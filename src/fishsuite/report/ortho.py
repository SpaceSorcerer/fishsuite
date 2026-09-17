"""Run-directory adapter for orthogonal MIAT/QKI figures."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..core import io
from ..core.ortho_profile import clipped_default_line, pick_punctum, render_ortho_figure, select_nuclei, nucleus_crop


def _display_levels(cfg):
    output = cfg.output
    partner = 'rna2' if cfg.channels.analysis_mode == 'rna_rna' else 'antibody'
    levels = []
    for name in ('rna', partner):
        lo = getattr(output, f'manual_{name}_min')
        hi = getattr(output, f'manual_{name}_max')
        if lo is None or hi is None or not np.isfinite([lo, hi]).all() or hi <= lo:
            raise ValueError(f'Fixed output.manual_{name}_min/max are required (finite min < max)')
        levels.append((float(lo), float(hi)))
    return levels


def _source_and_mask(run_dir, row, roster, run_config):
    metadata = row.to_dict()
    if roster is not None:
        match = roster.loc[roster.image == row.image]
        if len(match) != 1:
            raise ValueError(f'Expected one hierarchy entry for image {row.image!r}')
        metadata.update(match.iloc[0].dropna().to_dict())
    source = metadata.get('source_path')
    input_dir = run_config.get('input_dir')
    if source is not None and pd.notna(source):
        source = Path(str(source))
        if not source.is_absolute():
            if not input_dir:
                raise ValueError('Relative source_path requires run_config input_dir')
            source = Path(input_dir) / source
    elif input_dir:
        source = Path(input_dir) / str(row.image)
        if not source.is_file():
            discovery = run_config.get('config_resolved', {}).get('conditions', {})
            candidates = [im.path for im in io.discover_inputs(Path(input_dir),
                          recursive_discovery=discovery.get('recursive_discovery', False))
                          if im.path.name == str(row.image)]
            if len(candidates) != 1:
                raise ValueError(f'Missing or ambiguous source image {row.image!r} under {input_dir}')
            source = candidates[0]
    else:
        raise ValueError(f'Missing source_path/input_dir for {row.image!r}')
    if not source.is_file():
        raise FileNotFoundError(f'Missing source image: {source}')
    stem = metadata.get('output_stem')
    if stem is not None and pd.notna(stem):
        mask = run_dir / 'masks' / f'{stem}__nuclei_label_mask.tif'
    else:
        raise ValueError(f'Missing exact output_stem for {row.image!r}; provide the saved '
                         'mapping in nuclei_metrics.csv or resolved_experiment_hierarchy.csv')
    if not mask.is_file():
        raise FileNotFoundError(f'Missing saved nucleus mask: {mask}')
    return source.resolve(), mask.resolve()


def _channel_indices(cfg, image):
    partner = cfg.channels.rna2 if cfg.channels.analysis_mode == 'rna_rna' else cfg.channels.antibody
    indices = [cfg.channels.rna, partner, cfg.channels.dapi]
    indices = [v - 1 if cfg.channels.one_indexed and v > 0 else v for v in indices]
    if any(v < 0 for v in indices):
        auto = io.autodetect_channels(image)
        indices = [auto[name] if value < 0 else value
                   for name, value in zip(('rna', 'ab', 'dapi'), indices)]
    if len(set(indices)) != 3 or any(v < 0 or v >= image.n_channels for v in indices):
        raise ValueError(f'MIAT/QKI/DAPI require three distinct valid channel indices: {indices}')
    return indices


def _run_calibration(image, record, record_name):
    """Validate header dimensions against saved effective analysis calibration.

    The existing dapi_mask.build_dapi comparison is inline and XY-only; there
    is no shared helper. Relative tolerance here is the requested one percent,
    measured against the recorded run value, with no absolute tolerance.
    """
    values, sources = [], []
    for column, required in (('voxel_xy_nm', True), ('voxel_z_nm', False)):
        header = float(getattr(image, column))
        if not np.isfinite(header) or header <= 0:
            raise ValueError(f'Missing valid header {column}: {record.image}')
        saved = record.get(column, np.nan)
        if pd.isna(saved):
            if required:
                raise ValueError(f'Missing recorded XY calibration: {record.image} ({record_name}:{column})')
            provenance = f'image_header:{column}; recorded run value unavailable'
        else:
            saved = float(saved)
            if not np.isfinite(saved) or saved <= 0:
                raise ValueError(f'Invalid recorded calibration: {record.image}, {column}={saved}')
            if not np.isclose(header, saved, rtol=.01, atol=0):
                raise ValueError(f'calibration mismatch: {record.image}, {column}: '
                                 f'header={header:g} nm, run={saved:g} nm ({record_name})')
            provenance = f'image_header:{column}; validated against {record_name}:{column}'
        values.append(header / 1000.)
        sources.append(provenance)
    return (*values, *sources)


def _analysed_plane(record, n_z):
    """per_image_summary.z_plane is 1-based absolute; return validated 0-based Z."""
    value = record.get('z_plane', np.nan)
    if pd.isna(value):
        return None
    value = float(value)
    if not np.isfinite(value) or value != int(value) or not 1 <= value <= n_z:
        raise ValueError(f'Invalid analysed plane for {record.image}: z_plane={value} '
                         f'(expected 1-based integer in [1, {n_z}])')
    return int(value) - 1


def _spot_center(spots, nucleus_id, rule, stack, *, analysed_plane_z=None):
    spots = spots.copy()
    if 'peak_intensity' not in spots and 'spot_peak_intensity' in spots:
        spots['peak_intensity'] = spots['spot_peak_intensity']
    # Spot coordinates select XY only. The metrics' actual analysis plane wins
    # over spot Z fields, which may refer to extracted/projected coordinates.
    _, y, x = pick_punctum(spots, nucleus_id, rule)
    iy, ix = int(round(y)), int(round(x))
    if not (0 <= iy < stack.shape[2] and 0 <= ix < stack.shape[3]):
        raise ValueError(f'Punctum XY outside source image for nucleus {nucleus_id}')
    z = analysed_plane_z
    z_source = 'analysed_plane'
    if z is None:
        window = stack[0, :, max(0, iy-2):iy+3, max(0, ix-2):ix+3]
        scores = window.astype(float).mean(axis=(1, 2))
        if not np.isfinite(scores).all():
            raise ValueError('Non-finite MIAT intensity in fallback Z window')
        z = float(np.argmax(scores))
        z_source = 'miat_5x5_mean_argmax'
    if not np.isfinite(z) or z != int(z) or not 0 <= z < stack.shape[1]:
        raise ValueError(f'Invalid full-stack punctum Z: {z}')
    return (int(z), y, x), z_source


def render_run(run_dir, *, config=None, k=3, metric='nuclear_spot_count', seed=0,
               punctum='brightest', half_width_um=None, qki_min=None, miat_min=None, out=None, command=''):
    from .. import __version__
    from ..config.schema import FishsuiteConfig
    import matplotlib.pyplot as plt
    import tifffile
    import yaml

    run_dir = Path(run_dir).resolve()
    out = Path(out).resolve() if out else run_dir / 'ortho'
    config_path = run_dir / 'run_config.json'
    if not config_path.is_file():
        raise FileNotFoundError(f'Missing {config_path}')
    run_config = json.loads(config_path.read_text(encoding='utf-8'))
    saved = run_config.get('config_resolved')
    if not isinstance(saved, dict):
        raise ValueError('Missing run_config.json config_resolved')
    resolved = dict(saved)
    if config:
        overrides = yaml.safe_load(Path(config).read_text(encoding='utf-8'))
        if not isinstance(overrides, dict):
            raise ValueError('Config YAML must contain a mapping')
        # Only output options are needed for figure overrides; preserve run
        # acquisition settings unless explicitly supplied in the YAML.
        for key, value in overrides.items():
            resolved[key] = ({**resolved.get(key, {}), **value}
                             if isinstance(value, dict) else value)
    cfg = FishsuiteConfig.model_validate(resolved)
    levels = _display_levels(cfg)
    prefix = saved.get('output', {}).get('prefix') or ''
    nuclei_path = run_dir / f'{prefix}nuclei_metrics.csv'
    spots_path = run_dir / f'{prefix}spot_metrics.csv'
    for path in (nuclei_path, spots_path):
        if not path.is_file():
            raise FileNotFoundError(f'Missing {path}')
    nuclei = pd.read_csv(nuclei_path)
    spots = pd.read_csv(spots_path)
    summary_path = run_dir / f'{prefix}per_image_summary.csv'
    if not summary_path.is_file():
        raise FileNotFoundError(f'Missing run calibration/analysed-plane table: {summary_path}')
    summary = pd.read_csv(summary_path)
    if 'image' not in summary or summary.image.isna().any() or summary.image.duplicated().any():
        raise ValueError(f'Missing or duplicate image identities in {summary_path.name}')
    for name, frame, required in (
        ('nuclei', nuclei, {'image', 'nucleus_id', metric}),
        ('spots', spots, {'image', 'nucleus_id', 'x_px', 'y_px'}),
    ):
        missing = required - set(frame)
        if missing:
            raise ValueError(f'Missing {name} columns: {sorted(missing)}')
    arm_col = next((c for c in ('arm', 'group', 'condition') if c in nuclei), None)
    if arm_col is None:
        raise ValueError('Missing arm/group/condition column in nuclei_metrics.csv')
    selected = select_nuclei(nuclei, arm_col, metric, k, seed)
    if selected.empty:
        raise ValueError('No nuclei available for selection')
    hierarchy_path = run_dir / 'resolved_experiment_hierarchy.csv'
    roster = pd.read_csv(hierarchy_path) if hierarchy_path.is_file() else None
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'command.log').open('a', encoding='utf-8') as handle:
        handle.write(command + '\n')
    (out / 'versions.txt').write_text(f'fishsuite {__version__}\nnumpy {np.__version__}\n'
                                      f'pandas {pd.__version__}\n', encoding='utf-8')
    rows = []
    cached_image = None
    for index, row in selected.iterrows():
        if cached_image != row.image:
            source, mask_path = _source_and_mask(run_dir, row, roster, run_config)
            matches = summary.loc[summary.image == row.image]
            if len(matches) != 1:
                raise ValueError(f'Missing run calibration/analysed-plane record for {row.image}')
            record = matches.iloc[0]
            image = io.read_image(source)
            pixel_size, z_step, pixel_source, z_step_source = _run_calibration(
                image, record, summary_path.name)
            indices = _channel_indices(cfg, image)
            stack = np.stack([io.extract_channel(image, c, z_mode='3d') for c in indices])
            analysed_plane_z = _analysed_plane(record, stack.shape[1])
            labels = tifffile.imread(mask_path)
            if labels.shape != stack.shape[2:]:
                raise ValueError(f'Saved mask shape {labels.shape} differs from image XY {stack.shape[2:]}')
            cached_image = row.image
        mask = labels == row.nucleus_id
        if not mask.any():
            raise ValueError(f'Nucleus {row.nucleus_id} absent from {mask_path}')
        image_spots = spots.loc[spots.image == row.image]
        if 'channel' in image_spots:
            image_spots = image_spots.loc[image_spots.channel.isin(['rna1', 'rna'])]
        center, z_source = _spot_center(image_spots, row.nucleus_id, punctum, stack,
                                        analysed_plane_z=analysed_plane_z)
        crop_yx, half, used_half_um = nucleus_crop(mask, pixel_size, half_width_um)
        # The chord stays on punctum Y and is clipped to the nucleus-centred crop.
        ys, xs = np.nonzero(mask)
        endpoints = ((float(center[1]), float(max(xs.min(), crop_yx[1]-half, 0))),
                     (float(center[1]), float(min(xs.max(), crop_yx[1]+half, mask.shape[1]-1))))
        profile_width_px = 3
        fig = render_ortho_figure(stack, center, half, nucleus_mask=mask,
                                  pixel_size_um=pixel_size, z_step_um=z_step,
                                  display_levels=levels, line_endpoints=endpoints,
                                  profile_width_px=profile_width_px, analysed_plane_z=analysed_plane_z,
                                  qki_min=qki_min, miat_min=miat_min, run_dir=str(run_dir),
                                  arm=str(row[arm_col]),
                                  arm_color=cfg.conditions.group_colors.get(str(row[arm_col]), '#595959'),
                                  image=Path(str(row.image)).name, nucleus_id=int(row.nucleus_id),
                                  metric=metric, metric_value=float(row[metric]), arm_median=float(row.arm_median))
        slug = lambda text: re.sub(r'[^A-Za-z0-9]+', '-', str(text)).strip('-')
        name = f'ortho_{slug(row[arm_col])}_{slug(Path(str(row.image)).stem)}_nuc{int(row.nucleus_id)}'
        try:
            fig.savefig(out / f'{name}.png', dpi=600)
            with plt.rc_context({'svg.fonttype': 'none'}):
                fig.savefig(out / f'{name}.svg')
        finally:
            plt.close(fig)
        rows.append(dict(arm=row[arm_col], image=row.image, nucleus_id=row.nucleus_id,
                         metric=metric, metric_value=row[metric], arm_median=row.arm_median,
                         punctum_rule=punctum, seed=seed, k=k, half_width_um=used_half_um,
                         half_width_px=half, half_width_used_um=half*pixel_size,
                         crop_center_y=crop_yx[0], crop_center_x=crop_yx[1],
                         profile_width_px=profile_width_px,
                         punctum_z=center[0], punctum_y=center[1], punctum_x=center[2],
                         z_source=z_source, line_p0_y=endpoints[0][0], line_p0_x=endpoints[0][1],
                         line_p1_y=endpoints[1][0], line_p1_x=endpoints[1][1],
                         source_path=str(source), mask_path=str(mask_path),
                         pixel_size_um=pixel_size, z_step_um=z_step,
                         pixel_size_source=pixel_source, z_step_source=z_step_source,
                         analysed_plane_z=analysed_plane_z,
                         analysed_plane_record=(f'{summary_path.name}:z_plane (1-based absolute)'
                                                if analysed_plane_z is not None else None),
                         display_mode='manual', configured_display_mode=cfg.output.pub_contrast_mode,
                         miat_display_min=levels[0][0], miat_display_max=levels[0][1],
                         qki_display_min=levels[1][0], qki_display_max=levels[1][1],
                         qki_min=qki_min, miat_min=miat_min, figure=name))
    result = pd.DataFrame(rows)
    result.to_csv(out / 'ortho_selection.csv', index=False)
    return result
