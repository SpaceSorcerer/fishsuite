"""Serial physical-diameter QC using the run's plane and segmentation helpers."""
from __future__ import annotations

import gc
import json
import math
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from fishsuite.config.schema import FishsuiteConfig, NucleiCfg
from fishsuite.config.hierarchy import select_inputs
from . import io, segmentation
from .nuclear_size import resolve_nuclear_size_px
from .repro import write_versions_txt, write_command_log


def choose_images(images, maximum=1, seed=0):
    """Sort names, seed a permutation per condition, take its first N entries."""
    if maximum < 1:
        raise ValueError('max-images-per-condition must be positive')
    rng = np.random.default_rng(seed)
    chosen = []
    for condition in sorted({im.condition for im in images}):
        group = sorted((im for im in images if im.condition == condition),
                       key=lambda im: (im.path.name, str(im.path)))
        chosen.extend(group[int(i)] for i in rng.permutation(len(group))[:maximum])
    return chosen


def parse_diameters(value):
    try:
        values = [float(x) for x in value.split(',')]
    except ValueError as exc:
        raise ValueError('diameters must be comma-separated positive finite numbers') from exc
    if not values or any(not math.isfinite(d) or d <= 0 for d in values) or len(set(values)) != len(values):
        raise ValueError('diameters must be distinct positive finite numbers')
    return values


def run_sweep(config_path, input_dir, out, diameters, maximum=1, seed=0):
    from .modes.rna_rna import prepare_segmentation_planes, segmentation_params
    from .modes.rna_protein import _build_rna2_shim_cfg
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import ListedColormap
    from skimage.segmentation import find_boundaries
    from PIL import Image

    config_path, input_dir, out = map(Path, (config_path, input_dir, out))
    diameters = parse_diameters(','.join(map(str, diameters)))
    cfg = FishsuiteConfig.from_yaml(config_path)
    if cfg.channels.analysis_mode == 'rna_only':
        from .modes.rna_only import prepare_segmentation_planes, segmentation_params
    elif cfg.channels.analysis_mode not in ('rna_rna', 'rna_protein'):
        raise ValueError('seg-sweep supports rna_only, rna_rna and rna_protein presets')
    if cfg.nuclei.backend != 'cellpose':
        raise ValueError('diameter sweeps require the cellpose backend')
    if out.exists() and any(out.iterdir()):
        raise ValueError('seg-sweep output directory must be new or empty')
    conditions = cfg.conditions
    images = io.discover_inputs(input_dir, **{key: getattr(conditions, key) for key in (
        'subfolder_conditions', 'sec_only_folders', 'sec_only_files', 'filename_conditions',
        'strict_subfolders', 'exclude_subfolders', 'strict_filenames', 'recursive_discovery')})
    images = select_inputs(images, input_dir, cfg.input_file_subset) if cfg.input_file_subset else images
    images = choose_images(images, maximum, seed)
    if not images:
        raise ValueError('No images selected for seg-sweep')
    out.mkdir(parents=True, exist_ok=True)
    if not write_versions_txt(out, seed) or not write_command_log(out, config_path, out, seed):
        raise OSError('Could not write sweep provenance')
    with (out / 'command.log').open('a', encoding='utf-8') as log:
        log.write('\nseg-sweep arguments: ' + json.dumps(dict(config=str(config_path.resolve()),
            input=str(input_dir.resolve()), out=str(out.resolve()), diameters_um=diameters,
            max_images_per_condition=maximum, seed=seed)) + '\n')
        log.write('Selection: sorted names, seeded permutation, first N per condition.\n')
        log.write('Counts cover area and border filtering; ghost and nucleus sampling are downstream.\n')
    cfg = _build_rna2_shim_cfg(cfg) if cfg.channels.analysis_mode == 'rna_protein' else cfg
    rows, distributions = [], {d: [] for d in diameters}
    display = None
    # Only rendered row rasters persist; microscopy handles/masks are released per image.
    with tempfile.TemporaryDirectory(prefix='seg_sweep_', dir=out) as temporary:
        row_paths = []
        for image_index, item in enumerate(images):
            img = io.read_image(item.path)
            pixel_nm = cfg.foci.bigfish_voxel_size_nm
            if not (math.isfinite(pixel_nm) and pixel_nm > 0):
                pixel_nm = img.voxel_xy_nm
            scale = pixel_nm / 1000.0
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError(f'{item.path.name}: readable positive XY pixel size required')
            planes = prepare_segmentation_planes(img, item.path, cfg)
            dapi = planes['dapi_2d']
            if display is None:
                low, high = cfg.output.manual_dapi_min, cfg.output.manual_dapi_max
                if low is None or high is None or high <= low:
                    low, high = float(np.min(dapi)), float(np.max(dapi))
                    high = max(high, low + 1)
                display = (low, high)
                with (out / 'command.log').open('a', encoding='utf-8') as log:
                    log.write(f'Fixed DAPI display range for every panel: {display}\n')
            fig = Figure(figsize=(2.4 * len(diameters), 2.8), dpi=600)
            FigureCanvasAgg(fig)
            axes = fig.subplots(1, len(diameters), squeeze=False)[0]
            for axis, diameter in zip(axes, diameters):
                nuclear = cfg.nuclei.model_dump(exclude_unset=True)
                nuclear.pop('cellpose_diameter_px', None)
                nuclear['expected_diameter_um'] = diameter
                sweep_cfg = cfg.model_copy(deep=True)
                sweep_cfg.nuclei = NucleiCfg.model_validate(nuclear)
                size = resolve_nuclear_size_px(sweep_cfg.nuclei, scale, sweep_cfg.nuclei.cellpose_downsample_factor)
                diagnostics = {}
                labels = segmentation.segment_nuclei(dapi, backend=cfg.nuclei.backend,
                    params=segmentation_params(sweep_cfg, size), diagnostics=diagnostics)
                raw = diagnostics['labels_before_area']
                ids, counts = np.unique(raw[raw > 0], return_counts=True)
                removed = ids[~np.isin(ids, np.unique(labels))]
                small = ids[(counts < size.min_area_px) & np.isin(ids, removed)]
                large = ids[(counts > size.max_area_px) & np.isin(ids, removed)]
                area_rejected = np.where(np.isin(raw, removed), raw, 0)
                backend_rejected = diagnostics.get('labels_backend_area_rejected', np.zeros_like(raw))
                backend_small_count = len(np.unique(backend_rejected[backend_rejected > 0]))
                # Backend and final filters renumber independently; offset the QC IDs.
                area_rejected = np.where(backend_rejected > 0,
                    backend_rejected + int(raw.max()), area_rejected)
                n_area = len(removed) + backend_small_count
                before_border = len(np.unique(labels[labels > 0]))
                final = segmentation.exclude_border_labels(labels, size.border_margin_px) if cfg.nuclei.exclude_border else labels
                _, kept_counts = np.unique(final[final > 0], return_counts=True)
                areas = kept_counts * scale ** 2
                n_border = before_border - len(areas)
                distributions[diameter].extend(areas.tolist())
                rows.append(dict(image=item.path.relative_to(input_dir).as_posix(), condition=item.condition,
                    diameter_um=diameter, diameter_px_model_input=size.diameter_px,
                    n_kept=len(areas), n_rejected_small=len(small) + backend_small_count, n_rejected_large=len(large),
                    n_rejected_border=n_border, median_area_um2=float(np.median(areas)) if len(areas) else np.nan,
                    iqr_area_um2=float(np.percentile(areas, 75)-np.percentile(areas, 25)) if len(areas) else np.nan,
                    median_equivalent_diameter_um=float(np.median(np.sqrt(4*areas/np.pi))) if len(areas) else np.nan))
                axis.imshow(dapi, cmap='gray', vmin=display[0], vmax=display[1])
                for mask, color in ((final, '#56B4E9'), (area_rejected, '#E69F00'),
                                    (np.where((labels > 0) & (final == 0), labels, 0), '#CC79A7')):
                    boundary = find_boundaries(mask, mode='inner')
                    axis.imshow(np.ma.masked_where(~boundary, boundary), cmap=ListedColormap([color]), vmin=0, vmax=1)
                length = min(float(cfg.output.scalebar_um), dapi.shape[1]*scale*0.25)
                x, y = dapi.shape[1]*0.06, dapi.shape[0]*0.9
                axis.plot([x, x+length/scale], [y, y], color='white', linewidth=2)
                axis.text(x, y-3, f'{length:g} µm', color='white', fontsize=6)
                axis.set_title(f'{diameter:g} µm → {size.diameter_px:.3f} px\n'
                    f'{len(areas)} kept / {n_area} area / {n_border} border', fontsize=7)
                axis.set_axis_off()
            fig.suptitle(f'{item.path.name} | {item.condition}\nBlue: kept; orange: area rejected; magenta: border rejected', fontsize=7)
            fig.tight_layout()
            row_path = Path(temporary) / f'row_{image_index}.png'
            fig.savefig(row_path, dpi=600)
            row_paths.append(row_path)
            fig.clear()
            del axes, axis, fig, img, planes, dapi, labels, raw, final, diagnostics, area_rejected, backend_rejected, mask, boundary
            gc.collect()
        with Image.open(row_paths[0]) as first:
            canvas = Image.new('RGB', (first.width, first.height * len(row_paths)), 'white')
            height = first.height
        for index, row_path in enumerate(row_paths):
            with Image.open(row_path) as row_image:
                canvas.paste(row_image, (0, index * height))
        canvas.save(out / 'seg_sweep_contact_sheet.png', dpi=(600, 600))
        canvas.close()
    pd.DataFrame(rows).to_csv(out / 'seg_sweep_summary.csv', index=False)
    fig = Figure(figsize=(4, 3), dpi=600)
    FigureCanvasAgg(fig)
    axis = fig.subplots()
    all_areas = [a for values in distributions.values() for a in values]
    bins = np.linspace(0, max(all_areas, default=1)*1.05, 21)
    for diameter, values in distributions.items():
        axis.hist(values, bins=bins, histtype='step', label=f'{diameter:g} µm')
    axis.set(xlabel='Nuclear area (µm²), retained nuclei', ylabel='Count')
    axis.legend()
    fig.tight_layout()
    fig.savefig(out / 'seg_sweep_area_hist.png', dpi=600)
    fig.clear()
    return out
