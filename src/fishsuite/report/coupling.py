"""Descriptive, well-replicated single-plane MIAT/QKI coupling report."""
from __future__ import annotations

import colorsys
import hashlib
import shlex
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from . import figures as figlib
from . import workbook
from .coupling_stats import METRICS, prepare_data, summarize

ABUNDANCE = ('n_miat_spots', 'integrated_nuclear_miat')
ASSOCIATION = ('obs_minus_null_frac_miat_spots_qki_pos', 'qki_at_spots_minus_nuclear')
SHEET_ORDER = ['README', 'per_well', 'contrast', 'ratio_of_ratios',
               'within_well_correlation', 'sensitivity', 'all_arms']
PERMUTATION_NOTE = '2 v 2 wells: exact permutation has 6 allocations; smallest two-sided p = 0.333'
MARKERS = ('o', 's', '^', 'D', 'v', 'P', 'X')


def _well_markers(df, order):
    """Stable, distinct markers for the complete (arm, well) roster."""
    pairs = [(arm, well) for arm in order
             for well in sorted(df.loc[df.condition.eq(arm), 'well'].unique())]
    return {pair: MARKERS[i] if i < len(MARKERS) else f'${i + 1}$'
            for i, pair in enumerate(pairs)}


def _configuration(config, arms, treated):
    import yaml
    from matplotlib.colors import to_rgb

    cfg = yaml.safe_load(Path(config).read_text(encoding='utf-8')) if config else {}
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError('config must be a YAML mapping')
    requested = cfg.get('condition_order', [])
    overrides = cfg.get('group_colors', {})
    if not isinstance(requested, list) or not isinstance(overrides, dict):
        raise ValueError('condition_order must be a list and group_colors a mapping')
    order = list(dict.fromkeys([str(g) for g in requested if str(g) in arms] + list(arms)))
    defaults = {g: figlib.OKABE_ITO['sky'] if g == treated else '#7F7F7F' for g in order}
    defaults.update({str(k): str(v) for k, v in overrides.items()})
    colors = figlib.group_colors(order, defaults)
    hues = [colorsys.rgb_to_hsv(*to_rgb(c)) for c in colors.values()]
    red = any((h < .08 or h > .94) and s > .3 for h, s, v in hues)
    green = any(.20 < h < .48 and s > .3 for h, s, v in hues)
    if red and green:
        raise ValueError('group_colors must not pair red and green')
    return order, colors


def _axis_variants(ax, values):
    """Register display-only bounds with the shared full/focus exporter."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    lo, hi = (float(vals.min()), float(vals.max())) if len(vals) else (0., 1.)
    pad = max(hi - lo, abs(hi) * .1, .01) * .12

    def setter(variant):
        ax.set_ylim(min(0., lo - pad) if variant == 'full' else lo - pad, hi + pad)
    ax._replicate_simple_axis = setter


def _finish(fig, ax, destination, stem, title, filters, footer, manifest):
    figlib.no_box(fig, ax)
    extra = figlib.stamp_head(fig, textwrap.fill(title, 66), 'Descriptive; wells are the replicates', filters)
    ax.set_ylabel('\n'.join(textwrap.fill(line, 31) for line in ax.get_ylabel().splitlines()))
    ax.set_xlabel(textwrap.fill(ax.get_xlabel(), 65))
    foot = footer + '; axis: focus window'
    figlib.stamp_foot(fig, foot)
    bottom = max(.22, figlib.foot_band(foot, fig.get_figwidth(), fig.get_figheight(), 5.6) + .08)
    fig.subplots_adjust(left=.20, right=.97, top=.80-extra, bottom=bottom)
    figlib.save(fig, destination, stem, manifest, title + '; ' + filters, footer)


def _thresholds(df):
    parts = []
    for key in ('threshold_multiplier', 'miat_min_used', 'qki_min_used'):
        values = ','.join(f'{v:g}' for v in sorted(pd.to_numeric(df[key], errors='coerce').dropna().unique())) if key in df else 'missing'
        parts.append(f'{key}={values or "missing"}')
    return '; '.join(parts)


def _figures(df, sheets, treated, control, order, colors, source, out, seed):
    import matplotlib.pyplot as plt

    figlib.set_style()
    rng = np.random.default_rng(seed)
    base = df.loc[df.threshold_multiplier.eq(1.0)]
    footer = f'Input: {source}; thresholds: {_thresholds(df)}; single-plane; no nucleus-level tests'
    manifest = []
    pairs = list(base[['condition', 'well']].drop_duplicates().itertuples(index=False, name=None))
    shapes = _well_markers(df, order)
    metrics = sheets['sensitivity'].metric.tolist()
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(6.5, 4.8))
        values = []
        for x, arm in enumerate(order):
            wells = sorted(df.loc[df.condition.eq(arm), 'well'].unique())
            offsets = np.linspace(-.16, .16, len(wells)) if len(wells) > 1 else [0.]
            for offset, well in zip(offsets, wells):
                vals = base.loc[base.condition.eq(arm) & base.well.eq(well), metric].dropna().to_numpy()
                marker = shapes[arm, well]
                values.extend(vals)
                ax.scatter(x + offset + rng.uniform(-.06, .06, len(vals)), vals, s=12,
                           marker=marker, color=colors[arm], alpha=.35)
                if len(vals):
                    ax.scatter(x + offset, vals.mean(), s=70, marker=marker,
                               facecolors='white', edgecolors=colors[arm], linewidths=1.4,
                               label=f'{arm}: {well}', zorder=5)
                else:
                    ax.scatter([], [], marker=marker, color=colors[arm], label=f'{arm}: {well} (NA)')
        ax.set_xticks(range(len(order)), order)
        ax.set_ylabel(metric.replace('_', ' '))
        ax.legend(fontsize=6, loc='best')
        _axis_variants(ax, values)
        _finish(fig, ax, out, f'distribution_{metric}', metric.replace('_', ' '),
                'Filter: multiplier=1; finite metric; N=0 retained when defined. Small=nuclei; outlined=well means.', footer, manifest)

        fig, ax = plt.subplots(figsize=(6.5, 4.8))
        pw = sheets['per_well']
        pw = pw.loc[pw.threshold_multiplier.eq(1) & pw.condition.isin([treated, control])].copy()
        pw = pw.rename(columns={'condition': 'group', 'well': 'well_id', f'mean_{metric}': 'well_mean_of_field_values'})
        pw['endpoint'] = metric
        contrast_order = [a for a in order if a in (treated, control)]
        ctx = SimpleNamespace(group_order=contrast_order, colors=colors, reference=control,
                              technical_layer='none', footer=lambda text: footer)
        figlib.draw_replicate_simple(ax, ctx, metric, pw, pd.DataFrame(), pd.DataFrame(),
                                    pd.DataFrame(), metric.replace('_', ' '), None)
        # Keep the shared replicate-simple axes, but this panel requests dots only.
        for container in list(ax.containers):
            container.remove()
        for collection in list(ax.collections):
            collection.remove()
        for x, arm in enumerate(contrast_order):
            wells = pw.loc[pw.group.eq(arm)].sort_values('well_id')
            offsets = np.linspace(-.14, .14, len(wells)) if len(wells) > 1 else [0.]
            for offset, row in zip(offsets, wells.itertuples()):
                val = row.well_mean_of_field_values
                ax.scatter(x+offset, val, marker=shapes[arm, row.well_id], s=70,
                           facecolors='white', edgecolors=colors[arm], linewidths=1.4,
                           label=f'{arm}: {row.well_id}' + (' (NA)' if pd.isna(val) else ''))
        ax.legend(fontsize=6)
        _finish(fig, ax, out, f'well_means_{metric}', metric.replace('_', ' '),
                'Filter: treated/control; multiplier=1; finite per-nucleus metric within each well. Every well shown.', footer, manifest)

        fig, ax = plt.subplots(figsize=(6.5, 4.8))
        row = sheets['sensitivity'].set_index('metric').loc[metric]
        levels = sorted(df.threshold_multiplier.unique())
        positions = np.arange(len(levels))
        differences = [row[f'difference_multiplier_{level:g}'] for level in levels]
        ax.plot(positions, differences, marker='o', color=colors[treated])
        ax.set_xticks(positions, [f'{level:g}' for level in levels])
        ax.set_xlim(-.5, len(levels) - .5)
        for position, level, difference in zip(positions, levels, differences):
            if not np.isfinite(difference):
                reason = str(row[f'na_reason_multiplier_{level:g}'])
                ax.scatter([position], [0], transform=ax.get_xaxis_transform(),
                           marker='o', s=45, facecolors='none', edgecolors=colors[treated],
                           linewidths=1.2, clip_on=False, zorder=6)
                ax.annotate('NA\n' + textwrap.fill(reason, 25), xy=(position, 0),
                            xycoords=ax.get_xaxis_transform(), xytext=(0, 8),
                            textcoords='offset points', ha='center', va='bottom', fontsize=6,
                            bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1})
        ax.axhline(0, color='#7F7F7F', linewidth=.7)
        ax.set_xlabel('Threshold multiplier')
        ax.set_ylabel('Treated - control\n' + metric.replace('_', ' '))
        _axis_variants(ax, [0, *differences])
        _finish(fig, ax, out, f'sensitivity_{metric}', metric.replace('_', ' '),
                'Filter: treated/control; all thresholds; finite metric; equal weighting of wells. NA=open baseline marker with reason.', footer, manifest)

    correlations = sheets['within_well_correlation']
    for abundance in ABUNDANCE:
        for association in ASSOCIATION:
            fig, ax = plt.subplots(figsize=(6.5, 4.8))
            values = []
            for arm, well in pairs:
                subset = base.loc[base.condition.eq(arm) & base.well.eq(well) & base.n_miat_spots.gt(0)]
                subset = subset.dropna(subset=[abundance, association])
                r = correlations.loc[correlations.condition.eq(arm) & correlations.well.eq(well)
                    & correlations.threshold_multiplier.eq(1) & correlations.abundance_metric.eq(abundance)
                    & correlations.association_metric.eq(association)].iloc[0]
                detail = f'Pearson={r.pearson_r:.2f}, Spearman={r.spearman_r:.2f}' if not r.na_reason else f'r=NA; n={r.n_pairs} (gate unmet)'
                ax.scatter(subset[abundance], subset[association], marker=shapes[arm, well],
                           s=15, alpha=.6, color=colors[arm], label=f'{arm}/{well}: {detail}')
                values.extend(subset[association])
            ax.set_xlabel(abundance.replace('_', ' '))
            ax.set_ylabel(association.replace('_', ' '))
            ax.legend(fontsize=5.5, loc='best')
            _axis_variants(ax, values)
            _finish(fig, ax, out, f'scatter_{abundance}_vs_{association}', 'Within-well abundance versus association',
                    'Filter: multiplier=1; N>0; finite pairs. r requires >=10 pairs and >=5 distinct values in each variable.', footer, manifest)

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    values = []
    for arm, well in pairs:
        subset = base.loc[base.condition.eq(arm) & base.well.eq(well) & base.n_miat_spots.gt(0)].dropna(subset=['frac_miat_spots_qki_pos'])
        ax.scatter(subset.n_miat_spots, subset.frac_miat_spots_qki_pos, marker=shapes[arm, well],
                   color=colors[arm], alpha=.5, s=15, label=f'{arm}: {well}')
        values.extend(subset.frac_miat_spots_qki_pos)
    ax.set_xlabel('MIAT spots per nucleus (N)')
    ax.set_ylabel('QKI-positive fraction of MIAT spots')
    ax.legend(fontsize=6)
    _axis_variants(ax, [0, 1, *values])
    _finish(fig, ax, out, 'precision_funnel', 'Precision funnel',
            'Filter: multiplier=1; N>0; finite fraction. Descriptive points; no inferential bounds.', footer, manifest)
    pd.DataFrame(manifest).to_csv(out / 'figure_manifest.csv', index=False)


def build_coupling(assoc_csv, treated, control, out, config=None, seed=0, n_boot=2000):
    """Build a new report from saved per-nucleus association measurements."""
    from ..core import repro
    from ..core.qki_association import COLUMN_DEFINITIONS

    source = Path(assoc_csv).resolve()
    destination = Path(out).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('out must be new or empty; existing report files are not overwritten')
    raw = pd.read_csv(source, dtype={'condition': str, 'well': str, 'image': str})
    sheets = summarize(raw, treated, control, seed=seed, n_boot=n_boot)
    df = prepare_data(raw)
    order, colors = _configuration(config, list(dict.fromkeys(df.condition)), treated)
    notes = [dict(topic='input', detail=str(source)),
             dict(topic='input_sha256', detail=hashlib.sha256(source.read_bytes()).hexdigest()),
             dict(topic='thresholds used', detail=_thresholds(df)),
             dict(topic='display', detail=f'condition_order={order}; group_colors={colors}'),
             dict(topic='bootstrap settings', detail=f'seed={seed}; n_boot={n_boot}; with fewer than 3 wells per arm no interval is reported; read the individual well means. For >=3 wells per arm, exploratory 2.5/97.5 percentiles of treated-control difference; not confirmatory')]
    for metric in METRICS:
        notes.append(dict(topic=metric, detail=COLUMN_DEFINITIONS.get(metric, 'Definition missing')))
    for column in df:
        if column.startswith('na_reason_'):
            notes.append(dict(topic=column, detail=str(df[column].fillna('').value_counts().to_dict())))
    # Normalize the README's two descriptive columns without discarding any rows.
    readme = sheets['README'].copy()
    if len(readme.columns) == 2:
        readme.columns = ['topic', 'detail']
    sheets['README'] = pd.concat([readme, pd.DataFrame(notes)], ignore_index=True)
    descriptions = {name: f'{name}: single-plane; biological wells are replicates; descriptive, no nucleus-level tests.' for name in SHEET_ORDER}
    descriptions['contrast'] += ' ' + PERMUTATION_NOTE
    destination.mkdir(parents=True, exist_ok=True)
    workbook.write(destination / 'coupling_summary.xlsx', sheets, order=SHEET_ORDER, descriptions=descriptions)
    # The common writer reserves README wrapping for its legacy 'Read me' name.
    # Keep its table/header formatting and make this command's README readable too.
    from openpyxl import load_workbook
    from openpyxl.styles import Alignment
    book = load_workbook(destination / 'coupling_summary.xlsx')
    readme_sheet = book['README']
    readme_sheet.column_dimensions['A'].width = 48
    readme_sheet.column_dimensions['B'].width = 110
    for row in readme_sheet.iter_rows(min_row=3):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical='top')
        lines = max(len(textwrap.wrap(str(cell.value or ''), 46 if cell.column == 1 else 105)) for cell in row)
        readme_sheet.row_dimensions[row[0].row].height = max(30, 15 * lines)
    book.save(destination / 'coupling_summary.xlsx')
    book.close()
    _figures(df, sheets, treated, control, order, colors, source, destination / 'figures', seed)
    command = ['fishsuite', 'coupling', '--assoc-csv', str(source), '--treated', treated,
               '--control', control, '--seed', str(seed), '--n-boot', str(n_boot), '--out', str(destination)]
    if config:
        command += ['--config', str(Path(config).resolve())]
    if not repro.write_command_log(destination, str(Path(config).resolve()) if config else 'none', destination, seed,
                                  extra={'coupling_command': shlex.join(command), 'quantitation': 'single-plane'}):
        raise OSError('failed to write command.log')
    if not repro.write_versions_txt(destination, seed):
        raise OSError('failed to write versions.txt')
    return destination
