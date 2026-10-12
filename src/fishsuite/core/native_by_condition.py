"""CSV-only native condition figures, using the report hierarchy and renderer.

No source measurement, segmentation, detection, threshold, or gate is changed.
The public regenerator only writes a fresh output folder. The native downstream
hook additionally archives that new run's legacy well figures in place.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fishsuite.config.schema import ConditionsCfg
from fishsuite.report import aggregate as agg, endpoints as ep, figures as fig
from fishsuite.report.sensitivities import scalar_sensitivities
from fishsuite.report.stats import welch


def read_groups(path: Path) -> ConditionsCfg:
    raw = yaml.safe_load(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(raw, dict):
        raise ValueError('groups YAML must be a mapping')
    block = raw.get('conditions', raw)
    cfg = ConditionsCfg.model_validate(block)
    if not cfg.groups:
        raise ValueError('groups YAML must declare groups')
    return cfg


def collapse_conditions(well: pd.DataFrame) -> pd.DataFrame:
    """Equal weight per well, independent of its FOV or nucleus count."""
    rows = []
    for (group, endpoint), sub in well.groupby(['group', 'endpoint'], sort=False):
        values = sub.well_mean_of_field_values.replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(dict(condition=group, endpoint=endpoint, n_wells=len(values),
            n_FOVs=int(sub.n_fields.sum()), n_nuclei=int(sub.n_nuclei_in_well.sum()),
            mean_of_well_means=values.mean(), sd_across_wells=values.std(ddof=1)))
    return pd.DataFrame(rows)


def _load(run: Path, groups: ConditionsCfg):
    image = pd.read_csv(run/'per_image_summary.csv')
    nuclei = pd.read_csv(run/'nuclei_metrics.csv')
    cfg = json.loads((run/'run_config.json').read_text(encoding='utf-8-sig'))
    for col in ['image', 'condition', 'secondary_only']:
        if col not in image:
            raise ValueError(f'per_image_summary.csv missing {col}')
    if image.image.duplicated().any():
        raise ValueError('duplicate image identities in per_image_summary.csv')
    image['secondary_only'] = image.secondary_only.astype(str).str.lower().isin(['true','1','1.0'])
    from fishsuite.config.hierarchy import resolve_hierarchy, group_mapping
    labels = resolve_hierarchy(image, group_mapping(groups.groups), groups.well_from_image, strict=True)
    labels = agg.label_frame(labels, group_mapping(groups.groups), {}, groups.well_from_image)
    nuclei = nuclei.drop(columns=['condition','secondary_only','group','well_id','field',
                                   'excluded_field','exclusion_reason','sec_field','source_condition','field_id','source_path','output_stem'], errors='ignore')
    nuclei = nuclei.merge(labels, on='image', how='left', validate='many_to_one')
    if nuclei.group.isna().any():
        raise ValueError('nucleus image absent from per_image_summary.csv')
    for c in nuclei:
        if 'usable' in c:
            nuclei[c] = nuclei[c].astype(str).str.lower().isin(['true','1','1.0'])
    endpoints, absent = ep.resolve(nuclei, image)
    endpoints = ep.usable_endpoints(endpoints, absent)
    # Native pixel/morphology measurements not represented in the report registry.
    used = {e.column for e in endpoints}
    for c in nuclei:
        if c not in used and (c.startswith(('pearson_', 'manders_', 'li_icq', 'icq_', 'coloc_'))
                             or c in ['nucleus_area_px', 'cyto_area_px', 'cell_area_px']):
            if pd.to_numeric(nuclei[c], errors='coerce').notna().any():
                endpoints.append(ep.Endpoint(c, c, 'partner', 'fraction' if 'manders' in c else 'value', c.replace('_',' ')))
    endpoints = [e for e in endpoints if pd.to_numeric(
        (nuclei if e.level == 'nucleus' else image)[e.column], errors='coerce').notna().any()]
    if not endpoints:
        raise ValueError('no finite native endpoints in finished CSVs')
    # per_field_long normally omits controls. Include them for DISPLAY only;
    # retain original control flags in the nucleus table used for inference.
    display_nuc, display_labels = nuclei.copy(), labels.copy()
    display_nuc['secondary_only'] = False
    display_labels['secondary_only'] = False
    fields = agg.per_field_long(display_nuc, image, endpoints, display_labels, 0)
    wells = agg.per_well_long(fields)
    order = agg.resolve_group_order(labels, groups.resolved_group_order())
    if not order:
        raise ValueError('no biological conditions')
    contrasts = []
    for e in endpoints:
        print(f'Native condition statistics: {e.name}', flush=True)
        w = wells.loc[wells.endpoint.eq(e.name)]
        f = fields.loc[fields.endpoint.eq(e.name)]
        for group in order[1:]:
            a = w.loc[w.group.eq(group), 'well_mean_of_field_values'].dropna().to_numpy()
            b = w.loc[w.group.eq(order[0]), 'well_mean_of_field_values'].dropna().to_numpy()
            row = dict(endpoint=e.name, test_group=group, reference_group=order[0],
                       descriptive_only=e.descriptive_only, absolute_intensity=e.absolute_intensity)
            row.update(welch(a,b))
            row.update(scalar_sensitivities(e,nuclei,f,a,b,group,order[0]))
            contrasts.append(row)
    if labels.secondary_only.any():
        order.append(agg.SEC_ONLY_GROUP)
    from fishsuite.report.sensitivities import add_mixed_columns
    table = add_mixed_columns(pd.DataFrame(contrasts)) if contrasts else pd.DataFrame(contrasts)
    return cfg, nuclei, fields, wells, table, endpoints, order


def _render(run: Path, out: Path, groups: ConditionsCfg):
    cfg, nuclei, fields, wells, contrasts, endpoints, order = _load(run, groups)
    condition_dir = out/'figures/per_condition'
    overview_dir = out/'figures/00_overview'
    condition_dir.mkdir(parents=True, exist_ok=True)
    overview_dir.mkdir(parents=True, exist_ok=True)
    labels = ep.channel_labels(cfg)
    fig.set_style()
    ctx = fig.FigureContext(run,cfg,None,order,order[0],.05,{},labels,
        color_overrides=groups.group_colors,plot_style='replicate-simple',technical_layer='none')
    fig.prepare_axis_groups(ctx,endpoints,wells,fields)
    manifest = []
    for e in endpoints:
        print(f'Native condition figure: {e.name}', flush=True)
        fig.superplot_standalone(ctx,e.name,e.pretty(labels),e.pretty(labels),
            wells,fields,nuclei,contrasts,e.column if e.level=='nucleus' else None,
            condition_dir,e.name,manifest)
    # Six panels per overview page keeps figures readable and memory bounded.
    ordered = sorted(endpoints, key=lambda e: not e.primary)
    for start in range(0,len(ordered),6):
        stem = 'combined_output_panel' if start==0 else f'combined_output_panel_{start//6+1:02d}'
        specs = [dict(endpoint=e.name,ylabel=ep.short_title(e.name,e.pretty(labels),labels),nuc_column=e.column)
                 for e in ordered[start:start+6]]
        fig.composite_main(ctx,specs,[],[],wells,fields,nuclei,contrasts,overview_dir,manifest,stem=stem)
    # Compatibility entry point now resolves to the condition-level overview.
    shutil.copy2(overview_dir/'combined_output_panel_focus.png', overview_dir/'single_condition_panel.png')
    conditions = collapse_conditions(wells)
    workbook = out/'analysis_summary.xlsx'
    # Regeneration builds a clean workbook and never republishes the source workbook's sheets; in place, the
    # engine's own workbook receives only freshly built, stamped native sheets.
    from fishsuite.report.sensitivities import check_writable, stamp_statistics
    native = {'Per condition': conditions, 'Per well': wells, 'Per image': fields, 'Condition contrasts': contrasts}
    for name, table in native.items():
        check_writable(table, name)
    opts = dict(engine='openpyxl', mode='a', if_sheet_exists='replace') if (out == run and workbook.exists()) else dict(engine='openpyxl')
    with pd.ExcelWriter(workbook, **opts) as writer:
        for name, table in native.items():
            stamp_statistics(table, name, fresh=True).to_excel(writer,sheet_name=name,index=False)
    conditions.to_csv(out/'per_condition_summary.csv',index=False)
    fields.to_csv(out/'per_field_condition_summary.csv', index=False)
    wells.to_csv(out/'per_well_condition_summary.csv', index=False)
    from fishsuite.config.hierarchy import resolve_hierarchy, group_mapping
    full_roster = resolve_hierarchy(pd.read_csv(run/'per_image_summary.csv'),
        group_mapping(groups.groups), groups.well_from_image, strict=True)
    full_roster.to_csv(out/'condition_report_hierarchy.csv', index=False)
    attempt_path = run/'condition_output_status.json'
    attempt = json.loads(attempt_path.read_text(encoding='utf-8')).get('attempt_id') if attempt_path.is_file() else None
    provenance = dict(run=str(run), groups=groups.model_dump(), attempt_id=attempt,
        policy='Grouping only: source CSV values retained; YAML peak floors and nucleus filters are not applied.',
        hierarchy='nucleus -> FOV mean -> well mean -> condition mean; SD across well means',
        sources={name:hashlib.sha256((run/name).read_bytes()).hexdigest()
                 for name in ['nuclei_metrics.csv','per_image_summary.csv','run_config.json']},
        figures=manifest)
    (out/'native_by_condition.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    return dict(out=out,conditions=conditions,contrasts=contrasts)


def completed_attempt(run, attempt_id):
    """An old manifest from a resumed run cannot satisfy this attempt."""
    path = Path(run)/'native_by_condition.json'
    if not path.is_file():
        return False
    try:
        record = json.loads(path.read_text(encoding='utf-8'))
        return record.get('attempt_id') == attempt_id and all(
            hashlib.sha256((Path(run)/name).read_bytes()).hexdigest() == digest
            for name, digest in record['sources'].items())
    except (OSError, ValueError, KeyError):
        return False


def _frozen(run: Path) -> bool:
    # Only inspect the run and its ancestors, never recursively search data drives.
    def marker(name: str) -> bool:
        folded = name.casefold()
        return ('manifest' in folded or 'ledger' in folded or
                folded == 'report_lock.json' or 'frozen' in folded)

    for root in [run, *run.parents]:
        try:
            if any(p.is_file() and marker(p.name) for p in root.iterdir()):
                return True
        except OSError:
            continue
    return False


def regenerate(run: Path, groups: ConditionsCfg, out: Path | None = None):
    run = Path(run).resolve()
    if not groups.groups:
        raise ValueError('native-figures requires condition groups')
    if out is None:
        if _frozen(run):
            raise ValueError('frozen run: specify --out outside the run')
        out = run/('figures_by_condition_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out = Path(out).resolve()
    if out==run or out in run.parents or (out.is_relative_to(run) and
            (out.parent!=run or not out.name.startswith('figures_by_condition_'))):
        raise ValueError('refusing original run/figures: use a new external output directory')
    if out.is_relative_to(run) and _frozen(run):
        raise ValueError('frozen run: specify --out outside the run')
    if out.exists():
        raise ValueError('output must be a new directory; refusing overwrite')
    # Validate required inputs before allocating output.
    for name in ['per_image_summary.csv','nuclei_metrics.csv','run_config.json']:
        if not (run/name).is_file():
            raise ValueError(f'missing {run/name}')
    out.mkdir(parents=True)
    legacy = run/'figures'
    supplementary = out/'figures/per_well_supplementary'
    supplementary.mkdir(parents=True)
    result = _render(run,out,groups)  # copy legacy assets only after a successful render
    if legacy.is_dir():
        skip = _legacy_inferential(legacy)
        ignore = lambda d, names: [n for n in names if (Path(d)/n).resolve() in skip]
        for p in legacy.iterdir():
            if p.resolve() in skip:
                continue
            dest = out/'figures/per_image' if p.name=='per_image' else supplementary/p.name
            if p.is_dir():
                shutil.copytree(p,dest,ignore=ignore)
            else:
                shutil.copy2(p,dest)
    return result


_P_TEXT = __import__('re').compile(r'(mixed model p|\bp\s*[=<>]\s*[0-9.]|p-value|holm|MDE|minimum detectable)', __import__('re').I)


def _legacy_inferential(root: Path) -> set:
    """Legacy files that carry inference (unstamped statistical tables, figures whose SVG prints a p, Holm or
    MDE, and the raster/PDF twins of those figures); they are not republished by regeneration."""
    from fishsuite.report.sensitivities import is_statistical
    skip, stems = set(), set()
    for p in root.rglob('*'):
        if not p.is_file():
            continue
        suffix = p.suffix.lower()
        try:
            if suffix in ('.csv', '.tsv'):
                head = pd.read_csv(p, nrows=0, sep='\t' if suffix == '.tsv' else ',').columns
                if is_statistical(head):  # legacy inference is regenerated, never copied, whatever its stamp
                    skip.add(p.resolve())
            elif suffix == '.xlsx':
                skip.add(p.resolve())  # engine workbooks are not regenerated outputs
            elif suffix == '.svg' and _P_TEXT.search(p.read_text(encoding='utf-8', errors='replace')):
                skip.add(p.resolve()); stems.add((p.parent.resolve(), p.stem))
        except Exception:
            skip.add(p.resolve())  # unreadable: do not republish
    for p in root.rglob('*'):
        if p.is_file() and (p.parent.resolve(), p.stem) in stems:
            skip.add(p.resolve())
    return skip


def finalize_native(run: Path):
    """Called after the native downstream plots of a NEW run have finished."""
    run = Path(run).resolve()
    if not (run/'run_config.json').is_file():
        return None
    cfg = json.loads((run/'run_config.json').read_text(encoding='utf-8-sig'))
    groups = ConditionsCfg.model_validate(cfg.get('config_resolved',{}).get('conditions',{}))
    if not groups.groups:
        return None
    figures = run/'figures'
    supplementary = figures/'per_well_supplementary'
    supplementary.mkdir(parents=True,exist_ok=True)
    archive = supplementary
    if any(supplementary.iterdir()):
        archive = supplementary/('generation_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        archive.mkdir()
    for p in list(figures.iterdir()):
        if p.name in ['per_image','per_well_supplementary']:
            continue
        dest = archive/p.name
        if dest.exists():
            raise ValueError(f'refusing to overwrite supplementary figures: {dest}')
        p.rename(dest)
    return _render(run,run,groups)
