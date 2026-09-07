"""Read and render persisted standard-coloc results. No measurement/null engine imports."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter

from .aggregate import ReportInputError
from . import endpoints as ep
from .provenance import sha256, guard_output


@dataclass
class ExistingPanel:
    path: Path
    tables: dict
    sources: pd.DataFrame
    manifest: dict


def verify_pinned(path: Path, manifest: dict) -> str:
    path = Path(path).resolve()
    if not path.is_file():
        raise ReportInputError(f'missing source: {path}')
    records = manifest.get('named_files', []) + manifest.get('supplemental_files', [])
    matches = [r for r in records if Path(r['path']).resolve() == path]
    if len(matches) != 1 or not matches[0].get('sha256'):
        raise ReportInputError(f'missing unique frozen hash: {path}')
    digest = sha256(path)
    if digest.lower() != matches[0]['sha256'].lower():
        raise ReportInputError(f'stale source hash: {path}')
    return digest


def _groups(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if 'line' in frame:
        if 'group' in frame and not frame['group'].fillna('').equals(frame['line'].fillna('')):
            raise ReportInputError('conflicting panel line/group columns')
        frame['group'] = frame['line']
    return frame


def load_existing_panel(path: Path, figure_index: Path, baseline_manifest: Path,
                        figure_index_sha256: str | None = None) -> ExistingPanel:
    path, figure_index = Path(path), Path(figure_index)
    if not path.is_file() or not figure_index.is_file():
        raise ReportInputError(f'missing persisted panel or FIGURE_INDEX: {path}; {figure_index}')
    if baseline_manifest is None or not Path(baseline_manifest).is_file():
        raise ReportInputError('missing baseline manifest for persisted panel')
    manifest = json.loads(Path(baseline_manifest).read_text(encoding='utf-8'))
    digest = verify_pinned(path, manifest)
    # A0 pins the measurement workbook but does not pin this prose-only index.
    # Record its current bytes; deck specs additionally pin the A3 observed hash.
    index_digest = sha256(figure_index)
    if figure_index_sha256 and index_digest != figure_index_sha256:
        raise ReportInputError(f'stale figure index: {figure_index}')
    book = load_workbook(path, read_only=True, data_only=True)
    tables, sources = {}, []
    required = ('README', 'per_nucleus', 'per_FOV', 'per_well', 'contrasts',
                'line_profiles', 'overlays_index')
    try:
        for name in required:
            if name not in book.sheetnames:
                raise ReportInputError(f'missing panel sheet: {name}')
            ws = book[name]
            values = list(ws.values)
            if not values or len(set(values[0])) != len(values[0]):
                raise ReportInputError(f'missing or duplicate panel headers: {name}')
            # Use pandas' native dtype inference, without modifying any source cells.
            tables[name] = _groups(pd.read_excel(path, sheet_name=name))
            for ri, row in enumerate(ws.iter_rows(min_row=2),2):
                for ci, cell in enumerate(row,1):
                    sources.append(dict(file=str(path.resolve()), sha256=digest, sheet=name,
                                        cell=f'{get_column_letter(ci)}{ri}', value=cell.value))
    finally:
        book.close()
    sources.append(dict(file=str(figure_index.resolve()), sha256=index_digest,
                        sheet='', cell='', value=figure_index.read_text(encoding='utf-8')))
    required_cols = set(ep.A3_PARTNER_ADDITIONS) | set(ep.A3_PANEL_ALIASES)
    for name in ('per_nucleus', 'per_FOV', 'per_well'):
        missing = required_cols - set(tables[name])
        if missing:
            raise ReportInputError(f'missing {name} endpoints: {sorted(missing)}')
    pn = tables['per_nucleus']
    # The v3 contrasts sheet appends a second pooled-pixel table after two blank
    # rows. Preserve the whole sheet, but its secondary header/arms are not tests.
    raw_contrasts=tables['contrasts']
    separators=np.flatnonzero(raw_contrasts['endpoint'].isna().to_numpy())
    stop=int(separators[0]) if len(separators) else len(raw_contrasts)
    tables['endpoint_contrasts']=raw_contrasts.iloc[:stop].copy()
    if pn.duplicated(['image', 'nucleus_id']).any():
        raise ReportInputError('duplicate panel nucleus keys')
    for c in ('manders_m1_costes_only', 'manders_m2_costes_only'):
        if pn.loc[~pn['costes_converged'].eq(True), c].notna().any():
            raise ReportInputError('Costes failure incorrectly recorded as converged-only value')
    return ExistingPanel(path.resolve(), tables, pd.DataFrame(sources), manifest)


def validate_pairing(panel_nuclei: pd.DataFrame, spots: pd.DataFrame) -> None:
    """Independent observed forward numerator/denominator from nuclear spot flags."""
    keys = ['image', 'nucleus_id']
    required = set(keys + ['channel', 'in_nucleus', 'paired_at_0p3um'])
    if not required <= set(spots):
        raise ReportInputError(f'missing pairing spot fields: {sorted(required-set(spots))}')
    nuclear = spots.loc[(spots.channel == 'rna1') & spots.in_nucleus.eq(1) & (spots.nucleus_id > 0)]
    if not nuclear['paired_at_0p3um'].isin([0, 1, False, True]).all():
        raise ReportInputError('missing or invalid nuclear pairing calls')
    expected = nuclear.groupby(keys)['paired_at_0p3um'].mean()
    actual = panel_nuclei.set_index(keys)['paired_frac_rna1_at_partner']
    wanted = expected.reindex(actual.index)
    if not np.allclose(actual, wanted, atol=1e-12, rtol=1e-10, equal_nan=True):
        raise ReportInputError('panel nuclear pairing differs from persisted nuclear spot flags')


def integrate_panel(data: dict, panel: ExistingPanel, endpoints: list) -> tuple:
    """Validate full retained cohort, aliases and anchors before adding columns."""
    nuc = data['nuclei']
    pn = panel.tables['per_nucleus']
    keys = ['image', 'nucleus_id']
    a, b = nuc.set_index(keys).sort_index(), pn.set_index(keys).sort_index()
    if not a.index.equals(b.index):
        raise ReportInputError('persisted panel cohort mismatch: nucleus roster')
    for c in ('condition', 'group', 'secondary_only'):
        mask = ~a.secondary_only.eq(True) if c == 'group' else pd.Series(True,index=a.index)
        if c in a and c in b and not a.loc[mask,c].fillna('').equals(b.loc[mask,c].fillna('')):
            raise ReportInputError(f'persisted panel cohort mismatch: {c}')
    for pc, nc in (('n_rna1_nuclear_puncta', 'nuclear_spot_count'),
                   ('n_partner_nuclear_puncta', 'n_spots_protein')):
        if not np.array_equal(a[nc], b[pc]):
            raise ReportInputError(f'persisted panel anchor count mismatch: {pc}')
    by_name = {e.name:e for e in endpoints}
    for alias, name in ep.A3_PANEL_ALIASES.items():
        if name not in by_name:
            raise ReportInputError(f'missing original alias endpoint: {name}')
        if not np.allclose(a[by_name[name].column], b[alias], atol=1e-12, rtol=1e-10, equal_nan=True):
            raise ReportInputError(f'persisted alias differs: {alias}')
    for record in panel.manifest['endpoint_registry']:
        old = by_name.get(record['name'])
        if old is None:
            raise ReportInputError(f'frozen registry member missing: {record["name"]}')
        for attr in ('family', 'descriptive_only', 'absolute_intensity', 'excluded_from_holm', 'usability_flag'):
            if getattr(old, attr) != record[attr]:
                raise ReportInputError(f'frozen registry membership changed: {old.name}/{attr}')
    added = ep.a3_endpoints(panel.tables['per_well'].columns)
    columns = [e.column for e in added if e.column in pn and e.column not in nuc]
    merged = nuc.merge(pn[keys+columns], on=keys, how='left', validate='one_to_one')
    for endpoint in added:
        if endpoint.column not in merged:
            raise ReportInputError(f'missing A endpoint: {endpoint.column}')
    data['nuclei'] = merged
    return endpoints + added, added


def render_existing(panel: ExistingPanel, ctx, contrasts: pd.DataFrame, out_dir: Path) -> list:
    """Re-render every persisted per-well metric using the common replicate-simple key."""
    from . import figures as fig
    out_dir = guard_output(out_dir)
    fig.set_style()
    manifest = []
    pw = panel.tables['per_well']
    pf = panel.tables['per_FOV']
    pn = panel.tables['per_nucleus']
    pw = pw.loc[~pw.secondary_only.eq(True)]
    pf = pf.loc[~pf.secondary_only.eq(True)]
    for metric in panel.tables['endpoint_contrasts']['endpoint']:
        if metric not in pw:
            raise ReportInputError(f'missing persisted per-well metric: {metric}')
        name = ep.A3_PANEL_ALIASES.get(metric, metric)
        w = pw[['group','well_id',metric]].rename(columns={metric:'well_mean_of_field_values'}).assign(endpoint=name)
        f = pf[['group','well_id','image',metric]].rename(columns={metric:'field_value'}).assign(endpoint=name,level='nucleus')
        finite = pn.loc[~pn.secondary_only.eq(True)].groupby('image')[metric].count()
        f['n_nuclei_nonmissing'] = f.image.map(finite)
        r = contrasts.loc[contrasts.endpoint == name]
        # Historical descriptive rows are not re-tested for a plot. Added tests use
        # the explicitly amended report contrasts; all other p cells stay missing.
        if metric not in ep.A3_PARTNER_ADDITIONS and metric not in ep.A3_PANEL_ALIASES:
            r = r.iloc[:0]
        desc = panel.tables['contrasts'].loc[lambda x: x.endpoint == metric].iloc[0]
        fig.superplot_standalone(ctx, name, str(desc['label']), str(desc['unit']),
                                w, f, pn, r, None, out_dir, metric, manifest)
        manifest[-1].update(endpoint=metric, source_file=str(panel.path),
                            source_sheet='per_well', sha256=sha256(out_dir / manifest[-1]['png']),
                            svg_sha256=sha256(out_dir / manifest[-1]['svg']),
                            source_cells=';'.join(f'per_well!{get_column_letter(pw.columns.get_loc(metric)+1)}{i+2}' for i in pw.index),
                            report_cells=';'.join(f'Coloc per well!{get_column_letter(pw.columns.get_loc(metric)+1)}{i+3}' for i in pw.index))
    (out_dir/'FIGURE_INDEX.md').write_text('# Persisted standard panel\n\n'
        'WT #595959; QKI-KO #D67AE5. Nuclear anchors only. No new nulls.\n\n' +
        '\n'.join(f'- {r["png"]}: {r["endpoint"]}; source {panel.path} / per_well' for r in manifest), encoding='utf-8')
    return manifest
