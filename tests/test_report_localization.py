"""Localization accounting, hierarchy, frozen statistics and figure contract."""
from pathlib import Path
import json
import os
import hashlib

import numpy as np
import pandas as pd
import pytest

from fishsuite.report import aggregate as agg, endpoints as ep, figures as fig


def compare(old, new, keys, exclude=(), allowed_new=("rna1_cyto_spots_per_nucleus",)):
    old = old.set_index(keys).sort_index()
    new = new.set_index(keys).sort_index()
    assert old.index.is_unique and new.index.is_unique
    missing = old.index.difference(new.index)
    extra = new.index.difference(old.index)
    assert missing.empty, f'missing keys: {missing.tolist()}'
    assert set(extra.get_level_values('endpoint')) <= set(allowed_new), f'extra keys: {extra.tolist()}'
    new = new.loc[old.index]
    assert new.index.equals(old.index)
    for col in old:
        if col in exclude:
            continue
        x, y = old[col], new[col]
        # Frozen spreadsheet blank strings are represented as null in A0.
        if pd.api.types.is_numeric_dtype(x) or pd.api.types.is_bool_dtype(x):
            np.testing.assert_array_equal(x.isna(), y.isna(), err_msg=col)
            if pd.api.types.is_bool_dtype(x) or col.startswith('n_'):
                np.testing.assert_array_equal(x.to_numpy(), y.to_numpy(), err_msg=col)
            else:
                np.testing.assert_allclose(x.to_numpy(float), y.to_numpy(float), atol=1e-12, rtol=1e-10, equal_nan=True, err_msg=col)
        else:
            assert x.fillna('').astype(str).tolist() == y.fillna('').astype(str).tolist(), col

def fixture_frames():
    # FOV a: zero, all nuclear, all cyto, mixed; FOV b: one all-nuclear nucleus.
    nuc = pd.DataFrame(dict(image=['a'] * 4 + ['b'], nucleus_id=[1, 2, 3, 4, 1],
        nuclear_spot_count=[0, 2, 0, 1, 1], cyto_spot_count=[0, 0, 3, 3, 0],
        nuclear_spot_fraction=[np.nan, 1., 0., .25, 1.]))
    nuc['rna_spot_count'] = nuc.nuclear_spot_count + nuc.cyto_spot_count
    nuc['n_spots_rna1'] = nuc.rna_spot_count
    nuc['n_cytoplasmic_rna1_spots_per_cell'] = nuc.cyto_spot_count
    rows = []
    for image, ns in nuc.groupby('image', sort=False):
        sid = 1
        for n in ns.itertuples():
            for flag, count in [('in_nucleus', n.nuclear_spot_count),
                                ('in_cytoplasm', n.cyto_spot_count)]:
                for _ in range(count):
                    rows.append(dict(image=image, nucleus_id=n.nucleus_id,
                        spot_id=sid, channel='rna1', in_nucleus=int(flag == 'in_nucleus'),
                        in_cytoplasm=int(flag == 'in_cytoplasm')))
                    sid += 1
        rows.append(dict(image=image, nucleus_id=0, spot_id=sid, channel='rna1',
                         in_nucleus=0, in_cytoplasm=0))
    return nuc, pd.DataFrame(rows)


def test_zero_partitions_unassigned_and_duplicate():
    n, s = fixture_frames()
    original = n.copy(deep=True)
    a = agg.reconcile_localization(n, pd.concat([s, s.iloc[[0]]], ignore_index=True))
    assert a['status'] == 'ok'
    assert a['duplicate_rows'] == 1
    assert a['unassigned'].spot_count.sum() == 2
    assert len(a['per_nucleus']) == 5
    np.testing.assert_array_equal(a['per_nucleus'].source_nuclear_spots, [0, 2, 0, 1, 1])
    np.testing.assert_array_equal(a['per_nucleus'].source_cyto_spots, [0, 0, 3, 3, 0])
    np.testing.assert_array_equal(a['per_nucleus'].source_nuclear_fraction.isna(), n.nuclear_spot_fraction.isna())
    assert a['checks'].status.eq('ok').all()
    pd.testing.assert_frame_equal(n, original)


def test_conflicting_duplicate_and_unknown_assignment_are_reported():
    n, s = fixture_frames()
    bad = s.iloc[[0]].copy()
    bad['nucleus_id'] = 999
    a = agg.reconcile_localization(n, pd.concat([s, bad], ignore_index=True))
    assert a['status'] == 'blocked'
    assert 'conflicting spot identity' in ' '.join(a['errors'])
    s.loc[0, 'nucleus_id'] = 999
    a = agg.reconcile_localization(n, s)
    assert a['status'] == 'blocked'
    assert 'outside nucleus roster' in ' '.join(a['errors'])


def test_alias_mismatch_and_missing_are_not_replaced():
    n, s = fixture_frames()
    n.loc[0, 'n_spots_rna1'] = 12
    n = n.drop(columns='n_cytoplasmic_rna1_spots_per_cell')
    a = agg.reconcile_localization(n, s)
    checks = a['checks'].set_index('column')
    assert checks.loc['n_spots_rna1', 'status'] == 'mismatch'
    assert checks.loc['n_cytoplasmic_rna1_spots_per_cell', 'status'] == 'missing'
    assert n.loc[0, 'n_spots_rna1'] == 12
    assert 'n_cytoplasmic_rna1_spots_per_cell' not in n


def test_unequal_fov_mean_of_fractions_and_counts():
    n, _ = fixture_frames()
    n = n.assign(group='WT', well_id='WT_1', secondary_only=False)
    labels = n[['image', 'group', 'well_id', 'secondary_only']].drop_duplicates()
    p = labels.assign(condition='WT_1')
    endpoints = [e for e in ep.ENDPOINTS if e.name in (
        'rna1_nuclear_spot_fraction', 'rna1_nuclear_spots_per_nucleus',
        'rna1_cyto_spots_per_nucleus')]
    assert len(endpoints) == 3
    f = agg.per_field_long(n, p, endpoints, labels, qc_min_nuclei=5)
    w = agg.per_well_long(f).set_index('endpoint')
    assert w.loc['rna1_nuclear_spot_fraction', 'well_mean_of_field_values'] == pytest.approx((1.25 / 3 + 1) / 2)
    assert w.loc['rna1_nuclear_spots_per_nucleus', 'well_mean_of_field_values'] == .875
    assert w.loc['rna1_cyto_spots_per_nucleus', 'well_mean_of_field_values'] == .75
    # QC flags describe low-nucleus fields; they do not exclude them.
    assert f.image.nunique() == 2


def test_registry_amendment_is_only_cytoplasmic_count():
    addition = next(e for e in ep.ENDPOINTS if e.name == 'rna1_cyto_spots_per_nucleus')
    assert addition.column == 'cyto_spot_count'
    assert addition.family == 'detection'
    assert 'per nucleus and assigned cell territory' in addition.unit


@pytest.mark.parametrize('column,value', [('nuclear_spot_fraction', 0.),
    ('nuclear_spot_count', 1), ('cyto_spot_count', np.nan)])
def test_zero_na_or_count_changes_block(column, value):
    n, s = fixture_frames()
    n.loc[0, column] = value
    assert agg.reconcile_localization(n, s)['status'] == 'blocked'


def test_invalid_compartment_flags_block():
    n, s = fixture_frames()
    s.loc[0, ['in_nucleus', 'in_cytoplasm']] = [1, 1]
    assert agg.reconcile_localization(n, s)['status'] == 'blocked'
    s.loc[0, 'in_cytoplasm'] = np.nan
    assert agg.reconcile_localization(n, s)['status'] == 'blocked'


def test_missing_geometry_stays_missing():
    n, s = fixture_frames()
    t = agg.reconcile_localization(n, s)['territory']
    assert t.cyto_area_px.isna().all()
    assert not t.geometry_complete.any()
    assert t.area_partition_matches.isna().all()


@pytest.mark.parametrize('raw_roster', [False, True])
def test_renderer_missing_geometry_and_percent_artists(tmp_path, monkeypatch, raw_roster):
    import matplotlib.pyplot as plt
    monkeypatch.setitem(plt.rcParams, 'svg.fonttype', 'path')
    n, s = fixture_frames()
    n = n.assign(group='WT', well_id='WT_1', secondary_only=False,
                 cyto_estimation_method='voronoi', cyto_area_px=np.nan,
                 cell_area_px=pd.NA, nucleus_area_px=4, voxel_xy_um=.1)
    labels = n[['image', 'group', 'well_id', 'secondary_only']].drop_duplicates()
    eps = [e for e in ep.ENDPOINTS if e.name in ['rna1_nuclear_spot_fraction',
           'rna1_nuclear_spots_per_nucleus', 'rna1_cyto_spots_per_nucleus']]
    f = agg.per_field_long(n, labels.assign(condition='WT_1'), eps, labels, 5)
    w = agg.per_well_long(f)
    if raw_roster:
        n = n.drop(columns=['group', 'well_id'])
    original_nuclei = n.copy(deep=True)
    c = pd.DataFrame([dict(endpoint=e.name, test_group='QKI-KO', reference_group='WT', p_welch=.2)
                      for e in eps])
    ctx = fig.FigureContext(tmp_path, {}, None, ['WT'], 'WT', .05, {}, {'rna1': 'RNA1'})
    def fake_crop(*args):
        mask = np.zeros((10, 10), dtype=bool)
        mask[3:5, 3:5] = True
        return dict(crop=np.zeros((10, 10, 3)), png='fixture.png', nuclear_mask_path='fixture.tif',
            nuclear_mask=mask, x0=0, y0=0, um_per_px=.1, image_shape=(10, 10))
    monkeypatch.setattr(fig, 'crop_for', fake_crop)
    seen = []
    def check_save(canvas, out_dir, stem, manifest, *args):
        assert plt.rcParams['svg.fonttype'] == 'none'
        for ax in canvas.axes:
            if hasattr(ax, '_figure_data'):
                exported = ax._figure_data['nuclei']
                label_map = labels.set_index('image')
                assert exported.group.equals(exported.image.map(label_map.group))
                assert exported.well_id.equals(exported.image.map(label_map.well_id))
            artists = [a for a in ax.collections if a.get_gid() == 'well:WT']
            if artists:
                is_percent = '%' in ax.get_ylabel()
                value = float(artists[0].get_offsets()[0, 1])
                if is_percent:
                    assert value == pytest.approx((1.25 / 3 + 1) / 2 * 100)
                    seen.append('percent')
                else:
                    assert value in (.875, .75)
        plt.close(canvas)
        rec = dict(figure=stem, png=stem+'.png', svg=stem+'.svg')
        manifest.append(rec)
        return rec
    monkeypatch.setattr(fig, 'save', check_save)
    before = w.copy(deep=True)
    result = fig.render_localization(ctx, w, f, n, s, c, tmp_path / 'render', pub_dir=tmp_path)
    assert seen == ['percent', 'percent']
    assert result['crops'][0]['cyto_area_px'] is None
    assert result['crops'][0]['cell_area_px'] is None
    assert result['crops'][0]['territory_boundary'] == 'missing'
    pd.testing.assert_frame_equal(before, w)
    pd.testing.assert_frame_equal(original_nuclei, n)
    f.loc[0, 'field_value'] += .2
    with pytest.raises(agg.ReportInputError, match='full-roster source means'):
        fig.render_localization(ctx, w, f, n, s, c, tmp_path / 'blocked')


def test_persisted_territory_requires_spatial_identity():
    nuclear = np.array([[1, 0, 0], [0, 0, 0]])
    cell = np.array([[1, 1, 0], [0, 0, 0]])
    roster = pd.DataFrame(dict(nucleus_id=[1], cell_area_px=[2]))
    spots = pd.DataFrame(dict(nucleus_id=[1], x_px=[1], y_px=[0]))
    fig.validate_territory_mask(cell, nuclear, roster, spots)
    with pytest.raises(agg.ReportInputError, match='same-ID nuclei'):
        fig.validate_territory_mask(np.roll(cell, 1, axis=1), nuclear, roster, spots)
    spots['x_px'] = 2
    with pytest.raises(agg.ReportInputError, match='Assigned spots'):
        fig.validate_territory_mask(cell, nuclear, roster, spots)
    roster['cell_area_px'] = pd.NA
    with pytest.raises(agg.ReportInputError, match='Missing cell_area_px'):
        fig.validate_territory_mask(cell, nuclear, roster, spots)


RUN = Path('F:/Image Analysis Work/RNASEH2B_BIN1introns_2026_08_25/13b_FULL_HARMONIZED_T36_FIXEDNUCLEAR_2026-09-05/RUN_T36_fixed_2026-09-05_0915')
A0 = Path('E:/Claude/imaging-closeout_2026-09-07/A')
MICROGRAPHS = Path('F:/Image Analysis Work/RNASEH2B_BIN1introns_2026_08_25/DELIVERY_RNASEH2B_BIN1intron_2026-09-05_v3/05_micrographs')


@pytest.mark.skipif(not (RUN / 'nuclei_metrics.csv').is_file(), reason='recorded run missing')
def test_recorded_reconciliation_baseline_parity_and_proposal(tmp_path, monkeypatch):
    from fishsuite.report.build import build_report
    from PIL import Image
    m = json.loads((A0 / 'baseline_manifest.json').read_text())
    # Frozen A0 membership stays pinned; A2 cytoplasmic count and A20 absolute
    # intensity eligibility are explicit amendments, not raw-statistic parity.
    assert {f: len(v) for f, v in m['families'].items()} == dict(detection=4, localization=5, partner=22)
    pins = [p for p in m['named_files'] if p.get('exists') and p['path'].startswith('F:')]
    def verify_hashes():
        for p in pins:
            assert hashlib.sha256(Path(p['path']).read_bytes()).hexdigest().lower() == p['sha256'].lower()
    verify_hashes()
    n, s = pd.read_csv(RUN / 'nuclei_metrics.csv'), pd.read_csv(RUN / 'spot_metrics.csv')
    a = agg.reconcile_localization(n, s)
    assert a['status'] == 'ok', a['errors']
    assert a['duplicate_rows'] == 0
    assert len(a['per_nucleus']) == 521
    assert a['checks'].status.eq('ok').all()
    assert a['territory'].geometry_complete.all()
    assert a['territory'].area_partition_matches.all()
    assert a['per_nucleus'].source_nuclear_spots.sum() == 2701
    assert a['per_nucleus'].source_cyto_spots.sum() == 445
    assert a['unassigned'].spot_count.sum() == 1047
    assert a['per_nucleus'].source_nuclear_fraction.isna().sum() == 115
    out = Path(os.environ.get('FISHSUITE_LOCALIZATION_EVIDENCE', str(tmp_path)))
    out.mkdir(parents=True, exist_ok=True)
    r = build_report(RUN, out_dir=out / 'report', make_figures=False, coloc_panel=False,
        groups=['WT=WT_1,WT_2,WT_3', 'QKI-KO=KO_1,KO_2,KO_3'], reference='WT',
        group_order=['WT', 'QKI-KO'], nucleus_filter='all', sec_min_nuclei=10,
        sec_outlier_k=0, plot_style='replicate-simple', technical_layer='none')
    changed = 'rna1_cyto_spots_per_nucleus'
    family_sizes = (r['contrasts'].assign(_finite_test=lambda c: c.in_holm_family & c.p_welch.notna())
                    .groupby('family')._finite_test.sum().astype(int))
    assert family_sizes.to_dict() == dict(detection=7, localization=5, partner=23)
    additions = {
        'detection': {changed, 'protein_nuclear_mean', 'rna1_nuclear_above_floor_intensity'},
        'localization': set(),
        'partner': {'partner_mean_in_exact_rna1_footprint'},
    }
    for family, frozen in m['families'].items():
        c = r['contrasts']
        actual = set(c.loc[(c.family == family) & c.in_holm_family & c.p_welch.notna(), 'endpoint'])
        assert actual == set(frozen) | additions[family]
    compare(pd.DataFrame(m['baseline_per_well']), r['well'], ['endpoint', 'group', 'well_id'])
    compare(pd.DataFrame(m['baseline_per_field']), r['field'], ['endpoint', 'group', 'well_id', 'image'])
    # Keep every raw statistic and measurement comparison intact. Only declared
    # multiplicity changes and A20 absolute-intensity eligibility may differ.
    amended = {'p_welch_holm_within_family', 'significant_holm_0p05',
        'holm_family_size', 'mde_hedges_g_at_family_alpha', 'observed_g_reaches_mde'}
    old = pd.DataFrame(m['baseline_contrasts'])
    for row in old.itertuples():
        exclusions = set(amended) if row.family in ('detection', 'partner') else set()
        current = r['contrasts'].loc[r['contrasts'].endpoint == row.endpoint]
        if row.absolute_intensity:
            exclusions |= {'descriptive_only', 'excluded_from_holm', 'endpoint_note',
                'in_holm_family', 'holm_exclusion_reason', 'mde_hedges_g_alpha_0p05'}
            assert not current.descriptive_only.any()
            assert current.endpoint_note.eq(ep.INTENSITY_CAVEAT).all()
            assert current.in_holm_family.equals(~current.endpoint_absent_in_run)
        compare(old.loc[old.endpoint == row.endpoint], current,
                ['endpoint', 'reference_group', 'test_group'], exclusions)
    # Source columns remain unchanged including count zeros and fraction NA mask.
    new_n = pd.read_excel(r['xlsx'], sheet_name='Per nucleus', header=1)
    for col in ['nuclear_spot_count', 'cyto_spot_count', 'nuclear_spot_fraction', 'n_spots_rna1']:
        left = n.set_index(['image', 'nucleus_id'])[col].sort_index()
        right = new_n.set_index(['image', 'nucleus_id'])[col].sort_index()
        np.testing.assert_array_equal(left.isna(), right.isna())
        np.testing.assert_allclose(left, right, atol=1e-12, rtol=1e-10, equal_nan=True)
    ctx = fig.FigureContext(RUN, json.loads((RUN / 'run_config.json').read_text()),
        pd.read_csv(RUN / 'thresholds.csv'), ['WT', 'QKI-KO'], 'WT', .05, {}, r['labels'],
        plot_style='replicate-simple')
    fig.set_style()
    original_save = fig.save
    def check_real_artists(canvas, *args, **kwargs):
        for ax in canvas.axes:
            primary = [a for a in ax.collections if str(a.get_gid()).startswith('well:')]
            if primary:
                assert [len(a.get_offsets()) for a in primary] == [3, 3]
                assert not any(str(a.get_gid()).startswith('fov:') for a in ax.collections)
        return original_save(canvas, *args, **kwargs)
    monkeypatch.setattr(fig, 'save', check_real_artists)
    proposal = fig.render_localization(ctx, r['well'], r['field'], n, s, r['contrasts'],
                                     out / 'figures', pub_dir=MICROGRAPHS)
    assert len(proposal['crops']) == 2
    assert all(c['crop_status'] == 'available' for c in proposal['crops'])
    assert all(c['territory_boundary'] == 'missing' for c in proposal['crops'])
    for rec in proposal['figures']:
        with Image.open(out / 'figures' / rec['png']) as im:
            assert im.info['dpi'] == pytest.approx((600, 600), abs=.02)
        svg = (out / 'figures' / rec['svg']).read_text(encoding='utf-8')
        # Locked compact footer carries mixed-model and well-Welch p values;
        # Holm, Hedges g and MDE remain verified in the report contrasts above.
        for text in ['<text', '#595959', '#d67ae5', 'Welch', 'mixed model p']:
            assert text in svg, text
        assert 'descriptive, no test' not in svg
        if not rec.get('is_composite'):
            from fishsuite.report.stats import fmt_p
            endpoint = rec['figure'].removesuffix('_localization')
            contrast = r['contrasts'].loc[r['contrasts'].endpoint == endpoint].iloc[0]
            assert f"Welch (wells) p {fmt_p(contrast.p_welch)}" in svg
    verify_hashes()
    (out / 'acceptance.json').write_text(json.dumps(dict(
        source_nuclei=len(n), source_nuclear_spots=2701, source_cyto_spots=445,
        unassigned_spots=1047, zero_total_fraction_na=115, verified_hashes=len(pins),
        baseline_families={f:len(v) for f,v in m['families'].items()},
        proposed_families=family_sizes.to_dict(), unchanged_raw_statistics='pass; atol=1e-12 rtol=1e-10',
        amended_Holm='explicitly changed; not parity', missing='persisted cell/territory boundary masks'), indent=2))

@pytest.mark.parametrize('mutation', ['missing', 'extra', 'all_na_missing', 'all_na_extra'])
def test_baseline_roster_rejects_changed_keys(mutation):
    old = pd.DataFrame(dict(endpoint=['a', 'b'], group=['WT', 'WT'], value=[1., np.nan]))
    new = old.copy()
    if mutation in ('missing', 'all_na_missing'):
        new = new.drop(1 if mutation == 'all_na_missing' else 0)
    else:
        new = pd.concat([new, pd.DataFrame([dict(endpoint='unexpected', group='WT', value=np.nan if mutation == 'all_na_extra' else 1.)])])
    with pytest.raises(AssertionError, match='keys'):
        compare(old, new, ['endpoint', 'group'])


def test_baseline_roster_all_na_and_declared_addition():
    old = pd.DataFrame(dict(endpoint=['a'], group=['WT'], value=[np.nan]))
    new = pd.concat([old, old.assign(endpoint='rna1_cyto_spots_per_nucleus')])
    compare(old, new, ['endpoint', 'group'])
