"""Persisted-only panel import and frozen-cohort integration."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

ROOT = Path('F:/Image Analysis Work/RNASEH2B_BIN1introns_2026_08_25')
PANEL = ROOT / 'DELIVERY_RNASEH2B_BIN1intron_2026-09-05_v3/02_colocalization_panel/coloc_standard_panel.xlsx'
RUN = ROOT / '13b_FULL_HARMONIZED_T36_FIXEDNUCLEAR_2026-09-05/RUN_T36_fixed_2026-09-05_0915'
BASELINE = Path('E:/Claude/imaging-closeout_2026-09-07/A/baseline_manifest.json')


def test_import_exact_persisted_cells_and_anchors():
    from fishsuite.report.coloc_existing import load_existing_panel
    panel = load_existing_panel(PANEL, PANEL.with_name('FIGURE_INDEX.md'), BASELINE)
    original = pd.read_excel(PANEL, sheet_name='per_well')
    pd.testing.assert_frame_equal(panel.tables['per_well'][original.columns], original)
    assert panel.tables['per_well']['group'].equals(original['line'])
    assert len(panel.sources) > original.size
    assert {'paired_frac_rna1_at_partner', 'frac_called_coloc_partner_runthr'} <= set(panel.tables['per_nucleus'])
    assert panel.tables['per_nucleus'].loc[lambda x: ~x.costes_converged, 'manders_m1_costes_only'].isna().all()


def test_missing_stale_and_frozen_guard(tmp_path):
    from fishsuite.report.coloc_existing import load_existing_panel
    from fishsuite.report.provenance import guard_output
    from fishsuite.report.aggregate import ReportInputError
    with pytest.raises(ReportInputError, match='missing'):
        load_existing_panel(tmp_path/'missing.xlsx', tmp_path/'missing.md', BASELINE)
    with pytest.raises(ReportInputError, match='frozen'):
        guard_output(tmp_path/'DELIVERY_any'/'report')
    m = json.loads(BASELINE.read_text())
    for row in m['named_files']:
        if Path(row['path']) == PANEL:
            row['sha256'] = '0'*64
    stale = tmp_path/'stale.json'
    stale.write_text(json.dumps(m))
    with pytest.raises(ReportInputError, match='stale'):
        load_existing_panel(PANEL, PANEL.with_name('FIGURE_INDEX.md'), stale)
    with pytest.raises(ReportInputError, match='stale figure index'):
        load_existing_panel(PANEL, PANEL.with_name('FIGURE_INDEX.md'), BASELINE, '0'*64)


def test_hand_worked_nuclear_pairing_and_zero_anchors():
    from fishsuite.report.coloc_existing import validate_pairing
    spots = pd.DataFrame(dict(image=['a']*5, nucleus_id=[1,1,1,2,0],
                              channel=['rna1']*5, in_nucleus=[1,1,0,0,0],
                              paired_at_0p3um=[1,0,1,1,1]))
    panel = pd.DataFrame(dict(image=['a','a'], nucleus_id=[1,2],
                             paired_frac_rna1_at_partner=[.5,np.nan]))
    validate_pairing(panel, spots)
    panel.loc[0,'paired_frac_rna1_at_partner'] = 2/3
    with pytest.raises(RuntimeError, match='nuclear pairing'):
        validate_pairing(panel, spots)


def test_report_integration_no_nulls_raw_parity(tmp_path, monkeypatch):
    from fishsuite.report.build import build_report
    import fishsuite.report.build as build
    from fishsuite.report.endpoints import A3_PARTNER_ADDITIONS
    def forbidden(*a, **kw):
        pytest.fail('new-null invocation')
    monkeypatch.setattr(build, 'run_coloc_standard_panel', forbidden)
    result = build_report(RUN, tmp_path/'report', groups=['WT=WT_1,WT_2,WT_3', 'QKI-KO=KO_1,KO_2,KO_3'],
                          reference='WT', make_figures=False, existing_coloc=PANEL,
                          baseline_manifest=BASELINE)
    audit = pd.read_excel(result['xlsx'], sheet_name='Localization counts', header=1)
    unassigned = pd.read_excel(result['xlsx'], sheet_name='Localization unassigned', header=1)
    assert audit.source_nuclear_fraction.isna().sum() == 115
    assert unassigned.spot_count.sum() == 1047
    baseline = json.loads(BASELINE.read_text())
    old = pd.DataFrame(baseline['baseline_contrasts']).set_index('endpoint')
    new = result['contrasts'].set_index('endpoint')
    for name in old.index:
        amended = {'p_welch_holm_within_family','significant_holm_0p05','holm_family_size',
                   'mde_hedges_g_at_family_alpha','observed_g_reaches_mde'} if old.loc[name,'family'] in {'detection','partner'} else set()
        for col in old.columns:
            if col in amended:
                continue
            before,after=old.loc[name,col],new.loc[name,col]
            if before is None or pd.isna(before):
                assert pd.isna(after) or after=='', (name,col)
            elif isinstance(before,(int,float,np.number,bool)):
                np.testing.assert_allclose(float(after),float(before),atol=1e-12,rtol=1e-10,equal_nan=True)
            else:
                assert before==after,(name,col)
    for key,table,indices in [('baseline_per_field',result['field'],['endpoint','group','well_id','image']),
                              ('baseline_per_well',result['well'],['endpoint','group','well_id'])]:
        before=pd.DataFrame(baseline[key]).set_index(indices).sort_index()
        after=table.set_index(indices).sort_index().loc[before.index,before.columns]
        pd.testing.assert_frame_equal(before,after,check_dtype=False,atol=1e-12,rtol=1e-10)
    assert set(A3_PARTNER_ADDITIONS) <= set(new.index)
    assert set(new.loc[new.family=='partner'].holm_family_size.dropna()) == {27}
    assert not new.loc['cell_total_intensity_protein','in_holm_family']
    assert pd.isna(new.loc['paired_frac_rna1_at_partner_shuffle','p_welch'])
    assert 'reporter commit:' in (tmp_path/'report/versions.txt').read_text()
    assert 'producing engine commit: missing from run' in (tmp_path/'report/versions.txt').read_text()
    with pytest.raises(RuntimeError, match='persisted|gates'):
        build_report(RUN, tmp_path/'bad', existing_coloc=PANEL, baseline_manifest=BASELINE,
                     peak_floors={'rna1':999}, make_figures=False)


def test_hand_worked_reverse_calls_total_if_and_unequal_fovs():
    from fishsuite.report.aggregate import per_field_long, per_well_long
    from fishsuite.report.endpoints import a3_endpoints
    nuc=pd.DataFrame(dict(image=['a','a','b'],nucleus_id=[1,2,1],group=['WT']*3,
                          well_id=['WT_1']*3,secondary_only=[False]*3,
                          cell_total_intensity_protein=[10.,30.,80.],
                          nuclear_total_intensity_protein=[4.,8.,20.],
                          rna1_local_mean_at_protein_spots=[3.,9.,30.],
                          frac_called_coloc_partner_runthr=[2/4,0/2,1/1],
                          paired_frac_rna1_at_partner=[1/2,np.nan,1/1],
                          paired_frac_rna1_at_partner_minus_shuffle=[.5,np.nan,.25],
                          frac_called_coloc_partner_minus_shuffle_runthr=[.5,-.25,.25],
                          paired_frac_partner_at_rna1_minus_shuffle=[.25,0.,.5]))
    labels=nuc[['image','group','well_id','secondary_only']].drop_duplicates()
    from fishsuite.report.endpoints import A3_PARTNER_ADDITIONS
    endpoints=a3_endpoints(A3_PARTNER_ADDITIONS)
    fields=per_field_long(nuc,pd.DataFrame({'image':['a','b']}),endpoints,labels,5)
    wells=per_well_long(fields).set_index('endpoint')
    assert wells.loc['cell_total_intensity_protein','well_mean_of_field_values']==50
    assert wells.loc['nuclear_total_intensity_protein','well_mean_of_field_values']==13
    assert wells.loc['rna1_local_mean_at_partner_puncta','well_mean_of_field_values']==18
    assert wells.loc['frac_called_coloc_partner_runthr','well_mean_of_field_values']==.625
    assert wells.loc['paired_frac_rna1_at_partner','well_mean_of_field_values']==.75
    # Persisted excess values include negatives and undefined anchors. Average
    # defined nuclei inside each FOV, then give the two FOVs equal weight.
    assert wells.loc['paired_frac_rna1_at_partner_minus_shuffle','well_mean_of_field_values']==.375
    assert wells.loc['frac_called_coloc_partner_minus_shuffle_runthr','well_mean_of_field_values']==.1875
    assert wells.loc['paired_frac_partner_at_rna1_minus_shuffle','well_mean_of_field_values']==.3125


def test_line_group_conflict_and_cohort_rejected():
    from fishsuite.report.coloc_existing import _groups,load_existing_panel,integrate_panel
    from fishsuite.report.aggregate import load_run
    from fishsuite.report.endpoints import resolve
    with pytest.raises(RuntimeError,match='conflicting'):
        _groups(pd.DataFrame({'line':['WT'],'group':['QKI-KO']}))
    data=load_run(RUN,{'WT_1':'WT','WT_2':'WT','WT_3':'WT','KO_1':'QKI-KO','KO_2':'QKI-KO','KO_3':'QKI-KO'}, {})
    endpoints,_=resolve(data['nuclei'],data['per_image'])
    panel=load_existing_panel(PANEL,PANEL.with_name('FIGURE_INDEX.md'),BASELINE)
    panel.tables['per_nucleus']=panel.tables['per_nucleus'].iloc[1:]
    with pytest.raises(RuntimeError,match='cohort mismatch'):
        integrate_panel(data,panel,endpoints)


def test_default_rendering_omitted_localization_aliases(tmp_path):
    from test_report_localization import fixture_frames
    from fishsuite.report.build import build_report
    n, spots = fixture_frames()
    n = n.drop(columns=['rna_spot_count', 'n_spots_rna1', 'n_cytoplasmic_rna1_spots_per_cell'])
    n['nucleus_area_px'] = 100
    n['voxel_xy_um'] = .1
    run = tmp_path/'run'
    run.mkdir()
    n.to_csv(run/'nuclei_metrics.csv', index=False)
    spots.to_csv(run/'spot_metrics.csv', index=False)
    pd.DataFrame(dict(image=['a','b'], condition=['WT_1','KO_1'], secondary_only=[False,False])).to_csv(run/'per_image_summary.csv', index=False)
    (run/'run_config.json').write_text('{}')
    result = build_report(run, tmp_path/'report', groups=['WT=WT_1','KO=KO_1'],
                          make_figures=True, coloc_panel=False)
    audit = pd.read_excel(result['xlsx'], sheet_name='Localization counts', header=1)
    assert audit.source_nuclear_fraction.isna().sum() == 1
    assert (tmp_path/'report/localization/FIG_LOCALIZATION.svg').is_file()


