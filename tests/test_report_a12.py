import numpy as np
import pandas as pd
from fishsuite.report.sensitivities import mixed_model, flag_fov_outliers, scalar_sensitivities
from fishsuite.report.stats import welch


def fixture():
    rows = []
    for arm, shift in [('WT', 0), ('KO', 4)]:
        for w, offset in enumerate([-1, 0, 1]):
            for f, effect in enumerate([-.5, 0, .5]):
                for n, residual in enumerate([-.3, -.1, .1, .3]):
                    rows.append(dict(group=arm, well_id=f'{arm}{w}', image=f'{arm}{w}_{f}', value=10+shift+offset+effect+residual))
    return pd.DataFrame(rows)


def test_mixed_model_hand_worked_direction_and_counts():
    data = fixture()
    fit = mixed_model(data, 'KO', 'WT')
    assert fit['sensitivity_mixed_status'] == 'ok'
    assert np.isclose(fit['sensitivity_mixed_estimate'], 4, atol=1e-5)
    assert fit['sensitivity_mixed_se'] > 0
    assert 0 <= fit['sensitivity_mixed_p'] <= 1
    assert fit['sensitivity_n_nuclei_test'] == 36
    assert fit['sensitivity_n_fovs_test'] == 9
    assert fit['sensitivity_n_wells_test'] == 3
    gate = welch(np.array([13.,14.,15.]), np.array([9.,10.,11.]))
    assert gate['diff'] == 4


def test_no_pooled_nucleus_t_is_emitted(monkeypatch):
    import scipy.stats
    calls = []
    original = scipy.stats.ttest_ind
    def counted(a,b,**kwargs):
        calls.append((len(a),len(b),kwargs.get('equal_var')))
        return original(a,b,**kwargs)
    monkeypatch.setattr(scipy.stats, 'ttest_ind', counted)
    from types import SimpleNamespace
    data = fixture().rename(columns={'value':'measurement'})
    ep = SimpleNamespace(level='nucleus', column='measurement', usability_flag=None)
    field = data.groupby(['group','well_id','image'], as_index=False).measurement.mean().rename(columns={'measurement':'field_value'})
    result = scalar_sensitivities(ep, data, field, np.array([13.,14.,15.]), np.array([9.,10.,11.]), 'KO', 'WT')
    assert calls == [(3,3,True),(9,9,False)]
    assert 'sensitivity_student_well_p' in result
    assert 'sensitivity_welch_fov_p' in result
    assert not any('nucleus_t' in key or 'pooled_nuc' in key for key in result)


def test_outliers_minimum_planted_and_cap():
    def frame(values):
        return pd.DataFrame(dict(endpoint='count',group='WT',well_id='w',image=[str(i) for i in range(len(values))],field_value=values,n_nuclei_nonmissing=10))
    assert flag_fov_outliers(frame([0,100])).empty
    flags = flag_fov_outliers(frame([1,1.1,.9,100]))
    assert flags.image.tolist() == ['3']
    assert abs(flags.z.iloc[0]) > 2.5
    assert len(flag_fov_outliers(frame([-100,100]+[0]*10))) == 1
    both = pd.concat([frame([1,1.1,.9,100]), frame([1,1.1,.9,100]).assign(group='KO',well_id='k')])
    assert set(flag_fov_outliers(both).arm) == {'WT','KO'}
    assert flag_fov_outliers(frame([1,1,1])).empty


def test_four_configured_panels(tmp_path):
    from fishsuite.report.figures import publication_panel_paths
    cfg = {'channels':dict(analysis_mode='rna_protein',rna_label='MIAT 647',antibody_label='QKI 568',rna_lut='yellow',antibody_lut='magenta'), 'output':dict(manual_rna_min=10,manual_rna_max=100,manual_antibody_min=20,manual_antibody_max=200,manual_dapi_min=0,manual_dapi_max=200)}
    for name in ['merge_all','merge_MIAT_647_QKI_568','QKI_568_magenta','MIAT_647_yellow']:
        (tmp_path/f'field__{name}.png').write_bytes(b'fixture')
    panels = publication_panel_paths(tmp_path, 'field', cfg)
    assert len(panels) == 4
    assert not any('DAPI-only' in x['panel'] for x in panels)
    assert panels[2]['panel'] == 'QKI 568'


def test_plain_notes_numeric_contrasts(tmp_path):
    from fishsuite.report.slides import speaker_notes
    values = [dict(label='count: '+k,value=v,sheet='Contrasts') for k,v in dict(endpoint_plain='Puncta',test_group='KO',reference_group='WT',mean_test=14,mean_ref=10,diff=4,p_welch=.01,hedges_g=2,p_welch_holm_within_family=.02,mde_hedges_g_at_family_alpha=3,n_wells_test=3,n_wells_reference=3,sensitivity_n_nuclei_test=36,sensitivity_n_nuclei_reference=36,sensitivity_n_fovs_test=9,sensitivity_n_fovs_reference=9).items()]
    text = speaker_notes(tmp_path/'REPORT.xlsx',dict(values=values))
    assert text.startswith('Workbook: '+str((tmp_path/'REPORT.xlsx').resolve()))
    assert '14' in text and '0.01' in text and 'Levels of comparison' in text
    assert '!' not in text


def test_outlier_recomputation_preserves_headline():
    from fishsuite.report.sensitivities import outlier_sensitivity
    frame = pd.DataFrame([dict(endpoint='count',group=arm,well_id=str(w),image=f'{arm}{w}{f}',field_value=value,n_nuclei_nonmissing=10)
        for arm in ['WT','KO'] for w in range(3) for f,value in enumerate([w+1,w+1.1,w+.9,100+w])])
    gate = pd.DataFrame([dict(endpoint='count',test_group='KO',reference_group='WT',p_welch=1.,diff=0.)])
    before = gate.copy(deep=True)
    result = outlier_sensitivity(frame,gate)
    pd.testing.assert_frame_equal(gate,before)
    comparison = result.loc[result.record_type.eq('contrast')].iloc[0]
    assert comparison.with_exclusion_mean_test == 2
    assert comparison.with_exclusion_welch_p == 1
    assert len(result.loc[result.record_type.eq('flagged FOV')]) == 6


def test_mixed_missing_dependency_reports_blocker(monkeypatch):
    import builtins
    original = builtins.__import__
    def guarded(name,*args,**kwargs):
        if name == 'statsmodels.regression.mixed_linear_model':
            raise ImportError('fixture missing package')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    result = mixed_model(fixture(),'KO','WT')
    assert 'statsmodels missing' in result['sensitivity_mixed_status']
    assert np.isnan(result['sensitivity_mixed_p'])
