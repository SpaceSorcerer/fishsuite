"""Persisted statistics without validated mixed inference are refused, never repaired (2026-10-11)."""
import numpy as np
import pandas as pd
import pytest

from fishsuite.report import sensitivities as S

REBUILD = 'legacy statistics: rebuild from the source run with fishsuite >= '


def _legacy_contrasts():
    # Pre-70500c0 shape: status 'ok', a cached mixed p of 1e-30 as headline, no validation column.
    return pd.DataFrame([
        dict(endpoint='a', family='detection', test_group='KO', reference_group='WT', in_holm_family=True,
             sensitivity_mixed_status='ok', sensitivity_mixed_p=1e-30, p_mixed=1e-30, p_welch=.02,
             p_headline=1e-30, headline_test='mixed model', p_headline_holm=2e-30, p_mixed_holm=2e-30,
             p_welch_holm_within_family=.04, significant_headline_holm_0p05=True, diff=1.),
        dict(endpoint='b', family='detection', test_group='KO', reference_group='WT', in_holm_family=True,
             sensitivity_mixed_status='ok', sensitivity_mixed_p=1e-20, p_mixed=1e-20, p_welch=.3,
             p_headline=1e-20, headline_test='mixed model', p_headline_holm=1e-20, p_mixed_holm=1e-20,
             p_welch_holm_within_family=.3, significant_headline_holm_0p05=True, diff=-1.)])


@pytest.mark.parametrize('variant', ['no_column', 'not_validated', 'nan_marker', 'headline_only', 'holm_only'])
def test_legacy_or_unvalidated_table_is_refused(variant):
    c = _legacy_contrasts()
    if variant == 'not_validated':
        c['sensitivity_mixed_validation'] = ['validated', 'not_validated:PD failed']
    elif variant == 'nan_marker':
        c['sensitivity_mixed_validation'] = ['validated', np.nan]
    elif variant == 'headline_only':  # an incomplete schema must not escape
        c = c[['endpoint', 'p_headline', 'headline_test']]
    elif variant == 'holm_only':
        c = c[['endpoint', 'p_headline_holm']]
    with pytest.raises(S.LegacyStatisticsError, match=REBUILD):
        S.require_validated_mixed(c)


def test_validated_table_passes_unchanged():
    c = _legacy_contrasts().assign(sensitivity_mixed_validation='validated')
    out = S.require_validated_mixed(c)
    assert out is c
    welch_headline = _legacy_contrasts().assign(sensitivity_mixed_validation='not_validated:x', p_mixed=np.nan,
                                                sensitivity_mixed_p=np.nan, p_mixed_holm=np.nan,
                                                headline_test='Welch (well means)')
    assert S.require_validated_mixed(welch_headline) is welch_headline


def test_rebuild_message_names_a_commit():
    assert S.MIXED_VALIDATION_COMMIT != 'COMMIT_PENDING'
    assert S.REBUILD_MESSAGE == REBUILD + S.MIXED_VALIDATION_COMMIT


def test_workbook_write_refuses_legacy_sheet(tmp_path):
    from fishsuite.report import workbook
    with pytest.raises(S.LegacyStatisticsError, match=REBUILD):
        workbook.write(tmp_path / 'r.xlsx', {'Contrasts': _legacy_contrasts()}, {}, order=['Contrasts'])


def test_rn_readout_refuses_legacy_contrasts():
    from fishsuite.report.object_delivery import rn_readout
    with pytest.raises(S.LegacyStatisticsError, match=REBUILD):
        rn_readout(_legacy_contrasts(), {'ratio': {}})


def test_slide_cache_tracing_to_legacy_source_is_refused():
    contrasts = [('legacy',), ('endpoint', 'p_mixed'), ('a', 1e-5)]
    cache = [('cache',), ('slide', 'source_sheet', 'source_cell', 'label', 'value'), (1, 'Contrasts', 'B3', 'a: p_mixed', 1e-5)]
    with pytest.raises(S.LegacyStatisticsError, match=REBUILD):
        S.require_validated_rows(contrasts, 'Contrasts')
    # Codex case: the source column was cleared or dropped, yet the cache still carries the number.
    cleared = [('legacy',), ('endpoint', 'p_welch'), ('a', .8)]
    with pytest.raises(S.LegacyStatisticsError, match=REBUILD):
        S.require_validated_slide_cache(cache, {'Contrasts': cleared})
    validated = [('x',), ('endpoint', 'p_mixed', 'sensitivity_mixed_validation'), ('a', 1e-5, 'validated')]
    S.require_validated_slide_cache(cache, {'Contrasts': validated})


def test_final_delivery_narrative_does_not_use_legacy_mixed_p():
    from fishsuite.report import final_delivery
    spec = {'slides': [{'identity': 'q1_localization'}]}
    contrasts = _legacy_contrasts().assign(endpoint=['nuclear_spot_fraction_dapi', 'b'])
    body = final_delivery.narrative(spec, contrasts)['slides'][0]['body']
    assert 'detects' not in body and 'mixed model not validated (legacy result)' in body.lower()


def test_build_time_producers_match_the_declared_rule():
    fresh = pd.DataFrame(dict(endpoint=['x', 'y', 'z'], sensitivity_mixed_p=[np.nan, .02, .01], p_welch=[.8, .4, .03],
                              sensitivity_mixed_validation=['not_validated:failed', 'validated', 'validated'],
                              family=['f', 'f', 'g'], test_group=['t', 't', 't'], in_holm_family=[True, True, True]))
    out = S.add_headline_holm(S.add_mixed_columns(fresh), .05, ['family', 'test_group'])
    assert np.isnan(out.p_mixed.iloc[0]) and out.p_headline.tolist() == [.8, .02, .01]
    assert out.headline_test.tolist() == ['Welch (well means)', 'mixed model', 'mixed model']
    assert out.p_headline_holm.tolist() == pytest.approx([.8, .04, .01])
    assert np.isnan(out.p_mixed_holm.iloc[0]) and out.holm_family_size.tolist() == [2, 2, 1]
