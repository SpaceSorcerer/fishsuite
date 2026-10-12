"""Workbooks built before 70500c0 carry no sensitivity_mixed_validation column; their mixed p is never shown."""
import numpy as np
import pandas as pd
import pytest

from fishsuite.report.sensitivities import withhold_unvalidated_mixed

LEGACY_TEXT = 'mixed model not validated (legacy result)'


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


def test_legacy_table_withholds_every_mixed_p_and_falls_back_to_welch():
    c = withhold_unvalidated_mixed(_legacy_contrasts())
    assert c[['p_mixed', 'sensitivity_mixed_p', 'p_mixed_holm']].isna().all().all()
    assert c.p_headline.tolist() == [.02, .3]
    assert (c.headline_test == 'Welch (well means)').all()
    assert c.p_headline_holm.tolist() == pytest.approx([.04, .3])
    assert c.significant_headline_holm_0p05.tolist() == [True, False]
    assert c.sensitivity_mixed_validation.str.contains('legacy').all()


def test_validated_rows_keep_their_mixed_p():
    c = _legacy_contrasts().assign(sensitivity_mixed_validation=['validated', 'not_validated:x'])
    out = withhold_unvalidated_mixed(c)
    assert out.p_mixed.iloc[0] == 1e-30 and np.isnan(out.p_mixed.iloc[1])
    assert out.p_headline.tolist() == [1e-30, .3]
    assert out.headline_test.tolist() == ['mixed model', 'Welch (well means)']


def test_final_delivery_narrative_does_not_use_legacy_mixed_p():
    from fishsuite.report import final_delivery
    spec = {'slides': [{'identity': 'q1_localization'}]}
    contrasts = _legacy_contrasts().assign(endpoint=['nuclear_spot_fraction_dapi', 'b'])
    body = final_delivery.narrative(spec, contrasts)['slides'][0]['body']
    assert 'detects' not in body and LEGACY_TEXT in body.lower()
    body = final_delivery.narrative(spec, withhold_unvalidated_mixed(contrasts))['slides'][0]['body']
    assert 'detects' not in body and LEGACY_TEXT in body.lower()


def test_slide_notes_from_legacy_workbook_print_no_mixed_p(tmp_path):
    from fishsuite.report.slides import speaker_notes
    row = _legacy_contrasts().iloc[0].to_dict()
    definition = {'identity': 'x', 'values': [dict(sheet='Contrasts', label=f'a: {k}', value=v) for k, v in row.items()]}
    notes = speaker_notes(tmp_path / 'REPORT.xlsx', definition)
    assert '1e-30' not in notes and '1.00e-30' not in notes and '2e-30' not in notes
    assert 'mixed model p' not in notes and 'Wald p' not in notes
    assert LEGACY_TEXT in notes
    assert 'Holm p (well-level Welch) 0.04' in notes
