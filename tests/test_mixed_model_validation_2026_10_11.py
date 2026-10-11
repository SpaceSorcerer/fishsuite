"""mixed_model returns a p only from a validated fit (multi-optimizer, PD information gate)."""
import warnings

import numpy as np
import pandas as pd
import pytest

from fishsuite.report.sensitivities import mixed_model
from fishsuite.report.figures import mixed_p_text

pytest.importorskip('statsmodels')


def _frame(level, seed=0):
    """level: 'iid' (nucleus noise only), 'fov' (constant inside each FOV), 'well' (constant inside each well)."""
    rng = np.random.default_rng(seed)
    rows = []
    for arm, shift in [('WT', 0.), ('KO', .5 if level == 'iid' else 2.)]:
        for w in range(3):
            v_well = rng.normal()
            for f in range(3):
                v_fov = rng.normal()
                for _ in range(20 if level == 'iid' else 5):
                    value = {'iid': rng.normal(), 'fov': v_fov, 'well': v_well}[level]
                    rows.append(dict(group=arm, well_id=f'{arm}{w}', image=f'{arm}{w}_{f}', value=10 + shift + value))
    return pd.DataFrame(rows)


def _well_posed(seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for arm, shift in [('WT', 0.), ('KO', 1.)]:
        for w in range(4):
            v_well = rng.normal(scale=.7)
            for f in range(3):
                v_fov = rng.normal(scale=.4)
                for _ in range(15):
                    rows.append(dict(group=arm, well_id=f'{arm}{w}', image=f'{arm}{w}_{f}',
                                     value=10 + shift + v_well + v_fov + rng.normal()))
    return pd.DataFrame(rows)


def _statsmodels(data, optimizer):
    from statsmodels.regression.mixed_linear_model import MixedLM
    d = data.copy()
    d['arm'] = d.group.eq('KO').astype(float)
    d['well'] = d.groupby(['group', 'well_id'], sort=True).ngroup().astype(str)
    method = optimizer.split('/')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        from fishsuite.report.sensitivities import MAXITER
        return MixedLM.from_formula('value ~ arm', data=d, groups='well', re_formula='1',
                                    vc_formula={'fov': '0 + C(image)'}, use_sparse=method[0] == 'lbfgs').fit(
            reml=True, method=method, maxiter=MAXITER[method[0]], disp=False)


def test_well_posed_fit_is_validated_and_matches_statsmodels():
    out = mixed_model(_well_posed(), 'KO', 'WT')
    assert out['sensitivity_mixed_validation'] == 'validated'
    assert out['sensitivity_mixed_status'] == 'ok'
    assert out['sensitivity_mixed_min_eig'] > 0
    assert out['sensitivity_mixed_optimizer'] in {'lbfgs/bfgs', 'powell', 'nm'}
    ref = _statsmodels(_well_posed(), out['sensitivity_mixed_optimizer'])
    assert ref.converged
    assert abs(out['sensitivity_mixed_estimate'] - ref.fe_params['arm']) < 1e-6
    assert abs(out['sensitivity_mixed_se'] - ref.bse_fe['arm']) < 1e-6
    assert abs(out['sensitivity_mixed_p'] - ref.pvalues['arm']) < 1e-6


@pytest.mark.parametrize('level', ['fov', 'well'])
def test_zero_within_variance_is_not_validated(level):
    # Before 2026-10-11 these returned status 'ok' with a finite p from a non-PD information matrix.
    out = mixed_model(_frame(level), 'KO', 'WT')
    assert out['sensitivity_mixed_validation'].startswith('not_validated:')
    assert 'positive definite' in out['sensitivity_mixed_validation']
    assert out['sensitivity_mixed_status'] != 'ok'
    assert np.isnan(out['sensitivity_mixed_p'])
    assert np.isnan(out['sensitivity_mixed_estimate']) and np.isnan(out['sensitivity_mixed_se'])


def test_one_nucleus_per_fov_is_not_validated():
    rng = np.random.default_rng(0)
    data = pd.DataFrame([dict(group=arm, well_id=f'{arm}{w}', image=f'{arm}{w}_{f}', value=10 + s + rng.normal())
                         for arm, s in [('WT', 0), ('KO', 1)] for w in range(3) for f in range(3)])
    out = mixed_model(data, 'KO', 'WT')
    assert out['sensitivity_mixed_validation'].startswith('not_validated:')
    assert np.isnan(out['sensitivity_mixed_p'])


def test_lbfgs_nonconvergence_is_rescued_by_another_optimizer():
    data = _frame('iid', seed=4)  # seed chosen 2026-10-11: lbfgs nonconverged, Powell PD
    assert not _statsmodels(data, 'lbfgs/bfgs').converged
    out = mixed_model(data, 'KO', 'WT')
    assert out['sensitivity_mixed_validation'] == 'validated'
    assert out['sensitivity_mixed_optimizer'] in {'powell', 'nm'}
    assert np.isfinite(out['sensitivity_mixed_p'])


def test_constant_arms_not_validated():
    data = pd.DataFrame([dict(group=arm, well_id=f'{arm}{w}', image=f'{arm}{w}_{f}', value=100 + s)
                         for arm, s in [('WT', 0), ('KO', -73)] for w in range(2) for f in range(2) for _ in range(4)])
    out = mixed_model(data, 'KO', 'WT')
    assert out['sensitivity_mixed_validation'].startswith('not_validated:')
    assert np.isnan(out['sensitivity_mixed_p'])


def test_footer_text_never_prints_unvalidated_p():
    out = mixed_model(_frame('well'), 'KO', 'WT')
    stale = dict(out, sensitivity_mixed_p=1e-30, p_mixed=1e-30)  # a p must not leak even if present
    text = mixed_p_text(stale)
    assert text == 'mixed model not validated'
    assert ' p' not in text and 'e-' not in text
    good = mixed_model(_well_posed(), 'KO', 'WT')
    assert mixed_p_text(good).startswith('mixed model p ')


@pytest.mark.parametrize('level', ['fov', 'well'])
def test_footer_render_with_not_validated_contains_no_mixed_p(tmp_path, level):
    import matplotlib.pyplot as plt
    from fishsuite.report import figures as fig
    endpoint = 'rna1_spots_per_nucleus'
    out = mixed_model(_frame(level).replace({'group': {'KO': 'QKI-KO'}}), 'QKI-KO', 'WT')
    assert out['sensitivity_mixed_validation'].startswith('not_validated:')
    wells = pd.DataFrame(dict(endpoint=[endpoint] * 6, group=['WT'] * 3 + ['QKI-KO'] * 3,
                              well_id=['WT_1', 'WT_2', 'WT_3', 'KO_1', 'KO_2', 'KO_3'],
                              well_mean_of_field_values=[1., 2., 3., 4., 5., 6.]))
    # A stale finite p_mixed must not reach the footer when the fit is not validated.
    contrasts = pd.DataFrame([dict(out, endpoint=endpoint, test_group='QKI-KO', reference_group='WT',
                                   p_welch=.2, p_mixed=1e-30)])
    ctx = fig.FigureContext(tmp_path, {}, None, ['WT', 'QKI-KO'], 'WT', .05, {}, {}, plot_style='replicate-simple')
    canvas, ax = plt.subplots()
    try:
        foot = fig.draw_plot(ax, ctx, endpoint, wells, pd.DataFrame(), pd.DataFrame(), contrasts, 'Puncta', None)
        drawn = ' '.join(t.get_text() for t in ax.texts)
    finally:
        plt.close(canvas)
    assert 'mixed model not validated' in foot
    assert 'mixed model p' not in foot and '1e-30' not in foot and '1.00e-30' not in foot
    assert '1.00e-30' not in drawn and 'Welch (well means)' in drawn


@pytest.mark.parametrize('status,expected', [
    ('missing: endpoint has only field-level measurements', 'mixed model not fitted (field-level endpoint)'),
    ('blocked: mixed model did not converge', 'mixed model did not converge'),
    ('blocked: mixed model not validated (lbfgs/bfgs: did not converge; powell: information matrix not positive definite)',
     'mixed model not validated'),
    ('blocked: fewer than two wells per arm with finite nucleus values', 'mixed model not validated')])
def test_footer_reason_is_single_and_specific(tmp_path, status, expected):
    import matplotlib.pyplot as plt
    from fishsuite.report import figures as fig
    endpoint = 'rna1_spots_per_nucleus'
    wells = pd.DataFrame(dict(endpoint=[endpoint] * 6, group=['WT'] * 3 + ['QKI-KO'] * 3,
                              well_id=['w1', 'w2', 'w3', 'k1', 'k2', 'k3'],
                              well_mean_of_field_values=[1., 2., 3., 4., 5., 6.]))
    rows = pd.DataFrame([dict(endpoint=endpoint, test_group='QKI-KO', reference_group='WT',
                              p_mixed=np.nan, p_welch=.2, sensitivity_mixed_status=status)])
    ctx = fig.FigureContext(tmp_path, {}, None, ['WT', 'QKI-KO'], 'WT', .05, {}, {}, plot_style='replicate-simple')
    canvas, ax = plt.subplots()
    try:
        foot = fig.draw_replicate_simple(ax, ctx, endpoint, wells, pd.DataFrame(), pd.DataFrame(), rows, 'Puncta', None)
    finally:
        plt.close(canvas)
    assert expected in foot
    assert not ('not validated' in foot and 'did not converge' in foot)
    assert 'mixed model p' not in foot


def test_field_level_endpoint_status_from_scalar_sensitivities():
    from types import SimpleNamespace
    from fishsuite.report.sensitivities import scalar_sensitivities
    field = pd.DataFrame(dict(group=['KO'] * 3 + ['WT'] * 3, field_value=[1., 2., 3., 4., 5., 6.]))
    ep = SimpleNamespace(level='field', column='x', usability_flag=None)
    out = scalar_sensitivities(ep, None, field, np.array([1., 2.]), np.array([3., 4.]), 'KO', 'WT')
    assert mixed_p_text(out) == 'mixed model not fitted (field-level endpoint)'


def test_fit_that_hits_the_iteration_cap_is_never_validated(monkeypatch):
    from fishsuite.report import sensitivities as S
    monkeypatch.setattr(S, 'MAXITER', {'lbfgs': 1, 'powell': 1, 'nm': 2})
    out = S.mixed_model(_well_posed(), 'KO', 'WT')
    assert out['sensitivity_mixed_validation'].startswith('not_validated:')
    assert 'did not converge' in out['sensitivity_mixed_validation']
    assert np.isnan(out['sensitivity_mixed_p'])
