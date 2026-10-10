"""report.stats.mde_hedges_g: simulation-based Welch MDE in pooled-SD units, no Hedges j (2026-10-10)."""
import numpy as np
import pytest
from scipy import stats

from fishsuite.report import stats as st


def _realised_welch_power(g, alpha, n1, n2, sd1, sd2, reps=40_000, seed=12345):
    """Independent of stats.py: scipy's Welch test on fresh draws (different seed from the MDE search)."""
    rng = np.random.default_rng(seed)
    sp = np.sqrt(((n1 - 1) * sd1 ** 2 + (n2 - 1) * sd2 ** 2) / (n1 + n2 - 2))
    a = rng.normal(g * sp, sd1, (reps, n1))
    b = rng.normal(0.0, sd2, (reps, n2))
    return float(np.mean(stats.ttest_ind(a, b, axis=1, equal_var=False).pvalue < alpha))


@pytest.mark.parametrize("n", [2, 3, 4])
@pytest.mark.parametrize("sd1,sd2", [(1.0, 1.0), (1.0, 3.0)])
def test_simulated_power_at_the_mde_is_eighty_percent_at_two_three_and_four_per_arm(n, sd1, sd2):
    g = st.mde_hedges_g(0.05, n1=n, n2=n, sd1=sd1, sd2=sd2)
    assert _realised_welch_power(g, 0.05, n, n, sd1, sd2) == pytest.approx(0.80, abs=0.02)


@pytest.mark.parametrize("n1,n2,alpha", [(3, 3, 0.05 / 6), (4, 6, 0.05)])
def test_simulated_power_at_family_alpha_and_unequal_arms(n1, n2, alpha):
    g = st.mde_hedges_g(alpha, n1=n1, n2=n2, sd1=1.0, sd2=2.0)
    assert _realised_welch_power(g, alpha, n1, n2, 1.0, 2.0) == pytest.approx(0.80, abs=0.02)


def test_mde_is_reproducible_and_needs_at_least_5000_reps_by_default():
    assert st.MDE_SIM_REPS >= 5000
    assert st.mde_hedges_g(0.05, n1=3, n2=3, sd1=1.0, sd2=2.0) == st.mde_hedges_g(0.05, n1=3, n2=3, sd1=1.0, sd2=2.0)


def test_mde_exceeds_the_student_noncentral_t_value_at_small_n():
    # The Student/noncentral-t MDE gives the realised Welch test only ~70% power at 3 v 3.
    student = 3.0708  # statsmodels TTestIndPower nobs1=3, ratio=1, alpha=0.05, power=0.8
    from statsmodels.stats.power import TTestIndPower
    assert TTestIndPower().solve_power(nobs1=3, ratio=1.0, alpha=0.05, power=0.80) == pytest.approx(student, abs=1e-3)
    assert st.mde_hedges_g(0.05, n1=3, n2=3) > student


def test_mde_is_nan_for_invalid_designs():
    assert np.isnan(st.mde_hedges_g(0.05, n1=1, n2=1))
    assert np.isnan(st.mde_hedges_g(0.05, n1=2, n2=1))
    assert np.isnan(st.mde_hedges_g(0.05, n1=3, n2=3, sd1=0.0, sd2=0.0))
    assert np.isnan(st.mde_hedges_g(0.05, n1=3, n2=3, sd1=np.nan, sd2=1.0))
