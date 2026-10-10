"""report.stats.mde_hedges_g returns the population delta at 80% power for every n (2026-10-10)."""
import numpy as np
import pytest

from fishsuite.report import stats as st


@pytest.mark.parametrize("n", [2, 3, 4])
@pytest.mark.parametrize("alpha", [0.05, 0.05 / 4])
def test_mde_has_eighty_percent_power_at_two_three_and_four_per_arm(n, alpha):
    assert st._power_two_sided(st.mde_hedges_g(alpha, n1=n, n2=n), alpha, n, n) == pytest.approx(0.80, abs=1e-6)


def test_mde_is_nan_below_two_per_arm():
    assert np.isnan(st.mde_hedges_g(0.05, n1=1, n2=1))
    assert np.isnan(st.mde_hedges_g(0.05, n1=2, n2=1))


def test_welch_mde_equals_statsmodels_student_value_for_equal_sds_and_arms():
    from statsmodels.stats.power import TTestIndPower
    known = TTestIndPower().solve_power(nobs1=3, ratio=1.0, alpha=0.05, power=0.80)
    assert st.mde_hedges_g(0.05, n1=3, n2=3, sd1=2.0, sd2=2.0) == pytest.approx(known, abs=1e-4)
    assert st.mde_hedges_g(0.05, n1=3, n2=3) == pytest.approx(known, abs=1e-4)


@pytest.mark.parametrize("n1,n2,sd1,sd2,alpha", [(3, 3, 1.0, 3.0, 0.05), (3, 3, 1.0, 3.0, 0.05 / 6),
                                                 (2, 2, 1.0, 2.0, 0.05), (4, 6, 1.0, 3.0, 0.05)])
def test_welch_mde_has_eighty_percent_power_by_independent_monte_carlo(n1, n2, sd1, sd2, alpha):
    from fishsuite.report.dapi_delivery import _welch_power_monte_carlo

    def arm(n, sd):  # n values with sample SD exactly sd
        z = np.arange(n, dtype=float)
        return (z - z.mean()) / z.std(ddof=1) * sd

    g = st.mde_hedges_g(alpha, n1=n1, n2=n2, sd1=sd1, sd2=sd2)
    assert _welch_power_monte_carlo(g, alpha, arm(n1, sd1), arm(n2, sd2)) == pytest.approx(0.80, abs=0.003)


def test_welch_mde_differs_from_student_when_sds_differ():
    student = st.mde_hedges_g(0.05, n1=3, n2=3)
    assert st.mde_hedges_g(0.05, n1=3, n2=3, sd1=1.0, sd2=3.0) > student
    assert st.mde_hedges_g(0.05, n1=4, n2=6, sd1=1.0, sd2=3.0) != pytest.approx(st.mde_hedges_g(0.05, n1=4, n2=6), abs=1e-3)
    assert np.isnan(st.mde_hedges_g(0.05, n1=3, n2=3, sd1=0.0, sd2=0.0))
    assert np.isnan(st.mde_hedges_g(0.05, n1=3, n2=3, sd1=np.nan, sd2=1.0))


def test_realised_welch_power_at_the_mde_is_below_nominal_at_three_wells():
    """Recorded observation, not a calibration: with sample SDs the realised Welch test at 3 v 3 has
    less than the nominal 80% power at the MDE (simulation, seed 0, 40000 draws)."""
    from scipy import stats
    rng = np.random.default_rng(0)
    g = st.mde_hedges_g(0.05, n1=3, n2=3, sd1=1.0, sd2=1.0)
    a = rng.normal(g, 1.0, (40000, 3))
    b = rng.normal(0.0, 1.0, (40000, 3))
    rate = float(np.mean(stats.ttest_ind(a, b, axis=1, equal_var=False).pvalue < 0.05))
    assert 0.60 < rate < 0.78, rate
