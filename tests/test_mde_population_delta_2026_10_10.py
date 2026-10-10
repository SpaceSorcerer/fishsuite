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
