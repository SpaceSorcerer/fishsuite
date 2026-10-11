"""coloc_standard_panel and the report share one MDE implementation (2026-10-11)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from fishsuite.report import stats as rstats

PANEL_PATH = Path(__file__).resolve().parents[1] / "scripts" / "coloc_standard_panel.py"


@pytest.fixture(scope="module")
def panel():
    spec = importlib.util.spec_from_file_location("_panel_mde_2026_10_11", PANEL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("n1,n2,sd1,sd2", [(2, 2, None, None), (3, 3, .05, .2), (4, 3, .3, .1), (2, 5, 1., 1.)])
def test_panel_mde_equals_report_stats_mde(panel, n1, n2, sd1, sd2):
    assert panel.mde_hedges_g(n1=n1, n2=n2, sd1=sd1, sd2=sd2) == rstats.mde_hedges_g(
        0.05, 0.80, n1=n1, n2=n2, sd1=sd1, sd2=sd2)


def test_panel_has_no_private_power_routine(panel):
    assert not hasattr(panel, "_power_two_sided")


def test_panel_mde_nan_below_two_wells(panel):
    assert np.isnan(panel.mde_hedges_g(n1=1, n2=3, sd1=.1, sd2=.1))
