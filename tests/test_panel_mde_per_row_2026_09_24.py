"""coloc_standard_panel: the MDE on each contrasts row uses THAT row's finite well counts.

2026-09-24: a single design-level MDE was stamped on every row, so an endpoint
with no finite well means (e.g. the partner-anchored columns) still showed the
2 v 2 MDE while its own mde_note said TOO_FEW_WELLS.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PANEL_PATH = Path(__file__).resolve().parents[1] / "scripts" / "coloc_standard_panel.py"


@pytest.fixture
def panel():
    spec = importlib.util.spec_from_file_location("_panel_mde_2026_09_24", PANEL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _two_by_two(panel):
    endpoints = {c: np.nan for c, _ in panel.ALL_ENDPOINTS}
    rows = []
    for line, cond, wells, value in (("OE", "g2_Dox", ("d1", "d2"), (0.1, 0.2)),
                                     ("WT", "g2_NoDox", ("n1", "n2"), (0.3, 0.5))):
        for well, v in zip(wells, value):
            rows.append(dict(line=line, condition=cond, well_id=well, secondary_only=False,
                             image=well + ".vsi", nucleus_id=1,
                             **{**endpoints, "frac_called_coloc_runthr": v, "pearson_r_csp": v}))
    return pd.DataFrame(rows)


def test_rows_with_fewer_finite_wells_get_their_own_mde(panel):
    per_nucleus = _two_by_two(panel)
    _fov, well = panel.rollup(per_nucleus)
    panel.ARMS[:] = ["WT", "OE"]
    design = panel.mde_hedges_g(n1=2, n2=2)
    contrasts = panel.contrasts_table(well, per_nucleus, design,
                                      "frac_called_coloc_runthr", "frac_called_coloc").set_index("endpoint")
    col = "mde_hedges_g_alpha_0p05_power_0p80"
    assert contrasts.loc["frac_called_coloc_runthr", col] == pytest.approx(5.65, abs=0.01)
    assert contrasts.loc["pearson_r_csp", col] == pytest.approx(5.65, abs=0.01)
    empty = contrasts.loc["frac_called_coloc_partner"]
    assert empty.n_wells_test == 0 and empty.mde_note == "TOO_FEW_WELLS"
    assert np.isnan(empty[col])
    assert (contrasts[col] > 1e-6).where(contrasts[col].notna(), True).all()


def test_overlay_picks_carry_condition_and_well_for_the_field_label(panel, tmp_path):
    """c80f511 regrouped overlays on well_id and dropped `condition`, so every
    real panel run died in _save_field_overlay with KeyError: 'condition'."""
    per_nucleus = _two_by_two(panel)
    per_nucleus["n_rna1_nuclear_puncta"] = 3
    per_nucleus["n_called_coloc"] = 1
    panel.ARMS[:] = ["WT", "OE"]
    picks = panel.overlay_picks(per_nucleus, {"primary_obs": "frac_called_coloc_runthr"}, 1)
    assert [(p["line"], p["well_id"], p["condition"]) for p in picks] == [
        ("WT", "n1", "g2_NoDox"), ("WT", "n2", "g2_NoDox"),
        ("OE", "d1", "g2_Dox"), ("OE", "d2", "g2_Dox")]
    rgb = np.zeros((60, 80, 3))
    panel._save_field_overlay(tmp_path / "f.png", rgb, [], np.array([10]), np.array([10]),
                              np.array([True]), 130.0, {}, picks[0])
    assert (tmp_path / "f.png").stat().st_size > 0
