"""Scope-matched per-area endpoints, and the panel's object-fraction labels.

1. `rna1_spots_per_um2` divides TOTAL puncta by NUCLEAR area. That is a count
   normalised by nuclear size, not a concentration. `rna1_nuclear_spots_per_um2`
   and `protein_spots_per_um2` are the scope-matched companions.
2. `scripts/coloc_standard_panel.py` tested `col.startswith("frac_")`, but the
   six object-fraction columns are named `paired_frac_*`, so all six inherited
   the pixel fallback `role = pixel, descriptive`, `unit = ICQ -0.5..+0.5`.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import pytest

from fishsuite.report import aggregate as agg
from fishsuite.report import endpoints as ep

REPO = Path(__file__).resolve().parents[1]
PANEL = REPO / "scripts" / "coloc_standard_panel.py"
BY_NAME = {e.name: e for e in ep.ENDPOINTS}


# --------------------------------------------------------------- endpoints
def test_both_new_endpoints_are_declared():
    for n in ("rna1_nuclear_spots_per_um2", "protein_spots_per_um2"):
        assert n in BY_NAME, n


def test_protein_density_mirrors_its_parents_holm_treatment():
    child, parent = BY_NAME["protein_spots_per_um2"], BY_NAME["protein_spots_per_nucleus"]
    assert child.excluded_from_holm == parent.excluded_from_holm == \
        "proxy for absolute partner level"
    assert child.exploratory is parent.exploratory is True
    assert child.family == parent.family or child.family == "localization"


def test_new_endpoints_state_the_nuclear_scope_in_the_unit():
    for n in ("rna1_nuclear_spots_per_um2", "protein_spots_per_um2"):
        u = BY_NAME[n].unit
        assert "nuclear puncta" in u and "nuclear area" in u, (n, u)


def test_the_total_over_nuclear_endpoints_declare_the_mismatch():
    for n in ("rna1_spots_per_um2", "rna2_spots_per_um2"):
        e = BY_NAME[n]
        assert "TOTAL" in e.unit and "NUCLEAR" in e.unit, (n, e.unit)
        assert "SCOPE MISMATCH" in e.note, n


def _run(tmp_path, nuc_extra: dict) -> dict:
    run = tmp_path / "RUN"
    run.mkdir(parents=True, exist_ok=True)
    nuc = {"image": "a.vsi", "nucleus_id": 1, "n_spots_rna1": 10.0,
           "nuclear_spot_count": 4.0, "nucleus_area_px": 10000.0,
           "voxel_xy_um": 0.1}
    nuc.update(nuc_extra)
    pd.DataFrame([{"image": "a.vsi", "condition": "WT_1", "secondary_only": False}]
                 ).to_csv(run / "per_image_summary.csv", index=False)
    pd.DataFrame([nuc]).to_csv(run / "nuclei_metrics.csv", index=False)
    pd.DataFrame([{"image": "a.vsi", "channel": "rna1", "nucleus_id": 1,
                   "in_nucleus": True, "in_cytoplasm": False,
                   "x_px": 0.0, "y_px": 0.0, "z_slice": 0}]
                 ).to_csv(run / "spot_metrics.csv", index=False)
    (run / "run_config.json").write_text(json.dumps(
        {"config_resolved": {"channels": {"analysis_mode": "rna_protein"}}}),
        encoding="utf-8")
    return agg.load_run(run, {}, {})


def test_nuclear_density_is_computed_when_the_engine_did_not_emit_it(tmp_path):
    nuc = _run(tmp_path, {})["nuclei"].iloc[0]
    # 10000 px * 0.1^2 = 100 um2; 4 nuclear puncta -> 0.04
    assert nuc["nucleus_area_um2"] == pytest.approx(100.0)
    assert nuc["rna1_nuclear_spots_per_um2"] == pytest.approx(0.04)
    # and the total-over-nuclear column is unchanged at 10/100
    assert nuc["rna1_spots_per_um2"] == pytest.approx(0.10)


def test_the_engines_own_density_wins_over_recomputation(tmp_path):
    nuc = _run(tmp_path, {"nuclear_spot_density_per_um2": 0.123})["nuclei"].iloc[0]
    assert nuc["rna1_nuclear_spots_per_um2"] == pytest.approx(0.123)


def test_protein_density_from_the_engine_column(tmp_path):
    nuc = _run(tmp_path, {"nuclear_spot_density_per_um2_protein": 0.77,
                          "nuclear_spot_count_protein": 3.0})["nuclei"].iloc[0]
    assert nuc["protein_spots_per_um2"] == pytest.approx(0.77)


def test_protein_density_computed_from_the_count_when_absent(tmp_path):
    nuc = _run(tmp_path, {"nuclear_spot_count_protein": 3.0})["nuclei"].iloc[0]
    assert nuc["protein_spots_per_um2"] == pytest.approx(0.03)


def test_protein_density_absent_when_the_run_has_no_partner_spots(tmp_path):
    data = _run(tmp_path, {})
    _, absent = ep.resolve(data["nuclei"], data["per_image"])
    assert "protein_spots_per_um2" in absent
    assert "rna1_nuclear_spots_per_um2" not in absent


# ------------------------------------------------------------ panel labels
def test_panel_declares_both_object_fraction_prefixes():
    src = io.open(PANEL, encoding="utf-8").read()
    assert 'OBJECT_FRACTION_PREFIXES = ("frac_", "paired_frac_")' in src
    assert 'col.startswith("frac_")' not in src, "the frac_-only test survives"
    assert 'col.startswith(("frac_", "manders"))' not in src


def _prefixes():
    ns: dict = {}
    for line in io.open(PANEL, encoding="utf-8"):
        if line.startswith("OBJECT_FRACTION_PREFIXES"):
            exec(line, ns)
            return ns["OBJECT_FRACTION_PREFIXES"]
    raise AssertionError("constant not found")


@pytest.mark.parametrize("col", [
    "paired_frac_rna1_at_partner",
    "paired_frac_rna1_at_partner_shuffle",
    "paired_frac_rna1_at_partner_minus_shuffle",
    "paired_frac_partner_at_rna1",
    "paired_frac_partner_at_rna1_shuffle",
    "paired_frac_partner_at_rna1_minus_shuffle",
])
def test_the_six_paired_columns_are_object_fractions(col):
    pref = _prefixes()
    assert col.startswith(pref), col
    role = ("object, descriptive" if col.startswith(pref) else "pixel, descriptive")
    unit = "fraction 0-1" if col.startswith(pref + ("manders",)) else "ICQ -0.5..+0.5"
    assert role == "object, descriptive"
    assert unit == "fraction 0-1"


def test_pixel_columns_keep_the_pixel_labels():
    pref = _prefixes()
    for col in ("icq_runthr", "pearson_runthr"):
        assert not col.startswith(pref)
        unit = "fraction 0-1" if col.startswith(pref + ("manders",)) else (
            "r" if col.startswith("pearson") else "ICQ -0.5..+0.5")
        assert unit == ("r" if col.startswith("pearson") else "ICQ -0.5..+0.5")


def test_the_six_columns_are_the_ones_the_panel_emits():
    src = io.open(PANEL, encoding="utf-8").read()
    for col in ("paired_frac_rna1_at_partner", "paired_frac_partner_at_rna1"):
        assert f'("{col}", ' in src, col
