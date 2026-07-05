"""RNA1 over-detection QC-guard tests (Brian, 2026-07-05).

The guard is ADVISORY: it flags images whose RNA1 spots-per-nucleus is
implausibly high (the symptom of an out-of-focus dim RNA channel collapsing the
BigFISH auto-threshold into thousands of noise "spots") without changing
detection or dropping any image.

Two layers:
  * per-image ABSOLUTE cap in ``compute_qc_flags`` -> ``qc_overdetect_rna1``,
  * run-level ROBUST outlier in ``flag_overdetect_outliers`` ->
    ``qc_overdetect_rna1_run_outlier``.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from fishsuite.config.schema import FishsuiteConfig
from fishsuite.core.qc import compute_qc_flags, flag_overdetect_outliers


def _cfg(**qc_over):
    cfg = FishsuiteConfig()
    for k, v in qc_over.items():
        setattr(cfg.qc, k, v)
    return cfg


def _res(mean_rna1_spn, n_nuclei=10, n_spots=200):
    plane = (np.random.default_rng(0).integers(8000, 12000, size=(16, 16))).astype(np.uint16)
    return SimpleNamespace(
        per_image={"n_nuclei": n_nuclei, "mean_spots_per_nucleus_rna1": mean_rna1_spn},
        qc={"dapi_2d": plane, "rna_2d": plane},
        spots=pd.DataFrame({"x": range(n_spots)}),
        nuclei=pd.DataFrame({"label": range(n_nuclei)}),
    )


# ─── per-image absolute cap ─────────────────────────────────────────────────
def test_overdetect_flags_high_count():
    out = compute_qc_flags(_res(1500.0), _cfg())          # ~well11 MIAT-KD bug
    assert out["qc_rna1_spots_per_nucleus"] == 1500.0
    assert out["qc_overdetect_rna1"] is True
    assert "overdetect_rna1" in out["qc_flags"]
    assert out["qc_pass"] is False


def test_normal_count_not_flagged():
    out = compute_qc_flags(_res(20.0), _cfg())
    assert out["qc_overdetect_rna1"] is False
    assert "overdetect_rna1" not in out["qc_flags"]
    # A clean image (no other issues) still passes.
    assert out["qc_pass"] is True


def test_absolute_cap_disabled():
    out = compute_qc_flags(_res(1500.0), _cfg(qc_overdetect_rna1_max_per_nucleus=0.0))
    assert out["qc_overdetect_rna1"] is False


def test_falls_back_to_generic_mean_then_ratio():
    # No RNA1-specific key -> use n_spots/n_nuclei = 900/3 = 300 (== cap, not >).
    res = SimpleNamespace(
        per_image={"n_nuclei": 3},
        qc={}, spots=pd.DataFrame({"x": range(900)}), nuclei=pd.DataFrame({"l": range(3)}),
    )
    out = compute_qc_flags(res, _cfg(qc_overdetect_rna1_max_per_nucleus=200.0))
    assert out["qc_rna1_spots_per_nucleus"] == 300.0
    assert out["qc_overdetect_rna1"] is True


def test_zero_nuclei_is_nan_not_flagged():
    res = SimpleNamespace(
        per_image={"n_nuclei": 0},
        qc={}, spots=pd.DataFrame(), nuclei=pd.DataFrame(),
    )
    out = compute_qc_flags(res, _cfg())
    assert not np.isfinite(out["qc_rna1_spots_per_nucleus"])
    assert out["qc_overdetect_rna1"] is False


# ─── run-level robust outlier ───────────────────────────────────────────────
def test_run_outlier_flags_single_blowup():
    cfg = _cfg()
    # A tight cluster near ~40/nucleus plus one blown-up field at 900.
    rows = [{"qc_rna1_spots_per_nucleus": v, "qc_flags": "", "qc_pass": True}
            for v in [38.0, 41.0, 39.0, 42.0, 40.0, 37.0]]
    rows.append({"qc_rna1_spots_per_nucleus": 900.0, "qc_flags": "", "qc_pass": True})

    n = flag_overdetect_outliers(rows, cfg)
    assert n == 1
    assert rows[-1]["qc_overdetect_rna1_run_outlier"] is True
    assert "overdetect_rna1_outlier" in rows[-1]["qc_flags"]
    assert rows[-1]["qc_pass"] is False
    # The in-cluster images are untouched (column present, False).
    assert all(r["qc_overdetect_rna1_run_outlier"] is False for r in rows[:-1])
    assert all(r["qc_pass"] is True for r in rows[:-1])


def test_run_outlier_uniform_none_flagged():
    cfg = _cfg()
    rows = [{"qc_rna1_spots_per_nucleus": v, "qc_flags": "", "qc_pass": True}
            for v in [40.0, 41.0, 39.0, 42.0, 38.0]]
    assert flag_overdetect_outliers(rows, cfg) == 0
    assert all(r["qc_overdetect_rna1_run_outlier"] is False for r in rows)


def test_run_outlier_respects_small_signal_floor():
    """A 2x-median bump that is still below the small-signal floor never fires."""
    cfg = _cfg(qc_overdetect_min_per_nucleus_for_outlier=50.0)
    rows = [{"qc_rna1_spots_per_nucleus": v, "qc_flags": "", "qc_pass": True}
            for v in [2.0, 2.0, 2.0, 2.0, 2.0]]
    rows.append({"qc_rna1_spots_per_nucleus": 20.0, "qc_flags": "", "qc_pass": True})
    # 20 is a huge robust outlier vs a ~2 cluster, but < 50 floor -> not flagged.
    assert flag_overdetect_outliers(rows, cfg) == 0


def test_run_outlier_disabled_by_k():
    cfg = _cfg(qc_overdetect_robust_mad_k=0.0)
    rows = [{"qc_rna1_spots_per_nucleus": v, "qc_flags": "", "qc_pass": True}
            for v in [40.0, 41.0, 39.0, 900.0]]
    assert flag_overdetect_outliers(rows, cfg) == 0


def test_run_outlier_never_raises_on_garbage():
    assert flag_overdetect_outliers([], _cfg()) == 0
    assert flag_overdetect_outliers([{"nope": 1}, "junk", 5], _cfg()) == 0
