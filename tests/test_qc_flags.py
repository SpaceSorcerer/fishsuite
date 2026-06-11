"""Tests for the additive per-image QC-flag helper (2026-06-10)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from fishsuite.config.schema import FishsuiteConfig
from fishsuite.core.qc import compute_qc_flags


def _cfg():
    return FishsuiteConfig()


def test_overexposed_image_flags_saturation_and_fails():
    cfg = _cfg()
    # Fully saturated uint16 dapi + rna planes -> frac_saturated == 1.0.
    sat = np.full((16, 16), 65535, dtype=np.uint16)
    res = SimpleNamespace(
        per_image={"n_nuclei": 0},
        qc={"dapi_2d": sat, "rna_2d": sat},
        spots=pd.DataFrame(),          # zero spots
        nuclei=pd.DataFrame(),         # zero nuclei
    )
    out = compute_qc_flags(res, cfg)

    assert out["qc_frac_saturated_dapi"] > 0.0
    assert out["qc_frac_saturated_rna"] > 0.0
    # Saturation, low nuclei (0 < 5), and zero spots should all fire.
    flags = out["qc_flags"]
    assert "saturated_rna" in flags
    assert "saturated_dapi" in flags
    assert "low_nuclei" in flags
    assert "zero_spot" in flags
    assert out["qc_pass"] is False
    # rna2 / antibody planes absent -> their columns are NOT emitted.
    assert "qc_frac_saturated_rna2" not in out
    assert "qc_frac_saturated_antibody" not in out


def test_clean_image_passes():
    cfg = _cfg()
    rng = np.random.default_rng(0)
    # Mid-gray planes with mild texture -> not saturated.
    dapi = (rng.integers(8000, 12000, size=(32, 32))).astype(np.uint16)
    rna = (rng.integers(8000, 12000, size=(32, 32))).astype(np.uint16)
    res = SimpleNamespace(
        per_image={"n_nuclei": 12},
        qc={"dapi_2d": dapi, "rna_2d": rna},
        spots=pd.DataFrame({"x": range(20), "y": range(20)}),  # some spots
        nuclei=pd.DataFrame({"label": range(12)}),
    )
    out = compute_qc_flags(res, cfg)

    assert out["qc_frac_saturated_dapi"] == 0.0
    assert out["qc_frac_saturated_rna"] == 0.0
    assert out["qc_n_nuclei"] == 12
    assert out["qc_low_nuclei"] is False
    assert out["qc_zero_spot"] is False
    assert out["qc_flags"] == ""
    assert out["qc_pass"] is True
    # Focus score is a finite float for a textured plane.
    assert np.isfinite(out["qc_focus_score"])


def test_rna_protein_antibody_role_emitted():
    cfg = _cfg()
    plane = np.full((8, 8), 30000, dtype=np.uint16)  # not saturated
    res = SimpleNamespace(
        per_image={"n_nuclei": 7},
        qc={"dapi_2d": plane, "rna_2d": plane, "antibody_2d": plane},
        spots=pd.DataFrame({"x": [1, 2, 3]}),
        nuclei=pd.DataFrame({"label": range(7)}),
    )
    out = compute_qc_flags(res, cfg)
    assert "qc_frac_saturated_antibody" in out
    assert out["qc_frac_saturated_antibody"] == 0.0
    assert out["qc_pass"] is True


def test_compute_qc_flags_never_raises_on_garbage():
    cfg = _cfg()
    # res missing everything useful -> still returns a dict, no exception.
    res = SimpleNamespace(per_image={}, qc={}, spots=pd.DataFrame(), nuclei=pd.DataFrame())
    out = compute_qc_flags(res, cfg)
    assert isinstance(out, dict)
    assert "qc_flags" in out and "qc_pass" in out
    # 0 nuclei + 0 spots -> flagged.
    assert out["qc_low_nuclei"] is True
    assert out["qc_zero_spot"] is True
