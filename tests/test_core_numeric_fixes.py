"""Regression tests for numeric fixes in core: dense relabel after the area
filter, non-finite Costes thresholds, dtype-aware saturation cutoff, and
lowest-value abundance binning."""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pandas as pd

from fishsuite.config.schema import FishsuiteConfig
from fishsuite.core import metrics as _metrics
from fishsuite.core import segmentation as _seg
from fishsuite.core.qc import compute_qc_flags
from fishsuite.core.singlecell import compute_singlecell


def test_area_filter_relabels_densely(monkeypatch):
    labels = np.zeros((40, 40), dtype=np.int32)
    labels[1:11, 1:11] = 1    # 100 px, kept
    labels[15:17, 15:17] = 2  # 4 px, dropped
    labels[25:35, 25:35] = 3  # 100 px, kept

    import fishsuite.core._vendor.segmentation.segment_image as _si
    monkeypatch.setattr(_si, "run_backend", lambda backend, img, **kw: labels.copy())

    stats, diag = {}, {}
    out = _seg.segment_nuclei(np.zeros((40, 40), dtype=np.float32), backend="stardist",
                              params={"min_area": 50}, stats=stats, diagnostics=diag)
    assert sorted(np.unique(out[out > 0]).tolist()) == [1, 2]
    assert out.max() == 2
    assert stats["n_segmented"] == 3 and stats["n_area_excluded"] == 1
    # The QC copy keeps IDs consistent with the filtered labels.
    raw = diag["labels_before_area"]
    kept = np.isin(raw, np.unique(out[out > 0]))
    assert np.array_equal(np.where(kept, raw, 0), out)


def test_backend_min_area_never_exceeds_user_min_area(monkeypatch):
    seen = {}

    def fake(backend, img, **kw):
        seen.update(kw)
        return np.zeros(img.shape, dtype=np.int32)

    import fishsuite.core._vendor.segmentation.segment_image as _si
    monkeypatch.setattr(_si, "run_backend", fake)
    _seg.segment_nuclei(np.zeros((8, 8), dtype=np.float32), backend="stardist",
                        params={"min_area": 100})
    assert seen["min_area"] <= 100


def test_costes_nonfinite_threshold_returns_nan_metrics():
    rng = np.random.default_rng(0)
    r = rng.random(15)  # < 20 pixels -> Costes cannot resolve (inf)
    a = rng.random(15)
    out = _metrics.compute_coloc_metrics(r, a, thr_mode="costes")
    expected = _metrics._empty_metrics(15)
    assert out.keys() == expected.keys()
    for k, v in expected.items():
        if isinstance(v, float) and math.isnan(v):
            assert math.isnan(out[k]), k
        else:
            assert out[k] == v, k


def test_saturation_cutoff_uses_plane_dtype():
    sat8 = np.full((16, 16), 255, dtype=np.uint8)
    res = SimpleNamespace(per_image={"n_nuclei": 10}, qc={"dapi_2d": sat8},
                          spots=pd.DataFrame(), nuclei=pd.DataFrame())
    out = compute_qc_flags(res, FishsuiteConfig())
    assert out["qc_frac_saturated_dapi"] == 1.0
    assert "saturated_dapi" in out["qc_flags"]


def test_singlecell_bins_include_lowest_value():
    # Non-integer abundance -> quantile bins whose first edge is the minimum.
    abund = np.linspace(0.5, 10.5, 40)
    nm = pd.DataFrame({
        "condition": ["ctrl_1"] * 20 + ["kd_1"] * 20,
        "rna1_spot_count": abund,
        "metric_x": np.arange(40, dtype=float),
    })
    res = compute_singlecell(nm, abundance_col="rna1_spot_count",
                             metrics=["metric_x"], exclude_secondary=False)
    binned = res["metric_x"]["binned"]
    assert sum(row["pool_n"] for row in binned) == 40
