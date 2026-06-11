"""Isolated CPU unit test for the intensity-based coloc sampler.

Covers ``rna_rna._sample_partner_local_intensity`` — the per-spot RAW
partner-channel disk-mean used by the ``foci.compute_partner_intensity``
feature. Verifies:

  (a) correctness vs an independent brute-force disk-mean (a handful of spots),
  (b) edge spots (corners / partially + fully off-frame) do not crash and
      follow the documented semantics (partial disk -> in-bounds mean,
      fully off-image -> NaN),
  (c) it completes well under 2 s for ~4000 spots on a 1152x1152 frame —
      a regression guard proving the per-image pass cannot hang on this step.

These run single-process on synthetic numpy arrays — NO pipeline, NO GPU.
"""
import time

import numpy as np
import pandas as pd
import pytest

from fishsuite.core.modes.rna_rna import _sample_partner_local_intensity


def _brute_disk_mean(img, cy, cx, radius_px):
    """Independent reference: raw mean of ``img`` over an integer disk of
    radius ``radius_px`` centred at (round(cy), round(cx)), clipped to the
    image. Returns NaN if the disk is entirely off-image."""
    H, W = img.shape
    r = int(max(1, round(float(radius_px))))
    cyi, cxi = int(round(cy)), int(round(cx))
    vals = []
    for ddy in range(-r, r + 1):
        for ddx in range(-r, r + 1):
            if ddy * ddy + ddx * ddx <= radius_px ** 2:
                yy, xx = cyi + ddy, cxi + ddx
                if 0 <= yy < H and 0 <= xx < W:
                    vals.append(float(img[yy, xx]))
    return float(np.mean(vals)) if vals else float("nan")


@pytest.fixture(scope="module")
def synthetic_image():
    rng = np.random.default_rng(20260529)
    return (rng.random((1152, 1152), dtype=np.float64) * 4096.0).astype(np.float32)


def test_disk_mean_matches_bruteforce(synthetic_image):
    """A few interior spots: vectorized sampler == brute-force disk mean."""
    img = synthetic_image
    radius_px = 3.0
    coords = [(100.2, 200.7), (500.0, 500.0), (812.9, 333.1), (1000.4, 47.6)]
    df = pd.DataFrame(
        {"y_px": [c[0] for c in coords], "x_px": [c[1] for c in coords]}
    )
    got = _sample_partner_local_intensity(img, df, radius_px)
    for i, (cy, cx) in enumerate(coords):
        expect = _brute_disk_mean(img, cy, cx, radius_px)
        assert got[i] == pytest.approx(expect, rel=1e-9, abs=1e-9), (
            f"spot {i} ({cy},{cx}): got {got[i]} expected {expect}"
        )


def test_edge_and_offimage_spots(synthetic_image):
    """Corner/edge spots return the in-bounds (clipped) disk mean; a spot
    whose entire disk is off-image returns NaN; none of them crash."""
    img = synthetic_image
    H, W = img.shape
    radius_px = 4.0
    coords = [
        (0.0, 0.0),            # top-left corner (partial disk)
        (H - 1, W - 1),        # bottom-right corner (partial disk)
        (0.0, W // 2),         # top edge
        (H // 2, 0.0),         # left edge
        (-50.0, -50.0),        # fully off-image -> NaN
        (H + 100.0, W + 100.0),  # fully off-image -> NaN
    ]
    df = pd.DataFrame(
        {"y_px": [c[0] for c in coords], "x_px": [c[1] for c in coords]}
    )
    got = _sample_partner_local_intensity(img, df, radius_px)
    # First four: finite and equal to the clipped brute-force mean.
    for i in range(4):
        expect = _brute_disk_mean(img, coords[i][0], coords[i][1], radius_px)
        assert np.isfinite(got[i])
        assert got[i] == pytest.approx(expect, rel=1e-9, abs=1e-9)
    # Last two: disk entirely outside the frame -> NaN.
    assert np.isnan(got[4])
    assert np.isnan(got[5])


def test_empty_and_bad_radius(synthetic_image):
    """Empty table -> empty array; non-finite/zero radius is coerced (no crash)."""
    img = synthetic_image
    empty = _sample_partner_local_intensity(img, pd.DataFrame({"y_px": [], "x_px": []}), 3.0)
    assert empty.shape == (0,)
    df = pd.DataFrame({"y_px": [500.0], "x_px": [500.0]})
    for bad in (0.0, -1.0, np.nan, np.inf):
        v = _sample_partner_local_intensity(img, df, bad)
        assert v.shape == (1,) and np.isfinite(v[0])  # coerced radius -> sampled


def test_4000_spots_under_2s(synthetic_image):
    """Timing regression guard: ~4000 spots (incl. edge spots) must finish in
    well under 2 s. The pre-fix naive full-frame-mask path took minutes."""
    img = synthetic_image
    H, W = img.shape
    rng = np.random.default_rng(1)
    n = 4000
    ys = rng.integers(0, H, n).astype(float)
    xs = rng.integers(0, W, n).astype(float)
    # Sprinkle ~200 spots right on/near the borders.
    ys[:50] = 0.0
    ys[50:100] = H - 1
    xs[100:150] = 0.0
    xs[150:200] = W - 1
    df = pd.DataFrame({"y_px": ys, "x_px": xs})
    # radius ~1 px is the realistic BIN1/XRN2 case (130 nm spot / 130 nm px).
    radius_px = 1.0
    t0 = time.perf_counter()
    out = _sample_partner_local_intensity(img, df, radius_px)
    elapsed = time.perf_counter() - t0
    assert out.shape == (n,)
    assert np.isfinite(out).all()  # all sampled spots are in-frame
    assert elapsed < 2.0, f"sampling 4000 spots took {elapsed:.3f}s (expected < 2s)"
