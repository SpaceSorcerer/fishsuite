"""Per-spot Gaussian size fit: recovery, footprint units, saturation of the old estimator."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fishsuite.core.spot_size import (
    SIZE_FIT_COLUMNS,
    fit_spot_sizes,
    footprint_size_columns,
    size_rollup,
)

SEED = 0
VOXEL_UM = 0.065


def _synthetic(sigma_px, amp=2000.0, bg=300.0, pitch=20, grid=10, noise=True):
    rng = np.random.default_rng(SEED)
    n = pitch * grid + 2 * pitch
    img = np.full((n, n), float(bg))
    ys, xs = [], []
    for i in range(grid):
        for j in range(grid):
            cy, cx = pitch + i * pitch, pitch + j * pitch
            h = int(max(6, np.ceil(4 * sigma_px)))
            yy, xx = np.mgrid[cy - h:cy + h + 1, cx - h:cx + h + 1]
            img[cy - h:cy + h + 1, cx - h:cx + h + 1] += amp * np.exp(
                -(((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma_px ** 2)))
            ys.append(cy)
            xs.append(cx)
    if noise:
        img = rng.poisson(img).astype(np.float64)
    return img, pd.DataFrame({"y_px": ys, "x_px": xs})


@pytest.mark.parametrize("sigma_true", [0.9, 1.2, 1.6, 2.0])
def test_recovers_known_sigma_within_5pct(sigma_true):
    # Window must hold the spot: +/-3 sigma. 7 px is enough to sigma ~1.2;
    # wider sigmas get a window scaled to them, which is the documented knob.
    win = int(2 * np.ceil(3 * sigma_true) + 1)
    win = max(7, win)
    img, spots = _synthetic(sigma_true)
    res = fit_spot_sizes(img, spots, VOXEL_UM, window_px=win)
    ok = res[res["size_fit_ok"] == 1]
    assert len(ok) >= 0.95 * len(spots), f"only {len(ok)}/{len(spots)} fits ok"
    med = float(ok["size_fit_sigma_px"].median())
    assert abs(med - sigma_true) / sigma_true < 0.05, (
        f"sigma_true={sigma_true} recovered {med:.4f}")


def test_fwhm_um_is_sigma_times_constant_times_voxel():
    img, spots = _synthetic(1.2)
    res = fit_spot_sizes(img, spots, VOXEL_UM, window_px=9)
    k = 2.0 * np.sqrt(2.0 * np.log(2.0))
    ok = res[res["size_fit_ok"] == 1]
    assert np.allclose(ok["size_fit_fwhm_px"], k * ok["size_fit_sigma_px"])
    assert np.allclose(ok["size_fit_fwhm_um"], ok["size_fit_fwhm_px"] * VOXEL_UM)


def test_matches_scipy_curve_fit_on_a_subset():
    """The batched Levenberg-Marquardt must agree with a reference optimiser."""
    from scipy.optimize import curve_fit

    img, spots = _synthetic(1.4)
    win, half = 9, 4
    res = fit_spot_sizes(img, spots, VOXEL_UM, window_px=win)

    off = np.arange(-half, half + 1, dtype=float)
    yy, xx = np.meshgrid(off, off, indexing="ij")

    def model(coords, A, x0, y0, s, B):
        y_, x_ = coords
        return (A * np.exp(-(((x_ - x0) ** 2 + (y_ - y0) ** 2) / (2 * s ** 2)))
                + B).ravel()

    diffs = []
    for i in range(0, len(spots), 17):
        cy, cx = int(spots.y_px[i]), int(spots.x_px[i])
        crop = img[cy - half:cy + half + 1, cx - half:cx + half + 1].astype(float)
        p0 = [crop.max() - np.median(crop), 0.0, 0.0, 1.2, np.median(crop)]
        popt, _ = curve_fit(model, (yy, xx), crop.ravel(), p0=p0, maxfev=20000)
        diffs.append(abs(abs(popt[3]) - res.loc[i, "size_fit_sigma_px"]))
    assert max(diffs) < 0.02, f"max |sigma_batched - sigma_scipy| = {max(diffs):.4f}"


def test_edge_clipped_spots_are_not_ok_and_do_not_raise():
    img, _ = _synthetic(1.2)
    H, W = img.shape
    spots = pd.DataFrame({"y_px": [0, 1, H - 1, 20], "x_px": [0, W - 1, 3, 20]})
    res = fit_spot_sizes(img, spots, VOXEL_UM, window_px=7)
    assert list(res.columns) == SIZE_FIT_COLUMNS
    assert res.loc[0, "size_fit_ok"] == 0
    assert np.isnan(res.loc[0, "size_fit_sigma_px"])


def test_empty_and_bad_inputs():
    empty = fit_spot_sizes(np.zeros((10, 10)), pd.DataFrame({"y_px": [], "x_px": []}),
                           VOXEL_UM)
    assert len(empty) == 0 and list(empty.columns) == SIZE_FIT_COLUMNS
    bad = fit_spot_sizes(np.zeros((4, 4, 4)), pd.DataFrame({"y_px": [1], "x_px": [1]}),
                         VOXEL_UM)
    assert len(bad) == 1 and bad.loc[0, "size_fit_ok"] == 0


def test_footprint_area_um2_matches_stored_pixel_column():
    area_px = np.array([1.0, 12.0, 27.0, 208.0, np.nan])
    got = footprint_size_columns(area_px, VOXEL_UM)
    expect = area_px * VOXEL_UM ** 2
    assert np.allclose(got["footprint_area_um2"][:4], expect[:4])
    assert np.isnan(got["footprint_area_um2"][4])
    # equivalent diameter is the diameter of the disk of the same area
    d = got["footprint_equiv_diameter_um"].to_numpy()
    assert np.allclose(np.pi * (d[:4] / 2) ** 2, expect[:4])


def test_rollup_uses_only_ok_nuclear_spots():
    df = pd.DataFrame({
        "in_nucleus": [1, 1, 1, 0],
        "size_fit_ok": [1, 1, 0, 1],
        "size_fit_fwhm_um": [0.20, 0.30, 99.0, 5.0],
        "footprint_area_um2": [0.1, 0.3, 0.2, 9.0],
    })
    r = size_rollup(df, prefix="rna1")
    assert r["rna1_n_size_fit_ok"] == 2
    assert r["rna1_median_size_fit_fwhm_um"] == pytest.approx(0.25)
    assert r["rna1_median_footprint_area_um2"] == pytest.approx(0.2)


def test_adaptive_window_recovers_a_spot_wider_than_the_first_window():
    """sigma=5 exceeds half of a 9 px window, so a fixed window cannot measure it."""
    sigma_true = 5.0
    img, spots = _synthetic(sigma_true, pitch=54, grid=6)
    fixed = fit_spot_sizes(img, spots, VOXEL_UM, window_px=9, adaptive=False)
    grown = fit_spot_sizes(img, spots, VOXEL_UM, window_px=9, adaptive=True,
                           max_window_px=31)
    assert (fixed["size_fit_ok"] == 1).sum() == 0
    assert set(fixed["size_fit_flag"]) == {"wider_than_window"}
    assert (grown["size_fit_ok"] == 1).mean() > 0.95, "escalation failed to recover"
    assert grown["size_fit_window_used_px"].max() > 9
    med = float(grown[grown.size_fit_ok == 1]["size_fit_sigma_px"].median())
    assert abs(med - sigma_true) / sigma_true < 0.05, med


def test_escalation_never_replaces_a_good_fit_with_a_worse_one():
    img, spots = _synthetic(1.2, pitch=20)
    narrow = fit_spot_sizes(img, spots, VOXEL_UM, window_px=9, adaptive=False)
    wide = fit_spot_sizes(img, spots, VOXEL_UM, window_px=9, adaptive=True,
                          max_window_px=19)
    # Nothing needed escalation, so the two must be identical.
    assert (wide["size_fit_window_used_px"] == 9).all()
    assert np.allclose(narrow["size_fit_sigma_px"], wide["size_fit_sigma_px"],
                       equal_nan=True)


def test_flags_name_the_rejection_reason():
    rng = np.random.default_rng(SEED)
    # Flat noise: no object, so the Gaussian model explains nothing.
    flat = rng.poisson(np.full((64, 64), 300.0)).astype(float)
    spots = pd.DataFrame({"y_px": [20, 30, 40], "x_px": [20, 30, 40]})
    res = fit_spot_sizes(flat, spots, VOXEL_UM)
    assert set(res["size_fit_flag"]) <= {"low_r2", "center_drift",
                                         "wider_than_window", "bad_amplitude"}
    assert (res["size_fit_ok"] == 0).all()

    # An isolated spot 2 px from the supplied centre fits well but off-centre.
    img = np.full((80, 80), 300.0)
    yy, xx = np.mgrid[0:80, 0:80]
    img += 5000.0 * np.exp(-(((yy - 40) ** 2 + (xx - 42) ** 2) / (2 * 1.3 ** 2)))
    img = rng.poisson(img).astype(float)
    off = fit_spot_sizes(img, pd.DataFrame({"y_px": [40], "x_px": [40]}), VOXEL_UM,
                         adaptive=False)
    assert off.loc[0, "size_fit_flag"] == "center_drift"
    assert off.loc[0, "size_fit_ok"] == 0
    assert off.loc[0, "size_fit_r2"] > 0.9      # the model fit, the centre moved


def test_center_is_clamped_inside_the_window():
    """A runaway centre must be bounded, not reported as a coordinate off-crop."""
    rng = np.random.default_rng(SEED)
    img = np.full((64, 64), 300.0)
    yy, xx = np.mgrid[0:64, 0:64]
    img += 8000.0 * np.exp(-(((yy - 32) ** 2 + (xx - 44) ** 2) / (2 * 1.5 ** 2)))
    img = rng.poisson(img).astype(float)
    res = fit_spot_sizes(img, pd.DataFrame({"y_px": [32], "x_px": [32]}),
                         VOXEL_UM, window_px=9, adaptive=False)
    assert res.loc[0, "size_fit_ok"] == 0
    assert np.isfinite(res.loc[0, "size_fit_sigma_px"])


def test_schema_default_window_matches_the_fit_default():
    """A drift here silently backfills old runs on a different window."""
    import inspect

    from fishsuite.config.schema import FociCfg

    schema_default = FociCfg.model_fields["size_fit_window_px"].default
    fn_default = inspect.signature(fit_spot_sizes).parameters["window_px"].default
    assert schema_default == fn_default, (schema_default, fn_default)
    assert schema_default % 2 == 1


def test_backfill_uses_the_schema_default_for_a_run_without_the_field():
    """A run finished before this feature must still be fitted on the current default."""
    import inspect

    from fishsuite.core import sizefit_backfill

    src = inspect.getsource(sizefit_backfill.sizefit_run)
    assert "model_fields[\"size_fit_window_px\"].default" in src, (
        "backfill must read the schema default, not a duplicated literal")


def test_moment_estimator_saturates_while_the_fit_does_not():
    """Documents the defect this module exists to fix (rna_rna.py:224-282)."""
    from fishsuite.core.modes.rna_rna import _measure_spot_diameter_um

    moment, fitted, truth = [], [], []
    for s in (0.8, 1.6, 3.0):
        img, spots = _synthetic(s, pitch=28)
        moment.append(float(np.median(
            _measure_spot_diameter_um(img, spots, VOXEL_UM,
                                      fallback_diam_um=0.3))) / VOXEL_UM)
        res = fit_spot_sizes(img, spots, VOXEL_UM, window_px=int(2 * np.ceil(3 * s) + 1))
        fitted.append(float(res[res.size_fit_ok == 1]["size_fit_fwhm_px"].median()))
        truth.append(2.355 * s)

    true_span = truth[-1] / truth[0]
    assert moment[-1] / moment[0] < 0.5 * true_span, (
        "old estimator unexpectedly tracked size", moment)
    assert abs(fitted[-1] / fitted[0] - true_span) / true_span < 0.05, (
        "fit failed to track size", fitted, truth)
