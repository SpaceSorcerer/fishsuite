"""Per-spot size measurement by 2-D Gaussian fit (CPU, vectorised).

WHY THIS EXISTS
---------------
``rna_rna._measure_spot_diameter_um`` estimates a spot's width from the second
central moment of a FIXED 9x9 crop (``crop_half=4``, rna_rna.py:228). That
estimator is bounded by the crop, not by the spot: a uniformly filled 9x9 window
has ``var_y = var_x = (9**2 - 1) / 12 = 6.667``, so ``sigma = sqrt(13.33 / 2) =
2.58`` px and ``FWHM = 6.08`` px is a hard ceiling, while ``max(var / 2, 0.25)``
(rna_rna.py:279) puts a 1.18 px floor under it. Between those limits the
background pedestal left after a 10th-percentile subtraction dominates the
``r**2``-weighted sum, so the returned width compresses towards a near-constant
value. Measured on synthetic Gaussians (tests/test_spot_size_fit.py): a 4.4x
range in true FWHM (1.88 -> 8.24 px) came out as a 1.5x range (2.99 -> 4.55 px).
``spot_fwhm_px`` (rna_rna.py:3347) and ``spot_area_px`` (:3348) are unit
conversions of that same saturated quantity, so all three columns inherit it.

This module fits an actual model instead:

    z(x, y) = A * exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2)) + B

on a small window around each spot, by batched Levenberg-Marquardt. All spots
advance together in numpy, so a 138k-spot run costs seconds rather than the
minutes a per-spot ``scipy.optimize.curve_fit`` loop would.

A spot whose fitted sigma exceeds half its window is not measured, it is
extrapolated, so those spots are automatically REFITTED on a window scaled to
their own size (up to ``max_window_px``). ``size_fit_flag`` names the reason a
fit was rejected instead of leaving an opaque zero.

The outputs are ADDITIVE. Nothing here replaces ``spot_diameter_um`` /
``spot_fwhm_px`` / ``spot_area_px``; those columns keep their existing values so
old runs stay comparable.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

SIZE_FIT_COLUMNS = [
    "size_fit_sigma_px",
    "size_fit_fwhm_px",
    "size_fit_fwhm_um",
    "size_fit_amplitude",
    "size_fit_background",
    "size_fit_r2",
    "size_fit_window_used_px",
    "size_fit_ok",
    "size_fit_flag",
]

FOOTPRINT_SIZE_COLUMNS = [
    "footprint_area_um2",
    "footprint_equiv_diameter_um",
]

FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))   # 2.3548


def _empty_frame(n: int) -> pd.DataFrame:
    out = pd.DataFrame(index=pd.RangeIndex(n))
    for c in SIZE_FIT_COLUMNS:
        if c == "size_fit_ok":
            out[c] = 0
        elif c == "size_fit_flag":
            out[c] = "not_fitted"
        elif c == "size_fit_window_used_px":
            out[c] = np.nan
        else:
            out[c] = np.nan
    out["size_fit_ok"] = out["size_fit_ok"].astype(int)
    return out


def _odd(w: int) -> int:
    w = int(w)
    if w < 3:
        w = 3
    return w if w % 2 else w + 1


def _fit_once(
    img: np.ndarray,
    cy: np.ndarray,
    cx: np.ndarray,
    window_px: int,
    sigma_init: np.ndarray,
    max_iter: int,
) -> Tuple[np.ndarray, ...]:
    """Batched Levenberg-Marquardt on one window size.

    ``cy``/``cx`` are integer centres already known to sit far enough from the
    edge for this window. Returns (A, x0, y0, sigma, B, r2).
    """
    half = window_px // 2
    off = np.arange(-half, half + 1, dtype=np.intp)
    gy = cy[:, None, None] + off[None, :, None]
    gx = cx[:, None, None] + off[None, None, :]
    crops = img[gy, gx].reshape(cy.size, window_px * window_px)

    fo = off.astype(np.float64)
    yy, xx = np.meshgrid(fo, fo, indexing="ij")
    ys = yy.ravel()[None, :]
    xs = xx.ravel()[None, :]

    m = cy.size
    B = np.median(crops, axis=1)
    A = crops.max(axis=1) - B
    A = np.where(A > 0, A, 1.0)
    x0 = np.zeros(m)
    y0 = np.zeros(m)
    sg = np.clip(sigma_init.astype(np.float64), 0.4, float(half))

    d = (xs - x0[:, None]) ** 2 + (ys - y0[:, None]) ** 2
    g = np.exp(-d / (2.0 * sg[:, None] ** 2))
    r = A[:, None] * g + B[:, None] - crops
    cost = np.einsum("ij,ij->i", r, r)

    lam = np.full(m, 1e-3)
    active = np.ones(m, dtype=bool)
    eye = np.eye(5)

    for _ in range(int(max_iter)):
        if not active.any():
            break
        a = np.flatnonzero(active)
        ra, ga, da = r[a], g[a], d[a]
        Aa, x0a, y0a, sga = A[a], x0[a], y0[a], sg[a]
        s2 = sga[:, None] ** 2

        J = np.empty((a.size, ga.shape[1], 5))
        J[:, :, 0] = ga                                              # dA
        J[:, :, 1] = Aa[:, None] * ga * (xs - x0a[:, None]) / s2     # dx0
        J[:, :, 2] = Aa[:, None] * ga * (ys - y0a[:, None]) / s2     # dy0
        J[:, :, 3] = Aa[:, None] * ga * da / (sga[:, None] ** 3)     # dsigma
        J[:, :, 4] = 1.0                                             # dB

        JtJ = np.einsum("ikp,ikq->ipq", J, J)
        Jtr = np.einsum("ikp,ik->ip", J, ra)
        diag = np.maximum(np.diagonal(JtJ, axis1=1, axis2=2), 1e-12)
        damped = JtJ + lam[a][:, None, None] * eye[None, :, :] * diag[:, None, :]
        try:
            step = np.linalg.solve(damped, -Jtr)
        except np.linalg.LinAlgError:
            step = np.zeros((a.size, 5))
        step = np.where(np.isfinite(step), step, 0.0)

        A_n = Aa + step[:, 0]
        # A centre outside the crop is not a fit, it is a runaway: clamp it in
        # so the caller sees a bounded ``center_drift`` rejection rather than a
        # meaningless coordinate (observed x0 down to -6 px in a half=4 window
        # on the dense antibody channel).
        x0_n = np.clip(x0a + step[:, 1], -float(half), float(half))
        y0_n = np.clip(y0a + step[:, 2], -float(half), float(half))
        sg_n = np.clip(np.abs(sga + step[:, 3]), 0.15, float(half) * 2.0)
        B_n = B[a] + step[:, 4]

        d_n = (xs - x0_n[:, None]) ** 2 + (ys - y0_n[:, None]) ** 2
        g_n = np.exp(-d_n / (2.0 * sg_n[:, None] ** 2))
        r_n = A_n[:, None] * g_n + B_n[:, None] - crops[a]
        cost_n = np.einsum("ij,ij->i", r_n, r_n)

        better = cost_n < cost[a]
        acc = a[better]
        if acc.size:
            bi = np.flatnonzero(better)
            A[acc], x0[acc], y0[acc], sg[acc], B[acc] = (
                A_n[bi], x0_n[bi], y0_n[bi], sg_n[bi], B_n[bi])
            r[acc], g[acc], d[acc] = r_n[bi], g_n[bi], d_n[bi]
            rel = (cost[acc] - cost_n[bi]) / np.maximum(cost[acc], 1e-12)
            cost[acc] = cost_n[bi]
            lam[acc] = np.maximum(lam[acc] * 0.3, 1e-9)
            active[acc[rel < 1e-6]] = False
        rej = a[~better]
        if rej.size:
            lam[rej] *= 10.0
            active[rej[lam[rej] > 1e8]] = False

    mean_c = crops.mean(axis=1)
    ss_tot = np.einsum("ij,ij->i", crops - mean_c[:, None], crops - mean_c[:, None])
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.where(ss_tot > 0, 1.0 - cost / ss_tot, np.nan)
    return A, x0, y0, sg, B, r2


def fit_spot_sizes(
    plane_2d: np.ndarray,
    spots_df: pd.DataFrame,
    voxel_xy_um: float,
    *,
    window_px: int = 9,
    max_window_px: int = 15,
    adaptive: bool = True,
    sigma_init_px: float = 1.5,
    max_iter: int = 40,
    min_r2: float = 0.5,
    max_center_shift_px: float = 1.5,
    min_sigma_px: float = 0.3,
) -> pd.DataFrame:
    """Fit a 2-D Gaussian + constant background around every spot.

    Parameters
    ----------
    plane_2d
        The 2-D plane the spots were DETECTED on (raw intensities).
    spots_df
        Must carry ``y_px`` and ``x_px``. Row order is preserved.
    voxel_xy_um
        Lateral pixel size in micrometres, used only for the ``_um`` column.
    window_px
        Odd side length of the first-pass square window. 9 -> +/-4 px, chosen
        from a window-convergence scan on real 65 nm/px data (see the module
        docstring of tests/test_spot_size_fit.py).
    max_window_px, adaptive
        A spot whose first-pass sigma exceeds half its window is refitted on a
        window of ``2 * ceil(3 * sigma) + 1``, capped at ``max_window_px``, so
        genuinely large foci are measured rather than clipped.

    Returns
    -------
    DataFrame with :data:`SIZE_FIT_COLUMNS`, one row per input spot, indexed
    ``0..n-1``. ``size_fit_ok`` is 1 only for fits that converged inside their
    own window; ``size_fit_flag`` names the reason otherwise:

    ``edge``
        the window did not fit inside the frame
    ``wider_than_window``
        sigma exceeded half the widest window tried; the number is an
        extrapolation, not a measurement
    ``low_r2``
        the Gaussian + constant model does not describe the crop, e.g. a
        diffuse channel with no isolated object
    ``center_drift``
        the fitted centre moved more than ``max_center_shift_px``, usually a
        brighter neighbour inside the window
    ``bad_amplitude``
        non-positive or non-finite amplitude

    Failed fits keep their fitted numbers (still informative) but must be
    filtered on ``size_fit_ok`` before any summary.
    """
    n = len(spots_df)
    if n == 0:
        return _empty_frame(0)
    if plane_2d is None or np.ndim(plane_2d) != 2:
        return _empty_frame(n)

    w0 = _odd(window_px)
    wmax = max(_odd(max_window_px), w0)
    H, W = plane_2d.shape
    img = np.asarray(plane_2d, dtype=np.float64)

    cy_f = pd.to_numeric(spots_df["y_px"], errors="coerce").to_numpy(dtype=float)
    cx_f = pd.to_numeric(spots_df["x_px"], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(cy_f) & np.isfinite(cx_f)
    cy = np.where(finite, np.rint(cy_f), 0.0).astype(np.intp)
    cx = np.where(finite, np.rint(cx_f), 0.0).astype(np.intp)

    out = _empty_frame(n)

    sig = np.full(n, np.nan)
    amp = np.full(n, np.nan)
    bkg = np.full(n, np.nan)
    r2v = np.full(n, np.nan)
    shift = np.full(n, np.nan)
    win_used = np.full(n, np.nan)

    def _inside(w: int) -> np.ndarray:
        h = w // 2
        return finite & (cy >= h) & (cy < H - h) & (cx >= h) & (cx < W - h)

    first = np.flatnonzero(_inside(w0))
    if first.size == 0:
        out["size_fit_flag"] = "edge"
        return out

    A, x0, y0, sg, B, r2 = _fit_once(img, cy[first], cx[first], w0,
                                     np.full(first.size, float(sigma_init_px)),
                                     max_iter)
    sig[first], amp[first], bkg[first], r2v[first] = sg, A, B, r2
    shift[first] = np.hypot(x0, y0)
    win_used[first] = w0

    # Escalate ONLY the spots the window truncated, in +4 px steps, and only
    # where the first fit is otherwise sound (a low-r2 crop has no size to
    # measure, so a wider window would just swallow neighbours). The wider fit
    # REPLACES the narrower one only when it is itself sound — escalation must
    # never degrade a spot that was already measured.
    w = w0
    while adaptive and w < wmax:
        h = w // 2
        need = np.flatnonzero(
            (win_used == w) & (sig > h) & (r2v >= min_r2) & (amp > 0)
            & np.isfinite(sig) & (shift <= max_center_shift_px))
        if need.size == 0:
            break
        w_next = min(_odd(w + 4), wmax)
        if w_next <= w:
            break
        ins = _inside(w_next)
        keep = need[ins[need]]
        if keep.size == 0:
            break
        A2, x2, y2, s2, B2, r22 = _fit_once(img, cy[keep], cx[keep], w_next,
                                            sig[keep], max_iter)
        sh2 = np.hypot(x2, y2)
        take = (np.isfinite(s2) & (A2 > 0) & (r22 >= min_r2)
                & (sh2 <= max_center_shift_px))
        acc = keep[take]
        if acc.size:
            ti = np.flatnonzero(take)
            sig[acc], amp[acc], bkg[acc], r2v[acc] = (
                s2[ti], A2[ti], B2[ti], r22[ti])
            shift[acc] = sh2[ti]
            win_used[acc] = w_next
        w = w_next

    half_used = np.floor(win_used / 2.0)
    ok = (
        np.isfinite(sig) & np.isfinite(amp) & np.isfinite(r2v)
        & (amp > 0)
        & (sig >= float(min_sigma_px)) & (sig <= half_used)
        & (shift <= float(max_center_shift_px))
        & (r2v >= float(min_r2))
    )

    flag = np.full(n, "edge", dtype=object)
    fitted = np.isfinite(win_used)
    flag[fitted & ~(amp > 0)] = "bad_amplitude"
    flag[fitted & (amp > 0) & (r2v < float(min_r2))] = "low_r2"
    flag[fitted & (amp > 0) & (r2v >= float(min_r2))
         & (shift > float(max_center_shift_px))] = "center_drift"
    flag[fitted & (amp > 0) & (r2v >= float(min_r2))
         & (shift <= float(max_center_shift_px))
         & ((sig > half_used) | (sig < float(min_sigma_px)))] = "wider_than_window"
    flag[ok] = "ok"

    fwhm_px = FWHM_PER_SIGMA * sig
    out["size_fit_sigma_px"] = sig
    out["size_fit_fwhm_px"] = fwhm_px
    out["size_fit_fwhm_um"] = fwhm_px * float(voxel_xy_um)
    out["size_fit_amplitude"] = amp
    out["size_fit_background"] = bkg
    out["size_fit_r2"] = r2v
    out["size_fit_window_used_px"] = win_used
    out["size_fit_ok"] = ok.astype(int)
    out["size_fit_flag"] = flag
    return out


def footprint_size_columns(
    area_px,
    voxel_xy_um: float,
) -> pd.DataFrame:
    """Physical size of the EXACT half-maximum footprint.

    ``area_px`` is the per-spot pixel count already produced by
    ``_sample_qki_at_miat_footprint`` (column ``miat_footprint_area_px``).
    Converts it to um^2 and to the diameter of the disk of equal area, so a
    footprint can be reported in physical units without re-deriving it.
    """
    a = pd.to_numeric(pd.Series(np.asarray(area_px).ravel()),
                      errors="coerce").to_numpy(dtype=float)
    vx = float(voxel_xy_um)
    with np.errstate(invalid="ignore"):
        um2 = a * vx * vx
        equiv = 2.0 * np.sqrt(np.maximum(um2, 0.0) / np.pi)
    equiv = np.where(np.isfinite(um2), equiv, np.nan)
    return pd.DataFrame({
        "footprint_area_um2": um2,
        "footprint_equiv_diameter_um": equiv,
    })


def size_rollup(
    spots: pd.DataFrame,
    *,
    prefix: str,
    nuclear_only: bool = True,
) -> dict:
    """Median fitted FWHM (um) and median footprint area (um^2) for a group.

    Only ``size_fit_ok == 1`` rows contribute to the FWHM median. Returns an
    all-keys dict (NaN / 0 when empty) so the column set is stable across
    images and nuclei.
    """
    out = {
        f"{prefix}_median_size_fit_fwhm_um": float("nan"),
        f"{prefix}_median_footprint_area_um2": float("nan"),
        f"{prefix}_n_size_fit_ok": 0,
        f"{prefix}_frac_size_fit_ok": float("nan"),
    }
    if spots is None or len(spots) == 0:
        return out
    sub = spots
    if nuclear_only and "in_nucleus" in sub.columns:
        sub = sub[pd.to_numeric(sub["in_nucleus"], errors="coerce").fillna(0) > 0]
    if len(sub) == 0:
        return out
    if "size_fit_ok" in sub.columns:
        okc = pd.to_numeric(sub["size_fit_ok"], errors="coerce").fillna(0) > 0
        out[f"{prefix}_frac_size_fit_ok"] = float(okc.mean())
        if "size_fit_fwhm_um" in sub.columns:
            vals = pd.to_numeric(sub.loc[okc, "size_fit_fwhm_um"],
                                 errors="coerce").dropna()
            out[f"{prefix}_n_size_fit_ok"] = int(len(vals))
            if len(vals):
                out[f"{prefix}_median_size_fit_fwhm_um"] = float(vals.median())
    if "footprint_area_um2" in sub.columns:
        fa = pd.to_numeric(sub["footprint_area_um2"], errors="coerce").dropna()
        if len(fa):
            out[f"{prefix}_median_footprint_area_um2"] = float(fa.median())
    return out
