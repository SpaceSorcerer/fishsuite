"""Threshold-free single-plane MIAT/QKI colocalization metrics (2026-09-24).

Definitions follow ``MIAT_QKI_BASAL_Fig1_2026-09-24/_audit/metric_inventory.md``
section (c), items 1-4. All inputs are raw single-plane intensities on one
nucleus's nucleoplasm mask N (eligible pixels of that nucleus, nucleoli
excluded). Nothing here pools pixels or spots across nuclei and nothing here
is a test across nuclei: the Costes p is a per-nucleus descriptive tail
fraction.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import stats

MIN_PIXELS = 100
CCF_MAX_SHIFT_PX = 20
CCF_FLANK_PX = (15, 20)
COSTES_MIN_BLOCK_PX = 3
COSTES_MIN_BLOCKS = 10
COSTES_N_ITER = 200
RANK_CUTOFFS = (0.9, 0.75)


def null_midrank(observed: np.ndarray, null_draws: np.ndarray) -> np.ndarray:
    """Per-spot mid-rank of the observed value within its own null draws.

    ``null_draws`` has one column per spot and one row per draw (K rows).
    u_i = (#{d_ik < o_i} + 0.5 * #{d_ik = o_i}) / K.
    """
    observed = np.asarray(observed, dtype=float)
    draws = np.asarray(null_draws, dtype=float)
    if draws.ndim != 2 or draws.shape[1] != observed.size:
        raise ValueError("null_draws must have shape (n_draws, n_spots)")
    if draws.shape[0] == 0:
        return np.full(observed.size, np.nan)
    below = np.count_nonzero(draws < observed[None, :], axis=0)
    ties = np.count_nonzero(draws == observed[None, :], axis=0)
    return (below + 0.5 * ties) / draws.shape[0]


def rank_summary(u: np.ndarray) -> dict:
    u = np.asarray(u, dtype=float)
    if u.size == 0:
        return dict(mean_null_rank_qki_at_miat=np.nan, frac_spots_ge_null_q90=np.nan,
                    frac_spots_ge_null_q75=np.nan)
    return dict(mean_null_rank_qki_at_miat=float(u.mean()),
                frac_spots_ge_null_q90=float(np.mean(u >= RANK_CUTOFFS[0])),
                frac_spots_ge_null_q75=float(np.mean(u >= RANK_CUTOFFS[1])))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    da, db = a - a.mean(), b - b.mean()
    va, vb = float(da @ da), float(db @ db)
    if va <= 0 or vb <= 0:
        return float("nan")
    return float((da @ db) / np.sqrt(va * vb))


def masked_correlations(miat, qki, mask, min_pixels: int = MIN_PIXELS) -> dict:
    """Pearson r and Spearman rho (average ranks) over the pixels of ``mask``."""
    mask = np.asarray(mask, dtype=bool)
    m = np.asarray(miat, dtype=float)[mask]
    q = np.asarray(qki, dtype=float)[mask]
    out = dict(pearson=float("nan"), spearman=float("nan"), n_pixels=int(m.size), reason="")
    if m.size == 0:
        out["reason"] = "R0"
    elif m.size < min_pixels:
        out["reason"] = "LOW_PIX"
    elif np.ptp(m) == 0 or np.ptp(q) == 0:
        out["reason"] = "ZERO_VAR"
    else:
        out["pearson"] = _pearson(m, q)
        out["spearman"] = _pearson(stats.rankdata(m), stats.rankdata(q))
    return out


def _crop(mask):
    ys, xs = np.nonzero(mask)
    return slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1)


def ccf_curve(miat, qki, mask, *, axis: int, max_shift: int = CCF_MAX_SHIFT_PX,
              min_pairs: int = MIN_PIXELS):
    """Van Steensel CCF: r(d) = Pearson(M(p), Q(p + d e_axis)) over pixel pairs
    whose both members lie in ``mask``. ``axis=1`` shifts along x, ``axis=0``
    along y. A positive peak shift means QKI sits +d px from MIAT."""
    mask = np.asarray(mask, dtype=bool)
    shifts = np.arange(-int(max_shift), int(max_shift) + 1)
    r = np.full(shifts.size, np.nan)
    n = np.zeros(shifts.size, dtype=int)
    if not mask.any():
        return shifts, r, n
    sy, sx = _crop(mask)
    m = np.asarray(miat, dtype=float)[sy, sx]
    q = np.asarray(qki, dtype=float)[sy, sx]
    k = mask[sy, sx]
    if axis == 0:
        m, q, k = m.T, q.T, k.T
    width = k.shape[1]
    for i, d in enumerate(shifts):
        if abs(d) >= width:
            continue
        if d >= 0:
            a, b = slice(0, width - d), slice(d, width)
        else:
            a, b = slice(-d, width), slice(0, width + d)
        both = k[:, a] & k[:, b]
        n[i] = int(both.sum())
        if n[i] >= min_pairs:
            r[i] = _pearson(m[:, a][both], q[:, b][both])
    return shifts, r, n


def _crossing(shifts, r, start, step, half):
    j = start
    while 0 <= j + step < r.size and np.isfinite(r[j + step]) and r[j + step] >= half:
        j += step
    nxt = j + step
    if not (0 <= nxt < r.size) or not np.isfinite(r[nxt]):
        return float("nan")
    return float(shifts[nxt] + (half - r[nxt]) / (r[j] - r[nxt]) * (shifts[j] - shifts[nxt]))


def ccf_summary(shifts, r, flank: Sequence[int] = CCF_FLANK_PX) -> dict:
    """Peak (argmax; ties to the smallest |shift|), r(0), flank mean over
    flank[0] <= |d| <= flank[1], r(0) - flank, and FWHM in px measured at
    flank + (peak - flank)/2 with linear interpolation between samples."""
    shifts, r = np.asarray(shifts), np.asarray(r, dtype=float)
    out = dict(peak_shift_px=np.nan, peak_r=np.nan, r0=np.nan, flank_r=np.nan,
               r0_minus_flank=np.nan, fwhm_px=np.nan, reason="", fwhm_reason="")
    finite = np.isfinite(r)
    if not finite.any():
        out["reason"] = out["fwhm_reason"] = "NO_CCF"
        return out
    zero = np.flatnonzero(shifts == 0)
    out["r0"] = float(r[zero[0]]) if zero.size else np.nan
    peak_r = float(np.nanmax(r))
    candidates = np.flatnonzero(finite & (r == peak_r))
    peak = int(candidates[np.lexsort((shifts[candidates], np.abs(shifts[candidates])))[0]])
    out["peak_shift_px"], out["peak_r"] = int(shifts[peak]), peak_r
    ring = (np.abs(shifts) >= flank[0]) & (np.abs(shifts) <= flank[1]) & finite
    flank_r = float(r[ring].mean()) if ring.any() else float("nan")
    out["flank_r"] = flank_r
    out["r0_minus_flank"] = out["r0"] - flank_r
    if not np.isfinite(flank_r) or not peak_r - flank_r > 1e-12:
        out["fwhm_reason"] = "NO_PEAK_ABOVE_FLANK"
        return out
    half = flank_r + 0.5 * (peak_r - flank_r)
    left = _crossing(shifts, r, peak, -1, half)
    right = _crossing(shifts, r, peak, +1, half)
    if np.isfinite(left) and np.isfinite(right):
        out["fwhm_px"] = right - left
    else:
        out["fwhm_reason"] = "NO_HALF_CROSSING"
    return out


def _check_pixel_size(pixel_size_um):
    try:
        value = float(pixel_size_um)
    except (TypeError, ValueError) as exc:
        raise ValueError("pixel_size_um must come from image metadata; none given") from exc
    if not np.isfinite(value) or value <= 0:
        raise ValueError("pixel_size_um must come from image metadata and be finite and > 0")
    return value


def nucleus_ccf(miat, qki, mask, *, pixel_size_um, max_shift: int = CCF_MAX_SHIFT_PX,
                min_pairs: int = MIN_PIXELS, curves: list | None = None) -> dict:
    """Per-nucleus CCF along x and y with the summary columns of the chain."""
    px = _check_pixel_size(pixel_size_um)
    row = {}
    for name, axis in (("x", 1), ("y", 0)):
        shifts, r, n = ccf_curve(miat, qki, mask, axis=axis, max_shift=max_shift,
                                 min_pairs=min_pairs)
        s = ccf_summary(shifts, r)
        if curves is not None:
            curves.extend(dict(axis=name, shift_px=int(d), shift_um=float(d * px),
                               r=float(v), n_pairs=int(c)) for d, v, c in zip(shifts, r, n))
        if name == "x":
            row["ccf_r0"] = s["r0"]
            row["na_reason_ccf_r0"] = "" if np.isfinite(s["r0"]) else (s["reason"] or "LOW_PIX")
        row[f"ccf_peak_r_{name}"] = s["peak_r"]
        row[f"ccf_peak_shift_px_{name}"] = s["peak_shift_px"]
        row[f"ccf_peak_shift_um_{name}"] = (s["peak_shift_px"] * px
                                            if np.isfinite(s["peak_shift_px"]) else np.nan)
        row[f"ccf_peak_fwhm_px_{name}"] = s["fwhm_px"]
        row[f"ccf_r0_minus_flank_{name}"] = s["r0_minus_flank"]
        row[f"na_reason_ccf_peak_{name}"] = s["reason"]
        row[f"na_reason_ccf_peak_fwhm_px_{name}"] = s["fwhm_reason"]
        row[f"na_reason_ccf_r0_minus_flank_{name}"] = (
            "" if np.isfinite(s["r0_minus_flank"]) else (s["reason"] or "NO_FLANK"))
    return row


def costes_block_size(footprint_areas_px) -> int | None:
    """round(sqrt(median footprint area)) px, at least 3 px; None without footprints."""
    areas = np.asarray(list(footprint_areas_px), dtype=float)
    if areas.size == 0:
        return None
    return max(COSTES_MIN_BLOCK_PX, int(np.floor(np.sqrt(np.median(areas)) + 0.5)))


def costes_randomization(miat, qki, mask, block_px, n_iter, rng: np.random.Generator,
                         min_blocks: int = COSTES_MIN_BLOCKS,
                         min_pixels: int = MIN_PIXELS) -> dict:
    """Costes block-scramble randomization of r over the full blocks of ``mask``.

    The nucleus bounding box is tiled from its top-left corner into
    ``block_px`` x ``block_px`` blocks. Only blocks lying entirely inside the
    mask are used. QKI blocks are permuted among those positions (orientation
    kept), MIAT stays fixed, and r is recomputed over the same pixels.
    p = (1 + #{r_perm >= r_obs}) / (1 + n_iter): one-sided, per-nucleus,
    descriptive.
    """
    out = dict(costes_rand_n_blocks=0, costes_rand_r_obs=np.nan, costes_rand_null_mean_r=np.nan,
               costes_rand_r_obs_minus_null_mean=np.nan, costes_rand_p=np.nan, reason="")
    mask = np.asarray(mask, dtype=bool)
    if block_px is None:
        out["reason"] = "NO_BLOCK_SIZE"
        return out
    if not mask.any():
        out["reason"] = "R0"
        return out
    b = int(block_px)
    sy, sx = _crop(mask)
    k = mask[sy, sx]
    height, width = k.shape
    ph, pw = -height % b, -width % b
    k = np.pad(k, ((0, ph), (0, pw)), constant_values=False)
    m = np.pad(np.asarray(miat, dtype=float)[sy, sx], ((0, ph), (0, pw)))
    q = np.pad(np.asarray(qki, dtype=float)[sy, sx], ((0, ph), (0, pw)))
    nby, nbx = k.shape[0] // b, k.shape[1] // b

    def blocks(a):
        return a.reshape(nby, b, nbx, b).transpose(0, 2, 1, 3).reshape(nby * nbx, b * b)

    full = blocks(k).all(axis=1)
    count = int(full.sum())
    out["costes_rand_n_blocks"] = count
    if count < min_blocks or count * b * b < min_pixels:
        out["reason"] = "FEW_BLOCKS"
        return out
    mb, qb = blocks(m)[full], blocks(q)[full]
    mc, qc = mb - mb.mean(), qb - qb.mean()
    denominator = np.sqrt(float((mc * mc).sum()) * float((qc * qc).sum()))
    if denominator <= 0:
        out["reason"] = "ZERO_VAR"
        return out
    r_obs = float((mc * qc).sum() / denominator)
    order = np.stack([rng.permutation(count) for _ in range(int(n_iter))])
    r_perm = np.einsum("ij,nij->n", mc, qc[order]) / denominator
    out.update(costes_rand_r_obs=r_obs, costes_rand_null_mean_r=float(r_perm.mean()),
               costes_rand_r_obs_minus_null_mean=float(r_obs - r_perm.mean()),
               costes_rand_p=float((1 + np.count_nonzero(r_perm >= r_obs)) / (1 + int(n_iter))))
    return out
