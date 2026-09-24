"""Threshold-free single-plane MIAT/QKI colocalization metrics (2026-09-24).

Definitions follow ``MIAT_QKI_BASAL_Fig1_2026-09-24/_audit/metric_inventory.md``
section (c), items 1-4. All inputs are raw single-plane intensities on one
nucleus's nucleoplasm mask N (eligible pixels of that nucleus, nucleoli
excluded). Nothing here pools pixels or spots across nuclei and nothing here
is a test across nuclei: the Costes p is a per-nucleus descriptive tail
fraction. Round 2 (Astra review 2026-09-24): uniform-position percentile
score with randomized ties; Costes block size from per-channel ACF and the
theoretical PSF, on a frozen complete-block core with a coverage gate.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import stats

MIN_PIXELS = 100
CCF_MAX_SHIFT_PX = 20
CCF_FLANK_PX = (15, 20)
ACF_MAX_RADIUS_PX = 20
COSTES_MIN_BLOCKS = 10
COSTES_MIN_COVERAGE = 0.80
COSTES_N_ITER = 200
PSF_LATERAL_FACTOR = 0.51
UPP_CUTOFFS = (0.9, 0.75)


def uniform_position_percentile(observed, null_draws, rng: np.random.Generator) -> np.ndarray:
    """Per-spot uniform-position percentile score with randomized tie-breaking.

    ``null_draws`` is (K draws, n spots). R_i = #{d_ik < o_i} + J_i with J_i
    uniform on {0, ..., #{d_ik = o_i}}; u_i = R_i / K. If the observed spot is
    exchangeable with its K uniform-position placements, R_i is uniform on
    {0..K} exactly, ties included (a mid-rank would pile tied spots at 0.5).
    """
    observed = np.asarray(observed, dtype=float)
    draws = np.asarray(null_draws, dtype=float)
    if draws.ndim != 2 or draws.shape[1] != observed.size:
        raise ValueError("null_draws must have shape (n_draws, n_spots)")
    if draws.shape[0] == 0:
        return np.full(observed.size, np.nan)
    below = np.count_nonzero(draws < observed[None, :], axis=0)
    ties = np.count_nonzero(draws == observed[None, :], axis=0)
    jitter = rng.integers(0, ties + 1) if observed.size else np.zeros(0, dtype=int)
    return (below + jitter) / draws.shape[0]


def upp_chance_ge(k: int, cutoff: float) -> float:
    """P(u >= cutoff) under exchangeability with K draws: ranks uniform on {0..K}."""
    k = int(k)
    return (k - int(np.ceil(cutoff * k - 1e-9)) + 1) / (k + 1)


def upp_summary(u: np.ndarray, k: int) -> dict:
    u = np.asarray(u, dtype=float)
    out = dict(mean_uniform_position_percentile_qki=np.nan, frac_spots_upp_ge_0p90=np.nan,
               frac_spots_upp_ge_0p75=np.nan, n_spots_upp=int(u.size), chance_mean_upp=0.5,
               chance_frac_spots_upp_ge_0p90=upp_chance_ge(k, UPP_CUTOFFS[0]) if k else np.nan,
               chance_frac_spots_upp_ge_0p75=upp_chance_ge(k, UPP_CUTOFFS[1]) if k else np.nan)
    if u.size:
        out.update(mean_uniform_position_percentile_qki=float(u.mean()),
                   frac_spots_upp_ge_0p90=float(np.mean(u >= UPP_CUTOFFS[0] - 1e-12)),
                   frac_spots_upp_ge_0p75=float(np.mean(u >= UPP_CUTOFFS[1] - 1e-12)))
    return out


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


def psf_fwhm_px(emission_nm, numerical_aperture, pixel_nm) -> float:
    """Theoretical lateral PSF FWHM, 0.51 * lambda_em / NA, in pixels. Raises
    when any optical value is missing or invalid; there is no default."""
    values = []
    for name, value in (("emission wavelength (nm)", emission_nm),
                        ("objective NA", numerical_aperture), ("pixel size (nm)", pixel_nm)):
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"PSF needs {name} from the acquisition metadata; got {value!r}") from exc
        if not np.isfinite(number) or number <= 0:
            raise ValueError(f"PSF needs a finite positive {name}; got {value!r}")
        values.append(number)
    emission, na, pixel = values
    return PSF_LATERAL_FACTOR * emission / na / pixel


def acf_radial(x, mask, *, max_radius: int = ACF_MAX_RADIUS_PX, min_pairs: int = MIN_PIXELS):
    """Overlap-normalized, mean-centered 2-D autocorrelation of ``x`` over the
    pixel pairs whose BOTH endpoints lie in ``mask``, radially averaged.

    ACF(d) = sum over in-mask pairs (x_p - m)(x_{p+d} - m) / (n_pairs(d) * var),
    m and var over the mask. Radial bin k = round(|d|); each bin is
    pair-count weighted and kept only when it has >= ``min_pairs`` pairs.
    Returns (radii, acf, pairs, reason).
    """
    from scipy.signal import correlate

    mask = np.asarray(mask, dtype=bool)
    radii = np.arange(int(max_radius) + 1)
    acf = np.full(radii.size, np.nan)
    pairs = np.zeros(radii.size, dtype=np.int64)
    n = int(mask.sum())
    if n == 0:
        return radii, acf, pairs, "R0"
    if n < MIN_PIXELS:
        return radii, acf, pairs, "LOW_PIX"
    sy, sx = _crop(mask)
    k = mask[sy, sx]
    v = np.asarray(x, dtype=float)[sy, sx]
    mean = float(v[k].mean())
    var = float(((v[k] - mean) ** 2).mean())
    if not var > 0:
        return radii, acf, pairs, "ZERO_VAR"
    a = np.where(k, v - mean, 0.0)
    num = correlate(a, a, mode="full", method="fft")
    cnt = np.rint(correlate(k.astype(float), k.astype(float), mode="full", method="fft"))
    cy, cx = k.shape[0] - 1, k.shape[1] - 1
    r = int(max_radius)
    dy, dx = np.mgrid[-r:r + 1, -r:r + 1]
    y0, y1 = max(0, cy - r), min(num.shape[0], cy + r + 1)
    x0, x1 = max(0, cx - r), min(num.shape[1], cx + r + 1)
    window_num = np.zeros(dy.shape)
    window_cnt = np.zeros(dy.shape)
    window_num[y0 - cy + r:y1 - cy + r, x0 - cx + r:x1 - cx + r] = num[y0:y1, x0:x1]
    window_cnt[y0 - cy + r:y1 - cy + r, x0 - cx + r:x1 - cx + r] = cnt[y0:y1, x0:x1]
    bins = np.rint(np.hypot(dy, dx)).astype(int)
    for radius in radii:
        sel = (bins == radius) & (window_cnt > 0)
        total = float(window_cnt[sel].sum())
        pairs[radius] = int(total)
        if total >= min_pairs:
            acf[radius] = float(window_num[sel].sum()) / total / var
    return radii, acf, pairs, ""


def acf_fwhm_px(x, mask, *, max_radius: int = ACF_MAX_RADIUS_PX, min_pairs: int = MIN_PIXELS):
    """FWHM (px) of the central ACF peak: 2 x the radius at which the radial
    ACF first falls below 0.5, linearly interpolated. (nan, reason) when it
    does not fall to half height inside the estimable range."""
    radii, acf, _pairs, reason = acf_radial(x, mask, max_radius=max_radius, min_pairs=min_pairs)
    if reason:
        return float("nan"), reason
    for i in range(1, radii.size):
        if not np.isfinite(acf[i]):
            break
        if acf[i] < 0.5:
            r_half = radii[i - 1] + (acf[i - 1] - 0.5) / (acf[i - 1] - acf[i]) * (radii[i] - radii[i - 1])
            return float(2.0 * r_half), ""
    return float("nan"), "ACF_NO_HALF"


def costes_block_from_widths(fwhm_miat_px, fwhm_qki_px, psf_px):
    """b = ceil(max(PSF_FWHM_px, min(FWHM_ACF_MIAT_px, FWHM_ACF_QKI_px)))."""
    values = [float(fwhm_miat_px), float(fwhm_qki_px), float(psf_px)]
    if not all(np.isfinite(values)):
        return None
    return int(np.ceil(max(values[2], min(values[0], values[1])) - 1e-9))


def _complete_blocks(mask, block_px, oy, ox):
    """Complete b x b blocks of ``mask`` for a grid whose origin sits (oy, ox)
    px above/left of the mask bounding box. Returns (core mask, n blocks)."""
    b = int(block_px)
    sy, sx = _crop(mask)
    k = np.pad(mask[sy, sx], ((oy, 0), (ox, 0)), constant_values=False)
    k = np.pad(k, ((0, -k.shape[0] % b), (0, -k.shape[1] % b)), constant_values=False)
    nby, nbx = k.shape[0] // b, k.shape[1] // b
    full = k.reshape(nby, b, nbx, b).all(axis=(1, 3))
    core_pad = np.repeat(np.repeat(full, b, axis=0), b, axis=1)
    core = np.zeros(mask.shape, dtype=bool)
    height, width = sy.stop - sy.start, sx.stop - sx.start
    core[sy, sx] = core_pad[oy:oy + height, ox:ox + width]
    return core, int(full.sum())


def best_tile_phase(mask, block_px):
    """Grid phase (oy, ox) in [0, b)^2 maximising complete-block coverage,
    chosen from mask geometry only (ties: smallest oy, then ox). Returns
    ((oy, ox), core mask, n blocks, coverage = core pixels / mask pixels)."""
    mask = np.asarray(mask, dtype=bool)
    best = None
    for oy in range(int(block_px)):
        for ox in range(int(block_px)):
            core, count = _complete_blocks(mask, block_px, oy, ox)
            if best is None or count > best[2]:
                best = ((oy, ox), core, count)
    phase, core, count = best
    return phase, core, count, float(core.sum() / mask.sum())


def costes_randomization(miat, qki, mask, block_px, n_iter, rng: np.random.Generator,
                         min_blocks: int = COSTES_MIN_BLOCKS,
                         min_coverage: float = COSTES_MIN_COVERAGE) -> dict:
    """Costes block-scramble randomization on the frozen block core N_core.

    The tile phase maximising complete-block coverage is chosen from the mask
    alone; N_core (the union of complete blocks) is then frozen, and the
    observed r and every permutation (QKI blocks permuted among N_core block
    positions, orientation kept, MIAT fixed) use exactly N_core.
    p = (1 + #{r_perm >= r_obs}) / (1 + n_iter); one-sided, per-nucleus,
    descriptive. NA (MASK_BLOCK_COVERAGE) below ``min_blocks`` complete blocks
    or ``min_coverage`` of the mask.
    """
    out = dict(costes_tile_phase_y=np.nan, costes_tile_phase_x=np.nan, costes_rand_n_blocks=0,
               costes_core_coverage=np.nan, costes_rand_n_draws=int(n_iter),
               costes_rand_r_obs=np.nan, costes_rand_null_mean_r=np.nan,
               costes_rand_r_obs_minus_null_mean=np.nan, costes_rand_p=np.nan, reason="")
    mask = np.asarray(mask, dtype=bool)
    if block_px is None:
        out["reason"] = "NO_BLOCK_SIZE"
        return out
    if not mask.any():
        out["reason"] = "R0"
        return out
    b = int(block_px)
    (oy, ox), core, count, coverage = best_tile_phase(mask, b)
    out.update(costes_tile_phase_y=oy, costes_tile_phase_x=ox, costes_rand_n_blocks=count,
               costes_core_coverage=coverage)
    if count < min_blocks or coverage < min_coverage:
        out["reason"] = "MASK_BLOCK_COVERAGE"
        return out
    sy, sx = _crop(core)
    c = core[sy, sx]
    m = np.asarray(miat, dtype=float)[sy, sx]
    q = np.asarray(qki, dtype=float)[sy, sx]
    nby, nbx = c.shape[0] // b, c.shape[1] // b
    # core bbox starts on a block boundary and spans whole blocks
    c, m, q = c[:nby * b, :nbx * b], m[:nby * b, :nbx * b], q[:nby * b, :nbx * b]

    def blocks(a):
        return a.reshape(nby, b, nbx, b).transpose(0, 2, 1, 3).reshape(nby * nbx, b * b)

    full = blocks(c).all(axis=1)
    if int(full.sum()) != count:
        raise AssertionError("frozen core block count changed between phase search and scramble")
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


def nucleus_costes(miat, qki, mask, *, psf_px, n_iter, rng: np.random.Generator) -> dict:
    """ACF/PSF-sized Costes randomization for one nucleus, with every
    parameter recorded. No PSF -> NO_PSF (never a default block)."""
    fwhm_m, reason_m = acf_fwhm_px(miat, mask)
    fwhm_q, reason_q = acf_fwhm_px(qki, mask)
    psf = float(psf_px) if psf_px is not None else float("nan")
    block = costes_block_from_widths(fwhm_m, fwhm_q, psf) if np.isfinite(psf) else None
    row = dict(costes_psf_fwhm_px=psf, costes_acf_fwhm_px_miat=fwhm_m, costes_acf_fwhm_px_qki=fwhm_q,
               costes_block_px=float(block) if block is not None else float("nan"))
    result = costes_randomization(miat, qki, mask, block, n_iter, rng)
    reason = result.pop("reason")
    if block is None:
        reason = "NO_PSF" if not np.isfinite(psf) else ("ACF_" + (reason_m or reason_q).removeprefix("ACF_"))
    row.update(result)
    row["na_reason_costes_rand"] = reason
    return row


def nucleoplasm_sensitivity(miat, qki, dapi, labels, pixel_size_um, params: dict,
                            percentiles) -> "pd.DataFrame":
    """Nucleoplasm Pearson/Spearman with the nucleolus mask re-detected from
    DAPI at each intra-nuclear percentile (other nucleolus parameters fixed)."""
    import pandas as pd
    from .nucleolus import NucleolusParams, detect_nucleoli

    labels = np.asarray(labels)
    rows = []
    for percentile in percentiles:
        settings = NucleolusParams(**{**params, "intra_nuclear_percentile": float(percentile)})
        nucleoli = detect_nucleoli(labels, np.asarray(dapi), pixel_size_um, settings)
        for nucleus_id in np.unique(labels[labels > 0]):
            nucleus = labels == nucleus_id
            nucleolus = nucleoli == nucleus_id
            corr = masked_correlations(miat, qki, nucleus & ~nucleolus)
            rows.append(dict(nucleus_id=int(nucleus_id), percentile=percentile,
                             nucleolus_area_frac=float(nucleolus.sum() / nucleus.sum()),
                             n_pixels_nucleoplasm=corr["n_pixels"],
                             pearson_r_nucleoplasm=corr["pearson"],
                             spearman_rho_nucleoplasm=corr["spearman"], na_reason=corr["reason"]))
    return pd.DataFrame(rows)
