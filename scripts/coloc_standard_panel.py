"""Field-standard colocalization readouts for a COMPLETED fishsuite run.

CPU only. Read-only against the run directory: no segmentation, no spot
re-detection, no writes back into ``--run``. Adds the readouts a reviewer
outside this lab expects to see next to fishsuite's own rotation-null work:

  * per-nucleus Pearson r, Manders M1/M2 at a per-nucleus COSTES threshold,
    and Li's ICQ, reported beside the run's own columns for cross-check;
  * an object-based coloc fraction with a within-nucleus shuffle control;
  * a cytofluorogram (pooled nuclear pixels, log-density hexbin);
  * intensity line profiles straight across representative nuclei;
  * QC overlays that show WHICH puncta were called colocalized.

Definitions are documented in the workbook README sheet and in
``FIGURE_INDEX.md``.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import re
import os
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
from scipy import integrate, stats

from fishsuite.core import io as _io
from fishsuite.core import output as _out
from fishsuite.core import thresholds as _thr
from fishsuite.runner import (
    _compute_common_filename_prefix,
    _simplify_stem,
    _stem_with_condition,
)

# ------------------------------------------------------------------ constants
# Locked for this dataset by 14_REPORT_HARMONIZED_2026-09-03/make_figures_rnaseh2b.py.
LINE_COLORS = {"WT": "#333333", "QKI-KO": "#CC79A7", "Sec-Only": "#999999"}
# Arms in plotting order, REFERENCE first and TEST second; contrasts report
# test minus reference. Set once from the CLI / run conditions in main().
ARMS = ["WT", "QKI-KO"]
SEC_ONLY = "Sec-Only"
_ARM_FALLBACK_COLORS = ["#333333", "#CC79A7", "#0072B2", "#E69F00", "#009E73",
                        "#56B4E9", "#D55E00"]


def arm_color(name):
    if name in LINE_COLORS:
        return LINE_COLORS[name]
    i = ARMS.index(name) if name in ARMS else len(ARMS)
    return _ARM_FALLBACK_COLORS[i % len(_ARM_FALLBACK_COLORS)]

OKABE = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
         "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00",
         "purple": "#CC79A7", "black": "#000000"}
CALLED_COLOR = OKABE["vermillion"]      # punctum called colocalized
NOTCALLED_COLOR = OKABE["sky"]          # punctum not called
ALPHA = 0.05
POWER = 0.80
SHUFFLE_DRAWS = 200
CYTO_MAX_PIXELS = 2_000_000
# Costes-convergence floor below which the primary call falls back to the run's
# single batch threshold. A Costes-with-fallback column built from fewer than
# this fraction of converged nuclei is a mixture of two conventions whose arm
# mean tracks the convergence fraction, not the images.
COSTES_MIN_CONVERGENCE_PCT = 90.0
PER_NUCLEUS_PIXEL_CAP = 12_000          # cap on pixels kept per nucleus for pooling


# ---------------------------------------------------------------------- style
def set_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "text.color": "black", "axes.labelcolor": "black",
        "axes.edgecolor": "black", "axes.linewidth": 0.8,
        "xtick.color": "black", "ytick.color": "black",
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7,
        "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })


def shade(hex_color, frac):
    h = hex_color.lstrip("#")
    rgb = np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)]) / 255.0
    return tuple(1.0 - (1.0 - rgb) * float(np.clip(frac, 0.05, 1.0)))


def stars(p):
    if p is None or not np.isfinite(p):
        return "n/a"
    for thr, s in ((1e-4, "****"), (1e-3, "***"), (1e-2, "**"), (0.05, "*")):
        if p < thr:
            return s
    return "ns"


def fmt_p(p):
    if p is None or not np.isfinite(p):
        return "NA"
    p = float(p)
    return f"{p:.4g}" if p >= 1e-4 else f"{p:.2e}"


def wrap_foot(text, fig_w, size):
    width = max(60, int((fig_w - 0.40) * 72.0 / (size * 0.50)))
    out = []
    for line in str(text).splitlines():
        out.extend(textwrap.wrap(line, width) or [""])
    return out


NOTE_TEXT = ""


def stamp_foot(fig, text, size=5.6, y=0.012):
    if NOTE_TEXT:
        text = str(text).rstrip() + " " + NOTE_TEXT
    lines = wrap_foot(text, fig.get_figwidth(), size)
    fig.text(0.016, y, "\n".join(lines), ha="left", va="bottom", fontsize=size,
             color="#333333", linespacing=1.45)


def stamp_head(fig, title, subtitle, filt, wrap=132, y=0.982, sub_size=7.0,
               filt_size=6.0):
    """Title, then subtitle, then the gate band, stacked top-down. Returns the
    figure-fraction y of the bottom of the block so the axes can start below it."""
    hpt = fig.get_figheight() * 72.0
    fig.suptitle(title, y=y, fontsize=10.5, fontweight="bold")
    sub_lines = textwrap.wrap(str(subtitle), wrap) or [""]
    filt_lines = textwrap.wrap(str(filt), wrap) or [""]
    y1 = y - 13.0 / hpt
    fig.text(0.5, y1, chr(10).join(sub_lines), ha="center", va="top", fontsize=sub_size,
             color="#444444", linespacing=1.35)
    y2 = y1 - (len(sub_lines) * sub_size * 1.42 + 5.0) / hpt
    fig.text(0.5, y2, chr(10).join(filt_lines), ha="center", va="top", fontsize=filt_size,
             color="#444444", linespacing=1.35)
    return y2 - (len(filt_lines) * filt_size * 1.42) / hpt


def foot_bottom(fig, text, size, pad=0.042):
    """Figure-fraction bottom margin that leaves room for the footnote block."""
    if NOTE_TEXT:
        text = str(text).rstrip() + " " + NOTE_TEXT
    lines = wrap_foot(text, fig.get_figwidth(), size)
    return 0.012 + (len(lines) * size * 1.45) / (fig.get_figheight() * 72.0) + pad


# ---------------------------------------------------------------- statistics
def hedges_g(a, b):
    """Bias-corrected standardized mean difference. Same formula as
    14_REPORT_HARMONIZED_2026-09-03/build_report_rnaseh2b.py::hedges_g."""
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float("nan")
    s1, s2 = a.std(ddof=1), b.std(ddof=1)
    sp2 = ((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2)
    if not np.isfinite(sp2) or sp2 <= 0:
        return float("nan")
    d = (a.mean() - b.mean()) / math.sqrt(sp2)
    j = 1.0 - 3.0 / (4.0 * (n1 + n2) - 9.0)
    return float(j * d)


def welch(a, b, alpha=ALPHA):
    """Welch t of a against b; ``diff`` is mean(a) - mean(b)."""
    n1, n2 = len(a), len(b)
    out = dict(n_test=n1, n_ref=n2,
               mean_test=float(a.mean()) if n1 else np.nan,
               mean_ref=float(b.mean()) if n2 else np.nan,
               diff=np.nan, ci_low=np.nan, ci_high=np.nan, hedges_g=np.nan,
               t=np.nan, df=np.nan, p_welch=np.nan, welch_note="")
    if n1 < 2 or n2 < 2:
        out["welch_note"] = "fewer than 2 wells with a finite value in one group"
        return out
    v1, v2 = a.var(ddof=1), b.var(ddof=1)
    se = math.sqrt(v1 / n1 + v2 / n2)
    out["diff"] = float(a.mean() - b.mean())
    out["hedges_g"] = hedges_g(a, b)
    if se == 0:
        out["welch_note"] = "zero within-group variance in both groups; Welch t undefined"
        return out
    if v1 == 0 or v2 == 0:
        which = "test" if v1 == 0 else "reference"
        out["welch_note"] = (
            "zero within-group variance in the {} arm: every well gave the same value. "
            "Welch t and Hedges g are then driven by the OTHER arm's residual scatter and "
            "must not be read as evidence strength - the separation itself is the "
            "result.".format(which))
    df = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    tcrit = stats.t.ppf(1 - alpha / 2, df)
    t, p = stats.ttest_ind(a, b, equal_var=False)
    out.update(ci_low=out["diff"] - tcrit * se, ci_high=out["diff"] + tcrit * se,
               t=float(t), df=float(df), p_welch=float(p))
    return out


def _power_two_sided(d, alpha, n1, n2):
    """Power of the two-sided two-sample t test by quadrature over the chi-square
    mixing variable. scipy ``nct`` and statsmodels ``TTestIndPower`` are
    non-monotone in alpha at df=4 below ~0.004; this form is stable. Same
    routine as build_report_rnaseh2b.py::_power_two_sided."""
    df = n1 + n2 - 2
    if df < 1:
        return float("nan")
    ncp = d * math.sqrt(n1 * n2 / (n1 + n2))
    c = stats.t.ppf(1 - alpha / 2, df)
    chi2 = stats.chi2(df)
    hi = float(chi2.ppf(1 - 1e-12))
    up, _ = integrate.quad(
        lambda v: stats.norm.sf(c * math.sqrt(v / df) - ncp) * chi2.pdf(v), 0, hi, limit=400)
    lo, _ = integrate.quad(
        lambda v: stats.norm.cdf(-c * math.sqrt(v / df) - ncp) * chi2.pdf(v), 0, hi, limit=400)
    return float(up + lo)


def mde_hedges_g(alpha=ALPHA, power=POWER, n1=3, n2=3):
    if not np.isfinite(alpha) or alpha <= 0 or alpha >= 1:
        return float("nan")
    lo, hi = 0.0, 1.0
    while _power_two_sided(hi, alpha, n1, n2) < power and hi < 1e3:
        hi *= 2
    for _ in range(80):
        mid = (lo + hi) / 2
        if _power_two_sided(mid, alpha, n1, n2) < power:
            lo = mid
        else:
            hi = mid
    j = 1.0 - 3.0 / (4.0 * (n1 + n2) - 9.0)
    return float((lo + hi) / 2 * j)


def mean_ci(v, alpha=ALPHA):
    v = np.asarray([x for x in v if np.isfinite(x)], dtype=float)
    n = v.size
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    m = float(v.mean())
    if n < 2:
        return m, float("nan"), float("nan"), n
    se = float(v.std(ddof=1)) / math.sqrt(n)
    t = stats.t.ppf(1 - alpha / 2, n - 1)
    return m, m - t * se, m + t * se, n


# ------------------------------------------------------------------- costes
def regression_line(r, a, fit="tls"):
    """Slope and intercept of the line Costes walks down.

    ``tls`` is the total-least-squares (orthogonal / reduced-major-axis) fit
    used by Costes et al. 2004 and by Fiji's Coloc2, and is the default here
    because this tool exists to produce the field-standard readouts. ``ols`` is
    the ordinary-least-squares fit that ``fishsuite.core.thresholds.
    costes_threshold`` uses internally, kept so the two can be compared.
    """
    r = np.asarray(r, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    n = r.size
    dr = r - r.mean()
    da = a - a.mean()
    vr = float(dr @ dr) / n
    va = float(da @ da) / n
    cv = float(dr @ da) / n
    if fit == "ols":
        if vr <= 0:
            return float("nan"), float("nan")
        slope = cv / vr
    else:
        if cv == 0:
            return float("nan"), float("nan")
        slope = (va - vr + math.sqrt((va - vr) ** 2 + 4.0 * cv * cv)) / (2.0 * cv)
    return slope, float(a.mean()) - slope * float(r.mean())


def costes_np(r, a, max_thresholds=256, min_below=10, fit="tls"):
    """Costes automatic threshold, vectorized.

    Walks candidate rna1 thresholds DOWN the regression line
    ``a = slope*r + intercept`` and returns the first pair at which the Pearson
    r of the strictly-below-both-thresholds pixels is <= 0. Same procedure as
    ``fishsuite.core.thresholds.costes_threshold`` (and identical to it when
    ``fit='ols'``), but with the per-candidate O(n) pass done in numpy instead
    of Python and with an explicit convergence flag instead of that function's
    silent MAD fallback.

    Returns ``(r_thr, a_thr, converged)``. On non-convergence the thresholds are
    NaN and the caller supplies its own fallback. Non-convergence is
    informative, not a defect: it says the two channels stay positively
    correlated at every intensity level, so there is no uncorrelated-background
    regime for Costes to find.
    """
    r = np.asarray(r, dtype=np.float64).ravel()
    a = np.asarray(a, dtype=np.float64).ravel()
    n = r.size
    if n < 20:
        return float("nan"), float("nan"), False
    slope, intercept = regression_line(r, a, fit=fit)
    if not (np.isfinite(slope) and np.isfinite(intercept)):
        return float("nan"), float("nan"), False
    uniq = np.unique(r)[::-1]
    step = max(1, uniq.size // max_thresholds)
    for r_t in uniq[::step]:
        a_t = slope * float(r_t) + intercept
        m = (r < r_t) & (a < a_t)
        k = int(m.sum())
        if k < min_below:
            continue
        br = r[m]
        ba = a[m]
        bdr = br - br.mean()
        bda = ba - ba.mean()
        vr = float(bdr @ bdr)
        va = float(bda @ bda)
        if vr > 0 and va > 0:
            bp = float(bdr @ bda) / math.sqrt(vr * va)
            if bp <= 0:
                return float(r_t), max(0.0, float(a_t)), True
    return float("nan"), float("nan"), False


def pixel_metrics(r, a, r_thr, a_thr):
    """Pearson r, Li ICQ and Manders M1/M2 on one nucleus's paired pixels.

    Pearson and ICQ are threshold-free. M1 = fraction of rna1 intensity in
    pixels where the partner is >= ``a_thr``; M2 = fraction of partner intensity
    where rna1 is >= ``r_thr``. The ``>=`` matches
    ``fishsuite.core.metrics.compute_coloc_metrics``, so the only difference
    from the run's own Manders columns is WHICH threshold is used.
    """
    r = np.asarray(r, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    n = r.size
    out = dict(pearson_r=np.nan, li_icq=np.nan, manders_m1=np.nan, manders_m2=np.nan,
               frac_above_thr_rna1=np.nan, frac_above_thr_partner=np.nan)
    if n < 10:
        return out
    dr = r - r.mean()
    da = a - a.mean()
    vr = float(dr @ dr)
    va = float(da @ da)
    out["pearson_r"] = float((dr @ da) / math.sqrt(vr * va)) if (vr > 0 and va > 0) else 0.0
    out["li_icq"] = float(int(((dr * da) > 0).sum()) / float(n) - 0.5)
    if np.isfinite(r_thr) and np.isfinite(a_thr):
        rpos = r >= r_thr
        apos = a >= a_thr
        r_sum = float(r.sum())
        a_sum = float(a.sum())
        out["manders_m1"] = float(r[apos].sum() / r_sum) if r_sum > 0 else 0.0
        out["manders_m2"] = float(a[rpos].sum() / a_sum) if a_sum > 0 else 0.0
        out["frac_above_thr_rna1"] = float(rpos.sum()) / n
        out["frac_above_thr_partner"] = float(apos.sum()) / n
    return out


# ---------------------------------------------------------------- footprints
def footprint_offsets_for_image(rna1_2d, partner_2d, spots_df, voxel_xy_um,
                                default_spot_diameter_um, *, half_max_frac=0.5,
                                bg_percentile=10.0, window_pad_px=2,
                                min_window_half=4, max_window_half=12,
                                peak_search_half=1):
    """Per-spot EXACT half-max footprint: pixel offsets, area and partner mean.

    Same construction as
    ``fishsuite.core.modes.rna_rna._sample_qki_at_miat_footprint``, which is
    what produced the run's ``qki_at_miat_footprint`` /
    ``miat_footprint_area_px`` columns. That function returns only the mean and
    the area, so it cannot be reused directly: the shuffle control has to
    TRANSLATE the footprint, which needs its pixel offsets. The offsets are the
    only thing added here; the area and mean this returns are asserted against
    the run's own columns so the reconstruction is proven, not assumed.

    Returns a list of dicts with ``dy``, ``dx`` (offsets from the rounded spot
    centre), ``area_px``, ``partner_mean`` and ``method``
    (``halfmax_connected_component`` or ``fitted_radius_disk_fallback``).
    """
    from scipy import ndimage as _ndi

    n = len(spots_df)
    H, W = partner_2d.shape
    rna1_f = np.asarray(rna1_2d, dtype=np.float64)
    part_f = np.asarray(partner_2d, dtype=np.float64)
    ys_arr = np.rint(spots_df["y_px"].astype(float).to_numpy()).astype(np.intp)
    xs_arr = np.rint(spots_df["x_px"].astype(float).to_numpy()).astype(np.intp)
    if "spot_diameter_um" in spots_df.columns:
        diam_um = pd.to_numeric(spots_df["spot_diameter_um"], errors="coerce").to_numpy()
    else:
        diam_um = np.full(n, np.nan, dtype=np.float64)
    vx = max(float(voxel_xy_um), 1e-6)
    default_fwhm_px = float(default_spot_diameter_um) / vx
    struct8 = np.ones((3, 3), dtype=bool)
    out = []

    def _disk_fallback(cy, cx, fwhm_px):
        rad = max(1.0, float(fwhm_px) / 2.0)
        ri = int(max(1, round(rad)))
        yy, xx = np.mgrid[-ri:ri + 1, -ri:ri + 1]
        disk = (yy * yy + xx * xx) <= (rad * rad)
        ay = cy + yy[disk]
        ax = cx + xx[disk]
        inb = (ay >= 0) & (ay < H) & (ax >= 0) & (ax < W)
        if not inb.any():
            return dict(dy=np.empty(0, np.intp), dx=np.empty(0, np.intp),
                        area_px=0.0, partner_mean=float("nan"),
                        method="fitted_radius_disk_fallback")
        ay, ax = ay[inb], ax[inb]
        return dict(dy=(ay - cy).astype(np.intp), dx=(ax - cx).astype(np.intp),
                    area_px=float(ay.size), partner_mean=float(part_f[ay, ax].mean()),
                    method="fitted_radius_disk_fallback")

    for i in range(n):
        cy = int(np.clip(ys_arr[i], 0, H - 1))
        cx = int(np.clip(xs_arr[i], 0, W - 1))
        fwhm_px = diam_um[i] / vx if (np.isfinite(diam_um[i]) and diam_um[i] > 0) else default_fwhm_px
        if not (np.isfinite(fwhm_px) and fwhm_px > 0):
            fwhm_px = default_fwhm_px
        win_half = int(np.clip(round(fwhm_px) + window_pad_px, min_window_half, max_window_half))
        y0, y1 = max(0, cy - win_half), min(H, cy + win_half + 1)
        x0, x1 = max(0, cx - win_half), min(W, cx + win_half + 1)
        if (y1 - y0) < 3 or (x1 - x0) < 3:
            out.append(_disk_fallback(cy, cx, fwhm_px))
            continue
        win = rna1_f[y0:y1, x0:x1]
        sy0, sy1 = max(y0, cy - peak_search_half), min(y1, cy + peak_search_half + 1)
        sx0, sx1 = max(x0, cx - peak_search_half), min(x1, cx + peak_search_half + 1)
        inner = rna1_f[sy0:sy1, sx0:sx1]
        s_off = np.unravel_index(int(np.argmax(inner)), inner.shape)
        seed_y = sy0 + int(s_off[0])
        seed_x = sx0 + int(s_off[1])
        peak = float(rna1_f[seed_y, seed_x])
        bg = float(np.percentile(win, bg_percentile))
        if not (peak > bg):
            out.append(_disk_fallback(cy, cx, fwhm_px))
            continue
        thr = bg + half_max_frac * (peak - bg)
        lbl, _ = _ndi.label(win >= thr, structure=struct8)
        seed_lbl = int(lbl[seed_y - y0, seed_x - x0])
        if seed_lbl == 0:
            out.append(_disk_fallback(cy, cx, fwhm_px))
            continue
        ys_fp, xs_fp = np.where(lbl == seed_lbl)
        abs_y = ys_fp + y0
        abs_x = xs_fp + x0
        out.append(dict(dy=(abs_y - cy).astype(np.intp), dx=(abs_x - cx).astype(np.intp),
                        area_px=float(abs_y.size),
                        partner_mean=float(part_f[abs_y, abs_x].mean()),
                        method="halfmax_connected_component"))
    return out


def footprint_means_at(partner_2d, cys, cxs, dy, dx):
    """Mean partner intensity over one footprint SHAPE translated to each centre.

    Centres and their translated pixels are clipped into the frame, matching the
    engine's clip-to-edge sampling. Returns one mean per centre.
    """
    H, W = partner_2d.shape
    if cys.size == 0 or dy.size == 0:
        return np.empty(0, dtype=np.float64)
    ys = np.clip(cys[:, None] + dy[None, :], 0, H - 1)
    xs = np.clip(cxs[:, None] + dx[None, :], 0, W - 1)
    return partner_2d[ys, xs].mean(axis=1)


def shuffled_footprint_means(partner_2d, nuc_mask, nuc_ys, nuc_xs, dy, dx, n_draws, rng,
                             oversample=3, max_rounds=6):
    """Partner mean with this footprint placed at ``n_draws`` random in-nucleus
    positions, requiring EVERY translated pixel to stay inside the nucleus.

    Returns ``(means, relaxed)``. ``relaxed`` is True when the nucleus cannot
    contain the footprint at enough positions and the containment requirement
    was dropped to centre-in-nucleus with clip-to-edge, which is recorded per
    nucleus rather than hidden.
    """
    H, W = partner_2d.shape
    npix = nuc_ys.size
    if npix == 0 or dy.size == 0:
        return np.empty(0, dtype=np.float64), False
    keep_y, keep_x = [], []
    need = int(n_draws)
    for _ in range(max_rounds):
        if sum(a.size for a in keep_y) >= need:
            break
        k = max(need * oversample, 64)
        idx = rng.integers(0, npix, size=k)
        cy, cx = nuc_ys[idx], nuc_xs[idx]
        ys = cy[:, None] + dy[None, :]
        xs = cx[:, None] + dx[None, :]
        inb = ((ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)).all(axis=1)
        ok = np.zeros(cy.size, dtype=bool)
        if inb.any():
            yi, xi = ys[inb], xs[inb]
            ok[inb] = nuc_mask[yi, xi].all(axis=1)
        if ok.any():
            keep_y.append(cy[ok])
            keep_x.append(cx[ok])
    got = sum(a.size for a in keep_y)
    if got >= need:
        cys = np.concatenate(keep_y)[:need]
        cxs = np.concatenate(keep_x)[:need]
        return footprint_means_at(partner_2d, cys, cxs, dy, dx), False
    idx = rng.integers(0, npix, size=need)
    return footprint_means_at(partner_2d, nuc_ys[idx], nuc_xs[idx], dy, dx), True


# ------------------------------------------------------------------- run i/o
def read_recorded_plane(img,channel,z_1indexed):
    """Select the recorded C/Z on the lazy array before materializing pixels."""
    if not 0 <= channel < img.n_channels:
        raise IndexError(f'Channel {channel} out of range')
    z=max(0,min(img.n_z-1,int(z_1indexed)-1))
    return np.asarray(img.bio.get_image_dask_data('YX',T=0,C=channel,Z=z).compute())


class Run:
    """Read-only view of one completed fishsuite run directory."""

    def __init__(self, run_dir: Path, partner_slot=None, input_dir=None):
        self.dir = Path(run_dir).resolve()
        rc = json.loads((self.dir / "run_config.json").read_text(encoding="utf-8"))
        self.rc = rc
        self.cfg = rc["config_resolved"]
        self.ch = self.cfg["channels"]
        self.mode = self.ch["analysis_mode"]
        if self.mode not in ("rna_rna", "rna_protein"):
            raise SystemExit(f"analysis_mode {self.mode!r} has no partner channel; "
                             "this tool needs rna_rna or rna_protein")
        self.partner_slot = partner_slot or ("antibody" if self.mode == "rna_protein" else "rna2")
        if self.partner_slot not in ("antibody", "rna2"):
            raise SystemExit(f"--partner-slot {self.partner_slot!r} not understood")
        self.partner_key = "protein" if self.partner_slot == "antibody" else "rna2"
        self.partner_spot_channel = self.partner_key
        self.input_dir = Path(input_dir) if input_dir else Path(rc["input_dir"])

        self.summary = pd.read_csv(self.dir / "per_image_summary.csv")
        self.nuclei = pd.read_csv(self.dir / "nuclei_metrics.csv")
        self.thresholds = pd.read_csv(self.dir / "thresholds.csv")
        self.prefix = _compute_common_filename_prefix(
            [Path(s).stem for s in self.summary["image"].astype(str)])

        self.rna_label = str(self.ch.get("rna_label") or "RNA1")
        self.partner_label = str(
            self.ch.get("antibody_label") if self.partner_slot == "antibody"
            else self.ch.get("rna2_label") or "RNA2")
        self.dapi_label = str(self.ch.get("dapi_label") or "DAPI")

        sfx = "rna_protein" if self.partner_slot == "antibody" else "rna1_rna2"
        p_of = "protein" if self.partner_slot == "antibody" else "rna2"
        self.run_cols = dict(
            pearson=f"coloc_pearson_r_{sfx}",
            li_icq=f"coloc_li_icq_{sfx}",
            m1=f"manders_rna1_in_{p_of}",
            m2=f"manders_{p_of}_in_rna1",
            thr_rna1="coloc_mask_thr_rna1",
            thr_partner=f"coloc_mask_thr_{p_of}",
        )
        missing = [c for c in self.run_cols.values() if c not in self.nuclei.columns]
        if missing:
            raise SystemExit(f"nuclei_metrics.csv is missing {missing}")

    def channel_index(self, slot):
        one_ix = bool(self.ch.get("one_indexed", False))
        i = int(self.ch[slot])
        return (i - 1) if (one_ix and i > 0) else i

    def stem_for(self, image_name, condition):
        cond = None if (condition is None or (isinstance(condition, float) and pd.isna(condition))) \
            else str(condition)
        return _stem_with_condition(_simplify_stem(Path(image_name).stem, self.prefix), cond)

    def image_path(self, image_name):
        hits = sorted(self.input_dir.rglob(image_name))
        if len(hits) != 1:
            raise SystemExit(f"{image_name}: expected exactly 1 match under "
                             f"{self.input_dir}, found {len(hits)}")
        return hits[0]

    def planes(self, image_name, z_plane):
        """(dapi, rna1, partner) 2-D planes at the run's own z, exactly as the run
        extracted them: raw channel, no preprocessing, single plane.

        Select C/Z on BioIO's lazy array before computing, so unrelated planes
        are never materialized. Channel routing and the recorded one-based Z
        selection are identical to extract_channel_at_z.
        """
        img = _io.read_image(self.image_path(image_name))
        z = int(z_plane)
        try:
            out = []
            for slot in ("dapi", "rna", self.partner_slot):
                plane = read_recorded_plane(img, self.channel_index(slot), z)
                out.append(np.array(plane, copy=True))
                del plane
                gc.collect()
            vx = float(img.voxel_xy_nm)
        finally:
            try:
                img.bio.__exit__(None, None, None)
            except Exception:
                pass
            del img
            gc.collect()
        return out[0], out[1], out[2], vx

    def label_mask(self, stem):
        p = self.dir / "masks" / f"{stem}__nuclei_label_mask.tif"
        if not p.exists():
            raise SystemExit(f"missing nuclei label mask: {p}")
        from skimage.io import imread
        return imread(str(p))


def resolve_arm(condition, well_map, arm_rules):
    """Arm for one condition. Explicit --arm-map rules win, then a report
    per_well map, then a suffix heuristic (strip a trailing _<n>, take the last
    remaining token). Returns None when nothing matches, which is a hard error
    rather than a silent bin."""
    import fnmatch
    c = str(condition)
    for pat, arm in arm_rules:
        if fnmatch.fnmatch(c, pat) or fnmatch.fnmatch(c.upper(), pat.upper()):
            return arm
    if c in well_map:
        return well_map[c]
    base = re.sub(r"_\d+$", "", c)
    tok = base.split("_")[-1] if "_" in base else base
    return tok or None


def stratum_for(condition):
    """Leading token when a condition looks like <stratum>_<arm>_<rep>, else ''.
    Used only to shade points; never enters a test."""
    c = str(condition)
    base = re.sub(r"_\d+$", "", c)
    parts = base.split("_")
    return parts[0] if len(parts) >= 2 else ""


# ------------------------------------------------------------------ pass one
def compute_tables(run, spots, well_map, arm_rules, seed, costes_fit="tls",
                   anchor="rna1", pair_um=0.3, verbose=True):
    """Per-nucleus pixel + object colocalization for every nucleus in the run."""
    rows = []
    fp_checks = []
    _foci = run.cfg["foci"]
    default_spot_diam_um = 2.0 * float(_foci["bigfish_spot_radius_nm"]) / 1000.0
    _ab_r = (_foci.get("antibody_overrides", {}) or {}).get("bigfish_spot_radius_nm")
    default_partner_diam_um = 2.0 * float(
        _ab_r if _ab_r else _foci["bigfish_spot_radius_nm"]) / 1000.0
    pooled = {}       # line -> [rna1 array, partner array]
    nuc_index = {}    # (image, nucleus_id) -> row index, for later lookups
    fallback_used = 0

    nuc_by_img = {k: v for k, v in run.nuclei.groupby("image")}
    spots_by_img = {k: v for k, v in spots.groupby("image")}

    for img_i, srow in run.summary.reset_index(drop=True).iterrows():
        name = str(srow["image"])
        cond = srow["condition"]
        sec_only = bool(srow["secondary_only"])
        line = SEC_ONLY if sec_only else resolve_arm(cond, well_map, arm_rules)
        if line is None:
            raise SystemExit(
                "condition {!r} does not map to an arm; pass --arm-map "
                "'PATTERN=ARM'".format(cond))
        stem = run.stem_for(name, cond)
        z = int(srow["z_plane"])
        dapi2d, rna2d, par2d, vx_nm = run.planes(name, z)
        if abs(vx_nm - float(srow["voxel_xy_nm"])) > 1e-6:
            raise SystemExit(f"{name}: voxel_xy_nm image={vx_nm} csv={srow['voxel_xy_nm']}")
        labels = run.label_mask(stem)
        if labels.shape != rna2d.shape:
            raise SystemExit(f"{name}: mask {labels.shape} vs plane {rna2d.shape}")
        rna2d = rna2d.astype(np.float64)
        par2d = par2d.astype(np.float64)

        nm = nuc_by_img.get(name)
        if nm is None:
            continue
        sm = spots_by_img.get(name)
        sm_rna = sm[(sm["channel"] == "rna1") & (sm["in_nucleus"].astype(bool))] \
            if sm is not None else None
        sm_par = None
        if sm is not None and anchor in ("partner", "both"):
            sm_par = sm[(sm["channel"] == run.partner_spot_channel)
                        & (sm["in_nucleus"].astype(bool))]

        # EXACT punctum footprints for this image's nuclear rna1 spots. The
        # observed partner-in-footprint mean comes from the run's own
        # qki_at_miat_footprint column; these reconstructions supply only the
        # pixel OFFSETS the shuffle needs, and every one is cross-checked
        # against the run's stored area and mean.
        fps = None
        fp_index = {}
        if sm_rna is not None and len(sm_rna):
            fps = footprint_offsets_for_image(
                rna2d, par2d, sm_rna, float(vx_nm) / 1000.0, default_spot_diam_um)
            fp_area_rec = np.array([f["area_px"] for f in fps], dtype=float)
            fp_mean_rec = np.array([f["partner_mean"] for f in fps], dtype=float)
            fp_area_run = pd.to_numeric(
                sm_rna["miat_footprint_area_px"], errors="coerce").to_numpy(dtype=float)
            fp_mean_run = pd.to_numeric(
                sm_rna["qki_at_miat_footprint"], errors="coerce").to_numpy(dtype=float)
            fp_checks.append(dict(
                image=name, n_spots=len(fps),
                max_abs_area_diff=float(np.nanmax(np.abs(fp_area_rec - fp_area_run))),
                max_abs_mean_diff=float(np.nanmax(np.abs(fp_mean_rec - fp_mean_run))),
                n_disk_fallback=int(sum(1 for f in fps
                                        if f["method"] == "fitted_radius_disk_fallback")),
            ))
            fp_index = {int(k): j for j, k in enumerate(sm_rna.index)}

        # Partner-anchored footprints: built from the PARTNER channel and used to
        # sample rna1. The run stores no equivalent column for these, so unlike the
        # rna1 direction they are a RECONSTRUCTION and are labelled as such.
        fps_par = None
        fp_index_par = {}
        if sm_par is not None and len(sm_par):
            fps_par = footprint_offsets_for_image(
                par2d, rna2d, sm_par, float(vx_nm) / 1000.0, default_partner_diam_um)
            fp_index_par = {int(k): j for j, k in enumerate(sm_par.index)}

        for _, nrow in nm.iterrows():
            nid = int(nrow["nucleus_id"])
            mask = labels == nid
            npix = int(mask.sum())
            r = rna2d[mask]
            a = par2d[mask]
            run_thr_r = float(nrow[run.run_cols["thr_rna1"]])
            run_thr_a = float(nrow[run.run_cols["thr_partner"]])

            c_r, c_a, converged = costes_np(r, a, fit=costes_fit)
            o_r, o_a, o_conv = costes_np(r, a, fit="ols")
            if converged:
                thr_r, thr_a, src = c_r, c_a, "costes"
            else:
                thr_r, thr_a, src = run_thr_r, run_thr_a, "run_median_plus_k_mad"
                fallback_used += 1
            m_cos = pixel_metrics(r, a, thr_r, thr_a)
            m_run = pixel_metrics(r, a, run_thr_r, run_thr_a)

            # ---- object coloc on this nucleus's rna1 nuclear puncta ----------
            # Sampling region is the EXACT punctum footprint, never a disk.
            n_sp = 0
            n_called = 0
            n_called_run = 0
            frac_obs = np.nan
            frac_shuf = np.nan
            frac_obs_run = np.nan
            frac_shuf_run = np.nan
            fp_area_med = np.nan
            n_relaxed = 0
            if sm_rna is not None and fps is not None:
                sub = sm_rna[sm_rna["nucleus_id"].astype(int) == nid]
                n_sp = int(len(sub))
                if n_sp > 0:
                    # OBSERVED: the run's own partner-mean-over-footprint column.
                    obs = pd.to_numeric(sub["qki_at_miat_footprint"],
                                        errors="coerce").to_numpy(dtype=float)
                    areas = pd.to_numeric(sub["miat_footprint_area_px"],
                                          errors="coerce").to_numpy(dtype=float)
                    fp_area_med = float(np.nanmedian(areas))
                    called = obs > thr_a
                    called_run = obs > run_thr_a
                    n_called = int(np.nansum(called))
                    n_called_run = int(np.nansum(called_run))
                    frac_obs = float(np.nanmean(called))
                    frac_obs_run = float(np.nanmean(called_run))
                    ys, xs = np.nonzero(mask)
                    if ys.size:
                        rng = np.random.default_rng([int(seed), int(img_i), nid])
                        draws = np.empty((n_sp, SHUFFLE_DRAWS), dtype=np.float64)
                        for k, ridx in enumerate(sub.index):
                            f = fps[fp_index[int(ridx)]]
                            m, relaxed = shuffled_footprint_means(
                                par2d, mask, ys, xs, f["dy"], f["dx"],
                                SHUFFLE_DRAWS, rng)
                            n_relaxed += int(relaxed)
                            draws[k, :] = m if m.size == SHUFFLE_DRAWS else np.nan
                        frac_shuf = float(np.nanmean(
                            np.nanmean(draws > thr_a, axis=0)))
                        frac_shuf_run = float(np.nanmean(
                            np.nanmean(draws > run_thr_a, axis=0)))

            # ---- PARTNER-anchored: rna1 sampled over each partner punctum's
            # exact footprint. Mirror image of the block above; the observed
            # value is a reconstruction because the run stores no column for it.
            n_sp_p = 0
            frac_obs_p = np.nan
            frac_shuf_p = np.nan
            frac_obs_p_run = np.nan
            frac_shuf_p_run = np.nan
            fp_area_med_p = np.nan
            if sm_par is not None and fps_par is not None:
                subp = sm_par[sm_par["nucleus_id"].astype(int) == nid]
                n_sp_p = int(len(subp))
                if n_sp_p > 0:
                    obs_p = np.array(
                        [fps_par[fp_index_par[int(k)]]["partner_mean"] for k in subp.index],
                        dtype=float)
                    fp_area_med_p = float(np.nanmedian(
                        [fps_par[fp_index_par[int(k)]]["area_px"] for k in subp.index]))
                    frac_obs_p = float(np.nanmean(obs_p > thr_r))
                    frac_obs_p_run = float(np.nanmean(obs_p > run_thr_r))
                    ys, xs = np.nonzero(mask)
                    if ys.size:
                        rngp = np.random.default_rng([int(seed), 7, int(img_i), nid])
                        dr = np.empty((n_sp_p, SHUFFLE_DRAWS), dtype=np.float64)
                        for k, ridx in enumerate(subp.index):
                            f = fps_par[fp_index_par[int(ridx)]]
                            m, _rx = shuffled_footprint_means(
                                rna2d, mask, ys, xs, f["dy"], f["dx"], SHUFFLE_DRAWS, rngp)
                            dr[k, :] = m if m.size == SHUFFLE_DRAWS else np.nan
                        frac_shuf_p = float(np.nanmean(np.nanmean(dr > thr_r, axis=0)))
                        frac_shuf_p_run = float(
                            np.nanmean(np.nanmean(dr > run_thr_r, axis=0)))

            # ---- spot-to-spot pairing within pair_um, both directions.
            # Observed comes from the run's own paired_at_0p3um flag; the shuffle
            # re-draws the OTHER channel's spot positions uniformly in this
            # nucleus and recomputes the same nearest-neighbour rule.
            pair_r_obs = pair_r_shuf = np.nan
            pair_p_obs = pair_p_shuf = np.nan
            if sm_rna is not None:
                subr = sm_rna[sm_rna["nucleus_id"].astype(int) == nid]
                subp2 = (sm_par[sm_par["nucleus_id"].astype(int) == nid]
                         if sm_par is not None else None)
                if len(subr) and "paired_at_0p3um" in subr.columns:
                    pair_r_obs = float(pd.to_numeric(
                        subr["paired_at_0p3um"], errors="coerce").fillna(0).mean())
                if subp2 is not None and len(subp2) and "paired_at_0p3um" in subp2.columns:
                    pair_p_obs = float(pd.to_numeric(
                        subp2["paired_at_0p3um"], errors="coerce").fillna(0).mean())
                if subp2 is not None and len(subr) and len(subp2):
                    ys, xs = np.nonzero(mask)
                    if ys.size:
                        rng2 = np.random.default_rng([int(seed), 11, int(img_i), nid])
                        pr = float(pair_um) / (float(vx_nm) / 1000.0)
                        ry = subr["y_px"].to_numpy(float)
                        rx = subr["x_px"].to_numpy(float)
                        py = subp2["y_px"].to_numpy(float)
                        px = subp2["x_px"].to_numpy(float)
                        n_draw = min(SHUFFLE_DRAWS, 50)
                        fr = np.empty(n_draw)
                        fp_ = np.empty(n_draw)
                        for d in range(n_draw):
                            ip = rng2.integers(0, ys.size, size=py.size)
                            sy, sx = ys[ip].astype(float), xs[ip].astype(float)
                            d2 = ((ry[:, None] - sy[None, :]) ** 2
                                  + (rx[:, None] - sx[None, :]) ** 2)
                            fr[d] = float((d2.min(axis=1) <= pr * pr).mean())
                            ir = rng2.integers(0, ys.size, size=ry.size)
                            qy, qx = ys[ir].astype(float), xs[ir].astype(float)
                            d3 = ((py[:, None] - qy[None, :]) ** 2
                                  + (px[:, None] - qx[None, :]) ** 2)
                            fp_[d] = float((d3.min(axis=1) <= pr * pr).mean())
                        pair_r_shuf = float(fr.mean())
                        pair_p_shuf = float(fp_.mean())

            # ---- pooled nuclear pixels for the cytofluorogram ----------------
            if not sec_only:
                if npix > PER_NUCLEUS_PIXEL_CAP:
                    rng2 = np.random.default_rng([int(seed), 99, int(img_i), nid])
                    sel = rng2.choice(npix, size=PER_NUCLEUS_PIXEL_CAP, replace=False)
                    pr, pa = r[sel], a[sel]
                else:
                    pr, pa = r, a
                buf = pooled.setdefault(line, ([], []))
                buf[0].append(pr.astype(np.float32))
                buf[1].append(pa.astype(np.float32))

            nuc_index[(name, nid)] = len(rows)
            rows.append(dict(
                image=name, stem=stem, condition=cond, line=line,
                stratum=stratum_for(cond),
                secondary_only=sec_only, z_plane=z, nucleus_id=nid,
                n_pix=npix,
                costes_fit=costes_fit,
                costes_thr_rna1=c_r, costes_thr_partner=c_a,
                costes_converged=bool(converged), threshold_source=src,
                costes_ols_thr_rna1=o_r, costes_ols_thr_partner=o_a,
                costes_ols_converged=bool(o_conv),
                thr_rna1_used=thr_r, thr_partner_used=thr_a,
                run_thr_rna1=run_thr_r, run_thr_partner=run_thr_a,
                pearson_r_csp=m_cos["pearson_r"],
                li_icq_csp=m_cos["li_icq"],
                manders_m1_costes=m_cos["manders_m1"],
                manders_m2_costes=m_cos["manders_m2"],
                manders_m1_costes_only=m_cos["manders_m1"] if converged else np.nan,
                manders_m2_costes_only=m_cos["manders_m2"] if converged else np.nan,
                frac_above_costes_rna1=m_cos["frac_above_thr_rna1"],
                frac_above_costes_partner=m_cos["frac_above_thr_partner"],
                manders_m1_runthr=m_run["manders_m1"],
                manders_m2_runthr=m_run["manders_m2"],
                run_pearson_r=float(nrow[run.run_cols["pearson"]]),
                run_li_icq=float(nrow[run.run_cols["li_icq"]]),
                run_manders_m1=float(nrow[run.run_cols["m1"]]),
                run_manders_m2=float(nrow[run.run_cols["m2"]]),
                pearson_abs_diff_vs_run=abs(m_cos["pearson_r"] - float(nrow[run.run_cols["pearson"]])),
                li_icq_abs_diff_vs_run=abs(m_cos["li_icq"] - float(nrow[run.run_cols["li_icq"]])),
                manders_m1_runthr_abs_diff_vs_run=abs(m_run["manders_m1"] - float(nrow[run.run_cols["m1"]])),
                manders_m2_runthr_abs_diff_vs_run=abs(m_run["manders_m2"] - float(nrow[run.run_cols["m2"]])),
                n_rna1_nuclear_puncta=n_sp,
                median_footprint_area_px=fp_area_med,
                n_spots_shuffle_containment_relaxed=n_relaxed,
                n_called_coloc=n_called,
                frac_called_coloc=frac_obs,
                frac_called_coloc_shuffle=frac_shuf,
                frac_called_coloc_minus_shuffle=(frac_obs - frac_shuf)
                if (np.isfinite(frac_obs) and np.isfinite(frac_shuf)) else np.nan,
                n_called_coloc_runthr=n_called_run,
                n_partner_nuclear_puncta=n_sp_p,
                median_partner_footprint_area_px=fp_area_med_p,
                frac_called_coloc_partner=frac_obs_p,
                frac_called_coloc_partner_shuffle=frac_shuf_p,
                frac_called_coloc_partner_minus_shuffle=(frac_obs_p - frac_shuf_p)
                if (np.isfinite(frac_obs_p) and np.isfinite(frac_shuf_p)) else np.nan,
                frac_called_coloc_partner_runthr=frac_obs_p_run,
                frac_called_coloc_partner_shuffle_runthr=frac_shuf_p_run,
                frac_called_coloc_partner_minus_shuffle_runthr=(
                    frac_obs_p_run - frac_shuf_p_run)
                if (np.isfinite(frac_obs_p_run) and np.isfinite(frac_shuf_p_run)) else np.nan,
                paired_frac_rna1_at_partner=pair_r_obs,
                paired_frac_rna1_at_partner_shuffle=pair_r_shuf,
                paired_frac_rna1_at_partner_minus_shuffle=(pair_r_obs - pair_r_shuf)
                if (np.isfinite(pair_r_obs) and np.isfinite(pair_r_shuf)) else np.nan,
                paired_frac_partner_at_rna1=pair_p_obs,
                paired_frac_partner_at_rna1_shuffle=pair_p_shuf,
                paired_frac_partner_at_rna1_minus_shuffle=(pair_p_obs - pair_p_shuf)
                if (np.isfinite(pair_p_obs) and np.isfinite(pair_p_shuf)) else np.nan,
                frac_called_coloc_runthr=frac_obs_run,
                frac_called_coloc_shuffle_runthr=frac_shuf_run,
                frac_called_coloc_minus_shuffle_runthr=(frac_obs_run - frac_shuf_run)
                if (np.isfinite(frac_obs_run) and np.isfinite(frac_shuf_run)) else np.nan,
            ))
        if verbose:
            print(f"  {name}  z={z}  nuclei={len(nm)}  line={line}", flush=True)

    per_nucleus = pd.DataFrame(rows)
    pooled_out = {}
    for ln, (rl, al) in pooled.items():
        pooled_out[ln] = (np.concatenate(rl), np.concatenate(al))
    return per_nucleus, pooled_out, nuc_index, fallback_used, pd.DataFrame(fp_checks)


# ------------------------------------------------------------- rollups + test
PIXEL_ENDPOINTS = [
    ("pearson_r_csp", "Pearson r"),
    ("manders_m1_costes", "Manders M1"),
    ("manders_m2_costes", "Manders M2"),
    ("li_icq_csp", "Li ICQ"),
]
# What csp01 PLOTS. The Manders panels use the run's single batch threshold, not
# the Costes-with-fallback column: with Costes converging in a minority of nuclei
# that column is a mixture of two conventions and its arm means track the
# convergence fraction rather than the images. The Costes variants are computed,
# tabulated and plotted in csp01b.
CSP01_PANELS = [
    ("pearson_r_csp", "Pearson r"),
    ("li_icq_csp", "Li ICQ"),
    ("manders_m1_runthr", "Manders M1 (run batch threshold)"),
    ("manders_m2_runthr", "Manders M2 (run batch threshold)"),
]
CSP01B_PANELS = [
    ("manders_m1_runthr", "M1, run batch threshold" + chr(10) +
     "(all nuclei, one convention)"),
    ("manders_m1_costes_only", "M1, Costes threshold" + chr(10) +
     "(converged nuclei only)"),
    ("manders_m1_costes", "M1, Costes with fallback" + chr(10) +
     "(MIXTURE of both)"),
]
PIXEL_SENSITIVITY = [
    ("manders_m1_runthr", "Manders M1 at the run's batch threshold"),
    ("manders_m2_runthr", "Manders M2 at the run's batch threshold"),
    ("manders_m1_costes_only", "Manders M1, Costes-converged nuclei only"),
    ("manders_m2_costes_only", "Manders M2, Costes-converged nuclei only"),
]
OBJECT_ENDPOINTS = [
    ("frac_called_coloc", "object coloc fraction (observed)"),
    ("frac_called_coloc_shuffle", "object coloc fraction (shuffle control)"),
    ("frac_called_coloc_minus_shuffle", "object coloc fraction (observed - shuffle)"),
]
PARTNER_ANCHORED = [
    ("frac_called_coloc_partner_runthr",
     "partner-anchored coloc fraction (observed, batch threshold)"),
    ("frac_called_coloc_partner_shuffle_runthr",
     "partner-anchored coloc fraction (shuffle, batch threshold)"),
    ("frac_called_coloc_partner_minus_shuffle_runthr",
     "partner-anchored coloc fraction (observed - shuffle, batch threshold)"),
    ("frac_called_coloc_partner", "partner-anchored coloc fraction (Costes with fallback)"),
    ("frac_called_coloc_partner_shuffle",
     "partner-anchored coloc fraction (shuffle, Costes with fallback)"),
]
# Object-fraction columns are on 0-1 and are counts of OBJECTS, not pixel
# intensity-correlation quotients. Both spellings must be listed: the paired
# columns are named `paired_frac_*`, so a `frac_`-only test missed all six of
# them and they inherited the pixel fallback, `role = pixel, descriptive` and
# `unit = ICQ -0.5..+0.5`. The values were always correct; only the two label
# columns were wrong, and one of the six is a delivered headline.
OBJECT_FRACTION_PREFIXES = ("frac_", "paired_frac_")

SPOT_PAIRING = [
    ("paired_frac_rna1_at_partner", "fraction of rna1 puncta paired to a partner punctum"),
    ("paired_frac_rna1_at_partner_shuffle", "same, partner positions shuffled"),
    ("paired_frac_rna1_at_partner_minus_shuffle", "same, observed - shuffle"),
    ("paired_frac_partner_at_rna1", "fraction of partner puncta paired to an rna1 punctum"),
    ("paired_frac_partner_at_rna1_shuffle", "same, rna1 positions shuffled"),
    ("paired_frac_partner_at_rna1_minus_shuffle", "same, observed - shuffle"),
]
OBJECT_SENSITIVITY = [
    ("frac_called_coloc_runthr",
     "object coloc fraction, run batch threshold for every nucleus (observed)"),
    ("frac_called_coloc_shuffle_runthr",
     "object coloc fraction, run batch threshold (shuffle control)"),
    ("frac_called_coloc_minus_shuffle_runthr",
     "object coloc fraction, run batch threshold (observed - shuffle)"),
]
ALL_ENDPOINTS = (PIXEL_ENDPOINTS + PIXEL_SENSITIVITY + OBJECT_ENDPOINTS
                 + OBJECT_SENSITIVITY + PARTNER_ANCHORED + SPOT_PAIRING)


def rollup(per_nucleus):
    """nucleus -> FOV mean -> well mean, the convention used by the run's own
    report (per_well.csv column ``well_mean_of_fov_values``)."""
    cols = [c for c, _ in ALL_ENDPOINTS if c in per_nucleus.columns]
    fov = (per_nucleus
           .groupby(["line", "condition", "secondary_only", "image"], as_index=False)
           .agg(n_nuclei=("nucleus_id", "size"),
                **{c: (c, "mean") for c in cols}))
    well = (fov.groupby(["line", "condition", "secondary_only"], as_index=False)
            .agg(n_fov=("image", "size"), n_nuclei=("n_nuclei", "sum"),
                 **{c: (c, "mean") for c in cols}))
    sd = (fov.groupby(["line", "condition", "secondary_only"])[cols]
          .std(ddof=1).add_suffix("_sd_across_fov").reset_index())
    well = well.merge(sd, on=["line", "condition", "secondary_only"], how="left")
    well = well.rename(columns={"condition": "well_id"})
    fov = fov.rename(columns={"condition": "well_id"})
    return fov, well


def contrasts_table(well, per_nucleus, mde, primary_col, secondary_col):
    bio = well[~well["secondary_only"]]
    rows = []
    for col, label in ALL_ENDPOINTS:
        a = bio.loc[bio["line"] == ARMS[1], col].to_numpy(dtype=float)
        b = bio.loc[bio["line"] == ARMS[0], col].to_numpy(dtype=float)
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        primary = col == primary_col
        sens = col in dict(PIXEL_SENSITIVITY) or col in dict(OBJECT_SENSITIVITY)
        twin = col == secondary_col
        w = welch(a, b) if primary else dict(
            n_test=len(a), n_ref=len(b),
            mean_test=float(a.mean()) if len(a) else np.nan,
            mean_ref=float(b.mean()) if len(b) else np.nan,
            diff=(float(a.mean() - b.mean()) if len(a) and len(b) else np.nan),
            ci_low=np.nan, ci_high=np.nan, hedges_g=hedges_g(a, b),
            t=np.nan, df=np.nan, p_welch=np.nan, welch_note="")
        v = per_nucleus.loc[~per_nucleus["secondary_only"], col].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        rows.append(dict(
            endpoint=col, label=label,
            role="PRIMARY object endpoint, tested" if primary
            else ("SENSITIVITY twin of the primary, descriptive" if twin
                  else "SENSITIVITY, descriptive" if sens
                  else ("object, descriptive" if col.startswith(OBJECT_FRACTION_PREFIXES)
                        else "pixel, descriptive")),
            unit="fraction 0-1"
                 if col.startswith(OBJECT_FRACTION_PREFIXES + ("manders",)) else
                 ("r" if col.startswith("pearson") else "ICQ -0.5..+0.5"),
            arm_test=ARMS[1], arm_ref=ARMS[0],
            n_wells_test=w["n_test"], n_wells_ref=w["n_ref"],
            well_mean_test=w["mean_test"], well_mean_ref=w["mean_ref"],
            diff_test_minus_ref=w["diff"], ci95_low=w["ci_low"], ci95_high=w["ci_high"],
            hedges_g=w["hedges_g"], t=w["t"], df=w["df"], p_welch=w["p_welch"],
            stars=stars(w["p_welch"]) if primary else "n/a (not tested)",
            mde_hedges_g_alpha_0p05_power_0p80=mde,
            n_nuclei_biological=int(v.size),
            nucleus_pooled_mean=float(v.mean()) if v.size else np.nan,
            zero_variance_arm=bool(
                len(a) > 1 and len(b) > 1
                and (float(np.var(a, ddof=1)) == 0 or float(np.var(b, ddof=1)) == 0)),
            note=w["welch_note"] or ("" if primary else
                                     "descriptive only; the panel tests one object endpoint"),
        ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- rendering
def _lut(gray, name, floor, ceil):
    w = _out.lut_name_to_weights(name)
    return _out.apply_lut(gray, *w, floor=floor, ceil=ceil)


def merged_rgb(dapi, rna, partner, win, luts):
    layers = [_lut(dapi, luts["dapi"], *win["dapi"]),
              _lut(rna, luts["rna"], *win["rna"]),
              _lut(partner, luts["partner"], *win["partner"])]
    return _out._to_uint8(_out.merge_rgb_additive(layers))


def gray_rgb(plane, win):
    return _out._to_uint8(_lut(plane, "gray", *win))


def square_crop(shape, cy, cx, half):
    H, W = shape
    half = int(min(half, H // 2, W // 2))
    y0 = int(np.clip(cy - half, 0, H - 2 * half))
    x0 = int(np.clip(cx - half, 0, W - 2 * half))
    return y0, y0 + 2 * half, x0, x0 + 2 * half


# --------------------------------------------------------------- line profile
def chord_through(mask, cy, cx, orientation):
    """Endpoints of the chord through (cy, cx) along the region principal axis.

    skimage's ``orientation`` is the angle between the row axis and the major
    axis, so the unit direction is (dy, dx) = (-sin theta, cos theta). Walk
    outward in both directions in 0.5-px steps and keep the last in-mask point.
    """
    H, W = mask.shape
    ddy, ddx = -math.sin(orientation), math.cos(orientation)
    if not mask[int(round(cy)), int(round(cx))]:
        ys, xs = np.nonzero(mask)
        k = int(np.argmin((ys - cy) ** 2 + (xs - cx) ** 2))
        cy, cx = float(ys[k]), float(xs[k])
    ends = []
    for sgn in (+1.0, -1.0):
        last = (cy, cx)
        t = 0.0
        while True:
            t += 0.5
            y = cy + sgn * ddy * t
            x = cx + sgn * ddx * t
            iy, ix = int(round(y)), int(round(x))
            if iy < 0 or ix < 0 or iy >= H or ix >= W or not mask[iy, ix]:
                break
            last = (y, x)
        ends.append(last)
    return ends[0], ends[1]


def sample_profile(plane, src, dst):
    from skimage.measure import profile_line
    return profile_line(plane.astype(np.float64), src, dst, linewidth=1, order=1,
                        mode="constant", reduce_func=None).ravel()


# ------------------------------------------------------------------- figures
def save(fig, out_dir, stem, manifest, description, source, gate, n, test):
    png = out_dir / f"{stem}.png"
    svg = out_dir / f"{stem}.svg"
    fig.savefig(png, dpi=600)
    fig.savefig(svg)
    plt.close(fig)
    manifest.append(dict(figure=stem, png=str(png), svg=str(svg),
                         description=description, source=source,
                         gate=gate, n=n, test=test))
    print(f"  wrote {png.name} + .svg", flush=True)


def superplot_axes(ax, per_nucleus, well, col, label, title):
    lines = list(ARMS)
    for xi, ln in enumerate(lines):
        sub = per_nucleus[(~per_nucleus["secondary_only"]) & (per_nucleus["line"] == ln)]
        wells = sorted(sub["condition"].unique())
        nwell = max(1, len(wells))
        for wi, wname in enumerate(wells):
            v = sub.loc[sub["condition"] == wname, col].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            if not v.size:
                continue
            off = -0.24 + 0.48 * (wi / max(1, nwell - 1)) if nwell > 1 else 0.0
            jit = np.random.default_rng([0, xi, wi]).uniform(-0.06, 0.06, v.size)
            ax.scatter(np.full(v.size, xi + off) + jit, v, s=5.0,
                       facecolor=shade(arm_color(ln), 0.32 + 0.28 * wi),
                       edgecolor="none", alpha=0.55, zorder=2, rasterized=True)
        wm = well[(~well["secondary_only"]) & (well["line"] == ln)][col].to_numpy(dtype=float)
        wm = wm[np.isfinite(wm)]
        nwm = max(1, wm.size)
        for wi, m in enumerate(wm):
            off = -0.24 + 0.48 * (wi / max(1, nwm - 1)) if nwm > 1 else 0.0
            ax.scatter([xi + off], [m], s=52, marker="D",
                       facecolor=arm_color(ln), edgecolor="black", linewidth=0.7, zorder=4)
        if wm.size:
            m, lo, hi, _ = mean_ci(wm)
            ax.hlines(m, xi - 0.36, xi + 0.36, color="black", linewidth=1.4, zorder=5)
            if np.isfinite(lo):
                ax.vlines(xi + 0.42, lo, hi, color="black", linewidth=1.0, zorder=5)
                ax.hlines([lo, hi], xi + 0.38, xi + 0.46, color="black", linewidth=1.0, zorder=5)
    ax.set_xticks(range(len(lines)))
    ax.set_xticklabels(lines)
    ax.set_xlim(-0.62, len(lines) - 0.38)
    ax.set_ylabel(label)
    ax.set_title(title, fontsize=8.5)


def fig_pearson_manders(per_nucleus, well, ctx, out_dir, manifest):
    fig, axes = plt.subplots(1, 4, figsize=(11.6, 5.8))
    for ax, (col, label) in zip(axes, CSP01_PANELS):
        superplot_axes(ax, per_nucleus, well, col, label, label)
    bio = per_nucleus[~per_nucleus["secondary_only"]]
    n_nuc = len(bio)
    hb = stamp_head(fig, "Per-nucleus pixel colocalization, "
                    f"{ctx['rna_label']} x {ctx['partner_label']}",
                    "small dots = nuclei (shaded by well); diamonds = well means; "
                    "black bar = mean of the 3 well means with its 95% CI",
                    ctx["filt"], y=0.975)
    foot = (
        "Pixels: nuclear mask only, single z plane, the run's own autofocus plane. "
        "Pearson r and Li ICQ are threshold-free and reproduce the run's own columns "
        "exactly. Manders M1 = fraction of "
        f"{ctx['rna_label']} intensity in pixels where {ctx['partner_label']} >= its "
        f"threshold; M2 = fraction of {ctx['partner_label']} intensity where "
        f"{ctx['rna_label']} >= its threshold. "
        f"n = {n_nuc} biological nuclei in 6 wells "
        f"({ctx['arm_n_text']}). "
        "DESCRIPTIVE ONLY: no test is run on the pixel metrics; "
        "the panel's one tested endpoint is the object coloc fraction (csp04). "
        "THRESHOLD CHOICE: the Manders panels use the run's own batch median + 2.5 MAD "
        f"threshold for every nucleus. Costes ({ctx['costes_fit'].upper()} fit) converged "
        f"for only {ctx['costes_pct']:.1f}% of biological nuclei, because inside a nuclear "
        "mask both channels share the same intensity gradient and never become "
        "uncorrelated, so a Costes-with-fallback column mixes two conventions and its arm "
        "means track the convergence fraction rather than the images. That mixed column is "
        "computed and tabulated (manders_m*_costes) and is shown against the alternatives "
        "in csp01b; it is deliberately not plotted here. "
        f"Secondary-only fields are excluded here (n = {ctx['n_sec_nuc']} nuclei). "
        f"seed {ctx['seed']}.")
    fig.subplots_adjust(top=hb - 0.058, bottom=foot_bottom(fig, foot, 5.4),
                        left=0.055, right=0.985, wspace=0.36)
    stamp_foot(fig, foot, size=5.4)
    save(fig, out_dir, "csp01_pearson_manders_superplot", manifest,
         "SuperPlot of the four per-nucleus pixel colocalization coefficients, "
         + " vs ".join(ARMS) + ".",
         "coloc_standard_per_nucleus.csv / coloc_standard_per_well.csv",
         "nuclear mask, biological wells only; Manders at the run batch threshold, "
         "Pearson and Li ICQ threshold-free",
         f"{n_nuc} nuclei, 6 wells", "none (descriptive)")


def fig_manders_threshold_sensitivity(per_nucleus, well, ctx, out_dir, manifest):
    """The same Manders M1 under three threshold conventions, so the reader can
    see that the Costes-with-fallback column is a mixture and not a measurement."""
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 5.4), sharey=True)
    for ax, (col, label) in zip(axes, CSP01B_PANELS):
        superplot_axes(ax, per_nucleus, well, col, "Manders M1", label)
        n = int(per_nucleus.loc[~per_nucleus["secondary_only"], col].notna().sum())
        ax.text(0.5, 0.015, f"n = {n} nuclei", transform=ax.transAxes, ha="center",
                va="bottom", fontsize=6.4, color="#444444")
    axes[0].set_ylim(-0.04, 1.06)
    hb = stamp_head(fig, "Manders M1 under three threshold conventions",
                    "the same pixels and the same formula; only the threshold changes",
                    ctx["filt"], wrap=104, y=0.975)
    foot = (
        "LEFT: the run's own batch median + 2.5 MAD threshold applied to every nucleus - one "
        "convention, every nucleus, so arm means are comparable. MIDDLE: the per-nucleus "
        "Costes threshold, restricted to the nuclei where Costes converged - a Costes "
        "threshold sits far below the batch threshold, so almost all intensity falls above it "
        "and M1 saturates near 1. RIGHT: Costes where it converged and the batch threshold "
        "everywhere else - the column the panel was specified to produce. Its arm mean is a "
        "weighted average of the left and middle values with the weight set by the "
        f"convergence fraction ({ctx['costes_pct']:.1f}% of biological nuclei), so it moves "
        "when convergence moves and should not be read as a measure of colocalization. This "
        "figure exists so that is visible rather than buried. Descriptive; no test.")
    fig.subplots_adjust(top=hb - 0.090, bottom=foot_bottom(fig, foot, 5.4),
                        left=0.075, right=0.985, wspace=0.12)
    stamp_foot(fig, foot, size=5.4)
    save(fig, out_dir, "csp01b_manders_threshold_sensitivity", manifest,
         "Manders M1 under the run batch threshold, the Costes threshold on converged "
         "nuclei, and the Costes-with-fallback mixture.",
         "coloc_standard_per_nucleus.csv", "nuclear mask, biological wells only",
         f"{ctx['n_bio_nuc']} nuclei, 6 wells", "none (descriptive)")


def fig_cytofluorogram(pooled, ctx, out_dir, manifest, line, tag):
    r, a = pooled[line]
    rng = np.random.default_rng([ctx["seed"], 7])
    if r.size > CYTO_MAX_PIXELS:
        sel = rng.choice(r.size, size=CYTO_MAX_PIXELS, replace=False)
        r, a = r[sel], a[sel]
    r = r.astype(np.float64)
    a = a.astype(np.float64)
    c_r, c_a, conv = costes_np(r, a)
    pr = float(np.corrcoef(r, a)[0, 1])
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    hb = ax.hexbin(r, a, gridsize=110, bins="log", mincnt=1, cmap="magma_r",
                   linewidths=0.0)
    cb = fig.colorbar(hb, ax=ax, pad=0.02)
    cb.set_label("pixels per hexagon (log10 scale)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    if np.isfinite(c_r):
        ax.axvline(c_r, color=OKABE["blue"], linewidth=1.2, linestyle="--")
        ax.axhline(c_a, color=OKABE["green"], linewidth=1.2, linestyle="--")
        ax.text(0.98, 0.04, f"Costes T({ctx['rna_label']}) = {c_r:.0f}\n"
                            f"Costes T({ctx['partner_label']}) = {c_a:.0f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6.6)
    ax.text(0.03, 0.96, f"Pearson r = {pr:.3f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=8, fontweight="bold")
    ax.set_xlabel(f"{ctx['rna_label']} intensity (a.u.)")
    ax.set_ylabel(f"{ctx['partner_label']} intensity (a.u.)")
    hb = stamp_head(fig, f"Cytofluorogram - {line}",
                    "pooled nuclear pixels, log-density hexbin", ctx["filt"], wrap=72,
                    y=0.975)
    foot = (
        f"Pooled nuclear pixels from all {line} biological nuclei, subsampled to at most "
        f"{CYTO_MAX_PIXELS:,} pixels (seed {ctx['seed']}); {r.size:,} plotted. "
        + ("Dashed lines are the Costes thresholds computed on the POOLED sample, not the "
           "per-nucleus thresholds used in csp01. "
           if conv else
           "NO threshold lines are drawn: Costes did NOT converge on the pooled sample. The "
           "below-threshold pixels stay positively correlated all the way down, which is the "
           "same reason it fails on most single nuclei - inside a nuclear mask both channels "
           "carry one shared intensity gradient and there is no uncorrelated-background "
           "regime for Costes to find. The elongated single lobe below is that gradient. ")
        + "Pearson r is on the plotted pixels. Descriptive; no test.")
    fig.subplots_adjust(top=hb - 0.030, bottom=foot_bottom(fig, foot, 5.4, pad=0.075),
                        left=0.140, right=0.985)
    stamp_foot(fig, foot, size=5.4)
    save(fig, out_dir, f"csp02_cytofluorogram_{tag}", manifest,
         f"Cytofluorogram of pooled {line} nuclear pixels with pooled Costes thresholds.",
         "run image planes at the run's z, nuclear label masks",
         f"nuclear mask, {line} biological nuclei", f"{r.size} pixels", "none (descriptive)")
    return dict(line=line, n_pixels_plotted=int(r.size), pearson_r=pr,
                costes_thr_rna1=c_r, costes_thr_partner=c_a, costes_converged=bool(conv))


def fig_line_profiles(profiles, ctx, out_dir, manifest):
    n = len(profiles)
    fig, axes = plt.subplots(n, 2, figsize=(9.4, 1.72 * n),
                             gridspec_kw={"width_ratios": [1.0, 2.0]})
    if n == 1:
        axes = np.array([axes])
    for i, p in enumerate(profiles):
        axc, axp = axes[i, 0], axes[i, 1]
        axc.imshow(p["crop_rgb"], interpolation="nearest")
        axc.plot([p["src_x"], p["dst_x"]], [p["src_y"], p["dst_y"]],
                 color="white", linewidth=1.1, solid_capstyle="butt")
        axc.scatter([p["src_x"], p["dst_x"]], [p["src_y"], p["dst_y"]], s=8,
                    facecolor="white", edgecolor="black", linewidth=0.4, zorder=3)
        axc.set_xticks([])
        axc.set_yticks([])
        for s in axc.spines.values():
            s.set_visible(False)
        axc.set_ylabel(f"{p['line']} {p['well_id']}\nnucleus {p['nucleus_id']}",
                       fontsize=6.6, rotation=0, ha="right", va="center", labelpad=26)
        d = p["dist_um"]
        for key, lab, lut in (("rna1", ctx["rna_label"], ctx["luts"]["rna"]),
                              ("partner", ctx["partner_label"], ctx["luts"]["partner"]),
                              ("dapi", ctx["dapi_label"], ctx["luts"]["dapi"])):
            col = tuple(_out.lut_name_to_weights(lut))
            col = tuple(min(0.92, c) for c in col) if lut == "yellow" else col
            axp.plot(d, p[f"{key}_norm"], color=col, linewidth=1.15,
                     label=f"{lab} ({lut})")
        axp.set_ylim(-0.04, 1.06)
        axp.set_ylabel("normalized 0-1", fontsize=6.8)
        axr = axp.twinx()
        axr.set_ylim(0, 1)
        axr.set_yticks([0, 0.5, 1.0])
        axr.set_yticklabels([f"{p['raw_lo_rna1']:.0f}",
                             f"{0.5 * (p['raw_lo_rna1'] + p['raw_hi_rna1']):.0f}",
                             f"{p['raw_hi_rna1']:.0f}"], fontsize=6.0,
                            color=tuple(_out.lut_name_to_weights(ctx["luts"]["rna"])))
        axr.set_ylabel(f"{ctx['rna_label']} raw (a.u.)", fontsize=6.2)
        axr.spines["right"].set_visible(True)
        axr.spines["top"].set_visible(False)
        axp.tick_params(labelsize=6.6)
        if i == 0:
            axp.legend(fontsize=5.8, frameon=False, ncol=3, loc="upper right")
        if i == n - 1:
            axp.set_xlabel("distance along the principal axis (um)", fontsize=7.2)
        else:
            axp.set_xticklabels([])
    hb = stamp_head(fig, "Intensity line profiles across representative nuclei",
                    "left: 1:1 crop with the sampled line drawn; right: each channel in "
                    "its own LUT colour", ctx["filt"], y=0.988)
    foot = (
        "One nucleus per well: the nucleus whose nuclear "
        f"{ctx['rna_label']} punctum count is nearest that well's median. "
        "The line is the chord through the nucleus centroid along the region "
        "principal axis (skimage regionprops orientation), sampled with bilinear "
        "interpolation at 1 px spacing on the run's own z plane. Each trace is "
        "min-max normalized over the line so the three channels are comparable; the "
        f"right-hand axis carries the raw {ctx['rna_label']} range for that line. "
        f"Crop display window: {ctx['rna_label']} {ctx['win']['rna'][0]:.0f}-"
        f"{ctx['win']['rna'][1]:.0f}, {ctx['partner_label']} "
        f"{ctx['win']['partner'][0]:.0f}-{ctx['win']['partner'][1]:.0f}, "
        f"{ctx['dapi_label']} {ctx['win']['dapi'][0]:.0f}-{ctx['win']['dapi'][1]:.0f} "
        "a.u. - DISPLAY ONLY, no measurement uses it. Descriptive; no test.")
    fig.subplots_adjust(top=hb - 0.012, bottom=foot_bottom(fig, foot, 5.2),
                        left=0.115, right=0.925, hspace=0.28)
    stamp_foot(fig, foot, size=5.2)
    save(fig, out_dir, "csp03_line_profiles", manifest,
         "Line profiles through representative nuclei, one per well, each channel in its LUT colour.",
         "run image planes at the run's z, nuclei label masks, spot_metrics.csv",
         "nuclear mask, one representative nucleus per well",
         f"{n} nuclei ({n // 2} per arm)", "none (descriptive)")


def fig_object_fraction(well, contrasts, ctx, out_dir, manifest):
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 5.6), sharey=False)
    bio = well[~well["secondary_only"]]
    variants = [
        (ctx["primary_obs"], ctx["primary_shuf"],
         "A  " + ctx["primary_thr_label"] + chr(10) + "(PRIMARY, TESTED)"),
        (ctx["secondary_obs"], ctx["secondary_shuf"],
         "B  " + ctx["secondary_thr_label"] + chr(10) + "(sensitivity, not tested)"),
    ]
    row = contrasts[contrasts["endpoint"] == ctx["primary_obs"]].iloc[0]
    row_s = contrasts[contrasts["endpoint"] == ctx["secondary_obs"]].iloc[0]

    def _gap(obs_col, shuf_col):
        v = (bio[obs_col].to_numpy(dtype=float)
             - bio[shuf_col].to_numpy(dtype=float))
        v = v[np.isfinite(v)]
        return float(v.mean()) if v.size else float("nan")

    gap_a = _gap(ctx["primary_obs"], ctx["primary_shuf"])
    gap_b = _gap(ctx["secondary_obs"], ctx["secondary_shuf"])
    for ax, (obs_col, shuf_col, title) in zip(axes, variants):
        ytop = 0.0
        for xi, ln in enumerate(ARMS):
            x = float(xi)
            sub = bio[bio["line"] == ln].sort_values("well_id")
            for wi, (_, r) in enumerate(sub.iterrows()):
                ax.plot([x - 0.16, x + 0.16], [r[obs_col], r[shuf_col]],
                        color="#BBBBBB", linewidth=0.7, zorder=1)
                ax.scatter([x - 0.16], [r[obs_col]], s=58, marker="D",
                           facecolor=shade(arm_color(ln), 0.35 + 0.55 * wi / max(1, len(sub) - 1)),
                           edgecolor="black", linewidth=0.7, zorder=3)
                ax.scatter([x + 0.16], [r[shuf_col]], s=48, marker="o",
                           facecolor="white", edgecolor=arm_color(ln),
                           linewidth=1.1, zorder=3)
            for dx, col in ((-0.16, obs_col), (0.16, shuf_col)):
                v = sub[col].to_numpy(dtype=float)
                v = v[np.isfinite(v)]
                if v.size:
                    m, lo, hi, _ = mean_ci(v)
                    ax.hlines(m, x + dx - 0.10, x + dx + 0.10, color="black",
                              linewidth=1.4, zorder=4)
                    if np.isfinite(lo):
                        ax.vlines(x + dx, lo, hi, color="black", linewidth=0.9, zorder=2)
                        ytop = max(ytop, hi)
                    ytop = max(ytop, float(v.max()))
        ybar = ytop * 1.09
        is_primary = obs_col == ctx["primary_obs"]
        star = stars(row["p_welch"]) if is_primary else "not tested"
        xr = float(len(ARMS) - 1)
        ax.plot([-0.16, -0.16, xr - 0.16, xr - 0.16],
                [ybar * 0.975, ybar, ybar, ybar * 0.975], color="black", linewidth=0.9)
        ax.text((xr - 0.32) / 2.0, ybar * 1.012, star, ha="center", va="bottom",
                fontsize=10 if is_primary else 7,
                fontweight="bold" if is_primary else "normal")
        ax.set_xticks([float(i) for i in range(len(ARMS))])
        ax.set_xticklabels(list(ARMS))
        ax.set_xlim(-0.55, len(ARMS) - 0.45)
        ax.set_ylim(0, ybar * 1.24)
        ax.set_title(title, fontsize=8.2)
    axes[0].set_ylabel(f"fraction of nuclear {ctx['rna_label']} puncta called colocalized")
    axes[0].legend(handles=[
        Line2D([], [], marker="D", linestyle="", markersize=7, markerfacecolor="#888888",
               markeredgecolor="black", label="observed (per well)"),
        Line2D([], [], marker="o", linestyle="", markersize=7, markerfacecolor="white",
               markeredgecolor="#888888", label="within-nucleus shuffle control"),
    ], fontsize=6.6, frameon=False, loc="lower center", ncol=1,
        bbox_to_anchor=(0.5, -0.008))
    hb = stamp_head(fig, "Object-based colocalization of "
                    f"{ctx['rna_label']} puncta with {ctx['partner_label']}",
                    "one diamond per well (observed) paired with its shuffle control; "
                    "black bar = mean of 3 wells with its 95% CI", ctx["filt"], wrap=104,
                    y=0.975)
    foot = (
        f"CALLED COLOCALIZED: the mean {ctx['partner_label']} intensity over the punctum's "
        "EXACT footprint exceeds that nucleus's partner threshold. The footprint is the "
        f"connected component of {ctx['rna_label']} pixels at or above the local half-max "
        "(background-subtracted FWHM cut) containing the spot's peak pixel, so it scales "
        "with the punctum instead of being a fixed disk (median "
        f"{ctx['fp_area_median']:.0f} px, IQR {ctx['fp_area_q1']:.0f}-{ctx['fp_area_q3']:.0f}, "
        f"range {ctx['fp_area_min']:.0f}-{ctx['fp_area_max']:.0f}). The observed value is the "
        "run's own qki_at_miat_footprint column, not a recomputation. "
        "SHUFFLE CONTROL: the SAME footprint shape translated to random in-nucleus positions, "
        "every translated pixel required to stay inside the nucleus, "
        f"{SHUFFLE_DRAWS} draws per punctum, seed {ctx['seed']}, averaged. "
        "Rollup: nucleus -> FoV mean -> well mean, matching the run report's per_well "
        "convention. "
        f"PANEL A uses the {ctx['primary_thr_label']} and carries the panel's ONE test; "
        f"that choice was made {ctx['rule_why']}. Welch t on "
        f"{int(row['n_wells_ref'])} vs {int(row['n_wells_test'])} well means of the "
        f"observed fraction: p = {fmt_p(row['p_welch'])}, "
        f"Hedges g = {row['hedges_g']:.3g}, diff ({ARMS[1]} - {ARMS[0]}) = "
        f"{row['diff_test_minus_ref']:.4g} [95% CI {row['ci95_low']:.4g}, "
        f"{row['ci95_high']:.4g}]"
        + (". CAUTION: " + str(row["note"]) if str(row.get("note", "")).strip()
           and str(row.get("note", "")).strip().lower() != "nan" else ".")
        + " No multiplicity correction is applied because only one "
        "endpoint is tested. MDE at alpha 0.05 and 80% power is Hedges g = "
        f"{row['mde_hedges_g_alpha_0p05_power_0p80']:.2f}, so anything smaller is not "
        "detectable in this design and 'ns' here does not mean 'no difference'. "
        f"PANEL B repeats the calculation with the {ctx['secondary_thr_label']} and is NOT "
        f"tested (Hedges g {row_s['hedges_g']:.3g}). Costes converged in "
        f"{ctx['costes_pct']:.1f}% of biological nuclei. Read the two together: a conclusion "
        "that holds in A but not in B is a threshold artifact, not biology. "
        f"Observed minus shuffle, averaged over arms: panel A {gap_a:+.3f}, "
        f"panel B {gap_b:+.3f}; the smaller that gap, the more of the call rate is "
        f"explained by the nucleus's overall {ctx['partner_label']} level rather than by "
        "where the puncta sit. "
        f"n = {ctx['n_bio_nuc_with_puncta']} biological nuclei carrying at least one nuclear "
        f"{ctx['rna_label']} punctum. Stars: * p<0.05, ** p<0.01, *** p<0.001, **** p<1e-4, "
        "ns otherwise.")
    fig.subplots_adjust(top=hb - 0.092, bottom=foot_bottom(fig, foot, 5.2),
                        left=0.095, right=0.985, wspace=0.20)
    stamp_foot(fig, foot, size=5.2)
    save(fig, out_dir, "csp04_object_coloc_fraction", manifest,
         "Per-well object colocalization fraction with its within-nucleus shuffle "
         "control, " + " vs ".join(ARMS) + ".",
         "coloc_standard_per_well.csv / coloc_standard_contrasts",
         "nuclear rna1 puncta, biological wells only; panel A Costes-with-fallback "
         "threshold, panel B the run batch threshold",
         f"3 vs 3 wells, {ctx['n_bio_nuc_with_puncta']} nuclei",
         f"Welch t on well means, p = {fmt_p(row['p_welch'])}")


def fig_overlays(fields, ctx, out_dir, manifest):
    n = len(fields)
    ncol = 3
    nrow = int(math.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 3.62 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.axis("off")
    for ax, f in zip(axes, fields):
        ax.imshow(f["field_rgb"], interpolation="nearest")
        for (yy, xx) in f["outlines"]:
            ax.plot(xx, yy, color="#7A7A7A", linewidth=0.35, zorder=2)
        for (y, x, called) in f["puncta"]:
            ax.add_patch(Circle((x, y), radius=9.0, fill=False, linewidth=0.5,
                                edgecolor=CALLED_COLOR if called else NOTCALLED_COLOR,
                                zorder=3))
        y0, y1, x0, x1 = f["zoom_box"]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                               edgecolor="white", linewidth=0.8, zorder=4))
        ins = ax.inset_axes([0.62, 0.02, 0.36, 0.36])
        ins.imshow(f["zoom_rgb"], interpolation="nearest")
        for (y, x, called) in f["zoom_puncta"]:
            ins.add_patch(Circle((x, y), radius=6.0, fill=False, linewidth=0.8,
                                 edgecolor=CALLED_COLOR if called else NOTCALLED_COLOR,
                                 zorder=3))
        ins.set_xticks([])
        ins.set_yticks([])
        for s in ins.spines.values():
            s.set_edgecolor("white")
            s.set_linewidth(0.8)
        ax.text(0.02, 0.985, f"{f['line']}  {f['well_id']}\n{f['short']}",
                transform=ax.transAxes, ha="left", va="top", fontsize=6.2,
                color="white", linespacing=1.25)
        ax.text(0.02, 0.03,
                f"{f['n_called']}/{f['n_puncta']} called ({f['frac']:.2f})",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=6.4,
                color="white", fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    hb = stamp_head(fig, "QC overlay: which puncta were called colocalized",
                    f"{ctx['dapi_label']} in grey, nucleus outlines in grey, one circle per "
                    f"nuclear {ctx['rna_label']} punctum; inset is a 1:1 crop",
                    ctx["filt"], y=0.988)
    foot = (
        f"Circles: {CALLED_COLOR} = called colocalized, {NOTCALLED_COLOR} = not called. "
        f"A punctum is called when the mean {ctx['partner_label']} intensity over its EXACT "
        f"half-max footprint (median {ctx['fp_area_median']:.0f} px) exceeds that nucleus's "
        f"{ctx['partner_label']} threshold. Circle radius on the full field is a "
        "drawing size, NOT the measurement region; the inset is 1:1 with the raw pixels and "
        "carries a 5 um burned scale bar. Only nuclear "
        f"{ctx['rna_label']} puncta are drawn. One representative field per well, chosen as "
        "the field whose called fraction is nearest that well's own mean. Full-resolution "
        "per-field overlays are in the overlays/ subfolder and are indexed in the workbook. "
        "Read this figure for the threshold caveat: because the partner threshold is "
        "per nucleus and Costes converges in a minority of nuclei, the calls tend to be "
        "all-or-nothing WITHIN a nucleus rather than punctum-by-punctum. Descriptive; "
        "no test.")
    fig.subplots_adjust(top=hb - 0.012, bottom=foot_bottom(fig, foot, 5.2),
                        left=0.006, right=0.994, wspace=0.02, hspace=0.03)
    stamp_foot(fig, foot, size=5.2)
    save(fig, out_dir, "csp05_qc_overlay_called_coloc", manifest,
         "QC overlays showing every nuclear rna1 punctum coloured by its colocalization call.",
         "run image planes at the run's z, nuclei label masks, spot_metrics.csv",
         "one representative field per well", f"{n} fields", "none (QC)")


def fig_anchor_directions(per_nucleus, well, ctx, out_dir, manifest):
    """Both anchor directions side by side, each against its own shuffle control."""
    bio = well[~well["secondary_only"]]
    panels = [
        (ctx["primary_obs"], ctx["primary_shuf"],
         "A  {} at nuclear {} puncta".format(ctx["partner_label"], ctx["rna_label"])),
        ("frac_called_coloc_partner_runthr", "frac_called_coloc_partner_shuffle_runthr",
         "B  {} at nuclear {} puncta".format(ctx["rna_label"], ctx["partner_label"])),
        ("paired_frac_rna1_at_partner", "paired_frac_rna1_at_partner_shuffle",
         "C  {} puncta paired to a {} punctum".format(ctx["rna_label"], ctx["partner_label"])),
        ("paired_frac_partner_at_rna1", "paired_frac_partner_at_rna1_shuffle",
         "D  {} puncta paired to a {} punctum".format(ctx["partner_label"], ctx["rna_label"])),
    ]
    panels = [t for t in panels if t[0] in bio.columns and bio[t[0]].notna().any()]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels) + 1.2, 5.4))
    axes = np.atleast_1d(axes)
    for ax, (obs_col, shuf_col, title) in zip(axes, panels):
        ytop = 0.0
        for xi, ln in enumerate(ARMS):
            x = float(xi)
            sub_w = bio[bio["line"] == ln].sort_values("well_id")
            for wi, (_, r) in enumerate(sub_w.iterrows()):
                ax.plot([x - 0.16, x + 0.16], [r[obs_col], r[shuf_col]],
                        color="#BBBBBB", linewidth=0.7, zorder=1)
                ax.scatter([x - 0.16], [r[obs_col]], s=46, marker="D",
                           facecolor=shade(arm_color(ln),
                                           0.35 + 0.55 * wi / max(1, len(sub_w) - 1)),
                           edgecolor="black", linewidth=0.6, zorder=3)
                ax.scatter([x + 0.16], [r[shuf_col]], s=38, marker="o",
                           facecolor="white", edgecolor=arm_color(ln),
                           linewidth=1.0, zorder=3)
            for dx, col in ((-0.16, obs_col), (0.16, shuf_col)):
                v = sub_w[col].to_numpy(dtype=float)
                v = v[np.isfinite(v)]
                if v.size:
                    m, lo, hi, _ = mean_ci(v)
                    ax.hlines(m, x + dx - 0.10, x + dx + 0.10, color="black",
                              linewidth=1.3, zorder=4)
                    if np.isfinite(lo):
                        ax.vlines(x + dx, lo, hi, color="black", linewidth=0.9, zorder=2)
                        ytop = max(ytop, hi)
                    ytop = max(ytop, float(v.max()))
        ax.set_xticks([float(i) for i in range(len(ARMS))])
        ax.set_xticklabels(list(ARMS))
        ax.set_xlim(-0.55, len(ARMS) - 0.45)
        ax.set_ylim(0, max(ytop, 1e-6) * 1.18)
        ax.set_title(title, fontsize=7.6)
    axes[0].set_ylabel("fraction (diamond = observed, open circle = shuffle)")
    hb = stamp_head(fig, "Colocalization measured from both anchors",
                    "each panel: one diamond per well against its own shuffle control; "
                    "black bar = arm mean with 95% CI", ctx["filt"], wrap=118, y=0.975)
    foot = (
        "A and B are footprint-based: the partner mean over each "
        f"{ctx['rna_label']} punctum's exact footprint (A) and the {ctx['rna_label']} mean "
        f"over each {ctx['partner_label']} punctum's exact footprint (B), each called "
        "against the other channel's batch threshold. A's observed value is the run's own "
        "stored column; B's is a RECONSTRUCTION, because the run stores no partner-anchored "
        "footprint column, built with the identical algorithm. C and D are spot-to-spot: "
        f"the fraction of puncta with a nearest partner punctum within {ctx['pair_um']:.2f} "
        "um, taken from the run's own paired flag, against a shuffle that redraws the OTHER "
        "channel's positions uniformly inside the same nucleus. DESCRIPTIVE: none of these "
        "four is the panel's tested endpoint; that remains panel A of csp04. Reading the "
        "two anchors together guards against a result that only exists in one direction. "
        f"n = {ctx['n_bio_nuc']} biological nuclei.")
    fig.subplots_adjust(top=hb - 0.075, bottom=foot_bottom(fig, foot, 5.2),
                        left=0.085, right=0.985, wspace=0.28)
    stamp_foot(fig, foot, size=5.2)
    save(fig, out_dir, "csp06_anchor_directions", manifest,
         "Object colocalization from both anchor directions plus spot-to-spot pairing, "
         "each against its own shuffle control.",
         "coloc_standard_per_well.csv", "biological wells only",
         f"{ctx['n_bio_nuc']} nuclei", "none (descriptive)")


def fig_composite(per_nucleus, well, contrasts, pooled, fields, ctx, out_dir, manifest):
    fig = plt.figure(figsize=(13.2, 8.4))
    gs = fig.add_gridspec(2, 3, left=0.055, right=0.985, hspace=0.34, wspace=0.26)
    ax = fig.add_subplot(gs[0, 0])
    superplot_axes(ax, per_nucleus, well, "pearson_r_csp", "Pearson r", "A  Pearson r")
    ax = fig.add_subplot(gs[0, 1])
    superplot_axes(ax, per_nucleus, well, "manders_m1_runthr", "Manders M1",
                   f"B  M1: {ctx['rna_label']} in {ctx['partner_label']}"
                   " (run batch threshold)")
    ax = fig.add_subplot(gs[0, 2])
    bio = well[~well["secondary_only"]]
    for xi, ln in enumerate(ARMS):
        x = float(xi)
        sub = bio[bio["line"] == ln].sort_values("well_id")
        for wi, (_, r) in enumerate(sub.iterrows()):
            ax.plot([x - 0.16, x + 0.16],
                    [r[ctx["primary_obs"]], r[ctx["primary_shuf"]]],
                    color="#BBBBBB", linewidth=0.7, zorder=1)
            ax.scatter([x - 0.16], [r[ctx["primary_obs"]]], s=48, marker="D",
                       facecolor=shade(arm_color(ln), 0.35 + 0.55 * wi / max(1, len(sub) - 1)),
                       edgecolor="black", linewidth=0.6, zorder=3)
            ax.scatter([x + 0.16], [r[ctx["primary_shuf"]]], s=40, marker="o",
                       facecolor="white", edgecolor=arm_color(ln), linewidth=1.0, zorder=3)
    row = contrasts[contrasts["endpoint"] == ctx["primary_obs"]].iloc[0]
    ax.set_xticks([float(i) for i in range(len(ARMS))])
    ax.set_xticklabels(list(ARMS))
    ax.set_xlim(-0.55, len(ARMS) - 0.45)
    ax.set_ylabel("fraction called colocalized")
    ax.set_title(f"C  object coloc (D) vs shuffle (o)   {stars(row['p_welch'])}", fontsize=8.5)
    for i, (ln, tag) in enumerate(zip(ARMS[:2], ("D", "E"))):
        ax = fig.add_subplot(gs[1, i])
        r, a = pooled[ln]
        rng = np.random.default_rng([ctx["seed"], 7])
        if r.size > CYTO_MAX_PIXELS:
            sel = rng.choice(r.size, size=CYTO_MAX_PIXELS, replace=False)
            r, a = r[sel], a[sel]
        cinfo = ctx["cyto"][ln]
        ax.hexbin(r.astype(np.float64), a.astype(np.float64), gridsize=90, bins="log",
                  mincnt=1, cmap="magma_r", linewidths=0.0)
        if np.isfinite(cinfo["costes_thr_rna1"]):
            ax.axvline(cinfo["costes_thr_rna1"], color=OKABE["blue"], lw=1.0, ls="--")
            ax.axhline(cinfo["costes_thr_partner"], color=OKABE["green"], lw=1.0, ls="--")
        ax.set_xlabel(f"{ctx['rna_label']} (a.u.)")
        ax.set_ylabel(f"{ctx['partner_label']} (a.u.)")
        ax.set_title(f"{tag}  cytofluorogram {ln}   r = {cinfo['pearson_r']:.3f}", fontsize=8.5)
    ax = fig.add_subplot(gs[1, 2])
    f = fields[0]
    ax.imshow(f["zoom_rgb"], interpolation="nearest")
    for (y, x, called) in f["zoom_puncta"]:
        ax.add_patch(Circle((x, y), radius=6.0, fill=False, linewidth=0.8,
                            edgecolor=CALLED_COLOR if called else NOTCALLED_COLOR, zorder=3))
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"F  called-coloc overlay, {f['line']} {f['well_id']} (1:1 crop)", fontsize=8.5)
    ax.legend(handles=[
        Line2D([], [], marker="o", linestyle="", markersize=6, markerfacecolor="none",
               markeredgecolor=CALLED_COLOR, label="called colocalized"),
        Line2D([], [], marker="o", linestyle="", markersize=6, markerfacecolor="none",
               markeredgecolor=NOTCALLED_COLOR, label="not called")],
        fontsize=6.2, frameon=False, loc="lower left", labelcolor="white")
    foot = (
        "A-B: nuclear mask, single z plane; descriptive, no test, well means as diamonds. "
        "M1 here uses the run's batch median + 2.5 MAD threshold for every nucleus; the "
        "Costes-with-fallback variant is a mixture of two conventions and is shown "
        "separately in csp01b. C: nucleus -> FoV -> well rollup; Welch t on "
        f"well means of the OBSERVED fraction, p = {fmt_p(row['p_welch'])}, "
        f"Hedges g = {row['hedges_g']:.3g}, MDE Hedges g = "
        f"{row['mde_hedges_g_alpha_0p05_power_0p80']:.2f} at alpha 0.05 / 80% power; the "
        "open circles are the within-nucleus shuffle control and they sit close to the "
        "observed value. D-E: pooled nuclear pixels subsampled to at most "
        f"{CYTO_MAX_PIXELS:,}; POOLED Costes thresholds are drawn as dashed lines only "
        "where Costes converged on the pooled sample ("
        + (", ".join(f"{k}: {'converged' if v['costes_converged'] else 'did NOT converge'}"
                     for k, v in ctx["cyto"].items()))
        + "). F: circles are "
        "drawing sizes, not the exact-footprint measurement region. "
        f"n = {ctx['n_bio_nuc']} biological nuclei in 6 wells; secondary-only fields "
        f"excluded. Costes ({ctx['costes_fit'].upper()}) converged for "
        f"{ctx['costes_pct']:.1f}% of nuclei. {ctx['filt']} seed {ctx['seed']}.")
    hb = stamp_head(fig, "Standard colocalization panel: "
                    f"{ctx['rna_label']} x {ctx['partner_label']}",
                    "A-B per-nucleus pixel coefficients (descriptive) - C the one tested "
                    "object endpoint - D-E pooled-pixel cytofluorograms - F what the call "
                    "looks like on the image", ctx["filt"], y=0.982)
    gs.update(top=hb - 0.030, bottom=foot_bottom(fig, foot, 5.2, pad=0.075))
    stamp_foot(fig, foot, size=5.2)
    save(fig, out_dir, "FIG_COLOC_STANDARD", manifest,
         "Composite: pixel coefficients, the tested object endpoint, cytofluorograms and a "
         "called-coloc overlay.",
         "all coloc_standard_* tables in this folder",
         "biological wells only", f"{ctx['n_bio_nuc']} nuclei, 6 wells",
         f"one Welch t on well means, p = {fmt_p(row['p_welch'])}")


# ------------------------------------------------------------------ workbook
README_ROWS = [
    ("Scope", "Human (Homo sapiens). Probe design GRCh38 (hg38) / GENCODE v49. "
              "Single z plane per field, the run's own autofocus plane. The arms "
              "compared and the pixel size are in the provenance block below."),
    ("What this adds", "The field-standard colocalization readouts (Pearson, Manders with a "
                       "Costes threshold, Li ICQ, an object-based coloc fraction with a shuffle "
                       "control, a cytofluorogram and line profiles) next to fishsuite's own "
                       "rotation-null metrics. Nothing here replaces the rotation null; it is a "
                       "second, more familiar view of the same images."),
    ("Pixel mask", "Nuclear mask only: the run's own <stem>__nuclei_label_mask.tif, one label "
                   "per nucleus, on the run's own autofocus z plane. Raw channel values, no "
                   "preprocessing, exactly the array the run measured."),
    ("Pearson r", "Standard product-moment correlation of the two channels over the nuclear "
                  "pixels. Threshold-free, so it must reproduce the run's own column exactly."),
    ("Li ICQ", "Intensity correlation quotient: (fraction of pixels where both channels "
               "deviate from their means in the same direction) - 0.5. Range -0.5 to +0.5. "
               "Threshold-free, so it must also reproduce the run's own column."),
    ("Costes threshold", "Per nucleus. Walk candidate rna1 thresholds down the regression line "
                         "of partner on rna1 and stop at the first pair where the Pearson r of "
                         "the strictly-below-both pixels is <= 0. At most 256 candidates, at "
                         "least 10 below-threshold pixels required. The default fit is total "
                         "least squares (orthogonal), which is what Costes et al. 2004 and "
                         "Fiji's Coloc2 use; --costes-fit ols selects the ordinary-least-squares "
                         "fit that fishsuite.core.thresholds.costes_threshold uses internally, "
                         "and both are reported per nucleus. Where it does not converge the run's "
                         "own batch median + 2.5*MAD thresholds are used instead; column "
                         "threshold_source records which."),
    ("READ THIS: Costes often does not converge here",
     "Costes assumes that below some threshold the two channels are uncorrelated background. "
     "Inside a nuclear mask that assumption is frequently violated: both channels carry the same "
     "nucleus-wide intensity gradient, so the below-threshold pixels stay positively correlated "
     "all the way down and the search runs out of candidates. The measured convergence rate for "
     "this run is in the provenance block below and on every figure footer; it is well under "
     "100%. Non-convergence is a property of the data, not a failure of the code. Consequence: "
     "the columns manders_m1_costes / manders_m2_costes and the primary object endpoint mix two "
     "threshold conventions across nuclei. Use manders_m*_costes_only for the Costes-converged "
     "subset alone, and the *_runthr columns for a single convention applied to every nucleus."),
    ("Which Manders to read", "Read manders_m*_runthr. It applies one threshold to every "
                              "nucleus, so the arms are comparable. manders_m*_costes is the "
                              "column this panel was specified to produce, but with Costes "
                              "converging in a minority of nuclei its arm mean is a weighted "
                              "average of two conventions and moves with the convergence "
                              "fraction. Figure csp01b shows all three side by side."),
    ("Sensitivity columns", "Every *_runthr column repeats the same calculation with the run's "
                            "own batch median + 2.5*MAD threshold for EVERY nucleus, so it is "
                            "free of the threshold mixture. Every *_costes_only column is the "
                            "Costes-converged subset alone. Compare the three before believing a "
                            "difference between arms."),
    ("Manders M1 / M2", "M1 = fraction of rna1 intensity in pixels where the partner is >= its "
                        "threshold. M2 = fraction of partner intensity where rna1 is >= its "
                        "threshold. The >= matches fishsuite.core.metrics.compute_coloc_metrics. "
                        "Columns manders_m*_costes use the per-nucleus Costes threshold; "
                        "manders_m*_runthr use the run's own thresholds and are there only to "
                        "prove the arithmetic reproduces the run."),
    ("Why Manders differs from the run", "The run's pixel_coloc block uses threshold_mode=mad "
                                         "with threshold_scope=batch and k_mad=2.5: ONE pooled "
                                         "median + 2.5*MAD threshold per channel for every "
                                         "nucleus in the run. Costes is per nucleus and derived "
                                         "from the joint intensity distribution of that nucleus. "
                                         "Pearson and Li ICQ are threshold-free and therefore "
                                         "identical; Manders is threshold-dependent and is "
                                         "expected to differ."),
    ("Object-based call", "For each nuclear rna1 punctum (taken from the run's spot_metrics.csv; "
                          "no spot is re-detected), the mean partner intensity over the "
                          "punctum's EXACT FOOTPRINT. Called colocalized when that mean exceeds "
                          "the nucleus's partner threshold. Per nucleus, the fraction of its "
                          "puncta that are called. No fixed disk is used anywhere in this tool."),
    ("Footprint definition", "The exact footprint is the 8-connected component of rna1 pixels at "
                             "or above a local half-max threshold that contains the punctum's "
                             "peak pixel. Threshold = bg + 0.5*(peak - bg), with bg the 10th "
                             "percentile of a per-spot window whose half-width scales with that "
                             "spot's measured FWHM, so the region scales with the real extent of "
                             "each punctum. Source: the run's own columns qki_at_miat_footprint "
                             "(partner mean over those pixels) and miat_footprint_area_px (their "
                             "count), produced by fishsuite.core.modes.rna_rna."
                             "_sample_qki_at_miat_footprint under "
                             "foci.compute_footprint_enrichment. Those column names are legacy "
                             "MIAT/QKI names; here they mean rna1 and the partner channel. "
                             "Per-arm median pixel counts are in the provenance block below."),
    ("Footprint reconstruction", "The observed value is READ from the run's column. The shuffle "
                                 "has to translate the footprint, which needs its pixel offsets, "
                                 "and those are not stored, so the footprint is rebuilt with the "
                                 "same algorithm and then checked against the run's stored area "
                                 "and mean for every punctum. The maximum absolute difference is "
                                 "in the provenance block; a non-zero value there would "
                                 "invalidate the shuffle."),
    ("Shuffle control", "The SAME footprint shape, translated to random positions drawn uniformly "
                        "from that nucleus's own pixels, with every translated pixel required to "
                        "stay inside the nucleus; 200 draws per punctum, averaged. It measures "
                        "what the call rate would be if puncta of identical size and shape sat "
                        "at random places in the same nucleus, so observed - shuffle is the part "
                        "not explained by the nucleus's partner intensity and the puncta's sizes "
                        "alone. Where a nucleus cannot contain a footprint at enough positions "
                        "the containment rule is relaxed to centre-in-nucleus with clip-to-edge; "
                        "column n_spots_shuffle_containment_relaxed records that per nucleus."),
    ("Rollup", "nucleus -> FoV mean -> well mean, the same convention as the run report's "
               "per_well.csv column well_mean_of_fov_values. Well = biological replicate, "
               "FoV = technical replicate."),
    ("Test", "ONE test in this panel: Welch t on the per-well means of the OBSERVED "
             "object coloc fraction, test arm against reference arm, "
             "with Hedges g and the minimum detectable effect at alpha "
             "0.05 and 80% power. No multiplicity correction is applied because only one "
             "endpoint is tested. Every pixel metric and every sensitivity endpoint is "
             "descriptive and carries a CI, not a p-value. With 3 vs 3 wells the design is "
             "underpowered for anything but a large effect; read the MDE before reading the p."),
    ("Cytofluorogram", "Pooled nuclear pixels per arm, subsampled to at most 2,000,000, plotted "
                       "as a log-density hexbin with the POOLED Costes thresholds drawn. The "
                       "pooled thresholds are not the per-nucleus thresholds used elsewhere."),
    ("Line profiles", "One nucleus per well, the one whose nuclear rna1 punctum count is nearest "
                      "that well's median. The line is the chord through the nucleus centroid "
                      "along the region principal axis, sampled by bilinear interpolation."),
    ("Display windows", "The LUT windows used for the crops and overlays are DISPLAY ONLY. No "
                        "measurement in this workbook uses them."),
    ("Not done here", "No GPU, no segmentation, no spot re-detection, no writes into the run "
                      "directory, no red-and-green pairing."),
]


def questions_sheet(run, per_nucleus, per_well, ctx):
    """Per-well table answering the four biological questions in plain columns.

    Every column is pulled from either this panel's own tables or the run's
    nuclei_metrics.csv; nothing is recomputed here. Rolled up nucleus -> FoV mean
    -> well mean, the run report's convention.
    """
    rna, par = ctx["rna_label"], ctx["partner_label"]
    nm = run.nuclei.copy()
    key = per_nucleus[["image", "nucleus_id", "line", "condition",
                       "secondary_only"]].copy()
    nm = key.merge(nm, on=["image", "nucleus_id"], how="left",
                   suffixes=("", "_run"))

    # question 1 columns come straight from the engine's per-nucleus table
    q1 = {
        f"{rna} puncta per nucleus": "rna_spot_count",
        f"{rna} nuclear puncta per nucleus": "nuclear_spot_count",
        f"{rna} fraction of puncta that are nuclear": "nuclear_spot_fraction",
        f"{rna} punctum diameter um (median per nucleus)": None,
        f"{par} nuclear mean intensity": "protein_nuclear_mean",
        f"{par} nuclear to cytoplasmic ratio": "protein_nc_ratio",
        f"{par} nuclear puncta per nucleus": "nuclear_spot_count_protein",
    }
    rows = nm[["image", "line", "condition", "secondary_only"]].copy()
    for label, col in q1.items():
        rows[label] = nm[col].to_numpy() if (col and col in nm.columns) else np.nan

    # median punctum diameter per nucleus, from the run's spot table
    if ctx.get("rna_diam_by_nucleus") is not None:
        d = ctx["rna_diam_by_nucleus"]
        rows[f"{rna} punctum diameter um (median per nucleus)"] = (
            pd.MultiIndex.from_arrays([nm["image"], nm["nucleus_id"]])
            .map(d).to_numpy(dtype=float))

    # questions 2-4 come from this panel plus the run's rotation null
    p2 = per_nucleus.set_index(["image", "nucleus_id"])
    idx = pd.MultiIndex.from_arrays([nm["image"], nm["nucleus_id"]])

    def take(col):
        return p2[col].reindex(idx).to_numpy(dtype=float) if col in p2.columns else np.nan

    def take_run(col):
        return nm[col].to_numpy(dtype=float) if col in nm.columns else np.nan

    rows[f"{par} at {rna} puncta: fraction called colocalized"] = take(ctx["primary_obs"])
    rows[f"{par} at {rna} puncta: shuffle control"] = take(ctx["primary_shuf"])
    rows[f"{par} at {rna} puncta: observed minus shuffle"] = take(ctx["primary_diff"])
    rows[f"Manders fraction of {par} intensity inside the {rna} mask"] = take(
        "manders_m2_runthr")
    rows[f"Manders fraction of {rna} intensity inside the {par} mask"] = take(
        "manders_m1_runthr")
    rows["Pearson r over nuclear pixels"] = take("pearson_r_csp")
    rows[f"{par} rotation-null enrichment at {rna} puncta (engine, alongside)"] = take_run(
        "protein_rotation_enrichment_at_rna1_spots")
    rows[f"{rna} puncta paired to a {par} punctum within {ctx['pair_um']:.2f} um"] = take(
        "paired_frac_rna1_at_partner")
    rows[f"{rna} paired fraction: shuffle control"] = take(
        "paired_frac_rna1_at_partner_shuffle")
    rows[f"{rna} at {par} puncta: fraction called colocalized"] = take(
        "frac_called_coloc_partner_runthr")
    rows[f"{rna} at {par} puncta: shuffle control"] = take(
        "frac_called_coloc_partner_shuffle_runthr")
    rows[f"{par} puncta paired to a {rna} punctum within {ctx['pair_um']:.2f} um"] = take(
        "paired_frac_partner_at_rna1")
    rows[f"{par} paired fraction: shuffle control"] = take(
        "paired_frac_partner_at_rna1_shuffle")

    value_cols = [c for c in rows.columns
                  if c not in ("image", "line", "condition", "secondary_only")]
    fov = rows.groupby(["line", "condition", "secondary_only", "image"],
                       as_index=False)[value_cols].mean()
    well = fov.groupby(["line", "condition", "secondary_only"],
                       as_index=False)[value_cols].mean()
    well = well.rename(columns={"condition": "well"})
    order = [a for a in ARMS] + [SEC_ONLY]
    well["_o"] = well["line"].map({a: i for i, a in enumerate(order)}).fillna(99)
    well = well.sort_values(["_o", "well"]).drop(columns="_o").reset_index(drop=True)
    return well


QUESTION_GUIDE = [
    ("Question 1 - are the puncta and the partner level themselves different "
     "between arms?",
     "Columns 'puncta per nucleus', 'nuclear puncta per nucleus', 'fraction of puncta "
     "that are nuclear', 'punctum diameter um', partner 'nuclear mean intensity', "
     "'nuclear to cytoplasmic ratio' and 'nuclear puncta per nucleus'. All are read "
     "from the run's nuclei_metrics.csv, not recomputed."),
    ("Question 2 - is the partner enriched at the RNA puncta?",
     "Columns 'at ... puncta: fraction called colocalized' with its shuffle control and "
     "the difference, the two Manders fractions, Pearson r over nuclear pixels, and the "
     "engine's rotation-null enrichment carried alongside as an independent "
     "randomisation control."),
    ("Question 3 - do partner PUNCTA sit on RNA puncta?",
     "Column 'puncta paired to a ... punctum within the run's pair distance', from the "
     "run's own nearest-neighbour flag, with a shuffle control that redraws the other "
     "channel's positions uniformly inside the same nucleus."),
    ("Question 4 - and the reverse direction?",
     "The same trio anchored on the partner channel instead: RNA at partner puncta "
     "called fraction with shuffle, and partner puncta paired to an RNA punctum with "
     "shuffle. A real association should show in both directions."),
    ("How to read every number here",
     "One row per well. Well = biological replicate, FoV = technical replicate; values "
     "are nucleus -> FoV mean -> well mean. Secondary-only rows are the no-primary-probe "
     "control and are flagged, never pooled with the arms. No p-values on this sheet: "
     "the panel runs ONE test, on the endpoint named in the README sheet."),
]


def write_workbook(path, ctx, per_nucleus, per_fov, per_well, contrasts,
                   line_profiles, overlays_index, cyto_summary, questions=None):
    dynamic = [
        ("Arms compared",
         "Reference {!r} against test {!r}. Every contrast reports test minus "
         "reference.".format(ctx["arms"][0], ctx["arms"][1])),
        ("PRIMARY called-coloc rule (THIS RUN)",
         "The partner threshold behind the primary call and behind the one tested "
         "endpoint is the {}. Reason: {}. Costes converged in {:.2f}% of biological "
         "nuclei against a {:.0f}% floor. When Costes converges in fewer than that "
         "fraction, a Costes-with-fallback column is a mixture of two conventions whose "
         "arm mean tracks the convergence fraction rather than the images, so the run's "
         "single batch median + k*MAD threshold is used instead. The other variant is "
         "reported in full as the sensitivity twin ({}); both appear in per_nucleus, "
         "per_FOV, per_well and contrasts.".format(
             ctx["primary_thr_label"], ctx["rule_why"], ctx["costes_pct"],
             ctx["costes_floor"], ctx["secondary_obs"])),
        ("Primary object endpoint column", ctx["primary_obs"]),
        ("Sensitivity twin column", ctx["secondary_obs"]),
    ]
    if ctx.get("note"):
        dynamic.append(("Run-specific caveat", ctx["note"]))
    readme = pd.DataFrame(dynamic + list(README_ROWS), columns=["item", "definition"])
    prov = pd.DataFrame(sorted(ctx["provenance"].items()), columns=["key", "value"])
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        readme.to_excel(xw, sheet_name="README", index=False)
        prov.to_excel(xw, sheet_name="README", index=False, startrow=len(readme) + 3)
        per_nucleus.to_excel(xw, sheet_name="per_nucleus", index=False)
        per_fov.to_excel(xw, sheet_name="per_FOV", index=False)
        per_well.to_excel(xw, sheet_name="per_well", index=False)
        contrasts.to_excel(xw, sheet_name="contrasts", index=False)
        cyto_summary.to_excel(xw, sheet_name="contrasts", index=False,
                              startrow=len(contrasts) + 3)
        line_profiles.to_excel(xw, sheet_name="line_profiles", index=False)
        overlays_index.to_excel(xw, sheet_name="overlays_index", index=False)
        if questions is not None:
            hdr = pd.DataFrame({"A": [
                "Run: " + str(ctx["provenance"]["run_dir"]),
                "One row per well. " + " ".join(q[0] for q in QUESTION_GUIDE[:4]),
            ]})
            hdr.to_excel(xw, sheet_name="biological_questions", index=False,
                         header=False)
            questions.to_excel(xw, sheet_name="biological_questions", index=False,
                               startrow=3)
            pd.DataFrame(QUESTION_GUIDE, columns=["question", "which columns answer it"]
                         ).to_excel(xw, sheet_name="biological_questions", index=False,
                                    startrow=len(questions) + 6)
        if questions is not None:
            ws = xw.sheets["biological_questions"]
            ws.freeze_panes = "A5"
            ws.column_dimensions["A"].width = 18
        for name, df in (("per_nucleus", per_nucleus), ("per_FOV", per_fov),
                         ("per_well", per_well), ("contrasts", contrasts),
                         ("line_profiles", line_profiles),
                         ("overlays_index", overlays_index)):
            ws = xw.sheets[name]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
        ws = xw.sheets["README"]
        ws.column_dimensions["A"].width = 26
        ws.column_dimensions["B"].width = 130


# ----------------------------------------------------------------------- main
def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--partner-slot", choices=("antibody", "rna2"), default=None)
    p.add_argument("--report-per-well", type=Path, default=None,
                   help="report per_well.csv used only for the well_id -> line map")
    p.add_argument("--fields", type=int, default=1, help="representative fields per well")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--call-threshold", choices=("auto", "costes", "batch"), default="auto",
                   help="partner threshold behind the PRIMARY called-coloc rule and the "
                        "tested object endpoint. auto (default) = the run's single batch "
                        "median+k*MAD threshold whenever Costes converges in fewer than "
                        "COSTES_MIN_CONVERGENCE_PCT of biological nuclei, else "
                        "Costes-with-fallback. The other variant is always reported as a "
                        "sensitivity column.")
    p.add_argument("--anchor", choices=("rna1", "partner", "both"), default="both",
                   help="which channel's puncta anchor the object call. rna1 = partner "
                        "sampled over each rna1 punctum's footprint; partner = the reverse; "
                        "both (default) = both directions, falling back to rna1 alone when "
                        "the run detected no partner spots.")
    p.add_argument("--arm-map", action="append", default=[], metavar="PATTERN=ARM",
                   help="map a run condition to an arm, e.g. --arm-map '*_NT_*=NT'. "
                        "Repeatable; first match wins.")
    p.add_argument("--arm-order", default=None,
                   help="comma-separated arm order, REFERENCE first; contrasts report "
                        "test minus reference.")
    p.add_argument("--note", default=None,
                   help="one sentence appended to every figure footnote, for a caveat "
                        "the reader must not miss.")
    p.add_argument("--costes-fit", choices=("tls", "ols"), default="tls",
                   help="regression line Costes walks down; tls (Costes 2004 / Fiji "
                        "Coloc2, default) or ols (what fishsuite uses internally)")
    p.add_argument("--input-dir", type=Path, default=None)
    for ch in ("rna", "partner", "dapi"):
        p.add_argument(f"--{ch}-min", type=float, default=None)
        p.add_argument(f"--{ch}-max", type=float, default=None)
        p.add_argument(f"--{ch}-lut", default=None)
    return p.parse_args(argv)


def main(argv=None):
    t_start = datetime.now()
    args = parse_args(list(sys.argv[1:] if argv is None else argv))
    np.random.seed(args.seed)
    import random as _random
    _random.seed(args.seed)
    set_style()
    globals()["NOTE_TEXT"] = (args.note or "").strip()

    run = Run(args.run, partner_slot=args.partner_slot, input_dir=args.input_dir)
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "overlays").mkdir(exist_ok=True)

    # `fishsuite report` writes the arm column as `group`; older hand-built
    # per_well tables call it `line`. Accept either, group first, or the panel
    # silently falls back to the condition prefix and refuses --arm-order.
    well_map = {}
    if args.report_per_well and Path(args.report_per_well).exists():
        pw = pd.read_csv(args.report_per_well)
        arm_col = next((c for c in ("group", "line") if c in pw.columns), None)
        if arm_col and "well_id" in pw.columns:
            well_map = dict(pw[["well_id", arm_col]].dropna().drop_duplicates().values)
        else:
            print(f"  WARN: {args.report_per_well} has neither a 'group' nor a "
                  f"'line' column alongside 'well_id'; arms fall back to the "
                  f"condition prefix unless --arm-map is given", flush=True)

    arm_rules = []
    for spec in args.arm_map:
        if "=" not in spec:
            raise SystemExit("--arm-map wants PATTERN=ARM, got {!r}".format(spec))
        pat, arm = spec.split("=", 1)
        arm_rules.append((pat.strip(), arm.strip()))

    conds = run.summary.loc[~run.summary["secondary_only"].astype(bool), "condition"]
    found = []
    for c in conds.astype(str).unique():
        a = resolve_arm(c, well_map, arm_rules)
        if a is None:
            raise SystemExit("condition {!r} does not map to an arm; pass "
                             "--arm-map 'PATTERN=ARM'".format(c))
        if a not in found:
            found.append(a)
    if args.arm_order:
        order = [a.strip() for a in args.arm_order.split(",") if a.strip()]
        missing = [a for a in order if a not in found]
        if missing:
            raise SystemExit("--arm-order names {} but the run has arms {}".format(
                missing, found))
        extra = [a for a in found if a not in order]
        if extra:
            raise SystemExit("--arm-order omits {}; name every arm".format(extra))
        found = order
    if len(found) < 2:
        raise SystemExit("need at least 2 arms, found {}".format(found))
    ARMS[:] = found
    print("[0/6] arms: reference {!r} vs test {!r}".format(ARMS[0], ARMS[1]), flush=True)

    ocfg = run.cfg["output"]
    win = {
        "rna": (args.rna_min if args.rna_min is not None else ocfg.get("manual_rna_min"),
                args.rna_max if args.rna_max is not None else ocfg.get("manual_rna_max")),
        "dapi": (args.dapi_min if args.dapi_min is not None else ocfg.get("manual_dapi_min"),
                 args.dapi_max if args.dapi_max is not None else ocfg.get("manual_dapi_max")),
    }
    pk = "manual_antibody" if run.partner_slot == "antibody" else "manual_rna2"
    win["partner"] = (args.partner_min if args.partner_min is not None else ocfg.get(f"{pk}_min"),
                      args.partner_max if args.partner_max is not None else ocfg.get(f"{pk}_max"))
    luts = {
        "rna": args.rna_lut or run.ch.get("rna_lut"),
        "dapi": args.dapi_lut or run.ch.get("dapi_lut"),
        "partner": args.partner_lut or run.ch.get(
            "antibody_lut" if run.partner_slot == "antibody" else "rna2_lut"),
    }

    fp_cols = ["qki_at_miat_footprint", "miat_footprint_area_px"]
    have = set(pd.read_csv(run.dir / "spot_metrics.csv", nrows=0).columns)
    missing = [c for c in fp_cols + ["spot_diameter_um"] if c not in have]
    if missing:
        raise SystemExit(
            "spot_metrics.csv lacks {}; this run was made without "
            "foci.compute_footprint_enrichment, so there is no exact punctum "
            "footprint to sample. Re-run the engine with it enabled - this tool "
            "will not substitute a disk.".format(missing))
    _use = ["image", "channel", "nucleus_id", "in_nucleus", "x_px", "y_px",
            "spot_peak_intensity", "spot_diameter_um"] + fp_cols
    if "paired_at_0p3um" in have:
        _use.append("paired_at_0p3um")
    spots = pd.read_csv(run.dir / "spot_metrics.csv", usecols=_use)
    n_partner_spots = int(((spots["channel"] == run.partner_spot_channel)
                           & (spots["in_nucleus"].astype(bool))).sum())
    anchor = args.anchor
    if anchor in ("partner", "both") and n_partner_spots == 0:
        if anchor == "partner":
            raise SystemExit(
                "--anchor partner needs partner spots, but the run detected none "
                "(foci.detect_antibody_spots off?)")
        anchor = "rna1"
        print("      no partner spots in this run; anchor falls back to rna1 only",
              flush=True)
    print("      anchor: {} ({} partner nuclear puncta)".format(anchor, n_partner_spots),
          flush=True)
    pair_um = 0.3
    if "spot_coloc_pair_distance_um" in run.thresholds.columns:
        _pv = pd.to_numeric(run.thresholds["spot_coloc_pair_distance_um"],
                            errors="coerce").dropna().unique()
        if len(_pv) == 1:
            pair_um = float(_pv[0])
        elif len(_pv) > 1:
            raise SystemExit("spot_coloc_pair_distance_um is not constant across "
                             "images: {}".format(sorted(_pv)))
    print("      spot-spot pair distance: {} um (from the run)".format(pair_um),
          flush=True)

    print(f"[1/6] per-nucleus pixel + object colocalization "
          f"({len(run.summary)} fields)", flush=True)
    per_nucleus, pooled, nuc_index, n_fallback, fp_checks = compute_tables(
        run, spots, well_map, arm_rules, args.seed, costes_fit=args.costes_fit,
        anchor=anchor, pair_um=pair_um)

    n_run_rows = len(run.nuclei)
    if len(per_nucleus) != n_run_rows:
        raise SystemExit(f"per-nucleus rows {len(per_nucleus)} != nuclei_metrics rows {n_run_rows}")
    bio = per_nucleus[~per_nucleus["secondary_only"]]
    n_bio = len(bio)
    costes_pct = 100.0 * float(per_nucleus["costes_converged"].mean())
    costes_ols_pct = 100.0 * float(per_nucleus["costes_ols_converged"].mean())
    costes_pct_bio = 100.0 * float(bio["costes_converged"].mean())
    max_pearson_diff = float(per_nucleus["pearson_abs_diff_vs_run"].max())
    max_icq_diff = float(per_nucleus["li_icq_abs_diff_vs_run"].max())
    max_m1_diff = float(per_nucleus["manders_m1_runthr_abs_diff_vs_run"].max())
    max_m2_diff = float(per_nucleus["manders_m2_runthr_abs_diff_vs_run"].max())
    print(f"      nuclei {len(per_nucleus)} (biological {n_bio}); "
          f"Costes converged {costes_pct:.2f}% all / {costes_pct_bio:.2f}% biological; "
          f"fallback used {n_fallback}", flush=True)
    fp_area_max = float(fp_checks["max_abs_area_diff"].max()) if len(fp_checks) else float("nan")
    fp_mean_max = float(fp_checks["max_abs_mean_diff"].max()) if len(fp_checks) else float("nan")
    fp_fallback = int(fp_checks["n_disk_fallback"].sum()) if len(fp_checks) else 0
    fp_nspots = int(fp_checks["n_spots"].sum()) if len(fp_checks) else 0
    print(f"      footprint reconstruction vs run columns over {fp_nspots} nuclear "
          f"rna1 puncta: max |d area_px| = {fp_area_max:.3e}, "
          f"max |d partner mean| = {fp_mean_max:.3e}, "
          f"fitted-radius fallbacks = {fp_fallback}", flush=True)
    print(f"      cross-check vs run: max |dPearson| = {max_pearson_diff:.3e}, "
          f"max |dICQ| = {max_icq_diff:.3e}, max |dM1@runthr| = {max_m1_diff:.3e}, "
          f"max |dM2@runthr| = {max_m2_diff:.3e}", flush=True)

    fp_all = pd.to_numeric(
        spots.loc[(spots['channel'] == 'rna1') & (spots['in_nucleus'].astype(bool)),
                  'miat_footprint_area_px'], errors='coerce').dropna()
    img2line = dict(zip(per_nucleus['image'], per_nucleus['line']))
    _sp = spots[(spots['channel'] == 'rna1') & (spots['in_nucleus'].astype(bool))].copy()
    _sp['line'] = _sp['image'].map(img2line)
    def _fp_med(ln):
        v = pd.to_numeric(_sp.loc[_sp['line'] == ln, 'miat_footprint_area_px'],
                          errors='coerce').dropna()
        return round(float(v.median()), 3) if len(v) else float('nan')
    fp_med_all = round(float(fp_all.median()), 3) if len(fp_all) else float('nan')
    fp_med_wt = _fp_med(ARMS[0])
    fp_med_ko = _fp_med(ARMS[1])
    print(f"      exact footprint area px: median {fp_med_all:.1f} "
          f"({ARMS[0]} {fp_med_wt:.1f}, {ARMS[1]} {fp_med_ko:.1f}), "
          f"IQR {fp_all.quantile(0.25):.0f}-{fp_all.quantile(0.75):.0f}, "
          f"range {fp_all.min():.0f}-{fp_all.max():.0f}", flush=True)
    n_relaxed_total = int(per_nucleus['n_spots_shuffle_containment_relaxed'].sum())
    print(f"      shuffle containment relaxed for {n_relaxed_total} of {fp_nspots} "
          f"puncta", flush=True)

    if args.call_threshold == "auto":
        call_rule = "batch" if costes_pct_bio < COSTES_MIN_CONVERGENCE_PCT else "costes"
        rule_why = (
            "auto: Costes converged in {:.2f}% of biological nuclei, "
            "{} the {:.0f}% floor".format(
                costes_pct_bio,
                "below" if call_rule == "batch" else "at or above",
                COSTES_MIN_CONVERGENCE_PCT))
    else:
        call_rule = args.call_threshold
        rule_why = "forced by --call-threshold {}".format(args.call_threshold)
    if call_rule == "batch":
        primary_obs = "frac_called_coloc_runthr"
        primary_shuf = "frac_called_coloc_shuffle_runthr"
        primary_diff = "frac_called_coloc_minus_shuffle_runthr"
        primary_lab = "run batch threshold, every nucleus"
        secondary_obs = "frac_called_coloc"
        secondary_shuf = "frac_called_coloc_shuffle"
        secondary_lab = "Costes threshold with fallback"
        primary_thr_col = "run_thr_partner"
    else:
        primary_obs = "frac_called_coloc"
        primary_shuf = "frac_called_coloc_shuffle"
        primary_diff = "frac_called_coloc_minus_shuffle"
        primary_lab = "Costes threshold with fallback"
        secondary_obs = "frac_called_coloc_runthr"
        secondary_shuf = "frac_called_coloc_shuffle_runthr"
        secondary_lab = "run batch threshold, every nucleus"
        primary_thr_col = "thr_partner_used"
    per_nucleus["thr_partner_primary"] = per_nucleus[primary_thr_col]
    per_nucleus["primary_call_threshold_rule"] = call_rule
    print("      PRIMARY call threshold: {} ({})".format(call_rule, rule_why), flush=True)

    per_fov, per_well = rollup(per_nucleus)
    _bw = per_well[~per_well["secondary_only"]]
    _n_ref = int((_bw["line"] == ARMS[0]).sum())
    _n_test = int((_bw["line"] == ARMS[1]).sum())
    mde = mde_hedges_g(n1=_n_test, n2=_n_ref)
    print("      MDE Hedges g = {:.3f} at alpha {} / power {} with {} vs {} wells".format(
        mde, ALPHA, POWER, _n_ref, _n_test), flush=True)
    contrasts = contrasts_table(per_well, per_nucleus, mde, primary_obs, secondary_obs)

    ctx = dict(
        seed=args.seed, rna_label=run.rna_label, partner_label=run.partner_label,
        dapi_label=run.dapi_label, win=win, luts=luts,
        costes_pct=costes_pct_bio, costes_fit=args.costes_fit, n_bio_nuc=n_bio,
        arm_n_text=", ".join(f"{int((bio['line'] == a).sum())} {a}" for a in ARMS),
        n_sec_nuc=int(per_nucleus["secondary_only"].sum()),
        n_bio_nuc_with_puncta=int((bio["n_rna1_nuclear_puncta"] > 0).sum()),
        fp_area_median=fp_med_all, fp_area_median_wt=fp_med_wt,
        fp_area_median_ko=fp_med_ko,
        fp_area_q1=float(fp_all.quantile(0.25)) if len(fp_all) else float('nan'),
        fp_area_q3=float(fp_all.quantile(0.75)) if len(fp_all) else float('nan'),
        fp_area_min=float(fp_all.min()) if len(fp_all) else float('nan'),
        fp_area_max=float(fp_all.max()) if len(fp_all) else float('nan'),
        n_shuffle_relaxed=n_relaxed_total,
        primary_obs=primary_obs, primary_shuf=primary_shuf,
        primary_diff=primary_diff, secondary_obs=secondary_obs,
        secondary_shuf=secondary_shuf,
        primary_thr_label=primary_lab, secondary_thr_label=secondary_lab,
        call_rule=call_rule, rule_why=rule_why, note=(args.note or ""),
        arms=list(ARMS), costes_floor=COSTES_MIN_CONVERGENCE_PCT,
        pair_um=pair_um, anchor=anchor,
        rna_diam_by_nucleus=(
            spots[(spots["channel"] == "rna1") & (spots["in_nucleus"].astype(bool))]
            .groupby(["image", "nucleus_id"])["spot_diameter_um"].median()
            if "spot_diameter_um" in spots.columns else None),
        filt=(f"gate: nuclear mask only, single z plane; biological wells only "
              f"(secondary-only fields excluded); Costes per-nucleus thresholds with the run's "
              f"batch median + 2.5 MAD as fallback; alpha {ALPHA}."),
    )

    print("[2/6] cytofluorograms", flush=True)
    cyto = {}
    for ln in ARMS:
        tag = re.sub(r"[^A-Za-z0-9]+", "_", ln).strip("_")
        if ln in pooled:
            cyto[ln] = fig_cytofluorogram(pooled, ctx, out_dir, MANIFEST, ln, tag)
    ctx["cyto"] = cyto
    cyto_summary = pd.DataFrame(list(cyto.values()))

    print("[3/6] representative nuclei and fields", flush=True)
    profiles, prof_rows = build_line_profiles(run, per_nucleus, spots, ctx)
    fields, overlays_index = build_overlays(run, per_nucleus, spots, ctx, out_dir, args.fields)

    print("[4/6] figures", flush=True)
    fig_pearson_manders(per_nucleus, per_well, ctx, out_dir, MANIFEST)
    fig_manders_threshold_sensitivity(per_nucleus, per_well, ctx, out_dir, MANIFEST)
    fig_line_profiles(profiles, ctx, out_dir, MANIFEST)
    fig_object_fraction(per_well, contrasts, ctx, out_dir, MANIFEST)
    fig_overlays(fields, ctx, out_dir, MANIFEST)
    if per_nucleus['frac_called_coloc_partner_runthr'].notna().any() or per_nucleus['paired_frac_rna1_at_partner'].notna().any():
        fig_anchor_directions(per_nucleus, per_well, ctx, out_dir, MANIFEST)
    fig_composite(per_nucleus, per_well, contrasts, pooled, fields, ctx, out_dir, MANIFEST)

    print("[5/6] tables and workbook", flush=True)
    ctx["provenance"] = {
        "run_dir": str(run.dir),
        "input_dir": str(run.input_dir),
        "analysis_mode": run.mode,
        "partner_slot": run.partner_slot,
        "rna_label": run.rna_label,
        "partner_label": run.partner_label,
        "seed": args.seed,
        "object_sampling_region": "exact half-max punctum footprint (never a disk)",
        "footprint_source_columns": "qki_at_miat_footprint / miat_footprint_area_px",
        "footprint_area_px_median_all": fp_med_all,
        "footprint_area_px_median_WT": fp_med_wt,
        "footprint_area_px_median_QKI_KO": fp_med_ko,
        "footprint_recon_max_abs_area_diff_vs_run": fp_area_max,
        "footprint_recon_max_abs_partner_mean_diff_vs_run": fp_mean_max,
        "footprint_n_fitted_radius_fallback_spots": fp_fallback,
        "footprint_n_spots_checked": fp_nspots,
        "arms_reference_then_test": " -> ".join(ARMS),
        "primary_call_threshold_rule": call_rule,
        "primary_call_threshold_reason": rule_why,
        "primary_object_endpoint": primary_obs,
        "sensitivity_object_endpoint": secondary_obs,
        "costes_min_convergence_pct_floor": COSTES_MIN_CONVERGENCE_PCT,
        "shuffle_n_puncta_containment_relaxed": n_relaxed_total,
        "shuffle_draws": SHUFFLE_DRAWS,
        "cytofluorogram_max_pixels": CYTO_MAX_PIXELS,
        "run_pixel_coloc_threshold_mode": run.cfg["pixel_coloc"]["threshold_mode"],
        "run_pixel_coloc_threshold_scope": run.cfg["pixel_coloc"]["threshold_scope"],
        "run_pixel_coloc_k_mad": run.cfg["pixel_coloc"]["k_mad"],
        "n_nuclei_total": len(per_nucleus),
        "n_nuclei_biological": n_bio,
        "n_nuclei_secondary_only": ctx["n_sec_nuc"],
        "costes_converged_pct_all": round(costes_pct, 4),
        "costes_converged_pct_biological": round(costes_pct_bio, 4),
        "costes_fit": args.costes_fit,
        "costes_converged_pct_all_ols_variant": round(costes_ols_pct, 4),
        "n_nuclei_costes_fallback_to_run_threshold": int(n_fallback),
        "max_abs_pearson_diff_vs_run": max_pearson_diff,
        "max_abs_li_icq_diff_vs_run": max_icq_diff,
        "max_abs_manders_m1_runthr_diff_vs_run": max_m1_diff,
        "max_abs_manders_m2_runthr_diff_vs_run": max_m2_diff,
        "mde_hedges_g_alpha_0p05_power_0p80": mde,
        "built_utc": datetime.now(timezone.utc).isoformat(),
    }
    per_nucleus.to_csv(out_dir / "coloc_standard_per_nucleus.csv", index=False)
    per_fov.to_csv(out_dir / "coloc_standard_per_fov.csv", index=False)
    per_well.to_csv(out_dir / "coloc_standard_per_well.csv", index=False)
    contrasts.to_csv(out_dir / "coloc_standard_contrasts.csv", index=False)
    lp = pd.DataFrame(prof_rows)
    try:
        qsheet = questions_sheet(run, per_nucleus, per_well, ctx)
    except Exception as _e:
        print("      WARN: questions sheet not built: {}: {}".format(
            type(_e).__name__, _e), flush=True)
        qsheet = None
    write_workbook(out_dir / "coloc_standard_panel.xlsx", ctx, per_nucleus, per_fov,
                   per_well, contrasts, lp, overlays_index, cyto_summary, qsheet)
    if qsheet is not None:
        qsheet.to_csv(out_dir / "coloc_standard_biological_questions.csv", index=False)

    print("[6/6] provenance", flush=True)
    write_provenance(out_dir, ctx, MANIFEST, t_start)
    print(f"\ndone -> {out_dir}", flush=True)
    return 0


def build_line_profiles(run, per_nucleus, spots, ctx):
    """One nucleus per well: the one whose nuclear rna1 punctum count is nearest
    that well's median."""
    from skimage.measure import regionprops
    bio = per_nucleus[~per_nucleus["secondary_only"]]
    picks = []
    for ln in ARMS:
        sub = bio[bio["line"] == ln]
        for wname in sorted(sub["condition"].unique()):
            w = sub[(sub["condition"] == wname) & (sub["n_rna1_nuclear_puncta"] > 0)]
            if not len(w):
                continue
            med = float(w["n_rna1_nuclear_puncta"].median())
            k = int((w["n_rna1_nuclear_puncta"] - med).abs().idxmin())
            picks.append(per_nucleus.loc[k])
    profiles = []
    rows = []
    for p in picks:
        dapi, rna, par, vx_nm = run.planes(p["image"], int(p["z_plane"]))
        labels = run.label_mask(p["stem"])
        mask = labels == int(p["nucleus_id"])
        rp = [r for r in regionprops(mask.astype(np.uint8)) if r.label == 1][0]
        cy, cx = rp.centroid
        src, dst = chord_through(mask, cy, cx, rp.orientation)
        prof = {}
        for key, plane in (("rna1", rna), ("partner", par), ("dapi", dapi)):
            v = sample_profile(plane, src, dst)
            prof[key] = v
            lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
            prof[f"{key}_norm"] = (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)
            prof[f"raw_lo_{key}"] = lo
            prof[f"raw_hi_{key}"] = hi
        n = prof["rna1"].size
        um_per_px = float(vx_nm) / 1000.0
        length_px = math.hypot(dst[0] - src[0], dst[1] - src[1])
        prof["dist_um"] = np.linspace(0, length_px * um_per_px, n)
        ys, xs = np.nonzero(mask)
        half = int(max(60, 0.62 * max(ys.max() - ys.min(), xs.max() - xs.min())))
        y0, y1, x0, x1 = square_crop(mask.shape, cy, cx, half)
        prof["crop_rgb"] = merged_rgb(dapi[y0:y1, x0:x1], rna[y0:y1, x0:x1],
                                      par[y0:y1, x0:x1], ctx["win"], ctx["luts"])
        prof.update(src_y=src[0] - y0, src_x=src[1] - x0,
                    dst_y=dst[0] - y0, dst_x=dst[1] - x0,
                    line=p["line"], well_id=p["condition"], image=p["image"],
                    nucleus_id=int(p["nucleus_id"]))
        profiles.append(prof)
        for i in range(n):
            rows.append(dict(line=p["line"], well_id=p["condition"], image=p["image"],
                             nucleus_id=int(p["nucleus_id"]), sample_index=i,
                             distance_um=float(prof["dist_um"][i]),
                             rna1_raw=float(prof["rna1"][i]),
                             partner_raw=float(prof["partner"][i]),
                             dapi_raw=float(prof["dapi"][i]),
                             rna1_norm=float(prof["rna1_norm"][i]),
                             partner_norm=float(prof["partner_norm"][i]),
                             dapi_norm=float(prof["dapi_norm"][i])))
    return profiles, rows


def build_overlays(run, per_nucleus, spots, ctx, out_dir, n_per_well):
    """Representative field(s) per well: the field whose called fraction is
    nearest that well's own mean. Writes full-resolution per-field overlays."""
    from PIL import Image
    from skimage.measure import find_contours
    bio = per_nucleus[~per_nucleus["secondary_only"]]
    fov = (bio.groupby(["line", "condition", "image"], as_index=False)
           .agg(frac=(ctx["primary_obs"], "mean"),
                n_puncta=("n_rna1_nuclear_puncta", "sum"),
                n_called=("n_called_coloc", "sum")))
    picks = []
    for ln in ARMS:
        for wname in sorted(fov.loc[fov["line"] == ln, "condition"].unique()):
            sub = fov[(fov["line"] == ln) & (fov["condition"] == wname)].copy()
            sub["d"] = (sub["frac"] - sub["frac"].mean()).abs()
            picks.extend(sub.sort_values("d").head(int(n_per_well)).to_dict("records"))
    sp_nuc = spots[(spots["channel"] == "rna1") & (spots["in_nucleus"].astype(bool))]
    fields = []
    idx_rows = []
    for rec in picks:
        name = rec["image"]
        prow = per_nucleus[per_nucleus["image"] == name].iloc[0]
        dapi, rna, par, vx_nm = run.planes(name, int(prow["z_plane"]))
        labels = run.label_mask(prow["stem"])
        thr_by_nid = dict(zip(per_nucleus.loc[per_nucleus["image"] == name, "nucleus_id"],
                              per_nucleus.loc[per_nucleus["image"] == name,
                                              "thr_partner_primary"]))
        sub = sp_nuc[sp_nuc["image"] == name]
        cys = sub["y_px"].to_numpy(dtype=np.intp)
        cxs = sub["x_px"].to_numpy(dtype=np.intp)
        nids = sub["nucleus_id"].to_numpy(dtype=int)
        # Observed partner mean over the EXACT punctum footprint, straight from
        # the run's own column. No disk anywhere.
        means = pd.to_numeric(sub["qki_at_miat_footprint"],
                              errors="coerce").to_numpy(dtype=float)
        thrs = np.array([thr_by_nid.get(int(k), np.inf) for k in nids], dtype=float)
        called = means > thrs
        field_rgb = gray_rgb(dapi, ctx["win"]["dapi"])
        outlines = []
        for nid in np.unique(labels[labels > 0]):
            for c in find_contours((labels == nid).astype(float), 0.5):
                outlines.append((c[:, 0], c[:, 1]))
        # zoom on the nucleus with the most puncta in this field
        if nids.size:
            best = int(pd.Series(nids).value_counts().idxmax())
            ys, xs = np.nonzero(labels == best)
            zy, zx = float(ys.mean()), float(xs.mean())
        else:
            best = -1
            zy, zx = labels.shape[0] / 2.0, labels.shape[1] / 2.0
        zy0, zy1, zx0, zx1 = square_crop(labels.shape, zy, zx, 170)
        zoom_rgb = merged_rgb(dapi[zy0:zy1, zx0:zx1], rna[zy0:zy1, zx0:zx1],
                              par[zy0:zy1, zx0:zx1], ctx["win"], ctx["luts"])
        zoom_rgb = _out.burn_scale_bar(zoom_rgb, vx_nm, bar_um=5.0, height_px=5,
                                       margin_px=10, font_px=13)
        inz = (cys >= zy0) & (cys < zy1) & (cxs >= zx0) & (cxs < zx1)
        fields.append(dict(
            line=rec["line"], well_id=rec["condition"], image=name,
            short=Path(name).stem[-26:], field_rgb=field_rgb,
            outlines=outlines,
            puncta=list(zip(cys.tolist(), cxs.tolist(), called.tolist())),
            zoom_rgb=zoom_rgb, zoom_box=(zy0, zy1, zx0, zx1),
            zoom_puncta=list(zip((cys[inz] - zy0).tolist(), (cxs[inz] - zx0).tolist(),
                                 called[inz].tolist())),
            n_puncta=int(cys.size), n_called=int(called.sum()),
            frac=float(called.mean()) if cys.size else float("nan")))
        # full-resolution standalone overlays
        stem = str(prow["stem"])
        full_png = out_dir / "overlays" / f"{stem}__called_coloc_field.png"
        zoom_png = out_dir / "overlays" / f"{stem}__called_coloc_zoom.png"
        _save_field_overlay(full_png, field_rgb, outlines, cys, cxs, called, vx_nm,
                            ctx, rec, radius=11.0, bar_um=20.0)
        _save_zoom_overlay(zoom_png, zoom_rgb, cys[inz] - zy0, cxs[inz] - zx0, called[inz])
        idx_rows.append(dict(line=rec["line"], well_id=rec["condition"], image=name,
                             stem=stem, z_plane=int(prow["z_plane"]),
                             n_nuclear_rna1_puncta=int(cys.size),
                             n_called_coloc=int(called.sum()),
                             frac_called=float(called.mean()) if cys.size else np.nan,
                             zoom_y0=zy0, zoom_y1=zy1, zoom_x0=zx0, zoom_x1=zx1,
                             zoom_nucleus_id=best,
                             field_overlay_png=str(full_png),
                             zoom_overlay_png=str(zoom_png)))
    return fields, pd.DataFrame(idx_rows)


def _save_field_overlay(path, rgb, outlines, cys, cxs, called, vx_nm, ctx, rec,
                        radius=11.0, bar_um=20.0):
    h, w = rgb.shape[:2]
    fig = plt.figure(figsize=(w / 300.0, h / 300.0), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(rgb, interpolation="nearest")
    for (yy, xx) in outlines:
        ax.plot(xx, yy, color="#8A8A8A", linewidth=0.45, zorder=2)
    for y, x, c in zip(cys, cxs, called):
        ax.add_patch(Circle((x, y), radius=radius, fill=False, linewidth=0.7,
                            edgecolor=CALLED_COLOR if c else NOTCALLED_COLOR, zorder=3))
    bar_px = bar_um * 1000.0 / float(vx_nm)
    ax.plot([w - 40 - bar_px, w - 40], [h - 40, h - 40], color="white", linewidth=3.0)
    ax.text(w - 40 - bar_px / 2, h - 52, f"{bar_um:.0f} um", color="white",
            ha="center", va="bottom", fontsize=7)
    ax.text(12, 22, f"{rec['line']}  {rec['condition']}  |  "
                    f"{int(np.sum(called))}/{len(called)} called colocalized",
            color="white", fontsize=8, ha="left", va="top")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _save_zoom_overlay(path, rgb, cys, cxs, called):
    h, w = rgb.shape[:2]
    fig = plt.figure(figsize=(w / 100.0, h / 100.0), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(rgb, interpolation="nearest")
    for y, x, c in zip(cys, cxs, called):
        ax.add_patch(Circle((x, y), radius=7.0, fill=False, linewidth=1.0,
                            edgecolor=CALLED_COLOR if c else NOTCALLED_COLOR, zorder=3))
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    fig.savefig(path, dpi=100)
    plt.close(fig)


def write_provenance(out_dir, ctx, manifest, t_start):
    import PIL
    import scipy
    import skimage
    import fishsuite
    (out_dir / "command.log").write_text(
        "{}\ncwd={}\n{} {}\nelapsed_s={:.1f}\n".format(
            datetime.now(timezone.utc).isoformat(), os.getcwd(), sys.executable,
            " ".join(sys.argv), (datetime.now() - t_start).total_seconds()),
        encoding="utf-8")
    (out_dir / "versions.txt").write_text(
        "python={}\nfishsuite={}\nfishsuite.__file__={}\nnumpy={}\npandas={}\nscipy={}\n"
        "scikit-image={}\nmatplotlib={}\npillow={}\nseed={}\nplatform={}\n".format(
            sys.version.split()[0], fishsuite.__version__, fishsuite.__file__,
            np.__version__, pd.__version__, scipy.__version__, skimage.__version__,
            matplotlib.__version__, PIL.__version__, ctx["seed"], sys.platform),
        encoding="utf-8")
    lines = ["# Figure index - standard colocalization panel", "",
             f"Built {ctx['provenance']['built_utc']} from `{ctx['provenance']['run_dir']}`.",
             f"Seed {ctx['seed']}. Every chart is a 600-dpi PNG plus an editable-text SVG "
             "(`svg.fonttype='none'`), Arial, Okabe-Ito, no red-and-green pairing.", "",
             "| figure | what it shows | gate | n | test |", "|---|---|---|---|---|"]
    for m in manifest:
        lines.append("| `{}` | {} | {} | {} | {} |".format(
            m["figure"], m["description"], m["gate"], m["n"], m["test"]))
    lines += ["", "## Colours",
              *[f"- {a}: `{arm_color(a)}`" for a in ARMS],
              f"- called colocalized: `{CALLED_COLOR}`",
              f"- not called: `{NOTCALLED_COLOR}`", "",
              "## Reading order",
              "1. `csp01` - are the per-nucleus pixel coefficients where you expect them?",
              "1b. `csp01b` - what the threshold choice does to Manders. Read before csp04.",
              "2. `csp02` - what does the joint intensity distribution actually look like?",
              "3. `csp03` - does the signal track across a single nucleus?",
              "4. `csp04` - the one tested endpoint, against its shuffle control.",
              "5. `csp05` - do the calls sit on visible partner signal?",
              "6. `FIG_COLOC_STANDARD` - the composite."]
    (out_dir / "FIGURE_INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


MANIFEST = []

if __name__ == "__main__":
    raise SystemExit(main())
