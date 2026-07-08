"""Threshold computation — bit-identical port of fiji_scripts/Coloc_Core.py
and Coloc_Analysis.py.

The Fiji pipeline uses these conventions:
  - ``median(vals)``: numpy-style mid value (50th percentile, average of two
    middle values when n is even).
  - ``mad(vals)``: median absolute deviation around the median, RAW (not
    scaled by 1.4826).
  - ``coloc_threshold(vals, "mad")``: ``median + COLOC_K_MAD * mad``.

These functions take 1-D arrays / lists and return floats. They are designed
to match the Jython math line-for-line so the standalone suite produces
bit-identical thresholds.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional, Tuple

import numpy as np


def median(vals) -> float:
    """Mirror of Fiji ``Coloc_Core.median`` (average-of-two when n even)."""
    if vals is None:
        return 0.0
    v = list(vals)
    n = len(v)
    if n == 0:
        return 0.0
    s = sorted(v)
    if n % 2 == 1:
        return float(s[n // 2])
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def mad(vals, center: Optional[float] = None) -> float:
    """Median absolute deviation (raw, not scaled). Mirrors Fiji."""
    if vals is None:
        return 0.0
    v = list(vals)
    if not v:
        return 0.0
    if center is None:
        center = median(v)
    dev = [abs(float(x) - center) for x in v]
    return median(dev)


def percentile_sorted(vals_sorted, p: float) -> float:
    """Mirror of Fiji ``Coloc_Core.percentile`` — index by rounded position."""
    if not vals_sorted:
        return 0.0
    if p <= 0:
        return float(vals_sorted[0])
    if p >= 100:
        return float(vals_sorted[-1])
    idx = int(round((p / 100.0) * (len(vals_sorted) - 1)))
    idx = max(0, min(idx, len(vals_sorted) - 1))
    return float(vals_sorted[idx])


def coloc_threshold(
    vals,
    mode: str = "mad",
    *,
    k_mad: float = 2.0,
    percentile: float = 80.0,
    vals_other=None,  # for costes mode (paired)
) -> float:
    """Compute one-channel threshold. Mirrors Fiji ``coloc_threshold``."""
    v = list(vals) if vals is not None else []
    if not v:
        return float("inf")
    s = sorted(v)
    if mode == "percentile":
        return percentile_sorted(s, float(percentile))
    if mode == "costes":
        other = list(vals_other) if vals_other is not None else None
        if other is None or len(other) != len(v):
            # No paired channel (or a length mismatch that would break the
            # pixel pairing) -> Fiji MAD fallback. This is the single-channel
            # behavior and matches costes_threshold()'s own MAD fallback.
            med = median(s)
            m = mad(s, center=med)
            return med if m <= 0 else (med + float(k_mad) * m)
        # Paired Costes regression. costes_threshold(this, other) returns
        # (this_threshold, other_threshold); we want THIS channel's threshold.
        # ``v``/``other`` are kept in ORIGINAL pixel order (not sorted) so the
        # per-pixel pairing the regression relies on is preserved.
        r_thr, _ = costes_threshold(v, other)
        if not math.isfinite(r_thr):
            # Costes could not resolve (e.g. < 20 paired pixels) -> MAD fallback
            # so the caller never receives a non-finite threshold.
            med = median(s)
            m = mad(s, center=med)
            return med if m <= 0 else (med + float(k_mad) * m)
        return float(r_thr)
    # Default: MAD
    med = median(s)
    m = mad(s, center=med)
    return med if m <= 0 else (med + float(k_mad) * m)


def costes_threshold(rvals, avals) -> Tuple[float, float]:
    """Costes automatic threshold — bit-identical to Coloc_Analysis.py."""
    rvals = list(rvals)
    avals = list(avals)
    n = len(rvals)
    if n < 20:
        return (float("inf"), float("inf"))

    r_mean = sum(rvals) / float(n)
    a_mean = sum(avals) / float(n)
    num = 0.0
    den = 0.0
    for i in range(n):
        dr = rvals[i] - r_mean
        num += dr * (avals[i] - a_mean)
        den += dr * dr
    if den == 0:
        return (float("inf"), float("inf"))
    slope = num / den
    intercept = a_mean - slope * r_mean

    r_sorted = sorted(set(rvals), reverse=True)
    step = max(1, len(r_sorted) // 256)
    thresholds = r_sorted[::step]

    for r_t in thresholds:
        a_t = slope * r_t + intercept
        br = []
        ba = []
        for i in range(n):
            if rvals[i] < r_t and avals[i] < a_t:
                br.append(rvals[i])
                ba.append(avals[i])
        if len(br) < 10:
            continue
        bm_r = sum(br) / float(len(br))
        bm_a = sum(ba) / float(len(ba))
        bnum = 0.0
        bd_r = 0.0
        bd_a = 0.0
        for i in range(len(br)):
            dr = br[i] - bm_r
            da = ba[i] - bm_a
            bnum += dr * da
            bd_r += dr * dr
            bd_a += da * da
        if bd_r > 0 and bd_a > 0:
            bp = bnum / math.sqrt(bd_r * bd_a)
            if bp <= 0:
                return (r_t, max(0.0, a_t))
    # Fallback (Coloc_Analysis lines 116-126)
    k_mad = 2.0
    _r_med = median(rvals)
    _r_mad = mad(rvals, center=_r_med)
    _a_med = median(avals)
    _a_mad = mad(avals, center=_a_med)
    _r_fb = (_r_med + k_mad * 1.4826 * _r_mad) if _r_mad > 0 else _r_med
    _a_fb = (_a_med + k_mad * 1.4826 * _a_mad) if _a_mad > 0 else _a_med
    return (_r_fb, _a_fb)


def batch_threshold(
    pooled_values: Iterable[float],
    *,
    mode: str = "mad",
    k_mad: float = 2.0,
    percentile: float = 80.0,
) -> float:
    """Pre-scan: compute a single pooled threshold across many images."""
    return coloc_threshold(list(pooled_values), mode=mode, k_mad=k_mad, percentile=percentile)
