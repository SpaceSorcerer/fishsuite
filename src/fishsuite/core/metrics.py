"""Per-nucleus pixel-colocalization metrics.

Numpy port of fiji_scripts/Coloc_Analysis.py::compute_nucleus_coloc_metrics.
Math is line-for-line equivalent to the Jython implementation, but operates
on flat 1-D numpy arrays of the pixels within a nuclear mask.

Metrics returned (per nucleus):
  - n_pix, rna_mean, ab_mean
  - pearson_r, spearman_rho, li_icq, cosine_overlap
  - rna_thr, ab_thr, rna_frac_above_thr, ab_frac_above_thr
  - manders_m1, manders_m2
  - jaccard, dice, both_frac
  - ab_enrich_in_rna_high, rna_enrich_in_ab_high
  - sum_r, sum_a, sum_product, sum_min, min_frac_r, min_frac_a
"""
from __future__ import annotations

from typing import Optional, Dict

import numpy as np
from scipy import stats as _stats

from . import thresholds as _thr


def _empty_metrics(n_pix: int) -> Dict[str, float]:
    return dict(
        n_pix=n_pix,
        rna_mean=0.0, ab_mean=0.0,
        pearson_r=0.0, spearman_rho=0.0, li_icq=0.0, cosine_overlap=0.0,
        rna_thr=float("inf"), ab_thr=float("inf"),
        rna_frac_above_thr=0.0, ab_frac_above_thr=0.0,
        manders_m1=0.0, manders_m2=0.0,
        jaccard=0.0, dice=0.0, both_frac=0.0,
        ab_enrich_in_rna_high=0.0, rna_enrich_in_ab_high=0.0,
        sum_r=0.0, sum_a=0.0, sum_product=0.0,
        sum_min=0.0, min_frac_r=0.0, min_frac_a=0.0,
    )


def compute_coloc_metrics(
    rvals: np.ndarray,
    avals: np.ndarray,
    *,
    thr_mode: str = "mad",
    k_mad: float = 2.0,
    percentile: float = 80.0,
    r_thr_override: Optional[float] = None,
    a_thr_override: Optional[float] = None,
) -> Dict[str, float]:
    """Compute per-nucleus colocalization metrics on two 1-D pixel arrays.

    Bit-identical (within float64) to Coloc_Analysis.compute_nucleus_coloc_metrics.
    """
    r = np.asarray(rvals, dtype=np.float64).ravel()
    a = np.asarray(avals, dtype=np.float64).ravel()
    n = int(r.size)
    if n < 10:
        return _empty_metrics(n)

    r_sum = float(r.sum())
    a_sum = float(a.sum())
    r_mean = r_sum / n
    a_mean = a_sum / n

    dr = r - r_mean
    da = a - a_mean
    num = float(np.dot(dr, da))
    r_d2 = float(np.dot(dr, dr))
    a_d2 = float(np.dot(da, da))
    dot = float(np.dot(r, a))
    r2 = float(np.dot(r, r))
    a2 = float(np.dot(a, a))
    sum_min = float(np.minimum(r, a).sum())
    icq_pos = int(((dr * da) > 0).sum())

    pearson = (num / np.sqrt(r_d2 * a_d2)) if (r_d2 > 0 and a_d2 > 0) else 0.0
    cosine = (dot / np.sqrt(r2 * a2)) if (r2 > 0 and a2 > 0) else 0.0
    li_icq = (icq_pos / float(n)) - 0.5

    # Spearman via scipy (uses fractional ranks with tie-averaging — same as Fiji)
    if r.std() > 0 and a.std() > 0:
        spearman = float(_stats.spearmanr(r, a).statistic)
    else:
        spearman = 0.0

    # Thresholds
    if r_thr_override is not None and a_thr_override is not None:
        r_thr = float(r_thr_override)
        a_thr = float(a_thr_override)
    elif thr_mode == "costes":
        r_thr, a_thr = _thr.costes_threshold(r.tolist(), a.tolist())
    else:
        r_thr = _thr.coloc_threshold(r.tolist(), mode=thr_mode, k_mad=k_mad, percentile=percentile)
        a_thr = _thr.coloc_threshold(a.tolist(), mode=thr_mode, k_mad=k_mad, percentile=percentile)

    Rpos = r >= r_thr
    Apos = a >= a_thr
    both = int((Rpos & Apos).sum())
    r_pos = int(Rpos.sum())
    a_pos = int(Apos.sum())

    sumR_where_Apos = float(r[Apos].sum())
    sumA_where_Rpos = float(a[Rpos].sum())

    union = r_pos + a_pos - both
    jacc = (both / float(union)) if union > 0 else 0.0
    dice = (2.0 * both / float(r_pos + a_pos)) if (r_pos + a_pos) > 0 else 0.0
    both_frac = both / float(n)

    m1 = (sumR_where_Apos / r_sum) if r_sum > 0 else 0.0
    m2 = (sumA_where_Rpos / a_sum) if a_sum > 0 else 0.0

    rna_frac_above = r_pos / float(n)
    ab_frac_above = a_pos / float(n)

    ab_hi = float(a[Rpos].mean()) if r_pos > 0 else 0.0
    ab_lo = float(a[~Rpos].mean()) if (n - r_pos) > 0 else 0.0
    r_hi = float(r[Apos].mean()) if a_pos > 0 else 0.0
    r_lo = float(r[~Apos].mean()) if (n - a_pos) > 0 else 0.0

    ab_enrich = (ab_hi / ab_lo) if ab_lo > 0 else (ab_hi if ab_hi > 0 else 0.0)
    r_enrich = (r_hi / r_lo) if r_lo > 0 else (r_hi if r_hi > 0 else 0.0)

    min_frac_r = (sum_min / r_sum) if r_sum > 0 else 0.0
    min_frac_a = (sum_min / a_sum) if a_sum > 0 else 0.0

    return dict(
        n_pix=n,
        rna_mean=r_mean,
        ab_mean=a_mean,
        pearson_r=float(pearson),
        spearman_rho=float(spearman),
        li_icq=float(li_icq),
        cosine_overlap=float(cosine),
        rna_thr=float(r_thr),
        ab_thr=float(a_thr),
        rna_frac_above_thr=float(rna_frac_above),
        ab_frac_above_thr=float(ab_frac_above),
        manders_m1=float(m1),
        manders_m2=float(m2),
        jaccard=float(jacc),
        dice=float(dice),
        both_frac=float(both_frac),
        ab_enrich_in_rna_high=float(ab_enrich),
        rna_enrich_in_ab_high=float(r_enrich),
        sum_r=float(r_sum),
        sum_a=float(a_sum),
        sum_product=float(dot),
        sum_min=float(sum_min),
        min_frac_r=float(min_frac_r),
        min_frac_a=float(min_frac_a),
    )
