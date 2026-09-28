"""Nucleus-unit tests for the C panels: two-sided Wilcoxon signed-rank, median difference, matched-pairs
rank-biserial r, Holm step-down across the C family. Ported unchanged from fig1lib.c_superplots (2026-09-28)."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata, wilcoxon

ONE = [("C1_mean_UPP", "mean_uniform_position_percentile_qki", 0.5, "Mean UPP score of QKI\nunder MIAT footprints", "violin"),
       ("C2_frac_UPP_ge_0p90", "frac_spots_upp_ge_0p90", None, "Fraction of MIAT puncta\nwith UPP ≥ 0.90", "bar"),
       ("C3_pearson_nucleoplasm", "pearson_r_nucleoplasm", 0.0, "Pearson r (nucleoplasm pixels)", "violin"),
       ("C4_spearman_nucleoplasm", "spearman_rho_nucleoplasm", 0.0, "Spearman ρ (nucleoplasm pixels)", "violin")]
PAIR = [("C5_footprint_QKIpos_obs_vs_null", "frac_miat_footprint_area_qki_pos", "null_mean_frac_miat_footprint_area_qki_pos",
         "QKI-positive fraction of\nMIAT footprint area"),
        ("C6_frac_puncta_QKIpos_obs_vs_null", "frac_miat_spots_qki_pos", "null_mean_frac_miat_spots_qki_pos",
         "Fraction of MIAT puncta\nQKI-positive")]


def wilcoxon_effect(d) -> dict:
    """Two-sided Wilcoxon signed-rank on d (non-finite dropped, zero_method='wilcox') + median(d) + rank-biserial."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    r = wilcoxon(d, alternative='two-sided', zero_method='wilcox')
    nz = d[d != 0]
    rk = rankdata(np.abs(nz))
    rp, rm = rk[nz > 0].sum(), rk[nz < 0].sum()
    return dict(p=float(r.pvalue), n=int(len(d)), W=float(r.statistic), median_diff=float(np.median(d)),
                rank_biserial=float((rp - rm) / (rp + rm)))


def holm(ps) -> np.ndarray:
    """Holm step-down adjusted p values (monotone, capped at 1), returned in input order."""
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    m = len(ps)
    out = np.empty(m)
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (m - i) * ps[j]))
        out[j] = run
    return out


def run_tests(nv, fields, one=None, pair=None):
    """The six C tests over nuclei (pooled wells) plus per-well tests; adds p_holm across the family.

    Returns (T, one, pair); T[key] also carries ``well_mean_<field>`` for the plotted well means.
    """
    one = [list(x) for x in (one or ONE)]
    for x in one:
        if x[2] is None:
            x[2] = float(nv.chance_frac_spots_upp_ge_0p90.iloc[0])
    pair = pair or PAIR
    T = {}
    for key, col, ch, *_ in one:
        T[key] = wilcoxon_effect(nv[col].values - ch) | dict(
            comparison=f"vs chance {ch:.4g}", test="one-sample Wilcoxon signed-rank",
            **{f"well_{f}": wilcoxon_effect(nv.loc[nv.field == f, col].values - ch) for f in fields},
            **{f"well_mean_{f}": float(nv.loc[nv.field == f, col].mean()) for f in fields})
    for key, col, ncol, _ in pair:
        d = nv[[col, ncol, "field"]].dropna()
        T[key] = wilcoxon_effect(d[col].values - d[ncol].values) | dict(
            comparison="observed - matched null", test="paired Wilcoxon signed-rank",
            **{f"well_{f}": wilcoxon_effect(d.loc[d.field == f, col].values - d.loc[d.field == f, ncol].values) for f in fields},
            **{f"well_mean_{f}": float(d.loc[d.field == f, col].mean()) for f in fields})
    for k, h in zip(list(T), holm([T[k]["p"] for k in T])):
        T[k]["p_holm"] = float(h)
    return T, one, pair
