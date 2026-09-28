"""Correlation scatters (D panels): one marker and colour; pooled-puncta panels print Spearman + Pearson and say
"pooled"; per-nucleus panels print Spearman. Ported from fig1lib.d_scatters (2026-09-28)."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

from .style import COL, FILT, footer, fmt_p

PER_NUC_NOTE = "Each dot = one punctum. Per-nucleus tests: D8 (within-nucleus ρ, punctum MIAT vs QKI) and C4 (nucleoplasm pixel ρ)."


def scat(stem, x, y, xl, yl, pooled, note, saver, n_nuc_p=None, rows=None, footer_prefix="VPR noDox, fields 14 + 15 combined"):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
    fig = plt.figure(figsize=(2.9, 3.2)); ax = fig.add_axes([.24, .3, .7, .52])
    ax.scatter(x, y, s=8, facecolor=COL, alpha=.4, edgecolor='black', lw=.25)
    rs = spearmanr(x, y); rp = pearsonr(x, y) if pooled else None
    if pooled:
        t = (f"Spearman ρ = {rs.statistic:.2f} (p = {fmt_p(rs.pvalue)}); Pearson r = {rp.statistic:.2f} (p = {fmt_p(rp.pvalue)})\n"
             f"pooled across {len(x)} puncta from {n_nuc_p} nuclei")
    else:
        t = f"Spearman ρ = {rs.statistic:.2f}, p = {fmt_p(rs.pvalue)}\nn = {len(x)} nuclei (one point per nucleus)"
    ax.set_title(t, fontsize=5.8); ax.set_xlabel(xl); ax.set_ylabel(yl)
    footer(fig, [footer_prefix + ". " + note, FILT], width=62)
    saver.save(fig, stem)
    rows = rows if rows is not None else []
    rows.append(dict(panel=stem, x=xl, y=yl.replace("\n", " "), unit=("puncta pooled across nuclei" if pooled else "nuclei"), n=len(x),
                     n_nuclei=n_nuc_p if pooled else len(x), spearman_rho=rs.statistic, spearman_p=rs.pvalue,
                     pearson_r=rp.statistic if pooled else np.nan, pearson_p=rp.pvalue if pooled else np.nan))
    return rows


def build_d_round2(nv, sv, saver, px_um=0.13, footer_prefix="VPR noDox, fields 14 + 15 combined"):
    """The four round-2 D panels (D1, D2 pooled puncta; D4, D5 per nucleus); returns the stats rows."""
    rows = []
    nn = sv.groupby(["image", "nucleus_id"]).ngroups
    kw = dict(saver=saver, rows=rows, footer_prefix=footer_prefix)
    scat("D1_punctum_MIAT_vs_footprint_area_pooled_r2", sv.footprint_area_px.values * px_um ** 2, sv.miat_footprint_mean_raw.values,
         "MIAT punctum footprint area (µm²)", "MIAT punctum intensity\n(footprint mean, raw a.u.)", True, PER_NUC_NOTE, n_nuc_p=nn, **kw)
    scat("D2_punctum_MIAT_vs_punctum_QKI_pooled_r2", sv.miat_footprint_mean_raw.values, sv.footprint_mean_qki.values,
         "MIAT punctum intensity (footprint mean, raw a.u.)", "QKI under punctum\n(footprint mean, raw a.u.)", True, PER_NUC_NOTE, n_nuc_p=nn, **kw)
    scat("D4_puncta_per_nucleus_vs_nuclear_MIAT_r2", nv.integrated_nuclear_miat.values, nv.n_miat_spots.values.astype(float),
         "Integrated nuclear MIAT (raw a.u.)", "MIAT puncta per nucleus", False, "Each dot = one nucleus.", n_nuc_p=nn, **kw)
    scat("D5_puncta_per_nucleus_vs_nuclear_QKI_r2", nv.mean_nuclear_qki.values, nv.n_miat_spots.values.astype(float),
         "Mean nuclear QKI (raw a.u.)", "MIAT puncta per nucleus", False, "Each dot = one nucleus.", n_nuc_p=nn, **kw)
    return rows
