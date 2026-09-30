"""Two-condition (control vs perturbation) rendering for the locked figure-panel module (2026-09-28).

Render only: every value and every p value is supplied by the caller (read from an existing stats table);
nothing here fits a model or runs a test. Same locked look as the basal C/D/R/A panels:
violin of per-nucleus values, optional per-nucleus dots drawn over the well means, one 'o' circle per
biological unit in the condition colour (well_mean_style), a black line at the MEAN of the unit means,
significance bracket above all data. LUTs via style.colorize / check_luts (never green).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.stats import pearsonr, spearmanr

from .style import colorize, footer, fmt_p, stars, well_mean_style, check_luts

NOT_DETECTED = "not detected at this n"


def sig_label(p: float | None) -> str:
    if p is None or not np.isfinite(p):
        return ""
    return f"{stars(p)}  p = {fmt_p(p)}" if p < 0.05 else f"{NOT_DETECTED}  (p = {fmt_p(p)})"


def plot_two_arm(nuc: pd.DataFrame, units: pd.DataFrame, arms, colors: dict, ylab: str, title: str,
                 p: float | None, footer_lines, saver, stem: str, dots: bool, seed: int = 0,
                 ylim=None, chance: float | None = None, chance_label: str = "chance"):
    """nuc: columns arm, value (per nucleus; may be empty for an arm). units: columns arm, value (one row per
    biological unit, e.g. well). Returns the dict of plotted numbers."""
    rng = np.random.default_rng(seed)
    fig = plt.figure(figsize=(2.6, 3.4))
    ax = fig.add_axes([.27, .30, .58, .56])
    allv = []
    out = {}
    for x, arm in enumerate(arms):
        col = colors[arm]
        v = nuc.loc[nuc.arm == arm, "value"].to_numpy(float)
        v = v[np.isfinite(v)]
        u = units.loc[units.arm == arm, "value"].to_numpy(float)
        if len(v) > 1 and np.ptp(v) > 0:
            vp = ax.violinplot([v], positions=[x], widths=.75, showextrema=False)
            for b in vp['bodies']:
                b.set_facecolor(col); b.set_alpha(.18); b.set_edgecolor(col); b.set_linewidth(.6)
        if dots and len(v):
            ax.scatter(x + rng.uniform(-.18, .18, len(v)), v, s=5, facecolor=col, alpha=.45, edgecolor='black',
                       lw=.25, zorder=6)
        offs = np.linspace(-.12, .12, len(u)) if len(u) > 1 else np.zeros(len(u))
        for k, mm in enumerate(u):
            ax.scatter([x + offs[k]], [mm], **well_mean_style(arm, color=col))
        if len(u):
            ax.hlines(np.mean(u), x - .3, x + .3, color='black', lw=1.2, zorder=7)
        allv += list(v if (dots or len(v)) else []) + list(u)
        out[arm] = dict(n_nuclei=int(len(v)), n_units=int(len(u)), mean_of_unit_means=float(np.mean(u)) if len(u) else np.nan)
    if chance is not None:
        ax.axhline(chance, color='#555555', ls='--', lw=.8, zorder=0)
        ax.text(1.01, chance, f"{chance_label}\n{chance:.3g}", transform=ax.get_yaxis_transform(), fontsize=5.5,
                va='center', ha='left', color='#333333', clip_on=False)
        allv.append(chance)
    allv = np.asarray(allv, float)
    lo, hi = float(np.nanmin(allv)), float(np.nanmax(allv))
    span = (hi - lo) or abs(hi) or 1.0
    if ylim is None:
        ylim = (0.0 if lo >= 0 else lo - .06 * span, hi + .32 * span)
    ax.set_ylim(*ylim)
    s = sig_label(p)
    if s:
        yb = hi + .10 * span
        ax.plot([0, 0, 1, 1], [yb - .02 * span, yb, yb, yb - .02 * span], color='black', lw=.75)
        ax.text(.5, yb + .015 * span, s, ha='center', va='bottom', fontsize=6)
    ax.set_xlim(-.6, len(arms) - .4)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(arms, fontsize=6.5)
    for t, arm in zip(ax.get_xticklabels(), arms):
        t.set_color(colors[arm])
    ax.set_ylabel(ylab, fontsize=7)
    fig.text(.5, .955, title, ha='center', va='center', fontsize=7.5, fontweight='bold')
    footer(fig, list(footer_lines), width=64)
    saver.save(fig, stem)
    out["ylim"] = [float(ylim[0]), float(ylim[1])]
    return out


def fov_panel(planes: dict, lv: dict, px_um: float, title: str, footer_lines, saver, stem: str, scalebar_um: float = 20):
    check_luts()
    img = colorize(planes, lv=lv)
    fig = plt.figure(figsize=(3.4, 3.75), facecolor='white')
    ax = fig.add_axes([.02, .1, .96, .83]); ax.imshow(img, interpolation='nearest'); ax.set_axis_off()
    h, w = img.shape[:2]; L = scalebar_um / px_um; x1 = w * .96; y = h * .95
    ax.plot([x1 - L, x1], [y, y], color='white', lw=2, solid_capstyle='butt')
    ax.text(x1 - L / 2, y - h * .025, f"{scalebar_um:g} µm", color='white', ha='center', va='bottom', fontsize=6)
    fig.text(.5, .965, title, ha='center', fontsize=7)
    footer(fig, list(footer_lines))
    saver.save(fig, stem)


def crop_panel(planes: dict, lv: dict, px_um: float, box, title: str, footer_lines, saver, stem: str,
               mask=None, spots_xy=None, scalebar_um: float = 2):
    """box = (y0, y1, x0, x1); spots_xy = iterable of (x, y, radius_px) in full-image coordinates."""
    check_luts()
    y0, y1, x0, x1 = box
    img = colorize({c: a[y0:y1, x0:x1] for c, a in planes.items()}, lv=lv)
    fig = plt.figure(figsize=(2.6, 3.2), facecolor='white')
    ax = fig.add_axes([.04, .2, .92, .70]); ax.imshow(img, interpolation='nearest'); ax.set_axis_off()
    if mask is not None:
        ax.contour(mask[y0:y1, x0:x1], levels=[.5], colors='white', linewidths=.4)
    for (sx, sy, r) in (spots_xy or []):
        ax.add_patch(Circle((sx - x0, sy - y0), r, fill=False, ec='white', lw=.45))
    h_, w_ = img.shape[:2]; L = scalebar_um / px_um
    ax.plot([w_ * .95 - L, w_ * .95], [h_ * .95] * 2, color='white', lw=2, solid_capstyle='butt')
    ax.text(w_ * .95 - L / 2, h_ * .92, f"{scalebar_um:g} µm", color='white', ha='center', va='bottom', fontsize=6)
    fig.text(.5, .95, title, ha='center', va='center', fontsize=6.5)
    footer(fig, list(footer_lines), width=70)
    saver.save(fig, stem)


def scatter_one(x, y, color: str, xl: str, yl: str, title: str, pooled: bool, footer_lines, saver, stem: str,
                n_nuclei: int | None = None, xlim=None, ylim=None):
    """One condition per file. Pooled-puncta panels print Spearman + Pearson and say 'pooled'."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
    fig = plt.figure(figsize=(2.9, 3.3)); ax = fig.add_axes([.24, .28, .7, .5])
    ax.scatter(x, y, s=8, facecolor=color, alpha=.4, edgecolor='black', lw=.25)
    rs = spearmanr(x, y); rp = pearsonr(x, y)
    if pooled:
        t = (f"Spearman ρ = {rs.statistic:.2f} (p = {fmt_p(rs.pvalue)}); Pearson r = {rp.statistic:.2f} (p = {fmt_p(rp.pvalue)})\n"
             f"pooled across {len(x)} puncta from {n_nuclei} nuclei")
    else:
        t = (f"Spearman ρ = {rs.statistic:.2f} (p = {fmt_p(rs.pvalue)}); Pearson r = {rp.statistic:.2f} (p = {fmt_p(rp.pvalue)})\n"
             f"n = {len(x)} nuclei (one point per nucleus)")
    ax.set_title(t, fontsize=5.6)
    fig.text(.5, .965, title, ha='center', va='center', fontsize=7, fontweight='bold')
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    footer(fig, list(footer_lines), width=62)
    saver.save(fig, stem)
    return dict(panel=stem, x=xl, y=yl, unit="puncta pooled across nuclei" if pooled else "nuclei", n=int(len(x)),
                n_nuclei=n_nuclei if pooled else int(len(x)), spearman_rho=float(rs.statistic), spearman_p=float(rs.pvalue),
                pearson_r=float(rp.statistic), pearson_p=float(rp.pvalue))
