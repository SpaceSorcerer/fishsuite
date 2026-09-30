"""Well-mean superplot C panels for one condition: per-nucleus violin/column, one same-shape circle per well
(marker rule: same condition -> one marker and colour), mean-of-well-means bar, nucleus-unit Wilcoxon with
Holm stars and rank-biserial r. Ported from fig1lib.c_superplots (FIG1_IMAGING_v2 round 2, 2026-09-28)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .stats import run_tests
from .style import COL, NULLC, FILT, footer, stars, fmt_p, well_mean_style
from . import focus as _focus


def lab(t):
    return f"{stars(t['p_holm'])}\nHolm p = {fmt_p(t['p_holm'])}, n = {t['n']}\nr_rb = {t['rank_biserial']:.2f}"


def base_ax(w=2.3, h=3.2):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([.32, .33, .54, .54])
    return fig, ax


def violin(ax, x, v, color, alpha, edge):
    vp = ax.violinplot([v], positions=[x], widths=.7, showextrema=False)
    for b in vp['bodies']:
        b.set_facecolor(color); b.set_alpha(alpha); b.set_edgecolor(edge); b.set_linewidth(.5)


def _tag(arm):
    return arm.replace(" ", "")


def plot_one_sample(nv, fields, key, col, chance, ylab, shape, t, dots, rng, saver, arm="VPR noDox", suffix="_holm",
                    focus=False):
    """focus=True: y-window by the focus rule (fishsuite.figures.focus); nuclei outside are pinned and counted."""
    fig, ax = base_ax()
    v = nv[col].dropna().values
    wm = [nv.loc[nv.field == f, col].mean() for f in fields]
    fw = None
    if focus:
        frac = col.startswith("frac") or "UPP" in key
        dist = {arm: v} if dots or shape == "violin" else {}
        inc = list(wm) + [np.mean(wm), chance] + ([0.0] if shape != "violin" else [])
        fw = _focus.focus_window(dist, inc, "fraction" if frac else None, _focus.axes_height_pt(ax),
                                 _focus.text_band_pt(lab(t), 5.8), marker_pt=_WELL_R_PT)
        fw.n_above, fw.n_below = ({arm: int((v > fw.data_hi).sum())}, {arm: int((v < fw.data_lo).sum())})
    if shape == "violin":
        violin(ax, 0, v, COL, .15, COL)
        if fw is not None:
            _focus.clip_to_band(ax.collections[-1:], ax, fw)
    else:
        ax.bar(0, np.mean(wm), width=.55, color=COL, alpha=.15, edgecolor=COL, lw=.8, zorder=1)
    if dots and fw is not None:
        jx = rng.uniform(-.17, .17, len(v))
        ins = _focus.split_pinned(v, fw)[0]
        ax.scatter(jx[ins], v[ins], s=5, facecolor=COL, alpha=.45, edgecolor='black', lw=.25, zorder=6)
        _focus.draw_pinned(ax, jx, v, fw, COL)
    elif dots:
        ax.scatter(rng.uniform(-.17, .17, len(v)), v, s=5, facecolor=COL, alpha=.45, edgecolor='black', lw=.25, zorder=6)
    for k, mm in enumerate(wm):
        ax.scatter([(k - .5) * .2], [mm], **well_mean_style(arm, fields[k]))
    ax.hlines(np.mean(wm), -.32, .32, color='black', lw=1.2, zorder=4)
    ax.axhline(chance, color='#555555', ls='--', lw=.8, zorder=0)
    ax.text(1.01, chance, f"chance\n{chance:.3g}" if chance else "0", transform=ax.get_yaxis_transform(), fontsize=5.5,
            va='center', ha='left', color='#333333', clip_on=False)
    if fw is not None:
        ax.set_ylim(*fw.ylim)
        pt = (fw.ylim[1] - fw.ylim[0]) / _focus.axes_height_pt(ax)
        ax.text(0, fw.bracket_y - 2 * pt, lab(t), ha='center', va='bottom', fontsize=5.8)
        if fw.kind == "fraction":
            _focus.fraction_ticks(ax)
        _focus.draw_note(ax, fw, "nuclei", shown=dots)
    else:
        vals = v if dots or shape == "violin" else np.array(wm)
        lo, hi = min(vals.min(), chance), max(vals.max(), chance)
        span = hi - lo
        top = hi + (.08 if dots or shape == 'violin' else .16) * span
        ax.text(0, top, lab(t), ha='center', va='bottom', fontsize=5.8)
        ax.set_ylim(lo - .08 * span if chance == 0 else max(0, lo - .08 * span), top + .45 * span)
        if col.startswith("frac") or "UPP" in key:
            ax.set_ylim(0, max(1.0, top + .45 * span) if dots else ax.get_ylim()[1])
            ax.set_yticks([x for x in ax.get_yticks() if 0 <= x <= 1.0001])
    ax.set_xlim(-.6, .6); ax.set_xticks([0]); ax.set_xticklabels([arm]); ax.set_ylabel(ylab)
    footer(fig, [f"n = nuclei; {len(fields)} wells (fields {', '.join(fields)}). Circles = well means; bar = mean of well means"
                 + ("; small dots = nuclei" if dots else "") + f"; {'violin' if shape == 'violin' else 'column'} = per-nucleus distribution. "
                 f"One-sample Wilcoxon vs {chance:.3g}; stars from Holm across the six C tests; r_rb = matched-pairs rank-biserial; "
                 f"median diff = {t['median_diff']:.3g}.", FILT], width=58)
    stem = f"{key}{suffix}_{_tag(arm)}_{'dots' if dots else 'nodots'}"
    saver.save(fig, stem)
    return stem


_WELL_R_PT = float(np.sqrt(well_mean_style("")["s"]) / 2 + well_mean_style("")["lw"] / 2)


def plot_paired(nv, fields, key, col, ncol, ylab, t, dots, rng, saver, arm="VPR noDox", suffix="_holm"):
    d = nv[[col, ncol, "field"]].dropna()
    fig, ax = base_ax(2.6)
    for x, c, color in ((0, ncol, NULLC), (1, col, COL)):
        v = d[c].values
        violin(ax, x, v, color, .35 if color == NULLC else .15, '#777777' if color == NULLC else COL)
        if dots:
            ax.scatter(x + rng.uniform(-.17, .17, len(v)), v, s=5, facecolor=color, alpha=.6, edgecolor='black', lw=.25, zorder=6)
    wn = [d.loc[d.field == f, ncol].mean() for f in fields]
    wo = [d.loc[d.field == f, col].mean() for f in fields]
    for k in range(len(fields)):
        off = (k - .5) * .2
        ax.plot([off, 1 + off], [wn[k], wo[k]], color='black', lw=.8, zorder=4)
        ax.scatter([off], [wn[k]], **well_mean_style(arm, fields[k], color=NULLC))
        ax.scatter([1 + off], [wo[k]], **well_mean_style(arm, fields[k]))
    for x, mm in ((0, np.mean(wn)), (1, np.mean(wo))):
        ax.hlines(mm, x - .32, x + .32, color='black', lw=1.2, zorder=4)
    ax.set_ylim(0, 1.45)
    ax.plot([0, 0, 1, 1], [1.06, 1.08, 1.08, 1.06], color='black', lw=.75)
    ax.text(.5, 1.09, lab(t), ha='center', va='bottom', fontsize=5.8)
    ax.set_yticks(np.arange(0, 1.01, .2))
    ax.set_xlim(-.6, 1.6); ax.set_xticks([0, 1]); ax.set_xticklabels(["null", "observed"]); ax.set_ylabel(ylab); ax.set_xlabel(arm)
    footer(fig, [f"n = nuclei; {len(fields)} wells (fields {', '.join(fields)}). Circles = well means (lines join each well's null and observed); "
                 "bar = mean of well means" + ("; small dots = nuclei" if dots else "") + "; violin = per-nucleus distribution. "
                 "Null = mean over 200 uniform exact-footprint placements (seed 0). Paired Wilcoxon signed-rank within nucleus; stars from Holm "
                 f"across the six C tests; r_rb = matched-pairs rank-biserial; median diff (obs - null) = {t['median_diff']:.3g}.", FILT], width=66)
    stem = f"{key}{suffix}_{_tag(arm)}_{'dots' if dots else 'nodots'}"
    saver.save(fig, stem)
    return stem


def stats_table(T, fields) -> pd.DataFrame:
    rows = [dict(panel=k, test=t["test"], comparison=t["comparison"], n_nuclei=t["n"], median_diff=t["median_diff"],
                 rank_biserial=t["rank_biserial"], p_raw=t["p"], p_holm=t["p_holm"], stars_holm=stars(t["p_holm"]),
                 **{f"field{f}_{q}": t[f"well_{f}"][q] for f in fields for q in ("n", "p", "median_diff", "rank_biserial")})
            for k, t in T.items()]
    return pd.DataFrame(rows)


def build_c_panels(nv, fields, saver, arm="VPR noDox", seed=0, render=True, focus=False):
    """Run the six tests, Holm-adjust, render dots/nodots for each; returns the stats DataFrame.
    focus=True renders the one-sample panels in focus mode with stems '<key>_holm_focus_...' (paired panels keep
    their fixed 0-1.45 fraction axis and are not re-rendered)."""
    rng = np.random.default_rng(seed)
    T, one, pair = run_tests(nv, fields)
    if render:
        for key, col, chance, ylab, shape in one:
            for dots in (True, False):
                if focus:
                    plot_one_sample(nv, fields, key, col, chance, ylab, shape, T[key], dots, rng, saver, arm,
                                    suffix="_holm_focus", focus=True)
                else:
                    plot_one_sample(nv, fields, key, col, chance, ylab, shape, T[key], dots, rng, saver, arm)
        for key, col, ncol, ylab in (pair if not focus else ()):
            for dots in (True, False):
                plot_paired(nv, fields, key, col, ncol, ylab, T[key], dots, rng, saver, arm)
    return stats_table(T, fields)
