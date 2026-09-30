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
from . import focus as _focus

NOT_DETECTED = "not detected at this n"


def sig_label(p: float | None, unit: str | None = None) -> str:
    """unit (optional, e.g. 'slide-adjusted, 9 v 9 wells') is appended to the p text; default output unchanged."""
    if p is None or not np.isfinite(p):
        return ""
    if unit:
        return (f"{stars(p)}  p = {fmt_p(p)}\n{unit}" if p < 0.05
                else f"{NOT_DETECTED}\np = {fmt_p(p)}; {unit}")
    return f"{stars(p)}  p = {fmt_p(p)}" if p < 0.05 else f"{NOT_DETECTED}  (p = {fmt_p(p)})"


def plot_two_arm(nuc: pd.DataFrame, units: pd.DataFrame, arms, colors: dict, ylab: str, title: str,
                 p: float | None, footer_lines, saver, stem: str, dots: bool, seed: int = 0,
                 ylim=None, chance: float | None = None, chance_label: str = "chance",
                 sig_unit: str | None = None, focus: bool = False, focus_kind: str | None = None,
                 focus_unit: str = "nuclei"):
    """nuc: columns arm, value (per nucleus; may be empty for an arm). units: columns arm, value (one row per
    biological unit, e.g. well). Returns the dict of plotted numbers.
    focus=True: y-window by the focus rule (fishsuite.figures.focus; focus_kind None | 'ratio' | 'fraction');
    out-of-window nuclei are pinned as open markers and counted in a separate note. focus=False = unchanged output."""
    if focus and ylim is not None:
        raise ValueError("plot_two_arm: pass either ylim or focus=True, not both")
    rng = np.random.default_rng(seed)
    fig = plt.figure(figsize=(2.6, 3.4))
    ax = fig.add_axes([.27, .30, .58, .56])
    allv = []
    allu = []
    out = {}
    fw = None
    if focus:
        s_ = sig_label(p, sig_unit)
        fs_ = 6 if not sig_unit else 5.5
        vals = {a: nuc.loc[nuc.arm == a, "value"].to_numpy(float) for a in arms}
        inc = [units.loc[units.arm == a, "value"].to_numpy(float) for a in arms]
        inc = list(np.concatenate(inc)) + [np.mean(i) for i in inc if len(i)] + ([chance] if chance is not None else [])
        fw = _focus.focus_window(vals, inc, focus_kind, _focus.axes_height_pt(ax),
                                 _focus.text_band_pt(s_, fs_, 1.1), marker_pt=_WELL_R_PT)
        bodies = []
    for x, arm in enumerate(arms):
        col = colors[arm]
        v = nuc.loc[nuc.arm == arm, "value"].to_numpy(float)
        v = v[np.isfinite(v)]
        u = units.loc[units.arm == arm, "value"].to_numpy(float)
        if len(v) > 1 and np.ptp(v) > 0:
            vp = ax.violinplot([v], positions=[x], widths=.75, showextrema=False)
            for b in vp['bodies']:
                b.set_facecolor(col); b.set_alpha(.18); b.set_edgecolor(col); b.set_linewidth(.6)
            if fw is not None:
                bodies += vp['bodies']
        if dots and len(v) and fw is not None:
            jx = x + rng.uniform(-.18, .18, len(v))
            ins = _focus.split_pinned(v, fw)[0]
            ax.scatter(jx[ins], v[ins], s=5, facecolor=col, alpha=.45, edgecolor='black', lw=.25, zorder=6)
            _focus.draw_pinned(ax, jx, v, fw, col)
        elif dots and len(v):
            ax.scatter(x + rng.uniform(-.18, .18, len(v)), v, s=5, facecolor=col, alpha=.45, edgecolor='black',
                       lw=.25, zorder=6)
        offs = np.linspace(-.12, .12, len(u)) if len(u) > 1 else np.zeros(len(u))
        for k, mm in enumerate(u):
            ax.scatter([x + offs[k]], [mm], **well_mean_style(arm, color=col))
        if len(u):
            ax.hlines(np.mean(u), x - .3, x + .3, color='black', lw=1.2, zorder=7)
        allv += list(v if (dots or len(v)) else []) + list(u)
        allu += list(u)
        out[arm] = dict(n_nuclei=int(len(v)), n_units=int(len(u)), mean_of_unit_means=float(np.mean(u)) if len(u) else np.nan)
    if chance is not None:
        ax.axhline(chance, color='#555555', ls='--', lw=.8, zorder=0)
        ax.text(1.01, chance, f"{chance_label}\n{chance:.3g}", transform=ax.get_yaxis_transform(), fontsize=5.5,
                va='center', ha='left', color='#333333', clip_on=False)
        allv.append(chance)
    allv = np.asarray(allv, float)
    lo, hi = float(np.nanmin(allv)), float(np.nanmax(allv))
    span = (hi - lo) or abs(hi) or 1.0
    if fw is not None:
        return _finish_focus(fig, ax, fw, bodies, arms, colors, ylab, title, p, sig_unit, footer_lines, saver, stem,
                             out, focus_unit)
    auto = ylim is None
    if auto:
        ylim = (0.0 if lo >= 0 else lo - .06 * span, hi + .32 * span)
    ax.set_ylim(*ylim)
    s = sig_label(p, sig_unit)
    R = ylim[1] - ylim[0]
    guard = (not auto) and hi > ylim[1] - .2 * R
    if guard:  # data run past a caller-supplied window: keep the bracket inside the axes
        hi, span = ylim[1] - .24 * R, .8 * R
    if s:
        yb = _bracket_y(ax, hi, span, allu, s, 6 if not sig_unit else 5.5, auto, guard)
        ylim = ax.get_ylim()
        ax.plot([0, 0, 1, 1], [yb - .02 * span, yb, yb, yb - .02 * span], color='black', lw=.75)
        ax.text(.5, yb + .015 * span, s, ha='center', va='bottom', fontsize=6 if not sig_unit else 5.5, linespacing=1.1)
    ax.set_xlim(-.6, len(arms) - .4)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(arms, fontsize=6.5)
    for t, arm in zip(ax.get_xticklabels(), arms):
        t.set_color(colors[arm])
    ax.set_ylabel(ylab, fontsize=7)
    fig.text(.5, .955, title, ha='center', va='center', fontsize=7.5, fontweight='bold')
    if footer_lines:
        footer(fig, list(footer_lines), width=64)
    saver.save(fig, stem)
    out["ylim"] = [float(ylim[0]), float(ylim[1])]
    return out


_WELL_R_PT = float(np.sqrt(well_mean_style("")["s"]) / 2 + well_mean_style("")["lw"] / 2)


def _bracket_y(ax, hi, span, units, s, fs, auto, guard):
    """Default-mode bracket bar y: above the data maximum (hi + 10 % of the data span) AND with its ticks clear of every
    well-mean circle; on an automatic axis the top is raised until the text fits. Unchanged from b62484e whenever
    neither constraint binds."""
    yb = hi + .10 * span
    if guard or not len(units):
        return yb
    umax = float(np.nanmax(units))
    for _ in range(6):
        y0, y1 = ax.get_ylim()
        per_pt = (y1 - y0) / _focus.axes_height_pt(ax)
        yb = max(hi + .10 * span, umax + (_WELL_R_PT + 1.5) * per_pt + .02 * span)
        need = yb + .015 * span + ((s.count("\n") + 1) * fs * 1.25 + 1.0) * per_pt   # text height + 1 pt
        if not auto or need <= y1:
            break
        ax.set_ylim(y0, need)
    return yb


def _finish_focus(fig, ax, fw, bodies, arms, colors, ylab, title, p, sig_unit, footer_lines, saver, stem, out, unit):
    """Focus-mode tail of plot_two_arm: window, clipped violins, bracket in its own band, pinned-count note."""
    ax.set_ylim(*fw.ylim)
    _focus.clip_to_band(bodies, ax, fw)
    s = sig_label(p, sig_unit)
    pt = (fw.ylim[1] - fw.ylim[0]) / _focus.axes_height_pt(ax)
    if s and len(arms) == 2:
        yb = fw.bracket_y
        ax.plot([0, 0, 1, 1], [yb - 2 * pt, yb, yb, yb - 2 * pt], color='black', lw=.75)
        ax.text(.5, yb + 1 * pt, s, ha='center', va='bottom', fontsize=6 if not sig_unit else 5.5, linespacing=1.1)
    if fw.kind == 'fraction':
        _focus.fraction_ticks(ax)
    _focus.draw_note(ax, fw, unit)
    ax.set_xlim(-.6, len(arms) - .4)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(arms, fontsize=6.5)
    for t, arm in zip(ax.get_xticklabels(), arms):
        t.set_color(colors[arm])
    ax.set_ylabel(ylab, fontsize=7)
    fig.text(.5, .955, title, ha='center', va='center', fontsize=7.5, fontweight='bold')
    if footer_lines:
        footer(fig, list(footer_lines), width=64)
    saver.save(fig, stem)
    out["ylim"] = [float(fw.ylim[0]), float(fw.ylim[1])]
    out["focus"] = dict(data_lo=fw.data_lo, data_hi=fw.data_hi, bracket_y=fw.bracket_y, kind=fw.kind,
                        n_above=dict(fw.n_above), n_below=dict(fw.n_below), n_out=fw.n_out)
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


# ---- additions 2026-09-29 (imaging sets KD/OE): radial profile, per-unit ratio scatter, pixel-exact micrograph ----

def radial_profile(prof: dict, arms, colors: dict, xlab: str, ylab: str, title: str, saver, stem: str,
                   ylim=None, ref: float | None = 1.0):
    """prof[arm] = dict(r=..., obs=..., lo=..., hi=..., ctrl=...), all supplied by the caller (render only).
    Observed = solid line in the condition colour (+ CI band when lo/hi are finite); random-placement control = dashed."""
    from matplotlib.lines import Line2D
    fig = plt.figure(figsize=(2.6, 3.4))
    ax = fig.add_axes([.27, .34, .58, .52])
    allv = []
    for arm in arms:
        d = prof[arm]; col = colors[arm]
        r = np.asarray(d["r"], float); o = np.asarray(d["obs"], float); c = np.asarray(d["ctrl"], float)
        lo, hi = d.get("lo"), d.get("hi")
        if lo is not None and hi is not None:
            lo = np.asarray(lo, float); hi = np.asarray(hi, float)
            if np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)):
                ax.fill_between(r, lo, hi, color=col, alpha=.18, lw=0); allv += list(lo) + list(hi)
        ax.plot(r, o, color=col, lw=1.3, marker='o', ms=2.6, mec='black', mew=.3)
        ax.plot(r, c, color=col, lw=1.0, ls='--')
        allv += list(o) + list(c)
    if ref is not None:
        ax.axhline(ref, color='#555555', ls=':', lw=.7, zorder=0); allv.append(ref)
    allv = np.asarray(allv, float)
    lo_, hi_ = float(np.nanmin(allv)), float(np.nanmax(allv)); span = (hi_ - lo_) or 1.0
    ax.set_ylim(*(ylim or (lo_ - .06 * span, hi_ + .12 * span)))
    ax.set_xlabel(xlab, fontsize=7); ax.set_ylabel(ylab, fontsize=7)
    h = [Line2D([], [], color=colors[a], lw=1.3) for a in arms] + [Line2D([], [], color='#555555', lw=1.0, ls='--')]
    ax.legend(h, [a.replace("\n", " ") for a in arms] + ["random placement (dashed)"], fontsize=5.5, frameon=False,
              loc='upper center', bbox_to_anchor=(.5, -.2), ncol=1)
    fig.text(.5, .955, title, ha='center', va='center', fontsize=7.5, fontweight='bold')
    saver.save(fig, stem)
    return dict(ylim=[float(v) for v in ax.get_ylim()])


def ratio_scatter(units: pd.DataFrame, arms, colors: dict, xl: str, yl: str, title: str, saver, stem: str,
                  chance_slope: float | None = None, xlim=None, ylim=None):
    """units: columns arm, x, y (one row per biological unit, e.g. well). One 'o' per unit in the condition colour,
    a dashed least-squares line fitted across all units (descriptive; no p drawn) and an optional dotted chance line
    y = chance_slope * x. Returns the fit."""
    from matplotlib.lines import Line2D
    fig = plt.figure(figsize=(2.9, 3.4)); ax = fig.add_axes([.24, .34, .68, .52])
    x = units["x"].to_numpy(float); y = units["y"].to_numpy(float)
    for arm in arms:
        u = units[units.arm == arm]
        ax.scatter(u.x, u.y, **dict(well_mean_style(arm, color=colors[arm]), s=60))
    b, a = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1])
    xmax = float(np.nanmax(x)) * 1.08
    xx = np.array([0.0, xmax])
    ax.plot(xx, a + b * xx, color='black', ls='--', lw=.9, zorder=3)
    if chance_slope is not None:
        ax.plot(xx, chance_slope * xx, color='#777777', ls=':', lw=.9, zorder=2)
    ax.set_xlim(*(xlim or (0, xmax)))
    ax.set_ylim(*(ylim or (0, max(float(np.nanmax(y)), float(a + b * xmax)) * 1.12)))
    ax.set_xlabel(xl, fontsize=7); ax.set_ylabel(yl, fontsize=7)
    h = [Line2D([], [], marker='o', ls='', mfc=colors[a_], mec='black', ms=6) for a_ in arms] + \
        [Line2D([], [], color='black', ls='--', lw=.9)] + \
        ([Line2D([], [], color='#777777', ls=':', lw=.9)] if chance_slope is not None else [])
    lab = [a_.replace("\n", " ") for a_ in arms] + ["least-squares fit"] + (["chance"] if chance_slope is not None else [])
    ax.legend(h, lab, fontsize=5.5, frameon=False, loc='upper center', bbox_to_anchor=(.5, -.2), ncol=2)
    fig.text(.5, .955, title, ha='center', va='center', fontsize=7.5, fontweight='bold')
    saver.save(fig, stem)
    return dict(slope=float(b), intercept=float(a), pearson_r=r, r2=r * r, n_units=int(len(x)),
                xlim=[float(v) for v in ax.get_xlim()], ylim=[float(v) for v in ax.get_ylim()])


def micrograph(rgb8: np.ndarray, px_um: float, saver, stem: str, k: int = 1, labels=None,
               outline_color: str = '#D9D9D9', outline_lw: float = .5, scalebar_um: float = 10,
               count_text: str | None = None, strip_px: int = 120, outline_fields: bool = False):
    """Pixel-exact micrograph. rgb8 (uint8 HxWx3) is drawn at k output pixels per image pixel (integer k, so
    x-scale = y-scale and no resampling at 600 dpi). Scale bar and optional count text sit in a white strip BELOW
    the image, so image pixels are untouched; optional nuclear outlines are vector contours of ``labels``.
    Outlines are drawn only when ``labels`` holds a single nucleus (Brian 2026-09-30: never on whole fields);
    ``outline_fields=True`` overrides for a multi-nucleus field."""
    check_luts()
    rgb8 = np.asarray(rgb8)
    if rgb8.dtype != np.uint8 or rgb8.ndim != 3:
        raise ValueError("micrograph expects a uint8 HxWx3 array")
    h, w = rgb8.shape[:2]; k = int(k)
    W, H = w * k, h * k + strip_px
    def _inch(n):  # smallest float inch size that renders to exactly n pixels at 600 dpi (guards float truncation)
        v = n / 600
        while int(v * 600) < n:
            v = np.nextafter(v, np.inf)
        return v
    fig = plt.figure(figsize=(_inch(W), _inch(H)), dpi=600, facecolor='white')
    ax = fig.add_axes([0, strip_px / H, 1, h * k / H]); ax.set_axis_off()
    ax.imshow(rgb8, interpolation='nearest', extent=(0, w, h, 0))
    ax.set_xlim(0, w); ax.set_ylim(h, 0)
    n_nuc = 0 if labels is None else int(np.count_nonzero(np.unique(labels)))
    if labels is not None and (n_nuc == 1 or outline_fields):
        xs = np.arange(w) + .5; ys = np.arange(h) + .5
        for lab in np.unique(labels):
            if lab == 0:
                continue
            ax.contour(xs, ys, (labels == lab).astype(float), levels=[.5], colors=outline_color,
                       linewidths=outline_lw)
    sax = fig.add_axes([0, 0, 1, strip_px / H]); sax.set_axis_off(); sax.set_xlim(0, W); sax.set_ylim(0, strip_px)
    L = scalebar_um / px_um * k
    sax.plot([W - 20 - L, W - 20], [strip_px * .70] * 2, color='black', lw=1.5, solid_capstyle='butt')
    sax.text(W - 20 - L / 2, strip_px * .10, f"{scalebar_um:g} µm", ha='center', va='bottom', fontsize=5)
    if count_text:
        t = sax.text(20, strip_px * .45, count_text, ha='left', va='center', fontsize=5.5)
        t.set_gid(f"{stem}__count")
    saver.save(fig, stem)
    return dict(stem=stem, w_px=w, h_px=h, k=k, out_w=W, out_h=H, px_um_x=px_um, px_um_y=px_um,
                scale_x=k, scale_y=k)
