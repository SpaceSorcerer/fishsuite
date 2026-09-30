"""Focus mode (2026-09-30): a y-window that shrinks to the data that carry the point, never hiding means or statistics.

Rule (default when focus=True):
  * the data band contains every unit (well/sample) mean, the mean line, any reference/chance line, and the central 98 %
    (1st-99th percentile, order statistics, per arm) of the plotted per-point distribution; padded 8 % of its span;
  * the significance bracket and its text sit in a band ABOVE the data band, with marker-size headroom so the bracket
    never touches a well-mean circle (the band height is solved in points, so it is font/marker exact);
  * ratio / normalised metrics (reference = 1): 1 is always inside; the window is capped at max(2.5, rule) - the cap
    never cuts the rule, so in practice the rule decides;
  * fractions stay inside [0, 1] when every value does; non-negative data never get a negative axis;
  * points outside the data band are drawn pinned at its edge as small open markers, and a separate text object states
    "n above, n below axis". Violin bodies are clipped only outside the data band (the central 98 % is never cut).
Pure numpy + matplotlib; copied verbatim to the rnaseq-figure-style skill as templates/focus_axis.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FOCUS_Q = (1.0, 99.0)
FOCUS_PAD = 0.08
RATIO_CAP = 2.5
PIN_STYLE = dict(s=9, marker='o', facecolor='none', lw=.5, zorder=6, clip_on=False)
NOTE_GID = "focus_pinned_note"


@dataclass
class FocusWindow:
    ylim: tuple            # axis limits
    data_lo: float         # bottom of the data band (= ylim[0]); values below are pinned here
    data_hi: float         # top of the data band; values above are pinned here, the band above holds the bracket
    bracket_y: float       # y of the bracket bar (ticks hang down from it); text sits on top
    core: tuple            # unpadded (lo, hi) of means + central 98 % + must_include
    kind: str | None = None
    cap: float | None = None
    n_above: dict = field(default_factory=dict)
    n_below: dict = field(default_factory=dict)

    @property
    def n_out(self) -> int:
        return int(sum(self.n_above.values()) + sum(self.n_below.values()))


def central_range(v, q=FOCUS_Q):
    """(lo, hi) of the central range by order statistics (never interpolated), so >= q[1]-q[0] % of v lies inside."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return None
    return float(np.percentile(v, q[0], method='lower')), float(np.percentile(v, q[1], method='higher'))


def focus_window(values_by_arm: dict, must_include=(), kind: str | None = None, axes_height_pt: float = 100.0,
                 reserve_pt: float = 0.0, marker_pt: float = 6.0, pad: float = FOCUS_PAD, q=FOCUS_Q,
                 cap: float | None = None) -> FocusWindow:
    """values_by_arm: {arm: per-point values} (the plotted distribution). must_include: unit means, mean lines, chance.
    kind: None | 'ratio' | 'fraction'. reserve_pt: height of the bracket + text band (points); marker_pt: radius of the
    largest marker that can sit at the data top (well-mean circle) - the bracket clears it.
    cap: explicit hard top of the data band chosen by a person for one panel (e.g. 2.5 for an NT-normalised metric);
    it overrides the rule and MAY cut the central range - everything above is pinned and counted."""
    lo_c, hi_c = [], []
    for v in values_by_arm.values():
        r = central_range(v, q)
        if r is not None:
            lo_c.append(r[0]); hi_c.append(r[1])
    inc = [float(x) for x in np.ravel(np.asarray(list(must_include), float)) if np.isfinite(x)]
    if kind == 'ratio':
        inc.append(1.0)
    pool = lo_c + hi_c + inc
    if not pool:
        raise ValueError("focus_window: no finite values or means to frame")
    c_lo, c_hi = min(pool), max(pool)
    span = (c_hi - c_lo) or abs(c_hi) or 1.0
    d_lo, d_hi = c_lo - pad * span, c_hi + pad * span
    allv = np.concatenate([np.asarray(v, float).ravel() for v in values_by_arm.values()] + [np.asarray(inc, float)])
    allv = allv[np.isfinite(allv)]
    if kind == 'fraction' and len(allv) and allv.min() >= 0 and allv.max() <= 1:
        d_lo, d_hi = max(d_lo, 0.0), min(d_hi, 1.0)
    elif c_lo >= 0 and (not len(allv) or allv.min() >= 0):
        d_lo = max(d_lo, 0.0)
    if cap is not None:
        d_hi = float(cap)
        if d_hi <= d_lo:
            raise ValueError(f"focus_window: cap {cap} is not above the bottom of the data band {d_lo}")
    band = d_hi - d_lo
    head = marker_pt + 2.0                                  # well-mean radius + gap below the bracket ticks
    frac = min((head + reserve_pt) / max(axes_height_pt, 1.0), .6)
    W = band / (1 - frac)
    top = d_lo + W
    if kind == 'ratio':
        top = min(top, max(RATIO_CAP, top))                # documented cap; never binds against the rule
    bracket_y = d_hi + head * W / max(axes_height_pt, 1.0)
    fw = FocusWindow((d_lo, top), d_lo, d_hi, bracket_y, (c_lo, c_hi), kind, cap)
    for arm, v in values_by_arm.items():
        v = np.asarray(v, float); v = v[np.isfinite(v)]
        fw.n_above[arm] = int((v > d_hi).sum())
        fw.n_below[arm] = int((v < d_lo).sum())
    return fw


def text_band_pt(text: str, fontsize: float, linespacing: float = 1.2) -> float:
    """Height (pt) of the bracket ticks + a text block of len(lines) lines + top margin."""
    if not text:
        return 0.0
    return 2.0 + 1.5 + (text.count("\n") + 1) * fontsize * linespacing + 3.0


def axes_height_pt(ax) -> float:
    return float(ax.get_position().height * ax.figure.get_figheight() * 72)


def split_pinned(v, fw: FocusWindow):
    """Masks (inside, above, below) of v against the data band."""
    v = np.asarray(v, float)
    return (v >= fw.data_lo) & (v <= fw.data_hi), v > fw.data_hi, v < fw.data_lo


def draw_pinned(ax, x, v, fw: FocusWindow, color):
    """Pinned open markers at the data-band edge for points outside it; returns the number drawn."""
    x = np.asarray(x, float); v = np.asarray(v, float)
    _, up, dn = split_pinned(v, fw)
    if up.any():
        ax.scatter(x[up], np.full(up.sum(), fw.data_hi), edgecolor=color, **PIN_STYLE)
    if dn.any():
        ax.scatter(x[dn], np.full(dn.sum(), fw.data_lo), edgecolor=color, **PIN_STYLE)
    return int(up.sum() + dn.sum())


def clip_to_band(artists, ax, fw: FocusWindow, x0=-10.0, x1=10.0):
    """Clip violin bodies (or any patch collection) to the data band; the central 98 % lies inside by construction."""
    from matplotlib.patches import Rectangle
    for a in artists:
        a.set_clip_path(Rectangle((x0, fw.data_lo), x1 - x0, fw.data_hi - fw.data_lo, transform=ax.transData))


def pinned_note(fw: FocusWindow, unit: str = "points", shown: bool = True) -> str:
    """shown=False (points not drawn, e.g. a no-dots panel): the note says the out-of-band points are not shown."""
    a, b = sum(fw.n_above.values()), sum(fw.n_below.values())
    how = "drawn at edge" if shown else "(not shown)"
    if fw.cap is not None:
        return f"{a} {unit} above {fw.cap:g} {how}" + (f"; {b} below axis {how}" if b else "")
    return f"{a} {unit} above, {b} below axis " + ("(open markers)" if shown else "(not shown)")


def draw_note(ax, fw: FocusWindow, unit: str = "points", fontsize: float = 5.0, shown: bool = True):
    """Separate text object (gid 'focus_pinned_note') just above the axes, left-aligned."""
    t = ax.text(0.0, 1.015, pinned_note(fw, unit, shown), transform=ax.transAxes, fontsize=fontsize, ha='left',
                va='bottom', color='#333333', clip_on=False)
    t.set_gid(NOTE_GID)
    return t


def cap_axis(ax, fw: FocusWindow):
    """Hard-cap display: the y spine and ticks stop at the cap; the bracket band floats above it."""
    ax.spines['left'].set_bounds(fw.ylim[0], fw.cap)
    ax.set_yticks([t for t in ax.get_yticks() if fw.ylim[0] - 1e-9 <= t <= fw.cap + 1e-9])
    ax.set_ylim(*fw.ylim)


def fraction_ticks(ax):
    ax.set_yticks([t for t in ax.get_yticks() if -1e-9 <= t <= 1.0001])
