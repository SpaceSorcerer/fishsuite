"""Shared SuperPlot drawer for the post-run analysis subcommands.

Brian's group comparisons are drawn with ONE locked SuperPlot recipe
(``analysis.single_condition_plots._superplot_into_axes`` in the main
image-analysis-pipeline tree): per-unit dots jittered around each condition,
LARGE per-replicate MEAN markers (means = circles) with a dark edge, a violin
backdrop, and Welch-t significance brackets vs the left-most (reference)
condition. Okabe-Ito palette; NT / control = orange (#E69F00), the perturbation
= blue (#0072B2).

This module is a thin, import-safe bridge:

* :func:`get_locked_drawer` tries to import the LOCKED drawer (the canonical
  look). It searches, in order, (1) an already-importable ``analysis`` package,
  (2) ``$FISHSUITE_SUPERPLOT_PATH``, (3) the known lab default path. The import
  is LAZY (only when a figure is actually drawn) so importing fishsuite never
  depends on the external tree.
* :func:`superplot_into_axes` calls the locked drawer when available and
  otherwise falls back to a faithful, self-contained Okabe-Ito SuperPlot so the
  subcommands still render a correct figure on a machine without the locked tree
  (it never raises just because the external module is missing).

Nothing here is imported at package-import time by the CLI; the heavy plotting
imports live inside the functions.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np
import pandas as pd

# Okabe-Ito, colorblind-safe. Control/NT first (orange), perturbation second
# (blue), then green / vermillion / purple for any extra conditions.
OKABE_ITO: List[str] = ["#E69F00", "#0072B2", "#009E73", "#D55E00", "#CC79A7",
                        "#56B4E9", "#F0E442"]
CONTROL_COLOR = "#E69F00"   # NT / control -> orange
PERTURB_COLOR = "#0072B2"   # perturbation -> blue

# Default location of the locked drawer in Brian's lab (overridable via
# $FISHSUITE_SUPERPLOT_PATH). Kept out of the import path until a figure is
# actually requested.
_DEFAULT_LOCKED_PATH = r"F:\Image Analysis Work\image-analysis-pipeline\python"

_CONTROL_PATTERNS = ("nt", "control", "ctrl", "ctl", "scr", "scramble", "sic",
                     "si-c", "wt", "dmso", "veh", "vehicle", "mock", "untreated",
                     "none", "no dox", "nodox", "no-dox", "parent", "baseline")


def is_control_like(label: str) -> bool:
    """Heuristic: does ``label`` look like the control / non-targeting arm?

    Used only to decide draw ORDER (control-first) + the fallback color; it
    never alters which nuclei belong to which condition."""
    low = str(label).strip().lower()
    return any(low == p or low.startswith(p) for p in _CONTROL_PATTERNS)


def order_conditions_control_first(conditions: Sequence[str]) -> List[str]:
    """Return ``conditions`` with any control-like label(s) first, otherwise
    preserving the given order (stable)."""
    conds = list(dict.fromkeys(str(c) for c in conditions))  # de-dup, keep order
    ctrl = [c for c in conds if is_control_like(c)]
    rest = [c for c in conds if not is_control_like(c)]
    return ctrl + rest


def get_locked_drawer() -> Optional[Callable]:
    """Import and return the LOCKED ``_superplot_into_axes`` callable, or None.

    Never raises: a missing/broken external tree just yields None so the caller
    falls back to the vendored drawer."""
    # (1) already importable?
    try:
        from analysis.single_condition_plots import _superplot_into_axes  # type: ignore
        return _superplot_into_axes
    except Exception:
        pass
    # (2) env override, then (3) the lab default
    candidates = []
    env = os.environ.get("FISHSUITE_SUPERPLOT_PATH")
    if env:
        candidates.append(env)
    candidates.append(_DEFAULT_LOCKED_PATH)
    for cand in candidates:
        try:
            if cand and Path(cand).is_dir():
                if cand not in sys.path:
                    sys.path.insert(0, cand)
                from analysis.single_condition_plots import _superplot_into_axes  # type: ignore
                return _superplot_into_axes
        except Exception:
            continue
    return None


def _fallback_superplot(ax, df: pd.DataFrame, value_col: str, *, ylabel: str,
                        unit: str = "nucleus", pct: bool = False,
                        only_positive: bool = False,
                        condition_order: Optional[List[str]] = None,
                        annotate_stats: bool = True) -> bool:
    """Self-contained Okabe-Ito SuperPlot used only when the locked drawer is
    unavailable. Per-unit dots + LARGE per-replicate mean circles (dark edge) +
    violin backdrop + Welch-t stars vs the first condition. df needs columns
    ``condition``, ``image`` (= replicate id) and ``value_col``.
    """
    from scipy import stats as _st

    if not {value_col, "condition", "image"}.issubset(df.columns):
        return False
    d = df.copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d[d[value_col].notna()]
    if only_positive:
        d = d[d[value_col] > 0]
    if d.empty:
        return False

    conds = order_conditions_control_first(d["condition"].dropna().unique())
    if condition_order:
        want = [c for c in condition_order if c in set(conds)]
        conds = want + [c for c in conds if c not in set(want)]
    if not conds:
        return False

    rng = np.random.RandomState(20260704)
    group_rep_means: dict = {}
    for ci, cond in enumerate(conds):
        xi = ci + 1
        color = (CONTROL_COLOR if is_control_like(cond)
                 else OKABE_ITO[min(ci, len(OKABE_ITO) - 1)])
        if not is_control_like(cond) and ci == 1:
            color = PERTURB_COLOR
        sub = d[d["condition"] == cond]
        vals = sub[value_col].values
        n = len(vals)
        s_dot = 12 if n < 400 else (8 if n < 2000 else 4)
        a_dot = 0.18 if n < 400 else (0.10 if n < 2000 else 0.05)
        jx = rng.uniform(-0.26, 0.26, size=n)
        ax.scatter(np.full(n, xi) + jx, vals, s=s_dot, alpha=a_dot,
                   color=color, edgecolor="none", zorder=4)
        rep_means = []
        for rep, g in sub.groupby("image"):
            gv = g[value_col].dropna().values
            if gv.size:
                rep_means.append(float(np.mean(gv)))
        for m in rep_means:
            ax.scatter([xi], [m], s=200, facecolor=color, alpha=0.60,
                       edgecolor="#1f1f1f", linewidth=1.4, zorder=3)
        group_rep_means[cond] = rep_means
        try:
            vv = vals[np.isfinite(vals)]
            if vv.size > 1:
                parts = ax.violinplot([vv], positions=[xi], widths=0.85,
                                      showmeans=False, showmedians=False,
                                      showextrema=False)
                for pc in parts["bodies"]:
                    pc.set_facecolor(color)
                    pc.set_edgecolor("#202020")
                    pc.set_alpha(0.15)
                    pc.set_zorder(1)
        except Exception:
            pass

    ax.set_xticks(range(1, len(conds) + 1))
    ax.set_xticklabels(conds)
    ax.set_ylabel(ylabel)
    if pct:
        import matplotlib.ticker as mtick
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)

    # Welch-t on per-replicate means vs the first (reference) condition.
    if annotate_stats and len(conds) >= 2:
        ref = group_rep_means.get(conds[0], [])
        y0, y1 = ax.get_ylim()
        span = (y1 - y0) or 1.0
        top = y1
        for ci, cond in enumerate(conds[1:], start=1):
            other = group_rep_means.get(cond, [])
            if len(ref) >= 2 and len(other) >= 2:
                try:
                    p = float(_st.ttest_ind(ref, other, equal_var=False).pvalue)
                except Exception:
                    p = float("nan")
                star = ("***" if p < 0.001 else "**" if p < 0.01
                        else "*" if p < 0.05 else "n.s.")
                yb = top + span * 0.06 * ci
                ax.plot([1, ci + 1], [yb, yb], color="#333333", lw=1.0)
                ax.text((1 + ci + 1) / 2.0, yb, star, ha="center", va="bottom",
                        fontsize=9, color="#333333")
        ax.set_ylim(y0, top + span * (0.06 * len(conds) + 0.04))
    return True


def superplot_into_axes(ax, df: pd.DataFrame, value_col: str, *, ylabel: str,
                        unit: str = "nucleus", pct: bool = False,
                        only_positive: bool = False,
                        condition_order: Optional[List[str]] = None,
                        annotate_stats: bool = True) -> bool:
    """Draw a group-comparison SuperPlot into ``ax`` using the LOCKED drawer
    when available (canonical look, ``color_mode='by_condition'``) and a faithful
    Okabe-Ito fallback otherwise. Returns True if anything was drawn.

    df must carry ``condition`` (the group), ``image`` (the biological-replicate
    id) and ``value_col``.
    """
    drawer = get_locked_drawer()
    if drawer is not None:
        try:
            return bool(drawer(
                ax, df, value_col, ylabel=ylabel, unit=unit,
                only_positive=only_positive, pct=pct,
                condition_order=condition_order, annotate_stats=annotate_stats,
                color_mode="by_condition",
            ))
        except Exception:
            # locked drawer present but choked on this frame -> fall through
            pass
    return _fallback_superplot(
        ax, df, value_col, ylabel=ylabel, unit=unit, pct=pct,
        only_positive=only_positive, condition_order=condition_order,
        annotate_stats=annotate_stats,
    )
