"""Radial partner-enrichment profile — mean with a 95% confidence ribbon.

The 2-D analogue of a line scan, reported the way the line-scan literature
reports one. Where a line scan lays a fixed-length line through an object in
both channels, aligns the two profiles on their maximum, and averages a fixed
window either side of that maximum across objects, this module reads the
concentric-annulus profile ``rna_rna`` already computes around every detected
rna1 spot and averages it across objects the same way.

Two differences from a line scan, both in our favour and both worth stating in a
methods section:

* **No alignment step is needed, so no alignment bias is possible.** A line scan
  aligns on the observed maximum because the line's origin is arbitrary; that
  alignment can itself manufacture a peak from noise. Our rings are centred on
  the detected spot by construction, so the origin IS the object.
* **Each ring is an annulus, not a pair of points.** A ring averages every pixel
  at that distance rather than the two the line happens to cross, so it is
  isotropic and far less sensitive to which direction the line was drawn.

The ribbon is a 95% CONFIDENCE interval on the mean (t-based), not a standard
deviation and not a prediction interval: it answers "where is the mean ring
enrichment", which is the question a reader of this figure is asking.

The aggregation unit is one NUCLEUS per point (equal weight each), matching the
per-nucleus columns this reads. That deliberately differs from the pooled
spot-count-weighted profile in ``coloc_radial_profile.csv``, where a nucleus
with more spots counts for more; both are emitted, and the figure says which it
is drawing.

Two things this module must NOT do, both of which it did until 2026-08-10:

* **Pool conditions.** The runner hands it the run-wide concatenation of
  ``nuclei_metrics``. Averaging every condition into one ribbon on a preset with
  an over-expression arm, a non-targeting control and a no-probe control
  produces a number that describes no experiment, labelled as if it described
  the run. One panel per condition, always — with a single condition that
  degenerates to exactly one panel.
* **Average the no-primary / no-probe control in.** A ``secondary_only`` well is
  a background control, not a condition. It is drawn as its own explicitly
  labelled panel and excluded from every condition panel.

It must also agree with the table it visualises. With fixed-N sampling on and
``sampling.apply_to_rollups`` True, ``per_image_summary.csv``'s
``mean_<partner>_radial_enrichment_at_*`` / ``n_nuclei_in_*`` are restricted to
the sampled nuclei while ``nuclei_metrics`` keeps every visited one, so the
figure restricts too (``restrict_to_sampled``) and states on the chart which set
it drew.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ._superplot import OKABE_ITO, order_conditions_control_first

# Okabe-Ito, colorblind-safe. Vermillion is RESERVED for the no-enrichment
# reference line, so the per-condition palette excludes it — a condition drawn in
# the reference colour would be read as the reference.
_REFERENCE_COLOR = "#D55E00"
_CONDITION_COLORS: List[str] = [
    c for c in OKABE_ITO if c.upper() != _REFERENCE_COLOR.upper()
]
# The no-probe / no-primary control is deliberately achromatic: it is a floor to
# read the conditions against, not one more arm competing for attention.
_SECONLY_COLOR = "#666666"
# Retained for callers that drew the single-series figure before conditions were
# split out (the first palette entry is what a one-condition panel now uses).
_PROFILE_COLOR = _CONDITION_COLORS[0]

# ``<partner>_radial_enrichment_at_<0p25um>`` -> the 0p25um suffix.
_RING_COL_RE = re.compile(r"^(?P<partner>.+?)_radial_enrichment_at_(?P<suffix>[0-9p]+um)$")

# Per-nucleus verdict columns the figure must respect rather than average over.
SAMPLED_COL = "sampled_in_analysis"
SECONLY_COL = "secondary_only"
CONDITION_COL = "condition"

_SECONLY_LABEL = "secondary-only control (no probe)"


def _suffix_to_um(suffix: str) -> float:
    """``'0p25um'`` -> ``0.25``. Inverse of ``rna_rna._format_pair_um``."""
    return float(suffix[:-2].replace("p", "."))


def _bool_mask(series: pd.Series) -> np.ndarray:
    """Coerce a verdict column to a boolean mask without the ``astype`` trap.

    ``Series.astype(bool)`` on an object column reads every non-empty string as
    True — including ``"False"`` — and NaN as True. Both appear when the frame
    came back through a CSV round trip, and both would silently select the WRONG
    subset of nuclei, which is worse than not filtering at all. Anything not
    recognisably true is treated as false.
    """
    if series.dtype == bool:
        return series.to_numpy(dtype=bool)
    if pd.api.types.is_numeric_dtype(series):
        vals = pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        return vals != 0.0
    return np.asarray(
        [str(v).strip().lower() in ("true", "t", "yes", "y", "1") for v in series],
        dtype=bool,
    )


def find_ring_columns(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Radial enrichment columns in ``df``, ascending by ring radius.

    Each entry carries the column name, the ring's outer edge in µm, and the
    partner-channel token the column was named for (``rna2``, or ``protein``
    after rna_protein relabels). Returns an empty list when the frame carries no
    radial columns — the normal case, since the profile is opt-in.
    """
    out: List[Dict[str, Any]] = []
    for col in df.columns if df is not None else []:
        m = _RING_COL_RE.match(str(col))
        if m is None:
            continue
        try:
            um = _suffix_to_um(m.group("suffix"))
        except ValueError:
            continue
        out.append(dict(column=str(col), ring_um=um, partner=m.group("partner")))
    return sorted(out, key=lambda d: d["ring_um"])


def restrict_to_analysis_set(
    df: pd.DataFrame, *, restrict_to_sampled: bool = True
) -> Tuple[pd.DataFrame, str]:
    """Reduce a per-nucleus frame to the set the per-image rollups used.

    Returns ``(frame, human_label)``. The label goes ON the chart: a figure and
    the table it visualises must not report different numbers from the same run,
    and when they legitimately cover different sets the figure has to say which
    one it drew.

    ``restrict_to_sampled`` mirrors ``sampling.apply_to_rollups``. Pass False
    when that setting is off, because there the rollups were computed over ALL
    eligible nuclei and filtering here would manufacture the mismatch this exists
    to prevent. With sampling off the column is absent and both settings agree.
    """
    if df is None or len(df) == 0 or SAMPLED_COL not in df.columns:
        return df, "all analysed nuclei"
    if not restrict_to_sampled:
        return df, "all analysed nuclei (sampling.apply_to_rollups off)"
    keep = _bool_mask(df[SAMPLED_COL])
    return (
        df.loc[keep].reset_index(drop=True),
        "fixed-N sampled nuclei only (as in per_image_summary.csv)",
    )


def radial_profile_ci_table(
    df: pd.DataFrame,
    *,
    confidence: float = 0.95,
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """Per-ring ``mean`` / ``sd`` / ``n`` / ``ci_low`` / ``ci_high`` across objects.

    ``df`` is a per-nucleus frame (``nuclei_metrics.csv`` or the in-memory
    equivalent). One row per ring, ascending by radius. Non-finite values are
    dropped per ring, so a nucleus that could not be profiled reduces that ring's
    ``n`` instead of voiding the ring.

    ``group_col`` (normally ``"condition"``) computes the rings SEPARATELY per
    group and adds that column to the output, so the caller never has to pool
    arms that are not comparable. Left None the whole frame is one group, which
    is only correct for a frame that already holds a single condition.

    The interval is ``mean ± t(1-(1-confidence)/2, n-1) * sd / sqrt(n)``. With
    n < 2 there is no spread to estimate, so ``sd`` and both bounds are NaN
    rather than a zero-width ribbon that would imply certainty.
    """
    if group_col and df is not None and group_col in getattr(df, "columns", ()):
        blocks = []
        for key in df[group_col].astype(str).unique().tolist():
            sub = df.loc[df[group_col].astype(str) == key]
            tab = radial_profile_ci_table(sub, confidence=confidence)
            if len(tab):
                tab.insert(0, group_col, key)
                blocks.append(tab)
        if not blocks:
            return pd.DataFrame()
        return pd.concat(blocks, ignore_index=True)

    rings = find_ring_columns(df)
    rows = []
    for r in rings:
        vals = pd.to_numeric(df[r["column"]], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        n = int(vals.size)
        mean = float(vals.mean()) if n else float("nan")
        if n > 1:
            sd = float(vals.std(ddof=1))
            try:
                from scipy import stats as _stats

                tcrit = float(_stats.t.ppf(1.0 - (1.0 - confidence) / 2.0, n - 1))
            except Exception:
                tcrit = float("nan")
            half = tcrit * sd / math.sqrt(n)
            ci_low, ci_high = mean - half, mean + half
        else:
            sd = ci_low = ci_high = float("nan")
        rows.append(
            dict(
                ring_um=float(r["ring_um"]),
                partner=r["partner"],
                mean=mean,
                sd=sd,
                n_nuclei=n,
                ci_low=ci_low,
                ci_high=ci_high,
            )
        )
    return pd.DataFrame(rows)


def _series_plan(
    df: pd.DataFrame, *, group_col: str, confidence: float
) -> List[Dict[str, Any]]:
    """One entry per panel to draw: label, colour, CI table, sec-only flag.

    Condition panels come first in control-first order (the same ordering the
    SuperPlots use, so a reader meets the arms in the same sequence in every
    figure of the set); the pooled secondary-only control comes last.

    The secondary-only rows are split off BEFORE grouping. They carry a
    ``condition`` value of their own — that is how the well was labelled — so
    grouping first would fold a no-probe control into the arm it was acquired
    alongside and inflate or deflate that arm's ribbon with background.
    """
    if SECONLY_COL in df.columns:
        sec_mask = _bool_mask(df[SECONLY_COL])
    else:
        sec_mask = np.zeros(len(df), dtype=bool)
    real = df.loc[~sec_mask]
    sec = df.loc[sec_mask]

    plan: List[Dict[str, Any]] = []
    if group_col and group_col in real.columns and len(real):
        labels = order_conditions_control_first(
            real[group_col].astype(str).tolist()
        )
        for i, label in enumerate(labels):
            sub = real.loc[real[group_col].astype(str) == label]
            plan.append(
                dict(
                    label=str(label),
                    color=_CONDITION_COLORS[i % len(_CONDITION_COLORS)],
                    table=radial_profile_ci_table(sub, confidence=confidence),
                    is_seconly=False,
                )
            )
    elif len(real):
        plan.append(
            dict(
                label="all nuclei",
                color=_CONDITION_COLORS[0],
                table=radial_profile_ci_table(real, confidence=confidence),
                is_seconly=False,
            )
        )

    # Pooled, NOT split by condition: a no-probe control is one background
    # measurement, not one arm per well it happened to sit next to.
    if len(sec):
        plan.append(
            dict(
                label=_SECONLY_LABEL,
                color=_SECONLY_COLOR,
                table=radial_profile_ci_table(sec, confidence=confidence),
                is_seconly=True,
            )
        )

    def _drawable(entry: Dict[str, Any]) -> bool:
        tab = entry["table"]
        return bool(
            len(tab)
            and np.isfinite(tab["mean"].to_numpy(dtype=float)).any()
        )

    return [e for e in plan if _drawable(e)]


def _n_text(table: pd.DataFrame) -> str:
    n = table["n_nuclei"].to_numpy(dtype=int)
    lo, hi = int(n.min()), int(n.max())
    return f"n = {hi} nuclei" if lo == hi else f"n = {lo}–{hi} nuclei per ring"


def plot_radial_profile_ci(
    df: pd.DataFrame,
    out_png,
    *,
    confidence: float = 0.95,
    title: Optional[str] = None,
    group_col: Optional[str] = CONDITION_COL,
    restrict_to_sampled: bool = True,
) -> Optional[Path]:
    """Write the mean ± 95% CI radial profile PNG. Returns the path, or None.

    ONE PANEL PER CONDITION, on shared axes limits so the panels are directly
    comparable, plus a separate explicitly labelled panel for the pooled
    ``secondary_only`` (no-probe) control, which is excluded from every condition
    panel. A single-condition frame therefore draws exactly one panel.

    Returns None (writing nothing) when ``df`` carries no radial columns or no
    ring in any group has a finite mean — an absent figure is honest about an
    absent measurement, where an empty set of axes would not be.
    """
    if df is None or len(df) == 0 or not find_ring_columns(df):
        return None

    work, set_label = restrict_to_analysis_set(
        df, restrict_to_sampled=restrict_to_sampled
    )
    if work is None or len(work) == 0:
        return None

    plan = _series_plan(work, group_col=group_col or "", confidence=confidence)
    if not plan:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    partner = str(plan[0]["table"]["partner"].iloc[0])
    pct = int(round(confidence * 100))

    # Shared y-limits across panels, computed over every ribbon that exists, so a
    # difference between two panels is a difference in the data and not in the
    # scaling. The reference line at 1.0 is always inside the range.
    _all = np.concatenate([
        np.concatenate([
            e["table"]["mean"].to_numpy(dtype=float),
            e["table"]["ci_low"].to_numpy(dtype=float),
            e["table"]["ci_high"].to_numpy(dtype=float),
        ])
        for e in plan
    ])
    _all = _all[np.isfinite(_all)]
    lo_y, hi_y = float(min(_all.min(), 1.0)), float(max(_all.max(), 1.0))
    pad = 0.06 * (hi_y - lo_y) if hi_y > lo_y else 0.1
    ylim = (lo_y - pad, hi_y + pad)

    n_panels = len(plan)
    ncols = min(3, n_panels)
    nrows = int(math.ceil(n_panels / float(ncols)))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(1.0 + 3.4 * ncols, 0.6 + 3.0 * nrows),
        squeeze=False, sharey=True,
    )
    flat = [ax for row in axes for ax in row]

    for ax, entry in zip(flat, plan):
        tab = entry["table"]
        x = tab["ring_um"].to_numpy(dtype=float)
        y = tab["mean"].to_numpy(dtype=float)
        lo = tab["ci_low"].to_numpy(dtype=float)
        hi = tab["ci_high"].to_numpy(dtype=float)
        color = entry["color"]

        # Ribbon only where the interval exists (n >= 2 rings).
        finite_ci = np.isfinite(lo) & np.isfinite(hi)
        if finite_ci.any():
            ax.fill_between(
                x[finite_ci], lo[finite_ci], hi[finite_ci],
                color=color, alpha=0.22, linewidth=0,
                label=f"{pct}% CI of the mean",
            )
        ax.plot(
            x, y, marker="o", color=color, markersize=4.5, linewidth=1.6,
            linestyle="--" if entry["is_seconly"] else "-",
            label="mean across nuclei",
        )
        # 1.0 = the partner is no more concentrated in this ring than at random
        # in-nucleus positions at the same radius.
        ax.axhline(1.0, color=_REFERENCE_COLOR, linestyle="--", linewidth=1.1,
                   label="no enrichment (=1)")

        panel_title = entry["label"]
        if entry["is_seconly"]:
            panel_title += "\nbackground control — excluded from the panels above"
        ax.set_title(panel_title, fontsize=9)
        ax.set_xlabel("Distance from spot centre,\nring outer edge (µm)", fontsize=8)
        ax.set_ylim(*ylim)
        ax.tick_params(labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        # Keep the innermost ring's marker and its ribbon off the y-spine, where
        # they would be half-hidden — that ring is the one carrying the signal.
        ax.margins(x=0.06)
        # The unit of replication and the n belong on the panel they describe, not
        # in a shared footnote a reader has to map back onto six panels.
        ax.annotate(
            _n_text(tab), xy=(0.98, 0.03), xycoords="axes fraction",
            fontsize=7, va="bottom", ha="right", color="#333333",
        )

    for ax in flat[n_panels:]:
        ax.set_visible(False)
    for row in axes:
        row[0].set_ylabel(
            f"{partner} enrichment vs same-radius\nrandom in-nucleus null (ratio)",
            fontsize=8,
        )

    flat[0].legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle(
        title or "Partner enrichment vs distance from spot, per condition",
        fontsize=10,
    )

    # State the unit of replication, WHICH set of nuclei was used, and the
    # interval method ON the chart — a reader should not have to find the methods
    # section to know what the ribbon is or which nuclei it rests on.
    caption = (
        "1 nucleus = 1 observation; "
        f"ribbon = {pct}% CI (t-based) of the mean; "
        "rings centred on each detected spot (no alignment step).\n"
        f"Nuclei included: {set_label}. "
        "Conditions are never pooled"
        + (
            "; the secondary-only (no-probe) control is shown separately and is "
            "excluded from every condition panel."
            if any(e["is_seconly"] for e in plan) else "."
        )
    )
    fig.text(0.01, 0.005, caption, fontsize=6.5, va="bottom", ha="left",
             color="#444444")
    # Reserve the strips the suptitle and the caption occupy, or a multi-row grid
    # draws its panel titles straight through them.
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.94))

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return out_png
