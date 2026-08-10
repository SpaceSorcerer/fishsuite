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
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Okabe-Ito, colorblind-safe. Blue for the profile, vermillion for the
# no-enrichment reference — never a red/green pair.
_PROFILE_COLOR = "#0072B2"
_REFERENCE_COLOR = "#D55E00"

# ``<partner>_radial_enrichment_at_<0p25um>`` -> the 0p25um suffix.
_RING_COL_RE = re.compile(r"^(?P<partner>.+?)_radial_enrichment_at_(?P<suffix>[0-9p]+um)$")


def _suffix_to_um(suffix: str) -> float:
    """``'0p25um'`` -> ``0.25``. Inverse of ``rna_rna._format_pair_um``."""
    return float(suffix[:-2].replace("p", "."))


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


def radial_profile_ci_table(df: pd.DataFrame, *, confidence: float = 0.95) -> pd.DataFrame:
    """Per-ring ``mean`` / ``sd`` / ``n`` / ``ci_low`` / ``ci_high`` across objects.

    ``df`` is a per-nucleus frame (``nuclei_metrics.csv`` or the in-memory
    equivalent). One row per ring, ascending by radius. Non-finite values are
    dropped per ring, so a nucleus that could not be profiled reduces that ring's
    ``n`` instead of voiding the ring.

    The interval is ``mean ± t(1-(1-confidence)/2, n-1) * sd / sqrt(n)``. With
    n < 2 there is no spread to estimate, so ``sd`` and both bounds are NaN
    rather than a zero-width ribbon that would imply certainty.
    """
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


def plot_radial_profile_ci(
    df: pd.DataFrame,
    out_png,
    *,
    confidence: float = 0.95,
    title: Optional[str] = None,
) -> Optional[Path]:
    """Write the mean ± 95% CI radial profile PNG. Returns the path, or None.

    Returns None (writing nothing) when ``df`` carries no radial columns or no
    ring has a finite mean — an absent figure is honest about an absent
    measurement, where an empty set of axes would not be.
    """
    table = radial_profile_ci_table(df, confidence=confidence)
    if len(table) == 0 or not np.isfinite(table["mean"].to_numpy(dtype=float)).any():
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = table["ring_um"].to_numpy(dtype=float)
    y = table["mean"].to_numpy(dtype=float)
    lo = table["ci_low"].to_numpy(dtype=float)
    hi = table["ci_high"].to_numpy(dtype=float)
    n_max = int(table["n_nuclei"].max())
    n_min = int(table["n_nuclei"].min())
    partner = str(table["partner"].iloc[0])

    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    # Ribbon only where the interval exists (n >= 2 rings).
    finite_ci = np.isfinite(lo) & np.isfinite(hi)
    if finite_ci.any():
        ax.fill_between(
            x[finite_ci], lo[finite_ci], hi[finite_ci],
            color=_PROFILE_COLOR, alpha=0.22, linewidth=0,
            label=f"{int(round(confidence * 100))}% CI of the mean",
        )
    ax.plot(x, y, "-o", color=_PROFILE_COLOR, markersize=4.5, linewidth=1.6,
            label="mean across nuclei")
    # 1.0 = the partner is no more concentrated in this ring than at random
    # in-nucleus positions at the same radius.
    ax.axhline(1.0, color=_REFERENCE_COLOR, linestyle="--", linewidth=1.1,
               label="no enrichment (=1)")

    ax.set_xlabel("Distance from spot centre, ring outer edge (µm)")
    ax.set_ylabel(f"{partner} enrichment vs same-radius\nrandom in-nucleus null (ratio)")
    ax.set_title(title or "Partner enrichment vs distance from spot", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, loc="best")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    # Keep the innermost ring's marker and its ribbon off the y-spine, where they
    # would be half-hidden — that ring is the one carrying the signal.
    ax.margins(x=0.06)

    # State the unit of replication, the n, and the interval method ON the chart —
    # a reader should not have to find the methods section to know what the ribbon
    # is or how many objects it rests on.
    n_text = f"n = {n_max} nuclei" if n_min == n_max else f"n = {n_min}–{n_max} nuclei per ring"
    ax.annotate(
        f"{n_text}; 1 nucleus = 1 observation\n"
        f"ribbon = {int(round(confidence * 100))}% CI (t-based) of the mean\n"
        f"rings centred on each detected spot (no alignment step)",
        xy=(0.0, -0.34), xycoords="axes fraction", fontsize=6.5,
        va="top", ha="left", color="#444444",
    )

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return out_png
