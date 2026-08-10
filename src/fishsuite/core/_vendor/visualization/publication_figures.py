# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Publication-quality figure generation for colocalization analysis.

Generates figures suitable for journals like Nature Methods and Molecular Cell,
and for Brian's MIAT-QKI dissertation (Methods Chapter 3). All figures follow
consistent styling:
  - Arial font, sized for small (~5-7 in) journal-column plots:
      * 8pt default panel content (axes), 9pt axis labels, 7pt ticks/legend
      * applied via ``set_publication_style``; callers may override
  - 600 DPI raster default (Brian's stated preference). SVG is emitted only
    when explicitly requested via ``formats``; PDF is rarely useful for him.
  - Clean axes (top + right spines despined), tight margins, whitespace trimmed
  - Significance brackets placed above the upper whisker (not above outliers),
    unified thickness, p<0.001 reported as "p < 1e-3" with 3 sig-fig fallback
  - Star annotations available alongside or in place of numeric p ("ns", "*",
    "**", "***", "****")

Color policy (HARD RULE — Brian's accessibility + consistency feedback):
  - One palette across the whole pipeline. Same condition → same color in
    every figure. The ``CONDITION_COLORS`` dict at the top of this module is
    the source of truth.
  - Okabe-Ito 8-color palette, colorblind-safe; never red+green together.
  - Brian's two active designs are mapped explicitly:
      * H9 antisense oligo (3-condition):    Sec-Only, NT ASO, KD ASO
      * Epistasis factorial (4-condition):    WT, MIAT_KD, QKI_KO, DKO
    Synonyms (e.g. ``KO``, ``MIAT-KD``, ``Sec_Only``) resolve to the same
    color via ``_canonicalize_condition()``.

Figure types:
  1. Condition comparison boxplots (with individual data points)
  2. Multi-panel grids of metric comparisons
  3. Significance brackets + p-value/effect-size annotations
"""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Color palette (single source of truth — see HARD RULE in module docstring)
# ---------------------------------------------------------------------------

# Okabe-Ito 8-color colorblind-safe palette (matches the RNA-seq pipeline).
# Order: orange, sky blue, bluish green, yellow, blue, vermillion, reddish
# purple, black. Colors are stable across all figure generators.
OKABE_ITO = [
    "#E69F00",  # 0 orange
    "#56B4E9",  # 1 sky blue
    "#009E73",  # 2 bluish green
    "#F0E442",  # 3 yellow
    "#0072B2",  # 4 blue
    "#D55E00",  # 5 vermillion
    "#CC79A7",  # 6 reddish purple
    "#000000",  # 7 black
]

# Default ordered fallback palette for arbitrary conditions (avoids yellow at
# index 0 because yellow on white reads poorly as a fill color).
_DEFAULT_PALETTE = [
    OKABE_ITO[1],  # sky blue
    OKABE_ITO[5],  # vermillion (NOT pure red — distinguishable from green)
    OKABE_ITO[2],  # bluish green (used when vermillion is already taken)
    OKABE_ITO[0],  # orange
    OKABE_ITO[4],  # blue
    OKABE_ITO[6],  # reddish purple
    OKABE_ITO[7],  # black
    OKABE_ITO[3],  # yellow (last resort)
]

# Stable per-condition assignments. Same condition always renders the same
# color across H9 ASO and epistasis figures, so a reader scanning multiple
# panels in a thesis chapter doesn't have to relearn the legend.
#
# Rationale for assignments:
#   - Controls (Sec-Only, WT) → sky blue (calm, neutral baseline)
#   - Non-targeting (NT ASO)  → bluish green (close to control but distinct)
#   - Knockdown / knockout treatments → vermillion / orange (warm, "perturbed")
#   - Combined perturbations (DKO) → reddish purple
CONDITION_COLORS: Dict[str, str] = {
    # H9 antisense oligo design
    "Sec-Only": OKABE_ITO[1],   # sky blue
    "NT ASO":   OKABE_ITO[2],   # bluish green
    "KD ASO":   OKABE_ITO[5],   # vermillion
    # Epistasis 4-condition factorial
    "WT":       OKABE_ITO[1],   # sky blue (control == control)
    "MIAT_KD":  OKABE_ITO[5],   # vermillion
    "QKI_KO":   OKABE_ITO[0],   # orange
    "DKO":      OKABE_ITO[6],   # reddish purple
    # Generic two-group fallback
    "KO":       OKABE_ITO[5],   # vermillion (legacy default)
}


def _canonicalize_condition(label: str) -> str:
    """Normalize condition label spelling so synonyms map to one color.

    e.g. ``"miat-kd"`` -> ``"MIAT_KD"``, ``"sec only"`` -> ``"Sec-Only"``.
    Unknown labels pass through unchanged (the caller falls back to the
    ordered default palette).
    """
    if not isinstance(label, str):
        return str(label)
    s = label.strip()
    s_norm = s.replace(" ", "").replace("-", "").replace("_", "").lower()
    aliases = {
        "seconly":   "Sec-Only",
        "secondaryonly": "Sec-Only",
        "sec":       "Sec-Only",
        "ntaso":     "NT ASO",
        "nt":        "NT ASO",
        "nontargeting": "NT ASO",
        "kdaso":     "KD ASO",
        "kd":        "KD ASO",
        "wt":        "WT",
        "wildtype":  "WT",
        "control":   "WT",
        "miatkd":    "MIAT_KD",
        "qkiko":     "QKI_KO",
        "qki":       "QKI_KO",
        "dko":       "DKO",
        "doubleko":  "DKO",
        "ko":        "KO",
    }
    return aliases.get(s_norm, s)


def _get_colors(group_order: Sequence[str],
                colors: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Return a {group: color} mapping.

    Resolution order:
      1. Caller-supplied ``colors`` dict wins (verbatim, no canonicalization).
      2. Otherwise canonicalize each group label, look it up in
         ``CONDITION_COLORS``.
      3. Anything still unmapped falls back to ``_DEFAULT_PALETTE`` in the
         order encountered (so colors stay stable per-group-order, not
         per-call).
    """
    if colors:
        return dict(colors)
    out: Dict[str, str] = {}
    fallback_idx = 0
    for g in group_order:
        canon = _canonicalize_condition(g)
        if canon in CONDITION_COLORS:
            out[g] = CONDITION_COLORS[canon]
        else:
            out[g] = _DEFAULT_PALETTE[fallback_idx % len(_DEFAULT_PALETTE)]
            fallback_idx += 1
    return out


# ---------------------------------------------------------------------------
# Style + I/O helpers
# ---------------------------------------------------------------------------


def set_publication_style(font_size: int = 8, font_family: str = "Arial") -> None:
    """Configure matplotlib for publication-quality output.

    Defaults sized for typical journal-column figures (~3-7 in wide). For
    Brian's small image-pipeline boxplots that ratio (~1-2 pt per inch) reads
    cleanly at 600 DPI without overlapping. Callers can pass ``font_size=10``
    or ``11`` for thesis-body figures where the column is wider.
    """
    matplotlib.rcParams.update({
        "font.size": font_size,
        "font.family": font_family,
        "axes.titlesize": font_size + 1,
        "axes.labelsize": font_size + 1,   # axis labels slightly larger than ticks
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "legend.fontsize": font_size - 1,
        "legend.frameon": False,           # no box around legend (cleaner look)
        "figure.dpi": 600,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "lines.linewidth": 1.0,
        "pdf.fonttype": 42,                # TrueType fonts (editable in Illustrator)
        "ps.fonttype": 42,
        "svg.fonttype": "none",            # keep text as text in SVG
    })


def save_figure(
    fig: plt.Figure,
    output_path: str | Path,
    formats: Sequence[str] = ("png",),
    dpi: int = 600,
) -> None:
    """Save figure in one or more formats.

    Defaults to PNG-only at 600 DPI (Brian's stated preference). Pass
    ``formats=("png", "svg")`` if a vector copy is needed for figure assembly;
    PDF is supported but not emitted by default.

    Parameters
    ----------
    fig : matplotlib Figure
    output_path : str or Path
        Base path without extension.
    formats : sequence of str
        File formats to save. Default: ``("png",)``.
    dpi : int
        Resolution for raster formats. Default: 600.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(
            output_path.with_suffix(f".{fmt}"),
            format=fmt,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.05,
        )


# ---------------------------------------------------------------------------
# Single-metric condition comparison
# ---------------------------------------------------------------------------


def plot_condition_comparison(
    df: pd.DataFrame,
    metric: str,
    condition_col: str = "condition",
    group_order: Sequence[str] = ("WT", "KO"),
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    colors: Optional[Dict[str, str]] = None,
    style: str = "box_with_points",
    stats_result: Optional[Dict[str, Any]] = None,
    figsize: Tuple[float, float] = (2.6, 3.2),
    annotate_stars: bool = True,
) -> plt.Figure:
    """Plot comparison of a metric between conditions.

    Generates a publication-ready boxplot/violin with individual data points
    and optional significance annotations. The bracket sits above the upper
    whisker (not above outlier points), so distant outliers don't push the
    annotation off-screen.

    Parameters
    ----------
    df : pd.DataFrame
        Per-nucleus or per-spot metrics.
    metric : str
        Column name to plot on y-axis.
    condition_col : str
        Column containing condition labels.
    group_order : sequence of str
        Order of conditions on x-axis.
    ylabel : str, optional
        Y-axis label (defaults to a human-readable version of ``metric``).
    title : str, optional
        Plot title.
    colors : dict, optional
        Mapping of condition -> color. If omitted, the shared
        ``CONDITION_COLORS`` palette is used.
    style : str
        ``"box"``, ``"violin"``, or ``"box_with_points"``.
    stats_result : dict, optional
        Output from ``mann_whitney_test()`` (or omnibus dict) for significance.
    figsize : tuple
        Figure size in inches.
    annotate_stars : bool
        If True, append the canonical "ns"/"*"/"**"/"***"/"****" stars to
        the bracket label alongside the numeric p.

    Returns
    -------
    matplotlib.figure.Figure
    """
    group_order = list(group_order)
    colors = _get_colors(group_order, colors)
    palette = [colors.get(g, _DEFAULT_PALETTE[0]) for g in group_order]

    fig, ax = plt.subplots(figsize=figsize)

    plot_data = df[df[condition_col].isin(group_order)].copy()

    # Note: pass ``hue=condition_col`` + ``legend=False`` to satisfy
    # seaborn >= 0.13 (avoids the "palette without hue" deprecation warning).
    if style == "violin":
        sns.violinplot(
            data=plot_data, x=condition_col, y=metric,
            hue=condition_col, order=group_order, palette=palette,
            inner="box", cut=0, linewidth=0.8, legend=False, ax=ax,
        )
    elif style == "box_with_points":
        sns.boxplot(
            data=plot_data, x=condition_col, y=metric,
            hue=condition_col, order=group_order, palette=palette,
            width=0.55, linewidth=0.9, fliersize=0, legend=False, ax=ax,
            boxprops=dict(alpha=0.85),
        )
        sns.stripplot(
            data=plot_data, x=condition_col, y=metric,
            order=group_order, color="0.2", size=2.6, alpha=0.75,
            jitter=0.18, edgecolor="white", linewidth=0.3, ax=ax,
        )
    else:  # box only
        sns.boxplot(
            data=plot_data, x=condition_col, y=metric,
            hue=condition_col, order=group_order, palette=palette,
            width=0.55, linewidth=0.9, legend=False, ax=ax,
        )

    ax.set_xlabel("")
    ax.set_ylabel(ylabel or _format_metric_label(metric))
    if title:
        ax.set_title(title)

    # Significance bracket — anchored to whisker tops, not ylim
    if stats_result is not None and "p_value" in stats_result:
        is_omnibus = stats_result.get("is_omnibus", False)
        bracket_x2 = len(group_order) - 1 if is_omnibus else 1
        _add_significance_bracket(
            ax, stats_result, group_order,
            x1=0, x2=bracket_x2,
            data_for_anchor=plot_data, metric=metric, condition_col=condition_col,
            annotate_stars=annotate_stars,
        )

    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Multi-panel grid
# ---------------------------------------------------------------------------


def plot_multi_metric_comparison(
    df: pd.DataFrame,
    metrics: List[str],
    stats_df: Optional[pd.DataFrame] = None,
    condition_col: str = "condition",
    group_order: Sequence[str] = ("WT", "KO"),
    ncols: int = 4,
    panel_size: Tuple[float, float] = (2.4, 3.0),
    colors: Optional[Dict[str, str]] = None,
    annotate_stars: bool = True,
) -> plt.Figure:
    """Multi-panel figure comparing multiple metrics between conditions.

    Creates a grid of boxplots, one per metric, with significance annotations.
    This is the typical Figure 2 or 3 in a colocalization paper.
    """
    if not metrics:
        return plt.figure()

    group_order = list(group_order)
    n_metrics = len(metrics)
    nrows = int(np.ceil(n_metrics / ncols))
    fig_width = ncols * panel_size[0]
    fig_height = nrows * panel_size[1]

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), squeeze=False)

    colors_map = _get_colors(group_order, colors)
    palette = [colors_map.get(g, _DEFAULT_PALETTE[0]) for g in group_order]

    for i, metric in enumerate(metrics):
        row, col = divmod(i, ncols)
        ax = axes[row, col]

        if metric not in df.columns:
            ax.set_visible(False)
            continue

        plot_data = df[df[condition_col].isin(group_order)]

        sns.boxplot(
            data=plot_data, x=condition_col, y=metric,
            hue=condition_col, order=group_order, palette=palette,
            width=0.55, linewidth=0.9, fliersize=0, legend=False, ax=ax,
            boxprops=dict(alpha=0.85),
        )
        sns.stripplot(
            data=plot_data, x=condition_col, y=metric,
            order=group_order, color="0.2", size=2.0, alpha=0.7,
            jitter=0.16, edgecolor="white", linewidth=0.25, ax=ax,
        )

        ax.set_xlabel("")
        ax.set_ylabel(_format_metric_label(metric))
        ax.set_title(chr(65 + i), loc="left", fontweight="bold")  # A, B, C, ...

        # Significance if stats are available
        if stats_df is not None and metric in stats_df["metric"].values:
            row_stats = stats_df[stats_df["metric"] == metric].iloc[0]
            is_kw = "h_stat" in stats_df.columns
            bracket_x2 = len(group_order) - 1 if is_kw else 1
            _add_significance_bracket(
                ax,
                {"p_value": row_stats.get("p_adjusted", row_stats.get("p_value")),
                 "effect_size_r": row_stats.get("effect_size_r", None),
                 "is_omnibus": is_kw},
                group_order,
                x1=0, x2=bracket_x2,
                data_for_anchor=plot_data, metric=metric, condition_col=condition_col,
                annotate_stars=annotate_stars,
            )

        sns.despine(ax=ax)

    # Hide unused panels
    for i in range(n_metrics, nrows * ncols):
        row, col = divmod(i, ncols)
        axes[row, col].set_visible(False)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Labels + significance helpers
# ---------------------------------------------------------------------------


def _format_metric_label(metric: str) -> str:
    """Convert column name to human-readable axis label."""
    labels = {
        # Colocalization metrics
        "pearson_r": "Pearson r",
        "spearman_rho": "Spearman ρ",
        "li_icq": "Li's ICQ",
        "manders_m1": "Manders M1\n(RNA|Protein)",
        "manders_m2": "Manders M2\n(Protein|RNA)",
        "cosine_overlap": "Cosine overlap",
        "rna_spot_count": "RNA spots / nucleus",
        "coloc_spot_count": "Coloc. spots / nucleus",
        "coloc_spot_fraction": "Colocalization fraction",
        "ab_enrich_in_rna_high": "Protein enrichment\n(RNA-high regions)",
        "rna_enrich_in_ab_high": "RNA enrichment\n(protein-high regions)",
        "rna_mean_in_nucleus": "RNA mean intensity (a.u.)",
        "ab_mean_in_nucleus": "Protein mean intensity (a.u.)",
        # Spot shape / size
        "spot_fwhm_px": "Spot FWHM (px)",
        "mean_spot_fwhm_px": "Mean spot FWHM (px)",
        "median_spot_fwhm_px": "Median spot FWHM (px)",
        "spot_diameter_um": "Spot diameter (µm)",
        "spot_volume_um3": "Spot volume (µm³)",
        # Spot intensity / quality
        "spot_peak_intensity": "Spot peak intensity (a.u.)",
        # NEW honest names (preferred) + OLD-name back-compat keys -> same honest label
        "peak_intensity": "Peak intensity (brightest voxel)",
        "integrated_intensity_fit": "Peak intensity (brightest voxel)",
        "local_snr": "Local SNR",
        "rna_spot_mean_intensity_bgc_blend": "Mean spot intensity / cell\n(BG-corrected)",
        "rna_spot_total_peak_intensity": "Total RNA peak intensity / cell",
        "rna_spot_total_intensity_fit": "Total RNA peak intensity / cell",
        "rna_spot_total_intensity_bgc_blend": "Total RNA mass / cell\n(BG-corrected)",
        # Morphology metrics
        "area_um2": "Cell area (µm²)",
        "circularity": "Circularity",
        "aspect_ratio": "Aspect ratio",
        "solidity": "Solidity",
        "elongation": "Elongation",
        # Cytoplasm / N:C ratio metrics
        "rna_nc_ratio": "RNA N:C ratio",
        "ab_nc_ratio": "Protein N:C ratio",
        "rna_nuclear_mean": "RNA nuclear mean (a.u.)",
        "rna_cytoplasmic_mean": "RNA cytoplasmic mean (a.u.)",
        "ab_nuclear_mean": "Protein nuclear mean (a.u.)",
        "ab_cytoplasmic_mean": "Protein cytoplasmic mean (a.u.)",
        "nuclear_spot_count": "Nuclear spots / cell",
        "cyto_spot_count": "Cytoplasmic spots / cell",
        "nuclear_spot_fraction": "Nuclear spot fraction",
        # Spatial
        "spot_to_nuc_edge_um": "Distance to nuclear edge (µm)",
        "nucleus_area_px": "Nucleus area (px)",
    }
    return labels.get(metric, metric.replace("_", " "))


def _stars_for_p(p: float) -> str:
    """GraphPad-style star annotation for a p-value."""
    if p is None or np.isnan(p):
        return ""
    if p > 0.05:
        return "ns"
    if p > 0.01:
        return "*"
    if p > 0.001:
        return "**"
    if p > 0.0001:
        return "***"
    return "****"


def _format_pvalue(p: float, prefix: str = "") -> str:
    """Format a p-value for figure annotation.

    - p < 1e-4: scientific notation, 1 sig fig    (e.g. "p = 3e-05")
    - p < 0.001: scientific, 2 sig figs           (e.g. "p = 2.4e-04")
    - p >= 0.001: 3 decimal places                (e.g. "p = 0.012")
    - p >= 0.05: 2 decimal places, prefix "n.s."  (e.g. "n.s. (p = 0.27)")
    """
    if p is None or np.isnan(p):
        return ""
    if p < 1e-4:
        body = f"p = {p:.0e}"
    elif p < 1e-3:
        body = f"p = {p:.1e}"
    elif p < 0.05:
        body = f"p = {p:.3f}"
    else:
        body = f"p = {p:.2f}"
    full = f"{prefix}{body}"
    if p >= 0.05:
        full = f"n.s. ({full})"
    return full


def _whisker_top(values: Iterable[float]) -> Optional[float]:
    """Return the upper-whisker location (Q3 + 1.5*IQR clipped to data max)
    for a sequence of numeric values. Returns None for empty/all-NaN input.
    Used to anchor significance brackets above the whisker rather than above
    distant outliers, which reads as "above the box" to readers."""
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    upper_fence = q3 + 1.5 * iqr
    return float(min(arr.max(), upper_fence))


def _add_significance_bracket(
    ax: plt.Axes,
    stats_result: Dict[str, Any],
    group_order: Sequence[str],
    x1: int = 0,
    x2: int = 1,
    data_for_anchor: Optional[pd.DataFrame] = None,
    metric: Optional[str] = None,
    condition_col: str = "condition",
    annotate_stars: bool = True,
) -> None:
    """Add significance bracket with p-value (and effect size) annotation.

    The bracket is anchored to the maximum whisker top across the spanned
    groups when ``data_for_anchor`` is provided — that way distant outliers
    don't push the bracket off the panel. Falls back to the previous
    ylim-based placement if no anchor data is given.

    Parameters
    ----------
    ax : matplotlib Axes
    stats_result : dict
        Must contain "p_value"; optionally "effect_size_r", "is_omnibus".
    group_order : sequence of str
        Condition labels on the x-axis.
    x1, x2 : int
        X-axis positions for left and right ends of the bracket.
    data_for_anchor, metric, condition_col :
        If supplied, the bracket Y position is computed from the per-group
        whisker tops in this data frame.
    annotate_stars : bool
        Append GraphPad-style stars to the bracket label.
    """
    p = stats_result.get("p_value", 1.0)
    if p is None or np.isnan(p):
        return

    is_omnibus = stats_result.get("is_omnibus", False)
    prefix = "K-W " if is_omnibus else ""

    p_text = _format_pvalue(p, prefix=prefix)
    if annotate_stars:
        stars = _stars_for_p(p)
        if stars:
            p_text = f"{stars}\n{p_text}" if p < 0.05 else p_text

    # Effect size (skip for omnibus — epsilon-sq is reported elsewhere)
    r = stats_result.get("effect_size_r")
    if r is not None and not np.isnan(r) and not is_omnibus:
        p_text += f"\nr = {r:.2f}"

    # Anchor — prefer whisker tops over ylim so outliers don't push us off
    y_max = ax.get_ylim()[1]
    y_min = ax.get_ylim()[0]
    y_range = y_max - y_min if y_max > y_min else 1.0

    anchor_y = None
    if (data_for_anchor is not None and metric is not None
            and metric in data_for_anchor.columns
            and condition_col in data_for_anchor.columns):
        spanned_groups = list(group_order)[x1:x2 + 1]
        tops: List[float] = []
        for g in spanned_groups:
            sub = data_for_anchor[data_for_anchor[condition_col] == g][metric]
            top = _whisker_top(sub)
            if top is not None:
                tops.append(top)
        if tops:
            anchor_y = max(tops)

    if anchor_y is None:
        anchor_y = y_max - 0.05 * y_range

    bracket_y = anchor_y + 0.04 * y_range
    text_y = bracket_y + 0.02 * y_range

    ax.plot([x1, x1, x2, x2], [bracket_y, text_y, text_y, bracket_y],
            lw=0.9, c="black", clip_on=False)
    ax.text((x1 + x2) / 2, text_y, p_text,
            ha="center", va="bottom", fontsize=plt.rcParams["legend.fontsize"])

    # Extend ylim so annotation isn't clipped (count newlines for height)
    n_lines = 1 + p_text.count("\n")
    ax.set_ylim(y_min, text_y + (0.06 + 0.04 * n_lines) * y_range)
