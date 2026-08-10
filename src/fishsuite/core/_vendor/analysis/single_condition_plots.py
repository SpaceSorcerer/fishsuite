#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-condition exploratory plot panel.

Reads a Fiji pipeline output directory (nuclei_metrics.csv + spot_metrics.csv
+ per_image_summary.csv) and generates a multi-panel summary PNG:

    1. Spots-per-nucleus distribution (histogram + median markers per image)
    2. Spot count per image (bar chart, sec-only highlighted)
    3. Spot intensity distribution per image (overlaid histograms)
    4. Spot diameter distribution per image
    5. Nuclear vs cytoplasmic spot fraction per image
    6. Distance-to-nuclear-edge distribution for nuclear spots

Single-PNG output keeps everything skimmable for collaborator review.
Pure pandas + matplotlib; no statistics package — this is for exploration
not for publication-grade comparison stats (use run_postprocessing.py for
those).

Usage:
    python -m analysis.single_condition_plots --output-dir <run-dir>
    python -m analysis.single_condition_plots --output-dir <run-dir> --out custom_panel.png
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
# 2026-05-21 Brian: force headless Agg backend BEFORE importing pyplot.
# Otherwise matplotlib auto-picked backend_qtagg on this Windows install,
# which is not thread-safe and blew up with MemoryError under the parallel
# subplot renderer.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# 2026-05-18 Brian: long figure titles overlapped the plot box (and the
# italic subtitle below) in the first wrapped-title render. Use this helper
# at every ``ax.set_title(_wrap_title(...))`` call so any title >width chars breaks onto
# multiple lines. width=70 is roughly two figure-inches of text at the
# default title fontsize; width=50 is for figures dropped into the 3×2
# combined-composition panel where each cell has less horizontal real
# estate. Use the same helper for italic subtitles via ``_wrap_subtitle``
# so wrapped subtitles also expand the reserved bottom margin.
def _wrap_title(text: str, width: int = 70) -> str:
    """Wrap long figure titles to ``width`` chars. Returns multi-line string."""
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False)) or text


def _wrap_subtitle(text: str, width: int = 95) -> tuple[str, int]:
    """Wrap an italic subtitle and return (wrapped_text, n_lines). Caller
    uses ``n_lines`` to choose subplots_adjust(bottom=...) so the reserved
    bottom margin grows when the subtitle wraps to two lines."""
    wrapped = "\n".join(textwrap.wrap(text, width=width, break_long_words=False)) or text
    return wrapped, max(1, wrapped.count("\n") + 1)


def _resolve_col(df, *names):
    """Return the first of *names present in df.columns (back-compat for
    legacy CSVs that still carry the OLD *_intensity_fit names)."""
    for n in names:
        if n in df.columns:
            return n
    return names[0]


# 2026-05-29 Brian: SuperPlot publication-style controls.
#   * _SUPERPLOT_TITLE_FONTSIZE — reduce the (previously large/dominating)
#     SuperPlot title a notch so it stays readable but doesn't crowd the data.
#     The wrappers set ax.set_title(_wrap_title(...)) WITHOUT an explicit
#     fontsize, so a smaller default flows to every SuperPlot title.
#   * _SUPERPLOT_FILTER_SUBTITLE — a short filter/criteria line ("spot floor …;
#     per-image means, n=… FoV/condition") drawn under every SuperPlot via
#     _add_subtitle. Set per-run by main() from run_config.json / the run-dir
#     name; falls back to a sensible generic string when not set.
_SUPERPLOT_TITLE_FONTSIZE = 11.0
_SUPERPLOT_FILTER_SUBTITLE: str | None = None


def _set_log_axis_plain(ax, which: str = "x") -> None:
    """Put ``ax`` on a log scale for the given axis ("x", "y", or "both") with
    PLAIN-DECIMAL tick labels (a non-mathtext ScalarFormatter).

    2026-05-27 Brian: matplotlib's default log formatter emits mathtext tick
    labels ($\\mathdefault{10^{n}}$). The mathtext parser (pyparsing packrat
    cache) is process-global and historically broke under the per-panel thread
    pool with a spurious "ParseException at char 0". The standalone deck now
    renders serially (the real thread-safety fix), but routing EVERY log axis
    through this one helper is defense-in-depth: no log panel ever emits
    mathtext, so re-introducing parallelism later cannot reopen that hole, and
    the labels read as plain numbers (10, 100, 1000) which print cleaner anyway.
    """
    import matplotlib.ticker as _mticker
    axes = ("x", "y") if which == "both" else (which,)
    for w in axes:
        try:
            if w == "x":
                ax.set_xscale("log")
                axis = ax.xaxis
            else:
                ax.set_yscale("log")
                axis = ax.yaxis
            _sf = _mticker.ScalarFormatter(useMathText=False)
            _sf.set_scientific(False)
            axis.set_major_formatter(_sf)
            axis.set_minor_formatter(_mticker.NullFormatter())
        except Exception:
            pass


# 2026-05-20 Brian: Brian flagged that titles overlap data and legends are
# cramped across many figures (especially the single-panel summary and the
# combined / overview panels: 00, 56, 97, 98). This helper runs the FINAL
# layout pass just before each ``fig.savefig()`` call — after the existing
# ``_relabel_fig()`` text-substitution pass — and applies a small set of
# cheap fixes: long categorical x-tick labels get a 30° rotation, tight
# layout nudges subplots apart, and a bottom / right margin is reserved
# when the figure has an italic subtitle or an outside legend respectively.
# All steps are wrapped in defensive try/except so a layout-engine warning
# on one axis never blocks a render.
def _final_layout_polish(fig, *, has_subtitle: bool = False,
                         has_legend_outside: bool = False) -> None:
    """Final pre-savefig pass to fix overlapping labels.
    - Rotates long x-tick labels 30° (helps grouped/categorical x-axes)
    - Runs tight_layout to nudge subplots apart
    - Reserves bottom margin for italic subtitles (read off each axis'
      _subtitle_pad attribute, set by _add_subtitle, so a 1- vs 2-line
      subtitle reserves the right amount of space and never overlaps the
      rotated x-tick labels)
    - Reserves right margin when a legend lives outside the axes

    2026-05-26 (publication polish): the prior version ran
    ``subplots_adjust(bottom=...)`` AFTER ``tight_layout()`` with a FIXED
    0.18 margin. tight_layout re-packs the axes ignoring the figure.text
    subtitle, so on bar panels with two-line rotated x-tick labels the
    italic subtitle ended up sitting on top of the tick labels. Fix: run
    tight_layout FIRST with an explicit bottom ``rect`` reservation sized
    to the largest subtitle present, so the subtitle band is carved out
    of the axes region rather than overwritten afterwards.
    """
    import matplotlib.pyplot as _plt
    try:
        for ax in fig.axes:
            try:
                labels = ax.get_xticklabels()
                texts = [t.get_text() or "" for t in labels]
                # Keep horizontal: short labels, already-rotated axes, and
                # MULTI-LINE condition labels ("NT ASO\n(n=293 cells)") — those
                # read cleanly horizontal and look broken when rotated. Only
                # rotate long single-line categorical labels that would
                # otherwise collide.
                long_single_line = (
                    labels
                    and any(len(s) > 3 for s in texts)
                    and not any("\n" in s for s in texts)
                    and all(abs(float(t.get_rotation() or 0)) < 1 for t in labels)
                )
                if long_single_line:
                    _plt.setp(labels, rotation=30, ha="right")
            except Exception:
                pass
        # Largest subtitle reservation requested by any axis (set as a
        # fraction-of-figure-height pad on ax._subtitle_pad by _add_subtitle /
        # the SuperPlot below-axis legend). This pad is sized for a SINGLE-axis
        # standalone figure; on a dense multi-axis grid (the combined / per-
        # image / per-condition contact sheets) a 0.16 reservation of the whole
        # tall figure would carve out an enormous empty bottom band, so cap it
        # hard there — the GridSpec hspace already separates the rows.
        sub_pad = 0.0
        for ax in fig.axes:
            try:
                sub_pad = max(sub_pad, float(getattr(ax, "_subtitle_pad", 0.0) or 0.0))
            except Exception:
                pass
        n_axes = len([a for a in fig.axes if a.get_visible()])
        if n_axes > 2:
            sub_pad = min(sub_pad, 0.02)
        bottom_rect = sub_pad if sub_pad > 0 else (0.06 if has_subtitle else 0.0)
        right_rect = 0.86 if has_legend_outside else 1.0
        import warnings as _w
        # Dense multi-axis contact sheets (combined / per-image / per-condition)
        # contain a below-axis legend whose bbox makes tight_layout emit a
        # harmless "Axes not compatible with tight_layout" warning. The pack
        # still succeeds and the cells render correctly, so silence just that
        # warning to keep the run log clean.
        with _w.catch_warnings():
            _w.filterwarnings("ignore", message=".*not compatible with tight_layout.*")
            try:
                # rect = (left, bottom, right, top) in figure coords; carve out
                # the bottom band for the subtitle BEFORE packing so axes never
                # reclaim it.
                fig.tight_layout(rect=(0.0, bottom_rect, right_rect, 1.0))
            except Exception:
                try:
                    fig.tight_layout()
                except Exception:
                    pass
    except Exception:
        pass


# 2026-05-26 (publication polish): condition display-name map. The grouping
# key stays the internal SEC_ONLY_CONDITION ("sec-only") everywhere so the
# data pipeline / ordering logic is untouched, but anything the reader SEES
# (x-tick labels, block labels, legends) is title-cased to "Sec-Only" to match
# Brian's standard condition naming (NT ASO / KD ASO / Sec-Only).
_CONDITION_DISPLAY = {
    "sec-only": "Sec-Only",
    "sec only": "Sec-Only",
    "seconly": "Sec-Only",
}


def _display_condition(cond) -> str:
    """Return the publication display label for a condition grouping key.
    Only rewrites the internal sec-only key to title-case 'Sec-Only';
    every other label (NT ASO / KD ASO / WT / KO / ...) passes through
    unchanged."""
    try:
        key = str(cond).strip().lower()
    except Exception:
        return str(cond)
    return _CONDITION_DISPLAY.get(key, str(cond))

# Pull the shared Okabe-Ito palette + per-condition mapping so colors are
# stable between this exploratory panel and the publication figures.
try:
    from ..visualization.publication_figures import (
        OKABE_ITO,
        CONDITION_COLORS,
        _canonicalize_condition,
    )
except Exception:  # pragma: no cover — script run before package install
    OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#F0E442",
                 "#0072B2", "#D55E00", "#CC79A7", "#000000"]
    CONDITION_COLORS = {}
    def _canonicalize_condition(s):  # type: ignore[no-redef]
        return s


# Color palette: real-signal images get sky blue, sec-only stays neutral gray.
# (Both colorblind-safe; sky blue replaces the legacy matplotlib default for
# consistency with the publication-figure palette.)
COLOR_REAL = OKABE_ITO[1]   # sky blue
COLOR_SEC_ONLY = "#9e9e9e"  # neutral gray (intentional — gray reads as
                            # "background / control" everywhere in the panel)

# 2026-05-18 Brian: condition palette ≠ channel palette.
# Earlier figures used yellow for WT (matching the RNA1 LUT) and that
# collided with channel identity — a yellow WT-RNA1 box could not be
# disambiguated from a yellow KO-RNA2 box just by color. Channel identity
# now lives ONLY in figure titles / axis labels ("...— RNA1"); per-condition
# coloring is uniform across every by-condition figure. Okabe-Ito palette
# below is colorblind-safe AND none of these colors overlap with the
# publication-image LUTs (yellow=RNA1, magenta=RNA2, blue=DAPI,
# green=antibody) — so the same plot can sit next to a publication image
# without visual confusion.
# 2026-06-03 Brian: LOCKED 2-tone convention applied to EVERY project's
# per-condition fallback hex (this dict feeds _color_for_condition /
# CONDITION_COLORS — the family-colormap path in _CONDITION_CMAP_BY_KEY was
# already on-palette). CONTROL-type (WT / NT / Control / no-dox) -> orange
# #E69F00; PERTURBATION-type (KO / KD / OE) -> blue #0072B2; secondary-only
# -> neutral grey. NO vermillion #D55E00, NO green #009E73, NO reddish-purple
# #CC79A7. Prior values were OFF-palette (KO=vermillion, KD=reddish-purple)
# AND the WT/KO mapping was INVERTED (WT=blue). Now fully consistent and
# colorblind-safe. Scope-safe: only changes figures actively regenerated
# (BIN1 curated deliverables are not re-rendered).
_LOCAL_CONDITION_COLORS = {
    "WT":       "#E69F00",  # control-type -> Okabe-Ito orange (was blue, inverted)
    "KO":       "#0072B2",  # perturbation -> Okabe-Ito blue (was vermillion #D55E00)
    "KD":       "#0072B2",  # perturbation -> Okabe-Ito blue (was reddish-purple #CC79A7)
    "NT":       "#E69F00",  # control-type -> Okabe-Ito orange
    "OE":       "#0072B2",  # perturbation -> Okabe-Ito blue (was green #009E73)
    "MIAT OE":  "#0072B2",  # exact data label -> blue
    "NT ASO":   "#E69F00",  # exact H9 data label -> orange (control)
    "KD ASO":   "#0072B2",  # exact H9 data label -> blue (perturbation)
    "Control":  "#E69F00",  # exact data label -> Okabe-Ito orange
    "control":  "#E69F00",  # Okabe-Ito orange (was yellow #F0E442)
    "sec-only": "#9e9e9e",  # neutral gray — same as COLOR_SEC_ONLY
}
# Merge into the shared CONDITION_COLORS map without clobbering existing
# canonical entries (publication_figures.py owns the canonical map).
for _k, _v in _LOCAL_CONDITION_COLORS.items():
    CONDITION_COLORS.setdefault(_k, _v)

# Compartment colors: bluish green (nuclear) vs orange (cytoplasmic).
# The previous green+red pairing violates the no-red+green accessibility
# rule. Both colors below are Okabe-Ito and remain distinguishable for
# deuteranopia / protanopia readers.
#
# 2026-06-03 Brian: these are SHARED across projects (BIN1 / H9 / MIAT-OE).
# The default stays bluish-green/orange so already-finalized BIN1 and H9
# figures re-render byte-for-byte. A project whose LOCKED palette forbids
# green (e.g. MIAT-OE = orange+blue, NO GREEN) can override the nuclear /
# cytoplasmic compartment colors via env vars WITHOUT touching the default:
#   COMPARTMENT_NUCLEAR_COLOR / COMPARTMENT_CYTO_COLOR
# (the MIAT-OE downstream runner sets nuclear=blue #0072B2, cyto=orange).
import os as _os_compat
COLOR_NUCLEAR = _os_compat.environ.get(
    "COMPARTMENT_NUCLEAR_COLOR", OKABE_ITO[2])      # default bluish green
COLOR_CYTOPLASMIC = _os_compat.environ.get(
    "COMPARTMENT_CYTO_COLOR", OKABE_ITO[0])         # default orange

# 2026-06-03 Brian: the Okabe-Ito GREEN (#009E73, == OKABE_ITO[2]) is also
# used as a CATEGORICAL swatch in a couple of shared by-condition figures —
# the threshold triplet (≥1 / ≥5 / ≥10 spots) and the spot-count bin
# composition (0 / 1-4 / 5-9 / 10+). That green appears identically in BOTH
# conditions (it encodes the threshold/bin, not the condition), so it is NOT
# the OE-condition regression — but a project whose LOCKED palette forbids
# green entirely (MIAT-OE) can override JUST these categorical swatches via
# env var WITHOUT touching the OKABE_ITO list or any other project's figures.
# Default unchanged (green) so BIN1/H9 re-render byte-for-byte.
CATEGORICAL_GREEN = _os_compat.environ.get(
    "CATEGORICAL_GREEN_COLOR", OKABE_ITO[2])        # default bluish green

# 2026-05-19 Brian: when stacked bars for MULTIPLE conditions appear in the
# same panel and use a fixed two-tone fill (e.g. nuclear=blue / cyto=orange),
# the WT and KO bars are visually indistinguishable until you read the axis
# label. Encode the condition on the bar EDGE (sky-blue / vermillion /
# reddish-purple / bluish-green per the CONDITION_COLORS map) while keeping
# the segment fill encoding compartment / category. Result: each bar carries
# both axes of information independently — eye can read WT vs KO from the
# outline, and nuclear vs cyto (or count bin) from the fill, simultaneously.
# 2026-06-05 Brian: STRIP the condition-colored bar outline from ALL stacked
# composition bars. It was redundant with the condition LABEL beside each bar
# and competed with the stacked-bin fill colors. Setting this to 0 (plus
# edgecolor="none" at every barh call below) removes the edge everywhere this
# pattern is used, in one place. Kept as a named constant for back-compat.
COND_EDGE_LINEWIDTH = 0  # outline removed (was 2.5)


def _condition_outline_legend_handles(conds: list[str]) -> list:
    """DEPRECATED (2026-06-05 Brian): the condition-colored bar outline was
    removed from all composition stacked-bar plots, so there is no longer an
    outline to explain in the legend. Retained as a no-op returning [] for
    back-compat with any external caller. The condition is read from the
    LABEL beside each bar instead.
    """
    return []


def _apply_combined_legend(ax, fill_handles_labels: list, conds: list[str],
                           loc: str = "lower right", fontsize: int = 8,
                           bbox_to_anchor=None) -> None:
    """Render the legend for the stacked-bin FILL artists (e.g. Nuclear /
    Cytoplasmic, or the 0 / 1-4 / 5-9 / 10+ count bins).

    2026-06-05 Brian: the "Condition (outline):" block was REMOVED along with
    the condition-colored bar outline — the condition is now read from the
    LABEL beside each bar, so duplicating it in the legend was redundant. The
    ``conds`` argument is retained for signature back-compat but no longer
    adds an outline block.

    ``bbox_to_anchor`` (optional) places the legend OUTSIDE the axes (e.g.
    (1.01, 1.0) with loc="upper left" puts it to the right of the data) so
    it never overlaps a bar/segment; the caller reserves the matching margin.
    """
    # fill_handles_labels: list of (handle, label) tuples for the fill segments
    handles = [h for (h, _) in fill_handles_labels]
    labels = [l for (_, l) in fill_handles_labels]
    _leg_kw = dict(handles=handles, labels=labels, loc=loc,
                   fontsize=fontsize, framealpha=0.9)
    if bbox_to_anchor is not None:
        _leg_kw["bbox_to_anchor"] = bbox_to_anchor
        _leg_kw["borderaxespad"] = 0.0
    ax.legend(**_leg_kw)


# 2026-05-18 Brian: sec-only files (e.g. "WT 2nd Ab only-100x08.vsi",
# "KO 2nd Ab only-100x05.vsi") historically inherited their parent
# subfolder's biological condition string ("WT" / "KO") because they live
# inside the WT/ or KO/ subdirectory. The boolean ``secondary_only`` column
# correctly flags them, but the by-condition plots group by the condition
# string itself — so sec-only rows leak into the WT and KO boxes as a
# spurious zero-signal sub-cluster. Remapping these rows to a single
# "sec-only" condition BEFORE any plotting fixes the leak and gives
# sec-only its own visually-neutral gray column.
SEC_ONLY_CONDITION = "sec-only"


def _remap_sec_only_to_own_group(df: pd.DataFrame) -> pd.DataFrame:
    """Rewrite ``condition`` to ``"sec-only"`` for every row where the
    ``secondary_only`` flag is True. No-op when either column is missing
    (so legacy CSVs without ``secondary_only`` continue to render exactly
    as before). Returns a NEW dataframe; the caller's input is untouched."""
    if "secondary_only" not in df.columns or "condition" not in df.columns:
        return df
    # Coerce flag — CSVs round-trip as 'True' / 'False' strings.
    sec_mask = df["secondary_only"].astype(str).str.lower() == "true"
    if not sec_mask.any():
        return df
    df = df.copy()
    df.loc[sec_mask, "condition"] = SEC_ONLY_CONDITION
    return df


def load_outputs(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Load nuclei + spots + (optional) per-image summary CSVs.

    Applies the sec-only condition-remap at load time so EVERY downstream
    plot sees ``condition == "sec-only"`` for those rows — there is no
    other code path that strips them later. By-condition groupings
    therefore show three columns (WT, KO, sec-only) instead of leaking
    sec-only rows into WT/KO boxes.
    """
    nuc_path = output_dir / "nuclei_metrics.csv"
    spot_path = output_dir / "spot_metrics.csv"
    sum_path = output_dir / "per_image_summary.csv"
    if not nuc_path.exists():
        raise SystemExit(f"Missing required nuclei_metrics.csv in {output_dir}")
    nuc = pd.read_csv(nuc_path)
    spots = pd.read_csv(spot_path) if spot_path.exists() else pd.DataFrame()
    summary = pd.read_csv(sum_path) if sum_path.exists() else None
    nuc = _remap_sec_only_to_own_group(nuc)
    if len(spots):
        spots = _remap_sec_only_to_own_group(spots)
    if summary is not None and len(summary):
        summary = _remap_sec_only_to_own_group(summary)
    return nuc, spots, summary


def load_analysis_mode(output_dir: Path) -> str:
    """Return the ANALYSIS_MODE recorded in run_config.json.

    Recognized values include 'rna_only' (default), 'rna_rna', 'rna_protein',
    'protein_only', 'ab_ab'. Missing/invalid run_config falls back to
    'rna_only' so legacy Fiji output directories continue to work.
    """
    import json
    cfg = output_dir / "run_config.json"
    if not cfg.exists():
        return "rna_only"
    try:
        data = json.loads(cfg.read_text())
        mode = data.get("ANALYSIS_MODE") or data.get("analysis_mode")
        if mode:
            return str(mode)
        # Fallback: peek at resolved config (fishsuite layout)
        resolved = data.get("config_resolved", {})
        ch = (resolved.get("channels") or {}) if isinstance(resolved, dict) else {}
        m2 = ch.get("analysis_mode")
        if m2:
            return str(m2)
    except Exception:
        pass
    return "rna_only"


def load_condition_order(output_dir: Path) -> list[str]:
    """Read the user-defined CONDITION_ORDER from run_config.json if set.
    Returns an ordered list of condition labels; missing or invalid values
    fall back to alphabetical (the caller handles the alphabetical default
    when this returns an empty list)."""
    import json
    cfg = output_dir / "run_config.json"
    if not cfg.exists():
        return []
    try:
        data = json.loads(cfg.read_text())
        order = data.get("CONDITION_ORDER")
        if isinstance(order, list):
            return [str(c) for c in order if str(c).strip()]
        return []
    except Exception:
        return []


def order_conditions(conditions: list[str], preferred: list[str]) -> list[str]:
    """Return `conditions` reordered so that anything in `preferred` comes
    first in the listed order, with the rest alphabetical at the end.
    Used to drive consistent left-to-right ordering across the box plots
    and the comparison-stats tables.

    2026-05-18 Brian: ``sec-only`` is always pinned LAST regardless of
    where it appears in ``preferred`` / alphabetical order, so the
    biological conditions (WT, KO, ...) always read left-to-right with
    the control column on the far right. Sec-only is a no-probe staining
    control, not a biological treatment, and visually anchoring it last
    keeps WT-vs-KO comparisons immediate.
    """
    has_sec = SEC_ONLY_CONDITION in conditions
    bio_conds = [c for c in conditions if c != SEC_ONLY_CONDITION]
    bio_pref = [c for c in preferred if c in bio_conds]
    rest = sorted(c for c in bio_conds if c not in bio_pref)
    ordered = bio_pref + rest
    if has_sec:
        ordered.append(SEC_ONLY_CONDITION)
    return ordered


def load_metadata(output_dir: Path) -> dict:
    """Read EXPERIMENT_METADATA from run_config.json if present.
    Returns empty dict if no metadata or no config."""
    import json
    cfg = output_dir / "run_config.json"
    if not cfg.exists():
        return {}
    try:
        data = json.loads(cfg.read_text())
        return data.get("EXPERIMENT_METADATA", {}) or {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Config-driven channel labels (2026-05-19 Brian).
#
# Every USER-FACING reference to "RNA1" / "RNA2" / "DAPI" / "Exons" / "Introns"
# in this file's figure titles, axis labels, legends, and burned-in text gets
# remapped at render time to whatever the preset YAML's channels.rna_label,
# channels.rna2_label, channels.dapi_label resolved to. INTERNAL data API
# names (column names like ``n_active_tss_per_nucleus``,
# ``n_nuclear_rna1_rna2_overlap_per_nucleus``) are deliberately UNCHANGED —
# they are the contract between the pipeline CSVs and downstream tooling.
#
# The substitution is a string-walk post-pass over the figure's text artists,
# applied immediately before each ``fig.savefig()`` call (or before
# ``plt.close()`` for nested standalone-subplot renders). One token table
# loaded once per ``main()`` invocation; live preset edits propagate because
# load_labels() reads run_config.json on demand.
# ---------------------------------------------------------------------------


def load_labels(output_dir: Path) -> dict:
    """Read channel display labels from ``run_config.json``.

    Looks at ``config_resolved.channels.{rna_label, rna2_label, dapi_label}``
    (where fishsuite writes them) and falls back to the top-level
    ``CHANNEL_RNA_LABEL`` / ``CHANNEL_RNA2_LABEL`` / ``CHANNEL_DAPI_LABEL``
    keys that the runner also promotes. Defaults match the legacy hardcoded
    figure text so old run_config.json files (and missing files) render the
    same as before.
    """
    import json
    # 2026-05-29 Brian: ``antibody_label`` is also loaded so the rna_protein
    # coloc SuperPlots (07_coloc/) can title the second channel by its antibody
    # name (e.g. "XRN2") instead of the generic "protein". It is NOT fed to the
    # token-substitution _relabel_fig path (which only swaps RNA1/RNA2/DAPI) —
    # the coloc functions read it directly from _LABELS.
    out = {"rna_label": "RNA1", "rna2_label": "RNA2", "dapi_label": "DAPI",
           "antibody_label": "Protein"}
    cfg = output_dir / "run_config.json"
    if not cfg.exists():
        return out
    try:
        data = json.loads(cfg.read_text())
    except Exception:
        return out
    resolved = data.get("config_resolved", {}) or {}
    ch = (resolved.get("channels") or {}) if isinstance(resolved, dict) else {}
    for key, top_key in (
        ("rna_label",  "CHANNEL_RNA_LABEL"),
        ("rna2_label", "CHANNEL_RNA2_LABEL"),
        ("dapi_label", "CHANNEL_DAPI_LABEL"),
        ("antibody_label", "CHANNEL_ANTIBODY_LABEL"),
    ):
        v = ch.get(key) or data.get(top_key)
        if v:
            out[key] = str(v)
    return out


# Module-level label cache (set by main()). Falls through to the legacy
# defaults so functions that render before main() initializes (tests,
# standalone calls) still work.
_LABELS = {"rna_label": "RNA1", "rna2_label": "RNA2", "dapi_label": "DAPI",
           "antibody_label": "Protein"}


def _set_labels(labels: dict) -> None:
    """Install the active label dict (called once by main() at startup)."""
    global _LABELS
    _LABELS = {
        "rna_label":  str(labels.get("rna_label", "RNA1") or "RNA1"),
        "rna2_label": str(labels.get("rna2_label", "RNA2") or "RNA2"),
        "dapi_label": str(labels.get("dapi_label", "DAPI") or "DAPI"),
        # 2026-05-29 Brian: carried for the rna_protein coloc SuperPlots only;
        # NOT a token substituted by _relabel_fig (which handles RNA1/RNA2/DAPI).
        "antibody_label": str(labels.get("antibody_label", "Protein") or "Protein"),
    }


def _subst_user_text(s: str) -> str:
    """Replace hardcoded channel tokens in a USER-FACING string.

    Substitutes "RNA1" -> rna_label, "RNA2" -> rna2_label, "DAPI" -> dapi_label
    only when they appear as standalone tokens (not inside column-name
    substrings like ``frac_nuclear_rna1`` which are already lowercase). The
    matcher requires the token to NOT be immediately preceded or followed by
    an alphanumeric character or underscore, so ``rna_spot_count`` and
    similar internal identifiers are untouched.

    Order matters: RNA2 must substitute before RNA1 so "RNA1↔RNA2" doesn't
    half-replace.
    """
    if not isinstance(s, str) or not s:
        return s
    import re as _re
    rna1 = _LABELS["rna_label"]
    rna2 = _LABELS["rna2_label"]
    dapi = _LABELS["dapi_label"]
    # No-op fast path when labels match the defaults.
    if rna1 == "RNA1" and rna2 == "RNA2" and dapi == "DAPI":
        return s
    # Word-boundary substitution. The token boundaries here include hyphens,
    # arrows, parentheses, slashes, spaces — anything non-[A-Za-z0-9_].
    out = s
    out = _re.sub(r"(?<![A-Za-z0-9_])RNA2(?![A-Za-z0-9_])", rna2, out)
    out = _re.sub(r"(?<![A-Za-z0-9_])RNA1(?![A-Za-z0-9_])", rna1, out)
    # Only swap "DAPI" if the preset's dapi_label is non-default (DAPI is a
    # near-universal stain name and Brian's presets keep it).
    if dapi != "DAPI":
        out = _re.sub(r"(?<![A-Za-z0-9_])DAPI(?![A-Za-z0-9_])", dapi, out)
    return out


def _relabel_fig(fig) -> None:
    """Walk every text artist in ``fig`` and apply ``_subst_user_text`` to
    its string value. Covers axis titles, x/y labels, tick labels, legend
    entries, suptitle, free-form fig.text() annotations, and ax.text()
    annotations. Called immediately before ``fig.savefig``.

    No-op when the active label dict matches defaults (so legacy runs are
    bit-for-bit identical to before this helper existed).
    """
    if (_LABELS["rna_label"] == "RNA1"
            and _LABELS["rna2_label"] == "RNA2"
            and _LABELS["dapi_label"] == "DAPI"):
        return
    try:
        # Figure-level text artists (suptitle + fig.text() calls).
        for txt in list(fig.texts):
            try:
                txt.set_text(_subst_user_text(txt.get_text()))
            except Exception:
                pass
        for ax in fig.axes:
            # Axis titles + x/y labels.
            try:
                ax.set_title(_subst_user_text(ax.get_title()))
            except Exception:
                pass
            try:
                ax.set_xlabel(_subst_user_text(ax.get_xlabel()))
            except Exception:
                pass
            try:
                ax.set_ylabel(_subst_user_text(ax.get_ylabel()))
            except Exception:
                pass
            # Tick labels (x then y). matplotlib may return Text() with
            # empty strings before the figure is drawn — guard against that.
            try:
                xtl = [t.get_text() for t in ax.get_xticklabels()]
                if any(xtl):
                    # Pin the locator to the current tick positions before
                    # relabeling so matplotlib doesn't warn about a mismatch
                    # between a free locator and a fixed label list.
                    import matplotlib.ticker as _mt
                    ax.xaxis.set_major_locator(_mt.FixedLocator(ax.get_xticks()))
                    ax.set_xticklabels([_subst_user_text(t) for t in xtl])
            except Exception:
                pass
            try:
                ytl = [t.get_text() for t in ax.get_yticklabels()]
                if any(ytl):
                    import matplotlib.ticker as _mt
                    ax.yaxis.set_major_locator(_mt.FixedLocator(ax.get_yticks()))
                    ax.set_yticklabels([_subst_user_text(t) for t in ytl])
            except Exception:
                pass
            # Legend (handles + texts). Legend texts are Text artists too.
            try:
                leg = ax.get_legend()
                if leg is not None:
                    for txt in leg.get_texts():
                        try:
                            txt.set_text(_subst_user_text(txt.get_text()))
                        except Exception:
                            pass
            except Exception:
                pass
            # In-axes ax.text() annotations.
            try:
                for txt in list(ax.texts):
                    txt.set_text(_subst_user_text(txt.get_text()))
            except Exception:
                pass
    except Exception:
        # Never let a relabel pass crash a render.
        pass


_LEADING_NUM_RE = __import__("re").compile(r"^(\d+)")
# Match "well1", "well_1", "wells3-4", "wells3_4", or "control..." — keeps
# trailing dashes/digits so 'wells3-4' isn't truncated to 'wells3'.
_WELL_RE = __import__("re").compile(r"(wells?[\w\-]+|control[\w\-]*)", __import__("re").IGNORECASE)


def short_label(image_name: str) -> str:
    """Squeeze a long Fiji filename into something legible. Tries to keep
    the leading numeric prefix and any well/control identifier — falls back
    to the first 16 chars of the stem.

    Examples:
        "01_BA-04-24-2026-RNA-FISH-MIAT-H9-1-well1.vsi"     -> "01 well1"
        "08_BA-04-24-2026-RNA-FISH-MIAT-H9-1-wells3-4.vsi"  -> "08 wells3-4"
        "control_replicate2.vsi"                            -> "control_replicate2"
    """
    stem = image_name.rsplit(".", 1)[0]
    num = _LEADING_NUM_RE.match(stem)
    well = _WELL_RE.search(stem)
    if num and well:
        return f"{num.group(1)} {well.group(1)}"
    if well:
        return well.group(1)
    if num:
        return num.group(1)
    return stem[:16]


def display_label(image_name: str, condition: str | None = None,
                  unique_per_condition: bool = False) -> str:
    """Build the legend label for an image. Prefers the user-assigned
    condition label since that's what the experimenter actually wants to
    see. When multiple images share a condition (so the condition alone
    is ambiguous), suffixes the short image identifier in parentheses.

    Defensive against pandas NaN / non-string condition values (the
    nuclei_metrics CSV occasionally has float-NaN in the condition column
    when the GUI didn't assign one; coercing to "nan" then dropping it).
    """
    # Coerce to a clean string. None / NaN / "nan" all degrade to no-label.
    try:
        if condition is None:
            cond = ""
        elif isinstance(condition, float) and condition != condition:  # NaN
            cond = ""
        else:
            cond = str(condition).strip()
            if cond.lower() == "nan":
                cond = ""
    except Exception:
        cond = ""
    short = short_label(image_name)
    if cond and cond.lower() not in ("all", "none", "unknown"):
        # Title-case the internal sec-only key to the publication display name
        # (Sec-Only) so every per-image legend / tick label reads consistently.
        cond_disp = _display_condition(cond)
        return f"{cond_disp} ({short})" if unique_per_condition else cond_disp
    return short


def _build_image_labels(df: pd.DataFrame, image_col: str = "image",
                        cond_col: str = "condition") -> dict[str, str]:
    """Return {image_name: legend_label}. Uses condition labels when each
    condition maps to one image; falls back to '<condition> (<short>)' or
    just the short image identifier when conditions are missing/ambiguous."""
    if image_col not in df.columns:
        return {}
    if cond_col in df.columns:
        # Count distinct images per condition
        per_cond = df.groupby(cond_col)[image_col].nunique().to_dict()
    else:
        per_cond = {}
    out: dict[str, str] = {}
    for img in df[image_col].unique():
        sub = df[df[image_col] == img]
        cond = str(sub[cond_col].iloc[0]) if cond_col in sub.columns and len(sub) else ""
        ambiguous = per_cond.get(cond, 1) > 1
        out[img] = display_label(img, cond, unique_per_condition=ambiguous)
    return out


def is_secondary_only(row) -> bool:
    """Coerce the secondary_only column to bool; CSVs may have True/False/'True'/'False'."""
    v = row.get("secondary_only", False)
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() == "true"
    return bool(v)


def plot_spots_per_nucleus(ax, nuc: pd.DataFrame) -> None:
    """Histogram of rna_spot_count, colored per image, sec-only grayed."""
    if "rna_spot_count" not in nuc.columns:
        ax.set_visible(False)
        return
    images = nuc["image"].unique()
    labels = _build_image_labels(nuc)
    family_map = _build_family_color_map(nuc)
    max_count = int(nuc["rna_spot_count"].max() if len(nuc) else 1)
    bins = np.arange(0, max(max_count + 2, 20), 1) if max_count <= 50 else 30

    for img_name in images:
        sub = nuc[nuc["image"] == img_name]
        color = _image_family_color(family_map, img_name)
        ax.hist(
            sub["rna_spot_count"], bins=bins,
            alpha=0.5, label=f"{labels[img_name]} (n={len(sub)})",
            color=color,
        )
    ax.set_xlabel("RNA spots per nucleus")
    ax.set_ylabel("Number of nuclei")
    ax.set_title(_wrap_title("Spots-per-nucleus distribution\n"
                             "(color = condition family; shade = image)"))
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)


def plot_spot_count_per_image(ax, summary: pd.DataFrame | None, nuc: pd.DataFrame) -> None:
    """Bar chart: total + mean spots per image. Sec-only flagged."""
    if summary is not None and len(summary) > 0:
        labels_map = _build_image_labels(summary)
        labels = [labels_map.get(s, short_label(s)) for s in summary["image"]]
        means = summary["mean_spots_per_nucleus"]
        secs = [
            (str(v).lower() == "true") for v in summary["secondary_only"]
        ] if "secondary_only" in summary.columns else [False] * len(summary)
        colors = [COLOR_SEC_ONLY if s else COLOR_REAL for s in secs]
        ax.bar(labels, means, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_ylabel("Mean spots / nucleus")
        ax.set_title(_wrap_title("Mean spots per nucleus, by image\n(sec-only in gray)"))
        ax.tick_params(axis="x", rotation=20)
        for i, (m, s) in enumerate(zip(means, secs)):
            ax.text(i, m, f"{m:.1f}", ha="center", va="bottom", fontsize=9)
    else:
        labels_map = _build_image_labels(nuc)
        agg = nuc.groupby("image")["rna_spot_count"].mean().sort_index()
        ax.bar([labels_map.get(i, short_label(i)) for i in agg.index],
               agg.values, color=COLOR_REAL, edgecolor="black")
        ax.set_ylabel("Mean spots / nucleus")
        ax.set_title(_wrap_title("Mean spots per nucleus, by image"))
        ax.tick_params(axis="x", rotation=20)
    ax.grid(True, alpha=0.3, axis="y")


def plot_intensity_distribution(ax, spots: pd.DataFrame) -> None:
    """Per-image overlaid histograms of spot peak intensity."""
    if spots.empty or "spot_peak_intensity" not in spots.columns:
        ax.set_visible(False)
        return
    images = spots["image"].unique()
    labels = _build_image_labels(spots)
    family_map = _build_family_color_map(spots)
    for img_name in images:
        sub = spots[spots["image"] == img_name]
        color = _image_family_color(family_map, img_name)
        ax.hist(
            sub["spot_peak_intensity"], bins=40,
            alpha=0.5, label=f"{labels[img_name]} (n={len(sub)})",
            color=color,
        )
    ax.set_xlabel("Spot peak intensity (raw)")
    ax.set_ylabel("Spot count")
    ax.set_title(_wrap_title("Spot peak-intensity distribution\n"
                             "(color = condition family; shade = image)"))
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    counts_per_image = [(spots["image"] == img).sum() for img in images]
    if counts_per_image and max(counts_per_image) / max(min(counts_per_image), 1) > 50:
        ax.set_yscale("log")


def plot_diameter_distribution(ax, spots: pd.DataFrame) -> None:
    """Per-image overlaid histograms of spot diameter."""
    if spots.empty or "spot_diameter_um" not in spots.columns:
        ax.set_visible(False)
        return
    images = spots["image"].unique()
    labels = _build_image_labels(spots)
    finite = spots["spot_diameter_um"].dropna()
    if finite.empty:
        ax.set_visible(False)
        return
    bins = np.linspace(0, max(finite.max(), 0.001), 40)
    family_map = _build_family_color_map(spots)
    for img_name in images:
        sub = spots[spots["image"] == img_name]
        color = _image_family_color(family_map, img_name)
        ax.hist(
            sub["spot_diameter_um"].dropna(), bins=bins,
            alpha=0.5, label=f"{labels[img_name]} (n={len(sub.dropna(subset=['spot_diameter_um']))})",
            color=color,
        )
    ax.set_xlabel("Spot diameter (µm)")
    ax.set_ylabel("Spot count")
    ax.set_title(_wrap_title("Spot size distribution\n"
                             "(color = condition family; shade = image)"))
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)


def plot_nuclear_vs_cytoplasmic(ax, spots: pd.DataFrame,
                                condition_order: list[str] | None = None) -> None:
    """Stacked bar per image: nuclear vs cytoplasmic spot counts."""
    if spots.empty or "nucleus_id" not in spots.columns:
        ax.set_visible(False)
        return
    by_img = spots.groupby("image").agg(
        nuclear=("nucleus_id", lambda s: int((s > 0).sum())),
        cytoplasmic=("nucleus_id", lambda s: int((s == 0).sum())),
    )
    # Order images by CONDITION (deck order: NT → KD → Sec-Only), then image
    # name within, so the bars read in the same left-to-right condition order
    # as the rest of the deck instead of a raw lexical image sort.
    if "condition" in spots.columns:
        img_cond = spots.groupby("image")["condition"].first().to_dict()
        cond_order_local = order_conditions(
            list({str(c) for c in img_cond.values()}), condition_order or [])
        rank = {c: i for i, c in enumerate(cond_order_local)}
        ordered_imgs = sorted(
            by_img.index,
            key=lambda im: (rank.get(str(img_cond.get(im, "")), 999), str(im)))
        by_img = by_img.loc[ordered_imgs]
    else:
        by_img = by_img.sort_index()
    label_map = _build_image_labels(spots)
    labels = [label_map.get(i, short_label(i)) for i in by_img.index]
    x = np.arange(len(labels))
    ax.bar(x, by_img["nuclear"], label="nuclear", color=COLOR_NUCLEAR,
           edgecolor="black", linewidth=0.5)
    ax.bar(x, by_img["cytoplasmic"], bottom=by_img["nuclear"], label="cytoplasmic",
           color=COLOR_CYTOPLASMIC, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("Total spot count")
    ax.set_title(_wrap_title("Nuclear vs cytoplasmic spots per image"))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")


def plot_intensity_by_compartment(ax, spots: pd.DataFrame) -> None:
    """Stacked / overlaid histograms of spot peak intensity, split by
    nuclear vs cytoplasmic location. Useful for spotting whether
    cytoplasmic spots are systematically dimmer (likely noise) or
    comparable to nuclear (likely real biology)."""
    if spots.empty or "spot_peak_intensity" not in spots.columns or "nucleus_id" not in spots.columns:
        ax.set_visible(False)
        return
    real = spots[spots["secondary_only"].astype(str).str.lower() != "true"] if "secondary_only" in spots.columns else spots
    if real.empty:
        ax.set_visible(False)
        return
    nuclear = real[real["nucleus_id"] > 0]["spot_peak_intensity"].dropna()
    cyto = real[real["nucleus_id"] == 0]["spot_peak_intensity"].dropna()
    if nuclear.empty and cyto.empty:
        ax.set_visible(False)
        return
    bins = np.linspace(0, max(real["spot_peak_intensity"].max(), 1), 40)
    if not nuclear.empty:
        ax.hist(nuclear, bins=bins, alpha=0.6, label=f"nuclear (n={len(nuclear)})",
                color=COLOR_NUCLEAR)
    if not cyto.empty:
        ax.hist(cyto, bins=bins, alpha=0.6, label=f"cytoplasmic (n={len(cyto)})",
                color=COLOR_CYTOPLASMIC)
    ax.set_xlabel("Spot peak intensity (raw)")
    ax.set_ylabel("Spot count")
    ax.set_title(_wrap_title("Spot intensity by compartment\n(real-signal images only)"))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_cumulative_spots_per_cell(ax, nuc: pd.DataFrame) -> None:
    """Cumulative distribution function of spots-per-cell, per image.
    Reads as: 'fraction of cells with ≤ X spots'. Sec-only's CDF rises
    sharply (most cells have 0 spots); real images have a gentler ramp."""
    if "rna_spot_count" not in nuc.columns:
        ax.set_visible(False)
        return
    images = nuc["image"].unique()
    labels = _build_image_labels(nuc)
    family_map = _build_family_color_map(nuc)
    for img_name in images:
        sub = nuc[nuc["image"] == img_name]
        color = _image_family_color(family_map, img_name)
        counts = sorted(sub["rna_spot_count"])
        if not counts:
            continue
        x = np.array(counts)
        y = np.arange(1, len(x) + 1) / len(x)
        ax.plot(x, y, marker=".", linestyle="-", alpha=0.7,
                label=f"{labels[img_name]} (n={len(x)})", color=color)
    ax.set_xlabel("Spots per nucleus")
    ax.set_ylabel("Cumulative fraction of nuclei")
    ax.set_title(_wrap_title("CDF: nuclei with ≤ X spots (steeper = noise / no expression)\n"
                             "(color = condition family; shade = image)"))
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.02)


def plot_per_cell_expression(ax, nuc: pd.DataFrame) -> None:
    """Per-cell mean MIAT intensity histogram, per image. Reads as the
    expression-distribution view: are there clearly bimodal populations
    (high vs low expressors), or a continuous gradient?

    Uses rna_spot_mean_intensity_bgc_blend (mean BG-corrected intensity
    across spots in each nucleus). Cells with zero spots are excluded
    so the distribution reflects expression strength among expressing
    cells, not the on/off fraction (which is in the spots-per-nucleus
    plot)."""
    col = "rna_spot_mean_intensity_bgc_blend"
    if col not in nuc.columns:
        ax.set_visible(False)
        return
    expr = nuc[(nuc.get("rna_spot_count", 0) > 0) & nuc[col].notna() & (nuc[col] > 0)]
    if expr.empty:
        ax.set_visible(False)
        return

    images = expr["image"].unique()
    labels = _build_image_labels(expr)
    vmax = float(expr[col].quantile(0.99))
    bins = np.linspace(0, vmax * 1.05 if vmax > 0 else 1, 30)
    family_map = _build_family_color_map(expr)
    for img_name in images:
        sub = expr[expr["image"] == img_name]
        color = _image_family_color(family_map, img_name)
        ax.hist(sub[col], bins=bins, alpha=0.55,
                label=f"{labels[img_name]} (n={len(sub)})", color=color)
        med = sub[col].median()
        ax.axvline(med, color=color or COLOR_REAL, linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Mean spot intensity per cell (BG-corrected)")
    ax.set_ylabel("Number of cells")
    ax.set_title(_wrap_title("Per-cell expression intensity (spots-positive cells only; dashed = median)\n"
                             "(color = condition family; shade = image)"))
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)


def plot_total_rna_per_cell(ax, nuc: pd.DataFrame) -> None:
    """Per-cell TOTAL RNA mass distribution. Total = sum of integrated
    intensities across all spots in that cell. This is the abundance
    metric — separates "1 super-bright spot" cells from "many dim spots"
    cells. Logged x-axis if the dynamic range is large."""
    # Prefer the summed per-spot PEAK-pixel total when present; fall back to
    # the BG-corrected blend total (always present). The peak total is a
    # consistent intensity proxy (brightest voxel per spot), NOT a Gaussian
    # fit or a background-corrected value.
    col = _resolve_col(nuc, "rna_spot_total_peak_intensity", "rna_spot_total_intensity_fit")
    use_peak = col in nuc.columns
    if not use_peak or nuc[col].fillna(0).sum() == 0:
        col = "rna_spot_total_intensity_bgc_blend"
        use_peak = False
    if col not in nuc.columns:
        ax.set_visible(False)
        return

    expr = nuc[(nuc.get("rna_spot_count", 0) > 0) & nuc[col].notna() & (nuc[col] > 0)]
    if expr.empty:
        ax.set_visible(False)
        return

    images = expr["image"].unique()
    labels = _build_image_labels(expr)
    vmin = float(expr[col].quantile(0.01))
    vmax = float(expr[col].quantile(0.99))
    use_log = (vmax / max(vmin, 1)) > 100
    if use_log:
        bins = np.logspace(np.log10(max(vmin, 1)), np.log10(vmax * 1.05), 30)
    else:
        bins = np.linspace(0, vmax * 1.05, 30)

    family_map = _build_family_color_map(expr)
    for img_name in images:
        sub = expr[expr["image"] == img_name]
        color = _image_family_color(family_map, img_name)
        ax.hist(sub[col], bins=bins, alpha=0.55,
                label=f"{labels[img_name]} (n={len(sub)})", color=color)
    _kind = "peak" if use_peak else "BG-corrected"
    ax.set_xlabel(f"Summed {_kind} FISH intensity / nucleus (a.u.)")
    ax.set_ylabel("Number of nuclei")
    ax.set_title(_wrap_title("Per-nucleus total RNA intensity (distinguishes high vs low expressors)\n"
                             "(color = condition family; shade = image)"))
    if use_log:
        # Plain-decimal log axis via the shared thread-safe helper (no mathtext).
        _set_log_axis_plain(ax, "x")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)


def plot_distance_to_edge(ax, spots: pd.DataFrame) -> None:
    """Histogram of spot-to-nucleus-edge distances (nuclear spots only)."""
    if spots.empty or "spot_to_nuc_edge_um" not in spots.columns:
        ax.set_visible(False)
        return
    nuclear = spots[spots["nucleus_id"] > 0]
    if nuclear.empty:
        ax.set_visible(False)
        return
    images = nuclear["image"].unique()
    labels = _build_image_labels(nuclear)
    finite = nuclear["spot_to_nuc_edge_um"].dropna()
    if finite.empty:
        ax.set_visible(False)
        return
    bins = np.linspace(0, max(finite.max(), 0.001), 30)
    family_map = _build_family_color_map(nuclear)
    for img_name in images:
        sub = nuclear[nuclear["image"] == img_name]
        color = _image_family_color(family_map, img_name)
        d = sub["spot_to_nuc_edge_um"].dropna()
        if d.empty:
            continue
        ax.hist(d, bins=bins, alpha=0.5, label=f"{labels[img_name]} (n={len(d)})", color=color)
    ax.set_xlabel("Distance from spot to nuclear edge (µm)")
    ax.set_ylabel("Nuclear spot count")
    ax.set_title(_wrap_title("Spot peripheral-vs-interior bias (0 = at the envelope, larger = interior)\n"
                             "(color = condition family; shade = image)"))
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)


# Ordered fallback palette for arbitrary conditions. Drawn from Okabe-Ito
# (colorblind-safe) and tuned to match the publication-figure palette so the
# exploratory panel and the dissertation figures use the same colors.
# Yellow (OKABE_ITO[3]) is intentionally last because it reads poorly as a
# box fill on a white background.
_CONDITION_PALETTE = [
    OKABE_ITO[1],  # sky blue
    OKABE_ITO[5],  # vermillion
    OKABE_ITO[2],  # bluish green
    OKABE_ITO[0],  # orange
    OKABE_ITO[4],  # blue
    OKABE_ITO[6],  # reddish purple
    OKABE_ITO[7],  # black
    OKABE_ITO[3],  # yellow (last)
]


def _color_for_condition(cond: str, fallback_idx: int) -> str:
    """Resolve a stable color for a condition: prefer the shared
    CONDITION_COLORS map, fall back to the ordered palette by position.

    2026-05-18 Brian: tries the literal label first (so the lowercase
    "sec-only" we inject in _remap_sec_only_to_own_group hits the
    neutral-gray entry we registered above) before falling through to
    the publication-figure canonicalization. This keeps channel-color
    vs condition-color cleanly separated — no condition box should
    ever land on a color used for a channel (yellow / magenta /
    green / blue)."""
    if isinstance(cond, str) and cond in CONDITION_COLORS:
        return CONDITION_COLORS[cond]
    canon = _canonicalize_condition(cond) if isinstance(cond, str) else cond
    if canon in CONDITION_COLORS:
        return CONDITION_COLORS[canon]
    # Sec-only catch-all in case some upstream stripped the dash.
    if isinstance(cond, str) and cond.lower().replace(" ", "").replace("_", "").replace("-", "") in (
        "seconly", "secondaryonly", "sec"
    ):
        return CONDITION_COLORS.get(SEC_ONLY_CONDITION, COLOR_SEC_ONLY)
    return _CONDITION_PALETTE[fallback_idx % len(_CONDITION_PALETTE)]


# =====================================================================
# Condition color-FAMILIES with per-image hue variation (2026-05-25 Brian)
# =====================================================================
# Brian: every DISTRIBUTION-style figure (histogram / KDE / CDF / ranked /
# sorted-brightness) must color each line/patch by its CONDITION's color
# FAMILY, with a per-IMAGE hue gradient WITHIN that family so the eye reads
# "this is a KD image" (blue) vs "this is an NT image" (red/orange) vs
# "this is a Sec-Only image" (gray) at a glance, while still separating
# individual images by light->dark shade.
#
# Implementation: map each canonical condition to a matplotlib sequential
# colormap, then sample that colormap across the N images in the condition
# (avoiding the very-lightest / very-darkest extremes so every shade stays
# legible on white and distinct from the colormap's white endpoint).
#
# Mapping is driven by biology-NEUTRAL substring matching on the condition
# label (no assumption about what "KD"/"KO" does biologically — it is only a
# stable color assignment): KD/KO -> Blues, NT/WT/control -> Reds/Oranges,
# sec-only -> Greys, OE -> Greens. Unknown conditions cycle through a fixed
# list of remaining sequential colormaps so two unknown conditions never
# collide and never land on a red+green pairing within the same figure.
#
# These helpers are additive: the legacy ``color = COLOR_SEC_ONLY if sec
# else None`` callers are migrated to ``_image_family_color(...)`` so the
# recolor is on-by-default for every analysis mode without changing any
# function signature.
_CONDITION_CMAP_BY_KEY = {
    # canonical-ish keys (lowercased, dash/space-insensitive matching below)
    # 2026-05-29 Brian: lock the SuperPlot condition families to Okabe-Ito —
    # WT/NT = ORANGE (#E69F00 family, NOT pure red), KO/KD = BLUE (#0072B2
    # family). The Oranges colormap mid-tone (~0.62) lands on the Okabe-Ito
    # orange, so dots / per-image means / violins all read orange-vs-blue
    # (colorblind-safe, no red+green) consistently across every SuperPlot.
    "kd": "Blues",
    "ko": "Blues",
    "nt": "Oranges",
    "wt": "Oranges",
    "control": "Oranges",
    # 2026-06-03 Brian: MIAT-OE LOCKED palette = orange (Control) + blue (OE),
    # NO GREEN. Was "oe": "Greens" -> every MIAT-OE per-image shade + colormap
    # backdrop rendered green. OE is unique to this project (no WT/KO/NT/KD
    # condition normalizes to "oe"), so this is scope-safe for BIN1/H9.
    "oe": "Blues",
    "sec": "Greys",
    "seconly": "Greys",
    "secondaryonly": "Greys",
}
# Cmaps handed out to conditions that don't match any key above. Ordered so
# the first few are maximally distinct AND never produce a red+green clash
# for a typical 2–3 condition deck.
_FALLBACK_CMAPS = ["Blues", "Oranges", "Greys", "Purples", "BuGn", "PuRd"]


def _get_cmap(name):
    """Resolve a matplotlib colormap by name across versions. Prefers the
    non-deprecated ``matplotlib.colormaps[name]`` (mpl ≥3.5) and falls back to
    the legacy ``cm.get_cmap`` only if needed. Centralizes the lookup so the
    deprecation warning that was firing dozens of times per render is gone."""
    try:
        import matplotlib as _mpl
        return _mpl.colormaps[name]
    except Exception:
        import matplotlib.cm as _cm
        return _cm.get_cmap(name)


def _norm_cond_key(cond) -> str:
    """Normalize a condition string to a lookup key: lowercased with spaces,
    dashes and underscores stripped (so 'Sec-Only' / 'sec_only' / 'sec only'
    all collapse to 'seconly')."""
    try:
        c = str(cond).strip().lower()
    except Exception:
        return ""
    return c.replace(" ", "").replace("-", "").replace("_", "")


def _cmap_for_condition(cond, fallback_idx: int = 0) -> str:
    """Resolve the matplotlib sequential-colormap NAME for a condition's
    color family. Substring match on the normalized key so 'kdaso' -> Blues,
    'ntaso' -> Reds, 'seconly' -> Greys. Falls back to a stable cycling list
    keyed by ``fallback_idx`` for unrecognized conditions."""
    key = _norm_cond_key(cond)
    if not key:
        return _FALLBACK_CMAPS[fallback_idx % len(_FALLBACK_CMAPS)]
    # Direct hit first (cheap), then substring scan (so 'kdaso' matches 'kd').
    if key in _CONDITION_CMAP_BY_KEY:
        return _CONDITION_CMAP_BY_KEY[key]
    for k, cm in _CONDITION_CMAP_BY_KEY.items():
        if key.startswith(k) or k in key:
            return cm
    return _FALLBACK_CMAPS[fallback_idx % len(_FALLBACK_CMAPS)]


def _build_family_color_map(df: pd.DataFrame,
                            condition_order: list[str] | None = None,
                            image_col: str = "image",
                            cond_col: str = "condition") -> dict:
    """Return {image_name: hex_color}: each image colored by its condition's
    family colormap, with a light->dark hue gradient ACROSS the images within
    that condition. Images are sorted by name within each condition so the
    shade order is deterministic and stable across re-runs.

    The sampled fraction range is [0.45, 0.92] (mid-to-dark) so even a
    single-image condition gets a saturated, legible shade rather than the
    near-white colormap endpoint.
    """
    import matplotlib.cm as _cm
    if image_col not in df.columns:
        return {}
    out: dict = {}
    if cond_col in df.columns:
        conds_in = df[cond_col].dropna().unique().tolist()
        conds = order_conditions(conds_in, condition_order or [])
    else:
        conds = [None]
    for ci, cond in enumerate(conds):
        if cond is None:
            sub_imgs = sorted(str(i) for i in df[image_col].dropna().unique())
        else:
            sub_imgs = sorted(
                str(i) for i in df.loc[df[cond_col] == cond, image_col].dropna().unique()
            )
        if not sub_imgs:
            continue
        cmap = _get_cmap(_cmap_for_condition(cond, ci))
        n = len(sub_imgs)
        if n == 1:
            fracs = [0.72]
        else:
            lo, hi = 0.45, 0.92
            fracs = [lo + (hi - lo) * k / (n - 1) for k in range(n)]
        for img, fr in zip(sub_imgs, fracs):
            r, g, b, _ = cmap(fr)
            out[img] = "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))
    return out


def _image_family_color(family_map: dict, img_name, fallback: str | None = None) -> str | None:
    """Look up an image's family color; tolerant of str/non-str keys."""
    if not family_map:
        return fallback
    return family_map.get(str(img_name), family_map.get(img_name, fallback))


# 2026-06-03 Brian: conditions whose base bar/dot/median color must land on
# the EXACT Okabe-Ito lock (not the colormap mid-tone approximation). Now
# covers EVERY project under the LOCKED 2-tone convention: control-type
# (WT / NT / Control) -> orange #E69F00; perturbation-type (KO / KD / OE) ->
# blue #0072B2. (The Blues/Oranges colormap mid-tones the other path samples
# already read orange-vs-blue, but pinning the exact hex guarantees the
# headline base color is byte-exact on the lock.) Sec-only is handled
# separately (neutral grey) and is not pinned here.
_LOCKED_EXACT_BASE_COLORS = {
    "control": "#E69F00",  # Okabe-Ito orange (Control)
    "wt":      "#E69F00",  # Okabe-Ito orange (WT control-type)
    "nt":      "#E69F00",  # Okabe-Ito orange (NT control-type)
    "ntaso":   "#E69F00",  # Okabe-Ito orange (NT ASO control-type)
    "miatoe":  "#0072B2",  # Okabe-Ito blue (MIAT OE)
    "oe":      "#0072B2",  # Okabe-Ito blue (OE shorthand)
    "ko":      "#0072B2",  # Okabe-Ito blue (KO perturbation)
    "kd":      "#0072B2",  # Okabe-Ito blue (KD perturbation)
    "kdaso":   "#0072B2",  # Okabe-Ito blue (KD ASO perturbation)
}


def _condition_family_base_color(cond, fallback_idx: int = 0) -> str:
    """Return a single representative hex color from a condition's FAMILY
    colormap (mid-tone), so condition-level bars / labels read in the SAME
    color family as the per-image dots and distribution lines (KD = blue,
    NT = red, Sec-Only = gray). This intentionally overrides the legacy
    _color_for_condition mapping (which used Okabe-Ito green/orange for
    NT/KD) so the whole deck is color-consistent under the family scheme.

    2026-06-03 Brian: for the MIAT-OE LOCKED conditions (Control / MIAT OE)
    the base color is pinned to the EXACT Okabe-Ito hex (orange / blue) so
    bars/dots/medians read on Brian's locked 2-condition palette — no green.
    All other conditions keep the colormap mid-tone (BIN1/H9 unaffected)."""
    import matplotlib.cm as _cm
    _exact = _LOCKED_EXACT_BASE_COLORS.get(_norm_cond_key(cond))
    if _exact is not None:
        return _exact
    try:
        r, g, b, _ = _get_cmap(_cmap_for_condition(cond, fallback_idx))(0.62)
        return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))
    except Exception:
        return _color_for_condition(cond, fallback_idx)


def _annotate_pairwise_brackets(ax, group_means: dict, conditions: list[str],
                                 *, x_centers: dict | None = None,
                                 pct: bool = False, fontsize: float = 7.5) -> None:
    """Draw stacked pairwise stat brackets (vs the first/reference condition)
    on ANY by-condition axis whose x positions are 1..N (boxplot) or supplied
    via ``x_centers``. Same Welch-t(primary) + MWU(secondary) + n_FoV readout
    and neutral wording as the SuperPlots, so every comparison panel reads
    identically. Reserves top headroom; for a fraction axis it keeps the 0–1
    frame and packs brackets in the headroom below 1.0."""
    try:
        recs = _superplot_stats(group_means, conditions)
        if not recs:
            return
        if x_centers is None:
            x_centers = {c: i + 1 for i, c in enumerate(conditions)}
        all_means = [m for ms in group_means.values() for m in ms]
        if not all_means:
            return
        base_top = max(all_means)
        n_cmp = len(recs)
        y0, y1 = ax.get_ylim()
        log_y = ax.get_yscale() == "log"
        if log_y:
            return  # brackets on a log axis read poorly; skip (rare here)
        if pct:
            head = max(0.04, (1.0 - base_top) / (n_cmp + 1))
            bh = 0.012
        else:
            yspan = (y1 - y0) or 1.0
            ax.set_ylim(y0, max(y1, base_top) + yspan * (0.14 + 0.13 * n_cmp))
            y0, y1 = ax.get_ylim(); yspan = (y1 - y0) or 1.0
            head = yspan * 0.12; bh = yspan * 0.02
        # 2026-06-06 Brian (consistency with _superplot_into_axes): above each
        # bracket render ONLY a compact significance STAR (primary test =
        # Welch-t on per-image means; '*' <0.05 / '**' <0.01 / '***' <0.001 /
        # 'ns' otherwise / 'n/a' when n<2). The verbose "Welch p=…; MWU p=…;
        # n_FoV=…" text is REMOVED from the per-bracket label and moved to ONE
        # bottom-of-figure footnote (drawn by the standalone render path / the
        # owning grid from the records stashed below). Bracket LINES and which
        # comparisons are drawn are UNCHANGED — only the LABEL text + the
        # LOCATION of the details change.
        _drawn = []
        for ci, rec in enumerate(recs):
            if rec["a"] not in x_centers or rec["b"] not in x_centers:
                continue
            xa = x_centers[rec["a"]]; xb = x_centers[rec["b"]]
            bar_y = base_top + head * (ci + 0.6)
            ax.plot([xa, xa, xb, xb], [bar_y, bar_y + bh, bar_y + bh, bar_y],
                    lw=1.2, color="#1f1f1f", clip_on=False)
            star = (_stars_for_p(rec["p_t"])
                    if np.isfinite(rec.get("p_t", float("nan"))) else "n/a")
            ax.text((xa + xb) / 2.0, bar_y + bh + (0.004 if pct else bh * 0.4),
                    star, ha="center", va="bottom",
                    fontsize=max(fontsize + 2.5, 11), fontweight="bold",
                    color="#1f1f1f", clip_on=False)
            _drawn.append(rec)
        # Stash the per-bracket records on the axis so the render path can
        # compose the single bottom footnote (test names, star legend, per-
        # condition n_FoV). Extend if a panel calls this more than once.
        try:
            _prev = list(getattr(ax, "_superplot_stats_recs", None) or [])
            ax._superplot_stats_recs = _prev + _drawn
        except Exception:
            pass
    except Exception:
        pass


def _per_image_means_by_condition(df: pd.DataFrame, value_col: str,
                                   conditions: list[str], *,
                                   only_positive: bool = False) -> dict:
    """Return {condition: [per-image mean of value_col, ...]} for the given
    conditions. Used to feed the shared pairwise-stat bracket renderer from
    the box/collapsed panels."""
    out: dict = {}
    if value_col not in df.columns or "condition" not in df.columns or "image" not in df.columns:
        return out
    d = df.copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    if only_positive:
        d = d[d[value_col].notna() & (d[value_col] > 0)]
    else:
        d = d[d[value_col].notna()]
    for c in conditions:
        sub = d[d["condition"] == c]
        if sub.empty:
            out[c] = []
            continue
        out[c] = sub.groupby("image")[value_col].mean().dropna().tolist()
    return out


def _box_strip_with_image_means(ax, nuc: pd.DataFrame, value_col: str,
                                 only_expressing: bool = False,
                                 condition_order: list[str] | None = None) -> bool:
    """Reusable: box-plot of `value_col` by condition, with two scatter
    overlays — small dots for individual cells (per-cell spread) AND big
    diamonds for per-image means (biological-replicate spread). Returns
    True if anything was drawn."""
    if value_col not in nuc.columns or "condition" not in nuc.columns:
        return False
    df = nuc.copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    if only_expressing and "rna_spot_count" in df.columns:
        df = df[(pd.to_numeric(df["rna_spot_count"], errors="coerce") > 0) & df[value_col].notna() & (df[value_col] > 0)]
    else:
        df = df[df[value_col].notna()]
    if df.empty:
        return False

    conds_in_data = df["condition"].dropna().unique().tolist()
    conditions = order_conditions(conds_in_data, condition_order or [])
    if not conditions:
        return False
    data = [df[df["condition"] == c][value_col].values for c in conditions]

    # 2026-05-26 (publication polish): use the condition FAMILY base color
    # (KD=Blues, NT=Reds, Sec-Only=Greys) so box plots read in the SAME color
    # scheme as the SuperPlots / distributions / ranked bars everywhere in the
    # deck. Previously this used _color_for_condition (Okabe-Ito green/orange),
    # which made the same condition a different color than its SuperPlot.
    cond_colors = [_condition_family_base_color(c, i) for i, c in enumerate(conditions)]

    bp = ax.boxplot(data, tick_labels=[_display_condition(c) for c in conditions],
                    showfliers=False, patch_artist=True,
                    medianprops=dict(color="black", linewidth=1.5))
    for patch, color in zip(bp["boxes"], cond_colors):
        patch.set_facecolor(color); patch.set_alpha(0.55)

    rng = np.random.RandomState(0)
    for i, (cond, vals) in enumerate(zip(conditions, data), start=1):
        color = cond_colors[i - 1]
        # Per-cell strip — small, transparent
        if len(vals) > 0:
            jitter = (rng.random(len(vals)) - 0.5) * 0.25
            ax.plot(np.full_like(vals, i, dtype=float) + jitter, vals,
                    "o", markersize=2, alpha=0.35, color=color, zorder=2)
        # Per-image MEANS — diamonds with edge, drawn on top of the box.
        # 2026-05-18 Brian: shrank markersize 8 -> 5 and edgewidth 1.0 -> 0.7
        # ("per-image mean thing is too big" on the active-TSS plot). The
        # diamond shape is preserved so the legend marker still reads as
        # "biological-replicate mean" vs the round per-cell dots.
        if "image" in df.columns:
            img_means = df[df["condition"] == cond].groupby("image")[value_col].mean().values
            if len(img_means) > 0:
                jitter_im = (rng.random(len(img_means)) - 0.5) * 0.18
                ax.plot(np.full_like(img_means, i, dtype=float) + jitter_im, img_means,
                        "D", markersize=5, color=color,
                        markeredgecolor="black", markeredgewidth=0.7,
                        zorder=4, label="per-image mean" if i == 1 else None)
    return True


def plot_box_spots_by_condition(ax, nuc: pd.DataFrame,
                                 condition_order: list[str] | None = None) -> None:
    """Box plot of spots-per-cell by condition. Small dots = individual
    cells (per-cell spread); large diamonds = per-image means (biological
    replicate spread). Pairs visual cell-level + image-level dispersion
    in one panel — the proper way to show per-image-as-biological-
    replicate alongside per-cell-as-technical-replicate."""
    if not _box_strip_with_image_means(ax, nuc, "rna_spot_count",
                                       condition_order=condition_order):
        ax.set_visible(False); return
    ax.set_ylabel("Spots per nucleus")
    ax.set_title(_wrap_title("Spots per nucleus — by condition\n(◇ = per-image mean, • = per nucleus)"))
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=8, loc="upper right")
    # Pairwise stats on per-image means (Welch + MWU), vs the reference column.
    _conds = order_conditions(
        nuc["condition"].dropna().unique().tolist() if "condition" in nuc.columns else [],
        condition_order or [])
    _annotate_pairwise_brackets(
        ax, _per_image_means_by_condition(nuc, "rna_spot_count", _conds), _conds)


def plot_box_total_intensity_by_condition(ax, nuc: pd.DataFrame,
                                           condition_order: list[str] | None = None) -> None:
    """Box plot of total RNA mass per cell grouped by condition. Auto-logs
    y when range is wide. Excludes zero-spot cells so the box reflects
    expression strength among expressers. Per-image means overlaid as
    diamonds."""
    col = _resolve_col(nuc, "rna_spot_total_peak_intensity", "rna_spot_total_intensity_fit")
    use_peak = col in nuc.columns
    if not use_peak or pd.to_numeric(nuc.get(col, pd.Series()), errors="coerce").fillna(0).sum() == 0:
        col = "rna_spot_total_intensity_bgc_blend"
        use_peak = False
    if col not in nuc.columns:
        ax.set_visible(False); return
    if not _box_strip_with_image_means(ax, nuc, col, only_expressing=True,
                                        condition_order=condition_order):
        ax.set_visible(False); return
    # Auto-log if dynamic range is wide
    expr = nuc[(pd.to_numeric(nuc.get("rna_spot_count", 0), errors="coerce") > 0)
               & pd.to_numeric(nuc[col], errors="coerce").notna()
               & (pd.to_numeric(nuc[col], errors="coerce") > 0)]
    if not expr.empty:
        flat = pd.to_numeric(expr[col], errors="coerce").dropna().values
        if len(flat) and (flat.max() / max(flat[flat > 0].min() if (flat > 0).any() else 1, 1)) > 100:
            ax.set_yscale("log")
    _kind = "peak" if use_peak else "BG-corrected"
    ax.set_ylabel("Summed RNA FISH intensity per nucleus (a.u.)\n"
                  f"(summed per-spot {_kind} intensity)")
    ax.set_title(_wrap_title(
        "Total RNA intensity per nucleus — by condition\n"
        "(◇ = per-image mean, • = per nucleus, expressing only)"))
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=8, loc="upper right")


def plot_spot_brightness_rank(ax, spots: pd.DataFrame) -> None:
    """Sorted spot brightness curve, per image. Sort all spots in an image
    by peak intensity (descending) and plot rank vs intensity. The
    far-left "shoulder" reveals bright outliers — the candidate set for
    transcription sites. A flat tail = uniform single-mol-like spots; a
    steep curve = a few very bright + many dim spots."""
    col = _resolve_col(spots, "peak_intensity", "integrated_intensity_fit", "spot_peak_intensity")
    if spots.empty or col not in spots.columns:
        ax.set_visible(False)
        return
    images = spots["image"].unique()
    labels = _build_image_labels(spots)
    family_map = _build_family_color_map(spots)
    plotted = 0
    for img_name in images:
        sub = spots[spots["image"] == img_name]
        color = _image_family_color(family_map, img_name)
        # Only model-fit, non-zero spots
        if "fit_ok" in sub.columns:
            sub = sub[sub["fit_ok"].astype(str) == "1"]
        vals = sub[col].dropna()
        vals = vals[vals > 0]
        if vals.empty: continue
        sv = np.sort(vals.values)[::-1]
        x = np.arange(1, len(sv) + 1)
        ax.plot(x, sv, alpha=0.7, color=color, label=f"{labels.get(img_name, '?')} (n={len(sv)})")
        plotted += 1
    if plotted == 0:
        ax.set_visible(False)
        return
    ax.set_xlabel("Spot rank (1 = brightest)")
    ax.set_ylabel(f"Peak intensity ({col})")
    ax.set_title(_wrap_title("Sorted spot-brightness curve (left shoulder = bright outliers)\n"
                             "(color = condition family; shade = image)"))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3, which="both")


def plot_local_snr_distribution(ax, spots: pd.DataFrame) -> None:
    """Per-spot local SNR distribution, per image. Quality metric: spots
    with SNR > 5 are robustly above background; SNR < 2 are noise-tier.
    A real-signal image should have most spots > 5; sec-only typically
    sits low or disappears entirely after thresholding."""
    if spots.empty or "local_snr" not in spots.columns:
        ax.set_visible(False)
        return
    vals_all = spots["local_snr"].dropna()
    if vals_all.empty:
        ax.set_visible(False)
        return
    images = spots["image"].unique()
    labels = _build_image_labels(spots)
    vmax = float(vals_all.quantile(0.99)) * 1.1
    bins = np.linspace(0, max(vmax, 1), 40)
    family_map = _build_family_color_map(spots)
    for img_name in images:
        sub = spots[spots["image"] == img_name]
        color = _image_family_color(family_map, img_name)
        v = sub["local_snr"].dropna()
        if v.empty: continue
        ax.hist(v, bins=bins, alpha=0.55, color=color,
                label=f"{labels[img_name]} (n={len(v)})")
    ax.axvline(5, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Local SNR (peak / std of bg ring)")
    ax.set_ylabel("Spot count")
    ax.set_title(_wrap_title("Per-spot local SNR (dotted line = SNR=5 quality threshold)\n"
                             "(color = condition family; shade = image)"))
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)


def plot_spots_vs_nucleus_area(ax, nuc: pd.DataFrame) -> None:
    """Scatter of spots-per-cell vs nucleus area. Tests whether bigger cells
    have more transcripts (volume-scaled expression). A flat cloud = no
    relationship; positive trend = expression scales with cell size."""
    if "rna_spot_count" not in nuc.columns or "nucleus_area_px" not in nuc.columns:
        ax.set_visible(False)
        return
    conds_in = nuc["condition"].dropna().unique().tolist() if "condition" in nuc.columns else []
    conditions = order_conditions(conds_in, [])
    vox = _voxel_xy_um_from(nuc)
    area_um2 = pd.to_numeric(nuc["nucleus_area_px"], errors="coerce") * (vox ** 2)
    nuc = nuc.assign(_nuc_area_um2=area_um2)
    for i, cond in enumerate(conditions):
        sub = nuc[nuc["condition"] == cond]
        if sub.empty: continue
        ax.scatter(sub["_nuc_area_um2"], sub["rna_spot_count"],
                   s=18, alpha=0.65, color=_condition_family_base_color(cond, i),
                   edgecolor="white", linewidth=0.4,
                   label=f"{_display_condition(cond)} (n={len(sub)})")
    if not conditions:
        ax.scatter(nuc["_nuc_area_um2"], nuc["rna_spot_count"], s=18, alpha=0.55)
    ax.set_xlabel("Nucleus area (µm²)")
    ax.set_ylabel("Spots per nucleus")
    ax.set_title(_wrap_title("Spots per nucleus vs nucleus area\n(positive trend = volume-scaled expression)"))
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)


def plot_whole_nucleus_intensity_by_condition(ax, nuc: pd.DataFrame,
                                                condition_order: list[str] | None = None) -> None:
    """Whole-nucleus integrated RNA intensity per cell, by condition.
    Independent of spot detection — uses rna_mean_in_nucleus × area, so it
    works even on images where zero spots were called. Per-image means
    overlaid as diamonds.
    """
    if "rna_mean_in_nucleus" not in nuc.columns or "nucleus_area_px" not in nuc.columns:
        ax.set_visible(False); return
    df = nuc.copy()
    df["_whole_nuc_total"] = pd.to_numeric(df["rna_mean_in_nucleus"], errors="coerce") \
                             * pd.to_numeric(df["nucleus_area_px"], errors="coerce")
    if not _box_strip_with_image_means(ax, df, "_whole_nuc_total", only_expressing=False,
                                        condition_order=condition_order):
        ax.set_visible(False); return
    valid = df[df["_whole_nuc_total"].notna() & (df["_whole_nuc_total"] > 0)]
    if not valid.empty:
        flat = valid["_whole_nuc_total"].values
        if (flat.max() / max(flat[flat > 0].min() if (flat > 0).any() else 1, 1)) > 100:
            ax.set_yscale("log")
    ax.set_ylabel("Integrated RNA intensity in nucleus (a.u.)\n(mean nuclear RNA pixel × nucleus area)")
    ax.set_title(_wrap_title(
        "Whole-nucleus integrated RNA intensity — by condition\n"
        "(spot-detection-independent; ◇ = per-image mean, • = per nucleus)"))
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=8, loc="upper right")
    _conds = order_conditions(
        df["condition"].dropna().unique().tolist() if "condition" in df.columns else [],
        condition_order or [])
    if ax.get_yscale() != "log":
        _annotate_pairwise_brackets(
            ax, _per_image_means_by_condition(df, "_whole_nuc_total", _conds,
                                              only_positive=True), _conds)


def plot_spot_volume_distribution(ax, spots: pd.DataFrame) -> None:
    """Per-spot volume (or 2D footprint area) distribution. Real puncta cluster
    near the diffraction-limited size; a long tail flags clumps/aggregates."""
    col = "spot_volume_um3"
    if spots.empty or col not in spots.columns:
        ax.set_visible(False)
        return
    if "fit_ok" in spots.columns:
        sub_all = spots[spots["fit_ok"].astype(str) == "1"]
    else:
        sub_all = spots
    vals_all = sub_all[col].dropna()
    vals_all = vals_all[vals_all > 0]
    if vals_all.empty:
        ax.set_visible(False)
        return
    images = sub_all["image"].unique()
    labels = _build_image_labels(sub_all)
    vmax = float(vals_all.quantile(0.98)) * 1.05
    bins = np.linspace(0, max(vmax, 1e-3), 40)
    family_map = _build_family_color_map(sub_all)
    for img_name in images:
        sub = sub_all[sub_all["image"] == img_name]
        color = _image_family_color(family_map, img_name)
        v = sub[col].dropna()
        v = v[v > 0]
        if v.empty: continue
        ax.hist(v, bins=bins, alpha=0.55, color=color,
                label=f"{labels[img_name]} (n={len(v)})")
    ax.set_xlabel(f"Spot volume ({col})")
    ax.set_ylabel("Spot count")
    ax.set_title(_wrap_title("Per-spot volume / footprint distribution (fit-ok spots only)\n"
                             "(color = condition family; shade = image)"))
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)


# =====================================================================
# SuperPlots (Lord, Velle, Mullins & Fritz-Laylin 2020, JCB;
#   doi:10.1083/jcb.202001064) + per-condition ranked / waterfall views
# =====================================================================
# 2026-05-25 Brian: bring the cardiomyocyte deck's SuperPlot style into the
# rna_only single-condition deck. Each SuperPlot is one cluster per
# condition: small dots = per-unit observations (per nucleus, or per spot)
# colored by IMAGE using the same condition-family hue gradient as the
# distribution recolor; large outlined dots = per-IMAGE means (biological
# replicates); a faint violin backdrop shows the per-unit shape; and the
# inferential test is run on the PER-IMAGE MEANS (Welch t + Mann-Whitney)
# between condition pairs — never on pooled per-unit values (pseudo-
# replication). Annotation wording is observational/neutral (reports the
# test statistic + n_FoV; never "validates"/"confirms").
#
# Style is matched to build_cross_condition_deck_v3.py::_superplot_core so
# the single-condition deck and the cross-condition deck read identically.

def _superplot_stats(group_means: dict, conditions: list[str]) -> list[dict]:
    """Run Welch t-test + Mann-Whitney on PER-IMAGE MEANS for each adjacent
    condition pair (cond[i] vs cond[i+1]) AND for every pair vs the first
    condition. Returns a list of dicts describing each comparison. Pure
    description — no thresholds, no significance verdicts beyond the star
    glyph (which is a conventional readout, not an inferential claim)."""
    try:
        from scipy import stats as _stats
    except Exception:
        _stats = None
    out = []
    # Compare every condition against the FIRST condition (reference column),
    # which for this deck is the left-most after order_conditions (sec-only
    # is pinned last, so the reference is the first biological condition).
    if len(conditions) < 2:
        return out
    ref = conditions[0]
    ref_means = group_means.get(ref, [])
    for c in conditions[1:]:
        c_means = group_means.get(c, [])
        rec = {"a": ref, "b": c, "n_a": len(ref_means), "n_b": len(c_means),
               "p_t": float("nan"), "p_mw": float("nan")}
        if _stats is not None and len(ref_means) >= 2 and len(c_means) >= 2:
            import warnings as _w
            # The Sec-Only group is often all-zeros; Welch's t on a zero-
            # variance group triggers scipy's harmless "catastrophic
            # cancellation" RuntimeWarning. Suppress it so the run log stays
            # clean — the returned p-value is still valid.
            with _w.catch_warnings():
                _w.simplefilter("ignore", RuntimeWarning)
                try:
                    rec["p_t"] = float(_stats.ttest_ind(c_means, ref_means, equal_var=False)[1])
                except Exception:
                    pass
                try:
                    rec["p_mw"] = float(_stats.mannwhitneyu(c_means, ref_means, alternative="two-sided")[1])
                except Exception:
                    pass
        out.append(rec)
    return out


def _stars_for_p(p) -> str:
    """Conventional star glyph for a p-value (ns / * / ** / *** / ****)."""
    if p is None or not np.isfinite(p):
        return "n/a"
    if p < 1e-4: return "****"
    if p < 1e-3: return "***"
    if p < 1e-2: return "**"
    if p < 5e-2: return "*"
    return "ns"


def _fmt_p(p) -> str:
    """Format a p-value for an annotation: '<0.001' when tiny, else 3 sig
    decimals (e.g. 'p=0.004', 'p=0.062'). Returns 'p=n/a' for non-finite."""
    if p is None or not np.isfinite(p):
        return "p=n/a"
    if p < 1e-3:
        return "p<0.001"
    return f"p={p:.3f}"


def _stats_bracket_label(rec: dict) -> str:
    """Build the single-line bracket annotation shared by SuperPlots and the
    by-condition bar/box panels: stars + Welch-t p on per-image means
    (primary, Lord 2020) + Mann-Whitney U p (secondary) + n_FoV. Neutral
    wording — reports the test, never 'significant'/'confirms'."""
    pt = rec.get("p_t", float("nan"))
    pmw = rec.get("p_mw", float("nan"))
    star = _stars_for_p(pt) if np.isfinite(pt) else ""
    if np.isfinite(pt):
        body = f"Welch {_fmt_p(pt)}"
    else:
        body = "Welch p=n/a (n<2)"
    if np.isfinite(pmw):
        body += f"; MWU {_fmt_p(pmw)}"
    a = _display_condition(rec.get("a"))
    b = _display_condition(rec.get("b"))
    return f"{star} {a} vs {b}: {body}  (n_FoV={rec.get('n_a')}/{rec.get('n_b')})".strip()


def _superplot_into_axes(ax, df: pd.DataFrame, value_col: str, *,
                         ylabel: str, unit: str = "nucleus",
                         only_positive: bool = False,
                         cap_99: bool = False,
                         pct: bool = False,
                         log_y: bool = False,
                         condition_order: list[str] | None = None,
                         annotate_stats: bool = True,
                         color_mode: str = "by_image") -> bool:
    """Render a SuperPlot into ``ax``. ``unit`` is "nucleus" or "spot" and
    only changes legend/caption text. Returns True if anything was drawn.

    df must carry: condition, image, <value_col>. Each condition gets one
    x-cluster.

    ``color_mode`` (2026-06-11 Brian, ADDITIVE — default preserves the locked
    H9/BIN1 look so those decks are byte-unchanged unless a caller opts in):
      * "by_image" (DEFAULT): the established convention — every image in a
        condition gets its own light→dark hue from the condition family
        colormap (``_build_family_color_map``); per-nucleus dots + the
        per-image-MEAN markers share that per-image hue.
      * "by_condition": each CONDITION gets ONE hue (its family base color via
        ``_condition_family_base_color``). The big per-image-MEAN markers
        render in that condition hue at PARTIAL TRANSPARENCY (alpha≈0.55) with
        a dark edge so they stay legible UNDERNEATH the per-nucleus cloud; the
        per-nucleus dots take the SAME condition hue but lighter/more
        transparent. Violins, the dots-over-means z-order, the significance
        stars and the single bottom footnote are all UNCHANGED.
    """
    by_condition = (str(color_mode).lower() == "by_condition")
    if value_col not in df.columns or "condition" not in df.columns or "image" not in df.columns:
        return False
    d = df.copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    if only_positive:
        d = d[d[value_col].notna() & (d[value_col] > 0)]
    else:
        d = d[d[value_col].notna()]
    if d.empty:
        return False

    conds_in = d["condition"].dropna().unique().tolist()
    conditions = order_conditions(conds_in, condition_order or [])
    if not conditions:
        return False

    family_map = _build_family_color_map(d, condition_order=condition_order)
    rng = np.random.RandomState(20260525)

    cap = None
    if cap_99:
        try:
            cap = float(np.nanpercentile(d[value_col].values, 99))
        except Exception:
            cap = None

    group_means: dict = {}
    violin_data = []
    violin_pos = []
    x_centers = {}
    for ci, cond in enumerate(conditions):
        xi = ci + 1
        x_centers[cond] = xi
        sub = d[d["condition"] == cond]
        imgs = sorted(str(i) for i in sub["image"].dropna().unique())
        # 2026-06-11 Brian: in by_condition mode every image in THIS condition
        # shares the condition's single family base hue (no per-image gradient).
        cond_hue = _condition_family_base_color(cond, ci) if by_condition else None
        img_means = []
        for img in imgs:
            vals = sub.loc[sub["image"].astype(str) == img, value_col].dropna().values
            if len(vals) == 0:
                continue
            vals_plot = vals if cap is None else vals[vals <= cap]
            color = cond_hue if by_condition else (_image_family_color(family_map, img) or "#888888")
            n = len(vals_plot)
            s_dot = 12 if n < 400 else (8 if n < 2000 else 4)
            a_dot = 0.20 if n < 400 else (0.12 if n < 2000 else 0.06)
            if by_condition:
                # Lighter / more transparent per-nucleus cloud so the (also
                # condition-hued) per-image mean markers read clearly underneath.
                a_dot = a_dot * 0.80
            jx = rng.uniform(-0.28, 0.28, size=n)
            # 2026-06-06 Brian (FIX2): per-nucleus dots sit ON TOP (zorder=4)
            # of the large per-image-mean markers (zorder=3, below) so every
            # small dot that makes up a mean stays visible — the big rim now
            # reads BEHIND the cloud, not over it.
            ax.scatter(np.full(n, xi) + jx, vals_plot, s=s_dot, alpha=a_dot,
                       color=color, edgecolor="none", zorder=4)
            img_means.append((img, float(np.mean(vals))))
        # large per-image-mean dots — BEHIND the per-nucleus scatter (zorder=3
        # < scatter zorder=4) but ABOVE the violin backdrop (zorder=1). Kept
        # large + dark-edged so the rim still reads from behind the dots.
        # 2026-06-11 Brian (by_condition): the major dots take the condition hue
        # at partial transparency (alpha≈0.55) but KEEP a solid dark edge so they
        # stay visible under the per-nucleus cloud.
        for img, m in img_means:
            color = cond_hue if by_condition else (_image_family_color(family_map, img) or "#888888")
            m_disp = m if (cap is None or m <= cap) else cap
            if by_condition:
                ax.scatter([xi], [m_disp], s=200, facecolor=color, alpha=0.55,
                           edgecolor="#1f1f1f", linewidth=1.4, zorder=3)
            else:
                ax.scatter([xi], [m_disp], s=200, color=color,
                           edgecolor="#1f1f1f", linewidth=1.4, zorder=3)
        group_means[cond] = [m for _, m in img_means]
        # violin backdrop
        col_vals = sub[value_col].dropna().values
        if cap is not None:
            col_vals = col_vals[col_vals <= cap]
        violin_data.append(col_vals if len(col_vals) > 1 else np.array([np.nan, np.nan]))
        violin_pos.append(xi)

    try:
        parts = ax.violinplot(violin_data, positions=violin_pos, widths=0.85,
                              showmeans=False, showmedians=False, showextrema=False)
        for pc, (ci, cond) in zip(parts["bodies"], enumerate(conditions)):
            # Backdrop tinted with the CONDITION FAMILY (same family as the
            # per-image dots) so condition color reads identically everywhere.
            try:
                fam = _get_cmap(_cmap_for_condition(cond, ci))(0.55)
            except Exception:
                fam = _color_for_condition(cond, 0)
            pc.set_facecolor(fam)
            pc.set_edgecolor("#202020"); pc.set_alpha(0.15)
            pc.set_linewidth(0.8); pc.set_zorder(1)
    except Exception:
        pass

    ax.set_xticks(list(x_centers.values()))
    ax.set_xticklabels([_display_condition(c) for c in x_centers.keys()])
    ax.set_ylabel(ylabel)
    if pct:
        try:
            import matplotlib.ticker as _mtick
            # Display as a percentage (xmax=1.0 -> a fraction of 1 reads as
            # 100%) but DO NOT stretch the axis to the full 0–100% range.
            # 2026-06-05 Brian: coloc / pairing-fraction metrics live in a
            # narrow band; pinning to [0,1] made every condition look
            # identical. Autoscale to the DATA spread (per-unit points + means
            # already plotted) with a small symmetric margin so between-
            # condition differences are visible. The stats-headroom block below
            # opens room ABOVE the data for the brackets, relative to the data
            # max (not relative to 1.0). The tick LABELS still cap at 100% via a
            # MaxNLocator clamped to [0,1] so the axis never shows a meaningless
            # 120–140%.
            ax.yaxis.set_major_formatter(_mtick.PercentFormatter(xmax=1.0, decimals=0))
            ax.relim(); ax.autoscale(enable=True, axis="y")
            d0, d1 = ax.get_ylim()
            if not (np.isfinite(d0) and np.isfinite(d1)) or d1 <= d0:
                d0, d1 = 0.0, 1.0
            span = d1 - d0
            if span < 1e-9:
                # Degenerate (all values ~equal): fall back to a small window
                # centered on the value so the axis isn't zero-height.
                pad = max(abs(d1) * 0.05, 0.02)
                d0, d1 = d0 - pad, d1 + pad
                span = d1 - d0
            margin = span * 0.08
            lo = max(0.0, d0 - margin)          # fractions can't go below 0
            hi = min(1.0, d1 + margin)           # ... or above 1 (=100%)
            if hi <= lo:
                lo, hi = 0.0, 1.0
            ax.set_ylim(lo, hi)
            # Ticks chosen from the visible window, clamped to [0,1] so labels
            # stay within 0–100% even after the bracket headroom expands ylim.
            ax.yaxis.set_major_locator(
                _mtick.MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
        except Exception:
            pass
    if log_y:
        try:
            ax.set_yscale("log")
        except Exception:
            pass
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)

    # Stats annotation on per-image means (Welch-t primary + MWU secondary),
    # vs the first/reference (= left-most biological) condition.
    #
    # 2026-05-29 Brian: the brackets now sit near the TOP of the axes, INSIDE
    # a clear headroom band carved ABOVE the data — never in the middle of /
    # on top of the per-image-mean dots. We (1) extend the y-limit well above
    # the data max (fraction metrics -> top ~1.18–1.25 so 100% sits well below
    # the edge; unbounded -> data_max * ~1.20, plus a little extra per added
    # comparison), then (2) stack the brackets downward FROM the top so the
    # band between the data and the brackets stays clean.
    if annotate_stats and not log_y:
        try:
            stats_recs = _superplot_stats(group_means, conditions)
            n_cmp = max(1, len(stats_recs))
            all_means = [m for ms in group_means.values() for m in ms]
            base_top = max(all_means) if all_means else ax.get_ylim()[1]
            if cap is not None:
                base_top = min(base_top, cap)
            y0, y1 = ax.get_ylim()
            if pct:
                # 2026-06-05 Brian: fraction axis is now DATA-scaled (not pinned
                # to 0–100%). Open the bracket headroom RELATIVE TO THE DATA
                # SPREAD above the data max, exactly like the unbounded branch —
                # NOT relative to a fixed 1.0. This keeps the headroom
                # proportional to the (narrow) data band so the brackets sit
                # just above the points instead of way up near 100%.
                import matplotlib.ticker as _mt2
                yspan0 = (y1 - y0) or 1.0
                pos_top = max(base_top, y0)
                # Seat the top bracket a clear band above the data max; add a
                # little more per extra comparison. Never below the current data
                # ylim. Tick LABELS are re-clamped to [0,1] below so the visible
                # axis never reads >100% even though ylim may pass 1.0 slightly.
                y_top = max(y1, pos_top + yspan0 * (0.30 + 0.13 * (n_cmp - 1)))
                if not np.isfinite(y_top) or y_top <= y0:
                    y_top = y1 + yspan0 * (0.30 + 0.13 * n_cmp)
                ax.set_ylim(y0, y_top)
                # Place ticks only across the DATA-visible band, clamped to
                # [0,1] so PercentFormatter never prints a meaningless >100%.
                # Expanding ylim for the brackets must NOT add a >100% tick, so
                # we compute the ticks over the clamped band and pin them with a
                # FixedLocator (a plain locator would re-tick the full ylim).
                _tick_hi = min(1.0, max(y0, y1))
                _tick_lo = max(0.0, y0)
                if _tick_hi <= _tick_lo:
                    _tick_lo, _tick_hi = 0.0, 1.0
                _cand = _mt2.MaxNLocator(
                    nbins=6, steps=[1, 2, 2.5, 5, 10]).tick_values(_tick_lo, _tick_hi)
                _ticks = [t for t in _cand if _tick_lo - 1e-9 <= t <= _tick_hi + 1e-9]
                if not _ticks:
                    _ticks = list(np.linspace(_tick_lo, _tick_hi, 5))
                ax.yaxis.set_major_locator(_mt2.FixedLocator(_ticks))
                yspan = (y_top - y0) or 1.0
                bh = yspan * 0.02
                top_band = y_top - yspan * 0.10   # top bracket well below edge
                row_step = yspan * 0.085
            else:
                yspan0 = (y1 - y0) or 1.0
                # Top = data_max * ~1.28 (a clear band above the data), plus a
                # little more headroom per extra bracket so a 3+ condition deck
                # still keeps every bracket inside the axes. The bracket itself
                # seats ~10% below the top edge so its label never crowds the
                # axes top / the grey subtitle outside it.
                pos_top = max(base_top, 0.0)
                y_top = max(y1, pos_top * 1.28 + yspan0 * 0.13 * (n_cmp - 1))
                if not np.isfinite(y_top) or y_top <= y0:
                    y_top = y1 + yspan0 * (0.28 + 0.13 * n_cmp)
                ax.set_ylim(y0, y_top)
                yspan = (y_top - y0) or 1.0
                bh = yspan * 0.02
                top_band = y_top - yspan * 0.10   # top bracket well below edge
                row_step = yspan * 0.085
            # 2026-06-06 Brian (FIX1): above each bracket render ONLY a compact
            # significance STAR (based on the PRIMARY test = Welch-t on per-image
            # means; '*' <0.05 / '**' <0.01 / '***' <0.001 / 'ns' otherwise).
            # The verbose "Welch p=…; MWU p=…; n_FoV=…" text is REMOVED from the
            # per-bracket label — with 5–6 conditions it was unreadable clutter —
            # and moved to ONE footnote at the bottom of the figure (drawn by the
            # standalone render path from the records stashed below). The bracket
            # LINES and which comparisons are drawn are UNCHANGED; only the LABEL
            # text and the LOCATION of the details change.
            # Stack brackets DOWNWARD from the top of the axes.
            for ci, rec in enumerate(stats_recs):
                xa = x_centers[rec["a"]]; xb = x_centers[rec["b"]]
                bar_y = top_band - row_step * ci
                ax.plot([xa, xa, xb, xb], [bar_y, bar_y + bh, bar_y + bh, bar_y],
                        lw=1.2, color="#1f1f1f", clip_on=True)
                # Star marker only (primary = Welch-t). 'ns' when not significant,
                # 'n/a' when the test couldn't run (n<2) — never crashes.
                star = (_stars_for_p(rec["p_t"])
                        if np.isfinite(rec.get("p_t", float("nan"))) else "n/a")
                ax.text((xa + xb) / 2.0, bar_y + bh + (0.004 if pct else bh * 0.4),
                        star, ha="center", va="bottom", fontsize=11,
                        fontweight="bold", color="#1f1f1f", clip_on=False)
            # Stash the full per-bracket stat records on the axis so the render
            # path can compose the single bottom footnote (test names, star
            # legend, per-condition n_FoV). Detail lives at the bottom only.
            try:
                ax._superplot_stats_recs = stats_recs
            except Exception:
                pass
        except Exception:
            pass

    # Legend: per-unit dot + per-image-mean dot convention.
    # 2026-05-29 Brian (FIX3 — DETERMINISTIC): do NOT call ax.legend() here.
    # An in-axes legend (even anchored below via bbox_to_anchor) is repositioned
    # by tight_layout and could drift back over the data. Instead STASH the
    # handles/labels on the axis; the standalone render path draws them via a
    # FIXED-position fig.legend() in the reserved bottom band (bottom=0.20),
    # entirely OUTSIDE the axes. This removes the legend from the tight_layout /
    # bbox negotiation entirely, making the placement deterministic.
    try:
        from matplotlib.lines import Line2D
        unit_txt = "per nucleus" if unit == "nucleus" else "per spot"
        _handles = [
            Line2D([], [], marker="o", linestyle="", markersize=7,
                   markerfacecolor="#888", markeredgecolor="none", alpha=0.5,
                   label=f"{unit_txt} (colored by image)"),
            Line2D([], [], marker="o", linestyle="", markersize=12,
                   markerfacecolor="#888", markeredgecolor="#1f1f1f", markeredgewidth=1.4,
                   label="per-image mean"),
        ]
        ax._superplot_legend_handles = _handles
        ax._superplot_legend_labels = [h.get_label() for h in _handles]
    except Exception:
        pass

    # 2026-05-29 Brian: stash the filter/criteria subtitle + observed n_FoV on
    # the axis so the render path can draw it just UNDER the title (small,
    # clearly separated). Substance unchanged — pure annotation.
    try:
        n_by_cond = {c: len(group_means.get(c, [])) for c in conditions}
        ax._superplot_n_by_cond = n_by_cond
        ax._superplot_unit = unit
    except Exception:
        pass
    return True


def _superplot_filter_subtitle(ax) -> str:
    """Build the small filter/criteria subtitle for a SuperPlot axis. Prefers
    the per-run string set by main() (_SUPERPLOT_FILTER_SUBTITLE, which carries
    the spot floor pulled from run_config / the run-dir name); always appends
    the observed per-image-mean unit + n_FoV/condition read off the axis. Falls
    back to a sensible generic line when no per-run floor string is available.
    Neutral wording — reports the cutoffs + n, no 'significant'/'confirms'."""
    base = _SUPERPLOT_FILTER_SUBTITLE
    unit = getattr(ax, "_superplot_unit", "nucleus")
    unit_txt = "per-nucleus" if unit == "nucleus" else "per-spot"
    n_by = getattr(ax, "_superplot_n_by_cond", None)
    if n_by:
        n_txt = ", ".join(f"{_display_condition(c)} n={n}" for c, n in n_by.items())
        tail = f"{unit_txt} obs.; per-image means as replicates ({n_txt} FoV)"
    else:
        tail = f"{unit_txt} obs.; per-image means as biological replicates"
    if base:
        return f"{base}; {tail}"
    return tail


def _superplot_stats_footnote(ax) -> str:
    """Build the SINGLE bottom-of-figure stats footnote for a SuperPlot axis
    (2026-06-06 Brian, FIX1). The per-bracket labels now carry ONLY a star;
    the verbose detail moves here so it can be cropped later. Reports: the
    test(s) used + which one the star is based on, the star legend, and the
    per-comparison Welch/MWU p-values with n_FoV. Returns '' when no stats
    were computed (e.g. <2 conditions or n too small) so the caller can skip
    drawing it — no crash, graceful degrade.

    2026-06-06 (e): a panel may stash an extra one-line explanatory footnote on
    the axis via ``ax._superplot_extra_footnote`` (e.g. panel 76's z / empirical-p
    definition). It is appended here AND surfaces even on the <2-condition / no-
    stats early-return path, so the definition is never dropped."""
    extra = getattr(ax, "_superplot_extra_footnote", None)
    extra = str(extra).strip() if extra else ""
    recs = getattr(ax, "_superplot_stats_recs", None)
    if not recs:
        # No pairwise stats (e.g. a single condition) — still surface the extra
        # explanatory footnote when one was set; otherwise nothing to draw.
        return extra
    # Star legend + which test drives the star (primary = Welch-t on per-image
    # means; MWU reported alongside as a secondary check).
    legend = ("Significance star on the bracket = Welch's t-test on per-image "
              "(FoV) means (primary); Mann–Whitney U (MWU) reported as a "
              "secondary check. *** p<0.001, ** p<0.01, * p<0.05, ns = not "
              "significant. Comparisons vs the reference (left-most) condition.")
    parts = []
    for rec in recs:
        a = _display_condition(rec.get("a"))
        b = _display_condition(rec.get("b"))
        pt = rec.get("p_t", float("nan"))
        pmw = rec.get("p_mw", float("nan"))
        if np.isfinite(pt):
            body = f"Welch {_fmt_p(pt)}"
            star = _stars_for_p(pt)
        else:
            body = "Welch p=n/a (n<2)"
            star = "n/a"
        if np.isfinite(pmw):
            body += f", MWU {_fmt_p(pmw)}"
        parts.append(f"{a} vs {b}: {star}  ({body}; n_FoV={rec.get('n_a')}/"
                     f"{rec.get('n_b')})")
    base = legend + "   " + "  |  ".join(parts)
    if extra:
        base = base + "   " + extra
    return base


def _ranked_per_image_bars(ax, df: pd.DataFrame, value_col: str, *,
                           ylabel: str, unit_is_spot: bool = False,
                           agg: str = "mean", only_positive: bool = False,
                           condition_order: list[str] | None = None) -> bool:
    """Per-image ranked ("waterfall") bar chart: one bar per image, height =
    per-image aggregate of ``value_col`` (mean by default), bars colored by
    condition family hue, sorted DESCENDING within each condition block.
    Condition blocks are laid out left-to-right in order_conditions order
    with a small gap between blocks. Returns True if drawn."""
    if value_col not in df.columns or "image" not in df.columns or "condition" not in df.columns:
        return False
    d = df.copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    if only_positive:
        d = d[d[value_col].notna() & (d[value_col] > 0)]
    else:
        d = d[d[value_col].notna()]
    if d.empty:
        return False
    conds_in = d["condition"].dropna().unique().tolist()
    conditions = order_conditions(conds_in, condition_order or [])
    family_map = _build_family_color_map(d, condition_order=condition_order)
    labels_map = _build_image_labels(d)

    xs, heights, colors, xticklabels = [], [], [], []
    block_centers = {}
    x = 0
    for cond in conditions:
        sub = d[d["condition"] == cond]
        g = sub.groupby("image")[value_col]
        per_img = (g.mean() if agg == "mean" else g.median()).sort_values(ascending=False)
        if per_img.empty:
            continue
        start_x = x
        for img, h in per_img.items():
            xs.append(x); heights.append(float(h))
            colors.append(_image_family_color(family_map, img) or "#888888")
            xticklabels.append(labels_map.get(img, short_label(str(img))))
            x += 1
        block_centers[cond] = (start_x + x - 1) / 2.0
        x += 1  # gap between condition blocks
    if not xs:
        return False
    ax.bar(xs, heights, color=colors, edgecolor="black", linewidth=0.5, width=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels(xticklabels, rotation=45, ha="right", fontsize=7)
    # condition block labels BELOW the rotated per-image tick labels so they
    # don't overlap. Drawn in the condition FAMILY base color for consistency.
    for ci, (cond, cx) in enumerate(block_centers.items()):
        ax.text(cx, -0.34, _display_condition(cond), transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=11, fontweight="bold",
                color=_condition_family_base_color(cond, ci))
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    return True


def _collapse_to_condition(df: pd.DataFrame, value_col: str, *,
                           only_positive: bool = False) -> tuple[list, list, list]:
    """Collapse per-unit values to a per-image-mean list per condition, then
    return (conditions, per_image_mean_lists, condition_grand_means). Used by
    the per-condition COLLAPSED bar view (mean ± SD of per-image means)."""
    out_conds, out_lists, out_grand = [], [], []
    if value_col not in df.columns or "condition" not in df.columns or "image" not in df.columns:
        return out_conds, out_lists, out_grand
    d = df.copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    if only_positive:
        d = d[d[value_col].notna() & (d[value_col] > 0)]
    else:
        d = d[d[value_col].notna()]
    for cond in d["condition"].dropna().unique().tolist():
        sub = d[d["condition"] == cond]
        im = sub.groupby("image")[value_col].mean().dropna().tolist()
        if im:
            out_conds.append(cond); out_lists.append(im)
            out_grand.append(float(np.mean(im)))
    return out_conds, out_lists, out_grand


# ---- SuperPlot wrapper plot functions (layout-signature compatible) ----
# Each takes (ax, nuc_or_spots, condition_order=...) so it slots straight
# into the build_layout tuple list. They guard on column presence and call
# ax.set_visible(False) when there is nothing to draw.

def superplot_spots_per_cell(ax, nuc, condition_order=None):
    """SuperPlot — RNA spots per nucleus, by condition (per-image means)."""
    if not _superplot_into_axes(ax, nuc, "rna_spot_count", ylabel="Spots per nucleus",
                                unit="nucleus", condition_order=condition_order):
        ax.set_visible(False); return
    ax.set_title(_wrap_title("SuperPlot: spots per nucleus — by condition\n"
                             "(small = per nucleus, large = per-image mean; Welch + MWU on image means)"))


def superplot_total_rna_per_cell(ax, nuc, condition_order=None):
    """SuperPlot — total RNA mass per cell, expressing cells only."""
    col = _resolve_col(nuc, "rna_spot_total_peak_intensity", "rna_spot_total_intensity_fit")
    use_peak = col in nuc.columns
    if not use_peak or pd.to_numeric(nuc.get(col, pd.Series(dtype=float)),
                                     errors="coerce").fillna(0).sum() == 0:
        col = "rna_spot_total_intensity_bgc_blend"
        use_peak = False
    _kind = "peak" if use_peak else "BG-corrected"
    if not _superplot_into_axes(
            ax, nuc, col,
            ylabel=f"Summed RNA FISH intensity / nucleus (a.u.)\n(summed per-spot {_kind} intensity)",
            unit="nucleus", only_positive=True, cap_99=True,
            condition_order=condition_order):
        ax.set_visible(False); return
    ax.set_title(_wrap_title(
        "SuperPlot: total RNA intensity per nucleus — by condition\n"
        "(expressing nuclei; nucleus area matched; 99th-pct capped; Welch+MWU on image means)"))


def superplot_nc_ratio_per_cell(ax, nuc, condition_order=None):
    """SuperPlot — per-cell nuclear:cytoplasmic RNA spot fraction."""
    col = "nuclear_spot_fraction"
    if col not in nuc.columns:
        ax.set_visible(False); return
    if not _superplot_into_axes(ax, nuc, col, ylabel="Nuclear fraction of RNA spots",
                                unit="nucleus", pct=True, condition_order=condition_order):
        ax.set_visible(False); return
    ax.set_title(_wrap_title("SuperPlot: per-cell nuclear spot fraction — by condition\n"
                             "(small = per nucleus, large = per-image mean; Welch + MWU on image means)"))


def superplot_per_cell_expression(ax, nuc, condition_order=None):
    """SuperPlot — per-cell mean spot intensity (expressing cells only)."""
    col = "rna_spot_mean_intensity_bgc_blend"
    if not _superplot_into_axes(ax, nuc, col, ylabel="Mean spot intensity / cell\n(BG-corrected)",
                                unit="nucleus", only_positive=True, cap_99=True,
                                condition_order=condition_order):
        ax.set_visible(False); return
    ax.set_title(_wrap_title("SuperPlot: per-cell expression intensity — by condition\n"
                             "(expressing cells; 99th-pct capped; Welch + MWU on image means)"))


def superplot_spot_peak_intensity(ax, spots, condition_order=None):
    """Per-SPOT SuperPlot — spot peak intensity by condition."""
    if not _superplot_into_axes(ax, spots, "spot_peak_intensity",
                                ylabel="Spot peak intensity (raw)", unit="spot",
                                cap_99=True, condition_order=condition_order):
        ax.set_visible(False); return
    ax.set_title(_wrap_title("SuperPlot: spot peak intensity — by condition\n"
                             "(small = per spot, large = per-image mean; 99th-pct capped)"))


def superplot_spot_size(ax, spots, condition_order=None):
    """Per-SPOT SuperPlot — spot diameter (µm) by condition."""
    if not _superplot_into_axes(ax, spots, "spot_diameter_um",
                                ylabel="Spot diameter (µm)", unit="spot",
                                only_positive=True, cap_99=True, condition_order=condition_order):
        ax.set_visible(False); return
    ax.set_title(_wrap_title("SuperPlot: spot size (diameter) — by condition\n"
                             "(small = per spot, large = per-image mean; 99th-pct capped)"))


def superplot_spot_volume(ax, spots, condition_order=None):
    """Per-SPOT SuperPlot — spot volume (µm^3) by condition."""
    sub = spots
    if "fit_ok" in spots.columns:
        sub = spots[spots["fit_ok"].astype(str) == "1"]
    if not _superplot_into_axes(ax, sub, "spot_volume_um3",
                                ylabel="Spot volume (µm³)", unit="spot",
                                only_positive=True, cap_99=True, condition_order=condition_order):
        ax.set_visible(False); return
    ax.set_title(_wrap_title("SuperPlot: spot volume — by condition\n"
                             "(fit-ok spots; large = per-image mean; 99th-pct capped)"))


def superplot_local_snr(ax, spots, condition_order=None):
    """Per-SPOT SuperPlot — local SNR by condition."""
    if not _superplot_into_axes(ax, spots, "local_snr",
                                ylabel="Local SNR (peak / std bg ring)", unit="spot",
                                cap_99=True, condition_order=condition_order):
        ax.set_visible(False); return
    ax.axhline(5, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_title(_wrap_title("SuperPlot: per-spot local SNR — by condition\n"
                             "(dotted = SNR 5; large = per-image mean; 99th-pct capped)"))


def superplot_dist_to_edge(ax, spots, condition_order=None):
    """Per-SPOT SuperPlot — nuclear spot distance-to-edge by condition."""
    nuclear = spots[spots["nucleus_id"] > 0] if "nucleus_id" in spots.columns else spots
    if not _superplot_into_axes(ax, nuclear, "spot_to_nuc_edge_um",
                                ylabel="Spot-to-nuclear-edge distance (µm)", unit="spot",
                                only_positive=False, cap_99=True, condition_order=condition_order):
        ax.set_visible(False); return
    ax.set_title(_wrap_title("SuperPlot: nuclear spot distance-to-edge — by condition\n"
                             "(nuclear spots; large = per-image mean; 99th-pct capped)"))


# ---- Ranked / waterfall + per-condition collapsed wrappers ----

def ranked_spots_per_cell(ax, nuc, condition_order=None):
    """Per-image ranked bars of mean spots/nucleus, grouped by condition."""
    if not _ranked_per_image_bars(ax, nuc, "rna_spot_count",
                                  ylabel="Mean spots / nucleus (per image)",
                                  condition_order=condition_order):
        ax.set_visible(False); return
    ax.set_title(_wrap_title("Ranked per-image mean spots / nucleus\n"
                             "(bars sorted within each condition; color = condition family, shade = image)"))


def ranked_total_rna_per_cell(ax, nuc, condition_order=None):
    """Per-image ranked bars of mean total RNA mass / cell, by condition."""
    col = _resolve_col(nuc, "rna_spot_total_peak_intensity", "rna_spot_total_intensity_fit")
    use_peak = col in nuc.columns
    if not use_peak or pd.to_numeric(nuc.get(col, pd.Series(dtype=float)),
                                     errors="coerce").fillna(0).sum() == 0:
        col = "rna_spot_total_intensity_bgc_blend"
        use_peak = False
    _kind = "peak" if use_peak else "BG-corrected"
    if not _ranked_per_image_bars(
            ax, nuc, col, only_positive=True,
            ylabel="Mean summed RNA FISH intensity / nucleus, per image (a.u.)\n"
                   f"(summed per-spot {_kind} intensity)",
            condition_order=condition_order):
        ax.set_visible(False); return
    ax.set_title(_wrap_title("Ranked per-image mean total RNA intensity per nucleus\n"
                             "(expressing nuclei; bars sorted within each condition)"))


def ranked_nuclear_fraction(ax, nuc, condition_order=None):
    """Per-image ranked bars of mean nuclear spot fraction, by condition."""
    col = "nuclear_spot_fraction"
    if col not in nuc.columns:
        ax.set_visible(False); return
    if not _ranked_per_image_bars(ax, nuc, col,
                                  ylabel="Mean nuclear spot fraction (per image)",
                                  condition_order=condition_order):
        ax.set_visible(False); return
    try:
        import matplotlib.ticker as _mtick
        ax.yaxis.set_major_formatter(_mtick.PercentFormatter(xmax=1.0, decimals=0))
    except Exception:
        pass
    ax.set_title(_wrap_title("Ranked per-image mean nuclear spot fraction\n"
                             "(bars sorted within each condition; color = condition family, shade = image)"))


def collapsed_condition_means(ax, nuc, condition_order=None):
    """Per-condition COLLAPSED view: one bar per condition = grand mean of
    per-image means for spots/nucleus, error bar = SD across image means,
    individual per-image means overplotted as dots. Reads the biological-
    replicate spread directly at the condition level."""
    conds, lists, grand = _collapse_to_condition(nuc, "rna_spot_count")
    if not conds:
        ax.set_visible(False); return
    order = order_conditions(conds, condition_order or [])
    idx = {c: i for i, c in enumerate(conds)}
    order = [c for c in order if c in idx]
    xs = np.arange(len(order))
    means = [grand[idx[c]] for c in order]
    sds = [float(np.std(lists[idx[c]], ddof=1)) if len(lists[idx[c]]) > 1 else 0.0 for c in order]
    bar_colors = [_condition_family_base_color(c, i) for i, c in enumerate(order)]
    ax.bar(xs, means, yerr=sds, color=bar_colors, edgecolor="black",
           linewidth=0.6, alpha=0.55, capsize=5, zorder=2)
    rng = np.random.RandomState(7)
    for i, c in enumerate(order):
        vals = lists[idx[c]]
        jx = (rng.random(len(vals)) - 0.5) * 0.3
        ax.scatter(np.full(len(vals), i) + jx, vals, s=55, color=bar_colors[i],
                   edgecolor="#1f1f1f", linewidth=1.0, zorder=4)
    ax.set_xticks(xs); ax.set_xticklabels([_display_condition(c) for c in order])
    ax.set_ylabel("Spots per nucleus\n(grand mean of per-image means)")
    ax.grid(axis="y", alpha=0.25, linestyle="--"); ax.set_axisbelow(True)
    ax.set_title(_wrap_title("Per-condition collapsed: spots per nucleus\n"
                             "(bar = mean of image means, error = SD, dots = per-image means)"))
    # Pairwise stats on the per-image means (vs the reference column).
    gm = {c: lists[idx[c]] for c in order}
    _annotate_pairwise_brackets(ax, gm, order,
                                x_centers={c: i for i, c in enumerate(order)})


# =====================================================================
# rna_only NUCLEUS-LEVEL + NUCLEOLUS figures (2026-05-25 Brian)
# =====================================================================
# Brian: "see the nucleus vs each other" + "love to see the nucleolus info".
# These read per-nucleus columns that the nucleolus-aware pipeline build adds
# to nuclei_metrics.csv (nucleus_area_px, dapi_mean, dapi_cv,
# heterochromatin_fraction, rna_nc_ratio, nuclear_spot_fraction,
# nucleolus_area_px, nucleolus_fraction_of_nucleus, chromatin_dapi_mean) and
# the per-spot in_nucleus / in_nucleolus flags. Every function guards on
# column presence and calls ax.set_visible(False) when nothing applies, so
# the rna_rna deck and older nucleolus-free runs degrade gracefully. All
# reuse the SuperPlot / collapsed / family-color helpers above so condition
# colors (KD=Blues, NT=Reds, Sec-Only=Greys) read identically everywhere.

def _voxel_xy_um_from(df: pd.DataFrame, default: float = 0.13) -> float:
    """Pull the per-run voxel xy size (µm/px) from a 'voxel_xy_um' column if
    present, else fall back to Brian's 0.13 µm default. Used to convert px
    areas to µm²."""
    if df is not None and "voxel_xy_um" in df.columns:
        v = pd.to_numeric(df["voxel_xy_um"], errors="coerce").dropna()
        if len(v) and float(v.iloc[0]) > 0:
            return float(v.iloc[0])
    return default


def _nucleus_metric_superplot(ax, nuc, value_col, *, ylabel, title,
                              pct=False, condition_order=None,
                              transform=None, only_positive=False,
                              cap_99=False, hline=None, hline_label=None,
                              color_mode="by_image"):
    """Thin wrapper around _superplot_into_axes for a per-nucleus metric.
    Optional ``transform`` is applied to a COPY of the value column before
    plotting (e.g. px → µm²). ``hline`` draws a dashed horizontal reference
    line at that y-value (e.g. 1.0 = "no enrichment"). ``color_mode`` is passed
    straight through (default "by_image" = unchanged). Returns nothing; hides
    the axis on no data."""
    if value_col not in nuc.columns or "condition" not in nuc.columns:
        ax.set_visible(False); return
    d = nuc.copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    if transform is not None:
        d[value_col] = transform(d[value_col])
    if not _superplot_into_axes(ax, d, value_col, ylabel=ylabel, unit="nucleus",
                                pct=pct, only_positive=only_positive,
                                cap_99=cap_99, condition_order=condition_order,
                                color_mode=color_mode):
        ax.set_visible(False); return
    if hline is not None:
        try:
            ax.axhline(float(hline), color="#555555", linestyle="--",
                       linewidth=1.1, zorder=1.5,
                       label=(hline_label or None))
        except Exception:
            pass
    ax.set_title(_wrap_title(title))


def nucleus_area_superplot(ax, nuc, condition_order=None):
    """SuperPlot — nucleus area (µm²) by condition."""
    vox = _voxel_xy_um_from(nuc)
    _nucleus_metric_superplot(
        ax, nuc, "nucleus_area_px",
        ylabel="Nucleus area (µm²)", only_positive=True, cap_99=True,
        transform=lambda s: s * (vox ** 2), condition_order=condition_order,
        title=("SuperPlot: nucleus area — by condition\n"
               f"(px² → µm² at {vox:.3f} µm/px; large = per-image mean; 99th-pct capped)"))


def nucleus_dapi_mean_superplot(ax, nuc, condition_order=None):
    """SuperPlot — whole-nucleus mean DAPI intensity by condition."""
    _nucleus_metric_superplot(
        ax, nuc, "dapi_mean",
        ylabel="Whole-nucleus mean DAPI (raw)", cap_99=True,
        condition_order=condition_order,
        title=("SuperPlot: whole-nucleus mean DAPI — by condition\n"
               "(small = per nucleus, large = per-image mean; 99th-pct capped)"))


def nucleus_dapi_cv_superplot(ax, nuc, condition_order=None):
    """SuperPlot — within-nucleus DAPI coefficient of variation by condition.
    Higher CV = more heterogeneous chromatin texture."""
    _nucleus_metric_superplot(
        ax, nuc, "dapi_cv",
        ylabel="DAPI CV (within-nucleus)", cap_99=True,
        condition_order=condition_order,
        title=("SuperPlot: within-nucleus DAPI CV — by condition\n"
               "(higher = more heterogeneous DAPI texture; large = per-image mean)"))


def nucleus_heterochromatin_superplot(ax, nuc, condition_order=None):
    """SuperPlot — heterochromatin fraction of nucleus by condition."""
    _nucleus_metric_superplot(
        ax, nuc, "heterochromatin_fraction",
        ylabel="Heterochromatin fraction", pct=True,
        condition_order=condition_order,
        title=("SuperPlot: heterochromatin fraction — by condition\n"
               "(fraction of nuclear area above DAPI heterochromatin threshold)"))


def nucleus_nc_ratio_superplot(ax, nuc, condition_order=None):
    """SuperPlot — nuclear:cytoplasmic RNA pixel-intensity ratio by condition."""
    _nucleus_metric_superplot(
        ax, nuc, "rna_nc_ratio",
        ylabel="RNA nuclear:cytoplasmic ratio", only_positive=True, cap_99=True,
        condition_order=condition_order,
        title=("SuperPlot: RNA N/C pixel-intensity ratio — by condition\n"
               "(per-nucleus mean RNA pixel intensity, nuclear ÷ cytoplasmic; 99th-pct capped)"))


def nucleus_spot_fraction_superplot(ax, nuc, condition_order=None):
    """SuperPlot — per-nucleus nuclear fraction of RNA spots by condition."""
    _nucleus_metric_superplot(
        ax, nuc, "nuclear_spot_fraction",
        ylabel="Nuclear fraction of RNA spots", pct=True,
        condition_order=condition_order,
        title=("SuperPlot: per-nucleus nuclear spot fraction — by condition\n"
               "(fraction of each nucleus's RNA spots inside the nuclear mask)"))


# ---- Colocalization SuperPlots (07_coloc/) -------------------------------
# 2026-05-29 Brian: per-nucleus RNA × partner-channel colocalization rendered
# in the SAME SuperPlot style as every other by-condition figure (small dots =
# per nucleus colored by source image; large outlined dots = per-image means;
# violin backdrop; Welch + MWU on per-image means; n_FoV annotation; condition
# palette reused exactly — no red+green). These read the active channel labels
# directly from _LABELS so titles/axes name the real biology ("BIN1 Introns ×
# XRN2") rather than "rna1 × protein".
#
# Mode-gated: only built when the run carries two spot/intensity channels with
# per-nucleus coloc columns (rna_protein, rna_rna). For rna_only there are no
# coloc columns and these are NOT registered (see build_layout); each function
# also self-skips (hides its axis) if its source column is absent, so a stray
# call degrades cleanly rather than crashing.
#
# Floor↔coloc decoupling (verified in fishsuite core/modes/rna_rna.py:774 +
# core/metrics.py:compute_coloc_metrics): the per-nucleus Manders/Pearson are
# computed on RAW nuclear pixel intensities masked by the pixel-coloc MAD
# threshold (rna_thr_value / rna2_thr_value). They do NOT use the display/
# analysis floor (output.manual_antibody_min / apply_pub_contrast_floor_to_
# analysis), which only feeds the separate nuclear_above_floor_intensity_*
# columns. Raising the XRN2 floor therefore does not change these values.


def _coloc_partner_label() -> str:
    """Display label for the partner (non-RNA1) coloc channel.

    rna_protein runs name it by the antibody (e.g. "XRN2"); rna_rna runs name
    it by rna2_label. Picks antibody_label when it is non-default, else
    rna2_label, else "Protein"."""
    ab = _LABELS.get("antibody_label", "Protein")
    if ab and ab not in ("Protein", "Protein2"):
        return ab
    r2 = _LABELS.get("rna2_label", "RNA2")
    if r2 and r2 != "RNA2":
        return r2
    return ab or "Protein"


def _coloc_pair_title() -> str:
    """'<rna_label> × <partner_label>' for coloc figure titles."""
    return f"{_LABELS.get('rna_label', 'RNA1')} × {_coloc_partner_label()}"


def has_coloc_columns(nuc) -> bool:
    """True when the per-nucleus table carries the RNA × partner coloc columns
    (the marker columns the coloc SuperPlots plot). Used by build_layout to
    decide whether to register the 07_coloc deck."""
    if nuc is None or not hasattr(nuc, "columns"):
        return False
    markers = ("manders_rna1_in_protein", "manders_protein_in_rna1",
               "coloc_pearson_r_rna_protein", "paired_fraction_rna1_at_0p3um",
               "protein_enrichment_at_rna1_spots", "rna2_enrichment_at_rna1_spots",
               "protein_local_mean_at_rna1_spots", "rna2_local_mean_at_rna1_spots",
               # 2026-06-05 Brian: proper random-position-null enrichment column
               # (rna_protein relabels rna2->protein). Registers the 07_coloc deck
               # even on a run whose ONLY coloc output is the null enrichment.
               "protein_enrichment_vs_null_at_rna1_spots",
               "rna2_enrichment_vs_null_at_rna1_spots")
    return any(m in nuc.columns for m in markers)


def coloc_spot_pairing_superplot(ax, nuc, condition_order=None, color_mode="by_image"):
    """SuperPlot — per-nucleus fraction of RNA1 spots with a partner-channel
    spot within the pair distance (0.3 µm)."""
    rna = _LABELS.get("rna_label", "RNA1")
    partner = _coloc_partner_label()
    _nucleus_metric_superplot(
        ax, nuc, "paired_fraction_rna1_at_0p3um",
        ylabel=f"Fraction of {rna} spots paired with {partner} (≤0.3 µm)",
        pct=True, condition_order=condition_order, color_mode=color_mode,
        title=(f"SuperPlot: spot pairing — {_coloc_pair_title()} — by condition\n"
               f"(per-nucleus fraction of {rna} spots with a {partner} spot within 0.3 µm; "
               "large = per-image mean)"))


def coloc_manders_m1_superplot(ax, nuc, condition_order=None, color_mode="by_image"):
    """SuperPlot — Manders M1: fraction of RNA1 intensity inside the partner-
    positive pixel mask (RNA-in-protein)."""
    rna = _LABELS.get("rna_label", "RNA1")
    partner = _coloc_partner_label()
    _nucleus_metric_superplot(
        ax, nuc, "manders_rna1_in_protein",
        ylabel=f"Manders M1: fraction of {rna} signal in {partner}-positive pixels",
        pct=True, condition_order=condition_order, color_mode=color_mode,
        title=(f"SuperPlot: Manders M1 (M1, {rna} in {partner}) — by condition\n"
               f"(per-nucleus fraction of {rna} pixel intensity inside the {partner} mask; "
               "large = per-image mean)"))


def coloc_manders_m2_superplot(ax, nuc, condition_order=None, color_mode="by_image"):
    """SuperPlot — Manders M2: fraction of partner intensity inside the RNA1-
    positive pixel mask (protein-in-RNA)."""
    rna = _LABELS.get("rna_label", "RNA1")
    partner = _coloc_partner_label()
    _nucleus_metric_superplot(
        ax, nuc, "manders_protein_in_rna1",
        ylabel=f"Manders M2: fraction of {partner} signal in {rna}-positive pixels",
        pct=True, condition_order=condition_order, color_mode=color_mode,
        title=(f"SuperPlot: Manders M2 (M2, {partner} in {rna}) — by condition\n"
               f"(per-nucleus fraction of {partner} pixel intensity inside the {rna} mask; "
               "large = per-image mean)"))


def coloc_pearson_superplot(ax, nuc, condition_order=None, color_mode="by_image"):
    """SuperPlot — per-nucleus Pearson r between RNA1 and partner pixel
    intensities (whole-nucleus, all pixels, no floor)."""
    rna = _LABELS.get("rna_label", "RNA1")
    partner = _coloc_partner_label()
    _nucleus_metric_superplot(
        ax, nuc, "coloc_pearson_r_rna_protein",
        ylabel=f"Pearson r ({rna} vs {partner} pixel intensity)",
        condition_order=condition_order, color_mode=color_mode,
        title=(f"SuperPlot: Pearson r — {_coloc_pair_title()} — by condition\n"
               f"(per-nucleus pixel-intensity correlation, whole nucleus; "
               "large = per-image mean)"))


def _first_present_col(nuc, candidates):
    """Return the first candidate column name present in ``nuc``, else None.
    Lets the coloc functions accept BOTH the rna_protein-relabeled name
    (e.g. ``protein_enrichment_at_rna1_spots``) and the rna_rna name
    (``rna2_enrichment_at_rna1_spots``)."""
    if nuc is None or not hasattr(nuc, "columns"):
        return None
    for c in candidates:
        if c in nuc.columns:
            return c
    return None


def coloc_partner_intensity_at_rna_spots_superplot(ax, nuc, condition_order=None, color_mode="by_image"):
    """SuperPlot — intensity-based, FLOOR-ROBUST coloc. Per-nucleus mean RAW
    partner-channel (e.g. XRN2) intensity sampled in a spot-radius disk at the
    nucleus's RNA1 (intron) spots."""
    rna = _LABELS.get("rna_label", "RNA1")
    partner = _coloc_partner_label()
    col = _first_present_col(
        nuc, ("protein_local_mean_at_rna1_spots", "rna2_local_mean_at_rna1_spots"))
    if col is None:
        ax.set_visible(False); return
    _nucleus_metric_superplot(
        ax, nuc, col,
        ylabel=f"{partner} intensity at {rna} spots (raw)",
        only_positive=False, cap_99=True, condition_order=condition_order,
        color_mode=color_mode,
        title=(f"SuperPlot: {partner} intensity at {rna} spots (raw) — by condition\n"
               f"(per-nucleus mean RAW {partner} intensity in a spot-radius disk at each "
               f"{rna} spot; floor-robust; 99th-pct capped; large = per-image mean)"))


def coloc_partner_enrichment_at_rna_spots_superplot(ax, nuc, condition_order=None, color_mode="by_image"):
    """SuperPlot — intensity-based, FLOOR-ROBUST coloc enrichment ratio.
    Per-nucleus (mean partner intensity at RNA1 spots) ÷ (mean partner
    intensity over the whole nucleus). >1 ⇒ RNA1 foci sit at partner-bright
    sites. Dashed line at 1.0 = no enrichment."""
    rna = _LABELS.get("rna_label", "RNA1")
    partner = _coloc_partner_label()
    col = _first_present_col(
        nuc, ("protein_enrichment_at_rna1_spots", "rna2_enrichment_at_rna1_spots"))
    if col is None:
        ax.set_visible(False); return
    _nucleus_metric_superplot(
        ax, nuc, col,
        ylabel=f"{partner} enrichment at {rna} foci (×nuclear mean)",
        only_positive=True, cap_99=True, condition_order=condition_order,
        color_mode=color_mode,
        hline=1.0, hline_label="no enrichment (×1.0)",
        title=(f"SuperPlot: {partner} enrichment at {rna} foci — by condition\n"
               f"(per-nucleus {partner}-at-{rna}-spots ÷ {partner} whole-nucleus mean; "
               ">1 ⇒ foci at bright sites; floor-robust; dashed = no enrichment)"))


def _pooled_null_stats_by_condition(summary, condition_order=None) -> dict:
    """Per-condition pooled random-position-null stats for the proper coloc
    enrichment SuperPlot annotation.

    Reads per_image_summary (one row per FoV) and, for each condition, takes
    the mean pooled z and the mean empirical p across that condition's images
    (the empirical p / z are already pooled-then-rolled per image in fishsuite;
    averaging across the condition's FoVs gives a single condition-level
    summary for the small text annotation). Accepts BOTH the rna_protein-
    relabeled column names (``protein_pooled_*``) and the rna_rna names
    (``rna2_pooled_*``). Returns {condition: {"z": float, "p": float,
    "n": int}}; empty dict when the pooled columns are absent."""
    out: dict = {}
    if summary is None or not hasattr(summary, "columns") or "condition" not in summary.columns:
        return out
    z_col = _first_present_col(
        summary, ("protein_pooled_null_z_at_rna1_spots",
                  "rna2_pooled_null_z_at_rna1_spots"))
    p_col = _first_present_col(
        summary, ("protein_pooled_null_p_empirical_at_rna1_spots",
                  "rna2_pooled_null_p_empirical_at_rna1_spots"))
    if z_col is None and p_col is None:
        return out
    d = summary.copy()
    if z_col is not None:
        d[z_col] = pd.to_numeric(d[z_col], errors="coerce")
    if p_col is not None:
        d[p_col] = pd.to_numeric(d[p_col], errors="coerce")
    for cond, sub in d.groupby("condition", dropna=False):
        if cond is None or (isinstance(cond, float) and cond != cond):
            continue
        z_vals = sub[z_col].dropna().values if z_col is not None else np.array([])
        p_vals = sub[p_col].dropna().values if p_col is not None else np.array([])
        if len(z_vals) == 0 and len(p_vals) == 0:
            continue
        out[str(cond)] = {
            "z": float(np.mean(z_vals)) if len(z_vals) else float("nan"),
            "p": float(np.mean(p_vals)) if len(p_vals) else float("nan"),
            "n": int(max(len(z_vals), len(p_vals))),
        }
    return out


def coloc_partner_enrichment_vs_null_at_rna_spots_superplot(
        ax, nuc, summary=None, condition_order=None, color_mode="by_image"):
    """SuperPlot — PROPER colocalization statistic. Per-nucleus partner-channel
    (e.g. QKI) intensity at the RNA1 (MIAT) spots ÷ a per-nucleus RANDOM-POSITION
    null (same nucleus, N random spot-radius disks, nucleolar voids excluded).
    1.0 = no enrichment over chance; values sit ~1.0–1.3.

    Unlike the whole-nucleus enrichment SuperPlot (which divides by the bulk
    nuclear mean), the denominator here is a within-nucleus spatial null, so it
    controls for nuclear partner gradients and is the publication coloc metric.

    Reuses the established SuperPlot path: per-nucleus points shaded by source
    image, per-image MEANS as the tested biological replicates (Welch + MWU on
    the means), autoscaled y (NOT pinned to 0–1), dashed reference line at
    y=1.0. Each condition is annotated with its pooled null z / empirical-p
    (from per_image_summary) when available. Self-skips (hides its axis) when
    the enrichment-vs-null column is absent (older runs / non-rna_protein)."""
    rna = _LABELS.get("rna_label", "RNA1")
    partner = _coloc_partner_label()
    col = _first_present_col(
        nuc, ("protein_enrichment_vs_null_at_rna1_spots",
              "rna2_enrichment_vs_null_at_rna1_spots"))
    if col is None:
        # Optional panel — older runs / non-rna_protein lack this column.
        try:
            print(f"[coloc] enrichment-vs-null column absent "
                  f"(protein_enrichment_vs_null_at_rna1_spots) — skipping "
                  f"'{partner} at {rna} foci (vs null)' SuperPlot.")
        except Exception:
            pass
        ax.set_visible(False)
        return
    # only_positive=True: a 0 enrichment ratio means the partner was absent at
    # both the spots and the null draws -> undefined, drop it (matches the
    # whole-nucleus enrichment panel). cap_99 tames the rare hot-pixel nucleus.
    _nucleus_metric_superplot(
        ax, nuc, col,
        ylabel=f"enrichment vs random (×)",
        only_positive=True, cap_99=True, condition_order=condition_order,
        color_mode=color_mode,
        hline=1.0, hline_label="no enrichment (×1.0)",
        title=("QKI intensity at MIAT foci (vs per-nucleus null)\n"
               f"({partner} at {rna} spots ÷ per-nucleus random-position null; "
               ">1 ⇒ enriched; large = per-image mean; 99th-pct capped)"))
    if not ax.get_visible():
        return

    # 2026-06-06 (e): stash the z / empirical-p DEFINITION so the render path's
    # single bottom footnote (_superplot_stats_footnote) spells out what the
    # per-condition "pooled null z=…  p=…" annotation means. Set on the axis so
    # it is appended whether or not pairwise brackets were drawn.
    ax._superplot_extra_footnote = (
        "z = number of SDs the observed QKI-at-MIAT intensity sits above the "
        "per-nucleus random-position null; empirical p = fraction of null "
        "draws ≥ observed.")

    # Annotate each condition with its pooled null z / empirical-p (small text
    # just under the per-condition x-tick), consistent with how the other
    # SuperPlots surface stats. Pulled from per_image_summary; silent when the
    # pooled columns are absent.
    try:
        stats_by_cond = _pooled_null_stats_by_condition(summary, condition_order)
        if stats_by_cond:
            d = nuc.copy()
            d[col] = pd.to_numeric(d[col], errors="coerce")
            d = d[d[col].notna() & (d[col] > 0)]
            conds_in = d["condition"].dropna().unique().tolist() if "condition" in d.columns else []
            conditions = order_conditions(conds_in, condition_order or [])
            y0, y1 = ax.get_ylim()
            y_txt = y0 + (y1 - y0) * 0.015
            for xi, cond in enumerate(conditions, start=1):
                rec = stats_by_cond.get(str(cond))
                if not rec:
                    continue
                bits = []
                if np.isfinite(rec.get("z", float("nan"))):
                    bits.append(f"z={rec['z']:.1f}")
                if np.isfinite(rec.get("p", float("nan"))):
                    bits.append(f"{_stars_for_p(rec['p'])} {_fmt_p(rec['p'])}")
                if not bits:
                    continue
                ax.text(xi, y_txt, "pooled null\n" + "  ".join(bits),
                        ha="center", va="bottom", fontsize=6.8,
                        color="#1f1f1f", zorder=7, clip_on=False)
    except Exception:
        pass


# ---- Extended colocalization figures (2026-06-06 Brian) ------------------
# 5 publication coloc panels queued for the MIAT × QKI deck. Aesthetic =
# rnaseq-figure-style (Okabe-Ito condition palette, 600 DPI PNG, filter +
# stats in a croppable bottom footnote, agnostic framing). Coloc LUT:
# MIAT/640 = yellow, QKI/561 = magenta. All land in figures/07_coloc/.
# Each self-skips gracefully (no crash, no empty PNG) when its source column
# / CSV is absent so they are harmless on runs without the coloc backfill.
_QKI_MAGENTA = "#C51B8A"   # QKI / 561 LUT (readable magenta on white)
_MIAT_YELLOW = "#E6A700"   # MIAT / 640 LUT (readable gold on white)


def _coloc_fig_dir(out_dir) -> Path:
    """Return (and create) <out_dir>/figures/07_coloc/."""
    d = Path(out_dir) / "figures" / "07_coloc"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _rna1_spots_with_partner(spots):
    """rna1-channel spots carrying finite spot_peak_intensity AND
    partner_local_mean_intensity; None when the partner column is absent/empty
    (the self-skip trigger for the dose-dependence panels)."""
    if spots is None or not hasattr(spots, "columns"):
        return None
    if ("partner_local_mean_intensity" not in spots.columns
            or "spot_peak_intensity" not in spots.columns):
        return None
    d = spots.copy()
    if "channel" in d.columns:
        d = d[d["channel"].astype(str) == "rna1"]
    d["spot_peak_intensity"] = pd.to_numeric(d["spot_peak_intensity"], errors="coerce")
    d["partner_local_mean_intensity"] = pd.to_numeric(
        d["partner_local_mean_intensity"], errors="coerce")
    d = d[d["spot_peak_intensity"].notna() & d["partner_local_mean_intensity"].notna()]
    return d if not d.empty else None


def _pearson_spearman(x, y):
    """(pearson_r, pearson_p, spearman_rho, spearman_p); scipy path + numpy
    fallback. NaNs when n<3 or a channel is constant."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        if len(x) >= 2 and np.std(x) and np.std(y):
            return float(np.corrcoef(x, y)[0, 1]), float("nan"), float("nan"), float("nan")
        return float("nan"), float("nan"), float("nan"), float("nan")
    try:
        from scipy import stats as _st
        pr, pp = _st.pearsonr(x, y)
        sr, sp = _st.spearmanr(x, y)
        return float(pr), float(pp), float(sr), float(sp)
    except Exception:
        return float(np.corrcoef(x, y)[0, 1]), float("nan"), float("nan"), float("nan")


def render_coloc_panel_standalone(out_dir, idx, slug, draw_fn, *,
                                  figsize=(7.2, 5.4)):
    """Render a single-axis coloc panel (an in-grid drawer with signature
    ``draw_fn(ax)``) to its own PNG ``figures/07_coloc/{idx:02d}_{slug}.png``.

    Honors self-skip: if the drawer hides the axis (missing column / no data),
    NOTHING is written and the figure is closed — no empty PNG, no crash.
    A croppable bottom footnote is drawn from ``ax._coloc_extra_footnote`` (or,
    for a SuperPlot axis, ``_superplot_stats_footnote``). Returns the Path on
    success else None."""
    out_dir = Path(out_dir)
    fig, ax = plt.subplots(figsize=figsize, dpi=600)
    try:
        draw_fn(ax)
        if not ax.get_visible():
            plt.close(fig); return None
        foot = getattr(ax, "_coloc_extra_footnote", "") or _superplot_stats_footnote(ax)
        if foot:
            import textwrap as _tw
            _w_in, _h_in = fig.get_size_inches()
            _txt = "\n".join(_tw.wrap(str(foot), width=int(max(70, 11.0 * _w_in))))
            _n = _txt.count("\n") + 1
            _foot_in = 0.135 * _n + 0.18
            _new_h = _h_in + _foot_in
            fig.set_size_inches(_w_in, _new_h)
            _b0 = fig.subplotpars.bottom
            fig.subplots_adjust(bottom=min(0.6, (_b0 * _h_in + _foot_in) / _new_h))
            fig.text(0.5, 0.06 / _new_h, _txt, ha="center", va="bottom",
                     fontsize=6.5, color="#555555", linespacing=1.25)
        _relabel_fig(fig)
        out_pn = _coloc_fig_dir(out_dir) / f"{idx:02d}_{slug}.png"
        fig.savefig(out_pn, bbox_inches="tight", dpi=600)
        plt.close(fig)
        return out_pn
    except Exception as e:
        try:
            print(f"[coloc] standalone panel {idx} ({slug}) FAILED "
                  f"{type(e).__name__}: {e}")
        except Exception:
            pass
        plt.close(fig)
        return None


def coloc_dose_dependence_miat_vs_qki(ax, spots, condition_order=None):
    """Panel 77 (in-grid) — DOSE-DEPENDENCE. One point per rna1 (MIAT) spot:
    x = MIAT spot peak intensity, y = local partner (QKI) mean intensity in a
    spot-radius disk at that spot. Tests whether brighter MIAT foci sit at
    brighter QKI. Colored by condition; Pearson + Spearman r/p go in the
    croppable footnote. Self-skips (hides axis) when partner_local_mean_intensity
    is absent."""
    rna = _LABELS.get("rna_label", "RNA1")
    partner = _coloc_partner_label()
    d = _rna1_spots_with_partner(spots)
    if d is None:
        try:
            print("[coloc] partner_local_mean_intensity absent — skipping "
                  "dose-dependence panel 77.")
        except Exception:
            pass
        ax.set_visible(False); return
    conds_in = (d["condition"].dropna().unique().tolist()
                if "condition" in d.columns else [])
    conditions = order_conditions(conds_in, condition_order or []) or ["__all__"]
    for ci, cond in enumerate(conditions):
        sub = d[d["condition"] == cond] if (cond != "__all__" and "condition" in d.columns) else d
        if sub.empty:
            continue
        ax.scatter(sub["spot_peak_intensity"], sub["partner_local_mean_intensity"],
                   s=8, alpha=0.35, edgecolor="none",
                   color=_color_for_condition(cond, ci),
                   label=_display_condition(cond) if cond != "__all__" else None)
    # FIX 3 (2026-06-06): per-condition stats; Spearman (rank-based) is the
    # HEADLINE because it is robust to the per-section absolute-intensity scaling
    # (laser re-tuned per section). Pearson kept as a secondary check.
    real_conds = [c for c in conditions if c != "__all__"]
    cond_lines, foot_bits = [], []
    for cond in (real_conds or ["__all__"]):
        sub = d[d["condition"] == cond] if (cond != "__all__" and "condition" in d.columns) else d
        if sub.empty:
            continue
        cpr, cpp, csr, csp = _pearson_spearman(
            sub["spot_peak_intensity"].values,
            sub["partner_local_mean_intensity"].values)
        nm = _display_condition(cond) if cond != "__all__" else "all"
        cond_lines.append(f"{nm}: ρ={csr:.2f} ({_fmt_p(csp)})")
        foot_bits.append(f"{nm}: Spearman ρ={csr:.3f} ({_fmt_p(csp)}), "
                         f"Pearson r={cpr:.3f} ({_fmt_p(cpp)}), n={len(sub)}")
    pr, pp, sr, sp = _pearson_spearman(
        d["spot_peak_intensity"].values, d["partner_local_mean_intensity"].values)
    ax.set_xlabel(f"{rna} spot peak intensity (raw)")
    ax.set_ylabel(f"{partner} local mean at {rna} spot (raw)")
    ax.set_title(_wrap_title(
        f"Dose-dependence: {partner} intensity vs {rna} spot brightness — by condition"))
    if len(real_conds) > 1:
        ax.legend(fontsize=7, framealpha=0.9, markerscale=1.6, loc="lower right")
    ax.grid(alpha=0.25, linestyle="--"); ax.set_axisbelow(True)
    # In-grid headline Spearman (legible) so r/p reads without the footnote.
    if cond_lines:
        ax.text(0.035, 0.965, "Spearman ρ (peak vs partner)\n" + "\n".join(cond_lines),
                transform=ax.transAxes, ha="left", va="top", fontsize=8,
                color="#1f1f1f", zorder=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc",
                          alpha=0.88))
    # Clip the VIEW to a robust 99th-pct upper bound (extreme MIAT peaks stretch
    # the axis to ~65k and compress the bulk). Points are NOT dropped — the
    # Spearman/Pearson above use every spot.
    _peak = pd.to_numeric(d["spot_peak_intensity"], errors="coerce").values
    _p99 = float(np.nanpercentile(_peak, 99))
    if np.isfinite(_p99) and _p99 > 0:
        _lo = float(np.nanmin(_peak))
        ax.set_xlim(left=min(_lo, _p99 * 0.98), right=_p99)
    ax._coloc_extra_footnote = (
        f"Per-spot {rna} (MIAT) spots, all FoV pooled (n={len(d)}). Spearman ρ "
        f"(rank-based, robust to per-section absolute scaling) = headline; "
        f"Pearson r secondary. " + "  |  ".join(foot_bits) +
        f"  Pooled: ρ={sr:.3f} ({_fmt_p(sp)}), r={pr:.3f} ({_fmt_p(pp)}). "
        f"Point = one {rna} spot; y = {partner} mean in a spot-radius disk at "
        f"the spot; x clipped at the 99th pct for display only.")


def coloc_dose_dependence_faceted(spots, out_dir, condition_order=None):
    """Standalone 77b — the dose-dependence scatter FACETED one-panel-per-
    condition (shared axes) so per-condition trends read side-by-side. Saves
    figures/07_coloc/77b_coloc_dose_dependence_faceted.png. Self-skips (no PNG)
    when partner_local_mean_intensity is absent."""
    rna = _LABELS.get("rna_label", "RNA1")
    partner = _coloc_partner_label()
    d = _rna1_spots_with_partner(spots)
    if d is None:
        try:
            print("[coloc] partner_local_mean_intensity absent — skipping "
                  "faceted dose-dependence 77b.")
        except Exception:
            pass
        return None
    conds_in = (d["condition"].dropna().unique().tolist()
                if "condition" in d.columns else [])
    conditions = order_conditions(conds_in, condition_order or []) or ["__all__"]
    n = len(conditions)
    fig, axes = plt.subplots(1, n, figsize=(max(4.2, 3.5 * n), 4.4), dpi=600,
                             squeeze=False, sharex=True, sharey=True)
    foot_bits = []
    for ci, cond in enumerate(conditions):
        ax = axes[0][ci]
        sub = d[d["condition"] == cond] if (cond != "__all__" and "condition" in d.columns) else d
        col = _color_for_condition(cond, ci)
        if not sub.empty:
            ax.scatter(sub["spot_peak_intensity"], sub["partner_local_mean_intensity"],
                       s=8, alpha=0.35, edgecolor="none", color=col)
            pr, pp, sr, sp = _pearson_spearman(
                sub["spot_peak_intensity"].values,
                sub["partner_local_mean_intensity"].values)
            # FIX 3: Spearman ρ (+ p) is the legible headline; Pearson secondary.
            ax.text(0.045, 0.955, f"ρ={sr:.2f} ({_fmt_p(sp)})\nr={pr:.2f}",
                    transform=ax.transAxes, ha="left", va="top", fontsize=9,
                    color="#1f1f1f", zorder=8,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec="#cccccc", alpha=0.88))
            foot_bits.append(
                f"{_display_condition(cond)}: Spearman ρ={sr:.3f} ({_fmt_p(sp)}), "
                f"Pearson r={pr:.3f} ({_fmt_p(pp)}), n={len(sub)}")
        ax.set_title(_display_condition(cond) if cond != "__all__" else "all",
                     fontsize=10, color=col, fontweight="bold")
        ax.set_xlabel(f"{rna} spot peak (raw)")
        if ci == 0:
            ax.set_ylabel(f"{partner} local mean at {rna} spot")
        ax.grid(alpha=0.25, linestyle="--"); ax.set_axisbelow(True)
    # FIX 3: clip the shared x-view to a robust 99th-pct upper bound (extreme
    # MIAT peaks compress the bulk). View only — every spot is in the stats.
    _peak = pd.to_numeric(d["spot_peak_intensity"], errors="coerce").values
    _p99 = float(np.nanpercentile(_peak, 99))
    if np.isfinite(_p99) and _p99 > 0:
        _lo = float(np.nanmin(_peak))
        axes[0][0].set_xlim(left=min(_lo, _p99 * 0.98), right=_p99)
    fig.suptitle(_wrap_title(
        f"Dose-dependence (faceted): {partner} intensity vs {rna} spot brightness",
        width=84), fontsize=12, fontweight="bold")
    import textwrap as _tw
    foot = f"Per-spot {rna} (MIAT) spots.  " + "  |  ".join(foot_bits)
    fig.tight_layout(rect=(0, 0.11, 1, 0.93))
    fig.text(0.5, 0.02, "\n".join(_tw.wrap(foot, width=int(max(80, 13.0 * (3.5 * n))))),
             ha="center", va="bottom", fontsize=6.5, color="#555555", linespacing=1.25)
    _relabel_fig(fig)
    out_pn = _coloc_fig_dir(out_dir) / "77b_coloc_dose_dependence_faceted.png"
    fig.savefig(out_pn, dpi=600)
    plt.close(fig)
    return out_pn


def coloc_spot_vs_threshold_summary(nuc, summary, out_dir, condition_order=None):
    """Panel 80 (standalone) — spot-calling vs whole-nucleus threshold coloc,
    side by side. LEFT (HEADLINE): per-condition pooled QKI-at-MIAT enrichment
    vs the per-nucleus random-position null (>1 = enriched), annotated with the
    pooled null z + empirical p. RIGHT (washed-out CONTEXT): whole-nucleus
    Manders M1/M2 + Pearson r + a thresholded-intensity mass ratio — these
    dilute toward no-signal because QKI fills the nucleus. Saves
    figures/07_coloc/80_coloc_spot_vs_threshold.png. Accepts protein_*/rna2_*
    column names via _first_present_col. Self-skips (no PNG) when neither metric
    family is present."""
    rna = _LABELS.get("rna_label", "RNA1")
    partner = _coloc_partner_label()

    # ---- WITH spot-calling: per-condition pooled enrichment (headline) ----
    with_data = {}   # cond -> (enrich_mean, z_mean, p_mean, n)
    enr_col = (_first_present_col(
        summary, ("protein_pooled_enrichment_vs_null_at_rna1_spots",
                  "rna2_pooled_enrichment_vs_null_at_rna1_spots"))
        if summary is not None else None)
    null_stats = _pooled_null_stats_by_condition(summary, condition_order)
    if summary is not None and enr_col is not None and "condition" in summary.columns:
        s = summary.copy()
        s[enr_col] = pd.to_numeric(s[enr_col], errors="coerce")
        for cond, sub in s.groupby("condition", dropna=False):
            if cond is None:
                continue
            vals = sub[enr_col].dropna().values
            if len(vals) == 0:
                continue
            rec = null_stats.get(str(cond), {})
            with_data[str(cond)] = (float(np.mean(vals)),
                                    rec.get("z", float("nan")),
                                    rec.get("p", float("nan")), len(vals))

    # ---- WITHOUT spot-calling: whole-nucleus washed-out metrics ----
    m1c = _first_present_col(nuc, ("manders_rna1_in_protein",))
    m2c = _first_present_col(nuc, ("manders_protein_in_rna1",))
    prc = _first_present_col(nuc, ("coloc_pearson_r_rna_protein",))
    tnum = _first_present_col(nuc, ("protein_thresh_total_intensity_nuclear",
                                    "rna2_thresh_total_intensity_nuclear"))
    tden = _first_present_col(nuc, ("rna_thresh_total_intensity_nuclear",))
    has_without = any(c is not None for c in (m1c, m2c, prc))

    if not with_data and not has_without:
        try:
            print("[coloc] no spot-null or whole-nucleus coloc columns — "
                  "skipping panel 80.")
        except Exception:
            pass
        return None

    conds_present = list(with_data.keys())
    if "condition" in nuc.columns:
        conds_present += [c for c in nuc["condition"].dropna().unique().tolist()
                          if c not in conds_present]
    conditions = order_conditions(conds_present, condition_order or [])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 5.4), dpi=600)

    # LEFT: pooled enrichment headline bar
    for ci, cond in enumerate(conditions):
        rec = with_data.get(cond)
        h = rec[0] if rec else float("nan")
        col = _color_for_condition(cond, ci)
        axL.bar(ci, h if np.isfinite(h) else 0.0, color=col, edgecolor="black",
                linewidth=0.6, width=0.7)
        if rec:
            bits = []
            if np.isfinite(rec[1]):
                bits.append(f"z={rec[1]:.1f}")
            if np.isfinite(rec[2]):
                bits.append(f"{_stars_for_p(rec[2])} {_fmt_p(rec[2])}")
            if bits and np.isfinite(h):
                axL.text(ci, h, "pooled null\n" + "  ".join(bits), ha="center",
                         va="bottom", fontsize=7, color="#1f1f1f")
    axL.axhline(1.0, color="#555555", linestyle="--", linewidth=1.1,
                label="no enrichment (×1.0)")
    axL.set_xticks(list(range(len(conditions))))
    axL.set_xticklabels([_display_condition(c) for c in conditions])
    axL.set_ylabel(f"{partner} at {rna} foci ÷ per-nucleus null (×)")
    axL.set_title(_wrap_title(
        "WITH spot-calling  ·  HEADLINE METRIC\n"
        f"{partner} at {rna} foci ÷ per-nucleus random-position null "
        "(>1 = enriched)", width=46))
    axL.legend(fontsize=7); axL.grid(axis="y", alpha=0.25, linestyle="--")
    axL.set_axisbelow(True)
    # FIX 2 (2026-06-06): the modest enrichment (~1.05–1.15) is invisible against
    # the ×1.0 line when y starts at 0. Zoom to ~[0.9, max_bar*1.05] (keeps the
    # dashed 1.0 reference visible) so the enrichment reads clearly above 1.0.
    _finite_h = [rec[0] for rec in with_data.values()
                 if rec and np.isfinite(rec[0])]
    if _finite_h:
        _max_bar = max(_finite_h + [1.0])
        axL.set_ylim(0.9, _max_bar * 1.05 + 0.02)

    # RIGHT: whole-nucleus washed-out metrics (grouped condition-mean bars)
    metric_defs = []
    if m1c:
        metric_defs.append((f"Manders M1\n({rna} in {partner})", m1c))
    if m2c:
        metric_defs.append((f"Manders M2\n({partner} in {rna})", m2c))
    if prc:
        metric_defs.append(("Pearson r\n(whole nucleus)", prc))
    if tnum and tden:
        metric_defs.append((f"{partner}/{rna}\nthresh-mass ratio", None))
    if metric_defs:
        n_metrics = len(metric_defs)
        bar_w = 0.8 / max(1, len(conditions))
        for ci, cond in enumerate(conditions):
            sub = nuc[nuc["condition"] == cond] if "condition" in nuc.columns else nuc
            col = _color_for_condition(cond, ci)
            vals = []
            for mlabel, mcol in metric_defs:
                if mcol is not None:
                    vals.append(pd.to_numeric(sub[mcol], errors="coerce").mean())
                else:
                    num = pd.to_numeric(sub[tnum], errors="coerce")
                    den = pd.to_numeric(sub[tden], errors="coerce").replace(0, np.nan)
                    vals.append(float((num / den).mean()))
            xpos = np.arange(n_metrics) + (ci - (len(conditions) - 1) / 2.0) * bar_w
            axR.bar(xpos, vals, width=bar_w, color=col, edgecolor="black",
                    linewidth=0.5, label=_display_condition(cond))
        axR.set_xticks(np.arange(n_metrics))
        axR.set_xticklabels([m[0] for m in metric_defs], fontsize=8)
        axR.legend(fontsize=7)
    axR.set_ylabel("whole-nucleus coloc (condition mean)")
    axR.set_title(_wrap_title(
        "WITHOUT spot-calling  ·  context only\n"
        f"whole-nucleus Manders / Pearson — washed out ({partner} fills the "
        "nucleus)", width=46))
    axR.grid(axis="y", alpha=0.25, linestyle="--"); axR.set_axisbelow(True)

    fig.suptitle(_wrap_title(
        f"Colocalization: spot-calling vs whole-nucleus threshold — {rna} × {partner}",
        width=92), fontsize=12.5, fontweight="bold")
    import textwrap as _tw
    foot = (f"HEADLINE (left) = the per-nucleus random-position null enrichment "
            f"({partner} at {rna} foci ÷ a spatial control) — this is the metric "
            f"to read. The right-hand whole-nucleus metrics (Manders M1/M2, "
            f"Pearson r, thresholded-intensity mass ratio) are EXPECTED to wash "
            f"out toward no-signal because {partner} fills the nucleus — context "
            f"only, not the result.")
    fig.tight_layout(rect=(0, 0.10, 1, 0.93))
    fig.text(0.5, 0.015, "\n".join(_tw.wrap(foot, width=140)), ha="center",
             va="bottom", fontsize=6.5, color="#555555", linespacing=1.25)
    _relabel_fig(fig)
    out_pn = _coloc_fig_dir(out_dir) / "80_coloc_spot_vs_threshold.png"
    fig.savefig(out_pn, dpi=600)
    plt.close(fig)
    return out_pn


def _coloc_null_enrichment_by_condition(draws_df, summary=None, condition_order=None):
    """Panel-78 support (FIX 1, 2026-06-06) — put every FoV's random-position
    null on a common ENRICHMENT scale (observed/random, centered at 1.0) so FoV
    recorded at different ABSOLUTE QKI brightness (laser power re-tuned per
    section) pool legitimately within a condition.

    For each image: ``null_enrichment = pooled_null_value / (that image's stored
    pooled null mean from per_image_summary)`` — falling back to the image's own
    draw mean only when the summary mean is absent. Observed enrichment per FoV =
    ``pooled_obs / null_mean`` (== ``protein_pooled_enrichment_vs_null_at_rna1_spots``).

    The annotated z / empirical p are the VALIDATED within-nucleus pooled stats
    read straight from per_image_summary (IDENTICAL source to panel 76 via
    ``_pooled_null_stats_by_condition``) — NEVER recomputed from the pooled-draw
    SD, which the cross-FoV absolute-brightness spread would inflate.

    Returns ``{condition: {"null_enrichment": np.ndarray, "obs": [per-FoV],
    "obs_mean": float, "z": float, "p": float, "n_fov": int}}`` plus a
    ``"__pooled__"`` entry (all FoV; z/p = mean of the stored per-FoV stats).
    Empty dict when there are no usable draws."""
    out: dict = {}
    if (draws_df is None or not hasattr(draws_df, "columns")
            or not {"image", "condition", "pooled_null_value"}.issubset(set(draws_df.columns))):
        return out
    d = draws_df.copy()
    d["pooled_null_value"] = pd.to_numeric(d["pooled_null_value"], errors="coerce")
    d = d[d["pooled_null_value"].notna()]
    if d.empty:
        return out

    # Per-image stored null mean + stored per-FoV z/p from per_image_summary.
    nm_by_img: dict = {}
    z_by_img: dict = {}
    p_by_img: dict = {}
    if summary is not None and hasattr(summary, "columns") and "image" in summary.columns:
        nm_col = _first_present_col(
            summary, ("protein_pooled_null_mean_at_rna1_spots",
                      "rna2_pooled_null_mean_at_rna1_spots"))
        z_col = _first_present_col(
            summary, ("protein_pooled_null_z_at_rna1_spots",
                      "rna2_pooled_null_z_at_rna1_spots"))
        p_col = _first_present_col(
            summary, ("protein_pooled_null_p_empirical_at_rna1_spots",
                      "rna2_pooled_null_p_empirical_at_rna1_spots"))
        for _, srow in summary.iterrows():
            img = srow.get("image")
            if img is None:
                continue
            if nm_col is not None:
                nm_by_img[img] = pd.to_numeric(srow.get(nm_col), errors="coerce")
            if z_col is not None:
                z_by_img[img] = pd.to_numeric(srow.get(z_col), errors="coerce")
            if p_col is not None:
                p_by_img[img] = pd.to_numeric(srow.get(p_col), errors="coerce")

    # Per-condition pooled z/p — the SAME values panel 76 annotates.
    cond_stats = _pooled_null_stats_by_condition(summary, condition_order)

    # Build per-image enrichment arrays + observed enrichment.
    per_img: dict = {}   # image -> {"cond", "null_enrich", "obs_enrich"}
    for img, g in d.groupby("image"):
        nullv = g["pooled_null_value"].values.astype(float)
        nm = nm_by_img.get(img, float("nan"))
        if not (np.isfinite(nm) and nm > 0):
            nm = float(np.mean(nullv))     # fall back to the image's own draw mean
        if not (np.isfinite(nm) and nm > 0):
            continue
        null_enrich = nullv / nm
        obs_enrich = float("nan")
        if "pooled_obs" in g.columns:
            ov = pd.to_numeric(g["pooled_obs"], errors="coerce").dropna()
            if len(ov):
                obs_enrich = float(ov.iloc[0]) / nm
        cond_val = g["condition"].iloc[0]
        per_img[img] = {"cond": cond_val, "null_enrich": null_enrich,
                        "obs_enrich": obs_enrich}

    if not per_img:
        return out

    conds_in = d["condition"].dropna().unique().tolist()
    conditions = order_conditions(conds_in, condition_order or [])

    def _finite(vals):
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")

    for cond in conditions:
        imgs = [im for im, r in per_img.items() if r["cond"] == cond]
        if not imgs:
            continue
        ne = np.concatenate([per_img[im]["null_enrich"] for im in imgs])
        obs_list = [per_img[im]["obs_enrich"] for im in imgs]
        rec = cond_stats.get(str(cond), {})
        out[str(cond)] = {
            "null_enrichment": ne,
            "obs": obs_list,
            "obs_mean": _finite(obs_list),
            "z": rec.get("z", float("nan")),
            "p": rec.get("p", float("nan")),
            "n_fov": len(imgs),
        }

    # Pooled across all conditions (already on the common enrichment scale).
    all_imgs = list(per_img.keys())
    ne_all = np.concatenate([per_img[im]["null_enrich"] for im in all_imgs])
    obs_all = [per_img[im]["obs_enrich"] for im in all_imgs]
    out["__pooled__"] = {
        "null_enrichment": ne_all,
        "obs": obs_all,
        "obs_mean": _finite(obs_all),
        "z": _finite([z_by_img.get(im, float("nan")) for im in all_imgs]),
        "p": _finite([p_by_img.get(im, float("nan")) for im in all_imgs]),
        "n_fov": len(all_imgs),
    }
    return out


def _draw_coloc_null_overlay_axis(ax, rec, key, *, label=None,
                                  partner=None, rna=None):
    """Panel-78 single-axis renderer (CLARITY FIX, 2026-06-06 Brian: the null
    histogram "almost looked like the observed"). Draws the random-position null
    as a clearly-SECONDARY light-grey filled histogram explicitly labeled
    "random-position null", and the OBSERVED enrichment as a BOLD high-contrast
    magenta vertical line carrying a DIRECT arrow callout placed AT the line —
    "OBSERVED / <partner>@<rna> / z=…, p=…" — so the reader sees at a glance that
    the bold marked value is the real measurement sitting far out in the right
    tail of the grey null (not just a legend entry). x-axis is the enrichment
    scale centered at 1.0. Returns the callout text, or None when nothing was
    drawn (caller hides the axis)."""
    if partner is None:
        partner = _coloc_partner_label()
    if rna is None:
        rna = _LABELS.get("rna_label", "RNA1")
    if label is None:
        label = ("pooled (all conditions, per-FoV normalized)"
                 if key == "__pooled__" else _display_condition(key))
    if not rec:
        return None
    ne = np.asarray(rec.get("null_enrichment", []), float)
    ne = ne[np.isfinite(ne)]
    if len(ne) == 0:
        return None
    # NULL = neutral, secondary, light-grey fill (explicitly labeled).
    ax.hist(ne, bins=40, color="#C9C9C9", alpha=0.85, edgecolor="#9E9E9E",
            linewidth=0.4, density=True, zorder=1, label="random-position null")
    ax.axvline(1.0, color="#777777", linestyle=":", linewidth=1.0, zorder=2,
               label="no enrichment (×1.0)")
    # OBSERVED = bold high-contrast magenta line(s), one per FoV (pooled: mean).
    obs_vals = ([rec.get("obs_mean")] if key == "__pooled__"
                else list(rec.get("obs", [])))
    obs_vals = [float(v) for v in obs_vals if v is not None and np.isfinite(v)]
    for oi, ov in enumerate(obs_vals):
        ax.axvline(ov, color=_QKI_MAGENTA, linewidth=2.6, zorder=5,
                   label=("OBSERVED (per FoV)" if oi == 0 else None))
    # Give the right tail headroom so the OBSERVED line + callout read as "far
    # out in the right tail" with whitespace, not crammed at the axis edge.
    lo = min(float(ne.min()), 1.0)
    hi = max(float(ne.max()), (max(obs_vals) if obs_vals else 1.0))
    span = (hi - lo) or 1.0
    ax.set_xlim(lo - 0.05 * span, hi + 0.18 * span)
    # DIRECT callout AT the observed line (arrow), not just a legend entry.
    callout = None
    if obs_vals:
        ov_rep = float(np.mean(obs_vals))
        z = rec.get("z", float("nan")); p = rec.get("p", float("nan"))
        zp = []
        if np.isfinite(z):
            zp.append(f"z={z:.1f}")
        if np.isfinite(p):
            zp.append(_fmt_p(p))
        callout = f"OBSERVED\n{partner}@{rna}"
        if zp:
            callout += "\n" + ", ".join(zp)
        ax.annotate(
            callout, xy=(ov_rep, 0.55), xycoords=("data", "axes fraction"),
            xytext=(0.96, 0.94), textcoords="axes fraction",
            ha="right", va="top", fontsize=8.5, fontweight="bold",
            color=_QKI_MAGENTA, zorder=7,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=_QKI_MAGENTA,
                      lw=1.2, alpha=0.96),
            arrowprops=dict(arrowstyle="-|>", color=_QKI_MAGENTA, lw=1.8,
                            connectionstyle="arc3,rad=0.18"))
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
    ax.set_title(label, fontsize=10)
    ax.set_xlabel(f"{partner} enrichment at {rna} foci (observed / random-null)")
    ax.set_ylabel("density")
    ax.grid(alpha=0.2, linestyle="--"); ax.set_axisbelow(True)
    return callout


def coloc_null_distribution_overlay(out_dir, condition_order, draws_df, summary=None):
    """Panel 78 (standalone) — per-condition (+ pooled) histogram of the per-
    nucleus random-position null on an ENRICHMENT scale (observed/random,
    centered at 1.0), drawn as a clearly-SECONDARY light-grey fill, with the
    OBSERVED QKI-at-MIAT enrichment marked in the right tail by a BOLD magenta
    line + a direct arrow callout (CLARITY FIX 2026-06-06 — the null used to
    look like the observed), annotated with the VALIDATED pooled z + empirical p
    (same source as panel 76).

    FIX 1 (2026-06-06): the original panel histogrammed the ABSOLUTE
    ``pooled_null_value`` pooled across FoV — bimodal, because each section sat
    at a different absolute QKI brightness (laser re-tuned) — and RECOMPUTED z
    from that inflated cross-FoV SD, which contradicted the validated within-
    nucleus z (a dataset-rule violation: pooling absolute intensity across
    sections). Now every FoV's null is normalized by that image's stored null
    mean (``_coloc_null_enrichment_by_condition``), so pooling is unimodal ~1.0,
    and the annotated z/p are read from per_image_summary, not recomputed.

    Reads ``draws_df`` (coloc_null_draws.csv cols: image, condition, iter,
    pooled_null_value, pooled_obs) + ``summary`` (per_image_summary, for the
    per-image null mean and the validated z/p). Saves
    figures/07_coloc/78_coloc_null_distribution_overlay.png. Self-skips (no PNG)
    when draws_df is None/empty (backfill CSV absent)."""
    partner = _coloc_partner_label()
    rna = _LABELS.get("rna_label", "RNA1")
    if draws_df is None or not hasattr(draws_df, "columns") or len(draws_df) == 0:
        try:
            print("[coloc] coloc_null_draws absent — skipping null-overlay panel 78.")
        except Exception:
            pass
        return None
    if not {"condition", "pooled_null_value"}.issubset(set(draws_df.columns)):
        print("[coloc] coloc_null_draws missing required columns — skipping panel 78.")
        return None

    ann = _coloc_null_enrichment_by_condition(draws_df, summary, condition_order)
    if not ann:
        print("[coloc] coloc_null_draws yielded no usable enrichment — skipping panel 78.")
        return None
    conditions = [c for c in order_conditions(
        [k for k in ann.keys() if k != "__pooled__"], condition_order or [])]
    panels = conditions + (["__pooled__"] if "__pooled__" in ann else [])
    n = len(panels)
    ncol = min(3, n); nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.6 * nrow),
                             dpi=600, squeeze=False)
    for pi, key in enumerate(panels):
        ax = axes[pi // ncol][pi % ncol]
        rec = ann.get(key)
        label = ("pooled (all conditions, per-FoV normalized)"
                 if key == "__pooled__" else _display_condition(key))
        drew = _draw_coloc_null_overlay_axis(
            ax, rec, key, label=label, partner=partner, rna=rna)
        if drew is None:
            ax.set_visible(False)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].set_visible(False)
    fig.suptitle(_wrap_title(
        f"Random-position null vs observed {partner} enrichment at {rna} foci",
        width=82), fontsize=12.5, fontweight="bold")
    import textwrap as _tw
    foot = (f"Grey histogram = per-nucleus random-position null draws normalized "
            f"to ENRICHMENT (each FoV ÷ its own stored null mean, so FoV at "
            f"different absolute {partner} brightness pool on a common ~1.0 "
            f"scale); dotted = no enrichment (×1.0). Bold magenta line + callout "
            f"= OBSERVED {partner}-at-{rna} enrichment (one line per FoV), "
            f"sitting far out in the right tail of the grey null. z / empirical p "
            f"are the VALIDATED within-nucleus pooled stats from "
            f"per_image_summary (same source as the enrichment SuperPlot, panel "
            f"76) — not recomputed from the pooled-draw SD.")
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    fig.text(0.5, 0.015, "\n".join(_tw.wrap(foot, width=140)), ha="center",
             va="bottom", fontsize=6.5, color="#555555", linespacing=1.25)
    _relabel_fig(fig)
    out_pn = _coloc_fig_dir(out_dir) / "78_coloc_null_distribution_overlay.png"
    fig.savefig(out_pn, dpi=600)
    plt.close(fig)
    return out_pn


def coloc_radial_profile_plot(radial_df, out_dir, condition_order=None):
    """Panel 81 (standalone) — radial QKI profile around MIAT spots.

    CLARITY FIX (2026-06-06 Brian: lead with the rigorous metric, de-confuse the
    raw one). PRIMARY/larger LEFT panel = ENRICHMENT (obs ÷ null) vs annulus
    radius with a dashed 1.0 reference — this cancels ring geometry. SECONDARY/
    small RIGHT panel = the raw absolute QKI mean vs radius with the ~flat null
    mean ± SD band, captioned that it is within-condition only (laser power
    differs across sections — do NOT compare across conditions) and that its
    decline = local enrichment relaxing to the nucleoplasmic background (the null
    is ~flat), NOT an artifact. Reads ``radial_df`` (coloc_radial_profile.csv
    cols: image, condition, ring_um, obs_mean, null_mean, null_sd, enrichment,
    z, n_spots). Saves figures/07_coloc/81_coloc_radial_qki_profile.png.
    Self-skips (no PNG) when radial_df is None/empty."""
    partner = _coloc_partner_label()
    rna = _LABELS.get("rna_label", "RNA1")
    if radial_df is None or not hasattr(radial_df, "columns") or len(radial_df) == 0:
        try:
            print("[coloc] coloc_radial_profile absent — skipping radial panel 81.")
        except Exception:
            pass
        return None
    if not {"condition", "ring_um", "obs_mean", "null_mean"}.issubset(set(radial_df.columns)):
        print("[coloc] coloc_radial_profile missing required columns — skipping panel 81.")
        return None
    d = radial_df.copy()
    for c in ("ring_um", "obs_mean", "null_mean", "null_sd", "enrichment"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    conds_in = d["condition"].dropna().unique().tolist()
    conditions = order_conditions(conds_in, condition_order or [])
    # PRIMARY (left, larger) = enrichment (geometry-cancelled); SECONDARY (right,
    # small) = raw absolute, demoted per Brian (2026-06-06).
    fig, (axP, axR) = plt.subplots(
        1, 2, figsize=(12.0, 4.8), dpi=600,
        gridspec_kw={"width_ratios": [1.7, 1.0]})
    for ci, cond in enumerate(conditions):
        sub = d[d["condition"] == cond]
        if sub.empty:
            continue
        agg = sub.groupby("ring_um", as_index=False).mean(numeric_only=True)
        agg = agg.sort_values("ring_um")
        rings = agg["ring_um"].values
        obs = agg["obs_mean"].values
        nullm = agg["null_mean"].values
        nullsd = agg["null_sd"].values if "null_sd" in agg.columns else np.zeros_like(nullm)
        if "enrichment" in agg.columns:
            enr = agg["enrichment"].values
        else:
            enr = obs / np.where(nullm == 0, np.nan, nullm)
        col = _color_for_condition(cond, ci)
        # PRIMARY: enrichment vs ring (bold).
        axP.plot(rings, enr, "-o", color=col, linewidth=2.2, markersize=7,
                 label=_display_condition(cond))
        # SECONDARY: raw absolute observed + ~flat null mean ± SD band.
        axR.plot(rings, obs, "-o", color=col, linewidth=1.2, markersize=4,
                 label=f"{_display_condition(cond)} (obs)")
        axR.fill_between(rings, nullm - nullsd, nullm + nullsd, color=col, alpha=0.15)
        axR.plot(rings, nullm, "--", color=col, alpha=0.6, linewidth=1.0)
    # ---- PRIMARY: enrichment (the rigorous, geometry-cancelled metric) ----
    axP.axhline(1.0, color="#555555", linestyle="--", linewidth=1.2,
                label="no enrichment (×1.0)")
    axP.set_xlabel("annulus radius (µm)")
    axP.set_ylabel("enrichment vs random-position null (×)")
    axP.set_title(_wrap_title(
        f"{partner} enrichment around {rna} foci vs annulus radius  —  PRIMARY\n"
        "(obs ÷ null; geometry-cancelled; dashed = no enrichment ×1.0)",
        width=58), fontsize=11)
    axP.legend(fontsize=8); axP.grid(alpha=0.25, linestyle="--"); axP.set_axisbelow(True)
    # ---- SECONDARY: raw absolute (demoted; within-condition only) ----
    axR.set_xlabel("annulus radius (µm)")
    axR.set_ylabel(f"{partner} mean intensity (raw, a.u.)")
    axR.set_title(_wrap_title(
        f"Absolute {partner} (secondary; within-condition only)\n"
        "solid = observed, dashed = ~flat null ± SD — do NOT compare across "
        "sections (laser power differs)", width=50), fontsize=8.5)
    axR.legend(fontsize=7); axR.grid(alpha=0.25, linestyle="--"); axR.set_axisbelow(True)
    fig.suptitle(_wrap_title(f"Radial {partner} profile around {rna} foci", width=80),
                 fontsize=12.5, fontweight="bold")
    import textwrap as _tw
    foot = (f"Annulus rings centered on each {rna} spot; obs = mean {partner} in "
            f"the ring, null = per-nucleus random-position null; enrichment = "
            f"obs ÷ null (spot-weighted, pooled per condition). Read the PRIMARY "
            f"(left) enrichment panel — it cancels ring geometry. The raw "
            f"{partner} (right) declines with radius because {partner} is "
            f"locally enriched at {rna} and relaxes to the nucleoplasmic "
            f"background (the null is ~flat), so the decline is biology, not an "
            f"artifact; absolute values are within-condition only (laser power "
            f"differs across sections).")
    fig.tight_layout(rect=(0, 0.10, 1, 0.92))
    fig.text(0.5, 0.015, "\n".join(_tw.wrap(foot, width=140)), ha="center",
             va="bottom", fontsize=6.5, color="#555555", linespacing=1.25)
    _relabel_fig(fig)
    out_pn = _coloc_fig_dir(out_dir) / "81_coloc_radial_qki_profile.png"
    fig.savefig(out_pn, dpi=600)
    plt.close(fig)
    return out_pn


# ---- Nucleolus figures ----

def nucleolus_fraction_superplot(ax, nuc, condition_order=None):
    """SuperPlot — nucleolus area as a fraction of the nucleus, by condition."""
    _nucleus_metric_superplot(
        ax, nuc, "nucleolus_fraction_of_nucleus",
        ylabel="Nucleolus fraction of nucleus", pct=True, cap_99=True,
        condition_order=condition_order,
        title=("SuperPlot: nucleolus area fraction of nucleus — by condition\n"
               "(per-nucleus nucleolus area ÷ nucleus area; large = per-image mean)"))


def nucleolus_area_superplot(ax, nuc, condition_order=None):
    """SuperPlot — nucleolus area (µm²) by condition."""
    vox = _voxel_xy_um_from(nuc)
    _nucleus_metric_superplot(
        ax, nuc, "nucleolus_area_px",
        ylabel="Nucleolus area (µm²)", only_positive=True, cap_99=True,
        transform=lambda s: s * (vox ** 2), condition_order=condition_order,
        title=("SuperPlot: nucleolus area — by condition\n"
               f"(px² → µm² at {vox:.3f} µm/px; large = per-image mean; 99th-pct capped)"))


def _per_nucleus_nucleolar_spot_fraction(spots: pd.DataFrame) -> pd.DataFrame:
    """Build a per-nucleus table: for each (image, condition, nucleus_id),
    the fraction of that nucleus's IN-NUCLEUS spots that are ALSO in the
    nucleolus. Returns a DataFrame with columns
    [image, condition, nucleus_id, nucleolar_spot_fraction, n_nuclear_spots].
    Nuclei with zero in-nucleus spots are dropped (fraction undefined)."""
    needed = {"in_nucleus", "in_nucleolus", "nucleus_id", "image", "condition"}
    if spots is None or len(spots) == 0 or not needed.issubset(set(spots.columns)):
        return pd.DataFrame(columns=["image", "condition", "nucleus_id",
                                     "nucleolar_spot_fraction", "n_nuclear_spots"])
    d = spots.copy()
    d["in_nucleus"] = pd.to_numeric(d["in_nucleus"], errors="coerce")
    d["in_nucleolus"] = pd.to_numeric(d["in_nucleolus"], errors="coerce")
    d["nucleus_id"] = pd.to_numeric(d["nucleus_id"], errors="coerce")
    d = d[(d["in_nucleus"] == 1) & d["nucleus_id"].notna() & (d["nucleus_id"] > 0)]
    if d.empty:
        return pd.DataFrame(columns=["image", "condition", "nucleus_id",
                                     "nucleolar_spot_fraction", "n_nuclear_spots"])
    grp = d.groupby(["image", "condition", "nucleus_id"], dropna=False)
    out = grp["in_nucleolus"].agg(["mean", "count"]).reset_index()
    out = out.rename(columns={"mean": "nucleolar_spot_fraction",
                              "count": "n_nuclear_spots"})
    return out


def nucleolar_spot_fraction_superplot(ax, spots, condition_order=None):
    """SuperPlot — per-cell fraction of NUCLEAR RNA spots that fall in the
    nucleolus, by condition. Computed per nucleus from spot_metrics, then the
    SuperPlot collapses to per-image means as biological replicates."""
    tbl = _per_nucleus_nucleolar_spot_fraction(spots)
    if tbl.empty:
        ax.set_visible(False); return
    # treat each nucleus as a 'unit' row for the superplot (unit='nucleus').
    if not _superplot_into_axes(ax, tbl, "nucleolar_spot_fraction",
                                ylabel="Nucleolar fraction of nuclear RNA spots",
                                unit="nucleus", pct=True,
                                condition_order=condition_order):
        ax.set_visible(False); return
    ax.set_title(_wrap_title(
        "SuperPlot: per-cell nucleolar fraction of nuclear RNA — by condition\n"
        "(per nucleus: in-nucleolus ÷ in-nucleus spots; large = per-image mean)"))


def nucleolus_collapsed_mean_nuclear_in_nucleolus(ax, spots, condition_order=None):
    """Per-condition COLLAPSED bar: mean % of nuclear RNA spots that are in
    the nucleolus. Bar = grand mean of per-image means, error = SD across
    image means, dots = per-image means. Reads the per-nucleus nucleolar spot
    fraction table and aggregates to image then condition."""
    tbl = _per_nucleus_nucleolar_spot_fraction(spots)
    if tbl.empty:
        ax.set_visible(False); return
    # per-image mean of the per-nucleus fractions
    per_img = (tbl.groupby(["condition", "image"])["nucleolar_spot_fraction"]
                  .mean().reset_index())
    conds_in = per_img["condition"].dropna().unique().tolist()
    order = order_conditions(conds_in, condition_order or [])
    lists = {c: per_img.loc[per_img["condition"] == c,
                            "nucleolar_spot_fraction"].dropna().tolist()
             for c in order}
    order = [c for c in order if lists.get(c)]
    if not order:
        ax.set_visible(False); return
    xs = np.arange(len(order))
    means = [float(np.mean(lists[c])) * 100.0 for c in order]
    sds = [float(np.std(lists[c], ddof=1)) * 100.0 if len(lists[c]) > 1 else 0.0
           for c in order]
    bar_colors = [_condition_family_base_color(c, i) for i, c in enumerate(order)]
    ax.bar(xs, means, yerr=sds, color=bar_colors, edgecolor="black",
           linewidth=0.6, alpha=0.55, capsize=5, zorder=2)
    rng = np.random.RandomState(11)
    for i, c in enumerate(order):
        vals = [v * 100.0 for v in lists[c]]
        jx = (rng.random(len(vals)) - 0.5) * 0.3
        ax.scatter(np.full(len(vals), i) + jx, vals, s=55, color=bar_colors[i],
                   edgecolor="#1f1f1f", linewidth=1.0, zorder=4)
        ax.text(i, means[i] + (sds[i] if sds[i] else 0) + 0.5, f"{means[i]:.1f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels([_display_condition(c) for c in order])
    ax.set_ylabel("% of nuclear RNA spots in nucleolus\n(grand mean of per-image means)")
    ax.grid(axis="y", alpha=0.25, linestyle="--"); ax.set_axisbelow(True)
    ax.set_title(_wrap_title(
        "Per-condition collapsed: mean % nuclear RNA in nucleolus\n"
        "(bar = mean of image means, error = SD, dots = per-image means)"))
    # Pairwise stats on per-image means (in % units to match the axis).
    gm = {c: [v * 100.0 for v in lists[c]] for c in order}
    _annotate_pairwise_brackets(ax, gm, order,
                                x_centers={c: i for i, c in enumerate(order)})


# ---- rna_only PARITY-GAP single-channel figures (vs rna_rna deck) ----
# These re-use the single-channel-applicable rna_rna helpers / metrics that
# the legacy rna_only layout never wired up. Each guards on column presence.

def rna_only_pct_cells_with_spots(ax, nuc, condition_order=None):
    """% of cells with ≥1 / ≥5 / ≥10 RNA spots, by condition (grouped bars).
    Single-channel analogue of the rna_rna 01b composition companion."""
    _bar_pct_cells_with_spots(ax, nuc, "rna1", condition_order=condition_order)


def rna_only_spot_count_bin_composition(ax, nuc, condition_order=None):
    """Per-condition stacked bar: % of cells in each total-spot-count bin
    (0 / 1-4 / 5-9 / 10+). Single-channel analogue of the rna_rna bin
    composition companions."""
    _bar_spot_count_bin_composition(
        ax, nuc, "rna_spot_count",
        title="RNA spot-count bin composition — by condition",
        axis_label="% of cells",
        condition_order=condition_order,
        subtitle=("Each cell is binned by its total RNA spot count; bars show "
                  "the fraction of cells in each bin per condition."))


def rna_only_spot_density_superplot(ax, nuc, condition_order=None):
    """SuperPlot — nuclear spot density (spots per µm²) by condition.
    Reads the per-nucleus nuclear_spot_density_per_um2 column directly."""
    _nucleus_metric_superplot(
        ax, nuc, "nuclear_spot_density_per_um2",
        ylabel="Nuclear spot density (spots / µm²)", only_positive=False,
        cap_99=True, condition_order=condition_order,
        title=("SuperPlot: nuclear spot density — by condition\n"
               "(per-nucleus nuclear spots ÷ nucleus area in µm²; 99th-pct capped)"))


def rna_only_anisotropy_superplot(ax, spots, condition_order=None):
    """Per-SPOT SuperPlot — spot anisotropy (axial:lateral elongation) by
    condition. Single-channel; reads per-spot spot_anisotropy."""
    if not _superplot_into_axes(ax, spots, "spot_anisotropy",
                                ylabel="Spot anisotropy (z:xy)", unit="spot",
                                only_positive=True, cap_99=True,
                                condition_order=condition_order):
        ax.set_visible(False); return
    ax.set_title(_wrap_title("SuperPlot: per-spot anisotropy — by condition\n"
                             "(axial:lateral elongation; large = per-image mean; 99th-pct capped)"))


def rna_only_frac_spots_nuc_edge_superplot(ax, nuc, condition_order=None):
    """SuperPlot — per-nucleus fraction of spots near the nuclear edge, by
    condition. Reads the per-nucleus frac_spots_nuc_edge column."""
    _nucleus_metric_superplot(
        ax, nuc, "frac_spots_nuc_edge",
        ylabel="Fraction of spots at nuclear edge", pct=True,
        condition_order=condition_order,
        title=("SuperPlot: per-nucleus fraction of spots at nuclear edge — by condition\n"
               "(higher = more peripheral spot localization; large = per-image mean)"))


# =====================================================================
# 2026-05-27 Brian: NEW publication panels (N1–N6)
# =====================================================================
# All follow the existing rna_only conventions: family colors
# (KD=Blues / NT=Reds / Sec-Only=Greys) via _build_family_color_map /
# _image_family_color; condition order from order_conditions; per-image
# (FoV) mean stats via _superplot_stats (Welch-t primary + MWU secondary);
# agnostic titles/labels; any log axis uses the thread-safe non-mathtext
# _set_log_axis_plain helper. Each FAILS LOUD (no silent drop): if a
# required column is missing it prints a WARN and hides the axis, which the
# standalone render loop surfaces as a non-rendered panel.

def _warn_missing(slug: str, missing) -> None:
    """Print a LOUD warning naming the panel + the columns it could not find,
    so a missing-data panel is never silently blank."""
    print(f"  WARN: panel {slug} has no data — missing/empty column(s): "
          f"{sorted(missing) if not isinstance(missing, str) else missing}")


def fewer_brighter_joint(ax, nuc, condition_order=None):
    """N1 — Per-nucleus scatter of spot COUNT (x) vs MEAN PER-SPOT intensity
    (y), one point per nucleus colored by condition family, with each
    condition's median crosshair overlaid. Brian's core 'fewer-but-brighter'
    view: does the contrast shift count, per-spot brightness, or both."""
    xcol = "rna_spot_count"
    ycol = _resolve_col(nuc, "rna_spot_mean_peak_intensity", "rna_spot_mean_intensity_fit")
    missing = {c for c in (xcol, ycol, "condition", "image") if c not in nuc.columns}
    if missing:
        _warn_missing("spotprop_fewer_brighter_joint", missing); ax.set_visible(False); return
    d = nuc.copy()
    d[xcol] = pd.to_numeric(d[xcol], errors="coerce")
    d[ycol] = pd.to_numeric(d[ycol], errors="coerce")
    # Expressing nuclei only — mean per-spot intensity is undefined with 0 spots.
    d = d[d[xcol].notna() & d[ycol].notna() & (d[xcol] > 0) & (d[ycol] > 0)]
    if d.empty:
        _warn_missing("spotprop_fewer_brighter_joint", "all rows non-positive/NaN")
        ax.set_visible(False); return
    conds = order_conditions(d["condition"].dropna().unique().tolist(), condition_order or [])
    family_map = _build_family_color_map(d, condition_order=condition_order)
    for ci, cond in enumerate(conds):
        sub = d[d["condition"] == cond]
        if sub.empty:
            continue
        pt_colors = [_image_family_color(family_map, im) or _condition_family_base_color(cond, ci)
                     for im in sub["image"]]
        ax.scatter(sub[xcol], sub[ycol], s=10, alpha=0.30, color=pt_colors,
                   edgecolor="none", zorder=2)
        # median crosshair in the condition family base color
        mx = float(np.median(sub[xcol])); my = float(np.median(sub[ycol]))
        base = _condition_family_base_color(cond, ci)
        ax.axvline(mx, color=base, linestyle="--", linewidth=1.0, alpha=0.7, zorder=4)
        ax.axhline(my, color=base, linestyle="--", linewidth=1.0, alpha=0.7, zorder=4)
        ax.scatter([mx], [my], s=170, color=base, edgecolor="#1f1f1f",
                   linewidth=1.5, marker="P", zorder=6,
                   label=f"{_display_condition(cond)} median ({mx:.0f}, {my:.0f})")
        # 2-D KDE contours per condition when there are enough clean points.
        try:
            if len(sub) >= 25:
                from scipy.stats import gaussian_kde as _kde
                xy = np.vstack([sub[xcol].values, sub[ycol].values])
                k = _kde(xy)
                xg = np.linspace(sub[xcol].min(), sub[xcol].max(), 60)
                yg = np.linspace(sub[ycol].min(), sub[ycol].max(), 60)
                XX, YY = np.meshgrid(xg, yg)
                ZZ = k(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
                ax.contour(XX, YY, ZZ, levels=4, colors=[base], linewidths=0.8,
                           alpha=0.55, zorder=3)
        except Exception:
            pass
    ax.set_xlabel("Spots per nucleus (count)")
    ax.set_ylabel("Mean per-spot peak FISH intensity / nucleus (a.u.)")
    ax.grid(True, alpha=0.25, linestyle="--"); ax.set_axisbelow(True)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
    ax.set_title(_wrap_title(
        "Per-nucleus spot count vs mean per-spot intensity (by condition)\n"
        "(one point per expressing nucleus; dashed = per-condition medians; contours = 2-D KDE)"))


def loc_composition_stacked(ax, spots, condition_order=None):
    """N2 — Parts-of-a-whole stacked bars: for each condition, the MEAN
    per-nucleus fraction of spots in {nuclear (excl. nucleolus), nucleolar,
    cytoplasmic}. Per-nucleus fractions are computed first, then averaged
    across nuclei (each nucleus weighted equally). Bars sum to 1.0."""
    needed = {"in_nucleus_excluding_nucleolus", "in_nucleolus", "in_cytoplasm",
              "nucleus_id", "image", "condition"}
    missing = needed - set(spots.columns)
    if missing:
        _warn_missing("loc_composition_stacked", missing); ax.set_visible(False); return
    d = spots.copy()
    for c in ("in_nucleus_excluding_nucleolus", "in_nucleolus", "in_cytoplasm", "nucleus_id"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    # Assign each spot to exactly one compartment; nucleolar takes priority over
    # the nuclear-excluding flag so the three compartments are mutually exclusive.
    comp = np.where(d["in_nucleolus"] == 1, "nucleolar",
            np.where(d["in_nucleus_excluding_nucleolus"] == 1, "nuclear",
            np.where(d["in_cytoplasm"] == 1, "cytoplasmic", "other")))
    d["_comp"] = comp
    d = d[d["_comp"] != "other"]
    d = d[d["nucleus_id"].notna() & (d["nucleus_id"] > 0)]
    if d.empty:
        _warn_missing("loc_composition_stacked", "no spots with a compartment flag")
        ax.set_visible(False); return
    comps = ["nuclear", "nucleolar", "cytoplasmic"]
    # per-nucleus fractions, then mean across nuclei within a condition
    grp = d.groupby(["condition", "image", "nucleus_id"])["_comp"]
    per_nuc = grp.value_counts(normalize=True).unstack(fill_value=0.0)
    for c in comps:
        if c not in per_nuc.columns:
            per_nuc[c] = 0.0
    per_nuc = per_nuc.reset_index()
    conds = order_conditions(per_nuc["condition"].dropna().unique().tolist(),
                             condition_order or [])
    # Three distinct compartment colors (NOT red+green): nuclear=blue,
    # nucleolar=purple, cytoplasmic=grey.
    comp_colors = {"nuclear": "#1f5fb0", "nucleolar": "#7b3fa0", "cytoplasmic": "#9a9a9a"}
    comp_label = {"nuclear": "Nuclear (excl. nucleolus)", "nucleolar": "Nucleolar",
                  "cytoplasmic": "Cytoplasmic"}
    xs = np.arange(len(conds))
    bottoms = np.zeros(len(conds))
    n_nuclei = [int((per_nuc["condition"] == c).sum()) for c in conds]
    for comp in comps:
        vals = [float(per_nuc.loc[per_nuc["condition"] == c, comp].mean()) if (per_nuc["condition"] == c).any() else 0.0
                for c in conds]
        ax.bar(xs, vals, bottom=bottoms, width=0.62, color=comp_colors[comp],
               edgecolor="white", linewidth=0.8, label=comp_label[comp], zorder=2)
        for xi, (b, v) in enumerate(zip(bottoms, vals)):
            if v >= 0.06:
                ax.text(xi, b + v / 2.0, f"{v*100:.0f}%", ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
        bottoms = bottoms + np.array(vals)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{_display_condition(c)}\n(n={n})" for c, n in zip(conds, n_nuclei)])
    ax.set_ylabel("Mean per-nucleus fraction of spots")
    ax.set_ylim(0, 1.0)
    import matplotlib.ticker as _mt
    ax.yaxis.set_major_formatter(_mt.PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(axis="y", alpha=0.25, linestyle="--"); ax.set_axisbelow(True)
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=3, framealpha=0.9)
    try:
        ax._subtitle_pad = max(float(getattr(ax, "_subtitle_pad", 0.0) or 0.0), 0.12)
    except Exception:
        pass
    ax.set_title(_wrap_title(
        "Spot localization composition by condition (mean per-nucleus fractions)\n"
        "(per nucleus: spots split into nuclear / nucleolar / cytoplasmic, averaged across nuclei; bars sum to 100%)"))


def ecdf_priority_metrics(ax, nuc, condition_order=None):
    """N3 — Two stacked ECDF axes: per-nucleus spot count and per-nucleus total
    RNA intensity, by condition (family colors). KD-vs-NT KS statistic + p is
    annotated on each. Splits the host axis into two via its subplotspec so it
    renders as one standalone figure."""
    total_col = _resolve_col(nuc, "rna_spot_total_peak_intensity", "rna_spot_total_intensity_fit")
    needed = {"rna_spot_count", total_col, "condition", "image"}
    missing = needed - set(nuc.columns)
    if missing:
        _warn_missing("ecdf_priority_metrics", missing); ax.set_visible(False); return
    fig = ax.figure
    try:
        from matplotlib.gridspec import GridSpecFromSubplotSpec
        gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=ax.get_subplotspec(), hspace=0.55)
        ax.set_visible(False)
        ax_top = fig.add_subplot(gs[0, 0])
        ax_bot = fig.add_subplot(gs[1, 0])
    except Exception:
        # Fallback: single axis, count only.
        ax_top, ax_bot = ax, None

    d = nuc.copy()
    conds = order_conditions(d["condition"].dropna().unique().tolist(), condition_order or [])

    def _ecdf_panel(a, col, xlabel):
        for ci, cond in enumerate(conds):
            vals = pd.to_numeric(d.loc[d["condition"] == cond, col],
                                 errors="coerce").dropna().values
            if len(vals) == 0:
                continue
            xs = np.sort(vals)
            ys = np.arange(1, len(xs) + 1) / len(xs)
            a.step(xs, ys, where="post", color=_condition_family_base_color(cond, ci),
                   linewidth=1.8, label=f"{_display_condition(cond)} (n={len(xs)})")
        a.set_xlabel(xlabel); a.set_ylabel("Cumulative fraction of nuclei")
        a.set_ylim(0, 1.02)
        a.grid(True, alpha=0.25, linestyle="--"); a.set_axisbelow(True)
        a.legend(fontsize=7.5, loc="lower right", framealpha=0.9)
        # KD-vs-NT KS annotation (per-NUCLEUS distributions; descriptive).
        try:
            from scipy.stats import ks_2samp as _ks
            kd = next((c for c in conds if "kd" in _norm_cond_key(c)), None)
            nt = next((c for c in conds if "nt" in _norm_cond_key(c)), None)
            if kd is not None and nt is not None:
                a_kd = pd.to_numeric(d.loc[d["condition"] == kd, col], errors="coerce").dropna().values
                a_nt = pd.to_numeric(d.loc[d["condition"] == nt, col], errors="coerce").dropna().values
                if len(a_kd) >= 2 and len(a_nt) >= 2:
                    st = _ks(a_kd, a_nt)
                    a.text(0.02, 0.97,
                           f"{_display_condition(kd)} vs {_display_condition(nt)}: "
                           f"KS D={st.statistic:.3f}, {_fmt_p(st.pvalue)}",
                           transform=a.transAxes, ha="left", va="top", fontsize=7.5,
                           color="#1f1f1f",
                           bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#bbb", alpha=0.85))
        except Exception:
            pass

    _ecdf_panel(ax_top, "rna_spot_count", "Spots per nucleus (count)")
    if ax_bot is not None:
        _ecdf_panel(ax_bot, total_col,
                    "Summed peak FISH intensity / nucleus (a.u.)")
        ax_top.set_title(_wrap_title(
            "Cumulative distributions: spots/nucleus and total RNA intensity (by condition)\n"
            "(per-nucleus ECDFs; KS statistic computed on per-nucleus values)"), fontsize=10)


def nc_ratio_superplot(ax, nuc, condition_order=None):
    """N4 — SuperPlot of per-nucleus nuclear:cytoplasmic RNA ratio by condition
    (FoV-mean Welch + MWU stats via the shared SuperPlot helper)."""
    col = "rna_nc_ratio"
    if col not in nuc.columns:
        _warn_missing("loc_nc_ratio_superplot", {col}); ax.set_visible(False); return
    if not _superplot_into_axes(
            ax, nuc, col,
            ylabel="Nuclear : cytoplasmic RNA ratio",
            unit="nucleus", only_positive=True, cap_99=True,
            condition_order=condition_order):
        _warn_missing("loc_nc_ratio_superplot", "no positive finite values")
        ax.set_visible(False); return
    ax.axhline(1.0, color="#444", linestyle=":", linewidth=0.9, alpha=0.6, zorder=1)
    ax.set_title(_wrap_title(
        "Nuclear:cytoplasmic RNA ratio per nucleus (by condition)\n"
        "(dotted = parity (N=C); 99th-pct capped; Welch + MWU on per-image means)"))


def per_spot_intensity_violin(ax, spots, condition_order=None):
    """N5 — Per-SPOT distribution of PEAK-pixel intensity by condition
    (SuperPlot violin+dots; individual spots, family colors). Log-x is not used
    (this is a per-spot SuperPlot with a vertical value axis); when the range is
    wide the 99th-pct cap keeps the violins legible. Median per condition is the
    large per-image-mean marker drawn by the SuperPlot helper."""
    col = _resolve_col(spots, "peak_intensity", "integrated_intensity_fit", "spot_peak_intensity")
    if col not in spots.columns:
        _warn_missing("spotprop_per_spot_intensity_violin", {col}); ax.set_visible(False); return
    sub = spots
    if "fit_ok" in spots.columns:
        sub = spots[pd.to_numeric(spots["fit_ok"], errors="coerce") == 1]
    if not _superplot_into_axes(
            ax, sub, col,
            ylabel="Per-spot peak FISH intensity (a.u.)",
            unit="spot", only_positive=True, cap_99=True,
            condition_order=condition_order):
        _warn_missing("spotprop_per_spot_intensity_violin", "no positive finite values")
        ax.set_visible(False); return
    ax.set_title(_wrap_title(
        "Per-spot peak intensity distribution (by condition)\n"
        "(fit-ok spots; violin = all spots, large = per-image mean; 99th-pct capped)"))


def _fmt_pct_signed(v) -> str:
    """Compact signed percent for an effect-size readout: '+27%', '-8%',
    '0%'. Returns 'n/a' for non-finite."""
    if v is None or not np.isfinite(v):
        return "n/a"
    r = round(v)
    if r == 0:           # avoid the odd-looking '-0%' / '+0%'
        return "0%"
    return f"{r:+.0f}%"


def effect_size_summary(ax, nuc, condition_order=None):
    """N6 — Clean horizontal forest of NT→KD percent change for the headline
    per-nucleus metrics. Each row shows the percent change of per-image (FoV)
    means (point), its bootstrap 95% CI (whisker), the individual per-FoV
    deviations as small jittered dots (so the reader sees how many FoV and
    their spread behind each estimate — SuperPlot ethos), and a right-hand
    annotation column with the FoV-mean Welch + Mann-Whitney p. n_FoV and
    n_nuclei actually used are pulled from the data and shown in the subtitle
    and per row. Agnostic framing: 'NT→KD change', never 'significant'."""
    # Short, horizontal metric labels (publication-tight).
    metrics = [
        ("rna_spot_count", "Spots / nucleus"),
        (_resolve_col(nuc, "rna_spot_total_peak_intensity", "rna_spot_total_intensity_fit"),
         "Total RNA / nucleus"),
        ("nuclear_spot_fraction", "Nuclear fraction"),
        ("nuclear_spot_density_per_um2", "Nuclear density"),
    ]
    missing = {m for m, _ in metrics if m not in nuc.columns} | (
        {"condition", "image"} - set(nuc.columns))
    if missing:
        _warn_missing("effect_size_summary", missing); ax.set_visible(False); return
    conds = order_conditions(nuc["condition"].dropna().unique().tolist(), condition_order or [])
    kd = next((c for c in conds if "kd" in _norm_cond_key(c)), None)
    nt = next((c for c in conds if "nt" in _norm_cond_key(c)), None)
    if kd is None or nt is None:
        # An NT→KD contrast is undefined when both conditions are not present.
        # This is EXPECTED (not a data error) in the per-image / per-condition
        # passes, which call build_layout on a single-condition subset — so
        # degrade SILENTLY here. The loud _warn_missing path above is reserved
        # for genuinely missing COLUMNS, which IS a real problem.
        ax.set_visible(False); return
    rng = np.random.RandomState(20260527)

    def _img_means(cond, col):
        sub = nuc[nuc["condition"] == cond]
        v = pd.to_numeric(sub[col], errors="coerce")
        g = sub.assign(_v=v).dropna(subset=["_v"]).groupby("image")["_v"].mean()
        return g.values

    # Real n_FoV (distinct images) and n_nuclei per condition, from the subset.
    def _n_fov(cond):
        return int(nuc.loc[nuc["condition"] == cond, "image"].dropna().nunique())

    def _n_nuc(cond):
        return int((nuc["condition"] == cond).sum())

    nt_fov, kd_fov = _n_fov(nt), _n_fov(kd)
    nt_nuc, kd_nuc = _n_nuc(nt), _n_nuc(kd)

    rows = []  # (label, pct, lo, hi, p_t, p_mw, fov_devs, n_nt_fov, n_kd_fov)
    for col, label in metrics:
        kd_m = _img_means(kd, col); nt_m = _img_means(nt, col)
        if len(kd_m) < 1 or len(nt_m) < 1:
            rows.append((label, np.nan, np.nan, np.nan, np.nan, np.nan,
                         np.array([]), len(nt_m), len(kd_m))); continue
        nt_mean = float(np.mean(nt_m))
        pct = (float(np.mean(kd_m)) - nt_mean) / nt_mean * 100.0 if nt_mean != 0 else np.nan
        # bootstrap 95% CI of the percent change (resample per-image means)
        boots = []
        for _ in range(2000):
            bn = float(np.mean(rng.choice(nt_m, size=len(nt_m), replace=True)))
            bk = float(np.mean(rng.choice(kd_m, size=len(kd_m), replace=True)))
            if bn != 0:
                boots.append((bk - bn) / bn * 100.0)
        lo, hi = (np.nanpercentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
        # FoV-mean Welch (primary) + Mann-Whitney (secondary), same as SuperPlots.
        rec = _superplot_stats({nt: list(nt_m), kd: list(kd_m)}, [nt, kd])
        p_t = rec[0]["p_t"] if rec else float("nan")
        p_mw = rec[0]["p_mw"] if rec else float("nan")
        # Per-FoV deviation dots: each KD image's mean expressed as % change vs
        # the NT grand mean — the actual points the FoV-mean stats run on.
        fov_devs = ((kd_m - nt_mean) / nt_mean * 100.0) if nt_mean != 0 else np.array([])
        rows.append((label, pct, lo, hi, p_t, p_mw, np.asarray(fov_devs), len(nt_m), len(kd_m)))

    ys = np.arange(len(rows))[::-1]
    base = _condition_family_base_color(kd, conds.index(kd) if kd in conds else 0)
    # Light family tint for the per-FoV dots so they read as supporting detail.
    try:
        _r, _g, _b, _ = _get_cmap(_cmap_for_condition(kd, 0))(0.40)
        dot_color = "#%02x%02x%02x" % (int(_r * 255), int(_g * 255), int(_b * 255))
    except Exception:
        dot_color = base

    ax.axvline(0.0, color="#555", linestyle=":", linewidth=1.0, alpha=0.7, zorder=1)

    # Determine a clean x-range from CIs + per-FoV dots, with symmetric-ish
    # headroom and a reserved right margin for the annotation column.
    span_vals = []
    for r in rows:
        for v in (r[2], r[3], r[1]):
            if v is not None and np.isfinite(v):
                span_vals.append(v)
        if len(r[6]):
            span_vals.extend([v for v in r[6] if np.isfinite(v)])
    if not span_vals:
        ax.set_visible(False); return
    lo_x, hi_x = min(span_vals + [0.0]), max(span_vals + [0.0])
    pad = max(8.0, (hi_x - lo_x) * 0.10)
    data_lo, data_hi = lo_x - pad, hi_x + pad
    # Reserve ~42% of the axis width on the right for the aligned p-value column
    # so the longest 'MWU p = 0.686' line never touches the frame.
    ann_frac = 0.42
    full_lo = data_lo
    full_hi = data_hi + (data_hi - data_lo) * (ann_frac / (1.0 - ann_frac))
    ax.set_xlim(full_lo, full_hi)
    ann_x = data_hi + (full_hi - data_hi) * 0.10  # left-aligned annotation column

    # _fmt_p returns 'p=0.004' or 'p<0.001'; normalize to a spaced 'p = ...'.
    def _p_disp(p):
        s = _fmt_p(p)
        if s.startswith("p<"):
            return "p < " + s[2:]
        if s.startswith("p="):
            return "p = " + s[2:]
        return s

    for y, r in zip(ys, rows):
        label, pct, lo, hi, p_t, p_mw, fov_devs, n_nt, n_kd = r
        if not np.isfinite(pct):
            ax.text(ann_x, y, "n/a", ha="left", va="center", fontsize=8, color="#999")
            continue
        # bootstrap 95% CI whisker
        if np.isfinite(lo) and np.isfinite(hi):
            ax.plot([lo, hi], [y, y], color=base, linewidth=2.4, alpha=0.85, zorder=2,
                    solid_capstyle="round")
            for cap in (lo, hi):
                ax.plot([cap, cap], [y - 0.07, y + 0.07], color=base, linewidth=1.6,
                        alpha=0.85, zorder=2)
        # per-FoV deviation dots (small, jittered) — shows how many FoV + spread
        if len(fov_devs):
            jy = rng.uniform(-0.13, 0.13, size=len(fov_devs))
            ax.scatter(fov_devs, np.full(len(fov_devs), y) + jy, s=22, color=dot_color,
                       edgecolor="#1f1f1f", linewidth=0.5, alpha=0.85, zorder=3)
        # point estimate (large)
        ax.scatter([pct], [y], s=150, color=base, edgecolor="#1f1f1f",
                   linewidth=1.5, zorder=5)
        # value + CI just above the point (compact, consistent formatting).
        # Clamp horizontally so the label never overhangs the plot frame: if the
        # point sits near the left edge, left-align the label from a safe x.
        ci_txt = (f"  [{_fmt_pct_signed(lo)}, {_fmt_pct_signed(hi)}]"
                  if (np.isfinite(lo) and np.isfinite(hi)) else "")
        lbl = f"{_fmt_pct_signed(pct)}{ci_txt}"
        left_guard = data_lo + (data_hi - data_lo) * 0.18
        if pct < left_guard:
            ax.text(left_guard, y + 0.22, lbl, ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold", color="#1f1f1f", zorder=6)
        else:
            ax.text(pct, y + 0.22, lbl, ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold", color="#1f1f1f", zorder=6)
        # right-hand annotation column: aligned p-values + per-row FoV n.
        ax.text(ann_x, y + 0.16,
                f"Welch {_p_disp(p_t)}   MWU {_p_disp(p_mw)}",
                ha="left", va="center", fontsize=7.8, color="#1f1f1f")
        ax.text(ann_x, y - 0.20,
                f"FoV: {_display_condition(nt)} n={n_nt}, {_display_condition(kd)} n={n_kd}",
                ha="left", va="center", fontsize=7.0, color="#666")

    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=10)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    import matplotlib.ticker as _mt
    ax.xaxis.set_major_formatter(_mt.FuncFormatter(lambda v, _p: f"{v:+.0f}%" if v else "0%"))
    ax.set_xlabel(f"{_display_condition(nt)} → {_display_condition(kd)} change in per-image mean (%)")
    ax.grid(axis="x", alpha=0.22, linestyle="--"); ax.set_axisbelow(True)
    # A thin separator at the annotation-column boundary keeps it tidy.
    ax.axvline(data_hi + pad * 0.3, color="#ddd", linewidth=0.8, zorder=0)

    # Title (3 explicit lines, NOT auto-wrapped, so the headline / method /
    # n-readout stay on their own rows): headline, then the method gloss, then
    # the real n_FoV / n_nuclei pulled from this subset.
    sub = (f"{_display_condition(nt)}→{_display_condition(kd)} FoV: n={nt_fov} / {kd_fov}"
           f"   •   nuclei: {nt_nuc} / {kd_nuc}")
    ax.set_title(
        f"{_display_condition(nt)}→{_display_condition(kd)} effect-size summary\n"
        "point = % change of per-image means · whisker = bootstrap 95% CI · "
        "dots = individual FoV · p on per-image means\n"
        f"{sub}",
        fontsize=10, linespacing=1.35)
    # First title line slightly larger / bold for hierarchy.
    try:
        ax.title.set_fontsize(10)
    except Exception:
        pass


# =====================================================================
# rna_rna-mode-specific plotting helpers
# =====================================================================
# These functions read two-channel rna_rna output:
#   - per-nucleus CSV: rna_spot_count / n_spots_rna1 (== rna_spot_count),
#     n_spots_rna2, nuclear_spot_count, nuclear_spot_count_rna2,
#     cyto_spot_count, cyto_spot_count_rna2, paired_fraction_rna1_at_*,
#     paired_fraction_rna2_at_*, paired_spot_count_rna*_at_*,
#     median_nn_distance_rna1_um, median_nn_distance_rna2_um,
#     cell_total_intensity_rna1, cell_total_intensity_rna2
#   - per-spot CSV: includes a `channel` column ('rna1' / 'rna2') +
#     `nn_distance_um` + `paired_at_*` flag
# All plot functions degrade gracefully when columns are missing.

# Fixed colors for the two RNA channels in figures, matched to the
# pub-image LUTs (RNA1 = yellow, RNA2 = cyan) for visual consistency.
COLOR_RNA1 = OKABE_ITO[3]   # yellow (matches publication-image LUT)
COLOR_RNA2 = OKABE_ITO[1]   # sky blue (close to cyan / pub LUT)


def _find_pair_suffix(df: pd.DataFrame) -> str | None:
    """Look at column names like 'paired_fraction_rna1_at_0p3um' and
    return the suffix '0p3um'. Returns None if no such column found."""
    if df is None or len(df) == 0:
        return None
    prefix = "paired_fraction_rna1_at_"
    for c in df.columns:
        if isinstance(c, str) and c.startswith(prefix):
            return c[len(prefix):]
    return None


def plot_spots_per_nucleus_channel(ax, nuc: pd.DataFrame, channel: str) -> None:
    """Spots-per-nucleus distribution for one RNA channel.

    2026-05-22 Brian: colors NOW encode CONDITION (not channel). Channel
    identity lives in the title only. WT and KO get distinguishable colors
    so the histogram is readable. Sec-only stays gray.
    """
    col = "rna_spot_count" if channel == "rna1" else "n_spots_rna2"
    if col not in nuc.columns or "image" not in nuc.columns:
        ax.set_visible(False); return
    counts = pd.to_numeric(nuc[col], errors="coerce").dropna()
    if counts.empty:
        ax.set_visible(False); return
    max_count = int(counts.max() if len(counts) else 1)
    bins = np.arange(0, max(max_count + 2, 20), 1) if max_count <= 50 else 30
    # Aggregate by condition so one legend entry per condition (avoids
    # 10+ legend entries on multi-image runs). Sec-only stays separate.
    if "condition" not in nuc.columns:
        ax.set_visible(False); return
    is_sec = nuc["secondary_only"].astype(bool) if "secondary_only" in nuc.columns else pd.Series([False]*len(nuc), index=nuc.index)
    plot_groups = []  # list of (label, color, values)
    for cond in nuc["condition"].dropna().unique().tolist():
        if cond is None: continue
        real_mask = (nuc["condition"] == cond) & (~is_sec)
        sec_mask  = (nuc["condition"] == cond) & ( is_sec)
        if real_mask.any():
            vals = pd.to_numeric(nuc.loc[real_mask, col], errors="coerce").dropna()
            if len(vals):
                plot_groups.append((f"{cond} (n={int(real_mask.sum())})",
                                    _color_for_condition(cond, 0), vals))
        if sec_mask.any():
            vals = pd.to_numeric(nuc.loc[sec_mask, col], errors="coerce").dropna()
            if len(vals):
                plot_groups.append((f"sec-only {cond} (n={int(sec_mask.sum())})",
                                    COLOR_SEC_ONLY, vals))
    for label, color, vals in plot_groups:
        ax.hist(vals, bins=bins, alpha=0.55, label=label, color=color, edgecolor="black", linewidth=0.5)
    ax.set_xlabel(f"{channel.upper()} spots per nucleus")
    ax.set_ylabel("Number of nuclei")
    ax.set_title(_wrap_title(f"Spots-per-nucleus distribution — {channel.upper()}"))
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)


def plot_mean_spots_per_image_channel(ax, summary: pd.DataFrame | None,
                                       nuc: pd.DataFrame, channel: str) -> None:
    """Bar chart of mean-spots-per-nucleus, AGGREGATED PER CONDITION.

    2026-05-22 Brian: was per-image with channel color (yellow for both
    WT and KO — unreadable). Now: one bar per condition, colored by
    condition. Sec-only stays gray. Channel identity lives in title.
    """
    pre = "mean_spots_per_nucleus_rna1" if channel == "rna1" else "mean_spots_per_nucleus_rna2"
    col = "rna_spot_count" if channel == "rna1" else "n_spots_rna2"
    if "condition" not in nuc.columns or col not in nuc.columns:
        ax.set_visible(False); return
    is_sec = nuc["secondary_only"].astype(bool) if "secondary_only" in nuc.columns else pd.Series([False]*len(nuc), index=nuc.index)
    rows = []  # (label, color, mean)
    for cond in nuc["condition"].dropna().unique().tolist():
        if cond is None: continue
        real_mask = (nuc["condition"] == cond) & (~is_sec)
        sec_mask  = (nuc["condition"] == cond) & ( is_sec)
        if real_mask.any():
            vals = pd.to_numeric(nuc.loc[real_mask, col], errors="coerce").dropna()
            if len(vals):
                rows.append((cond, _color_for_condition(cond, 0), float(vals.mean())))
        if sec_mask.any():
            vals = pd.to_numeric(nuc.loc[sec_mask, col], errors="coerce").dropna()
            if len(vals):
                rows.append((f"sec-only {cond}", COLOR_SEC_ONLY, float(vals.mean())))
    if not rows:
        ax.set_visible(False); return
    labels = [r[0] for r in rows]
    colors = [r[1] for r in rows]
    means = [r[2] for r in rows]
    ax.bar(labels, means, color=colors, edgecolor="black", linewidth=0.8)
    for i, m in enumerate(means):
        ax.text(i, m, f"{m:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel(f"Mean {channel.upper()} spots / nucleus")
    ax.set_title(_wrap_title(f"Mean spots per nucleus — {channel.upper()}"))
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, alpha=0.3, axis="y")


def plot_cumulative_spots_both_channels(ax, nuc: pd.DataFrame) -> None:
    """Cumulative spots-per-cell, BOTH channels overlaid per image."""
    if "rna_spot_count" not in nuc.columns or "n_spots_rna2" not in nuc.columns:
        ax.set_visible(False); return
    images = nuc["image"].unique()
    labels = _build_image_labels(nuc)
    plotted = 0
    for img_name in images:
        sub = nuc[nuc["image"] == img_name]
        for col, color, ch in ((
            "rna_spot_count", COLOR_RNA1, "RNA1"),
            ("n_spots_rna2", COLOR_RNA2, "RNA2"),
        ):
            vals = pd.to_numeric(sub[col], errors="coerce").dropna().sort_values()
            if vals.empty: continue
            x = vals.values
            y = np.arange(1, len(x) + 1) / len(x)
            ax.plot(x, y, marker=".", linestyle="-", alpha=0.7, color=color,
                    label=f"{labels[img_name]} {ch} (n={len(x)})")
            plotted += 1
    if plotted == 0:
        ax.set_visible(False); return
    ax.set_xlabel("Spots per nucleus")
    ax.set_ylabel("Cumulative fraction of nuclei")
    ax.set_title(_wrap_title("CDF: spots per nucleus — both channels"))
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.02)


def plot_nuclear_vs_cytoplasmic_channel(ax, nuc: pd.DataFrame, channel: str) -> None:
    """Stacked bar per image: nuclear vs cytoplasmic spots for one channel.

    Reads aggregated counts from nuc (nuclear_spot_count / cyto_spot_count
    for RNA1, *_rna2 for RNA2) rather than re-summing per-spot rows — this
    is robust to the rna_rna per-spot file having a 'channel' column."""
    if channel == "rna1":
        nuc_col, cyt_col = "nuclear_spot_count", "cyto_spot_count"
    else:
        nuc_col, cyt_col = "nuclear_spot_count_rna2", "cyto_spot_count_rna2"
    if nuc_col not in nuc.columns or cyt_col not in nuc.columns:
        ax.set_visible(False); return
    by_img = nuc.groupby("image").agg(
        nuclear=(nuc_col, lambda s: int(pd.to_numeric(s, errors="coerce").fillna(0).sum())),
        cytoplasmic=(cyt_col, lambda s: int(pd.to_numeric(s, errors="coerce").fillna(0).sum())),
    ).sort_index()
    label_map = _build_image_labels(nuc)
    labels = [label_map.get(i, short_label(i)) for i in by_img.index]
    x = np.arange(len(labels))
    ax.bar(x, by_img["nuclear"], label="nuclear", color=COLOR_NUCLEAR,
           edgecolor="black", linewidth=0.5)
    ax.bar(x, by_img["cytoplasmic"], bottom=by_img["nuclear"],
           label="cytoplasmic", color=COLOR_CYTOPLASMIC,
           edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("Total spot count")
    ax.set_title(_wrap_title(f"Nuclear vs cytoplasmic spots — {channel.upper()}"))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")


def plot_paired_fraction_per_condition(ax, summary: pd.DataFrame | None,
                                         condition_order: list[str] | None = None) -> None:
    """Bar plot of paired fraction at the configured pairing distance,
    grouped by condition, both channels side-by-side. One bar per
    (condition, channel) where height = mean across images and error
    bar = SEM."""
    if summary is None or len(summary) == 0:
        ax.set_visible(False); return
    suffix = _find_pair_suffix(summary)
    if suffix is None:
        ax.set_visible(False); return
    col1 = f"paired_fraction_rna1_at_{suffix}"
    col2 = f"paired_fraction_rna2_at_{suffix}"
    if col1 not in summary.columns or col2 not in summary.columns:
        ax.set_visible(False); return
    df = summary.copy()
    df[col1] = pd.to_numeric(df[col1], errors="coerce")
    df[col2] = pd.to_numeric(df[col2], errors="coerce")
    conds_in_data = df["condition"].dropna().unique().tolist() if "condition" in df.columns else []
    conditions = order_conditions(conds_in_data, condition_order or []) if conds_in_data else ["(all)"]
    means1, sems1, means2, sems2 = [], [], [], []
    for c in conditions:
        sub = df if c == "(all)" else df[df["condition"] == c]
        v1 = sub[col1].dropna().values
        v2 = sub[col2].dropna().values
        means1.append(float(np.mean(v1)) if len(v1) else 0.0)
        sems1.append(float(np.std(v1, ddof=1) / max(np.sqrt(len(v1)), 1)) if len(v1) > 1 else 0.0)
        means2.append(float(np.mean(v2)) if len(v2) else 0.0)
        sems2.append(float(np.std(v2, ddof=1) / max(np.sqrt(len(v2)), 1)) if len(v2) > 1 else 0.0)
    x = np.arange(len(conditions))
    width = 0.38
    ax.bar(x - width / 2, means1, width, yerr=sems1, color=COLOR_RNA1,
           edgecolor="black", linewidth=0.5, capsize=4,
           label=f"RNA1 overlap @ {suffix} (RNA1↔RNA2)")
    ax.bar(x + width / 2, means2, width, yerr=sems2, color=COLOR_RNA2,
           edgecolor="black", linewidth=0.5, capsize=4,
           label=f"RNA2 overlap @ {suffix} (RNA1↔RNA2)")
    ax.set_xticks(x); ax.set_xticklabels(conditions, rotation=15)
    # 2026-05-18 Brian: "paired" wording was opaque (TODO: pull suffix
    # distance from cfg when accessible — currently hard-coded to the
    # 0.3 µm default per the spot_coloc.pair_distance_um config).
    ax.set_ylabel(
        f"Overlap fraction (RNA1↔RNA2 within {suffix.replace('p', '.')[:-2]} µm)"
    )
    ax.set_title(_wrap_title(f"Spot–spot overlap fraction — by condition\n"
                 f"(error bars = SEM across images)"))
    ax.set_ylim(0, max(0.05, max(means1 + means2) * 1.3))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")


def plot_nn_distance_distribution(ax, spots: pd.DataFrame) -> None:
    """Histogram of per-spot nearest-neighbor (partner-channel) distances,
    both RNA channels overlaid. Caps at the 99th percentile of finite
    values to avoid the long tail of unpaired spots (which have +inf)."""
    if spots is None or len(spots) == 0 or "nn_distance_um" not in spots.columns:
        ax.set_visible(False); return
    if "channel" not in spots.columns:
        ax.set_visible(False); return
    d = pd.to_numeric(spots["nn_distance_um"], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    if d.empty:
        ax.set_visible(False); return
    vmax = float(d.quantile(0.99))
    bins = np.linspace(0, max(vmax, 0.1), 50)
    for chan, color in (("rna1", COLOR_RNA1), ("rna2", COLOR_RNA2)):
        sub = spots[spots["channel"] == chan]
        vals = pd.to_numeric(sub["nn_distance_um"], errors="coerce")
        vals = vals.replace([np.inf, -np.inf], np.nan).dropna()
        if vals.empty: continue
        ax.hist(vals, bins=bins, alpha=0.55, color=color,
                label=f"{chan.upper()} (n={len(vals)})")
        ax.axvline(float(vals.median()), color=color, linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("NN distance to partner channel (µm)")
    ax.set_ylabel("Spot count")
    ax.set_title(_wrap_title("Nearest-neighbor distance distribution\n(dashed = median; x-axis to p99)"))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_rna1_vs_rna2_per_cell_scatter(ax, nuc: pd.DataFrame,
                                         condition_order: list[str] | None = None) -> None:
    """Scatter of total spots per cell: RNA1 (x) vs RNA2 (y), colored by
    condition, with an overall OLS regression line."""
    if "rna_spot_count" not in nuc.columns or "n_spots_rna2" not in nuc.columns:
        ax.set_visible(False); return
    df = nuc.copy()
    df["_x"] = pd.to_numeric(df["rna_spot_count"], errors="coerce")
    df["_y"] = pd.to_numeric(df["n_spots_rna2"], errors="coerce")
    df = df[df["_x"].notna() & df["_y"].notna()]
    if df.empty:
        ax.set_visible(False); return
    conds = sorted(df["condition"].dropna().unique()) if "condition" in df.columns else []
    conds = order_conditions(conds, condition_order or []) if conds else []
    plotted = 0
    for i, cond in enumerate(conds):
        sub = df[df["condition"] == cond]
        if sub.empty: continue
        ax.scatter(sub["_x"], sub["_y"], s=18, alpha=0.65,
                   color=_color_for_condition(cond, i),
                   edgecolor="white", linewidth=0.4,
                   label=f"{cond} (n={len(sub)})")
        plotted += 1
    if plotted == 0:
        ax.scatter(df["_x"], df["_y"], s=18, alpha=0.55)
    # OLS regression line on the pooled data
    x = df["_x"].values.astype(float)
    y = df["_y"].values.astype(float)
    if len(x) >= 2 and float(np.std(x)) > 0:
        m_, b_ = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, m_ * xs + b_, color="black", linestyle="--", linewidth=1.2,
                label=f"OLS: y = {m_:.2f}x + {b_:.2f}")
        # Pearson r
        try:
            r_ = float(np.corrcoef(x, y)[0, 1])
            ax.text(0.02, 0.98, f"Pearson r = {r_:.2f}", transform=ax.transAxes,
                    ha="left", va="top", fontsize=9,
                    bbox=dict(facecolor="white", edgecolor="gray", alpha=0.85, boxstyle="round"))
        except Exception:
            pass
    ax.set_xlabel("RNA1 spots per cell")
    ax.set_ylabel("RNA2 spots per cell")
    ax.set_title(_wrap_title("RNA1 vs RNA2 spots per cell\n(each dot = one nucleus)"))
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)


def plot_paired_spots_per_nucleus_distribution(ax, nuc: pd.DataFrame) -> None:
    """Histogram of paired-spot count per nucleus, both channels."""
    suffix = _find_pair_suffix(nuc)
    if suffix is None:
        ax.set_visible(False); return
    c1 = f"paired_spot_count_rna1_at_{suffix}"
    c2 = f"paired_spot_count_rna2_at_{suffix}"
    if c1 not in nuc.columns and c2 not in nuc.columns:
        ax.set_visible(False); return
    plotted = 0
    all_max = 0
    for col, color, ch in ((c1, COLOR_RNA1, "RNA1"), (c2, COLOR_RNA2, "RNA2")):
        if col not in nuc.columns: continue
        vals = pd.to_numeric(nuc[col], errors="coerce").dropna()
        if vals.empty: continue
        all_max = max(all_max, int(vals.max()))
    if all_max == 0:
        ax.set_visible(False); return
    bins = np.arange(0, max(all_max + 2, 5), 1)
    for col, color, ch in ((c1, COLOR_RNA1, "RNA1"), (c2, COLOR_RNA2, "RNA2")):
        if col not in nuc.columns: continue
        vals = pd.to_numeric(nuc[col], errors="coerce").dropna()
        if vals.empty: continue
        ax.hist(vals, bins=bins, alpha=0.6, color=color,
                label=f"{ch} (n={len(vals)}, mean={vals.mean():.2f})")
        plotted += 1
    if plotted == 0:
        ax.set_visible(False); return
    ax.set_xlabel(
        f"Overlapping spots per nucleus (RNA1↔RNA2 within "
        f"{suffix.replace('p', '.')[:-2]} µm)"
    )
    ax.set_ylabel("Number of nuclei")
    ax.set_title(_wrap_title("Overlapping spot count per nucleus\n(RNA1↔RNA2 within 0.3 µm)"))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_box_paired_fraction_by_condition(ax, nuc: pd.DataFrame,
                                            channel: str,
                                            condition_order: list[str] | None = None) -> None:
    """Per-nucleus paired-fraction by condition, box+strip+image-mean."""
    suffix = _find_pair_suffix(nuc)
    if suffix is None:
        ax.set_visible(False); return
    col = f"paired_fraction_rna1_at_{suffix}" if channel == "rna1" else f"paired_fraction_rna2_at_{suffix}"
    if col not in nuc.columns:
        ax.set_visible(False); return
    if not _box_strip_with_image_means(ax, nuc, col, only_expressing=False,
                                        condition_order=condition_order):
        ax.set_visible(False); return
    ax.set_ylabel(
        f"Overlap fraction — {channel.upper()} (RNA1↔RNA2 within 0.3 µm)"
    )
    ax.set_title(_wrap_title(
        f"Overlap fraction per nucleus — {channel.upper()}\n"
        f"(RNA1↔RNA2 within 0.3 µm; ◇ = per-image mean, • = per cell)"
    ))
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=8, loc="upper right")


def plot_box_cell_total_intensity_by_channel(ax, nuc: pd.DataFrame,
                                               channel: str,
                                               condition_order: list[str] | None = None) -> None:
    """Per-cell total RNA SPOT intensity by condition, one channel at a time.

    2026-05-18 Brian: previously used ``cell_total_intensity_rna{1,2}``,
    which sums every cell-mask pixel × channel value — so secondary-only
    (no-primary-FISH-probe) cells reported MILLIONS of intensity units of
    pure autofluorescence + camera offset, and Brian was reading that as
    "high RNA signal in sec-only" in figures.

    Fix: use ``rna_spot_total_peak_intensity`` / ``rna2_spot_total_peak_intensity``
    — the sum of *only* the detected spots' peak-pixel intensities. Sec-only
    cells correctly read 0 here (no spots detected → no spot intensity to
    sum). Falls back to the legacy contaminated column with an explicit
    autofluorescence subtitle if the spot-only column is missing in older
    runs."""
    if channel == "rna1":
        spot_col = _resolve_col(nuc, "rna_spot_total_peak_intensity", "rna_spot_total_intensity_fit")
    else:
        spot_col = _resolve_col(nuc, "rna2_spot_total_peak_intensity", "rna2_spot_total_intensity_fit")
    legacy_col = "cell_total_intensity_rna1" if channel == "rna1" else "cell_total_intensity_rna2"
    used_spot = spot_col in nuc.columns
    col = spot_col if used_spot else legacy_col
    if col not in nuc.columns:
        ax.set_visible(False); return
    if not _box_strip_with_image_means(ax, nuc, col, only_expressing=False,
                                        condition_order=condition_order):
        ax.set_visible(False); return
    valid = pd.to_numeric(nuc[col], errors="coerce").dropna()
    if not valid.empty and (valid.max() / max(valid[valid > 0].min() if (valid > 0).any() else 1, 1)) > 100:
        ax.set_yscale("log")
    if used_spot:
        ax.set_ylabel(f"Per-cell total {channel.upper()} SPOT intensity\n(sum of detected spot intensities — autofluorescence excluded)")
        ax.set_title(_wrap_title(f"Per-cell total {channel.upper()} SPOT intensity — by condition\n(◇ = per-image mean, • = per cell; sec-only = 0 by definition)"))
    else:
        ax.set_ylabel(f"Per-cell total {channel.upper()} intensity\n(integrated cell fluorescence — includes autofluorescence baseline)")
        ax.set_title(_wrap_title(f"Per-cell total {channel.upper()} intensity — by condition\n(◇ = per-image mean, • = per cell; includes background)"))
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=8, loc="upper right")


# ---------------------------------------------------------------------------
# Expanded rna_rna analysis helpers (2026-05-14) — biology-focused
# ---------------------------------------------------------------------------
# These helpers cover the per-channel breakdowns (intensity / size / SNR per
# channel), nuclear-vs-cytoplasmic stratification by channel, the exon/intron
# biology proxies (active TS, mature mRNA, burst size, TSS efficiency,
# nascent-to-mature ratio), and the cross-condition difference figures
# (quadrant scatter, anti-correlation flag, exclusive expression, volcano-
# like, effect-size bars).
#
# Every plot degrades gracefully when required columns are absent — the
# downstream loop in main() catches Exceptions per panel, but
# set_visible(False) is preferred so empty panels disappear cleanly.


def _channel_filter(spots: pd.DataFrame, channel: str) -> pd.DataFrame:
    """Return spots subset for one channel ('rna1' or 'rna2') if a 'channel'
    column is present, else returns the full df (rna_only-style data).
    """
    if "channel" not in spots.columns:
        return spots
    return spots[spots["channel"] == channel].copy()


def _hist_by_condition(
    ax, df: pd.DataFrame, value_col: str, bins,
    xlabel: str, ylabel: str, title: str,
    filter_positive: bool = False,
) -> None:
    """Helper: aggregate `value_col` by CONDITION (not image/channel) and
    draw overlaid histograms. WT/KO get distinct condition colors; sec-only
    gets gray. One legend entry per (condition, sec-only flag) group.
    """
    if "condition" not in df.columns or value_col not in df.columns:
        ax.set_visible(False); return
    is_sec = df["secondary_only"].astype(bool) if "secondary_only" in df.columns else pd.Series([False]*len(df), index=df.index)
    plot_groups = []
    for cond in df["condition"].dropna().unique().tolist():
        if cond is None: continue
        real_mask = (df["condition"] == cond) & (~is_sec)
        sec_mask  = (df["condition"] == cond) & ( is_sec)
        if real_mask.any():
            vals = pd.to_numeric(df.loc[real_mask, value_col], errors="coerce").dropna()
            if filter_positive: vals = vals[vals > 0]
            if len(vals):
                plot_groups.append((f"{cond} (n={len(vals)})",
                                    _color_for_condition(cond, 0), vals))
        if sec_mask.any():
            vals = pd.to_numeric(df.loc[sec_mask, value_col], errors="coerce").dropna()
            if filter_positive: vals = vals[vals > 0]
            if len(vals):
                plot_groups.append((f"sec-only {cond} (n={len(vals)})",
                                    COLOR_SEC_ONLY, vals))
    if not plot_groups:
        ax.set_visible(False); return
    for label, color, vals in plot_groups:
        ax.hist(vals, bins=bins, alpha=0.55, label=label,
                color=color, edgecolor="black", linewidth=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(_wrap_title(title))
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)


def plot_spot_peak_intensity_channel(ax, spots: pd.DataFrame, channel: str) -> None:
    """Per-condition spot peak-intensity histogram, ONE channel only."""
    sub = _channel_filter(spots, channel)
    if sub.empty or "spot_peak_intensity" not in sub.columns:
        ax.set_visible(False); return
    vals_all = sub["spot_peak_intensity"].dropna()
    if vals_all.empty:
        ax.set_visible(False); return
    bins = np.linspace(0, float(vals_all.quantile(0.99)) * 1.05, 40)
    _hist_by_condition(ax, sub, "spot_peak_intensity", bins,
                       xlabel=f"{channel.upper()} spot peak intensity",
                       ylabel="Spot count",
                       title=f"Spot peak-intensity distribution — {channel.upper()}")


def plot_spot_size_channel(ax, spots: pd.DataFrame, channel: str) -> None:
    """Per-condition spot diameter (µm) distribution."""
    sub = _channel_filter(spots, channel)
    if sub.empty or "spot_diameter_um" not in sub.columns:
        ax.set_visible(False); return
    vals_all = sub["spot_diameter_um"].dropna()
    vals_all = vals_all[vals_all > 0]
    if vals_all.empty:
        ax.set_visible(False); return
    bins = np.linspace(0, float(vals_all.max()), 40)
    _hist_by_condition(ax, sub, "spot_diameter_um", bins,
                       xlabel=f"{channel.upper()} spot diameter (µm)",
                       ylabel="Spot count",
                       title=f"Spot size distribution — {channel.upper()}",
                       filter_positive=True)


def plot_local_snr_channel(ax, spots: pd.DataFrame, channel: str) -> None:
    """Per-condition local SNR histogram. Useful for spot-quality QC."""
    sub = _channel_filter(spots, channel)
    if sub.empty or "local_snr" not in sub.columns:
        ax.set_visible(False); return
    vals_all = sub["local_snr"].dropna()
    if vals_all.empty:
        ax.set_visible(False); return
    vmax = float(vals_all.quantile(0.99)) * 1.1
    bins = np.linspace(0, max(vmax, 1), 40)
    _hist_by_condition(ax, sub, "local_snr", bins,
                       xlabel="Local SNR", ylabel="Spot count",
                       title=f"Per-spot local SNR — {channel.upper()}\n(dotted = SNR=5)")
    ax.axvline(5, color="black", linestyle=":", linewidth=0.8, alpha=0.5)


def plot_sorted_brightness_channel(ax, spots: pd.DataFrame, channel: str) -> None:
    """Sorted-brightness rank curve per channel (log-log).

    Per-image traces colored by their CONDITION (WT blue, KO orange,
    sec-only gray) — was channel color which collapsed WT/KO to one color.
    """
    sub = _channel_filter(spots, channel)
    col = _resolve_col(sub, "peak_intensity", "integrated_intensity_fit", "spot_peak_intensity")
    if sub.empty or col not in sub.columns or "condition" not in sub.columns:
        ax.set_visible(False); return
    plotted = 0
    labels_map = _build_image_labels(sub)
    family_map = _build_family_color_map(sub)
    for img_name in sub["image"].unique():
        s = sub[sub["image"] == img_name]
        if not len(s): continue
        cond = s["condition"].iloc[0]
        sec = bool(s["secondary_only"].iloc[0]) if "secondary_only" in s.columns else False
        color = _image_family_color(family_map, img_name) or (
            COLOR_SEC_ONLY if sec else _color_for_condition(cond, 0))
        vals = s[col].dropna()
        vals = vals[vals > 0]
        if vals.empty:
            continue
        sv = np.sort(vals.values)[::-1]
        x = np.arange(1, len(sv) + 1)
        ax.plot(x, sv, alpha=0.7, color=color,
                label=f"{labels_map.get(img_name, '?')} [{cond}{' sec' if sec else ''}] (n={len(sv)})")
        plotted += 1
    if plotted == 0:
        ax.set_visible(False); return
    ax.set_xlabel("Spot rank (1 = brightest)")
    ax.set_ylabel(f"{channel.upper()} peak intensity")
    ax.set_title(_wrap_title(f"Sorted spot-brightness curve — {channel.upper()}"))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3, which="both")


def plot_spots_vs_area_channel(ax, nuc: pd.DataFrame, channel: str,
                                 condition_order: list[str] | None = None) -> None:
    """Per-channel scatter of spots-per-cell vs nucleus area, colored by condition."""
    col = "rna_spot_count" if channel == "rna1" else "n_spots_rna2"
    if col not in nuc.columns or "nucleus_area_px" not in nuc.columns:
        ax.set_visible(False); return
    df = nuc.copy()
    conds = sorted(df["condition"].dropna().unique()) if "condition" in df.columns else []
    conds = order_conditions(conds, condition_order or []) if conds else []
    for i, cond in enumerate(conds):
        sub = df[df["condition"] == cond]
        if sub.empty: continue
        ax.scatter(sub["nucleus_area_px"], sub[col], s=18, alpha=0.65,
                   color=_color_for_condition(cond, i),
                   edgecolor="white", linewidth=0.4,
                   label=f"{cond} (n={len(sub)})")
    if not conds:
        ax.scatter(df["nucleus_area_px"], df[col], s=18, alpha=0.55)
    ax.set_xlabel("Nucleus area (px)")
    ax.set_ylabel(f"{channel.upper()} spots per cell")
    ax.set_title(_wrap_title(f"Spots per cell vs nucleus area — {channel.upper()}"))
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)


def plot_box_per_cell_expression_channel(ax, nuc: pd.DataFrame, channel: str,
                                           condition_order: list[str] | None = None) -> None:
    """Box plot of per-cell mean spot intensity (expressers only) by condition,
    one channel at a time. Uses *_spot_mean_peak_intensity columns."""
    if channel == "rna1":
        col = _resolve_col(nuc, "rna_spot_mean_peak_intensity", "rna_spot_mean_intensity_fit")
    else:
        col = _resolve_col(nuc, "rna2_spot_mean_peak_intensity", "rna2_spot_mean_intensity_fit")
    cnt_col = "rna_spot_count" if channel == "rna1" else "n_spots_rna2"
    if col not in nuc.columns or cnt_col not in nuc.columns:
        ax.set_visible(False); return
    df = nuc.copy()
    df["_filtered_intensity"] = pd.to_numeric(df[col], errors="coerce")
    df["rna_spot_count"] = pd.to_numeric(df[cnt_col], errors="coerce")
    if not _box_strip_with_image_means(ax, df, "_filtered_intensity", only_expressing=True,
                                        condition_order=condition_order):
        ax.set_visible(False); return
    ax.set_ylabel(f"Mean {channel.upper()} spot intensity\n(per cell, expressers only)")
    ax.set_title(_wrap_title(f"Per-cell expression intensity — {channel.upper()} by condition"))
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=8, loc="upper right")


def plot_box_nc_ratio_total_intensity(ax, nuc: pd.DataFrame, channel: str,
                                        condition_order: list[str] | None = None,
                                        spots: pd.DataFrame | None = None) -> None:
    """Box plot of N/C ratio of RNA SPOT intensity by condition.

    2026-05-18 Brian: previously used ``nc_ratio_total_intensity_rna{1,2}``,
    which is the ratio of nuclear- vs cytoplasmic-pixel sums — both
    contaminated by autofluorescence and camera offset, so sec-only cells
    landed on the same numeric N/C ratio as real signal (the ratio of two
    background numbers).

    Fix: re-derive a SPOT-based N/C ratio from ``spot_metrics`` when
    available — sum the per-spot PEAK-pixel intensity per nucleus split by
    ``in_nucleus``, then divide. Sec-only cells have no detected spots →
    NaN N/C → excluded from the box (correct: there is no spot signal to
    compute a ratio from). Falls back to the legacy pixel-sum column with
    an explicit subtitle if spot_metrics isn't provided."""
    spot_ratio_col = "_nc_ratio_spot_intensity"
    df = nuc.copy()
    used_spot = False
    pk_col = _resolve_col(spots, "peak_intensity", "integrated_intensity_fit", "spot_peak_intensity") \
        if spots is not None else "peak_intensity"
    if spots is not None and not spots.empty \
            and {"image", "nucleus_id", "channel", "in_nucleus", pk_col}.issubset(spots.columns):
        sp = spots[spots["channel"] == channel].copy()
        sp = sp[pd.to_numeric(sp["nucleus_id"], errors="coerce").notna()]
        sp[pk_col] = pd.to_numeric(sp[pk_col], errors="coerce")
        sp = sp[sp[pk_col].notna()]
        if not sp.empty:
            nuc_sum = (sp[sp["in_nucleus"].astype(bool)]
                       .groupby(["image", "nucleus_id"])[pk_col]
                       .sum().rename("_nuc_spot_int"))
            cyt_sum = (sp[~sp["in_nucleus"].astype(bool)]
                       .groupby(["image", "nucleus_id"])[pk_col]
                       .sum().rename("_cyt_spot_int"))
            joined = pd.concat([nuc_sum, cyt_sum], axis=1).reset_index()
            joined["_nuc_spot_int"] = joined["_nuc_spot_int"].fillna(0.0)
            joined["_cyt_spot_int"] = joined["_cyt_spot_int"].fillna(0.0)
            joined[spot_ratio_col] = joined["_nuc_spot_int"] / joined["_cyt_spot_int"].replace(0.0, np.nan)
            # Cells with zero cyto spot intensity but >0 nuclear are NaN'd (would be +inf)
            # to keep the box well-defined; those show up in the spot-count N/C plot anyway.
            df["nucleus_id"] = pd.to_numeric(df.get("nucleus_id"), errors="coerce")
            df = df.merge(joined[["image", "nucleus_id", spot_ratio_col]],
                          on=["image", "nucleus_id"], how="left")
            used_spot = True
    if used_spot:
        col = spot_ratio_col
    else:
        col = "nc_ratio_total_intensity_rna1" if channel == "rna1" else "nc_ratio_total_intensity_rna2"
        if col not in df.columns:
            ax.set_visible(False); return
    if not _box_strip_with_image_means(ax, df, col, only_expressing=False,
                                        condition_order=condition_order):
        ax.set_visible(False); return
    if used_spot:
        # 2026-05-18 Brian: emphasize "(spot intensity only)" in the label
        # so it's unambiguous that the autofluorescence baseline isn't
        # included — without this readers compare against the legacy
        # pixel-sum N/C ratio by reflex.
        ax.set_ylabel(f"N/C ratio of {channel.upper()} SPOT intensity\n(spot intensity only — autofluorescence baseline excluded)")
        ax.set_title(_wrap_title(f"N/C ratio of SPOT intensity — {channel.upper()}\n(per cell; >1 = nuclear-enriched; spot intensity only; sec-only excluded — no spots)"))
    else:
        ax.set_ylabel(f"N/C ratio of total {channel.upper()} intensity\n(integrated pixel sums — includes autofluorescence)")
        ax.set_title(_wrap_title(f"N/C ratio of total intensity — {channel.upper()}\n(per cell; >1 = nuclear-enriched; includes background)"))
    ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=8, loc="upper right")


def plot_box_nc_spot_count(ax, nuc: pd.DataFrame, channel: str,
                            condition_order: list[str] | None = None) -> None:
    """Box of (nuclear / cytoplasmic) spot count ratio per cell, per channel.
    +inf cells (no cyto spots) are clipped to a sentinel = 99 for plotting
    so the box doesn't collapse to one value."""
    if channel == "rna1":
        nuc_col, cyt_col = "nuclear_spot_count", "cyto_spot_count"
    else:
        nuc_col, cyt_col = "nuclear_spot_count_rna2", "cyto_spot_count_rna2"
    if nuc_col not in nuc.columns or cyt_col not in nuc.columns:
        ax.set_visible(False); return
    df = nuc.copy()
    nv = pd.to_numeric(df[nuc_col], errors="coerce")
    cv = pd.to_numeric(df[cyt_col], errors="coerce")
    df["_nc_spot_ratio"] = nv / cv.replace(0, np.nan)
    # Cells with no cyto spots but >0 nuc spots get a high sentinel
    df.loc[(cv == 0) & (nv > 0), "_nc_spot_ratio"] = np.nan  # excluded from box
    if not _box_strip_with_image_means(ax, df, "_nc_spot_ratio", only_expressing=False,
                                        condition_order=condition_order):
        ax.set_visible(False); return
    ax.set_ylabel(f"N/C spot count ratio — {channel.upper()}")
    ax.set_title(_wrap_title(f"N/C spot-count ratio — {channel.upper()}\n(per cell; cells with 0 cyto spots excluded)"))
    ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=8, loc="upper right")


def plot_nuclear_spots_distribution(ax, nuc: pd.DataFrame, channel: str) -> None:
    """Per-condition histogram of NUCLEAR spots-per-nucleus."""
    col = "nuclear_spot_count" if channel == "rna1" else "nuclear_spot_count_rna2"
    if col not in nuc.columns:
        ax.set_visible(False); return
    vals_all = pd.to_numeric(nuc[col], errors="coerce").dropna()
    if vals_all.empty:
        ax.set_visible(False); return
    max_count = int(vals_all.max() if len(vals_all) else 1)
    bins = np.arange(0, max(max_count + 2, 10), 1) if max_count <= 50 else 30
    _hist_by_condition(ax, nuc, col, bins,
                       xlabel=f"Nuclear {channel.upper()} spots per nucleus",
                       ylabel="Number of nuclei",
                       title=f"Nuclear spots per nucleus — {channel.upper()}")


def plot_cytoplasmic_spots_distribution(ax, nuc: pd.DataFrame, channel: str) -> None:
    """Per-condition histogram of CYTOPLASMIC spots-per-cell."""
    col = "cyto_spot_count" if channel == "rna1" else "cyto_spot_count_rna2"
    if col not in nuc.columns:
        ax.set_visible(False); return
    vals_all = pd.to_numeric(nuc[col], errors="coerce").dropna()
    if vals_all.empty or vals_all.sum() == 0:
        ax.set_visible(False); return
    max_count = int(vals_all.max() if len(vals_all) else 1)
    bins = np.arange(0, max(max_count + 2, 10), 1) if max_count <= 50 else 30
    _hist_by_condition(ax, nuc, col, bins,
                       xlabel=f"Cytoplasmic {channel.upper()} spots per cell",
                       ylabel="Number of cells",
                       title=f"Cytoplasmic spots per cell — {channel.upper()}")


def plot_predominantly_nuclear_fraction(ax, nuc: pd.DataFrame,
                                          condition_order: list[str] | None = None) -> None:
    """Bar chart per channel: fraction of cells where nuclear_spot_fraction > 0.5,
    grouped by condition (predominantly-nuclear localization frequency)."""
    bars_data = []
    for channel, nfrac_col, label_, color_ in (
        ("rna1", "nuclear_spot_fraction", "RNA1", COLOR_RNA1),
        ("rna2", "nuclear_spot_fraction_rna2", "RNA2", COLOR_RNA2),
    ):
        if nfrac_col not in nuc.columns:
            continue
        df = nuc.copy()
        df[nfrac_col] = pd.to_numeric(df[nfrac_col], errors="coerce")
        df = df[df[nfrac_col].notna()]
        if df.empty:
            continue
        bars_data.append((channel, nfrac_col, label_, color_, df))
    if not bars_data:
        ax.set_visible(False); return
    conds_in = []
    for _, _, _, _, df in bars_data:
        conds_in += df["condition"].dropna().unique().tolist() if "condition" in df.columns else []
    conds = order_conditions(sorted(set(conds_in)), condition_order or []) if conds_in else ["(all)"]
    x = np.arange(len(conds))
    width = 0.38
    for offset, (_, nfrac_col, label_, color_, df) in zip([-1, 1], bars_data):
        means = []
        for c in conds:
            sub = df if c == "(all)" else df[df["condition"] == c]
            if len(sub) == 0:
                means.append(0.0)
            else:
                means.append(float((sub[nfrac_col] > 0.5).sum()) / float(len(sub)))
        ax.bar(x + offset * width / 2, means, width, color=color_,
               edgecolor="black", linewidth=0.5, label=label_)
    ax.set_xticks(x); ax.set_xticklabels(conds, rotation=15)
    ax.set_ylabel("Fraction of cells predominantly nuclear\n(nuclear_spot_fraction > 0.5)")
    ax.set_title(_wrap_title("Predominantly-nuclear localization frequency"))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.05)


def plot_active_tss_per_nucleus(ax, nuc: pd.DataFrame,
                                  condition_order: list[str] | None = None) -> None:
    """Box plot of n_active_tss_per_nucleus (paired-AND-nuclear RNA1 spots)
    by condition. For exon/intron designs this approximates per-cell active
    transcription site count."""
    col = "n_nuclear_rna1_rna2_overlap_per_nucleus"
    if col not in nuc.columns:
        ax.set_visible(False); return
    if not _box_strip_with_image_means(ax, nuc, col, only_expressing=False,
                                        condition_order=condition_order):
        ax.set_visible(False); return
    ax.set_ylabel("Nuclear RNA1+RNA2 overlap per nucleus")
    # 2026-05-18 Brian: trimmed to its informative core per the title-wrap
    # task. "Nuclear RNA1 spots overlapping with RNA2 — per nucleus (within
    # 0.3 µm)" was >2 lines even wrapped at width=70.
    ax.set_title(_wrap_title("Nuclear RNA1+RNA2 overlap per nucleus (≤0.3 µm)"))
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=8, loc="upper right")


def plot_mature_mrna_per_cell(ax, nuc: pd.DataFrame, channel: str,
                                condition_order: list[str] | None = None) -> None:
    """Box of n_mature_mrna_<chan>_per_cell (cytoplasmic spots) by condition.
    Reflects the exported / mature mRNA pool. For exon probes this should
    track total mRNA abundance; for intron probes ~ 0."""
    col = "n_cytoplasmic_rna1_spots_per_cell" if channel == "rna1" else "n_cytoplasmic_rna2_spots_per_cell"
    if col not in nuc.columns:
        ax.set_visible(False); return
    if not _box_strip_with_image_means(ax, nuc, col, only_expressing=False,
                                        condition_order=condition_order):
        ax.set_visible(False); return
    ax.set_ylabel(f"Cytoplasmic {channel.upper()} spots per cell")
    ax.set_title(_wrap_title(f"Cytoplasmic {channel.upper()} spots per cell\n(cytoplasmic spots)"))
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=8, loc="upper right")


def plot_paired_only_nuc_vs_cyto(ax, spots: pd.DataFrame) -> None:
    """Stacked bar per image: paired-spot count split into nuclear vs
    cytoplasmic. Asks: 'Where do the co-localized punctae live?'"""
    if spots.empty or "in_nucleus" not in spots.columns:
        ax.set_visible(False); return
    suffix = _find_pair_suffix(spots) or "0p3um"
    pair_col = f"paired_at_{suffix}"
    if pair_col not in spots.columns:
        # Try the alternate path: 'colocalized' column from Fiji rna_rna
        if "colocalized" in spots.columns:
            pair_col = "colocalized"
        else:
            ax.set_visible(False); return
    paired = spots[pd.to_numeric(spots[pair_col], errors="coerce") == 1]
    if paired.empty:
        ax.set_visible(False); return
    by_img = paired.groupby("image").agg(
        nuclear=("in_nucleus", lambda s: int((pd.to_numeric(s, errors="coerce") == 1).sum())),
        cytoplasmic=("in_nucleus", lambda s: int((pd.to_numeric(s, errors="coerce") == 0).sum())),
    ).sort_index()
    if by_img.empty:
        ax.set_visible(False); return
    label_map = _build_image_labels(paired)
    labels = [label_map.get(i, short_label(i)) for i in by_img.index]
    x = np.arange(len(labels))
    ax.bar(x, by_img["nuclear"], label="overlapping · nuclear", color=COLOR_NUCLEAR,
           edgecolor="black", linewidth=0.5)
    ax.bar(x, by_img["cytoplasmic"], bottom=by_img["nuclear"],
           label="overlapping · cytoplasmic", color=COLOR_CYTOPLASMIC,
           edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("Overlapping spot count (RNA1↔RNA2 within 0.3 µm)")
    ax.set_title(_wrap_title("Overlapping spots — nuclear vs cytoplasmic\n"
                 "(RNA1↔RNA2 within 0.3 µm; where do the overlapping spots live?)"))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")


def plot_rna1_vs_rna2_intensity_scatter(ax, nuc: pd.DataFrame,
                                          condition_order: list[str] | None = None) -> None:
    """Per-cell total SPOT intensity scatter — RNA1 vs RNA2 with OLS + Pearson r.

    2026-05-18 Brian: switched from ``cell_total_intensity_rna{1,2}`` (whole-
    cell pixel sums, autofluorescence-contaminated) to the spot-only
    columns so the correlation reflects co-expression of detected RNA, not
    co-variation of camera offset / background."""
    c1 = _resolve_col(nuc, "rna_spot_total_peak_intensity", "rna_spot_total_intensity_fit")
    c2 = _resolve_col(nuc, "rna2_spot_total_peak_intensity", "rna2_spot_total_intensity_fit")
    used_spot = (c1 in nuc.columns and c2 in nuc.columns)
    if not used_spot:
        # Fallback to legacy contaminated columns if spot-only sums absent.
        c1, c2 = "cell_total_intensity_rna1", "cell_total_intensity_rna2"
    if c1 not in nuc.columns or c2 not in nuc.columns:
        ax.set_visible(False); return
    df = nuc.copy()
    df["_x"] = pd.to_numeric(df[c1], errors="coerce")
    df["_y"] = pd.to_numeric(df[c2], errors="coerce")
    df = df[df["_x"].notna() & df["_y"].notna() & (df["_x"] > 0) & (df["_y"] > 0)]
    if len(df) < 2:
        ax.set_visible(False); return
    conds = sorted(df["condition"].dropna().unique()) if "condition" in df.columns else []
    conds = order_conditions(conds, condition_order or []) if conds else []
    for i, cond in enumerate(conds):
        sub = df[df["condition"] == cond]
        if sub.empty: continue
        ax.scatter(sub["_x"], sub["_y"], s=18, alpha=0.65,
                   color=_color_for_condition(cond, i),
                   edgecolor="white", linewidth=0.4,
                   label=f"{cond} (n={len(sub)})")
    if not conds:
        ax.scatter(df["_x"], df["_y"], s=18, alpha=0.55)
    x = df["_x"].values.astype(float)
    y = df["_y"].values.astype(float)
    if len(x) >= 2 and float(np.std(x)) > 0:
        m_, b_ = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, m_ * xs + b_, color="black", linestyle="--", linewidth=1.2,
                label=f"OLS slope={m_:.2f}")
        try:
            r_ = float(np.corrcoef(x, y)[0, 1])
            ax.text(0.02, 0.98, f"Pearson r = {r_:.2f}", transform=ax.transAxes,
                    ha="left", va="top", fontsize=9,
                    bbox=dict(facecolor="white", edgecolor="gray", alpha=0.85, boxstyle="round"))
        except Exception:
            pass
    _suffix = "spot peak intensity" if used_spot else "pixel-sum intensity"
    ax.set_xlabel(f"Per-cell total RNA1 {_suffix}")
    ax.set_ylabel(f"Per-cell total RNA2 {_suffix}")
    ax.set_title(_wrap_title(f"Per-cell total RNA1 vs RNA2 {_suffix}"))
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")


def plot_transcription_efficiency_proxy(ax, nuc: pd.DataFrame,
                                          condition_order: list[str] | None = None) -> None:
    """Box plot per condition: (nuclear paired spots) / (RNA1 in-nucleus
    spots). For exon (RNA1) / intron (RNA2) design, this is the fraction of
    active transcription sites currently making nascent intron — a TSS
    EFFICIENCY proxy (per nucleus). Requires nuclear_spot_count > 0."""
    if "n_nuclear_rna1_rna2_overlap_per_nucleus" not in nuc.columns or "nuclear_spot_count" not in nuc.columns:
        ax.set_visible(False); return
    df = nuc.copy()
    nu_ = pd.to_numeric(df["nuclear_spot_count"], errors="coerce")
    df["_tss_eff"] = pd.to_numeric(df["n_nuclear_rna1_rna2_overlap_per_nucleus"], errors="coerce") / nu_.replace(0, np.nan)
    df = df[df["_tss_eff"].notna()]
    if df.empty:
        ax.set_visible(False); return
    if not _box_strip_with_image_means(ax, df, "_tss_eff", only_expressing=False,
                                        condition_order=condition_order):
        ax.set_visible(False); return
    ax.set_ylabel(
        "Fraction of nuclear RNA1 spots overlapping with RNA2"
    )
    ax.set_title(_wrap_title(
        "Fraction of nuclear RNA1 spots overlapping with RNA2 (within 0.3 µm)\n"
        "(by condition)"
    ))
    ax.tick_params(axis="x", rotation=15)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=8, loc="upper right")


def plot_burst_size_distribution(ax, nuc: pd.DataFrame,
                                   condition_order: list[str] | None = None) -> None:
    """Histogram per condition of n_active_tss_per_nucleus for cells with
    ≥1 active TSS. Right-skewed = bursting; concentrated = constitutive."""
    if "n_nuclear_rna1_rna2_overlap_per_nucleus" not in nuc.columns:
        ax.set_visible(False); return
    df = nuc.copy()
    df["_tss"] = pd.to_numeric(df["n_nuclear_rna1_rna2_overlap_per_nucleus"], errors="coerce")
    df = df[df["_tss"] >= 1]
    if df.empty:
        ax.set_visible(False); return
    conds = sorted(df["condition"].dropna().unique()) if "condition" in df.columns else []
    conds = order_conditions(conds, condition_order or []) if conds else ["(all)"]
    vmax = int(df["_tss"].max())
    bins = np.arange(0, max(vmax + 2, 6), 1)
    for i, cond in enumerate(conds):
        sub = df if cond == "(all)" else df[df["condition"] == cond]
        if sub.empty:
            continue
        ax.hist(sub["_tss"], bins=bins, alpha=0.55,
                color=_color_for_condition(cond, i),
                label=f"{cond} (n={len(sub)}, mean={sub['_tss'].mean():.1f})")
    ax.set_xlabel("Nuclear RNA1+RNA2 overlap count per nucleus (cells with ≥1)")
    ax.set_ylabel("Number of cells")
    ax.set_title(_wrap_title("Nuclear-overlap count distribution (cells with ≥1)\n(RNA1↔RNA2 within 0.3 µm)"))
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)


def plot_nascent_to_mature_ratio(ax, nuc: pd.DataFrame,
                                   condition_order: list[str] | None = None) -> None:
    """Box of (n_active_tss) / (n_mature_mrna_rna1) per cell, by condition.
    Captures the cell-state balance between active transcription and mature
    mRNA accumulation. Lower = more mature transcript relative to nascent."""
    if "n_nuclear_rna1_rna2_overlap_per_nucleus" not in nuc.columns or "n_cytoplasmic_rna1_spots_per_cell" not in nuc.columns:
        ax.set_visible(False); return
    df = nuc.copy()
    nas = pd.to_numeric(df["n_nuclear_rna1_rna2_overlap_per_nucleus"], errors="coerce")
    mat = pd.to_numeric(df["n_cytoplasmic_rna1_spots_per_cell"], errors="coerce")
    df["_ratio"] = nas / mat.replace(0, np.nan)
    df = df[df["_ratio"].notna() & np.isfinite(df["_ratio"])]
    if df.empty:
        ax.set_visible(False); return
    if not _box_strip_with_image_means(ax, df, "_ratio", only_expressing=False,
                                        condition_order=condition_order):
        ax.set_visible(False); return
    ax.set_ylabel("(Nuclear RNA1+RNA2 overlap) / (cytoplasmic RNA1)")
    ax.set_title(_wrap_title("(Nuclear RNA1+RNA2 overlap) / (cytoplasmic RNA1) — per cell\n(within 0.3 µm)"))
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=8, loc="upper right")
    valid = df["_ratio"].values
    if len(valid) and (valid.max() / max(valid[valid > 0].min() if (valid > 0).any() else 1, 1)) > 100:
        ax.set_yscale("log")


def plot_tss_to_edge_distance(ax, spots: pd.DataFrame, nuc: pd.DataFrame) -> None:
    """Histogram of distance from active-TSS spots to nuclear edge. Only
    available if spot_metrics has 'spot_to_nuc_edge_um' AND a paired_at flag."""
    if spots is None or len(spots) == 0:
        ax.set_visible(False); return
    if "spot_to_nuc_edge_um" not in spots.columns:
        ax.set_visible(False); return
    suffix = _find_pair_suffix(spots) or "0p3um"
    pair_col = f"paired_at_{suffix}"
    if pair_col not in spots.columns:
        if "colocalized" in spots.columns:
            pair_col = "colocalized"
        else:
            ax.set_visible(False); return
    sub = spots.copy()
    sub = sub[pd.to_numeric(sub[pair_col], errors="coerce") == 1]
    if "in_nucleus" in sub.columns:
        sub = sub[pd.to_numeric(sub["in_nucleus"], errors="coerce") == 1]
    if "channel" in sub.columns:
        sub = sub[sub["channel"] == "rna1"]
    d = pd.to_numeric(sub["spot_to_nuc_edge_um"], errors="coerce").dropna()
    if d.empty:
        ax.set_visible(False); return
    conds = sorted(sub["condition"].dropna().unique()) if "condition" in sub.columns else []
    bins = np.linspace(0, max(float(d.max()), 0.1), 30)
    for i, cond in enumerate(conds):
        s = sub[sub["condition"] == cond]
        v = pd.to_numeric(s["spot_to_nuc_edge_um"], errors="coerce").dropna()
        if v.empty: continue
        ax.hist(v, bins=bins, alpha=0.55,
                color=_color_for_condition(cond, i),
                label=f"{cond} (n={len(v)})")
    if not conds:
        ax.hist(d, bins=bins, alpha=0.65)
    ax.set_xlabel("Distance from nuclear overlap to nuclear edge (µm)")
    ax.set_ylabel("Overlapping nuclear RNA1 spot count")
    ax.set_title(_wrap_title("Nuclear-overlap-to-nuclear-edge distance\n(RNA1↔RNA2 within 0.3 µm; 0 = at envelope, larger = interior)"))
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)


def plot_coexpression_quadrants(ax, nuc: pd.DataFrame,
                                  condition_order: list[str] | None = None) -> None:
    """Per-cell quadrant chart: RNA1-only / RNA2-only / both / neither. One
    grouped-bar set per condition, sums to 100% of cells. Uses spot counts
    > 0 as the per-channel expression threshold."""
    if "rna_spot_count" not in nuc.columns or "n_spots_rna2" not in nuc.columns:
        ax.set_visible(False); return
    df = nuc.copy()
    a = pd.to_numeric(df["rna_spot_count"], errors="coerce").fillna(0) > 0
    b = pd.to_numeric(df["n_spots_rna2"], errors="coerce").fillna(0) > 0
    df["_quad"] = "neither"
    df.loc[a & ~b, "_quad"] = "RNA1 only"
    df.loc[~a & b, "_quad"] = "RNA2 only"
    df.loc[a & b, "_quad"] = "Both"
    conds = sorted(df["condition"].dropna().unique()) if "condition" in df.columns else []
    conds = order_conditions(conds, condition_order or []) if conds else ["(all)"]
    quad_order = ["Both", "RNA1 only", "RNA2 only", "neither"]
    quad_colors = {
        "Both": OKABE_ITO[5],         # vermillion
        "RNA1 only": COLOR_RNA1,      # yellow
        "RNA2 only": COLOR_RNA2,      # sky blue
        "neither": "#cccccc",
    }
    x = np.arange(len(conds))
    width = 0.18
    for i, q in enumerate(quad_order):
        pcts = []
        for c in conds:
            sub = df if c == "(all)" else df[df["condition"] == c]
            tot = max(1, len(sub))
            pcts.append(100.0 * float((sub["_quad"] == q).sum()) / float(tot))
        ax.bar(x + (i - 1.5) * width, pcts, width, color=quad_colors[q],
               edgecolor="black", linewidth=0.5, label=q)
    ax.set_xticks(x); ax.set_xticklabels(conds, rotation=15)
    ax.set_ylabel("% of cells")
    ax.set_title(_wrap_title("Co-expression quadrants — by condition\n(spot count > 0 = expressing)"))
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 105)


def plot_per_condition_pearson_r(ax, nuc: pd.DataFrame,
                                   condition_order: list[str] | None = None) -> None:
    """Per-condition Pearson r between RNA1 and RNA2 spot counts (per cell).
    Plots r ± 95% CI (Fisher z-transform). Flags r < 0 with a red asterisk —
    suggestive of mutually exclusive expression."""
    if "rna_spot_count" not in nuc.columns or "n_spots_rna2" not in nuc.columns:
        ax.set_visible(False); return
    df = nuc.copy()
    df["_x"] = pd.to_numeric(df["rna_spot_count"], errors="coerce")
    df["_y"] = pd.to_numeric(df["n_spots_rna2"], errors="coerce")
    df = df[df["_x"].notna() & df["_y"].notna()]
    conds = sorted(df["condition"].dropna().unique()) if "condition" in df.columns else []
    conds = order_conditions(conds, condition_order or []) if conds else ["(all)"]
    if not conds:
        ax.set_visible(False); return
    rs, los, his, ns = [], [], [], []
    for c in conds:
        sub = df if c == "(all)" else df[df["condition"] == c]
        if len(sub) < 3 or float(np.std(sub["_x"])) == 0 or float(np.std(sub["_y"])) == 0:
            rs.append(np.nan); los.append(np.nan); his.append(np.nan); ns.append(len(sub))
            continue
        r_ = float(np.corrcoef(sub["_x"], sub["_y"])[0, 1])
        n_ = len(sub)
        # Fisher z transform CI
        z = 0.5 * np.log((1 + r_) / (1 - r_)) if abs(r_) < 1 else np.nan
        se = 1.0 / np.sqrt(max(n_ - 3, 1))
        zlo = z - 1.96 * se if z == z else np.nan
        zhi = z + 1.96 * se if z == z else np.nan
        rlo = (np.exp(2 * zlo) - 1) / (np.exp(2 * zlo) + 1) if zlo == zlo else np.nan
        rhi = (np.exp(2 * zhi) - 1) / (np.exp(2 * zhi) + 1) if zhi == zhi else np.nan
        rs.append(r_); los.append(rlo); his.append(rhi); ns.append(n_)
    xpos = np.arange(len(conds))
    cond_colors = [_color_for_condition(c, i) for i, c in enumerate(conds)]
    bars = ax.bar(xpos, [r if r == r else 0 for r in rs], color=cond_colors,
                  edgecolor="black", linewidth=0.5)
    yerr_lo = [(r - lo) if (r == r and lo == lo) else 0 for r, lo in zip(rs, los)]
    yerr_hi = [(hi - r) if (r == r and hi == hi) else 0 for r, hi in zip(rs, his)]
    ax.errorbar(xpos, [r if r == r else 0 for r in rs],
                yerr=[yerr_lo, yerr_hi], fmt="none", ecolor="black", capsize=4)
    for i, r in enumerate(rs):
        if r == r and r < 0:
            ax.text(xpos[i], -0.05, "*", ha="center", va="top", fontsize=14, color="red")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"{c}\n(n={n})" for c, n in zip(conds, ns)], rotation=15)
    ax.set_ylabel("Pearson r (per-cell RNA1 vs RNA2 spots)")
    ax.set_title(_wrap_title("Per-condition RNA1↔RNA2 correlation\n(red * = r<0 → mutual exclusion candidate)"))
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, alpha=0.3, axis="y")


def plot_within_nucleus_paired_fraction(ax, spots: pd.DataFrame,
                                          condition_order: list[str] | None = None) -> None:
    """Per-condition bar chart: paired fraction restricted to NUCLEAR spots
    only (a stricter transcription-site coloc measure). Both channels."""
    if spots is None or len(spots) == 0 or "in_nucleus" not in spots.columns:
        ax.set_visible(False); return
    suffix = _find_pair_suffix(spots) or "0p3um"
    pair_col = f"paired_at_{suffix}"
    if pair_col not in spots.columns:
        if "colocalized" in spots.columns:
            pair_col = "colocalized"
        else:
            ax.set_visible(False); return
    sub = spots[pd.to_numeric(spots["in_nucleus"], errors="coerce") == 1].copy()
    if sub.empty:
        ax.set_visible(False); return
    conds = sorted(sub["condition"].dropna().unique()) if "condition" in sub.columns else []
    conds = order_conditions(conds, condition_order or []) if conds else ["(all)"]
    if "channel" not in sub.columns:
        ax.set_visible(False); return
    means1, means2 = [], []
    for c in conds:
        ss = sub if c == "(all)" else sub[sub["condition"] == c]
        for chan, dest in (("rna1", means1), ("rna2", means2)):
            s_ = ss[ss["channel"] == chan]
            if len(s_) == 0:
                dest.append(0.0)
            else:
                dest.append(float(pd.to_numeric(s_[pair_col], errors="coerce").fillna(0).mean()))
    x = np.arange(len(conds))
    width = 0.38
    ax.bar(x - width / 2, means1, width, color=COLOR_RNA1,
           edgecolor="black", linewidth=0.5, label="RNA1 nuclear-only")
    ax.bar(x + width / 2, means2, width, color=COLOR_RNA2,
           edgecolor="black", linewidth=0.5, label="RNA2 nuclear-only")
    ax.set_xticks(x); ax.set_xticklabels(conds, rotation=15)
    ax.set_ylabel(
        "Overlap fraction (nuclear spots only, RNA1↔RNA2 within 0.3 µm)"
    )
    ax.set_title(_wrap_title(
        "Within-nucleus overlap fraction — by condition\n"
        "(nuclear spots only; RNA1↔RNA2 within 0.3 µm)"
    ))
    ax.set_ylim(0, max(0.05, max(means1 + means2 + [0]) * 1.3))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")


def plot_cytoplasmic_nn_clustering(ax, spots: pd.DataFrame) -> None:
    """Histogram of nearest-neighbor distance among cytoplasmic spots of the
    SAME channel — proxy for granule formation. Computed per image, pooled."""
    if spots is None or len(spots) == 0:
        ax.set_visible(False); return
    if "in_nucleus" not in spots.columns or "x_px" not in spots.columns:
        ax.set_visible(False); return
    sub = spots[pd.to_numeric(spots["in_nucleus"], errors="coerce") == 0].copy()
    if sub.empty:
        ax.set_visible(False); return
    # Compute per-image, per-channel NN within same-channel cytoplasmic spots.
    nn_all = {"rna1": [], "rna2": []}
    try:
        from scipy.spatial import cKDTree
    except Exception:
        ax.set_visible(False); return
    for (img, chan), grp in sub.groupby(["image", "channel"]) if "channel" in sub.columns else []:
        if chan not in ("rna1", "rna2") or len(grp) < 2:
            continue
        xy = grp[["x_px", "y_px"]].astype(float).to_numpy()
        tree = cKDTree(xy)
        dists, _ = tree.query(xy, k=2)
        nn = dists[:, 1]
        # Convert px to µm via voxel_xy if available — fallback assumes 0.13 µm/px (Brian default).
        vox = 0.13
        if "voxel_xy_um" in grp.columns:
            v = pd.to_numeric(grp["voxel_xy_um"], errors="coerce").dropna()
            if len(v):
                vox = float(v.iloc[0])
        nn_all[chan].extend((nn * vox).tolist())
    if not (nn_all["rna1"] or nn_all["rna2"]):
        ax.set_visible(False); return
    all_vals = np.array(nn_all["rna1"] + nn_all["rna2"])
    if len(all_vals) == 0:
        ax.set_visible(False); return
    vmax = float(np.quantile(all_vals, 0.95))
    bins = np.linspace(0, max(vmax, 0.5), 40)
    for chan, color in (("rna1", COLOR_RNA1), ("rna2", COLOR_RNA2)):
        vals = nn_all[chan]
        if not vals:
            continue
        ax.hist(vals, bins=bins, alpha=0.55, color=color,
                label=f"{chan.upper()} (n={len(vals)}, median={np.median(vals):.2f} µm)")
    ax.set_xlabel("Same-channel NN distance, cytoplasmic spots (µm)")
    ax.set_ylabel("Spot count")
    ax.set_title(_wrap_title("Cytoplasmic clustering — same-channel NN\n(short NN = granule-like aggregation)"))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_volcano_like_per_metric(ax, nuc: pd.DataFrame,
                                   condition_order: list[str] | None = None) -> None:
    """Volcano-style: log2 fold-change of mean per condition vs first condition
    on the x-axis, -log10 p-value (Mann-Whitney U vs first condition) on y.
    One point per (metric, condition). Only run when there are ≥2 conditions
    with ≥3 cells each."""
    metric_cols = [
        ("rna_spot_count", "RNA1 spots/cell"),
        ("n_spots_rna2", "RNA2 spots/cell"),
        ("nuclear_spot_count", "RNA1 nuclear spots"),
        ("nuclear_spot_count_rna2", "RNA2 nuclear spots"),
        ("cyto_spot_count", "RNA1 cyto spots"),
        ("cyto_spot_count_rna2", "RNA2 cyto spots"),
        ("n_nuclear_rna1_rna2_overlap_per_nucleus", "Nuc RNA1+RNA2 overlap"),
        ("n_cytoplasmic_rna1_spots_per_cell", "Cyto RNA1 spots"),
        # 2026-05-18 Brian: switched intensity metrics to spot-only sums so
        # autofluorescence in sec-only / low-signal cells doesn't drive the
        # volcano. ``rna_spot_total_peak_intensity`` = nuc1 sum of detected-
        # spot peak intensities; ``rna2_spot_total_peak_intensity`` = same for RNA2.
        (_resolve_col(nuc, "rna_spot_total_peak_intensity", "rna_spot_total_intensity_fit"),
         "Total RNA1 spot int"),
        (_resolve_col(nuc, "rna2_spot_total_peak_intensity", "rna2_spot_total_intensity_fit"),
         "Total RNA2 spot int"),
    ]
    if "condition" not in nuc.columns:
        ax.set_visible(False); return
    conds_in = nuc["condition"].dropna().unique().tolist()
    conds = order_conditions(conds_in, condition_order or [])
    if len(conds) < 2:
        ax.set_visible(False); return
    ref = conds[0]
    try:
        from scipy.stats import mannwhitneyu
    except Exception:
        ax.set_visible(False); return
    plotted = 0
    for col, label_ in metric_cols:
        if col not in nuc.columns:
            continue
        ref_vals = pd.to_numeric(nuc[nuc["condition"] == ref][col], errors="coerce").dropna()
        if len(ref_vals) < 3:
            continue
        for i, c in enumerate(conds[1:], start=1):
            test_vals = pd.to_numeric(nuc[nuc["condition"] == c][col], errors="coerce").dropna()
            if len(test_vals) < 3:
                continue
            mref = float(ref_vals.mean())
            mtest = float(test_vals.mean())
            if mref <= 0 or mtest <= 0:
                continue
            log2fc = float(np.log2(mtest / mref))
            try:
                _, p = mannwhitneyu(ref_vals, test_vals, alternative="two-sided")
            except Exception:
                continue
            nlp = -float(np.log10(max(p, 1e-300)))
            color = _color_for_condition(c, i)
            ax.scatter(log2fc, nlp, s=42, alpha=0.75, color=color,
                       edgecolor="black", linewidth=0.5)
            ax.annotate(f"{label_}@{c}", (log2fc, nlp), xytext=(3, 3),
                        textcoords="offset points", fontsize=7, alpha=0.8)
            plotted += 1
    if plotted == 0:
        ax.set_visible(False); return
    ax.axhline(-np.log10(0.05), color="black", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.axvline(0, color="black", linewidth=0.7, alpha=0.5)
    ax.set_xlabel(f"log2 fold-change (vs {ref})")
    ax.set_ylabel("-log10 p (Mann-Whitney U)")
    ax.set_title(_wrap_title("Volcano-like: per-cell metrics across conditions"))
    ax.grid(True, alpha=0.3)


def plot_effect_size_bars(ax, nuc: pd.DataFrame,
                            condition_order: list[str] | None = None) -> None:
    """Cohen's d for per-cell metrics, condition X vs first condition. Top 8
    metrics by |d| across all non-reference conditions."""
    metric_cols = [
        ("rna_spot_count", "RNA1 spots"),
        ("n_spots_rna2", "RNA2 spots"),
        ("nuclear_spot_count", "RNA1 nuc"),
        ("cyto_spot_count", "RNA1 cyto"),
        ("n_nuclear_rna1_rna2_overlap_per_nucleus", "Nuc RNA1+RNA2 overlap"),
        ("n_cytoplasmic_rna1_spots_per_cell", "Cyto RNA1 spots"),
        # 2026-05-18 Brian: switched intensity metrics to spot-only sums
        # (autofluorescence excluded). See volcano above.
        (_resolve_col(nuc, "rna_spot_total_peak_intensity", "rna_spot_total_intensity_fit"),
         "Total RNA1 spot"),
        (_resolve_col(nuc, "rna2_spot_total_peak_intensity", "rna2_spot_total_intensity_fit"),
         "Total RNA2 spot"),
    ]
    if "condition" not in nuc.columns:
        ax.set_visible(False); return
    conds_in = nuc["condition"].dropna().unique().tolist()
    conds = order_conditions(conds_in, condition_order or [])
    if len(conds) < 2:
        ax.set_visible(False); return
    ref = conds[0]
    rows = []
    for col, label_ in metric_cols:
        if col not in nuc.columns:
            continue
        ref_vals = pd.to_numeric(nuc[nuc["condition"] == ref][col], errors="coerce").dropna()
        if len(ref_vals) < 3:
            continue
        for i, c in enumerate(conds[1:], start=1):
            test_vals = pd.to_numeric(nuc[nuc["condition"] == c][col], errors="coerce").dropna()
            if len(test_vals) < 3:
                continue
            pooled = np.sqrt((np.var(ref_vals, ddof=1) + np.var(test_vals, ddof=1)) / 2.0)
            if pooled <= 0:
                continue
            d = (float(test_vals.mean()) - float(ref_vals.mean())) / pooled
            rows.append((label_, c, d, _color_for_condition(c, i)))
    if not rows:
        ax.set_visible(False); return
    rows.sort(key=lambda r: -abs(r[2]))
    rows = rows[:12]
    labels = [f"{label_} ({c})" for label_, c, _, _ in rows]
    ds = [r[2] for r in rows]
    colors = [r[3] for r in rows]
    y = np.arange(len(rows))
    ax.barh(y, ds, color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(f"Cohen's d (vs {ref})")
    ax.set_title(_wrap_title("Per-condition effect sizes — top 12\n(|d|>0.8 large; <0.2 negligible)"))
    ax.grid(True, alpha=0.3, axis="x")


def plot_per_image_variance(ax, nuc: pd.DataFrame,
                              condition_order: list[str] | None = None) -> None:
    """Per-image CV (within condition) vs between-condition Δmean. Flags
    images that contribute outsize variance. One dot per (image, metric)."""
    if "condition" not in nuc.columns or "image" not in nuc.columns:
        ax.set_visible(False); return
    metric_cols = [
        ("rna_spot_count", "RNA1 spots"),
        ("n_spots_rna2", "RNA2 spots"),
        ("n_nuclear_rna1_rna2_overlap_per_nucleus", "Nuc RNA1+RNA2 overlap"),
        ("nc_ratio_total_intensity_rna1", "N/C RNA1"),
    ]
    conds_in = nuc["condition"].dropna().unique().tolist()
    conds = order_conditions(conds_in, condition_order or [])
    if len(conds) < 2:
        ax.set_visible(False); return
    ref = conds[0]
    plotted = 0
    for col, label_ in metric_cols:
        if col not in nuc.columns:
            continue
        ref_mean = float(pd.to_numeric(nuc[nuc["condition"] == ref][col], errors="coerce").dropna().mean())
        for i, c in enumerate(conds[1:], start=1):
            sub = nuc[nuc["condition"] == c]
            if sub.empty:
                continue
            cond_mean = float(pd.to_numeric(sub[col], errors="coerce").dropna().mean())
            if not np.isfinite(cond_mean) or not np.isfinite(ref_mean):
                continue
            dmean = cond_mean - ref_mean
            for img, grp in sub.groupby("image"):
                v = pd.to_numeric(grp[col], errors="coerce").dropna()
                if len(v) < 3 or v.mean() <= 0:
                    continue
                cv_ = float(v.std() / v.mean())
                ax.scatter(cv_, dmean, s=40, alpha=0.65,
                           color=_color_for_condition(c, i), edgecolor="black", linewidth=0.4)
                plotted += 1
    if plotted == 0:
        ax.set_visible(False); return
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xlabel("Within-image CV (per metric)")
    ax.set_ylabel(f"Between-condition Δmean (vs {ref})")
    ax.set_title(_wrap_title("Per-image variance vs cross-condition signal\n(top-right: noisy + big effect)"))
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Composition figures (52–56). 2026-05-18 Brian: "we need percentage of
# total spots that overlap in each thing — if percentage in one condition
# is more nuclear than cyto, that's interesting." Switch from absolute
# counts to compositional % per condition so the across-condition shift
# (e.g. WT 30% nuclear → KO 60% nuclear) reads visually instead of being
# buried in raw counts. All four panels are stacked horizontal bars with
# the % numbers burned into each segment.
# ---------------------------------------------------------------------------

# Two-tone palettes for the composition panels. All Okabe-Ito (colorblind
# safe) and none collide with the condition colors.
_COMP_NUC_COLOR  = OKABE_ITO[4]  # blue          (nuclear)
_COMP_CYTO_COLOR = OKABE_ITO[5]  # vermillion    (cytoplasmic)
_COMP_PAIRED_COLOR = OKABE_ITO[0]  # orange      (overlapping)
_COMP_SOLO_COLOR   = OKABE_ITO[3]  # yellow/sand (non-overlapping)


def _aggregate_per_condition(summary: pd.DataFrame, value_col: str,
                              weight_col: str | None,
                              condition_order: list[str] | None) -> tuple[list[str], list[float], list[int]]:
    """Return (conditions, mean-fractions, image-counts) for a composition
    bar, optionally weighted by ``weight_col`` (e.g. total spot count) so
    images with more events contribute proportionally. Sec-only images
    are kept — they end up with 0% / 0% by definition (no spots → empty
    bar), which is the correct visual signal."""
    if summary is None or len(summary) == 0 or value_col not in summary.columns:
        return [], [], []
    df = summary.copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    if "condition" not in df.columns:
        df["condition"] = "(all)"
    conds_in = df["condition"].dropna().unique().tolist()
    conds = order_conditions(conds_in, condition_order or [])
    means: list[float] = []
    ns: list[int] = []
    for c in conds:
        sub = df[df["condition"] == c]
        v = sub[value_col].copy()
        if weight_col and weight_col in sub.columns:
            w = pd.to_numeric(sub[weight_col], errors="coerce").fillna(0.0)
            # Drop rows where the weight is 0 — they have no events to
            # contribute to the fraction (otherwise sec-only's NaN frac
            # at 0-spot drags the bar down to 0 regardless of other
            # images' real fractions).
            mask = v.notna() & (w > 0)
            if mask.sum() == 0:
                means.append(float("nan")); ns.append(int(len(sub))); continue
            means.append(float(np.average(v[mask].values, weights=w[mask].values)))
        else:
            v = v.dropna()
            means.append(float(v.mean()) if len(v) else float("nan"))
        ns.append(int(len(sub)))
    return conds, means, ns


def _composition_stacked_bar(ax, summary: pd.DataFrame | None,
                              top_col: str, top_label: str, top_color: str,
                              bot_label: str, bot_color: str,
                              title: str, ylabel: str,
                              weight_col: str | None,
                              condition_order: list[str] | None) -> None:
    """Draw a stacked horizontal-bar composition panel.

    Each condition gets one bar of total length 100%. The ``top_col`` is
    the fraction filling the first (e.g. "nuclear") segment; ``1 - top``
    fills the rest. The % numbers are written inside each segment.
    Sec-only bars (where the fraction is NaN because no spots were
    detected) are rendered as a single neutral-gray bar with the label
    "no spots detected".
    """
    conds, fracs, ns = _aggregate_per_condition(summary, top_col, weight_col, condition_order)
    if not conds:
        ax.set_visible(False); return

    y = np.arange(len(conds))
    bar_h = 0.65

    for i, (cond, f) in enumerate(zip(conds, fracs)):
        # 2026-06-05 Brian: condition-colored bar outline REMOVED (redundant
        # with the condition label beside each bar; competed with fill colors).
        edge_c = "none"
        if not np.isfinite(f):
            # No spots → empty bar (sec-only's expected outcome)
            ax.barh(y[i], 100.0, height=bar_h, color="#dddddd",
                    edgecolor=edge_c, linewidth=COND_EDGE_LINEWIDTH)
            ax.text(50.0, y[i], f"{cond}: no spots detected",
                    ha="center", va="center", fontsize=10, color="#444")
            continue
        f_pct = float(f) * 100.0
        rest = 100.0 - f_pct
        ax.barh(y[i], f_pct, height=bar_h, color=top_color,
                edgecolor=edge_c, linewidth=COND_EDGE_LINEWIDTH,
                label=top_label if i == 0 else None)
        ax.barh(y[i], rest, left=f_pct, height=bar_h, color=bot_color,
                edgecolor=edge_c, linewidth=COND_EDGE_LINEWIDTH,
                label=bot_label if i == 0 else None)
        # Burn the % number into each segment, but only if the segment is
        # wide enough to fit text. Otherwise float it just outside.
        if f_pct >= 7:
            ax.text(f_pct / 2.0, y[i], f"{f_pct:.0f}%",
                    ha="center", va="center", fontsize=11,
                    color="white", fontweight="bold")
        else:
            ax.text(f_pct + 1, y[i], f"{f_pct:.0f}%",
                    ha="left", va="center", fontsize=9, color="black")
        if rest >= 7:
            ax.text(f_pct + rest / 2.0, y[i], f"{rest:.0f}%",
                    ha="center", va="center", fontsize=11,
                    color="white", fontweight="bold")
        else:
            ax.text(f_pct + rest - 1, y[i], f"{rest:.0f}%",
                    ha="right", va="center", fontsize=9, color="black")

    ax.set_yticks(y); ax.set_yticklabels([f"{c}\n(n={n} img)" for c, n in zip(conds, ns)])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel(ylabel)
    ax.set_title(_wrap_title(title))
    # Combined legend: segment fills (top_label / bot_label) + condition
    # outline patches so reader knows what the bar EDGE color means.
    from matplotlib.patches import Patch
    fill_hl = [
        (Patch(facecolor=top_color, edgecolor="black", linewidth=0.5), top_label),
        (Patch(facecolor=bot_color, edgecolor="black", linewidth=0.5), bot_label),
    ]
    _apply_combined_legend(ax, fill_hl, conds, loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.25, axis="x")
    # Only the bottom axis carries meaning here; clean up.
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


def plot_composition_rna1_nuc_vs_cyto(ax, summary: pd.DataFrame | None,
                                       condition_order: list[str] | None = None) -> None:
    """Composition panel: per condition, what % of RNA1 spots are nuclear
    vs cytoplasmic. Weighted by total RNA1 spot count so images with more
    events contribute proportionally."""
    _composition_stacked_bar(
        ax, summary,
        top_col="frac_nuclear_rna1",
        top_label="Nuclear", top_color=_COMP_NUC_COLOR,
        bot_label="Cytoplasmic", bot_color=_COMP_CYTO_COLOR,
        title="RNA1 spot localization — composition by condition\n(spot-count-weighted across images)",
        ylabel="% of RNA1 spots",
        weight_col="total_spots_rna1",
        condition_order=condition_order,
    )


def plot_composition_rna2_nuc_vs_cyto(ax, summary: pd.DataFrame | None,
                                       condition_order: list[str] | None = None) -> None:
    """Composition panel: per condition, what % of RNA2 spots are nuclear
    vs cytoplasmic."""
    _composition_stacked_bar(
        ax, summary,
        top_col="frac_nuclear_rna2",
        top_label="Nuclear", top_color=_COMP_NUC_COLOR,
        bot_label="Cytoplasmic", bot_color=_COMP_CYTO_COLOR,
        title="RNA2 spot localization — composition by condition\n(spot-count-weighted across images)",
        ylabel="% of RNA2 spots",
        weight_col="total_spots_rna2",
        condition_order=condition_order,
    )


def plot_composition_overlap_vs_solo_rna1(ax, summary: pd.DataFrame | None,
                                            condition_order: list[str] | None = None) -> None:
    """Composition panel: per condition, what % of RNA1 spots overlap
    with an RNA2 spot within 0.3 µm (i.e. are 'paired') vs not."""
    suffix = _find_pair_suffix(summary) if summary is not None else None
    if suffix is None:
        ax.set_visible(False); return
    col = f"paired_fraction_rna1_at_{suffix}"
    dist_um = suffix.replace("p", ".").replace("um", "")
    _composition_stacked_bar(
        ax, summary,
        top_col=col,
        top_label=f"Overlapping (≤{dist_um} µm of an RNA2)",
        top_color=_COMP_PAIRED_COLOR,
        bot_label="Non-overlapping",
        bot_color=_COMP_SOLO_COLOR,
        title=f"Fraction of RNA1 spots overlapping with RNA2 (within {dist_um} µm) — by condition",
        ylabel="% of RNA1 spots",
        weight_col="total_spots_rna1",
        condition_order=condition_order,
    )


def plot_composition_overlap_vs_solo_rna2(ax, summary: pd.DataFrame | None,
                                            condition_order: list[str] | None = None) -> None:
    """Composition panel: per condition, what % of RNA2 spots overlap
    with an RNA1 spot within 0.3 µm vs not."""
    suffix = _find_pair_suffix(summary) if summary is not None else None
    if suffix is None:
        ax.set_visible(False); return
    col = f"paired_fraction_rna2_at_{suffix}"
    dist_um = suffix.replace("p", ".").replace("um", "")
    _composition_stacked_bar(
        ax, summary,
        top_col=col,
        top_label=f"Overlapping (≤{dist_um} µm of an RNA1)",
        top_color=_COMP_PAIRED_COLOR,
        bot_label="Non-overlapping",
        bot_color=_COMP_SOLO_COLOR,
        title=f"Fraction of RNA2 spots overlapping with RNA1 (within {dist_um} µm) — by condition",
        ylabel="% of RNA2 spots",
        weight_col="total_spots_rna2",
        condition_order=condition_order,
    )


def plot_composition_summary_panel(ax, summary: pd.DataFrame | None,
                                    spots: pd.DataFrame | None = None,
                                    condition_order: list[str] | None = None) -> None:
    """3×2 grid combining figures 52–55 + 57 + 60 into one overview.

    The render loop hands us a single ``ax``; we hijack its figure and lay
    a 3×2 GridSpec on top so the panel renders as a proper combined view
    in both the per-figure pass AND the 00_combined_panel.png overview
    (where this just occupies one cell of the outer grid — gridspec is
    robust to nested grids).

    2026-05-18 Brian (round 2): added figures 57 (cross-channel localization
    composition) and 60 (overlap location split) per the title-wrap +
    comparisons task. Each cell of the inner grid uses width=50 for title
    wrapping because the cells are narrower than a stand-alone figure.

    2026-05-19 Brian: layout was too cramped — subplot titles overlapped
    bars in every cell. Fix: in the standalone-figure case, resize the
    figure to 16×16 so each cell gets enough vertical room; bump hspace
    to 0.50 / wspace to 0.30; enable _COMPACT_SUPPRESS_SUBTITLES so the
    italic explanatory subtitles (useful in standalone figures, clutter
    in the summary) don't render; and shrink each per-subplot title to
    11pt after the helper sets it.
    """
    if summary is None or len(summary) == 0:
        ax.set_visible(False); return

    fig = ax.figure
    # Find the SubplotSpec the host axis is sitting in so we can build a
    # local 3x2 GridSpecFromSubplotSpec on top of it. This way the panel
    # cooperates with the outer grid in the combined-panel rendering.
    try:
        host_ss = ax.get_subplotspec()
    except Exception:
        host_ss = None
    ax.set_visible(False)
    if host_ss is None:
        # Stand-alone figure case (per-figure pass): give the figure
        # enough room for 6 subplots with non-overlapping titles. The
        # outer pass-2 loop creates the figure at (7,5); resize it.
        # 2026-05-20 Brian: bumped 16×16 -> 30×24 so the 56 panel has
        # the same headroom as the 97/98 overviews.
        try:
            fig.set_size_inches(30, 24)
        except Exception:
            pass
        from matplotlib.gridspec import GridSpec as _GS
        inner = _GS(3, 2, figure=fig, hspace=0.50, wspace=0.30)
        cells = [fig.add_subplot(inner[i, j]) for i in range(3) for j in range(2)]
    else:
        from matplotlib.gridspec import GridSpecFromSubplotSpec
        inner = GridSpecFromSubplotSpec(3, 2, subplot_spec=host_ss,
                                         hspace=0.50, wspace=0.30)
        cells = [fig.add_subplot(inner[i, j]) for i in range(3) for j in range(2)]

    a1, a2, a3, a4, a5, a6 = cells
    # Suppress per-subplot italic subtitles for the duration of this
    # render — clutter in a 3×2 grid. Restore on exit.
    global _COMPACT_SUPPRESS_SUBTITLES
    _prev_suppress = _COMPACT_SUPPRESS_SUBTITLES
    _COMPACT_SUPPRESS_SUBTITLES = True
    try:
        plot_composition_rna1_nuc_vs_cyto(a1, summary, condition_order=condition_order)
        plot_composition_rna2_nuc_vs_cyto(a2, summary, condition_order=condition_order)
        plot_composition_overlap_vs_solo_rna1(a3, summary, condition_order=condition_order)
        plot_composition_overlap_vs_solo_rna2(a4, summary, condition_order=condition_order)
        # Row 3: cross-channel localization composition + overlap location split.
        plot_localization_composition_both_channels(a5, summary, condition_order=condition_order)
        # 60 needs the spot_metrics dataframe (per-spot in_nucleus flag); if it
        # wasn't piped through to the host (legacy callers) the cell just hides.
        if spots is not None and len(spots):
            plot_overlap_location_split(a6, spots, condition_order=condition_order)
        else:
            a6.set_visible(False)
        # Shrink each cell's title to 10pt so it fits cleanly above its
        # bars in the cramped 3×2 cell.
        # 2026-05-20 Brian: dropped 11 -> 10pt to match the bumped figsize
        # and the 97/98 overview-grid convention.
        for _cell in (a1, a2, a3, a4, a5, a6):
            try:
                if _cell.get_visible():
                    _cell.title.set_fontsize(10)
            except Exception:
                pass
    finally:
        _COMPACT_SUPPRESS_SUBTITLES = _prev_suppress


# ---------------------------------------------------------------------------
# 2026-05-19 Brian — CORE + COLOC overview panels (figures 97 + 98).
#
# These are not part of the per-figure PLOT_LAYOUT (they don't slot into the
# 4-column outer grid cleanly — they ARE their own multi-panel render).
# main() calls them once each, after Pass 2, into the figures/ directory.
#
# Each is a 2×3 grid of pre-existing figure functions, called via the same
# (ax, ...) entrypoints used in the per-figure pass. Compact-subtitle mode
# is enabled so per-cell italic subtitles don't crowd the bars.
# ---------------------------------------------------------------------------


def _render_overview_grid(out_path: Path,
                            cells_to_render,  # list of callables that take an ax
                            overall_title: str,
                            figsize: tuple[int, int] = (30, 24),
                            dpi: int = 600) -> None:
    """Render a 2×3 grid of pre-existing per-axis figure functions to a PNG.

    cells_to_render: list of 6 callables ``fn(ax) -> None``. Render order is
    row-major: (0,0), (0,1), (0,2), (1,0), (1,1), (1,2). Any callable that
    raises is caught and the corresponding cell is hidden.

    Compact mode is enabled for the duration so per-subplot subtitles don't
    fight the cramped cells. Per-subplot titles are shrunk to 10pt.

    2026-05-20 Brian: bumped default figsize from (20, 14) to (30, 24) so
    97_CORE_overview_panel and 98_COLOC_overview_panel get enough room for
    the 6 cells without titles crashing into bars / legends. Subplot title
    fontsize dropped 12 -> 10pt to match the figsize change.
    """
    from matplotlib.gridspec import GridSpec as _GS
    fig = plt.figure(figsize=figsize, dpi=dpi)
    inner = _GS(2, 3, figure=fig, hspace=0.55, wspace=0.30,
                 left=0.06, right=0.97, top=0.90, bottom=0.10)
    cells = [fig.add_subplot(inner[i, j]) for i in range(2) for j in range(3)]

    global _COMPACT_SUPPRESS_SUBTITLES
    _prev_suppress = _COMPACT_SUPPRESS_SUBTITLES
    _COMPACT_SUPPRESS_SUBTITLES = True
    try:
        for fn, cell in zip(cells_to_render, cells):
            try:
                fn(cell)
            except Exception as e:
                print(f"  WARN: overview cell raised {type(e).__name__}: {e}")
                cell.set_visible(False)
        # Shrink subplot titles to 10pt; they're set inside each fn via
        # ax.set_title(_wrap_title(...)).
        for cell in cells:
            try:
                if cell.get_visible():
                    cell.title.set_fontsize(10)
            except Exception:
                pass
    finally:
        _COMPACT_SUPPRESS_SUBTITLES = _prev_suppress

    fig.suptitle(overall_title, fontsize=16, fontweight="bold", y=0.97)
    _relabel_fig(fig)
    _final_layout_polish(fig, has_subtitle=False)
    fig.savefig(out_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def render_core_overview_panel(out_path: Path,
                                 nuc: pd.DataFrame,
                                 spots: pd.DataFrame,
                                 summary: pd.DataFrame | None,
                                 condition_order: list[str] | None) -> None:
    """97_CORE_overview_panel — the headline biological story.

    2×3 grid:
      (0,0) Fig 57 — Spot localization composition (RNA1 + RNA2 nuc vs cyto)
      (0,1) Fig 17b — Per-cell nuc vs cyto stacked, RNA1
      (0,2) Fig 18b — Per-cell nuc vs cyto stacked, RNA2
      (1,0) Fig 60 — Overlap location split (nuc vs cyto of overlap events)
      (1,1) Fig 35b — % nuclei with ≥1 overlap
      (1,2) Fig 40 — Nuclear RNA1+RNA2 overlap per nucleus (box+strip)
    """
    cells = [
        lambda ax: plot_localization_composition_both_channels(ax, summary, condition_order=condition_order),
        lambda ax: plot_per_cell_nc_stacked_rna1(ax, nuc, condition_order=condition_order),
        lambda ax: plot_per_cell_nc_stacked_rna2(ax, nuc, condition_order=condition_order),
        lambda ax: plot_overlap_location_split(ax, spots, condition_order=condition_order),
        lambda ax: plot_pct_nuclei_with_overlap(ax, nuc, condition_order=condition_order),
        lambda ax: plot_active_tss_per_nucleus(ax, nuc, condition_order=condition_order),
    ]
    _render_overview_grid(
        out_path, cells,
        overall_title="Core findings — spot localization + RNA1↔RNA2 overlap across conditions",
        figsize=(30, 24),
    )


def render_coloc_overview_panel(out_path: Path,
                                  nuc: pd.DataFrame,
                                  spots: pd.DataFrame,
                                  summary: pd.DataFrame | None,
                                  condition_order: list[str] | None) -> None:
    """98_COLOC_overview_panel — the colocalization / overlap story standalone.

    2×3 grid:
      (0,0) Fig 33 — Overlap fraction per condition (raw)
      (0,1) Fig 54 — Composition: overlap vs solo, RNA1
      (0,2) Fig 55 — Composition: overlap vs solo, RNA2
      (1,0) Fig 60 — Overlap location split (nuc vs cyto of overlap events)
      (1,1) Fig 36b — Composition of overlap location
      (1,2) Fig 62 — Nuclear overlap fraction of nuclear RNA1 spots
    """
    cells = [
        lambda ax: plot_paired_fraction_per_condition(ax, summary, condition_order=condition_order),
        lambda ax: plot_composition_overlap_vs_solo_rna1(ax, summary, condition_order=condition_order),
        lambda ax: plot_composition_overlap_vs_solo_rna2(ax, summary, condition_order=condition_order),
        lambda ax: plot_overlap_location_split(ax, spots, condition_order=condition_order),
        lambda ax: plot_composition_overlap_location(ax, spots, condition_order=condition_order),
        lambda ax: plot_nuclear_overlap_fraction_of_nuclear_rna1(ax, nuc, condition_order=condition_order),
    ]
    _render_overview_grid(
        out_path, cells,
        overall_title="RNA1↔RNA2 overlap — composition and location across conditions",
        figsize=(30, 24),
    )


# ---------------------------------------------------------------------------
# 2026-05-20 Brian — PI-focus figures (99–104).
#
# These match the ``PI_Focus`` Excel sheet being built in parallel. Each is a
# headline figure the PI sees first when reviewing a run. All 6 are gated to
# rna_rna mode (the 2×2 channel × compartment grid is meaningless without a
# second channel) — main() skips them silently for rna_only / rna_protein /
# ab_ab. Every defensive lookup uses ``in c.columns``-style checks so a
# missing column degrades to a hidden cell rather than a render crash.
#
# Channel/compartment layout convention (used by 99–102):
#     rows: row 0 = RNA1 ("Introns"), row 1 = RNA2 ("Exons")
#     cols: col 0 = Nuclear, col 1 = Cytoplasmic
# The user-facing "RNA1" / "RNA2" strings get re-mapped to whatever the
# active preset's rna_label / rna2_label resolved to (e.g. "Introns" /
# "Exons") via _relabel_fig() at savefig time — the same mechanism every
# other figure in this file uses.
# ---------------------------------------------------------------------------


def _pi_focus_box_strip(ax, df: pd.DataFrame, value_col: str,
                         condition_order: list[str] | None,
                         ylabel: str) -> bool:
    """Mini box+strip+image-mean-diamond helper for the PI_FOCUS grids.

    Renders WT/KO (+ sec-only) by condition for one (channel, compartment)
    cell. Returns True if data was drawn so callers can skip empty cells
    cleanly. Mirrors ``_box_strip_with_image_means`` but operates on an
    arbitrary numeric column passed in (so callers can supply derived
    values like per-pixel above-floor intensity).
    """
    if df is None or len(df) == 0 or value_col not in df.columns:
        return False
    if "condition" not in df.columns:
        return False
    sub = df.copy()
    sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce")
    sub = sub[sub[value_col].notna()]
    if sub.empty:
        return False
    conds_in_data = sub["condition"].dropna().unique().tolist()
    conditions = order_conditions(conds_in_data, condition_order or [])
    if not conditions:
        return False
    data = [sub[sub["condition"] == c][value_col].values for c in conditions]
    cond_colors = [_color_for_condition(c, i) for i, c in enumerate(conditions)]
    bp = ax.boxplot(data, tick_labels=conditions, showfliers=False,
                    patch_artist=True,
                    medianprops=dict(color="black", linewidth=1.5))
    for patch, color in zip(bp["boxes"], cond_colors):
        patch.set_facecolor(color); patch.set_alpha(0.55)
    rng = np.random.RandomState(0)
    for i, (cond, vals) in enumerate(zip(conditions, data), start=1):
        color = cond_colors[i - 1]
        if len(vals) > 0:
            jitter = (rng.random(len(vals)) - 0.5) * 0.25
            ax.plot(np.full_like(vals, i, dtype=float) + jitter, vals,
                    "o", markersize=2, alpha=0.35, color=color, zorder=2)
        if "image" in sub.columns:
            img_means = (sub[sub["condition"] == cond]
                         .groupby("image")[value_col].mean().values)
            if len(img_means) > 0:
                jitter_im = (rng.random(len(img_means)) - 0.5) * 0.18
                ax.plot(np.full_like(img_means, i, dtype=float) + jitter_im,
                        img_means, "D", markersize=5, color=color,
                        markeredgecolor="black", markeredgewidth=0.7,
                        zorder=4)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3, axis="y")
    return True


def _pi_focus_violin(ax, df: pd.DataFrame, value_col: str,
                      condition_order: list[str] | None,
                      ylabel: str) -> bool:
    """Violin + median bar + image-mean diamond helper.

    Same data as _pi_focus_box_strip but rendered as density violins —
    shows distribution shape directly when the strip-dot version gets too
    crowded. Returns True if data was drawn.
    """
    if df is None or len(df) == 0 or value_col not in df.columns:
        return False
    if "condition" not in df.columns:
        return False
    sub = df.copy()
    sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce")
    sub = sub[sub[value_col].notna()]
    if sub.empty:
        return False
    conds_in_data = sub["condition"].dropna().unique().tolist()
    conditions = order_conditions(conds_in_data, condition_order or [])
    if not conditions:
        return False
    data = [sub[sub["condition"] == c][value_col].values for c in conditions]
    cond_colors = [_color_for_condition(c, i) for i, c in enumerate(conditions)]

    # Drop empty conditions (matplotlib's violinplot errors on empty arrays).
    pairs = [(c, v, col) for c, v, col in zip(conditions, data, cond_colors)
             if len(v) > 0]
    if not pairs:
        return False
    used_conds = [p[0] for p in pairs]
    used_data = [p[1] for p in pairs]
    used_colors = [p[2] for p in pairs]
    positions = list(range(1, len(used_conds) + 1))

    vp = ax.violinplot(used_data, positions=positions, showmedians=False,
                       showextrema=False, widths=0.8)
    for body, color in zip(vp["bodies"], used_colors):
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(0.55)
        body.set_linewidth(0.8)

    # Median bar + Q1/Q3 marker per violin.
    for i, vals in zip(positions, used_data):
        if not len(vals):
            continue
        med = float(np.median(vals))
        q1, q3 = np.percentile(vals, [25, 75])
        ax.plot([i - 0.25, i + 0.25], [med, med], color="black", linewidth=1.8, zorder=4)
        ax.plot([i, i], [q1, q3], color="black", linewidth=1.0, zorder=3)

    # Per-image mean diamonds overlaid (same as box+strip helper).
    rng = np.random.RandomState(0)
    if "image" in sub.columns:
        for i, cond in zip(positions, used_conds):
            color = used_colors[positions.index(i)]
            img_means = (sub[sub["condition"] == cond]
                         .groupby("image")[value_col].mean().values)
            if len(img_means) > 0:
                jitter_im = (rng.random(len(img_means)) - 0.5) * 0.16
                ax.plot(np.full_like(img_means, i, dtype=float) + jitter_im,
                        img_means, "D", markersize=5, color=color,
                        markeredgecolor="black", markeredgewidth=0.7, zorder=5)

    ax.set_xticks(positions)
    ax.set_xticklabels(used_conds)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3, axis="y")
    return True


def render_pi_focus_spot_peak_intensity_violin(out_path: Path, spots: pd.DataFrame,
                                                condition_order: list[str] | None) -> None:
    """Violin variant of 03_spot_peak_intensity_by_compartment.

    2×2 grid (channel × compartment), per-spot peak intensity rendered as
    density violins instead of box+strip. Median bar + Q1-Q3 line + per-
    image mean diamonds overlaid. Cleaner read of distribution shape when
    spot counts are large.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=600)
    if spots is None or len(spots) == 0 \
            or "spot_peak_intensity" not in spots.columns \
            or "in_nucleus" not in spots.columns:
        for r in range(2):
            for c in range(2):
                axes[r, c].set_visible(False)
        fig.suptitle("Spot peak intensity (violins) — by compartment and channel "
                     "(no data)", fontsize=15, fontweight="bold", y=0.98)
        _relabel_fig(fig)
        _final_layout_polish(fig, has_subtitle=False)
        fig.savefig(out_path, bbox_inches="tight", dpi=600)
        plt.close(fig)
        return
    sp = spots.copy()
    sp["in_nucleus"] = pd.to_numeric(sp["in_nucleus"], errors="coerce").fillna(0).astype(int)
    for r, (ch_id, ch_disp) in enumerate(_PI_FOCUS_CHANNELS):
        if "channel" in sp.columns:
            sub_ch = sp[sp["channel"] == ch_id]
        elif ch_id == "rna1":
            sub_ch = sp
        else:
            sub_ch = sp.iloc[0:0]
        for c, (cp_id, cp_disp) in enumerate(_PI_FOCUS_COMPARTMENTS):
            ax = axes[r, c]
            mask = (sub_ch["in_nucleus"] == 1) if cp_id == "nuclear" \
                else (sub_ch["in_nucleus"] == 0)
            sub_cp = sub_ch[mask]
            ok = _pi_focus_violin(ax, sub_cp, "spot_peak_intensity",
                                   condition_order,
                                   ylabel=f"{cp_disp} spot peak intensity")
            if not ok:
                ax.set_visible(False); continue
            ax.set_title(_wrap_title(f"{ch_disp} — {cp_disp}"), fontsize=11)
    fig.suptitle("Spot peak intensity (violin distributions) — by compartment and channel",
                 fontsize=15, fontweight="bold", y=0.98)
    fig.text(0.5, 0.02,
             "BigFISH-fit per-spot peak intensity rendered as density violins. "
             "Bar = median; vertical line = Q1-Q3; diamond = per-image mean.",
             ha="center", fontsize=9, style="italic", color="#555")
    _relabel_fig(fig)
    _final_layout_polish(fig, has_subtitle=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=600)
    plt.close(fig)


# (channel_id, channel_display) and (compartment_id, compartment_display)
# tuple pairs used for the 2×2 grids. The display strings ride through
# _relabel_fig() before savefig so "RNA1" / "RNA2" become the configured
# rna_label / rna2_label.
_PI_FOCUS_CHANNELS = (("rna1", "RNA1"), ("rna2", "RNA2"))
_PI_FOCUS_COMPARTMENTS = (("nuclear", "Nuclear"), ("cytoplasmic", "Cytoplasmic"))


def render_pi_focus_spot_counts(out_path: Path, nuc: pd.DataFrame,
                                 condition_order: list[str] | None) -> None:
    """99_PI_FOCUS_spot_counts_per_compartment.

    2×2 grid: rows = channel (RNA1/RNA2), cols = compartment (Nuclear /
    Cytoplasmic). Y = spots per cell in that compartment. The headline
    PI figure — 4 boxes that immediately tell the story.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=600)
    col_map = {
        ("rna1", "nuclear"):      "nuclear_spot_count",
        ("rna1", "cytoplasmic"):  "cyto_spot_count",
        ("rna2", "nuclear"):      "nuclear_spot_count_rna2",
        ("rna2", "cytoplasmic"):  "cyto_spot_count_rna2",
    }
    for r, (ch_id, ch_disp) in enumerate(_PI_FOCUS_CHANNELS):
        for c, (cp_id, cp_disp) in enumerate(_PI_FOCUS_COMPARTMENTS):
            ax = axes[r, c]
            col = col_map[(ch_id, cp_id)]
            ok = _pi_focus_box_strip(ax, nuc, col, condition_order,
                                      ylabel=f"{cp_disp} spots per cell")
            if not ok:
                ax.set_visible(False); continue
            ax.set_title(_wrap_title(f"{ch_disp} — {cp_disp}"), fontsize=11)
    fig.suptitle("Spot counts per cell — by compartment and channel",
                 fontsize=15, fontweight="bold", y=0.98)
    fig.text(0.5, 0.02,
             "Per-cell spot counts in the named compartment for each channel "
             "(circle = single cell, diamond = per-image mean).",
             ha="center", fontsize=9, style="italic", color="#555")
    _relabel_fig(fig)
    _final_layout_polish(fig, has_subtitle=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=600)
    plt.close(fig)


def render_pi_focus_above_floor(out_path: Path, nuc: pd.DataFrame,
                                  condition_order: list[str] | None) -> None:
    """100_PI_FOCUS_above_floor_intensity.

    2×2 grid: Y = mean per-pixel above-floor intensity in that compartment
    for that channel. Computed at render time as
    ``nuclear_above_floor_intensity_<ch> / nucleus_area_px`` (or the
    cytoplasmic analogue). Cells where the floor column is NaN are
    skipped gracefully (column missing -> cell hidden).
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=600)
    # Column triples: (above_floor_col, area_col)
    col_map = {
        ("rna1", "nuclear"):      ("nuclear_above_floor_intensity_rna1",
                                    "nucleus_area_px"),
        ("rna1", "cytoplasmic"):  ("cytoplasmic_above_floor_intensity_rna1",
                                    "cyto_area_px"),
        ("rna2", "nuclear"):      ("nuclear_above_floor_intensity_rna2",
                                    "nucleus_area_px"),
        ("rna2", "cytoplasmic"):  ("cytoplasmic_above_floor_intensity_rna2",
                                    "cyto_area_px"),
    }
    for r, (ch_id, ch_disp) in enumerate(_PI_FOCUS_CHANNELS):
        for c, (cp_id, cp_disp) in enumerate(_PI_FOCUS_COMPARTMENTS):
            ax = axes[r, c]
            af_col, ar_col = col_map[(ch_id, cp_id)]
            if af_col not in nuc.columns or ar_col not in nuc.columns:
                ax.set_visible(False); continue
            tmp = nuc.copy()
            tmp[af_col] = pd.to_numeric(tmp[af_col], errors="coerce")
            tmp[ar_col] = pd.to_numeric(tmp[ar_col], errors="coerce")
            tmp = tmp[tmp[af_col].notna() & tmp[ar_col].notna() & (tmp[ar_col] > 0)]
            if tmp.empty:
                ax.set_visible(False); continue
            derived = f"_pi_pp_{ch_id}_{cp_id}"
            tmp[derived] = tmp[af_col] / tmp[ar_col]
            ok = _pi_focus_box_strip(ax, tmp, derived, condition_order,
                                      ylabel=f"{cp_disp} above-floor / px")
            if not ok:
                ax.set_visible(False); continue
            ax.set_title(_wrap_title(f"{ch_disp} — {cp_disp}"), fontsize=11)
    fig.suptitle("Above-floor pixel intensity per cell — by compartment and channel",
                 fontsize=15, fontweight="bold", y=0.98)
    fig.text(0.5, 0.02,
             "Per-pixel mean of above-floor intensity in the named compartment "
             "(diamond = per-image mean). Cells with no floor recorded are skipped.",
             ha="center", fontsize=9, style="italic", color="#555")
    _relabel_fig(fig)
    _final_layout_polish(fig, has_subtitle=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=600)
    plt.close(fig)


def render_pi_focus_spot_peak_intensity(out_path: Path, spots: pd.DataFrame,
                                          condition_order: list[str] | None) -> None:
    """101_PI_FOCUS_spot_peak_intensity_by_compartment.

    2×2: per-spot peak intensity (BigFISH fit) split by channel × in-/out-
    nucleus. ``in_nucleus`` truthy -> Nuclear column; falsy -> Cytoplasmic
    column. Defensive: if ``spot_peak_intensity`` or ``in_nucleus`` is
    missing, the corresponding cells hide.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=600)
    if spots is None or len(spots) == 0 \
            or "spot_peak_intensity" not in spots.columns \
            or getattr(spots, "in_nucleus", None) is None and "in_nucleus" not in spots.columns:
        for r in range(2):
            for c in range(2):
                axes[r, c].set_visible(False)
        # Still write the file (empty) so downstream sees a real PNG.
        fig.suptitle("Spot peak intensity — by compartment and channel "
                     "(no data)", fontsize=15, fontweight="bold", y=0.98)
        _relabel_fig(fig)
        _final_layout_polish(fig, has_subtitle=False)
        fig.savefig(out_path, bbox_inches="tight", dpi=600)
        plt.close(fig)
        return
    sp = spots.copy()
    sp["in_nucleus"] = pd.to_numeric(sp["in_nucleus"], errors="coerce").fillna(0).astype(int)
    for r, (ch_id, ch_disp) in enumerate(_PI_FOCUS_CHANNELS):
        # Channel filter (rna_rna data always has 'channel'; rna_only doesn't
        # — in which case rna2 row hides cleanly).
        if "channel" in sp.columns:
            sub_ch = sp[sp["channel"] == ch_id]
        elif ch_id == "rna1":
            sub_ch = sp
        else:
            sub_ch = sp.iloc[0:0]
        for c, (cp_id, cp_disp) in enumerate(_PI_FOCUS_COMPARTMENTS):
            ax = axes[r, c]
            mask = (sub_ch["in_nucleus"] == 1) if cp_id == "nuclear" \
                else (sub_ch["in_nucleus"] == 0)
            sub_cp = sub_ch[mask]
            ok = _pi_focus_box_strip(ax, sub_cp, "spot_peak_intensity",
                                      condition_order,
                                      ylabel=f"{cp_disp} spot peak intensity")
            if not ok:
                ax.set_visible(False); continue
            ax.set_title(_wrap_title(f"{ch_disp} — {cp_disp}"), fontsize=11)
    fig.suptitle("Spot peak intensity — by compartment and channel",
                 fontsize=15, fontweight="bold", y=0.98)
    fig.text(0.5, 0.02,
             "BigFISH-fit per-spot peak intensity, split by channel and "
             "in_nucleus / in_cytoplasm flag (diamond = per-image mean).",
             ha="center", fontsize=9, style="italic", color="#555")
    _relabel_fig(fig)
    _final_layout_polish(fig, has_subtitle=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=600)
    plt.close(fig)


def render_pi_focus_spot_size(out_path: Path, spots: pd.DataFrame,
                                condition_order: list[str] | None) -> None:
    """102_PI_FOCUS_spot_size_by_compartment.

    Same 2×2 layout as 101 but Y = per-spot ``spot_diameter_um``.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=600)
    if spots is None or len(spots) == 0 \
            or "spot_diameter_um" not in spots.columns \
            or "in_nucleus" not in spots.columns:
        for r in range(2):
            for c in range(2):
                axes[r, c].set_visible(False)
        fig.suptitle("Spot diameter (µm) — by compartment and channel "
                     "(no data)", fontsize=15, fontweight="bold", y=0.98)
        _relabel_fig(fig)
        _final_layout_polish(fig, has_subtitle=False)
        fig.savefig(out_path, bbox_inches="tight", dpi=600)
        plt.close(fig)
        return
    sp = spots.copy()
    sp["in_nucleus"] = pd.to_numeric(sp["in_nucleus"], errors="coerce").fillna(0).astype(int)
    for r, (ch_id, ch_disp) in enumerate(_PI_FOCUS_CHANNELS):
        if "channel" in sp.columns:
            sub_ch = sp[sp["channel"] == ch_id]
        elif ch_id == "rna1":
            sub_ch = sp
        else:
            sub_ch = sp.iloc[0:0]
        for c, (cp_id, cp_disp) in enumerate(_PI_FOCUS_COMPARTMENTS):
            ax = axes[r, c]
            mask = (sub_ch["in_nucleus"] == 1) if cp_id == "nuclear" \
                else (sub_ch["in_nucleus"] == 0)
            sub_cp = sub_ch[mask]
            ok = _pi_focus_box_strip(ax, sub_cp, "spot_diameter_um",
                                      condition_order,
                                      ylabel=f"{cp_disp} spot diameter (µm)")
            if not ok:
                ax.set_visible(False); continue
            ax.set_title(_wrap_title(f"{ch_disp} — {cp_disp}"), fontsize=11)
    fig.suptitle("Spot diameter (µm) — by compartment and channel",
                 fontsize=15, fontweight="bold", y=0.98)
    fig.text(0.5, 0.02,
             "Per-spot diameter (µm), split by channel and compartment "
             "(diamond = per-image mean).",
             ha="center", fontsize=9, style="italic", color="#555")
    _relabel_fig(fig)
    _final_layout_polish(fig, has_subtitle=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=600)
    plt.close(fig)


def render_pi_focus_localization_summary(out_path: Path, spots: pd.DataFrame,
                                          condition_order: list[str] | None) -> None:
    """103_PI_FOCUS_localization_summary.

    2×1 grid: row 0 = stacked bar of % nuclear vs % cytoplasmic per
    condition for RNA1; row 1 = same for RNA2. The two-row view answers
    "how does WT vs KO shift the nuc/cyto split for each channel?" at a
    glance.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 12), dpi=600)
    if spots is None or len(spots) == 0 \
            or "in_nucleus" not in spots.columns \
            or "condition" not in spots.columns:
        for r in range(2):
            axes[r].set_visible(False)
        fig.suptitle("Spot localization composition by condition (no data)",
                     fontsize=15, fontweight="bold", y=0.98)
        _relabel_fig(fig)
        _final_layout_polish(fig, has_subtitle=False)
        fig.savefig(out_path, bbox_inches="tight", dpi=600)
        plt.close(fig)
        return
    sp = spots.copy()
    sp["in_nucleus"] = pd.to_numeric(sp["in_nucleus"], errors="coerce").fillna(0).astype(int)
    for r, (ch_id, ch_disp) in enumerate(_PI_FOCUS_CHANNELS):
        ax = axes[r]
        if "channel" in sp.columns:
            sub_ch = sp[sp["channel"] == ch_id]
        elif ch_id == "rna1":
            sub_ch = sp
        else:
            sub_ch = sp.iloc[0:0]
        if sub_ch.empty:
            ax.set_visible(False); continue
        conds_in_data = sub_ch["condition"].dropna().unique().tolist()
        conditions = order_conditions(conds_in_data, condition_order or [])
        if not conditions:
            ax.set_visible(False); continue
        nuc_pcts = []
        cyto_pcts = []
        for cond in conditions:
            cs = sub_ch[sub_ch["condition"] == cond]
            n_total = len(cs)
            if n_total == 0:
                nuc_pcts.append(0.0); cyto_pcts.append(0.0); continue
            n_nuc = int((cs["in_nucleus"] == 1).sum())
            nuc_pcts.append(100.0 * n_nuc / n_total)
            cyto_pcts.append(100.0 - 100.0 * n_nuc / n_total)
        x = np.arange(len(conditions))
        ax.bar(x, nuc_pcts, color=COLOR_NUCLEAR, label="Nuclear")
        ax.bar(x, cyto_pcts, bottom=nuc_pcts, color=COLOR_CYTOPLASMIC,
               label="Cytoplasmic")
        ax.set_xticks(x); ax.set_xticklabels(conditions)
        ax.set_ylabel("% of spots")
        ax.set_ylim(0, 100)
        ax.set_title(_wrap_title(f"{ch_disp} — nuclear vs cytoplasmic"),
                     fontsize=11)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("Spot localization composition by condition",
                 fontsize=15, fontweight="bold", y=0.98)
    fig.text(0.5, 0.02,
             "Stacked % of spots in each compartment, per condition, for "
             "each channel.",
             ha="center", fontsize=9, style="italic", color="#555")
    _relabel_fig(fig)
    _final_layout_polish(fig, has_subtitle=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=600)
    plt.close(fig)


def render_headline_nuclear_retention(
    out_path: Path, nuc: pd.DataFrame, condition_order: list[str] | None,
) -> None:
    """Headline biology figure: nuclear retention shift WT vs KO.

    Shows fraction of total RNA spots that are nuclear, per channel, per
    condition, as bars. The pp shift (KO-WT) is annotated above each pair.
    This is the headline finding across all 3 genes: KO sequesters mRNA
    in the nucleus.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=600)
    conds = order_conditions(nuc["condition"].dropna().unique().tolist(), condition_order or [])
    real = nuc[nuc.get("secondary_only", False) != True] if "secondary_only" in nuc.columns else nuc
    channels = [
        ("rna1", "Introns", "nuclear_spot_fraction", axes[0]),
        ("rna2", "Exons",   "nuclear_spot_fraction_rna2", axes[1]),
    ]
    for ch_id, ch_disp, col, ax in channels:
        if col not in real.columns:
            ax.set_visible(False); continue
        vals_by_cond = []
        for c in conds:
            s = real.loc[real["condition"] == c, col].dropna()
            if not len(s):
                vals_by_cond.append((c, np.nan, np.nan, 0))
                continue
            mean = float(s.mean())
            sem = float(s.std() / np.sqrt(len(s))) if len(s) > 1 else 0.0
            vals_by_cond.append((c, mean, sem, len(s)))
        cond_names = [v[0] for v in vals_by_cond]
        means = [100 * v[1] if not np.isnan(v[1]) else 0 for v in vals_by_cond]
        sems = [100 * v[2] if not np.isnan(v[2]) else 0 for v in vals_by_cond]
        colors = [_color_for_condition(c, i) for i, c in enumerate(cond_names)]
        bars = ax.bar(cond_names, means, yerr=sems, capsize=5, color=colors, alpha=0.85, edgecolor="black", linewidth=1.5)
        for bar, m, n in zip(bars, means, [v[3] for v in vals_by_cond]):
            ax.text(bar.get_x() + bar.get_width()/2, m + 2, f"{m:.1f}%\nn={n}",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
        # Annotate WT→KO shift if both present
        if "WT" in cond_names and "KO" in cond_names:
            i_wt = cond_names.index("WT")
            i_ko = cond_names.index("KO")
            shift = means[i_ko] - means[i_wt]
            mid_x = (bars[i_wt].get_x() + bars[i_ko].get_x() + bars[i_wt].get_width()) / 2
            y_top = max(means) + 12
            ax.annotate("", xy=(bars[i_ko].get_x() + bars[i_ko].get_width()/2, y_top),
                        xytext=(bars[i_wt].get_x() + bars[i_wt].get_width()/2, y_top),
                        arrowprops=dict(arrowstyle="->", color="firebrick", lw=2))
            sign = "+" if shift >= 0 else ""
            ax.text(mid_x, y_top + 3, f"shift {sign}{shift:.1f} pp",
                    ha="center", va="bottom", fontsize=13, fontweight="bold", color="firebrick")
        ax.set_ylabel(f"Fraction of {ch_disp} spots nuclear (%)")
        ax.set_ylim(0, max(110, max(means) * 1.4))
        ax.set_title(_wrap_title(f"{ch_disp} — Nuclear Retention Shift"), fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("Nuclear retention shift (WT → KO)", fontsize=16, fontweight="bold")
    _relabel_fig(fig)
    _final_layout_polish(fig, has_subtitle=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=600)
    plt.close(fig)


def render_headline_compartment_redistribution(
    out_path: Path, nuc: pd.DataFrame, condition_order: list[str] | None,
) -> None:
    """Headline biology figure: compartment redistribution stacked bars.

    Per condition × channel, what fraction of spots live in nucleus vs
    cytoplasm. Visualizes the WT→KO shift away from cytoplasm.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=600)
    conds = order_conditions(nuc["condition"].dropna().unique().tolist(), condition_order or [])
    real = nuc[nuc.get("secondary_only", False) != True] if "secondary_only" in nuc.columns else nuc
    channels = [
        ("Introns", "nuclear_spot_count", "cyto_spot_count", axes[0]),
        ("Exons",   "nuclear_spot_count_rna2", "cyto_spot_count_rna2", axes[1]),
    ]
    for ch_disp, nuc_col, cyt_col, ax in channels:
        if nuc_col not in real.columns or cyt_col not in real.columns:
            ax.set_visible(False); continue
        cond_names, nuc_pcts, cyt_pcts = [], [], []
        for c in conds:
            s = real[real["condition"] == c]
            if not len(s): continue
            total = s[nuc_col].sum() + s[cyt_col].sum()
            if total <= 0: continue
            cond_names.append(c)
            nuc_pcts.append(100 * s[nuc_col].sum() / total)
            cyt_pcts.append(100 * s[cyt_col].sum() / total)
        if not cond_names: ax.set_visible(False); continue
        x = np.arange(len(cond_names))
        ax.bar(x, nuc_pcts, width=0.6, color="#3F51B5", edgecolor="black", linewidth=1.5, label="Nuclear")
        ax.bar(x, cyt_pcts, width=0.6, bottom=nuc_pcts, color="#FFC107", edgecolor="black", linewidth=1.5, label="Cytoplasmic")
        for i, (n, c) in enumerate(zip(nuc_pcts, cyt_pcts)):
            ax.text(i, n / 2, f"{n:.1f}%", ha="center", va="center", color="white", fontsize=12, fontweight="bold")
            ax.text(i, n + c / 2, f"{c:.1f}%", ha="center", va="center", color="black", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(cond_names)
        ax.set_ylabel(f"Fraction of {ch_disp} spots (%)")
        ax.set_ylim(0, 110)
        ax.set_title(_wrap_title(f"{ch_disp} — Compartment redistribution"), fontsize=13, fontweight="bold")
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("Where the spots live — nuclear vs cytoplasmic by condition",
                 fontsize=16, fontweight="bold")
    _relabel_fig(fig)
    _final_layout_polish(fig, has_subtitle=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=600)
    plt.close(fig)


def render_headline_ko_wt_log2fc_panel(
    out_path: Path, nuc: pd.DataFrame, condition_order: list[str] | None,
) -> None:
    """Headline biology figure: KO/WT log2 fold-change across all
    channel×compartment slices. At-a-glance "what changed in KO" panel.

    Shows log2(KO mean / WT mean) for: total/nuclear/cytoplasmic spots per
    cell, both channels. Bars colored by direction (red↑KO, blue↓KO).
    """
    fig, ax = plt.subplots(figsize=(12, 7), dpi=600)
    real = nuc[nuc.get("secondary_only", False) != True] if "secondary_only" in nuc.columns else nuc
    conds = order_conditions(real["condition"].dropna().unique().tolist(), condition_order or [])
    if "WT" not in conds or "KO" not in conds:
        ax.text(0.5, 0.5, "Need WT and KO to compute log2(KO/WT)",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        _relabel_fig(fig)
        fig.savefig(out_path, bbox_inches="tight", dpi=600)
        plt.close(fig)
        return
    metrics = [
        ("Introns total",     "n_spots_rna1"),
        ("Introns nuclear",   "nuclear_spot_count"),
        ("Introns cyto",      "cyto_spot_count"),
        ("Exons total",       "n_spots_rna2"),
        ("Exons nuclear",     "nuclear_spot_count_rna2"),
        ("Exons cyto",        "cyto_spot_count_rna2"),
        ("Nuc Introns+Exons coloc", "n_nuclear_rna1_rna2_overlap_per_nucleus"),
    ]
    labels, log2fcs = [], []
    for lbl, col in metrics:
        if col not in real.columns: continue
        wt = real.loc[real["condition"] == "WT", col].dropna()
        ko = real.loc[real["condition"] == "KO", col].dropna()
        if not len(wt) or not len(ko): continue
        wt_mean = max(float(wt.mean()), 0.01)  # avoid log2(0)
        ko_mean = max(float(ko.mean()), 0.01)
        log2fc = float(np.log2(ko_mean / wt_mean))
        labels.append(lbl)
        log2fcs.append(log2fc)
    colors = ["#D32F2F" if v > 0 else "#1976D2" for v in log2fcs]
    y = np.arange(len(labels))
    bars = ax.barh(y, log2fcs, color=colors, edgecolor="black", linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("log2(KO mean / WT mean)  per cell", fontsize=12)
    for bar, v in zip(bars, log2fcs):
        x = bar.get_width()
        ax.text(x + (0.05 if x >= 0 else -0.05), bar.get_y() + bar.get_height()/2,
                f"{v:+.2f} ({2**v:.2f}×)", ha="left" if x >= 0 else "right",
                va="center", fontsize=10, fontweight="bold")
    # Annotate direction
    ax.text(0.98, 0.97, "→ ↑ KO\n→ ↓ KO", transform=ax.transAxes,
            ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"))
    ax.set_title("KO / WT fold-change across compartments", fontsize=15, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")
    _relabel_fig(fig)
    _final_layout_polish(fig, has_subtitle=False)
    fig.savefig(out_path, bbox_inches="tight", dpi=600)
    plt.close(fig)


def render_headline_property_shifts(
    out_path: Path, nuc: pd.DataFrame, spots: pd.DataFrame,
    condition_order: list[str] | None,
) -> None:
    """Headline biology figure: spot property shifts per compartment.

    3-panel: (count, intensity, size) × (nuclear, cytoplasmic) × (Introns,
    Exons). Each panel shows KO/WT log2 fold-change as bars — at-a-glance
    answer to "are there fewer spots, or just dimmer/smaller ones?"
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), dpi=600)
    real = nuc[nuc.get("secondary_only", False) != True] if "secondary_only" in nuc.columns else nuc
    real_spots = spots[spots.get("secondary_only", False) != True] if "secondary_only" in spots.columns else spots
    if "WT" not in real["condition"].values or "KO" not in real["condition"].values:
        for ax in axes:
            ax.text(0.5, 0.5, "Need WT and KO", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
        fig.savefig(out_path, bbox_inches="tight", dpi=600)
        plt.close(fig); return

    # Panel data: (label, getter(real_or_spots, cond) -> array)
    def per_nuc(col, cond):
        if col not in real.columns: return np.array([])
        return real.loc[real["condition"] == cond, col].dropna().to_numpy()

    def per_spot(channel, compartment, value_col, cond):
        if not len(real_spots) or value_col not in real_spots.columns:
            return np.array([])
        sub = real_spots
        if "channel" in sub.columns:
            sub = sub[sub["channel"] == channel]
        if compartment == "nuclear" and "in_nucleus" in sub.columns:
            sub = sub[sub["in_nucleus"].astype(bool) == True]
        elif compartment == "cyto" and "in_cytoplasm" in sub.columns:
            sub = sub[sub["in_cytoplasm"].astype(bool) == True]
        return sub.loc[sub["condition"] == cond, value_col].dropna().to_numpy()

    # Panel 1: spots per cell (count) — use per_nuc on count columns
    # Panel 2: per-spot peak intensity — use per_spot
    # Panel 3: per-spot diameter — use per_spot
    panels = [
        ("Spots per cell", "count", [
            ("Introns nuc",  lambda c: per_nuc("nuclear_spot_count", c)),
            ("Introns cyto", lambda c: per_nuc("cyto_spot_count", c)),
            ("Exons nuc",    lambda c: per_nuc("nuclear_spot_count_rna2", c)),
            ("Exons cyto",   lambda c: per_nuc("cyto_spot_count_rna2", c)),
        ]),
        ("Per-spot peak intensity", "intensity", [
            ("Introns nuc",  lambda c: per_spot("rna1", "nuclear",  "spot_peak_intensity", c)),
            ("Introns cyto", lambda c: per_spot("rna1", "cyto",     "spot_peak_intensity", c)),
            ("Exons nuc",    lambda c: per_spot("rna2", "nuclear",  "spot_peak_intensity", c)),
            ("Exons cyto",   lambda c: per_spot("rna2", "cyto",     "spot_peak_intensity", c)),
        ]),
        ("Per-spot diameter (µm)", "size", [
            ("Introns nuc",  lambda c: per_spot("rna1", "nuclear",  "spot_diameter_um", c)),
            ("Introns cyto", lambda c: per_spot("rna1", "cyto",     "spot_diameter_um", c)),
            ("Exons nuc",    lambda c: per_spot("rna2", "nuclear",  "spot_diameter_um", c)),
            ("Exons cyto",   lambda c: per_spot("rna2", "cyto",     "spot_diameter_um", c)),
        ]),
    ]
    for ax, (title, key, rows) in zip(axes, panels):
        labels, log2fcs = [], []
        for lbl, get_fn in rows:
            wt = get_fn("WT")
            ko = get_fn("KO")
            if not len(wt) or not len(ko):
                labels.append(lbl); log2fcs.append(np.nan); continue
            wt_mean = max(float(wt.mean()), 1e-6)
            ko_mean = max(float(ko.mean()), 1e-6)
            log2fcs.append(float(np.log2(ko_mean / wt_mean)))
            labels.append(lbl)
        colors = ["#D32F2F" if (not np.isnan(v) and v > 0) else "#1976D2" for v in log2fcs]
        y = np.arange(len(labels))
        bars = ax.barh(y, [0 if np.isnan(v) else v for v in log2fcs],
                       color=colors, edgecolor="black", linewidth=1.2)
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=10)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_xlabel("log2(KO / WT)", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis="x")
        for bar, v in zip(bars, log2fcs):
            if np.isnan(v):
                ax.text(0, bar.get_y() + bar.get_height()/2, "  n/a",
                        va="center", fontsize=9, color="gray")
                continue
            x = bar.get_width()
            ax.text(x + (0.03 if x >= 0 else -0.03), bar.get_y() + bar.get_height()/2,
                    f"{v:+.2f} ({2**v:.2f}×)",
                    ha="left" if x >= 0 else "right", va="center", fontsize=9, fontweight="bold")
    fig.suptitle("Spot property shifts (count, intensity, size) by compartment — KO vs WT",
                 fontsize=15, fontweight="bold")
    fig.text(0.5, 0.02, "Are there fewer spots, or just dimmer/smaller ones? Each panel answers a different aspect.",
             ha="center", fontsize=10, style="italic", color="#555")
    _relabel_fig(fig)
    _final_layout_polish(fig, has_subtitle=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=600)
    plt.close(fig)


def render_pi_focus_overview_panel(out_path: Path, nuc: pd.DataFrame,
                                    spots: pd.DataFrame,
                                    condition_order: list[str] | None) -> None:
    """104_PI_FOCUS_overview_panel.

    3×2 publication-ready overview the PI sees at a glance. Cells:
      (0,0) spot counts: RNA1 Nuclear  |  (0,1) spot counts: RNA1 Cytoplasmic
      (1,0) spot counts: RNA2 Nuclear  |  (1,1) spot counts: RNA2 Cytoplasmic
      (2,0) RNA1 localization stack    |  (2,1) RNA2 localization stack
    """
    fig = plt.figure(figsize=(24, 18), dpi=600)
    from matplotlib.gridspec import GridSpec as _GS
    gs = _GS(3, 2, figure=fig, hspace=0.55, wspace=0.30,
              left=0.06, right=0.97, top=0.93, bottom=0.07)
    axes = [[fig.add_subplot(gs[i, j]) for j in range(2)] for i in range(3)]
    # Rows 0-1: spot counts per compartment, one (channel, compartment) per cell.
    col_map = {
        ("rna1", "nuclear"):      "nuclear_spot_count",
        ("rna1", "cytoplasmic"):  "cyto_spot_count",
        ("rna2", "nuclear"):      "nuclear_spot_count_rna2",
        ("rna2", "cytoplasmic"):  "cyto_spot_count_rna2",
    }
    for r, (ch_id, ch_disp) in enumerate(_PI_FOCUS_CHANNELS):
        for c, (cp_id, cp_disp) in enumerate(_PI_FOCUS_COMPARTMENTS):
            ax = axes[r][c]
            col = col_map[(ch_id, cp_id)]
            ok = _pi_focus_box_strip(ax, nuc, col, condition_order,
                                      ylabel=f"{cp_disp} spots per cell")
            if not ok:
                ax.set_visible(False); continue
            ax.set_title(_wrap_title(f"{ch_disp} — {cp_disp} spots per cell"),
                         fontsize=11)
    # Row 2: localization stacks per channel.
    if (spots is not None and len(spots) and "in_nucleus" in spots.columns
            and "condition" in spots.columns):
        sp = spots.copy()
        sp["in_nucleus"] = pd.to_numeric(sp["in_nucleus"], errors="coerce").fillna(0).astype(int)
        for c, (ch_id, ch_disp) in enumerate(_PI_FOCUS_CHANNELS):
            ax = axes[2][c]
            if "channel" in sp.columns:
                sub_ch = sp[sp["channel"] == ch_id]
            elif ch_id == "rna1":
                sub_ch = sp
            else:
                sub_ch = sp.iloc[0:0]
            if sub_ch.empty:
                ax.set_visible(False); continue
            conds_in_data = sub_ch["condition"].dropna().unique().tolist()
            conditions = order_conditions(conds_in_data, condition_order or [])
            if not conditions:
                ax.set_visible(False); continue
            nuc_pcts, cyto_pcts = [], []
            for cond in conditions:
                cs = sub_ch[sub_ch["condition"] == cond]
                n_total = len(cs)
                if n_total == 0:
                    nuc_pcts.append(0.0); cyto_pcts.append(0.0); continue
                n_nuc = int((cs["in_nucleus"] == 1).sum())
                nuc_pcts.append(100.0 * n_nuc / n_total)
                cyto_pcts.append(100.0 - 100.0 * n_nuc / n_total)
            x = np.arange(len(conditions))
            ax.bar(x, nuc_pcts, color=COLOR_NUCLEAR, label="Nuclear")
            ax.bar(x, cyto_pcts, bottom=nuc_pcts, color=COLOR_CYTOPLASMIC,
                   label="Cytoplasmic")
            ax.set_xticks(x); ax.set_xticklabels(conditions)
            ax.set_ylabel("% of spots")
            ax.set_ylim(0, 100)
            ax.set_title(_wrap_title(f"{ch_disp} — nuclear vs cytoplasmic"),
                         fontsize=11)
            ax.legend(loc="lower right", fontsize=8)
            ax.grid(True, alpha=0.3, axis="y")
    else:
        for c in range(2):
            axes[2][c].set_visible(False)
    fig.suptitle("PI focus overview — spot counts and localization across conditions",
                 fontsize=16, fontweight="bold", y=0.98)
    fig.text(0.5, 0.02,
             "Headline PI panel: per-compartment spot counts (rows 0-1) and "
             "nuclear/cytoplasmic localization stacks (row 2) for each channel.",
             ha="center", fontsize=9, style="italic", color="#555")
    _relabel_fig(fig)
    _final_layout_polish(fig, has_subtitle=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=600)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2026-05-24 Brian — Whole-nucleus + N/C pixel-intensity panels.
#
# Spot-detection-INDEPENDENT measures from nuclei_metrics.csv per-nucleus
# pixel statistics. Mirrors fig15 / fig16 in the cross-condition deck so
# every per-run figures/ subfolder carries the same "is nuclear retention
# happening at the pixel level?" readout that the cross-condition deck
# already showed for the 3-gene comparison.
#
# Both functions:
#   - Filter to real-probe nuclei (secondary_only == False) before plotting
#   - 2 panels side-by-side: Intron Spots channel | Exon Spots channel
#     (column names rna_* = rna1 = Intron, rna2_* = Exon by Brian's
#     wavelength → channel convention; the panel TITLES are hard-coded to
#     "Intron Spots channel" / "Exon Spots channel" so the per-run figure
#     reads identically to the cross-condition deck regardless of any
#     run_config label substitution)
#   - TNG palette (#4a78a8 WT, #c89048 KO) hard-coded so per-run figures
#     match the cross-condition deck exactly; other conditions fall back
#     to _color_for_condition
#   - 600 DPI, wide figure (16x8 in), large fonts (~20pt)
#   - Per-nucleus dots + bar (mean) + violin backdrop, condition order WT-then-KO
#   - Per-image mean diamonds overlaid (biological-replicate spread)
# ---------------------------------------------------------------------------

# TNG palette pinned to match fig15 / fig16 in the cross-condition deck.
# Hard-coded here (not pulled from _color_for_condition) so the per-run
# figure carries the IDENTICAL color scheme of the cross-condition deck
# even though the rest of single_condition_plots.py uses Okabe-Ito blue
# (#0072B2) for WT and vermillion (#D55E00) for KO.
_PIXEL_INTENSITY_COLOR_WT = "#4a78a8"  # TNG steel blue
_PIXEL_INTENSITY_COLOR_KO = "#c89048"  # TNG warm gold


def _pixel_intensity_color_for(cond: str, fallback_idx: int) -> str:
    """Resolve a color for a condition in the pixel-intensity panels. WT/KO
    keep the pinned TNG steel-blue/warm-gold (so the original cross-condition
    WT/KO decks stay byte-identical). Sec-only stays neutral gray. Anything
    else (NT ASO / KD ASO / OE / ...) uses the deck-wide condition FAMILY
    color (KD=Blues / NT=Reds / Sec=Greys) so this panel reads in the SAME
    scheme as every other by-condition figure in the rna_only deck.

    2026-05-26 (publication polish): previously the non-WT/KO fallback was
    _color_for_condition (Okabe-Ito green/orange), which made NT/KD a
    different color here than in the SuperPlots."""
    if isinstance(cond, str):
        c = cond.strip()
        if c == "WT":
            return _PIXEL_INTENSITY_COLOR_WT
        if c == "KO":
            return _PIXEL_INTENSITY_COLOR_KO
        if c == SEC_ONLY_CONDITION:
            return COLOR_SEC_ONLY
    return _condition_family_base_color(cond, fallback_idx)


def _pixel_intensity_two_panel(
    out_path: Path,
    nuc: pd.DataFrame,
    *,
    col_intron: str,
    col_exon: str,
    suptitle: str,
    ylabel: str,
    condition_order: list[str] | None,
    cap_99: bool = True,
    channel_titles: tuple[str, str] = ("Intron Spots channel", "Exon Spots channel"),
) -> None:
    """Two-panel SuperPlot-style figure (Intron Spots | Exon Spots) for a
    per-nucleus pixel-intensity metric. Real-probe nuclei only.

    2026-05-26 (publication polish): ``channel_titles`` lets the single-channel
    rna_only callers pass a meaningful panel title (the RNA label) instead of
    the two-channel-deck default. When the second (exon) column is absent or
    all-NaN, the figure collapses to a SINGLE centered panel rather than
    rendering an empty "Exon Spots channel" axis.

    Mirrors the cross-condition-deck fig15/fig16 convention adapted to
    per-run (single-dataset) context:
      - x-axis: per-condition (WT, KO, ...) — condition order respected
      - violin backdrop per condition (distribution shape)
      - per-nucleus dots jittered (per-cell spread)
      - bar showing the per-condition mean (over real-probe nuclei)
      - per-image mean diamonds overlaid (biological-replicate spread)
      - title: neutral descriptive
      - TNG palette WT/KO (matches cross-condition deck)
    """
    # Filter to real-probe nuclei (secondary_only == False).
    df_all = nuc.copy()
    if "secondary_only" in df_all.columns:
        sec_mask = df_all["secondary_only"].astype(str).str.lower() == "true"
        df_all = df_all[~sec_mask]
    # Also drop any condition that was remapped to sec-only at load time.
    if "condition" in df_all.columns:
        df_all = df_all[df_all["condition"] != SEC_ONLY_CONDITION]

    # Decide single- vs two-panel: collapse to one centered panel when the
    # exon column is missing or all-NaN (the single-channel rna_only case).
    def _has_data(c):
        if c not in df_all.columns:
            return False
        return pd.to_numeric(df_all[c], errors="coerce").notna().any()
    two_panel = _has_data(col_exon)

    # 2026-05-24 Brian: 16x8 wide / 600 DPI to match the per-run deck.
    if two_panel:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=600)
        panels = [(col_intron, channel_titles[0], axes[0]),
                  (col_exon,   channel_titles[1], axes[1])]
    else:
        fig, ax_one = plt.subplots(1, 1, figsize=(9, 8), dpi=600)
        panels = [(col_intron, channel_titles[0], ax_one)]

    drew_any = False
    rng = np.random.RandomState(20260524)

    for col, ch_title, ax in panels:
        # Per-panel data: drop NaN per-nucleus values.
        if col not in df_all.columns or "condition" not in df_all.columns:
            ax.set_visible(False); continue
        df_col = df_all[["condition", "image", col]].copy() \
            if "image" in df_all.columns else df_all[["condition", col]].copy()
        df_col[col] = pd.to_numeric(df_col[col], errors="coerce")
        df_col = df_col[df_col[col].notna()]
        if df_col.empty:
            ax.set_visible(False); continue

        # Condition order: respect run_config CONDITION_ORDER, then fall
        # back to alphabetical (with sec-only pinned last — already
        # stripped above). WT-then-KO is the universal default in Brian's
        # runs; CONDITION_ORDER in run_config enforces this when set.
        conds_in_data = df_col["condition"].dropna().unique().tolist()
        conditions = order_conditions(conds_in_data, condition_order or [])
        if not conditions:
            ax.set_visible(False); continue

        # Optional 99th-percentile y-cap so a few super-bright nuclei
        # don't compress the rest of the distribution.
        cap = None
        if cap_99:
            pool = df_col[col].values
            if len(pool):
                cap = float(np.nanpercentile(pool, 99))

        positions = list(range(1, len(conditions) + 1))
        cond_colors = [_pixel_intensity_color_for(c, i) for i, c in enumerate(conditions)]

        # ---- 1. Violin backdrop (distribution shape) ----
        violin_data = []
        for c in conditions:
            vals = df_col.loc[df_col["condition"] == c, col].dropna().values
            if cap is not None:
                vals = vals[vals <= cap]
            violin_data.append(vals if len(vals) > 1 else np.array([np.nan, np.nan]))
        try:
            vp = ax.violinplot(violin_data, positions=positions, widths=0.85,
                                showmeans=False, showmedians=False, showextrema=False)
            for body, color in zip(vp["bodies"], cond_colors):
                body.set_facecolor(color); body.set_edgecolor("#202020")
                body.set_alpha(0.18); body.set_linewidth(0.8); body.set_zorder(1)
        except Exception:
            pass

        # ---- 2. Per-condition mean BAR (thin, sits behind dots) ----
        means_per_cond = []
        for c in conditions:
            vals = df_col.loc[df_col["condition"] == c, col].dropna().values
            means_per_cond.append(float(np.mean(vals)) if len(vals) else float("nan"))
        ax.bar(positions, means_per_cond, width=0.55,
                color=cond_colors, alpha=0.35,
                edgecolor="#202020", linewidth=1.2, zorder=2)

        # ---- 3. Per-nucleus dots (per-cell spread) ----
        for i, c in zip(positions, conditions):
            color = cond_colors[i - 1]
            vals = df_col.loc[df_col["condition"] == c, col].dropna().values
            if not len(vals):
                continue
            vals_plot = vals if cap is None else vals[vals <= cap]
            if not len(vals_plot):
                continue
            # Scale dot size + alpha by N so dense conditions don't ink-out.
            s_dot = 14 if len(vals_plot) < 400 else (9 if len(vals_plot) < 2000 else 5)
            a_dot = 0.32 if len(vals_plot) < 400 else (0.16 if len(vals_plot) < 2000 else 0.08)
            jitter = (rng.random(len(vals_plot)) - 0.5) * 0.32
            ax.scatter(np.full(len(vals_plot), i) + jitter, vals_plot,
                       s=s_dot, alpha=a_dot, color=color,
                       edgecolor="none", zorder=3)

        # ---- 4. Per-image mean diamonds (biological-replicate spread) ----
        if "image" in df_col.columns:
            for i, c in zip(positions, conditions):
                color = cond_colors[i - 1]
                img_means = (df_col[df_col["condition"] == c]
                             .groupby("image")[col].mean().values)
                if not len(img_means):
                    continue
                # Cap displayed diamonds at the y-cap so they sit visibly
                # inside the panel rather than off the top edge.
                img_disp = img_means if cap is None else np.minimum(img_means, cap)
                jitter_im = (rng.random(len(img_disp)) - 0.5) * 0.20
                ax.scatter(np.full(len(img_disp), i) + jitter_im, img_disp,
                           s=180, color=color, edgecolor="#1f1f1f",
                           linewidth=1.6, marker="D", zorder=6)

        # ---- Axis cosmetics ----
        ax.set_xticks(positions)
        # 2026-05-24 Brian: include n=cells per condition under the label
        # so the dot-cloud density is interpretable at a glance.
        ax.set_xticklabels(
            [f"{_display_condition(c)}\n(n={int((df_col['condition']==c).sum())} cells)"
             for c in conditions],
            fontsize=18,
        )
        if cap is not None:
            ax.set_ylim(0, cap * 1.10)
        ax.tick_params(axis="y", labelsize=16)
        ax.set_title(ch_title, fontsize=20, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=18)
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        ax.set_axisbelow(True)
        # Pairwise stats on per-image means (Welch + MWU), vs the reference
        # column — consistent with every other by-condition panel.
        gm = {c: (df_col[df_col["condition"] == c].groupby("image")[col].mean()
                  .dropna().tolist() if "image" in df_col.columns else [])
              for c in conditions}
        _annotate_pairwise_brackets(
            ax, gm, conditions,
            x_centers={c: p for c, p in zip(conditions, positions)}, fontsize=12)
        drew_any = True

    # Suptitle uses the neutral descriptive text the caller passed in.
    fig.suptitle(suptitle, fontsize=20, fontweight="bold", y=0.995)
    # Legend block (WT/KO color swatches + diamond explanation), placed
    # below the panels so it doesn't crowd either axis.
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_handles = []
    legend_labels = []
    if drew_any:
        # Only add WT / KO entries that actually appear in the data.
        any_conditions = []
        for col in (col_intron, col_exon):
            if col in df_all.columns:
                sub = df_all[df_all[col].notna()] if col in df_all.columns else df_all
                any_conditions += sub["condition"].dropna().unique().tolist()
        seen = []
        for c in any_conditions:
            if c not in seen:
                seen.append(c)
        for i, c in enumerate(seen):
            legend_handles.append(Patch(facecolor=_pixel_intensity_color_for(c, i),
                                         edgecolor="#202020", alpha=0.55,
                                         label=_display_condition(c)))
            legend_labels.append(_display_condition(c))
        legend_handles.append(Line2D([0], [0], marker="D", color="w",
                                      markerfacecolor="#888", markeredgecolor="#1f1f1f",
                                      markersize=10, label="per-image mean"))
        legend_labels.append("per-image mean")
        legend_handles.append(Line2D([0], [0], marker="o", color="w",
                                      markerfacecolor="#888", markeredgecolor="none",
                                      markersize=7, label="per nucleus"))
        legend_labels.append("per nucleus")
        fig.legend(legend_handles, legend_labels,
                   loc="lower center", ncol=max(2, len(legend_labels)),
                   frameon=False, fontsize=14, bbox_to_anchor=(0.5, -0.02))
    _relabel_fig(fig)
    # Reserve bottom margin for legend; do NOT use tight_layout (which
    # would clip the legend) — manually adjust spacing instead.
    try:
        fig.subplots_adjust(left=0.07, right=0.97, top=0.90, bottom=0.18, wspace=0.20)
    except Exception:
        pass
    # 2026-06-06 Brian: per-bracket labels now carry ONLY a star; gather the
    # per-comparison stat detail from BOTH panels' stashed records into ONE
    # footnote at the very bottom (panel-tagged), below the WT/KO legend. The
    # star/test legend sentence is identical across panels, so we print it once
    # and then append each panel's tagged comparison list. bbox_inches="tight"
    # includes the text in the saved canvas. Degrades gracefully if no brackets.
    try:
        import textwrap as _tw
        _LEG_SENT = ("Significance star on the bracket = Welch's t-test on "
                     "per-image (FoV) means (primary); Mann–Whitney U (MWU) "
                     "reported as a secondary check. *** p<0.001, ** p<0.01, "
                     "* p<0.05, ns = not significant. Comparisons vs the "
                     "reference (left-most) condition.")
        _foot_parts = []
        for _ax in fig.axes:
            _full = _superplot_stats_footnote(_ax)
            if not _full:
                continue
            _tag = (_ax.get_title() or "").strip()
            # Strip the shared legend sentence; keep only this panel's
            # comparison list, tagged with the panel (Intron/Exon) title.
            _detail = _full.split(_LEG_SENT, 1)[-1].strip()
            _foot_parts.append((f"[{_tag}] " if _tag else "") + _detail)
        if _foot_parts:
            _w_in, _h_in = fig.get_size_inches()
            _wrap_w = int(max(90, 9.0 * _w_in))
            _full_text = _LEG_SENT + "   " + "    ".join(_foot_parts)
            _foot_txt = "\n".join(_tw.wrap(_full_text, width=_wrap_w))
            fig.text(0.5, 0.005, _foot_txt, ha="center", va="bottom",
                     fontsize=8, color="#555555", linespacing=1.25)
    except Exception:
        pass
    fig.savefig(out_path, bbox_inches="tight", dpi=600)
    plt.close(fig)


def render_whole_nucleus_pixel_intensity(
    out_path: Path, nuc: pd.DataFrame,
    condition_order: list[str] | None,
) -> None:
    """Whole-nucleus mean pixel intensity per nucleus, by condition.

    2-panel figure (Intron Spots | Exon Spots). Reads
    ``rna_nuclear_mean`` (rna1 / Intron) and ``rna2_nuclear_mean`` (rna2 /
    Exon) per-nucleus columns from nuclei_metrics.csv. Spot-detection-
    independent — measures the raw pixel mean inside each segmented
    nucleus, so the figure renders even on conditions where zero spots
    were called. Real-probe nuclei only (secondary_only == False).

    Mirrors fig15 in the cross-condition deck so per-run + cross-condition
    figures answer the same question with the same conventions.
    """
    _pixel_intensity_two_panel(
        out_path, nuc,
        col_intron="rna_nuclear_mean",
        col_exon="rna2_nuclear_mean",
        suptitle=("Whole-nucleus mean RNA pixel intensity per nucleus, by condition "
                  "(spot-detection-independent)"),
        ylabel="Mean RNA pixel intensity inside nucleus (a.u.)",
        condition_order=condition_order,
        cap_99=True,
        # "RNA1"/"RNA2" tokens are relabeled to the run's channel names by
        # _relabel_fig (e.g. MIAT-640) so the panel title is meaningful for any
        # probe — not the cross-condition deck's hard-coded "Intron Spots".
        channel_titles=("RNA1 channel", "RNA2 channel"),
    )


def render_nuc_cyto_pixel_intensity_ratio(
    out_path: Path, nuc: pd.DataFrame,
    condition_order: list[str] | None,
) -> None:
    """Nuclear / cytoplasmic pixel intensity ratio per nucleus, by condition.

    2-panel figure (Intron Spots | Exon Spots). Reads ``rna_nc_ratio``
    (rna1 / Intron) and ``rna2_nc_ratio`` (rna2 / Exon) per-nucleus
    columns from nuclei_metrics.csv. Higher values = more nuclear
    retention at the pixel level. Spot-detection-independent. Real-probe
    nuclei only (secondary_only == False).

    Falls back to computing the ratio from rna_nuclear_mean /
    rna_cytoplasmic_mean (and the rna2 analogues) when the pre-computed
    ratio columns are missing (legacy CSVs).

    Mirrors fig16 in the cross-condition deck.
    """
    df = nuc.copy()
    # Backfill rna_nc_ratio / rna2_nc_ratio if absent.
    if "rna_nc_ratio" not in df.columns and "rna_nuclear_mean" in df.columns \
            and "rna_cytoplasmic_mean" in df.columns:
        df["rna_nc_ratio"] = (
            pd.to_numeric(df["rna_nuclear_mean"], errors="coerce")
            / pd.to_numeric(df["rna_cytoplasmic_mean"], errors="coerce")
            .replace(0, np.nan)
        )
    if "rna2_nc_ratio" not in df.columns and "rna2_nuclear_mean" in df.columns \
            and "rna2_cytoplasmic_mean" in df.columns:
        df["rna2_nc_ratio"] = (
            pd.to_numeric(df["rna2_nuclear_mean"], errors="coerce")
            / pd.to_numeric(df["rna2_cytoplasmic_mean"], errors="coerce")
            .replace(0, np.nan)
        )
    _pixel_intensity_two_panel(
        out_path, df,
        col_intron="rna_nc_ratio",
        col_exon="rna2_nc_ratio",
        suptitle=("Nuclear / cytoplasmic RNA pixel-intensity ratio per nucleus, by condition "
                  "(spot-detection-independent)"),
        ylabel="Nuclear / cytoplasmic RNA pixel-intensity ratio (per nucleus)",
        condition_order=condition_order,
        cap_99=True,
        channel_titles=("RNA1 channel", "RNA2 channel"),
    )


# ---------------------------------------------------------------------------
# 2026-05-18 Brian — composition COMPANIONS to existing raw-count figures.
# For each raw-count figure where the underlying biological question is
# really "what fraction of cells / spots / nuclei does X?", add a `b`-suffix
# panel that shows the per-condition composition. The raw figures stay as
# they are — the b-suffix ones sit next to them and make WT-vs-KO shifts
# visually obvious instead of "compare two histograms by eye".
# ---------------------------------------------------------------------------


# 2026-05-19 Brian: module-level toggles for the summary / core / coloc
# overview panels. When a panel is rendering N pre-existing figures into a
# grid of small cells, each cell is too cramped for (a) its italic
# explanatory subtitle and (b) the per-subplot title is too large at 14pt.
# These toggles let the overview-panel render set "compact mode" before
# calling the per-figure helpers, then restore on exit.
#
#   _COMPACT_SUPPRESS_SUBTITLES — when True, _add_subtitle is a no-op.
#   _COMPACT_TITLE_FONTSIZE     — when not None, set on every per-axis title
#                                  after the helper sets it. Default None.
_COMPACT_SUPPRESS_SUBTITLES = False
_COMPACT_TITLE_FONTSIZE = None


def _add_subtitle(ax, text: str) -> None:
    """Render a short italic explanatory subtitle below the host axis.

    Hijacks the bottom margin of the figure: we expand the bottom margin
    (subplots_adjust(bottom=...)) and place the text in the reserved
    space using axis-bbox-relative coordinates. This lands the subtitle
    BELOW the multi-line x-tick labels (e.g. "WT\\n(n=146 cells)") instead
    of on top of them, which was the symptom in the first render.

    2026-05-18 Brian (round 2): wrap the subtitle to ~95 chars so long
    descriptions break onto a 2nd line instead of running off the figure
    or being clipped by ``bbox_inches='tight'``. When wrapping triggers,
    grow the reserved bottom margin (0.18 → 0.25) so the 2-line subtitle
    has somewhere to sit. Per-spec values from the title-wrap task.

    2026-05-19 Brian: honor _COMPACT_SUPPRESS_SUBTITLES — when an overview
    panel (figure 56, 97, 98) is rendering 6 plots into one figure, the
    per-subplot subtitle clutters cramped cells. The overview-panel render
    sets the toggle before calling these helpers."""
    if _COMPACT_SUPPRESS_SUBTITLES:
        return
    try:
        fig = ax.figure
        wrapped, n_lines = _wrap_subtitle(text, width=95)
        # 2026-05-26 (publication polish): DON'T call subplots_adjust here.
        # _final_layout_polish runs tight_layout(rect=...) AFTER this and would
        # overwrite any margin we set, re-packing the axes on top of the
        # subtitle (the overlap-with-x-tick-labels bug). Instead, record the
        # bottom band this subtitle needs on the axis; _final_layout_polish
        # reads the largest request across all axes and carves that band out of
        # the tight_layout rect BEFORE packing. We then place the text inside
        # that reserved band in figure coords.
        #
        # Band sizing: each subtitle line ~0.030 of fig height, plus 0.085
        # clearance below the (possibly rotated, possibly two-line) x-tick
        # labels so the italic text always lands beneath them.
        pad = 0.085 + 0.030 * n_lines
        try:
            prev = float(getattr(ax, "_subtitle_pad", 0.0) or 0.0)
        except Exception:
            prev = 0.0
        ax._subtitle_pad = max(prev, pad)
        # Anchor the subtitle to the AXIS using axes-fraction coordinates with
        # a negative y, so it tracks the axis through tight_layout(rect=...)
        # and always sits a fixed distance BELOW the (rotated, possibly
        # two-line) x-tick labels. This is robust for both the standalone
        # single-axis figures and the combined multi-axis overview panel,
        # where a fixed figure-y would stack every subtitle at the page
        # bottom. The reserved band (ax._subtitle_pad, honored by
        # _final_layout_polish) guarantees there is room for it.
        y_off = -0.20 if n_lines == 1 else -0.24
        ax.text(0.5, y_off, wrapped, transform=ax.transAxes,
                ha="center", va="top", fontsize=8, style="italic",
                color="#444", wrap=True, clip_on=False)
    except Exception:
        # Don't crash a figure render over a subtitle.
        pass


def _bar_pct_cells_with_spots(ax, nuc: pd.DataFrame, channel: str,
                                condition_order: list[str] | None = None) -> None:
    """01b / 02b: grouped bar of % cells with ≥1, ≥5, ≥10 spots, per condition.

    Reads the per-cell count column directly (rna_spot_count for RNA1,
    n_spots_rna2 for RNA2) so the bar matches the underlying nuclei_metrics
    rows exactly — no dependency on summary-level frac_nuclei_with_ge_*
    aggregation."""
    if channel == "rna1":
        count_col = "rna_spot_count"
    else:
        count_col = "n_spots_rna2" if "n_spots_rna2" in nuc.columns else "rna_spot_count"
    if count_col not in nuc.columns or "condition" not in nuc.columns:
        ax.set_visible(False); return
    df = nuc.copy()
    df[count_col] = pd.to_numeric(df[count_col], errors="coerce")
    df = df[df[count_col].notna()]
    if df.empty:
        ax.set_visible(False); return
    conds_in = df["condition"].dropna().unique().tolist()
    conds = order_conditions(conds_in, condition_order or [])
    thresholds = [1, 5, 10]
    width = 0.25
    x = np.arange(len(conds))
    for j, t in enumerate(thresholds):
        pcts = []
        for c in conds:
            sub = df[df["condition"] == c]
            pcts.append(100.0 * (sub[count_col] >= t).mean() if len(sub) else 0.0)
        # ≥1 = sky blue, ≥5 = green (CATEGORICAL_GREEN, env-overridable), ≥10 = yellow
        _thr_col = CATEGORICAL_GREEN if (j + 1) == 2 else OKABE_ITO[j + 1]
        bars = ax.bar(x + (j - 1) * width, pcts, width=width,
                       label=f"≥{t} spots", edgecolor="black", linewidth=0.5,
                       color=_thr_col)
        for xi, p in zip(x + (j - 1) * width, pcts):
            ax.text(xi, p + 1.0, f"{p:.0f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{_display_condition(c)}\n(n={int((df['condition']==c).sum())} cells)" for c in conds])
    ax.set_ylim(0, 105)
    ax.set_ylabel("% of cells")
    ax.set_title(_wrap_title(f"% of cells with ≥1, ≥5, ≥10 {channel.upper()} spots — by condition"))
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    _add_subtitle(ax,
        f"Reads each cell's {channel.upper()} spot count and asks: does it pass each threshold? "
        f"Bars show the proportion of cells in each condition that do.")


def _bar_spot_count_bin_composition(ax, nuc: pd.DataFrame, count_col: str,
                                      title: str, axis_label: str,
                                      condition_order: list[str] | None = None,
                                      subtitle: str | None = None) -> None:
    """08b/09b/10b/11b helper: per condition, % of cells in each spot-count
    bin (0, 1-4, 5-9, 10+). Renders as a stacked horizontal bar that sums
    to 100% per condition so cellular heterogeneity is comparable across
    WT / KO / sec-only at a glance."""
    if count_col not in nuc.columns or "condition" not in nuc.columns:
        ax.set_visible(False); return
    df = nuc.copy()
    df[count_col] = pd.to_numeric(df[count_col], errors="coerce")
    df = df[df[count_col].notna()]
    if df.empty:
        ax.set_visible(False); return
    bins = [(0, 0, "0", OKABE_ITO[7]),            # black
            (1, 4, "1-4", OKABE_ITO[1]),          # sky blue
            (5, 9, "5-9", CATEGORICAL_GREEN),     # green (env-overridable)
            (10, np.inf, "10+", OKABE_ITO[5])]    # vermillion
    conds_in = df["condition"].dropna().unique().tolist()
    conds = order_conditions(conds_in, condition_order or [])
    y = np.arange(len(conds))
    bar_h = 0.65

    for i, cond in enumerate(conds):
        sub = df[df["condition"] == cond]
        n = len(sub)
        # 2026-06-05 Brian: condition-colored bar outline REMOVED (redundant
        # with the condition label beside each bar; competed with fill colors).
        edge_c = "none"
        if n == 0:
            ax.barh(y[i], 100.0, height=bar_h, color="#dddddd",
                    edgecolor=edge_c, linewidth=COND_EDGE_LINEWIDTH)
            ax.text(50.0, y[i], f"{_display_condition(cond)}: no cells", ha="center", va="center",
                    fontsize=10, color="#444")
            continue
        cursor = 0.0
        for lo, hi, lbl, color in bins:
            mask = (sub[count_col] >= lo) & (sub[count_col] <= hi)
            pct = 100.0 * mask.mean()
            if pct <= 0:
                continue
            ax.barh(y[i], pct, left=cursor, height=bar_h, color=color,
                    edgecolor=edge_c, linewidth=COND_EDGE_LINEWIDTH,
                    label=lbl if i == 0 else None)
            if pct >= 7:
                ax.text(cursor + pct / 2.0, y[i], f"{pct:.0f}%",
                        ha="center", va="center", fontsize=10,
                        color="white" if lbl in ("0", "10+") else "black",
                        fontweight="bold")
            cursor += pct
    ax.set_yticks(y)
    ax.set_yticklabels([f"{_display_condition(c)}\n(n={int((df['condition']==c).sum())} cells)" for c in conds])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel(axis_label)
    ax.set_title(_wrap_title(title))
    # Combined legend: count-bin fills + condition outlines.
    # 2026-05-29 Brian (FIX3): the legend used to sit INSIDE the axes at
    # loc="lower right" and covered the KO bar's "11%" (5-9) segment. Move it
    # OUTSIDE, to the RIGHT of the axes, and reserve the right margin so it
    # never overlaps any bar/segment. bbox_inches="tight" on save keeps it.
    from matplotlib.patches import Patch
    fill_hl = [
        (Patch(facecolor=col, edgecolor="black", linewidth=0.5), lbl)
        for (lo, hi, lbl, col) in bins
    ]
    _apply_combined_legend(ax, fill_hl, conds,
                           loc="upper left", fontsize=8,
                           bbox_to_anchor=(1.01, 1.0))
    try:
        ax.figure.subplots_adjust(right=0.78)
    except Exception:
        pass
    ax.grid(True, alpha=0.25, axis="x")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    if subtitle:
        _add_subtitle(ax, subtitle)


def plot_pct_cells_with_spots_rna1(ax, nuc, condition_order=None):
    """01b: % cells with ≥1/≥5/≥10 RNA1 spots, by condition."""
    _bar_pct_cells_with_spots(ax, nuc, "rna1", condition_order=condition_order)


def plot_pct_cells_with_spots_rna2(ax, nuc, condition_order=None):
    """02b: % cells with ≥1/≥5/≥10 RNA2 spots, by condition."""
    _bar_pct_cells_with_spots(ax, nuc, "rna2", condition_order=condition_order)


def plot_nuclear_spots_bin_composition_rna1(ax, nuc, condition_order=None):
    """08b: % of cells with 0 / 1-4 / 5-9 / 10+ NUCLEAR RNA1 spots."""
    _bar_spot_count_bin_composition(
        ax, nuc, count_col="nuclear_spot_count",
        title="Nuclear RNA1 spot-count distribution — composition by condition",
        axis_label="% of cells",
        condition_order=condition_order,
        subtitle=("For each cell, bin its nuclear RNA1 spot count and ask which bin it "
                  "falls into. Stack shows the % of cells in each bin per condition."),
    )


def plot_nuclear_spots_bin_composition_rna2(ax, nuc, condition_order=None):
    """09b: % of cells with 0 / 1-4 / 5-9 / 10+ NUCLEAR RNA2 spots."""
    _bar_spot_count_bin_composition(
        ax, nuc, count_col="nuclear_spot_count_rna2",
        title="Nuclear RNA2 spot-count distribution — composition by condition",
        axis_label="% of cells",
        condition_order=condition_order,
        subtitle=("For each cell, bin its nuclear RNA2 spot count and ask which bin it "
                  "falls into. Stack shows the % of cells in each bin per condition."),
    )


def plot_cyto_spots_bin_composition_rna1(ax, nuc, condition_order=None):
    """10b: % of cells with 0 / 1-4 / 5-9 / 10+ CYTOPLASMIC RNA1 spots."""
    _bar_spot_count_bin_composition(
        ax, nuc, count_col="cyto_spot_count",
        title="Cytoplasmic RNA1 spot-count distribution — composition by condition",
        axis_label="% of cells",
        condition_order=condition_order,
        subtitle=("For each cell, bin its cytoplasmic RNA1 spot count and ask which bin "
                  "it falls into. Stack shows the % of cells in each bin per condition."),
    )


def plot_cyto_spots_bin_composition_rna2(ax, nuc, condition_order=None):
    """11b: % of cells with 0 / 1-4 / 5-9 / 10+ CYTOPLASMIC RNA2 spots."""
    _bar_spot_count_bin_composition(
        ax, nuc, count_col="cyto_spot_count_rna2",
        title="Cytoplasmic RNA2 spot-count distribution — composition by condition",
        axis_label="% of cells",
        condition_order=condition_order,
        subtitle=("For each cell, bin its cytoplasmic RNA2 spot count and ask which bin "
                  "it falls into. Stack shows the % of cells in each bin per condition."),
    )


def _bar_frac_nuclear_box_per_condition(ax, nuc: pd.DataFrame, channel: str,
                                          condition_order: list[str] | None = None) -> None:
    """15b / 16b: per-cell nuclear fraction of detected spots, boxed by
    condition with per-image-mean diamonds. Renders nuclear_spot_count /
    total_spot_count_for_that_channel for each cell. Cells with zero total
    spots are dropped (the fraction is undefined). Sec-only cells therefore
    fall out — they have no spots at all."""
    if channel == "rna1":
        nuc_col, cyt_col = "nuclear_spot_count", "cyto_spot_count"
    else:
        nuc_col, cyt_col = "nuclear_spot_count_rna2", "cyto_spot_count_rna2"
    if nuc_col not in nuc.columns or cyt_col not in nuc.columns:
        ax.set_visible(False); return
    df = nuc.copy()
    nv = pd.to_numeric(df[nuc_col], errors="coerce").fillna(0.0)
    cv = pd.to_numeric(df[cyt_col], errors="coerce").fillna(0.0)
    total = nv + cv
    df["_frac_nuclear"] = np.where(total > 0, nv / total, np.nan)
    if not _box_strip_with_image_means(ax, df, "_frac_nuclear", only_expressing=False,
                                        condition_order=condition_order):
        ax.set_visible(False); return
    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel(f"Fraction of {channel.upper()} spots in nucleus (per cell)")
    ax.set_title(_wrap_title(
        f"Fraction of {channel.upper()} spots that are nuclear — by condition\n"
        f"(per cell; cells with no {channel.upper()} spots excluded)"
    ))
    ax.axhline(0.5, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=8, loc="upper right")
    _add_subtitle(ax,
        f"Each cell contributes one number: nuclear {channel.upper()} spots divided by "
        f"total {channel.upper()} spots in that cell. 1.0 = fully nuclear, 0.0 = fully cytoplasmic.")


def plot_frac_nuclear_box_rna1(ax, nuc, condition_order=None):
    """15b: per-cell RNA1 nuclear fraction by condition."""
    _bar_frac_nuclear_box_per_condition(ax, nuc, "rna1", condition_order=condition_order)


def plot_frac_nuclear_box_rna2(ax, nuc, condition_order=None):
    """16b: per-cell RNA2 nuclear fraction by condition."""
    _bar_frac_nuclear_box_per_condition(ax, nuc, "rna2", condition_order=condition_order)


def _bar_per_cell_nc_stacked(ax, nuc: pd.DataFrame, channel: str,
                                condition_order: list[str] | None = None) -> None:
    """17b / 18b: per condition, the AVERAGE per-cell composition of
    nuclear vs cytoplasmic spots (mean of per-cell fractions across cells
    in that condition). Stacked horizontal bar, summing to 100%."""
    if channel == "rna1":
        nuc_col, cyt_col = "nuclear_spot_count", "cyto_spot_count"
    else:
        nuc_col, cyt_col = "nuclear_spot_count_rna2", "cyto_spot_count_rna2"
    if nuc_col not in nuc.columns or cyt_col not in nuc.columns:
        ax.set_visible(False); return
    df = nuc.copy()
    nv = pd.to_numeric(df[nuc_col], errors="coerce").fillna(0.0)
    cv = pd.to_numeric(df[cyt_col], errors="coerce").fillna(0.0)
    total = nv + cv
    df["_fn"] = np.where(total > 0, nv / total, np.nan)
    if "condition" not in df.columns:
        ax.set_visible(False); return
    conds_in = df["condition"].dropna().unique().tolist()
    conds = order_conditions(conds_in, condition_order or [])
    y = np.arange(len(conds))
    bar_h = 0.65
    for i, cond in enumerate(conds):
        sub = df[df["condition"] == cond]
        frac = sub["_fn"].dropna()
        n_used = len(frac)
        n_total = len(sub)
        # 2026-06-05 Brian: condition-colored bar outline REMOVED.
        edge_c = "none"
        if n_used == 0:
            ax.barh(y[i], 100.0, height=bar_h, color="#dddddd",
                    edgecolor=edge_c, linewidth=COND_EDGE_LINEWIDTH)
            ax.text(50.0, y[i],
                    f"{cond}: no cells with detected {channel.upper()} spots",
                    ha="center", va="center", fontsize=10, color="#444")
            continue
        nuc_pct = 100.0 * float(frac.mean())
        cyt_pct = 100.0 - nuc_pct
        ax.barh(y[i], nuc_pct, height=bar_h, color=_COMP_NUC_COLOR,
                edgecolor=edge_c, linewidth=COND_EDGE_LINEWIDTH,
                label="Nuclear" if i == 0 else None)
        ax.barh(y[i], cyt_pct, left=nuc_pct, height=bar_h, color=_COMP_CYTO_COLOR,
                edgecolor=edge_c, linewidth=COND_EDGE_LINEWIDTH,
                label="Cytoplasmic" if i == 0 else None)
        if nuc_pct >= 7:
            ax.text(nuc_pct / 2.0, y[i], f"{nuc_pct:.0f}%", ha="center", va="center",
                    fontsize=11, color="white", fontweight="bold")
        else:
            ax.text(nuc_pct + 1, y[i], f"{nuc_pct:.0f}%", ha="left", va="center",
                    fontsize=9, color="black")
        if cyt_pct >= 7:
            ax.text(nuc_pct + cyt_pct / 2.0, y[i], f"{cyt_pct:.0f}%",
                    ha="center", va="center", fontsize=11, color="white",
                    fontweight="bold")
        else:
            ax.text(nuc_pct + cyt_pct - 1, y[i], f"{cyt_pct:.0f}%",
                    ha="right", va="center", fontsize=9, color="black")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{c}\n(n={int(df[df['condition']==c]['_fn'].notna().sum())} cells)" for c in conds])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel(f"% of {channel.upper()} spots per cell (averaged across cells)")
    ax.set_title(_wrap_title(
        f"Average per-cell nuclear vs cytoplasmic — {channel.upper()}\n"
        f"(mean of per-cell fractions across each condition's cells)"
    ))
    from matplotlib.patches import Patch
    fill_hl = [
        (Patch(facecolor=_COMP_NUC_COLOR, edgecolor="black", linewidth=0.5), "Nuclear"),
        (Patch(facecolor=_COMP_CYTO_COLOR, edgecolor="black", linewidth=0.5), "Cytoplasmic"),
    ]
    _apply_combined_legend(ax, fill_hl, conds, loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.25, axis="x")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    _add_subtitle(ax,
        f"For each cell, compute the % of its {channel.upper()} spots that are nuclear. "
        f"Bar shows the average of those per-cell percentages across the condition.")


def plot_per_cell_nc_stacked_rna1(ax, nuc, condition_order=None):
    """17b: average per-cell RNA1 nuc vs cyto composition, per condition."""
    _bar_per_cell_nc_stacked(ax, nuc, "rna1", condition_order=condition_order)


def plot_per_cell_nc_stacked_rna2(ax, nuc, condition_order=None):
    """18b: average per-cell RNA2 nuc vs cyto composition, per condition."""
    _bar_per_cell_nc_stacked(ax, nuc, "rna2", condition_order=condition_order)


def _bar_pct_nuclei_with_overlap(ax, nuc: pd.DataFrame, count_col: str,
                                    title: str,
                                    condition_order: list[str] | None = None,
                                    subtitle: str | None = None) -> None:
    """35b / 40b helper: per condition, the % of nuclei with at least one
    overlap event (paired_spot_count_rna1_at_0p3um for 35b, or
    n_active_tss_per_nucleus for 40b). Single bar per condition."""
    if count_col not in nuc.columns or "condition" not in nuc.columns:
        ax.set_visible(False); return
    df = nuc.copy()
    df[count_col] = pd.to_numeric(df[count_col], errors="coerce")
    df = df[df[count_col].notna()]
    if df.empty:
        ax.set_visible(False); return
    conds_in = df["condition"].dropna().unique().tolist()
    conds = order_conditions(conds_in, condition_order or [])
    pcts = []
    ns = []
    for c in conds:
        sub = df[df["condition"] == c]
        ns.append(len(sub))
        pcts.append(100.0 * (sub[count_col] >= 1).mean() if len(sub) else 0.0)
    colors = [_color_for_condition(c, i) for i, c in enumerate(conds)]
    x = np.arange(len(conds))
    ax.bar(x, pcts, color=colors, edgecolor="black", linewidth=0.5)
    for xi, p, n in zip(x, pcts, ns):
        ax.text(xi, p + 1.0, f"{p:.0f}%", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(n={n} nuclei)" for c, n in zip(conds, ns)])
    ax.set_ylim(0, max(105.0, max(pcts + [0]) * 1.2))
    ax.set_ylabel("% of nuclei")
    ax.set_title(_wrap_title(title))
    ax.grid(True, alpha=0.3, axis="y")
    if subtitle:
        _add_subtitle(ax, subtitle)


def plot_pct_nuclei_with_overlap(ax, nuc, condition_order=None):
    """35b: % of nuclei with ≥1 RNA1↔RNA2 overlap event (within 0.3 µm)."""
    suffix = _find_pair_suffix(nuc)
    col = f"paired_spot_count_rna1_at_{suffix}" if suffix else None
    if col is None or col not in nuc.columns:
        # Try the more conventional 0p3um default
        col = "paired_spot_count_rna1_at_0p3um"
    _bar_pct_nuclei_with_overlap(
        ax, nuc, count_col=col,
        title="% of nuclei with ≥1 RNA1↔RNA2 overlap event — by condition",
        condition_order=condition_order,
        subtitle=("For each nucleus, ask: does it have at least one RNA1 spot within 0.3 µm "
                  "of an RNA2 spot? Bar shows the fraction of nuclei in each condition that do."),
    )


def plot_pct_nuclei_with_nuc_overlap(ax, nuc, condition_order=None):
    """40b: % of nuclei with ≥1 NUCLEAR RNA1↔RNA2 overlap event."""
    _bar_pct_nuclei_with_overlap(
        ax, nuc, count_col="n_nuclear_rna1_rna2_overlap_per_nucleus",
        title="% of nuclei with ≥1 nuclear RNA1↔RNA2 overlap — by condition",
        condition_order=condition_order,
        subtitle=("For each nucleus, ask: does it have at least one nuclear RNA1 spot within "
                  "0.3 µm of a nuclear RNA2 spot? Bar shows the fraction of nuclei that do."),
    )


def plot_composition_overlap_location(ax, spots: pd.DataFrame,
                                        condition_order: list[str] | None = None) -> None:
    """36b: of the spots that ARE overlapping (within 0.3 µm of a spot in
    the other channel), what % are in the nucleus vs cytoplasm? Stacked
    horizontal bar per condition. This re-frames fig 36 ("where do
    overlapping spots live in absolute count") as a per-condition
    composition so WT-vs-KO localization shifts read at a glance."""
    if spots.empty or "in_nucleus" not in spots.columns or "condition" not in spots.columns:
        ax.set_visible(False); return
    suffix = _find_pair_suffix(spots) or "0p3um"
    pair_col = f"paired_at_{suffix}"
    if pair_col not in spots.columns:
        if "colocalized" in spots.columns:
            pair_col = "colocalized"
        else:
            ax.set_visible(False); return
    df = spots.copy()
    df[pair_col] = pd.to_numeric(df[pair_col], errors="coerce")
    df["in_nucleus"] = pd.to_numeric(df["in_nucleus"], errors="coerce")
    paired = df[df[pair_col] == 1]
    if paired.empty:
        ax.set_visible(False); return
    conds_in = paired["condition"].dropna().unique().tolist()
    conds = order_conditions(conds_in, condition_order or [])
    y = np.arange(len(conds))
    bar_h = 0.65
    for i, cond in enumerate(conds):
        sub = paired[paired["condition"] == cond]
        n_overlap = len(sub)
        # 2026-06-05 Brian: condition-colored bar outline REMOVED.
        edge_c = "none"
        if n_overlap == 0:
            ax.barh(y[i], 100.0, height=bar_h, color="#dddddd",
                    edgecolor=edge_c, linewidth=COND_EDGE_LINEWIDTH)
            ax.text(50.0, y[i], f"{cond}: no overlap events",
                    ha="center", va="center", fontsize=10, color="#444")
            continue
        nuc_pct = 100.0 * float((sub["in_nucleus"] == 1).mean())
        cyt_pct = 100.0 - nuc_pct
        ax.barh(y[i], nuc_pct, height=bar_h, color=_COMP_NUC_COLOR,
                edgecolor=edge_c, linewidth=COND_EDGE_LINEWIDTH,
                label="Nuclear" if i == 0 else None)
        ax.barh(y[i], cyt_pct, left=nuc_pct, height=bar_h, color=_COMP_CYTO_COLOR,
                edgecolor=edge_c, linewidth=COND_EDGE_LINEWIDTH,
                label="Cytoplasmic" if i == 0 else None)
        if nuc_pct >= 7:
            ax.text(nuc_pct / 2.0, y[i], f"{nuc_pct:.0f}%", ha="center", va="center",
                    fontsize=11, color="white", fontweight="bold")
        else:
            ax.text(nuc_pct + 1, y[i], f"{nuc_pct:.0f}%", ha="left", va="center",
                    fontsize=9, color="black")
        if cyt_pct >= 7:
            ax.text(nuc_pct + cyt_pct / 2.0, y[i], f"{cyt_pct:.0f}%",
                    ha="center", va="center", fontsize=11, color="white",
                    fontweight="bold")
        else:
            ax.text(nuc_pct + cyt_pct - 1, y[i], f"{cyt_pct:.0f}%",
                    ha="right", va="center", fontsize=9, color="black")
    ax.set_yticks(y)
    ax.set_yticklabels([
        f"{c}\n(n={int((paired['condition']==c).sum())} overlap spots)"
        for c in conds
    ])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of overlap events")
    ax.set_title(_wrap_title("Overlap events — nuclear vs cytoplasmic composition by condition"))
    from matplotlib.patches import Patch
    fill_hl = [
        (Patch(facecolor=_COMP_NUC_COLOR, edgecolor="black", linewidth=0.5), "Nuclear"),
        (Patch(facecolor=_COMP_CYTO_COLOR, edgecolor="black", linewidth=0.5), "Cytoplasmic"),
    ]
    _apply_combined_legend(ax, fill_hl, conds, loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.25, axis="x")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    _add_subtitle(ax,
        "Take only the spots that overlap with a spot in the other channel (≤0.3 µm). "
        "Of those overlap events, what % live in the nucleus vs the cytoplasm?")


def plot_pct_cells_with_cyto_rna1(ax, nuc: pd.DataFrame,
                                    condition_order: list[str] | None = None) -> None:
    """41b: % of cells with ≥1 and ≥5 cytoplasmic RNA1 spots, per condition.
    Sec-only conditions have no spots → bar at 0, annotated 'n/a'."""
    if "cyto_spot_count" not in nuc.columns or "condition" not in nuc.columns:
        ax.set_visible(False); return
    df = nuc.copy()
    df["cyto_spot_count"] = pd.to_numeric(df["cyto_spot_count"], errors="coerce")
    df = df[df["cyto_spot_count"].notna()]
    if df.empty:
        ax.set_visible(False); return
    conds_in = df["condition"].dropna().unique().tolist()
    conds = order_conditions(conds_in, condition_order or [])
    thresholds = [1, 5]
    width = 0.35
    x = np.arange(len(conds))
    for j, t in enumerate(thresholds):
        pcts = []
        for c in conds:
            sub = df[df["condition"] == c]
            pcts.append(100.0 * (sub["cyto_spot_count"] >= t).mean() if len(sub) else 0.0)
        ax.bar(x + (j - 0.5) * width, pcts, width=width,
               label=f"≥{t} cyto-RNA1 spots",
               edgecolor="black", linewidth=0.5,
               color=OKABE_ITO[j + 1])
        for xi, p, c in zip(x + (j - 0.5) * width, pcts, conds):
            # Sec-only / any condition with 0% gets "n/a" annotation when it
            # also has no cells with any cyto RNA1 (otherwise just "0%").
            if c == SEC_ONLY_CONDITION and p == 0.0:
                ax.text(xi, 1.0, "n/a", ha="center", va="bottom",
                        fontsize=8, color="#666", style="italic")
            else:
                ax.text(xi, p + 1.0, f"{p:.0f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{_display_condition(c)}\n(n={int((df['condition']==c).sum())} cells)" for c in conds])
    ax.set_ylim(0, 105)
    ax.set_ylabel("% of cells")
    ax.set_title(_wrap_title("% of cells with cytoplasmic RNA1 spots — by condition"))
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    _add_subtitle(ax,
        "Reads each cell's cytoplasmic RNA1 spot count and asks: does it pass each threshold? "
        "Bars show the proportion of cells per condition. Sec-only labeled n/a (no spots).")


# ---------------------------------------------------------------------------
# 2026-05-18 Brian — figures 57–62: per-spot-type localization composition
# AND RNA1↔RNA2 comparisons.
#
# Brian: "ill want percentage of localization of each spot type based on rna
# type, maybe some more comparisons between each rna as well."
#
# These all consume the same nuclei_metrics / spot_metrics / per_image_summary
# CSVs the other figures use — no new pipeline output required. Each function
# follows the same shape: build a per-cell or per-bar dataframe, render the
# plot, add an italic subtitle, set a wrapped title.
# ---------------------------------------------------------------------------


def plot_localization_composition_both_channels(ax, summary: pd.DataFrame | None,
                                                  condition_order: list[str] | None = None) -> None:
    """Figure 57 — grouped stacked bar: per condition × channel, what % of
    spots are nuclear vs cytoplasmic.

    x-axis groups: WT/KO/sec-only (or whatever conditions exist). Within each
    group: two bars (RNA1, RNA2). Each bar split nuclear% (dark) vs
    cytoplasmic% (light). At-a-glance comparison of how nuclear retention
    differs between channels within each condition.

    Reads ``frac_nuclear_rna1`` / ``frac_nuclear_rna2`` weighted by
    ``total_spots_rna1`` / ``total_spots_rna2`` (already in
    per_image_summary). Sec-only condition shows empty bars — correct
    visual signal for a no-spot control.
    """
    if summary is None or len(summary) == 0:
        ax.set_visible(False); return
    needed = {"condition", "frac_nuclear_rna1", "frac_nuclear_rna2",
              "total_spots_rna1", "total_spots_rna2"}
    if not needed.issubset(set(summary.columns)):
        ax.set_visible(False); return

    conds_in = summary["condition"].dropna().unique().tolist()
    conds = order_conditions(conds_in, condition_order or [])
    if not conds:
        ax.set_visible(False); return

    # For each condition × channel, compute spot-count-weighted mean of
    # frac_nuclear (matches _aggregate_per_condition logic).
    def _wmean(rows: pd.DataFrame, value_col: str, weight_col: str) -> float:
        v = pd.to_numeric(rows[value_col], errors="coerce")
        w = pd.to_numeric(rows[weight_col], errors="coerce").fillna(0.0)
        mask = v.notna() & (w > 0)
        if mask.sum() == 0:
            return float("nan")
        return float(np.average(v[mask].values, weights=w[mask].values))

    bar_pairs = []  # list of (cond, channel, nuc_pct, cyt_pct)
    for cond in conds:
        sub = summary[summary["condition"] == cond]
        for ch_label, val_col, wgt_col in (
            ("RNA1", "frac_nuclear_rna1", "total_spots_rna1"),
            ("RNA2", "frac_nuclear_rna2", "total_spots_rna2"),
        ):
            f = _wmean(sub, val_col, wgt_col)
            if np.isfinite(f):
                bar_pairs.append((cond, ch_label, 100.0 * f, 100.0 - 100.0 * f))
            else:
                bar_pairs.append((cond, ch_label, float("nan"), float("nan")))

    # Lay bars: each condition gets a slot of 2 bars + gap. Within slot,
    # RNA1 on the left, RNA2 on the right.
    n_conds = len(conds)
    bar_w = 0.4
    intra_gap = 0.05
    inter_gap = 0.6
    xs = []
    for i in range(n_conds):
        base = i * (2 * bar_w + intra_gap + inter_gap)
        xs.append(base)
        xs.append(base + bar_w + intra_gap)
    # xs[2i] = RNA1 of cond i; xs[2i+1] = RNA2 of cond i.
    nuc_color = _COMP_NUC_COLOR
    cyt_color = _COMP_CYTO_COLOR
    cond_idx_map = {c: i for i, c in enumerate(conds)}
    for k, (cond, ch_label, nuc_pct, cyt_pct) in enumerate(bar_pairs):
        x = xs[k]
        # 2026-06-05 Brian: condition-colored bar outline REMOVED.
        edge_c = "none"
        if not np.isfinite(nuc_pct):
            ax.bar(x, 100, width=bar_w, color="#dddddd",
                   edgecolor=edge_c, linewidth=COND_EDGE_LINEWIDTH)
            ax.text(x, 50, "no\nspots", ha="center", va="center",
                    fontsize=8, color="#444")
            continue
        # Bar FILL encodes nuclear vs cytoplasmic (blue / vermillion).
        # Bar EDGE encodes condition (WT vs KO via CONDITION_COLORS).
        # Reader can decode both axes independently.
        ax.bar(x, nuc_pct, width=bar_w, color=nuc_color,
               edgecolor=edge_c, linewidth=COND_EDGE_LINEWIDTH,
               label="Nuclear" if k == 0 else None)
        ax.bar(x, cyt_pct, bottom=nuc_pct, width=bar_w, color=cyt_color,
               edgecolor=edge_c, linewidth=COND_EDGE_LINEWIDTH,
               label="Cytoplasmic" if k == 0 else None)
        if nuc_pct >= 8:
            ax.text(x, nuc_pct / 2.0, f"{nuc_pct:.0f}%",
                    ha="center", va="center", fontsize=9,
                    color="white", fontweight="bold")
        if cyt_pct >= 8:
            ax.text(x, nuc_pct + cyt_pct / 2.0, f"{cyt_pct:.0f}%",
                    ha="center", va="center", fontsize=9,
                    color="white", fontweight="bold")
        # Per-bar channel label below the bar (RNA1 / RNA2).
        ax.text(x, -3.0, ch_label, ha="center", va="top", fontsize=8,
                color="#333")

    # Condition labels under each pair (centered between RNA1 and RNA2 bar).
    cond_centers = [(xs[2 * i] + xs[2 * i + 1]) / 2.0 for i in range(n_conds)]
    ax.set_xticks(cond_centers)
    ax.set_xticklabels([f"\n{c}" for c in conds])  # extra newline so the
    # condition label sits BELOW the RNA1/RNA2 labels we added with
    # ax.text(x, -3.0, ...). The leading "\n" pushes the tick text down one
    # line in display coordinates without touching the axis spacing.
    ax.set_ylim(-12, 110)
    ax.set_ylabel("% of spots in channel")
    ax.set_title(_wrap_title(
        "Spot localization composition — both channels by condition"))
    from matplotlib.patches import Patch
    fill_hl = [
        (Patch(facecolor=nuc_color, edgecolor="black", linewidth=0.5), "Nuclear"),
        (Patch(facecolor=cyt_color, edgecolor="black", linewidth=0.5), "Cytoplasmic"),
    ]
    _apply_combined_legend(ax, fill_hl, conds, loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.25, axis="y")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    _add_subtitle(ax,
        "Of all detected spots in a channel, what fraction sit in nucleus vs "
        "cytoplasm? Bar OUTLINE = condition (WT/KO), bar FILL = compartment "
        "(nuclear / cytoplasmic). Compared across both RNAs and all conditions.")


def plot_rna1_vs_rna2_nuclear_fraction_scatter(ax, nuc: pd.DataFrame,
                                                  condition_order: list[str] | None = None) -> None:
    """Figure 58 — per-nucleus scatter, x = RNA1 nuclear fraction, y = RNA2
    nuclear fraction. Color by condition. y=x reference line.

    Above the y=x line: this nucleus keeps RNA2 more nuclear than RNA1.
    Below: RNA1 more nuclear than RNA2. Cells with 0 spots in either
    channel are dropped (the fraction is undefined).
    """
    needed = {"nuclear_spot_count", "cyto_spot_count",
              "nuclear_spot_count_rna2", "cyto_spot_count_rna2"}
    if not needed.issubset(set(nuc.columns)) or "condition" not in nuc.columns:
        ax.set_visible(False); return
    df = nuc.copy()
    n1 = pd.to_numeric(df["nuclear_spot_count"], errors="coerce").fillna(0.0)
    c1 = pd.to_numeric(df["cyto_spot_count"], errors="coerce").fillna(0.0)
    n2 = pd.to_numeric(df["nuclear_spot_count_rna2"], errors="coerce").fillna(0.0)
    c2 = pd.to_numeric(df["cyto_spot_count_rna2"], errors="coerce").fillna(0.0)
    t1 = n1 + c1
    t2 = n2 + c2
    df["_fn1"] = np.where(t1 > 0, n1 / t1, np.nan)
    df["_fn2"] = np.where(t2 > 0, n2 / t2, np.nan)
    df = df[df["_fn1"].notna() & df["_fn2"].notna()]
    if df.empty:
        ax.set_visible(False); return

    conds_in = df["condition"].dropna().unique().tolist()
    conds = order_conditions(conds_in, condition_order or [])
    plotted = 0
    for i, cond in enumerate(conds):
        sub = df[df["condition"] == cond]
        if sub.empty:
            continue
        ax.scatter(sub["_fn1"], sub["_fn2"], s=22, alpha=0.6,
                   color=_color_for_condition(cond, i),
                   edgecolor="white", linewidth=0.4,
                   label=f"{cond} (n={len(sub)})")
        plotted += 1
    if plotted == 0:
        ax.set_visible(False); return

    # y=x reference line.
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1.0,
            alpha=0.6, label="y = x (equal nuclear fraction)")

    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("RNA1 nuclear fraction (per cell)")
    ax.set_ylabel("RNA2 nuclear fraction (per cell)")
    ax.set_title(_wrap_title(
        "RNA1 vs RNA2 nuclear fraction — per cell"))
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    _add_subtitle(ax,
        "Per cell: does this cell preferentially keep RNA1 in nucleus, or "
        "RNA2? Above y=x = RNA2 more nuclear; below = RNA1 more nuclear.")


def plot_rna1_minus_rna2_nuclear_fraction(ax, nuc: pd.DataFrame,
                                             condition_order: list[str] | None = None) -> None:
    """Figure 59 — per-condition box+strip+image-mean of
    (frac_nuclear_rna1 − frac_nuclear_rna2) per cell. Positive ⇒ RNA1 more
    nuclear than RNA2 in this cell; negative ⇒ RNA2 more nuclear."""
    needed = {"nuclear_spot_count", "cyto_spot_count",
              "nuclear_spot_count_rna2", "cyto_spot_count_rna2"}
    if not needed.issubset(set(nuc.columns)) or "condition" not in nuc.columns:
        ax.set_visible(False); return
    df = nuc.copy()
    n1 = pd.to_numeric(df["nuclear_spot_count"], errors="coerce").fillna(0.0)
    c1 = pd.to_numeric(df["cyto_spot_count"], errors="coerce").fillna(0.0)
    n2 = pd.to_numeric(df["nuclear_spot_count_rna2"], errors="coerce").fillna(0.0)
    c2 = pd.to_numeric(df["cyto_spot_count_rna2"], errors="coerce").fillna(0.0)
    t1 = n1 + c1
    t2 = n2 + c2
    fn1 = np.where(t1 > 0, n1 / t1, np.nan)
    fn2 = np.where(t2 > 0, n2 / t2, np.nan)
    df["_delta"] = fn1 - fn2
    df = df[pd.Series(df["_delta"]).notna() & np.isfinite(df["_delta"])]
    if df.empty:
        ax.set_visible(False); return
    if not _box_strip_with_image_means(ax, df, "_delta", only_expressing=False,
                                        condition_order=condition_order):
        ax.set_visible(False); return
    ax.axhline(0.0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("RNA1 nuclear frac − RNA2 nuclear frac (per cell)")
    ax.set_title(_wrap_title(
        "Per-cell Δ nuclear fraction (RNA1 − RNA2) — by condition"))
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=8, loc="upper right")
    _add_subtitle(ax,
        "Per-cell difference in nuclear retention between RNA1 and RNA2. "
        "Each dot = one cell. Positive = RNA1 more nuclear than RNA2; "
        "negative = RNA2 more nuclear.")


def plot_overlap_location_split(ax, spots: pd.DataFrame,
                                  condition_order: list[str] | None = None) -> None:
    """Figure 60 — for each condition, stacked horizontal bar with TWO
    segments: % of all RNA1↔RNA2 overlapping pairs that are nuclear, %
    that are cytoplasmic.

    Different from 36b (which is the same data plotted as a stacked bar
    weighted by overlap-spot count). 60 takes the simpler framing: "of all
    overlap events in this condition, where do they happen?".
    """
    if spots.empty or "in_nucleus" not in spots.columns or "condition" not in spots.columns:
        ax.set_visible(False); return
    suffix = _find_pair_suffix(spots) or "0p3um"
    pair_col = f"paired_at_{suffix}"
    if pair_col not in spots.columns:
        if "colocalized" in spots.columns:
            pair_col = "colocalized"
        else:
            ax.set_visible(False); return
    df = spots.copy()
    df[pair_col] = pd.to_numeric(df[pair_col], errors="coerce")
    df["in_nucleus"] = pd.to_numeric(df["in_nucleus"], errors="coerce")
    paired = df[df[pair_col] == 1]
    if paired.empty:
        ax.set_visible(False); return

    conds_in = paired["condition"].dropna().unique().tolist()
    conds = order_conditions(conds_in, condition_order or [])
    y = np.arange(len(conds))
    bar_h = 0.65
    for i, cond in enumerate(conds):
        sub = paired[paired["condition"] == cond]
        n_overlap = len(sub)
        # 2026-06-05 Brian: condition-colored bar outline REMOVED.
        edge_c = "none"
        if n_overlap == 0:
            ax.barh(y[i], 100.0, height=bar_h, color="#dddddd",
                    edgecolor=edge_c, linewidth=COND_EDGE_LINEWIDTH)
            ax.text(50.0, y[i], f"{cond}: no overlap events",
                    ha="center", va="center", fontsize=10, color="#444")
            continue
        nuc_pct = 100.0 * float((sub["in_nucleus"] == 1).mean())
        cyt_pct = 100.0 - nuc_pct
        ax.barh(y[i], nuc_pct, height=bar_h, color=_COMP_NUC_COLOR,
                edgecolor=edge_c, linewidth=COND_EDGE_LINEWIDTH,
                label="Nuclear" if i == 0 else None)
        ax.barh(y[i], cyt_pct, left=nuc_pct, height=bar_h, color=_COMP_CYTO_COLOR,
                edgecolor=edge_c, linewidth=COND_EDGE_LINEWIDTH,
                label="Cytoplasmic" if i == 0 else None)
        if nuc_pct >= 7:
            ax.text(nuc_pct / 2.0, y[i], f"{nuc_pct:.0f}%", ha="center", va="center",
                    fontsize=11, color="white", fontweight="bold")
        else:
            ax.text(nuc_pct + 1, y[i], f"{nuc_pct:.0f}%", ha="left", va="center",
                    fontsize=9, color="black")
        if cyt_pct >= 7:
            ax.text(nuc_pct + cyt_pct / 2.0, y[i], f"{cyt_pct:.0f}%",
                    ha="center", va="center", fontsize=11, color="white",
                    fontweight="bold")
        else:
            ax.text(nuc_pct + cyt_pct - 1, y[i], f"{cyt_pct:.0f}%",
                    ha="right", va="center", fontsize=9, color="black")
    ax.set_yticks(y)
    ax.set_yticklabels([
        f"{c}\n(n={int((paired['condition']==c).sum())} overlap events)"
        for c in conds
    ])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of overlap events")
    ax.set_title(_wrap_title(
        "Overlap events — nuclear vs cytoplasmic split by condition"))
    from matplotlib.patches import Patch
    fill_hl = [
        (Patch(facecolor=_COMP_NUC_COLOR, edgecolor="black", linewidth=0.5), "Nuclear"),
        (Patch(facecolor=_COMP_CYTO_COLOR, edgecolor="black", linewidth=0.5), "Cytoplasmic"),
    ]
    _apply_combined_legend(ax, fill_hl, conds, loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.25, axis="x")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    _add_subtitle(ax,
        "When RNA1 and RNA2 spots overlap (within 0.3 µm), where in the "
        "cell does that happen?")


def plot_rna1_to_rna2_spot_ratio_per_cell(ax, nuc: pd.DataFrame,
                                             condition_order: list[str] | None = None) -> None:
    """Figure 61 — per-cell log10 box of (rna1_spot_count / rna2_spot_count)
    for cells with ≥1 of each, by condition. Higher = more RNA1 per RNA2.

    Skips zero-spot cells (denominator) and zero-numerator (so the log is
    finite). Sec-only collapses to empty by design.
    """
    if ("rna_spot_count" not in nuc.columns or "n_spots_rna2" not in nuc.columns
            or "condition" not in nuc.columns):
        ax.set_visible(False); return
    df = nuc.copy()
    r1 = pd.to_numeric(df["rna_spot_count"], errors="coerce")
    r2 = pd.to_numeric(df["n_spots_rna2"], errors="coerce")
    # Require strictly positive both channels so log10 is defined.
    mask = r1.notna() & r2.notna() & (r1 > 0) & (r2 > 0)
    df = df[mask].copy()
    if df.empty:
        ax.set_visible(False); return
    df["_log_ratio"] = np.log10(pd.to_numeric(df["rna_spot_count"], errors="coerce")
                                 / pd.to_numeric(df["n_spots_rna2"], errors="coerce"))
    if not _box_strip_with_image_means(ax, df, "_log_ratio", only_expressing=False,
                                        condition_order=condition_order):
        ax.set_visible(False); return
    ax.axhline(0.0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_ylabel("log10(RNA1 spot count / RNA2 spot count) per cell")
    ax.set_title(_wrap_title(
        "Per-cell RNA1:RNA2 spot ratio — by condition (log scale)"))
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=8, loc="upper right")
    _add_subtitle(ax,
        "Per-cell ratio of RNA1 to RNA2 spot count (log10 scale). "
        "Higher = more RNA1 per RNA2. Cells with 0 of either channel "
        "are dropped (ratio undefined).")


def plot_nuclear_overlap_fraction_of_nuclear_rna1(ax, nuc: pd.DataFrame,
                                                     condition_order: list[str] | None = None) -> None:
    """Figure 62 — for cells with ≥1 nuclear RNA1 spot: per-condition box of
    (nuclear RNA1+RNA2 overlapping spots) / (total nuclear RNA1 spots).

    Numerator = ``n_active_tss_per_nucleus`` (count of nuclear RNA1 spots
    that overlap an RNA2 spot within 0.3 µm).
    Denominator = ``nuclear_spot_count`` (total nuclear RNA1 spots).

    Reads as: of this cell's nuclear RNA1 spots, what fraction also
    co-locate with a nuclear RNA2 spot? Note: this is conceptually the
    same numerator/denominator as the existing 42 panel
    (plot_transcription_efficiency_proxy), but with neutral framing
    (no "TSS efficiency" / nascent-RNA jargon).
    """
    if ("n_nuclear_rna1_rna2_overlap_per_nucleus" not in nuc.columns
            or "nuclear_spot_count" not in nuc.columns
            or "condition" not in nuc.columns):
        ax.set_visible(False); return
    df = nuc.copy()
    nuclear_rna1 = pd.to_numeric(df["nuclear_spot_count"], errors="coerce")
    nuclear_overlap = pd.to_numeric(df["n_nuclear_rna1_rna2_overlap_per_nucleus"], errors="coerce")
    # Require ≥1 nuclear RNA1 spot.
    mask = nuclear_rna1.notna() & (nuclear_rna1 >= 1) & nuclear_overlap.notna()
    df = df[mask].copy()
    if df.empty:
        ax.set_visible(False); return
    df["_frac"] = (pd.to_numeric(df["n_nuclear_rna1_rna2_overlap_per_nucleus"], errors="coerce")
                    / pd.to_numeric(df["nuclear_spot_count"], errors="coerce"))
    df = df[df["_frac"].notna() & np.isfinite(df["_frac"])]
    # Clip very rarely-occurring >1 floating-point cases from off-by-one
    # voxel matching (the denominator is the per-nucleus count, numerator
    # is detected overlap pairs; both should agree on count semantics).
    df["_frac"] = df["_frac"].clip(0.0, 1.0)
    if df.empty:
        ax.set_visible(False); return
    if not _box_strip_with_image_means(ax, df, "_frac", only_expressing=False,
                                        condition_order=condition_order):
        ax.set_visible(False); return
    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel("Nuclear RNA1+RNA2 overlap / nuclear RNA1 (per cell)")
    ax.set_title(_wrap_title(
        "Nuclear overlap fraction of nuclear RNA1 — by condition"))
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=8, loc="upper right")
    _add_subtitle(ax,
        "Of this cell's nuclear RNA1 spots, what fraction also overlap "
        "with RNA2 (≤0.3 µm)? Computed per cell, plotted across conditions.")


def build_rna_rna_layout(nuc_df, spots_df, summary_df, cond_order_arg):
    """Build the rna_rna PLOT_LAYOUT: 45 panels covering per-channel
    distributions, N/C breakdowns, spot-spot colocalization, exon/intron
    biology proxies (active TSS, mature mRNA, burst size, TSS efficiency,
    nascent-to-mature ratio), two-RNA general analyses (quadrants, anti-
    correlation, exclusive expression, within-nucleus paired fraction,
    cytoplasmic clustering), and cross-condition difference figures (volcano-
    like, effect sizes, variance decomposition). Returns the same tuple
    format as the rna_only layout."""
    return [
        # === Block A: spot-count distributions per channel ===
        (1,  "spots_per_nucleus_distribution_rna1", "Spots per nucleus — RNA1",
            lambda ax, n=nuc_df: plot_spots_per_nucleus_channel(ax, n, "rna1")),
        (2,  "spots_per_nucleus_distribution_rna2", "Spots per nucleus — RNA2",
            lambda ax, n=nuc_df: plot_spots_per_nucleus_channel(ax, n, "rna2")),
        (3,  "cumulative_spots_per_cell_both_channels", "CDF: spots per cell — both channels",
            lambda ax, n=nuc_df: plot_cumulative_spots_both_channels(ax, n)),
        (4,  "mean_spots_per_image_rna1", "Mean spots per nucleus — RNA1",
            lambda ax, n=nuc_df, s=summary_df: plot_mean_spots_per_image_channel(ax, s, n, "rna1")),
        (5,  "mean_spots_per_image_rna2", "Mean spots per nucleus — RNA2",
            lambda ax, n=nuc_df, s=summary_df: plot_mean_spots_per_image_channel(ax, s, n, "rna2")),
        # === Block B: nuclear/cytoplasmic stratification per channel ===
        (6,  "nuclear_vs_cytoplasmic_rna1", "Nuclear vs cytoplasmic — RNA1",
            lambda ax, n=nuc_df: plot_nuclear_vs_cytoplasmic_channel(ax, n, "rna1")),
        (7,  "nuclear_vs_cytoplasmic_rna2", "Nuclear vs cytoplasmic — RNA2",
            lambda ax, n=nuc_df: plot_nuclear_vs_cytoplasmic_channel(ax, n, "rna2")),
        (8,  "nuclear_spots_distribution_rna1", "Nuclear spots distribution — RNA1",
            lambda ax, n=nuc_df: plot_nuclear_spots_distribution(ax, n, "rna1")),
        (9,  "nuclear_spots_distribution_rna2", "Nuclear spots distribution — RNA2",
            lambda ax, n=nuc_df: plot_nuclear_spots_distribution(ax, n, "rna2")),
        (10, "cytoplasmic_spots_distribution_rna1", "Cytoplasmic spots distribution — RNA1",
            lambda ax, n=nuc_df: plot_cytoplasmic_spots_distribution(ax, n, "rna1")),
        (11, "cytoplasmic_spots_distribution_rna2", "Cytoplasmic spots distribution — RNA2",
            lambda ax, n=nuc_df: plot_cytoplasmic_spots_distribution(ax, n, "rna2")),
        (12, "predominantly_nuclear_fraction", "Predominantly-nuclear cells (per channel)",
            lambda ax, n=nuc_df, co=cond_order_arg: plot_predominantly_nuclear_fraction(ax, n, condition_order=co)),
        (13, "box_nc_ratio_total_intensity_rna1", "N/C ratio total intensity — RNA1",
            lambda ax, n=nuc_df, sp=spots_df, co=cond_order_arg: plot_box_nc_ratio_total_intensity(ax, n, "rna1", condition_order=co, spots=sp)),
        (14, "box_nc_ratio_total_intensity_rna2", "N/C ratio total intensity — RNA2",
            lambda ax, n=nuc_df, sp=spots_df, co=cond_order_arg: plot_box_nc_ratio_total_intensity(ax, n, "rna2", condition_order=co, spots=sp)),
        (15, "box_nc_spot_count_rna1", "N/C spot-count ratio — RNA1",
            lambda ax, n=nuc_df, co=cond_order_arg: plot_box_nc_spot_count(ax, n, "rna1", condition_order=co)),
        (16, "box_nc_spot_count_rna2", "N/C spot-count ratio — RNA2",
            lambda ax, n=nuc_df, co=cond_order_arg: plot_box_nc_spot_count(ax, n, "rna2", condition_order=co)),
        # === Block C: per-cell intensity / expression boxes by condition ===
        (17, "box_spots_per_cell_rna1_by_condition", "Spots per cell — RNA1 by condition",
            lambda ax, n=nuc_df.rename(columns={"rna_spot_count": "rna_spot_count"}),
                co=cond_order_arg: plot_box_spots_by_condition(ax, n, condition_order=co)),
        (18, "box_spots_per_cell_rna2_by_condition", "Spots per cell — RNA2 by condition",
            lambda ax, n=nuc_df.assign(rna_spot_count=lambda d: d.get("n_spots_rna2", d.get("rna_spot_count"))),
                co=cond_order_arg: plot_box_spots_by_condition(ax, n, condition_order=co)),
        (19, "box_cell_total_intensity_rna1_by_condition", "Per-cell total RNA1 intensity",
            lambda ax, n=nuc_df, co=cond_order_arg: plot_box_cell_total_intensity_by_channel(ax, n, "rna1", condition_order=co)),
        (20, "box_cell_total_intensity_rna2_by_condition", "Per-cell total RNA2 intensity",
            lambda ax, n=nuc_df, co=cond_order_arg: plot_box_cell_total_intensity_by_channel(ax, n, "rna2", condition_order=co)),
        (21, "box_per_cell_expression_rna1", "Per-cell expression — RNA1",
            lambda ax, n=nuc_df, co=cond_order_arg: plot_box_per_cell_expression_channel(ax, n, "rna1", condition_order=co)),
        (22, "box_per_cell_expression_rna2", "Per-cell expression — RNA2",
            lambda ax, n=nuc_df, co=cond_order_arg: plot_box_per_cell_expression_channel(ax, n, "rna2", condition_order=co)),
        # === Block D: spot-quality histograms per channel ===
        (23, "spot_peak_intensity_rna1", "Spot peak intensity — RNA1",
            lambda ax, sp=spots_df: plot_spot_peak_intensity_channel(ax, sp, "rna1")),
        (24, "spot_peak_intensity_rna2", "Spot peak intensity — RNA2",
            lambda ax, sp=spots_df: plot_spot_peak_intensity_channel(ax, sp, "rna2")),
        (25, "spot_size_rna1", "Spot size distribution — RNA1",
            lambda ax, sp=spots_df: plot_spot_size_channel(ax, sp, "rna1")),
        (26, "spot_size_rna2", "Spot size distribution — RNA2",
            lambda ax, sp=spots_df: plot_spot_size_channel(ax, sp, "rna2")),
        (27, "local_snr_rna1", "Per-spot local SNR — RNA1",
            lambda ax, sp=spots_df: plot_local_snr_channel(ax, sp, "rna1")),
        (28, "local_snr_rna2", "Per-spot local SNR — RNA2",
            lambda ax, sp=spots_df: plot_local_snr_channel(ax, sp, "rna2")),
        (29, "sorted_brightness_rna1", "Sorted spot brightness — RNA1",
            lambda ax, sp=spots_df: plot_sorted_brightness_channel(ax, sp, "rna1")),
        (30, "sorted_brightness_rna2", "Sorted spot brightness — RNA2",
            lambda ax, sp=spots_df: plot_sorted_brightness_channel(ax, sp, "rna2")),
        (31, "spots_vs_area_rna1", "Spots vs nucleus area — RNA1",
            lambda ax, n=nuc_df, co=cond_order_arg: plot_spots_vs_area_channel(ax, n, "rna1", condition_order=co)),
        (32, "spots_vs_area_rna2", "Spots vs nucleus area — RNA2",
            lambda ax, n=nuc_df, co=cond_order_arg: plot_spots_vs_area_channel(ax, n, "rna2", condition_order=co)),
        # === Block E: inter-channel colocalization core ===
        (33, "overlap_fraction_per_condition", "Spot–spot overlap fraction — by condition",
            lambda ax, s=summary_df, co=cond_order_arg: plot_paired_fraction_per_condition(ax, s, condition_order=co)),
        (34, "nn_distance_distribution", "Nearest-neighbor distance distribution",
            lambda ax, sp=spots_df: plot_nn_distance_distribution(ax, sp)),
        (35, "overlapping_spots_per_nucleus", "Overlapping spots per nucleus",
            lambda ax, n=nuc_df: plot_paired_spots_per_nucleus_distribution(ax, n)),
        (36, "overlapping_spots_nuc_vs_cyto", "Overlapping spots: nuclear vs cytoplasmic",
            lambda ax, sp=spots_df: plot_paired_only_nuc_vs_cyto(ax, sp)),
        (37, "rna1_vs_rna2_per_cell_scatter", "RNA1 vs RNA2 spots per cell",
            lambda ax, n=nuc_df, co=cond_order_arg: plot_rna1_vs_rna2_per_cell_scatter(ax, n, condition_order=co)),
        (38, "rna1_vs_rna2_intensity_scatter", "RNA1 vs RNA2 per-cell intensity",
            lambda ax, n=nuc_df, co=cond_order_arg: plot_rna1_vs_rna2_intensity_scatter(ax, n, condition_order=co)),
        (39, "within_nucleus_overlap_fraction", "Within-nucleus overlap fraction",
            lambda ax, sp=spots_df, co=cond_order_arg: plot_within_nucleus_paired_fraction(ax, sp, condition_order=co)),
        # === Block F: nuclear-overlap / cytoplasmic-spot per-condition figures ===
        (40, "nuclear_rna1_rna2_overlap_per_nucleus", "Nuclear RNA1+RNA2 overlap per nucleus (≤0.3 µm)",
            lambda ax, n=nuc_df, co=cond_order_arg: plot_active_tss_per_nucleus(ax, n, condition_order=co)),
        (41, "cytoplasmic_rna1_spots_per_cell", "Cytoplasmic RNA1 spots per cell",
            lambda ax, n=nuc_df, co=cond_order_arg: plot_mature_mrna_per_cell(ax, n, "rna1", condition_order=co)),
        (42, "nuclear_rna1_overlap_fraction", "Fraction of nuclear RNA1 spots overlapping with RNA2",
            lambda ax, n=nuc_df, co=cond_order_arg: plot_transcription_efficiency_proxy(ax, n, condition_order=co)),
        (43, "nuclear_overlap_count_distribution", "Nuclear-overlap count distribution (cells with ≥1)",
            lambda ax, n=nuc_df, co=cond_order_arg: plot_burst_size_distribution(ax, n, condition_order=co)),
        (44, "nuclear_overlap_to_cyto_rna1_ratio", "(Nuclear RNA1+RNA2 overlap) / (cytoplasmic RNA1) — per cell",
            lambda ax, n=nuc_df, co=cond_order_arg: plot_nascent_to_mature_ratio(ax, n, condition_order=co)),
        (45, "nuclear_overlap_to_edge_distance", "Nuclear-overlap-to-nuclear-edge distance",
            lambda ax, sp=spots_df, n=nuc_df: plot_tss_to_edge_distance(ax, sp, n)),
        # === Block G: two-RNA general analyses ===
        (46, "coexpression_quadrants", "Co-expression quadrants",
            lambda ax, n=nuc_df, co=cond_order_arg: plot_coexpression_quadrants(ax, n, condition_order=co)),
        (47, "per_condition_pearson_r", "Per-condition RNA1↔RNA2 Pearson r",
            lambda ax, n=nuc_df, co=cond_order_arg: plot_per_condition_pearson_r(ax, n, condition_order=co)),
        (48, "cytoplasmic_clustering", "Cytoplasmic clustering NN",
            lambda ax, sp=spots_df: plot_cytoplasmic_nn_clustering(ax, sp)),
        # === Block H: cross-condition difference figures ===
        (49, "volcano_like_per_metric", "Volcano-like per-metric",
            lambda ax, n=nuc_df, co=cond_order_arg: plot_volcano_like_per_metric(ax, n, condition_order=co)),
        (50, "effect_size_bars", "Per-condition effect sizes (Cohen's d)",
            lambda ax, n=nuc_df, co=cond_order_arg: plot_effect_size_bars(ax, n, condition_order=co)),
        (51, "per_image_variance_decomposition", "Per-image variance decomposition",
            lambda ax, n=nuc_df, co=cond_order_arg: plot_per_image_variance(ax, n, condition_order=co)),
        # === Block I: COMPOSITION figures (2026-05-18 Brian) ============
        # Shift from absolute counts to per-condition % composition.
        # Stacked horizontal bars sum to 100% per condition; the
        # interesting cross-condition shift (e.g. "WT is 30% nuclear, KO
        # is 60% nuclear") reads visually instead of being buried in raw
        # counts. Sec-only bars render as a single "no spots detected"
        # gray bar — correct visual signal for a background control.
        (52, "composition_rna1_nuc_vs_cyto", "RNA1 nuclear vs cytoplasmic — composition",
            lambda ax, s=summary_df, co=cond_order_arg: plot_composition_rna1_nuc_vs_cyto(ax, s, condition_order=co)),
        (53, "composition_rna2_nuc_vs_cyto", "RNA2 nuclear vs cytoplasmic — composition",
            lambda ax, s=summary_df, co=cond_order_arg: plot_composition_rna2_nuc_vs_cyto(ax, s, condition_order=co)),
        (54, "composition_overlap_vs_solo_rna1", "RNA1 spots overlapping RNA2 — composition",
            lambda ax, s=summary_df, co=cond_order_arg: plot_composition_overlap_vs_solo_rna1(ax, s, condition_order=co)),
        (55, "composition_overlap_vs_solo_rna2", "RNA2 spots overlapping RNA1 — composition",
            lambda ax, s=summary_df, co=cond_order_arg: plot_composition_overlap_vs_solo_rna2(ax, s, condition_order=co)),
        (56, "composition_summary_panel", "Composition summary (3×2)",
            lambda ax, s=summary_df, sp=spots_df, co=cond_order_arg:
                plot_composition_summary_panel(ax, s, spots=sp, condition_order=co)),
        # === Block I-bis: per-spot-type localization + RNA1↔RNA2 comparisons
        # (2026-05-18 Brian, round 2). 57–62 sit alongside the composition
        # block; they re-frame the same nuclei_metrics + spot_metrics data
        # as cross-channel comparisons, not per-channel raw counts.
        (57, "localization_composition_both_channels",
            "Spot localization composition — both channels",
            lambda ax, s=summary_df, co=cond_order_arg:
                plot_localization_composition_both_channels(ax, s, condition_order=co)),
        (58, "rna1_vs_rna2_nuclear_fraction_scatter",
            "RNA1 vs RNA2 nuclear fraction — per cell",
            lambda ax, n=nuc_df, co=cond_order_arg:
                plot_rna1_vs_rna2_nuclear_fraction_scatter(ax, n, condition_order=co)),
        (59, "rna1_minus_rna2_nuclear_fraction",
            "Per-cell Δ nuclear fraction (RNA1 − RNA2)",
            lambda ax, n=nuc_df, co=cond_order_arg:
                plot_rna1_minus_rna2_nuclear_fraction(ax, n, condition_order=co)),
        (60, "overlap_location_split",
            "Overlap events — nuclear vs cytoplasmic split",
            lambda ax, sp=spots_df, co=cond_order_arg:
                plot_overlap_location_split(ax, sp, condition_order=co)),
        (61, "rna1_to_rna2_spot_ratio_per_cell",
            "Per-cell RNA1:RNA2 spot ratio (log10)",
            lambda ax, n=nuc_df, co=cond_order_arg:
                plot_rna1_to_rna2_spot_ratio_per_cell(ax, n, condition_order=co)),
        (62, "nuclear_overlap_fraction_of_nuclear_rna1",
            "Nuclear overlap fraction of nuclear RNA1",
            lambda ax, n=nuc_df, co=cond_order_arg:
                plot_nuclear_overlap_fraction_of_nuclear_rna1(ax, n, condition_order=co)),
        # === Block J: COMPOSITION COMPANIONS (b-suffix; 2026-05-18 Brian) =====
        # For each raw-count figure whose underlying biological question is
        # really "what fraction of cells / spots / nuclei does X?", add a
        # b-suffix panel that plots the per-condition composition. These
        # sit alphabetically next to their raw-count counterparts in the
        # figures/ directory (01_, 01b_, 02_, 02b_, ...) and make WT-vs-KO
        # shifts visually obvious. 5-tuple = (idx, slug, title, fn, suffix);
        # suffix "b" produces "01b_slug.png".
        (1,  "pct_cells_with_spots_rna1",
            "% cells with ≥1/≥5/≥10 RNA1 spots — by condition",
            lambda ax, n=nuc_df, co=cond_order_arg: plot_pct_cells_with_spots_rna1(ax, n, condition_order=co),
            "b"),
        (2,  "pct_cells_with_spots_rna2",
            "% cells with ≥1/≥5/≥10 RNA2 spots — by condition",
            lambda ax, n=nuc_df, co=cond_order_arg: plot_pct_cells_with_spots_rna2(ax, n, condition_order=co),
            "b"),
        (8,  "nuclear_spots_bin_composition_rna1",
            "Nuclear RNA1 spot-count bin composition — by condition",
            lambda ax, n=nuc_df, co=cond_order_arg: plot_nuclear_spots_bin_composition_rna1(ax, n, condition_order=co),
            "b"),
        (9,  "nuclear_spots_bin_composition_rna2",
            "Nuclear RNA2 spot-count bin composition — by condition",
            lambda ax, n=nuc_df, co=cond_order_arg: plot_nuclear_spots_bin_composition_rna2(ax, n, condition_order=co),
            "b"),
        (10, "cyto_spots_bin_composition_rna1",
            "Cytoplasmic RNA1 spot-count bin composition — by condition",
            lambda ax, n=nuc_df, co=cond_order_arg: plot_cyto_spots_bin_composition_rna1(ax, n, condition_order=co),
            "b"),
        (11, "cyto_spots_bin_composition_rna2",
            "Cytoplasmic RNA2 spot-count bin composition — by condition",
            lambda ax, n=nuc_df, co=cond_order_arg: plot_cyto_spots_bin_composition_rna2(ax, n, condition_order=co),
            "b"),
        (15, "frac_nuclear_box_rna1",
            "Per-cell nuclear fraction of RNA1 spots — by condition",
            lambda ax, n=nuc_df, co=cond_order_arg: plot_frac_nuclear_box_rna1(ax, n, condition_order=co),
            "b"),
        (16, "frac_nuclear_box_rna2",
            "Per-cell nuclear fraction of RNA2 spots — by condition",
            lambda ax, n=nuc_df, co=cond_order_arg: plot_frac_nuclear_box_rna2(ax, n, condition_order=co),
            "b"),
        (17, "per_cell_nc_stacked_rna1",
            "Average per-cell nuc vs cyto — RNA1 by condition",
            lambda ax, n=nuc_df, co=cond_order_arg: plot_per_cell_nc_stacked_rna1(ax, n, condition_order=co),
            "b"),
        (18, "per_cell_nc_stacked_rna2",
            "Average per-cell nuc vs cyto — RNA2 by condition",
            lambda ax, n=nuc_df, co=cond_order_arg: plot_per_cell_nc_stacked_rna2(ax, n, condition_order=co),
            "b"),
        (35, "pct_nuclei_with_overlap",
            "% of nuclei with ≥1 RNA1↔RNA2 overlap — by condition",
            lambda ax, n=nuc_df, co=cond_order_arg: plot_pct_nuclei_with_overlap(ax, n, condition_order=co),
            "b"),
        (36, "composition_overlap_location",
            "Overlap events — nuclear vs cytoplasmic composition",
            lambda ax, sp=spots_df, co=cond_order_arg: plot_composition_overlap_location(ax, sp, condition_order=co),
            "b"),
        (40, "pct_nuclei_with_nuc_overlap",
            "% of nuclei with ≥1 nuclear RNA1↔RNA2 overlap — by condition",
            lambda ax, n=nuc_df, co=cond_order_arg: plot_pct_nuclei_with_nuc_overlap(ax, n, condition_order=co),
            "b"),
        (41, "pct_cells_with_cyto_rna1",
            "% of cells with cytoplasmic RNA1 spots — by condition",
            lambda ax, n=nuc_df, co=cond_order_arg: plot_pct_cells_with_cyto_rna1(ax, n, condition_order=co),
            "b"),
        # ===== COLOCALIZATION SuperPlots (07_coloc) — appended only when the
        # per-nucleus coloc columns are present (always true for rna_rna /
        # rna_protein here). Same deck as the legacy layout. =====
    ] + _maybe_add_coloc_panels(nuc_df, cond_order_arg, summary_df, spots_df)


# Colocalization SuperPlot deck shared by BOTH layouts (rna_only-legacy +
# rna_rna). Returns [] when the nuclei table lacks coloc columns so a pure
# rna_only run is byte-identical to before. Figure indices 70-73 + the
# ``coloc_`` slug prefix route every panel into figures/07_coloc/ via
# _FIGURE_SLUG_SUBFOLDER. Titles/axes name channels from _LABELS.
def _maybe_add_coloc_panels(nuc_df, cond_order_arg, summary_df=None, spots_df=None):
    if not has_coloc_columns(nuc_df):
        return []
    return [
        (70, "coloc_superplot_spot_pairing",
            "SuperPlot: spot pairing — RNA × partner — by condition",
            lambda ax, n=nuc_df, co=cond_order_arg: coloc_spot_pairing_superplot(ax, n, condition_order=co)),
        (71, "coloc_superplot_manders_m1_rna_in_protein",
            "SuperPlot: Manders M1 (RNA in protein) — by condition",
            lambda ax, n=nuc_df, co=cond_order_arg: coloc_manders_m1_superplot(ax, n, condition_order=co)),
        (72, "coloc_superplot_manders_m2_protein_in_rna",
            "SuperPlot: Manders M2 (protein in RNA) — by condition",
            lambda ax, n=nuc_df, co=cond_order_arg: coloc_manders_m2_superplot(ax, n, condition_order=co)),
        (73, "coloc_superplot_pearson",
            "SuperPlot: Pearson r (RNA vs partner) — by condition",
            lambda ax, n=nuc_df, co=cond_order_arg: coloc_pearson_superplot(ax, n, condition_order=co)),
        # Intensity-based, spot-centric, FLOOR-ROBUST coloc (Brian 2026-05-29).
        (74, "coloc_superplot_partner_intensity_at_rna_spots",
            "SuperPlot: partner intensity at RNA spots (raw) — by condition",
            lambda ax, n=nuc_df, co=cond_order_arg: coloc_partner_intensity_at_rna_spots_superplot(ax, n, condition_order=co)),
        (75, "coloc_superplot_partner_enrichment_at_rna_spots",
            "SuperPlot: partner enrichment at RNA foci — by condition",
            lambda ax, n=nuc_df, co=cond_order_arg: coloc_partner_enrichment_at_rna_spots_superplot(ax, n, condition_order=co)),
        # 2026-06-05 Brian: PROPER coloc statistic — partner intensity at RNA
        # foci vs a per-nucleus random-position null (>1.0 = enriched over
        # chance). Optional: self-skips when the enrichment-vs-null column is
        # absent (older runs / non-rna_protein). Annotated with pooled null
        # z / empirical-p from per_image_summary.
        (76, "coloc_superplot_partner_enrichment_vs_null_at_rna_spots",
            "SuperPlot: partner enrichment at RNA foci (vs per-nucleus null) — by condition",
            lambda ax, n=nuc_df, s=summary_df, co=cond_order_arg: coloc_partner_enrichment_vs_null_at_rna_spots_superplot(ax, n, summary=s, condition_order=co)),
        # 2026-06-06 Brian (a): DOSE-DEPENDENCE — per-spot MIAT brightness vs
        # local QKI intensity. In-grid here (Pass-2 emits 77_coloc_dose_*.png
        # into 07_coloc/); the faceted 77b + standalone 78/80/81 are rendered
        # by main()'s coloc block. Self-skips when partner_local_mean_intensity
        # is absent.
        (77, "coloc_dose_dependence_miat_vs_qki",
            "Dose-dependence: partner intensity vs RNA spot brightness — by condition",
            lambda ax, sp=spots_df, co=cond_order_arg: coloc_dose_dependence_miat_vs_qki(ax, sp, condition_order=co)),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--output-dir", required=True,
        help="Path to a Fiji pipeline output directory containing the CSVs.",
    )
    parser.add_argument(
        "--out", default=None,
        help="Path for the output PNG (default: <output-dir>/single_condition_panel.png)",
    )
    parser.add_argument(
        "--prefix", default="",
        help="Output prefix used in the pipeline (matches OUTPUT_PREFIX). Empty by default.",
    )
    parser.add_argument(
        "--title", default=None,
        help="Custom panel title. Overrides EXPERIMENT_METADATA.experiment_name "
             "from run_config.json, which itself overrides the output dir name.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    if not out_dir.is_dir():
        sys.stderr.write(f"ERROR: not a directory: {out_dir}\n")
        return 1
    # 2026-05-28 Brian rule "no loose PNGs at run-dir root": route the back-compat
    # alias INTO figures/00_overview/ so it isn't a flat file at the run-dir root.
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = out_dir / "figures" / "00_overview" / "single_condition_panel.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)

    nuc, spots, summary = load_outputs(out_dir)
    metadata = load_metadata(out_dir)
    cond_order = load_condition_order(out_dir)
    analysis_mode = load_analysis_mode(out_dir)
    # 2026-05-19 Brian: pull channel display labels from run_config.json
    # before any plot function runs. _set_labels() installs the dict at
    # module scope; _relabel_fig() consults it on every savefig.
    _labels_for_run = load_labels(out_dir)
    _set_labels(_labels_for_run)
    print(f"Loaded {len(nuc)} nuclei, {len(spots)} spots, "
          f"{len(summary) if summary is not None else 0} image summaries; "
          f"ANALYSIS_MODE={analysis_mode}; "
          f"CONDITION_ORDER={cond_order or '(alphabetical)'}; "
          f"labels=rna:{_labels_for_run['rna_label']!r} "
          f"rna2:{_labels_for_run['rna2_label']!r} "
          f"dapi:{_labels_for_run['dapi_label']!r}")

    # 2026-05-29 Brian: build the per-run SuperPlot filter/criteria subtitle.
    # Pull the RNA spot intensity floor from the run-dir name (the canonical
    # place the manual peak-intensity floor is recorded, e.g.
    # "...RUN_introns1600-2200_XRN2-1100-4000..."), tagged with the RNA channel
    # label. Every SuperPlot then carries this floor + per-image-mean n line.
    try:
        import re as _re
        global _SUPERPLOT_FILTER_SUBTITLE
        _rna_lab = _labels_for_run.get("rna_label") or "RNA"
        _floor = None
        _m = _re.search(r"intron[s]?(\d+)", out_dir.name, _re.IGNORECASE)
        if _m is None:
            _m = _re.search(r"floor[_-]?(\d+)", out_dir.name, _re.IGNORECASE)
        if _m:
            _floor = int(_m.group(1))
        if _floor is not None:
            _SUPERPLOT_FILTER_SUBTITLE = f"{_rna_lab} spot floor {_floor}"
        else:
            _SUPERPLOT_FILTER_SUBTITLE = f"{_rna_lab} spots"
        print(f"  SuperPlot filter subtitle base: {_SUPERPLOT_FILTER_SUBTITLE!r}")
    except Exception as _e:
        print(f"  WARN: could not build SuperPlot filter subtitle: {_e}")

    # Title preference: CLI --title > metadata.experiment_name > folder name
    title_parts = []
    if args.title:
        title_parts.append(args.title)
    elif metadata.get("experiment_name"):
        title_parts.append(str(metadata["experiment_name"]))
    else:
        title_parts.append(out_dir.name)
    sub_bits = []
    if metadata.get("analyst"):
        sub_bits.append(f"analyst: {metadata['analyst']}")
    if metadata.get("date"):
        sub_bits.append(str(metadata["date"]))
    if metadata.get("notes"):
        sub_bits.append(str(metadata["notes"])[:80])

    def build_layout(nuc_df, spots_df, summary_df, cond_order_arg):
        """Return the (number, slug, title, render_fn) layout list. Built
        as a closure so per-image / per-condition passes can swap in a
        filtered nuc_df / spots_df without redefining 16 lambdas.

        Mode-aware: when run_config.json's ANALYSIS_MODE is 'rna_rna',
        returns the two-channel + spot-coloc layout (20 panels). Otherwise
        returns the legacy rna_only layout (16 panels). The legacy layout
        is unchanged so rna_only / rna_protein / Fiji runs behave exactly
        as before.
        """
        if analysis_mode == "rna_rna":
            return build_rna_rna_layout(nuc_df, spots_df, summary_df, cond_order_arg)
        # 2026-05-26 (publication polish, Brian): layout reordered to LEAD THE
        # STORY in Brian's stated priority — ① spot count / nucleus, ② total RNA
        # intensity / nucleus, ③ subcellular distribution (nuclear/cyto/
        # nucleolar), then ④ per-spot properties, ⑤ nucleus/chromatin, ⑥
        # nucleolus. The headline replicate-aware SuperPlots + by-condition
        # comparisons lead each block; the per-image distributions follow as
        # supporting detail. Every rna_only panel now carries a slug-prefix
        # routing entry in _FIGURE_SLUG_SUBFOLDER so subfolder placement is
        # driven by slug (stable) and the numeric index is free to encode the
        # story order without colliding with the shared rna_rna numeric map.
        return [
            # ===== ① SPOT COUNT PER NUCLEUS (lead) =====
            (1,  "sct_superplot_spots_per_nucleus", "SuperPlot: spots per nucleus — by condition",
                lambda ax, n=nuc_df, co=cond_order_arg: superplot_spots_per_cell(ax, n, condition_order=co)),
            (2,  "sct_box_spots_per_nucleus", "Spots per nucleus — by condition (box + per-image means)",
                lambda ax, n=nuc_df, co=cond_order_arg: plot_box_spots_by_condition(ax, n, condition_order=co)),
            (3,  "sct_collapsed_spots_per_nucleus", "Per-condition collapsed: spots per nucleus",
                lambda ax, n=nuc_df, co=cond_order_arg: collapsed_condition_means(ax, n, condition_order=co)),
            (4,  "sct_ranked_spots_per_nucleus", "Ranked per-image mean spots per nucleus",
                lambda ax, n=nuc_df, co=cond_order_arg: ranked_spots_per_cell(ax, n, condition_order=co)),
            (5,  "sct_pct_cells_with_spots", "% cells with ≥1/≥5/≥10 RNA spots — by condition",
                lambda ax, n=nuc_df, co=cond_order_arg: rna_only_pct_cells_with_spots(ax, n, condition_order=co)),
            (6,  "sct_spot_count_bin_composition", "RNA spot-count bin composition — by condition",
                lambda ax, n=nuc_df, co=cond_order_arg: rna_only_spot_count_bin_composition(ax, n, condition_order=co)),
            (7,  "sct_spots_per_nucleus_distribution", "Spots-per-nucleus distribution (per image)",
                lambda ax, n=nuc_df: plot_spots_per_nucleus(ax, n)),
            (8,  "sct_cumulative_spots_per_nucleus", "CDF: spots per nucleus (per image)",
                lambda ax, n=nuc_df: plot_cumulative_spots_per_cell(ax, n)),
            (9,  "sct_mean_spots_per_image", "Mean spots per nucleus, by image",
                lambda ax, n=nuc_df, s=summary_df: plot_spot_count_per_image(ax, s, n)),
            # ===== ② TOTAL RNA INTENSITY PER NUCLEUS =====
            (10, "rint_superplot_total_rna_intensity", "SuperPlot: total RNA intensity per nucleus — by condition",
                lambda ax, n=nuc_df, co=cond_order_arg: superplot_total_rna_per_cell(ax, n, condition_order=co)),
            (11, "rint_box_total_rna_intensity", "Total RNA intensity per nucleus — by condition",
                lambda ax, n=nuc_df, co=cond_order_arg: plot_box_total_intensity_by_condition(ax, n, condition_order=co)),
            (12, "rint_whole_nucleus_integrated", "Whole-nucleus integrated RNA intensity — by condition",
                lambda ax, n=nuc_df, co=cond_order_arg: plot_whole_nucleus_intensity_by_condition(ax, n, condition_order=co)),
            (13, "rint_ranked_total_rna_intensity", "Ranked per-image mean total RNA intensity per nucleus",
                lambda ax, n=nuc_df, co=cond_order_arg: ranked_total_rna_per_cell(ax, n, condition_order=co)),
            (14, "rint_superplot_per_nucleus_expression", "SuperPlot: per-nucleus mean spot intensity — by condition",
                lambda ax, n=nuc_df, co=cond_order_arg: superplot_per_cell_expression(ax, n, condition_order=co)),
            (15, "rint_per_nucleus_total_distribution", "Per-nucleus total RNA intensity distribution (per image)",
                lambda ax, n=nuc_df: plot_total_rna_per_cell(ax, n)),
            (16, "rint_per_nucleus_expression_distribution", "Per-nucleus expression intensity distribution (per image)",
                lambda ax, n=nuc_df: plot_per_cell_expression(ax, n)),
            (17, "rint_spot_peak_intensity_distribution", "Spot peak-intensity distribution (per image)",
                lambda ax, sp=spots_df: plot_intensity_distribution(ax, sp)),
            (18, "rint_sorted_spot_brightness", "Sorted spot-brightness curve (per image)",
                lambda ax, sp=spots_df: plot_spot_brightness_rank(ax, sp)),
            # ===== ③ SUBCELLULAR DISTRIBUTION (nuclear / cyto / nucleolar) =====
            (19, "loc_superplot_nuclear_fraction", "SuperPlot: per-nucleus nuclear spot fraction — by condition",
                lambda ax, n=nuc_df, co=cond_order_arg: superplot_nc_ratio_per_cell(ax, n, condition_order=co)),
            (20, "loc_superplot_nc_pixel_ratio", "SuperPlot: RNA N/C pixel-intensity ratio — by condition",
                lambda ax, n=nuc_df, co=cond_order_arg: nucleus_nc_ratio_superplot(ax, n, condition_order=co)),
            (21, "loc_ranked_nuclear_fraction", "Ranked per-image mean nuclear spot fraction",
                lambda ax, n=nuc_df, co=cond_order_arg: ranked_nuclear_fraction(ax, n, condition_order=co)),
            (22, "loc_nuclear_vs_cytoplasmic", "Nuclear vs cytoplasmic spots (per image)",
                lambda ax, sp=spots_df, co=cond_order_arg: plot_nuclear_vs_cytoplasmic(ax, sp, condition_order=co)),
            (23, "loc_superplot_spot_density", "SuperPlot: nuclear spot density (per µm²) — by condition",
                lambda ax, n=nuc_df, co=cond_order_arg: rna_only_spot_density_superplot(ax, n, condition_order=co)),
            (24, "loc_superplot_frac_spots_nuc_edge", "SuperPlot: per-nucleus fraction of spots at nuclear edge — by condition",
                lambda ax, n=nuc_df, co=cond_order_arg: rna_only_frac_spots_nuc_edge_superplot(ax, n, condition_order=co)),
            (25, "loc_superplot_spot_distance_to_edge", "SuperPlot: nuclear spot distance-to-edge — by condition",
                lambda ax, sp=spots_df, co=cond_order_arg: superplot_dist_to_edge(ax, sp, condition_order=co)),
            (26, "loc_spot_peripheral_bias_distribution", "Spot peripheral-vs-interior bias (per image)",
                lambda ax, sp=spots_df: plot_distance_to_edge(ax, sp)),
            # ===== ④ PER-SPOT PROPERTIES =====
            (27, "spotprop_superplot_peak_intensity", "SuperPlot: spot peak intensity — by condition",
                lambda ax, sp=spots_df, co=cond_order_arg: superplot_spot_peak_intensity(ax, sp, condition_order=co)),
            (28, "spotprop_superplot_size", "SuperPlot: spot size (diameter) — by condition",
                lambda ax, sp=spots_df, co=cond_order_arg: superplot_spot_size(ax, sp, condition_order=co)),
            (29, "spotprop_superplot_volume", "SuperPlot: spot volume — by condition",
                lambda ax, sp=spots_df, co=cond_order_arg: superplot_spot_volume(ax, sp, condition_order=co)),
            (30, "spotprop_superplot_local_snr", "SuperPlot: per-spot local SNR — by condition",
                lambda ax, sp=spots_df, co=cond_order_arg: superplot_local_snr(ax, sp, condition_order=co)),
            (31, "spotprop_superplot_anisotropy", "SuperPlot: per-spot anisotropy — by condition",
                lambda ax, sp=spots_df, co=cond_order_arg: rna_only_anisotropy_superplot(ax, sp, condition_order=co)),
            (32, "spotprop_size_distribution", "Spot size distribution (per image)",
                lambda ax, sp=spots_df: plot_diameter_distribution(ax, sp)),
            (33, "spotprop_volume_distribution", "Per-spot volume distribution (per image)",
                lambda ax, sp=spots_df: plot_spot_volume_distribution(ax, sp)),
            (34, "spotprop_local_snr_distribution", "Per-spot local SNR distribution (per image)",
                lambda ax, sp=spots_df: plot_local_snr_distribution(ax, sp)),
            (35, "spotprop_spots_vs_nucleus_area", "Spots per nucleus vs nucleus area",
                lambda ax, n=nuc_df: plot_spots_vs_nucleus_area(ax, n)),
            # ===== ⑤ NUCLEUS / CHROMATIN =====
            (36, "nucleus_area_superplot", "Nucleus area — by condition",
                lambda ax, n=nuc_df, co=cond_order_arg: nucleus_area_superplot(ax, n, condition_order=co)),
            (37, "nucleus_dapi_mean_superplot", "Whole-nucleus mean DAPI — by condition",
                lambda ax, n=nuc_df, co=cond_order_arg: nucleus_dapi_mean_superplot(ax, n, condition_order=co)),
            (38, "nucleus_dapi_cv_superplot", "Within-nucleus DAPI CV — by condition",
                lambda ax, n=nuc_df, co=cond_order_arg: nucleus_dapi_cv_superplot(ax, n, condition_order=co)),
            (39, "nucleus_heterochromatin_superplot", "Heterochromatin fraction — by condition",
                lambda ax, n=nuc_df, co=cond_order_arg: nucleus_heterochromatin_superplot(ax, n, condition_order=co)),
            (40, "nucleus_spot_fraction_superplot", "Per-nucleus nuclear spot fraction — by condition",
                lambda ax, n=nuc_df, co=cond_order_arg: nucleus_spot_fraction_superplot(ax, n, condition_order=co)),
            # ===== ⑥ NUCLEOLUS =====
            (41, "nucleolus_fraction_superplot", "Nucleolus area fraction of nucleus — by condition",
                lambda ax, n=nuc_df, co=cond_order_arg: nucleolus_fraction_superplot(ax, n, condition_order=co)),
            (42, "nucleolus_area_superplot", "Nucleolus area — by condition",
                lambda ax, n=nuc_df, co=cond_order_arg: nucleolus_area_superplot(ax, n, condition_order=co)),
            (43, "nucleolar_spot_fraction_superplot", "Per-nucleus nucleolar fraction of nuclear RNA — by condition",
                lambda ax, sp=spots_df, co=cond_order_arg: nucleolar_spot_fraction_superplot(ax, sp, condition_order=co)),
            (44, "nucleolus_collapsed_mean_nuclear_in_nucleolus", "Mean % nuclear RNA in nucleolus — by condition",
                lambda ax, sp=spots_df, co=cond_order_arg: nucleolus_collapsed_mean_nuclear_in_nucleolus(ax, sp, condition_order=co)),
            # ===== NEW publication panels (N1–N6, 2026-05-27 Brian) =====
            # Slug prefixes drive subfolder routing (see _FIGURE_SLUG_SUBFOLDER):
            #   spotprop_ -> 04_spot_properties, loc_ -> 03_localization,
            #   ovw_ -> 00_overview.
            (45, "spotprop_fewer_brighter_joint", "Per-nucleus spot count vs mean per-spot intensity — by condition",
                lambda ax, n=nuc_df, co=cond_order_arg: fewer_brighter_joint(ax, n, condition_order=co)),
            (46, "loc_composition_stacked", "Spot localization composition — by condition",
                lambda ax, sp=spots_df, co=cond_order_arg: loc_composition_stacked(ax, sp, condition_order=co)),
            (47, "ovw_ecdf_priority_metrics", "Cumulative distributions: spots/nucleus and total RNA intensity — by condition",
                lambda ax, n=nuc_df, co=cond_order_arg: ecdf_priority_metrics(ax, n, condition_order=co)),
            (48, "loc_nc_ratio_superplot", "Nuclear:cytoplasmic RNA ratio per nucleus — by condition",
                lambda ax, n=nuc_df, co=cond_order_arg: nc_ratio_superplot(ax, n, condition_order=co)),
            (49, "spotprop_per_spot_intensity_violin", "Per-spot peak intensity distribution — by condition",
                lambda ax, sp=spots_df, co=cond_order_arg: per_spot_intensity_violin(ax, sp, condition_order=co)),
            (50, "ovw_effect_size_summary", "NT→KD effect-size summary (per-image-mean stats)",
                lambda ax, n=nuc_df, co=cond_order_arg: effect_size_summary(ax, n, condition_order=co)),
            # ===== ⑦ COLOCALIZATION (07_coloc) — only when a partner channel
            # with per-nucleus coloc columns is present (rna_protein here; the
            # rna_rna layout gets its own copy). Appended via _maybe_add_coloc
            # below so a pure rna_only run never registers them. =====
        ] + _maybe_add_coloc_panels(nuc_df, cond_order_arg, summary_df, spots_df)

    PLOT_LAYOUT = build_layout(nuc, spots, summary, cond_order)

    # 2026-05-18 Brian: layout tuples are EITHER 4-tuple (idx, slug, title, fn)
    # OR 5-tuple (idx, slug, title, fn, suffix). The 5th element is a
    # filename suffix like "b" that lets composition COMPANIONS (e.g.
    # "01b_pct_cells_with_spots_rna1.png") sort alphabetically right next to
    # their raw-count counterparts ("01_spots_per_nucleus_distribution_rna1.png").
    # We normalize to a 5-tuple here so every downstream loop only sees one shape.
    def _normalize_layout(layout):
        out = []
        for row in layout:
            if len(row) == 4:
                idx, slug, title, fn = row
                out.append((idx, slug, title, fn, ""))
            elif len(row) == 5:
                out.append(tuple(row))
            else:
                raise ValueError(f"layout row must be 4- or 5-tuple, got {len(row)}")
        return out
    PLOT_LAYOUT = _normalize_layout(PLOT_LAYOUT)

    figures_dir = out_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    title_text = title_parts[0]
    subtitle_text = " · ".join(sub_bits) if sub_bits else None

    # Grid sizing: 4 cols, ceil(n / 4) rows. rna_only -> 4x4, rna_rna -> 5x4.
    # 2026-05-20 Brian: bumped figsize 24×PANEL_HEIGHT -> 30×max(24,...).
    # Per-subplot title fontsize dropped to 10pt. Combined panel doubles as
    # the back-compat ``single_condition_panel.png`` (copied below), so the
    # polish here flows through to that filename too.
    PANEL_COLS = 4
    PANEL_ROWS = int(np.ceil(len(PLOT_LAYOUT) / PANEL_COLS))
    PANEL_HEIGHT_IN = max(24, PANEL_ROWS * 4.5)

    # ── Pass 1: combined NxN panel (overview) ──
    # Per-panel PNGs (Pass 2) are saved at 600 DPI per Brian's figure-format
    # preference.  The combined / overview panel is a contact-sheet summary
    # (not a publication figure) and its Agg buffer at 600 DPI can exceed 1e8
    # pixels for large rna_rna layouts, causing a MemoryError on allocation.
    # 2026-06-20 fix: save the combined panel at COMBINED_DPI (150) so the
    # buffer stays well under the safe ~1.2e8-px cap.  Per-panel PNGs keep
    # their full PNG_DPI.  The savefig is also guarded: a MemoryError there
    # logs a clear warning and lets the run continue so per-panel PNGs + CSVs
    # (the important outputs) are never lost.
    PNG_DPI = 600          # per-panel standalone figures (Pass 2)
    COMBINED_DPI = 150     # combined overview panel only — avoids Agg MemoryError
    fig = plt.figure(figsize=(30, PANEL_HEIGHT_IN), dpi=COMBINED_DPI)
    gs = GridSpec(PANEL_ROWS, PANEL_COLS, figure=fig, hspace=0.7, wspace=0.4)
    axes = [fig.add_subplot(gs[i, j]) for i in range(PANEL_ROWS) for j in range(PANEL_COLS)]
    for (idx, _slug, _title, fn, _suf), ax in zip(PLOT_LAYOUT, axes):
        try:
            fn(ax)
        except Exception as e:
            print(f"  WARN: panel {idx} ({_slug}) raised {type(e).__name__}: {e}")
            ax.set_visible(False)
    fig.suptitle(title_text, fontsize=14, fontweight="bold")
    if subtitle_text:
        fig.text(0.5, 0.945, subtitle_text, ha="center", fontsize=9, color="#555")
    # Shrink every cell's title to 10pt so suptitle / subtitle don't get
    # crowded by per-axis titles in this densely-packed overview.
    for _cell in axes:
        try:
            if _cell.get_visible():
                _cell.title.set_fontsize(10)
        except Exception:
            pass
    combined_png = figures_dir / "00_combined_panel.png"
    # Combined panel saved at COMBINED_DPI (150), not PNG_DPI (600).
    # bbox_inches="tight" also skipped for the combined panel: tight layout
    # on a 30×N-inch figure can itself spike peak memory; the subplots_adjust
    # call below pins the margins instead.
    _relabel_fig(fig)
    # 2026-05-20 Brian: also reserve the single_condition_panel layout
    # margins. wspace=0.35 / hspace=0.55 stops grouped x-tick labels from
    # bumping into adjacent subplots in the densest configurations.
    try:
        fig.subplots_adjust(wspace=0.35, hspace=0.55)
    except Exception:
        pass
    _final_layout_polish(fig, has_subtitle=bool(subtitle_text))
    try:
        fig.savefig(combined_png, dpi=COMBINED_DPI)
        print(f"  Combined panel written: {combined_png} ({combined_png.stat().st_size // 1024} KB)")
    except MemoryError as _mem_err:
        print(f"  WARN: combined panel skipped — MemoryError saving {combined_png}: {_mem_err}")
        print("  Per-panel PNGs and CSVs are unaffected.")
        combined_png = None
    except Exception as _save_err:
        print(f"  WARN: combined panel skipped — {type(_save_err).__name__}: {_save_err}")
        combined_png = None
    plt.close(fig)
    # Back-compat: also write at out_dir/single_condition_panel.png so
    # older tools/recipes that look there still find it.
    # Guard: combined_png is None when the save was skipped (MemoryError guard).
    if combined_png is not None:
        import shutil
        shutil.copy2(combined_png, out_path)

    # ── Pass 2: individual subplots, one PNG per panel ──
    # Filename = f"{idx:02d}{suffix}_{slug}.png" so b-suffix composition
    # companions land right next to their raw-count counterparts on disk.
    # 2026-05-19 Brian: do NOT add a suptitle here. Every plot_* function
    # already calls ax.set_title(...) with its own descriptive title;
    # adding "<folder_name> — <title>" on top produced a duplicate-title
    # render (bold up top + regular below). The folder name lives in the
    # filename, the plot title lives on the axis. Don't duplicate.
    # 2026-05-27 Brian: standalone panels are now rendered SERIALLY.
    #
    # ROOT CAUSE of the intermittent / non-deterministic dropped panels:
    # matplotlib is NOT thread-safe. Two process-global, lock-free caches are
    # mutated on every render and corrupt under concurrency:
    #   1. the mathtext parser (pyparsing packrat cache) — concurrent parses of
    #      tick labels like $\mathdefault{10^{n}}$ raise a spurious
    #      "ParseException at char 0", and
    #   2. the pyplot state machine + Agg renderer allocation path — concurrent
    #      plt.subplots()/savefig() calls can silently mangle a figure so its
    #      panel never lands on disk.
    # The earlier ThreadPoolExecutor (max_workers≈4) is exactly what triggered
    # both: panels 15/24/25/26 (and any panel emitting mathtext) were
    # intermittently dropped — sometimes WITHOUT a caught warning, because the
    # corruption surfaced inside an internal C draw call, not the Python try.
    #
    # The only fully correct fix for "matplotlib is not thread-safe" is to never
    # touch its global state concurrently. We render serially. There are ~45
    # small (7×5 in) panels; the wall-clock cost is dominated by 600-DPI PNG
    # encoding either way, and the determinism is non-negotiable for a
    # publication deck. (A module-level Lock around the whole draw+save would
    # also work but yields zero parallelism in practice — the GIL-releasing PNG
    # encode is the only parallelizable part and it is a small slice — so the
    # serial loop is both simpler and equally fast here.)
    #
    # Every panel is wrapped so a single failure is LOUD (printed) and never
    # silently swallowed, and never takes down the rest of the deck.
    def _render_one_subplot(spec):
        idx, slug, title, fn, suffix = spec
        sub_fig = None
        try:
            # 2026-05-29 Brian (FIX2): the prior pass FORCED every standalone
            # panel into a 5.5×6.5 square; Brian disliked the square shape and
            # the long 2-line SuperPlot title was overlapping the in-axes
            # significance bracket ("text on text"). Default to a comfortable
            # WIDE frame (7×5, the pre-square shape) for ordinary panels; for
            # SuperPlot panels switch to a gentle PORTRAIT whose WIDTH scales
            # with the number of conditions (resized AFTER fn() draws so the
            # condition count is known). NOT square, NOT crowded.
            sub_fig, sub_ax = plt.subplots(figsize=(7.0, 5.0), dpi=PNG_DPI)
            fn(sub_ax)
            out_pn = figures_dir / f"{idx:02d}{suffix}_{slug}.png"
            # 2026-05-29 Brian (SuperPlot publication polish): for SuperPlot
            # panels (detected by the per-condition n stashed on the axis),
            # (1) size a sensible NON-SQUARE portrait frame scaled by n_cond,
            # (2) shorten the title to ONE concise line (drop the redundant
            #     " — by condition" tail + the verbose method parenthetical),
            # (3) reserve a clear top band so the title + small grey subtitle
            #     clear the in-axes significance bracket — never overlap it.
            _is_superplot = getattr(sub_ax, "_superplot_n_by_cond", None) is not None
            if _is_superplot:
                # 2026-05-29 Brian (FIX3 — FULLY DETERMINISTIC SuperPlot layout):
                # the prior pass mixed an axes-relative title (set_title pad=30),
                # a figure-coord subtitle, tight_layout, a post-hoc
                # subplots_adjust, AND bbox_inches="tight" — those negotiate
                # against each other and the title/subtitle kept colliding.
                # The deterministic recipe pins EVERY band by hand and bypasses
                # tight_layout / bbox="tight" entirely so nothing can move:
                #   * reserved figure regions: top=0.82, bottom=0.20,
                #     left=0.14, right=0.96  (axes live strictly inside these)
                #   * title   -> fig.suptitle(y=0.965)  (clearly above axes)
                #   * subtitle-> fig.text(0.5, 0.895)    (gap below title,
                #                                         above axes top=0.82)
                #   * legend  -> fig.legend(loc='lower center', y=0.005) in the
                #               reserved bottom band, OUTSIDE the axes
                # The in-axes significance bracket keeps its current placement.
                _n_by = getattr(sub_ax, "_superplot_n_by_cond", None) or {}
                n_cond = max(1, len(_n_by))
                # Sensible landscape-ish frame; widen a touch per extra condition
                # so a 3+ deck never crowds. NOT square.
                fig_w = max(6.5, 5.2 + 0.9 * n_cond)
                sub_fig.set_size_inches(fig_w, 6.0)
                # Build the short 1-line title from the axes title the wrapper
                # set, then CLEAR the axes title (we render it as a suptitle).
                _t = sub_ax.get_title()
                _t1 = ""
                if _t:
                    _t1 = _t.split("(", 1)[0]            # drop method parenthetical
                    _t1 = " ".join(_t1.split()).strip()  # collapse wrap/newlines
                    if _t1.lower().startswith("superplot:"):
                        _t1 = _t1.split(":", 1)[1].strip()
                    for _suf in (" — by condition", " - by condition",
                                 "— by condition", "- by condition"):
                        if _t1.endswith(_suf):
                            _t1 = _t1[: -len(_suf)].rstrip(" —-").strip()
                            break
                    _t1 = _t1.rstrip(" —-").strip()
                sub_ax.set_title("")
                # 2026-06-06 Brian (FIX1/FIX3): the per-bracket stat detail now
                # lives in ONE footnote at the very bottom of the figure. Build
                # + wrap it here, size the reserved bottom band to fit it BELOW
                # the legend (which sits below the x-axis labels), and grow the
                # figure height + axes-bottom margin so the footnote never
                # collides with the x-axis tick labels — for any condition count.
                _foot_raw = _superplot_stats_footnote(sub_ax)
                _foot_txt = ""
                _n_foot_lines = 0
                if _foot_raw:
                    import textwrap as _tw
                    # Wrap width scales with figure width so wide 5–6 condition
                    # frames don't waste a tall footnote block.
                    _wrap_w = int(max(70, 11.0 * fig_w))
                    _foot_txt = "\n".join(_tw.wrap(_foot_raw, width=_wrap_w))
                    _n_foot_lines = _foot_txt.count("\n") + 1
                # Lay out the BOTTOM stack in INCHES from the figure bottom, then
                # convert to figure fractions. Stack (top→bottom inside the
                # reserved band): x-axis tick labels, the per-nucleus/mean legend,
                # then the stats footnote at the very bottom. Each gets its own
                # inch band + a small gap so nothing overlaps — for ANY condition
                # count. The footnote band grows with its wrapped line count; if
                # there is no footnote (e.g. <2 conditions, n too small) it
                # contributes 0 and the layout degrades gracefully.
                _XLAB_IN = 1.05          # room for the long wrapped x-tick labels
                _LEG_IN = 0.30           # one-line legend
                _GAP_IN = 0.18           # gap between bands
                _foot_line_in = 0.135
                _foot_in = (_foot_line_in * _n_foot_lines + 0.10) if _n_foot_lines else 0.0
                _bottom_band_in = _XLAB_IN + _GAP_IN + _LEG_IN + (
                    (_GAP_IN + _foot_in) if _foot_in else 0.0)
                # Keep the axes plotting region a constant ~3.6" tall regardless
                # of how much the footnote pushes the bottom band down.
                _AX_IN = 3.6
                _TOP_IN = 1.30           # title + grey subtitle band above axes
                _fig_h = _TOP_IN + _AX_IN + _bottom_band_in
                sub_fig.set_size_inches(fig_w, _fig_h)
                _bottom = _bottom_band_in / _fig_h
                _top = 1.0 - (_TOP_IN / _fig_h)
                # Inch positions (from fig bottom) of each lower element's anchor.
                _foot_top_y = (0.06) / _fig_h                      # va=bottom anchor
                _legend_y = (_foot_in + _GAP_IN) / _fig_h          # loc=lower center
                # Reserve the fixed bands FIRST so axes occupy exactly the
                # interior box; suptitle/subtitle/legend live in the margins.
                sub_fig.subplots_adjust(top=_top, bottom=_bottom,
                                        left=0.14, right=0.96)
                # Title + grey subtitle anchored in the TOP band (positions in
                # inches from the figure top so they track the figure height,
                # which now varies with the footnote line count).
                _title_y = 1.0 - (0.32 / _fig_h)
                _subt_y = 1.0 - (0.78 / _fig_h)
                if _t1:
                    sub_fig.suptitle(_wrap_title(_t1, width=52), y=_title_y,
                                     fontsize=12.5, fontweight="bold")
                # Grey italic descriptor, BELOW the title, ABOVE the axes.
                _sub_txt, _ = _wrap_subtitle(
                    _superplot_filter_subtitle(sub_ax), width=90)
                sub_fig.text(0.5, _subt_y, _sub_txt, ha="center", va="top",
                             fontsize=8.5, color="#555555", style="italic",
                             linespacing=1.2)
                # Deterministic OUTSIDE-below legend (per-nucleus vs per-image
                # mean) in the reserved bottom band, ABOVE the footnote.
                _lh = getattr(sub_ax, "_superplot_legend_handles", None)
                _ll = getattr(sub_ax, "_superplot_legend_labels", None)
                if _lh and _ll:
                    sub_fig.legend(_lh, _ll, loc="lower center",
                                   bbox_to_anchor=(0.5, _legend_y), ncol=2,
                                   frameon=False, fontsize=8.5)
                # Single stats footnote at the very bottom (cropable).
                if _foot_txt:
                    sub_fig.text(0.5, _foot_top_y, _foot_txt, ha="center",
                                 va="bottom", fontsize=6.5, color="#555555",
                                 linespacing=1.25)
                _relabel_fig(sub_fig)
                # NO tight_layout, NO bbox_inches="tight": the reserved bands ARE
                # the layout, so the canvas/figure box is exactly what we set.
                sub_fig.savefig(out_pn, dpi=PNG_DPI)
            else:
                _relabel_fig(sub_fig)
                _final_layout_polish(sub_fig, has_subtitle=False)
                # 2026-06-06 Brian: by-condition box/collapsed panels that drew
                # pairwise stat brackets via _annotate_pairwise_brackets now
                # carry ONLY a star above each bar; the verbose detail moves to
                # ONE footnote at the very bottom of the figure (same approach as
                # the SuperPlots). Draw it here from the records stashed on the
                # axis. bbox_inches="tight" includes the fig.text in the saved
                # canvas, so we only need to push the axes up (subplots_adjust
                # bottom) so the footnote never overlaps the x-tick labels.
                # Degrades gracefully: no records -> no footnote, no change.
                _foot_raw = _superplot_stats_footnote(sub_ax)
                if _foot_raw:
                    import textwrap as _tw
                    _w_in, _h_in = sub_fig.get_size_inches()
                    _wrap_w = int(max(70, 11.0 * _w_in))
                    _foot_txt = "\n".join(_tw.wrap(_foot_raw, width=_wrap_w))
                    _n_lines = _foot_txt.count("\n") + 1
                    # Grow the figure height + carve a bottom band for the
                    # footnote so it sits BELOW the (already packed) x-tick
                    # labels. ~0.135"/line + a fixed gap below the labels.
                    _foot_in = 0.135 * _n_lines + 0.18
                    _new_h = _h_in + _foot_in
                    sub_fig.set_size_inches(_w_in, _new_h)
                    # Re-seat the existing bottom margin so the band we just
                    # added is empty space under the axes/x-labels (not stolen
                    # from them). Convert current bottom (inches) -> new fraction.
                    _b0 = sub_fig.subplotpars.bottom
                    _new_bottom = (_b0 * _h_in + _foot_in) / _new_h
                    sub_fig.subplots_adjust(bottom=min(0.6, _new_bottom))
                    sub_fig.text(0.5, 0.06 / _new_h, _foot_txt, ha="center",
                                 va="bottom", fontsize=6.5, color="#555555",
                                 linespacing=1.25)
                sub_fig.savefig(out_pn, bbox_inches="tight", dpi=PNG_DPI)
            plt.close(sub_fig)
            return True
        except Exception as e:
            import traceback as _tb
            print(f"  WARN: standalone panel {idx}{suffix} ({slug}) FAILED "
                  f"{type(e).__name__}: {e}")
            print("    " + _tb.format_exc().replace("\n", "\n    ").rstrip())
            try:
                if sub_fig is not None:
                    plt.close(sub_fig)
            except Exception:
                pass
            return False

    _n_ok = 0
    for _spec in PLOT_LAYOUT:
        if _render_one_subplot(_spec):
            _n_ok += 1
    print(f"  Rendered {_n_ok}/{len(PLOT_LAYOUT)} standalone panels.")
    if _n_ok != len(PLOT_LAYOUT):
        print(f"  WARN: {len(PLOT_LAYOUT) - _n_ok} standalone panel(s) did not "
              f"render — see the per-panel FAILED lines above.")

    # ── Pass 2b: dedicated CORE + COLOC overview panels ──
    # 2026-05-19 Brian: only meaningful in rna_rna mode (these panels
    # combine RNA1×RNA2 figures). Skip silently for rna_only.
    if analysis_mode == "rna_rna":
        try:
            render_core_overview_panel(
                figures_dir / "97_CORE_overview_panel.png",
                nuc, spots, summary, cond_order,
            )
        except Exception as e:
            print(f"  WARN: 97_CORE_overview_panel raised {type(e).__name__}: {e}")
        try:
            render_coloc_overview_panel(
                figures_dir / "98_COLOC_overview_panel.png",
                nuc, spots, summary, cond_order,
            )
        except Exception as e:
            print(f"  WARN: 98_COLOC_overview_panel raised {type(e).__name__}: {e}")

    # ── Pass 2e: extended colocalization figures (2026-06-06 Brian) ──
    # Native 07_coloc/ deck additions: 77b faceted dose-dependence, 80 spot-
    # vs-threshold summary (plot-only) + 78 null-distribution overlay, 81 radial
    # QKI profile (need the coloc backfill CSVs; self-skip when absent). Panel
    # 77 (in-grid) is already emitted by Pass 2 + organized into 07_coloc/. Only
    # runs when the per-nucleus coloc columns exist (rna_protein / rna_rna); a
    # pure rna_only run is unaffected. Each call is wrapped so one missing
    # column/CSV never takes down the rest.
    if has_coloc_columns(nuc):
        _draws_path = out_dir / "coloc_null_draws.csv"
        _radial_path = out_dir / "coloc_radial_profile.csv"
        _draws_df = pd.read_csv(_draws_path) if _draws_path.exists() else None
        _radial_df = pd.read_csv(_radial_path) if _radial_path.exists() else None
        _coloc_renders = (
            ("77b_coloc_dose_dependence_faceted",
                lambda: coloc_dose_dependence_faceted(spots, out_dir, cond_order)),
            ("80_coloc_spot_vs_threshold",
                lambda: coloc_spot_vs_threshold_summary(nuc, summary, out_dir, cond_order)),
            ("78_coloc_null_distribution_overlay",
                lambda: coloc_null_distribution_overlay(out_dir, cond_order, _draws_df, summary)),
            ("81_coloc_radial_qki_profile",
                lambda: coloc_radial_profile_plot(_radial_df, out_dir, cond_order)),
        )
        for _cname, _cfn in _coloc_renders:
            try:
                _cfn()
            except Exception as e:
                print(f"  WARN: coloc figure {_cname} raised {type(e).__name__}: {e}")

    # ── Pass 2c: headline figures (formerly PI_FOCUS 99-104) ──
    # 2026-05-20 Brian: headline figures matching the headline Excel sheet.
    # 2026-05-21 Brian: moved into figures/headline/ subfolder and stripped
    # the "PI_FOCUS" prefix so figure filenames stay readable. Gated to
    # rna_rna mode (2×2 channel×compartment grids meaningless for
    # single-channel data). Each render function is wrapped in try/except so
    # one missing column doesn't take down the rest of the suite.
    if analysis_mode == "rna_rna":
        _headline_dir = figures_dir / "headline"
        _headline_dir.mkdir(parents=True, exist_ok=True)
        _headline_renders = (
            ("01_spot_counts_per_compartment.png",
                lambda p: render_pi_focus_spot_counts(p, nuc, cond_order)),
            ("02_above_floor_intensity.png",
                lambda p: render_pi_focus_above_floor(p, nuc, cond_order)),
            ("03_spot_peak_intensity_by_compartment.png",
                lambda p: render_pi_focus_spot_peak_intensity(p, spots, cond_order)),
            ("03b_spot_peak_intensity_violins.png",
                lambda p: render_pi_focus_spot_peak_intensity_violin(p, spots, cond_order)),
            ("04_spot_size_by_compartment.png",
                lambda p: render_pi_focus_spot_size(p, spots, cond_order)),
            ("05_localization_summary.png",
                lambda p: render_pi_focus_localization_summary(p, spots, cond_order)),
            ("06_overview_panel.png",
                lambda p: render_pi_focus_overview_panel(p, nuc, spots, cond_order)),
            # 2026-05-22 Brian: biology-headline figures distilled from the
            # cross-gene findings (nuclear retention shift + compartment
            # redistribution + KO/WT fold-changes + property shifts).
            ("07_nuclear_retention_shift.png",
                lambda p: render_headline_nuclear_retention(p, nuc, cond_order)),
            ("08_compartment_redistribution.png",
                lambda p: render_headline_compartment_redistribution(p, nuc, cond_order)),
            ("09_ko_wt_log2fc_counts.png",
                lambda p: render_headline_ko_wt_log2fc_panel(p, nuc, cond_order)),
            ("10_property_shifts_per_compartment.png",
                lambda p: render_headline_property_shifts(p, nuc, spots, cond_order)),
        )
        for _fname, _fn in _headline_renders:
            try:
                _fn(_headline_dir / _fname)
            except Exception as e:
                print(f"  WARN: headline/{_fname} raised {type(e).__name__}: {e}")

    # ── Pass 2d: pixel-intensity panels (spot-detection-independent) ──
    # 2026-05-24 Brian: whole-nucleus pixel mean + N/C pixel-intensity
    # ratio per nucleus. Both panels read per-nucleus columns that exist
    # for every analysis mode (rna_only / rna_rna / rna_protein) when the
    # nucleus + cytoplasm masks were measured (the standard pipeline path).
    # NOT gated to rna_rna — the columns exist in all modes; the figure
    # just renders the Exon panel as empty when only rna1 is measured.
    # Lands in figures/12_pixel_intensity/ so the deliverable subfolder
    # name slots cleanly after the existing 11_cross_channel/ themed dir.
    _pixel_intensity_dir = figures_dir / "12_pixel_intensity"
    _pixel_intensity_dir.mkdir(parents=True, exist_ok=True)
    _pixel_intensity_renders = (
        ("fig_whole_nucleus_pixel_intensity.png",
            lambda p: render_whole_nucleus_pixel_intensity(p, nuc, cond_order)),
        ("fig_nuc_cyto_pixel_intensity_ratio.png",
            lambda p: render_nuc_cyto_pixel_intensity_ratio(p, nuc, cond_order)),
    )
    for _fname, _fn in _pixel_intensity_renders:
        try:
            _fn(_pixel_intensity_dir / _fname)
        except Exception as e:
            print(f"  WARN: 12_pixel_intensity/{_fname} raised "
                  f"{type(e).__name__}: {e}")

    # ── Pass 3: per-image figures ──
    # One 4x4 panel per image (filtered nuc/spots), saved into
    # figures/per_image/<short_label>__panel.png. Lets a reader open one
    # PNG per image and see only that image's data without the cross-
    # image overlay clutter.
    per_image_dir = figures_dir / "per_image"
    per_image_dir.mkdir(exist_ok=True)
    image_names = nuc["image"].unique() if "image" in nuc.columns else []
    img_label_map = _build_image_labels(nuc) if len(image_names) else {}
    n_per_image = 0
    for img_name in image_names:
        sub_nuc = nuc[nuc["image"] == img_name].copy()
        sub_spots = spots[spots["image"] == img_name].copy() if "image" in spots.columns else spots
        sub_summary = summary[summary["image"] == img_name] if (summary is not None and "image" in summary.columns) else summary
        sub_layout = _normalize_layout(build_layout(sub_nuc, sub_spots, sub_summary, cond_order))
        sub_rows = int(np.ceil(len(sub_layout) / PANEL_COLS))
        sub_h = max(18, sub_rows * 4.5)
        fig_i = plt.figure(figsize=(24, sub_h), dpi=PNG_DPI)
        gs_i = GridSpec(sub_rows, PANEL_COLS, figure=fig_i, hspace=0.7, wspace=0.4)
        axes_i = [fig_i.add_subplot(gs_i[i, j]) for i in range(sub_rows) for j in range(PANEL_COLS)]
        for (idx, _slug, _title, fn, _suf), ax_i in zip(sub_layout, axes_i):
            try:
                fn(ax_i)
            except Exception as e:
                print(f"  WARN: per-image[{img_name}] panel {idx} raised {type(e).__name__}: {e}")
                ax_i.set_visible(False)
        short = img_label_map.get(img_name, short_label(img_name)).replace(" ", "_").replace("/", "_")
        # 2026-05-19 Brian: drop folder-name prefix from suptitle.
        # Image label alone is the meaningful descriptor here.
        fig_i.suptitle(f"image: {img_label_map.get(img_name, img_name)}",
                       fontsize=14, fontweight="bold")
        if subtitle_text:
            fig_i.text(0.5, 0.945, subtitle_text, ha="center", fontsize=9, color="#555")
        out_i = per_image_dir / f"{short}__panel.png"
        _relabel_fig(fig_i)
        _final_layout_polish(fig_i, has_subtitle=bool(subtitle_text))
        # Per-image panels are contact sheets; save at COMBINED_DPI (150) to
        # stay under the Agg MemoryError threshold.  Guard just in case.
        try:
            fig_i.savefig(out_i, dpi=COMBINED_DPI)
        except MemoryError as _mem_err:
            print(f"  WARN: per-image panel {out_i.name} skipped — MemoryError: {_mem_err}")
        except Exception as _save_err:
            print(f"  WARN: per-image panel {out_i.name} skipped — {type(_save_err).__name__}: {_save_err}")
        plt.close(fig_i)
        n_per_image += 1

    # ── Pass 4: per-condition figures ──
    # One 4x4 panel per condition (filtered nuc/spots), saved into
    # figures/per_condition/<condition>__panel.png. Aggregates all
    # images in a condition; useful when N images per group >1.
    per_cond_dir = figures_dir / "per_condition"
    per_cond_dir.mkdir(exist_ok=True)
    n_per_cond = 0
    if "condition" in nuc.columns:
        cond_unique = nuc["condition"].dropna().unique().tolist()
        for cond in order_conditions(cond_unique, cond_order):
            sub_nuc = nuc[nuc["condition"] == cond].copy()
            sub_spots = spots[spots["condition"] == cond].copy() if "condition" in spots.columns else spots
            sub_summary = summary[summary["condition"] == cond] if (summary is not None and "condition" in summary.columns) else summary
            sub_layout = _normalize_layout(build_layout(sub_nuc, sub_spots, sub_summary, cond_order))
            sub_rows = int(np.ceil(len(sub_layout) / PANEL_COLS))
            sub_h = max(18, sub_rows * 4.5)
            fig_c = plt.figure(figsize=(24, sub_h), dpi=PNG_DPI)
            gs_c = GridSpec(sub_rows, PANEL_COLS, figure=fig_c, hspace=0.7, wspace=0.4)
            axes_c = [fig_c.add_subplot(gs_c[i, j]) for i in range(sub_rows) for j in range(PANEL_COLS)]
            for (idx, _slug, _title, fn, _suf), ax_c in zip(sub_layout, axes_c):
                try:
                    fn(ax_c)
                except Exception as e:
                    print(f"  WARN: per-condition[{cond}] panel {idx} raised {type(e).__name__}: {e}")
                    ax_c.set_visible(False)
            cond_safe = str(cond).replace(" ", "_").replace("/", "_")
            # 2026-05-19 Brian: drop folder-name prefix from suptitle.
            # Condition label alone is the meaningful descriptor here.
            fig_c.suptitle(f"condition: {cond}",
                           fontsize=14, fontweight="bold")
            if subtitle_text:
                fig_c.text(0.5, 0.945, subtitle_text, ha="center", fontsize=9, color="#555")
            out_c = per_cond_dir / f"{cond_safe}__panel.png"
            _relabel_fig(fig_c)
            _final_layout_polish(fig_c, has_subtitle=bool(subtitle_text))
            # Per-condition panels are contact sheets; save at COMBINED_DPI (150)
            # to stay under the Agg MemoryError threshold.  Guard just in case.
            try:
                fig_c.savefig(out_c, dpi=COMBINED_DPI)
            except MemoryError as _mem_err:
                print(f"  WARN: per-condition panel {out_c.name} skipped — MemoryError: {_mem_err}")
            except Exception as _save_err:
                print(f"  WARN: per-condition panel {out_c.name} skipped — {type(_save_err).__name__}: {_save_err}")
            plt.close(fig_c)
            n_per_cond += 1

    # ── Pass 5: organize figures into thematic subfolders ──
    # 2026-05-21 Brian: previously all 60+ standalone PNGs sat flat in
    # figures/. Reorganize into themed subfolders so the deliverable is
    # browsable.
    _organize_flat_figures_into_subfolders(figures_dir)

    _combined_label = str(combined_png) if combined_png is not None else "(combined panel skipped)"
    print(f"Wrote {_combined_label} (+ {len(PLOT_LAYOUT)} standalone subplots, "
          f"{n_per_image} per-image panels in {per_image_dir.name}/, "
          f"{n_per_cond} per-condition panels in {per_cond_dir.name}/)")
    return 0


# ── Subfolder organization map (filename prefix → subfolder name) ──
# 2026-05-21 Brian: figures flat-listed in figures/ are now sorted into
# themed subfolders for browsability. Match on the leading number+optional
# 'b' suffix (e.g. "01", "08b") to preserve the existing naming scheme.
_FIGURE_SUBFOLDER_MAP = {
    # 00_overview — top-level summary panels
    "00": "00_overview", "97": "00_overview", "98": "00_overview",
    # 01_spot_counts — distributions of total spots per cell
    "01": "01_spot_counts", "01b": "01_spot_counts",
    "02": "01_spot_counts", "02b": "01_spot_counts",
    "03": "01_spot_counts", "04": "01_spot_counts", "05": "01_spot_counts",
    # 02_localization — N vs C distributions, fractions
    "06": "02_localization", "07": "02_localization",
    "08": "02_localization", "08b": "02_localization",
    "09": "02_localization", "09b": "02_localization",
    "10": "02_localization", "10b": "02_localization",
    "11": "02_localization", "11b": "02_localization",
    "12": "02_localization",
    # 03_nc_box_plots — N/C ratio + box plots per channel
    "13": "03_nc_box_plots", "14": "03_nc_box_plots",
    "15": "03_nc_box_plots", "15b": "03_nc_box_plots",
    "16": "03_nc_box_plots", "16b": "03_nc_box_plots",
    "17": "03_nc_box_plots", "17b": "03_nc_box_plots",
    "18": "03_nc_box_plots", "18b": "03_nc_box_plots",
    # 04_intensity — per-cell intensity boxes
    "19": "04_intensity", "20": "04_intensity",
    "21": "04_intensity", "22": "04_intensity",
    # 05_spot_properties — peak, size, SNR, brightness, area
    "23": "05_spot_properties", "24": "05_spot_properties",
    "25": "05_spot_properties", "26": "05_spot_properties",
    "27": "05_spot_properties", "28": "05_spot_properties",
    "29": "05_spot_properties", "30": "05_spot_properties",
    "31": "05_spot_properties", "32": "05_spot_properties",
    # 06_colocalization — overlap & NN distance
    "33": "06_colocalization", "34": "06_colocalization",
    "35": "06_colocalization", "35b": "06_colocalization",
    "36": "06_colocalization", "36b": "06_colocalization",
    # 07_rna_rna_compare — RNA1 vs RNA2 scatter + correlations
    "37": "07_rna_rna_compare", "38": "07_rna_rna_compare",
    "47": "07_rna_rna_compare",
    # 08_overlap_advanced — deeper nuclear-overlap analyses
    "39": "08_overlap_advanced", "40": "08_overlap_advanced", "40b": "08_overlap_advanced",
    "41": "08_overlap_advanced", "41b": "08_overlap_advanced",
    "42": "08_overlap_advanced", "43": "08_overlap_advanced",
    "44": "08_overlap_advanced", "45": "08_overlap_advanced",
    "46": "08_overlap_advanced", "48": "08_overlap_advanced",
    # 09_statistics — volcano-style + variance decomposition
    "49": "09_statistics", "50": "09_statistics", "51": "09_statistics",
    # 10_composition — compositional plots
    "52": "10_composition", "53": "10_composition",
    "54": "10_composition", "55": "10_composition",
    "56": "10_composition", "57": "10_composition",
    # 11_cross_channel — RNA1-vs-RNA2 nuclear-fraction & ratio
    "58": "11_cross_channel", "59": "11_cross_channel",
    "60": "11_cross_channel", "61": "11_cross_channel",
    "62": "11_cross_channel",
    # 07_coloc — RNA × partner colocalization SuperPlots (numeric backstop;
    # primary routing is the ``coloc_`` slug prefix above).
    "70": "07_coloc", "71": "07_coloc", "72": "07_coloc", "73": "07_coloc",
    "74": "07_coloc", "75": "07_coloc", "76": "07_coloc",
}


# 2026-05-25 Brian: SLUG-prefix overrides for the rna_only SuperPlot + ranked
# deck. The numeric subfolder map (_FIGURE_SUBFOLDER_MAP) is shared with the
# rna_rna layout, where 17–29 mean different figures. Routing the new rna_only
# panels by their slug (which is unique) takes priority over the numeric map so
# they land in dedicated, browsable subfolders regardless of index reuse.
_FIGURE_SLUG_SUBFOLDER = (
    # 2026-05-26 (publication polish, Brian): rna_only deck reordered to lead
    # the story. Each panel's slug now starts with a story-block prefix, and
    # subfolders are numbered in the SAME priority order so the deliverable
    # reads top-to-bottom: ① spot count, ② RNA intensity, ③ localization,
    # ④ spot properties, ⑤ nucleus/chromatin, ⑥ nucleolus. Slug routing takes
    # priority over the numeric map (which is shared with the rna_rna layout),
    # so reusing indices 1-44 here is safe. "nucleolus_"/"nucleolar_" precede
    # the generic "nucleus_" prefix so they aren't swallowed.
    # 2026-05-27 Brian: NEW overview panels (N3 ECDF, N6 effect-size) route to
    # 00_overview via the ovw_ prefix (checked before the numeric map).
    ("ovw_", "00_overview"),
    ("sct_", "01_spot_count"),
    ("rint_", "02_rna_intensity"),
    ("loc_", "03_localization"),
    ("spotprop_", "04_spot_properties"),
    # 2026-05-29 Brian: RNA × partner-channel colocalization SuperPlots
    # (rna_protein / rna_rna). Checked before the generic numeric map.
    ("coloc_", "07_coloc"),
    ("nucleolus_", "06_nucleolus"),
    ("nucleolar_", "06_nucleolus"),
    ("nucleus_", "05_nucleus_chromatin"),
    # ---- legacy rna_only slugs (kept so older recipes / re-runs of an
    # un-renamed build still route correctly) ----
    ("superplot_", "04_spot_properties"),
    ("collapsed_condition_", "01_spot_count"),
    ("ranked_per_image_", "01_spot_count"),
    ("rna_only_spot_count_bin_composition", "01_spot_count"),
    ("rna_only_pct_cells_with_spots", "01_spot_count"),
    ("rna_only_spot_density_superplot", "03_localization"),
    ("rna_only_frac_spots_nuc_edge_superplot", "03_localization"),
    ("rna_only_anisotropy_superplot", "04_spot_properties"),
)


def _organize_flat_figures_into_subfolders(figures_dir):
    """Move flat PNGs in figures/ into thematic subfolders. Slug-prefix
    overrides (_FIGURE_SLUG_SUBFOLDER) are checked first, then the leading
    numeric token via _FIGURE_SUBFOLDER_MAP. Files matching neither are left
    in place. Existing subdirs (headline/, per_image/, per_condition/) are
    untouched.
    """
    import re, shutil
    pattern = re.compile(r"^(\d{1,3}b?)_(.*)\.png$", re.IGNORECASE)
    moved = 0
    for entry in sorted(figures_dir.iterdir()):
        if not entry.is_file() or entry.suffix.lower() != ".png":
            continue
        m = pattern.match(entry.name)
        if not m:
            continue
        key = m.group(1)
        slug = m.group(2)
        # Slug-prefix override first (rna_only SuperPlot / ranked deck).
        sub_name = None
        for _pref, _dir in _FIGURE_SLUG_SUBFOLDER:
            if slug.startswith(_pref):
                sub_name = _dir
                break
        if sub_name is None:
            sub_name = _FIGURE_SUBFOLDER_MAP.get(key)
        if sub_name is None:
            continue
        sub_dir = figures_dir / sub_name
        sub_dir.mkdir(parents=True, exist_ok=True)
        target = sub_dir / entry.name
        try:
            shutil.move(str(entry), str(target))
            moved += 1
        except Exception as e:
            print(f"  WARN: could not move {entry.name} -> {sub_name}/: {e}")
    if moved:
        print(f"  Organized {moved} figures into thematic subfolders.")


if __name__ == "__main__":
    sys.exit(main())
