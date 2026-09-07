"""Figures for ``fishsuite report``, in the locked lab style.

Rules this module enforces, from the ``rnaseq-figure-style`` skill and the
``fishsuite-rna-fish`` stats convention:

* Okabe-Ito colours only; red and green are never paired.
* 600-dpi PNG plus an editable-text SVG for every figure.
* Every figure carries a one-line filter label and a single croppable footnote
  holding the test, the star legend, the exact p-values, the minimum detectable
  effect and n per group.
* Every figure footer names the producing run directory, the run's detection
  thresholds, the segmentation model and version, the replicate unit and the test.
* SuperPlots are dots over means: nuclei shaded BY WELL inside their condition
  group, field means as open circles, WELL means as the tested diamonds, and a
  heavy line at the mean of the well means.
* The star comes from the RAW Welch p. The Holm-adjusted p and the minimum
  detectable effect are printed in the footnote, never hidden.
"""
from __future__ import annotations

import glob
import json
import os
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg           # noqa: E402
import matplotlib.pyplot as plt            # noqa: E402
import numpy as np                         # noqa: E402
import pandas as pd                        # noqa: E402
from matplotlib.patches import Rectangle   # noqa: E402

from .stats import SEED, fmt_p, stars      # noqa: E402

OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "grey": "#333333",
}
# Cycled for groups without a locked colour. Green and vermillion are never
# adjacent in this order, so a two-group figure can never be red plus green.
PALETTE_CYCLE = [OKABE_ITO["grey"], OKABE_ITO["purple"], OKABE_ITO["sky"],
                 OKABE_ITO["orange"], OKABE_ITO["blue"], OKABE_ITO["green"],
                 OKABE_ITO["vermillion"]]
# Locked condition colours, in two scopes that must not be confused.
#
# TWO_GROUP_IMAGING applies ONLY to a two-group d8 cardiomyocyte WT-versus-KO set:
# Brian's per-paper BIN1 pair. Applying it to a multi-group set silently recolours
# WT and QKI-KO away from the key that set's own figures already use, which is what
# happened to the five-line exon/intron report on 2026-09-04.
TWO_GROUP_IMAGING = {
    "wt": "#595959",
    "control": "#595959",
    "qki-ko": "#D67AE5",
    "qki_ko": "#D67AE5",
    "qkiko": "#D67AE5",
}
# ANY_SIZE applies at any number of groups: the shared condition-colour map for the
# hESC knockdown conditions, and the fixed grey for the detection-floor control.
ANY_SIZE = {
    "miat-kd": OKABE_ITO["orange"],
    "miat_kd": OKABE_ITO["orange"],
    "miat-oe": OKABE_ITO["sky"],
    "miat_oe": OKABE_ITO["sky"],
    "nt": "#595959",
    "nt aso": "#595959",
    "secondary-only": "#999999",
    "sec-only": "#999999",
}
# Kept for callers that want the union; group_colors() does NOT use it directly.
LOCKED_GROUP_COLORS = {**TWO_GROUP_IMAGING, **ANY_SIZE}

DPI = 600
CROP_PX = 256


def group_colors(group_order: Sequence[str],
                 overrides: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """One colour per condition group.

    Precedence, highest first: an explicit ``overrides`` entry (the ``group_colors``
    block of a run's ``report_groups.yaml``), then a locked condition colour, then
    the next unused Okabe-Ito entry.

    The two-group imaging pair (WT #595959, QKI-KO #D67AE5) is applied ONLY when
    the design HAS two biological groups. A multi-group set takes the key recorded
    beside its run, or the cycle; recolouring WT and QKI-KO inside a five-line set
    would contradict that set's own published key.
    """
    order = [g for g in group_order]
    locked = dict(ANY_SIZE)
    if len([g for g in order if str(g) != "Secondary-only"]) == 2:
        locked.update(TWO_GROUP_IMAGING)
    for k, v in (overrides or {}).items():
        locked[str(k).strip().lower()] = str(v)
    out: Dict[str, str] = {}
    spare = [c for c in PALETTE_CYCLE]
    for g in order:
        hex_ = locked.get(str(g).strip().lower())
        if hex_:
            out[g] = hex_
            if hex_ in spare:
                spare.remove(hex_)
    for g in order:
        if g not in out:
            out[g] = spare.pop(0) if spare else OKABE_ITO["blue"]
    out.setdefault("Secondary-only", locked.get("secondary-only", "#999999"))
    return out


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "text.color": "black", "axes.labelcolor": "black",
        "axes.edgecolor": "black", "axes.linewidth": 0.8,
        "xtick.color": "black", "ytick.color": "black",
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7,
        "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })


def no_box(fig, ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_linewidth(0.8)
        ax.spines[s].set_color("black")
    fig.patch.set_edgecolor("none")


def shade(hex_color: str, frac: float):
    """Lighten a hex colour toward white; frac in (0, 1], 1 is the pure colour."""
    h = hex_color.lstrip("#")
    rgb = np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)]) / 255.0
    return tuple(1.0 - (1.0 - rgb) * float(np.clip(frac, 0.05, 1.0)))


def wrap_foot(text: str, fig_w: float, size: float) -> List[str]:
    width = max(60, int((fig_w - 0.40) * 72.0 / (size * 0.50)))
    out: List[str] = []
    for line in str(text).splitlines():
        out.extend(textwrap.wrap(line, width) or [""])
    return out


def foot_band(text: str, fig_w: float, fig_h: float, size: float, pad: float = 0.030) -> float:
    n = len(wrap_foot(text, fig_w, size))
    return 0.014 + (n * size * 1.45) / (fig_h * 72.0) + pad


def stamp_foot(fig, text: str, size: float = 5.6, y: float = 0.014) -> None:
    fig.text(0.016, y, "\n".join(wrap_foot(text, fig.get_figwidth(), size)),
             ha="left", va="bottom", fontsize=size, color="#333333", linespacing=1.45)


def stamp_head(fig, title: str, subtitle: str, filt: str, wrap: int = 132) -> float:
    """Title, run banner and filter line.

    Both bands wrap to the figure width, so a long run name or a long threshold
    string can never be clipped off the edge of the page.
    """
    fig_w, fig_h = fig.get_figwidth(), fig.get_figheight()
    # One text line, as a fraction of the figure height. Spacing is derived from
    # the actual figure size rather than a constant, so a wrapped banner on a
    # short figure cannot overlap the filter line below it.
    line = lambda pt: pt * 1.35 / (fig_h * 72.0)               # noqa: E731
    shift = line(10.5) * str(title).count("\n")
    fig.suptitle(title, y=0.982, fontsize=10.5, fontweight="bold")
    sub_wrap = max(50, int((fig_w - 0.6) * 72.0 / (7.0 * 0.52)))
    sub = textwrap.fill(str(subtitle), sub_wrap)
    n_sub = sub.count("\n") + 1
    y_sub = 0.952 - shift
    fig.text(0.5, y_sub, sub, ha="center", va="top", fontsize=7.0, color="#444444",
             linespacing=1.35)
    filt_wrap = max(50, int((fig_w - 0.6) * 72.0 / (6.0 * 0.52)))
    filt_text = textwrap.fill(filt, min(wrap, filt_wrap))
    y_filt = y_sub - n_sub * line(7.0) - 0.004
    fig.text(0.5, y_filt, filt_text, ha="center", va="top", fontsize=6.0,
             color="#444444", linespacing=1.35)
    # How far the whole header block extends below where a one-line header would.
    used = (0.952 - y_filt) + (filt_text.count("\n") + 1) * line(6.0)
    return max(used - 0.075, 0.0)


def save(fig, out_dir: Path, stem: str, manifest: List[dict], description: str,
         source: str) -> Dict[str, str]:
    from .provenance import guard_output
    out_dir = guard_output(out_dir)
    guard_output(out_dir / f"{stem}.png")
    guard_output(out_dir / f"{stem}.svg")
    # Axes drawn by replicate-simple register a display-only scale setter.
    setters = [ax._replicate_simple_axis for ax in fig.axes
               if hasattr(ax, "_replicate_simple_axis")]
    variants = ("full", "focus") if setters else (None,)
    paths = {v: (guard_output(out_dir / f"{stem}{'_' + v if v else ''}.png"),
                 guard_output(out_dir / f"{stem}{'_' + v if v else ''}.svg")) for v in variants}
    out_dir.mkdir(parents=True, exist_ok=True)
    import re
    texts = [(t, t.get_text()) for t in fig.texts]
    if setters and not any("axis:" in text for _, text in texts):
        label = fig.text(.016, .008, "axis: focus window", fontsize=6, color="#333333")
        texts.append((label, label.get_text()))
    for variant in variants:
        for setter in setters:
            setter(variant)
        for artist, original in texts:
            artist.set_text(re.sub(r"(axis:\s*)focus(\s*)window",
                                   lambda m: m[1] + variant + m[2]
                                   + ("window" if variant == "focus" else "scale"), original)
                            if variant else original)
        png, svg = paths[variant]
        fig.savefig(png, dpi=DPI)
        fig.savefig(svg)
    plt.close(fig)
    png, svg = paths[variants[-1]]
    rec = {"figure": stem, "png": png.name, "svg": svg.name,
           "description": description, "source": source}
    if setters:
        rec.update(full_png=paths["full"][0].name, full_svg=paths["full"][1].name)
    manifest.append(rec)
    return rec


# ------------------------------------------------------------------- context


def validate_plot_options(plot_style: str, technical_layer: str) -> None:
    if plot_style not in ("superplot", "replicate-simple"):
        raise ValueError("plot_style must be superplot or replicate-simple")
    if technical_layer not in ("none", "fov"):
        raise ValueError("technical_layer must be none or fov")


class FigureContext:
    """Everything every figure footer must carry, assembled once per report."""

    def __init__(self, run_dir: Path, cfg: dict, thresholds: Optional[pd.DataFrame],
                 group_order: Sequence[str], reference: str, alpha: float,
                 excluded_fields: Dict[str, str], labels: Dict[str, str],
                 color_overrides: Optional[Dict[str, str]] = None,
                 nucleus_filter: str = "all",
                 n_nuclei_all: Optional[int] = None,
                 n_nuclei_after_nucleus_filter: Optional[int] = None,
                 plot_style: str = "superplot", technical_layer: str = "none"):
        validate_plot_options(plot_style, technical_layer)
        self.plot_style = plot_style
        self.technical_layer = technical_layer
        self.run_dir = Path(run_dir)
        self.run_name = self.run_dir.name
        self.run_path = str(self.run_dir)
        self.cfg = cfg
        self.group_order = list(group_order)
        self.reference = reference
        self.alpha = alpha
        self.excluded_fields = dict(excluded_fields)
        self.nucleus_filter = str(nucleus_filter or "all")
        self.n_nuclei_all = n_nuclei_all
        self.n_nuclei_after_nucleus_filter = n_nuclei_after_nucleus_filter
        self.channel_labels = dict(labels)
        self.colors = group_colors(self.group_order, color_overrides)
        self.thresholds = self._threshold_text(thresholds)
        self.segmentation = self._segmentation_text(cfg, self.run_dir)
        self.banner = (f"Run {self.run_name} | {self.thresholds} | {self.segmentation} | "
                       "replicate unit: the well | test: Welch t on well means")
        self.filt = self._filter_text()

    @staticmethod
    def _threshold_text(thr: Optional[pd.DataFrame]) -> str:
        if thr is None or not len(thr):
            return "detection thresholds: not recorded by this run"
        parts = []
        for col, name in (("rna_bigfish_log_threshold", "anchor"),
                          ("protein_bigfish_log_threshold", "partner")):
            if col in thr.columns:
                vals = sorted(pd.to_numeric(thr[col], errors="coerce").dropna().unique().tolist())
                if len(vals) == 1:
                    parts.append(f"{name} detection threshold {vals[0]:g} (harmonized)")
                elif vals:
                    parts.append(f"{name} detection threshold varies per image, "
                                 f"{min(vals):g} to {max(vals):g}")
        return "; ".join(parts) if parts else "detection thresholds: not recorded"

    @staticmethod
    def _segmentation_text(cfg: dict, run_dir: Path) -> str:
        """Segmentation backend, model and version, read off the run itself.

        The backend version is written to the run's ``versions.txt`` rather than
        into ``run_config.json``, so both files are read and neither is assumed.
        """
        backend = model = ""
        try:
            backend = str(cfg.get("SEGMENTATION_BACKEND")
                          or (cfg.get("config_resolved") or {}).get("nuclei", {}).get(
                              "backend") or "")
            nuc = (cfg.get("config_resolved") or {}).get("nuclei") or {}
            model = str(nuc.get(f"{backend}_model_type") or nuc.get(f"{backend}_model")
                        or nuc.get("cellpose_model_type") or "")
        except Exception:                                      # noqa: BLE001
            pass
        ver = ""
        vpath = Path(run_dir) / "versions.txt"
        if vpath.is_file():
            for line in vpath.read_text(encoding="utf-8", errors="replace").splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    if k.strip().lower() == (backend or "cellpose").lower():
                        ver = v.strip()
                        break
        if not backend:
            return "segmentation backend not recorded by this run"
        out = f"segmentation {backend}"
        if model:
            out += f" {model}"
        out += f" {ver}" if ver else " (version not recorded)"
        return out

    def _nucleus_filter_text(self) -> str:
        """The nucleus filter ACTUALLY applied, not an assumed one.

        Until 2026-09-05 this asserted "every segmented nucleus, no post-hoc
        nucleus filter" unconditionally, so every report built with
        ``nucleus_filter: sampled`` carried a false statement on every figure.
        """
        if self.nucleus_filter == "sampled":
            samp = (self.cfg or {}).get("sampling") or {}
            n = samp.get("n_per_unit")
            unit = {"per_image": "image", "per_well": "well"}.get(
                str(samp.get("unit") or "per_image"), str(samp.get("unit")))
            head = (f"the {n} nuclei per {unit} the run sampled"
                    if n else "the nuclei the run sampled")
            counts = ""
            if self.n_nuclei_after_nucleus_filter is not None and self.n_nuclei_all:
                counts = (f", {self.n_nuclei_after_nucleus_filter} of "
                          f"{self.n_nuclei_all} segmented")
            return f"post-hoc nucleus filter: {head}{counts}"
        if self.nucleus_filter in ("all", ""):
            return "every segmented nucleus, no post-hoc nucleus filter"
        return f"post-hoc nucleus filter: {self.nucleus_filter}"

    def _filter_text(self) -> str:
        ex = (f"; excluded fields: "
              + "; ".join(f"{k} ({v})" for k, v in sorted(self.excluded_fields.items()))
              if self.excluded_fields else "; no field excluded")
        return (f"Filter: {self._nucleus_filter_text()}, except "
                f"where an endpoint states the engine usability flag it applies"
                f"{ex}. Gate: Welch t on well means at alpha {self.alpha:g}; "
                f"reference group {self.reference}.")

    def footer(self, extra: str = "") -> str:
        base = (f"Produced by fishsuite report from run {self.run_path}. "
                f"{self.thresholds}. {self.segmentation}. Replicate unit: the well "
                f"(fields of view are technical replicates; nuclei are the measurement "
                f"unit). Test: Welch t on well means.")
        note = getattr(self, 'localization_note', '')
        if note:
            base += '\n' + note
        return (extra + "\n" + base) if extra else base


# ----------------------------------------------------------------- SuperPlot


def draw_superplot(ax, ctx: FigureContext, endpoint: str, well: pd.DataFrame,
                   field: pd.DataFrame, per_nucleus: pd.DataFrame,
                   contrasts: pd.DataFrame, ylabel: str, nuc_column: Optional[str],
                   scale: float = 1.0, clip_pct: Optional[float] = None,
                   compact: bool = False, hline_at: Optional[float] = None,
                   hline_label: str = "") -> str:
    """One SuperPlot into ``ax``; returns the footnote text.

    Dots are nuclei shaded BY WELL within their condition group and drawn over
    the means. Open circles are field means. Diamonds are well means, the tested
    points. The heavy line is the mean of the well means.
    """
    rng = np.random.default_rng(SEED)
    order = ctx.group_order
    if fraction_scale(endpoint) == 100:
        scale = 100.
        ylabel = ('BIN1 intron puncta: % nuclear (per nucleus)' if endpoint == 'rna1_nuclear_spot_fraction'
                  else ylabel.replace('fraction', 'percent') + (' (%)' if '%' not in ylabel and 'percentage points' not in ylabel else ''))
    pw = well[well["endpoint"] == endpoint] if len(well) else well
    pf = field[field["endpoint"] == endpoint] if len(field) else field
    crow = contrasts[contrasts["endpoint"] == endpoint] if len(contrasts) else contrasts

    ufilter = ""
    if len(crow):
        ufilter = str(crow.iloc[0].get("usability_filter", "none"))
        if ufilter == "none":
            ufilter = ""
    have_nuc = (nuc_column is not None and len(per_nucleus)
                and nuc_column in per_nucleus.columns and not ufilter)

    pooled = (pd.to_numeric(per_nucleus[nuc_column], errors="coerce").dropna().to_numpy() * scale
              if have_nuc else np.array([]))
    clip_hi, n_above = None, 0
    if clip_pct is not None and len(pooled):
        clip_hi = float(np.percentile(pooled, clip_pct))
        n_above = int((pooled > clip_hi).sum())
        if clip_hi <= 0:
            clip_hi = None

    seen: List[float] = []
    lows: List[float] = []
    n_data = 0
    n_nuc_by_group: Dict[str, int] = {}
    n_well_by_group: Dict[str, int] = {}
    for xi, group in enumerate(order):
        col = ctx.colors.get(group, OKABE_ITO["blue"])
        if have_nuc:
            sub = per_nucleus[per_nucleus["group"] == group].copy()
            sub["_v"] = pd.to_numeric(sub[nuc_column], errors="coerce") * scale
            keep = sub[np.isfinite(sub["_v"])]
            if clip_hi is not None:
                keep = keep[keep["_v"] <= clip_hi]
            v = keep["_v"].to_numpy()
            n_nuc_by_group[group] = int(np.isfinite(
                pd.to_numeric(sub[nuc_column], errors="coerce")).sum())
            if len(v):
                q1, q3 = np.percentile(v, [25, 75])
                ax.add_patch(Rectangle((xi - 0.30, q1), 0.60, max(q3 - q1, 1e-9),
                                       facecolor=col, alpha=0.10, edgecolor="#999999",
                                       linewidth=0.6, zorder=1))
                seen.append(float(np.nanmax(v)))
                lows.append(float(np.nanmin(v)))
                n_data += int(len(v))
            # Shade the per-nucleus cloud BY WELL, so a well is visible as a
            # replicate inside its condition group rather than being pooled away.
            wells = sorted(keep["well_id"].dropna().astype(str).unique())
            for k, wid in enumerate(wells):
                fr = 0.20 + 0.65 * (k / max(len(wells) - 1, 1))
                vv = keep.loc[keep["well_id"].astype(str) == wid, "_v"].to_numpy()
                jx = xi + rng.uniform(-0.235, 0.235, size=len(vv))
                ax.scatter(jx, vv, s=7, facecolor=[shade(col, fr)], edgecolor="black",
                           linewidths=0.16, alpha=0.80, zorder=6)
        fm = (pd.to_numeric(pf[pf["group"] == group]["field_value"], errors="coerce")
              .to_numpy() * scale) if len(pf) else np.array([])
        fm = fm[np.isfinite(fm)]
        if len(fm):
            ax.scatter(xi + np.linspace(-0.17, 0.17, len(fm)), fm, s=28,
                       facecolor="white", edgecolor=col, linewidths=1.0, marker="o",
                       zorder=4)
            seen.append(float(np.nanmax(fm)))
            lows.append(float(np.nanmin(fm)))
            n_data += int(len(fm))
        wm = (pd.to_numeric(pw[pw["group"] == group]["well_mean_of_field_values"],
                            errors="coerce").to_numpy() * scale) if len(pw) else np.array([])
        wm = wm[np.isfinite(wm)]
        n_well_by_group[group] = int(len(wm))
        if len(wm):
            ax.scatter(xi + np.linspace(-0.13, 0.13, len(wm)), wm, s=70, facecolor=col,
                       edgecolor="black", linewidths=0.9, marker="D", zorder=5)
            ax.hlines(float(np.mean(wm)), xi - 0.33, xi + 0.33, color=col,
                      linewidth=2.4, zorder=7)
            seen.append(float(np.nanmax(wm)))
            lows.append(float(np.nanmin(wm)))
            n_data += int(len(wm))

    if n_data and hline_at is not None:
        ax.axhline(hline_at, color="#666666", linestyle="--", linewidth=1.0, zorder=2)
        seen.append(float(hline_at))
        lows.append(float(hline_at))

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    for tick, group in zip(ax.get_xticklabels(), order):
        tick.set_color(ctx.colors.get(group, OKABE_ITO["blue"]))
    ax.set_ylabel(ylabel, fontsize=6.6 if compact else 8.5)
    ax.set_xlim(-0.62, len(order) - 0.38)

    if not n_data:
        ax.text(0.5, 0.5, "endpoint not emitted by this run\nall values not available",
                transform=ax.transAxes, ha="center", va="center", fontsize=8,
                color=OKABE_ITO["vermillion"])
        ax.set_yticks([])
        return ctx.footer(
            "Nothing is drawn because the source column is absent from this run; no "
            "placeholder value is substituted.")

    top, bot = max(seen), min(lows)
    if clip_hi is not None:
        top = max(clip_hi, top)
    span = (top - bot) if (top - bot) > 0 else (abs(top) if top else 1.0)
    ax.set_ylim(bot - 0.08 * span, top + 0.30 * span)
    if abs(scale - 100.0) < 1e-9:
        ax.set_ylim(min(bot - 0.08 * span, -2.0), min(top + 0.30 * span, 118.0))
        ax.set_yticks(np.arange(0, 101, 20))

    # Brackets: every non-reference group against the reference. The star is the
    # RAW Welch p; the Holm-adjusted value goes in the footnote.
    ref_x = order.index(ctx.reference) if ctx.reference in order else 0
    p_lines: List[str] = []
    level = 0
    for xi, group in enumerate(order):
        if group == ctx.reference:
            continue
        r = crow[crow["test_group"] == group] if len(crow) else crow
        if not len(r):
            continue
        r0 = r.iloc[0]
        p_raw = r0.get("p_welch", np.nan)
        p_holm = r0.get("p_welch_holm_within_family", np.nan)
        mde = r0.get("mde_hedges_g_at_family_alpha", np.nan)
        g = r0.get("hedges_g", np.nan)
        row_star = top + (0.13 + 0.13 * level) * span
        if fraction_scale(endpoint) == 100 and 'minus_shuffle' not in endpoint:
            row_star = min(row_star, 92 - 8 * level)
        ax.plot([ref_x, ref_x, xi, xi],
                [row_star - 0.035 * span, row_star, row_star, row_star - 0.035 * span],
                color="black", linewidth=0.8, zorder=8, clip_on=False)
        ax.text((ref_x + xi) / 2, row_star + 0.012 * span, stars(p_raw), ha="center",
                va="bottom", fontsize=7.0 if compact else 9.0, color="black",
                fontweight="bold", zorder=9)
        holm_txt = (f"Holm-adjusted p {fmt_p(p_holm)}"
                    if np.isfinite(pd.to_numeric(p_holm, errors="coerce"))
                    else f"no Holm adjustment ({r0.get('holm_exclusion_reason', '')})")
        mde_txt = (f"smallest Hedges g detectable at 80 percent power and the family "
                   f"alpha is {float(mde):.3g}"
                   if np.isfinite(pd.to_numeric(mde, errors="coerce"))
                   else "minimum detectable effect not defined at this number of wells")
        g_num = pd.to_numeric(g, errors="coerce")
        g_txt = (f", Hedges g {float(g_num):.3g}" if np.isfinite(g_num) else "")
        p_lines.append(f"{group} vs {ctx.reference}: raw Welch p {fmt_p(p_raw)} "
                       f"(the star), {holm_txt}{g_txt}, {mde_txt}.")
        ax.set_ylim(bot - 0.08 * span, row_star + 0.17 * span)
        level += 1

    if hline_at is not None and hline_label:
        ax.text(0.985, hline_at, hline_label, transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=5.6 if compact else 6.4, color="#666666")

    for xi, group in enumerate(order):
        nw = n_well_by_group.get(group, 0)
        lab = (f"{nw} wells, n={n_nuc_by_group.get(group, 0)} nuclei"
               if have_nuc else f"{nw} wells")
        ax.annotate(lab, xy=(xi, 0), xycoords=ax.get_xaxis_transform(),
                    xytext=(0, -14 if compact else -18), textcoords="offset points",
                    ha="center", va="top", fontsize=5.0 if compact else 5.8,
                    color="#555555", annotation_clip=False)
    if compact:
        ax.tick_params(labelsize=6.2)
        for t in ax.get_xticklabels():
            t.set_fontsize(6.6)

    marker_line = (
        "Dots are nuclei, shaded by WELL within their condition group; open circles are "
        "field means; diamonds are WELL means, the tested points; the heavy line is the "
        "mean of the well means; the light box is the per-nucleus interquartile range."
        if have_nuc else
        "Open circles are field means; diamonds are WELL means, the tested points; the "
        "heavy line is the mean of the well means.")
    star_line = ("Stars come from the RAW Welch p on well means: **** below 1e-4, "
                 "*** below 0.001, ** below 0.01, * below 0.05, ns otherwise.")
    n_line = "n per group: " + "; ".join(
        f"{g} {n_well_by_group.get(g, 0)} wells"
        + (f", {n_nuc_by_group.get(g, 0)} nuclei" if have_nuc else "")
        for g in order) + "."
    foot = marker_line + "\n" + star_line + "\n" + " ".join(p_lines) + "\n" + n_line
    if ufilter:
        foot += ("\nThis endpoint is restricted to nuclei the engine flagged usable for "
                 f"its null ({ufilter}), so the per-nucleus cloud is not drawn: it would "
                 "show unfiltered nuclei behind well means computed on the filtered set.")
    if clip_hi is not None and n_above:
        foot += (f"\nThe y axis is clipped at the {clip_pct:g}th percentile of pooled "
                 f"per-nucleus values ({clip_hi:.4g}); {n_above} nuclei plot above the "
                 "clip and are not drawn. No value is removed from any statistic: every "
                 "test runs on well means.")
    foot += f"\nSource: Per nucleus, Per field, Per well and Contrasts sheets; endpoint {endpoint}."
    if fraction_scale(endpoint) == 100 and 'minus_shuffle' not in endpoint:
        ax.set_ylim(0, 100)
    if endpoint == 'rna1_nuclear_spot_fraction':
        return ("Two-sided Welch on well means; replicate unit: well.\n"
                + " ".join(p_lines) + " " + n_line + f"\nRun: {ctx.run_path}")
    return ctx.footer(foot)


def fraction_scale(endpoint):
    from .endpoints import ENDPOINTS, a3_endpoints
    return 100. if any(e.name == endpoint and 'fraction' in e.unit
                       for e in (*ENDPOINTS, *a3_endpoints([endpoint]))) else 1.


def draw_replicate_simple(ax, ctx: FigureContext, endpoint: str, well: pd.DataFrame,
                          field: pd.DataFrame, per_nucleus: pd.DataFrame,
                          contrasts: pd.DataFrame, ylabel: str, nuc_column: Optional[str],
                          scale: float = 1.0, clip_pct: Optional[float] = None,
                          compact: bool = False, hline_at: Optional[float] = None,
                          hline_label: str = "") -> str:
    """Draw persisted well means without recomputing statistics or using nuclei.

    ``clip_pct`` is deliberately unused: a nucleus percentile must never hide a
    biological replicate. FOVs are optional technical context, never tested points.
    The signature matches draw_superplot for standalone, composite and native callers.
    """
    if fraction_scale(endpoint) == 100:
        scale = 100.
        ylabel = ('BIN1 intron puncta: % nuclear (per nucleus)' if endpoint == 'rna1_nuclear_spot_fraction'
                  else ylabel.replace('fraction', 'percent') + (' (%)' if '%' not in ylabel and 'percentage points' not in ylabel else ''))
    pw = well[well["endpoint"] == endpoint] if len(well) else well
    pf = field[field["endpoint"] == endpoint] if len(field) else field
    rows = contrasts[contrasts["endpoint"] == endpoint] if len(contrasts) else contrasts
    seen, counts, well_values, mean_bars = [], [], [], []
    for xi, group in enumerate(ctx.group_order):
        col = ctx.colors[group]
        wells = pw[pw["group"] == group].sort_values("well_id") if len(pw) else pw
        fields = pf[pf["group"] == group] if len(pf) else pf
        wm = (pd.to_numeric(wells["well_mean_of_field_values"], errors="coerce")
              .dropna().to_numpy() * scale) if len(wells) else np.array([])
        wm = wm[np.isfinite(wm)]
        fm = (pd.to_numeric(fields["field_value"], errors="coerce").dropna()
              .to_numpy() * scale) if len(fields) else np.array([])
        fm = fm[np.isfinite(fm)]
        if ctx.technical_layer == "fov" and len(fm):
            art = ax.scatter(xi + np.linspace(-.22, .22, len(fm)), fm, s=12,
                             facecolor=shade(col, .18), edgecolor=shade(col, .35),
                             linewidths=.5, zorder=2)
            art.set_gid(f"fov:{group}")
            seen.extend(fm)
        if len(wm):
            offsets = np.linspace(-.14, .14, len(wm)) if len(wm) > 1 else np.zeros(1)
            art = ax.scatter(xi + offsets, wm, s=42 if compact else 60,
                             facecolor=(*shade(col, 1), .45), edgecolor=col,
                             linewidths=1.1, zorder=5)
            art.set_gid(f"well:{group}")
            if len(wm) >= 2:
                mean, sd = float(wm.mean()), float(wm.std(ddof=1))
                bar = ax.bar(xi, mean, width=.6, facecolor=(*shade(col, 1), .35),
                             edgecolor=col, linewidth=1, zorder=1)[0]
                bar.set_gid(f"group-mean:{group}")
                mean_bars.append((bar, mean))
                ax.errorbar(xi, mean, yerr=sd, fmt="none", ecolor=col,
                            elinewidth=1, capsize=3, capthick=1, zorder=3,
                            label=f"group-sd:{group}")
                seen.extend([mean - sd, mean + sd])
            seen.extend(wm)
            well_values.extend(wm)
        nn = (int(pd.to_numeric(fields["n_nuclei_nonmissing"], errors="coerce").sum())
              if len(fields) and "n_nuclei_nonmissing" in fields else None)
        level = str(fields.iloc[0].get("level", "nucleus")) if len(fields) else ""
        counts.append(f"{group}: {len(wm)} wells, {len(fm)} FOVs"
                      + (f", {nn} defined nuclei" if nn is not None and level == "nucleus" else ""))
    ax.set_xticks(range(len(ctx.group_order)), ctx.group_order)
    for tick, group in zip(ax.get_xticklabels(), ctx.group_order):
        tick.set_color(ctx.colors[group])
    ax.set_xlim(-.6, len(ctx.group_order) - .4)
    ax.set_ylabel(ylabel, fontsize=6.6 if compact else 8.5)
    ax.tick_params(labelsize=6.2 if compact else 8)
    marker = ("Points are WELL means, the tested replicates; "
              "bar = mean of well means, error bar = ± SD. "
              + ("Muted small points are technical FOV means." if ctx.technical_layer == "fov"
                 else "No technical layer is drawn."))
    if not seen:
        ax.text(.5, .5, "No finite well means available", transform=ax.transAxes,
                ha="center", va="center")
        ax._replicate_simple_axis = lambda variant: None
        return ctx.footer(marker + "\nNo values are substituted.; axis: focus window")
    if hline_at is not None:
        ax.axhline(hline_at, color="#666666", linestyle="--", linewidth=.8)
        seen.append(hline_at)
        if hline_label:
            ax.text(.98, hline_at, hline_label, transform=ax.get_yaxis_transform(),
                    ha="right", va="bottom", fontsize=5.6 if compact else 6.4)
    low, high = min(seen), max(seen)
    from .endpoints import ENDPOINTS, a3_endpoints
    unit = next((e.unit for e in (*ENDPOINTS, *a3_endpoints([endpoint]))
                 if e.name == endpoint), "")
    percent = (fraction_scale(endpoint) == 100 or '%' in unit) and 'minus_shuffle' not in endpoint
    zoom = percent and bool(well_values) and min(well_values) > 50
    brackets = []

    def set_axis(variant):
        focused = variant == "focus"
        if percent:
            bottom = 50. if focused and zoom else min(0., low)
            base_top = 100.
            step = 10. if bottom == 50 else 20.
        else:
            # Round outward, including constant and all-zero data.
            width = max(high - low, abs(high) * .1, 1e-6)
            step = 10. ** np.floor(np.log10(width)) / 2.
            bottom = (np.floor(low / step) - 1) * step if focused else min(0., low)
            if low >= 0:
                bottom = max(0., bottom)
            base_top = (np.floor(high / step) + 1) * step
        reserve = .25 + .14 * max(len(brackets) - 1, 0)
        span = max(base_top - bottom, (high - bottom) / max(.1, 1. - reserve)) if brackets else base_top - bottom
        top = bottom + span
        if not percent:
            top = np.ceil(top / step) * step
            span = top - bottom
        for level, (line, text) in enumerate(brackets):
            y = high + (.13 + .14 * level) * span
            line.set_ydata([y, y])
            text.set_y(y + .02 * span)
        ax.set_ylim(bottom, top)
        for bar, mean in mean_bars:
            bar.set_y(bottom)
            bar.set_height(mean - bottom)
        if percent:
            ax.set_yticks(np.arange(bottom, 101, step))

    ax._replicate_simple_axis = set_axis
    ref_x = ctx.group_order.index(ctx.reference)
    details, level = [], 0
    for xi, group in enumerate(ctx.group_order):
        if group == ctx.reference or not len(rows):
            continue
        match = rows[rows["test_group"] == group]
        if "reference_group" in match:
            match = match[match["reference_group"] == ctx.reference]
        if not len(match):
            continue
        r = match.iloc[0]
        p = r.get("p_welch", np.nan)
        from .endpoints import ENDPOINTS, a3_endpoints
        definition = next((e for e in (*ENDPOINTS, *a3_endpoints([endpoint]))
                           if e.name == endpoint), None)
        descriptive = definition and (definition.descriptive_only or definition.absolute_intensity)
        if descriptive:
            label = 'descriptive, no test'
            annotation = ax.text(.5, .97, label, transform=ax.transAxes,
                                 ha='center', va='top', fontsize=6.2 if compact else 7)
            annotation.set_gid('test-status:' + endpoint)
            continue
        line, = ax.plot([ref_x, xi], [high, high], color="black", linewidth=.8)
        text = ax.text((ref_x + xi) / 2, high, f"{stars(p)}  p={fmt_p(p)}",
                       ha="center", va="bottom", fontsize=6.2 if compact else 7)
        brackets.append((line, text))
        def number(key):
            value = pd.to_numeric(r.get(key, np.nan), errors="coerce")
            return f"{value:.3g}" if np.isfinite(value) else "NA"
        details.append(f"{group} vs {ctx.reference}: raw Welch p {fmt_p(p)}; "
                       f"Holm p {fmt_p(r.get('p_welch_holm_within_family', np.nan))}; "
                       f"Hedges g {number('hedges_g')}; MDE g (80% power, alpha .05) "
                       f"{number('mde_hedges_g_alpha_0p05')}, family alpha "
                       f"{number('mde_hedges_g_at_family_alpha')}; "
                       f"usability filter: {r.get('usability_filter', 'none')}.")
        level += 1
    set_axis("focus")
    return ("Two-sided Welch on well means; replicate unit: well; raw-p stars. "
            "bar = mean of well means, error bar = ± SD.\n"
            + " ".join(details) + " n: " + "; ".join(counts) + ".\n"
            + f"Run: {ctx.run_path}; axis: focus window")


def draw_plot(ax, ctx: FigureContext, *args, **kwargs) -> str:
    """Shared endpoint dispatcher; both styles use the same palette and save path."""
    draw = draw_replicate_simple if ctx.plot_style == "replicate-simple" else draw_superplot
    return draw(ax, ctx, *args, **kwargs)


def layout_replicate_simple(fig, ax, ctx, title, foot):
    """Fit the compact standalone bands using rendered text dimensions."""
    import re
    renderer = fig.canvas.get_renderer()
    heading = fig.text(.5, .98, " ".join(title.split()), ha="center", va="top",
                       fontsize=10.5, fontweight="bold")
    while heading.get_window_extent(renderer).width > fig.bbox.width * .96 and heading.get_fontsize() > 9:
        heading.set_fontsize(max(9, heading.get_fontsize() - .25))
    # Wrap only after the single line has exhausted the permitted font range.
    if heading.get_window_extent(renderer).width > fig.bbox.width * .96:
        words, lines, line = title.split(), [], ""
        for word in words:
            candidate = (line + " " + word).strip()
            heading.set_text(candidate)
            if line and heading.get_window_extent(renderer).width > fig.bbox.width * .96:
                lines.append(line)
                line = word
            else:
                line = candidate
        heading.set_text("\n".join([*lines, line]))
    thresholds = ctx.thresholds.replace(" detection threshold", "").replace(" (harmonized)", "")
    thresholds = thresholds.replace("detection thresholds: not recorded by this run", "thresholds NA")
    run = ctx.run_name if len(ctx.run_name) <= 23 else "…" + ctx.run_name[-22:]
    # Full run provenance remains in the report; the printed header is two lines.
    header = (f"Run {run} | {thresholds}\n"
              f"Filter: {ctx.nucleus_filter}; {len(ctx.excluded_fields)} fields excluded")
    head = fig.text(.5, .875, header, ha="center", va="top", fontsize=7, linespacing=1.05)
    compact_foot = foot.split("\nRun:")[0]
    compact_foot = compact_foot.replace("Two-sided Welch on well means; replicate unit: well; raw-p stars.",
        "Two-sided Welch; well replicates; raw-p stars.")
    compact_foot = compact_foot.replace("MDE g (80% power, alpha .05)", "MDE g (80%, α .05)")
    compact_foot = compact_foot.replace("family alpha", "family α")
    compact_foot = compact_foot.replace("defined nuclei", "nuclei")
    if "axis: focus window" in foot:
        compact_foot += " axis: focus window"
    renderer = fig.canvas.get_renderer()
    # Wrap by measured glyph width, rather than a minimum character count that
    # was intended for the much wider superplot canvas.
    probe = fig.text(0, 0, "", fontsize=6)
    def wrap(text, max_pixels):
        lines = []
        for paragraph in text.splitlines():
            line = ""
            for word in paragraph.split():
                candidate = (line + " " + word).strip()
                probe.set_text(candidate)
                if line and probe.get_window_extent(renderer).width > max_pixels:
                    lines.append(line)
                    line = word
                else:
                    line = candidate
            lines.append(line)
        return "\n".join(lines)
    footer = fig.text(.035, .018, wrap(compact_foot, fig.bbox.width * .93),
                      fontsize=6, va="bottom", linespacing=1.05, color="#333333")
    probe.remove()
    # Keep exactly two header lines at 7 pt, compressing horizontally only for
    # unusually long run/threshold metadata.
    head.set_text(header)
    fig.canvas.draw()
    if head.get_window_extent().width > fig.bbox.width * .96:
        head.set_text(header.replace(f"Run {run} | ", ""))
    fig.canvas.draw()
    bottom = footer.get_window_extent().y1 / fig.bbox.height + .125  # clear gap above the footer for the x tick labels
    top = head.get_window_extent().y0 / fig.bbox.height - .035
    ax.set_position([.22, bottom, .74, top - bottom])
    ax.yaxis.label.set_size(7)


def superplot_standalone(ctx: FigureContext, endpoint: str, title: str, ylabel: str,
                         well: pd.DataFrame, field: pd.DataFrame,
                         per_nucleus: pd.DataFrame, contrasts: pd.DataFrame,
                         nuc_column: Optional[str], out_dir: Path, stem: str,
                         manifest: List[dict], scale: float = 1.0,
                         clip_pct: Optional[float] = None,
                         hline_at: Optional[float] = None, hline_label: str = "",
                         w: float = 7.6, h: float = 6.6) -> Dict[str, str]:
    if endpoint == "rna1_nuclear_spot_fraction":
        title = "BIN1 intron puncta, % nuclear"
    simple = ctx.plot_style == "replicate-simple"
    if simple:
        w, h = 2.8 + .9 * max(0, len(ctx.group_order) - 2), 3.2
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0.115, 0.300, 0.790, 0.560])
    foot = draw_plot(ax, ctx, endpoint, well, field, per_nucleus, contrasts,
                          ylabel, nuc_column, scale=scale, clip_pct=clip_pct,
                          hline_at=hline_at, hline_label=hline_label)
    no_box(fig, ax)
    if simple:
        layout_replicate_simple(fig, ax, ctx, title, foot)
        return save(fig, out_dir, stem, manifest,
                    f"{ctx.plot_style} of {endpoint} by condition group, wells as replicates.",
                    "Per nucleus / Per field / Per well / Contrasts sheets")
    shift = stamp_head(fig, title, ctx.banner, ctx.filt)
    if shift:
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0, pos.width, max(pos.height - shift, 0.20)])
    band = foot_band(foot, w, h, 5.6)
    pos = ax.get_position()
    if band > pos.y0:
        ax.set_position([pos.x0, band, pos.width, max(pos.y1 - band, 0.20)])
    stamp_foot(fig, foot)
    return save(fig, out_dir, stem, manifest,
                f"{ctx.plot_style} of {endpoint} by condition group, wells as replicates.",
                "Per nucleus / Per field / Per well / Contrasts sheets")


# --------------------------------------------------------------- micrographs


def validate_territory_mask(cell, nuclear, roster, assigned_spots):
    """Validate persisted full-frame territory labels without constructing any."""
    from .aggregate import ReportInputError
    if cell.shape != nuclear.shape:
        raise ReportInputError('Recorded territory mask and nuclear dimensions differ')
    if not np.isin(cell, [0, *roster.nucleus_id.tolist()]).all():
        raise ReportInputError('Recorded territory mask has labels outside nucleus roster')
    if 'cell_area_px' not in roster or roster.cell_area_px.isna().any():
        raise ReportInputError('Missing cell_area_px for recorded territory mask validation')
    if any(
            int((cell == r.nucleus_id).sum()) != r.cell_area_px for r in roster.itertuples()):
        raise ReportInputError('Recorded territory mask does not reconcile to cell_area_px')
    inside = nuclear > 0
    if not np.array_equal(cell[inside], nuclear[inside]):
        raise ReportInputError('Recorded territory mask does not contain same-ID nuclei')
    if {'x_px', 'y_px'} <= set(assigned_spots) and len(assigned_spots):
        xy = assigned_spots[['x_px', 'y_px']].apply(pd.to_numeric, errors='coerce').to_numpy(float)
        if not np.isfinite(xy).all():
            raise ReportInputError('Missing assigned-spot coordinates for territory validation')
        xy = np.rint(xy).astype(int)
        if ((xy < 0).any() or (xy[:, 0] >= cell.shape[1]).any() or (xy[:, 1] >= cell.shape[0]).any()
                or not np.array_equal(cell[xy[:, 1], xy[:, 0]], assigned_spots.nucleus_id)):
            raise ReportInputError('Assigned spots do not reconcile to recorded territory mask')


def render_localization(ctx: FigureContext, well: pd.DataFrame, field: pd.DataFrame,
                        nuclei: pd.DataFrame, spots: pd.DataFrame,
                        contrasts: pd.DataFrame, out_dir: Path,
                        pub_dir: Optional[Path] = None,
                        territory_masks: Optional[Dict[str, Path]] = None) -> dict:
    """Reusable localization proposal renderer, using existing report tables.

    Pass the complete source nucleus/spot roster and ungated well/FOV tables.
    Audits never overwrite measurements. Percent is display-only. Optional
    territory_masks must be persisted full-frame cell labels keyed by image;
    absent geometry is labelled missing and is NEVER regenerated here.
    This API is separate from the historical FIG_MAIN and preserves its layout.
    """
    from .aggregate import reconcile_localization, ReportInputError
    from copy import copy
    from .provenance import guard_output
    out_dir = guard_output(out_dir)
    for name in ('per_nucleus', 'unassigned', 'checks', 'territory'):
        guard_output(out_dir / f'localization_{name}.csv')
    for name in ('localization_crops.json', 'localization_manifest.json'):
        guard_output(out_dir / name)
    for stem in ('rna1_nuclear_spot_fraction_localization', 'rna1_nuclear_spots_per_nucleus_localization',
                 'rna1_cyto_spots_per_nucleus_localization', 'FIG_LOCALIZATION'):
        for suffix in ('.png', '.svg'):
            guard_output(out_dir / (stem + suffix))
    audit = reconcile_localization(nuclei, spots)
    if audit['status'] != 'ok':
        raise ReportInputError('Localization rendering blocked: ' + '; '.join(audit['errors']))
    if ctx.nucleus_filter not in ('all', '') or ctx.excluded_fields:
        raise ReportInputError('Localization proposal requires all retained nuclei and no excluded fields')
    specs = [dict(endpoint='rna1_nuclear_spot_fraction', column='nuclear_spot_fraction',
                  label='BIN1 intron puncta: % nuclear (per nucleus)', scale=100.),
             dict(endpoint='rna1_nuclear_spots_per_nucleus', column='nuclear_spot_count',
                  label='Nuclear puncta count', scale=1.),
             dict(endpoint='rna1_cyto_spots_per_nucleus', column='cyto_spot_count',
                  label='Assigned-cytoplasmic puncta count', scale=1.)]
    if any(s['endpoint'] not in set(well.endpoint) for s in specs):
        raise ReportInputError('Localization rendering needs all three registered endpoints')
    bio = nuclei.loc[~nuclei.secondary_only.astype(bool)] if 'secondary_only' in nuclei else nuclei
    for spec in specs:
        pf = field[field.endpoint == spec['endpoint']]
        expected = bio.groupby('image')[spec['column']].mean().sort_index()
        actual = pf.set_index('image').field_value.sort_index()
        if (not expected.index.equals(actual.index) or not np.allclose(expected, actual, atol=1e-12, rtol=1e-10, equal_nan=True)):
            raise ReportInputError('Localization field values differ from full-roster source means')
        expected_w = pf.groupby(['group', 'well_id']).field_value.mean().sort_index()
        actual_w = well[well.endpoint == spec['endpoint']].set_index(['group', 'well_id']).well_mean_of_field_values.sort_index()
        if (not expected_w.index.equals(actual_w.index) or not np.allclose(expected_w, actual_w, atol=1e-12, rtol=1e-10, equal_nan=True)):
            raise ReportInputError('Localization well values differ from equal-weight FOV means')
    # This view always uses the A1 main well points. No measurement or inferential
    # value is recalculated for a change in plot style or percent display units.
    view = copy(ctx)
    view.plot_style = 'replicate-simple'
    view.localization_note = ('Per nucleus and assigned cell territory. Nuclear fraction displayed ×100; stored as 0–1. '
        'Detection Holm and family-alpha MDE include the added cytoplasmic test; amended, not baseline parity.')
    view.filt = (f'Filter: all retained nuclei; zero counts retained; zero-total fraction NA; '
                 f'no added peak/nucleus gates. Test: two-sided well-mean Welch; alpha {ctx.alpha:g}.')
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for key in ['per_nucleus', 'unassigned', 'checks', 'territory']:
        audit[key].to_csv(guard_output(out_dir / f'localization_{key}.csv'), index=False)
    manifests = []
    rna_name = ctx.channel_labels.get('rna1', 'RNA1')
    for spec in specs:
        superplot_standalone(view, spec['endpoint'], f"{rna_name}: {spec['label']}",
            spec['label'] + ('\nper nucleus and assigned cell territory' if spec['scale'] == 1 else ''),
            well, field, nuclei, contrasts, spec['column'], out_dir,
            spec['endpoint'] + '_localization', manifests, scale=spec['scale'])
    pub_dir = pub_dir or publication_image_dir(ctx.run_dir)
    luts, lut_source = read_luts(ctx.run_dir, pub_dir)
    panels = []
    records = []
    for group in view.group_order:
        f = pick_field(field, group, 'rna1_nuclear_spot_fraction')
        n = pick_nucleus(nuclei, f['image'], 'nuclear_spot_fraction') if f else None
        record = dict(group=group, selection='FOV nearest group median nuclear fraction; nucleus nearest FOV median',
                      crop_status='missing', territory_boundary='missing')
        p = None
        if f and n and pub_dir:
            row = nuclei[(nuclei.image == f['image']) & (nuclei.nucleus_id == n['nucleus_id'])].iloc[0]
            vox = pd.to_numeric(row.get('voxel_xy_um', np.nan), errors='coerce')
            p = crop_for(ctx.run_dir, pub_dir, f['image'], n['nucleus_id'], vox)
            record.update(image=f['image'], well_id=f['well_id'], nucleus_id=n['nucleus_id'])
            for col in ['cyto_estimation_method', 'cyto_area_px', 'cell_area_px', 'nucleus_area_px', 'voxel_xy_um']:
                value = row.get(col)
                record[col] = None if pd.isna(value) else (value.item() if isinstance(value, np.generic) else value)
            if p is not None:
                record.update(crop_status='available', png=str(p['png']), nuclear_mask_path=p['nuclear_mask_path'],
                              x0=p['x0'], y0=p['y0'], width=p['crop'].shape[1], height=p['crop'].shape[0])
                mask_path = (territory_masks or {}).get(f['image'])
                if mask_path is not None and Path(mask_path).is_file():
                    import tifffile
                    cell = tifffile.imread(str(mask_path))
                    if cell.shape != p['image_shape']:
                        raise ReportInputError('Recorded territory mask and image dimensions differ')
                    roster = nuclei[nuclei.image == f['image']]
                    assigned = spots[(spots.channel == 'rna1') & (spots.image == f['image']) & (spots.nucleus_id > 0)]
                    validate_territory_mask(cell, tifffile.imread(p['nuclear_mask_path']), roster, assigned)
                    p['territory_mask'] = cell[p['y0']:p['y0'] + CROP_PX, p['x0']:p['x0'] + CROP_PX] == n['nucleus_id']
                    record.update(territory_boundary='recorded', territory_mask_path=str(mask_path))
                selected = spots[(spots.channel == 'rna1') & (spots.image == f['image']) & (spots.nucleus_id == n['nucleus_id'])].drop_duplicates(['image', 'channel', 'spot_id'])
                p.update(group=group, record=record, assigned_spots=selected)
        panels.append(p)
        records.append(record)
    (out_dir / 'localization_crops.json').write_text(json.dumps(records, indent=2, allow_nan=False), encoding='utf-8')
    canvas = plt.figure(figsize=(11.4, 10.0))
    canvas.text(.06, .975, f'{rna_name} localization and assigned counts', fontsize=14, va='top')
    canvas.text(.06, .935, 'Per nucleus and assigned cell territory; nuclei → equal-weight FOVs → wells', fontsize=9)
    canvas.text(.06, .905, view.filt, fontsize=6.4)
    canvas.text(.06, .875, '\n'.join(textwrap.wrap(view.banner, width=175)), fontsize=5.4, va='top')
    foots = []
    for k, spec in enumerate(specs):
        ax = canvas.add_axes([.08 + k * .32, .61, .235, .23])
        foot = draw_plot(ax, view, spec['endpoint'], well, field, nuclei, contrasts,
                         spec['label'], spec['column'], scale=spec['scale'], compact=True)
        foots.append(foot.splitlines()[1])
        if spec['endpoint'] == 'rna1_nuclear_spot_fraction':
            ax.set_title('BIN1 intron puncta, % nuclear', fontsize=7)
        canvas.text(.04 + k * .32, .855, chr(65 + k), fontsize=11, fontweight='bold')
    for k, (p, record) in enumerate(zip(panels, records)):
        ax = canvas.add_axes([.09 + k * .86 / len(panels), .245, .76 / len(panels), .26])
        ax.set_axis_off()
        ax.set_title(record['group'], color=view.colors[record['group']], fontsize=9)
        if p is None:
            ax.text(.5, .5, 'Documented crop missing', transform=ax.transAxes, ha='center')
            continue
        ax.imshow(p['crop'], interpolation='nearest')
        ax.contour(p['nuclear_mask'].astype(float), levels=[.5], colors=['#00FFFF'], linewidths=.7)
        if 'territory_mask' in p:
            ax.contour(p['territory_mask'].astype(float), levels=[.5], colors=['white'], linewidths=.8)
        ss = p['assigned_spots']
        if {'x_px', 'y_px'} <= set(ss):
            cyto = pd.to_numeric(ss.in_cytoplasm, errors='coerce').eq(1)
            ax.scatter(ss.loc[cyto, 'x_px'] - p['x0'], ss.loc[cyto, 'y_px'] - p['y0'],
                       facecolors='none', edgecolors='white', s=18, linewidths=.7)
        ax.set_xlim(0, p['crop'].shape[1] - 1)
        ax.set_ylim(p['crop'].shape[0] - 1, 0)
        _bar(ax, p['crop'].shape[0], p['um_per_px'])
        ax.text(.5, -.035, f"{record['well_id']} | nucleus {record['nucleus_id']} | {record.get('cyto_estimation_method', 'missing')}\n"
                f"Territory boundary: {record['territory_boundary']}", transform=ax.transAxes, ha='center', va='top', fontsize=6)
    canvas.text(.06, .535, 'D   Documented crops: cyan = retained nuclear outline; white rings = assigned cytoplasmic RNA1 spots', fontsize=8)
    canvas.text(.06, .57, 'cytoplasmic = outside the 2D nuclear mask within the assigned territory; single plane', fontsize=8)
    canvas.text(.06, .19, 'Recorded areas describe assigned territory, not membrane-bounded cells. Missing boundaries are not reconstructed.\n'
                'Unassigned spots are excluded from nucleus denominators and tabulated separately. Crop coordinates and geometry: localization_crops.json.', fontsize=6.5)
    for k, (name, color) in enumerate(luts):
        canvas.text(.06 + k * .27, .16, '■', color=color, fontsize=9)
        canvas.text(.076 + k * .27, .16, name, fontsize=7)
    footer = ('Two-sided Welch on well means; replicate unit: well.\n'
              + '\n'.join(foots) + f'\nRun: {ctx.run_path}')
    stamp_foot(canvas, footer, size=5.4, y=.025)
    save(canvas, out_dir, 'FIG_LOCALIZATION', manifests,
         'Localization proposal: percent nuclear and nuclear/cytoplasmic counts, equal-weight FOV to well; recorded crops, missing territory boundaries explicit.',
         'Source nuclei_metrics.csv / spot_metrics.csv; report Per field / Per well / Contrasts; localization_crops.json')
    (out_dir / 'localization_manifest.json').write_text(json.dumps(dict(
        figures=manifests, source_run=str(ctx.run_dir), channel_key_source=lut_source,
        duplicate_rows=audit['duplicate_rows'],
        audit_status=audit['status'], family_amendment='detection: added rna1_cyto_spots_per_nucleus; amended Holm, not parity'), indent=2), encoding='utf-8')
    return dict(audit=audit, figures=manifests, crops=records)


def publication_image_dir(run_dir: Path) -> Optional[Path]:
    """The run's publication images, preferring a post-hoc re-render if one exists."""
    run_dir = Path(run_dir)
    rer = sorted(run_dir.glob("publication_images*_rerender_*"))
    if rer:
        return rer[-1]
    plain = run_dir / "publication_images"
    return plain if plain.is_dir() else None


LUT_HEX = {"yellow": "#FFFF00", "magenta": "#FF00FF", "green": "#00FF00",
           "blue": "#4169FF", "cyan": "#00FFFF", "red": "#FF0000",
           "grey": "#BBBBBB", "gray": "#BBBBBB", "white": "#FFFFFF"}


# Channel slot -> the run_config key pair for that slot, per analysis mode. The
# engine writes rna2_lut for an rna_rna run and antibody_lut for rna_protein; both
# keys are always PRESENT, so reading a fixed slot list hands an rna_rna panel the
# antibody LUT. That produced a green exon channel on a magenta run (2026-09-04).
MODE_SLOTS = {
    "rna_rna": (("rna_label", "rna_lut"), ("rna2_label", "rna2_lut"),
                ("dapi_label", "dapi_lut")),
    "rna_protein": (("rna_label", "rna_lut"), ("antibody_label", "antibody_lut"),
                    ("dapi_label", "dapi_lut")),
    "rna_only": (("rna_label", "rna_lut"), ("dapi_label", "dapi_lut")),
    "protein_only": (("antibody_label", "antibody_lut"), ("dapi_label", "dapi_lut")),
    "ab_ab": (("antibody_label", "antibody_lut"), ("ab2_label", "ab2_lut"),
              ("dapi_label", "dapi_lut")),
}
RENDER_SLOTS = {
    "rna_rna": ("rna", "rna2", "dapi"),
    "rna_protein": ("rna", "ab", "dapi"),
    "rna_only": ("rna", "dapi"),
}


def read_luts(run_dir: Path, pub_dir: Optional[Path]):
    """Channel names and LUT colours, read from a file and keyed on the run's mode.

    Returns ``(entries, source)``. A run whose mode cannot be read yields an empty
    list and a source string saying so, rather than a guessed default: a wrong LUT
    key on a micrograph mislabels which channel a reader is looking at.
    """
    def hexof(name):
        return LUT_HEX.get(str(name).lower(), "#BBBBBB")

    cfg_path = Path(run_dir) / "run_config.json"
    mode = ""
    ch: Dict[str, object] = {}
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        ch = (cfg.get("config_resolved") or {}).get("channels") or {}
        mode = str(ch.get("analysis_mode") or cfg.get("ANALYSIS_MODE") or "")
    except Exception:                                          # noqa: BLE001
        pass

    if pub_dir is not None:
        rp = Path(pub_dir) / "render_params.json"
        if rp.is_file():
            try:
                r = json.loads(rp.read_text(encoding="utf-8"))
                lbl, lut = r.get("labels_used") or {}, r.get("luts_used") or {}
                rmode = str(r.get("analysis_mode") or mode)
                slots = RENDER_SLOTS.get(rmode)
                if lbl and lut and slots:
                    ent = [(str(lbl[s]), hexof(lut[s])) for s in slots
                           if lut.get(s) and lbl.get(s)]
                    if ent:
                        return ent, str(rp)
            except Exception:                                  # noqa: BLE001
                pass

    slots = MODE_SLOTS.get(mode)
    if not slots:
        return [], (f"analysis mode {mode!r} in {cfg_path} has no channel slot map, "
                    "so no LUT key is drawn rather than guessing one")
    ent = [(str(ch.get(lk, lk)), hexof(ch.get(ck, ""))) for lk, ck in slots
           if ch.get(ck)]
    if not ent:
        return [], f"no LUT was recorded for mode {mode!r} in {cfg_path}"
    return ent, str(cfg_path)


def merge_png_for(pub_dir: Path, image: str):
    """The all-channel merge PNG for one image, plus the per-image output stem."""
    core = str(image)
    if "." in os.path.basename(core):
        core = os.path.splitext(core)[0]
    cands = []
    for f in glob.glob(os.path.join(str(pub_dir), "*__merge_*.png")):
        base = os.path.basename(f)
        parts = base.split("__")
        if len(parts) >= 3 and parts[1] and core.endswith(parts[1]):
            merge = parts[2][:-4]
            cands.append((0 if merge == "merge_all" else 1, -len(parts[1]),
                          -merge.count("_"), f, "__".join(parts[:2]) + "__"))
    if not cands:
        return None
    cands.sort()
    return cands[0][3], cands[0][4]


def _bar(ax, npx, um_per_px, um=5.0, color="white"):
    if not um_per_px or not np.isfinite(um_per_px):
        return
    px = um / um_per_px
    y = npx * 0.93
    ax.plot([npx * 0.06, npx * 0.06 + px], [y, y], color=color, linewidth=2.4,
            solid_capstyle="butt")
    ax.text(npx * 0.06 + px / 2, y - npx * 0.035, f"{um:g} micrometres", color=color,
            ha="center", va="bottom", fontsize=6.0)


def pick_field(field: pd.DataFrame, group: str, endpoint: str) -> Optional[dict]:
    pf = field[(field["endpoint"] == endpoint) & (field["group"] == group)]
    pf = pf[np.isfinite(pd.to_numeric(pf["field_value"], errors="coerce"))]
    if not len(pf):
        return None
    med = float(pf["field_value"].median())
    r = pf.iloc[(pf["field_value"] - med).abs().argsort().iloc[0]]
    return {"image": str(r["image"]), "well_id": str(r["well_id"]),
            "field_value": float(r["field_value"])}


def pick_nucleus(per_nucleus: pd.DataFrame, image: str, column: str) -> Optional[dict]:
    sub = per_nucleus[per_nucleus["image"] == image].copy()
    if column not in sub.columns:
        return None
    sub["_v"] = pd.to_numeric(sub[column], errors="coerce")
    sub = sub[np.isfinite(sub["_v"])]
    if not len(sub):
        return None
    med = float(sub["_v"].median())
    r = sub.iloc[(sub["_v"] - med).abs().argsort().iloc[0]]
    return {"nucleus_id": int(r["nucleus_id"]), "value": float(r["_v"]),
            "n_candidates": int(len(sub))}


def crop_for(run_dir: Path, pub_dir: Path, image: str, nucleus_id: int,
             um_per_px: float) -> Optional[dict]:
    try:
        import tifffile
    except Exception:                                          # noqa: BLE001
        return None
    hit = merge_png_for(pub_dir, image)
    if hit is None:
        return None
    png, stem = hit
    mask = Path(run_dir) / "masks" / f"{stem}nuclei_label_mask.tif"
    if not (os.path.exists(png) and mask.is_file()):
        return None
    lab = tifffile.imread(str(mask))
    img = mpimg.imread(png)
    H, W = img.shape[:2]
    if lab.shape[:2] != (H, W):
        return None
    ys, xs = np.where(lab == nucleus_id)
    if not len(ys):
        return None
    cy, cx = int(round(ys.mean())), int(round(xs.mean()))
    half = CROP_PX // 2
    y0 = int(np.clip(cy - half, 0, max(H - CROP_PX, 0)))
    x0 = int(np.clip(cx - half, 0, max(W - CROP_PX, 0)))
    return {"crop": img[y0:y0 + CROP_PX, x0:x0 + CROP_PX], "png": png,
            "um_per_px": um_per_px, "image": image, "nucleus_id": nucleus_id,
            "x0": x0, "y0": y0, "nuclear_mask_path": str(mask),
            "nuclear_mask": lab[y0:y0 + CROP_PX, x0:x0 + CROP_PX] == nucleus_id,
            "image_shape": (H, W)}


def collect_panels(ctx: FigureContext, field: pd.DataFrame, per_nucleus: pd.DataFrame,
                   endpoint: str, nuc_column: Optional[str], run_dir: Path,
                   pub_dir: Optional[Path], um_per_px: float) -> List[dict]:
    if pub_dir is None:
        return []
    panels = []
    for group in ctx.group_order:
        f = pick_field(field, group, endpoint)
        if f is None:
            continue
        n = pick_nucleus(per_nucleus, f["image"], nuc_column) if nuc_column else None
        if n is None:
            continue
        c = crop_for(run_dir, pub_dir, f["image"], n["nucleus_id"], um_per_px)
        if c is None:
            continue
        c.update(group=group, well_id=f["well_id"], field_value=f["field_value"],
                 nucleus_value=n["value"], n_candidates=n["n_candidates"])
        panels.append(c)
    return panels


def draw_crop_row(fig, axes, panels, luts, bar_um=5.0):
    for ax, p in zip(axes, panels):
        ax.imshow(p["crop"], interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{p['group']}  well {p['well_id']}", fontsize=7.5,
                     color="#333333", pad=3)
        _bar(ax, p["crop"].shape[0], p["um_per_px"], um=bar_um)


def micrograph_standalone(ctx: FigureContext, panels: List[dict], luts,
                          lut_source: str, out_dir: Path, stem: str,
                          manifest: List[dict], endpoint: str) -> Optional[Dict[str, str]]:
    if not panels:
        return None
    n = len(panels)
    w, h = max(3.2 * n, 5.0), 5.4
    fig = plt.figure(figsize=(w, h))
    # Draw the header first: it returns how far the wrapped banner and filter
    # line push the body down, and the panels are placed under that.
    shift = stamp_head(fig, "Representative fields, one nucleus per condition group",
                       ctx.banner, ctx.filt)
    top = 0.80 - shift
    axes = [fig.add_axes([0.03 + i * (0.94 / n), 0.30, 0.94 / n - 0.02,
                          max(top - 0.30, 0.20)]) for i in range(n)]
    draw_crop_row(fig, axes, panels, luts)
    if luts:
        for k, (name, col) in enumerate(luts):
            fig.text(0.03 + k * 0.16, 0.245, "■", color=col, fontsize=9.4,
                     ha="left", va="center")
            fig.text(0.03 + k * 0.16 + 0.018, 0.245, name, color="#333333",
                     fontsize=6.4, ha="left", va="center")
    sel = "; ".join(
        f"{p['group']} well {p['well_id']}, image {p['image']}, nucleus "
        f"{p['nucleus_id']} of {p['n_candidates']} candidates" for p in panels)
    foot = ctx.footer(
        "Each panel is the nucleus closest to the median of its own field for "
        f"{endpoint}, inside the field closest to that group's median. Selection is "
        "deterministic, not chosen by eye, and no image is contrast-matched to another "
        "beyond the run's own shared display window.\n"
        f"Panels: {sel}.\nChannel names and colours read from {lut_source}.\n"
        f"Images read from {panels[0]['png']}.")
    band = foot_band(foot, w, h, 5.6)
    if band > 0.30:
        for i, ax in enumerate(axes):
            ax.set_position([0.03 + i * (0.94 / n), band, 0.94 / n - 0.02,
                             max(top - band, 0.20)])
        for t in fig.texts:
            if t.get_fontsize() == 6.4 or t.get_text() == "■":
                x, y = t.get_position()
                if abs(y - 0.245) < 1e-9:
                    t.set_position((x, band - 0.055))
    stamp_foot(fig, foot)
    return save(fig, out_dir, stem, manifest,
                "Representative micrograph crops, one per condition group.",
                "publication images plus the run's nuclei label masks")


# ----------------------------------------------------------------- composite


def composite_main(ctx: FigureContext, specs: Sequence[dict], panels: List[dict],
                   luts, well: pd.DataFrame, field: pd.DataFrame,
                   per_nucleus: pd.DataFrame, contrasts: pd.DataFrame,
                   out_dir: Path, manifest: List[dict],
                   stem: str = "FIG_MAIN") -> Dict[str, str]:
    """One composite: the primary endpoints as SuperPlots plus the micrograph row."""
    n_sp = len(specs)
    ncol = max(min(n_sp, 3), 1)
    nrow = int(np.ceil(n_sp / ncol))
    has_img = bool(panels)
    w = 3.5 * ncol + 0.9
    h = 3.1 * nrow + (3.0 if has_img else 0) + 2.4
    fig = plt.figure(figsize=(w, h))
    head_shift = stamp_head(fig, "Condition-group comparison, wells as biological "
                                 "replicates", ctx.banner, ctx.filt)
    top = 0.885 - head_shift
    body_h = 3.1 * nrow / h
    img_h = (3.0 / h) if has_img else 0.0
    axes = []
    for k, spec in enumerate(specs):
        r, c = divmod(k, ncol)
        ax = fig.add_axes([0.085 + c * (0.90 / ncol),
                           top - body_h * (r + 1) / nrow + 0.055,
                           0.90 / ncol - 0.085,
                           body_h / nrow - 0.115])
        axes.append((ax, spec))
    foots = []
    for k, (ax, spec) in enumerate(axes):
        f = draw_plot(ax, ctx, spec["endpoint"], well, field, per_nucleus,
                           contrasts, spec["ylabel"], spec.get("nuc_column"),
                           compact=True, scale=spec.get("scale", 1.0),
                           hline_at=spec.get("hline_at"),
                           hline_label=spec.get("hline_label", ""))
        no_box(fig, ax)
        # Panel letters live in FIGURE coordinates, clear of the axis label. In
        # axes coordinates a two-line y-axis label collides with the letter.
        pos = ax.get_position()
        fig.text(max(pos.x0 - 0.055, 0.004), min(pos.y1 + 0.012, 0.995),
                 chr(ord("A") + k), fontsize=10, fontweight="bold", va="bottom",
                 ha="left")
        foots.append(f.splitlines()[1]
                     if ctx.plot_style == "replicate-simple" and len(f.splitlines()) > 1
                     else (f.splitlines()[2] if len(f.splitlines()) > 2 else ""))
    if has_img:
        n = len(panels)
        y0 = top - body_h - img_h + 0.055
        iaxes = [fig.add_axes([0.085 + i * (0.86 / n), y0, 0.86 / n - 0.02,
                               img_h - 0.075]) for i in range(n)]
        draw_crop_row(fig, iaxes, panels, luts)
        ipos = iaxes[0].get_position()
        fig.text(max(ipos.x0 - 0.055, 0.004), min(ipos.y1 + 0.012, 0.995),
                 chr(ord("A") + n_sp), fontsize=10, fontweight="bold", va="bottom",
                 ha="left")
        if luts:
            for k, (name, col) in enumerate(luts):
                fig.text(0.085 + k * 0.14, y0 - 0.020, "■", color=col,
                         fontsize=9.4, ha="left", va="center")
                fig.text(0.085 + k * 0.14 + 0.016, y0 - 0.020, name, color="#333333",
                         fontsize=6.4, ha="left", va="center")
    lettered = "; ".join(f"{chr(ord('A') + k)} {s['endpoint']}"
                         for k, s in enumerate(specs))
    foot = ctx.footer(
        f"Panels: {lettered}"
        + (f"; {chr(ord('A') + n_sp)} representative micrographs." if has_img else ".")
        + "\nDots are nuclei shaded by WELL within their condition group; open circles "
          "are field means; diamonds are WELL means, the tested points; the heavy line "
          "is the mean of the well means.\nStars come from the RAW Welch p on well "
          "means: **** below 1e-4, *** below 0.001, ** below 0.01, * below 0.05, ns "
          "otherwise. The Holm-adjusted p and the minimum detectable effect for every "
          "panel are on that panel's standalone figure and in the Contrasts sheet.\n"
        + "\n".join(x for x in foots if x))
    if ctx.plot_style == "replicate-simple":
        foot = ctx.footer(f"Panels: {lettered}.\n"
                          "Points are well means; bar = mean of well means, error bar = ± SD. "
                          + ("Muted small points are FOV means. " if ctx.technical_layer == "fov" else "")
                          + "Two-sided Welch on wells; raw p above each comparison.\n"
                          + "\n".join(foots))
    stamp_foot(fig, foot)
    return save(fig, out_dir, stem, manifest,
                f"Composite ({ctx.plot_style}): primary endpoints by condition group plus representative "
                "micrographs.",
                "Per nucleus / Per field / Per well / Contrasts sheets and the run's "
                "publication images")
