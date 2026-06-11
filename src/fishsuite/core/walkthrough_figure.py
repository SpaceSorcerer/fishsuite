"""walkthrough_figure — publication PIPELINE-WALKTHROUGH composite (Brian, 2026-06-07).

A single, clean, labeled 8-panel figure that "shows how the MIAT x QKI RNA-FISH
pipeline works" using the pipeline's OWN per-step microscope PNGs (already
LUT-applied with a 25 um scalebar burned in) for panels A-F and H, plus ONE newly
rendered panel (G): the run's thresholded QKI (magenta) intensity field with the
detected MIAT spots overlaid as small yellow open markers — showing MIAT foci
sitting within the QKI intensity field. NO statistics / coloc-result panel.

Panels (row-major, 2x4):
  A  Nuclei (DAPI)                         <run>/pipeline_walkthrough  step01_DAPI_raw
  B  Nucleus segmentation (Cellpose)       <run>/pipeline_walkthrough  step03_nuclei_outlines_on_DAPI
  C  MIAT RNA-FISH (640)                   <run>/publication_images    *_yellow (MIAT)
  D  MIAT spot detection (BigFISH LoG)     <run>/pipeline_walkthrough  step07_*_spots_on_DAPI
  E  QKI immunofluorescence (561)          <run>/publication_images    *_magenta (QKI)
  F  QKI intensity layer (thresholded)     <run>/pipeline_walkthrough  step06_QKI_*_threshold_on_signal
  G  MIAT spots on thresholded QKI         RENDERED from pixels (this module)
  H  Merge: MIAT + QKI                     <run>/publication_images    merge_<MIAT>_<QKI>  (fallback step11_merge_all)

Panel G is rendered by REUSING coloc_backfill's VSI machinery (the SAME PLAIN-VSI
resolver + the SAME io z-recompute primitives the backfill uses): the QKI 2D plane
is recomputed deterministically at the DAPI autofocus z (NEVER read from
spot_metrics, which stores z=0), thresholded at the run's ``protein_threshold_value``
(fallback per-image ``thresholds.csv`` -> ``manual_antibody_min``), shown in magenta
with the run's display ceiling, with the image's detected MIAT (rna1) spots overlaid.

Reusable for the d4/d8/d15 timepoints: panels are located by the ``image_key``
PREFIX + the known step suffixes, and any missing panel self-skips with a warning
(no crash).

CLI::

    python -m fishsuite.core.walkthrough_figure --run-dir <run> --staging <staging>
        [--image <key>] [--out <png>]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .output import sanitize_condition_for_filename
# Reuse coloc_backfill's PLAIN-VSI resolver (single source of truth for finding
# the non-decon VSI under the staging tree).
from .coloc_backfill import _resolve_plain_vsi

PIXEL_UM_DEFAULT = 0.13          # MIAT-QKI hESC 100x voxel (130 nm/px)
SCALEBAR_UM_DEFAULT = 25.0       # matches the burned scalebar on the other panels
DEFAULT_OUT_REL = ("figures", "07_coloc", "79_pipeline_walkthrough.png")


# ===========================================================================
# PURE new-panel logic (panel G) — unit-tested on synthetic arrays
# ===========================================================================
def threshold_qki_plane(qki_2d: np.ndarray, threshold: float) -> np.ndarray:
    """Return a float COPY of ``qki_2d`` with sub-threshold pixels set to 0.0 and
    supra-threshold pixels (>= ``threshold``) kept verbatim.

    Pure + deterministic; does NOT mutate the input. Mirrors the run's QKI
    intensity gate (diffuse QKI IF is NOT spot-detected — it is thresholded).
    """
    arr = np.asarray(qki_2d, dtype=np.float64).copy()
    arr[arr < float(threshold)] = 0.0
    return arr


def _magenta_cmap():
    """Black -> magenta LUT with masked pixels drawn black (matches the QKI LUT
    used by the publication renderer + coloc_backfill montage)."""
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("magenta_k", ["black", "magenta"]).copy()
    cmap.set_bad("black")
    return cmap


def _draw_scalebar(ax, shape, pixel_um: float, bar_um: float,
                   *, color: str = "white", frac_margin: float = 0.045) -> None:
    """Burn a ``bar_um`` scalebar (bottom-right) into ``ax`` in data coords, so
    panel G matches the burned-in scalebars on the composed PNG panels."""
    from matplotlib.patches import Rectangle
    h_img, w_img = shape[0], shape[1]
    bar_px = float(bar_um) / float(pixel_um)
    x1 = w_img * (1.0 - frac_margin)
    x0 = x1 - bar_px
    y = h_img * (1.0 - frac_margin)
    bar_h = max(2.0, h_img * 0.012)
    ax.add_patch(Rectangle((x0, y - bar_h), bar_px, bar_h, color=color, ec="none",
                           zorder=5))


def render_panel_g(
    ax,
    qki_2d: np.ndarray,
    threshold: float,
    spots_xy: np.ndarray,
    *,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    pixel_um: float = PIXEL_UM_DEFAULT,
    scalebar_um: float = SCALEBAR_UM_DEFAULT,
    marker_color: str = "yellow",
    marker_size: float = 11.0,
    marker_lw: float = 0.4,
    add_scalebar: bool = True,
) -> Dict[str, Any]:
    """Render panel G onto ``ax``: the thresholded QKI field (magenta) with the
    detected MIAT spots as small yellow open markers.

    Sub-threshold QKI pixels are MASKED (drawn black, i.e. dimmed out) so only the
    supra-threshold QKI intensity field shows; supra pixels stretch to ``vmax``
    (the run's display ceiling). ``spots_xy`` is an ``(n, 2)`` array of
    ``(x_px, y_px)`` MIAT-spot centres; markers are placed at EXACTLY those
    coordinates. Returns a dict with the AxesImage, the spot PathCollection, the
    masked display array, the threshold and ``n_spots`` (for testing/inspection).
    """
    qki = np.asarray(qki_2d, dtype=np.float64)
    thr = float(threshold)
    disp = np.ma.masked_less(qki, thr)            # sub-threshold -> masked (black)

    if vmin is None:
        vmin = thr
    if vmax is None:
        finite = qki[np.isfinite(qki)]
        vmax = float(np.percentile(finite, 99.5)) if finite.size else thr + 1.0
    if not (vmax > vmin):
        vmax = vmin + 1.0

    im = ax.imshow(disp, cmap=_magenta_cmap(), vmin=vmin, vmax=vmax,
                   interpolation="nearest")

    spots = np.asarray(spots_xy, dtype=float).reshape(-1, 2)
    sc = ax.scatter(
        spots[:, 0], spots[:, 1],
        s=marker_size, facecolors="none", edgecolors=marker_color,
        linewidths=marker_lw, zorder=4,
    )
    # keep the image framing (scatter can otherwise expand the limits)
    ax.set_xlim(-0.5, qki.shape[1] - 0.5)
    ax.set_ylim(qki.shape[0] - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    if add_scalebar and pixel_um and scalebar_um:
        _draw_scalebar(ax, qki.shape, pixel_um, scalebar_um)

    return {
        "image": im,
        "scatter": sc,
        "masked": disp,
        "threshold": thr,
        "n_spots": int(spots.shape[0]),
    }


# ===========================================================================
# Panel / image resolution
# ===========================================================================
def _resolve_image_key(run_dir, image_key: Optional[str] = None) -> str:
    """Resolve the panel-prefix ``image_key``. If given, pass through. Otherwise
    discover the per-image prefixes from ``pipeline_walkthrough/*__step01_*.png``
    and prefer the OE g2-Dox image (its prefix contains ``MIAT_OE``)."""
    if image_key:
        return image_key
    wk = Path(run_dir) / "pipeline_walkthrough"
    prefixes: List[str] = []
    for p in sorted(wk.glob("*__step01_*.png")):
        idx = p.name.find("__step01")
        if idx > 0:
            prefixes.append(p.name[:idx])
    if not prefixes:
        raise FileNotFoundError(
            f"no '*__step01_*.png' panels under {wk}; cannot infer image_key"
        )
    oe = [k for k in prefixes if "MIAT_OE" in k]
    return sorted(oe)[0] if oe else sorted(prefixes)[0]


def _match_image_row(per_image: pd.DataFrame, image_key: str):
    """Map a panel-prefix ``image_key`` (``<sanitized_condition>__<short_base>``)
    to its ``per_image_summary`` row (whose ``image`` is the full VSI name).

    Match = the sanitized condition is the image_key prefix AND the (space->_)
    image stem ENDS WITH the image_key's trailing ``<short_base>`` token (the run
    strips a common leading prefix from the base). Falls back to base-token-only.
    """
    base_token = image_key.rsplit("__", 1)[-1]
    for _, row in per_image.iterrows():
        img = str(row["image"])
        stem_us = Path(img).stem.replace(" ", "_")
        cond_san = sanitize_condition_for_filename(str(row.get("condition", "")))
        if cond_san and image_key.startswith(cond_san) and stem_us.endswith(base_token):
            return row
    for _, row in per_image.iterrows():
        stem_us = Path(str(row["image"])).stem.replace(" ", "_")
        if stem_us.endswith(base_token):
            return row
    return None


def _find_panel_file(run_dir, candidates: List[Dict[str, Any]]) -> Optional[Path]:
    """Return the first file matching any candidate spec, else None.

    Each candidate: ``{subdir, pattern, require=[], require_any=[], exclude=[]}``.
    ``require`` = every substring must be in the filename; ``require_any`` = at
    least one must be; ``exclude`` = none may be. Substrings are matched
    case-insensitively against the file name.
    """
    run_dir = Path(run_dir)
    for spec in candidates:
        d = run_dir / spec["subdir"]
        if not d.is_dir():
            continue
        require = [s.lower() for s in spec.get("require", []) if s]
        require_any = [s.lower() for s in spec.get("require_any", []) if s]
        exclude = [s.lower() for s in spec.get("exclude", []) if s]
        for p in sorted(d.glob(spec["pattern"])):
            name = p.name.lower()
            if require and not all(s in name for s in require):
                continue
            if require_any and not any(s in name for s in require_any):
                continue
            if exclude and any(s in name for s in exclude):
                continue
            return p
    return None


def _panel_specs(image_key: str, rna_tok: Optional[str], prot_tok: Optional[str]
                 ) -> List[Dict[str, Any]]:
    """Ordered A-H panel definitions: letter, title, and either ``image``
    candidate specs (compose an existing PNG) or ``render_g`` (render panel G)."""
    k = image_key
    prot_req = {"require": [prot_tok]} if prot_tok else {
        "require_any": ["qki", "561", "magenta"]}
    rna_excl = [rna_tok] if rna_tok else ["miat", "640"]
    merge_filter = (
        {"require": [rna_tok, prot_tok], "exclude": ["dapi"]}
        if (rna_tok and prot_tok) else
        {"require_any": ["qki", "561"], "exclude": ["dapi", "all"]}
    )
    return [
        {"letter": "A", "title": "Nuclei (DAPI)", "kind": "image",
         "candidates": [{"subdir": "pipeline_walkthrough",
                         "pattern": f"{k}__step01_*.png"}]},
        {"letter": "B", "title": "Nucleus segmentation (Cellpose)", "kind": "image",
         "candidates": [{"subdir": "pipeline_walkthrough",
                         "pattern": f"{k}__step03_*.png"}]},
        {"letter": "C", "title": "MIAT RNA-FISH (640)", "kind": "image",
         "candidates": [{"subdir": "publication_images",
                         "pattern": f"{k}__*_yellow.png", "exclude": ["merge"]}]},
        {"letter": "D", "title": "MIAT spot detection (BigFISH LoG)", "kind": "image",
         "candidates": [{"subdir": "pipeline_walkthrough",
                         "pattern": f"{k}__step07_*.png"}]},
        {"letter": "E", "title": "QKI immunofluorescence (561)", "kind": "image",
         "candidates": [{"subdir": "publication_images",
                         "pattern": f"{k}__*_magenta.png", "exclude": ["merge"]}]},
        {"letter": "F", "title": "QKI intensity layer (thresholded)", "kind": "image",
         "candidates": [{"subdir": "pipeline_walkthrough",
                         "pattern": f"{k}__step06_*threshold_on_signal*.png",
                         "exclude": rna_excl, **prot_req}]},
        {"letter": "G", "title": "MIAT spots on thresholded QKI", "kind": "render_g",
         "candidates": []},
        {"letter": "H", "title": "Merge: MIAT + QKI", "kind": "image",
         "candidates": [{"subdir": "publication_images",
                         "pattern": f"{k}__merge_*.png", **merge_filter},
                        {"subdir": "pipeline_walkthrough",
                         "pattern": f"{k}__step11_merge_all*.png"}]},
    ]


# ===========================================================================
# Panel G — recompute QKI plane (REUSE coloc_backfill's machinery)
# ===========================================================================
def _recompute_qki_plane(cfg, vsi_path):
    """Recompute the QKI 2D analysis plane for ``vsi_path`` — mirrors
    ``coloc_backfill.backfill_run``'s z-recompute EXACTLY, reusing the same io
    primitives (``read_image`` / ``extract_channel_autofocus_with_idx`` /
    ``extract_channel_at_z``) and the same rna_protein antibody->rna2 channel
    shim. NEVER reads z from spot_metrics (it is 0). Returns ``(qki_2d, pix_um)``.
    """
    from . import io as _io  # local import so tests can monkeypatch read_image
    from .modes.rna_rna import _resolve_channels

    img = _io.read_image(vsi_path)
    mode = getattr(cfg.channels, "analysis_mode", "")
    if mode == "rna_protein":
        from .modes.rna_protein import _build_rna2_shim_cfg
        chan_cfg = _build_rna2_shim_cfg(cfg)
    else:
        chan_cfg = cfg
    dapi_idx, _rna_idx, rna2_idx = _resolve_channels(chan_cfg, img)

    z_start = cfg.z_stack.start_slice
    z_end = cfg.z_stack.end_slice
    if z_start is not None and z_start > img.n_z:
        z_start = 1
    if z_end is not None and z_end > img.n_z:
        z_end = img.n_z
    iw = bool(getattr(cfg.z_stack, "autofocus_intensity_weighted", False))
    dapi_z, _dapi_2d = _io.extract_channel_autofocus_with_idx(
        img, dapi_idx, z_start=z_start, z_end=z_end, intensity_weighted=iw
    )
    qki_2d = _io.extract_channel_at_z(img, rna2_idx, z_1indexed=dapi_z)

    vx = float(img.voxel_xy_nm) if (img.voxel_xy_nm == img.voxel_xy_nm
                                    and img.voxel_xy_nm > 0) else 130.0
    return np.asarray(qki_2d, dtype=np.float64), vx / 1000.0


def _resolve_qki_threshold(row, run_dir, image_value, rc: dict) -> float:
    """QKI display floor: ``protein_threshold_value`` from the per_image row;
    else a per-image ``thresholds.csv`` lookup; else ``manual_antibody_min`` from
    run_config; else 1250.0."""
    if row is not None:
        v = row.get("protein_threshold_value")
        if v is not None and v == v:  # not NaN
            return float(v)
    try:
        th = pd.read_csv(Path(run_dir) / "thresholds.csv")
        m = th[th["image"] == image_value] if "image" in th.columns else th.iloc[0:0]
        for col in ("protein_threshold_value", "protein_thresh_floor"):
            if col in m.columns and len(m) and m.iloc[0][col] == m.iloc[0][col]:
                return float(m.iloc[0][col])
    except Exception:
        pass
    out = rc.get("config_resolved", {}).get("output", {}) if rc else {}
    v = out.get("manual_antibody_min")
    return float(v) if v is not None else 1250.0


def _load_miat_spots(run_dir, image_value) -> np.ndarray:
    """MIAT (rna1) spot ``(x_px, y_px)`` for ``image_value`` from spot_metrics.csv;
    empty ``(0, 2)`` array if unavailable."""
    try:
        sm = pd.read_csv(Path(run_dir) / "spot_metrics.csv")
    except Exception:
        return np.empty((0, 2), dtype=float)
    sel = sm[(sm["image"] == image_value) & (sm["channel"] == "rna1")]
    if not len(sel):
        return np.empty((0, 2), dtype=float)
    return sel[["x_px", "y_px"]].astype(float).to_numpy()


def _panel_g_inputs(run_dir, staging_dir, input_dir, image_key, cfg, rc
                    ) -> Optional[Dict[str, Any]]:
    """Gather panel-G inputs (qki plane, threshold, MIAT spots, ceiling, pix_um),
    AVAILABILITY-GUARDED: returns None (panel G self-skips) if cfg is missing, the
    image row cannot be matched, no source tree is given, the PLAIN VSI is not
    found, or the VSI read fails."""
    if cfg is None:
        return None
    try:
        per_image = pd.read_csv(Path(run_dir) / "per_image_summary.csv")
    except Exception:
        return None
    row = _match_image_row(per_image, image_key)
    if row is None:
        return None
    image_value = str(row["image"])
    threshold = _resolve_qki_threshold(row, run_dir, image_value, rc)

    src = staging_dir or input_dir or (rc.get("input_dir") if rc else None)
    if not src:
        return None
    vsi = _resolve_plain_vsi(Path(src), image_value)
    if vsi is None:
        return None
    try:
        qki_2d, pix_um = _recompute_qki_plane(cfg, vsi)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"panel G: QKI re-read failed for {image_value}: "
                      f"{type(exc).__name__}: {exc}", UserWarning)
        return None

    out = rc.get("config_resolved", {}).get("output", {}) if rc else {}
    ceil = out.get("manual_antibody_max")
    spots = _load_miat_spots(run_dir, image_value)
    return {
        "qki_2d": qki_2d,
        "threshold": float(threshold),
        "vmax": float(ceil) if ceil is not None else None,
        "spots_xy": spots,
        "pix_um": float(pix_um),
        "image_value": image_value,
        "condition": str(row.get("condition", "")),
    }


# ===========================================================================
# Composite figure
# ===========================================================================
def _load_panel_image(path: Path) -> np.ndarray:
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def _annotate_panel(ax, letter: str, title: str) -> None:
    import matplotlib.patheffects as pe
    ax.text(0.03, 0.97, letter, transform=ax.transAxes, fontsize=16,
            fontweight="bold", va="top", ha="left", color="white",
            path_effects=[pe.withStroke(linewidth=2.5, foreground="black")])
    ax.set_title(title, fontsize=9.5)


def _placeholder_panel(ax, title: str, msg: str) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("0.95")
    ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center", va="center",
            fontsize=8.5, color="0.4", style="italic")


def _build_walkthrough_fig(run_dir, staging_dir=None, input_dir=None,
                           image_key=None):
    """Build the 8-panel walkthrough figure. Returns ``(fig, statuses)`` where
    ``statuses`` is the ordered A-H list of ``{letter, title, status, path}``
    (status in ok/rendered/missing/skipped). Missing/skipped panels emit a
    UserWarning and render a placeholder (so the grid always has 8 axes)."""
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    image_key = _resolve_image_key(run_dir, image_key)

    # run_config (optional): drives the rna/protein filename tokens + panel G.
    rc: Optional[dict] = None
    cfg = None
    rc_path = run_dir / "run_config.json"
    if rc_path.exists():
        try:
            rc = json.loads(rc_path.read_text())
            from ..config.schema import FishsuiteConfig
            cfg = FishsuiteConfig.model_validate(rc["config_resolved"])
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"walkthrough: could not load run_config.json "
                          f"({type(exc).__name__}: {exc}); panel G + token "
                          f"disambiguation degraded.", UserWarning)

    rna_tok = prot_tok = None
    if cfg is not None:
        rna_tok = sanitize_condition_for_filename(
            getattr(cfg.channels, "rna_label", "") or "") or None
        prot_tok = sanitize_condition_for_filename(
            getattr(cfg.channels, "antibody_label", "") or "") or None

    specs = _panel_specs(image_key, rna_tok, prot_tok)

    # figure-level label
    label = image_key
    g_inputs = _panel_g_inputs(run_dir, staging_dir, input_dir, image_key, cfg, rc)
    if g_inputs and g_inputs.get("condition"):
        base = image_key.rsplit("__", 1)[-1]
        label = f"{g_inputs['condition']} - {base}"

    fig, axes = plt.subplots(2, 4, figsize=(16.0, 8.7))
    axes = axes.ravel()
    statuses: List[Dict[str, Any]] = []

    for ax, spec in zip(axes, specs):
        letter, title = spec["letter"], spec["title"]
        if spec["kind"] == "render_g":
            if g_inputs is None:
                _placeholder_panel(ax, title, f"{title}\n(panel G unavailable:\n"
                                              f"VSI / run_config not found)")
                _annotate_panel(ax, letter, title)
                warnings.warn(f"walkthrough panel G ({title}) skipped: "
                              f"VSI / run_config unavailable.", UserWarning)
                statuses.append({"letter": letter, "title": title,
                                 "status": "skipped", "path": None})
                continue
            render_panel_g(
                ax, g_inputs["qki_2d"], g_inputs["threshold"],
                g_inputs["spots_xy"], vmax=g_inputs["vmax"],
                pixel_um=g_inputs["pix_um"], scalebar_um=SCALEBAR_UM_DEFAULT,
            )
            _annotate_panel(ax, letter, title)
            statuses.append({"letter": letter, "title": title,
                             "status": "rendered", "path": None})
            continue

        found = _find_panel_file(run_dir, spec["candidates"])
        if found is None:
            pats = ", ".join(c["pattern"] for c in spec["candidates"])
            _placeholder_panel(ax, title, f"{title}\n(panel {letter} missing)")
            _annotate_panel(ax, letter, title)
            warnings.warn(f"walkthrough panel {letter} ({title}) missing: "
                          f"no file matched [{pats}] under {run_dir}.", UserWarning)
            statuses.append({"letter": letter, "title": title,
                             "status": "missing", "path": None})
            continue
        ax.imshow(_load_panel_image(found))
        ax.set_xticks([])
        ax.set_yticks([])
        _annotate_panel(ax, letter, title)
        statuses.append({"letter": letter, "title": title,
                         "status": "ok", "path": str(found)})

    fig.suptitle(f"MIAT×QKI RNA-FISH pipeline — {label}",
                 fontsize=15, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    return fig, statuses


def build_walkthrough_figure(run_dir, staging_dir=None, input_dir=None,
                             image_key=None, out_path=None) -> str:
    """Build + save the pipeline-walkthrough composite PNG; return its path.

    ``image_key`` defaults to the OE g2-Dox image if present (else the first
    image). ``out_path`` defaults to
    ``<run_dir>/figures/07_coloc/79_pipeline_walkthrough.png``. Panels are located
    by the image_key prefix; any missing panel self-skips with a warning.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    fig, _statuses = _build_walkthrough_fig(
        run_dir, staging_dir=staging_dir, input_dir=input_dir, image_key=image_key
    )
    out = Path(out_path) if out_path else run_dir.joinpath(*DEFAULT_OUT_REL)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return str(out)


# ===========================================================================
# CLI
# ===========================================================================
def main(argv: Optional[List[str]] = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser(
        prog="python -m fishsuite.core.walkthrough_figure",
        description="Build the publication MIATxQKI pipeline-walkthrough composite "
                    "figure (8 panels, including the rendered MIAT-spots-on-"
                    "thresholded-QKI panel G) from a run's own step images.",
    )
    ap.add_argument("--run-dir", required=True, help="run output dir")
    ap.add_argument("--staging", default=None,
                    help="staging tree holding the PLAIN VSIs (for panel G); "
                         "defaults to input_dir / run_config input_dir")
    ap.add_argument("--input", default=None, help="alt source dir for VSIs")
    ap.add_argument("--image", default=None,
                    help="panel-prefix image_key (default: OE g2-Dox if present)")
    ap.add_argument("--out", default=None, help="output PNG path override")
    args = ap.parse_args(argv)

    out = build_walkthrough_figure(
        args.run_dir, staging_dir=args.staging, input_dir=args.input,
        image_key=args.image, out_path=args.out,
    )
    print("written:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
