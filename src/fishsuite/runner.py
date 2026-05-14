"""Top-level batch runner — discovers inputs, runs the pipeline per image, writes outputs.

Produces a Fiji-pipeline-compatible output directory layout so downstream
tools (combine_to_xlsx.py, single_condition_plots.py, R scripts) can
consume fishsuite output transparently.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, BarColumn, TextColumn, TimeRemainingColumn

from . import __version__
from .config.schema import FishsuiteConfig
from .core import io as _io
from .core import output as _out
from .core.modes import get_mode
from .core.parallel import auto_n_workers


_console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_output_dirs(output_dir: Path, cfg: FishsuiteConfig) -> Dict[str, Path]:
    out = {
        "qc_overlays": output_dir / "qc_overlays",
        "per_image_csv": output_dir / "per_image_csv",
        "masks": output_dir / "masks",
        "publication_images": output_dir / "publication_images",
        "pipeline_walkthrough": output_dir / "pipeline_walkthrough",
        "nuclei_popouts": output_dir / "nuclei_popouts",
    }
    for p in out.values():
        p.mkdir(parents=True, exist_ok=True)
    return out


def _stem_with_condition(stem: str, condition: str | None) -> str:
    """Compose ``<stem>__<condition_sanitized>`` for per-image output files.

    When the condition is missing or sanitizes to empty (e.g. unlabeled or
    blank), the bare stem is returned so we don't emit ugly trailing
    double-underscores. Sanitization rules: see
    ``output.sanitize_condition_for_filename``.

    Examples:
        ("H9-MIAT-ASOs-_03", "NT ASO")    -> "H9-MIAT-ASOs-_03__NT_ASO"
        ("H9-MIAT-ASOs-_10", "Sec-Only")  -> "H9-MIAT-ASOs-_10__Sec_Only"
        ("H9-MIAT-ASOs-_03", None)        -> "H9-MIAT-ASOs-_03"
    """
    csan = _out.sanitize_condition_for_filename(condition)
    return f"{stem}__{csan}" if csan else stem


def _write_per_image_csvs(per_dir: Path, stem: str,
                          nuclei_df: pd.DataFrame, spots_df: pd.DataFrame,
                          prefix: str = "") -> None:
    per_dir.mkdir(parents=True, exist_ok=True)
    if len(nuclei_df) > 0:
        nuclei_df.to_csv(per_dir / f"{prefix}{stem}__nuclei_metrics.csv", index=False)
    if len(spots_df) > 0:
        spots_df.to_csv(per_dir / f"{prefix}{stem}__spot_metrics.csv", index=False)


def _write_per_image_thresholds(mask_dir: Path, stem: str, thresholds: dict,
                                prefix: str = "") -> None:
    pd.DataFrame([thresholds]).to_csv(
        mask_dir / f"{prefix}{stem}__thresholds.csv", index=False
    )


def _write_masks(mask_dir: Path, stem: str, *,
                 labels: np.ndarray, rna_pos_mask: np.ndarray,
                 dapi_mask: np.ndarray | None = None,
                 prefix: str = "") -> None:
    _out.save_label_tiff(labels, mask_dir / f"{prefix}{stem}__nuclei_label_mask.tif")
    if rna_pos_mask is not None:
        _out.save_mask_tiff(rna_pos_mask, mask_dir / f"{prefix}{stem}__spot_mask.tif")
    if dapi_mask is not None:
        _out.save_mask_tiff(dapi_mask, mask_dir / f"{prefix}{stem}__dapi_mask.tif")


# ---------------------------------------------------------------------------
# Main batch entry point
# ---------------------------------------------------------------------------

def run_batch(
    config_path: Path,
    input_dir: Path,
    output_dir: Path,
    *,
    parallel: str | int = "auto",
    resume: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
) -> dict:
    """End-to-end batch run.

    Returns a dict with summary stats.
    """
    cfg = FishsuiteConfig.from_yaml(config_path)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dirs = _make_output_dirs(output_dir, cfg)

    prefix = cfg.output.prefix or ""

    images = _io.discover_inputs(
        input_dir,
        subfolder_conditions=cfg.conditions.subfolder_conditions,
        sec_only_folders=cfg.conditions.sec_only_folders,
        sec_only_files=cfg.conditions.sec_only_files,
    )
    if not images:
        raise RuntimeError(f"No images discovered under {input_dir}")

    # Apply per-file selection subset (Improvement 2). When ``input_file_subset``
    # is non-empty, filter the discovered list to only those entries. We accept
    # bare basenames, posix-style relative paths (matching what the GUI tree
    # writes), and absolute paths — comparing each form against the discovered
    # file. Empty list = include everything (legacy back-compat).
    subset = list(getattr(cfg, "input_file_subset", None) or [])
    if subset:
        total = len(images)
        # Normalise the subset entries for matching: stripped, forward-slash.
        norm = set()
        names_only = set()
        for s in subset:
            s2 = str(s).replace("\\", "/").strip()
            if not s2:
                continue
            norm.add(s2)
            names_only.add(Path(s2).name)
        kept: list = []
        for im in images:
            ip = Path(im.path)
            full_norm = ip.as_posix()
            try:
                rel_norm = ip.relative_to(input_dir).as_posix()
            except Exception:
                rel_norm = ""
            if (
                im.path.name in names_only
                or full_norm in norm
                or (rel_norm and rel_norm in norm)
            ):
                kept.append(im)
        _console.print(
            f"[bold]fishsuite v{__version__}[/bold] — found {total} images, "
            f"input_file_subset filter -> kept [bold]{len(kept)}[/bold] of {total}"
        )
        images = kept
        if not images:
            raise RuntimeError(
                f"input_file_subset matched 0 of {total} discovered files. "
                f"Subset entries: {subset[:5]}{'...' if len(subset) > 5 else ''}"
            )
    else:
        _console.print(f"[bold]fishsuite v{__version__}[/bold] — found [bold]{len(images)}[/bold] images")
    for im in images:
        _console.print(f"  {im.path.name:40s}  condition={im.condition!r:18s}  sec_only={im.sec_only}")

    if dry_run:
        _console.print("[yellow]--dry-run set, exiting before processing.[/yellow]")
        return dict(n_images=len(images))

    if isinstance(parallel, str) and parallel.lower() == "auto":
        n_workers = auto_n_workers()
    else:
        n_workers = max(1, int(parallel))
    _console.print(f"Parallel workers: [bold]{n_workers}[/bold]")

    # Reset batch contrast cache so a prior run's running-max floor/ceil
    # cannot leak into this run's publication / QC renders. Mirrors Fiji's
    # reset_batch_disp_ceil_cache() called once at pipeline start in
    # Coloc_Pipeline.py.
    _out.reset_batch_disp_range_cache()

    # ---- Batch-scope pixel-coloc threshold pre-pass -----------------------
    # When pixel_coloc.threshold_scope == 'batch', Fiji's pipeline runs a
    # pre-scan over every image, pools all raw nuclear RNA pixel values, and
    # computes ONE median + k_mad*MAD value applied uniformly to every image
    # in the run. See Coloc_Analysis.run_batch_prescan_for_thresholds()
    # (lines 2336-2666) — uses convertToFloatProcessor() with NO rolling-ball,
    # NO blur, NO median filter. Reproduced here.
    #
    # rna_rna mode: pools nuclear pixels for BOTH RNA channels INDEPENDENTLY
    # so each channel gets its own batch threshold (its own nuclear-pixel
    # distribution). The mode's collect_nuclear_rna_pixels returns
    # (rna_pixels, rna2_pixels); we pool each separately.
    batch_rna_threshold: float | None = None
    batch_rna2_threshold: float | None = None
    pc_cfg = getattr(cfg, "pixel_coloc", None)
    is_rna_rna = (cfg.channels.analysis_mode == "rna_rna")
    if pc_cfg is not None and getattr(pc_cfg, "threshold_scope", "per_image") == "batch":
        try:
            from .core import thresholds as _thr
            if is_rna_rna:
                from .core.modes.rna_rna import (
                    collect_nuclear_rna_pixels as _collect_two,
                )
                collect_nuclear_rna_pixels = None  # type: ignore[assignment]
            else:
                from .core.modes.rna_only import collect_nuclear_rna_pixels
                _collect_two = None  # type: ignore[assignment]
        except Exception as e:
            _console.print(
                f"[yellow]Could not import batch-threshold helpers ({e}); "
                f"falling back to per-image thresholds.[/yellow]"
            )
            collect_nuclear_rna_pixels = None  # type: ignore[assignment]
            _collect_two = None  # type: ignore[assignment]

        if (collect_nuclear_rna_pixels is not None) or (_collect_two is not None):
            _console.print(
                "[bold]PRE-SCAN[/bold] pixel-coloc threshold_scope=batch: "
                "collecting nuclear RNA pixels across all images..."
            )
            pooled_list: List[np.ndarray] = []
            pooled2_list: List[np.ndarray] = []
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                console=_console,
            ) as ppg:
                ptask = ppg.add_task("Pre-pass: nuclear-pixel pooling", total=len(images))
                for dimg in images:
                    try:
                        if _collect_two is not None:
                            v1, v2 = _collect_two(dimg.path, cfg=cfg)
                            if v1.size > 0:
                                pooled_list.append(v1)
                            if v2.size > 0:
                                pooled2_list.append(v2)
                        else:
                            vals = collect_nuclear_rna_pixels(dimg.path, cfg=cfg)
                            if vals.size > 0:
                                pooled_list.append(vals)
                    except Exception as e:
                        _console.print(
                            f"[yellow]Pre-scan failed on {dimg.path.name}: {e} — "
                            f"image excluded from pool[/yellow]"
                        )
                    ppg.advance(ptask)
            if pooled_list:
                pooled = np.concatenate(pooled_list)
                try:
                    batch_rna_threshold = float(_thr.coloc_threshold(
                        pooled.tolist(),
                        mode=pc_cfg.threshold_mode,
                        k_mad=float(pc_cfg.k_mad),
                        percentile=float(pc_cfg.percentile),
                    ))
                except Exception as e:
                    _console.print(
                        f"[yellow]Batch threshold computation failed ({e}); "
                        f"falling back to per-image.[/yellow]"
                    )
                    batch_rna_threshold = None
                else:
                    _console.print(
                        f"  pooled n_pixels (rna) = {pooled.size:,}  "
                        f"-> batch rna_threshold_value = [bold]{batch_rna_threshold:.4f}[/bold] "
                        f"(mode={pc_cfg.threshold_mode}, k_mad={pc_cfg.k_mad})"
                    )
            else:
                _console.print(
                    "[yellow]Pre-scan collected zero nuclear pixels across the "
                    "batch; falling back to per-image thresholds.[/yellow]"
                )
            if is_rna_rna and pooled2_list:
                pooled2 = np.concatenate(pooled2_list)
                try:
                    batch_rna2_threshold = float(_thr.coloc_threshold(
                        pooled2.tolist(),
                        mode=pc_cfg.threshold_mode,
                        k_mad=float(pc_cfg.k_mad),
                        percentile=float(pc_cfg.percentile),
                    ))
                except Exception as e:
                    _console.print(
                        f"[yellow]Batch rna2 threshold computation failed ({e}); "
                        f"falling back to per-image.[/yellow]"
                    )
                    batch_rna2_threshold = None
                else:
                    _console.print(
                        f"  pooled n_pixels (rna2) = {pooled2.size:,}  "
                        f"-> batch rna2_threshold_value = [bold]{batch_rna2_threshold:.4f}[/bold] "
                        f"(mode={pc_cfg.threshold_mode}, k_mad={pc_cfg.k_mad})"
                    )

    mode_fn = get_mode(cfg.channels.analysis_mode)

    per_image_rows: List[dict] = []
    nuclei_dfs: List[pd.DataFrame] = []
    spots_dfs: List[pd.DataFrame] = []
    morph_dfs: List[pd.DataFrame] = []
    threshold_rows: List[dict] = []
    failures: List[tuple] = []
    t_start = time.time()

    # Reorder images: process every non-sec-only image FIRST, then sec-only
    # controls. This guarantees the batch contrast cache is populated by
    # real-signal images before any sec-only image is rendered, so the
    # "consult-but-don't-update" sec-only path in _resolve_lut_range always
    # has a valid (floor, ceil) to render with. Without this, a sec-only
    # image encountered before any real image (e.g. file order
    # alphanumerically interleaves them) would fall back to its own
    # autofluorescence percentiles and the user would still see falsely
    # bright RNA signal in the pub PNG. Stable-sort preserves discovery
    # order within each group.
    images = sorted(images, key=lambda im: (1 if im.sec_only else 0,))

    # ---- Process each image ------------------------------------------------
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=_console,
    ) as progress:
        task = progress.add_task("Processing images", total=len(images))
        for i, dimg in enumerate(images):
            progress.update(task, description=f"[{i+1}/{len(images)}] {dimg.path.name}")
            raw_stem = dimg.path.stem
            # Embed the (sanitized) condition into every per-image output
            # filename so Brian doesn't have to cross-reference an image
            # number against the conditions table to know what condition
            # _10 belongs to. Top-level master CSVs already carry the
            # 'condition' column and are NOT renamed.
            stem = _stem_with_condition(raw_stem, dimg.condition)
            try:
                _mode_kwargs = dict(
                    condition=dimg.condition,
                    sec_only=dimg.sec_only,
                    cfg=cfg,
                )
                # Only forward the batch threshold to modes whose run()
                # signature accepts it (currently rna_only + rna_rna). Other
                # modes delegate to rna_only internally; the kwarg will reach
                # run_one through that path only if needed in the future.
                if (
                    batch_rna_threshold is not None
                    and cfg.channels.analysis_mode in ("rna_only", "rna_rna")
                ):
                    _mode_kwargs["precomputed_rna_threshold"] = batch_rna_threshold
                if (
                    batch_rna2_threshold is not None
                    and cfg.channels.analysis_mode == "rna_rna"
                ):
                    _mode_kwargs["precomputed_rna2_threshold"] = batch_rna2_threshold
                res = mode_fn(
                    dimg.path,
                    **_mode_kwargs,
                )

                # Accumulate master tables
                per_image_rows.append(res.per_image)
                if len(res.nuclei):
                    nuclei_dfs.append(res.nuclei)
                if len(res.spots):
                    spots_dfs.append(res.spots)
                if len(getattr(res, "morphology", pd.DataFrame())):
                    morph_dfs.append(res.morphology)
                # Inject user-typed channel labels into the per-image
                # threshold record so the human name flows into both
                # thresholds.csv and the per-image masks/<stem>__thresholds.csv.
                _ch = cfg.channels
                thr_with_labels = dict(res.thresholds)
                thr_with_labels.setdefault(
                    "dapi_label", getattr(_ch, "dapi_label", "DAPI"),
                )
                thr_with_labels.setdefault(
                    "rna_label", getattr(_ch, "rna_label", "RNA1"),
                )
                if cfg.channels.analysis_mode == "rna_rna":
                    thr_with_labels.setdefault(
                        "rna2_label", getattr(_ch, "rna2_label", "RNA2"),
                    )
                if cfg.channels.analysis_mode in ("rna_protein", "ab_ab", "protein_only"):
                    thr_with_labels.setdefault(
                        "antibody_label",
                        getattr(_ch, "antibody_label", "Protein"),
                    )
                if cfg.channels.analysis_mode == "ab_ab":
                    thr_with_labels.setdefault(
                        "ab2_label", getattr(_ch, "ab2_label", "Protein2"),
                    )
                # ImageResult is a dataclass — assign the merged dict back
                # so the per-image masks/<stem>__thresholds.csv (written from
                # res.thresholds further down) also carries the labels.
                res.thresholds = thr_with_labels
                threshold_rows.append(thr_with_labels)

                # Per-image CSVs
                if cfg.output.save_per_image_csv:
                    _write_per_image_csvs(
                        dirs["per_image_csv"], stem,
                        res.nuclei, res.spots, prefix=prefix,
                    )
                    # rna_rna: also write per-channel spot CSVs for
                    # convenience (the master spot_metrics.csv already has
                    # a `channel` column to disambiguate).
                    if (
                        cfg.channels.analysis_mode == "rna_rna"
                        and len(res.spots) > 0
                        and "channel" in res.spots.columns
                    ):
                        for label, suffix in (("rna1", "spot_metrics_rna1"),
                                              ("rna2", "spot_metrics_rna2")):
                            sub = res.spots[res.spots["channel"] == label]
                            if len(sub) > 0:
                                sub.to_csv(
                                    dirs["per_image_csv"]
                                    / f"{prefix}{stem}__{suffix}.csv",
                                    index=False,
                                )

                # Masks + per-image thresholds CSV
                if cfg.output.save_masks:
                    qc = res.qc
                    labels = qc.get("labels")
                    if labels is not None:
                        _write_masks(
                            dirs["masks"], stem,
                            labels=labels,
                            rna_pos_mask=qc.get("rna_pos_mask"),
                            dapi_mask=qc.get("dapi_mask"),
                            prefix=prefix,
                        )
                        # rna_rna: also write the rna2 threshold mask
                        rna2_mask = qc.get("rna2_pos_mask")
                        if rna2_mask is not None:
                            _out.save_mask_tiff(
                                rna2_mask,
                                dirs["masks"] / f"{prefix}{stem}__spot_mask_rna2.tif",
                            )
                    _write_per_image_thresholds(
                        dirs["masks"], stem, res.thresholds, prefix=prefix,
                    )

                # QC overlays
                if cfg.output.save_qc_overlays:
                    qc = res.qc
                    dapi = qc.get("dapi_2d")
                    rna = qc.get("rna_2d")
                    rna2 = qc.get("rna2_2d")
                    labels = qc.get("labels")
                    vx = float(qc.get("voxel_xy_nm", 65.0))
                    # Resolve channel labels (display only — defaults match
                    # the legacy generic role names so back-compat is preserved).
                    _ch = cfg.channels
                    _dapi_lbl = getattr(_ch, "dapi_label", "DAPI") or "DAPI"
                    _rna_lbl = getattr(_ch, "rna_label", "RNA1") or "RNA1"
                    _rna2_lbl = getattr(_ch, "rna2_label", "RNA2") or "RNA2"
                    if dapi is not None and rna is not None and labels is not None:
                        if rna2 is not None:
                            spots1 = qc.get("spots1", pd.DataFrame())
                            spots2 = qc.get("spots2", pd.DataFrame())
                            all_in_one = _out.render_all_in_one_qc_rna_rna(
                                dapi, rna, rna2, labels, spots1, spots2, vx,
                                sec_only=bool(dimg.sec_only),
                                dapi_label=_dapi_lbl,
                                rna_label=_rna_lbl,
                                rna2_label=_rna2_lbl,
                            )
                            _out.save_png(
                                all_in_one,
                                dirs["qc_overlays"]
                                / f"{prefix}{stem}__qc_dapi_rna1_rna2_nuclei_spots.png",
                            )
                        else:
                            spots_for_overlay = res.spots if len(res.spots) else pd.DataFrame()
                            all_in_one = _out.render_all_in_one_qc(
                                dapi, rna, labels, spots_for_overlay, vx,
                                sec_only=bool(dimg.sec_only),
                                dapi_label=_dapi_lbl,
                                rna_label=_rna_lbl,
                            )
                            _out.save_png(
                                all_in_one,
                                dirs["qc_overlays"]
                                / f"{prefix}{stem}__qc_dapi_rna_nuclei_spots.png",
                            )
                        seg_qc = _out.render_segmentation_qc(dapi, labels, vx)
                        _out.save_png(
                            seg_qc,
                            dirs["qc_overlays"]
                            / f"{prefix}{stem}__qc_nuclei_on_dapi.png",
                        )

                # Publication images
                if cfg.output.save_publication_images:
                    qc = res.qc
                    dapi = qc.get("dapi_2d")
                    rna = qc.get("rna_2d")
                    rna2 = qc.get("rna2_2d")
                    vx = float(qc.get("voxel_xy_nm", 65.0))
                    if dapi is not None and rna is not None:
                        _ch = cfg.channels
                        _out.save_publication_images_bundle(
                            dirs["publication_images"], f"{prefix}{stem}",
                            dapi, rna, vx,
                            rna2=rna2,
                            sec_only=bool(dimg.sec_only),
                            dapi_label=getattr(_ch, "dapi_label", None),
                            rna_label=getattr(_ch, "rna_label", None),
                            rna2_label=getattr(_ch, "rna2_label", None),
                            antibody_label=getattr(_ch, "antibody_label", None),
                            dapi_lut=getattr(_ch, "dapi_lut", None),
                            rna_lut=getattr(_ch, "rna_lut", None),
                            rna2_lut=getattr(_ch, "rna2_lut", None),
                            antibody_lut=getattr(_ch, "antibody_lut", None),
                        )

                # Pipeline walkthrough — skip sec_only (no RNA threshold pretense)
                if not dimg.sec_only:
                    qc = res.qc
                    dapi = qc.get("dapi_2d")
                    rna = qc.get("rna_2d")
                    rna2 = qc.get("rna2_2d")
                    labels = qc.get("labels")
                    dmask = qc.get("dapi_mask")
                    rmask = qc.get("rna_pos_mask")
                    rmask2 = qc.get("rna2_pos_mask")
                    vx = float(qc.get("voxel_xy_nm", 65.0))
                    if all(x is not None for x in (dapi, rna, labels, dmask, rmask)):
                        if rna2 is not None and rmask2 is not None:
                            _out.save_walkthrough_bundle_rna_rna(
                                dirs["pipeline_walkthrough"], f"{prefix}{stem}",
                                dapi=dapi, rna1=rna, rna2=rna2,
                                dapi_mask=dmask, labels=labels,
                                rna1_pos_mask=rmask, rna2_pos_mask=rmask2,
                                voxel_xy_nm=vx, sec_only=False,
                            )
                        else:
                            _out.save_walkthrough_bundle(
                                dirs["pipeline_walkthrough"], f"{prefix}{stem}",
                                dapi=dapi, rna=rna, dapi_mask=dmask,
                                labels=labels, rna_pos_mask=rmask,
                                voxel_xy_nm=vx,
                                sec_only=False,
                            )

                # Per-nucleus popouts — skip sec_only
                if not dimg.sec_only and len(res.nuclei):
                    qc = res.qc
                    dapi = qc.get("dapi_2d")
                    rna = qc.get("rna_2d")
                    labels = qc.get("labels")
                    vx = float(qc.get("voxel_xy_nm", 65.0))
                    if dapi is not None and rna is not None and labels is not None:
                        _out.save_nuclei_popouts(
                            dirs["nuclei_popouts"], f"{prefix}{stem}",
                            dapi=dapi, rna=rna, labels=labels,
                            spots_df=res.spots,
                            per_nuc_rows=res.nuclei.to_dict(orient="records"),
                            voxel_xy_nm=vx,
                            n_per_image=2,
                        )

            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                failures.append((dimg.path.name, repr(e), tb))
                if verbose:
                    _console.print(f"[red]FAIL {dimg.path.name}: {e}[/red]\n{tb}")
                else:
                    _console.print(f"[red]FAIL {dimg.path.name}: {e}[/red]")
            progress.advance(task)

    # ---- Write master CSVs (Fiji column order via union of per-image cols) -
    per_image_df = pd.DataFrame(per_image_rows)
    per_image_df.to_csv(output_dir / f"{prefix}per_image_summary.csv", index=False)

    nuclei_df = pd.concat(nuclei_dfs, ignore_index=True) if nuclei_dfs else pd.DataFrame()
    nuclei_df.to_csv(output_dir / f"{prefix}nuclei_metrics.csv", index=False)

    spots_df = pd.concat(spots_dfs, ignore_index=True) if spots_dfs else pd.DataFrame()
    spots_df.to_csv(output_dir / f"{prefix}spot_metrics.csv", index=False)

    morph_df = pd.concat(morph_dfs, ignore_index=True) if morph_dfs else pd.DataFrame()
    morph_df.to_csv(output_dir / f"{prefix}cell_morphology.csv", index=False)

    thr_df = pd.DataFrame(threshold_rows)
    thr_df.to_csv(output_dir / f"{prefix}thresholds.csv", index=False)

    # ---- Excel workbook ----------------------------------------------------
    try:
        with pd.ExcelWriter(output_dir / f"{prefix}analysis_summary.xlsx", engine="openpyxl") as xl:
            pd.DataFrame([
                dict(field="purpose", value="fishsuite end-to-end results"),
                dict(field="version", value=__version__),
                dict(field="config", value=str(config_path)),
                dict(field="input_dir", value=str(input_dir)),
                dict(field="output_dir", value=str(output_dir)),
                dict(field="n_images", value=len(images)),
            ]).to_excel(xl, sheet_name="How_to_read", index=False)
            per_image_df.to_excel(xl, sheet_name="Per_Image_Summary", index=False)
            if len(nuclei_df):
                nuclei_df.to_excel(xl, sheet_name="Per_Nucleus_Metrics", index=False)
            if len(spots_df):
                spots_df.to_excel(xl, sheet_name="Per_Spot_Metrics", index=False)
            if len(morph_df):
                morph_df.to_excel(xl, sheet_name="Cell_Morphology", index=False)
            if len(thr_df):
                thr_df.to_excel(xl, sheet_name="Thresholds", index=False)
    except Exception as e:
        _console.print(f"[yellow]Could not write Excel workbook: {e}[/yellow]")

    # ---- Provenance --------------------------------------------------------
    run_config = dict(
        package="fishsuite",
        version=__version__,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        run_start_utc=datetime.fromtimestamp(t_start, tz=timezone.utc).isoformat(),
        run_end_utc=datetime.now(tz=timezone.utc).isoformat(),
        runtime_s=round(time.time() - t_start, 2),
        n_workers=n_workers,
        config_path=str(config_path),
        config_resolved=cfg.model_dump(mode="json"),
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        n_images=len(images),
        failures=[dict(image=f[0], error=f[1]) for f in failures],
        # Fiji-parity keys so combine_to_xlsx.py / downstream scripts can
        # find the same fields they find in a Fiji run_config.json.
        ANALYSIS_MODE=cfg.channels.analysis_mode,
        Z_MODE=cfg.z_stack.mode,
        Z_START=cfg.z_stack.start_slice,
        Z_END=cfg.z_stack.end_slice,
        APPLIED_PROFILE=cfg.experiment.name,
        DISP_FLOOR_PERCENTILE=_out.RNA_FLOOR_PCT,
        DISP_CEIL_PERCENTILE=_out.RNA_CEIL_PCT,
        SEGMENTATION_BACKEND=cfg.nuclei.backend,
        STARDIST_PROB_THRESHOLD=cfg.nuclei.prob_threshold,
        SPOT_BACKEND=cfg.foci.backend,
        BIGFISH_VOXEL_SIZE_NM=cfg.foci.bigfish_voxel_size_nm,
        BIGFISH_VOXEL_Z_NM=cfg.foci.bigfish_voxel_z_nm,
        BIGFISH_SPOT_RADIUS_NM=cfg.foci.bigfish_spot_radius_nm,
        BIGFISH_SPOT_RADIUS_Z_NM=cfg.foci.bigfish_spot_radius_z_nm,
        BIGFISH_THRESHOLD=cfg.foci.threshold_override,
        NUC_MIN_AREA_PX=cfg.nuclei.min_area_px,
        EXCLUDE_BORDER_NUCLEI=cfg.nuclei.exclude_border,
        DO_FOCI=cfg.foci.enabled,
        DO_CYTOPLASM=cfg.cytoplasm.enabled,
        CONDITION_MODE=cfg.conditions.mode,
        CONDITION_ORDER=cfg.conditions.condition_order,
        FOLDER_CONDITION_MAP=cfg.conditions.subfolder_conditions,
        SAVE_PER_IMAGE_CSV=cfg.output.save_per_image_csv,
        SAVE_QC_OVERLAYS=cfg.output.save_qc_overlays,
        SAVE_MASKS=cfg.output.save_masks,
        SAVE_PUBLICATION_IMAGES=cfg.output.save_publication_images,
        # Channel labels — promote to top-level fields so downstream tooling
        # (Fiji parity, Brian's plotting scripts) can read the human names
        # without having to walk into config_resolved.channels.
        CHANNEL_DAPI_LABEL=getattr(cfg.channels, "dapi_label", "DAPI"),
        CHANNEL_RNA_LABEL=getattr(cfg.channels, "rna_label", "RNA1"),
        CHANNEL_RNA2_LABEL=getattr(cfg.channels, "rna2_label", "RNA2"),
        CHANNEL_ANTIBODY_LABEL=getattr(cfg.channels, "antibody_label", "Protein"),
        CHANNEL_AB2_LABEL=getattr(cfg.channels, "ab2_label", "Protein2"),
    )
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, default=str)

    _console.print(
        f"[green]Done[/green] in {run_config['runtime_s']}s  "
        f"-> {output_dir}  (failures: {len(failures)})"
    )

    # ── Auto-run downstream figure step ────────────────────────────────────
    # Mirror the Fiji pipeline's run_downstream_analysis: after writing CSVs
    # + run_config.json, kick off analysis.single_condition_plots so the
    # output dir always ends with a populated figures/ directory. Best-
    # effort: stdout/stderr stream to a log file inside output_dir; runner
    # success is independent of this step's exit code (a downstream-only
    # failure should not bubble up as a batch failure).
    try:
        import subprocess as _sp
        _down_cwd = Path(r"F:\Image Analysis Work\image-analysis-pipeline\python")
        if _down_cwd.is_dir():
            _log_path = output_dir / "_downstream_plots.log"
            with open(_log_path, "w", encoding="utf-8") as _lf:
                _lf.write(f"# downstream auto-run from fishsuite runner — {datetime.now(tz=timezone.utc).isoformat()}\n")
                _lf.flush()
                rc = _sp.call(
                    [sys.executable, "-m", "analysis.single_condition_plots",
                     "--output-dir", str(output_dir)],
                    cwd=str(_down_cwd),
                    stdout=_lf, stderr=_sp.STDOUT,
                )
                _lf.write(f"\n# exit code: {rc}\n")
            _console.print(f"[dim]downstream figures: exit={rc}, log={_log_path}[/dim]")
        else:
            _console.print(
                "[yellow]downstream skipped[/yellow]: "
                f"{_down_cwd} not found"
            )
    except Exception as _exc:
        _console.print(f"[yellow]downstream failed[/yellow]: {_exc!r}")

    return dict(
        n_images=len(images),
        n_nuclei=int(len(nuclei_df)),
        n_spots=int(len(spots_df)),
        failures=failures,
        runtime_s=run_config["runtime_s"],
        output_dir=str(output_dir),
    )
