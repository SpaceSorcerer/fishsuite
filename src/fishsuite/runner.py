"""Top-level batch runner — discovers inputs, runs the pipeline per image, writes outputs.

Produces a Fiji-pipeline-compatible output directory layout so downstream
tools (combine_to_xlsx.py, single_condition_plots.py, R scripts) can
consume fishsuite output transparently.
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, BarColumn, TextColumn, TimeRemainingColumn

from . import __version__
from .config.schema import FishsuiteConfig
from .core import io as _io
from .core import output as _out
from .core.excel_report import (
    write_analysis_summary_workbook,
    write_raw_data_workbook,
)
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
        "nucleolus_overlay": output_dir / "nucleolus_overlay",
    }
    for p in out.values():
        p.mkdir(parents=True, exist_ok=True)
    return out


def _stem_with_condition(stem: str, condition: str | None) -> str:
    """Compose ``<condition_sanitized>__<stem>`` for per-image output files.

    2026-05-22 Brian: condition FIRST so files sort by condition when
    browsing the directory (WT_1, WT_2, KO_3, KO_4 ... grouped together).

    Examples:
        ("H9-MIAT-ASOs-_03", "NT ASO")    -> "NT_ASO__H9-MIAT-ASOs-_03"
        ("H9-MIAT-ASOs-_10", "Sec-Only")  -> "Sec_Only__H9-MIAT-ASOs-_10"
        ("H9-MIAT-ASOs-_03", None)        -> "H9-MIAT-ASOs-_03"
    """
    csan = _out.sanitize_condition_for_filename(condition)
    return f"{csan}__{stem}" if csan else stem


def _compute_common_filename_prefix(stems: list[str]) -> str:
    """Find the longest common leading substring across input stems, trimmed
    back to the last separator ('_', '-', or ' ') so we don't slice through
    a meaningful token.

    Used to auto-strip boilerplate from output filenames:
      ["TRANK1-CAMK2D-WT-KO_10_6ADVMLE fast",
       "TRANK1-CAMK2D-WT-KO_13_50ADVMLE fast",
       "TRANK1-CAMK2D-WT-KO_17"]
    -> "TRANK1-CAMK2D-WT-KO_"
    so "TRANK1-CAMK2D-WT-KO_10_6ADVMLE fast" becomes "10_6ADVMLE fast"
    in output files. Stems with no shared prefix (or only 1 file) get
    the empty string back.
    """
    if len(stems) <= 1:
        return ""
    common = os.path.commonprefix(stems)
    if not common:
        return ""
    # Trim back to the last separator inside the common prefix so the
    # leftover starts at a meaningful token boundary.
    for sep in ("_", "-", " "):
        idx = common.rfind(sep)
        if idx >= 0:
            common = common[: idx + 1]
            break
    # Sanity: don't strip if the common prefix is shorter than 4 chars
    # (no real boilerplate to remove).
    if len(common) < 4:
        return ""
    return common


def _simplify_stem(stem: str, strip_prefix: str) -> str:
    """Strip the auto-detected common prefix and replace spaces with
    underscores so output filenames are filesystem-friendly and short.
    """
    out = stem
    if strip_prefix and out.startswith(strip_prefix):
        out = out[len(strip_prefix):]
    out = out.replace(" ", "_")
    return out or stem  # never return empty string


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
# Parallel pre-scan worker (module-level so it is picklable for ProcessPool).
# ---------------------------------------------------------------------------

def _prescan_one(args):
    """Segment ONE image and return pooled nuclear pixels + labels.

    Runs in a worker process. Returns
        (path_str, v1, v2_or_None, labels_or_None, err_or_None)
    so the parent can pool identically to the serial path. cfg is a pydantic
    model (picklable); is_rna_rna selects the collector. Byte-identical to the
    serial collectors — same functions, just invoked in a subprocess.
    """
    path, cfg, is_rna_rna = args
    # ``is_rna_rna`` selects the TWO-channel collector. For rna_protein the
    # 2nd channel is the PROTEIN/antibody channel (rna_protein's collector
    # maps it into the rna2 slot internally), so it uses the two-channel path.
    _mode = getattr(cfg.channels, "analysis_mode", "")
    try:
        if is_rna_rna and _mode == "rna_protein":
            from .core.modes.rna_protein import collect_nuclear_rna_pixels as _c2
            v1, v2, labels = _c2(path, cfg=cfg)
            return (str(path), v1, v2, labels, None)
        elif is_rna_rna:
            from .core.modes.rna_rna import collect_nuclear_rna_pixels as _c2
            v1, v2, labels = _c2(path, cfg=cfg)
            return (str(path), v1, v2, labels, None)
        else:
            from .core.modes.rna_only import collect_nuclear_rna_pixels as _c1
            vals, labels = _c1(path, cfg=cfg)
            return (str(path), vals, None, labels, None)
    except Exception as e:  # noqa: BLE001
        import traceback
        return (str(path), None, None, None, f"{e!r}\n{traceback.format_exc()}")


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
    # 2026-06-10: ADDITIVE reproducibility — lock the broad global RNG state at
    # the VERY START (before any discovery/stochastic step) and log what was
    # seeded. Wrapped so a seeding failure can never abort the run. cfg.seed
    # defaults to 0; this is complementary to foci.partner_null_seed.
    try:
        from .core.repro import set_global_seeds as _set_global_seeds
        _seeded = _set_global_seeds(int(getattr(cfg, "seed", 0)))
        _console.print(
            f"[dim]Global seeds set (seed={getattr(cfg, 'seed', 0)}): "
            f"{_seeded}[/dim]"
        )
    except Exception as _seed_err:  # pragma: no cover - defensive
        _console.print(
            f"[yellow]Global seed lock skipped ({_seed_err})[/yellow]"
        )
    # 2026-05-25 Brian: per-preset scale-bar length + label font. The render
    # functions read these as output-module constants; the per-image pass is
    # serial single-process so setting them once here is safe and propagates
    # everywhere (all-in-one QC, publication images, walkthrough). Defaults
    # (50 µm / 32 px) leave legacy presets byte-identical.
    try:
        from .core import output as _output_mod
        _output_mod.SCALEBAR_UM = float(cfg.output.scalebar_um)
        _output_mod.SCALEBAR_FONT_PX = int(cfg.output.scalebar_font_px)
    except Exception:
        pass
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dirs = _make_output_dirs(output_dir, cfg)

    # 2026-06-10: ADDITIVE provenance — emit versions.txt + command.log near the
    # START so they exist even if the run later fails. Crash-proof: any failure
    # is logged and the run continues unaffected.
    try:
        from .core.repro import write_run_metadata as _write_run_metadata
        _meta_extra = {
            "analysis_mode": getattr(cfg.channels, "analysis_mode", "?"),
            "z_mode": getattr(cfg.z_stack, "mode", "?"),
        }
        _write_run_metadata(
            output_dir,
            config_path,
            output_dir,
            int(getattr(cfg, "seed", 0)),
            extra=_meta_extra,
        )
    except Exception as _meta_err:  # pragma: no cover - defensive
        _console.print(
            f"[yellow]Run-metadata write skipped ({_meta_err})[/yellow]"
        )

    # 2026-07-03 Brian: IF-intensity antibody-validation mode. This is a
    # plate-level pipeline (per-well signal routing, exposure gate, fold-over-
    # secondary-only, cross-condition Welch stats, shared-display micrographs)
    # that does NOT fit the per-image spots contract of the FISH modes. Divert
    # to its self-contained batch runner HERE — after seeds + provenance are
    # written, BEFORE the generic discover/loop — so every existing FISH mode's
    # code path below is byte-for-byte untouched. Lazy import keeps package
    # import unaffected.
    if getattr(cfg.channels, "analysis_mode", "") == "if_intensity":
        from .core.modes.if_intensity import run_if_batch as _run_if_batch
        return _run_if_batch(
            cfg,
            config_path,
            input_dir,
            output_dir,
            dirs,
            dry_run=dry_run,
            verbose=verbose,
        )

    prefix = cfg.output.prefix or ""

    images = _io.discover_inputs(
        input_dir,
        subfolder_conditions=cfg.conditions.subfolder_conditions,
        sec_only_folders=cfg.conditions.sec_only_folders,
        sec_only_files=cfg.conditions.sec_only_files,
        filename_conditions=cfg.conditions.filename_conditions,
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
    # Cache of FINAL (post-border-exclude) nuclei labels produced by the
    # batch threshold pre-scan, keyed by str(image path). Reused in the main
    # analysis pass so each image is segmented exactly ONCE per run (avoids a
    # 2x segmentation cost with slow backends such as cellpose). Populated
    # ONLY when threshold_scope == 'batch'; empty otherwise, so the per-image
    # path's behavior is byte-identical to before (dict.get -> None).
    precomputed_labels_by_path: Dict[str, np.ndarray] = {}
    pc_cfg = getattr(cfg, "pixel_coloc", None)
    # 2026-05-28 Brian: rna_protein is a TWO-channel mode (RNA + protein) and
    # pools BOTH channels in the batch pre-scan, exactly like rna_rna pools
    # rna + rna2. ``is_rna_rna`` here means "two-channel batch pre-scan".
    is_rna_rna = (cfg.channels.analysis_mode in ("rna_rna", "rna_protein"))
    if pc_cfg is not None and getattr(pc_cfg, "threshold_scope", "per_image") == "batch":
        try:
            from .core import thresholds as _thr
            if cfg.channels.analysis_mode == "rna_protein":
                from .core.modes.rna_protein import (
                    collect_nuclear_rna_pixels as _collect_two,
                )
                collect_nuclear_rna_pixels = None  # type: ignore[assignment]
            elif is_rna_rna:
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
            # 2026-05-27 PERF: resolve segmentation pre-scan worker count.
            # Device-aware: directml forces 1 (single GPU, no VRAM sharing);
            # CPU "auto" budgets memory + cores. seg_workers=1 -> legacy serial
            # loop (byte-identical). Each worker caps its BLAS/torch threads so
            # N_workers x threads stays <= logical cores.
            from .core.parallel import (
                resolve_workers as _resolve_workers,
                _init_worker_threads,
            )
            _seg_device = getattr(cfg.nuclei, "cellpose_device", "cpu")
            _seg_workers = _resolve_workers(
                getattr(cfg.parallel, "seg_workers", 1),
                kind="seg", device=_seg_device,
            )
            _tpw = int(getattr(cfg.parallel, "threads_per_worker", 0) or 0)
            _console.print(
                f"[bold]PRE-SCAN[/bold] segmentation workers: "
                f"[bold]{_seg_workers}[/bold] (device={_seg_device}, "
                f"threads/worker={_tpw or 'unset'})"
            )

            def _accumulate(path_str, v1, v2, labels, err):
                if err is not None:
                    _console.print(
                        f"[yellow]Pre-scan failed on {Path(path_str).name}: "
                        f"{err.splitlines()[0]} — image excluded from pool[/yellow]"
                    )
                    return
                if v1 is not None and v1.size > 0:
                    pooled_list.append(v1)
                if is_rna_rna and v2 is not None and v2.size > 0:
                    pooled2_list.append(v2)
                if labels is not None:
                    precomputed_labels_by_path[path_str] = labels

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                console=_console,
            ) as ppg:
                ptask = ppg.add_task("Pre-pass: nuclear-pixel pooling", total=len(images))
                if _seg_workers <= 1:
                    # Serial path — byte-identical to the pre-2026-05-27 loop.
                    for dimg in images:
                        try:
                            if _collect_two is not None:
                                v1, v2, labels = _collect_two(dimg.path, cfg=cfg)
                                _accumulate(str(dimg.path), v1, v2, labels, None)
                            else:
                                vals, labels = collect_nuclear_rna_pixels(dimg.path, cfg=cfg)
                                _accumulate(str(dimg.path), vals, None, labels, None)
                        except Exception as e:
                            _accumulate(str(dimg.path), None, None, None, repr(e))
                        ppg.advance(ptask)
                else:
                    from concurrent.futures import ProcessPoolExecutor, as_completed
                    _tasks = [(dimg.path, cfg, bool(is_rna_rna)) for dimg in images]
                    with ProcessPoolExecutor(
                        max_workers=_seg_workers,
                        initializer=_init_worker_threads,
                        initargs=(_tpw,),
                    ) as _pool:
                        _futs = [_pool.submit(_prescan_one, t) for t in _tasks]
                        for _fut in as_completed(_futs):
                            path_str, v1, v2, labels, err = _fut.result()
                            _accumulate(path_str, v1, v2, labels, err)
                            ppg.advance(ptask)
            if pooled_list:
                pooled = np.concatenate(pooled_list)
                # Costes needs the PAIRED partner pixels. pooled/pooled2 are
                # accumulated from the SAME nuclear masks in the same order
                # (collect_nuclear_rna_pixels returns rna/rna2 over one mask),
                # so they are pixel-paired. None for mad/percentile (unchanged).
                _pooled_other = (
                    np.concatenate(pooled2_list).tolist()
                    if (is_rna_rna and pc_cfg.threshold_mode == "costes" and pooled2_list)
                    else None
                )
                try:
                    batch_rna_threshold = float(_thr.coloc_threshold(
                        pooled.tolist(),
                        mode=pc_cfg.threshold_mode,
                        k_mad=float(pc_cfg.k_mad),
                        percentile=float(pc_cfg.percentile),
                        vals_other=_pooled_other,
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
                # Paired partner pixels for the rna2 Costes threshold (same
                # pixel order as pooled2). None for mad/percentile (unchanged).
                _pooled2_other = (
                    np.concatenate(pooled_list).tolist()
                    if (pc_cfg.threshold_mode == "costes" and pooled_list)
                    else None
                )
                try:
                    batch_rna2_threshold = float(_thr.coloc_threshold(
                        pooled2.tolist(),
                        mode=pc_cfg.threshold_mode,
                        k_mad=float(pc_cfg.k_mad),
                        percentile=float(pc_cfg.percentile),
                        vals_other=_pooled2_other,
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

    # ---- Batch-scope pub-image contrast pre-pass --------------------------
    # When ``output.pub_contrast_mode == "auto_batch"`` (the default), we
    # pool RAW pixels per channel across every non-sec-only image, compute
    # ONE (floor, ceil) per channel from configured percentiles, and apply
    # those uniform values to every image's publication PNG (and to the
    # sec-only images too — they render with the same contrast as the
    # real-probe images so the dim no-probe control correctly appears dim).
    #
    # This is "true" batch contrast — different from the legacy running-max
    # cache in core/output.py, which is order-dependent and biased by the
    # first image processed. The runner stores the computed values in
    # ``run_config.json`` under "batch_contrast" so downstream tooling /
    # Brian's manual audits can read back exactly what was applied.
    #
    # Manual mode: skip the pre-scan, apply the user-typed
    # ``manual_*_min/max`` values verbatim per image.
    # Auto-per-image mode: skip the pre-scan, each image computes its own
    # percentiles (legacy / Fiji "auto_per_image" parity).
    batch_contrast: Dict[str, tuple[float, float]] = {}
    # Hoisted out of the conditional so the run_config.json provenance block
    # at the end of run_batch can always read it (manual / auto_per_image
    # modes still populate this string).
    _pub_mode = getattr(cfg.output, "pub_contrast_mode", "auto_batch")
    # 2026-05-20 Brian/Sam: reference_image mode runs the same auto_batch
    # whole-batch pre-scan first to populate dapi + antibody fallback floor/
    # ceil values; the per-channel RNA values are then OVERWRITTEN below by
    # the reference-image computation. This keeps DAPI sensible without
    # requiring the user to also pin manual_dapi_min/max (though they can,
    # and those pins take precedence in the per-image render branch).
    # 2026-05-21 Brian: also run the auto_batch pre-scan in MANUAL mode
    # when any of manual_dapi_min/max, manual_rna_min/max, manual_rna2_min/max
    # are unset. Lets users pin RNA channels manually but get a consistent
    # auto-batch DAPI (Sam-style "auto per image" but applied uniformly
    # across the batch — fixes the DAPI inconsistency from pinning to a
    # single reference value that doesn't fit every image).
    _manual_needs_autobatch = (
        _pub_mode == "manual"
        and (
            cfg.output.manual_dapi_min is None or cfg.output.manual_dapi_max is None
            or cfg.output.manual_rna_min is None or cfg.output.manual_rna_max is None
            or cfg.output.manual_rna2_min is None or cfg.output.manual_rna2_max is None
        )
    )
    if (
        cfg.output.save_publication_images
        and (_pub_mode in ("auto_batch", "reference_image") or _manual_needs_autobatch)
        and len(images) > 0
    ):
        _floor_pct = float(cfg.output.pub_contrast_floor_pct)
        _ceil_pct = float(cfg.output.pub_contrast_ceil_pct)
        _dapi_floor_pct = float(cfg.output.pub_contrast_dapi_floor_pct)
        _dapi_ceil_pct = float(cfg.output.pub_contrast_dapi_ceil_pct)
        _console.print(
            "[bold]PRE-SCAN[/bold] pub_contrast_mode=auto_batch: pooling raw "
            f"pixels per channel across {len(images)} images (FISH "
            f"p{_floor_pct}/p{_ceil_pct}, DAPI "
            f"p{_dapi_floor_pct}/p{_dapi_ceil_pct})..."
        )
        # Per-channel pooled flat arrays (whole-image pixel values from the
        # same z-slice the per-image pass will use). Sec-only images are
        # EXCLUDED from the pool — their dim/autofluorescent background
        # would pull the ceiling down and make real-probe images look
        # blown out. This mirrors the Fiji
        # ``compute_pub_images_batch_contrast`` skip-sec-only behavior
        # (Coloc_Analysis.py lines 3990-3992).
        _pool: Dict[str, List[np.ndarray]] = {
            "dapi": [], "rna": [], "rna2": [], "antibody": [],
        }

        # Pre-resolve channel indices once via cfg + one-indexed flag (we
        # don't have an image handle here, so unresolved (-1) entries get
        # auto-detected per-image inside the loop).
        _ch_cfg = cfg.channels
        _one_indexed = bool(getattr(_ch_cfg, "one_indexed", False))

        def _chan_idx(raw: int) -> int:
            return (raw - 1) if (_one_indexed and raw > 0) else raw

        _cfg_dapi = _chan_idx(int(getattr(_ch_cfg, "dapi", -1)))
        _cfg_rna = _chan_idx(int(getattr(_ch_cfg, "rna", -1)))
        _cfg_rna2 = _chan_idx(int(getattr(_ch_cfg, "rna2", -1)))
        _cfg_ab = _chan_idx(int(getattr(_ch_cfg, "antibody", -1)))

        _z_mode = cfg.z_stack.mode
        _z_start = cfg.z_stack.start_slice
        _z_end = cfg.z_stack.end_slice
        _mode = cfg.channels.analysis_mode
        _need_rna2 = (_mode == "rna_rna")
        _need_ab = (_mode in ("rna_protein", "ab_ab", "protein_only"))

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=_console,
        ) as ppg:
            ptask = ppg.add_task(
                "Pre-pass: pub-image pixel pooling", total=len(images),
            )
            for dimg in images:
                # Sec-only images don't contribute to the pool — see comment
                # above. They still get rendered later with the same uniform
                # (floor, ceil) so they correctly appear dim.
                if bool(dimg.sec_only):
                    ppg.advance(ptask)
                    continue
                try:
                    img_h = _io.read_image(dimg.path)
                    # Resolve unset (-1) channel indices per-image via
                    # autodetect. We do this lazily so the pre-scan honors
                    # the same channel resolution the per-image run_one
                    # path would arrive at.
                    auto = None
                    di = _cfg_dapi
                    ri = _cfg_rna
                    r2i = _cfg_rna2
                    abi = _cfg_ab
                    if di < 0 or ri < 0 or (_need_rna2 and r2i < 0) or (_need_ab and abi < 0):
                        auto = _io.autodetect_channels(img_h)
                        if di < 0:
                            di = auto.get("dapi", -1)
                        if ri < 0:
                            ri = auto.get("rna", -1)
                        if _need_rna2 and r2i < 0:
                            # rna_rna mode: derive rna2 from remaining channels
                            # (rna_rna._resolve_channels does this — copy
                            # the cheap fallback: pick the first remaining
                            # non-DAPI non-RNA channel).
                            used = {di, ri}
                            for k in range(img_h.n_channels):
                                if k not in used and k >= 0:
                                    r2i = k
                                    break
                        if _need_ab and abi < 0:
                            abi = auto.get("ab", -1)

                    # Clamp to valid range
                    def _clamp(idx: int) -> int:
                        if idx < 0:
                            return -1
                        return max(0, min(img_h.n_channels - 1, idx))

                    di = _clamp(di)
                    ri = _clamp(ri)
                    r2i = _clamp(r2i) if _need_rna2 else -1
                    abi = _clamp(abi) if _need_ab else -1

                    # 2026-05-18 Brian: lock pre-scan pool to DAPI's
                    # autofocus z so the pooled pixels match the slice that
                    # will actually be rendered + analyzed downstream. Without
                    # this, the pool mixes RNA-autofocus-z pixels into the
                    # contrast histogram while the per-image render uses
                    # DAPI-autofocus-z — auto_batch min/max ends up calibrated
                    # to a different plane than the one the user sees.
                    dapi_z_lock: Optional[int] = None
                    # 2026-05-24 Brian: autofocus_maxproj pre-scan — compute
                    # the per-image DAPI focus window once and MIP it for all
                    # channels, so the pooled pixel histogram matches the
                    # per-image render's anatomy.
                    afm_zs_1: Optional[int] = None
                    afm_ze_1: Optional[int] = None
                    if di >= 0:
                        if _z_mode == "autofocus":
                            dapi_z_lock, d2 = _io.extract_channel_autofocus_with_idx(
                                img_h, di, z_start=_z_start, z_end=_z_end,
                                intensity_weighted=bool(getattr(
                                    cfg.z_stack, "autofocus_intensity_weighted", False)),
                            )
                        elif _z_mode == "autofocus_maxproj":
                            (afm_zs_1, afm_ze_1), _afm_diag, d2 = _io.extract_dapi_focus_window(
                                img_h, di,
                                metric=cfg.z_stack.focus_metric,
                                threshold_frac=float(cfg.z_stack.focus_threshold_frac),
                                min_slices=int(cfg.z_stack.focus_window_min_slices),
                                max_slices=int(cfg.z_stack.focus_window_max_slices),
                                z_start=_z_start, z_end=_z_end,
                                fixed_n_slices=int(getattr(cfg.z_stack, "focus_window_fixed_n_slices", 0)),
                                min_intensity_frac_of_peak=float(getattr(cfg.z_stack, "focus_min_intensity_frac_of_peak", 0.0)),
                                intensity_weighted=bool(getattr(cfg.z_stack, "autofocus_intensity_weighted", False)),
                                central_fraction=float(getattr(cfg.z_stack, "focus_central_fraction", 0.0)),
                            )
                        else:
                            d2 = _io.extract_channel(
                                img_h, di, z_mode=_z_mode,
                                z_start=_z_start, z_end=_z_end,
                            )
                            if d2.ndim != 2:
                                d2 = d2.max(axis=0)
                        _pool["dapi"].append(d2.astype(np.float32).ravel())
                    if ri >= 0:
                        if dapi_z_lock is not None:
                            r2 = _io.extract_channel_at_z(img_h, ri, z_1indexed=dapi_z_lock)
                        elif afm_zs_1 is not None and afm_ze_1 is not None:
                            r2 = _io.extract_channel_in_z_range(
                                img_h, ri,
                                z_start_1indexed=afm_zs_1,
                                z_end_1indexed=afm_ze_1,
                                project="maxproj",
                            )
                        else:
                            r2 = _io.extract_channel(
                                img_h, ri, z_mode=_z_mode,
                                z_start=_z_start, z_end=_z_end,
                            )
                            if r2.ndim != 2:
                                r2 = r2.max(axis=0)
                        _pool["rna"].append(r2.astype(np.float32).ravel())
                    if _need_rna2 and r2i >= 0:
                        if dapi_z_lock is not None:
                            r22 = _io.extract_channel_at_z(img_h, r2i, z_1indexed=dapi_z_lock)
                        elif afm_zs_1 is not None and afm_ze_1 is not None:
                            r22 = _io.extract_channel_in_z_range(
                                img_h, r2i,
                                z_start_1indexed=afm_zs_1,
                                z_end_1indexed=afm_ze_1,
                                project="maxproj",
                            )
                        else:
                            r22 = _io.extract_channel(
                                img_h, r2i, z_mode=_z_mode,
                                z_start=_z_start, z_end=_z_end,
                            )
                            if r22.ndim != 2:
                                r22 = r22.max(axis=0)
                        _pool["rna2"].append(r22.astype(np.float32).ravel())
                    if _need_ab and abi >= 0:
                        if dapi_z_lock is not None:
                            ab2 = _io.extract_channel_at_z(img_h, abi, z_1indexed=dapi_z_lock)
                        elif afm_zs_1 is not None and afm_ze_1 is not None:
                            ab2 = _io.extract_channel_in_z_range(
                                img_h, abi,
                                z_start_1indexed=afm_zs_1,
                                z_end_1indexed=afm_ze_1,
                                project="maxproj",
                            )
                        else:
                            ab2 = _io.extract_channel(
                                img_h, abi, z_mode=_z_mode,
                                z_start=_z_start, z_end=_z_end,
                            )
                            if ab2.ndim != 2:
                                ab2 = ab2.max(axis=0)
                        _pool["antibody"].append(ab2.astype(np.float32).ravel())
                except Exception as e:
                    _console.print(
                        f"[yellow]Pub-contrast pre-scan failed on {dimg.path.name}: "
                        f"{e} — image excluded from pool[/yellow]"
                    )
                ppg.advance(ptask)

        # Reduce: per-channel (floor, ceil) from configured percentiles.
        for _ch_key, _bundle in _pool.items():
            if not _bundle:
                continue
            pooled = np.concatenate(_bundle)
            if _ch_key == "dapi":
                lo_pct, hi_pct = _dapi_floor_pct, _dapi_ceil_pct
            else:
                lo_pct, hi_pct = _floor_pct, _ceil_pct
            try:
                f = float(np.percentile(pooled, lo_pct))
                c = float(np.percentile(pooled, hi_pct))
            except Exception as e:
                _console.print(
                    f"[yellow]Batch contrast percentile for {_ch_key} failed "
                    f"({e}); falling back to per-image[/yellow]"
                )
                continue
            # 2026-05-18 Brian: bump the auto floor for RNA-class channels
            # (RNA1, RNA2, antibody) by pub_contrast_rna_floor_bump_pct.
            # DAPI is exempt — its histogram structure is different
            # (bright nuclei against dark background) and the 10/99.9
            # pair already handles it cleanly. Manual-mode contrast is
            # NOT bumped — explicit numbers shouldn't be second-guessed.
            if _ch_key in ("rna", "rna2", "antibody"):
                bump = float(
                    getattr(cfg.output, "pub_contrast_rna_floor_bump_pct", 0.0)
                ) / 100.0
                if bump > 0 and f > 0:
                    f = f * (1.0 + bump)
            if c <= f:
                c = f + 1.0
            batch_contrast[_ch_key] = (f, c)
            _console.print(
                f"  batch_contrast[{_ch_key}] = ({f:.2f}, {c:.2f})  "
                f"n_pixels={pooled.size:,}"
            )

    # ---- Reference-image (Sam-style) pub-image contrast override ----------
    # When ``output.pub_contrast_mode == "reference_image"``, override the
    # RNA1 / RNA2 (floor, ceil) values in batch_contrast with values
    # computed from a per-channel REFERENCE IMAGE + REGION. This replicates
    # Sam's (Brian's PI's) manual B&C tuning:
    #
    #   * Introns probe (typically RNA1 / 640): tune on a KO image. Floor
    #     just above the cytoplasmic noise tail so cytoplasm clips to black,
    #     but bright nuclear puncta remain. Apply to all images.
    #
    #   * Exons probe (typically RNA2 / 561): tune on a WT image. Floor just
    #     above the nuclear background tail so the nucleoplasm clips to
    #     black, but cytoplasmic exon spots remain visible. Apply to all
    #     images.
    #
    # Region semantics:
    #   - "cytoplasm" : Voronoi-expanded label minus the nucleus interior
    #   - "nucleus"   : the nucleus label mask
    #   - "all"       : the entire 2D plane (whole image percentile)
    #
    # If the named reference image is missing from the discovered batch, or
    # if segmentation / channel extraction fails, we log a yellow warning
    # and FALL BACK to the auto_batch percentile for that channel (which
    # the block above already populated). DAPI + antibody are unaffected.
    if (
        cfg.output.save_publication_images
        and _pub_mode == "reference_image"
        and len(images) > 0
    ):
        # Lazy-import the segmentation + morphology helpers so the cheaper
        # modes don't pay the StarDist / scikit-image import cost.
        try:
            from .core import segmentation as _seg
            from .core import morphology as _morph
            _ref_imports_ok = True
        except Exception as e:
            _console.print(
                f"[yellow]reference_image mode: could not import segmentation/"
                f"morphology helpers ({e}); falling back to auto_batch for "
                f"all channels.[/yellow]"
            )
            _ref_imports_ok = False

        if _ref_imports_ok:
            # Build a basename → DiscoveredImage lookup so we can resolve the
            # configured reference images to their full paths.
            _by_name: Dict[str, Any] = {im.path.name: im for im in images}

            _ch_cfg = cfg.channels
            _one_indexed = bool(getattr(_ch_cfg, "one_indexed", False))

            def _chan_idx2(raw: int) -> int:
                return (raw - 1) if (_one_indexed and raw > 0) else raw

            _cfg_dapi = _chan_idx2(int(getattr(_ch_cfg, "dapi", -1)))
            _cfg_rna = _chan_idx2(int(getattr(_ch_cfg, "rna", -1)))
            _cfg_rna2 = _chan_idx2(int(getattr(_ch_cfg, "rna2", -1)))

            _z_mode = cfg.z_stack.mode
            _z_start = cfg.z_stack.start_slice
            _z_end = cfg.z_stack.end_slice

            def _resolve_idx(img_h, want: int, role: str) -> int:
                """Resolve a channel index, auto-detecting when unset (-1)."""
                if want >= 0:
                    return max(0, min(img_h.n_channels - 1, want))
                try:
                    auto = _io.autodetect_channels(img_h)
                except Exception:
                    return -1
                return int(auto.get(role, -1))

            def _compute_ref_channel(
                ref_name: str,
                channel_role: str,        # "rna" or "rna2" (for autodetect)
                cfg_channel_idx: int,
                floor_region: str,
                floor_pct: float,
                ceil_region: str,
                ceil_pct: float,
            ) -> Optional[tuple]:
                """Compute (floor, ceil) on a single reference image.

                Returns None on any failure (caller falls back to
                batch_contrast / auto_batch). Logs verbosely so Brian can
                audit what region + percentile produced what value.
                """
                dimg = _by_name.get(ref_name)
                if dimg is None:
                    _console.print(
                        f"[yellow]reference_image: reference '{ref_name}' for "
                        f"{channel_role} not found in discovered batch — "
                        f"falling back to auto_batch for that channel.[/yellow]"
                    )
                    return None
                try:
                    img_h = _io.read_image(dimg.path)
                except Exception as e:
                    _console.print(
                        f"[yellow]reference_image: failed to open '{ref_name}' "
                        f"({e}) — falling back to auto_batch.[/yellow]"
                    )
                    return None

                # Resolve channel indices (DAPI for segmentation + the RNA
                # channel of interest).
                di = _resolve_idx(img_h, _cfg_dapi, "dapi")
                ri = _resolve_idx(img_h, cfg_channel_idx, channel_role)
                if di < 0 or ri < 0:
                    _console.print(
                        f"[yellow]reference_image: could not resolve channel "
                        f"indices on '{ref_name}' (dapi={di}, "
                        f"{channel_role}={ri}) — falling back to auto_batch.[/yellow]"
                    )
                    return None

                # DAPI autofocus + DAPI-locked RNA extraction (same pattern
                # the main pipeline uses).
                try:
                    if _z_mode == "autofocus":
                        dapi_z, dapi_2d = _io.extract_channel_autofocus_with_idx(
                            img_h, di, z_start=_z_start, z_end=_z_end,
                            intensity_weighted=bool(getattr(
                                cfg.z_stack, "autofocus_intensity_weighted", False)),
                        )
                        rna_2d = _io.extract_channel_at_z(
                            img_h, ri, z_1indexed=dapi_z,
                        )
                    else:
                        dapi_2d = _io.extract_channel(
                            img_h, di, z_mode=_z_mode,
                            z_start=_z_start, z_end=_z_end,
                        )
                        if dapi_2d.ndim != 2:
                            dapi_2d = dapi_2d.max(axis=0)
                        rna_2d = _io.extract_channel(
                            img_h, ri, z_mode=_z_mode,
                            z_start=_z_start, z_end=_z_end,
                        )
                        if rna_2d.ndim != 2:
                            rna_2d = rna_2d.max(axis=0)
                except Exception as e:
                    _console.print(
                        f"[yellow]reference_image: channel extraction failed on "
                        f"'{ref_name}' ({e}) — falling back to auto_batch.[/yellow]"
                    )
                    return None

                # Segment nuclei + build Voronoi cytoplasm (reuse the same
                # cfg.nuclei + cfg.cytoplasm parameters as the per-image
                # pass would).
                nuc_mask = None
                cyt_mask = None
                if floor_region in ("cytoplasm", "nucleus") or ceil_region in ("cytoplasm", "nucleus"):
                    try:
                        seg_params = dict(
                            min_area=cfg.nuclei.min_area_px,
                            max_area=cfg.nuclei.max_area_px,
                            prob_threshold=cfg.nuclei.prob_threshold,
                            nms_threshold=cfg.nuclei.nms_threshold,
                            n_tiles=cfg.nuclei.n_tiles,
                            stardist_model=cfg.nuclei.stardist_model,
                            stardist_gauss_sigma=cfg.nuclei.stardist_gauss_sigma,
                            stardist_postprocess=cfg.nuclei.stardist_postprocess,
                            stardist_postprocess_dilate_px=cfg.nuclei.stardist_postprocess_dilate_px,
                            stardist_postprocess_otsu_sigma=cfg.nuclei.stardist_postprocess_otsu_sigma,
                            stardist_postprocess_mask_closing_px=cfg.nuclei.stardist_postprocess_mask_closing_px,
                            label_smoothing_radius_px=cfg.nuclei.label_smoothing_radius_px,
                            diameter=cfg.nuclei.cellpose_diameter_px,
                            flow_threshold=cfg.nuclei.cellpose_flow_threshold,
                            cellprob_threshold=cfg.nuclei.cellpose_cellprob_threshold,
                            cellpose_model_type=cfg.nuclei.cellpose_model_type,
                            cellpose_device=getattr(cfg.nuclei, "cellpose_device", "cpu"),
                        )
                        labels = _seg.segment_nuclei(
                            dapi_2d, backend=cfg.nuclei.backend, params=seg_params,
                        )
                        if cfg.nuclei.exclude_border:
                            labels = _seg.exclude_border_labels(
                                labels, margin_px=cfg.nuclei.border_margin_px,
                            )
                        nuc_mask = labels > 0
                        if cfg.cytoplasm.enabled and labels.max() > 0:
                            cyt_labels = _morph.compute_cytoplasm_mask(
                                labels,
                                max_expand_px=cfg.cytoplasm.voronoi_max_expansion_px,
                            )
                            cyt_mask = (cyt_labels > 0) & (~nuc_mask)
                    except Exception as e:
                        _console.print(
                            f"[yellow]reference_image: segmentation failed on "
                            f"'{ref_name}' ({e}) — falling back to auto_batch "
                            f"for that channel.[/yellow]"
                        )
                        return None

                def _pixels_for_region(region: str) -> Optional[np.ndarray]:
                    if region == "all":
                        return rna_2d.astype(np.float32).ravel()
                    if region == "cytoplasm":
                        if cyt_mask is None or not cyt_mask.any():
                            return None
                        return rna_2d[cyt_mask].astype(np.float32)
                    if region == "nucleus":
                        if nuc_mask is None or not nuc_mask.any():
                            return None
                        return rna_2d[nuc_mask].astype(np.float32)
                    return None

                floor_px = _pixels_for_region(floor_region)
                ceil_px = _pixels_for_region(ceil_region)
                if floor_px is None or floor_px.size == 0:
                    _console.print(
                        f"[yellow]reference_image: floor region '{floor_region}' "
                        f"on '{ref_name}' produced zero pixels — falling back "
                        f"to auto_batch for that channel.[/yellow]"
                    )
                    return None
                if ceil_px is None or ceil_px.size == 0:
                    _console.print(
                        f"[yellow]reference_image: ceil region '{ceil_region}' "
                        f"on '{ref_name}' produced zero pixels — falling back "
                        f"to auto_batch for that channel.[/yellow]"
                    )
                    return None

                try:
                    f_val = float(np.percentile(floor_px, float(floor_pct)))
                    c_val = float(np.percentile(ceil_px, float(ceil_pct)))
                except Exception as e:
                    _console.print(
                        f"[yellow]reference_image: percentile failed on "
                        f"'{ref_name}' ({e}) — falling back to auto_batch.[/yellow]"
                    )
                    return None
                # 2026-05-21 Brian: ONLY fall back when ceil ≤ floor (truly
                # broken / inverted range). The original "ceil ≤ floor × 1.2"
                # heuristic fired surprisingly under percentile tweaks
                # (e.g. asking for Exons ceil p99.9 produced ceil < floor*1.2
                # because of a fluke in the cyto pixel distribution, then
                # whole-image p99.9 was lower than region p99.5 → ceiling
                # collapsed). User-set percentiles should be trusted as long
                # as the resulting range is non-zero.
                region_ceil = c_val
                if c_val <= f_val:
                    fallback_ceil = float(np.percentile(rna_2d, 99.9))
                    if fallback_ceil > f_val * 1.2:
                        c_val = fallback_ceil
                        _console.print(
                            f"[yellow]reference_image ceil ({channel_role}) "
                            f"fell back to whole-image p99.9 ({c_val:.0f}) "
                            f"because region p{ceil_pct} ({region_ceil:.0f}) "
                            f"was too close to floor ({f_val:.0f}).[/yellow]"
                        )
                    else:
                        c_val = float(rna_2d.max())
                        _console.print(
                            f"[yellow]reference_image ceil ({channel_role}) "
                            f"fell back to whole-image MAX ({c_val:.0f}) "
                            f"because whole-image p99.9 ({fallback_ceil:.0f}) "
                            f"was also too close to floor "
                            f"({f_val:.0f}).[/yellow]"
                        )
                if c_val <= f_val:
                    c_val = f_val + 1.0
                _console.print(
                    f"  Reference-image contrast ({channel_role}): "
                    f"floor=[bold]{f_val:.2f}[/bold] ({floor_region} p{floor_pct} "
                    f"of {ref_name}); "
                    f"ceil=[bold]{c_val:.2f}[/bold] ({ceil_region} p{ceil_pct} "
                    f"of {ref_name})"
                )
                return (f_val, c_val)

            # RNA1
            _ref_rna_name = getattr(cfg.output, "manual_rna_reference_image", None)
            if _ref_rna_name:
                _console.print(
                    "[bold]PRE-SCAN[/bold] pub_contrast_mode=reference_image: "
                    f"computing RNA1 (floor, ceil) from '{_ref_rna_name}' ..."
                )
                _res = _compute_ref_channel(
                    _ref_rna_name, "rna", _cfg_rna,
                    cfg.output.manual_rna_floor_region,
                    float(cfg.output.manual_rna_floor_pct),
                    cfg.output.manual_rna_ceil_region,
                    float(cfg.output.manual_rna_ceil_pct),
                )
                if _res is not None:
                    batch_contrast["rna"] = _res
            else:
                _console.print(
                    "[yellow]reference_image: manual_rna_reference_image not "
                    "set — RNA1 will use auto_batch fallback.[/yellow]"
                )

            # RNA2 (only meaningful in rna_rna mode, but harmless otherwise)
            _ref_rna2_name = getattr(cfg.output, "manual_rna2_reference_image", None)
            if _ref_rna2_name:
                _console.print(
                    "[bold]PRE-SCAN[/bold] pub_contrast_mode=reference_image: "
                    f"computing RNA2 (floor, ceil) from '{_ref_rna2_name}' ..."
                )
                _res2 = _compute_ref_channel(
                    _ref_rna2_name, "rna", _cfg_rna2,
                    cfg.output.manual_rna2_floor_region,
                    float(cfg.output.manual_rna2_floor_pct),
                    cfg.output.manual_rna2_ceil_region,
                    float(cfg.output.manual_rna2_ceil_pct),
                )
                if _res2 is not None:
                    batch_contrast["rna2"] = _res2
            else:
                if cfg.channels.analysis_mode == "rna_rna":
                    _console.print(
                        "[yellow]reference_image: manual_rna2_reference_image "
                        "not set — RNA2 will use auto_batch fallback.[/yellow]"
                    )

            # 2026-05-21 Brian/Sam: Manual override for DAPI / antibody when
            # reference_image mode is set but the user pinned explicit values.
            # The reference_image fields (manual_rna_reference_image etc.) only
            # apply to RNA1/RNA2, so DAPI + antibody default to the auto_batch
            # fallback populated in the pre-scan above — but the manual_*_min/
            # max pair takes precedence over that fallback when BOTH halves are
            # set. Mirrors the manual-mode block's behavior (no bump, verbatim).
            # Caught when CAMK2D auto_tune ran with manual_dapi_min=100,
            # manual_dapi_max=800 but produced DAPI batch_contrast = (28, 1695)
            # from the auto_batch percentile fallback.
            _mn = getattr(cfg.output, "manual_dapi_min", None)
            _mx = getattr(cfg.output, "manual_dapi_max", None)
            if _mn is not None and _mx is not None:
                try:
                    _f = float(_mn)
                    _c = float(_mx)
                    if _c <= _f:
                        _c = _f + 1.0
                    batch_contrast["dapi"] = (_f, _c)
                    _console.print(
                        f"  batch_contrast[dapi] = ({_f:.2f}, {_c:.2f})  "
                        f"(manual override in reference_image mode)"
                    )
                except (TypeError, ValueError):
                    pass
            _mn = getattr(cfg.output, "manual_antibody_min", None)
            _mx = getattr(cfg.output, "manual_antibody_max", None)
            if _mn is not None and _mx is not None:
                try:
                    _f = float(_mn)
                    _c = float(_mx)
                    if _c <= _f:
                        _c = _f + 1.0
                    batch_contrast["antibody"] = (_f, _c)
                    _console.print(
                        f"  batch_contrast[antibody] = ({_f:.2f}, {_c:.2f})  "
                        f"(manual override in reference_image mode)"
                    )
                except (TypeError, ValueError):
                    pass

    # ---- Manual-mode batch_contrast population (2026-05-20 Brian/Sam) -------
    # When ``output.pub_contrast_mode == "manual"``, the auto_batch and
    # reference_image pre-scan branches above are skipped, so batch_contrast
    # stays empty. That breaks two downstream consumers:
    #
    #   (a) ``analysis_floors`` lookup further down (line ~916) reads
    #       batch_contrast["rna" / "rna2"] to forward the floor into the
    #       rna_rna mode → without this block, analysis_floors comes through
    #       as None and every *_above_floor_intensity_* column ends up NaN
    #       (even when apply_pub_contrast_floor_to_analysis = True).
    #
    #   (b) ``apply_pub_contrast_floor_to_spots`` filter in rna_rna.py reads
    #       the same analysis_floors dict → without this block, spots are
    #       never filtered against the manual floor.
    #
    # Mirror the manual_*_min/max into batch_contrast so all three modes
    # (auto_batch, reference_image, manual) feed the same dict uniformly.
    # Manual mode is verbatim user-typed values — NO bump applied (the
    # bump is reserved for auto_batch percentile-derived floors).
    if cfg.output.pub_contrast_mode == "manual":
        _manual_pairs = (
            ("rna",      "manual_rna_min",      "manual_rna_max"),
            ("rna2",     "manual_rna2_min",     "manual_rna2_max"),
            ("dapi",     "manual_dapi_min",     "manual_dapi_max"),
            ("antibody", "manual_antibody_min", "manual_antibody_max"),
        )
        for _key, _min_attr, _max_attr in _manual_pairs:
            _mn = getattr(cfg.output, _min_attr, None)
            _mx = getattr(cfg.output, _max_attr, None)
            if _mn is not None and _mx is not None:
                try:
                    _f = float(_mn)
                    _c = float(_mx)
                except (TypeError, ValueError):
                    continue
                if _c <= _f:
                    _c = _f + 1.0
                batch_contrast[_key] = (_f, _c)
                _console.print(
                    f"  batch_contrast[{_key}] = ({_f:.2f}, {_c:.2f})  "
                    f"(manual pub_contrast_mode)"
                )

    mode_fn = get_mode(cfg.channels.analysis_mode)

    per_image_rows: List[dict] = []
    nuclei_dfs: List[pd.DataFrame] = []
    spots_dfs: List[pd.DataFrame] = []
    morph_dfs: List[pd.DataFrame] = []
    threshold_rows: List[dict] = []
    # 2026-06-06 Brian: optional NATIVE coloc-figure carriers (default OFF ->
    # these stay empty -> no extra CSV written -> byte-identical output).
    coloc_null_draws_dfs: List[pd.DataFrame] = []
    coloc_radial_dfs: List[pd.DataFrame] = []
    coloc_rotation_null_dfs: List[pd.DataFrame] = []
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
        # 2026-05-21 Brian: auto-strip the common leading prefix from all
        # input filenames so per-image outputs use short, readable names.
        # e.g. "TRANK1-CAMK2D-WT-KO_10_6ADVMLE fast" -> "10_6ADVMLE_fast".
        _strip_prefix = _compute_common_filename_prefix([im.path.stem for im in images])
        task = progress.add_task("Processing images", total=len(images))
        for i, dimg in enumerate(images):
            progress.update(task, description=f"[{i+1}/{len(images)}] {dimg.path.name}")
            raw_stem = _simplify_stem(dimg.path.stem, _strip_prefix)
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
                # signature accepts it (rna_only + rna_rna + rna_protein).
                # rna_protein delegates to rna_rna's run_one with the antibody
                # channel mapped into the rna2 slot, so it accepts the SAME
                # precomputed_rna_threshold / precomputed_rna2_threshold (the
                # rna2 threshold here is the pooled PROTEIN-channel threshold).
                if (
                    batch_rna_threshold is not None
                    and cfg.channels.analysis_mode in ("rna_only", "rna_rna", "rna_protein")
                ):
                    _mode_kwargs["precomputed_rna_threshold"] = batch_rna_threshold
                if (
                    batch_rna2_threshold is not None
                    and cfg.channels.analysis_mode in ("rna_rna", "rna_protein")
                ):
                    _mode_kwargs["precomputed_rna2_threshold"] = batch_rna2_threshold
                # Reuse the nuclei labels the batch threshold pre-scan already
                # computed for this image so it is segmented exactly once per
                # run. rna_only / rna_rna / rna_protein run_one accept this
                # kwarg; the dict is empty unless threshold_scope == 'batch', so
                # per-image runs pass nothing and behavior is unchanged. Gate on
                # a cached entry actually existing so other modes never get it.
                if cfg.channels.analysis_mode in ("rna_only", "rna_rna", "rna_protein"):
                    _cached_labels = precomputed_labels_by_path.get(str(dimg.path))
                    if _cached_labels is not None:
                        _mode_kwargs["precomputed_labels"] = _cached_labels
                # 2026-05-20 Brian/Sam: forward the resolved per-channel pub-
                # image contrast floor as a hard quantification floor when
                # output.apply_pub_contrast_floor_to_analysis is True. The
                # floor values come from the batch_contrast dict the runner
                # has already resolved via auto_batch / manual /
                # reference_image (whichever mode is active). We pass even
                # when None — the mode treats missing floors as NaN columns
                # so the schema stays stable.
                # ALSO forward when apply_pub_contrast_floor_to_spots is True
                # — same dict, separate downstream effect (filters detected
                # spots whose peak intensity falls below the channel's floor).
                # 2026-06-02 Brian: ALWAYS forward for the supported modes so the
                # thresholded-intensity feature can DEFAULT its floor to the
                # resolved spot floor (the pub-contrast RNA floor) even when both
                # apply_pub_contrast_floor_to_* toggles are OFF. The modes treat
                # a missing/None floor as NaN columns, so always forwarding is
                # schema-stable and harmless for the legacy code paths.
                if (
                    cfg.channels.analysis_mode in ("rna_rna", "rna_only", "rna_protein")
                ):
                    _rna_floor_pair = batch_contrast.get("rna")
                    # rna_protein: the 2nd-channel floor is the PROTEIN/antibody
                    # contrast (batch_contrast["antibody"]); rna_rna uses rna2.
                    if cfg.channels.analysis_mode == "rna_protein":
                        _ab_floor_pair = batch_contrast.get("antibody")
                        _mode_kwargs["analysis_floors"] = {
                            "rna": (
                                float(_rna_floor_pair[0])
                                if _rna_floor_pair is not None else None
                            ),
                            "antibody": (
                                float(_ab_floor_pair[0])
                                if _ab_floor_pair is not None else None
                            ),
                        }
                    else:
                        _rna2_floor_pair = batch_contrast.get("rna2")
                        _mode_kwargs["analysis_floors"] = {
                            "rna": (
                                float(_rna_floor_pair[0])
                                if _rna_floor_pair is not None else None
                            ),
                            "rna2": (
                                float(_rna2_floor_pair[0])
                                if _rna2_floor_pair is not None else None
                            ),
                        }
                res = mode_fn(
                    dimg.path,
                    **_mode_kwargs,
                )

                # 2026-06-10: ADDITIVE per-image QC flags. Computed in the
                # runner (one place) so every mode gets a consistent qc_*
                # column set without touching any mode's per_image keys. Merged
                # via dict.update() -> purely additive. Crash-proof: a QC
                # failure must never drop the image or abort the run.
                try:
                    from .core.qc import compute_qc_flags as _compute_qc_flags
                    if isinstance(res.per_image, dict):
                        res.per_image.update(_compute_qc_flags(res, cfg))
                except Exception as _qc_err:
                    _console.print(
                        f"[yellow]QC-flag computation skipped for "
                        f"{dimg.path.name} ({_qc_err})[/yellow]"
                    )

                # Accumulate master tables
                per_image_rows.append(res.per_image)
                if len(res.nuclei):
                    nuclei_dfs.append(res.nuclei)
                if len(res.spots):
                    spots_dfs.append(res.spots)
                if len(getattr(res, "morphology", pd.DataFrame())):
                    morph_dfs.append(res.morphology)
                # NATIVE coloc-figure carriers (present only when the gating
                # flags are on; absent by default -> nothing accumulated).
                _cnd = res.extra.get("coloc_null_draws") if isinstance(res.extra, dict) else None
                if isinstance(_cnd, pd.DataFrame) and len(_cnd):
                    coloc_null_draws_dfs.append(_cnd)
                _crp = res.extra.get("coloc_radial_profile") if isinstance(res.extra, dict) else None
                if isinstance(_crp, pd.DataFrame) and len(_crp):
                    coloc_radial_dfs.append(_crp)
                _crot = res.extra.get("coloc_rotation_null") if isinstance(res.extra, dict) else None
                if isinstance(_crot, pd.DataFrame) and len(_crot):
                    coloc_rotation_null_dfs.append(_crot)
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
                    # rna_rna / rna_protein: also write per-channel spot CSVs
                    # for convenience (the master spot_metrics.csv already has
                    # a `channel` column to disambiguate). rna_protein labels
                    # its 2nd channel 'protein' (not 'rna2'), so the suffix is
                    # spot_metrics_protein for protein-correct output.
                    if (
                        cfg.channels.analysis_mode in ("rna_rna", "rna_protein")
                        and len(res.spots) > 0
                        and "channel" in res.spots.columns
                    ):
                        if cfg.channels.analysis_mode == "rna_protein":
                            _ch_csv_split = (("rna1", "spot_metrics_rna1"),
                                             ("protein", "spot_metrics_protein"))
                        else:
                            _ch_csv_split = (("rna1", "spot_metrics_rna1"),
                                             ("rna2", "spot_metrics_rna2"))
                        for label, suffix in _ch_csv_split:
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
                    # 2026-05-28 Brian: rna_protein labels its 2nd channel as
                    # the PROTEIN (antibody) — drive the QC overlay's 2nd-channel
                    # label + LUT from antibody_label / antibody_lut so the
                    # overlay reads e.g. "XRN2", not "RNA2".
                    _is_rna_protein = (cfg.channels.analysis_mode == "rna_protein")
                    _dapi_lbl = getattr(_ch, "dapi_label", "DAPI") or "DAPI"
                    _rna_lbl = getattr(_ch, "rna_label", "RNA1") or "RNA1"
                    if _is_rna_protein:
                        _rna2_lbl = getattr(_ch, "antibody_label", "Protein") or "Protein"
                    else:
                        _rna2_lbl = getattr(_ch, "rna2_label", "RNA2") or "RNA2"
                    if dapi is not None and rna is not None and labels is not None:
                        if rna2 is not None:
                            spots1 = qc.get("spots1", pd.DataFrame())
                            spots2 = qc.get("spots2", pd.DataFrame())
                            # 2026-05-19 Brian: pass BOTH channel LUT colors
                            # through so QC overlay markers / fills match the
                            # preset's rna_lut / rna2_lut (was hardcoding
                            # yellow for RNA1 regardless of LUT).
                            _rna_lut_name = (
                                getattr(_ch, "rna_lut", None) or "yellow"
                            )
                            _rna_w = _out.lut_name_to_weights(
                                _rna_lut_name, (1.0, 1.0, 0.0),
                            )
                            _rna_color_u8 = (
                                int(_rna_w[0] * 255),
                                int(_rna_w[1] * 255),
                                int(_rna_w[2] * 255),
                            )
                            _rna2_lut_name = (
                                (getattr(_ch, "antibody_lut", None) or "green")
                                if _is_rna_protein
                                else (getattr(_ch, "rna2_lut", None) or "magenta")
                            )
                            _rna2_w = _out.lut_name_to_weights(
                                _rna2_lut_name, (1.0, 0.0, 1.0),
                            )
                            _rna2_color_u8 = (
                                int(_rna2_w[0] * 255),
                                int(_rna2_w[1] * 255),
                                int(_rna2_w[2] * 255),
                            )
                            # 2026-05-21 Brian: pass batch_contrast values
                            # into the QC overlay so RNA channels match
                            # publication_images/.
                            # 2026-05-22 Brian: DAPI specifically uses
                            # PER-IMAGE auto contrast in QC overlays. Some
                            # batches (e.g. CAMK2D) have nuclear DAPI
                            # medians varying 10× across images — a single
                            # batch DAPI floor either washes out dim
                            # nuclei or oversaturates bright ones. QC
                            # overlays are for visual inspection of nucleus
                            # + spot localization, so each image should
                            # display its own DAPI cleanly. Publication
                            # PNGs still use batch DAPI for cross-image
                            # comparability.
                            _qc_df, _qc_dc = None, None  # per-image auto
                            _qc_rf, _qc_rc = batch_contrast.get("rna", (None, None))
                            # 2026-05-28 Brian: rna_protein's 2nd-channel QC
                            # contrast is the antibody channel's.
                            _qc_r2f, _qc_r2c = batch_contrast.get(
                                "antibody" if _is_rna_protein else "rna2", (None, None)
                            )
                            all_in_one = _out.render_all_in_one_qc_rna_rna(
                                dapi, rna, rna2, labels, spots1, spots2, vx,
                                sec_only=bool(dimg.sec_only),
                                dapi_label=_dapi_lbl,
                                rna_label=_rna_lbl,
                                rna2_label=_rna2_lbl,
                                rna_color=_rna_color_u8,
                                rna_lut_weights=_rna_w,
                                rna2_color=_rna2_color_u8,
                                rna2_lut_weights=_rna2_w,
                                dapi_floor_override=_qc_df,
                                dapi_ceil_override=_qc_dc,
                                rna_floor_override=_qc_rf,
                                rna_ceil_override=_qc_rc,
                                rna2_floor_override=_qc_r2f,
                                rna2_ceil_override=_qc_r2c,
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
                    # Antibody/protein channel 2D — keyed by either name
                    # depending on the mode that produced this qc dict.
                    protein_2d = qc.get("antibody_2d")
                    if protein_2d is None:
                        protein_2d = qc.get("protein_2d")
                    # 2026-05-28 Brian: rna_protein loads the antibody channel
                    # into rna_rna's rna2 slot, so qc carries it under BOTH
                    # rna2_2d AND antibody_2d. The PROTEIN render path
                    # (protein=, antibody_label/lut) is canonical for the 2nd
                    # channel — suppress the rna2 slot so it is NOT rendered a
                    # second time as "RNA2" (dedup) and no RNA2 merges appear.
                    # The canonical rna_protein render set is exactly:
                    # DAPI + rna1 (BIN1 introns, rna_lut) + protein (XRN2,
                    # antibody_lut). rna_rna / rna_only are unaffected.
                    if cfg.channels.analysis_mode == "rna_protein":
                        rna2 = None
                    vx = float(qc.get("voxel_xy_nm", 65.0))
                    if dapi is not None and rna is not None:
                        _ch = cfg.channels
                        # Resolve per-channel (floor, ceil) overrides based on
                        # pub_contrast_mode. auto_batch -> use the pre-scan
                        # batch_contrast dict (one uniform value per channel
                        # for the whole run). manual -> use cfg.output.manual_*
                        # min/max (None means "fall through to percentiles").
                        # auto_per_image -> leave all overrides as None so the
                        # bundle falls back to per-image percentile logic.
                        _df = _dc = _rf = _rc = None
                        _r2f = _r2c = _abf = _abc = None
                        if _pub_mode == "auto_batch":
                            _df, _dc = batch_contrast.get("dapi", (None, None))
                            _rf, _rc = batch_contrast.get("rna", (None, None))
                            _r2f, _r2c = batch_contrast.get("rna2", (None, None))
                            _abf, _abc = batch_contrast.get("antibody", (None, None))
                        elif _pub_mode == "manual":
                            _df = cfg.output.manual_dapi_min
                            _dc = cfg.output.manual_dapi_max
                            _rf = cfg.output.manual_rna_min
                            _rc = cfg.output.manual_rna_max
                            _r2f = cfg.output.manual_rna2_min
                            _r2c = cfg.output.manual_rna2_max
                            _abf = cfg.output.manual_antibody_min
                            _abc = cfg.output.manual_antibody_max
                        elif _pub_mode == "reference_image":
                            # Sam-style: rna / rna2 floor+ceil come from the
                            # per-channel reference-image computation that
                            # populated batch_contrast in the pre-scan. DAPI
                            # + antibody come from the auto_batch fallback
                            # values (also in batch_contrast), with manual_*
                            # overrides applied on top when set (the BIN1
                            # preset pins manual_dapi_min/max explicitly).
                            _rf, _rc = batch_contrast.get("rna", (None, None))
                            _r2f, _r2c = batch_contrast.get("rna2", (None, None))
                            _df, _dc = batch_contrast.get("dapi", (None, None))
                            _abf, _abc = batch_contrast.get("antibody", (None, None))
                            # Manual DAPI / antibody overrides win when both
                            # min and max are supplied; partial pins fall
                            # through to the batch_contrast value.
                            if (cfg.output.manual_dapi_min is not None
                                    and cfg.output.manual_dapi_max is not None):
                                _df = cfg.output.manual_dapi_min
                                _dc = cfg.output.manual_dapi_max
                            if (cfg.output.manual_antibody_min is not None
                                    and cfg.output.manual_antibody_max is not None):
                                _abf = cfg.output.manual_antibody_min
                                _abc = cfg.output.manual_antibody_max
                        _out.save_publication_images_bundle(
                            dirs["publication_images"], f"{prefix}{stem}",
                            dapi, rna, vx,
                            rna2=rna2,
                            protein=protein_2d,
                            sec_only=bool(dimg.sec_only),
                            # Floor/ceil overrides (None = use legacy
                            # percentile path).
                            dapi_floor=_df, dapi_ceil=_dc,
                            rna_floor=_rf, rna_ceil=_rc,
                            rna2_floor=_r2f, rna2_ceil=_r2c,
                            ab_floor=_abf, ab_ceil=_abc,
                            # Per-channel percentile knobs (used when an
                            # override is None — pub_contrast_floor_pct
                            # also influences sec-only renders in auto_per_image
                            # mode).
                            dapi_floor_pct=float(cfg.output.pub_contrast_dapi_floor_pct),
                            dapi_ceil_pct=float(cfg.output.pub_contrast_dapi_ceil_pct),
                            rna_floor_pct=float(cfg.output.pub_contrast_floor_pct),
                            rna_ceil_pct=float(cfg.output.pub_contrast_ceil_pct),
                            rna2_floor_pct=float(cfg.output.pub_contrast_floor_pct),
                            rna2_ceil_pct=float(cfg.output.pub_contrast_ceil_pct),
                            # 2026-05-18 Brian: post-percentile floor bump
                            # for RNA-class channels (auto_per_image path).
                            # auto_batch path applies the bump inside the
                            # pre-scan; this only affects per-image renders
                            # that fall through the percentile resolver.
                            rna_floor_bump_pct=float(
                                getattr(cfg.output,
                                        "pub_contrast_rna_floor_bump_pct",
                                        0.0)
                            ),
                            save_tifs=bool(cfg.output.save_publication_tifs),
                            dapi_label=getattr(_ch, "dapi_label", None),
                            rna_label=getattr(_ch, "rna_label", None),
                            rna2_label=getattr(_ch, "rna2_label", None),
                            antibody_label=getattr(_ch, "antibody_label", None),
                            dapi_lut=getattr(_ch, "dapi_lut", None),
                            rna_lut=getattr(_ch, "rna_lut", None),
                            rna2_lut=getattr(_ch, "rna2_lut", None),
                            antibody_lut=getattr(_ch, "antibody_lut", None),
                        )

                # Pipeline walkthrough — emitted for both real and sec-only
                # images (Brian 2026-05-14: sec-only no-probe controls still
                # need walkthrough PNGs so QC reviewers can inspect the
                # nuclear segmentation + background-level "threshold" the
                # same way as real images). sec_only=True is propagated to
                # the renderer so contrast-cache pollution is avoided
                # (consult-but-don't-update behavior).
                qc = res.qc
                dapi = qc.get("dapi_2d")
                rna = qc.get("rna_2d")
                rna2 = qc.get("rna2_2d")
                labels = qc.get("labels")
                dmask = qc.get("dapi_mask")
                rmask = qc.get("rna_pos_mask")
                rmask2 = qc.get("rna2_pos_mask")
                cyt = qc.get("cyt_labels")
                vx = float(qc.get("voxel_xy_nm", 65.0))
                # spots dataframes — for the new step07/08 (rna_only) and
                # step07/08/09 (rna_rna) panels. res.spots already mirrors
                # the per-spot rows; for rna_rna we use the channel-specific
                # frames from qc since res.spots stacks them with a 'channel'
                # column.
                _spots_rna_only = res.spots if len(res.spots) else None
                _spots_rna1 = qc.get("spots1")
                _spots_rna2 = qc.get("spots2")
                if all(x is not None for x in (dapi, rna, labels, dmask, rmask)):
                    if rna2 is not None and rmask2 is not None:
                        _ch_wk = cfg.channels
                        # 2026-05-19 Brian: pass BOTH RNA LUT colors through.
                        _rna_lut_wk = (
                            getattr(_ch_wk, "rna_lut", None) or "yellow"
                        )
                        _rna_w_wk = _out.lut_name_to_weights(
                            _rna_lut_wk, (1.0, 1.0, 0.0),
                        )
                        _rna_color_wk = (
                            int(_rna_w_wk[0] * 255),
                            int(_rna_w_wk[1] * 255),
                            int(_rna_w_wk[2] * 255),
                        )
                        # 2026-05-28 Brian: rna_protein → 2nd-channel walkthrough
                        # panel uses the antibody LUT (e.g. green for XRN2).
                        _is_rp_wk = (cfg.channels.analysis_mode == "rna_protein")
                        _rna2_lut_wk = (
                            (getattr(_ch_wk, "antibody_lut", None) or "green")
                            if _is_rp_wk
                            else (getattr(_ch_wk, "rna2_lut", None) or "magenta")
                        )
                        _rna2_w_wk = _out.lut_name_to_weights(
                            _rna2_lut_wk, (1.0, 0.0, 1.0),
                        )
                        _rna2_color_wk = (
                            int(_rna2_w_wk[0] * 255),
                            int(_rna2_w_wk[1] * 255),
                            int(_rna2_w_wk[2] * 255),
                        )
                        # 2026-05-22 Brian: pull pub contrast for walkthrough
                        # the same way callout figures do (manual or auto_batch).
                        _wk_df = _wk_dc = _wk_rf = _wk_rc = _wk_r2f = _wk_r2c = None
                        try:
                            _wk_pm = getattr(cfg.output, "pub_contrast_mode", "auto_batch")
                            # 2026-05-28 Brian: rna_protein's 2nd-channel contrast
                            # is the antibody channel's (manual_antibody_* /
                            # batch_contrast["antibody"]), not rna2's.
                            if _wk_pm == "manual":
                                _wk_df = cfg.output.manual_dapi_min
                                _wk_dc = cfg.output.manual_dapi_max
                                _wk_rf = cfg.output.manual_rna_min
                                _wk_rc = cfg.output.manual_rna_max
                                if _is_rp_wk:
                                    _wk_r2f = cfg.output.manual_antibody_min
                                    _wk_r2c = cfg.output.manual_antibody_max
                                else:
                                    _wk_r2f = cfg.output.manual_rna2_min
                                    _wk_r2c = cfg.output.manual_rna2_max
                            else:
                                _wk_df, _wk_dc = batch_contrast.get("dapi", (None, None))
                                _wk_rf, _wk_rc = batch_contrast.get("rna", (None, None))
                                _wk_r2f, _wk_r2c = batch_contrast.get(
                                    "antibody" if _is_rp_wk else "rna2", (None, None)
                                )
                        except Exception:
                            pass
                        # 2026-06-05 Brian: is the display floor (rna*_floor_override)
                        # the ACTUAL spot-detection gate? Only when
                        # apply_pub_contrast_floor_to_spots is on. When off (e.g.
                        # MIAT-QKI: low cosmetic MIAT floor, spots from BigFISH
                        # LoG), the step05/06 floor-mask would flood each nucleus
                        # into a solid mass; pass False so channels WITH spots
                        # render the actual called spots instead.
                        _wk_floor_is_gate = bool(
                            getattr(cfg.output, "apply_pub_contrast_floor_to_spots", False)
                        )
                        _out.save_walkthrough_bundle_rna_rna(
                            dirs["pipeline_walkthrough"], f"{prefix}{stem}",
                            dapi=dapi, rna1=rna, rna2=rna2,
                            dapi_mask=dmask, labels=labels,
                            rna1_pos_mask=rmask, rna2_pos_mask=rmask2,
                            voxel_xy_nm=vx,
                            sec_only=bool(dimg.sec_only),
                            spots1=_spots_rna1, spots2=_spots_rna2,
                            cyt_labels=cyt,
                            rna_color=_rna_color_wk,
                            rna_lut_weights=_rna_w_wk,
                            rna2_color=_rna2_color_wk,
                            rna2_lut_weights=_rna2_w_wk,
                            # 2026-05-19 Brian: filename-embedded labels so
                            # step04_<rna_label>_raw.png reflects the preset.
                            # 2026-05-28: rna_protein → 2nd-channel label is the
                            # antibody label (e.g. XRN2), not "RNA2".
                            rna_label=getattr(_ch_wk, "rna_label", None),
                            rna2_label=(
                                getattr(_ch_wk, "antibody_label", None)
                                if _is_rp_wk
                                else getattr(_ch_wk, "rna2_label", None)
                            ),
                            dapi_label=getattr(_ch_wk, "dapi_label", None),
                            dapi_floor_override=_wk_df,
                            dapi_ceil_override=_wk_dc,
                            rna_floor_override=_wk_rf,
                            rna_ceil_override=_wk_rc,
                            rna2_floor_override=_wk_r2f,
                            rna2_ceil_override=_wk_r2c,
                            floor_is_spot_gate=_wk_floor_is_gate,
                        )
                    else:
                        _ch_wk2 = cfg.channels
                        # 2026-06-02 Brian: when the RNA spot floor is the
                        # active spot gate (apply_pub_contrast_floor_to_spots),
                        # render the step05/06/08 threshold panels at the floor
                        # (manual_rna_min) — not the pixel-coloc MAD mask, which
                        # is much lower and looked over-exposed. None => legacy
                        # rna_pos_mask behaviour.
                        _wk_spot_floor = None
                        try:
                            if (
                                bool(getattr(cfg.output, "apply_pub_contrast_floor_to_spots", False))
                                and getattr(cfg.output, "manual_rna_min", None) is not None
                                and float(cfg.output.manual_rna_min) > 0.0
                            ):
                                _wk_spot_floor = float(cfg.output.manual_rna_min)
                        except Exception:
                            _wk_spot_floor = None
                        _out.save_walkthrough_bundle(
                            dirs["pipeline_walkthrough"], f"{prefix}{stem}",
                            dapi=dapi, rna=rna, dapi_mask=dmask,
                            labels=labels, rna_pos_mask=rmask,
                            voxel_xy_nm=vx,
                            sec_only=bool(dimg.sec_only),
                            spots=_spots_rna_only,
                            cyt_labels=cyt,
                            # 2026-05-19 Brian: filename-embedded labels for
                            # rna_only mode walkthroughs (rna_rna already
                            # threads labels through).
                            rna_label=getattr(_ch_wk2, "rna_label", None),
                            dapi_label=getattr(_ch_wk2, "dapi_label", None),
                            spot_floor=_wk_spot_floor,
                        )

                # Per-nucleus popouts — skip sec_only
                if not dimg.sec_only and len(res.nuclei):
                    qc = res.qc
                    dapi = qc.get("dapi_2d")
                    rna = qc.get("rna_2d")
                    labels = qc.get("labels")
                    vx = float(qc.get("voxel_xy_nm", 65.0))
                    if dapi is not None and rna is not None and labels is not None:
                        # 2026-05-22 Brian: reuse the same _cb_* contrast values
                        # computed below for save_nuclei_callout_figure so the
                        # individual popout PNGs also use pub contrast. rna2 is
                        # passed so rna_rna mode shows both channels.
                        # NOTE: _cb_* are assigned in the callout block below;
                        # initialise to None here so they exist even if the
                        # callout block is restructured later.
                        _pop_df = _pop_dc = _pop_rf = _pop_rc = _pop_r2f = _pop_r2c = None
                        try:
                            _pop_pm = getattr(cfg.output, "pub_contrast_mode", "auto_batch")
                            # 2026-05-28 Brian: rna_protein → 2nd-channel popout
                            # contrast is the antibody channel's.
                            _is_rp_pop = (cfg.channels.analysis_mode == "rna_protein")
                            if _pop_pm == "manual":
                                _pop_df = cfg.output.manual_dapi_min
                                _pop_dc = cfg.output.manual_dapi_max
                                _pop_rf = cfg.output.manual_rna_min
                                _pop_rc = cfg.output.manual_rna_max
                                if _is_rp_pop:
                                    _pop_r2f = cfg.output.manual_antibody_min
                                    _pop_r2c = cfg.output.manual_antibody_max
                                else:
                                    _pop_r2f = cfg.output.manual_rna2_min
                                    _pop_r2c = cfg.output.manual_rna2_max
                            else:
                                _pop_df, _pop_dc = batch_contrast.get("dapi", (None, None))
                                _pop_rf, _pop_rc = batch_contrast.get("rna", (None, None))
                                _pop_r2f, _pop_r2c = batch_contrast.get(
                                    "antibody" if _is_rp_pop else "rna2", (None, None)
                                )
                        except Exception:
                            pass
                        _out.save_nuclei_popouts(
                            dirs["nuclei_popouts"], f"{prefix}{stem}",
                            dapi=dapi, rna=rna, labels=labels,
                            spots_df=res.spots,
                            per_nuc_rows=res.nuclei.to_dict(orient="records"),
                            voxel_xy_nm=vx,
                            n_per_image=2,
                            dapi_floor=_pop_df, dapi_ceil=_pop_dc,
                            rna_floor=_pop_rf, rna_ceil=_pop_rc,
                            rna2=qc.get("rna2_2d"),
                            rna2_floor=_pop_r2f, rna2_ceil=_pop_r2c,
                        )
                        # 2026-05-22 Brian: combined main-image + popout figure
                        # with red boxes showing where each crop came from.
                        # Pull contrast floors/ceils from the same source the
                        # publication images use so the callout panels look
                        # identical to the publication renders.
                        _cb_df = _cb_dc = _cb_rf = _cb_rc = _cb_r2f = _cb_r2c = None
                        try:
                            _pm = getattr(cfg.output, "pub_contrast_mode", "auto_batch")
                            if _pm == "manual":
                                _cb_df = cfg.output.manual_dapi_min
                                _cb_dc = cfg.output.manual_dapi_max
                                _cb_rf = cfg.output.manual_rna_min
                                _cb_rc = cfg.output.manual_rna_max
                                if _is_rp_pop:
                                    _cb_r2f = cfg.output.manual_antibody_min
                                    _cb_r2c = cfg.output.manual_antibody_max
                                else:
                                    _cb_r2f = cfg.output.manual_rna2_min
                                    _cb_r2c = cfg.output.manual_rna2_max
                            else:
                                _cb_df, _cb_dc = batch_contrast.get("dapi", (None, None))
                                _cb_rf, _cb_rc = batch_contrast.get("rna", (None, None))
                                _cb_r2f, _cb_r2c = batch_contrast.get(
                                    "antibody" if _is_rp_pop else "rna2", (None, None)
                                )
                        except Exception:
                            pass
                        try:
                            _out.save_nuclei_callout_figure(
                                dirs["nuclei_popouts"], f"{prefix}{stem}",
                                dapi=dapi, rna=rna, rna2=qc.get("rna2_2d"),
                                labels=labels,
                                per_nuc_rows=res.nuclei.to_dict(orient="records"),
                                voxel_xy_nm=vx,
                                n_popouts=4,
                                dapi_floor=_cb_df, dapi_ceil=_cb_dc,
                                rna_floor=_cb_rf, rna_ceil=_cb_rc,
                                rna2_floor=_cb_r2f, rna2_ceil=_cb_r2c,
                            )
                        except Exception as _exc:
                            _console.print(
                                f"[yellow]nuclei callout figure failed for {stem}: "
                                f"{type(_exc).__name__}: {_exc}[/yellow]"
                            )

                # 2026-05-22 Brian: nucleolus QC overlay — moved OUTSIDE
                # the sec_only gate so sec-only images also get overlays
                # (useful for visually confirming negative controls).
                qc_obj = res.qc
                _nucleolus_labels = qc_obj.get("nucleolus_labels")
                _dapi_for_overlay = qc_obj.get("dapi_2d")
                _labels_for_overlay = qc_obj.get("labels")
                if (
                    _nucleolus_labels is not None
                    and _dapi_for_overlay is not None
                    and _labels_for_overlay is not None
                ):
                    try:
                        from .core.nucleolus import render_nucleolus_overlay
                        import imageio.v3 as _iio
                        overlay = render_nucleolus_overlay(
                            _dapi_for_overlay, _labels_for_overlay, _nucleolus_labels,
                        )
                        _iio.imwrite(
                            dirs["nucleolus_overlay"]
                            / f"{prefix}{stem}__nucleolus_overlay.png",
                            overlay,
                        )
                    except Exception as _exc:
                        import traceback as _tb
                        _console.print(
                            f"[yellow]nucleolus overlay save failed for {stem}: "
                            f"{type(_exc).__name__}: {_exc}[/yellow]\n"
                            f"{_tb.format_exc()}"
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

    # 2026-07-05: ADDITIVE run-level RNA1 over-detection outlier flag. Needs the
    # whole batch (median + MAD across images), so it runs here after every
    # per_image dict is collected. Advisory only — mutates qc_* columns, never
    # drops/reorders an image. Crash-proof (warn, never abort).
    try:
        from .core.qc import flag_overdetect_outliers as _flag_overdetect_outliers
        _n_over = _flag_overdetect_outliers(per_image_rows, cfg)
        if _n_over:
            _console.print(
                f"[yellow]QC: {_n_over} image(s) flagged qc_overdetect_rna1_run_outlier "
                f"(RNA1 spots/nucleus robust-outlier vs run median)[/yellow]"
            )
    except Exception as _over_err:
        _console.print(
            f"[yellow]Run-level over-detection QC skipped ({_over_err})[/yellow]"
        )

    # ---- Write master CSVs (Fiji column order via union of per-image cols) -
    per_image_df = pd.DataFrame(per_image_rows)
    per_image_df.to_csv(output_dir / f"{prefix}per_image_summary.csv", index=False)

    nuclei_df = pd.concat(nuclei_dfs, ignore_index=True) if nuclei_dfs else pd.DataFrame()
    nuclei_df.to_csv(output_dir / f"{prefix}nuclei_metrics.csv", index=False)

    spots_df = pd.concat(spots_dfs, ignore_index=True) if spots_dfs else pd.DataFrame()
    spots_df.to_csv(output_dir / f"{prefix}spot_metrics.csv", index=False)

    morph_df = pd.concat(morph_dfs, ignore_index=True) if morph_dfs else pd.DataFrame()
    morph_df.to_csv(output_dir / f"{prefix}cell_morphology.csv", index=False)

    # NATIVE coloc-figure CSVs — written ONLY when the gating flags produced
    # carriers (default OFF -> no file -> byte-identical to legacy runs).
    if coloc_null_draws_dfs:
        pd.concat(coloc_null_draws_dfs, ignore_index=True).to_csv(
            output_dir / f"{prefix}coloc_null_draws.csv", index=False
        )
    if coloc_radial_dfs:
        pd.concat(coloc_radial_dfs, ignore_index=True).to_csv(
            output_dir / f"{prefix}coloc_radial_profile.csv", index=False
        )
    if coloc_rotation_null_dfs:
        pd.concat(coloc_rotation_null_dfs, ignore_index=True).to_csv(
            output_dir / f"{prefix}coloc_rotation_null.csv", index=False
        )

    thr_df = pd.DataFrame(threshold_rows)
    thr_df.to_csv(output_dir / f"{prefix}thresholds.csv", index=False)

    _run_start_utc = datetime.fromtimestamp(
        t_start, tz=timezone.utc,
    ).isoformat()

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
        SAVE_PUBLICATION_TIFS=cfg.output.save_publication_tifs,
        # Pub-image contrast — record the strategy used + the per-channel
        # absolute (floor, ceil) actually applied so Brian / downstream
        # tooling can read back what contrast each PNG was rendered with
        # without having to inspect the image histograms. Empty dict in
        # manual / auto_per_image modes (no batch-wide values were computed).
        PUB_CONTRAST_MODE=_pub_mode,
        PUB_CONTRAST_FLOOR_PCT=float(cfg.output.pub_contrast_floor_pct),
        PUB_CONTRAST_CEIL_PCT=float(cfg.output.pub_contrast_ceil_pct),
        PUB_CONTRAST_DAPI_FLOOR_PCT=float(cfg.output.pub_contrast_dapi_floor_pct),
        PUB_CONTRAST_DAPI_CEIL_PCT=float(cfg.output.pub_contrast_dapi_ceil_pct),
        # 2026-05-18 Brian: post-percentile floor bump for RNA-class
        # channels (rna, rna2, antibody). Applied AFTER the
        # pub_contrast_floor_pct percentile selection. 0 = disabled.
        PUB_CONTRAST_RNA_FLOOR_BUMP_PCT=float(
            getattr(cfg.output, "pub_contrast_rna_floor_bump_pct", 0.0)
        ),
        batch_contrast={
            k: {"floor": float(v[0]), "ceil": float(v[1])}
            for k, v in batch_contrast.items()
        },
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

    # ---- Excel workbook (PI-ready) ----------------------------------------
    # Written AFTER run_config.json so the Run_Config sheet can read the
    # flattened JSON straight off disk.
    try:
        _fallback_cols = write_analysis_summary_workbook(
            out_path=output_dir / f"{prefix}analysis_summary.xlsx",
            per_image_df=per_image_df,
            nuclei_df=nuclei_df,
            spots_df=spots_df,
            morph_df=morph_df,
            thr_df=thr_df,
            fishsuite_version=__version__,
            run_start_utc=_run_start_utc,
            config_path=Path(config_path),
            input_dir=Path(input_dir),
            output_dir=Path(output_dir),
            z_mode=str(cfg.z_stack.mode),
            # start_slice/end_slice are None when using single mode + per-image
            # file_overrides (no global z-window); coalesce to 0 so the Excel
            # workbook still writes (was crashing int(None) -> Excel skipped).
            z_start=int(cfg.z_stack.start_slice or 0),
            z_end=int(cfg.z_stack.end_slice or 0),
            images=images,
            n_workers=int(n_workers),
        )
        if _fallback_cols:
            _console.print(
                f"[yellow]Excel glossary fallback fired for "
                f"{len(_fallback_cols)} column(s): "
                f"{', '.join(_fallback_cols[:8])}"
                f"{' ...' if len(_fallback_cols) > 8 else ''}[/yellow]"
            )
    except Exception as e:
        import traceback as _tb
        _console.print(
            f"[yellow]Could not write Excel workbook: {e}[/yellow]\n"
            f"{_tb.format_exc()}"
        )

    # ---- Excel companion: raw-data workbook -------------------------------
    try:
        write_raw_data_workbook(
            out_path=output_dir / f"{prefix}analysis_raw_data.xlsx",
            per_image_df=per_image_df,
            nuclei_df=nuclei_df,
            spots_df=spots_df,
            morph_df=morph_df,
            fishsuite_version=__version__,
            run_start_utc=_run_start_utc,
            output_dir=Path(output_dir),
        )
    except Exception as e:
        import traceback as _tb
        _console.print(
            f"[yellow]Could not write raw-data Excel workbook: {e}[/yellow]\n"
            f"{_tb.format_exc()}"
        )

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
