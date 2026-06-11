"""rna_only — per-nucleus FISH spot count, intensity, N/C stratification.

Outputs Fiji-pipeline-compatible row dicts (per_image_summary / nuclei_metrics /
spot_metrics / cell_morphology / thresholds) so downstream tools (Brian's
combine_to_xlsx.py, single_condition_plots.py, R scripts) can consume the
fishsuite output transparently.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
from skimage import measure
from skimage.filters import threshold_otsu

from .. import io as _io
from .. import segmentation as _seg
from .. import spots as _spots
from .. import morphology as _morph
from .. import thresholds as _thr
from .. import metrics as _metrics


def _resolve_thresh_intensity_floor(cfg, analysis_floors, channel: str) -> float:
    """Resolve the thresholded-intensity floor for ``channel`` ('rna'|'rna2').

    Precedence (2026-06-02 Brian — see OutputCfg.rna_intensity_threshold):
      1. cfg.output.<channel>_intensity_threshold when set and > 0 (explicit pin).
      2. the spot floor the runner forwarded via analysis_floors[channel]
         (the resolved pub-contrast floor / manual_<channel>_min).
      3. cfg.output.manual_<channel>_min read directly (so the feature still
         works without the runner, e.g. in unit tests).
      4. NaN -> the thresholded columns are emitted but empty (schema stable).

    Returns a float (NaN when no floor is resolvable).
    """
    out = getattr(cfg, "output", None)
    # 1) explicit per-channel pin
    pin_attr = "rna_intensity_threshold" if channel == "rna" else "rna2_intensity_threshold"
    pin = getattr(out, pin_attr, None) if out is not None else None
    try:
        if pin is not None and float(pin) > 0:
            return float(pin)
    except (TypeError, ValueError):
        pass
    # 2) runner-forwarded spot floor
    if analysis_floors:
        v = analysis_floors.get(channel)
        try:
            if v is not None and float(v) > 0:
                return float(v)
        except (TypeError, ValueError):
            pass
    # 3) manual_<channel>_min read directly from config
    man_attr = "manual_rna_min" if channel == "rna" else "manual_rna2_min"
    man = getattr(out, man_attr, None) if out is not None else None
    try:
        if man is not None and float(man) > 0:
            return float(man)
    except (TypeError, ValueError):
        pass
    # 4) no floor resolvable
    return float("nan")


@dataclass
class ImageResult:
    image: str
    condition: str
    sec_only: bool
    per_image: dict
    nuclei: pd.DataFrame
    spots: pd.DataFrame
    morphology: pd.DataFrame
    thresholds: dict
    qc: dict
    extra: Dict[str, Any] = field(default_factory=dict)


def _safe_float(v) -> float:
    try:
        f = float(v)
        if f != f:
            return float("nan")
        return f
    except (TypeError, ValueError):
        return float("nan")


def _measure_spot_diameter_um(
    rna_2d: np.ndarray,
    spots_df: pd.DataFrame,
    voxel_xy_um: float,
    crop_half: int = 4,
    fallback_diam_um: Optional[float] = None,
) -> np.ndarray:
    """Per-spot FWHM diameter (µm) via moment-based 2D Gaussian estimator.

    See rna_rna._measure_spot_diameter_um for full notes. Duplicated here
    to keep rna_only self-contained (rna_only has no import dep on rna_rna).
    """
    n = len(spots_df)
    if n == 0:
        return np.array([], dtype=np.float32)
    if rna_2d.ndim != 2:
        return np.full(n, float(fallback_diam_um) if fallback_diam_um else 2.0 * voxel_xy_um,
                       dtype=np.float32)
    H, W = rna_2d.shape
    diameters_um = np.zeros(n, dtype=np.float32)
    fallback = float(fallback_diam_um) if fallback_diam_um else (2.0 * voxel_xy_um)
    ys_arr = spots_df["y_px"].astype(float).to_numpy()
    xs_arr = spots_df["x_px"].astype(float).to_numpy()
    for i in range(n):
        cy = int(round(ys_arr[i]))
        cx = int(round(xs_arr[i]))
        y0, y1 = max(0, cy - crop_half), min(H, cy + crop_half + 1)
        x0, x1 = max(0, cx - crop_half), min(W, cx + crop_half + 1)
        if (y1 - y0) < 3 or (x1 - x0) < 3:
            diameters_um[i] = fallback
            continue
        crop = rna_2d[y0:y1, x0:x1].astype(np.float32)
        bg = float(np.percentile(crop, 10))
        sig = np.clip(crop - bg, 0, None)
        total = float(sig.sum())
        if total <= 0:
            diameters_um[i] = fallback
            continue
        ys_ix, xs_ix = np.indices(sig.shape, dtype=np.float32)
        my = float((ys_ix * sig).sum() / total)
        mx = float((xs_ix * sig).sum() / total)
        var = float((((ys_ix - my) ** 2 + (xs_ix - mx) ** 2) * sig).sum() / total)
        sigma_px = float(np.sqrt(max(var / 2.0, 0.25)))
        fwhm_px = 2.355 * sigma_px
        diameters_um[i] = float(fwhm_px * voxel_xy_um)
    return diameters_um


def _median(values):
    vals = [v for v in values if v == v]
    if not vals:
        return float("nan")
    s = sorted(vals)
    n = len(s)
    return s[n // 2] if n % 2 == 1 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def run_one(
    path,
    *,
    condition: str,
    sec_only: bool,
    cfg,
    precomputed_rna_threshold: Optional[float] = None,
    precomputed_labels: Optional[np.ndarray] = None,
    analysis_floors: Optional[Dict[str, Any]] = None,
) -> ImageResult:
    """Run the rna_only pipeline on a single image.

    Parameters
    ----------
    precomputed_labels : np.ndarray or None
        When supplied (by the batch runner during a
        ``pixel_coloc.threshold_scope == 'batch'`` run), this is the FINAL
        (already border-excluded) nuclei label image produced by the
        pre-pass ``collect_nuclear_rna_pixels`` for THIS exact image. When
        not None, ``run_one`` reuses it verbatim and SKIPS both the
        ``segment_nuclei`` call and the ``exclude_border_labels`` call, so
        each image is segmented exactly once per batch run (avoids the 2x
        segmentation cost with slow backends such as cellpose). The collect
        helper builds an identical ``seg_params`` and applies identical
        border exclusion, so the cached labels are bit-equivalent to what
        this function would otherwise compute. When None, segmentation runs
        exactly as before (per-image / non-batch path is unchanged).
    precomputed_rna_threshold : float or None
        When supplied (typically by the batch runner during a
        ``pixel_coloc.threshold_scope == 'batch'`` run), this scalar is used
        verbatim as the pixel-coloc threshold for THIS image — bypassing the
        per-image median+k*MAD computation. The runner does a pre-pass that
        pools raw nuclear RNA pixels across all images and computes ONE
        median+k*MAD value, then passes it here so every image in the batch
        gets the same threshold (matches Fiji's ``COLOC_THR_SCOPE == 'batch'``
        pre-scan; see ``Coloc_Analysis.run_batch_prescan_for_thresholds``).
    """
    t0 = time.time()
    img = _io.read_image(path)

    one_indexed = bool(cfg.channels.one_indexed)
    def _chan(idx: int) -> int:
        return (idx - 1) if (one_indexed and idx > 0) else idx

    dapi_idx = _chan(cfg.channels.dapi)
    rna_idx = _chan(cfg.channels.rna)
    if dapi_idx < 0 or rna_idx < 0:
        auto = _io.autodetect_channels(img)
        if dapi_idx < 0:
            dapi_idx = auto["dapi"]
        if rna_idx < 0:
            rna_idx = auto["rna"]
    dapi_idx = max(0, min(img.n_channels - 1, dapi_idx))
    rna_idx = max(0, min(img.n_channels - 1, rna_idx))

    z_mode = cfg.z_stack.mode
    z_start = cfg.z_stack.start_slice
    z_end = cfg.z_stack.end_slice
    # 2026-05-22 Brian: per-image z-window override. If the current image's
    # file name matches a key in cfg.z_stack.file_overrides, use that
    # image-specific single_slice / start_slice / end_slice instead of the
    # batch default. Mirrors the rna_rna.py lookup pattern exactly.
    _file_overrides = getattr(cfg.z_stack, "file_overrides", {}) or {}
    _img_name = Path(path).name
    _ovr = _file_overrides.get(_img_name, {}) if _img_name in _file_overrides else {}
    if _ovr:
        if "start_slice" in _ovr:
            z_start = int(_ovr["start_slice"])
        if "end_slice" in _ovr:
            z_end = int(_ovr["end_slice"])
        try:
            from rich.console import Console as _C
            _C().print(f"  [dim]z-override: {_img_name} → start={z_start}, end={z_end}[/dim]")
        except Exception:
            pass
    # 2026-05-25 Brian: per-channel z (DAPI single plane, RNA maxproj). When
    # this image's override carries BOTH rna_start_slice and rna_end_slice,
    # detect spots on a MAXPROJ of the RNA channel over that 1-indexed
    # inclusive window while DAPI segmentation stays on the normal path.
    # When absent → RNA extraction is unchanged (full back-compat).
    rna_start_slice = _ovr.get("rna_start_slice") if _ovr else None
    rna_end_slice = _ovr.get("rna_end_slice") if _ovr else None
    rna_per_channel_z = (rna_start_slice is not None and rna_end_slice is not None)
    if rna_per_channel_z:
        rna_start_slice = int(rna_start_slice)
        rna_end_slice = int(rna_end_slice)
        if rna_start_slice > img.n_z:
            rna_start_slice = 1
        if rna_end_slice > img.n_z:
            rna_end_slice = img.n_z
        try:
            from rich.console import Console as _C
            _C().print(
                f"  [dim]rna z-override: {_img_name} → RNA maxproj "
                f"[{rna_start_slice},{rna_end_slice}][/dim]"
            )
        except Exception:
            pass
    if z_start is not None and z_start > img.n_z:
        z_start = 1
    if z_end is not None and z_end > img.n_z:
        z_end = img.n_z

    # 2026-05-24 Brian: autofocus_maxproj — per-image DAPI focus-window
    # detection, then MIP that window for all channels. Replaces the need
    # for per-image file_overrides on datasets with field-to-field focus
    # drift (e.g. BIN1 KO_100x02 needed 15-49 vs default 9-78).
    # 2026-05-28 Brian: dapi_autofocus_z holds DAPI's picked plane in
    # z_mode == "autofocus" so RNA (here) and the antibody channel (in the
    # rna_protein wrapper, via qc) can be locked to that SAME plane.
    dapi_autofocus_z: Optional[int] = None
    if z_mode == "autofocus_maxproj":
        (afm_zs, afm_ze), afm_diag, dapi_2d = _io.extract_dapi_focus_window(
            img, dapi_idx,
            metric=cfg.z_stack.focus_metric,
            threshold_frac=float(cfg.z_stack.focus_threshold_frac),
            min_slices=int(cfg.z_stack.focus_window_min_slices),
            max_slices=int(cfg.z_stack.focus_window_max_slices),
            z_start=z_start, z_end=z_end,
            fixed_n_slices=int(getattr(cfg.z_stack, "focus_window_fixed_n_slices", 0)),
            min_intensity_frac_of_peak=float(getattr(cfg.z_stack, "focus_min_intensity_frac_of_peak", 0.0)),
            intensity_weighted=bool(getattr(cfg.z_stack, "autofocus_intensity_weighted", False)),
            central_fraction=float(getattr(cfg.z_stack, "focus_central_fraction", 0.0)),
        )
        if rna_per_channel_z:
            rna_2d = _io.extract_channel(
                img, rna_idx, z_mode="maxproj",
                z_start=rna_start_slice, z_end=rna_end_slice,
            )
            if rna_2d.ndim != 2:
                rna_2d = rna_2d.max(axis=0)
        else:
            rna_2d = _io.extract_channel_in_z_range(
                img, rna_idx,
                z_start_1indexed=afm_zs, z_end_1indexed=afm_ze,
                project="maxproj",
            )
        try:
            from rich.console import Console as _C
            _C().print(
                f"  [dim][autofocus_maxproj] {Path(path).name}: "
                f"focus peak at z={afm_diag['peak_z']+1}, "
                f"window=[{afm_zs},{afm_ze}] "
                f"({afm_ze - afm_zs + 1} slices)[/dim]"
            )
        except Exception:
            pass
    elif z_mode == "autofocus":
        # 2026-05-28 Brian: autofocus now LOCKS the RNA channel to DAPI's
        # picked focal plane. Previously DAPI and RNA were autofocused
        # INDEPENDENTLY (each calling extract_channel(z_mode="autofocus")),
        # so they could land on different physical planes — spot xy then came
        # from a different plane than the nuclear mask. We autofocus DAPI once
        # (bounded by the z_start/z_end window, including per-image
        # file_overrides), then read RNA at that exact plane. The DAPI-lock
        # takes precedence over rna_per_channel_z in this mode.
        dapi_autofocus_z, dapi_2d = _io.extract_channel_autofocus_with_idx(
            img, dapi_idx, z_start=z_start, z_end=z_end,
            intensity_weighted=bool(getattr(cfg.z_stack, "autofocus_intensity_weighted", False)),
        )
        rna_2d = _io.extract_channel_at_z(img, rna_idx, z_1indexed=dapi_autofocus_z)
        try:
            from rich.console import Console as _C
            if rna_per_channel_z:
                _C().print(
                    f"  [dim]z-lock: {Path(path).name} → RNA locked to DAPI plane "
                    f"z={dapi_autofocus_z} (per-channel RNA maxproj override IGNORED "
                    f"under autofocus)[/dim]"
                )
            else:
                _C().print(
                    f"  [dim]z-lock: {Path(path).name} → all channels @ DAPI plane "
                    f"z={dapi_autofocus_z}[/dim]"
                )
        except Exception:
            pass
    else:
        dapi_2d = _io.extract_channel(img, dapi_idx, z_mode=z_mode, z_start=z_start, z_end=z_end)
        if dapi_2d.ndim != 2:
            dapi_2d = dapi_2d.max(axis=0)
        if rna_per_channel_z:
            rna_2d = _io.extract_channel(
                img, rna_idx, z_mode="maxproj",
                z_start=rna_start_slice, z_end=rna_end_slice,
            )
        else:
            rna_2d = _io.extract_channel(img, rna_idx, z_mode=z_mode, z_start=z_start, z_end=z_end)
        if rna_2d.ndim != 2:
            rna_2d = rna_2d.max(axis=0)

    voxel_xy_nm = _safe_float(img.voxel_xy_nm)
    if not (voxel_xy_nm > 0):
        voxel_xy_nm = 65.0  # H9 100x fallback
    voxel_z_nm = _safe_float(img.voxel_z_nm)
    if not (voxel_z_nm > 0):
        voxel_z_nm = 230.0
    voxel_xy_um = voxel_xy_nm / 1000.0
    voxel_z_um = voxel_z_nm / 1000.0

    # ---- DAPI threshold mask (for walkthrough step 02) ---------------------
    try:
        dapi_thr_val = float(threshold_otsu(dapi_2d))
    except Exception:
        dapi_thr_val = float(dapi_2d.mean())
    dapi_mask = (dapi_2d >= dapi_thr_val).astype(np.uint8) * 255

    # ---- nuclear segmentation ----------------------------------------------
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
        cellpose_downsample_factor=cfg.nuclei.cellpose_downsample_factor,
        cellpose_device=getattr(cfg.nuclei, "cellpose_device", "cpu"),
    )
    if precomputed_labels is not None:
        # Batch threshold_scope pre-pass already segmented + border-excluded
        # this exact image; reuse those labels and SKIP re-segmentation so
        # each image is segmented exactly once per run. The cached labels are
        # already border-excluded, so set the bookkeeping accordingly.
        labels = precomputed_labels
        n_after = int(labels.max())
        n_before = n_after
        n_border_excluded = 0
    else:
        labels = _seg.segment_nuclei(dapi_2d, backend=cfg.nuclei.backend, params=seg_params)
        n_before = int(labels.max())
        if cfg.nuclei.exclude_border:
            labels = _seg.exclude_border_labels(labels, margin_px=cfg.nuclei.border_margin_px)
        n_after = int(labels.max())
        n_border_excluded = n_before - n_after

    # ---- cytoplasm mask -----------------------------------------------------
    cyt_labels = None
    if cfg.cytoplasm.enabled and n_after > 0:
        cyt_labels = _morph.compute_cytoplasm_mask(
            labels, max_expand_px=cfg.cytoplasm.voronoi_max_expansion_px
        )

    # ---- spot detection -----------------------------------------------------
    spots_df = pd.DataFrame()
    thr_val = float("nan")
    # 2026-05-25 Brian: optionally also detect spots on secondary-only control
    # images (foci.detect_in_sec_only). Sec-only spot counts are REPORTED
    # (flow through per_image_summary + per-nucleus CSV like any image) for
    # background QC, never subtracted from sample images. Default False keeps
    # legacy behavior (sec-only images skip detection → zero spots).
    if cfg.foci.enabled and (not sec_only or getattr(cfg.foci, "detect_in_sec_only", False)):
        vx = cfg.foci.bigfish_voxel_size_nm
        vz = cfg.foci.bigfish_voxel_z_nm
        if vx <= 0:
            vx = voxel_xy_nm
        if vz <= 0:
            vz = voxel_z_nm
        try:
            spots_df = _spots.detect_spots(
                rna_2d,
                backend=cfg.foci.backend,
                voxel_xy_nm=float(vx),
                voxel_z_nm=float(vz),
                spot_radius_nm=cfg.foci.bigfish_spot_radius_nm,
                spot_radius_z_nm=cfg.foci.bigfish_spot_radius_z_nm,
                threshold_multiplier=cfg.foci.threshold_multiplier,
                threshold=cfg.foci.threshold_override,
                log_threshold=cfg.foci.log_threshold,
                log_spot_radius_px=cfg.foci.log_spot_radius_px,
            )
            thr_val = float(spots_df["threshold_used"].iloc[0]) if len(spots_df) else float("nan")
        except Exception:
            spots_df = pd.DataFrame()
            thr_val = float("nan")

    # ---- Pub-contrast floor as HARD spot-detection floor -------------------
    # rna_only parity with rna_rna (2026-05-25 Brian): when
    # output.apply_pub_contrast_floor_to_spots is True AND the runner forwarded
    # the resolved RNA floor via analysis_floors, drop spots whose peak
    # intensity is below the floor — BEFORE stratification so every per-nucleus
    # count reflects the filtered set. This filter previously existed ONLY in
    # rna_rna mode, so the MIAT (rna_only) manual_rna_min floor never actually
    # applied — every rna_only run reported RAW BigFISH detections regardless
    # of floor. BigFISH detection itself is unchanged; this is a strict
    # post-detection filter.
    if (
        bool(getattr(cfg.output, "apply_pub_contrast_floor_to_spots", False))
        and analysis_floors and len(spots_df)
    ):
        _rna_floor = analysis_floors.get("rna")
        if _rna_floor:
            from .rna_rna import _filter_spots_by_floor as _ffbf
            _n0 = len(spots_df)
            spots_df = _ffbf(spots_df, _rna_floor, rna_2d)
            _nd = _n0 - len(spots_df)
            if _nd:
                print(f"  [floor-filter] {Path(path).name}: dropped {_nd}/{_n0} "
                      f"spots below floor={float(_rna_floor):.1f}")

    # ---- spot diameter --------------------------------------------------
    # Per-spot diameter is MEASURED via a moment-based 2D Gaussian estimator
    # on a small crop around each spot center (see _measure_spot_diameter_um
    # above). The previous behavior — reporting the configured BigFISH spot
    # radius doubled, i.e. a single constant per image — was an error noted
    # by Brian; the resulting ``spot_diameter_um`` column had no variability,
    # which broke spot-size figures (figures/25,26).
    spot_radius_um = float(cfg.foci.bigfish_spot_radius_nm) / 1000.0
    default_spot_diameter_um = 2.0 * spot_radius_um
    default_spot_fwhm_px = default_spot_diameter_um / max(voxel_xy_um, 1e-6)
    default_spot_area_px = math.pi * (default_spot_fwhm_px / 2.0) ** 2

    # Stratify spots vs nuclei / cytoplasm
    if cyt_labels is not None and len(spots_df) > 0:
        spots_df = _morph.stratify_spots(spots_df, labels, cytoplasm_labels=cyt_labels)
    elif len(spots_df) > 0:
        spots_df = _morph.stratify_spots(spots_df, labels)

    # Measured per-spot diameter (µm). Attach as a column so the per-nucleus
    # aggregator and the per-spot row both see real per-spot values.
    if len(spots_df) > 0:
        spots_df = spots_df.copy() if not isinstance(spots_df, pd.DataFrame) else spots_df
        spots_df["spot_diameter_um"] = _measure_spot_diameter_um(
            rna_2d, spots_df, voxel_xy_um,
            fallback_diam_um=default_spot_diameter_um,
        )

    # ---- RNA-positive mask for walkthrough step 05/06 ----------------------
    # Pixel-coloc threshold — matches Fiji's Coloc_Analysis.coloc_threshold()
    # design: collect RAW (unpreprocessed) RNA pixel values from within
    # nuclei and compute the MAD-based threshold = median + k_mad * MAD.
    # Confirmed against F:\Image Analysis Work\image-analysis-pipeline\
    # fiji_scripts\Coloc_Pipeline.py (line 682, 752, 760-761): Fiji uses
    # rna2d.convertToFloatProcessor() directly — NO rolling-ball, NO blur,
    # NO median filter applied before the threshold. Spot detection has its
    # own preprocessing (RNA_DETECT_ROLLINGBALL) but it is NEVER applied to
    # the pixel-coloc threshold.
    #
    # 2026-05-11: Brian's eye-check observation that the previous threshold
    # was "too permissive in some, too critical in others" was because the
    # old code used BigFISH's spot-detection threshold (which adapts to LoG
    # response per image) rather than the per-image MAD over RAW nuclear
    # pixels. Switching to the Fiji-style MAD on raw nuclear pixels matches
    # what Brian's manual Fiji runs do.
    #
    # threshold_scope:
    #   'per_image' (H9 default per Brian 2026-05-11): each image gets its
    #     own median+k_mad*MAD from its own nuclear pixels — adapts to
    #     out-of-focus haze that varies field-to-field.
    #   'batch': a pooled threshold across all images (not done at the
    #     per-image stage; pooled threshold would be applied by the runner).
    pc_cfg = getattr(cfg, "pixel_coloc", None)
    # When the batch runner has done a pre-pass and supplied a single pooled
    # threshold for the whole run (scope == 'batch'), use it as-is for every
    # image instead of computing per-image. This matches Fiji's
    # COLOC_THR_SCOPE == 'batch' branch (Coloc_Pipeline.py lines 287-304 +
    # Coloc_Analysis.run_batch_prescan_for_thresholds).
    _scope = getattr(pc_cfg, "threshold_scope", "per_image") if pc_cfg is not None else "per_image"
    if (
        pc_cfg is not None
        and _scope == "batch"
        and precomputed_rna_threshold is not None
        and precomputed_rna_threshold == precomputed_rna_threshold  # not NaN
        and precomputed_rna_threshold > 0
    ):
        rna_thr_value = float(precomputed_rna_threshold)
    elif pc_cfg is not None and n_after > 0:
        # Per-image: collect raw nuclear RNA pixels (Fiji's
        # collect_nuclear_pixel_values_fast) and compute median+k*MAD.
        nuc_pixel_mask = labels > 0
        if nuc_pixel_mask.any():
            rvals_img = rna_2d[nuc_pixel_mask].astype(np.float64).tolist()
            try:
                rna_thr_value = float(_thr.coloc_threshold(
                    rvals_img,
                    mode=pc_cfg.threshold_mode,
                    k_mad=float(pc_cfg.k_mad),
                    percentile=float(pc_cfg.percentile),
                ))
            except Exception:
                rna_thr_value = float("nan")
        else:
            rna_thr_value = float("nan")
    else:
        rna_thr_value = float("nan")

    # Robust fallback chain — if the MAD computation failed or there were no
    # nuclei to pool from, fall back to BigFISH auto-threshold, then Otsu,
    # then a hardcoded 99th-percentile. This keeps the walkthrough+QC
    # rendering robust on edge-case images (e.g. sec-only controls).
    if not (rna_thr_value == rna_thr_value and rna_thr_value > 0):
        if thr_val == thr_val and thr_val > 0:
            rna_thr_value = float(thr_val)
        else:
            try:
                rna_thr_value = float(threshold_otsu(rna_2d))
            except Exception:
                rna_thr_value = float(np.percentile(rna_2d, 99.0))
    rna_pos_mask = (rna_2d >= rna_thr_value).astype(np.uint8) * 255

    # ---- per-nucleus rows ---------------------------------------------------
    nuc_rows: List[Dict[str, Any]] = []
    spot_rows: List[Dict[str, Any]] = []
    morph_rows: List[Dict[str, Any]] = []

    # regionprops for shape descriptors
    rp_props = ("label", "area", "perimeter", "centroid",
                "eccentricity", "solidity", "feret_diameter_max",
                "major_axis_length", "minor_axis_length")
    if n_after > 0:
        rp_table = measure.regionprops_table(labels, properties=rp_props)
        rp_df = pd.DataFrame(rp_table)
    else:
        rp_df = pd.DataFrame(columns=["label"])
    rp_by_id = {int(r["label"]): r for r in rp_df.to_dict(orient="records")}

    # ---- Thresholded RNA intensity in compartments (2026-06-02 Brian) ------
    # Resolve the settable floor for this measurement ONCE per image (defaults
    # to the spot floor when unset/0). Applied to the SAME RNA plane used for
    # spot detection (rna_2d, the objective-window MIP). Per-nucleus values are
    # computed inside the loop below via
    # _metrics.compute_thresholded_compartment_intensity for BOTH compartments.
    rna_thresh_floor = _resolve_thresh_intensity_floor(cfg, analysis_floors, "rna")

    # spot-id offsets so global ids are unique per image (incremented across all)
    spot_global_id = 0

    # Per-nucleus aggregation of spots
    spots_by_nid: Dict[int, pd.DataFrame] = {}
    if len(spots_df) > 0:
        for nid_val, grp in spots_df.groupby("nucleus_id"):
            try:
                spots_by_nid[int(nid_val)] = grp
            except (TypeError, ValueError):
                pass

    img_name = path.name

    for nid in range(1, n_after + 1):
        rp = rp_by_id.get(nid, {})
        nucleus_area_px = int(rp.get("area", 0))
        perim_px = float(rp.get("perimeter", 0.0))
        area_um2 = nucleus_area_px * (voxel_xy_um ** 2)
        perimeter_um = perim_px * voxel_xy_um

        nuc_mask = labels == nid
        if nuc_mask.any():
            rna_vals = rna_2d[nuc_mask].astype(np.float64)
            dapi_vals = dapi_2d[nuc_mask].astype(np.float64)
            rna_mean_in_nucleus = float(rna_vals.mean())
            sum_rna_intensity = float(rna_vals.sum())
            dapi_mean_in_nucleus = float(dapi_vals.mean())
        else:
            rna_mean_in_nucleus = float("nan")
            sum_rna_intensity = float("nan")
            dapi_mean_in_nucleus = float("nan")

        if cyt_labels is not None:
            cyt_mask = (cyt_labels == nid) & (~nuc_mask)
            if cyt_mask.any():
                rna_cyto = rna_2d[cyt_mask].astype(np.float64)
                rna_cytoplasmic_mean = float(rna_cyto.mean())
                cyto_area_px = int(cyt_mask.sum())
            else:
                rna_cytoplasmic_mean = float("nan")
                cyto_area_px = 0
        else:
            cyt_mask = None
            rna_cytoplasmic_mean = float("nan")
            cyto_area_px = 0

        rna_nc_ratio = (rna_mean_in_nucleus / rna_cytoplasmic_mean) \
            if (rna_cytoplasmic_mean and rna_cytoplasmic_mean > 0
                and not math.isnan(rna_cytoplasmic_mean)) else float("nan")

        # ---- Thresholded RNA intensity, per compartment (2026-06-02 Brian) --
        # Threshold-and-integrate: sum/mean/area/fraction of pixels whose RAW
        # RNA value >= rna_thresh_floor, computed SEPARATELY for nucleus and
        # cytoplasm. Pixel-thresholding (diffuse + punctate), independent of
        # the spot-caller. Cyto values are all-NaN (area 0) when no cytoplasm
        # mask exists for this nucleus.
        _tn = _metrics.compute_thresholded_compartment_intensity(
            rna_2d, nuc_mask, rna_thresh_floor
        )
        if cyt_mask is not None:
            _tc = _metrics.compute_thresholded_compartment_intensity(
                rna_2d, cyt_mask, rna_thresh_floor
            )
        else:
            _tc = dict(
                thresh_total_intensity=float("nan"),
                thresh_mean_intensity=float("nan"),
                thresh_pos_area_px=0,
                thresh_pos_fraction=float("nan"),
            )

        # Per-nucleus spots
        sub = spots_by_nid.get(nid, pd.DataFrame())
        nuc_spot_mask = (sub.get("in_nucleus", pd.Series(dtype=bool)) == True)
        cyt_spot_mask = (sub.get("in_cytoplasm", pd.Series(dtype=bool)) == True)
        nuclear_spot_count = int(nuc_spot_mask.sum())
        cyto_spot_count = int(cyt_spot_mask.sum())
        rna_spot_count = nuclear_spot_count + cyto_spot_count
        nuclear_spot_fraction = (nuclear_spot_count / float(rna_spot_count)) if rna_spot_count > 0 else float("nan")

        nuclear_spot_density_per_um2 = (
            rna_spot_count / area_um2
        ) if area_um2 > 0 else float("nan")

        # Per-nucleus intensity aggregates over its spots
        if rna_spot_count > 0 and "intensity_peak" in sub.columns:
            ipeaks = sub["intensity_peak"].astype(float)
            rna_spot_mean_intensity_bgc_blend = float(ipeaks.mean())
            rna_spot_total_intensity_bgc_blend = float(ipeaks.sum())
            rna_spot_median_intensity_bgc_blend = float(ipeaks.median())
            # Approximate "fit" total = same series (BigFISH peak is fit-like).
            rna_spot_mean_intensity_fit = float(ipeaks.mean())
            rna_spot_total_intensity_fit = float(ipeaks.sum())
            rna_spot_median_intensity_fit = float(ipeaks.median())
            rna_spot_intensity_cv_fit = (
                float(ipeaks.std()) / float(ipeaks.mean()) if float(ipeaks.mean()) > 0 else float("nan")
            )
            spot_fit_success_count = rna_spot_count  # all succeeded (BigFISH model)
            spot_fit_success_fraction = 1.0
        else:
            rna_spot_mean_intensity_bgc_blend = float("nan")
            rna_spot_total_intensity_bgc_blend = float("nan")
            rna_spot_median_intensity_bgc_blend = float("nan")
            rna_spot_mean_intensity_fit = float("nan")
            rna_spot_total_intensity_fit = float("nan")
            rna_spot_median_intensity_fit = float("nan")
            rna_spot_intensity_cv_fit = float("nan")
            spot_fit_success_count = 0
            spot_fit_success_fraction = float("nan")

        # Spot size aggregates — now computed from MEASURED per-spot diameters
        # (column added to spots_df above). Falls back to the BigFISH-nominal
        # default if the measured column is somehow missing (older runs).
        if rna_spot_count > 0 and len(sub) > 0 and "spot_diameter_um" in sub.columns:
            d_um = pd.to_numeric(sub["spot_diameter_um"], errors="coerce").to_numpy()
            d_um = d_um[np.isfinite(d_um) & (d_um > 0)]
            if d_um.size > 0:
                mean_spot_diameter_um = float(d_um.mean())
                _vx = max(voxel_xy_um, 1e-6)
                fwhm_px_arr = d_um / _vx
                mean_spot_fwhm_px = float(fwhm_px_arr.mean())
                median_spot_fwhm_px = float(np.median(fwhm_px_arr))
                area_px_arr = math.pi * (fwhm_px_arr / 2.0) ** 2
                mean_spot_area_px = float(area_px_arr.mean())
            else:
                mean_spot_diameter_um = default_spot_diameter_um
                mean_spot_fwhm_px = default_spot_fwhm_px
                median_spot_fwhm_px = default_spot_fwhm_px
                mean_spot_area_px = default_spot_area_px
        elif rna_spot_count > 0:
            mean_spot_diameter_um = default_spot_diameter_um
            mean_spot_fwhm_px = default_spot_fwhm_px
            median_spot_fwhm_px = default_spot_fwhm_px
            mean_spot_area_px = default_spot_area_px
        else:
            mean_spot_diameter_um = float("nan")
            mean_spot_fwhm_px = float("nan")
            median_spot_fwhm_px = float("nan")
            mean_spot_area_px = float("nan")
        mean_spot_volume_vox = (
            4.0 / 3.0 * math.pi * (default_spot_fwhm_px / 2.0) ** 2
            * (cfg.foci.bigfish_spot_radius_z_nm / voxel_z_nm)
        ) if rna_spot_count > 0 else float("nan")
        mean_spot_volume_um3 = (
            4.0 / 3.0 * math.pi
            * (default_spot_diameter_um / 2.0) ** 2
            * (2.0 * cfg.foci.bigfish_spot_radius_z_nm / 1000.0 / 2.0)
        ) if rna_spot_count > 0 else float("nan")
        mean_spot_anisotropy = (
            (cfg.foci.bigfish_spot_radius_z_nm / cfg.foci.bigfish_spot_radius_nm)
        ) if rna_spot_count > 0 else float("nan")
        mean_spot_local_snr = float("nan")

        nuc_row = {
            "image": img_name,
            "condition": condition,
            "secondary_only": sec_only,
            "experiment_id": "",
            "nucleus_id": int(nid),
            "nucleus_area_px": int(nucleus_area_px),
            "rna_mean_in_nucleus": rna_mean_in_nucleus,
            "rna_nuclear_mean": rna_mean_in_nucleus,
            "rna_cytoplasmic_mean": rna_cytoplasmic_mean,
            "rna_nc_ratio": rna_nc_ratio,
            "rna_spot_count": int(rna_spot_count),
            "nuclear_spot_count": int(nuclear_spot_count),
            "cyto_spot_count": int(cyto_spot_count),
            "nuclear_spot_fraction": nuclear_spot_fraction,
            "nuclear_spot_density_per_um2": nuclear_spot_density_per_um2,
            "mean_spot_diameter_um": mean_spot_diameter_um,
            "mean_spot_fwhm_px": mean_spot_fwhm_px,
            "median_spot_fwhm_px": median_spot_fwhm_px,
            "mean_spot_area_px": mean_spot_area_px,
            "mean_spot_volume_vox": mean_spot_volume_vox,
            "mean_spot_volume_um3": mean_spot_volume_um3,
            "mean_spot_anisotropy": mean_spot_anisotropy,
            "mean_spot_local_snr": mean_spot_local_snr,
            "rna_spot_mean_intensity_bgc_blend": rna_spot_mean_intensity_bgc_blend,
            "rna_spot_total_intensity_bgc_blend": rna_spot_total_intensity_bgc_blend,
            "rna_spot_median_intensity_bgc_blend": rna_spot_median_intensity_bgc_blend,
            "rna_spot_mean_intensity_fit": rna_spot_mean_intensity_fit,
            "rna_spot_total_intensity_fit": rna_spot_total_intensity_fit,
            "rna_spot_median_intensity_fit": rna_spot_median_intensity_fit,
            "rna_spot_intensity_cv_fit": rna_spot_intensity_cv_fit,
            "spot_fit_success_count": int(spot_fit_success_count),
            "spot_fit_success_fraction": spot_fit_success_fraction,
            "sum_rna_intensity": sum_rna_intensity,
            # ---- Thresholded RNA intensity per compartment (2026-06-02) ----
            # Third intensity measurement: pixels with RAW value >=
            # rna_thresh_floor, integrated separately in nucleus and cytoplasm.
            "rna_thresh_total_intensity_nuclear": _tn["thresh_total_intensity"],
            "rna_thresh_mean_intensity_nuclear": _tn["thresh_mean_intensity"],
            "rna_thresh_pos_area_px_nuclear": int(_tn["thresh_pos_area_px"]),
            "rna_thresh_pos_fraction_nuclear": _tn["thresh_pos_fraction"],
            "rna_thresh_total_intensity_cyto": _tc["thresh_total_intensity"],
            "rna_thresh_mean_intensity_cyto": _tc["thresh_mean_intensity"],
            "rna_thresh_pos_area_px_cyto": int(_tc["thresh_pos_area_px"]),
            "rna_thresh_pos_fraction_cyto": _tc["thresh_pos_fraction"],
            "rna_thresh_floor": rna_thresh_floor,
            "cyto_area_px": int(cyto_area_px),
            "cyto_estimation_method": "voronoi" if cyt_labels is not None else "",
            "n_voxels": int(nucleus_area_px),
            "n_pix": int(nucleus_area_px),
            "n_z_slices": int(img.n_z),
            "z_mode": z_mode,
            "z_range": f"{z_start}-{z_end}" if (z_start and z_end) else "",
            "autofocus_z": "",
            "voxel_xy_um": voxel_xy_um,
            "voxel_z_um": voxel_z_um,
            "rna_threshold_value": rna_thr_value,
            "rna_frac_above_thr": float((rna_2d >= rna_thr_value).sum()) / float(rna_2d.size),
            "frac_spots_nuc_edge": float("nan"),
            "dapi_mean_in_nucleus": dapi_mean_in_nucleus,
        }
        nuc_rows.append(nuc_row)

        # Morphology row (one per nucleus)
        major = float(rp.get("major_axis_length", 0.0))
        minor = float(rp.get("minor_axis_length", 0.0))
        feret_max_um = float(rp.get("feret_diameter_max", 0.0)) * voxel_xy_um
        feret_min_um = (minor * voxel_xy_um) if minor > 0 else float("nan")
        aspect_ratio = (major / minor) if minor > 0 else float("nan")
        roundness = (minor / major) if major > 0 else float("nan")
        elongation = aspect_ratio
        solidity = float(rp.get("solidity", float("nan")))
        circularity = (
            4.0 * math.pi * nucleus_area_px / (perim_px ** 2)
        ) if perim_px > 0 else float("nan")
        morph_rows.append({
            "image": img_name,
            "condition": condition,
            "experiment_id": "",
            "cell_id": int(nid),
            "nucleus_id": int(nid),
            "segmentation_mode": cfg.nuclei.backend,
            "area_um2": area_um2,
            "perimeter_um": perimeter_um,
            "circularity": circularity,
            "aspect_ratio": aspect_ratio,
            "roundness": roundness,
            "elongation": elongation,
            "solidity": solidity,
            "feret_max_um": feret_max_um,
            "feret_min_um": feret_min_um,
        })

    # Per-spot rows
    if len(spots_df) > 0:
        for _, r in spots_df.iterrows():
            spot_global_id += 1
            x_px = int(r.get("x_px", 0))
            y_px = int(r.get("y_px", 0))
            z_slice = int(r.get("z_slice", 0))
            ipeak = float(r.get("intensity_peak", float("nan")))
            nid_at = int(r.get("nucleus_id", 0))
            in_nuc = bool(r.get("in_nucleus", False))
            in_cyt = bool(r.get("in_cytoplasm", False))
            # Use the MEASURED per-spot diameter attached to spots_df above.
            spot_diam_um = float(r.get("spot_diameter_um", default_spot_diameter_um))
            if not (spot_diam_um == spot_diam_um and spot_diam_um > 0):
                spot_diam_um = default_spot_diameter_um
            spot_fwhm_px_val = spot_diam_um / max(voxel_xy_um, 1e-6)
            spot_area_px_val = math.pi * (spot_fwhm_px_val / 2.0) ** 2
            spot_rows.append({
                "image": img_name,
                "condition": condition,
                "secondary_only": sec_only,
                "experiment_id": "",
                "spot_id": int(spot_global_id),
                "nucleus_id": nid_at,
                "x_px": x_px,
                "y_px": y_px,
                # 2026-05-25 Brian: surface in_nucleus / in_cytoplasm flags on
                # the per-spot rows (rna_rna already does this). Needed by the
                # nucleolus subnuclear-region classifier and Excel Section G,
                # which gate nucleolar counts on in_nucleus. Pure-additive.
                "in_nucleus": int(in_nuc),
                "in_cytoplasm": int(in_cyt),
                "z_slice": z_slice,
                "z_position_um": z_slice * voxel_z_um,
                "spot_peak_intensity": ipeak,
                "quality": ipeak,
                "spot_fwhm_px": spot_fwhm_px_val,
                "fwhm_xy_px_fit": spot_fwhm_px_val,
                "fwhm_z_px_fit": (cfg.foci.bigfish_spot_radius_z_nm * 2.355 / voxel_z_nm),
                "sigma_xy_px_fit": spot_fwhm_px_val / 2.355,
                "sigma_z_px_fit": (cfg.foci.bigfish_spot_radius_z_nm / voxel_z_nm),
                "spot_diameter_um": spot_diam_um,
                "spot_area_px": spot_area_px_val,
                "spot_volume_vox": (
                    4.0 / 3.0 * math.pi * (spot_fwhm_px_val / 2.0) ** 2
                    * (cfg.foci.bigfish_spot_radius_z_nm / voxel_z_nm)
                ),
                "spot_volume_um3": (
                    4.0 / 3.0 * math.pi
                    * (spot_diam_um / 2.0) ** 2
                    * (cfg.foci.bigfish_spot_radius_z_nm / 1000.0)
                ),
                "spot_anisotropy": (
                    cfg.foci.bigfish_spot_radius_z_nm / cfg.foci.bigfish_spot_radius_nm
                ),
                "integrated_intensity_fit": ipeak,
                "rna_mean_raw_disk": ipeak,
                "rna_mean_bgc_blend": ipeak,
                "rna_sum_bgc_blend": ipeak * spot_area_px_val,
                "rna_sum_raw_disk": ipeak * spot_area_px_val,
                "rna_bg_blend": float("nan"),
                "rna_contrast_blend": float("nan"),
                "spot_bg_estimate": float("nan"),
                "spot_to_nuc_edge_um": float("nan"),
                "spot_to_nuc_centroid_um": float("nan"),
                "spot_to_nuc_edge_px": float("nan"),
                "spot_to_nuc_centroid_px": float("nan"),
                "local_snr": float("nan"),
                "fit_ok": 1,
                "n_voxels_sampled": int(spot_area_px_val),
                "z_fwhm_slices": float(cfg.foci.bigfish_spot_radius_z_nm * 2.355 / voxel_z_nm),
                "colocalized": 0,
                "coloc_partner_id": -1,
                "coloc_partner_dist_px": float("nan"),
                "coloc_partner_dist_um": float("nan"),
                "coloc_partner_intensity": float("nan"),
                "contrast_threshold": float("nan"),
            })

    nuclei_df = pd.DataFrame(nuc_rows)
    spots_out_df = pd.DataFrame(spot_rows)
    morph_df = pd.DataFrame(morph_rows)

    # ---- Nucleolus + chromatin (optional) ----------------------------------
    # 2026-05-25 Brian: rna_only parity with rna_rna. When cfg.nucleolus.enabled,
    # detect DAPI-low subnuclear regions (nucleoli) and add nucleolus / chromatin
    # columns to nuclei_df + spots_out_df. SINGLE-RNA adaptation: rna_rna runs
    # classify on rna1 AND rna2; rna_only has one RNA channel (spots_out_df), so
    # we classify that single channel only. Column names match rna_rna exactly so
    # the Excel/figure code consumes them identically (the single RNA maps to the
    # rna1-equivalent "Introns" row in Excel Section G). Pure-additive: no existing
    # columns are removed or renamed; if disabled this block is a no-op (byte-
    # identical legacy behavior). Guarded in try/except — nucleolus is "nice to
    # have" and must never crash the run.
    nucleolus_enabled = (
        getattr(cfg, "nucleolus", None) is not None
        and getattr(cfg.nucleolus, "enabled", False)
    )
    nucleolus_labels_for_qc = None
    if nucleolus_enabled:
        try:
            from ..nucleolus import (
                NucleolusParams,
                detect_nucleoli,
                chromatin_metrics_per_nucleus,
                classify_spots_by_subnuclear_region,
            )
            _ncfg = cfg.nucleolus
            _params = NucleolusParams(
                intra_nuclear_percentile=float(_ncfg.intra_nuclear_percentile),
                min_area_um2=float(_ncfg.min_area_um2),
                max_area_frac_of_nucleus=float(_ncfg.max_area_frac_of_nucleus),
                closing_radius_px=int(_ncfg.closing_radius_px),
                min_border_distance_px=int(getattr(_ncfg, "min_border_distance_px", 3)),
            )
            _pix_um = float(voxel_xy_nm) / 1000.0 if voxel_xy_nm else 0.13
            nucleolus_labels = detect_nucleoli(
                labels, dapi_2d, pixel_size_um=_pix_um, params=_params
            )
            nucleolus_labels_for_qc = nucleolus_labels  # exposed via qc dict below
            # Per-spot in_nucleolus + refined in_nucleus_excluding_nucleolus
            # (single RNA channel).
            if len(spots_out_df) and "x_px" in spots_out_df.columns:
                spots_out_df = classify_spots_by_subnuclear_region(
                    spots_out_df, labels, nucleolus_labels
                )
            # Merge chromatin metrics into nuclei_df by nucleus_id
            chrom_df = chromatin_metrics_per_nucleus(labels, dapi_2d, nucleolus_labels)
            if len(chrom_df) and len(nuclei_df) and "nucleus_id" in nuclei_df.columns:
                # Drop duplicate columns the merge would otherwise create.
                _to_drop = [
                    c for c in ["nucleus_area_px"]
                    if c in chrom_df.columns and c in nuclei_df.columns
                ]
                chrom_df = chrom_df.drop(columns=_to_drop)
                nuclei_df = nuclei_df.merge(chrom_df, on="nucleus_id", how="left")
        except Exception as _exc:
            import traceback as _tb
            print(
                f"  WARN: nucleolus detection failed on {img_name} "
                f"({type(_exc).__name__}: {_exc}); continuing without nucleolus cols.\n"
                f"{_tb.format_exc()}"
            )

    # ---- OPT-IN ghost-nucleus rejection ------------------------------------
    # 2026-05-29: drop empty 'ghost' shells (out-of-focus border debris that
    # cellpose segments off the aberrant single-plane border band). DEFAULT
    # OFF (cfg.nuclei.reject_ghost_nuclei) => this whole block is skipped and
    # behavior is byte-for-byte unchanged for every other dataset/preset. The
    # rule is POST-spot-detection: it needs rna_spot_count + dapi_cv (both
    # already on nuclei_df by here). See core.segmentation.identify_ghost_nuclei.
    if getattr(cfg.nuclei, "reject_ghost_nuclei", False) and len(nuclei_df) > 0:
        # dapi_cv is normally merged in the nucleolus block above; if nucleolus
        # is disabled, compute the per-nucleus chromatin metrics on the fly so
        # the filter still has its texture column.
        if "dapi_cv" not in nuclei_df.columns:
            try:
                from ..nucleolus import chromatin_metrics_per_nucleus as _chrom
                _cdf = _chrom(labels, dapi_2d, None)
                if len(_cdf) and "nucleus_id" in nuclei_df.columns:
                    _keep = [c for c in ["nucleus_id", "dapi_cv"] if c in _cdf.columns]
                    nuclei_df = nuclei_df.merge(_cdf[_keep], on="nucleus_id", how="left")
            except Exception as _gexc:
                print(f"  WARN: ghost-filter dapi_cv compute failed on {img_name} "
                      f"({type(_gexc).__name__}: {_gexc}); skipping ghost filter.")
        if "dapi_cv" in nuclei_df.columns:
            _ghost_ids = _seg.identify_ghost_nuclei(
                nuclei_df,
                max_dapi_cv=float(getattr(cfg.nuclei, "reject_ghost_max_dapi_cv", 0.12)),
                min_area_px=int(getattr(cfg.nuclei, "reject_ghost_min_area_px", 6000)),
            )
            if _ghost_ids:
                _gset = set(int(g) for g in _ghost_ids)
                # 1) drop from per-nucleus table
                nuclei_df = nuclei_df[~nuclei_df["nucleus_id"].isin(_gset)].reset_index(drop=True)
                # 2) zero those labels so saved masks / QC overlays / popouts /
                #    nuc_mask and downstream counts are all consistent.
                if labels is not None and labels.size:
                    labels = labels.copy()
                    labels[np.isin(labels, list(_gset))] = 0
                # 3) drop those nuclei's spots (if any) from the spot table.
                if len(spots_out_df) and "nucleus_id" in spots_out_df.columns:
                    spots_out_df = spots_out_df[
                        ~spots_out_df["nucleus_id"].isin(_gset)
                    ].reset_index(drop=True)
                print(f"  ghost-filter: dropped {len(_gset)} empty nucleus shell(s) "
                      f"on {img_name} (ids={sorted(_gset)})")

    # Per-image summary row (Fiji-compatible columns)
    if len(nuclei_df) > 0:
        spot_counts = nuclei_df["rna_spot_count"].astype(int).tolist()
        n = len(spot_counts)
        m = sum(spot_counts) / float(n) if n > 0 else 0.0
        sd = math.sqrt(sum((v - m) ** 2 for v in spot_counts) / float(n - 1)) if n > 1 else 0.0
        cv = (sd / m) if m > 0 else float("nan")
        spot_diameters = [v for v in nuclei_df["mean_spot_diameter_um"].tolist()
                          if v == v]
        densities = [v for v in nuclei_df["nuclear_spot_density_per_um2"].tolist()
                     if v == v]
        cell_mean_int_blend = [
            float(r) for r, c in zip(
                nuclei_df["rna_spot_mean_intensity_bgc_blend"].tolist(),
                nuclei_df["rna_spot_count"].tolist())
            if c > 0 and r == r
        ]
        cell_total_int_fit = [
            float(r) for r, c in zip(
                nuclei_df["rna_spot_total_intensity_fit"].tolist(),
                nuclei_df["rna_spot_count"].tolist())
            if c > 0 and r == r
        ]
        if cell_total_int_fit and len(cell_total_int_fit) > 1:
            tm = sum(cell_total_int_fit) / float(len(cell_total_int_fit))
            tv = sum((v - tm) ** 2 for v in cell_total_int_fit) / float(len(cell_total_int_fit) - 1)
            tcv = (math.sqrt(tv) / tm) if tm > 0 else float("nan")
        else:
            tcv = float("nan")

        # Per-image mean of the per-nucleus thresholded-intensity values
        # (2026-06-02 Brian). Plain mean over finite per-nucleus values.
        def _col_mean(col: str) -> float:
            if col not in nuclei_df.columns:
                return float("nan")
            v = pd.to_numeric(nuclei_df[col], errors="coerce").to_numpy()
            v = v[np.isfinite(v)]
            return float(v.mean()) if v.size else float("nan")

        per_image = {
            "image": img_name,
            "condition": condition,
            "secondary_only": sec_only,
            "nuclei_analyzed": int(n),
            "mean_spots_per_nucleus": round(m, 3),
            "median_spots_per_nucleus": round(_median(spot_counts), 3),
            "cv_spots_per_nucleus": round(cv, 4) if cv == cv else float("nan"),
            "frac_nuclei_with_ge_1_spot": round(sum(1 for v in spot_counts if v >= 1) / float(n), 4) if n > 0 else 0.0,
            "frac_nuclei_with_ge_5_spots": round(sum(1 for v in spot_counts if v >= 5) / float(n), 4) if n > 0 else 0.0,
            "frac_nuclei_with_ge_10_spots": round(sum(1 for v in spot_counts if v >= 10) / float(n), 4) if n > 0 else 0.0,
            "mean_spot_diameter_um": round(sum(spot_diameters) / float(len(spot_diameters)), 4) if spot_diameters else float("nan"),
            "median_spot_diameter_um": round(_median(spot_diameters), 4) if spot_diameters else float("nan"),
            "mean_nuclear_spot_density_per_um2": round(sum(densities) / float(len(densities)), 6) if densities else float("nan"),
            "mean_cell_intensity_blend": round(sum(cell_mean_int_blend) / float(len(cell_mean_int_blend)), 3) if cell_mean_int_blend else float("nan"),
            "median_cell_intensity_blend": round(_median(cell_mean_int_blend), 3) if cell_mean_int_blend else float("nan"),
            "mean_cell_total_intensity_fit": round(sum(cell_total_int_fit) / float(len(cell_total_int_fit)), 2) if cell_total_int_fit else float("nan"),
            "median_cell_total_intensity_fit": round(_median(cell_total_int_fit), 2) if cell_total_int_fit else float("nan"),
            "cv_cell_total_intensity_fit": round(tcv, 4) if tcv == tcv else float("nan"),
            "mean_spot_volume_um3": round(default_spot_diameter_um, 5) if cell_total_int_fit else float("nan"),
            "mean_spot_anisotropy": round(
                cfg.foci.bigfish_spot_radius_z_nm / cfg.foci.bigfish_spot_radius_nm, 3
            ) if cell_total_int_fit else float("nan"),
            "n_nuclei_border_excluded": int(n_border_excluded),
            "total_spots": int(len(spots_out_df)),
            "spots_in_nuclei": int((spots_out_df.get("nucleus_id", pd.Series(dtype=int)) > 0).sum()) if len(spots_out_df) else 0,
            # ---- Per-image means of thresholded RNA intensity (2026-06-02) --
            "rna_thresh_floor": rna_thresh_floor,
            "mean_rna_thresh_total_intensity_nuclear": _col_mean("rna_thresh_total_intensity_nuclear"),
            "mean_rna_thresh_mean_intensity_nuclear": _col_mean("rna_thresh_mean_intensity_nuclear"),
            "mean_rna_thresh_pos_area_px_nuclear": _col_mean("rna_thresh_pos_area_px_nuclear"),
            "mean_rna_thresh_pos_fraction_nuclear": _col_mean("rna_thresh_pos_fraction_nuclear"),
            "mean_rna_thresh_total_intensity_cyto": _col_mean("rna_thresh_total_intensity_cyto"),
            "mean_rna_thresh_mean_intensity_cyto": _col_mean("rna_thresh_mean_intensity_cyto"),
            "mean_rna_thresh_pos_area_px_cyto": _col_mean("rna_thresh_pos_area_px_cyto"),
            "mean_rna_thresh_pos_fraction_cyto": _col_mean("rna_thresh_pos_fraction_cyto"),
            "runtime_s": round(time.time() - t0, 3),
            "dapi_channel": int(dapi_idx),
            "rna_channel": int(rna_idx),
            "voxel_xy_nm": voxel_xy_nm,
            "voxel_z_nm": voxel_z_nm,
            "n_z": int(img.n_z),
        }
    else:
        per_image = {
            "image": img_name,
            "condition": condition,
            "secondary_only": sec_only,
            "nuclei_analyzed": 0,
            "mean_spots_per_nucleus": 0.0,
            "median_spots_per_nucleus": 0.0,
            "cv_spots_per_nucleus": float("nan"),
            "frac_nuclei_with_ge_1_spot": 0.0,
            "frac_nuclei_with_ge_5_spots": 0.0,
            "frac_nuclei_with_ge_10_spots": 0.0,
            "mean_spot_diameter_um": float("nan"),
            "median_spot_diameter_um": float("nan"),
            "mean_nuclear_spot_density_per_um2": float("nan"),
            "mean_cell_intensity_blend": float("nan"),
            "median_cell_intensity_blend": float("nan"),
            "mean_cell_total_intensity_fit": float("nan"),
            "median_cell_total_intensity_fit": float("nan"),
            "cv_cell_total_intensity_fit": float("nan"),
            "mean_spot_volume_um3": float("nan"),
            "mean_spot_anisotropy": float("nan"),
            "n_nuclei_border_excluded": int(n_border_excluded),
            "total_spots": 0,
            "spots_in_nuclei": 0,
            # Thresholded RNA intensity provenance (no nuclei -> NaN means).
            "rna_thresh_floor": rna_thresh_floor,
            "runtime_s": round(time.time() - t0, 3),
            "dapi_channel": int(dapi_idx),
            "rna_channel": int(rna_idx),
            "voxel_xy_nm": voxel_xy_nm,
            "voxel_z_nm": voxel_z_nm,
            "n_z": int(img.n_z),
        }

    thresholds = {
        "image": img_name,
        "rna_threshold_used": thr_val,
        "rna_threshold_value": rna_thr_value,
        "rna_threshold_method": (
            "pixel_coloc_mad" if (pc_cfg is not None
                                  and rna_thr_value == rna_thr_value
                                  and pc_cfg.threshold_mode == "mad")
            else ("pixel_coloc_" + pc_cfg.threshold_mode if pc_cfg is not None
                  else "fallback")
        ),
        "rna_threshold_k_mad": float(pc_cfg.k_mad) if pc_cfg is not None else float("nan"),
        "rna_threshold_scope": getattr(pc_cfg, "threshold_scope", "") if pc_cfg is not None else "",
        "dapi_threshold_method": "Otsu dark",
        "dapi_threshold_value": dapi_thr_val,
        "watershed": cfg.nuclei.stardist_postprocess in ("watershed_otsu", "watershed_triangle"),
        "nuc_min_area_px": cfg.nuclei.min_area_px,
        "exclude_border_nuclei": cfg.nuclei.exclude_border,
        "z_mode": z_mode,
        "z_start": z_start,
        "z_end": z_end,
        "segmentation_backend": cfg.nuclei.backend,
        "stardist_prob_threshold": cfg.nuclei.prob_threshold,
        "spot_backend": cfg.foci.backend,
        "bigfish_spot_radius_nm": cfg.foci.bigfish_spot_radius_nm,
        "bigfish_voxel_size_nm": voxel_xy_nm,
        "bigfish_voxel_z_nm": voxel_z_nm,
        "trackmate_threshold": "",
        "rna_detect_blur_sigma": "",
        "rna_detect_rollingball": "",
    }

    qc = dict(
        labels=labels,
        dapi_2d=dapi_2d,
        rna_2d=rna_2d,
        cyt_labels=cyt_labels,
        dapi_mask=dapi_mask,
        rna_pos_mask=rna_pos_mask,
        voxel_xy_nm=voxel_xy_nm,
        # 2026-05-25 Brian: rna_only parity — expose nucleolus labels so the
        # runner's generic nucleolus-overlay step (runner.py ~L1668) renders
        # <prefix><stem>__nucleolus_overlay.png into the nucleolus_overlay/ dir.
        # None when cfg.nucleolus.enabled is False (overlay step is skipped).
        nucleolus_labels=nucleolus_labels_for_qc,
        # 2026-05-28 Brian: DAPI's autofocus-picked 1-indexed plane (only set
        # when z_mode == "autofocus"; None otherwise). The rna_protein wrapper
        # reads this to lock the antibody channel to the SAME plane.
        dapi_autofocus_z=dapi_autofocus_z,
    )

    return ImageResult(
        image=img_name,
        condition=condition,
        sec_only=sec_only,
        per_image=per_image,
        nuclei=nuclei_df,
        spots=spots_out_df,
        morphology=morph_df,
        thresholds=thresholds,
        qc=qc,
        extra=dict(rna_thr_value=rna_thr_value, dapi_thr_value=dapi_thr_val,
                   n_border_excluded=n_border_excluded),
    )


from . import register_mode

@register_mode("rna_only")
def run(*args, **kwargs):
    return run_one(*args, **kwargs)


# Helper used by the batch-scope pre-pass in runner.py. Loads ONE image,
# segments nuclei (with border exclusion matching the main pass), and returns
# the raw nuclear RNA pixel values as a 1-D numpy array. Mirrors Fiji's
# collect_nuclear_pixel_values_fast() called inside
# run_batch_prescan_for_thresholds() (Coloc_Analysis.py lines 2336-2666):
# uses raw rna2d (NO rolling-ball, NO blur, NO median filter).
def collect_nuclear_rna_pixels(path, *, cfg) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(rna_nuclear_pixels, labels)`` for ONE image.

    Used by the batch runner's pre-pass to pool RNA pixels across the whole
    batch and compute a single median+k*MAD threshold. Bit-equivalent to the
    per-image nuclear pixel extraction inside ``run_one`` — same channel
    resolution, same z-collapse, same segmentation, same border exclusion,
    same ``labels > 0`` mask, same ``astype(np.float64)``.

    The SECOND return value is the FINAL (post-border-exclude) nuclei label
    image. The runner caches it keyed by image path and feeds it back into
    ``run_one`` via ``precomputed_labels=`` so each image is segmented exactly
    ONCE in a ``threshold_scope == 'batch'`` run (avoids a 2x segmentation
    cost with slow backends such as cellpose). On the empty-mask early return
    the labels are still returned so the runner can cache them.
    """
    img = _io.read_image(path)

    one_indexed = bool(cfg.channels.one_indexed)
    def _chan(idx: int) -> int:
        return (idx - 1) if (one_indexed and idx > 0) else idx

    dapi_idx = _chan(cfg.channels.dapi)
    rna_idx = _chan(cfg.channels.rna)
    if dapi_idx < 0 or rna_idx < 0:
        auto = _io.autodetect_channels(img)
        if dapi_idx < 0:
            dapi_idx = auto["dapi"]
        if rna_idx < 0:
            rna_idx = auto["rna"]
    dapi_idx = max(0, min(img.n_channels - 1, dapi_idx))
    rna_idx = max(0, min(img.n_channels - 1, rna_idx))

    z_mode = cfg.z_stack.mode
    z_start = cfg.z_stack.start_slice
    z_end = cfg.z_stack.end_slice
    # 2026-05-22 Brian: per-image z-window override (mirrors run_one). Keeps
    # the batch pixel-threshold pre-scan extracting the SAME pixels run_one
    # will analyze for this image.
    _file_overrides = getattr(cfg.z_stack, "file_overrides", {}) or {}
    _img_name = Path(path).name
    _ovr = _file_overrides.get(_img_name, {}) if _img_name in _file_overrides else {}
    if _ovr:
        if "start_slice" in _ovr:
            z_start = int(_ovr["start_slice"])
        if "end_slice" in _ovr:
            z_end = int(_ovr["end_slice"])
    # 2026-05-25 Brian: per-channel z. When BOTH rna_start_slice and
    # rna_end_slice are set for this image, pool the RNA pixels from a MAXPROJ
    # of the RNA channel over that window — SAME pixels run_one detects spots
    # on. DAPI handling is unchanged.
    rna_start_slice = _ovr.get("rna_start_slice") if _ovr else None
    rna_end_slice = _ovr.get("rna_end_slice") if _ovr else None
    rna_per_channel_z = (rna_start_slice is not None and rna_end_slice is not None)
    if rna_per_channel_z:
        rna_start_slice = int(rna_start_slice)
        rna_end_slice = int(rna_end_slice)
        if rna_start_slice > img.n_z:
            rna_start_slice = 1
        if rna_end_slice > img.n_z:
            rna_end_slice = img.n_z
    if z_start is not None and z_start > img.n_z:
        z_start = 1
    if z_end is not None and z_end > img.n_z:
        z_end = img.n_z

    # 2026-05-24 Brian: autofocus_maxproj — per-image DAPI focus-window
    # detection, then MIP that window for all channels. Replaces the need
    # for per-image file_overrides on datasets with field-to-field focus
    # drift (e.g. BIN1 KO_100x02 needed 15-49 vs default 9-78).
    if z_mode == "autofocus_maxproj":
        (afm_zs, afm_ze), afm_diag, dapi_2d = _io.extract_dapi_focus_window(
            img, dapi_idx,
            metric=cfg.z_stack.focus_metric,
            threshold_frac=float(cfg.z_stack.focus_threshold_frac),
            min_slices=int(cfg.z_stack.focus_window_min_slices),
            max_slices=int(cfg.z_stack.focus_window_max_slices),
            z_start=z_start, z_end=z_end,
            fixed_n_slices=int(getattr(cfg.z_stack, "focus_window_fixed_n_slices", 0)),
            min_intensity_frac_of_peak=float(getattr(cfg.z_stack, "focus_min_intensity_frac_of_peak", 0.0)),
            intensity_weighted=bool(getattr(cfg.z_stack, "autofocus_intensity_weighted", False)),
            central_fraction=float(getattr(cfg.z_stack, "focus_central_fraction", 0.0)),
        )
        if rna_per_channel_z:
            rna_2d = _io.extract_channel(
                img, rna_idx, z_mode="maxproj",
                z_start=rna_start_slice, z_end=rna_end_slice,
            )
            if rna_2d.ndim != 2:
                rna_2d = rna_2d.max(axis=0)
        else:
            rna_2d = _io.extract_channel_in_z_range(
                img, rna_idx,
                z_start_1indexed=afm_zs, z_end_1indexed=afm_ze,
                project="maxproj",
            )
        try:
            from rich.console import Console as _C
            _C().print(
                f"  [dim][autofocus_maxproj] {Path(path).name}: "
                f"focus peak at z={afm_diag['peak_z']+1}, "
                f"window=[{afm_zs},{afm_ze}] "
                f"({afm_ze - afm_zs + 1} slices)[/dim]"
            )
        except Exception:
            pass
    elif z_mode == "autofocus":
        # 2026-05-28 Brian: IDENTICAL z-lock logic to run_one (above). The
        # batch pre-pass pools threshold pixels and caches the nuclei labels;
        # locking RNA to DAPI's autofocus plane here keeps the pooled pixels
        # and the cached segmentation on the SAME plane the spots are detected
        # on in run_one (which uses the same window deterministically). The
        # DAPI-lock takes precedence over rna_per_channel_z under autofocus.
        dapi_autofocus_z, dapi_2d = _io.extract_channel_autofocus_with_idx(
            img, dapi_idx, z_start=z_start, z_end=z_end,
            intensity_weighted=bool(getattr(cfg.z_stack, "autofocus_intensity_weighted", False)),
        )
        rna_2d = _io.extract_channel_at_z(img, rna_idx, z_1indexed=dapi_autofocus_z)
        try:
            from rich.console import Console as _C
            if rna_per_channel_z:
                _C().print(
                    f"  [dim]z-lock (prescan): {Path(path).name} → RNA locked to DAPI "
                    f"plane z={dapi_autofocus_z} (per-channel RNA maxproj override "
                    f"IGNORED under autofocus)[/dim]"
                )
            else:
                _C().print(
                    f"  [dim]z-lock (prescan): {Path(path).name} → all channels @ DAPI "
                    f"plane z={dapi_autofocus_z}[/dim]"
                )
        except Exception:
            pass
    else:
        dapi_2d = _io.extract_channel(img, dapi_idx, z_mode=z_mode, z_start=z_start, z_end=z_end)
        if dapi_2d.ndim != 2:
            dapi_2d = dapi_2d.max(axis=0)
        if rna_per_channel_z:
            rna_2d = _io.extract_channel(
                img, rna_idx, z_mode="maxproj",
                z_start=rna_start_slice, z_end=rna_end_slice,
            )
        else:
            rna_2d = _io.extract_channel(img, rna_idx, z_mode=z_mode, z_start=z_start, z_end=z_end)
        if rna_2d.ndim != 2:
            rna_2d = rna_2d.max(axis=0)

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
        cellpose_downsample_factor=cfg.nuclei.cellpose_downsample_factor,
        cellpose_device=getattr(cfg.nuclei, "cellpose_device", "cpu"),
    )
    labels = _seg.segment_nuclei(dapi_2d, backend=cfg.nuclei.backend, params=seg_params)
    if cfg.nuclei.exclude_border:
        labels = _seg.exclude_border_labels(labels, margin_px=cfg.nuclei.border_margin_px)

    nuc_mask = labels > 0
    if not nuc_mask.any():
        return np.empty(0, dtype=np.float64), labels
    return rna_2d[nuc_mask].astype(np.float64), labels
