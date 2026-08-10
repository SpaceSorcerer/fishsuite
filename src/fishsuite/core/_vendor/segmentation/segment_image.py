#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Segment nuclei from a 2D DAPI image using one of three backends.

Designed as the canonical bridge between the Jython side (which only sees
TIFFs on disk) and the Python deep-learning libraries. The Jython pipeline
saves a DAPI TIFF, calls this script via subprocess, then reads the label
TIFF back in.

Usage
-----
    python -m segmentation.segment_image \\
        --input  /path/to/dapi.tif \\
        --backend stardist \\
        --output /path/to/labels.tif \\
        [--min-area 250] [--max-area 1e12]
        # StarDist-only
        [--prob-threshold 0.5]
        [--nms-threshold 0.4]
        [--n-tiles auto|<int>]
        [--model 2D_versatile_fluo]
        [--stardist-postprocess none|dilate|watershed_otsu|watershed_triangle]
        [--stardist-postprocess-dilate-px 30]
        [--stardist-postprocess-otsu-sigma 2.0]
        [--stardist-postprocess-mask-closing-px 5]
        # Cellpose-only
        [--diameter 0]                     # 0 = auto (slow on CPU)
        [--flow-threshold 0.4]
        [--cellprob-threshold 0.0]
        [--model-type cpsam]               # cellpose 4.x only ships cpsam

Exit codes
----------
    0  success
    2  backend not installed (e.g. asked for stardist but tensorflow missing)
    3  bad input image
    1  any other error
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import tifffile


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------

def segment_otsu(
    img: np.ndarray,
    *,
    gauss_sigma: float = 1.5,
    min_area: int = 250,
    max_area: float = 1e12,
    do_watershed: bool = True,
    threshold_method: str = "otsu",
) -> np.ndarray:
    """Reference implementation of the legacy Otsu + watershed pipeline.

    Mirrors fiji_scripts/Coloc_Core.py: blur -> threshold -> fill holes ->
    optional watershed -> connected components -> area filter. Used for
    head-to-head comparisons against the deep-learning backends. Pure
    scikit-image; no Java required.
    """
    from skimage import filters, morphology, measure, segmentation
    from scipy import ndimage as ndi

    # Coerce to 2D float for filtering
    if img.ndim != 2:
        raise ValueError(f"Otsu backend expects 2D input; got shape {img.shape}")
    arr = img.astype(np.float32)

    # Blur
    if gauss_sigma > 0:
        arr = filters.gaussian(arr, sigma=gauss_sigma, preserve_range=True)

    # Threshold
    if threshold_method == "otsu":
        thresh = filters.threshold_otsu(arr)
    elif threshold_method == "triangle":
        thresh = filters.threshold_triangle(arr)
    elif threshold_method == "li":
        thresh = filters.threshold_li(arr)
    else:
        raise ValueError(f"Unknown threshold method: {threshold_method}")
    binary = arr > thresh
    binary = ndi.binary_fill_holes(binary)

    # Watershed split via distance transform peaks
    if do_watershed:
        distance = ndi.distance_transform_edt(binary)
        # peak_local_max returns coords; use as markers
        from skimage.feature import peak_local_max
        coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=binary)
        markers = np.zeros(distance.shape, dtype=np.int32)
        for i, (y, x) in enumerate(coords, start=1):
            markers[y, x] = i
        labels = segmentation.watershed(-distance, markers, mask=binary)
    else:
        labels = measure.label(binary, connectivity=1)

    # Area filter: drop labels outside [min_area, max_area]
    if min_area > 0 or max_area < 1e18:
        out = np.zeros_like(labels, dtype=np.int32)
        new_id = 0
        for region in measure.regionprops(labels):
            if region.area < min_area or region.area > max_area:
                continue
            new_id += 1
            out[labels == region.label] = new_id
        labels = out

    return labels.astype(np.uint16)


def segment_stardist(
    img: np.ndarray,
    *,
    prob_threshold: float = 0.5,
    nms_threshold: float = 0.4,
    min_area: int = 250,
    model_name: str = "2D_versatile_fluo",
    n_tiles: "int | tuple | None" = None,
    gauss_sigma: float = 0.0,
) -> np.ndarray:
    """StarDist 2D segmentation via the pretrained fluorescent-nuclei model.

    The "2D_versatile_fluo" model was trained on a wide variety of
    fluorescent nuclei (DAPI, Hoechst, etc.) including dense / confluent
    cultures — generally the right default for DAPI images of monolayers.

    Parameters
    ----------
    img : 2D numpy array (any dtype). Internally normalized via percentile
          contrast as recommended by StarDist authors.
    prob_threshold : detector probability threshold; lower = more nuclei.
                     Default 0.5 (StarDist's own default for this model).
    nms_threshold  : non-maximum suppression IoU threshold for overlapping
                     candidate detections. Default 0.4.
    min_area       : labels with fewer pixels are dropped post-hoc.
    model_name     : "2D_versatile_fluo" (DAPI / nuclei), "2D_versatile_he"
                     (H&E histology — not relevant here), "2D_paper_dsb2018",
                     or "2D_demo".
    n_tiles        : how many tiles to split big images into for memory
                     safety (passed straight to StarDist). None = StarDist's
                     own auto. Pass an int (e.g. 4) to force tiling on
                     huge frames. For 2D, an int is broadcast by StarDist
                     to (n, n) over (Y, X).
    gauss_sigma    : Gaussian pre-blur sigma in pixels. 0 = no blur (legacy
                     behaviour). Critical at high magnification (100x @
                     0.065 µm/px): without pre-blur StarDist treats every
                     bright sub-nuclear feature (nucleolus / chromocentre)
                     as a separate nucleus. Sweep results on H9 hESC at
                     100x found sigma=3 with prob=0.5, nms=0.5,
                     min_area=10000 hit Brian's manual count (N=40)
                     exactly. Set 0 at lower magnifications.
    """
    try:
        from csbdeep.utils import normalize
        from stardist.models import StarDist2D
    except ImportError as exc:
        raise SystemExit(
            f"stardist or csbdeep not installed in this Python env: {exc}\n"
            "Install with: pip install stardist"
        )

    if img.ndim != 2:
        raise ValueError(f"StarDist 2D expects 2D input; got shape {img.shape}")

    # Optional Gaussian pre-blur (essential at high magnification — see docstring).
    if gauss_sigma > 0:
        from skimage import filters as _filters
        img = _filters.gaussian(img.astype(np.float32), sigma=gauss_sigma,
                                preserve_range=True)

    # Pretrained model is downloaded + cached on first use to ~/.keras/models/
    model = StarDist2D.from_pretrained(model_name)

    # Normalize to roughly [0, 1] using 1st-99.8th percentile (StarDist convention)
    img_norm = normalize(img, 1.0, 99.8, axis=(0, 1))

    # StarDist accepts n_tiles as int (broadcast) or tuple matching image
    # axes. None lets it pick (no tiling) — fine for 2304x2304 frames.
    predict_kwargs = dict(prob_thresh=prob_threshold, nms_thresh=nms_threshold)
    if n_tiles is not None:
        # int -> (n, n) for 2D; pass tuples through unchanged
        if isinstance(n_tiles, int):
            predict_kwargs["n_tiles"] = (n_tiles, n_tiles)
        else:
            predict_kwargs["n_tiles"] = n_tiles
    labels, _ = model.predict_instances(img_norm, **predict_kwargs)

    # Optional area filter (mirrors the Otsu backend)
    if min_area > 0:
        from skimage import measure
        out = np.zeros_like(labels, dtype=np.int32)
        new_id = 0
        for region in measure.regionprops(labels):
            if region.area < min_area:
                continue
            new_id += 1
            out[labels == region.label] = new_id
        labels = out

    return labels.astype(np.uint16)


# ---------------------------------------------------------------------------
# Cellpose model cache (2026-05-27 PERF).
#
# cpsam is a ~1.2 GB transformer. The pipeline loaded a fresh CellposeModel on
# EVERY segment_cellpose call — in a fishsuite batch run that is one ~1.2 GB
# load per image (9 loads for the H9 floor-500 run; profiled segmentation
# included this reload). Caching ONE model instance per (model_type, device)
# per process removes the redundant reloads AND relieves the per-worker memory
# pressure that caused the documented -p4 OOM (each worker now holds at most
# one cpsam, reused across all its images).
#
# Correctness: the cached model is the SAME object that would otherwise be
# rebuilt with identical constructor args, and CellposeModel.eval is a pure
# forward pass (no internal mutable state that carries between images that
# would change masks). Verified byte-identical masks with caching on vs off.
# The cache is OPT-OUT via env FISHSUITE_NO_MODEL_CACHE=1 (escape hatch; the
# Fiji subprocess path constructs one model per process anyway, so this cache
# is a no-op there — a single image per `python -m segmentation.segment_image`
# invocation means one entry, identical to before).
_CELLPOSE_MODEL_CACHE: dict = {}


def _get_cellpose_model(model_type: str, device: str):
    """Return a cached CellposeModel for (model_type, device), building once."""
    import os as _os
    if _os.environ.get("FISHSUITE_NO_MODEL_CACHE") == "1":
        return _construct_cellpose_model(model_type, device)
    key = (str(model_type), str(device).lower())
    m = _CELLPOSE_MODEL_CACHE.get(key)
    if m is None:
        m = _construct_cellpose_model(model_type, device)
        _CELLPOSE_MODEL_CACHE[key] = m
    return m


def _construct_cellpose_model(model_type: str, device: str):
    """Build a fresh CellposeModel (CPU or DirectML). No caching here."""
    from cellpose import models
    _dev = str(device or "cpu").lower()
    if _dev in ("directml", "dml"):
        try:
            return _build_cellpose_directml_model(model_type)
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write(
                f"WARNING: cellpose_device='directml' requested but DirectML "
                f"setup failed ({exc}); falling back to CPU.\n"
            )
            return models.CellposeModel(gpu=False, pretrained_model=model_type)
    return models.CellposeModel(gpu=False, pretrained_model=model_type)


def _build_cellpose_directml_model(model_type: str):
    """Build a cellpose CellposeModel running its transformer on a torch-directml
    GPU (Brian's AMD RX 6750 XT — no CUDA), with the two patches required for
    cpsam to run end-to-end on DirectML. Returns the model with its net on the
    DirectML device.

    OPT-IN only — reached solely when the caller passes device="directml".
    The default CPU path never imports torch_directml, so the production CPU
    env (which lacks torch-directml) is completely unaffected.

    Patches (verified working 2026-05-27, see _gpu_accel_investigation/
    DIRECTML_PATCH_ATTEMPT.md — byte-for-byte equivalent masks vs CPU, ~33x):
      (A) use_bfloat16=False: cpsam ships bf16 weights; DirectML has no
          BFloat16 kernel (hard abort). fp32 is fully supported.
      (B) Build the model on CPU, then move ONLY the net to the DirectML
          device (the expensive transformer forward). The cheap flow-dynamics
          / mask-reconstruction step uses torch.sparse_coo_tensor, which has
          no DirectML kernel, so we monkeypatch the single cellpose entry
          point (dynamics.resize_and_compute_masks) to force JUST that step
          back to CPU — mirrors cellpose's own Apple-MPS escape hatch. The
          arrays reaching that function are already numpy host arrays, so no
          extra device<->host copy is added.
    """
    import torch
    import torch_directml as tdml
    from cellpose import models, dynamics

    # Patch (B): force flow-dynamics to CPU (idempotent module-level guard).
    if not getattr(dynamics, "_dml_cpu_patched", False):
        _orig_rcm = dynamics.resize_and_compute_masks

        def _rcm_cpu(*a, **k):
            k["device"] = torch.device("cpu")
            return _orig_rcm(*a, **k)

        dynamics.resize_and_compute_masks = _rcm_cpu
        dynamics._dml_cpu_patched = True

    # Patch (A): fp32 weights, build on CPU. Some cellpose 4.x builds expose
    # use_bfloat16 on the constructor; guard for forward/back-compat.
    dev = tdml.device()
    try:
        model = models.CellposeModel(
            gpu=False, pretrained_model=model_type, use_bfloat16=False,
        )
    except TypeError:
        # Older/newer signature without use_bfloat16 — fall back and coerce
        # the net to fp32 manually so DirectML doesn't hit the bf16 abort.
        model = models.CellposeModel(gpu=False, pretrained_model=model_type)
        try:
            model.net = model.net.float()
        except Exception:
            pass
    # Patch (B) cont.: move ONLY the transformer net to the GPU.
    model.net.to(dev)
    model.device = dev
    return model


def segment_cellpose(
    img: np.ndarray,
    *,
    diameter: float = 0.0,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    min_area: int = 250,
    model_type: str = "cpsam",
    device: str = "cpu",
) -> np.ndarray:
    """Cellpose 2D segmentation via a pretrained model.

    Cellpose 4.x ships a single transformer generalist 'cpsam' (~1.2 GB);
    the older 'nuclei' / 'cyto' / 'cyto2' weights from cellpose 2/3 are
    not loadable in 4.x and will fall back to cpsam silently. We pin
    'cpsam' as the only valid choice here so the GUI never lies to the
    user. Slower than StarDist on CPU; use --diameter to make it usable.

    Parameters
    ----------
    img : 2D numpy array.
    diameter : expected nucleus diameter in pixels. 0 = auto-estimate. For
               H9 DAPI at ~0.065 µm/px and ~13 µm nuclei, ~200 px is typical;
               passing 0 (auto) is much slower on CPU.
    flow_threshold : higher = more permissive on shape. 0.4 is the default.
    cellprob_threshold : higher = fewer cells. 0.0 default.
    min_area : labels with fewer pixels are dropped post-hoc.
    model_type : cellpose 4.x supports only 'cpsam'. Other strings are
                 accepted for forward-compat but cellpose internally
                 maps unknown names to cpsam.
    device : "cpu" (default) = legacy CPU path, byte-for-byte unchanged.
             "directml" = OPT-IN GPU acceleration via torch-directml (AMD
             GPU, no CUDA). Only valid in an env that has torch-directml
             installed. The default never touches it, so the production
             CPU env is unaffected.
    """
    try:
        from cellpose import models
    except ImportError as exc:
        raise SystemExit(
            f"cellpose not installed in this Python env: {exc}\n"
            "Install with: pip install cellpose"
        )

    if img.ndim != 2:
        raise ValueError(f"Cellpose 2D expects 2D input; got shape {img.shape}")

    # Cellpose 4.x API: CellposeModel; the only model is 'cpsam', a ~1.2GB
    # transformer generalist. Pass via pretrained_model=. Note: cellpose
    # 4.x's eval() does NOT have a tile_norm parameter (verified against
    # cellpose 4.1.1 signature 2026-05); the GUI's CELLPOSE_TILE_NORM
    # field was removed because there is no equivalent kwarg to wire it
    # to. tile_overlap / bsize exist but control geometric tiling, not
    # tile-wise normalization.
    # 2026-05-27 PERF: fetch a process-cached model (built once per
    # (model_type, device)) instead of reloading the ~1.2 GB cpsam every call.
    # Default CPU path is byte-for-byte identical to the pre-2026-05-27
    # behavior — same constructor args, same eval — just without the reload.
    # The directml fallback-to-CPU-on-failure behavior is preserved inside
    # _construct_cellpose_model.
    model = _get_cellpose_model(model_type, device)
    eval_out = model.eval(
        img,
        diameter=diameter if diameter > 0 else None,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )
    masks = eval_out[0] if isinstance(eval_out, tuple) else eval_out

    if min_area > 0:
        from skimage import measure
        out = np.zeros_like(masks, dtype=np.int32)
        new_id = 0
        for region in measure.regionprops(masks):
            if region.area < min_area:
                continue
            new_id += 1
            out[masks == region.label] = new_id
        masks = out

    return masks.astype(np.uint16)


# ---------------------------------------------------------------------------
# StarDist post-processing
# ---------------------------------------------------------------------------

POSTPROCESS_MODES = ("none", "dilate", "watershed_otsu", "watershed_triangle")


def _filter_min_area(labels: np.ndarray, min_area: int) -> np.ndarray:
    """Drop labels smaller than min_area and renumber 1..N contiguously."""
    if min_area <= 0:
        return labels
    from skimage import measure
    out = np.zeros_like(labels, dtype=np.int32)
    new_id = 0
    for region in measure.regionprops(labels):
        if region.area < min_area:
            continue
        new_id += 1
        out[labels == region.label] = new_id
    return out


def postprocess_stardist_labels(
    labels: np.ndarray,
    dapi: np.ndarray,
    *,
    mode: str = "none",
    dilate_px: int = 30,
    otsu_sigma: float = 2.0,
    mask_closing_px: int = 5,
    min_area: int = 0,
) -> np.ndarray:
    """Refine StarDist labels by expanding them to the true nuclear extent.

    StarDist's star-convex contours can hug the bright DAPI core too tightly
    on confluent monolayers at high magnification (100x @ 0.065 µm/px on H9
    hESC), leaving a diffuse rim around each nucleus unsegmented. The four
    modes below trade off speed vs. accuracy:

      - "none"               : passthrough, no post-processing.
      - "dilate"             : uniformly expand each label by `dilate_px`
                                pixels via `skimage.segmentation.expand_labels`.
                                Fastest; ignores image content. Use when
                                seeds are already well-placed and you just
                                need a uniform rim.
      - "watershed_otsu"     : compute an Otsu mask on Gaussian-blurred DAPI,
                                close it by `mask_closing_px`, and use the
                                StarDist labels as seeds for a watershed
                                bounded by that mask. Result: each seed
                                grows until it hits the mask boundary or a
                                neighbouring seed. RECOMMENDED for 100x H9.
      - "watershed_triangle" : same as watershed_otsu but uses
                                `threshold_triangle` (more permissive on
                                dim / sparsely-stained DAPI). Use if Otsu
                                under-masks the field.

    After the expansion step the area filter (`min_area`) is re-applied so
    that post-processed labels which grow above the threshold get retained
    even if the underlying StarDist seed was below it.

    Parameters
    ----------
    labels : 2D int label image from StarDist.
    dapi   : 2D DAPI intensity image (any dtype). Used only by the
             watershed modes; ignored by "none" / "dilate".
    mode   : one of POSTPROCESS_MODES.
    dilate_px : pixels for uniform dilation (mode="dilate").
    otsu_sigma : Gaussian-blur sigma for the watershed mask (px).
    mask_closing_px : binary closing radius for the watershed mask (px).
                      Use 0 to skip the closing step.
    min_area : drop labels smaller than this AFTER post-processing.
               0 = no re-filter. Should match the segmenter's --min-area.

    Returns
    -------
    Renumbered uint16 label image of the same shape as `labels`.
    """
    if mode not in POSTPROCESS_MODES:
        raise ValueError(
            "Unknown postprocess mode %r. Choices: %s" % (mode, POSTPROCESS_MODES))

    if mode == "none" or int(labels.max()) == 0:
        return labels.astype(np.uint16)

    from skimage import filters, morphology, segmentation
    from scipy import ndimage as ndi

    if mode == "dilate":
        if dilate_px <= 0:
            return labels.astype(np.uint16)
        out = segmentation.expand_labels(labels, distance=int(dilate_px))
        out = _filter_min_area(out, int(min_area))
        return out.astype(np.uint16)

    # Watershed branches: need an intensity mask.
    if dapi is None:
        raise ValueError("watershed post-processing requires the DAPI image")
    if dapi.shape != labels.shape:
        raise ValueError(
            "DAPI shape %s does not match labels shape %s" % (dapi.shape, labels.shape))

    arr = dapi.astype(np.float32)
    if otsu_sigma > 0:
        arr = filters.gaussian(arr, sigma=float(otsu_sigma), preserve_range=True)

    if mode == "watershed_otsu":
        thresh = filters.threshold_otsu(arr)
    else:  # "watershed_triangle"
        thresh = filters.threshold_triangle(arr)
    mask = arr > thresh

    # Drop tiny mask debris (matches the sweep-validated F: segmentation_refine.py).
    mask = morphology.remove_small_objects(mask, min_size=500)
    if mask_closing_px and int(mask_closing_px) > 0:
        mask = morphology.binary_closing(
            mask, footprint=morphology.disk(int(mask_closing_px)))

    # Ensure every seed pixel lies inside the mask so watershed actually
    # propagates outward instead of dropping the marker.
    mask = mask | (labels > 0)

    # Sign convention matches segmentation_refine.py: negative blurred DAPI
    # makes nuclear interiors deep basins so watershed flows outward.
    distance = -arr
    out = segmentation.watershed(distance, markers=labels, mask=mask)
    out = _filter_min_area(out, int(min_area))
    return out.astype(np.uint16)


# ---------------------------------------------------------------------------
# Backend dispatch
# ---------------------------------------------------------------------------

BACKENDS = {
    "otsu": segment_otsu,
    "stardist": segment_stardist,
    "cellpose": segment_cellpose,
}


def run_backend(
    backend: str,
    img: np.ndarray,
    *,
    min_area: int,
    max_area: float,
    # StarDist
    prob_threshold: float,
    nms_threshold: float = 0.4,
    n_tiles: "int | None" = None,
    stardist_model: str = "2D_versatile_fluo",
    stardist_gauss_sigma: float = 0.0,
    stardist_postprocess: str = "none",
    stardist_postprocess_dilate_px: int = 30,
    stardist_postprocess_otsu_sigma: float = 2.0,
    stardist_postprocess_mask_closing_px: int = 5,
    # Cellpose
    diameter: float,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    cellpose_model_type: str = "cpsam",
    cellpose_device: str = "cpu",
) -> np.ndarray:
    """Dispatch to the named backend, passing only the kwargs each one accepts."""
    if backend == "otsu":
        return segment_otsu(img, min_area=min_area, max_area=max_area)
    if backend == "stardist":
        labels = segment_stardist(
            img,
            prob_threshold=prob_threshold,
            nms_threshold=nms_threshold,
            n_tiles=n_tiles,
            model_name=stardist_model,
            min_area=min_area,
            gauss_sigma=stardist_gauss_sigma,
        )
        if stardist_postprocess and stardist_postprocess != "none":
            try:
                labels = postprocess_stardist_labels(
                    labels, img,
                    mode=stardist_postprocess,
                    dilate_px=int(stardist_postprocess_dilate_px),
                    otsu_sigma=float(stardist_postprocess_otsu_sigma),
                    mask_closing_px=int(stardist_postprocess_mask_closing_px),
                    min_area=int(min_area),
                )
            except Exception as _pp_exc:
                # Post-processing is a nice-to-have; never let it crash the
                # whole subprocess. If Otsu fails on a degenerate channel
                # (wrong CH_DAPI, all-zero, saturated) we want StarDist's
                # raw labels so the run still produces a roi.zip and the
                # Jython side can flag the image rather than abort.
                sys.stderr.write(
                    "WARNING: stardist postprocess mode=%r failed (%s); "
                    "falling back to raw StarDist labels.\n"
                    % (stardist_postprocess, _pp_exc)
                )
        return labels
    if backend == "cellpose":
        return segment_cellpose(
            img,
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            model_type=cellpose_model_type,
            min_area=min_area,
            device=cellpose_device,
        )
    raise ValueError(f"Unknown backend: {backend!r}. Available: {sorted(BACKENDS)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", required=True, help="Input DAPI TIFF (2D).")
    parser.add_argument("--output", required=True, help="Output label TIFF (16-bit).")
    parser.add_argument(
        "--backend", required=True, choices=sorted(BACKENDS),
        help="Segmentation backend to use.",
    )
    parser.add_argument("--min-area", type=int, default=250,
                        help="Drop labels smaller than this many pixels.")
    parser.add_argument("--max-area", type=float, default=1e12,
                        help="Drop labels larger than this many pixels.")
    parser.add_argument("--prob-threshold", type=float, default=0.5,
                        help="(stardist only) detector probability threshold.")
    parser.add_argument("--nms-threshold", type=float, default=0.4,
                        help="(stardist only) non-maximum suppression IoU threshold "
                             "for overlapping detections. Default 0.4.")
    parser.add_argument("--n-tiles", type=str, default="auto",
                        help="(stardist only) tile count per axis. 'auto' (default) "
                             "lets StarDist pick. Pass an int (e.g. 4) on huge images "
                             "to bound memory: it will be broadcast to (n, n) for 2D.")
    parser.add_argument("--model", type=str, default="2D_versatile_fluo",
                        help="(stardist only) pretrained model name. Default "
                             "'2D_versatile_fluo' (DAPI / fluorescent nuclei).")
    parser.add_argument("--stardist-gauss-sigma", type=float, default=0.0,
                        help="(stardist only) Gaussian pre-blur sigma in pixels. "
                             "0 = no blur. Use ~3 at 100x @ 0.065 um/px so StarDist "
                             "doesn't fragment each nucleus into nucleolus-sized "
                             "pieces. Sweep-validated values for H9 100x: sigma=3 "
                             "with prob=0.5, nms=0.5, min_area=10000.")
    parser.add_argument("--stardist-postprocess", type=str, default="none",
                        choices=list(POSTPROCESS_MODES),
                        help="(stardist only) post-process StarDist labels to "
                             "expand them outward. 'none' (default) keeps the raw "
                             "star-convex contours. 'dilate' uniformly expands by "
                             "--stardist-postprocess-dilate-px. 'watershed_otsu' "
                             "uses StarDist labels as seeds for an Otsu-bounded "
                             "watershed — recommended for 100x H9 monolayers; "
                             "expands each seed to the actual nuclear extent. "
                             "'watershed_triangle' same but uses Triangle threshold "
                             "(more permissive for dim DAPI).")
    parser.add_argument("--stardist-postprocess-dilate-px", type=int, default=30,
                        help="(stardist + postprocess=dilate only) uniform "
                             "dilation distance in pixels via expand_labels. "
                             "Default 30. Useful range 5-80.")
    parser.add_argument("--stardist-postprocess-otsu-sigma", type=float, default=2.0,
                        help="(stardist + postprocess=watershed_* only) Gaussian "
                             "blur sigma applied to DAPI BEFORE the Otsu / Triangle "
                             "threshold used as the watershed mask. Default 2.0.")
    parser.add_argument("--stardist-postprocess-mask-closing-px", type=int, default=5,
                        help="(stardist + postprocess=watershed_* only) binary "
                             "closing radius (px) applied to the watershed mask "
                             "to fill small gaps. Default 5. Set 0 to skip.")
    parser.add_argument("--diameter", type=float, default=0.0,
                        help="(cellpose only) expected nucleus diameter px; 0 = auto.")
    parser.add_argument("--flow-threshold", type=float, default=0.4,
                        help="(cellpose only) higher = more permissive on cell shape. "
                             "Default 0.4.")
    parser.add_argument("--cellprob-threshold", type=float, default=0.0,
                        help="(cellpose only) higher = fewer cells (stricter). "
                             "Default 0.0.")
    parser.add_argument("--model-type", type=str, default="cpsam",
                        help="(cellpose only) pretrained model name. Cellpose 4.x "
                             "ships only 'cpsam'.")
    parser.add_argument("--cellpose-device", type=str, default="cpu",
                        choices=["cpu", "directml"],
                        help="(cellpose only) compute device. 'cpu' (default) = "
                             "legacy CPU path. 'directml' = OPT-IN GPU via "
                             "torch-directml (AMD GPU, no CUDA); requires "
                             "torch-directml in the env.")
    args = parser.parse_args()

    # Parse "auto" -> None for n_tiles; else int
    n_tiles_val: "int | None"
    if args.n_tiles is None or str(args.n_tiles).strip().lower() in ("", "auto", "none"):
        n_tiles_val = None
    else:
        try:
            n_tiles_val = int(args.n_tiles)
        except (TypeError, ValueError):
            sys.stderr.write(
                f"WARNING: --n-tiles={args.n_tiles!r} is not 'auto' or an int; "
                f"falling back to auto.\n"
            )
            n_tiles_val = None

    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.is_file():
        sys.stderr.write(f"ERROR: input not found: {in_path}\n")
        return 3

    img = tifffile.imread(in_path)
    if img.ndim == 3 and img.shape[0] == 1:
        img = img[0]
    if img.ndim != 2:
        sys.stderr.write(f"ERROR: expected 2D input; got shape {img.shape}\n")
        return 3

    t0 = time.time()
    try:
        labels = run_backend(
            args.backend, img,
            min_area=args.min_area, max_area=args.max_area,
            # StarDist
            prob_threshold=args.prob_threshold,
            nms_threshold=args.nms_threshold,
            n_tiles=n_tiles_val,
            stardist_model=args.model,
            stardist_gauss_sigma=args.stardist_gauss_sigma,
            stardist_postprocess=args.stardist_postprocess,
            stardist_postprocess_dilate_px=args.stardist_postprocess_dilate_px,
            stardist_postprocess_otsu_sigma=args.stardist_postprocess_otsu_sigma,
            stardist_postprocess_mask_closing_px=args.stardist_postprocess_mask_closing_px,
            # Cellpose
            diameter=args.diameter,
            flow_threshold=args.flow_threshold,
            cellprob_threshold=args.cellprob_threshold,
            cellpose_model_type=args.model_type,
            cellpose_device=args.cellpose_device,
        )
    except SystemExit as e:
        # Backend missing — report and use a distinct exit code
        sys.stderr.write(str(e) + "\n")
        return 2
    elapsed = time.time() - t0

    n_labels = int(labels.max())
    print(f"backend={args.backend}  nuclei={n_labels}  elapsed={elapsed:.2f}s  out={out_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(out_path, labels.astype(np.uint16))

    # Also write an ImageJ ROI archive next to the label TIFF so the Jython
    # side can load the per-nucleus ROIs natively via RoiManager.runCommand
    # ("Open", "<path>.zip"). One contour per label; coordinates in image px.
    # Failing here is non-fatal (the label TIFF is still authoritative) but
    # we surface the warning so callers know.
    roi_path = out_path.with_suffix(".roi.zip")
    try:
        write_imagej_roi_zip(labels, roi_path)
        print(f"  Wrote ROI archive: {roi_path}")
    except Exception as e:
        sys.stderr.write(f"  WARNING: could not write ROI archive ({e}); label TIFF still saved.\n")
    return 0


def write_imagej_roi_zip(labels: np.ndarray, out_zip: Path) -> None:
    """Convert a 2D label image to one ImageJ FreehandRoi per label and save
    as a ROI .zip archive readable by Fiji's RoiManager.

    Implementation:
        - For each label value 1..max, find the largest connected component
          (handles rare disconnected fragments by keeping only the largest).
        - Trace its outer contour with skimage.measure.find_contours.
        - Build a roifile.ImagejRoi via frompoints(...).
        - Append all to a single .zip via roifile.roiwrite.

    Notes:
        - find_contours returns (row, col) = (y, x). roifile expects (x, y).
        - Subpixel contour coords are kept as floats (roifile supports this).
    """
    from roifile import ImagejRoi, roiwrite
    from skimage import measure

    # roiwrite appends to existing ZIPs, so make sure we start fresh
    if out_zip.exists():
        out_zip.unlink()

    rois = []
    n_regions = 0
    for region in measure.regionprops(labels):
        n_regions += 1
        # Slice to the label's bounding box, build a binary submask
        minr, minc, maxr, maxc = region.bbox
        submask = (labels[minr:maxr, minc:maxc] == region.label).astype(np.uint8)
        # find_contours wants padding so contours close at borders
        padded = np.pad(submask, 1, mode="constant", constant_values=0)
        contours = measure.find_contours(padded, 0.5)
        if not contours:
            continue
        # Largest contour by length
        contour = max(contours, key=lambda c: len(c))
        # Map back to original image coords; (row,col) -> (x,y) and remove pad offset
        pts = np.column_stack([contour[:, 1] - 1 + minc, contour[:, 0] - 1 + minr])
        roi = ImagejRoi.frompoints(pts, name=f"label_{region.label:04d}")
        rois.append(roi)

    # Use ASCII -> instead of Unicode arrow: print() goes to Windows
    # cp1252 stdout by default, which raises UnicodeEncodeError on
    # non-Latin-1 chars. The exception escapes out of the function and
    # gets caught by the caller's try/except as "could not write ROI
    # archive", silently aborting before roiwrite even runs.
    print(f"  ROI archive: {n_regions} labeled regions -> {len(rois)} ROIs")
    if rois:
        roiwrite(out_zip, rois)
    else:
        # Always write the file so the Jython side gets a clear "0 ROIs"
        # signal instead of "file not found". Empty zipfile is still a
        # valid zip — RoiManager.runCommand("Open", path) loads it as
        # zero ROIs. Beats silent failure mode.
        import zipfile
        with zipfile.ZipFile(out_zip, "w"):
            pass
        sys.stderr.write(
            f"  WARNING: 0 ROIs to write — segmentation found no nuclei. "
            f"Wrote empty archive at {out_zip}.\n"
        )


if __name__ == "__main__":
    sys.exit(main())
