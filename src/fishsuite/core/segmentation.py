"""Nuclear segmentation wrapper.

Imports the existing Fiji-pipeline segmentation routines verbatim — no
algorithm reinvention. Single entry point: ``segment_nuclei``.

Also provides ``_smooth_label_boundaries`` — a per-label morphological
post-processing step (closing + opening with a disk SE) that rounds off
the sharp corners introduced by StarDist's star-convex polygon
predictions where neighboring instances meet.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np


# Inject the Fiji pipeline 'python/' folder onto sys.path so we can import
# its segmentation module. The package never modifies that source — read-only
# reuse.
_FIJI_PY = Path(r"F:\Image Analysis Work\image-analysis-pipeline\python")
if str(_FIJI_PY) not in sys.path:
    sys.path.insert(0, str(_FIJI_PY))


def _smooth_label_boundaries(labels: np.ndarray, radius: int) -> np.ndarray:
    """Round per-label boundaries via morphological closing + opening.

    For each label k in ``labels`` (background = 0):
        1) build a binary mask of that label,
        2) apply ``binary_closing`` then ``binary_opening`` with a disk SE
           of radius ``radius`` px,
        3) write k into the output where the smoothed mask is True AND the
           original pixel was either background or already labelled k
           (this prevents label bleed into a neighbor's territory).

    This is intentionally a per-label loop — O(N * mask_area * 2) where N
    is the number of labels, which is fine for the ~100 labels per H9
    field-of-view we see in practice. Pixels that were assigned to a
    different label originally are preserved exactly. Pixels that were
    background but are NOT claimed by any smoothed mask remain background;
    if two smoothed masks would have overlapped on a background pixel the
    first one in label order wins (acceptable — collisions are very rare
    at radii 3-8 px after watershed splitting has already done its job).

    Parameters
    ----------
    labels : 2D integer label image (background = 0).
    radius : disk SE radius in pixels. ``<= 0`` disables (returns input).
    """
    if radius is None or int(radius) <= 0:
        return labels
    if labels.ndim != 2:
        return labels
    from skimage.morphology import disk, binary_closing, binary_opening

    r = int(radius)
    se = disk(r)
    out = np.zeros_like(labels)
    # np.unique is sorted ascending; background (0) is skipped in the loop.
    uniq = np.unique(labels)
    for k in uniq:
        if k == 0:
            continue
        mask = labels == k
        if not mask.any():
            continue
        smoothed = binary_opening(binary_closing(mask, se), se)
        # Only claim pixels that were either background OR already this label.
        # This is the key anti-bleed constraint: a neighbor's pixels are
        # preserved exactly as the watershed assigned them.
        valid = smoothed & ((labels == 0) | (labels == k))
        # If two smoothed masks would collide on previously-background
        # pixels, the first-seen label wins (deterministic via sorted uniq).
        valid &= out == 0
        out[valid] = k
    # Preserve labels that smoothing erased entirely (e.g. tiny / thin masks
    # that don't survive opening with a large disk) by falling back to the
    # original assignment for those labels only.
    surviving = set(np.unique(out).tolist()) - {0}
    missing = [int(k) for k in uniq if k != 0 and int(k) not in surviving]
    if missing:
        for k in missing:
            # Restore original pixels of that label that are not now claimed
            # by some other smoothed label.
            restore = (labels == k) & (out == 0)
            out[restore] = k
    return out


def segment_nuclei(
    dapi_2d: np.ndarray,
    *,
    backend: str = "stardist",
    params: Dict[str, Any] | None = None,
) -> np.ndarray:
    """Segment nuclei in a 2D DAPI image.

    Parameters
    ----------
    dapi_2d : 2D float / int array.
    backend : "stardist" | "cellpose" | "otsu"
    params : dict with backend-specific knobs:
        min_area, max_area, prob_threshold, nms_threshold, n_tiles,
        stardist_model, stardist_gauss_sigma, stardist_postprocess,
        stardist_postprocess_dilate_px, stardist_postprocess_otsu_sigma,
        stardist_postprocess_mask_closing_px,
        label_smoothing_radius_px,
        diameter, flow_threshold, cellprob_threshold, cellpose_model_type
    """
    from segmentation.segment_image import run_backend
    p = dict(params or {})
    # 2026-05-13: separate the AUTHORITATIVE min/max area filter from the
    # backend's internal filter. The backend uses a coarse floor (1/2 of
    # user's value, min 250) so it doesn't drop labels that label smoothing
    # would otherwise round UP above the user's threshold. The final filter
    # is applied AFTER smoothing below — see Brian's Run R2 regression where
    # backend-side min_area=12000 dropped 305 labels that smoothing would
    # have lifted into compliance.
    _user_min_area = int(p.get("min_area", 250))
    _user_max_area = float(p.get("max_area", 1e12))
    _backend_min_area = max(250, _user_min_area // 2)
    kwargs = dict(
        min_area=_backend_min_area,
        max_area=_user_max_area,
        prob_threshold=float(p.get("prob_threshold", 0.5)),
        nms_threshold=float(p.get("nms_threshold", 0.4)),
        n_tiles=p.get("n_tiles"),
        stardist_model=str(p.get("stardist_model", "2D_versatile_fluo")),
        stardist_gauss_sigma=float(p.get("stardist_gauss_sigma", 0.0)),
        stardist_postprocess=str(p.get("stardist_postprocess", "none")),
        stardist_postprocess_dilate_px=int(p.get("stardist_postprocess_dilate_px", 30)),
        stardist_postprocess_otsu_sigma=float(p.get("stardist_postprocess_otsu_sigma", 2.0)),
        stardist_postprocess_mask_closing_px=int(p.get("stardist_postprocess_mask_closing_px", 5)),
        diameter=float(p.get("diameter", 0.0)),
        flow_threshold=float(p.get("flow_threshold", 0.4)),
        cellprob_threshold=float(p.get("cellprob_threshold", 0.0)),
        cellpose_model_type=str(p.get("cellpose_model_type", "cpsam")),
    )
    labels = run_backend(backend, dapi_2d, **kwargs)
    # Per-label boundary smoothing AFTER backend postprocess (watershed /
    # dilate / none / closing). Default radius 0 = disabled = current
    # behavior. Recommended 3-7 px to round off star-convex artifacts that
    # cause "sharp angle" splits between adjacent StarDist predictions.
    smooth_r = int(p.get("label_smoothing_radius_px", 0))
    if smooth_r > 0:
        labels = _smooth_label_boundaries(labels, smooth_r)
    # AUTHORITATIVE area filter applied AFTER smoothing so smoothing can
    # round labels up into compliance (vs the backend's coarse pre-smoothing
    # floor at _backend_min_area). Drop labels whose final area is outside
    # [_user_min_area, _user_max_area].
    if _user_min_area > _backend_min_area or _user_max_area < 1e12:
        from scipy.ndimage import sum as _ndi_sum
        _label_ids = np.unique(labels)
        _label_ids = _label_ids[_label_ids != 0]
        if len(_label_ids) > 0:
            _areas = _ndi_sum(np.ones_like(labels), labels, _label_ids)
            _bad = _label_ids[(_areas < _user_min_area) | (_areas > _user_max_area)]
            if len(_bad) > 0:
                _mask = np.isin(labels, _bad)
                labels = labels.copy()
                labels[_mask] = 0
    return labels


def exclude_border_labels(labels: np.ndarray, margin_px: int = 5) -> np.ndarray:
    """Drop any label touching the image border (within ``margin_px``)."""
    from skimage import measure
    if margin_px <= 0:
        margin_px = 1
    h, w = labels.shape
    out = np.zeros_like(labels, dtype=np.int32)
    new_id = 0
    border_drop = 0
    for region in measure.regionprops(labels):
        y0, x0, y1, x1 = region.bbox
        if (
            y0 < margin_px or x0 < margin_px
            or y1 > h - margin_px or x1 > w - margin_px
        ):
            border_drop += 1
            continue
        new_id += 1
        out[labels == region.label] = new_id
    return out.astype(np.uint16)
