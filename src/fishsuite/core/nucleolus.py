"""Nucleolus + chromatin metrics from DAPI signal.

Detects DAPI-low subnuclear regions (nucleoli) and computes per-nucleus
chromatin texture metrics. Pure functions — designed to be called
optionally after nuclear segmentation, with no dependencies on the rest
of the pipeline.

Biology rationale (Brian 2026-05-21):
  - Nucleoli appear as DAPI-low islands inside nuclei (the dense rRNA
    machinery excludes DNA). Useful as a fourth compartment alongside
    nucleus / cytoplasm — splicing/transcription spots inside vs outside
    the nucleolus tell different stories.
  - Heterochromatin (densely packed DNA) shows up as bright DAPI puncta.
    Chromatin texture (CV, fraction-above-threshold) per nucleus may
    correlate with transcriptional state.

Public API:
  - detect_nucleoli(nuc_labels, dapi_2d, **params) -> label map
  - chromatin_metrics_per_nucleus(nuc_labels, dapi_2d) -> DataFrame
  - classify_spots_by_subnuclear_region(spots, nuc_labels, nucleolus_labels)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class NucleolusParams:
    """Tuning knobs for nucleolus detection."""
    # Within-nucleus percentile that defines the DAPI-low threshold.
    # 25 = bottom quartile of nuclear DAPI counts as nucleolus candidate.
    intra_nuclear_percentile: float = 25.0
    # Minimum nucleolus area in microns² (~1 µm² typical lower bound).
    min_area_um2: float = 1.0
    # Maximum nucleolus area as a FRACTION of the parent nucleus area.
    # Caps the search so we don't accept a "nucleolus" that's the whole
    # nucleus (would happen for very dim nuclei).
    max_area_frac_of_nucleus: float = 0.6
    # Morphological closing (in pixels) to smooth nucleolus boundaries
    # before final mask. 0 disables.
    closing_radius_px: int = 2
    # 2026-05-22 Brian: minimum distance (in pixels) of nucleolus pixels
    # from the nucleus border. Nucleoli are typically central; this
    # rejects DAPI-low regions touching the edge (boundary artifacts,
    # peripheral chromatin rings). 5 px ≈ 0.65 µm at 0.13 µm/px.
    min_border_distance_px: int = 5


def detect_nucleoli(
    nuc_labels: np.ndarray,
    dapi_2d: np.ndarray,
    pixel_size_um: float,
    params: Optional[NucleolusParams] = None,
) -> np.ndarray:
    """Return a label map of nucleoli (one integer label per nucleolus).

    Labels match the parent nucleus_id where possible. If a single nucleus
    contains multiple disconnected nucleoli they share the parent label
    (caller can re-label per-nucleolus if needed).

    Args:
        nuc_labels: 2D integer array, 0 = background, >0 = nucleus IDs
        dapi_2d: 2D DAPI intensity image, same shape as nuc_labels
        pixel_size_um: physical pixel size (used to convert min_area_um2)
        params: NucleolusParams (uses defaults if None)

    Returns:
        2D integer array — 0 = not nucleolus, n = belongs to nucleus n's
        nucleolus.
    """
    if params is None:
        params = NucleolusParams()
    if nuc_labels.shape != dapi_2d.shape:
        raise ValueError(
            f"nuc_labels {nuc_labels.shape} and dapi_2d {dapi_2d.shape} must match"
        )

    out = np.zeros_like(nuc_labels, dtype=np.int32)
    min_area_px = max(1, int(round(params.min_area_um2 / (pixel_size_um ** 2))))

    # Iterate per-nucleus to compute an intra-nuclear threshold (nucleus
    # boundaries differ a lot in average DAPI intensity).
    unique_ids = np.unique(nuc_labels)
    unique_ids = unique_ids[unique_ids > 0]

    # Optional closing structure
    if params.closing_radius_px > 0:
        try:
            from skimage.morphology import disk, binary_closing
            se = disk(params.closing_radius_px)
        except Exception:
            se = None
    else:
        se = None

    # 2026-05-22 Brian: erode the nucleus mask by min_border_distance_px
    # so nucleolus candidates can ONLY arise away from the nuclear edge.
    # Real nucleoli are typically central; edge DAPI-low regions are
    # usually boundary artifacts or adjacent-cell bleed-through.
    border_distance = int(getattr(params, "min_border_distance_px", 0) or 0)

    # 2026-05-27 PERF: bounding-box crop per nucleus. Previously every
    # per-nucleus op (mask build, threshold, candidate AND, connected
    # components, per-component mask) ran on the FULL 2304x2304 frame inside
    # a per-nucleus loop -> O(N_nuclei * image * N_components). Profiled at
    # ~527 s (CPU) / ~572 s (DML env) for an 88-nucleus image = 68-87% of
    # run_one. The border-erosion was also a whole-image binary_erosion done
    # once up front. We now compute each nucleus's bounding box via
    # scipy.ndimage.find_objects and do ALL work inside that small ROI; the
    # border erosion is done per-ROI on the cropped nucleus mask. Every value
    # (the intra-nuclear percentile, the DAPI<=thr candidate, the connected
    # components, the size filters, the closing) is computed over the EXACT
    # same pixels as before — only the array sizes shrink — so the output
    # label map is byte-identical to the pre-2026-05-27 behavior. Verified
    # np.array_equal against the legacy path on the H9 floor-500 images.
    try:
        from scipy.ndimage import find_objects as _find_objects
        from scipy.ndimage import label as cc_label
        _have_scipy = True
    except Exception:
        _have_scipy = False

    if _have_scipy:
        slices = _find_objects(nuc_labels)

    # Uniform pad so EVERY morphological footprint (border erosion AND the
    # nucleolus closing) sees the true neighborhood at the ROI edge — making
    # the cropped computation byte-identical to the legacy full-frame one.
    # The largest footprint reach is max(border erosion radius, closing radius).
    _pad = max(int(border_distance), int(params.closing_radius_px) if se is not None else 0, 0)
    Hf, Wf = nuc_labels.shape

    for nid in unique_ids:
        nid_i = int(nid)
        # Resolve the bounding-box ROI for this label. find_objects returns a
        # list indexed by (label-1); skip labels with no slice (zero pixels).
        sl = None
        if _have_scipy and 1 <= nid_i <= len(slices):
            sl = slices[nid_i - 1]
        if sl is None:
            continue
        ys, xs = sl
        # Padded window (clamped to frame). All per-nucleus work happens here;
        # we write back only the inner (original-bbox) extent.
        y0 = max(0, ys.start - _pad); y1 = min(Hf, ys.stop + _pad)
        x0 = max(0, xs.start - _pad); x1 = min(Wf, xs.stop + _pad)
        sub_lab = nuc_labels[y0:y1, x0:x1]
        sub_dapi = dapi_2d[y0:y1, x0:x1]
        nuc_mask = sub_lab == nid
        nuc_area = int(nuc_mask.sum())
        if nuc_area < min_area_px * 2:
            continue
        nuc_dapi = sub_dapi[nuc_mask]
        thr = float(np.percentile(nuc_dapi, params.intra_nuclear_percentile))
        # Constrain candidate search to interior of nucleus (away from border).
        # 2026-05-27 PERF/CORRECTNESS: legacy eroded the GLOBAL foreground mask
        # (nuc_labels > 0) once, then ANDed per nucleus. For TOUCHING nuclei
        # the global foreground merges adjacent nuclei into one blob, so their
        # shared internal boundary is NOT eroded — eroding a lone nucleus mask
        # would wrongly strip that shared edge. We erode the GLOBAL foreground
        # cropped to the PADDED window (sub_lab > 0), which reproduces the
        # global erosion exactly within the ROI.
        interior_mask = nuc_mask
        if border_distance > 0:
            fg = (sub_lab > 0)
            try:
                from skimage.morphology import binary_erosion as _be, disk as _disk
                eroded = _be(fg, _disk(border_distance))
            except Exception:
                try:
                    from scipy.ndimage import binary_erosion as _be
                    eroded = _be(fg, iterations=border_distance)
                except Exception:
                    eroded = None
            if eroded is not None:
                _interior = nuc_mask & eroded
                # If erosion wipes the interior, fall back to whole nucleus.
                if _interior.any():
                    interior_mask = _interior
        candidate = (sub_dapi <= thr) & interior_mask

        # Connected components inside this nucleus (within the padded ROI)
        try:
            cc, _ = cc_label(candidate)
        except Exception:
            from skimage.measure import label as cc_label_sk
            cc = cc_label_sk(candidate)

        # Size-filter components
        max_area_px = int(nuc_area * params.max_area_frac_of_nucleus)
        out_sub = out[y0:y1, x0:x1]
        for comp_id in range(1, int(cc.max()) + 1):
            comp_mask = cc == comp_id
            comp_size = int(comp_mask.sum())
            if comp_size < min_area_px:
                continue
            if comp_size > max_area_px:
                # too big to plausibly be a nucleolus — likely the whole
                # nuclear interior of a dim nucleus
                continue
            if se is not None:
                comp_mask = binary_closing(comp_mask, se)
                # ensure still inside the parent nucleus
                comp_mask &= nuc_mask
            out_sub[comp_mask] = nid_i

    return out


def chromatin_metrics_per_nucleus(
    nuc_labels: np.ndarray,
    dapi_2d: np.ndarray,
    nucleolus_labels: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Per-nucleus DAPI / chromatin texture metrics.

    Returns one row per nucleus_id with:
      - nucleus_id, nucleus_area_px
      - dapi_mean, dapi_median, dapi_std, dapi_cv, dapi_p10, dapi_p90
      - heterochromatin_fraction: fraction of nuclear pixels >= 1.5x median
      - heterochromatin_density: bright-DAPI pixels per µm² (needs pixel size)
        (we leave the per-um² conversion to callers; just report _px2 here)
      - nucleolus_area_px (if nucleolus_labels provided), else NaN
      - nucleolus_fraction_of_nucleus_area (if nucleolus_labels provided)
      - nuc_excluding_nucleolus_dapi_mean (DAPI inside nucleus, OUTSIDE
        the nucleolus — this is the cleaner "chromatin" intensity)
    """
    unique_ids = np.unique(nuc_labels)
    unique_ids = unique_ids[unique_ids > 0]

    # 2026-05-27 PERF: same bounding-box crop as detect_nucleoli. Each
    # per-nucleus metric was computed via a whole-image `nuc_labels == nid`
    # scan; cropping to the label's bbox shrinks every op to the ROI while
    # computing over the identical pixels (byte-identical metrics). find_objects
    # returns slices indexed by (label-1).
    try:
        from scipy.ndimage import find_objects as _find_objects
        _slices = _find_objects(nuc_labels)
    except Exception:
        _slices = None

    rows = []
    for nid in unique_ids:
        nid_i = int(nid)
        sl = None
        if _slices is not None and 1 <= nid_i <= len(_slices):
            sl = _slices[nid_i - 1]
        if sl is None:
            continue
        sub_lab = nuc_labels[sl]
        sub_dapi = dapi_2d[sl]
        mask = sub_lab == nid
        area = int(mask.sum())
        if area == 0:
            continue
        vals = sub_dapi[mask].astype(np.float64)
        med = float(np.median(vals))
        mean = float(vals.mean())
        std = float(vals.std())
        cv = float(std / mean) if mean > 0 else float("nan")
        p10 = float(np.percentile(vals, 10))
        p90 = float(np.percentile(vals, 90))
        hetero_thr = 1.5 * med
        hetero_frac = float((vals >= hetero_thr).mean())

        if nucleolus_labels is not None:
            sub_nucleolus = nucleolus_labels[sl]
            nucleolus_mask = (sub_nucleolus == nid) & mask
            nucleolus_area = int(nucleolus_mask.sum())
            outside_nuc_mask = mask & (~nucleolus_mask)
            if outside_nuc_mask.any():
                outside_vals = sub_dapi[outside_nuc_mask].astype(np.float64)
                outside_mean = float(outside_vals.mean())
            else:
                outside_mean = float("nan")
            nucleolus_frac = float(nucleolus_area / area) if area > 0 else float("nan")
        else:
            nucleolus_area = 0
            outside_mean = float("nan")
            nucleolus_frac = float("nan")

        rows.append({
            "nucleus_id": int(nid),
            "nucleus_area_px": area,
            "dapi_mean": mean,
            "dapi_median": med,
            "dapi_std": std,
            "dapi_cv": cv,
            "dapi_p10": p10,
            "dapi_p90": p90,
            "heterochromatin_fraction": hetero_frac,
            "nucleolus_area_px": nucleolus_area,
            "nucleolus_fraction_of_nucleus": nucleolus_frac,
            "chromatin_dapi_mean": outside_mean,  # DAPI in nucleus minus nucleolus
        })

    return pd.DataFrame(rows)


def render_nucleolus_overlay(
    dapi_2d: np.ndarray,
    nuc_labels: np.ndarray,
    nucleolus_labels: np.ndarray,
    *,
    nucleus_outline_rgb: tuple = (255, 255, 255),
    nucleolus_color_rgb: tuple = (255, 140, 0),  # orange
    nucleolus_alpha: float = 0.55,
) -> np.ndarray:
    """Render a QC RGB image showing DAPI + nucleus outlines + nucleolus
    color overlay. Lets the user visually verify nucleolus segmentation.

    Returns uint8 HxWx3 RGB array. Save with imageio.imwrite() or PIL.
    """
    # Normalize DAPI to 0-1 using its own p1/p99.5
    p_lo = float(np.percentile(dapi_2d, 1.0))
    p_hi = float(np.percentile(dapi_2d, 99.5))
    if p_hi <= p_lo:
        p_hi = p_lo + 1.0
    dn = np.clip((dapi_2d.astype(np.float64) - p_lo) / (p_hi - p_lo), 0.0, 1.0)
    rgb = np.stack([dn, dn, dn], axis=-1)  # grayscale
    # Boost contrast slightly for QC
    rgb = np.clip(rgb * 1.1, 0.0, 1.0)

    # Nucleolus overlay (orange fill at ~55% opacity)
    nucleolus_mask = nucleolus_labels > 0
    if nucleolus_mask.any():
        oc = np.asarray(nucleolus_color_rgb, dtype=np.float64) / 255.0
        a = float(nucleolus_alpha)
        for c in range(3):
            rgb[..., c] = np.where(
                nucleolus_mask,
                rgb[..., c] * (1.0 - a) + oc[c] * a,
                rgb[..., c],
            )

    # Nucleus outlines (white)
    try:
        from scipy.ndimage import binary_erosion
        nuc_mask = nuc_labels > 0
        outline = nuc_mask & ~binary_erosion(nuc_mask, iterations=2)
        nc = np.asarray(nucleus_outline_rgb, dtype=np.float64) / 255.0
        for c in range(3):
            rgb[..., c] = np.where(outline, nc[c], rgb[..., c])
    except Exception:
        pass

    return (rgb * 255.0).astype(np.uint8)


def classify_spots_by_subnuclear_region(
    spots_df: pd.DataFrame,
    nuc_labels: np.ndarray,
    nucleolus_labels: np.ndarray,
) -> pd.DataFrame:
    """Add `in_nucleolus` column (0/1) and refine `in_nucleus` to
    exclude nucleolar spots (so the three compartments — nucleolus,
    nucleus-excluding-nucleolus, cytoplasm — are disjoint).

    Expects spots_df with columns x_px, y_px, in_nucleus, in_cytoplasm.
    Returns a copy with the new/updated columns.
    """
    out = spots_df.copy()
    if not len(out):
        out["in_nucleolus"] = []
        return out

    xi = out["x_px"].round().astype(int).clip(0, nuc_labels.shape[1] - 1)
    yi = out["y_px"].round().astype(int).clip(0, nuc_labels.shape[0] - 1)
    nucleolus_at_spot = nucleolus_labels[yi.values, xi.values]
    out["in_nucleolus"] = (nucleolus_at_spot > 0).astype(int)

    # Spots that were "in_nucleus" but are actually in the nucleolus —
    # downgrade the in_nucleus flag so the three compartments are disjoint.
    # (in_nucleolus is treated as a strict subset of in_nucleus.)
    if "in_nucleus" in out.columns:
        in_nuc = out["in_nucleus"].astype(bool)
        in_nucleolus = out["in_nucleolus"].astype(bool)
        out["in_nucleus_excluding_nucleolus"] = (in_nuc & ~in_nucleolus).astype(int)
    return out
