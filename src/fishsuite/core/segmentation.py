"""Nuclear segmentation wrapper.

Imports the existing Fiji-pipeline segmentation routines verbatim — no
algorithm reinvention. Single entry point: ``segment_nuclei``.

Also provides ``_smooth_label_boundaries`` — a per-label morphological
post-processing step (closing + opening with a disk SE) that rounds off
the sharp corners introduced by StarDist's star-convex polygon
predictions where neighboring instances meet.

Finally, this module owns the FIXED-N NUCLEUS SAMPLER
(``nucleus_pre_pass`` / ``resolve_nucleus_sampling``) — see
``resolve_nucleus_sampling`` for the ordering rules and the bias notes.
"""
from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Iterable, List, Sequence, Tuple

import numpy as np


# The segmentation backends now ship inside this package at
# ``core/_vendor/segmentation/`` — see ``_vendor/PROVENANCE.md`` for the source
# repository, commit and per-file checksums. They are copied verbatim; change
# behaviour by wrapping here, never by editing the vendored file.


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
    stats: Dict[str, Any] | None = None,
) -> np.ndarray:
    """Segment nuclei in a 2D DAPI image.

    Parameters
    ----------
    dapi_2d : 2D float / int array.
    backend : "stardist" | "cellpose" | "otsu"
    stats : optional out-dict. When supplied it is FILLED with the label
        bookkeeping the return value cannot carry —
        ``n_segmented`` (labels the backend produced, before the
        authoritative area filter) and ``n_area_excluded`` (labels the
        authoritative area filter zeroed). Purely additive: callers that
        pass nothing are unaffected, and the returned label image is
        identical either way.
    params : dict with backend-specific knobs:
        min_area, max_area, prob_threshold, nms_threshold, n_tiles,
        stardist_model, stardist_gauss_sigma, stardist_postprocess,
        stardist_postprocess_dilate_px, stardist_postprocess_otsu_sigma,
        stardist_postprocess_mask_closing_px,
        label_smoothing_radius_px,
        diameter, flow_threshold, cellprob_threshold, cellpose_model_type
    """
    from ._vendor.segmentation.segment_image import run_backend
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
        # 2026-05-27: OPT-IN GPU device selector. Default "cpu" => existing
        # CPU behavior is byte-for-byte unchanged (run_backend forwards "cpu"
        # to segment_cellpose, which takes the legacy gpu=False path). Only
        # "directml" enables the torch-directml GPU path (fp32 net on GPU,
        # sparse flow-dynamics forced back to CPU). Non-cellpose backends
        # ignore this kwarg.
        cellpose_device=str(p.get("cellpose_device", "cpu")),
    )
    # 2026-05-25: cellpose speed lever (no CUDA on Brian's AMD GPU). cpsam
    # runtime scales ~quadratically with pixel count, and H9 DAPI at
    # 0.065 µm/px is heavily oversampled for segmentation (nuclei ~200 px).
    # Downsample by `cellpose_downsample_factor`, segment on the smaller grid
    # with a proportionally smaller diameter + min_area, then upsample integer
    # labels (nearest-neighbour) back to full res. The AUTHORITATIVE full-res
    # area filter below is unaffected — it runs on the upsampled labels.
    # The field is named cellpose_downsample_factor for historical reasons but
    # the downsample applies to ANY backend (StarDist benefits too: at
    # 0.065 µm/px H9 nuclei ~200 px exceed StarDist's training scale and
    # over-segment; downsampling to ~0.13 µm/px — the cardiomyocyte scale —
    # fixes it, ~150x faster than full-res cellpose). diameter is only used by
    # cellpose (StarDist ignores it); scaling it is harmless.
    _ds = float(p.get("cellpose_downsample_factor", 1.0))
    _do_ds = (_ds > 1.0)
    seg_img = dapi_2d
    if backend == "cellpose":
        try:
            import os as _os, torch as _torch
            _torch.set_num_threads(int(_os.cpu_count() or 12))
        except Exception:
            pass
    if _do_ds:
        from skimage.transform import rescale as _rescale
        seg_img = _rescale(dapi_2d.astype(np.float32), 1.0 / _ds, order=1,
                           anti_aliasing=True, preserve_range=True)
        kwargs["diameter"] = float(kwargs.get("diameter", 0.0)) / _ds
        kwargs["min_area"] = max(1, int(kwargs["min_area"] / (_ds * _ds)))
    labels = run_backend(backend, seg_img, **kwargs)
    if _do_ds and labels.shape != dapi_2d.shape:
        from skimage.transform import resize as _resize
        labels = _resize(labels, dapi_2d.shape, order=0, preserve_range=True,
                         anti_aliasing=False).astype(np.int32)
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
    # `stats` records the two counts the label image itself cannot express:
    # how many labels the backend produced, and how many the area filter
    # zeroed. Zeroing does NOT renumber, so ``labels.max()`` afterwards is a
    # max label ID, not a count — hence the explicit bookkeeping.
    _n_segmented = int(np.count_nonzero(np.unique(labels)))
    _n_area_excluded = 0
    if _user_min_area > _backend_min_area or _user_max_area < 1e12:
        from scipy.ndimage import sum as _ndi_sum
        _label_ids = np.unique(labels)
        _label_ids = _label_ids[_label_ids != 0]
        if len(_label_ids) > 0:
            _areas = _ndi_sum(np.ones_like(labels), labels, _label_ids)
            _bad = _label_ids[(_areas < _user_min_area) | (_areas > _user_max_area)]
            if len(_bad) > 0:
                _n_area_excluded = int(len(_bad))
                _mask = np.isin(labels, _bad)
                labels = labels.copy()
                labels[_mask] = 0
    if stats is not None:
        stats["n_segmented"] = _n_segmented
        stats["n_area_excluded"] = _n_area_excluded
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


def identify_ghost_nuclei(
    nuclei_df,
    *,
    max_dapi_cv: float = 0.12,
    min_area_px: int = 6000,
    spot_count_col: str = "rna_spot_count",
    dapi_cv_col: str = "dapi_cv",
    area_col: str = "nucleus_area_px",
):
    """Identify empty 'ghost' nucleus shells via a POST-detection composite rule.

    Returns the list of ``nucleus_id`` values judged to be ghosts: segmented
    objects that are large, flat (low DAPI texture) AND carry zero detected RNA
    spots — i.e. out-of-focus debris / coverslip-edge ovals that cellpose
    segments off the aberrant border band seen in some SINGLE-PLANE snaps.

    OPT-IN. The caller only invokes this when ``cfg.nuclei.reject_ghost_nuclei``
    is True; default-off means every other dataset/preset is byte-for-byte
    unchanged.

    Rationale (2026-05-29 audit, BIN1 d8cMyo RNase WELLS12 run): on the
    low-contrast KO single planes NO single interior-DAPI metric separates the
    18 ghost shells from real nuclei — their DAPI CV / heterochromatin fraction
    / Laplacian variance are all embedded inside the real-KO distribution. The
    ONLY clean signal is 0 detected spots, but a real spotless nucleus can exist
    (one real KO z-stack nucleus had 0 spots). The composite rule
    ``spots==0 AND area>=min_area_px AND dapi_cv<=max_dapi_cv`` separates all 18
    ghosts (area 6908-8040 px, cv 0.072-0.100) from every real nucleus
    (the real 0-spot nucleus is only 3776 px; WT nuclei have cv>=0.144) with a
    safety margin and ZERO false drops in WT / z-stacks. A nucleus is required to
    satisfy ALL THREE conditions — each alone is intentionally insufficient.

    Parameters
    ----------
    nuclei_df : pandas.DataFrame with per-nucleus rows including the spot-count,
        DAPI-CV and area columns named below. Rows missing a required column or
        carrying NaN in ``dapi_cv`` are treated as NON-ghost (conservative).
    max_dapi_cv : ghost iff dapi_cv <= this (flat shell). Default 0.12.
    min_area_px : ghost iff area >= this (large oval). Default 6000.
    spot_count_col / dapi_cv_col / area_col : column names.

    Returns
    -------
    list[int] : nucleus_id values flagged as ghosts (possibly empty).
    """
    import math as _math
    if nuclei_df is None or len(nuclei_df) == 0:
        return []
    needed = [spot_count_col, dapi_cv_col, area_col, "nucleus_id"]
    if any(c not in nuclei_df.columns for c in needed):
        return []
    ghost_ids = []
    for _, row in nuclei_df.iterrows():
        cv = row.get(dapi_cv_col)
        try:
            cv = float(cv)
        except Exception:
            continue
        if cv != cv:  # NaN -> conservative keep
            continue
        try:
            spots = float(row.get(spot_count_col))
            area = float(row.get(area_col))
        except Exception:
            continue
        if spots == 0 and area >= float(min_area_px) and cv <= float(max_dapi_cv):
            ghost_ids.append(int(row["nucleus_id"]))
    return ghost_ids


# ===========================================================================
# FIXED-N NUCLEUS SAMPLER
# ===========================================================================
# Quantify the SAME number of nuclei in every field of view, so every
# condition is compared on an identical denominator instead of on however
# many nuclei happened to land in each frame.
#
# Two invariants govern the whole feature:
#
# 1. FILTER ORDER IS area -> border -> ghost -> sample. Sampling runs LAST,
#    over the set that survived every quality filter. Sampling BEFORE
#    filtering would put the variable denominator straight back.
#
# 2. THE ORDERING KEY COMES ONLY FROM THE DAPI CHANNEL AND GEOMETRY, never
#    from the analysis channels. Ranking nuclei by anything derived from the
#    reported readout is selection-on-the-outcome.
#
# Invariant 2 is why there is no ``dapi_rank`` and no ``area_rank`` order.
# DAPI integrated intensity tracks DNA content and nuclear area tracks cell
# size; both track cell-cycle stage, which correlates with total RNA — i.e.
# with the readout. Selecting the brightest or largest nuclei would bias the
# reported RNA measurement through the back door. They are excluded on
# purpose; do not add them.
#
# The sampler NEVER renumbers labels. It carries a set of selected nucleus
# IDs. Downstream code assumes dense label IDs 1..N (``exclude_border_labels``
# renumbers contiguously and the per-nucleus loops iterate
# ``range(1, n_after + 1)``), so relabelling here would silently corrupt every
# ID-keyed join in the pipeline.


@dataclass
class NucleusSampling:
    """Resolved per-unit nucleus selection.

    Attributes
    ----------
    eligible_ids : nucleus IDs that survived area + border + ghost filtering,
        sorted ascending.
    ordered_ids : ``eligible_ids`` in the APPLIED order (the draw order for
        ``random``; the scan order for ``raster`` / ``center_out``).
    selected_ids : the IDs actually quantified — the first ``n_target`` of
        ``ordered_ids``, or empty when the unit was dropped.
    rank : nucleus ID -> 1-based position in ``ordered_ids``. Assigned to
        EVERY eligible nucleus, not just the selected ones, so the applied
        order is fully recoverable and N can be revisited after the fact
        (``selected`` is exactly ``rank <= n_target``).
    short_of_target : True when fewer nuclei were eligible than requested.
    unit_dropped : True when ``on_short="drop_unit"`` fired.
    included_in_stats : ``n_eligible >= min_eligible`` — an ANNOTATION only.
        Nothing in the pipeline excludes an image on this flag.
    """

    n_target: int
    order: str
    unit_key: str
    seed_used: int
    eligible_ids: List[int] = field(default_factory=list)
    ordered_ids: List[int] = field(default_factory=list)
    selected_ids: List[int] = field(default_factory=list)
    rank: Dict[int, int] = field(default_factory=dict)
    # None = NOT KNOWN for this image, which is different from zero. The batch
    # threshold pre-scan hands the mode labels it already segmented and
    # area-filtered, so those two counts are unrecoverable on that path;
    # reporting 0 there would assert that nothing was filtered.
    n_segmented: int | None = 0
    n_area_excluded: int | None = 0
    n_border_excluded: int = 0
    n_ghost_excluded: int = 0
    short_of_target: bool = False
    unit_dropped: bool = False
    included_in_stats: bool = True

    @property
    def n_eligible(self) -> int:
        return len(self.eligible_ids)

    @property
    def n_sampled(self) -> int:
        return len(self.selected_ids)

    @property
    def selected_set(self) -> set:
        return set(self.selected_ids)


def nucleus_pre_pass(
    labels: np.ndarray,
    dapi_2d: np.ndarray,
    *,
    spot_y: np.ndarray | Sequence[float] | None = None,
    spot_x: np.ndarray | Sequence[float] | None = None,
) -> Dict[int, Dict[str, float]]:
    """Materialise the per-nucleus keys the sampler and ghost filter need.

    Runs ONCE per image, BEFORE the per-nucleus loop, over the labels that
    survived the area + border filters. Everything here is either geometry or
    DAPI — deliberately nothing from an analysis channel (see the ordering-key
    invariant above) except ``spot_count``, which the GHOST rule needs and the
    sampler never reads as an ordering key.

    Returns ``{label_id: {area, centroid_y, centroid_x, dapi_mean, dapi_cv,
    spot_count}}``. Only labels actually present in ``labels`` appear, so IDs
    zeroed by the area filter are absent and therefore never eligible.

    ``dapi_cv`` is ``std/mean`` with ddof=0 — the same definition
    ``nucleolus.chromatin_metrics_per_nucleus`` uses, so a ghost verdict here
    matches a ghost verdict there.
    """
    out: Dict[int, Dict[str, float]] = {}
    if labels is None or labels.size == 0:
        return out
    ids = np.unique(labels)
    ids = ids[ids != 0]
    if ids.size == 0:
        return out

    from scipy import ndimage as _ndi

    dapi_f = np.asarray(dapi_2d, dtype=np.float64)
    # scipy bins over 1..max(id) internally, so a gap in the label numbering
    # (left by the area filter, which zeroes without renumbering) makes it
    # divide by a zero count for the absent IDs. Those bins are discarded
    # below — only IDs in `ids` are read — so silence the numpy warning rather
    # than pre-compacting the label image, which would renumber it.
    with np.errstate(invalid="ignore", divide="ignore"):
        areas = _ndi.sum(np.ones_like(labels, dtype=np.float64), labels, ids)
        centroids = _ndi.center_of_mass(
            np.ones_like(labels, dtype=np.float64), labels, ids
        )
        dapi_means = _ndi.mean(dapi_f, labels, ids)
        dapi_sds = _ndi.standard_deviation(dapi_f, labels, ids)

    # Spot count per label: a plain lookup of the label image at each detected
    # spot's rounded pixel coordinate. Spots landing on background (label 0)
    # are ignored.
    spot_counts: Dict[int, int] = {}
    if spot_y is not None and spot_x is not None:
        sy = np.asarray(spot_y, dtype=np.float64)
        sx = np.asarray(spot_x, dtype=np.float64)
        if sy.size and sy.size == sx.size:
            h, w = labels.shape[:2]
            yi = np.clip(np.rint(sy).astype(np.int64), 0, h - 1)
            xi = np.clip(np.rint(sx).astype(np.int64), 0, w - 1)
            lab_at_spot = np.asarray(labels)[yi, xi]
            lab_at_spot = lab_at_spot[lab_at_spot != 0]
            if lab_at_spot.size:
                bc = np.bincount(lab_at_spot.astype(np.int64))
                for k in np.nonzero(bc)[0]:
                    spot_counts[int(k)] = int(bc[k])

    areas = np.atleast_1d(areas)
    dapi_means = np.atleast_1d(dapi_means)
    dapi_sds = np.atleast_1d(dapi_sds)
    if ids.size == 1 and not isinstance(centroids, list):
        centroids = [centroids]
    for i, lab in enumerate(ids.tolist()):
        lab = int(lab)
        mean = float(dapi_means[i])
        sd = float(dapi_sds[i])
        cy, cx = centroids[i]
        out[lab] = {
            "area": float(areas[i]),
            "centroid_y": float(cy),
            "centroid_x": float(cx),
            "dapi_mean": mean,
            "dapi_cv": float(sd / mean) if mean > 0 else float("nan"),
            "spot_count": int(spot_counts.get(lab, 0)),
        }
    return out


def spot_xy_columns(spots_df) -> Tuple[Any, Any]:
    """Pull (y, x) pixel coordinate arrays off a spot table.

    ``spots.detect_spots`` writes the canonical ``y_px`` / ``x_px``; some
    downstream frames carry bare ``y`` / ``x``. Resolving the naming in ONE
    place matters more than it looks: silently finding neither name yields zero
    spot counts for every nucleus, which would make the ghost rule (zero spots
    AND large AND flat) start firing on real nuclei. Returns (None, None) only
    when the table genuinely has no coordinates.
    """
    if spots_df is None or len(spots_df) == 0:
        return None, None
    cols = getattr(spots_df, "columns", ())
    for ycol, xcol in (("y_px", "x_px"), ("y", "x")):
        if ycol in cols and xcol in cols:
            return spots_df[ycol].to_numpy(), spots_df[xcol].to_numpy()
    return None, None


def derive_unit_rng(seed: int, unit_key: str) -> np.random.Generator:
    """Per-unit PCG64 generator keyed on a STABLE string, not on call order.

    Consuming one global RNG in file order would make the sample depend on
    which worker got which image, because images are dispatched to a parallel
    pool in nondeterministic completion order. Hashing the unit's own key
    (relative image path, or well name) into a ``SeedSequence`` spawn key
    gives each unit an independent stream that is identical at ``-p 1`` and
    ``-p 12``, and identical whether or not its neighbours were processed
    first.
    """
    key = int.from_bytes(
        hashlib.blake2b(str(unit_key).encode("utf-8"), digest_size=8).digest(),
        "big",
    )
    return np.random.default_rng(
        np.random.SeedSequence(entropy=int(seed), spawn_key=(key,))
    )


def order_nuclei(
    pre: Dict[int, Dict[str, float]],
    ids: Iterable[int],
    *,
    order: str = "random",
    rng: np.random.Generator | None = None,
    field_shape: Tuple[int, int] | None = None,
) -> List[int]:
    """Return ``ids`` in the applied sampling order.

    ``random`` — seeded uniform permutation. THE ONLY UNBIASED OPTION, and the
    one to use for any publication run.

    ``raster`` — strict top-to-bottom, left-to-right scan by centroid. This is
    SPATIALLY SYSTEMATIC AND THEREFORE BIASED: any gradient across the field
    (illumination falloff, focus tilt, edge-of-well confluency) is sampled
    non-uniformly, because the first N nuclei all come from the top band of
    the frame. Provided because reviewers ask for a literal "first N".

    ``center_out`` — ascending Euclidean distance from the field centre.
    DELIBERATELY CENTRE-BIASED: it systematically excludes the periphery. The
    optical argument for it is real (best PSF, flattest illumination), but it
    is a bias, and must never be described as unbiased.

    Every order ends on a total tie-break by nucleus ID, so the result is
    fully deterministic.
    """
    ids = sorted(int(i) for i in ids)
    if not ids:
        return []
    if order == "random":
        if rng is None:
            raise ValueError("order='random' requires a seeded rng")
        return [int(v) for v in rng.permutation(np.asarray(ids, dtype=np.int64))]
    if order == "raster":
        return sorted(
            ids,
            key=lambda i: (
                pre[i]["centroid_y"], pre[i]["centroid_x"], i,
            ),
        )
    if order == "center_out":
        if field_shape is not None:
            cy0 = (float(field_shape[0]) - 1.0) / 2.0
            cx0 = (float(field_shape[1]) - 1.0) / 2.0
        else:
            cy0 = float(np.mean([pre[i]["centroid_y"] for i in ids]))
            cx0 = float(np.mean([pre[i]["centroid_x"] for i in ids]))
        return sorted(
            ids,
            key=lambda i: (
                (pre[i]["centroid_y"] - cy0) ** 2 + (pre[i]["centroid_x"] - cx0) ** 2,
                pre[i]["centroid_y"], pre[i]["centroid_x"], i,
            ),
        )
    raise ValueError(f"unknown sampling order {order!r}")


def resolve_nucleus_sampling(
    pre: Dict[int, Dict[str, float]],
    *,
    n_target: int,
    unit_key: str,
    seed: int = 0,
    order: str = "random",
    ghost_ids: Iterable[int] = (),
    on_short: str = "keep",
    min_eligible: int = 0,
    field_shape: Tuple[int, int] | None = None,
    n_segmented: int | None = 0,
    n_area_excluded: int | None = 0,
    n_border_excluded: int = 0,
) -> NucleusSampling:
    """Resolve which nuclei this unit quantifies.

    ``pre`` is the ``nucleus_pre_pass`` table for the labels that already
    survived the area + border filters; ``ghost_ids`` are removed on top of
    that. Sampling therefore runs strictly last in the
    area -> border -> ghost -> sample chain, and a nucleus that failed ANY
    filter can never be selected.

    ``on_short`` — what to do when fewer nuclei are eligible than requested:
    ``keep`` takes every eligible nucleus (flagged short), ``drop_unit``
    quantifies none of them and records the unit as dropped, ``fail`` raises.
    """
    ghosts = {int(g) for g in ghost_ids}
    eligible = sorted(int(i) for i in pre.keys() if int(i) not in ghosts)
    n_target = int(n_target)

    rng = derive_unit_rng(seed, unit_key) if order == "random" else None
    ordered = order_nuclei(
        pre, eligible, order=order, rng=rng, field_shape=field_shape
    )
    ranks = {int(nid): i + 1 for i, nid in enumerate(ordered)}

    short = len(eligible) < n_target
    dropped = False
    if short and on_short == "fail":
        raise ValueError(
            f"nucleus sampling: unit {unit_key!r} has {len(eligible)} eligible "
            f"nuclei but sampling.n_per_unit is {n_target} "
            f"(sampling.on_short='fail')"
        )
    if short and on_short == "drop_unit":
        dropped = True
        selected: List[int] = []
    else:
        selected = list(ordered[:n_target])

    return NucleusSampling(
        n_target=n_target,
        order=str(order),
        unit_key=str(unit_key),
        seed_used=int(seed),
        eligible_ids=eligible,
        ordered_ids=ordered,
        selected_ids=selected,
        rank=ranks,
        n_segmented=None if n_segmented is None else int(n_segmented),
        n_area_excluded=None if n_area_excluded is None else int(n_area_excluded),
        n_border_excluded=int(n_border_excluded),
        n_ghost_excluded=int(len(ghosts & set(pre.keys()))),
        short_of_target=bool(short),
        unit_dropped=bool(dropped),
        included_in_stats=bool(len(eligible) >= int(min_eligible)),
    )


def sampling_per_image_cols(
    res: NucleusSampling, *, n_ghost_excluded: int = 0
) -> Dict[str, Any]:
    """The per-image-summary provenance block for one sampled image.

    Shared by every mode so the column names and their meaning cannot drift
    apart between rna_only and rna_rna. ``n_nuclei_border_excluded`` is NOT
    emitted here — the modes already report it, and duplicating it would give
    two columns that could disagree.

    Every stage of the filter chain is accounted for, so the drop from
    segmented to sampled is fully itemised and nothing vanishes unexplained.

    A count that is genuinely UNKNOWN for this image is written as NaN (an
    empty CSV cell), never as 0 — see ``NucleusSampling.n_segmented``.
    """
    return {
        "n_nuclei_segmented": (
            float("nan") if res.n_segmented is None else int(res.n_segmented)
        ),
        "n_nuclei_area_excluded": (
            float("nan") if res.n_area_excluded is None else int(res.n_area_excluded)
        ),
        "n_nuclei_ghost_excluded": int(n_ghost_excluded),
        "n_nuclei_eligible": int(res.n_eligible),
        "n_nuclei_sampled": int(res.n_sampled),
        "sampling_short_of_target": bool(res.short_of_target),
        "sampling_unit_dropped": bool(res.unit_dropped),
        "sampling_unit": str(res.unit_key),
        "sampling_order": str(res.order),
        "sampling_seed_used": int(res.seed_used),
        "sampling_n_target": int(res.n_target),
        # ANNOTATION ONLY — see SamplingCfg.min_eligible. Nothing in the
        # pipeline drops an image on this flag.
        "image_included_in_stats": bool(res.included_in_stats),
    }


def allocate_per_unit(unit_keys: Sequence[str], n_total: int) -> Dict[str, int]:
    """Split ``n_total`` equally across the images beneath one well.

    Used only by ``sampling.unit="per_well"``. Drawing flat from a pooled well
    would let one dense field of view supply most of the sample, which is the
    variable denominator the feature exists to remove — one crowded FOV would
    dominate its own well. Instead every image under the well gets
    ``n_total // k``, and the remainder goes to the first ``n_total % k``
    images in SORTED KEY ORDER (not processing order), so the split is
    identical at any worker count.
    """
    keys = sorted(str(k) for k in unit_keys)
    k = len(keys)
    if k == 0:
        return {}
    base, rem = divmod(int(n_total), k)
    return {key: base + (1 if i < rem else 0) for i, key in enumerate(keys)}
