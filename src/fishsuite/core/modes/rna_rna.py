"""rna_rna — two-RNA-channel FISH spot detection + spot-spot colocalization.

Mirrors the rna_only template but runs independent BigFISH spot detection on
two RNA channels (``cfg.channels.rna`` and ``cfg.channels.rna2``) and adds
between-channel spot-to-spot colocalization metrics:

  - per-spot ``nn_distance_um`` (nearest spot in the partner channel)
  - per-image and per-nucleus ``paired_fraction_at_<X>um`` (fraction of spots
    in each channel with a partner-channel spot within X µm; X is
    configurable via ``cfg.spot_coloc.pair_distance_um``, default 0.3 µm)
  - per-image and per-nucleus ``median_nn_distance_um``

Channels are loaded as DAPI / RNA1 / RNA2. Nuclear segmentation runs once on
DAPI (same single step as rna_only). Pixel-coloc MAD thresholds are computed
INDEPENDENTLY per channel — the runner's batch pre-scan pools nuclear pixels
separately for each. Per-channel BigFISH thresholds are auto-LoG, same
convention as rna_only.

Outputs:
  - Per-spot CSV columns include ``channel`` ('rna1' or 'rna2') plus spot
    metrics + nn_distance / paired flag.
  - Per-nucleus CSV columns include ``n_spots_rna1``, ``n_spots_rna2``,
    ``paired_fraction_rna1_at_Xum``, ``paired_fraction_rna2_at_Xum``,
    ``median_nn_distance_rna1_um``, ``median_nn_distance_rna2_um``.
  - Per-image summary row has the same two-channel split.
  - Thresholds row has both ``rna_threshold_value`` and ``rna2_threshold_value``.
"""
from __future__ import annotations

import math
import time
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
from .rna_only import (
    ImageResult, _safe_float, _median, _resolve_thresh_intensity_floor,
)
from . import register_mode


def _measure_spot_diameter_um(
    rna_2d: np.ndarray,
    spots_df: pd.DataFrame,
    voxel_xy_um: float,
    crop_half: int = 4,
    fallback_diam_um: Optional[float] = None,
) -> np.ndarray:
    """Per-spot FWHM diameter (µm) via moment-based 2D Gaussian estimator.

    For each detected spot (with 'y_px','x_px' columns) we extract a small
    crop around the center, subtract a local background (10th-percentile),
    clip negatives, and compute the second central moment of the signal in
    pixels. ``var = sigma_x^2 + sigma_y^2`` for a circular Gaussian, so the
    1-D ``sigma_px = sqrt(var/2)``. FWHM = 2.355 * sigma, then convert to µm
    by multiplying by ``voxel_xy_um``.

    Fast (one tiny crop + a couple of sums per spot, no optimization loop) —
    needed because a Run has thousands of spots and fitting Gaussians via
    scipy.optimize for every one is prohibitively slow.

    Returns an ndarray of length ``len(spots_df)`` in µm. Spots with zero
    foreground signal (rare; usually edge-of-image clipping) fall back to
    ``fallback_diam_um`` if provided, else ``2 * voxel_xy_um``.
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
    # Pull once for speed
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
        # var = sigma_x^2 + sigma_y^2 for a circular Gaussian -> divide by 2
        sigma_px = float(np.sqrt(max(var / 2.0, 0.25)))
        fwhm_px = 2.355 * sigma_px
        diameters_um[i] = float(fwhm_px * voxel_xy_um)
    return diameters_um


def _sample_partner_local_intensity(
    partner_2d: np.ndarray,
    spots_df: pd.DataFrame,
    radius_px: float,
) -> np.ndarray:
    """Per-spot mean RAW intensity of the PARTNER channel in a disk.

    2026-05-29 Brian — intensity-based, spot-centric, FLOOR-ROBUST coloc.
    For each detected spot (with 'y_px','x_px' columns) we sample the RAW
    pixel intensities of ``partner_2d`` in a disk of radius ``radius_px``
    centered at the spot centroid and return the per-spot MEAN.

    ``partner_2d`` is the OTHER channel's 2-D plane on the SAME shared,
    z-locked plane the spots were detected on (z-lock guarantees one plane),
    so the disk sample is co-registered with the spot. Intensities are RAW
    (NOT clipped by manual_antibody_min / apply_pub_contrast_floor) — that is
    the whole point: this metric does not move when the display/spot floor
    moves.

    Returns an ndarray of length ``len(spots_df)``. Spots whose disk falls
    entirely off-image return NaN.
    """
    n = len(spots_df)
    if n == 0:
        return np.array([], dtype=np.float64)
    if partner_2d.ndim != 2:
        return np.full(n, np.nan, dtype=np.float64)
    H, W = partner_2d.shape
    rad = float(radius_px)
    if not np.isfinite(rad) or rad <= 0:
        rad = 1.0
    r = int(max(1, round(rad)))

    # Pre-build the (small) disk offset stencil ONCE. ``dy``/``dx`` are the
    # in-disk offsets (length ~pi*r^2), so each spot costs O(r^2), NOT O(H*W).
    # This is the whole point of the local-window design: a real image has
    # thousands of spots (XRN2 ~3-5k / frame) and a full-frame boolean mask or
    # ``np.where`` per spot would be O(n_spots * H * W) -> minutes/hang.
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disk = (yy * yy + xx * xx) <= (rad ** 2)
    dy = yy[disk].astype(np.intp)
    dx = xx[disk].astype(np.intp)

    partner_f = partner_2d.astype(np.float64, copy=False)
    # Round + cast centroids vectorially; clip the *centre* into-bounds so a
    # finite (possibly off-by-a-pixel) centroid never indexes out of range.
    ys_arr = np.rint(spots_df["y_px"].astype(float).to_numpy()).astype(np.intp)
    xs_arr = np.rint(spots_df["x_px"].astype(float).to_numpy()).astype(np.intp)

    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        cy = ys_arr[i]
        cx = xs_arr[i]
        yi = cy + dy
        xi = cx + dx
        inb = (yi >= 0) & (yi < H) & (xi >= 0) & (xi < W)
        if not inb.any():
            # Disk falls entirely off-image (spot well outside the frame).
            continue
        out[i] = float(partner_f[yi[inb], xi[inb]].mean())
    return out


def _disk_stencil(radius_px: float) -> Tuple[np.ndarray, np.ndarray]:
    """In-disk integer (dy, dx) offsets for a disk of radius ``radius_px``.

    Mirrors the reference scripts' ``disk_stencil``: a square grid of side
    ``2*round(radius)+1`` keeping offsets with ``dy^2+dx^2 <= radius^2``. The
    SAME stencil is used for the observed AND the null sample, so the two are
    sampled identically (the whole point of the null).
    """
    rad = float(radius_px)
    if not np.isfinite(rad) or rad <= 0:
        rad = 1.0
    ri = int(max(1, round(rad)))
    yy, xx = np.mgrid[-ri:ri + 1, -ri:ri + 1]
    disk = (yy * yy + xx * xx) <= (rad * rad)
    return yy[disk].astype(np.intp), xx[disk].astype(np.intp)


def _disk_means_at(
    partner_2d: np.ndarray,
    cys: np.ndarray,
    cxs: np.ndarray,
    dy: np.ndarray,
    dx: np.ndarray,
) -> np.ndarray:
    """Vectorized per-center mean partner intensity in a disk stencil.

    For each (cy, cx) center, returns the mean of ``partner_2d`` over the disk
    stencil (dy, dx) centered there, with centers/disks CLIPPED into the frame
    (matching the reference scripts' ``disk_mean_at`` clip-to-edge semantics).
    Fully numpy-batched: shape (n_centers, n_disk) gather, then row-mean — no
    Python per-center loop. ``cys``/``cxs`` are int arrays of length n_centers.
    """
    H, W = partner_2d.shape
    if cys.size == 0:
        return np.empty(0, dtype=np.float64)
    # (n_centers, n_disk) clipped indices.
    ys = np.clip(cys[:, None] + dy[None, :], 0, H - 1)
    xs = np.clip(cxs[:, None] + dx[None, :], 0, W - 1)
    return partner_2d[ys, xs].mean(axis=1)


def _partner_null_for_nucleus(
    partner_2d: np.ndarray,
    spot_cys: np.ndarray,
    spot_cxs: np.ndarray,
    nuc_ys: np.ndarray,
    nuc_xs: np.ndarray,
    dy: np.ndarray,
    dx: np.ndarray,
    n_null: int,
    rng: np.random.Generator,
) -> Tuple[float, np.ndarray]:
    """Per-nucleus observed stat + the (n_null,) null distribution of the stat.

    observed = mean over the nucleus's rna1 spots of [disk-mean partner intensity
               at the spot] (the spot-centric coloc statistic).
    null     = for each of ``n_null`` iterations, draw ``len(spots)`` random
               positions UNIFORMLY from this nucleus's sampling pixels
               (``nuc_ys``/``nuc_xs`` — already nucleolus-excluded by the caller
               when requested), disk-sample the partner, and take the mean. The
               whole (n_null, n_spots) draw is numpy-batched, so 1000 draws x
               hundreds of nuclei is fast (the reference scripts looped per spot
               in Python — this vectorizes that).

    Returns ``(observed_stat, null_stats)`` where ``null_stats`` has shape
    (n_null,). Returns ``(nan, empty)`` if there are no spots or no sampling
    pixels.
    """
    n_sp = int(spot_cys.size)
    npix = int(nuc_ys.size)
    if n_sp == 0 or npix == 0:
        return float("nan"), np.empty(0, dtype=np.float64)
    obs_vals = _disk_means_at(partner_2d, spot_cys, spot_cxs, dy, dx)
    obs_stat = float(obs_vals.mean())
    # Batched null: draw (n_null * n_sp) random in-nucleus pixel indices at once,
    # disk-sample them all, reshape to (n_null, n_sp), and row-mean.
    idx = rng.integers(0, npix, size=n_null * n_sp)
    rcy = nuc_ys[idx]
    rcx = nuc_xs[idx]
    all_means = _disk_means_at(partner_2d, rcy, rcx, dy, dx)  # (n_null*n_sp,)
    null_stats = all_means.reshape(n_null, n_sp).mean(axis=1)
    return obs_stat, null_stats


def _annulus_stencils(bins_px) -> list:
    """Concentric-ring integer (dy, dx) offset masks for the radial profile.

    Given outer-edge radii ``bins_px`` (px, ascending), returns one (dy, dx)
    tuple per ring. Ring 0 is the inner DISK ``0 <= d <= bins_px[0]`` (includes
    the center pixel); ring ``i`` (i>=1) is the ANNULUS
    ``bins_px[i-1] < d <= bins_px[i]``. The rings are disjoint and tile the
    disk of radius ``bins_px[-1]``. Mirrors ``_disk_stencil`` (square grid +
    radius mask) so the SAME stencil samples the observed AND the null.
    """
    edges = [0.0] + [float(b) for b in bins_px]
    r_max = int(max(1, round(edges[-1])))
    yy, xx = np.mgrid[-r_max:r_max + 1, -r_max:r_max + 1]
    d = np.sqrt((yy * yy + xx * xx).astype(np.float64))
    out = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == 0:
            ring = (d >= lo) & (d <= hi)      # inner disk incl. center
        else:
            ring = (d > lo) & (d <= hi)       # annulus
        out.append((yy[ring].astype(np.intp), xx[ring].astype(np.intp)))
    return out


def _radial_profile_for_nucleus(
    partner_2d: np.ndarray,
    spot_cys: np.ndarray,
    spot_cxs: np.ndarray,
    nuc_ys: np.ndarray,
    nuc_xs: np.ndarray,
    stencils: list,
    n_null: int,
    rng: np.random.Generator,
) -> list:
    """Per-nucleus radial coloc profile: one ``(obs_mean, null_mean, null_sd,
    n_spots)`` tuple per concentric ring in ``stencils``.

    For each ring (annulus offset mask), ``obs_mean`` is the mean over this
    nucleus's rna1 spots of [mean partner intensity in the ring around the spot];
    the per-ring null draws ``n_null`` sets of ``len(spots)`` random IN-NUCLEUS
    positions (``nuc_ys``/``nuc_xs`` — already nucleolus-excluded by the caller
    when requested) and takes the same ring-mean. ONE shared set of random
    positions is reused across rings (deterministic via ``rng``), so the radial
    profile shares a common random reference frame. Returns ``(nan, nan, nan,
    0)`` per ring when there are no spots or no sampling pixels.
    """
    n_sp = int(spot_cys.size)
    npix = int(nuc_ys.size)
    if n_sp == 0 or npix == 0:
        return [(float("nan"), float("nan"), float("nan"), 0) for _ in stencils]
    # One shared draw of random in-nucleus centers reused for every ring.
    idx = rng.integers(0, npix, size=n_null * n_sp)
    rcy = nuc_ys[idx]
    rcx = nuc_xs[idx]
    out = []
    for (dy, dx) in stencils:
        if dy.size == 0:
            out.append((float("nan"), float("nan"), float("nan"), n_sp))
            continue
        obs_vals = _disk_means_at(partner_2d, spot_cys, spot_cxs, dy, dx)
        obs_stat = float(obs_vals.mean())
        all_means = _disk_means_at(partner_2d, rcy, rcx, dy, dx)
        null_stats = all_means.reshape(n_null, n_sp).mean(axis=1)
        nmean = float(null_stats.mean())
        nsd = float(null_stats.std(ddof=1)) if null_stats.size > 1 else 0.0
        out.append((obs_stat, nmean, nsd, n_sp))
    return out


def _resolve_channels(cfg, img) -> Tuple[int, int, int]:
    """Resolve dapi / rna / rna2 channel indices from cfg, with auto fallback.

    Raises ValueError if rna2 is missing or matches rna (rna_rna needs two
    distinct RNA channels).
    """
    one_indexed = bool(cfg.channels.one_indexed)
    def _chan(idx: int) -> int:
        return (idx - 1) if (one_indexed and idx > 0) else idx

    dapi_idx = _chan(cfg.channels.dapi)
    rna_idx = _chan(cfg.channels.rna)
    rna2_idx = _chan(cfg.channels.rna2)

    if dapi_idx < 0 or rna_idx < 0:
        auto = _io.autodetect_channels(img)
        if dapi_idx < 0:
            dapi_idx = auto["dapi"]
        if rna_idx < 0:
            rna_idx = auto["rna"]

    # rna2 has no auto-detect contract — rna_rna mode requires an explicit
    # second RNA channel.
    if rna2_idx < 0:
        raise ValueError(
            "rna_rna mode requires cfg.channels.rna2 to be set to a valid "
            "channel index (got -1). Set rna2 in the YAML preset."
        )
    if rna2_idx == rna_idx:
        raise ValueError(
            f"rna_rna mode requires two DISTINCT RNA channels; got rna={rna_idx} "
            f"== rna2={rna2_idx}. Pick different channels."
        )

    dapi_idx = max(0, min(img.n_channels - 1, dapi_idx))
    rna_idx = max(0, min(img.n_channels - 1, rna_idx))
    rna2_idx = max(0, min(img.n_channels - 1, rna2_idx))
    return dapi_idx, rna_idx, rna2_idx


def _compute_pixel_coloc_thr(
    img2d: np.ndarray,
    labels: np.ndarray,
    *,
    pc_cfg,
    precomputed: Optional[float],
    bigfish_auto_thr: float,
) -> float:
    """Compute (or accept the pre-scan) pixel-coloc threshold for ONE channel.

    Same MAD-on-raw-nuclear-pixels math as rna_only, factored out so we can
    apply it independently to rna and rna2.
    """
    _scope = getattr(pc_cfg, "threshold_scope", "per_image") if pc_cfg is not None else "per_image"
    if (
        pc_cfg is not None
        and _scope == "batch"
        and precomputed is not None
        and precomputed == precomputed  # not NaN
        and precomputed > 0
    ):
        return float(precomputed)

    if pc_cfg is not None and int(labels.max()) > 0:
        nuc_pixel_mask = labels > 0
        if nuc_pixel_mask.any():
            vals = img2d[nuc_pixel_mask].astype(np.float64).tolist()
            try:
                v = float(_thr.coloc_threshold(
                    vals,
                    mode=pc_cfg.threshold_mode,
                    k_mad=float(pc_cfg.k_mad),
                    percentile=float(pc_cfg.percentile),
                ))
                if v == v and v > 0:
                    return v
            except Exception:
                pass

    # Fallback chain: BigFISH auto -> Otsu -> 99th percentile
    if bigfish_auto_thr == bigfish_auto_thr and bigfish_auto_thr > 0:
        return float(bigfish_auto_thr)
    try:
        return float(threshold_otsu(img2d))
    except Exception:
        return float(np.percentile(img2d, 99.0))


def _spot_coords_um(spots_df: pd.DataFrame, voxel_xy_um: float, voxel_z_um: float) -> np.ndarray:
    """Return Nx3 (x, y, z) coordinates in microns for the cKDTree query.

    Empty -> shape (0, 3) so cKDTree builds happily and query returns inf.
    """
    if spots_df is None or len(spots_df) == 0:
        return np.empty((0, 3), dtype=np.float64)
    x_um = spots_df["x_px"].astype(float).to_numpy() * float(voxel_xy_um)
    y_um = spots_df["y_px"].astype(float).to_numpy() * float(voxel_xy_um)
    z_um = spots_df["z_slice"].astype(float).to_numpy() * float(voxel_z_um)
    return np.stack([x_um, y_um, z_um], axis=1)


def _nn_distances(
    src_coords: np.ndarray, partner_coords: np.ndarray
) -> np.ndarray:
    """For each row in src_coords, distance (µm) to the nearest partner.

    Empty partner -> array of inf. Empty source -> empty array. Uses
    scipy.spatial.cKDTree for efficient queries.
    """
    if src_coords.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    if partner_coords.shape[0] == 0:
        return np.full(src_coords.shape[0], np.inf, dtype=np.float64)
    from scipy.spatial import cKDTree
    tree = cKDTree(partner_coords)
    dists, _ = tree.query(src_coords, k=1)
    return np.asarray(dists, dtype=np.float64)


def _format_pair_um(pair_um: float) -> str:
    """Make a column-friendly distance suffix, e.g. 0.3 -> '0p3um'."""
    s = f"{float(pair_um):.3f}".rstrip("0").rstrip(".")
    return s.replace(".", "p") + "um"


def _filter_spots_by_floor(
    spots_df: pd.DataFrame, floor: float, image_2d: np.ndarray
) -> pd.DataFrame:
    """Drop spots whose peak intensity (sampled from ``image_2d`` at the spot
    xy) is below ``floor``. Returns the filtered dataframe.

    2026-05-20 Brian/Sam: this implements
    ``output.apply_pub_contrast_floor_to_spots``. We prefer the existing
    ``intensity_peak`` column populated by ``spots._spots_to_dataframe`` (the
    canonical peak-intensity value sampled at the spot's exact xy at detection
    time). Falls back to resampling ``image_2d`` at the spot's integer
    ``x_px``/``y_px`` when the column is missing. Returns the input unchanged
    if floor is None/<=0 or there's no way to evaluate peak intensity per
    spot.
    """
    if not len(spots_df) or floor is None or float(floor) <= 0:
        return spots_df
    # Prefer the canonical column written by spots._spots_to_dataframe.
    if "intensity_peak" in spots_df.columns:
        keep = spots_df["intensity_peak"].astype(float) >= float(floor)
    elif "peak_intensity" in spots_df.columns:
        # Defensive: legacy/alternate naming if upstream ever changes.
        keep = spots_df["peak_intensity"].astype(float) >= float(floor)
    elif {"y_px", "x_px"}.issubset(spots_df.columns):
        ys = spots_df["y_px"].astype(int).clip(0, image_2d.shape[0] - 1)
        xs = spots_df["x_px"].astype(int).clip(0, image_2d.shape[1] - 1)
        sampled = image_2d[ys.values, xs.values].astype(float)
        keep = sampled >= float(floor)
    elif {"y", "x"}.issubset(spots_df.columns):
        ys = spots_df["y"].astype(int).clip(0, image_2d.shape[0] - 1)
        xs = spots_df["x"].astype(int).clip(0, image_2d.shape[1] - 1)
        sampled = image_2d[ys.values, xs.values].astype(float)
        keep = sampled >= float(floor)
    else:
        return spots_df  # no way to evaluate — leave unfiltered
    out = spots_df.loc[keep.values].reset_index(drop=True)
    return out


def run_one(
    path,
    *,
    condition: str,
    sec_only: bool,
    cfg,
    precomputed_rna_threshold: Optional[float] = None,
    precomputed_rna2_threshold: Optional[float] = None,
    analysis_floors: Optional[Dict[str, float]] = None,
    precomputed_labels: Optional[np.ndarray] = None,
    rna2_is_antibody: bool = False,
) -> ImageResult:
    """Run the rna_rna pipeline on a single image.

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
        helper builds an identical ``seg_params``, uses the same DAPI 2D
        (dust-speck masking touches only the RNA channels, never DAPI), and
        applies identical border exclusion, so the cached labels are
        bit-equivalent to what this function would otherwise compute. When
        None, segmentation runs exactly as before (per-image / non-batch
        path is unchanged).
    precomputed_rna_threshold, precomputed_rna2_threshold : float or None
        When supplied (typically by the batch runner's pre-pass in
        ``pixel_coloc.threshold_scope == 'batch'`` mode), these scalars are
        used verbatim as the pixel-coloc thresholds for THIS image's rna and
        rna2 channels respectively — bypassing the per-image median+k*MAD
        computation. Each channel has its OWN pooled pre-scan.
    analysis_floors : dict or None
        2026-05-20 Brian/Sam: per-channel floor values (keys ``"rna"`` and
        ``"rna2"``) resolved from the publication-image contrast pipeline
        (manual / auto_batch / reference_image). Used by TWO independent
        flags:

        * ``cfg.output.apply_pub_contrast_floor_to_analysis`` — apply the
          floor as a hard pixel-intensity threshold to the per-nucleus /
          per-cell pixel intensity quantification. Pixels with value <
          floor are treated as 0 (cytoplasmic noise excluded from
          intensity sums). The raw (no-floor) intensity columns remain.
        * ``cfg.output.apply_pub_contrast_floor_to_spots`` — drop detected
          BigFISH spots whose peak intensity is below the channel's floor.
          Applied BEFORE per-nucleus stratification + pairing, so all
          downstream counts (nuclear/cyto, paired_fraction, etc.) reflect
          the filtered spot set. BigFISH LoG detection itself is unchanged.

        Both flags are independent — either, both, or neither can be set.
        When a channel's floor is missing/None (e.g. no global floor
        available because pub-image was auto_per_image), the corresponding
        above-floor analysis columns are NaN, and the spot filter is a
        no-op for that channel.
    rna2_is_antibody : bool
        2026-06-05 Brian: set True ONLY by ``rna_protein.run_one`` (which
        maps the diffuse antibody/protein channel into the rna2 slot). When
        True AND ``cfg.foci.detect_antibody_spots`` is False, the rna2
        (antibody) channel is NOT BigFISH spot-detected — an EMPTY spot set
        is produced instead, so a diffuse antibody stain (e.g. QKI IF) does
        not carpet the field with meaningless spots. The intensity-based
        coloc is UNAFFECTED: ``compute_partner_intensity`` samples antibody
        PIXELS at the rna1 spots (never the antibody spots), and pixel-coloc
        reads the rna2 PIXEL plane — both still compute. Default False keeps
        plain rna_rna byte-identical (both channels always detected); the
        flag is meaningless unless rna2 is the antibody channel.
    """
    t0 = time.time()
    img = _io.read_image(path)

    dapi_idx, rna_idx, rna2_idx = _resolve_channels(cfg, img)

    z_mode = cfg.z_stack.mode
    z_start = cfg.z_stack.start_slice
    z_end = cfg.z_stack.end_slice
    # 2026-05-22 Brian: per-image z-window override. If the current image's
    # file name matches a key in cfg.z_stack.file_overrides, use that
    # image-specific start_slice / end_slice instead of the batch default.
    _file_overrides = getattr(cfg.z_stack, "file_overrides", {}) or {}
    _img_name = Path(path).name
    if _img_name in _file_overrides:
        _ovr = _file_overrides[_img_name]
        if "start_slice" in _ovr:
            z_start = int(_ovr["start_slice"])
        if "end_slice" in _ovr:
            z_end = int(_ovr["end_slice"])
        try:
            from rich.console import Console as _C
            _C().print(f"  [dim]z-override: {_img_name} → start={z_start}, end={z_end}[/dim]")
        except Exception:
            pass
    if z_start is not None and z_start > img.n_z:
        z_start = 1
    if z_end is not None and z_end > img.n_z:
        z_end = img.n_z

    # 2026-05-18 Brian: autofocus mode now LOCKS all channels to DAPI's
    # picked focal plane. Previously each channel ran autofocus independently
    # — DAPI picked slice 42, RNA1 picked 38, RNA2 picked 45 — and spot xy
    # from a different physical plane than the nuclear mask led to
    # mis-assignment of nuclear-edge spots to cytoplasm.
    # 2026-05-24 Brian: autofocus_maxproj — per-image DAPI focus-window
    # detection, then MIP that window for all channels. Replaces per-image
    # file_overrides for datasets with field-to-field focus drift.
    dapi_autofocus_z: Optional[int] = None
    if z_mode == "autofocus":
        dapi_autofocus_z, dapi_2d = _io.extract_channel_autofocus_with_idx(
            img, dapi_idx, z_start=z_start, z_end=z_end,
            intensity_weighted=bool(getattr(cfg.z_stack, "autofocus_intensity_weighted", False)),
        )
        rna_2d = _io.extract_channel_at_z(img, rna_idx, z_1indexed=dapi_autofocus_z)
        rna2_2d = _io.extract_channel_at_z(img, rna2_idx, z_1indexed=dapi_autofocus_z)
        try:
            from rich.console import Console as _C
            _C().print(
                f"  [dim]z-lock: {Path(path).name} → all channels @ DAPI plane "
                f"z={dapi_autofocus_z}[/dim]"
            )
        except Exception:
            pass
    elif z_mode == "autofocus_maxproj":
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
        rna_2d = _io.extract_channel_in_z_range(
            img, rna_idx,
            z_start_1indexed=afm_zs, z_end_1indexed=afm_ze,
            project="maxproj",
        )
        rna2_2d = _io.extract_channel_in_z_range(
            img, rna2_idx,
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
    else:
        dapi_2d = _io.extract_channel(img, dapi_idx, z_mode=z_mode, z_start=z_start, z_end=z_end)
        if dapi_2d.ndim != 2:
            dapi_2d = dapi_2d.max(axis=0)
        rna_2d = _io.extract_channel(img, rna_idx, z_mode=z_mode, z_start=z_start, z_end=z_end)
        if rna_2d.ndim != 2:
            rna_2d = rna_2d.max(axis=0)
        rna2_2d = _io.extract_channel(img, rna2_idx, z_mode=z_mode, z_start=z_start, z_end=z_end)
        if rna2_2d.ndim != 2:
            rna2_2d = rna2_2d.max(axis=0)

    # ---- Image-level dust speck removal (optional) -------------------------
    # 2026-05-21 Brian: before any spot detection / per-pixel quantification,
    # detect and mask LARGE bright artifacts (dust, fluorescent debris,
    # sensor blowouts). Applied to RNA channels only; DAPI left alone
    # (bright DAPI clusters can be real heterochromatin).
    _speck_min = int(getattr(cfg.foci, "mask_dust_specks_min_size_px", 0) or 0)
    if _speck_min > 0:
        try:
            from ..image_preprocess import mask_dust_specks
            _speck_thr = float(getattr(cfg.foci, "mask_dust_specks_threshold_mad", 50.0))
            _speck_repl = str(getattr(cfg.foci, "mask_dust_specks_replacement", "median"))
            rna_2d_clean, n_specks_rna, npx_rna = mask_dust_specks(
                rna_2d, min_speck_size_px=_speck_min,
                brightness_threshold_mad=_speck_thr, replacement=_speck_repl,
            )
            rna2_2d_clean, n_specks_rna2, npx_rna2 = mask_dust_specks(
                rna2_2d, min_speck_size_px=_speck_min,
                brightness_threshold_mad=_speck_thr, replacement=_speck_repl,
            )
            rna_2d = rna_2d_clean
            rna2_2d = rna2_2d_clean
            if n_specks_rna > 0 or n_specks_rna2 > 0:
                try:
                    from rich.console import Console as _C
                    _C().print(
                        f"  [dim]dust-speck mask: rna1 {n_specks_rna} specks ({npx_rna} px), "
                        f"rna2 {n_specks_rna2} specks ({npx_rna2} px) replaced w/ median[/dim]"
                    )
                except Exception:
                    pass
        except Exception as _exc:
            print(f"  WARN: dust speck masking failed: {type(_exc).__name__}: {_exc}")

    voxel_xy_nm = _safe_float(img.voxel_xy_nm)
    if not (voxel_xy_nm > 0):
        voxel_xy_nm = 65.0
    voxel_z_nm = _safe_float(img.voxel_z_nm)
    if not (voxel_z_nm > 0):
        voxel_z_nm = 230.0
    voxel_xy_um = voxel_xy_nm / 1000.0
    voxel_z_um = voxel_z_nm / 1000.0

    # ---- DAPI threshold mask (walkthrough step 02) ------------------------
    try:
        dapi_thr_val = float(threshold_otsu(dapi_2d))
    except Exception:
        dapi_thr_val = float(dapi_2d.mean())
    dapi_mask = (dapi_2d >= dapi_thr_val).astype(np.uint8) * 255

    # ---- Nuclear segmentation (single pass on DAPI) ------------------------
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

    # ---- Cytoplasm mask ----------------------------------------------------
    cyt_labels = None
    if cfg.cytoplasm.enabled and n_after > 0:
        cyt_labels = _morph.compute_cytoplasm_mask(
            labels, max_expand_px=cfg.cytoplasm.voronoi_max_expansion_px
        )

    # ---- Spot detection: per channel, independent BigFISH calls ------------
    # Resolve per-channel BigFISH params via FociCfg.resolved_for. When the
    # override fields on rna_overrides / rna2_overrides are None, the shared
    # FociCfg values are used (back-compat path).
    rna1_params = cfg.foci.resolved_for("rna")
    rna2_params = cfg.foci.resolved_for("rna2")

    def _detect(rna_img: np.ndarray, params: Dict[str, Any]) -> Tuple[pd.DataFrame, float]:
        # 2026-05-21 Brian: previously sec-only images were skipped entirely
        # (returned empty DataFrame). That artificially produced 0 spots in
        # negative controls, masking any genuine autofluorescence and
        # preventing the controls from validating the detection pipeline.
        # Now sec-only runs through the same detector as real-probe; if the
        # control is truly clean, BigFISH LoG + the manual floor filter will
        # yield ~0 spots organically.
        if not cfg.foci.enabled:
            return pd.DataFrame(), float("nan")
        vx = cfg.foci.bigfish_voxel_size_nm
        vz = cfg.foci.bigfish_voxel_z_nm
        if vx <= 0:
            vx = voxel_xy_nm
        if vz <= 0:
            vz = voxel_z_nm
        try:
            df_ = _spots.detect_spots(
                rna_img,
                backend=cfg.foci.backend,
                voxel_xy_nm=float(vx),
                voxel_z_nm=float(vz),
                spot_radius_nm=float(params["bigfish_spot_radius_nm"]),
                spot_radius_z_nm=float(params["bigfish_spot_radius_z_nm"]),
                threshold_multiplier=float(params["threshold_multiplier"]),
                threshold=cfg.foci.threshold_override,
                log_threshold=cfg.foci.log_threshold,
                log_spot_radius_px=cfg.foci.log_spot_radius_px,
            )
            thr_ = float(df_["threshold_used"].iloc[0]) if len(df_) else float("nan")
            return df_, thr_
        except Exception:
            return pd.DataFrame(), float("nan")

    spots1_df, thr1_val = _detect(rna_2d, rna1_params)
    # 2026-06-05 Brian: diffuse-antibody opt-out. In rna_protein mode
    # (rna2_is_antibody=True), when cfg.foci.detect_antibody_spots is False the
    # antibody/protein channel is treated as a DIFFUSE INTENSITY channel: skip
    # BigFISH spot detection and produce an EMPTY spot set (the same empty-frame
    # representation _detect returns when foci are disabled, which every
    # downstream step already tolerates). The rna2 PIXEL plane (rna2_2d) is
    # still fully loaded, so pixel-coloc (Pearson/Manders/Li) and the
    # partner-intensity coloc — which samples rna2 PIXELS at the rna1 spots, NOT
    # rna2 spots — are completely unaffected. Plain rna_rna (rna2_is_antibody
    # False) ALWAYS detects rna2, so two-real-FISH-target runs are unchanged.
    if rna2_is_antibody and not bool(getattr(cfg.foci, "detect_antibody_spots", True)):
        spots2_df, thr2_val = pd.DataFrame(), float("nan")
        print(
            f"  [antibody-spots] {path.name}: detect_antibody_spots=False -> "
            f"skipping antibody (rna2) spot detection; intensity coloc unaffected"
        )
    else:
        spots2_df, thr2_val = _detect(rna2_2d, rna2_params)

    # ---- Pub-contrast floor as HARD spot-detection floor (Brian/Sam 2026-05-20) ----
    # When ``output.apply_pub_contrast_floor_to_spots`` is True AND the caller
    # (batch runner) passed in resolved per-channel floors via the
    # ``analysis_floors`` kwarg, drop spots whose peak intensity is below the
    # corresponding channel's pub-image floor. Applied BEFORE stratification
    # so all downstream metrics (nuclear/cyto counts, paired_fraction, etc.)
    # reflect the filtered spot set. BigFISH detection itself is unchanged —
    # this is a strict post-detection filter.
    if (
        bool(getattr(cfg.output, "apply_pub_contrast_floor_to_spots", False))
        and analysis_floors
    ):
        rna1_floor = analysis_floors.get("rna")
        rna2_floor = analysis_floors.get("rna2")
        if rna1_floor and len(spots1_df):
            n_before_1 = len(spots1_df)
            spots1_df = _filter_spots_by_floor(spots1_df, rna1_floor, rna_2d)
            n_dropped_1 = n_before_1 - len(spots1_df)
            if n_dropped_1:
                print(
                    f"  [floor-filter] {path.name} rna1: dropped {n_dropped_1}/{n_before_1} "
                    f"spots below floor={float(rna1_floor):.1f}"
                )
        if rna2_floor and len(spots2_df):
            n_before_2 = len(spots2_df)
            spots2_df = _filter_spots_by_floor(spots2_df, rna2_floor, rna2_2d)
            n_dropped_2 = n_before_2 - len(spots2_df)
            if n_dropped_2:
                print(
                    f"  [floor-filter] {path.name} rna2: dropped {n_dropped_2}/{n_before_2} "
                    f"spots below floor={float(rna2_floor):.1f}"
                )

    # Stratify spots vs nuclei / cytoplasm — same call as rna_only
    if cyt_labels is not None:
        if len(spots1_df) > 0:
            spots1_df = _morph.stratify_spots(spots1_df, labels, cytoplasm_labels=cyt_labels)
        if len(spots2_df) > 0:
            spots2_df = _morph.stratify_spots(spots2_df, labels, cytoplasm_labels=cyt_labels)
    else:
        if len(spots1_df) > 0:
            spots1_df = _morph.stratify_spots(spots1_df, labels)
        if len(spots2_df) > 0:
            spots2_df = _morph.stratify_spots(spots2_df, labels)

    # ---- Pixel-coloc thresholds (independent per channel) ------------------
    pc_cfg = getattr(cfg, "pixel_coloc", None)
    rna_thr_value = _compute_pixel_coloc_thr(
        rna_2d, labels, pc_cfg=pc_cfg,
        precomputed=precomputed_rna_threshold, bigfish_auto_thr=thr1_val,
    )
    rna2_thr_value = _compute_pixel_coloc_thr(
        rna2_2d, labels, pc_cfg=pc_cfg,
        precomputed=precomputed_rna2_threshold, bigfish_auto_thr=thr2_val,
    )
    rna_pos_mask = (rna_2d >= rna_thr_value).astype(np.uint8) * 255
    rna2_pos_mask = (rna2_2d >= rna2_thr_value).astype(np.uint8) * 255

    # ---- Spot-spot colocalization (cKDTree NN) -----------------------------
    sc_cfg = getattr(cfg, "spot_coloc", None)
    pair_um = float(getattr(sc_cfg, "pair_distance_um", 0.3)) if sc_cfg is not None else 0.3
    report_nn = bool(getattr(sc_cfg, "report_nn_distance", True)) if sc_cfg is not None else True
    pair_suffix = _format_pair_um(pair_um)

    coords1 = _spot_coords_um(spots1_df, voxel_xy_um, voxel_z_um)
    coords2 = _spot_coords_um(spots2_df, voxel_xy_um, voxel_z_um)
    nn1 = _nn_distances(coords1, coords2)  # rna1 -> nearest rna2
    nn2 = _nn_distances(coords2, coords1)  # rna2 -> nearest rna1
    paired1 = (nn1 <= pair_um) if nn1.size else np.empty(0, dtype=bool)
    paired2 = (nn2 <= pair_um) if nn2.size else np.empty(0, dtype=bool)

    if len(spots1_df) > 0:
        spots1_df = spots1_df.copy()
        spots1_df["nn_distance_um"] = nn1 if report_nn else np.full(len(spots1_df), np.nan)
        spots1_df[f"paired_at_{pair_suffix}"] = paired1.astype(int)
    if len(spots2_df) > 0:
        spots2_df = spots2_df.copy()
        spots2_df["nn_distance_um"] = nn2 if report_nn else np.full(len(spots2_df), np.nan)
        spots2_df[f"paired_at_{pair_suffix}"] = paired2.astype(int)

    # ---- Per-nucleus rows --------------------------------------------------
    img_name = path.name

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

    # Index per-channel spots by nucleus_id for cheap per-nucleus aggregation.
    # NOTE: this MUST run AFTER every per-spot column the per-nucleus
    # aggregation reads has been written onto spots1_df/spots2_df. groupby
    # snapshots the columns that exist at call time, so any column added to the
    # parent frame later (e.g. ``partner_local_mean_intensity``, added below in
    # the intensity-coloc block) would be INVISIBLE to the grouped sub-frames —
    # silently NaN-ing the per-nucleus rollup. The indexes are therefore built
    # at the END of all per-spot column construction (just above the
    # per-nucleus loop), not here. (Bug fix 2026-05-29: per-spot
    # ``partner_local_mean_intensity`` was 100% populated but the per-nucleus
    # ``*_local_mean_at_*_spots`` / ``*_enrichment_at_*_spots`` were all-NaN
    # because the index predated the column.)
    def _by_nid(df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
        out_: Dict[int, pd.DataFrame] = {}
        if len(df) > 0 and "nucleus_id" in df.columns:
            for nid_val, grp in df.groupby("nucleus_id"):
                try:
                    out_[int(nid_val)] = grp
                except (TypeError, ValueError):
                    pass
        return out_

    # Spot diameter defaults (used as a fallback when per-spot measurement
    # fails for edge-clipped or all-zero crops). The actual reported
    # ``spot_diameter_um`` per spot is MEASURED below via a moment-based
    # 2D Gaussian estimator on a local crop — see ``_measure_spot_diameter_um``.
    spot_radius_um = float(cfg.foci.bigfish_spot_radius_nm) / 1000.0
    default_spot_diameter_um = 2.0 * spot_radius_um
    default_spot_fwhm_px = default_spot_diameter_um / max(voxel_xy_um, 1e-6)
    default_spot_area_px = math.pi * (default_spot_fwhm_px / 2.0) ** 2

    # Per-spot diameter measurement on the 2D plane used for detection.
    # Resulting column ``spot_diameter_um`` is a per-spot FWHM diameter (µm).
    if len(spots1_df) > 0:
        spots1_df = spots1_df.copy() if not isinstance(spots1_df, pd.DataFrame) else spots1_df
        spots1_df["spot_diameter_um"] = _measure_spot_diameter_um(
            rna_2d, spots1_df, voxel_xy_um,
            fallback_diam_um=default_spot_diameter_um,
        )
    if len(spots2_df) > 0:
        spots2_df = spots2_df.copy() if not isinstance(spots2_df, pd.DataFrame) else spots2_df
        spots2_df["spot_diameter_um"] = _measure_spot_diameter_um(
            rna2_2d, spots2_df, voxel_xy_um,
            fallback_diam_um=default_spot_diameter_um,
        )

    # ---- Intensity-based, spot-centric, FLOOR-ROBUST coloc (Brian 2026-05-29)
    # For each spot, sample the RAW local intensity of the OTHER channel in a
    # disk of radius = the pipeline's nominal spot radius (bigfish_spot_radius_nm
    # / voxel) on the SAME z-locked plane. Direction:
    #   spots1 (RNA1/intron) -> mean RAW rna2 (partner) intensity at the spot
    #   spots2 (RNA2/partner) -> mean RAW rna1 intensity at the spot
    # The single column ``partner_local_mean_intensity`` is channel-agnostic
    # (the "partner" is whatever the row's own ``channel`` is NOT). It is RAW
    # (never floored), so it is robust to manual_antibody_min / pub-contrast
    # floor changes — complementing the binary Manders/pairing metrics.
    # GATED (Brian 2026-05-29): this per-spot sampling HANGS the parallel
    # per-image worker pool (works single-process only). Until the hang is
    # fixed it is OFF by default via ``cfg.foci.compute_partner_intensity``.
    # When False the ``partner_local_mean_intensity`` column is never produced,
    # the per-nucleus ``*_local_mean_at_*_spots`` / ``*_enrichment_at_*_spots``
    # blocks and their per-image means are omitted, and the two-channel output
    # is byte-equivalent to the pre-feature (binary-coloc-only) path. Figures
    # 74_/75_ self-skip on the absent columns; 70-73 still generate.
    compute_partner_intensity = bool(
        getattr(cfg.foci, "compute_partner_intensity", False)
    )
    if compute_partner_intensity:
        partner_disk_radius_px = max(1.0, spot_radius_um / max(voxel_xy_um, 1e-6))
        if len(spots1_df) > 0:
            spots1_df["partner_local_mean_intensity"] = _sample_partner_local_intensity(
                rna2_2d, spots1_df, partner_disk_radius_px,
            )
        if len(spots2_df) > 0:
            spots2_df["partner_local_mean_intensity"] = _sample_partner_local_intensity(
                rna_2d, spots2_df, partner_disk_radius_px,
            )

    # Build the per-nucleus spot indexes NOW — after nn_distance/paired,
    # spot_diameter, and (when enabled) partner_local_mean_intensity have all
    # been written — so groupby snapshots a frame that already carries every
    # column the per-nucleus aggregation reads.
    spots1_by_nid = _by_nid(spots1_df)
    spots2_by_nid = _by_nid(spots2_df)

    nuc_rows: List[Dict[str, Any]] = []
    spot_rows: List[Dict[str, Any]] = []
    morph_rows: List[Dict[str, Any]] = []
    spot_global_id = 0

    paired_col = f"paired_fraction_at_{pair_suffix}"

    # ---- Pub-contrast floor as analysis floor (Brian/Sam 2026-05-20) ------
    # When ``output.apply_pub_contrast_floor_to_analysis`` is True AND the
    # caller (batch runner) passed in resolved per-channel floors via the
    # ``analysis_floors`` kwarg, we ALSO compute per-pixel intensity sums
    # with values below the floor clipped to 0 — i.e. the same threshold
    # the viewer's eye uses to separate signal from cytoplasmic noise in
    # the publication image. When the floor isn't available (None) the
    # corresponding above-floor columns are emitted as NaN; the raw
    # (no-floor) columns are emitted regardless and stay byte-identical
    # to the legacy output.
    apply_floor = bool(getattr(cfg.output, "apply_pub_contrast_floor_to_analysis", False))
    _af = analysis_floors if (apply_floor and analysis_floors is not None) else {}
    floor_rna1 = _af.get("rna", None)
    floor_rna2 = _af.get("rna2", None)
    have_floor_rna1 = apply_floor and (floor_rna1 is not None) and (floor_rna1 == floor_rna1)
    have_floor_rna2 = apply_floor and (floor_rna2 is not None) and (floor_rna2 == floor_rna2)
    # Pre-compute the above-floor images once per channel. ``np.clip`` of
    # ``img - floor`` to 0 is the operation Sam's eye is doing on the pub
    # image when she sets the B&C min. Numerically:
    #   nuclear_above_floor_intensity_rna1 = sum_{p in nuclear_mask} max(rna_2d[p] - floor_rna1, 0)
    rna_above_floor = None
    rna2_above_floor = None
    if have_floor_rna1:
        rna_above_floor = np.clip(
            rna_2d.astype(np.float64) - float(floor_rna1), 0.0, None
        )
    if have_floor_rna2:
        rna2_above_floor = np.clip(
            rna2_2d.astype(np.float64) - float(floor_rna2), 0.0, None
        )

    # ---- Thresholded RNA intensity in compartments (2026-06-02 Brian) ------
    # Independent of apply_pub_contrast_floor_to_analysis (above): a THIRD
    # intensity measurement integrating RAW intensities of >=floor pixels,
    # per compartment, for BOTH channels. Floors default to each channel's
    # spot floor when unset/0 (see OutputCfg.rna_intensity_threshold). The
    # rna2 slot carries the antibody floor in rna_protein mode (the
    # rna_protein wrapper maps analysis_floors["antibody"] -> "rna2", and
    # cfg.output.rna2_intensity_threshold / manual_rna2_min serve both modes).
    rna_thresh_floor = _resolve_thresh_intensity_floor(cfg, analysis_floors, "rna")
    rna2_thresh_floor = _resolve_thresh_intensity_floor(cfg, analysis_floors, "rna2")

    # ---- Per-nucleus RANDOM-POSITION NULL for partner-at-rna1 coloc ---------
    # 2026-06-05 Brian: PIPELINE-NATIVE proper coloc statistic. GATED behind
    # cfg.foci.compute_partner_null_enrichment (requires compute_partner_intensity
    # so the partner channel is the one being sampled). For each nucleus the
    # partner (rna2 / QKI) intensity at the rna1 (MIAT) spots is compared to the
    # SAME spot count placed at random IN-NUCLEUS positions, repeated partner_null_n
    # times -> enrichment (obs/null_mean), z ((obs-null_mean)/null_sd), and a
    # per-image spot-count-weighted pooled enrichment / z / empirical-p. Disk
    # radius is the CANONICAL partner_null_disk_px (3.0 px = the validated
    # reference-script value), NOT the tiny compute_partner_intensity radius.
    # Reproduces qki_at_miat_null_ALLARMS_tm1.0.py. Default OFF -> no columns,
    # byte-equivalent output. Single-process is fine (compute_partner_intensity
    # already forces it); the null draws are numpy-batched for speed.
    compute_partner_null = (
        compute_partner_intensity
        and bool(getattr(cfg.foci, "compute_partner_null_enrichment", False))
    )
    # 2026-06-06 Brian: PIPELINE-NATIVE radial QKI-around-MIAT profile (separate
    # downstream figure). Sweeps concentric annuli around each rna1 spot; reuses
    # the SAME per-nucleus sampling block (nucleolus-excluded mask, same seed /
    # n) as the disk null but with its OWN rng so toggling it never perturbs the
    # disk-null draws (byte-identical pooled-null contract). Emitted ONLY via the
    # empty-default extra dict. Default OFF.
    compute_partner_radial = (
        compute_partner_intensity
        and bool(getattr(cfg.foci, "compute_partner_radial_profile", False))
    )
    # When EITHER feature is on we need the per-nucleus sampling block (nucleolus
    # detection + shared spot/mask prep). When radial is OFF this is exactly
    # ``compute_partner_null`` -> the legacy gate -> byte-identical behavior.
    _need_partner_block = compute_partner_null or compute_partner_radial
    # Surfaced-output carriers (None -> key never added to extra; defaults OFF).
    coloc_null_draws_df = None
    coloc_radial_df = None
    # Nucleolus-exclusion for the null sampling positions + observed spots.
    # When requested AND nucleolus is enabled, detect nucleoli ONCE here (the
    # post-loop nucleolus column block below REUSES this array, so detection
    # runs only once). The DAPI-poor nucleolar voids — which the partner (QKI)
    # also avoids — are removed from BOTH the random null positions and the
    # observed rna1 spots, so the enrichment cannot be inflated by mutual
    # nucleolar avoidance.
    _exclude_nucleolus_null = bool(
        getattr(cfg.foci, "exclude_nucleolus_from_partner_null", False)
    )
    _nucleolus_enabled_cfg = (
        getattr(cfg, "nucleolus", None) is not None
        and getattr(cfg.nucleolus, "enabled", False)
    )
    nucleolus_labels_precomputed = None
    if (_need_partner_block and _exclude_nucleolus_null and _nucleolus_enabled_cfg
            and n_after > 0):
        try:
            from ..nucleolus import NucleolusParams, detect_nucleoli
            _ncfg = cfg.nucleolus
            _np = NucleolusParams(
                intra_nuclear_percentile=float(_ncfg.intra_nuclear_percentile),
                min_area_um2=float(_ncfg.min_area_um2),
                max_area_frac_of_nucleus=float(_ncfg.max_area_frac_of_nucleus),
                closing_radius_px=int(_ncfg.closing_radius_px),
                min_border_distance_px=int(getattr(_ncfg, "min_border_distance_px", 3)),
            )
            _pix_um = float(voxel_xy_nm) / 1000.0 if voxel_xy_nm else 0.13
            nucleolus_labels_precomputed = detect_nucleoli(
                labels, dapi_2d, pixel_size_um=_pix_um, params=_np
            )
        except Exception as _nexc:
            print(
                f"  WARN: partner-null nucleolus detection failed on {img_name} "
                f"({type(_nexc).__name__}: {_nexc}); null uses whole nucleus."
            )
            nucleolus_labels_precomputed = None
    # RNG + disk stencil for the null (fixed seed -> deterministic).
    _null_n = int(getattr(cfg.foci, "partner_null_n", 1000))
    _null_disk_px = float(getattr(cfg.foci, "partner_null_disk_px", 3.0))
    _null_seed = int(getattr(cfg.foci, "partner_null_seed", 0))
    if compute_partner_null:
        _null_rng = np.random.default_rng(_null_seed)
        _null_dy, _null_dx = _disk_stencil(_null_disk_px)
        # Per-image pooled accumulators (spot-count weighted), matching the
        # reference script's pooling: obs_pool = sum(obs*n_sp)/sum(n_sp);
        # null_pool[it] = sum(null_stats[it]*n_sp)/sum(n_sp); empirical p from
        # the pooled null distribution.
        _null_obs_num = 0.0
        _null_w_den = 0.0
        _null_pool = np.zeros(_null_n, dtype=np.float64)
        _null_n_nuclei_used = 0
    if compute_partner_radial:
        # Radial annuli (outer edges µm -> px) + own rng (same SEED, separate
        # stream so the disk-null draws stay byte-identical) + per-ring
        # spot-count-weighted accumulators.
        _radial_bins_um = list(
            getattr(cfg.foci, "partner_radial_bins_um", [0.25, 0.5, 0.75, 1.0])
        )
        _pix_um_radial = float(voxel_xy_nm) / 1000.0 if voxel_xy_nm else 0.13
        _radial_bins_px = [
            float(b) / _pix_um_radial for b in _radial_bins_um if float(b) > 0
        ]
        _radial_stencils = _annulus_stencils(_radial_bins_px)
        _n_rings = len(_radial_stencils)
        _radial_rng = np.random.default_rng(_null_seed)
        _radial_obs_num = np.zeros(_n_rings, dtype=np.float64)
        _radial_nullmean_num = np.zeros(_n_rings, dtype=np.float64)
        _radial_nullsd_num = np.zeros(_n_rings, dtype=np.float64)
        _radial_w_den = np.zeros(_n_rings, dtype=np.float64)

    for nid in range(1, n_after + 1):
        rp = rp_by_id.get(nid, {})
        nucleus_area_px = int(rp.get("area", 0))
        perim_px = float(rp.get("perimeter", 0.0))
        area_um2 = nucleus_area_px * (voxel_xy_um ** 2)
        perimeter_um = perim_px * voxel_xy_um

        nuc_mask = labels == nid
        if nuc_mask.any():
            rna_vals = rna_2d[nuc_mask].astype(np.float64)
            rna2_vals = rna2_2d[nuc_mask].astype(np.float64)
            dapi_vals = dapi_2d[nuc_mask].astype(np.float64)
            rna_mean = float(rna_vals.mean())
            rna2_mean = float(rna2_vals.mean())
            sum_rna_intensity = float(rna_vals.sum())
            sum_rna2_intensity = float(rna2_vals.sum())
            dapi_mean = float(dapi_vals.mean())
            # ---- Per-nucleus MASK (Manders-style) colocalization ----------
            # 2026-05-28 Brian: bidirectional pixel-mask coloc on the rna ×
            # rna2 pixel pair WITHIN this nucleus mask, on the SAME shared
            # DAPI-locked plane the rest of the analysis uses. This is the
            # Manders/correlation view (DROPPED when rna_protein was routed
            # through rna_rna); it now lives in the shared core so BOTH modes
            # get it. ADDITIVE to (coexists with) the spot-spot coloc below.
            #
            # Masks use the SAME per-channel pixel-coloc thresholds already
            # resolved for this run (rna_thr_value / rna2_thr_value), passed as
            # overrides so the mask boundaries match the rest of the pipeline.
            #
            # Direction (compute_coloc_metrics convention; r=rna1, a=rna2):
            #   manders_m1 = sum(rna1 where rna2>=thr) / sum(rna1)
            #              = fraction of RNA1 intensity inside the RNA2 mask
            #                -> "RNA1 in RNA2"  (RNA2 masked, RNA1 measured)
            #   manders_m2 = sum(rna2 where rna1>=thr) / sum(rna2)
            #              = fraction of RNA2 intensity inside the RNA1 mask
            #                -> "RNA2 in RNA1"  (RNA1 masked, RNA2 measured)
            _cm = _metrics.compute_coloc_metrics(
                rna_vals, rna2_vals,
                thr_mode=pc_cfg.threshold_mode if pc_cfg is not None else "mad",
                k_mad=float(pc_cfg.k_mad) if pc_cfg is not None else 2.0,
                percentile=float(pc_cfg.percentile) if pc_cfg is not None else 80.0,
                r_thr_override=rna_thr_value,
                a_thr_override=rna2_thr_value,
            )
            # Above-floor nuclear sums (NaN when no global floor available)
            nuclear_above_floor_rna1 = (
                float(rna_above_floor[nuc_mask].sum()) if rna_above_floor is not None
                else float("nan")
            )
            nuclear_above_floor_rna2 = (
                float(rna2_above_floor[nuc_mask].sum()) if rna2_above_floor is not None
                else float("nan")
            )
        else:
            rna_mean = rna2_mean = sum_rna_intensity = sum_rna2_intensity = dapi_mean = float("nan")
            _cm = _metrics.compute_coloc_metrics(
                np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64),
            )
            nuclear_above_floor_rna1 = float("nan")
            nuclear_above_floor_rna2 = float("nan")

        if cyt_labels is not None:
            cyt_mask = (cyt_labels == nid) & (~nuc_mask)
            if cyt_mask.any():
                rna_cytoplasmic_mean = float(rna_2d[cyt_mask].astype(np.float64).mean())
                rna2_cytoplasmic_mean = float(rna2_2d[cyt_mask].astype(np.float64).mean())
                sum_rna_intensity_cyto = float(rna_2d[cyt_mask].astype(np.float64).sum())
                sum_rna2_intensity_cyto = float(rna2_2d[cyt_mask].astype(np.float64).sum())
                cyto_area_px = int(cyt_mask.sum())
                cytoplasmic_above_floor_rna1 = (
                    float(rna_above_floor[cyt_mask].sum()) if rna_above_floor is not None
                    else float("nan")
                )
                cytoplasmic_above_floor_rna2 = (
                    float(rna2_above_floor[cyt_mask].sum()) if rna2_above_floor is not None
                    else float("nan")
                )
            else:
                rna_cytoplasmic_mean = rna2_cytoplasmic_mean = float("nan")
                sum_rna_intensity_cyto = sum_rna2_intensity_cyto = 0.0
                cyto_area_px = 0
                cytoplasmic_above_floor_rna1 = (
                    0.0 if rna_above_floor is not None else float("nan")
                )
                cytoplasmic_above_floor_rna2 = (
                    0.0 if rna2_above_floor is not None else float("nan")
                )
        else:
            cyt_mask = None
            rna_cytoplasmic_mean = rna2_cytoplasmic_mean = float("nan")
            sum_rna_intensity_cyto = sum_rna2_intensity_cyto = 0.0
            cyto_area_px = 0
            cytoplasmic_above_floor_rna1 = float("nan")
            cytoplasmic_above_floor_rna2 = float("nan")

        # ---- Thresholded RNA intensity, per compartment, BOTH channels -----
        # (2026-06-02 Brian). Same threshold-and-integrate as rna_only, applied
        # to rna1 (rna_2d) AND rna2 (rna2_2d) at their respective floors. In
        # rna_protein mode rna2_2d is the antibody channel, so this is also the
        # protein-channel mirror Brian asked for.
        _empty_thresh = dict(
            thresh_total_intensity=float("nan"),
            thresh_mean_intensity=float("nan"),
            thresh_pos_area_px=0,
            thresh_pos_fraction=float("nan"),
        )
        _tn1 = _metrics.compute_thresholded_compartment_intensity(
            rna_2d, nuc_mask, rna_thresh_floor
        )
        _tn2 = _metrics.compute_thresholded_compartment_intensity(
            rna2_2d, nuc_mask, rna2_thresh_floor
        )
        if cyt_mask is not None:
            _tc1 = _metrics.compute_thresholded_compartment_intensity(
                rna_2d, cyt_mask, rna_thresh_floor
            )
            _tc2 = _metrics.compute_thresholded_compartment_intensity(
                rna2_2d, cyt_mask, rna2_thresh_floor
            )
        else:
            _tc1 = dict(_empty_thresh)
            _tc2 = dict(_empty_thresh)

        # N/C and frac_nuclear ratios for the above-floor variants. Mirror
        # the _nc_total semantics used for the raw sums above: ratio is NaN
        # when the denominator is 0 / NaN. frac_nuclear_above_floor is
        # nuclear / (nuclear + cytoplasmic) — bounded [0, 1] when both are
        # finite, NaN otherwise.
        def _nc_total_floor(n_sum: float, c_sum: float) -> float:
            if not (n_sum == n_sum and c_sum == c_sum):  # NaN check
                return float("nan")
            if c_sum > 0:
                return n_sum / c_sum
            return float("nan")

        def _frac_nuclear_floor(n_sum: float, c_sum: float) -> float:
            if not (n_sum == n_sum and c_sum == c_sum):  # NaN check
                return float("nan")
            tot = n_sum + c_sum
            if tot > 0:
                return n_sum / tot
            return float("nan")

        nc_ratio_above_floor_rna1 = _nc_total_floor(
            nuclear_above_floor_rna1, cytoplasmic_above_floor_rna1
        )
        nc_ratio_above_floor_rna2 = _nc_total_floor(
            nuclear_above_floor_rna2, cytoplasmic_above_floor_rna2
        )
        frac_nuclear_above_floor_rna1 = _frac_nuclear_floor(
            nuclear_above_floor_rna1, cytoplasmic_above_floor_rna1
        )
        frac_nuclear_above_floor_rna2 = _frac_nuclear_floor(
            nuclear_above_floor_rna2, cytoplasmic_above_floor_rna2
        )

        # "Cell" = nucleus + cytoplasm (matches rna_only semantics for the
        # mean_cell_total_intensity_fit metric Brian wants for both channels).
        cell_area_px = int(nucleus_area_px) + int(cyto_area_px)
        cell_total_intensity_rna1 = sum_rna_intensity + sum_rna_intensity_cyto
        cell_total_intensity_rna2 = sum_rna2_intensity + sum_rna2_intensity_cyto

        def _nc(nmean, cmean):
            if cmean and cmean > 0 and not math.isnan(cmean):
                return nmean / cmean
            return float("nan")
        rna_nc_ratio = _nc(rna_mean, rna_cytoplasmic_mean)
        rna2_nc_ratio = _nc(rna2_mean, rna2_cytoplasmic_mean)

        # N/C ratio of TOTAL pixel-sum intensity (not the mean-based ratio
        # above). Brian's figure suite plots both — the total-ratio is more
        # robust when nucleus and cytoplasm have very different areas.
        def _nc_total(n_sum, c_sum):
            if c_sum and c_sum > 0 and not math.isnan(c_sum):
                return n_sum / c_sum
            return float("nan")
        nc_ratio_total_intensity_rna1 = _nc_total(sum_rna_intensity, sum_rna_intensity_cyto)
        nc_ratio_total_intensity_rna2 = _nc_total(sum_rna2_intensity, sum_rna2_intensity_cyto)

        # Per-nucleus per-channel spot aggregates
        def _agg(sub):
            if sub is None or len(sub) == 0:
                return dict(
                    rna_spot_count=0,
                    nuclear_spot_count=0,
                    cyto_spot_count=0,
                    nuclear_spot_fraction=float("nan"),
                    nuclear_spot_density_per_um2=float("nan"),
                    mean_int=float("nan"),
                    total_int=float("nan"),
                    median_int=float("nan"),
                    int_cv=float("nan"),
                    median_nn_um=float("nan"),
                    paired_frac=float("nan"),
                    paired_count=0,
                )
            sub = sub.copy()
            nuc_spot_mask = sub.get("in_nucleus", pd.Series(dtype=bool)) == True
            cyt_spot_mask = sub.get("in_cytoplasm", pd.Series(dtype=bool)) == True
            n_in = int(nuc_spot_mask.sum())
            n_cy = int(cyt_spot_mask.sum())
            tot = n_in + n_cy
            nuc_frac = (n_in / float(tot)) if tot > 0 else float("nan")
            density = (tot / area_um2) if area_um2 > 0 else float("nan")
            if tot > 0 and "intensity_peak" in sub.columns:
                ipeaks = sub["intensity_peak"].astype(float)
                m_ = float(ipeaks.mean())
                t_ = float(ipeaks.sum())
                med_ = float(ipeaks.median())
                cv_ = (float(ipeaks.std()) / m_) if m_ > 0 else float("nan")
            else:
                m_ = t_ = med_ = cv_ = float("nan")
            if tot > 0 and "nn_distance_um" in sub.columns:
                nn_finite = sub["nn_distance_um"].replace([np.inf, -np.inf], np.nan).dropna()
                med_nn = float(nn_finite.median()) if len(nn_finite) else float("nan")
            else:
                med_nn = float("nan")
            if tot > 0 and f"paired_at_{pair_suffix}" in sub.columns:
                p_cnt = int(sub[f"paired_at_{pair_suffix}"].astype(int).sum())
                p_frac = p_cnt / float(tot)
            else:
                p_cnt = 0
                p_frac = float("nan")
            return dict(
                rna_spot_count=int(tot),
                nuclear_spot_count=int(n_in),
                cyto_spot_count=int(n_cy),
                nuclear_spot_fraction=nuc_frac,
                nuclear_spot_density_per_um2=density,
                mean_int=m_,
                total_int=t_,
                median_int=med_,
                int_cv=cv_,
                median_nn_um=med_nn,
                paired_frac=p_frac,
                paired_count=p_cnt,
            )

        a1 = _agg(spots1_by_nid.get(nid))
        a2 = _agg(spots2_by_nid.get(nid))

        # ---- Active transcription sites + mature mRNA per cell -----------
        # Active TS proxy: a spot that is BOTH in the nucleus AND has a
        # paired partner spot within pair_distance_um in the other channel.
        # For an exon-intron (or 5'-3' tiling) probe set, an "active TS"
        # appears as a co-localized punctum at the gene locus inside the
        # nucleus. Counting per nucleus on RNA1 (the canonical "primary"
        # channel) gives a per-cell active-TS count.
        # Mature mRNA proxy: cytoplasmic spots in the primary RNA channel
        # — these are exported transcripts. For exon-only probes this is
        # the mature mRNA pool; for intron probes this should be near 0
        # (intron retention/nuclear-localized signal).
        def _active_ts_count(sub):
            if sub is None or len(sub) == 0:
                return 0
            if "in_nucleus" not in sub.columns or f"paired_at_{pair_suffix}" not in sub.columns:
                return 0
            mask = (
                (sub["in_nucleus"].astype(bool) == True)
                & (sub[f"paired_at_{pair_suffix}"].astype(int) == 1)
            )
            return int(mask.sum())

        def _mature_mrna_count(sub):
            if sub is None or len(sub) == 0:
                return 0
            if "in_cytoplasm" not in sub.columns:
                return 0
            return int((sub["in_cytoplasm"].astype(bool) == True).sum())

        n_active_tss = _active_ts_count(spots1_by_nid.get(nid))
        n_active_tss_rna2 = _active_ts_count(spots2_by_nid.get(nid))
        n_mature_mrna_rna1 = _mature_mrna_count(spots1_by_nid.get(nid))
        n_mature_mrna_rna2 = _mature_mrna_count(spots2_by_nid.get(nid))

        # ---- Intensity-based, spot-centric, FLOOR-ROBUST coloc (per nucleus)
        # Mean partner-local-intensity over THIS nucleus's spots, plus an
        # enrichment ratio vs the whole-nucleus partner-channel mean:
        #   rna2_local_mean_at_rna1_spots = mean(partner_local @ this nuc's RNA1 spots)
        #   rna2_enrichment_at_rna1_spots = rna2_local_mean_at_rna1_spots / rna2_mean
        # >1 => RNA1 (intron) foci sit at partner-BRIGHT sites. Reciprocal for
        # RNA2 spots vs the RNA1 channel. (rna_protein relabels rna2->protein,
        # so these read protein_local_mean_at_rna1_spots /
        # protein_enrichment_at_rna1_spots etc.)
        def _partner_local_mean_over_spots(sub):
            if sub is None or len(sub) == 0 or "partner_local_mean_intensity" not in sub.columns:
                return float("nan")
            v = pd.to_numeric(sub["partner_local_mean_intensity"], errors="coerce").to_numpy()
            v = v[np.isfinite(v)]
            return float(v.mean()) if v.size else float("nan")

        def _enrichment(local_mean, nuc_channel_mean):
            if not (local_mean == local_mean):  # NaN
                return float("nan")
            if nuc_channel_mean and nuc_channel_mean > 0 and (nuc_channel_mean == nuc_channel_mean):
                return float(local_mean / nuc_channel_mean)
            return float("nan")

        rna2_local_mean_at_rna1_spots = _partner_local_mean_over_spots(spots1_by_nid.get(nid))
        rna1_local_mean_at_rna2_spots = _partner_local_mean_over_spots(spots2_by_nid.get(nid))
        # Partner channel whole-nucleus mean = rna2_mean for RNA1 spots (partner
        # is rna2), rna_mean for RNA2 spots (partner is rna1).
        rna2_enrichment_at_rna1_spots = _enrichment(rna2_local_mean_at_rna1_spots, rna2_mean)
        rna1_enrichment_at_rna2_spots = _enrichment(rna1_local_mean_at_rna2_spots, rna_mean)

        # ---- Per-nucleus RANDOM-POSITION NULL (partner @ rna1 spots) --------
        # 2026-06-05 Brian. observed = mean over THIS nucleus's rna1 (MIAT)
        # spots of [disk-mean partner (rna2/QKI) intensity]; null = same #spots
        # at random in-nucleus positions x partner_null_n, disk-sampled. The
        # nucleus's sampling pixels are the nucleus mask, MINUS nucleoli when
        # exclude_nucleolus_from_partner_null is on; observed rna1 spots inside a
        # nucleolus are likewise dropped. Pooled (spot-weighted) into the
        # per-image rollup. Mirrors qki_at_miat_null_ALLARMS_tm1.0.py.
        rna2_enrichment_vs_null_at_rna1_spots = float("nan")
        rna2_null_z_at_rna1_spots = float("nan")
        if _need_partner_block and nuc_mask.any():
            sub1 = spots1_by_nid.get(nid)
            if sub1 is not None and len(sub1) > 0 and {"y_px", "x_px"}.issubset(sub1.columns):
                # Sampling pixels = nucleus minus nucleoli (when on). SHARED by
                # the disk null AND the radial profile so they sample identically.
                _samp_mask = nuc_mask
                _nucleolus_in_nuc = None
                if nucleolus_labels_precomputed is not None:
                    _nucleolus_in_nuc = (nucleolus_labels_precomputed == nid) & nuc_mask
                    _nucleoplasm = nuc_mask & (~_nucleolus_in_nuc)
                    if _nucleoplasm.any():
                        _samp_mask = _nucleoplasm
                _nys, _nxs = np.where(_samp_mask)
                _scy = np.rint(sub1["y_px"].astype(float).to_numpy()).astype(np.intp)
                _scx = np.rint(sub1["x_px"].astype(float).to_numpy()).astype(np.intp)
                _scy = np.clip(_scy, 0, rna2_2d.shape[0] - 1)
                _scx = np.clip(_scx, 0, rna2_2d.shape[1] - 1)
                # Drop observed rna1 spots whose center is inside a nucleolus.
                if _nucleolus_in_nuc is not None:
                    _keep_sp = ~_nucleolus_in_nuc[_scy, _scx]
                    _scy = _scy[_keep_sp]
                    _scx = _scx[_keep_sp]
                if _scy.size > 0 and _nys.size > 0:
                    _rna2_2d_f = rna2_2d.astype(np.float64, copy=False)
                    if compute_partner_null:
                        _obs_stat, _null_stats = _partner_null_for_nucleus(
                            _rna2_2d_f,
                            _scy, _scx, _nys, _nxs,
                            _null_dy, _null_dx, _null_n, _null_rng,
                        )
                        if _null_stats.size:
                            _nmean = float(_null_stats.mean())
                            _nsd = float(_null_stats.std(ddof=1)) if _null_stats.size > 1 else 0.0
                            rna2_enrichment_vs_null_at_rna1_spots = (
                                _obs_stat / _nmean if _nmean > 0 else float("nan")
                            )
                            rna2_null_z_at_rna1_spots = (
                                (_obs_stat - _nmean) / _nsd if _nsd > 0 else float("nan")
                            )
                            _n_sp_used = int(_scy.size)
                            _null_obs_num += _obs_stat * _n_sp_used
                            _null_w_den += _n_sp_used
                            _null_pool += _null_stats * _n_sp_used
                            _null_n_nuclei_used += 1
                    if compute_partner_radial:
                        # Per-ring (obs, null_mean, null_sd, n_spots); accumulate
                        # spot-count-weighted into the per-(image, ring) rollup.
                        _rad = _radial_profile_for_nucleus(
                            _rna2_2d_f, _scy, _scx, _nys, _nxs,
                            _radial_stencils, _null_n, _radial_rng,
                        )
                        for _ri, (_ro, _rnm, _rnsd, _rnsp) in enumerate(_rad):
                            if _rnsp > 0 and _ro == _ro:  # finite observed
                                _radial_obs_num[_ri] += _ro * _rnsp
                                _radial_nullmean_num[_ri] += _rnm * _rnsp
                                _radial_nullsd_num[_ri] += _rnsd * _rnsp
                                _radial_w_den[_ri] += _rnsp

        # Per-nucleus row — column ordering: rna_only-compatible fields first
        # (so anything reading rna_only output still finds its columns), then
        # rna2/spot-coloc additions.
        nuc_row = {
            "image": img_name,
            "condition": condition,
            "secondary_only": sec_only,
            "experiment_id": "",
            "nucleus_id": int(nid),
            "nucleus_area_px": int(nucleus_area_px),
            # ---- RNA1 (rna) per-nucleus block — same names as rna_only ----
            "rna_mean_in_nucleus": rna_mean,
            "rna_nuclear_mean": rna_mean,
            "rna_cytoplasmic_mean": rna_cytoplasmic_mean,
            "rna_nc_ratio": rna_nc_ratio,
            "rna_spot_count": int(a1["rna_spot_count"]),
            "nuclear_spot_count": int(a1["nuclear_spot_count"]),
            "cyto_spot_count": int(a1["cyto_spot_count"]),
            "nuclear_spot_fraction": a1["nuclear_spot_fraction"],
            "nuclear_spot_density_per_um2": a1["nuclear_spot_density_per_um2"],
            "rna_spot_mean_intensity_bgc_blend": a1["mean_int"],
            "rna_spot_total_intensity_bgc_blend": a1["total_int"],
            "rna_spot_median_intensity_bgc_blend": a1["median_int"],
            "rna_spot_mean_intensity_fit": a1["mean_int"],
            "rna_spot_total_intensity_fit": a1["total_int"],
            "rna_spot_median_intensity_fit": a1["median_int"],
            "rna_spot_intensity_cv_fit": a1["int_cv"],
            "sum_rna_intensity": sum_rna_intensity,
            # ---- RNA2 per-nucleus block (rna_rna additions) --------------
            "rna2_mean_in_nucleus": rna2_mean,
            "rna2_nuclear_mean": rna2_mean,
            "rna2_cytoplasmic_mean": rna2_cytoplasmic_mean,
            "rna2_nc_ratio": rna2_nc_ratio,
            "n_spots_rna1": int(a1["rna_spot_count"]),
            "n_spots_rna2": int(a2["rna_spot_count"]),
            "nuclear_spot_count_rna2": int(a2["nuclear_spot_count"]),
            "cyto_spot_count_rna2": int(a2["cyto_spot_count"]),
            "nuclear_spot_fraction_rna2": a2["nuclear_spot_fraction"],
            "nuclear_spot_density_per_um2_rna2": a2["nuclear_spot_density_per_um2"],
            "rna2_spot_mean_intensity_fit": a2["mean_int"],
            "rna2_spot_total_intensity_fit": a2["total_int"],
            "rna2_spot_median_intensity_fit": a2["median_int"],
            "rna2_spot_intensity_cv_fit": a2["int_cv"],
            "sum_rna2_intensity": sum_rna2_intensity,
            # ---- Thresholded RNA intensity per compartment (2026-06-02) ----
            # Third intensity measurement (pixels with RAW value >= floor),
            # integrated separately in nucleus + cytoplasm, BOTH channels.
            "rna_thresh_total_intensity_nuclear": _tn1["thresh_total_intensity"],
            "rna_thresh_mean_intensity_nuclear": _tn1["thresh_mean_intensity"],
            "rna_thresh_pos_area_px_nuclear": int(_tn1["thresh_pos_area_px"]),
            "rna_thresh_pos_fraction_nuclear": _tn1["thresh_pos_fraction"],
            "rna_thresh_total_intensity_cyto": _tc1["thresh_total_intensity"],
            "rna_thresh_mean_intensity_cyto": _tc1["thresh_mean_intensity"],
            "rna_thresh_pos_area_px_cyto": int(_tc1["thresh_pos_area_px"]),
            "rna_thresh_pos_fraction_cyto": _tc1["thresh_pos_fraction"],
            "rna_thresh_floor": rna_thresh_floor,
            "rna2_thresh_total_intensity_nuclear": _tn2["thresh_total_intensity"],
            "rna2_thresh_mean_intensity_nuclear": _tn2["thresh_mean_intensity"],
            "rna2_thresh_pos_area_px_nuclear": int(_tn2["thresh_pos_area_px"]),
            "rna2_thresh_pos_fraction_nuclear": _tn2["thresh_pos_fraction"],
            "rna2_thresh_total_intensity_cyto": _tc2["thresh_total_intensity"],
            "rna2_thresh_mean_intensity_cyto": _tc2["thresh_mean_intensity"],
            "rna2_thresh_pos_area_px_cyto": int(_tc2["thresh_pos_area_px"]),
            "rna2_thresh_pos_fraction_cyto": _tc2["thresh_pos_fraction"],
            "rna2_thresh_floor": rna2_thresh_floor,
            # ---- Spot-spot colocalization ---------------------------------
            f"median_nn_distance_rna1_um": a1["median_nn_um"],
            f"median_nn_distance_rna2_um": a2["median_nn_um"],
            f"paired_fraction_rna1_at_{pair_suffix}": a1["paired_frac"],
            f"paired_fraction_rna2_at_{pair_suffix}": a2["paired_frac"],
            f"paired_spot_count_rna1_at_{pair_suffix}": int(a1["paired_count"]),
            f"paired_spot_count_rna2_at_{pair_suffix}": int(a2["paired_count"]),
            # ---- Per-nucleus MASK (Manders-style) colocalization ----------
            # 2026-05-28 Brian: bidirectional pixel-mask coloc (Manders M1/M2 +
            # correlation/overlap), additive to the spot-spot view above. From
            # ``compute_coloc_metrics`` on (rna1, rna2) within this nucleus,
            # thresholded at the run's rna_thr_value / rna2_thr_value.
            #
            # DIRECTION is explicit in the column name "<measured>_in_<mask>":
            #   manders_rna1_in_rna2 = M1 = sum(RNA1 where RNA2>=thr)/sum(RNA1)
            #       -> RNA2 is the MASK, RNA1 is measured ("how much RNA1 sits
            #          inside the RNA2-positive region").
            #   manders_rna2_in_rna1 = M2 = sum(RNA2 where RNA1>=thr)/sum(RNA2)
            #       -> RNA1 is the MASK, RNA2 is measured.
            # (_relabel_rna2_to_protein maps rna2->protein, leaving rna1 as-is,
            #  so rna_protein reads manders_rna1_in_protein / manders_protein_in_rna1.)
            "manders_rna1_in_rna2": _cm["manders_m1"],
            "manders_rna2_in_rna1": _cm["manders_m2"],
            "coloc_pearson_r_rna1_rna2": _cm["pearson_r"],
            "coloc_spearman_rho_rna1_rna2": _cm["spearman_rho"],
            "coloc_li_icq_rna1_rna2": _cm["li_icq"],
            "coloc_cosine_overlap_rna1_rna2": _cm["cosine_overlap"],
            "coloc_jaccard_rna1_rna2": _cm["jaccard"],
            "coloc_dice_rna1_rna2": _cm["dice"],
            "coloc_both_frac_rna1_rna2": _cm["both_frac"],
            "coloc_frac_above_thr_rna1": _cm["rna_frac_above_thr"],
            "coloc_frac_above_thr_rna2": _cm["ab_frac_above_thr"],
            # Enrichment of one channel inside the OTHER's positive mask.
            #   rna2_enrich_in_rna1_high: mean(RNA2 | RNA1>=thr)/mean(RNA2 | RNA1<thr)
            #   rna1_enrich_in_rna2_high: mean(RNA1 | RNA2>=thr)/mean(RNA1 | RNA2<thr)
            "rna2_enrich_in_rna1_high": _cm["ab_enrich_in_rna_high"],
            "rna1_enrich_in_rna2_high": _cm["rna_enrich_in_ab_high"],
            "coloc_mask_thr_rna1": _cm["rna_thr"],
            "coloc_mask_thr_rna2": _cm["ab_thr"],
            # ---- Intensity-based, spot-centric, FLOOR-ROBUST coloc ---------
            # 2026-05-29 Brian. GATED behind cfg.foci.compute_partner_intensity
            # (default OFF; see sampling block above). When OFF these columns
            # are NOT emitted, so the per-nucleus output is byte-equivalent to
            # the pre-feature path. When ON they are injected just below.
            # ---- Per-CELL (nucleus + cytoplasm) intensity totals -----------
            # Brian wants total RNA intensity per nucleus AND per cell for
            # BOTH channels. "_cell_" = nucleus + voronoi cytoplasm.
            "cell_total_intensity_rna1": cell_total_intensity_rna1,
            "cell_total_intensity_rna2": cell_total_intensity_rna2,
            "cell_area_px": int(cell_area_px),
            "sum_rna_intensity_cyto": sum_rna_intensity_cyto,
            "sum_rna2_intensity_cyto": sum_rna2_intensity_cyto,
            # ---- N/C ratios of TOTAL intensity (pixel sum) ----------------
            # Brian's figure suite requires both the mean-based N/C ratio
            # (above) AND the total-intensity N/C ratio. The total ratio is
            # less sensitive to area mismatches between nucleus + cytoplasm.
            "nuclear_total_intensity_rna1": sum_rna_intensity,
            "nuclear_total_intensity_rna2": sum_rna2_intensity,
            "cytoplasmic_total_intensity_rna1": sum_rna_intensity_cyto,
            "cytoplasmic_total_intensity_rna2": sum_rna2_intensity_cyto,
            "nc_ratio_total_intensity_rna1": nc_ratio_total_intensity_rna1,
            "nc_ratio_total_intensity_rna2": nc_ratio_total_intensity_rna2,
            # ---- Above-floor intensity variants (Brian/Sam 2026-05-20) ----
            # Pixels below the publication-image contrast floor (per channel)
            # are clipped to zero before summing — same threshold the eye
            # uses to judge nuclear-vs-cytoplasmic signal in the pub render.
            # Only added when output.apply_pub_contrast_floor_to_analysis is
            # True. When the resolved floor is unavailable, values are NaN.
            **(
                {
                    "nuclear_above_floor_intensity_rna1": nuclear_above_floor_rna1,
                    "nuclear_above_floor_intensity_rna2": nuclear_above_floor_rna2,
                    "cytoplasmic_above_floor_intensity_rna1": cytoplasmic_above_floor_rna1,
                    "cytoplasmic_above_floor_intensity_rna2": cytoplasmic_above_floor_rna2,
                    "nc_ratio_above_floor_intensity_rna1": nc_ratio_above_floor_rna1,
                    "nc_ratio_above_floor_intensity_rna2": nc_ratio_above_floor_rna2,
                    "frac_nuclear_above_floor_intensity_rna1": frac_nuclear_above_floor_rna1,
                    "frac_nuclear_above_floor_intensity_rna2": frac_nuclear_above_floor_rna2,
                }
                if apply_floor
                else {}
            ),
            # ---- Active TS + mature mRNA proxies --------------------------
            # Brian's exon/intron design: active TS = nuclear + paired spot
            # (a co-localized punctum at the gene locus). Mature mRNA =
            # cytoplasmic spots in the primary channel.
            "n_nuclear_rna1_rna2_overlap_per_nucleus": int(n_active_tss),
            "n_nuclear_rna2_rna1_overlap_per_nucleus": int(n_active_tss_rna2),
            "n_cytoplasmic_rna1_spots_per_cell": int(n_mature_mrna_rna1),
            "n_cytoplasmic_rna2_spots_per_cell": int(n_mature_mrna_rna2),
            # ---- Common metadata ------------------------------------------
            "cyto_area_px": int(cyto_area_px),
            "cyto_estimation_method": "voronoi" if cyt_labels is not None else "",
            "n_voxels": int(nucleus_area_px),
            "n_pix": int(nucleus_area_px),
            "n_z_slices": int(img.n_z),
            "z_mode": z_mode,
            "z_range": f"{z_start}-{z_end}" if (z_start and z_end) else "",
            "voxel_xy_um": voxel_xy_um,
            "voxel_z_um": voxel_z_um,
            "rna_threshold_value": rna_thr_value,
            "rna2_threshold_value": rna2_thr_value,
            "rna_frac_above_thr": float((rna_2d >= rna_thr_value).sum()) / float(rna_2d.size),
            "rna2_frac_above_thr": float((rna2_2d >= rna2_thr_value).sum()) / float(rna2_2d.size),
            "dapi_mean_in_nucleus": dapi_mean,
        }
        # GATED intensity-coloc per-nucleus columns (default OFF). Inserted here
        # (right after coloc_mask_thr_rna2 in the legacy ordering) only when
        # cfg.foci.compute_partner_intensity is True, so the OFF path is
        # byte-equivalent to the pre-feature output.
        if compute_partner_intensity:
            nuc_row["rna2_local_mean_at_rna1_spots"] = rna2_local_mean_at_rna1_spots
            nuc_row["rna2_enrichment_at_rna1_spots"] = rna2_enrichment_at_rna1_spots
            nuc_row["rna1_local_mean_at_rna2_spots"] = rna1_local_mean_at_rna2_spots
            nuc_row["rna1_enrichment_at_rna2_spots"] = rna1_enrichment_at_rna2_spots
        # GATED random-position-null coloc per-nucleus columns (default OFF).
        # Only emitted when cfg.foci.compute_partner_null_enrichment is True
        # (and compute_partner_intensity is on), so the OFF path stays byte-
        # equivalent. rna_protein relabels rna2->protein.
        if compute_partner_null:
            nuc_row["rna2_enrichment_vs_null_at_rna1_spots"] = rna2_enrichment_vs_null_at_rna1_spots
            nuc_row["rna2_null_z_at_rna1_spots"] = rna2_null_z_at_rna1_spots
        nuc_rows.append(nuc_row)

        # Morphology row (per-nucleus, single block — shape is channel-agnostic)
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

    # ---- Per-image pooled random-position-null rollup ----------------------
    # 2026-06-05 Brian. Spot-count-weighted pool of the per-nucleus observed
    # stat + null distribution (matching qki_at_miat_null_ALLARMS_tm1.0.py):
    #   obs_pool   = sum(obs_n * n_sp_n) / sum(n_sp_n)
    #   null_pool  = sum(null_stats_n * n_sp_n) / sum(n_sp_n)  (per iteration)
    #   enrichment = obs_pool / mean(null_pool)
    #   z          = (obs_pool - mean(null_pool)) / std(null_pool)
    #   empirical p= (#{null_pool >= obs_pool} + 1) / (n_null + 1)
    # Only computed when the feature is on; injected into per_image below.
    pooled_null_enrichment = float("nan")
    pooled_null_z = float("nan")
    pooled_null_p_empirical = float("nan")
    pooled_null_obs = float("nan")
    pooled_null_mean = float("nan")
    if compute_partner_null and _null_w_den > 0:
        _obs_pool = _null_obs_num / _null_w_den
        _np_pool = _null_pool / _null_w_den
        _np_mean = float(_np_pool.mean())
        _np_sd = float(_np_pool.std(ddof=1)) if _np_pool.size > 1 else 0.0
        pooled_null_obs = float(_obs_pool)
        pooled_null_mean = _np_mean
        pooled_null_enrichment = (_obs_pool / _np_mean) if _np_mean > 0 else float("nan")
        pooled_null_z = ((_obs_pool - _np_mean) / _np_sd) if _np_sd > 0 else float("nan")
        pooled_null_p_empirical = float(
            (np.sum(_np_pool >= _obs_pool) + 1) / (_null_n + 1)
        )
        # 2026-06-06 Brian: surface the pooled null vector + pooled observed for
        # the downstream null-distribution overlay. Reuses the SAME _np_pool /
        # _obs_pool computed above (NO recompute) -> ~zero cost. Gated; default
        # OFF -> the key is never added to extra (byte-identical carrier).
        if bool(getattr(cfg.foci, "save_partner_null_draws", False)):
            coloc_null_draws_df = pd.DataFrame(
                {
                    "image": img_name,
                    "condition": condition,
                    "iter": np.arange(_null_n, dtype=int),
                    "pooled_null_value": _np_pool,
                    "pooled_obs": float(_obs_pool),
                }
            )

    # ---- Per-image pooled radial QKI-around-MIAT profile -------------------
    # 2026-06-06 Brian. Spot-count-weighted pool of the per-nucleus per-ring
    # (obs, null_mean, null_sd) into one row per (image, ring). enrichment =
    # obs/null_mean; z = (obs-null_mean)/null_sd. Only built when the feature
    # is on AND at least one ring accumulated spots.
    if compute_partner_radial and float(_radial_w_den.sum()) > 0:
        _rad_rows = []
        for _ri in range(_n_rings):
            _w = float(_radial_w_den[_ri])
            if _w <= 0:
                continue
            _o = _radial_obs_num[_ri] / _w
            _nm = _radial_nullmean_num[_ri] / _w
            _nsd = _radial_nullsd_num[_ri] / _w
            _enr = (_o / _nm) if _nm > 0 else float("nan")
            _z = ((_o - _nm) / _nsd) if _nsd > 0 else float("nan")
            _rad_rows.append(
                {
                    "image": img_name,
                    "condition": condition,
                    "ring_um": float(_radial_bins_um[_ri]),
                    "obs_mean": float(_o),
                    "null_mean": float(_nm),
                    "null_sd": float(_nsd),
                    "enrichment": float(_enr),
                    "z": float(_z),
                    "n_spots": int(_w),
                }
            )
        if _rad_rows:
            coloc_radial_df = pd.DataFrame(_rad_rows)

    # ---- Per-spot rows (combined into one DataFrame with ``channel`` col) --
    def _emit_spot_rows(df: pd.DataFrame, label: str):
        nonlocal spot_global_id
        if df is None or len(df) == 0:
            return
        for _, r in df.iterrows():
            spot_global_id += 1
            x_px = int(r.get("x_px", 0))
            y_px = int(r.get("y_px", 0))
            z_slice = int(r.get("z_slice", 0))
            ipeak = float(r.get("intensity_peak", float("nan")))
            nid_at = int(r.get("nucleus_id", 0))
            in_nuc_flag = bool(r.get("in_nucleus", False))
            in_cyt_flag = bool(r.get("in_cytoplasm", False))
            # Use the MEASURED per-spot diameter (added on spots1_df/spots2_df
            # above via _measure_spot_diameter_um). Fall back to the nominal
            # default only if the column is somehow missing or non-finite.
            spot_diam_um = float(r.get("spot_diameter_um", default_spot_diameter_um))
            if not (spot_diam_um == spot_diam_um and spot_diam_um > 0):  # NaN or non-positive
                spot_diam_um = default_spot_diameter_um
            spot_fwhm_px_val = spot_diam_um / max(voxel_xy_um, 1e-6)
            spot_area_px_val = math.pi * (spot_fwhm_px_val / 2.0) ** 2
            spot_row = {
                "image": img_name,
                "condition": condition,
                "secondary_only": sec_only,
                "experiment_id": "",
                "channel": label,
                "spot_id": int(spot_global_id),
                "nucleus_id": nid_at,
                "in_nucleus": int(in_nuc_flag),
                "in_cytoplasm": int(in_cyt_flag),
                "x_px": x_px,
                "y_px": y_px,
                "z_slice": z_slice,
                "z_position_um": z_slice * voxel_z_um,
                "spot_peak_intensity": ipeak,
                "quality": ipeak,
                "spot_fwhm_px": spot_fwhm_px_val,
                "spot_diameter_um": spot_diam_um,
                "spot_area_px": spot_area_px_val,
                "integrated_intensity_fit": ipeak,
                "nn_distance_um": float(r.get("nn_distance_um", float("nan"))),
                f"paired_at_{pair_suffix}": int(r.get(f"paired_at_{pair_suffix}", 0)),
            }
            # GATED per-spot intensity-coloc column (default OFF). Omitted when
            # cfg.foci.compute_partner_intensity is False so spot_metrics.csv is
            # byte-equivalent to the pre-feature output.
            if compute_partner_intensity:
                # Intensity-based floor-robust coloc: mean RAW intensity of the
                # OTHER channel in a spot-radius disk at this spot's centroid.
                spot_row["partner_local_mean_intensity"] = float(
                    r.get("partner_local_mean_intensity", float("nan"))
                )
            spot_rows.append(spot_row)

    _emit_spot_rows(spots1_df, "rna1")
    _emit_spot_rows(spots2_df, "rna2")

    nuclei_df = pd.DataFrame(nuc_rows)
    spots_out_df = pd.DataFrame(spot_rows)
    morph_df = pd.DataFrame(morph_rows)

    # ---- Drop floater spots (optional, default True) -----------------------
    # 2026-05-21 Brian: spots with in_nucleus=False AND in_cytoplasm=False
    # are bare-field detections (off any segmented cell). Typically
    # autofluorescence/junk the LoG kernel found between cells. Drop them
    # so the spots_df + downstream counts only reflect within-cell signal.
    # Per-nucleus and per-cell aggregations already filter implicitly; this
    # cleans up the top-level total_spots_rna1/rna2 counts in
    # per_image_summary.csv and removes the floaters from spot_metrics.csv.
    # ---- "Exploded speck" outlier filter (off by default; tighter rule) ---
    # 2026-05-21 Brian: drop ONLY spots whose peak intensity is wildly
    # above the bulk of the rest. Compares per-image, per-channel peak vs
    # the 95th-percentile of peaks in that same image+channel. If a spot
    # is > N× the p95, it's an "absurd" outlier — much brighter than even
    # the brightest 5% of REAL spots — and is treated as a dust speck.
    # Does NOT drop based on absolute intensity (real spots can be very
    # bright) or FWHM (real biology can be small).
    _max_ratio = float(getattr(cfg.foci, "max_peak_over_p95_ratio", 0.0) or 0.0)
    if _max_ratio > 0 and len(spots_out_df) and "spot_peak_intensity" in spots_out_df.columns:
        grp_keys = [k for k in ("image", "channel") if k in spots_out_df.columns]
        if grp_keys:
            _before = len(spots_out_df)
            pk = spots_out_df["spot_peak_intensity"].astype(float)
            p95 = spots_out_df.groupby(grp_keys)["spot_peak_intensity"].transform(
                lambda s: float(np.percentile(s, 95)) if len(s) > 0 else 0.0
            )
            _keep = pk <= (p95 * _max_ratio + 1e-9)
            spots_out_df = spots_out_df.loc[_keep, :].reset_index(drop=True)
            _after = len(spots_out_df)
            _dropped = _before - _after
            if _dropped > 0:
                try:
                    from rich.console import Console as _C
                    _C().print(
                        f"  [dim]speck filter: dropped {_dropped} spots with "
                        f"peak > {_max_ratio}× per-image p95 (absurd outliers)[/dim]"
                    )
                except Exception:
                    pass
            # Mirror to channel-specific dataframes
            if "channel" in spots_out_df.columns and "spot_id" in spots_out_df.columns:
                _k1 = set(spots_out_df.loc[spots_out_df.channel == "rna1", "spot_id"].tolist())
                _k2 = set(spots_out_df.loc[spots_out_df.channel == "rna2", "spot_id"].tolist())
                if "spot_id" in spots1_df.columns:
                    spots1_df = spots1_df.loc[spots1_df.spot_id.isin(_k1), :].reset_index(drop=True)
                if "spot_id" in spots2_df.columns:
                    spots2_df = spots2_df.loc[spots2_df.spot_id.isin(_k2), :].reset_index(drop=True)

    _drop_floaters = getattr(cfg.foci, "drop_floater_spots", True)
    if _drop_floaters and len(spots_out_df) and "in_nucleus" in spots_out_df.columns and "in_cytoplasm" in spots_out_df.columns:
        _before = len(spots_out_df)
        _keep = (spots_out_df["in_nucleus"].astype(bool) | spots_out_df["in_cytoplasm"].astype(bool))
        spots_out_df = spots_out_df.loc[_keep, :].reset_index(drop=True)
        _after = len(spots_out_df)
        _dropped = _before - _after
        if _dropped > 0:
            try:
                from rich.console import Console as _C
                _C().print(
                    f"  [dim]drop_floater_spots: kept {_after}/{_before} "
                    f"({100*_after/_before:.1f}%); dropped {_dropped} bare-field detections[/dim]"
                )
            except Exception:
                pass
        # Also filter the per-channel detection-source dataframes so any
        # subsequent ops downstream see the same set. spots1_df / spots2_df
        # are referenced by qc rendering — keep them aligned.
        if "channel" in spots_out_df.columns:
            _keep1_ids = set(spots_out_df.loc[spots_out_df.channel == "rna1", "spot_id"].tolist()) if "spot_id" in spots_out_df.columns else None
            _keep2_ids = set(spots_out_df.loc[spots_out_df.channel == "rna2", "spot_id"].tolist()) if "spot_id" in spots_out_df.columns else None
            if _keep1_ids is not None and "spot_id" in spots1_df.columns:
                spots1_df = spots1_df.loc[spots1_df.spot_id.isin(_keep1_ids), :].reset_index(drop=True)
            if _keep2_ids is not None and "spot_id" in spots2_df.columns:
                spots2_df = spots2_df.loc[spots2_df.spot_id.isin(_keep2_ids), :].reset_index(drop=True)

    # ---- Nucleolus + chromatin (optional) ----------------------------------
    # 2026-05-21 Brian: when cfg.nucleolus.enabled, detect DAPI-low subnuclear
    # regions and add nucleolus / chromatin columns to nuclei_df + spots_df.
    # Pure-additive: no existing columns are removed or renamed; if disabled
    # this block is a no-op.
    nucleolus_enabled = getattr(cfg, "nucleolus", None) is not None and getattr(cfg.nucleolus, "enabled", False)
    nucleolus_labels_for_qc = None
    if nucleolus_enabled:
        try:
            from ..nucleolus import (
                NucleolusParams,
                detect_nucleoli,
                chromatin_metrics_per_nucleus,
                classify_spots_by_subnuclear_region,
                render_nucleolus_overlay,
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
            # Reuse the labels already detected for the partner-null exclusion
            # (computed once before the per-nucleus loop) so detection runs only
            # once per image. Same params (the precompute used the same NucleolusParams),
            # so the emitted nucleolus columns are byte-identical either way.
            if nucleolus_labels_precomputed is not None:
                nucleolus_labels = nucleolus_labels_precomputed
            else:
                nucleolus_labels = detect_nucleoli(
                    labels, dapi_2d, pixel_size_um=_pix_um, params=_params
                )
            nucleolus_labels_for_qc = nucleolus_labels  # exposed via qc dict below
            # Per-spot in_nucleolus + refined in_nucleus_excluding_nucleolus
            if len(spots_out_df) and "x_px" in spots_out_df.columns:
                spots_out_df = classify_spots_by_subnuclear_region(
                    spots_out_df, labels, nucleolus_labels
                )
            # Merge chromatin metrics into nuclei_df by nucleus_id
            chrom_df = chromatin_metrics_per_nucleus(labels, dapi_2d, nucleolus_labels)
            if len(chrom_df) and len(nuclei_df) and "nucleus_id" in nuclei_df.columns:
                # Drop duplicate nucleus_id/area columns the merge would create
                _to_drop = [c for c in ["nucleus_area_px"] if c in chrom_df.columns and c in nuclei_df.columns]
                chrom_df = chrom_df.drop(columns=_to_drop)
                nuclei_df = nuclei_df.merge(chrom_df, on="nucleus_id", how="left")
        except Exception as _exc:
            import traceback as _tb
            print(
                f"  WARN: nucleolus detection failed on {img_name} "
                f"({type(_exc).__name__}: {_exc}); continuing without nucleolus cols.\n"
                f"{_tb.format_exc()}"
            )

    # ---- Per-image summary -------------------------------------------------
    # 2026-05-22 Brian: image-level total_spots_rna1/rna2 report only in-cell
    # spots (in_nucleus OR in_cytoplasm). Floaters still live in spot_metrics
    # for audit, but the headline totals must match what the per-nucleus
    # metrics count (and what the QC overlay now displays) so the dashboards
    # are consistent end-to-end.
    def _in_cell_count(df: pd.DataFrame) -> int:
        if df is None or len(df) == 0:
            return 0
        if "in_nucleus" in df.columns and "in_cytoplasm" in df.columns:
            return int((df["in_nucleus"].astype(bool) | df["in_cytoplasm"].astype(bool)).sum())
        return int(len(df))
    total_spots1 = _in_cell_count(spots1_df)
    total_spots2 = _in_cell_count(spots2_df)
    # Also track raw detection counts (incl. floaters) for transparency.
    total_spots1_raw = int(len(spots1_df))
    total_spots2_raw = int(len(spots2_df))
    n_floaters_rna1 = total_spots1_raw - total_spots1
    n_floaters_rna2 = total_spots2_raw - total_spots2
    paired1_total = int(np.asarray(paired1, dtype=int).sum()) if paired1.size else 0
    paired2_total = int(np.asarray(paired2, dtype=int).sum()) if paired2.size else 0
    img_paired_frac_1 = (paired1_total / float(total_spots1)) if total_spots1 > 0 else float("nan")
    img_paired_frac_2 = (paired2_total / float(total_spots2)) if total_spots2 > 0 else float("nan")

    def _finite_median(arr: np.ndarray) -> float:
        if arr.size == 0:
            return float("nan")
        a = arr[np.isfinite(arr)]
        return float(np.median(a)) if a.size else float("nan")

    img_median_nn_1 = _finite_median(nn1)
    img_median_nn_2 = _finite_median(nn2)

    # Helper: compute mean/median/cv for an arbitrary column of nuclei_df,
    # optionally filtered to rows where ``cnt_col > 0`` (matches rna_only's
    # "only count expressing cells" rule for per-spot intensity aggregates).
    def _img_stats(col_: str, cnt_col_: Optional[str] = None) -> Tuple[float, float, float]:
        if len(nuclei_df) == 0 or col_ not in nuclei_df.columns:
            return float("nan"), float("nan"), float("nan")
        ser = pd.to_numeric(nuclei_df[col_], errors="coerce")
        if cnt_col_ and cnt_col_ in nuclei_df.columns:
            mask = pd.to_numeric(nuclei_df[cnt_col_], errors="coerce") > 0
            ser = ser[mask]
        vals = ser.dropna().tolist()
        if not vals:
            return float("nan"), float("nan"), float("nan")
        m_ = sum(vals) / float(len(vals))
        med_ = _median(vals)
        if len(vals) > 1 and m_ > 0:
            var_ = sum((v - m_) ** 2 for v in vals) / float(len(vals) - 1)
            cv_ = math.sqrt(var_) / m_
        else:
            cv_ = float("nan")
        return m_, med_, cv_

    if len(nuclei_df) > 0:
        counts1 = nuclei_df["rna_spot_count"].astype(int).tolist()
        counts2 = nuclei_df["n_spots_rna2"].astype(int).tolist()
        n = len(counts1)
        m1 = sum(counts1) / float(n) if n > 0 else 0.0
        m2 = sum(counts2) / float(n) if n > 0 else 0.0
        # CV of spots-per-nucleus, both channels
        if n > 1 and m1 > 0:
            sd1 = math.sqrt(sum((v - m1) ** 2 for v in counts1) / float(n - 1))
            cv_count_1 = sd1 / m1
        else:
            cv_count_1 = float("nan")
        if n > 1 and m2 > 0:
            sd2 = math.sqrt(sum((v - m2) ** 2 for v in counts2) / float(n - 1))
            cv_count_2 = sd2 / m2
        else:
            cv_count_2 = float("nan")

        # Per-channel intensity stats — spot total per cell (only-expressing)
        mean_tot_fit_1, med_tot_fit_1, cv_tot_fit_1 = _img_stats(
            "rna_spot_total_intensity_fit", "rna_spot_count")
        mean_tot_fit_2, med_tot_fit_2, cv_tot_fit_2 = _img_stats(
            "rna2_spot_total_intensity_fit", "n_spots_rna2")

        # Per-cell TOTAL RNA intensity (raw pixel sum, nucleus+cyto, BOTH
        # channels) — NOT spot-only. Always defined (no expressing-only
        # filter), since this is a pixel measurement.
        mean_cell_int_1, med_cell_int_1, cv_cell_int_1 = _img_stats(
            "cell_total_intensity_rna1")
        mean_cell_int_2, med_cell_int_2, cv_cell_int_2 = _img_stats(
            "cell_total_intensity_rna2")
        # Per-NUCLEUS TOTAL RNA intensity (raw pixel sum, nucleus only)
        mean_nuc_int_1, med_nuc_int_1, _ = _img_stats("sum_rna_intensity")
        mean_nuc_int_2, med_nuc_int_2, _ = _img_stats("sum_rna2_intensity")
        # N/C-ratio image-level rollups, both channels
        mean_nc_total_1, med_nc_total_1, _ = _img_stats("nc_ratio_total_intensity_rna1")
        mean_nc_total_2, med_nc_total_2, _ = _img_stats("nc_ratio_total_intensity_rna2")
        # ---- Above-floor intensity image-level rollups (Brian/Sam) -----
        # Only computed when the per-nucleus columns exist (i.e. when
        # apply_pub_contrast_floor_to_analysis is True). When the floor was
        # unavailable per-channel, _img_stats already returns NaN for an
        # all-NaN series.
        if apply_floor:
            mean_nuc_af_1, med_nuc_af_1, _ = _img_stats("nuclear_above_floor_intensity_rna1")
            mean_nuc_af_2, med_nuc_af_2, _ = _img_stats("nuclear_above_floor_intensity_rna2")
            mean_cyt_af_1, med_cyt_af_1, _ = _img_stats("cytoplasmic_above_floor_intensity_rna1")
            mean_cyt_af_2, med_cyt_af_2, _ = _img_stats("cytoplasmic_above_floor_intensity_rna2")
            mean_nc_af_1, med_nc_af_1, _ = _img_stats("nc_ratio_above_floor_intensity_rna1")
            mean_nc_af_2, med_nc_af_2, _ = _img_stats("nc_ratio_above_floor_intensity_rna2")
            mean_frac_af_1, med_frac_af_1, _ = _img_stats("frac_nuclear_above_floor_intensity_rna1")
            mean_frac_af_2, med_frac_af_2, _ = _img_stats("frac_nuclear_above_floor_intensity_rna2")
        # Active-TS + mature-mRNA image-level rollups
        mean_active_tss, med_active_tss, _ = _img_stats("n_nuclear_rna1_rna2_overlap_per_nucleus")
        mean_mature_1, med_mature_1, _ = _img_stats("n_cytoplasmic_rna1_spots_per_cell")
        mean_mature_2, med_mature_2, _ = _img_stats("n_cytoplasmic_rna2_spots_per_cell")
        # Intensity-based, spot-centric, FLOOR-ROBUST coloc image-level rollups
        # (mean over nuclei of the per-nucleus partner-local mean + enrichment).
        mean_partner_local_at_rna1, _, _ = _img_stats("rna2_local_mean_at_rna1_spots")
        mean_partner_enrich_at_rna1, _, _ = _img_stats("rna2_enrichment_at_rna1_spots")
        mean_partner_local_at_rna2, _, _ = _img_stats("rna1_local_mean_at_rna2_spots")
        mean_partner_enrich_at_rna2, _, _ = _img_stats("rna1_enrichment_at_rna2_spots")

        # frac_nuclei_with_ge_X_spots for both channels
        def _frac(ct, k):
            return round(sum(1 for v in ct if v >= k) / float(n), 4) if n > 0 else 0.0

        # Nuclear-vs-cyto stratification rollups, per channel
        # frac_nuclear = mean over nuclei of nuclear_spot_fraction
        nuc_frac_1_vals = [v for v in nuclei_df.get("nuclear_spot_fraction", pd.Series(dtype=float)).tolist()
                            if isinstance(v, (int, float)) and v == v]
        nuc_frac_2_vals = [v for v in nuclei_df.get("nuclear_spot_fraction_rna2", pd.Series(dtype=float)).tolist()
                            if isinstance(v, (int, float)) and v == v]
        # Per-image-level fractions: total nuclear spots / total spots
        if total_spots1 > 0:
            nuclear_spots_1 = int(
                pd.to_numeric(nuclei_df.get("nuclear_spot_count", pd.Series(dtype=int)), errors="coerce").fillna(0).sum())
            cyto_spots_1 = int(
                pd.to_numeric(nuclei_df.get("cyto_spot_count", pd.Series(dtype=int)), errors="coerce").fillna(0).sum())
            frac_nuclear_image_1 = nuclear_spots_1 / float(total_spots1)
        else:
            nuclear_spots_1 = cyto_spots_1 = 0
            frac_nuclear_image_1 = float("nan")
        if total_spots2 > 0:
            nuclear_spots_2 = int(
                pd.to_numeric(nuclei_df.get("nuclear_spot_count_rna2", pd.Series(dtype=int)), errors="coerce").fillna(0).sum())
            cyto_spots_2 = int(
                pd.to_numeric(nuclei_df.get("cyto_spot_count_rna2", pd.Series(dtype=int)), errors="coerce").fillna(0).sum())
            frac_nuclear_image_2 = nuclear_spots_2 / float(total_spots2)
        else:
            nuclear_spots_2 = cyto_spots_2 = 0
            frac_nuclear_image_2 = float("nan")

        per_image = {
            "image": img_name,
            "condition": condition,
            "secondary_only": sec_only,
            "nuclei_analyzed": int(n),
            # RNA1 default-named (compat with rna_only/rna_protein readers)
            "mean_spots_per_nucleus": round(m1, 3),
            "median_spots_per_nucleus": round(_median(counts1), 3),
            "cv_spots_per_nucleus": round(cv_count_1, 4) if cv_count_1 == cv_count_1 else float("nan"),
            "frac_nuclei_with_ge_1_spot": _frac(counts1, 1),
            "frac_nuclei_with_ge_5_spots": _frac(counts1, 5),
            "frac_nuclei_with_ge_10_spots": _frac(counts1, 10),
            # RNA1 + RNA2 named explicitly (counts)
            "mean_spots_per_nucleus_rna1": round(m1, 3),
            "mean_spots_per_nucleus_rna2": round(m2, 3),
            "median_spots_per_nucleus_rna1": round(_median(counts1), 3),
            "median_spots_per_nucleus_rna2": round(_median(counts2), 3),
            "cv_spots_per_nucleus_rna1": round(cv_count_1, 4) if cv_count_1 == cv_count_1 else float("nan"),
            "cv_spots_per_nucleus_rna2": round(cv_count_2, 4) if cv_count_2 == cv_count_2 else float("nan"),
            "frac_nuclei_with_ge_1_spot_rna1": _frac(counts1, 1),
            "frac_nuclei_with_ge_5_spots_rna1": _frac(counts1, 5),
            "frac_nuclei_with_ge_10_spots_rna1": _frac(counts1, 10),
            "frac_nuclei_with_ge_1_spot_rna2": _frac(counts2, 1),
            "frac_nuclei_with_ge_5_spots_rna2": _frac(counts2, 5),
            "frac_nuclei_with_ge_10_spots_rna2": _frac(counts2, 10),
            "total_spots_rna1": total_spots1,
            "total_spots_rna2": total_spots2,
            # 2026-05-22 raw detections (incl. floaters) for audit; the
            # primary total_spots_* columns above are in-cell only.
            "total_spots_rna1_raw": total_spots1_raw,
            "total_spots_rna2_raw": total_spots2_raw,
            "n_floater_spots_rna1": n_floaters_rna1,
            "n_floater_spots_rna2": n_floaters_rna2,
            # Spot totals split by compartment (per channel)
            "nuclear_spots_rna1": nuclear_spots_1,
            "cytoplasmic_spots_rna1": cyto_spots_1,
            "nuclear_spots_rna2": nuclear_spots_2,
            "cytoplasmic_spots_rna2": cyto_spots_2,
            "frac_nuclear_rna1": frac_nuclear_image_1,
            "frac_nuclear_rna2": frac_nuclear_image_2,
            # ---- Per-CELL (nucleus + cytoplasm) total RNA intensity ------
            # These are the columns Brian explicitly asked for: total RNA
            # intensity per cell for BOTH channels, with mean/median/CV.
            "mean_cell_total_intensity_fit_rna1": round(mean_cell_int_1, 2) if mean_cell_int_1 == mean_cell_int_1 else float("nan"),
            "median_cell_total_intensity_fit_rna1": round(med_cell_int_1, 2) if med_cell_int_1 == med_cell_int_1 else float("nan"),
            "cv_cell_total_intensity_fit_rna1": round(cv_cell_int_1, 4) if cv_cell_int_1 == cv_cell_int_1 else float("nan"),
            "mean_cell_total_intensity_fit_rna2": round(mean_cell_int_2, 2) if mean_cell_int_2 == mean_cell_int_2 else float("nan"),
            "median_cell_total_intensity_fit_rna2": round(med_cell_int_2, 2) if med_cell_int_2 == med_cell_int_2 else float("nan"),
            "cv_cell_total_intensity_fit_rna2": round(cv_cell_int_2, 4) if cv_cell_int_2 == cv_cell_int_2 else float("nan"),
            # Per-nucleus (nucleus-only) total RNA intensity, both channels
            "mean_nuc_total_intensity_rna1": round(mean_nuc_int_1, 2) if mean_nuc_int_1 == mean_nuc_int_1 else float("nan"),
            "median_nuc_total_intensity_rna1": round(med_nuc_int_1, 2) if med_nuc_int_1 == med_nuc_int_1 else float("nan"),
            "mean_nuc_total_intensity_rna2": round(mean_nuc_int_2, 2) if mean_nuc_int_2 == mean_nuc_int_2 else float("nan"),
            "median_nuc_total_intensity_rna2": round(med_nuc_int_2, 2) if med_nuc_int_2 == med_nuc_int_2 else float("nan"),
            # Spot-detected total intensity per cell (only-expressing cells)
            "mean_cell_total_spot_intensity_fit_rna1": round(mean_tot_fit_1, 2) if mean_tot_fit_1 == mean_tot_fit_1 else float("nan"),
            "median_cell_total_spot_intensity_fit_rna1": round(med_tot_fit_1, 2) if med_tot_fit_1 == med_tot_fit_1 else float("nan"),
            "cv_cell_total_spot_intensity_fit_rna1": round(cv_tot_fit_1, 4) if cv_tot_fit_1 == cv_tot_fit_1 else float("nan"),
            "mean_cell_total_spot_intensity_fit_rna2": round(mean_tot_fit_2, 2) if mean_tot_fit_2 == mean_tot_fit_2 else float("nan"),
            "median_cell_total_spot_intensity_fit_rna2": round(med_tot_fit_2, 2) if med_tot_fit_2 == med_tot_fit_2 else float("nan"),
            "cv_cell_total_spot_intensity_fit_rna2": round(cv_tot_fit_2, 4) if cv_tot_fit_2 == cv_tot_fit_2 else float("nan"),
            # ---- N/C ratio rollups (image-level mean/median) --------------
            "mean_nc_ratio_total_intensity_rna1": round(mean_nc_total_1, 4) if mean_nc_total_1 == mean_nc_total_1 else float("nan"),
            "median_nc_ratio_total_intensity_rna1": round(med_nc_total_1, 4) if med_nc_total_1 == med_nc_total_1 else float("nan"),
            "mean_nc_ratio_total_intensity_rna2": round(mean_nc_total_2, 4) if mean_nc_total_2 == mean_nc_total_2 else float("nan"),
            "median_nc_ratio_total_intensity_rna2": round(med_nc_total_2, 4) if med_nc_total_2 == med_nc_total_2 else float("nan"),
            # ---- Above-floor intensity rollups (Brian/Sam 2026-05-20) -----
            # Same channel-resolved pub-image floor as the per-nucleus
            # columns above. Only emitted when
            # output.apply_pub_contrast_floor_to_analysis is True.
            **(
                {
                    "mean_nuclear_above_floor_intensity_rna1": round(mean_nuc_af_1, 2) if mean_nuc_af_1 == mean_nuc_af_1 else float("nan"),
                    "median_nuclear_above_floor_intensity_rna1": round(med_nuc_af_1, 2) if med_nuc_af_1 == med_nuc_af_1 else float("nan"),
                    "mean_nuclear_above_floor_intensity_rna2": round(mean_nuc_af_2, 2) if mean_nuc_af_2 == mean_nuc_af_2 else float("nan"),
                    "median_nuclear_above_floor_intensity_rna2": round(med_nuc_af_2, 2) if med_nuc_af_2 == med_nuc_af_2 else float("nan"),
                    "mean_cytoplasmic_above_floor_intensity_rna1": round(mean_cyt_af_1, 2) if mean_cyt_af_1 == mean_cyt_af_1 else float("nan"),
                    "median_cytoplasmic_above_floor_intensity_rna1": round(med_cyt_af_1, 2) if med_cyt_af_1 == med_cyt_af_1 else float("nan"),
                    "mean_cytoplasmic_above_floor_intensity_rna2": round(mean_cyt_af_2, 2) if mean_cyt_af_2 == mean_cyt_af_2 else float("nan"),
                    "median_cytoplasmic_above_floor_intensity_rna2": round(med_cyt_af_2, 2) if med_cyt_af_2 == med_cyt_af_2 else float("nan"),
                    "mean_nc_ratio_above_floor_intensity_rna1": round(mean_nc_af_1, 4) if mean_nc_af_1 == mean_nc_af_1 else float("nan"),
                    "median_nc_ratio_above_floor_intensity_rna1": round(med_nc_af_1, 4) if med_nc_af_1 == med_nc_af_1 else float("nan"),
                    "mean_nc_ratio_above_floor_intensity_rna2": round(mean_nc_af_2, 4) if mean_nc_af_2 == mean_nc_af_2 else float("nan"),
                    "median_nc_ratio_above_floor_intensity_rna2": round(med_nc_af_2, 4) if med_nc_af_2 == med_nc_af_2 else float("nan"),
                    "mean_frac_nuclear_above_floor_intensity_rna1": round(mean_frac_af_1, 4) if mean_frac_af_1 == mean_frac_af_1 else float("nan"),
                    "median_frac_nuclear_above_floor_intensity_rna1": round(med_frac_af_1, 4) if med_frac_af_1 == med_frac_af_1 else float("nan"),
                    "mean_frac_nuclear_above_floor_intensity_rna2": round(mean_frac_af_2, 4) if mean_frac_af_2 == mean_frac_af_2 else float("nan"),
                    "median_frac_nuclear_above_floor_intensity_rna2": round(med_frac_af_2, 4) if med_frac_af_2 == med_frac_af_2 else float("nan"),
                }
                if apply_floor
                else {}
            ),
            # ---- Thresholded RNA intensity (per-image roll-up, 2026-06-02) -
            # Plain per-image mean of the per-nucleus thresholded values, both
            # channels, both compartments. Always emitted (NaN if no floor).
            "rna_thresh_floor": rna_thresh_floor,
            "rna2_thresh_floor": rna2_thresh_floor,
            "mean_rna_thresh_total_intensity_nuclear": _img_stats("rna_thresh_total_intensity_nuclear")[0],
            "mean_rna_thresh_mean_intensity_nuclear": _img_stats("rna_thresh_mean_intensity_nuclear")[0],
            "mean_rna_thresh_pos_area_px_nuclear": _img_stats("rna_thresh_pos_area_px_nuclear")[0],
            "mean_rna_thresh_pos_fraction_nuclear": _img_stats("rna_thresh_pos_fraction_nuclear")[0],
            "mean_rna_thresh_total_intensity_cyto": _img_stats("rna_thresh_total_intensity_cyto")[0],
            "mean_rna_thresh_mean_intensity_cyto": _img_stats("rna_thresh_mean_intensity_cyto")[0],
            "mean_rna_thresh_pos_area_px_cyto": _img_stats("rna_thresh_pos_area_px_cyto")[0],
            "mean_rna_thresh_pos_fraction_cyto": _img_stats("rna_thresh_pos_fraction_cyto")[0],
            "mean_rna2_thresh_total_intensity_nuclear": _img_stats("rna2_thresh_total_intensity_nuclear")[0],
            "mean_rna2_thresh_mean_intensity_nuclear": _img_stats("rna2_thresh_mean_intensity_nuclear")[0],
            "mean_rna2_thresh_pos_area_px_nuclear": _img_stats("rna2_thresh_pos_area_px_nuclear")[0],
            "mean_rna2_thresh_pos_fraction_nuclear": _img_stats("rna2_thresh_pos_fraction_nuclear")[0],
            "mean_rna2_thresh_total_intensity_cyto": _img_stats("rna2_thresh_total_intensity_cyto")[0],
            "mean_rna2_thresh_mean_intensity_cyto": _img_stats("rna2_thresh_mean_intensity_cyto")[0],
            "mean_rna2_thresh_pos_area_px_cyto": _img_stats("rna2_thresh_pos_area_px_cyto")[0],
            "mean_rna2_thresh_pos_fraction_cyto": _img_stats("rna2_thresh_pos_fraction_cyto")[0],
            # ---- Active TS + mature mRNA (per-image roll-up) -------------
            "mean_n_nuclear_rna1_rna2_overlap_per_nucleus": round(mean_active_tss, 3) if mean_active_tss == mean_active_tss else 0.0,
            "median_n_nuclear_rna1_rna2_overlap_per_nucleus": round(med_active_tss, 3) if med_active_tss == med_active_tss else 0.0,
            "mean_n_cytoplasmic_rna1_spots_per_cell": round(mean_mature_1, 3) if mean_mature_1 == mean_mature_1 else 0.0,
            "median_n_cytoplasmic_rna1_spots_per_cell": round(med_mature_1, 3) if med_mature_1 == med_mature_1 else 0.0,
            "mean_n_cytoplasmic_rna2_spots_per_cell": round(mean_mature_2, 3) if mean_mature_2 == mean_mature_2 else 0.0,
            "median_n_cytoplasmic_rna2_spots_per_cell": round(med_mature_2, 3) if med_mature_2 == med_mature_2 else 0.0,
            # ---- Intensity-based floor-robust coloc (per-image rollup) ----
            # GATED (default OFF). The four mean_*_at_*_spots keys are injected
            # after this if/else only when cfg.foci.compute_partner_intensity is
            # True, keeping the OFF path byte-equivalent to the pre-feature
            # per_image_summary.
            # ---- Spot-spot colocalization (between channels) -------------
            f"paired_fraction_rna1_at_{pair_suffix}": img_paired_frac_1,
            f"paired_fraction_rna2_at_{pair_suffix}": img_paired_frac_2,
            f"paired_count_rna1_at_{pair_suffix}": paired1_total,
            f"paired_count_rna2_at_{pair_suffix}": paired2_total,
            "median_nn_distance_rna1_um": img_median_nn_1,
            "median_nn_distance_rna2_um": img_median_nn_2,
            # ---- Threshold provenance ------------------------------------
            "rna_threshold_value": rna_thr_value,
            "rna2_threshold_value": rna2_thr_value,
            "rna_bigfish_log_threshold": thr1_val,
            "rna2_bigfish_log_threshold": thr2_val,
            "n_nuclei_border_excluded": int(n_border_excluded),
            "total_spots": int(total_spots1 + total_spots2),
            "runtime_s": round(time.time() - t0, 3),
            "dapi_channel": int(dapi_idx),
            "rna_channel": int(rna_idx),
            "rna2_channel": int(rna2_idx),
            "voxel_xy_nm": voxel_xy_nm,
            "voxel_z_nm": voxel_z_nm,
            "n_z": int(img.n_z),
        }
        # GATED intensity-coloc per-image rollup (default OFF; see flag above).
        if compute_partner_intensity:
            per_image["mean_rna2_local_mean_at_rna1_spots"] = round(mean_partner_local_at_rna1, 3) if mean_partner_local_at_rna1 == mean_partner_local_at_rna1 else float("nan")
            per_image["mean_rna2_enrichment_at_rna1_spots"] = round(mean_partner_enrich_at_rna1, 4) if mean_partner_enrich_at_rna1 == mean_partner_enrich_at_rna1 else float("nan")
            per_image["mean_rna1_local_mean_at_rna2_spots"] = round(mean_partner_local_at_rna2, 3) if mean_partner_local_at_rna2 == mean_partner_local_at_rna2 else float("nan")
            per_image["mean_rna1_enrichment_at_rna2_spots"] = round(mean_partner_enrich_at_rna2, 4) if mean_partner_enrich_at_rna2 == mean_partner_enrich_at_rna2 else float("nan")
        # GATED random-position-null per-image pooled rollup (default OFF).
        if compute_partner_null:
            per_image["rna2_pooled_enrichment_vs_null_at_rna1_spots"] = round(pooled_null_enrichment, 4) if pooled_null_enrichment == pooled_null_enrichment else float("nan")
            per_image["rna2_pooled_null_z_at_rna1_spots"] = round(pooled_null_z, 3) if pooled_null_z == pooled_null_z else float("nan")
            per_image["rna2_pooled_null_p_empirical_at_rna1_spots"] = pooled_null_p_empirical
            per_image["rna2_pooled_obs_at_rna1_spots"] = round(pooled_null_obs, 3) if pooled_null_obs == pooled_null_obs else float("nan")
            per_image["rna2_pooled_null_mean_at_rna1_spots"] = round(pooled_null_mean, 3) if pooled_null_mean == pooled_null_mean else float("nan")
            per_image["n_nuclei_partner_null"] = int(_null_n_nuclei_used)
            per_image["partner_null_n"] = int(_null_n)
            per_image["partner_null_disk_px"] = float(_null_disk_px)
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
            "mean_spots_per_nucleus_rna1": 0.0,
            "mean_spots_per_nucleus_rna2": 0.0,
            "median_spots_per_nucleus_rna1": 0.0,
            "median_spots_per_nucleus_rna2": 0.0,
            "cv_spots_per_nucleus_rna1": float("nan"),
            "cv_spots_per_nucleus_rna2": float("nan"),
            "frac_nuclei_with_ge_1_spot_rna1": 0.0,
            "frac_nuclei_with_ge_5_spots_rna1": 0.0,
            "frac_nuclei_with_ge_10_spots_rna1": 0.0,
            "frac_nuclei_with_ge_1_spot_rna2": 0.0,
            "frac_nuclei_with_ge_5_spots_rna2": 0.0,
            "frac_nuclei_with_ge_10_spots_rna2": 0.0,
            "total_spots_rna1": total_spots1,
            "total_spots_rna2": total_spots2,
            "total_spots_rna1_raw": total_spots1_raw,
            "total_spots_rna2_raw": total_spots2_raw,
            "n_floater_spots_rna1": n_floaters_rna1,
            "n_floater_spots_rna2": n_floaters_rna2,
            "nuclear_spots_rna1": 0,
            "cytoplasmic_spots_rna1": 0,
            "nuclear_spots_rna2": 0,
            "cytoplasmic_spots_rna2": 0,
            "frac_nuclear_rna1": float("nan"),
            "frac_nuclear_rna2": float("nan"),
            "mean_cell_total_intensity_fit_rna1": float("nan"),
            "median_cell_total_intensity_fit_rna1": float("nan"),
            "cv_cell_total_intensity_fit_rna1": float("nan"),
            "mean_cell_total_intensity_fit_rna2": float("nan"),
            "median_cell_total_intensity_fit_rna2": float("nan"),
            "cv_cell_total_intensity_fit_rna2": float("nan"),
            "mean_nuc_total_intensity_rna1": float("nan"),
            "median_nuc_total_intensity_rna1": float("nan"),
            "mean_nuc_total_intensity_rna2": float("nan"),
            "median_nuc_total_intensity_rna2": float("nan"),
            "mean_cell_total_spot_intensity_fit_rna1": float("nan"),
            "median_cell_total_spot_intensity_fit_rna1": float("nan"),
            "cv_cell_total_spot_intensity_fit_rna1": float("nan"),
            "mean_cell_total_spot_intensity_fit_rna2": float("nan"),
            "median_cell_total_spot_intensity_fit_rna2": float("nan"),
            "cv_cell_total_spot_intensity_fit_rna2": float("nan"),
            "mean_nc_ratio_total_intensity_rna1": float("nan"),
            "median_nc_ratio_total_intensity_rna1": float("nan"),
            "mean_nc_ratio_total_intensity_rna2": float("nan"),
            "median_nc_ratio_total_intensity_rna2": float("nan"),
            # Thresholded RNA intensity provenance (no nuclei -> NaN means).
            "rna_thresh_floor": rna_thresh_floor,
            "rna2_thresh_floor": rna2_thresh_floor,
            "mean_n_nuclear_rna1_rna2_overlap_per_nucleus": 0.0,
            "median_n_nuclear_rna1_rna2_overlap_per_nucleus": 0.0,
            "mean_n_cytoplasmic_rna1_spots_per_cell": 0.0,
            "median_n_cytoplasmic_rna1_spots_per_cell": 0.0,
            "mean_n_cytoplasmic_rna2_spots_per_cell": 0.0,
            "median_n_cytoplasmic_rna2_spots_per_cell": 0.0,
            # GATED intensity-coloc rollup keys injected after this if/else
            # (default OFF) — kept absent here for pre-feature byte-equivalence.
            f"paired_fraction_rna1_at_{pair_suffix}": img_paired_frac_1,
            f"paired_fraction_rna2_at_{pair_suffix}": img_paired_frac_2,
            f"paired_count_rna1_at_{pair_suffix}": paired1_total,
            f"paired_count_rna2_at_{pair_suffix}": paired2_total,
            "median_nn_distance_rna1_um": img_median_nn_1,
            "median_nn_distance_rna2_um": img_median_nn_2,
            "rna_threshold_value": rna_thr_value,
            "rna2_threshold_value": rna2_thr_value,
            "rna_bigfish_log_threshold": thr1_val,
            "rna2_bigfish_log_threshold": thr2_val,
            "n_nuclei_border_excluded": int(n_border_excluded),
            "total_spots": int(total_spots1 + total_spots2),
            "runtime_s": round(time.time() - t0, 3),
            "dapi_channel": int(dapi_idx),
            "rna_channel": int(rna_idx),
            "rna2_channel": int(rna2_idx),
            "voxel_xy_nm": voxel_xy_nm,
            "voxel_z_nm": voxel_z_nm,
            "n_z": int(img.n_z),
        }
        # GATED intensity-coloc per-image rollup (default OFF) — empty-image
        # fallback emits NaN for these only when the feature is enabled.
        if compute_partner_intensity:
            per_image["mean_rna2_local_mean_at_rna1_spots"] = float("nan")
            per_image["mean_rna2_enrichment_at_rna1_spots"] = float("nan")
            per_image["mean_rna1_local_mean_at_rna2_spots"] = float("nan")
            per_image["mean_rna1_enrichment_at_rna2_spots"] = float("nan")
        # GATED random-position-null per-image rollup — empty-image fallback.
        if compute_partner_null:
            per_image["rna2_pooled_enrichment_vs_null_at_rna1_spots"] = float("nan")
            per_image["rna2_pooled_null_z_at_rna1_spots"] = float("nan")
            per_image["rna2_pooled_null_p_empirical_at_rna1_spots"] = float("nan")
            per_image["rna2_pooled_obs_at_rna1_spots"] = float("nan")
            per_image["rna2_pooled_null_mean_at_rna1_spots"] = float("nan")
            per_image["n_nuclei_partner_null"] = 0
            per_image["partner_null_n"] = int(_null_n)
            per_image["partner_null_disk_px"] = float(_null_disk_px)

    # ---- Thresholds row ----------------------------------------------------
    _kmad = float(pc_cfg.k_mad) if pc_cfg is not None else float("nan")
    _scope = getattr(pc_cfg, "threshold_scope", "") if pc_cfg is not None else ""
    _mode = pc_cfg.threshold_mode if pc_cfg is not None else "fallback"
    thresholds = {
        "image": img_name,
        # --- RNA1 (rna) thresholds — full provenance block --------------
        "rna_threshold_used": thr1_val,         # BigFISH/LoG auto-derived
        "rna_threshold_value": rna_thr_value,   # pixel-coloc threshold actually used
        "rna_threshold_method": "pixel_coloc_" + _mode if pc_cfg is not None else "fallback",
        "rna_threshold_mode": _mode,
        "rna_threshold_k_mad": _kmad,
        "rna_threshold_scope": _scope,
        "rna_bigfish_log_threshold": thr1_val,
        # --- RNA2 thresholds — same set, suffixed -----------------------
        "rna2_threshold_used": thr2_val,
        "rna2_threshold_value": rna2_thr_value,
        "rna2_threshold_method": "pixel_coloc_" + _mode if pc_cfg is not None else "fallback",
        "rna2_threshold_mode": _mode,
        "rna2_threshold_k_mad": _kmad,
        "rna2_threshold_scope": _scope,
        "rna2_bigfish_log_threshold": thr2_val,
        # --- DAPI / segmentation / spot params --------------------------
        "dapi_threshold_method": "Otsu dark",
        "dapi_threshold_value": dapi_thr_val,
        "spot_coloc_pair_distance_um": pair_um,
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
        # ---- Per-channel resolved BigFISH params (overrides applied) ----
        # These are what was ACTUALLY USED for each channel. When the user
        # left rna2_overrides empty the rna1 and rna2 columns will match.
        # When they set rna2_overrides.threshold_multiplier=0.7 (etc.), the
        # rna2_* column reflects the override while the rna_* column shows
        # the shared FociCfg value.
        "rna_bigfish_spot_radius_nm": rna1_params["bigfish_spot_radius_nm"],
        "rna2_bigfish_spot_radius_nm": rna2_params["bigfish_spot_radius_nm"],
        "rna_bigfish_spot_radius_z_nm": rna1_params["bigfish_spot_radius_z_nm"],
        "rna2_bigfish_spot_radius_z_nm": rna2_params["bigfish_spot_radius_z_nm"],
        "rna_threshold_multiplier": rna1_params["threshold_multiplier"],
        "rna2_threshold_multiplier": rna2_params["threshold_multiplier"],
        "rna_only_nuclear_spots": rna1_params["only_nuclear_spots"],
        "rna2_only_nuclear_spots": rna2_params["only_nuclear_spots"],
        "rna_min_sep_px": rna1_params["min_sep_px"],
        "rna2_min_sep_px": rna2_params["min_sep_px"],
    }

    qc = dict(
        labels=labels,
        dapi_2d=dapi_2d,
        rna_2d=rna_2d,
        rna2_2d=rna2_2d,
        cyt_labels=cyt_labels,
        dapi_mask=dapi_mask,
        rna_pos_mask=rna_pos_mask,
        rna2_pos_mask=rna2_pos_mask,
        spots1=spots1_df,
        spots2=spots2_df,
        voxel_xy_nm=voxel_xy_nm,
        nucleolus_labels=nucleolus_labels_for_qc,
    )

    _extra = dict(
        rna_thr_value=rna_thr_value,
        rna2_thr_value=rna2_thr_value,
        dapi_thr_value=dapi_thr_val,
        n_border_excluded=n_border_excluded,
        pair_suffix=pair_suffix,
        pair_distance_um=pair_um,
        mode="rna_rna",
    )
    # 2026-06-06 Brian: NATIVE coloc-figure carriers — added ONLY when the
    # gating flags are on (defaults OFF -> keys absent -> byte-identical extra).
    if coloc_null_draws_df is not None:
        _extra["coloc_null_draws"] = coloc_null_draws_df
    if coloc_radial_df is not None:
        _extra["coloc_radial_profile"] = coloc_radial_df

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
        extra=_extra,
    )


@register_mode("rna_rna")
def run(*args, **kwargs):
    return run_one(*args, **kwargs)


# Helper used by the batch-scope pre-pass in runner.py. Loads ONE image,
# segments nuclei (with border exclusion matching the main pass), and
# returns BOTH rna and rna2 raw nuclear pixel arrays. Each is pooled
# SEPARATELY across the batch -> two batch thresholds.
def collect_nuclear_rna_pixels(path, *, cfg) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(rna_nuclear_pixels, rna2_nuclear_pixels, labels)`` for one image.

    Mirrors rna_only.collect_nuclear_rna_pixels but returns BOTH RNA channels.
    Same channel resolution, segmentation, border exclusion, and dtype as
    the per-image path in ``run_one``.

    The THIRD return value is the FINAL (post-border-exclude) nuclei label
    image. The runner caches it keyed by image path and feeds it back into
    ``run_one`` via ``precomputed_labels=`` so each image is segmented exactly
    ONCE in a ``threshold_scope == 'batch'`` run (avoids a 2x segmentation
    cost with slow backends such as cellpose). On the empty-mask early return
    the labels are still returned so the runner can cache them.
    """
    img = _io.read_image(path)
    dapi_idx, rna_idx, rna2_idx = _resolve_channels(cfg, img)

    z_mode = cfg.z_stack.mode
    z_start = cfg.z_stack.start_slice
    z_end = cfg.z_stack.end_slice
    # 2026-05-22 Brian: per-image z-window override. If the current image's
    # file name matches a key in cfg.z_stack.file_overrides, use that
    # image-specific start_slice / end_slice instead of the batch default.
    _file_overrides = getattr(cfg.z_stack, "file_overrides", {}) or {}
    _img_name = Path(path).name
    if _img_name in _file_overrides:
        _ovr = _file_overrides[_img_name]
        if "start_slice" in _ovr:
            z_start = int(_ovr["start_slice"])
        if "end_slice" in _ovr:
            z_end = int(_ovr["end_slice"])
        try:
            from rich.console import Console as _C
            _C().print(f"  [dim]z-override: {_img_name} → start={z_start}, end={z_end}[/dim]")
        except Exception:
            pass
    if z_start is not None and z_start > img.n_z:
        z_start = 1
    if z_end is not None and z_end > img.n_z:
        z_end = img.n_z

    # 2026-05-18 Brian: autofocus mode now LOCKS all channels to DAPI's
    # picked focal plane. Previously each channel ran autofocus independently
    # — DAPI picked slice 42, RNA1 picked 38, RNA2 picked 45 — and spot xy
    # from a different physical plane than the nuclear mask led to
    # mis-assignment of nuclear-edge spots to cytoplasm.
    # 2026-05-24 Brian: autofocus_maxproj — per-image DAPI focus-window
    # detection, then MIP that window for all channels. Replaces per-image
    # file_overrides for datasets with field-to-field focus drift.
    dapi_autofocus_z: Optional[int] = None
    if z_mode == "autofocus":
        dapi_autofocus_z, dapi_2d = _io.extract_channel_autofocus_with_idx(
            img, dapi_idx, z_start=z_start, z_end=z_end,
            intensity_weighted=bool(getattr(cfg.z_stack, "autofocus_intensity_weighted", False)),
        )
        rna_2d = _io.extract_channel_at_z(img, rna_idx, z_1indexed=dapi_autofocus_z)
        rna2_2d = _io.extract_channel_at_z(img, rna2_idx, z_1indexed=dapi_autofocus_z)
        try:
            from rich.console import Console as _C
            _C().print(
                f"  [dim]z-lock (prescan): {Path(path).name} → all channels @ DAPI "
                f"plane z={dapi_autofocus_z}[/dim]"
            )
        except Exception:
            pass
    elif z_mode == "autofocus_maxproj":
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
        rna_2d = _io.extract_channel_in_z_range(
            img, rna_idx,
            z_start_1indexed=afm_zs, z_end_1indexed=afm_ze,
            project="maxproj",
        )
        rna2_2d = _io.extract_channel_in_z_range(
            img, rna2_idx,
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
    else:
        dapi_2d = _io.extract_channel(img, dapi_idx, z_mode=z_mode, z_start=z_start, z_end=z_end)
        if dapi_2d.ndim != 2:
            dapi_2d = dapi_2d.max(axis=0)
        rna_2d = _io.extract_channel(img, rna_idx, z_mode=z_mode, z_start=z_start, z_end=z_end)
        if rna_2d.ndim != 2:
            rna_2d = rna_2d.max(axis=0)
        rna2_2d = _io.extract_channel(img, rna2_idx, z_mode=z_mode, z_start=z_start, z_end=z_end)
        if rna2_2d.ndim != 2:
            rna2_2d = rna2_2d.max(axis=0)

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
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            labels,
        )
    return (
        rna_2d[nuc_mask].astype(np.float64),
        rna2_2d[nuc_mask].astype(np.float64),
        labels,
    )
