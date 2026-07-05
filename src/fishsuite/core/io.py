"""Image I/O for fishsuite.

Wraps `bioio` for reading VSI / CZI / LIF / TIFF files. Auto-detects channels
(DAPI / RNA / antibody) from per-channel metadata or pixel-content heuristics.

The bioio + bffile dependency emits ``np.asarray(..., copy=False)`` calls that
only work on numpy >= 2.0. We pin numpy < 2 (for tensorflow / stardist
compatibility) and patch bffile at import time — see
``src/fishsuite/__init__.py::_apply_bffile_compat_patch``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Literal

import numpy as np


@dataclass
class ImageWrapper:
    """Lightweight handle around a bioio BioImage instance.

    Attributes
    ----------
    path : Path
        Source file path.
    bio : bioio.BioImage
        The underlying bioio reader (scene already set).
    scene_idx : int
        Active scene index (always 0 here for VSI/CZI multi-scene files
        where scene 0 is the main image; scene 1 is the macro overview).
    shape : tuple
        Shape in TCZYX order.
    channel_names : list[str]
        Per-channel names from metadata.
    voxel_xy_nm : float
        Physical pixel size in nm. NaN if unavailable.
    voxel_z_nm : float
        Physical Z step in nm. NaN if unavailable.
    n_channels : int
    n_z : int
    """
    path: Path
    bio: object
    scene_idx: int
    shape: Tuple[int, ...]
    channel_names: List[str]
    voxel_xy_nm: float
    voxel_z_nm: float
    n_channels: int
    n_z: int


def read_image(path: Path, scene: int = 0) -> ImageWrapper:
    """Open an image file (any bioio-supported format) and return a wrapper."""
    from bioio import BioImage

    p = Path(path)
    # 2026-06-10: cheap pre-flight sanity check BEFORE handing the file to
    # bioio/BioFormats. Malformed or truncated/0-byte files (e.g. a stub .vsi)
    # otherwise reach the BioFormats JVM probe and trigger a hard native
    # "Windows fatal exception: access violation" that no Python try/except can
    # catch — corrupting JVM state for the rest of the process. Raising a normal
    # ValueError here keeps such files on the ordinary per-image failure path
    # (logged + excluded) instead of crashing the JVM. Real microscopy files
    # are far larger than this floor, so this never rejects a valid image.
    try:
        _sz = p.stat().st_size
    except OSError as _e:
        raise ValueError(f"Cannot stat image file {p.name}: {_e}") from _e
    # 512 bytes is below the header size of every real microscopy/TIFF format
    # but well above stub/placeholder files used in tests and corrupt 0-byte
    # acquisitions. Container formats (VSI/CZI/LIF/ND2) and TIFFs all exceed it.
    if _sz < 512:
        raise ValueError(
            f"Image file {p.name} is too small ({_sz} bytes) to be a valid "
            f"microscopy image — refusing to hand it to BioFormats."
        )

    bio = BioImage(p)
    scenes = list(bio.scenes)
    if scene >= len(scenes):
        raise ValueError(
            f"Scene index {scene} out of range for {p.name} (has {len(scenes)})"
        )
    bio.set_scene(scene)
    # bio.shape returns TCZYX after set_scene
    shape = tuple(bio.shape)
    # Find C and Z axis lengths from dims order
    dims_order = bio.dims.order  # e.g. "TCZYX"
    c_idx = dims_order.index("C")
    z_idx = dims_order.index("Z")
    n_c = int(shape[c_idx])
    n_z = int(shape[z_idx])
    # Voxel sizes (in microns from bioio); convert to nm
    try:
        psx = bio.physical_pixel_sizes
        x_um = psx.X
        z_um = psx.Z
        voxel_xy_nm = float(x_um) * 1000.0 if x_um else float("nan")
        voxel_z_nm = float(z_um) * 1000.0 if z_um else float("nan")
    except Exception:
        voxel_xy_nm = float("nan")
        voxel_z_nm = float("nan")
    channel_names = [str(n) for n in (bio.channel_names or [])]
    return ImageWrapper(
        path=p,
        bio=bio,
        scene_idx=scene,
        shape=shape,
        channel_names=channel_names,
        voxel_xy_nm=voxel_xy_nm,
        voxel_z_nm=voxel_z_nm,
        n_channels=n_c,
        n_z=n_z,
    )


def extract_channel(
    img: ImageWrapper,
    channel_idx: int,
    z_mode: str = "maxproj",
    z_start: Optional[int] = None,
    z_end: Optional[int] = None,
) -> np.ndarray:
    """Pull a single channel from the image, applying z-collapse.

    Parameters
    ----------
    img : ImageWrapper
    channel_idx : int
        0-indexed channel.
    z_mode : str
        One of:
          - "single": take z_start (1-indexed, default mid-stack)
          - "maxproj": max-projection over [z_start, z_end] (1-indexed, inclusive)
          - "autofocus": pick the sharpest z-plane in [z_start, z_end] by
                         normalized variance of Laplacian
          - "3d": return the full 3D stack (no projection)
    z_start, z_end : int or None
        1-indexed inclusive slice bounds. If None, full range.

    Returns
    -------
    np.ndarray
        2D for single/maxproj/autofocus; 3D for "3d".
    """
    if channel_idx < 0 or channel_idx >= img.n_channels:
        raise IndexError(
            f"Channel {channel_idx} out of range (image has {img.n_channels})"
        )
    # Get full ZYX for this channel
    zyx = img.bio.get_image_data("ZYX", T=0, C=channel_idx)  # type: ignore[attr-defined]
    if zyx.ndim != 3:
        # Some readers may return 2D directly when Z==1
        if zyx.ndim == 2:
            zyx = zyx[None, :, :]
        else:
            raise ValueError(f"Unexpected channel shape {zyx.shape}")
    nz = zyx.shape[0]

    # Clamp z bounds to [1, nz]; map to 0-indexed half-open slice
    if z_start is None:
        zs0 = 0
    else:
        zs0 = max(0, int(z_start) - 1)
    if z_end is None:
        ze0 = nz
    else:
        ze0 = min(nz, int(z_end))
    if zs0 >= ze0:
        # Bad range: fall back to full stack
        zs0, ze0 = 0, nz

    sub = zyx[zs0:ze0]  # shape (k, Y, X)

    if z_mode == "3d":
        return sub
    if z_mode == "single":
        # If z_start is None, take the middle of sub; else first slice of sub
        if z_start is None:
            mid = sub.shape[0] // 2
            return sub[mid]
        return sub[0]
    if z_mode == "maxproj":
        return sub.max(axis=0)
    if z_mode == "autofocus":
        return _autofocus_plane(sub)
    if z_mode == "autofocus_maxproj":
        # autofocus_maxproj is normally dispatched by the mode runners,
        # which compute the focus window once on DAPI then MIP that window
        # for all channels. When extract_channel is invoked directly with
        # this z_mode (e.g. the runner's pre-scan pub-contrast pool path
        # that doesn't know the DAPI channel), fall back to a per-channel
        # autofocus window using default parameters on the channel itself.
        # This keeps the pre-scan pool consistent with the per-image
        # render (both MIP a focus-derived slab) without requiring the
        # pre-scan to know which channel is DAPI.
        (ws, we), _ = compute_focus_window(sub)
        return sub[ws : we + 1].max(axis=0)
    raise ValueError(f"Unknown z_mode {z_mode!r}")


def _autofocus_plane(stack: np.ndarray, *, intensity_weighted: bool = False) -> np.ndarray:
    """Pick the sharpest 2D plane from a 3D stack by normalized Laplacian variance.

    When ``intensity_weighted`` is True the per-plane score is multiplied by the
    plane mean (see ``_autofocus_plane_with_idx``) — more robust on thick stacks
    with a near-flat focus profile.
    """
    _, plane = _autofocus_plane_with_idx(stack, intensity_weighted=intensity_weighted)
    return plane


# ---------------------------------------------------------------------------
# Per-slice focus metrics (used by autofocus_maxproj z-mode)
# ---------------------------------------------------------------------------

def _focus_score_variance_of_laplacian(plane: np.ndarray) -> float:
    """Variance of the Laplacian-filtered plane (Pertuz et al. 2013).

    Standard focus operator for fluorescence microscopy. Normalized by
    mean intensity so the score is insensitive to the absolute brightness
    of the slice — only the spatial structure of the gradient matters.
    """
    from scipy import ndimage as ndi
    p = plane.astype(np.float32)
    m = float(p.mean())
    if m <= 0:
        return 0.0
    lap = ndi.laplace(p / m)
    return float(np.var(lap))


def _focus_score_tenengrad(plane: np.ndarray) -> float:
    """Sum of squared Sobel-gradient magnitudes ("Tenengrad" focus measure).

    Sharper response to oriented edges than the Laplacian; sometimes
    preferred when the focus profile is shallow. Normalized by mean
    intensity for brightness invariance.
    """
    from scipy import ndimage as ndi
    p = plane.astype(np.float32)
    m = float(p.mean())
    if m <= 0:
        return 0.0
    pn = p / m
    gx = ndi.sobel(pn, axis=1)
    gy = ndi.sobel(pn, axis=0)
    return float(np.sum(gx * gx + gy * gy))


def _focus_score_normalized_variance(plane: np.ndarray) -> float:
    """Variance / mean^2 ("normalized variance") — brightness-invariant.

    Cheapest of the three metrics; useful as a sanity-check fallback.
    """
    p = plane.astype(np.float32)
    m = float(p.mean())
    if m <= 0:
        return 0.0
    return float(np.var(p) / (m * m))


_FOCUS_METRICS = {
    "variance_of_laplacian": _focus_score_variance_of_laplacian,
    "tenengrad": _focus_score_tenengrad,
    "normalized_variance": _focus_score_normalized_variance,
}


def compute_focus_window(
    dapi_zstack: np.ndarray,
    *,
    metric: str = "variance_of_laplacian",
    threshold_frac: float = 0.5,
    min_slices: int = 3,
    max_slices: int = 0,
    outer_start: Optional[int] = None,
    outer_end: Optional[int] = None,
    fixed_n_slices: int = 0,
    min_intensity_frac_of_peak: float = 0.0,
    intensity_weighted: bool = False,
    central_fraction: float = 0.0,
) -> Tuple[Tuple[int, int], dict]:
    """Detect the per-image in-focus DAPI z-window for autofocus_maxproj mode.

    Two modes:
      - **FWHM-style (default, fixed_n_slices=0)**: walk outward from peak
        focus slice while score >= threshold_frac * peak; per-image window
        width VARIES depending on the focus profile.
      - **Fixed-N centered (fixed_n_slices > 0)**: window is exactly N
        slices wide, centered on the peak-focus slice (symmetric for odd
        N, with the extra slice trailing the peak for even N). Per-image
        window WIDTH IS CONSTANT — only the position adapts. If clamping
        to outer bounds would shrink the window below N, the window is
        SHIFTED (slid toward the bound that wasn't violated) instead of
        shrunk, so the integration depth stays consistent across images.
        Window only shrinks below N when the *outer-bound interval itself*
        is narrower than N.

    Parameters
    ----------
    dapi_zstack : np.ndarray
        Full DAPI 3D stack with shape (Z, Y, X) — 0-indexed along Z.
    metric : {"variance_of_laplacian", "tenengrad", "normalized_variance"}
        Per-slice sharpness metric. Default: variance_of_laplacian (Pertuz
        et al. 2013, standard for fluorescence microscopy).
    threshold_frac : float
        FWHM-style threshold. Slices with focus_score >= threshold_frac *
        peak_score are included in the window. 0.5 = FWHM. Unused when
        fixed_n_slices > 0.
    min_slices : int
        Minimum number of slices in the returned window. If the natural
        FWHM window is smaller, the window is symmetrically expanded
        around the peak slice (clamped to valid stack indices and to
        outer_start/outer_end if set). Unused when fixed_n_slices > 0.
    max_slices : int
        Maximum number of slices. 0 = no cap. Unused when fixed_n_slices > 0.
    outer_start, outer_end : int or None
        0-indexed inclusive outer bounds the returned window must lie
        within. If set, the focus-score peak search and window expansion
        are both restricted to this range. (Used by the runner to honor
        cfg.z_stack.start_slice / end_slice as hard ceilings on what the
        per-image autofocus is allowed to pick.)
    fixed_n_slices : int
        When > 0, switches to fixed-N centered mode (see above). The
        window is exactly N slices wide, positioned to keep the peak-
        focus slice centered (subject to outer-bound clamping). When 0
        (default), uses FWHM-style variable-N logic.
    min_intensity_frac_of_peak : float
        Pre-filter: slices whose mean intensity is below this fraction of
        the brightest slice's mean are disqualified (focus_score set to
        -inf) before the peak search. Guards against noisy/empty edge
        slices. 0.0 (default) = disabled.
    intensity_weighted : bool
        When True, each slice's focus score is MULTIPLIED by that slice's
        mean intensity — i.e. ``var(laplace(plane/mean)) * mean`` (the BIN1
        thick-stack fix; see ``_autofocus_plane_with_idx``). This pulls the
        focus peak toward the bright AND sharp nuclear plane instead of a
        noise-driven dim edge plane, which is exactly the failure the plain
        mean-normalized Laplacian variance hit on near-flat / noisy focus
        profiles (2026-05-31 Brian: the H9 33-plane DAPI stacks picked
        z=33/z=1 edges under the unweighted metric). Mirrors the
        single-plane ``autofocus`` z-mode's intensity weighting so
        ``autofocus_maxproj`` and ``autofocus`` use the SAME per-slice score.
        Wired from ``cfg.z_stack.autofocus_intensity_weighted``. Default
        False (legacy parity). Applied to the per-slice score BEFORE the
        min-intensity pre-filter overwrite and BEFORE peak search, so it
        composes with ``min_intensity_frac_of_peak``.
    central_fraction : float
        Robustness guard. When in ``(0, 1]``, the focus-PEAK search is
        additionally restricted to the central ``central_fraction`` of the
        (outer-bounded) stack — e.g. 0.6 keeps only the middle 60% of
        slices eligible to WIN the peak. This prevents the objective window
        from anchoring on a true stack-edge plane even if (after weighting)
        an edge plane still scores highest. The peak is constrained, but the
        fixed-N / FWHM window may still extend toward the edge from a central
        peak. 0.0 (default) = disabled (whole [lo, hi] range eligible).

    Returns
    -------
    (window_start, window_end), diagnostics
        ``window_start`` and ``window_end`` are inclusive 0-indexed slice
        indices into the original dapi_zstack. diagnostics is a dict:
        ``{"peak_z": int, "peak_score": float, "focus_scores": list[float],
        "window_size": int, "window_start": int, "window_end": int,
        "threshold_used": float, "expanded_to_min": bool,
        "clipped_to_max": bool, "metric": str}``. When fixed_n_slices > 0,
        the diagnostics dict also contains ``"fixed_n": True,
        "requested_n": int, "actual_n": int, "shifted_for_bounds": bool,
        "shrunk_by_bounds": bool``. ``actual_n`` equals ``requested_n``
        unless the outer-bound interval is narrower than N (in which case
        ``shrunk_by_bounds`` is True). ``shifted_for_bounds`` is True
        when the window was slid off-center to keep the requested N.
    """
    if dapi_zstack.ndim != 3:
        raise ValueError(
            f"compute_focus_window: expected 3D stack, got shape {dapi_zstack.shape}"
        )
    nz = int(dapi_zstack.shape[0])
    if nz == 0:
        raise ValueError("compute_focus_window: empty z-stack")

    # Resolve outer bounds (0-indexed inclusive). Default = whole stack.
    lo = 0 if outer_start is None else max(0, int(outer_start))
    hi = (nz - 1) if outer_end is None else min(nz - 1, int(outer_end))
    if lo > hi:
        # Bad outer bounds — fall back to whole stack
        lo, hi = 0, nz - 1

    score_fn = _FOCUS_METRICS.get(metric)
    if score_fn is None:
        raise ValueError(
            f"compute_focus_window: unknown metric {metric!r} "
            f"(valid: {sorted(_FOCUS_METRICS)})"
        )

    # Compute per-slice focus scores. We compute over the FULL stack so
    # diagnostics include the whole profile (useful for inspection),
    # but the peak / window expansion is restricted to [lo, hi].
    #
    # Slice means are needed both for the min-intensity pre-filter and for
    # intensity-weighting; compute them once.
    slice_means = np.asarray(
        [float(dapi_zstack[z].mean()) for z in range(nz)], dtype=float
    )
    focus_scores: list[float] = []
    for z in range(nz):
        s = float(score_fn(dapi_zstack[z]))
        if intensity_weighted:
            # var(laplace(plane/mean)) * mean — weight sharpness by plane
            # brightness so the peak favors the bright + sharp nuclear plane
            # over a noise-driven dim edge plane. Mirrors the single-plane
            # autofocus metric (_autofocus_plane_with_idx). mean>=0 always.
            s *= slice_means[z]
        focus_scores.append(s)

    # 2026-05-24 v7 Brian: min-intensity pre-filter. Disqualify slices whose
    # mean DAPI intensity is below ``min_intensity_frac_of_peak * max_slice_mean``
    # by overwriting their focus_score with -inf BEFORE peak search. Guards
    # against noisy/empty edge slices (top/bottom of stack) that score
    # spuriously high on variance_of_laplacian or normalized_variance.
    excluded_by_intensity = []
    if min_intensity_frac_of_peak and min_intensity_frac_of_peak > 0.0:
        max_mean = float(slice_means.max()) if slice_means.size else 0.0
        cutoff = max_mean * float(min_intensity_frac_of_peak)
        for z in range(nz):
            if slice_means[z] < cutoff:
                focus_scores[z] = float("-inf")
                excluded_by_intensity.append(z)

    # Determine the slice range eligible to WIN the focus peak. Default is the
    # full outer-bounded range [lo, hi]. The central_fraction guard shrinks
    # this to the middle fraction of [lo, hi] so a true stack-edge plane can
    # never anchor the window even if it scores highest (robustness against
    # near-flat / noisy edge-heavy focus profiles). The window itself (fixed-N
    # or FWHM) may still extend toward an edge from a central peak.
    peak_lo, peak_hi = lo, hi
    if central_fraction and 0.0 < float(central_fraction) < 1.0:
        span = hi - lo + 1
        keep = max(1, int(round(span * float(central_fraction))))
        margin = (span - keep) // 2
        peak_lo = lo + margin
        peak_hi = hi - (span - keep - margin)
        if peak_lo > peak_hi:  # degenerate — fall back to full range
            peak_lo, peak_hi = lo, hi

    # Find peak within the (possibly central-restricted) [peak_lo, peak_hi]
    sub_scores = focus_scores[peak_lo : peak_hi + 1]
    peak_offset = int(np.argmax(sub_scores))
    peak_z = peak_lo + peak_offset
    peak_score = float(focus_scores[peak_z])

    # ─── Branch: fixed-N centered window ──────────────────────────────────
    # When fixed_n_slices > 0, return exactly N slices centered on peak_z,
    # bypassing the FWHM threshold + min/max-slice machinery entirely.
    # Window is positioned to keep peak_z centered (symmetric for odd N);
    # for even N, the extra slice trails the peak (peak - n//2, peak + (n-1)//2).
    # If centered position would violate outer bounds, SHIFT (don't shrink)
    # the window toward the bound that wasn't violated, so integration
    # depth stays constant across the batch. Window only shrinks below N
    # when the outer-bound interval [lo, hi] itself is narrower than N.
    if fixed_n_slices and int(fixed_n_slices) > 0:
        n_req = int(fixed_n_slices)
        outer_span = hi - lo + 1
        if n_req >= outer_span:
            # Outer bounds force a smaller window than requested.
            ws, we = lo, hi
            actual_n = outer_span
            shifted = False
            shrunk = True
        else:
            # Center on peak (asymmetric for even N: half before, n-1-half after)
            half_lo = n_req // 2          # slices BEFORE peak
            half_hi = n_req - 1 - half_lo # slices AFTER peak (= (n-1)//2)
            ws = peak_z - half_lo
            we = peak_z + half_hi
            shifted = False
            # Slide the whole window toward the bound that wasn't violated
            if ws < lo:
                shift = lo - ws
                ws += shift
                we += shift
                shifted = True
            elif we > hi:
                shift = we - hi
                ws -= shift
                we -= shift
                shifted = True
            actual_n = we - ws + 1
            shrunk = False  # outer_span > n_req, so width is preserved
        diagnostics = {
            "metric": metric,
            "peak_z": int(peak_z),
            "peak_score": float(peak_score),
            "focus_scores": focus_scores,
            "window_size": int(actual_n),
            "window_start": int(ws),
            "window_end": int(we),
            "threshold_used": 0.0,
            "outer_start": int(lo),
            "outer_end": int(hi),
            "expanded_to_min": False,
            "clipped_to_max": False,
            "fixed_n": True,
            "requested_n": int(n_req),
            "actual_n": int(actual_n),
            "shifted_for_bounds": bool(shifted),
            "shrunk_by_bounds": bool(shrunk),
            "intensity_weighted": bool(intensity_weighted),
            "central_fraction": float(central_fraction),
            "peak_search_lo": int(peak_lo),
            "peak_search_hi": int(peak_hi),
        }
        return (int(ws), int(we)), diagnostics

    # ─── Default: FWHM-style variable-N window ────────────────────────────
    threshold = float(threshold_frac) * peak_score

    # FWHM-style window expansion: walk outward from peak while score >=
    # threshold. Stop on first slice below threshold (don't keep going
    # past a dip and rejoining).
    ws = peak_z
    while ws - 1 >= lo and focus_scores[ws - 1] >= threshold:
        ws -= 1
    we = peak_z
    while we + 1 <= hi and focus_scores[we + 1] >= threshold:
        we += 1

    expanded_to_min = False
    clipped_to_max = False

    # Enforce min_slices: symmetric expansion around peak, clamped to [lo, hi]
    mn = max(1, int(min_slices))
    while (we - ws + 1) < mn:
        expanded_to_min = True
        grew = False
        # Try to grow on the side that has more room (prefer symmetric)
        room_left = ws - lo
        room_right = hi - we
        if room_left == 0 and room_right == 0:
            break  # cannot grow further
        if room_left >= room_right and room_left > 0:
            ws -= 1
            grew = True
        elif room_right > 0:
            we += 1
            grew = True
        if not grew:
            break

    # Enforce max_slices ceiling: symmetric trim around peak
    if max_slices and max_slices > 0:
        mx = int(max_slices)
        if (we - ws + 1) > mx:
            clipped_to_max = True
            # Re-center on peak, take ±(mx//2) around it, clamped
            half = mx // 2
            # Try to keep peak inside the trimmed window
            ws_new = max(lo, peak_z - half)
            we_new = ws_new + mx - 1
            if we_new > hi:
                we_new = hi
                ws_new = max(lo, we_new - mx + 1)
            ws, we = ws_new, we_new

    diagnostics = {
        "metric": metric,
        "peak_z": int(peak_z),
        "peak_score": float(peak_score),
        "focus_scores": focus_scores,
        "window_size": int(we - ws + 1),
        "window_start": int(ws),
        "window_end": int(we),
        "threshold_used": float(threshold),
        "outer_start": int(lo),
        "outer_end": int(hi),
        "expanded_to_min": bool(expanded_to_min),
        "clipped_to_max": bool(clipped_to_max),
        "fixed_n": False,
        "intensity_weighted": bool(intensity_weighted),
        "central_fraction": float(central_fraction),
        "peak_search_lo": int(peak_lo),
        "peak_search_hi": int(peak_hi),
    }
    return (int(ws), int(we)), diagnostics


def _autofocus_plane_with_idx(
    stack: np.ndarray, *, intensity_weighted: bool = False
) -> "tuple[int, np.ndarray]":
    """Like ``_autofocus_plane`` but also returns the picked 0-indexed slice.

    Callers that need to lock other channels to the same focal plane (e.g.
    rna_rna mode, where RNA1/RNA2 spots must be measured at DAPI's z so the
    nuclear mask + spot xy come from the SAME physical plane) use this to
    grab the index, then re-extract the other channels in 'single' mode at
    that absolute z.

    Per-plane score:
      * default (``intensity_weighted=False``): ``var(laplace(plane/mean))`` —
        the legacy mean-normalized Laplacian variance (Pertuz et al. 2013).
      * ``intensity_weighted=True``: ``var(laplace(plane/mean)) * mean`` — the
        same sharpness term WEIGHTED by the plane mean intensity. 2026-05-28
        Brian: on thick (~16µm, nz≈79) d8 cMyo stacks the focus profile is
        near-flat across the cell-containing depth, so the unweighted score at
        a badly out-of-focus HIGH plane is only ~1.0–1.2× the in-focus score —
        noise then tips the pick to garbage upper planes (measured DAPI picks
        z=42–67 when the true in-focus nuclear plane is z≈20–24). Multiplying
        by the plane mean pulls the pick toward the bright AND sharp nuclear
        plane (validated: picks z≈18–22 on all 8 Dataset A images). The
        ``mean <= 0`` guard is preserved (score 0).
    """
    from scipy import ndimage as ndi
    if stack.ndim != 3:
        raise ValueError("expected 3D stack")
    if stack.shape[0] == 1:
        return 0, stack[0]
    best_idx = 0
    best_score = -np.inf
    for z in range(stack.shape[0]):
        plane = stack[z].astype(np.float32)
        # Normalize by mean to make the sharpness term insensitive to brightness
        mean = float(plane.mean())
        if mean <= 0:
            score = 0.0
        else:
            lap = ndi.laplace(plane / mean)
            score = float(np.var(lap))
            if intensity_weighted:
                # Weight sharpness by plane brightness so the pick favors the
                # bright in-focus nuclear plane over noise-driven upper planes.
                score *= mean
        if score > best_score:
            best_score = score
            best_idx = z
    return best_idx, stack[best_idx]


def extract_channel_autofocus_with_idx(
    img: ImageWrapper,
    channel_idx: int,
    z_start: Optional[int] = None,
    z_end: Optional[int] = None,
    *,
    intensity_weighted: bool = False,
) -> "tuple[int, np.ndarray]":
    """Autofocus a single channel; return (absolute_z_1indexed, 2D plane).

    The returned z is 1-indexed against the full stack (so it can be passed
    back into ``extract_channel(z_mode='single', z_start=z, z_end=z)`` to
    lock other channels to the same physical plane).

    ``intensity_weighted`` (default False) selects the per-plane focus score —
    see ``_autofocus_plane_with_idx``. Wired to ``z_stack.autofocus_intensity_weighted``
    at the mode/runner call sites for the single-plane ``autofocus`` z-mode.
    """
    if channel_idx < 0 or channel_idx >= img.n_channels:
        raise IndexError(
            f"Channel {channel_idx} out of range (image has {img.n_channels})"
        )
    zyx = img.bio.get_image_data("ZYX", T=0, C=channel_idx)  # type: ignore[attr-defined]
    if zyx.ndim == 2:
        zyx = zyx[None, :, :]
    nz = zyx.shape[0]
    zs0 = 0 if z_start is None else max(0, int(z_start) - 1)
    ze0 = nz if z_end is None else min(nz, int(z_end))
    if zs0 >= ze0:
        zs0, ze0 = 0, nz
    sub = zyx[zs0:ze0]
    local_idx, plane = _autofocus_plane_with_idx(
        sub, intensity_weighted=intensity_weighted
    )
    # local_idx is 0-indexed within the windowed substack; convert to
    # 1-indexed absolute z against the full stack.
    abs_z_1indexed = zs0 + local_idx + 1
    return abs_z_1indexed, plane


def extract_dapi_focus_window(
    img: ImageWrapper,
    dapi_channel_idx: int,
    *,
    metric: str = "variance_of_laplacian",
    threshold_frac: float = 0.5,
    min_slices: int = 3,
    max_slices: int = 0,
    z_start: Optional[int] = None,
    z_end: Optional[int] = None,
    fixed_n_slices: int = 0,
    min_intensity_frac_of_peak: float = 0.0,
    intensity_weighted: bool = False,
    central_fraction: float = 0.0,
) -> "tuple[Tuple[int, int], dict, np.ndarray]":
    """Compute the per-image in-focus DAPI z-window and return its MIP.

    Used by the autofocus_maxproj z-mode dispatch in the per-image mode
    runners. Returns the focus-window bounds (1-indexed, inclusive, for
    consistency with extract_channel_in_z_range), the diagnostics dict,
    and the DAPI MIP over that window — so the caller can re-use the
    same window for the RNA channels via extract_channel_in_z_range.

    z_start / z_end are 1-indexed inclusive outer bounds (same convention
    as cfg.z_stack.start_slice / end_slice). Pass None to use the whole
    stack.

    fixed_n_slices: when > 0, use fixed-N centered window mode (see
    compute_focus_window docstring). When 0 (default), use FWHM logic.
    """
    if dapi_channel_idx < 0 or dapi_channel_idx >= img.n_channels:
        raise IndexError(
            f"Channel {dapi_channel_idx} out of range (image has {img.n_channels})"
        )
    zyx = img.bio.get_image_data("ZYX", T=0, C=dapi_channel_idx)  # type: ignore[attr-defined]
    if zyx.ndim == 2:
        zyx = zyx[None, :, :]
    nz = zyx.shape[0]
    # Convert 1-indexed outer bounds to 0-indexed inclusive (compute_focus_window
    # takes 0-indexed bounds).
    outer_start_0 = None if z_start is None else max(0, int(z_start) - 1)
    outer_end_0 = None if z_end is None else min(nz - 1, int(z_end) - 1)

    (ws0, we0), diag = compute_focus_window(
        zyx,
        metric=metric,
        threshold_frac=threshold_frac,
        min_slices=min_slices,
        max_slices=max_slices,
        outer_start=outer_start_0,
        outer_end=outer_end_0,
        fixed_n_slices=fixed_n_slices,
        min_intensity_frac_of_peak=min_intensity_frac_of_peak,
        intensity_weighted=intensity_weighted,
        central_fraction=central_fraction,
    )
    dapi_mip = zyx[ws0 : we0 + 1].max(axis=0)
    # Convert window to 1-indexed inclusive for the caller (parity with the
    # rest of the io.py 1-indexed API).
    return (int(ws0 + 1), int(we0 + 1)), diag, dapi_mip


def extract_channel_in_z_range(
    img: ImageWrapper,
    channel_idx: int,
    *,
    z_start_1indexed: int,
    z_end_1indexed: int,
    project: Literal["maxproj", "mean", "none"] = "maxproj",
) -> np.ndarray:
    """Extract a non-DAPI channel over a pre-computed z-window.

    Companion to extract_dapi_focus_window: once the DAPI focus window is
    chosen, the RNA / antibody channels are pulled over the SAME slab and
    MIP'd so all channels are anatomically aligned.

    z_start_1indexed / z_end_1indexed are 1-indexed inclusive (matches
    the cfg.z_stack.* convention). project="maxproj" (default) returns
    the max projection over the slab; "mean" returns the mean projection;
    "none" returns the raw (Z, Y, X) sub-stack.
    """
    if channel_idx < 0 or channel_idx >= img.n_channels:
        raise IndexError(
            f"Channel {channel_idx} out of range (image has {img.n_channels})"
        )
    zyx = img.bio.get_image_data("ZYX", T=0, C=channel_idx)  # type: ignore[attr-defined]
    if zyx.ndim == 2:
        zyx = zyx[None, :, :]
    nz = zyx.shape[0]
    zs0 = max(0, int(z_start_1indexed) - 1)
    ze0 = min(nz, int(z_end_1indexed))  # half-open
    if zs0 >= ze0:
        zs0, ze0 = 0, nz
    sub = zyx[zs0:ze0]
    if project == "none":
        return sub
    if project == "mean":
        return sub.mean(axis=0)
    return sub.max(axis=0)


def extract_channel_at_z(
    img: ImageWrapper,
    channel_idx: int,
    z_1indexed: int,
) -> np.ndarray:
    """Extract a single 2D plane at an exact 1-indexed z (no autofocus, no maxproj).

    Used to lock RNA / antibody channels to DAPI's autofocus pick so the
    nuclear mask + spot xy come from the same physical plane.
    """
    if channel_idx < 0 or channel_idx >= img.n_channels:
        raise IndexError(
            f"Channel {channel_idx} out of range (image has {img.n_channels})"
        )
    zyx = img.bio.get_image_data("ZYX", T=0, C=channel_idx)  # type: ignore[attr-defined]
    if zyx.ndim == 2:
        return zyx
    nz = zyx.shape[0]
    z0 = max(0, min(nz - 1, int(z_1indexed) - 1))
    return zyx[z0]


def rna_plane_quality(plane: np.ndarray) -> dict:
    """Signal-quality readout for a single 2D RNA plane (2026-07-05 Brian).

    A per-image gauge of whether the RNA channel at the chosen focal plane
    carries *callable* single-molecule signal — used both to REPORT (per-image
    ``rna_focus_score`` / ``rna_dynamic_range`` columns) and to DRIVE the
    ``autofocus_channel == "auto"`` RNA-vs-DAPI anchor decision.

    Returns a dict with:
      * ``focus_score`` — ``var(laplace(plane / mean))``, the mean-normalized
        Laplacian variance (Pertuz et al. 2013). Higher = crisper structure /
        sharper puncta. Brightness-insensitive.
      * ``dynamic_range`` — ``(p99.9 - median) / (1.4826 * MAD)``, a robust SNR
        proxy: how far the brightest pixels stand above the background noise
        floor. Real puncta push it high; a flat / out-of-focus / pure-noise
        field stays near 0. This is the score the ``auto`` gate thresholds.
      * ``background_median`` / ``background_mad`` / ``top_p999`` — the raw
        components (handy for logging / debugging).

    Fully defensive — returns NaNs rather than raising on an empty/degenerate
    plane.
    """
    out = {
        "focus_score": float("nan"),
        "dynamic_range": float("nan"),
        "background_median": float("nan"),
        "background_mad": float("nan"),
        "top_p999": float("nan"),
    }
    arr = np.asarray(plane)
    if arr.size == 0:
        return out
    p = arr.astype(np.float32)
    # --- focus score (mean-normalized Laplacian variance) ---
    try:
        from scipy import ndimage as ndi

        m = float(p.mean())
        if m > 0:
            out["focus_score"] = float(np.var(ndi.laplace(p / m)))
        else:
            out["focus_score"] = 0.0
    except Exception:
        out["focus_score"] = float("nan")
    # --- robust dynamic range / spot SNR ---
    try:
        med = float(np.median(p))
        mad = float(np.median(np.abs(p - med)))
        p999 = float(np.percentile(p, 99.9))
        out["background_median"] = med
        out["background_mad"] = mad
        out["top_p999"] = p999
        denom = 1.4826 * mad
        if denom > 0:
            out["dynamic_range"] = (p999 - med) / denom
        elif p999 > med:
            # Zero MAD but a bright tail exists -> effectively unbounded SNR;
            # report a large finite sentinel so the auto gate still fires.
            out["dynamic_range"] = float("inf")
        else:
            out["dynamic_range"] = 0.0
    except Exception:
        pass
    return out


def resolve_autofocus_plane(
    img: ImageWrapper,
    *,
    dapi_idx: int,
    rna_idx: int,
    z_start: Optional[int] = None,
    z_end: Optional[int] = None,
    autofocus_channel: str = "dapi",
    intensity_weighted: bool = False,
    auto_rna_quality_min: float = 3.0,
) -> "tuple[int, str, dict]":
    """Pick the single autofocus plane, optionally anchored on the RNA channel.

    Companion to the LOCKED default DAPI-anchor path (the mode runners still
    call ``extract_channel_autofocus_with_idx`` directly for ``autofocus_channel
    == "dapi"`` so that path is byte-for-byte unchanged). This helper handles
    ONLY the opt-in ``"rna"`` / ``"auto"`` anchors.

    The one-plane invariant is preserved: exactly ONE absolute z is returned
    and the caller reads DAPI (segmentation), RNA and antibody all at that z —
    only the channel that CHOOSES the plane differs.

    Parameters
    ----------
    autofocus_channel : {"rna", "auto"}
        ``"rna"`` -> pick the sharpest RNA1 plane. ``"auto"`` -> RNA-anchor
        when the RNA-best plane's ``dynamic_range`` (see ``rna_plane_quality``)
        is >= ``auto_rna_quality_min``, else DAPI-anchor. (``"dapi"`` is a
        defensive alias for the DAPI-anchor branch — normal callers never pass
        it here.)
    intensity_weighted : bool
        Forwarded to the per-plane focus scorer (thick-stack fix), same as the
        DAPI path.
    auto_rna_quality_min : float
        RNA dynamic-range gate for ``"auto"``.

    Returns
    -------
    (abs_z_1indexed, channel_used, diag)
        ``channel_used`` is "rna" or "dapi". ``diag`` carries
        ``rna_z`` / ``dapi_z`` (1-indexed, may be None if not computed),
        ``rna_quality_score`` (the dynamic-range used for the auto decision),
        ``rna_quality_min`` (the threshold), ``rna_focus_score`` and the mode.
    """
    ch = str(autofocus_channel).lower()
    diag: dict = {
        "requested_channel": ch,
        "rna_z": None,
        "dapi_z": None,
        "rna_quality_score": float("nan"),
        "rna_quality_min": float(auto_rna_quality_min),
        "rna_focus_score": float("nan"),
    }

    def _dapi_pick() -> int:
        z, _ = extract_channel_autofocus_with_idx(
            img, dapi_idx, z_start=z_start, z_end=z_end,
            intensity_weighted=intensity_weighted,
        )
        diag["dapi_z"] = int(z)
        return int(z)

    # Defensive: an explicit "dapi" (or anything unrecognised) falls back to the
    # DAPI anchor so a bad config value can never crash a run.
    if ch not in ("rna", "auto"):
        return _dapi_pick(), "dapi", diag

    # Autofocus the RNA channel. On failure, fall back to DAPI.
    try:
        rna_z, rna_plane = extract_channel_autofocus_with_idx(
            img, rna_idx, z_start=z_start, z_end=z_end,
            intensity_weighted=intensity_weighted,
        )
        diag["rna_z"] = int(rna_z)
        q = rna_plane_quality(rna_plane)
        diag["rna_quality_score"] = float(q.get("dynamic_range", float("nan")))
        diag["rna_focus_score"] = float(q.get("focus_score", float("nan")))
    except Exception:
        return _dapi_pick(), "dapi", diag

    if ch == "rna":
        return int(rna_z), "rna", diag

    # ch == "auto": gate on RNA dynamic range.
    score = diag["rna_quality_score"]
    if np.isfinite(score) and score >= float(auto_rna_quality_min):
        return int(rna_z), "rna", diag
    return _dapi_pick(), "dapi", diag


def get_voxel_size_nm(img: ImageWrapper) -> Tuple[float, float]:
    """Return (xy_nm, z_nm) physical voxel size, or (NaN, NaN) if unavailable."""
    return (img.voxel_xy_nm, img.voxel_z_nm)


# ---------------------------------------------------------------------------
# Channel auto-detection
# ---------------------------------------------------------------------------

# Patterns from the Fiji pipeline (utils/channel_detection.py).
_DAPI_NAME_PATTERNS = ("dapi", "hoechst", "draq5", "405")
_RNA_NAME_PATTERNS = ("488", "fitc", "gfp", "555", "tritc", "cy3", "561")
_AB_NAME_PATTERNS = ("647", "cy5", "far red", "far-red", "640")


def _name_matches(name: str, patterns) -> bool:
    nl = str(name).lower()
    return any(p in nl for p in patterns)


def autodetect_channels(img: ImageWrapper) -> dict:
    """Return ``{'dapi': idx, 'rna': idx, 'ab': idx}`` (all 0-indexed; -1 = unknown).

    Strategy:
      1. Match channel_names against fluorophore/wavelength patterns.
      2. Fallback: rank by mean intensity (highest -> DAPI candidate).
    """
    names = img.channel_names
    n = img.n_channels
    out = {"dapi": -1, "rna": -1, "ab": -1}

    # Layer 1: name matching
    for i, name in enumerate(names):
        if out["dapi"] == -1 and _name_matches(name, _DAPI_NAME_PATTERNS):
            out["dapi"] = i
    for i, name in enumerate(names):
        if i in out.values():
            continue
        if out["rna"] == -1 and _name_matches(name, _RNA_NAME_PATTERNS):
            out["rna"] = i
    for i, name in enumerate(names):
        if i in out.values():
            continue
        if out["ab"] == -1 and _name_matches(name, _AB_NAME_PATTERNS):
            out["ab"] = i

    # Layer 2: heuristic — DAPI = highest mean (bright diffuse)
    if out["dapi"] == -1 and n > 0:
        means = []
        for c in range(n):
            p = extract_channel(img, c, z_mode="maxproj")
            means.append((float(p.mean()), c))
        means.sort(reverse=True)
        out["dapi"] = means[0][1]

    # If RNA / AB still missing, fill with whatever channels remain
    remaining = [c for c in range(n) if c not in out.values()]
    if out["rna"] == -1 and remaining:
        out["rna"] = remaining.pop(0)
    if out["ab"] == -1 and remaining:
        out["ab"] = remaining.pop(0)
    return out


# ---------------------------------------------------------------------------
# Input discovery
# ---------------------------------------------------------------------------

@dataclass
class DiscoveredImage:
    path: Path
    condition: str
    sec_only: bool
    subfolder: str


def discover_inputs(
    input_dir: Path,
    *,
    subfolder_conditions: Optional[dict] = None,
    sec_only_folders: Optional[List[str]] = None,
    sec_only_files: Optional[List[str]] = None,
    filename_conditions: Optional[List[List[str]]] = None,
    extensions: Tuple[str, ...] = (".vsi", ".czi", ".lif", ".nd2", ".tif", ".tiff"),
) -> List[DiscoveredImage]:
    """Walk an input dir and return image paths labelled by condition.

    Parameters
    ----------
    input_dir : Path
    subfolder_conditions : optional dict
        Mapping ``{subfolder_name: condition_label}``. If a subfolder is
        not in the map, its name is used as the condition.
    sec_only_folders : list of subfolder names whose images are flagged
        ``sec_only=True``.
    sec_only_files : list of filename substrings flagged ``sec_only=True``.
    filename_conditions : optional ORDERED list of ``[substring, condition]``
        pairs (2026-05-31 Brian). For FLAT folders whose CONDITION is encoded
        in the FILENAME (e.g. ``..._NT_02.vsi`` vs ``..._MIAT-KD_05.vsi``)
        there is otherwise no way to assign distinct non-sec conditions
        (flat-mode gives every file the single ``subfolder_conditions[""]``
        label). When set, the FIRST pair whose (case-insensitive) substring
        is found in the filename sets that file's condition. Evaluated AFTER
        the ``sec_only_*`` test: sec-only files keep their forced "Sec-Only"
        label and are NOT relabelled by this map (so a ``-NT_`` substring on a
        sec-only file can't steal it back). Default ``None`` / empty =
        legacy behaviour (no filename-based condition assignment).
    extensions : tuple of accepted file extensions (lowercase).
    """
    input_dir = Path(input_dir)
    sec_only_folders = set(sec_only_folders or [])
    sec_only_files = [s.lower() for s in (sec_only_files or [])]
    subfolder_conditions = subfolder_conditions or {}
    # Normalise filename_conditions to a list of (lower-substring, label) in
    # the user-supplied order (first match wins).
    fname_conds: List[Tuple[str, str]] = []
    for pair in (filename_conditions or []):
        if pair and len(pair) >= 2 and str(pair[0]).strip():
            fname_conds.append((str(pair[0]).lower(), str(pair[1])))

    out: List[DiscoveredImage] = []

    def _add(p: Path, subfolder: str):
        name_l = p.name.lower()
        is_sec = (
            subfolder in sec_only_folders
            or any(s in name_l for s in sec_only_files)
        )
        condition = subfolder_conditions.get(subfolder, subfolder)
        if is_sec:
            # Force the "Sec-Only" label when the file/folder is sec-only,
            # so downstream stats grouping is consistent.
            condition = subfolder_conditions.get(subfolder, "Sec-Only")
        elif fname_conds:
            # Filename-substring condition assignment (flat folders with the
            # condition encoded in the file name). First matching pair wins.
            for sub_l, label in fname_conds:
                if sub_l in name_l:
                    condition = label
                    break
        out.append(DiscoveredImage(
            path=p,
            condition=condition,
            sec_only=is_sec,
            subfolder=subfolder,
        ))

    # Subfolder-mode
    children = sorted(p for p in input_dir.iterdir() if p.is_dir() and not p.name.startswith("_"))
    if children:
        for sub in children:
            for f in sorted(sub.iterdir()):
                if f.suffix.lower() in extensions and not f.name.startswith("_"):
                    _add(f, sub.name)
    else:
        # Flat-mode
        for f in sorted(input_dir.iterdir()):
            if f.suffix.lower() in extensions and not f.name.startswith("_"):
                _add(f, "")
    return out
