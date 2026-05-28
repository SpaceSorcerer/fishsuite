"""Image-level pre-processing: dust speck / bright-artifact removal.

Detects pixels that are (a) far above the background and (b) part of a
connected bright region LARGER than a real diffraction-limited RNA-FISH
spot. Replaces those pixels with a local-median substitute so they no
longer contaminate spot detection, per-pixel intensity quantification,
or publication-image rendering.

Real RNA-FISH spots: ~5-20 px area at 0.13 µm/px (FWHM ~2-3 px).
Dust specks / fluorescent debris: usually 30-1000+ px, often in
saturated clusters.

Public API:
    mask_dust_specks(image, *, min_speck_size_px, brightness_threshold_mad)
        -> (cleaned_image, n_specks_found, total_speck_pixels)
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def mask_dust_specks(
    image: np.ndarray,
    *,
    min_speck_size_px: int = 100,
    brightness_threshold_mad: float = 50.0,
    replacement: str = "median",
) -> Tuple[np.ndarray, int, int]:
    """Detect and replace large bright artifacts in a 2D image.

    Algorithm:
      1. Compute robust threshold = median + brightness_threshold_mad * MAD
         (MAD scaled by 1.4826 to approximate σ for normal data).
      2. Mask all pixels above the threshold (bright pixels).
      3. Connected-component label the bright mask.
      4. Components with area ≥ min_speck_size_px are flagged as specks.
         Smaller components (real spots) are left alone.
      5. Replace speck pixels with the per-image median.

    Args:
        image: 2D numpy array.
        min_speck_size_px: minimum connected-component area to be treated
            as a dust speck (default 100; real spots are usually < 25 px).
        brightness_threshold_mad: how many MADs above the median a pixel
            must be to enter the bright-mask candidate set.
        replacement: "median" (use per-image median) or "local"
            (use 5-px-ring local median; slower).

    Returns:
        Tuple of (cleaned_image, num_specks_found, total_speck_pixels).
        Original image is NOT modified in place.
    """
    if image.ndim != 2:
        raise ValueError(f"mask_dust_specks expects 2D image, got shape {image.shape}")

    arr = image.astype(np.float64, copy=True)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    if mad <= 0:
        # uniform image — nothing to mask
        return image.copy(), 0, 0
    thr = med + brightness_threshold_mad * mad * 1.4826

    bright_mask = arr > thr
    if not bright_mask.any():
        return image.copy(), 0, 0

    try:
        from scipy.ndimage import label as cc_label
    except Exception:
        from skimage.measure import label as cc_label

    labels, n_components = cc_label(bright_mask)
    if n_components == 0:
        return image.copy(), 0, 0

    speck_mask = np.zeros_like(bright_mask)
    n_specks = 0
    for comp_id in range(1, int(labels.max()) + 1):
        comp = labels == comp_id
        comp_size = int(comp.sum())
        if comp_size >= min_speck_size_px:
            speck_mask |= comp
            n_specks += 1

    total_speck_px = int(speck_mask.sum())
    if total_speck_px == 0:
        return image.copy(), 0, 0

    cleaned = image.copy()
    if replacement == "median":
        cleaned[speck_mask] = type(image[0, 0])(med)
    elif replacement == "local":
        # Local 5-px-ring median substitution — slower but smoother.
        # Dilate the speck mask, compute median of NON-speck pixels in
        # the dilated ring, fill speck pixels with that.
        try:
            from scipy.ndimage import binary_dilation
            ring = binary_dilation(speck_mask, iterations=5) & ~speck_mask
            if ring.any():
                local_med = float(np.median(image[ring]))
                cleaned[speck_mask] = type(image[0, 0])(local_med)
            else:
                cleaned[speck_mask] = type(image[0, 0])(med)
        except Exception:
            cleaned[speck_mask] = type(image[0, 0])(med)
    else:
        raise ValueError(f"replacement must be 'median' or 'local', got {replacement!r}")

    return cleaned, n_specks, total_speck_px


def mask_dust_specks_3d(
    stack: np.ndarray,
    *,
    min_speck_size_px: int = 100,
    brightness_threshold_mad: float = 50.0,
    replacement: str = "median",
) -> Tuple[np.ndarray, int, int]:
    """3D wrapper: apply mask_dust_specks slice-by-slice."""
    if stack.ndim == 2:
        return mask_dust_specks(
            stack,
            min_speck_size_px=min_speck_size_px,
            brightness_threshold_mad=brightness_threshold_mad,
            replacement=replacement,
        )
    if stack.ndim != 3:
        raise ValueError(f"stack must be 2D or 3D, got shape {stack.shape}")
    cleaned = np.empty_like(stack)
    total_specks = 0
    total_px = 0
    for z in range(stack.shape[0]):
        cleaned_z, n_specks, n_px = mask_dust_specks(
            stack[z],
            min_speck_size_px=min_speck_size_px,
            brightness_threshold_mad=brightness_threshold_mad,
            replacement=replacement,
        )
        cleaned[z] = cleaned_z
        total_specks += n_specks
        total_px += n_px
    return cleaned, total_specks, total_px
