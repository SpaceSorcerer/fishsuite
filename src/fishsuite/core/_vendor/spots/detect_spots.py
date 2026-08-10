#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Detect RNA-FISH spots in a 2D (or 3D) image using one of two backends.

Backends:
    - log     : Laplacian-of-Gaussian (skimage.feature.blob_log). Equivalent
                in spirit to TrackMate's LoG detector. Threshold is required.
    - bigfish : bigfish.detection.detect_spots. Auto Otsu-on-LoG threshold —
                self-calibrating. Built explicitly for RNA FISH puncta.

Per-spot output schema (CSV) — uniform across backends and 2D/3D:

    spot_id, x_px, y_px, z_slice,
    intensity_peak, intensity_mean,
    integrated_intensity,         # analytical Gaussian-fit volume integral
    local_snr,                    # peak / std(local-bg-ring)
    sigma_xy_px, fwhm_xy_px,      # in-plane Gaussian shape
    sigma_z_px, fwhm_z_px,        # axial shape (3D only; NaN otherwise)
    volume_vox, volume_um3,       # ellipsoid volume at half-max
    anisotropy,                   # FWHM_z / FWHM_xy (PSF check; ~2.5-3 ideal)
    fit_ok,                       # 1 = Gaussian fit converged, 0 = fell back
    diameter_px

Plus a one-line metadata stamp on stderr summarizing the threshold used,
elapsed time, and total count.

Usage
-----
    python -m spots.detect_spots --input rna.tif --backend bigfish \\
        --output spots.csv [--voxel-size-nm 108]
        [--threshold 0.5]      # log only; bigfish auto-calibrates
        [--spot-radius-nm 150] # bigfish only

Exit codes
----------
    0  success
    2  backend not installed
    3  bad input image
    1  any other error
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import tifffile


def _normalize_to_2d(img: np.ndarray) -> np.ndarray:
    """Squeeze single-frame 3D arrays to 2D; reject true volumes for now."""
    if img.ndim == 3 and img.shape[0] == 1:
        return img[0]
    if img.ndim == 2:
        return img
    raise ValueError(
        f"Expected 2D image; got shape {img.shape}. 3D not yet wired."
    )


# ---------------------------------------------------------------------------
# Backend: skimage Laplacian-of-Gaussian (TrackMate-equivalent)
# ---------------------------------------------------------------------------

def detect_spots_log(
    img: np.ndarray,
    *,
    spot_radius_px: float = 2.5,
    threshold: float = 0.05,
    overlap: float = 0.0,
    num_sigma: int = 3,
) -> tuple[np.ndarray, float]:
    """skimage.feature.blob_log on a 2D image.

    Returns:
        spots: (N, 3) array of (y, x, sigma) — convert to (x_px, y_px) for output
        threshold_used: the threshold passed to blob_log

    Notes:
        - blob_log expects float input, normalized to ~[0, 1] for stable
          threshold semantics. We min-max normalize internally.
        - threshold is in the LoG-filtered-image domain, which is why TrackMate
          values (~0.5) don't translate directly. blob_log's natural scale
          gives a useful spot count around 0.02–0.10 on FISH-like data.
    """
    from skimage import feature, exposure

    img2d = _normalize_to_2d(img).astype(np.float32)
    img_norm = exposure.rescale_intensity(img2d, out_range=(0.0, 1.0))

    sigma_min = spot_radius_px / np.sqrt(2)  # blob_log convention
    sigma_max = spot_radius_px * np.sqrt(2)

    blobs = feature.blob_log(
        img_norm,
        min_sigma=sigma_min,
        max_sigma=sigma_max,
        num_sigma=num_sigma,
        threshold=threshold,
        overlap=overlap,
    )
    return blobs, threshold


# ---------------------------------------------------------------------------
# Backend: BigFISH detect_spots (auto-thresholded)
# ---------------------------------------------------------------------------

def detect_spots_bigfish(
    img: np.ndarray,
    *,
    voxel_size_nm: float = 108.0,
    spot_radius_nm: float = 150.0,
    voxel_z_nm: float = 230.0,
    spot_radius_z_nm: float = 300.0,
    threshold: float | None = None,
) -> tuple[np.ndarray, float]:
    """bigfish.detection.detect_spots on 2D OR 3D images.

    Auto-dispatches based on input dimensionality:
        2D image  → 2D detection, returns (N, 2) array of (y, x)
        3D stack  → 3D detection, returns (N, 3) array of (z, y, x)

    BigFISH applies a LoG filter at the spot scale, then thresholds the
    LoG response. By default the threshold is found automatically via Otsu
    on the LoG image, which calibrates to whatever signal/noise level the
    image has — no manual tuning per image.

    Args:
        voxel_size_nm: in-plane pixel size in nm.
        spot_radius_nm: expected in-plane spot radius in nm.
        voxel_z_nm: axial pixel size in nm (3D only). Confocal typical: 200-300.
        spot_radius_z_nm: axial spot radius in nm (3D only). 2-3x voxel_z is typical.
        threshold: pass a fixed threshold to override BigFISH's auto-Otsu.
    """
    try:
        from bigfish.detection import detect_spots as bf_detect_spots
    except ImportError as exc:
        raise SystemExit(
            f"bigfish not installed in this Python env: {exc}\n"
            "Install with: pip install big-fish"
        )

    if img.ndim == 3 and img.shape[0] > 1:
        # True 3D
        spots, used_threshold = bf_detect_spots(
            images=img,
            threshold=threshold,
            return_threshold=True,
            voxel_size=(int(voxel_z_nm), int(voxel_size_nm), int(voxel_size_nm)),
            spot_radius=(int(spot_radius_z_nm), int(spot_radius_nm), int(spot_radius_nm)),
        )
    else:
        # 2D (or single-frame 3D — squeeze to 2D)
        img2d = _normalize_to_2d(img)
        spots, used_threshold = bf_detect_spots(
            images=img2d,
            threshold=threshold,
            return_threshold=True,
            voxel_size=(int(voxel_size_nm), int(voxel_size_nm)),
            spot_radius=(int(spot_radius_nm), int(spot_radius_nm)),
        )
    return spots, float(used_threshold)


# ---------------------------------------------------------------------------
# Per-spot Gaussian fitting — gives sub-pixel localization, sigma in xy/z,
# integrated intensity, and a volume estimate.
# ---------------------------------------------------------------------------

def _fit_gaussian_2d(patch: np.ndarray, x_init: float, y_init: float, sigma_init: float = 1.5):
    """Fit a 2D Gaussian over a patch. Patch coordinates are local (0,0)-based
    relative to the patch top-left.

    Returns (amplitude, x_fit, y_fit, sigma, background) or None on failure.
    """
    try:
        from scipy.optimize import curve_fit
    except ImportError:
        return None

    H, W = patch.shape
    if H < 3 or W < 3:
        return None

    yy, xx = np.mgrid[0:H, 0:W]

    def model(coords, A, x0, y0, sigma, B):
        x, y = coords
        return A * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2.0 * sigma ** 2)) + B

    bg0 = float(np.median(patch))
    A0 = float(patch.max()) - bg0
    if A0 <= 0:
        return None
    try:
        popt, _ = curve_fit(
            model, (xx.ravel(), yy.ravel()), patch.ravel().astype(np.float64),
            p0=[A0, x_init, y_init, sigma_init, bg0],
            maxfev=300,
        )
    except (RuntimeError, ValueError, TypeError):
        return None

    A, x_fit, y_fit, sigma, B = popt
    sigma = abs(sigma)
    # Reject pathological fits: amplitude must be positive, sigma must be in
    # FISH-puncta range. At 108 nm/px, real puncta have sigma ~1.2-2.0 px;
    # sigma > 3 px almost always means the fit drifted into adjacent spots
    # in dense regions. Empirically validated against the H9 MIAT dataset
    # where 14% of "successful" fits at the old loose bound came back with
    # sigma > 3 px and inflated integrated intensities by 2-5x.
    if A <= 0 or sigma < 0.5 or sigma > 3.0:
        return None
    # Centroid must remain inside the patch — reject fits that drifted out
    if x_fit < 0 or x_fit >= W or y_fit < 0 or y_fit >= H:
        return None
    return float(A), float(x_fit), float(y_fit), float(sigma), float(B)


def _fit_gaussian_3d(patch: np.ndarray, x_init: float, y_init: float, z_init: float,
                     sigma_xy_init: float = 1.5, sigma_z_init: float = 1.0):
    """Fit a 3D Gaussian (axially anisotropic) over a 3D patch.

    Returns (amplitude, x_fit, y_fit, z_fit, sigma_xy, sigma_z, background) or None.
    """
    try:
        from scipy.optimize import curve_fit
    except ImportError:
        return None

    if patch.ndim != 3 or any(s < 3 for s in patch.shape):
        return None

    Z, Y, X = patch.shape
    zz, yy, xx = np.mgrid[0:Z, 0:Y, 0:X]

    def model(coords, A, x0, y0, z0, sxy, sz, B):
        x, y, z = coords
        return A * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2.0 * sxy ** 2)
                          - (z - z0) ** 2 / (2.0 * sz ** 2)) + B

    bg0 = float(np.median(patch))
    A0 = float(patch.max()) - bg0
    if A0 <= 0:
        return None
    try:
        popt, _ = curve_fit(
            model,
            (xx.ravel(), yy.ravel(), zz.ravel()),
            patch.ravel().astype(np.float64),
            p0=[A0, x_init, y_init, z_init, sigma_xy_init, sigma_z_init, bg0],
            maxfev=400,
        )
    except (RuntimeError, ValueError, TypeError):
        return None

    A, x_fit, y_fit, z_fit, sxy, sz, B = popt
    sxy = abs(sxy); sz = abs(sz)
    # FISH-puncta sanity bounds. In-plane: same as the 2D path. Axial:
    # bound widened from 3.5 to 5.0 (2026-05-05) — at 230 nm voxel
    # spacing, real puncta with FWHM_z ~1 µm have sigma_z ~2 px and the
    # confocal PSF tail can push converged sigma_z to 3-4 px legitimately.
    # The old 3.5 cap rejected ~56% of 3D fits in v22; 5.0 recovers them
    # while still flagging clumps / drifted fits as outliers.
    if A <= 0 or sxy < 0.5 or sxy > 3.0 or sz < 0.3 or sz > 5.0:
        return None
    if x_fit < 0 or x_fit >= X or y_fit < 0 or y_fit >= Y or z_fit < 0 or z_fit >= Z:
        return None
    return float(A), float(x_fit), float(y_fit), float(z_fit), float(sxy), float(sz), float(B)


_FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))  # ≈ 2.3548 — sigma → FWHM


# ---------------------------------------------------------------------------
# Colocalization pairing (RNA-RNA, two-probe).
#
# SCAFFOLDING — wired into the schema but not yet invoked from the pipeline.
# Activated when both spot lists are non-empty and a pairing radius is set.
# Returns, for each spot in `a_spots`, the index of the closest spot in
# `b_spots` within the radius (or -1) and the distance to that partner.
# ---------------------------------------------------------------------------

def pair_spots_by_distance(
    a_spots: np.ndarray, b_spots: np.ndarray,
    radius_xy_px: float, radius_z_px: float = 0.0,
    is_3d: bool = False,
) -> tuple[list[int], list[float]]:
    """Greedy nearest-neighbor coloc pairing. O(N*M) — fine for FISH where
    N, M are usually < 5000. Returns (partner_idx, partner_dist) lists of
    length N (one per a_spot).

    a_spots, b_spots: arrays from detect_spots_bigfish/log:
        2D: (N, 2) of (y, x)
        3D: (N, 3) of (z, y, x)

    radius_xy_px : max in-plane distance to count as colocalized.
    radius_z_px  : max axial distance (3D only).
    """
    n = len(a_spots)
    if n == 0 or len(b_spots) == 0:
        return [-1] * n, [float("nan")] * n
    partner_idx = [-1] * n
    partner_dist = [float("nan")] * n
    r2_xy = float(radius_xy_px) ** 2

    for i in range(n):
        if is_3d:
            za, ya, xa = a_spots[i]
        else:
            ya, xa = a_spots[i][:2]
            za = 0
        best_j = -1
        best_d = float("inf")
        for j in range(len(b_spots)):
            if is_3d:
                zb, yb, xb = b_spots[j]
                if abs(za - zb) > radius_z_px:
                    continue
            else:
                yb, xb = b_spots[j][:2]
                zb = 0
            dxy2 = (ya - yb) ** 2 + (xa - xb) ** 2
            if dxy2 > r2_xy:
                continue
            d = float(np.sqrt(dxy2 + (za - zb) ** 2))
            if d < best_d:
                best_d = d
                best_j = j
        if best_j >= 0:
            partner_idx[i] = best_j
            partner_dist[i] = best_d
    return partner_idx, partner_dist


# ---------------------------------------------------------------------------
# Common per-spot metric extraction (works for both backends)
# ---------------------------------------------------------------------------

def annotate_spots(
    img: np.ndarray, spots: np.ndarray, disk_radius_px: int = 3, is_3d: bool = False,
    voxel_size_nm: float = 108.0, voxel_z_nm: float = 230.0,
    fit_window_xy: int = 4, fit_window_z: int = 2,
) -> list[dict]:
    """Build per-spot dicts with peak / mean intensity, Gaussian-fit shape
    parameters (sigma_xy, sigma_z, FWHM, anisotropy), integrated intensity,
    volume in voxels and µm³, and local SNR. Uniform schema regardless of
    backend or 2D/3D input.

    spots shape:
        2D: (N, 2) for bigfish (y, x) or (N, 3) for blob_log (y, x, sigma)
        3D: (N, 3) (z, y, x) from bigfish 3D
    """
    vox_xy_um = voxel_size_nm / 1000.0
    vox_z_um = voxel_z_nm / 1000.0

    nan = float("nan")

    if is_3d:
        H, W = img.shape[1], img.shape[2]
        nz = img.shape[0]
        out = []
        for sid, row in enumerate(spots, start=1):
            z = int(row[0]); y = int(row[1]); x = int(row[2])
            z = max(0, min(nz - 1, z))

            # Sample 2D disk on the spot's z-slice for peak/mean (back-compat)
            y0 = max(0, y - disk_radius_px); y1 = min(H, y + disk_radius_px + 1)
            x0 = max(0, x - disk_radius_px); x1 = min(W, x + disk_radius_px + 1)
            disk_patch = img[z, y0:y1, x0:x1].astype(np.float32)
            if disk_patch.size == 0:
                peak = mean = nan
            else:
                peak = float(disk_patch.max())
                mean = float(disk_patch.mean())

            # Local background from a wider ring on the same z-slice
            ring_r = max(disk_radius_px + 3, fit_window_xy + 2)
            ry0 = max(0, y - ring_r); ry1 = min(H, y + ring_r + 1)
            rx0 = max(0, x - ring_r); rx1 = min(W, x + ring_r + 1)
            ring_patch = img[z, ry0:ry1, rx0:rx1].astype(np.float32)
            if ring_patch.size > disk_patch.size:
                ring_mask = np.ones_like(ring_patch, dtype=bool)
                # carve out central disk
                cy = y - ry0; cx = x - rx0
                yy_r, xx_r = np.mgrid[0:ring_patch.shape[0], 0:ring_patch.shape[1]]
                ring_mask &= ((yy_r - cy) ** 2 + (xx_r - cx) ** 2) > disk_radius_px ** 2
                bg_pixels = ring_patch[ring_mask]
                bg_std = float(bg_pixels.std()) if bg_pixels.size > 1 else nan
                bg_mean = float(bg_pixels.mean()) if bg_pixels.size > 0 else nan
            else:
                bg_std = bg_mean = nan
            local_snr = ((peak - bg_mean) / bg_std) if (bg_std and bg_std == bg_std and bg_std > 0) else nan

            # 3D Gaussian fit on a small box centered on the spot
            wz = fit_window_z; wxy = fit_window_xy
            pz0 = max(0, z - wz); pz1 = min(nz, z + wz + 1)
            py0 = max(0, y - wxy); py1 = min(H, y + wxy + 1)
            px0 = max(0, x - wxy); px1 = min(W, x + wxy + 1)
            patch3d = img[pz0:pz1, py0:py1, px0:px1].astype(np.float32)
            x_local = x - px0; y_local = y - py0; z_local = z - pz0

            fit_ok = 0
            sigma_xy = sigma_z = fwhm_xy = fwhm_z = anisotropy = nan
            integrated = nan
            volume_vox = nan; volume_um3 = nan

            res = _fit_gaussian_3d(patch3d, x_local, y_local, z_local)
            if res is not None:
                A, _xf, _yf, _zf, sxy, sz, _B = res
                sigma_xy = sxy
                sigma_z = sz
                fwhm_xy = sxy * _FWHM_FACTOR
                fwhm_z = sz * _FWHM_FACTOR
                anisotropy = (fwhm_z / fwhm_xy) if fwhm_xy > 0 else nan
                # Analytical 3D Gaussian volume integral
                integrated = A * (2.0 * np.pi) ** 1.5 * (sxy ** 2) * sz
                # Half-max ellipsoid volume
                volume_vox = (4.0 / 3.0) * np.pi * (fwhm_xy / 2.0) ** 2 * (fwhm_z / 2.0)
                volume_um3 = volume_vox * (vox_xy_um ** 2) * vox_z_um
                fit_ok = 1
            else:
                # Empirical fallback: bg-subtracted disk sum, no shape info
                if disk_patch.size > 0 and bg_mean == bg_mean:
                    integrated = float((disk_patch - bg_mean).sum())

            out.append({
                "spot_id": sid,
                "x_px": x,
                "y_px": y,
                "z_slice": z + 1,  # 1-based for downstream
                "intensity_peak": round(peak, 2),
                "intensity_mean": round(mean, 2),
                "integrated_intensity": round(integrated, 2) if integrated == integrated else nan,
                "local_snr": round(local_snr, 3) if local_snr == local_snr else nan,
                "sigma_xy_px": round(sigma_xy, 3) if sigma_xy == sigma_xy else nan,
                "fwhm_xy_px": round(fwhm_xy, 3) if fwhm_xy == fwhm_xy else nan,
                "sigma_z_px": round(sigma_z, 3) if sigma_z == sigma_z else nan,
                "fwhm_z_px": round(fwhm_z, 3) if fwhm_z == fwhm_z else nan,
                "volume_vox": round(volume_vox, 2) if volume_vox == volume_vox else nan,
                "volume_um3": round(volume_um3, 4) if volume_um3 == volume_um3 else nan,
                "anisotropy": round(anisotropy, 3) if anisotropy == anisotropy else nan,
                "fit_ok": fit_ok,
                "diameter_px": float(disk_radius_px) * 2,
            })
        return out

    # 2D path
    img2d = _normalize_to_2d(img).astype(np.float32)
    H, W = img2d.shape
    out = []
    for sid, row in enumerate(spots, start=1):
        y = int(row[0]); x = int(row[1])

        y0 = max(0, y - disk_radius_px); y1 = min(H, y + disk_radius_px + 1)
        x0 = max(0, x - disk_radius_px); x1 = min(W, x + disk_radius_px + 1)
        disk_patch = img2d[y0:y1, x0:x1]
        if disk_patch.size == 0:
            peak = mean = nan
        else:
            peak = float(disk_patch.max())
            mean = float(disk_patch.mean())

        ring_r = max(disk_radius_px + 3, fit_window_xy + 2)
        ry0 = max(0, y - ring_r); ry1 = min(H, y + ring_r + 1)
        rx0 = max(0, x - ring_r); rx1 = min(W, x + ring_r + 1)
        ring_patch = img2d[ry0:ry1, rx0:rx1]
        if ring_patch.size > disk_patch.size:
            yy_r, xx_r = np.mgrid[0:ring_patch.shape[0], 0:ring_patch.shape[1]]
            cy = y - ry0; cx = x - rx0
            ring_mask = ((yy_r - cy) ** 2 + (xx_r - cx) ** 2) > disk_radius_px ** 2
            bg_pixels = ring_patch[ring_mask]
            bg_std = float(bg_pixels.std()) if bg_pixels.size > 1 else nan
            bg_mean = float(bg_pixels.mean()) if bg_pixels.size > 0 else nan
        else:
            bg_std = bg_mean = nan
        local_snr = ((peak - bg_mean) / bg_std) if (bg_std and bg_std == bg_std and bg_std > 0) else nan

        # 2D Gaussian fit
        wxy = fit_window_xy
        py0 = max(0, y - wxy); py1 = min(H, y + wxy + 1)
        px0 = max(0, x - wxy); px1 = min(W, x + wxy + 1)
        patch2d = img2d[py0:py1, px0:px1]
        x_local = x - px0; y_local = y - py0

        fit_ok = 0
        sigma_xy = fwhm_xy = nan
        integrated = nan
        volume_vox = nan; volume_um3 = nan

        res = _fit_gaussian_2d(patch2d, x_local, y_local)
        if res is not None:
            A, _xf, _yf, sigma, _B = res
            sigma_xy = sigma
            fwhm_xy = sigma * _FWHM_FACTOR
            # Analytical 2D Gaussian integral
            integrated = A * 2.0 * np.pi * (sigma ** 2)
            # Spot footprint area at half-max (acts as 2D "volume" for symmetry)
            volume_vox = np.pi * (fwhm_xy / 2.0) ** 2
            volume_um3 = volume_vox * (vox_xy_um ** 2)  # really µm² in 2D, kept for schema parity
            fit_ok = 1
        else:
            if disk_patch.size > 0 and bg_mean == bg_mean:
                integrated = float((disk_patch - bg_mean).sum())

        spot_diameter_px = (
            float(row[2]) * 2.0 if len(row) >= 3 else float(disk_radius_px) * 2
        )
        out.append({
            "spot_id": sid,
            "x_px": x,
            "y_px": y,
            "z_slice": 1,
            "intensity_peak": round(peak, 2),
            "intensity_mean": round(mean, 2),
            "integrated_intensity": round(integrated, 2) if integrated == integrated else nan,
            "local_snr": round(local_snr, 3) if local_snr == local_snr else nan,
            "sigma_xy_px": round(sigma_xy, 3) if sigma_xy == sigma_xy else nan,
            "fwhm_xy_px": round(fwhm_xy, 3) if fwhm_xy == fwhm_xy else nan,
            "sigma_z_px": nan,
            "fwhm_z_px": nan,
            "volume_vox": round(volume_vox, 2) if volume_vox == volume_vox else nan,
            "volume_um3": round(volume_um3, 4) if volume_um3 == volume_um3 else nan,
            "anisotropy": nan,
            "fit_ok": fit_ok,
            "diameter_px": round(spot_diameter_px, 2),
        })
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", required=True, help="Input RNA TIFF (2D).")
    parser.add_argument("--output", required=True, help="Output CSV of spots.")
    parser.add_argument(
        "--backend", required=True, choices=["log", "bigfish"],
        help="Spot detection backend.",
    )
    parser.add_argument("--threshold", type=float, default=None,
                        help="Detection threshold. log: required (default 0.05). "
                             "bigfish: optional (None = auto Otsu-on-LoG).")
    parser.add_argument("--threshold-multiplier", type=float, default=1.0,
                        help="Multiply the auto-computed threshold by this factor. "
                             "Values <1.0 catch MORE spots (smaller / dimmer / noisier); "
                             "values >1.0 catch FEWER (stricter). 1.0 = use raw auto threshold. "
                             "Ignored when --threshold is set explicitly. "
                             "Typical small-spot sensitivity boost: 0.7-0.8. "
                             "Aggressive: 0.5.")
    parser.add_argument("--spot-radius-px", type=float, default=2.5,
                        help="(log) expected spot radius in pixels.")
    parser.add_argument("--voxel-size-nm", type=float, default=108.0,
                        help="(bigfish) in-plane pixel size in nm.")
    parser.add_argument("--spot-radius-nm", type=float, default=150.0,
                        help="(bigfish) expected in-plane spot radius in nm.")
    parser.add_argument("--voxel-z-nm", type=float, default=230.0,
                        help="(bigfish, 3D only) axial pixel size in nm. "
                             "Confocal typical: 200-300.")
    parser.add_argument("--spot-radius-z-nm", type=float, default=300.0,
                        help="(bigfish, 3D only) axial spot radius in nm. "
                             "2-3x voxel_z is typical for FISH puncta.")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.is_file():
        sys.stderr.write(f"ERROR: input not found: {in_path}\n")
        return 3

    img = tifffile.imread(in_path)
    is_3d = img.ndim == 3 and img.shape[0] > 1

    t0 = time.time()
    if args.backend == "log" and is_3d:
        sys.stderr.write(
            "ERROR: backend='log' is 2D-only; got 3D input. Use --backend bigfish for 3D.\n"
        )
        return 3
    try:
        if args.backend == "log":
            thr = args.threshold if args.threshold is not None else 0.05
            blobs, used_thr = detect_spots_log(
                img, spot_radius_px=args.spot_radius_px, threshold=thr,
            )
        elif args.backend == "bigfish":
            # Two-pass approach if a multiplier is requested with auto threshold:
            # run BigFISH once with auto to get the Otsu-on-LoG threshold,
            # multiply, then re-run with the scaled value explicitly. This is
            # cheap because BigFISH caches the LoG-filtered image internally.
            if args.threshold is None and args.threshold_multiplier != 1.0:
                _, _auto_thr = detect_spots_bigfish(
                    img,
                    voxel_size_nm=args.voxel_size_nm,
                    spot_radius_nm=args.spot_radius_nm,
                    voxel_z_nm=args.voxel_z_nm,
                    spot_radius_z_nm=args.spot_radius_z_nm,
                    threshold=None,
                )
                _scaled = float(_auto_thr) * float(args.threshold_multiplier)
                sys.stderr.write(
                    f"  bigfish threshold multiplier: auto={_auto_thr:.4f} × "
                    f"{args.threshold_multiplier:.3f} → scaled={_scaled:.4f}\n"
                )
                blobs, used_thr = detect_spots_bigfish(
                    img,
                    voxel_size_nm=args.voxel_size_nm,
                    spot_radius_nm=args.spot_radius_nm,
                    voxel_z_nm=args.voxel_z_nm,
                    spot_radius_z_nm=args.spot_radius_z_nm,
                    threshold=_scaled,
                )
            else:
                blobs, used_thr = detect_spots_bigfish(
                    img,
                    voxel_size_nm=args.voxel_size_nm,
                    spot_radius_nm=args.spot_radius_nm,
                    voxel_z_nm=args.voxel_z_nm,
                    spot_radius_z_nm=args.spot_radius_z_nm,
                    threshold=args.threshold,
                )
    except SystemExit as e:
        sys.stderr.write(str(e) + "\n")
        return 2
    elapsed = time.time() - t0

    rows = annotate_spots(
        img, blobs, is_3d=is_3d,
        voxel_size_nm=args.voxel_size_nm, voxel_z_nm=args.voxel_z_nm,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with out_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    else:
        out_path.write_text(
            "spot_id,x_px,y_px,z_slice,intensity_peak,intensity_mean,"
            "integrated_intensity,local_snr,sigma_xy_px,fwhm_xy_px,"
            "sigma_z_px,fwhm_z_px,volume_vox,volume_um3,anisotropy,"
            "fit_ok,diameter_px\n"
        )

    # Sanity check the threshold: BigFISH's auto-Otsu can degenerate on
    # dim images (sec-only controls, weak signal) and pick a threshold
    # below the image's 50th percentile — at which point it's splitting
    # noise from noise. Emit a clear warning so the Jython side can
    # surface it and downstream QC notices the run is suspect.
    img2d_for_stats = _normalize_to_2d(img) if (not is_3d) else None
    img_p50 = float(np.median(img2d_for_stats if img2d_for_stats is not None else img))
    img_p95 = float(np.percentile(img2d_for_stats if img2d_for_stats is not None else img, 95))
    threshold_warning = ""
    if used_thr < img_p50 and len(rows) > 0:
        threshold_warning = (
            f"WARNING: BigFISH/LoG threshold ({used_thr:.4f}) is below image median "
            f"({img_p50:.4f}); detector likely picking up noise (n={len(rows)} spots)."
        )
        sys.stderr.write(threshold_warning + "\n")

    # Sidecar threshold record so the Jython wrapper can attach the
    # per-image threshold to the thresholds.csv audit trail. One-line
    # key=value text file (Jython-safe, no JSON dep).
    sidecar = out_path.with_suffix(out_path.suffix + ".threshold.txt")
    try:
        sidecar.write_text(
            f"backend={args.backend}\n"
            f"threshold={used_thr:.6f}\n"
            f"image_median={img_p50:.4f}\n"
            f"image_p95={img_p95:.4f}\n"
            f"n_spots={len(rows)}\n"
            f"warning={threshold_warning}\n",
            encoding="utf-8",
        )
    except Exception as exc:
        sys.stderr.write(f"  (note: failed to write sidecar threshold record: {exc})\n")

    print(f"backend={args.backend}  spots={len(rows)}  threshold={used_thr:.4f}  "
          f"elapsed={elapsed:.2f}s  out={out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
