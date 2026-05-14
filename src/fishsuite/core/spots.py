"""Spot detection wrapper around BigFISH / LoG.

Uses the Fiji-pipeline implementations verbatim where possible. Single
entry point: ``detect_spots``.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

_FIJI_PY = Path(r"F:\Image Analysis Work\image-analysis-pipeline\python")
if str(_FIJI_PY) not in sys.path:
    sys.path.insert(0, str(_FIJI_PY))


def detect_spots(
    rna: np.ndarray,
    *,
    backend: str = "bigfish",
    voxel_xy_nm: float = 65.0,
    voxel_z_nm: float = 230.0,
    spot_radius_nm: float = 150.0,
    spot_radius_z_nm: float = 300.0,
    threshold_multiplier: float = 1.0,
    threshold: Optional[float] = None,
    log_threshold: float = 0.05,
    log_spot_radius_px: float = 2.5,
) -> pd.DataFrame:
    """Detect spots in a 2D or 3D RNA image.

    Returns a DataFrame with columns:
        spot_id, x_px, y_px, z_slice, intensity_peak, threshold_used
    """
    from spots.detect_spots import detect_spots_bigfish, detect_spots_log

    if backend == "bigfish":
        # Apply threshold multiplier by first running auto, then re-running with
        # multiplied threshold. This matches the Fiji `threshold_multiplier`
        # semantic.
        spots, auto_thr = detect_spots_bigfish(
            rna,
            voxel_size_nm=voxel_xy_nm,
            spot_radius_nm=spot_radius_nm,
            voxel_z_nm=voxel_z_nm,
            spot_radius_z_nm=spot_radius_z_nm,
            threshold=None,
        )
        used_threshold = auto_thr
        if threshold is not None:
            spots, used_threshold = detect_spots_bigfish(
                rna,
                voxel_size_nm=voxel_xy_nm,
                spot_radius_nm=spot_radius_nm,
                voxel_z_nm=voxel_z_nm,
                spot_radius_z_nm=spot_radius_z_nm,
                threshold=int(threshold),
            )
        elif abs(threshold_multiplier - 1.0) > 1e-6:
            t = max(1.0, auto_thr * float(threshold_multiplier))
            spots, used_threshold = detect_spots_bigfish(
                rna,
                voxel_size_nm=voxel_xy_nm,
                spot_radius_nm=spot_radius_nm,
                voxel_z_nm=voxel_z_nm,
                spot_radius_z_nm=spot_radius_z_nm,
                threshold=int(t),
            )
    elif backend == "log":
        spots, used_threshold = detect_spots_log(
            rna,
            spot_radius_px=log_spot_radius_px,
            threshold=float(log_threshold) * float(threshold_multiplier),
        )
    else:
        raise ValueError(f"Unknown spot backend: {backend!r}")

    # Convert into a uniform DataFrame
    return _spots_to_dataframe(spots, rna, used_threshold)


def _spots_to_dataframe(spots: np.ndarray, rna: np.ndarray, threshold_used: float) -> pd.DataFrame:
    if spots is None or len(spots) == 0:
        return pd.DataFrame(
            columns=["spot_id", "x_px", "y_px", "z_slice", "intensity_peak", "threshold_used"]
        )
    rows = []
    is_3d = rna.ndim == 3
    for i, row in enumerate(spots):
        if is_3d:
            # bigfish 3D returns (z, y, x)
            z, y, x = int(row[0]), int(row[1]), int(row[2])
            try:
                ipeak = float(rna[z, y, x])
            except Exception:
                ipeak = float("nan")
        else:
            if len(row) >= 3 and rna.ndim == 2:
                # blob_log returns (y, x, sigma)
                y, x = int(row[0]), int(row[1])
            else:
                # bigfish 2D returns (y, x)
                y, x = int(row[0]), int(row[1])
            z = 0
            try:
                ipeak = float(rna[y, x])
            except Exception:
                ipeak = float("nan")
        rows.append(
            dict(
                spot_id=i,
                x_px=x,
                y_px=y,
                z_slice=z,
                intensity_peak=ipeak,
                threshold_used=float(threshold_used),
            )
        )
    return pd.DataFrame(rows)
