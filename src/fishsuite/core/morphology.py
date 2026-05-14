"""Nuclear / cytoplasmic mask helpers + spot stratification."""
from __future__ import annotations

from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd


def compute_cytoplasm_mask(
    nucleus_labels: np.ndarray,
    *,
    max_expand_px: int = 80,
) -> np.ndarray:
    """Expand each nuclear label outward by ``max_expand_px`` pixels.

    Returns a label image where each pixel is assigned to the nearest
    nucleus (Voronoi-style), bounded by ``max_expand_px``. Cytoplasm mask
    for nucleus N = (expanded == N) & (nucleus_labels != N).
    """
    from skimage.segmentation import expand_labels
    return expand_labels(nucleus_labels, distance=int(max_expand_px))


def regionprops_table(
    labels: np.ndarray,
    intensity_image: Optional[np.ndarray] = None,
    *,
    voxel_xy_nm: float = 65.0,
) -> pd.DataFrame:
    """Per-nucleus shape + intensity descriptors using scikit-image regionprops."""
    from skimage import measure
    props = [
        "label", "area", "perimeter", "centroid",
        "eccentricity", "solidity",
        "feret_diameter_max",
    ]
    if intensity_image is not None:
        props += ["mean_intensity"]
    df = pd.DataFrame(
        measure.regionprops_table(labels, intensity_image=intensity_image, properties=props)
    )
    # Convert pixel area/perimeter to microns
    um_per_px = voxel_xy_nm / 1000.0
    df["area_um2"] = df["area"] * (um_per_px ** 2)
    df["perimeter_um"] = df["perimeter"] * um_per_px
    df["feret_diameter_max_um"] = df["feret_diameter_max"] * um_per_px
    df["centroid_x_um"] = df["centroid-1"] * um_per_px
    df["centroid_y_um"] = df["centroid-0"] * um_per_px
    # Circularity
    p2 = df["perimeter"] ** 2
    df["circularity"] = np.where(p2 > 0, 4 * np.pi * df["area"] / p2, 0.0)
    df = df.rename(columns={"label": "nucleus_id"})
    return df


def stratify_spots(
    spots: pd.DataFrame,
    nucleus_labels: np.ndarray,
    *,
    cytoplasm_labels: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Annotate each spot with `nucleus_id`, `in_nucleus`, `in_cytoplasm`."""
    if spots is None or len(spots) == 0:
        return spots.copy() if spots is not None else pd.DataFrame()
    out = spots.copy()
    h, w = nucleus_labels.shape
    # Clamp coords inside image bounds
    xs = np.clip(out["x_px"].to_numpy().astype(int), 0, w - 1)
    ys = np.clip(out["y_px"].to_numpy().astype(int), 0, h - 1)
    nuc_at_spot = nucleus_labels[ys, xs]
    out["nucleus_id"] = nuc_at_spot.astype(int)
    out["in_nucleus"] = nuc_at_spot > 0
    if cytoplasm_labels is not None:
        cyt_at_spot = cytoplasm_labels[ys, xs]
        # spot is "in cytoplasm" if it's inside the expanded label but NOT the nucleus
        out["in_cytoplasm"] = (cyt_at_spot > 0) & (~out["in_nucleus"])
        # If outside both, leave both False
        # If in cytoplasm, attach the parent nucleus id (cytoplasm shares label id)
        cyto_parent = cyt_at_spot.astype(int)
        out.loc[out["in_cytoplasm"], "nucleus_id"] = cyto_parent[out["in_cytoplasm"].to_numpy()]
    else:
        out["in_cytoplasm"] = False
    return out


def per_nucleus_spot_counts(
    spots: pd.DataFrame,
    nucleus_labels: np.ndarray,
) -> pd.DataFrame:
    """Aggregate stratified spot table -> per-nucleus counts."""
    n_labels = int(nucleus_labels.max())
    rows: List[Dict] = []
    for nid in range(1, n_labels + 1):
        s_in = ((spots.get("nucleus_id", pd.Series(dtype=int)) == nid) & spots.get("in_nucleus", False)).sum()
        s_cy = ((spots.get("nucleus_id", pd.Series(dtype=int)) == nid) & spots.get("in_cytoplasm", False)).sum()
        denom = float(s_in + s_cy) if (s_in + s_cy) > 0 else float("nan")
        nuc_frac = (s_in / denom) if (s_in + s_cy) > 0 else float("nan")
        rows.append(dict(
            nucleus_id=nid,
            spots_in_nucleus=int(s_in),
            spots_in_cytoplasm=int(s_cy),
            nuclear_fraction=nuc_frac,
            cytoplasmic_fraction=(1.0 - nuc_frac) if not np.isnan(nuc_frac) else float("nan"),
        ))
    return pd.DataFrame(rows)


def nuclear_cytoplasmic_intensity(
    image: np.ndarray,
    nucleus_labels: np.ndarray,
    cytoplasm_labels: np.ndarray,
) -> pd.DataFrame:
    """Per-nucleus mean intensity in nucleus and cytoplasm, plus N/C ratio."""
    n_labels = int(nucleus_labels.max())
    rows: List[Dict] = []
    img = image.astype(np.float64)
    for nid in range(1, n_labels + 1):
        nuc_mask = nucleus_labels == nid
        cyt_mask = (cytoplasm_labels == nid) & (~nuc_mask)
        nm = float(img[nuc_mask].mean()) if nuc_mask.any() else float("nan")
        cm = float(img[cyt_mask].mean()) if cyt_mask.any() else float("nan")
        nc = (nm / cm) if (cm and not np.isnan(cm) and cm > 0) else float("nan")
        rows.append(dict(
            nucleus_id=nid,
            nuclear_mean=nm,
            cytoplasmic_mean=cm,
            nc_ratio=nc,
        ))
    return pd.DataFrame(rows)
