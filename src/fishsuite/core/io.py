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
from typing import Optional, Tuple, List

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
    raise ValueError(f"Unknown z_mode {z_mode!r}")


def _autofocus_plane(stack: np.ndarray) -> np.ndarray:
    """Pick the sharpest 2D plane from a 3D stack by normalized Laplacian variance."""
    from scipy import ndimage as ndi
    if stack.ndim != 3:
        raise ValueError("expected 3D stack")
    if stack.shape[0] == 1:
        return stack[0]
    best_idx = 0
    best_score = -np.inf
    for z in range(stack.shape[0]):
        plane = stack[z].astype(np.float32)
        # Normalize by mean to make the score insensitive to brightness
        mean = plane.mean()
        if mean <= 0:
            score = 0.0
        else:
            lap = ndi.laplace(plane / mean)
            score = float(np.var(lap))
        if score > best_score:
            best_score = score
            best_idx = z
    return stack[best_idx]


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
    extensions : tuple of accepted file extensions (lowercase).
    """
    input_dir = Path(input_dir)
    sec_only_folders = set(sec_only_folders or [])
    sec_only_files = [s.lower() for s in (sec_only_files or [])]
    subfolder_conditions = subfolder_conditions or {}

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
