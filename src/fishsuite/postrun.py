"""Reusable helpers for measurements derived from completed FishSuite runs."""
from __future__ import annotations

import json
from numbers import Integral
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


CHANNEL_CONFIG_SOURCE = "run_config.json:config_resolved.channels"
SAMPLING_DEFAULTS = {
    "eligible_for_sampling": True,
    "sampled_in_analysis": True,
}
IMAGE_TRANSFORMS = ("none", "rot180", "rot90", "flipud")


def _channel_metadata(run_dir: Path) -> dict:
    path = Path(run_dir) / "run_config.json"
    if not path.is_file():
        return {}
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    channels = config.get("config_resolved", {}).get("channels")
    return channels if isinstance(channels, dict) else {}


def resolve_channels(
    run_dir: Path,
    role_keys: Sequence[str] = ("rna", "antibody", "dapi"),
    overrides: Mapping[str, int | None] | None = None,
) -> tuple[dict[str, int], dict[str, str]]:
    """Resolve zero-based channel indices without a silent positional default."""
    roles = tuple(str(role) for role in role_keys)
    if not roles or len(set(roles)) != len(roles):
        raise ValueError("channel roles must be nonempty and distinct")
    given = dict(overrides or {})
    unknown = sorted(set(given) - set(roles))
    if unknown:
        raise ValueError(f"channel overrides contain unknown roles: {unknown}")
    metadata = _channel_metadata(Path(run_dir))
    one_indexed = metadata.get("one_indexed", False)
    if not isinstance(one_indexed, bool):
        raise ValueError("channel one_indexed must be a boolean")
    offset = int(one_indexed)
    indices: dict[str, int] = {}
    sources: dict[str, str] = {}
    for role in roles:
        value = given.get(role)
        source = "override"
        if value is None:
            value = metadata.get(role)
            source = CHANNEL_CONFIG_SOURCE
        try:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                raise ValueError("channel index must have integer type")
            index = int(value) - (offset if source == CHANNEL_CONFIG_SOURCE else 0)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"cannot resolve the {role} channel index") from None
        if index < 0:
            raise ValueError(f"cannot resolve the {role} channel index")
        indices[role] = index
        sources[role] = source
    if len(set(indices.values())) != len(indices):
        raise ValueError(f"channel indices must be distinct, got {indices}")
    return indices, sources


def normalize_sampling_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Add true defaults only for sampling columns absent from an unsampled run."""
    normalized = frame.copy()
    defaulted = tuple(name for name in SAMPLING_DEFAULTS if name not in normalized)
    for name in defaulted:
        normalized[name] = SAMPLING_DEFAULTS[name]
    return normalized, defaulted


def transform_image(image, mode: str = "none"):
    """Apply a declared geometry transform to one two-dimensional image plane."""
    array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError("calibration image must be a two-dimensional plane")
    if mode == "none":
        return array
    if mode == "rot90" and (array.ndim < 2 or array.shape[0] != array.shape[1]):
        raise ValueError(f"rot90 needs a square image, got {array.shape}")
    if mode == "rot180":
        return np.rot90(array, 2)
    if mode == "rot90":
        return np.rot90(array, 1)
    if mode == "flipud":
        return np.flipud(array)
    raise ValueError(f"unknown image transform {mode!r}")


def footprint_union_summary(
    image2d,
    valid_mask,
    footprints_yx: Iterable[np.ndarray],
) -> dict[str, int | float]:
    """Integrate exact raster footprints, counting each pixel once per footprint."""
    image = np.asarray(image2d)
    if image.ndim != 2:
        raise ValueError("image2d must be two-dimensional")
    mask = np.asarray(valid_mask, dtype=bool)
    if mask.shape != image.shape:
        raise ValueError("valid_mask must have the same shape as image2d")
    footprints = tuple(footprints_yx)
    union = np.zeros(image.shape, dtype=bool)
    sum_pixels = 0
    sum_intensity = 0.0
    nonempty = 0
    height, width = image.shape
    for footprint in footprints:
        pixels = np.asarray(footprint)
        if pixels.size == 0:
            pixels = np.empty((0, 2), dtype=np.intp)
        if pixels.ndim != 2 or pixels.shape[1] != 2:
            raise ValueError("each footprint must have shape (n_pixels, 2)")
        if not np.issubdtype(pixels.dtype, np.integer):
            numeric = np.asarray(pixels, dtype=float)
            if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
                raise ValueError("footprint coordinates must be finite integers")
            pixels = numeric.astype(np.intp)
        else:
            pixels = pixels.astype(np.intp, copy=False)
        pixels = np.unique(pixels, axis=0)
        y, x = pixels[:, 0], pixels[:, 1]
        inside = (y >= 0) & (y < height) & (x >= 0) & (x < width)
        y, x = y[inside], x[inside]
        if y.size:
            keep = mask[y, x]
            y, x = y[keep], x[keep]
        if not y.size:
            continue
        nonempty += 1
        sum_pixels += int(y.size)
        sum_intensity += float(np.asarray(image[y, x], dtype=float).sum())
        union[y, x] = True
    union_px = int(np.count_nonzero(union))
    return {
        "n_footprints": len(footprints),
        "n_nonempty_footprints": nonempty,
        "union_px": union_px,
        "union_intensity_sum": float(np.asarray(image[union], dtype=float).sum()),
        "sum_footprint_intensity": sum_intensity,
        "overlap_px": sum_pixels - union_px,
    }
