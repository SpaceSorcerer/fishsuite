"""The one lateral pixel-size resolver (2026-10-10): CLI value, then the recorded value (preset, manifest
or image metadata), then FALLBACK_PIXEL_UM with a logged WARNING. The fallback is defined only here;
tests/test_pixel_size_2026_10_10.py fails on any new bare 0.13 literal elsewhere in src/."""
from __future__ import annotations

import logging
import math

FALLBACK_PIXEL_UM = 0.13   # MIAT-QKI hESC 100x, binned (130 nm/px); never correct for unbinned 65 nm sets

log = logging.getLogger(__name__)


def _valid(value) -> bool:
    try:
        return math.isfinite(float(value)) and float(value) > 0
    except (TypeError, ValueError):
        return False


def resolve_pixel_um(*, cli=None, recorded=None, where: str = "") -> float:
    """Pixel size in um: ``cli`` if valid, else ``recorded`` if valid, else FALLBACK_PIXEL_UM + WARNING."""
    for value in (cli, recorded):
        if _valid(value):
            return float(value)
    log.warning("pixel size not given%s; falling back to %g um/px (scale bars, areas and distances assume it)",
                f" for {where}" if where else "", FALLBACK_PIXEL_UM)
    return FALLBACK_PIXEL_UM


def nm_to_um(voxel_xy_nm):
    """voxel_xy_nm / 1000, or None when the recorded value is missing, non-finite or not positive."""
    return float(voxel_xy_nm) / 1000.0 if _valid(voxel_xy_nm) else None
