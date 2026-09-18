"""Resolve physical nuclear sizes without changing legacy pixel settings."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from fishsuite.config.schema import NucleiCfg


@dataclass(frozen=True)
class ResolvedNuclearSize:
    diameter_px: float
    native_diameter_px: float
    min_area_px: float
    max_area_px: float
    reject_ghost_min_area_px: float
    border_margin_px: float
    pixel_size_um: float | None
    downsample_factor: float
    sources: dict[str, str]
    grids: dict[str, str]
    inputs: dict[str, dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "pixel_size_um": self.pixel_size_um,
            "downsample_factor": self.downsample_factor,
            "sizes": {
                name: {
                    "value": getattr(self, name),
                    "grid": self.grids[name],
                    "source": self.sources[name],
                    **self.inputs[name],
                }
                for name in self.sources
            },
        }


def resolve_nuclear_size_px(
    cfg: NucleiCfg,
    pixel_size_um: float | None,
    downsample_factor: float,
) -> ResolvedNuclearSize:
    """Diameter uses the model grid; final area and border filters use native masks.

    Pass ``native_diameter_px`` to ``segment_nuclei``: that wrapper already
    divides diameter once when it downsamples. ``diameter_px`` records the
    resulting value handed to the model. A factor <= 1 preserves the wrapper's
    existing no-downsampling behavior.
    """
    twins = (
        ("diameter_px", "expected_diameter_um", "cellpose_diameter_px", 1),
        ("min_area_px", "min_area_um2", "min_area_px", 2),
        ("max_area_px", "max_area_um2", "max_area_px", 2),
        ("reject_ghost_min_area_px", "reject_ghost_min_area_um2", "reject_ghost_min_area_px", 2),
        ("border_margin_px", "border_margin_um", "border_margin_px", 1),
    )
    explicit = cfg.model_fields_set
    for _, um_key, px_key, _ in twins:
        if getattr(cfg, um_key) is not None and um_key in explicit and px_key in explicit:
            raise ValueError(f"nuclei.{um_key} and nuclei.{px_key} were both explicitly supplied; choose one unit")
    physical = any(getattr(cfg, um_key) is not None for _, um_key, _, _ in twins)
    valid_pixel_size = pixel_size_um is not None and math.isfinite(pixel_size_um) and pixel_size_um > 0
    if physical and not valid_pixel_size:
        raise ValueError("Physical nuclei settings require a readable positive XY pixel size (um/px)")
    factor = float(downsample_factor)
    if not math.isfinite(factor):
        raise ValueError("nuclei.cellpose_downsample_factor must be finite")
    factor = max(1.0, factor)
    values = {}
    sources = {}
    grids = {}
    inputs = {}
    for name, um_key, px_key, power in twins:
        um_value = getattr(cfg, um_key)
        if um_value is not None:
            value = float(um_value) / float(pixel_size_um) ** power
            source = "um"
            input_value = um_value
            input_unit = "um2" if power == 2 else "um"
        else:
            value = getattr(cfg, px_key)
            source = "px" if px_key in explicit else "default"
            input_value = value
            input_unit = "px2" if power == 2 else "px"
        values[name] = value
        sources[name] = source
        grids[name] = "native"
        inputs[name] = {"input_value": input_value, "input_unit": input_unit}
    values["native_diameter_px"] = values["diameter_px"]
    sources["native_diameter_px"] = sources["diameter_px"]
    grids["native_diameter_px"] = "native"
    inputs["native_diameter_px"] = dict(inputs["diameter_px"])
    values["diameter_px"] /= factor
    grids["diameter_px"] = "downsampled" if factor > 1 else "native"
    return ResolvedNuclearSize(
        **values,
        pixel_size_um=float(pixel_size_um) if valid_pixel_size else None,
        downsample_factor=factor,
        sources=sources,
        grids=grids,
        inputs=inputs,
    )
