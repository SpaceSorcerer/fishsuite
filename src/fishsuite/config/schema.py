"""Pydantic v2 schema for fishsuite YAML configs.

Mirrors the Fiji-pipeline YAML schema where keys overlap; adds a few
fishsuite-only knobs (parallel.*, input_discovery.*).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Any, Literal

import yaml
from pydantic import BaseModel, Field


class ExperimentCfg(BaseModel):
    name: str = "experiment"
    description: str = ""
    cell_line: str = ""
    date: str = ""
    analyst: str = ""


class ConditionsCfg(BaseModel):
    mode: Literal["auto", "explicit", "subfolders"] = "subfolders"
    subfolder_conditions: Dict[str, str] = Field(default_factory=dict)
    sec_only_folders: List[str] = Field(default_factory=list)
    sec_only_files: List[str] = Field(default_factory=list)
    condition_order: List[str] = Field(default_factory=list)
    min_nuclei_for_stats: int = 6


class ChannelsCfg(BaseModel):
    analysis_mode: Literal[
        "rna_only", "rna_protein", "rna_rna", "ab_ab", "protein_only", "pub_images"
    ] = "rna_only"
    # 0 = auto-detect; otherwise 0-indexed channel
    dapi: int = -1
    rna: int = -1
    rna2: int = -1
    antibody: int = -1
    antibody2: int = -1
    # Whether the numeric values above are 0-indexed (default) or 1-indexed Fiji-style
    one_indexed: bool = False

    # Human-readable channel labels. Default values match the historical
    # generic role names so legacy configs without these keys produce the
    # same output filenames / overlays as before. Users can set e.g.
    # ``rna_label: "MIAT-Cy5"`` and the label will flow into publication
    # image filenames, QC overlay legends, and per-image label rows in
    # thresholds.csv + run_config.json. Labels are sanitized before use
    # in filenames (see ``output.sanitize_condition_for_filename``); they
    # never affect CSV column names or the underlying channel-index logic.
    dapi_label: str = "DAPI"
    rna_label: str = "RNA1"
    rna2_label: str = "RNA2"
    antibody_label: str = "Protein"
    ab2_label: str = "Protein2"

    # Per-role LUT (lookup table) names — pseudo-color used to render each
    # channel in publication / QC images. Defaults match the historical
    # Blue / Yellow / Cyan / Magenta / Green so legacy configs render byte-
    # identical output. Accepts a named color (case-insensitive):
    # blue, yellow, cyan, magenta, green, red, orange, gray, fire.
    # Unknown names fall back to gray. See
    # ``output.lut_name_to_weights`` for the full lookup.
    # 2026-05-14 Brian: standard wavelength→color convention is
    #   647/Cy5  → yellow   (RNA1 typical)
    #   561/Cy3  → magenta  (RNA2 typical)
    #   488      → green    (antibody/protein typical)
    #   DAPI/405 → blue
    # Defaults below reflect the most-common channel-role assignments.
    # The GUI's "Detect channels" button auto-suggests colors based on
    # each channel's measured emission wavelength, so dataset variants
    # (e.g. RNA1 at 561 and RNA2 at 647) re-route automatically.
    dapi_lut: str = "blue"
    rna_lut: str = "yellow"
    rna2_lut: str = "magenta"
    antibody_lut: str = "green"
    ab2_lut: str = "magenta"


class ZStackCfg(BaseModel):
    mode: Literal["single", "maxproj", "autofocus", "3d"] = "autofocus"
    single_slice: int = 0
    start_slice: Optional[int] = None
    end_slice: Optional[int] = None


class NucleiCfg(BaseModel):
    backend: Literal["stardist", "cellpose", "otsu"] = "stardist"
    prob_threshold: float = 0.5
    nms_threshold: float = 0.5
    n_tiles: Optional[int] = None
    stardist_model: str = "2D_versatile_fluo"
    stardist_gauss_sigma: float = 3.0
    stardist_postprocess: Literal["none", "dilate", "watershed_otsu", "watershed_triangle"] = "watershed_otsu"
    stardist_postprocess_dilate_px: int = 30
    stardist_postprocess_otsu_sigma: float = 2.0
    stardist_postprocess_mask_closing_px: int = 5
    min_area_px: int = 10000
    max_area_px: float = 1e12
    # Per-label boundary smoothing applied AFTER watershed/dilate postprocess.
    # 0 disables (current behavior); recommend 3-7 px to round off the sharp
    # corners introduced by StarDist's star-convex polygon predictions where
    # neighboring instances meet. See `core.segmentation._smooth_label_boundaries`.
    label_smoothing_radius_px: int = 0
    cellpose_diameter_px: float = 0.0
    cellpose_flow_threshold: float = 0.4
    cellpose_cellprob_threshold: float = 0.0
    cellpose_model_type: str = "cpsam"
    exclude_border: bool = True
    border_margin_px: int = 5


class PixelColocCfg(BaseModel):
    threshold_mode: Literal["mad", "percentile", "costes"] = "mad"
    threshold_scope: Literal["batch", "per_image"] = "batch"
    k_mad: float = 2.0
    percentile: float = 80.0


class SpotColocCfg(BaseModel):
    """Spot-to-spot colocalization between two RNA channels (rna_rna mode).

    Drives the nearest-neighbor (cKDTree) pairing between rna1 and rna2 spots.
    pair_distance_um defaults to 0.3 µm — at the H9 100x voxel size (~0.065
    µm/px) this is ~4-5 px, ~roughly the diffraction limit.
    """
    pair_distance_um: float = 0.3
    report_nn_distance: bool = True


class FociChannelOverrideCfg(BaseModel):
    """Optional per-channel BigFISH parameter overrides for rna_rna mode.

    Every field defaults to ``None`` meaning "inherit from FociCfg". When a
    field is set, it replaces the shared FociCfg value for that channel only.
    The set of overrideable fields intentionally tracks the knobs Brian most
    often differs between RNA1 and RNA2 (different probe brightness or spot
    size); voxel-size / backend / threshold_override / LoG knobs stay shared
    to keep the override set small and the YAML readable.
    """
    bigfish_spot_radius_nm: Optional[float] = None
    bigfish_spot_radius_z_nm: Optional[float] = None
    threshold_multiplier: Optional[float] = None
    only_nuclear_spots: Optional[bool] = None
    # ``min_sep_px`` is consumed by the Fiji NMS pass (not the fishsuite
    # BigFISH wrapper); kept here for full per-channel parity with the Fiji
    # launcher so a single fishsuite YAML can drive both backends.
    min_sep_px: Optional[int] = None


class FociCfg(BaseModel):
    enabled: bool = True
    backend: Literal["bigfish", "log"] = "bigfish"
    bigfish_voxel_size_nm: float = 0.0  # 0 = auto
    bigfish_voxel_z_nm: float = 0.0  # 0 = auto
    bigfish_spot_radius_nm: float = 130.0
    bigfish_spot_radius_z_nm: float = 300.0
    threshold_multiplier: float = 0.7
    threshold_override: Optional[float] = None
    log_spot_radius_px: float = 2.5
    log_threshold: float = 0.05
    only_nuclear_spots: bool = False
    # Shared Fiji-NMS minimum-spot-separation knob — fishsuite's BigFISH
    # wrapper doesn't NMS today, but the field round-trips through YAML so
    # the same config drives the Fiji launcher consistently. Default 1 ≈
    # disabled.
    min_sep_px: int = 1
    # Per-channel overrides (used by rna_rna mode). When a field on either
    # override is None, the shared FociCfg value is used. ``rna_overrides``
    # applies to the first RNA channel (``channels.rna``); ``rna2_overrides``
    # applies to ``channels.rna2``.
    rna_overrides: FociChannelOverrideCfg = Field(default_factory=FociChannelOverrideCfg)
    rna2_overrides: FociChannelOverrideCfg = Field(default_factory=FociChannelOverrideCfg)

    def resolved_for(self, channel: Literal["rna", "rna2"]) -> Dict[str, Any]:
        """Return a dict of effective spot-detection params for ``channel``.

        Applies the matching per-channel override on top of the shared
        FociCfg values. Unset (``None``) overrides fall back to the shared
        value. Returned keys: ``bigfish_spot_radius_nm``,
        ``bigfish_spot_radius_z_nm``, ``threshold_multiplier``,
        ``only_nuclear_spots``, ``min_sep_px``.

        Unknown channel names raise ``ValueError`` (callers should pass only
        ``"rna"`` or ``"rna2"``).
        """
        if channel == "rna":
            ov = self.rna_overrides
        elif channel == "rna2":
            ov = self.rna2_overrides
        else:
            raise ValueError(
                f"FociCfg.resolved_for: channel must be 'rna' or 'rna2', got {channel!r}"
            )
        return {
            "bigfish_spot_radius_nm": (
                float(ov.bigfish_spot_radius_nm)
                if ov.bigfish_spot_radius_nm is not None
                else float(self.bigfish_spot_radius_nm)
            ),
            "bigfish_spot_radius_z_nm": (
                float(ov.bigfish_spot_radius_z_nm)
                if ov.bigfish_spot_radius_z_nm is not None
                else float(self.bigfish_spot_radius_z_nm)
            ),
            "threshold_multiplier": (
                float(ov.threshold_multiplier)
                if ov.threshold_multiplier is not None
                else float(self.threshold_multiplier)
            ),
            "only_nuclear_spots": (
                bool(ov.only_nuclear_spots)
                if ov.only_nuclear_spots is not None
                else bool(self.only_nuclear_spots)
            ),
            "min_sep_px": (
                int(ov.min_sep_px)
                if ov.min_sep_px is not None
                else int(self.min_sep_px)
            ),
        }


class CytoplasmCfg(BaseModel):
    enabled: bool = True
    voronoi_max_expansion_px: int = 80
    measure_nc_ratio: bool = True


class OutputCfg(BaseModel):
    save_qc_overlays: bool = True
    save_per_image_csv: bool = True
    save_masks: bool = True
    save_publication_images: bool = True
    prefix: str = ""


class ParallelCfg(BaseModel):
    workers: int | str = "auto"  # int or "auto"


class FishsuiteConfig(BaseModel):
    experiment: ExperimentCfg = Field(default_factory=ExperimentCfg)
    conditions: ConditionsCfg = Field(default_factory=ConditionsCfg)
    channels: ChannelsCfg = Field(default_factory=ChannelsCfg)
    z_stack: ZStackCfg = Field(default_factory=ZStackCfg)
    nuclei: NucleiCfg = Field(default_factory=NucleiCfg)
    pixel_coloc: PixelColocCfg = Field(default_factory=PixelColocCfg)
    spot_coloc: SpotColocCfg = Field(default_factory=SpotColocCfg)
    foci: FociCfg = Field(default_factory=FociCfg)
    cytoplasm: CytoplasmCfg = Field(default_factory=CytoplasmCfg)
    output: OutputCfg = Field(default_factory=OutputCfg)
    parallel: ParallelCfg = Field(default_factory=ParallelCfg)

    # Optional per-file selection subset. When non-empty, the batch runner
    # filters the discovered input list to only files whose ``Path.name`` (or
    # full path string) appears here. Empty list = include all discovered
    # files (legacy behavior). Values may be bare basenames (matched against
    # ``Path.name``) or absolute paths (matched against the full path); the
    # runner normalises both. Subfolder-mode condition assignment still
    # applies — selecting only files in subfolder X automatically keeps them
    # tagged with condition X. Populated from the GUI's per-file tree widget.
    input_file_subset: List[str] = Field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "FishsuiteConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data)

    def dump_yaml(self, path: Path | str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.model_dump(mode="json"), f, sort_keys=False)
