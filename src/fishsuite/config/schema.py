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
    mode: Literal["single", "maxproj", "autofocus", "autofocus_maxproj", "3d"] = "autofocus"
    single_slice: int = 0
    start_slice: Optional[int] = None
    end_slice: Optional[int] = None
    # 2026-05-22 Brian: per-image z-window overrides. Keys are full file
    # names (matched against Path(image).name). Values are dicts that may
    # contain start_slice and/or end_slice — those override the batch
    # defaults above for that specific image. Useful when one image has
    # an out-of-focus middle (or any other reason to use a different z
    # window than the rest of the batch).
    # Example:
    #   file_overrides:
    #     "KO BIN1-100x02 (decon).vsi":
    #       start_slice: 15
    #       end_slice: 49
    #
    # 2026-05-25 Brian: per-channel z (rna_only mode). In ADDITION to
    # start_slice/end_slice (which apply to the chosen z_mode for ALL
    # channels) and single_slice (the fixed DAPI plane), a file_override
    # may carry rna_start_slice + rna_end_slice (1-indexed, inclusive).
    # When BOTH are present, rna_only mode segments nuclei on the DAPI
    # single plane as usual but detects spots on a MAXPROJ of the RNA
    # channel over [rna_start_slice, rna_end_slice] (better puncta SNR
    # without blurring the DAPI segmentation). When absent, RNA extraction
    # is unchanged (follows the global z_mode). The override dict is a
    # free-form Dict[str, Any]; recognised keys are:
    #   single_slice, start_slice, end_slice, rna_start_slice, rna_end_slice
    # Example (DAPI single plane 12, RNA maxproj over slices 8-20):
    #   file_overrides:
    #     "H9-MIAT-ASOs-_02.vsi":
    #       single_slice: 12
    #       rna_start_slice: 8
    #       rna_end_slice: 20
    file_overrides: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    # ─── autofocus_maxproj mode parameters (2026-05-24 Brian) ──────────────
    # When ``mode == "autofocus_maxproj"`` the runner computes a per-image
    # in-focus DAPI z-window using ``focus_metric`` on the DAPI channel,
    # then MIPs JUST THAT WINDOW for ALL channels (DAPI + RNA1 + RNA2).
    # This eliminates per-image file_overrides hacks for datasets where
    # the in-focus DAPI slab varies per field (e.g. BIN1 KO_100x02 needed
    # 15-49 vs the batch default 9-78).
    #
    # The window-detection algorithm is FWHM-style:
    #   1. Compute focus_metric per z-slice on DAPI.
    #   2. Find peak slice (argmax).
    #   3. Walk outward from peak in both directions while the slice's
    #      score >= focus_threshold_frac * peak_score.
    #   4. Enforce focus_window_min_slices minimum (symmetrically expand
    #      around peak if the natural window is smaller).
    #   5. Enforce focus_window_max_slices ceiling (0 = no cap).
    #   6. Constrain to [start_slice, end_slice] outer bounds if set.

    # Per-slice sharpness metric for DAPI. ``variance_of_laplacian`` is the
    # default (Pertuz et al. 2013 — standard focus operator for
    # fluorescence microscopy: variance of the Laplacian-filtered image).
    # ``tenengrad`` = sum of squared Sobel-gradient magnitude (sharper
    # gradient response, sensitive to edges). ``normalized_variance`` =
    # variance / mean^2 (brightness-invariant, simplest).
    focus_metric: Literal[
        "variance_of_laplacian", "tenengrad", "normalized_variance"
    ] = "variance_of_laplacian"

    # FWHM-style threshold: include any slice whose focus score is >=
    # this fraction of the peak slice's score. 0.5 = "include slices that
    # are at least half as in-focus as the peak", which mirrors the FWHM
    # of the focus-vs-z curve typical for a well-resolved DAPI slab.
    focus_threshold_frac: float = 0.5

    # Minimum number of slices in the returned window. Guards against
    # degenerate 1-slice windows that defeat the whole point of MIP'ing
    # the in-focus slab. If the natural FWHM window is smaller than this,
    # the window is symmetrically expanded around the peak slice.
    focus_window_min_slices: int = 3

    # Maximum number of slices in the returned window. 0 = no cap (let
    # the FWHM threshold decide). Use a hard ceiling when you want to
    # prevent runaway windows on images with broad focus profiles
    # (e.g. heavily decon'd stacks where many slices score above the
    # threshold).
    focus_window_max_slices: int = 0

    # ─── Fixed-N centered focus window (2026-05-24 Brian) ──────────────────
    # When > 0, AUTOFOCUS_MAXPROJ uses a FIXED-N centered window instead
    # of the FWHM-style (variable-N) window. The window is exactly this
    # many slices wide, centered on the peak-focus slice. Symmetric +/-
    # half on each side; clamps to outer bounds if the peak is too close
    # to an edge. When 0 (default), falls back to FWHM logic using
    # focus_threshold_frac.
    # Rationale: fixed-N keeps MIP integration depth consistent across
    # images so signal aggregation isn't confounded by per-image z-window
    # width differences.
    focus_window_fixed_n_slices: int = 0

    # ─── Min-intensity pre-filter (2026-05-24 v7 Brian) ───────────────────
    # Before computing focus scores, EXCLUDE z-slices whose mean DAPI
    # intensity is below ``focus_min_intensity_frac_of_peak * max_slice_mean``.
    # Guards against the failure mode where noisy/empty edge slices have
    # high focus-metric scores driven by noise rather than real signal
    # (variance_of_laplacian + normalized_variance both vulnerable to this).
    #
    # Default 0.0 = disabled (no pre-filter, full backward compat).
    # Recommended 0.2-0.5 when stacks have substantial empty z-padding
    # at top/bottom: 0.3 means "must be at least 30% as bright as the
    # brightest slice in the stack to be considered for focus scoring".
    focus_min_intensity_frac_of_peak: float = 0.0


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
    # 2026-05-25: speed lever for the cpsam transformer on CPU (no CUDA on
    # Brian's AMD GPU). cpsam runtime scales ~quadratically with pixel count;
    # H9 DAPI at 0.065 µm/px is heavily oversampled for segmentation (nuclei
    # ~200 px). Downsample the DAPI by this factor before cellpose, segment on
    # the smaller grid with a proportionally smaller diameter, then upsample
    # the integer labels back to full resolution. 1.0 = off. 2.0 → ~0.13 µm/px
    # (same scale as the cardiomyocyte runs) ≈ 4× faster, no quality loss.
    # Applies to the cellpose backend only.
    cellpose_downsample_factor: float = 1.0
    # 2026-05-27: OPT-IN GPU acceleration for the cpsam transformer via
    # torch-directml (Brian's AMD RX 6750 XT — no CUDA). DEFAULT "cpu" keeps
    # the production CPU path byte-for-byte unchanged. "directml" builds the
    # cellpose net in fp32, moves only the network to the DirectML device, and
    # forces the sparse flow-dynamics step back to CPU (DirectML has no sparse
    # kernel) — see core.segmentation + segment_image.segment_cellpose. Only
    # usable from an env that has torch-directml installed (e.g. fishproc_dml);
    # the production fishproc env never selects it. cellpose backend only.
    cellpose_device: Literal["cpu", "directml"] = "cpu"
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
    # 2026-05-25 Brian: also run spot detection on secondary-only (no-probe)
    # control images. Default False = legacy behavior (sec-only images skip
    # spot detection and report zero spots). When True, sec-only images run
    # the SAME BigFISH/LoG detection as normal images and their spot counts
    # flow through per_image_summary / per-nucleus CSV exactly like any other
    # image — REPORTED for background QC, never subtracted from sample images.
    detect_in_sec_only: bool = False
    # 2026-05-22 Brian: ALL post-detection spot filters OFF. NoMIP testing
    # showed the floater filter dropped 70-87% of legitimate cytoplasmic
    # Exons because Voronoi cytoplasm (40 px max expansion) doesn't reach
    # the mature-mRNA halo at distance. BigFISH LoG + manual contrast floor
    # already provide enough cleanup; the floater rule was too aggressive.
    drop_floater_spots: bool = False
    min_spot_fwhm_px: float = 0.0
    max_spot_peak_robust_z: float = 0.0
    max_peak_over_p95_ratio: float = 0.0
    mask_dust_specks_min_size_px: int = 0
    mask_dust_specks_threshold_mad: float = 50.0
    mask_dust_specks_replacement: str = "median"
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


class NucleolusCfg(BaseModel):
    """DAPI-low subnuclear region detection (nucleolus).

    When enabled, finds DAPI-low connected regions inside each nucleus mask
    and classifies spots into nucleolus / nucleus-excluding-nucleolus /
    cytoplasm compartments. Also adds per-nucleus chromatin texture
    metrics (DAPI CV, heterochromatin fraction, chromatin-only mean).
    """
    enabled: bool = False  # opt-in
    intra_nuclear_percentile: float = 25.0  # bottom 25% of nuclear DAPI = nucleolus candidate
    min_area_um2: float = 1.0  # smallest acceptable nucleolus
    max_area_frac_of_nucleus: float = 0.6  # safety cap
    closing_radius_px: int = 2  # morphological smoothing
    # 2026-05-22 Brian: require nucleoli to sit at least N pixels from the
    # nucleus border. Nucleoli are typically central; this rejects edge
    # artifacts (peripheral low-DAPI rings, mask-boundary effects).
    # 5 px ≈ 0.65 µm at 0.13 µm/px. For typical 60-100 px cardiomyocyte
    # nuclei this is 5-8% of diameter — generous interior margin without
    # over-restricting nucleoli that legitimately approach the periphery.
    # Set 0 to disable.
    min_border_distance_px: int = 5


class OutputCfg(BaseModel):
    save_qc_overlays: bool = True
    save_per_image_csv: bool = True
    save_masks: bool = True
    save_publication_images: bool = True
    # 2026-05-18 Brian: stop emitting a .tif next to every .png by default —
    # cuts publication_images directory size roughly in half. Turn on only
    # when you need 16-bit TIFs for figure assembly in Illustrator/Fiji.
    save_publication_tifs: bool = False

    # ─── Pub-image contrast strategy (Fiji parity: PUB_IMG_CONTRAST_MODE) ──
    # 2026-05-18 Brian: three modes for choosing the floor/ceil applied to
    # each publication-image channel render:
    #   - "auto_batch"     : run a pre-scan over every non-sec-only image,
    #                        pool raw pixels per channel, compute ONE
    #                        floor=p(pub_contrast_floor_pct) +
    #                        ceil=p(pub_contrast_ceil_pct) per channel, and
    #                        apply that single (lo, hi) to every image in the
    #                        batch. The sec-only images render with the SAME
    #                        (lo, hi) so they correctly appear dim. This is
    #                        the default — gives true cross-image brightness
    #                        comparability (matches Fiji's batch contrast).
    #   - "auto_per_image" : legacy behavior — each image computes its own
    #                        percentile-based (lo, hi). Useful for one-off
    #                        renders where comparability doesn't matter.
    #   - "manual"         : use the manual_<channel>_min / manual_<channel>_max
    #                        pairs verbatim (Fiji's "type the Brightness/
    #                        Contrast min/max numbers" workflow). When a
    #                        channel pair is None, that channel falls back to
    #                        auto_per_image percentiles — so you can pin one
    #                        channel and leave the rest automatic.
    #   - "reference_image": Sam-style per-channel tuning. For each RNA
    #                        channel, a single reference image is named (e.g.
    #                        a KO image for the introns probe, a WT image for
    #                        the exons probe). On that reference image we
    #                        segment nuclei + Voronoi cytoplasm, then compute
    #                        the channel floor / ceil from the configured
    #                        anatomical REGION (cytoplasm / nucleus / all) at
    #                        the configured percentile. The resolved
    #                        (floor, ceil) is then applied verbatim to every
    #                        image in the batch (same dict semantics as
    #                        auto_batch). The rationale embedded in the
    #                        defaults: for an INTRONS probe (mostly nuclear
    #                        puncta), set floor = cytoplasm tail percentile
    #                        so cytoplasmic noise clips to black; ceil =
    #                        nuclear bright pixels. For an EXONS probe
    #                        (mostly cytoplasmic), reverse: floor = nuclear
    #                        tail (clip nucleoplasmic background), ceil =
    #                        cytoplasmic bright pixels. DAPI inherits the
    #                        auto_batch percentile path (or manual_dapi_*
    #                        when set).
    pub_contrast_mode: Literal[
        "auto_batch", "auto_per_image", "manual", "reference_image"
    ] = "auto_batch"

    # Percentile knobs used when pub_contrast_mode is "auto_batch" or
    # "auto_per_image". FISH channels (rna, rna2) share one floor/ceil pair;
    # DAPI uses a separate pair because its biology is fundamentally
    # different (nuclear signal fills most of the cell, so its background
    # floor sits much lower).
    pub_contrast_floor_pct: float = 98.0   # FISH channels — clip background
    pub_contrast_ceil_pct: float = 99.9
    # 2026-05-22 Brian: bumped DAPI floor percentile from p10 → p40.
    # Images with diffuse high-background DAPI (e.g. CAMK2D KO_9, TRANK1
    # sec-only KO_15/17/18) showed background in QC overlays because p10
    # picks up the background tail. p40 clips it without losing
    # nuclear cores (which sit well above p40).
    pub_contrast_dapi_floor_pct: float = 40.0
    pub_contrast_dapi_ceil_pct: float = 99.9

    # 2026-05-18 Brian: auto_batch / auto_per_image floor for RNA-class
    # channels (RNA1, RNA2, antibody) felt a hair too low — diffuse cytoplasmic
    # background still showed. This bump multiplies the auto-computed floor
    # by (1 + bump/100) AFTER percentile selection, ONLY for RNA-class
    # channels. DAPI is unaffected (its histogram is structurally different —
    # bright nuclei against dark background, and the existing 10/99.9 pair
    # already handles it cleanly). Set to 0 to disable.
    pub_contrast_rna_floor_bump_pct: float = 10.0

    # Manual per-channel min/max (None = inherit auto_per_image percentiles
    # for that channel). When pub_contrast_mode == "manual" and BOTH the
    # min and max for a channel are non-None, they're applied verbatim. If
    # only one is set, the other falls back to the per-image percentile so
    # the user can pin just a floor or just a ceiling.
    manual_dapi_min: Optional[float] = None
    manual_dapi_max: Optional[float] = None
    manual_rna_min: Optional[float] = None
    manual_rna_max: Optional[float] = None
    manual_rna2_min: Optional[float] = None
    manual_rna2_max: Optional[float] = None
    manual_antibody_min: Optional[float] = None
    manual_antibody_max: Optional[float] = None

    # ─── reference_image mode params (Sam-style per-channel tuning) ────────
    # The reference image is referenced by basename (matched against the
    # discovered images' Path.name). RNA1 reference: pick the image where
    # RNA1 nuclear puncta should be crisp and cytoplasm dark (e.g. a KO
    # image for an introns probe). The floor is set to
    # ``manual_rna_floor_pct`` percentile of the configured REGION pixels in
    # that reference image; the ceiling is the
    # ``manual_rna_ceil_pct`` percentile of the configured ceil region.
    # Defaults below encode the introns/exons rationale: introns floor =
    # cytoplasm tail (so cyto clips), ceil = nuclear bright pixels; exons
    # reverse it (floor = nuclear tail, ceil = cytoplasmic bright pixels).
    # If either reference image is missing from the discovered batch, the
    # runner logs a warning and falls back to auto_batch percentiles for
    # that channel.
    manual_rna_reference_image: Optional[str] = None       # basename of the .vsi
    manual_rna_floor_region: Literal["cytoplasm", "nucleus", "all"] = "cytoplasm"
    manual_rna_floor_pct: float = 99.5
    manual_rna_ceil_region: Literal["cytoplasm", "nucleus", "all"] = "nucleus"
    manual_rna_ceil_pct: float = 99.5

    manual_rna2_reference_image: Optional[str] = None
    manual_rna2_floor_region: Literal["cytoplasm", "nucleus", "all"] = "nucleus"
    manual_rna2_floor_pct: float = 99.5
    manual_rna2_ceil_region: Literal["cytoplasm", "nucleus", "all"] = "cytoplasm"
    manual_rna2_ceil_pct: float = 99.5

    # 2026-05-20 Brian's PI Sam: apply the publication-image contrast floor
    # (resolved from manual / auto_batch / reference_image mode) as a hard
    # threshold on per-pixel intensity quantification too. When True, the
    # per-nucleus / per-cell intensity columns get an "above-floor" variant
    # that excludes pixel values below the channel's display floor — same
    # threshold the viewer's eye is using to judge nuclear-vs-cytoplasmic
    # signal. The raw (no-floor) columns remain present for back-compat.
    apply_pub_contrast_floor_to_analysis: bool = False

    # 2026-05-20 Brian: ALSO filter detected spots by the pub-image contrast
    # floor. When True, BigFISH still does its LoG-based detection (unchanged),
    # but spots whose peak intensity falls below the channel's resolved floor
    # are dropped post-detection. Useful when the visual floor is the user's
    # trusted "signal vs noise" cutoff and they want spot counts to reflect
    # the same cut.
    apply_pub_contrast_floor_to_spots: bool = False

    # 2026-05-25 Brian: configurable publication / QC scale bar. These flow
    # into the output module's render globals at run start (runner.run_batch
    # sets output.SCALEBAR_UM / output.SCALEBAR_FONT_PX from these). Defaults
    # match the historical module constants so legacy configs render byte-
    # identical scale bars. ``scalebar_um`` is the bar length in microns;
    # ``scalebar_font_px`` is the label font size in pixels.
    scalebar_um: float = 50.0
    scalebar_font_px: int = 32

    prefix: str = ""


class ParallelCfg(BaseModel):
    workers: int | str = "auto"  # int or "auto"
    # 2026-05-27 PERF: asymmetric parallelism is now the FISHSUITE DEFAULT.
    # Empirically validated 3.9x end-to-end speedup on the H9 MIAT-KD ASO
    # cohort on Brian's box (12 physical / 24 logical / 128 GB) with
    # byte-identical numeric results vs the legacy serial path
    # (F:\Image Analysis Work\H9-MIAT-KD\_gpu_accel_investigation\PIPELINE_PERF.md).
    #
    #   seg_workers     — process count for the segmentation pre-scan
    #                     (cpsam is memory-heavy on CPU; on directml it MUST
    #                     stay 1-2 since processes can't share the 12 GB VRAM).
    #                     DEFAULT "auto" -> memory + core aware count on CPU
    #                     (caps at 8; ~6 on Brian's box), forced to 1 on the
    #                     directml device. 1 = legacy serial (still supported).
    #   main_workers    — process count for the main per-image pass (BigFISH +
    #                     measurement + figure rendering). Memory-light and
    #                     embarrassingly parallel. Default stays at 1
    #                     (memory-conservative; raising it 4x's transient RAM
    #                     during the figure-render phase, opt-in per-preset).
    #                     "auto" -> high on CPU.
    #   threads_per_worker — cap OMP/MKL/numexpr/torch threads INSIDE each
    #                     worker so N_workers x threads stays <= logical cores
    #                     (avoid BLAS oversubscription). DEFAULT 4 — with
    #                     seg_workers='auto' resolving to ~6, 6 x 4 = 24
    #                     matches the logical-core count exactly. 0 disables
    #                     the cap (legacy behavior; lets BLAS pick).
    seg_workers: int | str = "auto"
    main_workers: int | str = 1
    threads_per_worker: int = 4


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
    nucleolus: NucleolusCfg = Field(default_factory=NucleolusCfg)
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
