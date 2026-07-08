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
    # 2026-05-31 Brian: FLAT-folder, filename-encoded condition assignment.
    # An ORDERED list of ``[substring, condition_label]`` pairs. For datasets
    # acquired into a single flat folder where the condition lives in the file
    # NAME (e.g. ``H9-...-NT_02.vsi`` vs ``H9-...-MIAT-KD_05.vsi``) — flat-mode
    # discovery otherwise assigns every file the single ``subfolder_conditions[""]``
    # label, so NT vs KD can't be split. When non-empty, the FIRST pair whose
    # (case-insensitive) substring is found in a NON-sec-only file's name sets
    # that file's condition. Evaluated AFTER the sec_only_* test, so sec-only
    # files keep their forced "Sec-Only" label regardless. Default empty =
    # legacy behaviour (no filename-based assignment). Pairs are written as
    # 2-element lists in YAML, e.g.
    #   filename_conditions:
    #     - ["-NT_", "NT ASO"]
    #     - ["-MIAT-KD_", "KD ASO"]
    filename_conditions: List[List[str]] = Field(default_factory=list)
    condition_order: List[str] = Field(default_factory=list)
    min_nuclei_for_stats: int = 6


class ChannelsCfg(BaseModel):
    analysis_mode: Literal[
        "rna_only", "rna_protein", "rna_rna", "ab_ab", "protein_only",
        "pub_images", "if_intensity"
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

    # ─── single-plane "autofocus" mode metric (2026-05-28 Brian) ───────────
    # ONLY affects ``mode == "autofocus"`` (the single-plane pick used by
    # rna_only / rna_rna / rna_protein for the DAPI z-lock). Does NOT touch
    # autofocus_maxproj (which uses ``focus_metric`` above + a window walk).
    #
    # Default per-plane focus score is the mean-normalized Laplacian variance
    # ``var(laplace(plane/mean))``. On THICK stacks (~16µm, nz≈79; e.g. d8 cMyo
    # BIN1 datasets) the focus profile is near-flat across the cell-containing
    # depth, so a badly out-of-focus HIGH plane scores only ~1.0–1.2× the true
    # in-focus plane — noise then tips the pick to garbage upper planes
    # (measured DAPI picks z=42–67 when the in-focus nuclear plane is z≈20–24).
    #
    # When ``autofocus_intensity_weighted`` is True the score becomes
    # ``var(laplace(plane/mean)) * mean`` — the same sharpness term weighted by
    # plane brightness, which pulls the pick to the bright AND sharp nuclear
    # plane (validated: z≈18–22 on all 8 d8 cMyo Dataset A images, matching the
    # DAPI intensity peak). Leave False for thin stacks / legacy parity; set
    # True for thick stacks (the BIN1 d8 cMyo presets do).
    autofocus_intensity_weighted: bool = False

    # ─── RNA-anchored single-plane autofocus (2026-07-05 Brian) ────────────
    # ONLY affects ``mode == "autofocus"`` (the single-plane pick used by
    # rna_rna / rna_protein to z-lock every channel to ONE optical section).
    # Does NOT touch autofocus_maxproj, single, maxproj, or 3d.
    #
    # WHICH channel drives the single-plane pick:
    #   "dapi" (DEFAULT, LOCKED) — pick the sharpest DAPI plane, then read RNA
    #      + antibody at THAT plane. Byte-identical to all prior behaviour.
    #   "rna"  — pick the sharpest RNA1 plane instead, then read DAPI (seg) +
    #      RNA + antibody at THAT plane. Use when the dim single-molecule RNA
    #      target (e.g. MIAT) focuses on a DIFFERENT plane than the bright DAPI
    #      chromatin: the DAPI-best plane can read RNA out of focus, collapsing
    #      the BigFISH auto-threshold into thousands of noise "spots". Anchoring
    #      on RNA keeps the puncta in focus. The one-plane invariant (all
    #      channels at the SAME physical z, required for honest colocalization)
    #      is PRESERVED — only which channel chooses the plane changes.
    #   "auto" — compute a per-image RNA signal-quality score (dynamic range /
    #      spot-callability at the RNA-best plane) and RNA-anchor when it clears
    #      ``autofocus_auto_rna_quality_min``, else fall back to DAPI-anchor.
    #      The channel actually used is recorded per-image (``z_autofocus_channel_used``).
    #
    # Auditing: when this is "rna" or "auto", per_image_summary gains
    # ``z_autofocus_mode``, ``z_autofocus_channel_used``, ``z_plane`` (1-indexed
    # absolute), ``rna_focus_score``, ``rna_dynamic_range`` and
    # ``rna_n_confident_spots`` columns (plus the auto decision score/threshold
    # in "auto" mode). Default "dapi" emits NONE of these -> byte-identical CSV.
    autofocus_channel: Literal["dapi", "rna", "auto"] = "dapi"

    # Threshold on the per-image RNA dynamic-range / spot-callability score
    # used by ``autofocus_channel == "auto"`` to decide RNA-anchor vs DAPI-anchor.
    # The score is ``(p99.9 - median) / (1.4826 * MAD)`` of the RNA-best plane —
    # a robust SNR proxy for "are there real puncta standing above background?".
    # A focused single-molecule field (MIAT) scores high; a flat / out-of-focus
    # / pure-noise field scores low. RNA-anchor is used when score >= this value.
    # Only consulted when ``autofocus_channel == "auto"``. Default 3.0.
    autofocus_auto_rna_quality_min: float = 3.0

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

    # ─── Central-fraction peak guard (2026-05-31 Brian) ───────────────────
    # autofocus_maxproj robustness guard. When in (0, 1], the focus-PEAK
    # search is restricted to the central this-fraction of the (outer-
    # bounded) stack — e.g. 0.6 keeps only the middle 60% of slices eligible
    # to WIN the peak. The fixed-N / FWHM window can still extend toward an
    # edge from a central peak, but the ANCHOR can never be a true stack-edge
    # plane. Composes with autofocus_intensity_weighted: on the H9 33-plane
    # DAPI stacks, intensity-weighting fixed 9/10 windows, and 0.6 here pulled
    # the last one (_12, peak z=5 → window [1,10]) off the bottom edge to a
    # mid-stack peak z=7 → window [2,11]. Default 0.0 = disabled (whole
    # outer-bounded range eligible; full backward compat).
    focus_central_fraction: float = 0.0


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
    # 2026-05-29: OPT-IN ghost-nucleus rejection. DEFAULT False keeps every
    # other dataset/preset byte-for-byte unchanged. When True, a POST-spot-
    # detection composite rule drops empty 'ghost' shells — segmented objects
    # that are simultaneously (a) carry ZERO detected RNA spots, (b) are large
    # (area >= reject_ghost_min_area_px) and (c) flat / low-texture
    # (dapi_cv <= reject_ghost_max_dapi_cv). These are out-of-focus debris /
    # coverslip-edge ovals that cellpose segments off the aberrant border band
    # seen in some SINGLE-PLANE snaps (BIN1 d8cMyo RNase WELLS12 audit, 2026-05-29).
    # All three conditions are required — each alone is insufficient (the ghost
    # DAPI-CV band is embedded inside the real-nucleus distribution on these
    # low-contrast KO images; only the conjunction separates them with a safety
    # margin and ZERO real-nucleus loss in WT / z-stacks). See
    # core.segmentation.identify_ghost_nuclei.
    reject_ghost_nuclei: bool = False
    reject_ghost_max_dapi_cv: float = 0.12
    reject_ghost_min_area_px: int = 6000


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
    # 2026-07-04 Brian: per-channel ABSOLUTE minimum spot PEAK-intensity floor,
    # fully decoupled from the display/pub-contrast floor (manual_*_min) and from
    # output.apply_pub_contrast_floor_to_spots. When set, spots in THIS channel
    # with peak intensity < value are dropped right after BigFISH detection,
    # before stratification/pairing. None = no floor (current behavior, byte-
    # identical). Used to enforce the QKI antibody specificity floor.
    min_spot_peak_intensity: Optional[float] = None


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
    # 2026-05-28 Brian: per-channel overrides for the PROTEIN/antibody channel
    # (used by rna_protein mode, which now spot-detects the antibody channel at
    # the same depth rna_rna gives rna2). When a field is None the shared
    # FociCfg value is used — so legacy rna_protein presets without this block
    # behave identically. rna_protein maps the antibody channel into rna_rna's
    # rna2 slot internally; this override is copied into rna2_overrides on the
    # config shim, so the antibody channel can carry its own spot params (e.g. a
    # larger spot radius for diffuse XRN2 puncta) without touching rna2 in any
    # rna_rna preset.
    antibody_overrides: FociChannelOverrideCfg = Field(default_factory=FociChannelOverrideCfg)
    # 2026-05-29 Brian: intensity-based, spot-centric, FLOOR-ROBUST coloc.
    # When True, the two-channel (rna_rna / rna_protein) pipeline samples each
    # spot's RAW partner-channel local intensity (column
    # ``partner_local_mean_intensity``) and emits the per-nucleus
    # ``*_local_mean_at_*_spots`` / ``*_enrichment_at_*_spots`` columns plus the
    # per-image means (and figures 74_/75_). DEFAULT FALSE: this code path
    # currently HANGS the parallel per-image worker pool (works single-process
    # only), so it is GATED OFF until the hang is fixed. With it False the
    # two-channel analysis is byte-equivalent to the pre-feature path (binary
    # Manders/Pearson/pairing coloc only); figures 74_/75_ self-skip on the
    # absent columns while 70-73 still generate.
    compute_partner_intensity: bool = False
    # 2026-06-05 Brian: spot-detect the antibody/protein channel? RNA_PROTEIN
    # MODE ONLY. DEFAULT TRUE = existing behavior, completely unchanged: the
    # antibody channel (mapped into the rna_rna ``rna2`` slot) is BigFISH
    # spot-detected exactly as before. Set FALSE to treat the antibody channel
    # as a DIFFUSE INTENSITY channel: rna2/antibody spot detection is SKIPPED
    # (an empty spot set is produced), so a diffuse antibody stain (e.g. the
    # QKI IF, which otherwise carpets the field with meaningless "spots") is
    # NOT spotted. The INTENSITY-based colocalization is UNAFFECTED:
    # ``compute_partner_intensity`` samples the antibody-channel PIXELS at the
    # rna1 (MIAT) spots — it never reads the antibody SPOTS — so QKI-intensity-
    # at-MIAT-spots coloc, pixel-coloc (Pearson/Manders/Li), and all per-nucleus
    # antibody PIXEL metrics still compute. Consulted ONLY for the antibody
    # (rna2) channel of an rna_protein run: plain ``rna_rna`` (two real FISH
    # targets) ALWAYS spot-detects both channels regardless of this value, and
    # the rna1 channel is never affected.
    detect_antibody_spots: bool = True
    # 2026-06-05 Brian: PIPELINE-NATIVE proper colocalization statistic — the
    # per-nucleus RANDOM-POSITION NULL for "partner intensity at rna1 spots".
    # The existing ``compute_partner_intensity`` enrichment (observed / whole-
    # nucleus mean) has NO null model and NO error bar — washed out (~1.0) for a
    # dense-nuclear partner like the QKI IF. When True (requires
    # ``compute_partner_intensity: true``), for EACH nucleus the pipeline:
    #   observed   = mean over rna1 (MIAT) spots of [mean partner (QKI) intensity
    #                in a disk of radius ``partner_null_disk_px`` centered on the
    #                spot] — same disk-sample as the observed coloc;
    #   null       = the SAME number of spots placed at random IN-NUCLEUS
    #                positions, disk-sampled, repeated ``partner_null_n`` times
    #                (numpy-batched, deterministic via ``partner_null_seed``);
    #   enrichment = observed / null_mean ; z = (observed - null_mean) / null_sd.
    # Emits per nucleus ``rna2_enrichment_vs_null_at_rna1_spots`` +
    # ``rna2_null_z_at_rna1_spots`` (rna_protein relabels rna2->protein), and a
    # PER-IMAGE spot-count-weighted pooled rollup (pooled enrichment, pooled z,
    # empirical p) in per_image_summary. Reproduces the validated external script
    # ``qki_at_miat_null_ALLARMS_tm1.0.py`` (DISK_R=3 px, N_NULL=1000). DEFAULT
    # FALSE -> the columns are never emitted and the two-channel output is byte-
    # equivalent to the pre-feature path (back-compat: BIN1 / H9 unaffected).
    compute_partner_null_enrichment: bool = False
    # Number of random-position null draws per nucleus (reference scripts: 1000).
    partner_null_n: int = 1000
    # Disk radius (px) for BOTH the observed and the null partner-intensity
    # sample. CANONICAL = 3.0 px (~0.39 µm at 0.13 µm/px) — the value used by the
    # two validated reference scripts. This is INTENTIONALLY decoupled from the
    # tiny ``compute_partner_intensity`` radius (bigfish_spot_radius / voxel ≈ 1
    # px at the MIAT-QKI preset's 100 nm radius / 0.13 µm voxel), which would not
    # reproduce the validated method; pin it here so the null is reproducible.
    partner_null_disk_px: float = 3.0
    # Fixed RNG seed for the null draws -> deterministic across runs/processes.
    partner_null_seed: int = 0
    # When True AND ``nucleolus.enabled`` is True, EXCLUDE nucleolar pixels
    # (DAPI-poor voids the partner channel also avoids) from the random-null
    # sampling positions, AND drop rna1 spots whose center falls inside a
    # nucleolus before the observed/null stats. Tests whether the enrichment is
    # genuine partner-at-rna1 association rather than mutual nucleolar avoidance.
    # When nucleolus is NOT enabled this is a no-op and the null uses the whole
    # nucleus (current/legacy behavior). DEFAULT FALSE.
    exclude_nucleolus_from_partner_null: bool = False
    # 2026-06-06 Brian: PIPELINE-NATIVE coloc-figure outputs (downstream "make
    # coloc clear" null-overlay + radial QKI profile). All DEFAULT FALSE/empty
    # carrier -> emitted ONLY via the empty-default ImageResult.extra dict, so
    # the per_image / nuclei / spots tables stay byte-identical (BIN1 / H9
    # unaffected). save_partner_null_draws surfaces the per-image pooled
    # null vector (the 1000 draws the pooling block already computes then
    # discards) as ``extra["coloc_null_draws"]`` for the null-distribution
    # overlay; requires compute_partner_null_enrichment.
    save_partner_null_draws: bool = False
    # When True (requires compute_partner_intensity), sweep CONCENTRIC ANNULI of
    # increasing radius around each rna1 (MIAT) spot and compare the partner
    # (QKI) intensity in each ring to the SAME-ring intensity at random
    # in-nucleus positions (per-ring null, same seed / n as the disk null),
    # emitting a spot-count-weighted per-(image, ring) profile as
    # ``extra["coloc_radial_profile"]``. DEFAULT FALSE.
    compute_partner_radial_profile: bool = False
    # Outer-edge radii (µm) of the concentric annuli for the radial profile.
    # Ring 0 = inner disk [0, bins[0]]; ring i = (bins[i-1], bins[i]].
    partner_radial_bins_um: List[float] = Field(
        default_factory=lambda: [0.25, 0.5, 0.75, 1.0]
    )
    # 2026-06-19 Brian: PIPELINE-NATIVE rotation "proper background" null — a
    # STRICTER companion to the random-position null above. Where the position
    # null randomizes spot POSITIONS (and is inflated when the partner and the
    # rna1 spots merely SHARE a nuclear compartment), the rotation null rotates
    # each nucleus's rna1 (MIAT) spot CONSTELLATION about its OWN centroid
    # (registration-destroying, spatial-structure-PRESERVING): the spot pattern
    # keeps its internal geometry while its alignment to the fixed partner (QKI)
    # field is destroyed. observed > rotation-null therefore means the partner is
    # concentrated AT the spots BEYOND a shared compartment. When True (requires
    # ``compute_partner_intensity``), for EACH nucleus the pipeline emits per
    # nucleus ``rna2_rotation_enrichment_at_rna1_spots`` +
    # ``rna2_rotation_null_z_at_rna1_spots`` + ``rna2_rotation_null_p_at_rna1_spots``
    # + a null-calibrated ``rna2_rotation_assoc_fraction_at_rna1_spots`` (fraction
    # of observed spots whose partner disk-mean exceeds the per-nucleus rotation
    # single-position high percentile) + ``rotation_null_usable``, plus a per-image
    # spot-count-weighted pooled rollup. Uses the SAME disk radius / seed-derived
    # RNG / nucleolus handling as the position null so the two are directly
    # comparable. KEEP-N redraw (rotated spots leaving the in-mask region are
    # re-rotated by a fresh per-spot angle until in-mask, NOT dropped — dropping
    # biases enrichment LOW). N=1000 seeded. Native port of the adversarially-
    # validated ``rotation_null_prototype.py``. DEFAULT FALSE -> the columns are
    # never emitted and the output is byte-equivalent to the pre-feature path.
    compute_partner_rotation_null: bool = False
    # Number of seeded rotation iterations per nucleus (validated protocol: 1000).
    partner_rotation_n: int = 1000
    # Fixed RNG seed for the rotation/translation draws (deterministic). Distinct
    # streams are derived from this so toggling rotation never perturbs the
    # position-null draws (byte-identical pooled-null contract preserved).
    partner_rotation_seed: int = 0
    # Median first-pass in-mask retention below which a nucleus is marked NOT
    # usable for the rotation null (drops SPARSE nuclei whose constellation cannot
    # be rotated within the mask; does NOT drop dense nuclei). Validated default.
    partner_rotation_min_retention: float = 0.5
    # Percentile of the per-nucleus rotation SINGLE-POSITION partner distribution
    # used as the null-calibrated association threshold; a spot is "associated"
    # when its observed partner disk-mean exceeds this percentile. The chance
    # floor of the association fraction is (1 - pct/100), i.e. 0.05 at 95.
    partner_rotation_assoc_percentile: float = 95.0
    # When True (requires compute_partner_rotation_null), ALSO compute the
    # TRANSLATION companion null (rigid shift of the whole constellation). Emitted
    # per nucleus as ``rna2_translation_enrichment_at_rna1_spots`` +
    # ``rna2_translation_null_z_at_rna1_spots`` + ``translation_null_usable`` and a
    # per-image pooled rollup. FLAGGED UNRELIABLE for dense / space-filling spot
    # patterns (most rigid shifts push too many points out of mask -> few usable
    # nuclei); rotation is the robust headline. DEFAULT FALSE.
    compute_partner_translation_null: bool = False
    # Surface the pooled rotation null vector + pooled observed (the per-iteration
    # draws the pooling block computes then discards) as
    # ``extra["coloc_rotation_null"]`` for the null-distribution overlay; requires
    # compute_partner_rotation_null. DEFAULT FALSE -> the key is never added to
    # extra (byte-identical carrier).
    save_partner_rotation_null_draws: bool = False
    # 2026-07-07 Brian: PIPELINE-NATIVE MIAT x QKI ASSOCIATION metrics (approved
    # spec _SPEC_association_analysis_2026-07-06.md). Continuous, floor-robust,
    # AT-THE-PUNCTUM replacements for the binary "QKI-associated MIAT spots"
    # count (which conflated abundance with propensity). When True (rna_protein /
    # rna_rna path; the partner/rna2 channel is the protein, e.g. QKI), for EACH
    # rna1 (MIAT) spot the pipeline samples the partner (QKI) intensity over the
    # spot's EXACT half-max (FWHM) FOOTPRINT — the connected rna1 pixels
    # >= bg + 0.5*(local_peak - bg) within a small per-spot window — so the sample
    # SCALES with the spot's real size (a 0.3 um vs 1 um MIAT punctum covers a
    # different # of pixels), NOT a fixed disk. Emits per SPOT (spot_metrics.csv,
    # rna1 rows): ``qki_at_miat_footprint`` (raw mean QKI over the footprint),
    # ``miat_footprint_area_px``, and ``qki_footprint_enrichment``
    # (= qki_at_miat_footprint / that spot's-nucleus mean QKI; floor-robust,
    # PRIMARY per-spot metric). Emits per NUCLEUS (nuclei_metrics.csv):
    # ``qki_assoc_ratio_continuous`` (mean of qki_footprint_enrichment over the
    # nucleus's MIAT spots), ``coloc_moc`` (Manders Overlap Coefficient R = the
    # threshold-free raw-intensity cosine overlap), ``coloc_icq`` (Li's ICQ), and
    # ``qki_at_miat_foci_enrichment`` (mean QKI over the UNION of all this
    # nucleus's MIAT-footprint pixels / nuclear-mean QKI). ALSO emits three
    # EXTENSIVE (mass-based) sponge-CAPACITY columns over this nucleus's NUCLEAR
    # MIAT spots: ``qki_associated_with_miat`` (= sum of qki_at_miat_footprint *
    # miat_footprint_area_px = total QKI intensity inside MIAT-spot pixels),
    # ``miat_mass_nuclear`` (= sum of spot_peak_intensity * spot_area_px =
    # integrated MIAT signal), and ``capacity_qki_at_miat`` (= qki_associated_with_miat /
    # the nuclear QKI total ``nuclear_total_intensity_rna2`` [->
    # nuclear_total_intensity_protein] = fraction of the nucleus's QKI associated with
    # MIAT). All threshold-free and
    # emitted ALONGSIDE (never replacing) the MAD-thresholded Manders M1/M2 so a
    # gated-vs-ungated comparison is possible. When a partner (QKI) floor is set
    # via ``assoc_qki_floor`` a SECONDARY floor-gated variant
    # ``qki_assoc_ratio_gated_<floor>`` is ALSO emitted (MIAT spots whose footprint
    # QKI < floor contribute 0 -> reintroduces floor-sensitivity; shown for
    # contrast). DEFAULT FALSE -> none of these columns are emitted and the output
    # is byte-equivalent to the pre-feature path (BIN1 / H9 unaffected). Runs the
    # per-spot footprint loop in-process (like ``compute_partner_intensity``) ->
    # use -p 1 (the MIAT/QKI presets already force single-process).
    compute_footprint_enrichment: bool = False
    # Genuine partner (QKI) intensity floor for the SECONDARY floor-gated
    # association-ratio variant (``qki_assoc_ratio_gated_<floor>``). None / <= 0
    # => OFF (no gated column emitted; the continuous ratio is the primary). When
    # set, a MIAT spot whose ``qki_at_miat_footprint`` is below this value
    # contributes 0 to the gated per-nucleus ratio. Units = raw partner-channel
    # intensity (same scale as ``qki_at_miat_footprint``). DEFAULT None.
    assoc_qki_floor: Optional[float] = None

    def resolved_for(self, channel: Literal["rna", "rna2", "antibody"]) -> Dict[str, Any]:
        """Return a dict of effective spot-detection params for ``channel``.

        Applies the matching per-channel override on top of the shared
        FociCfg values. Unset (``None``) overrides fall back to the shared
        value. Returned keys: ``bigfish_spot_radius_nm``,
        ``bigfish_spot_radius_z_nm``, ``threshold_multiplier``,
        ``only_nuclear_spots``, ``min_sep_px``, ``min_spot_peak_intensity``.

        Unknown channel names raise ``ValueError`` (callers should pass only
        ``"rna"``, ``"rna2"``, or ``"antibody"``).
        """
        if channel == "rna":
            ov = self.rna_overrides
        elif channel == "rna2":
            ov = self.rna2_overrides
        elif channel == "antibody":
            ov = self.antibody_overrides
        else:
            raise ValueError(
                f"FociCfg.resolved_for: channel must be 'rna', 'rna2', or "
                f"'antibody', got {channel!r}"
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
            # 2026-07-04 Brian: per-channel absolute spot peak-intensity floor.
            # No shared FociCfg fallback (there is no FociCfg-level field) —
            # unset override => None => no floor (byte-identical legacy path).
            "min_spot_peak_intensity": (
                float(ov.min_spot_peak_intensity)
                if ov.min_spot_peak_intensity is not None
                else None
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

    # ─── Thresholded RNA intensity in compartments (2026-06-02 Brian) ──────
    # A THIRD intensity measurement, distinct from spot-based intensities
    # (rna_spot_total_peak_intensity) AND from the raw whole-nucleus pixel sum
    # (sum_rna_intensity, which has NO floor). For each nucleus, integrate the
    # RNA-channel intensity of ALL pixels whose RAW value >= this floor —
    # measured SEPARATELY within the nucleus and within the Voronoi cytoplasm.
    # This is PIXEL-thresholding (not spot detection): it captures all
    # above-floor signal (diffuse + punctate), independent of the spot-caller.
    # Mirrors a protein "threshold-and-integrate" approach. Emits, per nucleus,
    # for EACH compartment (nuclear / cyto):
    #   rna_thresh_total_intensity_*   sum of RAW intensities of >=floor pixels
    #   rna_thresh_mean_intensity_*    mean intensity of the >=floor pixels
    #   rna_thresh_pos_area_px_*       count of >=floor pixels
    #   rna_thresh_pos_fraction_*      >=floor pixel count / compartment area
    # (and the rna2_thresh_*_* equivalents in rna_rna / rna_protein modes).
    #
    # The floor is applied to the SAME RNA image plane used for spot detection
    # (the objective-window MIP). It is a SEPARATE knob from the pub-contrast
    # floor: the threshold-integrate columns are ALWAYS emitted regardless of
    # apply_pub_contrast_floor_to_* — they are NaN only when no floor can be
    # resolved at all.
    #
    # Default resolution when this field is None or <= 0:
    #   1. the resolved spot floor forwarded by the runner (the pub-contrast
    #      RNA floor — i.e. manual_rna_min when apply_pub_contrast_floor_to_spots
    #      is on), else
    #   2. manual_rna_min read directly from this config, else
    #   3. NaN (columns present but empty; schema stays stable).
    # Set this field to a positive number to pin the threshold explicitly,
    # independent of the spot floor.
    rna_intensity_threshold: Optional[float] = None
    # Second-channel (rna2 / antibody) floor for the same measurement, used by
    # rna_rna + rna_protein modes. Same None/<=0 default semantics, but the
    # default spot-floor source is the rna2 / antibody channel
    # (manual_rna2_min / manual_antibody_min).
    rna2_intensity_threshold: Optional[float] = None

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


class QcCfg(BaseModel):
    """Per-image QC-flag thresholds (ADDITIVE, 2026-06-10).

    These drive the INFORMATIONAL ``qc_*`` columns the runner merges into
    ``per_image_summary.csv``. They never drop or alter an image — they only
    decide which advisory flags fire. Defaults are neutral so legacy output is
    unaffected except for the added (advisory) columns.
    """
    # An image is flagged ``qc_low_nuclei`` when its nucleus count is below
    # this. Advisory only; the image is still analysed and reported.
    qc_min_nuclei: int = 5
    # A channel is flagged ``saturated_<role>`` when the fraction of pixels at
    # or above near-full-scale (0.999 * dtype_max) exceeds this fraction.
    qc_saturated_frac: float = 0.01
    # Minimum DAPI focus score (variance-of-Laplacian) below which an image is
    # flagged ``low_focus``. Default 0.0 -> focus NEVER flags (opt-in).
    qc_min_focus_score: float = 0.0

    # ─── RNA1 over-detection guard (2026-07-05 Brian, bulletproofing) ───────
    # ADVISORY-ONLY. Flags images whose RNA1 spots-per-nucleus is implausibly
    # high, which is the classic symptom of an out-of-focus dim RNA channel
    # collapsing the BigFISH auto-threshold into thousands of noise "spots"
    # (e.g. well11 MIAT-KD ~1500 spots/nucleus). It NEVER changes detection or
    # drops an image — it sets ``qc_overdetect_rna1`` + adds ``overdetect_rna1``
    # to ``qc_flags`` so downstream/you can EXCLUDE flagged images by hand.
    #
    # Two independent triggers (either fires the flag):
    #   1. ABSOLUTE cap: RNA1 spots/nucleus > this many. A "few hundred per
    #      nucleus" is already far above any real single-molecule lncRNA/mRNA
    #      count in these datasets. Set <= 0 to DISABLE the absolute cap.
    qc_overdetect_rna1_max_per_nucleus: float = 300.0
    #   2. ROBUST run-level outlier: RNA1 spots/nucleus exceeds
    #      ``median + k * MAD`` across the run AND is above the small-signal
    #      floor below. Computed once per run (needs the whole batch), so it is
    #      applied by the runner after all images are processed. Set k <= 0 to
    #      DISABLE the robust outlier test.
    qc_overdetect_robust_mad_k: float = 5.0
    # Small-signal floor for the robust test: a run whose RNA1 spots/nucleus are
    # all low won't spuriously flag a merely-2x-median image as "over-detected".
    # The robust outlier only fires when the image is ALSO above this absolute
    # spots/nucleus value.
    qc_overdetect_min_per_nucleus_for_outlier: float = 50.0


class IfIntensityCfg(BaseModel):
    """Immunofluorescence antibody-validation intensity mode (``if_intensity``).

    Drives the plate-level IF pipeline (per-well signal routing, exposure gate,
    per-nucleus + whole-FOV intensity, fold-over-secondary-only normalization,
    per-well biological-replicate stats, SuperPlots, and SHARED-display-range
    micrographs). Ported verbatim from the locked panQKI WT-vs-QKI-KO standalone
    (F:\\Image Analysis Work\\MIAT-QKI-Coloc\\WT-QKI-KO_2026_07_01\\_scripts).
    Segmentation params come from the shared ``nuclei`` block (cellpose/cpsam/
    DirectML), identical to the FISH modes. Only active when
    ``channels.analysis_mode == "if_intensity"``; every other mode ignores it.
    Human / Homo sapiens.
    """
    # ---- plate map --------------------------------------------------------
    # CSV with one row per well: columns well, genotype, staining_arm (or arm),
    # secondary. Genotype/arm/secondary are the biology source of truth; each
    # discovered image is matched to a well by parsing the well number from its
    # containing subfolder name (``well[-_ ]?(\\d+)``). If empty, the mode falls
    # back to parsing genotype/arm/secondary straight from the folder names.
    plate_layout_csv: str = ""
    # Well-number -> secondary override for conflict wells (e.g. folder tag says
    # 565 but the plate groups it as 647). Keys are stringified well numbers.
    well_secondary_overrides: Dict[str, str] = Field(default_factory=dict)
    # Image-file extensions ``_discover_wells`` globs inside each well subfolder,
    # in priority order (case-insensitive, no leading dot; multi-dot like
    # "ome.tif" is allowed). Default ["vsi"] = legacy byte-identical behaviour.
    # Set to e.g. ["ome.tif"] to quantify pre-generated MEAN-z-projection
    # OME-TIFFs (the volumetric if_intensity workflow) instead of raw .vsi.
    input_glob_exts: List[str] = Field(default_factory=lambda: ["vsi"])

    # ---- channel routing (substring-matched on OME channel_names) ---------
    dapi_channel_key: str = "405"
    # secondary label -> the CSU channel-name substring that carries its signal.
    # NOTE the intentional offset: secondary 647 -> "640" channel; 568/565 ->
    # "561" channel.
    signal_channel_map: Dict[str, str] = Field(
        default_factory=lambda: {"647": "640", "568": "561", "565": "561"}
    )

    # ---- metrics ----------------------------------------------------------
    cyto_ring_px: int = 12          # dilated-ring width grown from each nucleus
    exposure_tol_s: float = 1e-6    # intra-channel exposure-equality tolerance
    low_nuc_flag: int = 3           # flag FOVs with fewer than this many nuclei
    pixel_size_um: float = 0.0      # 0 = read from image metadata

    # ---- shared-display micrographs ("Sam's floor") ----------------------
    display_ceiling_pct: float = 99.5   # vmax from the SIGNAL (WT-primary) wells
    display_floor_pct: float = 50.0     # vmin from the secondary-only wells
    scalebar_um: float = 20.0
    # Micrograph source: "fov" projects the quantification FOVs themselves;
    # "zstack" pulls a separate z-stack folder and uses a CENTRAL in-focus
    # windowed max-projection (cleaner than a full-stack max).
    micrograph_source: Literal["fov", "zstack"] = "fov"
    micrograph_zstack_dir: str = ""
    micrograph_z_window_frac: float = 0.5   # central in-focus fraction to keep

    # ---- publication images (native port of the panQKI pub_images.py) -----
    # Renders, per representative well per secondary, from TWO sources:
    #   single_plane (representative FOV of the quantification set) and picked_z
    #   (the SINGLE best-focus z-plane -- var(laplace)*mean on DAPI, central band,
    #   NO MIP). Per well: a signal|DAPI|Merge channel panel + a standalone merge,
    #   plus one WT/KO/secondary-only composite per secondary/source. Uses the
    #   RAISED per-secondary display floors so WT cytoplasm reads near-zero;
    #   ceiling = pub_ceiling_pct of the WT-primary signal. Regenerable from a
    #   finished run WITHOUT the GPU via `fishsuite if-pub-images`.
    pub_images: bool = True            # render publication images (default ON)
    pub_sources: List[str] = Field(default_factory=lambda: ["single_plane", "picked_z"])
    # RAISED per-secondary signal display floor (vmin); drives WT cytoplasm to
    # near-zero. Keyed by secondary label ("647"/"568"). Ported values.
    pub_display_floors: Dict[str, float] = Field(
        default_factory=lambda: {"647": 5000.0, "568": 3500.0}
    )
    pub_ceiling_pct: float = 99.5      # vmax = this pct of the WT-primary signal
    # 2026-07-05 Brian: EXPLICIT signal ceiling (vmax) override, per secondary
    # and/or per (secondary, source). Keys may be bare ``"647"`` (both sources)
    # or source-scoped ``"647:single_plane"`` (that source only; the more-
    # specific key wins). When a secondary/source has an entry here it OVERRIDES
    # ``pub_ceiling_pct``; secondaries/sources with no entry keep the WT-primary
    # percentile behaviour. ``pub_display_floors`` accepts the SAME per-source
    # ``"SEC:source"`` key syntax. Default empty = percentile ceiling for all
    # (byte-identical legacy).
    pub_display_ceilings: Dict[str, float] = Field(default_factory=dict)
    # 2026-07-05 Brian: FIXED DAPI display range (vmin/vmax) applied to the DAPI
    # channel across ALL panels (so DAPI is not over-exposed; Brian's target is a
    # fixed ceiling ~8000). When BOTH are None, per-image DAPI min-max
    # normalization as before (legacy). One-sided is allowed — the unset side
    # falls back to the per-image min / max.
    pub_dapi_floor: Optional[float] = None
    pub_dapi_ceiling: Optional[float] = None
    # 2026-07-05 Brian: compute each panel's signal ceiling from THAT image's own
    # ``pub_ceiling_pct`` percentile instead of the shared WT-primary ceiling.
    # Slightly non-rigorous (breaks cross-panel brightness comparability) — a
    # fallback when a shared ceiling clips or washes out individual panels. An
    # explicit ``pub_display_ceilings`` entry still wins over it. Default False
    # (shared display is the default and the scientifically preferred mode).
    pub_per_image_ceiling: bool = False
    # Separate z-stack folder for the picked_z source (subfolders per well). When
    # empty, falls back to micrograph_zstack_dir.
    pub_zstack_dir: str = ""
    pub_z_central_frac: float = 0.8    # central band the best-focus plane is picked from
    pub_dapi_weight: float = 0.6       # DAPI dimming in the merge (locked)
    pub_signal_label: str = "signal"   # panel/label text (e.g. "QKI")

    # ---- misc -------------------------------------------------------------
    fig_seed: int = 42
    make_excel: bool = True


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
    qc: QcCfg = Field(default_factory=QcCfg)
    if_intensity: IfIntensityCfg = Field(default_factory=IfIntensityCfg)

    # Broad GLOBAL reproducibility seed (ADDITIVE, 2026-06-10). Seeds Python's
    # ``random``, NumPy, PYTHONHASHSEED, and (if present) torch at the very
    # start of each run via ``core.repro.set_global_seeds``. This is the broad
    # run-wide seed; it is COMPLEMENTARY to — and does NOT replace —
    # ``foci.partner_null_seed`` (the focused per-image partner-null
    # permutation seed). Default 0 is a valid, deterministic seed.
    seed: int = 0

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
