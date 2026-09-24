"""Single-plane raw-intensity MIAT/QKI occupancy and exact-placement correction."""
from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json

import numpy as np
import pandas as pd

from . import coloc_pixel_metrics as _cpm
from .footprint_null import (
    MiatFootprint,
    exact_footprint_position_null,
    full_footprint_is_valid,
)


_FRACTIONS = (
    "frac_miat_spots_qki_pos",
    "frac_qki_area_on_miat_area",
    "frac_qki_area_on_miat_footprints",
    "frac_miat_footprint_area_qki_pos",
)
_NULL_METRICS = (_FRACTIONS[0], _FRACTIONS[3])
_IDENTITY = ("threshold_multiplier", "miat_min_used", "qki_min_used",
             "image", "condition", "well", "nucleus_id", "measurement_plane")
_CONTEXT = (
    "nuclear_area_um2", "qki_pos_area_frac", "miat_pos_area_frac",
    "miat_footprint_area_frac", "mean_nuclear_qki", "mean_qki_at_miat_spots",
    "qki_at_spots_minus_nuclear", "integrated_nuclear_miat", "n_miat_spots",
    "n_miat_spots_qki_pos", "n_miat_spots_per_100um2", "median_footprint_area_px",
    "sat_frac_miat", "sat_frac_qki",
    "null_valid_center_count", "n_null_effective",
)
_EXPECTED = ("exp_frac_qki_area_on_miat_area", "exp_frac_qki_area_on_miat_footprints")
_ANALYTIC_CORRECTION = ("obs_minus_exp_frac_qki_area_on_miat_area",)
_CORRECTION = tuple(f"{prefix}_{metric}" for metric in _NULL_METRICS
                    for prefix in ("null_mean", "null_sd", "obs_minus_null", "null_ge_obs_frac"))
_REASONED = (_FRACTIONS + ("qki_pos_area_frac", "miat_pos_area_frac",
                           "miat_footprint_area_frac", "sat_frac_miat", "sat_frac_qki")
             + _EXPECTED + _ANALYTIC_CORRECTION + _CORRECTION)
LEGACY_NUCLEUS_COLUMNS = (_IDENTITY + _CONTEXT + _FRACTIONS + _EXPECTED + _ANALYTIC_CORRECTION + _CORRECTION
                          + tuple(f"na_reason_{name}" for name in _REASONED))
LEGACY_SPOT_COLUMNS = _IDENTITY + ("spot_id", "footprint_area_px", "footprint_mean_qki",
                                   "qki_positive")
# 2026-09-24 threshold-free additions (metric_inventory.md section c, items 1-4).
# Appended AFTER every legacy column so the legacy CSV prefix is unchanged.
_UPP = ("mean_uniform_position_percentile_qki", "frac_spots_upp_ge_0p90", "frac_spots_upp_ge_0p75")
_UPP_REFERENCE = ("n_spots_upp", "chance_mean_upp", "chance_frac_spots_upp_ge_0p90",
                  "chance_frac_spots_upp_ge_0p75")
_PIXEL = ("pearson_r_nucleoplasm", "spearman_rho_nucleoplasm", "n_pixels_nucleoplasm",
          "pearson_r_whole_nucleus_mask")
_CCF = ("ccf_r0",) + tuple(f"{name}_{axis}" for axis in ("x", "y") for name in (
    "ccf_peak_r", "ccf_peak_shift_px", "ccf_peak_shift_um", "ccf_peak_fwhm_px",
    "ccf_r0_minus_flank"))
_COSTES_PARAMS = ("costes_psf_fwhm_px", "costes_acf_fwhm_px_miat", "costes_acf_fwhm_px_qki",
                  "costes_block_px", "costes_tile_phase_y", "costes_tile_phase_x",
                  "costes_rand_n_blocks", "costes_core_coverage", "costes_rand_n_draws")
_COSTES_STATS = ("costes_rand_r_obs", "costes_rand_null_mean_r", "costes_rand_r_obs_minus_null_mean",
                 "costes_rand_p")
_NEW_REASONS = (tuple(f"na_reason_{name}" for name in _UPP)
                + ("na_reason_pearson_r_nucleoplasm", "na_reason_spearman_rho_nucleoplasm",
                   "na_reason_pearson_r_whole_nucleus_mask", "na_reason_ccf_r0")
                + tuple(f"na_reason_{name}_{axis}" for axis in ("x", "y")
                        for name in ("ccf_peak", "ccf_peak_fwhm_px", "ccf_r0_minus_flank"))
                + ("na_reason_costes_rand", "costes_psf_source"))
# Columns that depend on --seed (placement-null draws, UPP tie-breaks, Costes scramble).
SEED_DEPENDENT_NEW = _UPP + ("costes_rand_null_mean_r", "costes_rand_r_obs_minus_null_mean",
                             "costes_rand_p")
NEW_NUCLEUS_COLUMNS = _UPP + _UPP_REFERENCE + _PIXEL + _CCF + _COSTES_PARAMS + _COSTES_STATS + _NEW_REASONS
NUCLEUS_COLUMNS = list(LEGACY_NUCLEUS_COLUMNS + NEW_NUCLEUS_COLUMNS)
SPOT_COLUMNS = list(LEGACY_SPOT_COLUMNS + ("uniform_position_percentile_qki",))
# Signed correlation / shift / difference statistics: a treated/control ratio is undefined.
DIFFERENCE_SCALE_NEW = frozenset(
    ("pearson_r_nucleoplasm", "spearman_rho_nucleoplasm", "pearson_r_whole_nucleus_mask",
     "costes_rand_r_obs", "costes_rand_null_mean_r", "costes_rand_r_obs_minus_null_mean")
    + tuple(c for c in _CCF if not c.startswith("ccf_peak_fwhm_px")))
# Astra F3: parameters, randomization p-values and chance references are
# descriptive QC. They never enter well/arm contrasts, differences or ratios.
DESCRIPTIVE_ONLY = frozenset(_COSTES_PARAMS + ("costes_rand_p",) + _UPP_REFERENCE)

_DEFINITIONS = {
    "threshold_multiplier": "Multiplier applied to both raw thresholds; dimensionless",
    "miat_min_used": "MIAT threshold after multiplier; raw intensity",
    "qki_min_used": "QKI threshold after multiplier; raw intensity",
    "image": "Recorded image identifier; identifier",
    "condition": "Recorded condition; label",
    "well": "Recorded biological well; identifier",
    "nucleus_id": "Positive nuclear mask label, including nuclei without spots; identifier",
    "measurement_plane": "Measurement scope, always single-plane; label",
    "nuclear_area_um2": "Eligible nucleus area excluding masked regions; um2",
    "qki_pos_area_frac": "QKI threshold-positive eligible pixels divided by eligible pixels; fraction",
    "miat_pos_area_frac": "MIAT threshold-positive eligible pixels divided by eligible pixels; fraction",
    "miat_footprint_area_frac": "Union of eligible footprint pixels divided by eligible pixels; fraction",
    "mean_nuclear_qki": "Mean raw QKI over eligible nuclear pixels; raw intensity",
    "mean_qki_at_miat_spots": "Unweighted mean across eligible spots of footprint-mean raw QKI; raw intensity",
    "qki_at_spots_minus_nuclear": "Mean QKI at spots minus eligible nuclear mean QKI; raw intensity",
    "integrated_nuclear_miat": "Sum raw MIAT over eligible nuclear pixels; raw intensity pixel",
    "n_miat_spots": "Number of eligible complete nuclear MIAT footprints; count",
    "n_miat_spots_qki_pos": "Eligible spots whose footprint-mean raw QKI meets threshold; count",
    "n_miat_spots_per_100um2": "Eligible spot count per 100 um2 eligible nuclear area; count per 100 um2",
    "median_footprint_area_px": "Median eligible footprint area; pixels",
    "sat_frac_miat": "Eligible fraction at dtype maximum or observed maximum when shared by >=0.1% of eligible pixels; fraction",
    "sat_frac_qki": "Eligible fraction at dtype maximum or observed maximum when shared by >=0.1% of eligible pixels; fraction",
    "frac_miat_spots_qki_pos": "QKI-positive eligible spot count divided by eligible spot count; fraction",
    "frac_qki_area_on_miat_area": "QKI-positive pixels also MIAT-positive divided by QKI-positive pixels in eligible nucleus; area occupancy fraction",
    "frac_qki_area_on_miat_footprints": "QKI-positive pixels inside eligible footprint union divided by QKI-positive pixels in eligible nucleus; area occupancy fraction; UNCORRECTED for MIAT coverage and rises with coverage by chance; the coverage-corrected area statistic is obs_minus_null_frac_miat_footprint_area_qki_pos",
    "frac_miat_footprint_area_qki_pos": "QKI-positive pixels inside eligible footprint union divided by that union's own pixel count; fraction; each null draw uses its own union denominator",
    "null_valid_center_count": "Minimum number of admissible centers across eligible footprints, zero when no footprints; not a shared-center intersection count; count",
    "n_null_effective": "Common completed placement iterations across all eligible footprints, minimum valid_draw_count; zero with no footprints or any invalid null; sparse-domain draws are counted although statistics are withheld; count",
    "exp_frac_qki_area_on_miat_area": "Analytic independence expectation equal to eligible MIAT-positive area fraction; fraction",
    "exp_frac_qki_area_on_miat_footprints": "Analytic independence expectation equal to eligible footprint union area fraction; fraction",
    "obs_minus_exp_frac_qki_area_on_miat_area": "Observed QKI area occupancy on MIAT-positive area minus analytic independence expectation; fraction",
    "spot_id": "Input footprint spot_index; identifier",
    "footprint_area_px": "Complete eligible footprint pixel count; pixels",
    "footprint_mean_qki": "Mean raw QKI over complete footprint pixels; raw intensity",
    "qki_positive": "Whether footprint-mean raw QKI is >= qki_min_used; boolean",
    "uniform_position_percentile_qki": "Uniform-position percentile score of this spot: rank R of its footprint-mean raw QKI among its OWN K exact-footprint uniform-position placement draws, R = #{draw < observed} + J with J uniform on {0..#{draw = observed}} (randomized tie-breaking; replaces the round-1 mid-rank), divided by K = n_null_effective; threshold-free, invariant to monotone intensity transforms; exactly uniform on {0..K}/K only if MIAT centres are exchangeable with uniform admissible positions (mean 0.5); NaN when the placement null is NO_DOMAIN or SPARSE_DOMAIN; fraction",
    "mean_uniform_position_percentile_qki": "Mean over this nucleus's eligible spots of uniform_position_percentile_qki (each nucleus weighted equally at the well level; the spot-pooled well value is in coupling per_well); chance chance_mean_upp; fraction",
    "frac_spots_upp_ge_0p90": "Fraction of eligible spots with uniform_position_percentile_qki >= 0.9; finite-K chance is chance_frac_spots_upp_ge_0p90 = (K - ceil(0.9K) + 1)/(K + 1) (21/201 at K = 200), not 0.10; fraction",
    "frac_spots_upp_ge_0p75": "Fraction of eligible spots with uniform_position_percentile_qki >= 0.75; finite-K chance chance_frac_spots_upp_ge_0p75 (51/201 at K = 200); fraction",
    "n_spots_upp": "Spots scored by the uniform-position percentile (0 when undefined); weight for the spot-pooled well value; count; descriptive only",
    "chance_mean_upp": "Chance reference for mean_uniform_position_percentile_qki under exchangeability (0.5); descriptive only",
    "chance_frac_spots_upp_ge_0p90": "Finite-K chance reference (K - ceil(0.9K) + 1)/(K + 1) with K = n_null_effective; descriptive only",
    "chance_frac_spots_upp_ge_0p75": "Finite-K chance reference (K - ceil(0.75K) + 1)/(K + 1); descriptive only",
    "pearson_r_nucleoplasm": "Pearson r of raw MIAT vs raw QKI over the nucleoplasm mask N (eligible pixels of this nucleus, nucleoli excluded); offset- and gain-invariant; NaN below 100 pixels (LOW_PIX) or for a constant channel (ZERO_VAR); r",
    "spearman_rho_nucleoplasm": "Spearman rho (Pearson on average ranks) of raw MIAT vs raw QKI over the nucleoplasm mask N; same gates as pearson_r_nucleoplasm; rho",
    "n_pixels_nucleoplasm": "Pixel count of the nucleoplasm mask N used by the pixel metrics; pixels",
    "pearson_r_whole_nucleus_mask": "Comparator: Pearson r over the WHOLE nuclear label mask without nucleolar exclusion (the convention of the panel's pearson_r_csp); pearson_r_whole_nucleus_mask minus pearson_r_nucleoplasm is the nucleolar-exclusion contribution; r",
    "ccf_r0": "Van Steensel CCF at zero shift: Pearson r of M(y,x) vs Q(y,x) over pixel pairs inside N (equals pearson_r_nucleoplasm); r",
    "costes_psf_fwhm_px": "Theoretical lateral PSF FWHM 0.51 * lambda_em / NA in pixels, lambda_em = the longer of the MIAT and QKI emission wavelengths; source in costes_psf_source and command.log; descriptive only; pixels",
    "costes_psf_source": "Where objective NA and emission wavelengths came from (e.g. source VSI OME metadata path or explicit CLI values); text",
    "costes_acf_fwhm_px_miat": "FWHM of the radially averaged, overlap-normalized, mean-centered 2-D autocorrelation of raw MIAT over in-mask pixel pairs of N (2 x radius where ACF first < 0.5, <= 20 px); recorded, not used for the block (MIAT is not scrambled); descriptive only; pixels",
    "costes_acf_fwhm_px_qki": "Same ACF FWHM for raw QKI; descriptive only; pixels",
    "costes_block_px": "Costes block edge b = ceil(max(costes_psf_fwhm_px, costes_acf_fwhm_px_qki)): the ACF width of the SCRAMBLED channel (QKI; MIAT stays fixed). Round 3 change from round 2's min(MIAT, QKI) width, which gave blocks smaller than the QKI texture and an anticonservative null; descriptive only; pixels",
    "costes_tile_phase_y": "Grid phase (rows) maximising complete-block coverage of N, chosen from mask geometry only; descriptive only; pixels",
    "costes_tile_phase_x": "Grid phase (columns) maximising complete-block coverage of N; descriptive only; pixels",
    "costes_rand_n_blocks": "Complete b x b blocks in the frozen core N_core; descriptive only; count",
    "costes_core_coverage": "Pixels of N_core divided by pixels of N; Costes is NA (MASK_BLOCK_COVERAGE) below 0.80 or below 10 blocks; descriptive only; fraction",
    "costes_rand_n_draws": "Block-scramble permutations per nucleus (--n-costes); descriptive only; count",
    "costes_rand_r_obs": "Pearson r of raw MIAT vs raw QKI over exactly N_core (unpermuted); r",
    "costes_rand_null_mean_r": "Mean Pearson r over the block-scramble draws on N_core (QKI blocks permuted among N_core block positions, orientation kept, MIAT fixed); r",
    "costes_rand_r_obs_minus_null_mean": "costes_rand_r_obs minus costes_rand_null_mean_r; r",
    "costes_rand_p": "NOT_CALIBRATED (round-3 gate: 8.25% false positives at alpha 0.05 and 2.75% at 0.01 on 400 texture-matched independent-channel nuclei, above the binomial limits 6.75% / 2.0%): QC descriptor only, never evidence of colocalization. Costes randomization tail fraction (1 + #{r_perm >= r_obs}) / (1 + costes_rand_n_draws), one-sided, on N_core; seed recorded in command.log; per-nucleus descriptive; not a test across nuclei and never read as nucleus-level inference; excluded from coupling contrasts; like the placement null it does NOT remove MIAT/QKI co-preference for the same sub-nuclear compartment; fraction",
    "na_reason_costes_rand": "Undefined reason for every costes_rand_* value: empty when defined, R0, NO_PSF (optics unavailable to the library call), ACF_NO_HALF / ACF_LOW_PIX / ACF_ZERO_VAR (a channel ACF has no half-height crossing within 20 px or is not estimable), MASK_BLOCK_COVERAGE (<10 complete blocks or <80% of N in complete blocks), ZERO_VAR; code",
    "na_reason_ccf_r0": "Undefined reason for ccf_r0: empty when defined, LOW_PIX (<100 pixel pairs) or NO_CCF; code",
}
for _axis, _long in (("x", "x (columns)"), ("y", "y (rows)")):
    _DEFINITIONS.update({
        f"ccf_peak_r_{_axis}": f"Maximum of the Van Steensel CCF r(d), d = -20..+20 px step 1 px along {_long}, r(d) = Pearson of M(p) vs Q(p + d) over pairs with both pixels in N (>=100 pairs); r",
        f"ccf_peak_shift_px_{_axis}": f"Shift d of the CCF maximum along {_long} (ties to the smallest |d|); positive = QKI displaced +d px from MIAT; a non-zero value in most nuclei indicates channel misregistration; pixels",
        f"ccf_peak_shift_um_{_axis}": f"ccf_peak_shift_px_{_axis} times the pixel size read from the image metadata (manifest voxel_xy_nm), never a default; um",
        f"ccf_peak_fwhm_px_{_axis}": f"Full width of the CCF peak along {_long} at flank + (peak - flank)/2, linear interpolation; flank = mean r over 15 <= |d| <= 20 px; ~PSF width = punctum-scale association, broad = compartment sharing; pixels",
        f"ccf_r0_minus_flank_{_axis}": f"ccf_r0 minus the mean CCF r over 15 <= |d| <= 20 px along {_long}; punctum-scale excess over compartment-scale correlation; r",
        f"na_reason_ccf_peak_{_axis}": f"Undefined reason for ccf_peak_r_{_axis} / ccf_peak_shift_*_{_axis}: empty when defined, NO_CCF (no shift with >=100 pairs); code",
        f"na_reason_ccf_peak_fwhm_px_{_axis}": f"Undefined reason for ccf_peak_fwhm_px_{_axis}: empty when defined, NO_CCF, NO_PEAK_ABOVE_FLANK, NO_HALF_CROSSING (curve never falls to half height inside +-20 px); code",
        f"na_reason_ccf_r0_minus_flank_{_axis}": f"Undefined reason for ccf_r0_minus_flank_{_axis}: empty when defined, NO_CCF, NO_FLANK; code",
    })
for _metric in _UPP:
    _DEFINITIONS[f"na_reason_{_metric}"] = f"Undefined reason for {_metric}: empty when defined, N0 (no spots), NO_DOMAIN or SPARSE_DOMAIN (placement null withheld); code"
for _metric in ("pearson_r_nucleoplasm", "spearman_rho_nucleoplasm", "pearson_r_whole_nucleus_mask"):
    _DEFINITIONS[f"na_reason_{_metric}"] = f"Undefined reason for {_metric}: empty when defined, R0, LOW_PIX (<100 pixels), ZERO_VAR; code"
for _metric in _NULL_METRICS:
    for _prefix, _description in (
        ("null_mean", "Mean identical statistic over independent exact-footprint placements"),
        ("null_sd", "Population standard deviation (ddof=0) of identical placement-null statistic"),
        ("obs_minus_null", "Observed statistic minus placement-null mean"),
        ("null_ge_obs_frac", "(1 + number of completed null draws >= observed)/(1 + n_null_effective); per-nucleus descriptive; not a test across nuclei"),
    ):
        _DEFINITIONS[f"{_prefix}_{_metric}"] = f"{_description} for {_metric}; fraction"
for _metric in _REASONED:
    _DEFINITIONS[f"na_reason_{_metric}"] = (
        f"Undefined reason for {_metric}: empty when defined, N0 (no spots), Q0 (no QKI-positive pixels), "
        "U0 (empty footprint union), R0 (no eligible pixels), NO_DOMAIN (zero admissible centers or unusable exact placements), "
        "or SPARSE_DOMAIN (minimum admissible-center count below 50); code"
    )
COLUMN_DEFINITIONS = {name: f"{definition}; single-plane" for name, definition in _DEFINITIONS.items()}


def _positive(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _saturation(values: np.ndarray) -> float:
    if not values.size:
        return float("nan")
    maximum = values.max()
    at_maximum = float(np.count_nonzero(values == maximum) / values.size)
    if np.issubdtype(values.dtype, np.integer):
        dtype_maximum = np.iinfo(values.dtype).max
    elif np.issubdtype(values.dtype, np.floating):
        dtype_maximum = np.finfo(values.dtype).max
    else:
        dtype_maximum = 1
    return at_maximum if maximum == dtype_maximum or at_maximum >= 0.001 else 0.0


def _placement_rng(seed: int, image: str, nucleus_id: int) -> np.random.Generator:
    identity = json.dumps([int(seed), str(image), int(nucleus_id)],
                          ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    words = np.frombuffer(hashlib.sha256(identity).digest(), dtype="<u4")
    return np.random.default_rng(np.random.SeedSequence(list(map(int, words))))


def _costes_rng(seed: int, image: str, nucleus_id: int) -> np.random.Generator:
    # Same (seed, image, nucleus_id) binding as _placement_rng, in its own
    # stream, so the block scramble never shifts a placement-null draw.
    identity = json.dumps([int(seed), str(image), int(nucleus_id), "costes_block_scramble"],
                          ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    words = np.frombuffer(hashlib.sha256(identity).digest(), dtype="<u4")
    return np.random.default_rng(np.random.SeedSequence(list(map(int, words))))


def _stream_rng(seed: int, image: str, nucleus_id: int, tag: str) -> np.random.Generator:
    identity = json.dumps([int(seed), str(image), int(nucleus_id), tag],
                          ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    words = np.frombuffer(hashlib.sha256(identity).digest(), dtype="<u4")
    return np.random.default_rng(np.random.SeedSequence(list(map(int, words))))


def _pixel_and_costes_columns(miat, qki, labels, region, nucleus_id, *, pixel_size_um,
                              psf_fwhm_px, psf_source, n_costes, rng, curves):
    corr = _cpm.masked_correlations(miat, qki, region)
    whole = _cpm.masked_correlations(miat, qki, labels == nucleus_id)
    row = dict(pearson_r_nucleoplasm=corr["pearson"], spearman_rho_nucleoplasm=corr["spearman"],
               n_pixels_nucleoplasm=corr["n_pixels"],
               pearson_r_whole_nucleus_mask=whole["pearson"],
               na_reason_pearson_r_nucleoplasm=corr["reason"],
               na_reason_spearman_rho_nucleoplasm=corr["reason"],
               na_reason_pearson_r_whole_nucleus_mask=whole["reason"])
    row.update(_cpm.nucleus_ccf(miat, qki, region, pixel_size_um=pixel_size_um, curves=curves))
    row.update(_cpm.nucleus_costes(miat, qki, region, psf_px=psf_fwhm_px, n_iter=n_costes, rng=rng))
    row["costes_psf_source"] = str(psf_source or "")
    return row


def association_tables(
    miat: np.ndarray,
    qki: np.ndarray,
    labels: np.ndarray,
    footprints: Sequence[MiatFootprint],
    *,
    pixel_size_um: float,
    miat_min: float,
    qki_min: float,
    sensitivity: Sequence[float] = (0.8, 1.0, 1.25),
    n_null: int = 200,
    seed: int = 0,
    eligible_mask: np.ndarray | None = None,
    image: str = "",
    condition: str = "",
    well: str = "",
    n_costes: int = _cpm.COSTES_N_ITER,
    ccf_records: list | None = None,
    psf_fwhm_px: float | None = None,
    psf_source: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute all requested levels using the same retained exact placements.

    The caller supplies postrun-eligible footprints and the matching exclusion
    mask. Complete-mask validity is checked again per nucleus. Thresholds do
    not filter the footprint population; the MIAT threshold defines area only.
    Each null draw translates every footprint independently, allowing overlap,
    then takes the union to avoid double counting occupied pixels. The area
    null divides overlap by that draw's own union size. Independent SHA256 /
    SeedSequence streams bind the global seed to image and nucleus ID; sorted
    spot IDs determine the order within each independent nucleus stream.

    The threshold-free additions (placement-null rank, nucleoplasm Pearson /
    Spearman, CCF, Costes randomization) are computed once per nucleus and
    repeated on every threshold row. The Costes scramble and the percentile
    tie-breaks use their own seeded streams. ``ccf_records``, when a list,
    receives the per-nucleus CCF curves. ``psf_fwhm_px`` (from acquisition
    metadata) is required for Costes; without it Costes is NA (NO_PSF).
    """
    miat_min = _positive(miat_min, "miat_min")
    qki_min = _positive(qki_min, "qki_min")
    pixel_size_um = _positive(pixel_size_um, "pixel_size_um")
    levels = tuple(_positive(level, "sensitivity multiplier") for level in sensitivity)
    if not levels or len(set(levels)) != len(levels):
        raise ValueError("sensitivity must contain distinct positive multipliers")
    if not isinstance(n_null, (int, np.integer)) or n_null <= 0:
        raise ValueError("n_null must be a positive integer")
    for level in levels:
        _positive(miat_min * level, "scaled miat_min")
        _positive(qki_min * level, "scaled qki_min")
    miat, qki, labels = np.asarray(miat), np.asarray(qki), np.asarray(labels)
    if miat.ndim != 2 or qki.shape != miat.shape or labels.shape != miat.shape:
        raise ValueError("miat, qki and labels must be matching single-plane 2D arrays")
    if not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
        raise ValueError("labels must contain nonnegative integer labels")
    eligible = np.ones(labels.shape, dtype=bool) if eligible_mask is None else np.asarray(eligible_mask, dtype=bool)
    if eligible.shape != labels.shape:
        raise ValueError("eligible_mask must match the single-plane image shape")
    included = eligible & (labels > 0)
    if not np.isfinite(miat[included]).all() or not np.isfinite(qki[included]).all():
        raise ValueError("eligible nuclear raw intensities must all be finite")
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(n_costes, bool) or not isinstance(n_costes, (int, np.integer)) or n_costes <= 0:
        raise ValueError("n_costes must be a positive integer")
    nucleus_rows, spot_rows = [], []
    height, width = labels.shape
    for nucleus_id in np.unique(labels[labels > 0]):
        region = eligible & (labels == nucleus_id)
        region_size = int(region.sum())
        selected = [fp for fp in footprints
                    if 0 <= fp.center_y_px < height and 0 <= fp.center_x_px < width
                    and labels[fp.center_y_px, fp.center_x_px] == nucleus_id
                    and fp.full_mask_valid and not fp.invalid_reason
                    and full_footprint_is_valid(fp, region)]
        selected.sort(key=lambda fp: fp.spot_index)
        observed_means = np.asarray([qki[fp.y_px, fp.x_px].astype(float).mean() for fp in selected])
        union = np.zeros(labels.shape, dtype=bool)
        for fp in selected:
            union[fp.y_px, fp.x_px] = True
        rng = _placement_rng(seed, image, nucleus_id)
        nulls = [exact_footprint_position_null(qki, fp, region, n_null=n_null,
                    rng=rng,
                    retain_raw_draws=True) for fp in selected]
        usable = all(result.usable for result in nulls)
        support = min((result.valid_center_count for result in nulls), default=0)
        n_effective = min((result.valid_draw_count for result in nulls), default=0) if usable else 0
        domain_reason = "NO_DOMAIN" if not usable or support == 0 else "SPARSE_DOMAIN" if support < 50 else ""
        null_means = (np.stack([result.null_qki_raw for result in nulls], axis=1)
                      if selected and usable else np.empty((n_null, 0)))
        null_unions = []
        if usable and selected:
            translated = np.concatenate([
                (result.sampled_centers_yx[:, 0, None] + fp.dy_px) * width
                + result.sampled_centers_yx[:, 1, None] + fp.dx_px
                for fp, result in zip(selected, nulls)
            ], axis=1)
            null_unions = [np.unique(draw) for draw in translated]
        n_spots = len(selected)
        r_reason = "" if region_size else "R0"
        rank_reason = "N0" if not n_spots else domain_reason
        percentiles = (_cpm.uniform_position_percentile(
            observed_means, null_means, _stream_rng(seed, image, nucleus_id, "upp_tiebreak"))
            if not rank_reason else np.full(n_spots, np.nan))
        extra = _cpm.upp_summary(percentiles if not rank_reason else np.empty(0), n_effective)
        extra.update({f"na_reason_{name}": rank_reason for name in _UPP})
        curves = [] if ccf_records is not None else None
        extra.update(_pixel_and_costes_columns(
            miat, qki, labels, region, nucleus_id, pixel_size_um=pixel_size_um,
            psf_fwhm_px=psf_fwhm_px, psf_source=psf_source, n_costes=int(n_costes),
            rng=_costes_rng(seed, image, nucleus_id), curves=curves))
        if curves is not None:
            ccf_records.extend(dict(image=image, condition=condition, well=well,
                                    nucleus_id=int(nucleus_id), **c) for c in curves)
        area = float(region_size * pixel_size_um ** 2)
        mean_nuclear = float(qki[region].astype(float).mean()) if region_size else float("nan")
        mean_spots = float(observed_means.mean()) if n_spots else float("nan")
        for level in levels:
            identity = dict(threshold_multiplier=level, miat_min_used=miat_min * level,
                            qki_min_used=qki_min * level, image=image, condition=condition,
                            well=well, nucleus_id=int(nucleus_id), measurement_plane="single-plane")
            q_pos = region & (qki >= qki_min * level)
            m_pos = region & (miat >= miat_min * level)
            q_count = int(q_pos.sum())
            n_positive = int(np.count_nonzero(observed_means >= qki_min * level))
            q_reason = "" if q_count else "Q0"
            n_reason = "" if n_spots else "N0"
            union_count = int(union.sum())
            u_reason = "" if union_count else "U0"
            row = dict(identity, nuclear_area_um2=area,
                       qki_pos_area_frac=q_count / region_size if region_size else float("nan"),
                       miat_pos_area_frac=float(m_pos.sum() / region_size) if region_size else float("nan"),
                       miat_footprint_area_frac=float(union.sum() / region_size) if region_size else float("nan"),
                       mean_nuclear_qki=mean_nuclear, mean_qki_at_miat_spots=mean_spots,
                       qki_at_spots_minus_nuclear=mean_spots - mean_nuclear,
                       integrated_nuclear_miat=float(miat[region].astype(float).sum()),
                       n_miat_spots=n_spots, n_miat_spots_qki_pos=n_positive,
                       n_miat_spots_per_100um2=n_spots * 100 / area if area else float("nan"),
                       median_footprint_area_px=float(np.median([fp.area_px for fp in selected])) if n_spots else float("nan"),
                       sat_frac_miat=_saturation(miat[region]), sat_frac_qki=_saturation(qki[region]),
                       null_valid_center_count=support, n_null_effective=n_effective)
            row[_FRACTIONS[0]] = n_positive / n_spots if n_spots else float("nan")
            row[_FRACTIONS[1]] = float(np.count_nonzero(q_pos & m_pos) / q_count) if q_count else float("nan")
            row[_FRACTIONS[2]] = float(np.count_nonzero(q_pos & union) / q_count) if q_count else float("nan")
            row[_FRACTIONS[3]] = float(np.count_nonzero(q_pos & union) / union_count) if union_count else float("nan")
            row[_EXPECTED[0]], row[_EXPECTED[1]] = row["miat_pos_area_frac"], row["miat_footprint_area_frac"]
            row[_ANALYTIC_CORRECTION[0]] = row[_FRACTIONS[1]] - row[_EXPECTED[0]]
            for metric in _REASONED:
                row[f"na_reason_{metric}"] = ""
            for metric in ("qki_pos_area_frac", "miat_pos_area_frac", "miat_footprint_area_frac",
                           "sat_frac_miat", "sat_frac_qki") + _EXPECTED:
                row[f"na_reason_{metric}"] = r_reason
            for metric, reason in zip(_FRACTIONS, (n_reason, q_reason, q_reason, u_reason)):
                row[f"na_reason_{metric}"] = reason
            row[f"na_reason_{_ANALYTIC_CORRECTION[0]}"] = q_reason
            for metric in _NULL_METRICS:
                reason = n_reason if metric == _FRACTIONS[0] else u_reason
                if not reason:
                    reason = domain_reason
                if reason:
                    draws = None
                elif metric == _FRACTIONS[0]:
                    draws = np.mean(null_means >= qki_min * level, axis=1)
                else:
                    q_flat = q_pos.ravel()
                    draws = np.asarray([np.count_nonzero(q_flat[indices]) / indices.size for indices in null_unions])
                stats = ((float(draws.mean()), float(draws.std(ddof=0)),
                          float(row[metric] - draws.mean()),
                          float((1 + np.count_nonzero(draws >= row[metric])) / (1 + n_effective)))
                         if draws is not None else (float("nan"),) * 4)
                for prefix, value in zip(("null_mean", "null_sd", "obs_minus_null", "null_ge_obs_frac"), stats):
                    key = f"{prefix}_{metric}"
                    row[key], row[f"na_reason_{key}"] = value, reason
            row.update(extra)
            nucleus_rows.append(row)
            for fp, mean, percentile in zip(selected, observed_means, percentiles):
                spot_rows.append(dict(identity, spot_id=fp.spot_index, footprint_area_px=fp.area_px,
                                      footprint_mean_qki=float(mean), qki_positive=bool(mean >= qki_min * level),
                                      uniform_position_percentile_qki=float(percentile)))
    nuclei = pd.DataFrame(nucleus_rows, columns=NUCLEUS_COLUMNS)
    spots = pd.DataFrame(spot_rows, columns=SPOT_COLUMNS)
    # Preserve caller level order, with nuclei and spots deterministic within it.
    if not nuclei.empty:
        nuclei = pd.concat([nuclei[nuclei.threshold_multiplier == level] for level in levels], ignore_index=True)
    if not spots.empty:
        spots = pd.concat([spots[spots.threshold_multiplier == level] for level in levels], ignore_index=True)
    return nuclei, spots
