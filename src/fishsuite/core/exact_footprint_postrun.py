"""Post-run statistics for exact-footprint MIAT/QKI single-plane analysis.

The module consumes only canonical retained tables.  It never reads microscopy
planes, redetects spots, changes a selected z plane, or reruns a null model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from itertools import combinations, product
from math import exp, log
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats

from fishsuite.core.authoritative_sampling import correct_spot_sampling_flags


THRESHOLDS = (90, 95, 99)
_POPULATIONS = ("positive", "negative", "unusable")
UNSIGNED_POSITIVE = "unsigned_positive"
SIGNED_ADDITIVE = "signed_additive"
_VALID_ENDPOINT_SCALES = {UNSIGNED_POSITIVE, SIGNED_ADDITIVE}
EXPECTED_CHANNEL_INDICES = {"miat": 0, "qki": 1, "dapi": 2}
ALLOWED_QUANTITATION_PLANES = ("exact_recorded_single_z",)
DEFAULT_RATIO_OF_RATIOS_SPECS = (
    (
        "threshold_positive_spots_per_nucleus_q95",
        "n_spots_floor",
    ),
    (
        "miat_footprint_mass_q95_positive_spot_summed",
        "miat_footprint_mass_floor_spot_summed",
    ),
)


class PostrunValidationError(ValueError):
    """Raised when retained tables cannot support the locked post-run design."""


@dataclass(frozen=True)
class PostrunResults:
    """In-memory canonical products of the exact-footprint post-run layer."""

    corrected_spots: pd.DataFrame
    sampling_audit: dict[str, Any]
    quantitation_audit: dict[str, Any]
    ratio_specs: tuple[tuple[str, str], ...]
    nucleus_endpoints: pd.DataFrame
    endpoint_fov_means: pd.DataFrame
    endpoint_set_means: pd.DataFrame
    endpoint_inference: pd.DataFrame
    cartesian_contrasts: pd.DataFrame
    fov_correlations: pd.DataFrame
    set_correlations: pd.DataFrame
    correlation_inference: pd.DataFrame
    ratio_of_ratios: pd.DataFrame


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], *, table: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise PostrunValidationError(f"{table} is missing required columns: {missing}")


def _normalised_image_key(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.casefold()


def _bool_series(values: pd.Series, *, name: str) -> pd.Series:
    true = {"true", "t", "yes", "y", "1"}
    false = {"false", "f", "no", "n", "0"}

    def convert(value: Any) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)) and value in (0, 1):
            return bool(value)
        if isinstance(value, (float, np.floating)) and np.isfinite(value) and value in (0, 1):
            return bool(int(value))
        token = str(value).strip().casefold()
        if token in true:
            return True
        if token in false:
            return False
        raise PostrunValidationError(f"{name} is not an explicit boolean: {value!r}")

    return values.map(convert).astype(bool)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def _integer_series(values: pd.Series, *, name: str, minimum: int) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or (~np.isfinite(numeric.to_numpy(float))).any():
        raise PostrunValidationError(f"{name} contains missing or nonfinite values")
    if not np.equal(numeric.to_numpy(float), np.rint(numeric.to_numpy(float))).all():
        raise PostrunValidationError(f"{name} must contain integer values")
    if (numeric < minimum).any():
        raise PostrunValidationError(f"{name} must be at least {minimum}")
    return numeric.astype(np.int64)


def validate_quantitation_invariants(
    spots: pd.DataFrame,
    nuclei: pd.DataFrame,
    *,
    expected_channel_indices: dict[str, int] | None = None,
    allowed_quantitation_planes: Iterable[str] = ALLOWED_QUANTITATION_PLANES,
) -> dict[str, Any]:
    """Fail closed unless retained tables prove exact, locked single-z quantitation."""

    spot_required = {
        "image_key",
        "nucleus_id",
        "selected_z_1based",
        "selected_z_0based",
        "quantitation_plane",
        "miat_channel_index",
        "qki_channel_index",
        "dapi_channel_index",
    }
    nucleus_required = {
        "image_key",
        "nucleus_id",
        "z_mode",
        "z_range",
        "n_z_slices",
    }
    _require_columns(spots, spot_required, table="canonical spot table")
    _require_columns(nuclei, nucleus_required, table="canonical nucleus table")
    if spots.empty:
        raise PostrunValidationError(
            "canonical spot table is empty; selected-z and channel locks cannot be verified"
        )
    if nuclei.empty:
        raise PostrunValidationError(
            "canonical nucleus table is empty; singleton quantitation z cannot be verified"
        )

    expected = dict(EXPECTED_CHANNEL_INDICES)
    if expected_channel_indices is not None:
        expected = {str(key).casefold(): int(value) for key, value in expected_channel_indices.items()}
    if set(expected) != {"miat", "qki", "dapi"}:
        raise PostrunValidationError(
            "expected_channel_indices must define miat, qki, and dapi"
        )
    allowed_planes = {
        str(value).strip().casefold() for value in allowed_quantitation_planes
    }
    if not allowed_planes:
        raise PostrunValidationError("at least one single-plane quantitation value is required")

    spot_frame = spots.copy()
    nucleus_frame = nuclei.copy()
    for frame, table in ((spot_frame, "canonical spot table"), (nucleus_frame, "canonical nucleus table")):
        frame["_image_key_norm"] = _normalised_image_key(frame["image_key"])
        if frame["_image_key_norm"].eq("").any():
            raise PostrunValidationError(f"{table} contains an empty image_key")
        frame["_nucleus_id_norm"] = _integer_series(
            frame["nucleus_id"], name=f"{table} nucleus_id", minimum=1
        )
    if nucleus_frame.duplicated(["_image_key_norm", "_nucleus_id_norm"]).any():
        raise PostrunValidationError("canonical nucleus table contains duplicate nucleus keys")

    spot_frame["_selected_z_1based"] = _integer_series(
        spot_frame["selected_z_1based"], name="selected_z_1based", minimum=1
    )
    spot_frame["_selected_z_0based"] = _integer_series(
        spot_frame["selected_z_0based"], name="selected_z_0based", minimum=0
    )
    if not np.array_equal(
        spot_frame["_selected_z_0based"].to_numpy(),
        spot_frame["_selected_z_1based"].to_numpy() - 1,
    ):
        raise PostrunValidationError(
            "selected_z_0based must equal selected_z_1based minus one for every spot"
        )
    if spot_frame.groupby("_image_key_norm")["_selected_z_1based"].nunique().gt(1).any():
        raise PostrunValidationError("each image must have a single selected_z_1based")
    if (
        spot_frame.groupby(["_image_key_norm", "_nucleus_id_norm"])[
            "_selected_z_1based"
        ]
        .nunique()
        .gt(1)
        .any()
    ):
        raise PostrunValidationError("each image/nucleus must have a single selected_z")

    plane_values = spot_frame["quantitation_plane"].astype(str).str.strip().str.casefold()
    if plane_values.eq("").any() or not set(plane_values).issubset(allowed_planes):
        found = sorted(set(plane_values))
        raise PostrunValidationError(
            "canonical spots do not prove allowed single-plane quantitation; "
            f"found quantitation_plane values {found}"
        )
    projection_tokens = r"mip|projection|maximum.intensity|z.?stack|three.?dimensional|3d"
    if plane_values.str.contains(projection_tokens, regex=True, case=False).any():
        raise PostrunValidationError(
            "quantitation_plane contradicts single-plane quantitation"
        )

    for role in ("miat", "qki", "dapi"):
        column = f"{role}_channel_index"
        spot_frame[f"_{column}"] = _integer_series(
            spot_frame[column], name=column, minimum=0
        )
        if spot_frame.groupby("_image_key_norm")[f"_{column}"].nunique().gt(1).any():
            raise PostrunValidationError(
                f"{column} must be locked to one value within every image"
            )
        if not spot_frame[f"_{column}"].eq(expected[role]).all():
            observed = sorted(spot_frame[f"_{column}"].unique().tolist())
            raise PostrunValidationError(
                "canonical spots do not match expected channel indices: "
                f"{role} expected {expected[role]}, observed {observed}"
            )

    ranges = nucleus_frame["z_range"].astype(str).str.strip().str.extract(
        r"^(\d+)\s*-\s*(\d+)$"
    )
    if ranges.isna().any().any():
        raise PostrunValidationError(
            "canonical nucleus z_range must be an explicit integer singleton such as 21-21"
        )
    range_start = ranges[0].astype(np.int64)
    range_end = ranges[1].astype(np.int64)
    if not range_start.eq(range_end).all():
        raise PostrunValidationError(
            "canonical nucleus quantitation requires a singleton z_range for every nucleus"
        )
    if (range_start < 1).any():
        raise PostrunValidationError("canonical nucleus z_range is 1-based and must be positive")
    nucleus_frame["_selected_z_1based"] = range_start
    nucleus_frame["_n_z_slices"] = _integer_series(
        nucleus_frame["n_z_slices"], name="n_z_slices", minimum=1
    )
    if (nucleus_frame["_selected_z_1based"] > nucleus_frame["_n_z_slices"]).any():
        raise PostrunValidationError("nucleus singleton z_range exceeds n_z_slices")
    z_modes = nucleus_frame["z_mode"].astype(str).str.strip().str.casefold()
    if z_modes.eq("").any():
        raise PostrunValidationError("canonical nucleus z_mode is missing")
    if z_modes.str.contains(projection_tokens, regex=True, case=False).any():
        raise PostrunValidationError(
            "canonical nucleus z_mode contradicts single-plane quantitation"
        )
    per_image_nucleus = nucleus_frame.groupby("_image_key_norm", sort=False).agg(
        selected_z_1based=("_selected_z_1based", "nunique"),
        n_z_slices=("_n_z_slices", "nunique"),
        z_mode=("z_mode", "nunique"),
    )
    if per_image_nucleus.gt(1).any().any():
        raise PostrunValidationError(
            "nucleus z provenance must be locked within every image"
        )

    spot_image = spot_frame.groupby("_image_key_norm", sort=False).agg(
        selected_z_1based=("_selected_z_1based", "first")
    )
    nucleus_image = nucleus_frame.groupby("_image_key_norm", sort=False).agg(
        selected_z_1based=("_selected_z_1based", "first")
    )
    unknown_spot_images = sorted(set(spot_image.index).difference(nucleus_image.index))
    if unknown_spot_images:
        raise PostrunValidationError(
            f"spot images are absent from the nucleus table: {unknown_spot_images}"
        )
    joined = spot_image.join(
        nucleus_image,
        how="left",
        lsuffix="_spot",
        rsuffix="_nucleus",
    )
    if not joined["selected_z_1based_spot"].eq(
        joined["selected_z_1based_nucleus"]
    ).all():
        bad = joined.loc[
            ~joined["selected_z_1based_spot"].eq(
                joined["selected_z_1based_nucleus"]
            )
        ].index.tolist()
        raise PostrunValidationError(
            "spot selected_z disagrees with nucleus z_range for images: "
            f"{bad[:10]}"
        )

    nucleus_images = set(nucleus_image.index)
    spot_images = set(spot_image.index)
    return {
        "validation_status": "pass",
        "evidence_source": "canonical_spot_and_nucleus_tables",
        "n_spots": int(len(spot_frame)),
        "n_nuclei": int(len(nucleus_frame)),
        "n_nucleus_images": int(len(nucleus_images)),
        "n_spot_bearing_images": int(len(spot_images)),
        "n_images_without_spots": int(len(nucleus_images.difference(spot_images))),
        "selected_z_parity_pass": True,
        "single_selected_z_per_image_pass": True,
        "single_selected_z_per_nucleus_pass": True,
        "nucleus_singleton_z_range_pass": True,
        "spot_nucleus_z_agreement_pass": True,
        "channel_lock_within_image_pass": True,
        "expected_channel_indices_pass": True,
        "expected_channel_indices": expected,
        "quantitation_plane_values": sorted(set(plane_values)),
        "nucleus_z_mode_values": sorted(set(z_modes)),
        "projection_or_mip_contradiction": False,
        "zero_spot_image_channel_evidence": (
            "not_applicable_in_spot_table; z proven by singleton nucleus z_range"
            if nucleus_images.difference(spot_images)
            else "all nucleus images have spot-table channel evidence"
        ),
    }


def _population_mask(group: pd.DataFrame, percentile: int, population: str) -> pd.Series:
    expected = f"threshold_{population}" if population != "unusable" else "unusable"
    return group[f"population_label_q{percentile}"].astype(str).eq(expected)


def _spot_sum(group: pd.DataFrame, mask: pd.Series, channel: str) -> float:
    column = f"{channel}_footprint_sum_raw"
    values = pd.to_numeric(group.loc[mask, column], errors="coerce")
    if values.isna().any():
        raise PostrunValidationError(f"{column} contains missing/non-numeric values")
    return float(values.sum())


def _union_sum(
    group: pd.DataFrame,
    mask: pd.Series,
    pixels_by_spot: pd.DataFrame,
    channel: str,
) -> float:
    uids = set(group.loc[mask, "spot_uid"].astype(str))
    if not uids:
        return 0.0
    selected = pixels_by_spot.loc[pixels_by_spot["spot_uid"].astype(str).isin(uids)].copy()
    if selected.empty:
        raise PostrunValidationError("footprint pixels are missing for selected spot_uid values")
    raw_column = f"{channel}_raw"
    selected[raw_column] = pd.to_numeric(selected[raw_column], errors="coerce")
    if selected[raw_column].isna().any():
        raise PostrunValidationError(f"footprint pixels {raw_column} contains non-numeric values")
    conflicts = selected.groupby("flat_pixel_index", sort=False)[raw_column].nunique(dropna=False)
    if conflicts.gt(1).any():
        raise PostrunValidationError(
            f"overlapping footprints disagree on {raw_column} for a shared flat_pixel_index"
        )
    return float(selected.drop_duplicates("flat_pixel_index")[raw_column].sum())


def _mass_metrics(
    group: pd.DataFrame,
    pixels: pd.DataFrame,
    populations: dict[str, pd.Series],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for population, mask in populations.items():
        for channel in ("miat", "qki"):
            stem = f"{channel}_footprint_mass_{population}"
            metrics[f"{stem}_spot_summed"] = _spot_sum(group, mask, channel)
            metrics[f"{stem}_union_deduplicated"] = _union_sum(
                group, mask, pixels, channel
            )
    return metrics


def build_nucleus_endpoints(
    nuclei: pd.DataFrame,
    spots: pd.DataFrame,
    pixels: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Correct spot sampling flags and summarize from the complete nucleus roster.

    Returns ``(corrected_spots, sampling_audit, nucleus_endpoints)``.  The
    authoritative nucleus table is always the left side of the aggregation, so
    sampled or eligible nuclei with zero detected spots are retained.
    """

    nucleus_keys = {
        "image_key",
        "nucleus_id",
        "eligible_for_sampling",
        "sampled_in_analysis",
    }
    spot_columns = {
        *nucleus_keys,
        "spot_uid",
        "passes_miat_floor",
        "footprint_full_nucleus_valid",
        "stored_in_nucleolus",
        "null_candidate",
        "null_usable",
        "null_exclusion_reason",
        "miat_footprint_sum_raw",
        "qki_footprint_sum_raw",
        "qki_footprint_mean_raw",
        "qki_footprint_enrichment_vs_nucleus",
        "qki_footprint_enrichment_vs_nucleoplasm",
        *(f"population_label_q{q}" for q in THRESHOLDS),
    }
    pixel_columns = {"spot_uid", "flat_pixel_index", "miat_raw", "qki_raw"}
    _require_columns(nuclei, nucleus_keys, table="nucleus roster")
    _require_columns(spots, spot_columns, table="spot table")
    _require_columns(pixels, pixel_columns, table="footprint pixel table")
    if nuclei.duplicated(["image_key", "nucleus_id"]).any():
        raise PostrunValidationError("nucleus roster contains duplicate nucleus keys")
    if spots["spot_uid"].astype(str).duplicated().any():
        raise PostrunValidationError("spot table contains duplicate spot_uid values")
    unknown_pixels = set(pixels["spot_uid"].astype(str)).difference(
        set(spots["spot_uid"].astype(str))
    )
    if unknown_pixels:
        raise PostrunValidationError(
            f"footprint pixel table has unknown spot_uid values: {sorted(unknown_pixels)[:10]}"
        )

    corrected, sampling_audit = correct_spot_sampling_flags(spots, nuclei)
    roster = nuclei.copy()
    roster["_image_key_norm"] = _normalised_image_key(roster["image_key"])
    corrected = corrected.copy()
    corrected["_image_key_norm"] = _normalised_image_key(corrected["image_key"])
    corrected["nucleus_id"] = pd.to_numeric(corrected["nucleus_id"], errors="raise").astype(int)
    pixels_work = pixels.copy()
    pixels_work["spot_uid"] = pixels_work["spot_uid"].astype(str)
    spot_parent = corrected[
        ["spot_uid", "_image_key_norm", "nucleus_id"]
    ].copy()
    spot_parent["spot_uid"] = spot_parent["spot_uid"].astype(str)
    pixels_work = pixels_work.merge(
        spot_parent,
        on="spot_uid",
        how="left",
        validate="many_to_one",
    )
    spot_groups = {
        (str(image_key), int(nucleus_id)): group.copy()
        for (image_key, nucleus_id), group in corrected.groupby(
            ["_image_key_norm", "nucleus_id"], sort=False
        )
    }
    pixel_groups = {
        (str(image_key), int(nucleus_id)): group.copy()
        for (image_key, nucleus_id), group in pixels_work.groupby(
            ["_image_key_norm", "nucleus_id"], sort=False
        )
    }

    rows: list[dict[str, Any]] = []
    for source in roster.to_dict("records"):
        image_key = str(source["_image_key_norm"])
        nucleus_id = int(source["nucleus_id"])
        group = spot_groups.get(
            (image_key, nucleus_id), corrected.iloc[:0].copy()
        )
        group["spot_uid"] = group["spot_uid"].astype(str)
        footprint_pixels = pixel_groups.get(
            (image_key, nucleus_id), pixels_work.iloc[:0].copy()
        )
        all_mask = pd.Series(True, index=group.index, dtype=bool)
        floor = (
            _bool_series(group["passes_miat_floor"], name="passes_miat_floor")
            if len(group)
            else pd.Series(False, index=group.index, dtype=bool)
        )
        null_candidate = (
            _bool_series(group["null_candidate"], name="null_candidate")
            if len(group)
            else pd.Series(False, index=group.index, dtype=bool)
        )
        null_usable = (
            _bool_series(group["null_usable"], name="null_usable")
            if len(group)
            else pd.Series(False, index=group.index, dtype=bool)
        )
        valid = (
            _bool_series(
                group["footprint_full_nucleus_valid"],
                name="footprint_full_nucleus_valid",
            )
            if len(group)
            else pd.Series(False, index=group.index, dtype=bool)
        )
        nucleolar = (
            _bool_series(group["stored_in_nucleolus"], name="stored_in_nucleolus")
            if len(group)
            else pd.Series(False, index=group.index, dtype=bool)
        )
        populations: dict[str, pd.Series] = {"all": all_mask, "floor": floor}
        record = {key: value for key, value in source.items() if key != "_image_key_norm"}
        record.update(
            {
                "nucleus_uid": f"{image_key}:nucleus:{nucleus_id}",
                "n_spots_all": int(len(group)),
                "n_spots_floor": int(floor.sum()),
                "n_below_floor": int((~floor).sum()),
                "n_valid_exact_footprints": int(valid.sum()),
                "n_spots_nucleolar": int(nucleolar.sum()),
                "n_spots_non_nucleolar": int((~nucleolar).sum()),
                "n_null_candidate": int(null_candidate.sum()),
                "n_null_usable": int(null_usable.sum()),
                "n_null_excluded": int((floor & ~null_usable).sum()),
            }
        )
        reasons = group["null_exclusion_reason"].fillna("").astype(str)
        reason_columns = {
            "n_excluded_nucleolar": "nucleolar_spot",
            "n_excluded_low_first_pass_retention": "low_first_pass_retention",
            "n_excluded_footprint_not_full_nucleoplasm": "footprint_not_full_nucleoplasm",
            "n_excluded_insufficient_spots_for_rotation": "insufficient_spots_for_rotation",
        }
        known_reasons = set(reason_columns.values()) | {""}
        for column, reason in reason_columns.items():
            record[column] = int((floor & reasons.eq(reason)).sum())
        record["n_excluded_other"] = int(
            (floor & ~reasons.isin(known_reasons)).sum()
        )

        for percentile in THRESHOLDS:
            masks = {
                population: _population_mask(group, percentile, population)
                for population in _POPULATIONS
            }
            usable = masks["positive"] | masks["negative"]
            populations.update(
                {
                    f"q{percentile}_positive": masks["positive"],
                    f"q{percentile}_negative": masks["negative"],
                    f"q{percentile}_unusable": masks["unusable"],
                    f"q{percentile}_usable": usable,
                }
            )
            for population in _POPULATIONS:
                count_column = (
                    f"n_unusable_q{percentile}"
                    if population == "unusable"
                    else f"n_threshold_{population}_q{percentile}"
                )
                record[count_column] = int(masks[population].sum())
            record[f"threshold_positive_spots_per_nucleus_q{percentile}"] = int(
                masks["positive"].sum()
            )
            record[f"association_fraction_among_usable_q{percentile}"] = _safe_ratio(
                float(masks["positive"].sum()), float(usable.sum())
            )
            record[
                f"association_fraction_among_all_floor_spots_q{percentile}"
            ] = _safe_ratio(float(masks["positive"].sum()), float(floor.sum()))
            record[f"threshold_usability_coverage_q{percentile}"] = _safe_ratio(
                float(usable.sum()), float(floor.sum())
            )
            record[f"population_reconciliation_pass_q{percentile}"] = bool(
                int(floor.sum())
                == int(masks["positive"].sum())
                + int(masks["negative"].sum())
                + int(masks["unusable"].sum())
                and len(group) == int(floor.sum()) + int((~floor).sum())
            )
            if not record[f"population_reconciliation_pass_q{percentile}"]:
                raise PostrunValidationError(
                    f"population reconciliation failed for {image_key} nucleus {nucleus_id} q{percentile}"
                )

        record.update(_mass_metrics(group, footprint_pixels, populations))
        for percentile in THRESHOLDS:
            positive_key = f"q{percentile}_positive"
            usable_key = f"q{percentile}_usable"
            positive_mass = record[
                f"miat_footprint_mass_{positive_key}_spot_summed"
            ]
            usable_mass = record[f"miat_footprint_mass_{usable_key}_spot_summed"]
            floor_mass = record["miat_footprint_mass_floor_spot_summed"]
            record[
                f"associated_miat_mass_fraction_among_usable_q{percentile}"
            ] = _safe_ratio(positive_mass, usable_mass)
            record[
                f"associated_miat_mass_fraction_among_all_floor_spots_q{percentile}"
            ] = _safe_ratio(positive_mass, floor_mass)

        continuous_means = {
            "qki_footprint_mean": "qki_footprint_mean_raw",
            "qki_enrichment_vs_nucleus_mean": "qki_footprint_enrichment_vs_nucleus",
            "qki_enrichment_vs_nucleoplasm_mean": "qki_footprint_enrichment_vs_nucleoplasm",
        }
        for output_stem, source_column in continuous_means.items():
            values = pd.to_numeric(group[source_column], errors="coerce")
            record[f"{output_stem}_all"] = (
                float(values.loc[np.isfinite(values.to_numpy(float))].mean())
                if np.isfinite(values.to_numpy(float)).any()
                else np.nan
            )
            for percentile in THRESHOLDS:
                positive = populations[f"q{percentile}_positive"]
                selected_values = values.loc[positive]
                finite_values = selected_values.loc[
                    np.isfinite(selected_values.to_numpy(float))
                ]
                record[f"{output_stem}_q{percentile}_positive"] = (
                    float(finite_values.mean()) if len(finite_values) else np.nan
                )
        # Keep the historical q95 names as explicit primary aliases.
        record["threshold_positive_spots_per_nucleus"] = record[
            "threshold_positive_spots_per_nucleus_q95"
        ]
        record["association_fraction_among_usable"] = record[
            "association_fraction_among_usable_q95"
        ]
        record["association_fraction_among_all_floor_spots"] = record[
            "association_fraction_among_all_floor_spots_q95"
        ]
        record["associated_miat_mass_fraction_among_usable"] = record[
            "associated_miat_mass_fraction_among_usable_q95"
        ]
        record["associated_miat_mass_fraction_among_all"] = record[
            "associated_miat_mass_fraction_among_all_floor_spots_q95"
        ]
        rows.append(record)

    result = pd.DataFrame(rows).reset_index(drop=True)
    corrected = corrected.drop(columns=["_image_key_norm"]).reset_index(drop=True)
    return corrected, sampling_audit, result


def aggregate_endpoint_hierarchy(
    nuclei: pd.DataFrame,
    *,
    endpoints: Iterable[str],
    cohorts: Iterable[str] = ("sampled_primary", "all_eligible"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average nuclei equally within FOVs, then FOVs equally within 12 sets."""

    endpoint_names = list(dict.fromkeys(str(endpoint) for endpoint in endpoints))
    cohort_names = list(dict.fromkeys(str(cohort) for cohort in cohorts))
    if not endpoint_names:
        raise PostrunValidationError("at least one endpoint is required")
    required = {
        "image",
        "image_key",
        "nucleus_id",
        "slide",
        "arm",
        "replicate",
        "fov",
        "biological_set",
        "is_control",
        "eligible_for_sampling",
        "sampled_in_analysis",
        *endpoint_names,
    }
    _require_columns(nuclei, required, table="nucleus endpoint table")
    frame = nuclei.copy()
    frame["is_control"] = _bool_series(frame["is_control"], name="is_control")
    frame["eligible_for_sampling"] = _bool_series(
        frame["eligible_for_sampling"], name="eligible_for_sampling"
    )
    frame["sampled_in_analysis"] = _bool_series(
        frame["sampled_in_analysis"], name="sampled_in_analysis"
    )
    if (frame["sampled_in_analysis"] & ~frame["eligible_for_sampling"]).any():
        raise PostrunValidationError(
            "sampled_in_analysis may be true only for eligible_for_sampling nuclei"
        )
    frame = frame.loc[~frame["is_control"]].copy()
    if frame.empty:
        raise PostrunValidationError("no biological nuclei remain after control exclusion")

    cohort_gates = {
        "sampled_primary": "sampled_in_analysis",
        "all_eligible": "eligible_for_sampling",
    }
    unknown = sorted(set(cohort_names).difference(cohort_gates))
    if unknown:
        raise PostrunValidationError(f"unknown cohorts: {unknown}")

    fov_rows: list[dict[str, Any]] = []
    set_rows: list[dict[str, Any]] = []
    fov_keys = [
        "image",
        "image_key",
        "slide",
        "arm",
        "replicate",
        "fov",
        "biological_set",
    ]
    set_keys = ["slide", "arm", "replicate", "biological_set"]
    for cohort in cohort_names:
        selected = frame.loc[frame[cohort_gates[cohort]]].copy()
        if selected.empty:
            raise PostrunValidationError(f"cohort {cohort!r} contains no biological nuclei")
        biological_images = set(frame["image_key"].astype(str))
        selected_images = set(selected["image_key"].astype(str))
        missing_images = sorted(biological_images.difference(selected_images))
        if missing_images:
            raise PostrunValidationError(
                f"cohort {cohort!r} has zero nuclei in biological images: {missing_images}"
            )
        for endpoint in endpoint_names:
            selected[endpoint] = pd.to_numeric(selected[endpoint], errors="coerce")
            for key_values, group in selected.groupby(fov_keys, dropna=False, sort=False):
                values = group[endpoint].to_numpy(float)
                finite = np.isfinite(values)
                fov_rows.append(
                    {
                        "cohort": cohort,
                        "endpoint": endpoint,
                        **dict(zip(fov_keys, key_values)),
                        "is_control": False,
                        "n_nuclei_total": int(len(group)),
                        "n_nuclei_finite": int(finite.sum()),
                        "value": float(values[finite].mean()) if finite.any() else np.nan,
                    }
                )
            endpoint_fov = pd.DataFrame(
                [
                    row
                    for row in fov_rows
                    if row["cohort"] == cohort and row["endpoint"] == endpoint
                ]
            )
            for key_values, group in endpoint_fov.groupby(
                set_keys, dropna=False, sort=False
            ):
                values = pd.to_numeric(group["value"], errors="coerce").to_numpy(float)
                finite = np.isfinite(values)
                set_rows.append(
                    {
                        "cohort": cohort,
                        "endpoint": endpoint,
                        **dict(zip(set_keys, key_values)),
                        "n_fovs_total": int(len(group)),
                        "n_fovs_finite": int(finite.sum()),
                        "n_nuclei_total": int(group["n_nuclei_total"].sum()),
                        "n_nuclei_finite": int(group["n_nuclei_finite"].sum()),
                        "value": float(values[finite].mean()) if finite.any() else np.nan,
                        "complete_for_inference": bool(finite.all()),
                    }
                )

    fov = pd.DataFrame(fov_rows).reset_index(drop=True)
    sets = pd.DataFrame(set_rows).reset_index(drop=True)
    for (cohort, endpoint), group in sets.groupby(["cohort", "endpoint"], sort=False):
        arm_counts = group.groupby("arm").size().to_dict()
        if len(group) != 12 or arm_counts != {"KD": 6, "NT": 6}:
            raise PostrunValidationError(
                f"{cohort}/{endpoint} must yield exactly 12 independent sets "
                f"(6 NT and 6 KD), found {len(group)} with {arm_counts}"
            )
        if group["biological_set"].astype(str).duplicated().any():
            raise PostrunValidationError(
                f"{cohort}/{endpoint} has duplicate biological_set rows"
            )
    return fov, sets


def _finite_set_frame(group: pd.DataFrame) -> pd.DataFrame:
    required = {"slide", "arm", "replicate", "biological_set", "value"}
    _require_columns(group, required, table="biological-set endpoint rows")
    frame = group.copy()
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame.loc[np.isfinite(frame["value"].to_numpy(float))].reset_index(drop=True)


def _welch(group: pd.DataFrame, *, endpoint_scale: str) -> dict[str, Any]:
    frame = _finite_set_frame(group)
    nt = frame.loc[frame["arm"].eq("NT"), "value"].to_numpy(float)
    kd = frame.loc[frame["arm"].eq("KD"), "value"].to_numpy(float)
    if len(nt) < 2 or len(kd) < 2:
        raise PostrunValidationError("Welch inference needs at least two sets per arm")
    mean_nt, mean_kd = float(nt.mean()), float(kd.mean())
    difference = mean_kd - mean_nt
    variance_nt, variance_kd = float(nt.var(ddof=1)), float(kd.var(ddof=1))
    vm_nt, vm_kd = variance_nt / len(nt), variance_kd / len(kd)
    se2 = vm_nt + vm_kd
    se = float(np.sqrt(se2))
    if se == 0:
        t_value = float(np.sign(difference) * np.inf) if difference else 0.0
        degrees = float("inf")
        p_value = 0.0 if difference else 1.0
        ci_low = ci_high = difference
    else:
        denominator = vm_nt**2 / (len(nt) - 1) + vm_kd**2 / (len(kd) - 1)
        degrees = float(se2**2 / denominator) if denominator else float("inf")
        t_value = difference / se
        p_value = float(2 * stats.t.sf(abs(t_value), degrees))
        critical = float(stats.t.ppf(0.975, degrees))
        ci_low, ci_high = difference - critical * se, difference + critical * se
    ratios_allowed = endpoint_scale == UNSIGNED_POSITIVE
    ratio_defined = ratios_allowed and mean_nt > 0
    return {
        "test": "welch_unpaired_biological_set_means",
        "n_nt": int(len(nt)),
        "n_kd": int(len(kd)),
        "mean_nt": mean_nt,
        "mean_kd": mean_kd,
        "sd_nt": float(np.sqrt(variance_nt)),
        "sd_kd": float(np.sqrt(variance_kd)),
        "difference_kd_minus_nt": difference,
        "difference_standard_error": se,
        "difference_ci95_low": float(ci_low),
        "difference_ci95_high": float(ci_high),
        "ratio_kd_over_nt": float(mean_kd / mean_nt) if ratio_defined else np.nan,
        "percent_change_kd_vs_nt": (
            float(100 * difference / mean_nt) if ratio_defined else np.nan
        ),
        "ratio_effect_allowed": ratios_allowed,
        "welch_t": float(t_value),
        "welch_df": degrees,
        "welch_p_two_sided": p_value,
    }


def _slide_adjusted_ols(group: pd.DataFrame) -> dict[str, Any]:
    frame = _finite_set_frame(group)
    slides = sorted(frame["slide"].unique(), key=str)
    condition = frame["arm"].eq("KD").astype(float).to_numpy()
    columns = [np.ones(len(frame)), condition]
    columns.extend(frame["slide"].eq(slide).astype(float).to_numpy() for slide in slides[1:])
    design = np.column_stack(columns)
    response = frame["value"].to_numpy(float)
    rank = int(np.linalg.matrix_rank(design))
    residual_df = len(frame) - rank
    if rank != design.shape[1] or residual_df <= 0:
        raise PostrunValidationError("slide fixed-block design is not estimable")
    coefficients = np.linalg.lstsq(design, response, rcond=None)[0]
    residuals = response - design @ coefficients
    residual_variance = float(residuals @ residuals / residual_df)
    covariance = residual_variance * np.linalg.inv(design.T @ design)
    se = float(np.sqrt(max(covariance[1, 1], 0.0)))
    effect = float(coefficients[1])
    if se == 0:
        t_value = float(np.sign(effect) * np.inf) if effect else 0.0
        p_value = 0.0 if effect else 1.0
        ci_low = ci_high = effect
    else:
        t_value = effect / se
        p_value = float(2 * stats.t.sf(abs(t_value), residual_df))
        critical = float(stats.t.ppf(0.975, residual_df))
        ci_low, ci_high = effect - critical * se, effect + critical * se
    return {
        "test": "ols_condition_plus_slide_fixed_block",
        "coefficient_kd_minus_nt": effect,
        "standard_error": se,
        "t_value": float(t_value),
        "residual_df": int(residual_df),
        "p_two_sided": p_value,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
    }


def _exact_permutation(group: pd.DataFrame) -> dict[str, Any]:
    frame = _finite_set_frame(group).reset_index(drop=True)
    if len(frame) != len(group):
        raise PostrunValidationError("exact permutation requires every set value")
    choices_by_slide: list[list[tuple[int, ...]]] = []
    for slide, slide_group in frame.groupby("slide", sort=True):
        indices = tuple(int(index) for index in slide_group.index)
        n_kd = int(slide_group["arm"].eq("KD").sum())
        if n_kd == 0 or n_kd == len(slide_group):
            raise PostrunValidationError(f"slide {slide!r} does not contain both arms")
        choices_by_slide.append(list(combinations(indices, n_kd)))
    values = frame["value"].to_numpy(float)
    observed_mask = frame["arm"].eq("KD").to_numpy()
    observed = float(values[observed_mask].mean() - values[~observed_mask].mean())
    all_indices = set(range(len(frame)))
    null: list[float] = []
    for slide_choices in product(*choices_by_slide):
        kd_indices = set().union(*(set(choice) for choice in slide_choices))
        nt_indices = all_indices.difference(kd_indices)
        null.append(
            float(values[sorted(kd_indices)].mean() - values[sorted(nt_indices)].mean())
        )
    null_values = np.asarray(null, dtype=float)
    tolerance = np.finfo(float).eps * max(1.0, abs(observed)) * 16
    extreme = int(np.count_nonzero(np.abs(null_values) >= abs(observed) - tolerance))
    return {
        "test": "exact_within_slide_label_permutation",
        "observed_difference_kd_minus_nt": observed,
        "n_permutations": int(len(null_values)),
        "extreme_permutations_two_sided": extreme,
        "p_exact_two_sided": float(extreme / len(null_values)),
        "null_mean": float(null_values.mean()),
        "null_sd": float(null_values.std(ddof=0)),
    }


def _blank_inference_record() -> dict[str, Any]:
    return {
        "n_nt": np.nan,
        "n_kd": np.nan,
        "mean_nt": np.nan,
        "mean_kd": np.nan,
        "sd_nt": np.nan,
        "sd_kd": np.nan,
        "difference_kd_minus_nt": np.nan,
        "difference_standard_error": np.nan,
        "difference_ci95_low": np.nan,
        "difference_ci95_high": np.nan,
        "ratio_kd_over_nt": np.nan,
        "percent_change_kd_vs_nt": np.nan,
        "ratio_effect_allowed": False,
        "welch_t": np.nan,
        "welch_df": np.nan,
        "welch_p_two_sided": np.nan,
        "slide_adjusted_coefficient_kd_minus_nt": np.nan,
        "slide_adjusted_standard_error": np.nan,
        "slide_adjusted_t_value": np.nan,
        "slide_adjusted_residual_df": np.nan,
        "slide_adjusted_p_two_sided": np.nan,
        "slide_adjusted_ci95_low": np.nan,
        "slide_adjusted_ci95_high": np.nan,
        "permutation_observed_difference_kd_minus_nt": np.nan,
        "permutation_n_permutations": np.nan,
        "permutation_extreme_permutations_two_sided": np.nan,
        "permutation_p_exact_two_sided": np.nan,
        "permutation_null_mean": np.nan,
        "permutation_null_sd": np.nan,
    }


def _inference_row(
    group: pd.DataFrame,
    *,
    cohort: str,
    endpoint: str,
    endpoint_scale: str,
    analysis_role: str,
    excluded_set: str = "",
) -> dict[str, Any]:
    expected_permutations = 200 if excluded_set == "S1_KD_1" else 400
    welch = _welch(group, endpoint_scale=endpoint_scale)
    ols = _slide_adjusted_ols(group)
    permutation = _exact_permutation(group)
    if permutation["n_permutations"] != expected_permutations:
        raise PostrunValidationError(
            f"{analysis_role} must enumerate exactly {expected_permutations} labelings, "
            f"found {permutation['n_permutations']}"
        )
    row: dict[str, Any] = {
        "cohort": cohort,
        "endpoint": endpoint,
        "endpoint_scale": endpoint_scale,
        "analysis_role": analysis_role,
        "excluded_biological_set": excluded_set,
        "pairing_semantics": "none_independent_biological_sets",
        "inference_status": "complete",
        "n_complete_sets": int(len(group)),
        **welch,
    }
    row.update({f"slide_adjusted_{key}": value for key, value in ols.items()})
    row.update({f"permutation_{key}": value for key, value in permutation.items()})
    return row


def build_endpoint_inference(
    set_means: pd.DataFrame,
    *,
    endpoint_scales: dict[str, str],
    include_s1_kd1_exclusion_sensitivity: bool = False,
) -> pd.DataFrame:
    """Run locked independent-set tests or emit an explicit incomplete gate."""

    required = {
        "cohort",
        "endpoint",
        "slide",
        "arm",
        "replicate",
        "biological_set",
        "value",
    }
    _require_columns(set_means, required, table="set endpoint table")
    unknown_endpoints = sorted(set(set_means["endpoint"]).difference(endpoint_scales))
    if unknown_endpoints:
        raise PostrunValidationError(f"missing endpoint scale declarations: {unknown_endpoints}")
    invalid_scales = sorted(set(endpoint_scales.values()).difference(_VALID_ENDPOINT_SCALES))
    if invalid_scales:
        raise PostrunValidationError(f"invalid endpoint scales: {invalid_scales}")
    rows: list[dict[str, Any]] = []
    for (cohort, endpoint), group in set_means.groupby(["cohort", "endpoint"], sort=False):
        group = group.copy().reset_index(drop=True)
        if len(group) != 12:
            raise PostrunValidationError(
                f"{cohort}/{endpoint} must contain exactly 12 biological-set rows"
            )
        numeric_values = pd.to_numeric(group["value"], errors="coerce")
        finite = np.isfinite(numeric_values.to_numpy(float))
        if (
            endpoint_scales[endpoint] == UNSIGNED_POSITIVE
            and (numeric_values.loc[finite] < 0).any()
        ):
            raise PostrunValidationError(
                f"unsigned_positive endpoint {endpoint!r} must be nonnegative"
            )
        complete_flags = (
            _bool_series(group["complete_for_inference"], name="complete_for_inference").to_numpy()
            if "complete_for_inference" in group
            else np.ones(len(group), dtype=bool)
        )
        complete = finite & complete_flags
        if not complete.all():
            rows.append(
                {
                    "cohort": cohort,
                    "endpoint": endpoint,
                    "endpoint_scale": endpoint_scales[endpoint],
                    "analysis_role": "primary",
                    "excluded_biological_set": "",
                    "pairing_semantics": "none_independent_biological_sets",
                    "inference_status": "descriptive_only_incomplete",
                    "n_complete_sets": int(complete.sum()),
                    **_blank_inference_record(),
                }
            )
            # Restore the nonblank completeness count overwritten by the template.
            rows[-1]["n_complete_sets"] = int(complete.sum())
            continue
        rows.append(
            _inference_row(
                group,
                cohort=str(cohort),
                endpoint=str(endpoint),
                endpoint_scale=endpoint_scales[endpoint],
                analysis_role="primary",
            )
        )
        if include_s1_kd1_exclusion_sensitivity and cohort == "sampled_primary":
            sensitivity = group.loc[~group["biological_set"].astype(str).eq("S1_KD_1")].copy()
            if len(sensitivity) != 11:
                raise PostrunValidationError(
                    "S1_KD_1 whole-set sensitivity requires exactly one matching set"
                )
            rows.append(
                _inference_row(
                    sensitivity,
                    cohort=str(cohort),
                    endpoint=str(endpoint),
                    endpoint_scale=endpoint_scales[endpoint],
                    analysis_role="s1_kd1_whole_set_exclusion_sensitivity",
                    excluded_set="S1_KD_1",
                )
            )
    return pd.DataFrame(rows).reset_index(drop=True)


def _contrast_effect(nt_value: float, kd_value: float, endpoint_scale: str) -> dict[str, Any]:
    difference = kd_value - nt_value
    if endpoint_scale == UNSIGNED_POSITIVE:
        defined = nt_value > 0
        return {
            "kd_minus_nt": difference,
            "ratio_defined": defined,
            "kd_over_nt": float(kd_value / nt_value) if defined else np.nan,
            "percent_change_kd_vs_nt": (
                float(100 * difference / nt_value) if defined else np.nan
            ),
        }
    return {
        "kd_minus_nt": difference,
        "ratio_defined": False,
        "kd_over_nt": np.nan,
        "percent_change_kd_vs_nt": np.nan,
    }


def build_cartesian_contrasts(
    set_means: pd.DataFrame,
    *,
    endpoint_scales: dict[str, str],
    include_s1_kd1_exclusion_sensitivity: bool = False,
) -> pd.DataFrame:
    """Return within-slide NT×KD descriptive grids with no inferential fields."""

    required = {"cohort", "endpoint", "slide", "arm", "replicate", "biological_set", "value"}
    _require_columns(set_means, required, table="set endpoint table")
    rows: list[dict[str, Any]] = []

    def append_design(
        group: pd.DataFrame,
        *,
        cohort: str,
        endpoint: str,
        slide: Any,
        role: str,
        expected_kd: int,
        excluded: str,
    ) -> None:
        nt = group.loc[group["arm"].eq("NT")].sort_values("replicate", kind="stable")
        kd = group.loc[group["arm"].eq("KD")].sort_values("replicate", kind="stable")
        if len(nt) != 3 or len(kd) != expected_kd:
            raise PostrunValidationError(
                f"{role} slide {slide} requires 3 NT and {expected_kd} KD sets"
            )
        for order, (nt_row, kd_row) in enumerate(
            product(nt.itertuples(index=False), kd.itertuples(index=False)), start=1
        ):
            nt_value, kd_value = float(nt_row.value), float(kd_row.value)
            rows.append(
                {
                    "cohort": cohort,
                    "endpoint": endpoint,
                    "endpoint_scale": endpoint_scales[endpoint],
                    "analysis_role": role,
                    "slide": slide,
                    "cartesian_order_within_slide": order,
                    "n_nt_sets": 3,
                    "n_kd_sets": expected_kd,
                    "n_cartesian_contrasts": 3 * expected_kd,
                    "excluded_biological_set": excluded,
                    "nt_biological_set": str(nt_row.biological_set),
                    "nt_replicate": int(nt_row.replicate),
                    "nt_value": nt_value,
                    "kd_biological_set": str(kd_row.biological_set),
                    "kd_replicate": int(kd_row.replicate),
                    "kd_value": kd_value,
                    "pairing_semantics": "none_cartesian_cross_set_descriptive",
                    "replicate_is_pairing_key": False,
                    "inferential_test": "none_descriptive_only",
                    **_contrast_effect(nt_value, kd_value, endpoint_scales[endpoint]),
                }
            )

    for (cohort, endpoint), group in set_means.groupby(["cohort", "endpoint"], sort=False):
        if endpoint not in endpoint_scales:
            raise PostrunValidationError(f"missing endpoint scale for {endpoint!r}")
        values = pd.to_numeric(group["value"], errors="coerce")
        if len(group) != 12 or not np.isfinite(values.to_numpy(float)).all():
            raise PostrunValidationError(
                f"Cartesian contrasts require 12 finite rows for {cohort}/{endpoint}"
            )
        if endpoint_scales[endpoint] == UNSIGNED_POSITIVE and (values < 0).any():
            raise PostrunValidationError(
                f"unsigned_positive endpoint {endpoint!r} must be nonnegative"
            )
        for slide, slide_group in group.groupby("slide", sort=True):
            append_design(
                slide_group,
                cohort=str(cohort),
                endpoint=str(endpoint),
                slide=slide,
                role="primary_descriptive_full_cohort",
                expected_kd=3,
                excluded="",
            )
        if include_s1_kd1_exclusion_sensitivity and cohort == "sampled_primary":
            slide_one = group.loc[
                group["slide"].eq(1) & ~group["biological_set"].astype(str).eq("S1_KD_1")
            ]
            append_design(
                slide_one,
                cohort=str(cohort),
                endpoint=str(endpoint),
                slide=1,
                role="declared_incomplete_sensitivity",
                expected_kd=2,
                excluded="S1_KD_1",
            )
    return pd.DataFrame(rows).reset_index(drop=True)


def ratio_of_ratios(
    set_means: pd.DataFrame,
    *,
    numerator_endpoint: str,
    denominator_endpoint: str,
    cohort: str,
    endpoint_scales: dict[str, str],
) -> dict[str, Any]:
    """Compare two unsigned arm-mean KD/NT ratios without cross-arm pairing."""

    scales = {
        endpoint_scales.get(numerator_endpoint),
        endpoint_scales.get(denominator_endpoint),
    }
    if scales != {UNSIGNED_POSITIVE}:
        raise PostrunValidationError(
            "ratio-of-ratios requires both endpoint scales to be unsigned_positive"
        )

    frame = set_means.loc[
        set_means["cohort"].astype(str).eq(cohort)
        & set_means["endpoint"].astype(str).isin([numerator_endpoint, denominator_endpoint])
    ].copy()
    _require_columns(
        frame,
        {"biological_set", "arm", "endpoint", "value"},
        table="ratio-of-ratios set means",
    )
    wide = frame.pivot(index=["biological_set", "arm"], columns="endpoint", values="value").reset_index()
    if set(wide["arm"]) != {"NT", "KD"} or wide.groupby("arm").size().to_dict() != {"KD": 6, "NT": 6}:
        raise PostrunValidationError("ratio-of-ratios requires six independent sets per arm")
    values = wide[[numerator_endpoint, denominator_endpoint]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy(float)).all() or (values <= 0).any().any():
        raise PostrunValidationError(
            "ratio-of-ratios is defined only for finite strictly positive endpoints"
        )
    means: dict[str, tuple[float, float]] = {}
    variance_log = 0.0
    for arm in ("NT", "KD"):
        arm_values = wide.loc[wide["arm"].eq(arm), [numerator_endpoint, denominator_endpoint]].to_numpy(float)
        mean_num, mean_den = arm_values.mean(axis=0)
        means[arm] = (float(mean_num), float(mean_den))
        covariance = np.cov(arm_values, rowvar=False, ddof=1)
        gradient = np.array([1.0 / mean_num, -1.0 / mean_den])
        variance_log += float(gradient @ (covariance / len(arm_values)) @ gradient)
    nt_num, nt_den = means["NT"]
    kd_num, kd_den = means["KD"]
    numerator_ratio = kd_num / nt_num
    denominator_ratio = kd_den / nt_den
    effect = numerator_ratio / denominator_ratio
    log_effect = log(effect)
    standard_error = float(np.sqrt(max(variance_log, 0.0)))
    if standard_error == 0:
        low = high = effect
        z_value = float(np.sign(log_effect) * np.inf) if log_effect else 0.0
        p_value = 0.0 if log_effect else 1.0
    else:
        critical = float(stats.norm.ppf(0.975))
        low, high = exp(log_effect - critical * standard_error), exp(
            log_effect + critical * standard_error
        )
        z_value = log_effect / standard_error
        p_value = float(2 * stats.norm.sf(abs(z_value)))
    return {
        "cohort": cohort,
        "numerator_endpoint": numerator_endpoint,
        "denominator_endpoint": denominator_endpoint,
        "n_nt": 6,
        "n_kd": 6,
        "numerator_ratio_kd_over_nt": float(numerator_ratio),
        "denominator_ratio_kd_over_nt": float(denominator_ratio),
        "ratio_of_ratios": float(effect),
        "log_ratio_of_ratios": float(log_effect),
        "log_ratio_of_ratios_standard_error": standard_error,
        "ratio_of_ratios_ci95_low": float(low),
        "ratio_of_ratios_ci95_high": float(high),
        "z_value": float(z_value),
        "p_two_sided": p_value,
        "ci_method": "multivariate_delta_within_arm_covariance",
        "pairing_semantics": "none_arm_mean_ratio_of_ratios",
        "endpoint_scale": UNSIGNED_POSITIVE,
    }


def _correlation_value(
    x: np.ndarray,
    y: np.ndarray,
    *,
    method: str,
) -> dict[str, Any]:
    n_spots = int(len(x))
    if n_spots < 3:
        return {
            "n_spots": n_spots,
            "correlation": np.nan,
            "p_value": np.nan,
            "estimable": False,
            "fisher_estimable": False,
            "fisher_z": np.nan,
            "nonestimable_reason": "fewer_than_3_spots",
        }
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return {
            "n_spots": n_spots,
            "correlation": np.nan,
            "p_value": np.nan,
            "estimable": False,
            "fisher_estimable": False,
            "fisher_z": np.nan,
            "nonestimable_reason": "constant_vector",
        }
    if method == "pearson":
        estimate = stats.pearsonr(x, y)
    elif method == "spearman":
        estimate = stats.spearmanr(x, y)
    else:
        raise PostrunValidationError(f"unknown correlation method {method!r}")
    correlation = float(estimate.statistic)
    # Do not clip a boundary correlation into an invented finite Fisher value.
    boundary = bool(np.isclose(abs(correlation), 1.0, rtol=0.0, atol=8 * np.finfo(float).eps))
    return {
        "n_spots": n_spots,
        "correlation": correlation,
        "p_value": float(estimate.pvalue),
        "estimable": not boundary,
        "fisher_estimable": not boundary,
        "fisher_z": np.nan if boundary else float(np.arctanh(correlation)),
        "nonestimable_reason": "exact_boundary_correlation" if boundary else "",
    }


def compute_fov_correlations(
    spots: pd.DataFrame,
    nuclei: pd.DataFrame,
    *,
    cohorts: Iterable[str] = ("sampled_primary", "all_eligible"),
) -> pd.DataFrame:
    """Compute continuous correlations directly from spots in each biological FOV."""

    nucleus_columns = {
        "image",
        "image_key",
        "nucleus_id",
        "slide",
        "arm",
        "replicate",
        "fov",
        "biological_set",
        "is_control",
        "eligible_for_sampling",
        "sampled_in_analysis",
    }
    spot_columns = {
        "image_key",
        "nucleus_id",
        "footprint_full_nucleus_valid",
        "null_usable",
        "miat_footprint_mean_raw",
        "qki_footprint_mean_raw",
        "miat_footprint_enrichment_vs_nucleus",
        "qki_footprint_enrichment_vs_nucleus",
        *(f"population_label_q{q}" for q in THRESHOLDS),
    }
    _require_columns(nuclei, nucleus_columns, table="nucleus roster")
    _require_columns(spots, spot_columns, table="corrected spot table")
    roster = nuclei.copy()
    roster["is_control"] = _bool_series(roster["is_control"], name="is_control")
    roster["eligible_for_sampling"] = _bool_series(
        roster["eligible_for_sampling"], name="eligible_for_sampling"
    )
    roster["sampled_in_analysis"] = _bool_series(
        roster["sampled_in_analysis"], name="sampled_in_analysis"
    )
    roster["_image_key_norm"] = _normalised_image_key(roster["image_key"])
    roster["_nucleus_id_norm"] = pd.to_numeric(
        roster["nucleus_id"], errors="raise"
    ).astype(int)
    if roster.duplicated(["_image_key_norm", "_nucleus_id_norm"]).any():
        raise PostrunValidationError("nucleus roster contains duplicate nucleus keys")
    spot_frame = spots.copy()
    spot_frame["_image_key_norm"] = _normalised_image_key(spot_frame["image_key"])
    spot_frame["_nucleus_id_norm"] = pd.to_numeric(
        spot_frame["nucleus_id"], errors="raise"
    ).astype(int)
    cohort_gates = {
        "sampled_primary": "sampled_in_analysis",
        "all_eligible": "eligible_for_sampling",
    }
    cohort_names = list(dict.fromkeys(str(cohort) for cohort in cohorts))
    unknown = sorted(set(cohort_names).difference(cohort_gates))
    if unknown:
        raise PostrunValidationError(f"unknown cohorts: {unknown}")
    identity = [
        "image",
        "image_key",
        "slide",
        "arm",
        "replicate",
        "fov",
        "biological_set",
    ]
    pairs = {
        "raw": ("miat_footprint_mean_raw", "qki_footprint_mean_raw"),
        "within_nucleus_normalized": (
            "miat_footprint_enrichment_vs_nucleus",
            "qki_footprint_enrichment_vs_nucleus",
        ),
    }
    rows: list[dict[str, Any]] = []
    for cohort in cohort_names:
        selected_nuclei = roster.loc[
            ~roster["is_control"] & roster[cohort_gates[cohort]]
        ].copy()
        if selected_nuclei.empty:
            raise PostrunValidationError(f"cohort {cohort!r} contains no biological nuclei")
        biological_images = set(roster.loc[~roster["is_control"], "_image_key_norm"])
        selected_images = set(selected_nuclei["_image_key_norm"])
        missing = sorted(biological_images.difference(selected_images))
        if missing:
            raise PostrunValidationError(
                f"cohort {cohort!r} has zero nuclei in biological images: {missing}"
            )
        selected_keys = selected_nuclei[["_image_key_norm", "_nucleus_id_norm"]].drop_duplicates()
        cohort_spots = spot_frame.merge(
            selected_keys,
            on=["_image_key_norm", "_nucleus_id_norm"],
            how="inner",
            validate="many_to_one",
        )
        # One row per selected biological FOV is retained even when it has no spots.
        fov_roster = selected_nuclei[identity + ["_image_key_norm"]].drop_duplicates()
        if fov_roster["_image_key_norm"].duplicated().any():
            raise PostrunValidationError("one image_key maps to multiple FOV identities")
        for fov_source in fov_roster.to_dict("records"):
            fov_spots = cohort_spots.loc[
                cohort_spots["_image_key_norm"].eq(fov_source["_image_key_norm"])
            ].copy()
            local_valid = _bool_series(
                fov_spots["footprint_full_nucleus_valid"],
                name="footprint_full_nucleus_valid",
            )
            local_usable = _bool_series(
                fov_spots["null_usable"], name="null_usable"
            )
            populations: list[tuple[str, pd.Series, bool]] = [
                ("all_detected_exact_valid", local_valid, False)
            ]
            for percentile in THRESHOLDS:
                positive = fov_spots[f"population_label_q{percentile}"].astype(str).eq(
                    "threshold_positive"
                )
                populations.append(
                    (
                        f"threshold_positive_q{percentile}",
                        local_valid & local_usable & positive,
                        True,
                    )
                )
            identity_values = {key: fov_source[key] for key in identity}
            for population, population_gate, conditional in populations:
                for pair_name, (miat_column, qki_column) in pairs.items():
                    x = pd.to_numeric(fov_spots[miat_column], errors="coerce")
                    y = pd.to_numeric(fov_spots[qki_column], errors="coerce")
                    finite = np.isfinite(x.to_numpy(float)) & np.isfinite(y.to_numpy(float))
                    gate = population_gate.to_numpy(bool) & finite
                    x_values = x.to_numpy(float)[gate]
                    y_values = y.to_numpy(float)[gate]
                    for method in ("pearson", "spearman"):
                        correlation = _correlation_value(
                            x_values, y_values, method=method
                        )
                        if conditional:
                            correlation["p_value"] = np.nan
                        rows.append(
                            {
                                "cohort": cohort,
                                **identity_values,
                                "population": population,
                                "measurement_pair": pair_name,
                                "correlation_method": method,
                                "conditional_descriptive": conditional,
                                **correlation,
                            }
                        )
    return pd.DataFrame(rows).reset_index(drop=True)


def aggregate_correlation_hierarchy(fov_correlations: pd.DataFrame) -> pd.DataFrame:
    """Average finite Fisher z values equally across FOVs within each set."""

    keys = [
        "cohort",
        "population",
        "measurement_pair",
        "correlation_method",
        "conditional_descriptive",
        "slide",
        "arm",
        "replicate",
        "biological_set",
    ]
    required = {*keys, "image_key", "fisher_estimable", "fisher_z", "n_spots"}
    _require_columns(fov_correlations, required, table="FOV correlation table")
    rows: list[dict[str, Any]] = []
    for key_values, group in fov_correlations.groupby(keys, dropna=False, sort=False):
        estimable = _bool_series(group["fisher_estimable"], name="fisher_estimable")
        fisher = pd.to_numeric(group["fisher_z"], errors="coerce")
        finite = estimable & np.isfinite(fisher.to_numpy(float))
        mean_z = float(fisher.loc[finite].mean()) if finite.any() else np.nan
        rows.append(
            {
                **dict(zip(keys, key_values)),
                "n_fovs_total": int(len(group)),
                "n_fovs_estimable": int(finite.sum()),
                "n_spots_total": int(pd.to_numeric(group["n_spots"], errors="coerce").sum()),
                "fisher_z_mean": mean_z,
                "value": mean_z,
                "correlation_backtransformed": (
                    float(np.tanh(mean_z)) if np.isfinite(mean_z) else np.nan
                ),
                "complete_for_inference": bool(finite.all()),
                "incomplete_reason": "" if finite.all() else "one_or_more_fovs_nonestimable",
            }
        )
    result = pd.DataFrame(rows).reset_index(drop=True)
    combination_keys = ["cohort", "population", "measurement_pair", "correlation_method"]
    for combination, group in result.groupby(combination_keys, sort=False):
        arm_counts = group.groupby("arm").size().to_dict()
        if len(group) != 12 or arm_counts != {"KD": 6, "NT": 6}:
            raise PostrunValidationError(
                f"correlation hierarchy {combination} must yield 12 sets (6 NT, 6 KD), "
                f"found {len(group)} with {arm_counts}"
            )
    return result


def build_correlation_inference(set_correlations: pd.DataFrame) -> pd.DataFrame:
    """Infer all-spot Fisher-z effects; retain threshold-selected effects as descriptive."""

    required = {
        "cohort",
        "population",
        "measurement_pair",
        "correlation_method",
        "conditional_descriptive",
        "slide",
        "arm",
        "replicate",
        "biological_set",
        "value",
        "complete_for_inference",
    }
    _require_columns(set_correlations, required, table="set correlation table")
    rows: list[dict[str, Any]] = []
    grouping = ["cohort", "population", "measurement_pair", "correlation_method"]
    for key_values, group in set_correlations.groupby(grouping, sort=False):
        cohort, population, pair, method = key_values
        if len(group) != 12:
            raise PostrunValidationError(f"correlation group {key_values} must contain 12 sets")
        finite = np.isfinite(pd.to_numeric(group["value"], errors="coerce").to_numpy(float))
        complete_flags = _bool_series(
            group["complete_for_inference"], name="complete_for_inference"
        ).to_numpy()
        complete = finite & complete_flags
        conditional = bool(group["conditional_descriptive"].astype(bool).all())
        metadata = {
            "cohort": cohort,
            "population": population,
            "measurement_pair": pair,
            "correlation_method": method,
            "conditional_descriptive": conditional,
            "endpoint_scale": SIGNED_ADDITIVE,
            "pairing_semantics": "none_independent_biological_sets",
            "n_complete_sets": int(complete.sum()),
        }
        if conditional:
            descriptive = _blank_inference_record()
            finite_group = group.loc[finite].copy()
            nt = pd.to_numeric(
                finite_group.loc[finite_group["arm"].eq("NT"), "value"], errors="coerce"
            ).to_numpy(float)
            kd = pd.to_numeric(
                finite_group.loc[finite_group["arm"].eq("KD"), "value"], errors="coerce"
            ).to_numpy(float)
            descriptive.update(
                {
                    "n_nt": int(len(nt)),
                    "n_kd": int(len(kd)),
                    "mean_nt": float(nt.mean()) if len(nt) else np.nan,
                    "mean_kd": float(kd.mean()) if len(kd) else np.nan,
                    "difference_kd_minus_nt": (
                        float(kd.mean() - nt.mean()) if len(nt) and len(kd) else np.nan
                    ),
                }
            )
            rows.append(
                {
                    **metadata,
                    "analysis_role": "conditional_threshold_selected_descriptive",
                    "inference_status": (
                        "descriptive_conditional"
                        if complete.all()
                        else "descriptive_only_incomplete_conditional"
                    ),
                    **descriptive,
                }
            )
            rows[-1]["n_complete_sets"] = int(complete.sum())
            continue
        if not complete.all():
            rows.append(
                {
                    **metadata,
                    "analysis_role": "primary_continuous_all_spots",
                    "inference_status": "descriptive_only_incomplete",
                    **_blank_inference_record(),
                }
            )
            rows[-1]["n_complete_sets"] = int(complete.sum())
            continue
        infer_group = group.copy()
        record = _inference_row(
            infer_group,
            cohort=str(cohort),
            endpoint=f"{population}__{pair}__{method}__fisher_z",
            endpoint_scale=SIGNED_ADDITIVE,
            analysis_role="primary_continuous_all_spots",
        )
        record.update(metadata)
        rows.append(record)
    return pd.DataFrame(rows).reset_index(drop=True)


def default_endpoint_scales(nucleus_endpoints: pd.DataFrame) -> dict[str, str]:
    """Return the nonnegative endpoint registry used by the post-run workbook."""

    exact_names = {
        "n_spots_all",
        "n_spots_floor",
        "n_below_floor",
        "n_valid_exact_footprints",
        "n_spots_nucleolar",
        "n_spots_non_nucleolar",
        "n_null_candidate",
        "n_null_usable",
        "n_null_excluded",
        "n_excluded_nucleolar",
        "n_excluded_low_first_pass_retention",
        "n_excluded_footprint_not_full_nucleoplasm",
        "n_excluded_insufficient_spots_for_rotation",
        "n_excluded_other",
        "whole_nucleus_miat_sum_raw",
        "whole_nucleus_miat_mean_raw",
        "whole_nucleus_qki_sum_raw",
        "whole_nucleus_qki_mean_raw",
        "nucleoplasm_miat_sum_raw",
        "nucleoplasm_miat_mean_raw",
        "nucleoplasm_qki_sum_raw",
        "nucleoplasm_qki_mean_raw",
    }
    prefixes = (
        "n_threshold_positive_q",
        "n_threshold_negative_q",
        "n_unusable_q",
        "threshold_positive_spots_per_nucleus_q",
        "association_fraction_among_usable_q",
        "association_fraction_among_all_floor_spots_q",
        "threshold_usability_coverage_q",
        "miat_footprint_mass_",
        "qki_footprint_mass_",
        "associated_miat_mass_fraction_",
        "qki_footprint_mean_",
        "qki_enrichment_vs_nucleus_mean_",
        "qki_enrichment_vs_nucleoplasm_mean_",
    )
    registry: dict[str, str] = {}
    for column in nucleus_endpoints.columns:
        if column in exact_names or column.startswith(prefixes):
            numeric = pd.to_numeric(nucleus_endpoints[column], errors="coerce")
            present = numeric.loc[numeric.notna()]
            if len(present) and (present < 0).any():
                raise PostrunValidationError(
                    f"unsigned endpoint {column!r} contains negative values"
                )
            registry[column] = UNSIGNED_POSITIVE
    if not registry:
        raise PostrunValidationError("no default post-run endpoints were found")
    return registry


def _resolve_ratio_specs(
    ratio_specs: Iterable[tuple[str, str]] | None,
) -> tuple[tuple[str, str], ...]:
    source = DEFAULT_RATIO_OF_RATIOS_SPECS if ratio_specs is None else ratio_specs
    resolved: list[tuple[str, str]] = []
    for spec in source:
        if len(spec) != 2:
            raise PostrunValidationError(
                "each ratio-of-ratios spec must contain numerator and denominator endpoints"
            )
        numerator, denominator = (str(value).strip() for value in spec)
        if not numerator or not denominator or numerator == denominator:
            raise PostrunValidationError(
                "ratio-of-ratios endpoints must be distinct non-empty names"
            )
        resolved.append((numerator, denominator))
    if len(resolved) != len(set(resolved)):
        raise PostrunValidationError("ratio-of-ratios specs contain duplicates")
    return tuple(resolved)


def build_postrun_statistics(
    nuclei: pd.DataFrame,
    spots: pd.DataFrame,
    pixels: pd.DataFrame,
    *,
    endpoints: Iterable[str] | None = None,
    endpoint_scales: dict[str, str] | None = None,
    cohorts: Iterable[str] = ("sampled_primary", "all_eligible"),
    ratio_specs: Iterable[tuple[str, str]] | None = None,
) -> PostrunResults:
    """Build every downstream table from retained canonical inputs in memory."""

    quantitation_audit = validate_quantitation_invariants(spots, nuclei)
    corrected, sampling_audit, nucleus_endpoints = build_nucleus_endpoints(
        nuclei, spots, pixels
    )
    resolved_ratio_specs = _resolve_ratio_specs(ratio_specs)
    resolved_cohorts = tuple(dict.fromkeys(str(value) for value in cohorts))
    if not resolved_cohorts:
        raise PostrunValidationError("at least one cohort is required")
    resolved_scales = (
        dict(endpoint_scales)
        if endpoint_scales is not None
        else default_endpoint_scales(nucleus_endpoints)
    )
    endpoint_names = (
        list(dict.fromkeys(str(endpoint) for endpoint in endpoints))
        if endpoints is not None
        else list(resolved_scales)
    )
    missing_scales = sorted(set(endpoint_names).difference(resolved_scales))
    if missing_scales:
        raise PostrunValidationError(
            f"selected endpoints lack scale declarations: {missing_scales}"
        )
    ratio_endpoints = {
        endpoint for spec in resolved_ratio_specs for endpoint in spec
    }
    missing_ratio_endpoints = sorted(ratio_endpoints.difference(endpoint_names))
    if missing_ratio_endpoints:
        raise PostrunValidationError(
            "ratio-of-ratios specs require selected endpoints: "
            f"{missing_ratio_endpoints}; pass ratio_specs=[] to disable the locked defaults"
        )
    non_unsigned_ratio_endpoints = sorted(
        endpoint
        for endpoint in ratio_endpoints
        if resolved_scales.get(endpoint) != UNSIGNED_POSITIVE
    )
    if non_unsigned_ratio_endpoints:
        raise PostrunValidationError(
            "ratio-of-ratios specs require unsigned_positive endpoint scales: "
            f"{non_unsigned_ratio_endpoints}"
        )
    fov_means, set_means = aggregate_endpoint_hierarchy(
        nucleus_endpoints, endpoints=endpoint_names, cohorts=resolved_cohorts
    )
    inference = build_endpoint_inference(
        set_means,
        endpoint_scales={endpoint: resolved_scales[endpoint] for endpoint in endpoint_names},
        include_s1_kd1_exclusion_sensitivity=True,
    )
    complete_groups: list[pd.DataFrame] = []
    for _, group in set_means.groupby(["cohort", "endpoint"], sort=False):
        values = pd.to_numeric(group["value"], errors="coerce").to_numpy(float)
        complete = _bool_series(
            group["complete_for_inference"], name="complete_for_inference"
        ).to_numpy()
        if np.isfinite(values).all() and complete.all():
            complete_groups.append(group)
    cartesian_input = (
        pd.concat(complete_groups, ignore_index=True)
        if complete_groups
        else set_means.iloc[:0].copy()
    )
    cartesian = (
        build_cartesian_contrasts(
            cartesian_input,
            endpoint_scales={endpoint: resolved_scales[endpoint] for endpoint in endpoint_names},
            include_s1_kd1_exclusion_sensitivity=True,
        )
        if len(cartesian_input)
        else pd.DataFrame()
    )
    fov_correlations = compute_fov_correlations(
        corrected, nuclei, cohorts=resolved_cohorts
    )
    set_correlations = aggregate_correlation_hierarchy(fov_correlations)
    correlation_inference = build_correlation_inference(set_correlations)
    ratio_rows: list[dict[str, Any]] = []
    for cohort in resolved_cohorts:
        for numerator, denominator in resolved_ratio_specs:
            ratio_rows.append(
                ratio_of_ratios(
                    set_means,
                    numerator_endpoint=str(numerator),
                    denominator_endpoint=str(denominator),
                    cohort=cohort,
                    endpoint_scales=resolved_scales,
                )
            )
    return PostrunResults(
        corrected_spots=corrected,
        sampling_audit=sampling_audit,
        quantitation_audit=quantitation_audit,
        ratio_specs=resolved_ratio_specs,
        nucleus_endpoints=nucleus_endpoints,
        endpoint_fov_means=fov_means,
        endpoint_set_means=set_means,
        endpoint_inference=inference,
        cartesian_contrasts=cartesian,
        fov_correlations=fov_correlations,
        set_correlations=set_correlations,
        correlation_inference=correlation_inference,
        ratio_of_ratios=pd.DataFrame(ratio_rows),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)


def run_postrun_directory(
    source_run: str | Path,
    output_dir: str | Path,
    *,
    endpoints: Iterable[str] | None = None,
    endpoint_scales: dict[str, str] | None = None,
    cohorts: Iterable[str] = ("sampled_primary", "all_eligible"),
    ratio_specs: Iterable[tuple[str, str]] | None = None,
) -> PostrunResults:
    """Create a non-overwriting post-run revision from one completed source run.

    The corrected spot table and sampling audit are written before any summary
    is computed, making the hierarchy repair an explicit prerequisite rather
    than a downstream filter hidden inside statistics code.
    """

    source = Path(source_run).resolve()
    output = Path(output_dir).resolve()
    if output == source or output.is_relative_to(source):
        raise PostrunValidationError(
            "output_dir must be outside the completed source run"
        )
    if not source.is_dir():
        raise PostrunValidationError(f"completed source run does not exist: {source}")
    required_paths = {
        "spots": source / "spot_exact_footprint_metrics.csv.gz",
        "nuclei": source / "nucleus_exact_footprint_metrics.csv",
        "pixels": source / "footprint_pixels.csv.gz",
    }
    missing = [str(path) for path in required_paths.values() if not path.is_file()]
    if missing:
        raise PostrunValidationError(f"completed source run is missing canonical tables: {missing}")
    if output.exists() and any(output.iterdir()):
        raise PostrunValidationError(f"output directory already exists and is not empty: {output}")

    spots = pd.read_csv(required_paths["spots"], low_memory=False)
    nuclei = pd.read_csv(required_paths["nuclei"], low_memory=False)
    pixels = pd.read_csv(required_paths["pixels"], low_memory=False)
    quantitation_audit = validate_quantitation_invariants(spots, nuclei)
    ratio_spec_source = (
        "locked_default_q95_specs"
        if ratio_specs is None
        else "explicit_api_or_cli_specs"
    )
    resolved_ratio_specs = _resolve_ratio_specs(ratio_specs)
    resolved_cohorts = tuple(dict.fromkeys(str(value) for value in cohorts))
    if not resolved_cohorts:
        raise PostrunValidationError("at least one cohort is required")
    output.mkdir(parents=True, exist_ok=True)

    # Required ordering: persist the authority repair and audit before summaries.
    corrected, sampling_audit = correct_spot_sampling_flags(spots, nuclei)
    corrected_path = output / "corrected_spot_exact_footprint_metrics.csv.gz"
    audit_path = output / "sampling_correction_audit.json"
    quantitation_audit_path = output / "quantitation_invariant_audit.json"
    corrected.to_csv(corrected_path, index=False)
    audit_path.write_text(
        json.dumps(sampling_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    quantitation_audit_path.write_text(
        json.dumps(quantitation_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = build_postrun_statistics(
        nuclei,
        spots,
        pixels,
        endpoints=endpoints,
        endpoint_scales=endpoint_scales,
        cohorts=resolved_cohorts,
        ratio_specs=resolved_ratio_specs,
    )
    if result.sampling_audit != sampling_audit:
        raise PostrunValidationError("sampling correction audit changed during summary construction")
    if result.quantitation_audit != quantitation_audit:
        raise PostrunValidationError(
            "quantitation invariant audit changed during summary construction"
        )
    tables = {
        "nucleus_exact_footprint_endpoints.csv": result.nucleus_endpoints,
        "fov_endpoint_means.csv": result.endpoint_fov_means,
        "biological_set_endpoint_means.csv": result.endpoint_set_means,
        "endpoint_inference.csv": result.endpoint_inference,
        "cartesian_contrasts.csv": result.cartesian_contrasts,
        "fov_correlations.csv": result.fov_correlations,
        "biological_set_correlations.csv": result.set_correlations,
        "correlation_inference.csv": result.correlation_inference,
    }
    if len(result.ratio_of_ratios):
        tables["ratio_of_ratios.csv"] = result.ratio_of_ratios
    for filename, frame in tables.items():
        _write_csv(frame, output / filename)

    written = [
        corrected_path,
        audit_path,
        quantitation_audit_path,
        *(output / name for name in tables),
    ]
    manifest_ratio_specs = [
        {
            "numerator_endpoint": numerator,
            "denominator_endpoint": denominator,
            "endpoint_scale": UNSIGNED_POSITIVE,
            "threshold_scope": "q95_primary" if "q95" in numerator else "global",
        }
        for numerator, denominator in resolved_ratio_specs
    ]
    manifest = {
        "analysis": "MIAT_QKI_exact_footprint_postrun_statistics",
        "source_run": str(source),
        "source_tables": {
            role: {"path": str(path), "sha256": _sha256(path)}
            for role, path in required_paths.items()
        },
        "sampling_metadata_source": "authoritative_nucleus_key_join",
        "sampling_correction_audit": sampling_audit,
        "quantitation_invariant_audit": quantitation_audit,
        "quantitation_plane": (
            "validated from retained canonical tables: singleton reviewed z for all "
            f"{quantitation_audit['n_nucleus_images']} nucleus images; spot/nucleus z "
            f"agreement and locked MIAT/QKI/DAPI indices in all "
            f"{quantitation_audit['n_spot_bearing_images']} spot-bearing images; "
            f"{quantitation_audit['n_images_without_spots']} image(s) lack spot-table "
            "channel rows; no projection contradiction"
        ),
        "hierarchy": "nucleus -> equal-nucleus FOV mean -> equal-FOV biological-set mean",
        "pairing_semantics": "none; NT and KD biological sets are independent",
        "thresholds": {"primary": 95, "sensitivities": [90, 99]},
        "ratio_of_ratios_specs": manifest_ratio_specs,
        "ratio_of_ratios_spec_source": ratio_spec_source,
        "ratio_of_ratios_cohorts": list(resolved_cohorts),
        "written_files": [
            {"path": str(path), "sha256": _sha256(path)} for path in written
        ],
    }
    manifest_path = output / "postrun_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main(argv: Iterable[str] | None = None) -> int:
    """Command-line entry point for a non-overwriting completed-run revision."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    run_postrun_directory(args.source_run, args.output_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the public API
    raise SystemExit(main())
