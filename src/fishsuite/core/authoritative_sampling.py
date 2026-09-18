"""Repair image-aggregated sampling labels with authoritative nucleus keys."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


_TRUE = {"true", "t", "yes", "y", "1"}
_FALSE = {"false", "f", "no", "n", "0"}


def _require_columns(frame: pd.DataFrame, columns: set[str], *, table: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{table} is missing required columns: {missing}")


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and value in (0, 1):
        return bool(int(value))
    text = str(value).strip().casefold()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise ValueError(f"sampling flag is not an explicit boolean: {value!r}")


def _normalise_keys(frame: pd.DataFrame, *, table: str) -> pd.DataFrame:
    result = frame.copy()
    result["_image_key_norm"] = result["image_key"].astype(str).str.strip().str.casefold()
    if result["_image_key_norm"].eq("").any():
        raise ValueError(f"{table} contains an empty image_key")
    numeric = pd.to_numeric(result["nucleus_id"], errors="coerce")
    if numeric.isna().any() or (~np.isfinite(numeric.to_numpy(float))).any():
        raise ValueError(f"{table} contains a nonfinite nucleus_id")
    if not np.equal(numeric.to_numpy(float), np.rint(numeric.to_numpy(float))).all():
        raise ValueError(f"{table} contains a nonintegral nucleus_id")
    if (numeric < 1).any():
        raise ValueError(f"{table} contains nucleus_id below 1")
    result["_nucleus_id_norm"] = numeric.astype(np.int64)
    return result


def correct_spot_sampling_flags(
    spots: pd.DataFrame,
    nuclei: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return a spot table whose cohort flags come from exact nucleus keys.

    The incoming spot-level flags are retained with an ``_image_aggregate``
    suffix for provenance.  Nuclei with no spots remain represented in the
    audit, which prevents a spot-first join from silently shrinking a cohort.
    """

    required = {
        "image_key",
        "nucleus_id",
        "eligible_for_sampling",
        "sampled_in_analysis",
    }
    _require_columns(spots, required, table="spot table")
    _require_columns(nuclei, required, table="nucleus table")
    source_spots = _normalise_keys(spots, table="spot table")
    source_nuclei = _normalise_keys(nuclei, table="nucleus table")
    key_columns = ["_image_key_norm", "_nucleus_id_norm"]
    duplicated = source_nuclei.duplicated(key_columns, keep=False)
    if duplicated.any():
        keys = source_nuclei.loc[duplicated, key_columns].drop_duplicates().to_dict("records")
        raise ValueError(f"nucleus table contains duplicate nucleus keys: {keys[:10]}")

    for frame in (source_spots, source_nuclei):
        for field in ("eligible_for_sampling", "sampled_in_analysis"):
            frame[field] = frame[field].map(_as_bool)

    authoritative_columns = key_columns + [
        "eligible_for_sampling",
        "sampled_in_analysis",
    ]
    if "is_control" in source_nuclei.columns:
        source_nuclei["_authoritative_is_control"] = source_nuclei["is_control"].map(_as_bool)
        authoritative_columns.append("_authoritative_is_control")

    working = source_spots.rename(
        columns={
            "eligible_for_sampling": "eligible_for_sampling_image_aggregate",
            "sampled_in_analysis": "sampled_in_analysis_image_aggregate",
        }
    )
    working["_row_order"] = np.arange(len(working), dtype=np.int64)
    merged = working.merge(
        source_nuclei[authoritative_columns],
        on=key_columns,
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        missing = merged.loc[
            merged["_merge"].ne("both"), key_columns
        ].drop_duplicates().to_dict("records")
        raise ValueError(f"spot table has missing authoritative nucleus keys: {missing[:10]}")

    merged["eligible_for_sampling"] = merged["eligible_for_sampling"].map(_as_bool)
    merged["sampled_in_analysis"] = merged["sampled_in_analysis"].map(_as_bool)
    merged["sampling_metadata_source"] = "authoritative_nucleus_key_join"
    old_sampled = merged["sampled_in_analysis_image_aggregate"].to_numpy(bool)
    new_sampled = merged["sampled_in_analysis"].to_numpy(bool)
    old_eligible = merged["eligible_for_sampling_image_aggregate"].to_numpy(bool)
    new_eligible = merged["eligible_for_sampling"].to_numpy(bool)

    if "_authoritative_is_control" in merged.columns:
        spot_control = merged["_authoritative_is_control"].to_numpy(bool)
    elif "is_control" in merged.columns:
        spot_control = merged["is_control"].map(_as_bool).to_numpy(bool)
    else:
        spot_control = np.zeros(len(merged), dtype=bool)
    if "_authoritative_is_control" in source_nuclei.columns:
        nucleus_control = source_nuclei["_authoritative_is_control"].to_numpy(bool)
    elif "is_control" in source_nuclei.columns:
        nucleus_control = source_nuclei["is_control"].map(_as_bool).to_numpy(bool)
    else:
        nucleus_control = np.zeros(len(source_nuclei), dtype=bool)

    sampled_nuclei = source_nuclei["sampled_in_analysis"].to_numpy(bool)
    spot_keys = set(map(tuple, merged[key_columns].to_numpy()))
    sampled_nucleus_keys = source_nuclei.loc[
        source_nuclei["sampled_in_analysis"], key_columns
    ]
    sampled_zero_spot = sum(
        tuple(key) not in spot_keys for key in sampled_nucleus_keys.to_numpy()
    )
    audit: dict[str, Any] = {
        "correction_status": "pass",
        "sampling_metadata_source": "authoritative_nucleus_key_join",
        "n_spots": int(len(merged)),
        "n_nuclei": int(len(source_nuclei)),
        "n_spot_keys_missing_authority": 0,
        "n_spot_sampling_mismatches_corrected": int(np.count_nonzero(old_sampled != new_sampled)),
        "n_spot_eligibility_mismatches_corrected": int(np.count_nonzero(old_eligible != new_eligible)),
        "n_sampled_spots": int(np.count_nonzero(new_sampled)),
        "n_sampled_biological_spots": int(np.count_nonzero(new_sampled & ~spot_control)),
        "n_sampled_control_spots": int(np.count_nonzero(new_sampled & spot_control)),
        "n_sampled_nuclei": int(np.count_nonzero(sampled_nuclei)),
        "n_sampled_biological_nuclei": int(np.count_nonzero(sampled_nuclei & ~nucleus_control)),
        "n_sampled_control_nuclei": int(np.count_nonzero(sampled_nuclei & nucleus_control)),
        "n_sampled_zero_spot_nuclei": int(sampled_zero_spot),
    }
    corrected = (
        merged.sort_values("_row_order", kind="stable")
        .drop(
            columns=[
                "_image_key_norm",
                "_nucleus_id_norm",
                "_row_order",
                "_merge",
                "_authoritative_is_control",
            ],
            errors="ignore",
        )
        .reset_index(drop=True)
    )
    return corrected, audit

