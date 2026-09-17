"""Build and audit a fresh MIAT--QKI final-publication directory.

This downstream-only orchestrator consumes explicit retained sources.  It never
discovers projects, reselects z, projects image stacks, detects spots, or
segments nuclei.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import textwrap
import time
from typing import Any, Iterable

import h5py
import numpy as np
import pandas as pd
from PIL import Image

import fishsuite.core.exact_footprint_publication_plots as publication_plots_module
from fishsuite.core.exact_footprint_publication_micrographs import (
    PublicationMicrographOutputs,
    render_publication_micrograph_package,
    select_publication_micrograph_examples,
)
from fishsuite.core.exact_footprint_publication_plots import (
    PUBLICATION_FAMILIES,
    PublicationPlotOutputs,
    render_publication_plot_package,
)


@dataclass(frozen=True)
class FinalPublicationInputs:
    nucleus_endpoints: Path
    fov_endpoint_means: Path
    biological_set_endpoint_means: Path
    endpoint_inference: Path
    ratio_of_ratios: Path
    selected_planes_h5: Path
    image_audit: Path
    spot_calls: Path
    nucleus_qc: Path
    footprint_pixels: Path
    postrun_manifest: Path
    backfill_validation: Path
    backfill_parameters: Path
    catalog_input_dir: Path
    catalog_sha256_ledger: Path
    microscopy_methods: Path
    figure_terms: Path

    def __post_init__(self) -> None:
        for field in fields(self):
            object.__setattr__(self, field.name, Path(getattr(self, field.name)))


@dataclass(frozen=True)
class FinalPublicationOutputs:
    output_dir: Path
    manifest_path: Path
    source_hash_manifest_path: Path
    qa_json_path: Path
    qa_markdown_path: Path
    inventory_path: Path


_REQUIRED_TABLE_COLUMNS: dict[str, set[str]] = {
    "nucleus_endpoints": {
        "sampled_in_analysis",
        "include",
        "is_control",
        "secondary_only",
        "arm",
        "slide",
        "replicate",
        "biological_set",
        "image",
        "image_key",
        "fov",
        "nucleus_id",
        "n_spots_all",
        "n_spots_floor",
        "miat_footprint_mass_all_union_deduplicated",
        "miat_footprint_mass_floor_union_deduplicated",
        "association_fraction_among_usable_q95",
        "association_fraction_among_all_floor_spots_q95",
        "threshold_positive_spots_per_nucleus_q95",
        "miat_footprint_mass_q95_positive_union_deduplicated",
    },
    "fov_endpoint_means": {
        "cohort",
        "endpoint",
        "image",
        "image_key",
        "slide",
        "arm",
        "replicate",
        "fov",
        "biological_set",
        "is_control",
        "n_nuclei_total",
        "n_nuclei_finite",
        "value",
    },
    "biological_set_endpoint_means": {
        "cohort",
        "endpoint",
        "slide",
        "arm",
        "replicate",
        "biological_set",
        "n_fovs_total",
        "n_fovs_finite",
        "n_nuclei_total",
        "n_nuclei_finite",
        "value",
        "complete_for_inference",
    },
    "endpoint_inference": {
        "cohort",
        "endpoint",
        "analysis_role",
        "excluded_biological_set",
        "inference_status",
        "n_nt",
        "n_kd",
        "mean_nt",
        "mean_kd",
        "permutation_test",
        "permutation_n_permutations",
        "permutation_p_exact_two_sided",
    },
    "ratio_of_ratios": {
        "cohort",
        "numerator_endpoint",
        "denominator_endpoint",
        "ratio_of_ratios",
        "ratio_of_ratios_ci95_low",
        "ratio_of_ratios_ci95_high",
        "p_two_sided",
    },
    "image_audit": {
        "image_key",
        "image",
        "slide",
        "arm",
        "biological_set",
        "is_control",
        "secondary_only",
        "selected_z_1based",
        "voxel_xy_nm",
        "miat_channel_index",
        "qki_channel_index",
        "dapi_channel_index",
        "plane_lock_pass",
        "mask_qc_pass",
        "footprint_parity_pass",
        "population_reconciliation_pass",
        "image_qc_status",
        "analyzed_vsi_path",
        "source_checksum",
    },
    "spot_calls": {
        "image_key",
        "nucleus_id",
        "spot_uid",
        "null_usable",
        "qki_threshold_positive_q95",
        "stored_in_nucleolus",
        "qki_footprint_enrichment_vs_nucleoplasm",
        "center_y_px",
        "center_x_px",
        "footprint_method",
    },
    "nucleus_qc": {
        "image_key",
        "nucleus_id",
        "nucleus_uid",
        "nucleus_qc_status",
    },
    "footprint_pixels": {"spot_uid", "pixel_index", "y_px", "x_px"},
}

_TABLE_FIELDS = tuple(_REQUIRED_TABLE_COLUMNS)
_FIGURE_ENDPOINTS = tuple(
    panel.endpoint for family in PUBLICATION_FAMILIES for panel in family.panels
)
_RATIO_ALIASES = (
    (
        "threshold_positive_spots_per_nucleus_q95",
        "n_spots_all",
        "n_spots_floor",
    ),
    (
        "miat_footprint_mass_q95_positive_union_deduplicated",
        "miat_footprint_mass_all_union_deduplicated",
        "miat_footprint_mass_floor_union_deduplicated",
    ),
)


def _publication_layout_families() -> tuple[Any, ...]:
    """Wrap Task 1 text without changing its data, inference, or renderer."""

    families = []
    for family in publication_plots_module.PUBLICATION_FAMILIES:
        labels = family.labels
        if family.stem == "02_base_qki_association":
            labels = tuple(
                textwrap.fill(label, width=24, break_long_words=False)
                for label in family.labels
            )
        families.append(
            replace(
                family,
                subtitle=textwrap.fill(
                    family.subtitle, width=78, break_long_words=False
                ),
                labels=labels,
            )
        )
    return tuple(families)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"{name} is not valid UTF-8 JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain a JSON object: {path}")
    return payload


def _explicit_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().casefold()
    if text in {"true", "1", "yes", "pass", "complete"}:
        return True
    if text in {"false", "0", "no", "fail", "failed", ""}:
        return False
    raise ValueError(f"ambiguous boolean value: {value!r}")


def _validate_required_paths(inputs: FinalPublicationInputs) -> None:
    for field in fields(inputs):
        path = Path(getattr(inputs, field.name))
        if field.name == "catalog_input_dir":
            if not path.is_dir():
                raise FileNotFoundError(f"{field.name}: {path}")
        elif not path.is_file():
            raise FileNotFoundError(f"{field.name}: {path}")


def _read_tables(inputs: FinalPublicationInputs) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for name in _TABLE_FIELDS:
        path = Path(getattr(inputs, name))
        if path.suffix.casefold() not in {".csv", ".gz"}:
            raise ValueError(f"{name} must be an explicit CSV or CSV.GZ path: {path}")
        frame = pd.read_csv(path)
        missing = sorted(_REQUIRED_TABLE_COLUMNS[name].difference(frame.columns))
        if missing:
            raise ValueError(f"{name} is missing required columns: {missing}")
        tables[name] = frame
    return tables


def _validate_records(inputs: FinalPublicationInputs) -> None:
    postrun = _json(inputs.postrun_manifest, name="postrun_manifest")
    plane = str(postrun.get("quantitation_plane", "")).casefold()
    if "one reviewed z" not in plane or "no projection" not in plane:
        raise ValueError("postrun_manifest does not prove reviewed single-z/no projection")
    if "none" not in str(postrun.get("pairing_semantics", "")).casefold():
        raise ValueError("postrun_manifest does not record independent NT/KD groups")
    validation = _json(inputs.backfill_validation, name="backfill_validation")
    required_pass = (
        _explicit_bool(validation.get("full_parity_gate_pass")),
        _explicit_bool(validation.get("population_reconciliation_pass")),
        str(validation.get("run_status", "")).casefold() == "complete",
    )
    if not all(required_pass):
        raise ValueError("backfill_validation is not complete with full parity")
    parameters = _json(inputs.backfill_parameters, name="backfill_parameters")
    if int(parameters.get("n_null", -1)) != 1000:
        raise ValueError("backfill_parameters must record exactly 1,000 null placements")
    if int(parameters.get("primary_threshold_percentile", -1)) != 95:
        raise ValueError("backfill_parameters must record q95 as primary")
    geometry = str(parameters.get("placement_geometry", "")).casefold()
    if "exact_footprint" not in geometry:
        raise ValueError("backfill_parameters do not record exact-footprint placement")


def _validate_catalog(
    image_audit: pd.DataFrame, catalog_input: Path, catalog_ledger: Path
) -> dict[str, object]:
    root = catalog_input.resolve()
    ledger = pd.read_csv(catalog_ledger)
    required = {"source", "destination", "status", "bytes", "sha256"}
    missing = sorted(required.difference(ledger.columns))
    if missing:
        raise ValueError(f"catalog SHA-256 ledger is missing columns: {missing}")
    if ledger.empty:
        raise ValueError("catalog SHA-256 ledger is empty")
    destinations = ledger["destination"].astype(str).map(
        lambda value: str(Path(value).resolve())
    )
    if destinations.duplicated().any():
        raise ValueError("catalog SHA-256 ledger contains duplicate destinations")

    started = time.perf_counter()
    file_records: list[dict[str, object]] = []
    by_path: dict[Path, dict[str, object]] = {}
    digest_records: list[str] = []
    for row in ledger.to_dict("records"):
        path = Path(str(row["destination"])).resolve()
        try:
            relative = path.relative_to(root)
        except ValueError as error:
            raise ValueError(
                f"catalog ledger destination escapes explicit catalog input: {path}"
            ) from error
        recorded_sha256 = str(row["sha256"]).strip().casefold()
        if len(recorded_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in recorded_sha256
        ):
            raise ValueError(f"invalid SHA-256 in catalog ledger: {relative}")
        try:
            recorded_size = int(row["bytes"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid byte count in catalog ledger: {relative}") from error
        exists = path.is_file()
        actual_size = path.stat().st_size if exists else None
        computed_sha256 = _sha256_file(path) if exists else None
        passed = (
            exists
            and actual_size == recorded_size
            and computed_sha256 == recorded_sha256
        )
        record = {
            "path": str(path),
            "relative_path": relative.as_posix(),
            "size_bytes": actual_size,
            "recorded_size_bytes": recorded_size,
            "recorded_sha256": recorded_sha256,
            "computed_sha256": computed_sha256,
            "status": "pass" if passed else "fail",
        }
        file_records.append(record)
        by_path[path] = record
        digest_records.append(
            f"{relative.as_posix()}\t{recorded_size}\t{recorded_sha256}"
        )

    audit_link_failures: list[str] = []
    for row in image_audit.to_dict("records"):
        path = Path(str(row["analyzed_vsi_path"])).resolve()
        try:
            relative = path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"catalog source escapes explicit catalog input: {path}") from error
        checksum = str(row["source_checksum"]).strip().casefold()
        if len(checksum) != 64 or any(character not in "0123456789abcdef" for character in checksum):
            raise ValueError(f"invalid source checksum for catalog image: {relative}")
        ledger_record = by_path.get(path)
        if (
            ledger_record is None
            or ledger_record["status"] != "pass"
            or checksum != ledger_record["recorded_sha256"]
        ):
            audit_link_failures.append(relative.as_posix())

    failed = [record for record in file_records if record["status"] != "pass"]
    duration_seconds = time.perf_counter() - started
    aggregate = {
        "files_listed": len(file_records),
        "files_passed": len(file_records) - len(failed),
        "files_failed": len(failed),
        "bytes_listed": int(
            sum(int(record["recorded_size_bytes"]) for record in file_records)
        ),
        "bytes_hashed": int(
            sum(
                int(record["size_bytes"])
                for record in file_records
                if record["size_bytes"] is not None
            )
        ),
        "audit_image_records": len(image_audit),
        "audit_image_records_linked": len(image_audit) - len(audit_link_failures),
        "duration_seconds": duration_seconds,
        "verdict": "pass" if not failed and not audit_link_failures else "fail",
    }
    result = {
        "ledger_path": str(catalog_ledger.resolve()),
        "ledger_sha256": _sha256_file(catalog_ledger),
        "ledger_record_digest_sha256": hashlib.sha256(
            "\n".join(sorted(digest_records)).encode("utf-8")
        ).hexdigest(),
        "aggregate": aggregate,
        "files": file_records,
        "audit_link_failures": sorted(audit_link_failures),
    }
    if aggregate["verdict"] != "pass":
        raise ValueError(
            "catalog byte integrity failed: "
            f"{len(failed)} ledger file mismatch(es), "
            f"{len(audit_link_failures)} audit-link mismatch(es)"
        )
    return result


def _validate_h5_and_plane_parity(
    selected_planes_h5: Path, image_audit: pd.DataFrame
) -> pd.DataFrame:
    audit = image_audit.copy()
    audit["image_key"] = audit["image_key"].astype(str).str.casefold()
    if audit["image_key"].duplicated().any():
        raise ValueError("image_audit must contain one row per image_key")
    expected_channels = {"miat": 0, "qki": 1, "dapi": 2}
    rows: list[dict[str, object]] = []
    with h5py.File(selected_planes_h5, "r") as handle:
        if "images" not in handle:
            raise ValueError("selected_planes_h5 lacks the images group")
        keys = {str(key).casefold() for key in handle["images"].keys()}
        audit_keys = set(audit["image_key"])
        if keys != audit_keys:
            raise ValueError("selected_planes_h5 image keys do not equal image_audit")
        for record in audit.to_dict("records"):
            key = str(record["image_key"])
            group = handle["images"][key]
            if not _explicit_bool(group.attrs.get("complete", False)):
                raise ValueError(f"selected plane is incomplete: {key}")
            if not _explicit_bool(group.attrs.get("same_plane_all_channels", False)):
                raise ValueError(f"same-plane channel lock failed: {key}")
            z_audit = int(record["selected_z_1based"])
            z_h5 = int(group.attrs["selected_z_1based"])
            if z_audit != z_h5:
                raise ValueError(f"selected-z mismatch for {key}: {z_audit} != {z_h5}")
            for role, expected in expected_channels.items():
                audit_index = int(record[f"{role}_channel_index"])
                h5_index = int(group.attrs[f"{role}_channel_index"])
                if audit_index != expected or h5_index != expected:
                    raise ValueError(f"channel index mismatch for {key}/{role}")
            required_datasets = {
                "miat",
                "qki",
                "dapi",
                "nucleus_labels",
                "nucleolus_labels",
            }
            if not required_datasets.issubset(group.keys()):
                raise ValueError(f"selected plane datasets are incomplete: {key}")
            rows.append(
                {
                    "image_key": key,
                    "selected_z_1based_audit": z_audit,
                    "selected_z_1based_h5": z_h5,
                    "same_z_all_channels": True,
                    "projection_used": False,
                    "miat_channel_index": 0,
                    "qki_channel_index": 1,
                    "dapi_channel_index": 2,
                    "parity_verdict": "pass",
                }
            )
    return pd.DataFrame(rows)


def _validate_statistical_inputs(tables: dict[str, pd.DataFrame]) -> None:
    sets = tables["biological_set_endpoint_means"]
    complete = sets["complete_for_inference"].map(_explicit_bool)
    counts = (
        sets.loc[
            sets["cohort"].astype(str).eq("sampled_primary")
            & sets["endpoint"].astype(str).isin(_FIGURE_ENDPOINTS)
            & sets["arm"].astype(str).isin(("NT", "KD"))
            & complete
        ]
        .groupby(["endpoint", "arm"])["biological_set"]
        .nunique()
    )
    expected = pd.MultiIndex.from_product([_FIGURE_ENDPOINTS, ("NT", "KD")])
    counts = counts.reindex(expected, fill_value=0)
    if not counts.eq(6).all():
        raise ValueError("publication statistics require six complete sets per arm")
    inference = tables["endpoint_inference"]
    primary = inference.loc[
        inference["cohort"].astype(str).eq("sampled_primary")
        & inference["endpoint"].astype(str).isin(_FIGURE_ENDPOINTS)
        & inference["analysis_role"].astype(str).eq("primary")
        & inference["excluded_biological_set"].isna()
    ]
    if len(primary) != len(_FIGURE_ENDPOINTS):
        raise ValueError("endpoint_inference lacks one retained primary row per endpoint")
    if not (
        pd.to_numeric(primary["n_nt"], errors="coerce").eq(6)
        & pd.to_numeric(primary["n_kd"], errors="coerce").eq(6)
        & primary["permutation_test"].astype(str).eq(
            "exact_within_slide_label_permutation"
        )
    ).all():
        raise ValueError("primary inference is not six-by-six exact within-slide")
    if not pd.to_numeric(
        primary["permutation_n_permutations"], errors="coerce"
    ).eq(400).all():
        raise ValueError(
            "retained primary inference must contain exactly 400 within-slide "
            "label permutations for every displayed endpoint"
        )

    footprint_methods = (
        tables["spot_calls"]["footprint_method"]
        .astype(str)
        .str.strip()
        .str.casefold()
    )
    disk_derived = footprint_methods.str.contains("disk", regex=False, na=False)
    if disk_derived.any():
        methods = sorted(set(footprint_methods.loc[disk_derived]))
        raise ValueError(
            "disk-derived footprint methods cannot enter final exact-footprint "
            f"source data: {methods}"
        )


def _endpoint_values_equal(sets: pd.DataFrame, left: str, right: str) -> bool:
    keys = ["slide", "arm", "replicate", "biological_set"]
    frame = sets.loc[
        sets["cohort"].astype(str).eq("sampled_primary")
        & sets["endpoint"].astype(str).isin((left, right)),
        [*keys, "endpoint", "value"],
    ]
    pivot = frame.pivot(index=keys, columns="endpoint", values="value")
    if left not in pivot or right not in pivot or len(pivot) != 12:
        return False
    a = pd.to_numeric(pivot[left], errors="coerce").to_numpy(float)
    b = pd.to_numeric(pivot[right], errors="coerce").to_numpy(float)
    return bool(np.array_equal(a, b, equal_nan=False))


def _ratio_input_for_renderer(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    ratios = tables["ratio_of_ratios"].copy()
    additions: list[pd.DataFrame] = []
    for numerator, renderer_denominator, retained_denominator in _RATIO_ALIASES:
        direct = ratios.loc[
            ratios["cohort"].astype(str).eq("sampled_primary")
            & ratios["numerator_endpoint"].astype(str).eq(numerator)
            & ratios["denominator_endpoint"].astype(str).eq(renderer_denominator)
        ]
        if len(direct) == 1:
            continue
        if len(direct) != 0:
            raise ValueError(f"ambiguous ratio-of-ratios row for {numerator}")
        retained = ratios.loc[
            ratios["cohort"].astype(str).eq("sampled_primary")
            & ratios["numerator_endpoint"].astype(str).eq(numerator)
            & ratios["denominator_endpoint"].astype(str).eq(retained_denominator)
        ].copy()
        if len(retained) != 1:
            raise ValueError(f"missing retained ratio-of-ratios row for {numerator}")
        if not _endpoint_values_equal(
            tables["biological_set_endpoint_means"],
            renderer_denominator,
            retained_denominator,
        ):
            raise ValueError(
                f"cannot alias {retained_denominator} to {renderer_denominator}; "
                "retained set values differ"
            )
        retained["retained_denominator_endpoint"] = retained_denominator
        retained["denominator_endpoint"] = renderer_denominator
        retained["denominator_equivalence_verified"] = True
        retained["denominator_equivalence_unit"] = "12 retained biological-set values"
        additions.append(retained)
    if additions:
        ratios = pd.concat([ratios, *additions], ignore_index=True, sort=False)
    return ratios


def _source_metadata(
    inputs: FinalPublicationInputs,
    tables: dict[str, pd.DataFrame],
    *,
    catalog_integrity: dict[str, object],
) -> dict[str, dict[str, object]]:
    sources: dict[str, dict[str, object]] = {}
    for field in fields(inputs):
        name = field.name
        path = Path(getattr(inputs, name)).resolve()
        if name == "catalog_input_dir":
            sources[name] = {
                "path": str(path),
                "kind": "explicit_catalog_directory",
                "catalog_record_digest_sha256": catalog_integrity[
                    "ledger_record_digest_sha256"
                ],
                "image_records": len(tables["image_audit"]),
                "ledger_files": catalog_integrity["aggregate"]["files_listed"],
                "ledger_bytes": catalog_integrity["aggregate"]["bytes_listed"],
            }
            continue
        record: dict[str, object] = {
            "path": str(path),
            "kind": "explicit_file",
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        if name in tables:
            record["rows"] = len(tables[name])
            record["columns"] = len(tables[name].columns)
        sources[name] = record
    return sources


def _copy_verified(source: Path, destination: Path) -> dict[str, object]:
    shutil.copy2(source, destination)
    source_hash = _sha256_file(source)
    copied_hash = _sha256_file(destination)
    if source_hash != copied_hash:
        raise RuntimeError(f"copied file hash mismatch: {destination}")
    return {
        "source_path": str(source.resolve()),
        "copied_path": str(destination.resolve()),
        "source_sha256": source_hash,
        "copied_sha256": copied_hash,
    }


def _relative(output: Path, paths: Iterable[Path]) -> list[str]:
    return [path.resolve().relative_to(output.resolve()).as_posix() for path in paths]


_CSV_FLOAT_ATOL = 1e-12
_CSV_FLOAT_RTOL = 1e-12
_COMPARISON_SEMANTICS = (
    "exact strings/integers/booleans; floats within declared CSV-roundtrip tolerance"
)


def _json_scalar(value: object) -> object:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _compare_column_value(
    source: pd.DataFrame,
    source_index: object,
    source_column: str,
    generated: pd.DataFrame,
    generated_index: object,
    generated_column: str,
) -> dict[str, object]:
    source_value = source.at[source_index, source_column]
    generated_value = generated.at[generated_index, generated_column]
    source_missing = bool(pd.isna(source_value))
    generated_missing = bool(pd.isna(generated_value))
    base: dict[str, object] = {
        "source_column": source_column,
        "generated_column": generated_column,
        "source_value": _json_scalar(source_value),
        "generated_value": _json_scalar(generated_value),
    }
    if source_missing or generated_missing:
        passed = source_missing and generated_missing
        return {
            **base,
            "comparison": "exact_missing",
            "exactly_equal": passed,
            "within_tolerance": None,
            "absolute_difference": None,
            "relative_difference": None,
            "verdict": "pass" if passed else "fail",
        }

    source_dtype = source[source_column].dtype
    if pd.api.types.is_bool_dtype(source_dtype) or isinstance(
        source_value, (bool, np.bool_)
    ):
        passed = isinstance(generated_value, (bool, np.bool_)) and bool(
            source_value
        ) == bool(generated_value)
        return {
            **base,
            "comparison": "exact",
            "value_type": "boolean",
            "exactly_equal": passed,
            "within_tolerance": None,
            "absolute_difference": None,
            "relative_difference": None,
            "verdict": "pass" if passed else "fail",
        }
    if pd.api.types.is_integer_dtype(source_dtype) or isinstance(
        source_value, (int, np.integer)
    ):
        generated_is_integer = isinstance(generated_value, (int, np.integer)) or (
            isinstance(generated_value, (float, np.floating))
            and float(generated_value).is_integer()
        )
        passed = generated_is_integer and int(source_value) == int(generated_value)
        return {
            **base,
            "comparison": "exact",
            "value_type": "integer",
            "exactly_equal": passed,
            "within_tolerance": None,
            "absolute_difference": None,
            "relative_difference": None,
            "verdict": "pass" if passed else "fail",
        }
    if pd.api.types.is_numeric_dtype(source_dtype) or isinstance(
        source_value, (float, np.floating)
    ):
        try:
            source_float = float(source_value)
            generated_float = float(generated_value)
        except (TypeError, ValueError):
            passed = False
            absolute_difference = None
            relative_difference = None
            exactly_equal = False
        else:
            absolute_difference = abs(source_float - generated_float)
            scale = max(abs(source_float), abs(generated_float))
            relative_difference = absolute_difference / scale if scale else 0.0
            exactly_equal = source_float == generated_float
            passed = bool(
                np.isclose(
                    source_float,
                    generated_float,
                    atol=_CSV_FLOAT_ATOL,
                    rtol=_CSV_FLOAT_RTOL,
                    equal_nan=False,
                )
            )
        return {
            **base,
            "comparison": "float_tolerance",
            "value_type": "float",
            "exactly_equal": exactly_equal,
            "within_tolerance": passed,
            "absolute_difference": absolute_difference,
            "relative_difference": relative_difference,
            "verdict": "pass" if passed else "fail",
        }
    passed = type(source_value) is type(generated_value) and source_value == generated_value
    return {
        **base,
        "comparison": "exact",
        "value_type": "string_or_other",
        "exactly_equal": passed,
        "within_tolerance": None,
        "absolute_difference": None,
        "relative_difference": None,
        "verdict": "pass" if passed else "fail",
    }


def _comparison_summary(
    column_comparisons: list[dict[str, object]],
) -> dict[str, object]:
    float_nonzero = [
        item
        for item in column_comparisons
        if item["comparison"] == "float_tolerance"
        and item["absolute_difference"] is not None
        and float(item["absolute_difference"]) > 0
    ]
    absolute = [float(item["absolute_difference"]) for item in float_nonzero]
    relative = [float(item["relative_difference"]) for item in float_nonzero]
    passed = all(item["verdict"] == "pass" for item in column_comparisons)
    return {
        "comparison_semantics": _COMPARISON_SEMANTICS,
        "float_tolerance": {"atol": _CSV_FLOAT_ATOL, "rtol": _CSV_FLOAT_RTOL},
        "nonzero_float_differences": len(float_nonzero),
        "max_absolute_difference": max(absolute, default=0.0),
        "max_relative_difference": max(relative, default=0.0),
        "verdict": "pass_within_tolerance" if passed else "fail",
    }


def _compare_statistics(
    retained: pd.DataFrame, generated_path: Path
) -> list[dict[str, object]]:
    generated = pd.read_csv(generated_path)
    retained_primary = retained.loc[
        retained["cohort"].astype(str).eq("sampled_primary")
        & retained["endpoint"].astype(str).isin(_FIGURE_ENDPOINTS)
        & retained["analysis_role"].astype(str).eq("primary")
        & retained["excluded_biological_set"].isna()
    ].copy()
    expected_added = {
        "primary_p_value_column": "permutation_p_exact_two_sided",
        "inference_unit": "biological_set_mean",
        "paired_across_arms": False,
    }
    comparisons: list[dict[str, object]] = []
    for endpoint in _FIGURE_ENDPOINTS:
        source_rows = retained_primary.loc[
            retained_primary["endpoint"].astype(str).eq(endpoint)
        ]
        target_rows = generated.loc[generated["endpoint"].astype(str).eq(endpoint)]
        if len(source_rows) != 1 or len(target_rows) != 1:
            comparisons.append(
                {
                    "endpoint": endpoint,
                    "compared_columns": list(retained_primary.columns),
                    "column_comparisons": [],
                    "row_count_source": len(source_rows),
                    "row_count_generated": len(target_rows),
                    "comparison_semantics": _COMPARISON_SEMANTICS,
                    "float_tolerance": {
                        "atol": _CSV_FLOAT_ATOL,
                        "rtol": _CSV_FLOAT_RTOL,
                    },
                    "nonzero_float_differences": 0,
                    "max_absolute_difference": 0.0,
                    "max_relative_difference": 0.0,
                    "verdict": "fail",
                }
            )
            continue
        source_index = source_rows.index[0]
        target_index = target_rows.index[0]
        missing_columns = sorted(set(retained_primary.columns).difference(generated.columns))
        unexpected_columns = sorted(
            set(generated.columns).difference(retained_primary.columns).difference(expected_added)
        )
        column_comparisons = [
            _compare_column_value(
                retained_primary,
                source_index,
                column,
                generated,
                target_index,
                column,
            )
            for column in retained_primary.columns
            if column in generated.columns
        ]
        derived_checks = []
        for column, expected in expected_added.items():
            actual = generated.at[target_index, column] if column in generated else None
            passed = actual == expected
            derived_checks.append(
                {
                    "generated_column": column,
                    "expected_value": expected,
                    "generated_value": _json_scalar(actual),
                    "comparison": "exact_derived",
                    "verdict": "pass" if passed else "fail",
                }
            )
        summary = _comparison_summary(column_comparisons)
        if missing_columns or unexpected_columns or not all(
            item["verdict"] == "pass" for item in derived_checks
        ):
            summary["verdict"] = "fail"
        comparisons.append(
            {
                "endpoint": endpoint,
                "compared_columns": list(retained_primary.columns),
                "column_comparisons": column_comparisons,
                "derived_column_checks": derived_checks,
                "missing_canonical_columns": missing_columns,
                "unexpected_generated_columns": unexpected_columns,
                **summary,
            }
        )
    return comparisons


def _compare_ratios(
    retained: pd.DataFrame, generated_path: Path
) -> list[dict[str, object]]:
    generated = pd.read_csv(generated_path)
    comparisons: list[dict[str, object]] = []
    for numerator, renderer_denominator, retained_denominator in _RATIO_ALIASES:
        source = retained.loc[
            retained["cohort"].astype(str).eq("sampled_primary")
            & retained["numerator_endpoint"].astype(str).eq(numerator)
            & retained["denominator_endpoint"].astype(str).isin(
                (renderer_denominator, retained_denominator)
            )
        ]
        target = generated.loc[
            generated["numerator_endpoint"].astype(str).eq(numerator)
            & generated["denominator_endpoint"].astype(str).eq(renderer_denominator)
        ]
        if len(source) != 1 or len(target) != 1:
            comparisons.append(
                {
                    "numerator_endpoint": numerator,
                    "retained_denominator_endpoint": None,
                    "renderer_denominator_endpoint": renderer_denominator,
                    "compared_source_columns": list(retained.columns),
                    "column_comparisons": [],
                    "row_count_source": len(source),
                    "row_count_generated": len(target),
                    "comparison_semantics": _COMPARISON_SEMANTICS,
                    "float_tolerance": {
                        "atol": _CSV_FLOAT_ATOL,
                        "rtol": _CSV_FLOAT_RTOL,
                    },
                    "nonzero_float_differences": 0,
                    "max_absolute_difference": 0.0,
                    "max_relative_difference": 0.0,
                    "verdict": "fail",
                }
            )
            continue
        source_index = source.index[0]
        target_index = target.index[0]
        canonical_denominator = str(source.at[source_index, "denominator_endpoint"])
        column_map = {
            column: (
                "retained_denominator_endpoint"
                if column == "denominator_endpoint"
                and canonical_denominator != renderer_denominator
                else column
            )
            for column in retained.columns
        }
        missing_columns = sorted(
            source_column
            for source_column, generated_column in column_map.items()
            if generated_column not in generated.columns
        )
        column_comparisons = [
            _compare_column_value(
                retained,
                source_index,
                source_column,
                generated,
                target_index,
                generated_column,
            )
            for source_column, generated_column in column_map.items()
            if generated_column in generated.columns
        ]
        expected_added = {
            "denominator_endpoint": renderer_denominator,
            "denominator_equivalence_verified": True,
            "denominator_equivalence_unit": "12 retained biological-set values",
        }
        derived_checks = []
        if canonical_denominator != renderer_denominator:
            for column, expected in expected_added.items():
                actual = generated.at[target_index, column] if column in generated else None
                passed = actual == expected
                derived_checks.append(
                    {
                        "generated_column": column,
                        "expected_value": expected,
                        "generated_value": _json_scalar(actual),
                        "comparison": "exact_derived",
                        "verdict": "pass" if passed else "fail",
                    }
                )
        expected_generated = set(column_map.values()).union(
            expected_added if canonical_denominator != renderer_denominator else ()
        )
        unexpected_columns = sorted(set(generated.columns).difference(expected_generated))
        summary = _comparison_summary(column_comparisons)
        if missing_columns or unexpected_columns or not all(
            item["verdict"] == "pass" for item in derived_checks
        ):
            summary["verdict"] = "fail"
        comparisons.append(
            {
                "numerator_endpoint": numerator,
                "retained_denominator_endpoint": canonical_denominator,
                "renderer_denominator_endpoint": renderer_denominator,
                "compared_source_columns": list(retained.columns),
                "column_comparisons": column_comparisons,
                "derived_column_checks": derived_checks,
                "missing_canonical_columns": missing_columns,
                "unexpected_generated_columns": unexpected_columns,
                **summary,
            }
        )
    return comparisons


def _png_checks(paths: Iterable[Path]) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    for path in paths:
        with Image.open(path) as image:
            dpi = image.info.get("dpi", (0.0, 0.0))
            pass_dpi = all(abs(float(value) - 600.0) <= 0.01 for value in dpi[:2])
            checks.append(
                {
                    "path": str(path.resolve()),
                    "width_px": image.width,
                    "height_px": image.height,
                    "dpi": [float(value) for value in dpi[:2]],
                    "verdict": "pass" if pass_dpi else "fail",
                }
            )
    return checks


def _statistical_edge_checks(paths: Iterable[Path]) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    for path in paths:
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB"))
        border = np.concatenate(
            (
                pixels[:5].reshape(-1, 3),
                pixels[-5:].reshape(-1, 3),
                pixels[:, :5].reshape(-1, 3),
                pixels[:, -5:].reshape(-1, 3),
            )
        )
        clean = int(border.min()) >= 250
        checks.append(
            {
                "path": str(path.resolve()),
                "five_pixel_canvas_border_min": int(border.min()),
                "verdict": "pass" if clean else "fail",
            }
        )
    return checks


def _svg_checks(paths: Iterable[Path], hidden_labels: set[str]) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        leaked = sorted(label for label in hidden_labels if label and label in text)
        editable = "<text" in text
        checks.append(
            {
                "path": str(path.resolve()),
                "editable_text": editable,
                "hidden_labels_found": leaked,
                "verdict": "pass" if editable and not leaked else "fail",
            }
        )
    return checks


def _file_checks(output: Path, planned: Iterable[Path]) -> list[dict[str, object]]:
    paths = set(path for path in output.rglob("*") if path.is_file())
    paths.update(planned)
    checks: list[dict[str, object]] = []
    for path in sorted(paths, key=lambda item: str(item).casefold()):
        record: dict[str, object] = {
            "path": str(path.resolve()),
            "verdict": "pass",
        }
        if path.is_file():
            record.update({"bytes": path.stat().st_size, "sha256": _sha256_file(path)})
        else:
            record.update({"bytes": None, "sha256": None, "planned_qa_output": True})
        checks.append(record)
    return checks


def _qa_markdown(payload: dict[str, Any]) -> str:
    catalog = payload["catalog_byte_integrity"]["aggregate"]
    lines = [
        "# Final-publication QA",
        "",
        f"Automated status: **{payload['automated_status']}**",
        f"Visual status: **{payload['visual_status']}**",
        "",
        "Retained-value comparison policy: exact strings, integers, and booleans; "
        "floating-point values within the explicitly recorded CSV-roundtrip tolerance.",
        "",
        f"Catalog byte integrity: **{catalog['verdict']}**; "
        f"{catalog['files_passed']}/{catalog['files_listed']} files; "
        f"{catalog['bytes_hashed']} bytes streamed in "
        f"{catalog['duration_seconds']:.3f} seconds.",
        "",
        "## Automated checks",
        "",
        "| Check | Verdict |",
        "|---|---|",
    ]
    for check in payload["checks"]:
        lines.append(f"| {check['name']} | {check['verdict']} |")
    lines.extend(
        [
            "",
            "## File-by-file structural verdicts",
            "",
            "| File | Verdict |",
            "|---|---|",
        ]
    )
    for item in payload["file_checks"]:
        lines.append(f"| `{item['path']}` | {item['verdict']} |")
    lines.extend(
        [
            "",
            "## Visual inspections",
            "",
            "| File | Verdict | Evidence |",
            "|---|---|---|",
        ]
    )
    if payload["visual_inspections"]:
        for item in payload["visual_inspections"]:
            evidence = str(item["evidence"]).replace("|", "\\|")
            lines.append(
                f"| `{item['path']}` | {item['verdict']} | {evidence} |"
            )
    else:
        lines.append("| Pending | pending | Contact sheets/statistical PNGs not yet inspected |")
    return "\n".join(lines) + "\n"


def _write_output_inventory(output: Path, inventory_path: Path) -> None:
    rows = []
    for path in sorted(output.rglob("*"), key=lambda item: str(item).casefold()):
        if not path.is_file() or path == inventory_path:
            continue
        rows.append(
            {
                "relative_path": path.relative_to(output).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    pd.DataFrame(rows).to_csv(inventory_path, index=False)


def mark_publication_superseded(
    output_dir: str | Path,
    *,
    status: str,
    reason: str,
    replacement_path: str | Path | None = None,
) -> Path:
    """Mark a retained publication root unusable without deleting its contents."""

    output = Path(output_dir).resolve()
    if not output.is_dir():
        raise FileNotFoundError(f"publication root is missing: {output}")
    clean_status = str(status).strip()
    clean_reason = str(reason).strip()
    if not clean_status or not clean_reason:
        raise ValueError("supersession status and reason must be non-empty")
    readme = output / "README.md"
    if not readme.is_file():
        raise FileNotFoundError(f"publication README is missing: {readme}")
    retained_readme = output / "README_RETAINED_BEFORE_SUPERSESSION.md"
    if not retained_readme.exists():
        shutil.copy2(readme, retained_readme)
    replacement = (
        str(Path(replacement_path).resolve())
        if replacement_path is not None
        else "replacement not yet available"
    )
    notice = (
        "SUPERSEDED — DO NOT USE\n\n"
        f"Status: `{clean_status}`\n\n"
        f"Reason: {clean_reason}\n\n"
        f"Replacement: `{replacement}`\n\n"
        "All figures, data, manifests, and prior QA records are retained for audit. "
        "The original README is preserved verbatim in "
        "`README_RETAINED_BEFORE_SUPERSESSION.md`.\n"
    )
    marker = output / "SUPERSEDED_DO_NOT_USE.md"
    marker.write_text("# " + notice, encoding="utf-8")
    readme.write_text(notice, encoding="utf-8")
    inventory = output / "QA" / "output_inventory.csv"
    if inventory.is_file():
        _write_output_inventory(output, inventory)
    return marker


def _write_automated_qa(
    output: Path,
    plot_outputs: PublicationPlotOutputs,
    micro_outputs: PublicationMicrographOutputs,
    tables: dict[str, pd.DataFrame],
    plane_parity: pd.DataFrame,
    catalog_integrity: dict[str, object],
    qa_json_path: Path,
    qa_markdown_path: Path,
    inventory_path: Path,
) -> None:
    micro_manifest = _json(micro_outputs.manifest_path, name="micrograph manifest")
    examples = pd.DataFrame(micro_manifest["selection_manifest"])
    pngs = sorted(output.rglob("*.png"))
    svgs = sorted(output.rglob("*.svg"))
    hidden_labels = set(examples["biological_set"].astype(str))
    hidden_labels.update(f"Slide {value}" for value in examples["slide"].astype(str))
    png_checks = _png_checks(pngs)
    statistical_edge_checks = _statistical_edge_checks(
        sorted((output / "STATISTICS").glob("*.png"))
    )
    svg_checks = _svg_checks(svgs, hidden_labels)
    statistic_checks = _compare_statistics(
        tables["endpoint_inference"], plot_outputs.master_statistics_path
    )
    ratio_checks = _compare_ratios(
        tables["ratio_of_ratios"], plot_outputs.ratio_of_ratios_statistics_path
    )
    no_pdf = not any(output.rglob("*.pdf"))
    six_examples = (
        len(examples) == 6
        and examples["image_key"].nunique() == 6
        and examples.groupby("condition").size().to_dict() == {"MIAT-KD": 3, "NT": 3}
        and examples.groupby("condition")["biological_set"].nunique().eq(3).all()
    )
    checks = [
        {"name": "no_pdf_files", "verdict": "pass" if no_pdf else "fail"},
        {
            "name": "all_png_exactly_600_dpi",
            "verdict": "pass"
            if png_checks and all(item["verdict"] == "pass" for item in png_checks)
            else "fail",
        },
        {
            "name": "all_svg_text_editable_and_labels_hidden",
            "verdict": "pass"
            if svg_checks and all(item["verdict"] == "pass" for item in svg_checks)
            else "fail",
        },
        {
            "name": "statistical_text_inside_canvas",
            "verdict": "pass"
            if statistical_edge_checks
            and all(item["verdict"] == "pass" for item in statistical_edge_checks)
            else "fail",
        },
        {"name": "six_distinct_representatives", "verdict": "pass" if six_examples else "fail"},
        {
            "name": "exact_z_and_channel_parity",
            "verdict": "pass"
            if len(plane_parity) == len(tables["image_audit"])
            and plane_parity["parity_verdict"].eq("pass").all()
            else "fail",
        },
        {
            "name": "retained_primary_statistics_within_declared_csv_roundtrip_tolerance",
            "verdict": "pass"
            if all(str(item["verdict"]).startswith("pass") for item in statistic_checks)
            else "fail",
        },
        {
            "name": "retained_ratio_of_ratios_within_declared_csv_roundtrip_tolerance",
            "verdict": "pass"
            if all(str(item["verdict"]).startswith("pass") for item in ratio_checks)
            else "fail",
        },
        {
            "name": "catalog_ledger_byte_integrity",
            "verdict": str(catalog_integrity["aggregate"]["verdict"]),
        },
    ]
    payload: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "automated_status": "pass"
        if all(check["verdict"] == "pass" for check in checks)
        else "fail",
        "visual_status": "pending",
        "checks": checks,
        "row_counts": {name: len(table) for name, table in tables.items()},
        "png_checks": png_checks,
        "statistical_edge_checks": statistical_edge_checks,
        "svg_checks": svg_checks,
        "statistic_comparisons": statistic_checks,
        "ratio_of_ratios_comparisons": ratio_checks,
        "catalog_byte_integrity": catalog_integrity,
        "plane_channel_parity_rows": len(plane_parity),
        "visual_required_paths": [
            str(path.resolve())
            for path in (
                *sorted((output / "STATISTICS").glob("*.png")),
                *sorted((output / "MICROGRAPHS" / "contact_sheets").glob("*.png")),
            )
        ],
        "visual_inspections": [],
    }
    payload["file_checks"] = _file_checks(
        output, (qa_json_path, qa_markdown_path, inventory_path)
    )
    qa_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    qa_markdown_path.write_text(_qa_markdown(payload), encoding="utf-8")
    _write_output_inventory(output, inventory_path)
    if payload["automated_status"] != "pass":
        raise RuntimeError(f"automated final-publication QA failed: {qa_json_path}")


def generate_final_publication(
    inputs: FinalPublicationInputs,
    output_dir: str | Path,
    *,
    png_dpi: int = 600,
) -> FinalPublicationOutputs:
    """Generate one non-overwriting final-publication tree from explicit inputs."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    if not output.name.startswith("FINAL_PUBLICATION_"):
        raise ValueError("output directory name must start with FINAL_PUBLICATION_")
    if int(png_dpi) != 600:
        raise ValueError("final-publication PNG output must be exactly 600 dpi")
    _validate_required_paths(inputs)
    _validate_records(inputs)
    tables = _read_tables(inputs)
    _validate_statistical_inputs(tables)
    catalog_integrity = _validate_catalog(
        tables["image_audit"],
        inputs.catalog_input_dir,
        inputs.catalog_sha256_ledger,
    )
    plane_parity = _validate_h5_and_plane_parity(
        inputs.selected_planes_h5, tables["image_audit"]
    )
    biological_image_audit = tables["image_audit"].loc[
        tables["image_audit"]["arm"]
        .astype(str)
        .str.casefold()
        .isin(("nt", "kd", "miat-kd"))
        & ~tables["image_audit"]["is_control"].map(_explicit_bool)
        & ~tables["image_audit"]["secondary_only"].map(_explicit_bool)
    ].copy()
    selection = select_publication_micrograph_examples(
        biological_image_audit, tables["spot_calls"], tables["nucleus_qc"]
    )
    if len(selection.manifest) != 6:
        raise ValueError("micrograph selection did not yield exactly six examples")
    ratio_for_renderer = _ratio_input_for_renderer(tables)
    source_metadata = _source_metadata(
        inputs, tables, catalog_integrity=catalog_integrity
    )

    output.mkdir(parents=True, exist_ok=False)
    try:
        import matplotlib as mpl

        with mpl.rc_context({"svg.fonttype": "none"}):
            original_families = publication_plots_module.PUBLICATION_FAMILIES
            publication_plots_module.PUBLICATION_FAMILIES = (
                _publication_layout_families()
            )
            try:
                plot_outputs = render_publication_plot_package(
                    inputs.nucleus_endpoints,
                    inputs.fov_endpoint_means,
                    inputs.biological_set_endpoint_means,
                    inputs.endpoint_inference,
                    ratio_for_renderer,
                    output / "STATISTICS",
                    png_dpi=int(png_dpi),
                )
            finally:
                publication_plots_module.PUBLICATION_FAMILIES = original_families
            micro_outputs = render_publication_micrograph_package(
                inputs.selected_planes_h5,
                biological_image_audit,
                tables["spot_calls"],
                tables["nucleus_qc"],
                tables["footprint_pixels"],
                output / "MICROGRAPHS",
                representatives_per_arm=3,
                dpi=int(png_dpi),
            )
        methods_dir = output / "METHODS"
        source_dir = output / "SOURCE_DATA"
        qa_dir = output / "QA"
        for directory in (methods_dir, source_dir, qa_dir):
            directory.mkdir(exist_ok=False)
        copied_methods = {
            inputs.microscopy_methods.name: _copy_verified(
                inputs.microscopy_methods,
                methods_dir / inputs.microscopy_methods.name,
            ),
            inputs.figure_terms.name: _copy_verified(
                inputs.figure_terms, methods_dir / inputs.figure_terms.name
            ),
        }
        for path in plot_outputs.source_data_paths:
            shutil.copy2(path, source_dir / path.name)
        shutil.copy2(
            micro_outputs.selection_audit_path,
            source_dir / "micrograph_selection_audit.csv",
        )
        shutil.copy2(
            micro_outputs.source_data_path,
            source_dir / "micrograph_selection_manifest.csv",
        )
        plane_path = source_dir / "exact_z_channel_manifest.csv"
        plane_parity.to_csv(plane_path, index=False)
        references_path = source_dir / "retained_source_references.csv"
        pd.DataFrame(
            [
                {"source_name": name, **record}
                for name, record in source_metadata.items()
            ]
        ).to_csv(references_path, index=False)
        source_hash_manifest_path = source_dir / "source_hash_manifest.json"
        source_hash_manifest_path.write_text(
            json.dumps(
                {
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "input_discovery": "none; every input was explicit",
                    "sources": source_metadata,
                    "catalog_byte_integrity": catalog_integrity,
                    "canonical_large_tables": "retained by absolute path and SHA-256; not duplicated",
                    "ratio_denominator_compatibility": {
                        "rule": "alias only after exact equality of all 12 retained biological-set values",
                        "aliases": [
                            {
                                "renderer_endpoint": renderer,
                                "retained_endpoint": retained,
                            }
                            for _, renderer, retained in _RATIO_ALIASES
                        ],
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        micro_manifest = _json(
            micro_outputs.manifest_path, name="publication micrograph manifest"
        )
        figures = [
            *plot_outputs.png_paths,
            *plot_outputs.svg_paths,
            *micro_outputs.individual_png_paths,
            *micro_outputs.individual_svg_paths,
            *micro_outputs.contact_sheet_paths,
        ]
        manifest_path = output / "publication_manifest.json"
        qa_json_path = qa_dir / "qa_report.json"
        qa_markdown_path = qa_dir / "qa_report.md"
        inventory_path = qa_dir / "output_inventory.csv"
        manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "png_dpi": int(png_dpi),
            "figure_suffixes": [".png", ".svg"],
            "figure_files": _relative(output, figures),
            "pdf_files": [],
            "micrograph_examples": micro_manifest["selection_manifest"],
            "copied_methods": copied_methods,
            "source_hash_manifest": _relative(output, (source_hash_manifest_path,))[0],
            "output_inventory": _relative(output, (inventory_path,))[0],
            "qa_json": _relative(output, (qa_json_path,))[0],
            "qa_markdown": _relative(output, (qa_markdown_path,))[0],
            "scientific_contract": {
                "species": "human H9 hESC",
                "groups": "independent NT and MIAT-KD; no cross-arm pairing",
                "quantitation_plane": "exact_recorded_single_z",
                "same_z_all_channels": True,
                "projection_used": False,
                "qki_measurement": "inside exact detected MIAT footprint pixels only",
                "primary_inference": "exact two-sided within-slide label permutation",
                "claim_scope": "spatial QKI association; not proof of direct binding",
            },
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        readme_path = output / "README.md"
        readme_path.write_text(
            "# MIAT–QKI final publication outputs\n\n"
            "This fresh, non-overwriting package contains the final 600-dpi PNG and "
            "editable SVG statistical figures and six exact-single-z representative "
            "micrographs (three NT and three MIAT-KD). No projection was used.\n\n"
            "- `STATISTICS/`: four final statistical figure families and retained inference tables.\n"
            "- `MICROGRAPHS/`: six individual examples and two condition contact sheets.\n"
            "- `METHODS/`: hash-verified copies of the approved Methods and terminology documents.\n"
            "- `SOURCE_DATA/`: compact renderer source data, exact-z/channel audit, and source hashes.\n"
            "- `QA/`: numerical, structural, file-by-file, and visual QA records.\n\n"
            "q95 means QKI within a MIAT spot's exact detected footprint exceeded 95% "
            "of 1,000 same-nucleus KEEP-N randomized same-shape placements. Results "
            "support spatial QKI association with MIAT and do not by themselves prove "
            "direct binding or prove/disprove the sponge model.\n",
            encoding="utf-8",
        )
        _write_automated_qa(
            output,
            plot_outputs,
            micro_outputs,
            tables,
            plane_parity,
            catalog_integrity,
            qa_json_path,
            qa_markdown_path,
            inventory_path,
        )
        if any(output.rglob("*.pdf")):
            raise RuntimeError("PDF detected after final-publication generation")
        return FinalPublicationOutputs(
            output,
            manifest_path,
            source_hash_manifest_path,
            qa_json_path,
            qa_markdown_path,
            inventory_path,
        )
    except Exception as error:
        failure = output / "FAILED_GENERATION.json"
        failure.write_text(
            json.dumps(
                {
                    "status": "failed; retained and must not be reused",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        raise


def finalize_visual_qa(
    output_dir: str | Path,
    inspections: Iterable[dict[str, object]],
    *,
    replacement_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Record one immutable visual-QA pass and refresh the output inventory."""

    output = Path(output_dir)
    qa_json_path = output / "QA" / "qa_report.json"
    qa_markdown_path = output / "QA" / "qa_report.md"
    inventory_path = output / "QA" / "output_inventory.csv"
    payload = _json(qa_json_path, name="qa_report")
    if payload.get("visual_status") != "pending":
        raise RuntimeError("visual QA is already finalized")
    required = {str(Path(path).resolve()) for path in payload["visual_required_paths"]}
    normalized: list[dict[str, object]] = []
    for item in inspections:
        path = str(Path(str(item.get("path", ""))).resolve())
        verdict = str(item.get("verdict", "")).casefold()
        evidence = str(item.get("evidence", "")).strip()
        if verdict not in {"pass", "fail"} or not evidence:
            raise ValueError("each visual inspection needs pass/fail and evidence")
        normalized.append({"path": path, "verdict": verdict, "evidence": evidence})
    if {str(item["path"]) for item in normalized} != required or len(normalized) != len(required):
        raise ValueError("visual inspections must cover every required PNG exactly once")
    payload["visual_inspections"] = sorted(
        normalized, key=lambda item: str(item["path"]).casefold()
    )
    payload["visual_status"] = (
        "pass"
        if all(item["verdict"] == "pass" for item in normalized)
        else "fail"
    )
    payload["visual_finalized_utc"] = datetime.now(timezone.utc).isoformat()
    qa_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    qa_markdown_path.write_text(_qa_markdown(payload), encoding="utf-8")
    if payload["visual_status"] == "fail":
        mark_publication_superseded(
            output,
            status="visual_qa_failed",
            reason="visual_status=fail: one or more required visual inspections failed.",
            replacement_path=replacement_path,
        )
    _write_output_inventory(output, inventory_path)
    return qa_json_path, qa_markdown_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for field in fields(FinalPublicationInputs):
        parser.add_argument(
            "--" + field.name.replace("_", "-"),
            required=True,
            type=Path,
        )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--png-dpi", type=int, default=600)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    inputs = FinalPublicationInputs(
        **{field.name: getattr(args, field.name) for field in fields(FinalPublicationInputs)}
    )
    outputs = generate_final_publication(
        inputs, args.output_dir, png_dpi=args.png_dpi
    )
    print(
        json.dumps(
            {
                "status": "generated; visual QA pending",
                "output_dir": str(outputs.output_dir.resolve()),
                "manifest": str(outputs.manifest_path.resolve()),
                "qa": str(outputs.qa_json_path.resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
