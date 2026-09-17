"""Exact-single-plane MIAT/QKI historical-run reconstruction.

This module is deliberately separate from the normal FishSuite runner.  It
reuses persisted nucleus labels and detected MIAT centres, reads only the
recorded two-dimensional z plane, reconstructs each original half-maximum MIAT
footprint, and calibrates QKI at those exact pixels with the KEEP-N rotation
null.  It never performs autofocus, projection, segmentation, or spot
detection.

The functions below are split into small quantitative units so the expensive
VSI/HDF5 orchestration can be tested with real arrays and a tiny reader double.
"""
from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .footprint_null import (
    MiatFootprint,
    build_miat_footprints,
    encode_footprint_rle,
    full_footprint_is_valid,
    keep_n_footprint_rotation_null,
)


PLACEMENT_GEOMETRY = "rotated_center_translated_exact_footprint"
THRESHOLD_PERCENTILES = (90, 95, 99)


@dataclass(frozen=True)
class ExactFootprintParameters:
    """Quantitative parameters; display limits intentionally do not belong here."""

    miat_floor_raw: float = 364.0
    n_null: int = 1000
    global_seed: int = 0
    max_redraw: int = 1000
    min_first_pass_retention: float = 0.5
    min_valid_draw_fraction: float = 1.0
    min_valid_draws: int = 2
    threshold_percentiles: tuple[int, ...] = THRESHOLD_PERCENTILES
    primary_threshold_percentile: int = 95
    half_max_frac: float = 0.5
    background_percentile: float = 10.0
    window_pad_px: int = 2
    min_window_half: int = 4
    max_window_half: int = 12
    peak_search_half: int = 1
    disk_radii_px: tuple[int, ...] = (1, 3, 6)

    def __post_init__(self) -> None:
        if self.n_null < 1:
            raise ValueError("n_null must be positive")
        if self.max_redraw < 0:
            raise ValueError("max_redraw must be non-negative")
        if self.primary_threshold_percentile not in self.threshold_percentiles:
            raise ValueError("primary threshold must be one of threshold_percentiles")
        if tuple(sorted(set(self.threshold_percentiles))) != self.threshold_percentiles:
            raise ValueError("threshold_percentiles must be unique and sorted")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["placement_geometry"] = PLACEMENT_GEOMETRY
        payload["threshold_operator"] = "observed_qki_mean > own_linear_quantile"
        payload["center_rounding"] = "numpy_rint_ties_to_even"
        return payload


@dataclass
class ReconstructionResult:
    """All per-image products before serial checkpointing."""

    spot_metrics: pd.DataFrame
    pixel_metrics: pd.DataFrame
    footprints_by_nucleus: dict[int, tuple[MiatFootprint, ...]]
    null_groups: dict[int, dict[str, Any]]
    audit: dict[str, Any]
    planes: dict[str, np.ndarray]
    nucleus_labels: np.ndarray
    nucleolus_labels: np.ndarray


def _explicit_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    text = str(value).strip().casefold()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n", "", "nan", "none"}:
        return False
    raise ValueError(f"cannot interpret boolean value {value!r}")


def _finite_integer(value: Any, *, name: str, minimum: int | None = None) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite integer") from exc
    if not np.isfinite(numeric) or numeric != round(numeric):
        raise ValueError(f"{name} must be a finite integer")
    integer = int(numeric)
    if minimum is not None and integer < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return integer


def _casefold_basename(value: Any) -> str:
    return Path(str(value)).name.casefold()


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], *, table: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{table} is missing required columns: {missing}")


def stable_nucleus_seed(
    run_id: str,
    image_key: str,
    nucleus_id: int,
    global_seed: int,
) -> int:
    """Return an order/resume-independent unsigned 64-bit seed."""

    material = json.dumps(
        [str(run_id), str(image_key).casefold(), int(nucleus_id), int(global_seed)],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big", signed=False)


def diameter_um_to_px(values: Any, *, voxel_xy_nm: float) -> np.ndarray:
    """Convert persisted physical spot diameter to pixels."""

    voxel = float(voxel_xy_nm)
    if not np.isfinite(voxel) or voxel <= 0:
        raise ValueError("voxel_xy_nm must be finite and positive")
    diameters = np.asarray(values, dtype=np.float64)
    return diameters / (voxel / 1000.0)


def _unique_image_hierarchy(hierarchy: pd.DataFrame) -> pd.DataFrame:
    required = {
        "image",
        "image_key",
        "condition",
        "secondary_only",
        "slide",
        "arm",
        "source_arm",
        "replicate",
        "fov",
        "biological_set",
        "catalog_folder",
        "is_control",
        "eligible_for_sampling",
        "sampled_in_analysis",
    }
    _require_columns(hierarchy, required, table="authoritative hierarchy")
    identity = sorted(required.difference({"eligible_for_sampling", "sampled_in_analysis"}))
    rows: list[pd.Series] = []
    for image_key, group in hierarchy.groupby("image_key", sort=False, dropna=False):
        if not str(image_key).strip():
            raise ValueError("authoritative hierarchy contains an empty image_key")
        for column in identity:
            values = group[column].fillna("<NA>").astype(str).unique()
            if len(values) != 1:
                raise ValueError(
                    f"authoritative hierarchy is inconsistent for {image_key!r}: {column}"
                )
        row = group.iloc[0].copy()
        row["eligible_for_sampling"] = bool(
            group["eligible_for_sampling"].map(_explicit_bool).any()
        )
        row["sampled_in_analysis"] = bool(
            group["sampled_in_analysis"].map(_explicit_bool).any()
        )
        row["n_nuclei"] = int(group["nucleus_id"].nunique()) if "nucleus_id" in group else len(group)
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def _source_image_key(value: Any) -> str:
    path = Path(str(value))
    if not path.is_absolute():
        raise ValueError("native hierarchy source_path must record an absolute source path")
    return path.resolve(strict=False).as_posix().casefold()


def _uses_source_identity(hierarchy: pd.DataFrame) -> bool:
    """Also accept an explicitly exported source-keyed hierarchy without altering legacy keys."""
    if hierarchy.attrs.get("native_hierarchy"):
        return True
    if hierarchy.empty or not {"source_vsi", "image_key"}.issubset(hierarchy.columns):
        return False
    if not hierarchy["source_vsi"].map(lambda value: Path(str(value)).is_absolute()).all():
        return False
    return bool(hierarchy["image_key"].astype(str).eq(
        hierarchy["source_vsi"].map(_source_image_key)).all())


def _source_keys_for_table(frame: pd.DataFrame, hierarchy: pd.DataFrame, *, table: str) -> pd.Series:
    """Join via recorded source paths, or an unambiguous exact recorded image label."""
    roster = hierarchy[["image", "image_key", "source_vsi"]].drop_duplicates()
    if roster["image_key"].duplicated().any():
        raise ValueError("source_path identity must be unique in the image roster")
    if "source_path" in frame:
        keys = frame["source_path"].map(_source_image_key)
    else:
        if roster["image"].duplicated().any() and not frame.empty:
            raise ValueError(f"{table} requires recorded source_path for ambiguous image labels")
        lookup = roster.drop_duplicates("image").set_index("image")["image_key"]
        keys = frame["image"].map(lookup)
    if keys.isna().any() or not set(keys).issubset(set(roster["image_key"])):
        raise ValueError(f"{table} contains source_path/image identities absent from recorded hierarchy")
    expected_images = keys.map(roster.set_index("image_key")["image"])
    if not frame["image"].eq(expected_images).all():
        raise ValueError(f"{table} image label disagrees with recorded source_path hierarchy")
    return keys


def native_hierarchy_from_run(run_dir: str | Path) -> pd.DataFrame:
    """Adapt recorded native metadata without inferring experimental identities.

    Sampling-disabled exports represent the complete recorded nucleus population.
    The legacy slide field carries well_id as a native computation stratum, never
    an invented acquisition slide. Historical slide-blocked inference is separate.
    This function reads tables only and never writes or reads image pixels.
    """
    run = Path(run_dir)
    names = ("nuclei_metrics.csv", "resolved_experiment_hierarchy.csv",
             "per_image_summary.csv", "spot_metrics.csv", "run_config.json")
    for name in names:
        if not (run / name).is_file():
            raise FileNotFoundError(f"native hierarchy missing required run file: {run / name}")
    nuclei = pd.read_csv(run / names[0], dtype=str, keep_default_na=False)
    roster = pd.read_csv(run / names[1], dtype=str, keep_default_na=False)
    per_image = pd.read_csv(run / names[2], dtype=str, keep_default_na=False)
    config = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
    resolved = config.get("config_resolved", {})
    sampling = resolved.get("sampling", {}) if isinstance(resolved, Mapping) else {}
    if not isinstance(sampling, Mapping) or not isinstance(sampling.get("enabled"), bool):
        raise ValueError("run_config.json must record config_resolved.sampling.enabled as a boolean")
    native_provenance = {
        "slide": "well_id (native computation stratum; not acquisition slide)",
        "catalog_folder": "recorded in nuclei_metrics.csv",
        "image_identity": "normalized recorded source_path",
        "historical_slide_blocked_inference": "unsupported_native_well_strata",
    }
    if sampling["enabled"]:
        _require_columns(nuclei, {"eligible_for_sampling", "sampled_in_analysis"},
                         table="sampling enabled in run; nuclei_metrics.csv sampling records")
        native_provenance["sampling"] = (
            "enabled_in_run; mapped nuclei_metrics.csv.eligible_for_sampling and sampled_in_analysis")
    else:
        nuclei["eligible_for_sampling"] = "True"
        nuclei["sampled_in_analysis"] = "True"
        native_provenance["sampling"] = "disabled_in_run; all recorded nuclei eligible and sampled"
    if "catalog_folder" not in nuclei:
        nuclei["catalog_folder"] = "not_recorded"
        native_provenance["catalog_folder"] = "not_recorded (native hierarchy)"
    else:
        missing_catalog = nuclei["catalog_folder"].str.strip().eq("")
        if missing_catalog.any():
            nuclei.loc[missing_catalog, "catalog_folder"] = "not_recorded"
            native_provenance["catalog_folder"] = "not_recorded (native hierarchy); recorded values retained"
    # This compatibility key makes native grouping use the recorded well, not
    # an unknown slide sentinel. The actual source table remains checksummed.
    nuclei["slide"] = ""  # assigned by the recorded roster after the join below
    nucleus_columns = ["image", "nucleus_id", "slide", "catalog_folder",
                       "eligible_for_sampling", "sampled_in_analysis",
                       "z_mode", "z_range", "n_z_slices"]
    roster_columns = ["image", "condition", "secondary_only", "group",
                      "source_condition", "well_id", "field_id", "source_path"]
    _require_columns(nuclei, nucleus_columns, table="native nuclei_metrics.csv")
    _require_columns(roster, roster_columns, table="native resolved_experiment_hierarchy.csv")
    _require_columns(per_image, {"image", "condition", "secondary_only"},
                     table="native per_image_summary.csv")
    missing_parity = sorted({"miat_footprint_area_px", "qki_at_miat_footprint"}.difference(
        pd.read_csv(run / names[3], nrows=0).columns))
    if missing_parity:
        raise ValueError(f"native spot_metrics.csv missing {missing_parity}; "
                         "re-run fishsuite with foci.compute_footprint_enrichment: true — "
                         "the backfill verifies its reconstructed footprints against these run-time columns")
    for frame, columns, name in ((nuclei, nucleus_columns, names[0]),
                                 (roster, roster_columns, names[1]),
                                 (per_image, ["image", "condition", "secondary_only"], names[2])):
        for column in columns:
            if column == "slide":
                continue
            if frame[column].str.strip().eq("").any():
                raise ValueError(f"native {name} has missing values for {column}")
    roster["image_key"] = roster["source_path"].map(_source_image_key)
    if roster["image_key"].duplicated().any():
        raise ValueError("native resolved_experiment_hierarchy.csv requires unique source_path identities")
    source_roster = roster.rename(columns={"source_path": "source_vsi"})
    for frame, name in ((per_image, names[2]), (nuclei, names[0])):
        frame["image_key"] = _source_keys_for_table(frame, source_roster, table=name)
    for frame, name in ((roster, names[1]), (per_image, names[2])):
        if frame.empty or frame["image_key"].duplicated().any():
            raise ValueError(f"native {name} must have unique nonempty source_path identities")
    keys = set(roster["image_key"])
    if set(per_image["image_key"]) != keys or set(nuclei["image_key"]) != keys:
        raise ValueError("native image roster mismatch across resolved_experiment_hierarchy.csv, "
                         "per_image_summary.csv and nuclei_metrics.csv")
    nuclei["nucleus_id"] = nuclei["nucleus_id"].map(
        lambda value: _finite_integer(value, name="native nucleus_id", minimum=1))
    if nuclei.duplicated(["image_key", "nucleus_id"]).any():
        raise ValueError("native nuclei_metrics.csv contains duplicate nucleus identities")
    def recorded_bool(value: str) -> bool:
        if value.strip().casefold() not in {"true", "false", "1", "0", "yes", "no", "y", "n"}:
            raise ValueError(f"native hierarchy requires a recorded boolean, found {value!r}")
        return _explicit_bool(value)

    for flag in ("eligible_for_sampling", "sampled_in_analysis"):
        nuclei[flag] = nuclei[flag].map(recorded_bool)
    if (nuclei["sampled_in_analysis"] & ~nuclei["eligible_for_sampling"]).any():
        raise ValueError("native sampled_in_analysis requires eligible_for_sampling")
    roster["secondary_only"] = roster["secondary_only"].map(recorded_bool)
    # Require consistency wherever metadata is repeated in native exports.
    authoritative = roster.set_index("image_key")
    for frame, name in ((per_image, names[2]), (nuclei, names[0])):
        for column in roster_columns:
            if column not in frame:
                continue
            recorded = frame[column]
            expected = frame["image_key"].map(authoritative[column])
            if column == "secondary_only":
                recorded = recorded.map(recorded_bool)
            elif column == "source_path":
                recorded = recorded.map(_source_image_key)
                expected = expected.map(_source_image_key)
            if not recorded.eq(expected).all():
                raise ValueError(f"native {name} disagrees with recorded hierarchy: {column}")
    result = nuclei[["image_key", *nucleus_columns[1:]]].merge(
        roster[["image_key", *roster_columns]], on="image_key", validate="many_to_one")
    for target, source in {"arm": "group", "source_arm": "source_condition",
                           "replicate": "well_id", "fov": "field_id",
                           "biological_set": "well_id", "source_vsi": "source_path",
                           "well": "well_id", "is_control": "secondary_only"}.items():
        result[target] = result[source]
    result["slide"] = result["well_id"]
    columns = ["image", "image_key", "nucleus_id", "condition", "secondary_only",
               "slide", "arm", "source_arm", "replicate", "fov", "biological_set",
               "catalog_folder", "is_control", "eligible_for_sampling", "sampled_in_analysis",
               "source_vsi", "well", "z_mode", "z_range", "n_z_slices"]
    result = result[columns]
    if "output_stem" in roster:
        result["output_stem"] = result["image_key"].map(roster.set_index("image_key")["output_stem"])
    _unique_image_hierarchy(result)  # validate invariant image-level metadata
    result.attrs["native_hierarchy"] = True
    native_provenance["observed_native_design"] = [
        int(len(roster)), int(roster["condition"].nunique()), int(roster["well_id"].nunique())]
    result.attrs["native_provenance"] = native_provenance
    return result


def _subset_lookup(run_config: Mapping[str, Any]) -> tuple[dict[str, str], Path | None]:
    resolved = run_config.get("config_resolved", {})
    if not isinstance(resolved, Mapping):
        resolved = {}
    subset = resolved.get("input_file_subset", run_config.get("input_file_subset", []))
    if not isinstance(subset, Sequence) or isinstance(subset, (str, bytes)):
        raise ValueError("run_config input_file_subset must be a sequence")
    lookup: dict[str, str] = {}
    for item in subset:
        rel = str(item)
        key = _casefold_basename(rel)
        if key in lookup:
            raise ValueError(f"input_file_subset has duplicate basename {key!r}")
        lookup[key] = rel
    input_dir_value = run_config.get("input_dir")
    input_dir = Path(str(input_dir_value)) if input_dir_value else None
    return lookup, input_dir


def resolve_unique_mask_path(
    masks_dir: str | Path,
    image_name: str,
    condition: str,
) -> Path:
    """Resolve exactly one plausible persisted nucleus-label mask."""

    directory = Path(masks_dir)
    stem_us = Path(image_name).stem.replace(" ", "_")
    del condition  # hierarchy condition is intentionally not used to hide ambiguity
    suffix = "__nuclei_label_mask.tif"
    candidates: list[Path] = []
    for candidate in sorted(directory.glob("*" + suffix)):
        core = candidate.name[: -len(suffix)]
        parts = core.split("__")
        middle = parts[-1] if parts else core
        if not middle:
            continue
        # The persisted mask token is the terminal, sanitized portion of the
        # acquisition stem. Match only at a non-alphanumeric boundary so a
        # mask for field cannot silently resolve field2. Multiple
        # boundary-valid candidates remain an explicit ambiguity failure.
        name_match = bool(
            re.search(
                rf"(?:^|[^0-9A-Za-z]){re.escape(middle)}$",
                stem_us,
                flags=re.IGNORECASE,
            )
        )
        if name_match:
            candidates.append(candidate)
    if not candidates:
        raise FileNotFoundError(f"no saved nucleus mask for {image_name}")
    if len(candidates) != 1:
        raise ValueError(
            f"ambiguous saved nucleus masks for {image_name}: "
            f"{[str(path) for path in candidates]}"
        )
    return candidates[0]


def validate_vsi_companion(vsi_path: str | Path) -> tuple[Path, int]:
    """Require the Olympus ``_<stem>_`` directory and one nonzero ETS payload."""

    path = Path(vsi_path)
    if path.suffix.casefold() != ".vsi":
        raise ValueError(f"exact-footprint input must be a VSI file: {path}")
    companion = path.parent / f"_{path.stem}_"
    if not companion.is_dir():
        raise FileNotFoundError(f"VSI companion directory is missing: {companion}")
    ets_files = [item for item in companion.rglob("*.ets") if item.is_file()]
    nonzero = [item for item in ets_files if item.stat().st_size > 0]
    if len(nonzero) != 1:
        raise ValueError(
            f"VSI companion must contain exactly one nonzero ETS for {path.name}; "
            f"found {len(nonzero)}"
        )
    return nonzero[0], int(nonzero[0].stat().st_size)


def read_and_validate_label_mask(
    path: str | Path,
    *,
    expected_shape: tuple[int, int],
    expected_nucleus_ids: Iterable[int],
    spots: pd.DataFrame,
) -> np.ndarray:
    """Read uint16 labels and reconcile shape, IDs, and every (y, x) centre."""

    import tifffile

    source = Path(path)
    raw = np.asarray(tifffile.imread(source))
    if raw.dtype != np.uint16:
        raise ValueError(f"saved nucleus mask must be uint16, found {raw.dtype}")
    if raw.shape != tuple(expected_shape):
        raise ValueError(
            f"saved nucleus mask shape {raw.shape} does not match {expected_shape}"
        )
    labels = raw.astype(np.int32, copy=False)
    actual_ids = {int(value) for value in np.unique(labels) if int(value) > 0}
    expected_ids = {int(value) for value in expected_nucleus_ids}
    if actual_ids != expected_ids:
        raise ValueError(
            f"saved nucleus mask nucleus IDs differ: missing={sorted(expected_ids-actual_ids)}, "
            f"extra={sorted(actual_ids-expected_ids)}"
        )
    if not spots.empty:
        _require_columns(spots, {"y_px", "x_px", "nucleus_id"}, table="spot centres")
        for row in spots.itertuples(index=False):
            y = _finite_integer(row.y_px, name="spot y_px", minimum=0)
            x = _finite_integer(row.x_px, name="spot x_px", minimum=0)
            nid = _finite_integer(row.nucleus_id, name="spot nucleus_id", minimum=1)
            if y >= labels.shape[0] or x >= labels.shape[1] or int(labels[y, x]) != nid:
                raise ValueError(
                    f"spot centre (y={y}, x={x}) does not land on recorded parent "
                    f"nucleus {nid}"
                )
    return labels


def _native_spot_eligibility(
    spots: pd.DataFrame, labels: np.ndarray, *, image_key: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Select native nuclear centres; territory assignment is not containment.

    Reasons are exclusive: a recorded non-nuclear spot is counted first,
    including unassigned parent 0. A claimed nuclear centre off its parent is
    excluded only at distance <= 1 px; larger disagreement is corruption.
    """
    _require_columns(spots, {"in_nucleus", "y_px", "x_px", "nucleus_id"},
                     table="native RNA1 spots")
    work = spots.copy()
    keep = np.zeros(len(work), dtype=bool)
    counts = dict(n_input_spots=len(work), n_eligible_spots=0,
                  not_in_nucleus=0, centre_off_parent_label=0)
    for position, row in enumerate(work.itertuples(index=False)):
        flag = row.in_nucleus
        if pd.isna(flag) or str(flag).strip().casefold() in {"", "nan", "none"}:
            raise ValueError(f"native in_nucleus must record a boolean: {image_key}")
        if isinstance(flag, (int, float, np.integer, np.floating)):
            if flag not in (0, 1):
                raise ValueError(f"native in_nucleus must record a boolean: {flag!r}")
            nuclear = bool(flag)
        else:
            nuclear = _explicit_bool(flag)
        if not nuclear:
            counts["not_in_nucleus"] += 1
            continue
        centre = np.asarray([row.y_px, row.x_px], dtype=float)
        if not np.isfinite(centre).all():
            raise ValueError(f"native nuclear spot centre must be finite: {image_key}")
        # The native footprint sampler uses np.rint (rna_rna.py); persisted
        # native detections are already integer x_px/y_px, not fitted centres.
        y, x = (int(value) for value in np.rint(centre))
        nid = _finite_integer(row.nucleus_id, name="native nuclear parent", minimum=1)
        on_parent = (0 <= y < labels.shape[0] and 0 <= x < labels.shape[1]
                     and int(labels[y, x]) == nid)
        if not on_parent:
            # Only the four axial neighbours can be within 1 Euclidean pixel.
            # Do not admit these spots or relax the strict historical assertion.
            within_one = any(
                0 <= yy < labels.shape[0] and 0 <= xx < labels.shape[1]
                and int(labels[yy, xx]) == nid
                for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1))
            )
            if not within_one:
                raise ValueError(
                    f"native spot {getattr(row, 'spot_id', position)!r} in {image_key}: "
                    f"in_nucleus=True but centre (y={y}, x={x}) is > 1 px "
                    f"from recorded parent nucleus {nid} (or parent is absent)"
                )
            counts["centre_off_parent_label"] += 1
            continue
        keep[position] = True
        work.iloc[position, work.columns.get_loc("y_px")] = y
        work.iloc[position, work.columns.get_loc("x_px")] = x
    counts["n_eligible_spots"] = int(keep.sum())
    return work.loc[keep].copy(), counts


def _filter_native_spots(
    historical: pd.DataFrame, manifest: pd.DataFrame, hierarchy: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read saved masks only; retain per-image accounting even for empty images."""
    import tifffile

    work = historical.copy()
    keep = pd.Series(True, index=work.index)
    summaries = []
    for row in manifest.loc[manifest["selected_for_execution"]].itertuples(index=False):
        scoped = work["image_key"].eq(row.image_key) & work["channel"].eq("rna1")
        image_spots = work.loc[scoped]
        expected_ids = hierarchy.loc[hierarchy["image_key"].eq(row.image_key), "nucleus_id"]
        with tifffile.TiffFile(row.mask_path) as mask:
            shape = mask.series[0].shape
        if len(shape) != 2:
            raise ValueError(f"saved native nucleus mask must be 2D: {row.mask_path}")
        labels = read_and_validate_label_mask(
            row.mask_path, expected_shape=shape, expected_nucleus_ids=expected_ids,
            spots=image_spots.iloc[:0],
        )
        eligible, counts = _native_spot_eligibility(image_spots, labels, image_key=row.image_key)
        keep.loc[scoped] = False
        keep.loc[eligible.index] = True
        work.loc[eligible.index, ["y_px", "x_px"]] = eligible[["y_px", "x_px"]]
        summaries.append({"image": row.image, "image_key": row.image_key, **counts})
    return work.loc[keep].copy(), pd.DataFrame(summaries)


def build_image_manifest(
    per_image: pd.DataFrame,
    hierarchy: pd.DataFrame,
    spot_metrics: pd.DataFrame,
    *,
    run_config: Mapping[str, Any],
    run_dir: str | Path,
    validate_expected_design: bool = False,
    resolve_paths: bool = True,
) -> pd.DataFrame:
    """Build the image universe from per-image records and audited hierarchy.

    The spots table is used only for counts, so a zero-spot image cannot vanish.
    Filename contents are never parsed into experimental metadata.
    """

    _require_columns(
        per_image,
        {
            "image",
            "condition",
            "secondary_only",
            "z_plane",
            "dapi_channel",
            "rna_channel",
            "protein_channel",
            "voxel_xy_nm",
            "voxel_z_nm",
            "n_z",
        },
        table="per_image_summary",
    )
    _require_columns(spot_metrics, {"image", "channel", "spot_id"}, table="spot_metrics")
    source_identity = _uses_source_identity(hierarchy)
    base = per_image.copy()
    base["image_key"] = (_source_keys_for_table(base, hierarchy, table="per_image_summary.csv")
                         if source_identity else base["image"].map(_casefold_basename))
    if base["image_key"].duplicated().any():
        dup = sorted(base.loc[base["image_key"].duplicated(False), "image_key"].unique())
        raise ValueError(f"per_image_summary has duplicate images: {dup}")
    audited = _unique_image_hierarchy(hierarchy)
    audited["image_key"] = audited["image_key"].astype(str).str.casefold()
    if source_identity and "output_stem" in audited:
        if audited["output_stem"].astype(str).str.casefold().duplicated().any():
            raise ValueError("recorded output_stem must be unique per source_path; duplicate masks are ambiguous")
    hierarchy_columns = [
        "image",
        "image_key",
        "condition",
        "secondary_only",
        "slide",
        "arm",
        "source_arm",
        "replicate",
        "fov",
        "biological_set",
        "catalog_folder",
        "is_control",
        "eligible_for_sampling",
        "sampled_in_analysis",
        "n_nuclei",
    ]
    if "source_vsi" in audited.columns:
        hierarchy_columns.append("source_vsi")
    if source_identity:
        hierarchy_columns.extend(column for column in ("well", "output_stem") if column in audited)
    merged = base.merge(
        audited[hierarchy_columns],
        on="image_key",
        how="left",
        validate="one_to_one",
        suffixes=("_per_image", ""),
    )
    if merged["is_control"].isna().any():
        missing = sorted(merged.loc[merged["is_control"].isna(), "image_key"])
        raise ValueError(f"images are missing authoritative hierarchy rows: {missing}")
    for column in ("secondary_only", "is_control", "eligible_for_sampling", "sampled_in_analysis"):
        merged[column] = merged[column].map(_explicit_bool)
    merged["eligible_for_biological_inference"] = ~merged["is_control"]

    selected_z: list[int] = []
    n_z_values: list[int] = []
    for row in merged.itertuples(index=False):
        z = _finite_integer(row.z_plane, name="per_image_summary.z_plane", minimum=1)
        n_z = _finite_integer(row.n_z, name="per_image_summary.n_z", minimum=1)
        if z > n_z:
            raise ValueError(f"selected z {z} lies outside 1..{n_z} for {row.image}")
        selected_z.append(z)
        n_z_values.append(n_z)
    merged["selected_z_1based"] = selected_z
    merged["selected_z_0based"] = np.asarray(selected_z, dtype=int) - 1
    merged["z_source"] = "per_image_summary.z_plane"
    merged["plane_lock_pass"] = True
    merged["n_z"] = n_z_values
    merged = merged.rename(
        columns={
            "dapi_channel": "dapi_channel_index",
            "rna_channel": "miat_channel_index",
            "protein_channel": "qki_channel_index",
        }
    )
    counts = (
        spot_metrics.loc[spot_metrics["channel"].astype(str).eq("rna1")]
        .assign(image_key=lambda x: _source_keys_for_table(x, hierarchy, table="spot_metrics.csv")
                if source_identity else x["image"].map(_casefold_basename))
        .groupby("image_key", sort=False)
        .size()
    )
    merged["n_input_spots"] = merged["image_key"].map(counts).fillna(0).astype(int)

    if source_identity:
        subset, input_dir = {}, None
    else:
        subset, input_dir = _subset_lookup(run_config)
    merged["source_relpath"] = merged["image_key"].map(subset)
    if subset and merged["source_relpath"].isna().any():
        missing = sorted(merged.loc[merged["source_relpath"].isna(), "image_key"])
        raise ValueError(f"input_file_subset is missing images: {missing}")
    if source_identity:
        merged["analyzed_vsi_path"] = merged["source_vsi"]
    elif input_dir is None:
        merged["analyzed_vsi_path"] = merged["source_relpath"].fillna(merged["image"])
    else:
        merged["analyzed_vsi_path"] = merged["source_relpath"].map(
            lambda value: str(input_dir / str(value)) if pd.notna(value) else ""
        )
    merged["raw_source_vsi"] = (
        merged["source_vsi"].astype(str)
        if "source_vsi" in merged.columns
        else merged["analyzed_vsi_path"].astype(str)
    )
    merged["analysis_run_id"] = Path(run_dir).name
    merged["load_status"] = "pending"
    merged["mask_status"] = "pending"
    merged["mask_path"] = ""
    merged["height_px"] = np.nan
    merged["width_px"] = np.nan
    merged["source_checksum"] = ""
    merged["mask_checksum"] = ""
    merged["notes"] = ""
    mismatch_fields = ("image", "condition", "secondary_only")
    for field_name in mismatch_fields:
        left = merged[f"{field_name}_per_image"]
        right = merged[field_name]
        if field_name == "secondary_only":
            mismatch = left.map(_explicit_bool) != right.map(_explicit_bool)
        elif field_name == "image":
            mismatch = left.map(_casefold_basename) != right.map(_casefold_basename)
        else:
            mismatch = left.fillna("").astype(str) != right.fillna("").astype(str)
        merged[f"hierarchy_mismatch_{field_name}"] = mismatch
    merged["hierarchy_mismatch_fields"] = [
        ";".join(
            field_name
            for field_name in mismatch_fields
            if bool(row[f"hierarchy_mismatch_{field_name}"])
        )
        for row in merged.to_dict("records")
    ]
    merged["notes"] = merged["hierarchy_mismatch_fields"].map(
        lambda value: (
            f"authoritative hierarchy overrode per-image fields: {value}"
            if value
            else ""
        )
    )

    if resolve_paths:
        masks_dir = Path(run_dir) / "masks"
        for index, row in merged.iterrows():
            source_path = Path(str(row["analyzed_vsi_path"]))
            if not source_path.is_file():
                raise FileNotFoundError(f"analyzed VSI is missing: {source_path}")
            ets_path, ets_size = validate_vsi_companion(source_path)
            if source_identity and "output_stem" in merged:
                stem = str(row["output_stem"])
                if not stem.strip() or Path(stem).name != stem or stem in {".", ".."}:
                    raise ValueError("recorded output_stem must be a nonempty filename stem")
                mask_path = masks_dir / f"{stem}__nuclei_label_mask.tif"
                if not mask_path.is_file():
                    raise FileNotFoundError(f"recorded output_stem nucleus mask is missing: {mask_path}")
            else:
                mask_path = resolve_unique_mask_path(
                    masks_dir, str(row["image"]), str(row["condition"])
                )
            merged.at[index, "mask_path"] = str(mask_path)
            merged.at[index, "mask_status"] = "resolved_unique"
            merged.at[index, "companion_ets_path"] = str(ets_path)
            merged.at[index, "companion_ets_size_bytes"] = ets_size

    n_control = int(merged["is_control"].sum())
    n_biological = int((~merged["is_control"]).sum())
    n_sets = int(
        merged.loc[~merged["is_control"], "biological_set"].replace("", np.nan).nunique()
    )
    observed_design = (len(merged), n_biological, n_control, n_sets)
    if validate_expected_design and observed_design != (44, 37, 7, 12):
        raise ValueError(
            "audited design must be 44 images, 37 biological, 7 controls, "
            f"and 12 sets; found {observed_design}"
        )
    result = merged.reset_index(drop=True)
    result.attrs["observed_design"] = observed_design
    if source_identity:
        result.attrs["source_path_identity"] = True
    return result


def read_exact_selected_planes(
    vsi_path: str | Path,
    *,
    selected_z_1based: Any,
    channel_indices: Mapping[str, int],
    image_reader: Callable[[Path], Any] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read C0/C1/C2 directly as YX at one recorded z; never load a stack."""

    z = _finite_integer(selected_z_1based, name="selected_z_1based", minimum=1)
    if image_reader is None:
        from .io import read_image

        image_reader = read_image
    image = image_reader(Path(vsi_path))
    try:
        n_z = _finite_integer(image.n_z, name="image.n_z", minimum=1)
        if z > n_z:
            raise ValueError(f"selected_z_1based {z} is outside image range 1..{n_z}")
        required_roles = ("miat", "qki", "dapi")
        if set(channel_indices) != set(required_roles):
            raise ValueError(f"channel_indices must contain exactly {required_roles}")
        z0 = z - 1
        planes: dict[str, np.ndarray] = {}
        shape: tuple[int, int] | None = None
        for role in required_roles:
            channel = _finite_integer(channel_indices[role], name=f"{role}_channel", minimum=0)
            if channel >= int(image.n_channels):
                raise ValueError(f"{role} channel {channel} is outside image channel range")
            plane = np.asarray(
                image.bio.get_image_data("YX", T=0, C=channel, Z=z0)
            ).copy()
            if plane.ndim != 2:
                raise ValueError(f"direct {role} plane read returned shape {plane.shape}, expected YX")
            if shape is None:
                shape = plane.shape
            elif plane.shape != shape:
                raise ValueError("selected channel planes have different shapes")
            planes[role] = plane
        metadata = {
            "selected_z_1based": z,
            "selected_z_0based": z0,
            "n_z": n_z,
            "n_channels": int(image.n_channels),
            "voxel_xy_nm": float(getattr(image, "voxel_xy_nm", np.nan)),
            "voxel_z_nm": float(getattr(image, "voxel_z_nm", np.nan)),
            "height_px": int(shape[0]),
            "width_px": int(shape[1]),
            "plane_dtypes": {
                role: str(np.asarray(planes[role]).dtype) for role in required_roles
            },
            "plane_lock_pass": True,
            "reader_request": "YX/T0/Crole/Zselected_minus_one",
        }
        return planes, metadata
    finally:
        closed: set[int] = set()
        for candidate in (
            image,
            getattr(image, "bio", None),
            getattr(getattr(image, "bio", None), "reader", None),
        ):
            if candidate is None or id(candidate) in closed:
                continue
            close = getattr(candidate, "close", None)
            if callable(close):
                close()
                closed.add(id(candidate))


def _validate_loaded_plane_bundle(
    planes: Mapping[str, np.ndarray],
    loaded: Mapping[str, Any],
    *,
    image_key: str,
    selected_z_1based: int,
    channel_indices: Mapping[str, int],
    validate_expected_design: bool,
) -> None:
    required = {"miat", "qki", "dapi"}
    if set(planes) != required:
        raise ValueError(f"loaded planes differ from C0/C1/C2 roles for {image_key}")
    arrays = {role: np.asarray(array) for role, array in planes.items()}
    shapes = {array.shape for array in arrays.values()}
    if len(shapes) != 1 or any(array.ndim != 2 for array in arrays.values()):
        raise ValueError(f"loaded planes are not same-shape YX arrays for {image_key}")
    shape = next(iter(shapes))
    expected_z0 = int(selected_z_1based) - 1
    if int(loaded["selected_z_1based"]) != int(selected_z_1based):
        raise ValueError(f"plane reader changed the selected z for {image_key}")
    if int(loaded["selected_z_0based"]) != expected_z0:
        raise ValueError(f"plane reader did not perform exactly one 1-to-0 z conversion for {image_key}")
    if int(loaded.get("height_px", shape[0])) != shape[0] or int(
        loaded.get("width_px", shape[1])
    ) != shape[1]:
        raise ValueError(f"loaded plane dimensions disagree with metadata for {image_key}")
    loaded_dtypes = loaded.get("plane_dtypes")
    if isinstance(loaded_dtypes, Mapping):
        for role, array in arrays.items():
            if str(loaded_dtypes.get(role, "")) != str(array.dtype):
                raise ValueError(f"loaded {role} dtype metadata mismatch for {image_key}")
    n_channels = loaded.get("n_channels")
    if n_channels is not None and int(n_channels) <= max(channel_indices.values()):
        raise ValueError(f"loaded channel count cannot contain C0/C1/C2 for {image_key}")
    if validate_expected_design:
        if shape != (2304, 2304):
            raise ValueError(f"audited selected plane must be 2304x2304 for {image_key}")
        if any(array.dtype != np.dtype(np.uint16) for array in arrays.values()):
            raise ValueError(f"audited selected planes must be uint16 for {image_key}")
        if n_channels is None or int(n_channels) != 3:
            raise ValueError(f"audited acquisition must expose exactly 3 channels for {image_key}")


def _array_stats(values: np.ndarray, *, prefix: str) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return {
            f"{prefix}_mean_raw": np.nan,
            f"{prefix}_sum_raw": np.nan,
            f"{prefix}_median_raw": np.nan,
            f"{prefix}_sd_raw": np.nan,
            f"{prefix}_min_raw": np.nan,
            f"{prefix}_max_raw": np.nan,
        }
    return {
        f"{prefix}_mean_raw": float(finite.mean()),
        f"{prefix}_sum_raw": float(finite.sum()),
        f"{prefix}_median_raw": float(np.median(finite)),
        f"{prefix}_sd_raw": float(finite.std(ddof=0)),
        f"{prefix}_min_raw": float(finite.min()),
        f"{prefix}_max_raw": float(finite.max()),
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def _disk_or_annulus_mean(
    image: np.ndarray,
    center_y: int,
    center_x: int,
    inner_radius: float,
    outer_radius: float,
) -> float:
    radius = int(math.ceil(outer_radius))
    y0, y1 = max(0, center_y - radius), min(image.shape[0], center_y + radius + 1)
    x0, x1 = max(0, center_x - radius), min(image.shape[1], center_x + radius + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dist2 = (yy - center_y) ** 2 + (xx - center_x) ** 2
    keep = (dist2 <= outer_radius**2) & (dist2 > inner_radius**2)
    values = np.asarray(image[y0:y1, x0:x1], dtype=np.float64)[keep]
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def _local_window_values(
    image: np.ndarray,
    center_y: int,
    center_x: int,
    diameter_px: float,
    parameters: ExactFootprintParameters,
) -> np.ndarray:
    half = int(
        np.clip(
            round(float(diameter_px)) + parameters.window_pad_px,
            parameters.min_window_half,
            parameters.max_window_half,
        )
    )
    y0, y1 = max(0, center_y - half), min(image.shape[0], center_y + half + 1)
    x0, x1 = max(0, center_x - half), min(image.shape[1], center_x + half + 1)
    return np.asarray(image[y0:y1, x0:x1], dtype=np.float64)


def _null_defaults() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "null_candidate": False,
        "null_exclusion_reason": "",
        "null_seed": pd.NA,
        "null_n_requested": 0,
        "null_n_valid": 0,
        "null_max_redraw": 0,
        "placement_geometry": PLACEMENT_GEOMETRY,
        "null_mean_raw": np.nan,
        "null_sd_raw": np.nan,
        "null_min_raw": np.nan,
        "null_q05_raw": np.nan,
        "null_q25_raw": np.nan,
        "null_q50_raw": np.nan,
        "null_q75_raw": np.nan,
        "null_q90_raw": np.nan,
        "null_q95_raw": np.nan,
        "null_q99_raw": np.nan,
        "null_max_raw": np.nan,
        "null_delta_raw": np.nan,
        "null_enrichment_ratio": np.nan,
        "null_z": np.nan,
        "null_p_empirical": np.nan,
        "threshold_percentile": 95,
        "threshold_operator": "observed_qki_mean > own_linear_quantile",
        "qki_threshold_positive_q90": pd.NA,
        "qki_threshold_positive_q95": pd.NA,
        "qki_threshold_positive_q99": pd.NA,
        "qki_threshold_positive": pd.NA,
        "population_label_q90": "unusable",
        "population_label_q95": "unusable",
        "population_label_q99": "unusable",
        "null_usable": False,
        "population_label": "unusable",
        "first_pass_retention_mean": np.nan,
        "first_pass_retention_median": np.nan,
        "n_redraws": 0,
        "n_unplaceable": 0,
    }
    return payload


def _spot_metric_columns(parameters: ExactFootprintParameters) -> list[str]:
    hierarchy = [
        "image", "condition", "secondary_only", "is_control", "slide", "arm",
        "source_arm", "replicate", "fov", "biological_set", "catalog_folder",
        "eligible_for_sampling", "sampled_in_analysis",
    ]
    identity_and_plane = [
        "spot_uid", "image_key", "spot_id", "channel", "nucleus_uid",
        "nucleus_id", "analysis_run_id", "selected_z_1based",
        "selected_z_0based", "z_source", "n_z", "voxel_xy_nm", "voxel_z_nm",
        "miat_channel_index", "qki_channel_index", "dapi_channel_index",
        "quantitation_plane",
    ]
    geometry = [
        "x_px", "y_px", "center_x_px", "center_y_px",
        "spot_peak_intensity", "peak_intensity", "spot_diameter_um",
        "spot_diameter_px", "original_spot_area_px", "miat_floor_raw",
        "passes_miat_floor", "stored_in_nucleolus",
        "stored_in_nucleus_excluding_nucleolus",
        "recomputed_center_in_nucleolus", "nucleolar_center_discrepancy",
        "footprint_method", "footprint_fallback_reason", "footprint_area_px",
        "footprint_bbox_y0", "footprint_bbox_y1", "footprint_bbox_x0",
        "footprint_bbox_x1", "footprint_full_nucleus_valid",
        "footprint_full_nucleoplasm_valid", "footprint_invalid_reason",
        "footprint_rle_offset", "footprint_rle_n_runs",
    ]
    stat_suffixes = ("mean", "sum", "median", "sd", "min", "max")
    continuous = [
        "miat_center_raw",
        *(f"miat_footprint_{suffix}_raw" for suffix in stat_suffixes),
        "miat_local_window_bg_p10_raw",
        "miat_local_window_bg_median_raw",
        "miat_footprint_bgsub_mean_raw",
        "miat_footprint_bgsub_sum_raw",
        "qki_center_raw",
        *(f"qki_footprint_{suffix}_raw" for suffix in stat_suffixes),
        "miat_nucleus_mean_raw", "miat_nucleoplasm_mean_raw",
        "qki_nucleus_mean_raw", "qki_nucleoplasm_mean_raw",
        "miat_footprint_enrichment_vs_nucleus",
        "miat_footprint_enrichment_vs_nucleoplasm",
        "qki_footprint_enrichment_vs_nucleus",
        "qki_footprint_enrichment_vs_nucleoplasm",
        *(f"qki_disk_r{radius}_mean_raw" for radius in parameters.disk_radii_px),
        "qki_annulus_0_1_mean_raw", "qki_annulus_1_3_mean_raw",
        "qki_annulus_3_6_mean_raw", "sensitivity_metrics_role",
    ]
    return hierarchy + identity_and_plane + geometry + continuous + list(_null_defaults())


def summarize_spot_null_draws(
    observed_qki_raw: Any,
    draw_qki_raw: Any,
    *,
    require_all_draws: bool = True,
    min_valid_draws: int = 2,
) -> pd.DataFrame:
    """Summarize each spot's own null with linear q90/q95/q99 calls."""

    observed = np.asarray(observed_qki_raw, dtype=np.float64).reshape(-1)
    draws = np.asarray(draw_qki_raw, dtype=np.float64)
    if draws.ndim != 2 or draws.shape[0] != observed.size:
        raise ValueError("draw_qki_raw must have shape (n_spots, n_iterations)")
    n_requested = int(draws.shape[1])
    rows: list[dict[str, Any]] = []
    for obs, values in zip(observed, draws, strict=True):
        finite = values[np.isfinite(values)]
        usable = bool(
            np.isfinite(obs)
            and finite.size >= min_valid_draws
            and (not require_all_draws or finite.size == n_requested)
        )
        row = _null_defaults()
        row["null_n_requested"] = n_requested
        row["null_n_valid"] = int(finite.size)
        row["null_usable"] = usable
        if usable:
            quantiles = {
                percentile: float(np.percentile(finite, percentile, method="linear"))
                for percentile in (5, 25, 50, 75, 90, 95, 99)
            }
            mean = float(finite.mean())
            sd = float(finite.std(ddof=0))
            row.update(
                {
                    "null_mean_raw": mean,
                    "null_sd_raw": sd,
                    "null_min_raw": float(finite.min()),
                    "null_q05_raw": quantiles[5],
                    "null_q25_raw": quantiles[25],
                    "null_q50_raw": quantiles[50],
                    "null_q75_raw": quantiles[75],
                    "null_q90_raw": quantiles[90],
                    "null_q95_raw": quantiles[95],
                    "null_q99_raw": quantiles[99],
                    "null_max_raw": float(finite.max()),
                    "null_delta_raw": float(obs - mean),
                    "null_enrichment_ratio": _safe_ratio(float(obs), mean),
                    "null_z": float((obs - mean) / sd) if sd > 0 else np.nan,
                    "null_p_empirical": float(
                        (1 + np.count_nonzero(finite >= obs)) / (1 + finite.size)
                    ),
                    "qki_threshold_positive_q90": bool(obs > quantiles[90]),
                    "qki_threshold_positive_q95": bool(obs > quantiles[95]),
                    "qki_threshold_positive_q99": bool(obs > quantiles[99]),
                    "qki_threshold_positive": bool(obs > quantiles[95]),
                }
            )
            for percentile in THRESHOLD_PERCENTILES:
                row[f"population_label_q{percentile}"] = (
                    "threshold_positive"
                    if row[f"qki_threshold_positive_q{percentile}"]
                    else "threshold_negative"
                )
            row["population_label"] = row["population_label_q95"]
        rows.append(row)
    return pd.DataFrame(rows)


def validate_spot_population_invariants(spots: pd.DataFrame) -> dict[str, int]:
    """Fail closed on q90/q95/q99, q95-alias, and unusable semantics."""

    required = {
        "passes_miat_floor", "null_usable", "qki_threshold_positive",
        "population_label",
        *(f"qki_threshold_positive_q{p}" for p in THRESHOLD_PERCENTILES),
        *(f"population_label_q{p}" for p in THRESHOLD_PERCENTILES),
    }
    _require_columns(spots, required, table="spot population table")
    counts = {
        "n_rows": int(len(spots)),
        "n_below_floor": 0,
        "n_q90_positive": 0,
        "n_q95_positive": 0,
        "n_q99_positive": 0,
        "n_unusable": 0,
    }
    for index, row in spots.iterrows():
        passes_floor = _explicit_bool(row["passes_miat_floor"])
        usable = _explicit_bool(row["null_usable"])
        calls: dict[int, bool | None] = {}
        for percentile in THRESHOLD_PERCENTILES:
            value = row[f"qki_threshold_positive_q{percentile}"]
            calls[percentile] = None if pd.isna(value) else _explicit_bool(value)
        if not passes_floor:
            counts["n_below_floor"] += 1
            if usable or any(value is not None for value in calls.values()):
                raise ValueError(f"below-floor row {index} cannot have a usable null/call")
            labels = {
                str(row[f"population_label_q{p}"]) for p in THRESHOLD_PERCENTILES
            }
            if labels != {"below_miat_floor"} or str(row["population_label"]) != "below_miat_floor":
                raise ValueError(f"below-floor row {index} has inconsistent population labels")
            continue
        if not usable:
            counts["n_unusable"] += 1
            if any(value is not None for value in calls.values()):
                raise ValueError(f"unusable row {index} has a threshold call")
            labels = {
                str(row[f"population_label_q{p}"]) for p in THRESHOLD_PERCENTILES
            }
            if labels != {"unusable"} or str(row["population_label"]) != "unusable":
                raise ValueError(f"unusable row {index} leaked into positive/negative")
            if not pd.isna(row["qki_threshold_positive"]):
                raise ValueError(f"unusable row {index} has a primary threshold call")
            continue
        if any(value is None for value in calls.values()):
            raise ValueError(f"usable row {index} lacks q90/q95/q99 calls")
        if bool(calls[99]) and not bool(calls[95]) or bool(calls[95]) and not bool(calls[90]):
            raise ValueError(
                "threshold monotonicity q99 subset q95 subset q90 failed "
                f"for row {index}"
            )
        for percentile, call in calls.items():
            expected = "threshold_positive" if call else "threshold_negative"
            if str(row[f"population_label_q{percentile}"]) != expected:
                raise ValueError(f"q{percentile} label/call mismatch for row {index}")
            counts[f"n_q{percentile}_positive"] += int(bool(call))
        primary = row["qki_threshold_positive"]
        if pd.isna(primary) or _explicit_bool(primary) != bool(calls[95]):
            raise ValueError(f"q95 primary-call alias mismatch for row {index}")
        if str(row["population_label"]) != str(row["population_label_q95"]):
            raise ValueError(f"q95 primary-label alias mismatch for row {index}")
    return counts


def reconstruct_image_footprints(
    miat_2d: np.ndarray,
    qki_2d: np.ndarray,
    nucleus_labels: np.ndarray,
    nucleolus_labels: np.ndarray | None,
    spots: pd.DataFrame,
    image_metadata: Mapping[str, Any],
    *,
    parameters: ExactFootprintParameters | None = None,
    run_id: str,
    compute_null: bool = True,
    nucleus_hierarchy: pd.DataFrame | None = None,
) -> ReconstructionResult:
    """Reconstruct and quantify every persisted RNA1 spot in one image."""

    params = parameters or ExactFootprintParameters()
    miat = np.asarray(miat_2d)
    qki = np.asarray(qki_2d)
    labels = np.asarray(nucleus_labels)
    if miat.ndim != 2 or qki.shape != miat.shape or labels.shape != miat.shape:
        raise ValueError("MIAT, QKI, and nucleus labels must be same-shape 2D arrays")
    if nucleolus_labels is None:
        nucleoli = np.zeros_like(labels, dtype=np.int32)
    else:
        nucleoli = np.asarray(nucleolus_labels)
        if nucleoli.shape != labels.shape:
            raise ValueError("nucleolus labels must match nucleus labels")
    _require_columns(
        spots,
        {
            "spot_id",
            "channel",
            "nucleus_id",
            "x_px",
            "y_px",
            "spot_diameter_um",
            "spot_peak_intensity",
            "in_nucleolus",
        },
        table="RNA1 spot table",
    )
    work = spots.loc[spots["channel"].astype(str).eq("rna1")].copy().reset_index(drop=True)
    voxel_xy_nm = float(image_metadata["voxel_xy_nm"])
    diameters_px = diameter_um_to_px(work["spot_diameter_um"], voxel_xy_nm=voxel_xy_nm)
    spot_yx = work[["y_px", "x_px"]].to_numpy(dtype=float)
    footprints = build_miat_footprints(
        miat,
        spot_yx,
        diameters_px,
        partner_2d=qki,
        valid_mask=np.ones(miat.shape, dtype=bool),
        half_max_frac=params.half_max_frac,
        bg_percentile=params.background_percentile,
        window_pad_px=params.window_pad_px,
        min_window_half=params.min_window_half,
        max_window_half=params.max_window_half,
        peak_search_half=params.peak_search_half,
    )
    image_key = str(image_metadata.get("image_key", image_metadata.get("image", ""))).casefold()
    sampling_by_nucleus: dict[int, tuple[bool, bool]] = {}
    if nucleus_hierarchy is not None:
        _require_columns(
            nucleus_hierarchy,
            {
                "image_key",
                "nucleus_id",
                "eligible_for_sampling",
                "sampled_in_analysis",
            },
            table="nucleus hierarchy",
        )
        scoped_hierarchy = nucleus_hierarchy.loc[
            nucleus_hierarchy["image_key"].astype(str).str.casefold().eq(image_key)
        ].copy()
        if scoped_hierarchy.empty:
            raise ValueError(f"nucleus hierarchy has no rows for image {image_key!r}")
        for hierarchy_nucleus_id, group in scoped_hierarchy.groupby(
            "nucleus_id", sort=False, dropna=False
        ):
            nucleus_key = _finite_integer(
                hierarchy_nucleus_id, name="hierarchy nucleus_id", minimum=1
            )
            flags: list[bool] = []
            for field in ("eligible_for_sampling", "sampled_in_analysis"):
                values = group[field].map(_explicit_bool).unique()
                if len(values) != 1:
                    raise ValueError(
                        f"nucleus hierarchy is inconsistent for {image_key!r} "
                        f"nucleus {nucleus_key}: {field}"
                    )
                flags.append(bool(values[0]))
            sampling_by_nucleus[nucleus_key] = (flags[0], flags[1])
    overlap = np.zeros(miat.size, dtype=np.int32)
    for fp in footprints:
        overlap[fp.y_px * miat.shape[1] + fp.x_px] += 1
    spot_rows: list[dict[str, Any]] = []
    pixel_rows: list[dict[str, Any]] = []
    footprint_by_spot_uid: dict[str, MiatFootprint] = {}
    rle_offset = 0
    nucleolar_discrepancies = 0
    for position, (source, fp, diameter_px) in enumerate(
        zip(work.to_dict("records"), footprints, diameters_px, strict=True)
    ):
        nucleus_id = _finite_integer(source["nucleus_id"], name="nucleus_id", minimum=0)
        spot_id = _finite_integer(source["spot_id"], name="spot_id")
        spot_uid = f"{image_key}:rna1:{spot_id}"
        nucleus_uid = f"{image_key}:nucleus:{nucleus_id}"
        parent = labels == nucleus_id if nucleus_id > 0 else np.zeros_like(labels, dtype=bool)
        parent_nucleolus = (nucleoli == nucleus_id) & parent
        nucleoplasm = parent & ~parent_nucleolus
        full_nucleus_valid = full_footprint_is_valid(fp, parent)
        full_nucleoplasm_valid = full_footprint_is_valid(fp, nucleoplasm)
        center_y, center_x = fp.center_y_px, fp.center_x_px
        stored_nucleolar = _explicit_bool(source.get("in_nucleolus", False))
        recomputed_nucleolar = bool(
            nucleus_id > 0 and nucleoli[center_y, center_x] == nucleus_id
        )
        nucleolar_discrepancies += int(stored_nucleolar != recomputed_nucleolar)
        miat_values = np.asarray(miat[fp.y_px, fp.x_px], dtype=np.float64)
        qki_values = np.asarray(qki[fp.y_px, fp.x_px], dtype=np.float64)
        nuc_qki = np.asarray(qki[parent], dtype=np.float64)
        np_qki = np.asarray(qki[nucleoplasm], dtype=np.float64)
        nuc_miat = np.asarray(miat[parent], dtype=np.float64)
        np_miat = np.asarray(miat[nucleoplasm], dtype=np.float64)
        local = _local_window_values(miat, center_y, center_x, float(diameter_px), params)
        local_finite = local[np.isfinite(local)]
        bg_p10 = float(np.percentile(local_finite, 10)) if local_finite.size else np.nan
        bg_median = float(np.median(local_finite)) if local_finite.size else np.nan
        miat_stats = _array_stats(miat_values, prefix="miat_footprint")
        qki_stats = _array_stats(qki_values, prefix="qki_footprint")
        rle = encode_footprint_rle(fp, image_shape=miat.shape)
        passes_floor = bool(float(source["spot_peak_intensity"]) >= params.miat_floor_raw)
        invalid_reason = fp.invalid_reason or ""
        if not invalid_reason and not full_nucleus_valid:
            invalid_reason = "footprint_not_full_parent_nucleus"
        elif not invalid_reason and not bool(np.isfinite(qki_values).all()):
            invalid_reason = "nonfinite_qki_footprint"
        if not passes_floor:
            null_exclusion = "below_miat_floor"
        elif stored_nucleolar:
            null_exclusion = "nucleolar_spot"
        elif invalid_reason:
            null_exclusion = invalid_reason
        elif not full_nucleoplasm_valid:
            null_exclusion = "footprint_not_full_nucleoplasm"
        else:
            null_exclusion = "" if compute_null else "null_not_computed"
        null_candidate = bool(passes_floor and not stored_nucleolar and full_nucleoplasm_valid and not invalid_reason)
        if nucleus_hierarchy is not None:
            if nucleus_id not in sampling_by_nucleus:
                raise ValueError(
                    f"nucleus hierarchy is missing {image_key!r} nucleus {nucleus_id}"
                )
            eligible_for_sampling, sampled_in_analysis = sampling_by_nucleus[nucleus_id]
        else:
            eligible_for_sampling = _explicit_bool(
                image_metadata.get(
                    "eligible_for_sampling", source.get("eligible_for_sampling", False)
                )
            )
            sampled_in_analysis = _explicit_bool(
                image_metadata.get(
                    "sampled_in_analysis", source.get("sampled_in_analysis", False)
                )
            )
        row: dict[str, Any] = {
            **{key: image_metadata.get(key, source.get(key, "")) for key in (
                "image", "condition", "secondary_only", "is_control", "slide", "arm",
                "source_arm", "replicate", "fov", "biological_set", "catalog_folder",
            )},
            "eligible_for_sampling": eligible_for_sampling,
            "sampled_in_analysis": sampled_in_analysis,
            "spot_uid": spot_uid,
            "image_key": image_key,
            "spot_id": spot_id,
            "channel": "rna1",
            "nucleus_uid": nucleus_uid,
            "nucleus_id": nucleus_id,
            "analysis_run_id": str(image_metadata.get("analysis_run_id", run_id)),
            "selected_z_1based": int(image_metadata.get("selected_z_1based", 0)),
            "selected_z_0based": int(image_metadata.get(
                "selected_z_0based",
                int(image_metadata.get("selected_z_1based", 0)) - 1,
            )),
            "z_source": str(image_metadata.get("z_source", "per_image_summary.z_plane")),
            "n_z": int(image_metadata.get("n_z", 0)),
            "voxel_xy_nm": voxel_xy_nm,
            "voxel_z_nm": float(image_metadata.get("voxel_z_nm", np.nan)),
            "miat_channel_index": int(image_metadata.get("miat_channel_index", 0)),
            "qki_channel_index": int(image_metadata.get("qki_channel_index", 1)),
            "dapi_channel_index": int(image_metadata.get("dapi_channel_index", 2)),
            "quantitation_plane": "exact_recorded_single_z",
            "x_px": float(source["x_px"]),
            "y_px": float(source["y_px"]),
            "center_x_px": center_x,
            "center_y_px": center_y,
            "spot_peak_intensity": float(source["spot_peak_intensity"]),
            "peak_intensity": float(source.get("peak_intensity", source["spot_peak_intensity"])),
            "spot_diameter_um": float(source["spot_diameter_um"]),
            "spot_diameter_px": float(diameter_px),
            "original_spot_area_px": float(source.get("spot_area_px", np.nan)),
            "miat_floor_raw": params.miat_floor_raw,
            "passes_miat_floor": passes_floor,
            "stored_in_nucleolus": stored_nucleolar,
            "stored_in_nucleus_excluding_nucleolus": _explicit_bool(
                source.get("in_nucleus_excluding_nucleolus", not stored_nucleolar)
            ),
            "recomputed_center_in_nucleolus": recomputed_nucleolar,
            "nucleolar_center_discrepancy": stored_nucleolar != recomputed_nucleolar,
            "footprint_method": fp.method,
            "footprint_fallback_reason": fp.fallback_reason or "",
            "footprint_area_px": fp.area_px,
            "footprint_bbox_y0": int(fp.y_px.min()) if fp.area_px else pd.NA,
            "footprint_bbox_y1": int(fp.y_px.max() + 1) if fp.area_px else pd.NA,
            "footprint_bbox_x0": int(fp.x_px.min()) if fp.area_px else pd.NA,
            "footprint_bbox_x1": int(fp.x_px.max() + 1) if fp.area_px else pd.NA,
            "footprint_full_nucleus_valid": full_nucleus_valid,
            "footprint_full_nucleoplasm_valid": full_nucleoplasm_valid,
            "footprint_invalid_reason": invalid_reason,
            "footprint_rle_offset": rle_offset,
            "footprint_rle_n_runs": int(len(rle)),
            "miat_center_raw": float(miat[center_y, center_x]),
            **miat_stats,
            "miat_local_window_bg_p10_raw": bg_p10,
            "miat_local_window_bg_median_raw": bg_median,
            "miat_footprint_bgsub_mean_raw": float(miat_stats["miat_footprint_mean_raw"] - bg_p10),
            "miat_footprint_bgsub_sum_raw": float(
                miat_stats["miat_footprint_sum_raw"] - bg_p10 * fp.area_px
            ),
            "qki_center_raw": float(qki[center_y, center_x]),
            **qki_stats,
            "miat_nucleus_mean_raw": float(np.mean(nuc_miat)) if nuc_miat.size else np.nan,
            "miat_nucleoplasm_mean_raw": float(np.mean(np_miat)) if np_miat.size else np.nan,
            "qki_nucleus_mean_raw": float(np.mean(nuc_qki)) if nuc_qki.size else np.nan,
            "qki_nucleoplasm_mean_raw": float(np.mean(np_qki)) if np_qki.size else np.nan,
        }
        row["miat_footprint_enrichment_vs_nucleus"] = _safe_ratio(
            row["miat_footprint_mean_raw"], row["miat_nucleus_mean_raw"]
        )
        row["miat_footprint_enrichment_vs_nucleoplasm"] = _safe_ratio(
            row["miat_footprint_mean_raw"], row["miat_nucleoplasm_mean_raw"]
        )
        row["qki_footprint_enrichment_vs_nucleus"] = _safe_ratio(
            row["qki_footprint_mean_raw"], row["qki_nucleus_mean_raw"]
        )
        row["qki_footprint_enrichment_vs_nucleoplasm"] = _safe_ratio(
            row["qki_footprint_mean_raw"], row["qki_nucleoplasm_mean_raw"]
        )
        for radius in params.disk_radii_px:
            row[f"qki_disk_r{radius}_mean_raw"] = _disk_or_annulus_mean(
                qki, center_y, center_x, -1.0, float(radius)
            )
        for inner, outer in ((0, 1), (1, 3), (3, 6)):
            row[f"qki_annulus_{inner}_{outer}_mean_raw"] = _disk_or_annulus_mean(
                qki, center_y, center_x, float(inner), float(outer)
            )
        row["sensitivity_metrics_role"] = "non_primary_radius_sensitivity"
        row.update(_null_defaults())
        row["null_candidate"] = null_candidate
        row["null_exclusion_reason"] = null_exclusion
        if not passes_floor:
            for percentile in THRESHOLD_PERCENTILES:
                row[f"population_label_q{percentile}"] = "below_miat_floor"
            row["population_label"] = "below_miat_floor"
        spot_rows.append(row)
        footprint_by_spot_uid[spot_uid] = fp
        for pixel_index, (y_px, x_px, dy_px, dx_px) in enumerate(
            zip(fp.y_px, fp.x_px, fp.dy_px, fp.dx_px, strict=True)
        ):
            flat = int(y_px * miat.shape[1] + x_px)
            pixel_rows.append(
                {
                    "spot_uid": spot_uid,
                    "pixel_index": pixel_index,
                    "flat_pixel_index": flat,
                    "y_px": int(y_px),
                    "x_px": int(x_px),
                    "dy_px": int(dy_px),
                    "dx_px": int(dx_px),
                    "miat_raw": float(miat[y_px, x_px]),
                    "qki_raw": float(qki[y_px, x_px]),
                    "nucleus_label": int(labels[y_px, x_px]),
                    "nucleolus_label": int(nucleoli[y_px, x_px]),
                    "in_parent_nucleus": bool(labels[y_px, x_px] == nucleus_id),
                    "in_parent_nucleoplasm": bool(
                        labels[y_px, x_px] == nucleus_id and nucleoli[y_px, x_px] != nucleus_id
                    ),
                    "footprint_overlap_multiplicity": int(overlap[flat]),
                }
            )
        rle_offset += len(rle)

    spot_table = pd.DataFrame(spot_rows, columns=_spot_metric_columns(params))
    pixel_table = pd.DataFrame(pixel_rows)
    if pixel_table.empty:
        pixel_table = pd.DataFrame(
            columns=[
                "spot_uid", "pixel_index", "flat_pixel_index", "y_px", "x_px",
                "dy_px", "dx_px", "miat_raw", "qki_raw", "nucleus_label",
                "nucleolus_label", "in_parent_nucleus", "in_parent_nucleoplasm",
                "footprint_overlap_multiplicity",
            ]
        )
    null_groups: dict[int, dict[str, Any]] = {}
    footprints_by_nucleus: dict[int, tuple[MiatFootprint, ...]] = {}
    if not spot_table.empty:
        for nucleus_id, indexes in spot_table.groupby("nucleus_id", sort=False).groups.items():
            candidate_positions = [int(i) for i in indexes if bool(spot_table.at[i, "null_candidate"])]
            candidate_footprints = tuple(
                footprint_by_spot_uid[str(spot_table.at[i, "spot_uid"])] for i in candidate_positions
            )
            footprints_by_nucleus[int(nucleus_id)] = candidate_footprints
            if not compute_null:
                continue
            if len(candidate_positions) < 2:
                reason = "insufficient_spots_for_rotation"
                spot_table.loc[candidate_positions, "null_exclusion_reason"] = reason
                continue
            seed = stable_nucleus_seed(run_id, image_key, int(nucleus_id), params.global_seed)
            valid_mask = (
                (labels == int(nucleus_id))
                & (nucleoli != int(nucleus_id))
                & np.isfinite(qki)
            )
            null_result = keep_n_footprint_rotation_null(
                qki,
                candidate_footprints,
                valid_mask,
                n_null=params.n_null,
                rng=np.random.default_rng(seed),
                passes_miat_floor=[True] * len(candidate_footprints),
                threshold_percentile=float(params.primary_threshold_percentile),
                min_first_pass_retention=params.min_first_pass_retention,
                min_valid_draw_fraction=params.min_valid_draw_fraction,
                min_valid_draws=params.min_valid_draws,
                max_redraw=params.max_redraw,
            )
            summaries = summarize_spot_null_draws(
                [fp.observed_qki_raw for fp in candidate_footprints],
                null_result.null_qki_raw,
                require_all_draws=params.min_valid_draw_fraction == 1.0,
                min_valid_draws=params.min_valid_draws,
            )
            for local_index, table_index in enumerate(candidate_positions):
                summary = summaries.iloc[local_index].to_dict()
                # These two fields encode upstream eligibility/provenance and
                # must never be reset by the generic null-summary defaults.
                summary.pop("null_candidate", None)
                summary.pop("null_exclusion_reason", None)
                engine_spot = null_result.spots[local_index]
                usable = bool(summary["null_usable"] and engine_spot.null_usable)
                if not usable:
                    summary.update(
                        {
                            "qki_threshold_positive_q90": pd.NA,
                            "qki_threshold_positive_q95": pd.NA,
                            "qki_threshold_positive_q99": pd.NA,
                            "qki_threshold_positive": pd.NA,
                            "population_label_q90": "unusable",
                            "population_label_q95": "unusable",
                            "population_label_q99": "unusable",
                            "population_label": "unusable",
                        }
                    )
                summary["null_usable"] = usable
                summary["null_seed"] = seed
                summary["null_max_redraw"] = params.max_redraw
                summary["first_pass_retention_mean"] = null_result.mean_first_pass_retention
                summary["first_pass_retention_median"] = null_result.median_first_pass_retention
                summary["n_redraws"] = int(null_result.redraw_counts[local_index].sum())
                summary["n_unplaceable"] = int(
                    np.count_nonzero(
                        ~np.isfinite(null_result.null_qki_raw[local_index])
                    )
                )
                for key, value in summary.items():
                    spot_table.at[table_index, key] = value
                spot_table.at[table_index, "null_exclusion_reason"] = (
                    ""
                    if usable
                    else engine_spot.invalid_reason
                    or null_result.invalid_reason
                    or "null_unusable"
                )
            null_groups[int(nucleus_id)] = {
                "seed": seed,
                "spot_uid": spot_table.loc[candidate_positions, "spot_uid"].astype(str).to_numpy(),
                "observed_qki_raw": np.asarray(
                    [fp.observed_qki_raw for fp in candidate_footprints], dtype=np.float64
                ),
                "draw_qki_raw": null_result.null_qki_raw,
                "initial_angles_deg": null_result.initial_angles_deg,
                "placement_angles_deg": null_result.placement_angles_deg,
                "first_pass_valid": null_result.first_pass_valid,
                "redraw_counts": null_result.redraw_counts,
                "null_usable": null_result.usable,
                "unusable_reason": null_result.invalid_reason or "",
                "first_pass_retention_mean": null_result.mean_first_pass_retention,
                "first_pass_retention_median": null_result.median_first_pass_retention,
            }
    population_audit = validate_spot_population_invariants(spot_table)
    audit = {
        "n_input_spots": int(len(work)),
        "n_reconstructed_spots": int(len(spot_table)),
        "n_footprint_pixels": int(len(pixel_table)),
        "nucleolar_center_discrepancy_count": int(nucleolar_discrepancies),
        "same_plane_quantitation": True,
        "segmentation_reused": True,
        "spot_detection_reused": True,
        **population_audit,
    }
    return ReconstructionResult(
        spot_metrics=spot_table,
        pixel_metrics=pixel_table,
        footprints_by_nucleus=footprints_by_nucleus,
        null_groups=null_groups,
        audit=audit,
        planes={"miat": miat, "qki": qki},
        nucleus_labels=labels,
        nucleolus_labels=nucleoli,
    )


def calibrate_reconstruction_nulls(
    reconstruction: ReconstructionResult,
    *,
    parameters: ExactFootprintParameters,
    run_id: str,
) -> ReconstructionResult:
    """Apply KEEP-N nulls after the complete historical parity gate passes."""

    table = reconstruction.spot_metrics
    qki = reconstruction.planes["qki"]
    labels = reconstruction.nucleus_labels
    nucleoli = reconstruction.nucleolus_labels
    image_key = str(table["image_key"].iloc[0]).casefold() if len(table) else ""
    reconstruction.null_groups.clear()
    for nucleus_id, footprints in reconstruction.footprints_by_nucleus.items():
        candidate_index = table.index[
            table["nucleus_id"].eq(nucleus_id)
            & table["null_candidate"].map(_explicit_bool)
        ].tolist()
        if len(candidate_index) != len(footprints):
            raise ValueError(
                f"candidate footprint alignment failed for nucleus {nucleus_id}"
            )
        if len(candidate_index) < 2:
            table.loc[candidate_index, "null_exclusion_reason"] = (
                "insufficient_spots_for_rotation"
            )
            continue
        seed = stable_nucleus_seed(
            run_id, image_key, int(nucleus_id), parameters.global_seed
        )
        valid_mask = (
            (labels == int(nucleus_id))
            & (nucleoli != int(nucleus_id))
            & np.isfinite(qki)
        )
        null_result = keep_n_footprint_rotation_null(
            qki,
            footprints,
            valid_mask,
            n_null=parameters.n_null,
            rng=np.random.default_rng(seed),
            passes_miat_floor=[True] * len(footprints),
            threshold_percentile=float(parameters.primary_threshold_percentile),
            min_first_pass_retention=parameters.min_first_pass_retention,
            min_valid_draw_fraction=parameters.min_valid_draw_fraction,
            min_valid_draws=parameters.min_valid_draws,
            max_redraw=parameters.max_redraw,
        )
        summaries = summarize_spot_null_draws(
            [footprint.observed_qki_raw for footprint in footprints],
            null_result.null_qki_raw,
            require_all_draws=parameters.min_valid_draw_fraction == 1.0,
            min_valid_draws=parameters.min_valid_draws,
        )
        for local_index, table_index in enumerate(candidate_index):
            summary = summaries.iloc[local_index].to_dict()
            # Preserve the eligibility decision and write the explicit
            # exclusion provenance only after all numeric summary fields.
            summary.pop("null_candidate", None)
            summary.pop("null_exclusion_reason", None)
            engine_spot = null_result.spots[local_index]
            usable = bool(summary["null_usable"] and engine_spot.null_usable)
            if not usable:
                summary.update(
                    {
                        "qki_threshold_positive_q90": pd.NA,
                        "qki_threshold_positive_q95": pd.NA,
                        "qki_threshold_positive_q99": pd.NA,
                        "qki_threshold_positive": pd.NA,
                        "population_label_q90": "unusable",
                        "population_label_q95": "unusable",
                        "population_label_q99": "unusable",
                        "population_label": "unusable",
                    }
                )
            summary.update(
                {
                    "null_usable": usable,
                    "null_seed": seed,
                    "null_max_redraw": parameters.max_redraw,
                    "first_pass_retention_mean": null_result.mean_first_pass_retention,
                    "first_pass_retention_median": null_result.median_first_pass_retention,
                    "n_redraws": int(null_result.redraw_counts[local_index].sum()),
                    "n_unplaceable": int(
                        np.count_nonzero(
                            ~np.isfinite(null_result.null_qki_raw[local_index])
                        )
                    ),
                }
            )
            for key, value in summary.items():
                table.at[table_index, key] = value
            table.at[table_index, "null_exclusion_reason"] = (
                ""
                if usable
                else engine_spot.invalid_reason
                or null_result.invalid_reason
                or "null_unusable"
            )
        reconstruction.null_groups[int(nucleus_id)] = {
            "seed": seed,
            "spot_uid": table.loc[candidate_index, "spot_uid"].astype(str).to_numpy(),
            "observed_qki_raw": np.asarray(
                [footprint.observed_qki_raw for footprint in footprints],
                dtype=np.float64,
            ),
            "draw_qki_raw": null_result.null_qki_raw,
            "initial_angles_deg": null_result.initial_angles_deg,
            "placement_angles_deg": null_result.placement_angles_deg,
            "first_pass_valid": null_result.first_pass_valid,
            "redraw_counts": null_result.redraw_counts,
            "null_usable": null_result.usable,
            "unusable_reason": null_result.invalid_reason or "",
            "first_pass_retention_mean": null_result.mean_first_pass_retention,
            "first_pass_retention_median": null_result.median_first_pass_retention,
            "unplaceable_count": null_result.unplaceable_count,
        }
    reconstruction.spot_metrics = table
    validate_spot_population_invariants(table)
    return reconstruction


def validate_historical_parity(
    reconstructed: pd.DataFrame,
    historical_spots: pd.DataFrame,
    *,
    qki_rtol: float = 1e-12,
    qki_atol: float = 1e-12,
) -> dict[str, Any]:
    """Hard gate exact area and historical QKI footprint mean parity."""

    keys = ["image_key", "channel", "spot_id"]
    historical_rna1 = historical_spots
    if "channel" in historical_rna1.columns:
        historical_rna1 = historical_rna1.loc[
            historical_rna1["channel"].astype(str).eq("rna1")
        ]
    if reconstructed.empty and historical_rna1.empty:
        return {"n_spots": 0, "area_exact": True, "qki_allclose": True}
    _require_columns(
        reconstructed,
        {*keys, "footprint_area_px", "qki_footprint_mean_raw"},
        table="reconstructed spots",
    )
    historical = historical_spots.copy()
    if "image_key" not in historical:
        historical["image_key"] = historical["image"].map(_casefold_basename)
    _require_columns(
        historical,
        {*keys, "miat_footprint_area_px", "qki_at_miat_footprint"},
        table="historical spots",
    )
    merged = reconstructed.merge(
        historical[keys + ["miat_footprint_area_px", "qki_at_miat_footprint"]],
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("footprint parity key mismatch between reconstructed and historical spots")
    reconstructed_area = pd.to_numeric(
        merged["footprint_area_px"], errors="coerce"
    ).to_numpy(float)
    historical_area = pd.to_numeric(
        merged["miat_footprint_area_px"], errors="coerce"
    ).to_numpy(float)
    area_exact = bool(
        np.isfinite(reconstructed_area).all()
        and np.isfinite(historical_area).all()
        and np.equal(reconstructed_area, np.rint(reconstructed_area)).all()
        and np.equal(historical_area, np.rint(historical_area)).all()
        and np.array_equal(reconstructed_area, historical_area)
    )
    if not area_exact:
        raise ValueError("MIAT footprint area parity failed")
    qki_allclose = bool(
        np.allclose(
            pd.to_numeric(merged["qki_footprint_mean_raw"]).to_numpy(float),
            pd.to_numeric(merged["qki_at_miat_footprint"]).to_numpy(float),
            rtol=qki_rtol,
            atol=qki_atol,
            equal_nan=True,
        )
    )
    if not qki_allclose:
        raise ValueError("QKI footprint parity failed")
    return {"n_spots": int(len(merged)), "area_exact": True, "qki_allclose": True}


def _correlation_result(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    n = int(len(x))
    if n < 3:
        return {
            "n_spots": n,
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
            "estimable": False,
            "nonestimable_reason": "fewer_than_3_spots",
        }
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return {
            "n_spots": n,
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
            "estimable": False,
            "nonestimable_reason": "constant_vector",
        }
    pearson = stats.pearsonr(x, y)
    spearman = stats.spearmanr(x, y)
    return {
        "n_spots": n,
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "estimable": True,
        "nonestimable_reason": "",
    }


def correlation_records(spots: pd.DataFrame, *, nucleus_id: int) -> pd.DataFrame:
    """Continuous MIAT/QKI correlations for explicit, non-ambiguous populations."""

    frame = spots.loc[spots["nucleus_id"].eq(nucleus_id)].copy()
    valid = frame["footprint_full_nucleus_valid"].map(_explicit_bool)
    finite_raw = np.isfinite(pd.to_numeric(frame["miat_footprint_mean_raw"], errors="coerce")) & np.isfinite(
        pd.to_numeric(frame["qki_footprint_mean_raw"], errors="coerce")
    )
    non_nucleolar = ~frame["stored_in_nucleolus"].map(_explicit_bool)
    threshold_eligible = frame["null_candidate"].map(_explicit_bool)
    populations: list[tuple[str, pd.Series, bool]] = [
        ("all_detected_exact_valid", valid & finite_raw, False),
        (
            "non_nucleolar_exact_valid",
            valid & finite_raw & non_nucleolar & threshold_eligible,
            False,
        ),
    ]
    for percentile in THRESHOLD_PERCENTILES:
        column = f"qki_threshold_positive_q{percentile}"
        calls = frame[column].map(lambda value: value is True or isinstance(value, np.bool_) and bool(value))
        usable = frame["null_usable"].map(_explicit_bool)
        populations.append(
            (f"threshold_positive_q{percentile}", valid & finite_raw & usable & calls, True)
        )
    pairs = {
        "raw": ("miat_footprint_mean_raw", "qki_footprint_mean_raw"),
        "within_nucleus_normalized": (
            "miat_footprint_enrichment_vs_nucleus",
            "qki_footprint_enrichment_vs_nucleus",
        ),
    }
    rows: list[dict[str, Any]] = []
    for population, gate, conditional in populations:
        for pair_name, (miat_col, qki_col) in pairs.items():
            x = pd.to_numeric(frame.loc[gate, miat_col], errors="coerce").to_numpy(float)
            y = pd.to_numeric(frame.loc[gate, qki_col], errors="coerce").to_numpy(float)
            finite = np.isfinite(x) & np.isfinite(y)
            rows.append(
                {
                    "nucleus_id": int(nucleus_id),
                    "population": population,
                    "measurement_pair": pair_name,
                    "conditional_descriptive": conditional,
                    **_correlation_result(x[finite], y[finite]),
                }
            )
    return pd.DataFrame(rows)


def aggregate_nucleus_metrics(
    spots: pd.DataFrame,
    pixels: pd.DataFrame,
    nuclei: pd.DataFrame,
    *,
    miat_2d: np.ndarray,
    qki_2d: np.ndarray,
    nucleus_labels: np.ndarray,
    nucleolus_labels: np.ndarray,
) -> pd.DataFrame:
    """Left-join spot summaries onto the complete saved-nucleus universe."""

    labels = np.asarray(nucleus_labels)
    nucleoli = np.asarray(nucleolus_labels)
    rows: list[dict[str, Any]] = []
    for source in nuclei.to_dict("records"):
        image_key = str(source.get("image_key", "")).casefold()
        nucleus_id = int(source["nucleus_id"])
        group = spots.loc[
            spots["image_key"].astype(str).str.casefold().eq(image_key)
            & spots["nucleus_id"].eq(nucleus_id)
        ].copy()
        parent = labels == nucleus_id
        nucleoplasm = parent & (nucleoli != nucleus_id)
        floor = group["passes_miat_floor"].map(_explicit_bool) if len(group) else pd.Series(dtype=bool)
        usable = group["null_usable"].map(_explicit_bool) if len(group) else pd.Series(dtype=bool)
        calls_by_percentile = {
            percentile: (
                group[f"qki_threshold_positive_q{percentile}"].map(
                    lambda value: value is True
                    or isinstance(value, np.bool_) and bool(value)
                )
                & usable
                if len(group)
                else pd.Series(dtype=bool)
            )
            for percentile in THRESHOLD_PERCENTILES
        }
        positive = calls_by_percentile[95]
        labels_pop = group["population_label"].astype(str) if len(group) else pd.Series(dtype=str)
        all_mass = float(pd.to_numeric(group.get("miat_footprint_sum_raw"), errors="coerce").sum()) if len(group) else 0.0
        pos_mass = float(pd.to_numeric(group.loc[positive, "miat_footprint_sum_raw"], errors="coerce").sum()) if len(group) else 0.0
        group_uids = set(group["spot_uid"].astype(str)) if len(group) else set()
        pos_uids = set(group.loc[positive, "spot_uid"].astype(str)) if len(group) else set()
        p_all = pixels.loc[pixels["spot_uid"].astype(str).isin(group_uids)].drop_duplicates("flat_pixel_index")
        p_pos = pixels.loc[pixels["spot_uid"].astype(str).isin(pos_uids)].drop_duplicates("flat_pixel_index")
        all_union = float(pd.to_numeric(p_all.get("miat_raw"), errors="coerce").sum()) if len(p_all) else 0.0
        pos_union = float(pd.to_numeric(p_pos.get("miat_raw"), errors="coerce").sum()) if len(p_pos) else 0.0
        qki_all_mass = float(
            pd.to_numeric(group.get("qki_footprint_sum_raw"), errors="coerce").sum()
        ) if len(group) else 0.0
        qki_pos_mass = float(
            pd.to_numeric(
                group.loc[positive, "qki_footprint_sum_raw"], errors="coerce"
            ).sum()
        ) if len(group) else 0.0
        qki_all_union = float(
            pd.to_numeric(p_all.get("qki_raw"), errors="coerce").sum()
        ) if len(p_all) else 0.0
        qki_pos_union = float(
            pd.to_numeric(p_pos.get("qki_raw"), errors="coerce").sum()
        ) if len(p_pos) else 0.0
        usable_mass = float(pd.to_numeric(group.loc[usable, "miat_footprint_sum_raw"], errors="coerce").sum()) if len(group) else 0.0
        record: dict[str, Any] = dict(source)
        record.update(
            {
                "nucleus_uid": f"{image_key}:nucleus:{nucleus_id}",
                "n_spots_all": int(len(group)),
                "n_spots_floor": int(floor.sum()),
                "n_spots_nucleolar": int(group["stored_in_nucleolus"].map(_explicit_bool).sum()) if len(group) else 0,
                "n_spots_non_nucleolar": int((~group["stored_in_nucleolus"].map(_explicit_bool)).sum()) if len(group) else 0,
                "n_valid_exact_footprints": int(group["footprint_full_nucleus_valid"].map(_explicit_bool).sum()) if len(group) else 0,
                "n_non_nucleolar_exact_valid": int(
                    (
                        ~group["stored_in_nucleolus"].map(_explicit_bool)
                        & group["footprint_full_nucleus_valid"].map(_explicit_bool)
                    ).sum()
                ) if len(group) else 0,
                "n_null_candidate": int(group["null_candidate"].map(_explicit_bool).sum()) if len(group) else 0,
                "n_threshold_positive": int(labels_pop.eq("threshold_positive").sum()),
                "n_threshold_negative": int(labels_pop.eq("threshold_negative").sum()),
                "n_unusable": int(labels_pop.eq("unusable").sum()),
                "n_below_floor": int(labels_pop.eq("below_miat_floor").sum()),
                "threshold_positive_spots_per_nucleus": int(labels_pop.eq("threshold_positive").sum()),
                "miat_footprint_mass_all_spot_summed": all_mass,
                "miat_footprint_mass_positive_spot_summed": pos_mass,
                "miat_footprint_mass_all_union_deduplicated": all_union,
                "miat_footprint_mass_positive_union_deduplicated": pos_union,
                "qki_footprint_mass_all_spot_summed": qki_all_mass,
                "qki_footprint_mass_positive_spot_summed": qki_pos_mass,
                "qki_footprint_mass_all_union_deduplicated": qki_all_union,
                "qki_footprint_mass_positive_union_deduplicated": qki_pos_union,
                "association_fraction_among_usable": _safe_ratio(float(positive.sum()), float(usable.sum())),
                "association_fraction_among_all_floor_spots": _safe_ratio(float(positive.sum()), float(floor.sum())),
                "associated_miat_mass_fraction_among_usable": _safe_ratio(pos_mass, usable_mass),
                "associated_miat_mass_fraction_among_all": _safe_ratio(pos_mass, all_mass),
                "qki_footprint_mean_all": float(pd.to_numeric(group.get("qki_footprint_mean_raw"), errors="coerce").mean()) if len(group) else np.nan,
                "qki_footprint_mean_threshold_positive": float(pd.to_numeric(group.loc[positive, "qki_footprint_mean_raw"], errors="coerce").mean()) if positive.any() else np.nan,
                "qki_enrichment_mean_all": float(pd.to_numeric(group.get("qki_footprint_enrichment_vs_nucleus"), errors="coerce").mean()) if len(group) else np.nan,
                "qki_enrichment_mean_threshold_positive": float(pd.to_numeric(group.loc[positive, "qki_footprint_enrichment_vs_nucleus"], errors="coerce").mean()) if positive.any() else np.nan,
                "whole_nucleus_miat_sum_raw": float(np.asarray(miat_2d)[parent].sum()),
                "whole_nucleus_miat_mean_raw": float(np.asarray(miat_2d)[parent].mean()) if parent.any() else np.nan,
                "whole_nucleus_qki_sum_raw": float(np.asarray(qki_2d)[parent].sum()),
                "whole_nucleus_qki_mean_raw": float(np.asarray(qki_2d)[parent].mean()) if parent.any() else np.nan,
                "nucleoplasm_miat_sum_raw": float(np.asarray(miat_2d)[nucleoplasm].sum()),
                "nucleoplasm_miat_mean_raw": float(np.asarray(miat_2d)[nucleoplasm].mean()) if nucleoplasm.any() else np.nan,
                "nucleoplasm_qki_sum_raw": float(np.asarray(qki_2d)[nucleoplasm].sum()),
                "nucleoplasm_qki_mean_raw": float(np.asarray(qki_2d)[nucleoplasm].mean()) if nucleoplasm.any() else np.nan,
            }
        )
        record["population_reconciliation_pass"] = bool(
            record["n_spots_floor"]
            == record["n_threshold_positive"] + record["n_threshold_negative"] + record["n_unusable"]
            and record["n_spots_all"] == record["n_spots_floor"] + record["n_below_floor"]
        )
        for percentile in THRESHOLD_PERCENTILES:
            population_labels = (
                group[f"population_label_q{percentile}"].astype(str)
                if len(group)
                else pd.Series(dtype=str)
            )
            record[f"n_threshold_positive_q{percentile}"] = int(
                population_labels.eq("threshold_positive").sum()
            )
            record[f"n_threshold_negative_q{percentile}"] = int(
                population_labels.eq("threshold_negative").sum()
            )
            record[f"n_unusable_q{percentile}"] = int(
                population_labels.eq("unusable").sum()
            )
            record[f"population_reconciliation_pass_q{percentile}"] = bool(
                record["n_spots_floor"]
                == record[f"n_threshold_positive_q{percentile}"]
                + record[f"n_threshold_negative_q{percentile}"]
                + record[f"n_unusable_q{percentile}"]
                and record["n_spots_all"]
                == record["n_spots_floor"] + record["n_below_floor"]
            )
        if len(group):
            for corr in correlation_records(group, nucleus_id=nucleus_id).to_dict("records"):
                stem = f"corr_{corr['population']}_{corr['measurement_pair']}"
                for key in (
                    "n_spots", "pearson_r", "pearson_p", "spearman_rho", "spearman_p",
                    "estimable", "nonestimable_reason", "conditional_descriptive",
                ):
                    record[f"{stem}_{key}"] = corr[key]
        rows.append(record)
    return pd.DataFrame(rows)


def _h5_token(value: Any, *, keep_dot: bool = True) -> str:
    text = str(value).casefold()
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789_-" + ("." if keep_dot else ""))
    token = "".join(char if char in allowed else "_" for char in text).strip("_")
    if not token:
        token = "unnamed"
    if token != text:
        token = f"{token}__{hashlib.sha256(text.encode('utf-8')).hexdigest()[:10]}"
    return token


def write_null_h5_group(
    handle: Any,
    *,
    nucleus_uid: str,
    image_key: str,
    nucleus_id: int,
    selected_z_1based: int,
    seed: int,
    spot_uid: Any,
    observed_qki_raw: Any,
    draw_qki_raw: Any,
    initial_angles_deg: Any,
    placement_angles_deg: Any,
    first_pass_valid: Any,
    redraw_counts: Any,
    null_usable: bool,
    unusable_reason: str,
    root: str = "nuclei",
    max_redraw: int | None = None,
    min_valid_draw_fraction: float | None = None,
    threshold_percentile: float = 95.0,
    first_pass_retention_mean: float | None = None,
    first_pass_retention_median: float | None = None,
) -> str:
    """Write one complete per-nucleus null group to an already-open HDF5 file."""

    path = f"{root.strip('/')}/{_h5_token(nucleus_uid, keep_dot=True)}"
    if path in handle:
        raise ValueError(f"HDF5 null group already exists: {path}")
    group = handle.create_group(path)
    string_dtype = __import__("h5py").string_dtype(encoding="utf-8")
    group.create_dataset("spot_uid", data=np.asarray(spot_uid, dtype=object), dtype=string_dtype)
    observed = np.asarray(observed_qki_raw, dtype=np.float64)
    draws = np.asarray(draw_qki_raw, dtype=np.float64)
    initial = np.asarray(initial_angles_deg, dtype=np.float64)
    placed = np.asarray(placement_angles_deg, dtype=np.float64)
    first = np.asarray(first_pass_valid, dtype=bool)
    redraw = np.asarray(redraw_counts, dtype=np.int32)
    if draws.ndim != 2 or draws.shape[0] != observed.size:
        raise ValueError("draw_qki_raw shape must match observed_qki_raw")
    expected = draws.shape
    if placed.shape != expected or first.shape != expected or redraw.shape != expected:
        raise ValueError("placement provenance matrices must match draw_qki_raw")
    if initial.shape != (draws.shape[1],):
        raise ValueError("initial_angles_deg must have one value per iteration")
    placement_code = np.full(expected, 2, dtype=np.uint8)
    finite = np.isfinite(draws)
    placement_code[first & finite] = 0
    placement_code[~first & finite] = 1
    compression = "gzip" if draws.size else None
    for name, data in (
        ("observed_qki_raw", observed),
        ("draw_qki_raw", draws),
        ("initial_angle_deg", initial),
        ("placement_angle_deg", placed),
        ("first_pass_valid", first),
        ("redraw_counts", redraw),
        ("placement_code", placement_code),
    ):
        kwargs = {"compression": compression, "shuffle": True} if np.asarray(data).ndim and np.asarray(data).size and compression else {}
        group.create_dataset(name, data=data, **kwargs)
    group.attrs.update(
        {
            "nucleus_uid": str(nucleus_uid),
            "image_key": str(image_key).casefold(),
            "nucleus_id": int(nucleus_id),
            "selected_z_1based": int(selected_z_1based),
            "seed": np.uint64(seed),
            "n_requested": int(draws.shape[1]),
            "n_iterations": int(draws.shape[1]),
            "threshold_percentile": float(threshold_percentile),
            "threshold_operator": "observed_qki_mean > own_linear_quantile",
            "quantile_method": "linear",
            "valid_mask_definition": "same parent nucleus AND outside recomputed nucleolus AND finite QKI",
            "placement_geometry": PLACEMENT_GEOMETRY,
            "redraw_fraction": float(np.count_nonzero(redraw > 0) / redraw.size)
            if redraw.size
            else 0.0,
            "unplaceable_fraction": float(np.count_nonzero(~finite) / finite.size)
            if finite.size
            else 0.0,
            "redraw_count_total": int(redraw.sum()),
            "unplaceable_count": int(np.count_nonzero(~finite)),
            "null_usable": bool(null_usable),
            "unusable_reason": str(unusable_reason),
            "complete": True,
        }
    )
    if max_redraw is not None:
        group.attrs["max_redraw"] = int(max_redraw)
    if min_valid_draw_fraction is not None:
        group.attrs["min_valid_draw_fraction"] = float(min_valid_draw_fraction)
        group.attrs["require_all_draws"] = bool(min_valid_draw_fraction == 1.0)
    if first_pass_retention_mean is not None:
        group.attrs["first_pass_retention_mean"] = float(
            first_pass_retention_mean
        )
    if first_pass_retention_median is not None:
        group.attrs["first_pass_retention_median"] = float(
            first_pass_retention_median
        )
    return path


def write_selected_plane_h5_group(
    handle: Any,
    *,
    image_key: str,
    selected_z_1based: int,
    planes: Mapping[str, np.ndarray],
    nucleus_labels: np.ndarray,
    nucleolus_labels: np.ndarray,
    channel_indices: Mapping[str, int],
) -> str:
    """Write the exact selected planes and reused/recomputed masks."""

    path = f"images/{_h5_token(image_key, keep_dot=True)}"
    if path in handle:
        raise ValueError(f"selected-plane HDF5 group already exists: {path}")
    required = {"miat", "qki", "dapi"}
    if set(planes) != required or set(channel_indices) != required:
        raise ValueError("planes and channel_indices must contain miat, qki, and dapi")
    arrays = {key: np.asarray(value) for key, value in planes.items()}
    shape = arrays["miat"].shape
    if any(array.shape != shape or array.ndim != 2 for array in arrays.values()):
        raise ValueError("all selected planes must be same-shape two-dimensional arrays")
    labels = np.asarray(nucleus_labels)
    nucleoli = np.asarray(nucleolus_labels)
    if labels.shape != shape or nucleoli.shape != shape:
        raise ValueError("saved masks must match selected planes")
    group = handle.create_group(path)
    for name, array in {**arrays, "nucleus_labels": labels, "nucleolus_labels": nucleoli}.items():
        group.create_dataset(name, data=array, compression="gzip", shuffle=True)
    group.attrs.update(
        {
            "image_key": str(image_key).casefold(),
            "selected_z_1based": int(selected_z_1based),
            "selected_z_0based": int(selected_z_1based) - 1,
            "miat_channel_index": int(channel_indices["miat"]),
            "qki_channel_index": int(channel_indices["qki"]),
            "dapi_channel_index": int(channel_indices["dapi"]),
            "same_plane_all_channels": True,
            "complete": True,
        }
    )
    return path


def create_timestamped_output_dir(
    output_root: str | Path,
    *,
    source_run_dir: str | Path,
    timestamp: str | None = None,
    prefix: str = "EXACT_FOOTPRINT_BACKFILL_",
) -> Path:
    """Create a new output directory and refuse the completed source run path."""

    stamp = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    root = Path(output_root)
    candidate = root / f"{prefix}{stamp}"
    source = Path(source_run_dir).resolve(strict=False)
    resolved_candidate = candidate.resolve(strict=False)
    if resolved_candidate == source or source in resolved_candidate.parents:
        raise ValueError(
            "output directory cannot equal or be inside the historical run"
        )
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _atomic_write_dataframe(
    frame: pd.DataFrame,
    path: Path,
    *,
    compression: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, compression=compression)
    os.replace(temporary, path)


def _atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _combine_csv_fragments(
    fragments: Sequence[Path],
    output_path: Path,
    *,
    compressed: bool,
) -> None:
    """Stream fragments into one table without materializing the project."""

    temporary = output_path.with_name(output_path.name + ".tmp")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    opener_out = gzip.open if compressed else open
    with opener_out(temporary, "wt", encoding="utf-8", newline="") as target:
        wrote_header = False
        for fragment in fragments:
            if not fragment.is_file() or fragment.stat().st_size == 0:
                continue
            opener_in = gzip.open if fragment.suffix == ".gz" else open
            with opener_in(fragment, "rt", encoding="utf-8", newline="") as source:
                header = source.readline()
                if not header.strip():
                    continue
                if not wrote_header:
                    target.write(header)
                    wrote_header = True
                first_data = source.readline()
                if not first_data:
                    continue
                target.write(first_data)
                for line in source:
                    target.write(line)
    os.replace(temporary, output_path)


def _default_nucleolus_builder(
    run_config: Mapping[str, Any],
    labels: np.ndarray,
    dapi_2d: np.ndarray,
    voxel_xy_nm: float,
) -> np.ndarray:
    from ..config.schema import FishsuiteConfig
    from .coloc_backfill import _build_nucleolus_labels

    cfg = FishsuiteConfig.model_validate(run_config["config_resolved"])
    required = bool(
        getattr(cfg.foci, "exclude_nucleolus_from_partner_null", False)
        and getattr(getattr(cfg, "nucleolus", None), "enabled", False)
    )
    result = _build_nucleolus_labels(cfg, labels, dapi_2d, voxel_xy_nm)
    if result is None and required:
        raise RuntimeError(
            "nucleolus reconstruction returned no label image while exclusion "
            "is required; refusing to run a whole-nucleus null"
        )
    if result is None:
        return np.zeros_like(labels, dtype=np.int32)
    return np.asarray(result, dtype=np.int32)


def _null_draw_index(
    image_key: str,
    nucleus_id: int,
    payload: Mapping[str, Any],
) -> pd.DataFrame:
    draws = np.asarray(payload["draw_qki_raw"], dtype=np.float64)
    first = np.asarray(payload["first_pass_valid"], dtype=bool)
    redraw = np.asarray(payload["redraw_counts"], dtype=np.int32)
    angles = np.asarray(payload["initial_angles_deg"], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for draw_id in range(draws.shape[1]):
        finite = np.isfinite(draws[:, draw_id])
        first_n = int(first[:, draw_id].sum())
        rows.append(
            {
                "nucleus_uid": f"{image_key}:nucleus:{int(nucleus_id)}",
                "draw_id": draw_id,
                "initial_angle_deg": float(angles[draw_id]),
                "first_pass_valid_n": first_n,
                "first_pass_retention": float(first_n / draws.shape[0]),
                "redrawn_n": int((~first[:, draw_id] & finite).sum()),
                "unplaceable_n": int((~finite).sum()),
                "complete_keep_n": bool(finite.all()),
            }
        )
    return pd.DataFrame(rows)


def _write_selected_image_atomic(
    path: Path,
    *,
    image_key: str,
    selected_z_1based: int,
    planes: Mapping[str, np.ndarray],
    nucleus_labels: np.ndarray,
    nucleolus_labels: np.ndarray,
    channel_indices: Mapping[str, int],
) -> None:
    import h5py

    token = _h5_token(image_key, keep_dot=True)
    pending_key = f"_pending_{token}"
    with h5py.File(path, "a") as handle:
        final_path = f"images/{token}"
        pending_path = f"images/{pending_key}"
        if pending_path in handle:
            del handle[pending_path]
        if final_path in handle:
            del handle[final_path]
        created = write_selected_plane_h5_group(
            handle,
            image_key=pending_key,
            selected_z_1based=selected_z_1based,
            planes=planes,
            nucleus_labels=nucleus_labels,
            nucleolus_labels=nucleolus_labels,
            channel_indices=channel_indices,
        )
        handle[created].attrs["image_key"] = str(image_key).casefold()
        handle.move(created, final_path)
        handle.flush()


def _write_null_image_atomic(
    path: Path,
    *,
    image_key: str,
    selected_z_1based: int,
    null_groups: Mapping[int, Mapping[str, Any]],
    parameters: ExactFootprintParameters,
) -> None:
    import h5py

    token = _h5_token(image_key, keep_dot=True)
    pending = f"_pending_images/{token}"
    final = f"images/{token}"
    with h5py.File(path, "a") as handle:
        if pending in handle:
            del handle[pending]
        if final in handle:
            del handle[final]
        image_group = handle.create_group(pending)
        image_group.attrs.update(
            {
                "image_key": str(image_key).casefold(),
                "selected_z_1based": int(selected_z_1based),
                "placement_geometry": PLACEMENT_GEOMETRY,
                "primary_threshold_percentile": int(
                    parameters.primary_threshold_percentile
                ),
                "complete": False,
            }
        )
        # Preserve an explicit empty schema for zero-spot images. Consumers
        # can distinguish a completed image with zero candidate nuclei from a
        # truncated image that never reached null-group creation.
        image_group.require_group("nuclei")
        for nucleus_id, payload in null_groups.items():
            write_null_h5_group(
                handle,
                nucleus_uid=f"{image_key}:nucleus:{int(nucleus_id)}",
                image_key=image_key,
                nucleus_id=int(nucleus_id),
                selected_z_1based=selected_z_1based,
                seed=int(payload["seed"]),
                spot_uid=payload["spot_uid"],
                observed_qki_raw=payload["observed_qki_raw"],
                draw_qki_raw=payload["draw_qki_raw"],
                initial_angles_deg=payload["initial_angles_deg"],
                placement_angles_deg=payload["placement_angles_deg"],
                first_pass_valid=payload["first_pass_valid"],
                redraw_counts=payload["redraw_counts"],
                null_usable=bool(payload["null_usable"]),
                unusable_reason=str(payload["unusable_reason"]),
                root=f"{pending}/nuclei",
                max_redraw=parameters.max_redraw,
                min_valid_draw_fraction=parameters.min_valid_draw_fraction,
                threshold_percentile=parameters.primary_threshold_percentile,
                first_pass_retention_mean=float(
                    payload["first_pass_retention_mean"]
                ),
                first_pass_retention_median=float(
                    payload["first_pass_retention_median"]
                ),
            )
        image_group.attrs["complete"] = True
        handle.require_group("images")
        handle.move(pending, final)
        handle.flush()


def _read_cached_selected_image(
    path: Path, image_key: str
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    import h5py

    token = _h5_token(image_key, keep_dot=True)
    with h5py.File(path, "r") as handle:
        group = handle[f"images/{token}"]
        if not bool(group.attrs.get("complete", False)):
            raise ValueError(f"selected-plane cache is incomplete for {image_key}")
        planes = {role: group[role][:] for role in ("miat", "qki", "dapi")}
        return planes, group["nucleus_labels"][:], group["nucleolus_labels"][:]


def _parameter_fingerprint(parameters: ExactFootprintParameters) -> str:
    material = json.dumps(
        parameters.to_dict(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _execution_fingerprint(
    *,
    parameter_fingerprint: str,
    source_table_fingerprint: str,
    selected_image_keys: Sequence[str],
    phase1_only: bool,
    analysis_scope: str,
) -> str:
    material = json.dumps(
        {
            "parameter_fingerprint": parameter_fingerprint,
            "source_table_fingerprint": source_table_fingerprint,
            "selected_image_keys": list(selected_image_keys),
            "phase1_only": bool(phase1_only),
            "analysis_scope": str(analysis_scope),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _select_execution_keys(
    manifest: pd.DataFrame,
    image_keys: Sequence[str] | None,
) -> tuple[list[str], str]:
    """Normalize an optional smoke roster only after the full manifest exists."""

    available = [str(value).casefold() for value in manifest["image_key"]]
    if image_keys is None:
        return available, "full_manifest"
    requested = [(_source_image_key(value) if manifest.attrs.get("source_path_identity")
                  else _casefold_basename(value)) for value in image_keys]
    if not requested:
        raise ValueError("image_keys cannot be an empty sequence")
    if len(set(requested)) != len(requested):
        raise ValueError("image_keys contains duplicates after basename normalization")
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"requested image_keys are absent from the full manifest: {unknown}")
    requested_set = set(requested)
    selected = [key for key in available if key in requested_set]
    return selected, "smoke_subset"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_table_fingerprint(source_run: Path, hierarchy_file: Path) -> str:
    paths = [
        source_run / "per_image_summary.csv",
        source_run / "spot_metrics.csv",
        source_run / "nuclei_metrics.csv",
        source_run / "run_config.json",
        hierarchy_file,
    ]
    payload = [(str(path.resolve()), _sha256_file(path)) for path in paths]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _software_provenance() -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[3]

    def git_value(*arguments: str) -> str:
        try:
            completed = subprocess.run(
                ["git", "-C", str(repository), *arguments],
                check=True,
                capture_output=True,
                text=True,
                timeout=20,
            )
            return completed.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return "unavailable"

    packages: dict[str, str] = {}
    for distribution in (
        "numpy", "pandas", "scipy", "h5py", "bioio",
        "bioio-bioformats", "tifffile",
    ):
        try:
            packages[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            packages[distribution] = "not_installed"
    module_paths = [
        Path(__file__).resolve(),
        Path(__file__).with_name("footprint_null.py").resolve(),
    ]
    return {
        "generated_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "python_version": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "repository": str(repository),
        "worktree_commit": git_value("rev-parse", "HEAD"),
        "worktree_branch": git_value("branch", "--show-current"),
        "base_commit_main": git_value("merge-base", "HEAD", "main"),
        "java_home": os.environ.get("JAVA_HOME", "not_set"),
        "bff_java_version": os.environ.get("BFF_JAVA_VERSION", "not_set"),
        "bff_java_fetch": os.environ.get("BFF_JAVA_FETCH", "not_set"),
        "analysis_module_sha256": {
            path.name: _sha256_file(path) for path in module_paths if path.is_file()
        },
    }


def _input_checksum_table(
    manifest: pd.DataFrame,
    *,
    source_run: Path,
    hierarchy_file: Path,
) -> pd.DataFrame:
    sources: list[tuple[str, str, Path]] = [
        ("per_image_summary", "", source_run / "per_image_summary.csv"),
        ("spot_metrics", "", source_run / "spot_metrics.csv"),
        ("nuclei_metrics", "", source_run / "nuclei_metrics.csv"),
        ("run_config", "", source_run / "run_config.json"),
        ("authoritative_hierarchy", "", hierarchy_file),
    ]
    for row in manifest.to_dict("records"):
        image_key = str(row["image_key"]).casefold()
        for role, column in (
            ("analyzed_vsi", "analyzed_vsi_path"),
            ("raw_source_vsi", "raw_source_vsi"),
            ("nucleus_mask", "mask_path"),
            ("vsi_companion_ets", "companion_ets_path"),
        ):
            value = str(row.get(column, ""))
            if value:
                sources.append((role, image_key, Path(value)))
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for role, image_key, path in sources:
        resolved = path.resolve(strict=False)
        key = (role, image_key, str(resolved).casefold())
        if key in seen:
            continue
        seen.add(key)
        if not resolved.is_file():
            raise FileNotFoundError(f"checksum input is missing: {resolved}")
        stat = resolved.stat()
        rows.append(
            {
                "source_role": role,
                "image_key": image_key,
                "path": str(resolved),
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "sha256": _sha256_file(resolved),
                "unchanged_postrun": pd.NA,
            }
        )
    return pd.DataFrame(rows)


def _verify_input_checksums_unchanged(checksums: pd.DataFrame) -> pd.DataFrame:
    verified = checksums.copy()
    unchanged: list[bool] = []
    for row in verified.to_dict("records"):
        path = Path(str(row["path"]))
        stat = path.stat()
        same = bool(
            int(stat.st_size) == int(row["size_bytes"])
            and int(stat.st_mtime_ns) == int(row["mtime_ns"])
            and _sha256_file(path) == str(row["sha256"])
        )
        unchanged.append(same)
    verified["unchanged_postrun"] = unchanged
    if not all(unchanged):
        changed = verified.loc[
            ~verified["unchanged_postrun"].map(_explicit_bool), "path"
        ].tolist()
        raise RuntimeError(f"immutable source inputs changed during execution: {changed}")
    return verified


def _acquisition_metadata_table(manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for image in manifest.to_dict("records"):
        for role, column in (
            ("miat", "miat_channel_index"),
            ("qki", "qki_channel_index"),
            ("dapi", "dapi_channel_index"),
        ):
            rows.append(
                {
                    "image_key": image["image_key"],
                    "image": image["image"],
                    "channel_role": role,
                    "channel_index": int(image[column]),
                    "selected_z_1based": int(image["selected_z_1based"]),
                    "selected_z_0based": int(image["selected_z_0based"]),
                    "n_z": int(image["n_z"]),
                    "voxel_xy_nm": float(image["voxel_xy_nm"]),
                    "voxel_z_nm": float(image["voxel_z_nm"]),
                    "height_px": image.get("height_px", np.nan),
                    "width_px": image.get("width_px", np.nan),
                    "exposure": np.nan,
                    "exposure_status": "not_available_in_completed_run_tables",
                    "gain": np.nan,
                    "gain_status": "not_available_in_completed_run_tables",
                    "detector": "",
                    "detector_status": "not_available_in_completed_run_tables",
                    "laser_power": np.nan,
                    "laser_power_status":
                        "absent_from_vsi_ome_metadata_unverified",
                    "quantitative_normalization_priority":
                        "within_nucleus_and_within_nucleoplasm",
                }
            )
    return pd.DataFrame(rows)


def _spot_population_summary(frame: pd.DataFrame) -> dict[str, Any]:
    floor = frame["passes_miat_floor"].map(_explicit_bool)
    result: dict[str, Any] = {
        "n_spots_all": int(len(frame)),
        "n_spots_floor": int(floor.sum()),
        "n_below_floor": int((~floor).sum()),
    }
    for percentile in THRESHOLD_PERCENTILES:
        labels = frame[f"population_label_q{percentile}"].astype(str)
        positive = int(labels.eq("threshold_positive").sum())
        negative = int(labels.eq("threshold_negative").sum())
        unusable = int(labels.eq("unusable").sum())
        result[f"n_threshold_positive_q{percentile}"] = positive
        result[f"n_threshold_negative_q{percentile}"] = negative
        result[f"n_unusable_q{percentile}"] = unusable
        result[f"reconciliation_pass_q{percentile}"] = bool(
            int(floor.sum()) == positive + negative + unusable
            and len(frame) == int(floor.sum()) + int((~floor).sum())
        )
    return result


def _population_reconciliation_table(
    spots: pd.DataFrame,
    nuclei: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {"level": "project", "key": "project", **_spot_population_summary(spots)}
    ]
    for image_key in manifest["image_key"].astype(str):
        group = spots.loc[spots["image_key"].astype(str).eq(image_key)]
        rows.append(
            {"level": "image", "key": image_key, **_spot_population_summary(group)}
        )
    for source in nuclei.to_dict("records"):
        row = {
            "level": "nucleus",
            "key": str(source["nucleus_uid"]),
            "n_spots_all": int(source["n_spots_all"]),
            "n_spots_floor": int(source["n_spots_floor"]),
            "n_below_floor": int(source["n_below_floor"]),
        }
        for percentile in THRESHOLD_PERCENTILES:
            for metric in (
                "n_threshold_positive", "n_threshold_negative", "n_unusable"
            ):
                row[f"{metric}_q{percentile}"] = int(
                    source[f"{metric}_q{percentile}"]
                )
            row[f"reconciliation_pass_q{percentile}"] = bool(
                source[f"population_reconciliation_pass_q{percentile}"]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _data_dictionary_table(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for table_name, frame in tables.items():
        for column in frame.columns:
            if str(column).endswith("_raw"):
                unit = "raw detector intensity"
            elif str(column).endswith("_px"):
                unit = "pixel"
            elif str(column).endswith("_nm"):
                unit = "nanometer"
            else:
                unit = ""
            rows.append(
                {
                    "table": table_name,
                    "column": str(column),
                    "dtype": str(frame[column].dtype),
                    "unit": unit,
                    "description": f"Canonical {table_name} field: {column}",
                }
            )
    return pd.DataFrame(rows)


def _methods_dictionary_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "method_id": "exact_recorded_single_z",
                "role": "primary_quantitation_plane",
                "definition": "All channels read as YX at per_image_summary.z_plane minus one.",
                "primary": True,
            },
            {
                "method_id": "half_max_exact_footprint",
                "role": "miat_detection_footprint",
                "definition": "Persisted MIAT centre with original half-maximum connected footprint.",
                "primary": True,
            },
            {
                "method_id": PLACEMENT_GEOMETRY,
                "role": "keep_n_null",
                "definition": "Rotate centres and translate unchanged exact footprint pixels.",
                "primary": True,
            },
            {
                "method_id": "own_linear_q90_q95_q99",
                "role": "threshold_sensitivity",
                "definition": "Strict observed greater than each spot's own linear null quantile; q95 primary.",
                "primary": True,
            },
            {
                "method_id": "r1_r3_r6_disks_and_annuli",
                "role": "radius_sensitivity",
                "definition": "Non-primary local QKI radius sensitivity metrics.",
                "primary": False,
            },
            {
                "method_id": "display_ranges",
                "role": "rendering_only",
                "definition": "DAPI/MIAT/QKI display windows have no quantitative effect.",
                "primary": False,
            },
        ]
    )


def _finalize_provenance_outputs(
    *,
    output: Path,
    manifest: pd.DataFrame,
    input_checksums: pd.DataFrame,
    spot_path: Path,
    pixel_path: Path,
    nucleus_path: Path | None,
    null_index_path: Path | None,
) -> dict[str, Any]:
    verified_checksums = _verify_input_checksums_unchanged(input_checksums)
    _atomic_write_dataframe(verified_checksums, output / "input_checksums.csv")
    acquisition = _acquisition_metadata_table(manifest)
    _atomic_write_dataframe(acquisition, output / "acquisition_metadata.csv")
    spots = pd.read_csv(spot_path)
    nuclei = (
        pd.read_csv(nucleus_path)
        if nucleus_path is not None and nucleus_path.is_file()
        else pd.DataFrame()
    )
    reconciliation = _population_reconciliation_table(spots, nuclei, manifest)
    if not all(
        reconciliation[
            [f"reconciliation_pass_q{p}" for p in THRESHOLD_PERCENTILES]
        ]
        .fillna(False)
        .to_numpy(bool)
        .ravel()
    ):
        raise RuntimeError("population reconciliation failed")
    _atomic_write_dataframe(
        reconciliation, output / "population_reconciliation.csv"
    )
    table_headers: dict[str, pd.DataFrame] = {
        "image_manifest": manifest,
        "spot_exact_footprint_metrics": spots.iloc[0:0],
        "footprint_pixels": pd.read_csv(pixel_path, nrows=0),
        "acquisition_metadata": acquisition.iloc[0:0],
        "population_reconciliation": reconciliation.iloc[0:0],
    }
    if not nuclei.empty:
        table_headers["nucleus_exact_footprint_metrics"] = nuclei.iloc[0:0]
    if null_index_path is not None and null_index_path.is_file():
        table_headers["null_draw_index"] = pd.read_csv(null_index_path, nrows=0)
    _atomic_write_dataframe(
        _data_dictionary_table(table_headers), output / "data_dictionary.csv"
    )
    _atomic_write_dataframe(
        _methods_dictionary_table(), output / "methods_dictionary.csv"
    )
    _atomic_write_json(_software_provenance(), output / "software_provenance.json")
    return {
        "input_checksum_rows": int(len(verified_checksums)),
        "input_immutability_pass": True,
        "population_reconciliation_rows": int(len(reconciliation)),
        "population_reconciliation_pass": True,
        "laser_power_status": "absent_from_vsi_ome_metadata_unverified",
    }


def _h5_complete(path: Path, group_path: str) -> bool:
    if not path.is_file():
        return False
    import h5py

    try:
        with h5py.File(path, "r") as handle:
            return bool(
                group_path in handle
                and handle[group_path].attrs.get("complete", False)
            )
    except (OSError, KeyError, ValueError):
        return False


def _h5_checkpoint_complete(
    path: Path,
    *,
    group_path: str,
    pending_path: str,
    image_key: str,
) -> bool:
    if not path.is_file():
        return False
    import h5py

    try:
        with h5py.File(path, "r") as handle:
            if pending_path in handle or group_path not in handle:
                return False
            group = handle[group_path]
            return bool(
                group.attrs.get("complete", False)
                and str(group.attrs.get("image_key", "")).casefold()
                == str(image_key).casefold()
            )
    except (OSError, KeyError, ValueError):
        return False


def _assert_no_pending_h5_groups(path: Path) -> None:
    if not path.is_file():
        return
    import h5py

    with h5py.File(path, "r") as handle:
        image_pending = (
            [
                name
                for name in handle["images"].keys()
                if str(name).startswith("_pending_")
            ]
            if "images" in handle
            else []
        )
        null_pending = (
            list(handle["_pending_images"].keys())
            if "_pending_images" in handle
            else []
        )
        if image_pending or null_pending:
            raise RuntimeError(
                f"HDF5 contains pending image groups: "
                f"images={image_pending}, nulls={null_pending}"
            )


def _csv_fragment_has_header(path: Path, *, compressed: bool) -> bool:
    if not path.is_file():
        return False
    opener = gzip.open if compressed else open
    try:
        with opener(path, "rt", encoding="utf-8", newline="") as handle:
            return bool(handle.readline().strip())
    except (OSError, EOFError, UnicodeError):
        return False


def _verify_phase1_checkpoint(
    *,
    marker: Path,
    base_spots: Path,
    pixels: Path,
    selected_h5: Path,
    image_key: str,
) -> None:
    token = _h5_token(image_key, keep_dot=True)
    good = (
        marker.is_file()
        and _csv_fragment_has_header(base_spots, compressed=True)
        and _csv_fragment_has_header(pixels, compressed=True)
        and _h5_checkpoint_complete(
            selected_h5,
            group_path=f"images/{token}",
            pending_path=f"images/_pending_{token}",
            image_key=image_key,
        )
    )
    if not good:
        raise RuntimeError(
            f"phase-1 checkpoint marker/artifact mismatch for {image_key}"
        )


def _verify_phase2_checkpoint(
    *,
    marker: Path,
    spots: Path,
    nuclei: Path,
    null_index: Path,
    null_h5: Path,
    image_key: str,
) -> None:
    token = _h5_token(image_key, keep_dot=True)
    good = (
        marker.is_file()
        and _csv_fragment_has_header(spots, compressed=True)
        and _csv_fragment_has_header(nuclei, compressed=False)
        and _csv_fragment_has_header(null_index, compressed=True)
        and _h5_checkpoint_complete(
            null_h5,
            group_path=f"images/{token}",
            pending_path=f"_pending_images/{token}",
            image_key=image_key,
        )
    )
    if not good:
        raise RuntimeError(
            f"phase-2 checkpoint marker/artifact mismatch for {image_key}"
        )


def _resource_snapshot(path: Path) -> dict[str, int]:
    free_disk = int(shutil.disk_usage(path).free)
    rss = -1
    available_ram = -1
    try:
        import psutil

        rss = int(psutil.Process(os.getpid()).memory_info().rss)
        available_ram = int(psutil.virtual_memory().available)
    except (ImportError, OSError):
        pass
    return {
        "rss_bytes": rss,
        "available_ram_bytes": available_ram,
        "output_free_bytes": free_disk,
    }


def _enforce_resource_gates(
    snapshot: Mapping[str, int],
    *,
    preflight: bool,
) -> None:
    gib = 1024**3
    minimum_disk = 20 * gib if preflight else 10 * gib
    if int(snapshot["output_free_bytes"]) < minimum_disk:
        raise RuntimeError(
            f"insufficient output disk: {snapshot['output_free_bytes']} bytes free"
        )
    available = int(snapshot["available_ram_bytes"])
    if preflight and available >= 0 and available < 16 * gib:
        raise RuntimeError(f"insufficient available RAM: {available} bytes")
    rss = int(snapshot["rss_bytes"])
    if not preflight and rss >= 0 and rss > 16 * gib:
        raise RuntimeError(f"process working set exceeded 16 GiB: {rss} bytes")


def run_exact_footprint_backfill(
    run_dir: str | Path,
    hierarchy_path: str | Path | None,
    output_dir: str | Path,
    *,
    parameters: ExactFootprintParameters | None = None,
    resume: bool = False,
    plane_reader: Callable[..., tuple[dict[str, np.ndarray], dict[str, Any]]] = read_exact_selected_planes,
    nucleolus_builder: Callable[[Mapping[str, Any], np.ndarray, np.ndarray, float], np.ndarray] = _default_nucleolus_builder,
    null_calibrator: Callable[..., ReconstructionResult] = calibrate_reconstruction_nulls,
    validate_expected_design: bool = True,
    image_keys: Sequence[str] | None = None,
    phase1_only: bool = False,
    allow_output_inside_run: bool = False,
) -> dict[str, Any]:
    """Run the two-phase, serial, resumable exact-footprint reconstruction.

    Phase 1 reconstructs and parity-checks every image and caches its selected
    planes.  No null calculation can start until all phase-1 markers and the
    project-wide parity gate exist.  Phase 2 reads only that cache, performs the
    KEEP-N null image by image, and writes final fragments plus atomic HDF5 image
    groups.  Canonical CSVs are assembled by streaming the fragments.
    """

    params = parameters or ExactFootprintParameters()
    source_run = Path(run_dir)
    native = hierarchy_path is None
    hierarchy_file = (source_run / "resolved_experiment_hierarchy.csv"
                      if native else Path(hierarchy_path))
    if native:
        validate_expected_design = False
    native_provenance = ({"design_check": "not_applicable_native_hierarchy",
                          "hierarchy_source": "native_run_outputs"} if native else {})
    output = Path(output_dir)
    resolved_output = output.resolve(strict=False)
    resolved_source_run = source_run.resolve(strict=False)
    if (
        resolved_output == resolved_source_run
        or (resolved_source_run in resolved_output.parents
            and not (native and allow_output_inside_run))
    ):
        raise ValueError(
            "output directory cannot equal or be inside the historical run"
        )
    if output.exists() and not resume:
        raise FileExistsError(f"output directory already exists: {output}")
    if resume and not output.is_dir():
        raise FileNotFoundError(f"resume output directory does not exist: {output}")

    # Construct and validate the complete authoritative roster before applying
    # any optional smoke subset.  No output is created until that full-roster
    # validation succeeds.
    per_image = pd.read_csv(source_run / "per_image_summary.csv")
    historical_spots = pd.read_csv(source_run / "spot_metrics.csv")
    hierarchy = native_hierarchy_from_run(source_run) if native else pd.read_csv(hierarchy_file)
    if native:
        native_provenance.update(hierarchy.attrs["native_provenance"])
    run_config = json.loads(
        (source_run / "run_config.json").read_text(encoding="utf-8")
    )
    source_identity = _uses_source_identity(hierarchy)
    if source_identity:
        raw_input_dir = Path(run_config["input_dir"]) if run_config.get("input_dir") else None
    else:
        _subset, raw_input_dir = _subset_lookup(run_config)
    if raw_input_dir is not None:
        resolved_raw_input = raw_input_dir.resolve(strict=False)
        if (
            resolved_output == resolved_raw_input
            or resolved_raw_input in resolved_output.parents
        ):
            raise ValueError(
                "output directory cannot equal or be inside the raw input tree"
            )
    manifest = build_image_manifest(
        per_image,
        hierarchy,
        historical_spots,
        run_config=run_config,
        run_dir=source_run,
        validate_expected_design=validate_expected_design,
        resolve_paths=True,
    )
    if native:
        print("DESIGN CHECK not_applicable_native_hierarchy: observed design "
              f"(images, biological, controls, sets) = {manifest.attrs.get('observed_design')}",
              flush=True)
    selected_image_keys, analysis_scope = _select_execution_keys(manifest, image_keys)
    selected_key_set = set(selected_image_keys)
    biological_inference_output = bool(
        analysis_scope == "full_manifest" and not phase1_only
    )
    manifest["selected_for_execution"] = manifest["image_key"].isin(selected_key_set)
    manifest["analysis_scope"] = analysis_scope
    manifest["execution_phase"] = (
        "phase1_only" if phase1_only else "full_two_phase"
    )
    manifest["eligible_for_biological_inference_output"] = (
        manifest["selected_for_execution"]
        & ~manifest["is_control"].map(_explicit_bool)
        & biological_inference_output
    )
    manifest.loc[
        ~manifest["selected_for_execution"], "load_status"
    ] = "not_selected_smoke"

    fingerprint = _parameter_fingerprint(params)
    if native:
        # Reject pre-eligibility checkpoints, which reconstructed a different
        # population. The explicit-hierarchy fingerprint remains unchanged.
        fingerprint = hashlib.sha256(
            (fingerprint + ":native_spot_eligibility_v1").encode("utf-8")
        ).hexdigest()
    source_fingerprint = _source_table_fingerprint(source_run, hierarchy_file)
    execution_fingerprint = _execution_fingerprint(
        parameter_fingerprint=fingerprint,
        source_table_fingerprint=source_fingerprint,
        selected_image_keys=selected_image_keys,
        phase1_only=phase1_only,
        analysis_scope=analysis_scope,
    )
    if validate_expected_design:
        _enforce_resource_gates(
            _resource_snapshot(output.parent), preflight=True
        )
    run_started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = output / "_checkpoints"
    checkpoints.mkdir(exist_ok=True)
    parameter_path = output / "analysis_parameters.json"
    if resume:
        if not parameter_path.is_file():
            raise ValueError("resume output lacks analysis_parameters.json")
        recorded = json.loads(parameter_path.read_text(encoding="utf-8"))
        if recorded.get("parameter_fingerprint") != fingerprint:
            raise ValueError("resume parameters do not match the existing output")
        if recorded.get("source_table_fingerprint") != source_fingerprint:
            raise ValueError("resume source tables do not match the existing output")
        if recorded.get("execution_fingerprint") != execution_fingerprint:
            raise ValueError(
                "resume execution scope/selected images/phase do not match "
                "the existing output"
            )
    else:
        _atomic_write_json(
            {
                **params.to_dict(),
                **native_provenance,
                "parameter_fingerprint": fingerprint,
                "source_run_dir": str(source_run.resolve()),
                "hierarchy_path": str(hierarchy_file.resolve()),
                "two_phase_full_parity_before_null": True,
                "design_check_skipped": not bool(validate_expected_design),
                "observed_design": list(manifest.attrs.get("observed_design", ())),
                "audited_design": [44, 37, 7, 12],
                "source_table_fingerprint": source_fingerprint,
                "execution_fingerprint": execution_fingerprint,
                "selected_image_keys": selected_image_keys,
                "phase1_only": bool(phase1_only),
                "analysis_scope": analysis_scope,
                "eligible_for_biological_inference": biological_inference_output,
                "display_only_ranges": {
                    "dapi": [334, 5500],
                    "miat_primary": [400, 4000],
                    "miat_optional": [700, 5000],
                    "qki": [555, 3000],
                    "quantitative_effect": "none",
                },
                "laser_power_status":
                    "absent_from_vsi_ome_metadata_unverified",
            },
            parameter_path,
        )

    manifest_path = output / "image_manifest.csv"
    selected_h5 = output / "selected_planes_and_masks.h5"
    null_h5 = output / "exact_footprint_nulls.h5"
    hierarchy_keys = hierarchy.copy()
    hierarchy_keys["image_key"] = hierarchy_keys["image_key"].astype(str).str.casefold()
    historical = historical_spots.copy()
    historical["image_key"] = (
        _source_keys_for_table(historical, hierarchy, table="spot_metrics.csv")
        if source_identity else historical["image"].map(_casefold_basename))
    if native:
        historical, eligibility = _filter_native_spots(historical, manifest, hierarchy_keys)
        count_columns = ["n_input_spots", "n_eligible_spots", "not_in_nucleus",
                         "centre_off_parent_label"]
        native_provenance["native_spot_eligibility"] = {
            "rule": "recorded in_nucleus=True AND rounded centre on saved parent label",
            "center_rounding": "numpy_rint_ties_to_even; native persisted centres are integers",
            "reason_precedence": ["not_in_nucleus", "centre_off_parent_label"],
            "hard_error": "in_nucleus=True and distance to parent label > 1 Euclidean px",
            "scope": "selected execution images; RNA1 spots before floor and nucleolus gates",
            "summary_file": "spot_eligibility_summary.csv",
            "totals": {column: int(eligibility[column].sum()) for column in count_columns},
        }
    checksum_path = output / "input_checksums.csv"
    if resume:
        if not checksum_path.is_file():
            raise ValueError("resume output lacks input_checksums.csv")
        input_checksums = _verify_input_checksums_unchanged(
            pd.read_csv(checksum_path)
        )
    else:
        input_checksums = _input_checksum_table(
            manifest, source_run=source_run, hierarchy_file=hierarchy_file
        )
        _atomic_write_dataframe(input_checksums, checksum_path)
    if native:
        # A rejected resume must not rewrite previously published provenance.
        _atomic_write_dataframe(eligibility, output / "spot_eligibility_summary.csv")
        recorded_parameters = json.loads(parameter_path.read_text(encoding="utf-8"))
        recorded_parameters.update(native_provenance)
        _atomic_write_json(recorded_parameters, parameter_path)
    source_hash = (
        input_checksums.loc[
            input_checksums["source_role"].eq("analyzed_vsi"),
            ["image_key", "sha256"],
        ]
        .drop_duplicates("image_key")
        .set_index("image_key")["sha256"]
    )
    mask_hash = (
        input_checksums.loc[
            input_checksums["source_role"].eq("nucleus_mask"),
            ["image_key", "sha256"],
        ]
        .drop_duplicates("image_key")
        .set_index("image_key")["sha256"]
    )
    manifest["source_checksum"] = manifest["image_key"].map(source_hash).fillna("")
    manifest["mask_checksum"] = manifest["image_key"].map(mask_hash).fillna("")
    if resume and manifest_path.is_file():
        prior_manifest = pd.read_csv(manifest_path)
        prior_manifest["image_key"] = prior_manifest["image_key"].astype(str).str.casefold()
        if set(prior_manifest["image_key"]) != set(manifest["image_key"]):
            raise ValueError("resume image manifest keys do not match the existing output")
        status_columns = [
            column
            for column in (
                "load_status", "mask_status", "height_px", "width_px", "notes"
            )
            if column in prior_manifest.columns
        ]
        prior_status = prior_manifest.set_index("image_key")[status_columns]
        for column in status_columns:
            manifest[column] = manifest["image_key"].map(prior_status[column])

    # Phase 1: every exact footprint must pass historical parity before any null.
    phase1_fragments: list[Path] = []
    phase1_spot_fragments: list[Path] = []
    execution_indices = list(
        manifest.index[manifest["selected_for_execution"].map(_explicit_bool)]
    )
    for manifest_index in execution_indices:
        image_row = manifest.loc[manifest_index]
        image_key = str(image_row["image_key"]).casefold()
        image_started = time.perf_counter()
        token = hashlib.sha256(image_key.encode("utf-8")).hexdigest()[:16]
        checkpoint = checkpoints / token
        checkpoint.mkdir(exist_ok=True)
        phase1_done = checkpoint / "phase1.done.json"
        base_spot_path = checkpoint / "spot_base.csv.gz"
        pixel_path = checkpoint / "footprint_pixels.csv.gz"
        phase1_spot_fragments.append(base_spot_path)
        phase1_fragments.append(pixel_path)
        if phase1_done.is_file():
            _verify_phase1_checkpoint(
                marker=phase1_done,
                base_spots=base_spot_path,
                pixels=pixel_path,
                selected_h5=selected_h5,
                image_key=image_key,
            )
            manifest.at[manifest_index, "load_status"] = "phase1_complete"
            continue
        planes, loaded = plane_reader(
            Path(str(image_row["analyzed_vsi_path"])),
            selected_z_1based=int(image_row["selected_z_1based"]),
            channel_indices={
                "miat": int(image_row["miat_channel_index"]),
                "qki": int(image_row["qki_channel_index"]),
                "dapi": int(image_row["dapi_channel_index"]),
            },
        )
        _validate_loaded_plane_bundle(
            planes,
            loaded,
            image_key=image_key,
            selected_z_1based=int(image_row["selected_z_1based"]),
            channel_indices={
                "miat": int(image_row["miat_channel_index"]),
                "qki": int(image_row["qki_channel_index"]),
                "dapi": int(image_row["dapi_channel_index"]),
            },
            validate_expected_design=validate_expected_design,
        )
        if int(loaded["n_z"]) != int(image_row["n_z"]):
            raise ValueError(f"loaded n_z differs from recorded manifest for {image_key}")
        for field_name in ("voxel_xy_nm", "voxel_z_nm"):
            recorded_value = float(image_row[field_name])
            loaded_value = float(loaded[field_name])
            if not np.isfinite(recorded_value):
                raise ValueError(
                    f"recorded {field_name} must be finite for {image_key}"
                )
            if not np.isfinite(loaded_value):
                raise ValueError(
                    f"loaded {field_name} must be finite for {image_key}"
                )
            if not np.isclose(
                recorded_value, loaded_value, rtol=1e-9, atol=1e-9
            ):
                raise ValueError(
                    f"loaded {field_name} differs from recorded manifest for {image_key}"
                )
        image_spots = historical.loc[
            historical["image_key"].eq(image_key)
            & historical["channel"].astype(str).eq("rna1")
        ].copy()
        expected_ids = set(
            pd.to_numeric(
                hierarchy_keys.loc[
                    hierarchy_keys["image_key"].eq(image_key), "nucleus_id"
                ],
                errors="raise",
            ).astype(int)
        )
        labels = read_and_validate_label_mask(
            Path(str(image_row["mask_path"])),
            expected_shape=planes["miat"].shape,
            expected_nucleus_ids=expected_ids,
            spots=image_spots,
        )
        nucleolus_result = nucleolus_builder(
            run_config,
            labels,
            planes["dapi"],
            float(image_row["voxel_xy_nm"]),
        )
        if nucleolus_result is None:
            raise RuntimeError(
                f"nucleolus reconstruction returned None for {image_key}; "
                "refusing to disable exclusion silently"
            )
        nucleoli = np.asarray(nucleolus_result, dtype=np.int32)
        if nucleoli.shape != labels.shape:
            raise RuntimeError(
                f"nucleolus reconstruction shape mismatch for {image_key}: "
                f"{nucleoli.shape} versus {labels.shape}"
            )
        invalid_nucleolus = (nucleoli > 0) & (nucleoli != labels)
        if bool(invalid_nucleolus.any()):
            raise RuntimeError(
                f"nucleolus reconstruction contains labels outside their parent "
                f"nucleus for {image_key}"
            )
        image_nuclei = hierarchy_keys.loc[
            hierarchy_keys["image_key"].eq(image_key)
        ].drop_duplicates(["image_key", "nucleus_id"])
        reconstruction = reconstruct_image_footprints(
            planes["miat"],
            planes["qki"],
            labels,
            nucleoli,
            image_spots,
            image_row.to_dict(),
            parameters=params,
            run_id=source_run.name,
            compute_null=False,
            nucleus_hierarchy=image_nuclei,
        )
        parity = validate_historical_parity(
            reconstruction.spot_metrics,
            image_spots.loc[image_spots["channel"].astype(str).eq("rna1")],
        )
        _atomic_write_dataframe(
            reconstruction.spot_metrics, base_spot_path, compression="gzip"
        )
        _atomic_write_dataframe(
            reconstruction.pixel_metrics, pixel_path, compression="gzip"
        )
        _write_selected_image_atomic(
            selected_h5,
            image_key=image_key,
            selected_z_1based=int(image_row["selected_z_1based"]),
            planes=planes,
            nucleus_labels=labels,
            nucleolus_labels=nucleoli,
            channel_indices={
                "miat": int(image_row["miat_channel_index"]),
                "qki": int(image_row["qki_channel_index"]),
                "dapi": int(image_row["dapi_channel_index"]),
            },
        )
        selected_token = _h5_token(image_key, keep_dot=True)
        if not _h5_checkpoint_complete(
            selected_h5,
            group_path=f"images/{selected_token}",
            pending_path=f"images/_pending_{selected_token}",
            image_key=image_key,
        ):
            raise RuntimeError(
                f"selected-plane HDF5 readback failed for {image_key}"
            )
        audit_payload = dict(reconstruction.audit)
        plane_shape = tuple(planes["miat"].shape)
        del planes, loaded, labels, nucleoli, reconstruction
        gc.collect()
        resources = _resource_snapshot(output)
        if validate_expected_design:
            _enforce_resource_gates(resources, preflight=False)
        phase1_elapsed = float(time.perf_counter() - image_started)
        _atomic_write_json(
            {
                **parity,
                **audit_payload,
                "image_key": image_key,
                "phase": "phase1",
                "image_elapsed_s": phase1_elapsed,
                "total_elapsed_s": float(time.perf_counter() - run_started),
                **resources,
            },
            phase1_done,
        )
        manifest.at[manifest_index, "load_status"] = "phase1_complete"
        manifest.at[manifest_index, "mask_status"] = "loaded_reused"
        manifest.at[manifest_index, "height_px"] = int(plane_shape[0])
        manifest.at[manifest_index, "width_px"] = int(plane_shape[1])
        manifest.at[manifest_index, "phase1_elapsed_s"] = phase1_elapsed
        manifest.at[manifest_index, "post_phase1_rss_bytes"] = resources["rss_bytes"]
        print(
            f"PHASE1 {len(phase1_fragments)}/{len(selected_image_keys)} "
            f"{image_key} z={int(image_row['selected_z_1based'])}/"
            f"{int(image_row['n_z'])} spots={parity['n_spots']} "
            f"elapsed={phase1_elapsed:.2f}s rss={resources['rss_bytes']} "
            f"free={resources['output_free_bytes']}",
            flush=True,
        )
        _atomic_write_dataframe(manifest, manifest_path)

    phase1_markers = [
        checkpoints / hashlib.sha256(str(key).casefold().encode("utf-8")).hexdigest()[:16] / "phase1.done.json"
        for key in selected_image_keys
    ]
    if not all(path.is_file() for path in phase1_markers):
        raise RuntimeError("project-wide parity gate cannot pass: phase-1 images are incomplete")
    parity_records = [json.loads(path.read_text(encoding="utf-8")) for path in phase1_markers]
    parity_total = int(sum(int(record["n_spots"]) for record in parity_records))
    historical_total = int(
        (
            historical["channel"].astype(str).eq("rna1")
            & historical["image_key"].isin(selected_key_set)
        ).sum()
    )
    if parity_total != historical_total:
        raise ValueError(
            f"project parity count mismatch: reconstructed {parity_total}, historical {historical_total}"
        )
    if (
        validate_expected_design
        and analysis_scope == "full_manifest"
        and historical_total != 23_829
    ):
        raise ValueError(
            f"audited historical run must contain 23829 RNA1 spots, found {historical_total}"
        )
    gate_path = output / (
        "full_historical_parity_gate.json"
        if analysis_scope == "full_manifest"
        else "smoke_subset_historical_parity_gate.json"
    )
    _atomic_write_json(
        {
            "pass": True,
            "n_images": int(len(selected_image_keys)),
            "n_spots": parity_total,
            "area_exact_all_images": True,
            "qki_allclose_all_images": True,
            "null_started_only_after_this_gate": True,
            "analysis_scope": analysis_scope,
            "selected_image_keys": selected_image_keys,
            "phase1_only": bool(phase1_only),
            "eligible_for_biological_inference": biological_inference_output,
            "execution_fingerprint": execution_fingerprint,
        },
        gate_path,
    )

    if analysis_scope == "smoke_subset":
        _atomic_write_text(
            "SMOKE SUBSET ONLY. This output contains only the explicitly selected "
            "images and must not be used for biological inference.\n",
            output / "SMOKE_SUBSET_DO_NOT_USE_FOR_INFERENCE.txt",
        )

    if phase1_only:
        _combine_csv_fragments(
            phase1_spot_fragments,
            output / "spot_exact_footprint_metrics.csv.gz",
            compressed=True,
        )
        _combine_csv_fragments(
            phase1_fragments,
            output / "footprint_pixels.csv.gz",
            compressed=True,
        )
        _atomic_write_text(
            "PHASE 1 ONLY. Historical exact-footprint parity passed for the "
            "selected execution roster, but no KEEP-N nulls were calculated. "
            "Do not use this output for biological inference.\n",
            output / "PHASE1_ONLY_DO_NOT_USE_FOR_INFERENCE.txt",
        )
        _assert_no_pending_h5_groups(selected_h5)
        _atomic_write_dataframe(manifest, manifest_path)
        provenance = _finalize_provenance_outputs(
            output=output,
            manifest=manifest,
            input_checksums=input_checksums,
            spot_path=output / "spot_exact_footprint_metrics.csv.gz",
            pixel_path=output / "footprint_pixels.csv.gz",
            nucleus_path=None,
            null_index_path=None,
        )
        summary = {
            "run_status": "phase1_only_complete",
            **native_provenance,
            "analysis_scope": analysis_scope,
            "phase1_only": True,
            "selected_image_keys": selected_image_keys,
            "n_images_phase1_complete": int(
                manifest["load_status"].eq("phase1_complete").sum()
            ),
            "n_images_full_roster": int(len(manifest)),
            "n_spots": parity_total,
            "full_parity_gate_pass": analysis_scope == "full_manifest",
            "eligible_for_biological_inference": False,
            "execution_fingerprint": execution_fingerprint,
            "output_dir": str(output.resolve()),
            **provenance,
        }
        _atomic_write_json(summary, output / "validation_report.json")
        return summary

    # Phase 2: null calibration and nucleus rollup from the exact-plane cache.
    final_spot_fragments: list[Path] = []
    nucleus_fragments: list[Path] = []
    null_index_fragments: list[Path] = []
    for manifest_index in execution_indices:
        image_row = manifest.loc[manifest_index]
        image_key = str(image_row["image_key"]).casefold()
        image_started = time.perf_counter()
        token = hashlib.sha256(image_key.encode("utf-8")).hexdigest()[:16]
        checkpoint = checkpoints / token
        phase2_done = checkpoint / "phase2.done.json"
        final_spot_path = checkpoint / "spot_final.csv.gz"
        nucleus_path = checkpoint / "nucleus_final.csv"
        null_index_path = checkpoint / "null_draw_index.csv.gz"
        final_spot_fragments.append(final_spot_path)
        nucleus_fragments.append(nucleus_path)
        null_index_fragments.append(null_index_path)
        if phase2_done.is_file():
            _verify_phase2_checkpoint(
                marker=phase2_done,
                spots=final_spot_path,
                nuclei=nucleus_path,
                null_index=null_index_path,
                null_h5=null_h5,
                image_key=image_key,
            )
            manifest.at[manifest_index, "load_status"] = "complete"
            continue
        planes, labels, nucleoli = _read_cached_selected_image(selected_h5, image_key)
        image_spots = historical.loc[historical["image_key"].eq(image_key)].copy()
        image_nuclei = hierarchy_keys.loc[
            hierarchy_keys["image_key"].eq(image_key)
        ].drop_duplicates(["image_key", "nucleus_id"])
        reconstruction = reconstruct_image_footprints(
            planes["miat"],
            planes["qki"],
            labels,
            nucleoli,
            image_spots,
            image_row.to_dict(),
            parameters=params,
            run_id=source_run.name,
            compute_null=False,
            nucleus_hierarchy=image_nuclei,
        )
        reconstruction = null_calibrator(
            reconstruction,
            parameters=params,
            run_id=source_run.name,
        )
        nucleus_table = aggregate_nucleus_metrics(
            reconstruction.spot_metrics,
            reconstruction.pixel_metrics,
            image_nuclei,
            miat_2d=planes["miat"],
            qki_2d=planes["qki"],
            nucleus_labels=labels,
            nucleolus_labels=nucleoli,
        )
        if native:
            # Eligibility can empty an image. Keep its zero-spot nuclei and
            # correlation columns so CSV fragments have the populated schema.
            for index in nucleus_table.index[nucleus_table["n_spots_all"].eq(0)]:
                nid = int(nucleus_table.at[index, "nucleus_id"])
                for corr in correlation_records(
                    reconstruction.spot_metrics, nucleus_id=nid,
                ).to_dict("records"):
                    stem = f"corr_{corr['population']}_{corr['measurement_pair']}"
                    for key in (
                        "n_spots", "pearson_r", "pearson_p", "spearman_rho", "spearman_p",
                        "estimable", "nonestimable_reason", "conditional_descriptive",
                    ):
                        nucleus_table.at[index, f"{stem}_{key}"] = corr[key]
        reconciliation_columns = [
            f"population_reconciliation_pass_q{p}"
            for p in THRESHOLD_PERCENTILES
        ]
        if not nucleus_table[reconciliation_columns].fillna(False).to_numpy(bool).all():
            raise RuntimeError(
                f"nucleus population reconciliation failed for {image_key}"
            )
        null_indices = [
            _null_draw_index(image_key, nucleus_id, payload)
            for nucleus_id, payload in reconstruction.null_groups.items()
        ]
        null_index = (
            pd.concat(null_indices, ignore_index=True)
            if null_indices
            else pd.DataFrame(
                columns=[
                    "nucleus_uid", "draw_id", "initial_angle_deg",
                    "first_pass_valid_n", "first_pass_retention", "redrawn_n",
                    "unplaceable_n", "complete_keep_n",
                ]
            )
        )
        _atomic_write_dataframe(
            reconstruction.spot_metrics, final_spot_path, compression="gzip"
        )
        _atomic_write_dataframe(nucleus_table, nucleus_path)
        _atomic_write_dataframe(null_index, null_index_path, compression="gzip")
        _write_null_image_atomic(
            null_h5,
            image_key=image_key,
            selected_z_1based=int(image_row["selected_z_1based"]),
            null_groups=reconstruction.null_groups,
            parameters=params,
        )
        null_token = _h5_token(image_key, keep_dot=True)
        if not _h5_checkpoint_complete(
            null_h5,
            group_path=f"images/{null_token}",
            pending_path=f"_pending_images/{null_token}",
            image_key=image_key,
        ):
            raise RuntimeError(f"null HDF5 readback failed for {image_key}")
        spot_table = reconstruction.spot_metrics
        null_telemetry = {
            "n_null_candidate_spots": int(
                spot_table["null_candidate"].map(_explicit_bool).sum()
            ),
            "n_null_usable_spots": int(
                spot_table["null_usable"].map(_explicit_bool).sum()
            ),
            "n_unplaceable_total": int(
                pd.to_numeric(spot_table["n_unplaceable"], errors="coerce")
                .fillna(0)
                .sum()
            ),
            "n_redraws_total": int(
                pd.to_numeric(spot_table["n_redraws"], errors="coerce")
                .fillna(0)
                .sum()
            ),
        }
        for percentile in THRESHOLD_PERCENTILES:
            labels_for_threshold = spot_table[
                f"population_label_q{percentile}"
            ].astype(str)
            for label in (
                "threshold_positive", "threshold_negative", "unusable",
                "below_miat_floor",
            ):
                null_telemetry[f"n_q{percentile}_{label}"] = int(
                    labels_for_threshold.eq(label).sum()
                )
        n_spots_complete = int(len(spot_table))
        n_nuclei_complete = int(len(nucleus_table))
        n_null_nuclei = int(len(reconstruction.null_groups))
        del planes, labels, nucleoli, reconstruction
        gc.collect()
        resources = _resource_snapshot(output)
        if validate_expected_design:
            _enforce_resource_gates(resources, preflight=False)
        phase2_elapsed = float(time.perf_counter() - image_started)
        _atomic_write_json(
            {
                "image_key": image_key,
                "phase": "phase2",
                "n_spots": n_spots_complete,
                "n_nuclei": n_nuclei_complete,
                "n_null_nuclei": n_null_nuclei,
                "image_elapsed_s": phase2_elapsed,
                "total_elapsed_s": float(time.perf_counter() - run_started),
                **null_telemetry,
                **resources,
            },
            phase2_done,
        )
        manifest.at[manifest_index, "load_status"] = "complete"
        manifest.at[manifest_index, "phase2_elapsed_s"] = phase2_elapsed
        manifest.at[manifest_index, "post_phase2_rss_bytes"] = resources["rss_bytes"]
        print(
            f"PHASE2 {len(final_spot_fragments)}/{len(selected_image_keys)} "
            f"{image_key} spots={n_spots_complete} null_nuclei={n_null_nuclei} "
            f"usable={null_telemetry['n_null_usable_spots']} "
            f"q95+={null_telemetry['n_q95_threshold_positive']} "
            f"unusable={null_telemetry['n_q95_unusable']} "
            f"elapsed={phase2_elapsed:.2f}s rss={resources['rss_bytes']} "
            f"free={resources['output_free_bytes']}",
            flush=True,
        )
        _atomic_write_dataframe(manifest, manifest_path)
        del nucleus_table, null_index, spot_table
        gc.collect()

    if not all(
        path.is_file()
        for path in [
            checkpoints / hashlib.sha256(str(key).casefold().encode("utf-8")).hexdigest()[:16] / "phase2.done.json"
            for key in selected_image_keys
        ]
    ):
        raise RuntimeError("phase-2 checkpoint reconciliation failed")
    _combine_csv_fragments(
        final_spot_fragments,
        output / "spot_exact_footprint_metrics.csv.gz",
        compressed=True,
    )
    _combine_csv_fragments(
        phase1_fragments,
        output / "footprint_pixels.csv.gz",
        compressed=True,
    )
    _combine_csv_fragments(
        null_index_fragments,
        output / "null_draw_index.csv.gz",
        compressed=True,
    )
    _combine_csv_fragments(
        nucleus_fragments,
        output / "nucleus_exact_footprint_metrics.csv",
        compressed=False,
    )
    _assert_no_pending_h5_groups(selected_h5)
    _assert_no_pending_h5_groups(null_h5)
    _atomic_write_dataframe(manifest, manifest_path)
    provenance = _finalize_provenance_outputs(
        output=output,
        manifest=manifest,
        input_checksums=input_checksums,
        spot_path=output / "spot_exact_footprint_metrics.csv.gz",
        pixel_path=output / "footprint_pixels.csv.gz",
        nucleus_path=output / "nucleus_exact_footprint_metrics.csv",
        null_index_path=output / "null_draw_index.csv.gz",
    )
    summary = {
        "run_status": "complete",
        **native_provenance,
        "analysis_scope": analysis_scope,
        "phase1_only": False,
        "selected_image_keys": selected_image_keys,
        "n_images_complete": int(manifest["load_status"].eq("complete").sum()),
        "n_images_full_roster": int(len(manifest)),
        "n_spots": parity_total,
        "n_controls": int(
            (
                manifest["selected_for_execution"].map(_explicit_bool)
                & manifest["is_control"].map(_explicit_bool)
            ).sum()
        ),
        "n_biological": int(
            (
                manifest["selected_for_execution"].map(_explicit_bool)
                & ~manifest["is_control"].map(_explicit_bool)
            ).sum()
        ),
        "full_parity_gate_pass": analysis_scope == "full_manifest",
        "eligible_for_biological_inference": biological_inference_output,
        "execution_fingerprint": execution_fingerprint,
        "design_check_skipped": not bool(validate_expected_design),
        "observed_design": list(manifest.attrs.get("observed_design", ())),
        "audited_design": [44, 37, 7, 12],
        "output_dir": str(output.resolve()),
        **provenance,
    }
    _atomic_write_json(summary, output / "validation_report.json")
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reconstruct exact MIAT footprints on recorded single z planes."
    )
    parser.add_argument("--run", required=True, help="Completed FishSuite run")
    parser.add_argument("--hierarchy", help="Authoritative nuclei_coloc_derived.csv; otherwise use recorded native hierarchy")
    parser.add_argument("--output-root", help="Parent for a new timestamped output")
    parser.add_argument("--miat-floor", type=float, default=364.0)
    parser.add_argument("--n-null", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-redraw", type=int, default=1000)
    parser.add_argument("--min-first-pass-retention", type=float, default=0.5)
    parser.add_argument("--min-valid-draw-fraction", type=float, default=1.0)
    parser.add_argument(
        "--image-key",
        action="append",
        help=(
            "Run only this manifest image (repeatable). The complete 44-image "
            "manifest is still validated and the output is marked smoke-only."
        ),
    )
    parser.add_argument(
        "--phase1-only",
        action="store_true",
        help="Stop after exact-footprint parity; do not calculate KEEP-N nulls.",
    )
    parser.add_argument(
        "--skip-design-check",
        action="store_true",
        help=(
            "Skip the audited MIAT/QKI design assertion (44 images, 37 biological, "
            "7 controls, 12 sets) so a run with a different design can be processed. "
            "Validation is ON by default. The observed design tuple is logged and "
            "recorded in the output provenance."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate roster/paths without reading pixels")
    parser.add_argument("--resume", type=Path, help="Resume an existing exact-footprint output directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.  Full serial orchestration is added below this core."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    run_dir = Path(args.run)
    hierarchy_path = Path(args.hierarchy) if args.hierarchy else None
    required = [
        run_dir / "per_image_summary.csv",
        run_dir / "spot_metrics.csv",
        run_dir / "nuclei_metrics.csv",
        run_dir / "run_config.json",
    ]
    if hierarchy_path is not None:
        required.append(hierarchy_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error(f"missing required inputs: {missing}")
    run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    validate_design = hierarchy_path is not None and not args.skip_design_check
    try:
        hierarchy = (native_hierarchy_from_run(run_dir) if hierarchy_path is None
                     else pd.read_csv(hierarchy_path))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    manifest = build_image_manifest(
        pd.read_csv(run_dir / "per_image_summary.csv"),
        hierarchy,
        pd.read_csv(run_dir / "spot_metrics.csv"),
        run_config=run_config,
        run_dir=run_dir,
        validate_expected_design=validate_design,
        resolve_paths=True,
    )
    if hierarchy_path is None:
        print("DESIGN CHECK not_applicable_native_hierarchy: observed design "
              f"(images, biological, controls, sets) = {manifest.attrs.get('observed_design')}")
    elif not validate_design:
        print(
            "DESIGN CHECK SKIPPED (--skip-design-check): observed design "
            f"(images, biological, controls, sets) = {manifest.attrs.get('observed_design')}; "
            "audited MIAT/QKI design is (44, 37, 7, 12)"
        )
    selected_image_keys, analysis_scope = _select_execution_keys(
        manifest, args.image_key
    )
    resolved = run_config.get("config_resolved", {})
    foci = resolved.get("foci", {}) if isinstance(resolved, Mapping) else {}
    nucleolus = resolved.get("nucleolus", {}) if isinstance(resolved, Mapping) else {}
    nucleolus_required = bool(
        foci.get("exclude_nucleolus_from_partner_null", False)
        and nucleolus.get("enabled", False)
    )
    if args.dry_run:
        ets_total = int(
            pd.to_numeric(
                manifest.get("companion_ets_size_bytes", pd.Series(dtype=float)),
                errors="coerce",
            ).fillna(0).sum()
        )
        print(
            f"DRY-RUN PASS: {len(manifest)} images; "
            f"{int((~manifest['is_control']).sum())} biological; "
            f"{int(manifest['is_control'].sum())} controls; exact recorded z only; "
            f"unique masks; VSI companions={len(manifest)} ({ets_total} bytes); "
            f"nucleolus exclusion required={nucleolus_required}; "
            f"analysis scope={analysis_scope}; selected images="
            f"{len(selected_image_keys)}; phase1 only={bool(args.phase1_only)}"
        )
        return 0
    parameters = ExactFootprintParameters(
        miat_floor_raw=args.miat_floor,
        n_null=args.n_null,
        global_seed=args.seed,
        max_redraw=args.max_redraw,
        min_first_pass_retention=args.min_first_pass_retention,
        min_valid_draw_fraction=args.min_valid_draw_fraction,
    )
    if args.resume is not None:
        output = Path(args.resume)
        resume = True
    else:
        if not args.output_root and hierarchy_path is not None:
            parser.error("--output-root is required for a new production run")
        if analysis_scope == "smoke_subset":
            prefix = "EXACT_FOOTPRINT_SMOKE_SUBSET_"
        elif args.phase1_only:
            prefix = "EXACT_FOOTPRINT_PHASE1_ONLY_"
        else:
            prefix = "EXACT_FOOTPRINT_BACKFILL_"
        output = (Path(args.output_root) if args.output_root else run_dir.resolve().parent) / (
            prefix + datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        resume = False
    summary = run_exact_footprint_backfill(
        run_dir,
        hierarchy_path,
        output,
        parameters=parameters,
        resume=resume,
        validate_expected_design=validate_design,
        image_keys=args.image_key,
        phase1_only=args.phase1_only,
        **({"allow_output_inside_run": bool(args.output_root)}
           if hierarchy_path is None else {}),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
