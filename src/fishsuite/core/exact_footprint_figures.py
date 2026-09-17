"""Exact-single-plane MIAT x QKI representative and threshold figures.

This module is deliberately display-only.  It consumes frozen raw planes,
footprint pixels, and canonical calls; it never performs detection, z-plane
selection, projection, segmentation, or null calculation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


POSITIVE_COLOR = "#D55E00"
NEGATIVE_COLOR = "#0072B2"
UNUSABLE_COLOR = "#7F7F7F"

PRIMARY_DISPLAY_WINDOWS: dict[str, tuple[float, float]] = {
    "dapi": (334.0, 8000.0),
    "miat": (400.0, 4000.0),
    "qki": (555.0, 3000.0),
}
DISPLAY_SENSITIVITY_WINDOWS: dict[str, tuple[float, float]] = {
    "dapi_prior_qa": (334.0, 5500.0),
    "miat_high": (700.0, 5000.0),
}


def display_channel(raw: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Map a raw channel to uint8 without mutating the quantitative array."""

    lo = float(vmin)
    hi = float(vmax)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError("display limits must be finite with vmax > vmin")
    source = np.asarray(raw)
    scaled = ((source.astype(np.float64, copy=True) - lo) / (hi - lo)) * 255.0
    return np.floor(np.clip(scaled, 0.0, 255.0) + 0.5).astype(np.uint8)


def render_channel_rgb(
    raw: np.ndarray,
    channel: str,
    *,
    display_window: tuple[float, float] | None = None,
) -> np.ndarray:
    """Apply a fixed wavelength LUT to a display-only uint8 copy."""

    role = str(channel).strip().casefold()
    if role not in PRIMARY_DISPLAY_WINDOWS:
        raise ValueError(f"unknown channel role: {channel!r}")
    window = display_window or PRIMARY_DISPLAY_WINDOWS[role]
    gray = display_channel(raw, *window)
    zero = np.zeros_like(gray)
    if role == "dapi":
        return np.stack((zero, zero, gray), axis=-1)
    if role == "miat":
        return np.stack((gray, gray, zero), axis=-1)
    return np.stack((gray, zero, gray), axis=-1)


@dataclass(frozen=True)
class SelectedPlaneData:
    """Frozen 2D raw planes and masks from one complete HDF5 image group."""

    image_key: str
    selected_z_1based: int
    selected_z_0based: int
    channel_indices: dict[str, int]
    planes: dict[str, np.ndarray]
    nucleus_labels: np.ndarray
    nucleolus_labels: np.ndarray


@dataclass(frozen=True)
class FigurePackageOutputs:
    """Paths and frozen audits emitted by one figure-only package render."""

    output_dir: Path
    selection: RepresentativeSelection
    clean_grid_paths: tuple[Path, ...]
    walkthrough_paths: tuple[Path, ...]
    overlay_paths: tuple[Path, ...]
    sidecar_paths: tuple[Path, ...]
    threshold_summary_path: Path


@dataclass(frozen=True)
class ControlPanelHooks:
    """Explicit QC-only keys for the clean zero and artifact anomaly panels."""

    zero_control_image_key: str
    anomaly_image_key: str
    anomaly_nucleus_ids: tuple[int, ...] = (35, 30)


@dataclass(frozen=True)
class ControlDiagnosticOutputs:
    """Paths emitted by the standalone omission-control diagnostic renderer."""

    output_dir: Path
    ledger_path: Path
    ledger_figure_paths: tuple[Path, ...]
    zero_example_paths: tuple[Path, ...]
    anomaly_paths: tuple[Path, ...]
    sidecar_path: Path


@dataclass(frozen=True)
class PreparedRendererAudits:
    """Fail-closed adapters derived from recorded backfill QC artifacts."""

    image_audit: pd.DataFrame
    nucleus_qc_roster: pd.DataFrame


def load_selected_plane(
    h5_path: str | Path,
    image_key: str,
    *,
    expected_z_1based: int | None = None,
    expected_channel_indices: Mapping[str, int] | None = None,
) -> SelectedPlaneData:
    """Load one exact cached plane and fail closed on z/channel disagreement."""

    import h5py

    target = str(image_key).casefold()
    path = Path(h5_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        if "images" not in handle:
            raise ValueError("selected-plane HDF5 lacks the images root")
        matches = [
            group
            for group in handle["images"].values()
            if str(group.attrs.get("image_key", "")).casefold() == target
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected one selected-plane group for {target!r}; found {len(matches)}"
            )
        group = matches[0]
        if not bool(group.attrs.get("complete", False)):
            raise ValueError(f"selected-plane group for {target!r} is incomplete")
        if not bool(group.attrs.get("same_plane_all_channels", False)):
            raise ValueError("selected-plane group does not lock all channels to one z")
        z1 = int(group.attrs["selected_z_1based"])
        z0 = int(group.attrs["selected_z_0based"])
        if z0 != z1 - 1:
            raise ValueError("selected z metadata is not an exact 1-based to 0-based conversion")
        if expected_z_1based is not None and z1 != int(expected_z_1based):
            raise ValueError(
                f"selected z mismatch: HDF5={z1}, expected={int(expected_z_1based)}"
            )
        channel_indices = {
            role: int(group.attrs[f"{role}_channel_index"])
            for role in ("miat", "qki", "dapi")
        }
        if expected_channel_indices is not None:
            expected = {
                role: int(expected_channel_indices[role])
                for role in ("miat", "qki", "dapi")
            }
            if channel_indices != expected:
                raise ValueError(
                    f"channel index mismatch: HDF5={channel_indices}, expected={expected}"
                )
        required = {
            "miat",
            "qki",
            "dapi",
            "nucleus_labels",
            "nucleolus_labels",
        }
        missing = sorted(required - set(group.keys()))
        if missing:
            raise ValueError(f"selected-plane group lacks datasets: {missing}")
        planes = {
            role: np.asarray(group[role][:])
            for role in ("miat", "qki", "dapi")
        }
        shape = planes["miat"].shape
        if any(
            plane.ndim != 2
            or plane.shape != shape
            or plane.dtype != np.uint16
            for plane in planes.values()
        ):
            raise ValueError(
                "selected microscopy planes must be same-shape 2D uint16 arrays"
            )
        nucleus_labels = np.asarray(group["nucleus_labels"][:])
        nucleolus_labels = np.asarray(group["nucleolus_labels"][:])
        if (
            nucleus_labels.shape != shape
            or nucleolus_labels.shape != shape
            or not np.issubdtype(nucleus_labels.dtype, np.integer)
            or not np.issubdtype(nucleolus_labels.dtype, np.integer)
        ):
            raise ValueError("selected label masks must be same-shape integer arrays")
    return SelectedPlaneData(
        image_key=target,
        selected_z_1based=z1,
        selected_z_0based=z0,
        channel_indices=channel_indices,
        planes=planes,
        nucleus_labels=nucleus_labels,
        nucleolus_labels=nucleolus_labels,
    )


@dataclass(frozen=True)
class CropBounds:
    """Half-open native-pixel crop bounds with recorded physical geometry."""

    y0: int
    y1: int
    x0: int
    x1: int
    margin_px: int
    voxel_xy_nm: float
    was_clamped: bool

    @property
    def is_square(self) -> bool:
        return (self.y1 - self.y0) == (self.x1 - self.x0)

    @property
    def width_um(self) -> float:
        return (self.x1 - self.x0) * self.voxel_xy_nm / 1000.0

    @property
    def height_um(self) -> float:
        return (self.y1 - self.y0) * self.voxel_xy_nm / 1000.0


def _clamped_interval(start: int, stop: int, size: int, limit: int) -> tuple[int, int]:
    extra = int(size) - (int(stop) - int(start))
    lo = int(start) - extra // 2
    hi = lo + int(size)
    if size >= limit:
        return 0, int(limit)
    if lo < 0:
        hi -= lo
        lo = 0
    if hi > limit:
        lo -= hi - limit
        hi = int(limit)
    return int(lo), int(hi)


def nucleus_square_crop(
    nucleus_labels: np.ndarray,
    nucleus_id: int,
    voxel_xy_nm: float,
    *,
    margin_um: float = 2.0,
) -> CropBounds:
    """Return the complete nucleus bbox + margin, squared and field-clamped."""

    labels = np.asarray(nucleus_labels)
    if labels.ndim != 2:
        raise ValueError("nucleus_labels must be a two-dimensional label image")
    voxel = float(voxel_xy_nm)
    if not np.isfinite(voxel) or voxel <= 0:
        raise ValueError("voxel_xy_nm must be finite and positive")
    margin = float(margin_um)
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("margin_um must be finite and non-negative")
    yy, xx = np.nonzero(labels == int(nucleus_id))
    if yy.size == 0:
        raise ValueError(f"nucleus_id {nucleus_id} is absent from the label image")
    margin_px = int(np.ceil(margin * 1000.0 / voxel))
    raw_y0 = int(yy.min()) - margin_px
    raw_y1 = int(yy.max()) + 1 + margin_px
    raw_x0 = int(xx.min()) - margin_px
    raw_x1 = int(xx.max()) + 1 + margin_px
    side = max(raw_y1 - raw_y0, raw_x1 - raw_x0)
    y0, y1 = _clamped_interval(raw_y0, raw_y1, side, labels.shape[0])
    x0, x1 = _clamped_interval(raw_x0, raw_x1, side, labels.shape[1])
    was_clamped = bool(
        y0 != raw_y0 - (side - (raw_y1 - raw_y0)) // 2
        or x0 != raw_x0 - (side - (raw_x1 - raw_x0)) // 2
        or (y1 - y0) != side
        or (x1 - x0) != side
    )
    return CropBounds(
        y0=y0,
        y1=y1,
        x0=x0,
        x1=x1,
        margin_px=margin_px,
        voxel_xy_nm=voxel,
        was_clamped=was_clamped,
    )


@dataclass(frozen=True)
class OverlaySpec:
    """Vector instructions derived only from one stored exact footprint."""

    spot_uid: str
    boundary_segments_xy: np.ndarray
    center_xy: tuple[float, float]
    call_class: str
    color: str
    line_style: str
    draw_center_x: bool
    footprint_method: str


def _explicit_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    raise ValueError(f"value is not an explicit boolean: {value!r}")


def exact_footprint_outline_segments(
    footprint_yx: np.ndarray,
    *,
    x_offset: int = 0,
    y_offset: int = 0,
) -> np.ndarray:
    """Trace exterior pixel edges of an exact footprint in native coordinates."""

    pixels = np.asarray(footprint_yx)
    if pixels.ndim != 2 or pixels.shape[1] != 2 or pixels.shape[0] == 0:
        raise ValueError("footprint_yx must be a non-empty (n_pixels, 2) array")
    if not np.isfinite(pixels.astype(float)).all():
        raise ValueError("footprint coordinates must be finite")
    integer = pixels.astype(np.int64)
    if not np.array_equal(integer.astype(float), pixels.astype(float)):
        raise ValueError("footprint coordinates must be integer pixels")
    pixel_set = {(int(y), int(x)) for y, x in integer}
    if len(pixel_set) != len(integer):
        raise ValueError("exact footprint contains duplicate pixel coordinates")
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for y, x in sorted(pixel_set):
        left = float(x - x_offset) - 0.5
        right = left + 1.0
        top = float(y - y_offset) - 0.5
        bottom = top + 1.0
        if (y - 1, x) not in pixel_set:
            segments.append(((left, top), (right, top)))
        if (y + 1, x) not in pixel_set:
            segments.append(((left, bottom), (right, bottom)))
        if (y, x - 1) not in pixel_set:
            segments.append(((left, top), (left, bottom)))
        if (y, x + 1) not in pixel_set:
            segments.append(((right, top), (right, bottom)))
    return np.asarray(segments, dtype=np.float64)


def build_call_overlay_specs(
    spots: pd.DataFrame,
    footprint_pixels: pd.DataFrame,
    *,
    percentile: int = 95,
    crop: CropBounds | None = None,
) -> list[OverlaySpec]:
    """Build q-call styles on exact stored footprints; centers stay separate."""

    if percentile not in {90, 95, 99}:
        raise ValueError("percentile must be one of 90, 95, or 99")
    required_spots = {
        "spot_uid",
        "center_x_px",
        "center_y_px",
        "null_usable",
        f"qki_threshold_positive_q{percentile}",
        "footprint_method",
    }
    missing = sorted(required_spots - set(spots.columns))
    if missing:
        raise ValueError(f"spot table lacks overlay columns: {missing}")
    missing_pixels = sorted({"spot_uid", "y_px", "x_px"} - set(footprint_pixels.columns))
    if missing_pixels:
        raise ValueError(f"footprint pixel table lacks columns: {missing_pixels}")
    if spots["spot_uid"].astype(str).duplicated().any():
        raise ValueError("spot_uid must be unique in the overlay spot table")
    x_offset = int(crop.x0) if crop is not None else 0
    y_offset = int(crop.y0) if crop is not None else 0
    pixel_groups = {
        str(uid): group
        for uid, group in footprint_pixels.groupby("spot_uid", sort=False)
    }
    specs: list[OverlaySpec] = []
    for row in spots.assign(_uid=spots["spot_uid"].astype(str)).sort_values(
        "_uid", kind="mergesort"
    ).to_dict("records"):
        uid = str(row["spot_uid"])
        if uid not in pixel_groups:
            raise ValueError(f"spot_uid {uid!r} has no exact footprint pixels")
        usable = _explicit_bool(row["null_usable"])
        if usable:
            call_value = row[f"qki_threshold_positive_q{percentile}"]
            if pd.isna(call_value):
                raise ValueError(f"usable spot {uid!r} lacks a q{percentile} call")
            positive = _explicit_bool(call_value)
            call_class = "threshold_positive" if positive else "threshold_negative"
            color = POSITIVE_COLOR if positive else NEGATIVE_COLOR
            line_style = "-" if positive else "--"
        else:
            call_class = "unusable"
            color = UNUSABLE_COLOR
            line_style = ":"
        group = pixel_groups[uid]
        segments = exact_footprint_outline_segments(
            group[["y_px", "x_px"]].to_numpy(),
            x_offset=x_offset,
            y_offset=y_offset,
        )
        specs.append(
            OverlaySpec(
                spot_uid=uid,
                boundary_segments_xy=segments,
                center_xy=(
                    float(row["center_x_px"]) - x_offset,
                    float(row["center_y_px"]) - y_offset,
                ),
                call_class=call_class,
                color=color,
                line_style=line_style,
                draw_center_x=not usable,
                footprint_method=str(row["footprint_method"]),
            )
        )
    return specs


@dataclass(frozen=True)
class RepresentativeSelection:
    """Complete deterministic field/nucleus selection audit and compact manifest."""

    field_audit: pd.DataFrame
    nucleus_audit: pd.DataFrame
    manifest: pd.DataFrame


def _require_columns(frame: pd.DataFrame, required: set[str], *, table: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{table} lacks required columns: {missing}")


def _qc_pass(value: object) -> bool:
    return str(value).strip().casefold() not in {
        "fail",
        "failed",
        "exclude",
        "excluded",
    }


def _bool_series(series: pd.Series, *, name: str) -> pd.Series:
    try:
        return series.map(_explicit_bool).astype(bool)
    except ValueError as exc:
        raise ValueError(f"{name} contains a non-boolean value") from exc


def _median_mad(values: pd.Series | np.ndarray) -> tuple[float, float]:
    array = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
    array = array[np.isfinite(array)]
    if not array.size:
        return np.nan, np.nan
    median = float(np.median(array))
    return median, float(np.median(np.abs(array - median)))


def _slide_order(value: object) -> tuple[int, str]:
    text = str(value).strip().casefold()
    match = re.search(r"\d+", text)
    return (int(match.group()) if match else 10**9, text)


def _arm_order(value: object) -> tuple[int, str]:
    text = str(value).strip().casefold()
    return ({"nt": 0, "kd": 1}.get(text, 2), text)


def format_clean_grid_row_label(
    slide: object,
    arm: object,
    biological_set: object,
) -> str:
    """Return a publication-facing row label without float slide notation."""

    slide_text = str(slide).strip()
    numeric_match = re.fullmatch(
        r"(?:slide\s*)?([+-]?\d+(?:\.\d+)?)",
        slide_text,
        flags=re.IGNORECASE,
    )
    if numeric_match:
        numeric_slide = float(numeric_match.group(1))
        slide_label = (
            str(int(numeric_slide))
            if np.isfinite(numeric_slide) and numeric_slide.is_integer()
            else numeric_match.group(1)
        )
    else:
        slide_label = re.sub(
            r"^slide\s*", "", slide_text, flags=re.IGNORECASE
        )

    arm_text = str(arm).strip()
    arm_label = {
        "nt": "NT",
        "kd": "MIAT-KD",
        "miat-kd": "MIAT-KD",
    }.get(arm_text.casefold(), arm_text)
    return f"Slide {slide_label} — {arm_label}\n{biological_set}"


def _nucleus_rows_for_image(
    image_key: str,
    spots: pd.DataFrame,
    nuclei: pd.DataFrame,
) -> pd.DataFrame:
    scoped_spots = spots.loc[spots["image_key"].eq(image_key)]
    scoped_nuclei = nuclei.loc[
        nuclei["image_key"].eq(image_key)
        & nuclei["nucleus_qc_status"].map(_qc_pass)
    ]
    rows: list[dict[str, object]] = []
    for nucleus in scoped_nuclei.sort_values(
        "nucleus_id", kind="mergesort"
    ).to_dict("records"):
        nucleus_id = int(nucleus["nucleus_id"])
        group = scoped_spots.loc[
            pd.to_numeric(
                scoped_spots["nucleus_id"], errors="coerce"
            ).eq(nucleus_id)
        ]
        usable = (
            _bool_series(group["null_usable"], name="null_usable")
            if len(group)
            else pd.Series(dtype=bool)
        )
        calls = pd.Series(False, index=group.index, dtype=bool)
        if usable.any():
            calls.loc[usable] = _bool_series(
                group.loc[usable, "qki_threshold_positive_q95"],
                name="qki_threshold_positive_q95",
            )
        positive = usable & calls
        negative = usable & ~calls
        nonnucleolar = (
            ~_bool_series(
                group["stored_in_nucleolus"], name="stored_in_nucleolus"
            )
            if len(group)
            else pd.Series(dtype=bool)
        )
        n_usable = int(usable.sum())
        rows.append(
            {
                "image_key": image_key,
                "nucleus_uid": str(
                    nucleus.get(
                        "nucleus_uid", f"{image_key}:nucleus:{nucleus_id}"
                    )
                ),
                "nucleus_id": nucleus_id,
                "nucleus_qc_status": str(nucleus["nucleus_qc_status"]),
                "n_usable": n_usable,
                "n_positive_q95": int(positive.sum()),
                "n_negative_q95": int(negative.sum()),
                "n_unusable": int(len(group) - n_usable),
                "association_fraction_q95": (
                    float(positive.sum() / n_usable)
                    if n_usable
                    else np.nan
                ),
                "n_spots_non_nucleolar": int(nonnucleolar.sum()),
            }
        )
    return pd.DataFrame(rows)


def select_representatives(
    image_audit: pd.DataFrame,
    spots: pd.DataFrame,
    nuclei: pd.DataFrame,
) -> RepresentativeSelection:
    """Select robust-medoid fields and objective nuclei without cherry-picking.

    Field q95 association is computed directly as sum(positive)/sum(usable).
    The nucleus roster is authoritative, so QC-passing nuclei with zero spots
    contribute zeros to the per-field median non-nucleolar spot count.
    """

    image_required = {
        "image_key",
        "image",
        "slide",
        "arm",
        "is_control",
        "secondary_only",
        "selected_z_1based",
        "voxel_xy_nm",
        "plane_lock_pass",
        "mask_qc_pass",
        "footprint_parity_pass",
        "population_reconciliation_pass",
        "image_qc_status",
    }
    spot_required = {
        "image_key",
        "nucleus_id",
        "spot_uid",
        "null_usable",
        "qki_threshold_positive_q95",
        "stored_in_nucleolus",
        "qki_footprint_enrichment_vs_nucleoplasm",
    }
    nucleus_required = {
        "image_key",
        "nucleus_id",
        "nucleus_uid",
        "nucleus_qc_status",
    }
    _require_columns(image_audit, image_required, table="image audit")
    _require_columns(spots, spot_required, table="corrected spot table")
    _require_columns(nuclei, nucleus_required, table="nucleus QC roster")
    images = image_audit.copy()
    spot_table = spots.copy()
    nucleus_table = nuclei.copy()
    for frame in (images, spot_table, nucleus_table):
        frame["image_key"] = frame["image_key"].astype(str).str.casefold()
    if images["image_key"].duplicated().any():
        raise ValueError("image audit must contain one row per image_key")
    if spot_table["spot_uid"].astype(str).duplicated().any():
        raise ValueError("corrected spot table spot_uid values must be unique")
    if nucleus_table.duplicated(["image_key", "nucleus_id"]).any():
        raise ValueError(
            "nucleus QC roster must be unique by image_key/nucleus_id"
        )
    known_images = set(images["image_key"])
    if not set(spot_table["image_key"]).issubset(known_images):
        raise ValueError(
            "corrected spot table contains image_key values absent from image audit"
        )
    if not set(nucleus_table["image_key"]).issubset(known_images):
        raise ValueError(
            "nucleus QC roster contains image_key values absent from image audit"
        )

    field_rows: list[dict[str, object]] = []
    nucleus_cache: dict[str, pd.DataFrame] = {}
    pass_columns = (
        "plane_lock_pass",
        "mask_qc_pass",
        "footprint_parity_pass",
        "population_reconciliation_pass",
    )
    for image in images.sort_values(
        "image_key", kind="mergesort"
    ).to_dict("records"):
        key = str(image["image_key"])
        group = spot_table.loc[spot_table["image_key"].eq(key)]
        nucleus_rows = _nucleus_rows_for_image(
            key, spot_table, nucleus_table
        )
        nucleus_cache[key] = nucleus_rows
        usable = (
            _bool_series(group["null_usable"], name="null_usable")
            if len(group)
            else pd.Series(dtype=bool)
        )
        calls = pd.Series(False, index=group.index, dtype=bool)
        if usable.any():
            calls.loc[usable] = _bool_series(
                group.loc[usable, "qki_threshold_positive_q95"],
                name="qki_threshold_positive_q95",
            )
        positive = usable & calls
        negative = usable & ~calls
        n_usable = int(usable.sum())
        counts = (
            nucleus_rows["n_spots_non_nucleolar"].astype(float)
            if len(nucleus_rows)
            else pd.Series(dtype=float)
        )
        enrich = pd.to_numeric(
            group.loc[
                usable, "qki_footprint_enrichment_vs_nucleoplasm"
            ],
            errors="coerce",
        )
        mixed_nucleus = bool(
            len(nucleus_rows)
            and (
                (nucleus_rows["n_positive_q95"] > 0)
                & (nucleus_rows["n_negative_q95"] > 0)
            ).any()
        )
        field_both = bool(positive.any() and negative.any())
        hard_checks = {
            column: _explicit_bool(image[column])
            for column in pass_columns
        }
        hard_eligible = bool(
            not _explicit_bool(image["is_control"])
            and not _explicit_bool(image["secondary_only"])
            and all(hard_checks.values())
            and _qc_pass(image["image_qc_status"])
            and len(nucleus_rows) > 0
            and n_usable > 0
        )
        record = dict(image)
        record.update(
            {
                "hard_eligible": hard_eligible,
                "n_qc_passing_nuclei": int(len(nucleus_rows)),
                "n_usable": n_usable,
                "n_positive_q95": int(positive.sum()),
                "n_negative_q95": int(negative.sum()),
                "n_unusable": int(len(group) - n_usable),
                "unusable_fraction": (
                    float((len(group) - n_usable) / len(group))
                    if len(group)
                    else np.nan
                ),
                "field_has_both_classes": field_both,
                "mixed_nucleus_exists": mixed_nucleus,
                "content_tier": (
                    1 if mixed_nucleus else (2 if field_both else 3)
                ),
                "association_fraction_q95": (
                    float(positive.sum() / n_usable)
                    if n_usable
                    else np.nan
                ),
                "median_non_nucleolar_spots_per_nucleus": (
                    float(counts.median()) if len(counts) else np.nan
                ),
                "median_qki_enrichment_vs_nucleoplasm": (
                    float(enrich[np.isfinite(enrich)].median())
                    if np.isfinite(enrich).any()
                    else np.nan
                ),
                "rank_within_tier": pd.NA,
                "selected": False,
                "exclusion_or_fallback_reason": "",
            }
        )
        if not hard_eligible:
            reasons: list[str] = []
            if _explicit_bool(image["is_control"]) or _explicit_bool(
                image["secondary_only"]
            ):
                reasons.append("control")
            reasons.extend(
                column for column, passed in hard_checks.items() if not passed
            )
            if not _qc_pass(image["image_qc_status"]):
                reasons.append("image_qc_failed")
            if not len(nucleus_rows):
                reasons.append("no_qc_passing_nuclei")
            if n_usable == 0:
                reasons.append("no_q95_usable_spots")
            record["exclusion_or_fallback_reason"] = ";".join(reasons)
        field_rows.append(record)
    fields = pd.DataFrame(field_rows)

    metrics = (
        "association_fraction_q95",
        "median_non_nucleolar_spots_per_nucleus",
        "median_qki_enrichment_vs_nucleoplasm",
    )
    grouped = fields.groupby(["slide", "arm"], sort=False).groups
    for indexes in grouped.values():
        cell_indexes = list(indexes)
        hard = fields.loc[cell_indexes]
        hard = hard.loc[hard["hard_eligible"]]
        if hard.empty:
            continue
        for metric in metrics:
            median, mad = _median_mad(hard[metric])
            fields.loc[
                cell_indexes, f"cell_median_{metric}"
            ] = median
            fields.loc[cell_indexes, f"cell_mad_{metric}"] = mad
            values = pd.to_numeric(
                fields.loc[cell_indexes, metric], errors="coerce"
            ).to_numpy(float)
            term = np.full(len(values), np.nan, dtype=float)
            if np.isfinite(median) and np.isfinite(mad) and mad > 0:
                finite = np.isfinite(values)
                term[finite] = np.abs(values[finite] - median) / mad
            fields.loc[
                cell_indexes, f"distance_term_{metric}"
            ] = term
        distance_columns = [
            f"distance_term_{metric}" for metric in metrics
        ]
        fields.loc[cell_indexes, "medoid_distance"] = fields.loc[
            cell_indexes, distance_columns
        ].sum(axis=1, skipna=True)
        highest_tier = int(hard["content_tier"].min())
        candidates = fields.loc[cell_indexes]
        candidates = candidates.loc[
            candidates["hard_eligible"]
            & candidates["content_tier"].eq(highest_tier)
        ].copy()
        assoc_median = float(
            fields.loc[
                cell_indexes,
                "cell_median_association_fraction_q95",
            ].iloc[0]
        )
        candidates["_assoc_abs_dev"] = (
            pd.to_numeric(
                candidates["association_fraction_q95"], errors="coerce"
            )
            - assoc_median
        ).abs()
        candidates["_unusable_sort"] = pd.to_numeric(
            candidates["unusable_fraction"], errors="coerce"
        ).fillna(np.inf)
        candidates = candidates.sort_values(
            [
                "medoid_distance",
                "_assoc_abs_dev",
                "_unusable_sort",
                "n_usable",
                "image_key",
            ],
            ascending=[True, True, True, False, True],
            kind="mergesort",
        )
        fields.loc[
            candidates.index, "rank_within_tier"
        ] = np.arange(1, len(candidates) + 1)
        fields.loc[candidates.index[0], "selected"] = True
        lower = (
            fields.index.isin(cell_indexes)
            & fields["hard_eligible"]
            & fields["content_tier"].ne(highest_tier)
        )
        fields.loc[
            lower, "exclusion_or_fallback_reason"
        ] = "lower_available_content_tier"

    nucleus_audits: list[pd.DataFrame] = []
    manifests: list[dict[str, object]] = []
    selected_fields = fields.loc[fields["selected"]].copy()
    selected_fields["_slide_order"] = selected_fields["slide"].map(
        _slide_order
    )
    selected_fields["_arm_order"] = selected_fields["arm"].map(_arm_order)
    selected_fields = selected_fields.sort_values(
        ["_slide_order", "_arm_order", "image_key"], kind="mergesort"
    )
    for field in selected_fields.to_dict("records"):
        key = str(field["image_key"])
        audit = nucleus_cache[key].copy()
        if audit.empty:
            raise ValueError(
                f"selected field {key!r} has no QC-passing nuclei"
            )
        assoc_median, assoc_mad = _median_mad(
            audit["association_fraction_q95"]
        )
        log_counts = np.log1p(
            audit["n_spots_non_nucleolar"].astype(float)
        )
        log_median, log_mad = _median_mad(log_counts)
        audit[
            "field_median_association_fraction_q95"
        ] = assoc_median
        audit["field_mad_association_fraction_q95"] = assoc_mad
        audit[
            "field_median_log1p_non_nucleolar_spots"
        ] = log_median
        audit[
            "field_mad_log1p_non_nucleolar_spots"
        ] = log_mad
        assoc_values = pd.to_numeric(
            audit["association_fraction_q95"], errors="coerce"
        ).to_numpy(float)
        assoc_term = np.zeros(len(audit), dtype=float)
        if np.isfinite(assoc_mad) and assoc_mad > 0:
            finite = np.isfinite(assoc_values)
            assoc_term[finite] = (
                np.abs(assoc_values[finite] - assoc_median) / assoc_mad
            )
        count_term = np.zeros(len(audit), dtype=float)
        if np.isfinite(log_mad) and log_mad > 0:
            count_term = (
                np.abs(log_counts.to_numpy(float) - log_median) / log_mad
            )
        audit["nucleus_distance"] = assoc_term + count_term
        audit["content_tier"] = np.where(
            (audit["n_positive_q95"] > 0)
            & (audit["n_negative_q95"] > 0),
            "mixed",
            np.where(
                audit["n_positive_q95"] > 0,
                "positive_bearing",
                np.where(
                    audit["n_negative_q95"] > 0,
                    "negative_bearing",
                    "no_usable",
                ),
            ),
        )
        audit = audit.sort_values(
            ["nucleus_distance", "n_unusable", "nucleus_id"],
            kind="mergesort",
        ).reset_index(drop=True)
        audit["rank"] = np.arange(1, len(audit) + 1)
        audit["selected"] = False
        audit["selection_role"] = ""
        audit["fallback_reason"] = ""
        mixed = audit.index[
            audit["content_tier"].eq("mixed")
        ].tolist()
        if mixed:
            chosen = [(mixed[0], "mixed")]
        elif bool(field["field_has_both_classes"]):
            positive = audit.index[
                audit["n_positive_q95"] > 0
            ].tolist()
            negative = audit.index[
                audit["n_negative_q95"] > 0
            ].tolist()
            if not positive or not negative:
                raise ValueError(
                    f"field {key!r} class summary disagrees with nucleus roster"
                )
            chosen = [
                (positive[0], "positive"),
                (negative[0], "negative"),
            ]
            audit[
                "fallback_reason"
            ] = "classes_split_across_nuclei"
        else:
            chosen = [(0, "representative")]
            audit["fallback_reason"] = "one_q95_class_absent"
        for position, role in chosen:
            audit.loc[position, "selected"] = True
            audit.loc[position, "selection_role"] = role
            row = audit.loc[position]
            manifests.append(
                {
                    "slide": field["slide"],
                    "arm": field["arm"],
                    "image_key": key,
                    "image": field["image"],
                    "biological_set": field.get("biological_set", ""),
                    "replicate": field.get("replicate", ""),
                    "fov": field.get("fov", ""),
                    "selected_z_1based": int(
                        field["selected_z_1based"]
                    ),
                    "voxel_xy_nm": float(field["voxel_xy_nm"]),
                    "nucleus_uid": row["nucleus_uid"],
                    "nucleus_id": int(row["nucleus_id"]),
                    "selection_role": role,
                }
            )
        audit["slide"] = field["slide"]
        audit["arm"] = field["arm"]
        nucleus_audits.append(audit)
    nucleus_audit = (
        pd.concat(nucleus_audits, ignore_index=True)
        if nucleus_audits
        else pd.DataFrame()
    )
    manifest = pd.DataFrame(manifests)
    if not manifest.empty:
        manifest["_slide_order"] = manifest["slide"].map(_slide_order)
        manifest["_arm_order"] = manifest["arm"].map(_arm_order)
        manifest = (
            manifest.sort_values(
                [
                    "_slide_order",
                    "_arm_order",
                    "image_key",
                    "selection_role",
                ],
                kind="mergesort",
            )
            .drop(columns=["_slide_order", "_arm_order"])
            .reset_index(drop=True)
        )
    return RepresentativeSelection(
        field_audit=fields.sort_values(
            "image_key", kind="mergesort"
        ).reset_index(drop=True),
        nucleus_audit=nucleus_audit,
        manifest=manifest,
    )


def _table_and_hash(
    source: pd.DataFrame | str | Path,
) -> tuple[pd.DataFrame, str, str]:
    if isinstance(source, pd.DataFrame):
        frame = source.copy()
        payload = frame.to_csv(index=False, lineterminator="\n").encode(
            "utf-8"
        )
        return frame, hashlib.sha256(payload).hexdigest(), "<dataframe>"
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    return frame, _sha256_file(path), str(path.resolve())


def prepare_renderer_audits(
    image_manifest: pd.DataFrame | str | Path,
    nucleus_metrics: pd.DataFrame | str | Path,
    population_reconciliation: pd.DataFrame | str | Path,
    full_parity_gate: Mapping[str, Any] | str | Path,
) -> PreparedRendererAudits:
    """Derive renderer audits only from explicit completed-backfill evidence.

    image_manifest.csv supplies hierarchy, exact z/channel locks, qc_pass,
    load_status, and mask_status.  full_historical_parity_gate.json supplies
    the all-image exact-area/QKI parity proof.  Image rows of
    population_reconciliation.csv supply q90/q95/q99 reconciliation.
    nucleus_exact_footprint_metrics.csv supplies recorded include/exclude
    decisions. Missing or contradictory evidence raises instead of passing.
    """

    manifest, _, _ = _table_and_hash(image_manifest)
    nucleus_table, _, _ = _table_and_hash(nucleus_metrics)
    reconciliation, _, _ = _table_and_hash(population_reconciliation)
    if isinstance(full_parity_gate, Mapping):
        parity = dict(full_parity_gate)
    else:
        parity_path = Path(full_parity_gate)
        if not parity_path.is_file():
            raise FileNotFoundError(parity_path)
        parity = json.loads(parity_path.read_text(encoding="utf-8"))
        if not isinstance(parity, dict):
            raise ValueError("full parity gate must contain a JSON object")
    manifest_required = {
        "image_key",
        "image",
        "slide",
        "arm",
        "biological_set",
        "replicate",
        "fov",
        "is_control",
        "secondary_only",
        "selected_z_1based",
        "voxel_xy_nm",
        "miat_channel_index",
        "qki_channel_index",
        "dapi_channel_index",
        "plane_lock_pass",
        "load_status",
        "mask_status",
        "qc_pass",
    }
    _require_columns(
        manifest, manifest_required, table="completed image manifest"
    )
    _require_columns(
        nucleus_table,
        {
            "image_key",
            "nucleus_id",
            "nucleus_uid",
            "include",
            "exclude_reason",
        },
        table="nucleus exact-footprint metrics",
    )
    _require_columns(
        reconciliation,
        {
            "level",
            "key",
            "reconciliation_pass_q90",
            "reconciliation_pass_q95",
            "reconciliation_pass_q99",
        },
        table="population reconciliation",
    )
    images = manifest.copy()
    images["image_key"] = images["image_key"].astype(str).str.casefold()
    if images["image_key"].duplicated().any():
        raise ValueError("completed image manifest contains duplicate image_key rows")
    known = set(images["image_key"])
    parity_required = {
        "pass",
        "area_exact_all_images",
        "qki_allclose_all_images",
        "n_images",
        "selected_image_keys",
    }
    missing_parity = sorted(parity_required - set(parity))
    if missing_parity:
        raise ValueError(
            f"full parity gate lacks required keys: {missing_parity}"
        )
    parity_bools = (
        _explicit_bool(parity["pass"]),
        _explicit_bool(parity["area_exact_all_images"]),
        _explicit_bool(parity["qki_allclose_all_images"]),
    )
    if not all(parity_bools):
        raise ValueError("full parity gate did not pass exact area and QKI parity")
    if int(parity["n_images"]) != len(images):
        raise ValueError("full parity gate n_images disagrees with image manifest")
    parity_keys = [
        str(value).casefold() for value in parity["selected_image_keys"]
    ]
    if len(parity_keys) != len(set(parity_keys)) or set(parity_keys) != known:
        raise ValueError(
            "full parity gate selected_image_keys disagree with image manifest"
        )
    if "analysis_scope" in parity and str(
        parity["analysis_scope"]
    ) != "full_manifest":
        raise ValueError("renderer adapter requires a full_manifest parity gate")

    image_reconciliation = reconciliation.loc[
        reconciliation["level"].astype(str).str.casefold().eq("image")
    ].copy()
    image_reconciliation["image_key"] = (
        image_reconciliation["key"].astype(str).str.casefold()
    )
    if image_reconciliation["image_key"].duplicated().any():
        raise ValueError("population reconciliation has duplicate image rows")
    if set(image_reconciliation["image_key"]) != known:
        raise ValueError(
            "population reconciliation image keys disagree with image manifest"
        )
    reconciliation_columns = [
        f"reconciliation_pass_q{percentile}"
        for percentile in (90, 95, 99)
    ]
    for column in reconciliation_columns:
        image_reconciliation[column] = _bool_series(
            image_reconciliation[column], name=column
        )
    image_reconciliation["population_reconciliation_pass"] = (
        image_reconciliation[reconciliation_columns].all(axis=1)
    )
    images = images.merge(
        image_reconciliation[
            ["image_key", "population_reconciliation_pass"]
        ],
        on="image_key",
        how="left",
        validate="one_to_one",
    )
    images["mask_qc_pass"] = (
        images["load_status"].astype(str).str.casefold().eq("complete")
        & images["mask_status"]
        .astype(str)
        .str.casefold()
        .eq("loaded_reused")
    )
    images["footprint_parity_pass"] = True
    images["image_qc_status"] = np.where(
        _bool_series(images["qc_pass"], name="qc_pass"),
        "pass",
        "failed",
    )
    images["mask_qc_source"] = (
        "image_manifest.load_status=complete AND mask_status=loaded_reused"
    )
    images["footprint_parity_source"] = (
        "full_historical_parity_gate exact-area + QKI all-images pass"
    )
    images["population_reconciliation_source"] = (
        "population_reconciliation image q90/q95/q99 conjunction"
    )

    nuclei = nucleus_table.copy()
    nuclei["image_key"] = nuclei["image_key"].astype(str).str.casefold()
    if nuclei.duplicated(["image_key", "nucleus_id"]).any():
        raise ValueError(
            "nucleus exact-footprint metrics contains duplicate nucleus keys"
        )
    if not set(nuclei["image_key"]).issubset(known):
        raise ValueError(
            "nucleus exact-footprint metrics contains unknown image_key values"
        )
    included = _bool_series(nuclei["include"], name="nucleus include")
    reasons: list[str] = []
    for include, value in zip(
        included, nuclei["exclude_reason"], strict=True
    ):
        reason = "" if pd.isna(value) else str(value).strip()
        if include and reason:
            raise ValueError(
                "included nucleus has a contradictory recorded exclude_reason"
            )
        if not include and not reason:
            reason = "recorded_include_false_without_reason"
        reasons.append(reason)
    nuclei["nucleus_qc_status"] = np.where(
        included, "pass", "failed"
    )
    nuclei["nucleus_qc_reason"] = reasons
    roster = nuclei[
        [
            "image_key",
            "nucleus_id",
            "nucleus_uid",
            "nucleus_qc_status",
            "nucleus_qc_reason",
        ]
    ].copy()
    return PreparedRendererAudits(
        image_audit=images,
        nucleus_qc_roster=roster,
    )


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(value: object) -> str:
    text = str(value).strip().casefold()
    slug = "".join(
        character
        if character.isalnum() or character in {"_", "-", "."}
        else "_"
        for character in text
    ).strip("._")
    return slug or "unnamed"


def _merge_rgb(*channels: np.ndarray) -> np.ndarray:
    arrays = [np.asarray(channel, dtype=np.uint16) for channel in channels]
    if not arrays or any(array.shape != arrays[0].shape for array in arrays):
        raise ValueError("RGB merge inputs must be non-empty and same-shape")
    return np.clip(np.sum(arrays, axis=0), 0, 255).astype(np.uint8)


def _display_rgb_set(
    planes: Mapping[str, np.ndarray],
    windows: Mapping[str, tuple[float, float]],
) -> dict[str, np.ndarray]:
    rendered = {
        role: render_channel_rgb(
            planes[role], role, display_window=windows[role]
        )
        for role in ("dapi", "miat", "qki")
    }
    rendered["merge"] = _merge_rgb(
        rendered["dapi"], rendered["miat"], rendered["qki"]
    )
    return rendered


def _save_rgb_png(array: np.ndarray, path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array, dtype=np.uint8), mode="RGB").save(
        path, format="PNG", compress_level=6
    )


def _write_tiff(path: Path, array: np.ndarray) -> None:
    import tifffile

    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, np.asarray(array), photometric="minisblack")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _save_composite(
    figure: Any,
    base_path: Path,
    *,
    dpi: int,
) -> tuple[Path, Path, Path]:
    paths = tuple(
        base_path.with_suffix(f".{suffix}") for suffix in ("png", "pdf", "svg")
    )
    base_path.parent.mkdir(parents=True, exist_ok=True)
    for path in paths:
        figure.savefig(
            path,
            dpi=int(dpi),
            bbox_inches="tight",
            pad_inches=0.04,
            facecolor="black",
        )
    return paths


def _scale_bar_record(voxel_xy_nm: float, target_um: float) -> dict[str, float | int]:
    pixel_um = float(voxel_xy_nm) / 1000.0
    if not np.isfinite(pixel_um) or pixel_um <= 0:
        raise ValueError("voxel_xy_nm must be finite and positive")
    pixels = max(1, int(round(float(target_um) / pixel_um)))
    actual = pixels * pixel_um
    return {
        "requested_um": float(target_um),
        "pixels": pixels,
        "actual_um": actual,
        "absolute_error_um": abs(actual - float(target_um)),
    }


def _draw_scale_bar(
    axis: Any,
    shape: tuple[int, int],
    *,
    voxel_xy_nm: float,
    target_um: float,
) -> dict[str, float | int]:
    record = _scale_bar_record(voxel_xy_nm, target_um)
    height, width = shape
    length = int(record["pixels"])
    if length >= width:
        raise ValueError(
            f"{target_um:g} um scale bar ({length} px) does not fit image width {width}"
        )
    x1 = width * 0.94
    x0 = x1 - length
    y = height * 0.92
    axis.plot(
        [x0, x1],
        [y, y],
        color="white",
        linewidth=2.0,
        solid_capstyle="butt",
        zorder=20,
    )
    axis.text(
        (x0 + x1) / 2,
        y - max(1.0, height * 0.025),
        f"{target_um:g} µm",
        color="white",
        fontsize=7,
        ha="center",
        va="bottom",
        zorder=20,
    )
    return record


def _show_rgb(axis: Any, image: np.ndarray, title: str) -> None:
    axis.imshow(image, interpolation="nearest")
    axis.set_title(title, color="white", fontsize=8)
    axis.set_xlim(-0.5, image.shape[1] - 0.5)
    axis.set_ylim(image.shape[0] - 0.5, -0.5)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_facecolor("black")


def _show_gray_qki(axis: Any, qki: np.ndarray, title: str) -> None:
    gray = display_channel(qki, *PRIMARY_DISPLAY_WINDOWS["qki"])
    axis.imshow(
        gray,
        cmap="gray",
        vmin=0,
        vmax=255,
        interpolation="nearest",
    )
    axis.set_title(title, color="white", fontsize=8)
    axis.set_xlim(-0.5, gray.shape[1] - 0.5)
    axis.set_ylim(gray.shape[0] - 0.5, -0.5)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_facecolor("black")


def _draw_overlay_specs(
    axis: Any,
    specs: list[OverlaySpec],
    *,
    linewidth: float = 0.8,
) -> None:
    from matplotlib.collections import LineCollection

    for spec in specs:
        axis.add_collection(
            LineCollection(
                spec.boundary_segments_xy,
                colors=[spec.color],
                linewidths=linewidth,
                linestyles=spec.line_style,
                zorder=8,
            )
        )
        if spec.draw_center_x:
            axis.scatter(
                [spec.center_xy[0]],
                [spec.center_xy[1]],
                marker="x",
                s=12,
                color=UNUSABLE_COLOR,
                linewidths=0.7,
                zorder=9,
            )


def _draw_miat_footprints(
    axis: Any,
    spots: pd.DataFrame,
    pixels: pd.DataFrame,
    *,
    crop: CropBounds | None = None,
    show_indices: bool = False,
) -> None:
    from matplotlib.collections import LineCollection

    specs = build_call_overlay_specs(
        spots, pixels, percentile=95, crop=crop
    )
    for index, spec in enumerate(specs):
        fallback = "fallback" in spec.footprint_method.casefold()
        axis.add_collection(
            LineCollection(
                spec.boundary_segments_xy,
                colors=["white"],
                linewidths=0.75,
                linestyles=":" if fallback else "-",
                zorder=8,
            )
        )
        axis.scatter(
            [spec.center_xy[0]],
            [spec.center_xy[1]],
            marker="+",
            s=9,
            color="#56B4E9",
            linewidths=0.55,
            zorder=9,
        )
        if show_indices:
            axis.text(
                spec.center_xy[0] + 0.8,
                spec.center_xy[1] - 0.8,
                str(index + 1),
                color="white",
                fontsize=5,
                zorder=10,
            )


def _pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    return plt


def _draw_mask_outlines(
    axis: Any,
    nucleus_labels: np.ndarray,
    nucleolus_labels: np.ndarray,
) -> None:
    nuclei = np.asarray(nucleus_labels) > 0
    nucleoli = np.asarray(nucleolus_labels) > 0
    if nuclei.any() and not nuclei.all():
        axis.contour(
            nuclei.astype(np.uint8),
            levels=[0.5],
            colors=["white"],
            linewidths=0.45,
            alpha=0.9,
            zorder=6,
        )
    if nucleoli.any() and not nucleoli.all():
        axis.contour(
            nucleoli.astype(np.uint8),
            levels=[0.5],
            colors=["#E69F00"],
            linewidths=0.45,
            alpha=0.9,
            zorder=7,
        )


def _draw_crop_box(axis: Any, crop: CropBounds) -> None:
    from matplotlib.patches import Rectangle

    axis.add_patch(
        Rectangle(
            (crop.x0 - 0.5, crop.y0 - 0.5),
            crop.x1 - crop.x0,
            crop.y1 - crop.y0,
            fill=False,
            edgecolor="white",
            linewidth=0.8,
            zorder=12,
        )
    )


def _render_clean_grid(
    selected_fields: pd.DataFrame,
    crops_by_image: Mapping[str, CropBounds],
    loaded_by_image: Mapping[str, SelectedPlaneData],
    *,
    windows: Mapping[str, tuple[float, float]],
    output_base: Path,
    dpi: int,
    sensitivity_label: str | None = None,
) -> tuple[Path, Path, Path]:
    plt = _pyplot()
    rows = selected_fields.copy()
    rows["_slide_order"] = rows["slide"].map(_slide_order)
    rows["_arm_order"] = rows["arm"].map(_arm_order)
    rows = rows.sort_values(
        ["_slide_order", "_arm_order", "image_key"], kind="mergesort"
    )
    figure, axes = plt.subplots(
        len(rows),
        4,
        figsize=(11.0, max(2.5, 2.45 * len(rows))),
        squeeze=False,
        facecolor="black",
    )
    for row_index, row in enumerate(rows.to_dict("records")):
        key = str(row["image_key"]).casefold()
        loaded = loaded_by_image[key]
        rgb = _display_rgb_set(loaded.planes, windows)
        titles = ("DAPI", "MIAT", "QKI", "DAPI + MIAT + QKI")
        for column, (role, title) in enumerate(
            zip(("dapi", "miat", "qki", "merge"), titles, strict=True)
        ):
            axis = axes[row_index, column]
            _show_rgb(axis, rgb[role], title)
            if role == "merge":
                _draw_crop_box(axis, crops_by_image[key])
                _draw_scale_bar(
                    axis,
                    rgb[role].shape[:2],
                    voxel_xy_nm=float(row["voxel_xy_nm"]),
                    target_um=20.0,
                )
            if column == 0:
                axis.set_ylabel(
                    format_clean_grid_row_label(
                        row["slide"], row["arm"], row["biological_set"]
                    ),
                    color="white",
                    fontsize=8,
                )
        axes[row_index, 0].text(
            0.01,
            0.02,
            (
                f'z={int(row["selected_z_1based"])} (1-based); '
                "single plane; no projection"
            ),
            color="white",
            fontsize=6,
            transform=axes[row_index, 0].transAxes,
            ha="left",
            va="bottom",
        )
    suffix = (
        f" — display sensitivity: {sensitivity_label}"
        if sensitivity_label
        else ""
    )
    figure.suptitle(
        "MIAT × QKI representative exact analyzed planes" + suffix,
        color="white",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    paths = _save_composite(figure, output_base, dpi=dpi)
    plt.close(figure)
    return paths


def _render_null_ladder(
    axis: Any,
    nucleus_spots: pd.DataFrame,
) -> None:
    ordered = nucleus_spots.assign(
        _uid=nucleus_spots["spot_uid"].astype(str)
    ).sort_values("_uid", kind="mergesort")
    axis.set_facecolor("white")
    if ordered.empty:
        axis.text(0.5, 0.5, "No detected MIAT spots", ha="center", va="center")
        axis.set_axis_off()
        return
    y_positions = np.arange(len(ordered))
    for y, row in zip(y_positions, ordered.to_dict("records"), strict=True):
        q05 = float(row["null_q05_raw"])
        q99 = float(row["null_q99_raw"])
        observed = float(row["qki_footprint_mean_raw"])
        usable = _explicit_bool(row["null_usable"])
        if usable and np.isfinite(q05) and np.isfinite(q99):
            axis.hlines(y, q05, q99, color="#7F7F7F", linewidth=1.0)
            for percentile, color, height in (
                (90, "#56B4E9", 0.19),
                (95, "#0072B2", 0.27),
                (99, "#000000", 0.34),
            ):
                value = float(row[f"null_q{percentile}_raw"])
                axis.vlines(
                    value,
                    y - height,
                    y + height,
                    color=color,
                    linewidth=0.8,
                )
            point_color = (
                POSITIVE_COLOR
                if _explicit_bool(row["qki_threshold_positive_q95"])
                else NEGATIVE_COLOR
            )
            axis.scatter(
                [observed], [y], color=point_color, s=11, zorder=5
            )
        elif np.isfinite(observed):
            axis.scatter(
                [observed],
                [y],
                color=UNUSABLE_COLOR,
                marker="x",
                s=12,
                linewidths=0.7,
            )
    axis.set_yticks(y_positions)
    axis.set_yticklabels(
        [str(uid) for uid in ordered["spot_uid"]], fontsize=4
    )
    axis.invert_yaxis()
    axis.set_xlabel("Raw QKI camera units", fontsize=6)
    axis.set_title(
        "Per-spot null ladder\nq05–q99, q90/q95/q99 ticks, observed point",
        fontsize=7,
    )
    axis.tick_params(axis="x", labelsize=5)


def _render_overlay_png(
    qki_crop: np.ndarray,
    specs: list[OverlaySpec],
    *,
    percentile: int,
    output_path: Path,
    voxel_xy_nm: float,
    dpi: int,
) -> None:
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(4.0, 4.0), facecolor="black")
    _show_gray_qki(
        axis,
        qki_crop,
        (
            f"per-spot rotation-null q{percentile}"
            + (" — primary" if percentile == 95 else "")
            + "\nraw QKI background; no global intensity cutoff"
        ),
    )
    _draw_overlay_specs(axis, specs, linewidth=1.0)
    _draw_scale_bar(
        axis,
        qki_crop.shape,
        voxel_xy_nm=voxel_xy_nm,
        target_um=5.0,
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.03,
        facecolor="black",
    )
    plt.close(figure)


def _render_walkthrough(
    loaded: SelectedPlaneData,
    field_spots: pd.DataFrame,
    footprint_pixels: pd.DataFrame,
    *,
    nucleus_id: int,
    selection_role: str,
    crop: CropBounds,
    slide: object,
    arm: object,
    biological_set: object,
    voxel_xy_nm: float,
    output_dir: Path,
    dpi: int,
) -> tuple[tuple[Path, Path, Path], tuple[Path, ...]]:
    plt = _pyplot()
    nucleus_spots = field_spots.loc[
        pd.to_numeric(
            field_spots["nucleus_id"], errors="coerce"
        ).eq(int(nucleus_id))
    ].copy()
    if nucleus_spots.empty:
        raise ValueError(
            f"selected nucleus {nucleus_id} has no spots for walkthrough"
        )
    relevant_uids = set(field_spots["spot_uid"].astype(str))
    field_pixels = footprint_pixels.loc[
        footprint_pixels["spot_uid"].astype(str).isin(relevant_uids)
    ]
    nucleus_uids = set(nucleus_spots["spot_uid"].astype(str))
    nucleus_pixels = field_pixels.loc[
        field_pixels["spot_uid"].astype(str).isin(nucleus_uids)
    ]
    full_rgb = _display_rgb_set(loaded.planes, PRIMARY_DISPLAY_WINDOWS)
    crop_slice = np.s_[crop.y0 : crop.y1, crop.x0 : crop.x1]
    crop_planes = {
        role: plane[crop_slice] for role, plane in loaded.planes.items()
    }
    crop_rgb = _display_rgb_set(crop_planes, PRIMARY_DISPLAY_WINDOWS)

    figure, axes = plt.subplots(
        2,
        5,
        figsize=(16.0, 7.0),
        facecolor="black",
        constrained_layout=True,
        gridspec_kw={"wspace": 0.12, "hspace": 0.2},
    )
    _show_rgb(axes[0, 0], full_rgb["dapi"], "DAPI + saved outlines")
    _draw_mask_outlines(
        axes[0, 0], loaded.nucleus_labels, loaded.nucleolus_labels
    )
    _show_rgb(axes[0, 1], full_rgb["miat"], "Raw MIAT")
    _show_rgb(
        axes[0, 2],
        full_rgb["miat"],
        "MIAT exact footprints + detected centers",
    )
    _draw_miat_footprints(
        axes[0, 2], field_spots, field_pixels, show_indices=False
    )
    _show_rgb(axes[0, 3], full_rgb["qki"], "Raw QKI")
    _show_rgb(axes[0, 4], full_rgb["merge"], "Three-channel merge + zoom")
    _draw_crop_box(axes[0, 4], crop)
    _draw_scale_bar(
        axes[0, 4],
        loaded.planes["miat"].shape,
        voxel_xy_nm=voxel_xy_nm,
        target_um=20.0,
    )

    _show_rgb(
        axes[1, 0],
        crop_rgb["miat"],
        "MIAT exact footprints + spot indices",
    )
    _draw_miat_footprints(
        axes[1, 0],
        nucleus_spots,
        nucleus_pixels,
        crop=crop,
        show_indices=True,
    )
    _draw_scale_bar(
        axes[1, 0],
        crop_planes["miat"].shape,
        voxel_xy_nm=voxel_xy_nm,
        target_um=5.0,
    )
    overlay_paths: list[Path] = []
    for column, percentile in enumerate((90, 95, 99), start=1):
        specs = build_call_overlay_specs(
            nucleus_spots,
            nucleus_pixels,
            percentile=percentile,
            crop=crop,
        )
        _show_gray_qki(
            axes[1, column],
            crop_planes["qki"],
            (
                f"per-spot rotation-null q{percentile}"
                + (" — PRIMARY" if percentile == 95 else "")
                + "\nraw QKI; no global cutoff"
            ),
        )
        _draw_overlay_specs(axes[1, column], specs, linewidth=1.0)
        if percentile == 95:
            _draw_scale_bar(
                axes[1, column],
                crop_planes["qki"].shape,
                voxel_xy_nm=voxel_xy_nm,
                target_um=5.0,
            )
        overlay_path = output_dir / f"q{percentile}_overlay.png"
        _render_overlay_png(
            crop_planes["qki"],
            specs,
            percentile=percentile,
            output_path=overlay_path,
            voxel_xy_nm=voxel_xy_nm,
            dpi=dpi,
        )
        overlay_paths.append(overlay_path)
    _render_null_ladder(axes[1, 4], nucleus_spots)
    axes[1, 4].set_facecolor("white")
    plane_title = (
        f"{slide} {arm} · {biological_set} · nucleus {nucleus_id} "
        f"({selection_role})\n"
        f"single analyzed plane z={loaded.selected_z_1based} (1-based); "
        "DAPI/MIAT/QKI same z; no projection"
    )
    figure.suptitle(plane_title, color="white", fontsize=10)
    paths = _save_composite(
        figure, output_dir / "threshold_walkthrough", dpi=dpi
    )
    plt.close(figure)
    return paths, tuple(overlay_paths)


def _validate_render_spots(spots: pd.DataFrame) -> None:
    required = {
        "image_key",
        "nucleus_id",
        "spot_uid",
        "selected_z_1based",
        "selected_z_0based",
        "miat_channel_index",
        "qki_channel_index",
        "dapi_channel_index",
        "quantitation_plane",
        "center_x_px",
        "center_y_px",
        "footprint_method",
        "null_usable",
        "qki_footprint_mean_raw",
        "null_q05_raw",
        "null_q90_raw",
        "null_q95_raw",
        "null_q99_raw",
        *(
            f"qki_threshold_positive_q{percentile}"
            for percentile in (90, 95, 99)
        ),
        *(
            f"population_label_q{percentile}"
            for percentile in (90, 95, 99)
        ),
    }
    _require_columns(spots, required, table="renderer spot table")
    for row in spots.to_dict("records"):
        uid = str(row["spot_uid"])
        z1 = int(row["selected_z_1based"])
        if int(row["selected_z_0based"]) != z1 - 1:
            raise ValueError(f"spot {uid!r} has mismatched selected z metadata")
        if str(row["quantitation_plane"]) != "exact_recorded_single_z":
            raise ValueError(f"spot {uid!r} is not from the exact recorded single z")
        usable = _explicit_bool(row["null_usable"])
        calls: dict[int, bool] = {}
        for percentile in (90, 95, 99):
            value = row[f"qki_threshold_positive_q{percentile}"]
            if usable:
                if pd.isna(value):
                    raise ValueError(
                        f"usable spot {uid!r} lacks q{percentile} call"
                    )
                calls[percentile] = _explicit_bool(value)
                expected_label = (
                    "threshold_positive"
                    if calls[percentile]
                    else "threshold_negative"
                )
                if str(row[f"population_label_q{percentile}"]) != expected_label:
                    raise ValueError(
                        f"spot {uid!r} has inconsistent q{percentile} label"
                    )
            else:
                if not pd.isna(value):
                    raise ValueError(
                        f"unusable spot {uid!r} has a q{percentile} call"
                    )
                label = str(row[f"population_label_q{percentile}"])
                if label in {"threshold_positive", "threshold_negative"}:
                    raise ValueError(
                        f"unusable spot {uid!r} has a classified q{percentile} label"
                    )
        if usable and (
            (calls[99] and not calls[95])
            or (calls[95] and not calls[90])
        ):
            raise ValueError(f"spot {uid!r} violates q99 subset q95 subset q90")


def _expected_channel_indices(field: Mapping[str, Any]) -> dict[str, int] | None:
    columns = {
        role: f"{role}_channel_index"
        for role in ("miat", "qki", "dapi")
    }
    if not all(column in field and not pd.isna(field[column]) for column in columns.values()):
        return None
    return {
        role: int(field[column]) for role, column in columns.items()
    }


def _threshold_sensitivity_rows(
    manifest: pd.DataFrame,
    spots: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for selected in manifest.to_dict("records"):
        group = spots.loc[
            spots["image_key"].eq(str(selected["image_key"]).casefold())
            & pd.to_numeric(
                spots["nucleus_id"], errors="coerce"
            ).eq(int(selected["nucleus_id"]))
        ]
        for percentile in (90, 95, 99):
            labels = group[f"population_label_q{percentile}"].astype(str)
            positive = int(labels.eq("threshold_positive").sum())
            negative = int(labels.eq("threshold_negative").sum())
            usable = positive + negative
            rows.append(
                {
                    "slide": selected["slide"],
                    "arm": selected["arm"],
                    "image_key": selected["image_key"],
                    "nucleus_uid": selected["nucleus_uid"],
                    "nucleus_id": int(selected["nucleus_id"]),
                    "selection_role": selected["selection_role"],
                    "percentile": percentile,
                    "is_primary": percentile == 95,
                    "n_positive": positive,
                    "n_negative": negative,
                    "n_unusable": int(len(group) - usable),
                    "association_fraction_among_usable": (
                        float(positive / usable) if usable else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def render_exact_footprint_package(
    selected_planes_h5: str | Path,
    corrected_spots: pd.DataFrame | str | Path,
    footprint_pixels: pd.DataFrame | str | Path,
    nuclei: pd.DataFrame | str | Path,
    image_audit: pd.DataFrame | str | Path,
    output_dir: str | Path,
    *,
    dpi: int = 600,
    renderer_version: str = "exact-footprint-figures-v1",
    render_miat_sensitivity: bool = False,
    render_dapi_prior_sensitivity: bool = False,
    source_hashes: Mapping[str, str] | None = None,
) -> FigurePackageOutputs:
    """Write the complete display-only representative exact-plane package.

    The output directory must not already exist.  This function reads only
    frozen 2D HDF5 planes and canonical tables; no image-analysis engine is
    imported or called.
    """

    h5_path = Path(selected_planes_h5)
    if not h5_path.is_file():
        raise FileNotFoundError(h5_path)
    spots, spots_hash, spots_source = _table_and_hash(corrected_spots)
    pixels, pixels_hash, pixels_source = _table_and_hash(footprint_pixels)
    nucleus_table, nuclei_hash, nuclei_source = _table_and_hash(nuclei)
    images, images_hash, images_source = _table_and_hash(image_audit)
    _validate_render_spots(spots)
    _require_columns(
        pixels,
        {"spot_uid", "pixel_index", "y_px", "x_px"},
        table="footprint pixel table",
    )
    if pixels.duplicated(["spot_uid", "pixel_index"]).any():
        raise ValueError(
            "footprint pixel table must be unique by spot_uid/pixel_index"
        )
    spots = spots.copy()
    pixels = pixels.copy()
    spots["image_key"] = spots["image_key"].astype(str).str.casefold()
    images["image_key"] = images["image_key"].astype(str).str.casefold()
    nucleus_table["image_key"] = (
        nucleus_table["image_key"].astype(str).str.casefold()
    )
    selection = select_representatives(images, spots, nucleus_table)
    selected_fields = selection.field_audit.loc[
        selection.field_audit["selected"]
    ].copy()
    if selected_fields["image_key"].nunique() != 4:
        raise ValueError(
            "publication package requires exactly four selected biological fields"
        )
    selected_cells = selected_fields[["slide", "arm"]].drop_duplicates()
    if len(selected_cells) != 4:
        raise ValueError(
            "publication package requires one field in each of four slide×arm cells"
        )
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(
            f"figure output directory already exists; use a new path: {output}"
        )
    output.mkdir(parents=True, exist_ok=False)

    hashes = {
        "selected_planes_h5_sha256": _sha256_file(h5_path),
        "corrected_spot_table_sha256": spots_hash,
        "footprint_pixels_sha256": pixels_hash,
        "nucleus_table_sha256": nuclei_hash,
        "image_audit_sha256": images_hash,
    }
    if source_hashes:
        hashes.update({str(key): str(value) for key, value in source_hashes.items()})
    sources = {
        "selected_planes_h5": str(h5_path.resolve()),
        "corrected_spot_table": spots_source,
        "footprint_pixels": pixels_source,
        "nucleus_table": nuclei_source,
        "image_audit": images_source,
    }
    manifest = selection.manifest.copy()
    nucleus_audit = selection.nucleus_audit.copy()
    crop_columns = (
        "crop_y0",
        "crop_y1",
        "crop_x0",
        "crop_x1",
        "crop_width_um",
        "crop_height_um",
        "crop_square",
        "crop_clamped",
    )
    for column in crop_columns[:6]:
        manifest[column] = np.nan
        nucleus_audit[column] = np.nan
    for column in crop_columns[6:]:
        manifest[column] = False
        nucleus_audit[column] = pd.Series(
            pd.array([pd.NA] * len(nucleus_audit), dtype="boolean"),
            index=nucleus_audit.index,
        )
    loaded_by_image: dict[str, SelectedPlaneData] = {}
    crops_by_image: dict[str, CropBounds] = {}
    walkthrough_paths: list[Path] = []
    overlay_product_paths: list[Path] = []
    sidecar_paths: list[Path] = []

    for field in selected_fields.to_dict("records"):
        key = str(field["image_key"]).casefold()
        loaded = load_selected_plane(
            h5_path,
            key,
            expected_z_1based=int(field["selected_z_1based"]),
            expected_channel_indices=_expected_channel_indices(field),
        )
        loaded_by_image[key] = loaded
        field_spots = spots.loc[spots["image_key"].eq(key)].copy()
        if field_spots.empty:
            raise ValueError(f"selected image {key!r} has no renderer spot rows")
        if not pd.to_numeric(
            field_spots["selected_z_1based"], errors="coerce"
        ).eq(loaded.selected_z_1based).all():
            raise ValueError(f"selected z mismatch inside spot rows for {key!r}")
        for role in ("miat", "qki", "dapi"):
            if not pd.to_numeric(
                field_spots[f"{role}_channel_index"], errors="coerce"
            ).eq(loaded.channel_indices[role]).all():
                raise ValueError(
                    f"channel index mismatch inside spot rows for {key!r}: {role}"
                )
        field_pixels = pixels.loc[
            pixels["spot_uid"].astype(str).isin(
                set(field_spots["spot_uid"].astype(str))
            )
        ].copy()
        if set(field_spots["spot_uid"].astype(str)) - set(
            field_pixels["spot_uid"].astype(str)
        ):
            raise ValueError(
                f"selected image {key!r} has spots without exact footprint pixels"
            )
        field_root = output / "fields" / _slug(key)
        full_root = field_root / "full"
        full_root.mkdir(parents=True, exist_ok=False)
        for role in ("dapi", "miat", "qki"):
            _write_tiff(
                full_root / f"{role}_raw_uint16.tif",
                loaded.planes[role],
            )
        _write_tiff(
            full_root / "nucleus_labels_raw_integer.tif",
            loaded.nucleus_labels,
        )
        _write_tiff(
            full_root / "nucleolus_labels_raw_integer.tif",
            loaded.nucleolus_labels,
        )
        full_rgb = _display_rgb_set(
            loaded.planes, PRIMARY_DISPLAY_WINDOWS
        )
        for role, array in full_rgb.items():
            _save_rgb_png(
                array, full_root / f"{role}_fixed_display.png"
            )
        full_sidecar = full_root / "provenance.json"
        _write_json(
            full_sidecar,
            {
                "display_only": True,
                "renderer_version": renderer_version,
                "image_key": key,
                "selected_z_1based": loaded.selected_z_1based,
                "selected_z_0based": loaded.selected_z_0based,
                "channel_indices": loaded.channel_indices,
                "display_windows": PRIMARY_DISPLAY_WINDOWS,
                "quantitation": {
                    "single_plane": True,
                    "same_z_all_channels": True,
                    "projection": "none",
                    "rendered_images_used_for_quantitation": False,
                },
                "voxel_xy_nm": float(field["voxel_xy_nm"]),
                "spot_uids": sorted(field_spots["spot_uid"].astype(str)),
                "source_paths": sources,
                "source_hashes": hashes,
            },
        )
        sidecar_paths.append(full_sidecar)

        selected_nuclei = manifest.loc[manifest["image_key"].eq(key)]
        for manifest_index, selected in selected_nuclei.iterrows():
            nucleus_id = int(selected["nucleus_id"])
            role = str(selected["selection_role"])
            crop = nucleus_square_crop(
                loaded.nucleus_labels,
                nucleus_id,
                float(field["voxel_xy_nm"]),
            )
            crops_by_image.setdefault(key, crop)
            crop_values: dict[str, object] = {
                "crop_y0": crop.y0,
                "crop_y1": crop.y1,
                "crop_x0": crop.x0,
                "crop_x1": crop.x1,
                "crop_width_um": crop.width_um,
                "crop_height_um": crop.height_um,
                "crop_square": crop.is_square,
                "crop_clamped": crop.was_clamped,
            }
            for column, value in crop_values.items():
                manifest.loc[manifest_index, column] = value
            nucleus_match = (
                nucleus_audit["image_key"].eq(key)
                & pd.to_numeric(
                    nucleus_audit["nucleus_id"], errors="coerce"
                ).eq(nucleus_id)
                & nucleus_audit["selected"].map(_explicit_bool)
            )
            for column, value in crop_values.items():
                nucleus_audit.loc[nucleus_match, column] = value
            crop_root = field_root / f"nucleus_{nucleus_id}_{_slug(role)}"
            crop_root.mkdir(parents=True, exist_ok=False)
            crop_slice = np.s_[crop.y0 : crop.y1, crop.x0 : crop.x1]
            crop_planes = {
                channel: plane[crop_slice]
                for channel, plane in loaded.planes.items()
            }
            for channel in ("dapi", "miat", "qki"):
                _write_tiff(
                    crop_root / f"{channel}_raw_uint16.tif",
                    crop_planes[channel],
                )
            _write_tiff(
                crop_root / "nucleus_labels_raw_integer.tif",
                loaded.nucleus_labels[crop_slice],
            )
            _write_tiff(
                crop_root / "nucleolus_labels_raw_integer.tif",
                loaded.nucleolus_labels[crop_slice],
            )
            crop_rgb = _display_rgb_set(
                crop_planes, PRIMARY_DISPLAY_WINDOWS
            )
            for channel, array in crop_rgb.items():
                _save_rgb_png(
                    array, crop_root / f"{channel}_fixed_display.png"
                )
            rendered, overlay_paths = _render_walkthrough(
                loaded,
                field_spots,
                field_pixels,
                nucleus_id=nucleus_id,
                selection_role=role,
                crop=crop,
                slide=field["slide"],
                arm=field["arm"],
                biological_set=field.get("biological_set", ""),
                voxel_xy_nm=float(field["voxel_xy_nm"]),
                output_dir=crop_root,
                dpi=dpi,
            )
            walkthrough_paths.extend(rendered)
            overlay_product_paths.extend(overlay_paths)
            nucleus_spots = field_spots.loc[
                pd.to_numeric(
                    field_spots["nucleus_id"], errors="coerce"
                ).eq(nucleus_id)
            ]
            sidecar = crop_root / "provenance.json"
            _write_json(
                sidecar,
                {
                    "display_only": True,
                    "renderer_version": renderer_version,
                    "image_key": key,
                    "slide": str(field["slide"]),
                    "arm": str(field["arm"]),
                    "biological_set": str(
                        field.get("biological_set", "")
                    ),
                    "nucleus_uid": str(selected["nucleus_uid"]),
                    "nucleus_id": nucleus_id,
                    "selection_role": role,
                    "selected_z_1based": loaded.selected_z_1based,
                    "selected_z_0based": loaded.selected_z_0based,
                    "channel_indices": loaded.channel_indices,
                    "crop": crop_values,
                    "voxel_xy_nm": float(field["voxel_xy_nm"]),
                    "display_windows": PRIMARY_DISPLAY_WINDOWS,
                    "scale_bars": {
                        "full": _scale_bar_record(
                            float(field["voxel_xy_nm"]), 20.0
                        ),
                        "zoom": _scale_bar_record(
                            float(field["voxel_xy_nm"]), 5.0
                        ),
                    },
                    "spot_uids": sorted(
                        nucleus_spots["spot_uid"].astype(str)
                    ),
                    "thresholds": {
                        "primary": 95,
                        "sensitivity": [90, 99],
                        "operator": "observed_qki_mean > own_linear_quantile",
                        "global_qki_intensity_cutoff": None,
                    },
                    "quantitation": {
                        "single_plane": True,
                        "same_z_all_channels": True,
                        "projection": "none",
                        "footprint_region": "exact_detected_miat_pixels",
                        "rendered_images_used_for_quantitation": False,
                    },
                    "source_paths": sources,
                    "source_hashes": hashes,
                },
            )
            sidecar_paths.append(sidecar)

    for column, value in hashes.items():
        manifest[column] = value
    field_audit = selection.field_audit.copy()
    field_audit.to_csv(
        output / "representative_field_selection_audit.csv", index=False
    )
    nucleus_audit.to_csv(
        output / "representative_nucleus_selection_audit.csv", index=False
    )
    manifest.to_csv(
        output / "representative_selection_manifest.csv", index=False
    )
    threshold_summary = _threshold_sensitivity_rows(manifest, spots)
    threshold_summary_path = output / "threshold_sensitivity_summary.csv"
    threshold_summary.to_csv(threshold_summary_path, index=False)
    selection = RepresentativeSelection(
        field_audit=field_audit,
        nucleus_audit=nucleus_audit,
        manifest=manifest,
    )

    clean_paths: list[Path] = list(
        _render_clean_grid(
            selected_fields,
            crops_by_image,
            loaded_by_image,
            windows=PRIMARY_DISPLAY_WINDOWS,
            output_base=output / "clean_micrograph_grid",
            dpi=dpi,
        )
    )
    if render_miat_sensitivity:
        miat_windows = dict(PRIMARY_DISPLAY_WINDOWS)
        miat_windows["miat"] = DISPLAY_SENSITIVITY_WINDOWS["miat_high"]
        clean_paths.extend(
            _render_clean_grid(
                selected_fields,
                crops_by_image,
                loaded_by_image,
                windows=miat_windows,
                output_base=output
                / "clean_micrograph_grid__miat_700_5000_display_sensitivity",
                dpi=dpi,
                sensitivity_label="MIAT 700–5000",
            )
        )
    if render_dapi_prior_sensitivity:
        dapi_windows = dict(PRIMARY_DISPLAY_WINDOWS)
        dapi_windows["dapi"] = DISPLAY_SENSITIVITY_WINDOWS[
            "dapi_prior_qa"
        ]
        clean_paths.extend(
            _render_clean_grid(
                selected_fields,
                crops_by_image,
                loaded_by_image,
                windows=dapi_windows,
                output_base=output
                / "clean_micrograph_grid__dapi_334_5500_prior_qa_sensitivity",
                dpi=dpi,
                sensitivity_label="prior-QA DAPI 334–5500",
            )
        )
    clean_sidecar = output / "clean_micrograph_grid.provenance.json"
    _write_json(
        clean_sidecar,
        {
            "display_only": True,
            "renderer_version": renderer_version,
            "primary_display_windows": PRIMARY_DISPLAY_WINDOWS,
            "primary_dapi_window_reason": (
                "global 334–8000 avoids clipping in the visually audited "
                "slide-2 representative fields"
            ),
            "optional_display_sensitivities": {
                "miat_700_5000_rendered": bool(render_miat_sensitivity),
                "dapi_334_5500_prior_qa_rendered": bool(
                    render_dapi_prior_sensitivity
                ),
            },
            "quantitation": {
                "single_plane": True,
                "same_z_all_channels": True,
                "projection": "none",
                "rendered_images_used_for_quantitation": False,
            },
            "source_paths": sources,
            "source_hashes": hashes,
            "selected_rows": manifest.to_dict("records"),
        },
    )
    sidecar_paths.append(clean_sidecar)
    return FigurePackageOutputs(
        output_dir=output,
        selection=selection,
        clean_grid_paths=tuple(clean_paths),
        walkthrough_paths=tuple(walkthrough_paths),
        overlay_paths=tuple(overlay_product_paths),
        sidecar_paths=tuple(sidecar_paths),
        threshold_summary_path=threshold_summary_path,
    )


def _nucleus_bbox(labels: np.ndarray, nucleus_id: int) -> tuple[int, int, int, int]:
    yy, xx = np.nonzero(np.asarray(labels) == int(nucleus_id))
    if not yy.size:
        raise ValueError(
            f"anomaly nucleus_id {nucleus_id} is absent from the saved mask"
        )
    return (
        int(yy.min()),
        int(yy.max()) + 1,
        int(xx.min()),
        int(xx.max()) + 1,
    )


def _draw_nucleus_callouts(
    axis: Any,
    labels: np.ndarray,
    nucleus_ids: tuple[int, ...],
) -> None:
    from matplotlib.patches import Rectangle

    for nucleus_id in nucleus_ids:
        y0, y1, x0, x1 = _nucleus_bbox(labels, nucleus_id)
        axis.add_patch(
            Rectangle(
                (x0 - 0.5, y0 - 0.5),
                x1 - x0,
                y1 - y0,
                fill=False,
                edgecolor="#E69F00",
                linewidth=1.0,
                zorder=12,
            )
        )
        axis.text(
            x0,
            max(0, y0 - 1),
            f"nucleus {nucleus_id}",
            color="#E69F00",
            fontsize=6,
            zorder=13,
        )


def render_control_diagnostics(
    selected_planes_h5: str | Path,
    corrected_spots: pd.DataFrame | str | Path,
    footprint_pixels: pd.DataFrame | str | Path,
    image_audit: pd.DataFrame | str | Path,
    output_dir: str | Path,
    *,
    hooks: ControlPanelHooks,
    dpi: int = 600,
    renderer_version: str = "exact-footprint-figures-v1",
) -> ControlDiagnosticOutputs:
    """Render QC-only omission controls, a zero example, and anomaly callouts."""

    h5_path = Path(selected_planes_h5)
    spots, spots_hash, spots_source = _table_and_hash(corrected_spots)
    pixels, pixels_hash, pixels_source = _table_and_hash(footprint_pixels)
    images, images_hash, images_source = _table_and_hash(image_audit)
    _require_columns(
        images,
        {
            "image_key",
            "image",
            "slide",
            "is_control",
            "secondary_only",
            "selected_z_1based",
            "voxel_xy_nm",
            "mask_qc_pass",
            "image_qc_status",
            "qc_flags",
        },
        table="control image audit",
    )
    _require_columns(
        spots,
        {
            "image_key",
            "spot_uid",
            "nucleus_id",
            "center_x_px",
            "center_y_px",
            "null_usable",
            "qki_threshold_positive_q95",
            "footprint_method",
        },
        table="control spot table",
    )
    _require_columns(
        pixels,
        {"spot_uid", "pixel_index", "y_px", "x_px"},
        table="control footprint pixels",
    )
    images = images.copy()
    spots = spots.copy()
    images["image_key"] = images["image_key"].astype(str).str.casefold()
    spots["image_key"] = spots["image_key"].astype(str).str.casefold()
    controls = images.loc[
        images["is_control"].map(_explicit_bool)
        & images["secondary_only"].map(_explicit_bool)
    ].copy()
    if controls.empty:
        raise ValueError("control image audit contains no full-omission controls")
    zero_key = str(hooks.zero_control_image_key).casefold()
    anomaly_key = str(hooks.anomaly_image_key).casefold()
    known = set(controls["image_key"])
    if zero_key not in known or anomaly_key not in known:
        raise ValueError(
            "zero/anomaly control hooks must name rows in the control audit"
        )
    if len(spots.loc[spots["image_key"].eq(zero_key)]) != 0:
        raise ValueError("configured zero-control example does not have zero MIAT calls")
    zero_row = controls.loc[controls["image_key"].eq(zero_key)].iloc[0]
    zero_flags = (
        ""
        if pd.isna(zero_row["qc_flags"])
        else str(zero_row["qc_flags"]).strip().casefold()
    )
    if "zero_spot" not in zero_flags:
        raise ValueError(
            "configured zero-control example lacks the recorded zero_spot QC flag"
        )
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(
            f"control output directory already exists; use a new path: {output}"
        )
    output.mkdir(parents=True, exist_ok=False)

    loaded: dict[str, SelectedPlaneData] = {}
    ledger_rows: list[dict[str, object]] = []
    for row in controls.sort_values("image_key", kind="mergesort").to_dict(
        "records"
    ):
        key = str(row["image_key"])
        plane = load_selected_plane(
            h5_path,
            key,
            expected_z_1based=int(row["selected_z_1based"]),
            expected_channel_indices=_expected_channel_indices(row),
        )
        loaded[key] = plane
        count = int(spots["image_key"].eq(key).sum())
        recorded_flags = (
            ""
            if pd.isna(row["qc_flags"])
            else str(row["qc_flags"]).strip()
        )
        has_recorded_zero_spot = "zero_spot" in recorded_flags.casefold()
        if count > 0 and has_recorded_zero_spot:
            raise ValueError(
                f"control {key!r} has MIAT calls despite recorded zero_spot QC"
            )
        if count == 0 and has_recorded_zero_spot:
            control_signal_qc_status = "expected_negative_zero_spot"
        elif count == 0:
            control_signal_qc_status = "zero_spot_not_recorded"
        elif key == anomaly_key:
            control_signal_qc_status = "artifact_bearing_miat_calls"
        elif _qc_pass(row["image_qc_status"]):
            control_signal_qc_status = "recorded_signal_qc_pass"
        else:
            control_signal_qc_status = "recorded_signal_qc_failed"
        if key == zero_key:
            designation = "clean expected-negative zero-call example"
        elif key == anomaly_key:
            designation = "artifact-bearing retained diagnostic"
        else:
            designation = str(
                row.get("control_designation", "retained diagnostic")
            )
        ledger_rows.append(
            {
                "image_key": key,
                "image": row["image"],
                "slide": row["slide"],
                "selected_z_1based": plane.selected_z_1based,
                "voxel_xy_nm": float(row["voxel_xy_nm"]),
                "miat_call_count": count,
                "qki_background_median_raw": float(
                    np.median(plane.planes["qki"])
                ),
                "qki_background_p95_raw": float(
                    np.percentile(plane.planes["qki"], 95)
                ),
                "mask_qc_status": (
                    "pass" if _explicit_bool(row["mask_qc_pass"]) else "failed"
                ),
                "plane_integrity_status": "pass",
                "recorded_image_qc_status": str(row["image_qc_status"]),
                "recorded_qc_flags": recorded_flags,
                "control_signal_qc_status": control_signal_qc_status,
                "designation": designation,
                "combined_primary_omission": True,
                "biological_inference_eligible": False,
            }
        )
    ledger = pd.DataFrame(ledger_rows)
    ledger_path = output / "omission_control_ledger.csv"
    ledger.to_csv(ledger_path, index=False)

    plt = _pyplot()
    n_columns = min(4, len(ledger))
    n_rows = int(np.ceil(len(ledger) / n_columns))
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(3.3 * n_columns, 3.2 * n_rows),
        squeeze=False,
        facecolor="black",
    )
    for axis in axes.ravel():
        axis.set_axis_off()
        axis.set_facecolor("black")
    for axis, row in zip(
        axes.ravel(), ledger.to_dict("records"), strict=False
    ):
        key = str(row["image_key"])
        rgb = _display_rgb_set(
            loaded[key].planes, PRIMARY_DISPLAY_WINDOWS
        )
        _show_rgb(
            axis,
            rgb["merge"],
            (
                f'{key}\nz={int(row["selected_z_1based"])}; '
                f'MIAT calls={int(row["miat_call_count"])}\n'
                f'{row["designation"]}'
            ),
        )
    figure.suptitle(
        "Slide-2 full-omission controls — combined MIAT probe + QKI primary",
        color="white",
        fontsize=10,
    )
    figure.tight_layout()
    ledger_figure_paths = _save_composite(
        figure, output / "omission_control_thumbnail_ledger", dpi=dpi
    )
    plt.close(figure)

    zero = loaded[zero_key]
    zero_rgb = _display_rgb_set(
        zero.planes, PRIMARY_DISPLAY_WINDOWS
    )
    figure, axes = plt.subplots(
        1, 4, figsize=(12.0, 3.2), facecolor="black"
    )
    for axis, role, title in zip(
        axes,
        ("dapi", "miat", "qki", "merge"),
        ("DAPI", "MIAT", "QKI", "Merge"),
        strict=True,
    ):
        _show_rgb(axis, zero_rgb[role], title)
    _draw_scale_bar(
        axes[-1],
        zero.planes["miat"].shape,
        voxel_xy_nm=float(
            ledger.set_index("image_key").loc[zero_key, "voxel_xy_nm"]
        ),
        target_um=20.0,
    )
    figure.suptitle(
        (
            f"{zero_key}: expected-negative zero-call omission control\n"
            f"single analyzed plane z={zero.selected_z_1based} (1-based); "
            "same z; no projection; QC only"
        ),
        color="white",
        fontsize=9,
    )
    figure.tight_layout()
    zero_paths = _save_composite(
        figure, output / "zero_control_example", dpi=dpi
    )
    plt.close(figure)

    anomaly = loaded[anomaly_key]
    anomaly_spots = spots.loc[
        spots["image_key"].eq(anomaly_key)
    ].copy()
    anomaly_uids = set(anomaly_spots["spot_uid"].astype(str))
    anomaly_pixels = pixels.loc[
        pixels["spot_uid"].astype(str).isin(anomaly_uids)
    ]
    if set(anomaly_spots["spot_uid"].astype(str)) - set(
        anomaly_pixels["spot_uid"].astype(str)
    ):
        raise ValueError("anomaly control has spots without exact footprint pixels")
    anomaly_rgb = _display_rgb_set(
        anomaly.planes, PRIMARY_DISPLAY_WINDOWS
    )
    figure, axes = plt.subplots(
        1, 4, figsize=(12.0, 3.2), facecolor="black"
    )
    _show_rgb(axes[0], anomaly_rgb["dapi"], "DAPI + anomaly nuclei")
    _draw_mask_outlines(
        axes[0], anomaly.nucleus_labels, anomaly.nucleolus_labels
    )
    _draw_nucleus_callouts(
        axes[0], anomaly.nucleus_labels, hooks.anomaly_nucleus_ids
    )
    _show_rgb(
        axes[1], anomaly_rgb["miat"], "MIAT + exact footprint boundaries"
    )
    _draw_miat_footprints(
        axes[1], anomaly_spots, anomaly_pixels
    )
    _draw_nucleus_callouts(
        axes[1], anomaly.nucleus_labels, hooks.anomaly_nucleus_ids
    )
    _show_gray_qki(
        axes[2],
        anomaly.planes["qki"],
        "QKI + q95 calls (diagnostic only)",
    )
    anomaly_specs = build_call_overlay_specs(
        anomaly_spots, anomaly_pixels, percentile=95
    )
    _draw_overlay_specs(axes[2], anomaly_specs)
    _show_rgb(axes[3], anomaly_rgb["merge"], "Merge")
    _draw_nucleus_callouts(
        axes[3], anomaly.nucleus_labels, hooks.anomaly_nucleus_ids
    )
    figure.suptitle(
        (
            f"{anomaly_key}: artifact-bearing omission control retained diagnostically\n"
            f"single analyzed plane z={anomaly.selected_z_1based} (1-based); "
            "excluded from biological selection and inference"
        ),
        color="white",
        fontsize=9,
    )
    figure.tight_layout()
    anomaly_paths = _save_composite(
        figure, output / "artifact_anomaly_panel", dpi=dpi
    )
    plt.close(figure)

    sidecar_path = output / "control_diagnostics.provenance.json"
    _write_json(
        sidecar_path,
        {
            "display_only": True,
            "renderer_version": renderer_version,
            "control_scope": (
                "combined MIAT-probe and QKI-primary omission"
            ),
            "scope_limits": [
                "does not isolate MIAT-probe specificity",
                "does not isolate QKI-primary specificity",
                "does not establish slide-1 background",
            ],
            "excluded_from_biological_selection_and_inference": True,
            "zero_control_image_key": zero_key,
            "anomaly_image_key": anomaly_key,
            "anomaly_nucleus_ids": list(hooks.anomaly_nucleus_ids),
            "display_windows": PRIMARY_DISPLAY_WINDOWS,
            "quantitation": {
                "single_plane": True,
                "same_z_all_channels": True,
                "projection": "none",
            },
            "source_paths": {
                "selected_planes_h5": str(h5_path.resolve()),
                "corrected_spot_table": spots_source,
                "footprint_pixels": pixels_source,
                "image_audit": images_source,
            },
            "source_hashes": {
                "selected_planes_h5_sha256": _sha256_file(h5_path),
                "corrected_spot_table_sha256": spots_hash,
                "footprint_pixels_sha256": pixels_hash,
                "image_audit_sha256": images_hash,
            },
        },
    )
    return ControlDiagnosticOutputs(
        output_dir=output,
        ledger_path=ledger_path,
        ledger_figure_paths=ledger_figure_paths,
        zero_example_paths=zero_paths,
        anomaly_paths=anomaly_paths,
        sidecar_path=sidecar_path,
    )
