"""Saved exact-footprint single-plane adapter for QKI association metrics."""
from __future__ import annotations

import shlex
from pathlib import Path

import numpy as np
import pandas as pd

from .exact_footprint_figures import load_selected_plane
from .exact_footprint_postrun import validate_quantitation_invariants
from .footprint_null import MiatFootprint, full_footprint_is_valid
from . import repro

REQUIRED_FILES = (
    "spot_exact_footprint_metrics.csv.gz",
    "nucleus_exact_footprint_metrics.csv",
    "footprint_pixels.csv.gz",
    "selected_planes_and_masks.h5",
    "image_manifest.csv",
)


def _require(frame, names, description):
    missing = sorted(set(names) - set(frame.columns))
    if missing:
        raise ValueError(f"{description} missing columns: {', '.join(missing)}")


def _integer(values, name):
    numbers = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(numbers).all() or not np.equal(numbers, np.floor(numbers)).all():
        raise ValueError(f"{name} must contain finite integers")
    return numbers.astype(np.int64)


def _truth(value):
    token = str(value).strip().casefold()
    if token in {"true", "1", "1.0"}:
        return True
    if token in {"false", "0", "0.0"}:
        return False
    raise ValueError(f"null_candidate must be a recorded boolean, found {value!r}")


def _restore_footprints(rows, pixels, data, eligible):
    footprints, identifiers = [], {}
    for index, row in enumerate(rows.itertuples(index=False)):
        if not _truth(row.null_candidate):
            continue
        block = pixels.loc[pixels.spot_uid == row.spot_uid].sort_values("pixel_index")
        if len(block) != int(row.footprint_area_px) or block.empty:
            raise ValueError(f"missing or inconsistent retained footprint pixels: {row.spot_uid}")
        y = _integer(block.y_px, "y_px")
        x = _integer(block.x_px, "x_px")
        dy = _integer(block.dy_px, "dy_px")
        dx = _integer(block.dx_px, "dx_px")
        cy, cx = int(row.center_y_px), int(row.center_x_px)
        if not np.array_equal(y - cy, dy) or not np.array_equal(x - cx, dx):
            raise ValueError(f"retained footprint offsets disagree with center: {row.spot_uid}")
        coords = np.column_stack((y, x))
        region = eligible & (data.nucleus_labels == int(row.nucleus_id))
        if len(np.unique(coords, axis=0)) != len(coords) or not full_footprint_is_valid(coords, region):
            raise ValueError(f"retained eligible footprint is outside parent nucleoplasm: {row.spot_uid}")
        if not (0 <= cy < region.shape[0] and 0 <= cx < region.shape[1]) or not region[cy, cx]:
            raise ValueError(f"retained footprint center is outside parent nucleoplasm: {row.spot_uid}")
        for role in ("miat", "qki"):
            if not np.array_equal(block[f"{role}_raw"].to_numpy(), data.planes[role][y, x]):
                raise ValueError(f"retained {role} pixels disagree with selected plane: {row.spot_uid}")
        identifiers[index] = row.spot_id
        fallback = getattr(row, "footprint_fallback_reason", "")
        footprints.append(MiatFootprint(
            spot_index=index, center_y_px=cy, center_x_px=cx,
            y_px=y, x_px=x, dy_px=dy, dx_px=dx,
            method=str(row.footprint_method), fallback_reason=str(fallback) or None,
            full_mask_valid=True, invalid_reason=None,
            observed_miat_raw=float(data.planes["miat"][y, x].mean()),
            observed_qki_raw=float(data.planes["qki"][y, x].mean()),
        ))
    return footprints, identifiers


def run_qki_association(run_dir, out, *, miat_min, qki_min,
                        sensitivity=(0.8, 1.0, 1.25), n_null=200, seed=0):
    from .qki_association import association_tables, COLUMN_DEFINITIONS

    levels = tuple(float(value) for value in sensitivity)
    if not levels or any(not np.isfinite(v) or v <= 0 for v in (miat_min, qki_min, *levels)):
        raise ValueError("MIAT/QKI thresholds and sensitivity multipliers must be finite and > 0")
    if len(set(levels)) != len(levels):
        raise ValueError("sensitivity multipliers must be unique")
    if isinstance(n_null, bool) or not isinstance(n_null, (int, np.integer)) or n_null <= 0:
        raise ValueError("n_null must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not np.isfinite(np.asarray(levels) * miat_min).all() or not np.isfinite(np.asarray(levels) * qki_min).all():
        raise ValueError("scaled thresholds must be finite")
    source, destination = Path(run_dir).resolve(), Path(out).resolve()
    missing = [name for name in REQUIRED_FILES if not (source / name).is_file()]
    if missing:
        raise ValueError(f"run directory missing retained exact-footprint files: {', '.join(missing)}")
    if destination == source or source in destination.parents or destination in source.parents:
        raise ValueError("--out must be outside the source run directory and its ancestors")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError("--out must be a new or empty directory; existing results are never overwritten")
    spots = pd.read_csv(source / REQUIRED_FILES[0], keep_default_na=False, dtype={"spot_id": str, "spot_uid": str})
    nuclei = pd.read_csv(source / REQUIRED_FILES[1], keep_default_na=False)
    pixels = pd.read_csv(source / REQUIRED_FILES[2], keep_default_na=False, dtype={"spot_uid": str})
    manifest = pd.read_csv(source / REQUIRED_FILES[4], keep_default_na=False,
                           dtype={"condition": str, "well": str})
    _require(spots, ["image_key", "spot_id", "spot_uid", "nucleus_id", "center_y_px", "center_x_px",
        "footprint_area_px", "footprint_method", "null_candidate", "selected_z_1based",
        "selected_z_0based", "quantitation_plane", "miat_channel_index", "qki_channel_index", "dapi_channel_index"], "spot table")
    _require(nuclei, ["image_key", "nucleus_id", "z_mode", "z_range", "n_z_slices"], "nucleus table")
    _require(pixels, ["spot_uid", "pixel_index", "y_px", "x_px", "dy_px", "dx_px", "miat_raw", "qki_raw"], "footprint pixel table")
    _require(manifest, ["image_key", "voxel_xy_nm", "selected_z_1based", "miat_channel_index", "qki_channel_index", "dapi_channel_index"], "image manifest")
    for frame in (spots, nuclei, manifest):
        frame["image_key"] = frame.image_key.astype(str).str.strip().str.casefold()
        if frame.image_key.eq("").any():
            raise ValueError("image_key cannot be empty")
    if manifest.empty or manifest.image_key.duplicated().any():
        raise ValueError("image manifest must contain unique image keys")
    if spots.spot_uid.duplicated().any() or spots.duplicated(["image_key", "spot_id"]).any():
        raise ValueError("spot table contains duplicate source identifiers")
    if nuclei.duplicated(["image_key", "nucleus_id"]).any():
        raise ValueError("nucleus table contains duplicate source identifiers")
    for frame in (spots, nuclei):
        if not set(frame.image_key).issubset(set(manifest.image_key)):
            raise ValueError("table contains image keys absent from image manifest")
        frame["nucleus_id"] = _integer(frame.nucleus_id, "nucleus_id")
    for column in ("center_y_px", "center_x_px", "footprint_area_px"):
        spots[column] = _integer(spots[column], column)
    if not set(pixels.spot_uid).issubset(set(spots.spot_uid)):
        raise ValueError("pixel table contains spot identifiers absent from spot table")
    all_nuclei, all_spots = [], []
    for record in manifest.sort_values("image_key").itertuples(index=False):
        key = record.image_key
        image_spots = spots.loc[spots.image_key == key].sort_values("spot_id")
        image_nuclei = nuclei.loc[nuclei.image_key == key]
        try:
            scale = float(record.voxel_xy_nm) / 1000.0
        except (ValueError, TypeError) as exc:
            raise ValueError(f"image manifest voxel_xy_nm must be finite and > 0: {key}") from exc
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"image manifest voxel_xy_nm must be finite and > 0: {key}")
        metadata_values = _integer(pd.Series([record.selected_z_1based, record.miat_channel_index,
            record.qki_channel_index, record.dapi_channel_index]), "selected plane metadata")
        channels = dict(zip(("miat", "qki", "dapi"), metadata_values[1:]))
        z = int(metadata_values[0])
        if z < 1 or any(value < 0 for value in channels.values()) or len(set(channels.values())) != 3:
            raise ValueError("selected plane metadata must specify positive z and three distinct channel indices")
        if not image_spots.empty:
            validate_quantitation_invariants(image_spots, image_nuclei, expected_channel_indices=channels)
        data = load_selected_plane(source / REQUIRED_FILES[3], key,
            expected_z_1based=z, expected_channel_indices=channels)
        if set(image_nuclei.nucleus_id) != set(np.unique(data.nucleus_labels[data.nucleus_labels > 0])):
            raise ValueError(f"nucleus roster disagrees with selected label mask: {key}")
        if not image_nuclei.z_range.astype(str).str.fullmatch(rf"\s*{z}\s*-\s*{z}\s*").all():
            raise ValueError(f"nucleus singleton z_range disagrees with selected plane: {key}")
        z_modes = image_nuclei.z_mode.astype(str).str.strip()
        if z_modes.eq("").any() or z_modes.str.contains(r"mip|projection|maximum.intensity|z.?stack|three.?dimensional|3d", case=False).any():
            raise ValueError(f"nucleus z_mode does not prove single-plane quantitation: {key}")
        if (_integer(image_nuclei.n_z_slices, "n_z_slices") < z).any():
            raise ValueError(f"nucleus selected z exceeds recorded slice count: {key}")
        if "voxel_xy_nm" in image_spots and not np.allclose(pd.to_numeric(image_spots.voxel_xy_nm), scale * 1000, rtol=0, atol=1e-6):
            raise ValueError(f"spot and manifest voxel_xy_nm disagree: {key}")
        eligible = (data.nucleus_labels > 0) & (data.nucleolus_labels != data.nucleus_labels)
        footprints, identifiers = _restore_footprints(image_spots, pixels, data, eligible)
        nuc_table, spot_table = association_tables(data.planes["miat"], data.planes["qki"],
            data.nucleus_labels, footprints, pixel_size_um=scale, miat_min=miat_min,
            qki_min=qki_min, sensitivity=levels, n_null=n_null, seed=seed,
            eligible_mask=eligible, image=key, condition=str(getattr(record, "condition", "")),
            well=str(getattr(record, "well", "")))
        spot_table["spot_id"] = spot_table.spot_id.map(identifiers)
        all_nuclei.append(nuc_table)
        all_spots.append(spot_table)
    nucleus_table = pd.concat(all_nuclei, ignore_index=True)
    spot_table = pd.concat(all_spots, ignore_index=True)
    destination.mkdir(parents=True, exist_ok=True)
    nucleus_table.to_csv(destination / "qki_association_per_nucleus.csv", index=False, lineterminator="\n")
    spot_table.to_csv(destination / "qki_association_per_spot.csv", index=False, lineterminator="\n")
    columns = list(dict.fromkeys([*nucleus_table.columns, *spot_table.columns]))
    lines = ["# Single-plane QKI association columns", "",
        "- Eligible spots retain source null_candidate eligibility and exact saved pixels; miat_min affects MIAT area only.",
        "- Missing source condition/well values are empty strings; identifiers are never inferred.",
        "- Areas are native-pixel area occupancy fractions. Every metric is single-plane.",
        "- Raw `frac_qki_area_on_miat_footprints` is UNCORRECTED for MIAT coverage and rises with coverage by chance; the coverage-corrected area statistic is `obs_minus_null_frac_miat_footprint_area_qki_pos`.",
        "- An undefined denominator gives NaN plus its reason (N0, Q0, U0, or R0 for eligible-area denominators); a defined denominator with zero numerator gives 0.0 plus an empty reason. Null summaries additionally require supported placements; the plus-one empirical tail fraction is 1.0 when all null draws tie an observed zero.",
        "", "## What the placement null does and does not control", "",
        "It places each unchanged footprint uniformly over admissible positions in the eligible nuclear region. It removes the effect of MIAT abundance/coverage and of global nuclear QKI level under this uniform-position reference.",
        "It does NOT remove association caused by MIAT puncta and QKI both avoiding or preferring the same nuclear sub-regions within that eligible region (e.g. around nucleolar exclusion zones or the nuclear periphery).",
        'A positive obs minus null means "more QKI at MIAT puncta than at random eligible nuclear positions", not molecular binding. Single-plane measurement.',
        "", "## Column definitions", ""]
    for name in columns:
        if name not in COLUMN_DEFINITIONS:
            raise ValueError(f"missing column definition: {name}")
        definition = COLUMN_DEFINITIONS[name]
        if name == "spot_id":
            definition = "Saved source spot identifier, retained without renumbering; identifier; single-plane."
        lines.append(f"- `{name}`: {definition}")
    (destination / "qki_association_columns.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    command = shlex.join(["fishsuite", "qki-assoc", "--run-dir", str(source), "--miat-min", str(miat_min),
        "--qki-min", str(qki_min), "--sensitivity", ",".join(map(str, levels)), "--n-null", str(n_null),
        "--seed", str(seed), "--out", str(destination)])
    if not repro.write_command_log(destination, source / "image_manifest.csv", destination, seed,
            extra={"qki_assoc_command": command, "consumed_files": ", ".join(REQUIRED_FILES), "quantitation": "single-plane"}):
        raise OSError("failed to write command.log")
    if not repro.write_versions_txt(destination, seed):
        raise OSError("failed to write versions.txt")
    return destination
