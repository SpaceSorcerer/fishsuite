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


def _ccf_field_tables(records):
    """Per-field CCF curves (mean over nuclei of per-nucleus r(d)) and the
    per-field registration summary. Pixels are never pooled across nuclei."""
    curve_columns = ["image", "condition", "well", "axis", "shift_px", "shift_um",
                     "mean_r_over_nuclei", "sd_r_over_nuclei", "n_nuclei_finite", "n_nuclei"]
    reg_columns = ["image", "condition", "well", "axis", "pixel_size_um", "peak_shift_px",
                   "peak_shift_um", "peak_r", "r0", "r0_minus_flank", "fwhm_px",
                   "n_nuclei_finite", "median_nucleus_peak_shift_px",
                   "frac_nuclei_peak_at_zero", "na_reason"]
    if not records:
        return pd.DataFrame(columns=curve_columns), pd.DataFrame(columns=reg_columns)
    from .coloc_pixel_metrics import ccf_summary
    frame = pd.DataFrame(records)
    keys = ["image", "condition", "well", "axis", "shift_px"]
    curves = (frame.groupby(keys, sort=True)
              .agg(shift_um=("shift_um", "first"),
                   mean_r_over_nuclei=("r", "mean"),
                   sd_r_over_nuclei=("r", lambda v: float(v.dropna().std(ddof=1)) if v.notna().sum() > 1 else np.nan),
                   n_nuclei_finite=("r", "count"), n_nuclei=("nucleus_id", "nunique"))
              .reset_index())
    rows = []
    for (image, condition, well, axis), group in curves.groupby(["image", "condition", "well", "axis"], sort=True):
        group = group.sort_values("shift_px")
        summary = ccf_summary(group.shift_px.to_numpy(), group.mean_r_over_nuclei.to_numpy())
        nuc = frame[(frame.image == image) & (frame.axis == axis)]
        peaks = []
        for _, curve in nuc.groupby("nucleus_id", sort=True):
            curve = curve.sort_values("shift_px")
            peak = ccf_summary(curve.shift_px.to_numpy(), curve.r.to_numpy())["peak_shift_px"]
            if np.isfinite(peak):
                peaks.append(peak)
        zero = group.loc[group.shift_px.eq(0)]
        pixel = float(group.shift_um.iloc[-1] / group.shift_px.iloc[-1]) if group.shift_px.iloc[-1] else np.nan
        rows.append(dict(image=image, condition=condition, well=well, axis=axis, pixel_size_um=pixel,
                         peak_shift_px=summary["peak_shift_px"],
                         peak_shift_um=summary["peak_shift_px"] * pixel if np.isfinite(summary["peak_shift_px"]) else np.nan,
                         peak_r=summary["peak_r"], r0=summary["r0"], r0_minus_flank=summary["r0_minus_flank"],
                         fwhm_px=summary["fwhm_px"],
                         n_nuclei_finite=int(zero.n_nuclei_finite.iloc[0]) if len(zero) else 0,
                         median_nucleus_peak_shift_px=float(np.median(peaks)) if peaks else np.nan,
                         frac_nuclei_peak_at_zero=float(np.mean(np.asarray(peaks) == 0)) if peaks else np.nan,
                         na_reason=summary["reason"] or summary["fwhm_reason"]))
    return curves[curve_columns], pd.DataFrame(rows, columns=reg_columns)


def run_qki_association(run_dir, out, *, miat_min, qki_min,
                        sensitivity=(0.8, 1.0, 1.25), n_null=200, seed=0,
                        n_costes=200, conditions=None, image_keys=None):
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
    if isinstance(n_costes, bool) or not isinstance(n_costes, (int, np.integer)) or n_costes <= 0:
        raise ValueError("n_costes must be a positive integer")
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
    selected_manifest = manifest
    if conditions:
        wanted = {str(c) for c in conditions}
        selected_manifest = selected_manifest[selected_manifest.condition.astype(str).isin(wanted)]
    if image_keys:
        wanted = {str(k).strip().casefold() for k in image_keys}
        selected_manifest = selected_manifest[selected_manifest.image_key.isin(wanted)]
    if selected_manifest.empty:
        raise ValueError("the condition / image filter matched no image in the manifest")
    all_nuclei, all_spots, ccf_records = [], [], []
    for record in selected_manifest.sort_values("image_key").itertuples(index=False):
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
            well=str(getattr(record, "well", "")), n_costes=int(n_costes), ccf_records=ccf_records)
        spot_table["spot_id"] = spot_table.spot_id.map(identifiers)
        all_nuclei.append(nuc_table)
        all_spots.append(spot_table)
    nucleus_table = pd.concat(all_nuclei, ignore_index=True)
    spot_table = pd.concat(all_spots, ignore_index=True)
    destination.mkdir(parents=True, exist_ok=True)
    nucleus_table.to_csv(destination / "qki_association_per_nucleus.csv", index=False, lineterminator="\n")
    spot_table.to_csv(destination / "qki_association_per_spot.csv", index=False, lineterminator="\n")
    ccf_curves, ccf_registration = _ccf_field_tables(ccf_records)
    ccf_curves.to_csv(destination / "qki_association_ccf_per_field.csv", index=False, lineterminator="\n")
    ccf_registration.to_csv(destination / "qki_association_ccf_registration.csv", index=False, lineterminator="\n")
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
        "", "## Threshold-free additions (2026-09-24)", "",
        "- `mean_null_rank_qki_at_miat`, `frac_spots_ge_null_q90`, `frac_spots_ge_null_q75`: each spot's footprint-mean raw QKI is ranked against its OWN placement-null draws (the same draws as above); no QKI cutoff; chance 0.5 / 0.10 / 0.25. Invariant to any monotone intensity transform, so neither the QKI offset nor the nuclear QKI level moves the chance level.",
        "- `pearson_r_nucleoplasm`, `spearman_rho_nucleoplasm`: over the nucleoplasm mask N (nucleoli excluded); `pearson_r_whole_nucleus_mask` is the nucleolus-inclusive comparator.",
        f"- Van Steensel CCF: shifts -20..+20 px (step 1 px) along x and y, pixel size from the manifest voxel_xy_nm; per-field curves in `qki_association_ccf_per_field.csv` (mean over nuclei of per-nucleus r(d)) and the channel-registration check in `qki_association_ccf_registration.csv`.",
        f"- Costes randomization: b x b QKI blocks (b = round(sqrt(median footprint area of the image)), >= 3 px) permuted within N, MIAT fixed, n_costes = {int(n_costes)} draws, seed = {int(seed)} bound to (image, nucleus_id) in a stream separate from the placement null. `costes_rand_p` is per-nucleus DESCRIPTIVE, not a test across nuclei; wells remain the replicates.",
        "- These statistics share the placement null's limit: none removes MIAT and QKI co-preferring the same nuclear sub-compartment.",
        "", "## Column definitions", ""]
    for name in columns:
        if name not in COLUMN_DEFINITIONS:
            raise ValueError(f"missing column definition: {name}")
        definition = COLUMN_DEFINITIONS[name]
        if name == "spot_id":
            definition = "Saved source spot identifier, retained without renumbering; identifier; single-plane."
        lines.append(f"- `{name}`: {definition}")
    (destination / "qki_association_columns.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    argv = ["fishsuite", "qki-assoc", "--run-dir", str(source), "--miat-min", str(miat_min),
        "--qki-min", str(qki_min), "--sensitivity", ",".join(map(str, levels)), "--n-null", str(n_null),
        "--seed", str(seed), "--n-costes", str(int(n_costes))]
    for condition in conditions or ():
        argv += ["--condition", str(condition)]
    command = shlex.join(argv + ["--out", str(destination)])
    extra = {"qki_assoc_command": command, "consumed_files": ", ".join(REQUIRED_FILES),
             "quantitation": "single-plane", "n_costes": int(n_costes),
             "costes_seed_stream": "sha256([seed, image_key, nucleus_id, 'costes_block_scramble'])",
             "ccf_shift_range_px": "-20..20 step 1",
             "images_processed": len(selected_manifest)}
    if image_keys:
        extra["image_keys_filter"] = "; ".join(sorted(str(k) for k in image_keys))
    if not repro.write_command_log(destination, source / "image_manifest.csv", destination, seed,
            extra=extra):
        raise OSError("failed to write command.log")
    if not repro.write_versions_txt(destination, seed):
        raise OSError("failed to write versions.txt")
    return destination
