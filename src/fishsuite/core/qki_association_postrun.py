"""Saved exact-footprint single-plane adapter for QKI association metrics."""
from __future__ import annotations

import json
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


OPTICS_KEYS = ("numerical_aperture", "emission_nm_miat", "emission_nm_qki")
_NUCLEOLUS_KEYS = ("intra_nuclear_percentile", "min_area_um2", "max_area_frac_of_nucleus",
                   "closing_radius_px", "min_border_distance_px")


def _explicit_optics(optics):
    missing = [k for k in OPTICS_KEYS if optics.get(k) is None]
    if missing:
        raise ValueError("explicit optics must give " + ", ".join(OPTICS_KEYS) + "; missing " + ", ".join(missing))
    values = {k: float(optics[k]) for k in OPTICS_KEYS}
    label = "explicit (" + ", ".join(f"{k}={values[k]:g}" for k in OPTICS_KEYS) + ")"
    return values, label


def _metadata_optics(image_key, channels):
    """Objective NA and per-channel emission wavelength from the source image's
    OME metadata. Raises when the file or any value is unavailable."""
    path = Path(str(image_key))
    if not path.is_file():
        raise ValueError(
            f"Costes PSF needs objective numerical aperture (NA) and emission wavelengths; the source "
            f"image {image_key} is not readable and no explicit optics were given "
            "(--objective-na / --emission-nm-miat / --emission-nm-qki)")
    from . import io as _io
    try:
        img = _io.read_image(path)
    except Exception as exc:
        raise ValueError(f"Costes PSF needs objective numerical aperture (NA) and emission wavelengths; "
                         f"OME metadata of {image_key} is unreadable ({exc}); supply --objective-na / "
                         "--emission-nm-miat / --emission-nm-qki") from exc
    try:
        meta = img.bio.ome_metadata
        image = meta.images[0]
        objective_id = image.objective_settings.id if image.objective_settings else None
        objectives = [o for inst in meta.instruments for o in inst.objectives]
        chosen = [o for o in objectives if o.id == objective_id] or (objectives if len(objectives) == 1 else [])
        na = chosen[0].lens_na if chosen else None
        channel_meta = image.pixels.channels
        emission = {}
        for role in ("miat", "qki"):
            index = int(channels[role])
            channel = channel_meta[index] if index < len(channel_meta) else None
            value = channel.emission_wavelength if channel is not None else None
            unit = str(getattr(channel, "emission_wavelength_unit", "")) if channel is not None else ""
            if value is not None and "NANOMETER" not in unit.upper() and unit:
                raise ValueError(f"emission wavelength unit {unit} for {role} in {image_key} is not nm")
            emission[role] = value
    finally:
        try:
            img.bio.__exit__(None, None, None)
        except Exception:
            pass
    missing = [name for name, value in (("objective numerical aperture (NA)", na),
                                        ("MIAT emission wavelength", emission["miat"]),
                                        ("QKI emission wavelength", emission["qki"])) if value is None]
    if missing:
        raise ValueError(f"source OME metadata of {image_key} lacks " + ", ".join(missing)
                         + "; supply --objective-na / --emission-nm-miat / --emission-nm-qki")
    values = dict(numerical_aperture=float(na), emission_nm_miat=float(emission["miat"]),
                  emission_nm_qki=float(emission["qki"]))
    return values, f"source OME metadata: {image_key}"


def _nucleolus_parameters(source):
    """Production nucleolus parameters and their source, from the backfill's
    analysis_parameters.json -> source_run_dir/run_config.json. (None, reason)
    when not recorded."""
    import json
    parameters = source / "analysis_parameters.json"
    if not parameters.is_file():
        return None, "analysis_parameters.json absent"
    run_dir = json.loads(parameters.read_text(encoding="utf-8")).get("source_run_dir")
    config = Path(str(run_dir)) / "run_config.json" if run_dir else None
    if config is None or not config.is_file():
        return None, f"source run_config.json not found (source_run_dir={run_dir!r})"
    resolved = json.loads(config.read_text(encoding="utf-8")).get("config_resolved", {})
    block = resolved.get("nucleolus") or {}
    missing = [k for k in _NUCLEOLUS_KEYS if k not in block]
    if missing:
        return None, f"{config} lacks nucleolus keys {missing}"
    return {k: block[k] for k in _NUCLEOLUS_KEYS}, str(config)


def _nucleolus_sensitivity_tables(rows, production_percentile):
    per_nucleus = pd.DataFrame(rows)
    if per_nucleus.empty:
        return per_nucleus, pd.DataFrame()
    per_well = (per_nucleus.groupby(["condition", "well", "percentile"], sort=True)
                .agg(n_nuclei=("nucleus_id", "size"),
                     mean_nucleolus_area_frac=("nucleolus_area_frac", "mean"),
                     well_mean_pearson_r_nucleoplasm=("pearson_r_nucleoplasm", "mean"),
                     well_mean_spearman_rho_nucleoplasm=("spearman_rho_nucleoplasm", "mean"),
                     saved_mask_reproduced_frac=("saved_mask_reproduced", "mean"))
                .reset_index())
    base = per_well[per_well.percentile.eq(production_percentile)].set_index(["condition", "well"])
    key = pd.MultiIndex.from_frame(per_well[["condition", "well"]])
    per_well["delta_pearson_vs_production"] = (
        per_well.well_mean_pearson_r_nucleoplasm.to_numpy()
        - base.well_mean_pearson_r_nucleoplasm.reindex(key).to_numpy())
    per_well["delta_spearman_vs_production"] = (
        per_well.well_mean_spearman_rho_nucleoplasm.to_numpy()
        - base.well_mean_spearman_rho_nucleoplasm.reindex(key).to_numpy())
    per_well["production_percentile"] = production_percentile
    return per_nucleus, per_well


def run_qki_association(run_dir, out, *, miat_min, qki_min,
                        sensitivity=(0.8, 1.0, 1.25), n_null=200, seed=0,
                        n_costes=200, conditions=None, image_keys=None, optics=None,
                        nucleolus_percentiles=()):
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
    if optics is not None:
        explicit_optics = _explicit_optics(optics)
    nucleolus_params, nucleolus_source = _nucleolus_parameters(source)
    percentiles = tuple(float(v) for v in (nucleolus_percentiles or ()))
    if percentiles and nucleolus_params is None:
        raise ValueError("nucleolus sensitivity needs the production nucleolus parameters: "
                         + nucleolus_source)
    all_nuclei, all_spots, ccf_records, sensitivity_rows, optics_log = [], [], [], [], {}
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
        from .coloc_pixel_metrics import nucleoplasm_sensitivity, psf_fwhm_px
        optic_values, optic_source = (explicit_optics if optics is not None
                                      else _metadata_optics(key, channels))
        emission = max(optic_values["emission_nm_miat"], optic_values["emission_nm_qki"])
        psf_px = psf_fwhm_px(emission, optic_values["numerical_aperture"], scale * 1000.0)
        optics_log[key] = dict(optic_values, source=optic_source, psf_fwhm_px=psf_px)
        if percentiles:
            table = nucleoplasm_sensitivity(data.planes["miat"], data.planes["qki"], data.planes["dapi"],
                                            data.nucleus_labels, scale, nucleolus_params, percentiles)
            if float(nucleolus_params["intra_nuclear_percentile"]) in percentiles:
                from .nucleolus import NucleolusParams, detect_nucleoli
                production = detect_nucleoli(data.nucleus_labels, data.planes["dapi"], scale,
                                             NucleolusParams(**nucleolus_params))
                same = {int(n): bool(np.array_equal(production == n, data.nucleolus_labels == n))
                        for n in np.unique(data.nucleus_labels[data.nucleus_labels > 0])}
            else:
                same = {}
            for row in table.to_dict("records"):
                row.update(image=key, condition=str(getattr(record, "condition", "")),
                           well=str(getattr(record, "well", "")),
                           saved_mask_reproduced=same.get(row["nucleus_id"], np.nan)
                           if row["percentile"] == float(nucleolus_params["intra_nuclear_percentile"]) else np.nan)
                sensitivity_rows.append(row)
        nuc_table, spot_table = association_tables(data.planes["miat"], data.planes["qki"],
            data.nucleus_labels, footprints, pixel_size_um=scale, miat_min=miat_min,
            qki_min=qki_min, sensitivity=levels, n_null=n_null, seed=seed,
            eligible_mask=eligible, image=key, condition=str(getattr(record, "condition", "")),
            well=str(getattr(record, "well", "")), n_costes=int(n_costes), ccf_records=ccf_records,
            psf_fwhm_px=psf_px, psf_source=optic_source)
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
    if percentiles:
        sens_nucleus, sens_well = _nucleolus_sensitivity_tables(
            sensitivity_rows, float(nucleolus_params["intra_nuclear_percentile"]))
        sens_nucleus.to_csv(destination / "qki_association_nucleolus_sensitivity_per_nucleus.csv",
                            index=False, lineterminator="\n")
        sens_well.to_csv(destination / "qki_association_nucleolus_sensitivity_per_well.csv",
                         index=False, lineterminator="\n")
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
        "- Uniform-position percentile score (`uniform_position_percentile_qki`, `mean_uniform_position_percentile_qki`, `frac_spots_upp_ge_0p90`, `frac_spots_upp_ge_0p75`): each spot's footprint-mean raw QKI is ranked against its OWN uniform-position placement draws (the same draws as above), ties broken at random; no QKI cutoff. Chance references are finite-K: mean 0.5, P(u >= 0.9) = (K - ceil(0.9K) + 1)/(K + 1) = 21/201 at K = 200, P(u >= 0.75) = 51/201. The score is exactly uniform only if MIAT centres are exchangeable with uniform admissible positions; departures from that (e.g. MIAT preferring a sub-compartment) are part of what it measures. Coupling adds a spot-pooled well value beside the per-nucleus mean.",
        "- `pearson_r_nucleoplasm`, `spearman_rho_nucleoplasm`: over the nucleoplasm mask N (nucleoli excluded); `pearson_r_whole_nucleus_mask` is the nucleolus-inclusive comparator.",
        f"- Van Steensel CCF: shifts -20..+20 px (step 1 px) along x and y, pixel size from the manifest voxel_xy_nm; per-field curves in `qki_association_ccf_per_field.csv` (mean over nuclei of per-nucleus r(d)) and the channel-registration check in `qki_association_ccf_registration.csv`.",
        f"- Costes randomization (round 3): **`costes_rand_p` is NOT_CALIBRATED** (texture-matched synthetic gate failed: 8.25% false positives at alpha 0.05, 2.75% at 0.01, n = 400) and is a QC descriptor only. b = ceil(max(PSF FWHM, ACF FWHM of the scrambled channel QKI)) px (round 2 used min(MIAT, QKI), which undersized the blocks), PSF = 0.51 lambda_em / NA from the recorded optics, ACF = overlap-normalized mean-centered 2-D autocorrelation over in-mask pairs, radially averaged; tile phase chosen from mask geometry to maximise complete-block coverage; observed r and all {int(n_costes)} permutations on the frozen core N_core; NA (MASK_BLOCK_COVERAGE) below 10 blocks or 80% coverage; seed = {int(seed)} bound to (image, nucleus_id) in its own stream. `costes_rand_p` is per-nucleus DESCRIPTIVE, not a test across nuclei, and is excluded from coupling contrasts; wells remain the replicates.",
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
    if optics is not None:
        argv += ["--objective-na", f"{explicit_optics[0]['numerical_aperture']:g}",
                 "--emission-nm-miat", f"{explicit_optics[0]['emission_nm_miat']:g}",
                 "--emission-nm-qki", f"{explicit_optics[0]['emission_nm_qki']:g}"]
    if percentiles:
        argv += ["--nucleolus-sensitivity", ",".join(f"{v:g}" for v in percentiles)]
    command = shlex.join(argv + ["--out", str(destination)])
    extra = {"qki_assoc_command": command, "consumed_files": ", ".join(REQUIRED_FILES),
             "quantitation": "single-plane", "n_costes": int(n_costes),
             "costes_seed_stream": "sha256([seed, image_key, nucleus_id, 'costes_block_scramble'])",
             "ccf_shift_range_px": "-20..20 step 1",
             "images_processed": len(selected_manifest),
             "upp_tiebreak_stream": "sha256([seed, image_key, nucleus_id, 'upp_tiebreak'])",
             "psf_formula": "0.51 * max(emission_nm_miat, emission_nm_qki) / objective_na / voxel_xy_nm",
             "costes_block_rule": "ceil(max(psf_fwhm_px, acf_fwhm_px_qki)) [scrambled channel = QKI; round-3 change from min(miat, qki)]; >=10 blocks and >=0.80 coverage",
             "costes_p_calibration": "NOT_CALIBRATED (QC only)",
             "nucleolus_params": (json.dumps(nucleolus_params, sort_keys=True) if nucleolus_params is not None
                                  else "not recorded: " + nucleolus_source + "; saved nucleolus_labels used as-is"),
             "nucleolus_params_source": nucleolus_source,
             "nucleolus_sensitivity_percentiles": ",".join(f"{v:g}" for v in percentiles) or "none"}
    distinct = {(v["numerical_aperture"], v["emission_nm_miat"], v["emission_nm_qki"]) for v in optics_log.values()}
    if len(distinct) == 1:
        na, em_m, em_q = next(iter(distinct))
        extra.update(objective_na=na, emission_nm_miat=em_m, emission_nm_qki=em_q)
    extra["optics_per_image"] = json.dumps(optics_log, sort_keys=True)
    if image_keys:
        extra["image_keys_filter"] = "; ".join(sorted(str(k) for k in image_keys))
    if not repro.write_command_log(destination, source / "image_manifest.csv", destination, seed,
            extra=extra):
        raise OSError("failed to write command.log")
    if not repro.write_versions_txt(destination, seed):
        raise OSError("failed to write versions.txt")
    return destination
