"""Final-publication micrographs from frozen MIAT--QKI exact-single-z data."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from fishsuite.core.exact_footprint_figures import (
    PRIMARY_DISPLAY_WINDOWS, _draw_overlay_specs, _explicit_bool, _pyplot,
    _qc_pass, _require_columns, _sha256_file, _show_gray_qki, _show_rgb,
    build_call_overlay_specs, load_selected_plane, nucleus_square_crop,
    render_channel_rgb,
)

_PANELS = ("DAPI", "MIAT", "QKI", "merge", "q95 overlay")
_Q95_DEFINITION = "q95: QKI greater than 95% of 1,000 same-nucleus KEEP-N randomized same-shape placements."
_Q95_CLASSES = ("threshold_positive", "threshold_negative", "unusable")
_CHANNELS = {"miat": 0, "qki": 1, "dapi": 2}
_CROP_MARGIN_UM = 2.0


@dataclass(frozen=True)
class PublicationMicrographSelection:
    field_audit: pd.DataFrame
    nucleus_audit: pd.DataFrame
    manifest: pd.DataFrame


@dataclass(frozen=True)
class PublicationMicrographOutputs:
    output_dir: Path
    individual_png_paths: tuple[Path, ...]
    individual_svg_paths: tuple[Path, ...]
    contact_sheet_paths: tuple[Path, ...]
    selection_audit_path: Path
    source_data_path: Path
    manifest_path: Path


def _condition(arm: object) -> str:
    text = str(arm).strip().casefold()
    if text == "nt":
        return "NT"
    if text in {"kd", "miat-kd"}:
        return "MIAT-KD"
    raise ValueError(f"unsupported publication condition: {arm!r}")


def _central_distance(values: pd.Series) -> tuple[pd.Series, float, float]:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric.loc[np.isfinite(numeric)]
    if finite.empty:
        return pd.Series(np.nan, index=values.index, dtype=float), np.nan, np.nan
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    if mad == 0:
        result = pd.Series(np.nan, index=values.index, dtype=float)
        finite_mask = np.isfinite(numeric)
        result.loc[finite_mask & numeric.eq(median)] = 0.0
        result.loc[finite_mask & ~numeric.eq(median)] = np.inf
        return result, median, mad
    return (numeric - median).abs() / mad, median, mad


def _field_audit(images: pd.DataFrame, spots: pd.DataFrame, nuclei: pd.DataFrame) -> pd.DataFrame:
    _require_columns(images, {"image_key", "image", "slide", "arm", "biological_set", "is_control", "secondary_only", "selected_z_1based", "voxel_xy_nm", "miat_channel_index", "qki_channel_index", "dapi_channel_index", "plane_lock_pass", "mask_qc_pass", "footprint_parity_pass", "population_reconciliation_pass", "image_qc_status"}, table="image audit")
    _require_columns(spots, {"image_key", "nucleus_id", "spot_uid", "null_usable", "qki_threshold_positive_q95", "stored_in_nucleolus", "qki_footprint_enrichment_vs_nucleoplasm"}, table="corrected spot table")
    _require_columns(nuclei, {"image_key", "nucleus_id", "nucleus_uid", "nucleus_qc_status"}, table="nucleus QC roster")
    images, spots, nuclei = images.copy(), spots.copy(), nuclei.copy()
    for frame in (images, spots, nuclei):
        frame["image_key"] = frame["image_key"].astype(str).str.casefold()
    if images["image_key"].duplicated().any():
        raise ValueError("image audit must contain one row per image_key")
    if spots["spot_uid"].astype(str).duplicated().any():
        raise ValueError("corrected spot table spot_uid values must be unique")
    if nuclei.duplicated(["image_key", "nucleus_id"]).any():
        raise ValueError("nucleus QC roster must be unique by image_key/nucleus_id")
    records: list[dict[str, object]] = []
    checks = ("plane_lock_pass", "mask_qc_pass", "footprint_parity_pass", "population_reconciliation_pass")
    for image in images.sort_values("image_key", kind="mergesort").to_dict("records"):
        key = str(image["image_key"])
        roster = nuclei.loc[nuclei["image_key"].eq(key) & nuclei["nucleus_qc_status"].map(_qc_pass)]
        roster_ids = set(pd.to_numeric(roster["nucleus_id"], errors="coerce").dropna().astype(int))
        group = spots.loc[spots["image_key"].eq(key) & pd.to_numeric(spots["nucleus_id"], errors="coerce").isin(roster_ids)]
        usable = group["null_usable"].map(_explicit_bool) if len(group) else pd.Series(dtype=bool)
        calls = pd.Series(False, index=group.index, dtype=bool)
        if usable.any():
            calls.loc[usable] = group.loc[usable, "qki_threshold_positive_q95"].map(_explicit_bool)
        nonnucleolar = ~group["stored_in_nucleolus"].map(_explicit_bool) if len(group) else pd.Series(dtype=bool)
        n_usable = int(usable.sum())
        qki = pd.to_numeric(group.loc[usable, "qki_footprint_enrichment_vs_nucleoplasm"], errors="coerce")
        indices = {role: int(image[f"{role}_channel_index"]) for role in _CHANNELS}
        if indices != _CHANNELS:
            raise ValueError(f"image audit channel indices must be canonical: {indices}")
        passed = {name: _explicit_bool(image[name]) for name in checks}
        is_control = bool(
            _explicit_bool(image["is_control"])
            or _explicit_bool(image["secondary_only"])
        )
        row = dict(image)
        row.update({"condition": "CONTROL" if is_control else _condition(image["arm"]), "source_image": image["image"], "eligible": bool(not is_control and all(passed.values()) and _qc_pass(image["image_qc_status"]) and len(roster) > 0 and n_usable > 0), "n_qc_passing_nuclei": int(len(roster)), "n_usable_q95": n_usable, "n_q95_positive": int((usable & calls).sum()), "association_fraction_q95": int((usable & calls).sum()) / n_usable if n_usable else np.nan, "non_nucleolar_spots_per_qc_nucleus": float(nonnucleolar.sum() / len(roster)) if len(roster) else np.nan, "median_qki_enrichment_vs_nucleoplasm": float(qki.median()) if qki.notna().any() else np.nan})
        records.append(row)
    return pd.DataFrame(records)


def _select_nuclei_for_fields(
    field_manifest: pd.DataFrame,
    spots: pd.DataFrame,
    nuclei: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Choose one field-relative central QC-passing nucleus per field."""
    audits: list[pd.DataFrame] = []
    selected_rows: list[dict[str, object]] = []
    for field in field_manifest.to_dict("records"):
        key = str(field["image_key"])
        roster = nuclei.loc[
            nuclei["image_key"].eq(key)
            & nuclei["nucleus_qc_status"].map(_qc_pass)
        ].sort_values("nucleus_id", kind="mergesort")
        rows: list[dict[str, object]] = []
        for nucleus in roster.to_dict("records"):
            nucleus_id = int(nucleus["nucleus_id"])
            group = spots.loc[
                spots["image_key"].eq(key)
                & pd.to_numeric(spots["nucleus_id"], errors="coerce").eq(
                    nucleus_id
                )
            ]
            usable = (
                group["null_usable"].map(_explicit_bool)
                if len(group)
                else pd.Series(dtype=bool)
            )
            calls = pd.Series(False, index=group.index, dtype=bool)
            if usable.any():
                calls.loc[usable] = group.loc[
                    usable, "qki_threshold_positive_q95"
                ].map(_explicit_bool)
            positive = usable & calls
            nonnucleolar = (
                ~group["stored_in_nucleolus"].map(_explicit_bool)
                if len(group)
                else pd.Series(dtype=bool)
            )
            enrichment = pd.to_numeric(
                group.loc[usable, "qki_footprint_enrichment_vs_nucleoplasm"],
                errors="coerce",
            )
            rows.append(
                {
                    **field,
                    "nucleus_id": nucleus_id,
                    "nucleus_uid": str(nucleus["nucleus_uid"]),
                    "nucleus_qc_status": str(nucleus["nucleus_qc_status"]),
                    "n_scoped_spots": int(len(group)),
                    "n_null_usable": int(usable.sum()),
                    "n_q95_positive": int(positive.sum()),
                    "n_q95_negative": int((usable & ~calls).sum()),
                    "n_q95_unusable": int(len(group) - usable.sum()),
                    "association_fraction_q95_nucleus": (
                        float(positive.sum() / usable.sum())
                        if usable.sum()
                        else np.nan
                    ),
                    "n_non_nucleolar_spots": int(nonnucleolar.sum()),
                    "log1p_non_nucleolar_spots": float(
                        np.log1p(nonnucleolar.sum())
                    ),
                    "median_qki_enrichment_nucleus": (
                        float(enrichment.median())
                        if enrichment.notna().any()
                        else np.nan
                    ),
                }
            )
        audit = pd.DataFrame(rows)
        if audit.empty:
            raise ValueError(f"selected image {key!r} lacks a QC-passing nucleus")
        total = pd.Series(0.0, index=audit.index)
        for metric in (
            "association_fraction_q95_nucleus",
            "log1p_non_nucleolar_spots",
        ):
            distance, median, mad = _central_distance(audit[metric])
            audit[f"field_median_{metric}"] = median
            audit[f"field_mad_{metric}"] = mad
            audit[f"nucleus_distance_{metric}"] = distance
            total = total.add(distance.fillna(np.inf), fill_value=0.0)
        audit["nucleus_centrality_distance"] = total
        audit = audit.sort_values(
            ["nucleus_centrality_distance", "n_q95_unusable", "nucleus_id"],
            kind="mergesort",
        ).reset_index(drop=True)
        audit["nucleus_rank"] = np.arange(1, len(audit) + 1)
        audit["selected"] = False
        audit.loc[0, "selected"] = True
        audit["nucleus_selection_rationale"] = (
            "field_relative_median_mad_centrality"
        )
        selected_rows.append(dict(audit.iloc[0]))
        audits.append(audit)
    return pd.concat(audits, ignore_index=True), pd.DataFrame(selected_rows)


def select_publication_micrograph_examples(image_audit: pd.DataFrame, spots: pd.DataFrame, nuclei: pd.DataFrame, *, representatives_per_arm: int = 3) -> PublicationMicrographSelection:
    """Select three condition-relative central fields per arm, distinct sets first."""
    if int(representatives_per_arm) != 3:
        raise ValueError("final publication package requires exactly 3 examples per arm")
    fields = _field_audit(image_audit, spots, nuclei)
    metrics = ("association_fraction_q95", "non_nucleolar_spots_per_qc_nucleus", "median_qki_enrichment_vs_nucleoplasm")
    fields["condition_centrality_distance"] = np.nan
    for condition in ("NT", "MIAT-KD"):
        index = fields.index[fields["condition"].eq(condition)].tolist()
        candidates = fields.loc[index].loc[fields.loc[index, "eligible"]]
        if len(candidates) < 3:
            raise ValueError(f"{condition} has fewer than 3 valid candidates")
        total = pd.Series(0.0, index=candidates.index)
        for metric in metrics:
            distance, median, mad = _central_distance(candidates[metric])
            fields.loc[index, f"condition_median_{metric}"] = median
            fields.loc[index, f"condition_mad_{metric}"] = mad
            fields.loc[candidates.index, f"condition_distance_{metric}"] = distance
            total = total.add(distance.fillna(np.inf), fill_value=0.0)
        fields.loc[candidates.index, "condition_centrality_distance"] = total
    parts: list[pd.DataFrame] = []
    for condition in ("NT", "MIAT-KD"):
        candidates = fields.loc[fields["condition"].eq(condition) & fields["eligible"]].sort_values(["condition_centrality_distance", "image_key"], kind="mergesort")
        selected = candidates.drop_duplicates("biological_set", keep="first").head(3).copy()
        if len(selected) < 3:
            selected = pd.concat([selected, candidates.loc[~candidates["image_key"].isin(selected["image_key"])].head(3 - len(selected))])
        if len(selected) != 3:
            raise ValueError(f"{condition} cannot provide exactly 3 examples")
        selected["example_number"] = range(1, 4)
        selected["selection_rationale"] = "condition_relative_median_mad_centrality"
        parts.append(selected)
    field_manifest = pd.concat(parts, ignore_index=True)[["condition", "example_number", "image_key", "source_image", "slide", "biological_set", "selected_z_1based", "voxel_xy_nm", "miat_channel_index", "qki_channel_index", "dapi_channel_index", "association_fraction_q95", "non_nucleolar_spots_per_qc_nucleus", "median_qki_enrichment_vs_nucleoplasm", "condition_centrality_distance", "selection_rationale"]]
    nucleus_audit, manifest = _select_nuclei_for_fields(
        field_manifest, spots, nuclei
    )
    fields["selected"] = fields["image_key"].isin(set(manifest["image_key"]))
    return PublicationMicrographSelection(fields, nucleus_audit, manifest)


def _input_table(value: pd.DataFrame | str | Path, *, name: str) -> tuple[pd.DataFrame, dict[str, str]]:
    if isinstance(value, pd.DataFrame):
        frame = value.copy()
        columns = sorted(frame.columns.astype(str).tolist())
        text = frame.loc[:, columns].copy()
        for column in columns:
            text[column] = text[column].map(lambda item: "" if pd.isna(item) else str(item))
        payload = text.sort_values(columns, kind="mergesort").to_csv(index=False, lineterminator="\n").encode("utf-8")
        return frame, {"mode": "deterministic_dataframe_csv", "sha256": hashlib.sha256(payload).hexdigest()}
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.casefold() == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix.casefold() == ".tsv":
        frame = pd.read_csv(path, sep="\t")
    elif path.suffix.casefold() in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    else:
        raise ValueError(f"{name} table path has unsupported format: {path}")
    return frame, {"mode": "file_bytes_sha256", "sha256": _sha256_file(path)}


def _merge(*channels: np.ndarray) -> np.ndarray:
    return np.clip(sum(channel.astype(np.uint16) for channel in channels), 0, 255).astype(np.uint8)


def _crop(array: np.ndarray, bounds: Any) -> np.ndarray:
    return array[bounds.y0:bounds.y1, bounds.x0:bounds.x1]


def _save(figure: Any, path: Path, *, dpi: int) -> tuple[Path, Path]:
    png, svg = path.with_suffix(".png"), path.with_suffix(".svg")
    path.parent.mkdir(parents=True, exist_ok=True)
    for target in (png, svg):
        figure.savefig(target, dpi=dpi, bbox_inches="tight", pad_inches=0.04, facecolor="black")
    return png, svg


def render_publication_micrograph_package(selected_planes_h5: str | Path, image_audit: pd.DataFrame | str | Path, spots: pd.DataFrame | str | Path, nuclei: pd.DataFrame | str | Path, footprint_pixels: pd.DataFrame | str | Path, output_dir: str | Path, *, representatives_per_arm: int = 3, dpi: int = 600) -> PublicationMicrographOutputs:
    """Render exact-plane individual/contact-sheet PNG+SVG at 600 dpi only."""
    if int(dpi) != 600:
        raise ValueError("final publication micrographs require exactly 600 dpi")
    image_audit, image_hash = _input_table(image_audit, name="image audit")
    spots, spots_hash = _input_table(spots, name="spot calls")
    nuclei, nuclei_hash = _input_table(nuclei, name="nucleus QC")
    footprint_pixels, pixels_hash = _input_table(footprint_pixels, name="footprint pixels")
    _require_columns(footprint_pixels, {"spot_uid", "pixel_index", "y_px", "x_px"}, table="footprint pixel table")
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    selected = select_publication_micrograph_examples(image_audit, spots, nuclei, representatives_per_arm=representatives_per_arm)
    table, roster = spots.copy(), nuclei.copy()
    table["image_key"], roster["image_key"] = table["image_key"].astype(str).str.casefold(), roster["image_key"].astype(str).str.casefold()
    manifest = selected.manifest.copy().reset_index(drop=True)
    nucleus_audit = selected.nucleus_audit.copy()
    crop_columns = {
        "crop_y0": np.nan,
        "crop_y1": np.nan,
        "crop_x0": np.nan,
        "crop_x1": np.nan,
        "requested_margin_um": _CROP_MARGIN_UM,
        "requested_margin_px": np.nan,
        "crop_clamped": False,
        "crop_height_px": np.nan,
        "crop_width_px": np.nan,
        "crop_height_um": np.nan,
        "crop_width_um": np.nan,
    }
    for column, default in crop_columns.items():
        manifest[column] = default
        nucleus_audit[column] = default

    rendered: dict[str, dict[str, object]] = {}
    for manifest_index, entry in manifest.iterrows():
        key = str(entry["image_key"])
        expected_indices = {role: int(entry[f"{role}_channel_index"]) for role in _CHANNELS}
        plane = load_selected_plane(selected_planes_h5, key, expected_z_1based=int(entry["selected_z_1based"]), expected_channel_indices=expected_indices)
        nucleus_id = int(entry["nucleus_id"])
        valid = roster.loc[
            roster["image_key"].eq(key)
            & pd.to_numeric(roster["nucleus_id"], errors="coerce").eq(nucleus_id)
            & roster["nucleus_qc_status"].map(_qc_pass)
        ]
        if len(valid) != 1:
            raise ValueError(
                f"selected image {key!r} nucleus {nucleus_id} is not uniquely QC-passing"
            )
        if str(valid.iloc[0]["nucleus_uid"]) != str(entry["nucleus_uid"]):
            raise ValueError(
                f"selected image {key!r} nucleus UID disagrees with the QC roster"
            )
        bounds = nucleus_square_crop(
            plane.nucleus_labels,
            nucleus_id,
            float(entry["voxel_xy_nm"]),
            margin_um=_CROP_MARGIN_UM,
        )
        scoped = table.loc[table["image_key"].eq(key) & pd.to_numeric(table["nucleus_id"], errors="coerce").eq(nucleus_id)].copy()
        usable = scoped["null_usable"].map(_explicit_bool)
        calls = pd.Series(False, index=scoped.index, dtype=bool)
        if usable.any():
            calls.loc[usable] = scoped.loc[
                usable, "qki_threshold_positive_q95"
            ].map(_explicit_bool)
        scoped_counts = {
            "n_scoped_spots": int(len(scoped)),
            "n_null_usable": int(usable.sum()),
            "n_q95_positive": int((usable & calls).sum()),
            "n_q95_negative": int((usable & ~calls).sum()),
            "n_q95_unusable": int(len(scoped) - usable.sum()),
        }
        for column, value in scoped_counts.items():
            if int(entry[column]) != value:
                raise ValueError(
                    f"selected image {key!r} nucleus {nucleus_id} {column} "
                    "disagrees with the scoped spot table"
                )
        pixels = footprint_pixels.loc[footprint_pixels["spot_uid"].astype(str).isin(scoped["spot_uid"].astype(str))]
        specs = build_call_overlay_specs(scoped, pixels, percentile=95, crop=bounds)
        display = {role: _crop(render_channel_rgb(plane.planes[role], role), bounds) for role in _CHANNELS}
        display["merge"] = _merge(display["dapi"], display["miat"], display["qki"])
        qki = _crop(plane.planes["qki"], bounds)
        crop_values = {
            "crop_y0": int(bounds.y0),
            "crop_y1": int(bounds.y1),
            "crop_x0": int(bounds.x0),
            "crop_x1": int(bounds.x1),
            "requested_margin_um": _CROP_MARGIN_UM,
            "requested_margin_px": int(bounds.margin_px),
            "crop_clamped": bool(bounds.was_clamped),
            "crop_height_px": int(bounds.y1 - bounds.y0),
            "crop_width_px": int(bounds.x1 - bounds.x0),
            "crop_height_um": float(bounds.height_um),
            "crop_width_um": float(bounds.width_um),
        }
        for column, value in {**crop_values, **scoped_counts}.items():
            manifest.loc[manifest_index, column] = value
        audit_match = (
            nucleus_audit["image_key"].eq(key)
            & pd.to_numeric(nucleus_audit["nucleus_id"], errors="coerce").eq(
                nucleus_id
            )
        )
        if int(audit_match.sum()) != 1:
            raise ValueError(
                f"nucleus selection audit cannot resolve {key!r} nucleus {nucleus_id}"
            )
        for column, value in crop_values.items():
            nucleus_audit.loc[audit_match, column] = value
        rendered[key] = {
            "display": display,
            "qki": qki,
            "specs": specs,
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent)
    )
    plt, png_rel, svg_rel = _pyplot(), [], []
    try:
        for entry in manifest.to_dict("records"):
            key = str(entry["image_key"])
            content = rendered[key]
            display = content["display"]
            qki = content["qki"]
            specs = content["specs"]
            figure, axes = plt.subplots(1, 5, figsize=(15, 3.2), facecolor="black")
            for axis, panel in zip(axes[:4], _PANELS[:4], strict=True): _show_rgb(axis, display[panel.lower()], panel)
            _show_gray_qki(axes[4], qki, "q95 overlay"); _draw_overlay_specs(axes[4], specs)
            figure.suptitle(f"{entry['condition']} example {int(entry['example_number'])} — {_Q95_DEFINITION}", color="white", fontsize=8); figure.tight_layout()
            base = temporary / "individual" / f"{entry['condition'].lower().replace('-', '_')}_example_{int(entry['example_number'])}"
            png, svg = _save(figure, base, dpi=dpi); plt.close(figure)
            png_rel.append(png.relative_to(temporary)); svg_rel.append(svg.relative_to(temporary))
        contact_rel: list[Path] = []
        for condition in ("NT", "MIAT-KD"):
            entries = manifest.loc[manifest["condition"].eq(condition)]
            figure, axes = plt.subplots(3, 5, figsize=(15, 9), facecolor="black", squeeze=False)
            for row, entry in enumerate(entries.to_dict("records")):
                content = rendered[str(entry["image_key"])]
                for axis, panel in zip(axes[row, :4], _PANELS[:4], strict=True): _show_rgb(axis, content["display"][panel.lower()], panel)
                _show_gray_qki(axes[row, 4], content["qki"], "q95 overlay"); _draw_overlay_specs(axes[row, 4], content["specs"]); axes[row, 0].set_ylabel(f"{condition} example {int(entry['example_number'])}", color="white", fontsize=8)
            figure.suptitle(_Q95_DEFINITION, color="white", fontsize=8); figure.tight_layout()
            contact_paths = _save(figure, temporary / "contact_sheets" / f"{condition.lower().replace('-', '_')}_contact_sheet", dpi=dpi); plt.close(figure)
            contact_rel.extend(path.relative_to(temporary) for path in contact_paths)
        audit_rel = Path("selection_audit.csv")
        source_rel = Path("display_source_data.csv")
        manifest_rel = Path("publication_micrograph_manifest.json")
        nucleus_audit.to_csv(temporary / audit_rel, index=False)
        manifest.to_csv(temporary / source_rel, index=False)
        payload = {"quantitation_plane": "exact_recorded_single_z", "projection_used": False, "same_z_all_channels": True, "display_only": True, "display_windows": PRIMARY_DISPLAY_WINDOWS, "panels": list(_PANELS), "panel_channel_order": ["dapi", "miat", "qki", "merge", "q95_overlay"], "q95_overlay_classes": list(_Q95_CLASSES), "q95_definition": _Q95_DEFINITION, "selection_manifest": json.loads(manifest.to_json(orient="records")), "selection_rule": {"field": "condition_relative_median_mad_centrality", "nucleus": "field_relative_median_mad_centrality", "zero_mad_policy": "median_matches_zero_finite_deviations_infinite"}, "input_hashes": {"selected_planes_h5": {"mode": "file_bytes_sha256", "sha256": _sha256_file(selected_planes_h5)}, "image_audit": image_hash, "spot_calls": spots_hash, "nucleus_qc": nuclei_hash, "footprint_pixels": pixels_hash}, "files": {"individual_png": [path.as_posix() for path in png_rel], "individual_svg": [path.as_posix() for path in svg_rel], "contact_sheets": [path.as_posix() for path in contact_rel], "pdf": []}}
        (temporary / manifest_rel).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    pngs = tuple(output / path for path in png_rel)
    svgs = tuple(output / path for path in svg_rel)
    contacts = tuple(output / path for path in contact_rel)
    audit_path = output / audit_rel
    source_path = output / source_rel
    manifest_path = output / manifest_rel
    return PublicationMicrographOutputs(output, pngs, svgs, contacts, audit_path, source_path, manifest_path)
