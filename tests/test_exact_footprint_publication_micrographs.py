"""Contracts for final-publication exact-single-z micrographs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
from PIL import Image


def _selection_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    images: list[dict[str, object]] = []
    nuclei: list[dict[str, object]] = []
    spots: list[dict[str, object]] = []
    for arm in ("NT", "KD"):
        for number in range(1, 7):
            key = f"{arm.lower()}_set_{number}.vsi"
            images.append({"image_key": key, "image": key, "slide": 1 if number <= 3 else 2, "arm": arm, "biological_set": f"{arm}_set_{number}", "is_control": False, "secondary_only": False, "selected_z_1based": 9, "voxel_xy_nm": 1000.0, "miat_channel_index": 0, "qki_channel_index": 1, "dapi_channel_index": 2, "plane_lock_pass": True, "mask_qc_pass": True, "footprint_parity_pass": True, "population_reconciliation_pass": True, "image_qc_status": "pass"})
            nuclei.append({"image_key": key, "nucleus_id": 1, "nucleus_uid": f"{key}:nucleus:1", "nucleus_qc_status": "pass"})
            for spot in range(4):
                spots.append({"image_key": key, "nucleus_id": 1, "spot_uid": f"{key}:spot:{spot}", "null_usable": spot < 3, "qki_threshold_positive_q95": spot == 0, "stored_in_nucleolus": False, "qki_footprint_enrichment_vs_nucleoplasm": float(number)})
    images.extend([dict(images[0], image_key="nt_control.vsi", is_control=True), dict(images[1], image_key="nt_failed_parity.vsi", footprint_parity_pass=False)])
    return pd.DataFrame(images), pd.DataFrame(spots), pd.DataFrame(nuclei)


def test_condition_relative_selection_is_three_per_arm_distinct_and_auditable():
    """Catches extreme-effect, slide-relative, or QC/control selection drift."""
    from fishsuite.core.exact_footprint_publication_micrographs import select_publication_micrograph_examples
    images, spots, nuclei = _selection_tables()
    selected = select_publication_micrograph_examples(images, spots, nuclei)
    assert selected.manifest.groupby("condition").size().to_dict() == {"NT": 3, "MIAT-KD": 3}
    assert selected.manifest.groupby("condition")["biological_set"].nunique().to_dict() == {"NT": 3, "MIAT-KD": 3}
    audit = selected.field_audit.set_index("image_key")
    assert not bool(audit.loc["nt_control.vsi", "eligible"])
    assert not bool(audit.loc["nt_failed_parity.vsi", "eligible"])


def test_omission_control_is_excluded_before_condition_normalization():
    """Catches a valid OMISSION arm raising before its control flag is applied."""
    from fishsuite.core.exact_footprint_publication_micrographs import (
        select_publication_micrograph_examples,
    )

    images, spots, nuclei = _selection_tables()
    omission = dict(images.iloc[0])
    omission.update(
        {
            "image_key": "dual_omission.vsi",
            "image": "dual_omission.vsi",
            "arm": "OMISSION",
            "biological_set": "",
            "is_control": True,
            "secondary_only": True,
        }
    )
    selected = select_publication_micrograph_examples(
        pd.concat([images, pd.DataFrame([omission])], ignore_index=True),
        spots,
        nuclei,
    )

    row = selected.field_audit.set_index("image_key").loc["dual_omission.vsi"]
    assert row["condition"] == "CONTROL"
    assert not bool(row["eligible"])


def test_zero_mad_penalizes_finite_outlier_and_selects_exact_central_keys():
    """Catches zero-MAD outliers tying the true condition median at distance zero."""
    from fishsuite.core.exact_footprint_publication_micrographs import (
        select_publication_micrograph_examples,
    )

    images, spots, nuclei = _selection_tables()
    nt = spots["image_key"].str.startswith("nt_set_")
    spots.loc[nt, "qki_footprint_enrichment_vs_nucleoplasm"] = 1.0
    spots.loc[spots["image_key"].eq("nt_set_1.vsi"), "qki_footprint_enrichment_vs_nucleoplasm"] = 100.0

    selected = select_publication_micrograph_examples(images, spots, nuclei)

    nt_keys = selected.manifest.loc[
        selected.manifest["condition"].eq("NT"), "image_key"
    ].tolist()
    assert nt_keys == ["nt_set_2.vsi", "nt_set_3.vsi", "nt_set_4.vsi"]
    audit = selected.field_audit.set_index("image_key")
    assert audit.loc["nt_set_2.vsi", "condition_distance_median_qki_enrichment_vs_nucleoplasm"] == 0.0
    assert np.isinf(
        audit.loc[
            "nt_set_1.vsi",
            "condition_distance_median_qki_enrichment_vs_nucleoplasm",
        ]
    )


def test_selection_uses_spots_only_from_the_qc_passing_nucleus_roster():
    """Catches failed-nucleus spots inflating q95 eligibility or centrality."""
    from fishsuite.core.exact_footprint_publication_micrographs import select_publication_micrograph_examples
    images, spots, nuclei = _selection_tables()
    nuclei = pd.concat([nuclei, pd.DataFrame([{"image_key": "nt_set_3.vsi", "nucleus_id": 99, "nucleus_uid": "nt_set_3.vsi:nucleus:99", "nucleus_qc_status": "failed"}])], ignore_index=True)
    failed = pd.DataFrame([{"image_key": "nt_set_3.vsi", "nucleus_id": 99, "spot_uid": f"nt_set_3.vsi:failed:{n}", "null_usable": True, "qki_threshold_positive_q95": True, "stored_in_nucleolus": False, "qki_footprint_enrichment_vs_nucleoplasm": 100.0} for n in range(10)])
    row = select_publication_micrograph_examples(images, pd.concat([spots, failed], ignore_index=True), nuclei).field_audit.set_index("image_key").loc["nt_set_3.vsi"]
    assert row["n_qc_passing_nuclei"] == 1
    assert row["n_usable_q95"] == 3
    assert row["association_fraction_q95"] == pytest.approx(1 / 3)
    assert row["median_qki_enrichment_vs_nucleoplasm"] == 3.0


def _render_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    images, spots, nuclei = _selection_tables()
    spots = spots.copy()
    spots["qki_threshold_positive_q95"] = False
    spots["center_y_px"] = 15 + spots["spot_uid"].str[-1].astype(int)
    spots["center_x_px"] = 15 + spots["spot_uid"].str[-1].astype(int)
    spots["footprint_method"] = "half_max_component"
    extra_nuclei: list[dict[str, object]] = []
    extra_spots: list[dict[str, object]] = []
    for key in images.loc[images["image_key"].str.contains("_set_"), "image_key"]:
        for nucleus_id in (2, 3):
            extra_nuclei.append(
                {
                    "image_key": key,
                    "nucleus_id": nucleus_id,
                    "nucleus_uid": f"{key}:nucleus:{nucleus_id}",
                    "nucleus_qc_status": "pass",
                }
            )
            for spot_number in range(4):
                center = 34 + spot_number if nucleus_id == 2 else 51 + spot_number
                extra_spots.append(
                    {
                        "image_key": key,
                        "nucleus_id": nucleus_id,
                        "spot_uid": f"{key}:nucleus:{nucleus_id}:spot:{spot_number}",
                        "null_usable": True,
                        "qki_threshold_positive_q95": (
                            spot_number < (2 if nucleus_id == 2 else 4)
                        ),
                        "stored_in_nucleolus": False,
                        "qki_footprint_enrichment_vs_nucleoplasm": float(nucleus_id),
                        "center_y_px": center,
                        "center_x_px": center,
                        "footprint_method": "half_max_component",
                    }
                )
    nuclei = pd.concat([nuclei, pd.DataFrame(extra_nuclei)], ignore_index=True)
    spots = pd.concat([spots, pd.DataFrame(extra_spots)], ignore_index=True)
    pixels = pd.DataFrame({"spot_uid": spots["spot_uid"], "pixel_index": 0, "y_px": spots["center_y_px"], "x_px": spots["center_x_px"]})
    return images, spots, nuclei, pixels


def _write_selected_planes(path: Path, keys: list[str]) -> None:
    with h5py.File(path, "w") as handle:
        root = handle.create_group("images")
        for key in keys:
            group = root.create_group(key)
            group.attrs.update({"image_key": key, "selected_z_1based": 9, "selected_z_0based": 8, "miat_channel_index": 0, "qki_channel_index": 1, "dapi_channel_index": 2, "same_plane_all_channels": True, "complete": True})
            base = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
            for role, offset in (("miat", 400), ("qki", 555), ("dapi", 334)):
                group.create_dataset(role, data=base + offset)
            labels = np.zeros((64, 64), dtype=np.int32)
            labels[10:25, 10:25] = 1
            labels[30:45, 30:45] = 2
            labels[48:62, 48:62] = 3
            group.create_dataset("nucleus_labels", data=labels)
            group.create_dataset("nucleolus_labels", data=np.zeros_like(labels))


def _render(tmp_path: Path, output_name: str, *, inputs: tuple[object, object, object, object] | None = None):
    from fishsuite.core.exact_footprint_publication_micrographs import render_publication_micrograph_package
    images, spots, nuclei, pixels = _render_tables()
    h5_path = tmp_path / "selected_planes.h5"
    _write_selected_planes(h5_path, images.loc[images["image_key"].str.contains("_set_"), "image_key"].tolist())
    return render_publication_micrograph_package(h5_path, *(inputs or (images, spots, nuclei, pixels)), tmp_path / output_name), h5_path, (images, spots, nuclei, pixels)


def test_renderer_outputs_exact_plane_panels_and_clean_visible_labels(tmp_path: Path):
    """Catches projections, non-final formats, omitted overlay classes, or leaked IDs."""
    outputs, _, _ = _render(tmp_path, "micrographs")
    assert len(outputs.individual_png_paths) == len(outputs.individual_svg_paths) == 6
    assert {path.suffix for path in (*outputs.individual_png_paths, *outputs.contact_sheet_paths)} == {".png", ".svg"}
    assert not list(outputs.output_dir.rglob("*.pdf"))
    with Image.open(outputs.individual_png_paths[0]) as image:
        assert image.info["dpi"][0] == pytest.approx(600, rel=0.01)
    manifest = json.loads(outputs.manifest_path.read_text(encoding="utf-8"))
    assert manifest["quantitation_plane"] == "exact_recorded_single_z"
    assert manifest["projection_used"] is False
    assert manifest["panels"] == ["DAPI", "MIAT", "QKI", "merge", "q95 overlay"]
    assert manifest["panel_channel_order"] == ["dapi", "miat", "qki", "merge", "q95_overlay"]
    assert manifest["q95_overlay_classes"] == [
        "threshold_positive",
        "threshold_negative",
        "unusable",
    ]
    assert {row["nucleus_id"] for row in manifest["selection_manifest"]} == {2}
    required_provenance = {
        "nucleus_id",
        "nucleus_uid",
        "crop_y0",
        "crop_y1",
        "crop_x0",
        "crop_x1",
        "requested_margin_um",
        "requested_margin_px",
        "crop_clamped",
        "crop_height_px",
        "crop_width_px",
        "n_scoped_spots",
        "n_null_usable",
        "n_q95_positive",
        "n_q95_negative",
        "n_q95_unusable",
    }
    assert all(required_provenance <= set(row) for row in manifest["selection_manifest"])
    assert all(row["crop_y0"] == 28 and row["crop_y1"] == 47 for row in manifest["selection_manifest"])
    assert all(row["crop_x0"] == 28 and row["crop_x1"] == 47 for row in manifest["selection_manifest"])
    assert all(row["requested_margin_um"] == 2.0 for row in manifest["selection_manifest"])
    assert all(row["requested_margin_px"] == 2 for row in manifest["selection_manifest"])
    assert all(row["crop_height_px"] == row["crop_width_px"] == 19 for row in manifest["selection_manifest"])
    assert all(row["n_scoped_spots"] == 4 for row in manifest["selection_manifest"])
    assert all(row["n_null_usable"] == 4 for row in manifest["selection_manifest"])
    assert all(row["n_q95_positive"] == row["n_q95_negative"] == 2 for row in manifest["selection_manifest"])
    for group in ("individual_png", "individual_svg", "contact_sheets"):
        for relative in manifest["files"][group]:
            assert "/" in relative
            assert (outputs.manifest_path.parent / relative).is_file()
    for svg_path in (*outputs.individual_svg_paths, *[path for path in outputs.contact_sheet_paths if path.suffix == ".svg"]):
        svg = svg_path.read_text(encoding="utf-8")
        assert "q95: QKI greater than 95%" in svg
        for forbidden in ("Slide 1", "Slide 2", "NT_set_", "KD_set_", "biological_set"):
            assert forbidden not in svg

    source = pd.read_csv(outputs.source_data_path)
    audit = pd.read_csv(outputs.selection_audit_path)
    assert required_provenance <= set(source.columns)
    assert required_provenance <= set(audit.columns)
    assert source["nucleus_id"].eq(2).all()
    assert audit.loc[audit["selected"].astype(bool), "nucleus_id"].eq(2).all()


def test_renderer_fails_closed_on_audit_to_hdf_channel_index_disagreement(tmp_path: Path):
    """Catches a swapped cached HDF5 channel being silently rendered as QKI."""
    from fishsuite.core.exact_footprint_publication_micrographs import (
        render_publication_micrograph_package,
        select_publication_micrograph_examples,
    )
    images, spots, nuclei, pixels = _render_tables()
    h5_path = tmp_path / "swapped.h5"
    _write_selected_planes(h5_path, images.loc[images["image_key"].str.contains("_set_"), "image_key"].tolist())
    selected = select_publication_micrograph_examples(images, spots, nuclei)
    late_key = str(selected.manifest.iloc[-1]["image_key"])
    with h5py.File(h5_path, "r+") as handle:
        handle["images"][late_key].attrs["qki_channel_index"] = 0
    output = tmp_path / "swapped_micrographs"
    with pytest.raises(ValueError, match="channel index mismatch"):
        render_publication_micrograph_package(h5_path, images, spots, nuclei, pixels, output)
    assert not output.exists()


def test_renderer_records_deterministic_hashes_for_all_in_memory_inputs(tmp_path: Path):
    """Catches missing provenance hashes for tables that drive selection and overlays."""
    outputs, _, _ = _render(tmp_path, "hash_micrographs")
    hashes = json.loads(outputs.manifest_path.read_text(encoding="utf-8"))["input_hashes"]
    assert set(hashes) == {"selected_planes_h5", "image_audit", "spot_calls", "nucleus_qc", "footprint_pixels"}
    assert hashes["selected_planes_h5"]["mode"] == "file_bytes_sha256"
    for name in ("image_audit", "spot_calls", "nucleus_qc", "footprint_pixels"):
        assert hashes[name]["mode"] == "deterministic_dataframe_csv"
        assert len(hashes[name]["sha256"]) == 64


def test_renderer_uses_canonical_file_bytes_when_table_paths_are_supplied(tmp_path: Path):
    """Catches replacing source-table hashes with an in-memory reserialization."""
    images, spots, nuclei, pixels = _render_tables()
    paths: list[Path] = []
    for name, frame in (("image_audit", images), ("spot_calls", spots), ("nucleus_qc", nuclei), ("footprint_pixels", pixels)):
        path = tmp_path / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths.append(path)
    outputs, _, _ = _render(tmp_path, "file_hash_micrographs", inputs=tuple(paths))
    hashes = json.loads(outputs.manifest_path.read_text(encoding="utf-8"))["input_hashes"]
    for name, path in zip(("image_audit", "spot_calls", "nucleus_qc", "footprint_pixels"), paths, strict=True):
        assert hashes[name] == {"mode": "file_bytes_sha256", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
