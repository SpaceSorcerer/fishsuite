"""Tests for exact-plane MIAT x QKI representative figure rendering."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest


def test_display_window_returns_copy_without_touching_raw_plane():
    """Catches a renderer that clips or rescales the quantitative input in place."""
    from fishsuite.core.exact_footprint_figures import display_channel

    raw = np.array([[0, 334, 2917, 5500, 6000]], dtype=np.uint16)
    before = raw.copy()

    rendered = display_channel(raw, 334, 5500)
    alternate = display_channel(raw, 334, 2419)

    assert rendered.dtype == np.uint8
    assert rendered.tolist() == [[0, 0, 128, 255, 255]]
    assert not np.array_equal(rendered, alternate)
    np.testing.assert_array_equal(raw, before)


def test_clean_grid_row_label_formats_integral_slides_and_kd_arm():
    """Catches publication labels that expose float slides or shorthand KD."""
    from fishsuite.core.exact_footprint_figures import (
        format_clean_grid_row_label,
    )

    assert (
        format_clean_grid_row_label(1.0, "NT", "S1_NT_1")
        == "Slide 1 — NT\nS1_NT_1"
    )
    assert (
        format_clean_grid_row_label(2.0, "KD", "S2_KD_3")
        == "Slide 2 — MIAT-KD\nS2_KD_3"
    )


def test_nucleus_crop_keeps_complete_edge_nucleus_and_stays_square():
    """Catches crop math that clips an edge nucleus or applies margin asymmetrically."""
    from fishsuite.core.exact_footprint_figures import nucleus_square_crop

    labels = np.zeros((10, 12), dtype=np.int32)
    labels[1:4, 0:2] = 7

    crop = nucleus_square_crop(labels, nucleus_id=7, voxel_xy_nm=1000.0)

    assert (crop.y0, crop.y1, crop.x0, crop.x1) == (0, 7, 0, 7)
    assert crop.margin_px == 2
    assert crop.is_square
    assert crop.was_clamped
    assert crop.width_um == crop.height_um == 7.0
    ys, xs = np.nonzero(labels == 7)
    assert crop.y0 <= int(ys.min()) and crop.y1 > int(ys.max())
    assert crop.x0 <= int(xs.min()) and crop.x1 > int(xs.max())


def _normalized_segments(segments):
    return {
        tuple(sorted((tuple(map(float, segment[0])), tuple(map(float, segment[1])))))
        for segment in np.asarray(segments)
    }


def test_overlay_uses_exact_footprint_edges_and_keeps_centers_separate():
    """Catches substitution of a disk/ellipse for the quantified footprint."""
    from fishsuite.core.exact_footprint_figures import build_call_overlay_specs

    spots = pd.DataFrame(
        [
            {
                "spot_uid": "s-pos",
                "center_x_px": 1,
                "center_y_px": 1,
                "null_usable": True,
                "qki_threshold_positive_q95": True,
                "footprint_method": "half_max_component",
                "spot_diameter_px": 99.0,
            },
            {
                "spot_uid": "s-neg",
                "center_x_px": 5,
                "center_y_px": 5,
                "null_usable": True,
                "qki_threshold_positive_q95": False,
                "footprint_method": "half_max_component",
                "spot_diameter_px": 0.1,
            },
            {
                "spot_uid": "s-unusable",
                "center_x_px": 8,
                "center_y_px": 8,
                "null_usable": False,
                "qki_threshold_positive_q95": pd.NA,
                "footprint_method": "fallback_single_pixel",
                "spot_diameter_px": 50.0,
            },
        ]
    )
    pixels = pd.DataFrame(
        [
            {"spot_uid": "s-pos", "y_px": 1, "x_px": 1},
            {"spot_uid": "s-pos", "y_px": 1, "x_px": 2},
            {"spot_uid": "s-pos", "y_px": 2, "x_px": 1},
            {"spot_uid": "s-neg", "y_px": 5, "x_px": 5},
            {"spot_uid": "s-unusable", "y_px": 8, "x_px": 8},
        ]
    )

    specs = build_call_overlay_specs(spots, pixels, percentile=95)

    assert [spec.spot_uid for spec in specs] == ["s-neg", "s-pos", "s-unusable"]
    assert [spec.call_class for spec in specs] == [
        "threshold_negative",
        "threshold_positive",
        "unusable",
    ]
    assert [spec.line_style for spec in specs] == ["--", "-", ":"]
    assert [spec.color for spec in specs] == ["#0072B2", "#D55E00", "#7F7F7F"]
    assert specs[-1].draw_center_x
    assert specs[1].center_xy == (1.0, 1.0)
    # Hand-derived perimeter of the three-pixel L; diameter never enters.
    assert _normalized_segments(specs[1].boundary_segments_xy) == {
        ((0.5, 0.5), (1.5, 0.5)),
        ((0.5, 0.5), (0.5, 1.5)),
        ((1.5, 0.5), (2.5, 0.5)),
        ((2.5, 0.5), (2.5, 1.5)),
        ((1.5, 1.5), (2.5, 1.5)),
        ((0.5, 1.5), (0.5, 2.5)),
        ((0.5, 2.5), (1.5, 2.5)),
        ((1.5, 1.5), (1.5, 2.5)),
    }


def test_representative_selection_is_deterministic_and_uses_direct_fov_ratio():
    """Catches row-order/cherry-pick drift and averaging nucleus fractions first."""
    from fishsuite.core.exact_footprint_figures import select_representatives

    image_audit = pd.DataFrame(
        [
            {
                "image_key": key,
                "image": key,
                "slide": "S1",
                "arm": "NT",
                "biological_set": key[0],
                "replicate": key[0],
                "fov": 1,
                "is_control": False,
                "secondary_only": False,
                "selected_z_1based": 7,
                "voxel_xy_nm": 100.0,
                "plane_lock_pass": True,
                "mask_qc_pass": True,
                "footprint_parity_pass": True,
                "population_reconciliation_pass": True,
                "image_qc_status": "pass",
            }
            for key in ("b.vsi", "A.vsi")
        ]
    )
    nuclei = pd.DataFrame(
        [
            {
                "image_key": key,
                "nucleus_uid": f"{key}:nucleus:{nucleus_id}",
                "nucleus_id": nucleus_id,
                "nucleus_qc_status": "pass",
            }
            for key in ("b.vsi", "A.vsi")
            for nucleus_id in (1, 2)
        ]
    )
    spots = pd.DataFrame(
        [
            {
                "image_key": key,
                "nucleus_id": 1,
                "spot_uid": f"{key}:spot:{spot_id}",
                "null_usable": True,
                "qki_threshold_positive_q95": spot_id < 3,
                "stored_in_nucleolus": False,
                "qki_footprint_enrichment_vs_nucleoplasm": 1.0,
            }
            for key in ("b.vsi", "A.vsi")
            for spot_id in range(4)
        ]
    )

    first = select_representatives(image_audit, spots, nuclei)
    second = select_representatives(
        image_audit.sample(frac=1.0, random_state=2),
        spots.sample(frac=1.0, random_state=3),
        nuclei.sample(frac=1.0, random_state=4),
    )

    assert first.manifest["image_key"].tolist() == ["a.vsi"]
    pd.testing.assert_frame_equal(first.manifest, second.manifest)
    selected_field = first.field_audit.loc[first.field_audit["selected"]].iloc[0]
    assert selected_field["association_fraction_q95"] == 0.75
    assert selected_field["median_non_nucleolar_spots_per_nucleus"] == 2.0
    selected_nucleus_audit = first.nucleus_audit.loc[
        first.nucleus_audit["image_key"].eq("a.vsi")
    ]
    assert len(selected_nucleus_audit) == 2
    assert selected_nucleus_audit.loc[
        selected_nucleus_audit["selected"], "nucleus_id"
    ].tolist() == [1]


def test_primary_dapi_render_uses_global_334_to_8000_window():
    """Catches regression to the visually clipped 5500 DAPI ceiling."""
    from fishsuite.core.exact_footprint_figures import render_channel_rgb

    raw = np.array([[334, 5500, 8000]], dtype=np.uint16)
    rendered = render_channel_rgb(raw, "dapi")

    assert rendered.shape == (1, 3, 3)
    assert rendered[0, 0].tolist() == [0, 0, 0]
    assert 0 < int(rendered[0, 1, 2]) < 255
    assert rendered[0, 2].tolist() == [0, 0, 255]


def test_selected_plane_loader_fails_on_z_or_channel_mismatch(tmp_path):
    """Catches accidental use of a different z/channel or any stack-like input."""
    import h5py

    from fishsuite.core.exact_footprint_figures import load_selected_plane

    path = tmp_path / "selected_planes_and_masks.h5"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("images/field.vsi")
        group.attrs.update(
            {
                "image_key": "field.vsi",
                "selected_z_1based": 9,
                "selected_z_0based": 8,
                "miat_channel_index": 0,
                "qki_channel_index": 1,
                "dapi_channel_index": 2,
                "same_plane_all_channels": True,
                "complete": True,
            }
        )
        for role, value in (("miat", 11), ("qki", 22), ("dapi", 33)):
            group.create_dataset(
                role, data=np.full((5, 6), value, dtype=np.uint16)
            )
        group.create_dataset(
            "nucleus_labels", data=np.ones((5, 6), dtype=np.int32)
        )
        group.create_dataset(
            "nucleolus_labels", data=np.zeros((5, 6), dtype=np.int32)
        )

    loaded = load_selected_plane(
        path,
        "FIELD.VSI",
        expected_z_1based=9,
        expected_channel_indices={"miat": 0, "qki": 1, "dapi": 2},
    )

    assert loaded.selected_z_1based == 9
    assert loaded.planes["qki"].shape == (5, 6)
    assert loaded.planes["qki"].dtype == np.uint16
    with pytest.raises(ValueError, match="selected z"):
        load_selected_plane(path, "field.vsi", expected_z_1based=8)
    with pytest.raises(ValueError, match="channel index"):
        load_selected_plane(
            path,
            "field.vsi",
            expected_z_1based=9,
            expected_channel_indices={"miat": 1, "qki": 0, "dapi": 2},
        )


def test_package_renderer_writes_exact_plane_publication_products(tmp_path):
    """Catches missing raw/vector products or sidecars that obscure plane provenance."""
    import h5py
    import tifffile

    from fishsuite.core.exact_footprint_figures import (
        render_exact_footprint_package,
    )

    cells = [
        ("S1", "NT", "s1_nt.vsi"),
        ("S1", "KD", "s1_kd.vsi"),
        ("S2", "NT", "s2_nt.vsi"),
        ("S2", "KD", "s2_kd.vsi"),
    ]
    image_audit = pd.DataFrame(
        [
            {
                "image_key": key,
                "image": key,
                "slide": slide,
                "arm": arm,
                "biological_set": f"{slide}_{arm}_1",
                "replicate": 1,
                "fov": 1,
                "is_control": False,
                "secondary_only": False,
                "selected_z_1based": 9,
                "voxel_xy_nm": 1000.0,
                "miat_channel_index": 0,
                "qki_channel_index": 1,
                "dapi_channel_index": 2,
                "plane_lock_pass": True,
                "mask_qc_pass": True,
                "footprint_parity_pass": True,
                "population_reconciliation_pass": True,
                "image_qc_status": "pass",
            }
            for slide, arm, key in cells
        ]
    )
    nuclei = pd.DataFrame(
        [
            {
                "image_key": key,
                "nucleus_uid": f"{key}:nucleus:1",
                "nucleus_id": 1,
                "nucleus_qc_status": "pass",
            }
            for _, _, key in cells
        ]
    )
    spot_rows = []
    pixel_rows = []
    for _, _, key in cells:
        for spot_id, positive, y, x in (
            (0, True, 30, 30),
            (1, False, 34, 34),
        ):
            uid = f"{key}:rna1:{spot_id}"
            spot_rows.append(
                {
                    "image_key": key,
                    "nucleus_id": 1,
                    "spot_uid": uid,
                    "selected_z_1based": 9,
                    "selected_z_0based": 8,
                    "miat_channel_index": 0,
                    "qki_channel_index": 1,
                    "dapi_channel_index": 2,
                    "quantitation_plane": "exact_recorded_single_z",
                    "center_x_px": x,
                    "center_y_px": y,
                    "null_usable": True,
                    "qki_threshold_positive_q90": positive,
                    "qki_threshold_positive_q95": positive,
                    "qki_threshold_positive_q99": False,
                    "population_label_q90": (
                        "threshold_positive" if positive else "threshold_negative"
                    ),
                    "population_label_q95": (
                        "threshold_positive" if positive else "threshold_negative"
                    ),
                    "population_label_q99": "threshold_negative",
                    "stored_in_nucleolus": False,
                    "qki_footprint_enrichment_vs_nucleoplasm": (
                        1.2 if positive else 0.9
                    ),
                    "qki_footprint_mean_raw": 1800.0 if positive else 900.0,
                    "null_q05_raw": 700.0,
                    "null_q90_raw": 1000.0,
                    "null_q95_raw": 1100.0,
                    "null_q99_raw": 1900.0,
                    "footprint_method": "half_max_component",
                }
            )
            pixel_rows.append(
                {"spot_uid": uid, "pixel_index": 0, "y_px": y, "x_px": x}
            )
    spots = pd.DataFrame(spot_rows)
    pixels = pd.DataFrame(pixel_rows)
    h5_path = tmp_path / "selected_planes_and_masks.h5"
    with h5py.File(h5_path, "w") as handle:
        root = handle.create_group("images")
        for index, (_, _, key) in enumerate(cells):
            group = root.create_group(key)
            group.attrs.update(
                {
                    "image_key": key,
                    "selected_z_1based": 9,
                    "selected_z_0based": 8,
                    "miat_channel_index": 0,
                    "qki_channel_index": 1,
                    "dapi_channel_index": 2,
                    "same_plane_all_channels": True,
                    "complete": True,
                }
            )
            base = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
            group.create_dataset("miat", data=base + index)
            group.create_dataset("qki", data=base + 555)
            group.create_dataset("dapi", data=base + 334)
            labels = np.zeros((64, 64), dtype=np.int32)
            labels[24:41, 24:41] = 1
            group.create_dataset("nucleus_labels", data=labels)
            group.create_dataset(
                "nucleolus_labels", data=np.zeros_like(labels)
            )

    output_dir = tmp_path / "figure_package"
    outputs = render_exact_footprint_package(
        h5_path,
        spots,
        pixels,
        nuclei,
        image_audit,
        output_dir,
        dpi=72,
    )

    assert {path.suffix for path in outputs.clean_grid_paths} == {
        ".png",
        ".pdf",
        ".svg",
    }
    assert len(outputs.walkthrough_paths) == 12
    assert all(path.is_file() for path in outputs.walkthrough_paths)
    assert (output_dir / "representative_field_selection_audit.csv").is_file()
    assert (output_dir / "representative_nucleus_selection_audit.csv").is_file()
    assert (output_dir / "representative_selection_manifest.csv").is_file()
    assert (output_dir / "threshold_sensitivity_summary.csv").is_file()
    raw_path = (
        output_dir
        / "fields"
        / "s1_nt.vsi"
        / "full"
        / "dapi_raw_uint16.tif"
    )
    raw = tifffile.imread(raw_path)
    assert raw.dtype == np.uint16
    assert int(raw[0, 0]) == 334
    sidecar = json.loads(
        (
            output_dir
            / "fields"
            / "s1_nt.vsi"
            / "nucleus_1_mixed"
            / "provenance.json"
        ).read_text(encoding="utf-8")
    )
    assert sidecar["display_only"] is True
    assert sidecar["display_windows"]["dapi"] == [334.0, 8000.0]
    assert sidecar["quantitation"]["projection"] == "none"
    assert sidecar["quantitation"]["same_z_all_channels"] is True
    assert sidecar["selected_z_1based"] == 9


def test_control_hooks_render_zero_example_and_anomaly_without_biology(tmp_path):
    """Catches omission controls being dropped or mislabeled as biological evidence."""
    import h5py

    from fishsuite.core.exact_footprint_figures import (
        ControlPanelHooks,
        render_control_diagnostics,
    )

    image_audit = pd.DataFrame(
        [
            {
                "image_key": key,
                "image": key,
                "slide": "S2",
                "arm": "secondary_only",
                "is_control": True,
                "secondary_only": True,
                "selected_z_1based": 7,
                "voxel_xy_nm": 1000.0,
                "mask_qc_pass": True,
                "image_qc_status": (
                    "failed" if key == "kd42.vsi" else "pass"
                ),
                "qc_flags": "zero_spot" if key == "kd42.vsi" else "",
            }
            for key in ("kd42.vsi", "nt21.vsi")
        ]
    )
    spots = pd.DataFrame(
        [
            {
                "image_key": "nt21.vsi",
                "spot_uid": f"nt21.vsi:rna1:{spot_id}",
                "nucleus_id": nucleus_id,
                "center_x_px": x,
                "center_y_px": y,
                "null_usable": True,
                "qki_threshold_positive_q95": spot_id == 0,
                "footprint_method": "half_max_component",
            }
            for spot_id, nucleus_id, y, x in (
                (0, 35, 30, 30),
                (1, 30, 45, 45),
            )
        ]
    )
    pixels = pd.DataFrame(
        [
            {
                "spot_uid": row["spot_uid"],
                "pixel_index": 0,
                "y_px": row["center_y_px"],
                "x_px": row["center_x_px"],
            }
            for row in spots.to_dict("records")
        ]
    )
    h5_path = tmp_path / "controls.h5"
    with h5py.File(h5_path, "w") as handle:
        root = handle.create_group("images")
        for key in ("kd42.vsi", "nt21.vsi"):
            group = root.create_group(key)
            group.attrs.update(
                {
                    "image_key": key,
                    "selected_z_1based": 7,
                    "selected_z_0based": 6,
                    "miat_channel_index": 0,
                    "qki_channel_index": 1,
                    "dapi_channel_index": 2,
                    "same_plane_all_channels": True,
                    "complete": True,
                }
            )
            base = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
            group.create_dataset("miat", data=base)
            group.create_dataset("qki", data=base + 555)
            group.create_dataset("dapi", data=base + 334)
            labels = np.zeros((64, 64), dtype=np.int32)
            labels[24:38, 24:38] = 35
            labels[40:54, 40:54] = 30
            group.create_dataset("nucleus_labels", data=labels)
            group.create_dataset(
                "nucleolus_labels", data=np.zeros_like(labels)
            )

    outputs = render_control_diagnostics(
        h5_path,
        spots,
        pixels,
        image_audit,
        tmp_path / "control_diagnostics",
        hooks=ControlPanelHooks(
            zero_control_image_key="kd42.vsi",
            anomaly_image_key="nt21.vsi",
            anomaly_nucleus_ids=(35, 30),
        ),
        dpi=72,
    )

    ledger = pd.read_csv(outputs.ledger_path)
    assert ledger["image_key"].tolist() == ["kd42.vsi", "nt21.vsi"]
    assert ledger.set_index("image_key").loc["kd42.vsi", "miat_call_count"] == 0
    assert ledger.set_index("image_key").loc["nt21.vsi", "miat_call_count"] == 2
    assert (
        ledger.set_index("image_key").loc["kd42.vsi", "mask_qc_status"]
        == "pass"
    )
    assert (
        ledger.set_index("image_key").loc[
            "kd42.vsi", "control_signal_qc_status"
        ]
        == "expected_negative_zero_spot"
    )
    assert (
        ledger.set_index("image_key").loc[
            "kd42.vsi", "recorded_image_qc_status"
        ]
        == "failed"
    )
    assert all(path.is_file() for path in outputs.ledger_figure_paths)
    assert all(path.is_file() for path in outputs.zero_example_paths)
    assert all(path.is_file() for path in outputs.anomaly_paths)
    scope = json.loads(outputs.sidecar_path.read_text(encoding="utf-8"))
    assert scope["control_scope"] == "combined MIAT-probe and QKI-primary omission"
    assert scope["excluded_from_biological_selection_and_inference"] is True


def test_prepare_renderer_audits_uses_only_recorded_qc_and_parity_evidence():
    """Catches adapters that silently fill missing QC/reconciliation as passing."""
    from fishsuite.core.exact_footprint_figures import prepare_renderer_audits

    manifest = pd.DataFrame(
        [
            {
                "image_key": key,
                "image": key,
                "slide": 1,
                "arm": "NT",
                "biological_set": f"NT_{index}",
                "replicate": index,
                "fov": 1,
                "is_control": False,
                "secondary_only": False,
                "selected_z_1based": 9,
                "voxel_xy_nm": 100.0,
                "miat_channel_index": 0,
                "qki_channel_index": 1,
                "dapi_channel_index": 2,
                "plane_lock_pass": True,
                "load_status": "complete",
                "mask_status": "loaded_reused",
                "qc_pass": index == 1,
            }
            for index, key in ((1, "a.vsi"), (2, "b.vsi"))
        ]
    )
    nucleus_metrics = pd.DataFrame(
        [
            {
                "image_key": "a.vsi",
                "nucleus_id": 1,
                "nucleus_uid": "a.vsi:nucleus:1",
                "include": True,
                "exclude_reason": "",
            },
            {
                "image_key": "b.vsi",
                "nucleus_id": 2,
                "nucleus_uid": "b.vsi:nucleus:2",
                "include": False,
                "exclude_reason": "recorded_qc_exclusion",
            },
        ]
    )
    reconciliation = pd.DataFrame(
        [
            {
                "level": "image",
                "key": key,
                "reconciliation_pass_q90": True,
                "reconciliation_pass_q95": True,
                "reconciliation_pass_q99": key == "a.vsi",
            }
            for key in ("a.vsi", "b.vsi")
        ]
    )
    parity = {
        "pass": True,
        "area_exact_all_images": True,
        "qki_allclose_all_images": True,
        "n_images": 2,
        "selected_image_keys": ["a.vsi", "b.vsi"],
    }

    prepared = prepare_renderer_audits(
        manifest, nucleus_metrics, reconciliation, parity
    )

    image_qc = prepared.image_audit.set_index("image_key")
    assert bool(image_qc.loc["a.vsi", "footprint_parity_pass"])
    assert bool(image_qc.loc["a.vsi", "population_reconciliation_pass"])
    assert not bool(image_qc.loc["b.vsi", "population_reconciliation_pass"])
    assert image_qc.loc["a.vsi", "image_qc_status"] == "pass"
    assert image_qc.loc["b.vsi", "image_qc_status"] == "failed"
    nucleus_qc = prepared.nucleus_qc_roster.set_index("nucleus_uid")
    assert nucleus_qc.loc["a.vsi:nucleus:1", "nucleus_qc_status"] == "pass"
    assert nucleus_qc.loc["b.vsi:nucleus:2", "nucleus_qc_status"] == "failed"
    assert (
        nucleus_qc.loc["b.vsi:nucleus:2", "nucleus_qc_reason"]
        == "recorded_qc_exclusion"
    )
    bad_parity = dict(parity, selected_image_keys=["a.vsi"])
    with pytest.raises(ValueError, match="selected_image_keys"):
        prepare_renderer_audits(
            manifest, nucleus_metrics, reconciliation, bad_parity
        )
