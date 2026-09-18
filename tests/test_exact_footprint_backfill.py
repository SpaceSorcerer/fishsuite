"""Tests for the exact-single-plane MIAT/QKI reconstruction backfill.

Each test protects a scientific boundary: the stored z plane, exact MIAT
footprints, KEEP-N threshold semantics, population retention, and resumable
serial outputs.  External VSI reading is represented by a complete tiny reader
double; the quantitative reconstruction itself always exercises real arrays.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

import fishsuite.core.exact_footprint_backfill as exact_footprint_backfill
from fishsuite.core.exact_footprint_backfill import (
    ExactFootprintParameters,
    _build_parser,
    _null_draw_index,
    aggregate_nucleus_metrics,
    build_image_manifest,
    calibrate_reconstruction_nulls,
    correlation_records,
    create_timestamped_output_dir,
    diameter_um_to_px,
    read_exact_selected_planes,
    read_and_validate_label_mask,
    reconstruct_image_footprints,
    run_exact_footprint_backfill,
    resolve_unique_mask_path,
    stable_nucleus_seed,
    summarize_spot_null_draws,
    validate_historical_parity,
    validate_spot_population_invariants,
    write_null_h5_group,
    write_selected_plane_h5_group,
)


def _hierarchy_rows() -> pd.DataFrame:
    rows = []
    for slide in (1, 2):
        for arm in ("NT", "KD"):
            for replicate in (1, 2, 3):
                image = f"S{slide}_{arm}_{replicate}.vsi"
                rows.append(
                    {
                        "image": image,
                        "image_key": image.casefold(),
                        "nucleus_id": 1,
                        "condition": f"S{slide}_{arm}_{replicate}",
                        "secondary_only": False,
                        "slide": slide,
                        "arm": arm,
                        "source_arm": arm,
                        "replicate": replicate,
                        "fov": replicate,
                        "biological_set": f"S{slide}_{arm}_{replicate}",
                        "catalog_folder": f"S{slide}_{arm}_{replicate}",
                        "is_control": False,
                        "eligible_for_sampling": True,
                        "sampled_in_analysis": replicate == 1,
                    }
                )
    for control_index in range(7):
        image = f"control_{control_index}.vsi"
        rows.append(
            {
                "image": image,
                "image_key": image.casefold(),
                "nucleus_id": 1,
                "condition": "Sec-Only",
                "secondary_only": True,
                "slide": 1 if control_index < 3 else 2,
                "arm": "CONTROL",
                "source_arm": "NT" if control_index % 2 == 0 else "KD",
                "replicate": np.nan,
                "fov": control_index,
                "biological_set": "",
                "catalog_folder": "FULL_OMISSION",
                "is_control": True,
                "eligible_for_sampling": True,
                "sampled_in_analysis": False,
            }
        )
    # Add biological FOVs without changing the 12 independent sets: 37 total
    # images plus seven controls = the audited 44-image universe.
    for index in range(25):
        base = rows[index % 12].copy()
        base["image"] = f"extra_{index}_{base['image']}"
        base["image_key"] = base["image"].casefold()
        base["fov"] = 100 + index
        rows.append(base)
    return pd.DataFrame(rows)


def _manifest_inputs(tmp_path: Path):
    hierarchy = _hierarchy_rows()
    images = hierarchy.drop_duplicates("image_key")
    per_image = images[["image", "condition", "secondary_only"]].copy()
    per_image["z_plane"] = 4
    per_image["dapi_channel"] = 2
    per_image["rna_channel"] = 0
    per_image["protein_channel"] = 1
    per_image["voxel_xy_nm"] = 100.0
    per_image["voxel_z_nm"] = 300.0
    per_image["n_z"] = 9
    per_image["nuclei_analyzed"] = 1
    spots = pd.DataFrame(
        {
            "image": per_image.loc[per_image.index[:-1], "image"],
            "channel": "rna1",
            "spot_id": 1,
        }
    )
    subset = [f"input/{name}" for name in per_image["image"]]
    run_config = {
        "input_dir": str(tmp_path / "input"),
        "config_resolved": {"input_file_subset": subset},
    }
    return per_image, hierarchy, spots, run_config


def test_manifest_preserves_44_images_controls_12_sets_and_zero_spot(tmp_path):
    per_image, hierarchy, spots, run_config = _manifest_inputs(tmp_path)

    manifest = build_image_manifest(
        per_image,
        hierarchy,
        spots,
        run_config=run_config,
        run_dir=tmp_path / "historical",
        validate_expected_design=True,
        resolve_paths=False,
    )

    assert len(manifest) == 44
    assert manifest["is_control"].sum() == 7
    assert (~manifest["is_control"]).sum() == 37
    assert manifest.loc[~manifest["is_control"], "biological_set"].nunique() == 12
    zero_spot_image = per_image.iloc[-1]["image"].casefold()
    assert manifest.set_index("image_key").loc[zero_spot_image, "n_input_spots"] == 0
    assert manifest["selected_z_0based"].eq(3).all()
    assert manifest["plane_lock_pass"].all()


def test_authoritative_hierarchy_overrides_duplicate_roster_fields_and_audits_mismatch(
    tmp_path,
):
    per_image, hierarchy, spots, run_config = _manifest_inputs(tmp_path)
    target_key = str(per_image.iloc[0]["image"]).casefold()
    expected = hierarchy.loc[
        hierarchy["image_key"].astype(str).str.casefold().eq(target_key)
    ].iloc[0]
    per_image.loc[0, "condition"] = "WRONG_RUN_TABLE_CONDITION"
    per_image.loc[0, "secondary_only"] = not bool(expected["secondary_only"])
    manifest = build_image_manifest(
        per_image,
        hierarchy,
        spots,
        run_config=run_config,
        run_dir=tmp_path / "historical",
        validate_expected_design=True,
        resolve_paths=False,
    ).set_index("image_key")
    row = manifest.loc[target_key]
    assert row["condition"] == expected["condition"]
    assert bool(row["secondary_only"]) == bool(expected["secondary_only"])
    assert row["hierarchy_mismatch_condition"]
    assert row["hierarchy_mismatch_secondary_only"]
    assert set(row["hierarchy_mismatch_fields"].split(";")) == {
        "condition", "secondary_only"
    }


def test_exact_plane_reader_uses_yx_and_same_recorded_z_for_all_channels(tmp_path):
    calls = []

    class FakeBio:
        shape = (1, 3, 8, 5, 6)
        dims = type("Dims", (), {"order": "TCZYX"})()

        def get_image_data(self, order, **kwargs):
            calls.append((order, kwargs.copy()))
            return np.full((5, 6), 100 * kwargs["C"] + kwargs["Z"], np.uint16)

    class FakeImage:
        bio = FakeBio()
        n_z = 8
        n_channels = 3
        voxel_xy_nm = 65.0
        voxel_z_nm = 210.0

    planes, metadata = read_exact_selected_planes(
        tmp_path / "image.vsi",
        selected_z_1based=5,
        channel_indices={"miat": 0, "qki": 1, "dapi": 2},
        image_reader=lambda _path: FakeImage(),
    )

    assert calls == [
        ("YX", {"T": 0, "C": 0, "Z": 4}),
        ("YX", {"T": 0, "C": 1, "Z": 4}),
        ("YX", {"T": 0, "C": 2, "Z": 4}),
    ]
    assert set(planes) == {"miat", "qki", "dapi"}
    assert metadata["selected_z_0based"] == 4
    assert int(planes["qki"][0, 0]) == 104


def test_exact_plane_reader_closes_reader_on_success_and_failure(tmp_path):
    live = {"count": 0, "max": 0}

    class ClosingBio:
        def __init__(self, fail=False):
            self.fail = fail
            self.closed = False

        def get_image_data(self, order, **kwargs):
            if self.fail and kwargs["C"] == 1:
                raise RuntimeError("injected read failure")
            return np.ones((3, 4), np.uint16)

        def close(self):
            if not self.closed:
                self.closed = True
                live["count"] -= 1

    class ClosingImage:
        n_z = 3
        n_channels = 3
        voxel_xy_nm = 65.0
        voxel_z_nm = 210.0

        def __init__(self, fail=False):
            live["count"] += 1
            live["max"] = max(live["max"], live["count"])
            self.bio = ClosingBio(fail)

    channels = {"miat": 0, "qki": 1, "dapi": 2}
    read_exact_selected_planes(
        tmp_path / "ok.vsi", selected_z_1based=2, channel_indices=channels,
        image_reader=lambda _path: ClosingImage(),
    )
    assert live == {"count": 0, "max": 1}
    with pytest.raises(RuntimeError, match="injected"):
        read_exact_selected_planes(
            tmp_path / "bad.vsi", selected_z_1based=2, channel_indices=channels,
            image_reader=lambda _path: ClosingImage(fail=True),
        )
    assert live == {"count": 0, "max": 1}


def test_plane_reader_rejects_missing_or_out_of_range_selected_z(tmp_path):
    class FakeImage:
        n_z = 3
        n_channels = 3

    with pytest.raises(ValueError, match="selected_z_1based"):
        read_exact_selected_planes(
            tmp_path / "x.vsi",
            selected_z_1based=np.nan,
            channel_indices={"miat": 0, "qki": 1, "dapi": 2},
            image_reader=lambda _path: FakeImage(),
        )
    with pytest.raises(ValueError, match="outside"):
        read_exact_selected_planes(
            tmp_path / "x.vsi",
            selected_z_1based=4,
            channel_indices={"miat": 0, "qki": 1, "dapi": 2},
            image_reader=lambda _path: FakeImage(),
        )


def test_diameter_conversion_is_um_divided_by_voxel_um():
    got = diameter_um_to_px(np.array([0.26, 0.52]), voxel_xy_nm=130.0)
    np.testing.assert_allclose(got, [2.0, 4.0])
    with pytest.raises(ValueError, match="voxel_xy_nm"):
        diameter_um_to_px([0.26], voxel_xy_nm=0)


def _synthetic_reconstruction():
    miat = np.full((15, 15), 10, dtype=np.uint16)
    qki = np.full((15, 15), 20, dtype=np.uint16)
    labels = np.zeros((15, 15), dtype=np.int32)
    labels[1:14, 1:14] = 1
    nucleolus = np.zeros_like(labels)
    nucleolus[10:13, 10:13] = 1
    # Each exact half-max component is two pixels: peak plus one 60-count pixel.
    for (y, x), q_values in [((4, 4), (30, 50)), ((8, 8), (70, 90)), ((11, 11), (110, 130))]:
        miat[y, x] = 100
        miat[y, x + 1] = 60
        qki[y, x], qki[y, x + 1] = q_values
    spots = pd.DataFrame(
        {
            "image": "field.vsi",
            "image_key": "field.vsi",
            "condition": "S1_NT_1",
            "secondary_only": False,
            "spot_id": [10, 20, 30],
            "channel": "rna1",
            "nucleus_id": 1,
            "x_px": [4, 8, 11],
            "y_px": [4, 8, 11],
            "spot_peak_intensity": 100.0,
            "peak_intensity": 100.0,
            "spot_diameter_um": 0.2,
            "miat_footprint_area_px": 2.0,
            "qki_at_miat_footprint": [40.0, 80.0, 120.0],
            "in_nucleolus": [0, 0, 1],
            "in_nucleus_excluding_nucleolus": [1, 1, 0],
        }
    )
    hierarchy = {
        "image": "field.vsi",
        "image_key": "field.vsi",
        "condition": "S1_NT_1",
        "secondary_only": False,
        "is_control": False,
        "slide": 1,
        "arm": "NT",
        "source_arm": "NT",
        "replicate": 1,
        "fov": 1,
        "biological_set": "S1_NT_1",
        "catalog_folder": "S1_NT_1",
        "selected_z_1based": 5,
        "voxel_xy_nm": 100.0,
        "eligible_for_sampling": True,
        "sampled_in_analysis": True,
    }
    params = ExactFootprintParameters(
        miat_floor_raw=50, n_null=4, max_redraw=20, global_seed=7
    )
    return reconstruct_image_footprints(
        miat,
        qki,
        labels,
        nucleolus,
        spots,
        hierarchy,
        parameters=params,
        run_id="run-A",
        compute_null=False,
    )


def test_reconstruction_keeps_continuous_metrics_for_nucleolar_spots():
    result = _synthetic_reconstruction()
    table = result.spot_metrics.set_index("spot_id")

    assert list(table["footprint_area_px"]) == [2, 2, 2]
    assert table.loc[10, "miat_footprint_mean_raw"] == 80.0
    assert table.loc[20, "qki_footprint_mean_raw"] == 80.0
    assert table.loc[30, "qki_footprint_mean_raw"] == 120.0
    assert bool(table.loc[30, "stored_in_nucleolus"])
    assert table.loc[30, "null_exclusion_reason"] == "nucleolar_spot"
    assert table.loc[30, "population_label"] == "unusable"
    assert pd.isna(table.loc[30, "qki_threshold_positive"])
    assert result.pixel_metrics.groupby("spot_uid").size().tolist() == [2, 2, 2]


def test_invalid_empty_footprint_cannot_enter_null_or_intensity_endpoints():
    """Catches a disk-derived footprint contributing pixels or endpoint mass."""
    miat = np.full((7, 7), 3, dtype=np.uint16)
    qki = np.full((7, 7), 11, dtype=np.uint16)
    labels = np.zeros((7, 7), dtype=np.int32)
    labels[1:6, 1:6] = 1
    spots = pd.DataFrame(
        {
            "spot_id": [1],
            "channel": ["rna1"],
            "nucleus_id": [1],
            "x_px": [3],
            "y_px": [3],
            "spot_diameter_um": [0.2],
            "spot_peak_intensity": [100.0],
            "in_nucleolus": [False],
        }
    )
    image = {
        "image": "flat.vsi",
        "image_key": "flat.vsi",
        "condition": "S1_NT_1",
        "secondary_only": False,
        "is_control": False,
        "slide": 1,
        "arm": "NT",
        "source_arm": "NT",
        "replicate": 1,
        "fov": 1,
        "biological_set": "S1_NT_1",
        "catalog_folder": "S1_NT_1",
        "selected_z_1based": 5,
        "voxel_xy_nm": 100.0,
        "eligible_for_sampling": True,
        "sampled_in_analysis": True,
    }
    reconstruction = reconstruct_image_footprints(
        miat,
        qki,
        labels,
        np.zeros_like(labels),
        spots,
        image,
        parameters=ExactFootprintParameters(miat_floor_raw=50, n_null=3),
        run_id="flat-run",
        compute_null=False,
    )
    row = reconstruction.spot_metrics.iloc[0]

    assert row["footprint_method"] == "invalid_empty_footprint"
    assert row["footprint_fallback_reason"] == "flat_or_no_contrast"
    assert row["footprint_invalid_reason"] == "empty_footprint"
    assert int(row["footprint_area_px"]) == 0
    assert not bool(row["null_candidate"])
    assert row["null_exclusion_reason"] == "empty_footprint"
    assert reconstruction.pixel_metrics.empty

    nuclei = pd.DataFrame(
        [{"image_key": "flat.vsi", "nucleus_id": 1, "nucleus_qc_status": "pass"}]
    )
    endpoint = aggregate_nucleus_metrics(
        reconstruction.spot_metrics,
        reconstruction.pixel_metrics,
        nuclei,
        miat_2d=miat,
        qki_2d=qki,
        nucleus_labels=labels,
        nucleolus_labels=np.zeros_like(labels),
    ).iloc[0]
    assert int(endpoint["n_spots_all"]) == 1
    assert int(endpoint["n_valid_exact_footprints"]) == 0
    assert endpoint["miat_footprint_mass_all_spot_summed"] == 0.0
    assert endpoint["miat_footprint_mass_all_union_deduplicated"] == 0.0
    assert endpoint["qki_footprint_mass_all_spot_summed"] == 0.0
    assert endpoint["qki_footprint_mass_all_union_deduplicated"] == 0.0


def test_reconstruction_uses_nucleus_specific_sampling_flags():
    miat = np.full((12, 12), 10, dtype=np.uint16)
    qki = np.full((12, 12), 20, dtype=np.uint16)
    labels = np.zeros((12, 12), dtype=np.int32)
    labels[1:11, 1:6] = 1
    labels[1:11, 6:11] = 2
    miat[4, 3] = 100
    miat[4, 4] = 60
    miat[7, 8] = 100
    miat[7, 9] = 60
    spots = pd.DataFrame(
        {
            "spot_id": [1, 2],
            "channel": "rna1",
            "nucleus_id": [1, 2],
            "x_px": [3, 8],
            "y_px": [4, 7],
            "spot_diameter_um": 0.2,
            "spot_peak_intensity": 100.0,
            "in_nucleolus": False,
        }
    )
    image_metadata = {
        "image": "two-nuclei.vsi",
        "image_key": "two-nuclei.vsi",
        "condition": "S1_NT_1",
        "secondary_only": False,
        "is_control": False,
        "slide": 1,
        "arm": "NT",
        "source_arm": "NT",
        "replicate": 1,
        "fov": 1,
        "biological_set": "S1_NT_1",
        "catalog_folder": "S1_NT_1",
        "selected_z_1based": 5,
        "voxel_xy_nm": 100.0,
        # Image-level aggregation is deliberately True for both fields.
        "eligible_for_sampling": True,
        "sampled_in_analysis": True,
    }
    nucleus_hierarchy = pd.DataFrame(
        {
            "image_key": ["two-nuclei.vsi", "two-nuclei.vsi"],
            "nucleus_id": [1, 2],
            "eligible_for_sampling": [True, False],
            "sampled_in_analysis": [True, False],
        }
    )

    result = reconstruct_image_footprints(
        miat,
        qki,
        labels,
        np.zeros_like(labels),
        spots,
        image_metadata,
        parameters=ExactFootprintParameters(miat_floor_raw=50, n_null=3),
        run_id="run-hierarchy",
        compute_null=False,
        nucleus_hierarchy=nucleus_hierarchy,
    )

    table = result.spot_metrics.sort_values("nucleus_id")
    assert table["eligible_for_sampling"].tolist() == [True, False]
    assert table["sampled_in_analysis"].tolist() == [True, False]


@pytest.mark.parametrize("two_phase", [False, True])
def test_low_retention_null_preserves_candidate_and_exclusion_reason(
    monkeypatch, two_phase
):
    original_null = exact_footprint_backfill.keep_n_footprint_rotation_null

    def forced_low_retention(*args, **kwargs):
        result = original_null(*args, **kwargs)
        spots = tuple(
            replace(
                spot,
                invalid_reason="low_first_pass_retention",
                null_usable=False,
                association_call=None,
            )
            for spot in result.spots
        )
        return replace(
            result,
            spots=spots,
            mean_first_pass_retention=0.25,
            median_first_pass_retention=0.25,
            usable=False,
            invalid_reason="low_first_pass_retention",
        )

    monkeypatch.setattr(
        exact_footprint_backfill,
        "keep_n_footprint_rotation_null",
        forced_low_retention,
    )
    base = _synthetic_reconstruction()
    source = base.spot_metrics.copy()
    source["in_nucleolus"] = source["stored_in_nucleolus"]
    source["in_nucleus_excluding_nucleolus"] = source[
        "stored_in_nucleus_excluding_nucleolus"
    ]
    parameters = ExactFootprintParameters(
        miat_floor_raw=50, n_null=4, max_redraw=20, global_seed=7
    )
    if two_phase:
        result = calibrate_reconstruction_nulls(
            base, parameters=parameters, run_id="run-A"
        )
    else:
        result = reconstruct_image_footprints(
            base.planes["miat"],
            base.planes["qki"],
            base.nucleus_labels,
            base.nucleolus_labels,
            source,
            base.spot_metrics.iloc[0].to_dict(),
            parameters=parameters,
            run_id="run-A",
            compute_null=True,
        )

    table = result.spot_metrics.set_index("spot_id")
    candidates = table.loc[[10, 20]]
    assert candidates["null_candidate"].map(bool).all()
    assert not candidates["null_usable"].map(bool).any()
    assert candidates["null_exclusion_reason"].eq(
        "low_first_pass_retention"
    ).all()
    assert not bool(table.loc[30, "null_candidate"])
    assert table.loc[30, "null_exclusion_reason"] == "nucleolar_spot"


def test_parity_gate_checks_exact_area_and_qki_mean():
    result = _synthetic_reconstruction()
    original = pd.DataFrame(
        {
            "image_key": "field.vsi",
            "channel": "rna1",
            "spot_id": [10, 20, 30],
            "miat_footprint_area_px": [2.0, 2.0, 2.0],
            "qki_at_miat_footprint": [40.0, 80.0, 120.0],
        }
    )
    report = validate_historical_parity(result.spot_metrics, original)
    assert report == {"n_spots": 3, "area_exact": True, "qki_allclose": True}

    wrong = original.copy()
    wrong.loc[0, "qki_at_miat_footprint"] = 41.0
    with pytest.raises(ValueError, match="QKI footprint parity"):
        validate_historical_parity(result.spot_metrics, wrong)


def test_nucleolar_discrepancy_is_reported_without_overwriting_stored_flag():
    result = _synthetic_reconstruction()
    table = result.spot_metrics.set_index("spot_id")
    assert table.loc[30, "stored_in_nucleolus"]
    assert table.loc[30, "recomputed_center_in_nucleolus"]
    assert not result.audit["nucleolar_center_discrepancy_count"]

    altered = result.spot_metrics.copy()
    altered.loc[altered["spot_id"].eq(30), "stored_in_nucleolus"] = False
    # The audit helper is exercised by passing the altered stored source back
    # through a fresh reconstruction; recomputed geometry must win only in the
    # separate audit column, never overwrite the stored flag.
    source = _synthetic_reconstruction().spot_metrics
    assert "recomputed_center_in_nucleolus" in source


def test_both_nucleolar_disagreement_directions_retain_measurements_and_exclude_null():
    base = _synthetic_reconstruction()
    # Recover the persisted-like source, then deliberately reverse a non-
    # nucleolar and a nucleolar stored centre call.
    source = pd.DataFrame(
        {
            "image": "field.vsi", "image_key": "field.vsi",
            "condition": "S1_NT_1", "secondary_only": False,
            "spot_id": [10, 20, 30], "channel": "rna1", "nucleus_id": 1,
            "x_px": [4, 8, 11], "y_px": [4, 8, 11],
            "spot_peak_intensity": 100.0, "peak_intensity": 100.0,
            "spot_diameter_um": 0.2,
            "in_nucleolus": [True, False, False],
            "in_nucleus_excluding_nucleolus": [False, True, True],
        }
    )
    metadata = base.spot_metrics.iloc[0].to_dict()
    result = reconstruct_image_footprints(
        base.planes["miat"], base.planes["qki"], base.nucleus_labels,
        base.nucleolus_labels, source, metadata,
        parameters=ExactFootprintParameters(miat_floor_raw=50, n_null=3),
        run_id="disagreement", compute_null=False,
    )
    table = result.spot_metrics.set_index("spot_id")
    assert result.audit["nucleolar_center_discrepancy_count"] == 2
    assert table.loc[10, "stored_in_nucleolus"]
    assert not table.loc[10, "recomputed_center_in_nucleolus"]
    assert table.loc[10, "null_exclusion_reason"] == "nucleolar_spot"
    assert not table.loc[30, "stored_in_nucleolus"]
    assert table.loc[30, "recomputed_center_in_nucleolus"]
    assert table.loc[30, "null_exclusion_reason"] == "footprint_not_full_nucleoplasm"
    assert np.isfinite(table.loc[[10, 30], "qki_footprint_mean_raw"]).all()


def test_stable_seed_is_order_independent_and_keyed_by_nucleus():
    a = stable_nucleus_seed("run", "image-a", 7, 11)
    b = stable_nucleus_seed("run", "image-a", 7, 11)
    c = stable_nucleus_seed("run", "image-a", 8, 11)
    assert a == b
    assert a != c
    assert 0 <= a < 2**64


def test_null_summary_uses_own_linear_quantiles_and_strict_calls():
    observed = np.array([10.0, 9.5, 5.0])
    draws = np.array([[0.0, 10.0], [1.0, 9.0], [5.0, 5.0]])
    summary = summarize_spot_null_draws(observed, draws, require_all_draws=True)

    np.testing.assert_allclose(summary["null_q95_raw"], [9.5, 8.6, 5.0])
    assert summary["qki_threshold_positive_q95"].tolist() == [True, True, False]
    assert summary["qki_threshold_positive_q90"].tolist() == [True, True, False]
    assert summary["population_label_q90"].tolist() == [
        "threshold_positive", "threshold_positive", "threshold_negative"
    ]
    assert summary["population_label_q95"].tolist() == [
        "threshold_positive", "threshold_positive", "threshold_negative"
    ]
    assert summary["population_label_q99"].tolist() == [
        "threshold_positive", "threshold_positive", "threshold_negative"
    ]
    assert summary["population_label"].tolist() == summary["population_label_q95"].tolist()
    assert summary["null_p_empirical"].tolist() == pytest.approx([2 / 3, 1 / 3, 1.0])
    assert np.isfinite(summary["null_z"]).tolist() == [True, True, False]


def test_require_all_draws_marks_every_threshold_and_label_unusable():
    summary = summarize_spot_null_draws(
        [10.0], [[1.0, np.nan, 3.0]], require_all_draws=True
    ).iloc[0]
    assert not summary["null_usable"]
    for percentile in (90, 95, 99):
        assert pd.isna(summary[f"qki_threshold_positive_q{percentile}"])
        assert summary[f"population_label_q{percentile}"] == "unusable"


def test_null_h5_roundtrip_retains_draw_provenance_and_reproduces_calls(tmp_path):
    spot_uid = np.array(["img:rna1:1", "img:rna1:2"], dtype=object)
    observed = np.array([10.0, 9.5])
    draws = np.array([[0.0, 10.0], [1.0, 9.0]])
    summary = summarize_spot_null_draws(observed, draws, require_all_draws=True)
    path = tmp_path / "nulls.h5"
    with h5py.File(path, "w") as handle:
        group_path = write_null_h5_group(
            handle,
            nucleus_uid="img:nucleus:1",
            image_key="img",
            nucleus_id=1,
            selected_z_1based=5,
            seed=17,
            spot_uid=spot_uid,
            observed_qki_raw=observed,
            draw_qki_raw=draws,
            initial_angles_deg=np.array([90.0, 180.0]),
            placement_angles_deg=np.array([[90.0, 180.0], [90.0, 200.0]]),
            first_pass_valid=np.array([[True, True], [True, False]]),
            redraw_counts=np.array([[0, 0], [0, 1]], np.int32),
            null_usable=True,
            unusable_reason="",
            max_redraw=5,
            threshold_percentile=90,
            first_pass_retention_mean=0.75,
            first_pass_retention_median=0.75,
        )
    with h5py.File(path, "r") as handle:
        group = handle[group_path]
        np.testing.assert_array_equal(group["draw_qki_raw"], draws)
        assert group.attrs["placement_geometry"] == "rotated_center_translated_exact_footprint"
        assert group.attrs["threshold_operator"] == "observed_qki_mean > own_linear_quantile"
        assert group.attrs["quantile_method"] == "linear"
        assert group.attrs["redraw_fraction"] == pytest.approx(0.25)
        assert group.attrs["unplaceable_fraction"] == pytest.approx(0.0)
        assert group.attrs["max_redraw"] == 5
        assert group.attrs["threshold_percentile"] == 90
        reread = summarize_spot_null_draws(
            group["observed_qki_raw"][:], group["draw_qki_raw"][:], require_all_draws=True
        )
    np.testing.assert_allclose(reread["null_q95_raw"], summary["null_q95_raw"])
    assert reread["qki_threshold_positive_q95"].tolist() == summary[
        "qki_threshold_positive_q95"
    ].tolist()


def test_all_missing_low_retention_draws_are_unplaceable_in_h5_and_index(tmp_path):
    draws = np.full((2, 3), np.nan)
    payload = {
        "initial_angles_deg": np.array([10.0, 20.0, 30.0]),
        "first_pass_valid": np.zeros((2, 3), dtype=bool),
        "redraw_counts": np.zeros((2, 3), dtype=np.int32),
        "draw_qki_raw": draws,
    }
    index = _null_draw_index("img", 1, payload)
    assert index["unplaceable_n"].tolist() == [2, 2, 2]
    assert not index["complete_keep_n"].any()

    path = tmp_path / "low-retention.h5"
    with h5py.File(path, "w") as handle:
        group_path = write_null_h5_group(
            handle,
            nucleus_uid="img:nucleus:1",
            image_key="img",
            nucleus_id=1,
            selected_z_1based=5,
            seed=1,
            spot_uid=["a", "b"],
            observed_qki_raw=[1.0, 2.0],
            draw_qki_raw=draws,
            initial_angles_deg=payload["initial_angles_deg"],
            placement_angles_deg=np.full((2, 3), np.nan),
            first_pass_valid=payload["first_pass_valid"],
            redraw_counts=payload["redraw_counts"],
            null_usable=False,
            unusable_reason="low_first_pass_retention",
        )
    with h5py.File(path, "r") as handle:
        assert handle[group_path].attrs["unplaceable_fraction"] == 1.0
        assert handle[group_path].attrs["unplaceable_count"] == 6


def test_selected_plane_h5_stores_exact_planes_and_masks(tmp_path):
    path = tmp_path / "planes.h5"
    planes = {
        "miat": np.full((3, 4), 11, np.uint16),
        "qki": np.full((3, 4), 22, np.uint16),
        "dapi": np.full((3, 4), 33, np.uint16),
    }
    labels = np.arange(12, dtype=np.int32).reshape(3, 4)
    nucleoli = (labels == 5).astype(np.int32)
    with h5py.File(path, "w") as handle:
        write_selected_plane_h5_group(
            handle,
            image_key="Field.VSI",
            selected_z_1based=7,
            planes=planes,
            nucleus_labels=labels,
            nucleolus_labels=nucleoli,
            channel_indices={"miat": 0, "qki": 1, "dapi": 2},
        )
    with h5py.File(path, "r") as handle:
        group = handle["images/field.vsi"]
        assert group.attrs["selected_z_0based"] == 6
        np.testing.assert_array_equal(group["qki"], planes["qki"])
        np.testing.assert_array_equal(group["nucleus_labels"], labels)


def _correlation_spots() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "nucleus_id": 1,
            "miat_footprint_mean_raw": [1.0, 2.0, 3.0, 4.0, 8.0],
            "qki_footprint_mean_raw": [2.0, 4.0, 6.0, 8.0, 1.0],
            "miat_footprint_enrichment_vs_nucleus": [0.5, 1, 1.5, 2, 4],
            "qki_footprint_enrichment_vs_nucleus": [0.5, 1, 1.5, 2, 0.25],
            "footprint_full_nucleus_valid": [True] * 5,
            "footprint_full_nucleoplasm_valid": [True, True, True, False, False],
            "stored_in_nucleolus": [False, False, False, False, True],
            "null_candidate": [True, True, True, False, False],
            "qki_threshold_positive_q90": [True, True, True, False, False],
            "qki_threshold_positive_q95": [True, True, False, False, False],
            "qki_threshold_positive_q99": [True, False, False, False, False],
            "null_usable": [True, True, True, True, False],
        }
    )


def test_correlations_expose_all_non_nucleolar_and_threshold_populations():
    records = correlation_records(_correlation_spots(), nucleus_id=1)
    populations = set(records["population"])
    assert populations == {
        "all_detected_exact_valid",
        "non_nucleolar_exact_valid",
        "threshold_positive_q90",
        "threshold_positive_q95",
        "threshold_positive_q99",
    }
    by_pop = records.loc[records["measurement_pair"].eq("raw")].set_index("population")
    assert by_pop.loc["all_detected_exact_valid", "n_spots"] == 5
    assert by_pop.loc["non_nucleolar_exact_valid", "n_spots"] == 3
    assert by_pop.loc["threshold_positive_q95", "n_spots"] == 2
    assert not by_pop.loc["threshold_positive_q95", "estimable"]
    assert by_pop.loc["threshold_positive_q95", "conditional_descriptive"]


def test_threshold_correlation_requires_both_true_call_and_usable_null():
    frame = _correlation_spots()
    frame.loc[1, "null_usable"] = False
    frame.loc[1, "qki_threshold_positive_q95"] = True
    records = correlation_records(frame, nucleus_id=1)
    q95 = records.loc[
        records["population"].eq("threshold_positive_q95")
        & records["measurement_pair"].eq("raw")
    ].iloc[0]
    assert q95["n_spots"] == 1


def test_correlations_report_constant_or_under_three_as_nonestimable():
    frame = _correlation_spots()
    frame["qki_footprint_mean_raw"] = 3.0
    records = correlation_records(frame, nucleus_id=1)
    raw_all = records.loc[
        records["population"].eq("all_detected_exact_valid")
        & records["measurement_pair"].eq("raw")
    ].iloc[0]
    assert not raw_all["estimable"]
    assert raw_all["nonestimable_reason"] == "constant_vector"


def test_nucleus_aggregation_reconciles_population_and_union_mass():
    result = _synthetic_reconstruction()
    spots = result.spot_metrics.copy()
    spots["qki_threshold_positive_q95"] = [True, False, pd.NA]
    spots["qki_threshold_positive_q90"] = [True, True, pd.NA]
    spots["qki_threshold_positive_q99"] = [False, False, pd.NA]
    spots["qki_threshold_positive"] = spots["qki_threshold_positive_q95"]
    spots["null_usable"] = [True, True, False]
    spots["population_label"] = ["threshold_positive", "threshold_negative", "unusable"]
    nuclei = pd.DataFrame(
        {
            "image_key": ["field.vsi"],
            "image": ["field.vsi"],
            "nucleus_id": [1],
            "eligible_for_sampling": [True],
            "sampled_in_analysis": [True],
        }
    )
    out = aggregate_nucleus_metrics(
        spots,
        result.pixel_metrics,
        nuclei,
        miat_2d=result.planes["miat"],
        qki_2d=result.planes["qki"],
        nucleus_labels=result.nucleus_labels,
        nucleolus_labels=result.nucleolus_labels,
    )
    row = out.iloc[0]
    assert row["n_spots_all"] == 3
    assert row["n_threshold_positive"] == 1
    assert row["n_threshold_negative"] == 1
    assert row["n_unusable"] == 1
    assert row["population_reconciliation_pass"]
    assert row["miat_footprint_mass_all_spot_summed"] == 480.0
    assert row["miat_footprint_mass_all_union_deduplicated"] == 480.0


def test_union_mass_deduplicates_intentional_overlap_and_positive_subset():
    spots = pd.DataFrame(
        {
            "image_key": ["img", "img"], "nucleus_id": [1, 1],
            "spot_uid": ["img:rna1:1", "img:rna1:2"],
            "passes_miat_floor": [True, True], "stored_in_nucleolus": [False, False],
            "footprint_full_nucleus_valid": [True, True], "null_candidate": [True, True],
            "null_usable": [True, True],
            "qki_threshold_positive_q90": [True, False],
            "qki_threshold_positive_q95": [True, False],
            "qki_threshold_positive_q99": [True, False],
            "population_label_q90": ["threshold_positive", "threshold_negative"],
            "population_label_q95": ["threshold_positive", "threshold_negative"],
            "population_label_q99": ["threshold_positive", "threshold_negative"],
            "population_label": ["threshold_positive", "threshold_negative"],
            "miat_footprint_sum_raw": [13.0, 17.0],
            "qki_footprint_sum_raw": [4.0, 6.0],
            "miat_footprint_mean_raw": [6.5, 8.5],
            "qki_footprint_mean_raw": [2.0, 3.0],
            "miat_footprint_enrichment_vs_nucleus": [1.0, 1.1],
            "qki_footprint_enrichment_vs_nucleus": [1.0, 1.2],
        }
    )
    pixels = pd.DataFrame(
        {
            "spot_uid": ["img:rna1:1", "img:rna1:1", "img:rna1:2", "img:rna1:2"],
            "flat_pixel_index": [1, 2, 2, 3],
            "miat_raw": [5.0, 8.0, 8.0, 9.0],
            "qki_raw": [1.0, 3.0, 3.0, 3.0],
        }
    )
    labels = np.ones((2, 2), dtype=np.int32)
    out = aggregate_nucleus_metrics(
        spots, pixels,
        pd.DataFrame({"image_key": ["img"], "image": ["img"], "nucleus_id": [1]}),
        miat_2d=np.array([[5, 8], [9, 1]]), qki_2d=np.ones((2, 2)),
        nucleus_labels=labels, nucleolus_labels=np.zeros_like(labels),
    ).iloc[0]
    assert out["miat_footprint_mass_all_spot_summed"] == 30.0
    assert out["miat_footprint_mass_all_union_deduplicated"] == 22.0
    assert out["miat_footprint_mass_positive_spot_summed"] == 13.0
    assert out["miat_footprint_mass_positive_union_deduplicated"] == 13.0
    assert out["qki_footprint_mass_all_spot_summed"] == 10.0
    assert out["qki_footprint_mass_all_union_deduplicated"] == 7.0
    assert out["qki_footprint_mass_positive_union_deduplicated"] == 4.0


def test_controls_remain_raw_but_are_ineligible_for_biological_inference():
    per_image, hierarchy, spots, run_config = _manifest_inputs(Path("C:/tmp"))
    manifest = build_image_manifest(
        per_image,
        hierarchy,
        spots,
        run_config=run_config,
        run_dir=Path("C:/tmp/historical"),
        validate_expected_design=True,
        resolve_paths=False,
    )
    assert len(manifest.loc[manifest["is_control"]]) == 7
    assert not manifest.loc[manifest["is_control"], "eligible_for_biological_inference"].any()
    assert manifest.loc[~manifest["is_control"], "eligible_for_biological_inference"].all()


def test_display_ranges_are_not_quantitative_parameters():
    fields = set(ExactFootprintParameters.__dataclass_fields__)
    assert not any("display" in field or "vmin" in field or "vmax" in field for field in fields)


def test_timestamped_output_is_new_and_refuses_historical_run_path(tmp_path):
    historical = tmp_path / "completed-run"
    historical.mkdir()
    out = create_timestamped_output_dir(
        tmp_path / "analysis",
        source_run_dir=historical,
        timestamp="20260828-120000",
    )
    assert out.name == "EXACT_FOOTPRINT_BACKFILL_20260828-120000"
    assert out.is_dir()
    with pytest.raises(FileExistsError):
        create_timestamped_output_dir(
            tmp_path / "analysis",
            source_run_dir=historical,
            timestamp="20260828-120000",
        )
    with pytest.raises(ValueError, match="historical run"):
        create_timestamped_output_dir(
            historical.parent,
            source_run_dir=historical,
            timestamp="completed-run",
            prefix="",
        )


def test_checkpoint_metadata_is_json_serializable():
    params = ExactFootprintParameters()
    json.dumps(params.to_dict())


def test_zero_spot_image_keeps_nucleus_and_parity_is_well_defined():
    base = _synthetic_reconstruction()
    empty_source = pd.DataFrame(
        columns=[
            "image", "image_key", "condition", "secondary_only", "spot_id",
            "channel", "nucleus_id", "x_px", "y_px", "spot_peak_intensity",
            "peak_intensity", "spot_diameter_um", "in_nucleolus",
            "in_nucleus_excluding_nucleolus", "miat_footprint_area_px",
            "qki_at_miat_footprint",
        ]
    )
    result = reconstruct_image_footprints(
        base.planes["miat"], base.planes["qki"], base.nucleus_labels,
        base.nucleolus_labels, empty_source,
        {
            "image": "zero.vsi", "image_key": "zero.vsi", "condition": "Sec-Only",
            "secondary_only": True, "is_control": True, "slide": 2,
            "arm": "CONTROL", "source_arm": "KD", "replicate": np.nan,
            "fov": 42, "biological_set": "", "catalog_folder": "FULL_OMISSION",
            "eligible_for_sampling": True, "sampled_in_analysis": False,
            "selected_z_1based": 9, "voxel_xy_nm": 100.0,
        },
        parameters=ExactFootprintParameters(miat_floor_raw=50, n_null=3),
        run_id="run-zero", compute_null=False,
    )
    assert result.spot_metrics.empty
    assert validate_historical_parity(result.spot_metrics, empty_source) == {
        "n_spots": 0, "area_exact": True, "qki_allclose": True
    }
    nuclei = pd.DataFrame(
        {"image_key": ["zero.vsi"], "image": ["zero.vsi"], "nucleus_id": [1]}
    )
    out = aggregate_nucleus_metrics(
        result.spot_metrics, result.pixel_metrics, nuclei,
        miat_2d=base.planes["miat"], qki_2d=base.planes["qki"],
        nucleus_labels=base.nucleus_labels, nucleolus_labels=base.nucleolus_labels,
    )
    assert len(out) == 1
    assert out.iloc[0]["n_spots_all"] == 0
    assert out.iloc[0]["population_reconciliation_pass"]


def _write_synthetic_completed_run(tmp_path: Path):
    run = tmp_path / "completed"
    source = tmp_path / "input" / "S1_NT_1"
    masks = run / "masks"
    source.mkdir(parents=True)
    masks.mkdir(parents=True)
    vsi = source / "field.vsi"
    vsi.write_bytes(b"synthetic-reader-placeholder")
    ets_dir = source / "_field_" / "stack1"
    ets_dir.mkdir(parents=True)
    (ets_dir / "frame_t_0.ets").write_bytes(b"nonzero")

    base = _synthetic_reconstruction()
    historical = base.spot_metrics[
        ["image", "condition", "secondary_only", "spot_id", "channel", "nucleus_id",
         "x_px", "y_px", "spot_peak_intensity", "peak_intensity", "spot_diameter_um",
         "stored_in_nucleolus", "stored_in_nucleus_excluding_nucleolus",
         "footprint_area_px", "qki_footprint_mean_raw"]
    ].rename(
        columns={
            "stored_in_nucleolus": "in_nucleolus",
            "stored_in_nucleus_excluding_nucleolus": "in_nucleus_excluding_nucleolus",
            "footprint_area_px": "miat_footprint_area_px",
            "qki_footprint_mean_raw": "qki_at_miat_footprint",
        }
    )
    historical["z_slice"] = [0, 999, -10]
    historical.to_csv(run / "spot_metrics.csv", index=False)
    pd.DataFrame(
        {
            "image": ["field.vsi"], "condition": ["S1_NT_1"],
            "secondary_only": [False], "z_plane": [5], "dapi_channel": [2],
            "rna_channel": [0], "protein_channel": [1], "voxel_xy_nm": [100.0],
            "voxel_z_nm": [300.0], "n_z": [9], "nuclei_analyzed": [1],
        }
    ).to_csv(run / "per_image_summary.csv", index=False)
    hierarchy = pd.DataFrame(
        {
            "image": ["field.vsi"], "image_key": ["field.vsi"], "nucleus_id": [1],
            "condition": ["S1_NT_1"], "secondary_only": [False], "slide": [1],
            "arm": ["NT"], "source_arm": ["NT"], "replicate": [1], "fov": [1],
            "biological_set": ["S1_NT_1"], "catalog_folder": ["S1_NT_1"],
            "is_control": [False], "eligible_for_sampling": [True],
            "sampled_in_analysis": [True], "source_vsi": [str(vsi)],
        }
    )
    hierarchy_path = tmp_path / "nuclei_coloc_derived.csv"
    hierarchy.to_csv(hierarchy_path, index=False)
    hierarchy.to_csv(run / "nuclei_metrics.csv", index=False)
    (run / "run_config.json").write_text(
        json.dumps(
            {
                "input_dir": str(tmp_path / "input"),
                "config_resolved": {
                    "input_file_subset": ["S1_NT_1/field.vsi"],
                    "channels": {"rna": 0, "antibody": 1, "dapi": 2},
                },
            }
        ),
        encoding="utf-8",
    )
    import tifffile

    tifffile.imwrite(
        masks / "S1_NT_1__field__nuclei_label_mask.tif",
        base.nucleus_labels.astype(np.uint16),
    )
    planes = {
        "miat": base.planes["miat"],
        "qki": base.planes["qki"],
        "dapi": np.full_like(base.planes["miat"], 500),
    }
    return run, hierarchy_path, planes


def _append_synthetic_image(
    run: Path,
    hierarchy_path: Path,
    *,
    image: str,
    with_spots: bool,
) -> None:
    condition = "S1_NT_1"
    source = run.parent / "input" / condition
    (source / image).write_bytes(b"synthetic-reader-placeholder")
    ets = source / f"_{Path(image).stem}_" / "stack1"
    ets.mkdir(parents=True)
    (ets / "frame_t_0.ets").write_bytes(b"nonzero")
    per_image = pd.read_csv(run / "per_image_summary.csv")
    extra = per_image.iloc[[0]].copy()
    extra["image"] = image
    pd.concat([per_image, extra], ignore_index=True).to_csv(
        run / "per_image_summary.csv", index=False
    )
    hierarchy = pd.read_csv(hierarchy_path)
    h_extra = hierarchy.iloc[[0]].copy()
    h_extra["image"] = image
    h_extra["image_key"] = image.casefold()
    h_extra["fov"] = int(hierarchy["fov"].max()) + 1
    h_extra["source_vsi"] = str(source / image)
    hierarchy = pd.concat([hierarchy, h_extra], ignore_index=True)
    hierarchy.to_csv(hierarchy_path, index=False)
    hierarchy.to_csv(run / "nuclei_metrics.csv", index=False)
    spots = pd.read_csv(run / "spot_metrics.csv")
    if with_spots:
        s_extra = spots.copy()
        s_extra["image"] = image
        spots = pd.concat([spots, s_extra], ignore_index=True)
    spots.to_csv(run / "spot_metrics.csv", index=False)
    config = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
    config["config_resolved"]["input_file_subset"].append(f"{condition}/{image}")
    (run / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
    import tifffile

    original_mask = next((run / "masks").glob("*__field__nuclei_label_mask.tif"))
    labels = tifffile.imread(original_mask)
    tifffile.imwrite(
        run / "masks" / f"{condition}__{Path(image).stem}__nuclei_label_mask.tif",
        labels,
    )


def test_serial_runner_writes_seven_canonical_outputs_and_resumes(tmp_path, monkeypatch):
    run, hierarchy_path, planes = _write_synthetic_completed_run(tmp_path)
    output = tmp_path / "new-output"
    calls = []
    reconstruction_hierarchies = []
    original_reconstruct = exact_footprint_backfill.reconstruct_image_footprints

    def reconstruct_with_hierarchy_audit(*args, **kwargs):
        hierarchy = kwargs.get("nucleus_hierarchy")
        reconstruction_hierarchies.append(
            None
            if hierarchy is None
            else hierarchy[["image_key", "nucleus_id"]].to_dict("records")
        )
        return original_reconstruct(*args, **kwargs)

    monkeypatch.setattr(
        exact_footprint_backfill,
        "reconstruct_image_footprints",
        reconstruct_with_hierarchy_audit,
    )

    def plane_reader(path, *, selected_z_1based, channel_indices):
        calls.append((Path(path).name, selected_z_1based, dict(channel_indices)))
        return planes, {
            "selected_z_1based": 5, "selected_z_0based": 4, "n_z": 9,
            "voxel_xy_nm": 100.0, "voxel_z_nm": 300.0,
            "height_px": 15, "width_px": 15, "plane_lock_pass": True,
        }

    summary = run_exact_footprint_backfill(
        run,
        hierarchy_path,
        output,
        parameters=ExactFootprintParameters(
            miat_floor_raw=50, n_null=3, max_redraw=20, global_seed=2
        ),
        plane_reader=plane_reader,
        nucleolus_builder=lambda _rc, labels, _dapi, _voxel: np.zeros_like(labels),
        validate_expected_design=False,
    )
    expected = {
        "image_manifest.csv",
        "spot_exact_footprint_metrics.csv.gz",
        "footprint_pixels.csv.gz",
        "exact_footprint_nulls.h5",
        "null_draw_index.csv.gz",
        "nucleus_exact_footprint_metrics.csv",
        "selected_planes_and_masks.h5",
    }
    assert expected.issubset({path.name for path in output.iterdir()})
    assert summary["n_images_complete"] == 1
    assert summary["n_spots"] == 3
    assert calls == [("field.vsi", 5, {"miat": 0, "qki": 1, "dapi": 2})]
    manifest = pd.read_csv(output / "image_manifest.csv")
    assert manifest["phase1_elapsed_s"].notna().all()
    assert manifest["phase2_elapsed_s"].notna().all()
    checkpoint = next((output / "_checkpoints").glob("*/phase2.done.json"))
    telemetry = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert telemetry["phase"] == "phase2"
    assert {"rss_bytes", "output_free_bytes", "n_null_usable_spots"} <= telemetry.keys()
    written_spots = pd.read_csv(output / "spot_exact_footprint_metrics.csv.gz")
    assert len(written_spots) == 3
    assert set(written_spots["population_label_q95"]) <= {
        "threshold_positive", "threshold_negative", "unusable"
    }
    with h5py.File(output / "selected_planes_and_masks.h5", "r") as handle:
        assert handle["images/field.vsi"].attrs["complete"]
    with h5py.File(output / "exact_footprint_nulls.h5", "r") as handle:
        assert handle["images/field.vsi"].attrs["complete"]
    assert reconstruction_hierarchies == [
        [{"image_key": "field.vsi", "nucleus_id": 1}],
        [{"image_key": "field.vsi", "nucleus_id": 1}],
    ]

    run_exact_footprint_backfill(
        run,
        hierarchy_path,
        output,
        parameters=ExactFootprintParameters(
            miat_floor_raw=50, n_null=3, max_redraw=20, global_seed=2
        ),
        resume=True,
        plane_reader=lambda *_args, **_kwargs: pytest.fail("resume reread pixels"),
        nucleolus_builder=lambda _rc, labels, _dapi, _voxel: np.zeros_like(labels),
        validate_expected_design=False,
    )


def test_runner_keeps_zero_spot_image_and_all_its_nuclei_through_exports(tmp_path):
    run, hierarchy_path, planes = _write_synthetic_completed_run(tmp_path)
    _append_synthetic_image(run, hierarchy_path, image="zero.vsi", with_spots=False)

    def plane_reader(_path, *, selected_z_1based, channel_indices):
        return planes, {
            "selected_z_1based": 5, "selected_z_0based": 4, "n_z": 9,
            "voxel_xy_nm": 100.0, "voxel_z_nm": 300.0,
            "height_px": 15, "width_px": 15, "plane_lock_pass": True,
        }

    output = tmp_path / "zero-output"
    run_exact_footprint_backfill(
        run, hierarchy_path, output,
        parameters=ExactFootprintParameters(miat_floor_raw=50, n_null=3),
        plane_reader=plane_reader,
        nucleolus_builder=lambda _rc, labels, _dapi, _voxel: np.zeros_like(labels),
        validate_expected_design=False,
    )
    manifest = pd.read_csv(output / "image_manifest.csv")
    nuclei = pd.read_csv(output / "nucleus_exact_footprint_metrics.csv")
    spots = pd.read_csv(output / "spot_exact_footprint_metrics.csv.gz")
    assert set(manifest["image_key"]) == {"field.vsi", "zero.vsi"}
    assert manifest.set_index("image_key").loc["zero.vsi", "load_status"] == "complete"
    assert len(nuclei.loc[nuclei["image_key"].eq("zero.vsi")]) == 1
    assert not spots["image_key"].eq("zero.vsi").any()
    with h5py.File(output / "selected_planes_and_masks.h5", "r") as handle:
        assert handle["images/zero.vsi"].attrs["complete"]
    with h5py.File(output / "exact_footprint_nulls.h5", "r") as handle:
        assert handle["images/zero.vsi"].attrs["complete"]
        assert len(handle["images/zero.vsi/nuclei"]) == 0


def test_global_parity_barrier_checks_last_image_before_first_null(tmp_path):
    run, hierarchy_path, planes = _write_synthetic_completed_run(tmp_path)
    _append_synthetic_image(run, hierarchy_path, image="field2.vsi", with_spots=True)
    historical = pd.read_csv(run / "spot_metrics.csv")
    last = historical.index[historical["image"].eq("field2.vsi")][-1]
    historical.loc[last, "qki_at_miat_footprint"] += 1
    historical.to_csv(run / "spot_metrics.csv", index=False)
    null_calls = []

    def plane_reader(_path, *, selected_z_1based, channel_indices):
        return planes, {
            "selected_z_1based": 5, "selected_z_0based": 4, "n_z": 9,
            "voxel_xy_nm": 100.0, "voxel_z_nm": 300.0,
            "height_px": 15, "width_px": 15, "plane_lock_pass": True,
        }

    with pytest.raises(ValueError, match="QKI footprint parity"):
        run_exact_footprint_backfill(
            run, hierarchy_path, tmp_path / "barrier-output",
            parameters=ExactFootprintParameters(miat_floor_raw=50, n_null=3),
            plane_reader=plane_reader,
            nucleolus_builder=lambda _rc, labels, _dapi, _voxel: np.zeros_like(labels),
            null_calibrator=lambda *args, **kwargs: null_calls.append(True),
            validate_expected_design=False,
        )
    assert null_calls == []


def test_runner_parity_failure_occurs_before_null_calibration(tmp_path):
    run, hierarchy_path, planes = _write_synthetic_completed_run(tmp_path)
    spots = pd.read_csv(run / "spot_metrics.csv")
    spots.loc[0, "qki_at_miat_footprint"] += 1
    spots.to_csv(run / "spot_metrics.csv", index=False)
    called = []

    def plane_reader(_path, *, selected_z_1based, channel_indices):
        return planes, {
            "selected_z_1based": selected_z_1based,
            "selected_z_0based": selected_z_1based - 1,
            "n_z": 9, "voxel_xy_nm": 100.0, "voxel_z_nm": 300.0,
            "height_px": 15, "width_px": 15, "plane_lock_pass": True,
        }

    with pytest.raises(ValueError, match="QKI footprint parity"):
        run_exact_footprint_backfill(
            run,
            hierarchy_path,
            tmp_path / "failed-output",
            parameters=ExactFootprintParameters(miat_floor_raw=50, n_null=3),
            plane_reader=plane_reader,
            nucleolus_builder=lambda _rc, labels, _dapi, _voxel: np.zeros_like(labels),
            null_calibrator=lambda *args, **kwargs: called.append(True),
            validate_expected_design=False,
        )
    assert called == []


def test_runner_resume_requires_matching_parameter_fingerprint(tmp_path):
    run, hierarchy_path, planes = _write_synthetic_completed_run(tmp_path)

    def plane_reader(_path, *, selected_z_1based, channel_indices):
        return planes, {
            "selected_z_1based": 5, "selected_z_0based": 4, "n_z": 9,
            "voxel_xy_nm": 100.0, "voxel_z_nm": 300.0,
            "height_px": 15, "width_px": 15, "plane_lock_pass": True,
        }

    output = tmp_path / "resume-fingerprint"
    run_exact_footprint_backfill(
        run, hierarchy_path, output,
        parameters=ExactFootprintParameters(miat_floor_raw=50, n_null=3),
        plane_reader=plane_reader,
        nucleolus_builder=lambda _rc, labels, _dapi, _voxel: np.zeros_like(labels),
        validate_expected_design=False,
    )
    with pytest.raises(ValueError, match="parameters do not match"):
        run_exact_footprint_backfill(
            run, hierarchy_path, output,
            parameters=ExactFootprintParameters(miat_floor_raw=50, n_null=4),
            resume=True,
            plane_reader=plane_reader,
            nucleolus_builder=lambda _rc, labels, _dapi, _voxel: np.zeros_like(labels),
            validate_expected_design=False,
        )


def test_resume_rejects_marker_when_selected_h5_group_is_missing(tmp_path):
    run, hierarchy_path, planes = _write_synthetic_completed_run(tmp_path)

    def plane_reader(_path, *, selected_z_1based, channel_indices):
        return planes, {
            "selected_z_1based": 5, "selected_z_0based": 4, "n_z": 9,
            "voxel_xy_nm": 100.0, "voxel_z_nm": 300.0,
            "height_px": 15, "width_px": 15, "plane_lock_pass": True,
        }

    output = tmp_path / "broken-resume"
    params = ExactFootprintParameters(miat_floor_raw=50, n_null=3)
    run_exact_footprint_backfill(
        run, hierarchy_path, output, parameters=params, plane_reader=plane_reader,
        nucleolus_builder=lambda _rc, labels, _dapi, _voxel: np.zeros_like(labels),
        validate_expected_design=False,
    )
    with h5py.File(output / "selected_planes_and_masks.h5", "a") as handle:
        del handle["images/field.vsi"]
    with pytest.raises(RuntimeError, match="phase-1 checkpoint"):
        run_exact_footprint_backfill(
            run, hierarchy_path, output, parameters=params, resume=True,
            plane_reader=plane_reader,
            nucleolus_builder=lambda _rc, labels, _dapi, _voxel: np.zeros_like(labels),
            validate_expected_design=False,
        )


def test_runner_fails_closed_when_nucleolus_reconstruction_is_missing(tmp_path):
    run, hierarchy_path, planes = _write_synthetic_completed_run(tmp_path)

    def plane_reader(_path, *, selected_z_1based, channel_indices):
        return planes, {
            "selected_z_1based": 5, "selected_z_0based": 4, "n_z": 9,
            "voxel_xy_nm": 100.0, "voxel_z_nm": 300.0,
            "height_px": 15, "width_px": 15, "plane_lock_pass": True,
        }

    with pytest.raises(RuntimeError, match="nucleolus reconstruction"):
        run_exact_footprint_backfill(
            run, hierarchy_path, tmp_path / "fail-closed",
            parameters=ExactFootprintParameters(miat_floor_raw=50, n_null=3),
            plane_reader=plane_reader,
            nucleolus_builder=lambda _rc, _labels, _dapi, _voxel: None,
            validate_expected_design=False,
        )


@pytest.mark.parametrize("field_name", ["voxel_xy_nm", "voxel_z_nm"])
def test_runner_rejects_nonfinite_loaded_voxel_metadata(tmp_path, field_name):
    run, hierarchy_path, planes = _write_synthetic_completed_run(tmp_path)

    def plane_reader(_path, *, selected_z_1based, channel_indices):
        metadata = {
            "selected_z_1based": selected_z_1based,
            "selected_z_0based": selected_z_1based - 1,
            "n_z": 9,
            "voxel_xy_nm": 100.0,
            "voxel_z_nm": 300.0,
            "height_px": 15,
            "width_px": 15,
            "plane_lock_pass": True,
        }
        metadata[field_name] = np.nan
        return planes, metadata

    with pytest.raises(ValueError, match=rf"loaded {field_name}.*finite"):
        run_exact_footprint_backfill(
            run,
            hierarchy_path,
            tmp_path / f"nonfinite-{field_name}",
            parameters=ExactFootprintParameters(miat_floor_raw=50, n_null=3),
            plane_reader=plane_reader,
            nucleolus_builder=lambda _rc, labels, _dapi, _voxel: np.zeros_like(
                labels
            ),
            validate_expected_design=False,
        )


def test_mask_resolution_requires_exactly_one_plausible_match(tmp_path):
    masks = tmp_path / "masks"
    masks.mkdir()
    one = masks / "S1_NT_1__field__nuclei_label_mask.tif"
    one.write_bytes(b"one")
    assert resolve_unique_mask_path(masks, "prefix_field.vsi", "S1_NT_1") == one
    two = masks / "other__field__nuclei_label_mask.tif"
    two.write_bytes(b"two")
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_unique_mask_path(masks, "prefix_field.vsi", "S1_NT_1")
    one.unlink()
    two.unlink()
    with pytest.raises(FileNotFoundError, match="no saved nucleus mask"):
        resolve_unique_mask_path(masks, "prefix_field.vsi", "S1_NT_1")


def test_label_mask_reconciliation_is_yx_ordered_and_checks_ids(tmp_path):
    import tifffile

    path = tmp_path / "mask.tif"
    labels = np.zeros((6, 12), dtype=np.uint16)
    labels[2:5, 7:11] = 4
    tifffile.imwrite(path, labels)
    spots = pd.DataFrame({"y_px": [3], "x_px": [9], "nucleus_id": [4]})
    got = read_and_validate_label_mask(
        path, expected_shape=(6, 12), expected_nucleus_ids={4}, spots=spots
    )
    assert got[3, 9] == 4
    with pytest.raises(ValueError, match="recorded parent nucleus"):
        read_and_validate_label_mask(
            path,
            expected_shape=(6, 12),
            expected_nucleus_ids={4},
            spots=pd.DataFrame({"y_px": [9], "x_px": [3], "nucleus_id": [4]}),
        )
    with pytest.raises(ValueError, match="nucleus IDs"):
        read_and_validate_label_mask(
            path, expected_shape=(6, 12), expected_nucleus_ids={4, 5}, spots=spots
        )


def test_fractional_historical_area_cannot_pass_integer_cast_parity():
    result = _synthetic_reconstruction()
    original = pd.DataFrame(
        {
            "image_key": "field.vsi", "channel": "rna1",
            "spot_id": [10, 20, 30],
            "miat_footprint_area_px": [2.5, 2.0, 2.0],
            "qki_at_miat_footprint": [40.0, 80.0, 120.0],
        }
    )
    with pytest.raises(ValueError, match="area parity"):
        validate_historical_parity(result.spot_metrics, original)


def test_zero_spot_reconstruction_has_the_complete_canonical_spot_schema():
    base = _synthetic_reconstruction()
    empty_source = pd.DataFrame(
        columns=[
            "spot_id", "channel", "nucleus_id", "x_px", "y_px",
            "spot_diameter_um", "spot_peak_intensity", "in_nucleolus",
        ]
    )
    empty = reconstruct_image_footprints(
        base.planes["miat"],
        base.planes["qki"],
        base.nucleus_labels,
        base.nucleolus_labels,
        empty_source,
        base.spot_metrics.iloc[0].to_dict(),
        parameters=ExactFootprintParameters(miat_floor_raw=50, n_null=3),
        run_id="run",
        compute_null=False,
    )
    assert list(empty.spot_metrics.columns) == list(base.spot_metrics.columns)
    assert empty.spot_metrics.empty


def test_population_validator_locks_below_floor_aliases_and_monotonic_calls():
    base = _synthetic_reconstruction()
    source = (
        base.spot_metrics[
            [
                "image", "image_key", "condition", "secondary_only", "spot_id",
                "channel", "nucleus_id", "x_px", "y_px",
                "spot_peak_intensity", "peak_intensity", "spot_diameter_um",
                "stored_in_nucleolus",
                "stored_in_nucleus_excluding_nucleolus",
            ]
        ]
        .rename(
            columns={
                "stored_in_nucleolus": "in_nucleolus",
                "stored_in_nucleus_excluding_nucleolus":
                    "in_nucleus_excluding_nucleolus",
            }
        )
        .copy()
    )
    result = reconstruct_image_footprints(
        base.planes["miat"],
        base.planes["qki"],
        base.nucleus_labels,
        base.nucleolus_labels,
        source,
        base.spot_metrics.iloc[0].to_dict(),
        parameters=ExactFootprintParameters(miat_floor_raw=10_000, n_null=3),
        run_id="run",
        compute_null=False,
    )
    for percentile in (90, 95, 99):
        assert result.spot_metrics[f"population_label_q{percentile}"].eq(
            "below_miat_floor"
        ).all()
        assert result.spot_metrics[
            f"qki_threshold_positive_q{percentile}"
        ].isna().all()
    assert result.spot_metrics["population_label"].eq("below_miat_floor").all()
    validate_spot_population_invariants(result.spot_metrics)

    summary = summarize_spot_null_draws(
        [10.0, 20.0], [[1.0, 2.0, 3.0], [5.0, 10.0, 15.0]]
    )
    summary["passes_miat_floor"] = True
    validate_spot_population_invariants(summary)
    corrupt = summary.copy()
    corrupt.loc[0, "qki_threshold_positive_q90"] = False
    corrupt.loc[0, "qki_threshold_positive_q95"] = True
    with pytest.raises(ValueError, match="q99.*q95.*q90|monotonic"):
        validate_spot_population_invariants(corrupt)


def test_resume_rejects_surviving_pending_h5_group(tmp_path):
    run, hierarchy_path, planes = _write_synthetic_completed_run(tmp_path)

    def plane_reader(_path, *, selected_z_1based, channel_indices):
        return planes, {
            "selected_z_1based": selected_z_1based,
            "selected_z_0based": selected_z_1based - 1,
            "n_z": 9,
            "voxel_xy_nm": 100.0,
            "voxel_z_nm": 300.0,
            "height_px": 15,
            "width_px": 15,
            "plane_lock_pass": True,
        }

    output = tmp_path / "pending-resume"
    params = ExactFootprintParameters(miat_floor_raw=50, n_null=3)
    run_exact_footprint_backfill(
        run,
        hierarchy_path,
        output,
        parameters=params,
        plane_reader=plane_reader,
        nucleolus_builder=lambda _rc, labels, _dapi, _voxel: np.zeros_like(labels),
        validate_expected_design=False,
    )
    with h5py.File(output / "selected_planes_and_masks.h5", "a") as handle:
        handle.create_group("images/_pending_field.vsi")
    with pytest.raises(RuntimeError, match="phase-1 checkpoint"):
        run_exact_footprint_backfill(
            run,
            hierarchy_path,
            output,
            parameters=params,
            resume=True,
            plane_reader=plane_reader,
            nucleolus_builder=lambda _rc, labels, _dapi, _voxel: np.zeros_like(labels),
            validate_expected_design=False,
        )


def test_runner_rejects_output_nested_inside_completed_run(tmp_path):
    run, hierarchy_path, _planes = _write_synthetic_completed_run(tmp_path)
    with pytest.raises(ValueError, match="inside.*historical run"):
        run_exact_footprint_backfill(
            run,
            hierarchy_path,
            run / "nested-output",
            validate_expected_design=False,
        )


def test_runner_emits_required_provenance_and_reconciliation_files(tmp_path):
    run, hierarchy_path, planes = _write_synthetic_completed_run(tmp_path)

    def plane_reader(_path, *, selected_z_1based, channel_indices):
        return planes, {
            "selected_z_1based": selected_z_1based,
            "selected_z_0based": selected_z_1based - 1,
            "n_z": 9,
            "n_channels": 3,
            "voxel_xy_nm": 100.0,
            "voxel_z_nm": 300.0,
            "height_px": 15,
            "width_px": 15,
            "plane_dtypes": {"miat": "uint16", "qki": "uint16", "dapi": "uint16"},
            "plane_lock_pass": True,
        }

    output = tmp_path / "provenance-output"
    run_exact_footprint_backfill(
        run,
        hierarchy_path,
        output,
        parameters=ExactFootprintParameters(miat_floor_raw=50, n_null=3),
        plane_reader=plane_reader,
        nucleolus_builder=lambda _rc, labels, _dapi, _voxel: np.zeros_like(labels),
        validate_expected_design=False,
    )
    required = {
        "software_provenance.json",
        "input_checksums.csv",
        "acquisition_metadata.csv",
        "population_reconciliation.csv",
        "data_dictionary.csv",
        "methods_dictionary.csv",
    }
    assert required.issubset({path.name for path in output.iterdir()})
    acquisition = pd.read_csv(output / "acquisition_metadata.csv")
    assert acquisition["laser_power_status"].eq(
        "absent_from_vsi_ome_metadata_unverified"
    ).all()
    reconciliation = pd.read_csv(output / "population_reconciliation.csv")
    assert {"project", "image", "nucleus"}.issubset(
        set(reconciliation["level"])
    )


def test_smoke_subset_preserves_full_roster_but_runs_only_selected_phase1(tmp_path):
    run, hierarchy_path, planes = _write_synthetic_completed_run(tmp_path)
    _append_synthetic_image(run, hierarchy_path, image="zero.vsi", with_spots=False)
    calls = []

    def plane_reader(path, *, selected_z_1based, channel_indices):
        calls.append(Path(path).name)
        return planes, {
            "selected_z_1based": selected_z_1based,
            "selected_z_0based": selected_z_1based - 1,
            "n_z": 9,
            "voxel_xy_nm": 100.0,
            "voxel_z_nm": 300.0,
            "height_px": 15,
            "width_px": 15,
            "plane_lock_pass": True,
        }

    output = tmp_path / "smoke-subset"
    summary = run_exact_footprint_backfill(
        run,
        hierarchy_path,
        output,
        parameters=ExactFootprintParameters(miat_floor_raw=50, n_null=3),
        image_keys=["zero.vsi"],
        phase1_only=True,
        plane_reader=plane_reader,
        nucleolus_builder=lambda _rc, labels, _dapi, _voxel: np.zeros_like(labels),
        null_calibrator=lambda *_args, **_kwargs: pytest.fail(
            "phase1-only smoke must never start null calibration"
        ),
        validate_expected_design=False,
    )

    manifest = pd.read_csv(output / "image_manifest.csv")
    assert set(manifest["image_key"]) == {"field.vsi", "zero.vsi"}
    assert manifest["analysis_scope"].eq("smoke_subset").all()
    assert not manifest["eligible_for_biological_inference_output"].any()
    assert manifest.set_index("image_key").loc["zero.vsi", "selected_for_execution"]
    assert (
        manifest.set_index("image_key").loc["field.vsi", "load_status"]
        == "not_selected_smoke"
    )
    assert calls == ["zero.vsi"]
    assert summary["run_status"] == "phase1_only_complete"
    assert summary["analysis_scope"] == "smoke_subset"
    assert not summary["eligible_for_biological_inference"]
    assert not (output / "exact_footprint_nulls.h5").exists()
    assert not (output / "full_historical_parity_gate.json").exists()
    assert (output / "smoke_subset_historical_parity_gate.json").is_file()
    assert (output / "SMOKE_SUBSET_DO_NOT_USE_FOR_INFERENCE.txt").is_file()
    parameters = json.loads(
        (output / "analysis_parameters.json").read_text(encoding="utf-8")
    )
    assert parameters["selected_image_keys"] == ["zero.vsi"]
    assert parameters["phase1_only"] is True
    assert parameters["analysis_scope"] == "smoke_subset"
    assert parameters["execution_fingerprint"]
    with h5py.File(output / "selected_planes_and_masks.h5", "r") as handle:
        assert set(handle["images"]) == {"zero.vsi"}


def test_full_manifest_is_validated_before_smoke_subset_selection(tmp_path):
    run, hierarchy_path, _planes = _write_synthetic_completed_run(tmp_path)
    output = tmp_path / "must-not-be-created"
    with pytest.raises(ValueError, match="44 images"):
        run_exact_footprint_backfill(
            run,
            hierarchy_path,
            output,
            image_keys=["field.vsi"],
            phase1_only=True,
            validate_expected_design=True,
        )
    assert not output.exists()


def test_cli_accepts_repeated_image_keys_and_phase1_only():
    args = _build_parser().parse_args(
        [
            "--run", "run",
            "--hierarchy", "hierarchy.csv",
            "--output-root", "output",
            "--image-key", "KD_1_17.vsi",
            "--image-key", "control42.vsi",
            "--phase1-only",
        ]
    )
    assert args.image_key == ["KD_1_17.vsi", "control42.vsi"]
    assert args.phase1_only
