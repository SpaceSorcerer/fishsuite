from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

import fishsuite.core.exact_footprint_backfill as backfill
from fishsuite.cli import cli
from fishsuite.core.qki_association_postrun import REQUIRED_FILES, run_qki_association
from fishsuite.core.exact_footprint_postrun import run_postrun_directory
from test_exact_footprint_backfill import (
    _append_synthetic_image,
    _write_synthetic_completed_run,
)


@pytest.fixture
def native_run(tmp_path):
    run, explicit_path, planes = _write_synthetic_completed_run(tmp_path)
    original_spots = pd.read_csv(run / "spot_metrics.csv")
    original_spots["in_nucleus"] = True
    images = ["field.vsi", "opaque_b.vsi", "opaque_c.vsi", "opaque_d.vsi"]
    for image in images[1:]:
        _append_synthetic_image(run, explicit_path, image=image, with_spots=False)
    old_hierarchy = pd.read_csv(explicit_path)
    recorded, explicit, spot_frames = [], [], []
    for index, image in enumerate(images):
        condition = ["Control", "Control", "OE", "OE"][index]
        well = ["A01", "A02", "B01", "B02"][index]
        source = old_hierarchy.set_index("image").loc[image, "source_vsi"]
        source_condition = f"recorded_{well}"
        recorded.append(dict(image=image, condition=condition, secondary_only=False,
                             source_path=source, source_condition=source_condition,
                             well_id=well, group=condition, field_id="001",
                             output_stem=f"S1_NT_1__{Path(image).stem}"))
        explicit.append(dict(image=image, image_key=Path(source).resolve().as_posix().casefold(), nucleus_id=1,
                             condition=condition, secondary_only=False, slide=well,
                             arm=condition, source_arm=source_condition, replicate=well,
                             fov="001", biological_set=well, catalog_folder=source_condition,
                             is_control=False, eligible_for_sampling=True,
                             sampled_in_analysis=True, source_vsi=source, well=well,
                             z_mode="single_plane", z_range="5-5", n_z_slices=9,
                             output_stem=f"S1_NT_1__{Path(image).stem}"))
        image_spots = original_spots.copy()
        image_spots["image"] = image
        image_spots["condition"] = condition
        spot_frames.append(image_spots)
    pd.concat(spot_frames, ignore_index=True).to_csv(run / "spot_metrics.csv", index=False)
    pd.DataFrame(recorded).to_csv(run / "resolved_experiment_hierarchy.csv", index=False)
    explicit_frame = pd.DataFrame(explicit)
    explicit_frame.to_csv(explicit_path, index=False)
    explicit_frame[["image", "nucleus_id", "condition", "secondary_only", "slide",
                    "eligible_for_sampling", "sampled_in_analysis", "catalog_folder",
                    "z_mode", "z_range", "n_z_slices"]].to_csv(
        run / "nuclei_metrics.csv", index=False)
    nuclei_path = run / "nuclei_metrics.csv"
    nuclei = pd.read_csv(nuclei_path)
    nuclei["slide"] = "recorded_acquisition_slide"
    nuclei.to_csv(nuclei_path, index=False)
    config_path = run / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["config_resolved"]["sampling"] = {"enabled": True}
    config_path.write_text(json.dumps(config), encoding="utf-8")
    per_image = pd.read_csv(run / "per_image_summary.csv")
    per_image["condition"] = per_image.image.map(explicit_frame.set_index("image").condition)
    per_image.to_csv(run / "per_image_summary.csv", index=False)
    plane_file = run / "synthetic_selected_planes.npz"
    np.savez(plane_file, **planes)

    def reader(_path, *, selected_z_1based, channel_indices):
        assert selected_z_1based == 5
        assert channel_indices == {"miat": 0, "qki": 1, "dapi": 2}
        with np.load(plane_file) as saved:
            loaded = {name: saved[name] for name in saved.files}
        return loaded, dict(selected_z_1based=5, selected_z_0based=4, n_z=9,
                            voxel_xy_nm=100.0, voxel_z_nm=300.0,
                            height_px=15, width_px=15, plane_lock_pass=True)

    return run, explicit_path, reader


def _execute(native_run, output, hierarchy=None):
    run, _, reader = native_run
    return backfill.run_exact_footprint_backfill(
        run, hierarchy, output,
        parameters=backfill.ExactFootprintParameters(miat_floor_raw=50, n_null=3,
                                                     max_redraw=20, global_seed=2),
        plane_reader=reader,
        nucleolus_builder=lambda _rc, labels, _dapi, _voxel: np.zeros_like(labels),
        **({"validate_expected_design": False} if hierarchy is not None else {}),
    )


def test_native_hierarchy_retains_recorded_identifiers_and_flags(native_run):
    run, _, _ = native_run
    hierarchy = backfill.native_hierarchy_from_run(run)
    assert set(hierarchy.condition) == {"Control", "OE"}
    assert set(hierarchy.replicate) == {"A01", "A02", "B01", "B02"}
    assert set(hierarchy.fov) == {"001"}
    assert set(hierarchy.slide) == {"A01", "A02", "B01", "B02"}
    roster = pd.read_csv(run / "resolved_experiment_hierarchy.csv")
    assert set(hierarchy.image_key) == {
        Path(path).resolve().as_posix().casefold() for path in roster.source_path
    }
    assert hierarchy.eligible_for_sampling.all()
    assert hierarchy.sampled_in_analysis.all()
    assert not hierarchy.is_control.any()
    assert len(hierarchy) == 4


def test_native_backfill_completes_and_satisfies_qki_adapter(native_run, tmp_path):
    output = tmp_path / "native-output"
    summary = _execute(native_run, output)
    assert summary["n_images_complete"] == 4
    assert summary["n_spots"] == 12
    assert all((output / name).is_file() for name in REQUIRED_FILES)
    spots = pd.read_csv(output / "spot_exact_footprint_metrics.csv.gz")
    pixels = pd.read_csv(output / "footprint_pixels.csv.gz")
    assert len(spots) == spots.spot_uid.nunique() == pixels.spot_uid.nunique() == 12
    assert spots.footprint_area_px.eq(2).all()
    assert set(spots.condition) == {"Control", "OE"}
    assert set(spots.replicate) == {"A01", "A02", "B01", "B02"}
    provenance = json.loads((output / "analysis_parameters.json").read_text(encoding="utf-8"))
    assert provenance["design_check"] == "not_applicable_native_hierarchy"
    assert provenance["observed_design"] == [4, 4, 0, 4]
    assert provenance["observed_native_design"] == [4, 2, 4]
    assert provenance["slide"] == "well_id (native computation stratum; not acquisition slide)"
    assert provenance["sampling"] == (
        "enabled_in_run; mapped nuclei_metrics.csv.eligible_for_sampling and sampled_in_analysis"
    )
    association = tmp_path / "qki-output"
    run_qki_association(output, association, miat_min=50, qki_min=50, n_null=3, seed=2)
    nuclei = pd.read_csv(association / "qki_association_per_nucleus.csv")
    assert set(nuclei.well) == {"A01", "A02", "B01", "B02"}
    assert set(nuclei.condition) == {"Control", "OE"}


def test_native_and_explicit_hierarchy_produce_identical_footprint_tables(native_run, tmp_path):
    native = tmp_path / "native"
    explicit = tmp_path / "explicit"
    _execute(native_run, native)
    _execute(native_run, explicit, native_run[1])
    for name in ("spot_exact_footprint_metrics.csv.gz", "nucleus_exact_footprint_metrics.csv",
                 "footprint_pixels.csv.gz"):
        pd.testing.assert_frame_equal(pd.read_csv(native / name), pd.read_csv(explicit / name))


@pytest.mark.parametrize("missing", ["nuclei_metrics.csv", "resolved_experiment_hierarchy.csv"])
def test_native_hierarchy_names_missing_required_file(native_run, missing):
    run, _, _ = native_run
    (run / missing).unlink()
    with pytest.raises((ValueError, FileNotFoundError), match=missing):
        backfill.native_hierarchy_from_run(run)


@pytest.mark.parametrize("missing", ["eligible_for_sampling", "sampled_in_analysis"])
def test_native_hierarchy_rejects_unrecorded_required_metadata(native_run, missing):
    run, _, _ = native_run
    path = run / "nuclei_metrics.csv"
    pd.read_csv(path).drop(columns=missing).to_csv(path, index=False)
    with pytest.raises(ValueError, match=missing):
        backfill.native_hierarchy_from_run(run)


def test_native_hierarchy_does_not_parse_well_identifiers_as_numbers(native_run):
    run, _, _ = native_run
    path = run / "resolved_experiment_hierarchy.csv"
    frame = pd.read_csv(path, dtype=str)
    frame["well_id"] = ["001", "002", "003", "004"]
    frame.to_csv(path, index=False)
    hierarchy = backfill.native_hierarchy_from_run(run)
    assert set(hierarchy.replicate) == {"001", "002", "003", "004"}
    assert set(hierarchy.biological_set) == {"001", "002", "003", "004"}


def test_cli_help_and_native_dry_run_create_no_files(native_run, tmp_path):
    run, _, _ = native_run
    before = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}
    runner = CliRunner()
    help_result = runner.invoke(cli, ["footprint-backfill", "--help"])
    assert help_result.exit_code == 0, help_result.output
    for flag in ("--run", "--hierarchy", "--output-root", "--miat-floor", "--n-null",
                 "--seed", "--dry-run"):
        assert flag in help_result.output
    result = runner.invoke(cli, ["footprint-backfill", "--run", str(run), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "observed design" in result.output.lower()
    assert {path.relative_to(tmp_path) for path in tmp_path.rglob("*")} == before
    args = backfill._build_parser().parse_args(["--run", str(run), "--dry-run"])
    assert args.hierarchy is None


@pytest.mark.parametrize("missing", ["miat_footprint_area_px", "qki_at_miat_footprint"])
def test_native_dry_run_rejects_missing_parity_column(native_run, tmp_path, missing):
    run, _, _ = native_run
    path = run / "spot_metrics.csv"
    pd.read_csv(path).drop(columns=missing).to_csv(path, index=False)
    before = {item.relative_to(tmp_path) for item in tmp_path.rglob("*")}
    result = CliRunner().invoke(cli, ["footprint-backfill", "--run", str(run), "--dry-run"])
    assert result.exit_code != 0
    assert missing in result.output or missing in str(result.exception)
    assert (
        "re-run fishsuite with foci.compute_footprint_enrichment: true — "
        "the backfill verifies its reconstructed footprints against these run-time columns"
    ) in result.output
    assert {item.relative_to(tmp_path) for item in tmp_path.rglob("*")} == before


def test_native_dry_run_uses_recorded_source_without_config_subset(native_run, tmp_path):
    run, _, _ = native_run
    config_path = run / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    del config["config_resolved"]["input_file_subset"]
    config_path.write_text(json.dumps(config), encoding="utf-8")
    before = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}
    result = CliRunner().invoke(cli, ["footprint-backfill", "--run", str(run), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert {path.relative_to(tmp_path) for path in tmp_path.rglob("*")} == before


@pytest.mark.parametrize("corruption", ["duplicate", "mismatching_condition"])
def test_native_hierarchy_rejects_duplicate_or_conflicting_roster(native_run, corruption):
    run, _, _ = native_run
    path = run / "resolved_experiment_hierarchy.csv"
    roster = pd.read_csv(path, dtype=str)
    if corruption == "duplicate":
        roster = pd.concat([roster, roster.iloc[[0]]], ignore_index=True)
        error = "duplicate|unique"
    else:
        roster.loc[0, "condition"] = "unrecorded_condition"
        error = "condition|mismatch|disagree"
    roster.to_csv(path, index=False)
    with pytest.raises(ValueError, match=error):
        backfill.native_hierarchy_from_run(run)


def test_native_roster_checksum_and_resume_detect_changed_field(native_run, tmp_path):
    run, _, reader = native_run
    output = tmp_path / "resume-native"
    _execute(native_run, output)
    checksums = pd.read_csv(output / "input_checksums.csv")
    roster_path = run / "resolved_experiment_hierarchy.csv"
    assert roster_path.resolve() in {Path(path).resolve() for path in checksums.path}
    roster = pd.read_csv(roster_path, dtype=str)
    roster.loc[0, "field_id"] = "999"
    roster.to_csv(roster_path, index=False)
    with pytest.raises(ValueError, match="source tables do not match"):
        backfill.run_exact_footprint_backfill(
            run, None, output, resume=True,
            parameters=backfill.ExactFootprintParameters(miat_floor_raw=50, n_null=3,
                                                         max_redraw=20, global_seed=2),
            plane_reader=reader,
            nucleolus_builder=lambda _rc, labels, _dapi, _voxel: np.zeros_like(labels),
        )


def test_native_main_defaults_to_timestamped_sibling(native_run, monkeypatch):
    run, _, _ = native_run
    calls = []

    def capture(source, hierarchy, output, **kwargs):
        calls.append((Path(source), hierarchy, Path(output)))
        return {"run_status": "mocked_no_pixels"}

    monkeypatch.setattr(backfill, "run_exact_footprint_backfill", capture)
    assert backfill.main(["--run", str(run)]) == 0
    assert len(calls) == 1
    source, hierarchy, output = calls[0]
    assert source == run
    assert hierarchy is None
    assert output.parent == run.parent
    assert output.name.startswith("EXACT_FOOTPRINT_BACKFILL_")
    assert not output.exists()


def test_native_enabled_sampling_requires_both_recorded_flags(native_run, tmp_path):
    run, _, _ = native_run
    path = run / "nuclei_metrics.csv"
    missing = ["eligible_for_sampling", "sampled_in_analysis"]
    pd.read_csv(path).drop(columns=missing).to_csv(path, index=False)
    before = {item.relative_to(tmp_path) for item in tmp_path.rglob("*")}
    result = CliRunner().invoke(cli, ["footprint-backfill", "--run", str(run), "--dry-run"])
    assert result.exit_code != 0
    assert all(column in result.output for column in missing), result.output
    assert {item.relative_to(tmp_path) for item in tmp_path.rglob("*")} == before


def test_native_cli_explicit_output_root_allows_child_of_source(native_run, monkeypatch):
    run, _, reader = native_run
    execute = backfill.run_exact_footprint_backfill

    def synthetic_execution(*args, **kwargs):
        return execute(
            *args, **kwargs, plane_reader=reader,
            nucleolus_builder=lambda _rc, labels, _dapi, _voxel: np.zeros_like(labels),
        )

    monkeypatch.setattr(backfill, "run_exact_footprint_backfill", synthetic_execution)
    result = CliRunner().invoke(cli, ["footprint-backfill", "--run", str(run),
                                     "--output-root", str(run), "--miat-floor", "50",
                                     "--n-null", "3", "--seed", "2"])
    assert result.exit_code == 0, f"{result.output}\n{result.exception}"
    outputs = list(run.glob("EXACT_FOOTPRINT_BACKFILL_*"))
    assert len(outputs) == 1
    assert all((outputs[0] / name).is_file() for name in REQUIRED_FILES)


def test_native_hierarchy_rejects_missing_like_sampling_flag(native_run):
    run, _, _ = native_run
    path = run / "nuclei_metrics.csv"
    frame = pd.read_csv(path, dtype=str)
    frame.loc[0, "sampled_in_analysis"] = "nan"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="sampled_in_analysis|boolean"):
        backfill.native_hierarchy_from_run(run)


def test_native_hierarchy_rejects_relative_source_path(native_run):
    run, _, _ = native_run
    path = run / "resolved_experiment_hierarchy.csv"
    frame = pd.read_csv(path, dtype=str)
    frame.loc[0, "source_path"] = "input/field.vsi"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="source_path|absolute"):
        backfill.native_hierarchy_from_run(run)


def test_native_qki_adapter_preserves_numeric_well_identifiers(native_run, tmp_path):
    run, _, _ = native_run
    path = run / "resolved_experiment_hierarchy.csv"
    frame = pd.read_csv(path, dtype=str)
    frame["well_id"] = ["001", "002", "003", "004"]
    frame.to_csv(path, index=False)
    output = tmp_path / "numeric-wells"
    _execute(native_run, output)
    association = tmp_path / "numeric-wells-qki"
    run_qki_association(output, association, miat_min=50, qki_min=50, n_null=3, seed=2)
    nuclei = pd.read_csv(association / "qki_association_per_nucleus.csv", dtype={"well": str})
    assert set(nuclei.well) == {"001", "002", "003", "004"}


def _set_sampling(run, value):
    path = run / "run_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["config_resolved"]["sampling"] = {"enabled": value}
    path.write_text(json.dumps(config), encoding="utf-8")


def test_native_disabled_sampling_derives_flags_and_records_aliases(native_run, tmp_path):
    run, _, _ = native_run
    _set_sampling(run, False)
    path = run / "nuclei_metrics.csv"
    frame = pd.read_csv(path).drop(columns=["slide", "catalog_folder",
                                           "eligible_for_sampling", "sampled_in_analysis"])
    frame.to_csv(path, index=False)
    hierarchy = backfill.native_hierarchy_from_run(run)
    assert hierarchy.eligible_for_sampling.all()
    assert hierarchy.sampled_in_analysis.all()
    assert hierarchy.catalog_folder.eq("not_recorded").all()
    assert set(hierarchy.slide) == {"A01", "A02", "B01", "B02"}
    output = tmp_path / "disabled-sampling"
    _execute(native_run, output)
    for name in ("analysis_parameters.json", "validation_report.json"):
        provenance = json.loads((output / name).read_text(encoding="utf-8"))
        assert provenance["sampling"] == "disabled_in_run; all recorded nuclei eligible and sampled"
        assert provenance["catalog_folder"] == "not_recorded (native hierarchy)"
        assert provenance["slide"] == "well_id (native computation stratum; not acquisition slide)"
        assert provenance["observed_native_design"] == [4, 2, 4]


def test_native_enabled_sampling_preserves_recorded_false_flags(native_run):
    run, _, _ = native_run
    path = run / "nuclei_metrics.csv"
    frame = pd.read_csv(path)
    frame["eligible_for_sampling"] = [True, True, False, False]
    frame["sampled_in_analysis"] = [True, False, False, False]
    frame.to_csv(path, index=False)
    hierarchy = backfill.native_hierarchy_from_run(run).set_index("image")
    assert hierarchy.loc["field.vsi", "sampled_in_analysis"]
    assert not hierarchy.loc["opaque_b.vsi", "sampled_in_analysis"]
    assert hierarchy.loc["opaque_b.vsi", "eligible_for_sampling"]
    assert not hierarchy.loc["opaque_c.vsi", "eligible_for_sampling"]
    assert not hierarchy.loc["opaque_d.vsi", "sampled_in_analysis"]


def test_native_missing_sampling_configuration_fails_without_defaults(native_run):
    run, _, _ = native_run
    path = run / "run_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    del config["config_resolved"]["sampling"]["enabled"]
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="sampling.*enabled|enabled.*sampling"):
        backfill.native_hierarchy_from_run(run)


def test_native_duplicate_source_path_rejected(native_run):
    run, _, _ = native_run
    path = run / "resolved_experiment_hierarchy.csv"
    frame = pd.read_csv(path, dtype=str)
    frame.loc[1, "source_path"] = frame.loc[0, "source_path"]
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate|unique"):
        backfill.native_hierarchy_from_run(run)


def test_native_duplicate_basenames_remain_distinct_through_backfill_and_qki(native_run, tmp_path):
    run, _, _ = native_run
    roster_path = run / "resolved_experiment_hierarchy.csv"
    roster = pd.read_csv(roster_path, dtype=str)
    source_by_image, stem_by_image = {}, {}
    for row in roster.itertuples(index=False):
        directory = tmp_path / "duplicate_sources" / row.well_id
        directory.mkdir(parents=True)
        source = directory / "field.vsi"
        source.write_bytes(b"synthetic-reader-placeholder")
        ets = directory / "_field_" / "stack1"
        ets.mkdir(parents=True)
        (ets / "frame_t_0.ets").write_bytes(b"nonzero")
        source_by_image[row.image] = str(source)
        stem_by_image[row.image] = row.output_stem
    roster["source_path"] = roster.image.map(source_by_image)
    roster["image"] = "field.vsi"
    roster.to_csv(roster_path, index=False)
    for name in ("per_image_summary.csv", "nuclei_metrics.csv", "spot_metrics.csv"):
        path = run / name
        frame = pd.read_csv(path)
        frame["source_path"] = frame.image.map(source_by_image)
        frame["output_stem"] = frame.image.map(stem_by_image)
        frame["image"] = "field.vsi"
        frame.to_csv(path, index=False)
    config_path = run / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["input_dir"] = str(tmp_path / "duplicate_sources")
    config["config_resolved"]["input_file_subset"] = [f"{well}/field.vsi" for well in roster.well_id]
    config_path.write_text(json.dumps(config), encoding="utf-8")
    hierarchy = backfill.native_hierarchy_from_run(run)
    expected_keys = {Path(path).resolve().as_posix().casefold() for path in source_by_image.values()}
    assert set(hierarchy.image_key) == expected_keys
    output = tmp_path / "duplicate-basename-backfill"
    _execute(native_run, output)
    manifest = pd.read_csv(output / "image_manifest.csv")
    spots = pd.read_csv(output / "spot_exact_footprint_metrics.csv.gz")
    assert set(manifest.image_key) == expected_keys
    assert set(spots.image_key) == expected_keys
    assert spots.spot_uid.nunique() == len(spots) == 12
    assert spots.groupby("image_key").size().eq(3).all()
    association = tmp_path / "duplicate-basename-qki"
    run_qki_association(output, association, miat_min=50, qki_min=50, n_null=3, seed=2)
    nuclei = pd.read_csv(association / "qki_association_per_nucleus.csv")
    assert len(nuclei) == 12
    assert set(nuclei.threshold_multiplier) == {0.8, 1.0, 1.25}
    for _, rows in nuclei.groupby("threshold_multiplier"):
        assert len(rows[["image", "nucleus_id"]].drop_duplicates()) == 4
        assert set(rows.image) == expected_keys
    assert set(nuclei.well) == {"A01", "A02", "B01", "B02"}


def test_native_strata_cannot_enter_historical_slide_blocked_statistics(native_run, tmp_path):
    source = tmp_path / "native-for-historical-guard"
    _execute(native_run, source)
    historical_output = tmp_path / "historical-statistics"
    with pytest.raises(ValueError, match="native well_id strata are not acquisition-slide blocks"):
        run_postrun_directory(source, historical_output)
    assert not historical_output.exists()


def test_native_smoke_subset_accepts_normalized_source_path_key(native_run, tmp_path):
    run, _, reader = native_run
    roster = pd.read_csv(run / "resolved_experiment_hierarchy.csv")
    key = Path(roster.iloc[0].source_path).resolve().as_posix().casefold()
    output = tmp_path / "native-smoke-subset"
    summary = backfill.run_exact_footprint_backfill(
        run, None, output, image_keys=[key], phase1_only=True,
        parameters=backfill.ExactFootprintParameters(miat_floor_raw=50, n_null=3),
        plane_reader=reader,
        nucleolus_builder=lambda _rc, labels, _dapi, _voxel: np.zeros_like(labels),
        null_calibrator=lambda *_args, **_kwargs: pytest.fail("smoke phase1 ran null calibration"),
    )
    assert summary["run_status"] == "phase1_only_complete"
    assert summary["selected_image_keys"] == [key]
    assert summary["n_images_phase1_complete"] == 1
    manifest = pd.read_csv(output / "image_manifest.csv")
    assert len(manifest) == 4
    assert manifest.loc[manifest.selected_for_execution, "image_key"].tolist() == [key]


def test_native_duplicate_output_stem_cannot_reuse_saved_mask(native_run, tmp_path):
    run, _, _ = native_run
    path = run / "resolved_experiment_hierarchy.csv"
    roster = pd.read_csv(path, dtype=str)
    assert roster.loc[0, "source_path"] != roster.loc[1, "source_path"]
    roster.loc[1, "output_stem"] = roster.loc[0, "output_stem"]
    roster.to_csv(path, index=False)
    before = {item.relative_to(tmp_path) for item in tmp_path.rglob("*")}
    result = CliRunner().invoke(cli, ["footprint-backfill", "--run", str(run), "--dry-run"])
    assert result.exit_code != 0, result.output
    error = result.output + str(result.exception)
    assert "output_stem" in error
    assert "duplicate" in error.lower() or "unique" in error.lower()
    assert {item.relative_to(tmp_path) for item in tmp_path.rglob("*")} == before


@pytest.mark.parametrize("escaping_stem", ["../outside", r"..\outside"])
def test_native_output_stem_cannot_escape_masks_directory(native_run, tmp_path, escaping_stem):
    run, _, _ = native_run
    path = run / "resolved_experiment_hierarchy.csv"
    roster = pd.read_csv(path, dtype=str)
    roster.loc[0, "output_stem"] = escaping_stem
    roster.to_csv(path, index=False)
    before = {item.relative_to(tmp_path) for item in tmp_path.rglob("*")}
    result = CliRunner().invoke(cli, ["footprint-backfill", "--run", str(run), "--dry-run"])
    assert result.exit_code != 0, result.output
    assert "output_stem" in result.output + str(result.exception)
    assert {item.relative_to(tmp_path) for item in tmp_path.rglob("*")} == before
