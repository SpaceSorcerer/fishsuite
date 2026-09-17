from __future__ import annotations

import json

import h5py
import numpy as np
import pandas as pd
import pytest

import fishsuite.core.exact_footprint_backfill as backfill
from test_backfill_native_hierarchy import _execute, native_run


def test_native_autofocus_search_window_resolves_to_recorded_selected_plane(native_run):
    run, explicit, _ = native_run
    assert pd.read_csv(run / "nuclei_metrics.csv").z_range.eq("2-8").all()
    assert pd.read_csv(explicit).z_range.eq("5-5").all()
    hierarchy = backfill.native_hierarchy_from_run(run)
    assert hierarchy.z_range.eq("5-5").all()
    assert pd.read_csv(run / "nuclei_metrics.csv").z_range.eq("2-8").all()


def test_native_selected_plane_is_resolved_per_image_not_from_first_image(native_run):
    run, _, _ = native_run
    path = run / "per_image_summary.csv"
    per_image = pd.read_csv(path)
    per_image["z_plane"] = [5, 6, 7, 8]
    per_image.to_csv(path, index=False)
    hierarchy = backfill.native_hierarchy_from_run(run)
    expected = dict(zip(per_image.image, ["5-5", "6-6", "7-7", "8-8"]))
    assert hierarchy.set_index("image").z_range.to_dict() == expected


@pytest.mark.parametrize("invalid", [None, "bad", 0, -1, 5.5, 10])
def test_native_rejects_invalid_selected_plane(native_run, invalid):
    run, _, _ = native_run
    path = run / "per_image_summary.csv"
    per_image = pd.read_csv(path).astype({"z_plane": object})
    per_image.loc[0, "z_plane"] = invalid
    per_image.to_csv(path, index=False)
    with pytest.raises(ValueError, match="z_plane|selected z|outside"):
        backfill.native_hierarchy_from_run(run)


def test_native_requires_selected_plane_column(native_run):
    run, _, _ = native_run
    path = run / "per_image_summary.csv"
    pd.read_csv(path).drop(columns="z_plane").to_csv(path, index=False)
    with pytest.raises(ValueError, match="z_plane"):
        backfill.native_hierarchy_from_run(run)


@pytest.mark.parametrize("table,column,value", [
    ("per_image_summary.csv", "n_z", 8),
    ("nuclei_metrics.csv", "n_z_slices", 8),
    ("nuclei_metrics.csv", "n_z_slices", 9.5),
    ("per_image_summary.csv", "n_z", 0),
])
def test_native_rejects_conflicting_or_invalid_stack_depth(native_run, table, column, value):
    run, _, _ = native_run
    path = run / table
    frame = pd.read_csv(path).astype({column: object})
    frame.loc[0, column] = value
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="n_z|depth|slice"):
        backfill.native_hierarchy_from_run(run)


@pytest.mark.parametrize("mode", ["autofocus_maxproj", "maxproj", "3d"])
@pytest.mark.parametrize("record", ["nuclei", "config", "both"])
def test_native_projection_or_3d_mode_rejected_before_pixel_read(native_run, tmp_path, mode, record):
    run, hierarchy, _ = native_run
    if record in {"nuclei", "both"}:
        path = run / "nuclei_metrics.csv"
        nuclei = pd.read_csv(path)
        nuclei["z_mode"] = mode
        nuclei.to_csv(path, index=False)
    if record in {"config", "both"}:
        path = run / "run_config.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        config["config_resolved"]["z_stack"]["mode"] = mode
        path.write_text(json.dumps(config), encoding="utf-8")

    def no_pixels(*_args, **_kwargs):
        pytest.fail("incompatible quantitation mode must fail before reading pixels")

    with pytest.raises(ValueError, match="mode|projection|single.plane|3d|autofocus"):
        _execute((run, hierarchy, no_pixels), tmp_path / "rejected")


def test_native_saved_planes_and_z_indices_match_recorded_selection(native_run, tmp_path):
    run, _, _ = native_run
    output = tmp_path / "selected-plane"
    _execute(native_run, output)
    manifest = pd.read_csv(output / "image_manifest.csv")
    assert manifest.selected_z_1based.eq(5).all()
    assert manifest.selected_z_0based.eq(4).all()
    with np.load(run / "synthetic_selected_planes.npz") as expected, h5py.File(
        output / "selected_planes_and_masks.h5", "r"
    ) as saved:
        assert len(saved["images"]) == 4
        for group in saved["images"].values():
            assert group.attrs["selected_z_1based"] == 5
            assert group.attrs["selected_z_0based"] == 4
            for channel in ("miat", "qki", "dapi"):
                np.testing.assert_array_equal(group[channel][()], expected[channel])


@pytest.mark.parametrize("field,wrong", [("selected_z_1based", 6), ("selected_z_0based", 5)])
def test_native_rejects_reader_changing_selected_plane(native_run, tmp_path, field, wrong):
    run, hierarchy, reader = native_run

    def tampered(*args, **kwargs):
        planes, metadata = reader(*args, **kwargs)
        metadata[field] = wrong
        return planes, metadata

    with pytest.raises(ValueError, match="selected z|1-to-0 z conversion"):
        _execute((run, hierarchy, tampered), tmp_path / "tampered-reader")


@pytest.mark.parametrize("corruption", ["selected_z_1based", "selected_z_0based", "channel_pixel"])
def test_native_checks_saved_cache_immediately_after_write(native_run, tmp_path, monkeypatch, corruption):
    original = backfill._write_selected_image_atomic

    def corrupt_saved_cache(path, **kwargs):
        original(path, **kwargs)
        with h5py.File(path, "a") as saved:
            group = next(group for group in saved["images"].values()
                         if group.attrs["image_key"] == kwargs["image_key"])
            if corruption == "channel_pixel":
                group["miat"][0, 0] = int(group["miat"][0, 0]) + 1
            else:
                group.attrs[corruption] = int(group.attrs[corruption]) + 1

    monkeypatch.setattr(backfill, "_write_selected_image_atomic", corrupt_saved_cache)
    output = tmp_path / "corrupt-written-cache"
    with pytest.raises(ValueError, match="selected|cache|plane"):
        _execute(native_run, output)
    assert not (output / "exact_footprint_nulls.h5").exists()


def test_native_resume_rejects_consistent_but_wrong_saved_plane_indices(native_run, tmp_path):
    run, _, reader = native_run
    output = tmp_path / "resume-corrupt-plane"
    _execute(native_run, output)
    with h5py.File(output / "selected_planes_and_masks.h5", "a") as saved:
        group = next(iter(saved["images"].values()))
        group.attrs["selected_z_1based"] = 6
        group.attrs["selected_z_0based"] = 5
    with pytest.raises(ValueError, match="selected|cache|plane"):
        backfill.run_exact_footprint_backfill(
            run, None, output, resume=True,
            parameters=backfill.ExactFootprintParameters(
                miat_floor_raw=50, n_null=3, max_redraw=20, global_seed=2,
            ),
            plane_reader=reader,
            nucleolus_builder=lambda _rc, labels, _dapi, _voxel: np.zeros_like(labels),
        )
