from __future__ import annotations

import json

import h5py
import numpy as np
import pandas as pd
import pytest

import fishsuite.core.exact_footprint_backfill as backfill
from test_backfill_native_hierarchy import _execute, native_run


COUNTS = [
    "n_input_spots", "n_eligible_spots", "not_in_nucleus", "centre_off_parent_label",
]


@pytest.fixture
def eligibility_run(native_run):
    run, _, _ = native_run
    path = run / "spot_metrics.csv"
    spots = pd.read_csv(path)
    spots["in_nucleus"] = True
    spots.to_csv(path, index=False)
    return native_run


def _append_spot(run, *, spot_id, y, x, in_nucleus):
    path = run / "spot_metrics.csv"
    spots = pd.read_csv(path)
    extra = spots.loc[spots.image.eq("field.vsi")].iloc[[0]].copy()
    extra["spot_id"] = spot_id
    extra["y_px"], extra["x_px"] = y, x
    extra["in_nucleus"] = in_nucleus
    pd.concat([spots, extra], ignore_index=True).to_csv(path, index=False)


def _mixed_spots(run):
    _append_spot(run, spot_id=101, y=0, x=4, in_nucleus=False)
    _append_spot(run, spot_id=102, y=0, x=4, in_nucleus=True)
    _append_spot(run, spot_id=103, y=4, x=4, in_nucleus=False)


def _assert_accounting(output, expected):
    summary = pd.read_csv(output / "spot_eligibility_summary.csv")
    assert len(summary) == summary.image_key.nunique() == 4
    assert summary[COUNTS].sum().to_dict() == expected
    assert summary.n_input_spots.eq(
        summary.n_eligible_spots + summary.not_in_nucleus + summary.centre_off_parent_label
    ).all()
    provenance = json.loads((output / "analysis_parameters.json").read_text(encoding="utf-8"))
    eligibility = provenance["native_spot_eligibility"]
    assert eligibility["summary_file"] == "spot_eligibility_summary.csv"
    assert eligibility["rule"]
    assert eligibility["totals"] == expected
    return summary


def test_native_excludes_territory_and_edge_spots_with_exclusive_reasons(eligibility_run, tmp_path):
    run, _, _ = eligibility_run
    _mixed_spots(run)
    output = tmp_path / "mixed"
    result = _execute(eligibility_run, output)
    assert result["n_spots"] == 12
    summary = _assert_accounting(output, dict(zip(COUNTS, [15, 12, 2, 1])))
    affected = summary.loc[summary.image_key.str.endswith("/field.vsi")]
    assert affected[COUNTS].iloc[0].tolist() == [6, 3, 2, 1]
    spots = pd.read_csv(output / "spot_exact_footprint_metrics.csv.gz")
    assert set(spots.spot_id) == {10, 20, 30}


def test_native_non_nuclear_unassigned_spot_is_counted_without_parent_assertion(eligibility_run, tmp_path):
    run, _, _ = eligibility_run
    _append_spot(run, spot_id=101, y=0, x=0, in_nucleus=False)
    path = run / "spot_metrics.csv"
    spots = pd.read_csv(path)
    spots.loc[spots.spot_id.eq(101), "nucleus_id"] = 0
    spots.to_csv(path, index=False)
    output = tmp_path / "unassigned"
    result = _execute(eligibility_run, output)
    assert result["n_spots"] == 12
    _assert_accounting(output, dict(zip(COUNTS, [13, 12, 1, 0])))


def test_eligible_footprints_and_null_draws_equal_eligible_only_source(eligibility_run, tmp_path):
    run, _, _ = eligibility_run
    clean_spots = pd.read_csv(run / "spot_metrics.csv")
    _mixed_spots(run)
    mixed, clean = tmp_path / "mixed", tmp_path / "clean"
    _execute(eligibility_run, mixed)
    clean_spots.to_csv(run / "spot_metrics.csv", index=False)
    _execute(eligibility_run, clean)
    for name in (
        "spot_exact_footprint_metrics.csv.gz", "nucleus_exact_footprint_metrics.csv",
        "footprint_pixels.csv.gz",
    ):
        pd.testing.assert_frame_equal(pd.read_csv(mixed / name), pd.read_csv(clean / name))
    with h5py.File(mixed / "exact_footprint_nulls.h5", "r") as left, h5py.File(
        clean / "exact_footprint_nulls.h5", "r"
    ) as right:
        left_datasets, right_datasets = {}, {}
        left.visititems(lambda key, value: left_datasets.update({key: value[()]})
                        if isinstance(value, h5py.Dataset) else None)
        right.visititems(lambda key, value: right_datasets.update({key: value[()]})
                         if isinstance(value, h5py.Dataset) else None)
        assert left_datasets.keys() == right_datasets.keys()
        assert any(key.endswith("draw_qki_raw") for key in left_datasets)
        for key in left_datasets:
            np.testing.assert_array_equal(left_datasets[key], right_datasets[key])


def test_native_claimed_nuclear_spot_over_one_pixel_from_parent_is_hard_error(eligibility_run, tmp_path):
    run, _, _ = eligibility_run
    # Parent starts at (1, 1): this diagonal is sqrt(2), not one pixel away.
    _append_spot(run, spot_id=101, y=0, x=0, in_nucleus=True)
    output = tmp_path / "corrupt"
    with pytest.raises(ValueError, match="in_nucleus|parent"):
        _execute(eligibility_run, output)
    assert not (output / "exact_footprint_nulls.h5").exists()


@pytest.mark.parametrize("invalid", [None, "unknown", 2])
def test_native_rejects_invalid_recorded_nuclear_flag(eligibility_run, tmp_path, invalid):
    run, _, _ = eligibility_run
    path = run / "spot_metrics.csv"
    spots = pd.read_csv(path).astype({"in_nucleus": object})
    spots.loc[0, "in_nucleus"] = invalid
    spots.to_csv(path, index=False)
    with pytest.raises(ValueError, match="in_nucleus|boolean"):
        _execute(eligibility_run, tmp_path / "invalid")


def test_native_rejects_missing_recorded_nuclear_flag(eligibility_run, tmp_path):
    run, _, _ = eligibility_run
    path = run / "spot_metrics.csv"
    pd.read_csv(path).drop(columns="in_nucleus").to_csv(path, index=False)
    with pytest.raises(ValueError, match="in_nucleus"):
        _execute(eligibility_run, tmp_path / "missing")


def test_native_eligibility_uses_ties_to_even_rounding(eligibility_run, tmp_path):
    run, _, _ = eligibility_run
    path = run / "spot_metrics.csv"
    spots = pd.read_csv(path)
    spots.loc[0, "y_px"] = 3.5
    spots.loc[0, "x_px"] = 4.5
    spots.to_csv(path, index=False)
    _append_spot(run, spot_id=101, y=0.5, x=4, in_nucleus=True)
    output = tmp_path / "rounding"
    _execute(eligibility_run, output)
    _assert_accounting(output, dict(zip(COUNTS, [13, 12, 0, 1])))
    reconstructed = pd.read_csv(output / "spot_exact_footprint_metrics.csv.gz")
    assert reconstructed.footprint_area_px.eq(2).all()
    assert 101 not in reconstructed.spot_id.values


def test_image_with_no_eligible_spots_is_accounted_and_completes(eligibility_run, tmp_path):
    run, _, _ = eligibility_run
    path = run / "spot_metrics.csv"
    spots = pd.read_csv(path)
    spots.loc[spots.image.eq("field.vsi"), "in_nucleus"] = False
    spots.to_csv(path, index=False)
    output = tmp_path / "empty-eligible-image"
    result = _execute(eligibility_run, output)
    assert result["n_images_complete"] == 4
    assert result["n_spots"] == 9
    summary = _assert_accounting(output, dict(zip(COUNTS, [12, 9, 3, 0])))
    assert summary.loc[summary.image_key.str.endswith("/field.vsi"), "n_eligible_spots"].item() == 0


def test_resume_preserves_eligibility_accounting(eligibility_run, tmp_path):
    run, _, reader = eligibility_run
    _mixed_spots(run)
    output = tmp_path / "resumed"
    _execute(eligibility_run, output)
    expected = dict(zip(COUNTS, [15, 12, 2, 1]))
    before = _assert_accounting(output, expected)
    backfill.run_exact_footprint_backfill(
        run, None, output, resume=True,
        parameters=backfill.ExactFootprintParameters(
            miat_floor_raw=50, n_null=3, max_redraw=20, global_seed=2,
        ),
        plane_reader=reader,
        nucleolus_builder=lambda _rc, labels, _dapi, _voxel: np.zeros_like(labels),
    )
    pd.testing.assert_frame_equal(before, _assert_accounting(output, expected))


def test_resume_changed_mask_preserves_published_eligibility_files(eligibility_run, tmp_path):
    import tifffile

    run, _, reader = eligibility_run
    output = tmp_path / "changed-mask-resume"
    _execute(eligibility_run, output)
    published = {
        name: (output / name).read_bytes()
        for name in ("analysis_parameters.json", "spot_eligibility_summary.csv")
    }
    first = pd.read_csv(run / "spot_metrics.csv").iloc[0]
    mask_path = run / "masks" / "S1_NT_1__field__nuclei_label_mask.tif"
    labels = tifffile.imread(mask_path)
    y, x = int(np.rint(first.y_px)), int(np.rint(first.x_px))
    assert labels[y, x] == first.nucleus_id
    assert labels[y, x + 1] == first.nucleus_id
    labels[y, x] = 0
    tifffile.imwrite(mask_path, labels)
    with pytest.raises(RuntimeError, match="immutable source inputs changed"):
        backfill.run_exact_footprint_backfill(
            run, None, output, resume=True,
            parameters=backfill.ExactFootprintParameters(
                miat_floor_raw=50, n_null=3, max_redraw=20, global_seed=2,
            ),
            plane_reader=reader,
            nucleolus_builder=lambda _rc, mask, _dapi, _voxel: np.zeros_like(mask),
        )
    for name, original_bytes in published.items():
        assert (output / name).read_bytes() == original_bytes, name


def test_explicit_hierarchy_keeps_strict_parent_assertion(eligibility_run, tmp_path):
    run, hierarchy, _ = eligibility_run
    _append_spot(run, spot_id=101, y=0, x=4, in_nucleus=False)
    output = tmp_path / "historical-invalid"
    with pytest.raises(ValueError, match="recorded parent nucleus"):
        _execute(eligibility_run, output, hierarchy)
    assert not (output / "spot_eligibility_summary.csv").exists()


def test_explicit_hierarchy_has_no_native_eligibility_provenance(eligibility_run, tmp_path):
    output = tmp_path / "historical-valid"
    _execute(eligibility_run, output, eligibility_run[1])
    assert not (output / "spot_eligibility_summary.csv").exists()
    provenance = json.loads((output / "analysis_parameters.json").read_text(encoding="utf-8"))
    assert "native_spot_eligibility" not in provenance
