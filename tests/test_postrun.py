import json

import numpy as np
import pandas as pd
import pytest


def load_postrun():
    import importlib
    return importlib.import_module("fishsuite.postrun")


def test_resolve_channels_uses_run_metadata_and_partial_overrides(tmp_path):
    postrun = load_postrun()
    config = {
        "config_resolved": {
            "channels": {"one_indexed": True, "rna": 2, "antibody": 1, "dapi": 3},
        },
    }
    (tmp_path / "run_config.json").write_text(json.dumps(config), encoding="utf-8")

    indices, sources = postrun.resolve_channels(
        tmp_path,
        role_keys=("rna", "antibody", "dapi"),
        overrides={"antibody": 4},
    )

    assert indices == {"rna": 1, "antibody": 4, "dapi": 2}
    assert sources == {
        "rna": "run_config.json:config_resolved.channels",
        "antibody": "override",
        "dapi": "run_config.json:config_resolved.channels",
    }


def test_resolve_channels_rejects_missing_or_duplicate_indices(tmp_path):
    postrun = load_postrun()
    with pytest.raises(ValueError, match="cannot resolve.*rna"):
        postrun.resolve_channels(tmp_path, role_keys=("rna",))

    with pytest.raises(ValueError, match="distinct"):
        postrun.resolve_channels(
            tmp_path,
            role_keys=("rna", "antibody"),
            overrides={"rna": 0, "antibody": 0},
        )


def test_normalize_sampling_columns_defaults_only_absent_columns():
    postrun = load_postrun()
    original = pd.DataFrame({"nucleus_id": [1, 2], "eligible_for_sampling": [True, False]})

    normalized, defaulted = postrun.normalize_sampling_columns(original)

    assert defaulted == ("sampled_in_analysis",)
    assert normalized["eligible_for_sampling"].tolist() == [True, False]
    assert normalized["sampled_in_analysis"].tolist() == [True, True]
    assert "sampled_in_analysis" not in original


def test_transform_image_preserves_values_and_checks_rot90_geometry():
    postrun = load_postrun()
    rectangular = np.arange(6).reshape(2, 3)
    assert np.array_equal(postrun.transform_image(rectangular, "none"), rectangular)
    assert np.array_equal(postrun.transform_image(rectangular, "rot180"), rectangular[::-1, ::-1])
    assert np.array_equal(postrun.transform_image(rectangular, "flipud"), rectangular[::-1, :])
    with pytest.raises(ValueError, match="square"):
        postrun.transform_image(rectangular, "rot90")

    square = np.arange(9).reshape(3, 3)
    assert np.array_equal(postrun.transform_image(square, "rot90"), np.rot90(square))


def test_footprint_union_summary_counts_shared_pixels_once():
    postrun = load_postrun()
    image = np.arange(16, dtype=float).reshape(4, 4)
    valid = np.zeros((4, 4), dtype=bool)
    valid[:3, :3] = True
    footprints = [
        np.array([[0, 0], [0, 1], [1, 1]]),
        np.array([[0, 1], [1, 1], [2, 2], [3, 3]]),
        np.array([[3, 3]]),
    ]

    result = postrun.footprint_union_summary(image, valid, footprints)

    assert result == {
        "n_footprints": 3,
        "n_nonempty_footprints": 2,
        "union_px": 4,
        "union_intensity_sum": 16.0,
        "sum_footprint_intensity": 22.0,
        "overlap_px": 2,
    }


def test_footprint_union_summary_keeps_zero_as_a_measured_zero():
    postrun = load_postrun()
    result = postrun.footprint_union_summary(
        np.zeros((2, 2)), np.ones((2, 2), dtype=bool), [],
    )

    assert result["union_px"] == 0
    assert result["union_intensity_sum"] == 0.0
    assert result["n_footprints"] == 0
