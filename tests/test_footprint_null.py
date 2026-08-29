"""Focused tests for the exact MIAT-footprint rotation-null engine.

Every expected pixel, intensity, angle, threshold, and count in this file is
hand-derived.  In particular, none of the assertions rebuilds a footprint or
null distribution with production helpers.
"""
from __future__ import annotations

from importlib import import_module

import numpy as np
import pandas as pd
import pytest


def _api():
    return import_module("fishsuite.core.footprint_null")


def _manual_footprint(api, spot_index, center_yx, pixels_yx):
    pixels = np.asarray(pixels_yx, dtype=np.intp)
    center_y, center_x = center_yx
    return api.MiatFootprint(
        spot_index=spot_index,
        center_y_px=center_y,
        center_x_px=center_x,
        y_px=pixels[:, 0],
        x_px=pixels[:, 1],
        dy_px=pixels[:, 0] - center_y,
        dx_px=pixels[:, 1] - center_x,
        method="half_max_component",
        fallback_reason=None,
        full_mask_valid=True,
        invalid_reason=None,
        observed_miat_raw=1.0,
        observed_qki_raw=np.nan,
    )


def _raw_manual_footprint(api, spot_index, center_yx, y_px, x_px):
    center_y, center_x = center_yx
    y_arr = np.asarray(y_px)
    x_arr = np.asarray(x_px)
    return api.MiatFootprint(
        spot_index=spot_index,
        center_y_px=center_y,
        center_x_px=center_x,
        y_px=y_arr,
        x_px=x_arr,
        dy_px=y_arr - center_y,
        dx_px=x_arr - center_x,
        method="half_max_component",
        fallback_reason=None,
        full_mask_valid=True,
        invalid_reason=None,
        observed_miat_raw=1.0,
        observed_qki_raw=np.nan,
    )


def test_build_miat_footprints_preserves_exact_component_and_qki_mean():
    """Catches replacing the connected component with a fixed disk or bbox."""
    api = _api()
    miat = np.zeros((7, 7), dtype=float)
    miat[2:5, 2:5] = 10.0
    qki = np.zeros((7, 7), dtype=float)
    qki[2:5, 2:5] = np.arange(9, dtype=float).reshape(3, 3) * 11.0

    [footprint] = api.build_miat_footprints(
        miat,
        np.array([[3.0, 3.0]]),
        spot_diameter_px=np.array([2.0]),
        partner_2d=qki,
        min_window_half=2,
    )

    assert footprint.method == "half_max_component"
    assert footprint.fallback_reason is None
    assert footprint.area_px == 9
    assert set(zip(footprint.y_px.tolist(), footprint.x_px.tolist())) == {
        (2, 2), (2, 3), (2, 4),
        (3, 2), (3, 3), (3, 4),
        (4, 2), (4, 3), (4, 4),
    }
    assert set(zip(footprint.dy_px.tolist(), footprint.dx_px.tolist())) == {
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 0), (0, 1),
        (1, -1), (1, 0), (1, 1),
    }
    assert footprint.observed_qki_raw == pytest.approx(44.0)


def test_build_miat_footprints_flat_crop_fails_closed_without_disk_pixels():
    """Catches synthetic disk pixels entering an exact-footprint workflow."""
    api = _api()
    [footprint] = api.build_miat_footprints(
        np.full((7, 7), 3.0),
        np.array([[3.0, 3.0]]),
        spot_diameter_px=np.array([2.0]),
        min_window_half=2,
    )

    assert footprint.method == "invalid_empty_footprint"
    assert footprint.fallback_reason == "flat_or_no_contrast"
    assert footprint.area_px == 0
    assert footprint.pixels_yx.shape == (0, 2)
    assert not footprint.full_mask_valid
    assert footprint.invalid_reason == "empty_footprint"
    assert np.isnan(footprint.observed_miat_raw)


def test_build_miat_footprints_tiny_window_fails_closed_without_disk_pixels():
    """Catches the edge-window branch synthesizing radius-derived pixels."""
    api = _api()
    [footprint] = api.build_miat_footprints(
        np.array([[10.0]]),
        np.array([[0.0, 0.0]]),
        spot_diameter_px=np.array([2.0]),
        min_window_half=0,
    )

    assert footprint.method == "invalid_empty_footprint"
    assert footprint.fallback_reason == "window_too_small"
    assert footprint.area_px == 0
    assert footprint.invalid_reason == "empty_footprint"


def test_build_miat_footprints_defensive_zero_seed_component_has_no_disk(
    monkeypatch,
):
    """Catches the defensive seed-label branch reintroducing a fitted disk."""
    api = _api()
    monkeypatch.setattr(
        api.ndimage,
        "label",
        lambda mask, structure: (np.zeros_like(mask, dtype=np.int32), 0),
    )
    miat = np.zeros((7, 7), dtype=float)
    miat[3, 3] = 10.0

    [footprint] = api.build_miat_footprints(
        miat,
        np.array([[3.0, 3.0]]),
        spot_diameter_px=np.array([2.0]),
        min_window_half=2,
    )

    assert footprint.method == "invalid_empty_footprint"
    assert footprint.fallback_reason == "seed_below_half_max"
    assert footprint.area_px == 0
    assert footprint.invalid_reason == "empty_footprint"


def test_footprint_rle_round_trip_preserves_absolute_pixels():
    """Catches lossy or order-dependent footprint serialization."""
    api = _api()
    pixels = np.array([[0, 1], [0, 2], [2, 4], [3, 0], [3, 1]], dtype=int)

    encoded = api.encode_footprint_rle(pixels, image_shape=(4, 5))
    decoded = api.decode_footprint_rle(encoded, image_shape=(4, 5))

    assert encoded.tolist() == [[1, 2], [14, 3]]
    assert decoded.tolist() == [[0, 1], [0, 2], [2, 4], [3, 0], [3, 1]]


def test_footprint_rle_rejects_fractional_coordinates():
    """Catches silent float-to-integer truncation during serialization."""
    api = _api()

    with pytest.raises(ValueError, match="integer-valued"):
        api.encode_footprint_rle(
            np.array([[1.5, 2.0], [2.0, 2.0]]),
            image_shape=(4, 5),
        )


def test_footprint_rle_rejects_duplicate_pixels():
    """Catches silent deduplication that changes footprint cardinality."""
    api = _api()

    with pytest.raises(ValueError, match="duplicate"):
        api.encode_footprint_rle(
            np.array([[1, 2], [1, 2]], dtype=int),
            image_shape=(4, 5),
        )


def test_footprint_rle_decode_rejects_fractional_run_fields():
    """Catches silent truncation of corrupted persisted RLE runs."""
    api = _api()

    with pytest.raises(ValueError, match="integer-valued"):
        api.decode_footprint_rle(
            np.array([[1.5, 2.0]]),
            image_shape=(4, 5),
        )


def test_full_footprint_mask_validity_checks_every_pixel_not_only_center():
    """Catches center-only mask validation."""
    api = _api()
    miat = np.zeros((7, 7), dtype=float)
    miat[2:5, 2:5] = 10.0
    valid_mask = np.ones((7, 7), dtype=bool)
    valid_mask[2, 2] = False  # footprint corner; center remains valid

    [footprint] = api.build_miat_footprints(
        miat,
        np.array([[3.0, 3.0]]),
        spot_diameter_px=np.array([2.0]),
        valid_mask=valid_mask,
        min_window_half=2,
    )

    assert valid_mask[3, 3]
    assert footprint.full_mask_valid is False
    assert footprint.invalid_reason == "footprint_outside_valid_mask"
    assert api.full_footprint_is_valid(footprint, valid_mask) is False


@pytest.mark.parametrize(
    "pixels",
    [
        np.array([[1.5, 2.0]]),
        np.array([[1, 2], [1, 2]], dtype=int),
        np.array([[np.nan, 2.0]]),
        np.array([1, 2], dtype=int),
        np.array([[1, 2, 3]], dtype=int),
    ],
)
def test_full_footprint_is_valid_raw_array_fails_closed_on_malformed_pixels(pixels):
    """Catches lossy casting, duplicate acceptance, and malformed-array raises."""
    api = _api()

    assert api.full_footprint_is_valid(
        pixels, np.ones((5, 5), dtype=bool)
    ) is False


def test_rigid_rotation_rejects_whole_angle_then_redraws_constellation():
    """Catches per-spot repair: the rejected 90-degree angle must leave no draw."""
    api = _api()
    miat = np.zeros((7, 7), dtype=float)
    miat[2, 2] = 10.0
    miat[2, 4] = 10.0
    footprints = api.build_miat_footprints(
        miat,
        np.array([[2.0, 2.0], [2.0, 4.0]]),
        spot_diameter_px=np.array([1.0, 1.0]),
        min_window_half=1,
    )
    qki = np.add.outer(np.arange(7) * 10.0, np.arange(7, dtype=float))
    valid_mask = np.ones((7, 7), dtype=bool)
    valid_mask[1, 3] = False  # 90 degrees invalid for only the right-hand spot

    result = api.rigid_footprint_rotation_null(
        qki,
        footprints,
        valid_mask,
        n_null=1,
        candidate_angles_deg=[90.0, 180.0],
        threshold_percentile=95.0,
        passes_miat_floor=[True, True],
    )

    assert result.usable is True
    assert result.rejected_angle_count == 1
    assert result.accepted_angles_deg.tolist() == [180.0]
    assert result.null_qki_raw.tolist() == [[24.0], [22.0]]
    assert [spot.observed_qki_raw for spot in result.spots] == [22.0, 24.0]
    assert [spot.association_call for spot in result.spots] == [False, True]
    assert [spot.null_p_empirical for spot in result.spots] == [1.0, 0.5]


def test_rigid_rotation_rejects_angle_that_collapses_distinct_pixels():
    """Catches accepting a rasterized footprint with fewer unique pixels."""
    api = _api()
    footprints = [
        _manual_footprint(api, 0, (2, 2), [[0, 0], [1, 0]]),
        _manual_footprint(api, 1, (2, 3), [[2, 3]]),
    ]

    result = api.rigid_footprint_rotation_null(
        np.arange(49, dtype=float).reshape(7, 7),
        footprints,
        np.ones((7, 7), dtype=bool),
        n_null=1,
        candidate_angles_deg=[11.0, 180.0],
        passes_miat_floor=[True, True],
    )

    assert result.rejected_angle_count == 1
    assert result.accepted_angles_deg.tolist() == [180.0]
    assert result.null_qki_raw.shape == (2, 1)


def test_nonfinite_observed_qki_is_unusable_without_poisoning_finite_spot():
    """Catches classifying NaN observed QKI as threshold-negative."""
    api = _api()
    footprints = [
        _manual_footprint(api, 0, (2, 1), [[2, 1]]),
        _manual_footprint(api, 1, (2, 3), [[2, 3]]),
    ]
    qki = np.ones((5, 5), dtype=float)
    qki[2, 1] = np.nan
    qki[2, 3] = 7.0
    qki[1, 2] = 3.0
    qki[3, 2] = 4.0

    result = api.rigid_footprint_rotation_null(
        qki,
        footprints,
        np.ones((5, 5), dtype=bool),
        n_null=1,
        candidate_angles_deg=[90.0],
        passes_miat_floor=[True, True],
    )

    assert result.usable is False
    assert result.invalid_reason == "nonfinite_spot_values"
    assert result.spots[0].null_usable is False
    assert result.spots[0].invalid_reason == "nonfinite_observed_qki"
    assert result.spots[0].association_call is None
    assert np.isnan(result.spots[0].null_p_empirical)
    assert result.spots[1].null_usable is True
    assert result.spots[1].association_call is True


def test_nonfinite_candidate_qki_rejects_whole_angle_then_redraws():
    """Catches accepting an angle that samples NaN partner pixels."""
    api = _api()
    footprints = [
        _manual_footprint(api, 0, (2, 1), [[2, 1]]),
        _manual_footprint(api, 1, (2, 3), [[2, 3]]),
    ]
    qki = np.ones((5, 5), dtype=float)
    qki[1, 2] = np.nan  # sampled by the right spot only at 90 degrees

    result = api.rigid_footprint_rotation_null(
        qki,
        footprints,
        np.ones((5, 5), dtype=bool),
        n_null=1,
        candidate_angles_deg=[90.0, 180.0],
        passes_miat_floor=[True, True],
    )

    assert result.rejected_angle_count == 1
    assert result.accepted_angles_deg.tolist() == [180.0]
    assert all(spot.null_usable for spot in result.spots)


def test_out_of_bounds_observed_footprint_returns_unusable_not_index_error():
    """Catches indexing reconstructed pixels before full-mask validation."""
    api = _api()
    footprints = [
        _manual_footprint(api, 0, (2, 1), [[99, 1]]),
        _manual_footprint(api, 1, (2, 3), [[2, 3]]),
    ]

    result = api.rigid_footprint_rotation_null(
        np.ones((5, 5), dtype=float),
        footprints,
        np.ones((5, 5), dtype=bool),
        n_null=1,
        candidate_angles_deg=[90.0],
        passes_miat_floor=[True, True],
    )

    assert result.usable is False
    assert result.invalid_reason == "invalid_observed_footprint"
    assert result.spots[0].full_mask_valid is False
    assert result.spots[0].invalid_reason == "footprint_out_of_bounds"
    assert np.isnan(result.spots[0].observed_qki_raw)
    assert result.spots[0].association_call is None


def test_negative_observed_pixel_returns_unusable_without_index_wraparound():
    """Catches NumPy negative-index wraparound before validity checking."""
    api = _api()
    footprints = [
        _manual_footprint(api, 0, (2, 1), [[-1, 1]]),
        _manual_footprint(api, 1, (2, 3), [[2, 3]]),
    ]

    result = api.rigid_footprint_rotation_null(
        np.ones((5, 5), dtype=float),
        footprints,
        np.ones((5, 5), dtype=bool),
        n_null=1,
        candidate_angles_deg=[90.0],
        passes_miat_floor=[True, True],
    )

    assert result.invalid_reason == "invalid_observed_footprint"
    assert result.spots[0].invalid_reason == "footprint_out_of_bounds"
    assert np.isnan(result.spots[0].observed_qki_raw)


@pytest.mark.parametrize(
    ("bad_y", "bad_x", "expected_reason"),
    [
        ([2, 3], [2], "footprint_coordinate_length_mismatch"),
        ([2.5], [2.0], "footprint_coordinates_not_integer"),
        ([2, 2], [2, 2], "duplicate_footprint_pixels"),
    ],
)
def test_malformed_observed_pixel_arrays_return_explicit_unusable(
    bad_y, bad_x, expected_reason
):
    """Catches malformed reconstructed footprints reaching array indexing."""
    api = _api()
    footprints = [
        _raw_manual_footprint(api, 0, (2, 2), bad_y, bad_x),
        _manual_footprint(api, 1, (2, 3), [[2, 3]]),
    ]

    result = api.rigid_footprint_rotation_null(
        np.ones((5, 5), dtype=float),
        footprints,
        np.ones((5, 5), dtype=bool),
        n_null=1,
        candidate_angles_deg=[90.0],
        passes_miat_floor=[True, True],
    )

    assert result.invalid_reason == "invalid_observed_footprint"
    assert result.spots[0].invalid_reason == expected_reason
    assert result.spots[0].association_call is None


def test_association_call_requires_observed_strictly_greater_than_own_threshold():
    """Catches >= classification and any pooled rather than per-spot threshold."""
    api = _api()
    miat = np.zeros((5, 5), dtype=float)
    miat[2, 1] = 10.0
    miat[2, 3] = 10.0
    footprints = api.build_miat_footprints(
        miat,
        np.array([[2.0, 1.0], [2.0, 3.0]]),
        spot_diameter_px=np.array([1.0, 1.0]),
        min_window_half=1,
    )
    result = api.rigid_footprint_rotation_null(
        np.full((5, 5), 7.0),
        footprints,
        np.ones((5, 5), dtype=bool),
        n_null=3,
        candidate_angles_deg=[90.0, 180.0, 270.0],
        passes_miat_floor=[True, True],
    )

    spot = result.spots[0]
    assert spot.null_threshold_raw == pytest.approx(7.0)
    assert spot.observed_qki_raw == pytest.approx(7.0)
    assert spot.association_call is False
    assert spot.null_p_empirical == pytest.approx(1.0)


def test_nonconstant_null_pins_linear_q95_strict_call_and_empirical_p():
    """Catches percentile-method drift, pooled thresholds, and wrong p tails."""
    api = _api()
    footprints = [
        _manual_footprint(api, 0, (2, 1), [[2, 1]]),
        _manual_footprint(api, 1, (2, 3), [[2, 3]]),
    ]
    qki = np.ones((5, 5), dtype=float)
    qki[2, 1] = 9.5
    qki[3, 2] = 0.0
    qki[2, 3] = 10.0

    result = api.rigid_footprint_rotation_null(
        qki,
        footprints,
        np.ones((5, 5), dtype=bool),
        n_null=2,
        candidate_angles_deg=[90.0, 180.0],
        passes_miat_floor=[True, True],
    )

    spot = result.spots[0]
    assert result.null_qki_raw[0].tolist() == [0.0, 10.0]
    assert spot.null_threshold_raw == pytest.approx(9.5)
    assert spot.observed_qki_raw == pytest.approx(9.5)
    assert spot.association_call is False
    assert spot.null_p_empirical == pytest.approx(2.0 / 3.0)


def test_one_spot_constellation_is_explicitly_unusable():
    """Catches treating an immobile one-spot rotation as a valid null."""
    api = _api()
    miat = np.zeros((5, 5), dtype=float)
    miat[2, 2] = 10.0
    [footprint] = api.build_miat_footprints(
        miat,
        np.array([[2.0, 2.0]]),
        spot_diameter_px=np.array([1.0]),
        min_window_half=1,
    )

    result = api.rigid_footprint_rotation_null(
        np.full((5, 5), 7.0),
        [footprint],
        np.ones((5, 5), dtype=bool),
        n_null=3,
        candidate_angles_deg=[90.0, 180.0, 270.0],
        passes_miat_floor=[True],
    )

    assert result.usable is False
    assert result.invalid_reason == "insufficient_spots_for_rotation"
    assert result.spots[0].null_usable is False
    assert result.spots[0].association_call is None


def test_population_summaries_partition_floor_rows_and_reconcile_all_rows():
    """Catches dropping unusable rows or counting them as non-associated."""
    api = _api()
    rows = [
        {"passes_miat_floor": True, "null_usable": True,
         "association_call": True, "observed_qki_raw": 10.0},
        {"passes_miat_floor": True, "null_usable": True,
         "association_call": False, "observed_qki_raw": 4.0},
        {"passes_miat_floor": True, "null_usable": False,
         "association_call": None, "observed_qki_raw": np.nan},
        {"passes_miat_floor": False, "null_usable": True,
         "association_call": None, "observed_qki_raw": 2.0},
        {"passes_miat_floor": False, "null_usable": False,
         "association_call": None, "observed_qki_raw": 1.0},
    ]

    summary = api.summarize_footprint_null_populations(rows)

    assert {name: summary[name]["n"] for name in
            ("all", "floor", "assoc", "nonassoc", "unusable")} == {
        "all": 5, "floor": 3, "assoc": 1, "nonassoc": 1, "unusable": 1,
    }
    assert summary["reconciliation"] == {
        "below_floor_n": 2,
        "floor_equals_assoc_plus_nonassoc_plus_unusable": True,
        "all_equals_floor_plus_below_floor": True,
    }
    assert summary["assoc"]["mean_observed_qki_raw"] == pytest.approx(10.0)
    assert summary["nonassoc"]["mean_observed_qki_raw"] == pytest.approx(4.0)


@pytest.mark.parametrize("invalid_call", [None, np.nan, pd.NA, "true", 1])
def test_population_summary_treats_nonboolean_call_as_unusable(invalid_call):
    """Catches truthiness-based classification of CSV/pandas missing values."""
    api = _api()
    rows = [
        {"passes_miat_floor": True, "null_usable": True,
         "association_call": True, "observed_qki_raw": 10.0},
        {"passes_miat_floor": True, "null_usable": True,
         "association_call": False, "observed_qki_raw": 4.0},
        {"passes_miat_floor": True, "null_usable": True,
         "association_call": invalid_call, "observed_qki_raw": 7.0},
    ]

    summary = api.summarize_footprint_null_populations(rows)

    assert summary["assoc"]["n"] == 1
    assert summary["nonassoc"]["n"] == 1
    assert summary["unusable"]["n"] == 1
    assert summary["floor"]["n"] == 3
    assert summary["reconciliation"][
        "floor_equals_assoc_plus_nonassoc_plus_unusable"
    ] is True
