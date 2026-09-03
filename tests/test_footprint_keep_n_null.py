"""Hand-derived tests for the authoritative exact-footprint KEEP-N null."""
from __future__ import annotations

from importlib import import_module

import numpy as np
import pytest


def _api():
    return import_module("fishsuite.core.footprint_null")


def _footprint(api, index, center_yx, pixels_yx):
    pixels = np.asarray(pixels_yx, dtype=np.intp)
    center_y, center_x = center_yx
    return api.MiatFootprint(
        spot_index=index,
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
        observed_miat_raw=10.0,
        observed_qki_raw=np.nan,
    )


def _two_single_pixel_footprints(api):
    return [
        _footprint(api, 0, (2, 1), [[2, 1]]),
        _footprint(api, 1, (2, 3), [[2, 3]]),
    ]


class _SequenceRng:
    """Small deterministic random source for hand-derived redraw angles."""

    def __init__(self, values):
        self._values = iter(float(value) for value in values)

    def uniform(self, low, high, size=None):
        count = 1 if size is None else int(np.prod(size))
        values = [next(self._values) for _ in range(count)]
        if size is None:
            return values[0]
        return np.asarray(values, dtype=float).reshape(size)


def test_keep_n_public_api_is_exported():
    """Catches an implementation that is inaccessible to downstream runners."""
    api = _api()

    assert "KeepNFootprintRotationNullResult" in api.__all__
    assert "keep_n_footprint_rotation_null" in api.__all__


def test_keep_n_shared_valid_angle_places_every_spot_without_redraw():
    """Catches independent initial angles or missing provenance."""
    api = _api()
    result = api.keep_n_footprint_rotation_null(
        np.add.outer(np.arange(5) * 10.0, np.arange(5, dtype=float)),
        _two_single_pixel_footprints(api),
        np.ones((5, 5), dtype=bool),
        n_null=1,
        rng=_SequenceRng([]),
        min_valid_draws=1,
    )

    assert result.initial_angles_deg.tolist() == [90.0]
    assert result.placement_angles_deg.tolist() == [[90.0], [90.0]]
    assert result.first_pass_valid.tolist() == [[True], [True]]
    assert result.first_pass_valid.dtype == np.bool_
    assert result.redraw_counts.tolist() == [[0], [0]]
    assert np.issubdtype(result.redraw_counts.dtype, np.integer)
    assert result.null_qki_raw.tolist() == [[32.0], [12.0]]
    assert result.mean_first_pass_retention == pytest.approx(1.0)
    assert result.median_first_pass_retention == pytest.approx(1.0)
    assert result.unplaceable_count == 0
    assert result.unplaceable_fraction == pytest.approx(0.0)


def test_keep_n_rotates_center_then_translates_asymmetric_exact_footprint():
    """Catches rotation/resampling of the footprint's stored offset geometry."""
    api = _api()
    footprints = [
        _footprint(api, 0, (5, 5), [[5, 5], [5, 6], [6, 5]]),
        _footprint(api, 1, (5, 9), [[5, 9]]),
    ]
    qki = np.add.outer(
        np.arange(12, dtype=float) * 100.0,
        np.arange(12, dtype=float),
    )
    result = api.keep_n_footprint_rotation_null(
        qki,
        footprints,
        np.ones((12, 12), dtype=bool),
        n_null=1,
        rng=_SequenceRng([]),
        min_valid_draws=1,
    )

    # Center (5,5) rotates to (7,7); unchanged offsets
    # {(0,0),(0,1),(1,0)} place pixels {(7,7),(7,8),(8,7)}.
    assert result.null_qki_raw[0, 0] == pytest.approx(
        (707.0 + 708.0 + 807.0) / 3.0
    )
    assert result.placement_angles_deg[0, 0] == pytest.approx(90.0)
    assert result.redraw_counts[0, 0] == 0
    assert (
        result.placement_geometry
        == "rotated_center_translated_exact_footprint"
    )


def test_keep_n_redraws_only_invalid_spot_and_never_drops_it():
    """Catches whole-angle redraw, movement of valid spots, or drop-N behavior."""
    api = _api()
    valid_mask = np.ones((5, 5), dtype=bool)
    valid_mask[1, 2] = False  # right spot invalid at shared 90 degrees
    result = api.keep_n_footprint_rotation_null(
        np.add.outer(np.arange(5) * 10.0, np.arange(5, dtype=float)),
        _two_single_pixel_footprints(api),
        valid_mask,
        n_null=1,
        rng=_SequenceRng([180.0]),
        min_valid_draws=1,
    )

    assert result.initial_angles_deg.tolist() == [90.0]
    assert result.first_pass_valid.tolist() == [[True], [False]]
    assert result.placement_angles_deg.tolist() == [[90.0], [180.0]]
    assert result.redraw_counts.tolist() == [[0], [1]]
    assert result.null_qki_raw.shape == (2, 1)
    assert result.null_qki_raw.tolist() == [[32.0], [21.0]]
    assert np.isfinite(result.null_qki_raw).all()
    assert result.unplaceable_count == 0


def test_keep_n_unplaceable_draw_stays_nan_without_observed_fallback():
    """Catches the legacy observed-position fallback."""
    api = _api()
    valid_mask = np.ones((5, 5), dtype=bool)
    valid_mask[1, 2] = False
    result = api.keep_n_footprint_rotation_null(
        np.add.outer(np.arange(5) * 10.0, np.arange(5, dtype=float)),
        _two_single_pixel_footprints(api),
        valid_mask,
        n_null=1,
        rng=_SequenceRng([90.0, 90.0]),
        max_redraw=2,
        min_valid_draws=1,
    )

    assert result.null_qki_raw.shape == (2, 1)
    assert result.null_qki_raw[0, 0] == pytest.approx(32.0)
    assert np.isnan(result.null_qki_raw[1, 0])
    assert np.isnan(result.placement_angles_deg[1, 0])
    assert result.redraw_counts.tolist() == [[0], [2]]
    assert result.unplaceable_count == 1
    assert result.unplaceable_fraction == pytest.approx(0.5)
    assert result.spots[1].null_usable is False
    assert result.spots[1].association_call is None
    assert result.spots[1].observed_qki_raw == pytest.approx(23.0)


def test_keep_n_defaults_to_all_requested_draws_with_point95_opt_in():
    """Catches making the 95% sensitivity gate the primary default."""
    api = _api()
    valid_mask = np.ones((5, 5), dtype=bool)
    valid_mask[1, 2] = False
    kwargs = dict(
        partner_2d=np.ones((5, 5), dtype=float),
        footprints=_two_single_pixel_footprints(api),
        valid_mask=valid_mask,
        n_null=20,
        max_redraw=0,
        min_valid_draws=2,
    )
    primary = api.keep_n_footprint_rotation_null(
        **kwargs, rng=_SequenceRng([180.0] * 17)
    )
    sensitivity = api.keep_n_footprint_rotation_null(
        **kwargs,
        rng=_SequenceRng([180.0] * 17),
        min_valid_draw_fraction=0.95,
    )

    assert primary.valid_draw_fractions.tolist() == pytest.approx([0.95, 0.95])
    assert all(spot.null_usable is False for spot in primary.spots)
    assert sensitivity.valid_draw_fractions.tolist() == pytest.approx([0.95, 0.95])
    assert all(spot.null_usable is True for spot in sensitivity.spots)


def test_keep_n_low_first_pass_retention_makes_all_calls_unusable():
    """Catches applying per-spot gates or redrawing after the nucleus gate fails."""
    api = _api()
    footprints = [
        _footprint(api, 0, (2, 1), [[2, 1]]),
        _footprint(api, 1, (2, 2), [[2, 2]]),
        _footprint(api, 2, (2, 3), [[2, 3]]),
    ]
    valid_mask = np.ones((5, 5), dtype=bool)
    valid_mask[1, 2] = False
    valid_mask[3, 2] = False
    result = api.keep_n_footprint_rotation_null(
        np.ones((5, 5), dtype=float),
        footprints,
        valid_mask,
        n_null=1,
        rng=_SequenceRng([]),
        min_valid_draws=1,
    )

    assert result.first_pass_valid[:, 0].tolist() == [False, True, False]
    assert result.redraw_counts[:, 0].tolist() == [0, 0, 0]
    assert np.isnan(result.placement_angles_deg[[0, 2], 0]).all()
    assert result.placement_angles_deg[1, 0] == pytest.approx(90.0)
    assert result.median_first_pass_retention == pytest.approx(1.0 / 3.0)
    assert result.unplaceable_count == 0
    assert result.unplaceable_fraction == pytest.approx(0.0)
    assert result.usable is False
    assert result.invalid_reason == "low_first_pass_retention"
    assert all(spot.null_usable is False for spot in result.spots)
    assert all(spot.association_call is None for spot in result.spots)


def test_keep_n_uses_each_spots_own_linear_q95_and_finite_draw_empirical_p():
    """Catches pooled thresholds, non-linear q95, or the wrong empirical tail."""
    api = _api()
    qki = np.ones((5, 5), dtype=float)
    qki[2, 1] = 9.5
    qki[3, 2] = 0.0
    qki[2, 3] = 10.0
    result = api.keep_n_footprint_rotation_null(
        qki,
        _two_single_pixel_footprints(api),
        np.ones((5, 5), dtype=bool),
        n_null=2,
        rng=_SequenceRng([]),
        passes_miat_floor=[True, True],
    )

    assert result.null_qki_raw[0].tolist() == [0.0, 10.0]
    assert result.spots[0].null_threshold_raw == pytest.approx(9.5)
    assert result.spots[0].association_call is False
    assert result.spots[0].null_p_empirical == pytest.approx(2.0 / 3.0)
    assert result.spots[1].association_call is True
    assert result.spots[1].null_p_empirical == pytest.approx(1.0 / 3.0)


def test_keep_n_center_translation_cannot_introduce_raster_collision():
    """Catches reintroducing nearest-rounded per-pixel mask rotation."""
    api = _api()
    footprints = [
        _footprint(api, 0, (5, 5), [[5, 5], [6, 5]]),
        _footprint(api, 1, (15, 15), [[15, 15]]),
    ]
    result = api.keep_n_footprint_rotation_null(
        np.arange(900, dtype=float).reshape(30, 30),
        footprints,
        np.ones((30, 30), dtype=bool),
        n_null=4,
        rng=_SequenceRng([69.0]),
        min_valid_draws=2,
    )

    assert result.initial_angles_deg.tolist() == [90.0, 180.0, 270.0, 69.0]
    assert result.first_pass_valid[:, 3].tolist() == [True, True]
    assert result.placement_angles_deg[:, 3].tolist() == [69.0, 69.0]
    assert result.redraw_counts[:, 3].tolist() == [0, 0]
    assert np.isfinite(result.null_qki_raw[:, 3]).all()


def test_keep_n_nonfinite_qki_redraws_only_affected_spot():
    """Catches accepting NaN samples or redrawing an unaffected sibling."""
    api = _api()
    qki = np.ones((5, 5), dtype=float)
    qki[1, 2] = np.nan
    result = api.keep_n_footprint_rotation_null(
        qki,
        _two_single_pixel_footprints(api),
        np.ones((5, 5), dtype=bool),
        n_null=1,
        rng=_SequenceRng([180.0]),
        min_valid_draws=1,
    )

    assert result.first_pass_valid.tolist() == [[True], [False]]
    assert result.placement_angles_deg.tolist() == [[90.0], [180.0]]
    assert result.redraw_counts.tolist() == [[0], [1]]
    assert np.isfinite(result.null_qki_raw).all()


def test_keep_n_seeded_rng_is_deterministic():
    """Catches use of global RNG state or unstable draw ordering."""
    api = _api()
    kwargs = dict(
        partner_2d=np.arange(49, dtype=float).reshape(7, 7),
        footprints=[
            _footprint(api, 0, (3, 2), [[3, 2]]),
            _footprint(api, 1, (3, 4), [[3, 4]]),
        ],
        valid_mask=np.ones((7, 7), dtype=bool),
        n_null=5,
        min_valid_draws=2,
    )
    first = api.keep_n_footprint_rotation_null(
        **kwargs, rng=np.random.default_rng(42)
    )
    second = api.keep_n_footprint_rotation_null(
        **kwargs, rng=np.random.default_rng(42)
    )

    assert np.array_equal(first.initial_angles_deg, second.initial_angles_deg)
    assert np.array_equal(first.placement_angles_deg, second.placement_angles_deg)
    assert np.array_equal(first.first_pass_valid, second.first_pass_valid)
    assert np.array_equal(first.redraw_counts, second.redraw_counts)
    assert np.array_equal(first.null_qki_raw, second.null_qki_raw)


def test_keep_n_one_spot_is_unusable_with_deterministic_shapes():
    """Catches treating an immobile one-spot constellation as a valid null."""
    api = _api()
    result = api.keep_n_footprint_rotation_null(
        np.ones((5, 5), dtype=float),
        [_footprint(api, 0, (2, 2), [[2, 2]])],
        np.ones((5, 5), dtype=bool),
        n_null=2,
        rng=_SequenceRng([]),
    )

    assert result.usable is False
    assert result.invalid_reason == "insufficient_spots_for_rotation"
    assert result.null_qki_raw.shape == (1, 2)
    assert np.isnan(result.null_qki_raw).all()
    assert result.placement_angles_deg.shape == (1, 2)
    assert np.isnan(result.placement_angles_deg).all()
    assert result.first_pass_valid.shape == (1, 2)
    assert not result.first_pass_valid.any()
    assert result.redraw_counts.tolist() == [[0, 0]]
    assert result.spots[0].association_call is None


def test_keep_n_rejects_min_valid_draws_above_requested_draws():
    """Catches a caller typo that would otherwise make every spot unusable."""
    api = _api()
    with pytest.raises(ValueError, match="min_valid_draws cannot exceed n_null"):
        api.keep_n_footprint_rotation_null(
            np.ones((5, 5), dtype=float),
            _two_single_pixel_footprints(api),
            np.ones((5, 5), dtype=bool),
            n_null=2,
            min_valid_draws=3,
        )


def test_keep_n_explicit_none_placement_masks_is_byte_identical_to_legacy():
    """Catches the optional extension changing the established default path."""
    api = _api()
    kwargs = dict(
        partner_2d=np.arange(49, dtype=float).reshape(7, 7),
        footprints=[
            _footprint(api, 0, (3, 2), [[3, 2], [4, 2]]),
            _footprint(api, 1, (3, 4), [[3, 4]]),
        ],
        valid_mask=np.ones((7, 7), dtype=bool),
        n_null=7,
        min_valid_draws=2,
    )
    legacy = api.keep_n_footprint_rotation_null(
        **kwargs,
        rng=np.random.default_rng(12345),
    )
    explicit_none = api.keep_n_footprint_rotation_null(
        **kwargs,
        rng=np.random.default_rng(12345),
        placement_masks=None,
    )

    for field in (
        "null_qki_raw",
        "initial_angles_deg",
        "placement_angles_deg",
        "first_pass_valid",
        "redraw_counts",
        "valid_draw_counts",
        "valid_draw_fractions",
    ):
        legacy_value = getattr(legacy, field)
        explicit_value = getattr(explicit_none, field)
        assert explicit_value.tobytes() == legacy_value.tobytes()
        assert np.array_equal(explicit_value, legacy_value, equal_nan=True)
    for field in (
        "placement_geometry",
        "mean_first_pass_retention",
        "median_first_pass_retention",
        "unplaceable_count",
        "unplaceable_fraction",
        "usable",
        "invalid_reason",
    ):
        assert getattr(explicit_none, field) == getattr(legacy, field)
    for explicit_spot, legacy_spot in zip(
        explicit_none.spots, legacy.spots, strict=True
    ):
        assert explicit_spot.spot_index == legacy_spot.spot_index
        assert explicit_spot.null_usable is legacy_spot.null_usable
        assert explicit_spot.association_call is legacy_spot.association_call
        assert explicit_spot.null_threshold_raw == pytest.approx(
            legacy_spot.null_threshold_raw
        )
        assert explicit_spot.null_p_empirical == pytest.approx(
            legacy_spot.null_p_empirical
        )


def test_keep_n_uses_each_footprints_mask_for_initial_and_redraw_placements():
    """Catches pooled domains or using the wrong spot's mask during redraw."""
    api = _api()
    first_mask = np.zeros((5, 5), dtype=bool)
    first_mask[3, 2] = True  # left spot at shared 90-degree angle
    second_mask = np.zeros((5, 5), dtype=bool)
    second_mask[2, 1] = True  # right spot only after 180-degree redraw

    result = api.keep_n_footprint_rotation_null(
        np.add.outer(np.arange(5) * 10.0, np.arange(5, dtype=float)),
        _two_single_pixel_footprints(api),
        np.ones((5, 5), dtype=bool),
        n_null=1,
        rng=_SequenceRng([180.0]),
        placement_masks=[first_mask, second_mask],
        min_valid_draws=1,
    )

    assert result.initial_angles_deg.tolist() == [90.0]
    assert result.first_pass_valid.tolist() == [[True], [False]]
    assert result.placement_angles_deg.tolist() == [[90.0], [180.0]]
    assert result.redraw_counts.tolist() == [[0], [1]]
    assert result.null_qki_raw.tolist() == [[32.0], [21.0]]


def test_keep_n_per_footprint_empty_domain_marks_only_that_spot_unplaceable():
    """Catches dropping the spot, falling back to observed, or failing siblings."""
    api = _api()
    first_mask = np.zeros((5, 5), dtype=bool)
    first_mask[3, 2] = True
    second_mask = np.zeros((5, 5), dtype=bool)

    result = api.keep_n_footprint_rotation_null(
        np.add.outer(np.arange(5) * 10.0, np.arange(5, dtype=float)),
        _two_single_pixel_footprints(api),
        np.ones((5, 5), dtype=bool),
        n_null=1,
        rng=_SequenceRng([180.0]),
        placement_masks=[first_mask, second_mask],
        max_redraw=1,
        min_valid_draws=1,
    )

    assert result.null_qki_raw[0, 0] == pytest.approx(32.0)
    assert np.isnan(result.null_qki_raw[1, 0])
    assert result.redraw_counts.tolist() == [[0], [1]]
    assert result.unplaceable_count == 1
    assert result.spots[0].null_usable is True
    assert result.spots[1].null_usable is False
    assert result.spots[1].invalid_reason == "insufficient_valid_null_draws"
    assert result.spots[1].association_call is None
    assert result.usable is True


@pytest.mark.parametrize(
    ("placement_masks", "message"),
    [
        ([np.ones((5, 5), dtype=bool)], "one mask per footprint"),
        (
            [
                np.ones((5, 5), dtype=bool),
                np.ones((4, 5), dtype=bool),
            ],
            "same shape as partner_2d",
        ),
    ],
)
def test_keep_n_rejects_invalid_per_footprint_mask_sequences(
    placement_masks, message
):
    """Catches silent mask reuse when compartment metadata are malformed."""
    api = _api()

    with pytest.raises(ValueError, match=message):
        api.keep_n_footprint_rotation_null(
            np.ones((5, 5), dtype=float),
            _two_single_pixel_footprints(api),
            np.ones((5, 5), dtype=bool),
            n_null=1,
            placement_masks=placement_masks,
            min_valid_draws=1,
        )
