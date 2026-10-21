from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fishsuite.core.exact_footprint_common import (
    bool_series,
    fisher_estimable,
    fisher_z,
    median_mad,
    parse_bool,
    robust_distance,
)


@pytest.mark.parametrize(
    "value, expected",
    [(True, True), ("TRUE", True), (" t ", True), ("Yes", True), ("1", True), (1, True),
     (1.0, True), (np.bool_(False), False), ("f", False), ("NO", False), (0, False)],
)
def test_parse_bool_accepts_explicit_tokens(value, expected):
    assert parse_bool(value) is expected


@pytest.mark.parametrize("value", [np.nan, None, pd.NA, "", "nan", "y", "n", 2, "maybe"])
def test_parse_bool_rejects_missing_and_unknown_values(value):
    with pytest.raises(ValueError, match="explicit boolean"):
        parse_bool(value)


def test_bool_series_uses_the_caller_exception_class():
    class Custom(ValueError):
        pass

    with pytest.raises(Custom, match="flag"):
        bool_series(pd.Series([True, np.nan]), name="flag", error=Custom)


def test_fisher_z_is_nan_and_non_estimable_at_the_boundary():
    assert fisher_z(0.5) == pytest.approx(np.arctanh(0.5))
    for r in (1.0, -1.0, 1 - 1e-13, np.nan):
        assert np.isnan(fisher_z(r))
        assert not fisher_estimable(r)
    assert np.isfinite(fisher_z(1 - 1e-11))


def test_robust_distance_zero_mad_policy():
    values = pd.Series([1.0, 1.0, 1.0, 2.0, np.nan])
    median, mad = median_mad(values)
    assert (median, mad) == (1.0, 0.0)
    distance = robust_distance(values, median, mad)
    assert distance[:3].tolist() == [0.0, 0.0, 0.0]
    assert np.isinf(distance[3]) and np.isnan(distance[4])
