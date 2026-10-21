"""Small helpers shared by the exact-footprint backfill, statistics and figures."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

_TRUE_TOKENS = frozenset({"true", "t", "yes", "1"})
_FALSE_TOKENS = frozenset({"false", "f", "no", "0"})

# Correlations this close to +/-1 have no finite Fisher z.
FISHER_Z_BOUNDARY = 1.0 - 1e-12


def parse_bool(value: Any, *, name: str = "value", error: type[Exception] = ValueError) -> bool:
    """Parse one explicit boolean; missing or unrecognised values raise."""

    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer, float, np.floating)) and not isinstance(value, bool):
        if np.isfinite(value) and value in (0, 1):
            return bool(int(value))
        raise error(f"{name} is not an explicit boolean: {value!r}")
    if isinstance(value, str):
        token = value.strip().casefold()
        if token in _TRUE_TOKENS:
            return True
        if token in _FALSE_TOKENS:
            return False
    raise error(f"{name} is not an explicit boolean: {value!r}")


def bool_series(values: pd.Series, *, name: str, error: type[Exception] = ValueError) -> pd.Series:
    """Parse a column of explicit booleans with :func:`parse_bool`."""

    return values.map(lambda value: parse_bool(value, name=name, error=error)).astype(bool)


def require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    table: str,
    error: type[Exception] = ValueError,
) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise error(f"{table} is missing required columns: {missing}")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def median_mad(values: pd.Series | np.ndarray) -> tuple[float, float]:
    """Median and unscaled median absolute deviation of the finite values."""

    array = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
    array = array[np.isfinite(array)]
    if not array.size:
        return np.nan, np.nan
    median = float(np.median(array))
    return median, float(np.median(np.abs(array - median)))


def robust_distance(values: pd.Series | np.ndarray, median: float, mad: float) -> np.ndarray:
    """|value - median| / MAD; NaN for non-finite values.

    With zero MAD, values equal to the median are at distance 0 and every other
    finite value is infinitely far.
    """

    array = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
    out = np.full(array.shape, np.nan, dtype=float)
    if not (np.isfinite(median) and np.isfinite(mad)):
        return out
    finite = np.isfinite(array)
    deviation = np.abs(array[finite] - median)
    if mad > 0:
        out[finite] = deviation / mad
    else:
        out[finite] = np.where(deviation == 0, 0.0, np.inf)
    return out


def fisher_estimable(values: Any) -> Any:
    """True where a correlation is finite and strictly inside the boundary."""

    r = np.asarray(values, dtype=float)
    result = np.isfinite(r) & (np.abs(np.where(np.isfinite(r), r, 0.0)) < FISHER_Z_BOUNDARY)
    return bool(result) if result.ndim == 0 else result


def fisher_z(values: Any) -> Any:
    """Fisher z of a correlation; NaN at or beyond the boundary."""

    r = np.asarray(values, dtype=float)
    estimable = np.asarray(fisher_estimable(r))
    z = np.full(r.shape, np.nan, dtype=float)
    z[estimable] = np.arctanh(r[estimable])
    return float(z) if z.ndim == 0 else z


def q95_call_masks(spots: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (usable, positive, negative) q95 masks aligned to ``spots``.

    Calls are read only for null-usable spots; unusable spots are neither
    positive nor negative.
    """

    usable = (
        bool_series(spots["null_usable"], name="null_usable")
        if len(spots)
        else pd.Series(False, index=spots.index, dtype=bool)
    )
    calls = pd.Series(False, index=spots.index, dtype=bool)
    if usable.any():
        calls.loc[usable] = bool_series(
            spots.loc[usable, "qki_threshold_positive_q95"],
            name="qki_threshold_positive_q95",
        )
    return usable, usable & calls, usable & ~calls
