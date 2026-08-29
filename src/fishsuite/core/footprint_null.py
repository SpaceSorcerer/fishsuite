"""Pure exact-footprint construction and rigid rotation-null calculations.

The exact raster pixels used for each MIAT spot stay attached to construction
provenance. A rigid-null candidate is accepted only when every transformed
pixel of every footprint is inside the supplied validity mask. The older
fixed-disk rotation null in fishsuite.core.modes.rna_rna is intentionally not
imported or changed here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class MiatFootprint:
    """One exact MIAT footprint and its construction provenance."""

    spot_index: int
    center_y_px: int
    center_x_px: int
    y_px: np.ndarray
    x_px: np.ndarray
    dy_px: np.ndarray
    dx_px: np.ndarray
    method: str
    fallback_reason: str | None
    full_mask_valid: bool
    invalid_reason: str | None
    observed_miat_raw: float
    observed_qki_raw: float

    @property
    def area_px(self) -> int:
        return int(self.y_px.size)

    @property
    def center_yx(self) -> tuple[int, int]:
        return self.center_y_px, self.center_x_px

    @property
    def pixels_yx(self) -> np.ndarray:
        if self.area_px == 0:
            return np.empty((0, 2), dtype=np.intp)
        return np.column_stack((self.y_px, self.x_px))


@dataclass(frozen=True)
class FootprintNullSpotResult:
    """Per-spot observed value, own null calibration, and classification."""

    spot_index: int
    center_y_px: int
    center_x_px: int
    y_px: np.ndarray
    x_px: np.ndarray
    dy_px: np.ndarray
    dx_px: np.ndarray
    method: str
    fallback_reason: str | None
    full_mask_valid: bool
    invalid_reason: str | None
    observed_qki_raw: float
    null_threshold_raw: float
    null_p_empirical: float
    null_usable: bool
    association_call: bool | None
    passes_miat_floor: bool

    @property
    def area_px(self) -> int:
        return int(self.y_px.size)


@dataclass(frozen=True)
class FootprintRotationNullResult:
    """Strict rigid-null output for one nucleus's MIAT constellation."""

    spots: tuple[FootprintNullSpotResult, ...]
    null_qki_raw: np.ndarray
    accepted_angles_deg: np.ndarray
    rejected_angle_count: int
    usable: bool
    invalid_reason: str | None


@dataclass(frozen=True)
class KeepNFootprintRotationNullResult:
    """Authoritative exact-footprint KEEP-N null for one nucleus."""

    spots: tuple[FootprintNullSpotResult, ...]
    placement_geometry: str
    null_qki_raw: np.ndarray
    initial_angles_deg: np.ndarray
    placement_angles_deg: np.ndarray
    first_pass_valid: np.ndarray
    redraw_counts: np.ndarray
    valid_draw_counts: np.ndarray
    valid_draw_fractions: np.ndarray
    mean_first_pass_retention: float
    median_first_pass_retention: float
    unplaceable_count: int
    unplaceable_fraction: float
    usable: bool
    invalid_reason: str | None


def _as_spot_yx(spot_yx: Any) -> np.ndarray:
    arr = np.asarray(spot_yx, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("spot_yx must have shape (n_spots, 2)")
    if not np.isfinite(arr).all():
        raise ValueError("spot_yx must contain only finite coordinates")
    return arr


def _as_diameters(
    spot_diameter_px: Any,
    n_spots: int,
    default_spot_diameter_px: float,
) -> np.ndarray:
    if not np.isfinite(default_spot_diameter_px) or default_spot_diameter_px <= 0:
        raise ValueError("default_spot_diameter_px must be finite and positive")
    if spot_diameter_px is None:
        return np.full(n_spots, float(default_spot_diameter_px), dtype=np.float64)
    raw = np.asarray(spot_diameter_px, dtype=np.float64)
    if raw.ndim == 0:
        out = np.full(n_spots, float(raw), dtype=np.float64)
    else:
        out = raw.reshape(-1).copy()
        if out.size != n_spots:
            raise ValueError("spot_diameter_px must be scalar or one value per spot")
    out[~np.isfinite(out) | (out <= 0)] = float(default_spot_diameter_px)
    return out


def _validated_pixel_arrays(
    y_px: np.ndarray,
    x_px: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool, str | None]:
    y_raw = np.asarray(y_px)
    x_raw = np.asarray(x_px)
    if y_raw.ndim != 1 or x_raw.ndim != 1:
        empty = np.empty(0, dtype=np.intp)
        return empty, empty, False, "footprint_coordinates_not_1d"
    if y_raw.size != x_raw.size:
        empty = np.empty(0, dtype=np.intp)
        return empty, empty, False, "footprint_coordinate_length_mismatch"
    if y_raw.size == 0:
        empty = np.empty(0, dtype=np.intp)
        return empty, empty, False, "empty_footprint"
    for raw in (y_raw, x_raw):
        if not (
            np.issubdtype(raw.dtype, np.integer)
            or np.issubdtype(raw.dtype, np.floating)
        ):
            empty = np.empty(0, dtype=np.intp)
            return empty, empty, False, "footprint_coordinates_not_integer"
        if np.issubdtype(raw.dtype, np.floating):
            if not bool(np.isfinite(raw).all()):
                empty = np.empty(0, dtype=np.intp)
                return empty, empty, False, "footprint_coordinates_not_finite"
            if not bool((raw == np.rint(raw)).all()):
                empty = np.empty(0, dtype=np.intp)
                return empty, empty, False, "footprint_coordinates_not_integer"
    try:
        y_int = _require_integer_valued(y_raw, "footprint coordinates").astype(
            np.intp, copy=False
        )
        x_int = _require_integer_valued(x_raw, "footprint coordinates").astype(
            np.intp, copy=False
        )
    except ValueError:
        empty = np.empty(0, dtype=np.intp)
        return empty, empty, False, "footprint_coordinates_not_integer"
    pixels = np.column_stack((y_int, x_int))
    if np.unique(pixels, axis=0).shape[0] != pixels.shape[0]:
        return y_int, x_int, False, "duplicate_footprint_pixels"
    height, width = valid_mask.shape
    in_bounds = (
        (y_int >= 0) & (y_int < height) & (x_int >= 0) & (x_int < width)
    )
    if not bool(in_bounds.all()):
        return y_int, x_int, False, "footprint_out_of_bounds"
    if not bool(valid_mask[y_int, x_int].all()):
        return y_int, x_int, False, "footprint_outside_valid_mask"
    return y_int, x_int, True, None


def _mask_validity(
    y_px: np.ndarray,
    x_px: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[bool, str | None]:
    _y_int, _x_int, valid, reason = _validated_pixel_arrays(
        y_px, x_px, valid_mask
    )
    return valid, reason


def full_footprint_is_valid(
    footprint: MiatFootprint | np.ndarray,
    valid_mask: np.ndarray,
) -> bool:
    """Return whether every footprint pixel is in-bounds and mask-valid."""
    mask = np.asarray(valid_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("valid_mask must be two-dimensional")
    if isinstance(footprint, MiatFootprint):
        y_px, x_px = footprint.y_px, footprint.x_px
    else:
        pixels = np.asarray(footprint)
        if pixels.size == 0:
            return False
        if pixels.ndim != 2 or pixels.shape[1] != 2:
            return False
        y_px = pixels[:, 0]
        x_px = pixels[:, 1]
    valid, _reason = _mask_validity(y_px, x_px, mask)
    return valid


def build_miat_footprints(
    miat_2d: np.ndarray,
    spot_yx: Any,
    spot_diameter_px: Any = None,
    *,
    default_spot_diameter_px: float = 1.0,
    partner_2d: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    half_max_frac: float = 0.5,
    bg_percentile: float = 10.0,
    window_pad_px: int = 2,
    min_window_half: int = 4,
    max_window_half: int = 12,
    peak_search_half: int = 1,
) -> list[MiatFootprint]:
    """Construct exact half-max components and fail closed when unavailable."""
    miat = np.asarray(miat_2d)
    if miat.ndim != 2:
        raise ValueError("miat_2d must be two-dimensional")
    height, width = miat.shape
    spots = _as_spot_yx(spot_yx)
    diameters = _as_diameters(
        spot_diameter_px, spots.shape[0], default_spot_diameter_px
    )
    partner = None if partner_2d is None else np.asarray(partner_2d)
    if partner is not None and partner.shape != miat.shape:
        raise ValueError("partner_2d must have the same shape as miat_2d")
    if valid_mask is None:
        mask = np.ones(miat.shape, dtype=bool)
    else:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != miat.shape:
            raise ValueError("valid_mask must have the same shape as miat_2d")
    if not (0.0 < float(half_max_frac) <= 1.0):
        raise ValueError("half_max_frac must be in (0, 1]")
    if not (0.0 <= float(bg_percentile) <= 100.0):
        raise ValueError("bg_percentile must be in [0, 100]")
    if min_window_half < 0 or max_window_half < min_window_half:
        raise ValueError("window half-width bounds are invalid")
    if peak_search_half < 0:
        raise ValueError("peak_search_half must be non-negative")

    miat_f = miat.astype(np.float64, copy=False)
    partner_f = None if partner is None else partner.astype(np.float64, copy=False)
    structure8 = np.ones((3, 3), dtype=bool)
    output: list[MiatFootprint] = []

    for spot_index, ((spot_y, spot_x), diameter_px) in enumerate(zip(spots, diameters)):
        center_y = int(np.clip(np.rint(spot_y), 0, height - 1))
        center_x = int(np.clip(np.rint(spot_x), 0, width - 1))
        window_half = int(np.clip(
            round(float(diameter_px)) + int(window_pad_px),
            int(min_window_half),
            int(max_window_half),
        ))
        y0 = max(0, center_y - window_half)
        y1 = min(height, center_y + window_half + 1)
        x0 = max(0, center_x - window_half)
        x1 = min(width, center_x + window_half + 1)

        method = "half_max_component"
        fallback_reason: str | None = None
        if (y1 - y0) < 3 or (x1 - x0) < 3:
            method = "invalid_empty_footprint"
            fallback_reason = "window_too_small"
            y_px = np.empty(0, dtype=np.intp)
            x_px = np.empty(0, dtype=np.intp)
        else:
            window = miat_f[y0:y1, x0:x1]
            sy0 = max(y0, center_y - peak_search_half)
            sy1 = min(y1, center_y + peak_search_half + 1)
            sx0 = max(x0, center_x - peak_search_half)
            sx1 = min(x1, center_x + peak_search_half + 1)
            inner = miat_f[sy0:sy1, sx0:sx1]
            seed_offset = np.unravel_index(int(np.argmax(inner)), inner.shape)
            seed_y = sy0 + int(seed_offset[0])
            seed_x = sx0 + int(seed_offset[1])
            peak = float(miat_f[seed_y, seed_x])
            background = float(np.percentile(window, bg_percentile))
            if not peak > background:
                method = "invalid_empty_footprint"
                fallback_reason = "flat_or_no_contrast"
                y_px = np.empty(0, dtype=np.intp)
                x_px = np.empty(0, dtype=np.intp)
            else:
                threshold = background + half_max_frac * (peak - background)
                labels, _n_labels = ndimage.label(
                    window >= threshold, structure=structure8
                )
                seed_label = int(labels[seed_y - y0, seed_x - x0])
                if seed_label == 0:
                    method = "invalid_empty_footprint"
                    fallback_reason = "seed_below_half_max"
                    y_px = np.empty(0, dtype=np.intp)
                    x_px = np.empty(0, dtype=np.intp)
                else:
                    local_y, local_x = np.where(labels == seed_label)
                    y_px = (local_y + y0).astype(np.intp, copy=False)
                    x_px = (local_x + x0).astype(np.intp, copy=False)

        full_mask_valid, invalid_reason = _mask_validity(y_px, x_px, mask)
        observed_miat = (
            float(miat_f[y_px, x_px].mean()) if y_px.size else float("nan")
        )
        observed_qki = (
            float(partner_f[y_px, x_px].mean())
            if partner_f is not None and y_px.size
            else float("nan")
        )
        output.append(MiatFootprint(
            spot_index=spot_index,
            center_y_px=center_y,
            center_x_px=center_x,
            y_px=y_px,
            x_px=x_px,
            dy_px=(y_px - center_y).astype(np.intp, copy=False),
            dx_px=(x_px - center_x).astype(np.intp, copy=False),
            method=method,
            fallback_reason=fallback_reason,
            full_mask_valid=full_mask_valid,
            invalid_reason=invalid_reason,
            observed_miat_raw=observed_miat,
            observed_qki_raw=observed_qki,
        ))
    return output


def _require_integer_valued(values: Any, field_name: str) -> np.ndarray:
    """Validate exact integer values before any dtype conversion."""
    raw = np.asarray(values)
    if np.issubdtype(raw.dtype, np.integer):
        if np.issubdtype(raw.dtype, np.unsignedinteger):
            if raw.size and int(raw.max()) > np.iinfo(np.int64).max:
                raise ValueError(f"{field_name} exceed int64 range")
        return raw.astype(np.int64, copy=False)
    if not np.issubdtype(raw.dtype, np.floating):
        raise ValueError(f"{field_name} must be numeric and integer-valued")
    numeric = raw.astype(np.float64, copy=False)
    if not bool(np.isfinite(numeric).all()):
        raise ValueError(f"{field_name} must be finite and integer-valued")
    if not bool((numeric == np.rint(numeric)).all()):
        raise ValueError(f"{field_name} must be integer-valued")
    int64_info = np.iinfo(np.int64)
    if bool(((numeric < int64_info.min) | (numeric > int64_info.max)).any()):
        raise ValueError(f"{field_name} exceed int64 range")
    return numeric.astype(np.int64)


def encode_footprint_rle(
    footprint_or_pixels: MiatFootprint | np.ndarray,
    *,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Encode absolute pixels as sorted row-major [start, length] runs."""
    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("image_shape must contain positive dimensions")
    pixels = (
        footprint_or_pixels.pixels_yx
        if isinstance(footprint_or_pixels, MiatFootprint)
        else np.asarray(footprint_or_pixels)
    )
    if pixels.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError("footprint pixels must have shape (n_pixels, 2)")
    pixels_i = _require_integer_valued(pixels, "footprint coordinates")
    y_px = pixels_i[:, 0]
    x_px = pixels_i[:, 1]
    inside = (y_px >= 0) & (y_px < height) & (x_px >= 0) & (x_px < width)
    if not bool(inside.all()):
        raise ValueError("footprint pixels must lie inside image_shape")
    flat = y_px * width + x_px
    if np.unique(flat).size != flat.size:
        raise ValueError("footprint contains duplicate pixels")
    flat.sort()
    groups = np.split(flat, np.flatnonzero(np.diff(flat) != 1) + 1)
    return np.asarray(
        [[int(group[0]), int(group.size)] for group in groups],
        dtype=np.int64,
    )


def decode_footprint_rle(
    encoded: Any,
    *,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Decode row-major [start, length] runs to sorted absolute [y, x]."""
    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("image_shape must contain positive dimensions")
    runs = np.asarray(encoded)
    if runs.size == 0:
        return np.empty((0, 2), dtype=np.intp)
    if runs.ndim != 2 or runs.shape[1] != 2:
        raise ValueError("encoded footprint must have shape (n_runs, 2)")
    runs = _require_integer_valued(runs, "RLE run fields")
    if (runs[:, 0] < 0).any() or (runs[:, 1] <= 0).any():
        raise ValueError("RLE starts must be non-negative and lengths positive")
    flat = np.concatenate([
        np.arange(start, start + length, dtype=np.int64)
        for start, length in runs
    ])
    if (flat >= height * width).any():
        raise ValueError("encoded footprint exceeds image_shape")
    if np.unique(flat).size != flat.size:
        raise ValueError("encoded footprint contains overlapping runs")
    flat.sort()
    return np.column_stack((flat // width, flat % width)).astype(
        np.intp, copy=False
    )


def _rotate_footprint_pixels(
    footprint: MiatFootprint,
    centroid_yx: tuple[float, float],
    angle_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-pixel rigid rotation, preserving the input sample cardinality."""
    centroid_y, centroid_x = centroid_yx
    angle_rad = np.radians(float(angle_deg))
    cosine = float(np.cos(angle_rad))
    sine = float(np.sin(angle_rad))
    offset_y = footprint.y_px.astype(np.float64) - centroid_y
    offset_x = footprint.x_px.astype(np.float64) - centroid_x
    y_px = np.rint(
        centroid_y + offset_y * cosine - offset_x * sine
    ).astype(np.intp)
    x_px = np.rint(
        centroid_x + offset_y * sine + offset_x * cosine
    ).astype(np.intp)
    return y_px, x_px


def _make_spot_result(
    footprint: MiatFootprint,
    *,
    full_mask_valid: bool,
    invalid_reason: str | None,
    observed_qki_raw: float,
    null_threshold_raw: float,
    null_p_empirical: float,
    null_usable: bool,
    association_call: bool | None,
    passes_miat_floor: bool,
) -> FootprintNullSpotResult:
    return FootprintNullSpotResult(
        spot_index=footprint.spot_index,
        center_y_px=footprint.center_y_px,
        center_x_px=footprint.center_x_px,
        y_px=footprint.y_px,
        x_px=footprint.x_px,
        dy_px=footprint.dy_px,
        dx_px=footprint.dx_px,
        method=footprint.method,
        fallback_reason=footprint.fallback_reason,
        full_mask_valid=full_mask_valid,
        invalid_reason=invalid_reason,
        observed_qki_raw=observed_qki_raw,
        null_threshold_raw=null_threshold_raw,
        null_p_empirical=null_p_empirical,
        null_usable=null_usable,
        association_call=association_call,
        passes_miat_floor=passes_miat_floor,
    )


def _unusable_rotation_result(
    footprints: tuple[MiatFootprint, ...],
    observed: np.ndarray,
    floor_flags: np.ndarray,
    validity: Sequence[tuple[bool, str | None]],
    reason: str,
    null_qki_raw: np.ndarray | None = None,
    accepted_angles_deg: Sequence[float] = (),
    rejected_angle_count: int = 0,
) -> FootprintRotationNullResult:
    spots = tuple(
        _make_spot_result(
            footprint,
            full_mask_valid=validity[index][0],
            invalid_reason=validity[index][1],
            observed_qki_raw=float(observed[index]),
            null_threshold_raw=float("nan"),
            null_p_empirical=float("nan"),
            null_usable=False,
            association_call=None,
            passes_miat_floor=bool(floor_flags[index]),
        )
        for index, footprint in enumerate(footprints)
    )
    if null_qki_raw is None:
        null_qki_raw = np.empty((len(footprints), 0), dtype=np.float64)
    return FootprintRotationNullResult(
        spots=spots,
        null_qki_raw=null_qki_raw,
        accepted_angles_deg=np.asarray(accepted_angles_deg, dtype=np.float64),
        rejected_angle_count=int(rejected_angle_count),
        usable=False,
        invalid_reason=reason,
    )


def rigid_footprint_rotation_null(
    partner_2d: np.ndarray,
    footprints: Sequence[MiatFootprint],
    valid_mask: np.ndarray,
    *,
    n_null: int,
    rng: np.random.Generator | None = None,
    candidate_angles_deg: Iterable[float] | None = None,
    centroid_yx: tuple[float, float] | None = None,
    threshold_percentile: float = 95.0,
    passes_miat_floor: Sequence[bool] | None = None,
    max_angle_draws: int | None = None,
) -> FootprintRotationNullResult:
    """Run a strict whole-constellation exact-footprint rotation null.

    One candidate angle transforms every pixel of every footprint. If any
    transformed pixel is out of bounds or mask-invalid, the entire angle is
    rejected and redrawn. Spots are never dropped, repaired independently, or
    restored to observed positions. Each accepted spot sample has exactly the
    same number of pixel samples as its observed footprint.

    Thresholds are computed from each spot's own accepted null vector.
    Association requires observed > threshold. Empirical p is
    (1 + count(null >= observed)) / (1 + n_valid).
    """
    partner = np.asarray(partner_2d)
    mask = np.asarray(valid_mask, dtype=bool)
    if partner.ndim != 2:
        raise ValueError("partner_2d must be two-dimensional")
    if mask.shape != partner.shape:
        raise ValueError("valid_mask must have the same shape as partner_2d")
    if n_null <= 0:
        raise ValueError("n_null must be positive")
    if not (0.0 <= float(threshold_percentile) <= 100.0):
        raise ValueError("threshold_percentile must be in [0, 100]")

    footprints = tuple(footprints)
    n_spots = len(footprints)
    if passes_miat_floor is None:
        floor_flags = np.ones(n_spots, dtype=bool)
    else:
        floor_flags = np.asarray(passes_miat_floor, dtype=bool).reshape(-1)
        if floor_flags.size != n_spots:
            raise ValueError("passes_miat_floor must have one value per footprint")

    partner_f = partner.astype(np.float64, copy=False)
    validated = [
        _validated_pixel_arrays(fp.y_px, fp.x_px, mask) for fp in footprints
    ]
    validity = [(item[2], item[3]) for item in validated]
    observed = np.full(n_spots, np.nan, dtype=np.float64)
    if n_spots == 0:
        return _unusable_rotation_result(
            footprints, observed, floor_flags, validity, "empty_constellation"
        )
    if not all(item[0] for item in validity):
        return _unusable_rotation_result(
            footprints,
            observed,
            floor_flags,
            validity,
            "invalid_observed_footprint",
        )
    for index, (y_px, x_px, _valid, _reason) in enumerate(validated):
        observed_pixels = partner_f[y_px, x_px]
        if bool(np.isfinite(observed_pixels).all()):
            observed[index] = float(observed_pixels.mean())
    if n_spots < 2:
        return _unusable_rotation_result(
            footprints,
            observed,
            floor_flags,
            validity,
            "insufficient_spots_for_rotation",
        )

    if centroid_yx is None:
        centroid = (
            float(np.mean([fp.center_y_px for fp in footprints])),
            float(np.mean([fp.center_x_px for fp in footprints])),
        )
    else:
        centroid = (float(centroid_yx[0]), float(centroid_yx[1]))
        if not bool(np.isfinite(np.asarray(centroid)).all()):
            raise ValueError("centroid_yx must contain finite coordinates")

    if candidate_angles_deg is not None:
        candidates = iter(candidate_angles_deg)

        def next_angle() -> float:
            return float(next(candidates))

        attempts_limit = max_angle_draws
    else:
        generator = np.random.default_rng() if rng is None else rng

        def next_angle() -> float:
            return float(generator.uniform(0.0, 360.0))

        attempts_limit = (
            max_angle_draws if max_angle_draws is not None
            else max(100, n_null * 100)
        )

    accepted_angles: list[float] = []
    accepted_values: list[np.ndarray] = []
    rejected = 0
    attempts = 0
    while len(accepted_angles) < n_null:
        if attempts_limit is not None and attempts >= attempts_limit:
            break
        try:
            angle = next_angle()
        except StopIteration:
            break
        attempts += 1
        if not np.isfinite(angle):
            rejected += 1
            continue

        rotated: list[tuple[np.ndarray, np.ndarray]] = []
        candidate_valid = True
        for footprint in footprints:
            y_px, x_px = _rotate_footprint_pixels(footprint, centroid, angle)
            if y_px.size != footprint.area_px:
                raise RuntimeError("rotation changed footprint sample cardinality")
            valid, _invalid_reason = _mask_validity(y_px, x_px, mask)
            if not valid:
                candidate_valid = False
                break
            rotated.append((y_px, x_px))
        if not candidate_valid:
            rejected += 1
            continue

        candidate_values = []
        for y_px, x_px in rotated:
            partner_pixels = partner_f[y_px, x_px]
            if not bool(np.isfinite(partner_pixels).all()):
                candidate_valid = False
                break
            candidate_values.append(float(partner_pixels.mean()))
        if not candidate_valid:
            rejected += 1
            continue
        accepted_angles.append(angle)
        accepted_values.append(np.asarray(candidate_values, dtype=np.float64))

    null_qki = (
        np.column_stack(accepted_values)
        if accepted_values
        else np.empty((n_spots, 0), dtype=np.float64)
    )
    usable = (
        len(accepted_angles) == n_null
        and null_qki.shape == (n_spots, n_null)
        and bool(np.isfinite(null_qki).all())
    )
    if not usable:
        return _unusable_rotation_result(
            footprints,
            observed,
            floor_flags,
            validity,
            "insufficient_valid_rigid_angles",
            null_qki_raw=null_qki,
            accepted_angles_deg=accepted_angles,
            rejected_angle_count=rejected,
        )

    spot_usable = np.isfinite(observed)
    thresholds = np.full(n_spots, np.nan, dtype=np.float64)
    empirical_p = np.full(n_spots, np.nan, dtype=np.float64)
    thresholds[spot_usable] = np.percentile(
        null_qki[spot_usable],
        float(threshold_percentile),
        axis=1,
        method="linear",
    )
    empirical_p[spot_usable] = (
        1.0
        + np.sum(
            null_qki[spot_usable] >= observed[spot_usable, None],
            axis=1,
        )
    ) / (1.0 + null_qki.shape[1])
    calls: list[bool | None] = [
        bool(observed[index] > thresholds[index])
        if spot_usable[index] and floor_flags[index] else None
        for index in range(n_spots)
    ]
    spots = tuple(
        _make_spot_result(
            footprint,
            full_mask_valid=validity[index][0],
            invalid_reason=(
                validity[index][1]
                if spot_usable[index] else "nonfinite_observed_qki"
            ),
            observed_qki_raw=float(observed[index]),
            null_threshold_raw=float(thresholds[index]),
            null_p_empirical=float(empirical_p[index]),
            null_usable=bool(spot_usable[index]),
            association_call=calls[index],
            passes_miat_floor=bool(floor_flags[index]),
        )
        for index, footprint in enumerate(footprints)
    )
    return FootprintRotationNullResult(
        spots=spots,
        null_qki_raw=null_qki,
        accepted_angles_deg=np.asarray(accepted_angles, dtype=np.float64),
        rejected_angle_count=rejected,
        usable=bool(spot_usable.all()),
        invalid_reason=(
            None if bool(spot_usable.all()) else "nonfinite_spot_values"
        ),
    )


def _keep_n_initial_angles(
    n_null: int,
    rng: np.random.Generator,
) -> np.ndarray:
    fixed = np.asarray([90.0, 180.0, 270.0], dtype=np.float64)
    n_random = max(0, n_null - fixed.size)
    random_angles = (
        np.asarray(rng.uniform(0.0, 360.0, size=n_random), dtype=np.float64)
        if n_random else np.empty(0, dtype=np.float64)
    )
    return np.concatenate((fixed[:n_null], random_angles))[:n_null]


def _validated_footprint_offsets(
    footprint: MiatFootprint,
) -> tuple[np.ndarray, np.ndarray, bool, str | None]:
    dy_raw = np.asarray(footprint.dy_px)
    dx_raw = np.asarray(footprint.dx_px)
    if dy_raw.ndim != 1 or dx_raw.ndim != 1:
        empty = np.empty(0, dtype=np.intp)
        return empty, empty, False, "footprint_offsets_not_1d"
    if (
        dy_raw.size != dx_raw.size
        or dy_raw.size != np.asarray(footprint.y_px).size
        or dx_raw.size != np.asarray(footprint.x_px).size
    ):
        empty = np.empty(0, dtype=np.intp)
        return empty, empty, False, "footprint_offset_length_mismatch"
    try:
        dy = _require_integer_valued(
            dy_raw, "footprint offsets"
        ).astype(np.intp, copy=False)
        dx = _require_integer_valued(
            dx_raw, "footprint offsets"
        ).astype(np.intp, copy=False)
        y_px = _require_integer_valued(
            footprint.y_px, "footprint coordinates"
        ).astype(np.intp, copy=False)
        x_px = _require_integer_valued(
            footprint.x_px, "footprint coordinates"
        ).astype(np.intp, copy=False)
    except ValueError:
        empty = np.empty(0, dtype=np.intp)
        return empty, empty, False, "footprint_offsets_not_integer"
    if not (
        np.array_equal(dy, y_px - int(footprint.center_y_px))
        and np.array_equal(dx, x_px - int(footprint.center_x_px))
    ):
        return dy, dx, False, "footprint_offsets_mismatch"
    if np.unique(np.column_stack((dy, dx)), axis=0).shape[0] != dy.size:
        return dy, dx, False, "duplicate_footprint_offsets"
    return dy, dx, True, None


def _rotated_center_translated_footprint_pixels(
    footprint: MiatFootprint,
    dy_px: np.ndarray,
    dx_px: np.ndarray,
    centroid_yx: tuple[float, float],
    angle_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate the spot center, then translate the exact stored raster mask."""
    centroid_y, centroid_x = centroid_yx
    angle_rad = np.radians(float(angle_deg))
    cosine = float(np.cos(angle_rad))
    sine = float(np.sin(angle_rad))
    center_offset_y = float(footprint.center_y_px) - centroid_y
    center_offset_x = float(footprint.center_x_px) - centroid_x
    placed_center_y = int(np.rint(
        centroid_y
        + center_offset_y * cosine
        - center_offset_x * sine
    ))
    placed_center_x = int(np.rint(
        centroid_x
        + center_offset_y * sine
        + center_offset_x * cosine
    ))
    return placed_center_y + dy_px, placed_center_x + dx_px


def _unusable_keep_n_result(
    footprints: tuple[MiatFootprint, ...],
    observed: np.ndarray,
    floor_flags: np.ndarray,
    validity: Sequence[tuple[bool, str | None]],
    initial_angles: np.ndarray,
    reason: str,
) -> KeepNFootprintRotationNullResult:
    n_spots = len(footprints)
    n_null = int(initial_angles.size)
    null_qki = np.full((n_spots, n_null), np.nan, dtype=np.float64)
    placement_angles = np.full((n_spots, n_null), np.nan, dtype=np.float64)
    first_pass_valid = np.zeros((n_spots, n_null), dtype=bool)
    redraw_counts = np.zeros((n_spots, n_null), dtype=np.int32)
    valid_draw_counts = np.zeros(n_spots, dtype=np.int64)
    valid_draw_fractions = np.zeros(n_spots, dtype=np.float64)
    spots = tuple(
        _make_spot_result(
            footprint,
            full_mask_valid=validity[index][0],
            invalid_reason=(
                validity[index][1]
                or (
                    "nonfinite_observed_qki"
                    if not np.isfinite(observed[index]) else reason
                )
            ),
            observed_qki_raw=float(observed[index]),
            null_threshold_raw=float("nan"),
            null_p_empirical=float("nan"),
            null_usable=False,
            association_call=None,
            passes_miat_floor=bool(floor_flags[index]),
        )
        for index, footprint in enumerate(footprints)
    )
    return KeepNFootprintRotationNullResult(
        spots=spots,
        placement_geometry="rotated_center_translated_exact_footprint",
        null_qki_raw=null_qki,
        initial_angles_deg=initial_angles,
        placement_angles_deg=placement_angles,
        first_pass_valid=first_pass_valid,
        redraw_counts=redraw_counts,
        valid_draw_counts=valid_draw_counts,
        valid_draw_fractions=valid_draw_fractions,
        mean_first_pass_retention=0.0,
        median_first_pass_retention=0.0,
        unplaceable_count=0,
        unplaceable_fraction=0.0,
        usable=False,
        invalid_reason=reason,
    )


def keep_n_footprint_rotation_null(
    partner_2d: np.ndarray,
    footprints: Sequence[MiatFootprint],
    valid_mask: np.ndarray,
    *,
    n_null: int,
    rng: np.random.Generator | None = None,
    passes_miat_floor: Sequence[bool] | None = None,
    threshold_percentile: float = 95.0,
    min_first_pass_retention: float = 0.5,
    min_valid_draw_fraction: float = 1.0,
    min_valid_draws: int = 2,
    max_redraw: int = 1000,
) -> KeepNFootprintRotationNullResult:
    """Authoritative exact-footprint KEEP-N rotation null.

    Each iteration starts with one shared angle for the full constellation.
    Spot centers rotate about the constellation centroid; each exact original
    footprint is then translated to its rotated center with its stored dy/dx
    offsets unchanged. The irregular raster mask is never rotated or resampled.
    A placement that fails full-pixel geometry, mask, unique-cardinality, or
    finite-QKI validation is redrawn independently. Valid first-pass footprints
    keep the shared angle. Unplaceable draws remain NaN and are never replaced
    with observed pixels.

    The primary valid-draw gate requires every requested draw by default.
    A lower fraction is available only as an explicit sensitivity setting.
    """
    partner = np.asarray(partner_2d)
    mask = np.asarray(valid_mask, dtype=bool)
    if partner.ndim != 2:
        raise ValueError("partner_2d must be two-dimensional")
    if mask.shape != partner.shape:
        raise ValueError("valid_mask must have the same shape as partner_2d")
    if n_null <= 0:
        raise ValueError("n_null must be positive")
    if max_redraw < 0:
        raise ValueError("max_redraw must be non-negative")
    if min_valid_draws < 1:
        raise ValueError("min_valid_draws must be positive")
    if min_valid_draws > n_null:
        raise ValueError("min_valid_draws cannot exceed n_null")
    if not (0.0 <= float(threshold_percentile) <= 100.0):
        raise ValueError("threshold_percentile must be in [0, 100]")
    if not (0.0 <= float(min_first_pass_retention) <= 1.0):
        raise ValueError("min_first_pass_retention must be in [0, 1]")
    if not (0.0 <= float(min_valid_draw_fraction) <= 1.0):
        raise ValueError("min_valid_draw_fraction must be in [0, 1]")

    generator = np.random.default_rng() if rng is None else rng
    initial_angles = _keep_n_initial_angles(n_null, generator)
    footprints = tuple(footprints)
    n_spots = len(footprints)
    if passes_miat_floor is None:
        floor_flags = np.ones(n_spots, dtype=bool)
    else:
        floor_flags = np.asarray(passes_miat_floor, dtype=bool).reshape(-1)
        if floor_flags.size != n_spots:
            raise ValueError("passes_miat_floor must have one value per footprint")

    partner_f = partner.astype(np.float64, copy=False)
    validated = [
        _validated_pixel_arrays(fp.y_px, fp.x_px, mask) for fp in footprints
    ]
    validated_offsets = [
        _validated_footprint_offsets(fp) for fp in footprints
    ]
    for index, offset_result in enumerate(validated_offsets):
        if validated[index][2] and not offset_result[2]:
            y_px, x_px, _valid, _reason = validated[index]
            validated[index] = (
                y_px, x_px, False, offset_result[3]
            )
    validity = [(item[2], item[3]) for item in validated]
    observed = np.full(n_spots, np.nan, dtype=np.float64)
    if n_spots == 0:
        return _unusable_keep_n_result(
            footprints,
            observed,
            floor_flags,
            validity,
            initial_angles,
            "empty_constellation",
        )
    if not all(item[0] for item in validity):
        return _unusable_keep_n_result(
            footprints,
            observed,
            floor_flags,
            validity,
            initial_angles,
            "invalid_observed_footprint",
        )
    for index, (y_px, x_px, _valid, _reason) in enumerate(validated):
        observed_pixels = partner_f[y_px, x_px]
        if bool(np.isfinite(observed_pixels).all()):
            observed[index] = float(observed_pixels.mean())
    if n_spots < 2:
        return _unusable_keep_n_result(
            footprints,
            observed,
            floor_flags,
            validity,
            initial_angles,
            "insufficient_spots_for_rotation",
        )

    centroid = (
        float(np.mean([fp.center_y_px for fp in footprints])),
        float(np.mean([fp.center_x_px for fp in footprints])),
    )
    null_qki = np.full((n_spots, n_null), np.nan, dtype=np.float64)
    placement_angles = np.full((n_spots, n_null), np.nan, dtype=np.float64)
    first_pass_valid = np.zeros((n_spots, n_null), dtype=bool)
    redraw_counts = np.zeros((n_spots, n_null), dtype=np.int32)

    def placement_value(
        spot_index: int,
        angle_deg: float,
    ) -> float | None:
        if not np.isfinite(angle_deg):
            return None
        footprint = footprints[spot_index]
        dy_px, dx_px, _valid, _reason = validated_offsets[spot_index]
        y_px, x_px = _rotated_center_translated_footprint_pixels(
            footprint,
            dy_px,
            dx_px,
            centroid,
            float(angle_deg),
        )
        valid, _invalid_reason = _mask_validity(y_px, x_px, mask)
        if not valid:
            return None
        partner_pixels = partner_f[y_px, x_px]
        if not bool(np.isfinite(partner_pixels).all()):
            return None
        return float(partner_pixels.mean())

    for null_index, initial_angle in enumerate(initial_angles):
        for spot_index in range(n_spots):
            value = placement_value(spot_index, float(initial_angle))
            if value is not None:
                first_pass_valid[spot_index, null_index] = True
                placement_angles[spot_index, null_index] = initial_angle
                null_qki[spot_index, null_index] = value

    first_pass_retention = first_pass_valid.mean(axis=0)
    mean_retention = float(first_pass_retention.mean())
    median_retention = float(np.median(first_pass_retention))
    nucleus_gate = bool(median_retention >= min_first_pass_retention)
    if nucleus_gate:
        invalid_spot_indices, invalid_null_indices = np.nonzero(
            ~first_pass_valid
        )
        for spot_index, null_index in zip(
            invalid_spot_indices,
            invalid_null_indices,
            strict=True,
        ):
            for redraw_count in range(1, max_redraw + 1):
                redraw_angle = float(generator.uniform(0.0, 360.0))
                redraw_counts[spot_index, null_index] = redraw_count
                value = placement_value(int(spot_index), redraw_angle)
                if value is not None:
                    placement_angles[spot_index, null_index] = redraw_angle
                    null_qki[spot_index, null_index] = value
                    break

    valid_draw_mask = np.isfinite(null_qki)
    valid_draw_counts = valid_draw_mask.sum(axis=1).astype(np.int64)
    valid_draw_fractions = valid_draw_counts.astype(np.float64) / float(n_null)
    observed_finite = np.isfinite(observed)
    spot_draw_gate = (
        (valid_draw_counts >= min_valid_draws)
        & (valid_draw_fractions >= min_valid_draw_fraction)
    )
    spot_usable = nucleus_gate & observed_finite & spot_draw_gate

    thresholds = np.full(n_spots, np.nan, dtype=np.float64)
    empirical_p = np.full(n_spots, np.nan, dtype=np.float64)
    calls: list[bool | None] = [None] * n_spots
    for spot_index in range(n_spots):
        if not spot_usable[spot_index]:
            continue
        finite_draws = null_qki[spot_index, valid_draw_mask[spot_index]]
        thresholds[spot_index] = float(np.percentile(
            finite_draws,
            float(threshold_percentile),
            method="linear",
        ))
        empirical_p[spot_index] = float(
            (1.0 + np.count_nonzero(finite_draws >= observed[spot_index]))
            / (1.0 + finite_draws.size)
        )
        if floor_flags[spot_index]:
            calls[spot_index] = bool(
                observed[spot_index] > thresholds[spot_index]
            )

    spots = []
    for spot_index, footprint in enumerate(footprints):
        if not nucleus_gate:
            spot_reason = "low_first_pass_retention"
        elif not observed_finite[spot_index]:
            spot_reason = "nonfinite_observed_qki"
        elif not spot_draw_gate[spot_index]:
            spot_reason = "insufficient_valid_null_draws"
        else:
            spot_reason = validity[spot_index][1]
        spots.append(_make_spot_result(
            footprint,
            full_mask_valid=validity[spot_index][0],
            invalid_reason=spot_reason,
            observed_qki_raw=float(observed[spot_index]),
            null_threshold_raw=float(thresholds[spot_index]),
            null_p_empirical=float(empirical_p[spot_index]),
            null_usable=bool(spot_usable[spot_index]),
            association_call=calls[spot_index],
            passes_miat_floor=bool(floor_flags[spot_index]),
        ))

    redraw_exhausted = (redraw_counts > 0) & ~valid_draw_mask
    unplaceable_count = int(np.count_nonzero(redraw_exhausted))
    total_draws = n_spots * n_null
    unplaceable_fraction = (
        float(unplaceable_count / total_draws) if total_draws else 0.0
    )
    any_spot_usable = bool(spot_usable.any())
    if not nucleus_gate:
        result_reason = "low_first_pass_retention"
    elif not any_spot_usable:
        result_reason = "no_usable_spots"
    else:
        result_reason = None
    return KeepNFootprintRotationNullResult(
        spots=tuple(spots),
        placement_geometry="rotated_center_translated_exact_footprint",
        null_qki_raw=null_qki,
        initial_angles_deg=initial_angles,
        placement_angles_deg=placement_angles,
        first_pass_valid=first_pass_valid,
        redraw_counts=redraw_counts,
        valid_draw_counts=valid_draw_counts,
        valid_draw_fractions=valid_draw_fractions,
        mean_first_pass_retention=mean_retention,
        median_first_pass_retention=median_retention,
        unplaceable_count=unplaceable_count,
        unplaceable_fraction=unplaceable_fraction,
        usable=bool(nucleus_gate and any_spot_usable),
        invalid_reason=result_reason,
    )


def _row_value(row: Mapping[str, Any] | Any, name: str, default: Any) -> Any:
    if isinstance(row, Mapping):
        return row.get(name, default)
    return getattr(row, name, default)


def _explicit_bool(value: Any) -> bool | None:
    """Accept only actual Python/NumPy booleans, never generic truthiness."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return None


def _population_summary(rows: Sequence[Mapping[str, Any] | Any]) -> dict[str, Any]:
    values = np.asarray([
        _row_value(row, "observed_qki_raw", np.nan) for row in rows
    ], dtype=np.float64)
    finite = values[np.isfinite(values)]
    return {
        "n": len(rows),
        "n_finite_observed_qki_raw": int(finite.size),
        "mean_observed_qki_raw": (
            float(finite.mean()) if finite.size else float("nan")
        ),
        "median_observed_qki_raw": (
            float(np.median(finite)) if finite.size else float("nan")
        ),
    }


def summarize_footprint_null_populations(
    rows: Sequence[Mapping[str, Any] | FootprintNullSpotResult],
) -> dict[str, Any]:
    """Summarize all/floor/assoc/nonassoc/unusable without dropping rows.

    All retains every input row. Floor contains every MIAT-floor-passing row.
    Within floor, rows partition exactly into positive, negative, and unusable.
    Below-floor rows remain in all and in the explicit reconciliation count.
    """
    all_rows = list(rows)
    floor_rows: list[Mapping[str, Any] | Any] = []
    assoc_rows: list[Mapping[str, Any] | Any] = []
    nonassoc_rows: list[Mapping[str, Any] | Any] = []
    unusable_rows: list[Mapping[str, Any] | Any] = []

    for row in all_rows:
        passes_floor = _explicit_bool(
            _row_value(row, "passes_miat_floor", False)
        )
        if passes_floor is not True:
            continue
        floor_rows.append(row)
        null_usable = _explicit_bool(_row_value(row, "null_usable", False))
        call = _explicit_bool(_row_value(row, "association_call", None))
        if null_usable is not True or call is None:
            unusable_rows.append(row)
        elif call is True:
            assoc_rows.append(row)
        else:
            nonassoc_rows.append(row)

    below_floor_n = len(all_rows) - len(floor_rows)
    floor_reconciles = (
        len(floor_rows)
        == len(assoc_rows) + len(nonassoc_rows) + len(unusable_rows)
    )
    all_reconciles = len(all_rows) == len(floor_rows) + below_floor_n
    return {
        "all": _population_summary(all_rows),
        "floor": _population_summary(floor_rows),
        "assoc": _population_summary(assoc_rows),
        "nonassoc": _population_summary(nonassoc_rows),
        "unusable": _population_summary(unusable_rows),
        "reconciliation": {
            "below_floor_n": below_floor_n,
            "floor_equals_assoc_plus_nonassoc_plus_unusable": floor_reconciles,
            "all_equals_floor_plus_below_floor": all_reconciles,
        },
    }


__all__ = [
    "FootprintNullSpotResult",
    "FootprintRotationNullResult",
    "KeepNFootprintRotationNullResult",
    "MiatFootprint",
    "build_miat_footprints",
    "decode_footprint_rle",
    "encode_footprint_rle",
    "full_footprint_is_valid",
    "keep_n_footprint_rotation_null",
    "rigid_footprint_rotation_null",
    "summarize_footprint_null_populations",
]
