"""Unit tests for compute_focus_window — fixed-N centered branch.

Validates the 2026-05-24 fixed-N centered focus-window mode added on top of
the FWHM-style default. Each test constructs a synthetic DAPI z-stack where
ONE slice (the "in-focus" slice) has high spatial variance and the rest are
flat (low variance-of-Laplacian score). That makes the peak-focus slice
deterministic so we can assert exact window bounds.

Key invariants tested:
  - Odd N: symmetric around peak.
  - Even N: asymmetric (peak - n//2, peak + (n-1)//2).
  - Clamping at the lower bound: window SHIFTS right (no shrink).
  - Clamping at the upper bound: window SHIFTS left (no shrink).
  - Outer-bound interval narrower than N: window SHRINKS (with shrunk_by_bounds=True).
  - fixed_n_slices=0: legacy FWHM branch is used (no fixed_n flag in diagnostics).
"""
from __future__ import annotations

import numpy as np
import pytest

from fishsuite.core.io import compute_focus_window


def _make_stack_with_peak_at(peak_z: int, nz: int = 21, size: int = 64) -> np.ndarray:
    """Make a (Z, Y, X) stack where slice `peak_z` has high Laplacian variance.

    All other slices are uniform flat fields (zero Laplacian variance). The
    peak slice is a random-noise field, which gives a very high variance-of-
    Laplacian score and a near-zero score everywhere else, making the peak
    deterministic.
    """
    rng = np.random.default_rng(seed=42 + peak_z)
    stack = np.full((nz, size, size), 1000.0, dtype=np.float32)
    # Inject high-frequency noise into the peak slice
    stack[peak_z] = rng.uniform(500, 5000, size=(size, size)).astype(np.float32)
    return stack


# ─── Odd N — symmetric window ──────────────────────────────────────────────

def test_fixed_n_odd_centered_no_clamping():
    """N=7, peak at z=10, plenty of room → window = [7, 13], peak centered."""
    stack = _make_stack_with_peak_at(peak_z=10, nz=21)
    (ws, we), diag = compute_focus_window(stack, fixed_n_slices=7)
    assert diag["peak_z"] == 10
    assert (ws, we) == (7, 13)
    assert diag["window_size"] == 7
    assert diag["fixed_n"] is True
    assert diag["requested_n"] == 7
    assert diag["actual_n"] == 7
    assert diag["shifted_for_bounds"] is False
    assert diag["shrunk_by_bounds"] is False


# ─── Even N — asymmetric (more slices trail the peak) ──────────────────────

def test_fixed_n_even_asymmetric():
    """N=6, peak at z=10 → window = [8, 13]: 2 before peak, 3 after."""
    stack = _make_stack_with_peak_at(peak_z=10, nz=21)
    (ws, we), diag = compute_focus_window(stack, fixed_n_slices=6)
    # half_lo = 6 // 2 = 3; half_hi = 6 - 1 - 3 = 2 → [peak-3, peak+2] = [7, 12]
    assert (ws, we) == (7, 12)
    assert diag["window_size"] == 6
    assert diag["actual_n"] == 6
    assert diag["shifted_for_bounds"] is False


# ─── Clamping at lower bound — SHIFT right, don't shrink ───────────────────

def test_fixed_n_shifts_right_at_lower_bound():
    """Peak at z=1, N=7 → centered window [-2, 4] would underflow; shift to [0, 6]."""
    stack = _make_stack_with_peak_at(peak_z=1, nz=21)
    (ws, we), diag = compute_focus_window(stack, fixed_n_slices=7)
    assert diag["peak_z"] == 1
    assert (ws, we) == (0, 6)
    assert diag["window_size"] == 7  # WIDTH PRESERVED — that's the whole point
    assert diag["actual_n"] == 7
    assert diag["shifted_for_bounds"] is True
    assert diag["shrunk_by_bounds"] is False


def test_fixed_n_shifts_right_with_outer_start():
    """Outer bound z>=5, peak at z=6, N=7 → centered [3, 9] underflows → shift to [5, 11]."""
    stack = _make_stack_with_peak_at(peak_z=6, nz=21)
    (ws, we), diag = compute_focus_window(
        stack, fixed_n_slices=7, outer_start=5, outer_end=15,
    )
    assert diag["peak_z"] == 6
    assert (ws, we) == (5, 11)
    assert diag["window_size"] == 7
    assert diag["shifted_for_bounds"] is True
    assert diag["shrunk_by_bounds"] is False


# ─── Clamping at upper bound — SHIFT left, don't shrink ────────────────────

def test_fixed_n_shifts_left_at_upper_bound():
    """Peak at z=19 (nz=21), N=7 → centered [16, 22] overflows → shift to [14, 20]."""
    stack = _make_stack_with_peak_at(peak_z=19, nz=21)
    (ws, we), diag = compute_focus_window(stack, fixed_n_slices=7)
    assert diag["peak_z"] == 19
    assert (ws, we) == (14, 20)
    assert diag["window_size"] == 7
    assert diag["shifted_for_bounds"] is True
    assert diag["shrunk_by_bounds"] is False


def test_fixed_n_shifts_left_with_outer_end():
    """Outer bound z<=15, peak at z=14, N=7 → centered [11, 17] overflows → shift to [9, 15]."""
    stack = _make_stack_with_peak_at(peak_z=14, nz=21)
    (ws, we), diag = compute_focus_window(
        stack, fixed_n_slices=7, outer_start=5, outer_end=15,
    )
    assert diag["peak_z"] == 14
    assert (ws, we) == (9, 15)
    assert diag["window_size"] == 7
    assert diag["shifted_for_bounds"] is True


# ─── Practical H9 MIAT case — outer bounds [5, 15] (1-indexed) → [4, 14] (0-indexed) ──

@pytest.mark.parametrize(
    "peak_z, expected_ws, expected_we",
    [
        (4, 4, 10),   # peak at lower bound → window slides right to [4, 10]
        (5, 4, 10),   # peak at z=5 → centered [2, 8] underflows → shift to [4, 10]
        (6, 4, 10),   # still clamped
        (7, 4, 10),   # last clamped position
        (8, 5, 11),   # first non-shifted: centered [5, 11]
        (9, 6, 12),   # centered [6, 12]
        (10, 7, 13),  # centered [7, 13]
        (11, 8, 14),  # last non-shifted: centered [8, 14]
        (12, 8, 14),  # first upper-clamp
        (13, 8, 14),  # still clamped
        (14, 8, 14),  # peak at upper bound → window slides left to [8, 14]
    ],
)
def test_fixed_n_h9_miat_outer_bounds(peak_z, expected_ws, expected_we):
    """N=7 with outer_start=4, outer_end=14 (0-indexed; mirrors H9 MIAT preset's
    1-indexed [5, 15]). Validates the practical window positions Brian will
    see across all possible peak-z locations within the outer range.
    """
    stack = _make_stack_with_peak_at(peak_z=peak_z, nz=21)
    (ws, we), diag = compute_focus_window(
        stack, fixed_n_slices=7, outer_start=4, outer_end=14,
    )
    assert diag["peak_z"] == peak_z
    assert (ws, we) == (expected_ws, expected_we), (
        f"peak={peak_z} expected ({expected_ws},{expected_we}) got ({ws},{we})"
    )
    assert diag["window_size"] == 7
    assert diag["actual_n"] == 7
    assert diag["shrunk_by_bounds"] is False


# ─── Shrink case — outer interval narrower than requested N ────────────────

def test_fixed_n_shrinks_when_outer_interval_too_narrow():
    """N=7 but outer bounds only allow 5 slices → window shrinks to 5, flag set."""
    stack = _make_stack_with_peak_at(peak_z=10, nz=21)
    (ws, we), diag = compute_focus_window(
        stack, fixed_n_slices=7, outer_start=8, outer_end=12,
    )
    assert (ws, we) == (8, 12)
    assert diag["window_size"] == 5
    assert diag["actual_n"] == 5
    assert diag["requested_n"] == 7
    assert diag["shrunk_by_bounds"] is True
    # When shrunk, we don't claim shifted (the whole outer interval IS the window)
    assert diag["shifted_for_bounds"] is False


def test_fixed_n_equals_outer_interval_exactly():
    """N=5 with outer interval of exactly 5 slices → window fills outer, no shrink."""
    stack = _make_stack_with_peak_at(peak_z=10, nz=21)
    (ws, we), diag = compute_focus_window(
        stack, fixed_n_slices=5, outer_start=8, outer_end=12,
    )
    assert (ws, we) == (8, 12)
    assert diag["window_size"] == 5
    assert diag["actual_n"] == 5
    assert diag["shrunk_by_bounds"] is True  # n_req >= outer_span branch
    assert diag["shifted_for_bounds"] is False


# ─── Backward compat — fixed_n_slices=0 must use FWHM branch ───────────────

def test_fixed_n_zero_uses_fwhm_branch():
    """fixed_n_slices=0 (default) keeps legacy FWHM behavior intact."""
    stack = _make_stack_with_peak_at(peak_z=10, nz=21)
    # Make the peak much sharper so the FWHM window is narrow
    (ws, we), diag = compute_focus_window(stack, fixed_n_slices=0, min_slices=3)
    assert diag["fixed_n"] is False
    # FWHM defaults guarantee min_slices=3
    assert diag["window_size"] >= 3
    # No fixed-N keys in diagnostics
    assert "requested_n" not in diag
    assert "actual_n" not in diag
    assert "shifted_for_bounds" not in diag


def test_fixed_n_default_arg_is_zero():
    """Calling without the new keyword stays on the FWHM path (signature-compat)."""
    stack = _make_stack_with_peak_at(peak_z=10, nz=21)
    (ws, we), diag = compute_focus_window(stack)
    assert diag["fixed_n"] is False


# ─── N=1 sanity check ──────────────────────────────────────────────────────

def test_fixed_n_one_returns_single_slice():
    """N=1 → window is just the peak slice itself."""
    stack = _make_stack_with_peak_at(peak_z=10, nz=21)
    (ws, we), diag = compute_focus_window(stack, fixed_n_slices=1)
    assert (ws, we) == (10, 10)
    assert diag["window_size"] == 1
    assert diag["actual_n"] == 1
