"""Unit tests for the thresholded RNA intensity in compartments feature.

Brian 2026-06-02. Covers:
  * the core threshold-and-integrate helper
    (``metrics.compute_thresholded_compartment_intensity``) on synthetic
    nucleus + cytoplasm masks with a known RNA array (pixels above/below the
    floor) — asserting total / mean / area / fraction per compartment, and
    that the floor is respected; and
  * the floor-resolution precedence
    (``rna_only._resolve_thresh_intensity_floor``) — that the DEFAULT floor is
    the spot floor (``manual_rna_min``) when the dedicated knob is unset, that
    an explicit ``rna_intensity_threshold`` pin overrides it, and that a
    runner-forwarded ``analysis_floors`` value is used as the default.

These tests exercise the load-bearing math + config wiring WITHOUT touching
the GPU pipeline (no segmentation / spot detection / image IO).
"""
from __future__ import annotations

import math

import numpy as np

from fishsuite.core.metrics import compute_thresholded_compartment_intensity
from fishsuite.core.modes.rna_only import _resolve_thresh_intensity_floor
from fishsuite.config.schema import FishsuiteConfig


# ---------------------------------------------------------------------------
# Core threshold-and-integrate helper
# ---------------------------------------------------------------------------

def _synthetic_scene():
    """Build a 4x4 RNA plane + disjoint nucleus / cytoplasm masks.

    Layout (row, col) — intensities chosen so the >=floor selection is obvious
    at floor = 100:

        nucleus pixels (mask_nuc): (0,0)=300, (0,1)=150, (1,0)=50, (1,1)=80
            -> above floor (>=100): {300, 150}  (2 pixels)
        cytoplasm pixels (mask_cyt): (2,2)=400, (2,3)=400, (3,2)=20, (3,3)=90
            -> above floor (>=100): {400, 400}  (2 pixels)

    Everything else is 0 and unmasked.
    """
    img = np.zeros((4, 4), dtype=np.uint16)
    img[0, 0] = 300
    img[0, 1] = 150
    img[1, 0] = 50
    img[1, 1] = 80
    img[2, 2] = 400
    img[2, 3] = 400
    img[3, 2] = 20
    img[3, 3] = 90

    mask_nuc = np.zeros((4, 4), dtype=bool)
    mask_nuc[0, 0] = mask_nuc[0, 1] = mask_nuc[1, 0] = mask_nuc[1, 1] = True

    mask_cyt = np.zeros((4, 4), dtype=bool)
    mask_cyt[2, 2] = mask_cyt[2, 3] = mask_cyt[3, 2] = mask_cyt[3, 3] = True
    return img, mask_nuc, mask_cyt


def test_thresholded_intensity_nucleus_and_cytoplasm_known_values():
    img, mask_nuc, mask_cyt = _synthetic_scene()
    floor = 100.0

    # --- Nucleus: above-floor pixels are 300 and 150 (4 total in mask) ---
    n = compute_thresholded_compartment_intensity(img, mask_nuc, floor)
    assert n["thresh_pos_area_px"] == 2
    assert n["thresh_total_intensity"] == 450.0          # 300 + 150 (RAW sum)
    assert n["thresh_mean_intensity"] == 225.0           # 450 / 2
    assert n["thresh_pos_fraction"] == 0.5               # 2 of 4 nucleus pixels

    # --- Cytoplasm: above-floor pixels are 400 and 400 (4 total in mask) ---
    c = compute_thresholded_compartment_intensity(img, mask_cyt, floor)
    assert c["thresh_pos_area_px"] == 2
    assert c["thresh_total_intensity"] == 800.0          # 400 + 400
    assert c["thresh_mean_intensity"] == 400.0
    assert c["thresh_pos_fraction"] == 0.5

    # The two compartments are measured INDEPENDENTLY (no leakage).
    assert n["thresh_total_intensity"] != c["thresh_total_intensity"]


def test_thresholded_intensity_floor_is_respected():
    """Raising the floor must drop the now-below-floor pixels from the sum."""
    img, mask_nuc, _ = _synthetic_scene()

    # floor = 200 keeps only the 300 pixel in the nucleus.
    n_hi = compute_thresholded_compartment_intensity(img, mask_nuc, 200.0)
    assert n_hi["thresh_pos_area_px"] == 1
    assert n_hi["thresh_total_intensity"] == 300.0
    assert n_hi["thresh_mean_intensity"] == 300.0
    assert n_hi["thresh_pos_fraction"] == 0.25           # 1 of 4

    # floor = 500 selects NOTHING in the nucleus -> total 0, mean NaN, area 0.
    n_none = compute_thresholded_compartment_intensity(img, mask_nuc, 500.0)
    assert n_none["thresh_pos_area_px"] == 0
    assert n_none["thresh_total_intensity"] == 0.0
    assert math.isnan(n_none["thresh_mean_intensity"])
    assert n_none["thresh_pos_fraction"] == 0.0


def test_thresholded_intensity_inclusive_at_exact_floor():
    """A pixel whose value EQUALS the floor is included (>= semantics)."""
    img = np.array([[100, 99]], dtype=np.uint16)
    mask = np.array([[True, True]], dtype=bool)
    out = compute_thresholded_compartment_intensity(img, mask, 100.0)
    assert out["thresh_pos_area_px"] == 1                # only the ==100 pixel
    assert out["thresh_total_intensity"] == 100.0


def test_thresholded_intensity_no_floor_returns_nan_but_area_zero():
    """When no floor is resolvable (None / NaN / <=0) the value columns are
    NaN but thresh_pos_area_px stays 0 (never NaN) so the int column is
    schema-stable."""
    img, mask_nuc, _ = _synthetic_scene()
    for bad_floor in (None, float("nan"), 0.0, -5.0):
        out = compute_thresholded_compartment_intensity(img, mask_nuc, bad_floor)
        assert out["thresh_pos_area_px"] == 0
        assert math.isnan(out["thresh_total_intensity"])
        assert math.isnan(out["thresh_mean_intensity"])
        # compartment HAS pixels, so fraction is 0.0 (not NaN) for a bad floor.
        assert out["thresh_pos_fraction"] == 0.0


def test_thresholded_intensity_empty_mask():
    """An empty compartment mask -> area 0, value columns NaN, fraction NaN."""
    img, _, _ = _synthetic_scene()
    empty = np.zeros((4, 4), dtype=bool)
    out = compute_thresholded_compartment_intensity(img, empty, 100.0)
    assert out["thresh_pos_area_px"] == 0
    assert math.isnan(out["thresh_total_intensity"])
    assert math.isnan(out["thresh_mean_intensity"])
    assert math.isnan(out["thresh_pos_fraction"])        # 0-pixel compartment


# ---------------------------------------------------------------------------
# Floor-resolution precedence (default = spot floor)
# ---------------------------------------------------------------------------

def test_resolve_floor_defaults_to_spot_floor_manual_rna_min():
    """With no dedicated knob set, the threshold floor DEFAULTS to the spot
    floor (manual_rna_min), read directly from config (no runner needed)."""
    cfg = FishsuiteConfig()
    cfg.output.manual_rna_min = 800.0
    cfg.output.rna_intensity_threshold = None            # dedicated knob unset
    floor = _resolve_thresh_intensity_floor(cfg, None, "rna")
    assert floor == 800.0


def test_resolve_floor_explicit_pin_overrides_spot_floor():
    """An explicit rna_intensity_threshold pin takes precedence over both the
    runner-forwarded floor AND manual_rna_min."""
    cfg = FishsuiteConfig()
    cfg.output.manual_rna_min = 800.0
    cfg.output.rna_intensity_threshold = 1250.0
    floor = _resolve_thresh_intensity_floor(
        cfg, {"rna": 800.0}, "rna",
    )
    assert floor == 1250.0


def test_resolve_floor_uses_runner_forwarded_analysis_floor_as_default():
    """When the dedicated knob is unset, a runner-forwarded analysis_floors
    value (the resolved spot floor) is used as the default — preferred over
    manual_rna_min when both are present."""
    cfg = FishsuiteConfig()
    cfg.output.manual_rna_min = 500.0                    # would be fallback #3
    cfg.output.rna_intensity_threshold = None
    floor = _resolve_thresh_intensity_floor(
        cfg, {"rna": 900.0}, "rna",                      # fallback #2 wins
    )
    assert floor == 900.0


def test_resolve_floor_rna2_channel_uses_rna2_sources():
    """The rna2 channel resolves against rna2_intensity_threshold /
    manual_rna2_min / analysis_floors['rna2'] — independent of the rna1
    channel's floor."""
    cfg = FishsuiteConfig()
    cfg.output.manual_rna2_min = 650.0
    cfg.output.rna2_intensity_threshold = None
    # No analysis_floors -> falls back to manual_rna2_min.
    assert _resolve_thresh_intensity_floor(cfg, None, "rna2") == 650.0
    # Explicit pin on rna2 wins.
    cfg.output.rna2_intensity_threshold = 1000.0
    assert _resolve_thresh_intensity_floor(cfg, None, "rna2") == 1000.0


def test_resolve_floor_nan_when_nothing_set():
    """No knob, no analysis_floors, no manual_rna_min -> NaN (columns emitted
    but empty; schema stays stable)."""
    cfg = FishsuiteConfig()
    floor = _resolve_thresh_intensity_floor(cfg, None, "rna")
    assert math.isnan(floor)
