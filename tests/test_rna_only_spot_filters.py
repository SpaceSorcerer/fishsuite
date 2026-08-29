"""Regression tests for per-channel spot filtering in ``rna_only``.

The expensive image reader and BigFISH detector are replaced with deterministic
test doubles, while ``rna_only.run_one`` still performs its real stratification,
per-nucleus aggregation, and spot export.  The stage observations below pin the
required order:

    detect -> absolute peak floor -> stratify -> nuclear-only gate -> diameter

This matters because filtering only the final export would leave per-nucleus
counts and diameter summaries computed from spots that should have been removed.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional

import numpy as np
import pandas as pd
import pytest

from fishsuite.config.schema import FishsuiteConfig
from fishsuite.core import io as _io
from fishsuite.core import morphology as _morph
from fishsuite.core import spots as _spots
from fishsuite.core.modes import rna_only as _rna_only


_LOW_NUCLEAR_PEAK = 100.0
_HIGH_NUCLEAR_PEAK = 900.0
_HIGH_CYTOPLASMIC_PEAK = 800.0


def _detected_spots() -> pd.DataFrame:
    """Three hand-checked spots: two nuclear and one cytoplasmic."""
    return pd.DataFrame(
        {
            "x_px": [5, 6, 10],
            "y_px": [5, 6, 5],
            "z_slice": [1, 1, 1],
            "intensity_peak": [
                _LOW_NUCLEAR_PEAK,
                _HIGH_NUCLEAR_PEAK,
                _HIGH_CYTOPLASMIC_PEAK,
            ],
            "threshold_used": [42.0, 42.0, 42.0],
        }
    )


def _base_cfg() -> FishsuiteConfig:
    cfg = FishsuiteConfig()
    cfg.channels.analysis_mode = "rna_only"
    cfg.channels.dapi = 0
    cfg.channels.rna = 1
    cfg.z_stack.mode = "maxproj"
    cfg.nuclei.exclude_border = False
    cfg.cytoplasm.enabled = True
    cfg.cytoplasm.voronoi_max_expansion_px = 4
    cfg.foci.enabled = True
    cfg.foci.bigfish_spot_radius_nm = 130.0
    cfg.foci.bigfish_spot_radius_z_nm = 300.0
    cfg.foci.threshold_multiplier = 0.7
    cfg.foci.only_nuclear_spots = False
    cfg.output.apply_pub_contrast_floor_to_spots = False
    return cfg


def _run_synthetic(
    monkeypatch: pytest.MonkeyPatch,
    cfg: FishsuiteConfig,
    *,
    observe_stratify: Optional[Callable[[pd.DataFrame], None]] = None,
    observe_diameter: Optional[Callable[[pd.DataFrame], None]] = None,
):
    """Run the real single-channel pipeline on a controlled one-nucleus field."""
    labels = np.zeros((16, 16), dtype=np.int32)
    labels[4:8, 4:8] = 1

    dapi = np.full((16, 16), 10.0, dtype=np.float32)
    dapi[labels == 1] = 5000.0
    rna = np.full((16, 16), 10.0, dtype=np.float32)
    for row in _detected_spots().itertuples(index=False):
        rna[int(row.y_px), int(row.x_px)] = float(row.intensity_peak)

    fake_image = SimpleNamespace(
        n_channels=2,
        n_z=1,
        voxel_xy_nm=100.0,
        voxel_z_nm=300.0,
    )
    detect_calls = []

    monkeypatch.setattr(_io, "read_image", lambda _path: fake_image)

    def _extract(_img, channel, **_kwargs):
        return dapi.copy() if int(channel) == 0 else rna.copy()

    monkeypatch.setattr(_io, "extract_channel", _extract)

    def _detect(_image, **kwargs):
        detect_calls.append(dict(kwargs))
        return _detected_spots().copy()

    monkeypatch.setattr(_spots, "detect_spots", _detect)

    real_stratify = _morph.stratify_spots

    def _stratify(spots, nucleus_labels, *, cytoplasm_labels=None):
        if observe_stratify is not None:
            observe_stratify(spots.copy())
        return real_stratify(
            spots,
            nucleus_labels,
            cytoplasm_labels=cytoplasm_labels,
        )

    monkeypatch.setattr(_morph, "stratify_spots", _stratify)

    def _diameter(_image, spots, _voxel_xy_um, *, fallback_diam_um):
        if observe_diameter is not None:
            observe_diameter(spots.copy())
        return np.full(len(spots), float(fallback_diam_um), dtype=float)

    monkeypatch.setattr(_rna_only, "_measure_spot_diameter_um", _diameter)

    result = _rna_only.run_one(
        Path("synthetic_rna_only.tif"),
        condition="test",
        sec_only=False,
        cfg=cfg,
        precomputed_labels=labels,
    )
    return result, detect_calls


def test_rna_only_resolves_rna_override_for_detection(monkeypatch):
    """A channel override, not the shared fallback, drives RNA detection."""
    cfg = _base_cfg()
    cfg.foci.rna_overrides.bigfish_spot_radius_nm = 260.0
    cfg.foci.rna_overrides.bigfish_spot_radius_z_nm = 480.0
    cfg.foci.rna_overrides.threshold_multiplier = 1.6

    _result, detect_calls = _run_synthetic(monkeypatch, cfg)

    assert len(detect_calls) == 1
    assert detect_calls[0]["spot_radius_nm"] == 260.0
    assert detect_calls[0]["spot_radius_z_nm"] == 480.0
    assert detect_calls[0]["threshold_multiplier"] == 1.6


def test_rna_only_resolved_radii_drive_reported_geometry_and_provenance(monkeypatch):
    """Reported spot geometry and audit fields use the effective RNA radii."""
    cfg = _base_cfg()
    cfg.foci.min_sep_px = 2
    cfg.foci.rna_overrides.bigfish_spot_radius_nm = 260.0
    cfg.foci.rna_overrides.bigfish_spot_radius_z_nm = 480.0
    cfg.foci.rna_overrides.threshold_multiplier = 1.6
    cfg.foci.rna_overrides.only_nuclear_spots = True
    cfg.foci.rna_overrides.min_sep_px = 4

    result, _calls = _run_synthetic(monkeypatch, cfg)

    # Hand-derived from XY radius 260 nm, Z radius 480 nm, XY voxel 100 nm,
    # and Z voxel 300 nm.  The shared defaults (130/300 nm) must not leak in.
    spot = result.spots.iloc[0]
    assert float(spot["spot_diameter_um"]) == pytest.approx(0.52)
    assert float(spot["spot_fwhm_px"]) == pytest.approx(5.2)
    assert float(spot["sigma_z_px_fit"]) == pytest.approx(1.6)
    assert float(spot["fwhm_z_px_fit"]) == pytest.approx(3.768)
    assert float(spot["z_fwhm_slices"]) == pytest.approx(3.768)
    assert float(spot["spot_anisotropy"]) == pytest.approx(480.0 / 260.0)
    assert float(spot["spot_volume_um3"]) == pytest.approx(
        4.0 / 3.0 * np.pi * 0.26**2 * 0.48
    )

    nucleus = result.nuclei.iloc[0]
    assert float(nucleus["mean_spot_anisotropy"]) == pytest.approx(480.0 / 260.0)
    assert float(nucleus["mean_spot_volume_um3"]) == pytest.approx(
        4.0 / 3.0 * np.pi * 0.26**2 * 0.48
    )
    assert float(result.per_image["mean_spot_anisotropy"]) == pytest.approx(
        round(480.0 / 260.0, 3)
    )
    assert float(result.per_image["mean_spot_volume_um3"]) == pytest.approx(
        round(4.0 / 3.0 * np.pi * 0.26**2 * 0.48, 5)
    )

    assert result.thresholds["bigfish_spot_radius_nm"] == 260.0
    assert result.thresholds["bigfish_spot_radius_z_nm"] == 480.0
    assert result.thresholds["rna_bigfish_spot_radius_nm"] == 260.0
    assert result.thresholds["rna_bigfish_spot_radius_z_nm"] == 480.0
    assert result.thresholds["rna_threshold_multiplier"] == 1.6
    assert result.thresholds["rna_only_nuclear_spots"] is True
    assert result.thresholds["rna_min_sep_px"] == 4
    assert result.thresholds["rna_min_spot_peak_intensity"] is None


def test_rna_only_min_peak_floor_runs_before_stratification(monkeypatch, capsys):
    """The absolute RNA peak floor removes low rows before compartment lookup."""
    cfg = _base_cfg()
    cfg.foci.rna_overrides.min_spot_peak_intensity = 500.0

    peaks_at_stratify = []

    def _observe_stratify(spots):
        peaks_at_stratify.extend(spots["intensity_peak"].astype(float).tolist())

    result, _calls = _run_synthetic(
        monkeypatch,
        cfg,
        observe_stratify=_observe_stratify,
    )

    assert peaks_at_stratify == [
        _HIGH_NUCLEAR_PEAK,
        _HIGH_CYTOPLASMIC_PEAK,
    ]
    assert sorted(result.spots["spot_peak_intensity"].astype(float).tolist()) == [
        _HIGH_CYTOPLASMIC_PEAK,
        _HIGH_NUCLEAR_PEAK,
    ]
    nucleus = result.nuclei.iloc[0]
    assert int(nucleus["rna_spot_count"]) == 2
    assert int(nucleus["nuclear_spot_count"]) == 1
    assert int(nucleus["cyto_spot_count"]) == 1
    assert int(result.per_image["total_spots"]) == 2
    assert result.thresholds["rna_min_spot_peak_intensity"] == 500.0
    assert result.per_image["rna_min_spot_peak_intensity"] == 500.0
    assert result.per_image["spots_dropped_below_rna_min_peak"] == 1
    assert (
        "[spot-floor] synthetic_rna_only.tif rna: dropped 1/3 spots below "
        "min_spot_peak_intensity=500.0"
    ) in capsys.readouterr().out


def test_rna_only_nuclear_only_runs_after_stratification_before_outputs(monkeypatch):
    """Nuclear-only keeps stratification real but gates diameter, counts, and export."""
    cfg = _base_cfg()
    cfg.foci.rna_overrides.only_nuclear_spots = True

    peaks_at_stratify = []
    diameter_inputs = []

    def _observe_stratify(spots):
        peaks_at_stratify.extend(spots["intensity_peak"].astype(float).tolist())

    def _observe_diameter(spots):
        diameter_inputs.append(spots.copy())

    result, _calls = _run_synthetic(
        monkeypatch,
        cfg,
        observe_stratify=_observe_stratify,
        observe_diameter=_observe_diameter,
    )

    # The cytoplasmic spot reaches real stratification so its compartment is
    # known.  Only then is it removed from every downstream consumer.
    assert peaks_at_stratify == [
        _LOW_NUCLEAR_PEAK,
        _HIGH_NUCLEAR_PEAK,
        _HIGH_CYTOPLASMIC_PEAK,
    ]

    assert len(diameter_inputs) == 1
    assert diameter_inputs[0]["intensity_peak"].astype(float).tolist() == [
        _LOW_NUCLEAR_PEAK,
        _HIGH_NUCLEAR_PEAK,
    ]
    assert diameter_inputs[0]["in_nucleus"].astype(bool).tolist() == [True, True]
    assert result.spots["spot_peak_intensity"].astype(float).tolist() == [
        _LOW_NUCLEAR_PEAK,
        _HIGH_NUCLEAR_PEAK,
    ]

    nucleus = result.nuclei.iloc[0]
    assert int(nucleus["rna_spot_count"]) == 2
    assert int(nucleus["nuclear_spot_count"]) == 2
    assert int(nucleus["cyto_spot_count"]) == 0
    assert int(result.per_image["total_spots"]) == 2


@pytest.mark.parametrize("override_false", [False, True], ids=["unset", "explicit-false"])
def test_rna_only_false_or_unset_filters_preserve_all_detected_spots(
    monkeypatch,
    override_false,
):
    """Legacy behavior is preserved when the peak floor is unset and nuclear-only is off."""
    cfg = _base_cfg()
    cfg.foci.rna_overrides.min_spot_peak_intensity = None
    if override_false:
        # Explicit per-channel False must win over a True shared fallback.
        cfg.foci.only_nuclear_spots = True
        cfg.foci.rna_overrides.only_nuclear_spots = False

    result, _calls = _run_synthetic(monkeypatch, cfg)

    assert sorted(result.spots["spot_peak_intensity"].astype(float).tolist()) == [
        _LOW_NUCLEAR_PEAK,
        _HIGH_CYTOPLASMIC_PEAK,
        _HIGH_NUCLEAR_PEAK,
    ]
    nucleus = result.nuclei.iloc[0]
    assert int(nucleus["rna_spot_count"]) == 3
    assert int(nucleus["nuclear_spot_count"]) == 2
    assert int(nucleus["cyto_spot_count"]) == 1
    assert int(result.per_image["total_spots"]) == 3
