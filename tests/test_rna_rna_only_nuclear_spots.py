"""Regression coverage for per-channel ``only_nuclear_spots`` in two-channel modes.

The detector fixture deliberately returns biologically plausible exported
(cytoplasmic) RNA spots.  The real mode code still performs compartment
stratification, pairing, partner-intensity sampling, footprint sampling, and
the position/rotation nulls; only image I/O and spot detection are replaced by
small deterministic test doubles.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import gaussian_filter
from skimage.draw import disk

from fishsuite.config.schema import FishsuiteConfig
from fishsuite.core import io as _io
from fishsuite.core.io import ImageWrapper
from fishsuite.core.modes import rna_protein as _rna_protein
from fishsuite.core.modes import rna_rna as _rna_rna


H = W = 128
CY = CX = 64
NUCLEAR_RNA1 = (
    (50, 56),
    (53, 68),
    (58, 76),
    (64, 60),
    (67, 72),
    (73, 55),
    (76, 66),
    (70, 80),
)
NUCLEAR_RNA2 = tuple((y + 1, x + 1) for y, x in NUCLEAR_RNA1)
CYTOPLASMIC_RNA1 = (64, 93)
CYTOPLASMIC_RNA2 = (68, 93)


class _FakeBio:
    def __init__(self, czyx: np.ndarray):
        self._czyx = czyx

    def get_image_data(self, order: str, *, T: int = 0, C: int = 0):  # noqa: N803
        assert order == "ZYX"
        return self._czyx[C]


def _labels() -> np.ndarray:
    labels = np.zeros((H, W), dtype=np.int32)
    rr, cc = disk((CY, CX), 25, shape=labels.shape)
    labels[rr, cc] = 1
    return labels


def _paint_blobs(base: np.ndarray, positions, amplitude: float) -> np.ndarray:
    impulses = np.zeros_like(base, dtype=np.float32)
    for y, x in positions:
        impulses[y, x] += float(amplitude)
    return base + gaussian_filter(impulses, 1.0)


def _fake_image(*, flat_rna1: bool = False) -> ImageWrapper:
    labels = _labels()
    dapi = np.full((H, W), 5.0, dtype=np.float32)
    dapi[labels > 0] = 2500.0

    # The cytoplasmic RNA1 punctum is present in the pixels in every run; the
    # test varies only whether the detector reports it.  Thus pixel-coloc stays
    # fixed while spot-derived results expose any leaked exported spot.
    rna1 = _paint_blobs(
        np.full((H, W), 20.0, dtype=np.float32),
        (*NUCLEAR_RNA1, CYTOPLASMIC_RNA1),
        4000.0,
    )
    if flat_rna1:
        # A flat crop forces the footprint sampler onto its nominal-diameter
        # disk fallback, making the resolved RNA1 radius observable.
        rna1 = np.full((H, W), 20.0, dtype=np.float32)

    yy, xx = np.indices((H, W), dtype=np.float32)
    partner = 90.0 + 0.35 * xx + 0.15 * yy
    partner = _paint_blobs(partner, NUCLEAR_RNA1, 1800.0)
    # Make the exported spot strongly partner-bright so a leak has an obvious,
    # deterministic effect on partner, footprint, and null-model statistics.
    partner = _paint_blobs(partner, (CYTOPLASMIC_RNA1,), 9000.0)

    czyx = np.stack([dapi, rna1, partner], axis=0)[:, None, :, :]
    return ImageWrapper(
        path=Path("synthetic_exported_rna.tif"),
        bio=_FakeBio(czyx),
        scene_idx=0,
        shape=(1, 3, 1, H, W),
        channel_names=["DAPI", "RNA1", "PARTNER"],
        voxel_xy_nm=130.0,
        voxel_z_nm=300.0,
        n_channels=3,
        n_z=1,
    )


def _detected_spots(positions, plane: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "spot_id": i,
                "y_px": int(y),
                "x_px": int(x),
                "z_slice": 0,
                "intensity_peak": float(plane[y, x]),
                "threshold_used": 1.0,
            }
            for i, (y, x) in enumerate(positions)
        ]
    )


def _base_cfg(mode: str) -> FishsuiteConfig:
    cfg = FishsuiteConfig()
    cfg.channels.analysis_mode = mode
    cfg.channels.dapi = 0
    cfg.channels.rna = 1
    if mode == "rna_rna":
        cfg.channels.rna2 = 2
    else:
        cfg.channels.antibody = 2
    cfg.z_stack.mode = "maxproj"
    cfg.nuclei.backend = "otsu"
    cfg.nuclei.exclude_border = False
    cfg.cytoplasm.enabled = True
    cfg.cytoplasm.voronoi_max_expansion_px = 20
    cfg.foci.enabled = True
    cfg.foci.drop_floater_spots = False
    cfg.foci.bigfish_voxel_size_nm = 130.0
    cfg.foci.bigfish_voxel_z_nm = 300.0
    cfg.foci.bigfish_spot_radius_nm = 130.0
    cfg.pixel_coloc.threshold_scope = "per_image"
    return cfg


def _set_resolved_nuclear_flags(
    cfg: FishsuiteConfig,
    *,
    rna1_only: bool,
    partner_only: bool,
    inherit_true: bool = False,
) -> None:
    """Set independently resolved flags, optionally exercising inheritance."""
    if inherit_true:
        cfg.foci.only_nuclear_spots = True
        cfg.foci.rna_overrides.only_nuclear_spots = None if rna1_only else False
        partner_override = None if partner_only else False
    else:
        cfg.foci.only_nuclear_spots = False
        cfg.foci.rna_overrides.only_nuclear_spots = bool(rna1_only)
        partner_override = bool(partner_only)

    if cfg.channels.analysis_mode == "rna_rna":
        cfg.foci.rna2_overrides.only_nuclear_spots = partner_override
    else:
        cfg.foci.antibody_overrides.only_nuclear_spots = partner_override


def _run(
    monkeypatch,
    *,
    mode: str,
    include_cytoplasmic_rna1: bool,
    include_cytoplasmic_partner: bool,
    rna1_only: bool,
    partner_only: bool,
    inherit_true: bool = False,
    full_coloc: bool = False,
    configure_cfg=None,
    force_fallback_geometry: bool = False,
    flat_rna1: bool = False,
):
    img = _fake_image(flat_rna1=flat_rna1)
    cfg = _base_cfg(mode)
    _set_resolved_nuclear_flags(
        cfg,
        rna1_only=rna1_only,
        partner_only=partner_only,
        inherit_true=inherit_true,
    )
    if configure_cfg is not None:
        configure_cfg(cfg)
    if full_coloc:
        cfg.foci.compute_partner_intensity = True
        cfg.foci.compute_footprint_enrichment = True
        cfg.foci.compute_partner_null_enrichment = True
        cfg.foci.partner_null_n = 64
        cfg.foci.partner_null_disk_px = 2.0
        cfg.foci.partner_null_seed = 7
        cfg.foci.compute_partner_rotation_null = True
        cfg.foci.partner_rotation_n = 64
        cfg.foci.partner_rotation_seed = 11
        # The contaminated baseline deliberately includes an exported spot.
        # Zero keeps the rotation calculation defined so the regression can
        # show that the leaked coordinate changes it rather than merely making
        # the nucleus unusable.
        cfg.foci.partner_rotation_min_retention = 0.0
        cfg.foci.compute_partner_radial_profile = True

    if force_fallback_geometry:
        # Model a real failed/edge-clipped moment fit: the production exporter
        # and footprint sampler must then apply each channel's nominal radius.
        monkeypatch.setattr(
            _rna_rna,
            "_measure_spot_diameter_um",
            lambda _image, spots, _voxel_xy_um, *, fallback_diam_um: np.full(
                len(spots), np.nan, dtype=float
            ),
        )

    rna1_positions = list(NUCLEAR_RNA1)
    partner_positions = list(NUCLEAR_RNA2)
    if include_cytoplasmic_rna1:
        rna1_positions.append(CYTOPLASMIC_RNA1)
    if include_cytoplasmic_partner:
        partner_positions.append(CYTOPLASMIC_RNA2)

    def fake_detect(plane: np.ndarray, **_kwargs) -> pd.DataFrame:
        # RNA1 has a ~20 background; partner has a ~120 background.
        positions = rna1_positions if float(np.median(plane)) < 60.0 else partner_positions
        return _detected_spots(positions, plane)

    monkeypatch.setattr(_io, "read_image", lambda _path: img)
    monkeypatch.setattr(_rna_rna._spots, "detect_spots", fake_detect)
    mode_module = _rna_rna if mode == "rna_rna" else _rna_protein
    return mode_module.run_one(
        img.path,
        condition="condition",
        sec_only=False,
        cfg=cfg,
        precomputed_labels=_labels(),
    )


def _partner_name(mode: str) -> str:
    return "rna2" if mode == "rna_rna" else "protein"


@pytest.mark.parametrize("mode", ["rna_rna", "rna_protein"])
@pytest.mark.parametrize(
    ("rna1_only", "partner_only"),
    [(True, False), (False, True)],
)
def test_resolved_only_nuclear_spots_is_independent_per_channel(
    mode, rna1_only, partner_only, monkeypatch
):
    """A true resolved flag removes only that channel's exported spots.

    The true value is inherited from the shared setting; the false value is a
    per-channel override.  This catches either channel consulting the wrong
    override (including rna_protein failing to use antibody_overrides).
    """
    res = _run(
        monkeypatch,
        mode=mode,
        include_cytoplasmic_rna1=True,
        include_cytoplasmic_partner=True,
        rna1_only=rna1_only,
        partner_only=partner_only,
        inherit_true=True,
    )
    partner = _partner_name(mode)
    by_channel = {
        channel: frame.reset_index(drop=True)
        for channel, frame in res.spots.groupby("channel")
    }

    rna1 = by_channel["rna1"]
    partner_spots = by_channel[partner]
    assert len(rna1) == len(NUCLEAR_RNA1) + (not rna1_only)
    assert int(rna1["in_cytoplasm"].sum()) == int(not rna1_only)
    assert len(partner_spots) == len(NUCLEAR_RNA2) + (not partner_only)
    assert int(partner_spots["in_cytoplasm"].sum()) == int(not partner_only)

    nucleus = res.nuclei.iloc[0]
    assert int(nucleus["cyto_spot_count"]) == int(not rna1_only)
    assert int(nucleus[f"cyto_spot_count_{partner}"]) == int(not partner_only)
    assert int(res.per_image["total_spots_rna1"]) == len(rna1)
    assert int(res.per_image[f"total_spots_{partner}"]) == len(partner_spots)


@pytest.mark.parametrize("mode", ["rna_rna", "rna_protein"])
def test_nuclear_only_exported_spot_cannot_change_downstream_coloc(mode, monkeypatch):
    """Filtering must precede partner, footprint, position-null, and rotation.

    Adding a detected cytoplasmic RNA1 spot to an otherwise identical image
    must leave every spot-derived colocalization result unchanged when RNA1's
    resolved flag is true.  A late output-only filter would fail this test.
    """
    clean = _run(
        monkeypatch,
        mode=mode,
        include_cytoplasmic_rna1=False,
        include_cytoplasmic_partner=False,
        rna1_only=True,
        partner_only=False,
        full_coloc=True,
    )
    contaminated = _run(
        monkeypatch,
        mode=mode,
        include_cytoplasmic_rna1=True,
        include_cytoplasmic_partner=False,
        rna1_only=True,
        partner_only=False,
        full_coloc=True,
    )
    partner = _partner_name(mode)

    spot_cols = [
        "channel",
        "x_px",
        "y_px",
        "in_nucleus",
        "in_cytoplasm",
        "partner_local_mean_intensity",
        "qki_at_miat_footprint",
        "miat_footprint_area_px",
        "qki_footprint_enrichment",
        "nn_distance_um",
        "paired_at_0p3um",
    ]
    pd.testing.assert_frame_equal(
        clean.spots[spot_cols].reset_index(drop=True),
        contaminated.spots[spot_cols].reset_index(drop=True),
        check_dtype=False,
    )

    nucleus_cols = [
        "n_spots_rna1",
        "cyto_spot_count",
        f"{partner}_local_mean_at_rna1_spots",
        "qki_assoc_ratio_continuous",
        "qki_at_miat_foci_enrichment",
        f"{partner}_enrichment_vs_null_at_rna1_spots",
        f"{partner}_null_z_at_rna1_spots",
        f"{partner}_rotation_enrichment_at_rna1_spots",
        f"{partner}_rotation_null_z_at_rna1_spots",
        f"{partner}_rotation_null_p_at_rna1_spots",
        "median_nn_distance_rna1_um",
        "median_nn_distance_rna2_um".replace("rna2", partner),
        "paired_fraction_rna1_at_0p3um",
        "paired_fraction_rna2_at_0p3um".replace("rna2", partner),
        f"{partner}_radial_enrichment_at_0p25um",
    ]
    pd.testing.assert_frame_equal(
        clean.nuclei[nucleus_cols].reset_index(drop=True),
        contaminated.nuclei[nucleus_cols].reset_index(drop=True),
        check_dtype=False,
        rtol=0.0,
        atol=0.0,
    )

    image_keys = [
        "total_spots_rna1",
        f"mean_{partner}_local_mean_at_rna1_spots",
        "mean_qki_assoc_ratio_continuous",
        f"{partner}_pooled_enrichment_vs_null_at_rna1_spots",
        f"{partner}_pooled_null_z_at_rna1_spots",
        f"{partner}_pooled_rotation_enrichment_at_rna1_spots",
        f"{partner}_pooled_rotation_null_z_at_rna1_spots",
        f"{partner}_pooled_rotation_null_p_empirical_at_rna1_spots",
        "paired_fraction_rna1_at_0p3um",
        "paired_fraction_rna2_at_0p3um".replace("rna2", partner),
        "median_nn_distance_rna1_um_all_spots_in_frame",
        "median_nn_distance_rna2_um_all_spots_in_frame".replace("rna2", partner),
        f"{partner}_radial_pooled_enrichment_at_0p25um",
        f"mean_{partner}_radial_enrichment_at_0p25um",
    ]
    for key in image_keys:
        assert contaminated.per_image[key] == pytest.approx(
            clean.per_image[key], rel=0.0, abs=0.0, nan_ok=True
        ), key

    pd.testing.assert_frame_equal(
        clean.extra["coloc_radial_profile"].reset_index(drop=True),
        contaminated.extra["coloc_radial_profile"].reset_index(drop=True),
        check_dtype=False,
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("mode", ["rna_rna", "rna_protein"])
def test_resolved_channel_radii_drive_fallback_geometry_and_provenance(
    mode, monkeypatch
):
    """Each channel's effective radius drives its own fallback and audit data."""

    def _configure(cfg):
        cfg.foci.rna_overrides.bigfish_spot_radius_nm = 260.0
        cfg.foci.rna_overrides.bigfish_spot_radius_z_nm = 480.0
        cfg.foci.rna_overrides.threshold_multiplier = 1.6
        cfg.foci.rna_overrides.min_sep_px = 3
        cfg.foci.rna_overrides.min_spot_peak_intensity = 11.0

        partner_override = (
            cfg.foci.rna2_overrides
            if mode == "rna_rna"
            else cfg.foci.antibody_overrides
        )
        partner_override.bigfish_spot_radius_nm = 390.0
        partner_override.bigfish_spot_radius_z_nm = 720.0
        partner_override.threshold_multiplier = 1.9
        partner_override.min_sep_px = 5
        partner_override.min_spot_peak_intensity = 22.0
        cfg.foci.compute_footprint_enrichment = True

    result = _run(
        monkeypatch,
        mode=mode,
        include_cytoplasmic_rna1=False,
        include_cytoplasmic_partner=False,
        rna1_only=False,
        partner_only=False,
        configure_cfg=_configure,
        force_fallback_geometry=True,
        flat_rna1=True,
    )
    partner = _partner_name(mode)

    rna1_spots = result.spots[result.spots["channel"] == "rna1"]
    partner_spots = result.spots[result.spots["channel"] == partner]
    assert set(rna1_spots["spot_diameter_um"].astype(float)) == {0.52}
    assert set(partner_spots["spot_diameter_um"].astype(float)) == {0.78}
    # Diameter 0.52 um at 0.13 um/px gives fallback radius 2 px: 13 pixels.
    assert set(rna1_spots["miat_footprint_area_px"].astype(float)) == {13.0}

    prefix = "rna2" if mode == "rna_rna" else "protein"
    expected = {
        "bigfish_spot_radius_nm": 260.0,
        "rna_bigfish_spot_radius_nm": 260.0,
        "rna_bigfish_spot_radius_z_nm": 480.0,
        "rna_threshold_multiplier": 1.6,
        "rna_min_sep_px": 3,
        "rna_min_spot_peak_intensity": 11.0,
        f"{prefix}_bigfish_spot_radius_nm": 390.0,
        f"{prefix}_bigfish_spot_radius_z_nm": 720.0,
        f"{prefix}_threshold_multiplier": 1.9,
        f"{prefix}_min_sep_px": 5,
        f"{prefix}_min_spot_peak_intensity": 22.0,
    }
    for key, value in expected.items():
        assert result.thresholds[key] == value, key


@pytest.mark.parametrize("mode", ["rna_rna", "rna_protein"])
def test_false_retains_exported_spot_and_lets_it_affect_coloc(mode, monkeypatch):
    """An explicit false preserves exported biology and its downstream effect."""
    clean = _run(
        monkeypatch,
        mode=mode,
        include_cytoplasmic_rna1=False,
        include_cytoplasmic_partner=False,
        rna1_only=False,
        partner_only=False,
        full_coloc=True,
    )
    exported = _run(
        monkeypatch,
        mode=mode,
        include_cytoplasmic_rna1=True,
        include_cytoplasmic_partner=False,
        rna1_only=False,
        partner_only=False,
        full_coloc=True,
    )
    partner = _partner_name(mode)

    exported_rna = exported.spots[exported.spots["channel"] == "rna1"]
    assert len(exported_rna) == len(NUCLEAR_RNA1) + 1
    assert int(exported_rna["in_cytoplasm"].sum()) == 1
    assert int(exported.nuclei.iloc[0]["cyto_spot_count"]) == 1
    assert int(exported.per_image["total_spots_rna1"]) == (
        int(clean.per_image["total_spots_rna1"]) + 1
    )

    changed = [
        f"{partner}_local_mean_at_rna1_spots",
        "qki_assoc_ratio_continuous",
        f"{partner}_enrichment_vs_null_at_rna1_spots",
        f"{partner}_rotation_enrichment_at_rna1_spots",
    ]
    for key in changed:
        before = float(clean.nuclei.iloc[0][key])
        after = float(exported.nuclei.iloc[0][key])
        assert np.isfinite(before) and np.isfinite(after), key
        assert not np.isclose(before, after, rtol=1e-6, atol=1e-9), key
