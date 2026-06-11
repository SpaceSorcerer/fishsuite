"""Radial QKI-around-MIAT profile — unit tests (Brian, 2026-06-06).

PIPELINE-NATIVE radial colocalization profile: instead of a single disk-mean
partner (QKI) intensity at the rna1 (MIAT) spots, sweep CONCENTRIC ANNULI of
increasing radius around each spot and compare the partner intensity in each
ring to the SAME-ring intensity at random in-nucleus positions (the per-ring
null). Gated behind ``foci.compute_partner_radial_profile`` (requires
``compute_partner_intensity``); default OFF -> no ``coloc_radial_profile`` in
res.extra (byte-equivalent BIN1 / H9).

Two families of GPU-free tests:
  (a) helper-level on a synthetic partner with a bright ring at a KNOWN radius
      -> the matching annulus has the peak enrichment; determinism via seed;
      nucleolus exclusion raises the per-ring null floor;
  (b) end-to-end through run_one: default OFF -> key absent; ON -> the
      ``coloc_radial_profile`` DataFrame is emitted with the spec columns.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import gaussian_filter

from fishsuite.config.schema import FishsuiteConfig
from fishsuite.core import io as _io
from fishsuite.core.io import ImageWrapper
from fishsuite.core.modes import rna_rna as _rna_rna
from fishsuite.core.modes.rna_rna import (
    _annulus_stencils,
    _radial_profile_for_nucleus,
)


# ---------------------------------------------------------------------------
# (a) HELPER-LEVEL: annulus stencils + known-radius peak
# ---------------------------------------------------------------------------
def test_annulus_stencils_partition_by_radius():
    """Concentric rings: ring 0 is the inner disk (includes the center); every
    higher ring is an annulus whose offsets fall in (edge[i], edge[i+1]]; rings
    are disjoint and non-empty for coarse bins."""
    bins = [3.0, 6.0, 9.0, 12.0]
    stencils = _annulus_stencils(bins)
    assert len(stencils) == len(bins)
    edges = [0.0] + bins
    seen = set()
    for i, (dy, dx) in enumerate(stencils):
        assert dy.shape == dx.shape
        assert dy.size > 0
        d = np.sqrt(dy.astype(float) ** 2 + dx.astype(float) ** 2)
        assert np.all(d <= edges[i + 1] + 1e-9)
        if i == 0:
            assert np.any((dy == 0) & (dx == 0))  # center in ring 0
        else:
            assert np.all(d > edges[i] - 1e-9)
        # disjoint
        for (yy, xx) in zip(dy.tolist(), dx.tolist()):
            assert (yy, xx) not in seen
            seen.add((yy, xx))


def test_radial_enrichment_peaks_at_known_annulus():
    """A bright ring painted at d in (6, 9] around a single spot -> the matching
    annulus (ring index 2 for bins [3,6,9,12]) has the maximal enrichment."""
    H = W = 60
    cy = cx = 30
    partner = np.full((H, W), 10.0)
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    partner[(d > 6.0) & (d <= 9.0)] = 1000.0  # bright ring in the 3rd annulus

    bins = [3.0, 6.0, 9.0, 12.0]
    stencils = _annulus_stencils(bins)
    spot_cy = np.array([cy]); spot_cx = np.array([cx])
    nys, nxs = np.where(np.ones((H, W), bool))
    rng = np.random.default_rng(0)
    rows = _radial_profile_for_nucleus(
        partner, spot_cy, spot_cx, nys, nxs, stencils, 500, rng
    )
    assert len(rows) == len(bins)
    enr = []
    for (obs, nmean, nsd, nsp) in rows:
        assert nsp == 1
        enr.append(obs / nmean if nmean > 0 else np.nan)
    enr = np.array(enr)
    # ring index 2 == the (6, 9] annulus holds the bright ring
    assert int(np.nanargmax(enr)) == 2
    assert enr[2] > 1.5


def test_radial_profile_is_deterministic_with_seed():
    """Fixed seed -> identical per-ring null means (reproducibility)."""
    H = W = 50
    partner = np.random.default_rng(5).uniform(0, 100, (H, W))
    bins = [3.0, 6.0, 9.0]
    stencils = _annulus_stencils(bins)
    spot_cy = np.array([20, 25, 30]); spot_cx = np.array([20, 25, 30])
    nys, nxs = np.where(np.ones((H, W), bool))
    r1 = _radial_profile_for_nucleus(
        partner, spot_cy, spot_cx, nys, nxs, stencils, 300, np.random.default_rng(0)
    )
    r2 = _radial_profile_for_nucleus(
        partner, spot_cy, spot_cx, nys, nxs, stencils, 300, np.random.default_rng(0)
    )
    for (a, b) in zip(r1, r2):
        assert a[0] == b[0]              # obs identical
        assert a[1] == pytest.approx(b[1], abs=0.0)  # null_mean identical
        assert a[2] == pytest.approx(b[2], abs=0.0)  # null_sd identical


def test_radial_nucleolus_exclusion_raises_null_floor():
    """Excluding the DIM (nucleolar) void from the random sampling positions
    RAISES the per-ring null floor (random ring-samples can no longer be
    centered in the void) — mirrors the disk-null nucleolus test."""
    H = W = 60
    partner = np.full((H, W), 10.0)
    partner[:, :30] = 100.0  # bright left, dim right ("nucleolus")
    bins = [2.0, 4.0]
    stencils = _annulus_stencils(bins)
    spot_cy = np.array([10, 20, 30, 40]); spot_cx = np.array([8, 10, 12, 8])  # bright side
    full = np.ones((H, W), bool)
    nucleolus = np.zeros((H, W), bool); nucleolus[:, 30:] = True
    nucleoplasm = full & (~nucleolus)
    nys_f, nxs_f = np.where(full)
    nys_e, nxs_e = np.where(nucleoplasm)
    r_full = _radial_profile_for_nucleus(
        partner, spot_cy, spot_cx, nys_f, nxs_f, stencils, 1000, np.random.default_rng(0)
    )
    r_excl = _radial_profile_for_nucleus(
        partner, spot_cy, spot_cx, nys_e, nxs_e, stencils, 1000, np.random.default_rng(0)
    )
    for (a, b) in zip(r_full, r_excl):
        assert a[0] == b[0]            # observed unchanged (spots not excluded)
        assert b[1] > a[1]             # exclusion raises the null floor per ring


# ---------------------------------------------------------------------------
# Synthetic 3-channel stack (DAPI + rna1 spots + a dense-nuclear partner) for
# the end-to-end run_one ON/OFF contract.
# ---------------------------------------------------------------------------
DAPI_C, RNA_C, PART_C = 0, 1, 2
NZ = 4
H = W = 200


class _FakeBio:
    def __init__(self, czyx: np.ndarray):
        self._czyx = czyx

    def get_image_data(self, order: str, *, T: int = 0, C: int = 0):  # noqa: N803
        assert order == "ZYX"
        return self._czyx[C]


def _nuclei_centers():
    return [(70, 70), (70, 130), (130, 100)]


def _dapi_plane():
    from skimage.draw import disk
    img = np.random.default_rng(11).uniform(0.0, 20.0, (H, W)).astype(np.float32)
    for (cy, cx) in _nuclei_centers():
        rr, cc = disk((cy, cx), 28, shape=img.shape)
        img[rr, cc] += 3000.0
        hr, hc = disk((cy, cx), 7, shape=img.shape)
        img[hr, hc] = 200.0
    return img


def _rna_spot_plane():
    img = np.random.default_rng(22).uniform(2.0, 8.0, (H, W)).astype(np.float32)
    blob = np.zeros_like(img)
    rng = np.random.default_rng(33)
    for (cy, cx) in _nuclei_centers():
        for k in range(8):
            ang = 2 * np.pi * k / 8
            y = int(cy + 15 * np.sin(ang)); x = int(cx + 15 * np.cos(ang))
            blob[y, x] += float(rng.uniform(3000.0, 6000.0))
    return img + gaussian_filter(blob, 1.1)


def _partner_plane_dense_nuclear():
    from skimage.draw import disk
    img = np.random.default_rng(44).uniform(2.0, 8.0, (H, W)).astype(np.float32)
    for (cy, cx) in _nuclei_centers():
        rr, cc = disk((cy, cx), 28, shape=img.shape)
        img[rr, cc] = 800.0
        hr, hc = disk((cy, cx), 7, shape=img.shape)
        img[hr, hc] = 20.0
    return img


def _czyx() -> np.ndarray:
    planes = [_dapi_plane(), _rna_spot_plane(), _partner_plane_dense_nuclear()]
    return np.stack(
        [np.stack([p] * NZ, axis=0) for p in planes], axis=0
    ).astype(np.float32)


@pytest.fixture()
def fake_img() -> ImageWrapper:
    czyx = _czyx()
    return ImageWrapper(
        path="synthetic_partner_radial.tif",
        bio=_FakeBio(czyx),
        scene_idx=0,
        shape=(1, 3, NZ, H, W),
        channel_names=["DAPI", "RNA", "PART"],
        voxel_xy_nm=130.0,
        voxel_z_nm=300.0,
        n_channels=3,
        n_z=NZ,
    )


def _base_cfg() -> FishsuiteConfig:
    cfg = FishsuiteConfig()
    cfg.channels.dapi = DAPI_C
    cfg.channels.rna = RNA_C
    cfg.channels.rna2 = PART_C
    cfg.channels.analysis_mode = "rna_rna"
    cfg.nuclei.backend = "otsu"
    cfg.nuclei.min_area_px = 120
    cfg.nuclei.max_area_px = 10_000_000
    cfg.nuclei.exclude_border = True
    cfg.nuclei.border_margin_px = 3
    cfg.z_stack.mode = "maxproj"
    cfg.cytoplasm.enabled = True
    cfg.foci.enabled = True
    cfg.foci.backend = "bigfish"
    cfg.foci.threshold_multiplier = 1.0
    cfg.foci.drop_floater_spots = False
    cfg.pixel_coloc.threshold_scope = "per_image"
    return cfg


def _run(cfg, img, monkeypatch):
    monkeypatch.setattr(_io, "read_image", lambda p: img)
    return _rna_rna.run_one(
        Path(img.path), condition="cond", sec_only=False, cfg=cfg,
    )


# ---------------------------------------------------------------------------
# (b) END-TO-END: ON emits coloc_radial_profile; default OFF -> absent
# ---------------------------------------------------------------------------
def test_radial_default_off_no_extra(fake_img, monkeypatch):
    """compute_partner_radial_profile defaults OFF -> the key is ABSENT in
    res.extra even with partner-intensity / null ON (byte-equivalent carrier)."""
    cfg = _base_cfg()
    cfg.foci.compute_partner_intensity = True
    cfg.foci.compute_partner_null_enrichment = True
    cfg.foci.partner_null_n = 100
    res = _run(cfg, fake_img, monkeypatch)
    assert "coloc_radial_profile" not in res.extra


def test_radial_on_emits_profile(fake_img, monkeypatch):
    """compute_partner_radial_profile ON -> res.extra carries a DataFrame with
    one row per ring and the spec columns; enrichment / z are finite."""
    cfg = _base_cfg()
    cfg.foci.compute_partner_intensity = True
    cfg.foci.compute_partner_radial_profile = True
    cfg.foci.partner_radial_bins_um = [0.25, 0.5, 0.75, 1.0]
    cfg.foci.partner_null_n = 100
    res = _run(cfg, fake_img, monkeypatch)

    assert "coloc_radial_profile" in res.extra
    rad = res.extra["coloc_radial_profile"]
    assert isinstance(rad, pd.DataFrame)
    assert {
        "image", "condition", "ring_um", "obs_mean", "null_mean",
        "null_sd", "enrichment", "z", "n_spots",
    }.issubset(rad.columns)
    assert len(rad) == 4
    assert sorted(rad["ring_um"].tolist()) == [0.25, 0.5, 0.75, 1.0]
    assert np.isfinite(pd.to_numeric(rad["enrichment"], errors="coerce")).any()


def test_radial_on_is_deterministic(fake_img, monkeypatch):
    """Same seed -> identical radial enrichment across two runs."""
    cfg = _base_cfg()
    cfg.foci.compute_partner_intensity = True
    cfg.foci.compute_partner_radial_profile = True
    cfg.foci.partner_null_n = 100
    r1 = _run(cfg, fake_img, monkeypatch).extra["coloc_radial_profile"]
    r2 = _run(cfg, fake_img, monkeypatch).extra["coloc_radial_profile"]
    np.testing.assert_allclose(
        r1["enrichment"].to_numpy(dtype=float),
        r2["enrichment"].to_numpy(dtype=float),
        rtol=0, atol=0, equal_nan=True,
    )
