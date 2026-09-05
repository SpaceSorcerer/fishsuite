"""Per-image nuclear counts must describe the IMAGE, not the sampled subset.

Defect (2026-09-05, miatqki-rerun-stage): under fixed-N sampling with
``apply_to_rollups``, ``nuclear_spots_rna1`` / ``cytoplasmic_spots_rna1`` summed
only the sampled nuclei while ``frac_nuclear_rna1`` divided by the whole-image
``total_spots_rna1``. One MIAT x QKI image reported 263 nuclear spots from 10
sampled nuclei against 890 in-nucleus spots over 36 nuclei.

The invariant asserted here is internal consistency, not a fixed number:
``nuclear_spots_* + cytoplasmic_spots_* == total_spots_*`` and
``frac_nuclear_* == nuclear_spots_* / total_spots_*``, with the counts equal to
the sum over ALL nuclei in ``nuclei_metrics.csv``.
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

H = W = 260
NZ = 4
DAPI_C, RNA_C, PART_C = 0, 1, 2
# 6 nuclei so a sample of 2 is a strict subset
NUCLEI = [(60, 60), (60, 130), (60, 200), (150, 60), (150, 130), (150, 200)]


class _Bio:
    def __init__(self, czyx):
        self._c = czyx

    def get_image_data(self, order, *, T=0, C=0):  # noqa: N803
        return self._c[C]


def _dapi_plane():
    from skimage.draw import disk

    img = np.random.default_rng(11).uniform(0.0, 20.0, (H, W)).astype(np.float32)
    for cy, cx in NUCLEI:
        rr, cc = disk((cy, cx), 24, shape=img.shape)
        img[rr, cc] += 3000.0
    return img


def _spot_plane(seed_bg, seed_amp):
    """Nuclear spots in every nucleus plus cytoplasmic spots just outside."""
    img = np.random.default_rng(seed_bg).uniform(2.0, 8.0, (H, W)).astype(np.float32)
    blob = np.zeros_like(img)
    rng = np.random.default_rng(seed_amp)
    pts = []
    for cy, cx in NUCLEI:
        pts += [(int(cy + 11 * np.sin(2 * np.pi * k / 6)),
                 int(cx + 11 * np.cos(2 * np.pi * k / 6))) for k in range(6)]
        pts += [(int(cy + 31 * np.sin(2 * np.pi * k / 3)),
                 int(cx + 31 * np.cos(2 * np.pi * k / 3))) for k in range(3)]
    for y, x in pts:
        blob[np.clip(y, 0, H - 1), np.clip(x, 0, W - 1)] += float(rng.uniform(3000, 6000))
    return img + gaussian_filter(blob, 1.1)


@pytest.fixture()
def fake_img() -> ImageWrapper:
    czyx = np.stack(
        [np.stack([p] * NZ, 0)
         for p in (_dapi_plane(), _spot_plane(22, 33), _spot_plane(44, 55))], 0
    ).astype(np.float32)
    return ImageWrapper(path="per_image_nuclear.tif", bio=_Bio(czyx), scene_idx=0,
                        shape=(1, 3, NZ, H, W), channel_names=["DAPI", "RNA", "PART"],
                        voxel_xy_nm=130.0, voxel_z_nm=300.0, n_channels=3, n_z=NZ)


def _cfg(sample_n=None) -> FishsuiteConfig:
    cfg = FishsuiteConfig()
    cfg.channels.dapi, cfg.channels.rna, cfg.channels.rna2 = DAPI_C, RNA_C, PART_C
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
    cfg.pixel_coloc.threshold_scope = "per_image"
    if sample_n is not None:
        cfg.sampling.enabled = True
        cfg.sampling.n_per_unit = sample_n
        cfg.sampling.unit = "per_image"
        cfg.sampling.apply_to_rollups = True
    return cfg


def _run(cfg, img, monkeypatch):
    monkeypatch.setattr(_io, "read_image", lambda p: img)
    return _rna_rna.run_one(Path(img.path), condition="c", sec_only=False, cfg=cfg)


def _sum(df, col):
    return int(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())


CHANNELS = [
    ("rna1", "nuclear_spots_rna1", "cytoplasmic_spots_rna1", "total_spots_rna1",
     "frac_nuclear_rna1", "nuclear_spot_count", "cyto_spot_count"),
    ("rna2", "nuclear_spots_rna2", "cytoplasmic_spots_rna2", "total_spots_rna2",
     "frac_nuclear_rna2", "nuclear_spot_count_rna2", "cyto_spot_count_rna2"),
]


def test_sampling_actually_selects_a_strict_subset(fake_img, monkeypatch):
    """Guard: without this the sampled tests below would be vacuous."""
    res = _run(_cfg(sample_n=2), fake_img, monkeypatch)
    nuc = res.nuclei
    assert "sampled_in_analysis" in nuc.columns
    n_all = len(nuc)
    n_sampled = int(nuc["sampled_in_analysis"].astype(bool).sum())
    assert n_all >= 4, "fixture produced only %d nuclei" % n_all
    assert 0 < n_sampled < n_all, "sampled %d of %d" % (n_sampled, n_all)


@pytest.mark.parametrize("sample_n", [None, 2])
@pytest.mark.parametrize("ch", CHANNELS, ids=[c[0] for c in CHANNELS])
def test_per_image_nuclear_counts_cover_all_nuclei(ch, sample_n, fake_img, monkeypatch):
    _, k_nuc, k_cyto, k_tot, k_frac, c_nuc, c_cyto = ch
    res = _run(_cfg(sample_n=sample_n), fake_img, monkeypatch)
    pi, nuc = res.per_image, res.nuclei

    # nuclei_metrics.csv always holds EVERY nucleus, sampled or not
    assert _sum(nuc, c_nuc) == int(pi[k_nuc]), (
        "%s = %d but the sum over all %d nuclei is %d"
        % (k_nuc, int(pi[k_nuc]), len(nuc), _sum(nuc, c_nuc))
    )
    assert _sum(nuc, c_cyto) == int(pi[k_cyto])


@pytest.mark.parametrize("sample_n", [None, 2])
@pytest.mark.parametrize("ch", CHANNELS, ids=[c[0] for c in CHANNELS])
def test_nuclear_plus_cytoplasmic_equals_total(ch, sample_n, fake_img, monkeypatch):
    _, k_nuc, k_cyto, k_tot, k_frac, _, _ = ch
    pi = _run(_cfg(sample_n=sample_n), fake_img, monkeypatch).per_image
    assert int(pi[k_nuc]) + int(pi[k_cyto]) == int(pi[k_tot]), (
        "%s %d + %s %d != %s %d"
        % (k_nuc, int(pi[k_nuc]), k_cyto, int(pi[k_cyto]), k_tot, int(pi[k_tot]))
    )


@pytest.mark.parametrize("sample_n", [None, 2])
@pytest.mark.parametrize("ch", CHANNELS, ids=[c[0] for c in CHANNELS])
def test_frac_nuclear_matches_its_own_numerator_and_denominator(
    ch, sample_n, fake_img, monkeypatch
):
    _, k_nuc, _, k_tot, k_frac, _, _ = ch
    pi = _run(_cfg(sample_n=sample_n), fake_img, monkeypatch).per_image
    assert int(pi[k_tot]) > 0
    assert float(pi[k_frac]) == pytest.approx(int(pi[k_nuc]) / float(pi[k_tot]))


@pytest.mark.parametrize("ch", CHANNELS, ids=[c[0] for c in CHANNELS])
def test_sampling_does_not_change_the_image_level_counts(ch, fake_img, monkeypatch):
    """The image-level counts describe the image, so sampling must not move them."""
    _, k_nuc, k_cyto, k_tot, k_frac, _, _ = ch
    unsampled = _run(_cfg(), fake_img, monkeypatch).per_image
    sampled = _run(_cfg(sample_n=2), fake_img, monkeypatch).per_image
    for k in (k_nuc, k_cyto, k_tot):
        assert int(sampled[k]) == int(unsampled[k]), k
    assert float(sampled[k_frac]) == pytest.approx(float(unsampled[k_frac]))
