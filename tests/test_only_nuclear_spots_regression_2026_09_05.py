"""Regression guard: only_nuclear_spots must leave ZERO in_nucleus=0 rows.

The filter was implemented on the unmerged branch ``codex/fix-only-nuclear-spots``
(da5b350, 2026-08-29) and main never carried it, so from 2026-08-29 to 2026-09-05
every rna_rna / rna_protein preset setting the flag asserted a restriction the
engine did not apply. The MIAT x QKI control arm measured the consequence: the
nuclear spot set matched the 2026-08-28 record bit-exactly while 7,574
extra-nuclear spots were retained across 41 images, moving frac_nuclear_rna1 and
every metric pooled over all rna1 spots.

These assert the OUTCOME (row counts in the exported spot table), not the code
path, so they stay valid across refactors of where the gate lives.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import gaussian_filter

from fishsuite.config.schema import FishsuiteConfig, FociChannelOverrideCfg
from fishsuite.core import io as _io
from fishsuite.core.io import ImageWrapper
from fishsuite.core.modes import rna_protein as _rna_protein
from fishsuite.core.modes import rna_rna as _rna_rna

H = W = 200
NZ = 4
DAPI_C, RNA_C, PART_C = 0, 1, 2
NUCLEI = [(70, 70), (70, 130), (130, 100)]
OUTSIDE = [(20, 20), (20, 180), (180, 20), (180, 180), (100, 20)]


class _Bio:
    def __init__(self, czyx):
        self._c = czyx

    def get_image_data(self, order, *, T=0, C=0):  # noqa: N803
        assert order == "ZYX"
        return self._c[C]


def _dapi_plane():
    from skimage.draw import disk

    img = np.random.default_rng(11).uniform(0.0, 20.0, (H, W)).astype(np.float32)
    for cy, cx in NUCLEI:
        rr, cc = disk((cy, cx), 26, shape=img.shape)
        img[rr, cc] += 3000.0
    return img


def _spot_plane(seed_bg, seed_amp):
    """Spots inside the nuclei AND well outside them, so the gate has work to do."""
    img = np.random.default_rng(seed_bg).uniform(2.0, 8.0, (H, W)).astype(np.float32)
    blob = np.zeros_like(img)
    rng = np.random.default_rng(seed_amp)
    pts = []
    for cy, cx in NUCLEI:
        pts += [(int(cy + 12 * np.sin(2 * np.pi * k / 8)),
                 int(cx + 12 * np.cos(2 * np.pi * k / 8))) for k in range(8)]
    for cy, cx in OUTSIDE:
        pts += [(int(cy + 6 * np.sin(2 * np.pi * k / 4)),
                 int(cx + 6 * np.cos(2 * np.pi * k / 4))) for k in range(4)]
    for y, x in pts:
        blob[np.clip(y, 0, H - 1), np.clip(x, 0, W - 1)] += float(rng.uniform(3000, 6000))
    return img + gaussian_filter(blob, 1.1)


@pytest.fixture()
def fake_img() -> ImageWrapper:
    czyx = np.stack(
        [np.stack([p] * NZ, 0)
         for p in (_dapi_plane(), _spot_plane(22, 33), _spot_plane(44, 55))], 0
    ).astype(np.float32)
    return ImageWrapper(path="only_nuclear_regression.tif", bio=_Bio(czyx), scene_idx=0,
                        shape=(1, 3, NZ, H, W), channel_names=["DAPI", "RNA", "PART"],
                        voxel_xy_nm=130.0, voxel_z_nm=300.0, n_channels=3, n_z=NZ)


def _cfg(mode="rna_rna") -> FishsuiteConfig:
    cfg = FishsuiteConfig()
    cfg.channels.dapi = DAPI_C
    cfg.channels.rna = RNA_C
    if mode == "rna_rna":
        cfg.channels.rna2 = PART_C
    else:
        cfg.channels.antibody = PART_C
        cfg.channels.rna2 = -1
    cfg.channels.analysis_mode = mode
    cfg.nuclei.backend = "otsu"
    cfg.nuclei.min_area_px = 120
    cfg.nuclei.max_area_px = 10_000_000
    cfg.nuclei.exclude_border = True
    cfg.nuclei.border_margin_px = 3
    cfg.z_stack.mode = "maxproj"
    cfg.foci.enabled = True
    cfg.foci.backend = "bigfish"
    cfg.foci.threshold_multiplier = 1.0
    cfg.pixel_coloc.threshold_scope = "per_image"
    return cfg


def _run(cfg, img, monkeypatch):
    monkeypatch.setattr(_io, "read_image", lambda p: img)
    mod = _rna_rna if cfg.channels.analysis_mode == "rna_rna" else _rna_protein
    return mod.run_one(Path(img.path), condition="c", sec_only=False, cfg=cfg)


def _counts(spots, channel):
    if spots is None or len(spots) == 0:
        return 0, 0
    sub = spots[spots["channel"].astype(str) == channel]
    inn = pd.to_numeric(sub["in_nucleus"], errors="coerce").fillna(0).astype(int)
    return int((inn == 1).sum()), int((inn == 0).sum())


# ===========================================================================
# The fixture must actually produce extra-nuclear spots, or the tests below
# would pass on an engine that never filters anything.
# ===========================================================================
def test_fixture_produces_extranuclear_spots_when_gate_is_off(fake_img, monkeypatch):
    res = _run(_cfg(), fake_img, monkeypatch)
    for ch in ("rna1", "rna2"):
        n_in, n_out = _counts(res.spots, ch)
        assert n_in > 0, ch
        assert n_out > 0, "%s: no extra-nuclear spots, the gate tests would be vacuous" % ch


# ===========================================================================
# rna channel
# ===========================================================================
def test_shared_gate_leaves_no_extranuclear_rows(fake_img, monkeypatch):
    cfg = _cfg()
    cfg.foci.only_nuclear_spots = True
    res = _run(cfg, fake_img, monkeypatch)
    for ch in ("rna1", "rna2"):
        n_in, n_out = _counts(res.spots, ch)
        assert n_out == 0, "%s retained %d in_nucleus=0 rows" % (ch, n_out)
        assert n_in > 0


def test_rna_override_gates_only_that_channel(fake_img, monkeypatch):
    cfg = _cfg()
    cfg.foci.rna_overrides = FociChannelOverrideCfg(only_nuclear_spots=True)
    res = _run(cfg, fake_img, monkeypatch)
    assert _counts(res.spots, "rna1")[1] == 0
    assert _counts(res.spots, "rna2")[1] > 0, "rna2 must be untouched by an rna override"


# ===========================================================================
# antibody channel in rna_protein, via antibody_overrides. This is the case the
# RNASEH2B and QKI arm-2 presets actually use.
# ===========================================================================
def test_antibody_override_gates_the_protein_channel(fake_img, monkeypatch):
    cfg = _cfg("rna_protein")
    cfg.foci.antibody_overrides = FociChannelOverrideCfg(only_nuclear_spots=True)
    res = _run(cfg, fake_img, monkeypatch)
    n_in, n_out = _counts(res.spots, "protein")
    assert n_out == 0, "protein retained %d in_nucleus=0 rows" % n_out
    assert n_in > 0
    # the RNA channel carries no gate here and must keep its extra-nuclear spots
    assert _counts(res.spots, "rna1")[1] > 0


def test_antibody_override_off_retains_extranuclear_protein_rows(fake_img, monkeypatch):
    res = _run(_cfg("rna_protein"), fake_img, monkeypatch)
    assert _counts(res.spots, "protein")[1] > 0


# ===========================================================================
# The gate must precede the analysis, not just the export: per-nucleus counts
# and the pooled rotation null must see the filtered set.
# ===========================================================================
def test_gate_reaches_per_nucleus_counts(fake_img, monkeypatch):
    cfg = _cfg()
    cfg.foci.only_nuclear_spots = True
    res = _run(cfg, fake_img, monkeypatch)
    nuc = res.nuclei
    total = pd.to_numeric(nuc["rna_spot_count"], errors="coerce").fillna(0)
    nuclear = pd.to_numeric(nuc["nuclear_spot_count"], errors="coerce").fillna(0)
    assert (total == nuclear).all(), "cytoplasmic spots survived into the per-nucleus counts"
    cyto = pd.to_numeric(nuc["cyto_spot_count"], errors="coerce").fillna(0)
    assert (cyto == 0).all()


# ===========================================================================
# fs-calib's scoped anchor flag is a DIFFERENT restriction and must still work
# with the gate off - that is the case it exists for.
# ===========================================================================
def test_partner_anchored_nuclear_anchor_flag_still_works_with_gate_off(
    fake_img, monkeypatch
):
    cfg = _cfg()
    cfg.foci.only_nuclear_spots = False
    cfg.foci.compute_partner_intensity = True
    cfg.foci.compute_partner_rotation_null = True
    cfg.foci.compute_partner_anchored_rotation_null = True
    cfg.foci.partner_rotation_n = 60
    cfg.foci.partner_anchored_null_nuclear_anchors_only = True
    res = _run(cfg, fake_img, monkeypatch)
    # the spot table keeps its extra-nuclear rows: this flag is not a spot gate
    assert _counts(res.spots, "rna2")[1] > 0
    col = "n_partner_anchors_at_rna2_spots"
    assert col in res.nuclei.columns
    anchors = pd.to_numeric(res.nuclei[col], errors="coerce").fillna(0)
    assert (anchors >= 0).all()
    assert anchors.sum() > 0
