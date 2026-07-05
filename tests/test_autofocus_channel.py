"""RNA-anchored single-plane autofocus tests (Brian, 2026-07-05).

Feature: ``z_stack.autofocus_channel`` = "dapi" (default, unchanged) / "rna"
(pick the sharpest RNA1 plane, lock every channel to it) / "auto" (RNA-anchor
when the RNA channel clears a signal-quality threshold, else DAPI-anchor).

These tests cover:
  * schema field defaults + validation,
  * ``io.rna_plane_quality`` (focusability + dynamic-range readout),
  * ``io.resolve_autofocus_plane`` routing: "rna" picks a DIFFERENT plane than
    DAPI on a synthetic stack whose DAPI-best and RNA-best planes differ, and
    "auto" gates on the RNA quality score.

The default DAPI-anchor path is exercised by the existing
``test_autofocus_zlock.py`` (which this feature leaves byte-identical).
"""
from __future__ import annotations

import numpy as np
import pytest

from fishsuite.core import io as _io
from fishsuite.core.io import ImageWrapper
from fishsuite.config.schema import FishsuiteConfig, ZStackCfg


# ── synthetic stack: DAPI sharpest at one plane, RNA at a DIFFERENT plane ──
DAPI_C = 0
RNA_C = 1
AB_C = 2
NZ = 12
YX = (32, 32)
DAPI_SHARP_Z0 = 4
RNA_SHARP_Z0 = 9
AB_SHARP_Z0 = 1


class _FakeBio:
    def __init__(self, czyx: np.ndarray):
        self._czyx = czyx

    def get_image_data(self, order: str, *, T: int = 0, C: int = 0):  # noqa: N803
        assert order == "ZYX"
        return self._czyx[C]


def _sharp_plane(rng: np.random.Generator) -> np.ndarray:
    """Bright single-pixel spikes on a dark bg -> high focus + dynamic range."""
    plane = rng.uniform(5.0, 15.0, size=YX).astype(np.float32)
    for _ in range(20):
        y = int(rng.integers(2, YX[0] - 2))
        x = int(rng.integers(2, YX[1] - 2))
        plane[y, x] += 4000.0
    return plane


def _blurry_plane(rng: np.random.Generator) -> np.ndarray:
    yy, xx = np.mgrid[0 : YX[0], 0 : YX[1]].astype(np.float32)
    return (50.0 + 0.5 * yy + 0.5 * xx + rng.uniform(0.0, 2.0, size=YX)).astype(np.float32)


def _build_channel(sharp_z0: int, rng: np.random.Generator) -> np.ndarray:
    zyx = np.stack([_blurry_plane(rng) for _ in range(NZ)], axis=0)
    zyx[sharp_z0] = _sharp_plane(rng)
    return zyx


@pytest.fixture()
def fake_img() -> ImageWrapper:
    rng = np.random.default_rng(20260705)
    czyx = np.stack(
        [
            _build_channel(DAPI_SHARP_Z0, rng),
            _build_channel(RNA_SHARP_Z0, rng),
            _build_channel(AB_SHARP_Z0, rng),
        ],
        axis=0,
    )
    return ImageWrapper(
        path="synthetic.tif",
        bio=_FakeBio(czyx),
        scene_idx=0,
        shape=(1, 3, NZ, YX[0], YX[1]),
        channel_names=["DAPI", "RNA", "AB"],
        voxel_xy_nm=65.0,
        voxel_z_nm=230.0,
        n_channels=3,
        n_z=NZ,
    )


# ─── schema ────────────────────────────────────────────────────────────────
def test_schema_defaults():
    z = FishsuiteConfig().z_stack
    assert z.autofocus_channel == "dapi"          # default = unchanged behaviour
    assert z.autofocus_auto_rna_quality_min == 3.0


def test_schema_accepts_rna_and_auto():
    assert ZStackCfg(autofocus_channel="rna").autofocus_channel == "rna"
    assert ZStackCfg(autofocus_channel="auto").autofocus_channel == "auto"


def test_schema_rejects_bad_channel():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ZStackCfg(autofocus_channel="qki")


# ─── rna_plane_quality ───────────────────────────────────────────────────────
def test_rna_plane_quality_sharp_vs_flat():
    rng = np.random.default_rng(1)
    sharp = _sharp_plane(rng)
    flat = np.full(YX, 100.0, dtype=np.float32)

    qs = _io.rna_plane_quality(sharp)
    qf = _io.rna_plane_quality(flat)

    # Sharp punctate plane: large dynamic range + finite positive focus.
    assert np.isfinite(qs["focus_score"]) and qs["focus_score"] > 0
    assert qs["dynamic_range"] > 10.0
    # Perfectly flat plane: no dynamic range, no structure.
    assert qf["dynamic_range"] == 0.0

    # Degenerate input never raises.
    assert isinstance(_io.rna_plane_quality(np.empty((0, 0))), dict)


# ─── resolve_autofocus_plane routing ────────────────────────────────────────
def test_rna_anchor_picks_rna_plane(fake_img):
    """autofocus_channel='rna' picks RNA's sharp plane, NOT DAPI's."""
    dapi_z, _ = _io.extract_channel_autofocus_with_idx(fake_img, DAPI_C)
    picked_z, ch, diag = _io.resolve_autofocus_plane(
        fake_img, dapi_idx=DAPI_C, rna_idx=RNA_C, autofocus_channel="rna",
    )
    assert ch == "rna"
    assert picked_z == RNA_SHARP_Z0 + 1           # 1-indexed RNA sharp plane
    assert picked_z != dapi_z                     # genuinely different from DAPI
    assert diag["rna_z"] == RNA_SHARP_Z0 + 1


def test_dapi_alias_falls_back(fake_img):
    """Defensive: an explicit 'dapi' routes to the DAPI anchor."""
    picked_z, ch, _ = _io.resolve_autofocus_plane(
        fake_img, dapi_idx=DAPI_C, rna_idx=RNA_C, autofocus_channel="dapi",
    )
    dapi_z, _ = _io.extract_channel_autofocus_with_idx(fake_img, DAPI_C)
    assert ch == "dapi"
    assert picked_z == dapi_z == DAPI_SHARP_Z0 + 1


def test_auto_uses_rna_when_quality_high(fake_img):
    """auto: high RNA dynamic range -> RNA-anchor."""
    picked_z, ch, diag = _io.resolve_autofocus_plane(
        fake_img, dapi_idx=DAPI_C, rna_idx=RNA_C,
        autofocus_channel="auto", auto_rna_quality_min=3.0,
    )
    assert ch == "rna"
    assert picked_z == RNA_SHARP_Z0 + 1
    assert np.isfinite(diag["rna_quality_score"]) and diag["rna_quality_score"] >= 3.0


def test_auto_falls_back_to_dapi_when_quality_below_threshold(fake_img):
    """auto: an unreachably high threshold forces the DAPI anchor."""
    dapi_z, _ = _io.extract_channel_autofocus_with_idx(fake_img, DAPI_C)
    picked_z, ch, _ = _io.resolve_autofocus_plane(
        fake_img, dapi_idx=DAPI_C, rna_idx=RNA_C,
        autofocus_channel="auto", auto_rna_quality_min=1e12,
    )
    assert ch == "dapi"
    assert picked_z == dapi_z == DAPI_SHARP_Z0 + 1


def test_one_plane_invariant_preserved(fake_img):
    """Whichever channel anchors, ALL channels read at the SAME single plane."""
    picked_z, _, _ = _io.resolve_autofocus_plane(
        fake_img, dapi_idx=DAPI_C, rna_idx=RNA_C, autofocus_channel="rna",
    )
    dapi_2d = _io.extract_channel_at_z(fake_img, DAPI_C, z_1indexed=picked_z)
    rna_2d = _io.extract_channel_at_z(fake_img, RNA_C, z_1indexed=picked_z)
    ab_2d = _io.extract_channel_at_z(fake_img, AB_C, z_1indexed=picked_z)
    czyx = fake_img.bio._czyx
    np.testing.assert_array_equal(dapi_2d, czyx[DAPI_C][picked_z - 1])
    np.testing.assert_array_equal(rna_2d, czyx[RNA_C][picked_z - 1])
    np.testing.assert_array_equal(ab_2d, czyx[AB_C][picked_z - 1])
