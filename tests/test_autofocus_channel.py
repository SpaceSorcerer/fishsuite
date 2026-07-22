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


# ════════════════════════════════════════════════════════════════════════════
# JOINT multi-channel autofocus (2026-07-18 Brian): collapse the z-stack to a
# SINGLE shared plane in focus for DAPI AND rna AND rna2 (partner) at once, so
# colocalization is measured on one jointly-focused optical section.
# ════════════════════════════════════════════════════════════════════════════

# A distinct fixture from ``fake_img`` above: the single-sharp-plane stack has
# no well-defined JOINT optimum (three disjoint delta peaks -> the product is
# noise-dominated between them). Here each channel gets a SMOOTH Gaussian focus
# profile in z (a high-frequency texture whose amplitude is Gaussian-weighted
# about the channel's focus plane), so the per-channel focus curves OVERLAP and
# their normalized product/min has a single, computable maximum.
#
# Focus-plane centers (0-indexed) are DELIBERATELY asymmetric so the joint pick
# differs from EVERY single-channel best:
#   DAPI center 3, RNA center 9, partner center 7.
# With equal-width Gaussians the normalized product peaks near the centroid and
# the normalized min peaks near the DAPI/RNA midpoint; both land on z0=6 for
# this configuration (verified below), i.e. 1-indexed plane 7 — distinct from
# DAPI-best(4), RNA-best(10) and partner-best(8).
JOINT_NZ = 13
JOINT_YX = (48, 48)
JOINT_DAPI_Z0 = 3
JOINT_RNA_Z0 = 9
JOINT_PARTNER_Z0 = 7
JOINT_EXPECTED_Z0 = 6            # 0-indexed joint optimum
JOINT_EXPECTED_Z1 = JOINT_EXPECTED_Z0 + 1   # 1-indexed


def _joint_czyx() -> np.ndarray:
    """(C=3, Z, Y, X) stack with smooth, overlapping per-channel focus curves.

    The SAME high-frequency texture ``H`` is shared by all channels (so the
    per-channel normalized focus curves are identical in shape, just shifted in
    z); only the Gaussian center differs. Bright constant background dominates
    the mean so ``var(laplace(plane/mean))`` tracks the Gaussian amplitude^2 and
    peaks exactly at each channel's center.
    """
    rng = np.random.default_rng(20260718)
    npix = JOINT_YX[0] * JOINT_YX[1]
    H = np.zeros(npix, dtype=np.float32)
    spike_idx = rng.choice(npix, size=int(0.05 * npix), replace=False)
    H[spike_idx] = rng.uniform(200.0, 400.0, size=spike_idx.size).astype(np.float32)
    H = H.reshape(JOINT_YX)
    bg = 1000.0
    sigma = 2.0

    def _chan(center: int) -> np.ndarray:
        planes = []
        for z in range(JOINT_NZ):
            amp = float(np.exp(-0.5 * ((z - center) / sigma) ** 2))
            planes.append((bg + amp * H).astype(np.float32))
        return np.stack(planes, axis=0)

    return np.stack(
        [_chan(JOINT_DAPI_Z0), _chan(JOINT_RNA_Z0), _chan(JOINT_PARTNER_Z0)], axis=0
    )


@pytest.fixture()
def joint_img() -> ImageWrapper:
    czyx = _joint_czyx()
    return ImageWrapper(
        path="joint_synthetic.tif",
        bio=_FakeBio(czyx),
        scene_idx=0,
        shape=(1, 3, JOINT_NZ, JOINT_YX[0], JOINT_YX[1]),
        channel_names=["DAPI", "RNA", "RNA2"],
        voxel_xy_nm=65.0,
        voxel_z_nm=230.0,
        n_channels=3,
        n_z=JOINT_NZ,
    )


# ─── schema ──────────────────────────────────────────────────────────────────
def test_schema_accepts_joint_and_reduce_default():
    z = ZStackCfg(autofocus_channel="joint")
    assert z.autofocus_channel == "joint"
    assert z.autofocus_joint_reduce == "product"        # default reducer
    assert ZStackCfg(autofocus_joint_reduce="geomean").autofocus_joint_reduce == "geomean"
    assert ZStackCfg(autofocus_joint_reduce="min").autofocus_joint_reduce == "min"


def test_schema_rejects_bad_reduce():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ZStackCfg(autofocus_joint_reduce="mean")


# ─── the single-channel best planes really are distinct (fixture sanity) ─────
def test_joint_fixture_single_channel_bests_are_distinct(joint_img):
    d, _ = _io.extract_channel_autofocus_with_idx(joint_img, DAPI_C)
    r, _ = _io.extract_channel_autofocus_with_idx(joint_img, RNA_C)
    p, _ = _io.extract_channel_autofocus_with_idx(joint_img, AB_C)
    assert d == JOINT_DAPI_Z0 + 1
    assert r == JOINT_RNA_Z0 + 1
    assert p == JOINT_PARTNER_Z0 + 1


# ─── joint resolver: product ─────────────────────────────────────────────────
def test_joint_product_picks_joint_plane(joint_img):
    """reduce='product' lands on the plane that jointly maximizes all three,
    which is DIFFERENT from every single-channel best."""
    picked_z, diag = _io.resolve_joint_autofocus_plane(
        joint_img, dapi_idx=DAPI_C, rna_idx=RNA_C, partner_idx=AB_C,
        reduce="product",
    )
    assert picked_z == JOINT_EXPECTED_Z1
    assert diag["reduce"] == "product"
    # genuinely a JOINT compromise: not any single channel's own best plane
    assert picked_z != JOINT_DAPI_Z0 + 1
    assert picked_z != JOINT_RNA_Z0 + 1
    # per-channel focus scores at the chosen plane are reported for QC
    assert set(diag["per_channel_focus_score"]) == {"dapi", "rna", "partner"}


# ─── joint resolver: min ─────────────────────────────────────────────────────
def test_joint_min_picks_joint_plane(joint_img):
    """reduce='min' (worst-channel focus) also lands on the joint plane."""
    picked_z, diag = _io.resolve_joint_autofocus_plane(
        joint_img, dapi_idx=DAPI_C, rna_idx=RNA_C, partner_idx=AB_C,
        reduce="min",
    )
    assert picked_z == JOINT_EXPECTED_Z1
    assert diag["reduce"] == "min"


# ─── geomean is a monotone transform of product -> same pick ─────────────────
def test_joint_geomean_matches_product(joint_img):
    zp, _ = _io.resolve_joint_autofocus_plane(
        joint_img, dapi_idx=DAPI_C, rna_idx=RNA_C, partner_idx=AB_C, reduce="product",
    )
    zg, _ = _io.resolve_joint_autofocus_plane(
        joint_img, dapi_idx=DAPI_C, rna_idx=RNA_C, partner_idx=AB_C, reduce="geomean",
    )
    assert zp == zg == JOINT_EXPECTED_Z1


# ─── partner optional: DAPI+rna only ─────────────────────────────────────────
def test_joint_without_partner_is_dapi_rna_only(joint_img):
    """partner_idx=None -> joint over DAPI+rna only; still a valid single plane
    that is the DAPI/RNA compromise (centroid of centers 3 and 9 -> z0=6)."""
    picked_z, diag = _io.resolve_joint_autofocus_plane(
        joint_img, dapi_idx=DAPI_C, rna_idx=RNA_C, partner_idx=None, reduce="product",
    )
    assert 1 <= picked_z <= JOINT_NZ
    assert set(diag["per_channel_focus_score"]) == {"dapi", "rna"}
    assert picked_z == JOINT_EXPECTED_Z1            # (3+9)/2 = 6 -> 1-indexed 7


# ─── regression: the DAPI anchor is untouched by the joint work ──────────────
def test_joint_fixture_dapi_anchor_unchanged(joint_img):
    """autofocus_channel='dapi' still returns DAPI's OWN sharpest plane."""
    picked_z, ch, _ = _io.resolve_autofocus_plane(
        joint_img, dapi_idx=DAPI_C, rna_idx=RNA_C, autofocus_channel="dapi",
    )
    dapi_z, _ = _io.extract_channel_autofocus_with_idx(joint_img, DAPI_C)
    assert ch == "dapi"
    assert picked_z == dapi_z == JOINT_DAPI_Z0 + 1


# ─── one-plane invariant holds for the joint pick too ────────────────────────
def test_joint_one_plane_invariant(joint_img):
    picked_z, _ = _io.resolve_joint_autofocus_plane(
        joint_img, dapi_idx=DAPI_C, rna_idx=RNA_C, partner_idx=AB_C, reduce="product",
    )
    czyx = joint_img.bio._czyx
    for c in (DAPI_C, RNA_C, AB_C):
        plane = _io.extract_channel_at_z(joint_img, c, z_1indexed=picked_z)
        np.testing.assert_array_equal(plane, czyx[c][picked_z - 1])


# ════════════════════════════════════════════════════════════════════════════
# NUCLEAR-MASKED joint scoring (2026-07-19 Brian): on real data the whole-plane
# joint pick is hijacked by MIAT's CYTOPLASMIC autofluorescent haze (its var-of-
# Laplacian peaks on cytoplasmic texture, not nuclear puncta). Fix: score joint
# focus ONLY inside a DAPI-derived nuclear mask. This fixture reproduces the
# failure mode: a bright DAPI nuclear blob whose IN-nucleus high-frequency
# content (all channels) is sharpest at plane B, surrounded by a strong
# cytoplasmic haze whose OUT-of-nucleus texture is sharpest at a DIFFERENT plane
# A. Whole-plane scoring is dragged to A (haze dominates the frame area); masked
# scoring recovers the true nuclear plane B.
# ════════════════════════════════════════════════════════════════════════════
MASK_NZ = 11
MASK_YX = (48, 48)
MASK_OUT_PEAK_Z0 = 2        # cytoplasmic-haze focus plane (0-indexed) = "A"
MASK_IN_PEAK_Z0 = 8         # nuclear-content focus plane (0-indexed) = "B"


def _smooth_nuclear_envelope(yx, center, rho) -> np.ndarray:
    yy, xx = np.mgrid[0 : yx[0], 0 : yx[1]].astype(np.float32)
    r2 = (yy - center[0]) ** 2 + (xx - center[1]) ** 2
    return np.exp(-0.5 * r2 / (rho * rho)).astype(np.float32)   # 1 at center -> 0 away


def _masked_czyx() -> "tuple[np.ndarray, np.ndarray]":
    rng = np.random.default_rng(20260719)
    ny, nx = MASK_YX
    w_nuc = _smooth_nuclear_envelope(MASK_YX, (ny / 2.0, nx / 2.0), rho=6.0)
    w_cyto = (1.0 - w_nuc).astype(np.float32)

    def _hf() -> np.ndarray:
        f = np.zeros(ny * nx, dtype=np.float32)
        idx = rng.choice(ny * nx, size=int(0.4 * ny * nx), replace=False)
        f[idx] = rng.uniform(60.0, 160.0, size=idx.size).astype(np.float32)
        return f.reshape(MASK_YX)

    H_in, H_out = _hf(), _hf()       # nuclear-content vs cytoplasmic-haze textures
    sigma = 1.6

    def _amp(z: int, center: int) -> float:
        return float(np.exp(-0.5 * ((z - center) / sigma) ** 2))

    def _chan(is_dapi: bool) -> np.ndarray:
        planes = []
        for z in range(MASK_NZ):
            p = np.full(MASK_YX, 100.0, dtype=np.float32)
            if is_dapi:
                p = p + 2800.0 * w_nuc            # bright, smooth nuclear blob (for the mask)
            p = p + _amp(z, MASK_IN_PEAK_Z0) * H_in * w_nuc     # nuclear content sharp at B
            p = p + _amp(z, MASK_OUT_PEAK_Z0) * H_out * w_cyto  # cytoplasmic haze sharp at A
            planes.append(p.astype(np.float32))
        return np.stack(planes, axis=0)

    czyx = np.stack([_chan(True), _chan(False), _chan(False)], axis=0)
    return czyx, w_nuc


@pytest.fixture()
def masked_img() -> ImageWrapper:
    czyx, _ = _masked_czyx()
    return ImageWrapper(
        path="masked_synthetic.tif",
        bio=_FakeBio(czyx),
        scene_idx=0,
        shape=(1, 3, MASK_NZ, MASK_YX[0], MASK_YX[1]),
        channel_names=["DAPI", "RNA", "RNA2"],
        voxel_xy_nm=65.0,
        voxel_z_nm=230.0,
        n_channels=3,
        n_z=MASK_NZ,
    )


# ─── schema ──────────────────────────────────────────────────────────────────
def test_schema_joint_nuclear_mask_default_off():
    assert FishsuiteConfig().z_stack.autofocus_joint_nuclear_mask is False
    assert ZStackCfg(autofocus_joint_nuclear_mask=True).autofocus_joint_nuclear_mask is True


# ─── whole-plane (mask OFF) is hijacked by the cytoplasmic haze ──────────────
def test_joint_whole_plane_hijacked_by_cytoplasm(masked_img):
    picked_z, diag = _io.resolve_joint_autofocus_plane(
        masked_img, dapi_idx=DAPI_C, rna_idx=RNA_C, partner_idx=AB_C,
        reduce="product", nuclear_mask=False,
    )
    assert diag.get("nuclear_mask_used") is False
    # dragged to the cytoplasmic-haze plane A, NOT the true nuclear plane B
    assert picked_z == MASK_OUT_PEAK_Z0 + 1
    assert picked_z != MASK_IN_PEAK_Z0 + 1


# ─── masked (mask ON) recovers the true nuclear plane ────────────────────────
def test_joint_nuclear_mask_recovers_nuclear_plane(masked_img):
    picked_z, diag = _io.resolve_joint_autofocus_plane(
        masked_img, dapi_idx=DAPI_C, rna_idx=RNA_C, partner_idx=AB_C,
        reduce="product", nuclear_mask=True,
    )
    assert diag.get("nuclear_mask_used") is True
    assert 0.01 <= diag.get("nuclear_mask_coverage", 0.0) < 0.9   # a real, partial mask
    assert picked_z == MASK_IN_PEAK_Z0 + 1        # nuclear content wins
    # and it genuinely changed the answer vs whole-plane
    assert picked_z != MASK_OUT_PEAK_Z0 + 1


def test_joint_nuclear_mask_also_works_for_min(masked_img):
    picked_z, diag = _io.resolve_joint_autofocus_plane(
        masked_img, dapi_idx=DAPI_C, rna_idx=RNA_C, partner_idx=AB_C,
        reduce="min", nuclear_mask=True,
    )
    assert diag.get("nuclear_mask_used") is True
    assert picked_z == MASK_IN_PEAK_Z0 + 1


# ─── empty / degenerate mask falls back to whole-plane scoring ───────────────
def test_joint_nuclear_mask_empty_falls_back():
    # DAPI is a flat constant (no nucleus) -> mask derivation yields nothing ->
    # fall back to whole-plane and report it.
    rng = np.random.default_rng(11)
    ny, nx = 32, 32
    nz = 9
    def _chan(const):
        base = np.full((nz, ny, nx), float(const), dtype=np.float32)
        # give RNA channels a sharp plane so the pick is still well-defined
        if const == 0.0:
            base[5] += rng.uniform(0, 500, size=(ny, nx)).astype(np.float32)
        return base
    dapi = np.full((nz, ny, nx), 1000.0, dtype=np.float32)   # perfectly flat DAPI
    czyx = np.stack([dapi, _chan(0.0), _chan(0.0)], axis=0)
    img = ImageWrapper(
        path="flatdapi.tif", bio=_FakeBio(czyx), scene_idx=0,
        shape=(1, 3, nz, ny, nx), channel_names=["DAPI", "RNA", "RNA2"],
        voxel_xy_nm=65.0, voxel_z_nm=230.0, n_channels=3, n_z=nz,
    )
    z_on, diag_on = _io.resolve_joint_autofocus_plane(
        img, dapi_idx=0, rna_idx=1, partner_idx=2, nuclear_mask=True,
    )
    z_off, _ = _io.resolve_joint_autofocus_plane(
        img, dapi_idx=0, rna_idx=1, partner_idx=2, nuclear_mask=False,
    )
    assert diag_on.get("nuclear_mask_used") is False        # fell back
    assert z_on == z_off                                    # identical to whole-plane


# ─── default (mask OFF) leaves the ORIGINAL joint pick byte-for-byte ─────────
def test_joint_nuclear_mask_default_unchanged(joint_img):
    z_default, _ = _io.resolve_joint_autofocus_plane(
        joint_img, dapi_idx=DAPI_C, rna_idx=RNA_C, partner_idx=AB_C, reduce="product",
    )
    z_explicit_off, _ = _io.resolve_joint_autofocus_plane(
        joint_img, dapi_idx=DAPI_C, rna_idx=RNA_C, partner_idx=AB_C,
        reduce="product", nuclear_mask=False,
    )
    assert z_default == z_explicit_off == JOINT_EXPECTED_Z1
