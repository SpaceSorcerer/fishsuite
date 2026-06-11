"""Autofocus z-lock regression test (Brian, 2026-05-28).

Requirement: in ``z_mode == "autofocus"`` the pipeline must detect spots /
measure every non-DAPI channel on the EXACT SAME single z-plane that DAPI
segmentation uses. Before this fix DAPI and RNA were autofocused
INDEPENDENTLY (each called ``extract_channel(z_mode="autofocus")``) and could
land on different planes.

This test exercises the smallest refactored unit — the io helpers the mode
runners now call in the autofocus branch:
  - ``extract_channel_autofocus_with_idx`` picks DAPI's sharpest plane and
    returns its 1-indexed z.
  - ``extract_channel_at_z`` then reads any other channel at that EXACT z.

We build a synthetic multi-channel z-stack where DAPI is sharpest at a known
plane and RNA is sharpest at a DIFFERENT plane, then assert the plane USED for
RNA equals the DAPI-chosen plane (NOT RNA's own independent best). A
full ImageWrapper / bioio reader is unnecessary: we wrap the numpy stacks in a
tiny fake ``bio`` shim exposing ``get_image_data("ZYX", T=0, C=idx)``.
"""
from __future__ import annotations

import numpy as np
import pytest

from fishsuite.core import io as _io
from fishsuite.core.io import ImageWrapper


# Channel layout for the synthetic stack.
DAPI_C = 0
RNA_C = 1
AB_C = 2

NZ = 12
YX = (32, 32)

# Planes (0-indexed) where each channel is engineered to be SHARPEST. They are
# all DIFFERENT so an independent per-channel autofocus would disagree.
DAPI_SHARP_Z0 = 5
RNA_SHARP_Z0 = 9
AB_SHARP_Z0 = 2


class _FakeBio:
    """Minimal stand-in for bioio.BioImage.get_image_data used by the io helpers.

    Stores a (C, Z, Y, X) array and returns the requested channel's ZYX slab.
    """

    def __init__(self, czyx: np.ndarray):
        self._czyx = czyx

    def get_image_data(self, order: str, *, T: int = 0, C: int = 0):  # noqa: N803
        assert order == "ZYX"
        assert T == 0
        return self._czyx[C]


def _sharp_plane(rng: np.random.Generator) -> np.ndarray:
    """A high-variance-of-Laplacian plane: bright point sources on dark bg."""
    plane = rng.uniform(5.0, 15.0, size=YX).astype(np.float32)
    # Scatter a handful of sharp single-pixel spikes -> high Laplacian variance.
    for _ in range(20):
        y = int(rng.integers(2, YX[0] - 2))
        x = int(rng.integers(2, YX[1] - 2))
        plane[y, x] += 4000.0
    return plane


def _blurry_plane(rng: np.random.Generator) -> np.ndarray:
    """A low-variance-of-Laplacian plane: smooth gradient, no sharp edges."""
    yy, xx = np.mgrid[0 : YX[0], 0 : YX[1]].astype(np.float32)
    return (50.0 + 0.5 * yy + 0.5 * xx + rng.uniform(0.0, 2.0, size=YX)).astype(
        np.float32
    )


def _build_channel(sharp_z0: int, rng: np.random.Generator) -> np.ndarray:
    """ZYX stack: one sharp plane at ``sharp_z0``, blurry everywhere else."""
    zyx = np.stack([_blurry_plane(rng) for _ in range(NZ)], axis=0)
    zyx[sharp_z0] = _sharp_plane(rng)
    return zyx


@pytest.fixture()
def fake_img() -> ImageWrapper:
    rng = np.random.default_rng(20260528)
    czyx = np.stack(
        [
            _build_channel(DAPI_SHARP_Z0, rng),  # DAPI
            _build_channel(RNA_SHARP_Z0, rng),   # RNA  (different plane)
            _build_channel(AB_SHARP_Z0, rng),    # antibody (different plane)
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


def test_fixture_is_nondegenerate(fake_img):
    """Sanity: each channel's independent autofocus picks its OWN sharp plane,
    so locking actually changes the RNA / AB plane (the bug we are guarding)."""
    dapi_z, _ = _io.extract_channel_autofocus_with_idx(fake_img, DAPI_C)
    rna_z, _ = _io.extract_channel_autofocus_with_idx(fake_img, RNA_C)
    ab_z, _ = _io.extract_channel_autofocus_with_idx(fake_img, AB_C)
    # 1-indexed
    assert dapi_z == DAPI_SHARP_Z0 + 1
    assert rna_z == RNA_SHARP_Z0 + 1
    assert ab_z == AB_SHARP_Z0 + 1
    assert dapi_z != rna_z != ab_z  # all distinct -> independent autofocus disagrees


def test_rna_locked_to_dapi_plane(fake_img):
    """The locked RNA plane must equal DAPI's autofocus plane, NOT RNA's own."""
    dapi_z, dapi_plane = _io.extract_channel_autofocus_with_idx(fake_img, DAPI_C)
    rna_indep_z, _ = _io.extract_channel_autofocus_with_idx(fake_img, RNA_C)

    # This is exactly what the autofocus branch in rna_only / rna_rna / the
    # rna_protein wrapper now does: read RNA at DAPI's chosen plane.
    rna_locked = _io.extract_channel_at_z(fake_img, RNA_C, z_1indexed=dapi_z)

    # The locked RNA plane is DAPI's plane, not RNA's independent best.
    assert dapi_z != rna_indep_z
    expected = fake_img.bio.get_image_data("ZYX", T=0, C=RNA_C)[dapi_z - 1]
    np.testing.assert_array_equal(rna_locked, expected)
    # And it is NOT the plane RNA's own autofocus would have used.
    rna_indep = fake_img.bio.get_image_data("ZYX", T=0, C=RNA_C)[rna_indep_z - 1]
    assert not np.array_equal(rna_locked, rna_indep)


def test_antibody_locked_to_dapi_plane(fake_img):
    """rna_protein analog: antibody must also lock to DAPI's plane."""
    dapi_z, _ = _io.extract_channel_autofocus_with_idx(fake_img, DAPI_C)
    ab_indep_z, _ = _io.extract_channel_autofocus_with_idx(fake_img, AB_C)

    ab_locked = _io.extract_channel_at_z(fake_img, AB_C, z_1indexed=dapi_z)

    assert dapi_z != ab_indep_z
    expected = fake_img.bio.get_image_data("ZYX", T=0, C=AB_C)[dapi_z - 1]
    np.testing.assert_array_equal(ab_locked, expected)


def test_autofocus_window_bounds_dapi_pick(fake_img):
    """z_start/z_end (the middle-80% / file_overrides window) still bound the
    DAPI autofocus. With a window that excludes the true DAPI sharp plane, the
    pick is the sharpest plane WITHIN the window; RNA still locks to it."""
    # Window 1..4 (1-indexed inclusive) excludes DAPI's sharp plane (z0=5 -> z=6).
    dapi_z, _ = _io.extract_channel_autofocus_with_idx(
        fake_img, DAPI_C, z_start=1, z_end=4
    )
    assert 1 <= dapi_z <= 4
    rna_locked = _io.extract_channel_at_z(fake_img, RNA_C, z_1indexed=dapi_z)
    expected = fake_img.bio.get_image_data("ZYX", T=0, C=RNA_C)[dapi_z - 1]
    np.testing.assert_array_equal(rna_locked, expected)


# ---------------------------------------------------------------------------
# Intensity-weighted focus metric (Brian, 2026-05-28).
#
# On thick stacks the legacy mean-normalized Laplacian variance
# var(laplace(plane/mean)) can be tipped by NOISE to a dim, badly out-of-focus
# plane (dividing by a small mean inflates the normalized lap-var). The fix is
# var(laplace(plane/mean)) * mean — weighting sharpness by brightness pulls the
# pick to the bright AND sharp nuclear plane. These tests build a stack whose
# true in-focus plane (bright + structured) differs from a high-frequency NOISE
# plane (dim) that has higher RAW normalized lap-var, and assert the legacy
# metric picks the noise plane while intensity_weighted=True picks the bright one.
# ---------------------------------------------------------------------------
_IW_YX = (48, 48)
_IW_INFOCUS_Z0 = 1   # bright, structured -> true in-focus plane
_IW_NOISE_Z0 = 3     # dim, high-frequency speckle -> high raw normalized lap-var


def _iw_blurry(seed: int, base: float) -> np.ndarray:
    r = np.random.default_rng(seed)
    yy, xx = np.mgrid[0 : _IW_YX[0], 0 : _IW_YX[1]].astype(np.float32)
    return (base + 0.3 * yy + 0.3 * xx + r.uniform(0.0, 2.0, _IW_YX)).astype(np.float32)


def _iw_infocus(seed: int) -> np.ndarray:
    """Bright (high-mean), structured nuclear-like blobs -> the plane we WANT."""
    from scipy.ndimage import gaussian_filter
    from skimage.draw import disk

    r = np.random.default_rng(seed)
    img = r.uniform(40.0, 60.0, _IW_YX).astype(np.float32)
    for (cy, cx) in [(16, 16), (16, 32), (32, 24)]:
        rr, cc = disk((cy, cx), 7, shape=_IW_YX)
        img[rr, cc] += 1500.0
    return gaussian_filter(img, 0.6)


def _iw_noise(seed: int) -> np.ndarray:
    """DIM, high-frequency speckle: high var(laplace(plane/mean)) but low mean."""
    r = np.random.default_rng(seed)
    return r.uniform(0.5, 6.0, _IW_YX).astype(np.float32)


def _iw_stack() -> np.ndarray:
    planes = [
        _iw_blurry(1, 20.0),
        _iw_infocus(2),       # z=1: in-focus (bright)
        _iw_blurry(3, 25.0),
        _iw_noise(4),         # z=3: noise (dim, high raw lap-var)
        _iw_blurry(5, 18.0),
    ]
    return np.stack(planes, axis=0).astype(np.float32)


def test_legacy_metric_picks_noise_plane():
    """Guard: the legacy (unweighted) metric is the BUG — it picks the dim,
    high-frequency NOISE plane, not the bright in-focus plane."""
    stack = _iw_stack()
    idx, _ = _io._autofocus_plane_with_idx(stack, intensity_weighted=False)
    assert idx == _IW_NOISE_Z0, f"expected legacy to pick noise z={_IW_NOISE_Z0}, got {idx}"


def test_intensity_weighted_picks_bright_in_focus_plane():
    """Fix: intensity_weighted=True picks the bright, structured in-focus plane."""
    stack = _iw_stack()
    idx, _ = _io._autofocus_plane_with_idx(stack, intensity_weighted=True)
    assert idx == _IW_INFOCUS_Z0, (
        f"expected intensity-weighted to pick in-focus z={_IW_INFOCUS_Z0}, got {idx}"
    )
    # And the two metrics genuinely DISAGREE on this stack (non-degenerate).
    legacy, _ = _io._autofocus_plane_with_idx(stack, intensity_weighted=False)
    assert legacy != idx


def test_intensity_weighted_default_is_unweighted_legacy():
    """Default (no kwarg) must be the legacy metric so untouched call sites and
    thin-stack presets are byte-identical."""
    stack = _iw_stack()
    default_idx, _ = _io._autofocus_plane_with_idx(stack)
    legacy_idx, _ = _io._autofocus_plane_with_idx(stack, intensity_weighted=False)
    assert default_idx == legacy_idx == _IW_NOISE_Z0


def test_extract_channel_autofocus_threads_intensity_weighted():
    """extract_channel_autofocus_with_idx forwards intensity_weighted to the
    plane picker (the kwarg the mode/runner call sites pass)."""
    czyx = _iw_stack()[None, :, :, :]  # (C=1, Z, Y, X)
    img = ImageWrapper(
        path="iw_synthetic.tif",
        bio=_FakeBio(czyx),
        scene_idx=0,
        shape=(1, 1, czyx.shape[1], _IW_YX[0], _IW_YX[1]),
        channel_names=["DAPI"],
        voxel_xy_nm=65.0,
        voxel_z_nm=230.0,
        n_channels=1,
        n_z=czyx.shape[1],
    )
    z_legacy, _ = _io.extract_channel_autofocus_with_idx(img, 0)
    z_iw, _ = _io.extract_channel_autofocus_with_idx(img, 0, intensity_weighted=True)
    # 1-indexed absolute z.
    assert z_legacy == _IW_NOISE_Z0 + 1
    assert z_iw == _IW_INFOCUS_Z0 + 1
