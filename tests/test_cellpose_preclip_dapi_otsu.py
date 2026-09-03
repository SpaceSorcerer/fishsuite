"""Cellpose DAPI Otsu pre-clip (default off) + NucleiCfg unknown-key rejection.

Ported verbatim 2026-09-03 from the stage-09 proposal worktree
``F:\\Image Analysis Work\\RNASEH2B_BIN1introns_2026_08_25
\\09_CODE_PROPOSAL_FISHSUITE_DAPI_FLOOR_2026-09-01\\src\\fishsuite``, which is the
engine that produced run 11 (`11_FULL_CPSAMV2_OTSU_2026-09-01`). The feature had
never been merged into this repository, and ``NucleiCfg`` silently dropped the
key, so run-11's preset ran without the segmentation step it asked for and still
exited 0. These tests cover both halves of that failure.
"""
from __future__ import annotations

import numpy as np
import pytest
from skimage.filters import threshold_otsu

from fishsuite.config.schema import FishsuiteConfig, NucleiCfg
from fishsuite.core import segmentation as _seg


# ===========================================================================
# Schema: the key round-trips, defaults off, and unknown keys are rejected.
# ===========================================================================
def test_preclip_defaults_off():
    assert NucleiCfg().cellpose_preclip_dapi_otsu is False


@pytest.mark.parametrize("value", [True, False])
def test_preclip_round_trips_through_yaml_dict(value):
    cfg = FishsuiteConfig.model_validate(
        {"nuclei": {"backend": "cellpose", "cellpose_preclip_dapi_otsu": value}}
    )
    assert cfg.nuclei.cellpose_preclip_dapi_otsu is value
    # and survives a dump/reload cycle
    again = FishsuiteConfig.model_validate(cfg.model_dump())
    assert again.nuclei.cellpose_preclip_dapi_otsu is value


def test_unknown_nuclei_key_is_rejected_and_named():
    with pytest.raises(Exception) as exc:
        FishsuiteConfig.model_validate({"nuclei": {"cellpose_preclip_dapi_ostu": True}})
    # the typo must appear in the message; a rejection that does not name the
    # offending key is barely better than the silent drop it replaces.
    assert "cellpose_preclip_dapi_ostu" in str(exc.value)


def test_known_nuclei_keys_still_accepted():
    cfg = FishsuiteConfig.model_validate(
        {"nuclei": {"backend": "cellpose", "cellpose_model_type": "cpsam_v2",
                    "cellpose_diameter_px": 200.0, "min_area_px": 16000,
                    "exclude_border": True, "border_margin_px": 20}}
    )
    assert cfg.nuclei.cellpose_model_type == "cpsam_v2"


# ===========================================================================
# Segmentation: the pre-clip changes the MODEL INPUT, and only for cellpose.
# ===========================================================================
def _dapi_with_haze(seed=0, size=128):
    """Bright nuclei on a dim, non-zero haze — the case the pre-clip targets."""
    rng = np.random.default_rng(seed)
    img = rng.uniform(80.0, 140.0, (size, size)).astype(np.float32)  # haze
    from skimage.draw import disk
    for (cy, cx) in [(40, 40), (40, 90), (90, 65)]:
        rr, cc = disk((cy, cx), 16, shape=img.shape)
        img[rr, cc] += 3000.0
    return img


def _capture_seg_input(monkeypatch):
    """Record the array handed to the vendored backend."""
    seen = {}

    def fake_run_backend(backend, img, **kwargs):
        seen["backend"] = backend
        seen["img"] = np.array(img, copy=True)
        return np.zeros(img.shape, dtype=np.int32)

    import fishsuite.core._vendor.segmentation.segment_image as _si
    monkeypatch.setattr(_si, "run_backend", fake_run_backend)
    return seen


def test_preclip_off_passes_raw_dapi(monkeypatch):
    img = _dapi_with_haze()
    seen = _capture_seg_input(monkeypatch)
    _seg.segment_nuclei(img, backend="cellpose",
                        params={"cellpose_preclip_dapi_otsu": False, "min_area": 1})
    assert np.array_equal(seen["img"], img)


def test_preclip_on_zeroes_below_otsu(monkeypatch):
    img = _dapi_with_haze()
    thr = float(threshold_otsu(img))
    seen = _capture_seg_input(monkeypatch)
    _seg.segment_nuclei(img, backend="cellpose",
                        params={"cellpose_preclip_dapi_otsu": True, "min_area": 1})
    got = seen["img"]
    assert not np.array_equal(got, img)
    # every below-threshold pixel is zeroed; every other pixel is untouched.
    below = img < thr
    assert (got[below] == 0).all()
    assert np.array_equal(got[~below], img[~below])
    # the haze was non-zero to begin with, so this is a real change.
    assert below.any() and (img[below] > 0).all()


def test_preclip_ignored_for_non_cellpose_backend(monkeypatch):
    img = _dapi_with_haze()
    seen = _capture_seg_input(monkeypatch)
    _seg.segment_nuclei(img, backend="otsu",
                        params={"cellpose_preclip_dapi_otsu": True, "min_area": 1})
    assert np.array_equal(seen["img"], img)


def test_preclip_floor_is_reported_in_stats(monkeypatch):
    img = _dapi_with_haze()
    _capture_seg_input(monkeypatch)
    stats = {}
    _seg.segment_nuclei(img, backend="cellpose",
                        params={"cellpose_preclip_dapi_otsu": True, "min_area": 1},
                        stats=stats)
    assert stats["cellpose_dapi_otsu_floor"] == pytest.approx(float(threshold_otsu(img)))

    stats_off = {}
    _seg.segment_nuclei(img, backend="cellpose",
                        params={"cellpose_preclip_dapi_otsu": False, "min_area": 1},
                        stats=stats_off)
    assert "cellpose_dapi_otsu_floor" not in stats_off
