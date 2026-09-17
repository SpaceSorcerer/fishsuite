from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fishsuite.config.schema import FishsuiteConfig
from fishsuite.core.modes import rna_only, rna_rna


class SegmentationReached(Exception):
    pass


def _cfg():
    cfg = FishsuiteConfig()
    cfg.channels.dapi = 0
    cfg.channels.rna = 1
    cfg.channels.rna2 = 2
    cfg.channels.one_indexed = False
    cfg.z_stack.mode = "maxproj"
    cfg.nuclei = cfg.nuclei.model_copy(update={
        "expected_diameter_um": 11.0,
        "min_area_um2": 60.0,
        "max_area_um2": 120.0,
        "border_margin_um": 1.3,
        "cellpose_downsample_factor": 2,
    })
    return cfg


def _mock_image(monkeypatch, module, voxel_nm):
    img = SimpleNamespace(n_channels=3, n_z=1, voxel_xy_nm=voxel_nm,
                          voxel_z_nm=230.0)
    monkeypatch.setattr(module._io, "read_image", lambda path: img)
    monkeypatch.setattr(module._io, "extract_channel",
                        lambda *args, **kwargs: np.ones((16, 16), dtype=np.float32))


def _run(module, prescan, cfg):
    path = Path("synthetic_scale.tif")
    if prescan:
        return module.collect_nuclear_rna_pixels(path, cfg=cfg)
    return module.run_one(path, condition="synthetic", sec_only=False, cfg=cfg)


@pytest.mark.parametrize("module", [rna_only, rna_rna])
@pytest.mark.parametrize("prescan", [False, True])
@pytest.mark.parametrize("voxel_nm,diameter,area,max_area", [
    (130.0, 84.61538461538461, 3550.295857988165, 7100.59171597633),
    (65.0, 169.23076923076923, 14201.18343195266, 28402.36686390532),
])
def test_rna_routes_pass_native_sizes(monkeypatch, module, prescan,
                                      voxel_nm, diameter, area, max_area):
    cfg = _cfg()
    _mock_image(monkeypatch, module, voxel_nm)

    def segment(image, *, params, **kwargs):
        assert params["diameter"] == pytest.approx(diameter)
        assert params["min_area"] == pytest.approx(area)
        assert params["max_area"] == pytest.approx(max_area)
        assert params["cellpose_downsample_factor"] == 2
        raise SegmentationReached

    monkeypatch.setattr(module._seg, "segment_nuclei", segment)
    with pytest.raises(SegmentationReached):
        _run(module, prescan, cfg)


@pytest.mark.parametrize("module", [rna_only, rna_rna])
@pytest.mark.parametrize("prescan", [False, True])
def test_rna_routes_reject_missing_scale(monkeypatch, module, prescan):
    _mock_image(monkeypatch, module, float("nan"))
    monkeypatch.setattr(module._seg, "segment_nuclei",
                        lambda *a, **k: pytest.fail("segmentation must not run"))
    with pytest.raises(ValueError, match="synthetic_scale.tif"):
        _run(module, prescan, _cfg())


@pytest.mark.parametrize("module", [rna_only, rna_rna])
@pytest.mark.parametrize("prescan", [False, True])
def test_rna_routes_honor_explicit_scale(monkeypatch, module, prescan):
    cfg = _cfg()
    cfg.foci.bigfish_voxel_size_nm = 130.0
    _mock_image(monkeypatch, module, 65.0)

    def segment(image, *, params, **kwargs):
        assert params["diameter"] == pytest.approx(84.61538461538461)
        raise SegmentationReached

    monkeypatch.setattr(module._seg, "segment_nuclei", segment)
    with pytest.raises(SegmentationReached):
        _run(module, prescan, cfg)


@pytest.mark.parametrize("module", [rna_only, rna_rna])
def test_rna_prescan_resolves_native_border(monkeypatch, module):
    cfg = _cfg()
    cfg.nuclei.exclude_border = True
    _mock_image(monkeypatch, module, 130.0)
    monkeypatch.setattr(module._seg, "segment_nuclei",
                        lambda image, **kwargs: np.zeros_like(image, dtype=np.int32))
    margins = []

    def border(labels, *, margin_px):
        margins.append(margin_px)
        return labels

    monkeypatch.setattr(module._seg, "exclude_border_labels", border)
    _run(module, True, cfg)
    assert margins == [10]


@pytest.mark.parametrize("module", [rna_only, rna_rna])
@pytest.mark.parametrize("prescan", [False, True])
def test_rna_routes_preserve_legacy_pixels(monkeypatch, module, prescan):
    cfg = _cfg()
    cfg.nuclei = cfg.nuclei.model_copy(update={
        "expected_diameter_um": None, "min_area_um2": None,
        "max_area_um2": None, "border_margin_um": None,
        "cellpose_diameter_px": 100.0, "min_area_px": 900,
        "max_area_px": 30000,
    })
    _mock_image(monkeypatch, module, float("nan"))

    def segment(image, *, params, **kwargs):
        assert params["diameter"] == 100.0
        assert params["min_area"] == 900
        assert params["max_area"] == 30000
        raise SegmentationReached

    monkeypatch.setattr(module._seg, "segment_nuclei", segment)
    with pytest.raises(SegmentationReached):
        _run(module, prescan, cfg)


@pytest.mark.parametrize("module", [rna_only, rna_rna])
def test_rna_thresholds_keep_flat_resolved_minimum(monkeypatch, module):
    cfg = _cfg()
    cfg.foci.enabled = False
    cfg.cytoplasm.enabled = False
    cfg.nuclei.exclude_border = False
    _mock_image(monkeypatch, module, 130.0)
    monkeypatch.setattr(module._seg, "segment_nuclei",
                        lambda image, **kwargs: np.zeros_like(image, dtype=np.int32))
    result = _run(module, False, cfg)
    assert "resolved_nuclear_size" not in result.thresholds
    assert result.thresholds["nuc_min_area_px"] == pytest.approx(3550.295857988165)


@pytest.mark.parametrize("module", [rna_only, rna_rna])
def test_rna_ghost_filter_preserves_fractional_area(monkeypatch, module):
    cfg = _cfg()
    cfg.nuclei.reject_ghost_nuclei = True
    cfg.nuclei.reject_ghost_min_area_um2 = 60.0
    cfg.nuclei.exclude_border = False
    cfg.foci.enabled = False
    cfg.cytoplasm.enabled = False
    _mock_image(monkeypatch, module, 130.0)
    labels = np.zeros((16, 16), dtype=np.int32)
    labels[4:8, 4:8] = 1
    monkeypatch.setattr(module._seg, "segment_nuclei", lambda *a, **k: labels)
    areas = []

    def identify(frame, *, min_area_px, **kwargs):
        areas.append(min_area_px)
        return []

    monkeypatch.setattr(module._seg, "identify_ghost_nuclei", identify)
    _run(module, False, cfg)
    assert areas == pytest.approx([3550.295857988165])
