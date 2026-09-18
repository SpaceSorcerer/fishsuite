import pytest

from fishsuite.config.schema import NucleiCfg


def resolve(cfg, pixel_size_um, downsample_factor):
    from fishsuite.core.nuclear_size import resolve_nuclear_size_px

    return resolve_nuclear_size_px(cfg, pixel_size_um, downsample_factor)


def test_um_only_config_does_not_conflict_with_pixel_defaults():
    cfg = NucleiCfg(expected_diameter_um=11.0, min_area_um2=60.0)
    sizes = resolve(cfg, 0.13, 2.0)
    assert sizes.diameter_px == pytest.approx(42.308, abs=0.001)
    assert sizes.min_area_px == pytest.approx(3550.3, abs=0.1)
    assert sizes.sources["diameter_px"] == "um"
    assert sizes.sources["max_area_px"] == "default"


@pytest.mark.parametrize("pixel_size,downsample,want", [
    (0.13, 2.0, 42.308),
    (0.065, 2.0, 84.615),
    (0.13, 1.0, 84.615),
])
def test_diameter_is_scaled_once_for_model_grid(pixel_size, downsample, want):
    sizes = resolve(NucleiCfg(expected_diameter_um=11.0), pixel_size, downsample)
    assert sizes.diameter_px == pytest.approx(want, abs=0.001)
    assert sizes.grids["native_diameter_px"] == "native"


@pytest.mark.parametrize("pixel_size,want", [(0.13, 3550.3), (0.065, 14201.2)])
def test_area_filters_use_restored_native_grid(pixel_size, want):
    cfg = NucleiCfg(min_area_um2=60.0, max_area_um2=60.0,
                    reject_ghost_min_area_um2=60.0)
    sizes = resolve(cfg, pixel_size, 2.0)
    for name in ("min_area_px", "max_area_px", "reject_ghost_min_area_px"):
        assert getattr(sizes, name) == pytest.approx(want, abs=0.1)
        assert sizes.grids[name] == "native"


def test_border_margin_uses_native_linear_scale():
    sizes = resolve(NucleiCfg(border_margin_um=0.65), 0.13, 2.0)
    assert sizes.border_margin_px == pytest.approx(5.0)
    assert sizes.grids["border_margin_px"] == "native"


@pytest.mark.parametrize("um_key,px_key", [
    ("expected_diameter_um", "cellpose_diameter_px"),
    ("min_area_um2", "min_area_px"),
    ("max_area_um2", "max_area_px"),
    ("reject_ghost_min_area_um2", "reject_ghost_min_area_px"),
    ("border_margin_um", "border_margin_px"),
])
def test_explicit_unit_twins_raise_clear_error(um_key, px_key):
    with pytest.raises(ValueError, match=um_key):
        resolve(NucleiCfg(**{um_key: 10.0, px_key: 10}), 0.13, 2.0)


@pytest.mark.parametrize("pixel_size", [None, 0, -0.13, float("nan"), float("inf")])
def test_um_size_rejects_unreadable_pixel_size(pixel_size):
    with pytest.raises(ValueError, match="pixel size"):
        resolve(NucleiCfg(expected_diameter_um=11.0), pixel_size, 2.0)


def test_legacy_pixels_remain_native_and_diameter_scales_once():
    cfg = NucleiCfg(cellpose_diameter_px=100, min_area_px=123,
                    max_area_px=45678, reject_ghost_min_area_px=6000,
                    border_margin_px=5)
    sizes = resolve(cfg, None, 2.0)
    assert sizes.diameter_px == 50.0
    assert sizes.native_diameter_px == 100.0
    assert sizes.min_area_px == 123
    assert sizes.max_area_px == 45678
    assert sizes.reject_ghost_min_area_px == 6000
    assert sizes.border_margin_px == 5
    assert set(sizes.sources.values()) == {"px"}


def test_default_auto_diameter_and_defaults_remain_unchanged():
    sizes = resolve(NucleiCfg(), None, 1.0)
    assert sizes.diameter_px == 0
    assert sizes.min_area_px == 10000
    assert sizes.max_area_px == 1e12
    assert sizes.reject_ghost_min_area_px == 6000
    assert sizes.border_margin_px == 5
    assert set(sizes.sources.values()) == {"default"}


def test_null_um_twin_is_inactive():
    sizes = resolve(NucleiCfg(expected_diameter_um=None, cellpose_diameter_px=100),
                    None, 2.0)
    assert sizes.diameter_px == 50
    assert sizes.sources["diameter_px"] == "px"


def test_resolved_metadata_reports_input_output_source_and_grid():
    sizes = resolve(NucleiCfg(expected_diameter_um=11), 0.13, 2)
    report = sizes.as_dict()
    assert report["pixel_size_um"] == 0.13
    assert report["downsample_factor"] == 2
    diameter = report["sizes"]["diameter_px"]
    assert diameter["value"] == pytest.approx(42.308, abs=0.001)
    assert diameter["input_value"] == 11
    assert diameter["input_unit"] == "um"
    assert diameter["grid"] == "downsampled"
    assert diameter["source"] == "um"


def test_dump_yaml_preserves_explicit_nuclear_units_and_default_sources(tmp_path):
    from fishsuite.config.schema import FishsuiteConfig

    cfg = FishsuiteConfig(nuclei=NucleiCfg(expected_diameter_um=11.0))
    path = tmp_path / "config.yaml"
    cfg.dump_yaml(path)
    reloaded = FishsuiteConfig.from_yaml(path)
    sizes = resolve(reloaded.nuclei, 0.13, 2.0)
    assert sizes.diameter_px == pytest.approx(42.308, abs=0.001)
    assert sizes.sources["diameter_px"] == "um"
    assert sizes.sources["min_area_px"] == "default"


@pytest.mark.parametrize("kwargs,pixel_size,factor,want,model_shape", [
    ({"expected_diameter_um": 11.0}, 0.13, 2.0, 42.307692, (16, 16)),
    ({"expected_diameter_um": 11.0}, 0.065, 2.0, 84.615385, (16, 16)),
    ({"expected_diameter_um": 11.0}, 0.13, 1.0, 84.615385, (32, 32)),
    ({"cellpose_diameter_px": 100.0}, None, 2.0, 50.0, (16, 16)),
])
def test_segmentation_hands_resolved_diameter_to_model_once(
    monkeypatch, kwargs, pixel_size, factor, want, model_shape,
):
    import numpy as np
    from fishsuite.core import segmentation
    from fishsuite.core._vendor.segmentation import segment_image

    observed = {}

    def fake_backend(backend, image, **model_kwargs):
        observed.update(model_kwargs)
        observed["shape"] = image.shape
        return np.ones(image.shape, dtype=np.int32)

    monkeypatch.setattr(segment_image, "run_backend", fake_backend)
    sizes = resolve(NucleiCfg(**kwargs), pixel_size, factor)
    labels = segmentation.segment_nuclei(
        np.ones((32, 32), dtype=np.float32), backend="cellpose",
        params={"diameter": sizes.native_diameter_px,
                "cellpose_downsample_factor": factor, "min_area": 1},
    )
    assert observed["diameter"] == pytest.approx(want, abs=0.000001)
    assert observed["shape"] == model_shape
    assert labels.shape == (32, 32)
