from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from fishsuite.config.schema import FishsuiteConfig
from fishsuite.core.modes import rna_only, rna_rna
from fishsuite.core.nuclear_size import resolve_nuclear_size_px


def config():
    return FishsuiteConfig.model_validate({
        "channels": {"dapi": 3, "rna": 1, "rna2": 2, "one_indexed": True},
        "z_stack": {"mode": "autofocus", "start_slice": 1, "end_slice": 8,
                    "file_overrides": {"field.vsi": {"start_slice": 2, "end_slice": 6}}},
        "nuclei": {"expected_diameter_um": 11.0, "min_area_um2": 67.6,
                   "cellpose_downsample_factor": 2, "cellpose_device": "cpu"},
    })


@pytest.mark.parametrize("anchor", ["dapi", "joint", "rna", "auto"])
def test_prepare_preserves_channels_overrides_and_locked_plane(monkeypatch, anchor):
    cfg = config()
    cfg.z_stack.autofocus_channel = anchor
    img = SimpleNamespace(n_z=8, n_channels=3)
    planes = [np.full((4, 5), n, dtype=np.uint16) for n in range(3)]
    extract = Mock(side_effect=lambda image, ch, **kw: planes[ch])
    monkeypatch.setattr(rna_rna._io, "extract_channel_at_z", extract)
    dapi = Mock(return_value=(4, planes[2]))
    joint = Mock(return_value=(4, {"per_channel_focus_score": {"dapi": 0.75}}))
    other = Mock(return_value=(4, "rna", {}))
    monkeypatch.setattr(rna_rna._io, "extract_channel_autofocus_with_idx", dapi)
    monkeypatch.setattr(rna_rna._io, "resolve_joint_autofocus_plane", joint)
    monkeypatch.setattr(rna_rna._io, "resolve_autofocus_plane", other)
    out = rna_rna.prepare_segmentation_planes(img, Path("field.vsi"), cfg)
    assert (out["dapi_idx"], out["rna_idx"], out["rna2_idx"]) == (2, 0, 1)
    assert (out["z_start"], out["z_end"], out["dapi_autofocus_z"]) == (2, 6, 4)
    assert out["dapi_2d"] is planes[2]
    assert out["rna_2d"] is planes[0]
    assert out["rna2_2d"] is planes[1]
    called = dapi if anchor == "dapi" else joint if anchor == "joint" else other
    assert called.call_count == 1
    assert called.call_args.kwargs["z_start"] == 2
    assert called.call_args.kwargs["z_end"] == 6
    assert all(c.kwargs == {"z_1indexed": 4} for c in extract.call_args_list)
    assert extract.call_count == (2 if anchor == "dapi" else 3)
    if anchor == "joint":
        assert joint.call_args.kwargs["partner_idx"] == 1
        assert out["_z_af_extra"]["joint_focus_dapi"] == 0.75


def test_prepare_focus_window_reuses_shared_io(monkeypatch):
    cfg = config()
    cfg.z_stack.mode = "autofocus_maxproj"
    img = SimpleNamespace(n_z=8, n_channels=3)
    dapi = np.zeros((4, 5), dtype=np.uint16)
    focus = Mock(return_value=((3, 5), {"peak_z": 3}, dapi))
    extract = Mock(return_value=dapi)
    monkeypatch.setattr(rna_rna._io, "extract_dapi_focus_window", focus)
    monkeypatch.setattr(rna_rna._io, "extract_channel_in_z_range", extract)
    out = rna_rna.prepare_segmentation_planes(img, Path("field.vsi"), cfg)
    assert out["dapi_2d"] is dapi
    assert focus.call_args.args == (img, 2)
    assert [call.args[1] for call in extract.call_args_list] == [0, 1]
    assert all(call.kwargs == {"z_start_1indexed": 3, "z_end_1indexed": 5,
                               "project": "maxproj"} for call in extract.call_args_list)


@pytest.mark.parametrize("mode", [rna_rna, rna_only])
def test_segmentation_params_keep_native_grid_and_device(mode):
    cfg = config()
    cfg.nuclei.cellpose_device = "directml"
    size = resolve_nuclear_size_px(cfg.nuclei, 0.13, 2)
    out = mode.segmentation_params(cfg, size)
    assert out["diameter"] == pytest.approx(84.61538461538461)
    assert out["min_area"] == pytest.approx(4000.0)
    assert out["cellpose_downsample_factor"] == 2
    assert out["cellpose_device"] == "directml"


@pytest.mark.parametrize("mode", [rna_rna, rna_only])
def test_run_one_uses_both_shared_helpers_before_segmenter(monkeypatch, mode):
    cfg = config()
    cfg.foci.bigfish_voxel_size_nm = 130.0
    img = SimpleNamespace(n_z=8, n_channels=3, voxel_xy_nm=130.0, voxel_z_nm=210.0)
    dapi = np.ones((4, 5), dtype=np.uint16)
    prepare = Mock(return_value={
        "dapi_idx": 2, "rna_idx": 0, "rna2_idx": 1,
        "z_mode": "autofocus", "z_start": 2, "z_end": 6,
        "dapi_autofocus_z": 4, "_z_af_extra": {},
        "dapi_2d": dapi, "rna_2d": dapi, "rna2_2d": dapi,
    })
    params = Mock(return_value={"sentinel": True})
    monkeypatch.setattr(mode._io, "read_image", lambda path: img)
    monkeypatch.setattr(mode, "prepare_segmentation_planes", prepare)
    monkeypatch.setattr(mode, "segmentation_params", params)

    class StopAtSegmentation(Exception):
        pass

    segment = Mock(side_effect=StopAtSegmentation)
    monkeypatch.setattr(mode._seg, "segment_nuclei", segment)
    with pytest.raises(StopAtSegmentation):
        mode.run_one(Path("field.vsi"), condition="control", sec_only=False, cfg=cfg)
    prepare.assert_called_once_with(img, Path("field.vsi"), cfg)
    assert params.call_count == 1
    assert segment.call_args.args[0] is dapi
    assert segment.call_args.kwargs["params"] == {"sentinel": True}


@pytest.mark.parametrize("z_mode", ["autofocus", "maxproj", "autofocus_maxproj"])
def test_rna_only_preserves_dapi_lock_and_rna_window(monkeypatch, z_mode):
    cfg = config()
    cfg.z_stack.mode = z_mode
    cfg.z_stack.file_overrides["field.vsi"].update(rna_start_slice=3, rna_end_slice=5)
    img = SimpleNamespace(n_z=8, n_channels=3)
    dapi, rna = np.zeros((4, 5)), np.ones((4, 5))
    focus = Mock(return_value=(4, dapi))
    window = Mock(return_value=((2, 6), {"peak_z": 3}, dapi))
    single = Mock(return_value=rna)
    extract = Mock(side_effect=lambda image, channel, **kw: dapi if channel == 2 else rna)
    monkeypatch.setattr(rna_only._io, "extract_channel_autofocus_with_idx", focus)
    monkeypatch.setattr(rna_only._io, "extract_dapi_focus_window", window)
    monkeypatch.setattr(rna_only._io, "extract_channel_at_z", single)
    monkeypatch.setattr(rna_only._io, "extract_channel", extract)
    out = rna_only.prepare_segmentation_planes(img, Path("field.vsi"), cfg)
    assert out["dapi_idx"] == 2 and out["rna_idx"] == 0
    assert out["dapi_2d"] is dapi and out["rna_2d"] is rna
    assert out["z_start"] == 2 and out["z_end"] == 6
    if z_mode == "autofocus":
        assert out["dapi_autofocus_z"] == 4
        single.assert_called_once_with(img, 0, z_1indexed=4)
        extract.assert_not_called()
    else:
        assert extract.call_args.kwargs == {"z_mode": "maxproj", "z_start": 3, "z_end": 5}
        single.assert_not_called()


@pytest.mark.parametrize("mode", [rna_only, rna_rna])
def test_run_and_sweep_preserve_real_area_and_border_filters(tmp_path, monkeypatch, mode):
    from test_nucleus_sampling import _reader_probe_failure
    reason = _reader_probe_failure()
    if reason:
        pytest.skip(reason)
    import pandas as pd
    import tifffile
    from click.testing import CliRunner
    from fishsuite.cli import cli
    from fishsuite.core._vendor.segmentation import segment_image

    root = tmp_path / "inputs"
    root.mkdir()
    image_path = root / "field.ome.tif"
    yy, xx = np.indices((96, 96))
    image = np.stack([100 + yy + xx, 200 + 2 * yy + xx, 300 + yy + 2 * xx]).astype(np.uint16)
    tifffile.imwrite(image_path, image, ome=True, photometric="minisblack", metadata={
        "axes": "CYX", "PhysicalSizeX": 0.13, "PhysicalSizeXUnit": "µm",
        "PhysicalSizeY": 0.13, "PhysicalSizeYUnit": "µm",
    })
    cfg = FishsuiteConfig.model_validate({
        "channels": {"analysis_mode": mode.__name__.rsplit(".", 1)[-1],
                     "dapi": 0, "rna": 1, "rna2": 2, "one_indexed": False},
        "z_stack": {"mode": "single", "start_slice": 1, "end_slice": 1},
        "conditions": {"subfolder_conditions": {"": "control"}},
        "nuclei": {"backend": "cellpose", "cellpose_device": "cpu",
                   "expected_diameter_um": 11.0, "cellpose_downsample_factor": 1,
                   "min_area_px": 400, "max_area_px": 10000,
                   "exclude_border": True, "border_margin_px": 1,
                   "label_smoothing_radius_px": 0, "reject_ghost_nuclei": False},
        "foci": {"enabled": False, "bigfish_voxel_size_nm": 130.0},
        "cytoplasm": {"enabled": False}, "nucleolus": {"enabled": False},
        "sampling": {"enabled": True, "n_per_unit": 1, "min_eligible": 1},
    })
    planted = np.zeros((96, 96), dtype=np.uint16)
    planted[30:55, 30:55] = 1  # valid: 625 native pixels
    planted[:25, 65:90] = 2  # border: 625 native pixels
    planted[65:80, 15:35] = 3  # small: 300 pixels, above vendor floor 250
    model_calls = []

    def evaluate(plane, **kwargs):
        assert plane.shape == (96, 96)
        assert kwargs["diameter"] == pytest.approx(84.61538461538461)
        model_calls.append(kwargs)
        return planted.copy(), None

    # The sole replacement: the expensive model. TIFF I/O, segmentation wrappers,
    # vendor area filtering, native area filtering, border removal and CLI are real.
    monkeypatch.setattr(segment_image, "_get_cellpose_model",
                        lambda model_type, device: SimpleNamespace(eval=evaluate))
    result = mode.run_one(image_path, condition="control", sec_only=False, cfg=cfg)
    expected = np.zeros((96, 96), dtype=np.uint16)
    expected[30:55, 30:55] = 1
    np.testing.assert_array_equal(result.qc["labels"], expected)
    assert result.qc["labels"].dtype == np.uint16
    assert len(result.nuclei) == 1
    assert result.per_image["n_nuclei_area_excluded"] == 1
    assert result.per_image["n_nuclei_border_excluded"] == 1
    assert result.per_image["n_nuclei_sampled"] == 1

    config_path = tmp_path / "config.yaml"
    cfg.dump_yaml(config_path)
    output = tmp_path / "sweep"
    invoked = CliRunner().invoke(cli, ["seg-sweep", "-c", str(config_path),
        "-i", str(root), "--diameters-um", "11", "--out", str(output)])
    assert invoked.exit_code == 0, invoked.output + repr(invoked.exception)
    summary = pd.read_csv(output / "seg_sweep_summary.csv")
    assert len(summary) == 1
    assert summary.n_kept.tolist() == [1]
    assert summary.n_rejected_small.tolist() == [1]
    assert summary.n_rejected_border.tolist() == [1]
    assert summary.median_area_um2.tolist() == pytest.approx([10.5625])
    assert len(model_calls) == 2
