import logging
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fishsuite.config.schema import FishsuiteConfig
from fishsuite.core.modes import if_intensity as mode


@pytest.mark.parametrize("metadata", [None, 0, -1, float("nan")])
def test_missing_pixel_metadata_requires_explicit_override(metadata):
    with pytest.raises(ValueError, match="synthetic.tif.*if_intensity.pixel_size_um"):
        mode._resolve_pixel_size_um(FishsuiteConfig(), metadata, "synthetic.tif")


def test_explicit_pixel_size_override_warns(caplog):
    cfg = FishsuiteConfig(if_intensity={"pixel_size_um": 0.13})
    with caplog.at_level(logging.WARNING):
        assert mode._resolve_pixel_size_um(cfg, 0, "synthetic.tif") == 0.13
    assert "if_intensity.pixel_size_um" in caplog.text


def test_pixel_metadata_used_without_override():
    assert mode._resolve_pixel_size_um(FishsuiteConfig(), 0.065, "synthetic.tif") == 0.065


def test_segmentation_uses_resolved_physical_sizes(monkeypatch):
    cfg = FishsuiteConfig(nuclei={"expected_diameter_um": 11, "min_area_um2": 60,
                                  "cellpose_downsample_factor": 2})
    captured = {}

    def segment(image, *, backend, params):
        captured.update(params)
        return np.zeros(image.shape, dtype=np.int32)

    monkeypatch.setattr(mode._seg, "segment_nuclei", segment)
    mode._segment_dapi(np.zeros((8, 8), dtype=np.float32), cfg, pixel_size_um=0.13)
    # segment_nuclei receives native pixels and divides by d at model input.
    assert captured["diameter"] == pytest.approx(84.6153846)
    assert captured["min_area"] == pytest.approx(3550.295858)


def _mock_plate(monkeypatch, metadata):
    monkeypatch.setattr(mode, "_install_asarray_shim", lambda: None)
    monkeypatch.setattr(mode, "_build_plate_map", lambda *a, **k: {1: {}})
    monkeypatch.setattr(mode, "_build_comparison_groups", lambda *a: ({}, {"WT": [], "KO": []}))
    monkeypatch.setattr(mode, "_discover_wells", lambda *a, **k: {1: [Path("synthetic.tif")]})

    def load(*args, metadata_only=False):
        assert metadata_only
        return metadata

    monkeypatch.setattr(mode, "_load_fov", load)


def test_if_dryrun_prints_resolution_without_writing_files(monkeypatch, tmp_path, capsys):
    _mock_plate(monkeypatch, 0.13)
    cfg = FishsuiteConfig(nuclei={"expected_diameter_um": 11, "cellpose_downsample_factor": 2})
    result = mode.run_if_batch(cfg, "unused.yaml", tmp_path, tmp_path, {}, dry_run=True)
    assert result["dry_run"]
    assert "RESOLVED NUCLEAR SIZE" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []


def test_if_real_resolution_sidecar_preserves_run_config(tmp_path):
    original = '{"existing": "untouched"}\n'
    (tmp_path / "run_config.json").write_text(original)
    records = []
    mode._record_nuclear_size(FishsuiteConfig(), 0.13, "synthetic.tif", tmp_path, records)
    assert (tmp_path / "run_config.json").read_text() == original
    assert json.loads((tmp_path / "resolved_nuclear_size.json").read_text()) == records


def test_if_real_resolution_does_not_create_run_config(tmp_path):
    mode._record_nuclear_size(FishsuiteConfig(), 0.13, "synthetic.tif", tmp_path, [])
    assert (tmp_path / "resolved_nuclear_size.json").exists()
    assert not (tmp_path / "run_config.json").exists()


def test_if_dryrun_rejects_missing_metadata(monkeypatch, tmp_path):
    _mock_plate(monkeypatch, 0)
    with pytest.raises(ValueError, match="synthetic.tif.*if_intensity.pixel_size_um"):
        mode.run_if_batch(FishsuiteConfig(), "unused.yaml", tmp_path, tmp_path, {}, dry_run=True)


def test_if_dryrun_accepts_explicit_pixel_override(monkeypatch, tmp_path):
    _mock_plate(monkeypatch, 0)
    cfg = FishsuiteConfig(if_intensity={"pixel_size_um": 0.13})
    assert mode.run_if_batch(cfg, "unused.yaml", tmp_path, tmp_path, {}, dry_run=True)["dry_run"]


def test_if_provenance_reports_physical_diameter(tmp_path):
    cfg = FishsuiteConfig(nuclei={"expected_diameter_um": 11})
    mode._append_provenance(tmp_path, cfg, 64, 0.13)
    note = (tmp_path / "command.log").read_text()
    assert "expected_diameter_um=11.0" in note
    assert "resolved_nuclear_size.json" in note
    assert "run_config.json" not in note
    assert "diameter=0.0 " not in note


def test_if_provenance_keeps_legacy_diameter(tmp_path):
    cfg = FishsuiteConfig(nuclei={"cellpose_diameter_px": 100})
    mode._append_provenance(tmp_path, cfg, 64, 0.13)
    note = (tmp_path / "command.log").read_text()
    assert "diameter=100.0 device=cpu " in note


@pytest.mark.parametrize("override", [0.13, None])
def test_if_csv_preserves_metadata_and_records_override(tmp_path, override):
    per_fov = pd.DataFrame([dict(file="synthetic.tif", genotype="WT", arm="primary",
                                secondary="s", nucleus_count=1)])
    empty = pd.DataFrame()
    mode._write_csvs(tmp_path, "", per_fov, empty, empty, empty, empty, empty,
                     None, {}, {}, {"synthetic.tif": 0.065}, 0.13,
                     pixel_size_um_config_override=override)
    summary = pd.read_csv(tmp_path / "per_image_summary.csv")
    assert summary.loc[0, "voxel_xy_nm"] == 65.0
    value = summary.loc[0, "pixel_size_um_config_override"]
    assert pd.isna(value) if override is None else value == 0.13


def test_if_batch_keeps_metadata_separate_from_override(monkeypatch, tmp_path):
    cfg = FishsuiteConfig(if_intensity={"pixel_size_um": 0.13}, output={"save_masks": False})
    dapi = cfg.if_intensity.dapi_channel_key
    plate = {1: dict(qki_channel="qki", genotype="WT", arm="primary", secondary="s")}
    monkeypatch.setattr(mode, "_install_asarray_shim", lambda: None)
    monkeypatch.setattr(mode, "_build_plate_map", lambda *a, **k: plate)
    monkeypatch.setattr(mode, "_build_comparison_groups", lambda *a: ({}, {"WT": [], "KO": []}))
    monkeypatch.setattr(mode, "_discover_wells", lambda *a, **k: {1: [Path("synthetic.tif")]})
    image = np.zeros((8, 8))
    monkeypatch.setattr(mode, "_load_fov", lambda *a: ({dapi: image, "qki": image}, {}, 0, 0.065))
    scales = []

    def segment(*a, pixel_size_um):
        scales.append(pixel_size_um)
        return np.zeros((8, 8), dtype=np.int32)

    monkeypatch.setattr(mode, "_segment_dapi", segment)
    fov = dict(nucleus_count=0, nuc_mean_qki=0, ratio_qki_over_dapi=0, exp_qki_s=0)
    monkeypatch.setattr(mode, "_quantify_fov", lambda *a: (fov, pd.DataFrame()))
    monkeypatch.setattr(mode, "_check_exposures", lambda *a: (pd.DataFrame(), None))
    monkeypatch.setattr(mode, "_conflict_well_check", lambda *a: None)
    monkeypatch.setattr(mode, "_pool_seconly", lambda *a: {})
    monkeypatch.setattr(mode, "_add_corrected_columns", lambda *a: None)
    monkeypatch.setattr(mode, "_aggregate_wells", lambda *a: pd.DataFrame())
    monkeypatch.setattr(mode, "_add_well_fold", lambda *a: None)
    monkeypatch.setattr(mode, "_build_stats", lambda *a: pd.DataFrame())
    captured = {}

    class StopAfterCsv(Exception):
        pass

    def capture_csv(*args, **kwargs):
        captured["metadata"] = args[-2]
        captured["figure_scale"] = args[-1]
        captured.update(kwargs)
        raise StopAfterCsv

    monkeypatch.setattr(mode, "_write_csvs", capture_csv)
    with pytest.raises(StopAfterCsv):
        mode.run_if_batch(cfg, "unused.yaml", tmp_path, tmp_path, {})
    assert scales == [0.13]
    assert captured["metadata"] == {"synthetic.tif": 0.065}
    assert captured["figure_scale"] == 0.13
    assert captured["pixel_size_um_config_override"] == 0.13
