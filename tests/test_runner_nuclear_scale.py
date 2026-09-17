import json
from types import SimpleNamespace

import pytest

from fishsuite import runner
from fishsuite.config.schema import FishsuiteConfig


def test_dry_run_resolved_sizes(tmp_path, monkeypatch, capsys):
    from fishsuite.core import repro
    monkeypatch.setattr(repro, 'set_global_seeds', lambda *_: {})
    monkeypatch.setattr(repro, 'write_run_metadata', lambda *a, **k: None)
    root = tmp_path / 'input'
    root.mkdir()
    for name in ['a.tif', 'b.tif', 'c.tif']:
        (root / name).touch()
    monkeypatch.setattr(runner._io, 'read_image', lambda path: SimpleNamespace(voxel_xy_nm=65.0 if path.name == 'c.tif' else 130.0))
    config = tmp_path / 'config.yaml'
    config.write_text('nuclei:\n  expected_diameter_um: 11\n  cellpose_downsample_factor: 2\n')
    runner.run_batch(config, root, tmp_path / 'out', dry_run=True)
    assert capsys.readouterr().out.count('RESOLVED NUCLEAR SIZE') == 2
    assert not (tmp_path / 'out' / 'run_config.json').exists()
    assert not (tmp_path / 'out' / 'resolved_nuclear_size.json').exists()


@pytest.mark.parametrize('override', [0, 130])
def test_pixel_only_dry_run_never_opens_images(tmp_path, monkeypatch, override):
    from fishsuite.core import repro
    monkeypatch.setattr(repro, 'set_global_seeds', lambda *_: {})
    monkeypatch.setattr(repro, 'write_run_metadata', lambda *a, **k: None)
    root = tmp_path / 'input'
    root.mkdir()
    (root / 'a.tif').touch()
    opened = []
    def read(path):
        opened.append(path)
        return SimpleNamespace(voxel_xy_nm=130)
    monkeypatch.setattr(runner._io, 'read_image', read)
    config = tmp_path / 'config.yaml'
    config.write_text(f'nuclei:\n  cellpose_diameter_px: 100\nfoci:\n  bigfish_voxel_size_nm: {override}\n')
    runner.run_batch(config, root, tmp_path / 'out', dry_run=True)
    assert opened == []
    assert not (tmp_path / 'out' / 'run_config.json').exists()


@pytest.mark.parametrize('physical', [False, True])
def test_real_rna_output_provenance_and_threshold_schema(tmp_path, monkeypatch, physical):
    import numpy as np
    import pandas as pd
    from fishsuite.core import repro
    from fishsuite.core.modes import rna_only

    monkeypatch.setattr(repro, 'set_global_seeds', lambda *_: {})
    monkeypatch.setattr(repro, 'write_run_metadata', lambda *a, **k: None)
    root = tmp_path / 'input'
    root.mkdir()
    (root / 'a.tif').touch()
    cfg = FishsuiteConfig(
        channels={'analysis_mode': 'rna_only', 'dapi': 0, 'rna': 1, 'one_indexed': False},
        z_stack={'mode': 'maxproj'}, foci={'enabled': False}, cytoplasm={'enabled': False},
        pixel_coloc={'threshold_scope': 'per_image'},
        nuclei={'expected_diameter_um': 11} if physical else {'cellpose_diameter_px': 100},
        output={'save_publication_images': False, 'save_qc_overlays': False,
                'save_masks': True, 'pub_contrast_mode': 'manual'},
    )
    config = tmp_path / 'config.yaml'
    cfg.dump_yaml(config)
    opened = []
    def read(path):
        opened.append(path)
        return SimpleNamespace(n_channels=2, n_z=1, voxel_xy_nm=130, voxel_z_nm=230)
    monkeypatch.setattr(runner._io, 'read_image', read)
    monkeypatch.setattr(runner._io, 'extract_channel', lambda *a, **k: np.ones((16, 16), dtype=np.float32))
    monkeypatch.setattr(rna_only._seg, 'segment_nuclei', lambda image, **k: np.zeros_like(image, dtype=np.int32))
    original_keys = set()
    def mode(*args, **kwargs):
        result = rna_only.run_one(*args, **kwargs)
        original_keys.update(result.thresholds)
        return result
    monkeypatch.setattr(runner, 'get_mode', lambda _: mode)
    class OutputsWritten(BaseException):
        pass
    def stop_before_reports(**kwargs):
        raise OutputsWritten
    monkeypatch.setattr(runner, 'write_analysis_summary_workbook', stop_before_reports)
    out = tmp_path / 'out'
    with pytest.raises(OutputsWritten):
        runner.run_batch(config, root, out, parallel=1)
    assert len(opened) == (2 if physical else 1)
    run_config = json.loads((out / 'run_config.json').read_text())
    assert run_config['failures'] == []
    records = run_config['resolved_nuclear_size']
    assert json.loads((out / 'resolved_nuclear_size.json').read_text()) == records
    assert records[0]['sizes']['native_diameter_px']['value'] == pytest.approx(84.61538461538461 if physical else 100)
    expected = original_keys | {'dapi_label', 'rna_label'}
    assert 'resolved_nuclear_size' not in expected
    for path in [out / 'thresholds.csv', *list((out / 'masks').glob('*__thresholds.csv'))]:
        assert set(pd.read_csv(path).columns) == expected
    assert len(list((out / 'masks').glob('*__thresholds.csv'))) == 1


def test_fractional_native_area_filter(monkeypatch):
    import numpy as np
    from fishsuite.core.segmentation import segment_nuclei
    from fishsuite.core._vendor.segmentation import segment_image
    labels = np.zeros((30, 30), dtype=np.int32)
    labels[2:22, 2:22] = 1
    monkeypatch.setattr(segment_image, 'run_backend', lambda *a, **k: labels)
    result = segment_nuclei(np.ones((30, 30)), backend='otsu', params={'min_area': 400.5})
    assert not result.any()


def test_dry_run_missing_physical_scale_names_image(tmp_path, monkeypatch):
    from fishsuite.core import repro
    monkeypatch.setattr(repro, 'set_global_seeds', lambda *_: {})
    monkeypatch.setattr(repro, 'write_run_metadata', lambda *a, **k: None)
    root = tmp_path / 'input'
    root.mkdir()
    (root / 'unscaled.tif').touch()
    monkeypatch.setattr(runner._io, 'read_image', lambda path: SimpleNamespace(voxel_xy_nm=0))
    config = tmp_path / 'config.yaml'
    config.write_text('nuclei:\n  expected_diameter_um: 11\n')
    with pytest.raises(ValueError, match='unscaled.tif.*positive XY pixel size'):
        runner.run_batch(config, root, tmp_path / 'out', dry_run=True)


def test_sampling_methods_physical_limits():
    cfg = FishsuiteConfig(nuclei={'min_area_um2': 60, 'border_margin_um': 2})
    text = runner._sampling_methods_text(cfg, cfg.resolved_sampling(), None)
    assert 'minimum 60.0 um2' in text
    assert 'within 2.0 um' in text
    assert 'outside 10000' not in text
