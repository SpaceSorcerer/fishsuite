from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from fishsuite.cli import cli
from fishsuite.core import io


def test_diagnostics_capture_vendor_area_rejections_without_changing_masks(monkeypatch, tmp_path):
    import sys
    from fishsuite.core import segmentation
    from fishsuite.core._vendor.segmentation import segment_image
    monkeypatch.setitem(sys.modules, 'cellpose', SimpleNamespace(models=object()))
    model_masks = np.zeros((50, 50), dtype=np.int32)
    model_masks[5:15, 5:15] = 1  # 100 model px: vendor floor rejects this
    model_masks[20:45, 20:45] = 2  # 625 model px: survives vendor, fails native 4000
    model = SimpleNamespace(eval=lambda *a, **k: (model_masks.copy(), None))
    original = lambda *a: model
    monkeypatch.setattr(segment_image, '_get_cellpose_model', original)
    params = dict(min_area=4000, diameter=84.61538461538461, cellpose_downsample_factor=2)
    normal = segmentation.segment_nuclei(np.ones((100, 100)), backend='cellpose', params=params)
    diagnostics = {}
    observed = segmentation.segment_nuclei(np.ones((100, 100)), backend='cellpose', params=params,
                                         diagnostics=diagnostics)
    assert np.array_equal(normal, observed)
    assert np.count_nonzero(diagnostics['labels_backend_area_rejected']) == 400
    assert np.count_nonzero(diagnostics['labels_before_area']) == 2500
    assert segment_image._get_cellpose_model is original
    root = tmp_path / 'images'
    root.mkdir()
    (root / 'field.tif').touch()
    config = tmp_path / 'preset.yaml'
    config.write_text('''channels: {analysis_mode: rna_rna, dapi: 0, rna: 1, rna2: 2}
z_stack: {mode: single, start_slice: 1, end_slice: 1}
nuclei: {backend: cellpose, min_area_px: 4000, cellpose_downsample_factor: 2, exclude_border: false}
conditions: {subfolder_conditions: {"": control}}
''')
    monkeypatch.setattr(io, 'read_image', lambda p: SimpleNamespace(n_channels=3, n_z=1, voxel_xy_nm=130.0))
    monkeypatch.setattr(io, 'extract_channel', lambda *a, **k: np.ones((100, 100)))
    out = tmp_path / 'out'
    result = CliRunner().invoke(cli, ['seg-sweep', '-c', str(config), '-i', str(root),
        '--diameters-um', '11', '--out', str(out)])
    assert result.exit_code == 0, result.output + repr(result.exception)
    row = pd.read_csv(out / 'seg_sweep_summary.csv').iloc[0]
    assert row.n_rejected_small == 2
    assert row.n_kept == 0
    assert np.isnan(row.median_area_um2)
    model.eval = lambda *a, **k: (_ for _ in ()).throw(RuntimeError('planted backend failure'))
    with pytest.raises(RuntimeError, match='planted backend failure'):
        segmentation.segment_nuclei(np.ones((100, 100)), backend='cellpose', params=params, diagnostics={})
    assert segment_image._get_cellpose_model is original


@pytest.mark.parametrize('scale,expected_area,expected_px', [
    (0.13, 6.76, [42.30769230769231, 50.0]),
    (0.065, 1.69, [84.61538461538461, 100.0]),
])
@pytest.mark.parametrize('mode', ['rna_rna', 'rna_protein', 'rna_only'])
def test_sweep_model_units_and_outputs(tmp_path, monkeypatch, scale, expected_area, expected_px, mode):
    from fishsuite.core import seg_sweep
    from fishsuite.core._vendor.segmentation import segment_image
    root = tmp_path / 'inputs'
    root.mkdir()
    for name in ['b.tif', 'a.tif']:
        (root / name).touch()
    config = tmp_path / 'preset.yaml'
    config.write_text('''channels:
  analysis_mode: rna_rna
  dapi: 0
  rna: 1
  rna2: 2
  antibody: 2
z_stack:
  mode: single
  start_slice: 1
  end_slice: 1
nuclei:
  backend: cellpose
  cellpose_device: directml
  cellpose_downsample_factor: 2
  min_area_px: 300
  exclude_border: true
  border_margin_px: 1
conditions:
  subfolder_conditions: {"": control}
''')
    config.write_text(config.read_text().replace('analysis_mode: rna_rna', f'analysis_mode: {mode}'))
    opened = []
    def read(path):
        opened.append(path.name)
        return SimpleNamespace(n_channels=3, n_z=1, voxel_xy_nm=scale * 1000)
    monkeypatch.setattr(io, 'read_image', read)
    monkeypatch.setattr(io, 'extract_channel', lambda *a, **k: np.ones((100, 100), dtype=np.uint16))
    diameters = []
    def backend(name, plane, **kwargs):
        assert name == 'cellpose'
        assert kwargs['cellpose_device'] == 'directml'
        assert plane.shape == (50, 50)
        diameters.append(kwargs['diameter'])
        labels = np.zeros((50, 50), dtype=np.int32)
        labels[15:25, 15:25] = 1
        labels[30:35, 30:35] = 2
        labels[:10, 35:45] = 3
        return labels
    monkeypatch.setattr(segment_image, 'run_backend', backend)
    out = tmp_path / 'out'
    result = CliRunner().invoke(cli, ['seg-sweep', '-c', str(config), '-i', str(root),
        '--diameters-um', '11,13', '--max-images-per-condition', '2', '--out', str(out)])
    assert result.exit_code == 0, result.output + repr(result.exception)
    table = pd.read_csv(out / 'seg_sweep_summary.csv')
    assert len(table) == 4
    assert diameters == pytest.approx(expected_px * 2)
    assert table.diameter_px_model_input.tolist() == pytest.approx(expected_px * 2)
    assert table.n_kept.tolist() == [1] * 4
    assert table.n_rejected_small.tolist() == [1] * 4
    assert table.n_rejected_border.tolist() == [1] * 4
    assert table.median_area_um2.tolist() == pytest.approx([expected_area] * 4)
    assert table.iqr_area_um2.tolist() == [0.0] * 4
    assert sorted(opened) == ['a.tif', 'b.tif']
    for name in ['seg_sweep_contact_sheet.png', 'seg_sweep_area_hist.png', 'command.log', 'versions.txt']:
        assert (out / name).stat().st_size > 0
    from PIL import Image
    with Image.open(out / 'seg_sweep_contact_sheet.png') as sheet:
        assert sheet.info['dpi'] == pytest.approx((600, 600), abs=0.01)


def test_selection_is_order_independent_and_seeded():
    from fishsuite.core.seg_sweep import choose_images
    images = [io.DiscoveredImage(Path(f'{i}.tif'), c, False, '')
              for c in ['b', 'a'] for i in range(8)]
    selected = choose_images(images, 2, 0)
    assert selected == choose_images(list(reversed(images)), 2, 0)
    assert selected != choose_images(images, 2, 1)
    assert [im.condition for im in selected] == ['a', 'a', 'b', 'b']


@pytest.mark.parametrize('diameters', ['0', '-1', 'nan', 'inf', 'x', '11,11'])
def test_invalid_diameters_fail_before_reading(tmp_path, diameters):
    config = tmp_path / 'preset.yaml'
    config.write_text('{}')
    result = CliRunner().invoke(cli, ['seg-sweep', '-c', str(config), '-i', str(tmp_path),
        '--diameters-um', diameters, '--out', str(tmp_path / 'out')])
    assert result.exit_code != 0
    assert 'diameter' in result.output.lower()
