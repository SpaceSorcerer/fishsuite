"""Synthetic run-directory integration for the orthogonal figure command."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import tifffile
from click.testing import CliRunner

from fishsuite.cli import cli


def test_ortho_help_registered():
    result = CliRunner().invoke(cli, ['ortho', '--help'])
    assert result.exit_code == 0, result.output
    assert '--half-width-um' in result.output
    assert '--qki-min' in result.output and '--miat-min' in result.output


@pytest.fixture
def synthetic_run(tmp_path, monkeypatch):
    from fishsuite.core import io

    run = tmp_path / 'run'
    run.mkdir()
    (run / 'masks').mkdir()
    source = tmp_path / 'synthetic.tif'
    source.write_bytes(b'synthetic reader boundary')
    cfg = {'channels': {'analysis_mode': 'rna_protein', 'rna': 0,
                        'antibody': 1, 'dapi': 2},
           'z_stack': {'mode': 'autofocus'},
           'output': {'manual_rna_min': 0, 'manual_rna_max': 100,
                      'manual_antibody_min': 0, 'manual_antibody_max': 200}}
    (run / 'run_config.json').write_text(json.dumps(
        {'input_dir': str(tmp_path), 'config_resolved': cfg}))
    pd.DataFrame([{'image': 'synthetic.tif', 'condition': 'well A', 'group': 'arm A',
                   'nucleus_id': 1, 'nuclear_spot_count': 2}]).to_csv(
                       run / 'nuclei_metrics.csv', index=False)
    pd.DataFrame([{'image': 'synthetic.tif', 'channel': 'rna1', 'nucleus_id': 1,
                   'x_px': 16, 'y_px': 15, 'z_slice': 0, 'spot_peak_intensity': 100},
                  {'image': 'synthetic.tif', 'channel': 'protein', 'nucleus_id': 1,
                   'x_px': 5, 'y_px': 5, 'z_slice': 0, 'spot_peak_intensity': 1000}]).to_csv(
                       run / 'spot_metrics.csv', index=False)
    pd.DataFrame([{'image': 'synthetic.tif', 'source_path': str(source),
                   'output_stem': 'well_A__synthetic'}]).to_csv(
                       run / 'resolved_experiment_hierarchy.csv', index=False)
    pd.DataFrame([{'image': 'synthetic.tif', 'voxel_xy_nm': 130.,
                   'voxel_z_nm': 210.}]).to_csv(run / 'per_image_summary.csv', index=False)
    labels = np.zeros((32, 32), dtype=np.uint16)
    labels[8:24, 7:26] = 1
    tifffile.imwrite(run / 'masks' / 'well_A__synthetic__nuclei_label_mask.tif', labels)
    z, y, x = np.indices((12, 32, 32))
    blob = np.exp(-((z-7)**2 + (y-15)**2 + (x-16)**2)/2)
    stack = np.stack([100*blob, 200*blob, np.broadcast_to(labels, blob.shape)])
    image = SimpleNamespace(n_channels=3, n_z=12, voxel_xy_nm=130., voxel_z_nm=210.,
                            bio=SimpleNamespace(get_image_data=lambda order, T, C: stack[C]))
    monkeypatch.setattr(io, 'read_image', lambda path: image)
    return run, cfg


def test_ortho_renders_records_fallback_and_exact_arguments(synthetic_run):
    run, _ = synthetic_run
    out = run / 'figures'
    args = ['ortho', '--run-dir', str(run), '--out', str(out), '--k', '1',
            '--half-width-um', '2', '--seed', '17']
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    selected = pd.read_csv(out / 'ortho_selection.csv')
    assert selected.loc[0, 'arm'] == 'arm A'
    assert selected.loc[0, 'punctum_z'] == 7
    assert selected.loc[0, 'punctum_y'] == 15
    assert selected.loc[0, 'punctum_x'] == 16
    assert selected.loc[0, 'z_source'] == 'miat_5x5_mean_argmax'
    assert selected.loc[0, 'line_p0_x'] == 7
    assert selected.loc[0, 'line_p1_x'] == 25
    assert selected.loc[0, 'arm_median'] == 2
    assert len(list(out.glob('*.png'))) == 1
    assert len(list(out.glob('*.svg'))) == 1
    assert next(out.glob('*.png')).name == 'ortho_arm-A_synthetic_nuc1.png'
    assert '<text' in next(out.glob('*.svg')).read_text()
    from PIL import Image
    with Image.open(next(out.glob('*.png'))) as image:
        assert image.info['dpi'][0] == pytest.approx(600, abs=.01)
    from subprocess import list2cmdline
    # 2026-09-17 review: command.log is now a keyed provenance block (the bare
    # line was not runnable), so the exact command is asserted on its own key
    # and the runnable -m reproduction is asserted alongside it.
    log = (out / 'command.log').read_text()
    assert f"command: {list2cmdline(['fishsuite', *args])}" in log
    assert '-m fishsuite.cli ortho ' in log


def test_ortho_missing_display_levels_fails_before_images(synthetic_run, monkeypatch):
    run, cfg = synthetic_run
    cfg['output']['manual_antibody_max'] = None
    (run / 'run_config.json').write_text(json.dumps({'config_resolved': cfg}))
    from fishsuite.core import io
    monkeypatch.setattr(io, 'read_image', lambda path: pytest.fail('Must fail before reading images'))
    result = CliRunner().invoke(cli, ['ortho', '--run-dir', str(run)])
    assert result.exit_code != 0
    assert 'manual_antibody_min/max' in result.output


def test_ortho_rejects_missing_mask(synthetic_run):
    run, _ = synthetic_run
    (run / 'masks' / 'well_A__synthetic__nuclei_label_mask.tif').unlink()
    result = CliRunner().invoke(cli, ['ortho', '--run-dir', str(run), '--k', '1'])
    assert result.exit_code != 0
    assert 'nuclei_label_mask.tif' in result.output


def test_legacy_mask_without_explicit_mapping_is_rejected(synthetic_run):
    from fishsuite.report.ortho import _source_and_mask
    run, _ = synthetic_run
    row = pd.read_csv(run / 'nuclei_metrics.csv').iloc[0]
    rc = json.loads((run / 'run_config.json').read_text())
    with pytest.raises(ValueError, match='output_stem'):
        _source_and_mask(run, row, None, rc)


def test_analysed_plane_overrides_spot_z_and_brightest_plane():
    from fishsuite.report.ortho import _spot_center
    spots = pd.DataFrame([{'nucleus_id': 1, 'x_px': 4, 'y_px': 3,
                           'z_px': 2, 'z_slice': 0, 'peak_intensity': 100}])
    stack = np.zeros((3, 8, 10, 10))
    stack[0, 6, 1:6, 2:7] = 1000
    center, source = _spot_center(spots, 1, 'brightest', stack, analysed_plane_z=2)
    assert center == (2, 3, 4)
    assert source == 'analysed_plane'


def test_unrecorded_plane_uses_argmax_even_with_legacy_spot_z():
    from fishsuite.report.ortho import _spot_center
    spots = pd.DataFrame([{'nucleus_id': 1, 'x_px': 4, 'y_px': 3,
                           'z_px': 2, 'z_slice': 1, 'z_source': 'stack_0based',
                           'peak_intensity': 100}])
    stack = np.zeros((3,8,10,10))
    stack[0,6,1:6,2:7] = 1000
    center, source = _spot_center(spots,1,'brightest',stack)
    assert center == (6,3,4)
    assert source == 'miat_5x5_mean_argmax'


def test_prefixed_master_tables_and_config_overlay(synthetic_run, tmp_path, monkeypatch):
    from fishsuite.report import ortho
    run, cfg = synthetic_run
    cfg['output']['prefix'] = 'EXP_'
    for name in ('nuclei_metrics.csv', 'spot_metrics.csv', 'per_image_summary.csv'):
        (run / name).rename(run / ('EXP_' + name))
    (run / 'run_config.json').write_text(json.dumps({'config_resolved': cfg}))
    override = tmp_path / 'display.yaml'
    override.write_text('output:\n  manual_antibody_max: 300\n')
    class Figure:
        def savefig(self, *args, **kwargs):
            pass
    monkeypatch.setattr(ortho, 'render_ortho_figure', lambda *args, **kwargs: Figure())
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, 'close', lambda figure: None)
    result = CliRunner().invoke(cli, ['ortho', '--run-dir', str(run), '--k', '1',
                                    '--config', str(override)])
    assert result.exit_code == 0, result.output
    selected = pd.read_csv(run / 'ortho' / 'ortho_selection.csv')
    assert selected.loc[0, 'qki_display_max'] == 300
    assert selected.loc[0, 'miat_display_max'] == 100


@pytest.mark.parametrize('xy,z', [(65.,210.),(130.,420.)])
def test_calibration_mismatch_names_image_and_both_values(synthetic_run,xy,z):
    run,_ = synthetic_run
    pd.DataFrame([{'image':'synthetic.tif','voxel_xy_nm':xy,'voxel_z_nm':z}]).to_csv(
        run/'per_image_summary.csv',index=False)
    result = CliRunner().invoke(cli,['ortho','--run-dir',str(run),'--k','1'])
    assert result.exit_code != 0
    assert 'calibration mismatch' in result.output
    assert 'synthetic.tif' in result.output
    assert str(65 if xy == 65 else 420) in result.output
    assert str(130 if xy == 65 else 210) in result.output


def test_plane_calibration_and_selection_provenance(synthetic_run,monkeypatch):
    from fishsuite.report import ortho
    import matplotlib.pyplot as plt
    run,_ = synthetic_run
    pd.DataFrame([{'image':'synthetic.tif','voxel_xy_nm':129.,'voxel_z_nm':210.,
                   'z_plane':3}]).to_csv(run/'per_image_summary.csv',index=False)
    pd.DataFrame([{'image':'synthetic.tif','protein_threshold_value':42}]).to_csv(
        run/'thresholds.csv',index=False)
    captured = {}
    original = ortho.render_ortho_figure
    def render(*args,**kwargs):
        fig = original(*args,**kwargs)
        captured['center'] = args[1]
        captured['plane'] = kwargs['analysed_plane_z']
        raw = next(ax for ax in fig.axes if ax.get_label() == 'raw')
        captured['raw_max'] = raw.lines[0].get_ydata().max()
        # File writing is tested by the existing full rendering test.
        fig.savefig = lambda *args,**kwargs: None
        return fig
    monkeypatch.setattr(ortho,'render_ortho_figure',render)
    result = CliRunner().invoke(cli,['ortho','--run-dir',str(run),'--k','1',
                                    '--seed','17','--punctum','median','--half-width-um','2'])
    assert result.exit_code == 0, result.output
    row = pd.read_csv(run/'ortho'/'ortho_selection.csv').iloc[0]
    assert captured['center'][0] == captured['plane'] == row.punctum_z == 2
    assert captured['raw_max'] < 1  # Peak is at z=7, analysed z=2 is deliberately faint.
    assert row.z_source == 'analysed_plane'
    assert row.punctum_rule == 'median'
    assert row.seed == 17 and row.k == 1 and row.half_width_um == 2
    assert row.profile_width_px == 3 and row.metric == 'nuclear_spot_count'
    assert row.pixel_size_um == .13 and row.z_step_um == .21
    assert 'per_image_summary.csv:voxel_xy_nm' in row.pixel_size_source
    assert 'per_image_summary.csv:voxel_z_nm' in row.z_step_source
    assert pd.isna(row.qki_min) and pd.isna(row.miat_min)
    assert row.display_mode == 'manual'
    assert row.configured_display_mode == 'auto_batch'
    assert 'per_image_summary.csv:z_plane' in row.analysed_plane_record


@pytest.mark.parametrize('z', [0,2.5,13])
def test_invalid_analysed_plane_is_not_silently_replaced(synthetic_run,z):
    run,_ = synthetic_run
    table = pd.read_csv(run/'per_image_summary.csv')
    table['z_plane'] = z
    table.to_csv(run/'per_image_summary.csv',index=False)
    result = CliRunner().invoke(cli,['ortho','--run-dir',str(run),'--k','1'])
    assert result.exit_code != 0
    assert 'analysed plane' in result.output


def test_missing_run_calibration_is_not_silently_trusted(synthetic_run):
    run,_ = synthetic_run
    pd.DataFrame([{'image':'synthetic.tif','voxel_z_nm':210}]).to_csv(
        run/'per_image_summary.csv',index=False)
    result = CliRunner().invoke(cli,['ortho','--run-dir',str(run),'--k','1'])
    assert result.exit_code != 0
    assert 'recorded XY calibration' in result.output


@pytest.mark.parametrize('xy_nm', [65.,130.])
def test_calibration_preserves_both_acquisition_pixel_sizes(xy_nm):
    from fishsuite.report.ortho import _run_calibration
    image = SimpleNamespace(voxel_xy_nm=xy_nm, voxel_z_nm=210.)
    record = pd.Series({'image':'synthetic.tif','voxel_xy_nm':xy_nm,'voxel_z_nm':210.})
    xy,z,xy_source,z_source = _run_calibration(image,record,'per_image_summary.csv')
    assert xy == xy_nm/1000.
    assert z == .21
    assert 'validated against per_image_summary.csv:voxel_xy_nm' in xy_source
    assert 'validated against per_image_summary.csv:voxel_z_nm' in z_source


def test_user_minima_only_and_auto_crop_provenance(synthetic_run,monkeypatch):
    from fishsuite.report import ortho
    run,_ = synthetic_run
    pd.DataFrame([{'image':'synthetic.tif','protein_threshold_value':2787.5}]).to_csv(run/'thresholds.csv',index=False)
    captured = []
    original = ortho.render_ortho_figure
    def render(*args,**kwargs):
        fig = original(*args,**kwargs)
        raw = next(ax for ax in fig.axes if ax.get_label() == 'raw')
        captured.append([line.get_label() for line in raw.lines])
        y0,y1,x0,x1 = fig._ortho_crop_bounds
        assert y0 <= 8 and y1 >= 24 and x0 <= 7 and x1 >= 26
        fig.savefig = lambda *args,**kwargs: None
        return fig
    monkeypatch.setattr(ortho,'render_ortho_figure',render)
    for options in ([],['--qki-min','42','--miat-min','21']):
        result = CliRunner().invoke(cli,['ortho','--run-dir',str(run),'--k','1',*options])
        assert result.exit_code == 0,result.output
    assert not any('analysis' in label for label in captured[0])
    assert 'QKI analysis min (user)' in captured[1]
    assert 'MIAT analysis min (user)' in captured[1]
    row = pd.read_csv(run/'ortho'/'ortho_selection.csv').iloc[0]
    assert row.half_width_um == pytest.approx(19*.13/2+1.5)
    assert row.half_width_used_um == pytest.approx(row.half_width_px*.13)
    assert row.qki_min == 42 and row.miat_min == 21
