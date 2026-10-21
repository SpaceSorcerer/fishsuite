import csv
import pytest
from click.testing import CliRunner
from fishsuite.cli import cli


@pytest.mark.parametrize('failures, expected', [([], 0), ([('bad.tif', 'error', '')], 2)])
def test_cli_image_failure_exit(tmp_path, monkeypatch, failures, expected):
    from fishsuite import runner
    config = tmp_path / 'config.yaml'
    config.write_text('{}')
    monkeypatch.setattr(runner, 'run_batch', lambda **kw: {'failures': failures})
    result = CliRunner().invoke(cli, ['run', '-c', str(config), '-i', str(tmp_path),
                                    '-o', str(tmp_path / 'out')])
    assert result.exit_code == expected


def test_batch_records_failures_and_continues(tmp_path, monkeypatch, capsys):
    from fishsuite import runner
    from fishsuite.config.schema import FishsuiteConfig
    import subprocess
    cfg = FishsuiteConfig()
    cfg.output.pub_contrast_mode = 'manual'
    cfg.pixel_coloc.threshold_scope = 'per_image'
    monkeypatch.setattr(FishsuiteConfig, 'from_yaml', lambda p: cfg)
    monkeypatch.setattr(subprocess, 'call', lambda *a, **kw: 0)
    inputs = tmp_path / 'input'
    inputs.mkdir()
    paths = [inputs / 'a.tif', inputs / 'b.tif']
    for path in paths:
        path.touch()
    visited = []
    def fail(path, **kwargs):
        visited.append(path)
        raise ValueError('bad, image\nread failed')
    monkeypatch.setattr(runner, 'get_mode', lambda mode: fail)
    out = tmp_path / 'out'
    result = runner.run_batch(tmp_path / 'config.yaml', inputs, out, parallel=1)
    assert visited == paths
    with (out / 'failed_images.csv').open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [dict(path=str(p), exception_type='ValueError', message='bad, image\nread failed') for p in paths]
    assert result['n_ok'] == 0
    assert result['n_failed'] == 2
    assert 'n_ok=0 / n_failed=2' in capsys.readouterr().out


def test_grouped_image_failure_keeps_exit_two(tmp_path, monkeypatch):
    import json
    import subprocess
    from fishsuite import runner
    from fishsuite.config.schema import FishsuiteConfig
    cfg = FishsuiteConfig()
    cfg.output.pub_contrast_mode = 'manual'
    cfg.pixel_coloc.threshold_scope = 'per_image'
    cfg.conditions.subfolder_conditions = {'': 'WT'}
    cfg.conditions.groups = {'group': ['WT']}
    monkeypatch.setattr(FishsuiteConfig, 'from_yaml', lambda p: cfg)
    monkeypatch.setattr(subprocess, 'call', lambda *a, **kw: 0)
    def fail(path, **kwargs):
        raise ValueError('bad image')
    monkeypatch.setattr(runner, 'get_mode', lambda mode: fail)
    inputs = tmp_path / 'input'
    inputs.mkdir()
    (inputs / 'bad.tif').touch()
    config = tmp_path / 'config.yaml'
    config.write_text('{}')
    out = tmp_path / 'out'
    result = CliRunner().invoke(cli, ['run', '-c', str(config), '-i', str(inputs), '-o', str(out)])
    assert result.exit_code == 2, result.output
    assert 'CONDITION FIGURES INCOMPLETE' in result.output
    assert json.loads((out / 'condition_output_status.json').read_text())['status'] == 'failed'
    assert len(list(csv.DictReader((out / 'failed_images.csv').open()))) == 1


def _stub_batch(tmp_path, monkeypatch, cfg, mode_fn):
    import subprocess
    from fishsuite import runner
    from fishsuite.config.schema import FishsuiteConfig
    monkeypatch.setattr(FishsuiteConfig, 'from_yaml', lambda p: cfg)
    monkeypatch.setattr(subprocess, 'call', lambda *a, **kw: 0)
    monkeypatch.setattr(runner, 'get_mode', lambda mode: mode_fn)
    inputs = tmp_path / 'input'
    inputs.mkdir()
    (inputs / 'a.tif').touch()
    return lambda: runner.run_batch(tmp_path / 'config.yaml', inputs, tmp_path / 'out')


def test_sampling_shortfall_aborts_the_run(tmp_path, monkeypatch):
    from fishsuite.config.schema import FishsuiteConfig
    from fishsuite.core.segmentation import SamplingShortfallError
    cfg = FishsuiteConfig()
    cfg.output.pub_contrast_mode = 'manual'
    cfg.pixel_coloc.threshold_scope = 'per_image'

    def short(path, **kwargs):
        raise SamplingShortfallError('too few nuclei')
    run = _stub_batch(tmp_path, monkeypatch, cfg, short)
    with pytest.raises(SamplingShortfallError):
        run()


@pytest.mark.parametrize('pub, spots, analysis, expected', [
    (False, False, False, False),
    (False, True, False, True),
    (False, False, True, True),
    (True, False, False, True),
])
def test_contrast_prescan_runs_when_any_consumer_needs_it(tmp_path, monkeypatch,
                                                         pub, spots, analysis, expected):
    from fishsuite import runner
    from fishsuite.config.schema import FishsuiteConfig
    cfg = FishsuiteConfig()
    cfg.output.pub_contrast_mode = 'auto_batch'
    cfg.output.save_publication_images = pub
    cfg.output.apply_pub_contrast_floor_to_spots = spots
    cfg.output.apply_pub_contrast_floor_to_analysis = analysis
    cfg.pixel_coloc.threshold_scope = 'per_image'
    reads = []

    def read_image(path, *a, **kw):
        reads.append(path)
        raise OSError('unreadable')
    monkeypatch.setattr(runner._io, 'read_image', read_image)

    def fail(path, **kwargs):
        raise ValueError('bad image')
    run = _stub_batch(tmp_path, monkeypatch, cfg, fail)
    run()
    assert bool(reads) is expected


def test_preview_runs_only_the_named_image(tmp_path, monkeypatch):
    import yaml
    from fishsuite import runner
    config = tmp_path / 'config.yaml'
    config.write_text('seed: 3\n')
    image = tmp_path / 'input' / 'b.tif'
    image.parent.mkdir()
    image.touch()
    seen = {}
    monkeypatch.setattr(runner, 'run_batch', lambda **kw: seen.update(kw) or {})
    out = tmp_path / 'out'
    result = CliRunner().invoke(cli, ['preview', str(image), '-c', str(config), '-o', str(out)])
    assert result.exit_code == 0, result.output
    assert seen['config_path'] == out / 'preview_config.yaml'
    assert seen['input_dir'] == image.parent
    assert yaml.safe_load(seen['config_path'].read_text()) == {'seed': 3, 'input_file_subset': ['b.tif']}
