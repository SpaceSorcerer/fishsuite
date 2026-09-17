from click.testing import CliRunner
from fishsuite.cli import cli


def test_resume_rejected_before_runner(tmp_path, monkeypatch):
    from fishsuite import runner
    config = tmp_path / 'config.yaml'
    config.write_text('{}')
    calls = []
    monkeypatch.setattr(runner, 'run_batch', lambda **kw: calls.append(kw) or {})
    result = CliRunner().invoke(cli, ['run', '-c', str(config), '-i', str(tmp_path),
                                    '-o', str(tmp_path / 'out'), '--resume'])
    assert result.exit_code == 2
    assert '--resume is not implemented; re-run into a new output directory' in result.output
    assert calls == []
