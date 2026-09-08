import os
import sys
import json
import importlib

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import pytest
import yaml

from fishsuite.gui.report_tab import ReportOptions, build_command, prepare_inputs


@pytest.fixture
def run(tmp_path):
    path = tmp_path / 'run'
    path.mkdir()
    (path / 'per_image_summary.csv').write_text('image,condition,secondary_only\na,WT_1,false\nb,KO_1,false\n')
    (path / 'nuclei_metrics.csv').write_text('image,nucleus_id\na,1\n')
    (path / 'run_config.json').write_text(json.dumps({'config_resolved': {}}))
    return path


@pytest.mark.parametrize('advanced', [False, True])
def test_commands(run, tmp_path, advanced):
    out = tmp_path / 'report_new'
    spec = tmp_path / 'spec.yaml'
    panel = tmp_path / 'panel.xlsx'
    panel.touch()
    correction = tmp_path / 'correction.csv'
    correction.write_text('level,image,nucleus_id,corrected_mean\n')
    spec.write_text(yaml.safe_dump({'existing_coloc': str(panel)}))
    options = ReportOptions(run, out, plot_style='superplot' if advanced else 'replicate-simple',
        deck=advanced, micrographs=advanced, dapi=advanced, correction=advanced,
        deck_spec=str(spec) if advanced else '', corrected_csv=str(correction))
    groups, staged = prepare_inputs(options, 'groups: {WT: [WT_1], KO: [KO_1]}')
    command = build_command(options, groups, staged)
    expected = [sys.executable, '-u', '-m', 'fishsuite.cli', 'report', '--run', str(run),
        '--groups-file', str(groups), '--out', str(out), '--plot-style', options.plot_style]
    if advanced:
        expected += ['--deck-spec', str(staged), '--deck', '--micrograph-slides', 'per-well']
        saved = yaml.safe_load(staged.read_text())
        assert saved['dapi_localization'] is True
        assert saved['secondary_corrected_csv'] == str(correction)
    assert command == expected
    assert build_command(options, groups, native=True) == [sys.executable, '-u', '-m',
        'fishsuite.cli', 'native-figures', '--run', str(run), '--groups', str(groups), '--out', str(out)]
    assert not out.exists()  # native-figures requires an uncreated destination


def test_validation(run, tmp_path):
    with pytest.raises(ValueError, match='DAPI'):
        prepare_inputs(ReportOptions(run, tmp_path / 'out', correction=True), 'groups: {WT: [WT_1]}')
    (run / 'nuclei_metrics.csv').unlink()
    with pytest.raises(ValueError, match='nuclei_metrics.csv'):
        prepare_inputs(ReportOptions(run, tmp_path / 'out'), 'groups: {WT: [WT_1]}')


def test_widget_and_run_groups(run, monkeypatch):
    pytest.importorskip('PySide6')
    from PySide6.QtWidgets import QApplication
    main = importlib.import_module('fishsuite.gui.main')
    monkeypatch.setattr(main._state, 'load_settings', lambda: {})
    monkeypatch.setattr(main._state, 'save_settings', lambda *_: None)
    app = QApplication.instance() or QApplication([])
    window = main.FishsuiteWindow()
    tab = window.report_tab
    tab.run_path.setText(str(run))
    tab.load_run()
    assert 'WT_1' in tab.groups_editor.toPlainText()
    assert tab.plot_style.currentText() == 'replicate-simple'
    assert tab.output_path.text().startswith(str(run.parent / 'report_'))
    assert window._tabs.indexOf(tab) >= 0
    window.run_groups.setPlainText('WT: [WT_1]\nKO: [KO_1]')
    assert window._read_widgets_into_cfg()['conditions']['groups'] == {'WT': ['WT_1'], 'KO': ['KO_1']}
    window.close()
    app.processEvents()


def test_missing_qt_hint(monkeypatch, capsys):
    main = importlib.import_module('fishsuite.gui.main')
    monkeypatch.setattr(main, '_QT_OK', False)
    assert main.main([]) == 1
    assert 'pip install PySide6' in capsys.readouterr().err


def test_cli_missing_qt_hint(monkeypatch):
    from click.testing import CliRunner
    from fishsuite.cli import cli
    main = importlib.import_module('fishsuite.gui.main')
    monkeypatch.setattr(main, '_QT_OK', False)
    monkeypatch.setattr(sys, 'argv', ['fishsuite', 'gui'])
    result = CliRunner().invoke(cli, ['gui'])
    assert result.exit_code == 1
    assert 'pip install PySide6' in result.output
