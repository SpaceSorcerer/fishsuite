"""Completed-run report controls; subprocess arguments remain shell-free."""
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import csv
import json
import sys
import yaml

from fishsuite.config.schema import ConditionsCfg


def parse_groups(text):
    raw = yaml.safe_load(text) or {}
    if not isinstance(raw, dict):
        raise ValueError('Groups must be a YAML mapping: condition: [well1, well2]')
    return ConditionsCfg(groups=raw).groups


def validate_run(run):
    missing = [n for n in ('per_image_summary.csv', 'nuclei_metrics.csv', 'run_config.json')
               if not (run / n).is_file()]
    if missing:
        raise ValueError('Missing completed-run files: ' + ', '.join(missing))


@dataclass
class ReportOptions:
    run: Path
    out: Path
    plot_style: str = 'replicate-simple'
    deck: bool = False
    micrographs: bool = False
    dapi: bool = False
    correction: bool = False
    deck_spec: str = ''
    corrected_csv: str = ''


def build_command(options, groups, spec=None, native=False):
    cmd = [sys.executable, '-u', '-m', 'fishsuite.cli',
           'native-figures' if native else 'report', '--run', str(options.run),
           '--groups' if native else '--groups-file', str(groups), '--out', str(options.out)]
    if not native:
        cmd += ['--plot-style', options.plot_style]
        if spec:
            cmd += ['--deck-spec', str(spec)]
        if options.deck:
            cmd += ['--deck']
        if options.micrographs:
            cmd += ['--micrograph-slides', 'per-well']
    return cmd


def prepare_inputs(options, groups_text, native=False):
    """Validate before staging in a fresh sibling; never create the CLI output."""
    validate_run(options.run)
    run, out = options.run.resolve(), options.out.resolve()
    if out.exists() or out == run or out.is_relative_to(run) or run.is_relative_to(out):
        raise ValueError('Choose a NEW output folder outside the run.')
    raw = yaml.safe_load(groups_text)
    if not isinstance(raw, dict):
        raise ValueError('Groups YAML must contain a groups mapping.')
    block = raw.get('conditions', raw)
    cfg = ConditionsCfg.model_validate(block)
    if not cfg.groups:
        raise ValueError('Declare at least one condition group.')
    template = None
    if not native:
        if options.correction and not options.dapi:
            raise ValueError('Sec-only correction requires DAPI localization in this report backend.')
        if options.micrographs and not options.deck:
            raise ValueError('Per-well micrograph slides require Deck.')
        if options.deck or options.dapi or options.correction or options.deck_spec:
            source = Path(options.deck_spec)
            if not options.deck_spec or not source.is_file():
                raise ValueError('Select an existing validated deck spec for deck/DAPI/correction.')
            template = yaml.safe_load(source.read_text(encoding='utf-8-sig'))
            if not isinstance(template, dict) or not template.get('existing_coloc'):
                raise ValueError('Deck spec must reference an existing_coloc persisted panel.')
            # Relative asset references are interpreted beside the selected spec.
            for key in ('existing_coloc', 'baseline_manifest', 'secondary_corrected_csv'):
                if template.get(key):
                    p = Path(template[key])
                    template[key] = str((source.parent / p).resolve() if not p.is_absolute() else p)
            if not Path(template['existing_coloc']).is_file():
                raise ValueError('Deck spec persisted panel is missing.')
            template['dapi_localization'] = options.dapi
            correction = options.corrected_csv.strip() or template.get('secondary_corrected_csv', '')
            template.pop('secondary_corrected_csv', None)
            if options.correction:
                if not correction or not Path(correction).is_file():
                    raise ValueError('Select the matched secondary-corrected CSV.')
                template['secondary_corrected_csv'] = str(Path(correction).resolve())
    stage = out.parent / (out.name + '_inputs_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    stage.mkdir(parents=True, exist_ok=False)
    groups = stage / 'groups.yaml'
    groups.write_text(yaml.safe_dump(block, sort_keys=False), encoding='utf-8')
    spec = None
    if template is not None:
        spec = stage / 'deck_spec.yaml'
        spec.write_text(yaml.safe_dump(template, sort_keys=False), encoding='utf-8')
    return groups, spec


try:
    from PySide6.QtWidgets import (QWidget, QVBoxLayout, QFormLayout, QHBoxLayout,
        QLineEdit, QPushButton, QPlainTextEdit, QComboBox, QCheckBox, QFileDialog, QLabel)
except ImportError:
    pass  # The main entry point supplies the plain Qt install hint.
else:
    from .runner_proc import PipelineRunner

    class ReportTab(QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            layout = QVBoxLayout(self)
            form = QFormLayout()
            layout.addLayout(form)
            self.run_path = self._picker(form, 'Completed run', folder=True)
            self.run_path.editingFinished.connect(self.load_run)
            self.groups_path = self._picker(form, 'Groups YAML (optional)', callback=self.load_groups)
            self.groups_editor = QPlainTextEdit('groups: {}')
            self.groups_editor.setMaximumHeight(110)
            form.addRow('Condition → wells (YAML)', self.groups_editor)
            self.plot_style = QComboBox()
            self.plot_style.addItems(['replicate-simple', 'superplot'])
            form.addRow('Plot style', self.plot_style)
            self.deck_spec = self._picker(form, 'Validated deck spec')
            self.corrected_csv = self._picker(form, 'Matched sec-only corrected CSV')
            row = QHBoxLayout()
            self.deck = QCheckBox('Deck')
            self.micrographs = QCheckBox('Micrograph slides per-well')
            self.dapi = QCheckBox('DAPI localization')
            self.correction = QCheckBox('Sec-only correction')
            for check in (self.deck, self.micrographs, self.dapi, self.correction):
                row.addWidget(check)
            form.addRow(row)
            note = QLabel('Deck/DAPI require a validated spec and persisted panel. Correction also requires DAPI.\n'
                          'Native figures use grouping only. Persisted-panel reports force replicate-simple.')
            note.setWordWrap(True)
            layout.addWidget(note)
            self.output_path = self._picker(form, 'New output folder', folder=True)
            self.build = QPushButton('Build')
            self.native = QPushButton('Native figures by condition')
            self.stop = QPushButton('Stop')
            self.stop.setEnabled(False)
            buttons = QHBoxLayout()
            for button in (self.build, self.native, self.stop):
                buttons.addWidget(button)
            layout.addLayout(buttons)
            self.log = QPlainTextEdit()
            self.log.setReadOnly(True)
            self.log.setMaximumBlockCount(20000)
            layout.addWidget(self.log, 1)
            self.runner = PipelineRunner(self)
            self.runner.line.connect(lambda line: self.log.appendPlainText(line.rstrip('\n')))
            self.runner.finished.connect(self._finished)
            self.build.clicked.connect(lambda: self.start(False))
            self.native.clicked.connect(lambda: self.start(True))
            self.stop.clicked.connect(self.runner.stop)

        def _picker(self, form, label, folder=False, callback=None):
            edit = QLineEdit()
            row = QHBoxLayout()
            row.addWidget(edit)
            button = QPushButton('Browse…')
            row.addWidget(button)
            def pick():
                path = (QFileDialog.getExistingDirectory(self, label) if folder else
                        QFileDialog.getOpenFileName(self, label)[0])
                if path:
                    edit.setText(path)
                    if edit is self.run_path:
                        self.load_run()
                    elif callback:
                        callback()
            button.clicked.connect(pick)
            form.addRow(label, row)
            return edit

        def load_groups(self):
            try:
                self.groups_editor.setPlainText(Path(self.groups_path.text()).read_text(encoding='utf-8-sig'))
            except OSError as exc:
                self.log.appendPlainText(str(exc))

        def load_run(self):
            try:
                run = Path(self.run_path.text()).resolve()
                validate_run(run)
                self.output_path.setText(str(run.parent / ('report_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))))
                cfg = json.loads((run / 'run_config.json').read_text(encoding='utf-8-sig'))
                conditions = cfg.get('config_resolved', cfg).get('conditions', {})
                groups = conditions.get('groups') or {}
                if not groups:
                    # Persisted condition names are the run's well-folder identities;
                    # do not guess biological pooling from name prefixes.
                    with (run / 'per_image_summary.csv').open(encoding='utf-8-sig', newline='') as f:
                        wells = sorted({r['condition'] for r in csv.DictReader(f)
                                        if r.get('secondary_only', '').lower() not in ('true', '1', '1.0')})
                    groups = {well: [well] for well in wells}
                self.groups_path.clear()
                self.groups_editor.setPlainText(yaml.safe_dump({**conditions, 'groups': groups}, sort_keys=False))
                self.log.appendPlainText('Run files found. Review conditions / biological wells / technical FOVs. Without saved groups the suggested identity map has one well per condition; assign wells before condition inference.')
            except (OSError, ValueError, KeyError) as exc:
                self.log.appendPlainText(str(exc))

        def start(self, native=False):
            if self.runner.is_running():
                return
            try:
                if not self.run_path.text().strip() or not self.output_path.text().strip():
                    raise ValueError('Select a completed run and a new output folder.')
                options = ReportOptions(Path(self.run_path.text()).resolve(), Path(self.output_path.text()).resolve(),
                    self.plot_style.currentText(), self.deck.isChecked(), self.micrographs.isChecked(),
                    self.dapi.isChecked(), self.correction.isChecked(), self.deck_spec.text(), self.corrected_csv.text())
                groups, spec = prepare_inputs(options, self.groups_editor.toPlainText(), native)
                self.build.setEnabled(False)
                self.native.setEnabled(False)
                self.stop.setEnabled(True)
                self.runner.start(build_command(options, groups, spec, native), None, options.out)
            except (OSError, ValueError, yaml.YAMLError) as exc:
                self.log.appendPlainText(str(exc))

        def _finished(self, success, output):
            self.build.setEnabled(True)
            self.native.setEnabled(True)
            self.stop.setEnabled(False)
            self.log.appendPlainText(('Complete: ' if success else 'Failed: ') + output)
