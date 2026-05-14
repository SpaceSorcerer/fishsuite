"""Main PySide6 window for the fishsuite GUI launcher.

Comprehensive, modern, tabbed launcher built for Brian's dissertation work.
Eleven tabs (Experiment, Conditions, Channels, Z-Stack, Nuclei, Foci,
Pixel Coloc, Cytoplasm, Output, Run, YAML Preview) each with a colored
readiness dot. Dynamic visibility — backend-specific options only show
for the chosen backend. Live YAML preview. Subprocess management via
:mod:`runner_proc`. Settings persistence via :mod:`state`.

This file is large but self-contained — each tab is built by a private
``_build_tab_<name>`` method and is registered in ``_TABS`` so the readiness
loop can walk them generically. The single-file layout (vs splitting per
tab) is intentional: every tab shares the same widget map and read/write
hooks, and splitting would multiply boilerplate without simplifying.
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

# ---------------------------------------------------------------------------
# Qt import — headless-safe.
# ---------------------------------------------------------------------------
try:
    from PySide6.QtCore import Qt, QTimer, Signal, QSize
    from PySide6.QtGui import (
        QAction, QCloseEvent, QFont, QIcon, QKeySequence, QPalette, QShortcut,
        QTextCursor,
    )
    from PySide6.QtWidgets import (
        QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
        QFormLayout, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
        QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMenu,
        QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea,
        QSizePolicy, QSpinBox, QSplitter, QStackedWidget, QStyleFactory,
        QTabWidget, QTextEdit, QToolBar, QToolButton, QTreeWidget,
        QTreeWidgetItem, QVBoxLayout, QWidget,
    )
    _QT_OK = True
    _QT_ERR = None
except Exception as _exc:  # pragma: no cover - headless environments
    _QT_OK = False
    _QT_ERR = _exc

from . import readiness as _ready
from . import state as _state
from .runner_proc import PipelineRunner
from .widgets import (
    COLOR_GREEN, COLOR_GREY, COLOR_RED, COLOR_YELLOW, COLOR_TEAL,
    DARK_QSS, LIGHT_QSS,
)
if _QT_OK:
    from .widgets import (
        FolderConditionTable, IntSliderSpin, LabeledRow, ReorderableList,
        SectionHeader, SliderSpin, StatusDot, make_dot_icon, status_color,
    )


# ---------------------------------------------------------------------------
# Tab registry — drives the per-tab dot updates + iteration.
# Each entry: (key, display_label, readiness_func_name)
# ---------------------------------------------------------------------------

_TABS: List[tuple] = [
    ("experiment", "Experiment & Paths"),
    ("conditions", "Conditions"),
    ("channels", "Channels & Mode"),
    ("zstack", "Z-Stack"),
    ("nuclei", "Nuclei"),
    ("foci", "Foci / Spots"),
    ("pixel_coloc", "Pixel Coloc"),
    ("cytoplasm", "Cytoplasm"),
    ("output", "Output"),
    ("run", "Run"),
    ("yaml", "YAML Preview"),
]


# Tuning presets layered on top of the base preset. Mirror Fiji's "apply_profile"
# UX but keep it focused on the handful of tuning axes Brian iterates on.
TUNING_OVERLAYS: Dict[str, Dict[str, Any]] = {
    "(no overlay)": {},
    "Strict spots (tm=0.7)": {"foci": {"threshold_multiplier": 0.7}},
    "Default spots (tm=0.5)": {"foci": {"threshold_multiplier": 0.5}},
    "Permissive spots (tm=0.4)": {"foci": {"threshold_multiplier": 0.4}},
    "Strict pixel coloc (k_mad=3.0)": {"pixel_coloc": {"k_mad": 3.0}},
    "Default pixel coloc (k_mad=2.5)": {"pixel_coloc": {"k_mad": 2.5}},
    "Permissive pixel coloc (k_mad=2.0)": {"pixel_coloc": {"k_mad": 2.0}},
    "StarDist permissive (prob=0.3)": {"nuclei": {"prob_threshold": 0.3}},
    "StarDist strict (prob=0.5)": {"nuclei": {"prob_threshold": 0.5}},
}


if _QT_OK:

    class FishsuiteWindow(QMainWindow):
        """Top-level window."""

        def __init__(self):
            super().__init__()
            self.setWindowTitle("fishsuite — launcher")
            self.resize(1280, 880)
            self.setMinimumSize(QSize(1100, 740))

            # --- model state -------------------------------------------------
            self._settings: Dict[str, Any] = _state.load_settings()
            self._cfg: Dict[str, Any] = _state.schema_defaults()
            self._preset_overrides: Dict[str, Any] = {}  # changes on top of preset
            self._current_preset_stem: str = self._settings.get(
                "last_preset", _state.DEFAULT_PRESET_STEM
            )

            # --- runner ------------------------------------------------------
            self._runner = PipelineRunner(self)
            self._runner.line.connect(self._on_run_line)
            self._runner.progress.connect(self._on_run_progress)
            self._runner.finished.connect(self._on_run_finished)
            self._runner.phase_changed.connect(self._on_run_phase)
            self._last_output_dir: Optional[Path] = None

            # --- widget references keyed by tab key --------------------------
            self._tab_widgets: Dict[str, QWidget] = {}
            self._tab_indices: Dict[str, int] = {}
            self._statuses: Dict[str, str] = {}
            # Suppresses _on_field_changed while we're bulk-populating widgets
            # from a preset/overlay (otherwise SliderSpin.valueChanged would
            # fire mid-load and snapshot widgets that haven't been updated yet,
            # silently zeroing dapi/rna/etc. back to their construction default).
            self._loading_widgets: bool = False

            # Build top-level UI.
            self._build_ui()
            self._wire_shortcuts()

            # Initial population: load preset, apply settings, refresh dots.
            self._populate_preset_combo()
            self._select_preset(self._current_preset_stem, push_to_settings=False)
            self._restore_settings()
            self._refresh_readiness()
            self._refresh_yaml_preview()

            # Auto-refresh YAML on a debounced timer so heavy edits don't lag.
            self._yaml_timer = QTimer(self)
            self._yaml_timer.setInterval(180)
            self._yaml_timer.setSingleShot(True)
            self._yaml_timer.timeout.connect(self._refresh_yaml_preview)

            # Theme persistence — tri-state: system / light / dark.
            # Apply the saved theme. The combo widget itself is configured
            # below in _build_ui (it was earlier in init order so we delay
            # the visible-state sync until after the combo exists).
            self._apply_theme(_state.resolve_theme(self._settings))
            if hasattr(self, "_theme_combo"):
                self._sync_theme_combo()

        # ==================================================================
        # UI construction
        # ==================================================================

        def _build_ui(self) -> None:
            central = QWidget()
            self.setCentralWidget(central)
            outer = QVBoxLayout(central)
            outer.setContentsMargins(10, 10, 10, 10)
            outer.setSpacing(8)

            # ---- Top bar (preset + quick start + theme switcher) ----
            top = QFrame()
            top.setObjectName("topBar")
            top.setFrameShape(QFrame.NoFrame)
            tlay = QHBoxLayout(top)
            tlay.setContentsMargins(12, 8, 12, 8)
            tlay.setSpacing(10)

            tlay.addWidget(QLabel("<b>fishsuite</b>"))

            tlay.addWidget(QLabel("Preset:"))
            self._preset_combo = QComboBox()
            self._preset_combo.setMinimumWidth(220)
            self._preset_combo.currentTextChanged.connect(self._on_preset_selected)
            tlay.addWidget(self._preset_combo)

            self._load_preset_btn = QPushButton("Load")
            self._load_preset_btn.setToolTip("Reload selected preset (discarding unsaved tweaks)")
            self._load_preset_btn.clicked.connect(self._on_load_preset_btn)
            tlay.addWidget(self._load_preset_btn)

            self._save_preset_btn = QPushButton("Save as…")
            self._save_preset_btn.setToolTip("Save current settings as a new preset YAML")
            self._save_preset_btn.clicked.connect(self._on_save_preset_btn)
            tlay.addWidget(self._save_preset_btn)

            tlay.addSpacing(20)
            tlay.addWidget(QLabel("Quick overlay:"))
            self._overlay_combo = QComboBox()
            for k in TUNING_OVERLAYS.keys():
                self._overlay_combo.addItem(k)
            tlay.addWidget(self._overlay_combo)
            apply_overlay = QPushButton("Apply overlay")
            apply_overlay.clicked.connect(self._on_apply_overlay)
            tlay.addWidget(apply_overlay)

            tlay.addStretch(1)

            # Theme switcher: Light / Dark / System (auto). Persisted across
            # sessions via state.py. ``System`` resolves to whatever the OS
            # palette reports at runtime (re-resolved on each apply).
            tlay.addWidget(QLabel("Theme:"))
            self._theme_combo = QComboBox()
            self._theme_combo.addItem("System (auto)", userData="system")
            self._theme_combo.addItem("Light", userData="light")
            self._theme_combo.addItem("Dark", userData="dark")
            self._theme_combo.setToolTip(
                "Choose UI theme. 'System' follows your OS light/dark setting."
            )
            self._theme_combo.currentIndexChanged.connect(self._on_theme_changed)
            tlay.addWidget(self._theme_combo)

            outer.addWidget(top)

            # ---- Tabs ----
            self._tabs = QTabWidget()
            self._tabs.setMovable(False)
            self._tabs.setDocumentMode(True)
            outer.addWidget(self._tabs, 1)

            # Build every tab.
            self._build_tab_experiment()
            self._build_tab_conditions()
            self._build_tab_channels()
            self._build_tab_zstack()
            self._build_tab_nuclei()
            self._build_tab_foci()
            self._build_tab_pixel_coloc()
            self._build_tab_cytoplasm()
            self._build_tab_output()
            self._build_tab_run()
            self._build_tab_yaml()

            # ---- Footer ----
            footer = QHBoxLayout()
            footer.setContentsMargins(2, 0, 2, 0)
            self._footer_status = QLabel("ready")
            self._footer_status.setObjectName("footerStatus")
            self._footer_status.setProperty("state", "idle")
            footer.addWidget(self._footer_status)
            footer.addStretch(1)
            self._footer_output = QLabel("")
            self._footer_output.setObjectName("footerOutput")
            footer.addWidget(self._footer_output)
            outer.addLayout(footer)

        # ------------------------------------------------------------------
        # Tab 1 — Experiment & Paths
        # ------------------------------------------------------------------
        def _build_tab_experiment(self) -> None:
            w = QWidget()
            scroll = self._wrap_scroll(w)
            v = QVBoxLayout(w)
            v.setSpacing(8)
            v.setContentsMargins(16, 16, 16, 16)

            v.addWidget(SectionHeader(
                "Experiment metadata",
                "Free-text labels. Saved to the per-run config and into the dataset's provenance JSON."
            ))
            form = QFormLayout()
            form.setSpacing(8)
            self.exp_name = QLineEdit()
            self.exp_desc = QLineEdit()
            self.exp_cell = QLineEdit()
            self.exp_date = QLineEdit()
            self.exp_analyst = QLineEdit()
            for w_, key in [
                (self.exp_name, "name"),
                (self.exp_desc, "description"),
                (self.exp_cell, "cell_line"),
                (self.exp_date, "date"),
                (self.exp_analyst, "analyst"),
            ]:
                w_.textChanged.connect(self._on_field_changed)
                w_.setProperty("cfg_path", ("experiment", key))
            form.addRow("Experiment name", self.exp_name)
            form.addRow("Description", self.exp_desc)
            form.addRow("Cell line", self.exp_cell)
            form.addRow("Date", self.exp_date)
            form.addRow("Analyst", self.exp_analyst)
            v.addLayout(form)

            v.addWidget(SectionHeader(
                "Paths",
                "Input directory must contain images (flat or in subfolders). "
                "Output dir is auto-built per run."
            ))
            grid = QGridLayout()
            grid.setSpacing(8)
            grid.addWidget(QLabel("Input directory"), 0, 0)
            self.input_edit = QLineEdit()
            self.input_edit.textChanged.connect(self._on_field_changed)
            grid.addWidget(self.input_edit, 0, 1)
            in_browse = QPushButton("Browse…")
            in_browse.clicked.connect(self._browse_input)
            grid.addWidget(in_browse, 0, 2)

            grid.addWidget(QLabel("Output base"), 1, 0)
            self.output_base_edit = QLineEdit()
            self.output_base_edit.textChanged.connect(self._on_field_changed)
            grid.addWidget(self.output_base_edit, 1, 1)
            out_browse = QPushButton("Browse…")
            out_browse.clicked.connect(self._browse_output)
            grid.addWidget(out_browse, 1, 2)

            grid.addWidget(QLabel("Run tag"), 2, 0)
            self.tag_edit = QLineEdit()
            self.tag_edit.setPlaceholderText("e.g. h9-kd-sweep1")
            self.tag_edit.textChanged.connect(self._on_field_changed)
            grid.addWidget(self.tag_edit, 2, 1, 1, 2)

            grid.addWidget(QLabel("Computed output dir"), 3, 0)
            self.output_preview = QLabel("(set input + tag above)")
            self.output_preview.setObjectName("outputPreview")
            self.output_preview.setWordWrap(True)
            grid.addWidget(self.output_preview, 3, 1, 1, 2)
            v.addLayout(grid)

            # ---- Files in input dir (per-file checkable tree) ------------
            # Improvement 2 — the user can pick exactly which images to
            # include in the run, with subfolders surfaced as parent items.
            # Empty selection (or every leaf checked) = include all files
            # (legacy behavior). Selection is persisted per-input-dir via
            # state.file_subset_by_input.
            v.addWidget(SectionHeader(
                "Files in input dir",
                "Check the files you want to include in this run. Right-click "
                "a row for quick actions (include only this, exclude, "
                "mark sec-only, etc.). Empty selection = include all.",
            ))
            ftools = QHBoxLayout()
            self.file_tree_refresh_btn = QPushButton("Refresh")
            self.file_tree_refresh_btn.setToolTip(
                "Rescan the input directory and rebuild the tree below."
            )
            self.file_tree_refresh_btn.clicked.connect(self._on_file_tree_refresh)
            ftools.addWidget(self.file_tree_refresh_btn)
            self.file_tree_all_btn = QPushButton("Include all")
            self.file_tree_all_btn.clicked.connect(
                lambda: self._set_all_file_tree_checked(True)
            )
            ftools.addWidget(self.file_tree_all_btn)
            self.file_tree_none_btn = QPushButton("Include none")
            self.file_tree_none_btn.clicked.connect(
                lambda: self._set_all_file_tree_checked(False)
            )
            ftools.addWidget(self.file_tree_none_btn)
            self.file_tree_excl_sec_btn = QPushButton("Exclude all sec-only")
            self.file_tree_excl_sec_btn.setToolTip(
                "Uncheck every file whose basename is currently listed under "
                "Conditions → 'Secondary-only individual files (overrides)'."
            )
            self.file_tree_excl_sec_btn.clicked.connect(self._exclude_all_sec_only)
            ftools.addWidget(self.file_tree_excl_sec_btn)
            ftools.addWidget(QLabel("Filter:"))
            self.file_filter_edit = QLineEdit()
            self.file_filter_edit.setPlaceholderText("substring (case-insensitive)")
            self.file_filter_edit.textChanged.connect(self._apply_file_tree_filter)
            ftools.addWidget(self.file_filter_edit, 1)
            v.addLayout(ftools)

            # Live status line: "Included: 8 of 12 files (KO: 3, WT: 3, Sec-only: 2 excluded)"
            self.file_tree_summary = QLabel("(no files scanned yet)")
            self.file_tree_summary.setObjectName("previewChip")
            self.file_tree_summary.setWordWrap(True)
            v.addWidget(self.file_tree_summary)

            self.file_tree = QTreeWidget()
            self.file_tree.setHeaderLabels(["File", "Condition", "Sec-only"])
            self.file_tree.setRootIsDecorated(True)
            self.file_tree.setAlternatingRowColors(True)
            self.file_tree.setMinimumHeight(220)
            self.file_tree.itemChanged.connect(self._on_file_tree_item_changed)
            # Right-click context menu — see amendment from coordinator.
            self.file_tree.setContextMenuPolicy(Qt.CustomContextMenu)
            self.file_tree.customContextMenuRequested.connect(
                self._on_file_tree_context_menu
            )
            v.addWidget(self.file_tree, 1)

            # Internal state for the tree:
            self._file_tree_data: Dict[str, Any] = {}
            # When True, _on_file_tree_item_changed is a no-op (we're bulk-
            # updating checks).
            self._file_tree_loading: bool = False

            v.addStretch(0)

            self._register_tab("experiment", scroll, "Experiment & Paths")

        # ------------------------------------------------------------------
        # Tab 2 — Conditions
        # ------------------------------------------------------------------
        def _build_tab_conditions(self) -> None:
            w = QWidget()
            scroll = self._wrap_scroll(w)
            v = QVBoxLayout(w)
            v.setSpacing(8)
            v.setContentsMargins(16, 16, 16, 16)

            v.addWidget(SectionHeader(
                "Conditions & dataset layout",
                "Map subfolders (or filename patterns) to human-readable condition labels."
            ))
            form = QFormLayout()
            self.cond_mode = QComboBox()
            self.cond_mode.addItems(["subfolders", "auto", "explicit"])
            self.cond_mode.setToolTip(
                "subfolders — input dir contains one folder per condition "
                "(each folder name maps to a condition label below).\n"
                "auto — derive condition labels from filename patterns.\n"
                "explicit — every file's condition is assigned manually."
            )
            self.cond_mode.currentTextChanged.connect(self._on_field_changed)
            self.cond_mode.currentTextChanged.connect(lambda _t: self._refresh_conditions_visibility())
            form.addRow("Conditions mode", self.cond_mode)

            self.min_nuc = QSpinBox()
            self.min_nuc.setRange(1, 1000)
            self.min_nuc.setValue(6)
            self.min_nuc.valueChanged.connect(self._on_field_changed)
            form.addRow("min_nuclei_for_stats", self.min_nuc)
            v.addLayout(form)

            self.subf_box = QGroupBox("Subfolder → condition")
            self.subf_box.setToolTip("Each row maps a folder name to a human-readable condition label.")
            sbv = QVBoxLayout(self.subf_box)
            self.subf_table = FolderConditionTable()
            self.subf_table.changed.connect(self._on_field_changed)
            sbv.addWidget(self.subf_table)
            v.addWidget(self.subf_box)

            self.sec_box = QGroupBox("Secondary-only folders (controls)")
            self.sec_box.setToolTip("Folders whose images are no-probe / secondary-antibody-only controls.")
            sbv2 = QVBoxLayout(self.sec_box)
            self.sec_list = QListWidget()
            self.sec_list.setSelectionMode(QListWidget.MultiSelection)
            self.sec_list.itemSelectionChanged.connect(self._on_field_changed)
            sec_btns = QHBoxLayout()
            add_sec = QPushButton("+ Add folder")
            add_sec.clicked.connect(self._add_sec_folder)
            rm_sec = QPushButton("Remove selected")
            rm_sec.clicked.connect(self._remove_sec_folder)
            sec_btns.addWidget(add_sec)
            sec_btns.addWidget(rm_sec)
            sec_btns.addStretch(1)
            sbv2.addWidget(self.sec_list)
            sbv2.addLayout(sec_btns)
            v.addWidget(self.sec_box)

            # Sec-only individual file overrides (ported from Fiji GUI).
            # When a dataset doesn't follow the lab's naming convention and a
            # whole folder isn't the right granularity, you can flag specific
            # files here. Stored as `conditions.sec_only_files` in the YAML.
            self.sec_files_box = QGroupBox("Secondary-only individual files (overrides)")
            self.sec_files_box.setToolTip(
                "Specific files to flag as secondary-only controls when their "
                "names don't match auto-detection patterns. Listed by basename "
                "(matches against just the filename, no path)."
            )
            sfbv = QVBoxLayout(self.sec_files_box)
            self.sec_files_list = QListWidget()
            self.sec_files_list.setSelectionMode(QListWidget.MultiSelection)
            self.sec_files_list.itemSelectionChanged.connect(self._on_field_changed)
            sec_file_btns = QHBoxLayout()
            add_sec_f = QPushButton("+ Add files…")
            add_sec_f.clicked.connect(self._add_sec_files)
            rm_sec_f = QPushButton("Remove selected")
            rm_sec_f.clicked.connect(self._remove_sec_file)
            sec_file_btns.addWidget(add_sec_f)
            sec_file_btns.addWidget(rm_sec_f)
            sec_file_btns.addStretch(1)
            sfbv.addWidget(self.sec_files_list)
            sfbv.addLayout(sec_file_btns)
            v.addWidget(self.sec_files_box)

            self.order_box = QGroupBox("Condition order (drag to reorder)")
            self.order_box.setToolTip("Order used by downstream plotting / stats.")
            obv = QVBoxLayout(self.order_box)
            self.order_list = ReorderableList()
            self.order_list.changed.connect(self._on_field_changed)
            order_btns = QHBoxLayout()
            sync_btn = QPushButton("Sync from subfolder map")
            sync_btn.setToolTip("Replace this list with the unique condition labels in the table above.")
            sync_btn.clicked.connect(self._sync_order_from_table)
            order_btns.addWidget(sync_btn)
            order_btns.addStretch(1)
            obv.addWidget(self.order_list)
            obv.addLayout(order_btns)
            v.addWidget(self.order_box)

            v.addStretch(1)
            self._register_tab("conditions", scroll, "Conditions")

        def _refresh_conditions_visibility(self) -> None:
            mode = self.cond_mode.currentText()
            self.subf_box.setVisible(mode == "subfolders")
            self.sec_box.setVisible(mode == "subfolders")
            # Sec-only file overrides are useful in any mode (the schema field
            # is mode-agnostic), but they're most relevant when sec_box is
            # visible — keep them together for UX consistency.
            self.sec_files_box.setVisible(mode == "subfolders")
            self.order_box.setVisible(True)

        def _add_sec_folder(self) -> None:
            # Suggest folder names from input_dir if present.
            in_dir = self.input_edit.text()
            initial = in_dir if in_dir and Path(in_dir).is_dir() else str(Path.home())
            chosen = QFileDialog.getExistingDirectory(self, "Pick secondary-only folder", initial)
            if not chosen:
                return
            name = Path(chosen).name
            if any(self.sec_list.item(i).text() == name for i in range(self.sec_list.count())):
                return
            self.sec_list.addItem(name)
            self._on_field_changed()

        def _remove_sec_folder(self) -> None:
            for it in self.sec_list.selectedItems():
                self.sec_list.takeItem(self.sec_list.row(it))
            self._on_field_changed()

        def _add_sec_files(self) -> None:
            """Add one or more sec-only file basenames (Fiji-style override)."""
            in_dir = self.input_edit.text()
            initial = in_dir if in_dir and Path(in_dir).is_dir() else str(Path.home())
            paths, _ = QFileDialog.getOpenFileNames(
                self,
                "Pick secondary-only files",
                initial,
                "Images (*.vsi *.czi *.tif *.tiff *.nd2 *.lif *.oib *.oif);;All files (*)",
            )
            if not paths:
                return
            existing = {self.sec_files_list.item(i).text()
                        for i in range(self.sec_files_list.count())}
            for p in paths:
                name = Path(p).name
                if name not in existing:
                    self.sec_files_list.addItem(name)
                    existing.add(name)
            self._on_field_changed()

        def _remove_sec_file(self) -> None:
            for it in self.sec_files_list.selectedItems():
                self.sec_files_list.takeItem(self.sec_files_list.row(it))
            self._on_field_changed()

        def _sync_order_from_table(self) -> None:
            sub = self.subf_table.to_dict()
            labels: List[str] = []
            for v in sub.values():
                if v and v not in labels:
                    labels.append(v)
            self.order_list.set_items(labels)
            self._on_field_changed()

        # ------------------------------------------------------------------
        # Tab 3 — Channels & analysis mode
        # ------------------------------------------------------------------
        def _build_tab_channels(self) -> None:
            w = QWidget()
            scroll = self._wrap_scroll(w)
            v = QVBoxLayout(w)
            v.setSpacing(8)
            v.setContentsMargins(16, 16, 16, 16)

            # ---- Detected channels (read-only metadata from first VSI/CZI) ----
            # User clicks "Detect" → we open the first non-sec-only file in the
            # input dir with bioio and surface its per-channel metadata names
            # (e.g. "640 CSU / Cy5 (em ~668 nm)"). The role-spinner rows below
            # then show the resolved channel name beside each index, so the
            # user sees ch index → microscope channel → user label → role at
            # a glance. Persisted across sessions via state.last_detected_channels.
            detect_box = QGroupBox("Detected channels (from first file)")
            detect_box.setToolTip(
                "Click 'Detect' to read the first non-sec-only image in the "
                "input directory and surface its per-channel metadata. The "
                "channel index spinners below stay editable — this panel is "
                "read-only context to help you pick the right index per role."
            )
            db_lay = QVBoxLayout(detect_box)
            db_btn_row = QHBoxLayout()
            self.detect_channels_btn = QPushButton("Detect channels from sample file")
            self.detect_channels_btn.setToolTip(
                "Reads the first non-sec-only image in the input directory "
                "and shows its channel-name metadata + voxel size below."
            )
            self.detect_channels_btn.clicked.connect(self._on_detect_channels)
            db_btn_row.addWidget(self.detect_channels_btn)
            db_btn_row.addStretch(1)
            db_lay.addLayout(db_btn_row)
            self.detected_channels_label = QLabel(
                "(no file scanned yet — click Detect to read the first file)"
            )
            self.detected_channels_label.setObjectName("previewChip")
            self.detected_channels_label.setWordWrap(True)
            self.detected_channels_label.setTextInteractionFlags(
                Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard
            )
            db_lay.addWidget(self.detected_channels_label)
            v.addWidget(detect_box)

            v.addWidget(SectionHeader(
                "Analysis mode",
                "Pick the experiment type. Channel inputs below are filtered to match."
            ))
            form = QFormLayout()
            self.mode_combo = QComboBox()
            self.mode_combo.addItems([
                "rna_only", "rna_protein", "rna_rna", "ab_ab", "protein_only", "pub_images",
            ])
            self.mode_combo.setToolTip(
                "rna_only — DAPI + 1 RNA channel (spots + per-cell counts).\n"
                "rna_protein — DAPI + RNA + antibody (per-pixel + per-spot coloc).\n"
                "rna_rna — DAPI + 2 RNA channels (spot-on-spot NN coloc per nucleus).\n"
                "ab_ab — DAPI + 2 antibody channels (pixel-based coloc only).\n"
                "protein_only — DAPI + antibody (intensity quantification).\n"
                "pub_images — render per-channel + composite PNG/TIF, no quantification."
            )
            self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
            form.addRow("Analysis mode", self.mode_combo)

            self.one_indexed_chk = QCheckBox("Channel indices are 1-indexed (Fiji-style)")
            self.one_indexed_chk.toggled.connect(self._on_field_changed)
            form.addRow("", self.one_indexed_chk)
            v.addLayout(form)

            v.addWidget(SectionHeader(
                "Channel indices + labels",
                "Use -1 for auto-detect. Labels are human-readable names "
                "that appear in publication filenames, QC overlay legends, "
                "and thresholds.csv. Leave blank to use the role default "
                "(e.g. \"RNA1\").",
            ))
            chan_grid = QGridLayout()
            chan_grid.setSpacing(8)
            self.chan_dapi = self._make_chan_spin()
            self.chan_rna = self._make_chan_spin()
            self.chan_rna2 = self._make_chan_spin()
            self.chan_ab = self._make_chan_spin()
            self.chan_ab2 = self._make_chan_spin()

            # Per-channel label QLineEdits — placeholder shows the schema
            # default so users see what they get when leaving the field blank.
            self.chan_dapi_label = self._make_label_edit("DAPI")
            self.chan_rna_label = self._make_label_edit("RNA1")
            self.chan_rna2_label = self._make_label_edit("RNA2")
            self.chan_ab_label = self._make_label_edit("Protein")
            self.chan_ab2_label = self._make_label_edit("Protein2")

            # Per-channel LUT (pseudo-color) combos — 2026-05-13 issue 4
            # ("we need to ensure on the dual rna channels work that we are
            # able to fully select color outputs and all"). User picks the
            # LUT used by save_publication_images_bundle to tint each
            # channel; defaults match the historical Blue/Yellow/Cyan/
            # Magenta/Green so existing YAML configs render unchanged.
            self.chan_dapi_lut = self._make_lut_combo("blue")
            self.chan_rna_lut = self._make_lut_combo("yellow")
            self.chan_rna2_lut = self._make_lut_combo("cyan")
            self.chan_ab_lut = self._make_lut_combo("magenta")
            self.chan_ab2_lut = self._make_lut_combo("green")

            # Per-role "Detected name" italic labels — show the resolved
            # microscope channel name (from last detect-channels scan) for
            # whichever channel index is currently selected. Update live
            # when the user changes the spinner.
            def _make_resolved_label() -> QLabel:
                rl = QLabel("(detect to populate)")
                rl.setObjectName("resolvedChannelLabel")
                rl.setStyleSheet("font-style: italic; padding-left: 6px;")
                rl.setWordWrap(False)
                return rl

            self.chan_dapi_resolved = _make_resolved_label()
            self.chan_rna_resolved = _make_resolved_label()
            self.chan_rna2_resolved = _make_resolved_label()
            self.chan_ab_resolved = _make_resolved_label()
            self.chan_ab2_resolved = _make_resolved_label()

            # Refresh the resolved-label whenever the spinner changes.
            for sp in (self.chan_dapi, self.chan_rna, self.chan_rna2,
                       self.chan_ab, self.chan_ab2):
                sp.valueChanged.connect(self._refresh_resolved_channel_labels)

            self._chan_rows = {
                "dapi":      (QLabel("DAPI"),       self.chan_dapi,  self.chan_dapi_label,  self.chan_dapi_lut,  self.chan_dapi_resolved),
                "rna":       (QLabel("RNA"),        self.chan_rna,   self.chan_rna_label,   self.chan_rna_lut,   self.chan_rna_resolved),
                "rna2":      (QLabel("RNA #2"),     self.chan_rna2,  self.chan_rna2_label,  self.chan_rna2_lut,  self.chan_rna2_resolved),
                "antibody":  (QLabel("Antibody"),   self.chan_ab,    self.chan_ab_label,    self.chan_ab_lut,    self.chan_ab_resolved),
                "antibody2": (QLabel("Antibody #2"), self.chan_ab2,  self.chan_ab2_label,   self.chan_ab2_lut,   self.chan_ab2_resolved),
            }
            # Column headers
            chan_grid.addWidget(QLabel("Role"), 0, 0)
            chan_grid.addWidget(QLabel("Channel #"), 0, 1)
            chan_grid.addWidget(QLabel("Detected name"), 0, 2)
            chan_grid.addWidget(QLabel("Label"), 0, 3)
            chan_grid.addWidget(QLabel("Color (LUT)"), 0, 4)
            for i, (k, row) in enumerate(self._chan_rows.items(), start=1):
                lbl, sp, le, lc, rl = row
                chan_grid.addWidget(lbl, i, 0)
                chan_grid.addWidget(sp, i, 1)
                chan_grid.addWidget(rl, i, 2)
                chan_grid.addWidget(le, i, 3)
                chan_grid.addWidget(lc, i, 4)
            chan_grid.setColumnStretch(2, 1)
            chan_grid.setColumnStretch(3, 1)
            v.addLayout(chan_grid)

            v.addWidget(SectionHeader("Channel preview"))
            self.chan_preview = QLabel("(set analysis mode + indices)")
            self.chan_preview.setObjectName("previewChip")
            self.chan_preview.setWordWrap(True)
            v.addWidget(self.chan_preview)

            v.addStretch(1)
            self._refresh_channel_visibility()
            self._register_tab("channels", scroll, "Channels & Mode")

        def _make_chan_spin(self) -> QSpinBox:
            sp = QSpinBox()
            sp.setRange(-1, 31)
            sp.setSpecialValueText("auto (-1)")
            sp.setValue(-1)
            sp.valueChanged.connect(self._on_field_changed)
            return sp

        def _make_open_combo(
            self,
            *,
            items: List[str],
            default: str,
            tooltip: str = "",
        ) -> QComboBox:
            """Editable QComboBox seeded with a known-good list.

            Use for fields where there is a small set of *recommended* values
            (model names, postprocess steps) but custom strings should still
            round-trip through YAML. Unknown values loaded from a preset are
            inserted into the dropdown so the GUI never silently drops them.
            """
            cb = QComboBox()
            cb.setEditable(True)
            for it in items:
                cb.addItem(it)
            ix = cb.findText(default)
            cb.setCurrentIndex(ix if ix >= 0 else 0)
            if tooltip:
                cb.setToolTip(tooltip)
            cb.currentTextChanged.connect(self._on_field_changed)
            cb.editTextChanged.connect(self._on_field_changed)
            return cb

        def _set_open_combo(self, cb: QComboBox, value: str) -> None:
            """Set an editable combo's text; insert ``value`` if not in items.

            Keeps the dropdown a superset of every value ever loaded so the
            user can re-pick it later. Safe to call with empty/None.
            """
            v = str(value or "").strip()
            if not v:
                return
            cb.blockSignals(True)
            ix = cb.findText(v)
            if ix < 0:
                cb.addItem(v)
                ix = cb.findText(v)
            cb.setCurrentIndex(ix)
            cb.setEditText(v)
            cb.blockSignals(False)

        def _make_lut_combo(self, default_name: str) -> QComboBox:
            """ComboBox for selecting a per-channel LUT (pseudo-color).

            Items match the names accepted by output.lut_name_to_weights().
            Default selection matches the historical hard-coded color so
            unchanged configs produce unchanged output.
            """
            cb = QComboBox()
            for name in ("blue", "yellow", "cyan", "magenta", "green",
                         "red", "orange", "gray", "fire"):
                cb.addItem(name)
            ix = cb.findText(default_name)
            cb.setCurrentIndex(ix if ix >= 0 else 0)
            cb.currentTextChanged.connect(self._on_field_changed)
            return cb

        def _make_label_edit(self, default_placeholder: str) -> QLineEdit:
            """QLineEdit for a user-typed channel label.

            Placeholder = the schema default; leaving the field blank means
            "use the role default" (no need to retype "DAPI" in every preset).
            """
            le = QLineEdit()
            le.setPlaceholderText(default_placeholder)
            le.setMaxLength(40)
            le.textChanged.connect(self._on_field_changed)
            return le

        def _on_mode_changed(self, *_a) -> None:
            self._on_field_changed()
            self._refresh_channel_visibility()
            # Foci tab gates the RNA2 overrides group on analysis_mode == rna_rna.
            # The foci builder may not have run yet on first call — guard it.
            try:
                self._refresh_foci_visibility()
            except AttributeError:
                pass

        def _refresh_channel_visibility(self) -> None:
            mode = self.mode_combo.currentText()
            required = {
                "rna_only": {"dapi", "rna"},
                "rna_protein": {"dapi", "rna", "antibody"},
                "rna_rna": {"dapi", "rna", "rna2"},
                "ab_ab": {"dapi", "antibody", "antibody2"},
                "protein_only": {"dapi", "antibody"},
                "pub_images": set(),
            }.get(mode, set())
            display_names = {
                "dapi": "DAPI", "rna": "RNA", "rna2": "RNA #2",
                "antibody": "Antibody", "antibody2": "Antibody #2",
            }
            for k, row in self._chan_rows.items():
                lbl, sp, le = row[0], row[1], row[2]
                lc = row[3] if len(row) > 3 else None
                rl = row[4] if len(row) > 4 else None
                show = (k in required) or mode == "pub_images"
                lbl.setVisible(show)
                sp.setVisible(show)
                le.setVisible(show)
                if lc is not None:
                    lc.setVisible(show)
                if rl is not None:
                    rl.setVisible(show)
                tag = "required" if k in required else "optional"
                lbl.setText(f"{display_names[k]}  ({tag})")

        # ------------------------------------------------------------------
        # Detect-channels (Improvement 1) — read first non-sec-only file in
        # the input dir and surface its per-channel metadata.
        # ------------------------------------------------------------------
        def _on_detect_channels(self) -> None:
            in_dir = self.input_edit.text().strip()
            if not in_dir or not Path(in_dir).is_dir():
                QMessageBox.warning(
                    self, "fishsuite",
                    "Set an Input directory on the 'Experiment & Paths' tab "
                    "before detecting channels."
                )
                return
            # Pick the first non-sec-only image we can find. Use the same
            # extensions as the runner. Walk subfolders one level deep so a
            # subfolder-mode dataset (KO/, WT/, ...) still finds something.
            from ..core import io as _coreio
            try:
                subfolder_conditions = self._cfg.get("conditions", {}).get("subfolder_conditions") or {}
                sec_folders = self._cfg.get("conditions", {}).get("sec_only_folders") or []
                sec_files = self._cfg.get("conditions", {}).get("sec_only_files") or []
                imgs = _coreio.discover_inputs(
                    Path(in_dir),
                    subfolder_conditions=subfolder_conditions,
                    sec_only_folders=sec_folders,
                    sec_only_files=sec_files,
                )
            except Exception as e:
                QMessageBox.warning(
                    self, "fishsuite",
                    f"Could not scan input dir for sample file:\n{e}"
                )
                return
            # Prefer non-sec-only; fall back to the first image if every file
            # is flagged sec-only (rare).
            sample = next((im for im in imgs if not im.sec_only), None)
            if sample is None and imgs:
                sample = imgs[0]
            if sample is None:
                QMessageBox.information(
                    self, "fishsuite",
                    f"No supported images found under:\n{in_dir}"
                )
                return
            self.detect_channels_btn.setEnabled(False)
            self.detected_channels_label.setText(
                f"Reading metadata from {sample.path.name}…"
            )
            QApplication.processEvents()
            info = _state.detect_channels_from_file(str(sample.path))
            self.detect_channels_btn.setEnabled(True)
            if info.get("error"):
                self.detected_channels_label.setText(
                    f"Failed to read {sample.path.name}: {info['error']}"
                )
                return
            # Render: 4-line block with ch index, name, fluorophore/emission.
            lines = [
                f"Source: {Path(info['source_file']).name}",
            ]
            vx = info.get("voxel_xy_nm")
            vz = info.get("voxel_z_nm")
            if vx or vz:
                vx_s = f"{vx:.1f} nm" if vx else "?"
                vz_s = f"{vz:.1f} nm" if vz else "?"
                lines.append(f"Voxel: XY={vx_s}  Z={vz_s}")
            for i, n in enumerate(info["names"]):
                lines.append(f"  Ch {i}: {_state.format_channel_metadata_row(n)}")
            self.detected_channels_label.setText("\n".join(lines))
            # Persist + refresh per-role resolved labels.
            self._settings["last_detected_channels"] = {
                "names": list(info["names"]),
                "source_file": info["source_file"],
                "voxel_xy_nm": info.get("voxel_xy_nm"),
                "voxel_z_nm": info.get("voxel_z_nm"),
            }
            _state.save_settings(self._capture_settings_for_save())
            self._refresh_resolved_channel_labels()
            # Auto-suggest LUT colors based on each role's emission wavelength.
            # Brian's convention (2026-05-14): 647→yellow, 561→magenta,
            # 488→green, DAPI→blue. Only suggests when the detector has a
            # confident emission match; user can override afterward.
            self._auto_suggest_luts_from_detection(info["names"])

        def _auto_suggest_luts_from_detection(self, names: List[str]) -> None:
            """Apply Brian's wavelength → LUT mapping to the channel LUT combos.

            For each role's channel spinbox, looks up the resolved channel
            name and queries ``state.suggested_lut_for_channel_name``.
            If a confident LUT is suggested, set the corresponding combo.
            """
            if not names:
                return
            mapping = [
                (self.chan_dapi, self.chan_dapi_lut),
                (self.chan_rna, self.chan_rna_lut),
                (self.chan_rna2, self.chan_rna2_lut),
                (self.chan_ab, self.chan_ab_lut),
                (self.chan_ab2, self.chan_ab2_lut),
            ]
            for sp, combo in mapping:
                idx = int(sp.value())
                if idx < 0 or idx >= len(names):
                    continue
                suggested = _state.suggested_lut_for_channel_name(names[idx])
                if suggested is None:
                    continue
                # Use findText so we only set a value the combo actually has.
                i = combo.findText(suggested, Qt.MatchFixedString)
                if i >= 0:
                    combo.setCurrentIndex(i)

        def _refresh_resolved_channel_labels(self) -> None:
            """Update the per-role italic "→ Ch N = name" labels.

            Uses the most-recently-detected channel-name list (from
            state.last_detected_channels). Safe to call before any detect
            has happened — labels just stay at their placeholder text.
            """
            det = (self._settings.get("last_detected_channels") or {})
            names: List[str] = list(det.get("names") or [])
            mapping = [
                (self.chan_dapi, self.chan_dapi_resolved),
                (self.chan_rna, self.chan_rna_resolved),
                (self.chan_rna2, self.chan_rna2_resolved),
                (self.chan_ab, self.chan_ab_resolved),
                (self.chan_ab2, self.chan_ab2_resolved),
            ]
            for sp, rl in mapping:
                idx = int(sp.value())
                if not names:
                    rl.setText("(detect to populate)")
                    continue
                if idx < 0:
                    rl.setText("→ auto-detect at runtime")
                    continue
                if idx >= len(names):
                    rl.setText(f"→ Ch {idx} (out of range: only {len(names)} channels)")
                    continue
                rl.setText(
                    f"→ Ch {idx}: {_state.format_channel_metadata_row(names[idx])}"
                )

        # ------------------------------------------------------------------
        # Tab 4 — Z-Stack
        # ------------------------------------------------------------------
        def _build_tab_zstack(self) -> None:
            w = QWidget()
            scroll = self._wrap_scroll(w)
            v = QVBoxLayout(w)
            v.setSpacing(8)
            v.setContentsMargins(16, 16, 16, 16)

            v.addWidget(SectionHeader(
                "Z-stack handling",
                "Set start/end to the IN-FOCUS slice range; autofocus then picks the best slice inside that window."
            ))
            form = QFormLayout()
            self.z_mode = QComboBox()
            self.z_mode.addItems(["autofocus", "single", "maxproj", "3d"])
            self.z_mode.setToolTip(
                "autofocus — pick the single sharpest plane within "
                "[start_slice, end_slice] (recommended).\n"
                "single — use a fixed slice (single_slice).\n"
                "maxproj — collapse the substack via max-projection (FLATTENS Z; "
                "tends to merge spots / overlapping nuclei — verify visually).\n"
                "3d — full 3D substack analysis (slow; only when you actually need "
                "z-resolved spot positions)."
            )
            self.z_mode.currentTextChanged.connect(self._on_zmode_changed)
            form.addRow("Mode", self.z_mode)

            self.z_start = QSpinBox()
            self.z_start.setRange(1, 999)
            self.z_start.setValue(5)
            self.z_start.valueChanged.connect(self._on_field_changed)
            form.addRow("Start slice (1-based)", self.z_start)

            self.z_end = QSpinBox()
            self.z_end.setRange(1, 999)
            self.z_end.setValue(15)
            self.z_end.valueChanged.connect(self._on_field_changed)
            form.addRow("End slice (1-based)", self.z_end)

            self.z_single = QSpinBox()
            self.z_single.setRange(1, 999)
            self.z_single.setValue(10)
            self.z_single.valueChanged.connect(self._on_field_changed)
            form.addRow("Single slice", self.z_single)

            self._z_start_row_label = form.labelForField(self.z_start)
            self._z_end_row_label = form.labelForField(self.z_end)
            self._z_single_row_label = form.labelForField(self.z_single)

            note = QLabel(
                "Tip: a too-wide window can pull autofocus onto an out-of-focus plane "
                "(StarDist will under-segment). H9 100x default is 5-15."
            )
            note.setObjectName("tipLabel")
            note.setWordWrap(True)
            form.addRow(note)

            v.addLayout(form)
            v.addStretch(1)
            self._refresh_zstack_visibility()
            self._register_tab("zstack", scroll, "Z-Stack")

        def _on_zmode_changed(self, *_a) -> None:
            self._refresh_zstack_visibility()
            self._on_field_changed()

        def _refresh_zstack_visibility(self) -> None:
            mode = self.z_mode.currentText()
            show_range = mode in ("autofocus", "range")
            show_single = mode == "single"
            for w, lbl in [
                (self.z_start, self._z_start_row_label),
                (self.z_end, self._z_end_row_label),
            ]:
                w.setVisible(show_range)
                if lbl is not None:
                    lbl.setVisible(show_range)
            self.z_single.setVisible(show_single)
            if self._z_single_row_label is not None:
                self._z_single_row_label.setVisible(show_single)

        # ------------------------------------------------------------------
        # Tab 5 — Nuclei segmentation
        # ------------------------------------------------------------------
        def _build_tab_nuclei(self) -> None:
            w = QWidget()
            scroll = self._wrap_scroll(w)
            v = QVBoxLayout(w)
            v.setSpacing(8)
            v.setContentsMargins(16, 16, 16, 16)

            v.addWidget(SectionHeader("Nuclear segmentation", "Backend + tuning."))
            form = QFormLayout()
            self.nuc_backend = QComboBox()
            self.nuc_backend.addItems(["stardist", "cellpose", "otsu"])
            self.nuc_backend.setToolTip(
                "stardist — recommended for fluorescent DAPI nuclei.\n"
                "cellpose — alternative DL backend (slower without CUDA; this "
                "machine has no CUDA so cpsam falls back to CPU).\n"
                "otsu — classical Otsu thresholding + watershed; fast, no DL, "
                "tends to merge touching cells."
            )
            self.nuc_backend.currentTextChanged.connect(self._on_nuc_backend_changed)
            form.addRow("Backend", self.nuc_backend)

            self.nuc_min_area = QSpinBox()
            self.nuc_min_area.setRange(1, 10_000_000)
            self.nuc_min_area.setValue(10000)
            self.nuc_min_area.setSuffix(" px")
            self.nuc_min_area.valueChanged.connect(self._on_field_changed)
            form.addRow("min_area_px", self.nuc_min_area)

            self.nuc_max_area = QDoubleSpinBox()
            self.nuc_max_area.setRange(1.0, 1e15)
            self.nuc_max_area.setDecimals(0)
            self.nuc_max_area.setValue(1e12)
            self.nuc_max_area.setSuffix(" px")
            self.nuc_max_area.valueChanged.connect(self._on_field_changed)
            form.addRow("max_area_px", self.nuc_max_area)

            self.exclude_border = QCheckBox("Exclude nuclei touching image border")
            self.exclude_border.setChecked(True)
            self.exclude_border.toggled.connect(self._on_field_changed)
            form.addRow("", self.exclude_border)

            self.border_margin = QSpinBox()
            self.border_margin.setRange(0, 200)
            self.border_margin.setValue(5)
            self.border_margin.setSuffix(" px")
            self.border_margin.valueChanged.connect(self._on_field_changed)
            form.addRow("border_margin_px", self.border_margin)
            v.addLayout(form)

            # StarDist group
            self.stardist_box = QGroupBox("StarDist options")
            sgrp = QFormLayout(self.stardist_box)
            self.sd_prob = SliderSpin(0.05, 0.95, 0.05, decimals=2, value=0.5)
            self.sd_prob.valueChanged.connect(self._on_field_changed)
            sgrp.addRow("prob_threshold", self.sd_prob)
            self.sd_nms = SliderSpin(0.05, 0.95, 0.05, decimals=2, value=0.5)
            self.sd_nms.valueChanged.connect(self._on_field_changed)
            sgrp.addRow("nms_threshold", self.sd_nms)
            # StarDist model — editable combo. Known pretrained names are
            # offered as a dropdown; custom names typed in the line edit
            # are still accepted (round-trips through YAML).
            self.sd_model = self._make_open_combo(
                items=[
                    "2D_versatile_fluo",
                    "2D_paper_dsb2018",
                    "2D_versatile_he",
                    "2D_demo",
                ],
                default="2D_versatile_fluo",
                tooltip=(
                    "Pretrained StarDist model name.\n"
                    "  • 2D_versatile_fluo — default for DAPI / fluorescent nuclei.\n"
                    "  • 2D_paper_dsb2018 — DSB2018 weights (smaller nuclei).\n"
                    "  • 2D_versatile_he — H&E histology (not relevant for FISH).\n"
                    "  • 2D_demo — small demo model, poor accuracy, fast load.\n"
                    "You can type a custom model name if you've downloaded one."
                ),
            )
            sgrp.addRow("model", self.sd_model)
            self.sd_gauss = SliderSpin(0.0, 8.0, 0.5, decimals=1, value=3.0)
            self.sd_gauss.valueChanged.connect(self._on_field_changed)
            sgrp.addRow("gauss_sigma (pre-blur)", self.sd_gauss)
            self.sd_postprocess = QComboBox()
            self.sd_postprocess.addItems(["none", "dilate", "watershed_otsu", "watershed_triangle"])
            self.sd_postprocess.setToolTip(
                "Post-processing applied to the raw StarDist instance masks.\n"
                "  • none — use StarDist output directly.\n"
                "  • dilate — radial dilation by dilate_px on every label.\n"
                "  • watershed_otsu — Otsu-thresholded DAPI watershed seeded by "
                "StarDist labels (expands each label out to the full nuclear "
                "extent; recommended at high mag where StarDist hugs the DAPI core).\n"
                "  • watershed_triangle — same idea but uses Triangle thresholding "
                "(more permissive than Otsu)."
            )
            self.sd_postprocess.currentTextChanged.connect(self._on_field_changed)
            sgrp.addRow("post-process", self.sd_postprocess)
            self.sd_dilate = QSpinBox()
            self.sd_dilate.setRange(0, 200)
            self.sd_dilate.setValue(30)
            self.sd_dilate.setSuffix(" px")
            self.sd_dilate.valueChanged.connect(self._on_field_changed)
            sgrp.addRow("dilate_px", self.sd_dilate)
            self.sd_otsu_sigma = SliderSpin(0.0, 8.0, 0.5, decimals=1, value=2.0)
            self.sd_otsu_sigma.valueChanged.connect(self._on_field_changed)
            sgrp.addRow("postprocess_otsu_sigma", self.sd_otsu_sigma)
            self.sd_close = QSpinBox()
            self.sd_close.setRange(0, 50)
            self.sd_close.setValue(5)
            self.sd_close.setSuffix(" px")
            self.sd_close.valueChanged.connect(self._on_field_changed)
            sgrp.addRow("postprocess_mask_closing_px", self.sd_close)
            v.addWidget(self.stardist_box)

            # Cellpose group
            self.cellpose_box = QGroupBox("Cellpose options")
            cgrp = QFormLayout(self.cellpose_box)
            # Cellpose model_type — editable combo. cpsam is the only built-in
            # model that ships with Cellpose 4.x; the older nuclei/cyto/cyto2/
            # cyto3 names round-trip from legacy YAMLs but require the user to
            # have the weights downloaded locally.
            self.cp_model = self._make_open_combo(
                items=["cpsam", "nuclei", "cyto", "cyto2", "cyto3"],
                default="cpsam",
                tooltip=(
                    "Cellpose model to load.\n"
                    "  • cpsam — the only built-in model in Cellpose 4.x "
                    "(~1.2 GB transformer generalist).\n"
                    "  • nuclei / cyto / cyto2 / cyto3 — legacy Cellpose 2/3 "
                    "weights. Listed here so old YAMLs round-trip, but loading "
                    "them in Cellpose 4 requires you've downloaded the file "
                    "manually. Prefer StarDist for nuclei-only segmentation."
                ),
            )
            cgrp.addRow("model_type", self.cp_model)
            self.cp_diam = QDoubleSpinBox()
            self.cp_diam.setRange(0.0, 500.0)
            self.cp_diam.setDecimals(1)
            self.cp_diam.setValue(0.0)
            self.cp_diam.setSuffix(" px (0 = auto)")
            self.cp_diam.valueChanged.connect(self._on_field_changed)
            cgrp.addRow("diameter_px", self.cp_diam)
            self.cp_flow = SliderSpin(0.0, 1.0, 0.05, decimals=2, value=0.4)
            self.cp_flow.valueChanged.connect(self._on_field_changed)
            cgrp.addRow("flow_threshold", self.cp_flow)
            self.cp_prob = SliderSpin(-6.0, 6.0, 0.1, decimals=2, value=0.0)
            self.cp_prob.valueChanged.connect(self._on_field_changed)
            cgrp.addRow("cellprob_threshold", self.cp_prob)
            v.addWidget(self.cellpose_box)

            # Otsu group
            self.otsu_box = QGroupBox("Otsu options")
            ogrp = QFormLayout(self.otsu_box)
            ogrp.addRow(QLabel("(no extra parameters — uses min_area_px above + exclude_border)"))
            v.addWidget(self.otsu_box)

            v.addStretch(1)
            self._refresh_nuc_visibility()
            self._register_tab("nuclei", scroll, "Nuclei")

        def _on_nuc_backend_changed(self, *_a) -> None:
            self._refresh_nuc_visibility()
            self._on_field_changed()

        def _refresh_nuc_visibility(self) -> None:
            b = self.nuc_backend.currentText()
            self.stardist_box.setVisible(b == "stardist")
            self.cellpose_box.setVisible(b == "cellpose")
            self.otsu_box.setVisible(b == "otsu")

        # ------------------------------------------------------------------
        # Tab 6 — Foci / Spots
        # ------------------------------------------------------------------
        def _build_tab_foci(self) -> None:
            w = QWidget()
            scroll = self._wrap_scroll(w)
            v = QVBoxLayout(w)
            v.setSpacing(8)
            v.setContentsMargins(16, 16, 16, 16)

            v.addWidget(SectionHeader("Foci / spot detection"))
            self.foci_enabled = QCheckBox("Enable spot detection")
            self.foci_enabled.setChecked(True)
            self.foci_enabled.toggled.connect(self._on_field_changed)
            v.addWidget(self.foci_enabled)

            form = QFormLayout()
            self.foci_backend = QComboBox()
            self.foci_backend.addItems(["bigfish", "log"])
            self.foci_backend.setToolTip(
                "bigfish — Big-FISH LoG-based detection with auto Otsu "
                "threshold (recommended for diffraction-limited RNA-FISH).\n"
                "log — classical scikit-image LoG blob detection (fixed "
                "threshold; legacy / Fiji-parity)."
            )
            self.foci_backend.currentTextChanged.connect(self._on_foci_backend_changed)
            form.addRow("Backend", self.foci_backend)

            self.only_nuclear = QCheckBox("Restrict to nuclear spots only")
            self.only_nuclear.toggled.connect(self._on_field_changed)
            form.addRow("", self.only_nuclear)
            v.addLayout(form)

            # BigFISH group (shared / RNA1 defaults).
            # In rna_rna mode this group provides the RNA1 settings AND the
            # defaults for RNA2 when the "Use same as RNA1" checkbox is on.
            self.bigfish_box = QGroupBox("BigFISH options (shared / RNA1)")
            bgrp = QFormLayout(self.bigfish_box)
            self.bf_vx = QDoubleSpinBox()
            self.bf_vx.setRange(0.0, 1000.0)
            self.bf_vx.setDecimals(1)
            self.bf_vx.setSuffix(" nm (0 = auto from VSI)")
            self.bf_vx.valueChanged.connect(self._on_field_changed)
            bgrp.addRow("voxel xy (nm)", self.bf_vx)
            self.bf_vz = QDoubleSpinBox()
            self.bf_vz.setRange(0.0, 5000.0)
            self.bf_vz.setDecimals(1)
            self.bf_vz.setSuffix(" nm (0 = auto)")
            self.bf_vz.valueChanged.connect(self._on_field_changed)
            bgrp.addRow("voxel z (nm)", self.bf_vz)
            self.bf_rad = QDoubleSpinBox()
            self.bf_rad.setRange(10.0, 1000.0)
            self.bf_rad.setDecimals(1)
            self.bf_rad.setValue(130.0)
            self.bf_rad.setSuffix(" nm")
            self.bf_rad.valueChanged.connect(self._on_field_changed)
            bgrp.addRow("spot_radius_nm", self.bf_rad)
            self.bf_rad_z = QDoubleSpinBox()
            self.bf_rad_z.setRange(10.0, 3000.0)
            self.bf_rad_z.setDecimals(1)
            self.bf_rad_z.setValue(300.0)
            self.bf_rad_z.setSuffix(" nm")
            self.bf_rad_z.valueChanged.connect(self._on_field_changed)
            bgrp.addRow("spot_radius_z_nm", self.bf_rad_z)
            self.bf_tm = SliderSpin(0.2, 1.5, 0.05, decimals=2, value=0.7)
            self.bf_tm.valueChanged.connect(self._on_field_changed)
            bgrp.addRow("threshold_multiplier", self.bf_tm)
            self.bf_thr_override = QDoubleSpinBox()
            self.bf_thr_override.setRange(-1.0, 1e9)
            self.bf_thr_override.setDecimals(2)
            self.bf_thr_override.setSpecialValueText("(auto)")
            self.bf_thr_override.setValue(-1.0)
            self.bf_thr_override.valueChanged.connect(self._on_field_changed)
            bgrp.addRow("threshold_override", self.bf_thr_override)
            self.bf_min_sep = QSpinBox()
            self.bf_min_sep.setRange(1, 50)
            self.bf_min_sep.setValue(1)
            self.bf_min_sep.setSuffix(" px (Fiji NMS only)")
            self.bf_min_sep.valueChanged.connect(self._on_field_changed)
            bgrp.addRow("min_sep_px", self.bf_min_sep)
            v.addWidget(self.bigfish_box)

            # ----------------------------------------------------------------
            # RNA2 overrides group — only visible when analysis_mode == rna_rna.
            # The checkbox at the top toggles "inherit from RNA1 (defaults)"
            # vs. "use my own RNA2 values". When inherited, the YAML emits an
            # EMPTY rna2_overrides block (every field is None) so the schema's
            # FociChannelOverrideCfg falls back to FociCfg defaults.
            # ----------------------------------------------------------------
            self.rna2_override_box = QGroupBox("RNA2 BigFISH overrides")
            r2grp = QFormLayout(self.rna2_override_box)
            self.rna2_use_same = QCheckBox("Use same settings as RNA1")
            self.rna2_use_same.setChecked(True)
            self.rna2_use_same.toggled.connect(self._on_rna2_use_same_toggled)
            r2grp.addRow("", self.rna2_use_same)

            self.bf_rad_r2 = QDoubleSpinBox()
            self.bf_rad_r2.setRange(10.0, 1000.0)
            self.bf_rad_r2.setDecimals(1)
            self.bf_rad_r2.setValue(130.0)
            self.bf_rad_r2.setSuffix(" nm")
            self.bf_rad_r2.valueChanged.connect(self._on_field_changed)
            r2grp.addRow("spot_radius_nm (RNA2)", self.bf_rad_r2)

            self.bf_rad_z_r2 = QDoubleSpinBox()
            self.bf_rad_z_r2.setRange(10.0, 3000.0)
            self.bf_rad_z_r2.setDecimals(1)
            self.bf_rad_z_r2.setValue(300.0)
            self.bf_rad_z_r2.setSuffix(" nm")
            self.bf_rad_z_r2.valueChanged.connect(self._on_field_changed)
            r2grp.addRow("spot_radius_z_nm (RNA2)", self.bf_rad_z_r2)

            self.bf_tm_r2 = SliderSpin(0.2, 1.5, 0.05, decimals=2, value=0.7)
            self.bf_tm_r2.valueChanged.connect(self._on_field_changed)
            r2grp.addRow("threshold_multiplier (RNA2)", self.bf_tm_r2)

            self.only_nuclear_r2 = QCheckBox("Restrict to nuclear spots (RNA2)")
            self.only_nuclear_r2.toggled.connect(self._on_field_changed)
            r2grp.addRow("", self.only_nuclear_r2)

            self.bf_min_sep_r2 = QSpinBox()
            self.bf_min_sep_r2.setRange(1, 50)
            self.bf_min_sep_r2.setValue(1)
            self.bf_min_sep_r2.setSuffix(" px (Fiji NMS only)")
            self.bf_min_sep_r2.valueChanged.connect(self._on_field_changed)
            r2grp.addRow("min_sep_px (RNA2)", self.bf_min_sep_r2)
            v.addWidget(self.rna2_override_box)

            # LoG group
            self.log_box = QGroupBox("LoG options")
            lgrp = QFormLayout(self.log_box)
            self.lg_rad = SliderSpin(0.5, 10.0, 0.1, decimals=1, value=2.5)
            self.lg_rad.valueChanged.connect(self._on_field_changed)
            lgrp.addRow("log_spot_radius_px", self.lg_rad)
            self.lg_thr = SliderSpin(0.001, 1.0, 0.005, decimals=3, value=0.05)
            self.lg_thr.valueChanged.connect(self._on_field_changed)
            lgrp.addRow("log_threshold", self.lg_thr)
            v.addWidget(self.log_box)

            v.addStretch(1)
            self._refresh_foci_visibility()
            self._register_tab("foci", scroll, "Foci / Spots")

        def _on_foci_backend_changed(self, *_a) -> None:
            self._refresh_foci_visibility()
            self._on_field_changed()

        def _on_rna2_use_same_toggled(self, *_a) -> None:
            """When 'use same as RNA1' is checked, gray out the RNA2 widgets so
            it's visually clear they will inherit. Unchecked = enabled."""
            inherit = bool(self.rna2_use_same.isChecked())
            for w in (
                self.bf_rad_r2, self.bf_rad_z_r2, self.bf_tm_r2,
                self.only_nuclear_r2, self.bf_min_sep_r2,
            ):
                w.setEnabled(not inherit)
            self._on_field_changed()

        def _refresh_foci_visibility(self) -> None:
            b = self.foci_backend.currentText()
            self.bigfish_box.setVisible(b == "bigfish")
            self.log_box.setVisible(b == "log")
            # RNA2 overrides only relevant in rna_rna mode + bigfish backend.
            try:
                mode = self.mode_combo.currentText()
            except Exception:
                mode = "rna_only"
            self.rna2_override_box.setVisible(
                (b == "bigfish") and (mode == "rna_rna")
            )

        # ------------------------------------------------------------------
        # Tab 7 — Pixel Colocalization
        # ------------------------------------------------------------------
        def _build_tab_pixel_coloc(self) -> None:
            w = QWidget()
            scroll = self._wrap_scroll(w)
            v = QVBoxLayout(w)
            v.setSpacing(8)
            v.setContentsMargins(16, 16, 16, 16)

            v.addWidget(SectionHeader(
                "Pixel-level RNA threshold",
                "Computed as median + k*MAD over RAW nuclear pixel values (no preprocessing)."
            ))
            form = QFormLayout()
            self.pc_mode = QComboBox()
            self.pc_mode.addItems(["mad", "percentile", "costes"])
            self.pc_mode.setToolTip(
                "mad — threshold = median + k_mad × MAD over nuclear pixels "
                "(robust; default k_mad = 2.0–2.5).\n"
                "percentile — threshold = the chosen percentile of nuclear "
                "pixels (e.g. 80%).\n"
                "costes — Costes' regression-based automatic threshold "
                "(no tunable knobs)."
            )
            self.pc_mode.currentTextChanged.connect(self._on_pc_mode_changed)
            form.addRow("threshold_mode", self.pc_mode)

            self.pc_scope = QComboBox()
            self.pc_scope.addItems(["batch", "per_image"])
            self.pc_scope.setToolTip(
                "batch — ONE global threshold pooled from all nuclear pixels in "
                "the run (recommended for cross-condition comparison).\n"
                "per_image — each image gets its own threshold (only for very "
                "heterogeneous batches; destroys cross-image comparability)."
            )
            self.pc_scope.currentTextChanged.connect(self._on_field_changed)
            form.addRow("threshold_scope", self.pc_scope)
            scope_note = QLabel(
                "batch = ONE global threshold for the whole run (recommended for cross-condition comparison). "
                "per_image = each image gets its own threshold (only for very heterogeneous batches)."
            )
            scope_note.setObjectName("scopeNote")
            scope_note.setWordWrap(True)
            form.addRow(scope_note)

            self.pc_kmad = SliderSpin(0.5, 6.0, 0.1, decimals=1, value=2.0)
            self.pc_kmad.valueChanged.connect(self._on_field_changed)
            self._pc_kmad_label = QLabel("k_mad")
            form.addRow(self._pc_kmad_label, self.pc_kmad)

            self.pc_pct = SliderSpin(50.0, 99.9, 0.5, decimals=1, value=80.0)
            self.pc_pct.valueChanged.connect(self._on_field_changed)
            self._pc_pct_label = QLabel("percentile")
            form.addRow(self._pc_pct_label, self.pc_pct)

            v.addLayout(form)
            v.addStretch(1)
            self._refresh_pc_visibility()
            self._register_tab("pixel_coloc", scroll, "Pixel Coloc")

        def _on_pc_mode_changed(self, *_a) -> None:
            self._refresh_pc_visibility()
            self._on_field_changed()

        def _refresh_pc_visibility(self) -> None:
            mode = self.pc_mode.currentText()
            self.pc_kmad.setVisible(mode == "mad")
            self._pc_kmad_label.setVisible(mode == "mad")
            self.pc_pct.setVisible(mode == "percentile")
            self._pc_pct_label.setVisible(mode == "percentile")

        # ------------------------------------------------------------------
        # Tab 8 — Cytoplasm & N/C
        # ------------------------------------------------------------------
        def _build_tab_cytoplasm(self) -> None:
            w = QWidget()
            scroll = self._wrap_scroll(w)
            v = QVBoxLayout(w)
            v.setSpacing(8)
            v.setContentsMargins(16, 16, 16, 16)

            v.addWidget(SectionHeader(
                "Cytoplasm / N:C ratio",
                "Cytoplasm is approximated as a Voronoi expansion around each nucleus."
            ))
            self.cyt_enabled = QCheckBox("Enable cytoplasm estimation")
            self.cyt_enabled.setChecked(True)
            self.cyt_enabled.toggled.connect(self._on_field_changed)
            v.addWidget(self.cyt_enabled)

            form = QFormLayout()
            self.cyt_vme = QSpinBox()
            self.cyt_vme.setRange(1, 1000)
            self.cyt_vme.setValue(80)
            self.cyt_vme.setSuffix(" px")
            self.cyt_vme.valueChanged.connect(self._on_field_changed)
            form.addRow("voronoi_max_expansion_px", self.cyt_vme)
            self.cyt_nc = QCheckBox("Measure nuclear/cytoplasmic ratio")
            self.cyt_nc.setChecked(True)
            self.cyt_nc.toggled.connect(self._on_field_changed)
            form.addRow("", self.cyt_nc)
            v.addLayout(form)
            v.addStretch(1)
            self._register_tab("cytoplasm", scroll, "Cytoplasm")

        # ------------------------------------------------------------------
        # Tab 9 — Output
        # ------------------------------------------------------------------
        def _build_tab_output(self) -> None:
            w = QWidget()
            scroll = self._wrap_scroll(w)
            v = QVBoxLayout(w)
            v.setSpacing(8)
            v.setContentsMargins(16, 16, 16, 16)

            v.addWidget(SectionHeader("Output writers"))
            self.o_qc = QCheckBox("Save QC overlays (PNG)")
            self.o_qc.setChecked(True)
            self.o_per_image = QCheckBox("Save per-image CSVs")
            self.o_per_image.setChecked(True)
            self.o_masks = QCheckBox("Save mask TIFFs (nuclei + spot mask + DAPI)")
            self.o_masks.setChecked(True)
            self.o_pub = QCheckBox("Save publication images (DAPI/RNA/merge in PNG + TIF)")
            self.o_pub.setChecked(True)
            for c in (self.o_qc, self.o_per_image, self.o_masks, self.o_pub):
                c.toggled.connect(self._on_field_changed)
                v.addWidget(c)

            form = QFormLayout()
            self.o_prefix = QLineEdit()
            self.o_prefix.setPlaceholderText("(blank = no prefix)")
            self.o_prefix.textChanged.connect(self._on_field_changed)
            form.addRow("Filename prefix", self.o_prefix)

            self.parallel_combo = QComboBox()
            self.parallel_combo.setEditable(True)
            self.parallel_combo.addItems(["auto", "1", "2", "4", "6", "8", "12", "16"])
            self.parallel_combo.currentTextChanged.connect(self._on_field_changed)
            form.addRow("Parallel workers", self.parallel_combo)

            self.skip_down = QCheckBox("Skip downstream figure step (analysis.single_condition_plots)")
            self.skip_down.toggled.connect(self._on_field_changed)
            form.addRow("", self.skip_down)
            v.addLayout(form)
            v.addStretch(1)
            self._register_tab("output", scroll, "Output")

        # ------------------------------------------------------------------
        # Tab 10 — Run
        # ------------------------------------------------------------------
        def _build_tab_run(self) -> None:
            w = QWidget()
            v = QVBoxLayout(w)
            v.setSpacing(8)
            v.setContentsMargins(16, 16, 16, 16)

            header = QHBoxLayout()
            self.run_status_dot = StatusDot(diameter=18)
            self.run_status_dot.set_status("yellow")
            header.addWidget(self.run_status_dot)
            self.run_status_label = QLabel("Idle")
            f = QFont()
            f.setBold(True)
            f.setPointSize(13)
            self.run_status_label.setFont(f)
            header.addWidget(self.run_status_label)
            header.addStretch(1)
            self.run_output_label = QLabel("output: (not yet set)")
            self.run_output_label.setObjectName("runOutputPath")
            header.addWidget(self.run_output_label)
            v.addLayout(header)

            # Run buttons
            btn_row = QHBoxLayout()
            self.run_btn = QPushButton("▶  Start")
            self.run_btn.setObjectName("runButton")
            self.run_btn.clicked.connect(self._on_run_clicked)
            self.run_btn.setShortcut(QKeySequence("Ctrl+R"))
            self.run_btn.setToolTip("Ctrl+R")
            btn_row.addWidget(self.run_btn)

            self.stop_btn = QPushButton("■  Stop")
            self.stop_btn.setObjectName("stopButton")
            self.stop_btn.clicked.connect(self._on_stop_clicked)
            self.stop_btn.setEnabled(False)
            self.stop_btn.setShortcut(QKeySequence("Escape"))
            self.stop_btn.setToolTip("Esc")
            btn_row.addWidget(self.stop_btn)

            self.open_btn = QPushButton("Open output dir")
            self.open_btn.clicked.connect(self._on_open_output)
            self.open_btn.setEnabled(False)
            btn_row.addWidget(self.open_btn)

            self.reveal_btn = QPushButton("Reveal in Explorer")
            self.reveal_btn.clicked.connect(self._on_reveal_output)
            self.reveal_btn.setEnabled(False)
            btn_row.addWidget(self.reveal_btn)
            btn_row.addStretch(1)
            v.addLayout(btn_row)

            # Progress bar + description
            self.prog_bar = QProgressBar()
            self.prog_bar.setRange(0, 1)
            self.prog_bar.setValue(0)
            self.prog_bar.setTextVisible(True)
            v.addWidget(self.prog_bar)
            self.prog_desc = QLabel("Waiting…")
            self.prog_desc.setObjectName("progDesc")
            v.addWidget(self.prog_desc)

            # Log
            v.addWidget(SectionHeader("Live log"))
            self.log_view = QPlainTextEdit()
            self.log_view.setObjectName("logView")
            self.log_view.setReadOnly(True)
            self.log_view.setMaximumBlockCount(20000)
            v.addWidget(self.log_view, 1)

            # Run history
            v.addWidget(SectionHeader("Recent runs"))
            self.history_list = QListWidget()
            self.history_list.setMaximumHeight(140)
            self.history_list.itemDoubleClicked.connect(self._on_history_open)
            v.addWidget(self.history_list)

            self._register_tab("run", w, "Run")
            self._refresh_history()

        # ------------------------------------------------------------------
        # Tab 11 — YAML Preview
        # ------------------------------------------------------------------
        def _build_tab_yaml(self) -> None:
            w = QWidget()
            v = QVBoxLayout(w)
            v.setSpacing(8)
            v.setContentsMargins(16, 16, 16, 16)
            v.addWidget(SectionHeader(
                "Live YAML preview",
                "This is the exact YAML written under output_dir/_run_config.yaml."
            ))
            self.yaml_view = QPlainTextEdit()
            self.yaml_view.setObjectName("yamlView")
            self.yaml_view.setReadOnly(True)
            v.addWidget(self.yaml_view, 1)

            btn_row = QHBoxLayout()
            copy_btn = QPushButton("Copy to clipboard")
            copy_btn.clicked.connect(self._copy_yaml)
            btn_row.addWidget(copy_btn)
            export_btn = QPushButton("Export YAML…")
            export_btn.clicked.connect(self._export_yaml)
            btn_row.addWidget(export_btn)
            btn_row.addStretch(1)
            v.addLayout(btn_row)
            self._register_tab("yaml", w, "YAML Preview")

        # ==================================================================
        # Utility / helpers
        # ==================================================================

        def _wrap_scroll(self, w: QWidget) -> QScrollArea:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(w)
            return scroll

        def _register_tab(self, key: str, widget: QWidget, label: str) -> None:
            idx = self._tabs.addTab(widget, label)
            self._tab_widgets[key] = widget
            self._tab_indices[key] = idx
            # Initial yellow dot.
            self._tabs.setTabIcon(idx, make_dot_icon(COLOR_YELLOW))

        def _set_tab_status(self, key: str, status: str) -> None:
            self._statuses[key] = status
            idx = self._tab_indices.get(key)
            if idx is None:
                return
            self._tabs.setTabIcon(idx, make_dot_icon(status_color(status)))

        def _wire_shortcuts(self) -> None:
            # Tab navigation Ctrl+1..9 (Qt only allows up to digit 9; we use 0 for tab 10).
            for i, (k, _label) in enumerate(_TABS):
                if i < 9:
                    seq = QKeySequence(f"Ctrl+{i+1}")
                else:
                    seq = QKeySequence("Ctrl+0")
                sc = QShortcut(seq, self)
                sc.activated.connect(lambda key=k: self._tabs.setCurrentIndex(self._tab_indices[key]))

        def _apply_qss(self, *, dark: bool) -> None:
            """Apply the chosen QSS stylesheet AND a matching QPalette.

            Setting the palette in addition to the stylesheet means widgets we
            don't explicitly QSS (file dialogs, message boxes, third-party
            popups) still pick up the correct colours rather than reverting to
            the OS default. This is what fixes the "some widgets are
            unreadable in dark mode" class of bug at its root.
            """
            qss = DARK_QSS if dark else LIGHT_QSS
            app = QApplication.instance()
            app.setStyleSheet(qss)
            self._apply_palette(dark=dark)
            # Keep the legacy boolean in settings so other code that reads
            # ``dark_mode`` (or older user setting files) still works.
            self._settings["dark_mode"] = bool(dark)
            # Re-paint everything explicitly so style changes propagate even
            # when widgets cached their previous colour scheme.
            for w in app.allWidgets():
                w.style().unpolish(w)
                w.style().polish(w)
                w.update()

        def _apply_palette(self, *, dark: bool) -> None:
            """Set the application QPalette to match the active QSS theme.

            Colour roles match the QSS so that ``QPalette``-driven widgets
            (which is most of Qt's built-ins under the hood) render
            consistently. WCAG AA contrast is satisfied for every text role.
            """
            from PySide6.QtGui import QColor as _QC
            p = QPalette()
            if dark:
                window = _QC("#0f172a"); base = _QC("#0f172a"); alt = _QC("#1f2937")
                text = _QC("#e5e7eb"); strong = _QC("#ffffff"); muted = _QC("#a3aab8")
                button = _QC("#1f2937"); button_text = _QC("#e5e7eb")
                highlight = _QC(COLOR_TEAL); hl_text = _QC("#ffffff")
                disabled = _QC("#6b7280"); link = _QC("#7dd3fc")
                tooltip_bg = _QC("#1f2937"); tooltip_fg = _QC("#e5e7eb")
            else:
                window = _QC("#fafafa"); base = _QC("#ffffff"); alt = _QC("#f3f4f6")
                text = _QC("#1f2937"); strong = _QC("#111827"); muted = _QC("#5b6470")
                button = _QC("#ffffff"); button_text = _QC("#1f2937")
                highlight = _QC(COLOR_TEAL); hl_text = _QC("#ffffff")
                disabled = _QC("#9ca3af"); link = _QC("#0a7d6b")
                tooltip_bg = _QC("#1f2937"); tooltip_fg = _QC("#f9fafb")
            for grp in (QPalette.Active, QPalette.Inactive):
                p.setColor(grp, QPalette.Window, window)
                p.setColor(grp, QPalette.WindowText, text)
                p.setColor(grp, QPalette.Base, base)
                p.setColor(grp, QPalette.AlternateBase, alt)
                p.setColor(grp, QPalette.Text, text)
                p.setColor(grp, QPalette.Button, button)
                p.setColor(grp, QPalette.ButtonText, button_text)
                p.setColor(grp, QPalette.BrightText, strong)
                p.setColor(grp, QPalette.Highlight, highlight)
                p.setColor(grp, QPalette.HighlightedText, hl_text)
                p.setColor(grp, QPalette.PlaceholderText, muted)
                p.setColor(grp, QPalette.Link, link)
                p.setColor(grp, QPalette.ToolTipBase, tooltip_bg)
                p.setColor(grp, QPalette.ToolTipText, tooltip_fg)
            # Disabled-state overrides — Qt uses these for greyed widgets.
            for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
                p.setColor(QPalette.Disabled, role, disabled)
            QApplication.instance().setPalette(p)

        # -- theme switcher logic ------------------------------------------
        def _system_is_dark(self) -> bool:
            """Heuristic: is the OS palette darker than midpoint?"""
            try:
                pal = QApplication.instance().palette()
                bg = pal.color(QPalette.Window)
                # luminance approximation
                lum = 0.2126 * bg.redF() + 0.7152 * bg.greenF() + 0.0722 * bg.blueF()
                return lum < 0.5
            except Exception:
                return False

        def _apply_theme(self, name: str) -> None:
            """Apply a theme by name: 'system' | 'light' | 'dark'."""
            if name not in _state.VALID_THEMES:
                name = "system"
            self._settings["theme"] = name
            dark = self._system_is_dark() if name == "system" else (name == "dark")
            self._apply_qss(dark=dark)

        def _sync_theme_combo(self) -> None:
            target = self._settings.get("theme", "system")
            for i in range(self._theme_combo.count()):
                if self._theme_combo.itemData(i) == target:
                    self._theme_combo.blockSignals(True)
                    self._theme_combo.setCurrentIndex(i)
                    self._theme_combo.blockSignals(False)
                    return

        # ==================================================================
        # Preset wiring
        # ==================================================================

        def _populate_preset_combo(self) -> None:
            self._preset_combo.blockSignals(True)
            try:
                self._preset_combo.clear()
                presets = _state.list_presets()
                for p in presets:
                    self._preset_combo.addItem(p.stem)
            finally:
                self._preset_combo.blockSignals(False)

        def _select_preset(self, stem: str, *, push_to_settings: bool) -> None:
            stems = [_state.list_presets()[i].stem for i in range(len(_state.list_presets()))]
            chosen = stem if stem in stems else (stems[0] if stems else "")
            if not chosen:
                return
            self._current_preset_stem = chosen
            self._preset_combo.blockSignals(True)
            self._preset_combo.setCurrentText(chosen)
            self._preset_combo.blockSignals(False)
            preset = _state.load_preset(chosen)
            base = _state.schema_defaults()
            merged = _state.merge_config(base, preset)
            self._cfg = merged
            self._cfg_to_widgets()
            if push_to_settings:
                self._settings["last_preset"] = chosen
                _state.save_settings(self._settings)

        def _on_preset_selected(self, stem: str) -> None:
            if not stem or stem == self._current_preset_stem:
                return
            self._select_preset(stem, push_to_settings=True)
            self._refresh_readiness()
            self._refresh_yaml_preview()

        def _on_load_preset_btn(self) -> None:
            self._select_preset(self._preset_combo.currentText(), push_to_settings=True)
            self._refresh_readiness()
            self._refresh_yaml_preview()
            self._toast(f"Loaded preset: {self._current_preset_stem}")

        def _on_save_preset_btn(self) -> None:
            from PySide6.QtWidgets import QInputDialog
            name, ok = QInputDialog.getText(
                self, "Save preset", "New preset name (alphanumerics / _ - .):",
                text=f"{self._current_preset_stem}_custom",
            )
            if not ok or not name.strip():
                return
            try:
                cfg = self._read_widgets_into_cfg()
                path = _state.save_preset_as(name.strip(), cfg)
            except Exception as e:
                QMessageBox.critical(self, "fishsuite", f"Could not save preset:\n{e}")
                return
            self._populate_preset_combo()
            self._preset_combo.setCurrentText(path.stem)
            self._current_preset_stem = path.stem
            self._toast(f"Saved preset: {path}")

        def _on_apply_overlay(self) -> None:
            name = self._overlay_combo.currentText()
            overlay = TUNING_OVERLAYS.get(name, {})
            if not overlay:
                self._toast("No overlay applied (none selected)")
                return
            cur = self._read_widgets_into_cfg()
            merged = _state.merge_config(cur, overlay)
            self._cfg = merged
            self._cfg_to_widgets()
            self._refresh_readiness()
            self._refresh_yaml_preview()
            self._toast(f"Applied overlay: {name}")

        # ==================================================================
        # Widget <-> cfg dict marshalling
        # ==================================================================

        def _cfg_to_widgets(self) -> None:
            """Populate every widget from self._cfg.

            Sets the ``_loading_widgets`` flag for the duration so any
            valueChanged signals fired mid-load are ignored — otherwise the
            first slider that updates would snapshot the half-loaded widget
            state into self._cfg, clobbering the fields we haven't gotten
            to yet (was silently zeroing dapi/rna channels back to -1).
            """
            self._loading_widgets = True
            try:
                self._cfg_to_widgets_impl()
            finally:
                self._loading_widgets = False
            # Re-snapshot the widget state back into self._cfg so the two
            # are consistent (any defaults the schema added on the merge
            # path are now reflected in widgets and vice versa), then do
            # ONE consolidated refresh.
            self._cfg = self._read_widgets_into_cfg()
            self._refresh_readiness()
            self._refresh_output_preview()
            self._refresh_channel_preview()
            if hasattr(self, "_yaml_timer"):
                self._yaml_timer.start()

        def _cfg_to_widgets_impl(self) -> None:
            c = self._cfg
            # ---- experiment ----
            e = c.get("experiment", {})
            for w_, k in [
                (self.exp_name, "name"), (self.exp_desc, "description"),
                (self.exp_cell, "cell_line"), (self.exp_date, "date"),
                (self.exp_analyst, "analyst"),
            ]:
                w_.blockSignals(True)
                w_.setText(str(e.get(k, "") or ""))
                w_.blockSignals(False)
            # ---- conditions ----
            co = c.get("conditions", {})
            self.cond_mode.blockSignals(True)
            self.cond_mode.setCurrentText(str(co.get("mode", "subfolders")))
            self.cond_mode.blockSignals(False)
            self.min_nuc.blockSignals(True)
            self.min_nuc.setValue(int(co.get("min_nuclei_for_stats", 6)))
            self.min_nuc.blockSignals(False)
            self.subf_table.load_dict(co.get("subfolder_conditions") or {})
            self.sec_list.blockSignals(True)
            self.sec_list.clear()
            for f in co.get("sec_only_folders") or []:
                self.sec_list.addItem(str(f))
            self.sec_list.blockSignals(False)
            self.sec_files_list.blockSignals(True)
            self.sec_files_list.clear()
            for f in co.get("sec_only_files") or []:
                self.sec_files_list.addItem(str(f))
            self.sec_files_list.blockSignals(False)
            self.order_list.set_items(list(co.get("condition_order") or []))
            self._refresh_conditions_visibility()
            # ---- channels ----
            ch = c.get("channels", {})
            self.mode_combo.blockSignals(True)
            self.mode_combo.setCurrentText(str(ch.get("analysis_mode", "rna_only")))
            self.mode_combo.blockSignals(False)
            self.one_indexed_chk.blockSignals(True)
            self.one_indexed_chk.setChecked(bool(ch.get("one_indexed", False)))
            self.one_indexed_chk.blockSignals(False)
            for sp, key in [
                (self.chan_dapi, "dapi"), (self.chan_rna, "rna"),
                (self.chan_rna2, "rna2"), (self.chan_ab, "antibody"),
                (self.chan_ab2, "antibody2"),
            ]:
                sp.blockSignals(True)
                sp.setValue(int(ch.get(key, -1)))
                sp.blockSignals(False)
            # Channel labels (free text). Defaults from the schema match
            # the role names ("DAPI", "RNA1", ...) — show those as
            # placeholders rather than baking them into the field so a
            # blank YAML stays blank when the user just opens the GUI.
            _label_defaults = {
                "dapi_label": "DAPI", "rna_label": "RNA1",
                "rna2_label": "RNA2", "antibody_label": "Protein",
                "ab2_label": "Protein2",
            }
            for le, key in [
                (self.chan_dapi_label, "dapi_label"),
                (self.chan_rna_label, "rna_label"),
                (self.chan_rna2_label, "rna2_label"),
                (self.chan_ab_label, "antibody_label"),
                (self.chan_ab2_label, "ab2_label"),
            ]:
                le.blockSignals(True)
                val = str(ch.get(key, "") or "")
                # If the YAML stored the schema default verbatim, treat it
                # as "use the default" so the field shows the placeholder
                # rather than a redundant hard-coded value.
                if val == _label_defaults[key]:
                    le.setText("")
                else:
                    le.setText(val)
                le.blockSignals(False)
            # Channel LUT combos. Schema defaults are blue/yellow/cyan/magenta/green.
            _lut_defaults = {
                "dapi_lut": "blue", "rna_lut": "yellow",
                "rna2_lut": "cyan", "antibody_lut": "magenta",
                "ab2_lut": "green",
            }
            for cb, key in [
                (self.chan_dapi_lut, "dapi_lut"),
                (self.chan_rna_lut, "rna_lut"),
                (self.chan_rna2_lut, "rna2_lut"),
                (self.chan_ab_lut, "antibody_lut"),
                (self.chan_ab2_lut, "ab2_lut"),
            ]:
                cb.blockSignals(True)
                val = str(ch.get(key, _lut_defaults[key]) or _lut_defaults[key]).lower()
                ix = cb.findText(val)
                cb.setCurrentIndex(ix if ix >= 0 else cb.findText(_lut_defaults[key]))
                cb.blockSignals(False)
            self._refresh_channel_visibility()
            self._refresh_channel_preview()
            # Refresh the per-role "Detected name" labels — safe before any
            # detect has run (just shows the placeholder).
            try:
                self._refresh_resolved_channel_labels()
            except AttributeError:
                pass
            # ---- z-stack ----
            z = c.get("z_stack", {})
            self.z_mode.blockSignals(True)
            self.z_mode.setCurrentText(str(z.get("mode", "autofocus")))
            self.z_mode.blockSignals(False)
            self.z_start.blockSignals(True)
            self.z_start.setValue(int(z.get("start_slice") or 5))
            self.z_start.blockSignals(False)
            self.z_end.blockSignals(True)
            self.z_end.setValue(int(z.get("end_slice") or 15))
            self.z_end.blockSignals(False)
            self.z_single.blockSignals(True)
            self.z_single.setValue(int(z.get("single_slice") or 10))
            self.z_single.blockSignals(False)
            self._refresh_zstack_visibility()
            # ---- nuclei ----
            n = c.get("nuclei", {})
            self.nuc_backend.blockSignals(True)
            self.nuc_backend.setCurrentText(str(n.get("backend", "stardist")))
            self.nuc_backend.blockSignals(False)
            self.nuc_min_area.blockSignals(True)
            self.nuc_min_area.setValue(int(n.get("min_area_px", 10000)))
            self.nuc_min_area.blockSignals(False)
            self.nuc_max_area.blockSignals(True)
            self.nuc_max_area.setValue(float(n.get("max_area_px", 1e12)))
            self.nuc_max_area.blockSignals(False)
            self.exclude_border.blockSignals(True)
            self.exclude_border.setChecked(bool(n.get("exclude_border", True)))
            self.exclude_border.blockSignals(False)
            self.border_margin.blockSignals(True)
            self.border_margin.setValue(int(n.get("border_margin_px", 5)))
            self.border_margin.blockSignals(False)
            self.sd_prob.setValue(float(n.get("prob_threshold", 0.5)))
            self.sd_nms.setValue(float(n.get("nms_threshold", 0.5)))
            self._set_open_combo(
                self.sd_model, str(n.get("stardist_model", "2D_versatile_fluo")),
            )
            self.sd_gauss.setValue(float(n.get("stardist_gauss_sigma", 3.0)))
            self.sd_postprocess.blockSignals(True)
            self.sd_postprocess.setCurrentText(str(n.get("stardist_postprocess", "watershed_otsu")))
            self.sd_postprocess.blockSignals(False)
            self.sd_dilate.blockSignals(True)
            self.sd_dilate.setValue(int(n.get("stardist_postprocess_dilate_px", 30)))
            self.sd_dilate.blockSignals(False)
            self.sd_otsu_sigma.setValue(float(n.get("stardist_postprocess_otsu_sigma", 2.0)))
            self.sd_close.blockSignals(True)
            self.sd_close.setValue(int(n.get("stardist_postprocess_mask_closing_px", 5)))
            self.sd_close.blockSignals(False)
            self._set_open_combo(
                self.cp_model, str(n.get("cellpose_model_type", "cpsam")),
            )
            self.cp_diam.blockSignals(True)
            self.cp_diam.setValue(float(n.get("cellpose_diameter_px", 0.0)))
            self.cp_diam.blockSignals(False)
            self.cp_flow.setValue(float(n.get("cellpose_flow_threshold", 0.4)))
            self.cp_prob.setValue(float(n.get("cellpose_cellprob_threshold", 0.0)))
            self._refresh_nuc_visibility()
            # ---- foci ----
            fc = c.get("foci", {})
            self.foci_enabled.blockSignals(True)
            self.foci_enabled.setChecked(bool(fc.get("enabled", True)))
            self.foci_enabled.blockSignals(False)
            self.foci_backend.blockSignals(True)
            self.foci_backend.setCurrentText(str(fc.get("backend", "bigfish")))
            self.foci_backend.blockSignals(False)
            self.only_nuclear.blockSignals(True)
            self.only_nuclear.setChecked(bool(fc.get("only_nuclear_spots", False)))
            self.only_nuclear.blockSignals(False)
            self.bf_vx.blockSignals(True)
            self.bf_vx.setValue(float(fc.get("bigfish_voxel_size_nm", 0.0)))
            self.bf_vx.blockSignals(False)
            self.bf_vz.blockSignals(True)
            self.bf_vz.setValue(float(fc.get("bigfish_voxel_z_nm", 0.0)))
            self.bf_vz.blockSignals(False)
            self.bf_rad.blockSignals(True)
            self.bf_rad.setValue(float(fc.get("bigfish_spot_radius_nm", 130.0)))
            self.bf_rad.blockSignals(False)
            self.bf_rad_z.blockSignals(True)
            self.bf_rad_z.setValue(float(fc.get("bigfish_spot_radius_z_nm", 300.0)))
            self.bf_rad_z.blockSignals(False)
            self.bf_tm.setValue(float(fc.get("threshold_multiplier", 0.7)))
            self.bf_thr_override.blockSignals(True)
            ov = fc.get("threshold_override")
            self.bf_thr_override.setValue(float(ov) if ov is not None else -1.0)
            self.bf_thr_override.blockSignals(False)
            self.lg_rad.setValue(float(fc.get("log_spot_radius_px", 2.5)))
            self.lg_thr.setValue(float(fc.get("log_threshold", 0.05)))
            # Shared min_sep_px (used by Fiji NMS; round-trips on fishsuite side)
            try:
                self.bf_min_sep.blockSignals(True)
                self.bf_min_sep.setValue(int(fc.get("min_sep_px", 1)))
                self.bf_min_sep.blockSignals(False)
            except Exception:
                pass
            # ---- foci.rna2_overrides ----
            r2 = fc.get("rna2_overrides", {}) or {}
            r2_has_any = any(
                r2.get(k) is not None for k in (
                    "bigfish_spot_radius_nm",
                    "bigfish_spot_radius_z_nm",
                    "threshold_multiplier",
                    "only_nuclear_spots",
                    "min_sep_px",
                )
            )
            self.rna2_use_same.blockSignals(True)
            self.rna2_use_same.setChecked(not r2_has_any)
            self.rna2_use_same.blockSignals(False)
            # Populate the RNA2 widgets — when r2 omits a field, fall back to
            # the shared FociCfg value so unchecking "use same" reveals the
            # current effective settings rather than zeros.
            self.bf_rad_r2.blockSignals(True)
            self.bf_rad_r2.setValue(float(
                r2.get("bigfish_spot_radius_nm")
                if r2.get("bigfish_spot_radius_nm") is not None
                else fc.get("bigfish_spot_radius_nm", 130.0)
            ))
            self.bf_rad_r2.blockSignals(False)
            self.bf_rad_z_r2.blockSignals(True)
            self.bf_rad_z_r2.setValue(float(
                r2.get("bigfish_spot_radius_z_nm")
                if r2.get("bigfish_spot_radius_z_nm") is not None
                else fc.get("bigfish_spot_radius_z_nm", 300.0)
            ))
            self.bf_rad_z_r2.blockSignals(False)
            self.bf_tm_r2.setValue(float(
                r2.get("threshold_multiplier")
                if r2.get("threshold_multiplier") is not None
                else fc.get("threshold_multiplier", 0.7)
            ))
            self.only_nuclear_r2.blockSignals(True)
            self.only_nuclear_r2.setChecked(bool(
                r2.get("only_nuclear_spots")
                if r2.get("only_nuclear_spots") is not None
                else fc.get("only_nuclear_spots", False)
            ))
            self.only_nuclear_r2.blockSignals(False)
            self.bf_min_sep_r2.blockSignals(True)
            self.bf_min_sep_r2.setValue(int(
                r2.get("min_sep_px")
                if r2.get("min_sep_px") is not None
                else fc.get("min_sep_px", 1)
            ))
            self.bf_min_sep_r2.blockSignals(False)
            self._on_rna2_use_same_toggled()
            self._refresh_foci_visibility()
            # ---- pixel coloc ----
            pc = c.get("pixel_coloc", {})
            self.pc_mode.blockSignals(True)
            self.pc_mode.setCurrentText(str(pc.get("threshold_mode", "mad")))
            self.pc_mode.blockSignals(False)
            self.pc_scope.blockSignals(True)
            self.pc_scope.setCurrentText(str(pc.get("threshold_scope", "batch")))
            self.pc_scope.blockSignals(False)
            self.pc_kmad.setValue(float(pc.get("k_mad", 2.0)))
            self.pc_pct.setValue(float(pc.get("percentile", 80.0)))
            self._refresh_pc_visibility()
            # ---- cytoplasm ----
            cy = c.get("cytoplasm", {})
            self.cyt_enabled.blockSignals(True)
            self.cyt_enabled.setChecked(bool(cy.get("enabled", True)))
            self.cyt_enabled.blockSignals(False)
            self.cyt_vme.blockSignals(True)
            self.cyt_vme.setValue(int(cy.get("voronoi_max_expansion_px", 80)))
            self.cyt_vme.blockSignals(False)
            self.cyt_nc.blockSignals(True)
            self.cyt_nc.setChecked(bool(cy.get("measure_nc_ratio", True)))
            self.cyt_nc.blockSignals(False)
            # ---- output ----
            o = c.get("output", {})
            self.o_qc.blockSignals(True); self.o_qc.setChecked(bool(o.get("save_qc_overlays", True))); self.o_qc.blockSignals(False)
            self.o_per_image.blockSignals(True); self.o_per_image.setChecked(bool(o.get("save_per_image_csv", True))); self.o_per_image.blockSignals(False)
            self.o_masks.blockSignals(True); self.o_masks.setChecked(bool(o.get("save_masks", True))); self.o_masks.blockSignals(False)
            self.o_pub.blockSignals(True); self.o_pub.setChecked(bool(o.get("save_publication_images", True))); self.o_pub.blockSignals(False)
            self.o_prefix.blockSignals(True); self.o_prefix.setText(str(o.get("prefix", "") or "")); self.o_prefix.blockSignals(False)
            par = c.get("parallel", {})
            self.parallel_combo.blockSignals(True)
            self.parallel_combo.setCurrentText(str(par.get("workers", "auto")))
            self.parallel_combo.blockSignals(False)
            # ---- input_file_subset (Improvement 2 round-trip) ----
            # If a YAML carries an explicit subset, apply it to the tree by
            # syncing it into the per-input-dir saved-subset cache so the
            # next refresh shows the right check-states.
            subset_from_cfg = list(c.get("input_file_subset") or [])
            in_dir = self.input_edit.text().strip() if hasattr(self, "input_edit") else ""
            if subset_from_cfg and in_dir:
                sub_map = dict(self._settings.get("file_subset_by_input") or {})
                sub_map[in_dir] = subset_from_cfg
                self._settings["file_subset_by_input"] = sub_map
                # If the tree was already built, re-apply check states.
                if hasattr(self, "file_tree") and self.file_tree.topLevelItemCount() > 0:
                    self._rebuild_file_tree()

        def _read_widgets_into_cfg(self) -> Dict[str, Any]:
            """Read every widget and return a fresh config dict."""
            base = _state.schema_defaults()
            base["experiment"] = {
                "name": self.exp_name.text(),
                "description": self.exp_desc.text(),
                "cell_line": self.exp_cell.text(),
                "date": self.exp_date.text(),
                "analyst": self.exp_analyst.text(),
            }
            base["conditions"] = {
                "mode": self.cond_mode.currentText(),
                "subfolder_conditions": self.subf_table.to_dict(),
                "sec_only_folders": [self.sec_list.item(i).text()
                                     for i in range(self.sec_list.count())],
                "sec_only_files": [self.sec_files_list.item(i).text()
                                   for i in range(self.sec_files_list.count())],
                "condition_order": self.order_list.get_items(),
                "min_nuclei_for_stats": int(self.min_nuc.value()),
            }
            # Channel labels: blank text = use the schema default. We store
            # the placeholder default verbatim when the field is empty so a
            # round-trip through YAML doesn't lose the field entirely.
            _label_defaults = {
                "dapi_label": "DAPI", "rna_label": "RNA1",
                "rna2_label": "RNA2", "antibody_label": "Protein",
                "ab2_label": "Protein2",
            }
            def _read_label(le: QLineEdit, key: str) -> str:
                t = le.text().strip()
                return t if t else _label_defaults[key]
            base["channels"] = {
                "analysis_mode": self.mode_combo.currentText(),
                "dapi": int(self.chan_dapi.value()),
                "rna": int(self.chan_rna.value()),
                "rna2": int(self.chan_rna2.value()),
                "antibody": int(self.chan_ab.value()),
                "antibody2": int(self.chan_ab2.value()),
                "one_indexed": bool(self.one_indexed_chk.isChecked()),
                "dapi_label": _read_label(self.chan_dapi_label, "dapi_label"),
                "rna_label": _read_label(self.chan_rna_label, "rna_label"),
                "rna2_label": _read_label(self.chan_rna2_label, "rna2_label"),
                "antibody_label": _read_label(self.chan_ab_label, "antibody_label"),
                "ab2_label": _read_label(self.chan_ab2_label, "ab2_label"),
                # Per-channel LUT selection (pseudo-color used in pub images).
                # Defaults match the historical hard-coded colors so a config
                # that omits *_lut renders byte-identically.
                "dapi_lut": str(self.chan_dapi_lut.currentText() or "blue"),
                "rna_lut": str(self.chan_rna_lut.currentText() or "yellow"),
                "rna2_lut": str(self.chan_rna2_lut.currentText() or "cyan"),
                "antibody_lut": str(self.chan_ab_lut.currentText() or "magenta"),
                "ab2_lut": str(self.chan_ab2_lut.currentText() or "green"),
            }
            base["z_stack"] = {
                "mode": self.z_mode.currentText(),
                "single_slice": int(self.z_single.value()),
                "start_slice": int(self.z_start.value()),
                "end_slice": int(self.z_end.value()),
            }
            base["nuclei"] = {
                "backend": self.nuc_backend.currentText(),
                "prob_threshold": float(self.sd_prob.value()),
                "nms_threshold": float(self.sd_nms.value()),
                "n_tiles": None,
                "stardist_model": self.sd_model.currentText().strip() or "2D_versatile_fluo",
                "stardist_gauss_sigma": float(self.sd_gauss.value()),
                "stardist_postprocess": self.sd_postprocess.currentText(),
                "stardist_postprocess_dilate_px": int(self.sd_dilate.value()),
                "stardist_postprocess_otsu_sigma": float(self.sd_otsu_sigma.value()),
                "stardist_postprocess_mask_closing_px": int(self.sd_close.value()),
                "min_area_px": int(self.nuc_min_area.value()),
                "max_area_px": float(self.nuc_max_area.value()),
                "cellpose_diameter_px": float(self.cp_diam.value()),
                "cellpose_flow_threshold": float(self.cp_flow.value()),
                "cellpose_cellprob_threshold": float(self.cp_prob.value()),
                "cellpose_model_type": self.cp_model.currentText().strip() or "cpsam",
                "exclude_border": bool(self.exclude_border.isChecked()),
                "border_margin_px": int(self.border_margin.value()),
            }
            base["pixel_coloc"] = {
                "threshold_mode": self.pc_mode.currentText(),
                "threshold_scope": self.pc_scope.currentText(),
                "k_mad": float(self.pc_kmad.value()),
                "percentile": float(self.pc_pct.value()),
            }
            override = float(self.bf_thr_override.value())
            # RNA2 overrides: emit explicit numeric values when the user
            # unchecked "Use same as RNA1"; emit an empty dict otherwise so
            # the schema's FociChannelOverrideCfg falls back to FociCfg.
            if bool(self.rna2_use_same.isChecked()):
                rna2_overrides = {
                    "bigfish_spot_radius_nm": None,
                    "bigfish_spot_radius_z_nm": None,
                    "threshold_multiplier": None,
                    "only_nuclear_spots": None,
                    "min_sep_px": None,
                }
            else:
                rna2_overrides = {
                    "bigfish_spot_radius_nm": float(self.bf_rad_r2.value()),
                    "bigfish_spot_radius_z_nm": float(self.bf_rad_z_r2.value()),
                    "threshold_multiplier": float(self.bf_tm_r2.value()),
                    "only_nuclear_spots": bool(self.only_nuclear_r2.isChecked()),
                    "min_sep_px": int(self.bf_min_sep_r2.value()),
                }
            base["foci"] = {
                "enabled": bool(self.foci_enabled.isChecked()),
                "backend": self.foci_backend.currentText(),
                "bigfish_voxel_size_nm": float(self.bf_vx.value()),
                "bigfish_voxel_z_nm": float(self.bf_vz.value()),
                "bigfish_spot_radius_nm": float(self.bf_rad.value()),
                "bigfish_spot_radius_z_nm": float(self.bf_rad_z.value()),
                "threshold_multiplier": float(self.bf_tm.value()),
                "threshold_override": None if override < 0 else override,
                "log_spot_radius_px": float(self.lg_rad.value()),
                "log_threshold": float(self.lg_thr.value()),
                "only_nuclear_spots": bool(self.only_nuclear.isChecked()),
                "min_sep_px": int(self.bf_min_sep.value()),
                "rna2_overrides": rna2_overrides,
            }
            base["cytoplasm"] = {
                "enabled": bool(self.cyt_enabled.isChecked()),
                "voronoi_max_expansion_px": int(self.cyt_vme.value()),
                "measure_nc_ratio": bool(self.cyt_nc.isChecked()),
            }
            base["output"] = {
                "save_qc_overlays": bool(self.o_qc.isChecked()),
                "save_per_image_csv": bool(self.o_per_image.isChecked()),
                "save_masks": bool(self.o_masks.isChecked()),
                "save_publication_images": bool(self.o_pub.isChecked()),
                "prefix": self.o_prefix.text(),
            }
            workers_raw = self.parallel_combo.currentText().strip()
            if workers_raw.lower() == "auto":
                workers: Any = "auto"
            else:
                try:
                    workers = int(workers_raw)
                except ValueError:
                    workers = "auto"
            base["parallel"] = {"workers": workers}
            # Per-file subset (Improvement 2). We keep the runtime tree as the
            # source of truth and snapshot into cfg here; if the tree hasn't
            # been built yet (e.g. headless), fall back to whatever was
            # previously stored under self._cfg.
            existing_subset = list(self._cfg.get("input_file_subset") or [])
            base["input_file_subset"] = existing_subset
            return base

        # ==================================================================
        # Change handlers
        # ==================================================================

        def _on_field_changed(self, *_a) -> None:
            if self._loading_widgets:
                return
            self._cfg = self._read_widgets_into_cfg()
            self._refresh_readiness()
            self._refresh_output_preview()
            self._refresh_channel_preview()
            # Schedule a YAML refresh (debounced).
            if hasattr(self, "_yaml_timer"):
                self._yaml_timer.start()

        def _refresh_readiness(self) -> None:
            statuses = _ready.evaluate_all(
                self._cfg,
                input_dir=self.input_edit.text(),
                output_base=self.output_base_edit.text(),
                run_tag=self.tag_edit.text(),
            )
            for k, st in statuses.items():
                self._set_tab_status(k, st)
            ok = _ready.overall_ready(statuses)
            self.run_btn.setEnabled(ok and not self._runner.is_running())
            if ok:
                self._footer_status.setText("Ready to run")
                self._set_footer_state("ok")
            else:
                self._footer_status.setText("Fix red tabs before running")
                self._set_footer_state("bad")
            # Update the run-tab dot color directly too.
            self.run_status_dot.set_status(statuses.get("run", "yellow"))

        def _refresh_output_preview(self) -> None:
            base = self.output_base_edit.text() or str(_state.DEFAULT_OUTPUT_BASE)
            tag = self.tag_edit.text() or "run"
            preview = _state.compute_output_dir(base, tag)
            self.output_preview.setText(str(preview))
            self.run_output_label.setText(f"output: {preview}")
            self._footer_output.setText(f"output: {preview}")

        def _refresh_channel_preview(self) -> None:
            mode = self.mode_combo.currentText()
            # Use the user-typed labels (or the placeholder defaults when
            # blank) so the live preview reads e.g. "Ch 0 = MIAT-Cy5"
            # rather than the generic role name.
            def _eff(le: QLineEdit, default: str) -> str:
                t = le.text().strip()
                return t if t else default
            display = {
                "DAPI": _eff(self.chan_dapi_label, "DAPI"),
                "RNA": _eff(self.chan_rna_label, "RNA1"),
                "RNA #2": _eff(self.chan_rna2_label, "RNA2"),
                "Antibody": _eff(self.chan_ab_label, "Protein"),
                "Antibody #2": _eff(self.chan_ab2_label, "Protein2"),
            }
            indices = {
                "DAPI": int(self.chan_dapi.value()),
                "RNA": int(self.chan_rna.value()),
                "RNA #2": int(self.chan_rna2.value()),
                "Antibody": int(self.chan_ab.value()),
                "Antibody #2": int(self.chan_ab2.value()),
            }
            keep = {
                "rna_only": ["DAPI", "RNA"],
                "rna_protein": ["DAPI", "RNA", "Antibody"],
                "rna_rna": ["DAPI", "RNA", "RNA #2"],
                "ab_ab": ["DAPI", "Antibody", "Antibody #2"],
                "protein_only": ["DAPI", "Antibody"],
                "pub_images": ["DAPI", "RNA", "RNA #2", "Antibody", "Antibody #2"],
            }.get(mode, [])
            parts = []
            for name in keep:
                v = indices[name]
                lbl = display[name]
                parts.append(f"Ch {v if v >= 0 else 'auto'} = {lbl}")
            base_idx = " | ".join(parts) if parts else "(no channels for this mode)"
            mode_note = f"analysis_mode = {mode}\n{base_idx}"
            self.chan_preview.setText(mode_note)

        # ==================================================================
        # File-selection tree (Improvement 2)
        # ==================================================================

        def _on_file_tree_refresh(self) -> None:
            """Rescan the current input dir and rebuild the tree."""
            in_dir = self.input_edit.text().strip()
            if not in_dir or not Path(in_dir).is_dir():
                self.file_tree.clear()
                self.file_tree_summary.setText(
                    "(set Input directory above, then click Refresh)"
                )
                self._file_tree_data = {}
                return
            data = _state.scan_input_dir_tree(in_dir)
            self._file_tree_data = data
            self._rebuild_file_tree()

        def _rebuild_file_tree(self) -> None:
            """Populate self.file_tree from self._file_tree_data + saved subset."""
            data = self._file_tree_data
            self._file_tree_loading = True
            try:
                self.file_tree.clear()
                in_dir = self.input_edit.text().strip()
                saved_map = (self._settings.get("file_subset_by_input") or {})
                saved_subset = list(saved_map.get(in_dir) or [])
                # Empty saved subset = include all (legacy back-compat). Otherwise
                # check just the items in the saved subset.
                saved_set = set(saved_subset)
                check_all = (len(saved_set) == 0)

                subfolder_conditions = (
                    self._cfg.get("conditions", {}).get("subfolder_conditions") or {}
                )
                sec_files = set(
                    self._cfg.get("conditions", {}).get("sec_only_files") or []
                )
                sec_folders = set(
                    self._cfg.get("conditions", {}).get("sec_only_folders") or []
                )

                def _add_leaf(parent_item: Optional[QTreeWidgetItem], finfo: Dict[str, str],
                              subfolder_name: str) -> QTreeWidgetItem:
                    # Resolve the file's condition + sec-only flag.
                    cond = subfolder_conditions.get(subfolder_name, subfolder_name)
                    is_sec = (
                        finfo["name"] in sec_files
                        or subfolder_name in sec_folders
                    )
                    label_cond = cond or "(unassigned)"
                    sec_label = "yes" if is_sec else ""
                    leaf = QTreeWidgetItem([finfo["name"], label_cond, sec_label])
                    leaf.setData(0, Qt.UserRole, {
                        "rel": finfo["rel"],
                        "name": finfo["name"],
                        "subfolder": subfolder_name,
                        "is_sec": is_sec,
                    })
                    leaf.setFlags(leaf.flags() | Qt.ItemIsUserCheckable)
                    checked = check_all or (finfo["rel"] in saved_set) or (finfo["name"] in saved_set)
                    leaf.setCheckState(0, Qt.Checked if checked else Qt.Unchecked)
                    if parent_item is not None:
                        parent_item.addChild(leaf)
                    else:
                        self.file_tree.addTopLevelItem(leaf)
                    self._apply_leaf_visual(leaf)
                    return leaf

                # Subfolders (subfolder layout).
                for sub in data.get("subfolders", []):
                    sub_name = sub["name"]
                    files = sub.get("files", [])
                    cond = subfolder_conditions.get(sub_name, sub_name)
                    parent = QTreeWidgetItem([
                        f"{sub_name}  ({len(files)} files)",
                        cond or "(unassigned)",
                        "",
                    ])
                    parent.setData(0, Qt.UserRole, {
                        "is_folder": True,
                        "subfolder": sub_name,
                    })
                    # Folder parents are tristate so checking the parent toggles
                    # all leaves under it.
                    parent.setFlags(
                        parent.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsAutoTristate
                    )
                    self.file_tree.addTopLevelItem(parent)
                    for finfo in files:
                        _add_leaf(parent, finfo, sub_name)
                    parent.setExpanded(True)

                # Root-level files (flat layout).
                root_files = data.get("root_files", [])
                if root_files:
                    if data.get("subfolders"):
                        root_parent = QTreeWidgetItem([
                            f"(root, {len(root_files)} files)",
                            "",
                            "",
                        ])
                        root_parent.setFlags(
                            root_parent.flags() | Qt.ItemIsUserCheckable
                            | Qt.ItemIsAutoTristate
                        )
                        root_parent.setData(0, Qt.UserRole, {"is_folder": True,
                                                             "subfolder": ""})
                        self.file_tree.addTopLevelItem(root_parent)
                        for finfo in root_files:
                            _add_leaf(root_parent, finfo, "")
                        root_parent.setExpanded(True)
                    else:
                        for finfo in root_files:
                            _add_leaf(None, finfo, "")

                self.file_tree.resizeColumnToContents(0)
                self.file_tree.resizeColumnToContents(1)
                self.file_tree.resizeColumnToContents(2)
            finally:
                self._file_tree_loading = False
            self._refresh_file_tree_summary()
            self._apply_file_tree_filter()

        def _iter_leaf_items(self) -> List[QTreeWidgetItem]:
            """Return every leaf (file) item in the tree."""
            out: List[QTreeWidgetItem] = []
            root = self.file_tree.invisibleRootItem()
            stack: List[QTreeWidgetItem] = [root.child(i) for i in range(root.childCount())]
            while stack:
                it = stack.pop()
                if it is None:
                    continue
                data = it.data(0, Qt.UserRole) or {}
                if data.get("is_folder"):
                    for ci in range(it.childCount()):
                        stack.append(it.child(ci))
                else:
                    out.append(it)
            return out

        def _apply_leaf_visual(self, leaf: QTreeWidgetItem) -> None:
            """Dim the leaf row's text when unchecked, normal when checked.

            Implements the 'visual distinction' amendment: unchecked rows
            are unmistakably grey so the include/exclude state is readable
            at a glance.
            """
            checked = leaf.checkState(0) == Qt.Checked
            from PySide6.QtGui import QBrush, QColor
            if checked:
                # Clear any custom foreground — use the palette default.
                brush = QBrush(self.palette().color(QPalette.Text))
            else:
                brush = QBrush(QColor("#9aa0a6"))  # COLOR_GREY-ish, muted
            for col in range(self.file_tree.columnCount()):
                leaf.setForeground(col, brush)

        def _on_file_tree_item_changed(self, item: QTreeWidgetItem, _col: int) -> None:
            if self._file_tree_loading:
                return
            data = item.data(0, Qt.UserRole) or {}
            if not data.get("is_folder"):
                self._apply_leaf_visual(item)
            self._persist_file_subset()
            self._refresh_file_tree_summary()

        def _set_all_file_tree_checked(self, checked: bool) -> None:
            self._file_tree_loading = True
            try:
                for leaf in self._iter_leaf_items():
                    leaf.setCheckState(0, Qt.Checked if checked else Qt.Unchecked)
                    self._apply_leaf_visual(leaf)
            finally:
                self._file_tree_loading = False
            self._persist_file_subset()
            self._refresh_file_tree_summary()

        def _exclude_all_sec_only(self) -> None:
            """Uncheck every leaf whose basename is in cfg.conditions.sec_only_files.

            Implements the 'Exclude all sec-only' amendment quick-action.
            """
            sec_set = set(
                self._cfg.get("conditions", {}).get("sec_only_files") or []
            )
            sec_folders = set(
                self._cfg.get("conditions", {}).get("sec_only_folders") or []
            )
            if not sec_set and not sec_folders:
                QMessageBox.information(
                    self, "fishsuite",
                    "No sec-only files or folders are configured yet "
                    "(see Conditions tab). Nothing to exclude."
                )
                return
            self._file_tree_loading = True
            n = 0
            try:
                for leaf in self._iter_leaf_items():
                    d = leaf.data(0, Qt.UserRole) or {}
                    if d.get("name") in sec_set or d.get("subfolder") in sec_folders:
                        if leaf.checkState(0) == Qt.Checked:
                            leaf.setCheckState(0, Qt.Unchecked)
                            n += 1
                        self._apply_leaf_visual(leaf)
            finally:
                self._file_tree_loading = False
            self._persist_file_subset()
            self._refresh_file_tree_summary()
            self._toast(f"Excluded {n} sec-only file(s).")

        def _apply_file_tree_filter(self) -> None:
            needle = (self.file_filter_edit.text() or "").strip().lower()
            for leaf in self._iter_leaf_items():
                if not needle:
                    leaf.setHidden(False)
                else:
                    d = leaf.data(0, Qt.UserRole) or {}
                    leaf.setHidden(needle not in d.get("name", "").lower())
            # Hide folder parents whose every child is hidden, so the user
            # isn't left staring at empty group headers when the filter is
            # restrictive.
            root = self.file_tree.invisibleRootItem()
            for i in range(root.childCount()):
                parent = root.child(i)
                d = parent.data(0, Qt.UserRole) or {}
                if not d.get("is_folder"):
                    continue
                any_visible = any(
                    not parent.child(j).isHidden()
                    for j in range(parent.childCount())
                )
                parent.setHidden(not any_visible)

        def _refresh_file_tree_summary(self) -> None:
            """Update the live summary chip above the tree."""
            leaves = self._iter_leaf_items()
            total = len(leaves)
            if total == 0:
                self.file_tree_summary.setText(
                    "(no files in the tree — set Input directory and click Refresh)"
                )
                return
            included = 0
            per_cond: Dict[str, int] = {}
            sec_excluded = 0
            for leaf in leaves:
                d = leaf.data(0, Qt.UserRole) or {}
                checked = leaf.checkState(0) == Qt.Checked
                if checked:
                    included += 1
                    cond = leaf.text(1) or "(unassigned)"
                    per_cond[cond] = per_cond.get(cond, 0) + 1
                else:
                    if d.get("is_sec"):
                        sec_excluded += 1
            cond_parts = ", ".join(
                f"{k}: {v}" for k, v in sorted(per_cond.items())
            )
            extra = ""
            if sec_excluded:
                extra = f"; Sec-only: {sec_excluded} excluded"
            summary = (
                f"Included: {included} of {total} files"
                + (f" ({cond_parts}{extra})" if cond_parts or extra else "")
            )
            if included == total:
                summary += "  —  empty subset → run includes ALL files"
            self.file_tree_summary.setText(summary)

        def _persist_file_subset(self) -> None:
            """Write the current checked-leaves to settings + cfg overrides.

            Stores the list under ``state.file_subset_by_input[input_dir]`` so
            different input dirs keep independent selections. Triggers a
            normal _on_field_changed so YAML preview + readiness refresh.
            """
            in_dir = self.input_edit.text().strip()
            if not in_dir:
                return
            leaves = self._iter_leaf_items()
            total = len(leaves)
            checked: List[str] = []
            for leaf in leaves:
                if leaf.checkState(0) == Qt.Checked:
                    d = leaf.data(0, Qt.UserRole) or {}
                    rel = d.get("rel") or d.get("name") or ""
                    if rel:
                        checked.append(rel)
            # If every file is checked, treat as "include all" and clear the
            # subset (legacy back-compat for downstream YAML).
            if total > 0 and len(checked) == total:
                subset_for_cfg: List[str] = []
            else:
                subset_for_cfg = checked
            sub_map = dict(self._settings.get("file_subset_by_input") or {})
            sub_map[in_dir] = subset_for_cfg
            self._settings["file_subset_by_input"] = sub_map
            # Snap into cfg so YAML preview shows it.
            self._cfg["input_file_subset"] = list(subset_for_cfg)
            if hasattr(self, "_yaml_timer"):
                self._yaml_timer.start()

        def _on_file_tree_context_menu(self, point) -> None:
            """Right-click menu over the tree (Brian's amendment)."""
            it = self.file_tree.itemAt(point)
            if it is None:
                return
            d = it.data(0, Qt.UserRole) or {}
            menu = QMenu(self.file_tree)
            is_folder = bool(d.get("is_folder"))
            if not is_folder:
                act_only = menu.addAction("Include only this")
                act_excl = menu.addAction("Exclude this")
                act_sec = menu.addAction("Mark as sec-only")
                menu.addSeparator()
                act_all = menu.addAction("Include all in this folder")
                act_none = menu.addAction("Exclude all in this folder")
            else:
                act_only = None
                act_excl = None
                act_sec = None
                act_all = menu.addAction("Include all in this folder")
                act_none = menu.addAction("Exclude all in this folder")
            chosen = menu.exec_(self.file_tree.viewport().mapToGlobal(point))
            if chosen is None:
                return
            self._file_tree_loading = True
            try:
                if not is_folder and chosen == act_only:
                    for leaf in self._iter_leaf_items():
                        leaf.setCheckState(
                            0, Qt.Checked if leaf is it else Qt.Unchecked
                        )
                        self._apply_leaf_visual(leaf)
                elif not is_folder and chosen == act_excl:
                    it.setCheckState(0, Qt.Unchecked)
                    self._apply_leaf_visual(it)
                elif not is_folder and chosen == act_sec:
                    # Add this file's basename to cfg.conditions.sec_only_files
                    # AND to the visible list on the Conditions tab.
                    name = d.get("name") or ""
                    if name:
                        existing = {
                            self.sec_files_list.item(i).text()
                            for i in range(self.sec_files_list.count())
                        }
                        if name not in existing:
                            self.sec_files_list.addItem(name)
                        # Update leaf display "Sec-only" column + flag.
                        it.setText(2, "yes")
                        d["is_sec"] = True
                        it.setData(0, Qt.UserRole, d)
                elif chosen in (act_all, act_none):
                    want = (chosen == act_all)
                    if is_folder:
                        for ci in range(it.childCount()):
                            leaf = it.child(ci)
                            leaf.setCheckState(
                                0, Qt.Checked if want else Qt.Unchecked
                            )
                            self._apply_leaf_visual(leaf)
                    else:
                        # Apply to this leaf's siblings under the same folder
                        # (or the whole tree if the leaf has no folder parent).
                        parent = it.parent()
                        if parent is None:
                            for leaf in self._iter_leaf_items():
                                leaf.setCheckState(
                                    0, Qt.Checked if want else Qt.Unchecked
                                )
                                self._apply_leaf_visual(leaf)
                        else:
                            for ci in range(parent.childCount()):
                                leaf = parent.child(ci)
                                leaf.setCheckState(
                                    0, Qt.Checked if want else Qt.Unchecked
                                )
                                self._apply_leaf_visual(leaf)
            finally:
                self._file_tree_loading = False
            self._persist_file_subset()
            self._refresh_file_tree_summary()
            # _on_field_changed wires conditions tab + YAML preview.
            self._on_field_changed()

        # ==================================================================
        # YAML preview + helpers
        # ==================================================================

        def _refresh_yaml_preview(self) -> None:
            try:
                text = _state.cfg_to_yaml_str(self._cfg)
            except Exception as e:
                text = f"# (failed to render YAML: {e})"
            self.yaml_view.setPlainText(text)

        def _toast(self, msg: str) -> None:
            self._footer_status.setText(msg)
            self._set_footer_state("info")
            QTimer.singleShot(2500, self._refresh_readiness)

        def _set_footer_state(self, state: str) -> None:
            """Switch the footer status label colour via a QSS property.

            ``state`` is one of: 'ok' | 'bad' | 'info' | 'idle'.
            The actual colour comes from the active theme stylesheet so the
            footer is readable in both light AND dark modes.
            """
            self._footer_status.setProperty("state", state)
            # Force the stylesheet system to re-evaluate selectors that match
            # on the property we just changed.
            st = self._footer_status.style()
            st.unpolish(self._footer_status)
            st.polish(self._footer_status)
            self._footer_status.update()

        # ==================================================================
        # Browse / paths
        # ==================================================================

        def _browse_input(self) -> None:
            initial = self.input_edit.text() or str(Path.home())
            chosen = QFileDialog.getExistingDirectory(self, "Pick input directory", initial)
            if chosen:
                self.input_edit.setText(chosen)
                # Refresh the per-file tree for the new dir (Improvement 2).
                try:
                    self._on_file_tree_refresh()
                except AttributeError:
                    pass

        def _browse_output(self) -> None:
            initial = self.output_base_edit.text() or str(Path.home())
            chosen = QFileDialog.getExistingDirectory(self, "Pick output base directory", initial)
            if chosen:
                self.output_base_edit.setText(chosen)

        # ==================================================================
        # Settings
        # ==================================================================

        def _restore_settings(self) -> None:
            s = self._settings
            self.input_edit.blockSignals(True)
            self.input_edit.setText(str(s.get("last_input_dir") or ""))
            self.input_edit.blockSignals(False)
            self.output_base_edit.blockSignals(True)
            self.output_base_edit.setText(str(s.get("last_output_base") or _state.DEFAULT_OUTPUT_BASE))
            self.output_base_edit.blockSignals(False)
            self.tag_edit.blockSignals(True)
            self.tag_edit.setText(str(s.get("last_tag") or "run"))
            self.tag_edit.blockSignals(False)
            self.skip_down.blockSignals(True)
            self.skip_down.setChecked(bool(s.get("skip_downstream")))
            self.skip_down.blockSignals(False)
            # Re-apply any saved overrides on top of the loaded preset.
            ov = s.get("last_overrides") or {}
            if ov:
                self._cfg = _state.merge_config(self._cfg, ov)
                self._cfg_to_widgets()
            # Restore window geometry
            wg = s.get("window") or {}
            if wg.get("w") and wg.get("h"):
                self.resize(int(wg["w"]), int(wg["h"]))
            if wg.get("x") is not None and wg.get("y") is not None:
                self.move(int(wg["x"]), int(wg["y"]))
            self._refresh_output_preview()
            # If an input dir was saved, populate the file tree + resolved
            # channel labels so the user sees state immediately on launch.
            try:
                self._on_file_tree_refresh()
            except AttributeError:
                pass
            try:
                self._refresh_resolved_channel_labels()
            except AttributeError:
                pass

        def _capture_settings_for_save(self) -> Dict[str, Any]:
            # ``theme`` is updated by _apply_theme, but if the user never
            # touched the switcher we still want the resolved key written.
            self._settings.setdefault("theme", _state.resolve_theme(self._settings))
            self._settings.update({
                "last_preset": self._current_preset_stem,
                "last_input_dir": self.input_edit.text(),
                "last_output_base": self.output_base_edit.text(),
                "last_tag": self.tag_edit.text(),
                "skip_downstream": bool(self.skip_down.isChecked()),
                "last_overrides": self._cfg,
                "window": {"w": self.width(), "h": self.height(),
                           "x": self.x(), "y": self.y()},
            })
            return self._settings

        def closeEvent(self, ev: QCloseEvent) -> None:
            try:
                s = self._capture_settings_for_save()
                _state.save_settings(s)
            except Exception:
                pass
            super().closeEvent(ev)

        def _on_theme_changed(self, *_args) -> None:
            """User picked a new theme — apply + persist."""
            name = self._theme_combo.currentData() or "system"
            self._apply_theme(str(name))
            _state.save_settings(self._capture_settings_for_save())

        # ==================================================================
        # Run flow
        # ==================================================================

        def _on_run_clicked(self) -> None:
            # Sanity recheck just before launching.
            statuses = _ready.evaluate_all(
                self._cfg,
                input_dir=self.input_edit.text(),
                output_base=self.output_base_edit.text(),
                run_tag=self.tag_edit.text(),
            )
            if not _ready.overall_ready(statuses):
                QMessageBox.warning(
                    self, "fishsuite",
                    "Some required fields are missing or invalid. Check the red dots."
                )
                return
            in_dir = Path(self.input_edit.text())
            if not in_dir.is_dir():
                QMessageBox.critical(self, "fishsuite",
                                     f"Input dir does not exist:\n{in_dir}")
                return
            out_dir = _state.compute_output_dir(
                self.output_base_edit.text() or str(_state.DEFAULT_OUTPUT_BASE),
                self.tag_edit.text(),
            )
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                QMessageBox.critical(self, "fishsuite",
                                     f"Could not create output dir:\n{out_dir}\n{e}")
                return
            # Write per-run YAML
            run_yaml = out_dir / "_run_config.yaml"
            try:
                with open(run_yaml, "w", encoding="utf-8") as f:
                    yaml.safe_dump(self._cfg, f, sort_keys=False, default_flow_style=False)
            except Exception as e:
                QMessageBox.critical(self, "fishsuite",
                                     f"Could not write _run_config.yaml:\n{e}")
                return

            py = str(_state.DEFAULT_PYTHON_EXE) if _state.DEFAULT_PYTHON_EXE.is_file() else sys.executable
            cmd_pipeline = [
                py, "-m", "fishsuite.cli", "run",
                "--config", str(run_yaml),
                "--input-dir", str(in_dir),
                "--output-dir", str(out_dir),
                "--parallel", str(self._cfg.get("parallel", {}).get("workers", "auto")),
            ]
            cmd_down: Optional[List[str]] = None
            down_cwd: Optional[Path] = None
            if not self.skip_down.isChecked():
                cmd_down = [py, "-m", _state.DEFAULT_DOWNSTREAM_MODULE,
                            "--output-dir", str(out_dir)]
                down_cwd = _state.DEFAULT_DOWNSTREAM_CWD if _state.DEFAULT_DOWNSTREAM_CWD.is_dir() else None

            self._last_output_dir = out_dir
            self._clear_log()
            self._append_log(f"[gui] preset       : {self._current_preset_stem}\n")
            self._append_log(f"[gui] input dir    : {in_dir}\n")
            self._append_log(f"[gui] output dir   : {out_dir}\n")
            self._append_log(f"[gui] run config   : {run_yaml}\n")
            self._append_log(f"[gui] cmd (pipe)   : {' '.join(cmd_pipeline)}\n")
            if cmd_down:
                self._append_log(f"[gui] cmd (down)   : {' '.join(cmd_down)}  (cwd={down_cwd})\n")
            self._append_log("\n")

            self._set_running_ui(True)
            self.run_status_dot.set_status("yellow")
            self.run_status_label.setText("Running…")
            self.prog_bar.setRange(0, 0)  # indeterminate until first progress line
            self.prog_desc.setText("Starting pipeline…")
            # Switch user to the Run tab automatically.
            self._tabs.setCurrentIndex(self._tab_indices["run"])
            # Save settings before launching so we don't lose them on a crash.
            _state.save_settings(self._capture_settings_for_save())
            self._runner.start(cmd_pipeline, cmd_down, out_dir, downstream_cwd=down_cwd)

        def _on_stop_clicked(self) -> None:
            if not self._runner.is_running():
                return
            ret = QMessageBox.question(
                self, "fishsuite", "Stop the running pipeline?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if ret == QMessageBox.Yes:
                self._runner.stop()

        def _on_run_line(self, text: str) -> None:
            self._append_log(text)

        def _on_run_progress(self, cur: int, tot: int, desc: str) -> None:
            if tot > 0:
                self.prog_bar.setRange(0, tot)
                self.prog_bar.setValue(cur)
                self.prog_bar.setFormat(f"{cur}/{tot}  %p%")
            self.prog_desc.setText(desc)

        def _on_run_phase(self, phase: str) -> None:
            if phase == "pipeline":
                self.run_status_label.setText("Pipeline running…")
            elif phase == "downstream":
                self.run_status_label.setText("Downstream figures running…")
            elif phase == "idle":
                pass

        def _on_run_finished(self, success: bool, out_dir: str) -> None:
            self._set_running_ui(False)
            if success:
                self.run_status_label.setText("Done")
                self.run_status_dot.set_status("green")
                self.prog_bar.setRange(0, 1)
                self.prog_bar.setValue(1)
                self.prog_desc.setText("Complete.")
            else:
                self.run_status_label.setText("Failed")
                self.run_status_dot.set_status("red")
                self.prog_bar.setRange(0, 1)
                self.prog_bar.setValue(0)
            if out_dir:
                self.open_btn.setEnabled(True)
                self.reveal_btn.setEnabled(True)
                _state.append_run_history(self._settings, {
                    "output_dir": out_dir,
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "success": bool(success),
                    "preset": self._current_preset_stem,
                    "tag": self.tag_edit.text(),
                })
                _state.save_settings(self._capture_settings_for_save())
                self._refresh_history()

        def _set_running_ui(self, running: bool) -> None:
            self.run_btn.setEnabled(not running)
            self.stop_btn.setEnabled(running)
            for k, idx in self._tab_indices.items():
                if k != "run" and k != "yaml":
                    self._tabs.setTabEnabled(idx, not running)

        def _append_log(self, text: str) -> None:
            cursor = self.log_view.textCursor()
            cursor.movePosition(QTextCursor.End)
            cursor.insertText(text)
            self.log_view.setTextCursor(cursor)
            self.log_view.ensureCursorVisible()

        def _clear_log(self) -> None:
            self.log_view.clear()

        def _on_open_output(self) -> None:
            if not self._last_output_dir or not Path(self._last_output_dir).exists():
                return
            try:
                os.startfile(str(self._last_output_dir))  # type: ignore[attr-defined]
            except Exception as e:
                QMessageBox.warning(self, "fishsuite", f"Could not open output dir:\n{e}")

        def _on_reveal_output(self) -> None:
            if not self._last_output_dir:
                return
            try:
                if os.name == "nt":
                    subprocess.Popen(["explorer", "/select,", str(self._last_output_dir)])
                else:
                    os.startfile(str(self._last_output_dir.parent))  # type: ignore[attr-defined]
            except Exception as e:
                QMessageBox.warning(self, "fishsuite", f"Could not reveal:\n{e}")

        # ==================================================================
        # Run history
        # ==================================================================

        def _refresh_history(self) -> None:
            self.history_list.clear()
            for entry in (self._settings.get("run_history") or [])[:5]:
                ts = entry.get("ts", "")
                tag = entry.get("tag", "")
                ok = "OK " if entry.get("success") else "FAIL"
                preset = entry.get("preset", "")
                out = entry.get("output_dir", "")
                item = QListWidgetItem(f"[{ok}]  {ts}  preset={preset}  tag={tag}\n        {out}")
                item.setData(Qt.UserRole, out)
                self.history_list.addItem(item)

        def _on_history_open(self, item: QListWidgetItem) -> None:
            out = item.data(Qt.UserRole)
            if not out:
                return
            try:
                os.startfile(str(out))  # type: ignore[attr-defined]
            except Exception as e:
                QMessageBox.warning(self, "fishsuite", f"Could not open:\n{e}")

        # ==================================================================
        # YAML preview actions
        # ==================================================================

        def _copy_yaml(self) -> None:
            cb = QApplication.instance().clipboard()
            cb.setText(self.yaml_view.toPlainText())
            self._toast("YAML copied to clipboard")

        def _export_yaml(self) -> None:
            initial = self.output_base_edit.text() or str(Path.home())
            path, _flt = QFileDialog.getSaveFileName(
                self, "Export YAML",
                str(Path(initial) / f"fishsuite_{self.tag_edit.text() or 'run'}.yaml"),
                "YAML (*.yaml *.yml)"
            )
            if not path:
                return
            try:
                Path(path).write_text(self.yaml_view.toPlainText(), encoding="utf-8")
                self._toast(f"Exported: {path}")
            except Exception as e:
                QMessageBox.warning(self, "fishsuite", f"Could not export:\n{e}")


# ---------------------------------------------------------------------------
# Module entry — used by both the CLI subcommand and `python -m fishsuite.gui`.
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    """Launch the GUI. Returns 0 on clean shutdown, 1 if Qt unavailable."""
    import argparse
    parser = argparse.ArgumentParser(prog="fishsuite.gui",
                                     description="fishsuite desktop launcher")
    parser.add_argument("--print-toolkit", action="store_true",
                        help="Print which UI toolkit is available, then exit.")
    args = parser.parse_args(argv)
    if args.print_toolkit:
        print("PySide6" if _QT_OK else "(none)")
        return 0
    if not _QT_OK:
        sys.stderr.write(
            f"fishsuite gui: PySide6 is not available in this environment:\n  {_QT_ERR}\n"
            f"Install with: pip install PySide6\n"
        )
        return 1
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("fishsuite")
    # Fusion gives a consistent baseline across Windows / macOS / Linux and
    # respects QPalette overrides cleanly. The native Windows11 style ignores
    # several of our QSS rules (particularly QSpinBox arrows and combo
    # popups), which is the underlying cause of the original readability
    # complaints.
    if "Fusion" in QStyleFactory.keys():
        app.setStyle(QStyleFactory.create("Fusion"))
    win = FishsuiteWindow()
    win.show()
    return int(app.exec())


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
