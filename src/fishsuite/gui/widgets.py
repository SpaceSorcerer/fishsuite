"""Small reusable Qt widgets for the fishsuite GUI.

All widgets degrade gracefully if PySide6 is missing (import succeeds but
classes are unusable) — only the main entry point checks for Qt.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

try:
    from PySide6.QtCore import Qt, Signal, QSize
    from PySide6.QtGui import (
        QColor,
        QFont,
        QIcon,
        QPainter,
        QPalette,
        QPixmap,
    )
    from PySide6.QtWidgets import (
        QAbstractSpinBox,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFormLayout,
        QFrame,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QListWidget,
        QPushButton,
        QSizePolicy,
        QSlider,
        QSpinBox,
        QStyle,
        QStyleOption,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
    _QT_OK = True
except Exception:  # pragma: no cover
    _QT_OK = False


# Status colors (Okabe-Ito-adjacent palette tuned for traffic-light readability)
COLOR_RED = "#d32f2f"      # vermillion
COLOR_YELLOW = "#e8a33d"   # amber
COLOR_GREEN = "#0f9d58"    # green
COLOR_GREY = "#9aa0a6"
COLOR_TEAL = "#0a7d6b"     # primary accent (Run button)
COLOR_TEAL_HOVER = "#0b8d78"

# Theme-aware muted text/border colors. These are the SAME variables referenced
# by both QSS stylesheets AND inline setStyleSheet calls in main.py via the
# ``muted_color(dark)`` and ``divider_color(dark)`` helpers below. Centralising
# them here means we never have a light hex hard-wired into a widget that the
# dark stylesheet then cannot override (the original bug).
MUTED_LIGHT = "#5b6470"     # darker grey-blue: 5.4:1 on #ffffff (WCAG AA pass)
MUTED_DARK = "#a3aab8"      # light grey-blue: 7.4:1 on #0f172a   (WCAG AAA)
DIVIDER_LIGHT = "#e5e7eb"
DIVIDER_DARK = "#334155"
ACCENT_BG_LIGHT = "#eef4f3"  # top bar / preview chip backgrounds
ACCENT_BG_DARK = "#172033"


def muted_color(dark: bool) -> str:
    """Return the muted/secondary text color for the active theme."""
    return MUTED_DARK if dark else MUTED_LIGHT


def divider_color(dark: bool) -> str:
    return DIVIDER_DARK if dark else DIVIDER_LIGHT


def accent_bg(dark: bool) -> str:
    return ACCENT_BG_DARK if dark else ACCENT_BG_LIGHT


def panel_bg(dark: bool) -> str:
    """Background for inline preview chips (channel preview, etc.)."""
    return "#1f2937" if dark else "#f3f4f6"


def make_dot_icon(color_hex: str, size: int = 14) -> "QIcon":
    """Build a circular colored icon for use in tab labels."""
    if not _QT_OK:
        return None  # type: ignore[return-value]
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    color = QColor(color_hex)
    p.setBrush(color)
    p.setPen(QColor(color).darker(140))
    p.drawEllipse(1, 1, size - 2, size - 2)
    p.end()
    return QIcon(pm)


def status_color(code: str) -> str:
    return {"red": COLOR_RED, "yellow": COLOR_YELLOW, "green": COLOR_GREEN}.get(code, COLOR_GREY)


# ---------------------------------------------------------------------------
# StatusDotLabel — used in headers
# ---------------------------------------------------------------------------

if _QT_OK:

    class StatusDot(QLabel):
        """A circular colored indicator that can be placed inline.

        The dot is drawn with a contrasting outline so that the three status
        colours (red / yellow / green) remain visible on either light or dark
        backgrounds — the outline is a darkened version of the fill, which
        gives at least a ~2:1 edge against any reasonable tab background.
        """

        def __init__(self, diameter: int = 14, parent=None):
            super().__init__(parent)
            self._d = diameter
            self._color = COLOR_GREY
            self.setFixedSize(QSize(diameter + 4, diameter + 4))

        def set_status(self, code: str) -> None:
            self._color = status_color(code)
            self.update()

        def paintEvent(self, _ev):
            p = QPainter(self)
            p.setRenderHint(QPainter.Antialiasing)
            c = QColor(self._color)
            p.setBrush(c)
            # 2px outline using a darkened fill — visible on dark theme too.
            pen = QColor(c).darker(170)
            p.setPen(pen)
            p.drawEllipse(2, 2, self._d, self._d)
            p.end()

    class SectionHeader(QWidget):
        """Bold section header with subtle underline + optional subtitle.

        Uses object names + QSS class selectors instead of inline style so
        the colours follow the global light/dark theme automatically.
        """

        def __init__(self, title: str, subtitle: str = "", parent=None):
            super().__init__(parent)
            v = QVBoxLayout(self)
            v.setContentsMargins(0, 8, 0, 6)
            v.setSpacing(2)
            lbl = QLabel(title)
            lbl.setObjectName("sectionHeaderTitle")
            f = QFont()
            f.setBold(True)
            f.setPointSize(12)  # bumped from 11 — sections now scan faster
            lbl.setFont(f)
            v.addWidget(lbl)
            if subtitle:
                sub = QLabel(subtitle)
                sub.setObjectName("sectionHeaderSubtitle")
                sub.setWordWrap(True)
                v.addWidget(sub)
            line = QFrame()
            line.setObjectName("sectionHeaderDivider")
            line.setFrameShape(QFrame.HLine)
            v.addWidget(line)

    class LabeledRow(QWidget):
        """A horizontal row: <label> <widget> [unit-label] — used inside forms."""

        def __init__(self, label: str, widget: QWidget, unit: str = "",
                     tooltip: str = "", parent=None):
            super().__init__(parent)
            h = QHBoxLayout(self)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(8)
            self._lbl = QLabel(label)
            self._lbl.setMinimumWidth(180)
            self._lbl.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
            if tooltip:
                self._lbl.setToolTip(tooltip)
                widget.setToolTip(tooltip)
            h.addWidget(self._lbl)
            h.addWidget(widget, 1)
            if unit:
                u = QLabel(unit)
                u.setObjectName("unitLabel")
                h.addWidget(u)

    class SliderSpin(QWidget):
        """A horizontal QSlider + QDoubleSpinBox kept in sync.

        Emits ``valueChanged(float)`` whenever either widget changes.
        """

        valueChanged = Signal(float)

        def __init__(self, lo: float, hi: float, step: float, *,
                     decimals: int = 2, value: Optional[float] = None,
                     unit: str = "", parent=None):
            super().__init__(parent)
            self._lo = float(lo)
            self._hi = float(hi)
            self._step = float(step)
            self._decimals = int(decimals)
            self._syncing = False

            self._spin = QDoubleSpinBox()
            self._spin.setRange(self._lo, self._hi)
            self._spin.setSingleStep(self._step)
            self._spin.setDecimals(self._decimals)
            self._spin.setMinimumWidth(90)
            if unit:
                self._spin.setSuffix(f" {unit}")

            self._slider = QSlider(Qt.Horizontal)
            self._slider.setMinimumWidth(140)
            # Slider works in integer ticks; map ticks <-> float.
            ticks = max(1, int(round((self._hi - self._lo) / self._step)))
            self._slider.setRange(0, ticks)

            h = QHBoxLayout(self)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(8)
            h.addWidget(self._slider, 1)
            h.addWidget(self._spin, 0)

            v0 = float(value) if value is not None else self._lo
            self.setValue(v0)

            self._slider.valueChanged.connect(self._on_slider)
            self._spin.valueChanged.connect(self._on_spin)

        # --- helpers ---
        def _float_to_tick(self, v: float) -> int:
            return int(round((v - self._lo) / self._step))

        def _tick_to_float(self, t: int) -> float:
            return round(self._lo + t * self._step, max(self._decimals + 1, 6))

        def _on_slider(self, t: int):
            if self._syncing:
                return
            self._syncing = True
            try:
                v = self._tick_to_float(t)
                self._spin.setValue(v)
            finally:
                self._syncing = False
            self.valueChanged.emit(self.value())

        def _on_spin(self, v: float):
            if self._syncing:
                return
            self._syncing = True
            try:
                self._slider.setValue(self._float_to_tick(v))
            finally:
                self._syncing = False
            self.valueChanged.emit(self.value())

        # --- public ---
        def value(self) -> float:
            return float(self._spin.value())

        def setValue(self, v: float) -> None:
            v = max(self._lo, min(self._hi, float(v)))
            self._syncing = True
            try:
                self._spin.setValue(v)
                self._slider.setValue(self._float_to_tick(v))
            finally:
                self._syncing = False

    class IntSliderSpin(SliderSpin):
        valueChangedInt = Signal(int)

        def __init__(self, lo: int, hi: int, step: int = 1, value: Optional[int] = None,
                     unit: str = "", parent=None):
            super().__init__(lo, hi, step, decimals=0, value=value, unit=unit, parent=parent)
            self._spin.valueChanged.connect(lambda v: self.valueChangedInt.emit(int(round(v))))

        def value(self) -> int:  # type: ignore[override]
            return int(round(self._spin.value()))

        def setValue(self, v: int) -> None:  # type: ignore[override]
            super().setValue(int(round(v)))

    class FolderConditionTable(QWidget):
        """Editable two-column table: subfolder name -> condition label."""

        changed = Signal()

        def __init__(self, parent=None):
            super().__init__(parent)
            self._tbl = QTableWidget(0, 2)
            self._tbl.setHorizontalHeaderLabels(["Subfolder name", "Condition label"])
            self._tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            self._tbl.verticalHeader().setVisible(False)
            self._tbl.setSelectionBehavior(QTableWidget.SelectRows)
            self._tbl.itemChanged.connect(lambda *_: self.changed.emit())

            btns = QHBoxLayout()
            btns.setContentsMargins(0, 0, 0, 0)
            add_btn = QPushButton("+ Add row")
            rm_btn = QPushButton("Remove selected")
            add_btn.clicked.connect(self._add_row)
            rm_btn.clicked.connect(self._remove_selected)
            btns.addWidget(add_btn)
            btns.addWidget(rm_btn)
            btns.addStretch(1)

            v = QVBoxLayout(self)
            v.setContentsMargins(0, 0, 0, 0)
            v.setSpacing(6)
            v.addWidget(self._tbl)
            v.addLayout(btns)

        def _add_row(self):
            r = self._tbl.rowCount()
            self._tbl.insertRow(r)
            self._tbl.setItem(r, 0, QTableWidgetItem(""))
            self._tbl.setItem(r, 1, QTableWidgetItem(""))
            self.changed.emit()

        def _remove_selected(self):
            rows = sorted({i.row() for i in self._tbl.selectedIndexes()}, reverse=True)
            if not rows:
                # If nothing selected, remove the last row.
                if self._tbl.rowCount():
                    self._tbl.removeRow(self._tbl.rowCount() - 1)
            else:
                for r in rows:
                    self._tbl.removeRow(r)
            self.changed.emit()

        def to_dict(self) -> dict:
            d: dict = {}
            for r in range(self._tbl.rowCount()):
                k_item = self._tbl.item(r, 0)
                v_item = self._tbl.item(r, 1)
                k = k_item.text().strip() if k_item else ""
                v = v_item.text().strip() if v_item else ""
                if k:
                    d[k] = v or k
            return d

        def load_dict(self, d: dict) -> None:
            self._tbl.blockSignals(True)
            try:
                self._tbl.setRowCount(0)
                for k, v in (d or {}).items():
                    r = self._tbl.rowCount()
                    self._tbl.insertRow(r)
                    self._tbl.setItem(r, 0, QTableWidgetItem(str(k)))
                    self._tbl.setItem(r, 1, QTableWidgetItem(str(v)))
            finally:
                self._tbl.blockSignals(False)
            self.changed.emit()

    class ReorderableList(QListWidget):
        """List widget with drag-reorder enabled (used for condition_order)."""

        changed = Signal()

        def __init__(self, parent=None):
            super().__init__(parent)
            self.setDragDropMode(QListWidget.InternalMove)
            self.setSelectionMode(QListWidget.SingleSelection)
            self.model().rowsMoved.connect(lambda *_: self.changed.emit())

        def set_items(self, items: List[str]) -> None:
            self.clear()
            for it in items:
                self.addItem(it)
            self.changed.emit()

        def get_items(self) -> List[str]:
            return [self.item(i).text() for i in range(self.count())]

else:
    StatusDot = None  # type: ignore
    SectionHeader = None  # type: ignore
    LabeledRow = None  # type: ignore
    SliderSpin = None  # type: ignore
    IntSliderSpin = None  # type: ignore
    FolderConditionTable = None  # type: ignore
    ReorderableList = None  # type: ignore


# ---------------------------------------------------------------------------
# Stylesheet
# ---------------------------------------------------------------------------

def _build_qss(*, dark: bool) -> str:
    """Build the full stylesheet for a theme.

    The previous version of this file had two large, partly-overlapping QSS
    blobs with two recurring problems:
      1. Several widgets set a custom ``background`` without an explicit
         ``color`` — so the foreground inherited from the *parent* QSS rule
         (often the wrong one in dark mode).
      2. Popup widgets (QComboBox dropdowns, QMenu, QToolTip) were not
         styled at all and fell back to the system palette, which on Win11
         is bright white even in our dark theme.

    This unified builder fixes both: **every** background-setting rule has
    a paired colour, and the dropdown / menu / tooltip / scrollbar / disabled
    state are all explicitly themed. Contrast targets WCAG AA (4.5:1 for body
    text, 3:1 for large text and UI affordances).
    """
    if dark:
        bg_window = "#0f172a"
        bg_pane = "#111827"
        bg_tab = "#1f2937"
        bg_tab_sel = "#111827"
        bg_tab_hover = "#2d3a52"
        bg_input = "#0f172a"
        bg_input_disabled = "#1a2235"
        bg_button = "#1f2937"
        bg_button_hover = "#334155"
        bg_button_disabled = "#1a2235"
        bg_chip = "#1f2937"
        bg_log = "#020617"
        bg_yaml = "#0b1220"
        bg_header = "#1f2937"
        bg_groupbox = "#111827"
        bg_dropdown = "#1f2937"
        bg_dropdown_hover = "#334155"
        bg_tooltip = "#1f2937"
        bg_topbar = ACCENT_BG_DARK
        fg_text = "#e5e7eb"          # 13.2:1 on bg_window
        fg_strong = "#ffffff"
        fg_muted = MUTED_DARK         # used for hints, units
        fg_subtle = "#cbd5e1"         # used on tabs, header section captions
        fg_disabled = "#6b7280"
        fg_log = "#cbd5e1"
        fg_yaml = "#e2e8f0"
        fg_tooltip = "#e5e7eb"
        border = "#334155"
        border_strong = "#475569"
        border_subtle = "#1f2937"
        run_disabled_bg = "#334155"
        run_disabled_fg = "#9ca3af"
        stop_hover_bg = "#3f1d1d"
        scroll_handle = "#475569"
        scroll_handle_hover = "#64748b"
    else:
        bg_window = "#fafafa"
        bg_pane = "#ffffff"
        bg_tab = "#eef0f3"
        bg_tab_sel = "#ffffff"
        bg_tab_hover = "#dde2ea"
        bg_input = "#ffffff"
        bg_input_disabled = "#f3f4f6"
        bg_button = "#ffffff"
        bg_button_hover = "#eef0f3"
        bg_button_disabled = "#f3f4f6"
        bg_chip = "#f3f4f6"
        bg_log = "#0b1220"           # log stays dark in both modes (mono code)
        bg_yaml = "#0b1220"
        bg_header = "#eef0f3"
        bg_groupbox = "#ffffff"
        bg_dropdown = "#ffffff"
        bg_dropdown_hover = "#dde9e6"
        bg_tooltip = "#1f2937"        # dark tooltip looks better in light mode
        bg_topbar = ACCENT_BG_LIGHT
        fg_text = "#1f2937"           # 13.4:1 on bg_window
        fg_strong = "#111827"
        fg_muted = MUTED_LIGHT
        fg_subtle = "#374151"
        fg_disabled = "#9ca3af"
        fg_log = "#e2e8f0"
        fg_yaml = "#e2e8f0"
        fg_tooltip = "#f9fafb"
        border = "#cbd0d8"
        border_strong = "#9aa1ac"
        border_subtle = DIVIDER_LIGHT
        run_disabled_bg = "#cbd5e1"
        run_disabled_fg = "#6b7280"   # was #f3f4f6 — that was unreadable
        stop_hover_bg = "#fef2f2"
        scroll_handle = "#cbd5e1"
        scroll_handle_hover = "#94a3b8"

    return f"""
/* ===== Base ===== */
QWidget {{
    font-family: "Segoe UI", "SF Pro Text", "Noto Sans", sans-serif;
    font-size: 11pt;
    color: {fg_text};
}}
QMainWindow, QDialog, QScrollArea, QScrollArea > QWidget > QWidget {{
    background: {bg_window};
    color: {fg_text};
}}
QLabel {{ background: transparent; color: {fg_text}; }}
QToolTip {{
    background: {bg_tooltip};
    color: {fg_tooltip};
    border: 1px solid {border};
    padding: 4px 6px;
}}

/* ===== Top bar (object-named in main.py) ===== */
QFrame#topBar {{
    background: {bg_topbar};
    border: 1px solid {border};
    border-radius: 6px;
}}
QFrame#topBar QLabel {{ color: {fg_text}; }}

/* ===== Tabs ===== */
QTabWidget::pane {{
    border: 1px solid {border_subtle};
    border-radius: 6px;
    top: -1px;
    background: {bg_pane};
}}
QTabBar {{ qproperty-drawBase: 0; background: transparent; }}
QTabBar::tab {{
    padding: 8px 14px;
    margin-right: 2px;
    background: {bg_tab};
    border: 1px solid {border_subtle};
    border-bottom: 0;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    color: {fg_subtle};
}}
QTabBar::tab:selected {{
    background: {bg_tab_sel};
    color: {fg_strong};
    font-weight: 600;
}}
QTabBar::tab:hover:!selected {{ background: {bg_tab_hover}; color: {fg_text}; }}

/* ===== Group boxes / forms ===== */
QGroupBox {{
    border: 1px solid {border_subtle};
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 14px;
    background: {bg_groupbox};
    color: {fg_text};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 6px;
    color: {fg_subtle};
    font-weight: 600;
}}

/* ===== SectionHeader children (object-named in widgets.py) ===== */
QLabel#sectionHeaderTitle {{ color: {fg_strong}; }}
QLabel#sectionHeaderSubtitle {{ color: {fg_muted}; }}
QFrame#sectionHeaderDivider {{ background: {border_subtle}; max-height: 1px; border: 0; }}
QLabel#unitLabel {{ color: {fg_muted}; }}

/* ===== Inputs ===== */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox,
QPlainTextEdit, QTextEdit, QListWidget, QTableWidget {{
    background: {bg_input};
    color: {fg_text};
    border: 1px solid {border};
    border-radius: 4px;
    padding: 4px 6px;
    selection-background-color: {COLOR_TEAL};
    selection-color: white;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus,
QPlainTextEdit:focus, QTextEdit:focus {{
    border-color: {COLOR_TEAL};
}}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled,
QComboBox:disabled, QPlainTextEdit:disabled, QTextEdit:disabled {{
    background: {bg_input_disabled};
    color: {fg_disabled};
    border-color: {border_subtle};
}}

/* QComboBox dropdown — the popup is a separate top-level QListView and
   inherits the platform palette unless we style it explicitly here. */
QComboBox::drop-down {{
    border: 0;
    width: 22px;
}}
QComboBox QAbstractItemView {{
    background: {bg_dropdown};
    color: {fg_text};
    border: 1px solid {border};
    selection-background-color: {COLOR_TEAL};
    selection-color: white;
    outline: 0;
}}

/* Spin box up/down buttons — keep their default arrows but with our colours */
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    background: transparent;
    border: 0;
    width: 16px;
}}

/* ===== Buttons ===== */
QPushButton, QToolButton {{
    background: {bg_button};
    color: {fg_text};
    border: 1px solid {border};
    border-radius: 4px;
    padding: 6px 12px;
}}
QPushButton:hover, QToolButton:hover {{
    background: {bg_button_hover};
    border-color: {border_strong};
}}
QPushButton:disabled, QToolButton:disabled {{
    background: {bg_button_disabled};
    color: {fg_disabled};
    border-color: {border_subtle};
}}
QPushButton:focus, QToolButton:focus {{
    border-color: {COLOR_TEAL};
}}

/* Run/Stop accent buttons */
QPushButton#runButton {{
    background: {COLOR_TEAL};
    color: white;
    border: 1px solid {COLOR_TEAL};
    padding: 10px 22px;
    font-weight: 600;
    font-size: 12pt;
    border-radius: 6px;
}}
QPushButton#runButton:hover {{
    background: {COLOR_TEAL_HOVER};
    border-color: {COLOR_TEAL_HOVER};
}}
QPushButton#runButton:disabled {{
    background: {run_disabled_bg};
    color: {run_disabled_fg};
    border-color: {run_disabled_bg};
}}
QPushButton#stopButton {{
    background: {bg_button};
    color: {COLOR_RED};
    border: 1px solid {COLOR_RED};
    font-weight: 600;
}}
QPushButton#stopButton:hover {{ background: {stop_hover_bg}; }}
QPushButton#stopButton:disabled {{
    color: {fg_disabled};
    border-color: {border};
    background: {bg_button_disabled};
}}

/* ===== Log + YAML views (always dark, monospace) ===== */
QPlainTextEdit#logView, QPlainTextEdit#yamlView {{
    background: {bg_log};
    color: {fg_log};
    border: 1px solid {border_subtle};
    font-family: "Cascadia Mono", "Consolas", monospace;
    font-size: 10pt;
}}

/* ===== Inline chip-style labels ===== */
QLabel#previewChip {{
    background: {bg_chip};
    color: {fg_text};
    border-radius: 4px;
    padding: 8px;
    font-family: "Cascadia Mono", "Consolas", monospace;
}}
QLabel#outputPreview, QLabel#runOutputPath {{
    color: {COLOR_TEAL};
    font-family: "Cascadia Mono", "Consolas", monospace;
    background: transparent;
}}
QLabel#footerOutput, QLabel#tipLabel, QLabel#scopeNote {{
    color: {fg_muted};
    background: transparent;
}}
QLabel#progDesc {{ color: {fg_subtle}; background: transparent; }}
QLabel#footerStatus {{ background: transparent; }}
QLabel#footerStatus[state="ok"]   {{ color: {COLOR_GREEN}; }}
QLabel#footerStatus[state="bad"]  {{ color: {COLOR_RED};   }}
QLabel#footerStatus[state="info"] {{ color: {COLOR_TEAL};  }}
QLabel#footerStatus[state="idle"] {{ color: {fg_muted};    }}

QScrollArea {{ border: 0; background: transparent; }}

/* ===== Sliders ===== */
QSlider::groove:horizontal {{
    height: 6px;
    background: {border_subtle};
    border-radius: 3px;
}}
QSlider::sub-page:horizontal {{
    background: {COLOR_TEAL};
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    background: {bg_input};
    border: 2px solid {COLOR_TEAL};
    width: 14px;
    margin: -5px 0;
    border-radius: 8px;
}}
QSlider::handle:horizontal:hover {{ background: {bg_button_hover}; }}

/* ===== Tables / headers ===== */
QHeaderView::section {{
    background: {bg_header};
    color: {fg_subtle};
    border: 0;
    border-right: 1px solid {border_subtle};
    border-bottom: 1px solid {border_subtle};
    padding: 6px 8px;
    font-weight: 600;
}}
QTableWidget {{
    gridline-color: {border_subtle};
    alternate-background-color: {bg_chip};
}}
QTableWidget::item:selected, QListWidget::item:selected {{
    background: {COLOR_TEAL};
    color: white;
}}

/* ===== Checkboxes / radios ===== */
QCheckBox, QRadioButton {{ background: transparent; color: {fg_text}; spacing: 6px; }}
QCheckBox:disabled, QRadioButton:disabled {{ color: {fg_disabled}; }}

/* ===== Menus ===== */
QMenu {{
    background: {bg_dropdown};
    color: {fg_text};
    border: 1px solid {border};
}}
QMenu::item:selected {{ background: {bg_dropdown_hover}; color: {fg_strong}; }}
QMenu::separator {{ height: 1px; background: {border_subtle}; margin: 4px 6px; }}

/* ===== Scrollbars ===== */
QScrollBar:vertical {{
    background: {bg_window};
    width: 12px;
    margin: 0;
    border: 0;
}}
QScrollBar::handle:vertical {{
    background: {scroll_handle};
    border-radius: 4px;
    min-height: 24px;
    margin: 2px;
}}
QScrollBar::handle:vertical:hover {{ background: {scroll_handle_hover}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{
    background: {bg_window};
    height: 12px;
    margin: 0;
    border: 0;
}}
QScrollBar::handle:horizontal {{
    background: {scroll_handle};
    border-radius: 4px;
    min-width: 24px;
    margin: 2px;
}}
QScrollBar::handle:horizontal:hover {{ background: {scroll_handle_hover}; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

/* ===== Progress bar ===== */
QProgressBar {{
    background: {bg_chip};
    color: {fg_text};
    border: 1px solid {border_subtle};
    border-radius: 4px;
    text-align: center;
    min-height: 18px;
}}
QProgressBar::chunk {{
    background: {COLOR_TEAL};
    border-radius: 3px;
}}
"""


LIGHT_QSS = _build_qss(dark=False)
DARK_QSS = _build_qss(dark=True)
