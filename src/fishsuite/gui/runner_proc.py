"""Subprocess management for the fishsuite GUI.

Spawns ``fishsuite run`` (and optionally the downstream
single-condition-plots step) as child processes, streams stdout line-by-line
through Qt signals so the UI thread can append to the log without
crossing the Qt threading boundary.

Lifecycle:

    runner = PipelineRunner(parent=window)
    runner.line.connect(window.on_line)
    runner.progress.connect(window.on_progress)
    runner.finished.connect(window.on_finished)
    runner.start(cmd_pipeline, cmd_downstream_or_None, output_dir)

Signals:
    line(str)      - one stdout line from any child process
    progress(int, int, str)
                   - (current, total, description) parsed from rich progress
                     lines (e.g. "[3/12] H9-NT...") or "Pre-pass: 5/12"
    finished(bool, str)
                   - (success, output_dir) emitted exactly once when the
                     run (pipeline + optional downstream) completes
    phase_changed(str)
                   - "pipeline" / "downstream" / "idle"

Threading: we use a daemon ``threading.Thread`` per child process to drain
stdout, and Qt signals (queued connections from non-Qt threads) to push
text into the UI. The child's stdout is line-buffered text mode.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import List, Optional

try:
    from PySide6.QtCore import QObject, Signal
    _QT_OK = True
except Exception:  # pragma: no cover - lets module import on bare Python
    QObject = object  # type: ignore[assignment]

    class _DummySignal:
        def __init__(self, *a, **kw):
            self._slots = []

        def connect(self, slot):
            self._slots.append(slot)

        def emit(self, *a, **kw):
            for s in self._slots:
                s(*a, **kw)

    Signal = _DummySignal  # type: ignore[assignment]
    _QT_OK = False


# Regex for "[N/M] description" progress markers emitted by rich.
_PROG_RE = re.compile(r"\[(\d+)/(\d+)\]\s+(.*)")
# Pre-pass:    "Pre-pass: 5/12"
_PREPASS_RE = re.compile(r"Pre-pass[^:]*:\s+(\d+)/(\d+)")
# Final done line:   "Done in 12.34s"
_DONE_RE = re.compile(r"\bDone in [\d.]+s\b", re.IGNORECASE)


class PipelineRunner(QObject):
    line = Signal(str)                       # raw stdout text
    progress = Signal(int, int, str)         # (current, total, label)
    finished = Signal(bool, str)             # (success, output_dir)
    phase_changed = Signal(str)              # "pipeline" / "downstream" / "idle"

    def __init__(self, parent=None):
        if _QT_OK:
            super().__init__(parent)
        self._proc: Optional[subprocess.Popen] = None
        self._reader: Optional[threading.Thread] = None
        self._cmd_downstream: Optional[List[str]] = None
        self._downstream_cwd: Optional[Path] = None
        self._output_dir: Optional[Path] = None
        self._pipeline_ok: bool = False
        self._phase: str = "idle"
        self._stop_requested: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(
        self,
        cmd_pipeline: List[str],
        cmd_downstream: Optional[List[str]],
        output_dir: Path,
        *,
        downstream_cwd: Optional[Path] = None,
    ) -> None:
        if self.is_running():
            self.line.emit("[gui] already running; ignoring start\n")
            return
        self._output_dir = output_dir
        self._cmd_downstream = cmd_downstream
        self._downstream_cwd = downstream_cwd
        self._pipeline_ok = False
        self._stop_requested = False
        self._set_phase("pipeline")
        self._spawn(cmd_pipeline, cwd=None)

    def stop(self) -> None:
        self._stop_requested = True
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.kill()
                self.line.emit("[gui] stop requested — child killed.\n")
            except Exception as e:
                self.line.emit(f"[gui] kill failed: {e}\n")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _set_phase(self, phase: str) -> None:
        self._phase = phase
        self.phase_changed.emit(phase)

    def _spawn(self, cmd: List[str], cwd: Optional[Path]) -> None:
        try:
            creationflags = 0
            if os.name == "nt":
                creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            env = os.environ.copy()
            # Force unbuffered Python stdout so rich progress lines flush.
            env.setdefault("PYTHONUNBUFFERED", "1")
            self._proc = subprocess.Popen(
                cmd,
                cwd=str(cwd) if cwd else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                creationflags=creationflags,
            )
        except Exception as e:
            self.line.emit(f"[gui] failed to launch: {e}\n")
            self._finish(False)
            return
        self.line.emit(f"[gui] launched: {' '.join(cmd)}\n")
        self._reader = threading.Thread(target=self._drain, args=(self._proc,), daemon=True)
        self._reader.start()

    def _drain(self, proc: subprocess.Popen) -> None:
        try:
            assert proc.stdout is not None
            for raw in proc.stdout:
                self._handle_line(raw)
        except Exception as e:
            self.line.emit(f"[gui] reader error: {e}\n")
        finally:
            rc = proc.wait()
            self._on_child_exit(rc)

    def _handle_line(self, line: str) -> None:
        self.line.emit(line)
        # Parse progress markers (best-effort; never crash on parse error).
        try:
            m = _PROG_RE.search(line)
            if m:
                cur = int(m.group(1))
                tot = int(m.group(2))
                desc = m.group(3).strip()
                self.progress.emit(cur, tot, desc)
                return
            m = _PREPASS_RE.search(line)
            if m:
                cur = int(m.group(1))
                tot = int(m.group(2))
                self.progress.emit(cur, tot, "Pre-pass: nuclear-pixel pooling")
                return
        except Exception:
            pass

    def _on_child_exit(self, rc: int) -> None:
        self.line.emit(f"\n[gui] child exited rc={rc} (phase={self._phase})\n")
        if self._phase == "pipeline":
            self._pipeline_ok = (rc == 0)
            if self._stop_requested:
                self._finish(False)
                return
            if rc == 0 and self._cmd_downstream is not None:
                self._set_phase("downstream")
                self._spawn(self._cmd_downstream, cwd=self._downstream_cwd)
                return
            self._finish(rc == 0)
            return
        if self._phase == "downstream":
            if rc != 0:
                self.line.emit("[gui] downstream figure step failed (pipeline output still valid).\n")
            self._finish(self._pipeline_ok)
            return
        self._finish(rc == 0)

    def _finish(self, success: bool) -> None:
        self._set_phase("idle")
        out = str(self._output_dir) if self._output_dir else ""
        self.finished.emit(bool(success), out)
        self._proc = None
        self._reader = None
