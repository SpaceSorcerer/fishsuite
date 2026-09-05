"""Two provenance/interop fixes, 2026-09-05.

1. `scripts/coloc_standard_panel.py` read the arm from a `line` column, but
   `fishsuite report` writes it as `group`, so the panel silently fell back to
   the condition prefix and refused `--arm-order`.
2. `versions.txt` recorded `fishsuite_version` but not the commit, so which
   engine produced a run was evidenced only by logs outside the run directory.
"""
from __future__ import annotations

import importlib.util
import io
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from fishsuite.core.repro import engine_git_commit, write_versions_txt

REPO = Path(__file__).resolve().parents[1]
PANEL = REPO / "scripts" / "coloc_standard_panel.py"


# --------------------------------------------------------------- arm column
def _well_map_from(pw: pd.DataFrame) -> dict:
    """The panel's resolution rule, mirrored (the script is not importable:
    it executes argparse at module scope)."""
    arm_col = next((c for c in ("group", "line") if c in pw.columns), None)
    if arm_col and "well_id" in pw.columns:
        return dict(pw[["well_id", arm_col]].dropna().drop_duplicates().values)
    return {}


def test_panel_source_prefers_group_then_line():
    src = io.open(PANEL, encoding="utf-8").read()
    assert '("group", "line")' in src, "panel no longer prefers group then line"
    assert 'if {"line", "well_id"}.issubset' not in src, "old line-only gate survives"


def test_group_column_resolves_arms():
    pw = pd.DataFrame({"well_id": ["WT_1", "KO_1"], "group": ["WT", "QKI-KO"]})
    assert _well_map_from(pw) == {"WT_1": "WT", "KO_1": "QKI-KO"}


def test_line_column_still_resolves_arms():
    pw = pd.DataFrame({"well_id": ["WT_1", "KO_1"], "line": ["WT", "QKI-KO"]})
    assert _well_map_from(pw) == {"WT_1": "WT", "KO_1": "QKI-KO"}


def test_group_wins_when_both_are_present():
    pw = pd.DataFrame({"well_id": ["WT_1"], "group": ["WT"], "line": ["stale"]})
    assert _well_map_from(pw) == {"WT_1": "WT"}


def test_neither_column_yields_an_empty_map_not_a_crash():
    pw = pd.DataFrame({"well_id": ["WT_1"], "condition": ["WT_1"]})
    assert _well_map_from(pw) == {}


def test_a_real_report_per_well_has_group_not_line():
    """The mismatch this fixes: the report's own writer emits `group`."""
    src = io.open(REPO / "src" / "fishsuite" / "report" / "build.py",
                  encoding="utf-8").read()
    assert '"image", "condition", "group", "well_id"' in src


# ------------------------------------------------------------ engine commit
def test_engine_git_commit_matches_git_here():
    got = engine_git_commit()
    assert got != "unknown"
    sha = got.split("-")[0]
    assert len(sha) == 40 and all(c in "0123456789abcdef" for c in sha)
    real = subprocess.run(("git", "-C", str(REPO), "rev-parse", "HEAD"),
                          capture_output=True, text=True, check=True).stdout.strip()
    assert sha == real


def test_engine_git_commit_flags_a_dirty_tree_honestly():
    got = engine_git_commit()
    dirty = bool(subprocess.run(
        ("git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"),
        capture_output=True, text=True, check=True).stdout.strip())
    assert got.endswith("-dirty") == dirty


def test_versions_txt_records_the_commit(tmp_path):
    assert write_versions_txt(tmp_path, 0) is True
    lines = (tmp_path / "versions.txt").read_text(encoding="utf-8").splitlines()
    rec = [l for l in lines if l.startswith("fishsuite_git_commit:")]
    assert len(rec) == 1, lines[:6]
    assert rec[0].split(":", 1)[1].strip() == engine_git_commit()
    # it sits next to the version it disambiguates
    assert lines.index(rec[0]) == 1 + next(
        i for i, l in enumerate(lines) if l.startswith("fishsuite_version:"))


def test_engine_git_commit_returns_unknown_outside_a_repo(tmp_path, monkeypatch):
    """A checkout exported without .git must say unknown, not raise or lie."""
    import fishsuite.core.repro as _repro
    fake = tmp_path / "a" / "b" / "c"
    fake.mkdir(parents=True)
    monkeypatch.setattr(_repro, "__file__", str(fake / "repro.py"))
    assert _repro.engine_git_commit() == "unknown"


def test_engine_git_commit_never_raises(monkeypatch):
    import fishsuite.core.repro as _repro

    def boom(*a, **k):
        raise OSError("git missing")

    monkeypatch.setattr(subprocess, "run", boom)
    assert _repro.engine_git_commit() == "unknown"
