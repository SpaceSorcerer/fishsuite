"""The absolute-path CI gate.

A hard-coded machine path shipped to an outside user is the worst failure mode
available: if a directory of that name happens to exist on their machine, the
pipeline reads the wrong data and reports a number instead of an error. The gate
is scoped to what ships (derived from the wheel allow-list in pyproject.toml), so
a shipped preset can never carry an absolute path as an active value.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "check_no_absolute_paths.py"
_PRESETS = _REPO / "src" / "fishsuite" / "config" / "presets"


def _load():
    spec = importlib.util.spec_from_file_location("_abs_path_gate", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_abs_path_gate"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gate():
    return _load()


def _fake_presets_dir(root: Path) -> Path:
    """A scratch ``.../config/presets/`` — the path shape the exemption keys on."""
    d = root / "src" / "fishsuite" / "config" / "presets"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# THE GATE STILL WORKS
# ---------------------------------------------------------------------------
def test_the_matcher_self_test_passes(gate):
    """A gate that cannot fail is not a gate — the first version of this one was
    a grep that matched nothing."""
    gate._self_test()


def test_the_repo_is_clean(gate, monkeypatch, capsys):
    monkeypatch.chdir(_REPO)
    assert gate.main(["check", "src"]) == 0
    out = capsys.readouterr().out
    assert "exempted" in out and "dated run-record preset" in out


def test_a_shipped_preset_with_an_absolute_path_still_fails(gate, tmp_path):
    """The exemption must not swallow the presets that reach an outside user.

    Deliberately does NOT mutate a real preset: a test that edits a tracked file
    leaves the repository broken if it is interrupted. The two halves of the claim
    are checked separately — a shipped name IS gated, and a gated file with a bad
    path DOES report a finding.
    """
    shipped = gate.shipped_preset_names(_REPO / "pyproject.toml")
    assert shipped, "the wheel allowlist could not be read"
    for name in shipped:
        assert gate.is_gated_preset(_PRESETS / name, shipped), (
            f"{name} ships in the wheel but is exempt from the gate"
        )

    # Under a real config/presets path, which is what is_gated_preset keys on.
    victim = _fake_presets_dir(tmp_path) / sorted(shipped)[0]
    victim.write_text(
        'input_dir: "F:/Image Analysis Work/oops"\n', encoding="utf-8"
    )
    assert gate.is_gated_preset(victim, shipped)
    findings = gate.check_yaml(victim)
    assert len(findings) == 1 and "f:/" in findings[0]


def test_a_dated_preset_with_an_absolute_path_is_exempt(gate, tmp_path):
    """The other half: the same content under a dated run-record name is skipped,
    because the wheel does not ship it and the path IS the provenance."""
    shipped = gate.shipped_preset_names(_REPO / "pyproject.toml")
    record = _fake_presets_dir(tmp_path) / "some_lab_run_2026-07-03.yaml"
    record.write_text(
        'plate_layout_csv: "F:/Image Analysis Work/x/plate_layout.csv"\n',
        encoding="utf-8",
    )
    assert not gate.is_gated_preset(record, shipped)
    # The checker itself would still flag it — the exemption is in the WALK, so
    # the matcher stays honest and re-including the file re-gates it.
    assert gate.check_yaml(record)


# ---------------------------------------------------------------------------
# THE ALLOWLIST IS THE SINGLE SOURCE OF TRUTH
# ---------------------------------------------------------------------------
def test_the_allowlist_is_read_from_pyproject(gate):
    shipped = gate.shipped_preset_names(_REPO / "pyproject.toml")
    assert shipped
    for name in shipped:
        assert (_PRESETS / name).exists(), f"{name} is force-included but absent"


def test_an_unreadable_allowlist_checks_everything(gate, tmp_path):
    """Fail CLOSED. 'Could not tell what ships' must never mean 'exempt it'."""
    assert gate.shipped_preset_names(tmp_path / "nope.toml") is None
    assert gate.is_gated_preset(_PRESETS / "anything_2026-01-01.yaml", None)


def test_non_preset_files_are_never_exempt(gate):
    assert gate.is_gated_preset(
        Path("src/fishsuite/core/_superplot.py"), {"generic_100x_0p065.yaml"}
    )
    assert gate.is_gated_preset(Path("src/fishsuite/config/schema.py"), set())


# ---------------------------------------------------------------------------
# THE RESTORED PROVENANCE
# ---------------------------------------------------------------------------
def test_no_shipped_preset_carries_a_placeholder():
    """<SET_ME> belongs in a PORTABLE preset if anywhere, never as the record of
    a run that happened."""
    gate = _load()
    shipped = gate.shipped_preset_names(_REPO / "pyproject.toml") or set()
    for name in shipped:
        assert "<SET_ME>" not in (_PRESETS / name).read_text(encoding="utf-8")


def test_the_wheel_is_deny_all_then_allow():
    """Only the allow-listed portable presets ship. If the deny-all ever became
    an allow-all, any local scratch preset with a real path would go out."""
    text = (_REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert 'exclude = [\n    "src/fishsuite/config/presets/*.yaml",' in text
