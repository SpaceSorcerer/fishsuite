"""CLI layer for the post-run utilities (Brian, 2026-06-18).

The two post-run utilities — ``coloc_backfill`` (CPU re-read of QKI coloc extras
onto a finished run) and ``walkthrough_figure`` (the 8-panel pipeline-walkthrough
composite) — were previously reachable ONLY via ``python -m fishsuite.core.<mod>``.
They are now first-class, friendly ``fishsuite`` subcommands so a non-expert can
run them:

    fishsuite backfill    --run <run_dir> [--staging ... --input ... --seed N
                                           --no-null-draws --no-radial --no-montage]
    fishsuite walkthrough --run <run_dir> [--image KEY --out PNG --staging ... --input ...]
    fishsuite postrun     --run <run_dir>   # one-shot: runs ALL post-run utilities

These tests prove the CLI WIRING (Click arg parsing, default/auto-detect
resolution, the on/off toggles, plain-English errors, and that each subcommand
dispatches to the correct underlying function with the right kwargs) WITHOUT
touching any VSI / GPU — the heavy functions (``backfill_run`` /
``build_walkthrough_figure``) are monkeypatched so we assert on the kwargs they
receive, not on real image I/O (that path is covered by the dedicated
``test_coloc_backfill`` / ``test_walkthrough_figure`` suites).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from fishsuite import cli as _cli


# ===========================================================================
# Helpers
# ===========================================================================
def _make_run_dir(tmp_path: Path, *, with_input_dir: bool = True) -> Path:
    """A minimal completed-run dir: run_config.json (+ optional recorded input_dir)
    and the two CSVs the utilities expect. No VSI / masks (the heavy functions are
    monkeypatched, so the contents past existence are irrelevant)."""
    run = tmp_path / "run"
    run.mkdir()
    rc = {"config_resolved": {}, "output_dir": str(run)}
    if with_input_dir:
        rc["input_dir"] = str(tmp_path / "staging")
        (tmp_path / "staging").mkdir(exist_ok=True)
    (run / "run_config.json").write_text(json.dumps(rc))
    (run / "per_image_summary.csv").write_text("image,condition\n")
    (run / "spot_metrics.csv").write_text("image,channel,x_px,y_px,nucleus_id\n")
    return run


# ===========================================================================
# (1) registration / help — the post-run commands are discoverable
# ===========================================================================
def test_postrun_subcommands_are_registered():
    runner = CliRunner()
    res = runner.invoke(_cli.cli, ["--help"])
    assert res.exit_code == 0
    for name in ("backfill", "walkthrough", "postrun"):
        assert name in res.output


def test_backfill_help_states_cpu_only_and_reuse():
    """The CPU-only + reuse-masks/spots clarification MUST be plain in --help."""
    runner = CliRunner()
    res = runner.invoke(_cli.cli, ["backfill", "--help"])
    assert res.exit_code == 0
    low = res.output.lower()
    assert "cpu" in low and "gpu" in low
    assert "mask" in low and "spot" in low
    # the toggles are documented
    for opt in ("--no-null-draws", "--no-radial", "--no-montage", "--seed",
                "--staging", "--input"):
        assert opt in res.output


def test_walkthrough_help_documents_options():
    runner = CliRunner()
    res = runner.invoke(_cli.cli, ["walkthrough", "--help"])
    assert res.exit_code == 0
    for opt in ("--image", "--out", "--staging", "--input", "--run"):
        assert opt in res.output


def test_postrun_help_explains_oneshot():
    runner = CliRunner()
    res = runner.invoke(_cli.cli, ["postrun", "--help"])
    assert res.exit_code == 0
    low = res.output.lower()
    assert "all" in low  # "runs ALL post-run utilities" / similar


# ===========================================================================
# (2) backfill — dispatch + kwargs + toggles + auto-detect
# ===========================================================================
def test_backfill_dispatches_with_defaults(tmp_path, monkeypatch):
    run = _make_run_dir(tmp_path)
    captured = {}

    def fake_backfill(run_dir, staging_dir=None, input_dir=None, **kw):
        captured["run_dir"] = str(run_dir)
        captured["staging_dir"] = staging_dir
        captured["input_dir"] = input_dir
        captured.update(kw)
        return {"written": {"coloc_null_draws": str(run / "coloc_null_draws.csv")},
                "gate": {"n_fail": 0}}

    monkeypatch.setattr(_cli, "_backfill_run", fake_backfill, raising=False)
    runner = CliRunner()
    res = runner.invoke(_cli.cli, ["backfill", "--run", str(run)])
    assert res.exit_code == 0, res.output
    assert captured["run_dir"] == str(run)
    # defaults: all three products ON, staging/input auto-detected (None -> the
    # underlying fn falls back to run_config input_dir)
    assert captured["do_null_draws"] is True
    assert captured["do_radial"] is True
    assert captured["do_montage"] is True
    assert captured["seed"] == 0
    assert captured["staging_dir"] is None
    assert captured["input_dir"] is None


def test_backfill_toggles_turn_products_off(tmp_path, monkeypatch):
    run = _make_run_dir(tmp_path)
    captured = {}

    def fake_backfill(run_dir, staging_dir=None, input_dir=None, **kw):
        captured.update(kw)
        return {"written": {}, "gate": {"n_fail": 0}}

    monkeypatch.setattr(_cli, "_backfill_run", fake_backfill, raising=False)
    runner = CliRunner()
    res = runner.invoke(_cli.cli, [
        "backfill", "--run", str(run),
        "--no-null-draws", "--no-radial", "--no-montage", "--seed", "7",
    ])
    assert res.exit_code == 0, res.output
    assert captured["do_null_draws"] is False
    assert captured["do_radial"] is False
    assert captured["do_montage"] is False
    assert captured["seed"] == 7


def test_backfill_passes_staging_and_input(tmp_path, monkeypatch):
    run = _make_run_dir(tmp_path)
    staging = tmp_path / "stg"
    staging.mkdir()
    captured = {}

    def fake_backfill(run_dir, staging_dir=None, input_dir=None, **kw):
        captured["staging_dir"] = str(staging_dir) if staging_dir else None
        captured["input_dir"] = str(input_dir) if input_dir else None
        return {"written": {}, "gate": {"n_fail": 0}}

    monkeypatch.setattr(_cli, "_backfill_run", fake_backfill, raising=False)
    runner = CliRunner()
    res = runner.invoke(_cli.cli, [
        "backfill", "--run", str(run), "--staging", str(staging),
    ])
    assert res.exit_code == 0, res.output
    assert captured["staging_dir"] == str(staging)


# ===========================================================================
# (3) friendly errors — no raw tracebacks for user-fixable conditions
# ===========================================================================
def test_backfill_missing_run_dir_is_friendly():
    runner = CliRunner()
    res = runner.invoke(_cli.cli, ["backfill", "--run", "Z:/does/not/exist"])
    assert res.exit_code != 0
    # Click's Path(exists=True) rejects it OR our handler does — either way a
    # plain message, never a Python traceback.
    assert "Traceback" not in res.output


def test_backfill_no_staging_recoverable_gives_clear_error(tmp_path, monkeypatch):
    """A run dir with NO recorded input_dir and no --staging -> the underlying
    fn raises ValueError; the CLI must surface a plain actionable message
    telling the user to pass --staging, not a traceback."""
    run = _make_run_dir(tmp_path, with_input_dir=False)

    def fake_backfill(run_dir, staging_dir=None, input_dir=None, **kw):
        raise ValueError("no staging_dir / input_dir given and run_config has no input_dir")

    monkeypatch.setattr(_cli, "_backfill_run", fake_backfill, raising=False)
    runner = CliRunner()
    res = runner.invoke(_cli.cli, ["backfill", "--run", str(run)])
    assert res.exit_code != 0
    assert "Traceback" not in res.output
    assert "--staging" in res.output


def test_backfill_missing_masks_message_is_actionable(tmp_path, monkeypatch):
    """A FileNotFoundError from the underlying fn (e.g. missing masks/CSVs) is
    presented as a plain-English message, not a traceback."""
    run = _make_run_dir(tmp_path)

    def fake_backfill(run_dir, staging_dir=None, input_dir=None, **kw):
        raise FileNotFoundError("run_dir missing per_image_summary.csv or spot_metrics.csv")

    monkeypatch.setattr(_cli, "_backfill_run", fake_backfill, raising=False)
    runner = CliRunner()
    res = runner.invoke(_cli.cli, ["backfill", "--run", str(run)])
    assert res.exit_code != 0
    assert "Traceback" not in res.output
    assert "per_image_summary" in res.output or "finished" in res.output.lower()


# ===========================================================================
# (4) walkthrough — dispatch + kwargs + default out
# ===========================================================================
def test_walkthrough_dispatches_with_defaults(tmp_path, monkeypatch):
    run = _make_run_dir(tmp_path)
    captured = {}

    def fake_walk(run_dir, staging_dir=None, input_dir=None, image_key=None,
                  out_path=None):
        captured["run_dir"] = str(run_dir)
        captured["staging_dir"] = staging_dir
        captured["input_dir"] = input_dir
        captured["image_key"] = image_key
        captured["out_path"] = out_path
        return str(run / "figures" / "07_coloc" / "79_pipeline_walkthrough.png")

    monkeypatch.setattr(_cli, "_build_walkthrough_figure", fake_walk, raising=False)
    runner = CliRunner()
    res = runner.invoke(_cli.cli, ["walkthrough", "--run", str(run)])
    assert res.exit_code == 0, res.output
    assert captured["run_dir"] == str(run)
    # defaults auto-resolved by the underlying fn
    assert captured["image_key"] is None
    assert captured["out_path"] is None
    assert captured["staging_dir"] is None


def test_walkthrough_passes_image_and_out(tmp_path, monkeypatch):
    run = _make_run_dir(tmp_path)
    captured = {}

    def fake_walk(run_dir, staging_dir=None, input_dir=None, image_key=None,
                  out_path=None):
        captured["image_key"] = image_key
        captured["out_path"] = out_path
        return str(out_path)

    monkeypatch.setattr(_cli, "_build_walkthrough_figure", fake_walk, raising=False)
    runner = CliRunner()
    out_png = tmp_path / "myfig.png"
    res = runner.invoke(_cli.cli, [
        "walkthrough", "--run", str(run),
        "--image", "g2_wDox__g2-Dox_01", "--out", str(out_png),
    ])
    assert res.exit_code == 0, res.output
    assert captured["image_key"] == "g2_wDox__g2-Dox_01"
    assert captured["out_path"] == str(out_png)


def test_walkthrough_error_is_friendly(tmp_path, monkeypatch):
    run = _make_run_dir(tmp_path)

    def fake_walk(*a, **k):
        raise FileNotFoundError("no '*__step01_*.png' panels under ...; cannot infer image_key")

    monkeypatch.setattr(_cli, "_build_walkthrough_figure", fake_walk, raising=False)
    runner = CliRunner()
    res = runner.invoke(_cli.cli, ["walkthrough", "--run", str(run)])
    assert res.exit_code != 0
    assert "Traceback" not in res.output


# ===========================================================================
# (5) postrun — one-shot runs ALL utilities in order with a final summary
# ===========================================================================
def test_postrun_runs_all_utilities(tmp_path, monkeypatch):
    run = _make_run_dir(tmp_path)
    calls = []

    def fake_backfill(run_dir, staging_dir=None, input_dir=None, **kw):
        calls.append("backfill")
        return {"written": {"coloc_null_draws": str(run / "coloc_null_draws.csv"),
                            "montage": str(run / "m.png")},
                "gate": {"n_fail": 0}}

    def fake_walk(run_dir, staging_dir=None, input_dir=None, image_key=None,
                  out_path=None):
        calls.append("walkthrough")
        return str(run / "figures" / "07_coloc" / "79_pipeline_walkthrough.png")

    monkeypatch.setattr(_cli, "_backfill_run", fake_backfill, raising=False)
    monkeypatch.setattr(_cli, "_build_walkthrough_figure", fake_walk, raising=False)
    runner = CliRunner()
    res = runner.invoke(_cli.cli, ["postrun", "--run", str(run)])
    assert res.exit_code == 0, res.output
    # both ran, backfill BEFORE walkthrough
    assert calls == ["backfill", "walkthrough"]
    # a per-step progress line + a final file summary
    low = res.output.lower()
    assert "backfill" in low and "walkthrough" in low
    assert "79_pipeline_walkthrough.png" in res.output


def test_postrun_continues_if_one_step_fails(tmp_path, monkeypatch):
    """postrun is the 'just make my figures' button — if one utility fails it
    reports the failure plainly and STILL attempts the others, then exits
    non-zero so the user knows something went wrong (no traceback)."""
    run = _make_run_dir(tmp_path)
    calls = []

    def fake_backfill(*a, **k):
        calls.append("backfill")
        raise FileNotFoundError("no nuclei masks found")

    def fake_walk(*a, **k):
        calls.append("walkthrough")
        return str(run / "fig.png")

    monkeypatch.setattr(_cli, "_backfill_run", fake_backfill, raising=False)
    monkeypatch.setattr(_cli, "_build_walkthrough_figure", fake_walk, raising=False)
    runner = CliRunner()
    res = runner.invoke(_cli.cli, ["postrun", "--run", str(run)])
    # walkthrough still attempted despite backfill failing
    assert calls == ["backfill", "walkthrough"]
    assert "Traceback" not in res.output
    assert res.exit_code != 0  # surfaced the failure


def test_postrun_forwards_staging(tmp_path, monkeypatch):
    run = _make_run_dir(tmp_path)
    staging = tmp_path / "stg2"
    staging.mkdir()
    seen = {}

    def fake_backfill(run_dir, staging_dir=None, input_dir=None, **kw):
        seen["backfill_staging"] = str(staging_dir) if staging_dir else None
        return {"written": {}, "gate": {"n_fail": 0}}

    def fake_walk(run_dir, staging_dir=None, input_dir=None, image_key=None,
                  out_path=None):
        seen["walk_staging"] = str(staging_dir) if staging_dir else None
        return str(run / "fig.png")

    monkeypatch.setattr(_cli, "_backfill_run", fake_backfill, raising=False)
    monkeypatch.setattr(_cli, "_build_walkthrough_figure", fake_walk, raising=False)
    runner = CliRunner()
    res = runner.invoke(_cli.cli, ["postrun", "--run", str(run),
                                   "--staging", str(staging)])
    assert res.exit_code == 0, res.output
    assert seen["backfill_staging"] == str(staging)
    assert seen["walk_staging"] == str(staging)
