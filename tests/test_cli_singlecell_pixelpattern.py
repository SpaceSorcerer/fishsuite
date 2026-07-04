"""CLI layer for the single-cell + pixel-pattern post-run utilities (2026-07-04).

Two more first-class ``fishsuite`` post-run subcommands sit alongside
``backfill`` / ``walkthrough`` / ``postrun``:

    fishsuite singlecell   --run <run_dir> [--abundance-col ... --group-a ...
                                            --group-b ... --no-figures --no-excel]
    fishsuite pixelpattern --run <run_dir> [--staging ... --secondary-match ...
                                            --no-figures --no-excel]

These tests prove the CLI WIRING (Click arg parsing, defaults, the on/off
toggles, plain-English errors, and that each subcommand dispatches to the correct
underlying function with the right kwargs) WITHOUT touching any image / GPU: the
heavy functions (``singlecell_run`` / ``pixelpattern_run``) are monkeypatched, so
we assert on the kwargs they receive, not on real I/O.
"""
from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from fishsuite import cli as _cli


def _make_run_dir(tmp_path: Path, *, with_input_dir: bool = True) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    rc = {"config_resolved": {}, "output_dir": str(run)}
    if with_input_dir:
        rc["input_dir"] = str(tmp_path / "staging")
        (tmp_path / "staging").mkdir(exist_ok=True)
    (run / "run_config.json").write_text(json.dumps(rc))
    (run / "nuclei_metrics.csv").write_text("image,condition,nucleus_id,nuclear_spot_count\n")
    (run / "per_image_summary.csv").write_text("image,condition\n")
    (run / "masks").mkdir(exist_ok=True)
    return run


# ===========================================================================
# registration / help
# ===========================================================================
def test_new_subcommands_are_registered():
    res = CliRunner().invoke(_cli.cli, ["--help"])
    assert res.exit_code == 0
    for name in ("singlecell", "pixelpattern"):
        assert name in res.output


def test_singlecell_help_states_cpu_and_options():
    res = CliRunner().invoke(_cli.cli, ["singlecell", "--help"])
    assert res.exit_code == 0
    low = res.output.lower()
    assert "cpu" in low
    for opt in ("--abundance-col", "--group-a", "--group-b", "--no-figures",
                "--no-excel", "--run"):
        assert opt in res.output


def test_pixelpattern_help_states_cpu_reuse_and_secondary():
    res = CliRunner().invoke(_cli.cli, ["pixelpattern", "--help"])
    assert res.exit_code == 0
    low = res.output.lower()
    assert "cpu" in low and "gpu" in low
    assert "mask" in low
    for opt in ("--staging", "--secondary-match", "--no-figures", "--run"):
        assert opt in res.output


# ===========================================================================
# singlecell dispatch
# ===========================================================================
def test_singlecell_dispatches_with_defaults(tmp_path, monkeypatch):
    run = _make_run_dir(tmp_path)
    captured = {}

    def fake_sc(run_dir, **kw):
        captured["run_dir"] = str(run_dir)
        captured.update(kw)
        return {"written": {"excel": str(run / "single_cell_analysis.xlsx")},
                "n_metrics": 10, "n_figures": 5, "out_dir": str(run)}

    monkeypatch.setattr(_cli, "_singlecell_run", fake_sc, raising=False)
    res = CliRunner().invoke(_cli.cli, ["singlecell", "--run", str(run)])
    assert res.exit_code == 0, res.output
    assert captured["run_dir"] == str(run)
    assert captured["abundance_col"] is None
    assert captured["group_a"] is None and captured["group_b"] is None
    assert captured["exclude_secondary"] is True
    assert captured["do_figures"] is True and captured["do_excel"] is True
    assert captured["out_subdir"] == "singlecell"


def test_singlecell_toggles_and_overrides(tmp_path, monkeypatch):
    run = _make_run_dir(tmp_path)
    captured = {}

    def fake_sc(run_dir, **kw):
        captured.update(kw)
        return {"written": {}, "n_metrics": 0, "n_figures": 0, "out_dir": str(run)}

    monkeypatch.setattr(_cli, "_singlecell_run", fake_sc, raising=False)
    res = CliRunner().invoke(_cli.cli, [
        "singlecell", "--run", str(run), "--abundance-col", "rna_spot_count",
        "--group-a", "NT", "--group-b", "KD", "--include-secondary",
        "--no-figures", "--no-excel", "--seed", "3",
    ])
    assert res.exit_code == 0, res.output
    assert captured["abundance_col"] == "rna_spot_count"
    assert captured["group_a"] == "NT" and captured["group_b"] == "KD"
    assert captured["exclude_secondary"] is False
    assert captured["do_figures"] is False and captured["do_excel"] is False
    assert captured["seed"] == 3


def test_singlecell_missing_nuclei_metrics_is_friendly(tmp_path, monkeypatch):
    run = _make_run_dir(tmp_path)

    def fake_sc(run_dir, **kw):
        raise FileNotFoundError("run_dir missing nuclei_metrics.csv")

    monkeypatch.setattr(_cli, "_singlecell_run", fake_sc, raising=False)
    res = CliRunner().invoke(_cli.cli, ["singlecell", "--run", str(run)])
    assert res.exit_code != 0
    assert "Traceback" not in res.output
    assert "nuclei_metrics" in res.output


# ===========================================================================
# pixelpattern dispatch
# ===========================================================================
def test_pixelpattern_dispatches_with_defaults(tmp_path, monkeypatch):
    run = _make_run_dir(tmp_path)
    captured = {}

    def fake_pp(run_dir, staging_dir=None, input_dir=None, **kw):
        captured["run_dir"] = str(run_dir)
        captured["staging_dir"] = staging_dir
        captured["input_dir"] = input_dir
        captured.update(kw)
        return {"written": {"metrics": str(run / "pixel_pattern_metrics.csv")},
                "n_images": 3, "n_nuclei": 90, "n_figures": 6,
                "out_dir": str(run), "stain_qc": {}}

    monkeypatch.setattr(_cli, "_pixelpattern_run", fake_pp, raising=False)
    res = CliRunner().invoke(_cli.cli, ["pixelpattern", "--run", str(run)])
    assert res.exit_code == 0, res.output
    assert captured["run_dir"] == str(run)
    assert captured["staging_dir"] is None and captured["input_dir"] is None
    assert captured["secondary_match"] is None
    assert captured["do_figures"] is True and captured["do_excel"] is True
    assert captured["out_subdir"] == "pixelpattern"


def test_pixelpattern_passes_staging_secondary_toggles(tmp_path, monkeypatch):
    run = _make_run_dir(tmp_path)
    staging = tmp_path / "stg"
    staging.mkdir()
    captured = {}

    def fake_pp(run_dir, staging_dir=None, input_dir=None, **kw):
        captured["staging_dir"] = str(staging_dir) if staging_dir else None
        captured.update(kw)
        return {"written": {}, "n_images": 0, "n_nuclei": 0, "n_figures": 0,
                "out_dir": str(run), "stain_qc": {}}

    monkeypatch.setattr(_cli, "_pixelpattern_run", fake_pp, raising=False)
    res = CliRunner().invoke(_cli.cli, [
        "pixelpattern", "--run", str(run), "--staging", str(staging),
        "--secondary-match", "well12", "--no-figures", "--no-excel",
    ])
    assert res.exit_code == 0, res.output
    assert captured["staging_dir"] == str(staging)
    assert captured["secondary_match"] == "well12"
    assert captured["do_figures"] is False and captured["do_excel"] is False


def test_pixelpattern_missing_masks_is_friendly(tmp_path, monkeypatch):
    run = _make_run_dir(tmp_path)

    def fake_pp(run_dir, staging_dir=None, input_dir=None, **kw):
        raise FileNotFoundError("no saved nuclei masks dir found")

    monkeypatch.setattr(_cli, "_pixelpattern_run", fake_pp, raising=False)
    res = CliRunner().invoke(_cli.cli, ["pixelpattern", "--run", str(run)])
    assert res.exit_code != 0
    assert "Traceback" not in res.output
    assert "mask" in res.output.lower()


def test_pixelpattern_no_raw_images_runtimeerror_is_friendly(tmp_path, monkeypatch):
    run = _make_run_dir(tmp_path)

    def fake_pp(run_dir, staging_dir=None, input_dir=None, **kw):
        raise RuntimeError("no nuclei processed -- check --staging points at the raw images")

    monkeypatch.setattr(_cli, "_pixelpattern_run", fake_pp, raising=False)
    res = CliRunner().invoke(_cli.cli, ["pixelpattern", "--run", str(run)])
    assert res.exit_code != 0
    assert "Traceback" not in res.output
