"""Provenance fixes raised by the three-lens review of 2026-09-17.

Source review: RESULTS_PROD_miat500_qki1050_LoG174_20260917-211001/_review/
review_summary.md section 4. One test group per item:

a. ``footprint-backfill`` wrote no ``command.log``.
b. ``--miat-floor`` defaulted to a literal 364.0 with no reference to the run.
c. ``display_only_ranges`` was a hardcoded block contradicting the run's levels.
d. ``report/ortho.py`` wrote a 3-line ``versions.txt`` and a non-runnable command.
e. the over-detection flag never said that filtering on ``qc_pass`` drops an
   induced arm wholesale.
f. ``coloc_standard_panel`` keyed wells on the condition, emitted a 1.65e-24
   MDE, and NaN-ed its own primary endpoint without a reason.
g. ``run_config.json`` identified the preset by path only and ``command.log``
   carried no quoted, copy-pasteable command.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from subprocess import list2cmdline

import numpy as np
import pandas as pd
import pytest

import fishsuite.core.exact_footprint_backfill as backfill
from fishsuite.core import repro
from fishsuite.config.schema import FishsuiteConfig

from test_backfill_native_hierarchy import native_run, _execute  # noqa: F401
from test_ortho_cli import synthetic_run  # noqa: F401

REPO = Path(__file__).resolve().parents[1]
PANEL_PATH = REPO / "scripts" / "coloc_standard_panel.py"
BACKFILL_PATH = REPO / "src" / "fishsuite" / "core" / "exact_footprint_backfill.py"


@pytest.fixture(scope="module")
def panel():
    spec = importlib.util.spec_from_file_location("_panel_prov_2026_09_17", PANEL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ a + b
@pytest.fixture
def stub_backfill(monkeypatch, tmp_path):
    """Run ``main`` without pixels: capture the parameters, make the output dir."""
    captured = {}

    def _stub(run_dir, hierarchy_path, output_dir, *, parameters=None, **kwargs):
        captured["parameters"] = parameters
        captured["output"] = Path(output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return {"n_images_complete": 0}

    monkeypatch.setattr(backfill, "run_exact_footprint_backfill", _stub)
    return captured


def _run_dir_with_output(native_run, run_config_output):
    run = native_run[0]
    path = run / "run_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["config_resolved"]["output"] = run_config_output
    path.write_text(json.dumps(config), encoding="utf-8")
    return run


def test_backfill_writes_command_log_with_resolved_arguments(
    native_run, stub_backfill, tmp_path, monkeypatch
):
    run = _run_dir_with_output(native_run, {"manual_rna_min": 500.0, "manual_rna_max": 4000.0})
    monkeypatch.setenv("PYTHONPATH", str(REPO / "src"))
    argv = ["--run", str(run), "--output-root", str(tmp_path / "root"),
            "--n-null", "7", "--seed", "3"]
    assert backfill.main(argv) == 0

    log = (stub_backfill["output"] / "command.log").read_text(encoding="utf-8")
    fields = dict(line.split(": ", 1) for line in log.splitlines() if ": " in line)
    assert fields["backfill_argv"] == list2cmdline(argv)
    assert float(fields["miat_floor"]) == 500.0
    assert fields["miat_floor_source"] == "run_config.json:config_resolved.output.manual_rna_min"
    assert fields["n_null"] == "7"
    assert fields["seed"] == "3"
    assert fields["hierarchy"] == "native"
    assert fields["python_executable"] == sys.executable
    assert fields["PYTHONPATH"] == str(REPO / "src")


def test_backfill_command_log_names_the_explicit_hierarchy(native_run, stub_backfill, tmp_path):
    run, explicit, _ = native_run
    assert backfill.main(["--run", str(run), "--hierarchy", str(explicit),
                          "--output-root", str(tmp_path / "root"),
                          "--miat-floor", "123.5", "--skip-design-check"]) == 0
    log = (stub_backfill["output"] / "command.log").read_text(encoding="utf-8")
    fields = dict(line.split(": ", 1) for line in log.splitlines() if ": " in line)
    assert fields["hierarchy"] == str(explicit)
    assert float(fields["miat_floor"]) == 123.5
    assert fields["miat_floor_source"] == "--miat-floor"


def test_miat_floor_defaults_to_the_recorded_pin(native_run, stub_backfill, tmp_path):
    run = _run_dir_with_output(
        native_run, {"manual_rna_min": 500.0, "rna_intensity_threshold": 612.0})
    assert backfill.main(["--run", str(run), "--output-root", str(tmp_path / "root")]) == 0
    assert stub_backfill["parameters"].miat_floor_raw == 612.0


def test_miat_floor_literal_fallback_is_warned(native_run, stub_backfill, tmp_path, capsys):
    run = _run_dir_with_output(native_run, {})
    assert backfill.main(["--run", str(run), "--output-root", str(tmp_path / "root")]) == 0
    assert stub_backfill["parameters"].miat_floor_raw == backfill.LEGACY_MIAT_FLOOR_RAW
    out = capsys.readouterr().out
    assert "WARNING" in out and "364" in out


# ---------------------------------------------------------------------- c
def test_display_only_ranges_are_read_from_the_run(native_run, tmp_path):
    run = _run_dir_with_output(native_run, {
        "manual_dapi_min": 334.0, "manual_dapi_max": 5500.0,
        "manual_rna_min": 500.0, "manual_rna_max": 4000.0,
        "manual_antibody_min": 1050.0, "manual_antibody_max": 3000.0,
        "pub_contrast_mode": "manual"})
    _execute(native_run, tmp_path / "out")
    ranges = json.loads(
        (tmp_path / "out" / "analysis_parameters.json").read_text(encoding="utf-8")
    )["display_only_ranges"]
    assert ranges["dapi"] == [334.0, 5500.0]
    assert ranges["rna"] == [500.0, 4000.0]
    assert ranges["partner"] == [1050.0, 3000.0]
    assert ranges["source"] == "run_config.json:config_resolved.output"
    assert ranges["quantitative_effect"] == "none"
    assert run.is_dir()


def test_hardcoded_display_block_is_gone():
    src = BACKFILL_PATH.read_text(encoding="utf-8")
    for literal in ('"miat_primary": [400, 4000]', '"miat_optional": [700, 5000]',
                    '"dapi": [334, 5500]', '"qki": [555, 3000]'):
        assert literal not in src, f"hardcoded display literal survives: {literal}"


# ---------------------------------------------------------------------- d
def test_ortho_versions_txt_uses_the_shared_writer(synthetic_run):
    from click.testing import CliRunner
    from fishsuite.cli import cli

    run, _ = synthetic_run
    out = run / "figures"
    result = CliRunner().invoke(cli, ["ortho", "--run-dir", str(run), "--out", str(out),
                                      "--k", "1", "--seed", "17"])
    assert result.exit_code == 0, result.output
    versions = (out / "versions.txt").read_text(encoding="utf-8")
    for key in ("fishsuite_version:", "fishsuite_git_commit:", "fishsuite_git_dirty:",
                "fishsuite_source_path:", "global_seed: 17", "python:",
                "python_executable:", "numpy:", "pandas:"):
        assert key in versions, f"versions.txt lacks {key}"


def test_ortho_command_log_is_runnable(synthetic_run, monkeypatch):
    from click.testing import CliRunner
    from fishsuite.cli import cli

    run, _ = synthetic_run
    out = run / "figures"
    monkeypatch.setenv("PYTHONPATH", str(REPO / "src"))
    args = ["ortho", "--run-dir", str(run), "--out", str(out), "--k", "1"]
    assert CliRunner().invoke(cli, args).exit_code == 0
    log = (out / "command.log").read_text(encoding="utf-8")
    assert f"command: {list2cmdline(['fishsuite', *args])}" in log
    assert f"python_executable: {sys.executable}" in log
    assert f"PYTHONPATH: {REPO / 'src'}" in log
    line = next(l for l in log.splitlines() if l.startswith("reproduction: "))
    assert line.startswith(f"reproduction: set PYTHONPATH={REPO / 'src'}; ")
    assert "-m fishsuite.cli ortho " in line


def test_reproduction_line_flags_a_console_script_from_another_source(monkeypatch, tmp_path):
    import shutil

    other = tmp_path / "other-env"
    (other / "Scripts").mkdir(parents=True)
    (other / "Scripts" / "fishsuite.exe").write_bytes(b"x")
    monkeypatch.setattr(shutil, "which", lambda name: str(other / "Scripts" / "fishsuite.exe"))
    note = repro.console_script_note()
    assert "fishsuite.exe" in note and "NOT" in note

    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert "console script" in repro.console_script_note()


def test_reproduction_command_is_the_module_form():
    line = repro.reproduction_command(["ortho", "--run-dir", "C:/a path/run"])
    src = str(Path(repro.__file__).resolve().parents[2])
    assert line.startswith(f"set PYTHONPATH={src}; ")
    assert f"{sys.executable}" in line
    assert "-m fishsuite.cli ortho" in line
    assert '"C:/a path/run"' in line


# ---------------------------------------------------------------------- e
def test_overdetect_advisory_states_the_arm_consequence():
    from fishsuite.core.qc import OVERDETECT_ADVISORY

    assert "advisory" in OVERDETECT_ADVISORY
    assert "induced/overexpression arm" in OVERDETECT_ADVISORY
    assert "filtering on qc_pass drops the entire arm" in OVERDETECT_ADVISORY
    assert "do not filter on it for arm comparisons" in OVERDETECT_ADVISORY


def test_flagged_rows_carry_the_advisory_into_the_run_summary():
    from fishsuite.core.qc import OVERDETECT_ADVISORY, flag_overdetect_outliers

    cfg = FishsuiteConfig()
    rows = [{"qc_rna1_spots_per_nucleus": v, "qc_flags": "", "qc_pass": True}
            for v in [38.0, 41.0, 39.0, 42.0, 40.0, 37.0]]
    rows.append({"qc_rna1_spots_per_nucleus": 900.0, "qc_flags": "", "qc_pass": True})
    assert flag_overdetect_outliers(rows, cfg) == 1
    assert rows[-1]["qc_overdetect_advisory"] == OVERDETECT_ADVISORY
    assert all(r["qc_overdetect_advisory"] == "" for r in rows[:-1])


def test_runner_warning_carries_the_advisory():
    src = (REPO / "src" / "fishsuite" / "runner.py").read_text(encoding="utf-8")
    assert "_OVERDETECT_ADVISORY" in src, "runner warning does not use the shared advisory"


# ---------------------------------------------------------------------- f
def test_well_key_comes_from_the_recorded_hierarchy(panel):
    per_nucleus = pd.DataFrame([
        dict(line="OE", condition="g2_Dox", well_id="g2_Dox_01", secondary_only=False,
             image="a.vsi", nucleus_id=1, pearson_r=0.10),
        dict(line="OE", condition="g2_Dox", well_id="g2_Dox_02", secondary_only=False,
             image="b.vsi", nucleus_id=1, pearson_r=0.20),
        dict(line="WT", condition="g2_NoDox", well_id="g2_NoDox_01", secondary_only=False,
             image="c.vsi", nucleus_id=1, pearson_r=0.30),
        dict(line="WT", condition="g2_NoDox", well_id="g2_NoDox_02", secondary_only=False,
             image="d.vsi", nucleus_id=1, pearson_r=0.40),
    ])
    _fov, well = panel.rollup(per_nucleus)
    assert sorted(well["well_id"]) == ["g2_Dox_01", "g2_Dox_02", "g2_NoDox_01", "g2_NoDox_02"]
    assert int((well["line"] == "OE").sum()) == 2
    assert "condition" in well.columns


def test_run_well_ids_load_from_resolved_experiment_hierarchy(panel, tmp_path):
    hierarchy = tmp_path / "resolved_experiment_hierarchy.csv"
    pd.DataFrame([{"image": "a.vsi", "condition": "g2_Dox", "well_id": "g2_Dox_01"},
                  {"image": "b.vsi", "condition": "g2_Dox", "well_id": "g2_Dox_02"}]
                 ).to_csv(hierarchy, index=False)
    mapping, source = panel.recorded_well_ids(tmp_path)
    assert mapping == {"a.vsi": "g2_Dox_01", "b.vsi": "g2_Dox_02"}
    assert source == "resolved_experiment_hierarchy.csv:well_id"

    hierarchy.unlink()
    mapping, source = panel.recorded_well_ids(tmp_path)
    assert mapping == {}
    assert source == "per_image_summary.csv:condition"


@pytest.mark.parametrize("n,expected", [(2, 5.65), (3, 3.07), (5, 2.02)])
def test_mde_is_the_effect_detectable_at_eighty_percent_power(panel, n, expected):
    mde = panel.mde_hedges_g(n1=n, n2=n)
    assert mde == pytest.approx(expected, abs=0.01)
    assert panel._power_two_sided(mde, 0.05, n, n) == pytest.approx(0.80, abs=1e-3)


def test_single_well_arms_get_nan_not_a_fake_floor(panel):
    assert np.isnan(panel.mde_hedges_g(n1=1, n2=1))
    assert np.isnan(panel.mde_hedges_g(n1=2, n2=1))
    assert panel.mde_reason(1, 1) == "TOO_FEW_WELLS"
    assert panel.mde_reason(2, 2) == ""


def test_nan_primary_carries_its_reason(panel):
    endpoints = {c: np.nan for c, _ in panel.ALL_ENDPOINTS}
    per_nucleus = pd.DataFrame([
        dict(line="OE", condition="g2_Dox", well_id="g2_Dox_01", secondary_only=False,
             image="a.vsi", nucleus_id=1, **{**endpoints, "frac_called_coloc_runthr": 0.10}),
        dict(line="WT", condition="g2_NoDox", well_id="g2_NoDox_01", secondary_only=False,
             image="c.vsi", nucleus_id=1, **{**endpoints, "frac_called_coloc_runthr": 0.30}),
    ])
    _fov, well = panel.rollup(per_nucleus)
    panel.ARMS[:] = ["WT", "OE"]
    contrasts = panel.contrasts_table(well, per_nucleus, float("nan"),
                                      "frac_called_coloc_runthr", "frac_called_coloc")
    row = contrasts.loc[contrasts.endpoint == "frac_called_coloc_runthr"].iloc[0]
    assert not np.isfinite(row["p_welch"])
    assert row["nan_reason"].startswith("TOO_FEW_WELLS")
    assert "n_wells_test=1" in row["nan_reason"]
    assert np.isnan(row["mde_hedges_g_alpha_0p05_power_0p80"])
    assert "nan_reason" in contrasts.columns


def test_readme_states_the_primary_nan_reason(panel):
    rows = panel.primary_readme_rows(
        "frac_called_coloc_runthr", "TOO_FEW_WELLS: n_wells_test=1, n_wells_ref=1")
    joined = ["{} | {}".format(item, definition) for item, definition in rows]
    assert any("frac_called_coloc_runthr" in text and "TOO_FEW_WELLS" in text
               and "not read that NaN as 'no difference'" in text for text in joined)
    assert any("PRIMARY endpoint NOT testable" in item for item, _ in rows)
    assert panel.primary_readme_rows("frac_called_coloc_runthr", "") == []


# ---------------------------------------------------------------------- g
def test_command_log_carries_a_quoted_run_command(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["fishsuite", "run", "--config",
                                      r"F:\Image Analysis Work\a preset.yaml"])
    assert repro.write_command_log(tmp_path, "cfg.yaml", tmp_path, 0)
    log = (tmp_path / "command.log").read_text(encoding="utf-8")
    line = next(l for l in log.splitlines() if l.startswith("run_command: "))
    assert line == "run_command: " + list2cmdline(sys.argv)
    assert '"F:\\Image Analysis Work\\a preset.yaml"' in line


def test_run_config_records_the_preset_hash(tmp_path, monkeypatch):
    from fishsuite import runner as _runner

    (tmp_path / "KO").mkdir()
    (tmp_path / "KO" / "imgA.vsi").write_bytes(b"x")

    def _fake_mode(p, **kw):
        raise RuntimeError(p.name)

    monkeypatch.setattr(_runner, "get_mode", lambda _m: _fake_mode)
    cfg = FishsuiteConfig()
    cfg.pixel_coloc.threshold_scope = "per_image"
    cfg_path = tmp_path / "cfg.yaml"
    cfg.dump_yaml(cfg_path)
    out = tmp_path / "out"
    try:
        _runner.run_batch(cfg_path, tmp_path, out, parallel=1, dry_run=False)
    except Exception:
        pass
    expected = hashlib.sha256(cfg_path.read_bytes()).hexdigest()
    assert json.loads((out / "run_config.json").read_text(encoding="utf-8")
                      )["preset_sha256"] == expected
    assert f"preset_sha256: {expected}" in (out / "command.log").read_text(encoding="utf-8")
