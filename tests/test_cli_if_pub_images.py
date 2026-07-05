"""CLI layer for the ``if-pub-images`` post-run utility (2026-07-05).

A first-class ``fishsuite`` post-run subcommand alongside ``backfill`` /
``walkthrough`` / ``postrun`` / ``singlecell`` / ``pixelpattern``:

    fishsuite if-pub-images --run <run_dir> [--staging ... --zstack ...
                                             --source ... --floor SEC=VALUE
                                             --ceiling-pct ... --scalebar ...
                                             --label ...]

These tests prove the CLI WIRING (Click arg parsing, defaults, the repeatable
--source / --floor options, the SEC=VALUE floor parsing, plain-English errors,
and that the subcommand dispatches to the correct underlying function with the
right kwargs) WITHOUT touching any image / GPU: the heavy renderer
(``regenerate_pub_images``) is monkeypatched, so we assert on the kwargs it
receives, not on real I/O.
"""
from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from fishsuite import cli as _cli


def _make_run_dir(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    (run / "per_well.csv").write_text(
        "well,genotype,arm,secondary,qki_channel\n8,WT,primary,647,640\n"
    )
    (run / "per_fov.csv").write_text("well,file,nucleus_count\n8,a.vsi,10\n")
    return run


# ===========================================================================
# registration / help
# ===========================================================================
def test_if_pub_images_registered():
    res = CliRunner().invoke(_cli.cli, ["--help"])
    assert res.exit_code == 0
    assert "if-pub-images" in res.output


def test_if_pub_images_help_states_cpu_no_gpu_and_options():
    res = CliRunner().invoke(_cli.cli, ["if-pub-images", "--help"])
    assert res.exit_code == 0
    low = res.output.lower()
    assert "cpu" in low and "gpu" in low
    assert "no mip" in low or "max-projection" in low
    for opt in ("--staging", "--zstack", "--source", "--floor", "--label", "--run"):
        assert opt in res.output


# ===========================================================================
# dispatch
# ===========================================================================
def test_if_pub_images_dispatches_with_defaults(tmp_path, monkeypatch):
    run = _make_run_dir(tmp_path)
    captured = {}

    def fake_run(run_dir, **kw):
        captured["run_dir"] = str(run_dir)
        captured.update(kw)
        return {"out_dir": str(run), "channels": 24, "merge": 24,
                "composite": 4, "sources": ["single_plane", "picked_z"]}

    monkeypatch.setattr(_cli, "_if_pub_images_run", fake_run, raising=False)
    res = CliRunner().invoke(_cli.cli, ["if-pub-images", "--run", str(run)])
    assert res.exit_code == 0, res.output
    assert captured["run_dir"] == str(run)
    # unset options pass through as None / empty so the renderer applies its
    # if_run_context.json / built-in defaults.
    assert captured["staging_dir"] is None and captured["zstack_dir"] is None
    assert captured["sources"] is None
    assert captured["floors"] is None
    assert captured["ceiling_pct"] is None
    assert captured["scalebar_um"] is None
    assert captured["label"] is None


def test_if_pub_images_passes_sources_floors_and_label(tmp_path, monkeypatch):
    run = _make_run_dir(tmp_path)
    staging = tmp_path / "stg"; staging.mkdir()
    zstack = tmp_path / "zs"; zstack.mkdir()
    captured = {}

    def fake_run(run_dir, **kw):
        captured.update(kw)
        return {"out_dir": str(run), "channels": 2, "merge": 2, "composite": 1,
                "sources": ["single_plane"]}

    monkeypatch.setattr(_cli, "_if_pub_images_run", fake_run, raising=False)
    res = CliRunner().invoke(_cli.cli, [
        "if-pub-images", "--run", str(run), "--staging", str(staging),
        "--zstack", str(zstack), "--source", "single_plane",
        "--source", "picked_z", "--floor", "647=5000", "--floor", "568=3500",
        "--ceiling-pct", "99.9", "--scalebar", "25", "--label", "QKI",
    ])
    assert res.exit_code == 0, res.output
    assert captured["staging_dir"] == str(staging)
    assert captured["zstack_dir"] == str(zstack)
    assert captured["sources"] == ["single_plane", "picked_z"]
    assert captured["floors"] == {"647": 5000.0, "568": 3500.0}
    assert captured["ceiling_pct"] == 99.9
    assert captured["scalebar_um"] == 25.0
    assert captured["label"] == "QKI"


def test_if_pub_images_bad_floor_is_rejected(tmp_path, monkeypatch):
    run = _make_run_dir(tmp_path)
    monkeypatch.setattr(_cli, "_if_pub_images_run",
                        lambda *a, **k: {}, raising=False)
    res = CliRunner().invoke(_cli.cli, [
        "if-pub-images", "--run", str(run), "--floor", "647:5000",
    ])
    assert res.exit_code != 0
    assert "SEC=VALUE" in res.output


def test_if_pub_images_missing_per_well_is_friendly(tmp_path, monkeypatch):
    run = _make_run_dir(tmp_path)

    def fake_run(run_dir, **kw):
        raise FileNotFoundError("no per_well.csv in run")

    monkeypatch.setattr(_cli, "_if_pub_images_run", fake_run, raising=False)
    res = CliRunner().invoke(_cli.cli, ["if-pub-images", "--run", str(run)])
    assert res.exit_code != 0
    assert "Traceback" not in res.output
    assert "per_well" in res.output.lower()
