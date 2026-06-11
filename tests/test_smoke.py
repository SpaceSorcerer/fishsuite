"""Smoke test — import the package."""

def test_import():
    import fishsuite
    assert fishsuite.__version__


def test_schema_h9_preset_loads():
    from pathlib import Path
    from fishsuite.config.schema import FishsuiteConfig
    p = Path(__file__).resolve().parents[1] / "src" / "fishsuite" / "config" / "presets" / "h9_hesc_100x.yaml"
    cfg = FishsuiteConfig.from_yaml(p)
    assert cfg.channels.analysis_mode == "rna_only"
    assert cfg.channels.dapi == 2
    assert cfg.channels.rna == 0


def test_metrics_pearson_identity():
    """Identical channels -> Pearson = 1, ICQ = 0.25 (numerical), Manders = 1."""
    import numpy as np
    from fishsuite.core.metrics import compute_coloc_metrics
    rng = np.random.default_rng(7)
    a = rng.integers(0, 4096, size=1000).astype(np.float64)
    m = compute_coloc_metrics(a, a)
    assert abs(m["pearson_r"] - 1.0) < 1e-9
    # When channels are identical, every pixel is on the y=x line -> sum_min = sum_r = sum_a
    assert abs(m["sum_min"] - m["sum_r"]) < 1e-6
    assert abs(m["min_frac_r"] - 1.0) < 1e-9


def test_metrics_orthogonal_pearson_zero():
    import numpy as np
    from fishsuite.core.metrics import compute_coloc_metrics
    n = 5000
    rng = np.random.default_rng(0)
    a = rng.normal(1000, 100, n)
    b = rng.normal(1000, 100, n)
    m = compute_coloc_metrics(a, b)
    assert abs(m["pearson_r"]) < 0.05


def test_fluorophore_for_channel_name_known_lasers():
    """640 CSU -> Cy5; 405 CSU -> DAPI; 561 -> Cy3; 488 -> GFP/AF488."""
    from fishsuite.gui.state import (
        fluorophore_for_channel_name, format_channel_metadata_row,
    )
    info = fluorophore_for_channel_name("640 CSU")
    assert info["laser_nm"] == 640
    assert "Cy5" in info["fluor"]
    assert info["em_nm"] == 668
    info = fluorophore_for_channel_name("405 CSU")
    assert info["laser_nm"] == 405
    assert "DAPI" in info["fluor"]
    info = fluorophore_for_channel_name("Cy3 561")
    assert info["laser_nm"] == 561
    info = fluorophore_for_channel_name("488 nm")
    assert info["laser_nm"] == 488
    # Unknown channel name falls back gracefully.
    info = fluorophore_for_channel_name("Brightfield")
    assert info["laser_nm"] is None
    assert info["fluor"] is None
    # format function adds fluor + emission when known.
    s = format_channel_metadata_row("640 CSU")
    assert "640 CSU" in s
    assert "668" in s


def test_scan_input_dir_tree_subfolder_layout(tmp_path):
    """scan_input_dir_tree walks one level deep and surfaces .vsi/.czi leaves."""
    from fishsuite.gui.state import scan_input_dir_tree
    (tmp_path / "KO").mkdir()
    (tmp_path / "WT").mkdir()
    (tmp_path / "KO" / "image01.vsi").write_bytes(b"x")
    (tmp_path / "KO" / "image02.vsi").write_bytes(b"x")
    (tmp_path / "WT" / "image03.vsi").write_bytes(b"x")
    # Unsupported extension should be ignored.
    (tmp_path / "WT" / "ignore.txt").write_bytes(b"x")
    # Underscore-prefixed dir should be skipped.
    (tmp_path / "_seg_cache").mkdir()
    (tmp_path / "_seg_cache" / "ignored.vsi").write_bytes(b"x")

    tree = scan_input_dir_tree(str(tmp_path))
    sub_names = sorted(s["name"] for s in tree["subfolders"])
    assert sub_names == ["KO", "WT"]
    ko = next(s for s in tree["subfolders"] if s["name"] == "KO")
    assert sorted(f["name"] for f in ko["files"]) == ["image01.vsi", "image02.vsi"]
    wt = next(s for s in tree["subfolders"] if s["name"] == "WT")
    assert [f["name"] for f in wt["files"]] == ["image03.vsi"]
    assert tree["root_files"] == []


def test_runner_input_file_subset_filter(tmp_path, monkeypatch):
    """runner.run_batch filters discovered images by cfg.input_file_subset."""
    # Stand up a tiny fake input dir with KO + WT subfolders. We don't run
    # the actual pipeline (no real VSI data); just check that the filter
    # logic in run_batch keeps only the requested file and raises a clear
    # error when the subset matches zero files.
    from fishsuite.config.schema import FishsuiteConfig
    from fishsuite import runner as _runner

    (tmp_path / "KO").mkdir()
    (tmp_path / "WT").mkdir()
    (tmp_path / "KO" / "imgA.vsi").write_bytes(b"x")
    (tmp_path / "KO" / "imgB.vsi").write_bytes(b"x")
    (tmp_path / "WT" / "imgC.vsi").write_bytes(b"x")

    # We intercept the work AFTER subset filtering so we don't actually open
    # the dummy .vsi files. Patch the mode dispatch to raise a sentinel.
    class _Sentinel(Exception):
        pass

    seen: list = []

    def _fake_mode(p, **kw):  # pragma: no cover - only path-collection matters
        seen.append(p.name)
        raise _Sentinel(p.name)

    monkeypatch.setattr(_runner, "get_mode", lambda _m: _fake_mode)

    # Write a minimal config to YAML with input_file_subset set. Disable the
    # batch threshold pre-pass so the runner doesn't try to open the dummy
    # .vsi files — only the subset-filter logic is under test here.
    cfg = FishsuiteConfig()
    cfg.input_file_subset = ["imgA.vsi"]
    cfg.pixel_coloc.threshold_scope = "per_image"
    cfg_path = tmp_path / "cfg.yaml"
    cfg.dump_yaml(cfg_path)

    try:
        _runner.run_batch(cfg_path, tmp_path, tmp_path / "out", parallel=1, dry_run=False)
    except Exception:
        # The mode fn raises Sentinel inside the per-image loop; the runner
        # records it as a failure and keeps going, eventually finishing.
        pass

    # Only imgA.vsi should have been dispatched.
    assert seen == ["imgA.vsi"], seen

    # Now subset that matches nothing -> RuntimeError.
    cfg.input_file_subset = ["doesnotexist.vsi"]
    cfg.dump_yaml(cfg_path)
    import pytest
    with pytest.raises(RuntimeError, match="matched 0"):
        _runner.run_batch(cfg_path, tmp_path, tmp_path / "out2", parallel=1, dry_run=False)


def test_discover_filename_conditions_flat_folder(tmp_path):
    """2026-05-31: flat-folder, filename-encoded condition assignment.

    Mirrors the H9 05-05 session layout: a single flat folder whose condition
    lives in the file NAME (NT / MIAT-KD / Sec-only). Asserts:
      * the ordered filename_conditions substring map splits NT vs KD,
      * sec_only_files still forces "Sec-Only" + sec_only=True and is NOT
        relabelled by the filename map (even though it shares the prefix),
      * an empty filename_conditions map = legacy behaviour (single condition).
    """
    from fishsuite.core.io import discover_inputs

    # Flat folder, filename-labelled (no subdirs -> flat mode).
    for nm in (
        "H9-X-ASO-NT_02.vsi", "H9-X-ASO-NT_03.vsi",
        "H9-X-ASO-MIAT-KD_05.vsi", "H9-X-ASO-MIAT-KD_06.vsi",
        "H9-X-Sec-only_10.vsi",
    ):
        (tmp_path / nm).write_bytes(b"x")

    imgs = discover_inputs(
        tmp_path,
        sec_only_files=["sec-only"],
        filename_conditions=[["-nt_", "NT ASO"], ["-miat-kd_", "KD ASO"]],
    )
    by = {i.path.name: i for i in imgs}
    assert by["H9-X-ASO-NT_02.vsi"].condition == "NT ASO"
    assert by["H9-X-ASO-NT_02.vsi"].sec_only is False
    assert by["H9-X-ASO-MIAT-KD_05.vsi"].condition == "KD ASO"
    # Sec-only must win and NOT be relabelled by the filename map.
    assert by["H9-X-Sec-only_10.vsi"].condition == "Sec-Only"
    assert by["H9-X-Sec-only_10.vsi"].sec_only is True

    from collections import Counter
    counts = Counter(i.condition for i in imgs)
    assert counts == {"NT ASO": 2, "KD ASO": 2, "Sec-Only": 1}

    # Back-compat: no filename_conditions -> every non-sec file gets the single
    # flat-mode condition (empty string), legacy behaviour preserved.
    imgs2 = discover_inputs(tmp_path, sec_only_files=["sec-only"])
    nonsec2 = {i.condition for i in imgs2 if not i.sec_only}
    assert nonsec2 == {""}
