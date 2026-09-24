"""Regression: the GUI condition path must resolve the same roster as the runner.

2026-09-24 audit. Before the fix, gui/main.py::_hierarchy_roster and
gui/readiness.py::conditions_status called discover_inputs WITHOUT
exclude_subfolders / strict_subfolders / strict_filenames, so an excluded
folder re-entered the GUI roster as an unassigned biological well and the GUI
refused to launch a run the CLI accepts. Separately, strict_subfolders raised
on a sec_only_folders entry that (per the documented rule) must not be listed
in subfolder_conditions.
"""
import logging

import pytest

from fishsuite.config.schema import ConditionsCfg
from fishsuite.core.io import discover_inputs

TREE = ["A/img_01.tif", "A/img_02.tif", "B/img_03.tif", "B/img_04.tif",
        "C_unlisted/img_05.tif", "SecOnly/img_06.tif"]
GROUPED = dict(mode="subfolders", subfolder_conditions={"A": "Cond A", "B": "Cond B"},
               sec_only_folders=["SecOnly"], exclude_subfolders=["C_unlisted"],
               groups={"Cond A": ["Cond A"], "Cond B": ["Cond B"]})


def _tree(root):
    for name in TREE:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return root


def _runner_roster(root, block):
    from fishsuite.config.hierarchy import discovery_roster
    from fishsuite.core.io import discover_from_conditions
    conditions = ConditionsCfg.model_validate(block)
    return discovery_roster(discover_from_conditions(root, conditions), root, conditions)


def test_readiness_honours_exclude_subfolders(tmp_path):
    from fishsuite.gui.readiness import conditions_status
    root = _tree(tmp_path / "images")
    assert conditions_status({"conditions": GROUPED}, input_dir=str(root)) == "green"


def test_readiness_reports_red_when_runner_would_reject(tmp_path):
    from fishsuite.gui.readiness import conditions_status
    root = _tree(tmp_path / "images")
    groups = dict(GROUPED["groups"], junk=["C_unlisted"])
    block = dict(GROUPED, exclude_subfolders=[], strict_subfolders=True, groups=groups)
    assert conditions_status({"conditions": block}, input_dir=str(root)) == "red"


def test_runner_roster_excludes_folder_and_keeps_controls(tmp_path):
    root = _tree(tmp_path / "images")
    roster = _runner_roster(root, GROUPED)
    assert sorted(roster.image) == ["img_01.tif", "img_02.tif", "img_03.tif", "img_04.tif", "img_06.tif"]
    bio = roster.loc[~roster.secondary_only]
    assert bio.groupby("group").size().to_dict() == {"Cond A": 2, "Cond B": 2}
    assert roster.loc[roster.secondary_only, "group"].tolist() == ["Secondary-only"]


def test_strict_subfolders_accepts_unmapped_sec_only_folder(tmp_path, caplog):
    root = _tree(tmp_path / "images")
    with caplog.at_level(logging.WARNING):
        images = discover_inputs(root, subfolder_conditions={"A": "Cond A", "B": "Cond B"},
                                 sec_only_folders=["SecOnly"], exclude_subfolders=["C_unlisted"],
                                 strict_subfolders=True)
    by_name = {im.path.name: im for im in images}
    assert by_name["img_06.tif"].sec_only and by_name["img_06.tif"].condition == "Sec-Only"
    assert "SecOnly" not in caplog.text
    with pytest.raises(ValueError, match="C_unlisted"):
        discover_inputs(root, subfolder_conditions={"A": "Cond A", "B": "Cond B"},
                        sec_only_folders=["SecOnly"], strict_subfolders=True)


def test_gui_preview_matches_runner_roster(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    import importlib
    from PySide6.QtWidgets import QApplication
    main = importlib.import_module("fishsuite.gui.main")
    monkeypatch.setattr(main._state, "load_settings", lambda: {})
    monkeypatch.setattr(main._state, "save_settings", lambda *_: None)
    app = QApplication.instance() or QApplication([])
    window = main.FishsuiteWindow()
    root = _tree(tmp_path / "images")
    window._cfg["conditions"].update(GROUPED)
    window._cfg_to_widgets()
    window.input_edit.setText(str(root))
    assert window._preview_hierarchy(), window.hierarchy_status.text()
    gui = window._hierarchy_roster()
    runner = _runner_roster(root, GROUPED)
    assert gui[["image", "well_id", "group"]].equals(runner[["image", "well_id", "group"]])
    window.close()
    app.processEvents()
