"""Astra F9: wells must be a composite (biological_set, slide, well) or a
validated globally unique well_id; ambiguous reuse fails loudly."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from fishsuite.core.well_key import resolve_well_ids

PANEL_PATH = Path(__file__).resolve().parents[1] / "scripts" / "coloc_standard_panel.py"


def _frame(wells, conditions, **extra):
    return pd.DataFrame(dict(well=wells, condition=conditions,
                             image=[f"i{i}.vsi" for i in range(len(wells))], **extra))


def test_unique_well_ids_pass_and_are_unchanged():
    keys, source = resolve_well_ids(_frame(["A1", "A2", "B1"], ["c", "c", "t"]))
    assert keys.tolist() == ["A1", "A2", "B1"]
    assert source == "well (validated globally unique)"


def test_well_label_reused_across_conditions_fails_loudly():
    with pytest.raises(ValueError, match="ambiguous well"):
        resolve_well_ids(_frame(["A1", "A1"], ["control", "treated"]))


def test_composite_key_disambiguates_reused_labels():
    frame = _frame(["A1", "A1", "A1"], ["control", "treated", "control"],
                   biological_set=["s1", "s1", "s2"], slide=["1", "2", "1"])
    keys, source = resolve_well_ids(frame)
    assert keys.tolist() == ["s1|1|A1", "s1|2|A1", "s2|1|A1"]
    assert source == "biological_set|slide|well"


def test_composite_key_still_one_condition_per_physical_well():
    frame = _frame(["A1", "A1"], ["control", "treated"], biological_set=["s1", "s1"], slide=["1", "1"])
    with pytest.raises(ValueError, match="ambiguous well"):
        resolve_well_ids(frame)


def test_coupling_rejects_reused_well_label():
    from fishsuite.report.coupling_stats import prepare_data
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from test_coupling_report import toy_table
    df = toy_table()
    df.loc[df.well.eq("T1"), "well"] = "C1"
    with pytest.raises(ValueError, match="ambiguous well"):
        prepare_data(df)


def test_panel_hierarchy_rejects_reused_well_id(tmp_path):
    spec = importlib.util.spec_from_file_location("_panel_wellkey", PANEL_PATH)
    panel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(panel)
    pd.DataFrame([{"image": "a.vsi", "condition": "g2_Dox", "well_id": "W1"},
                  {"image": "b.vsi", "condition": "g2_NoDox", "well_id": "W1"}]
                 ).to_csv(tmp_path / "resolved_experiment_hierarchy.csv", index=False)
    with pytest.raises(ValueError, match="ambiguous well"):
        panel.recorded_well_ids(tmp_path)
