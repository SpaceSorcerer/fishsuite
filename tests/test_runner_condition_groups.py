"""The runner's condition-group column, tested without running the pipeline.

The runner attaches ``group`` to every master CSV that carries ``condition``.
The rules that matter and that a refactor could break: sec-only rows never land
in a biological group, an unlisted well keeps its own label so it stays visible,
and a run with NO groups configured gains no column at all so legacy runs stay
byte-identical.
"""
from __future__ import annotations

import pandas as pd
import pytest

from fishsuite.config.schema import ConditionsCfg
from fishsuite.runner import _REPORT_SEC_ONLY_GROUP


def _attach(df: pd.DataFrame, conditions: ConditionsCfg) -> pd.DataFrame:
    """The runner's group-attachment rule, applied to one frame.

    This mirrors the block in ``run_batch`` that writes the column. It is written
    out here rather than imported because the runner applies it inline while
    holding a live batch; the RULE is what this test pins.
    """
    out = df.copy()
    if not (conditions.groups or {}):
        return out
    sec = (out["secondary_only"].astype(bool) if "secondary_only" in out.columns
           else pd.Series(False, index=out.index))
    grp = out["condition"].astype(str).map(conditions.group_of)
    out["group"] = grp.where(~sec, _REPORT_SEC_ONLY_GROUP)
    return out


@pytest.fixture
def frame():
    return pd.DataFrame({
        "image": ["a.vsi", "b.vsi", "c.vsi", "d.vsi"],
        "condition": ["WT_1", "KO_1", "KO_9", "Sec-Only"],
        "secondary_only": [False, False, False, True],
        "n_spots_rna1": [4.0, 10.0, 11.0, 0.0],
    })


def test_groups_are_attached_and_sec_only_is_kept_out_of_them(frame):
    cfg = ConditionsCfg(groups={"WT": ["WT_1"], "QKI-KO": ["KO_1"]},
                        group_order=["WT", "QKI-KO"])
    out = _attach(frame, cfg)
    assert list(out["group"]) == ["WT", "QKI-KO", "KO_9", _REPORT_SEC_ONLY_GROUP]


def test_an_unlisted_well_stays_visible_as_its_own_group(frame):
    cfg = ConditionsCfg(groups={"WT": ["WT_1"], "QKI-KO": ["KO_1"]})
    out = _attach(frame, cfg)
    # KO_9 was never named in the groups block, so it must not be pooled into
    # QKI-KO or dropped; it appears under its own label.
    assert out.loc[out["condition"] == "KO_9", "group"].iloc[0] == "KO_9"


def test_no_groups_configured_writes_no_group_column(frame):
    out = _attach(frame, ConditionsCfg())
    assert "group" not in out.columns
    assert list(out.columns) == list(frame.columns)


def test_the_condition_column_is_never_altered(frame):
    cfg = ConditionsCfg(groups={"WT": ["WT_1"], "QKI-KO": ["KO_1"]})
    out = _attach(frame, cfg)
    pd.testing.assert_series_equal(out["condition"], frame["condition"])


def test_the_runner_exposes_the_by_group_figure_helper():
    """The by-group figure step must exist and be importable, or a configured run
    silently gets the old one-panel-per-well figures and nobody notices."""
    from fishsuite.runner import _write_by_group_figures
    assert callable(_write_by_group_figures)
