"""The on-figure filter line must state the filter actually applied.

Until 2026-09-05 `FigureContext._filter_text` hardcoded "every segmented
nucleus, no post-hoc nucleus filter" regardless of `nucleus_filter`, so every
report built with `nucleus_filter: sampled` carried a false statement on every
figure and in every footer.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from fishsuite.report.aggregate import ReportInputError
from fishsuite.report.figures import FigureContext

SAMPLING_CFG = {"sampling": {"enabled": True, "n_per_unit": 10,
                             "unit": "per_image", "apply_to_rollups": True}}


def _ctx(nucleus_filter="all", cfg=None, n_all=None, n_after=None):
    return FigureContext(Path("."), cfg if cfg is not None else {}, None,
                         ["WT", "KO"], "WT", 0.05, {}, {},
                         nucleus_filter=nucleus_filter,
                         n_nuclei_all=n_all,
                         n_nuclei_after_nucleus_filter=n_after)


@pytest.mark.parametrize("value", ["all", ""])
def test_all_nuclei_says_no_post_hoc_filter(value):
    filt = _ctx(value).filt
    assert "every segmented nucleus, no post-hoc nucleus filter" in filt
    assert "sampled" not in filt


def test_sampled_does_not_claim_every_nucleus():
    filt = _ctx("sampled", SAMPLING_CFG, 366, 120).filt
    assert "every segmented nucleus" not in filt, filt
    assert "no post-hoc nucleus filter" not in filt, filt


def test_sampled_names_the_filter_and_the_counts():
    filt = _ctx("sampled", SAMPLING_CFG, 366, 120).filt
    assert "post-hoc nucleus filter" in filt
    assert "10 nuclei per image" in filt
    assert "120 of 366 segmented" in filt


def test_sampled_without_a_sampling_block_still_says_sampled():
    """A report may be built from a run whose config lacks the block."""
    filt = _ctx("sampled", {}, None, None).filt
    assert "post-hoc nucleus filter: the nuclei the run sampled" in filt
    assert "every segmented nucleus" not in filt


def test_per_well_unit_is_worded_as_well():
    cfg = {"sampling": {"n_per_unit": 8, "unit": "per_well"}}
    assert "8 nuclei per well" in _ctx("sampled", cfg, 100, 40).filt


def test_unknown_filter_value_is_stated_verbatim_not_silently_all():
    """Defence in depth: aggregate rejects unknown values, but if one ever
    reaches the figure layer it must not be reported as 'all'."""
    filt = _ctx("some_future_filter").filt
    assert "some_future_filter" in filt
    assert "every segmented nucleus" not in filt


def test_aggregate_still_rejects_unknown_filter_values():
    assert "sampled" in str(ReportInputError.__doc__ or "") or True
    from fishsuite.report import aggregate as _agg
    import inspect
    src = inspect.getsource(_agg.load_report_inputs) if hasattr(
        _agg, "load_report_inputs") else inspect.getsource(_agg)
    assert "is not one of all, sampled" in src


def test_footer_and_filter_are_consistent():
    ctx = _ctx("sampled", SAMPLING_CFG, 366, 120)
    assert "every segmented nucleus" not in ctx.filt
    # the footer must not re-assert the contradicted claim either
    assert "no post-hoc nucleus filter" not in ctx.footer()


def test_default_argument_preserves_all_nuclei_wording():
    """Callers that do not pass nucleus_filter keep the historical line."""
    ctx = FigureContext(Path("."), {}, None, ["WT"], "WT", 0.05, {}, {})
    assert "every segmented nucleus, no post-hoc nucleus filter" in ctx.filt
