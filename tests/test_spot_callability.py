"""Spot-callability diagnostic — is the detector actually discriminating?

Encodes a hard-won lab result as a guardrail. ``foci.detect_in_sec_only`` already
runs the SAME detector on the secondary-only (no-probe) control images, and both
count sets already flow through ``per_image_summary`` — so the one division that
reveals whether a channel has any thresholdable object at all was computable, and
nobody saw it.

On this lab's own prior data a diffuse, nucleoplasm-filling antibody channel gave
roughly 200 "spots"/nucleus in sample and 218 in the secondary-only control: a
ratio near 0.9, i.e. no discrimination whatsoever. That is why mask-based
colocalization (Manders, ICQ) washed out on that partner — the mask was
background texture. The diagnostic now says so out loud.

It WARNS and never auto-switches: silently reinterpreting a user's declared
punctate channel as diffuse would be a worse failure than the one being warned
about.

Synthetic row dicts only — no images, no pipeline run needed.
"""
from __future__ import annotations

import numpy as np
import pytest

from fishsuite.config.schema import FishsuiteConfig
from fishsuite.core.qc import (
    flag_spot_callability,
    spot_callability_channels,
)


def _cfg(*, detect_in_sec_only=True, min_ratio=2.0) -> FishsuiteConfig:
    cfg = FishsuiteConfig()
    cfg.foci.detect_in_sec_only = detect_in_sec_only
    cfg.foci.min_spot_signal_to_control = min_ratio
    return cfg


def _row(image, *, sec_only, rna1, rna2=None):
    r = {
        "image": image,
        "secondary_only": sec_only,
        "mean_spots_per_nucleus_rna1": rna1,
    }
    if rna2 is not None:
        r["mean_spots_per_nucleus_rna2"] = rna2
    return r


# ---------------------------------------------------------------------------
# DEFAULTS
# ---------------------------------------------------------------------------
def test_default_threshold_is_two():
    assert FishsuiteConfig().foci.min_spot_signal_to_control == 2.0


def test_channel_discovery_reads_the_keys_not_a_hard_coded_list():
    """rna_protein relabels rna2 -> protein, so the channel set must come from
    the emitted keys or the diagnostic silently skips protein channels."""
    rows = [
        {"mean_spots_per_nucleus_rna1": 1.0, "mean_spots_per_nucleus_protein": 2.0},
        {"mean_spots_per_nucleus_rna1": 1.0},
    ]
    assert spot_callability_channels(rows) == ["protein", "rna1"]


# ---------------------------------------------------------------------------
# WHEN THE DIAGNOSTIC DOES NOT APPLY
# ---------------------------------------------------------------------------
def test_no_seconly_images_means_no_diagnostic():
    rows = [_row("a.tif", sec_only=False, rna1=30.0)]
    assert flag_spot_callability(rows, _cfg()) == []
    assert "spot_rate_signal_to_control_rna1" not in rows[0]


def test_no_sample_images_means_no_diagnostic():
    rows = [_row("sec.tif", sec_only=True, rna1=5.0)]
    assert flag_spot_callability(rows, _cfg()) == []


def test_detect_in_sec_only_off_means_no_diagnostic():
    """Without it the controls skip detection and report a structural zero, so
    every ratio would be infinite — reassuring and meaningless."""
    rows = [
        _row("a.tif", sec_only=False, rna1=30.0),
        _row("sec.tif", sec_only=True, rna1=0.0),
    ]
    assert flag_spot_callability(rows, _cfg(detect_in_sec_only=False)) == []
    assert "spot_rate_signal_to_control_rna1" not in rows[0]


# ---------------------------------------------------------------------------
# THE THREE EMITTED RATE COLUMNS
# ---------------------------------------------------------------------------
def test_emits_all_three_rates_on_every_row():
    """They are run-level constants, so a reader should not have to re-derive
    them per image."""
    rows = [
        _row("a.tif", sec_only=False, rna1=40.0),
        _row("b.tif", sec_only=False, rna1=20.0),
        _row("sec.tif", sec_only=True, rna1=6.0),
    ]
    flag_spot_callability(rows, _cfg())
    for r in rows:
        assert r["spot_rate_sample_per_nucleus_rna1"] == pytest.approx(30.0)
        assert r["spot_rate_seconly_per_nucleus_rna1"] == pytest.approx(6.0)
        assert r["spot_rate_signal_to_control_rna1"] == pytest.approx(5.0)


def test_ratio_is_sample_over_control():
    rows = [
        _row("a.tif", sec_only=False, rna1=10.0),
        _row("sec.tif", sec_only=True, rna1=4.0),
    ]
    flag_spot_callability(rows, _cfg())
    assert rows[0]["spot_rate_signal_to_control_rna1"] == pytest.approx(2.5)


def test_each_punctate_channel_is_reported_separately():
    """One channel can be callable while its partner is not — which is exactly
    the MIAT-FISH-against-diffuse-antibody case."""
    rows = [
        _row("a.tif", sec_only=False, rna1=100.0, rna2=200.0),
        _row("sec.tif", sec_only=True, rna1=2.0, rna2=218.0),
    ]
    warnings = flag_spot_callability(rows, _cfg())
    assert rows[0]["spot_rate_signal_to_control_rna1"] == pytest.approx(50.0)
    assert rows[0]["spot_rate_signal_to_control_rna2"] == pytest.approx(200.0 / 218.0)
    assert len(warnings) == 1 and "[rna2]" in warnings[0]


def test_zero_control_rate_gives_nan_not_an_infinite_ratio():
    rows = [
        _row("a.tif", sec_only=False, rna1=30.0),
        _row("sec.tif", sec_only=True, rna1=0.0),
    ]
    warnings = flag_spot_callability(rows, _cfg())
    assert np.isnan(rows[0]["spot_rate_signal_to_control_rna1"])
    assert warnings == []


# ---------------------------------------------------------------------------
# THE WARNING
# ---------------------------------------------------------------------------
def test_the_measured_qki_case_warns():
    """~200 spots/nucleus in sample vs ~218 in the no-probe control: ratio 0.92,
    no discrimination at all. This is the result the guardrail encodes."""
    rows = [
        _row("sample.tif", sec_only=False, rna1=200.0),
        _row("seconly.tif", sec_only=True, rna1=218.0),
    ]
    warnings = flag_spot_callability(rows, _cfg())
    assert len(warnings) == 1
    assert rows[0]["spot_rate_signal_to_control_rna1"] < 1.0


def test_a_callable_channel_does_not_warn():
    rows = [
        _row("sample.tif", sec_only=False, rna1=60.0),
        _row("seconly.tif", sec_only=True, rna1=3.0),
    ]
    assert flag_spot_callability(rows, _cfg()) == []


def test_warning_names_the_intensity_floor_as_the_lever():
    rows = [
        _row("sample.tif", sec_only=False, rna1=200.0),
        _row("seconly.tif", sec_only=True, rna1=218.0),
    ]
    text = flag_spot_callability(rows, _cfg())[0]
    assert "min_spot_peak_intensity" in text


def test_warning_states_that_threshold_multiplier_will_not_help():
    """A relative threshold rescales sample and control together: textured
    secondary-antibody background produces genuine LoG maxima, so raising the
    multiplier cannot buy specificity."""
    rows = [
        _row("sample.tif", sec_only=False, rna1=200.0),
        _row("seconly.tif", sec_only=True, rna1=218.0),
    ]
    text = flag_spot_callability(rows, _cfg())[0]
    assert "threshold_multiplier will NOT improve specificity" in text
    assert "LoG maxima" in text


def test_warning_says_nothing_was_switched():
    """Warn, never auto-switch — reinterpreting a declared channel would be
    worse than the problem."""
    rows = [
        _row("sample.tif", sec_only=False, rna1=200.0),
        _row("seconly.tif", sec_only=True, rna1=218.0),
    ]
    text = flag_spot_callability(rows, _cfg())[0]
    assert "Nothing was changed or switched" in text


def test_declared_channel_config_is_untouched():
    cfg = _cfg()
    before = cfg.model_dump(mode="json")
    rows = [
        _row("sample.tif", sec_only=False, rna1=200.0),
        _row("seconly.tif", sec_only=True, rna1=218.0),
    ]
    flag_spot_callability(rows, cfg)
    assert cfg.model_dump(mode="json") == before


def test_threshold_is_configurable():
    rows = [
        _row("a.tif", sec_only=False, rna1=30.0),
        _row("sec.tif", sec_only=True, rna1=10.0),
    ]
    assert flag_spot_callability(rows, _cfg(min_ratio=2.0)) == []
    rows2 = [dict(r) for r in rows]
    assert len(flag_spot_callability(rows2, _cfg(min_ratio=5.0))) == 1


# ---------------------------------------------------------------------------
# DEFENSIVENESS — an advisory pass must never break a completed run
# ---------------------------------------------------------------------------
def test_nonfinite_and_missing_values_are_tolerated():
    rows = [
        _row("a.tif", sec_only=False, rna1=float("nan")),
        _row("b.tif", sec_only=False, rna1=40.0),
        {"image": "c.tif", "secondary_only": False},  # channel key absent
        _row("sec.tif", sec_only=True, rna1=4.0),
    ]
    flag_spot_callability(rows, _cfg())
    assert rows[0]["spot_rate_sample_per_nucleus_rna1"] == pytest.approx(40.0)
    assert rows[0]["spot_rate_signal_to_control_rna1"] == pytest.approx(10.0)


def test_runner_runs_the_pass_before_building_the_csv():
    """The diagnostic mutates the per-image row dicts IN PLACE, so it has to run
    before ``per_image_df`` is constructed. If it were moved after, the three
    rate columns would vanish from per_image_summary.csv with nothing failing."""
    import inspect

    import fishsuite.runner as runner

    src = inspect.getsource(runner)
    assert src.index("flag_spot_callability") < src.index(
        "per_image_df = pd.DataFrame(per_image_rows)"
    )


def test_bad_input_returns_no_warnings_and_does_not_raise():
    assert flag_spot_callability([], _cfg()) == []
    assert flag_spot_callability([None, 7], _cfg()) == []

    class _NoFoci:
        pass

    assert flag_spot_callability(
        [_row("a.tif", sec_only=False, rna1=1.0)], _NoFoci()
    ) == []
