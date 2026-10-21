"""Ortho / linked-set intensity profile sampled over a strip of configurable width and aggregation (Brian 2026-10-20).
Default (no strip set) = legacy 3-px mean, byte-identical; the drawn line unchanged, the strip not drawn. Goldens
captured from ce546fa before the change by tests/data/capture_ortho_profile_strip_golden.py."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.text import Text
from skimage.measure import profile_line

import fishsuite.core.ortho_profile as op
from fishsuite.figures import linked_set
from test_linked_ortho_outlines_2026_09_30 import _Run

DATA = Path(__file__).parent / "data"
PX = 0.065
LINES = {"horizontal": ((48., 30.), (48., 66.)), "oblique": ((35.2, 31.7), (60.9, 64.3))}
FOOTER_OLD = "Profiles normalised to each channel's own min–max along this line (display only)."
FOOTER_05 = ("Profiles normalised to each channel's own min–max along this line (display only); "
             "sampled over a 0.5 µm wide strip, mean across the strip.")
FOOTER_1MAX = ("Profiles normalised to each channel's own min–max along this line (display only); "
               "sampled over a 1 µm wide strip, maximum across the strip.")


def _synthetic():
    rng = np.random.default_rng(20261020)
    z, y, x = np.indices((12, 96, 96))
    blob = np.exp(-((z - 6) ** 2 / 4 + (y - 48) ** 2 / 30 + (x - 48) ** 2 / 30))
    noise = rng.normal(0, 1, (3, 12, 96, 96))
    return np.stack([blob * 100, blob * 70, blob * 20]) + noise * np.array([5., 4., 2.])[:, None, None, None]


def _rows_stack(seed):
    rows = np.random.default_rng(seed).uniform(0, 100, 96)
    plane = np.broadcast_to(rows[:, None], (96, 96)).astype(float)
    return rows, plane, np.broadcast_to(np.stack([plane, 2 * plane, plane])[:, None], (3, 12, 96, 96)).copy()


def _mask():
    m = np.zeros((96, 96), bool); m[20:76, 20:76] = True
    return m


def _render(stack, ends=None, **kw):
    return op.render_ortho_figure(stack, (6, 48, 48), 30, nucleus_mask=_mask(), pixel_size_um=PX, z_step_um=.2,
                                  display_levels=((0, 100), (0, 100)), line_endpoints=ends, **kw)


def _traces(fig, labels=("MIAT", "QKI")):
    tr = {l.get_label(): l for l in next(a for a in fig.axes if a.get_label() == "raw").lines}
    return (np.asarray(tr[labels[0]].get_xdata()), np.asarray(tr[labels[0]].get_ydata()),
            np.asarray(tr[labels[1]].get_ydata()))


def _linked(tmp_path, **kw):
    run = _Run(tmp_path)
    orig = linked_set.read_stack
    linked_set.read_stack = lambda r, k: run.stack
    try:
        params = linked_set.build_linked_set(run, "14", 1, 0.5, **kw)
    finally:
        linked_set.read_stack = orig
    return run, params


def _fig(run, prefix):
    return next(f for s, (f, _) in run.saver.figs.items() if s.startswith(prefix))


def test_strip_width_px_rounding():
    assert op.strip_width_px(0.195, PX) == 3
    assert op.strip_width_px(0.5, PX) == 8
    assert op.strip_width_px(1.0, PX) == 15
    assert op.strip_width_px(0.01, PX) == 1
    for bad in (0, -1, np.nan):
        with pytest.raises(ValueError):
            op.strip_width_px(bad, PX)


@pytest.mark.parametrize("name", list(LINES))
def test_strip_0195_and_default_reproduce_pre_change_golden(name):
    g = np.load(DATA / "ortho_profile_strip_golden_ce546fa.npz")
    stack = _synthetic()
    d, v = op.line_profile(stack[:, 6], *LINES[name], width_px=op.strip_width_px(0.195, PX), pixel_size_um=PX)
    np.testing.assert_array_equal(d, g[f"{name}_distance"])
    np.testing.assert_array_equal(v, g[f"{name}_values"])
    for kw in (dict(), dict(profile_strip_um=0.195), dict(profile_width_px=3)):
        fig = _render(stack, LINES[name], **kw)
        x, y0, y1 = _traces(fig)
        np.testing.assert_array_equal(x, g[f"{name}_raw_x"])
        np.testing.assert_array_equal(y0, g[f"{name}_raw_y0"])
        np.testing.assert_array_equal(y1, g[f"{name}_raw_y1"])
        assert fig._ortho_geometry["profile_width_px"] == 3
        plt.close(fig)


@pytest.mark.parametrize("kw", [dict(), dict(profile_strip_um=0.195)], ids=["default", "strip0195"])
def test_linked_set_reproduces_pre_change_golden(tmp_path, kw):
    g = np.load(DATA / "linked_set_b2_b4_golden_ce546fa.npz")
    run, params = _linked(tmp_path, **kw)
    b4 = pd.read_csv(run.data / "B4_line_profile_values_field14_nuc1.csv")
    for c in ("distance_um", "miat", "qki", "dapi"):
        np.testing.assert_array_equal(b4[c].to_numpy(), g[f"b4_{c}"])
    x, y0, y1 = _traces(_fig(run, "B2_"), ("MIAT-640", "QKI-561"))
    np.testing.assert_array_equal(x, g["b2_raw_x"])
    np.testing.assert_array_equal(y0, g["b2_raw_y0"])
    np.testing.assert_array_equal(y1, g["b2_raw_y1"])
    if kw:
        assert params["line"]["profile_width_px"] == 3 and params["line"]["profile_strip_um"] == 0.195
    else:                                                               # default: params and footers as before
        assert not {"profile_width_px", "profile_strip_um", "profile_strip_agg"} & set(params["line"])
        assert FOOTER_OLD in _fig(run, "B2_")._ortho_geometry["footer"]
        assert any("; 3-px-wide line, analysed plane" in t.get_text() for t in _fig(run, "B4_").findobj(Text))


def test_strip_05_mean_averages_expected_pixels():
    rows, plane, stack = _rows_stack(7)
    r, ends = 48, ((48., 30.), (48., 66.))
    # 8 perpendicular samples at row offsets ±0.5 … ±3.5, bilinear: rows r-4 … r+4 weighted .5,1,…,1,.5 over 8.
    w = np.r_[.5, np.ones(7), .5] / 8
    expected = float(w @ rows[r - 4:r + 5])
    _, v = op.line_profile(stack[:, 6], *ends, width_px=op.strip_width_px(0.5, PX), pixel_size_um=PX)
    np.testing.assert_allclose(v[0], expected, rtol=1e-12)
    np.testing.assert_allclose(v[1], 2 * expected, rtol=1e-12)
    np.testing.assert_array_equal(v[0], profile_line(plane, *ends, linewidth=8, order=1, mode="constant",
                                                     reduce_func=np.mean))
    fig = _render(stack, ends, profile_strip_um=0.5)
    _, y0, _ = _traces(fig)
    np.testing.assert_allclose(y0, expected, rtol=1e-12)
    geo = fig._ortho_geometry
    assert geo["profile_width_px"] == 8 and geo["profile_strip_um"] == 0.5 and geo["profile_strip_agg"] == "mean"
    assert geo["profile_strip_um_sampled"] == pytest.approx(8 * PX)
    plt.close(fig)
    _, v3 = op.line_profile(stack[:, 6], *ends, width_px=3, pixel_size_um=PX)
    np.testing.assert_allclose(v3[0], rows[r - 1:r + 2].mean(), rtol=1e-12)
    assert abs(v3[0][0] - expected) > 1e-6                              # the rows discriminate the two widths


def test_strip_1um_max_takes_maximum_across_strip():
    rows, plane, stack = _rows_stack(11)
    r, ends = 48, ((48., 30.), (48., 66.))
    expected = rows[r - 7:r + 8].max()                                  # 15 samples at integer row offsets -7 … +7
    _, v = op.line_profile(stack[:, 6], *ends, width_px=op.strip_width_px(1.0, PX), pixel_size_um=PX, agg="max")
    np.testing.assert_array_equal(v[0], expected)
    np.testing.assert_array_equal(v[1], 2 * expected)
    np.testing.assert_array_equal(v[0], profile_line(plane, *ends, linewidth=15, order=1, mode="constant",
                                                     reduce_func=np.max))
    fig = _render(stack, ends, profile_strip_um=1.0, profile_strip_agg="max")
    _, y0, _ = _traces(fig)
    np.testing.assert_array_equal(y0, expected)
    assert fig._ortho_geometry["profile_strip_agg"] == "max" and fig._ortho_geometry["profile_width_px"] == 15
    assert FOOTER_1MAX in fig._ortho_geometry["footer"]
    plt.close(fig)
    _, vm = op.line_profile(stack[:, 6], *ends, width_px=15, pixel_size_um=PX)
    assert vm[0][0] < expected                                          # mean differs from max on these rows


def test_strip_argument_validation():
    stack = _synthetic()
    with pytest.raises(ValueError):
        _render(stack, LINES["oblique"], profile_strip_um=0.5, profile_width_px=3)
    with pytest.raises(ValueError):
        _render(stack, LINES["oblique"], profile_strip_agg="max")
    with pytest.raises(ValueError):
        _render(stack, LINES["oblique"], profile_strip_um=0.5, profile_strip_agg="median")
    with pytest.raises(ValueError):
        op.line_profile(stack[:, 6], *LINES["oblique"], pixel_size_um=PX, agg="sum")


def test_footer_states_strip_only_when_set():
    stack = _synthetic()
    fig = _render(stack, LINES["oblique"])
    assert FOOTER_OLD in fig._ortho_geometry["footer"] and "sampled over" not in fig._ortho_geometry["footer"]
    plt.close(fig)
    fig = _render(stack, LINES["oblique"], profile_strip_um=0.5)
    assert FOOTER_05 in fig._ortho_geometry["footer"]
    assert "sampled over a 0.5 µm wide strip, mean" in " ".join(t.get_text() for t in fig.texts)
    plt.close(fig)
    fig = _render(stack, LINES["oblique"], profile_strip_um=0.195)
    assert "(display only); sampled over a 0.195 µm wide strip, mean across the strip." in fig._ortho_geometry["footer"]
    plt.close(fig)


def _drawn_inventory(fig):
    out = []
    for ax in fig.axes:
        if ax.get_label() in ("raw", "normalised"):
            continue
        for l in ax.lines:
            out.append((ax.get_label(), "line", l.get_label(), l.get_linewidth(), l.get_linestyle(), l.get_color(),
                        tuple(np.round(np.asarray(l.get_xdata(), float), 9)),
                        tuple(np.round(np.asarray(l.get_ydata(), float), 9))))
        for c in ax.collections:
            out.append((ax.get_label(), "coll", c.get_label(), tuple(np.atleast_1d(c.get_linewidths())),
                        tuple(np.atleast_1d(c.get_sizes())) if hasattr(c, "get_sizes") else ()))
        out += [(ax.get_label(), "patch", p.get_label(), p.get_linewidth()) for p in ax.patches]
    return out


def test_drawn_line_weight_unchanged_and_strip_not_drawn():
    stack = _synthetic()
    figs = {"default": _render(stack, LINES["oblique"]),
            "0.5": _render(stack, LINES["oblique"], profile_strip_um=0.5),
            "1max": _render(stack, LINES["oblique"], profile_strip_um=1.0, profile_strip_agg="max")}
    ref = _drawn_inventory(figs["default"])
    assert _drawn_inventory(figs["0.5"]) == ref and _drawn_inventory(figs["1max"]) == ref
    for fig in figs.values():
        axes = {a.get_label(): a for a in fig.axes}
        for key in ("xy", "xz"):
            prof = [l for l in axes[key].lines if l.get_label() == "measured profile line"]
            assert len(prof) == 1 and prof[0].get_linewidth() == 1.0 and prof[0].get_linestyle() == ":"
        dots = [c for c in axes["xy"].collections if hasattr(c, "get_sizes") and len(c.get_sizes())]
        assert [tuple(c.get_sizes()) for c in dots] == [(4.0,)]
        cross = [l for l in axes["xy"].lines if l.get_label() == "section position"]
        assert len(cross) == 2 and all(l.get_linewidth() == .5 for l in cross)
        plt.close(fig)


@pytest.mark.parametrize("strip,agg,npx", [(0.5, None, 8), (1.0, "max", 15)])
def test_linked_set_strip_footer_and_params(tmp_path, strip, agg, npx):
    run, params = _linked(tmp_path, profile_strip_um=strip, profile_strip_agg=agg)
    word = "maximum" if agg == "max" else "mean"
    assert params["line"]["profile_strip_um"] == strip and params["line"]["profile_width_px"] == npx
    assert params["line"]["profile_strip_agg"] == (agg or "mean")
    b2 = _fig(run, "B2_")
    phrase = f"sampled over a {strip:g} µm wide strip, {word} across the strip"
    assert f"(display only); {phrase}." in b2._ortho_geometry["footer"] and b2._ortho_geometry["profile_width_px"] == npx
    assert any(f"profile {phrase}" in t.get_text() for t in _fig(run, "B4_").findobj(Text))
    assert _fig(run, "B3_").axes[0].lines[0].get_linewidth() == .8       # B3 drawn line untouched


def _section_axes_clean(fig):
    axes = {a.get_label(): a for a in fig.axes}
    for key in ("xz", "yz"):
        ax = axes[key]
        assert len(ax.get_xticks()) == 0 and len(ax.get_yticks()) == 0
        assert not [l for l in ax.lines if l.get_label() == "analysed plane"]
        assert all(l.get_clip_on() for l in ax.lines if not l.get_label().startswith("scale bar"))


def test_b2_section_axes_have_no_outside_ticks(tmp_path):
    """2026-10-21: no unlabelled analysed-plane ticks beside XZ / YZ when the z label sits on the XY merge."""
    stack = _synthetic()
    for kw in (dict(), dict(profile_strip_um=1.0, profile_strip_agg="max")):
        fig = _render(stack, LINES["oblique"], analysed_plane_z=6, show_z_slice_labels=True, **kw)
        _section_axes_clean(fig)
        assert any(t.get_text() == "z = 7" for t in {a.get_label(): a for a in fig.axes}["xy"].texts)
        plt.close(fig)
    run, _ = _linked(tmp_path, profile_strip_um=1.0, profile_strip_agg="max")
    _section_axes_clean(_fig(run, "B2_"))
    fig = _render(stack, LINES["oblique"], analysed_plane_z=6)          # labelled ticks kept on the default path
    assert len([l for a in fig.axes for l in a.lines if l.get_label() == "analysed plane"]) == 2
    plt.close(fig)
