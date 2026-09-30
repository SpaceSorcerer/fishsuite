"""Focus mode (2026-09-30): range rule, pinned counts, bracket/marker headroom, render, fraction/ratio kinds."""
import numpy as np
import pandas as pd
import pytest
from matplotlib.transforms import Bbox

from fishsuite.figures import focus, style, twoarm
from fishsuite.figures.c_superplots import plot_one_sample, lab


def _heavy(seed=0, n=300):
    rng = np.random.default_rng(seed)
    a = np.r_[rng.lognormal(2.0, .5, n), [400.0, 650.0]]          # two far outliers blow up the full-range axis
    b = np.r_[rng.lognormal(1.6, .5, n), [-50.0]]
    return {"NT": a, "KD": b}


@pytest.mark.parametrize("seed", range(5))
def test_range_rule_means_central98_pins(seed):
    vals = _heavy(seed)
    units = [np.mean(vals["NT"][:100]), np.mean(vals["NT"][100:200]), np.mean(vals["KD"][:100])]
    far = focus.focus_window(vals, units + [999.0])                 # a mean far outside the bulk still frames
    assert far.data_lo <= 999.0 <= far.data_hi
    fw = focus.focus_window(vals, units, None, axes_height_pt=137, reserve_pt=20, marker_pt=5.7)
    for m in units:
        assert fw.data_lo <= m <= fw.data_hi
    for arm, v in vals.items():
        inside = (v >= fw.data_lo) & (v <= fw.data_hi)
        assert inside.mean() >= .98
        assert fw.n_above[arm] == int((v > fw.data_hi).sum())
        assert fw.n_below[arm] == int((v < fw.data_lo).sum())
        lo, hi = focus.central_range(v)
        assert fw.data_lo <= lo and hi <= fw.data_hi
    assert fw.n_above["NT"] >= 2 and fw.n_below["KD"] >= 1       # the planted outliers are pinned, not dropped
    assert fw.ylim[0] == fw.data_lo and fw.data_hi < fw.bracket_y < fw.ylim[1]


def test_small_n_order_statistics_keep_all_points():
    v = {"a": np.arange(10.0)}
    fw = focus.focus_window(v, [4.5])
    assert fw.n_out == 0


def test_fraction_and_nonnegative_and_ratio():
    rng = np.random.default_rng(1)
    fr = {"a": rng.uniform(0, 1, 400)}
    fw = focus.focus_window(fr, [.5], "fraction")
    assert fw.data_lo >= 0 and fw.data_hi <= 1
    nn = {"a": rng.uniform(0, 1, 400)}
    assert focus.focus_window(nn, [.5]).data_lo >= 0
    r = {"a": rng.normal(.6, .05, 400)}
    fw = focus.focus_window(r, [.6], "ratio")
    assert fw.data_hi >= 1.0                                        # NT = 1 reference always inside


def _two_arm(tmp_path, focus_on, p=0.003, dots=True, kind=None):
    style.apply_style()
    vals = _heavy(3)
    nuc = pd.DataFrame({"arm": np.repeat(list(vals), [len(v) for v in vals.values()]), "value": np.concatenate(list(vals.values()))})
    units = pd.DataFrame({"arm": ["NT"] * 3 + ["KD"] * 3,
                          "value": [vals["NT"][i::3].mean() for i in range(3)] + [vals["KD"][i::3].mean() for i in range(3)]})
    sv = style.Saver(tmp_path, keep_open=True, write=True)
    out = twoarm.plot_two_arm(nuc, units, ("NT", "KD"), {"NT": "#595959", "KD": "#E69F00"}, "y", "t", p, ["f"], sv,
                              "P", dots=dots, focus=focus_on, focus_kind=kind)
    return sv.figures["P"], out, vals, units


@pytest.mark.parametrize("dots", [True, False])
def test_render_bracket_inside_and_clear_of_circles(tmp_path, dots):
    fig, out, vals, units = _two_arm(tmp_path, True, dots=dots)
    ax = fig.axes[0]
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    axbb = ax.get_window_extent(r)
    ylim = ax.get_ylim()
    assert (units.value.between(*ylim)).all()
    sig = [t for t in ax.texts if "p = " in t.get_text()][0]
    tb = sig.get_window_extent(r)
    assert tb.y1 <= axbb.y1 + .5 and tb.y0 >= axbb.y0                # bracket text inside the axes
    bracket = [l for l in ax.lines if len(l.get_xdata()) == 4][0]
    yb_px = ax.transData.transform((0, min(bracket.get_ydata())))[1]
    big = [c for c in ax.collections if hasattr(c, "get_sizes") and len(c.get_sizes()) and c.get_sizes()[0] == 110]
    assert len(big) == 6
    r_px = (np.sqrt(110) / 2 + .5) * fig.dpi / 72                    # well-mean circle radius incl. edge
    for c in big:
        cy = ax.transData.transform(c.get_offsets())[:, 1]
        assert (cy + r_px < yb_px).all()                              # no circle touches the bracket ticks
    note = [t for t in ax.texts if t.get_gid() == focus.NOTE_GID]
    assert len(note) == 1
    n_out = out["focus"]["n_out"]
    assert f"{sum(out['focus']['n_above'].values())} nuclei above" in note[0].get_text()
    assert n_out >= 3
    pins = [c for c in ax.collections if hasattr(c, "get_sizes") and len(c.get_sizes()) and c.get_sizes()[0] == 9]
    assert sum(len(c.get_offsets()) for c in pins) == (n_out if dots else 0)
    assert (tmp_path / "P.png").stat().st_size > 0


def test_focus_window_is_tighter_than_full(tmp_path):
    _, full, _, _ = _two_arm(tmp_path / "a", False)
    _, foc, _, _ = _two_arm(tmp_path / "b", True)
    assert (foc["ylim"][1] - foc["ylim"][0]) < .5 * (full["ylim"][1] - full["ylim"][0])


def test_focus_and_ylim_conflict(tmp_path):
    with pytest.raises(ValueError):
        twoarm.plot_two_arm(pd.DataFrame({"arm": ["a"], "value": [1.0]}), pd.DataFrame({"arm": ["a"], "value": [1.0]}),
                            ("a",), {"a": "#595959"}, "y", "t", None, [], style.Saver("x", write=False), "s", dots=False,
                            ylim=(0, 1), focus=True)


def test_one_sample_focus_fraction(tmp_path):
    style.apply_style()
    rng = np.random.default_rng(0)
    nv = pd.DataFrame({"field": np.repeat(["14", "15"], 150), "frac_x": np.r_[rng.beta(2, 8, 299), [1.0]]})
    t = dict(p_holm=1e-5, n=300, rank_biserial=.5, median_diff=.1)
    sv = style.Saver(tmp_path, keep_open=True, write=False)
    stem = plot_one_sample(nv, ["14", "15"], "C9", "frac_x", .1, "y", "violin", t, True, rng, sv, focus=True,
                           suffix="_holm_focus")
    ax = sv.figures[stem].axes[0]
    lo, hi = ax.get_ylim()
    assert lo >= 0
    assert all(0 <= y <= 1.0001 for y in ax.get_yticks())
    assert any(t_.get_gid() == focus.NOTE_GID for t_ in ax.texts)
    assert any(t_.get_text() == lab(t) for t_ in ax.texts)


@pytest.mark.parametrize("dots", [True, False])
def test_default_bracket_above_high_data_on_zero_based_axis(tmp_path, dots):
    """Regression (b62484e): the caller-window guard fired on automatic zero-based axes when data sat high,
    putting the bracket inside the data. The bracket must sit above every plotted value and clear the circles."""
    style.apply_style()
    rng = np.random.default_rng(0)
    nuc = pd.DataFrame({"arm": np.repeat(["NT", "KD"], 60), "value": np.r_[rng.normal(105, 2, 60), rng.normal(101, 2, 60)]})
    units = pd.DataFrame({"arm": ["NT"] * 3 + ["KD"] * 3, "value": [104.5, 105.2, 105.9, 100.4, 101.1, 101.8]})
    sv = style.Saver(tmp_path, keep_open=True, write=False)
    twoarm.plot_two_arm(nuc, units, ("NT", "KD"), {"NT": "#595959", "KD": "#E69F00"}, "y", "t", 0.004, [], sv, "H",
                        dots=dots)
    fig = sv.figures["H"]; ax = fig.axes[0]
    assert ax.get_ylim()[0] == 0.0
    bracket = [l for l in ax.lines if len(l.get_xdata()) == 4][0]
    yb_lo = min(bracket.get_ydata())
    shown = nuc.value.max() if dots else units.value.max()
    assert max(bracket.get_ydata()) > max(nuc.value.max(), units.value.max())
    assert yb_lo > shown
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    r_px = (np.sqrt(110) / 2 + .5) * fig.dpi / 72
    yb_px = ax.transData.transform((0, yb_lo))[1]
    assert (ax.transData.transform(units[["value"]].assign(x=0)[["x", "value"]].to_numpy())[:, 1] + r_px < yb_px).all()
    sig = [t for t in ax.texts if "p = " in t.get_text()][0]
    assert sig.get_window_extent(r).y1 <= ax.get_window_extent(r).y1 + .5


@pytest.mark.parametrize("dots", [True, False])
def test_focus_cap_hard_top(tmp_path, dots):
    style.apply_style()
    rng = np.random.default_rng(2)
    v = {"NT": rng.lognormal(0, .7, 300), "KD": rng.lognormal(-.6, .7, 300)}
    nuc = pd.DataFrame({"arm": np.repeat(["NT", "KD"], 300), "value": np.r_[v["NT"], v["KD"]]})
    units = pd.DataFrame({"arm": ["NT"] * 3 + ["KD"] * 3, "value": [v["NT"][i::3].mean() for i in range(3)] + [v["KD"][i::3].mean() for i in range(3)]})
    sv = style.Saver(tmp_path, keep_open=True, write=False)
    out = twoarm.plot_two_arm(nuc, units, ("NT", "KD"), {"NT": "#595959", "KD": "#E69F00"}, "y", "t", 0.006, [], sv, "K",
                              dots=dots, focus=True, focus_kind="ratio", focus_cap=2.5)
    ax = sv.figures["K"].axes[0]
    n_above = int((nuc.value > 2.5).sum())
    assert out["focus"]["data_hi"] == 2.5 and sum(out["focus"]["n_above"].values()) == n_above > 0
    assert max(ax.get_yticks()) <= 2.5 and ax.get_ylim()[1] > 2.5
    note = [t for t in ax.texts if t.get_gid() == focus.NOTE_GID][0].get_text()
    assert note.startswith(f"{n_above} nuclei above 2.5 " + ("drawn at edge" if dots else "(not shown)"))
    bracket = [l for l in ax.lines if len(l.get_xdata()) == 4][0]
    assert min(bracket.get_ydata()) > 2.5
    pins = [c for c in ax.collections if hasattr(c, "get_sizes") and len(c.get_sizes()) and c.get_sizes()[0] == 9]
    assert sum(len(c.get_offsets()) for c in pins) == (out["focus"]["n_out"] if dots else 0)
    with pytest.raises(ValueError):
        twoarm.plot_two_arm(nuc, units, ("NT", "KD"), {"NT": "#595959", "KD": "#E69F00"}, "y", "t", 0.006, [], sv, "K2",
                            dots=dots, focus_cap=2.5)


@pytest.mark.parametrize("dots", [True, False])
def test_caller_ylim_bracket_above_shown_data(tmp_path, dots):
    """Caller-supplied ylim whose data reach the top (the C05 nodots case): bracket above every visible value."""
    style.apply_style()
    rng = np.random.default_rng(1)
    nuc = pd.DataFrame({"arm": np.repeat(["NT", "KD"], 80), "value": np.r_[rng.uniform(.6, .98, 80), rng.uniform(.5, .95, 80)]})
    units = pd.DataFrame({"arm": ["NT"] * 2 + ["KD"] * 2, "value": [.90, .92, .80, .85]})
    sv = style.Saver(tmp_path, keep_open=True, write=False)
    twoarm.plot_two_arm(nuc, units, ("NT", "KD"), {"NT": "#595959", "KD": "#E69F00"}, "y", "t", 0.2, [], sv, "Y",
                        dots=dots, ylim=(0.4, 1.02))
    ax = sv.figures["Y"].axes[0]
    y0, y1 = ax.get_ylim()
    ys = np.concatenate([c.get_offsets()[:, 1] for c in ax.collections if type(c).__name__ == "PathCollection"])
    vis = ys[(ys >= y0) & (ys <= y1)]
    bracket = [l for l in ax.lines if len(l.get_xdata()) == 4][0]
    assert min(bracket.get_ydata()) > vis.max()
    assert y0 == 0.4


def test_nodots_cap_note_says_not_shown(tmp_path):
    style.apply_style()
    rng = np.random.default_rng(2)
    nuc = pd.DataFrame({"arm": np.repeat(["NT", "KD"], 300), "value": rng.lognormal(0, .8, 600)})
    units = pd.DataFrame({"arm": ["NT"] * 3 + ["KD"] * 3, "value": [1, 1.1, .9, .5, .6, .4]})
    sv = style.Saver(tmp_path, keep_open=True, write=False)
    out = twoarm.plot_two_arm(nuc, units, ("NT", "KD"), {"NT": "#595959", "KD": "#E69F00"}, "y", "t", 0.01, [], sv, "N",
                              dots=False, focus=True, focus_kind="ratio", focus_cap=2.5)
    note = [t for t in sv.figures["N"].axes[0].texts if t.get_gid() == focus.NOTE_GID][0].get_text()
    assert note == f"{sum(out['focus']['n_above'].values())} nuclei above 2.5 (not shown)"
