import matplotlib; matplotlib.use("Agg")
import numpy as np, pandas as pd, pytest
from fishsuite.figures import style, twoarm

def _data():
    r = np.random.default_rng(0)
    nuc = pd.DataFrame({"arm": ["A"] * 60 + ["B"] * 60, "value": np.r_[r.normal(1, .3, 60), r.normal(.5, .3, 60)]})
    units = pd.DataFrame({"arm": ["A"] * 3 + ["B"] * 3, "value": [1, 1.1, .9, .5, .6, .4]})
    return nuc, units

def _dot_alphas(tmp_path, **kw):
    nuc, units = _data(); sv = style.Saver(tmp_path, keep_open=True)
    twoarm.plot_two_arm(nuc, units, ("A", "B"), {"A": "#595959", "B": "#E69F00"}, "y", "t", .01, [], sv, "x",
                        dots=True, seed=0, **kw)
    ax = sv.figures["x"].axes[0]
    return [c.get_alpha() for c in ax.collections if hasattr(c, "get_sizes") and len(c.get_sizes()) and c.get_sizes()[0] == 5], ax

@pytest.mark.parametrize("focus", [False, True])
def test_default_alpha_unchanged_and_override(tmp_path, focus):
    kw = dict(focus=True, focus_kind="ratio", focus_cap=1.5) if focus else {}
    a0, _ = _dot_alphas(tmp_path / "a", **kw); a1, ax = _dot_alphas(tmp_path / "b", dot_alpha=.12, **kw)
    assert a0 == [.45, .45] and a1 == [.12, .12]
