import numpy as np
import pandas as pd
import pytest

from fishsuite.figures import style, twoarm


def _data():
    rng = np.random.default_rng(0)
    arms = ("NT ASO", "MIAT-KD ASO")
    nuc = pd.DataFrame({"arm": np.repeat(arms, 50), "value": np.r_[rng.normal(10, 2, 50), rng.normal(5, 2, 50)]})
    units = pd.DataFrame({"arm": np.repeat(arms, 3), "value": [10, 11, 9, 5, 6, 4]})
    return arms, nuc, units


@pytest.mark.parametrize("dots", [True, False])
def test_two_arm_markers_and_p_text(tmp_path, dots):
    style.apply_style()
    arms, nuc, units = _data()
    cols = {"NT ASO": "#595959", "MIAT-KD ASO": "#E69F00"}
    sv = style.Saver(tmp_path, keep_open=True, write=False)
    out = twoarm.plot_two_arm(nuc, units, arms, cols, "y", "t", 0.0123, ["f"], sv, "C_x", dots=dots)
    fig = sv.figures["C_x"]
    ax = fig.axes[0]
    texts = [t.get_text() for t in ax.texts]
    assert any("p = 0.012" in t and t.startswith("*") for t in texts)
    big = [c for c in ax.collections if hasattr(c, "get_sizes") and len(c.get_sizes()) and c.get_sizes()[0] == 110]
    assert len(big) == 6
    for c in big:
        assert len(c.get_paths()) == 1
    faces = {tuple(np.round(c.get_facecolor()[0][:3], 3)) for c in big}
    assert len(faces) == 2
    assert out["NT ASO"]["n_units"] == 3 and out["MIAT-KD ASO"]["mean_of_unit_means"] == pytest.approx(5.0)


def test_not_detected_wording():
    assert twoarm.sig_label(0.333).startswith(twoarm.NOT_DETECTED)
    assert "ns" not in twoarm.sig_label(0.333)
    assert twoarm.sig_label(None) == ""


def test_lock_defaults_unchanged():
    assert style.LV == {"miat": (500.0, 2250.0), "qki": (1050.0, 3746.0), "dapi": (607.0, 9000.0)}
    assert style.WELL_MEAN_MARKER["marker"] == "o"


def test_build_linked_set_new_kwargs_keep_locked_defaults():
    import inspect
    from fishsuite.figures import linked_set
    sig = inspect.signature(linked_set.build_linked_set).parameters
    assert sig["channel_labels"].default == ("MIAT-640", "QKI-561")
    assert sig["qki_min"].default == 1050 and sig["miat_min"].default == 500
    assert sig["arm_color"].default is None and sig["b4_floors"].default is None
    src = inspect.getsource(linked_set.build_linked_set)
    assert "b4_floors.get(key)" in src and "axhline(fv" in src
