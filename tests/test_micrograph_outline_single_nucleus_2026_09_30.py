import matplotlib
matplotlib.use("Agg")
import numpy as np
from fishsuite.figures import twoarm


class _Saver:
    def __init__(self):
        self.figs = []

    def save(self, fig, stem, **kw):
        self.figs.append(fig)
        return {}


def _n_contours(labels, **kw):
    sv = _Saver(); rgb = np.zeros((40, 40, 3), np.uint8)
    twoarm.micrograph(rgb, 0.065, sv, "t", labels=labels, scalebar_um=1, **kw)
    ax = sv.figs[0].axes[0]
    return sum(1 for c in ax.get_children() if "Contour" in type(c).__name__)


def _labels(n):
    lab = np.zeros((40, 40), int)
    for i in range(n):
        lab[2 + 12 * i: 10 + 12 * i, 5:15] = i + 1
    return lab


def test_field_has_no_outline_by_default():
    assert _n_contours(_labels(3)) == 0


def test_single_nucleus_keeps_outline():
    assert _n_contours(_labels(1)) >= 1


def test_field_override():
    assert _n_contours(_labels(3), outline_fields=True) >= 1
