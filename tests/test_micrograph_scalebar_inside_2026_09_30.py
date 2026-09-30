import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest
from fishsuite.figures import twoarm


class _Saver:
    def __init__(self):
        self.figs = []

    def save(self, fig, stem, **kw):
        self.figs.append(fig)
        return {}


def _render(h, w, px_um, bar_um, k=1):
    sv = _Saver()
    info = twoarm.micrograph(np.zeros((h, w, 3), np.uint8), px_um, sv, "t", k=k, scalebar_um=bar_um)
    fig = sv.figs[0]
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3], info


@pytest.mark.parametrize("px_um,bar_um,h,w,k", [(0.065, 5, 300, 320, 1), (0.13, 20, 400, 500, 1),
                                                 (0.13, 5, 150, 140, 2)])
def test_bar_inside_true_length_no_strip(px_um, bar_um, h, w, k):
    img, info = _render(h, w, px_um, bar_um, k)
    assert img.shape[:2] == (h * k, w * k)          # no added rows/columns
    x0, y0, x1, y1 = info["bar_xyxy_px"]
    row = img[int((y0 + y1) / 2 * k), :, 0] > 200   # white run through the bar's middle row
    runs = np.diff(np.r_[0, row.astype(int), 0]); st = np.flatnonzero(runs == 1); en = np.flatnonzero(runs == -1)
    measured = int((en - st).max())
    assert abs(measured - bar_um / px_um * k) <= 1 * k
    assert y1 < h and x1 < w                          # bar lies over the image


def test_strip_argument_ignored():
    sv = _Saver()
    info = twoarm.micrograph(np.zeros((50, 60, 3), np.uint8), 0.13, sv, "t", scalebar_um=2, strip_px=120)
    assert (info["out_h"], info["out_w"]) == (50, 60)
