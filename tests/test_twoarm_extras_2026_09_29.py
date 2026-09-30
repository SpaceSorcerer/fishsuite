import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from PIL import Image

from fishsuite.figures import style, twoarm


def test_sig_label_default_unchanged():
    assert twoarm.sig_label(0.005) == "**  p = 0.005"
    assert twoarm.sig_label(0.2) == "not detected at this n  (p = 0.2)"
    assert "9 v 9 wells" in twoarm.sig_label(0.01, "9 v 9 wells")


def test_radial_and_ratio(tmp_path):
    style.apply_style()
    sv = style.Saver(tmp_path)
    r = np.arange(5) * .13
    prof = {a: dict(r=r, obs=1.2 - r * .1, lo=None, hi=None, ctrl=np.full(5, 1.04)) for a in ("A", "B")}
    out = twoarm.radial_profile(prof, ("A", "B"), {"A": "#595959", "B": "#E69F00"}, "x", "y", "t", sv, "rad")
    assert (tmp_path / "rad.png").exists() and out["ylim"][0] < 1.0
    u = pd.DataFrame(dict(arm=["A", "A", "B", "B"], x=[1., 2., 3., 4.], y=[.2, .4, .6, .8]))
    f = twoarm.ratio_scatter(u, ("A", "B"), {"A": "#595959", "B": "#E69F00"}, "x", "y", "t", sv, "rat", chance_slope=.1)
    assert abs(f["slope"] - .2) < 1e-9 and abs(f["pearson_r"] - 1) < 1e-9 and f["n_units"] == 4


def test_micrograph_pixel_exact(tmp_path):
    style.apply_style()
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (40, 60, 3), dtype=np.uint8)
    sv = style.Saver(tmp_path)
    g = twoarm.micrograph(img, .1, sv, "m", k=3, scalebar_um=1)
    png = np.asarray(Image.open(tmp_path / "m.png").convert("RGB"))
    assert png.shape == (40 * 3, 60 * 3, 3)  # scale bar inside the image, no strip (2026-09-30)
    up = np.repeat(np.repeat(img, 3, 0), 3, 1)
    assert np.array_equal(png[:50], up[:50])  # untouched above the bar and its label
    assert g["scale_x"] == g["scale_y"] == 3


def test_micrograph_pixel_exact_awkward_height(tmp_path):
    style.apply_style()
    img = np.random.default_rng(1).integers(0, 256, (1894, 16, 3), dtype=np.uint8)
    twoarm.micrograph(img, .065, style.Saver(tmp_path), "m", k=1, scalebar_um=0.5, strip_px=120)
    png = np.asarray(Image.open(tmp_path / "m.png").convert("RGB"))
    assert png.shape[0] == 1894
    assert np.array_equal(png[:1700], img[:1700])
