"""Linked ortho: nuclear outline on every nucleus-level panel (B2 XY merge + singles + XZ + YZ, B3 zoom), never on the
B1 whole-field panel (Brian 2026-09-30). Checked on rendered pixels."""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest

from fishsuite.figures import linked_set

OUTLINE_RGB = np.array([0x8b, 0xa6, 0xc4])


class _Saver:
    def __init__(self):
        self.figs = {}

    def save(self, fig, stem):
        fig.set_dpi(300)
        fig.canvas.draw()
        self.figs[stem] = (fig, np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(int).copy())


class _Run:
    arm = "VPR noDox"
    run_dir = "X/run"

    def __init__(self, tmp, H=120, W=160, Z=30, z0=15):
        yy, xx = np.mgrid[:H, :W]
        lab = np.zeros((H, W), np.int32)
        lab[((yy - 60) / 22.0) ** 2 + ((xx - 60) / 30.0) ** 2 <= 1] = 1
        lab[((yy - 60) / 15.0) ** 2 + ((xx - 130) / 15.0) ** 2 <= 1] = 2
        stack = np.zeros((3, Z, H, W), np.float32)
        stack[2, 8:22] = np.where(lab > 0, 5000.0, 700.0)
        stack[0, :, 60, 50:70] = 900.0
        self.stack, self.z0, self.key, self.data, self.saver = stack, z0, "img_14.vsi", tmp, _Saver()
        self.plane = dict(dapi=stack[2, z0], miat=stack[0, z0], qki=stack[1, z0], nucleus_labels=lab, z0=z0)
        self.spots = pd.DataFrame(dict(image=self.key, nucleus_id=1, spot_id=[1, 2, 3], center_y_px=[60.0, 60.0, 61.0],
                                       center_x_px=[52.0, 60.0, 68.0], miat_footprint_mean_raw=[900.0, 1200.0, 800.0],
                                       footprint_mean_qki=[1100.0, 1100.0, 1100.0]))

    def selection(self):
        return {"pixel_size_um": [0.065], "z_step_um_nominal": [0.21]}

    def image_key(self, f):
        return self.key

    def load_plane(self, key):
        return self.plane

    def spots_all(self):
        return self.spots


def _pix(fig, img, ax):
    bb = ax.get_window_extent()
    h = img.shape[0]
    return img[int(h - bb.y1) + 1:int(h - bb.y0) - 1, int(bb.x0) + 1:int(bb.x1) - 1]


def _n_outline(px, tol=3):
    return int((np.abs(px - OUTLINE_RGB).max(axis=-1) <= tol).sum())


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    run = _Run(tmp_path_factory.mktemp("ls"))
    orig = linked_set.read_stack
    linked_set.read_stack = lambda r, k: run.stack
    try:
        linked_set.build_linked_set(run, "14", 1, 0.5)
        linked_set.build_linked_set(run, "14", 1, 0.5, mix=True)
    finally:
        linked_set.read_stack = orig
    return run


@pytest.mark.parametrize("sfx", ["", "_merge_MIATxQKI"])
def test_b2_every_nucleus_panel_has_outline(rendered, sfx):
    fig, img = rendered.saver.figs[f"B2_nucleus_ortho_VPRnoDox_field14_nuc1_fishsuite{sfx}"]
    labels = {a.get_label() for a in fig.axes}
    want = {"xy", "xy_miat", "xy_qki", "xz", "yz"} | ({"xy_dapi"} if not sfx else set())
    assert want <= labels
    for ax in fig.axes:
        if ax.get_label() in want:
            assert _n_outline(_pix(fig, img, ax)) > 50, ax.get_label()
            assert any(c.get_label() == "nucleus outline" for c in ax.collections), ax.get_label()


@pytest.mark.parametrize("sfx", ["", "_merge_MIATxQKI"])
def test_b3_zoom_has_outline(rendered, sfx):
    fig, img = rendered.saver.figs[f"B3_nucleus_zoom_line_VPRnoDox_field14_nuc1{sfx}"]
    ax = fig.axes[0]
    cs = [c for c in ax.collections if hasattr(c, "allsegs")]
    assert cs, "no contour"
    verts = np.concatenate([np.asarray(s) for s in cs[0].allsegs[0]])
    disp = ax.transData.transform(verts)
    h = img.shape[0]
    bright = [img[int(h - y) - 1:int(h - y) + 2, int(x) - 1:int(x) + 2].min(axis=-1).max() > 100 for x, y in disp[::5]
              if 1 <= int(h - y) < h - 1 and 1 <= int(x) < img.shape[1] - 1]
    assert np.mean(bright) > .8


@pytest.mark.parametrize("sfx", ["", "_merge_MIATxQKI"])
def test_b1_field_has_no_outline(rendered, sfx):
    fig, img = rendered.saver.figs[f"B1_FOV_ortho_VPRnoDox_field14_nuc1_box{sfx}"]
    for ax in fig.axes:
        assert _n_outline(_pix(fig, img, ax)) == 0
        assert not any(c.get_label() == "nucleus outline" or hasattr(c, "allsegs") for c in ax.collections)
