"""B2 ortho (render_ortho_figure) fixes, Brian 2026-09-30: XZ/YZ nucleus edges = XY outline extent (not the 1-row
chord at the section row), true physical aspect, crosshairs 'none'|'single'|'all', dotted profile line by default."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from fishsuite.core.ortho_profile import render_ortho_figure

PX, DZ, AX = .065, .21, .878


def _oblique(H=160, W=160, Z=40):
    yy, xx = np.mgrid[:H, :W]
    u, v = (xx - 80) * .7071 + (yy - 80) * .7071, -(xx - 80) * .7071 + (yy - 80) * .7071
    mask = (u / 55.) ** 2 + (v / 25.) ** 2 <= 1          # tilted ellipse, like KD_2_07 n16
    stack = np.zeros((3, Z, H, W), np.float32)
    stack[:, 10:30] = np.where(mask, 1000., 100.)
    return stack, mask


def _fig(**kw):
    stack, mask = _oblique()
    ys, xs = np.nonzero(mask)
    cy = int(ys.max()) - 8                                 # off-centre punctum near the nucleus tip
    cx = int(np.nonzero(mask[cy])[0].mean())
    fig = render_ortho_figure(stack, (20, cy, cx), None, nucleus_mask=mask, pixel_size_um=PX, z_step_um=DZ,
                              axial_scale_factor=AX, display_levels=((0, 1000), (0, 1000)), analysed_plane_z=20,
                              line_endpoints=((cy, cx - 10.), (cy, cx + 5.)), **kw)
    fig.canvas.draw()
    return fig, {a.get_label(): a for a in fig.axes}, mask, (cy, cx)


def _outline(ax):
    return np.concatenate([c.get_segments() for c in ax.collections if c.get_label() == "nucleus outline"][0])


def _edges(ax, i):
    segs = np.array([c.get_segments() for c in ax.collections if c.get_label() == "nucleus outline"][0])
    return sorted(segs[:, 0, i])


def test_xz_yz_edges_match_xy_outline_extent():
    fig, ax, mask, (cy, cx) = _fig()
    v = _outline(ax["xy"])
    xz, yz = _edges(ax["xz"], 0), _edges(ax["yz"], 1)
    assert len(xz) == 2 and len(yz) == 2
    np.testing.assert_allclose(xz, [v[:, 0].min(), v[:, 0].max()], atol=1)
    np.testing.assert_allclose(yz, [v[:, 1].min(), v[:, 1].max()], atol=1)
    row = np.nonzero(mask[cy])[0]                            # the old chord is strictly narrower here
    assert row.max() - row.min() + 1 < xz[1] - xz[0] - 2
    plt.close(fig)


def test_xz_yz_aspect_is_physical():
    fig, ax, _, _ = _fig()
    want = DZ * AX / PX
    assert fig._ortho_geometry["axial_data_aspect"] == pytest.approx(want, rel=1e-9)

    def per_inch(a):
        bb = a.get_window_extent()
        (x0, x1), (y0, y1) = a.get_xlim(), a.get_ylim()
        return abs(x1 - x0) / (bb.width / fig.dpi), abs(y1 - y0) / (bb.height / fig.dpi)
    xzx, xzy = per_inch(ax["xz"])
    yzx, yzy = per_inch(ax["yz"])
    xyx, xyy = per_inch(ax["xy"])
    assert xzx / xzy == pytest.approx(want, rel=.02)       # pixels per inch: lateral / axial
    assert yzy / yzx == pytest.approx(want, rel=.02)
    um = [xyx * PX, xyy * PX, xzx * PX, xzy * DZ * AX, yzx * DZ * AX, yzy * PX]  # displayed µm per inch, all equal
    np.testing.assert_allclose(um, um[0], rtol=.02)
    plt.close(fig)


def test_profile_line_dotted_by_default_and_solid_on_request():
    fig, ax, _, _ = _fig()
    for k in ("xy", "xz"):
        ls = [l.get_linestyle() for l in ax[k].lines if l.get_label() == "measured profile line"]
        assert ls == [":"], k
    plt.close(fig)
    fig, ax, _, _ = _fig(profile_line_style="solid")
    assert [l.get_linestyle() for l in ax["xy"].lines if l.get_label() == "measured profile line"] == ["-"]
    plt.close(fig)


@pytest.mark.parametrize("mode,n_xy,n_ortho", [("single", 2, 0), ("none", 0, 0), ("all", 2, 2)])
def test_crosshair_modes(mode, n_xy, n_ortho):
    fig, ax, _, _ = _fig(**({} if mode == "single" else {"crosshairs": mode}))
    n = lambda a: sum(l.get_label() == "section position" for l in a.lines)
    assert n(ax["xy"]) == n_xy
    assert n(ax["xz"]) == n_ortho and n(ax["yz"]) == n_ortho
    for k in ("xy_miat", "xy_qki"):
        assert n(ax[k]) == 0
    plt.close(fig)


def test_bad_options_raise():
    with pytest.raises(ValueError):
        _fig(crosshairs="double")
    with pytest.raises(ValueError):
        _fig(profile_line_style="dashed")
