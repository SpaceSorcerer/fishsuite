"""Physical-scale geometry of the orthogonal-view figure (2026-09-24).

XY, XZ and YZ must share one inches-per-µm scale (lateral and axial), the
figure must be sized from physical extents so no panel clips or overlaps at
any z-depth:crop ratio, scale bars must match their labels, and one channel
naming must be used in titles, legend and footer.
"""
import importlib
import itertools

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest


def core():
    return importlib.import_module('fishsuite.core.ortho_profile')


def _render(stack, center, half, dx, dz, **kw):
    mask = kw.pop('nucleus_mask', None)
    if mask is None:
        mask = np.zeros(stack.shape[2:], bool)
        cy, cx = center[1], center[2]
        mask[cy-half//2:cy+half//2+1, cx-half//2:cx+half//2+1] = True
    kw.setdefault('display_levels', ((0, 100), (0, 100)))
    fig = core().render_ortho_figure(stack, center, half, nucleus_mask=mask,
                                     pixel_size_um=dx, z_step_um=dz, **kw)
    fig.canvas.draw()
    return fig, {a.get_label(): a for a in fig.axes}


def _inches_per_data(ax, fig):
    t = ax.transData
    ox, oy = t.transform((0, 0))
    ux, _ = t.transform((1, 0))
    _, uy = t.transform((0, 1))
    return abs(ux-ox)/fig.dpi, abs(uy-oy)/fig.dpi


def _sphere_stack(radius_um, dx, dz, nz, nyx):
    z, y, x = np.indices((nz, nyx, nyx)).astype(float)
    cz, cyx = (nz-1)/2, (nyx-1)/2
    r2 = ((z-cz)*dz)**2 + ((y-cyx)*dx)**2 + ((x-cyx)*dx)**2
    ball = (r2 <= radius_um**2).astype(float)*100
    return np.stack([ball, ball, ball]), (int(cz), int(cyx), int(cyx))


@pytest.mark.parametrize('include_dapi', [False, True])
@pytest.mark.parametrize('factor', [1.0, 0.85])
def test_anisotropic_sphere_is_circular_in_xz_and_yz(include_dapi, factor):
    dx, dz, radius = .13, .21, 3.
    # The sphere is built at the *effective* z step so it is physically round
    # after the axial correction the renderer is told to apply.
    stack, center = _sphere_stack(radius, dx, dz*factor, 45, 81)
    fig, axes = _render(stack, center, 40, dx, dz, axial_scale_factor=factor,
                        include_dapi=include_dapi, dapi_display_level=(0, 100),
                        nucleus_mask=np.ones((81, 81), bool))
    xy_in_x, xy_in_y = _inches_per_data(axes['xy'], fig)
    for key in ('xz', 'yz'):
        ax = axes[key]
        mask = ax.images[0].get_array()[..., 0] > .5
        rows, cols = np.nonzero(mask)
        ix, iy = _inches_per_data(ax, fig)
        width_in = (cols.max()-cols.min()+1)*ix
        height_in = (rows.max()-rows.min()+1)*iy
        assert width_in/height_in == pytest.approx(1, rel=.05), key
    # One µm-per-inch for every image panel, lateral and axial.
    for key in ('xz', 'yz'):
        ix, iy = _inches_per_data(axes[key], fig)
        lateral, axial = (ix, iy) if key == 'xz' else (iy, ix)
        assert lateral/dx == pytest.approx(xy_in_x/dx, rel=1e-3)
        assert axial/(dz*factor) == pytest.approx(xy_in_x/dx, rel=1e-3)
    assert xy_in_x == pytest.approx(xy_in_y, rel=1e-3)
    plt.close(fig)


def _inside(inner, outer, tol=.5):
    return (inner.x0 >= outer.x0-tol and inner.y0 >= outer.y0-tol and
            inner.x1 <= outer.x1+tol and inner.y1 <= outer.y1+tol)


def _overlap(a, b, tol=.5):
    return (min(a.x1, b.x1)-max(a.x0, b.x0) > tol and
            min(a.y1, b.y1)-max(a.y0, b.y0) > tol)


@pytest.mark.parametrize('include_dapi', [False, True])
@pytest.mark.parametrize('ratio', [.2, .5, 1., 2., 3., 10.])
def test_no_panel_clips_or_overlaps_at_any_depth_ratio(ratio, include_dapi):
    dx, half = .1, 20
    n = 2*half+1
    dz = ratio*dx  # z window = n slices, so z extent / crop width = ratio
    stack = np.random.default_rng(0).random((3, n, 64, 64))*100
    fig, axes = _render(stack, (half, 32, 32), half, dx, dz,
                        include_dapi=include_dapi, dapi_display_level=(0, 100),
                        show_scale_bars=True, analysed_plane_z=half,
                        arm='VPR noDox', image='UD-MIAT-FISH-QKI-IF-VPR-no Dox_15.vsi',
                        nucleus_id=120, metric='nuclear_spot_count', metric_value=3, arm_median=3,
                        run_dir='F:/Image Analysis Work/MIAT-QKI-Coloc/UD/_OEvWT_2026-09/'
                                'RUN_PROD_OEvControl_PLAIN_jointAF_diam11um_LoG174_miat500_qki1050_20260917-203051')
    renderer = fig.canvas.get_renderer()
    fig_box = fig.bbox
    geo = fig._ortho_geometry
    assert geo['z_extent_um'] == pytest.approx(n*dz)
    assert geo['z_extent_um']/geo['crop_width_um'] == pytest.approx(ratio)
    for label, ax in axes.items():
        assert _inside(ax.get_window_extent(renderer), fig_box), label
        assert _inside(ax.get_tightbbox(renderer), fig_box), label
    for (la, a), (lb, b) in itertools.combinations(axes.items(), 2):
        assert not _overlap(a.get_window_extent(renderer), b.get_window_extent(renderer)), (la, lb)
    for la, a in axes.items():
        title = a.title.get_window_extent(renderer)
        for lb, b in axes.items():
            if la != lb and a.get_title():
                assert not _overlap(title, b.get_window_extent(renderer)), (la, lb)
    # Every rendered artist (legend, anchored header, annotations) is on the page.
    assert _inside(fig.get_tightbbox(renderer).transformed(fig.dpi_scale_trans), fig_box),         'figure content extends past the page'
    from matplotlib.offsetbox import AnchoredOffsetbox
    anchored = [a for a in fig.artists if isinstance(a, AnchoredOffsetbox)]
    assert anchored
    for artist in anchored:
        assert _inside(artist.get_window_extent(renderer), fig_box), 'header'
    legend = axes['raw'].get_legend()
    assert _inside(legend.get_window_extent(renderer), axes['raw'].get_window_extent(renderer)),         'profile legend clipped by the raw-intensity panel'
    for text in legend.get_texts():
        assert _inside(text.get_window_extent(renderer), axes['raw'].get_window_extent(renderer))
    # An axial bar is always drawn on both orthogonal views (A2).
    for key in ('xz', 'yz'):
        assert 'scale bar axial' in _bars(axes[key]), (key, ratio)
    for text in fig.texts:
        box = text.get_window_extent(renderer)
        assert _inside(box, fig_box), text.get_text()
        for label, ax in axes.items():
            assert not _overlap(box, ax.get_tightbbox(renderer)), (text.get_text()[:30], label)
    # XZ shares the XY x axis; YZ shares the XY y axis.
    xy, xz, yz = (axes[k].get_window_extent(renderer) for k in ('xy', 'xz', 'yz'))
    assert xz.x0 == pytest.approx(xy.x0, abs=1) and xz.x1 == pytest.approx(xy.x1, abs=1)
    assert xz.y1 < xy.y0
    assert yz.y0 == pytest.approx(xy.y0, abs=1) and yz.y1 == pytest.approx(xy.y1, abs=1)
    assert yz.x0 > xy.x1
    # Physical aspect is never stretched to square.
    assert (xz.height/xz.width) == pytest.approx(ratio, rel=.02)
    assert (yz.width/yz.height) == pytest.approx(ratio, rel=.02)
    plt.close(fig)


def _bars(ax):
    return {line.get_label(): line for line in ax.lines if line.get_label().startswith('scale bar')}


@pytest.mark.parametrize('factor', [1.0, .85])
def test_scale_bar_lengths_lateral_and_axial(factor):
    dx, dz, half = .13, .21, 48
    stack = np.zeros((3, 97, 128, 128))
    fig, axes = _render(stack, (48, 64, 64), half, dx, dz, axial_scale_factor=factor)
    inch_per_um = _inches_per_data(axes['xy'], fig)[0]/dx
    dz_eff = dz*factor
    # Expected bar lengths derived here from the inputs, not from production
    # metadata: 97 px crop and 97 z slices shown; nice value <= 40 % of extent.
    def nice(extent_um):
        return max(v for v in (.1, .2, .5, 1, 2, 5, 10, 20) if v <= .4*extent_um)
    lateral_um, axial_um = nice(97*dx), nice(97*dz_eff)
    independent = {'scale bar': lateral_um, 'scale bar lateral': lateral_um,
                   'scale bar axial': axial_um}
    expected = {('xy', 'scale bar'): (0, dx), ('xz', 'scale bar lateral'): (0, dx),
                ('xz', 'scale bar axial'): (1, dz_eff), ('yz', 'scale bar axial'): (0, dz_eff),
                ('yz', 'scale bar lateral'): (1, dx)}
    for (key, name), (axis, um_per_px) in expected.items():
        line = _bars(axes[key])[name]
        data = line.get_xdata() if axis == 0 else line.get_ydata()
        length_px = abs(data[1]-data[0])
        bar_um = independent[name]
        assert line._scale_bar_um == bar_um
        assert length_px == pytest.approx(bar_um/um_per_px)
        # Measured on the page: the bar is bar_um long at the shared scale.
        pts = axes[key].transData.transform(np.column_stack([line.get_xdata(), line.get_ydata()]))
        assert np.linalg.norm(pts[1]-pts[0])/fig.dpi == pytest.approx(bar_um*inch_per_um, rel=1e-3)
        label = line._scale_bar_text
        assert label.get_text().startswith(f'{bar_um:g} µm')
        assert ('axial' in name) == ('(z)' in label.get_text())
    geo = fig._ortho_geometry
    assert geo['axial_scale_factor'] == factor
    assert geo['z_step_effective_um'] == pytest.approx(dz_eff)
    footer = ' '.join(t.get_text() for t in fig.texts)
    assert f'axial scale factor {factor:g}' in footer
    plt.close(fig)


def test_axial_scale_factor_must_be_positive():
    stack = np.zeros((3, 9, 32, 32))
    for bad in (0, -1, float('nan')):
        with pytest.raises(ValueError, match='axial_scale_factor'):
            _render(stack, (4, 16, 16), 4, .13, .21, axial_scale_factor=bad)


@pytest.mark.parametrize('include_dapi', [False, True])
def test_channel_labels_consistent_in_titles_legend_and_footer(include_dapi):
    stack = np.random.default_rng(1).random((3, 21, 64, 64))*100
    labels = ('MIAT-640', 'QKI-561')
    fig, axes = _render(stack, (10, 32, 32), 10, .13, .21, channel_labels=labels,
                        include_dapi=include_dapi, dapi_display_level=(0, 100), qki_min=40)
    titles = ' '.join(ax.get_title() for ax in axes.values())
    legend = [t.get_text() for t in axes['raw'].get_legend().get_texts()]
    footer = ' '.join(t.get_text() for t in fig.texts)
    for label in labels:
        assert f'XY {label}' in titles
        assert label in legend
        assert label in footer
    everything = titles + ' '.join(legend) + footer
    for bare in ('MIAT', 'QKI'):
        # Every occurrence of the bare name is part of the preset label.
        assert everything.count(bare) == everything.count(f'{bare}-')
    plt.close(fig)


def test_display_levels_identical_across_panels_and_recorded():
    stack = np.zeros((3, 21, 64, 64))
    stack[0, 10, 32, 32] = 55.
    stack[1, 10, 32, 32] = 80.
    stack[2, 10, 32, 32] = 300.
    levels = ((10, 110), (30, 130))
    fig, axes = _render(stack, (10, 32, 32), 10, .13, .21, display_levels=levels,
                        include_dapi=True, dapi_display_level=(100, 500))
    y0, y1, x0, x1 = fig._ortho_crop_bounds
    z0, z1 = fig._ortho_geometry['z_range_0based']
    xy = axes['xy'].images[0].get_array()[32-y0, 32-x0]
    xz = axes['xz'].images[0].get_array()[10-z0, 32-x0]
    yz = axes['yz'].images[0].get_array()[32-y0, 10-z0]
    np.testing.assert_allclose(xy, xz)
    np.testing.assert_allclose(xy, yz)
    a, b, d = .45, .5, .5
    np.testing.assert_allclose(xy, [min(a+b, 1), a, min(b+d, 1)])
    np.testing.assert_allclose(axes['xy_miat'].images[0].get_array()[32-y0, 32-x0], [a, a, 0])
    np.testing.assert_allclose(axes['xy_qki'].images[0].get_array()[32-y0, 32-x0], [b, 0, b])
    np.testing.assert_allclose(axes['xy_dapi'].images[0].get_array()[32-y0, 32-x0], [0, 0, d])
    footer = ' '.join(t.get_text() for t in fig.texts)
    assert '10–110' in footer and '30–130' in footer and '100–500' in footer
    assert fig._ortho_geometry['display_levels'] == {
        'rna1': dict(label='MIAT', min=10., max=110.),
        'partner': dict(label='QKI', min=30., max=130.),
        'dapi': dict(label='DAPI', min=100., max=500.)}
    plt.close(fig)


def test_crosshairs_and_footer_state_z_range_and_interpolation():
    stack = np.zeros((3, 30, 64, 64))
    fig, axes = _render(stack, (12, 30, 34), 10, .13, .21, analysed_plane_z=12)
    y0, y1, x0, x1 = fig._ortho_crop_bounds
    z0, z1 = fig._ortho_geometry['z_range_0based']
    assert (z0, z1) == (2, 23)

    def cross(ax):
        lines = [l for l in ax.lines if l.get_label() == 'section position']
        return sorted((tuple(l.get_xdata()), tuple(l.get_ydata())) for l in lines)
    assert any(xs == (34-x0, 34-x0) for xs, _ in cross(axes['xy']))
    assert any(ys == (30-y0, 30-y0) for _, ys in cross(axes['xy']))
    assert any(ys == (12-z0, 12-z0) for _, ys in cross(axes['xz']))
    assert any(xs == (12-z0, 12-z0) for xs, _ in cross(axes['yz']))
    footer = ' '.join(t.get_text() for t in fig.texts)
    assert 'z 3–23 of 30' in footer
    assert 'nearest-neighbour' in footer
    assert fig._ortho_geometry['interpolation'] == 'nearest'
    plt.close(fig)


def test_duplicate_channel_labels_keep_both_display_levels():
    """M1: metadata is keyed by channel role, so equal labels cannot overwrite."""
    stack = np.zeros((3, 21, 64, 64))
    fig, axes = _render(stack, (10, 32, 32), 10, .13, .21, display_levels=((10, 110), (30, 130)),
                        channel_labels=('probe', 'probe'))
    levels = fig._ortho_geometry['display_levels']
    assert levels['rna1'] == dict(label='probe', min=10., max=110.)
    assert levels['partner'] == dict(label='probe', min=30., max=130.)
    plt.close(fig)


@pytest.mark.parametrize('ratio', [3., 10.])
def test_profile_panel_fits_its_full_legend_at_deep_stacks(ratio):
    """A1: at deep stacks the image scale shrinks; the profile legend must not clip."""
    dx, half = .1, 20
    stack = np.random.default_rng(2).random((3, 2*half+1, 64, 64))*100
    fig, axes = _render(stack, (half, 32, 32), half, dx, ratio*dx, qki_min=40, miat_min=20,
                        channel_labels=('MIAT-640', 'QKI-561'))
    renderer = fig.canvas.get_renderer()
    raw = axes['raw'].get_window_extent(renderer)
    legend = axes['raw'].get_legend()
    assert len(legend.get_texts()) == 8
    assert _inside(legend.get_window_extent(renderer), raw)
    assert raw.width/fig.dpi >= 3.0
    # The wider profile panel pushes XY/XZ right; they still share their x axis.
    xy, xz = (axes[k].get_window_extent(renderer) for k in ('xy', 'xz'))
    assert xz.x0 == pytest.approx(xy.x0, abs=1)
    assert not _overlap(raw, xz)
    plt.close(fig)
