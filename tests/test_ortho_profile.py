import importlib
import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def core():
    return importlib.import_module('fishsuite.core.ortho_profile')


@pytest.fixture
def stack():
    z, y, x = np.indices((40, 128, 128))
    blob = np.exp(-((z-20)**2/8 + (y-64)**2/18 + (x-64)**2/18))
    return np.stack([blob*100, blob*70, blob*20])


def test_select_median_shuffle_invariant():
    df = pd.DataFrame({'arm':['a']*6+['b']*6, 'image':['im']*12,
                       'nucleus_id':range(12), 'metric':[0,2,2,4,4,9]*2})
    a = core().select_nuclei(df, 'arm', 'metric', 3, 3)
    b = core().select_nuclei(df.sample(frac=1, random_state=9), 'arm', 'metric', 3, 3)
    pd.testing.assert_frame_equal(a, b)
    assert list(a.arm_median) == [3]*6
    assert set(a.metric) <= {2,4}


def test_punctum_rules_and_missing_z():
    df = pd.DataFrame({'nucleus_id':[1]*3, 'z_px':[10,20,30], 'y_px':[2]*3,
                       'x_px':[3]*3, 'peak_intensity':[9,2,4]})
    assert core().pick_punctum(df, 1) == (10,2,3)
    assert core().pick_punctum(df, 1, 'median') == (30,2,3)
    assert np.isnan(core().pick_punctum(df.drop(columns='z_px'), 1)[0])


def test_sections_and_profile(stack):
    m = core()
    sections = m.ortho_sections(stack, (20,64,64), 12)
    assert np.unravel_index(sections['xz'][0].argmax(), sections['xz'][0].shape) == (12,12)
    assert np.unravel_index(sections['yz'][0].argmax(), sections['yz'][0].shape) == (12,12)
    dist, intensity = m.line_profile(stack[:,20], (64,40), (64,88), pixel_size_um=.13)
    assert abs(intensity[0].argmax()-intensity[1].argmax()) <= 1
    assert dist[-1] == pytest.approx(48*.13)


def test_fixed_levels_aspect_and_annotations(stack):
    mask = np.zeros((128,128), bool)
    mask[45:85,45:85] = True
    kw = dict(nucleus_mask=mask, pixel_size_um=.13, z_step_um=.21,
              display_levels=((0,100),(0,100)), run_dir='synthetic', qki_min=35)
    f1 = core().render_ortho_figure(stack, (20,64,64), 24, **kw)
    altered = stack.copy()
    altered[:,20,50,50] = 10000
    f2 = core().render_ortho_figure(altered, (20,64,64), 24, **kw)
    axes = {a.get_label():a for a in f1.axes}
    assert axes['xz'].get_aspect() == pytest.approx(.21/.13)
    np.testing.assert_equal(axes['xy'].images[0].get_array()[24,24],
                            f2.axes[0].images[0].get_array()[24,24])
    assert len(axes['raw'].lines) == 7
    assert len(axes['normalised'].lines) == 3  # traces and in-range QKI threshold
    assert set(axes['normalised']._outside_trace_bounds) == {
        'MIAT display min 0: outside trace range',
        'MIAT display max 100: outside trace range',
        'QKI display min 0: outside trace range',
        'QKI display max 100: outside trace range',
    }
    assert any('synthetic' in t.get_text() for t in f1.texts)
    assert axes['xy'].collections
    plt.close(f1)
    plt.close(f2)


def test_legacy_profile_unchanged():
    from skimage.measure import profile_line
    plane = np.arange(100).reshape(10,10)
    actual = core().sample_profile(plane, (1,2), (7,8))
    expected = profile_line(plane.astype(float), (1,2), (7,8), linewidth=1,
                            order=1, mode='constant', reduce_func=None).ravel()
    np.testing.assert_array_equal(actual, expected)


def test_punctum_excludes_assigned_cytoplasmic_spots():
    df = pd.DataFrame({'nucleus_id':[1,1], 'z_px':[1,2], 'y_px':[3,4],
                       'x_px':[5,6], 'peak_intensity':[10,100], 'in_nucleus':[1,0]})
    assert core().pick_punctum(df, 1) == (1,3,5)

def test_orthogonal_panels_share_physical_scale(stack):
    mask = np.ones((128,128),bool)
    fig = core().render_ortho_figure(stack,(20,64,64),24,nucleus_mask=mask,
          pixel_size_um=.13,z_step_um=.21,display_levels=((0,100),(0,100)))
    fig.canvas.draw()
    axes = {a.get_label():a for a in fig.axes}
    def scale(ax,vector):
        return np.linalg.norm(ax.transData.transform(vector)-ax.transData.transform((0,0)))
    assert scale(axes['xy'],(1,0)) == pytest.approx(scale(axes['xz'],(1,0)),rel=.01)
    assert scale(axes['xy'],(0,1)) == pytest.approx(scale(axes['yz'],(0,1)),rel=.01)
    assert axes['yz'].get_aspect() == pytest.approx(.13/.21)
    plt.close(fig)


def test_profile_requires_explicit_keyword_pixel_size():
    plane = np.ones((3, 12, 12))
    with pytest.raises(TypeError, match='pixel_size_um'):
        core().line_profile(plane, (4, 2), (4, 8))
    with pytest.raises(TypeError):
        core().line_profile(plane, (4, 2), (4, 8), 3, .13)


def test_punctum_does_not_assume_z_slice_is_full_stack_z():
    df = pd.DataFrame({'nucleus_id':[1], 'z_slice':[8], 'y_px':[2],
                       'x_px':[3], 'peak_intensity':[9]})
    assert np.isnan(core().pick_punctum(df, 1)[0])


def test_default_profile_matches_visible_line_and_requested_width(stack):
    mask = np.ones((128,128), bool)
    endpoints = core().clipped_default_line(mask, (20,64,64), 12)
    assert endpoints == ((64.,52.),(64.,76.))
    fig = core().render_ortho_figure(stack, (20,64,64), 12, nucleus_mask=mask,
          pixel_size_um=.13, z_step_um=.21, display_levels=((0,100),(0,100)),
          profile_width_px=5)
    axes = {a.get_label():a for a in fig.axes}
    np.testing.assert_array_equal(axes['xy'].lines[2].get_xdata(), [0,24])
    expected_distance, expected_values = core().line_profile(stack[:,20], *endpoints,
                                                   width_px=5, pixel_size_um=.13)
    np.testing.assert_array_equal(axes['raw'].lines[0].get_xdata(), expected_distance)
    np.testing.assert_array_equal(axes['raw'].lines[0].get_ydata(), expected_values[0])
    plt.close(fig)


def test_normalised_limits_and_out_of_range_bound_annotations(stack):
    fig = core().render_ortho_figure(stack, (20,64,64), 12,
          nucleus_mask=np.ones((128,128),bool), pixel_size_um=.13, z_step_um=.21,
          display_levels=((-100,1000),(-100,1000)), qki_min=500)
    normal = next(ax for ax in fig.axes if ax.get_label() == 'normalised')
    assert normal.get_ylim() == pytest.approx((-.05,1.05))
    assert normal.get_ylabel() == 'normalised (display only)'
    assert len(normal.lines) == 2
    labels = normal._outside_trace_bounds
    assert len(labels) == 5
    assert any('QKI analysis min (user)' in label and '500' in label for label in labels)
    assert all('outside' in label for label in labels)
    plt.close(fig)


def test_analysed_plane_ticks_preserve_orthogonal_geometry(stack):
    fig = core().render_ortho_figure(stack, (20,64,64), 12,
          nucleus_mask=np.ones((128,128),bool), pixel_size_um=.13, z_step_um=.21,
          display_levels=((0,100),(0,100)), analysed_plane_z=17)
    axes = {a.get_label():a for a in fig.axes}
    tick_xz = next(line for line in axes['xz'].lines if line.get_label() == 'analysed plane')
    tick_yz = next(line for line in axes['yz'].lines if line.get_label() == 'analysed plane')
    np.testing.assert_array_equal(tick_xz.get_ydata(), [9,9])
    assert np.mean(tick_yz.get_xdata()) == pytest.approx(9)
    assert np.ptp(tick_yz.get_xdata()) == 0
    assert np.ptp(tick_yz.get_ydata()) > 0
    assert any('z = 18' in t.get_text() for t in axes['xz'].texts)
    assert any('z = 18' in t.get_text() for t in axes['yz'].texts)
    assert axes['xz'].get_aspect() == pytest.approx(.21/.13)
    assert axes['yz'].get_aspect() == pytest.approx(.13/.21)
    plt.close(fig)


def test_analysed_plane_ticks_do_not_expand_edge_crop(stack):
    fig = core().render_ortho_figure(stack, (20,0,0), 1,
          nucleus_mask=np.pad(np.ones((2,2),bool),((0,126),(0,126))), pixel_size_um=.13, z_step_um=.21,
          display_levels=((0,100),(0,100)), analysed_plane_z=20)
    axes = {a.get_label():a for a in fig.axes}
    assert axes['xz'].get_xlim() == pytest.approx((-.5,1.5))
    assert axes['yz'].get_ylim() == pytest.approx((1.5,-.5))
    plt.close(fig)


def test_auto_crop_contains_bbox_and_crosshair_stays_on_punctum(stack):
    mask = np.zeros((128,128),bool)
    mask[20:105,35:95] = True
    fig = core().render_ortho_figure(stack,(20,25,40),nucleus_mask=mask,
          pixel_size_um=.13,z_step_um=.21,display_levels=((0,100),(0,100)))
    y0,y1,x0,x1 = fig._ortho_crop_bounds
    assert y0 <= 20 and y1 >= 105 and x0 <= 35 and x1 >= 95
    xy = next(ax for ax in fig.axes if ax.get_label() == 'xy')
    assert xy.lines[0].get_xdata()[0] == 40-x0
    assert xy.lines[1].get_ydata()[0] == 25-y0
    bars = [line for line in xy.lines if line.get_label() == 'scale bar']
    assert len(bars) == 1
    length = np.ptp(bars[0].get_xdata())*.13
    assert length == pytest.approx(core().scale_bar_length((x1-x0)*.13))
    assert round(length) in {1,2,5,10,20}
    assert length <= (x1-x0)*.13*.40
    plt.close(fig)


# 2026-09-24 round 2 (A2): sub-µm candidates so shallow z windows still get an
# axial bar; nothing fits below .25 µm.
@pytest.mark.parametrize('panel_width,expected',[(4,1),(8,2),(12.5,5),(20,5),(25,10),(40,10),(50,20),(100,20),(2,.5),(1,.2),(.5,.2),(.3,.1),(.2,None)])
def test_round_scale_bar(panel_width,expected):
    assert core().scale_bar_length(panel_width) == expected


def test_header_channels_and_explicit_analysis_minima(stack):
    kw = dict(nucleus_mask=np.ones((128,128),bool),pixel_size_um=.13,z_step_um=.21,
              display_levels=((0,100),(0,100)),arm='arm A',image='example.vsi',
              nucleus_id=12,analysed_plane_z=20,metric='nuclear_spot_count',metric_value=8,arm_median=7)
    for minimum in (None,35):
        fig = core().render_ortho_figure(stack,(20,64,64),24,qki_min=minimum,**kw)
        axes = {ax.get_label():ax for ax in fig.axes}
        labels = [line.get_label() for line in axes['raw'].lines]
        assert ('QKI analysis min (user)' in labels) == (minimum is not None)
        assert not any('analysis threshold' in label for label in labels)
        assert all(word in fig._ortho_header for word in ('arm A','example.vsi','nucleus 12','z = 21 (1-based)','8 vs arm median 7'))
        a = axes['xy_miat'].images[0].get_array()
        b = axes['xy_qki'].images[0].get_array()
        merged = axes['xy'].images[0].get_array()
        np.testing.assert_array_equal(merged,np.clip(a+b,0,1))
        plt.close(fig)


def test_full_merge_has_dapi_calibrated_bars_and_linked_profile_line(stack):
    mask = np.zeros((128,128), bool)
    mask[45:85,45:85] = True
    fig = core().render_ortho_figure(
        stack, (20,64,64), 12, nucleus_mask=mask, pixel_size_um=.13, z_step_um=.21,
        display_levels=((0,100),(0,100)), dapi_display_level=(0,100),
        include_dapi=True, channel_labels=('BIN1 introns','RNASEH2B'),
        profile_labels=('BIN1 introns','RNASEH2B'), show_scale_bars=True,
        show_z_slice_labels=True, analysed_plane_z=17,
        line_endpoints=((64.,52.),(64.,76.)))
    axes = {a.get_label(): a for a in fig.axes}
    assert {'xy_dapi','xy_miat','xy_qki','xy','xz','yz','raw','normalised'} <= set(axes)
    assert axes['xy_dapi'].get_title() == 'XY DAPI'
    assert 'BIN1 introns' in axes['xy'].get_title()
    assert axes['xy'].images[0].get_array().shape[-1] == 3
    np.testing.assert_allclose(axes['xy_miat'].images[0].get_array()[..., 2], 0)
    assert len([line for line in axes['xy'].lines if line.get_label() == 'scale bar']) == 1
    assert len([line for line in axes['xz'].lines if line.get_label().startswith('scale bar')]) == 2
    assert len([line for line in axes['yz'].lines if line.get_label().startswith('scale bar')]) == 2
    assert [t.get_text() for t in axes['xy'].texts].count('z = 18') == 1
    assert sum(t.get_text() == 'z = 18' for ax in (axes['xz'], axes['yz']) for t in ax.texts) == 0
    profile_lines = [line for line in axes['xy'].lines if line.get_label() == 'measured profile line']
    assert len(profile_lines) == 1
    np.testing.assert_array_equal(profile_lines[0].get_xdata(), [0,24])
    assert not any(line.get_label() == 'measured profile point' for line in axes['yz'].lines)
    assert len([line for line in axes['xz'].lines if line.get_label() == 'measured profile line']) == 1
    np.testing.assert_array_equal(axes['raw'].lines[0].get_xdata(),
                                  core().line_profile(stack[:,20], (64,52), (64,76),
                                                       pixel_size_um=.13)[0])
    plt.close(fig)


def test_real_length_header_footer_and_panel_titles_fit(stack):
    from matplotlib.offsetbox import AnchoredOffsetbox
    fig = core().render_ortho_figure(stack,(20,64,64),24,
          nucleus_mask=np.ones((128,128),bool),pixel_size_um=.13,z_step_um=.21,
          display_levels=((0,100),(0,100)),arm='g2 noDox (control)',
          image='UD-MIAT-FISH-QKI-IF-VPR-no Dox_15.vsi',nucleus_id=120,
          analysed_plane_z=17,metric='nuclear_spot_count',metric_value=300,arm_median=250,
          run_dir='F:/Image Analysis Work/MIAT-QKI-Coloc/UD/_OEvWT_2026-09/'
                  'RUN_QC2_OEvControl_PLAIN_jointAF_diam11um_autoLoG_footprintcols_20260917-153923')
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = list(fig.texts)+[a for a in fig.artists if isinstance(a,AnchoredOffsetbox)]
    for artist in texts:
        bounds = artist.get_window_extent(renderer)
        assert bounds.x0 >= 0 and bounds.x1 <= fig.bbox.x1
        assert bounds.y0 >= 0 and bounds.y1 <= fig.bbox.y1
    axes = {ax.get_label():ax for ax in fig.axes}
    xz_title = axes['xz'].title.get_window_extent(renderer)
    for text in axes['xy'].texts:
        assert not xz_title.overlaps(text.get_window_extent(renderer))
    plt.close(fig)


@pytest.mark.parametrize('nz',[20,40,60])
@pytest.mark.parametrize('minimum',[None,42])
def test_fixed_xy_width_and_orthogonal_layout_independent_of_stack_depth(nz,minimum):
    stack = np.zeros((3,nz,128,128))
    mask = np.zeros((128,128),bool)
    mask[32:96,32:96] = True
    fig = core().render_ortho_figure(stack,(nz//2,64,64),48,nucleus_mask=mask,
          pixel_size_um=.13,z_step_um=.21,display_levels=((0,100),(0,100)),
          qki_min=minimum,miat_min=minimum,analysed_plane_z=nz//2)
    fig.canvas.draw()
    # 2026-09-24: figure size is computed from physical extents (was a fixed
    # 11 x 6.5 in, which clipped YZ / collided XZ with the footer on deep stacks).
    width, height = fig.get_size_inches()
    np.testing.assert_allclose((width, height), fig._ortho_geometry['figure_size_in'])
    axes = {ax.get_label():ax for ax in fig.axes}
    xy,xz,raw,normal = [axes[key].get_position() for key in ('xy','xz','raw','normalised')]
    assert xy.width*width == pytest.approx(2.3)
    assert xy.height*height == pytest.approx(2.3)
    assert xz.x0 == pytest.approx(xy.x0)
    assert xz.width == pytest.approx(xy.width)
    assert xz.height*height == pytest.approx(2.3*nz*.21/(97*.13))
    assert 0 < (xy.y0-xz.y1)*height < .6
    assert raw.x0 == pytest.approx(axes['xy_miat'].get_position().x0)
    assert raw.x1 == pytest.approx(axes['xy_qki'].get_position().x1)
    assert normal.x0 == pytest.approx(axes['yz'].get_position().x0)
    legend = axes['raw'].get_legend()
    assert legend._ncols == len(legend.get_texts())
    assert legend.get_frame().get_alpha() == pytest.approx(.6)
    legend_bounds = legend.get_window_extent(fig.canvas.get_renderer())
    raw_bounds = axes['raw'].get_window_extent()
    assert legend_bounds.x0 >= raw_bounds.x0
    assert legend_bounds.x1 <= raw_bounds.x1
    assert legend_bounds.y0 >= raw_bounds.y0
    assert legend_bounds.y1 <= raw_bounds.y1
    renderer = fig.canvas.get_renderer()
    normal_ax = axes['normalised']
    normal_label = normal_ax.yaxis.label.get_window_extent(renderer)
    assert normal_label.x1 <= fig.bbox.x1
    assert not normal_label.overlaps(axes['xz'].get_window_extent())
    for text in axes['xz'].texts:
        assert not text.get_window_extent(renderer).overlaps(normal_ax.get_window_extent())
    for text in axes['yz'].texts:
        assert not text.get_window_extent(renderer).overlaps(normal_ax.title.get_window_extent(renderer))
    plt.close(fig)
