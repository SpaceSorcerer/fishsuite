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
              display_levels=((0,100),(0,100)), run_dir='synthetic', qki_threshold=35)
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
    assert {text.get_text() for text in axes['normalised'].texts} == {
        'MIAT display min 0: outside trace range',
        'MIAT display max 100: outside trace range',
        'QKI display min 0: outside trace range',
        'QKI display max 100: outside trace range',
    }
    assert any(t.get_text() == 'synthetic' for t in f1.texts)
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
          display_levels=((-100,1000),(-100,1000)), qki_threshold=500)
    normal = next(ax for ax in fig.axes if ax.get_label() == 'normalised')
    assert normal.get_ylim() == pytest.approx((-.05,1.05))
    assert normal.get_ylabel() == "normalised to each channel's own min–max along this line (display only)"
    assert len(normal.lines) == 2
    labels = [text.get_text() for text in normal.texts]
    assert len(labels) == 5
    assert any('QKI analysis threshold' in label and '500' in label for label in labels)
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
    assert np.ptp(tick_yz.get_ydata()) == 0
    assert any('analysed plane' in t.get_text() and '17' in t.get_text() for t in axes['xz'].texts)
    assert any('analysed plane' in t.get_text() and '17' in t.get_text() for t in axes['yz'].texts)
    assert axes['xz'].get_aspect() == pytest.approx(.21/.13)
    assert axes['yz'].get_aspect() == pytest.approx(.13/.21)
    plt.close(fig)


def test_analysed_plane_ticks_do_not_expand_edge_crop(stack):
    fig = core().render_ortho_figure(stack, (20,0,0), 1,
          nucleus_mask=np.ones((128,128),bool), pixel_size_um=.13, z_step_um=.21,
          display_levels=((0,100),(0,100)), analysed_plane_z=20)
    axes = {a.get_label():a for a in fig.axes}
    assert axes['xz'].get_xlim() == pytest.approx((-.5,1.5))
    assert axes['yz'].get_ylim() == pytest.approx((1.5,-.5))
    plt.close(fig)
