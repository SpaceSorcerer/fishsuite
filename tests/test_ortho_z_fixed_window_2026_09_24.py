"""Render-only fixed-height z window for ortho figures (2026-09-24).

W um tall in effective (axially corrected) um, centred on the nucleus DAPI
half-max midpoint, shifted (not shrunk) to stay inside the stack; if the
stack is shorter than W the full stack is shown and flagged.
"""
import importlib

import numpy as np
import pytest


def core():
    return importlib.import_module('fishsuite.core.ortho_profile')


def _dapi(nz, center, sigma, nyx=30):
    z = np.arange(nz)[:, None, None]
    d = 100 + 1000*np.exp(-0.5*((z-center)/sigma)**2)*np.ones((1, nyx, nyx))
    m = np.zeros((nyx, nyx), bool)
    m[8:22, 8:22] = True
    return d, m


@pytest.mark.parametrize('center,sigma', [(30, 5.0), (40, 9.0), (45, 3.0)])
def test_window_height_constant_across_nuclei(center, sigma):
    d, m = _dapi(78, center, sigma)
    z0, z1, info = core().fixed_z_window(d, m, dz_eff_um=0.21*0.878, window_um=10.5)
    n = int(round(10.5/(0.21*0.878)))
    assert z1 - z0 == n
    mid = (info['core_first_0based'] + info['core_last_0based'])/2
    assert abs((z0 + z1 - 1)/2 - mid) <= 1
    assert not info['full_stack_shorter_than_window']


def test_window_shifted_inside_stack_not_shrunk():
    d, m = _dapi(78, 3, 3.0)
    z0, z1, info = core().fixed_z_window(d, m, dz_eff_um=0.21, window_um=6.3)
    assert (z0, z1) == (0, 30)
    d, m = _dapi(78, 76, 3.0)
    z0, z1, _ = core().fixed_z_window(d, m, dz_eff_um=0.21, window_um=6.3)
    assert (z0, z1) == (48, 78)


def test_stack_shorter_than_window_shows_full_stack_and_flags():
    d, m = _dapi(20, 10, 3.0)
    z0, z1, info = core().fixed_z_window(d, m, dz_eff_um=0.21, window_um=10.5)
    assert (z0, z1) == (0, 20) and info['full_stack_shorter_than_window']


def test_fixed_window_must_include_section_plane():
    d, m = _dapi(78, 60, 3.0)
    with pytest.raises(ValueError):
        core().fixed_z_window(d, m, dz_eff_um=0.21, window_um=2.1, must_include=(5, 60))


def test_fixed_window_footer_note():
    note = core().fixed_z_window_note(dict(window_um=10.5, n_planes=57, full_stack_shorter_than_window=False,
                                           core_first_0based=18, core_last_0based=54))
    assert 'fixed 10.5 µm window' in note and 'DAPI half-max midpoint' in note
    note2 = core().fixed_z_window_note(dict(window_um=10.5, n_planes=20, full_stack_shorter_than_window=True,
                                            core_first_0based=3, core_last_0based=15))
    assert 'stack shorter than the 10.5 µm window' in note2


def test_cli_exposes_fixed_mode():
    from click.testing import CliRunner
    from fishsuite.cli import cli
    out = CliRunner().invoke(cli, ['ortho', '--help']).output
    assert 'fixed' in out and '--z-window-um' in out


def test_axial_scale_note_replaces_nominal_wording_in_footer():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    stack = np.zeros((3, 40, 64, 64))
    mask = np.zeros((64, 64), bool)
    mask[25:36, 25:36] = True
    note = 'GLOX n≈1.333 / oil n=1.518; first-order correction'
    fig = core().render_ortho_figure(stack, (20, 30, 30), 10, nucleus_mask=mask, pixel_size_um=.13,
                                     z_step_um=.21, display_levels=((0, 100), (0, 100)),
                                     axial_scale_factor=0.878, axial_scale_note=note)
    footer = ' '.join(t.get_text() for t in fig.texts)
    assert 'axial scale factor 0.878 (' + note in footer.replace('\n', ' ')
    assert 'no refractive-index correction' not in footer
    plt.close(fig)
    from click.testing import CliRunner
    from fishsuite.cli import cli
    assert '--axial-scale-note' in CliRunner().invoke(cli, ['ortho', '--help']).output
