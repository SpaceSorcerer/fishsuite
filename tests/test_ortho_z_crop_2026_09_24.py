"""Render-only z crop of the ortho figure to the nucleus's own axial extent (2026-09-24).

The nucleus z extent is the contiguous run of planes around the peak of the
mean DAPI signal inside the saved 2-D nucleus mask whose min-max normalised
value is >= 0.5 (axial half-maximum), padded by ceil(margin_um / z step)
planes each side and clipped to the stack. Display only: no metric changes.
"""
import importlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest


def core():
    return importlib.import_module('fishsuite.core.ortho_profile')


def _dapi_gauss(nz=78, center=40, sigma_planes=8.0, nyx=40):
    z = np.arange(nz)[:, None, None]
    prof = np.exp(-0.5*((z-center)/sigma_planes)**2)
    dapi = 100 + 1000*prof*np.ones((1, nyx, nyx))
    mask = np.zeros((nyx, nyx), bool)
    mask[10:30, 10:30] = True
    return dapi, mask


def test_nucleus_z_extent_is_axial_half_max_plus_margin():
    dapi, mask = _dapi_gauss()
    z0, z1, info = core().nucleus_z_extent(dapi, mask, z_step_um=0.21, margin_um=1.0)
    hw = int(np.floor(8.0*np.sqrt(2*np.log(2))))  # half-max half-width in planes = 9
    margin = int(np.ceil(1.0/0.21))  # 5 planes
    assert (info['core_first_0based'], info['core_last_0based']) == (40-hw, 40+hw)
    assert (z0, z1) == (40-hw-margin, 40+hw+margin+1)
    assert info['margin_planes'] == margin


def test_nucleus_z_extent_clips_to_stack_and_keeps_required_planes():
    dapi, mask = _dapi_gauss(nz=30, center=2, sigma_planes=3)
    z0, z1, info = core().nucleus_z_extent(dapi, mask, z_step_um=0.21, margin_um=1.0,
                                           must_include=(20,))
    assert z0 == 0 and z1 == 21


def test_nucleus_z_extent_rejects_empty_mask():
    dapi, mask = _dapi_gauss()
    with pytest.raises(ValueError):
        core().nucleus_z_extent(dapi, np.zeros_like(mask), z_step_um=0.21, margin_um=1.0)


def test_render_honours_z_range_and_states_crop_in_footer():
    stack = np.zeros((3, 78, 64, 64))
    mask = np.zeros((64, 64), bool)
    mask[25:36, 25:36] = True
    fig = core().render_ortho_figure(stack, (40, 30, 30), 10, nucleus_mask=mask,
                                     pixel_size_um=.13, z_step_um=.21,
                                     display_levels=((0, 100), (0, 100)),
                                     analysed_plane_z=40, z_range=(26, 55),
                                     z_crop_note='Z cropped to nucleus DAPI half-max ± 1 µm')
    fig.canvas.draw()
    assert fig._ortho_geometry['z_range_0based'] == (26, 55)
    assert abs(fig._ortho_geometry['z_extent_um'] - 29*.21) < 1e-9
    footer = ' '.join(t.get_text() for t in fig.texts)
    assert 'z 27–55 of 78' in footer
    assert 'Z cropped to nucleus DAPI half-max' in footer
    plt.close(fig)


def test_render_rejects_z_range_excluding_section_plane():
    stack = np.zeros((3, 78, 64, 64))
    mask = np.zeros((64, 64), bool)
    mask[25:36, 25:36] = True
    with pytest.raises(ValueError):
        core().render_ortho_figure(stack, (40, 30, 30), 10, nucleus_mask=mask,
                                   pixel_size_um=.13, z_step_um=.21,
                                   display_levels=((0, 100), (0, 100)), z_range=(0, 20))


def test_cli_exposes_z_crop_options():
    from click.testing import CliRunner
    from fishsuite.cli import cli
    out = CliRunner().invoke(cli, ['ortho', '--help']).output
    assert '--z-crop' in out and '--z-margin-um' in out
