"""Pure selection, orthogonal sections and figures for MIAT/QKI stacks.

Coordinates are zero-based pixels; endpoints are (y, x). Channel order is
MIAT, QKI, DAPI. Display bounds are always explicit and shared between figures.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from skimage.measure import profile_line as _profile_line


def select_nuclei(nuclei_df, arm_col, metric_col, k, seed):
    """Nearest k to each arm median; seeded ties stable under row permutation."""
    if k < 1:
        raise ValueError('k must be positive')
    keys = [arm_col, 'image', 'nucleus_id']
    df = nuclei_df.copy()
    if df[keys].isna().any().any() or df.duplicated(keys).any():
        raise ValueError('Nucleus identities must be nonmissing and unique per arm/image')
    df[metric_col] = pd.to_numeric(df[metric_col], errors='raise')
    if not np.isfinite(df[metric_col]).all():
        raise ValueError('Selection metric must be finite for every nucleus')
    df = df.sort_values(keys).reset_index(drop=True)
    df['arm_median'] = df.groupby(arm_col)[metric_col].transform('median')
    df['_distance'] = (df[metric_col] - df.arm_median).abs()
    df['_tie'] = np.random.default_rng(seed).random(len(df))
    return (df.sort_values([arm_col, '_distance', '_tie']).groupby(arm_col, sort=True)
            .head(k).drop(columns=['_distance', '_tie']).reset_index(drop=True))


def pick_punctum(spots_df, nucleus_id, rule='brightest'):
    """Choose by peak intensity. A missing z is returned as NaN, never inferred."""
    if rule not in ('brightest', 'median'):
        raise ValueError('Unknown punctum rule')
    df = spots_df.loc[spots_df.nucleus_id == nucleus_id].copy()
    if 'in_nucleus' in df:
        df = df.loc[df.in_nucleus.astype(str).str.lower().isin(['1', '1.0', 'true'])]
    if df.empty:
        raise ValueError(f'No puncta for nucleus {nucleus_id}')
    intensity_col = next((c for c in ('peak_intensity', 'intensity') if c in df), None)
    if intensity_col is None:
        raise ValueError('Missing punctum peak_intensity')
    if not np.isfinite(df[[intensity_col, 'y_px', 'x_px']].to_numpy(dtype=float)).all():
        raise ValueError('Punctum intensity and XY must be finite')
    zcol = 'z_px' if 'z_px' in df else None
    df['_rank'] = (-df[intensity_col] if rule == 'brightest' else
                   (df[intensity_col] - df[intensity_col].median()).abs())
    row = df.sort_values(['_rank', 'y_px', 'x_px'] + ([zcol] if zcol else [])).iloc[0]
    return (float(row[zcol]) if zcol else float('nan'), float(row.y_px), float(row.x_px))


def _bounds(stack, center, half):
    if np.ndim(stack) != 4 or half < 1 or int(half) != half:
        raise ValueError('Expected CZYX stack and positive integer half width')
    center = np.asarray(center, dtype=float)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError('Center must be finite ZYX')
    z, y, x = np.rint(center).astype(int)
    if any(c < 0 or c >= n for c, n in zip((z,y,x), stack.shape[1:])):
        raise ValueError('Center outside stack')
    bounds = [(max(0,c-half), min(n,c+half+1)) for c,n in zip((z,y,x),stack.shape[1:])]
    return (z,y,x), bounds


def ortho_sections(stack_czyx, center_zyx, half_width_px):
    """Return C-Y-X, C-Z-X and C-Z-Y sections, clipped at image boundaries."""
    (z,y,x), ((z0,z1),(y0,y1),(x0,x1)) = _bounds(stack_czyx,center_zyx,half_width_px)
    return dict(xy=stack_czyx[:,z,y0:y1,x0:x1],
                xz=stack_czyx[:,z0:z1,y,x0:x1],
                yz=stack_czyx[:,z0:z1,y0:y1,x])


def line_profile(plane_cyx, p0, p1, width_px=3, *, pixel_size_um):
    """Mean across a perpendicular strip; return distance_um and C-by-N values."""
    if np.ndim(plane_cyx) != 3 or width_px < 1 or int(width_px) != width_px:
        raise ValueError('Expected CYX plane and positive integer profile width')
    if not np.isfinite(pixel_size_um) or pixel_size_um <= 0:
        raise ValueError('Pixel size must be positive')
    for p in (p0,p1):
        if len(p) != 2 or not np.isfinite(p).all() or any(v < 0 or v > n-1 for v,n in zip(p,plane_cyx.shape[1:])):
            raise ValueError('Profile endpoint outside plane')
    values = np.stack([_profile_line(p.astype(float),p0,p1,linewidth=width_px,
                                     order=1,mode='constant',reduce_func=np.mean)
                       for p in plane_cyx])
    distance = np.linspace(0, math.dist(p0,p1)*pixel_size_um,values.shape[1])
    return distance, values


def default_line(nucleus_mask, center_zyx):
    """Horizontal line at the punctum y spanning the saved nucleus bounding box."""
    ys,xs = np.nonzero(nucleus_mask)
    if not len(xs):
        raise ValueError('Nucleus mask is empty')
    return (float(center_zyx[1]),float(xs.min())), (float(center_zyx[1]),float(xs.max()))


def clipped_default_line(nucleus_mask, center_zyx, half_width_px):
    """Intersect the nucleus bounding-box chord with the displayed XY crop."""
    if np.ndim(nucleus_mask) != 2 or half_width_px < 1 or int(half_width_px) != half_width_px:
        raise ValueError('Expected 2-D nucleus mask and positive integer half width')
    center = np.asarray(center_zyx, dtype=float)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError('Center must be finite ZYX')
    _, y, x = np.rint(center).astype(int)
    if not (0 <= y < nucleus_mask.shape[0] and 0 <= x < nucleus_mask.shape[1]):
        raise ValueError('Center outside nucleus mask image')
    p0, p1 = default_line(nucleus_mask, center)
    x0 = max(p0[1], 0, x-half_width_px)
    x1 = min(p1[1], nucleus_mask.shape[1]-1, x+half_width_px)
    if x0 > x1:
        raise ValueError('Nucleus bounding box does not intersect displayed crop')
    return (float(y),float(x0)), (float(y),float(x1))


def nucleus_crop(nucleus_mask, pixel_size_um, half_width_um=None):
    ys, xs = np.nonzero(nucleus_mask)
    if not len(xs):
        raise ValueError('Nucleus mask is empty')
    if not np.isfinite(pixel_size_um) or pixel_size_um <= 0:
        raise ValueError('Pixel size must be positive')
    center_yx = tuple(np.rint([(ys.min()+ys.max())/2, (xs.min()+xs.max())/2]).astype(int))
    requested = (max(ys.max()-ys.min()+1, xs.max()-xs.min()+1)*pixel_size_um/2 + 1.5
                 if half_width_um is None else float(half_width_um))
    if not np.isfinite(requested) or requested <= 0:
        raise ValueError('Half width must be finite and positive')
    half_px = max(1, int(np.ceil(requested/pixel_size_um)))
    return center_yx, half_px, requested


def scale_bar_length(panel_width_um):
    candidates = [v for v in (1, 2, 5, 10, 20) if v <= panel_width_um*.40]
    return max(candidates) if candidates else None


# Layout constants, inches. Every image panel shares one inches-per-µm scale.
_XY_WIDTH_IN = 2.3
_MAX_ORTHO_BLOCK_IN = 6.0
_PROFILE_HEIGHT_IN = 1.7
_BAR_MARGIN_IN = .08


def _text_width_in(text, fontsize):
    return len(text)*fontsize*.56/72


def _draw_scale_bar(ax, *, orientation, bar_um, um_per_data, extent_data, panel_in,
                    corner, name, outside='right', fontsize=7):
    """Draw a bar of ``bar_um`` in data units (length = µm / µm-per-pixel).

    The label sits inside the panel in white when it takes at most half the
    panel's width/height; otherwise it goes just outside the panel
    (``outside='right'`` or ``'below'``) in black so it stays legible.
    """
    nxd, nyd = extent_data
    w_in, h_in = panel_in
    xpi, ypi = nxd/w_in, nyd/h_in
    m = _BAR_MARGIN_IN
    length = bar_um/um_per_data
    left, right, top, bottom = -.5, nxd-.5, -.5, nyd-.5
    text = f'{bar_um:g} µm' + (' (z)' if 'axial' in name else '')
    label_w, label_h = _text_width_in(text, fontsize), fontsize*1.25/72
    if orientation == 'h':
        x0 = left + m*xpi if corner.endswith('left') else right - m*xpi - length
        y = bottom - m*ypi if corner.startswith('bottom') else top + m*ypi
        xs, ys = [x0, x0+length], [y, y]
        inside = m + label_w <= w_in - .02 and m + label_h + .03 <= .45*h_in
        anchor, offset, ha, va = (x0, y), (0, 1.5), 'left', 'bottom'
    else:
        x0 = right - m*xpi
        y0 = bottom - m*ypi if corner.startswith('bottom') else top + m*ypi
        y1 = y0 - length if corner.startswith('bottom') else y0 + length
        xs, ys = [x0, x0], [y0, y1]
        inside = m + label_w + 3/72 <= .5*w_in and h_in >= .6
        anchor, offset, ha, va = (x0, (y0+y1)/2), (-3, 0), 'right', 'center'
    (line,) = ax.plot(xs, ys, color='white', lw=2, solid_capstyle='butt', label=name)
    if inside:
        label = ax.annotate(text, anchor, xytext=offset, textcoords='offset points',
                            color='white', fontsize=fontsize, ha=ha, va=va)
    elif outside == 'below':
        label = ax.annotate(text, (anchor[0], 0), xycoords=('data', 'axes fraction'),
                            xytext=(0, -2), textcoords='offset points', color='black',
                            fontsize=fontsize, ha='left' if orientation == 'h' else 'right',
                            va='top', annotation_clip=False)
    else:
        label = ax.annotate(text, (1, anchor[1]), xycoords=('axes fraction', 'data'),
                            xytext=(3, 0), textcoords='offset points', color='black',
                            fontsize=fontsize, ha='left', va='center', annotation_clip=False)
    line._scale_bar_um = bar_um
    line._scale_bar_text = label
    return line


def render_ortho_figure(stack_czyx, center_zyx, half_width_px=None, *, nucleus_mask,
                        pixel_size_um, z_step_um, display_levels,
                        line_endpoints=None, qki_min=None, miat_min=None, run_dir='',
                        profile_width_px=3, analysed_plane_z=None, arm='', arm_color='#595959',
                        image='', nucleus_id=None, metric='', metric_value=None, arm_median=None,
                        include_dapi=False, full_merge=None, dapi_display_level=None,
                        channel_labels=('MIAT', 'QKI'), dapi_label='DAPI',
                        profile_labels=None,
                        show_scale_bars=False, show_z_slice_labels=False,
                        show_cross_section=True, axial_scale_factor=1.0):
    """Render calibrated XY/XZ/YZ sections and the linked intensity profiles.

    Geometry: every image panel is drawn at one inches-per-µm scale, lateral
    and axial. XZ sits below XY and shares its x axis; YZ sits right of XY and
    shares its y axis. The figure size is computed from the physical extents
    (crop width/height and z window × z step × ``axial_scale_factor``), so the
    orthogonal views keep their true aspect at any stack depth and never clip.
    ``axial_scale_factor`` multiplies the nominal z step (refractive-index
    correction; 1.0 = nominal) and is stated in the footer.

    XY carries a lateral µm bar; XZ and YZ each carry a lateral and an axial
    ("(z)") bar whose pixel length is µm / (z step × axial factor).
    ``show_scale_bars`` adds lateral bars to the single-channel panels.
    ``include_dapi=True`` (or ``full_merge=True``) adds a DAPI panel and DAPI in
    the merges. Channel names come from ``channel_labels`` and are used in the
    titles, profile legend and footer. LUT: channel 1 yellow, channel 2
    magenta, DAPI blue; display levels are shared by every panel.
    """
    import matplotlib.pyplot as plt
    import textwrap
    from matplotlib.collections import LineCollection
    from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea
    from pathlib import PureWindowsPath
    from skimage.measure import find_contours
    if full_merge is not None:
        include_dapi = bool(full_merge)
    levels = np.asarray(display_levels, dtype=float)
    if levels.shape != (2,2) or not np.isfinite(levels).all() or np.any(levels[:,1] <= levels[:,0]):
        raise ValueError('Provide finite increasing MIAT and QKI display levels')
    if len(channel_labels) != 2 or any(not str(label).strip() for label in channel_labels):
        raise ValueError('channel_labels must contain two nonempty labels')
    if profile_labels is not None and (len(profile_labels) != 2 or
                                       any(not str(label).strip() for label in profile_labels)):
        raise ValueError('profile_labels must contain two nonempty labels')
    if include_dapi:
        if dapi_display_level is None:
            raise ValueError('dapi_display_level is required when include_dapi/full_merge is enabled')
        dapi_level = np.asarray(dapi_display_level, dtype=float)
        if dapi_level.shape != (2,) or not np.isfinite(dapi_level).all() or dapi_level[1] <= dapi_level[0]:
            raise ValueError('Provide a finite increasing dapi_display_level')
        if not str(dapi_label).strip():
            raise ValueError('dapi_label must be nonempty')
    else:
        dapi_level = None
    if not all(np.isfinite(v) and v > 0 for v in (pixel_size_um,z_step_um)):
        raise ValueError('Voxel sizes must be positive')
    try:
        factor = float(axial_scale_factor)
    except (TypeError, ValueError):
        factor = float('nan')
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError('axial_scale_factor must be finite and positive')
    if stack_czyx.shape[0] < 3 or nucleus_mask.shape != stack_czyx.shape[2:]:
        raise ValueError('Expected MIAT/QKI/DAPI stack and matching 2-D nucleus mask')
    for minimum in (qki_min, miat_min):
        if minimum is not None and not np.isfinite(minimum):
            raise ValueError('User analysis minima must be finite')
    crop_yx, auto_half, _ = nucleus_crop(nucleus_mask, pixel_size_um)
    half = auto_half if half_width_px is None else half_width_px
    (z,y,x), _ = _bounds(stack_czyx,center_zyx,half)
    _, ((z0,z1),(y0,y1),(x0,x1)) = _bounds(stack_czyx,(z,*crop_yx),half)
    if analysed_plane_z is not None:
        if (not np.isfinite(analysed_plane_z) or int(analysed_plane_z) != analysed_plane_z
                or not 0 <= analysed_plane_z < stack_czyx.shape[1]):
            raise ValueError('Analysed plane must be a zero-based full-stack Z index')
    sections = dict(xy=stack_czyx[:,z,y0:y1,x0:x1],
                    xz=stack_czyx[:,z0:z1,y,x0:x1], yz=stack_czyx[:,z0:z1,y0:y1,x])
    if line_endpoints is None:
        p0,p1 = default_line(nucleus_mask,(z,y,x))
        endpoints = ((y,max(x0,p0[1])), (y,min(x1-1,p1[1])))
    else:
        endpoints = line_endpoints
    distance,values = line_profile(stack_czyx[:,z],*endpoints,width_px=profile_width_px,
                                   pixel_size_um=pixel_size_um)
    label0, label1 = (str(channel_labels[0]), str(channel_labels[1]))
    profile_label0, profile_label1 = ((str(profile_labels[0]), str(profile_labels[1]))
                                      if profile_labels is not None else (label0, label1))

    # Physical geometry: one scale (inches per µm) for every image panel.
    dz_eff = z_step_um*factor
    nx, ny, nzs = x1-x0, y1-y0, z1-z0
    crop_w_um, crop_h_um, z_um = nx*pixel_size_um, ny*pixel_size_um, nzs*dz_eff
    scale = min(_XY_WIDTH_IN/crop_w_um, _MAX_ORTHO_BLOCK_IN/(crop_w_um+z_um),
                _MAX_ORTHO_BLOCK_IN/(crop_h_um+z_um))
    w_in, h_in, z_in = crop_w_um*scale, crop_h_um*scale, z_um*scale
    n_single = 3 if include_dapi else 2
    margin_l, margin_r, margin_t = .65, .7, .8
    gap_s, gap_o, gap_v = .3, .45, .55
    norm_w = min(max(z_in, 1.9), 2.6)
    fig_width = (margin_l + (n_single+1)*w_in + n_single*gap_s + gap_o +
                 max(z_in, norm_w) + margin_r)

    short_run = '/'.join(PureWindowsPath(str(run_dir)).parts[-2:])
    level_text = [f'{label0} {levels[0,0]:g}–{levels[0,1]:g}',
                  f'{label1} {levels[1,0]:g}–{levels[1,1]:g}']
    merge_text = [f'{label0} yellow', f'{label1} magenta']
    if include_dapi:
        level_text.append(f'{dapi_label} {dapi_level[0]:g}–{dapi_level[1]:g}')
        merge_text.append(f'{dapi_label} blue')
    footer = [
        (f'{short_run} | ' if short_run else '') +
        'single plane analysed; orthogonal views are raw stack sections; display levels '
        'fixed and identical across all panels and arms.',
        f'Voxel {pixel_size_um:g} µm (XY) × {z_step_um:g} µm (Z step); axial scale factor '
        f'{factor:g} (1 = nominal z, no refractive-index correction) → {dz_eff:.4g} µm per '
        f'displayed Z slice. Sections (1-based, dotted crosshairs): XY at z {z+1}, XZ at y {y+1}, '
        f'YZ at x {x+1}. Z shown: z {z0+1}–{z1} of {stack_czyx.shape[1]} ({z_um:.2f} µm).',
        'Display min–max: ' + '; '.join(level_text) + ' | Merge: ' + ', '.join(merge_text) +
        ' | Pixels drawn nearest-neighbour (no interpolation); all image panels at '
        f'{scale:.3f} in per µm, lateral and axial.',
        "Profiles normalised to each channel's own min–max along this line (display only).",
    ]
    wrap = max(40, int(fig_width*72/(7*.56)) - 4)
    footer_lines = [part for line in footer for part in
                    textwrap.wrap(line, wrap, break_on_hyphens=False, break_long_words=False)]
    footer_h = .12 + .15*len(footer_lines)
    row2_h = max(z_in + .3, _PROFILE_HEIGHT_IN + .45)
    fig_height = margin_t + h_in + gap_v + row2_h + footer_h

    with plt.rc_context({'font.family':'Arial', 'font.size':8, 'svg.fonttype':'none'}):
        fig = plt.figure(figsize=(fig_width, fig_height))

    def place(label, left_in, top_in, width_in, height_in):
        return fig.add_axes([left_in/fig_width, (fig_height-top_in-height_in)/fig_height,
                             width_in/fig_width, height_in/fig_height], label=label)

    single_x = [margin_l + i*(w_in+gap_s) for i in range(n_single)]
    xy_x = margin_l + n_single*(w_in+gap_s)
    yz_x = xy_x + w_in + gap_o
    row2_top = margin_t + h_in + gap_v
    raw_w = n_single*w_in + (n_single-1)*gap_s
    if include_dapi:
        dapi = place('xy_dapi', single_x[0], margin_t, w_in, h_in)
        miat = place('xy_miat', single_x[1], margin_t, w_in, h_in)
        qki = place('xy_qki', single_x[2], margin_t, w_in, h_in)
        xy = place('xy', xy_x, margin_t, w_in, h_in)
    else:
        xy = place('xy', xy_x, margin_t, w_in, h_in)
        miat = place('xy_miat', single_x[0], margin_t, w_in, h_in)
        qki = place('xy_qki', single_x[1], margin_t, w_in, h_in)
    yz = place('yz', yz_x, margin_t, z_in, h_in)
    xz = place('xz', xy_x, row2_top, w_in, z_in)
    raw = place('raw', margin_l, row2_top, raw_w, _PROFILE_HEIGHT_IN)
    normal = place('normalised', yz_x, row2_top, norm_w, _PROFILE_HEIGHT_IN)

    def rgb(plane, channel=None):
        a,b = [np.clip((plane[i]-lo)/(hi-lo),0,1) for i,(lo,hi) in enumerate(levels)]
        if include_dapi:
            d = np.clip((plane[2]-dapi_level[0])/(dapi_level[1]-dapi_level[0]),0,1)
        else:
            d = np.zeros_like(a)
        if channel == 0:
            b = np.zeros_like(b)
            d = np.zeros_like(d)
        elif channel == 1:
            a = np.zeros_like(a)
            d = np.zeros_like(d)
        if channel == 2:
            return np.stack([np.zeros_like(d),np.zeros_like(d),d],axis=-1)
        return np.stack([np.clip(a+b,0,1),a,np.clip(b+d,0,1)],axis=-1)
    merge_names = ([dapi_label] if include_dapi else []) + [label0, label1]
    axial_aspect = dz_eff/pixel_size_um
    panels = [
        (xy, 'xy', (x-x0,y-y0), 1, f'XY merge\n({" + ".join(merge_names)})', None),
        (miat, 'xy', None, 1, f'XY {label0}', 0),
        (qki, 'xy', None, 1, f'XY {label1}', 1),
        (xz, 'xz', (x-x0,z-z0), axial_aspect, 'XZ merge', None),
        (yz, 'yz', (z-z0,y-y0), 1/axial_aspect, 'YZ merge', None),
    ]
    if include_dapi:
        panels.insert(1, (dapi, 'xy', None, 1, f'XY {dapi_label}', 2))
    for ax,key,cross,aspect,title,channel in panels:
        plane = sections[key].transpose(0,2,1) if key == 'yz' else sections[key]
        ax.imshow(rgb(plane,channel),aspect=aspect,interpolation='nearest')
        if cross is not None:
            ax.axvline(cross[0],color='white',lw=.5,ls=':',label='section position')
            ax.axhline(cross[1],color='white',lw=.5,ls=':',label='section position')
        ax.set_title(title, fontsize=8, pad=9)
        ax.set_xticks([])
        ax.set_yticks([])
    if analysed_plane_z is not None:
        label = f'z = {int(analysed_plane_z)+1}'
        if z0 <= analysed_plane_z < z1:
            local_z = analysed_plane_z-z0
            xz.plot([1.01,1.04],[local_z]*2,color='#526d88',lw=1.5,label='analysed plane',
                    transform=xz.get_yaxis_transform(),clip_on=False)
            yz.plot([local_z]*2,[-.005,-.02],color='#526d88',lw=1.5,
                    label='analysed plane',transform=yz.get_xaxis_transform(),clip_on=False)
            if not show_z_slice_labels:
                xz.text(1.055,local_z,label,color='#526d88',fontsize=7,va='center',ha='left',rotation=90,
                        transform=xz.get_yaxis_transform(),clip_on=False)
                yz.text(local_z,-.026,label,color='#526d88',fontsize=7,ha='center',va='top',
                        transform=yz.get_xaxis_transform(),clip_on=False)
        else:
            if not show_z_slice_labels:
                for ax in (xz,yz):
                    ax.text(0,-.13,label+' outside crop',transform=ax.transAxes,
                            color='#526d88',fontsize=7,va='top')
        if show_z_slice_labels:
            xy.text(.96, .055, label, transform=xy.transAxes, color='#8ba6c4',
                    fontsize=7, ha='right', va='bottom', clip_on=True)
    elif show_z_slice_labels:
        label = f'z = {z+1} (displayed)'
        xy.text(.96, .055, label, transform=xy.transAxes, color='#8ba6c4',
                fontsize=7, ha='right', va='bottom', clip_on=True)
    contours = find_contours(np.pad(nucleus_mask.astype(float),1),.5)
    xy.add_collection(LineCollection([np.column_stack((c[:,1]-1-x0,c[:,0]-1-y0)) for c in contours],
                                     colors='#8ba6c4',linewidths=.8))
    if show_cross_section:
        xy.plot([p[1]-x0 for p in endpoints],[p[0]-y0 for p in endpoints],color='white',lw=1.0,
                label='measured profile line')
        xy.scatter([p[1]-x0 for p in endpoints],[p[0]-y0 for p in endpoints],s=4,color='white',zorder=5)
        # The measured XY segment projected into XZ, drawn at the displayed XY
        # plane (z). In YZ the x-varying segment collapses to the crosshair.
        xz.plot([p[1]-x0 for p in endpoints], [z-z0]*2, color='white', lw=1.0,
                label='measured profile line')
    # Reset bounds after markers so annotations cannot expand the image extent.
    for ax in ((dapi,xy,miat,qki) if include_dapi else (xy,miat,qki)):
        ax.set_xlim(-.5,nx-.5)
        ax.set_ylim(ny-.5,-.5)
    xz.set_xlim(-.5,nx-.5); xz.set_ylim(nzs-.5,-.5)
    yz.set_xlim(-.5,nzs-.5); yz.set_ylim(ny-.5,-.5)

    lateral_bar = scale_bar_length(crop_w_um)
    lateral_bar_y = scale_bar_length(crop_h_um)
    axial_bar = scale_bar_length(z_um)
    singles = [dapi, miat, qki] if include_dapi else [miat, qki]
    if lateral_bar is not None:
        for ax in [xy] + (singles if show_scale_bars else []):
            _draw_scale_bar(ax, orientation='h', bar_um=lateral_bar, um_per_data=pixel_size_um,
                            extent_data=(nx, ny), panel_in=(w_in, h_in), corner='bottom-left',
                            name='scale bar')
        _draw_scale_bar(xz, orientation='h', bar_um=lateral_bar, um_per_data=pixel_size_um,
                        extent_data=(nx, nzs), panel_in=(w_in, z_in), corner='bottom-left',
                        name='scale bar lateral', outside='below')
    if axial_bar is not None:
        _draw_scale_bar(xz, orientation='v', bar_um=axial_bar, um_per_data=dz_eff,
                        extent_data=(nx, nzs), panel_in=(w_in, z_in), corner='bottom-right',
                        name='scale bar axial', outside='below')
        _draw_scale_bar(yz, orientation='h', bar_um=axial_bar, um_per_data=dz_eff,
                        extent_data=(nzs, ny), panel_in=(z_in, h_in), corner='bottom-left',
                        name='scale bar axial')
    if lateral_bar_y is not None:
        _draw_scale_bar(yz, orientation='v', bar_um=lateral_bar_y, um_per_data=pixel_size_um,
                        extent_data=(nzs, ny), panel_in=(z_in, h_in), corner='top-right',
                        name='scale bar lateral')

    outside = []
    for i,(name,color,user_min) in enumerate(((profile_label0,'#ffff00',miat_min),
                                               (profile_label1,'#ff00ff',qki_min))):
        v = values[i]
        low,high = v.min(),v.max()
        span = high-low
        norm = lambda a: (a-low)/span if span > 0 else np.zeros_like(a)
        raw.plot(distance,v,color=color,label=name)
        normal.plot(distance,norm(v),color=color,label=name)
        bounds = [(f'{name} display {side}',bound,'--',.7)
                  for side,bound in zip(('min','max'),levels[i])]
        if user_min is not None:
            bounds.append((f'{name} analysis min (user)',user_min,'-',1))
        for label,bound,style,width in bounds:
            raw.axhline(bound,color=color,ls=style,lw=width,label=label)
            value = float(norm(bound)) if span > 0 else float('nan')
            if np.isfinite(value) and 0 <= value <= 1:
                normal.axhline(value,color=color,ls=style,lw=width)
            else:
                outside.append(f'{label} {bound:g}: outside trace range')
    for ax,title in ((raw,'Raw intensity'),(normal,'Min–max profile')):
        ax.set_facecolor('#353535')
        ax.set_title(title,fontsize=9,pad=9)
        ax.set_xlabel('Distance (µm)',fontsize=8)
        ax.tick_params(labelsize=7)
    raw.set_ylabel('Intensity (a.u.)',fontsize=8)
    legend_columns = len(raw.get_legend_handles_labels()[0])
    raw.legend(fontsize=min(6,36/legend_columns),ncol=legend_columns,loc='upper right',
               framealpha=.6,handlelength=1.,handletextpad=.3,columnspacing=.5,borderpad=.3)
    normal.set_ylim(-.05,1.05)
    normal.yaxis.tick_right()
    normal.yaxis.set_label_position('right')
    normal.tick_params(axis='y',pad=2)
    normal.set_ylabel('normalised (display only)',fontsize=8,labelpad=2)
    normal._outside_trace_bounds = outside
    plane_text = (f'analysed plane z = {int(analysed_plane_z)+1} (1-based)'
                  if analysed_plane_z is not None else f'analysed plane unavailable; displayed z = {z+1} (1-based)')
    comparison = (f'{metric} = {metric_value:g} vs arm median {arm_median:g}'
                  if metric_value is not None and arm_median is not None else '')
    details = ' | '.join(str(v) for v in (image, f'nucleus {nucleus_id}',plane_text,comparison) if v)
    header = HPacker(children=[TextArea(str(arm),textprops=dict(color=arm_color,size=8,weight='bold')),
                              TextArea(' | '+details,textprops=dict(size=8))],align='center',pad=0,sep=0)
    fig.add_artist(AnchoredOffsetbox(loc='upper left',child=header,pad=0,frameon=False,
                                    bbox_to_anchor=(.25/fig_width,1-.12/fig_height),
                                    bbox_transform=fig.transFigure,borderpad=0))
    fig._ortho_header = f'{arm} | {details}'
    for i, line in enumerate(footer_lines):
        fig.text(.25/fig_width, (.08 + .15*(len(footer_lines)-1-i))/fig_height, line,
                 fontsize=7, va='bottom')
    fig._ortho_crop_bounds = (y0,y1,x0,x1)
    display = {label0: tuple(float(v) for v in levels[0]), label1: tuple(float(v) for v in levels[1])}
    if include_dapi:
        display[str(dapi_label)] = tuple(float(v) for v in dapi_level)
    fig._ortho_geometry = dict(
        inches_per_um=scale, figure_size_in=(fig_width, fig_height),
        pixel_size_um=float(pixel_size_um), z_step_um=float(z_step_um),
        axial_scale_factor=factor, z_step_effective_um=dz_eff,
        crop_width_um=crop_w_um, crop_height_um=crop_h_um, z_extent_um=z_um,
        z_range_0based=(int(z0), int(z1)), n_z_total=int(stack_czyx.shape[1]),
        section_zyx_0based=(int(z), int(y), int(x)), interpolation='nearest',
        channel_labels=(label0, label1), display_levels=display,
        scale_bars_um=dict(lateral=lateral_bar, lateral_y=lateral_bar_y, axial=axial_bar),
        footer=' '.join(footer))
    from matplotlib.text import Text
    for artist in fig.findobj(Text):
        artist.set_fontfamily('Arial')
    return fig


def chord_through(mask, cy, cx, orientation):
    """Endpoints of the chord through (cy, cx) along the region principal axis.

    skimage's ``orientation`` is the angle between the row axis and the major
    axis, so the unit direction is (dy, dx) = (-sin theta, cos theta). Walk
    outward in both directions in 0.5-px steps and keep the last in-mask point.
    """
    H, W = mask.shape
    ddy, ddx = -math.sin(orientation), math.cos(orientation)
    if not mask[int(round(cy)), int(round(cx))]:
        ys, xs = np.nonzero(mask)
        k = int(np.argmin((ys - cy) ** 2 + (xs - cx) ** 2))
        cy, cx = float(ys[k]), float(xs[k])
    ends = []
    for sgn in (+1.0, -1.0):
        last = (cy, cx)
        t = 0.0
        while True:
            t += 0.5
            y = cy + sgn * ddy * t
            x = cx + sgn * ddx * t
            iy, ix = int(round(y)), int(round(x))
            if iy < 0 or ix < 0 or iy >= H or ix >= W or not mask[iy, ix]:
                break
            last = (y, x)
        ends.append(last)
    return ends[0], ends[1]


def sample_profile(plane, src, dst):
    from skimage.measure import profile_line
    return profile_line(plane.astype(np.float64), src, dst, linewidth=1, order=1,
                        mode="constant", reduce_func=None).ravel()
