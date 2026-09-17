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


def render_ortho_figure(stack_czyx, center_zyx, half_width_px=None, *, nucleus_mask,
                        pixel_size_um, z_step_um, display_levels,
                        line_endpoints=None, qki_min=None, miat_min=None, run_dir='',
                        profile_width_px=3, analysed_plane_z=None, arm='', arm_color='#595959',
                        image='', nucleus_id=None, metric='', metric_value=None, arm_median=None):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea
    from pathlib import PureWindowsPath
    from skimage.measure import find_contours
    levels = np.asarray(display_levels, dtype=float)
    if levels.shape != (2,2) or not np.isfinite(levels).all() or np.any(levels[:,1] <= levels[:,0]):
        raise ValueError('Provide finite increasing MIAT and QKI display levels')
    if not all(np.isfinite(v) and v > 0 for v in (pixel_size_um,z_step_um)):
        raise ValueError('Voxel sizes must be positive')
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
    with plt.rc_context({'font.family':'Arial', 'font.size':8, 'svg.fonttype':'none'}):
        fig = plt.figure(figsize=(11,6.5))
    nx, ny, nz = x1-x0, y1-y0, (z1-z0)*z_step_um/pixel_size_um
    panel_width = 2.3
    grid = fig.add_gridspec(2,4,left=.045,right=(.045*11+panel_width*4.39)/11,
                           bottom=.14,top=.88,wspace=.13,hspace=.20,
                           height_ratios=(2.3,2.13))
    def image_axes(cell, width, height, label):
        position = cell.get_position(fig)
        ax = fig.add_subplot(cell,label=label)
        ax.set_position([position.x0,position.y1-height/6.5,width/11,height/6.5])
        return ax
    unit = panel_width/nx
    # XY width is independent of stack depth; orthogonal axes retain physical scale.
    xy = image_axes(grid[0,2],panel_width,ny*unit,'xy')
    miat = image_axes(grid[0,0],panel_width,ny*unit,'xy_miat')
    qki = image_axes(grid[0,1],panel_width,ny*unit,'xy_qki')
    yz = image_axes(grid[0,3],nz*unit,ny*unit,'yz')
    xz = image_axes(grid[1,2],panel_width,nz*unit,'xz')
    raw = fig.add_subplot(grid[1,:2],label='raw')
    normal = fig.add_subplot(grid[1,3],label='normalised')
    def rgb(plane, channel=None):
        a,b = [np.clip((plane[i]-lo)/(hi-lo),0,1) for i,(lo,hi) in enumerate(levels)]
        if channel == 0:
            b = np.zeros_like(b)
        elif channel == 1:
            a = np.zeros_like(a)
        return np.stack([np.clip(a+b,0,1),a,b],axis=-1)
    for ax,key,cross,aspect,title,channel in (
            (xy,'xy',(x-x0,y-y0),1,'XY merge',None),
            (miat,'xy',None,1,'XY MIAT',0), (qki,'xy',None,1,'XY QKI',1),
            (xz,'xz',(x-x0,z-z0),z_step_um/pixel_size_um,'XZ merge',None),
            (yz,'yz',(z-z0,y-y0),pixel_size_um/z_step_um,'YZ merge',None)):
        plane = sections[key].transpose(0,2,1) if key == 'yz' else sections[key]
        ax.imshow(rgb(plane,channel),aspect=aspect,interpolation='nearest')
        if cross is not None:
            ax.axvline(cross[0],color='white',lw=.5,ls=':')
            ax.axhline(cross[1],color='white',lw=.5,ls=':')
        ax.set_title(title,fontsize=9,pad=9)
        ax.set_xticks([])
        ax.set_yticks([])
    if analysed_plane_z is not None:
        label = f'z = {int(analysed_plane_z)+1}'
        if z0 <= analysed_plane_z < z1:
            local_z = analysed_plane_z-z0
            xz.plot([1.01,1.04],[local_z]*2,color='#526d88',lw=1.5,label='analysed plane',
                    transform=xz.get_yaxis_transform(),clip_on=False)
            xz.text(1.055,local_z,label,color='#526d88',fontsize=7,va='center',ha='left',rotation=90,
                    transform=xz.get_yaxis_transform(),clip_on=False)
            yz.plot([local_z]*2,[-.005,-.02],color='#526d88',lw=1.5,
                    label='analysed plane',transform=yz.get_xaxis_transform(),clip_on=False)
            yz.text(local_z,-.026,label,color='#526d88',fontsize=7,ha='center',va='top',
                    transform=yz.get_xaxis_transform(),clip_on=False)
        else:
            for ax in (xz,yz):
                ax.text(0,-.13,label+' outside crop',transform=ax.transAxes,
                        color='#526d88',fontsize=7,va='top')
    contours = find_contours(np.pad(nucleus_mask.astype(float),1),.5)
    xy.add_collection(LineCollection([np.column_stack((c[:,1]-1-x0,c[:,0]-1-y0)) for c in contours],
                                     colors='#8ba6c4',linewidths=.8))
    xy.plot([p[1]-x0 for p in endpoints],[p[0]-y0 for p in endpoints],color='white',lw=.8)
    # Reset bounds after markers so annotations cannot expand the image extent.
    for ax in (xy,miat,qki):
        ax.set_xlim(-.5,nx-.5)
        ax.set_ylim(ny-.5,-.5)
    xz.set_xlim(-.5,nx-.5); xz.set_ylim(z1-z0-.5,-.5)
    yz.set_xlim(-.5,z1-z0-.5); yz.set_ylim(ny-.5,-.5)
    bar_um = scale_bar_length(nx*pixel_size_um)
    if bar_um is not None:
        xy.plot([.06*nx,.06*nx+bar_um/pixel_size_um],[.91*ny]*2,color='white',lw=2,label='scale bar')
        xy.text(.06*nx,.85*ny,f'{bar_um:g} µm',color='white',fontsize=8)
    outside = []
    for i,(name,color,user_min) in enumerate((('MIAT','#ffff00',miat_min),('QKI','#ff00ff',qki_min))):
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
                                    bbox_to_anchor=(.04,.975),bbox_transform=fig.transFigure,borderpad=0))
    fig._ortho_header = f'{arm} | {details}'
    short_run = '/'.join(PureWindowsPath(str(run_dir)).parts[-2:])
    fig.text(.025,.038,short_run+' | single plane analysed; orthogonal views are raw stack sections, display levels identical across arms',fontsize=7)
    fig.text(.025,.016,"Profiles normalised to each channel's own min–max along this line (display only).",fontsize=7)
    fig._ortho_crop_bounds = (y0,y1,x0,x1)
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


