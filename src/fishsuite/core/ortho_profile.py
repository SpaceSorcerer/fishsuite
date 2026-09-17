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


def render_ortho_figure(stack_czyx, center_zyx, half_width_px, *, nucleus_mask,
                        pixel_size_um, z_step_um, display_levels,
                        line_endpoints=None, qki_threshold=None, run_dir='',
                        profile_width_px=3, analysed_plane_z=None):
    """Render additive yellow/magenta sections and raw/normalised line profiles.

    Normalised traces use each trace's own min/max (labelled explicitly);
    image displays use ONLY the supplied common bounds. A 2-D saved DAPI
    segmentation outline is drawn on XY only, never extruded into fabricated 3-D.
    """
    import matplotlib.pyplot as plt
    from skimage.measure import find_contours
    levels = np.asarray(display_levels, dtype=float)
    if levels.shape != (2,2) or not np.isfinite(levels).all() or np.any(levels[:,1] <= levels[:,0]):
        raise ValueError('Provide finite increasing MIAT and QKI display levels')
    if not all(np.isfinite(v) and v > 0 for v in (pixel_size_um,z_step_um)):
        raise ValueError('Voxel sizes must be positive')
    if stack_czyx.shape[0] < 3 or nucleus_mask.shape != stack_czyx.shape[2:]:
        raise ValueError('Expected MIAT/QKI/DAPI stack and matching 2-D nucleus mask')
    (z,y,x), ((z0,z1),(y0,y1),(x0,x1)) = _bounds(stack_czyx,center_zyx,half_width_px)
    if analysed_plane_z is not None:
        if (not np.isfinite(analysed_plane_z) or int(analysed_plane_z) != analysed_plane_z
                or not 0 <= analysed_plane_z < stack_czyx.shape[1]):
            raise ValueError('Analysed plane must be a zero-based full-stack Z index')
    sections = ortho_sections(stack_czyx,center_zyx,half_width_px)
    endpoints = line_endpoints or clipped_default_line(nucleus_mask,(z,y,x),half_width_px)
    distance,values = line_profile(stack_czyx[:,z],*endpoints,width_px=profile_width_px,
                                   pixel_size_um=pixel_size_um)
    fig = plt.figure(figsize=(10,8))
    # Allocate physical panel sizes explicitly so adjacent sections share scale.
    nx, ny, nz = x1-x0, y1-y0, (z1-z0)*z_step_um/pixel_size_um
    gap = 8
    unit = min(8/(nx+nz+gap), 4.1/(ny+nz+gap))  # inches per XY pixel
    left, top = 1., 7.3
    xy = fig.add_axes([left/10,(top-ny*unit)/8,nx*unit/10,ny*unit/8],label='xy')
    xz = fig.add_axes([left/10,(top-(ny+gap+nz)*unit)/8,nx*unit/10,nz*unit/8],label='xz')
    yz = fig.add_axes([(left+(nx+gap)*unit)/10,(top-ny*unit)/8,nz*unit/10,ny*unit/8],label='yz')
    raw = fig.add_axes([.1,.15,.35,.19],label='raw')
    normal = fig.add_axes([.57,.15,.35,.19],label='normalised')
    def rgb(plane):
        a,b = [np.clip((plane[i]-lo)/(hi-lo),0,1) for i,(lo,hi) in enumerate(levels)]
        return np.stack([np.clip(a+b,0,1),a,b],axis=-1)
    for ax,key,cross,aspect in ((xy,'xy',(x-x0,y-y0),1),
                                (xz,'xz',(x-x0,z-z0),z_step_um/pixel_size_um),
                                (yz,'yz',(z-z0,y-y0),pixel_size_um/z_step_um)):
        plane = sections[key].transpose(0,2,1) if key == 'yz' else sections[key]
        ax.imshow(rgb(plane),aspect=aspect,interpolation='nearest')
        ax.axvline(cross[0],color='white',lw=.5,ls=':')
        ax.axhline(cross[1],color='white',lw=.5,ls=':')
        ax.set_title(key.upper())
        ax.set_xticks([])
        ax.set_yticks([])
    if analysed_plane_z is not None:
        label = f'analysed plane (z={int(analysed_plane_z)})'
        if z0 <= analysed_plane_z < z1:
            local_z = analysed_plane_z-z0
            xz.plot([.02,.1],[local_z]*2,color='#8ba6c4',lw=1.5,label='analysed plane',
                    transform=xz.get_yaxis_transform())
            xz.text(.12,local_z,label,color='#8ba6c4',fontsize=6,va='center',
                    transform=xz.get_yaxis_transform())
            yz.plot([local_z-.4,local_z+.4],[.97,.97],color='#8ba6c4',lw=1.5,
                    label='analysed plane',transform=yz.get_xaxis_transform())
            yz.text(local_z,.92,label,color='#8ba6c4',fontsize=6,ha='center',va='top',
                    transform=yz.get_xaxis_transform())
        else:
            for ax in (xz,yz):
                ax.text(.02,.98,label+' outside crop',transform=ax.transAxes,
                        color='#8ba6c4',fontsize=6,va='top')
    # Collections preserve contours as editable vector paths.
    from matplotlib.collections import LineCollection
    contours = find_contours(np.pad(nucleus_mask.astype(float),1),.5)
    xy.add_collection(LineCollection([np.column_stack((c[:,1]-1-x0,c[:,0]-1-y0)) for c in contours],
                                     colors='#8ba6c4',linewidths=.8))
    xy.plot([p[1]-x0 for p in endpoints],[p[0]-y0 for p in endpoints],color='white',lw=.8)
    xy.set_xlim(-.5,x1-x0-.5)
    xy.set_ylim(y1-y0-.5,-.5)
    bar_um = min(5., (x1-x0)*pixel_size_um/4)
    xy.plot([2,2+bar_um/pixel_size_um],[y1-y0-4]*2,color='white',lw=2)
    xy.text(2,y1-y0-6,f'{bar_um:g} µm',color='white',fontsize=8)
    annotation_count = 0
    for i,(name,color) in enumerate((('MIAT','#ffff00'),('QKI','#ff00ff'))):
        v = values[i]
        low,high = v.min(),v.max()
        span = high-low
        norm = lambda a: (a-low)/span if span > 0 else np.zeros_like(a)
        raw.plot(distance,v,color=color,label=name)
        normal.plot(distance,norm(v),color=color,label=name)
        bounds = [(f'{name} display {side}',bound,'--',.7)
                  for side,bound in zip(('min','max'),levels[i])]
        if i == 1 and qki_threshold is not None and np.isfinite(qki_threshold):
            bounds.append(('QKI analysis threshold',qki_threshold,'-',1))
        for label,bound,style,width in bounds:
            raw.axhline(bound,color=color,ls=style,lw=width,
                        label=label if style == '-' else None)
            value = float(norm(bound)) if span > 0 else float('nan')
            if np.isfinite(value) and 0 <= value <= 1:
                normal.axhline(value,color=color,ls=style,lw=width)
            else:
                normal.text(.02,.97-.1*annotation_count,
                            f'{label} {bound:g}: outside trace range',
                            transform=normal.transAxes,color=color,fontsize=5,va='top')
                annotation_count += 1
    for ax,title in ((raw,'Raw intensity'),(normal,'Per-trace min–max normalised')):
        ax.set_facecolor('#353535')
        ax.set_title(title,fontsize=10)
        ax.set_xlabel('Distance (µm)')
        ax.legend(fontsize=7)
    normal.set_ylim(-.05,1.05)
    normal.set_ylabel("normalised to each channel's own min–max along this line (display only)",
                      fontsize=6,wrap=True)
    fig.text(.02,.02,str(run_dir),fontsize=6,wrap=True)
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


