"""Linked orthogonal set B1-B4 for one nucleus: FOV ortho with nucleus box + line (B1), fishsuite nucleus ortho with
the same crosshair and line (B2), nucleus zoom with the line and on-line puncta (B3), MIAT/QKI/DAPI line profile (B4).
All four read P0/P1 and the marked spot from one variable. Ported from fig1lib.linked_set (2026-09-28).

Line rule (objective): among straight lines through the marked punctum (brightest footprint-mean MIAT in the
nucleus), 0-179 deg in 1 deg steps, clipped to the nucleus mask, pick the one whose perpendicular band <= R_PX
captures the most puncta centres; ties -> smallest summed perpendicular distance, then smallest angle.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from .style import AXIAL, COL, LINE_DAPI, LINE_MIAT, LINE_QKI, LV, colorize, footer


def nucleus_box(mask):
    ys, xs = np.nonzero(mask)
    return (xs.min() - 8, ys.min() - 8, xs.max() - xs.min() + 16, ys.max() - ys.min() + 16)


def marked_spot(ns: pd.DataFrame):
    """Brightest punctum of the nucleus by footprint-mean MIAT (the menu rep_punctum rule)."""
    mark = ns.sort_values("miat_footprint_mean_raw", ascending=False).iloc[0]
    return mark, float(mark.center_y_px), float(mark.center_x_px)


def select_line(mask, ns: pd.DataFrame, my: float, mx: float, R_PX: float = 2.0):
    """Objective line through (my, mx). Returns dict(angle_deg, d, t0, t1, P0, P1, hit, tt, perp, n_hit)."""
    def chord(theta):
        d = np.array([np.sin(theta), np.cos(theta)])
        ends = []
        for sgn in (-1, 1):
            t = 0.0
            while True:
                y, x = my + sgn * (t + .5) * d[0], mx + sgn * (t + .5) * d[1]
                iy, ix = int(round(y)), int(round(x))
                if not (0 <= iy < mask.shape[0] and 0 <= ix < mask.shape[1]) or not mask[iy, ix]:
                    break
                t += .5
            ends.append(sgn * t)
        return d, ends[0], ends[1]

    best = None
    for deg in range(180):
        th = np.deg2rad(deg)
        d, t0, t1 = chord(th)
        rel = np.c_[ns.center_y_px - my, ns.center_x_px - mx]
        tt = rel @ d
        perp = np.abs(rel @ np.array([-d[1], d[0]]))
        hit = (perp <= R_PX) & (tt >= t0) & (tt <= t1)
        score = (int(hit.sum()), -float(perp[hit].sum()), -deg)
        if best is None or score > best[0]:
            best = (score, deg, d, t0, t1, hit.copy(), tt.copy(), perp.copy())
    (score, deg, d, t0, t1, hit, tt, perp) = best
    P0 = (my + t0 * d[0], mx + t0 * d[1])
    P1 = (my + t1 * d[0], mx + t1 * d[1])
    return dict(angle_deg=deg, d=d, t0=t0, t1=t1, P0=P0, P1=P1, hit=hit, tt=tt, perp=perp, n_hit=score[0])


def line_rule_text(R_PX, px):
    return (f"Among straight lines through the marked punctum (angles 0-179 deg, 1 deg steps) clipped to the nucleus mask, "
            f"the one whose perpendicular distance <= {R_PX} px ({R_PX * px:.2f} µm) captures the most detected MIAT puncta "
            "centres; ties -> smallest summed perpendicular distance, then smallest angle.")


def linked_geometry(run, field_id, nucleus_id, R_PX=2.0, spots=None):
    """Everything B1-B4 share, computed from the h5 analysed plane + mask and the per-punctum table (no raw stack)."""
    sel = run.selection()
    px = sel["pixel_size_um"][0]
    key = run.image_key(field_id)
    P = run.load_plane(key)
    mask = P["nucleus_labels"] == int(nucleus_id)
    spots = run.spots_all() if spots is None else spots
    ns = spots[(spots.image == key) & (spots.nucleus_id == int(nucleus_id))].reset_index(drop=True)
    mark, my, mx = marked_spot(ns)
    L = select_line(mask, ns, my, mx, R_PX)
    on = ns[L["hit"]].assign(pos_um=(L["tt"][L["hit"]] - L["t0"]) * px, perp_px=L["perp"][L["hit"]]).sort_values("pos_um")
    return dict(key=key, plane=P, mask=mask, ns=ns, mark=mark, my=my, mx=mx, line=L, on=on, px=px,
                dz=sel["z_step_um_nominal"][0], box=nucleus_box(mask))


def params_dict(G, field_id, nucleus_id, label, R_PX, z_window, stack_chk):
    L, on = G["line"], G["on"]
    return dict(label=label,
                nucleus=dict(field=str(field_id), image=G["key"], nucleus_id=int(nucleus_id), box_xywh_px=[int(v) for v in G["box"]]),
                marked_spot=dict(spot_id=int(G["mark"].spot_id), y=G["my"], x=G["mx"], z0=G["plane"]["z0"],
                                 crosshair_yx=[int(round(G["my"])), int(round(G["mx"]))]),
                line=dict(rule=line_rule_text(R_PX, G["px"]), angle_deg=L["angle_deg"], p0_yx=list(L["P0"]), p1_yx=list(L["P1"]),
                          length_um=float((L["t1"] - L["t0"]) * G["px"]), n_puncta_on_line=int(len(on)), n_puncta_in_nucleus=int(len(G["ns"]))),
                puncta_on_line=on[["spot_id", "center_y_px", "center_x_px", "pos_um", "perp_px", "miat_footprint_mean_raw",
                                   "footprint_mean_qki"]].to_dict(orient="records"),
                z_window=[int(z_window[0]), int(z_window[1])] if z_window is not None else None,
                stack_matches_h5=stack_chk, pixel_um=G["px"], z_step_um=G["dz"], axial=AXIAL)


def _scalebar(ax, um, shape, px):
    h, w = shape; L = um / px; x1 = w * .96; y = h * .95
    ax.plot([x1 - L, x1], [y, y], color='white', lw=2, solid_capstyle='butt')
    ax.text(x1 - L / 2, y - h * .025, f"{um:g} µm", color='white', ha='center', va='bottom', fontsize=6)


def read_stack(run, key):
    from pathlib import Path
    from fishsuite.core import io as fio
    img = fio.read_image(Path(run.raw_path(key)))
    return np.stack([fio.extract_channel(img, c, z_mode="3d") for c in (0, 1, 2)])  # miat, qki, dapi


def xz_section(sub, cy):
    """XZ section of a (channel, z, y, x) sub-stack at row cy; row i = plane i (z increases downward, as in B2)."""
    return {"dapi": sub[2, :, cy, :], "miat": sub[0, :, cy, :], "qki": sub[1, :, cy, :]}


def build_linked_set(run, field_id, nucleus_id, rep_distance, label="", mix=False, R_PX=2.0, stem_suffix=None,
                     z_window_um=10.5, write_params=True, channel_labels=("MIAT-640", "QKI-561"), qki_min=1050,
                     miat_min=500, arm_color=None, b4_floors=None):
    """Render B1-B4 for <arm> field <field_id>, nucleus <nucleus_id>; returns the params dict (also written as JSON).

    channel_labels / qki_min / miat_min / arm_color default to the locked basal values (two-condition datasets pass
    their own run values). b4_floors = {"miat": (value, source), "qki": (value, source)} draws each floor as a
    horizontal line on its own B4 axis (MIAT solid-dash gold, QKI dash magenta).
    mix=True renders MIAT x QKI merges only (no DAPI) for B1/B3/B2 with suffix _merge_MIATxQKI; B4 and params are skipped.
    stem_suffix overrides the B1 stem tail (round 1 used ``_box`` without the nucleus id for field 15 nucleus 11).
    """
    from pathlib import Path
    from fishsuite.core.ortho_profile import fixed_z_window, line_profile, nucleus_crop, render_ortho_figure

    saver = run.saver
    tag = run.arm.replace(" ", "")
    SFX = "_merge_MIATxQKI" if mix else ""
    _m = (lambda d: {k: v for k, v in d.items() if k != "dapi"}) if mix else (lambda d: d)
    FLD = str(field_id); NID = int(nucleus_id); LBL = f" ({label})" if label else ""
    G = linked_geometry(run, FLD, NID, R_PX)
    P, mask, px, DZ, BOX = G["plane"], G["mask"], G["px"], G["dz"], G["box"]
    my, mx, on, L = G["my"], G["mx"], G["on"], G["line"]
    P0, P1 = L["P0"], L["P1"]
    ys, xs = np.nonzero(mask)
    mk_id = int(G["mark"].spot_id)

    stack = read_stack(run, G["key"])
    z0 = P["z0"]
    chk = {c: bool(np.array_equal(stack[i, z0], P[c])) for i, c in enumerate(("miat", "qki", "dapi"))}
    assert all(chk.values()), chk
    cy, cx = int(round(my)), int(round(mx))
    zz0, zz1, _ = fixed_z_window(stack[2], mask, dz_eff_um=DZ * AXIAL, window_um=z_window_um, must_include=[z0])

    # B1: FOV ortho with box + line
    sub = stack[:, zz0:zz1]; nzw = sub.shape[1]; H, W = stack.shape[2:]
    xy = colorize(_m({"dapi": stack[2, z0], "miat": stack[0, z0], "qki": stack[1, z0]}))
    xz = colorize(_m(xz_section(sub, cy)))  # row i = plane zz0 + i, drawn top-down like B2 (render_ortho_figure)
    yz = colorize(_m({"dapi": sub[2, :, :, cx].T, "miat": sub[0, :, :, cx].T, "qki": sub[1, :, :, cx].T}))
    zum = nzw * DZ * AXIAL; wum = W * px; hum = H * px; sc = 3.4 / wum; gap = .05
    fw = (wum + zum) * sc + .25 + gap; fh = (hum + zum) * sc + .6 + gap
    fig = plt.figure(figsize=(fw, fh), facecolor='white')
    l = .1 / fw; b = .3 / fh
    axy = fig.add_axes([l, b + (zum * sc + gap) / fh, wum * sc / fw, hum * sc / fh])
    axz = fig.add_axes([l, b, wum * sc / fw, zum * sc / fh])
    ayz = fig.add_axes([l + (wum * sc + gap) / fw, b + (zum * sc + gap) / fh, zum * sc / fw, hum * sc / fh])
    axy.imshow(xy, extent=[0, wum, hum, 0], interpolation='nearest')
    axz.imshow(xz, extent=[0, wum, zum, 0], interpolation='nearest', aspect='auto')  # z increases downward
    ayz.imshow(yz, extent=[0, zum, hum, 0], interpolation='nearest', aspect='auto')
    zpos = (z0 - zz0 + .5) * DZ * AXIAL
    for a in (axy, axz, ayz):
        a.set_xticks([]); a.set_yticks([])
    axy.axhline((cy + .5) * px, color='white', lw=.4, ls=':'); axy.axvline((cx + .5) * px, color='white', lw=.4, ls=':')
    axz.axvline((cx + .5) * px, color='white', lw=.4, ls=':'); axz.axhline(zpos, color='white', lw=.4, ls=':')
    ayz.axhline((cy + .5) * px, color='white', lw=.4, ls=':'); ayz.axvline(zpos, color='white', lw=.4, ls=':')
    axy.add_patch(Rectangle((BOX[0] * px, BOX[1] * px), BOX[2] * px, BOX[3] * px, fill=False, ec='white', lw=.8))
    axy.plot([(P0[1] + .5) * px, (P1[1] + .5) * px], [(P0[0] + .5) * px, (P1[0] + .5) * px], color='white', lw=.6)
    axy.plot([wum * .96 - 20, wum * .96], [hum * .95] * 2, color='white', lw=2)
    axy.text(wum * .96 - 10, hum * .93, "20 µm", color='white', ha='center', fontsize=6)
    axz.plot([wum * .02] * 2, [zum * .1, zum * .1 + 5], color='white', lw=2)
    axz.text(wum * .03, zum * .1 + 2.5, "5 µm (z)", color='white', va='center', fontsize=5)
    fig.text(.5, 1 - .1 / fh, f"{run.arm}, field {FLD}: orthogonal views (merge){LBL}", ha='center', va='top', fontsize=7)
    footer(fig, [f"Box = nucleus {NID}; line = B3 profile line; dotted = sections through the marked MIAT punctum. "
                 f"Z: {nzw} planes x {DZ:g} µm x {AXIAL} = {zum:.1f} µm."], width=int(fw * 72 / 2.6))
    b1_tail = stem_suffix if stem_suffix is not None else f"_nuc{NID}_box"
    saver.save(fig, f"B1_FOV_ortho_{tag}_field{FLD}{b1_tail}{SFX}")

    # B3: zoom of nucleus (analysed plane) with line + on-line puncta
    pad = int(np.ceil(1.5 / px))
    y0c, y1c = max(ys.min() - pad, 0), min(ys.max() + pad + 1, H)
    x0c, x1c = max(xs.min() - pad, 0), min(xs.max() + pad + 1, W)
    zoom = colorize(_m({"dapi": P["dapi"][y0c:y1c, x0c:x1c], "miat": P["miat"][y0c:y1c, x0c:x1c], "qki": P["qki"][y0c:y1c, x0c:x1c]}))
    fig = plt.figure(figsize=(2.8, 3.15), facecolor='white')
    ax = fig.add_axes([.03, .12, .94, .8]); ax.imshow(zoom, interpolation='nearest'); ax.set_axis_off()
    ax.contour(mask[y0c:y1c, x0c:x1c], levels=[.5], colors='white', linewidths=.4)
    ax.plot([P0[1] - x0c, P1[1] - x0c], [P0[0] - y0c, P1[0] - y0c], color='white', lw=.8)
    for s in on.itertuples():
        ax.add_patch(Circle((s.center_x_px - x0c, s.center_y_px - y0c), 4.5, fill=False, ec='white', lw=.5))
    ax.plot([mx - x0c - 9, mx - x0c - 5], [my - y0c] * 2, color='white', lw=.7)
    ax.plot([mx - x0c + 5, mx - x0c + 9], [my - y0c] * 2, color='white', lw=.7)
    _scalebar(ax, 2, zoom.shape[:2], px)
    fig.text(.5, .955, f"{run.arm}, field {FLD}, nucleus {NID}: profile line" + (f"\n{label}" if label else ""), ha='center', va='center', fontsize=7)
    footer(fig, [f"Line through the marked punctum (ticks) crossing {len(on)} MIAT puncta (circles); plane z = {z0 + 1}."])
    saver.save(fig, f"B3_nucleus_zoom_line_{tag}_field{FLD}_nuc{NID}{SFX}")

    # B2: fishsuite zoomed nucleus ortho, same spot + same line
    _, half, _ = nucleus_crop(mask, px)
    fig = render_ortho_figure(stack, (z0, cy, cx), half, nucleus_mask=mask, pixel_size_um=px, z_step_um=DZ,
                              display_levels=[LV["miat"], LV["qki"]], line_endpoints=(P0, P1), profile_width_px=3,
                              analysed_plane_z=z0, qki_min=qki_min, miat_min=miat_min, run_dir=str(run.run_dir), arm=run.arm, arm_color=arm_color or COL,
                              image=Path(G["key"]).name, nucleus_id=NID, metric="representative-rule distance",
                              metric_value=float(rep_distance), arm_median=0.0, include_dapi=not mix, dapi_display_level=LV["dapi"],
                              channel_labels=tuple(channel_labels), dapi_label="DAPI", axial_scale_factor=AXIAL,
                              show_scale_bars=True, show_z_slice_labels=True, show_cross_section=True,
                              z_range=(zz0, zz1), z_crop_note=f"fixed {z_window_um:g} µm effective z window (display only)",
                              axial_scale_note="GLOX n≈1.333 / oil n=1.518; first-order correction")
    saver.save(fig, f"B2_nucleus_ortho_{tag}_field{FLD}_nuc{NID}_fishsuite{SFX}")
    if mix:
        return None

    # B4: line profile
    dist, vals = line_profile(np.stack([P["miat"], P["qki"], P["dapi"]]), P0, P1, width_px=3, pixel_size_um=px)
    prof = pd.DataFrame(dict(distance_um=dist, miat=vals[0], qki=vals[1], dapi=vals[2]))
    data = run.data
    data.mkdir(parents=True, exist_ok=True)
    prof.to_csv(data / f"B4_line_profile_values_field{FLD}_nuc{NID}.csv", index=False)
    fig = plt.figure(figsize=(4.2, 2.5), facecolor='white')
    ax = fig.add_axes([.13, .3, .56, .55]); ax2 = ax.twinx(); ax3 = ax.twinx()
    ax3.spines["right"].set_position(("axes", 1.33)); ax2.spines["right"].set_visible(True); ax3.spines["right"].set_visible(True)
    ax.plot(dist, prof.miat, color=LINE_MIAT, lw=1.1, label=f"MIAT ({channel_labels[0].split('-')[-1]})")
    ax2.plot(dist, prof.qki, color=LINE_QKI, lw=1.1, label=f"QKI ({channel_labels[1].split('-')[-1]})")
    ax3.plot(dist, prof.dapi, color=LINE_DAPI, lw=.8, ls='--', label="DAPI")
    for s in on.itertuples():
        ax.axvline(s.pos_um, color='#888888', lw=.5, ls=':', zorder=0)
        ax.plot(s.pos_um, 1.02, marker='v', ms=4, transform=ax.get_xaxis_transform(), clip_on=False,
                color='black' if s.spot_id == mk_id else '#888888')
    for key, a_, c_ in (("miat", ax, LINE_MIAT), ("qki", ax2, LINE_QKI)):
        if b4_floors and b4_floors.get(key) is not None:
            fv = float(b4_floors[key][0])
            a_.axhline(fv, color=c_, lw=.8, ls=(0, (4, 2)), zorder=1,
                       label=f"{'MIAT' if key == 'miat' else 'QKI'} floor {fv:g}")
            lo_, hi_ = a_.get_ylim()
            a_.set_ylim(min(lo_, fv - .05 * (hi_ - lo_)), max(hi_, fv + .05 * (hi_ - lo_)))
    ax.set_xlabel("Distance along line (µm)"); ax.set_xlim(0, dist.max())
    ax.set_ylabel("MIAT (raw a.u.)", color='#8A7400'); ax2.set_ylabel("QKI (raw a.u.)", color=LINE_QKI)
    ax3.set_ylabel("DAPI (raw a.u.)", color=LINE_DAPI)
    h = sum((a.get_legend_handles_labels()[0] for a in (ax, ax2, ax3)), [])
    fig.legend(h, [x.get_label() for x in h], loc='upper center', ncol=3, frameon=False, fontsize=6, bbox_to_anchor=(.42, 1.0))
    footer(fig, [f"{run.arm} field {FLD} nucleus {NID}{LBL}; 3-px-wide line, analysed plane z = {z0 + 1}. Triangles = MIAT puncta "
                 f"on the line (n = {len(on)}; black = marked punctum). n = 1 line, illustrative."], width=95)
    saver.save(fig, f"B4_line_profile_{tag}_field{FLD}_nuc{NID}")

    params = params_dict(G, FLD, NID, label, R_PX, (zz0, zz1), chk)
    if write_params:
        (data / f"B_linked_params_field{FLD}_nuc{NID}.json").write_text(json.dumps(params, indent=1, default=float))
    return params
