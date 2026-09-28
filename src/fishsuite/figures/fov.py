"""Whole-FOV analysed-plane panels (merge, single channels, MIAT x QKI merge) and native 16-bit TIFF export.
Ported from fig1lib.fov (2026-09-28)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .style import PX_UM, colorize, footer

LABEL = {"DAPI": "DAPI", "MIAT": "MIAT RNA-FISH", "QKI": "QKI IF", "merge": "Merge (MIAT yellow, QKI magenta, DAPI blue)",
         "merge_MIATxQKI": "MIAT x QKI merge (MIAT yellow, QKI magenta; no DAPI)"}
CH = {"DAPI": ("dapi",), "MIAT": ("miat",), "QKI": ("qki",), "merge": ("dapi", "miat", "qki"), "merge_MIATxQKI": ("miat", "qki")}


def build_fov(run, field_id, channels=("merge", "DAPI", "MIAT", "QKI", "merge_MIATxQKI"), px=PX_UM):
    P = run.load_plane(run.image_key(field_id)); fld = str(field_id); tag = run.arm.replace(" ", "")
    for chname in channels:
        img = colorize({c: P[c] for c in CH[chname]})
        fig = plt.figure(figsize=(3.4, 3.75), facecolor='white')
        ax = fig.add_axes([.02, .1, .96, .83]); ax.imshow(img, interpolation='nearest'); ax.set_axis_off()
        h, w = img.shape[:2]; L = 20 / px; x1 = w * .96; y = h * .95
        ax.plot([x1 - L, x1], [y, y], color='white', lw=2, solid_capstyle='butt')
        ax.text(x1 - L / 2, y - h * .025, "20 µm", color='white', ha='center', va='bottom', fontsize=6)
        fig.text(.5, .965, f"{run.arm}, field {fld}: {LABEL[chname]}", ha='center', fontsize=7)
        footer(fig, [f"H9 hESC; analysed plane z = {P['z0'] + 1} (1-based). Display: MIAT 500-2250, QKI 1050-3746, DAPI 607-9000 raw a.u."])
        run.saver.save(fig, f"A_FOV_{tag}_field{fld}_{chname}")


def export_native_tiffs(run, out_dir, fields=("14", "15"), nuclei=(), zstack=True, px=PX_UM, z_step_um=0.21):
    """Raw (no display scaling) per-channel TIFFs: FOV analysed plane, and for each (field, nucleus, z0, z1) the nucleus
    crop of the analysed plane plus, if zstack, the crop of the display z window read from the raw VSI (read-only)."""
    import tifffile
    from .linked_set import read_stack
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True); rows = []
    for fld in fields:
        P = run.load_plane(run.image_key(fld))
        for c in ("miat", "qki", "dapi"):
            a = P[c]; fn = out / f"FOV_field{fld}_z{P['z0'] + 1}_{c.upper()}.tif"
            tifffile.imwrite(fn, a.astype(np.uint16) if a.max() <= 65535 else a, resolution=(1 / px, 1 / px),
                             metadata={"unit": "um", "axes": "YX"}, imagej=True)
            rows.append((fn.name, a.dtype.name, a.shape, int(a.min()), int(a.max())))
    stacks = {}
    for fld, nid, z0w, z1w in nuclei:
        key = run.image_key(fld)
        P = run.load_plane(key); m = P["nucleus_labels"] == int(nid); ys, xs = np.nonzero(m)
        pad = int(np.ceil(1.5 / px)); H, W = m.shape
        y0, y1, x0, x1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, H), max(xs.min() - pad, 0), min(xs.max() + pad + 1, W)
        for c in ("miat", "qki", "dapi"):
            a = P[c][y0:y1, x0:x1]; fn = out / f"nucleus_field{fld}_nuc{nid}_plane_z{P['z0'] + 1}_{c.upper()}_crop_y{y0}-{y1}_x{x0}-{x1}.tif"
            tifffile.imwrite(fn, a.astype(np.uint16), resolution=(1 / px, 1 / px), metadata={"unit": "um", "axes": "YX"}, imagej=True)
            rows.append((fn.name, "uint16", a.shape, int(a.min()), int(a.max())))
        tifffile.imwrite(out / f"nucleus_field{fld}_nuc{nid}_mask_crop_y{y0}-{y1}_x{x0}-{x1}.tif", m[y0:y1, x0:x1].astype(np.uint8) * 255, imagej=True)
        if zstack:
            if fld not in stacks:
                stacks[fld] = read_stack(run, key)
            st = stacks[fld]
            assert np.array_equal(st[0, P["z0"]], P["miat"]) and np.array_equal(st[1, P["z0"]], P["qki"]) and np.array_equal(st[2, P["z0"]], P["dapi"])
            for i, c in enumerate(("miat", "qki", "dapi")):
                a = st[i, z0w:z1w, y0:y1, x0:x1]
                fn = out / f"nucleus_field{fld}_nuc{nid}_zstack_z{z0w + 1}-{z1w}_{c.upper()}_crop_y{y0}-{y1}_x{x0}-{x1}.tif"
                tifffile.imwrite(fn, a.astype(np.uint16), resolution=(1 / px, 1 / px),
                                 metadata={"unit": "um", "axes": "ZYX", "spacing": z_step_um}, imagej=True)
                rows.append((fn.name, "uint16", a.shape, int(a.min()), int(a.max())))
    return rows
