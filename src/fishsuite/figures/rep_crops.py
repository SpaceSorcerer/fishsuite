"""Representative colocalization nucleus crops (merge / MIAT / QKI / DAPI / MIAT x QKI, with and without outlines)
with the four metric values beneath. Ported from fig1lib.rep_crops (2026-09-28)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from .style import PX_UM, colorize, footer

CHANNELS = ("merge", "MIAT", "QKI", "DAPI", "merge_MIATxQKI")


def build_rep_crops(run, picks, nv, sv, channels=CHANNELS, px=PX_UM):
    import h5py
    H5 = {}
    with h5py.File(run.h5, "r") as f:
        keys = {p["image"] for p in picks}
        for k in f["images"]:
            g = f["images"][k]
            if g.attrs["image_key"] in keys:
                H5[g.attrs["image_key"]] = ({c: g[c][()] for c in ("dapi", "miat", "qki", "nucleus_labels")}
                                            | {"z0": int(g.attrs["selected_z_0based"])})
    rows = []
    for rank, p in enumerate(picks, 1):
        P = H5[p["image"]]; nid = int(p["nucleus_id"]); mask = P["nucleus_labels"] == nid
        ys, xs = np.nonzero(mask); pad = int(np.ceil(1.5 / px)); H, W = mask.shape
        y0, y1, x0, x1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, H), max(xs.min() - pad, 0), min(xs.max() + pad + 1, W)
        ns = sv[(sv.image == p["image"]) & (sv.nucleus_id == nid)]
        row = nv[(nv.image == p["image"]) & (nv.nucleus_id == nid)].iloc[0]
        vals = (f"mean UPP {row.mean_uniform_position_percentile_qki:.3f} | Pearson r {row.pearson_r_nucleoplasm:.3f} | "
                f"Spearman ρ {row.spearman_rho_nucleoplasm:.3f} | frac puncta QKI+ {row.frac_miat_spots_qki_pos:.3f} ({int(row.n_miat_spots)} puncta)")
        rows.append(dict(rank=rank, field=p["field"], nucleus_id=nid, robust_distance=p["d"], z0=P["z0"], crop_yx=[int(y0), int(y1), int(x0), int(x1)],
                         mean_upp=row.mean_uniform_position_percentile_qki, pearson=row.pearson_r_nucleoplasm,
                         spearman=row.spearman_rho_nucleoplasm, frac_qki_pos=row.frac_miat_spots_qki_pos, n_puncta=int(row.n_miat_spots)))
        sl = (slice(y0, y1), slice(x0, x1))
        views = (("merge", {"dapi": P["dapi"][sl], "miat": P["miat"][sl], "qki": P["qki"][sl]}),
                 ("MIAT", {"miat": P["miat"][sl]}), ("QKI", {"qki": P["qki"][sl]}), ("DAPI", {"dapi": P["dapi"][sl]}),
                 ("merge_MIATxQKI", {"miat": P["miat"][sl], "qki": P["qki"][sl]}))
        for ch, planes in views:
            if ch not in channels:
                continue
            img = colorize(planes)
            for outl in (True, False):
                fig = plt.figure(figsize=(2.6, 3.2), facecolor='white')
                ax = fig.add_axes([.04, .2, .92, .72]); ax.imshow(img, interpolation='nearest'); ax.set_axis_off()
                if outl:
                    ax.contour(mask[sl], levels=[.5], colors='white', linewidths=.4)
                    for s in ns.itertuples():
                        ax.add_patch(Circle((s.center_x_px - x0, s.center_y_px - y0), max(np.sqrt(s.footprint_area_px / np.pi), 1.0) + 1.5,
                                            fill=False, ec='white', lw=.45))
                h_, w_ = img.shape[:2]; L = 2 / px
                ax.plot([w_ * .95 - L, w_ * .95], [h_ * .95] * 2, color='white', lw=2, solid_capstyle='butt')
                ax.text(w_ * .95 - L / 2, h_ * .92, "2 µm", color='white', ha='center', va='bottom', fontsize=6)
                lbl = {"merge": "merge", "MIAT": "MIAT", "QKI": "QKI", "DAPI": "DAPI", "merge_MIATxQKI": "MIAT x QKI merge"}[ch]
                fig.text(.5, .95, f"Representative (closest to median) #{rank}\nfield {p['field']}, nucleus {nid}: {lbl}", ha='center', va='center', fontsize=6.5)
                footer(fig, [vals, f"Plane z = {P['z0'] + 1}." + (" Outline = nucleus mask; circles = MIAT puncta (footprint-area radius + 1.5 px)." if outl else "")], width=70)
                run.saver.save(fig, f"R_coloc_rep{rank}_field{p['field']}_nuc{nid}_{ch}_{'outlines' if outl else 'plain'}")
    return pd.DataFrame(rows)
