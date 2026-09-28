"""Rebuild the basal v2 figure gallery (A FOVs, B linked sets, R representative crops, C Holm superplots, D scatters,
MIAT x QKI merges, native TIFFs, stats workbook, index.html) from one PanelRun. Driver logic of 04/06/07 (2026-09-28)."""
from __future__ import annotations

import html
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from . import selection
from .c_superplots import build_c_panels
from .config import PanelRun
from .d_scatters import build_d_round2
from .fov import build_fov, export_native_tiffs
from .linked_set import build_linked_set
from .rep_crops import build_rep_crops
from .style import apply_style, check_luts

STEPS = ("select", "fov", "linked", "reps", "c", "d", "mix", "tiff", "index")

GROUPS = [("A. Whole-FOV analysed planes", lambda n: n.startswith("A_") and "MIATxQKI" not in n),
          ("B. Linked ortho sets (same nucleus, spot and line across B1-B4)", lambda n: n.startswith("B") and "MIATxQKI" not in n),
          ("R. Representative colocalization nuclei (closest to median)", lambda n: n.startswith("R_coloc") and "MIATxQKI" not in n),
          ("C. Colocalization superplots (nucleus-unit Wilcoxon, Holm stars)", lambda n: n.startswith("C")),
          ("D. Correlation scatters", lambda n: n.startswith("D")),
          ("MIAT x QKI merges (no DAPI)", lambda n: "MIATxQKI" in n)]


def seed_all(seed=0):
    np.random.seed(seed)
    random.seed(seed)


def write_index(out_dir: Path, title: str):
    pan = out_dir / "panels"
    pngs = sorted(p.name for p in pan.glob("*.png"))

    def card(n):
        s = n[:-4]
        nat = f" · <a href='panels/{s}_native.svg'>SVG native</a>" if (pan / f"{s}_native.svg").exists() else ""
        return (f"<div class='c'><a href='panels/{n}'><img src='panels/{n}'></a><br>{html.escape(s)}<br>"
                f"<a href='panels/{s}.png'>PNG</a> · <a href='panels/{s}.svg'>SVG</a>{nat} · <a href='panels/{s}.pdf'>PDF</a></div>")
    H = ["<!doctype html><html><head><meta charset='utf-8'><title>" + html.escape(title) + "</title><style>body{font-family:Arial;margin:20px}"
         ".g{display:flex;flex-wrap:wrap;gap:14px}.c{width:300px;font-size:12px}.c img{width:300px;border:1px solid #ccc}</style></head><body>",
         f"<h1>{html.escape(title)}</h1><p><a href='stats.xlsx'>stats.xlsx</a> · <a href='data/'>data/</a> · "
         "<a href='native_tiff/'>native_tiff/</a> (raw 16-bit). 'SVG native' = rasters embedded at acquisition resolution.</p>"]
    for t, f in GROUPS:
        cards = [card(n) for n in pngs if f(n)]
        if cards:
            H.append(f"<h2>{html.escape(t)}</h2><div class='g'>" + "".join(cards) + "</div>")
    H.append("</body></html>")
    (out_dir / "index.html").write_text("\n".join(H), encoding="utf8")
    return len(pngs)


def build_gallery(run: PanelRun, steps=STEPS, rep_field="14", fov_fields=None, extra_linked=(), seed=0, log=print):
    """extra_linked: iterable of (field, nucleus_id, label) built in addition to the arm representative and the
    objective rep_field pick (e.g. ("14", 14, "delivery Fig1A nucleus"))."""
    apply_style()
    check_luts()
    seed_all(seed)
    data = run.data
    data.mkdir(parents=True, exist_ok=True)
    nv, sv = run.tables()
    fields = sorted(nv.field.unique())
    fov_fields = list(fov_fields or fields)
    rep = run.selection()["reps"][run.arm]
    rep_fld = str(rep["well"]).split("_")[-1]
    out = {}

    sel_path = data / "round2_selections.json"
    if "select" in steps or not sel_path.exists():
        sel = {"field14_rep" if str(rep_field) == "14" else f"field{rep_field}_rep": selection.field_rep(nv, rep_field),
               "coloc_reps": selection.coloc_reps(nv)}
        sel_path.write_text(json.dumps(sel, indent=1, default=float))
    sel = json.loads(sel_path.read_text())
    fr = sel.get("field14_rep") or sel[f"field{rep_field}_rep"]

    linked = [(rep_fld, int(rep["nucleus_id"]), float(rep["distance"]), "")]
    linked.append((str(rep_field), int(fr["nucleus_id"]), float(fr["distance"]), f"objective field-{rep_field} representative"))
    for f, n, lab in extra_linked:
        r = nv[(nv.field == str(f)) & (nv.nucleus_id == int(n))].iloc[0]
        linked.append((str(f), int(n), selection.robust_distance(r, fr["medians"], fr["scale"]), lab))

    if "fov" in steps:
        for f in fov_fields:
            build_fov(run, f, channels=("merge", "DAPI", "MIAT", "QKI"))
    if "linked" in steps:
        B = {}
        for f, n, dist, lab in linked:
            B[f"field{f}_nuc{n}"] = build_linked_set(run, f, n, dist, label=lab)
            log(f"linked set field {f} nucleus {n}: line {B[f'field{f}_nuc{n}']['line']['angle_deg']} deg")
        (data / "linked_sets.json").write_text(json.dumps({k: {q: v[q] for q in ("label", "nucleus", "marked_spot", "line", "z_window")}
                                                           for k, v in B.items()}, indent=1, default=float))
        out["linked"] = B
    tables = {}
    if "reps" in steps:
        tables["coloc_representatives"] = build_rep_crops(run, sel["coloc_reps"]["picks"], nv, sv, channels=("merge", "MIAT", "QKI", "DAPI"))
    if "c" in steps:
        tables["C_holm_effect"] = build_c_panels(nv, fields, run.saver, arm=run.arm, seed=seed)
    if "d" in steps:
        tables["D_correlations"] = pd.DataFrame(build_d_round2(nv, sv, run.saver, footer_prefix=f"{run.arm}, fields {' + '.join(fields)} combined"))
    if "mix" in steps:
        for f in fov_fields:
            build_fov(run, f, channels=("merge_MIATxQKI",))
        for f, n, dist, lab in linked:
            build_linked_set(run, f, n, dist, label=lab, mix=True)
        build_rep_crops(run, sel["coloc_reps"]["picks"], nv, sv, channels=("merge_MIATxQKI",))
    if "tiff" in steps:
        pj = {}
        for f, n, _, _ in linked:
            p = data / f"B_linked_params_field{f}_nuc{n}.json"
            if p.exists():
                pj[(f, n)] = json.loads(p.read_text())["z_window"]
        rows = export_native_tiffs(run, run.out_dir / "native_tiff", fields=fov_fields,
                                   nuclei=[(f, n, z[0], z[1]) for (f, n), z in pj.items()])
        pd.DataFrame(rows, columns=["file", "dtype", "shape", "min", "max"]).to_csv(run.out_dir / "native_tiff" / "manifest.csv", index=False)
    for name, df in tables.items():
        df.to_csv(data / f"{name}.csv", index=False)
    if tables and not run.native_only:
        with pd.ExcelWriter(run.out_dir / "stats.xlsx") as xw:
            for name, df in tables.items():
                df.to_excel(xw, sheet_name=name[:31], index=False)
    if "index" in steps and not run.native_only:
        out["n_png"] = write_index(run.out_dir, f"Basal MIAT x QKI figure panels: {run.arm}")
    out["tables"] = tables
    return out
