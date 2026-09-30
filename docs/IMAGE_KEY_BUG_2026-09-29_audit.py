"""Audit for the PanelRun.image_key bug (2026-09-29): two DIFFERENT field labels sharing identical micrograph pixels.

Units hashed (sha256 of the decoded pixel array + shape + dtype):
  * every raster embedded (data:image/...;base64) in every .svg,
  * every .tif/.tiff array,
  * standalone .png files with no .svg / _native.svg twin.
Rasters smaller than MIN_SIDE px on a side or with < MIN_UNIQUE distinct values (colorbars, blanks) are skipped and counted.
Labels (canonical field, nucleus) are parsed from the file name; composites (no field in name) are 'composite' and are
only used to report which figures embed a flagged raster.
Enumeration is a bounded os.scandir walk (depth <= MAX_DEPTH) of each listed output folder -- no content grep.
Usage: python IMAGE_KEY_BUG_2026-09-29_audit.py <out_csv.gz>
"""
import base64
import hashlib
import io
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image

FOLDERS = [
    r"F:\Image Analysis Work\MIAT_QKI_BASAL_Fig1_2026-09-24\FIG1_IMAGING_v2_2026-09-28",
    r"F:\Publication Work\PRESENTATION_MIAT_KD_OE_2026-09-26\CONFOCAL_GALLERY_v2_2026-09-28",
    r"F:\Publication Work\PRESENTATION_MIAT_KD_OE_2026-09-26\LINKED_ORTHO_2026-09-28",
    r"F:\Publication Work\05_FINALIZATION_2026-09-23\G_figure_rebuild_2026-09-28\confocal_selection_2026-09-29",
    r"F:\Publication Work\05_FINALIZATION_2026-09-23\G_figure_rebuild_2026-09-28\confocal_panels_draft3",
    r"F:\Publication Work\05_FINALIZATION_2026-09-23\G_figure_rebuild_2026-09-28\confocal_panels_draft3_20260929",
    r"F:\Publication Work\05_FINALIZATION_2026-09-23\fig_drafts_2026-09-29\Fig1",
    r"F:\Publication Work\05_FINALIZATION_2026-09-23\fig_drafts_2026-09-29\Fig1_supp",
    r"F:\Publication Work\05_FINALIZATION_2026-09-23\fig_drafts_2026-09-29\Fig4",
]
MAX_DEPTH, MIN_SIDE, MIN_UNIQUE = 5, 32, 16
B64 = re.compile(rb'href="data:image/(png|jpeg|jpg);base64,([^"]+)"')
FIELD_RES = [  # case-insensitive; first match wins
    re.compile(r"((?:KD|NT)[-_]\d+_\d+)", re.I),
    re.compile(r"(g2[-+]?(?:no)?Dox_\d+)", re.I),
    re.compile(r"field_?(\d+)(?!\d)", re.I),
]
NUC = re.compile(r"nuc(\d+|ALL)")


def walk(root):
    stack = [(Path(root), 0)]
    while stack:
        d, depth = stack.pop()
        with os.scandir(d) as it:
            for e in it:
                if e.is_dir():
                    if depth < MAX_DEPTH:
                        stack.append((Path(e.path), depth + 1))
                elif e.name.lower().endswith((".svg", ".tif", ".tiff", ".png")):
                    yield Path(e.path)


def canon_field(name):
    for r in FIELD_RES:
        m = r.search(name)
        if m:
            f = m.group(1)
            if f.isdigit():  # basal UD fields: g2 noDox 03/04 are the same raw files as OE g2-noDox_03/04
                return f"g2-noDox_{int(f):02d}" if int(f) in (3, 4) else f"UD_{int(f):02d}"
            if f.lower().startswith("g2"):
                return ("g2-noDox_" if "nodox" in f.lower() else "g2-Dox_") + f.rsplit("_", 1)[1]
            return f[:2].upper() + f[2:]
    return None


def arr_hash(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str((a.shape, a.dtype.str)).encode() + a.tobytes()).hexdigest()


def keep(a):
    return min(a.shape[:2]) >= MIN_SIDE and len(np.unique(a.reshape(-1)[:: max(1, a.size // 200000)])) >= MIN_UNIQUE


def rasters(p):
    s = p.suffix.lower()
    if s == ".svg":
        for i, m in enumerate(B64.finditer(p.read_bytes())):
            yield i, np.asarray(Image.open(io.BytesIO(base64.b64decode(m.group(2)))))
    elif s in (".tif", ".tiff"):
        yield 0, tifffile.imread(p)
    else:
        yield 0, np.asarray(Image.open(p))


def main(out_csv):
    if "--relabel" in sys.argv:  # re-run labelling/flagging on an existing hash table (no F: reads)
        return analyse(pd.read_csv(out_csv))
    rows, per_folder = [], {}
    for root in FOLDERS:
        files = sorted(walk(root))
        svg_stems = {f.with_suffix("") for f in files if f.suffix.lower() == ".svg"}
        n_tiles = n_skip = 0
        for f in files:
            if f.suffix.lower() == ".png":
                st = f.with_suffix("")
                if st in svg_stems or Path(str(st) + "_native") in svg_stems or Path(str(st).removesuffix("_preview")) in svg_stems:
                    continue
            try:
                for i, a in rasters(f):
                    if not keep(a):
                        n_skip += 1
                        continue
                    n_tiles += 1
                    rel = str(f.relative_to(root))
                    m = NUC.search(f.stem)
                    rows.append(dict(folder=root, file=rel, raster=i, shape="x".join(map(str, a.shape)),
                                     field=canon_field(f.stem), nucleus=m.group(1) if m else None,
                                     superseded="_superseded" in rel, pixel_sha256=arr_hash(a)))
            except Exception as e:  # unreadable file is reported, not hidden
                rows.append(dict(folder=root, file=str(f.relative_to(root)), raster=-1, error=repr(e)))
        per_folder[root] = (len(files), n_tiles, n_skip)
        print(f"{root}\n  files={len(files)} tiles_hashed={n_tiles} skipped_small_or_flat={n_skip}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    analyse(df)


def analyse(df):
    stem = df.file.map(lambda s: re.split(r"[\\/]", s)[-1].rsplit(".", 1)[0])
    df["field"] = stem.map(canon_field)
    df["nucleus"] = stem.map(lambda s: (NUC.search(s) or [None, None])[1])
    df["superseded"] = df.file.str.contains("_superseded")
    for fo, g in df.groupby("folder", sort=False):
        print(f"{fo}: tiles_hashed={g.pixel_sha256.notna().sum()} labelled={g.field.notna().sum()} "
              f"superseded={g.superseded.sum()}")
    lab = df[df.field.notna() & df.pixel_sha256.notna()]
    for name, sub in (("CURRENT", lab[~lab.superseded]), ("SUPERSEDED", lab[lab.superseded])):
        n = sub.groupby("pixel_sha256").field.nunique()
        bad = sub[sub.pixel_sha256.isin(n[n > 1].index)]
        print(f"\n{name}: flagged hashes (>=2 different fields share pixels) = {bad.pixel_sha256.nunique()}, files = {bad.file.nunique()}")
        for fn in sorted(set(bad.folder + " | " + bad.file)):
            print("   ", fn)
    cur = set(lab[~lab.superseded].pixel_sha256)
    buggy_only = set(lab[lab.superseded].pixel_sha256) - cur
    comp = df[df.field.isna() & ~df.superseded & df.pixel_sha256.isin(buggy_only)]
    print("\ncurrent composites embedding a superseded-only raster:", sorted(set(comp.folder + " | " + comp.file)) or "none")
    errs = df[df.raster == -1]
    print("read errors:", len(errs))
    for r in errs.itertuples():
        print("   ", r.folder, r.file, getattr(r, "error", ""))
    dup = lab[lab.nucleus.notna() & ~lab.superseded].groupby("pixel_sha256").filter(
        lambda g: g.nucleus.nunique() > 1 and g.field.nunique() == 1)
    print("same-field, different-nucleus shared hashes (context):", dup.pixel_sha256.nunique(),
          "shapes:", sorted(dup["shape"].unique()))


if __name__ == "__main__":
    main(sys.argv[1])
