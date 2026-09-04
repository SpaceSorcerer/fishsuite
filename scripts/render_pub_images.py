"""Post-hoc re-render of fishsuite publication_images for a COMPLETED run at a
new manual display window. CPU only: no segmentation, no spot detection.

Reuses fishsuite's own LUT / uint8 / scale-bar code via
``core.output.save_publication_images_bundle`` so a render at the run's
original window is byte-identical to the run's own PNGs.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from fishsuite.core import io as _io
from fishsuite.core import output as _out
from fishsuite.runner import (
    _compute_common_filename_prefix,
    _simplify_stem,
    _stem_with_condition,
)


def _parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--images", default=None,
                   help="comma-separated substrings or globs matched against per_image_summary.image")
    p.add_argument("--input-dir", type=Path, default=None)
    p.add_argument("--tifs", action="store_true")
    p.add_argument("--z-from-overrides", action="store_true")
    for ch in ("rna", "rna2", "dapi", "ab"):
        p.add_argument("--{}-min".format(ch), type=float, default=None)
        p.add_argument("--{}-max".format(ch), type=float, default=None)
        p.add_argument("--{}-lut".format(ch), default=None,
                       help="LUT name as accepted by the bundle (blue/yellow/magenta/cyan/"
                            "green/...). Default: the run_config value.")
    for ch in ("rna", "rna2", "ab"):
        p.add_argument("--{}-label".format(ch), default=None,
                       help="Channel label. Default: the run_config value. The bundle "
                            "derives output filenames from labels.")
    return p.parse_args(argv)


def _select(images, patterns):
    if not patterns:
        return list(images)
    sel = []
    for im in images:
        for pat in patterns:
            if fnmatch.fnmatch(im, pat) or pat in im:
                sel.append(im)
                break
    return sel


def _resolve_z(row, name, zov, use_ov):
    z = row.get("z_plane", None)
    if z is not None and pd.notna(z):
        return int(z)
    if not use_ov:
        raise SystemExit(
            "{}: z_plane missing in per_image_summary.csv; pass --z-from-overrides "
            "to fall back to z_stack.file_overrides".format(name))
    for key, ov in (zov or {}).items():
        if key == name or Path(key).name == name or Path(key).stem == Path(name).stem:
            s, e = ov.get("start_slice"), ov.get("end_slice")
            if s is not None and s == e:
                return int(s)
            raise SystemExit(
                "{}: file_override start_slice={} end_slice={} is not a single "
                "plane".format(name, s, e))
    raise SystemExit(
        "{}: no z_plane and no single-plane file_override - refusing to guess".format(name))


def main(argv=None):
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    run = args.run.resolve()
    rc = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
    cfg = rc["config_resolved"]
    ch, ocfg = cfg["channels"], cfg["output"]
    mode = ch["analysis_mode"]
    if mode not in ("rna_rna", "rna_protein", "rna_only"):
        raise SystemExit("unsupported analysis_mode {!r}".format(mode))

    win = {}
    for key, cli_lo, cli_hi, m_lo, m_hi in (
        ("dapi", args.dapi_min, args.dapi_max, "manual_dapi_min", "manual_dapi_max"),
        ("rna", args.rna_min, args.rna_max, "manual_rna_min", "manual_rna_max"),
        ("rna2", args.rna2_min, args.rna2_max, "manual_rna2_min", "manual_rna2_max"),
        ("ab", args.ab_min, args.ab_max, "manual_antibody_min", "manual_antibody_max"),
    ):
        lo = cli_lo if cli_lo is not None else ocfg.get(m_lo)
        hi = cli_hi if cli_hi is not None else ocfg.get(m_hi)
        win[key] = (None if lo is None else float(lo), None if hi is None else float(hi))

    # LUT / label overrides. None keeps the run_config value, so default output is
    # unchanged. The bundle builds filenames from the labels and LUT names.
    lut = {"dapi": args.dapi_lut or ch.get("dapi_lut"),
           "rna": args.rna_lut or ch.get("rna_lut"),
           "rna2": args.rna2_lut or ch.get("rna2_lut"),
           "ab": args.ab_lut or ch.get("antibody_lut")}
    lbl = {"dapi": ch.get("dapi_label"),
           "rna": args.rna_label or ch.get("rna_label"),
           "rna2": args.rna2_label or ch.get("rna2_label"),
           "ab": args.ab_label or ch.get("antibody_label")}

    out_dir = Path(args.out) if args.out else (
        run / "publication_images_rerender_{:%Y%m%d-%H%M%S}".format(datetime.now()))
    out_dir.mkdir(parents=True, exist_ok=True)

    summ = pd.read_csv(run / "per_image_summary.csv")
    roster = [Path(s).stem for s in summ["image"].astype(str)]
    prefix = _compute_common_filename_prefix(roster)
    patterns = [s for s in (args.images or "").split(",") if s.strip()] or None
    picked = _select(summ["image"].astype(str).tolist(), patterns)
    if not picked:
        raise SystemExit("--images {!r} matched 0 of {} rows".format(args.images, len(summ)))

    input_dir = Path(args.input_dir) if args.input_dir else Path(rc["input_dir"])
    one_ix = bool(ch.get("one_indexed", False))

    def _cidx(i):
        i = int(i)
        return (i - 1) if (one_ix and i > 0) else i

    _out.SCALEBAR_UM = float(ocfg["scalebar_um"])
    _out.SCALEBAR_FONT_PX = int(ocfg["scalebar_font_px"])
    zov = cfg.get("z_stack", {}).get("file_overrides", {})

    written = []
    for name in picked:
        row = summ.loc[summ["image"].astype(str) == name].iloc[0]
        hits = sorted(input_dir.rglob(name))
        if len(hits) != 1:
            raise SystemExit("{}: expected exactly 1 match under {}, found {}: {}".format(
                name, input_dir, len(hits), hits))
        z = _resolve_z(row, name, zov, args.z_from_overrides)
        img = _io.read_image(hits[0])
        vx_img, vx_csv = float(img.voxel_xy_nm), float(row["voxel_xy_nm"])
        if abs(vx_img - vx_csv) > 1e-6:
            raise SystemExit("{}: voxel_xy_nm image={} csv={}".format(name, vx_img, vx_csv))
        dapi = _io.extract_channel_at_z(img, _cidx(ch["dapi"]), z_1indexed=z)
        rna = _io.extract_channel_at_z(img, _cidx(ch["rna"]), z_1indexed=z)
        second = None
        if mode == "rna_rna":
            second = _io.extract_channel_at_z(img, _cidx(ch["rna2"]), z_1indexed=z)
        elif mode == "rna_protein":
            second = _io.extract_channel_at_z(img, _cidx(ch["antibody"]), z_1indexed=z)
        cond = None if pd.isna(row["condition"]) else str(row["condition"])
        stem = _stem_with_condition(_simplify_stem(Path(name).stem, prefix), cond)
        paths = _out.save_publication_images_bundle(
            out_dir, stem, dapi, rna, vx_csv,
            rna2=second if mode == "rna_rna" else None,
            protein=second if mode == "rna_protein" else None,
            sec_only=bool(row["secondary_only"]),
            dapi_floor=win["dapi"][0], dapi_ceil=win["dapi"][1],
            rna_floor=win["rna"][0], rna_ceil=win["rna"][1],
            rna2_floor=win["rna2"][0], rna2_ceil=win["rna2"][1],
            ab_floor=win["ab"][0], ab_ceil=win["ab"][1],
            dapi_floor_pct=float(ocfg["pub_contrast_dapi_floor_pct"]),
            dapi_ceil_pct=float(ocfg["pub_contrast_dapi_ceil_pct"]),
            rna_floor_pct=float(ocfg["pub_contrast_floor_pct"]),
            rna_ceil_pct=float(ocfg["pub_contrast_ceil_pct"]),
            rna2_floor_pct=float(ocfg["pub_contrast_floor_pct"]),
            rna2_ceil_pct=float(ocfg["pub_contrast_ceil_pct"]),
            rna_floor_bump_pct=float(ocfg.get("pub_contrast_rna_floor_bump_pct", 0.0)),
            save_tifs=bool(args.tifs),
            dapi_label=lbl["dapi"], rna_label=lbl["rna"],
            rna2_label=lbl["rna2"], antibody_label=lbl["ab"],
            dapi_lut=lut["dapi"], rna_lut=lut["rna"],
            rna2_lut=lut["rna2"], antibody_lut=lut["ab"],
        )
        written.extend(paths)
        print("{}  z={}  vx={}  stem={}  -> {} files".format(name, z, vx_csv, stem, len(paths)))

    import PIL
    import fishsuite
    (out_dir / "command.log").write_text(
        "{}\ncwd={}\n{} {}\n".format(datetime.now().isoformat(), os.getcwd(),
                                     sys.executable, " ".join(sys.argv)), encoding="utf-8")
    (out_dir / "versions.txt").write_text(
        "python={}\nfishsuite={}\nfishsuite.__file__={}\nnumpy={}\npandas={}\npillow={}\n".format(
            sys.version.split()[0], fishsuite.__version__, fishsuite.__file__,
            np.__version__, pd.__version__, PIL.__version__), encoding="utf-8")
    (out_dir / "render_params.json").write_text(json.dumps({
        "run": str(run), "input_dir": str(input_dir), "analysis_mode": mode,
        "prefix": prefix, "n_roster": len(roster), "images": picked,
        "window": {k: list(v) for k, v in win.items()},
        "luts_used": lut, "labels_used": lbl,
        "lut_overrides": {k: v for k, v in (("dapi", args.dapi_lut), ("rna", args.rna_lut),
                                            ("rna2", args.rna2_lut), ("ab", args.ab_lut))
                          if v is not None},
        "label_overrides": {k: v for k, v in (("rna", args.rna_label),
                                              ("rna2", args.rna2_label),
                                              ("ab", args.ab_label)) if v is not None},
        "scalebar_um": _out.SCALEBAR_UM, "scalebar_font_px": _out.SCALEBAR_FONT_PX,
        "save_tifs": bool(args.tifs), "n_files": len(written),
    }, indent=2), encoding="utf-8")
    print("\n{} files -> {}".format(len(written), out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
