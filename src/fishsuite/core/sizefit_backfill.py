"""CPU backfill of the per-spot Gaussian size fit onto a finished run.

Reads a run's ``spot_metrics.csv``, re-opens each source image, re-extracts the
SAME z-plane the run recorded (``per_image_summary.z_plane``, the z-locked DAPI
autofocus index), and fits a 2-D Gaussian + constant background around every
spot on its own detection plane.

Writes THREE new files beside the run's tables and modifies NOTHING that is
already there:

  ``spot_metrics_sizefit.csv``   one row per spot: keys + the ``size_fit_*``
                                 columns (+ footprint um^2 where the run stored
                                 ``miat_footprint_area_px``)
  ``sizefit_per_nucleus.csv``    median fitted FWHM (um) / footprint area (um^2)
                                 per (image, channel, nucleus), nuclear spots
  ``sizefit_per_image.csv``      the same rollup per (image, channel)

plus ``sizefit_command.log`` and ``sizefit_versions.txt`` for provenance.

Size is invariant to a multiplicative rescale of the plane, so a run that
pedestal-normalised rna1 before detection needs no special handling here.
"""
from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .coloc_backfill import _resolve_plain_vsi
from .spot_size import (
    SIZE_FIT_COLUMNS,
    fit_spot_sizes,
    footprint_size_columns,
    size_rollup,
)

# spot_metrics channel label -> which resolved channel index carries its plane.
# rna_protein renames rna2 to "protein" on the way out; the plane is the same.
_LABEL_TO_SLOT = {"rna1": "rna", "rna2": "rna2", "protein": "rna2"}

_KEY_COLS = ["image", "condition", "channel", "spot_id", "nucleus_id",
             "in_nucleus", "x_px", "y_px", "z_slice"]


def _write_provenance(run_dir: Path, argv: List[str], params: Dict) -> None:
    (run_dir / "sizefit_command.log").write_text(
        "written_utc: {}\nargv: {}\nparams: {}\n".format(
            datetime.now(timezone.utc).isoformat(),
            " ".join(argv), json.dumps(params, default=str)),
        encoding="utf-8")
    lines = [
        f"written_utc: {datetime.now(timezone.utc).isoformat()}",
        f"python: {platform.python_version()}",
        f"python_executable: {sys.executable}",
        f"platform: {platform.platform()}",
        f"numpy: {np.__version__}",
        f"pandas: {pd.__version__}",
    ]
    try:
        from .. import __version__ as _v
        lines.insert(1, f"fishsuite_version: {_v}")
    except Exception:
        pass
    try:
        import scipy
        lines.append(f"scipy: {scipy.__version__}")
    except Exception:
        pass
    (run_dir / "sizefit_versions.txt").write_text("\n".join(lines) + "\n",
                                                  encoding="utf-8")


def sizefit_run(
    run_dir,
    input_dir=None,
    *,
    window_px: Optional[int] = None,
    voxel_xy_nm: Optional[float] = None,
    limit_images: Optional[int] = None,
    verbose: bool = True,
) -> Dict:
    from . import io as _io
    from ..config.schema import FishsuiteConfig
    from .modes.rna_rna import _resolve_channels

    run_dir = Path(run_dir)
    rc = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    cfg = FishsuiteConfig.model_validate(rc["config_resolved"])
    per_image = pd.read_csv(run_dir / "per_image_summary.csv")
    spots = pd.read_csv(run_dir / "spot_metrics.csv")

    win = int(window_px if window_px else getattr(cfg.foci, "size_fit_window_px", 7))
    src = Path(input_dir or rc.get("input_dir"))
    if not src.is_dir():
        raise FileNotFoundError(
            f"source image tree not readable: {src} — pass --input-dir")

    mode = getattr(cfg.channels, "analysis_mode", "")
    if mode == "rna_protein":
        from .modes.rna_protein import _build_rna2_shim_cfg
        chan_cfg = _build_rna2_shim_cfg(cfg)
    else:
        chan_cfg = cfg

    z_start_cfg = cfg.z_stack.start_slice
    z_end_cfg = cfg.z_stack.end_slice
    iw = bool(getattr(cfg.z_stack, "autofocus_intensity_weighted", False))

    rows: List[pd.DataFrame] = []
    n_ok_img = 0
    skipped: List[str] = []
    it = per_image.itertuples(index=False)
    for k, prow in enumerate(it):
        if limit_images is not None and k >= limit_images:
            break
        image_name = str(getattr(prow, "image"))
        sm = spots[spots["image"] == image_name]
        if len(sm) == 0:
            continue
        vsi = _resolve_plain_vsi(src, image_name)
        if vsi is None:
            skipped.append(f"{image_name}: source image not found under {src}")
            continue
        try:
            img = _io.read_image(vsi)
            idx_map = dict(zip(("dapi", "rna", "rna2"),
                               _resolve_channels(chan_cfg, img)))
            z_rec = getattr(prow, "z_plane", None)
            z_use = int(z_rec) if z_rec is not None and z_rec == z_rec else None
            if z_use is None:
                z_start, z_end = z_start_cfg, z_end_cfg
                if z_start is not None and z_start > img.n_z:
                    z_start = 1
                if z_end is not None and z_end > img.n_z:
                    z_end = img.n_z
                z_use, _ = _io.extract_channel_autofocus_with_idx(
                    img, idx_map["dapi"], z_start=z_start, z_end=z_end,
                    intensity_weighted=iw)
            vx_nm = voxel_xy_nm
            if vx_nm is None:
                vx_nm = getattr(prow, "voxel_xy_nm", None)
            if vx_nm is None or vx_nm != vx_nm or float(vx_nm) <= 0:
                vx_nm = float(img.voxel_xy_nm) if img.voxel_xy_nm == img.voxel_xy_nm \
                    and img.voxel_xy_nm > 0 else 65.0
            vx_um = float(vx_nm) / 1000.0

            planes: Dict[str, np.ndarray] = {}
            for label, grp in sm.groupby("channel"):
                slot = _LABEL_TO_SLOT.get(str(label))
                if slot is None:
                    skipped.append(f"{image_name}: unknown channel label {label!r}")
                    continue
                if slot not in planes:
                    planes[slot] = _io.extract_channel_at_z(
                        img, idx_map[slot], z_1indexed=int(z_use))
                g = grp.reset_index(drop=True)
                fit = fit_spot_sizes(planes[slot], g, vx_um, window_px=win)
                keep = [c for c in _KEY_COLS if c in g.columns]
                out = pd.concat([g[keep].reset_index(drop=True), fit], axis=1)
                out["voxel_xy_nm"] = float(vx_nm)
                out["z_plane"] = int(z_use)
                out["size_fit_window_px"] = int(win)
                if "miat_footprint_area_px" in g.columns:
                    fp = footprint_size_columns(
                        g["miat_footprint_area_px"].to_numpy(), vx_um)
                    out = pd.concat([out, fp], axis=1)
                rows.append(out)
            n_ok_img += 1
            if verbose:
                print(f"  [{k + 1}/{len(per_image)}] {image_name} z={z_use} "
                      f"spots={len(sm)}", flush=True)
        except Exception as exc:  # per-image failure must not kill the batch
            skipped.append(f"{image_name}: {type(exc).__name__}: {exc}")
            if verbose:
                print(f"  SKIP {image_name}: {type(exc).__name__}: {exc}", flush=True)

    if not rows:
        raise RuntimeError(f"no image produced a size fit; skips: {skipped[:5]}")

    allfit = pd.concat(rows, ignore_index=True)
    allfit.to_csv(run_dir / "spot_metrics_sizefit.csv", index=False)

    nuc_rows, img_rows = [], []
    for (im, ch), g in allfit.groupby(["image", "channel"], sort=True):
        cond = g["condition"].iloc[0] if "condition" in g.columns else ""
        r = {"image": im, "condition": cond, "channel": ch,
             "n_spots": int(len(g))}
        r.update(size_rollup(g, prefix="spot", nuclear_only=True))
        img_rows.append(r)
        if "nucleus_id" not in g.columns:
            continue
        for nid, gn in g.groupby("nucleus_id", sort=True):
            if int(nid) < 1:
                continue
            rn = {"image": im, "condition": cond, "channel": ch,
                  "nucleus_id": int(nid), "n_spots": int(len(gn))}
            rn.update(size_rollup(gn, prefix="spot", nuclear_only=True))
            nuc_rows.append(rn)

    pd.DataFrame(img_rows).to_csv(run_dir / "sizefit_per_image.csv", index=False)
    pd.DataFrame(nuc_rows).to_csv(run_dir / "sizefit_per_nucleus.csv", index=False)

    frac_ok = float((allfit["size_fit_ok"] == 1).mean()) if len(allfit) else float("nan")
    _write_provenance(run_dir, list(sys.argv),
                      {"window_px": win, "input_dir": str(src),
                       "images_fitted": n_ok_img, "spots": int(len(allfit)),
                       "frac_size_fit_ok": frac_ok, "skipped": skipped})
    return {"run_dir": str(run_dir), "images_fitted": n_ok_img,
            "spots": int(len(allfit)), "frac_size_fit_ok": frac_ok,
            "skipped": skipped}


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        description="Backfill the per-spot Gaussian size fit onto a finished run.")
    ap.add_argument("--run", required=True)
    ap.add_argument("--input-dir", default=None)
    ap.add_argument("--window-px", type=int, default=None)
    ap.add_argument("--limit-images", type=int, default=None)
    a = ap.parse_args(argv)
    res = sizefit_run(a.run, input_dir=a.input_dir, window_px=a.window_px,
                      limit_images=a.limit_images)
    print(json.dumps(res, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
