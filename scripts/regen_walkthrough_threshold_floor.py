#!/usr/bin/env python
"""Standalone re-render of the H9 MIAT-KD walkthrough THRESHOLD panels at the
spot floor (800), overwriting the misleading pixel-coloc-MAD-thresholded
step05/06/08 panels in two existing runs.

WHY: in rna_only mode the original pipeline rendered the step05/06/08
"threshold" panels from the pixel-coloc MAD threshold (rna_pos_mask), which is
much lower than the spot floor (e.g. ~290 vs 800). That made the panels look
wildly over-exposed and misrepresented the actual spot gate. The real gate is
output.manual_rna_min = 800 (apply_pub_contrast_floor_to_spots: true), confirmed
in the run logs:  "[floor-filter] ...: dropped N/M spots below floor=800.0".

This script reproduces step05/06 (MIAT >= 800 yellow-on-black + overlay) and
step08 (saved detected spots overlaid on the 800-gated background) and labels
every panel "MIAT-640 >= spot floor 800". It does NOT touch spot detection, the
floor filter, pixel_coloc, or any quantitative output (spot_metrics /
nuclei_metrics are read-only here).

Reproduction is exact:
  * raw MIAT channel read from the SAME .vsi via fishsuite.core.io.read_image
  * MAX-PROJECT over the SAME objective z-window the run used, parsed per-image
    from "<vsi>: focus peak at z=..., window=[a,b]" in <run>_run.log
    (window is 0-indexed inclusive; runner does sub[a:b+1].max(0))
  * threshold at 800 (pixels >= 800 yellow, matching step05 style)
  * step08 overlays the SAVED detected spots (per-image spot_metrics x_px/y_px),
    which are already floor-filtered to >=800 by the run.

Stem -> .vsi mapping is taken from the per-image spot_metrics 'image' column
(robust: run 2 has corrected/swapped condition labels in the stem, so the stem
prefix must NOT be trusted for condition).

Env: conda run -n fishproc_dml python regen_walkthrough_threshold_floor.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import fishsuite  # applies bffile compat patch on import
from fishsuite.core.io import read_image
from fishsuite.core.output import (
    save_png,
    burn_scale_bar,
    burn_corner_title,
    draw_spot_markers,
    apply_lut,
    _to_uint8,
    _resolve_lut_range,
    RNA_FLOOR_PCT,
    RNA_CEIL_PCT,
    SCALEBAR_UM,
    SCALEBAR_HEIGHT_PX,
    SCALEBAR_FONT_PX,
)

SPOT_FLOOR = 800.0
RNA_CH = 0  # MIAT = channel 0 (0-indexed) in both runs

RUNS = [
    {
        "name": "RERUN_IWFOCUS_floor800_no07",
        "run_dir": Path(r"F:\Image Analysis Work\H9-MIAT-KD\RERUN_IWFOCUS_floor800_no07_20260531-100336"),
        "log": Path(r"F:\Image Analysis Work\H9-MIAT-KD\RERUN_IWFOCUS_floor800_no07_20260531-100336_run.log"),
        "raw_root": Path(r"F:\Raw Images\H9-MIAT-KD-ASO\H9-MIAT-KD-05-08-2026"),
    },
    {
        "name": "RERUN_0505_corrected_labels_floor800",
        "run_dir": Path(r"F:\Image Analysis Work\H9-MIAT-KD\RERUN_0505_corrected_labels_floor800_20260531-112658"),
        "log": Path(r"F:\Image Analysis Work\H9-MIAT-KD\RERUN_0505_corrected_labels_floor800_20260531-112658_run.log"),
        "raw_root": Path(r"F:\Raw Images\H9-MIAT-KD-ASO\H9-MIAT-KD-ASO-05-05-2026\H9-MIAT-KD-ASO-05-05-2026"),
    },
]

_RNA_FN = "MIAT_640"           # matches the existing step05/06/08 filename token
_THR_TITLE = f"MIAT-640 ≥ spot floor {SPOT_FLOOR:g}"

_WIN_RE = re.compile(r"^\s*(?P<vsi>\S+\.vsi):\s*focus peak at z=\d+,\s*window=\[(?P<a>\d+),(?P<b>\d+)\]")


def parse_focus_windows(log_path: Path) -> dict[str, tuple[int, int]]:
    """vsi filename -> (a, b) 0-indexed inclusive z-window. Last line wins
    (logs can repeat a file; the final emitted window is the one used)."""
    wins: dict[str, tuple[int, int]] = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _WIN_RE.match(line)
        if m:
            wins[m.group("vsi")] = (int(m.group("a")), int(m.group("b")))
    return wins


def find_raw(raw_root: Path, vsi_name: str) -> Path | None:
    """Locate the exact .vsi under raw_root (recursive; handles flat or
    Folder_* layouts). Excludes the Olympus _<stem>_ data subfolders."""
    hits = [p for p in raw_root.rglob(vsi_name) if p.is_file() and p.suffix.lower() == ".vsi"]
    # prefer the shortest path (top-level file, not a nested copy)
    hits.sort(key=lambda p: len(p.parts))
    return hits[0] if hits else None


def stem_to_vsi(spot_metrics_csv: Path) -> str | None:
    """Read the 'image' column (the source .vsi) from a per-image spot_metrics."""
    try:
        df = pd.read_csv(spot_metrics_csv, usecols=["image"])
    except Exception:
        df = pd.read_csv(spot_metrics_csv)
        if "image" not in df.columns:
            return None
    vals = df["image"].dropna().unique()
    return str(vals[0]) if len(vals) else None


def render_one(run: dict, stem: str, vsi_name: str, win: tuple[int, int]) -> dict:
    """Re-render step05/06/08 for one image; overwrite in place. Returns info."""
    run_dir: Path = run["run_dir"]
    walk = run_dir / "pipeline_walkthrough"
    raw = find_raw(run["raw_root"], vsi_name)
    if raw is None:
        raise FileNotFoundError(f"raw vsi not found for {vsi_name} under {run['raw_root']}")

    img = read_image(raw)
    zyx = img.bio.get_image_data("ZYX", T=0, C=RNA_CH)
    if zyx.ndim == 2:
        zyx = zyx[None, :, :]
    nz = zyx.shape[0]
    a, b = win
    a = max(0, a)
    b = min(nz - 1, b)
    rna = zyx[a : b + 1].max(axis=0)
    vx = float(img.voxel_xy_nm) if img.voxel_xy_nm and not np.isnan(img.voxel_xy_nm) else 65.0

    # Contrast ceil for the step06 grayscale base: same percentile path the
    # walkthrough uses (no manual ceil override is in play for the gray base;
    # the floor is anchored to the spot floor per the code fix).
    rf, rc = _resolve_lut_range(rna, RNA_FLOOR_PCT, RNA_CEIL_PCT, batch_key=None, is_sec_only=False)

    pos = np.asarray(rna) >= SPOT_FLOOR

    # ---- Step 05: threshold mask yellow on black + title --------------------
    s05 = np.zeros((rna.shape[0], rna.shape[1], 3), dtype=np.uint8)
    s05[pos] = (255, 255, 0)
    s05 = burn_corner_title(s05, _THR_TITLE)
    p5 = walk / f"{stem}__step05_{_RNA_FN}_threshold_yellow.png"
    save_png(s05, p5)

    # ---- Step 06: threshold overlay on grayscale RNA (floor anchored to 800)
    base = _to_uint8(apply_lut(rna, 1.0, 1.0, 1.0, floor=SPOT_FLOOR * 0.75, ceil=rc))
    s06 = base.copy()
    s06[pos] = (0.5 * np.array([255, 255, 0], np.float32) + 0.5 * base[pos].astype(np.float32)).astype(np.uint8)
    s06 = burn_corner_title(s06, _THR_TITLE)
    s06 = burn_scale_bar(s06, vx, bar_um=SCALEBAR_UM, height_px=SCALEBAR_HEIGHT_PX, font_px=SCALEBAR_FONT_PX)
    p6 = walk / f"{stem}__step06_{_RNA_FN}_threshold_on_signal.png"
    save_png(s06, p6)

    # ---- Step 08: saved detected spots on the 800-gated background ----------
    n_spots = 0
    sm = run_dir / "per_image_csv" / f"{stem}__spot_metrics.csv"
    spots = None
    if sm.exists():
        spots = pd.read_csv(sm)
        # the saved spots are already floor-filtered (>=800) by the run
        if "x_px" in spots.columns and "y_px" in spots.columns and len(spots):
            n_spots = int(len(spots))
    p8 = walk / f"{stem}__step08_spots_on_{_RNA_FN}_threshold.png"
    if n_spots > 0:
        s08 = np.zeros((rna.shape[0], rna.shape[1], 3), dtype=np.uint8)
        s08[pos] = (200, 200, 0)
        s08 = draw_spot_markers(s08, spots, color=(255, 255, 255), radius=4, thickness=2,
                                size_mode="auto", voxel_xy_nm=vx)
        s08 = burn_corner_title(s08, _THR_TITLE)
        s08 = burn_scale_bar(s08, vx, bar_um=SCALEBAR_UM, height_px=SCALEBAR_HEIGHT_PX, font_px=SCALEBAR_FONT_PX)
        save_png(s08, p8)

    return {
        "stem": stem, "vsi": vsi_name, "win": (a, b), "nz": nz,
        "frac_pos": float(pos.mean()), "n_spots": n_spots,
        "rna_max": int(rna.max()), "rna_p999": int(np.percentile(rna, 99.9)),
        "step08_written": n_spots > 0,
    }


def main() -> int:
    for run in RUNS:
        print(f"\n=== {run['name']} ===")
        wins = parse_focus_windows(run["log"])
        walk = run["run_dir"] / "pipeline_walkthrough"
        # one stem per per-image spot_metrics csv
        sms = sorted((run["run_dir"] / "per_image_csv").glob("*__spot_metrics.csv"))
        print(f"  {len(sms)} images; {len(wins)} focus-window entries in log")
        for sm in sms:
            stem = sm.name.replace("__spot_metrics.csv", "")
            vsi = stem_to_vsi(sm)
            if vsi is None:
                print(f"  [SKIP] {stem}: no 'image' column"); continue
            if vsi not in wins:
                print(f"  [SKIP] {stem}: vsi {vsi} not in focus-window log"); continue
            # only render where the original walkthrough step05 exists (skip
            # sec-only / images without a walkthrough)
            if not (walk / f"{stem}__step05_{_RNA_FN}_threshold_yellow.png").exists():
                print(f"  [SKIP] {stem}: no existing step05 panel"); continue
            try:
                info = render_one(run, stem, vsi, wins[vsi])
                print(f"  [OK] {stem}: win{info['win']} nz={info['nz']} "
                      f"pos={info['frac_pos']*100:.2f}% spots={info['n_spots']} "
                      f"rnaMax={info['rna_max']} step08={'Y' if info['step08_written'] else 'n'}")
            except Exception as e:
                print(f"  [ERR] {stem}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
