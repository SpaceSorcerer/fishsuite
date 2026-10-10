"""Per-nucleus shared-floor MIAT x QKI colocalization metrics from an exact-footprint backfill dir (2026-10-10).

Reads image_manifest.csv, selected_planes_and_masks.h5 (selected plane, nucleus and nucleolus labels),
spot_exact_footprint_metrics.csv.gz (the backfill's eligible native nuclear MIAT spots; per-image counts in
spot_eligibility_summary.csv) and footprint_pixels.csv.gz. Writes coloc_extra_metrics.csv, command.log and
versions.txt. Unit = nucleus; mask N = nucleus label minus its nucleolus label. Floors and pixel size are
passed in; nothing is estimated from the data.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fishsuite.core import coloc_pixel_metrics as cpm
from fishsuite.core import repro
from fishsuite.core.exact_footprint_figures import load_selected_plane

SPOTS, PIXELS = "spot_exact_footprint_metrics.csv.gz", "footprint_pixels.csv.gz"
MANIFEST, PLANES, ELIGIBILITY = "image_manifest.csv", "selected_planes_and_masks.h5", "spot_eligibility_summary.csv"
IDENTITY = ("image", "image_key", "condition", "slide", "arm", "replicate", "fov", "biological_set")
PIXEL_TOLERANCE = 0.01


def nucleus_rows(record, data, spots, pixels, *, miat_floor, qki_floor, pixel_um):
    miat, qki = data.planes["miat"], data.planes["qki"]
    labels, nucleoli = data.nucleus_labels, data.nucleolus_labels
    ident = {k: getattr(record, k) for k in IDENTITY if hasattr(record, k)}
    cy = np.rint(spots.center_y_px.to_numpy(float)).astype(int)
    cx = np.rint(spots.center_x_px.to_numpy(float)).astype(int)
    inside = (cy >= 0) & (cy < labels.shape[0]) & (cx >= 0) & (cx < labels.shape[1])
    if not inside.all():
        raise ValueError(f"spot centre outside the selected plane: {record.image_key}")
    parent = spots.nucleus_id.to_numpy(int)
    in_nucleoplasm = (labels[cy, cx] == parent) & (nucleoli[cy, cx] != parent)
    rows = []
    for nid in np.unique(labels[labels > 0]):
        mask = (labels == nid) & (nucleoli != nid)
        pick = (parent == nid) & in_nucleoplasm
        own = spots.loc[pick]
        fps = [pixels.get(uid) for uid in own.spot_uid]
        has_fp = np.array([fp is not None for fp in fps], dtype=bool)
        centres = np.column_stack((own.center_y_px.to_numpy(float), own.center_x_px.to_numpy(float)))[has_fp]
        row = dict(ident, nucleus_id=int(nid), n_pixels_nucleoplasm=int(mask.sum()),
                   n_spots_parent=int((parent == nid).sum()), n_spots_nucleoplasm=int(pick.sum()),
                   n_spots_without_footprint=int((~has_fp).sum()))
        row |= cpm.above_floor_pearson_icq(miat, qki, mask, miat_floor=miat_floor, qki_floor=qki_floor)
        row |= cpm.manders_shared_floor(miat, qki, mask, miat_floor=miat_floor, qki_floor=qki_floor)
        row |= cpm.object_coloc(qki, mask, centres, [fp for fp in fps if fp is not None],
                                qki_floor=qki_floor, pixel_um=pixel_um)
        row |= cpm.local_pearson_at_spots(miat, qki, mask, centres)
        rows.append(row | dict(miat_floor=miat_floor, qki_floor=qki_floor, pixel_um=pixel_um,
                               measurement_plane_z_1based=int(data.selected_z_1based)))
    return rows


def run(backfill: Path, out: Path, *, miat_floor, qki_floor, pixel_um) -> pd.DataFrame:
    manifest = pd.read_csv(backfill / MANIFEST)
    manifest["image_key"] = manifest.image_key.astype(str).str.casefold()
    spots = pd.read_csv(backfill / SPOTS)
    spots["image_key"] = spots.image_key.astype(str).str.casefold()
    pix = pd.read_csv(backfill / PIXELS, usecols=["spot_uid", "pixel_index", "y_px", "x_px"])
    pixels = {uid: (g.y_px.to_numpy(int), g.x_px.to_numpy(int))
              for uid, g in pix.sort_values(["spot_uid", "pixel_index"]).groupby("spot_uid", sort=False)}
    rows = []
    for record in manifest.sort_values("image_key").itertuples(index=False):
        recorded = float(record.voxel_xy_nm) / 1000.0
        if not abs(recorded - pixel_um) <= PIXEL_TOLERANCE * pixel_um:
            raise ValueError(f"--pixel-um {pixel_um:g} disagrees with manifest voxel_xy_nm "
                             f"{record.voxel_xy_nm} for {record.image_key}")
        channels = {c: int(getattr(record, f"{c}_channel_index")) for c in ("miat", "qki", "dapi")}
        data = load_selected_plane(backfill / PLANES, record.image_key,
                                   expected_z_1based=int(record.selected_z_1based), expected_channel_indices=channels)
        rows += nucleus_rows(record, data, spots.loc[spots.image_key == record.image_key], pixels,
                             miat_floor=miat_floor, qki_floor=qki_floor, pixel_um=pixel_um)
    table = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "coloc_extra_metrics.csv", index=False, lineterminator="\n")
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backfill", required=True, type=Path, help="exact-footprint backfill output dir")
    parser.add_argument("--miat-floor", required=True, type=float, help="absolute MIAT floor, raw a.u.")
    parser.add_argument("--qki-floor", required=True, type=float, help="absolute QKI floor, raw a.u.")
    parser.add_argument("--pixel-um", required=True, type=float, help="lateral pixel size, um (checked vs manifest)")
    parser.add_argument("--out", type=Path, default=None, help="default <backfill>/coloc_extra_metrics")
    parser.add_argument("--seed", type=int, default=0, help="recorded only; every metric is deterministic")
    args = parser.parse_args(argv)
    out = args.out or args.backfill / "coloc_extra_metrics"
    repro.set_global_seeds(args.seed)
    table = run(args.backfill, out, miat_floor=args.miat_floor, qki_floor=args.qki_floor, pixel_um=args.pixel_um)
    elig = args.backfill / ELIGIBILITY
    ok = repro.write_command_log(out, args.backfill, out, args.seed, extra={
        "floors": f"miat {args.miat_floor:g}, qki {args.qki_floor:g} raw a.u., fixed, shared by every image",
        "pixel_um": args.pixel_um,
        "spot_rule": f"rows of {SPOTS} (backfill-eligible native nuclear spots) whose rounded centre lies on "
                     "the parent nucleus label and off its nucleolus label in the saved masks",
        "eligibility_summary": (f"{ELIGIBILITY} n_eligible_spots total "
                                f"{int(pd.read_csv(elig).n_eligible_spots.sum())}" if elig.is_file() else "absent"),
        "object_rule": f"QKI objects = {cpm.OBJECT_CONNECTIVITY}-connected components of QKI > floor within N",
        "local_pearson": f"{cpm.LOCAL_WINDOW_PX}x{cpm.LOCAL_WINDOW_PX} px window within N, >= {cpm.LOCAL_MIN_PIXELS} px",
        "n_nuclei": len(table)})
    ok &= repro.write_versions_txt(out, args.seed)
    if not ok:
        raise OSError("provenance writer failed")
    print(f"{len(table)} nuclei -> {out / 'coloc_extra_metrics.csv'}")


if __name__ == "__main__":
    main()
