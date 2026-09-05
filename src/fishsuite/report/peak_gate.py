"""Post-hoc peak-intensity gate on a finished run's spots.

fishsuite's spot-level floor gate runs AFTER detection, so applying the same
floor to a finished run's ``spot_metrics.csv`` and re-deriving the per-nucleus
counts reproduces a gated run exactly. This module does that, so a report can be
built at a floor the run itself did not apply, without re-detecting.

The aggregation rule is the engine's own, not a reimplementation of convenience:
the denominator of the nuclear fraction is ``n_in + n_cytoplasm``, NOT the number
of rows carrying that nucleus id. A spot flagged neither in-nucleus nor
in-cytoplasm is excluded from every count. Boundary is ``>=``, so a spot exactly
at the floor is KEPT, and a floor that is None, NaN or non-positive leaves that
channel untouched.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# In order of preference. Every current fishsuite build writes the first; all
# carry the same value, so which is found changes whether the comparison can be
# made, never its result.
PEAK_COLS = ("peak_intensity", "intensity_peak", "spot_peak_intensity",
             "rna_spot_mean_peak_intensity")

CHANNEL_ALIASES = {"rna": "rna1", "rna1": "rna1", "intron": "rna1",
                   "rna2": "rna2", "exon": "rna2",
                   "protein": "protein", "antibody": "protein", "ab": "protein"}


class PeakGateError(RuntimeError):
    """Raised when a peak floor cannot be applied to a run."""


def parse_peak_floors(spec: str) -> Dict[str, float]:
    """``"rna=1000,rna2=1200"`` -> ``{"rna1": 1000.0, "rna2": 1200.0}``."""
    out: Dict[str, float] = {}
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise PeakGateError(
                f"--peak-floor entry {part!r} is not CHANNEL=VALUE, for example "
                "rna=1000,rna2=1200")
        name, value = part.split("=", 1)
        key = CHANNEL_ALIASES.get(name.strip().lower())
        if key is None:
            raise PeakGateError(
                f"--peak-floor channel {name.strip()!r} is not one of "
                f"{sorted(set(CHANNEL_ALIASES))}")
        try:
            out[key] = float(value)
        except ValueError as exc:
            raise PeakGateError(
                f"--peak-floor value for {name.strip()!r} is not a number: "
                f"{value!r}") from exc
    return out


def resolve_peak_column(spots: pd.DataFrame) -> str:
    for c in PEAK_COLS:
        if c in spots.columns:
            return c
    raise PeakGateError(
        f"spot_metrics.csv carries none of {list(PEAK_COLS)}, so no peak floor "
        "can be applied to this run")


def gate_spots(spots: pd.DataFrame, floors: Dict[str, float]
               ) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Drop spots below their channel's floor. Returns the kept rows and a record."""
    if "channel" not in spots.columns:
        raise PeakGateError("spot_metrics.csv has no 'channel' column")
    record: Dict[str, object] = {"peak_column": "", "per_channel": {}}
    if not len(spots) or not floors:
        record["peak_column"] = "not needed; no floor requested"
        return spots.copy(), record
    peak_col = resolve_peak_column(spots)
    record["peak_column"] = peak_col
    chan = spots["channel"].astype(str)
    peaks = pd.to_numeric(spots[peak_col], errors="coerce")
    keep = pd.Series(True, index=spots.index)
    for ch, floor in floors.items():
        if floor is None:
            continue
        f = float(floor)
        if not np.isfinite(f) or f <= 0:
            record["per_channel"][ch] = {"floor": f, "applied": False,
                                         "reason": "floor is not a positive number"}
            continue
        in_ch = (chan == ch)
        before = int(in_ch.sum())
        # Boundary is >=: a spot exactly at the floor is KEPT.
        ok = in_ch & (peaks >= f)
        keep &= ~in_ch | (peaks >= f)
        record["per_channel"][ch] = {
            "floor": f, "applied": True, "spots_before": before,
            "spots_kept": int(ok.sum()), "spots_dropped": int(before - ok.sum()),
            "fraction_kept": float(ok.sum() / before) if before else float("nan")}
    return spots.loc[keep.values].reset_index(drop=True), record


def repair_pairing(gated: pd.DataFrame, pair_distance_um: float,
                   voxel_xy_um: float, voxel_z_um: float
                   ) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Recompute per-spot nearest-neighbour distance and the paired flag.

    A peak floor removes partner spots, so the run-time ``nn_distance_um`` and
    ``paired_at_0p3um`` columns answer a question about a spot set that no longer
    exists: a spot whose only partner was dropped still reads as paired. They are
    recomputed here against the SURVIVING partner set, per image, matching the
    engine's own rule (nearest neighbour in three dimensions with the run's voxel
    size, paired when the distance is at or below the pairing distance).

    Returns the frame with both columns rewritten and a record of what changed.
    """
    from scipy.spatial import cKDTree

    rec: Dict[str, object] = {"pair_distance_um": float(pair_distance_um),
                              "voxel_xy_um": float(voxel_xy_um),
                              "voxel_z_um": float(voxel_z_um),
                              "recomputed": False}
    need = {"image", "channel", "x_px", "y_px"}
    if gated is None or not len(gated) or not need <= set(gated.columns):
        rec["note"] = "no coordinate columns, so pairing was left as the run wrote it"
        return gated, rec
    if not (pair_distance_um and pair_distance_um > 0):
        rec["note"] = "no pairing distance recorded by the run"
        return gated, rec

    out = gated.reset_index(drop=True)
    z = (pd.to_numeric(out["z_slice"], errors="coerce").fillna(0.0)
         if "z_slice" in out.columns else pd.Series(0.0, index=out.index))
    coords = np.column_stack([
        pd.to_numeric(out["x_px"], errors="coerce").to_numpy() * float(voxel_xy_um),
        pd.to_numeric(out["y_px"], errors="coerce").to_numpy() * float(voxel_xy_um),
        z.to_numpy() * float(voxel_z_um)])
    nn = np.full(len(out), np.inf, dtype=float)
    channels = [c for c in out["channel"].astype(str).unique()]
    pos = np.arange(len(out))
    ch_all = out["channel"].astype(str).to_numpy()
    for _, idx in out.groupby(out["image"].astype(str), sort=False).indices.items():
        idx = np.asarray(idx)
        for ch in channels:
            same = ch_all[idx] == ch
            src, partner = idx[same], idx[~same]
            if not len(src) or not len(partner):
                continue                       # no partner: stays inf, so unpaired
            tree = cKDTree(coords[partner])
            d, _ = tree.query(coords[src], k=1)
            nn[src] = np.asarray(d, dtype=float)
    _ = pos

    before = None
    pair_col = next((c for c in out.columns if c.startswith("paired_at_")), None)
    if pair_col is not None:
        before = float(pd.to_numeric(out[pair_col], errors="coerce").mean())
    out["nn_distance_um"] = nn
    flag = (nn <= float(pair_distance_um)).astype(int)
    if pair_col is None:
        pair_col = f"paired_at_{str(pair_distance_um).replace('.', 'p')}um"
    out[pair_col] = flag
    rec.update(recomputed=True, pair_column=pair_col,
               paired_fraction_before=before,
               paired_fraction_after=float(flag.mean()),
               n_spots=int(len(out)))
    return out, rec


def _per_nucleus_counts(sub: pd.DataFrame, area_um2: pd.Series) -> pd.DataFrame:
    """Counts per (image, nucleus) from gated spots of ONE channel.

    Mirrors the engine's aggregation: the denominator is in-nucleus plus
    in-cytoplasm, so a spot flagged as neither is excluded from every count.
    """
    if not len(sub):
        return pd.DataFrame(columns=["image", "nucleus_id", "_tot", "_in", "_cy"])
    d = sub.copy()
    d["_in"] = d.get("in_nucleus", pd.Series(False, index=d.index)).astype(bool).astype(int)
    d["_cy"] = d.get("in_cytoplasm", pd.Series(False, index=d.index)).astype(bool).astype(int)
    g = (d.groupby(["image", "nucleus_id"], sort=False)[["_in", "_cy"]]
         .sum().reset_index())
    g["_tot"] = g["_in"] + g["_cy"]
    return g


def apply_to_nuclei(nuclei: pd.DataFrame, gated: pd.DataFrame,
                    voxel_xy_um: Optional[float]) -> Tuple[pd.DataFrame, List[str]]:
    """Recompute the count endpoints of ``nuclei`` from the GATED spots.

    Only the columns a peak floor actually changes are rewritten. Every other
    column is left exactly as the run wrote it, and the ones that a floor makes
    stale but that cannot be re-derived post hoc are listed in the return value so
    the report can say so rather than quietly carrying a pre-gate value.
    """
    out = nuclei.copy()
    if "nucleus_id" not in out.columns or "image" not in out.columns:
        raise PeakGateError("nuclei_metrics.csv needs image and nucleus_id columns")

    area_um2 = None
    if voxel_xy_um and "nucleus_area_px" in out.columns:
        area_um2 = pd.to_numeric(out["nucleus_area_px"], errors="coerce") * float(voxel_xy_um) ** 2

    for channel, cols in (
        ("rna1", {"total": "n_spots_rna1", "alt_total": "rna_spot_count",
                  "nuc": "nuclear_spot_count", "cyt": "cyto_spot_count",
                  "frac": "nuclear_spot_fraction",
                  "dens": "nuclear_spot_density_per_um2"}),
        ("rna2", {"total": "n_spots_rna2", "alt_total": None,
                  "nuc": "nuclear_spot_count_rna2", "cyt": "cyto_spot_count_rna2",
                  "frac": "nuclear_spot_fraction_rna2",
                  "dens": "nuclear_spot_density_per_um2_rna2"}),
        ("protein", {"total": "n_spots_protein", "alt_total": None,
                     "nuc": "nuclear_spot_count_protein",
                     "cyt": "cyto_spot_count_protein",
                     "frac": "nuclear_spot_fraction_protein",
                     "dens": "nuclear_spot_density_per_um2_protein"}),
    ):
        sub = gated[gated["channel"].astype(str) == channel]
        counts = _per_nucleus_counts(sub, area_um2)
        merged = out[["image", "nucleus_id"]].merge(
            counts, on=["image", "nucleus_id"], how="left")
        tot = merged["_tot"].fillna(0).to_numpy() if "_tot" in merged else np.zeros(len(out))
        n_in = merged["_in"].fillna(0).to_numpy() if "_in" in merged else np.zeros(len(out))
        n_cy = merged["_cy"].fillna(0).to_numpy() if "_cy" in merged else np.zeros(len(out))
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(tot > 0, n_in / np.where(tot > 0, tot, 1), np.nan)
        for key, values in (("total", tot), ("alt_total", tot), ("nuc", n_in),
                            ("cyt", n_cy), ("frac", frac)):
            col = cols.get(key)
            if col and col in out.columns:
                out[col] = values
        if cols.get("dens") and cols["dens"] in out.columns and area_um2 is not None:
            a = area_um2.to_numpy()
            with np.errstate(divide="ignore", invalid="ignore"):
                out[cols["dens"]] = np.where(a > 0, tot / np.where(a > 0, a, 1), np.nan)

    if area_um2 is not None:
        out["nucleus_area_um2"] = area_um2
        if "n_spots_rna1" in out.columns:
            a = area_um2.to_numpy()
            with np.errstate(divide="ignore", invalid="ignore"):
                out["rna1_spots_per_um2"] = np.where(
                    a > 0, pd.to_numeric(out["n_spots_rna1"], errors="coerce").to_numpy()
                    / np.where(a > 0, a, 1), np.nan)

    # Columns a peak floor makes stale but that cannot be re-derived from
    # spot_metrics.csv alone. Naming them is the point: a report must not carry a
    # pre-gate value as though the gate had been applied to it.
    stale = [c for c in (
        "nuclear_above_floor_intensity_rna1", "nuclear_above_floor_intensity_rna2",
        "nc_ratio_above_floor_intensity_rna1", "nc_ratio_above_floor_intensity_rna2",
        "rna_thresh_total_intensity_nuclear", "rna2_thresh_total_intensity_nuclear",
        "manders_rna1_in_rna2", "manders_rna2_in_rna1",
        "coloc_pearson_r_rna1_rna2", "protein_rotation_enrichment_at_rna1_spots",
        "protein_enrichment_vs_null_at_rna1_spots",
    ) if c in out.columns]
    return out, stale
