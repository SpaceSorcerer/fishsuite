"""pixel_pattern -- CPU per-nucleus PIXEL-LEVEL pattern metrics for a run.

A POST-RUN utility (sibling of ``coloc_backfill``): it REUSES a completed run's
saved nucleus masks and re-reads only the raw channel pixels (CPU; never
re-segments or touches the GPU), then computes per-nucleus PIXEL-pattern metrics
that need no spot calling:

  * PERINUCLEAR / RADIAL index -- mean intensity in the outer 25% depth band /
    inner 50% band (>1 = rim/edge-enriched), plus a 10-bin edge->center radial
    profile. Computed for RNA1, the partner (rna2/antibody) channel AND DAPI, and
    for a designated SECONDARY-ONLY control => a one-number stain QC: if the
    perinuclear rim is present in secondary-only (no primary antibody), it is
    technical, not biological.
  * GINI + TOP-5% / TOP-10% concentration per channel (0 = uniform, 1 = all
    signal in one pixel) over the nucleoplasm (nucleus minus nucleolus).
  * FOCI-BAND counts -- the number of partner spots per nucleus above a set of
    intensity floors (from the run's spot_metrics.csv), with a secondary-only
    specificity anchor.
  * DECILE intensity-sweep -- partner intensity across the RNA1 intensity range
    and vice versa (nucleoplasm pixels), per condition.

Reproduction contract (mirrors ``coloc_backfill``): the analysis z-plane is
RECOMPUTED deterministically from the raw file with the SAME intensity-weighted
autofocus + z-window the run used (``extract_channel_autofocus_with_idx``) and
every channel is read at that exact plane; the SAVED nucleus label mask is reused
so ``nucleus_id`` aligns; nucleoli are recomputed over those labels only when the
run used nucleolus exclusion. All resolution helpers are imported from
``coloc_backfill`` (single source of truth).

GENERIC: channels are resolved from the run config (RNA1 = primary FISH, partner
= rna2/antibody slot, DAPI); nothing is hardcoded to MIAT/QKI. rna_only runs
(no partner) still get RNA1 + DAPI metrics.

Note: the rotation "proper background" null is the DOCUMENTED CANONICAL coloc
statistic for this pipeline (see POSTRUN_UTILITIES.md); this module is
complementary pixel-pattern evidence, not a coloc replacement.

Outputs (under ``<run>/deliverables/pixelpattern/`` by default):
  * ``pixel_pattern_metrics.csv``  -- one row per nucleus
  * ``_radial_per_nucleus.csv`` / ``_sweep_per_nucleus.csv`` / foci-band table
  * ``pixel_pattern.xlsx``         -- explorable workbook
  * ``figures/*.png``              -- SuperPlots + radial/sweep + the stain-QC panel
  * ``PIXEL_PATTERN_FINDINGS.md``  -- plain-language readout incl. the stain QC

CLI::

    python -m fishsuite.core.pixel_pattern --run-dir <run> [--staging <raw>]
        [--secondary-match well12] [--no-figures] [--no-excel]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Reuse the backfill resolution helpers -- SINGLE SOURCE OF TRUTH for how a
# completed run's raw VSI / saved masks / nucleoli are located + read.
from .coloc_backfill import (
    _resolve_plain_vsi,
    _resolve_mask_path,
    _read_label_tiff,
    _build_nucleolus_labels,
)

MIN_NP_PIX = 500          # min nucleoplasm pixels to trust the metrics
N_DECILE = 10             # intensity-sweep bins
DEFAULT_FOCI_FLOORS = [0, 5000, 8000, 10000, 12000, 15000]


# ===========================================================================
# PURE metric helpers (operate on numpy arrays)
# ===========================================================================
def gini(x) -> float:
    """Gini coefficient of nonnegative values (0=uniform, 1=all-in-one-pixel)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    if np.any(x < 0):
        x = x - x.min()
    s = x.sum()
    if s <= 0:
        return float("nan")
    xs = np.sort(x)
    n = xs.size
    idx = np.arange(1, n + 1)
    return float((2.0 * np.sum(idx * xs)) / (n * s) - (n + 1.0) / n)


def top_frac(x, frac: float) -> float:
    """Fraction of total signal held by the brightest ``frac`` of pixels."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    s = x.sum()
    if s <= 0 or x.size == 0:
        return float("nan")
    k = max(1, int(round(x.size * frac)))
    thr = np.partition(x, x.size - k)[x.size - k]
    return float(x[x >= thr].sum() / s)


def _radial_profile(vb: np.ndarray, dvals: np.ndarray) -> np.ndarray:
    """Mean of bg-subtracted values ``vb`` in 10 depth bins (0=edge,1=center)."""
    bins = np.linspace(0, 1, 11)
    bidx = np.clip(np.digitize(dvals, bins) - 1, 0, 9)
    prof = np.full(10, np.nan)
    for b in range(10):
        sel = bidx == b
        if sel.sum() > 0:
            prof[b] = vb[sel].mean()
    return prof


def _perinuc_index(vb: np.ndarray, dvals: np.ndarray) -> float:
    edge = vb[dvals < 0.25]
    cen = vb[dvals >= 0.5]
    if cen.size and edge.size and cen.mean() > 0:
        return float(edge.mean() / cen.mean())
    return float("nan")


# ===========================================================================
# PER-IMAGE processing (raw + saved mask; reuses backfill resolution)
# ===========================================================================
def _resolve_pp_channels(cfg, img):
    """Return (dapi_idx, rna_idx, partner_idx-or-None) using the rna_protein
    antibody->rna2 shim so the partner index is correct in every mode."""
    mode = getattr(cfg.channels, "analysis_mode", "")
    if mode == "rna_protein":
        from .modes.rna_protein import _build_rna2_shim_cfg
        chan_cfg = _build_rna2_shim_cfg(cfg)
    else:
        chan_cfg = cfg
    from .modes.rna_rna import _resolve_channels
    dapi_idx, rna_idx, rna2_idx = _resolve_channels(chan_cfg, img)
    partner = rna2_idx if (rna2_idx is not None and rna2_idx >= 0) else None
    return dapi_idx, rna_idx, partner


def compute_pixel_pattern_for_image(
    chans: Dict[str, np.ndarray],
    labels: np.ndarray,
    nucleolus_labels: Optional[np.ndarray],
    partner_spots_by_nid: Dict[int, np.ndarray],
    *,
    pixel_um: float,
    image: str = "",
    condition: str = "",
    cond2: str = "",
) -> Tuple[List[dict], List[dict], List[dict]]:
    """Return (rows, radial_records, sweep_records) for one image.

    ``chans`` maps role -> 2D array for the roles present ("rna1", "partner",
    "dapi"). ``partner_spots_by_nid`` maps nucleus_id -> (n,) peak-intensity array
    of partner spots (for the decile context; foci-band is computed separately).
    """
    from scipy import ndimage as ndi
    from scipy.stats import pearsonr, spearmanr

    roles = list(chans.keys())
    ref = chans["rna1"] if "rna1" in chans else chans[roles[0]]
    if any(chans[r].shape != ref.shape for r in roles) or ref.shape != labels.shape:
        return [], [], []

    bg = {r: float(np.percentile(chans[r], 5)) for r in roles}
    slices = ndi.find_objects(labels)
    uniq = np.unique(labels)
    uniq = uniq[uniq > 0]
    rows, radial_records, sweep_records = [], [], []

    for L in uniq:
        sl = slices[L - 1]
        if sl is None:
            continue
        sub_lab = labels[sl]
        nucmask = sub_lab == L
        # nucleolus of THIS nucleus within the crop (label image carries parent
        # nucleus id; a boolean mask is also accepted -- mirrors the engine).
        if nucleolus_labels is not None:
            nl_crop = np.asarray(nucleolus_labels)[sl]
            nucleomask = (nl_crop & nucmask) if nl_crop.dtype == bool else (nl_crop == L)
        else:
            nucleomask = np.zeros_like(nucmask, dtype=bool)
        npmask = nucmask & (~nucleomask)
        if npmask.sum() < MIN_NP_PIX:
            npmask = nucmask
        edt = ndi.distance_transform_edt(nucmask)
        dmax = edt.max() if edt.max() > 0 else 1.0
        depth = edt / dmax
        dvals = depth[npmask]

        row = dict(image=image, condition=condition, cond2=cond2, well=condition,
                   nucleus_id=int(L), np_pix=int(npmask.sum()),
                   nucleolus_pix=int(nucleomask.sum()))
        for role in roles:
            crop = chans[role][sl]
            v_np = crop[npmask].astype(float)
            vb = np.clip(v_np - bg[role], 0, None)
            row[f"{role}_mean"] = float(v_np.mean())
            row[f"{role}_gini"] = gini(vb)
            row[f"{role}_gini_raw"] = gini(v_np)
            row[f"{role}_cv"] = float(v_np.std() / v_np.mean()) if v_np.mean() > 0 else float("nan")
            row[f"{role}_top5pct_frac"] = top_frac(vb, 0.05)
            row[f"{role}_top10pct_frac"] = top_frac(vb, 0.10)
            row[f"{role}_perinuc_index"] = _perinuc_index(vb, dvals)
            prof = _radial_profile(vb, dvals)
            radial_records.append(dict(image=image, condition=condition, cond2=cond2,
                                       nucleus_id=int(L), channel=role,
                                       **{f"d{b}": prof[b] for b in range(10)}))
        # RNA1 x partner pixel relationship + decile sweep (nucleoplasm, bg-sub)
        if "rna1" in chans and "partner" in chans:
            m = np.clip(chans["rna1"][sl][npmask] - bg["rna1"], 0, None)
            q = np.clip(chans["partner"][sl][npmask] - bg["partner"], 0, None)
            if q.size > 20 and q.std() > 0 and m.std() > 0:
                row["pearson_rna1_partner_allpix"] = float(pearsonr(q, m)[0])
                row["spearman_rna1_partner_allpix"] = float(spearmanr(q, m)[0])
                try:
                    qr = pd.qcut(q, N_DECILE, labels=False, duplicates="drop")
                    mr = pd.qcut(m, N_DECILE, labels=False, duplicates="drop")
                    partner_at_m = [float(q[mr == b].mean()) if np.any(mr == b) else np.nan for b in range(N_DECILE)]
                    rna1_at_q = [float(m[qr == b].mean()) if np.any(qr == b) else np.nan for b in range(N_DECILE)]
                    sweep_records.append(dict(image=image, condition=condition, cond2=cond2, nucleus_id=int(L),
                                              **{f"partner_at_rna1_d{b}": partner_at_m[b] for b in range(N_DECILE)},
                                              **{f"rna1_at_partner_d{b}": rna1_at_q[b] for b in range(N_DECILE)}))
                except Exception:
                    pass
            else:
                row["pearson_rna1_partner_allpix"] = float("nan")
                row["spearman_rna1_partner_allpix"] = float("nan")
        rows.append(row)
    return rows, radial_records, sweep_records


# ===========================================================================
# I/O SHELL
# ===========================================================================
def _cond2(condition: str) -> str:
    """Coarse group label: control-like -> its label; else 'perturbation';
    secondary -> 'Sec'. Only used for grouping/plot color."""
    from ._superplot import is_control_like
    c = str(condition)
    if "sec" in c.lower():
        return "Sec"
    return c.split("_")[0].split("-")[0]


def pixelpattern_run(
    run_dir,
    staging_dir=None,
    input_dir=None,
    *,
    secondary_match: Optional[str] = None,
    foci_floors: Optional[List[float]] = None,
    do_excel: bool = True,
    do_figures: bool = True,
    out_subdir: str = "pixelpattern",
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """Compute per-nucleus pixel-pattern metrics on a completed run.

    Loops images from ``per_image_summary.csv``; resolves the raw VSI under
    ``staging_dir`` (or the run's recorded ``input_dir``); reads DAPI/RNA1/partner
    at the recomputed analysis z-plane; reuses the saved nucleus labels;
    recomputes nucleoli; computes the pixel-pattern metrics; appends the
    foci-band counts; writes the tables + Excel + figures.
    """
    from . import io as _io

    run_dir = Path(run_dir)
    rc_path = run_dir / "run_config.json"
    if not rc_path.exists():
        raise FileNotFoundError(f"no run_config.json in {run_dir}")
    rc = json.loads(rc_path.read_text(encoding="utf-8"))
    from ..config.schema import FishsuiteConfig
    cfg = FishsuiteConfig.model_validate(rc["config_resolved"])

    pi_path = run_dir / "per_image_summary.csv"
    nm_path = run_dir / "nuclei_metrics.csv"
    sm_path = run_dir / "spot_metrics.csv"
    if not pi_path.exists() or not nm_path.exists():
        raise FileNotFoundError(
            f"run_dir missing per_image_summary.csv or nuclei_metrics.csv: {run_dir}")
    per_image = pd.read_csv(pi_path)
    nuc = pd.read_csv(nm_path)
    masks_dir = run_dir / "masks"
    if not masks_dir.is_dir():
        raise FileNotFoundError(f"no saved nuclei masks dir found: {masks_dir}")

    # partner spot table (peak intensities per nucleus) for the foci-band metric
    partner_channel = None
    spot_metrics = None
    if sm_path.exists():
        spot_metrics = pd.read_csv(
            sm_path, usecols=lambda c: c in (
                "image", "channel", "nucleus_id", "in_nucleus",
                "peak_intensity", "spot_peak_intensity"))
        chans_present = set(spot_metrics["channel"].unique())
        partner_channel = "protein" if "protein" in chans_present else (
            "rna2" if "rna2" in chans_present else None)

    src = staging_dir or input_dir or rc.get("input_dir")
    if src is None:
        raise ValueError("no staging_dir / input_dir given and run_config has no input_dir")
    src = Path(src)

    z_start_cfg = cfg.z_stack.start_slice
    z_end_cfg = cfg.z_stack.end_slice
    iw = bool(getattr(cfg.z_stack, "autofocus_intensity_weighted", False))

    all_rows: List[dict] = []
    all_radial: List[dict] = []
    all_sweep: List[dict] = []
    n_ok = 0

    for _, prow in per_image.iterrows():
        image_name = str(prow["image"])
        condition = str(prow.get("condition", ""))
        try:
            vsi = _resolve_plain_vsi(src, image_name)
            if vsi is None:
                print(f"  SKIP {image_name}: raw VSI not found under {src}")
                continue
            mask_path = _resolve_mask_path(masks_dir, image_name, condition)
            if mask_path is None:
                print(f"  SKIP {image_name}: nuclei_label_mask not found in {masks_dir}")
                continue
            img = _io.read_image(vsi)
            dapi_idx, rna_idx, partner_idx = _resolve_pp_channels(cfg, img)

            z_start, z_end = z_start_cfg, z_end_cfg
            if z_start is not None and z_start > img.n_z:
                z_start = 1
            if z_end is not None and z_end > img.n_z:
                z_end = img.n_z
            dapi_z, dapi_2d = _io.extract_channel_autofocus_with_idx(
                img, dapi_idx, z_start=z_start, z_end=z_end, intensity_weighted=iw)
            chans: Dict[str, np.ndarray] = {"dapi": dapi_2d.astype(np.float32)}
            if rna_idx is not None and rna_idx >= 0:
                chans["rna1"] = _io.extract_channel_at_z(img, rna_idx, z_1indexed=dapi_z).astype(np.float32)
            if partner_idx is not None:
                chans["partner"] = _io.extract_channel_at_z(img, partner_idx, z_1indexed=dapi_z).astype(np.float32)

            labels = _read_label_tiff(mask_path)
            ref_shape = chans.get("rna1", chans["dapi"]).shape
            if labels.shape != ref_shape:
                print(f"  WARN {image_name}: mask shape {labels.shape} != {ref_shape}; skip")
                continue
            voxel_xy_nm = float(img.voxel_xy_nm) if (img.voxel_xy_nm == img.voxel_xy_nm and img.voxel_xy_nm > 0) else 130.0
            pixel_um = voxel_xy_nm / 1000.0
            nucleolus_labels = _build_nucleolus_labels(cfg, labels, dapi_2d, voxel_xy_nm)

            c2 = _cond2(condition)
            rows, rad, swe = compute_pixel_pattern_for_image(
                chans, labels, nucleolus_labels, {},
                pixel_um=pixel_um, image=image_name, condition=condition, cond2=c2)
            all_rows.extend(rows)
            all_radial.extend(rad)
            all_sweep.extend(swe)
            n_ok += 1
            if verbose:
                print(f"  OK {image_name}: {len(rows)} nuclei")
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR {image_name}: {type(exc).__name__}: {exc}")
            continue

    if not all_rows:
        raise RuntimeError("no nuclei processed -- check --staging points at the raw images")

    df = pd.DataFrame(all_rows)
    # join per-nucleus abundance (for the dose-response context in the Excel)
    keep = [k for k in ("image", "nucleus_id", "nuclear_spot_count", "rna_spot_count",
                        "nuclear_total_intensity_rna1", "sum_rna_intensity") if k in nuc.columns]
    if keep:
        df = df.merge(nuc[keep].drop_duplicates(["image", "nucleus_id"]),
                      on=["image", "nucleus_id"], how="left")

    # ---- foci-band counts (partner spots per nucleus at intensity floors) ----
    floors = list(foci_floors) if foci_floors else DEFAULT_FOCI_FLOORS
    foci_band = None
    if spot_metrics is not None and partner_channel is not None:
        pk_col = "peak_intensity" if "peak_intensity" in spot_metrics.columns else "spot_peak_intensity"
        ps = spot_metrics[(spot_metrics["channel"] == partner_channel)
                          & (spot_metrics.get("in_nucleus", 1) == 1)].copy()
        ps[pk_col] = pd.to_numeric(ps[pk_col], errors="coerce")
        for fl in floors:
            fc = ps[ps[pk_col] >= fl].groupby(["image", "nucleus_id"]).size().rename(f"partner_foci_ge{int(fl)}").reset_index()
            df = df.merge(fc, on=["image", "nucleus_id"], how="left")
            df[f"partner_foci_ge{int(fl)}"] = df[f"partner_foci_ge{int(fl)}"].fillna(0).astype(int)
        # per-group summary + secondary anchor
        fb_rows = []
        for fl in floors:
            col = f"partner_foci_ge{int(fl)}"
            grp = df.groupby("cond2")[col].mean()
            fb_rows.append({"floor": fl, **{f"{g}_foci_per_nuc": round(float(grp.get(g, np.nan)), 3) for g in grp.index}})
        foci_band = pd.DataFrame(fb_rows)

    # ---- secondary-only control designation (configurable) ----
    is_sec = df["cond2"] == "Sec"
    if secondary_match:
        sec_mask = is_sec & (df["image"].astype(str).str.contains(secondary_match, case=False)
                             | df["condition"].astype(str).str.contains(secondary_match, case=False))
    else:
        sec_mask = is_sec
    df["is_secondary_control"] = sec_mask

    out_dir = run_dir / "deliverables" / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, str] = {}
    metrics_csv = out_dir / "pixel_pattern_metrics.csv"
    df.to_csv(metrics_csv, index=False)
    written["metrics"] = str(metrics_csv)
    rad_df = pd.DataFrame(all_radial)
    rad_df.to_csv(out_dir / "_radial_per_nucleus.csv", index=False)
    swe_df = pd.DataFrame(all_sweep)
    swe_df.to_csv(out_dir / "_sweep_per_nucleus.csv", index=False)
    if foci_band is not None:
        foci_band.to_csv(out_dir / "foci_band_summary.csv", index=False)
        written["foci_band"] = str(out_dir / "foci_band_summary.csv")

    # ---- NT-vs-condition Welch on well means ----
    welch = _welch_table(df)
    welch.to_csv(out_dir / "nt_vs_condition_wellmeans_welch.csv", index=False)
    written["welch"] = str(out_dir / "nt_vs_condition_wellmeans_welch.csv")

    figs: List[str] = []
    if do_figures:
        figs = _make_figures(df, rad_df, swe_df, welch, out_dir / "figures",
                             chan_labels=_chan_labels(rc), secondary_match=secondary_match)
        written["n_figures"] = str(len(figs))
    if do_excel:
        xlsx = out_dir / "pixel_pattern.xlsx"
        _write_excel(df, welch, rad_df, swe_df, foci_band, xlsx, chan_labels=_chan_labels(rc))
        written["excel"] = str(xlsx)

    stainqc = _stain_qc(df, secondary_match)
    _write_findings(df, welch, stainqc, out_dir / "PIXEL_PATTERN_FINDINGS.md", figs)
    written["findings"] = str(out_dir / "PIXEL_PATTERN_FINDINGS.md")

    if verbose:
        vc = df["cond2"].value_counts().to_dict()
        print(f"[pixelpattern] {n_ok} images, {len(df)} nuclei {vc} | {len(figs)} figures")
        if stainqc:
            print(f"[pixelpattern] STAIN-QC partner perinuclear index: {stainqc}")
    return {"out_dir": str(out_dir), "written": written, "n_images": n_ok,
            "n_nuclei": int(len(df)), "n_figures": len(figs), "stain_qc": stainqc}


def _chan_labels(rc: dict) -> Dict[str, str]:
    ch = (rc.get("config_resolved", {}) or {}).get("channels", {}) or {}
    return {"rna1": ch.get("rna_label") or "RNA1",
            "partner": ch.get("antibody_label") or ch.get("rna2_label") or "Partner",
            "dapi": ch.get("dapi_label") or "DAPI"}


def _welch_table(df: pd.DataFrame) -> pd.DataFrame:
    """Well-mean Welch t control vs perturbation for every numeric metric."""
    from scipy.stats import ttest_ind
    from ._superplot import is_control_like
    groups = [g for g in df["cond2"].unique() if g != "Sec"]
    ctrl = [g for g in groups if is_control_like(g)]
    pert = [g for g in groups if not is_control_like(g)]
    gA = ctrl[0] if ctrl else (groups[0] if groups else None)
    gB = pert[0] if pert else (groups[1] if len(groups) > 1 else None)
    skip = {"nucleus_id", "np_pix", "nucleolus_pix"}
    metric_cols = [c for c in df.columns
                   if c not in skip and pd.api.types.is_numeric_dtype(df[c])
                   and df[c].notna().sum() > 4 and c not in ("image",)]
    rows = []
    for col in metric_cols:
        wm = df[df["cond2"].isin([gA, gB])].groupby(["cond2", "well"])[col].mean().reset_index()
        a = wm[wm.cond2 == gA][col].dropna().values if gA else np.array([])
        b = wm[wm.cond2 == gB][col].dropna().values if gB else np.array([])
        if len(a) >= 2 and len(b) >= 2:
            t, p = ttest_ind(a, b, equal_var=False)
        else:
            t, p = np.nan, np.nan
        rows.append({"metric": col, "A_group": gA, "B_group": gB,
                     "A_mean": float(np.mean(a)) if len(a) else np.nan,
                     "B_mean": float(np.mean(b)) if len(b) else np.nan,
                     "A_n_wells": len(a), "B_n_wells": len(b),
                     "B_minus_A": (float(np.mean(b) - np.mean(a)) if len(a) and len(b) else np.nan),
                     "pct_change": (float(100 * (np.mean(b) - np.mean(a)) / np.mean(a))
                                    if len(a) and len(b) and np.mean(a) else np.nan),
                     "welch_t": float(t) if t == t else np.nan,
                     "welch_p": float(p) if p == p else np.nan})
    return pd.DataFrame(rows)


def _stain_qc(df: pd.DataFrame, secondary_match: Optional[str]) -> Dict[str, Any]:
    """Partner perinuclear index for control / perturbation / secondary-only ->
    a one-number stain sanity check."""
    if "partner_perinuc_index" not in df.columns:
        return {}
    out: Dict[str, Any] = {}
    for g in df["cond2"].unique():
        sub = df[df["cond2"] == g]
        out[str(g)] = round(float(sub["partner_perinuc_index"].mean()), 3)
    sec = df[df["is_secondary_control"]] if "is_secondary_control" in df.columns else df[df["cond2"] == "Sec"]
    if len(sec):
        out["secondary_control"] = round(float(sec["partner_perinuc_index"].mean()), 3)
        real = df[df["cond2"] != "Sec"]["partner_perinuc_index"].mean()
        if real and out["secondary_control"] >= 0.9 * real:
            out["flag"] = ("perinuclear rim present in secondary-only "
                           "(>=90% of the antibody rim) -> likely technical, not specific")
        else:
            out["flag"] = "secondary-only rim well below antibody rim -> perinuclear signal looks specific"
    return out


# ===========================================================================
# Figures
# ===========================================================================
def _make_figures(df, rad, swe, welch, fig_dir: Path, *, chan_labels: Dict[str, str],
                  secondary_match: Optional[str]) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ._superplot import (superplot_into_axes, OKABE_ITO,
                             order_conditions_control_first)

    fig_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    dfk = df[df["cond2"] != "Sec"].copy()
    conds = order_conditions_control_first([g for g in dfk["cond2"].unique()])
    p_lab = chan_labels.get("partner", "Partner")

    # ---- SuperPlots of the key pixel metrics ----
    sp_specs = [
        ("partner_gini", f"{p_lab} nucleoplasm Gini", False),
        ("partner_top5pct_frac", f"{p_lab} signal in top-5% pixels", False),
        ("partner_perinuc_index", f"{p_lab} edge/center ratio", False),
        ("partner_cv", f"{p_lab} intensity CV", False),
        ("pearson_rna1_partner_allpix", f"RNA1 x {p_lab} pixel Pearson r", False),
        ("rna1_gini", "RNA1 nucleoplasm Gini", False),
    ]
    i = 0
    for col, yl, pct in sp_specs:
        if col not in dfk.columns:
            continue
        sub = dfk[["cond2", "well", col]].dropna().rename(columns={"cond2": "condition", "well": "image"})
        if sub.empty:
            continue
        i += 1
        fig, ax = plt.subplots(figsize=(4.6, 5.0))
        ok = superplot_into_axes(ax, sub, col, ylabel=yl, unit="nucleus",
                                 condition_order=conds, pct=pct)
        wp = welch[welch.metric == col]["welch_p"]
        title = yl + (f"\nWelch p={float(wp.values[0]):.3g} (well means)" if len(wp) and wp.values[0] == wp.values[0] else "")
        ax.set_title(title, fontsize=8.5)
        fig.tight_layout()
        p = fig_dir / f"sp{i:02d}_{col}.png"
        fig.savefig(p, dpi=600, bbox_inches="tight")
        plt.close(fig)
        if ok:
            written.append(str(p))

    # ---- STAIN-QC panel: partner perinuclear index NT/KD/secondary-only ----
    if "partner_perinuc_index" in df.columns:
        sub3 = df[["cond2", "well", "partner_perinuc_index"]].dropna().copy()
        sub3["cond2"] = sub3["cond2"].replace({"Sec": "Secondary-only"})
        sub3 = sub3.rename(columns={"cond2": "condition", "well": "image"})
        order3 = order_conditions_control_first([c for c in sub3["condition"].unique() if c != "Secondary-only"]) + ["Secondary-only"]
        fig, ax = plt.subplots(figsize=(5.2, 5.2))
        superplot_into_axes(ax, sub3, "partner_perinuc_index",
                            ylabel=f"{p_lab} edge/center ratio (perinuclear index)",
                            unit="nucleus", condition_order=order3)
        ax.axhline(1.0, color="#888", lw=0.8, ls=":")
        ax.set_title(f"STAIN QC: {p_lab} perinuclear index vs secondary-only", fontsize=9)
        fig.tight_layout()
        p = fig_dir / "stainqc_partner_perinuc_index.png"
        fig.savefig(p, dpi=600, bbox_inches="tight")
        plt.close(fig)
        written.append(str(p))

    # ---- radial profiles (partner NT vs KD vs secondary-only) ----
    if not rad.empty:
        DBINS = [f"d{b}" for b in range(10)]
        xdepth = (np.arange(10) + 0.5) / 10.0

        def norm_profile(block):
            P = block[DBINS].values.astype(float)
            mu = np.nanmean(P, axis=1, keepdims=True)
            mu[mu == 0] = np.nan
            return P / mu

        ccol = {}
        for k, c in enumerate(order_conditions_control_first(list(rad["cond2"].unique()))):
            ccol[c] = OKABE_ITO[min(k, len(OKABE_ITO) - 1)]
        fig, ax = plt.subplots(figsize=(5.8, 4.6))
        for c in rad["cond2"].unique():
            b = rad[(rad.channel == "partner") & (rad.cond2 == c)]
            if b.empty:
                continue
            Pn = norm_profile(b)
            m = np.nanmean(Pn, axis=0)
            se = np.nanstd(Pn, axis=0) / np.sqrt(np.maximum(1, np.sum(~np.isnan(Pn), axis=0)))
            lbl = "Secondary-only" if c == "Sec" else c
            ax.plot(xdepth, m, "-o", color=ccol.get(c, "#666"), label=f"{lbl} (n={len(b)})", ms=4)
            ax.fill_between(xdepth, m - se, m + se, color=ccol.get(c, "#666"), alpha=0.18)
        ax.axhline(1.0, color="#888", lw=0.8, ls=":")
        ax.set_xlabel("normalized depth (0 = nuclear edge  ->  1 = center)")
        ax.set_ylabel(f"relative {p_lab} intensity (/ nucleus mean)")
        ax.set_title(f"{p_lab} radial profile: specific or present in secondary-only?")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25, ls="--")
        fig.tight_layout()
        p = fig_dir / "radial_partner_with_seconly.png"
        fig.savefig(p, dpi=600, bbox_inches="tight")
        plt.close(fig)
        written.append(str(p))

    # ---- decile intensity sweep ----
    if not swe.empty:
        ccol = {}
        for k, c in enumerate(order_conditions_control_first([g for g in swe["cond2"].unique() if g != "Sec"])):
            ccol[c] = OKABE_ITO[min(k, len(OKABE_ITO) - 1)]
        for prefix, xlabel, title, fname in (
            ("partner_at_rna1", "RNA1 intensity decile (1=dim -> 10=bright)",
             f"{p_lab} across the RNA1 intensity range", "sweep_partner_at_rna1.png"),
            ("rna1_at_partner", f"{p_lab} intensity decile (1=dim -> 10=bright)",
             f"RNA1 across the {p_lab} intensity range", "sweep_rna1_at_partner.png")):
            cols = [f"{prefix}_d{b}" for b in range(10)]
            if not set(cols).issubset(swe.columns):
                continue
            fig, ax = plt.subplots(figsize=(5.4, 4.6))
            xd = np.arange(1, 11)
            for c in ccol:
                b = swe[swe.cond2 == c]
                if b.empty:
                    continue
                P = b[cols].values.astype(float)
                mu = np.nanmean(P, axis=1, keepdims=True)
                mu[mu == 0] = np.nan
                Pn = P / mu
                m = np.nanmean(Pn, axis=0)
                se = np.nanstd(Pn, axis=0) / np.sqrt(np.maximum(1, np.sum(~np.isnan(Pn), axis=0)))
                ax.plot(xd, m, "-o", color=ccol[c], label=f"{c} (n={len(b)})", ms=4)
                ax.fill_between(xd, m - se, m + se, color=ccol[c], alpha=0.18)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("relative partner intensity (/ nucleus mean)")
            ax.set_title(title)
            ax.legend()
            ax.grid(alpha=0.25, ls="--")
            fig.tight_layout()
            p = fig_dir / fname
            fig.savefig(p, dpi=600, bbox_inches="tight")
            plt.close(fig)
            written.append(str(p))
    return written


# ===========================================================================
# Excel + findings
# ===========================================================================
def _write_excel(df, welch, rad, swe, foci_band, xlsx: Path, *, chan_labels):
    from openpyxl import load_workbook
    DBINS = [f"d{b}" for b in range(10)]
    p_lab = chan_labels.get("partner", "Partner")
    readme = pd.DataFrame({
        "Sheet": ["per_nucleus", "nt_vs_condition_welch", "radial_profile",
                  "intensity_sweep", "foci_band", "stain_qc"],
        "What it is": [
            "One row per nucleus: pixel-level pattern metrics over the nucleoplasm (nucleus minus nucleolus). Roles: rna1=primary FISH, partner=rna2/antibody, dapi=DNA.",
            "Control vs perturbation on per-WELL means (biological replicate), Welch t-test.",
            "Mean relative intensity in 10 depth bins (0=nuclear edge, 1=center), per channel and condition.",
            "Partner intensity across the RNA1 intensity deciles (and vice-versa), per condition.",
            "Partner spots per nucleus above a set of intensity floors (foci-band), with the secondary-only anchor.",
            "Stain QC: partner perinuclear index for control / perturbation / secondary-only. Rim present in secondary-only => technical, not specific.",
        ]})
    legend = pd.DataFrame({"Column / metric": [
        "cond2", "well", "*_gini", "*_gini_raw", "*_cv", "*_top5pct_frac / top10pct_frac",
        "*_perinuc_index", "pearson_rna1_partner_allpix", "partner_foci_ge<floor>", "is_secondary_control"],
        "Meaning": [
        "Coarse group label (control prefix / perturbation / Sec = secondary-only).",
        "Well = one biological replicate.",
        "Gini of pixel intensity (0=uniform, 1=all-in-one-pixel); bg-subtracted. Higher = more 'bunched'.",
        "Gini on raw pixels (no bg subtraction), sensitivity check.",
        "Coefficient of variation (SD/mean) of pixel intensity.",
        "Fraction of total signal in the brightest 5% / 10% of pixels.",
        "Outer-25%-depth band mean / inner-50% band mean; >1 = edge/rim (perinuclear) enriched.",
        "Pixel-wise Pearson r of RNA1 vs partner over the whole nucleoplasm.",
        "Count of partner spots in the nucleus with peak intensity >= <floor>.",
        f"True if this nucleus is in the designated secondary-only control ({p_lab}).",
    ]})
    xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        readme.to_excel(xw, "How_to_read", index=False, startrow=0)
        legend.to_excel(xw, "How_to_read", index=False, startrow=len(readme) + 3)
        df.to_excel(xw, "per_nucleus", index=False)
        welch.to_excel(xw, "nt_vs_condition_welch", index=False)
        if not rad.empty:
            radt = rad.groupby(["channel", "cond2"])[DBINS].mean().reset_index()
            radt.columns = ["channel", "condition"] + [f"depthbin_{b}_edge0_center1" for b in range(10)]
            radt.to_excel(xw, "radial_profile", index=False)
        if not swe.empty:
            swcols = [c for c in swe.columns if c.startswith(("partner_at_rna1_d", "rna1_at_partner_d"))]
            swt = swe.groupby("cond2")[swcols].mean().reset_index()
            swt.to_excel(xw, "intensity_sweep", index=False)
        if foci_band is not None:
            foci_band.to_excel(xw, "foci_band", index=False)
    wb = load_workbook(xlsx)
    for ws in wb.worksheets:
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
    wb.save(xlsx)


def _write_findings(df, welch, stainqc, md_path: Path, figs: List[str]) -> None:
    lines = ["# Pixel-pattern analysis (per-nucleus, no spot calling)", "",
             f"- Nuclei: {len(df)} ({df['cond2'].value_counts().to_dict()})", ""]
    if stainqc:
        lines += ["## Stain QC (partner perinuclear index)"]
        for k, v in stainqc.items():
            lines.append(f"- {k}: {v}")
        lines.append("")
    # top Welch signals
    w = welch.dropna(subset=["welch_p"]).sort_values("welch_p")
    lines.append("## Strongest control-vs-perturbation differences (well-mean Welch)")
    for _, r in w.head(12).iterrows():
        lines.append(f"- {r['metric']}: A={r['A_mean']:.4g} B={r['B_mean']:.4g} "
                     f"({r['pct_change']:+.1f}%), Welch p={r['welch_p']:.3g}")
    lines.append("")
    lines.append(f"## Figures ({len(figs)})")
    for f in figs:
        lines.append(f"- {Path(f).name}")
    lines.append("")
    lines.append("_Reuses saved masks + raw pixels; z-plane recomputed like coloc_backfill. "
                 "The rotation 'proper background' null remains the canonical coloc metric._")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines), encoding="utf-8")


# ===========================================================================
# CLI
# ===========================================================================
def main(argv: Optional[List[str]] = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(
        prog="python -m fishsuite.core.pixel_pattern",
        description="Per-nucleus PIXEL-pattern metrics on a completed run (CPU; "
                    "reuses saved masks, re-reads raw channels).")
    ap.add_argument("--run-dir", required=True, help="completed run output dir")
    ap.add_argument("--staging", default=None, help="raw-image folder (auto from run if omitted)")
    ap.add_argument("--input", default=None, help="alternate raw-image folder")
    ap.add_argument("--secondary-match", default=None,
                    help="substring picking the CLEAN secondary-only well/condition "
                         "(e.g. 'well12'); default = all secondary-only nuclei")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--no-excel", action="store_true")
    ap.add_argument("--out-subdir", default="pixelpattern")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    res = pixelpattern_run(
        args.run_dir, staging_dir=args.staging, input_dir=args.input,
        secondary_match=args.secondary_match,
        do_excel=not args.no_excel, do_figures=not args.no_figures,
        out_subdir=args.out_subdir, seed=args.seed)
    print("written:", json.dumps(res["written"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
