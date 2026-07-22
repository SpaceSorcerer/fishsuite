"""if_intensity — plate-level immunofluorescence antibody-validation intensity mode.

Faithful port of the LOCKED panQKI WT-vs-QKI-KO standalone pipeline
(``F:\\Image Analysis Work\\MIAT-QKI-Coloc\\WT-QKI-KO_2026_07_01\\_scripts\\``:
``config.py`` / ``01_segment_nuclei.py`` / ``02_quantify.py`` / ``03_figures.py`` /
``04_build_excel.py``). The math/logic below is ported verbatim from those scripts;
the only change is that every parameter is sourced from ``cfg`` (``cfg.if_intensity``,
``cfg.nuclei``, ``cfg.output``, ``cfg.experiment``) instead of the standalone's
``config.py`` module constants, and comparison groups are auto-derived from the
plate map rather than hardcoded well numbers.

Unlike the FISH modes this is a PLATE-LEVEL pipeline (per-well signal routing,
exposure gate, fold-over-secondary-only normalization, cross-condition Welch
stats, SuperPlots, SHARED-display micrographs), so it does not fit the per-image
``ImageResult`` contract. ``runner.run_batch`` diverts to ``run_if_batch`` (see the
guard at runner.py ~L259) after seeds + provenance are written. The per-image
``run`` below is only a registry entry so ``get_mode('if_intensity')`` resolves.

Quantification reproduces the standalone EXACTLY: it forces the same
``bioio_bioformats`` reader + scene-picker + ``get_image_data('CYX', T=0, Z=0)``
loader (float64 channels), reuses fishsuite ``segmentation.segment_nuclei`` with
the identical cpsam/DirectML params, and runs skimage ``regionprops_table``
``intensity_mean`` per nucleus. Human / Homo sapiens.
"""
from __future__ import annotations

import re
import datetime
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import register_mode
from .. import segmentation as _seg
from . import if_report


# ---------------------------------------------------------------------------
# Registry entry (plate-level modes never run per-image — see module docstring).
# ---------------------------------------------------------------------------
@register_mode("if_intensity")
def run(path, *, condition, sec_only, cfg, **_ignored):
    raise NotImplementedError(
        "if_intensity is a plate-level mode; it runs via run_if_batch() "
        "dispatched from runner.run_batch, not per-image."
    )


# ---------------------------------------------------------------------------
# numpy 1.x/2.x bioio asarray shim (verbatim from the standalone). Applied
# LAZILY at the start of run_if_batch (not at import) so merely importing this
# module for registration never mutates np.asarray for the other FISH modes.
# ---------------------------------------------------------------------------
_ASARRAY_SHIM_INSTALLED = False


def _install_asarray_shim():
    global _ASARRAY_SHIM_INSTALLED
    if _ASARRAY_SHIM_INSTALLED:
        return
    _orig_asarray = np.asarray

    def _patched_asarray(a, dtype=None, order=None, *, copy=None, like=None):
        if copy:
            return np.array(a, dtype=dtype, order=order, copy=True, like=like)
        return _orig_asarray(a, dtype=dtype, order=order, like=like)

    np.asarray = _patched_asarray
    _ASARRAY_SHIM_INSTALLED = True


# ---------------------------------------------------------------------------
# Plate map + FOV discovery (ported from config.py — paths from disk, biology
# from the plate CSV; comparison groups auto-derived, no hardcoded wells).
# ---------------------------------------------------------------------------
_WELL_RE = re.compile(r"well[-_ ]?(\d{1,2})", re.IGNORECASE)


def _well_num_from_folder(name: str):
    m = _WELL_RE.search(name)
    return int(m.group(1)) if m else None


def _seq_num_from_file(path: Path) -> int:
    """Acquisition sequence number = the integer token (<=3 digits) before the date."""
    stem = path.stem
    toks = stem.split("_")
    for t in reversed(toks):
        if t.isdigit() and len(t) <= 3:   # seq numbers are small; date is 8 digits
            return int(t)
    return 0


def _normalize_secondary(raw) -> str:
    """Collapse a raw secondary label to a canonical arm: 647 -> '647'; 568/565 -> '568'."""
    s = str(raw)
    if "647" in s:
        return "647"
    if "568" in s or "565" in s:
        return "568"
    return s.strip()


def _secondary_to_channel(cfg, secondary: str) -> str:
    """Map a well's (normalized) secondary to its QKI-signal CSU-channel substring.

    Uses cfg.if_intensity.signal_channel_map; note the intentional offset
    (secondary 647 -> '640' channel, 568/565 -> '561'). First map key that is a
    substring of the secondary wins; iteration preserves insertion order.
    """
    s = str(secondary)
    for key, ch in cfg.if_intensity.signal_channel_map.items():
        if key in s:
            return ch
    raise ValueError(
        f"Unknown secondary '{secondary}' (no signal_channel_map key matches)"
    )


def _resolve_plate_columns(df: pd.DataFrame):
    """Pick the well / genotype / arm / secondary columns from a plate CSV."""
    lower = {c.lower().strip(): c for c in df.columns}

    def pick(*cands):
        for c in cands:
            if c in lower:
                return lower[c]
        return None

    well_col = pick("well", "well_num", "well_number")
    geno_col = pick("genotype", "geno", "line")
    arm_col = pick("staining_arm", "arm", "staining")
    sec_col = pick(
        "secondary", "secondary_label_in_layout", "secondary_antibody",
        "secondary_label", "secondary_label_in_folder_name",
    )
    return well_col, geno_col, arm_col, sec_col


def _build_plate_map(cfg, verbose: bool = True) -> dict:
    """Build {well:int -> dict(genotype, arm, secondary, qki_channel)} from the plate CSV.

    Genotype/arm/secondary are the biology source of truth. Applies
    cfg.if_intensity.well_secondary_overrides (stringified well -> secondary),
    normalizes the secondary to a canonical arm, then derives the QKI channel.
    """
    plate: dict = {}
    csv_path = cfg.if_intensity.plate_layout_csv
    if csv_path and Path(csv_path).exists():
        df = pd.read_csv(csv_path)
        well_col, geno_col, arm_col, sec_col = _resolve_plate_columns(df)
        if not all([well_col, geno_col, arm_col, sec_col]):
            raise RuntimeError(
                f"plate_layout_csv missing required columns; found {list(df.columns)} "
                f"(need well / genotype / staining_arm|arm / secondary)"
            )
        for _, r in df.iterrows():
            try:
                w = int(r[well_col])
            except Exception:
                continue
            geno = "KO" if "KO" in str(r[geno_col]).upper() else "WT"
            # Match "primary" FIRST — the label "primary+secondary" contains the
            # substring "sec", so a naive "sec in ..." test mislabels primaries.
            arm = "primary" if "primary" in str(r[arm_col]).lower() else "seconly"
            sec = _normalize_secondary(r[sec_col])
            plate[w] = dict(genotype=geno, arm=arm, secondary=sec)
    else:
        raise RuntimeError(
            f"plate_layout_csv not found: {csv_path!r} "
            f"(if_intensity requires a plate map)"
        )

    # --- apply well-secondary overrides (conflict wells) ---
    for wk, sv in (cfg.if_intensity.well_secondary_overrides or {}).items():
        try:
            w = int(wk)
        except Exception:
            continue
        if w in plate:
            new_sec = _normalize_secondary(sv)
            if plate[w]["secondary"] != new_sec and verbose:
                print(f"  [PLATE] well-{w} secondary override: "
                      f"{plate[w]['secondary']} -> {new_sec}")
            plate[w]["secondary"] = new_sec

    # --- derive QKI channel per well ---
    for w, v in plate.items():
        v["qki_channel"] = _secondary_to_channel(cfg, v["secondary"])
    return plate


def _seconly_wells_for(plate: dict, genotype: str, secondary: str) -> list:
    """Secondary-only wells matching a genotype + secondary (for background/fold)."""
    return sorted(
        w for w, v in plate.items()
        if v["arm"] == "seconly" and v["genotype"] == genotype
        and v["secondary"] == str(secondary)
    )


def _build_comparison_groups(plate: dict):
    """Auto-derive per-secondary WT/KO primary groups + the pooled n3 groups.

    Per secondary arm: group A = primary WT wells, group B = primary KO wells.
    Pooled = all primary WT vs all primary KO. Secondaries are ordered by
    descending primary-well count (tie-break by label) so the arm with more
    replicates (the headline) comes first — matches the standalone's 647-then-568.
    """
    prim_secs = [v["secondary"] for v in plate.values() if v["arm"] == "primary"]
    uniq = sorted(set(prim_secs), key=lambda s: (-prim_secs.count(s), s))
    per_secondary = {}
    for sec in uniq:
        wt = sorted(w for w, v in plate.items()
                    if v["arm"] == "primary" and v["genotype"] == "WT" and v["secondary"] == sec)
        ko = sorted(w for w, v in plate.items()
                    if v["arm"] == "primary" and v["genotype"] == "KO" and v["secondary"] == sec)
        per_secondary[sec] = dict(WT=wt, KO=ko)
    pooled = dict(
        WT=sorted(w for w, v in plate.items() if v["arm"] == "primary" and v["genotype"] == "WT"),
        KO=sorted(w for w, v in plate.items() if v["arm"] == "primary" and v["genotype"] == "KO"),
    )
    return per_secondary, pooled


def _discover_wells(raw_dir: Path, exts=("vsi",)) -> dict:
    """{well:int -> [Path,...]} parsing well number from each subfolder name.

    Globs the given image extensions (priority order; case-insensitive, no
    leading dot; multi-dot like "ome.tif" allowed) inside each well subfolder,
    deduping while preserving discovery order, then sorts by acquisition
    sequence. Default ("vsi",) reproduces the legacy raw-.vsi behaviour exactly.
    Subfolders with no well match (e.g. ``_temp``) are skipped.
    """
    raw_dir = Path(raw_dir)
    out: dict = {}
    if not raw_dir.is_dir():
        return out
    clean_exts = [str(e).lstrip(".").lower() for e in (exts or ("vsi",)) if str(e).strip()]
    if not clean_exts:
        clean_exts = ["vsi"]
    for sub in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        w = _well_num_from_folder(sub.name)
        if w is None:
            continue
        found: list = []
        seen = set()
        for ext in clean_exts:
            for f in sub.iterdir():
                if f.is_file() and f.name.lower().endswith("." + ext) and f not in seen:
                    seen.add(f)
                    found.append(f)
        files = sorted(found, key=_seq_num_from_file)
        if files:
            out[w] = files
    return out


# ---------------------------------------------------------------------------
# Image I/O (ported verbatim from 02_quantify.py — forced bioio_bioformats
# reader + scene-picker + get_image_data('CYX', T=0, Z=0)).
# ---------------------------------------------------------------------------
def _pick_scene(img):
    """Largest single-plane (Z==1) scene with >=3 channels (as in the standalone)."""
    best = None
    for sc in img.scenes:
        img.set_scene(sc)
        d = img.dims
        c = d.C if "C" in d.order else 1
        z = d.Z if "Z" in d.order else 1
        area = d.Y * d.X
        if c >= 3 and z == 1 and (best is None or area > best[1]):
            best = (sc, area)
    return best[0] if best else None


def _exposures(img, scene_index, idx):
    """OME per-channel exposure_time (seconds); NaN where unreadable (never crash)."""
    out = {k: float("nan") for k in idx}
    try:
        ome = img.ome_metadata
        images = ome.images
        im_meta = images[scene_index] if scene_index is not None and scene_index < len(images) else images[0]
        planes = list(im_meta.pixels.planes)
        for k, ci in idx.items():
            for pl in planes:
                if getattr(pl, "the_c", None) == ci:
                    et = getattr(pl, "exposure_time", None)
                    if et is not None:
                        out[k] = float(et)
                    break
    except Exception as exc:
        print(f"    [exposure][WARN] OME read failed: {exc}")
    return out


def _load_fov(path, channel_keys, dapi_key):
    """Return (chans{key:2D float64}, exp{key:sec}, dapi_idx:int, px_um:float).

    Verbatim standalone loader: forced bioio_bioformats.Reader, largest Z==1
    C>=3 scene, get_image_data('CYX', T=0, Z=0), float64 channels.
    """
    from bioio import BioImage
    import bioio_bioformats

    img = BioImage(str(path), reader=bioio_bioformats.Reader)
    sc = _pick_scene(img)
    if sc is None:
        raise RuntimeError(f"no valid 3ch single-plane scene: {path}")
    img.set_scene(sc)
    names = [str(n) for n in list(img.channel_names or [])]
    idx = {}
    for key in channel_keys:
        hit = [i for i, n in enumerate(names) if key in n]
        if len(hit) != 1:
            raise RuntimeError(f"channel '{key}' ambiguous/absent in {path}: {names}")
        idx[key] = hit[0]
    arr = img.get_image_data("CYX", T=0, Z=0)
    chans = {k: np.asarray(arr[idx[k]]).astype(np.float64) for k in idx}
    scene_index = getattr(img, "current_scene_index", 0)
    exp = _exposures(img, scene_index, idx)
    px_um = 0.0
    try:
        pxs = img.physical_pixel_sizes
        if pxs is not None and getattr(pxs, "X", None):
            px_um = float(pxs.X)
    except Exception:
        pass
    return chans, exp, idx[dapi_key], px_um


# ---------------------------------------------------------------------------
# Segmentation (reuse fishsuite segment_nuclei exactly as the standalone did).
# ---------------------------------------------------------------------------
def _segment_dapi(dapi_f32, cfg):
    params = dict(
        min_area=cfg.nuclei.min_area_px,
        max_area=cfg.nuclei.max_area_px,
        diameter=cfg.nuclei.cellpose_diameter_px,
        flow_threshold=cfg.nuclei.cellpose_flow_threshold,
        cellprob_threshold=cfg.nuclei.cellpose_cellprob_threshold,
        cellpose_model_type=cfg.nuclei.cellpose_model_type,
        cellpose_downsample_factor=cfg.nuclei.cellpose_downsample_factor,
        cellpose_device=cfg.nuclei.cellpose_device,
    )
    labels = _seg.segment_nuclei(dapi_f32, backend=cfg.nuclei.backend, params=params)
    if cfg.nuclei.exclude_border:
        labels = _seg.exclude_border_labels(labels, margin_px=cfg.nuclei.border_margin_px)
    return labels.astype(np.int32)


# ---------------------------------------------------------------------------
# Per-nucleus / whole-FOV metrics (ported verbatim from 02_quantify.py).
# ---------------------------------------------------------------------------
def _cyto_ring_labels(labels, cfg):
    """Perinuclear ring label image (grown cyto_ring_px into background)."""
    ring_px = int(cfg.if_intensity.cyto_ring_px)
    try:
        from skimage.segmentation import expand_labels
        expanded = expand_labels(labels, distance=ring_px)
        ring = np.where(labels == 0, expanded, 0)
        return ring.astype(np.int32), True
    except Exception:
        from skimage.morphology import binary_dilation, disk
        union = binary_dilation(labels > 0, disk(ring_px)) & (labels == 0)
        return union.astype(np.int32), False


def _per_nucleus_table(labels, qki):
    from skimage.measure import regionprops_table
    props = regionprops_table(labels, intensity_image=qki,
                              properties=["label", "area", "intensity_mean"])
    df = pd.DataFrame(props)
    df["integrated"] = df["area"] * df["intensity_mean"]
    return df


def _quantify_fov(chans, exp, qki_key, dapi_key, labels, signal_channels, cfg):
    """Whole-FOV + per-nucleus + cyto-ring metrics for one FOV (verbatim math)."""
    from skimage.measure import regionprops_table

    qki = chans[qki_key]
    dapi = chans[dapi_key]

    n_nuc = int((np.unique(labels) != 0).sum())
    qki_mean = float(qki.mean())
    dapi_mean = float(dapi.mean())
    total_qki = float(qki.sum())
    total_dapi = float(dapi.sum())
    ratio = qki_mean / dapi_mean if dapi_mean > 0 else float("nan")  # == total_qki/total_dapi

    # ---- per-nucleus (within DAPI mask, RAW) ----
    if n_nuc > 0:
        nuc = _per_nucleus_table(labels, qki)
        nuc_mean = float(nuc["intensity_mean"].mean())
        nuc_int = float(nuc["integrated"].mean())
    else:
        nuc = pd.DataFrame(columns=["label", "area", "intensity_mean", "integrated"])
        nuc_mean = float("nan"); nuc_int = float("nan")

    # ---- cytoplasmic ring ----
    ring, have_expand = _cyto_ring_labels(labels, cfg)
    if have_expand and n_nuc > 0 and ring.max() > 0:
        cyto = regionprops_table(ring, intensity_image=qki,
                                 properties=["label", "intensity_mean"])
        cyto_df = pd.DataFrame(cyto)
        cyto_mean = float(cyto_df["intensity_mean"].mean())
        merged = nuc.merge(cyto_df, on="label", how="left", suffixes=("_nuc", "_cyto"))
        merged["nuc_over_cyto"] = merged["intensity_mean_nuc"] / merged["intensity_mean_cyto"]
        nuc_over_cyto = float(merged["nuc_over_cyto"].replace([np.inf, -np.inf], np.nan).mean())
        nuc = merged.rename(columns={"intensity_mean_nuc": "intensity_mean",
                                     "intensity_mean_cyto": "cyto_mean"})
    else:
        ring_vals = qki[ring > 0]
        cyto_mean = float(ring_vals.mean()) if ring_vals.size else float("nan")
        nuc_over_cyto = nuc_mean / cyto_mean if cyto_mean and cyto_mean > 0 else float("nan")
        nuc["cyto_mean"] = float("nan")
        nuc["nuc_over_cyto"] = float("nan")

    fov = dict(
        nucleus_count=n_nuc,
        qki_mean=qki_mean, dapi_mean=dapi_mean,
        total_qki=total_qki, total_dapi=total_dapi,
        ratio_qki_over_dapi=ratio,
        nuc_mean_qki=nuc_mean, nuc_integrated_qki=nuc_int,
        cyto_mean_qki=cyto_mean, nuc_over_cyto=nuc_over_cyto,
    )
    # aux whole-FOV channel means (mean_640, mean_561, ... , mean_<dapi>)
    for ch in signal_channels:
        fov[f"mean_{ch}"] = float(chans[ch].mean())
    fov[f"mean_{dapi_key}"] = dapi_mean
    # routed-channel + per-channel exposures
    fov["exp_qki_s"] = exp.get(qki_key, float("nan"))
    fov["exp_dapi_s"] = exp.get(dapi_key, float("nan"))
    for ch in signal_channels:
        fov[f"exp_{ch}_s"] = exp.get(ch, float("nan"))
    return fov, nuc


# ---------------------------------------------------------------------------
# Stats (Welch primary + Student + Cohen's d; verbatim from 02_quantify.py).
# ---------------------------------------------------------------------------
def _cohens_d(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    sa, sb = np.std(a, ddof=1), np.std(b, ddof=1)
    denom = np.sqrt((sa ** 2 + sb ** 2) / 2.0)
    return float((np.mean(a) - np.mean(b)) / denom) if denom > 0 else float("nan")


def _sig_stars(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _welch(a, b):
    from scipy.stats import ttest_ind
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    b = np.asarray(b, float); b = b[~np.isnan(b)]
    out = dict(
        n_a=len(a), n_b=len(b),
        mean_a=float(np.mean(a)) if len(a) else float("nan"),
        sd_a=float(np.std(a, ddof=1)) if len(a) > 1 else float("nan"),
        mean_b=float(np.mean(b)) if len(b) else float("nan"),
        sd_b=float(np.std(b, ddof=1)) if len(b) > 1 else float("nan"),
    )
    if len(a) >= 2 and len(b) >= 2:
        _, wp = ttest_ind(a, b, equal_var=False)   # Welch, PRIMARY
        _, sp = ttest_ind(a, b, equal_var=True)     # Student, reported alongside
        out.update(welch_p=float(wp), student_p=float(sp),
                   cohens_d=_cohens_d(a, b), sig=_sig_stars(wp))
    else:
        out.update(welch_p=float("nan"), student_p=float("nan"),
                   cohens_d=float("nan"), sig="n/a (n<2)")
    out["direction"] = ("A>B" if out["mean_a"] > out["mean_b"] else "B>A")
    return out


# ---------------------------------------------------------------------------
# Normalization + aggregation (verbatim from 02_quantify.py).
# ---------------------------------------------------------------------------
def _pool_seconly(per_fov, n_pixels):
    """Pooled sec-only per (genotype, channel) for background subtraction / fold."""
    sec = per_fov[per_fov["arm"] == "seconly"]
    pooled = {}
    for (geno, ch), grp in sec.groupby(["genotype", "qki_channel"]):
        nz = grp[grp["nucleus_count"] > 0]
        m1_base = (n_pixels * nz["qki_mean"].sum() / nz["nucleus_count"].sum()
                   if len(nz) else float("nan"))
        pooled[(geno, ch)] = dict(
            qki_mean=float(grp["qki_mean"].mean()),
            ratio=float(grp["ratio_qki_over_dapi"].mean()),
            nuc_mean=float(grp["nuc_mean_qki"].mean()),
            m1_base=m1_base, n_fov=len(grp))
    return pooled


def _add_corrected_columns(per_fov, pooled, n_pixels):
    """Per-FOV background-corrected + fold columns (fold uses pooled sec-only)."""
    def g(row, field):
        return pooled.get((row["genotype"], row["qki_channel"]), {}).get(field, float("nan"))

    per_fov["sec_qki_mean_pooled"] = per_fov.apply(lambda r: g(r, "qki_mean"), axis=1)
    per_fov["sec_ratio_pooled"] = per_fov.apply(lambda r: g(r, "ratio"), axis=1)
    per_fov["sec_nuc_mean_pooled"] = per_fov.apply(lambda r: g(r, "nuc_mean"), axis=1)
    per_fov["sec_m1_base"] = per_fov.apply(lambda r: g(r, "m1_base"), axis=1)
    # d8-CM "Per-nucleus" M1 (aux): background-subtract BEFORE x N_PIXELS / count.
    per_fov["M1_pn"] = np.where(
        per_fov["nucleus_count"] > 0,
        (per_fov["qki_mean"] - per_fov["sec_qki_mean_pooled"]) * n_pixels / per_fov["nucleus_count"],
        np.nan)
    per_fov["ratio_bgcorr"] = per_fov["ratio_qki_over_dapi"] - per_fov["sec_ratio_pooled"]
    per_fov["nuc_mean_bgcorr"] = per_fov["nuc_mean_qki"] - per_fov["sec_nuc_mean_pooled"]
    # per-FOV fold (for the SuperPlot cloud; canonical tested fold is per-well)
    per_fov["fold_qki_mean"] = per_fov["qki_mean"] / per_fov["sec_qki_mean_pooled"]
    per_fov["fold_ratio"] = per_fov["ratio_qki_over_dapi"] / per_fov["sec_ratio_pooled"]
    per_fov["fold_nuc_mean"] = per_fov["nuc_mean_qki"] / per_fov["sec_nuc_mean_pooled"]


_AGG_METRICS = ["qki_mean", "dapi_mean", "total_qki", "ratio_qki_over_dapi",
                "nuc_mean_qki", "nuc_integrated_qki", "cyto_mean_qki", "nuc_over_cyto",
                "nucleus_count", "M1_pn", "ratio_bgcorr", "nuc_mean_bgcorr",
                "fold_qki_mean", "fold_ratio", "fold_nuc_mean"]


def _aggregate_wells(per_fov):
    rows = []
    for w, grp in per_fov.groupby("well"):
        r = dict(well=w, genotype=grp["genotype"].iloc[0], arm=grp["arm"].iloc[0],
                 secondary=grp["secondary"].iloc[0], qki_channel=grp["qki_channel"].iloc[0],
                 n_fov=len(grp))
        for m in _AGG_METRICS:
            r[m] = float(grp[m].mean())
        rows.append(r)
    return pd.DataFrame(rows).sort_values("well").reset_index(drop=True)


def _add_well_fold(per_well, plate):
    """Canonical fold = primary well mean / mean of matched sec-only WELL means."""
    fold_src = {"nuc_mean_qki": "wellfold_nuc_mean",
                "ratio_qki_over_dapi": "wellfold_ratio",
                "qki_mean": "wellfold_qki_mean"}
    for out in fold_src.values():
        per_well[out] = np.nan
    for i, row in per_well.iterrows():
        if row["arm"] != "primary":
            continue
        sec_wells = _seconly_wells_for(plate, row["genotype"], row["secondary"])
        sub = per_well[per_well["well"].isin(sec_wells)]
        for src, out in fold_src.items():
            denom = float(sub[src].mean()) if not sub.empty else float("nan")
            per_well.at[i, out] = row[src] / denom if denom and denom > 0 else float("nan")


def _build_stats(per_well, plate, per_secondary, pooled):
    """Design BOTH: per-secondary raw absolute, 3-group w/ floor, pooled fold n3."""
    stats = []

    def vals(wells, col):
        return per_well[per_well["well"].isin(wells)][col].dropna().values

    # (a) PER-SECONDARY raw absolute intensity
    abs_metrics = ["nuc_mean_qki", "ratio_qki_over_dapi", "M1_pn", "nuc_integrated_qki"]
    for sec, groups in per_secondary.items():
        for metric in abs_metrics:
            wt = vals(groups["WT"], metric); ko = vals(groups["KO"], metric)
            s = _welch(wt, ko)
            floor_wt = _seconly_wells_for(plate, "WT", sec)
            floor_ko = _seconly_wells_for(plate, "KO", sec)
            s.update(analysis="per_secondary_raw", secondary=sec, metric=metric,
                     group_A="WT_primary", group_B="KO_primary",
                     floor_WT=float(np.nanmean(vals(floor_wt, metric))) if floor_wt else np.nan,
                     floor_KO=float(np.nanmean(vals(floor_ko, metric))) if floor_ko else np.nan)
            stats.append(s)

    # (b) 3-group primary vs secondary-only floor, per secondary + genotype
    for sec, groups in per_secondary.items():
        for metric in ["nuc_mean_qki", "ratio_qki_over_dapi"]:
            for geno in ("WT", "KO"):
                prim_w = groups[geno]
                sec_w = _seconly_wells_for(plate, geno, sec)
                s = _welch(vals(prim_w, metric), vals(sec_w, metric))
                s.update(analysis="primary_vs_seconly", secondary=sec, metric=metric,
                         genotype=geno, group_A=f"{geno}_primary", group_B=f"{geno}_seconly")
                stats.append(s)

    # (c) POOLED fold n3 (dimensionless, poolable across secondaries)
    wt_lbl = ",".join(str(w) for w in pooled["WT"])
    ko_lbl = ",".join(str(w) for w in pooled["KO"])
    for metric in ["wellfold_nuc_mean", "wellfold_ratio", "wellfold_qki_mean"]:
        s = _welch(vals(pooled["WT"], metric), vals(pooled["KO"], metric))
        s.update(analysis="pooled_fold_n3", secondary="pooled", metric=metric,
                 group_A=f"WT_primary({wt_lbl})", group_B=f"KO_primary({ko_lbl})")
        stats.append(s)

    df = pd.DataFrame(stats)
    print("\n=== STATS SUMMARY ===")
    for _, r in df.iterrows():
        print(f"  [{r['analysis']:>18}] sec={r.get('secondary','')} {r['metric']:>22} "
              f"{r.get('group_A','A')}={r['mean_a']:.4g}(n{r['n_a']}) vs "
              f"{r.get('group_B','B')}={r['mean_b']:.4g}(n{r['n_b']})  "
              f"p={r['welch_p']}  d={r['cohens_d']}  {r['sig']}  {r['direction']}")
    return df


# ---------------------------------------------------------------------------
# Exposure gate (WARN-not-RAISE) + conflict-well check (verbatim behavior).
# ---------------------------------------------------------------------------
def _check_exposures(per_fov, dapi_key, cfg):
    """Assert routed-QKI-channel exposure is identical within each channel group.

    WARN-not-RAISE: loud banner + status='MISMATCH' row, but execution CONTINUES.
    """
    tol = float(cfg.if_intensity.exposure_tol_s)
    print("\n=== EXPOSURE ASSERTION (routed QKI channel) ===")
    rows = []
    ok = True
    for ch, grp in per_fov.groupby("qki_channel"):
        exps = grp["exp_qki_s"].dropna().unique()
        match = (len(exps) <= 1) or (float(np.nanmax(exps) - np.nanmin(exps)) <= tol)
        status = "OK" if match else "MISMATCH"
        if not match:
            ok = False
            print(f"  !!!!!!!!!! EXPOSURE MISMATCH in QKI channel {ch}: {exps} !!!!!!!!!!")
            print(f"  Absolute-intensity WT/KO/sec-only comparison in ch{ch} is NOT valid "
                  f"until re-matched.")
        else:
            print(f"  ch{ch}: exposure {exps if len(exps) else '(unreadable)'} -> {status} "
                  f"(n={len(grp)} FOV)")
        rows.append(dict(qki_channel=ch, exposures=str(list(exps)), status=status,
                         n_fov=len(grp)))
    dexps = per_fov["exp_dapi_s"].dropna().unique()
    dmatch = (len(dexps) <= 1) or (float(np.nanmax(dexps) - np.nanmin(dexps)) <= tol)
    if not dmatch:
        print(f"  [WARN] DAPI exposure not uniform: {dexps} (affects /DAPI metric)")
    rows.append(dict(qki_channel=f"{dapi_key}(DAPI)", exposures=str(list(dexps)),
                     status="OK" if dmatch else "MISMATCH", n_fov=len(per_fov)))
    if ok and dmatch:
        print("  All compared groups exposure-matched (setting-level). Proceeding.")
    return pd.DataFrame(rows), ("OK" if ok and dmatch else "MISMATCH")


def _conflict_well_check(per_fov, plate, signal_channels, cfg):
    """Empirically report which signal channel carries a conflict well's background.

    Generic over cfg.if_intensity.well_secondary_overrides (the standalone's
    well-2 565-vs-647 check). Writes one row per override well; does NOT auto-flip.
    """
    overrides = cfg.if_intensity.well_secondary_overrides or {}
    if not overrides:
        return pd.DataFrame()
    rows = []
    print("\n=== CONFLICT-WELL empirical channel check ===")
    # invert signal_channel_map: channel substring -> a representative secondary label
    ch_to_sec = {}
    for sec_key, ch in cfg.if_intensity.signal_channel_map.items():
        ch_to_sec.setdefault(ch, _normalize_secondary(sec_key))
    for wk, override_sec in overrides.items():
        try:
            w = int(wk)
        except Exception:
            continue
        wsub = per_fov[per_fov["well"] == w]
        if wsub.empty:
            print(f"  well {w} not present — cannot check.")
            continue
        row = dict(well=w)
        means = {}
        for ch in signal_channels:
            means[ch] = float(wsub[f"mean_{ch}"].mean())
            row[f"mean_{ch}"] = means[ch]
        # reference sec-only WT well routed to each channel (exclude the conflict well)
        for ch in signal_channels:
            refw = [ww for ww, v in plate.items()
                    if ww != w and v["arm"] == "seconly" and v["genotype"] == "WT"
                    and _secondary_to_channel(cfg, v["secondary"]) == ch]
            if refw:
                rw = sorted(refw)[0]
                rsub = per_fov[per_fov["well"] == rw]
                row[f"ref_well{rw}_{ch}"] = float(rsub[f"mean_{ch}"].mean()) if not rsub.empty else float("nan")
        higher = max(means, key=means.get) if means else None
        higher_sec = ch_to_sec.get(higher, "?")
        carrier = f"{higher_sec} ({higher} channel)" if higher else "?"
        override_norm = _normalize_secondary(override_sec)
        verdict = ("consistent with override (" + override_norm + ")"
                   if higher_sec == override_norm
                   else "conflicts with override (" + override_norm + ")")
        row["higher_signal_channel"] = carrier
        row["verdict"] = verdict
        row["config_override_secondary"] = override_norm
        rows.append(row)
        print(f"  well{w}: " + "  ".join(f"mean_{ch}={means[ch]:.1f}" for ch in signal_channels))
        print(f"    -> higher-signal channel = {carrier}; {verdict}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Orchestrator.
# ---------------------------------------------------------------------------
def _condition_label(genotype, arm, secondary):
    return f"{genotype} {arm} {secondary}"


def _sanitize(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", str(s)).strip("_") or "if"


def run_if_batch(cfg, config_path, input_dir, output_dir, dirs,
                 *, dry_run=False, verbose=False) -> dict:
    """Whole IF plate pipeline. Returns a summary dict."""
    _install_asarray_shim()
    t0 = time.time()
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    prefix = cfg.output.prefix or ""

    # ---- channels to read (dedup, preserving first-seen order) ----
    dapi_key = cfg.if_intensity.dapi_channel_key
    signal_channels = []
    for ch in cfg.if_intensity.signal_channel_map.values():
        if ch not in signal_channels:
            signal_channels.append(ch)
    channel_keys = list(signal_channels) + [dapi_key]

    # ---- plate map + comparison groups (auto-derived, no hardcoded wells) ----
    plate = _build_plate_map(cfg, verbose=True)
    per_secondary, pooled = _build_comparison_groups(plate)
    print(f"\n[if_intensity] plate: {len(plate)} wells; "
          f"secondaries={list(per_secondary)}; "
          f"pooled WT={pooled['WT']} vs KO={pooled['KO']}")

    # ---- discover FOVs on disk, match to wells ----
    wells_on_disk = _discover_wells(input_dir, exts=cfg.if_intensity.input_glob_exts)
    manifest = []
    for w in sorted(wells_on_disk):
        if w not in plate:
            print(f"  [WARN] well {w} on disk but not in plate map; skipping")
            continue
        for f in wells_on_disk[w]:
            manifest.append((f, w))
    n_fov_total = len(manifest)
    print(f"[if_intensity] discovered {n_fov_total} FOV(s) across "
          f"{len(set(w for _, w in manifest))} well(s) under {input_dir}")

    if dry_run:
        print("[if_intensity] --dry-run set, exiting before processing.")
        return dict(n_wells=len(set(w for _, w in manifest)), n_fov=n_fov_total,
                    dry_run=True)

    if n_fov_total == 0:
        raise RuntimeError(f"No FOVs discovered under {input_dir} (check plate map / subfolders)")

    masks_dir = dirs.get("masks", output_dir / "masks")
    save_masks = bool(cfg.output.save_masks)

    # ---- segment + quantify every FOV ----
    print("\n=== Segment + quantify Set-2 FOVs ===")
    fov_rows, nuc_rows = [], []
    fov_runtime, fov_dapi_ch, fov_px_um = {}, {}, {}
    rep_images = {}   # well -> (dapi2d, qki2d, secondary) first FOV, for micrographs
    n_pixels = None
    for f, w in manifest:
        pv = plate[w]
        qki_key = pv["qki_channel"]
        meta = dict(well=w, genotype=pv["genotype"], arm=pv["arm"],
                    secondary=pv["secondary"], qki_channel=qki_key,
                    file=f.name, fov_seq=_seq_num_from_file(f))
        ft = time.time()
        try:
            chans, exp, dapi_idx, px_um = _load_fov(f, channel_keys, dapi_key)
            dapi_f32 = chans[dapi_key].astype(np.float32)   # bit-identical to raw uint16->f32
            labels = _segment_dapi(dapi_f32, cfg)
            if save_masks:
                masks_dir.mkdir(parents=True, exist_ok=True)
                mp = masks_dir / f"well{w}_{pv['genotype']}_{pv['arm']}_{f.stem}__labels.npy"
                np.save(mp, labels.astype(np.int32))
            fov, nuc = _quantify_fov(chans, exp, qki_key, dapi_key, labels, signal_channels, cfg)
        except Exception as exc:
            print(f"  [ERROR] well{w} {f.name}: {exc}")
            continue
        if n_pixels is None:
            n_pixels = int(chans[dapi_key].size)
        fov_runtime[f.name] = round(time.time() - ft, 2)
        fov_dapi_ch[f.name] = int(dapi_idx)
        fov_px_um[f.name] = px_um
        if w not in rep_images:
            rep_images[w] = (chans[dapi_key], chans[qki_key], pv["secondary"])
        fov_rows.append({**meta, **fov})
        for _, nr in nuc.iterrows():
            nuc_rows.append({**meta, "nucleus_label": int(nr["label"]),
                             "area_px": float(nr["area"]),
                             "nuc_mean_qki": float(nr["intensity_mean"]),
                             "nuc_integrated_qki": float(nr["integrated"]),
                             "cyto_mean_qki": float(nr.get("cyto_mean", np.nan)),
                             "nuc_over_cyto": float(nr.get("nuc_over_cyto", np.nan))})
        print(f"  well{w} {f.name} ch{qki_key}: N={fov['nucleus_count']} "
              f"nuc_mean={fov['nuc_mean_qki']:.1f} ratio={fov['ratio_qki_over_dapi']:.4f} "
              f"exp_qki={fov['exp_qki_s']}")

    per_fov = pd.DataFrame(fov_rows)
    per_nucleus = pd.DataFrame(nuc_rows)
    if per_fov.empty:
        raise RuntimeError("No FOVs quantified — check masks / raw paths.")
    if n_pixels is None:
        n_pixels = int(cfg.if_intensity.pixel_size_um and 0) or 0

    # ---- exposure gate + conflict-well check ----
    exposure_report, exposure_status = _check_exposures(per_fov, dapi_key, cfg)
    conflict_report = _conflict_well_check(per_fov, plate, signal_channels, cfg)

    # ---- pool sec-only, corrected/fold cols, aggregate to well, well-fold ----
    pooled_sec = _pool_seconly(per_fov, n_pixels)
    _add_corrected_columns(per_fov, pooled_sec, n_pixels)
    per_well = _aggregate_wells(per_fov)
    _add_well_fold(per_well, plate)

    # ---- stats (design BOTH) ----
    stats_df = _build_stats(per_well, plate, per_secondary, pooled)

    seconly_rows = [dict(genotype=k[0], qki_channel=k[1], **v) for k, v in pooled_sec.items()]
    seconly_df = pd.DataFrame(seconly_rows)

    # ---- resolve pixel size for figures / scalebar ----
    px_um = float(cfg.if_intensity.pixel_size_um or 0.0)
    if px_um <= 0:
        px_vals = [v for v in fov_px_um.values() if v and v > 0]
        px_um = float(px_vals[0]) if px_vals else 0.2167

    # ---- write CSVs (IF deliverables + standard master CSVs) ----
    _write_csvs(output_dir, prefix, per_fov, per_nucleus, per_well, stats_df,
                exposure_report, seconly_df, conflict_report, fov_runtime,
                fov_dapi_ch, fov_px_um, px_um)

    # ---- figures / micrographs / excel (never abort the run on failure) ----
    fig_dir = output_dir / "figures"
    micro_dir = output_dir / "micrographs"
    montage_dir = output_dir / "montages"
    try:
        if_report.make_superplots(fig_dir, per_well, per_nucleus, stats_df,
                                  per_secondary, pooled, plate, cfg)
    except Exception as exc:
        print(f"  [WARN] superplots failed (run continues): {exc}")
    try:
        if_report.make_micrographs(micro_dir, montage_dir, plate, per_secondary,
                                   rep_images, px_um, cfg)
    except Exception as exc:
        print(f"  [WARN] micrographs failed (run continues): {exc}")
    # Native publication images (channel panels + merges + composites, both
    # sources, all wells) — CPU render; regenerable without the GPU. Always write
    # the run-context sidecar so `fishsuite if-pub-images` can regenerate later.
    try:
        from . import if_pub_images
        if_pub_images.write_run_context(output_dir, cfg, input_dir, px_um)
        if cfg.if_intensity.pub_images:
            print("\n=== Publication images (native, both sources, all wells) ===")
            if_pub_images.make_pub_images_live(output_dir, plate, per_fov,
                                               input_dir, cfg, px_um)
    except Exception as exc:
        print(f"  [WARN] publication images failed (run continues): {exc}")
    if cfg.if_intensity.make_excel:
        try:
            xlsx_path = output_dir / f"{_sanitize(cfg.experiment.name)}_IF_intensity.xlsx"
            if_report.build_excel(xlsx_path, plate, per_nucleus, per_fov, per_well,
                                  stats_df, seconly_df, exposure_report,
                                  conflict_report, cfg)
        except Exception as exc:
            print(f"  [WARN] Excel workbook failed (run continues): {exc}")

    _append_provenance(output_dir, cfg, n_pixels, px_um)

    # ---- plain-language console summary ----
    n_wells = int(per_well["well"].nunique())
    headline = _headline_wt_ko(stats_df, per_secondary)
    print("\n================ IF-INTENSITY SUMMARY ================")
    print(f"  wells quantified : {n_wells}   FOVs : {len(per_fov)}   nuclei : {len(per_nucleus)}")
    if headline is not None:
        sec, mean_wt, mean_ko, wp = headline
        print(f"  headline (secondary {sec}, nuclear QKI mean): "
              f"WT={mean_wt:.1f} vs KO={mean_ko:.1f}  Welch p={wp}")
    print(f"  exposure gate    : {exposure_status}")
    print(f"  outputs          : {output_dir}")
    print(f"  elapsed          : {time.time() - t0:.0f}s")
    print("======================================================")

    return dict(
        n_wells=n_wells,
        n_fov=int(len(per_fov)),
        n_nuclei=int(len(per_nucleus)),
        exposure_status=exposure_status,
        headline=None if headline is None else dict(
            secondary=headline[0], nuc_mean_wt=headline[1],
            nuc_mean_ko=headline[2], welch_p=headline[3]),
    )


def _headline_wt_ko(stats_df, per_secondary):
    """The per_secondary_raw nuc_mean_qki block for the first (headline) secondary."""
    if stats_df.empty or not per_secondary:
        return None
    sec = next(iter(per_secondary))
    sub = stats_df[(stats_df["analysis"] == "per_secondary_raw")
                   & (stats_df["secondary"] == sec)
                   & (stats_df["metric"] == "nuc_mean_qki")]
    if sub.empty:
        return None
    r = sub.iloc[0]
    return sec, float(r["mean_a"]), float(r["mean_b"]), r["welch_p"]


def _write_csvs(output_dir, prefix, per_fov, per_nucleus, per_well, stats_df,
                exposure_report, seconly_df, conflict_report, fov_runtime,
                fov_dapi_ch, fov_px_um, px_um):
    output_dir = Path(output_dir)

    # --- IF-native deliverable CSVs (exact standalone schemas) ---
    per_nucleus.to_csv(output_dir / "per_nucleus.csv", index=False)
    per_fov.to_csv(output_dir / "per_fov.csv", index=False)
    per_well.to_csv(output_dir / "per_well.csv", index=False)
    stats_df.to_csv(output_dir / "stats.csv", index=False)
    exposure_report.to_csv(output_dir / "exposure_report.csv", index=False)
    seconly_df.to_csv(output_dir / "seconly_pooled.csv", index=False)
    if conflict_report is not None and not conflict_report.empty:
        conflict_report.to_csv(output_dir / "well2_channel_check.csv", index=False)

    # --- STANDARD master CSVs (baseline columns + IF metrics) ---
    pis = per_fov.copy()
    pis["image"] = pis["file"]
    pis["condition"] = [
        _condition_label(g, a, s)
        for g, a, s in zip(pis["genotype"], pis["arm"], pis["secondary"])
    ]
    pis["secondary_only"] = pis["arm"].eq("seconly")
    pis["nuclei_analyzed"] = pis["nucleus_count"]
    pis["runtime_s"] = pis["file"].map(fov_runtime)
    pis["dapi_channel"] = pis["file"].map(fov_dapi_ch)
    pis["voxel_xy_nm"] = pis["file"].map(lambda f: (fov_px_um.get(f) or px_um) * 1000.0)
    pis["voxel_z_nm"] = np.nan
    pis["n_z"] = 1
    base_cols = ["image", "condition", "secondary_only", "nuclei_analyzed",
                 "runtime_s", "dapi_channel", "voxel_xy_nm", "voxel_z_nm", "n_z"]
    other_cols = [c for c in per_fov.columns]  # meta + IF per-FOV metrics
    pis = pis[base_cols + other_cols]
    pis.to_csv(output_dir / f"{prefix}per_image_summary.csv", index=False)

    nm = per_nucleus.copy()
    if nm.empty:
        nm = pd.DataFrame(columns=[
            "image", "condition", "secondary_only", "nucleus_id", "nucleus_area_px",
            "well", "genotype", "arm", "secondary", "qki_channel", "file", "fov_seq",
            "nuc_mean_qki", "nuc_integrated_qki", "cyto_mean_qki", "nuc_over_cyto"])
    else:
        nm["image"] = nm["file"]
        nm["condition"] = [
            _condition_label(g, a, s)
            for g, a, s in zip(nm["genotype"], nm["arm"], nm["secondary"])
        ]
        nm["secondary_only"] = nm["arm"].eq("seconly")
        nm["nucleus_id"] = nm["nucleus_label"]
        nm["nucleus_area_px"] = nm["area_px"]
        nm = nm[[
            "image", "condition", "secondary_only", "nucleus_id", "nucleus_area_px",
            "well", "genotype", "arm", "secondary", "qki_channel", "file", "fov_seq",
            "nuc_mean_qki", "nuc_integrated_qki", "cyto_mean_qki", "nuc_over_cyto"]]
    nm.to_csv(output_dir / f"{prefix}nuclei_metrics.csv", index=False)

    for p in ["per_nucleus.csv", "per_fov.csv", "per_well.csv", "stats.csv",
              "exposure_report.csv", "seconly_pooled.csv",
              f"{prefix}per_image_summary.csv", f"{prefix}nuclei_metrics.csv"]:
        print(f"[write] {output_dir / p}")


def _append_provenance(output_dir, cfg, n_pixels, px_um):
    """Append an IF-specific note to command.log (runner already wrote the header)."""
    try:
        with open(Path(output_dir) / "command.log", "a", encoding="utf-8") as f:
            f.write(f"\n# --- if_intensity mode ({datetime.datetime.now().isoformat()}) ---\n")
            f.write(f"# python={sys.version.split()[0]} platform={platform.platform()}\n")
            f.write(f"# N_PIXELS={n_pixels} pixel_size_um={px_um} "
                    f"cyto_ring_px={cfg.if_intensity.cyto_ring_px} "
                    f"exposure_tol_s={cfg.if_intensity.exposure_tol_s}\n")
            f.write(f"# seg backend={cfg.nuclei.backend} model={cfg.nuclei.cellpose_model_type} "
                    f"diameter={cfg.nuclei.cellpose_diameter_px} device={cfg.nuclei.cellpose_device} "
                    f"seed={cfg.seed} fig_seed={cfg.if_intensity.fig_seed}\n")
    except Exception as exc:
        print(f"  [WARN] provenance append failed: {exc}")
