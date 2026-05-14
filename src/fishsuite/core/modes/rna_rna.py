"""rna_rna — two-RNA-channel FISH spot detection + spot-spot colocalization.

Mirrors the rna_only template but runs independent BigFISH spot detection on
two RNA channels (``cfg.channels.rna`` and ``cfg.channels.rna2``) and adds
between-channel spot-to-spot colocalization metrics:

  - per-spot ``nn_distance_um`` (nearest spot in the partner channel)
  - per-image and per-nucleus ``paired_fraction_at_<X>um`` (fraction of spots
    in each channel with a partner-channel spot within X µm; X is
    configurable via ``cfg.spot_coloc.pair_distance_um``, default 0.3 µm)
  - per-image and per-nucleus ``median_nn_distance_um``

Channels are loaded as DAPI / RNA1 / RNA2. Nuclear segmentation runs once on
DAPI (same single step as rna_only). Pixel-coloc MAD thresholds are computed
INDEPENDENTLY per channel — the runner's batch pre-scan pools nuclear pixels
separately for each. Per-channel BigFISH thresholds are auto-LoG, same
convention as rna_only.

Outputs:
  - Per-spot CSV columns include ``channel`` ('rna1' or 'rna2') plus spot
    metrics + nn_distance / paired flag.
  - Per-nucleus CSV columns include ``n_spots_rna1``, ``n_spots_rna2``,
    ``paired_fraction_rna1_at_Xum``, ``paired_fraction_rna2_at_Xum``,
    ``median_nn_distance_rna1_um``, ``median_nn_distance_rna2_um``.
  - Per-image summary row has the same two-channel split.
  - Thresholds row has both ``rna_threshold_value`` and ``rna2_threshold_value``.
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
from skimage import measure
from skimage.filters import threshold_otsu

from .. import io as _io
from .. import segmentation as _seg
from .. import spots as _spots
from .. import morphology as _morph
from .. import thresholds as _thr
from .rna_only import ImageResult, _safe_float, _median
from . import register_mode


def _resolve_channels(cfg, img) -> Tuple[int, int, int]:
    """Resolve dapi / rna / rna2 channel indices from cfg, with auto fallback.

    Raises ValueError if rna2 is missing or matches rna (rna_rna needs two
    distinct RNA channels).
    """
    one_indexed = bool(cfg.channels.one_indexed)
    def _chan(idx: int) -> int:
        return (idx - 1) if (one_indexed and idx > 0) else idx

    dapi_idx = _chan(cfg.channels.dapi)
    rna_idx = _chan(cfg.channels.rna)
    rna2_idx = _chan(cfg.channels.rna2)

    if dapi_idx < 0 or rna_idx < 0:
        auto = _io.autodetect_channels(img)
        if dapi_idx < 0:
            dapi_idx = auto["dapi"]
        if rna_idx < 0:
            rna_idx = auto["rna"]

    # rna2 has no auto-detect contract — rna_rna mode requires an explicit
    # second RNA channel.
    if rna2_idx < 0:
        raise ValueError(
            "rna_rna mode requires cfg.channels.rna2 to be set to a valid "
            "channel index (got -1). Set rna2 in the YAML preset."
        )
    if rna2_idx == rna_idx:
        raise ValueError(
            f"rna_rna mode requires two DISTINCT RNA channels; got rna={rna_idx} "
            f"== rna2={rna2_idx}. Pick different channels."
        )

    dapi_idx = max(0, min(img.n_channels - 1, dapi_idx))
    rna_idx = max(0, min(img.n_channels - 1, rna_idx))
    rna2_idx = max(0, min(img.n_channels - 1, rna2_idx))
    return dapi_idx, rna_idx, rna2_idx


def _compute_pixel_coloc_thr(
    img2d: np.ndarray,
    labels: np.ndarray,
    *,
    pc_cfg,
    precomputed: Optional[float],
    bigfish_auto_thr: float,
) -> float:
    """Compute (or accept the pre-scan) pixel-coloc threshold for ONE channel.

    Same MAD-on-raw-nuclear-pixels math as rna_only, factored out so we can
    apply it independently to rna and rna2.
    """
    _scope = getattr(pc_cfg, "threshold_scope", "per_image") if pc_cfg is not None else "per_image"
    if (
        pc_cfg is not None
        and _scope == "batch"
        and precomputed is not None
        and precomputed == precomputed  # not NaN
        and precomputed > 0
    ):
        return float(precomputed)

    if pc_cfg is not None and int(labels.max()) > 0:
        nuc_pixel_mask = labels > 0
        if nuc_pixel_mask.any():
            vals = img2d[nuc_pixel_mask].astype(np.float64).tolist()
            try:
                v = float(_thr.coloc_threshold(
                    vals,
                    mode=pc_cfg.threshold_mode,
                    k_mad=float(pc_cfg.k_mad),
                    percentile=float(pc_cfg.percentile),
                ))
                if v == v and v > 0:
                    return v
            except Exception:
                pass

    # Fallback chain: BigFISH auto -> Otsu -> 99th percentile
    if bigfish_auto_thr == bigfish_auto_thr and bigfish_auto_thr > 0:
        return float(bigfish_auto_thr)
    try:
        return float(threshold_otsu(img2d))
    except Exception:
        return float(np.percentile(img2d, 99.0))


def _spot_coords_um(spots_df: pd.DataFrame, voxel_xy_um: float, voxel_z_um: float) -> np.ndarray:
    """Return Nx3 (x, y, z) coordinates in microns for the cKDTree query.

    Empty -> shape (0, 3) so cKDTree builds happily and query returns inf.
    """
    if spots_df is None or len(spots_df) == 0:
        return np.empty((0, 3), dtype=np.float64)
    x_um = spots_df["x_px"].astype(float).to_numpy() * float(voxel_xy_um)
    y_um = spots_df["y_px"].astype(float).to_numpy() * float(voxel_xy_um)
    z_um = spots_df["z_slice"].astype(float).to_numpy() * float(voxel_z_um)
    return np.stack([x_um, y_um, z_um], axis=1)


def _nn_distances(
    src_coords: np.ndarray, partner_coords: np.ndarray
) -> np.ndarray:
    """For each row in src_coords, distance (µm) to the nearest partner.

    Empty partner -> array of inf. Empty source -> empty array. Uses
    scipy.spatial.cKDTree for efficient queries.
    """
    if src_coords.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    if partner_coords.shape[0] == 0:
        return np.full(src_coords.shape[0], np.inf, dtype=np.float64)
    from scipy.spatial import cKDTree
    tree = cKDTree(partner_coords)
    dists, _ = tree.query(src_coords, k=1)
    return np.asarray(dists, dtype=np.float64)


def _format_pair_um(pair_um: float) -> str:
    """Make a column-friendly distance suffix, e.g. 0.3 -> '0p3um'."""
    s = f"{float(pair_um):.3f}".rstrip("0").rstrip(".")
    return s.replace(".", "p") + "um"


def run_one(
    path,
    *,
    condition: str,
    sec_only: bool,
    cfg,
    precomputed_rna_threshold: Optional[float] = None,
    precomputed_rna2_threshold: Optional[float] = None,
) -> ImageResult:
    """Run the rna_rna pipeline on a single image.

    Parameters
    ----------
    precomputed_rna_threshold, precomputed_rna2_threshold : float or None
        When supplied (typically by the batch runner's pre-pass in
        ``pixel_coloc.threshold_scope == 'batch'`` mode), these scalars are
        used verbatim as the pixel-coloc thresholds for THIS image's rna and
        rna2 channels respectively — bypassing the per-image median+k*MAD
        computation. Each channel has its OWN pooled pre-scan.
    """
    t0 = time.time()
    img = _io.read_image(path)

    dapi_idx, rna_idx, rna2_idx = _resolve_channels(cfg, img)

    z_mode = cfg.z_stack.mode
    z_start = cfg.z_stack.start_slice
    z_end = cfg.z_stack.end_slice
    if z_start is not None and z_start > img.n_z:
        z_start = 1
    if z_end is not None and z_end > img.n_z:
        z_end = img.n_z

    dapi_2d = _io.extract_channel(img, dapi_idx, z_mode=z_mode, z_start=z_start, z_end=z_end)
    if dapi_2d.ndim != 2:
        dapi_2d = dapi_2d.max(axis=0)
    rna_2d = _io.extract_channel(img, rna_idx, z_mode=z_mode, z_start=z_start, z_end=z_end)
    if rna_2d.ndim != 2:
        rna_2d = rna_2d.max(axis=0)
    rna2_2d = _io.extract_channel(img, rna2_idx, z_mode=z_mode, z_start=z_start, z_end=z_end)
    if rna2_2d.ndim != 2:
        rna2_2d = rna2_2d.max(axis=0)

    voxel_xy_nm = _safe_float(img.voxel_xy_nm)
    if not (voxel_xy_nm > 0):
        voxel_xy_nm = 65.0
    voxel_z_nm = _safe_float(img.voxel_z_nm)
    if not (voxel_z_nm > 0):
        voxel_z_nm = 230.0
    voxel_xy_um = voxel_xy_nm / 1000.0
    voxel_z_um = voxel_z_nm / 1000.0

    # ---- DAPI threshold mask (walkthrough step 02) ------------------------
    try:
        dapi_thr_val = float(threshold_otsu(dapi_2d))
    except Exception:
        dapi_thr_val = float(dapi_2d.mean())
    dapi_mask = (dapi_2d >= dapi_thr_val).astype(np.uint8) * 255

    # ---- Nuclear segmentation (single pass on DAPI) ------------------------
    seg_params = dict(
        min_area=cfg.nuclei.min_area_px,
        max_area=cfg.nuclei.max_area_px,
        prob_threshold=cfg.nuclei.prob_threshold,
        nms_threshold=cfg.nuclei.nms_threshold,
        n_tiles=cfg.nuclei.n_tiles,
        stardist_model=cfg.nuclei.stardist_model,
        stardist_gauss_sigma=cfg.nuclei.stardist_gauss_sigma,
        stardist_postprocess=cfg.nuclei.stardist_postprocess,
        stardist_postprocess_dilate_px=cfg.nuclei.stardist_postprocess_dilate_px,
        stardist_postprocess_otsu_sigma=cfg.nuclei.stardist_postprocess_otsu_sigma,
        stardist_postprocess_mask_closing_px=cfg.nuclei.stardist_postprocess_mask_closing_px,
        label_smoothing_radius_px=cfg.nuclei.label_smoothing_radius_px,
        diameter=cfg.nuclei.cellpose_diameter_px,
        flow_threshold=cfg.nuclei.cellpose_flow_threshold,
        cellprob_threshold=cfg.nuclei.cellpose_cellprob_threshold,
        cellpose_model_type=cfg.nuclei.cellpose_model_type,
    )
    labels = _seg.segment_nuclei(dapi_2d, backend=cfg.nuclei.backend, params=seg_params)
    n_before = int(labels.max())
    if cfg.nuclei.exclude_border:
        labels = _seg.exclude_border_labels(labels, margin_px=cfg.nuclei.border_margin_px)
    n_after = int(labels.max())
    n_border_excluded = n_before - n_after

    # ---- Cytoplasm mask ----------------------------------------------------
    cyt_labels = None
    if cfg.cytoplasm.enabled and n_after > 0:
        cyt_labels = _morph.compute_cytoplasm_mask(
            labels, max_expand_px=cfg.cytoplasm.voronoi_max_expansion_px
        )

    # ---- Spot detection: per channel, independent BigFISH calls ------------
    # Resolve per-channel BigFISH params via FociCfg.resolved_for. When the
    # override fields on rna_overrides / rna2_overrides are None, the shared
    # FociCfg values are used (back-compat path).
    rna1_params = cfg.foci.resolved_for("rna")
    rna2_params = cfg.foci.resolved_for("rna2")

    def _detect(rna_img: np.ndarray, params: Dict[str, Any]) -> Tuple[pd.DataFrame, float]:
        if not (cfg.foci.enabled and not sec_only):
            return pd.DataFrame(), float("nan")
        vx = cfg.foci.bigfish_voxel_size_nm
        vz = cfg.foci.bigfish_voxel_z_nm
        if vx <= 0:
            vx = voxel_xy_nm
        if vz <= 0:
            vz = voxel_z_nm
        try:
            df_ = _spots.detect_spots(
                rna_img,
                backend=cfg.foci.backend,
                voxel_xy_nm=float(vx),
                voxel_z_nm=float(vz),
                spot_radius_nm=float(params["bigfish_spot_radius_nm"]),
                spot_radius_z_nm=float(params["bigfish_spot_radius_z_nm"]),
                threshold_multiplier=float(params["threshold_multiplier"]),
                threshold=cfg.foci.threshold_override,
                log_threshold=cfg.foci.log_threshold,
                log_spot_radius_px=cfg.foci.log_spot_radius_px,
            )
            thr_ = float(df_["threshold_used"].iloc[0]) if len(df_) else float("nan")
            return df_, thr_
        except Exception:
            return pd.DataFrame(), float("nan")

    spots1_df, thr1_val = _detect(rna_2d, rna1_params)
    spots2_df, thr2_val = _detect(rna2_2d, rna2_params)

    # Stratify spots vs nuclei / cytoplasm — same call as rna_only
    if cyt_labels is not None:
        if len(spots1_df) > 0:
            spots1_df = _morph.stratify_spots(spots1_df, labels, cytoplasm_labels=cyt_labels)
        if len(spots2_df) > 0:
            spots2_df = _morph.stratify_spots(spots2_df, labels, cytoplasm_labels=cyt_labels)
    else:
        if len(spots1_df) > 0:
            spots1_df = _morph.stratify_spots(spots1_df, labels)
        if len(spots2_df) > 0:
            spots2_df = _morph.stratify_spots(spots2_df, labels)

    # ---- Pixel-coloc thresholds (independent per channel) ------------------
    pc_cfg = getattr(cfg, "pixel_coloc", None)
    rna_thr_value = _compute_pixel_coloc_thr(
        rna_2d, labels, pc_cfg=pc_cfg,
        precomputed=precomputed_rna_threshold, bigfish_auto_thr=thr1_val,
    )
    rna2_thr_value = _compute_pixel_coloc_thr(
        rna2_2d, labels, pc_cfg=pc_cfg,
        precomputed=precomputed_rna2_threshold, bigfish_auto_thr=thr2_val,
    )
    rna_pos_mask = (rna_2d >= rna_thr_value).astype(np.uint8) * 255
    rna2_pos_mask = (rna2_2d >= rna2_thr_value).astype(np.uint8) * 255

    # ---- Spot-spot colocalization (cKDTree NN) -----------------------------
    sc_cfg = getattr(cfg, "spot_coloc", None)
    pair_um = float(getattr(sc_cfg, "pair_distance_um", 0.3)) if sc_cfg is not None else 0.3
    report_nn = bool(getattr(sc_cfg, "report_nn_distance", True)) if sc_cfg is not None else True
    pair_suffix = _format_pair_um(pair_um)

    coords1 = _spot_coords_um(spots1_df, voxel_xy_um, voxel_z_um)
    coords2 = _spot_coords_um(spots2_df, voxel_xy_um, voxel_z_um)
    nn1 = _nn_distances(coords1, coords2)  # rna1 -> nearest rna2
    nn2 = _nn_distances(coords2, coords1)  # rna2 -> nearest rna1
    paired1 = (nn1 <= pair_um) if nn1.size else np.empty(0, dtype=bool)
    paired2 = (nn2 <= pair_um) if nn2.size else np.empty(0, dtype=bool)

    if len(spots1_df) > 0:
        spots1_df = spots1_df.copy()
        spots1_df["nn_distance_um"] = nn1 if report_nn else np.full(len(spots1_df), np.nan)
        spots1_df[f"paired_at_{pair_suffix}"] = paired1.astype(int)
    if len(spots2_df) > 0:
        spots2_df = spots2_df.copy()
        spots2_df["nn_distance_um"] = nn2 if report_nn else np.full(len(spots2_df), np.nan)
        spots2_df[f"paired_at_{pair_suffix}"] = paired2.astype(int)

    # ---- Per-nucleus rows --------------------------------------------------
    img_name = path.name

    # regionprops for shape descriptors
    rp_props = ("label", "area", "perimeter", "centroid",
                "eccentricity", "solidity", "feret_diameter_max",
                "major_axis_length", "minor_axis_length")
    if n_after > 0:
        rp_table = measure.regionprops_table(labels, properties=rp_props)
        rp_df = pd.DataFrame(rp_table)
    else:
        rp_df = pd.DataFrame(columns=["label"])
    rp_by_id = {int(r["label"]): r for r in rp_df.to_dict(orient="records")}

    # Index per-channel spots by nucleus_id for cheap per-nucleus aggregation
    def _by_nid(df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
        out_: Dict[int, pd.DataFrame] = {}
        if len(df) > 0 and "nucleus_id" in df.columns:
            for nid_val, grp in df.groupby("nucleus_id"):
                try:
                    out_[int(nid_val)] = grp
                except (TypeError, ValueError):
                    pass
        return out_

    spots1_by_nid = _by_nid(spots1_df)
    spots2_by_nid = _by_nid(spots2_df)

    # Spot diameter defaults (same as rna_only)
    spot_radius_um = float(cfg.foci.bigfish_spot_radius_nm) / 1000.0
    default_spot_diameter_um = 2.0 * spot_radius_um
    default_spot_fwhm_px = default_spot_diameter_um / max(voxel_xy_um, 1e-6)
    default_spot_area_px = math.pi * (default_spot_fwhm_px / 2.0) ** 2

    nuc_rows: List[Dict[str, Any]] = []
    spot_rows: List[Dict[str, Any]] = []
    morph_rows: List[Dict[str, Any]] = []
    spot_global_id = 0

    paired_col = f"paired_fraction_at_{pair_suffix}"

    for nid in range(1, n_after + 1):
        rp = rp_by_id.get(nid, {})
        nucleus_area_px = int(rp.get("area", 0))
        perim_px = float(rp.get("perimeter", 0.0))
        area_um2 = nucleus_area_px * (voxel_xy_um ** 2)
        perimeter_um = perim_px * voxel_xy_um

        nuc_mask = labels == nid
        if nuc_mask.any():
            rna_vals = rna_2d[nuc_mask].astype(np.float64)
            rna2_vals = rna2_2d[nuc_mask].astype(np.float64)
            dapi_vals = dapi_2d[nuc_mask].astype(np.float64)
            rna_mean = float(rna_vals.mean())
            rna2_mean = float(rna2_vals.mean())
            sum_rna_intensity = float(rna_vals.sum())
            sum_rna2_intensity = float(rna2_vals.sum())
            dapi_mean = float(dapi_vals.mean())
        else:
            rna_mean = rna2_mean = sum_rna_intensity = sum_rna2_intensity = dapi_mean = float("nan")

        if cyt_labels is not None:
            cyt_mask = (cyt_labels == nid) & (~nuc_mask)
            if cyt_mask.any():
                rna_cytoplasmic_mean = float(rna_2d[cyt_mask].astype(np.float64).mean())
                rna2_cytoplasmic_mean = float(rna2_2d[cyt_mask].astype(np.float64).mean())
                sum_rna_intensity_cyto = float(rna_2d[cyt_mask].astype(np.float64).sum())
                sum_rna2_intensity_cyto = float(rna2_2d[cyt_mask].astype(np.float64).sum())
                cyto_area_px = int(cyt_mask.sum())
            else:
                rna_cytoplasmic_mean = rna2_cytoplasmic_mean = float("nan")
                sum_rna_intensity_cyto = sum_rna2_intensity_cyto = 0.0
                cyto_area_px = 0
        else:
            rna_cytoplasmic_mean = rna2_cytoplasmic_mean = float("nan")
            sum_rna_intensity_cyto = sum_rna2_intensity_cyto = 0.0
            cyto_area_px = 0

        # "Cell" = nucleus + cytoplasm (matches rna_only semantics for the
        # mean_cell_total_intensity_fit metric Brian wants for both channels).
        cell_area_px = int(nucleus_area_px) + int(cyto_area_px)
        cell_total_intensity_rna1 = sum_rna_intensity + sum_rna_intensity_cyto
        cell_total_intensity_rna2 = sum_rna2_intensity + sum_rna2_intensity_cyto

        def _nc(nmean, cmean):
            if cmean and cmean > 0 and not math.isnan(cmean):
                return nmean / cmean
            return float("nan")
        rna_nc_ratio = _nc(rna_mean, rna_cytoplasmic_mean)
        rna2_nc_ratio = _nc(rna2_mean, rna2_cytoplasmic_mean)

        # N/C ratio of TOTAL pixel-sum intensity (not the mean-based ratio
        # above). Brian's figure suite plots both — the total-ratio is more
        # robust when nucleus and cytoplasm have very different areas.
        def _nc_total(n_sum, c_sum):
            if c_sum and c_sum > 0 and not math.isnan(c_sum):
                return n_sum / c_sum
            return float("nan")
        nc_ratio_total_intensity_rna1 = _nc_total(sum_rna_intensity, sum_rna_intensity_cyto)
        nc_ratio_total_intensity_rna2 = _nc_total(sum_rna2_intensity, sum_rna2_intensity_cyto)

        # Per-nucleus per-channel spot aggregates
        def _agg(sub):
            if sub is None or len(sub) == 0:
                return dict(
                    rna_spot_count=0,
                    nuclear_spot_count=0,
                    cyto_spot_count=0,
                    nuclear_spot_fraction=float("nan"),
                    nuclear_spot_density_per_um2=float("nan"),
                    mean_int=float("nan"),
                    total_int=float("nan"),
                    median_int=float("nan"),
                    int_cv=float("nan"),
                    median_nn_um=float("nan"),
                    paired_frac=float("nan"),
                    paired_count=0,
                )
            sub = sub.copy()
            nuc_spot_mask = sub.get("in_nucleus", pd.Series(dtype=bool)) == True
            cyt_spot_mask = sub.get("in_cytoplasm", pd.Series(dtype=bool)) == True
            n_in = int(nuc_spot_mask.sum())
            n_cy = int(cyt_spot_mask.sum())
            tot = n_in + n_cy
            nuc_frac = (n_in / float(tot)) if tot > 0 else float("nan")
            density = (tot / area_um2) if area_um2 > 0 else float("nan")
            if tot > 0 and "intensity_peak" in sub.columns:
                ipeaks = sub["intensity_peak"].astype(float)
                m_ = float(ipeaks.mean())
                t_ = float(ipeaks.sum())
                med_ = float(ipeaks.median())
                cv_ = (float(ipeaks.std()) / m_) if m_ > 0 else float("nan")
            else:
                m_ = t_ = med_ = cv_ = float("nan")
            if tot > 0 and "nn_distance_um" in sub.columns:
                nn_finite = sub["nn_distance_um"].replace([np.inf, -np.inf], np.nan).dropna()
                med_nn = float(nn_finite.median()) if len(nn_finite) else float("nan")
            else:
                med_nn = float("nan")
            if tot > 0 and f"paired_at_{pair_suffix}" in sub.columns:
                p_cnt = int(sub[f"paired_at_{pair_suffix}"].astype(int).sum())
                p_frac = p_cnt / float(tot)
            else:
                p_cnt = 0
                p_frac = float("nan")
            return dict(
                rna_spot_count=int(tot),
                nuclear_spot_count=int(n_in),
                cyto_spot_count=int(n_cy),
                nuclear_spot_fraction=nuc_frac,
                nuclear_spot_density_per_um2=density,
                mean_int=m_,
                total_int=t_,
                median_int=med_,
                int_cv=cv_,
                median_nn_um=med_nn,
                paired_frac=p_frac,
                paired_count=p_cnt,
            )

        a1 = _agg(spots1_by_nid.get(nid))
        a2 = _agg(spots2_by_nid.get(nid))

        # ---- Active transcription sites + mature mRNA per cell -----------
        # Active TS proxy: a spot that is BOTH in the nucleus AND has a
        # paired partner spot within pair_distance_um in the other channel.
        # For an exon-intron (or 5'-3' tiling) probe set, an "active TS"
        # appears as a co-localized punctum at the gene locus inside the
        # nucleus. Counting per nucleus on RNA1 (the canonical "primary"
        # channel) gives a per-cell active-TS count.
        # Mature mRNA proxy: cytoplasmic spots in the primary RNA channel
        # — these are exported transcripts. For exon-only probes this is
        # the mature mRNA pool; for intron probes this should be near 0
        # (intron retention/nuclear-localized signal).
        def _active_ts_count(sub):
            if sub is None or len(sub) == 0:
                return 0
            if "in_nucleus" not in sub.columns or f"paired_at_{pair_suffix}" not in sub.columns:
                return 0
            mask = (
                (sub["in_nucleus"].astype(bool) == True)
                & (sub[f"paired_at_{pair_suffix}"].astype(int) == 1)
            )
            return int(mask.sum())

        def _mature_mrna_count(sub):
            if sub is None or len(sub) == 0:
                return 0
            if "in_cytoplasm" not in sub.columns:
                return 0
            return int((sub["in_cytoplasm"].astype(bool) == True).sum())

        n_active_tss = _active_ts_count(spots1_by_nid.get(nid))
        n_active_tss_rna2 = _active_ts_count(spots2_by_nid.get(nid))
        n_mature_mrna_rna1 = _mature_mrna_count(spots1_by_nid.get(nid))
        n_mature_mrna_rna2 = _mature_mrna_count(spots2_by_nid.get(nid))

        # Per-nucleus row — column ordering: rna_only-compatible fields first
        # (so anything reading rna_only output still finds its columns), then
        # rna2/spot-coloc additions.
        nuc_row = {
            "image": img_name,
            "condition": condition,
            "secondary_only": sec_only,
            "experiment_id": "",
            "nucleus_id": int(nid),
            "nucleus_area_px": int(nucleus_area_px),
            # ---- RNA1 (rna) per-nucleus block — same names as rna_only ----
            "rna_mean_in_nucleus": rna_mean,
            "rna_nuclear_mean": rna_mean,
            "rna_cytoplasmic_mean": rna_cytoplasmic_mean,
            "rna_nc_ratio": rna_nc_ratio,
            "rna_spot_count": int(a1["rna_spot_count"]),
            "nuclear_spot_count": int(a1["nuclear_spot_count"]),
            "cyto_spot_count": int(a1["cyto_spot_count"]),
            "nuclear_spot_fraction": a1["nuclear_spot_fraction"],
            "nuclear_spot_density_per_um2": a1["nuclear_spot_density_per_um2"],
            "rna_spot_mean_intensity_bgc_blend": a1["mean_int"],
            "rna_spot_total_intensity_bgc_blend": a1["total_int"],
            "rna_spot_median_intensity_bgc_blend": a1["median_int"],
            "rna_spot_mean_intensity_fit": a1["mean_int"],
            "rna_spot_total_intensity_fit": a1["total_int"],
            "rna_spot_median_intensity_fit": a1["median_int"],
            "rna_spot_intensity_cv_fit": a1["int_cv"],
            "sum_rna_intensity": sum_rna_intensity,
            # ---- RNA2 per-nucleus block (rna_rna additions) --------------
            "rna2_mean_in_nucleus": rna2_mean,
            "rna2_nuclear_mean": rna2_mean,
            "rna2_cytoplasmic_mean": rna2_cytoplasmic_mean,
            "rna2_nc_ratio": rna2_nc_ratio,
            "n_spots_rna1": int(a1["rna_spot_count"]),
            "n_spots_rna2": int(a2["rna_spot_count"]),
            "nuclear_spot_count_rna2": int(a2["nuclear_spot_count"]),
            "cyto_spot_count_rna2": int(a2["cyto_spot_count"]),
            "nuclear_spot_fraction_rna2": a2["nuclear_spot_fraction"],
            "nuclear_spot_density_per_um2_rna2": a2["nuclear_spot_density_per_um2"],
            "rna2_spot_mean_intensity_fit": a2["mean_int"],
            "rna2_spot_total_intensity_fit": a2["total_int"],
            "rna2_spot_median_intensity_fit": a2["median_int"],
            "rna2_spot_intensity_cv_fit": a2["int_cv"],
            "sum_rna2_intensity": sum_rna2_intensity,
            # ---- Spot-spot colocalization ---------------------------------
            f"median_nn_distance_rna1_um": a1["median_nn_um"],
            f"median_nn_distance_rna2_um": a2["median_nn_um"],
            f"paired_fraction_rna1_at_{pair_suffix}": a1["paired_frac"],
            f"paired_fraction_rna2_at_{pair_suffix}": a2["paired_frac"],
            f"paired_spot_count_rna1_at_{pair_suffix}": int(a1["paired_count"]),
            f"paired_spot_count_rna2_at_{pair_suffix}": int(a2["paired_count"]),
            # ---- Per-CELL (nucleus + cytoplasm) intensity totals -----------
            # Brian wants total RNA intensity per nucleus AND per cell for
            # BOTH channels. "_cell_" = nucleus + voronoi cytoplasm.
            "cell_total_intensity_rna1": cell_total_intensity_rna1,
            "cell_total_intensity_rna2": cell_total_intensity_rna2,
            "cell_area_px": int(cell_area_px),
            "sum_rna_intensity_cyto": sum_rna_intensity_cyto,
            "sum_rna2_intensity_cyto": sum_rna2_intensity_cyto,
            # ---- N/C ratios of TOTAL intensity (pixel sum) ----------------
            # Brian's figure suite requires both the mean-based N/C ratio
            # (above) AND the total-intensity N/C ratio. The total ratio is
            # less sensitive to area mismatches between nucleus + cytoplasm.
            "nuclear_total_intensity_rna1": sum_rna_intensity,
            "nuclear_total_intensity_rna2": sum_rna2_intensity,
            "cytoplasmic_total_intensity_rna1": sum_rna_intensity_cyto,
            "cytoplasmic_total_intensity_rna2": sum_rna2_intensity_cyto,
            "nc_ratio_total_intensity_rna1": nc_ratio_total_intensity_rna1,
            "nc_ratio_total_intensity_rna2": nc_ratio_total_intensity_rna2,
            # ---- Active TS + mature mRNA proxies --------------------------
            # Brian's exon/intron design: active TS = nuclear + paired spot
            # (a co-localized punctum at the gene locus). Mature mRNA =
            # cytoplasmic spots in the primary channel.
            "n_active_tss_per_nucleus": int(n_active_tss),
            "n_active_tss_per_nucleus_rna2": int(n_active_tss_rna2),
            "n_mature_mrna_rna1_per_cell": int(n_mature_mrna_rna1),
            "n_mature_mrna_rna2_per_cell": int(n_mature_mrna_rna2),
            # ---- Common metadata ------------------------------------------
            "cyto_area_px": int(cyto_area_px),
            "cyto_estimation_method": "voronoi" if cyt_labels is not None else "",
            "n_voxels": int(nucleus_area_px),
            "n_pix": int(nucleus_area_px),
            "n_z_slices": int(img.n_z),
            "z_mode": z_mode,
            "z_range": f"{z_start}-{z_end}" if (z_start and z_end) else "",
            "voxel_xy_um": voxel_xy_um,
            "voxel_z_um": voxel_z_um,
            "rna_threshold_value": rna_thr_value,
            "rna2_threshold_value": rna2_thr_value,
            "rna_frac_above_thr": float((rna_2d >= rna_thr_value).sum()) / float(rna_2d.size),
            "rna2_frac_above_thr": float((rna2_2d >= rna2_thr_value).sum()) / float(rna2_2d.size),
            "dapi_mean_in_nucleus": dapi_mean,
        }
        nuc_rows.append(nuc_row)

        # Morphology row (per-nucleus, single block — shape is channel-agnostic)
        major = float(rp.get("major_axis_length", 0.0))
        minor = float(rp.get("minor_axis_length", 0.0))
        feret_max_um = float(rp.get("feret_diameter_max", 0.0)) * voxel_xy_um
        feret_min_um = (minor * voxel_xy_um) if minor > 0 else float("nan")
        aspect_ratio = (major / minor) if minor > 0 else float("nan")
        roundness = (minor / major) if major > 0 else float("nan")
        elongation = aspect_ratio
        solidity = float(rp.get("solidity", float("nan")))
        circularity = (
            4.0 * math.pi * nucleus_area_px / (perim_px ** 2)
        ) if perim_px > 0 else float("nan")
        morph_rows.append({
            "image": img_name,
            "condition": condition,
            "experiment_id": "",
            "cell_id": int(nid),
            "nucleus_id": int(nid),
            "segmentation_mode": cfg.nuclei.backend,
            "area_um2": area_um2,
            "perimeter_um": perimeter_um,
            "circularity": circularity,
            "aspect_ratio": aspect_ratio,
            "roundness": roundness,
            "elongation": elongation,
            "solidity": solidity,
            "feret_max_um": feret_max_um,
            "feret_min_um": feret_min_um,
        })

    # ---- Per-spot rows (combined into one DataFrame with ``channel`` col) --
    def _emit_spot_rows(df: pd.DataFrame, label: str):
        nonlocal spot_global_id
        if df is None or len(df) == 0:
            return
        for _, r in df.iterrows():
            spot_global_id += 1
            x_px = int(r.get("x_px", 0))
            y_px = int(r.get("y_px", 0))
            z_slice = int(r.get("z_slice", 0))
            ipeak = float(r.get("intensity_peak", float("nan")))
            nid_at = int(r.get("nucleus_id", 0))
            in_nuc_flag = bool(r.get("in_nucleus", False))
            in_cyt_flag = bool(r.get("in_cytoplasm", False))
            spot_rows.append({
                "image": img_name,
                "condition": condition,
                "secondary_only": sec_only,
                "experiment_id": "",
                "channel": label,
                "spot_id": int(spot_global_id),
                "nucleus_id": nid_at,
                "in_nucleus": int(in_nuc_flag),
                "in_cytoplasm": int(in_cyt_flag),
                "x_px": x_px,
                "y_px": y_px,
                "z_slice": z_slice,
                "z_position_um": z_slice * voxel_z_um,
                "spot_peak_intensity": ipeak,
                "quality": ipeak,
                "spot_fwhm_px": default_spot_fwhm_px,
                "spot_diameter_um": default_spot_diameter_um,
                "spot_area_px": default_spot_area_px,
                "integrated_intensity_fit": ipeak,
                "nn_distance_um": float(r.get("nn_distance_um", float("nan"))),
                f"paired_at_{pair_suffix}": int(r.get(f"paired_at_{pair_suffix}", 0)),
            })

    _emit_spot_rows(spots1_df, "rna1")
    _emit_spot_rows(spots2_df, "rna2")

    nuclei_df = pd.DataFrame(nuc_rows)
    spots_out_df = pd.DataFrame(spot_rows)
    morph_df = pd.DataFrame(morph_rows)

    # ---- Per-image summary -------------------------------------------------
    total_spots1 = int(len(spots1_df))
    total_spots2 = int(len(spots2_df))
    paired1_total = int(np.asarray(paired1, dtype=int).sum()) if paired1.size else 0
    paired2_total = int(np.asarray(paired2, dtype=int).sum()) if paired2.size else 0
    img_paired_frac_1 = (paired1_total / float(total_spots1)) if total_spots1 > 0 else float("nan")
    img_paired_frac_2 = (paired2_total / float(total_spots2)) if total_spots2 > 0 else float("nan")

    def _finite_median(arr: np.ndarray) -> float:
        if arr.size == 0:
            return float("nan")
        a = arr[np.isfinite(arr)]
        return float(np.median(a)) if a.size else float("nan")

    img_median_nn_1 = _finite_median(nn1)
    img_median_nn_2 = _finite_median(nn2)

    # Helper: compute mean/median/cv for an arbitrary column of nuclei_df,
    # optionally filtered to rows where ``cnt_col > 0`` (matches rna_only's
    # "only count expressing cells" rule for per-spot intensity aggregates).
    def _img_stats(col_: str, cnt_col_: Optional[str] = None) -> Tuple[float, float, float]:
        if len(nuclei_df) == 0 or col_ not in nuclei_df.columns:
            return float("nan"), float("nan"), float("nan")
        ser = pd.to_numeric(nuclei_df[col_], errors="coerce")
        if cnt_col_ and cnt_col_ in nuclei_df.columns:
            mask = pd.to_numeric(nuclei_df[cnt_col_], errors="coerce") > 0
            ser = ser[mask]
        vals = ser.dropna().tolist()
        if not vals:
            return float("nan"), float("nan"), float("nan")
        m_ = sum(vals) / float(len(vals))
        med_ = _median(vals)
        if len(vals) > 1 and m_ > 0:
            var_ = sum((v - m_) ** 2 for v in vals) / float(len(vals) - 1)
            cv_ = math.sqrt(var_) / m_
        else:
            cv_ = float("nan")
        return m_, med_, cv_

    if len(nuclei_df) > 0:
        counts1 = nuclei_df["rna_spot_count"].astype(int).tolist()
        counts2 = nuclei_df["n_spots_rna2"].astype(int).tolist()
        n = len(counts1)
        m1 = sum(counts1) / float(n) if n > 0 else 0.0
        m2 = sum(counts2) / float(n) if n > 0 else 0.0
        # CV of spots-per-nucleus, both channels
        if n > 1 and m1 > 0:
            sd1 = math.sqrt(sum((v - m1) ** 2 for v in counts1) / float(n - 1))
            cv_count_1 = sd1 / m1
        else:
            cv_count_1 = float("nan")
        if n > 1 and m2 > 0:
            sd2 = math.sqrt(sum((v - m2) ** 2 for v in counts2) / float(n - 1))
            cv_count_2 = sd2 / m2
        else:
            cv_count_2 = float("nan")

        # Per-channel intensity stats — spot total per cell (only-expressing)
        mean_tot_fit_1, med_tot_fit_1, cv_tot_fit_1 = _img_stats(
            "rna_spot_total_intensity_fit", "rna_spot_count")
        mean_tot_fit_2, med_tot_fit_2, cv_tot_fit_2 = _img_stats(
            "rna2_spot_total_intensity_fit", "n_spots_rna2")

        # Per-cell TOTAL RNA intensity (raw pixel sum, nucleus+cyto, BOTH
        # channels) — NOT spot-only. Always defined (no expressing-only
        # filter), since this is a pixel measurement.
        mean_cell_int_1, med_cell_int_1, cv_cell_int_1 = _img_stats(
            "cell_total_intensity_rna1")
        mean_cell_int_2, med_cell_int_2, cv_cell_int_2 = _img_stats(
            "cell_total_intensity_rna2")
        # Per-NUCLEUS TOTAL RNA intensity (raw pixel sum, nucleus only)
        mean_nuc_int_1, med_nuc_int_1, _ = _img_stats("sum_rna_intensity")
        mean_nuc_int_2, med_nuc_int_2, _ = _img_stats("sum_rna2_intensity")
        # N/C-ratio image-level rollups, both channels
        mean_nc_total_1, med_nc_total_1, _ = _img_stats("nc_ratio_total_intensity_rna1")
        mean_nc_total_2, med_nc_total_2, _ = _img_stats("nc_ratio_total_intensity_rna2")
        # Active-TS + mature-mRNA image-level rollups
        mean_active_tss, med_active_tss, _ = _img_stats("n_active_tss_per_nucleus")
        mean_mature_1, med_mature_1, _ = _img_stats("n_mature_mrna_rna1_per_cell")
        mean_mature_2, med_mature_2, _ = _img_stats("n_mature_mrna_rna2_per_cell")

        # frac_nuclei_with_ge_X_spots for both channels
        def _frac(ct, k):
            return round(sum(1 for v in ct if v >= k) / float(n), 4) if n > 0 else 0.0

        # Nuclear-vs-cyto stratification rollups, per channel
        # frac_nuclear = mean over nuclei of nuclear_spot_fraction
        nuc_frac_1_vals = [v for v in nuclei_df.get("nuclear_spot_fraction", pd.Series(dtype=float)).tolist()
                            if isinstance(v, (int, float)) and v == v]
        nuc_frac_2_vals = [v for v in nuclei_df.get("nuclear_spot_fraction_rna2", pd.Series(dtype=float)).tolist()
                            if isinstance(v, (int, float)) and v == v]
        # Per-image-level fractions: total nuclear spots / total spots
        if total_spots1 > 0:
            nuclear_spots_1 = int(
                pd.to_numeric(nuclei_df.get("nuclear_spot_count", pd.Series(dtype=int)), errors="coerce").fillna(0).sum())
            cyto_spots_1 = int(
                pd.to_numeric(nuclei_df.get("cyto_spot_count", pd.Series(dtype=int)), errors="coerce").fillna(0).sum())
            frac_nuclear_image_1 = nuclear_spots_1 / float(total_spots1)
        else:
            nuclear_spots_1 = cyto_spots_1 = 0
            frac_nuclear_image_1 = float("nan")
        if total_spots2 > 0:
            nuclear_spots_2 = int(
                pd.to_numeric(nuclei_df.get("nuclear_spot_count_rna2", pd.Series(dtype=int)), errors="coerce").fillna(0).sum())
            cyto_spots_2 = int(
                pd.to_numeric(nuclei_df.get("cyto_spot_count_rna2", pd.Series(dtype=int)), errors="coerce").fillna(0).sum())
            frac_nuclear_image_2 = nuclear_spots_2 / float(total_spots2)
        else:
            nuclear_spots_2 = cyto_spots_2 = 0
            frac_nuclear_image_2 = float("nan")

        per_image = {
            "image": img_name,
            "condition": condition,
            "secondary_only": sec_only,
            "nuclei_analyzed": int(n),
            # RNA1 default-named (compat with rna_only/rna_protein readers)
            "mean_spots_per_nucleus": round(m1, 3),
            "median_spots_per_nucleus": round(_median(counts1), 3),
            "cv_spots_per_nucleus": round(cv_count_1, 4) if cv_count_1 == cv_count_1 else float("nan"),
            "frac_nuclei_with_ge_1_spot": _frac(counts1, 1),
            "frac_nuclei_with_ge_5_spots": _frac(counts1, 5),
            "frac_nuclei_with_ge_10_spots": _frac(counts1, 10),
            # RNA1 + RNA2 named explicitly (counts)
            "mean_spots_per_nucleus_rna1": round(m1, 3),
            "mean_spots_per_nucleus_rna2": round(m2, 3),
            "median_spots_per_nucleus_rna1": round(_median(counts1), 3),
            "median_spots_per_nucleus_rna2": round(_median(counts2), 3),
            "cv_spots_per_nucleus_rna1": round(cv_count_1, 4) if cv_count_1 == cv_count_1 else float("nan"),
            "cv_spots_per_nucleus_rna2": round(cv_count_2, 4) if cv_count_2 == cv_count_2 else float("nan"),
            "frac_nuclei_with_ge_1_spot_rna1": _frac(counts1, 1),
            "frac_nuclei_with_ge_5_spots_rna1": _frac(counts1, 5),
            "frac_nuclei_with_ge_10_spots_rna1": _frac(counts1, 10),
            "frac_nuclei_with_ge_1_spot_rna2": _frac(counts2, 1),
            "frac_nuclei_with_ge_5_spots_rna2": _frac(counts2, 5),
            "frac_nuclei_with_ge_10_spots_rna2": _frac(counts2, 10),
            "total_spots_rna1": total_spots1,
            "total_spots_rna2": total_spots2,
            # Spot totals split by compartment (per channel)
            "nuclear_spots_rna1": nuclear_spots_1,
            "cytoplasmic_spots_rna1": cyto_spots_1,
            "nuclear_spots_rna2": nuclear_spots_2,
            "cytoplasmic_spots_rna2": cyto_spots_2,
            "frac_nuclear_rna1": frac_nuclear_image_1,
            "frac_nuclear_rna2": frac_nuclear_image_2,
            # ---- Per-CELL (nucleus + cytoplasm) total RNA intensity ------
            # These are the columns Brian explicitly asked for: total RNA
            # intensity per cell for BOTH channels, with mean/median/CV.
            "mean_cell_total_intensity_fit_rna1": round(mean_cell_int_1, 2) if mean_cell_int_1 == mean_cell_int_1 else float("nan"),
            "median_cell_total_intensity_fit_rna1": round(med_cell_int_1, 2) if med_cell_int_1 == med_cell_int_1 else float("nan"),
            "cv_cell_total_intensity_fit_rna1": round(cv_cell_int_1, 4) if cv_cell_int_1 == cv_cell_int_1 else float("nan"),
            "mean_cell_total_intensity_fit_rna2": round(mean_cell_int_2, 2) if mean_cell_int_2 == mean_cell_int_2 else float("nan"),
            "median_cell_total_intensity_fit_rna2": round(med_cell_int_2, 2) if med_cell_int_2 == med_cell_int_2 else float("nan"),
            "cv_cell_total_intensity_fit_rna2": round(cv_cell_int_2, 4) if cv_cell_int_2 == cv_cell_int_2 else float("nan"),
            # Per-nucleus (nucleus-only) total RNA intensity, both channels
            "mean_nuc_total_intensity_rna1": round(mean_nuc_int_1, 2) if mean_nuc_int_1 == mean_nuc_int_1 else float("nan"),
            "median_nuc_total_intensity_rna1": round(med_nuc_int_1, 2) if med_nuc_int_1 == med_nuc_int_1 else float("nan"),
            "mean_nuc_total_intensity_rna2": round(mean_nuc_int_2, 2) if mean_nuc_int_2 == mean_nuc_int_2 else float("nan"),
            "median_nuc_total_intensity_rna2": round(med_nuc_int_2, 2) if med_nuc_int_2 == med_nuc_int_2 else float("nan"),
            # Spot-detected total intensity per cell (only-expressing cells)
            "mean_cell_total_spot_intensity_fit_rna1": round(mean_tot_fit_1, 2) if mean_tot_fit_1 == mean_tot_fit_1 else float("nan"),
            "median_cell_total_spot_intensity_fit_rna1": round(med_tot_fit_1, 2) if med_tot_fit_1 == med_tot_fit_1 else float("nan"),
            "cv_cell_total_spot_intensity_fit_rna1": round(cv_tot_fit_1, 4) if cv_tot_fit_1 == cv_tot_fit_1 else float("nan"),
            "mean_cell_total_spot_intensity_fit_rna2": round(mean_tot_fit_2, 2) if mean_tot_fit_2 == mean_tot_fit_2 else float("nan"),
            "median_cell_total_spot_intensity_fit_rna2": round(med_tot_fit_2, 2) if med_tot_fit_2 == med_tot_fit_2 else float("nan"),
            "cv_cell_total_spot_intensity_fit_rna2": round(cv_tot_fit_2, 4) if cv_tot_fit_2 == cv_tot_fit_2 else float("nan"),
            # ---- N/C ratio rollups (image-level mean/median) --------------
            "mean_nc_ratio_total_intensity_rna1": round(mean_nc_total_1, 4) if mean_nc_total_1 == mean_nc_total_1 else float("nan"),
            "median_nc_ratio_total_intensity_rna1": round(med_nc_total_1, 4) if med_nc_total_1 == med_nc_total_1 else float("nan"),
            "mean_nc_ratio_total_intensity_rna2": round(mean_nc_total_2, 4) if mean_nc_total_2 == mean_nc_total_2 else float("nan"),
            "median_nc_ratio_total_intensity_rna2": round(med_nc_total_2, 4) if med_nc_total_2 == med_nc_total_2 else float("nan"),
            # ---- Active TS + mature mRNA (per-image roll-up) -------------
            "mean_n_active_tss_per_nucleus": round(mean_active_tss, 3) if mean_active_tss == mean_active_tss else 0.0,
            "median_n_active_tss_per_nucleus": round(med_active_tss, 3) if med_active_tss == med_active_tss else 0.0,
            "mean_n_mature_mrna_rna1_per_cell": round(mean_mature_1, 3) if mean_mature_1 == mean_mature_1 else 0.0,
            "median_n_mature_mrna_rna1_per_cell": round(med_mature_1, 3) if med_mature_1 == med_mature_1 else 0.0,
            "mean_n_mature_mrna_rna2_per_cell": round(mean_mature_2, 3) if mean_mature_2 == mean_mature_2 else 0.0,
            "median_n_mature_mrna_rna2_per_cell": round(med_mature_2, 3) if med_mature_2 == med_mature_2 else 0.0,
            # ---- Spot-spot colocalization (between channels) -------------
            f"paired_fraction_rna1_at_{pair_suffix}": img_paired_frac_1,
            f"paired_fraction_rna2_at_{pair_suffix}": img_paired_frac_2,
            f"paired_count_rna1_at_{pair_suffix}": paired1_total,
            f"paired_count_rna2_at_{pair_suffix}": paired2_total,
            "median_nn_distance_rna1_um": img_median_nn_1,
            "median_nn_distance_rna2_um": img_median_nn_2,
            # ---- Threshold provenance ------------------------------------
            "rna_threshold_value": rna_thr_value,
            "rna2_threshold_value": rna2_thr_value,
            "rna_bigfish_log_threshold": thr1_val,
            "rna2_bigfish_log_threshold": thr2_val,
            "n_nuclei_border_excluded": int(n_border_excluded),
            "total_spots": int(total_spots1 + total_spots2),
            "runtime_s": round(time.time() - t0, 3),
            "dapi_channel": int(dapi_idx),
            "rna_channel": int(rna_idx),
            "rna2_channel": int(rna2_idx),
            "voxel_xy_nm": voxel_xy_nm,
            "voxel_z_nm": voxel_z_nm,
            "n_z": int(img.n_z),
        }
    else:
        per_image = {
            "image": img_name,
            "condition": condition,
            "secondary_only": sec_only,
            "nuclei_analyzed": 0,
            "mean_spots_per_nucleus": 0.0,
            "median_spots_per_nucleus": 0.0,
            "cv_spots_per_nucleus": float("nan"),
            "frac_nuclei_with_ge_1_spot": 0.0,
            "frac_nuclei_with_ge_5_spots": 0.0,
            "frac_nuclei_with_ge_10_spots": 0.0,
            "mean_spots_per_nucleus_rna1": 0.0,
            "mean_spots_per_nucleus_rna2": 0.0,
            "median_spots_per_nucleus_rna1": 0.0,
            "median_spots_per_nucleus_rna2": 0.0,
            "cv_spots_per_nucleus_rna1": float("nan"),
            "cv_spots_per_nucleus_rna2": float("nan"),
            "frac_nuclei_with_ge_1_spot_rna1": 0.0,
            "frac_nuclei_with_ge_5_spots_rna1": 0.0,
            "frac_nuclei_with_ge_10_spots_rna1": 0.0,
            "frac_nuclei_with_ge_1_spot_rna2": 0.0,
            "frac_nuclei_with_ge_5_spots_rna2": 0.0,
            "frac_nuclei_with_ge_10_spots_rna2": 0.0,
            "total_spots_rna1": total_spots1,
            "total_spots_rna2": total_spots2,
            "nuclear_spots_rna1": 0,
            "cytoplasmic_spots_rna1": 0,
            "nuclear_spots_rna2": 0,
            "cytoplasmic_spots_rna2": 0,
            "frac_nuclear_rna1": float("nan"),
            "frac_nuclear_rna2": float("nan"),
            "mean_cell_total_intensity_fit_rna1": float("nan"),
            "median_cell_total_intensity_fit_rna1": float("nan"),
            "cv_cell_total_intensity_fit_rna1": float("nan"),
            "mean_cell_total_intensity_fit_rna2": float("nan"),
            "median_cell_total_intensity_fit_rna2": float("nan"),
            "cv_cell_total_intensity_fit_rna2": float("nan"),
            "mean_nuc_total_intensity_rna1": float("nan"),
            "median_nuc_total_intensity_rna1": float("nan"),
            "mean_nuc_total_intensity_rna2": float("nan"),
            "median_nuc_total_intensity_rna2": float("nan"),
            "mean_cell_total_spot_intensity_fit_rna1": float("nan"),
            "median_cell_total_spot_intensity_fit_rna1": float("nan"),
            "cv_cell_total_spot_intensity_fit_rna1": float("nan"),
            "mean_cell_total_spot_intensity_fit_rna2": float("nan"),
            "median_cell_total_spot_intensity_fit_rna2": float("nan"),
            "cv_cell_total_spot_intensity_fit_rna2": float("nan"),
            "mean_nc_ratio_total_intensity_rna1": float("nan"),
            "median_nc_ratio_total_intensity_rna1": float("nan"),
            "mean_nc_ratio_total_intensity_rna2": float("nan"),
            "median_nc_ratio_total_intensity_rna2": float("nan"),
            "mean_n_active_tss_per_nucleus": 0.0,
            "median_n_active_tss_per_nucleus": 0.0,
            "mean_n_mature_mrna_rna1_per_cell": 0.0,
            "median_n_mature_mrna_rna1_per_cell": 0.0,
            "mean_n_mature_mrna_rna2_per_cell": 0.0,
            "median_n_mature_mrna_rna2_per_cell": 0.0,
            f"paired_fraction_rna1_at_{pair_suffix}": img_paired_frac_1,
            f"paired_fraction_rna2_at_{pair_suffix}": img_paired_frac_2,
            f"paired_count_rna1_at_{pair_suffix}": paired1_total,
            f"paired_count_rna2_at_{pair_suffix}": paired2_total,
            "median_nn_distance_rna1_um": img_median_nn_1,
            "median_nn_distance_rna2_um": img_median_nn_2,
            "rna_threshold_value": rna_thr_value,
            "rna2_threshold_value": rna2_thr_value,
            "rna_bigfish_log_threshold": thr1_val,
            "rna2_bigfish_log_threshold": thr2_val,
            "n_nuclei_border_excluded": int(n_border_excluded),
            "total_spots": int(total_spots1 + total_spots2),
            "runtime_s": round(time.time() - t0, 3),
            "dapi_channel": int(dapi_idx),
            "rna_channel": int(rna_idx),
            "rna2_channel": int(rna2_idx),
            "voxel_xy_nm": voxel_xy_nm,
            "voxel_z_nm": voxel_z_nm,
            "n_z": int(img.n_z),
        }

    # ---- Thresholds row ----------------------------------------------------
    _kmad = float(pc_cfg.k_mad) if pc_cfg is not None else float("nan")
    _scope = getattr(pc_cfg, "threshold_scope", "") if pc_cfg is not None else ""
    _mode = pc_cfg.threshold_mode if pc_cfg is not None else "fallback"
    thresholds = {
        "image": img_name,
        # --- RNA1 (rna) thresholds — full provenance block --------------
        "rna_threshold_used": thr1_val,         # BigFISH/LoG auto-derived
        "rna_threshold_value": rna_thr_value,   # pixel-coloc threshold actually used
        "rna_threshold_method": "pixel_coloc_" + _mode if pc_cfg is not None else "fallback",
        "rna_threshold_mode": _mode,
        "rna_threshold_k_mad": _kmad,
        "rna_threshold_scope": _scope,
        "rna_bigfish_log_threshold": thr1_val,
        # --- RNA2 thresholds — same set, suffixed -----------------------
        "rna2_threshold_used": thr2_val,
        "rna2_threshold_value": rna2_thr_value,
        "rna2_threshold_method": "pixel_coloc_" + _mode if pc_cfg is not None else "fallback",
        "rna2_threshold_mode": _mode,
        "rna2_threshold_k_mad": _kmad,
        "rna2_threshold_scope": _scope,
        "rna2_bigfish_log_threshold": thr2_val,
        # --- DAPI / segmentation / spot params --------------------------
        "dapi_threshold_method": "Otsu dark",
        "dapi_threshold_value": dapi_thr_val,
        "spot_coloc_pair_distance_um": pair_um,
        "watershed": cfg.nuclei.stardist_postprocess in ("watershed_otsu", "watershed_triangle"),
        "nuc_min_area_px": cfg.nuclei.min_area_px,
        "exclude_border_nuclei": cfg.nuclei.exclude_border,
        "z_mode": z_mode,
        "z_start": z_start,
        "z_end": z_end,
        "segmentation_backend": cfg.nuclei.backend,
        "stardist_prob_threshold": cfg.nuclei.prob_threshold,
        "spot_backend": cfg.foci.backend,
        "bigfish_spot_radius_nm": cfg.foci.bigfish_spot_radius_nm,
        "bigfish_voxel_size_nm": voxel_xy_nm,
        "bigfish_voxel_z_nm": voxel_z_nm,
        # ---- Per-channel resolved BigFISH params (overrides applied) ----
        # These are what was ACTUALLY USED for each channel. When the user
        # left rna2_overrides empty the rna1 and rna2 columns will match.
        # When they set rna2_overrides.threshold_multiplier=0.7 (etc.), the
        # rna2_* column reflects the override while the rna_* column shows
        # the shared FociCfg value.
        "rna_bigfish_spot_radius_nm": rna1_params["bigfish_spot_radius_nm"],
        "rna2_bigfish_spot_radius_nm": rna2_params["bigfish_spot_radius_nm"],
        "rna_bigfish_spot_radius_z_nm": rna1_params["bigfish_spot_radius_z_nm"],
        "rna2_bigfish_spot_radius_z_nm": rna2_params["bigfish_spot_radius_z_nm"],
        "rna_threshold_multiplier": rna1_params["threshold_multiplier"],
        "rna2_threshold_multiplier": rna2_params["threshold_multiplier"],
        "rna_only_nuclear_spots": rna1_params["only_nuclear_spots"],
        "rna2_only_nuclear_spots": rna2_params["only_nuclear_spots"],
        "rna_min_sep_px": rna1_params["min_sep_px"],
        "rna2_min_sep_px": rna2_params["min_sep_px"],
    }

    qc = dict(
        labels=labels,
        dapi_2d=dapi_2d,
        rna_2d=rna_2d,
        rna2_2d=rna2_2d,
        cyt_labels=cyt_labels,
        dapi_mask=dapi_mask,
        rna_pos_mask=rna_pos_mask,
        rna2_pos_mask=rna2_pos_mask,
        spots1=spots1_df,
        spots2=spots2_df,
        voxel_xy_nm=voxel_xy_nm,
    )

    return ImageResult(
        image=img_name,
        condition=condition,
        sec_only=sec_only,
        per_image=per_image,
        nuclei=nuclei_df,
        spots=spots_out_df,
        morphology=morph_df,
        thresholds=thresholds,
        qc=qc,
        extra=dict(
            rna_thr_value=rna_thr_value,
            rna2_thr_value=rna2_thr_value,
            dapi_thr_value=dapi_thr_val,
            n_border_excluded=n_border_excluded,
            pair_suffix=pair_suffix,
            pair_distance_um=pair_um,
            mode="rna_rna",
        ),
    )


@register_mode("rna_rna")
def run(*args, **kwargs):
    return run_one(*args, **kwargs)


# Helper used by the batch-scope pre-pass in runner.py. Loads ONE image,
# segments nuclei (with border exclusion matching the main pass), and
# returns BOTH rna and rna2 raw nuclear pixel arrays. Each is pooled
# SEPARATELY across the batch -> two batch thresholds.
def collect_nuclear_rna_pixels(path, *, cfg) -> Tuple[np.ndarray, np.ndarray]:
    """Return (rna_nuclear_pixels, rna2_nuclear_pixels) for one image.

    Mirrors rna_only.collect_nuclear_rna_pixels but returns BOTH channels.
    Same channel resolution, segmentation, border exclusion, and dtype as
    the per-image path in ``run_one``.
    """
    img = _io.read_image(path)
    dapi_idx, rna_idx, rna2_idx = _resolve_channels(cfg, img)

    z_mode = cfg.z_stack.mode
    z_start = cfg.z_stack.start_slice
    z_end = cfg.z_stack.end_slice
    if z_start is not None and z_start > img.n_z:
        z_start = 1
    if z_end is not None and z_end > img.n_z:
        z_end = img.n_z

    dapi_2d = _io.extract_channel(img, dapi_idx, z_mode=z_mode, z_start=z_start, z_end=z_end)
    if dapi_2d.ndim != 2:
        dapi_2d = dapi_2d.max(axis=0)
    rna_2d = _io.extract_channel(img, rna_idx, z_mode=z_mode, z_start=z_start, z_end=z_end)
    if rna_2d.ndim != 2:
        rna_2d = rna_2d.max(axis=0)
    rna2_2d = _io.extract_channel(img, rna2_idx, z_mode=z_mode, z_start=z_start, z_end=z_end)
    if rna2_2d.ndim != 2:
        rna2_2d = rna2_2d.max(axis=0)

    seg_params = dict(
        min_area=cfg.nuclei.min_area_px,
        max_area=cfg.nuclei.max_area_px,
        prob_threshold=cfg.nuclei.prob_threshold,
        nms_threshold=cfg.nuclei.nms_threshold,
        n_tiles=cfg.nuclei.n_tiles,
        stardist_model=cfg.nuclei.stardist_model,
        stardist_gauss_sigma=cfg.nuclei.stardist_gauss_sigma,
        stardist_postprocess=cfg.nuclei.stardist_postprocess,
        stardist_postprocess_dilate_px=cfg.nuclei.stardist_postprocess_dilate_px,
        stardist_postprocess_otsu_sigma=cfg.nuclei.stardist_postprocess_otsu_sigma,
        stardist_postprocess_mask_closing_px=cfg.nuclei.stardist_postprocess_mask_closing_px,
        label_smoothing_radius_px=cfg.nuclei.label_smoothing_radius_px,
        diameter=cfg.nuclei.cellpose_diameter_px,
        flow_threshold=cfg.nuclei.cellpose_flow_threshold,
        cellprob_threshold=cfg.nuclei.cellpose_cellprob_threshold,
        cellpose_model_type=cfg.nuclei.cellpose_model_type,
    )
    labels = _seg.segment_nuclei(dapi_2d, backend=cfg.nuclei.backend, params=seg_params)
    if cfg.nuclei.exclude_border:
        labels = _seg.exclude_border_labels(labels, margin_px=cfg.nuclei.border_margin_px)

    nuc_mask = labels > 0
    if not nuc_mask.any():
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    return (
        rna_2d[nuc_mask].astype(np.float64),
        rna2_2d[nuc_mask].astype(np.float64),
    )
