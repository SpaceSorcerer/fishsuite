"""coloc_backfill — CPU re-read QKI-at-MIAT coloc backfill for trusted runs.

2026-06-06 Brian. The trusted MIAT x QKI run was produced BEFORE the native
coloc-extra outputs existed, so it lacks ``coloc_null_draws.csv`` (the 1000
pooled null draws), ``coloc_radial_profile.csv`` and a QKI montage. Re-running on
the GPU is undesirable. This module re-reads ONLY the QKI (antibody) channel
pixels from the source VSI (CPU), reuses the run's SAVED nucleus masks + MIAT
spot coordinates, and emits those three artifacts WITHOUT re-segmenting or
re-detecting spots.

It is a thin I/O shell around a PURE numpy core
(:func:`_compute_coloc_extras_for_image`) that reuses the *exact* null math the
engine uses — the module-level ``_partner_null_for_nucleus`` /
``_radial_profile_for_nucleus`` / ``_disk_stencil`` / ``_annulus_stencils`` from
:mod:`fishsuite.core.modes.rna_rna` — so the backfill reproduces the engine
bit-for-bit (single source of truth). The shell SELF-VALIDATES by reproducing the
run's stored ``per_image_summary.protein_pooled_*`` columns; a per-image
PASS/FAIL table is printed and a loud WARNING is raised on gross mismatch (it does
NOT hard-crash — we inspect).

Architecture facts the backfill depends on (verified against the trusted run):
  * UD run = ``ANALYSIS_MODE=rna_protein``; MIAT = ``rna`` (640), QKI = ``antibody``
    (561). The antibody channel is mapped into rna_rna's ``rna2`` slot via the
    ``rna_protein._build_rna2_shim_cfg`` shim, so the QKI channel index is the
    resolved ``rna2`` index.
  * ``spot_metrics.z_slice == 0`` for every spot (the extracted-plane index, NOT
    the DAPI autofocus plane). The analysis z-plane is therefore RECOMPUTED
    deterministically from the VSI with the SAME ``intensity_weighted`` +
    z_start/z_end the run used (``extract_channel_autofocus_with_idx``); QKI is
    read at that exact plane (``extract_channel_at_z``). NEVER read z from
    ``spot_metrics``.
  * Saved masks: ``masks/<COND>__<base>__nuclei_label_mask.tif`` — reused so
    ``nucleus_id`` aligns with ``spot_metrics``. NO nucleolus label mask is
    persisted, so nucleoli are RECOMPUTED via ``detect_nucleoli(labels, dapi_2d,
    ...)`` over the SAVED labels (deterministic given labels + the recomputed
    dapi_2d -> reproduces the engine's nucleolus exclusion exactly).
  * PLAIN VSI only — the staging tree carries both ``<name>.vsi`` (PLAIN) and
    ``<name> (decon).vsi``; the backfill reads the PLAIN file and never the
    ``(decon)`` one.

CLI::

    python -m fishsuite.core.coloc_backfill --run-dir <run> --staging <staging>
        [--input <dir>] [--no-null-draws] [--no-radial] [--no-montage] [--seed 0]

PROGRESS LOG
  2026-06-06 (montage redesign): the QKI montage is now the MEAN per-nucleus-
    normalized enrichment patch at MIAT spots vs matched random in-nucleus
    positions (``_mean_enrichment_patch`` + ``_matched_random_centers``, same
    seed/nucleolus-exclusion as the null) on a shared 1.0-centred diverging LUT,
    replacing the old raw single-crop tiling (modest ~10% enrichment now visible;
    per-nucleus norm makes cross-section pooling dataset-rule-legal). Example raw
    crops kept as a small strip from ONE image. Same filename, 600 DPI.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Reuse the engine's null math — SINGLE SOURCE OF TRUTH. Duplicating it would
# break the bit-for-bit reproduction contract.
from .modes.rna_rna import (
    _disk_stencil,
    _annulus_stencils,
    _partner_null_for_nucleus,
    _radial_profile_for_nucleus,
    _rotation_null_for_nucleus,
)


# ===========================================================================
# PURE CORE — operates only on in-memory numpy arrays.
# ===========================================================================
def _nucleolus_in_nucleus(nucleolus_labels, nid, nuc_mask):
    """Return the boolean nucleolus mask WITHIN this nucleus, mirroring the
    engine: integer label image -> ``(labels == nid) & nuc_mask``; a plain
    boolean mask -> ``nucleolus & nuc_mask`` (synthetic-test convenience)."""
    if nucleolus_labels is None:
        return None
    arr = np.asarray(nucleolus_labels)
    if arr.dtype == bool:
        nin = arr & nuc_mask
    else:
        nin = (arr == nid) & nuc_mask
    return nin


def _compute_coloc_extras_for_image(
    qki_2d: np.ndarray,
    labels: np.ndarray,
    spots_xy_by_nucleus: Dict[int, Any],
    nucleolus_labels: Optional[np.ndarray],
    *,
    disk_px: float = 3.0,
    n_null: int = 1000,
    seed: int = 0,
    radial_bins_px: Optional[List[float]] = None,
    do_null_draws: bool = True,
    do_radial: bool = True,
    do_montage: bool = False,
    do_rotation: bool = False,
    rotation_min_retention: float = 0.5,
    montage_n_nuclei: int = 6,
    montage_max_spots: int = 8,
    montage_crop_half: int = 12,
    image: str = "",
    condition: str = "",
) -> Dict[str, Any]:
    """Per-image coloc extras from in-memory arrays — EXACT engine reproduction.

    Mirrors :mod:`fishsuite.core.modes.rna_rna` (the per-nucleus null block ~L1467
    + the spot-count-weighted pooling ~L1758). For each nucleus ``nid`` (iterated
    1..labels.max() — the SAME order the engine uses, which is what makes the
    shared RNG reproduce the engine's draws):

      * sampling pixels = nucleus mask MINUS nucleoli (when ``nucleolus_labels``
        given); observed MIAT spots inside a nucleolus are dropped;
      * ``_partner_null_for_nucleus`` gives the observed disk-mean QKI at the
        spots + the (n_null,) null distribution; pooled spot-count-weighted;
      * ``_radial_profile_for_nucleus`` (own rng, same seed, separate stream —
        byte-identical to the engine) gives per-ring obs/null.

    Parameters
    ----------
    spots_xy_by_nucleus : dict[int, ndarray | DataFrame]
        ``nid -> (n, 2)`` array of ``(x_px, y_px)`` MIAT spot centres (a DataFrame
        with ``x_px``/``y_px`` columns is also accepted).
    nucleolus_labels : ndarray or None
        Integer nucleolus label image (parent-nucleus ids) OR boolean mask OR
        None (no exclusion).

    Returns a dict with ``null_summary`` (dict or None), ``null_draws_rows``
    (DataFrame or None), ``radial_rows`` (list of dicts) and ``montage_crops``
    (list of dicts).
    """
    qki_f = np.asarray(qki_2d, dtype=np.float64)
    labels = np.asarray(labels)
    H, Wd = qki_f.shape
    n_after = int(labels.max()) if labels.size else 0
    n_null = int(n_null)

    # ---- null accumulators (spot-count weighted) --------------------------
    null_rng = np.random.default_rng(int(seed))
    ndy, ndx = _disk_stencil(float(disk_px))
    null_obs_num = 0.0
    null_w_den = 0.0
    null_pool = np.zeros(n_null, dtype=np.float64)
    n_nuclei_used = 0

    # ---- radial accumulators (own rng, same seed, separate stream) --------
    do_rad = bool(do_radial) and radial_bins_px is not None and len(radial_bins_px) > 0
    if do_rad:
        stencils = _annulus_stencils(list(radial_bins_px))
        n_rings = len(stencils)
        radial_rng = np.random.default_rng(int(seed))
        r_obs = np.zeros(n_rings, dtype=np.float64)
        r_nm = np.zeros(n_rings, dtype=np.float64)
        r_nsd = np.zeros(n_rings, dtype=np.float64)
        r_w = np.zeros(n_rings, dtype=np.float64)

    # ---- rotation "proper background" accumulators (own rng stream) --------
    # 2026-06-19 Brian: retrofit the rotation null for OLD runs. Rotates each
    # nucleus's spot constellation about its OWN centroid (keep-N redraw), pools
    # spot-count-weighted over rot-USABLE nuclei (full-length keep-N null -> pools
    # per-iteration exactly like the position null). Separate rng (seed offset) so
    # toggling it never perturbs the position/radial draws.
    do_rot = bool(do_rotation)
    if do_rot:
        rot_rng = np.random.default_rng(int(seed) + 101)
        rot_obs_num = 0.0
        rot_w_den = 0.0
        rot_pool = np.zeros(n_null, dtype=np.float64)
        rot_n_used = 0

    montage_candidates: List[tuple] = []

    for nid in range(1, n_after + 1):
        nuc_mask = labels == nid
        if not nuc_mask.any():
            continue
        sub = spots_xy_by_nucleus.get(nid)
        if sub is None or len(sub) == 0:
            continue
        if isinstance(sub, pd.DataFrame):
            sub = sub[["x_px", "y_px"]].astype(float).to_numpy()
        sub = np.asarray(sub, dtype=float).reshape(-1, 2)

        # sampling mask = nucleus minus nucleolus (when requested)
        samp_mask = nuc_mask
        nin = _nucleolus_in_nucleus(nucleolus_labels, nid, nuc_mask)
        if nin is not None:
            nucleoplasm = nuc_mask & (~nin)
            if nucleoplasm.any():
                samp_mask = nucleoplasm
        nys, nxs = np.where(samp_mask)

        scx = np.rint(sub[:, 0]).astype(np.intp)
        scy = np.rint(sub[:, 1]).astype(np.intp)
        scy = np.clip(scy, 0, H - 1)
        scx = np.clip(scx, 0, Wd - 1)
        # drop observed spots whose centre is inside a nucleolus
        if nin is not None:
            keep = ~nin[scy, scx]
            scy = scy[keep]
            scx = scx[keep]

        if scy.size == 0 or nys.size == 0:
            continue

        if do_null_draws:
            obs_stat, null_stats = _partner_null_for_nucleus(
                qki_f, scy, scx, nys, nxs, ndy, ndx, n_null, null_rng
            )
            if null_stats.size:
                n_sp = int(scy.size)
                null_obs_num += obs_stat * n_sp
                null_w_den += n_sp
                null_pool += null_stats * n_sp
                n_nuclei_used += 1

        if do_rad:
            rad = _radial_profile_for_nucleus(
                qki_f, scy, scx, nys, nxs, stencils, n_null, radial_rng
            )
            for ri, (ro, rnm, rnsd, rnsp) in enumerate(rad):
                if rnsp > 0 and ro == ro:  # finite observed
                    r_obs[ri] += ro * rnsp
                    r_nm[ri] += rnm * rnsp
                    r_nsd[ri] += rnsd * rnsp
                    r_w[ri] += rnsp

        if do_rot:
            cy0 = float(scy.mean())
            cx0 = float(scx.mean())
            rot = _rotation_null_for_nucleus(
                qki_f, scy, scx, samp_mask, (cy0, cx0), ndy, ndx, n_null, rot_rng,
                min_retention=float(rotation_min_retention),
            )
            rns = rot["null_stats"]
            if rot["usable"] and rns.size == n_null:
                n_sp = int(scy.size)
                rot_obs_num += rot["obs"] * n_sp
                rot_w_den += n_sp
                rot_pool += rns * n_sp
                rot_n_used += 1

        if do_montage:
            montage_candidates.append((nid, scy.copy(), scx.copy(), nys, nxs))

    # ---- pooled null summary + per-draw frame -----------------------------
    null_summary = None
    null_draws_rows = None
    if do_null_draws and null_w_den > 0:
        obs_pool = null_obs_num / null_w_den
        np_pool = null_pool / null_w_den
        np_mean = float(np_pool.mean())
        np_sd = float(np_pool.std(ddof=1)) if np_pool.size > 1 else 0.0
        enrichment = (obs_pool / np_mean) if np_mean > 0 else float("nan")
        z = ((obs_pool - np_mean) / np_sd) if np_sd > 0 else float("nan")
        p_emp = float((np.sum(np_pool >= obs_pool) + 1) / (n_null + 1))
        null_summary = {
            "pooled_obs": float(obs_pool),
            "pooled_null_mean": np_mean,
            "pooled_null_sd": np_sd,
            "pooled_null_enrichment": float(enrichment),
            "pooled_null_z": float(z),
            "pooled_null_p_empirical": p_emp,
            "n_nuclei_used": int(n_nuclei_used),
            "n_null": n_null,
            "disk_px": float(disk_px),
        }
        null_draws_rows = pd.DataFrame(
            {
                "image": image,
                "condition": condition,
                "iter": np.arange(n_null, dtype=int),
                "pooled_null_value": np_pool,
                "pooled_obs": float(obs_pool),
            }
        )

    # ---- pooled radial rows ----------------------------------------------
    radial_rows: List[Dict[str, Any]] = []
    if do_rad and float(r_w.sum()) > 0:
        for ri in range(n_rings):
            w = float(r_w[ri])
            if w <= 0:
                continue
            o = r_obs[ri] / w
            nm = r_nm[ri] / w
            nsd = r_nsd[ri] / w
            radial_rows.append(
                {
                    "ring_idx": ri,
                    "ring_px": float(radial_bins_px[ri]),
                    "obs_mean": float(o),
                    "null_mean": float(nm),
                    "null_sd": float(nsd),
                    "enrichment": float(o / nm) if nm > 0 else float("nan"),
                    "z": float((o - nm) / nsd) if nsd > 0 else float("nan"),
                    "n_spots": int(w),
                }
            )

    # ---- pooled rotation "proper background" summary + per-draw frame ------
    rotation_summary = None
    rotation_draws_rows = None
    if do_rot and rot_w_den > 0:
        ro_pool = rot_obs_num / rot_w_den
        rp_pool = rot_pool / rot_w_den
        rp_mean = float(rp_pool.mean())
        rp_sd = float(rp_pool.std(ddof=1)) if rp_pool.size > 1 else 0.0
        rot_enr = (ro_pool / rp_mean) if rp_mean > 0 else float("nan")
        rot_z = ((ro_pool - rp_mean) / rp_sd) if rp_sd > 0 else float("nan")
        rot_p = float((np.sum(rp_pool >= ro_pool) + 1) / (n_null + 1))
        rotation_summary = {
            "pooled_obs": float(ro_pool),
            "pooled_rotation_null_mean": rp_mean,
            "pooled_rotation_null_sd": rp_sd,
            "pooled_rotation_enrichment": float(rot_enr),
            "pooled_rotation_null_z": float(rot_z),
            "pooled_rotation_p_empirical": rot_p,
            "n_nuclei_used": int(rot_n_used),
            "n_null": n_null,
            "disk_px": float(disk_px),
        }
        rotation_draws_rows = pd.DataFrame(
            {
                "image": image,
                "condition": condition,
                "iter": np.arange(n_null, dtype=int),
                "pooled_null_value": rp_pool,
                "pooled_obs": float(ro_pool),
            }
        )

    # ---- montage crops ----------------------------------------------------
    montage_crops: List[Dict[str, Any]] = []
    if do_montage and montage_candidates:
        montage_crops = _montage_crops_for_image(
            qki_f, montage_candidates,
            crop_half=int(montage_crop_half),
            n_nuclei=int(montage_n_nuclei),
            max_spots=int(montage_max_spots),
            seed=int(seed),
        )

    return {
        "null_summary": null_summary,
        "null_draws_rows": null_draws_rows,
        "radial_rows": radial_rows,
        "montage_crops": montage_crops,
        "rotation_summary": rotation_summary,
        "rotation_draws_rows": rotation_draws_rows,
    }


def _crop_at(img2d: np.ndarray, cy: int, cx: int, half: int) -> np.ndarray:
    """Square crop of side ``2*half+1`` centred at (cy, cx), clipped-and-padded
    so off-edge crops keep the fixed shape (pad with the image minimum)."""
    H, Wd = img2d.shape
    out = np.full((2 * half + 1, 2 * half + 1), float(img2d.min()), dtype=np.float64)
    y0, y1 = cy - half, cy + half + 1
    x0, x1 = cx - half, cx + half + 1
    sy0, sx0 = max(0, y0), max(0, x0)
    sy1, sx1 = min(H, y1), min(Wd, x1)
    if sy1 <= sy0 or sx1 <= sx0:
        return out
    out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img2d[sy0:sy1, sx0:sx1]
    return out


def _montage_crops_for_image(
    qki_2d: np.ndarray,
    candidates: List[tuple],
    *,
    crop_half: int = 12,
    n_nuclei: int = 6,
    max_spots: int = 8,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """For the highest-MIAT-spot nuclei, return QKI crops at MIAT spots (obs) vs
    the SAME count of random in-nucleus (nucleolus-excluded) positions (null).

    ``candidates`` is the ``(nid, scy, scx, nys, nxs)`` list collected by the pure
    core (spots already clipped + nucleolus-dropped; nys/nxs already
    nucleolus-excluded). Deterministic via a dedicated ``seed`` rng so it never
    perturbs the null/radial streams.
    """
    rng = np.random.default_rng(int(seed))
    # rank nuclei by spot count (descending), take the top n_nuclei
    ranked = sorted(candidates, key=lambda c: -int(c[1].size))
    out: List[Dict[str, Any]] = []
    for (nid, scy, scx, nys, nxs) in ranked[: int(n_nuclei)]:
        n_sp = int(scy.size)
        if n_sp == 0 or nys.size == 0:
            continue
        k = min(int(max_spots), n_sp)
        # deterministic spot subset (first k after a seeded shuffle of indices)
        sp_idx = rng.permutation(n_sp)[:k]
        obs_crops = [_crop_at(qki_2d, int(scy[i]), int(scx[i]), crop_half) for i in sp_idx]
        # k random in-nucleus positions
        ridx = rng.integers(0, nys.size, size=k)
        null_crops = [_crop_at(qki_2d, int(nys[j]), int(nxs[j]), crop_half) for j in ridx]
        out.append(
            {
                "nucleus_id": int(nid),
                "n_spots": n_sp,
                "obs_crops": obs_crops,
                "null_crops": null_crops,
            }
        )
    return out


# ===========================================================================
# MEAN ENRICHMENT PATCH — the rigorous + visually clear headline.
#
# A single raw QKI crop at a MIAT spot looks identical to a random crop because
# the QKI-at-MIAT enrichment is only ~8-15% over random. Averaging MANY crops,
# AFTER per-nucleus normalization (crop / nucleus-mean QKI -> "enrichment
# units"), makes the systematic central enrichment emerge while REMOVING the
# per-section absolute-brightness differences (laser re-tuned per section) that
# our dataset rule forbids mixing. Crops are therefore poolable across
# images/conditions once normalized.
# ===========================================================================
def _nucleus_sampling_pixels(labels, nucleolus_labels, nid):
    """``(nys, nxs)`` sampling pixels for nucleus ``nid`` = nucleus MINUS the
    nucleolus when given — the SAME region the null draws from. Reuses
    ``_nucleolus_in_nucleus`` (single source of truth for the exclusion)."""
    nuc_mask = labels == nid
    if not nuc_mask.any():
        empty = np.empty(0, dtype=np.intp)
        return empty, empty
    samp_mask = nuc_mask
    nin = _nucleolus_in_nucleus(nucleolus_labels, nid, nuc_mask)
    if nin is not None:
        nucleoplasm = nuc_mask & (~nin)
        if nucleoplasm.any():
            samp_mask = nucleoplasm
    return np.where(samp_mask)


def _mean_enrichment_patch(
    qki_2d: np.ndarray,
    labels: np.ndarray,
    nucleolus_labels: Optional[np.ndarray],
    centers_xy_by_nucleus: Dict[int, Any],
    half_px: int,
    *,
    normalize_by: str = "nucleus_mean",
) -> tuple:
    """Average per-nucleus-normalized QKI crops centred on the given positions.

    For each nucleus ``nid`` in ``centers_xy_by_nucleus`` (``nid -> (n, 2)``
    ``(x_px, y_px)`` array, or a DataFrame with ``x_px``/``y_px``): extract a
    ``(2*half+1, 2*half+1)`` QKI crop at each centre, divide it by that nucleus's
    mean QKI over the sampling region (nucleolus-excluded, matching the null), and
    average every such enrichment crop across all centres in all nuclei.

    ``normalize_by='nucleus_mean'`` (the only supported mode) puts every crop in
    enrichment units (QKI / nucleus-mean), so crops from different sections/lasers
    are poolable. Centres inside a nucleolus are dropped (mirrors the core's spot
    drop). Returns ``(mean_patch, n_used)``; ``n_used == 0`` -> an all-NaN patch.
    """
    if normalize_by != "nucleus_mean":
        raise ValueError(
            f"unsupported normalize_by={normalize_by!r}; only 'nucleus_mean'"
        )
    qki_f = np.asarray(qki_2d, dtype=np.float64)
    labels = np.asarray(labels)
    half = int(half_px)
    side = 2 * half + 1
    H, Wd = qki_f.shape
    acc = np.zeros((side, side), dtype=np.float64)
    n_used = 0

    for nid_raw, sub in centers_xy_by_nucleus.items():
        try:
            nid = int(nid_raw)
        except (TypeError, ValueError):
            continue
        if nid < 1:
            continue
        nys, nxs = _nucleus_sampling_pixels(labels, nucleolus_labels, nid)
        if nys.size == 0:
            continue
        nuc_mean = float(qki_f[nys, nxs].mean())
        if not (nuc_mean > 0):
            continue
        if isinstance(sub, pd.DataFrame):
            sub = sub[["x_px", "y_px"]].astype(float).to_numpy()
        sub = np.asarray(sub, dtype=float).reshape(-1, 2)
        if sub.size == 0:
            continue
        nuc_mask = labels == nid
        nin = _nucleolus_in_nucleus(nucleolus_labels, nid, nuc_mask)
        for (x, y) in sub:
            cy = int(np.clip(round(y), 0, H - 1))
            cx = int(np.clip(round(x), 0, Wd - 1))
            if nin is not None and nin[cy, cx]:
                continue  # observed spot inside a nucleolus -> dropped
            acc += _crop_at(qki_f, cy, cx, half) / nuc_mean
            n_used += 1

    if n_used == 0:
        return np.full((side, side), np.nan, dtype=np.float64), 0
    return acc / n_used, n_used


def _matched_random_centers(
    labels: np.ndarray,
    nucleolus_labels: Optional[np.ndarray],
    centers_xy_by_nucleus: Dict[int, Any],
    *,
    seed: int = 0,
) -> Dict[int, np.ndarray]:
    """Per-nucleus matched random control for ``_mean_enrichment_patch``.

    For each nucleus draws the SAME number of random in-nucleus
    (nucleolus-excluded) positions as it has MIAT centres — the SAME sampling
    region the engine's null uses — with a dedicated seeded RNG. Nuclei are
    iterated in ascending id order so the seed -> draw sequence is deterministic.
    Returns ``nid -> (n, 2)`` ``(x_px, y_px)`` arrays, directly consumable by
    ``_mean_enrichment_patch``.
    """
    rng = np.random.default_rng(int(seed))
    labels = np.asarray(labels)
    H, Wd = labels.shape
    out: Dict[int, np.ndarray] = {}
    for nid_raw in sorted(centers_xy_by_nucleus, key=lambda k: int(k)):
        nid = int(nid_raw)
        if nid < 1:
            continue
        sub = centers_xy_by_nucleus[nid_raw]
        if isinstance(sub, pd.DataFrame):
            sub = sub[["x_px", "y_px"]].astype(float).to_numpy()
        sub = np.asarray(sub, dtype=float).reshape(-1, 2)
        nys, nxs = _nucleus_sampling_pixels(labels, nucleolus_labels, nid)
        if nys.size == 0 or sub.size == 0:
            continue
        # match the count AFTER dropping observed spots that fall in a nucleolus
        nuc_mask = labels == nid
        nin = _nucleolus_in_nucleus(nucleolus_labels, nid, nuc_mask)
        if nin is not None:
            cy = np.clip(np.rint(sub[:, 1]).astype(np.intp), 0, H - 1)
            cx = np.clip(np.rint(sub[:, 0]).astype(np.intp), 0, Wd - 1)
            n_sp = int((~nin[cy, cx]).sum())
        else:
            n_sp = int(sub.shape[0])
        if n_sp == 0:
            continue
        idx = rng.integers(0, nys.size, size=n_sp)
        out[nid] = np.column_stack([nxs[idx], nys[idx]]).astype(float)  # (x, y)
    return out


def _central_disk_mean(patch: np.ndarray, r: float = 3.0) -> float:
    """Mean of ``patch`` over the central disk of radius ``r`` px (reuses the
    engine ``_disk_stencil`` so the disk matches the coloc statistic's disk)."""
    h = patch.shape[0] // 2
    dy, dx = _disk_stencil(float(r))
    ys = np.clip(h + dy, 0, patch.shape[0] - 1)
    xs = np.clip(h + dx, 0, patch.shape[1] - 1)
    return float(np.asarray(patch)[ys, xs].mean())


# ===========================================================================
# I/O SHELL
# ===========================================================================
def _first_present(d: dict, *keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _read_label_tiff(path: Path) -> np.ndarray:
    """Read a saved 16-bit label TIFF back to an int32 label array (companion to
    ``output.save_label_tiff``)."""
    try:
        import tifffile
        arr = tifffile.imread(str(path))
    except Exception:
        from PIL import Image
        arr = np.asarray(Image.open(str(path)))
    return np.asarray(arr).astype(np.int32)


def _resolve_plain_vsi(staging_dir: Path, image_name: str) -> Optional[Path]:
    """Find the PLAIN ``<image_name>`` VSI under the staging tree (recursive),
    NEVER a ``(decon)`` variant. ``image_name`` is the exact PLAIN basename from
    ``per_image_summary.image`` (it does not contain '(decon)')."""
    target = image_name
    matches = [
        p for p in staging_dir.rglob(target)
        if p.is_file() and "(decon)" not in p.name
    ]
    if matches:
        # shortest path wins (avoid nested _temp copies if any)
        matches.sort(key=lambda p: len(str(p)))
        return matches[0]
    return None


def _resolve_mask_path(masks_dir: Path, image_name: str, condition: str) -> Optional[Path]:
    """Resolve the saved ``*__nuclei_label_mask.tif`` for this image.

    The mask base is the image stem with spaces->underscores and the run's common
    leading prefix stripped (e.g. ``UD-MIAT-FISH-QKI-IF-g2-no Dox_03`` ->
    ``g2-no_Dox_03``). We match by: the sanitized image stem ENDSWITH the mask's
    middle ``__``-token, preferring the candidate whose leading token matches the
    sanitized condition and whose middle token is longest (most specific).
    """
    from .output import sanitize_condition_for_filename

    stem_us = Path(image_name).stem.replace(" ", "_")
    cond_san = sanitize_condition_for_filename(condition)
    suffix = "__nuclei_label_mask.tif"
    best = None
    best_score = -1
    for c in sorted(masks_dir.glob("*" + suffix)):
        core = c.name[: -len(suffix)]
        parts = core.split("__")
        if len(parts) >= 2:
            mid = parts[-1]
            condtok = "__".join(parts[:-1])
        else:
            mid, condtok = core, ""
        if not mid:
            continue
        if stem_us.endswith(mid) or mid in stem_us:
            score = len(mid) + (10_000 if (cond_san and condtok == cond_san) else 0)
            if score > best_score:
                best_score = score
                best = c
    return best


def _build_nucleolus_labels(cfg, labels, dapi_2d, voxel_xy_nm):
    """Recompute the nucleolus label image over the SAVED labels — only when the
    run used nucleolus exclusion (mirrors rna_rna's precompute gate). Returns the
    label image, or None when exclusion was off / detection fails."""
    excl = bool(getattr(cfg.foci, "exclude_nucleolus_from_partner_null", False))
    ncfg = getattr(cfg, "nucleolus", None)
    enabled = ncfg is not None and bool(getattr(ncfg, "enabled", False))
    if not (excl and enabled) or int(labels.max()) <= 0:
        return None
    try:
        from .nucleolus import NucleolusParams, detect_nucleoli
        params = NucleolusParams(
            intra_nuclear_percentile=float(ncfg.intra_nuclear_percentile),
            min_area_um2=float(ncfg.min_area_um2),
            max_area_frac_of_nucleus=float(ncfg.max_area_frac_of_nucleus),
            closing_radius_px=int(ncfg.closing_radius_px),
            min_border_distance_px=int(getattr(ncfg, "min_border_distance_px", 3)),
        )
        pix_um = float(voxel_xy_nm) / 1000.0 if voxel_xy_nm else 0.13
        return detect_nucleoli(labels, dapi_2d, pixel_size_um=pix_um, params=params)
    except Exception as exc:  # noqa: BLE001
        print(f"  WARN: nucleolus recompute failed ({type(exc).__name__}: {exc}); "
              f"null uses whole nucleus.")
        return None


def _render_mean_enrichment_montage(
    miat_patch: np.ndarray,
    rand_patch: np.ndarray,
    out_png: Path,
    *,
    n_miat: int,
    n_rand: int,
    example_crops: Optional[List[Dict[str, Any]]] = None,
    example_vmin: Optional[float] = None,
    example_vmax: Optional[float] = None,
    condition_label: str = "",
    pixel_um: Optional[float] = None,
    disk_r_px: float = 3.0,
) -> None:
    """Headline figure: the MEAN per-nucleus-normalized QKI enrichment patch at
    MIAT spots vs at matched random in-nucleus positions, two heatmaps on ONE
    shared colorbar centred at 1.0 (enrichment = QKI / nucleus-mean). A modest
    ~10% central enrichment that is invisible in a single raw crop is obvious in
    the mean. A small strip of example RAW crops (from a single image, so the
    absolute brightness is comparable) sits below for context. 600 DPI.

    ``miat_patch`` / ``rand_patch`` are the ``(side, side)`` mean enrichment
    patches from :func:`_mean_enrichment_patch` (already pooled across images,
    valid because each crop was per-nucleus normalized).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
    from matplotlib import gridspec

    side = int(miat_patch.shape[0])
    half = side // 2

    # Symmetric-around-1.0 scale from a robust deviation (ignore a single hot
    # pixel via the 99th percentile; floor so a near-flat pair still shows color).
    both = np.concatenate([np.asarray(miat_patch).ravel(),
                           np.asarray(rand_patch).ravel()])
    both = both[np.isfinite(both)]
    maxdev = float(np.percentile(np.abs(both - 1.0), 99)) if both.size else 0.1
    maxdev = max(maxdev, 0.05)
    vmin_e, vmax_e = 1.0 - maxdev, 1.0 + maxdev
    norm = TwoSlopeNorm(vcenter=1.0, vmin=vmin_e, vmax=vmax_e)
    cmap = "PuOr"  # diverging, colorblind-safe; neutral at the 1.0 center

    have_examples = bool(example_crops)
    nrows = 2 if have_examples else 1
    height_ratios = [3.0, 1.4] if have_examples else [3.0]
    fig = plt.figure(figsize=(6.4, 3.1 + (1.7 if have_examples else 0.0)))
    gs = gridspec.GridSpec(
        nrows, 1, height_ratios=height_ratios, hspace=0.42,
        top=0.86, bottom=0.10, figure=fig
    )

    # ---- top: the two mean enrichment patches + shared colorbar -----------
    gtop = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs[0], width_ratios=[1, 1, 0.06], wspace=0.18
    )
    ax_m = fig.add_subplot(gtop[0, 0])
    ax_r = fig.add_subplot(gtop[0, 1])
    cax = fig.add_subplot(gtop[0, 2])

    crop_um = (side * pixel_um) if pixel_um else None
    for ax, patch, who, n in ((ax_m, miat_patch, "MIAT", n_miat),
                              (ax_r, rand_patch, "random", n_rand)):
        im = ax.imshow(np.asarray(patch), cmap=cmap, norm=norm,
                       interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        cval = float(np.asarray(patch)[half, half])
        dval = _central_disk_mean(patch, r=disk_r_px)
        ax.set_title(f"mean QKI enrichment\nat {who} (n={n} spots)", fontsize=8.5)
        # annotate centre-pixel + central-disk (r=disk_r_px) enrichment
        ax.text(0.5, -0.10,
                f"center={cval:.2f}x  |  r{disk_r_px:g}px disk={dval:.2f}x",
                transform=ax.transAxes, ha="center", va="top", fontsize=7.5)
        # mark the centre
        ax.plot([half], [half], marker="+", ms=6, mew=0.8, color="0.15")

    cb = fig.colorbar(im, cax=cax)
    cb.set_label("QKI / nucleus-mean (enrichment)", fontsize=7.5)
    cb.ax.tick_params(labelsize=6.5)

    sub = f" - {condition_label}" if condition_label else ""
    fig.suptitle(
        f"QKI enrichment at MIAT vs random in-nucleus{sub}",
        fontsize=10, y=0.985,
    )
    scale = f"crop {side}px" + (f" ~{crop_um:.1f} um" if crop_um else "")
    fig.text(
        0.5, 0.015,
        f"per-nucleus normalized, pooled across images; {scale}",
        ha="center", va="bottom", fontsize=7, color="0.35",
    )

    # ---- bottom: example RAW crops strip (single image -> comparable) ------
    if have_examples:
        magenta = LinearSegmentedColormap.from_list(
            "magenta_k", ["black", "magenta"]
        )
        obs, nul = [], []
        for c in example_crops:
            for oc, nc in zip(c["obs_crops"], c["null_crops"]):
                obs.append(oc)
                nul.append(nc)
        ncol = min(len(obs), 8)
        if ncol > 0:
            obs, nul = obs[:ncol], nul[:ncol]
            gbot = gridspec.GridSpecFromSubplotSpec(
                2, ncol, subplot_spec=gs[1], wspace=0.08, hspace=0.08
            )
            ev0 = example_vmin if example_vmin is not None else float(
                np.percentile(np.asarray(obs), 5))
            ev1 = example_vmax if example_vmax is not None else float(
                np.percentile(np.asarray(obs), 99.5))
            for j in range(ncol):
                for row, crops in ((0, obs), (1, nul)):
                    ax = fig.add_subplot(gbot[row, j])
                    ax.imshow(np.asarray(crops[j]), cmap=magenta,
                              vmin=ev0, vmax=ev1, interpolation="nearest")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if j == 0:
                        ax.set_ylabel("MIAT" if row == 0 else "rand",
                                      fontsize=7)
            fig.text(0.5, gs[1].get_position(fig).y1 + 0.005,
                     "example raw QKI crops (single image; magenta = QKI/561)",
                     ha="center", va="bottom", fontsize=7.5)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close(fig)


def backfill_run(
    run_dir,
    staging_dir=None,
    input_dir=None,
    *,
    do_null_draws: bool = True,
    do_radial: bool = True,
    do_montage: bool = True,
    do_rotation: bool = False,
    seed: int = 0,
    tol_rel: float = 0.02,
    verbose: bool = True,
) -> dict:
    """Re-read QKI pixels from each image's PLAIN VSI and emit the coloc extras.

    Loops images from ``per_image_summary.csv``; resolves the PLAIN VSI under
    ``staging_dir`` (or ``input_dir`` / the run's recorded ``input_dir``);
    rebuilds the cfg from ``run_config.json``; resolves the QKI channel via the
    rna_protein antibody->rna2 shim; RECOMPUTES the analysis z-plane
    deterministically (never reads z from spot_metrics); loads the SAVED nucleus
    label mask; recomputes nucleoli over those labels; pulls MIAT (rna1) spot
    x_px/y_px/nucleus_id from ``spot_metrics.csv``; calls the pure core; writes
    ``coloc_null_draws.csv`` / ``coloc_null_summary.csv`` /
    ``coloc_radial_profile.csv`` and the montage PNG. SELF-VALIDATES against the
    stored ``protein_pooled_*`` columns and prints a PASS/FAIL table.

    Returns a summary dict.
    """
    from . import io as _io  # local import so tests can monkeypatch read_image

    run_dir = Path(run_dir)
    rc_path = run_dir / "run_config.json"
    if not rc_path.exists():
        raise FileNotFoundError(f"no run_config.json in {run_dir}")
    rc = json.loads(rc_path.read_text())
    from ..config.schema import FishsuiteConfig
    cfg = FishsuiteConfig.model_validate(rc["config_resolved"])

    pi_path = run_dir / "per_image_summary.csv"
    sm_path = run_dir / "spot_metrics.csv"
    if not pi_path.exists() or not sm_path.exists():
        raise FileNotFoundError(
            f"run_dir missing per_image_summary.csv or spot_metrics.csv: {run_dir}"
        )
    per_image = pd.read_csv(pi_path)
    spot_metrics = pd.read_csv(sm_path)
    masks_dir = run_dir / "masks"

    # source tree for the PLAIN VSIs
    src = staging_dir or input_dir or rc.get("input_dir")
    if src is None:
        raise ValueError("no staging_dir / input_dir given and run_config has no input_dir")
    src = Path(src)

    # cfg-driven null params (what the engine used)
    disk_px = float(getattr(cfg.foci, "partner_null_disk_px", 3.0))
    n_null = int(getattr(cfg.foci, "partner_null_n", 1000))
    null_seed = int(getattr(cfg.foci, "partner_null_seed", seed))
    radial_bins_um = list(getattr(cfg.foci, "partner_radial_bins_um", None)
                          or [0.25, 0.5, 0.75, 1.0])
    rot_min_ret = float(getattr(cfg.foci, "partner_rotation_min_retention", 0.5))

    # QKI display floor/ceil for the montage (manual_antibody_min/max).
    vmin = _first_present(rc.get("config_resolved", {}).get("output", {}),
                          "manual_antibody_min", "manual_rna2_min")
    vmax = _first_present(rc.get("config_resolved", {}).get("output", {}),
                          "manual_antibody_max", "manual_rna2_max")
    vmin = float(vmin) if vmin is not None else None
    vmax = float(vmax) if vmax is not None else None

    # rna2 shim so channel resolution yields the QKI/antibody index in rna2.
    mode = getattr(cfg.channels, "analysis_mode", "")
    if mode == "rna_protein":
        from .modes.rna_protein import _build_rna2_shim_cfg
        chan_cfg = _build_rna2_shim_cfg(cfg)
    else:
        chan_cfg = cfg
    from .modes.rna_rna import _resolve_channels

    z_start_cfg = cfg.z_stack.start_slice
    z_end_cfg = cfg.z_stack.end_slice
    iw = bool(getattr(cfg.z_stack, "autofocus_intensity_weighted", False))

    all_draws: List[pd.DataFrame] = []
    all_radial: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, Any]] = []
    gate_rows: List[Dict[str, Any]] = []
    all_rot_draws: List[pd.DataFrame] = []
    rot_summary_rows: List[Dict[str, Any]] = []
    # Mean-enrichment montage accumulators. Each per-image patch is already
    # per-nucleus normalized (enrichment units) so pooling across images is valid
    # (no absolute brightness mixed) — accumulate as a spot-weighted sum.
    patch_half = 10                       # 21x21 px ~= 2.7 um at 0.13 um/px
    miat_acc: Optional[np.ndarray] = None
    rand_acc: Optional[np.ndarray] = None
    miat_w = 0
    rand_w = 0
    conditions_seen: set = set()
    last_pix_um: Optional[float] = None
    # Example RAW crops are taken from a SINGLE image only (consistent laser ->
    # comparable absolute brightness); never pooled across sections.
    example_crops: Optional[List[Dict[str, Any]]] = None
    example_vmin: Optional[float] = None
    example_vmax: Optional[float] = None

    for _, prow in per_image.iterrows():
        image_name = str(prow["image"])
        condition = str(prow.get("condition", ""))
        try:
            vsi = _resolve_plain_vsi(src, image_name)
            if vsi is None:
                print(f"  SKIP {image_name}: PLAIN VSI not found under {src}")
                continue
            mask_path = _resolve_mask_path(masks_dir, image_name, condition)
            if mask_path is None:
                print(f"  SKIP {image_name}: nuclei_label_mask not found in {masks_dir}")
                continue

            img = _io.read_image(vsi)
            dapi_idx, rna_idx, rna2_idx = _resolve_channels(chan_cfg, img)

            # z-window with the engine's n_z clamp
            z_start, z_end = z_start_cfg, z_end_cfg
            if z_start is not None and z_start > img.n_z:
                z_start = 1
            if z_end is not None and z_end > img.n_z:
                z_end = img.n_z
            dapi_z, dapi_2d = _io.extract_channel_autofocus_with_idx(
                img, dapi_idx, z_start=z_start, z_end=z_end, intensity_weighted=iw
            )
            qki_2d = _io.extract_channel_at_z(img, rna2_idx, z_1indexed=dapi_z)

            # saved labels (reused so nucleus_id aligns with spot_metrics)
            labels = _read_label_tiff(mask_path)
            if labels.shape != qki_2d.shape:
                print(f"  WARN {image_name}: mask shape {labels.shape} != QKI "
                      f"{qki_2d.shape}; skipping")
                continue

            voxel_xy_nm = float(img.voxel_xy_nm) if img.voxel_xy_nm == img.voxel_xy_nm \
                and img.voxel_xy_nm > 0 else 65.0
            nucleolus_labels = _build_nucleolus_labels(cfg, labels, dapi_2d, voxel_xy_nm)

            # MIAT (rna1) spots for this image, grouped by nucleus_id (>=1)
            sm = spot_metrics[(spot_metrics["image"] == image_name)
                              & (spot_metrics["channel"] == "rna1")]
            spots_by_nid: Dict[int, np.ndarray] = {}
            for nid, grp in sm.groupby("nucleus_id"):
                try:
                    nid = int(nid)
                except (TypeError, ValueError):
                    continue
                if nid < 1:
                    continue
                spots_by_nid[nid] = grp[["x_px", "y_px"]].astype(float).to_numpy()

            pix_um = voxel_xy_nm / 1000.0 if voxel_xy_nm else 0.13
            radial_bins_px = [float(b) / pix_um for b in radial_bins_um if float(b) > 0]

            out = _compute_coloc_extras_for_image(
                qki_2d, labels, spots_by_nid, nucleolus_labels,
                disk_px=disk_px, n_null=n_null, seed=null_seed,
                radial_bins_px=radial_bins_px,
                do_null_draws=do_null_draws, do_radial=do_radial,
                do_montage=do_montage, do_rotation=do_rotation,
                rotation_min_retention=rot_min_ret,
                image=image_name, condition=condition,
            )

            rs = out.get("rotation_summary")
            if rs is not None:
                rot_summary_rows.append({
                    "image": image_name, "condition": condition,
                    "pooled_obs": rs["pooled_obs"],
                    "pooled_rotation_null_mean": rs["pooled_rotation_null_mean"],
                    "pooled_rotation_null_sd": rs["pooled_rotation_null_sd"],
                    "pooled_rotation_enrichment": rs["pooled_rotation_enrichment"],
                    "pooled_rotation_null_z": rs["pooled_rotation_null_z"],
                    "pooled_rotation_p_empirical": rs["pooled_rotation_p_empirical"],
                    "n_nuclei_used": rs["n_nuclei_used"],
                })
            if out.get("rotation_draws_rows") is not None:
                all_rot_draws.append(out["rotation_draws_rows"])

            s = out["null_summary"]
            if s is not None:
                summary_rows.append({
                    "image": image_name, "condition": condition,
                    "pooled_obs": s["pooled_obs"],
                    "pooled_null_mean": s["pooled_null_mean"],
                    "pooled_null_sd": s["pooled_null_sd"],
                    "pooled_null_enrichment": s["pooled_null_enrichment"],
                    "pooled_null_z": s["pooled_null_z"],
                    "pooled_null_p_empirical": s["pooled_null_p_empirical"],
                    "n_nuclei_used": s["n_nuclei_used"],
                })
                gate_rows.append(_gate_compare(prow, s, tol_rel=tol_rel))
            if out["null_draws_rows"] is not None:
                all_draws.append(out["null_draws_rows"])
            if out["radial_rows"]:
                rdf = pd.DataFrame(out["radial_rows"])
                rdf.insert(0, "condition", condition)
                rdf.insert(0, "image", image_name)
                rdf["ring_um"] = [radial_bins_um[i] for i in rdf["ring_idx"]]
                all_radial.append(rdf)
            if do_montage:
                # Headline: pool the per-nucleus-normalized MEAN enrichment patch
                # at MIAT spots vs matched random in-nucleus positions (same seed
                # + nucleolus exclusion as the null).
                mp, mn = _mean_enrichment_patch(
                    qki_2d, labels, nucleolus_labels, spots_by_nid, patch_half,
                    normalize_by="nucleus_mean",
                )
                if mn > 0 and np.isfinite(mp).all():
                    miat_acc = mp * mn if miat_acc is None else miat_acc + mp * mn
                    miat_w += mn
                rc_centers = _matched_random_centers(
                    labels, nucleolus_labels, spots_by_nid, seed=null_seed,
                )
                rp, rn = _mean_enrichment_patch(
                    qki_2d, labels, nucleolus_labels, rc_centers, patch_half,
                    normalize_by="nucleus_mean",
                )
                if rn > 0 and np.isfinite(rp).all():
                    rand_acc = rp * rn if rand_acc is None else rand_acc + rp * rn
                    rand_w += rn
                conditions_seen.add(condition)
                last_pix_um = pix_um
                # example RAW crops from the FIRST contributing image only
                if example_crops is None and out["montage_crops"]:
                    example_crops = out["montage_crops"]
                    example_vmin = (vmin if vmin is not None
                                    else float(np.percentile(qki_2d, 5)))
                    example_vmax = (vmax if vmax is not None
                                    else float(np.percentile(qki_2d, 99.5)))
            if verbose:
                tag = ""
                if s is not None and gate_rows:
                    tag = "PASS" if gate_rows[-1]["pass"] else "FAIL"
                print(f"  OK {image_name}: nuclei_used={s['n_nuclei_used'] if s else 0} {tag}")
        except Exception as exc:  # noqa: BLE001
            import traceback
            print(f"  ERROR {image_name}: {type(exc).__name__}: {exc}")
            if verbose:
                traceback.print_exc()
            continue

    # ---- write outputs ----------------------------------------------------
    written: Dict[str, str] = {}
    if all_draws:
        p = run_dir / "coloc_null_draws.csv"
        pd.concat(all_draws, ignore_index=True).to_csv(p, index=False)
        written["coloc_null_draws"] = str(p)
    if summary_rows:
        p = run_dir / "coloc_null_summary.csv"
        pd.DataFrame(summary_rows).to_csv(p, index=False)
        written["coloc_null_summary"] = str(p)
    if all_radial:
        cols = ["image", "condition", "ring_um", "obs_mean", "null_mean",
                "null_sd", "enrichment", "z", "n_spots"]
        rad = pd.concat(all_radial, ignore_index=True)
        rad = rad[[c for c in cols if c in rad.columns]]
        p = run_dir / "coloc_radial_profile.csv"
        rad.to_csv(p, index=False)
        written["coloc_radial_profile"] = str(p)
    if all_rot_draws:
        p = run_dir / "coloc_rotation_null_draws.csv"
        pd.concat(all_rot_draws, ignore_index=True).to_csv(p, index=False)
        written["coloc_rotation_null_draws"] = str(p)
    if rot_summary_rows:
        p = run_dir / "coloc_rotation_null_summary.csv"
        pd.DataFrame(rot_summary_rows).to_csv(p, index=False)
        written["coloc_rotation_null_summary"] = str(p)
    if do_montage and miat_w > 0 and rand_w > 0:
        miat_patch = miat_acc / miat_w
        rand_patch = rand_acc / rand_w
        _conds = sorted(c for c in conditions_seen if c)
        if len(_conds) > 2:
            cond_label = f"all {len(_conds)} conditions pooled"
        elif _conds:
            cond_label = "+".join(_conds)
        else:
            cond_label = "all conditions"
        png = run_dir / "figures" / "07_coloc" / "79_coloc_qki_montage_at_miat_vs_random.png"
        _render_mean_enrichment_montage(
            miat_patch, rand_patch, png,
            n_miat=int(miat_w), n_rand=int(rand_w),
            example_crops=example_crops, example_vmin=example_vmin,
            example_vmax=example_vmax, condition_label=cond_label,
            pixel_um=last_pix_um, disk_r_px=disk_px,
        )
        written["montage"] = str(png)

    # ---- self-validation gate report -------------------------------------
    gate = _print_gate(gate_rows, tol_rel=tol_rel) if gate_rows else {"max_abs_delta": None,
                                                                      "n_fail": 0,
                                                                      "n_pass": 0}
    return {
        "n_images": int(len(per_image)),
        "n_validated": len(gate_rows),
        "written": written,
        "gate": gate,
    }


def _gate_compare(prow, s, *, tol_rel: float = 0.02) -> Dict[str, Any]:
    """Compare recomputed pooled stats to the stored protein_pooled_* (rna2_*
    fallback) for one image.

    Also computes the per-image ``"pass"`` verdict HERE (single source of truth)
    so the caller's per-image log + ``_print_gate`` agree — PASS when enrichment
    AND obs AND null_mean each reproduce the stored value within ``tol_rel``.
    """
    def stored(*keys):
        for k in keys:
            if k in prow and prow[k] == prow[k]:  # not NaN
                return float(prow[k])
        return float("nan")

    st_obs = stored("protein_pooled_obs_at_rna1_spots", "rna2_pooled_obs_at_rna1_spots")
    st_nm = stored("protein_pooled_null_mean_at_rna1_spots", "rna2_pooled_null_mean_at_rna1_spots")
    st_enr = stored("protein_pooled_enrichment_vs_null_at_rna1_spots",
                    "rna2_pooled_enrichment_vs_null_at_rna1_spots")
    st_z = stored("protein_pooled_null_z_at_rna1_spots", "rna2_pooled_null_z_at_rna1_spots")
    st_p = stored("protein_pooled_null_p_empirical_at_rna1_spots",
                  "rna2_pooled_null_p_empirical_at_rna1_spots")

    d_obs = abs(s["pooled_obs"] - st_obs)
    d_nm = abs(s["pooled_null_mean"] - st_nm)
    d_enr = abs(s["pooled_null_enrichment"] - st_enr)
    d_enr_rel = (d_enr / abs(st_enr)) if st_enr else float("inf")
    d_obs_rel = (d_obs / abs(st_obs)) if st_obs else float("inf")
    d_nm_rel = (d_nm / abs(st_nm)) if st_nm else float("inf")
    return {
        "image": str(prow["image"]),
        "rec_enr": s["pooled_null_enrichment"], "st_enr": st_enr, "d_enr": d_enr,
        "rec_obs": s["pooled_obs"], "st_obs": st_obs, "d_obs": d_obs,
        "rec_nm": s["pooled_null_mean"], "st_nm": st_nm, "d_nm": d_nm,
        "rec_z": s["pooled_null_z"], "st_z": st_z,
        "rec_p": s["pooled_null_p_empirical"], "st_p": st_p,
        # PASS when enrichment AND obs AND null_mean each reproduce within tol_rel.
        "d_enr_rel": d_enr_rel,
        "d_obs_rel": d_obs_rel,
        "d_nm_rel": d_nm_rel,
        "pass": bool(d_enr_rel <= tol_rel and d_obs_rel <= tol_rel and d_nm_rel <= tol_rel),
    }


def _print_gate(gate_rows: List[Dict[str, Any]], *, tol_rel: float = 0.02) -> Dict[str, Any]:
    print("\n" + "=" * 78)
    print("SELF-VALIDATION GATE - recomputed pooled stats vs stored protein_pooled_*")
    print("=" * 78)
    print(f"{'image':<42}{'enr(rec/st)':>16}{'d_enr':>9}{'obs d%':>9}  res")
    n_fail = 0
    max_abs = 0.0
    for g in gate_rows:
        # Reuse the per-image verdict computed by _gate_compare (single source of
        # truth); fall back to recomputing only if a row predates it.
        ok = g.get("pass")
        if ok is None:
            ok = (g["d_enr_rel"] <= tol_rel and g["d_obs_rel"] <= tol_rel
                  and g["d_nm_rel"] <= tol_rel)
            g["pass"] = bool(ok)
        if not ok:
            n_fail += 1
        max_abs = max(max_abs, g["d_enr"], g["d_obs"], g["d_nm"])
        nm = g["image"]
        nm = (nm[:39] + "...") if len(nm) > 42 else nm
        print(f"{nm:<42}{g['rec_enr']:.4f}/{g['st_enr']:.4f}"
              f"{g['d_enr']:>9.4f}{100 * g['d_obs_rel']:>8.2f}%   "
              f"{'PASS' if ok else 'FAIL'}")
    print("-" * 78)
    print(f"images validated: {len(gate_rows)} | PASS: {len(gate_rows) - n_fail} "
          f"| FAIL: {n_fail} | max |d| (obs/null/enr): {max_abs:.4f}")
    if n_fail:
        print("\n" + "!" * 78)
        print(f"!!! WARNING: {n_fail} image(s) FAILED the reproduction gate "
              f"(tol_rel={tol_rel}). The recomputed z-plane / masks / sampling do "
              f"NOT match the trusted run. INSPECT before trusting the backfilled "
              f"artifacts.")
        print("!" * 78)
    else:
        print("ALL images reproduce the trusted run within tolerance. "
              "z-plane + masks + sampling confirmed.")
    print("=" * 78 + "\n")
    return {"max_abs_delta": max_abs, "n_fail": n_fail,
            "n_pass": len(gate_rows) - n_fail}


# ===========================================================================
# CLI
# ===========================================================================
def main(argv: Optional[List[str]] = None) -> int:
    # The gate report is plain ASCII, but reconfigure stdout to UTF-8 (replace
    # on error) so ANY future glyph + the Windows cp1252 console can never crash
    # the report after the CSVs are already written. Guarded for older streams
    # that lack reconfigure().
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser(
        prog="python -m fishsuite.core.coloc_backfill",
        description="Backfill coloc_null_draws / coloc_radial_profile / QKI "
                    "montage onto a trusted MIATxQKI run (CPU, reuse masks).",
    )
    ap.add_argument("--run-dir", required=True, help="trusted run output dir")
    ap.add_argument("--staging", default=None,
                    help="staging tree holding the PLAIN VSIs "
                         "(defaults to input_dir / run_config input_dir)")
    ap.add_argument("--input", default=None, help="alt source dir for VSIs")
    ap.add_argument("--no-null-draws", action="store_true")
    ap.add_argument("--no-radial", action="store_true")
    ap.add_argument("--no-montage", action="store_true")
    ap.add_argument("--rotation", action="store_true",
                    help="ALSO compute the rotation 'proper background' null "
                         "(keep-N constellation redraw) -> coloc_rotation_null_"
                         "summary.csv + coloc_rotation_null_draws.csv. OFF by "
                         "default (opt-in retrofit).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol-rel", type=float, default=0.02)
    args = ap.parse_args(argv)

    res = backfill_run(
        args.run_dir,
        staging_dir=args.staging,
        input_dir=args.input,
        do_null_draws=not args.no_null_draws,
        do_radial=not args.no_radial,
        do_montage=not args.no_montage,
        do_rotation=args.rotation,
        seed=args.seed,
        tol_rel=args.tol_rel,
    )
    print("written:", json.dumps(res["written"], indent=2))
    gate = res.get("gate", {})
    return 1 if gate.get("n_fail", 0) else 0


if __name__ == "__main__":
    sys.exit(main())
