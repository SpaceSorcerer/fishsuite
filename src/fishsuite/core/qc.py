"""Per-image QC flags for fishsuite (ADDITIVE, 2026-06-10).

``compute_qc_flags(res, cfg)`` returns a dict of NEW, INFORMATIONAL QC columns
that the runner merges into each image's ``per_image`` dict before it is
appended to ``per_image_summary.csv``. Computing these in the runner (one place)
keeps every analysis mode (rna_only / rna_rna / rna_protein / ab_ab /
protein_only) consistent without touching any mode's existing per-image keys.

These flags NEVER drop, exclude, or alter any image — they are advisory only.
Nothing here changes an existing column, default, or numeric result.

Emitted columns
---------------
- ``qc_frac_saturated_<role>`` for each analysed 2D plane present in ``res.qc``
  (roles: dapi / rna / rna2 / antibody). Fraction of pixels at/above the
  near-full-scale cutoff (``0.999 * dtype_max``). Roles whose plane is absent
  are simply not emitted (no spurious NaN column).
- ``qc_focus_score`` — DAPI focus sharpness (variance of the Laplacian of the
  mean-normalised DAPI plane). NaN on failure. Higher = sharper.
- ``qc_n_nuclei`` — mirror of ``per_image['n_nuclei']`` (or ``len(res.nuclei)``).
- ``qc_low_nuclei`` (bool) — ``n_nuclei < cfg.qc.qc_min_nuclei``.
- ``qc_zero_spot`` (bool) — image had 0 detected RNA spots.
- ``qc_flags`` (str) — comma-joined active flag names ("" if clean).
- ``qc_pass`` (bool) — True iff ``qc_flags`` is empty.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np


# Role -> candidate qc-dict keys holding the analysed 2D plane for that role.
# First present key wins. (rna_protein stashes the antibody plane under both
# "antibody_2d" and "rna2_2d"; we list antibody first so it is labelled as
# antibody when present.)
_ROLE_PLANE_KEYS = {
    "dapi": ("dapi_2d",),
    "rna": ("rna_2d",),
    "rna2": ("rna2_2d",),
    "antibody": ("antibody_2d",),
}


def _saturated_fraction(plane: np.ndarray, sat_cut: float) -> float:
    """Fraction of pixels >= sat_cut in a 2D plane. Defensive."""
    arr = np.asarray(plane)
    if arr.size == 0:
        return float("nan")
    return float((arr >= sat_cut).sum()) / float(arr.size)


def _focus_score(plane: np.ndarray) -> float:
    """Variance of the Laplacian of the mean-normalised plane (sharpness)."""
    try:
        from scipy.ndimage import laplace

        arr = np.asarray(plane).astype(np.float64)
        m = arr.mean()
        if not np.isfinite(m) or m == 0:
            return float("nan")
        norm = arr / m
        return float(np.var(laplace(norm)))
    except Exception:
        return float("nan")


def compute_qc_flags(res: Any, cfg: Any, dtype_max: int = 65535) -> Dict[str, Any]:
    """Compute additive per-image QC flags from a mode ``ImageResult``.

    Reads ``res.per_image`` (dict), ``res.qc`` (dict of analysed planes),
    ``res.spots`` (DataFrame), ``res.nuclei`` (DataFrame). Robust to missing
    pieces; callers wrap this in try/except so a failure can never abort a run.

    Parameters
    ----------
    res : object
        Mode result with ``.per_image`` / ``.qc`` / ``.spots`` / ``.nuclei``.
    cfg : FishsuiteConfig
        Used for ``cfg.qc.qc_min_nuclei``, ``cfg.qc.qc_saturated_frac``,
        ``cfg.qc.qc_min_focus_score``.
    dtype_max : int
        Full-scale value of the analysed planes (uint16 -> 65535).

    Returns
    -------
    dict
        New QC columns (see module docstring). Safe to ``.update()`` onto an
        existing per_image dict — all keys are ``qc_`` prefixed and new.
    """
    out: Dict[str, Any] = {}

    qc_cfg = getattr(cfg, "qc", None)
    min_nuclei = int(getattr(qc_cfg, "qc_min_nuclei", 5)) if qc_cfg is not None else 5
    sat_frac_thr = (
        float(getattr(qc_cfg, "qc_saturated_frac", 0.01)) if qc_cfg is not None else 0.01
    )
    min_focus = (
        float(getattr(qc_cfg, "qc_min_focus_score", 0.0)) if qc_cfg is not None else 0.0
    )

    sat_cut = 0.999 * float(dtype_max)

    per_image = getattr(res, "per_image", None)
    if not isinstance(per_image, dict):
        per_image = {}
    qc = getattr(res, "qc", None)
    if not isinstance(qc, dict):
        qc = {}

    # ---- n_nuclei -------------------------------------------------------
    n_nuclei = per_image.get("n_nuclei")
    if n_nuclei is None:
        try:
            n_nuclei = int(len(res.nuclei))
        except Exception:
            n_nuclei = 0
    try:
        n_nuclei = int(n_nuclei)
    except Exception:
        n_nuclei = 0
    out["qc_n_nuclei"] = n_nuclei

    # ---- spots ----------------------------------------------------------
    try:
        n_spots = int(len(res.spots))
    except Exception:
        n_spots = 0

    # ---- saturation per present role -----------------------------------
    active_flags = []
    for role, keys in _ROLE_PLANE_KEYS.items():
        plane = None
        for k in keys:
            v = qc.get(k)
            if v is not None:
                plane = v
                break
        if plane is None:
            continue
        try:
            frac = _saturated_fraction(plane, sat_cut)
        except Exception:
            frac = float("nan")
        out[f"qc_frac_saturated_{role}"] = frac
        try:
            if np.isfinite(frac) and frac > sat_frac_thr:
                active_flags.append(f"saturated_{role}")
        except Exception:
            pass

    # ---- focus score (DAPI plane) --------------------------------------
    focus = float("nan")
    try:
        dapi_plane = qc.get("dapi_2d")
        if dapi_plane is not None:
            focus = _focus_score(dapi_plane)
    except Exception:
        focus = float("nan")
    out["qc_focus_score"] = focus
    # Focus flags only when a positive threshold is configured (default 0 ->
    # never flags) AND a finite score is available.
    try:
        if min_focus > 0.0 and np.isfinite(focus) and focus < min_focus:
            active_flags.append("low_focus")
    except Exception:
        pass

    # ---- count-based flags ---------------------------------------------
    low_nuclei = bool(n_nuclei < min_nuclei)
    zero_spot = bool(n_spots == 0)
    out["qc_low_nuclei"] = low_nuclei
    out["qc_zero_spot"] = zero_spot
    if low_nuclei:
        active_flags.append("low_nuclei")
    if zero_spot:
        active_flags.append("zero_spot")

    # ---- summary --------------------------------------------------------
    qc_flags = ",".join(active_flags)
    out["qc_flags"] = qc_flags
    out["qc_pass"] = bool(qc_flags == "")

    return out
