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
    overdetect_cap = (
        float(getattr(qc_cfg, "qc_overdetect_rna1_max_per_nucleus", 300.0))
        if qc_cfg is not None else 300.0
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

    # ---- RNA1 over-detection guard (2026-07-05, ADVISORY) --------------
    # RNA1 spots-per-nucleus. Prefer the mode's own RNA1-specific per-image
    # value (correct for rna_rna/rna_protein where res.spots pools BOTH
    # channels); fall back to the generic mean, then to n_spots/n_nuclei
    # (rna_only). NaN when there are no nuclei (over-detection is undefined).
    spn = per_image.get("mean_spots_per_nucleus_rna1")
    if spn is None:
        spn = per_image.get("mean_spots_per_nucleus")
    if spn is None:
        spn = (float(n_spots) / n_nuclei) if n_nuclei > 0 else float("nan")
    try:
        spn = float(spn)
    except Exception:
        spn = float("nan")
    out["qc_rna1_spots_per_nucleus"] = spn
    overdetect = False
    try:
        if overdetect_cap > 0.0 and np.isfinite(spn) and spn > overdetect_cap:
            overdetect = True
    except Exception:
        overdetect = False
    out["qc_overdetect_rna1"] = bool(overdetect)
    if overdetect:
        active_flags.append("overdetect_rna1")

    # ---- summary --------------------------------------------------------
    qc_flags = ",".join(active_flags)
    out["qc_flags"] = qc_flags
    out["qc_pass"] = bool(qc_flags == "")

    return out


def flag_overdetect_outliers(rows, cfg) -> int:
    """Run-level robust RNA1 over-detection outlier flag (2026-07-05, ADVISORY).

    The per-image :func:`compute_qc_flags` applies an ABSOLUTE spots/nucleus
    cap. This second, complementary pass needs the WHOLE run, so the runner
    calls it once after every image's per_image dict has been collected. It
    flags images whose RNA1 spots-per-nucleus exceeds ``median + k*MAD`` across
    the run AND is above a small-signal floor — catching a single blown-up
    field even when its count sits below the absolute cap.

    Mutates each row dict IN PLACE (never drops/reorders rows):
      * adds ``qc_overdetect_rna1_run_outlier`` (bool) to every row, and
      * when it fires, appends ``overdetect_rna1_outlier`` to that row's
        ``qc_flags`` and sets ``qc_pass`` False.

    Purely advisory — it changes NO detection result. Fully defensive: on any
    problem it leaves rows untouched (adds the column as False) and returns 0.
    Returns the number of images flagged.
    """
    try:
        rows = list(rows)
    except Exception:
        return 0

    qc_cfg = getattr(cfg, "qc", None)
    k = float(getattr(qc_cfg, "qc_overdetect_robust_mad_k", 5.0)) if qc_cfg is not None else 5.0
    floor = (
        float(getattr(qc_cfg, "qc_overdetect_min_per_nucleus_for_outlier", 50.0))
        if qc_cfg is not None else 50.0
    )

    # Seed the column as False on every row up-front (so the column always
    # exists in per_image_summary once this pass runs).
    valid = []
    for r in rows:
        if isinstance(r, dict):
            r.setdefault("qc_overdetect_rna1_run_outlier", False)
            v = r.get("qc_rna1_spots_per_nucleus")
            try:
                v = float(v)
            except Exception:
                v = float("nan")
            if np.isfinite(v):
                valid.append(v)

    # Disabled (k<=0) or too few points to estimate a robust spread.
    if k <= 0.0 or len(valid) < 3:
        return 0

    arr = np.asarray(valid, dtype=float)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    if not np.isfinite(mad) or mad <= 0.0:
        return 0
    cutoff = med + k * 1.4826 * mad

    n_flagged = 0
    for r in rows:
        if not isinstance(r, dict):
            continue
        try:
            v = float(r.get("qc_rna1_spots_per_nucleus"))
        except Exception:
            continue
        if np.isfinite(v) and v > cutoff and v > floor:
            r["qc_overdetect_rna1_run_outlier"] = True
            n_flagged += 1
            # Fold into the human-readable flag summary without duplicating.
            existing = str(r.get("qc_flags", "") or "")
            tags = [t for t in existing.split(",") if t]
            if "overdetect_rna1_outlier" not in tags:
                tags.append("overdetect_rna1_outlier")
            r["qc_flags"] = ",".join(tags)
            r["qc_pass"] = False
    return n_flagged


# Prefix of the per-image spot-rate keys each punctate channel emits. The
# channel suffix is whatever follows it (rna1 / rna2, or protein / antibody once
# rna_protein has relabelled), so discovering channels from the keys keeps this
# working across modes without a hard-coded channel list.
_SPOTS_PER_NUCLEUS_PREFIX = "mean_spots_per_nucleus_"


def _mean_finite(values) -> float:
    arr = np.asarray([v for v in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def spot_callability_channels(rows) -> list:
    """Punctate channel suffixes present in ``rows``, sorted for stable output."""
    found = set()
    for r in rows:
        if not isinstance(r, dict):
            continue
        for k in r.keys():
            if isinstance(k, str) and k.startswith(_SPOTS_PER_NUCLEUS_PREFIX):
                suffix = k[len(_SPOTS_PER_NUCLEUS_PREFIX):]
                if suffix:
                    found.add(suffix)
    return sorted(found)


def flag_spot_callability(rows, cfg) -> list:
    """Run-level SPOT-CALLABILITY diagnostic (2026-08-10, ADVISORY).

    Answers a question the pipeline could always compute but never showed: does
    the spot detector actually DISCRIMINATE in this channel? ``detect_in_sec_only``
    already runs the same detector on the secondary-only (no-probe) control
    images, and both count sets already reach ``per_image_summary``, so the
    comparison was one division away from being visible.

    For each punctate channel this adds three columns to EVERY row (they are
    run-level constants, so a reader does not have to re-derive them per image):

      * ``spot_rate_sample_per_nucleus_<ch>``   mean over sample images
      * ``spot_rate_seconly_per_nucleus_<ch>``  mean over secondary-only images
      * ``spot_rate_signal_to_control_<ch>``    the ratio of the two

    A ratio near 1 means the detector finds as many "spots" in the no-probe
    control as in the sample — the channel has no thresholdable object, and any
    mask-based colocalization metric computed from it (Manders, ICQ, Jaccard,
    Dice) is measuring textured background. Threshold-free correlation plus a
    rotation null is the honest read for such a channel.

    Returns a list of human-readable warning strings (empty when everything is
    above ``foci.min_spot_signal_to_control``, or when the diagnostic does not
    apply). It NEVER changes a detection result and NEVER reinterprets a channel:
    silently switching a user's declared punctate channel to a diffuse treatment
    would be a worse failure than the one it is warning about. Fully defensive —
    on any problem it returns no warnings and leaves rows untouched.
    """
    try:
        rows = list(rows)
    except Exception:
        return []

    foci_cfg = getattr(cfg, "foci", None)
    if foci_cfg is None:
        return []
    # Without detect_in_sec_only the control images skip detection entirely and
    # report a structural zero, which would make every ratio infinite and the
    # diagnostic meaningless rather than reassuring.
    if not bool(getattr(foci_cfg, "detect_in_sec_only", False)):
        return []
    min_ratio = float(getattr(foci_cfg, "min_spot_signal_to_control", 2.0))

    sample_rows, seconly_rows = [], []
    for r in rows:
        if not isinstance(r, dict):
            continue
        (seconly_rows if bool(r.get("secondary_only")) else sample_rows).append(r)
    if not seconly_rows or not sample_rows:
        return []

    warnings = []
    for ch in spot_callability_channels(rows):
        key = f"{_SPOTS_PER_NUCLEUS_PREFIX}{ch}"
        sample_rate = _mean_finite(r.get(key, float("nan")) for r in sample_rows)
        seconly_rate = _mean_finite(r.get(key, float("nan")) for r in seconly_rows)
        ratio = (
            sample_rate / seconly_rate
            if (np.isfinite(sample_rate) and np.isfinite(seconly_rate)
                and seconly_rate > 0)
            else float("nan")
        )
        for r in rows:
            if isinstance(r, dict):
                r[f"spot_rate_sample_per_nucleus_{ch}"] = sample_rate
                r[f"spot_rate_seconly_per_nucleus_{ch}"] = seconly_rate
                r[f"spot_rate_signal_to_control_{ch}"] = ratio
        if np.isfinite(ratio) and ratio < min_ratio:
            warnings.append(
                f"SPOT CALLABILITY [{ch}]: signal-to-control ratio "
                f"{ratio:.2f} is below foci.min_spot_signal_to_control="
                f"{min_ratio:g} ({sample_rate:.1f} spots/nucleus in sample vs "
                f"{seconly_rate:.1f} in secondary-only). This channel has no "
                f"thresholdable object to mask, so mask-based colocalization "
                f"(Manders, ICQ, Jaccard, Dice) on it is measuring background "
                f"texture. The lever is the ABSOLUTE intensity floor "
                f"min_spot_peak_intensity (per-channel override). Raising "
                f"threshold_multiplier will NOT improve specificity: textured "
                f"secondary-antibody background produces genuine LoG maxima, so "
                f"a relative threshold rescales sample and control together. "
                f"Nothing was changed or switched — if the channel really is "
                f"diffuse, use threshold-free correlation with a rotation null."
            )
    return warnings
