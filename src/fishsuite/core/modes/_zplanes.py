"""Shared z-plane extraction and plane cleanup for the RNA analysis modes.

The per-image pass (``run_one``) and the batch pre-scan
(``collect_nuclear_rna_pixels``) of every RNA mode build their 2-D planes
here, so both always read the same pixels from the same z.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .. import io as _io

# file_overrides keys the RNA modes honour; anything else gets a warning.
OVERRIDE_KEYS = ("single_slice", "start_slice", "end_slice", "rna_start_slice", "rna_end_slice")


def _dim(msg: str) -> None:
    try:
        from rich.console import Console as _C
        _C().print(f"  [dim]{msg}[/dim]")
    except Exception:
        pass


def _to_2d(arr: np.ndarray) -> np.ndarray:
    return arr if arr.ndim == 2 else arr.max(axis=0)


def extract_planes(
    img, path, cfg, *, dapi_idx: int, rna_idx: int,
    rna2_idx: Optional[int] = None,
) -> Dict[str, Any]:
    """Return the DAPI / RNA (/ partner) 2-D planes for one image.

    Honours ``z_stack.file_overrides`` (start/end window and the optional
    RNA-only maxproj window) and every autofocus anchor. Single-plane
    autofocus locks all channels to one plane and ignores the RNA window.
    The RNA window, when used, applies to the partner channel too so
    colocalisation always compares pixels from the same z source.
    ``rna2_2d`` is None when ``rna2_idx`` is None.
    """
    zc = cfg.z_stack
    z_mode = zc.mode
    z_start = zc.start_slice
    z_end = zc.end_slice
    img_name = Path(path).name
    ovr = (getattr(zc, "file_overrides", {}) or {}).get(img_name) or {}
    if z_mode == "single" and int(getattr(zc, "single_slice", 0) or 0) > 0:
        # A fixed plane (1-indexed); 0 keeps the start_slice / mid-stack rule.
        z_start = z_end = int(zc.single_slice)
    if ovr:
        if z_mode == "single" and ovr.get("single_slice"):
            z_start = z_end = int(ovr["single_slice"])
        if "start_slice" in ovr:
            z_start = int(ovr["start_slice"])
        if "end_slice" in ovr:
            z_end = int(ovr["end_slice"])
        _dim(f"z-override: {img_name} → start={z_start}, end={z_end}")
        unknown = sorted(set(ovr) - set(OVERRIDE_KEYS))
        if unknown:
            print(f"  WARN: {img_name}: z_stack.file_overrides keys {unknown} "
                  f"are not implemented by this mode and were ignored.")
    rna_zs = ovr.get("rna_start_slice")
    rna_ze = ovr.get("rna_end_slice")
    rna_window = rna_zs is not None and rna_ze is not None
    if rna_window:
        rna_zs = int(rna_zs)
        rna_ze = int(rna_ze)
        if rna_zs > img.n_z:
            rna_zs = 1
        if rna_ze > img.n_z:
            rna_ze = img.n_z
        _dim(f"rna z-override: {img_name} → RNA maxproj [{rna_zs},{rna_ze}]")
    elif ("rna_start_slice" in ovr) != ("rna_end_slice" in ovr):
        print(f"  WARN: {img_name}: rna_start_slice and rna_end_slice must be "
              f"set together; the RNA z-window override was ignored.")
    if z_start is not None and z_start > img.n_z:
        z_start = 1
    if z_end is not None and z_end > img.n_z:
        z_end = img.n_z

    def _rna_window(idx: int) -> np.ndarray:
        return _to_2d(_io.extract_channel(
            img, idx, z_mode="maxproj", z_start=rna_zs, z_end=rna_ze,
        ))

    rna2_2d: Optional[np.ndarray] = None
    dapi_autofocus_z: Optional[int] = None
    z_af_extra: Dict[str, Any] = {}
    if z_mode == "autofocus":
        af_channel = str(getattr(zc, "autofocus_channel", "dapi")).lower()
        af_iw = bool(getattr(zc, "autofocus_intensity_weighted", False))
        if af_channel == "joint":
            # One plane jointly in focus for DAPI + RNA (+ partner).
            reduce = str(getattr(zc, "autofocus_joint_reduce", "product")).lower()
            picked_z, joint_diag = _io.resolve_joint_autofocus_plane(
                img, dapi_idx, rna_idx, partner_idx=rna2_idx,
                z_start=z_start, z_end=z_end,
                intensity_weighted=af_iw, reduce=reduce,
                central_fraction=float(getattr(zc, "focus_central_fraction", 0.0)),
                min_intensity_frac=float(
                    getattr(zc, "focus_min_intensity_frac_of_peak", 0.0)
                ),
                nuclear_mask=bool(getattr(zc, "autofocus_joint_nuclear_mask", False)),
            )
            dapi_autofocus_z = int(picked_z)
            dapi_2d = _io.extract_channel_at_z(img, dapi_idx, z_1indexed=dapi_autofocus_z)
            z_af_extra = {
                "z_autofocus_mode": "joint",
                "z_autofocus_channel_used": "joint",
                "z_plane": int(dapi_autofocus_z),
                "z_autofocus_joint_reduce": reduce,
                "nuclear_mask_used": bool(joint_diag.get("nuclear_mask_used", False)),
                "nuclear_mask_coverage": round(
                    float(joint_diag.get("nuclear_mask_coverage", 0.0)), 4
                ),
            }
            try:
                for cn, cv in joint_diag.get("per_channel_focus_score", {}).items():
                    z_af_extra[f"joint_focus_{cn}"] = round(float(cv), 4)
            except Exception:
                pass
            anchor_desc = "joint"
        elif af_channel not in ("rna", "auto"):
            # Default: DAPI picks the plane, every channel is read there.
            dapi_autofocus_z, dapi_2d = _io.extract_channel_autofocus_with_idx(
                img, dapi_idx, z_start=z_start, z_end=z_end,
                intensity_weighted=af_iw,
            )
            anchor_desc = "DAPI"
        else:
            # RNA-anchored (or auto) single-plane pick.
            picked_z, ch_used, af_diag = _io.resolve_autofocus_plane(
                img, dapi_idx=dapi_idx, rna_idx=rna_idx,
                z_start=z_start, z_end=z_end,
                autofocus_channel=af_channel, intensity_weighted=af_iw,
                auto_rna_quality_min=float(
                    getattr(zc, "autofocus_auto_rna_quality_min", 3.0)
                ),
            )
            dapi_autofocus_z = int(picked_z)
            dapi_2d = _io.extract_channel_at_z(img, dapi_idx, z_1indexed=dapi_autofocus_z)
            z_af_extra = {
                "z_autofocus_mode": af_channel,
                "z_autofocus_channel_used": ch_used,
                "z_plane": int(dapi_autofocus_z),
            }
            if af_channel == "auto":
                q = af_diag.get("rna_quality_score", float("nan"))
                z_af_extra["z_autofocus_rna_quality_score"] = (
                    round(float(q), 4) if q == q else float("nan")
                )
                z_af_extra["z_autofocus_rna_quality_min"] = float(
                    af_diag.get("rna_quality_min", float("nan"))
                )
            anchor_desc = str(ch_used).upper()
        rna_2d = _io.extract_channel_at_z(img, rna_idx, z_1indexed=dapi_autofocus_z)
        if rna2_idx is not None:
            rna2_2d = _io.extract_channel_at_z(img, rna2_idx, z_1indexed=dapi_autofocus_z)
        if z_af_extra and af_channel != "joint":
            # RNA signal quality at the chosen plane.
            try:
                qual = _io.rna_plane_quality(rna_2d)
                z_af_extra["rna_focus_score"] = round(float(qual["focus_score"]), 4)
                z_af_extra["rna_dynamic_range"] = round(float(qual["dynamic_range"]), 4)
            except Exception:
                pass
        _dim(f"z-lock: {img_name} → all channels @ {anchor_desc} plane "
             f"z={dapi_autofocus_z}"
             + (" (RNA z-window override ignored under autofocus)" if rna_window else ""))
    elif z_mode == "autofocus_maxproj":
        (afm_zs, afm_ze), afm_diag, dapi_2d = _io.extract_dapi_focus_window(
            img, dapi_idx,
            metric=zc.focus_metric,
            threshold_frac=float(zc.focus_threshold_frac),
            min_slices=int(zc.focus_window_min_slices),
            max_slices=int(zc.focus_window_max_slices),
            z_start=z_start, z_end=z_end,
            fixed_n_slices=int(getattr(zc, "focus_window_fixed_n_slices", 0)),
            min_intensity_frac_of_peak=float(getattr(zc, "focus_min_intensity_frac_of_peak", 0.0)),
            intensity_weighted=bool(getattr(zc, "autofocus_intensity_weighted", False)),
            central_fraction=float(getattr(zc, "focus_central_fraction", 0.0)),
        )

        def _focus_window(idx: int) -> np.ndarray:
            if rna_window:
                return _rna_window(idx)
            return _io.extract_channel_in_z_range(
                img, idx, z_start_1indexed=afm_zs, z_end_1indexed=afm_ze,
                project="maxproj",
            )

        rna_2d = _focus_window(rna_idx)
        if rna2_idx is not None:
            rna2_2d = _focus_window(rna2_idx)
        _dim(f"[autofocus_maxproj] {img_name}: focus peak at z={afm_diag['peak_z']+1}, "
             f"window=[{afm_zs},{afm_ze}] ({afm_ze - afm_zs + 1} slices)")
    else:
        dapi_2d = _to_2d(_io.extract_channel(
            img, dapi_idx, z_mode=z_mode, z_start=z_start, z_end=z_end,
        ))

        def _global(idx: int) -> np.ndarray:
            if rna_window:
                return _rna_window(idx)
            return _to_2d(_io.extract_channel(
                img, idx, z_mode=z_mode, z_start=z_start, z_end=z_end,
            ))

        rna_2d = _global(rna_idx)
        if rna2_idx is not None:
            rna2_2d = _global(rna2_idx)
    return {
        "z_mode": z_mode,
        "z_start": z_start,
        "z_end": z_end,
        "dapi_autofocus_z": dapi_autofocus_z,
        "_z_af_extra": z_af_extra,
        "dapi_2d": dapi_2d,
        "rna_2d": rna_2d,
        "rna2_2d": rna2_2d,
    }


def mask_dust_specks_planes(cfg, *planes: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Mask large bright artifacts in each RNA plane (no-op when disabled).

    DAPI is never passed here: bright DAPI clusters can be real chromatin.
    """
    speck_min = int(getattr(cfg.foci, "mask_dust_specks_min_size_px", 0) or 0)
    if speck_min <= 0:
        return planes
    try:
        from ..image_preprocess import mask_dust_specks
        thr = float(getattr(cfg.foci, "mask_dust_specks_threshold_mad", 50.0))
        repl = str(getattr(cfg.foci, "mask_dust_specks_replacement", "median"))
        out, counts = [], []
        for plane in planes:
            clean, n_specks, n_px = mask_dust_specks(
                plane, min_speck_size_px=speck_min,
                brightness_threshold_mad=thr, replacement=repl,
            )
            out.append(clean)
            counts.append((n_specks, n_px))
        if any(n for n, _ in counts):
            _dim("dust-speck mask: " + ", ".join(
                f"rna{'' if i == 0 else i + 1} {n} specks ({px} px)"
                for i, (n, px) in enumerate(counts)
            ) + f" replaced w/ {repl}")
        return tuple(out)
    except Exception as exc:
        print(f"  WARN: dust speck masking failed: {type(exc).__name__}: {exc}")
        return planes
