"""if_pub_images — native publication-image renderer for the ``if_intensity`` mode.

Faithful port of the LOCKED panQKI standalone ``pub_images.py``
(``F:\\Image Analysis Work\\MIAT-QKI-Coloc\\WT-QKI-KO_2026_07_01\\_scripts\\pub_images.py``)
folded into fishsuite so the publication images are produced NATIVELY by the mode
(and regenerable from a finished run WITHOUT the GPU) rather than by a one-off
script. The display math is reproduced verbatim; only the parameters are sourced
from ``cfg.if_intensity`` (or CLI flags on the regenerate path) and the plate /
representative wells are derived from the run rather than hardcoded.

CPU-only. Reads VSIs with ``bioio_bioformats`` (no cellpose / DirectML / GPU),
picks a single plane, and renders with matplotlib.

Two image SOURCES (both produced by default)
--------------------------------------------
  * ``single_plane`` — a representative FOV per well from the single-plane
    quantification set (the FOV with the most nuclei per ``per_fov`` counts;
    middle-FOV fallback). Plane used as acquired.
  * ``picked_z``     — the SINGLE best-focus z-plane from a z-stack folder,
    chosen by intensity-weighted focus on DAPI = ``var(laplace(plane)) *
    mean(plane)`` searched ONLY within the central ``pub_z_central_frac`` (0.8 =>
    skip the bottom 10% + top 10%). A SINGLE PICKED PLANE — never a
    max-projection / MIP.

Per representative well per secondary per source
------------------------------------------------
  (a) CHANNEL-PANEL ``..._channels.png``  : [signal in its wavelength LUT] .
      [DAPI blue] . [Merge], each panel with its OWN scalebar.
  (a') Standalone MERGE ``..._merge.png``.
  (b) One per-secondary/source COMPOSITE ``composite_<sec>_<source>.png`` : grid,
      rows = WT primary / KO primary / secondary-only, cols = signal / DAPI /
      Merge, every panel its own scalebar, row labels + column headers.

Locked display conventions (reused, not re-derived)
---------------------------------------------------
  * LUT by wavelength (``if_report._lut``): 647/640 -> yellow, 568/565/561 ->
    magenta, DAPI/405 -> blue.
  * DAPI dimmed by ``DAPI_WEIGHT`` (=0.6) IN THE MERGE ONLY.
  * Signal display floor = the RAISED per-secondary floor
    (``pub_display_floors`` default 647=5000 / 568=3500) so WT cytoplasm reads
    near-zero; ceiling = ``pub_ceiling_pct`` (99.5th) percentile of the
    WT-PRIMARY signal, matched per secondary and per source. The SAME (lo, hi) is
    applied to WT / KO / secondary-only (never per-image auto-contrast).

CRITICAL CHANNEL RULE
---------------------
For a 647 well ONLY the 640-channel signal + DAPI are read/rendered; for a
568/565 well ONLY the 561-channel signal + DAPI. The empty (no-fluorophore)
channel is never read. The routed channel comes from each well's ``qki_channel``.

Human / Homo sapiens.
"""
from __future__ import annotations

import datetime
import json
import platform
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import if_intensity as _ifi
from . import if_report as _rep

# Reuse the LOCKED display conventions (do not duplicate 03's helpers).
_lut = _rep._lut
_norm = _rep._norm
_focus_scores = _rep._focus_scores

DAPI_WEIGHT = 0.6                        # dim DAPI in the merge (locked)
DEFAULT_PUB_FLOORS = {"647": 5000.0, "568": 3500.0}
DEFAULT_CEIL_PCT = 99.5
DEFAULT_Z_CENTRAL_FRAC = 0.8
DEFAULT_SCALEBAR_UM = 20.0


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _alexa_label(sec) -> str:
    return "Alexa-647" if "647" in str(sec) else "Alexa-568"


def _slabel(source: str) -> str:
    return "single-plane" if source == "single_plane" else "picked-z"


def _resolve_source_secondary(mapping, sec, source):
    """Look up a per-secondary / per-(secondary,source) display value.

    Keys may be bare ``"647"`` (applies to BOTH sources) or source-scoped
    ``"647:single_plane"`` (that source only). The MORE-SPECIFIC source-scoped
    key wins. Returns a float, or ``None`` when no key matches (or the value is
    unparseable) so callers can fall back to their own default.
    """
    if not mapping:
        return None
    for key in (f"{sec}:{source}", str(sec)):
        if key in mapping:
            try:
                return float(mapping[key])
            except (TypeError, ValueError):
                return None
    return None


def _open(path):
    from bioio import BioImage
    import bioio_bioformats
    return BioImage(str(path), reader=bioio_bioformats.Reader)


def _pick_scene(img, want_z1=True):
    """Largest >=3-channel scene (Z==1 preferred; fall back to any Z)."""
    best = None
    for sc in img.scenes:
        img.set_scene(sc)
        d = img.dims
        c = d.C if "C" in d.order else 1
        z = d.Z if "Z" in d.order else 1
        area = d.Y * d.X
        ok = (c >= 3) and ((z == 1) if want_z1 else (z >= 1))
        if ok and (best is None or area > best[1]):
            best = (sc, area)
    return best[0] if best else None


def _chan_index(names, key):
    hit = [i for i, n in enumerate(names) if key in str(n)]
    if len(hit) != 1:
        raise RuntimeError(f"channel '{key}' ambiguous/absent in {names}")
    return hit[0]


# ---------------------------------------------------------------------------
# Image reads (CPU) — read ONLY DAPI + the well's own routed signal channel.
# ---------------------------------------------------------------------------
def _read_single_plane(path, dapi_key, qki_key, well, sec):
    img = _open(path)
    sc = _pick_scene(img, want_z1=True) or _pick_scene(img, want_z1=False)
    if sc is None:
        raise RuntimeError(f"no valid >=3ch scene: {path}")
    img.set_scene(sc)
    names = [str(n) for n in (img.channel_names or [])]
    didx = _chan_index(names, dapi_key)
    qidx = _chan_index(names, qki_key)
    arr = img.get_image_data("CYX", T=0, Z=0)
    dapi = np.asarray(arr[didx]).astype(np.float32)
    qki = np.asarray(arr[qidx]).astype(np.float32)
    print(f"      [single_plane] well{well} (sec {sec} -> ch '{qki_key}') "
          f"{Path(path).name}  DAPI ch={didx} signal ch={qidx}")
    return dapi, qki


def _pick_best_plane(path, dapi_key, qki_key, central_frac, well, sec):
    """Single best-focus plane (var(laplace)*mean on DAPI, central band). NO MIP."""
    img = _open(path)
    sc = _pick_scene(img, want_z1=False)
    if sc is None:
        raise RuntimeError(f"no valid >=3ch scene: {path}")
    img.set_scene(sc)
    names = [str(n) for n in (img.channel_names or [])]
    didx = _chan_index(names, dapi_key)
    qidx = _chan_index(names, qki_key)
    dstack = np.asarray(img.get_image_data("ZYX", T=0, C=didx)).astype(np.float32)
    qstack = np.asarray(img.get_image_data("ZYX", T=0, C=qidx)).astype(np.float32)
    nz = dstack.shape[0]
    if nz == 1:
        z0 = 0
    else:
        lo_z = int(nz * (1.0 - central_frac) / 2.0)
        hi_z = nz - lo_z
        if hi_z <= lo_z:
            lo_z, hi_z = 0, nz
        scores = _focus_scores(dstack)
        z0 = lo_z + int(np.argmax(scores[lo_z:hi_z]))
    lo_z = int(nz * (1.0 - central_frac) / 2.0)
    print(f"      [picked_z] well{well} (sec {sec} -> ch '{qki_key}') "
          f"{Path(path).name}  nz={nz} central[{lo_z},{nz - lo_z}) -> picked plane "
          f"z={z0} (single plane, NO MIP)  DAPI ch={didx} signal ch={qidx}")
    return dstack[z0], qstack[z0]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _qki01(qki, lo, hi):
    if hi > lo:
        return np.clip((qki - lo) / (hi - lo), 0, 1)
    return _norm(qki)


def _dapi01(dapi, dapi_lo=None, dapi_hi=None):
    """DAPI display normalization.

    When NEITHER bound is given, per-image min-max (``_norm``) exactly as before
    (byte-identical back-compat). When a fixed bound is supplied, use a fixed
    window; the unset side falls back to the per-image min / max — so passing
    only a fixed ceiling (Brian's ``--dapi-ceiling 8000``) caps over-exposure
    without moving the per-image black point.
    """
    if dapi_lo is None and dapi_hi is None:
        return _norm(dapi)
    lo = float(dapi_lo) if dapi_lo is not None else float(np.min(dapi))
    hi = float(dapi_hi) if dapi_hi is not None else float(np.max(dapi))
    if hi > lo:
        return np.clip((dapi - lo) / (hi - lo), 0, 1)
    return _norm(dapi)


def _add_scalebar(ax, shape, px_um, um):
    h, w = shape
    bar_px = um / px_um if px_um else um
    x0, y0 = w * 0.66, h * 0.93
    import matplotlib.patches as mpatches
    ax.add_patch(mpatches.Rectangle((x0, y0), bar_px, h * 0.013, color="white", ec="none"))
    ax.text(x0 + bar_px / 2, y0 - h * 0.02, f"{um:g} um", color="white",
            ha="center", va="bottom", fontsize=8, fontweight="bold")


def _render_channels(dapi, qki, sec, lo, hi, label, suptitle, out_path, px_um,
                     scalebar_um, dapi_lo=None, dapi_hi=None):
    import matplotlib.pyplot as plt
    q01 = _qki01(qki, lo, hi)
    qki_rgb = _lut(q01, sec)
    dapi01 = _dapi01(dapi, dapi_lo, dapi_hi)
    dapi_rgb = _lut(dapi01, "405")
    merge = np.clip(_lut(q01, sec) + DAPI_WEIGHT * _lut(dapi01, "405"), 0, 1)
    panels = [(qki_rgb, f"{label} ({_alexa_label(sec)})"), (dapi_rgb, "DAPI"), (merge, "Merge")]
    fig, axes = plt.subplots(1, 3, figsize=(4.2 * 3, 4.6), dpi=150)
    for ax, (rgb, ttl) in zip(axes, panels):
        ax.imshow(rgb); ax.axis("off")
        ax.set_title(ttl, fontsize=11, fontweight="bold", color="black", pad=6)
        _add_scalebar(ax, qki.shape, px_um, scalebar_um)
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=1.02)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.01, wspace=0.03)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=600, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out_path}")


def _render_merge(dapi, qki, sec, lo, hi, title, out_path, px_um, scalebar_um,
                  dapi_lo=None, dapi_hi=None):
    import matplotlib.pyplot as plt
    q01 = _qki01(qki, lo, hi)
    merge = np.clip(_lut(q01, sec)
                    + DAPI_WEIGHT * _lut(_dapi01(dapi, dapi_lo, dapi_hi), "405"), 0, 1)
    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    ax.imshow(merge); ax.axis("off")
    ax.set_title(title, fontsize=10, fontweight="bold", color="black", pad=6)
    _add_scalebar(ax, qki.shape, px_um, scalebar_um)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=600, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out_path}")


def _render_composite(rows, sec, lo, hi, label, source_label, out_path, px_um,
                      scalebar_um, his=None, dapi_lo=None, dapi_hi=None):
    """rows = [(row_label, dapi, qki), ...]; cols = signal | DAPI | Merge.

    ``his`` (optional) = a per-row signal-ceiling list matching ``rows`` (used by
    --per-image-ceiling); when None every row uses the shared scalar ``hi``.
    """
    import matplotlib.pyplot as plt
    nrow = len(rows)
    col_titles = [f"{label} ({_alexa_label(sec)})", "DAPI", "Merge"]
    fig, axes = plt.subplots(nrow, 3, figsize=(4.2 * 3, 4.4 * nrow), dpi=150)
    if nrow == 1:
        axes = axes[None, :]
    for r, (row_label, dapi, qki) in enumerate(rows):
        row_hi = his[r] if his is not None else hi
        q01 = _qki01(qki, lo, row_hi)
        dapi01 = _dapi01(dapi, dapi_lo, dapi_hi)
        imgs = [_lut(q01, sec), _lut(dapi01, "405"),
                np.clip(_lut(q01, sec) + DAPI_WEIGHT * _lut(dapi01, "405"), 0, 1)]
        for c in range(3):
            ax = axes[r, c]
            ax.imshow(imgs[c]); ax.axis("off")
            _add_scalebar(ax, qki.shape, px_um, scalebar_um)
            if r == 0:
                ax.set_title(col_titles[c], fontsize=12, fontweight="bold",
                             color="black", pad=8)
        axes[r, 0].text(-0.06, 0.5, row_label, transform=axes[r, 0].transAxes,
                        fontsize=12, fontweight="bold", color="black",
                        ha="right", va="center", rotation=90)
    fig.suptitle(f"{label} immunofluorescence — {_alexa_label(sec)} secondary ({source_label})",
                 fontsize=14, fontweight="bold", y=1.005)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.95, bottom=0.01,
                        wspace=0.03, hspace=0.05)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=600, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out_path}")


# ---------------------------------------------------------------------------
# QC: cytoplasmic vs nuclear signal within the raised display window.
# ---------------------------------------------------------------------------
def _qc_nuclear_dominance(dapi, qki, lo, hi, tag):
    try:
        from skimage.filters import threshold_otsu
        thr = float(threshold_otsu(dapi))
    except Exception:
        thr = float(np.percentile(dapi, 80))
    nuc = dapi >= thr
    cyt = ~nuc
    nuc_med = float(np.median(qki[nuc])) if nuc.any() else float("nan")
    cyt_med = float(np.median(qki[cyt])) if cyt.any() else float("nan")
    span = max(hi - lo, 1e-6)
    nuc_disp = float(np.clip((nuc_med - lo) / span, 0, 1))
    cyt_disp = float(np.clip((cyt_med - lo) / span, 0, 1))
    print(f"      [QC] {tag}: nuclear signal med={nuc_med:8.1f} (disp {nuc_disp:4.2f}) | "
          f"cyto/bg med={cyt_med:8.1f} (disp {cyt_disp:4.2f}) | floor={lo:.0f} ceil={hi:.0f}")
    return dict(tag=tag, nuc_med=nuc_med, cyt_med=cyt_med,
                nuc_disp=nuc_disp, cyt_disp=cyt_disp, lo=lo, hi=hi)


# ---------------------------------------------------------------------------
# Well grouping + FOV picking
# ---------------------------------------------------------------------------
def _group_by_secondary(plate: dict) -> dict:
    """{secondary -> dict(WT=[...primary], KO=[...primary], seconly=[...])}, the
    secondary with more primary wells first (headline)."""
    secs = sorted(
        {str(v["secondary"]) for v in plate.values()},
        key=lambda s: (-sum(1 for vv in plate.values()
                            if str(vv["secondary"]) == s and vv["arm"] == "primary"), s),
    )
    groups = {}
    for sec in secs:
        wt = sorted(w for w, v in plate.items()
                    if v["arm"] == "primary" and v["genotype"] == "WT" and str(v["secondary"]) == sec)
        ko = sorted(w for w, v in plate.items()
                    if v["arm"] == "primary" and v["genotype"] == "KO" and str(v["secondary"]) == sec)
        seconly = sorted(w for w, v in plate.items()
                         if v["arm"] == "seconly" and str(v["secondary"]) == sec)
        groups[sec] = dict(WT=wt, KO=ko, seconly=seconly)
    return groups


def _pick_densest_fov(files, counts):
    scored = [(counts.get(f.name, -1), f) for f in files]
    if any(s >= 0 for s, _ in scored):
        best = max(scored, key=lambda t: t[0])
        print(f"      [pick FOV] {best[1].name} (N={best[0]} nuclei, densest)")
        return best[1]
    mid = files[len(files) // 2]
    print(f"      [pick FOV] {mid.name} (middle FOV; no counts)")
    return mid


def _genotype_disp(g):
    return "WT" if str(g).upper() == "WT" else "KO"


# ---------------------------------------------------------------------------
# Core renderer (config-agnostic; both the live and regenerate paths call this).
# ---------------------------------------------------------------------------
def build_pub_images(out_dir, plate, *, single_plane_dir=None, zstack_dir=None,
                     counts=None, floors=None, ceiling_pct=DEFAULT_CEIL_PCT,
                     sources=None, scalebar_um=DEFAULT_SCALEBAR_UM,
                     z_central_frac=DEFAULT_Z_CENTRAL_FRAC, px_um=0.2167,
                     dapi_key="405", signal_label="signal", ceilings=None,
                     dapi_floor=None, dapi_ceiling=None, per_image_ceiling=False):
    """Render the full publication-image set. Returns a dict summary.

    ``plate`` : {well:int -> dict(genotype, arm, secondary, qki_channel)}.
    ``counts``: {file_basename -> nucleus_count} for densest-FOV picking (opt).

    Display-range control (all optional; defaults reproduce prior behaviour):
      * ``floors``   : signal FLOOR (vmin) per secondary. Keys ``"647"`` (both
                       sources) or ``"647:single_plane"`` (that source only).
      * ``ceilings`` : EXPLICIT signal ceiling (vmax) per secondary/source (same
                       key syntax). Overrides ``ceiling_pct`` for that pair; when
                       absent that pair uses the WT-primary percentile as before.
      * ``dapi_floor`` / ``dapi_ceiling`` : FIXED DAPI vmin/vmax across all
                       panels. When both are None, per-image DAPI normalization
                       (legacy). One-sided is allowed (other side = per-image).
      * ``per_image_ceiling`` : when True, each panel's signal ceiling comes from
                       THAT image's own ``ceiling_pct`` percentile instead of the
                       shared WT-primary ceiling. Slightly non-rigorous (breaks
                       cross-panel comparability); an explicit ``ceilings`` entry
                       still wins over it. Default False (shared display).
    """
    import matplotlib
    matplotlib.use("Agg")

    _ifi._install_asarray_shim()
    out_dir = Path(out_dir)
    floors = {str(k): float(v) for k, v in (floors or DEFAULT_PUB_FLOORS).items()}
    ceilings = {str(k): float(v) for k, v in (ceilings or {}).items()}
    dapi_floor = None if dapi_floor is None else float(dapi_floor)
    dapi_ceiling = None if dapi_ceiling is None else float(dapi_ceiling)
    per_image_ceiling = bool(per_image_ceiling)
    counts = counts or {}
    sources = list(sources or ["single_plane", "picked_z"])

    sp_wells = _ifi._discover_wells(Path(single_plane_dir)) if single_plane_dir else {}
    pz_wells = _ifi._discover_wells(Path(zstack_dir)) if zstack_dir else {}
    groups = _group_by_secondary(plate)

    out_dir.mkdir(parents=True, exist_ok=True)
    written = {"channels": 0, "merge": 0, "composite": 0}
    qc_rows = []
    channel_rule_log = []
    range_log = []   # applied (vmin,vmax) per source/secondary/channel

    for source in sources:
        if source == "single_plane" and not sp_wells:
            print(f"  [WARN] source 'single_plane' requested but no single-plane "
                  f"wells discovered under {single_plane_dir}; skipping")
            continue
        if source == "picked_z" and not pz_wells:
            print(f"  [WARN] source 'picked_z' requested but no z-stack wells "
                  f"discovered under {zstack_dir}; skipping")
            continue
        slab = _slabel(source)
        sdir = out_dir / source
        sdir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== PUBLICATION IMAGES source: {source} ===")

        def _read(well, sec):
            qki_key = str(plate[well]["qki_channel"])
            channel_rule_log.append((source, well, sec, qki_key))
            if source == "single_plane":
                files = sp_wells.get(well)
                if not files:
                    return None
                f = _pick_densest_fov(files, counts)
                return _read_single_plane(f, dapi_key, qki_key, well, sec)
            files = pz_wells.get(well)
            if not files:
                return None
            return _pick_best_plane(files[0], dapi_key, qki_key, z_central_frac, well, sec)

        for sec, grp in groups.items():
            cache = {}
            # WT-primary first -> shared ceiling = ceiling_pct of pooled WT-primary
            wt_pix = []
            for w in grp["WT"]:
                got = _read(w, sec)
                if got is not None:
                    cache[w] = got
                    wt_pix.append(got[1].ravel())

            # ---- signal FLOOR (vmin): per-source-aware, per-secondary fallback ----
            lo = _resolve_source_secondary(floors, sec, source)
            if lo is None:
                lo = float(DEFAULT_PUB_FLOORS.get(str(sec), 0.0))

            # ---- signal CEILING (vmax): explicit > per-image > WT-primary pct ----
            explicit_hi = _resolve_source_secondary(ceilings, sec, source)
            use_per_image_hi = per_image_ceiling and explicit_hi is None
            if explicit_hi is not None:
                hi = explicit_hi
                hi_src = "explicit"
            elif wt_pix:
                pool = np.concatenate(wt_pix)
                hi = float(np.percentile(pool, ceiling_pct))
                if hi <= lo:
                    hi = float(np.percentile(pool, 99.9))
                    if hi <= lo:
                        lo, hi = float(np.percentile(pool, 2)), float(np.percentile(pool, 99.5))
                hi_src = f"WT-primary {ceiling_pct}p"
            else:
                hi = lo + 1.0
                hi_src = "fallback (no WT-primary pixels)"
            if use_per_image_hi:
                hi_src = f"PER-IMAGE {ceiling_pct}p (--per-image-ceiling)"

            def _panel_hi(qki_img, _lo=lo, _hi=hi, _use=use_per_image_hi):
                """Per-panel signal ceiling: this image's own percentile when
                --per-image-ceiling is active (and no explicit ceiling was set),
                else the shared ``hi``."""
                if not _use:
                    return _hi
                r = qki_img.ravel()
                h = float(np.percentile(r, ceiling_pct))
                if h <= _lo:
                    h = float(np.percentile(r, 99.9))
                    if h <= _lo:
                        h = _lo + 1.0
                return h

            dapi_msg = ("per-image" if (dapi_floor is None and dapi_ceiling is None)
                        else f"[{'per-img' if dapi_floor is None else f'{dapi_floor:.0f}'},"
                             f"{'per-img' if dapi_ceiling is None else f'{dapi_ceiling:.0f}'}]")
            hi_msg = "per-image" if use_per_image_hi else f"{hi:.1f}"
            print(f"  [display] secondary {sec} ({slab}): signal floor={lo:.1f} "
                  f"ceiling={hi_msg} ({hi_src}) | DAPI {dapi_msg}")
            range_log.append(dict(
                source=source, secondary=str(sec),
                signal_lo=f"{lo:.1f}",
                signal_hi=("per-image" if use_per_image_hi else f"{hi:.1f}"),
                signal_hi_source=hi_src,
                dapi_lo=("per-image" if dapi_floor is None else f"{dapi_floor:.1f}"),
                dapi_hi=("per-image" if dapi_ceiling is None else f"{dapi_ceiling:.1f}"),
            ))

            # read the remaining wells
            for w in grp["KO"] + grp["seconly"]:
                got = _read(w, sec)
                if got is not None:
                    cache[w] = got

            # per-well channel + merge + QC
            for arm_key, wells in (("WT", grp["WT"]), ("KO", grp["KO"]),
                                   ("seconly", grp["seconly"])):
                for w in wells:
                    if w not in cache:
                        continue
                    dapi, qki = cache[w]
                    v = plate[w]
                    gdisp = _genotype_disp(v["genotype"])
                    if v["arm"] == "primary":
                        ttl = f"{gdisp} — {signal_label}"
                        tok = f"{gdisp}_primary_well{w}"
                    else:
                        ttl = f"Secondary-only ({gdisp})"
                        tok = f"seconly_{gdisp}_well{w}"
                    supt = f"{ttl} — {_alexa_label(sec)}, {slab}"
                    base = f"pub_{sec}_{tok}_{slab}"
                    phi = _panel_hi(qki)
                    _render_channels(dapi, qki, sec, lo, phi, signal_label, supt,
                                     sdir / f"{base}_channels.png", px_um, scalebar_um,
                                     dapi_lo=dapi_floor, dapi_hi=dapi_ceiling)
                    _render_merge(dapi, qki, sec, lo, phi, supt,
                                  sdir / f"{base}_merge.png", px_um, scalebar_um,
                                  dapi_lo=dapi_floor, dapi_hi=dapi_ceiling)
                    written["channels"] += 1
                    written["merge"] += 1
                    qc_rows.append(_qc_nuclear_dominance(dapi, qki, lo, phi,
                                                         f"{sec}/{tok}/{slab}"))

            # composite: first WT primary / first KO primary / first sec-only
            rowdefs = []
            for arm_key, wells, rl in (
                ("WT", grp["WT"], f"WT\n{signal_label}"),
                ("KO", grp["KO"], f"KO\n{signal_label}"),
                ("seconly", grp["seconly"], "Secondary-\nonly"),
            ):
                for w in wells:
                    if w in cache:
                        dapi, qki = cache[w]
                        rowdefs.append((rl, dapi, qki))
                        break
            if len(rowdefs) >= 2:
                comp_his = ([_panel_hi(qki) for (_, _, qki) in rowdefs]
                            if use_per_image_hi else None)
                _render_composite(rowdefs, sec, lo, hi, signal_label, slab,
                                  out_dir / f"composite_{sec}_{slab}.png",
                                  px_um, scalebar_um, his=comp_his,
                                  dapi_lo=dapi_floor, dapi_hi=dapi_ceiling)
                written["composite"] += 1

    _write_provenance(out_dir, floors, ceiling_pct, sources, scalebar_um,
                      z_central_frac, px_um, signal_label, qc_rows, channel_rule_log,
                      ceilings=ceilings, dapi_floor=dapi_floor,
                      dapi_ceiling=dapi_ceiling, per_image_ceiling=per_image_ceiling,
                      range_log=range_log)
    print(f"\n[publication_images] wrote {written['channels']} channel panels, "
          f"{written['merge']} merges, {written['composite']} composites -> {out_dir}")
    return dict(out_dir=str(out_dir), **written,
                sources=[s for s in sources
                         if (s == "single_plane" and sp_wells) or (s == "picked_z" and pz_wells)])


def _write_provenance(out_dir, floors, ceiling_pct, sources, scalebar_um,
                      z_central_frac, px_um, signal_label, qc_rows, channel_rule_log,
                      *, ceilings=None, dapi_floor=None, dapi_ceiling=None,
                      per_image_ceiling=False, range_log=None):
    import matplotlib as _mpl
    out_dir = Path(out_dir)
    ceilings = ceilings or {}
    range_log = range_log or []
    with open(out_dir / "versions.txt", "w", encoding="utf-8") as f:
        f.write(f"# publication_images versions ({datetime.datetime.now().isoformat()})\n")
        f.write(f"python: {sys.version.split()[0]}\nnumpy: {np.__version__}\n")
        f.write(f"matplotlib: {_mpl.__version__}\nplatform: {platform.platform()}\n")
        f.write(f"signal_label={signal_label}; DAPI_WEIGHT={DAPI_WEIGHT}; "
                f"pub_display_floors={floors}\n")
        f.write(f"explicit signal ceilings (override the percentile)={ceilings or '{}'}\n")
        if per_image_ceiling:
            f.write(f"ceiling = PER-IMAGE {ceiling_pct}th pct (--per-image-ceiling; "
                    f"NOT cross-panel comparable) unless an explicit ceiling is set\n")
        else:
            f.write(f"ceiling = {ceiling_pct}th pct of WT-primary signal "
                    f"(per secondary, per source) unless an explicit ceiling is set\n")
        dfloor = "per-image" if dapi_floor is None else dapi_floor
        dceil = "per-image" if dapi_ceiling is None else dapi_ceiling
        f.write(f"DAPI display range: floor={dfloor}; ceiling={dceil} "
                f"(per-image => min-max normalization)\n")
        f.write(f"sources={sources}; scalebar_um={scalebar_um}; pixel_size_um={px_um}\n")
        f.write(f"picked_z = single best-focus plane (var(laplace)*mean on DAPI), "
                f"central {z_central_frac} of planes, NO MIP\n")
        f.write("channel rule: 647->640+DAPI only; 568/565->561+DAPI only\n")
        f.write("600 DPI PNG; CPU only (bioio_bioformats read, no GPU/cellpose)\n")
    with open(out_dir / "channel_rule_log.txt", "w", encoding="utf-8") as f:
        f.write("# routed channel per read (confirms the critical channel rule)\n")
        f.write("source\twell\tsecondary\trouted_signal_channel\n")
        for src, w, sec, ch in channel_rule_log:
            f.write(f"{src}\twell{w}\t{sec}\t{ch}\n")
    if qc_rows:
        with open(out_dir / "qc_nuclear_dominance.txt", "w", encoding="utf-8") as f:
            # --- applied display (vmin,vmax) per source / secondary / channel ---
            f.write("# APPLIED DISPLAY RANGES (vmin,vmax) per source / secondary / channel\n")
            f.write("source\tsecondary\tsignal_lo\tsignal_hi\tsignal_hi_source"
                    "\tdapi_lo\tdapi_hi\n")
            for rl in range_log:
                f.write(f"{rl['source']}\t{rl['secondary']}\t{rl['signal_lo']}\t"
                        f"{rl['signal_hi']}\t{rl['signal_hi_source']}\t"
                        f"{rl['dapi_lo']}\t{rl['dapi_hi']}\n")
            f.write("#\n")
            f.write("# Nuclear vs cyto/background signal within the shared display window.\n")
            f.write("# disp = (median - floor)/(ceil - floor), clipped [0,1]; "
                    "cyto disp ~0 => cytoplasmic signal reads very low (goal met).\n")
            f.write("tag\tnuc_med\tcyt_med\tnuc_disp\tcyt_disp\tfloor\tceil\n")
            for q in qc_rows:
                f.write(f"{q['tag']}\t{q['nuc_med']:.1f}\t{q['cyt_med']:.1f}\t"
                        f"{q['nuc_disp']:.3f}\t{q['cyt_disp']:.3f}\t{q['lo']:.0f}\t{q['hi']:.0f}\n")
    print(f"  [OK] provenance -> {out_dir / 'versions.txt'}, channel_rule_log.txt, "
          f"qc_nuclear_dominance.txt")


# ---------------------------------------------------------------------------
# Live path (called from run_if_batch) + run-context sidecar.
# ---------------------------------------------------------------------------
def write_run_context(output_dir, cfg, input_dir, px_um):
    """Write if_run_context.json so publication images are regenerable (CPU)
    from a finished run without re-reading the config or touching the GPU."""
    ic = cfg.if_intensity
    zdir = ic.pub_zstack_dir or ic.micrograph_zstack_dir or ""
    ctx = dict(
        input_dir=str(input_dir),
        zstack_dir=str(zdir),
        dapi_channel_key=ic.dapi_channel_key,
        pixel_size_um=float(px_um),
        pub_display_floors={str(k): float(v) for k, v in dict(ic.pub_display_floors).items()},
        pub_display_ceilings={str(k): float(v) for k, v in dict(ic.pub_display_ceilings).items()},
        pub_ceiling_pct=float(ic.pub_ceiling_pct),
        pub_dapi_floor=(None if ic.pub_dapi_floor is None else float(ic.pub_dapi_floor)),
        pub_dapi_ceiling=(None if ic.pub_dapi_ceiling is None else float(ic.pub_dapi_ceiling)),
        pub_per_image_ceiling=bool(ic.pub_per_image_ceiling),
        pub_sources=list(ic.pub_sources),
        pub_z_central_frac=float(ic.pub_z_central_frac),
        scalebar_um=float(ic.scalebar_um),
        pub_signal_label=ic.pub_signal_label,
    )
    try:
        with open(Path(output_dir) / "if_run_context.json", "w", encoding="utf-8") as f:
            json.dump(ctx, f, indent=2)
    except Exception as exc:
        print(f"  [WARN] if_run_context.json write failed: {exc}")
    return ctx


def make_pub_images_live(output_dir, plate, per_fov, input_dir, cfg, px_um):
    """Render publication images during a live run (per_fov gives FOV counts)."""
    ic = cfg.if_intensity
    counts = {}
    if per_fov is not None and not per_fov.empty and "file" in per_fov.columns \
            and "nucleus_count" in per_fov.columns:
        counts = {str(r["file"]): int(r["nucleus_count"]) for _, r in per_fov.iterrows()}
    zdir = ic.pub_zstack_dir or ic.micrograph_zstack_dir or ""
    return build_pub_images(
        Path(output_dir) / "publication_images", plate,
        single_plane_dir=input_dir, zstack_dir=(zdir or None), counts=counts,
        floors={str(k): float(v) for k, v in dict(ic.pub_display_floors).items()},
        ceilings={str(k): float(v) for k, v in dict(ic.pub_display_ceilings).items()},
        ceiling_pct=float(ic.pub_ceiling_pct), sources=list(ic.pub_sources),
        scalebar_um=float(ic.scalebar_um), z_central_frac=float(ic.pub_z_central_frac),
        px_um=float(px_um), dapi_key=ic.dapi_channel_key,
        signal_label=ic.pub_signal_label,
        dapi_floor=(None if ic.pub_dapi_floor is None else float(ic.pub_dapi_floor)),
        dapi_ceiling=(None if ic.pub_dapi_ceiling is None else float(ic.pub_dapi_ceiling)),
        per_image_ceiling=bool(ic.pub_per_image_ceiling),
    )


# ---------------------------------------------------------------------------
# Regenerate path (CPU-only; NO GPU, NO re-segmentation).
# ---------------------------------------------------------------------------
def _plate_from_per_well(run_dir: Path) -> dict:
    p = run_dir / "per_well.csv"
    if not p.exists():
        raise FileNotFoundError(f"no per_well.csv in {run_dir}")
    df = pd.read_csv(p)
    plate = {}
    for _, r in df.iterrows():
        w = int(r["well"])
        plate[w] = dict(genotype=str(r["genotype"]), arm=str(r["arm"]),
                        secondary=str(r["secondary"]).split(".")[0],
                        qki_channel=str(r["qki_channel"]).split(".")[0])
    return plate


def _counts_from_per_fov(run_dir: Path) -> dict:
    p = run_dir / "per_fov.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    if "file" not in df.columns or "nucleus_count" not in df.columns:
        return {}
    return {str(r["file"]): int(r["nucleus_count"]) for _, r in df.iterrows()}


def _px_um_from_run(run_dir: Path, fallback=0.2167) -> float:
    for name in ("per_image_summary.csv", "if_intensity_per_image_summary.csv"):
        for p in run_dir.glob(name if name.startswith("if") else f"*{name}"):
            try:
                df = pd.read_csv(p)
                if "voxel_xy_nm" in df.columns:
                    vals = df["voxel_xy_nm"].dropna()
                    if len(vals) and float(vals.iloc[0]) > 0:
                        return float(vals.iloc[0]) / 1000.0
            except Exception:
                pass
    return fallback


def regenerate_pub_images(run_dir, *, staging_dir=None, zstack_dir=None,
                          sources=None, floors=None, ceiling_pct=None,
                          scalebar_um=None, z_central_frac=None, label=None,
                          ceilings=None, dapi_floor=None, dapi_ceiling=None,
                          per_image_ceiling=None, out_subdir="publication_images"):
    """Regenerate the publication images from a COMPLETED if_intensity run,
    CPU-only (no GPU, no re-segmentation). The plate map is rebuilt from the
    run's own per_well.csv; FOV counts from per_fov.csv; the source dirs, floors,
    etc. default from if_run_context.json and are overridable via arguments."""
    run_dir = Path(run_dir)
    ctx = {}
    ctx_path = run_dir / "if_run_context.json"
    if ctx_path.exists():
        try:
            ctx = json.loads(ctx_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  [WARN] could not read if_run_context.json: {exc}")

    plate = _plate_from_per_well(run_dir)
    counts = _counts_from_per_fov(run_dir)

    single_plane_dir = staging_dir or ctx.get("input_dir") or None
    zstack_dir = zstack_dir or ctx.get("zstack_dir") or None
    if zstack_dir and not Path(zstack_dir).is_dir():
        print(f"  [WARN] z-stack dir not found: {zstack_dir}; picked_z will be skipped")
        zstack_dir = None
    if single_plane_dir and not Path(single_plane_dir).is_dir():
        print(f"  [WARN] single-plane dir not found: {single_plane_dir}; "
              f"single_plane will be skipped")
        single_plane_dir = None

    floors = floors or ctx.get("pub_display_floors") or DEFAULT_PUB_FLOORS
    ceilings = ceilings or ctx.get("pub_display_ceilings") or {}
    ceiling_pct = ceiling_pct if ceiling_pct is not None else ctx.get("pub_ceiling_pct", DEFAULT_CEIL_PCT)
    if dapi_floor is None:
        dapi_floor = ctx.get("pub_dapi_floor")
    if dapi_ceiling is None:
        dapi_ceiling = ctx.get("pub_dapi_ceiling")
    if per_image_ceiling is None:
        per_image_ceiling = bool(ctx.get("pub_per_image_ceiling", False))
    sources = sources or ctx.get("pub_sources") or ["single_plane", "picked_z"]
    scalebar_um = scalebar_um if scalebar_um is not None else ctx.get("scalebar_um", DEFAULT_SCALEBAR_UM)
    z_central_frac = z_central_frac if z_central_frac is not None else ctx.get("pub_z_central_frac", DEFAULT_Z_CENTRAL_FRAC)
    label = label or ctx.get("pub_signal_label") or "signal"
    px_um = _px_um_from_run(run_dir, fallback=float(ctx.get("pixel_size_um") or 0.2167))
    dapi_key = ctx.get("dapi_channel_key", "405")

    if not single_plane_dir and not zstack_dir:
        raise ValueError(
            "no source images found: pass --staging (single-plane raw dir) and/or "
            "--zstack (z-stack dir). The run recorded none in if_run_context.json."
        )

    print(f"[if-pub-images] CPU-only regenerate for {run_dir}")
    print(f"  plate wells={sorted(plate)}  single_plane_dir={single_plane_dir}")
    print(f"  zstack_dir={zstack_dir}  sources={sources}  floors={floors}  px_um={px_um}")
    print(f"  ceilings={ceilings or '{}'}  dapi_floor={dapi_floor}  "
          f"dapi_ceiling={dapi_ceiling}  per_image_ceiling={bool(per_image_ceiling)}")
    return build_pub_images(
        run_dir / out_subdir, plate, single_plane_dir=single_plane_dir,
        zstack_dir=zstack_dir, counts=counts, floors=floors,
        ceilings=ceilings, ceiling_pct=float(ceiling_pct), sources=sources,
        scalebar_um=float(scalebar_um), z_central_frac=float(z_central_frac),
        px_um=float(px_um), dapi_key=dapi_key, signal_label=label,
        dapi_floor=dapi_floor, dapi_ceiling=dapi_ceiling,
        per_image_ceiling=bool(per_image_ceiling),
    )
