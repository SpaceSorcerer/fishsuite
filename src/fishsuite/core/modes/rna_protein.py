"""rna_protein — RNA-FISH + antibody/protein full two-channel analysis.

2026-05-28 Brian: UPGRADED from the old Phase-2 stub (which ran rna_only +
per-nucleus pixel coloc only). The PROTEIN/antibody channel is now analyzed at
the SAME depth ``rna_rna`` gives its 2nd RNA channel:

  * BigFISH spot detection on the protein channel (Brian wants to try spotting
    XRN2), with per-spot diameter, peak intensity, stratification.
  * Per-nucleus protein metrics: spot counts (nuclear / cytoplasmic), per-spot
    intensity mean / total / median / CV, N/C ratio (mean- and total-based),
    per-cell totals, spot-property distributions.
  * Full RNA × protein colocalization — the SAME spot-level (nn_distance_um,
    paired_at_<X>um) + pixel-coloc threshold machinery rna_rna computes between
    rna and rna2, here between rna and the protein channel.
  * z-lock (protein read at the EXACT DAPI-segmentation plane in autofocus
    mode) + per-image ``file_overrides`` — inherited verbatim from rna_rna.
  * Batch pre-scan that pools + caches BOTH channels (so each image is
    segmented once and the pooled pixel-coloc thresholds are per-channel).

DESIGN (remap, not duplicate): rna_protein reuses rna_rna's mature
two-channel core by building a shallow config SHIM that maps the antibody
channel into rna_rna's ``rna2`` slot (channel index, label, LUT, contrast
min/max, per-channel spot overrides). It calls ``rna_rna.run_one`` with that
shim, then RELABELS the rna2-* output columns / dict keys to PROTEIN semantics
so every CSV / Excel / figure reads e.g. "XRN2" — never "RNA2". This avoids
logic drift: there is ONE two-channel analysis core, and ``rna_rna`` behavior
is completely unchanged (it never sees the shim — the shim is built only inside
this module).

The qc dict keeps the rna2-named keys (``rna2_2d``, ``rna2_pos_mask``,
``spots2``, ...) so the runner's two-channel QC-overlay / walkthrough / mask
rendering — which the runner now also routes for rna_protein — works without a
second rendering path. We ALSO stash ``antibody_2d`` in qc so the publication-
image bundle renders the protein channel with the antibody LUT.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from . import register_mode
from . import rna_rna as _rna_rna
from .rna_only import ImageResult


# ---------------------------------------------------------------------------
# Config shim: map the antibody channel into rna_rna's rna2 slot.
# ---------------------------------------------------------------------------
def _build_rna2_shim_cfg(cfg):
    """Deep-copy ``cfg`` and remap the PROTEIN/antibody channel into the rna2
    slot so ``rna_rna``'s two-channel pipeline analyzes (rna, antibody).

    Copies, into the rna2 slot:
      * ``channels.rna2``      <- ``channels.antibody``
      * ``channels.rna2_label``<- ``channels.antibody_label``
      * ``channels.rna2_lut``  <- ``channels.antibody_lut``
      * ``foci.rna2_overrides``<- ``foci.antibody_overrides``
      * ``output.manual_rna2_min/max`` <- ``output.manual_antibody_min/max``

    and flips ``channels.analysis_mode`` to ``"rna_rna"`` so rna_rna's
    ``_resolve_channels`` accepts the (now-populated) rna2 slot. The shim is a
    SEPARATE pydantic object — the caller's cfg is never mutated, and no
    rna_rna preset is touched.

    NOTE: ``foci.detect_antibody_spots`` rides along verbatim on the deep copy.
    rna_protein passes its value to ``rna_rna.run_one`` via the explicit
    ``rna2_is_antibody=True`` + the flag, so that when it is False the rna2
    (antibody) channel is NOT spot-detected (the diffuse QKI IF carpet fix).
    The flag is consulted ONLY when ``rna2_is_antibody=True`` — plain rna_rna,
    which never builds this shim, is never affected.
    """
    shim = cfg.model_copy(deep=True)
    ch = shim.channels
    ch.rna2 = ch.antibody
    ch.rna2_label = getattr(ch, "antibody_label", "Protein")
    ch.rna2_lut = getattr(ch, "antibody_lut", "green")
    ch.analysis_mode = "rna_rna"
    # Per-channel spot-detection overrides for the protein channel.
    shim.foci.rna2_overrides = cfg.foci.antibody_overrides.model_copy(deep=True)
    # Publication / analysis-floor contrast for the protein channel.
    if getattr(shim.output, "manual_antibody_min", None) is not None:
        shim.output.manual_rna2_min = shim.output.manual_antibody_min
    if getattr(shim.output, "manual_antibody_max", None) is not None:
        shim.output.manual_rna2_max = shim.output.manual_antibody_max
    return shim


# ---------------------------------------------------------------------------
# Output relabeling: rna2-* -> protein-* in the analysis tables / dicts.
# ---------------------------------------------------------------------------
def _relabel_rna2_to_protein(name: str) -> str:
    """Map an rna2-flavored column / key name to its protein-flavored name.

    Handles both word-boundary ``rna2`` tokens (``n_spots_rna2`` ->
    ``n_spots_protein``) and the embedded ``_rna2_`` / ``_rna1_rna2_`` patterns
    used in the cross-channel coloc columns. Names without an rna2 token are
    returned unchanged.
    """
    if "rna2" not in name:
        return name
    out = name
    # Cross-channel overlap / pairing columns: keep "rna1" (the RNA channel)
    # but rename the rna2 half to protein so the meaning is explicit.
    out = out.replace("rna1_rna2", "rna_protein")
    out = out.replace("rna2_rna1", "protein_rna")
    # Generic rna2 -> protein.
    out = out.replace("rna2", "protein")
    return out


def _relabel_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with every rna2-flavored column renamed to protein-*.

    On a name COLLISION (an rna2-renamed column would clash with an existing
    column) the rna2 source is dropped in favor of the existing one — this
    never happens with the current rna_rna schema but keeps the rename safe.
    """
    if df is None or len(df.columns) == 0:
        return df
    rename: Dict[str, str] = {}
    for c in df.columns:
        nc = _relabel_rna2_to_protein(c)
        if nc != c:
            rename[c] = nc
    if not rename:
        return df
    # Drop any source whose target already exists to avoid duplicate columns.
    targets = set(rename.values())
    existing = set(df.columns)
    safe = {k: v for k, v in rename.items() if v not in (existing - set(rename))}
    return df.rename(columns=safe)


def _relabel_dict(d: dict) -> dict:
    """Return a new dict with rna2-flavored keys renamed to protein-*."""
    if not isinstance(d, dict):
        return d
    out: Dict[str, Any] = {}
    for k, v in d.items():
        nk = _relabel_rna2_to_protein(k) if isinstance(k, str) else k
        out[nk] = v
    return out


def _relabel_spot_channel(df: pd.DataFrame) -> pd.DataFrame:
    """Rename the per-spot ``channel`` value 'rna2' -> 'protein' (in place-safe)."""
    if df is None or "channel" not in df.columns or len(df) == 0:
        return df
    df = df.copy()
    df["channel"] = df["channel"].replace({"rna2": "protein"})
    return df


# ---------------------------------------------------------------------------
# Mode entry point.
# ---------------------------------------------------------------------------
def run_one(
    path,
    *,
    condition: str,
    sec_only: bool,
    cfg,
    precomputed_rna_threshold: Optional[float] = None,
    precomputed_rna2_threshold: Optional[float] = None,
    precomputed_labels: Optional[np.ndarray] = None,
    analysis_floors: Optional[Dict[str, Any]] = None,
) -> ImageResult:
    """Run the upgraded rna_protein pipeline on a single image.

    Delegates the full two-channel analysis to ``rna_rna.run_one`` via the
    antibody->rna2 config shim, then relabels rna2-* outputs to protein-*.

    The ``precomputed_rna2_threshold`` kwarg (forwarded by the batch runner)
    is the pooled PROTEIN-channel pixel-coloc threshold — the runner pools the
    antibody channel into the rna2 slot during the batch pre-scan (which uses
    this module's :func:`collect_nuclear_rna_pixels`). ``analysis_floors`` may
    carry an ``"antibody"`` key; we map it into the ``"rna2"`` slot the
    rna_rna core reads.
    """
    shim = _build_rna2_shim_cfg(cfg)

    # Map any antibody analysis-floor into the rna2 slot rna_rna reads.
    floors2 = dict(analysis_floors) if analysis_floors else None
    if floors2 is not None and "antibody" in floors2 and floors2.get("rna2") is None:
        floors2["rna2"] = floors2.get("antibody")

    res = _rna_rna.run_one(
        path,
        condition=condition,
        sec_only=sec_only,
        cfg=shim,
        precomputed_rna_threshold=precomputed_rna_threshold,
        precomputed_rna2_threshold=precomputed_rna2_threshold,
        analysis_floors=floors2,
        precomputed_labels=precomputed_labels,
        # 2026-06-05 Brian: the rna2 slot here IS the antibody/protein channel.
        # When cfg.foci.detect_antibody_spots is False, rna_rna SKIPS rna2
        # spot detection (diffuse QKI IF carpet fix). The flag rides on the
        # shim's foci; rna2_is_antibody=True scopes the skip to rna_protein
        # ONLY, so plain rna_rna can never trigger it.
        rna2_is_antibody=True,
    )

    # ---- Relabel analysis tables: rna2-* -> protein-* ----------------------
    res.nuclei = _relabel_df_columns(res.nuclei)
    res.spots = _relabel_spot_channel(res.spots)
    res.spots = _relabel_df_columns(res.spots)
    res.morphology = _relabel_df_columns(res.morphology)
    res.per_image = _relabel_dict(res.per_image)
    res.thresholds = _relabel_dict(res.thresholds)
    res.extra = _relabel_dict(res.extra)
    res.extra["mode"] = "rna_protein"
    # 2026-06-06 Brian: NATIVE coloc-figure carriers ride in extra as DataFrames.
    # Route their COLUMNS through the same rna2->protein relabel so they stay
    # consistent with per_image_summary (protein_*). Their current column sets
    # carry no rna2 token (this is a defensive no-op), but it future-proofs the
    # schema and keeps the relabel contract uniform.
    for _ck in ("coloc_null_draws", "coloc_radial_profile", "coloc_rotation_null"):
        _cv = res.extra.get(_ck)
        if isinstance(_cv, pd.DataFrame):
            res.extra[_ck] = _relabel_df_columns(_cv)

    # ---- qc: keep rna2-named keys for the runner's two-channel renderers,
    # but ALSO expose antibody_2d so the publication-image bundle paints the
    # protein channel with the antibody LUT. The protein 2D IS rna_rna's
    # rna2_2d (the antibody channel was loaded into the rna2 slot).
    if isinstance(res.qc, dict) and res.qc.get("rna2_2d") is not None:
        res.qc.setdefault("antibody_2d", res.qc.get("rna2_2d"))

    return res


@register_mode("rna_protein")
def run(*args, **kwargs):
    return run_one(*args, **kwargs)


# ---------------------------------------------------------------------------
# Batch pre-scan helper — pool BOTH channels (rna + protein) like rna_rna.
# ---------------------------------------------------------------------------
def collect_nuclear_rna_pixels(path, *, cfg):
    """Return ``(rna_nuclear_pixels, protein_nuclear_pixels, labels)``.

    Mirrors :func:`rna_rna.collect_nuclear_rna_pixels` but resolves the 2nd
    channel as the PROTEIN/antibody channel (via the antibody->rna2 shim), so
    the runner's batch threshold pre-scan pools the antibody channel
    SEPARATELY and caches the (border-excluded) nuclei labels — identical
    z-lock + file_overrides handling, each image segmented once per run.
    """
    shim = _build_rna2_shim_cfg(cfg)
    return _rna_rna.collect_nuclear_rna_pixels(path, cfg=shim)
