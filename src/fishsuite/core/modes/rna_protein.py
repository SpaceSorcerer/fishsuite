"""rna_protein — RNA-FISH + antibody coloc (Phase-2 stub; reuses rna_only).

Currently runs the rna_only path and adds per-nucleus pixel coloc on the
RNA + antibody channels using ``fishsuite.core.metrics.compute_coloc_metrics``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import io as _io
from .. import metrics as _metrics
from . import register_mode, rna_only as _rna_only


@register_mode("rna_protein")
def run(path, *, condition: str, sec_only: bool, cfg):
    result = _rna_only.run_one(path, condition=condition, sec_only=sec_only, cfg=cfg)
    if sec_only:
        return result
    img = _io.read_image(path)
    one_indexed = bool(cfg.channels.one_indexed)
    def _chan(idx):
        return (idx - 1) if (one_indexed and idx > 0) else idx
    ab_idx = _chan(cfg.channels.antibody)
    if ab_idx < 0:
        auto = _io.autodetect_channels(img)
        ab_idx = auto["ab"]
    if ab_idx < 0:
        return result
    ab_2d = _io.extract_channel(
        img, ab_idx,
        z_mode=cfg.z_stack.mode,
        z_start=cfg.z_stack.start_slice,
        z_end=cfg.z_stack.end_slice,
    )
    if ab_2d.ndim != 2:
        ab_2d = ab_2d.max(axis=0)
    labels = result.qc["labels"]
    rna_2d = result.qc["rna_2d"]
    rows = []
    for nid in range(1, int(labels.max()) + 1):
        mask = labels == nid
        if not mask.any():
            continue
        m = _metrics.compute_coloc_metrics(
            rna_2d[mask].astype(np.float64),
            ab_2d[mask].astype(np.float64),
            thr_mode=cfg.pixel_coloc.threshold_mode,
            k_mad=cfg.pixel_coloc.k_mad,
            percentile=cfg.pixel_coloc.percentile,
        )
        m["nucleus_id"] = nid
        rows.append(m)
    coloc_df = pd.DataFrame(rows)
    if len(coloc_df):
        # Merge into nuclei
        result.nuclei = result.nuclei.merge(
            coloc_df, on="nucleus_id", how="left", suffixes=("", "_coloc")
        )
    return result
