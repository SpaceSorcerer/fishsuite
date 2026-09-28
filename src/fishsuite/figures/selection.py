"""Objective representative-nucleus rules (ported from 04_select_r2.py, 2026-09-28).

field_rep: arm medians over UPP-defined nuclei, 1.4826*MAD scaling, Euclidean norm over
    (mean UPP, puncta count, mean nuclear QKI); candidates = one field, not touching the border.
coloc_reps: robust z over (mean UPP, Pearson, Spearman, frac puncta QKI+) with all four defined; a metric with
    MAD = 0 is scaled by 1.2533 * mean absolute deviation from the median; top k by norm, both wells.
"""
from __future__ import annotations

import numpy as np

K_REP = ["mean_uniform_position_percentile_qki", "n_miat_spots", "mean_nuclear_qki"]
K_COLOC = ["mean_uniform_position_percentile_qki", "pearson_r_nucleoplasm", "spearman_rho_nucleoplasm", "frac_miat_spots_qki_pos"]


def robust_distance(row, medians, scale):
    return float(np.sqrt(sum(((row[k] - medians[k]) / scale[k]) ** 2 for k in medians)))


def field_rep(nv, field_id):
    ref = nv[nv.mean_uniform_position_percentile_qki.notna()]
    med = {k: float(ref[k].median()) for k in K_REP}
    mad = {k: float(1.4826 * np.median(np.abs(ref[k] - med[k]))) for k in K_REP}
    c = ref[~ref.touches_border & (ref.field == str(field_id))].copy()
    c["d"] = np.sqrt(sum(((c[k] - med[k]) / mad[k]) ** 2 for k in K_REP))
    c = c.sort_values(["d", "nucleus_id"])
    return dict(rule=f"arm medians over UPP-defined nuclei, 1.4826*MAD scaling, candidates = field {field_id}, not touching border",
                medians=med, scale=mad, nucleus_id=int(c.iloc[0].nucleus_id), distance=float(c.iloc[0].d),
                values={k: float(c.iloc[0][k]) for k in K_REP}, top5=c.head(5)[["nucleus_id", "d"] + K_REP].to_dict(orient="records"))


def coloc_reps(nv, k=3):
    r2 = nv.dropna(subset=K_COLOC)
    med = {q: float(r2[q].median()) for q in K_COLOC}
    mad = {q: float(1.4826 * np.median(np.abs(r2[q] - med[q]))) for q in K_COLOC}
    fallback = {}
    for q in K_COLOC:
        if mad[q] == 0:
            mad[q] = float(1.2533 * np.mean(np.abs(r2[q] - med[q])))
            fallback[q] = "1.2533*meanAD (MAD = 0)"
    c2 = r2[~r2.touches_border].copy()
    c2["d"] = np.sqrt(sum(((c2[q] - med[q]) / mad[q]) ** 2 for q in K_COLOC))
    c2 = c2.sort_values(["d", "field", "nucleus_id"])
    return dict(rule="robust z = (x - arm median) / (1.4826*MAD) over nuclei with all four metrics defined; candidates not touching "
                     f"border, both wells; top {k} by Euclidean norm; a metric with MAD = 0 is scaled by 1.2533*mean absolute deviation from the median",
                medians=med, scale=mad, scale_fallback=fallback, n_pool=int(len(r2)), n_candidates=int(len(c2)),
                picks=c2.head(k)[["field", "image", "nucleus_id", "d"] + K_COLOC].to_dict(orient="records"))
