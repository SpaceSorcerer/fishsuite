"""SENSITIVITY-ONLY placement nulls for the uniform-position percentile score.

Astra fig1 review F1 (2026-09-24): the uniform exact-footprint placement null
does not control for MIAT and QKI both preferring the same DAPI/chromatin
density or radial zone. Two sensitivity references are added here; neither
replaces the uniform-null headline.

Texture-matched null. For one footprint, the admissible positions are the
existing null's (every translated footprint pixel inside the eligible,
nucleolus-excluded nucleus and on finite QKI). Each admissible position gets
(a) the footprint-mean DAPI at that placement and (b) the normalized radial
position of the placed centre (Euclidean distance to the nuclear boundary /
the nucleus's maximum inscribed distance). Each covariate is converted to a
within-nucleus quantile over that footprint's admissible positions (mid-rank
ECDF). Two matching methods:

- ``strata`` (default): fixed equal-probability bins, n_dapi x n_radial. The
  bins are defined from the admissible set alone, before looking at the
  observed position, so under uniform MIAT placement within the stratum the
  observed position is exchangeable with the K draws and the finite-K
  references (0.5, 21/201 at K = 200) are exact. A stratum with fewer than
  ``min_positions`` admissible positions is NA (SPARSE_STRATUM). There is no
  merging (Astra round-2 F1: an observation-dependent merge breaks
  exchangeability).
- ``knn``: the k admissible positions nearest to the observed position in
  (DAPI quantile, radial quantile) space (Euclidean, the observed position
  included, stable row-major tie order). The candidate set is centred on the
  observation, so the finite-K references are NOMINAL for this method.

The K null placements are drawn uniformly, with replacement, from the matched
set by the unchanged ``exact_footprint_position_null`` sampler restricted to
that candidate domain.

Rotation null. The existing exact-footprint KEEP-N rotation null
(``keep_n_footprint_rotation_null``: constellation rotated about the SPOT
centroid, per-spot redraws for placements leaving the mask) is reused
unchanged, and each spot is ranked against its own rotation draws.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from . import coloc_pixel_metrics as _cpm
from .footprint_null import (
    build_exact_footprint_candidate_domain,
    exact_footprint_position_null,
    keep_n_footprint_rotation_null,
)

SENSITIVITY_LABEL = "SENSITIVITY ONLY (never a replacement headline)"
# Every column carrying one of these prefixes is a sensitivity column: coupling
# drops them before any summary (Astra round-2 F2).
SENSITIVITY_COLUMN_PREFIXES = ("upp_texture_matched_", "upp_rotation_", "texture_",
                               "na_reason_upp_texture_matched", "na_reason_upp_rotation")
METHODS = ("strata", "knn")
CALIBRATION = {"strata": "EXACT_FINITE_K_FIXED_STRATA", "knn": "NOMINAL_KNN_OBSERVATION_CENTRED"}


def is_sensitivity_column(name: str) -> bool:
    return str(name).startswith(SENSITIVITY_COLUMN_PREFIXES)


@dataclass(frozen=True)
class TextureNullParams:
    n_dapi_bins: int = 5
    n_radial_bins: int = 5
    min_positions: int = 20
    method: str = "strata"
    knn_k: int = 200

    def __post_init__(self):
        for name in ("n_dapi_bins", "n_radial_bins", "min_positions", "knn_k"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")

    @property
    def label(self) -> str:
        return (f"strata_{self.n_dapi_bins}x{self.n_radial_bins}" if self.method == "strata"
                else f"knn_k{self.knn_k}")

    def as_dict(self) -> dict:
        return dict(method=self.method, n_dapi_bins=int(self.n_dapi_bins),
                    n_radial_bins=int(self.n_radial_bins), min_positions=int(self.min_positions),
                    knn_k=int(self.knn_k), label=self.label, calibration=CALIBRATION[self.method])


def admissible_centers(footprint, placement_mask, partner) -> np.ndarray:
    """Centres (row-major) whose whole translated footprint lies in
    ``placement_mask`` on finite ``partner`` pixels (the rule of
    ``exact_footprint_position_null``)."""
    mask = np.asarray(placement_mask, dtype=bool)
    values = np.asarray(partner, dtype=float)
    centers = np.argwhere(mask).astype(np.intp)
    height, width = mask.shape
    keep = np.ones(centers.shape[0], dtype=bool)
    for dy, dx in zip(np.asarray(footprint.dy_px, int), np.asarray(footprint.dx_px, int)):
        y, x = centers[:, 0] + dy, centers[:, 1] + dx
        inside = (y >= 0) & (y < height) & (x >= 0) & (x < width)
        ok = np.zeros_like(keep)
        ok[inside] = mask[y[inside], x[inside]] & np.isfinite(values[y[inside], x[inside]])
        keep &= ok
    return centers[keep]


def footprint_means_at(image, centers, footprint) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    total = np.zeros(centers.shape[0])
    dys, dxs = np.asarray(footprint.dy_px, int), np.asarray(footprint.dx_px, int)
    for dy, dx in zip(dys, dxs):
        total += image[centers[:, 0] + dy, centers[:, 1] + dx]
    return total / dys.size


def normalized_radial_map(nucleus_mask) -> np.ndarray:
    """Distance to the nuclear boundary / max inscribed distance; 0 outside.
    The image edge counts as boundary (the mask is padded with background)."""
    mask = np.asarray(nucleus_mask, dtype=bool)
    if not mask.any():
        return np.zeros(mask.shape)
    distance = ndimage.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
    return np.where(mask, distance / distance.max(), 0.0)


def quantile_bins(values, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Mid-rank within-set quantile q = (#< + 0.5 #=)/n and bin floor(q * n_bins).
    Tied values share q and bin."""
    values = np.asarray(values, dtype=float)
    order = np.sort(values)
    below = np.searchsorted(order, values, side="left")
    upto = np.searchsorted(order, values, side="right")
    q = (below + 0.5 * (upto - below)) / values.size
    return q, np.minimum((q * n_bins).astype(int), n_bins - 1)


def select_stratum(dapi_bin, radial_bin, observed_index: int, *, min_positions: int):
    """Fixed stratum of the observed position: (members | None, NA reason).
    No merge: a sparse stratum is NA (SPARSE_STRATUM)."""
    dapi_bin, radial_bin = np.asarray(dapi_bin), np.asarray(radial_bin)
    members = (dapi_bin == dapi_bin[observed_index]) & (radial_bin == radial_bin[observed_index])
    if members.sum() >= min_positions:
        return members, ""
    return None, "SPARSE_STRATUM"


def select_knn(dapi_q, radial_q, observed_index: int, *, k: int):
    """k nearest admissible positions in (DAPI quantile, radial quantile) space,
    observed included; ties broken by row-major order. (members | None, reason,
    max covariate distance in the set)."""
    points = np.column_stack((dapi_q, radial_q))
    if points.shape[0] < k:
        return None, "KNN_SPARSE", float("nan")
    distance = np.hypot(*(points - points[observed_index]).T)
    order = np.argsort(distance, kind="stable")[:k]
    members = np.zeros(points.shape[0], dtype=bool)
    members[order] = True
    return members, "", float(distance[order].max())


def _spot_na(reason, **extra):
    row = dict(upp_texture_matched_qki=np.nan, texture_dapi_footprint_mean=np.nan,
               texture_dapi_quantile=np.nan, texture_dapi_bin=-1, texture_radial_norm=np.nan,
               texture_radial_quantile=np.nan, texture_radial_bin=-1, texture_stratum_n_positions=0,
               texture_null_mean_dapi_quantile=np.nan, texture_null_mean_radial_quantile=np.nan,
               texture_match_max_distance=np.nan, na_reason_upp_texture_matched_spot=reason)
    row.update(extra)
    return row


def texture_matched_spot_scores(qki, dapi, nucleus_mask, region, footprints, observed, *,
                                n_null: int, params: TextureNullParams,
                                rng: np.random.Generator, tie_rng: np.random.Generator,
                                return_draw_centers: bool = False) -> list[dict]:
    """One row per footprint (caller order); draws from ``rng`` in that order.
    ``return_draw_centers`` adds ``_matched_centers`` / ``_draw_centers`` (test hook)."""
    radial = normalized_radial_map(nucleus_mask)
    rows = []
    for fp, obs in zip(footprints, observed):
        centers = admissible_centers(fp, region, qki)
        if centers.shape[0] == 0:
            rows.append(_spot_na("NO_DOMAIN"))
            continue
        hit = np.flatnonzero((centers[:, 0] == fp.center_y_px) & (centers[:, 1] == fp.center_x_px))
        if hit.size != 1:
            rows.append(_spot_na("OBSERVED_NOT_ADMISSIBLE"))
            continue
        index = int(hit[0])
        dapi_means = footprint_means_at(dapi, centers, fp)
        radial_values = radial[centers[:, 0], centers[:, 1]]
        dq, dbin = quantile_bins(dapi_means, params.n_dapi_bins)
        rq, rbin = quantile_bins(radial_values, params.n_radial_bins)
        covariates = dict(texture_dapi_footprint_mean=float(dapi_means[index]),
                          texture_dapi_quantile=float(dq[index]), texture_radial_norm=float(radial_values[index]),
                          texture_radial_quantile=float(rq[index]))
        if params.method == "strata":
            members, reason = select_stratum(dbin, rbin, index, min_positions=params.min_positions)
            covariates.update(texture_dapi_bin=int(dbin[index]), texture_radial_bin=int(rbin[index]))
            max_distance = float("nan")
        else:
            members, reason, max_distance = select_knn(dq, rq, index, k=params.knn_k)
        if members is None:
            size = (int(np.count_nonzero((dbin == dbin[index]) & (rbin == rbin[index])))
                    if params.method == "strata" else int(centers.shape[0]))
            rows.append(_spot_na(reason, **covariates, texture_stratum_n_positions=size))
            continue
        stratum = np.zeros(region.shape, dtype=bool)
        stratum[centers[members, 0], centers[members, 1]] = True
        result = exact_footprint_position_null(
            qki, fp, region, n_null=n_null, rng=rng, retain_raw_draws=True,
            candidate_domain=build_exact_footprint_candidate_domain(stratum))
        if not result.usable or result.valid_center_count != int(members.sum()):
            raise RuntimeError("texture stratum disagrees with the placement-null admissible set")
        lookup = np.full(region.shape, -1, dtype=np.intp)
        lookup[centers[:, 0], centers[:, 1]] = np.arange(centers.shape[0])
        drawn = lookup[result.sampled_centers_yx[:, 0], result.sampled_centers_yx[:, 1]]
        score = _cpm.uniform_position_percentile(np.asarray([obs]), result.null_qki_raw[:, None], tie_rng)
        row = dict(covariates, upp_texture_matched_qki=float(score[0]),
                   texture_stratum_n_positions=int(members.sum()),
                   texture_null_mean_dapi_quantile=float(dq[drawn].mean()),
                   texture_null_mean_radial_quantile=float(rq[drawn].mean()),
                   texture_match_max_distance=max_distance,
                   na_reason_upp_texture_matched_spot="")
        row.setdefault("texture_dapi_bin", -1)
        row.setdefault("texture_radial_bin", -1)
        if return_draw_centers:
            row["_matched_centers"] = centers[members]
            row["_draw_centers"] = result.sampled_centers_yx
        rows.append(row)
    return rows


def rotation_spot_scores(qki, region, footprints, observed, *, n_null: int,
                         rng: np.random.Generator, tie_rng: np.random.Generator) -> tuple[list[dict], dict]:
    if not footprints:
        return [], dict(upp_rotation_median_first_pass_retention=np.nan, na_reason_upp_rotation="N0")
    result = keep_n_footprint_rotation_null(qki, footprints, region, n_null=n_null, rng=rng)
    rows = []
    for index, (spot, obs) in enumerate(zip(result.spots, observed)):
        if not spot.null_usable:
            rows.append(dict(upp_rotation_qki=np.nan,
                             na_reason_upp_rotation_spot=str(spot.invalid_reason or "unusable")))
            continue
        draws = result.null_qki_raw[index]
        draws = draws[np.isfinite(draws)]
        score = _cpm.uniform_position_percentile(np.asarray([obs]), draws[:, None], tie_rng)
        rows.append(dict(upp_rotation_qki=float(score[0]), na_reason_upp_rotation_spot=""))
    nucleus = dict(upp_rotation_median_first_pass_retention=float(result.median_first_pass_retention),
                   na_reason_upp_rotation=str(result.invalid_reason or ""))
    return rows, nucleus


def _summary(prefix, scores, k):
    scores = np.asarray(scores, dtype=float)
    finite = scores[np.isfinite(scores)]
    return {f"{prefix}_mean_qki": float(finite.mean()) if finite.size else np.nan,
            f"{prefix}_frac_ge_0p90": float(np.mean(finite >= 0.9 - 1e-12)) if finite.size else np.nan,
            f"{prefix}_frac_ge_0p75": float(np.mean(finite >= 0.75 - 1e-12)) if finite.size else np.nan,
            f"{prefix}_n_spots_scored": int(finite.size),
            f"{prefix}_n_spots_na": int(scores.size - finite.size),
            f"{prefix}_chance_mean": 0.5,
            f"{prefix}_chance_frac_ge_0p90": _cpm.upp_chance_ge(k, 0.9),
            f"{prefix}_chance_frac_ge_0p75": _cpm.upp_chance_ge(k, 0.75)}


def nucleus_texture_summary(spot_rows, k, params: TextureNullParams) -> dict:
    out = _summary("upp_texture_matched", [row["upp_texture_matched_qki"] for row in spot_rows], k)
    reasons = Counter(row["na_reason_upp_texture_matched_spot"] for row in spot_rows
                      if row["na_reason_upp_texture_matched_spot"])
    out["upp_texture_matched_na_reason_counts"] = ";".join(f"{k_}={v}" for k_, v in sorted(reasons.items()))
    out["upp_texture_matched_method"] = params.label
    out["upp_texture_matched_calibration"] = CALIBRATION[params.method]
    out["na_reason_upp_texture_matched"] = ("N0" if not spot_rows
                                            else "ALL_SPOTS_NA" if out["upp_texture_matched_n_spots_scored"] == 0
                                            else "")
    return out


def nucleus_rotation_summary(spot_rows, nucleus, k) -> dict:
    out = _summary("upp_rotation", [row["upp_rotation_qki"] for row in spot_rows], k)
    out.update(nucleus)
    if spot_rows and not out["na_reason_upp_rotation"] and out["upp_rotation_n_spots_scored"] == 0:
        out["na_reason_upp_rotation"] = "ALL_SPOTS_NA"
    return out


def sensitivity_tables(nuclei, spots, *, level: float = 1.0):
    """Per-well and per-arm DESCRIPTIVE comparison of uniform vs texture-matched
    (and rotation, when present) UPP, with covariate balance, support and NA
    rates. Well = the validated well key; nucleus means are equal-weighted
    within a well, wells equal-weighted within an arm."""
    import pandas as pd
    from .well_key import resolve_well_ids

    nuc = nuclei[nuclei.threshold_multiplier == level].copy()
    spt = spots[spots.threshold_multiplier == level].copy()
    nuc["well_key"] = resolve_well_ids(nuc)[0]
    spt["well_key"] = resolve_well_ids(spt)[0]
    rotation = "upp_rotation_mean_qki" in nuc
    rows = []
    for (condition, well), group in nuc.groupby(["condition", "well_key"], sort=True):
        s = spt[(spt.condition == condition) & (spt.well_key == well)]
        scored = s[s.upp_texture_matched_qki.notna() & s.uniform_position_percentile_qki.notna()]
        support = scored.texture_stratum_n_positions
        row = dict(condition=condition, well=well, method=str(group.upp_texture_matched_method.iloc[0]),
                   calibration=str(group.upp_texture_matched_calibration.iloc[0]),
                   n_nuclei=len(group), n_nuclei_with_spots=int((group.n_miat_spots > 0).sum()),
                   n_spots=len(s),
                   n_nuclei_upp_uniform=int(group.mean_uniform_position_percentile_qki.notna().sum()),
                   n_nuclei_upp_texture_matched=int(group.upp_texture_matched_mean_qki.notna().sum()),
                   well_mean_upp_uniform=group.mean_uniform_position_percentile_qki.mean(),
                   well_mean_upp_texture_matched=group.upp_texture_matched_mean_qki.mean(),
                   well_frac_ge_0p90_upp_uniform=group.frac_spots_upp_ge_0p90.mean(),
                   well_frac_ge_0p90_upp_texture_matched=group.upp_texture_matched_frac_ge_0p90.mean(),
                   n_spots_texture_scored=int(s.upp_texture_matched_qki.notna().sum()),
                   n_spots_texture_na=int(s.upp_texture_matched_qki.isna().sum()),
                   texture_na_rate_spots=float(s.upp_texture_matched_qki.isna().mean()) if len(s) else np.nan,
                   texture_na_reason_counts=";".join(
                       f"{k}={v}" for k, v in sorted(Counter(
                           r for r in s.na_reason_upp_texture_matched_spot.fillna("") if r).items())),
                   support_min_positions=float(support.min()) if len(support) else np.nan,
                   support_median_positions=float(support.median()) if len(support) else np.nan,
                   balance_uniform_dapi_q=float((s.texture_dapi_quantile - 0.5).mean()),
                   balance_uniform_radial_q=float((s.texture_radial_quantile - 0.5).mean()),
                   balance_matched_dapi_q=float((scored.texture_dapi_quantile
                                                 - scored.texture_null_mean_dapi_quantile).mean()),
                   balance_matched_radial_q=float((scored.texture_radial_quantile
                                                   - scored.texture_null_mean_radial_quantile).mean()),
                   balance_matched_abs_dapi_q=float((scored.texture_dapi_quantile
                                                     - scored.texture_null_mean_dapi_quantile).abs().mean()),
                   balance_matched_abs_radial_q=float((scored.texture_radial_quantile
                                                       - scored.texture_null_mean_radial_quantile).abs().mean()),
                   balance_uniform_abs_dapi_q=float((s.texture_dapi_quantile - 0.5).abs().mean()),
                   balance_uniform_abs_radial_q=float((s.texture_radial_quantile - 0.5).abs().mean()),
                   spots_both_scored=len(scored),
                   spot_pooled_upp_uniform_on_both_scored=scored.uniform_position_percentile_qki.mean(),
                   spot_pooled_upp_texture_matched_on_both_scored=scored.upp_texture_matched_qki.mean(),
                   chance_mean=0.5,
                   chance_frac_ge_0p90=float(group.upp_texture_matched_chance_frac_ge_0p90.iloc[0]))
        if rotation:
            row.update(n_nuclei_upp_rotation=int(group.upp_rotation_mean_qki.notna().sum()),
                       well_mean_upp_rotation=group.upp_rotation_mean_qki.mean(),
                       well_frac_ge_0p90_upp_rotation=group.upp_rotation_frac_ge_0p90.mean(),
                       rotation_na_rate_spots=float(s.upp_rotation_qki.isna().mean()) if len(s) else np.nan)
        rows.append(row)
    per_well = pd.DataFrame(rows)
    value_columns = [c for c in per_well.columns
                     if c.startswith(("well_mean_", "well_frac_", "spot_pooled_", "balance_", "support_"))
                     or c.endswith("_na_rate_spots")]
    per_arm = (per_well.groupby(["condition", "method", "calibration"], sort=True)
               .agg(n_wells=("well", "size"), n_nuclei=("n_nuclei", "sum"), n_spots=("n_spots", "sum"),
                    n_spots_texture_na=("n_spots_texture_na", "sum"),
                    **{f"mean_of_{c}": (c, "mean") for c in value_columns})
               .reset_index())
    per_arm["texture_na_rate_spots_pooled"] = per_arm.n_spots_texture_na / per_arm.n_spots
    per_arm["note"] = f"{SENSITIVITY_LABEL}; descriptive; wells are the replicates; no test"
    return per_well, per_arm
