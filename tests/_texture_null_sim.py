"""Realistic synthetic nuclei for calibrating the SENSITIVITY-ONLY texture null.

Each nucleus: irregular (Fourier-perturbed, rotated) ellipse, 1-2 nucleolar
holes excluded from the eligible mask, variable exact footprints (ellipses of
random size/orientation, 5-21 px), a smooth nonlinear DAPI field (chromatin
texture G plus a peripheral rim, with pixel noise) and smooth QKI. No field
has a boundary aligned with the quantile strata.

Scenarios (per nucleus, independent across nuclei):
  confound    : QKI and MIAT placement both depend on (G, radial) only.
  independent : QKI depends on (G, radial); MIAT placed uniformly.
  coloc       : like independent, plus QKI +delta under each MIAT footprint.

``nucleus_summaries`` returns one row per nucleus with the nucleus-mean score
under the uniform null and each texture method; replicates are fields of
independent nuclei. Nucleus means are the independent calibration unit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

from fishsuite.core import coloc_pixel_metrics as cpm
from fishsuite.core.footprint_null import MiatFootprint, exact_footprint_position_null
from fishsuite.core.texture_null import (TextureNullParams, admissible_centers, normalized_radial_map,
                                         texture_matched_spot_scores)

SIZE = 112
K = 200


def _smooth(rng, sigma):
    field = ndimage.gaussian_filter(rng.normal(size=(SIZE, SIZE)), sigma)
    return (field - field.mean()) / field.std()


def _footprint(rng, index, y, x):
    a, b = rng.uniform(1.0, 2.6, size=2)
    theta = rng.uniform(0, np.pi)
    r = int(np.ceil(max(a, b)))
    dy, dx = np.mgrid[-r:r + 1, -r:r + 1]
    u = dx * np.cos(theta) + dy * np.sin(theta)
    v = -dx * np.sin(theta) + dy * np.cos(theta)
    keep = (u / a) ** 2 + (v / b) ** 2 <= 1.0
    keep[r, r] = True
    dy, dx = dy[keep], dx[keep]
    return MiatFootprint(index, int(y), int(x), y + dy, x + dx, dy, dx,
                         "synthetic", None, True, None, 100.0, 0.0)


def nucleus(rng, scenario, *, delta=80.0):
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(float)
    cy, cx = SIZE / 2 + rng.uniform(-3, 3, 2)
    a, b = rng.uniform(34, 46), rng.uniform(28, 40)
    phi = rng.uniform(0, np.pi)
    u = (xx - cx) * np.cos(phi) + (yy - cy) * np.sin(phi)
    v = -(xx - cx) * np.sin(phi) + (yy - cy) * np.cos(phi)
    angle = np.arctan2(v / b, u / a)
    wobble = 1 + sum(rng.uniform(-0.06, 0.06) * np.cos(k * angle + rng.uniform(0, 2 * np.pi)) for k in (2, 3, 4))
    mask = np.hypot(u / a, v / b) <= wobble
    eligible = mask.copy()
    inner = ndimage.binary_erosion(mask, iterations=10)
    for _ in range(rng.integers(1, 3)):
        hy, hx = np.argwhere(inner)[rng.integers(inner.sum())]
        ha, hb, ht = rng.uniform(3, 6), rng.uniform(2.5, 5), rng.uniform(0, np.pi)
        hu = (xx - hx) * np.cos(ht) + (yy - hy) * np.sin(ht)
        hv = -(xx - hx) * np.sin(ht) + (yy - hy) * np.cos(ht)
        eligible &= (hu / ha) ** 2 + (hv / hb) ** 2 > 1
    radial = normalized_radial_map(mask)
    chromatin = _smooth(rng, 4.0)
    dapi = (700 + 900 / (1 + np.exp(-1.6 * chromatin)) + 350 * np.exp(-6 * radial)
            + rng.normal(0, 40, (SIZE, SIZE)))
    qki = 1000 + 90 * np.tanh(chromatin) + 70 * np.sqrt(radial) + 45 * _smooth(rng, 2.0)
    n_spots = int(rng.integers(6, 16))
    footprints, attempts = [], 0
    while len(footprints) < n_spots and attempts < 200:
        attempts += 1
        fp = _footprint(rng, len(footprints), 0, 0)
        centers = admissible_centers(fp, eligible, qki)
        if centers.shape[0] == 0:
            continue
        if scenario == "confound":
            w = np.exp(1.4 * np.tanh(chromatin[centers[:, 0], centers[:, 1]])
                       + 1.4 * np.sqrt(radial[centers[:, 0], centers[:, 1]]))
            pick = centers[rng.choice(centers.shape[0], p=w / w.sum())]
        else:
            pick = centers[rng.integers(centers.shape[0])]
        footprints.append(_footprint_at(fp, *pick))
    if scenario == "coloc":
        for fp in footprints:
            qki[fp.y_px, fp.x_px] += delta
    return dict(mask=mask, eligible=eligible, dapi=dapi, qki=qki, footprints=footprints)


def _footprint_at(fp, y, x):
    return MiatFootprint(fp.spot_index, int(y), int(x), y + fp.dy_px, x + fp.dx_px, fp.dy_px, fp.dx_px,
                         fp.method, None, True, None, 100.0, 0.0)


def score_nucleus(nuc, rng, methods):
    region, qki = nuc["eligible"], nuc["qki"]
    fps = nuc["footprints"]
    observed = np.asarray([qki[fp.y_px, fp.x_px].mean() for fp in fps])
    uniform = []
    for fp in fps:
        result = exact_footprint_position_null(qki, fp, region, n_null=K, rng=rng, retain_raw_draws=True)
        uniform.append(result.null_qki_raw)
    scores = {"uniform": cpm.uniform_position_percentile(observed, np.stack(uniform, axis=1), rng)}
    na = {"uniform": 0}
    for name, params in methods.items():
        rows = texture_matched_spot_scores(qki, nuc["dapi"], nuc["mask"], region, fps, observed,
                                           n_null=K, params=params, rng=rng, tie_rng=rng)
        values = np.asarray([r["upp_texture_matched_qki"] for r in rows])
        scores[name] = values
        na[name] = int(np.isnan(values).sum())
    return scores, na


DEFAULT_METHODS = {"strata_5x5": TextureNullParams(),
                   "strata_8x8": TextureNullParams(n_dapi_bins=8, n_radial_bins=8),
                   "strata_10x10": TextureNullParams(n_dapi_bins=10, n_radial_bins=10),
                   "knn_k200": TextureNullParams(method="knn", knn_k=200)}


def nucleus_summaries(scenario, n_fields, n_nuclei, seed, methods=None, **kw):
    methods = DEFAULT_METHODS if methods is None else methods
    rows = []
    for field in range(n_fields):
        rng = np.random.default_rng([seed, field])
        for index in range(n_nuclei):
            nuc = nucleus(rng, scenario, **kw)
            scores, na = score_nucleus(nuc, rng, methods)
            for name, values in scores.items():
                finite = values[np.isfinite(values)]
                rows.append(dict(scenario=scenario, field=field, nucleus=index, method=name,
                                 n_spots=len(values), n_na=na[name],
                                 nucleus_mean=float(finite.mean()) if finite.size else np.nan,
                                 nucleus_frac_ge_0p90=float(np.mean(finite >= 0.9 - 1e-12))
                                 if finite.size else np.nan))
    return pd.DataFrame(rows)


def field_rejections(summaries, alpha=0.05):
    """Per (scenario, method): empirical rate at which a two-sided one-sample t
    test of the field's nucleus means against 0.5 rejects at ``alpha``, plus
    the mean field bias. Each field's nuclei are independent."""
    from scipy import stats
    out = []
    for (scenario, method), group in summaries.groupby(["scenario", "method"], sort=False):
        rejections, bias, bias_q90 = [], [], []
        for _, field in group.groupby("field"):
            values = field.nucleus_mean.dropna().to_numpy()
            rejections.append(stats.ttest_1samp(values, 0.5).pvalue < alpha)
            bias.append(values.mean() - 0.5)
            bias_q90.append(field.nucleus_frac_ge_0p90.dropna().mean() - 21 / 201)
        out.append(dict(scenario=scenario, method=method, n_fields=len(rejections),
                        n_nuclei=len(group), spot_na_rate=group.n_na.sum() / group.n_spots.sum(),
                        rejection_rate=float(np.mean(rejections)), mean_field_bias=float(np.mean(bias)),
                        sd_field_bias=float(np.std(bias, ddof=1)),
                        mean_field_bias_frac_ge_0p90=float(np.mean(bias_q90))))
    return pd.DataFrame(out)
