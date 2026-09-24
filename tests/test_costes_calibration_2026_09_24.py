"""Costes block-scramble calibration gate (round 3, 2026-09-24).

Synthetic independent-channel nuclei with textures matched to the real UD
basal data (MIAT punctate, ACF FWHM ~3.5 px; QKI diffuse, ~7.7 px; source:
implB_realdata_check/round2_882cdcd per-nucleus ACF widths) on irregular
nuclear masks (random ellipses with 1-2 nucleolar holes). At alpha 0.05 the
false-positive rate must not exceed the one-sided 95% binomial upper limit
for p = 0.05 at the number of estimable nuclei. Colocalized nuclei must still
reach small p. The rates at alpha 0.05 and 0.01 are written to
COSTES_CALIBRATION_OUT when that environment variable names a file.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest
from scipy import ndimage, stats

from fishsuite.core import coloc_pixel_metrics as cpm

N_ESTIMABLE = 400      # estimable nuclei required (coverage gate NAs are drawn past)
MAX_DRAWN = 6000
SIGMA_MIAT, SIGMA_QKI = 1.22, 2.43
PSF_PX = 0.51 * 668.0 / 1.5 / 130.0   # real optics: NA 1.5, 668 nm, 130 nm px


def _nucleus_mask(rng, shape=(120, 120)):
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    cy, cx = shape[0] / 2 + rng.uniform(-3, 3), shape[1] / 2 + rng.uniform(-3, 3)
    a, b, t = rng.uniform(40, 52), rng.uniform(32, 44), rng.uniform(0, np.pi)
    u = (xx - cx) * np.cos(t) + (yy - cy) * np.sin(t)
    v = -(xx - cx) * np.sin(t) + (yy - cy) * np.cos(t)
    mask = (u / a) ** 2 + (v / b) ** 2 <= 1
    for _ in range(rng.integers(1, 3)):
        r = rng.uniform(4, 7)
        oy, ox = cy + rng.uniform(-0.4, 0.4) * b, cx + rng.uniform(-0.4, 0.4) * a
        mask &= (yy - oy) ** 2 + (xx - ox) ** 2 > r * r
    return mask


def _puncta(rng, shape):
    img = np.zeros(shape)
    n = rng.integers(20, 60)
    img[rng.integers(0, shape[0], n), rng.integers(0, shape[1], n)] = rng.uniform(0.5, 1.5, n)
    return ndimage.gaussian_filter(img, SIGMA_MIAT)


def _miat_like(rng, shape, puncta=None):
    base = _puncta(rng, shape) if puncta is None else puncta
    return base * 4000 + 500 + rng.normal(0, 15, shape)


def _qki_like(rng, shape):
    return ndimage.gaussian_filter(rng.normal(size=shape), SIGMA_QKI) * 300 + 1500 + rng.normal(0, 5, shape)


@pytest.fixture(scope="module")
def null_run():
    ps, widths_m, widths_q, blocks, na = [], [], [], [], 0
    for seed in range(MAX_DRAWN):
        if len(ps) >= N_ESTIMABLE:
            break
        rng = np.random.default_rng(10_000 + seed)
        mask = _nucleus_mask(rng)
        out = cpm.nucleus_costes(_miat_like(rng, mask.shape), _qki_like(rng, mask.shape), mask,
                                 psf_px=PSF_PX, n_iter=200, rng=np.random.default_rng(seed))
        widths_m.append(out["costes_acf_fwhm_px_miat"])
        widths_q.append(out["costes_acf_fwhm_px_qki"])
        blocks.append(out["costes_block_px"])
        if out["na_reason_costes_rand"]:
            na += 1
        else:
            ps.append(out["costes_rand_p"])
    return dict(p=np.asarray(ps), acf_m=np.asarray(widths_m), acf_q=np.asarray(widths_q),
                block=np.asarray(blocks), n_na=na, n_drawn=len(widths_m))


def test_synthetic_textures_match_real_data(null_run):
    assert np.nanmedian(null_run["acf_m"]) == pytest.approx(3.5, rel=0.15)
    assert np.nanmedian(null_run["acf_q"]) == pytest.approx(7.7, rel=0.15)


def test_costes_false_positive_rate_within_binomial_upper_limit(null_run):
    p = null_run["p"]
    n = p.size
    assert n >= 400, f"only {n} estimable nuclei (NA {null_run['n_na']})"
    fpr05, fpr01 = float(np.mean(p < 0.05)), float(np.mean(p < 0.01))
    limit05 = stats.binom.ppf(0.95, n, 0.05) / n
    limit01 = stats.binom.ppf(0.95, n, 0.01) / n
    target = os.environ.get("COSTES_CALIBRATION_OUT")
    if target:
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(dict(n_estimable=n, n_na=null_run["n_na"], n_drawn=null_run["n_drawn"],
                           na_rate=null_run["n_na"] / null_run["n_drawn"], fpr_alpha_0p05=fpr05,
                           limit_alpha_0p05=limit05, fpr_alpha_0p01=fpr01, limit_alpha_0p01=limit01,
                           median_block_px=float(np.nanmedian(null_run["block"])),
                           median_acf_miat=float(np.nanmedian(null_run["acf_m"])),
                           median_acf_qki=float(np.nanmedian(null_run["acf_q"]))), handle, indent=1)
    calibrated = fpr05 <= limit05
    # The documented status must agree with the gate: a p may only be called
    # CALIBRATED when the texture-matched false-positive rate passes.
    assert cpm.COSTES_P_CALIBRATION == ("CALIBRATED" if calibrated else "NOT_CALIBRATED"), (
        f"FPR {fpr05:.4f} at alpha 0.05 vs limit {limit05:.4f} (n={n})")


def test_not_calibrated_status_is_documented_everywhere_p_is_shown():
    from fishsuite.core.qki_association import COLUMN_DEFINITIONS
    from fishsuite.report import coupling_stats
    if cpm.COSTES_P_CALIBRATION != "NOT_CALIBRATED":
        pytest.skip("calibrated")
    assert "NOT_CALIBRATED" in COLUMN_DEFINITIONS["costes_rand_p"]
    import inspect
    assert "NOT_CALIBRATED" in inspect.getsource(coupling_stats._readme)
    assert "NOT_CALIBRATED" in inspect.getsource(coupling_stats._descriptive_qc)


def test_costes_colocalized_positive_control_reaches_small_p():
    ps = []
    for seed in range(1000):
        if len(ps) >= 40:
            break
        rng = np.random.default_rng(90_000 + seed)
        mask = _nucleus_mask(rng)
        puncta = _puncta(rng, mask.shape)
        qki = _qki_like(rng, mask.shape) + puncta * 1500   # QKI enriched on MIAT puncta
        out = cpm.nucleus_costes(_miat_like(rng, mask.shape, puncta), qki, mask, psf_px=PSF_PX,
                                 n_iter=200, rng=np.random.default_rng(seed))
        if not out["na_reason_costes_rand"]:
            ps.append(out["costes_rand_p"])
    assert len(ps) >= 30
    assert np.mean(np.asarray(ps) < 0.01) >= 0.9
