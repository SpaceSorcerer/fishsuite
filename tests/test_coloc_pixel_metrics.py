"""Synthetic truth tests for the 2026-09-24 basal colocalization metrics.

Uniform-position percentile score vs own placement null, nucleoplasm-masked
Pearson/Spearman, Van Steensel CCF, and Costes block-scramble randomization
with ACF/PSF-derived block size (Astra round 2). Every expected value is
derived from the construction of the synthetic image, not the implementation.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import ndimage

from fishsuite.core import coloc_pixel_metrics as cpm
from fishsuite.core.footprint_null import MiatFootprint


def _disk(shape=(96, 96), radius=40):
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    cy, cx = shape[0] // 2, shape[1] // 2
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2


def _smooth_field(seed, shape=(96, 96), sigma=2.0):
    rng = np.random.default_rng(seed)
    return ndimage.gaussian_filter(rng.normal(size=shape), sigma) * 100 + 1000


def _gauss_acf_fwhm(sigma):
    # white noise * Gaussian(sigma): ACF(r) = exp(-r^2 / (4 sigma^2)); half at 2 sigma sqrt(ln 2)
    return 4.0 * sigma * math.sqrt(math.log(2.0))


# ------------------------------------------- uniform-position percentile
def test_upp_finite_k_chance_references():
    assert cpm.upp_chance_ge(200, 0.9) == pytest.approx(21 / 201)
    assert cpm.upp_chance_ge(200, 0.75) == pytest.approx(51 / 201)
    assert cpm.upp_chance_ge(10, 0.9) == pytest.approx(2 / 11)


def test_upp_randomized_rank_literal_without_ties():
    observed = np.array([5.0, 0.5])
    draws = np.array([[1.0, 1.0], [2.0, 2.0], [6.0, 3.0], [7.0, 4.0]])
    u = cpm.uniform_position_percentile(observed, draws, np.random.default_rng(0))
    assert u.tolist() == [2 / 4, 0 / 4]


def test_upp_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        cpm.uniform_position_percentile(np.ones(3), np.ones((10, 2)), np.random.default_rng(0))


@pytest.mark.parametrize("tied", [False, True])
def test_upp_is_calibrated_at_finite_k_with_and_without_ties(tied):
    """Exchangeable obs + K draws: u must be uniform on {0..K}/K. With heavy ties
    (tied=True: integer values 0..3) a mid-rank score would put P(u>=0.9) near 0;
    the randomized rank keeps it at (K - ceil(0.9K) + 1)/(K + 1)."""
    rng = np.random.default_rng(42)
    k, n = 200, 20000
    values = rng.integers(0, 4, size=(k + 1, n)).astype(float) if tied else rng.normal(size=(k + 1, n))
    u = cpm.uniform_position_percentile(values[0], values[1:], np.random.default_rng(1))
    sd = math.sqrt(21 / 201 * (1 - 21 / 201) / n)
    assert abs(np.mean(u >= 0.9) - 21 / 201) < 4 * sd
    assert abs(np.mean(u) - 0.5) < 4 * math.sqrt(1 / 12 / n)
    assert set(np.unique(np.rint(u * k))) <= set(range(k + 1))


def test_upp_summary_uses_finite_k_chance():
    s = cpm.upp_summary(np.array([0.95, 0.9, 0.75, 0.5, 0.1]), 200)
    assert s["mean_uniform_position_percentile_qki"] == pytest.approx(0.64)
    assert s["frac_spots_upp_ge_0p90"] == 2 / 5
    assert s["frac_spots_upp_ge_0p75"] == 3 / 5
    assert s["chance_frac_spots_upp_ge_0p90"] == pytest.approx(21 / 201)
    assert s["chance_frac_spots_upp_ge_0p75"] == pytest.approx(51 / 201)
    assert s["chance_mean_upp"] == 0.5
    assert s["n_spots_upp"] == 5


# ----------------------------------------------------- pearson/spearman
def test_identical_channels_pearson_and_spearman_are_one():
    m = _smooth_field(1)
    out = cpm.masked_correlations(m, m.copy(), _disk())
    assert out["pearson"] == pytest.approx(1.0, abs=1e-12)
    assert out["spearman"] == pytest.approx(1.0, abs=1e-12)
    assert out["reason"] == ""


def test_masked_correlation_is_offset_and_gain_invariant():
    m, q = _smooth_field(1), _smooth_field(2)
    base = cpm.masked_correlations(m, q, _disk())
    moved = cpm.masked_correlations(m, 3.0 * q + 500.0, _disk())
    assert moved["pearson"] == pytest.approx(base["pearson"], abs=1e-12)
    assert moved["spearman"] == pytest.approx(base["spearman"], abs=1e-12)


def test_masked_correlation_uses_only_masked_pixels():
    m, q = _smooth_field(1), _smooth_field(2)
    mask = _disk()
    q_out = q.copy()
    q_out[~mask] = m[~mask] * 7
    assert cpm.masked_correlations(m, q_out, mask)["pearson"] == pytest.approx(
        cpm.masked_correlations(m, q, mask)["pearson"], abs=1e-12)


def test_masked_correlation_minimum_pixels_and_zero_variance():
    m, q = _smooth_field(1), _smooth_field(2)
    small = np.zeros(m.shape, dtype=bool)
    small[:9, :11] = True
    out = cpm.masked_correlations(m, q, small)
    assert np.isnan(out["pearson"]) and out["reason"] == "LOW_PIX"
    out = cpm.masked_correlations(m, np.full(m.shape, 7.0), _disk())
    assert np.isnan(out["pearson"]) and out["reason"] == "ZERO_VAR"


# ---------------------------------------------------------------- CCF
def test_ccf_identical_channels_peak_at_zero_and_r0_one():
    m = _smooth_field(3)
    s = cpm.nucleus_ccf(m, m.copy(), _disk(), pixel_size_um=0.13)
    assert s["ccf_r0"] == pytest.approx(1.0, abs=1e-12)
    assert s["ccf_peak_shift_px_x"] == 0 and s["ccf_peak_shift_px_y"] == 0
    assert s["ccf_peak_r_x"] == pytest.approx(1.0, abs=1e-12)
    assert s["ccf_peak_shift_um_x"] == 0.0


@pytest.mark.parametrize("k", [3, -2, 5])
def test_ccf_qki_shifted_by_k_px_along_x_peaks_at_k_times_pixel(k):
    m = _smooth_field(4)
    q = np.roll(m, k, axis=1)
    s = cpm.nucleus_ccf(m, q, _disk(), pixel_size_um=0.13)
    assert s["ccf_peak_shift_px_x"] == k
    assert s["ccf_peak_shift_um_x"] == pytest.approx(k * 0.13, abs=1e-12)
    assert s["ccf_peak_r_x"] == pytest.approx(1.0, abs=1e-12)
    assert s["ccf_peak_r_x"] > s["ccf_r0"]


def test_ccf_qki_shifted_along_y_moves_only_the_y_peak():
    m = _smooth_field(5)
    q = np.roll(m, -4, axis=0)
    s = cpm.nucleus_ccf(m, q, _disk(), pixel_size_um=0.2)
    assert s["ccf_peak_shift_px_y"] == -4
    assert s["ccf_peak_shift_um_y"] == pytest.approx(-0.8, abs=1e-12)
    assert s["ccf_peak_shift_px_x"] == 0


def test_ccf_requires_a_pixel_size_from_metadata():
    m = _smooth_field(3)
    for bad in (None, 0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            cpm.nucleus_ccf(m, m, _disk(), pixel_size_um=bad)


def test_ccf_curve_pairs_only_inside_mask():
    m, q = _smooth_field(6), _smooth_field(7)
    mask = _disk()
    shifts, r, n = cpm.ccf_curve(m, q, mask, axis=1, max_shift=2)
    assert shifts.tolist() == [-2, -1, 0, 1, 2]
    both = mask[:, :-1] & mask[:, 1:]
    a, b = m[:, :-1][both], q[:, 1:][both]
    assert n[3] == both.sum()
    assert r[3] == pytest.approx(np.corrcoef(a, b)[0, 1], abs=1e-12)


def test_ccf_irregular_mask_never_wraps():
    """Astra F6: signal placed so that a wrap-around would pair it must not count."""
    m = np.zeros((40, 40))
    q = np.zeros((40, 40))
    mask = np.zeros((40, 40), dtype=bool)
    mask[5:35, 2:12] = True       # left strip
    mask[5:35, 28:38] = True      # right strip, 26 px away
    rng = np.random.default_rng(0)
    m[mask] = rng.normal(size=mask.sum())
    q[:, 28:38] = m[:, 2:12]      # QKI(right) copies MIAT(left): only reachable at +26 px
    shifts, r, n = cpm.ccf_curve(m, q, mask, axis=1, max_shift=20)
    assert np.nanmax(np.abs(r)) < 0.3
    assert n[shifts == 20][0] == 0 or np.isnan(r[shifts == 20][0]) or abs(r[shifts == 20][0]) < 0.3


def test_ccf_fwhm_literal_on_a_triangle():
    shifts = np.arange(-20, 21)
    r = np.clip(1.0 - np.abs(shifts) / 4.0, 0.0, None)
    s = cpm.ccf_summary(shifts, r)
    assert s["peak_shift_px"] == 0
    assert s["fwhm_px"] == pytest.approx(4.0, abs=1e-12)
    assert s["r0_minus_flank"] == pytest.approx(1.0, abs=1e-12)
    assert s["fwhm_reason"] == ""


def test_ccf_flat_curve_has_no_fwhm():
    shifts = np.arange(-20, 21)
    s = cpm.ccf_summary(shifts, np.full(shifts.size, 0.3))
    assert np.isnan(s["fwhm_px"]) and s["fwhm_reason"] == "NO_PEAK_ABOVE_FLANK"


# ------------------------------------------------ ACF / PSF block size
@pytest.mark.parametrize("sigma", [1.5, 3.0])
def test_acf_fwhm_recovers_gaussian_correlation_length(sigma):
    field = _smooth_field(7, shape=(200, 200), sigma=sigma)
    mask = np.ones(field.shape, dtype=bool)
    fwhm, reason = cpm.acf_fwhm_px(field, mask)
    assert reason == ""
    assert fwhm == pytest.approx(_gauss_acf_fwhm(sigma), rel=0.12)


def test_acf_uses_only_in_mask_pairs():
    field = _smooth_field(8, shape=(120, 120), sigma=2.0)
    mask = _disk((120, 120), 50)
    outside = field.copy()
    outside[~mask] = 1e6  # garbage outside N must not move the ACF
    a, _ = cpm.acf_fwhm_px(field, mask)
    b, _ = cpm.acf_fwhm_px(outside, mask)
    assert a == pytest.approx(b, abs=1e-9)


def test_acf_without_half_crossing_inside_estimable_range_is_na():
    field = _smooth_field(13, shape=(160, 160), sigma=6.0)   # FWHM ~20 px
    fwhm, reason = cpm.acf_fwhm_px(field, np.ones(field.shape, dtype=bool), max_radius=3)
    assert np.isnan(fwhm) and reason == "ACF_NO_HALF"


def test_psf_fwhm_from_emission_and_na_fails_loudly_without_metadata():
    assert cpm.psf_fwhm_px(668.0, 1.5, 130.0) == pytest.approx(0.51 * 668 / 1.5 / 130)
    for bad in [(None, 1.5, 130.0), (668.0, None, 130.0), (668.0, 0.0, 130.0), (float("nan"), 1.5, 130.0)]:
        with pytest.raises(ValueError):
            cpm.psf_fwhm_px(*bad)


def test_block_size_rule_uses_smaller_acf_and_psf_floor():
    # b = ceil(max(PSF, min(FWHM_MIAT, FWHM_QKI)))
    assert cpm.costes_block_from_widths(6.2, 13.0, 1.7) == 7
    assert cpm.costes_block_from_widths(13.0, 6.2, 1.7) == 7
    assert cpm.costes_block_from_widths(2.1, 3.0, 4.4) == 5
    assert cpm.costes_block_from_widths(float("nan"), 3.0, 1.7) is None


def test_block_size_from_acf_on_two_channels_of_known_width():
    m = _smooth_field(9, shape=(200, 200), sigma=1.5)
    q = _smooth_field(10, shape=(200, 200), sigma=3.0)
    out = cpm.nucleus_costes(m, q, np.ones(m.shape, dtype=bool), psf_px=1.7, n_iter=20,
                             rng=np.random.default_rng(0))
    assert out["costes_acf_fwhm_px_miat"] == pytest.approx(_gauss_acf_fwhm(1.5), rel=0.12)
    assert out["costes_acf_fwhm_px_qki"] == pytest.approx(_gauss_acf_fwhm(3.0), rel=0.12)
    assert out["costes_block_px"] == math.ceil(max(1.7, out["costes_acf_fwhm_px_miat"]))


# ------------------------------------------ tile phase + coverage gate
def test_tile_phase_maximises_complete_block_coverage_from_geometry():
    mask = np.zeros((50, 50), dtype=bool)
    mask[10:40, 10:40] = True
    mask[7, 7] = True          # shifts the bbox origin by 3 px: phase 0 is misaligned for b=5
    phase, core, n_blocks, coverage = cpm.best_tile_phase(mask, 5)
    assert n_blocks == 36 and core.sum() == 900
    assert coverage == pytest.approx(900 / 901)
    brute = max(cpm._complete_blocks(mask, 5, oy, ox)[1] for oy in range(5) for ox in range(5))
    assert n_blocks == brute


def test_irregular_mask_below_80pct_coverage_is_na():
    mask = _disk((96, 96), 30)
    m, q = _smooth_field(11), _smooth_field(12)
    out = cpm.costes_randomization(m, q, mask, 11, 50, np.random.default_rng(0))
    assert out["costes_core_coverage"] < 0.8
    assert np.isnan(out["costes_rand_p"]) and out["reason"] == "MASK_BLOCK_COVERAGE"


def test_fewer_than_ten_blocks_is_na():
    mask = np.zeros((40, 40), dtype=bool)
    mask[0:9, 0:9] = True  # 9 blocks of 3 px, 100 % coverage
    out = cpm.costes_randomization(_smooth_field(1, (40, 40)), _smooth_field(2, (40, 40)), mask, 3, 50,
                                   np.random.default_rng(0))
    assert out["costes_core_coverage"] == 1.0
    assert out["reason"] == "MASK_BLOCK_COVERAGE"


def test_costes_observed_and_null_both_on_frozen_core():
    m, q = _smooth_field(8), _smooth_field(9)
    mask = np.zeros(m.shape, dtype=bool)
    mask[10:40, 10:40] = True
    out = cpm.costes_randomization(m, q, mask, 3, 50, np.random.default_rng(0))
    assert out["costes_rand_n_blocks"] == 100 and out["costes_core_coverage"] == 1.0
    assert out["costes_rand_r_obs"] == pytest.approx(np.corrcoef(m[mask], q[mask])[0, 1], abs=1e-12)
    assert out["costes_rand_n_draws"] == 50


# ---------------------------------------- Costes calibration (Astra F10)
def _coloc_pair(seed, coupling):
    rng = np.random.default_rng(seed)
    shared = ndimage.gaussian_filter(rng.normal(size=(96, 96)), 2.0)
    shared /= shared.std()
    m = shared + 0.3 * rng.normal(size=shared.shape)
    q = coupling * shared + 0.3 * rng.normal(size=shared.shape)
    return m * 100 + 1000, q * 100 + 1000


def test_costes_positive_effect_scales_with_coupling_and_null_is_centred():
    """Not a floor test: the null mean must sit at ~0 and r_obs - null must
    track the true coupling; p is reported, not used as the pass criterion."""
    mask = np.ones((96, 96), dtype=bool)
    strong = cpm.nucleus_costes(*_coloc_pair(1, 1.0), mask, psf_px=1.7, n_iter=200,
                                rng=np.random.default_rng(0))
    weak = cpm.nucleus_costes(*_coloc_pair(1, 0.05), mask, psf_px=1.7, n_iter=200,
                              rng=np.random.default_rng(0))

    def truth(c):  # corr(S + 0.3e1, cS + 0.3e2) with var(S) = 1
        return c / (math.sqrt(1 + 0.09) * math.sqrt(c * c + 0.09))

    for out, c in ((strong, 1.0), (weak, 0.05)):
        assert out["na_reason_costes_rand"] == ""
        assert abs(out["costes_rand_null_mean_r"]) < 0.05
        assert out["costes_rand_r_obs_minus_null_mean"] == pytest.approx(truth(c), abs=0.08)
    assert weak["costes_rand_r_obs_minus_null_mean"] < strong["costes_rand_r_obs_minus_null_mean"] - 0.5
    assert strong["costes_rand_p"] < 0.05


def test_costes_independent_autocorrelated_channels_false_positive_rate():
    """100 independent nuclei with spatially autocorrelated channels at the
    ACF/PSF block size. Under a calibrated test X ~ Bin(100, 0.05); P(X >= 13)
    = 0.0015, so more than 12 rejections at alpha 0.05 fails the test."""
    mask = np.ones((96, 96), dtype=bool)
    ps = []
    for seed in range(100):
        m = _smooth_field(1000 + seed, sigma=2.0)
        q = _smooth_field(5000 + seed, sigma=2.0)
        out = cpm.nucleus_costes(m, q, mask, psf_px=1.7, n_iter=200, rng=np.random.default_rng(seed))
        assert out["na_reason_costes_rand"] == ""
        ps.append(out["costes_rand_p"])
    assert int(np.sum(np.asarray(ps) < 0.05)) <= 12


def test_nucleus_costes_without_psf_is_na_not_defaulted():
    out = cpm.nucleus_costes(_smooth_field(1), _smooth_field(2), _disk(), psf_px=None, n_iter=10,
                             rng=np.random.default_rng(0))
    assert out["na_reason_costes_rand"] == "NO_PSF" and np.isnan(out["costes_block_px"])


# ---------------------------------------- rank metric through the chain API
def _footprint(index, y, x, radius=1):
    dy, dx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    dy, dx = dy.ravel(), dx.ravel()
    return MiatFootprint(index, y, x, y + dy, x + dx, dy, dx,
                         "synthetic", None, True, None, 100.0, 0.0)


def _spots_on_random_qki(seed, colocalized):
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(1, 11), 48)[:, None] * np.ones((1, 96), dtype=int)
    qki = ndimage.gaussian_filter(rng.normal(size=labels.shape), 1.0) * 100 + 1000
    miat = np.full(labels.shape, 10.0)
    footprints = []
    for nucleus in range(10):
        centers = rng.choice(np.arange(3, 45), 30), rng.choice(np.arange(3, 93), 30)
        for y, x in zip(*centers):
            fp = _footprint(len(footprints), nucleus * 48 + int(y), int(x))
            footprints.append(fp)
            miat[fp.y_px, fp.x_px] = 500.0
            if colocalized:
                qki[fp.y_px, fp.x_px] += 400.0
    return miat, qki, labels, footprints


def _assoc(data, **kw):
    from fishsuite.core.qki_association import association_tables
    options = dict(pixel_size_um=0.13, miat_min=100, qki_min=1050, sensitivity=(1.0,),
                   n_null=200, seed=0, psf_fwhm_px=1.7, psf_source="synthetic")
    options.update(kw)
    return association_tables(*data, **options)


def test_independent_channels_percentile_score_is_at_finite_k_chance():
    nuclei, spots = _assoc(_spots_on_random_qki(21, colocalized=False))
    n_spots = int(nuclei.n_miat_spots.sum())
    assert n_spots >= 250
    assert abs(float(spots.uniform_position_percentile_qki.mean()) - 0.5) < 0.05
    assert abs(float(nuclei.mean_uniform_position_percentile_qki.mean()) - 0.5) < 0.05
    assert abs(float(nuclei.frac_spots_upp_ge_0p90.mean()) - 21 / 201) < 0.05
    assert set(nuclei.chance_frac_spots_upp_ge_0p90.round(12)) == {round(21 / 201, 12)}


def test_colocalized_puncta_score_near_one_no_qki_cutoff_involved():
    nuclei, _ = _assoc(_spots_on_random_qki(22, colocalized=True))
    assert float(nuclei.mean_uniform_position_percentile_qki.min()) > 0.9
    assert float(nuclei.frac_spots_upp_ge_0p90.min()) > 0.8
    high, _ = _assoc(_spots_on_random_qki(22, colocalized=True), qki_min=1e9)
    assert np.allclose(high.mean_uniform_position_percentile_qki, nuclei.mean_uniform_position_percentile_qki)


def test_new_metrics_are_threshold_free_across_sensitivity_levels():
    nuclei, _ = _assoc(_spots_on_random_qki(23, colocalized=True), sensitivity=(0.8, 1.0, 1.25), n_null=50)
    cols = ["mean_uniform_position_percentile_qki", "pearson_r_nucleoplasm", "spearman_rho_nucleoplasm",
            "ccf_r0", "ccf_peak_shift_um_x", "costes_rand_p", "costes_block_px"]
    for _, group in nuclei.groupby("nucleus_id"):
        for col in cols:
            vals = group[col].to_numpy(dtype=float)
            assert np.allclose(vals, vals[0], equal_nan=True), col


def test_chain_records_costes_parameters_and_ccf_centred():
    nuclei, _ = _assoc(_spots_on_random_qki(24, colocalized=True))
    assert set(nuclei.costes_psf_source) == {"synthetic"}
    assert nuclei.costes_psf_fwhm_px.eq(1.7).all()
    assert nuclei.costes_rand_n_draws.eq(200).all()
    finite = nuclei.dropna(subset=["costes_block_px"])
    assert (finite.costes_block_px >= np.ceil(1.7)).all()
    assert set(nuclei.ccf_peak_shift_px_x) == {0}
    assert set(nuclei.ccf_peak_shift_px_y) == {0}


def test_new_metrics_do_not_perturb_existing_placement_null_columns():
    from fishsuite.core import qki_association as qa
    data = _spots_on_random_qki(25, colocalized=True)
    nuclei, spots = _assoc(data)
    legacy = list(qa.LEGACY_NUCLEUS_COLUMNS)
    assert list(nuclei.columns[:len(legacy)]) == legacy
    assert list(spots.columns[:len(qa.LEGACY_SPOT_COLUMNS)]) == list(qa.LEGACY_SPOT_COLUMNS)
    off, off_spots = _assoc(data, n_costes=1, psf_fwhm_px=None)
    assert nuclei[legacy].equals(off[legacy])
    assert spots[list(qa.LEGACY_SPOT_COLUMNS)].equals(off_spots[list(qa.LEGACY_SPOT_COLUMNS)])


def test_ccf_records_collected_per_nucleus_and_axis():
    records = []
    _assoc(_spots_on_random_qki(26, colocalized=False), n_null=20, image="img", condition="c",
           well="w", ccf_records=records)
    assert len(records) == 10 * 2 * 41
    assert {r["axis"] for r in records} == {"x", "y"}
    assert {r["shift_px"] for r in records} == set(range(-20, 21))
    assert all(r["image"] == "img" and r["well"] == "w" for r in records)


def test_every_new_column_has_a_definition():
    from fishsuite.core import qki_association as qa
    for name in qa.NUCLEUS_COLUMNS + qa.SPOT_COLUMNS:
        assert name in qa.COLUMN_DEFINITIONS, name
    assert "descriptive" in qa.COLUMN_DEFINITIONS["costes_rand_p"]
    assert "not a test across nuclei" in qa.COLUMN_DEFINITIONS["costes_rand_p"]
    assert "uniform-position percentile" in qa.COLUMN_DEFINITIONS["uniform_position_percentile_qki"].lower()


# ------------------------------------------------ nucleolus sensitivity (F5)
def test_nucleoplasm_sensitivity_recomputes_mask_per_percentile():
    from fishsuite.core.nucleolus import NucleolusParams, detect_nucleoli
    rng = np.random.default_rng(3)
    labels = np.zeros((80, 80), dtype=np.int32)
    labels[_disk((80, 80), 35)] = 1
    yy, xx = np.mgrid[:80, :80]
    dapi = 5000 - 3000 * np.exp(-((yy - 40) ** 2 + (xx - 40) ** 2) / (2 * 8.0 ** 2)) \
        + rng.normal(0, 150, (80, 80))
    shared = ndimage.gaussian_filter(rng.normal(size=(80, 80)), 2)
    miat = 1000 + 300 * shared + (dapi - 5000) * 0.05 + rng.normal(0, 20, (80, 80))
    qki = 1000 + 200 * shared + (dapi - 5000) * 0.05 + rng.normal(0, 20, (80, 80))
    params = dict(intra_nuclear_percentile=25.0, min_area_um2=1.0, max_area_frac_of_nucleus=0.6,
                  closing_radius_px=2, min_border_distance_px=5)
    table = cpm.nucleoplasm_sensitivity(miat, qki, dapi, labels, 0.13, params, (20, 25, 30))
    assert table.percentile.tolist() == [20, 25, 30]
    frac = table.nucleolus_area_frac.to_numpy()
    assert frac[0] <= frac[1] <= frac[2] and frac[2] > frac[0]
    ref = detect_nucleoli(labels, dapi, 0.13, NucleolusParams(**params))
    expected = cpm.masked_correlations(miat, qki, (labels == 1) & (ref != 1))
    row = table.set_index("percentile").loc[25]
    assert row.pearson_r_nucleoplasm == pytest.approx(expected["pearson"], abs=1e-12)
    assert row.spearman_rho_nucleoplasm == pytest.approx(expected["spearman"], abs=1e-12)
