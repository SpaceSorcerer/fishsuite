"""Synthetic truth tests for the 2026-09-24 basal colocalization metrics.

Rank vs own placement null, nucleoplasm-masked Pearson/Spearman, Van Steensel
CCF and Costes block-scramble randomization. Every expected value is derived
from the construction of the synthetic image, not from the implementation.
"""
from __future__ import annotations

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


def _puncta(seed, shape=(96, 96), n=40, sigma=1.2, mask=None):
    rng = np.random.default_rng(seed)
    img = np.zeros(shape)
    ys, xs = np.nonzero(_disk(shape, 34) if mask is None else mask)
    pick = rng.choice(ys.size, n, replace=False)
    img[ys[pick], xs[pick]] = 1.0
    return ndimage.gaussian_filter(img, sigma) * 5000, ys[pick], xs[pick]


# ------------------------------------------------------------- midrank
def test_null_midrank_literal_with_ties():
    observed = np.array([5.0, 1.0, 3.0])
    draws = np.array([[1.0, 1.0, 1.0],
                      [2.0, 2.0, 3.0],
                      [5.0, 3.0, 3.0],
                      [6.0, 4.0, 4.0]])
    u = cpm.null_midrank(observed, draws)
    # spot 0: below = {1, 2} -> 2, ties = {5} -> 0.5 ; (2 + 0.5) / 4
    # spot 1: below = 0, ties = {1} -> 0.5 ; 0.5 / 4
    # spot 2: below = {1} -> 1, ties = {3, 3} -> 1 ; 2 / 4
    assert u.tolist() == [2.5 / 4, 0.5 / 4, 2.0 / 4]


def test_null_midrank_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        cpm.null_midrank(np.ones(3), np.ones((10, 2)))


def test_rank_summary_chance_levels_and_cutoffs():
    u = np.array([0.95, 0.9, 0.75, 0.5, 0.1])
    s = cpm.rank_summary(u)
    assert s["mean_null_rank_qki_at_miat"] == pytest.approx(np.mean(u))
    assert s["frac_spots_ge_null_q90"] == 2 / 5
    assert s["frac_spots_ge_null_q75"] == 3 / 5


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
    q_out[~mask] = m[~mask] * 7  # perfect correlation OUTSIDE the mask only
    assert cpm.masked_correlations(m, q_out, mask)["pearson"] == pytest.approx(
        cpm.masked_correlations(m, q, mask)["pearson"], abs=1e-12)


def test_masked_correlation_minimum_pixels_and_zero_variance():
    m, q = _smooth_field(1), _smooth_field(2)
    small = np.zeros(m.shape, dtype=bool)
    small[:9, :11] = True  # 99 pixels
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
    q = np.roll(m, k, axis=1)  # QKI content displaced by +k px in x
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
    # hand-computed at +1: pairs (y, x) and (y, x+1) both inside mask
    both = mask[:, :-1] & mask[:, 1:]
    a, b = m[:, :-1][both], q[:, 1:][both]
    assert n[3] == both.sum()
    assert r[3] == pytest.approx(np.corrcoef(a, b)[0, 1], abs=1e-12)


def test_ccf_fwhm_literal_on_a_triangle():
    shifts = np.arange(-20, 21)
    r = np.clip(1.0 - np.abs(shifts) / 4.0, 0.0, None)  # peak 1 at 0, flank mean 0
    s = cpm.ccf_summary(shifts, r)
    assert s["peak_shift_px"] == 0
    assert s["fwhm_px"] == pytest.approx(4.0, abs=1e-12)   # half level 0.5 at +-2
    assert s["r0_minus_flank"] == pytest.approx(1.0, abs=1e-12)
    assert s["fwhm_reason"] == ""


def test_ccf_flat_curve_has_no_fwhm():
    shifts = np.arange(-20, 21)
    s = cpm.ccf_summary(shifts, np.full(shifts.size, 0.3))
    assert np.isnan(s["fwhm_px"]) and s["fwhm_reason"] == "NO_PEAK_ABOVE_FLANK"


# ------------------------------------------------------------- Costes
def test_costes_block_size_from_median_footprint_area():
    assert cpm.costes_block_size([9, 9, 25]) == 3
    assert cpm.costes_block_size([30, 36, 40]) == 6      # sqrt(36) = 6
    assert cpm.costes_block_size([4, 4]) == 3            # floor at 3 px
    assert cpm.costes_block_size([]) is None


def test_costes_colocalized_puncta_gives_small_p():
    mask = _disk()
    miat, _, _ = _puncta(11, mask=None)
    noise = np.random.default_rng(12).normal(0, 5, miat.shape)
    qki = 0.8 * miat + 800 + noise
    out = cpm.costes_randomization(miat, qki, mask, block_px=3, n_iter=200,
                                   rng=np.random.default_rng(0))
    assert out["reason"] == ""
    assert out["costes_rand_r_obs"] > 0.9
    assert out["costes_rand_p"] == pytest.approx(1 / 201)
    assert out["costes_rand_p"] < 0.05


def test_costes_independent_channels_p_not_small():
    mask = _disk()
    ps = []
    for seed in range(30):
        m, q = _smooth_field(100 + seed, sigma=1.0), _smooth_field(500 + seed, sigma=1.0)
        out = cpm.costes_randomization(m, q, mask, block_px=3, n_iter=200,
                                       rng=np.random.default_rng(seed))
        ps.append(out["costes_rand_p"])
    ps = np.asarray(ps)
    assert np.median(ps) > 0.2
    assert np.mean(ps < 0.05) <= 0.2


def test_costes_same_seed_same_answer_and_few_blocks_reason():
    m, q = _smooth_field(1), _smooth_field(2)
    a = cpm.costes_randomization(m, q, _disk(), 3, 200, np.random.default_rng(0))
    b = cpm.costes_randomization(m, q, _disk(), 3, 200, np.random.default_rng(0))
    assert a == b
    tiny = np.zeros(m.shape, dtype=bool)
    tiny[:6, :12] = True  # 8 full 3x3 blocks < 10
    out = cpm.costes_randomization(m, q, tiny, 3, 200, np.random.default_rng(0))
    assert np.isnan(out["costes_rand_p"]) and out["reason"] == "FEW_BLOCKS"


def test_costes_block_permutation_keeps_miat_fixed_and_qki_multiset():
    # r_obs is computed on exactly the pixels of full blocks
    m, q = _smooth_field(8), _smooth_field(9)
    mask = np.zeros(m.shape, dtype=bool)
    mask[10:40, 10:40] = True  # 10 x 10 blocks of 3 px, all full
    out = cpm.costes_randomization(m, q, mask, 3, 50, np.random.default_rng(0))
    assert out["costes_rand_n_blocks"] == 100
    assert out["costes_rand_r_obs"] == pytest.approx(
        np.corrcoef(m[mask], q[mask])[0, 1], abs=1e-12)


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


def _assoc(data):
    from fishsuite.core.qki_association import association_tables
    return association_tables(*data, pixel_size_um=0.13, miat_min=100, qki_min=1050,
                              sensitivity=(1.0,), n_null=200, seed=0)


def test_independent_channels_mean_chance_rank_is_one_half():
    nuclei, spots = _assoc(_spots_on_random_qki(21, colocalized=False))
    n_spots = int(nuclei.n_miat_spots.sum())
    assert n_spots >= 250
    pooled = float(spots.null_midrank_qki.mean())
    assert abs(pooled - 0.5) < 0.05
    assert abs(float(nuclei.mean_null_rank_qki_at_miat.mean()) - 0.5) < 0.05
    assert abs(float(nuclei.frac_spots_ge_null_q90.mean()) - 0.10) < 0.05


def test_colocalized_puncta_rank_near_one_no_qki_cutoff_involved():
    nuclei, _ = _assoc(_spots_on_random_qki(22, colocalized=True))
    assert float(nuclei.mean_null_rank_qki_at_miat.min()) > 0.9
    assert float(nuclei.frac_spots_ge_null_q90.min()) > 0.8
    # qki_min far above every pixel leaves the rank untouched
    from fishsuite.core.qki_association import association_tables
    high, _ = association_tables(*_spots_on_random_qki(22, colocalized=True),
                                 pixel_size_um=0.13, miat_min=100, qki_min=1e9,
                                 sensitivity=(1.0,), n_null=200, seed=0)
    assert np.allclose(high.mean_null_rank_qki_at_miat, nuclei.mean_null_rank_qki_at_miat)


def test_new_metrics_are_threshold_free_across_sensitivity_levels():
    from fishsuite.core.qki_association import association_tables
    nuclei, _ = association_tables(*_spots_on_random_qki(23, colocalized=True),
                                   pixel_size_um=0.13, miat_min=100, qki_min=1050,
                                   sensitivity=(0.8, 1.0, 1.25), n_null=50, seed=0)
    cols = ["mean_null_rank_qki_at_miat", "pearson_r_nucleoplasm", "spearman_rho_nucleoplasm",
            "ccf_r0", "ccf_peak_shift_um_x", "costes_rand_p"]
    for _, group in nuclei.groupby("nucleus_id"):
        for col in cols:
            vals = group[col].to_numpy(dtype=float)
            assert np.allclose(vals, vals[0], equal_nan=True), col


def test_colocalized_chain_costes_small_and_ccf_centred():
    nuclei, _ = _assoc(_spots_on_random_qki(24, colocalized=True))
    assert float(nuclei.costes_rand_p.max()) < 0.05
    assert set(nuclei.costes_block_px) == {3}
    assert set(nuclei.ccf_peak_shift_px_x) == {0}
    assert set(nuclei.ccf_peak_shift_px_y) == {0}


def test_new_metrics_do_not_perturb_existing_placement_null_columns():
    from fishsuite.core import qki_association as qa
    data = _spots_on_random_qki(25, colocalized=True)
    nuclei, spots = _assoc(data)
    legacy = [c for c in qa.LEGACY_NUCLEUS_COLUMNS]
    assert list(nuclei.columns[:len(legacy)]) == legacy
    assert list(spots.columns[:len(qa.LEGACY_SPOT_COLUMNS)]) == list(qa.LEGACY_SPOT_COLUMNS)
    # Costes draws come from their own stream: turning them off leaves every legacy value equal
    off, off_spots = qa.association_tables(*data, pixel_size_um=0.13, miat_min=100, qki_min=1050,
                                           sensitivity=(1.0,), n_null=200, seed=0, n_costes=1)
    pd_equal = nuclei[legacy].equals(off[legacy])
    assert pd_equal
    assert spots[list(qa.LEGACY_SPOT_COLUMNS)].equals(off_spots[list(qa.LEGACY_SPOT_COLUMNS)])


def test_ccf_records_collected_per_nucleus_and_axis():
    from fishsuite.core.qki_association import association_tables
    records = []
    association_tables(*_spots_on_random_qki(26, colocalized=False), pixel_size_um=0.13,
                       miat_min=100, qki_min=1050, sensitivity=(1.0,), n_null=20, seed=0,
                       image="img", condition="c", well="w", ccf_records=records)
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
