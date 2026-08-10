"""Per-image rollups of the pixel-coloc coefficients, and the NaN/rename fixes.

Covers four changes that share the same code path (2026-08-10):

  1. The nine pixel-coloc coefficients existed PER NUCLEUS ONLY, so the lead
     metric (threshold-free nuclear Pearson) could not be tested at the lab's
     replicate unit — the per-image mean — without hand-aggregation. Each now
     carries ``mean_`` / ``median_`` / ``sd_`` / ``n_nuclei_in_``, ungated.
  3. A nucleus with < 10 pixels returned 0.0 for every coefficient, which is
     indistinguishable from a genuine measurement of zero colocalization. Now
     NaN.
  4. ``median_nn_distance_rna*_um`` named TWO different quantities — all spots
     in the frame (per image) vs the spots of one cell (per nucleus). The
     per-image one is renamed to say so, and the other definition gets its own
     rollup.
  5. Four values ``compute_coloc_metrics`` computed and returned but the emit
     block never copied out.

Synthetic arrays only — no image fixtures.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import gaussian_filter

from fishsuite.config.schema import FishsuiteConfig
from fishsuite.core import io as _io
from fishsuite.core import metrics as _metrics
from fishsuite.core.io import ImageWrapper
from fishsuite.core.modes import rna_rna as _rna_rna
from fishsuite.core.modes.rna_rna import (
    _COLOC_ROLLUP_COLS,
    coloc_rollup_columns,
    rollup_mean_median_sd_n,
)


# ---------------------------------------------------------------------------
# (3) DEGENERATE NUCLEUS -> NaN, NOT 0.0
# ---------------------------------------------------------------------------
def test_degenerate_nucleus_returns_nan_not_zero():
    """< 10 pixels -> every coefficient NaN. 0.0 would read as a real
    measurement of zero colocalization and drag every mean toward zero."""
    r = np.arange(5, dtype=float)
    a = np.arange(5, dtype=float)
    out = _metrics.compute_coloc_metrics(r, a)
    for key in (
        "pearson_r", "spearman_rho", "li_icq", "cosine_overlap",
        "manders_m1", "manders_m2", "jaccard", "dice", "both_frac",
        "rna_mean", "ab_mean", "rna_frac_above_thr", "ab_frac_above_thr",
        "sum_r", "sum_a", "sum_product", "sum_min", "min_frac_r", "min_frac_a",
        "ab_enrich_in_rna_high", "rna_enrich_in_ab_high",
    ):
        assert np.isnan(out[key]), f"{key} should be NaN for a degenerate nucleus"


def test_degenerate_nucleus_keeps_npix_and_inf_threshold_tell():
    """``n_pix`` stays an honest int and the thresholds stay +inf — that
    infinity is the unique on-disk tell identifying a degenerate row."""
    out = _metrics.compute_coloc_metrics(np.zeros(3), np.zeros(3))
    assert out["n_pix"] == 3
    assert not isinstance(out["n_pix"], float) or float(out["n_pix"]) == 3.0
    assert out["rna_thr"] == float("inf")
    assert out["ab_thr"] == float("inf")


def test_ten_pixels_is_computed_not_degenerate():
    """The boundary is n_pix < 10, so exactly 10 pixels is a real computation.
    Perfectly correlated input must give Pearson 1.0, not NaN."""
    r = np.arange(10, dtype=float)
    out = _metrics.compute_coloc_metrics(r, 2.0 * r + 1.0)
    assert out["n_pix"] == 10
    assert out["pearson_r"] == pytest.approx(1.0)


def test_degenerate_nan_is_dropped_by_the_rollup_not_propagated():
    """One unmeasurable nucleus must reduce n, not void the whole image."""
    df = pd.DataFrame({"coloc_pearson_r_rna1_rna2": [0.8, float("nan"), 0.6]})
    out = coloc_rollup_columns(df)
    assert out["n_nuclei_in_coloc_pearson_r_rna1_rna2"] == 2
    assert out["mean_coloc_pearson_r_rna1_rna2"] == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# (1) ROLLUP HELPER SEMANTICS
# ---------------------------------------------------------------------------
def test_rollup_mean_median_sd_n_basic():
    mean, median, sd, n = rollup_mean_median_sd_n([1.0, 2.0, 3.0, 4.0])
    assert (mean, median, n) == (pytest.approx(2.5), pytest.approx(2.5), 4)
    assert sd == pytest.approx(np.std([1.0, 2.0, 3.0, 4.0], ddof=1))


def test_rollup_sd_is_nan_for_a_single_nucleus():
    """One observation has no spread. Reporting sd=0.0 would claim certainty
    the single measurement does not provide."""
    mean, median, sd, n = rollup_mean_median_sd_n([2.5])
    assert (mean, median, n) == (pytest.approx(2.5), pytest.approx(2.5), 1)
    assert np.isnan(sd)


def test_rollup_empty_and_all_nonfinite_give_n_zero():
    for vals in ([], [float("nan")], [np.inf, -np.inf, float("nan")]):
        mean, median, sd, n = rollup_mean_median_sd_n(vals)
        assert n == 0
        assert np.isnan(mean) and np.isnan(median) and np.isnan(sd)


def test_rollup_columns_key_set_is_frame_independent():
    """A missing column or an empty frame must still yield the key — otherwise
    per_image_summary.csv goes ragged between images."""
    full = coloc_rollup_columns(
        pd.DataFrame({c: [0.5, 0.6] for c in _COLOC_ROLLUP_COLS})
    )
    empty = coloc_rollup_columns(pd.DataFrame())
    assert set(full) == set(empty)
    assert len(full) == 4 * len(_COLOC_ROLLUP_COLS) == 36
    for col in _COLOC_ROLLUP_COLS:
        assert empty[f"n_nuclei_in_{col}"] == 0
        assert np.isnan(empty[f"mean_{col}"])


def test_pearson_is_the_first_rollup_metric():
    """Threshold-free nuclear Pearson is the lead coloc metric: it is the only
    one of the nine needing no thresholdable object in either channel."""
    assert _COLOC_ROLLUP_COLS[0] == "coloc_pearson_r_rna1_rna2"


# ---------------------------------------------------------------------------
# END-TO-END synthetic stack (DAPI + two punctate RNA channels)
# ---------------------------------------------------------------------------
DAPI_C, RNA_C, RNA2_C = 0, 1, 2
NZ = 4
H = W = 200


class _FakeBio:
    def __init__(self, czyx: np.ndarray):
        self._czyx = czyx

    def get_image_data(self, order: str, *, T: int = 0, C: int = 0):  # noqa: N803
        assert order == "ZYX"
        return self._czyx[C]


def _nuclei_centers():
    return [(70, 70), (70, 130), (130, 100)]


def _dapi_plane():
    from skimage.draw import disk

    img = np.random.default_rng(11).uniform(0.0, 20.0, (H, W)).astype(np.float32)
    for (cy, cx) in _nuclei_centers():
        rr, cc = disk((cy, cx), 28, shape=img.shape)
        img[rr, cc] += 3000.0
    return img


def _spot_plane(seed: int, offset: int = 0):
    img = np.random.default_rng(seed).uniform(2.0, 8.0, (H, W)).astype(np.float32)
    blob = np.zeros_like(img)
    rng = np.random.default_rng(seed + 1)
    for (cy, cx) in _nuclei_centers():
        for k in range(8):
            ang = 2 * np.pi * k / 8
            y = int(cy + 15 * np.sin(ang)) + offset
            x = int(cx + 15 * np.cos(ang))
            blob[y, x] += float(rng.uniform(3000.0, 6000.0))
    return img + gaussian_filter(blob, 1.1)


def _czyx() -> np.ndarray:
    planes = [_dapi_plane(), _spot_plane(22), _spot_plane(44, offset=2)]
    return np.stack(
        [np.stack([p] * NZ, axis=0) for p in planes], axis=0
    ).astype(np.float32)


@pytest.fixture()
def fake_img() -> ImageWrapper:
    return ImageWrapper(
        path="synthetic_coloc_rollups.tif",
        bio=_FakeBio(_czyx()),
        scene_idx=0,
        shape=(1, 3, NZ, H, W),
        channel_names=["DAPI", "RNA1", "RNA2"],
        voxel_xy_nm=130.0,
        voxel_z_nm=300.0,
        n_channels=3,
        n_z=NZ,
    )


def _base_cfg() -> FishsuiteConfig:
    cfg = FishsuiteConfig()
    cfg.channels.dapi = DAPI_C
    cfg.channels.rna = RNA_C
    cfg.channels.rna2 = RNA2_C
    cfg.channels.analysis_mode = "rna_rna"
    cfg.nuclei.backend = "otsu"
    cfg.nuclei.min_area_px = 120
    cfg.nuclei.max_area_px = 10_000_000
    cfg.nuclei.exclude_border = True
    cfg.nuclei.border_margin_px = 3
    cfg.z_stack.mode = "maxproj"
    cfg.cytoplasm.enabled = True
    cfg.foci.enabled = True
    cfg.foci.backend = "bigfish"
    cfg.foci.threshold_multiplier = 1.0
    cfg.foci.drop_floater_spots = False
    cfg.pixel_coloc.threshold_scope = "per_image"
    return cfg


def _run(cfg, img, monkeypatch):
    monkeypatch.setattr(_io, "read_image", lambda p: img)
    return _rna_rna.run_one(
        Path(img.path), condition="cond", sec_only=False, cfg=cfg,
    )


def test_all_nine_rollups_emitted_ungated(fake_img, monkeypatch):
    """No feature flag needed: the nine coefficients get their four rollup keys
    on a plain rna_rna run, because the per-image mean is the replicate unit."""
    res = _run(_base_cfg(), fake_img, monkeypatch)
    for col in _COLOC_ROLLUP_COLS:
        assert col in res.nuclei.columns, f"per-nucleus {col} missing"
        for prefix in ("mean_", "median_", "sd_", "n_nuclei_in_"):
            assert f"{prefix}{col}" in res.per_image, f"{prefix}{col} missing"


def test_rollup_matches_hand_aggregation_of_the_nucleus_column(fake_img, monkeypatch):
    """The emitted mean/median/sd must equal a hand aggregation of the
    per-nucleus column — the exact step this change removes the need for."""
    res = _run(_base_cfg(), fake_img, monkeypatch)
    col = "coloc_pearson_r_rna1_rna2"
    vals = pd.to_numeric(res.nuclei[col], errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    assert vals.size >= 2
    assert res.per_image[f"mean_{col}"] == pytest.approx(float(vals.mean()))
    assert res.per_image[f"median_{col}"] == pytest.approx(float(np.median(vals)))
    assert res.per_image[f"sd_{col}"] == pytest.approx(float(vals.std(ddof=1)))
    assert res.per_image[f"n_nuclei_in_{col}"] == int(vals.size)


def test_n_nuclei_in_is_the_denominator_not_the_row_count(fake_img, monkeypatch):
    """n_nuclei_in_* counts the nuclei that actually contributed a finite
    value, which is what a reviewer asks for."""
    res = _run(_base_cfg(), fake_img, monkeypatch)
    col = "coloc_pearson_r_rna1_rna2"
    n_finite = int(
        np.isfinite(
            pd.to_numeric(res.nuclei[col], errors="coerce").to_numpy(dtype=float)
        ).sum()
    )
    assert res.per_image[f"n_nuclei_in_{col}"] == n_finite


def test_rollups_honour_nucleus_sampling(fake_img, monkeypatch):
    """With sampling on and apply_to_rollups True, the rollups must cover the
    sampled rows only — and n_nuclei_in_* must reveal it."""
    cfg = _base_cfg()
    cfg.sampling.enabled = True
    cfg.sampling.n_per_unit = 1
    cfg.sampling.seed = 3
    cfg.sampling.apply_to_rollups = True
    res = _run(cfg, fake_img, monkeypatch)

    col = "coloc_pearson_r_rna1_rna2"
    nuc = res.nuclei
    # Every visited nucleus is still a row; only the rollups restrict.
    assert len(nuc) >= 2
    sampled = nuc[nuc["sampled_in_analysis"].astype(bool)]
    assert len(sampled) == 1
    assert res.per_image[f"n_nuclei_in_{col}"] == 1
    assert res.per_image[f"mean_{col}"] == pytest.approx(
        float(pd.to_numeric(sampled[col], errors="coerce").iloc[0])
    )


def test_rollups_cover_every_nucleus_when_sampling_off(fake_img, monkeypatch):
    res = _run(_base_cfg(), fake_img, monkeypatch)
    col = "coloc_pearson_r_rna1_rna2"
    assert res.per_image[f"n_nuclei_in_{col}"] == len(res.nuclei)


# ---------------------------------------------------------------------------
# (5) THE FOUR COMPUTED-BUT-DISCARDED VALUES
# ---------------------------------------------------------------------------
def test_overlap_coefficient_family_is_emitted(fake_img, monkeypatch):
    res = _run(_base_cfg(), fake_img, monkeypatch)
    for col in (
        "coloc_sum_min_rna1_rna2", "coloc_sum_product_rna1_rna2",
        "coloc_min_frac_rna1", "coloc_min_frac_rna2",
    ):
        assert col in res.nuclei.columns
    assert np.isfinite(
        pd.to_numeric(res.nuclei["coloc_min_frac_rna1"], errors="coerce")
    ).any()


def test_min_frac_columns_are_sum_min_over_each_channel_total():
    """min_frac_<ch> = sum(min(r, a)) / sum(<ch>): that channel's overlapping
    fraction. Threshold-free, so unlike Manders it does not move with the mask."""
    r = np.array([10.0] * 10)
    a = np.array([4.0] * 10)
    out = _metrics.compute_coloc_metrics(r, a)
    assert out["sum_min"] == pytest.approx(40.0)
    assert out["min_frac_r"] == pytest.approx(40.0 / 100.0)
    assert out["min_frac_a"] == pytest.approx(1.0)


def test_sum_product_is_the_raw_dot_product():
    r = np.arange(1.0, 11.0)
    a = np.arange(1.0, 11.0)
    out = _metrics.compute_coloc_metrics(r, a)
    assert out["sum_product"] == pytest.approx(float(np.dot(r, a)))


# ---------------------------------------------------------------------------
# (4) THE median_nn_distance DISAMBIGUATION
# ---------------------------------------------------------------------------
def test_per_image_nn_median_is_renamed_to_state_its_scope(fake_img, monkeypatch):
    """The per-image value covers every spot in the frame; the old name is gone
    from per_image so it can no longer be confused with the per-nucleus one."""
    res = _run(_base_cfg(), fake_img, monkeypatch)
    for ch in ("rna1", "rna2"):
        assert f"median_nn_distance_{ch}_um_all_spots_in_frame" in res.per_image
        assert f"median_nn_distance_{ch}_um" not in res.per_image


def test_per_nucleus_nn_median_column_name_is_untouched(fake_img, monkeypatch):
    """Only the PER-IMAGE column was renamed. The per-nucleus column keeps its
    name — renaming both would have broken the per-nucleus readers too."""
    res = _run(_base_cfg(), fake_img, monkeypatch)
    for ch in ("rna1", "rna2"):
        assert f"median_nn_distance_{ch}_um" in res.nuclei.columns


def test_per_nucleus_nn_rollup_is_added_and_differs_from_frame_scope(fake_img, monkeypatch):
    """The new rollup aggregates the per-NUCLEUS definition. The two scopes are
    genuinely different quantities, which is why one name for both was a bug."""
    res = _run(_base_cfg(), fake_img, monkeypatch)
    for ch in ("rna1", "rna2"):
        col = f"median_nn_distance_{ch}_um"
        for prefix in ("mean_", "median_", "sd_"):
            assert f"{prefix}{col}_per_nucleus" in res.per_image
        assert f"n_nuclei_in_{col}" in res.per_image
        vals = pd.to_numeric(res.nuclei[col], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            assert res.per_image[f"mean_{col}_per_nucleus"] == pytest.approx(
                float(vals.mean())
            )


# ---------------------------------------------------------------------------
# ADDITIVITY: nothing pre-existing changed
# ---------------------------------------------------------------------------
def test_existing_per_nucleus_coloc_columns_still_present(fake_img, monkeypatch):
    res = _run(_base_cfg(), fake_img, monkeypatch)
    for col in (
        "manders_rna1_in_rna2", "manders_rna2_in_rna1",
        "coloc_pearson_r_rna1_rna2", "coloc_spearman_rho_rna1_rna2",
        "coloc_li_icq_rna1_rna2", "coloc_cosine_overlap_rna1_rna2",
        "coloc_jaccard_rna1_rna2", "coloc_dice_rna1_rna2",
        "coloc_both_frac_rna1_rna2", "coloc_mask_thr_rna1", "coloc_mask_thr_rna2",
    ):
        assert col in res.nuclei.columns


# ---------------------------------------------------------------------------
# rna_protein RELABELING of the new names
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "rna2_name, protein_name",
    [
        ("mean_coloc_pearson_r_rna1_rna2", "mean_coloc_pearson_r_rna_protein"),
        ("sd_manders_rna1_in_rna2", "sd_manders_rna1_in_protein"),
        ("n_nuclei_in_coloc_dice_rna1_rna2", "n_nuclei_in_coloc_dice_rna_protein"),
        ("coloc_sum_min_rna1_rna2", "coloc_sum_min_rna_protein"),
        ("coloc_min_frac_rna2", "coloc_min_frac_protein"),
        ("coloc_min_frac_rna1", "coloc_min_frac_rna1"),  # RNA side unchanged
        ("median_nn_distance_rna2_um_all_spots_in_frame",
         "median_nn_distance_protein_um_all_spots_in_frame"),
        ("mean_median_nn_distance_rna2_um_per_nucleus",
         "mean_median_nn_distance_protein_um_per_nucleus"),
        ("rna2_radial_enrichment_at_0p25um", "protein_radial_enrichment_at_0p25um"),
    ],
)
def test_new_columns_relabel_cleanly_for_rna_protein(rna2_name, protein_name):
    """rna_protein maps the antibody channel through rna_rna's rna2 slot, so
    every new name has to survive the rna2 -> protein rewrite intact."""
    from fishsuite.core.modes.rna_protein import _relabel_rna2_to_protein

    assert _relabel_rna2_to_protein(rna2_name) == protein_name


def test_relabeling_the_new_names_creates_no_collisions():
    """A collision would make the relabeler DROP one of the two columns."""
    from fishsuite.core.modes.rna_protein import _relabel_rna2_to_protein

    names = []
    for base in _COLOC_ROLLUP_COLS:
        names.extend(f"{p}{base}" for p in ("mean_", "median_", "sd_", "n_nuclei_in_"))
    names += [
        "coloc_sum_min_rna1_rna2", "coloc_sum_product_rna1_rna2",
        "coloc_min_frac_rna1", "coloc_min_frac_rna2",
        "median_nn_distance_rna1_um_all_spots_in_frame",
        "median_nn_distance_rna2_um_all_spots_in_frame",
    ]
    relabelled = [_relabel_rna2_to_protein(n) for n in names]
    assert len(relabelled) == len(set(relabelled))


# ---------------------------------------------------------------------------
# EXCEL LEGEND — every new column must be documented, and stay documented
# ---------------------------------------------------------------------------
def test_excel_glossary_base_list_matches_the_rollup_list():
    """excel_report duplicates the nine base names rather than importing them
    (importing would drag skimage into every workbook write). This is the guard
    that makes the duplication safe: drift fails here instead of silently
    leaving columns undocumented."""
    from fishsuite.core.excel_report import _COLOC_ROLLUP_BASES

    assert tuple(b for b, _, _ in _COLOC_ROLLUP_BASES) == _COLOC_ROLLUP_COLS


def test_every_new_rollup_column_has_a_legend_entry(fake_img, monkeypatch):
    from fishsuite.core.excel_report import PER_IMAGE_GLOSSARY

    res = _run(_base_cfg(), fake_img, monkeypatch)
    for col in _COLOC_ROLLUP_COLS:
        for prefix in ("mean_", "median_", "sd_", "n_nuclei_in_"):
            key = f"{prefix}{col}"
            assert key in res.per_image
            assert key in PER_IMAGE_GLOSSARY, f"{key} is emitted but undocumented"


def test_renamed_and_new_nn_columns_are_documented():
    from fishsuite.core.excel_report import (
        PER_IMAGE_GLOSSARY, PER_NUCLEUS_GLOSSARY,
    )

    for ch in ("rna1", "rna2"):
        assert f"median_nn_distance_{ch}_um_all_spots_in_frame" in PER_IMAGE_GLOSSARY
        # The stale per-image key must be gone, or the legend would still claim
        # the frame-scope column exists under the colliding name.
        assert f"median_nn_distance_{ch}_um" not in PER_IMAGE_GLOSSARY
        # The per-nucleus entry keeps its name — only per-image was renamed.
        assert f"median_nn_distance_{ch}_um" in PER_NUCLEUS_GLOSSARY
        assert f"mean_median_nn_distance_{ch}_um_per_nucleus" in PER_IMAGE_GLOSSARY


def test_overlap_family_columns_are_documented():
    from fishsuite.core.excel_report import PER_NUCLEUS_GLOSSARY

    for col in (
        "coloc_sum_min_rna1_rna2", "coloc_sum_product_rna1_rna2",
        "coloc_min_frac_rna1", "coloc_min_frac_rna2",
    ):
        assert col in PER_NUCLEUS_GLOSSARY


def test_nn_legend_states_the_cross_channel_direction():
    """The legend used to describe a within-channel spacing metric, but the code
    measures RNA1 -> nearest RNA2. Wrong units of meaning, right units of length."""
    from fishsuite.core.excel_report import (
        PER_IMAGE_GLOSSARY, PER_NUCLEUS_GLOSSARY,
    )

    per_img = PER_IMAGE_GLOSSARY["median_nn_distance_rna1_um_all_spots_in_frame"][2]
    assert "RNA2" in per_img
    per_nuc = PER_NUCLEUS_GLOSSARY["median_nn_distance_rna1_um"][2]
    assert "RNA2" in per_nuc


def test_no_surviving_nucleus_hits_the_degenerate_path(fake_img, monkeypatch):
    """The +inf threshold tell must be absent: with any realistic
    nuclei.min_area_px a surviving nucleus cannot have fewer than 10 pixels, so
    the 0.0 -> NaN change is a no-op on real data."""
    res = _run(_base_cfg(), fake_img, monkeypatch)
    for col in ("coloc_mask_thr_rna1", "coloc_mask_thr_rna2"):
        vals = pd.to_numeric(res.nuclei[col], errors="coerce").to_numpy(dtype=float)
        assert np.isfinite(vals).all(), f"{col} carries a degenerate +inf row"
