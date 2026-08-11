"""Radial profile promoted to first-class, reported line-scan style.

The radial/annulus profile is the metric that actually fits a DIFFUSE partner:
it needs no thresholdable object in the partner channel, only the partner's
intensity as a function of distance from an object in the punctate channel. It
is the 2-D analogue of the line scan the source method used for its nuclear arm.

Until now it wrote ONLY the long-format ``coloc_radial_profile.csv`` through an
"extra carrier" — no per-nucleus columns and no per-image columns — so it could
not be tested at the replicate level at all. This covers (2026-08-10):

  * per-nucleus enrichment-by-distance columns
  * per-image columns: the spot-weighted pool AND the equal-weight
    per-nucleus rollup (the replicate-level statistic)
  * the mean ± 95% CI profile table and figure
  * the pixel-size fail-loud: a substituted µm/px silently rescales every
    distance bin, so the radial path now refuses to run on one

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
from fishsuite.core.io import ImageWrapper
from fishsuite.core.modes import rna_rna as _rna_rna
from fishsuite.core.modes.rna_rna import (
    radial_column_suffixes,
    radial_per_nucleus_columns,
    radial_pooled_per_image_columns,
)
from fishsuite.core.radial_profile_figure import (
    find_ring_columns,
    plot_radial_profile_ci,
    radial_profile_ci_table,
    restrict_to_analysis_set,
    _series_plan,
)


# ---------------------------------------------------------------------------
# COLUMN NAMING + SHAPING HELPERS
# ---------------------------------------------------------------------------
def test_radial_column_suffixes_match_the_pair_distance_convention():
    """Same 'distance in a column name' convention as
    ``paired_fraction_rna1_at_0p3um``, so the schema reads consistently."""
    assert radial_column_suffixes([0.25, 0.5, 0.75, 1.0]) == [
        "0p25um", "0p5um", "0p75um", "1um",
    ]


def test_radial_column_suffixes_drop_nonpositive_bins():
    """Ring construction drops non-positive bins, so the suffix list must too —
    otherwise ring i gets reported under bin i's label."""
    assert radial_column_suffixes([0.0, 0.5, 1.0]) == ["0p5um", "1um"]


def test_per_nucleus_columns_derive_enrichment_and_z():
    per_ring = [(20.0, 10.0, 2.0, 7), (12.0, 10.0, 4.0, 7)]
    out = radial_per_nucleus_columns(per_ring, ["0p25um", "0p5um"])
    assert out["rna2_radial_obs_at_0p25um"] == pytest.approx(20.0)
    assert out["rna2_radial_null_mean_at_0p25um"] == pytest.approx(10.0)
    assert out["rna2_radial_enrichment_at_0p25um"] == pytest.approx(2.0)
    assert out["rna2_radial_null_z_at_0p25um"] == pytest.approx(5.0)
    assert out["rna2_radial_enrichment_at_0p5um"] == pytest.approx(1.2)
    assert out["rna2_radial_null_z_at_0p5um"] == pytest.approx(0.5)


def test_per_nucleus_n_spots_is_one_column_not_one_per_ring():
    """Every ring samples the same nucleus's spots, so a per-ring count would be
    the same number repeated."""
    out = radial_per_nucleus_columns([(1.0, 1.0, 1.0, 5)] * 3,
                                     ["0p25um", "0p5um", "0p75um"])
    assert out["rna2_radial_n_spots"] == 5
    assert sum(1 for k in out if k.endswith("_n_spots")) == 1


def test_per_nucleus_columns_keep_their_key_set_for_an_unprofiled_nucleus():
    """A nucleus with no spots yields NaN, never an absent column."""
    sfx = ["0p25um", "0p5um"]
    profiled = radial_per_nucleus_columns([(1.0, 1.0, 1.0, 3)] * 2, sfx)
    missing = radial_per_nucleus_columns(None, sfx)
    assert set(profiled) == set(missing)
    assert missing["rna2_radial_n_spots"] == 0
    assert np.isnan(missing["rna2_radial_enrichment_at_0p25um"])


def test_pooled_per_image_columns_are_spot_weighted_means():
    obs = np.array([200.0, 100.0])
    nullmean = np.array([100.0, 100.0])
    nullsd = np.array([20.0, 40.0])
    wden = np.array([10.0, 10.0])
    out = radial_pooled_per_image_columns(obs, nullmean, nullsd, wden,
                                          ["0p25um", "0p5um"])
    assert out["rna2_radial_pooled_obs_at_0p25um"] == pytest.approx(20.0)
    assert out["rna2_radial_pooled_enrichment_at_0p25um"] == pytest.approx(2.0)
    assert out["n_spots_radial_at_0p25um"] == 10
    assert out["rna2_radial_pooled_enrichment_at_0p5um"] == pytest.approx(1.0)


def test_pooled_per_image_columns_nan_on_zero_weight():
    out = radial_pooled_per_image_columns(
        np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), ["0p25um"]
    )
    assert np.isnan(out["rna2_radial_pooled_enrichment_at_0p25um"])
    assert out["n_spots_radial_at_0p25um"] == 0


# ---------------------------------------------------------------------------
# MEAN +- 95% CI TABLE AND FIGURE
# ---------------------------------------------------------------------------
def test_find_ring_columns_sorts_by_radius_and_reads_the_partner_token():
    df = pd.DataFrame(columns=[
        "rna2_radial_enrichment_at_1p0um",
        "rna2_radial_enrichment_at_0p25um",
        "rna2_radial_obs_at_0p25um",          # not an enrichment column
        "coloc_pearson_r_rna1_rna2",          # unrelated
    ])
    rings = find_ring_columns(df)
    assert [r["ring_um"] for r in rings] == [0.25, 1.0]
    assert {r["partner"] for r in rings} == {"rna2"}


def test_find_ring_columns_handles_the_protein_relabel():
    """rna_protein renames rna2 -> protein, so the figure must follow."""
    df = pd.DataFrame(columns=["protein_radial_enrichment_at_0p5um"])
    rings = find_ring_columns(df)
    assert len(rings) == 1 and rings[0]["partner"] == "protein"


def test_ci_table_matches_a_hand_computed_t_interval():
    from scipy import stats as _stats

    vals = [1.0, 1.5, 2.0, 2.5]
    df = pd.DataFrame({"rna2_radial_enrichment_at_0p5um": vals})
    tab = radial_profile_ci_table(df)
    assert len(tab) == 1
    row = tab.iloc[0]
    mean = float(np.mean(vals))
    sd = float(np.std(vals, ddof=1))
    half = float(_stats.t.ppf(0.975, len(vals) - 1)) * sd / np.sqrt(len(vals))
    assert row["n_nuclei"] == 4
    assert row["mean"] == pytest.approx(mean)
    assert row["ci_low"] == pytest.approx(mean - half)
    assert row["ci_high"] == pytest.approx(mean + half)


def test_ci_table_drops_nonfinite_per_ring():
    df = pd.DataFrame({
        "rna2_radial_enrichment_at_0p5um": [1.0, float("nan"), 3.0],
    })
    row = radial_profile_ci_table(df).iloc[0]
    assert row["n_nuclei"] == 2
    assert row["mean"] == pytest.approx(2.0)


def test_ci_table_gives_no_interval_for_a_single_object():
    """One object cannot support a confidence interval; a zero-width ribbon
    would imply a certainty the single observation does not carry."""
    df = pd.DataFrame({"rna2_radial_enrichment_at_0p5um": [1.7]})
    row = radial_profile_ci_table(df).iloc[0]
    assert row["n_nuclei"] == 1
    assert row["mean"] == pytest.approx(1.7)
    assert np.isnan(row["ci_low"]) and np.isnan(row["ci_high"])


def test_ci_table_is_empty_without_radial_columns():
    assert len(radial_profile_ci_table(pd.DataFrame({"a": [1, 2]}))) == 0


def test_figure_is_written_with_ring_columns(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "rna2_radial_enrichment_at_0p25um": 2.0 + rng.normal(0, 0.1, 12),
        "rna2_radial_enrichment_at_0p5um": 1.5 + rng.normal(0, 0.1, 12),
        "rna2_radial_enrichment_at_1p0um": 1.0 + rng.normal(0, 0.1, 12),
    })
    out = plot_radial_profile_ci(df, tmp_path / "radial.png")
    assert out is not None and out.exists() and out.stat().st_size > 0


def test_figure_returns_none_rather_than_drawing_empty_axes(tmp_path):
    """No measurement -> no figure. Empty axes would look like a null result."""
    assert plot_radial_profile_ci(pd.DataFrame({"a": [1]}), tmp_path / "x.png") is None
    allnan = pd.DataFrame({"rna2_radial_enrichment_at_0p5um": [np.nan, np.nan]})
    assert plot_radial_profile_ci(allnan, tmp_path / "y.png") is None


# ---------------------------------------------------------------------------
# CONDITIONS ARE NEVER POOLED (2026-08-10)
#
# The runner hands this the RUN-WIDE concatenation of nuclei_metrics. Averaging
# a 6-condition preset into one ribbon labelled "mean across nuclei" produces a
# number describing no experiment — an over-expression arm, its non-targeting
# control and a no-probe control in one line. These are the regression tests for
# that; the figure-level ones assert the plan, because a PNG cannot be read back.
# ---------------------------------------------------------------------------
def _cond_frame():
    """Two conditions with clearly different enrichment, plus a sec-only well."""
    return pd.DataFrame({
        "condition": ["NT"] * 4 + ["MIAT-OE"] * 4 + ["NT"] * 2,
        "secondary_only": [False] * 8 + [True] * 2,
        "rna2_radial_enrichment_at_0p25um": [1.0, 1.0, 1.0, 1.0,
                                             3.0, 3.0, 3.0, 3.0,
                                             0.5, 0.5],
    })


def test_ci_table_groups_on_condition_instead_of_pooling():
    tab = radial_profile_ci_table(_cond_frame(), group_col="condition")
    assert set(tab["condition"]) == {"NT", "MIAT-OE"}
    # NT here is 4 real + 2 sec-only rows; the TABLE groups only (the sec-only
    # split is the figure's job), so this asserts grouping, not the split.
    oe = tab.loc[tab["condition"] == "MIAT-OE"].iloc[0]
    assert oe["mean"] == pytest.approx(3.0) and oe["n_nuclei"] == 4
    # Pooling would land between the arms and describe neither.
    pooled = radial_profile_ci_table(_cond_frame()).iloc[0]
    assert 1.0 < pooled["mean"] < 3.0


def test_ci_table_ungrouped_is_unchanged_without_the_column():
    """The old single-series call must keep working byte-for-byte."""
    df = pd.DataFrame({"rna2_radial_enrichment_at_0p5um": [1.0, 2.0, 3.0]})
    a = radial_profile_ci_table(df)
    b = radial_profile_ci_table(df, group_col="condition")  # column absent
    assert a.equals(b)
    assert "condition" not in a.columns


def test_one_panel_per_condition_and_seconly_is_its_own_series():
    plan = _series_plan(_cond_frame(), group_col="condition", confidence=0.95)
    labels = [e["label"] for e in plan]
    # Control-first ordering, then the pooled background control LAST.
    assert labels[0] == "NT" and "MIAT-OE" in labels
    assert plan[-1]["is_seconly"] and "secondary-only" in labels[-1]
    assert sum(1 for e in plan if e["is_seconly"]) == 1


def test_seconly_nuclei_are_excluded_from_the_condition_they_sit_in():
    """A no-primary / no-probe well carries a `condition` of its own — the arm it
    was acquired alongside. Grouping before splitting would fold background into
    that arm's ribbon."""
    plan = _series_plan(_cond_frame(), group_col="condition", confidence=0.95)
    nt = next(e for e in plan if e["label"] == "NT")
    row = nt["table"].iloc[0]
    assert row["n_nuclei"] == 4          # the 2 sec-only rows are NOT in here
    assert row["mean"] == pytest.approx(1.0)   # not dragged toward 0.5


def test_every_condition_gets_a_distinct_colour_and_none_is_the_reference_colour():
    from fishsuite.core.radial_profile_figure import _REFERENCE_COLOR

    df = pd.DataFrame({
        "condition": [f"c{i}" for i in range(6)],
        "rna2_radial_enrichment_at_0p25um": [1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
    })
    plan = _series_plan(df, group_col="condition", confidence=0.95)
    colors = [e["color"] for e in plan]
    assert len(plan) == 6
    assert len(set(colors)) == 6, "a repeated colour makes two arms one series"
    assert _REFERENCE_COLOR not in colors


def test_multi_condition_figure_is_written(tmp_path):
    out = plot_radial_profile_ci(_cond_frame(), tmp_path / "by_cond.png")
    assert out is not None and out.exists() and out.stat().st_size > 0


# ---------------------------------------------------------------------------
# THE FIGURE AND THE TABLE IT VISUALISES MUST COVER THE SAME NUCLEI (2026-08-10)
#
# The modes RETURN `nuclei_df_all` (every visited nucleus) while the per-image
# rollups are restricted to the sampled set, so an unfiltered figure reported a
# mean and an n over nuclei that per_image_summary.csv had excluded.
# ---------------------------------------------------------------------------
def _sampled_frame():
    return pd.DataFrame({
        "sampled_in_analysis": [True, True, False, False],
        "rna2_radial_enrichment_at_0p25um": [2.0, 2.0, 10.0, 10.0],
    })


def test_restrict_drops_the_non_sampled_nuclei_and_says_so():
    kept, label = restrict_to_analysis_set(_sampled_frame())
    assert len(kept) == 2
    assert "sampled" in label and "per_image_summary" in label


def test_restrict_is_a_no_op_when_apply_to_rollups_is_off():
    """There the rollups covered ALL eligible nuclei, so filtering the figure
    would MANUFACTURE the mismatch this exists to prevent."""
    kept, label = restrict_to_analysis_set(
        _sampled_frame(), restrict_to_sampled=False
    )
    assert len(kept) == 4
    assert "all analysed nuclei" in label


def test_restrict_is_a_no_op_when_sampling_never_ran():
    df = pd.DataFrame({"rna2_radial_enrichment_at_0p25um": [1.0, 2.0]})
    kept, label = restrict_to_analysis_set(df)
    assert len(kept) == 2 and label == "all analysed nuclei"


def test_the_figures_mean_matches_the_restricted_set_not_the_full_one():
    kept, _ = restrict_to_analysis_set(_sampled_frame())
    row = radial_profile_ci_table(kept).iloc[0]
    assert row["n_nuclei"] == 2
    assert row["mean"] == pytest.approx(2.0)     # not 6.0, the unfiltered mean


def test_verdict_columns_survive_a_csv_round_trip(tmp_path):
    """`Series.astype(bool)` reads the STRING 'False' as True, so a frame that
    came back through a CSV would have selected every nucleus while claiming to
    have selected the sample."""
    p = tmp_path / "nuclei_metrics.csv"
    _sampled_frame().to_csv(p, index=False)
    kept, _ = restrict_to_analysis_set(pd.read_csv(p, dtype={"sampled_in_analysis": str}))
    assert len(kept) == 2

    secs = _cond_frame()
    secs["secondary_only"] = secs["secondary_only"].map({True: "True", False: "False"})
    plan = _series_plan(secs, group_col="condition", confidence=0.95)
    nt = next(e for e in plan if e["label"] == "NT")
    assert nt["table"].iloc[0]["n_nuclei"] == 4


def test_runner_ties_the_figures_nucleus_set_to_apply_to_rollups():
    """Passing restrict unconditionally would filter a run whose rollups were
    deliberately left unrestricted — the same mismatch, reversed."""
    import inspect

    import fishsuite.runner as runner

    src = inspect.getsource(runner.run_batch)
    call = src[src.index("plot_radial_profile_ci("):]
    assert "restrict_to_sampled" in call[:600]
    assert "apply_to_rollups" in call[:600]


# ---------------------------------------------------------------------------
# END-TO-END synthetic stack: DAPI + punctate rna1 + DIFFUSE nuclear partner
# ---------------------------------------------------------------------------
DAPI_C, RNA_C, PART_C = 0, 1, 2
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
        hr, hc = disk((cy, cx), 7, shape=img.shape)
        img[hr, hc] = 200.0
    return img


def _rna_spot_plane():
    img = np.random.default_rng(22).uniform(2.0, 8.0, (H, W)).astype(np.float32)
    blob = np.zeros_like(img)
    rng = np.random.default_rng(33)
    for (cy, cx) in _nuclei_centers():
        for k in range(8):
            ang = 2 * np.pi * k / 8
            y = int(cy + 15 * np.sin(ang))
            x = int(cx + 15 * np.cos(ang))
            blob[y, x] += float(rng.uniform(3000.0, 6000.0))
    return img + gaussian_filter(blob, 1.1)


def _partner_plane_diffuse():
    """A nucleoplasm-filling partner — the case that has no thresholdable
    object, and so the case the radial profile exists for."""
    from skimage.draw import disk

    img = np.random.default_rng(44).uniform(2.0, 8.0, (H, W)).astype(np.float32)
    for (cy, cx) in _nuclei_centers():
        rr, cc = disk((cy, cx), 28, shape=img.shape)
        img[rr, cc] = 800.0
        hr, hc = disk((cy, cx), 7, shape=img.shape)
        img[hr, hc] = 20.0
    return img


def _czyx() -> np.ndarray:
    planes = [_dapi_plane(), _rna_spot_plane(), _partner_plane_diffuse()]
    return np.stack(
        [np.stack([p] * NZ, axis=0) for p in planes], axis=0
    ).astype(np.float32)


def _make_img(voxel_xy_nm) -> ImageWrapper:
    return ImageWrapper(
        path="synthetic_radial_first_class.tif",
        bio=_FakeBio(_czyx()),
        scene_idx=0,
        shape=(1, 3, NZ, H, W),
        channel_names=["DAPI", "RNA", "PART"],
        voxel_xy_nm=voxel_xy_nm,
        voxel_z_nm=300.0,
        n_channels=3,
        n_z=NZ,
    )


@pytest.fixture()
def fake_img() -> ImageWrapper:
    return _make_img(130.0)


def _base_cfg() -> FishsuiteConfig:
    cfg = FishsuiteConfig()
    cfg.channels.dapi = DAPI_C
    cfg.channels.rna = RNA_C
    cfg.channels.rna2 = PART_C
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


def _radial_cfg() -> FishsuiteConfig:
    cfg = _base_cfg()
    cfg.foci.compute_partner_intensity = True
    cfg.foci.compute_partner_radial_profile = True
    cfg.foci.partner_radial_bins_um = [0.25, 0.5, 0.75, 1.0]
    cfg.foci.partner_null_n = 60
    return cfg


def _run(cfg, img, monkeypatch):
    monkeypatch.setattr(_io, "read_image", lambda p: img)
    return _rna_rna.run_one(
        Path(img.path), condition="cond", sec_only=False, cfg=cfg,
    )


_SUFFIXES = ["0p25um", "0p5um", "0p75um", "1um"]


def test_radial_off_emits_no_radial_columns(fake_img, monkeypatch):
    """Default OFF stays byte-equivalent: no per-nucleus and no per-image
    radial columns appear."""
    res = _run(_base_cfg(), fake_img, monkeypatch)
    assert not [c for c in res.nuclei.columns if "_radial_" in str(c)]
    assert not [k for k in res.per_image if "_radial_" in str(k)]


def test_radial_on_emits_per_nucleus_columns(fake_img, monkeypatch):
    res = _run(_radial_cfg(), fake_img, monkeypatch)
    for sfx in _SUFFIXES:
        for field in ("obs", "null_mean", "enrichment", "null_z"):
            assert f"rna2_radial_{field}_at_{sfx}" in res.nuclei.columns
    assert "rna2_radial_n_spots" in res.nuclei.columns
    enr = pd.to_numeric(
        res.nuclei["rna2_radial_enrichment_at_0p25um"], errors="coerce"
    )
    assert np.isfinite(enr).any()


def test_radial_on_emits_both_per_image_views(fake_img, monkeypatch):
    """The spot-weighted pool AND the equal-weight per-nucleus rollup, so a
    per-image test can read either straight from per_image_summary."""
    res = _run(_radial_cfg(), fake_img, monkeypatch)
    for sfx in _SUFFIXES:
        assert f"rna2_radial_pooled_enrichment_at_{sfx}" in res.per_image
        assert f"rna2_radial_pooled_null_z_at_{sfx}" in res.per_image
        assert f"n_spots_radial_at_{sfx}" in res.per_image
        rcol = f"rna2_radial_enrichment_at_{sfx}"
        for prefix in ("mean_", "median_", "sd_", "n_nuclei_in_"):
            assert f"{prefix}{rcol}" in res.per_image


def test_per_image_radial_mean_matches_the_nucleus_columns(fake_img, monkeypatch):
    """The replicate-level statistic must equal an equal-weight mean over the
    per-nucleus column — not the spot-weighted pool."""
    res = _run(_radial_cfg(), fake_img, monkeypatch)
    rcol = "rna2_radial_enrichment_at_0p25um"
    vals = pd.to_numeric(res.nuclei[rcol], errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    assert vals.size >= 2
    assert res.per_image[f"mean_{rcol}"] == pytest.approx(float(vals.mean()))
    assert res.per_image[f"n_nuclei_in_{rcol}"] == int(vals.size)


def test_pooled_per_image_agrees_with_the_long_format_csv(fake_img, monkeypatch):
    """Wide per-image columns and coloc_radial_profile.csv come from ONE set of
    accumulators, so they must not disagree."""
    res = _run(_radial_cfg(), fake_img, monkeypatch)
    rad = res.extra["coloc_radial_profile"]
    for _, row in rad.iterrows():
        sfx = radial_column_suffixes([float(row["ring_um"])])[0]
        assert res.per_image[f"rna2_radial_pooled_enrichment_at_{sfx}"] == pytest.approx(
            float(row["enrichment"]), nan_ok=True
        )
        assert res.per_image[f"n_spots_radial_at_{sfx}"] == int(row["n_spots"])


def test_radial_per_image_rollup_honours_sampling(fake_img, monkeypatch):
    cfg = _radial_cfg()
    cfg.sampling.enabled = True
    cfg.sampling.n_per_unit = 1
    cfg.sampling.seed = 3
    cfg.sampling.apply_to_rollups = True
    res = _run(cfg, fake_img, monkeypatch)
    rcol = "rna2_radial_enrichment_at_0p25um"
    assert len(res.nuclei) >= 2
    assert res.per_image[f"n_nuclei_in_{rcol}"] <= 1


def test_radial_columns_feed_the_ci_figure(fake_img, monkeypatch, tmp_path):
    """End to end: the per-nucleus columns a run emits are exactly what the
    mean ± 95% CI figure consumes."""
    res = _run(_radial_cfg(), fake_img, monkeypatch)
    tab = radial_profile_ci_table(res.nuclei)
    assert list(tab["ring_um"]) == [0.25, 0.5, 0.75, 1.0]
    assert (tab["n_nuclei"] > 0).any()
    out = plot_radial_profile_ci(res.nuclei, tmp_path / "e2e_radial.png")
    assert out is not None and out.exists()


def test_radial_still_deterministic_after_promotion(fake_img, monkeypatch):
    cfg = _radial_cfg()
    r1 = _run(cfg, fake_img, monkeypatch)
    r2 = _run(cfg, fake_img, monkeypatch)
    np.testing.assert_allclose(
        pd.to_numeric(r1.nuclei["rna2_radial_enrichment_at_0p5um"], errors="coerce"),
        pd.to_numeric(r2.nuclei["rna2_radial_enrichment_at_0p5um"], errors="coerce"),
        rtol=0, atol=0, equal_nan=True,
    )


# ---------------------------------------------------------------------------
# PIXEL-SIZE FAIL-LOUD
# ---------------------------------------------------------------------------
def test_radial_refuses_an_undeclared_pixel_size(monkeypatch):
    """The µm -> px conversion is the only place the bins become pixels, so a
    substituted default rescales every ring while looking entirely plausible.
    Refuse instead of silently reporting the wrong distances."""
    img = _make_img(0.0)  # image declares no lateral pixel size
    with pytest.raises(ValueError, match="lateral pixel size"):
        _run(_radial_cfg(), img, monkeypatch)


def test_undeclared_pixel_size_is_still_fine_with_the_radial_profile_off(monkeypatch):
    """The refusal is scoped to the radial path. Every other feature keeps its
    existing substitute-and-continue behaviour."""
    img = _make_img(0.0)
    res = _run(_base_cfg(), img, monkeypatch)
    assert len(res.nuclei) > 0


def test_explicit_config_voxel_size_satisfies_the_radial_path(monkeypatch):
    """A user stating the pixel size is a declaration, not a guess, so
    foci.bigfish_voxel_size_nm unblocks the radial profile."""
    img = _make_img(0.0)
    cfg = _radial_cfg()
    cfg.foci.bigfish_voxel_size_nm = 130.0
    res = _run(cfg, img, monkeypatch)
    assert "rna2_radial_enrichment_at_0p25um" in res.nuclei.columns


def test_pixel_size_sets_the_ring_radii():
    """Halving the µm/px doubles every ring's radius in PIXELS. This is the
    rescaling the fail-loud exists to prevent: the column still says "0.25 µm"
    while sampling a ring twice that size, and nothing in the output betrays it.

    Tested at the conversion rather than end-to-end, because the ring means only
    diverge when the partner varies over the distances involved — a flat partner
    would hide a 2x error entirely."""
    from fishsuite.core.modes.rna_rna import _annulus_stencils

    bin_um = 0.25
    fine_px = bin_um / (65.0 / 1000.0)      # ~3.85 px
    coarse_px = bin_um / (130.0 / 1000.0)   # ~1.92 px
    assert fine_px == pytest.approx(2.0 * coarse_px)
    n_fine = _annulus_stencils([fine_px])[0][0].size
    n_coarse = _annulus_stencils([coarse_px])[0][0].size
    assert n_fine > n_coarse
    # ~area ratio: a 2x radius covers roughly 4x the pixels.
    assert n_fine / n_coarse > 2.5
