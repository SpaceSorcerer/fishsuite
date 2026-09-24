"""Texture-matched placement null for the uniform-position percentile score (UPP).

SENSITIVITY ONLY (Astra fig1 review F1). The texture-matched null draws each
footprint's K placements only from admissible positions in the same
within-nucleus stratum of (footprint-mean DAPI quantile) x (normalized radial
position quantile). It must remove association that is due only to MIAT and
QKI sharing a DAPI-defined zone, while keeping punctum-level association.

Tolerances: under exchangeability each spot score is uniform on {0..K}/K
(variance (K+2)/(12K) ~= 1/12), so the pooled spot mean has SE
sqrt(1/(12 n)); the fraction >= 0.90 is Bernoulli(21/201). Every tolerance
below is 4 SE at the realised number of scored spots (two-sided
P(|Z| > 4) ~= 6e-5 per assertion if spots were independent; spots in one
nucleus share a QKI field, so 4 rather than 3 SE leaves room for that).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import ndimage

from fishsuite.core.footprint_null import MiatFootprint, exact_footprint_position_null

K = 200
CHANCE_Q90 = 21 / 201


def _footprint(index, y, x, radius=1):
    dy, dx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    dy, dx = dy.ravel(), dx.ravel()
    return MiatFootprint(index, int(y), int(x), y + dy, x + dx, dy, dx,
                         "synthetic", None, True, None, 100.0, 0.0)


def _field(kind, *, n_rows=8, n_cols=10, tile=72, radius=30, n_spots=15, seed=0):
    """One image of n_rows x n_cols disk nuclei with a known generative model.

    shared_zone : DAPI-bright zone (x > cx + 4); QKI +300 in the zone; MIAT
                  centres 80 % in the zone. No MIAT-QKI association beyond it.
    coloc       : flat DAPI texture; QKI +400 under every MIAT footprint.
    independent : DAPI-bright zone; QKI independent smooth noise; MIAT uniform.
    """
    rng = np.random.default_rng(seed)
    height, width = n_rows * tile, n_cols * tile
    labels = np.zeros((height, width), dtype=np.int32)
    zone = np.zeros((height, width), dtype=bool)
    yy, xx = np.mgrid[0:tile, 0:tile]
    c = tile // 2
    disk = (yy - c) ** 2 + (xx - c) ** 2 <= radius ** 2
    local_zone = disk & (xx - c > 4)
    inner = ndimage.binary_erosion(disk, structure=np.ones((3, 3), bool))
    footprints, label = [], 0
    for row in range(n_rows):
        for col in range(n_cols):
            label += 1
            oy, ox = row * tile, col * tile
            labels[oy:oy + tile, ox:ox + tile][disk] = label
            if kind != "coloc":
                zone[oy:oy + tile, ox:ox + tile] = local_zone
            if kind == "shared_zone":
                in_zone = np.argwhere(inner & local_zone)
                out_zone = np.argwhere(inner & ~local_zone)
                picks = [in_zone[rng.integers(len(in_zone))] if rng.random() < 0.8
                         else out_zone[rng.integers(len(out_zone))] for _ in range(n_spots)]
            else:
                pool = np.argwhere(inner)
                picks = [pool[rng.integers(len(pool))] for _ in range(n_spots)]
            for y, x in picks:
                footprints.append(_footprint(len(footprints), oy + y, ox + x))
    noise = ndimage.gaussian_filter(rng.normal(size=(height, width)), 2.0)
    qki = 1000.0 + 60.0 * noise / noise.std()
    dapi = 1000.0 + 800.0 * zone + rng.normal(0.0, 30.0, size=(height, width))
    miat = np.zeros((height, width))
    for fp in footprints:
        miat[fp.y_px, fp.x_px] = 100.0
    if kind == "shared_zone":
        qki = qki + 300.0 * zone
    if kind == "coloc":
        for fp in footprints:
            qki[fp.y_px, fp.x_px] += 400.0
    return miat, qki, labels, footprints, dapi


def _tables(data, **kwargs):
    from fishsuite.core.qki_association import association_tables
    from fishsuite.core.texture_null import TextureNullParams
    miat, qki, labels, footprints, dapi = data
    options = dict(pixel_size_um=0.13, miat_min=50, qki_min=1100, sensitivity=(1.0,),
                   n_null=K, seed=0, n_costes=5, dapi=dapi,
                   texture_null=TextureNullParams(), rotation_upp=True)
    options.update(kwargs)
    return association_tables(miat, qki, labels, footprints, **options)


def _pooled(spots, column):
    values = spots[column].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return values, float(values.mean()), 4.0 * np.sqrt(1.0 / (12.0 * values.size))


@pytest.fixture(scope="module")
def shared_zone():
    return _tables(_field("shared_zone", seed=11))


@pytest.fixture(scope="module")
def coloc():
    return _tables(_field("coloc", seed=12))


@pytest.fixture(scope="module")
def independent():
    return _tables(_field("independent", seed=13))


# --------------------------------------------------------------- acceptance (1)
def test_shared_dapi_zone_inflates_uniform_upp_but_not_texture_matched(shared_zone):
    nuclei, spots = shared_zone
    uni, uni_mean, uni_tol = _pooled(spots, "uniform_position_percentile_qki")
    tex, tex_mean, tex_tol = _pooled(spots, "upp_texture_matched_qki")
    assert tex.size >= 0.9 * len(spots)  # the null scores almost every spot
    assert uni_mean > 0.5 + uni_tol and uni_mean > 0.6  # confounded reference is inflated
    assert abs(tex_mean - 0.5) < tex_tol, (tex_mean, tex_tol)
    frac = float(np.mean(tex >= 0.9 - 1e-12))
    assert abs(frac - CHANCE_Q90) < 4 * np.sqrt(CHANCE_Q90 * (1 - CHANCE_Q90) / tex.size)
    well = nuclei.upp_texture_matched_mean_qki.mean()
    assert abs(well - 0.5) < 4.0 * np.sqrt(1.0 / (12.0 * nuclei.upp_texture_matched_n_spots_scored.min())
                                           / len(nuclei))


# --------------------------------------------------------------- acceptance (2)
def test_true_punctum_colocalization_in_uniform_texture_scores_near_one(coloc):
    nuclei, spots = coloc
    for column in ("uniform_position_percentile_qki", "upp_texture_matched_qki"):
        values, mean, _ = _pooled(spots, column)
        assert mean > 0.9, (column, mean)
        assert np.mean(values >= 0.9 - 1e-12) > 0.8, column
    assert nuclei.upp_texture_matched_mean_qki.min() > 0.8


# --------------------------------------------------------------- acceptance (3)
def test_independent_channels_score_near_half_under_both_nulls(independent):
    _, spots = independent
    for column in ("uniform_position_percentile_qki", "upp_texture_matched_qki"):
        values, mean, tol = _pooled(spots, column)
        assert abs(mean - 0.5) < tol, (column, mean, tol)
        frac = float(np.mean(values >= 0.9 - 1e-12))
        assert abs(frac - CHANCE_Q90) < 4 * np.sqrt(CHANCE_Q90 * (1 - CHANCE_Q90) / values.size), column


def test_rotation_upp_reported_near_half_for_independent_and_high_for_coloc(independent, coloc):
    _, spots = independent
    values, mean, tol = _pooled(spots, "upp_rotation_qki")
    assert values.size > 0.5 * len(spots)
    assert abs(mean - 0.5) < tol, (mean, tol)
    _, mean_coloc, _ = _pooled(coloc[1], "upp_rotation_qki")
    assert mean_coloc > 0.9


# --------------------------------------------------------------- acceptance (4)
def test_stratum_kept_when_it_has_enough_positions():
    from fishsuite.core.texture_null import select_stratum
    dapi_bin = np.zeros(100, int)
    radial_bin = np.repeat(np.arange(5), 20)
    members, merged, partner, reason = select_stratum(dapi_bin, radial_bin, 45, n_radial=5, min_positions=20)
    assert not merged and partner == -1 and reason == ""
    assert members.sum() == 20 and members[40:60].all()


def test_sparse_stratum_merges_with_the_larger_adjacent_radial_bin():
    from fishsuite.core.texture_null import select_stratum
    radial_bin = np.array([0] * 30 + [1] * 5 + [2] * 18)
    dapi_bin = np.zeros(radial_bin.size, int)
    members, merged, partner, reason = select_stratum(dapi_bin, radial_bin, 32, n_radial=5, min_positions=20)
    assert merged and partner == 0 and reason == ""
    assert members.sum() == 35 and not members[35:].any()


def test_merge_only_within_the_same_dapi_bin_and_edge_bin_has_one_neighbour():
    from fishsuite.core.texture_null import select_stratum
    radial_bin = np.array([4] * 5 + [3] * 30 + [3] * 30)
    dapi_bin = np.array([2] * 5 + [1] * 30 + [2] * 30)
    members, merged, partner, reason = select_stratum(dapi_bin, radial_bin, 0, n_radial=5, min_positions=20)
    assert merged and partner == 3 and reason == ""
    assert members.sum() == 35 and members[35:].all() and not members[5:35].any()


def test_still_sparse_after_merge_is_na_with_reason():
    from fishsuite.core.texture_null import select_stratum
    radial_bin = np.array([0] * 4 + [1] * 5 + [2] * 6)
    dapi_bin = np.zeros(radial_bin.size, int)
    members, merged, partner, reason = select_stratum(dapi_bin, radial_bin, 6, n_radial=5, min_positions=20)
    assert members is None and merged and partner == 2
    assert reason == "SPARSE_STRATUM_AFTER_MERGE"


def test_tiny_nucleus_marks_spots_na_and_records_counts():
    labels = np.zeros((20, 20), np.int32)
    labels[5:14, 5:14] = 1  # 9x9: 49 admissible centres for a 3x3 footprint
    rng = np.random.default_rng(0)
    qki, dapi = rng.normal(1000, 50, (20, 20)), rng.normal(1000, 50, (20, 20))
    fps = [_footprint(0, 8, 8), _footprint(1, 10, 11)]
    miat = np.zeros((20, 20))
    nuclei, spots = _tables((miat, qki, labels, fps, dapi))
    row = nuclei.iloc[0]
    assert np.isnan(row.upp_texture_matched_mean_qki)
    assert row.na_reason_upp_texture_matched == "ALL_SPOTS_NA"
    assert row.upp_texture_matched_n_spots_na == 2 and row.upp_texture_matched_n_spots_scored == 0
    assert "SPARSE_STRATUM_AFTER_MERGE=2" in row.upp_texture_matched_na_reason_counts
    assert spots.na_reason_upp_texture_matched_spot.eq("SPARSE_STRATUM_AFTER_MERGE").all()
    assert spots.texture_stratum_merged.all()
    # A 1x1 grid of strata with the gate at 20 scores both spots.
    from fishsuite.core.texture_null import TextureNullParams
    nuclei, spots = _tables((miat, qki, labels, fps, dapi),
                            texture_null=TextureNullParams(n_dapi_bins=1, n_radial_bins=1, min_positions=20))
    assert nuclei.iloc[0].upp_texture_matched_n_spots_scored == 2
    assert spots.texture_stratum_n_positions.eq(49).all()


def test_no_spots_gives_n0():
    labels = np.zeros((20, 20), np.int32)
    labels[2:18, 2:18] = 1
    flat = np.full((20, 20), 1000.0)
    nuclei, spots = _tables((np.zeros((20, 20)), flat, labels, [], flat))
    assert nuclei.iloc[0].na_reason_upp_texture_matched == "N0"
    assert nuclei.iloc[0].na_reason_upp_rotation == "N0"
    assert spots.empty


# ------------------------------------------------------------ construction
def test_admissible_centres_match_the_existing_placement_null_domain():
    from fishsuite.core.texture_null import admissible_centers
    labels = np.zeros((40, 40), np.int32)
    labels[4:36, 6:30] = 1
    region = labels == 1
    region[15:20, 15:20] = False  # nucleolus-like hole
    qki = np.random.default_rng(1).normal(1000, 50, (40, 40))
    fp = _footprint(0, 10, 10, radius=2)
    centers = admissible_centers(fp, region, qki)
    reference = exact_footprint_position_null(qki, fp, region, n_null=5, rng=np.random.default_rng(0))
    assert centers.shape[0] == reference.valid_center_count


def test_radial_position_is_boundary_distance_over_max_inscribed_distance():
    from fishsuite.core.texture_null import normalized_radial_map
    mask = np.zeros((41, 41), bool)
    yy, xx = np.mgrid[0:41, 0:41]
    mask[(yy - 20) ** 2 + (xx - 20) ** 2 <= 15 ** 2] = True
    radial = normalized_radial_map(mask)
    assert radial[20, 20] == pytest.approx(1.0)
    assert radial[20, 5] < 0.1 and np.all(radial[~mask] == 0)
    edge = np.zeros((10, 10), bool)
    edge[:, :5] = True  # touches the image border: border is a boundary too
    assert normalized_radial_map(edge)[5, 0] < normalized_radial_map(edge)[5, 2]


def test_within_nucleus_quantile_bins_are_quintiles():
    from fishsuite.core.texture_null import quantile_bins
    values = np.arange(100, dtype=float)
    q, bins = quantile_bins(values, 5)
    assert np.bincount(bins).tolist() == [20] * 5
    assert q.min() > 0 and q.max() < 1
    q, bins = quantile_bins(np.ones(10), 5)
    assert np.unique(bins).size == 1  # ties share a bin


# ------------------------------------------------------------ non-interference
def test_existing_columns_are_value_identical_and_absent_by_default():
    from fishsuite.core.qki_association import association_tables, NUCLEUS_COLUMNS, SPOT_COLUMNS
    data = _field("independent", n_rows=2, n_cols=2, seed=3)
    on_nuc, on_spot = _tables(data)
    miat, qki, labels, fps, _ = data
    off_nuc, off_spot = association_tables(miat, qki, labels, fps, pixel_size_um=0.13, miat_min=50,
                                           qki_min=1100, sensitivity=(1.0,), n_null=K, seed=0, n_costes=5)
    assert list(off_nuc.columns) == NUCLEUS_COLUMNS and list(off_spot.columns) == SPOT_COLUMNS
    pd.testing.assert_frame_equal(on_nuc[NUCLEUS_COLUMNS], off_nuc)
    pd.testing.assert_frame_equal(on_spot[SPOT_COLUMNS], off_spot)
    assert on_nuc.columns[len(NUCLEUS_COLUMNS):].str.contains("texture|rotation").all()


def test_texture_null_is_seeded_and_needs_dapi():
    from fishsuite.core.texture_null import TextureNullParams
    data = _field("independent", n_rows=1, n_cols=2, seed=4)
    a, b = _tables(data)[1], _tables(data)[1]
    pd.testing.assert_frame_equal(a, b)
    c = _tables(data, seed=1)[1]
    assert not np.allclose(a.upp_texture_matched_qki, c.upp_texture_matched_qki, equal_nan=True)
    with pytest.raises(ValueError, match="dapi"):
        _tables(data, dapi=None)
    with pytest.raises(ValueError):
        TextureNullParams(n_dapi_bins=0)


def test_column_definitions_label_sensitivity_only():
    from fishsuite.core.qki_association import COLUMN_DEFINITIONS, TEXTURE_NUCLEUS_COLUMNS, \
        TEXTURE_SPOT_COLUMNS
    for name in TEXTURE_NUCLEUS_COLUMNS + TEXTURE_SPOT_COLUMNS:
        assert "SENSITIVITY ONLY" in COLUMN_DEFINITIONS[name], name


# ------------------------------------------------------------ adapter / summary
def test_adapter_writes_sensitivity_tables_and_logs_parameters(tmp_path):
    import h5py
    from fishsuite.core.qki_association_postrun import run_qki_association
    from fishsuite.core.texture_null import TextureNullParams
    from test_qki_association import _assoc_cached_run, OPTICS
    root = _assoc_cached_run(tmp_path / "source")
    plain = run_qki_association(root, tmp_path / "plain", miat_min=10, qki_min=10, n_null=10, optics=OPTICS)
    out = run_qki_association(root, tmp_path / "tex", miat_min=10, qki_min=10, n_null=10, optics=OPTICS,
                              texture_null=TextureNullParams(min_positions=1), rotation_upp=True)
    base = pd.read_csv(plain / "qki_association_per_nucleus.csv")
    tex = pd.read_csv(out / "qki_association_per_nucleus.csv")
    pd.testing.assert_frame_equal(tex[base.columns], base)
    assert "upp_texture_matched_mean_qki" in tex
    per_well = pd.read_csv(out / "texture_null_per_well.csv")
    per_arm = pd.read_csv(out / "texture_null_per_arm.csv")
    assert {"well_mean_upp_uniform", "well_mean_upp_texture_matched", "well_frac_ge_0p90_upp_uniform",
            "well_frac_ge_0p90_upp_texture_matched", "texture_na_rate_spots"} <= set(per_well.columns)
    assert len(per_arm) == 1
    log = (out / "command.log").read_text(encoding="utf-8")
    assert "texture_null_params" in log and "SENSITIVITY ONLY" in log and "--texture-null" in log
    md = (out / "qki_association_columns.md").read_text(encoding="utf-8")
    assert "SENSITIVITY ONLY" in md
    plain_log = (plain / "command.log").read_text(encoding="utf-8")
    assert "texture_null_params" not in plain_log and "--texture-null" not in plain_log
    assert not (plain / "texture_null_per_well.csv").exists()
    assert (plain / "qki_association_columns.md").read_text(encoding="utf-8").count("SENSITIVITY ONLY") == 0


def test_cli_flags(tmp_path):
    from click.testing import CliRunner
    from fishsuite.cli import cli
    from test_qki_association import _assoc_cached_run
    root = _assoc_cached_run(tmp_path / "source")
    result = CliRunner().invoke(cli, ["qki-assoc", "--run-dir", str(root), "--miat-min", "10", "--qki-min", "10",
        "--objective-na", "1.5", "--emission-nm-miat", "668", "--emission-nm-qki", "603", "--n-null", "10",
        "--texture-null", "--texture-dapi-bins", "3", "--texture-radial-bins", "2",
        "--texture-min-positions", "1", "--rotation-upp", "--out", str(tmp_path / "cli")])
    assert result.exit_code == 0, result.output
    log = (tmp_path / "cli" / "command.log").read_text(encoding="utf-8")
    assert '"n_dapi_bins": 3' in log and '"n_radial_bins": 2' in log and '"min_positions": 1' in log
