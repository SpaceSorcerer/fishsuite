"""Texture-matched placement null for the uniform-position percentile score (UPP).

SENSITIVITY ONLY (Astra fig1 review F1; texture-null review round 2).

Calibration (round-2 F7/F8): realistic synthetic nuclei from
``_texture_null_sim`` (irregular masks with nucleolar holes, variable exact
footprints, smooth nonlinear DAPI and radial fields whose boundaries do not
align with any stratum). The calibration unit is the NUCLEUS mean score;
nuclei are independent. For each of N_FIELDS independent fields of N_NUCLEI
nuclei, a two-sided one-sample t test of the nucleus means against 0.5 at
alpha 0.05 is run, and the empirical rejection rate across fields is gated.
Under a calibrated null the rejection count is Binomial(N_FIELDS, 0.05);
the gate is its one-sided 99.9 % upper limit. No spot-level SE is used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import _texture_null_sim as sim
from fishsuite.core.footprint_null import MiatFootprint, exact_footprint_position_null

K = 200
N_FIELDS, N_NUCLEI = 40, 12
ALPHA = 0.05
MAX_FALSE_REJECTIONS = int(stats.binom.ppf(0.999, N_FIELDS, ALPHA))  # 7 of 40


def _footprint(index, y, x, radius=1):
    dy, dx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    dy, dx = dy.ravel(), dx.ravel()
    return MiatFootprint(index, int(y), int(x), y + dy, x + dx, dy, dx,
                         "synthetic", None, True, None, 100.0, 0.0)


def _tables(data, **kwargs):
    from fishsuite.core.qki_association import association_tables
    from fishsuite.core.texture_null import TextureNullParams
    miat, qki, labels, footprints, dapi, eligible = data
    options = dict(pixel_size_um=0.13, miat_min=50, qki_min=1100, sensitivity=(1.0,),
                   n_null=K, seed=0, n_costes=5, dapi=dapi, eligible_mask=eligible,
                   texture_null=TextureNullParams(), rotation_upp=True)
    options.update(kwargs)
    return association_tables(miat, qki, labels, footprints, **options)


def _tiled(scenario, n, seed):
    """n realistic nuclei tiled side by side into one labelled image."""
    rng = np.random.default_rng(seed)
    size = sim.SIZE
    labels = np.zeros((size, size * n), np.int32)
    eligible = np.zeros_like(labels, bool)
    qki = np.full(labels.shape, 1000.0)
    dapi = np.full(labels.shape, 1000.0)
    miat = np.zeros(labels.shape)
    fps = []
    for i in range(n):
        nuc = sim.nucleus(rng, scenario)
        sl = np.s_[:, i * size:(i + 1) * size]
        labels[sl][nuc["mask"]] = i + 1
        eligible[sl] = nuc["eligible"]
        qki[sl], dapi[sl] = nuc["qki"], nuc["dapi"]
        for fp in nuc["footprints"]:
            shifted = MiatFootprint(len(fps), fp.center_y_px, fp.center_x_px + i * size, fp.y_px,
                                    fp.x_px + i * size, fp.dy_px, fp.dx_px, "synthetic", None, True, None,
                                    100.0, 0.0)
            miat[shifted.y_px, shifted.x_px] = 100.0
            fps.append(shifted)
    return miat, qki, labels, fps, dapi, eligible


@pytest.fixture(scope="module")
def calibration():
    frames = [sim.nucleus_summaries(s, N_FIELDS, N_NUCLEI, 100 + i)
              for i, s in enumerate(("confound", "independent", "coloc"))]
    return sim.field_rejections(pd.concat(frames, ignore_index=True)).set_index(["scenario", "method"])


# --------------------------------------------------------------- acceptance (3)
@pytest.mark.parametrize("method", ["uniform", "strata_5x5", "strata_8x8", "strata_10x10", "knn_k200"])
def test_independent_channels_are_calibrated_at_the_nucleus_level(calibration, method):
    row = calibration.loc[("independent", method)]
    assert row.n_fields == N_FIELDS
    assert row.rejection_rate * N_FIELDS <= MAX_FALSE_REJECTIONS, row.to_dict()


# --------------------------------------------------------------- acceptance (1)
def test_shared_texture_confound_inflates_uniform_and_is_largely_removed_by_matching(calibration):
    uniform = calibration.loc[("confound", "uniform")]
    assert uniform.rejection_rate >= 0.9 and uniform.mean_field_bias > 0.05
    for method in ("strata_5x5", "strata_8x8", "strata_10x10", "knn_k200"):
        row = calibration.loc[("confound", method)]
        # Residual confounding within cells is expected; the gate is that matching
        # removes at least 75 % of the uniform null's bias. The rejection rate
        # is reported, not gated (see implC round 2).
        assert abs(row.mean_field_bias) <= 0.25 * uniform.mean_field_bias, (method, row.to_dict())


# --------------------------------------------------------------- acceptance (2)
@pytest.mark.parametrize("method", ["uniform", "strata_5x5", "strata_8x8", "strata_10x10", "knn_k200"])
def test_true_punctum_colocalization_is_detected_by_every_null(calibration, method):
    row = calibration.loc[("coloc", method)]
    assert row.rejection_rate >= 0.9 and row.mean_field_bias > 0.1, row.to_dict()


# --------------------------------------------------------------- acceptance (4)
def test_stratum_kept_when_it_has_enough_positions():
    from fishsuite.core.texture_null import select_stratum
    dapi_bin = np.zeros(100, int)
    radial_bin = np.repeat(np.arange(5), 20)
    members, reason = select_stratum(dapi_bin, radial_bin, 45, min_positions=20)
    assert reason == "" and members.sum() == 20 and members[40:60].all()


def test_sparse_stratum_is_na_and_never_merged():
    from fishsuite.core.texture_null import select_stratum
    radial_bin = np.array([0] * 30 + [1] * 5 + [2] * 18)
    dapi_bin = np.zeros(radial_bin.size, int)
    members, reason = select_stratum(dapi_bin, radial_bin, 32, min_positions=20)
    assert members is None and reason == "SPARSE_STRATUM"


def test_one_radial_bin_sparse_stratum_is_na_not_a_crash():
    from fishsuite.core.texture_null import select_stratum
    members, reason = select_stratum(np.zeros(7, int), np.zeros(7, int), 3, min_positions=20)
    assert members is None and reason == "SPARSE_STRATUM"


def test_knn_takes_k_nearest_in_quantile_space_and_is_na_below_k():
    from fishsuite.core.texture_null import select_knn
    dq = np.array([0.1, 0.12, 0.5, 0.9, 0.11])
    rq = np.array([0.5, 0.5, 0.5, 0.5, 0.52])
    members, reason, dmax = select_knn(dq, rq, 0, k=3)
    assert reason == "" and members.tolist() == [True, True, False, False, True]
    assert dmax == pytest.approx(np.hypot(0.01, 0.02))  # farthest member: index 4
    members, reason, _ = select_knn(dq, rq, 0, k=6)
    assert members is None and reason == "KNN_SPARSE"


def test_tiny_nucleus_marks_spots_na_and_records_counts():
    from fishsuite.core.texture_null import TextureNullParams
    labels = np.zeros((20, 20), np.int32)
    labels[5:14, 5:14] = 1  # 9x9: 49 admissible centres for a 3x3 footprint
    rng = np.random.default_rng(0)
    qki, dapi = rng.normal(1000, 50, (20, 20)), rng.normal(1000, 50, (20, 20))
    fps = [_footprint(0, 8, 8), _footprint(1, 10, 11)]
    data = (np.zeros((20, 20)), qki, labels, fps, dapi, None)
    nuclei, spots = _tables(data)
    row = nuclei.iloc[0]
    assert np.isnan(row.upp_texture_matched_mean_qki)
    assert row.na_reason_upp_texture_matched == "ALL_SPOTS_NA"
    assert row.upp_texture_matched_n_spots_na == 2 and row.upp_texture_matched_n_spots_scored == 0
    assert row.upp_texture_matched_na_reason_counts == "SPARSE_STRATUM=2"
    assert spots.na_reason_upp_texture_matched_spot.eq("SPARSE_STRATUM").all()
    nuclei, spots = _tables(data, texture_null=TextureNullParams(n_dapi_bins=1, n_radial_bins=1))
    assert nuclei.iloc[0].upp_texture_matched_n_spots_scored == 2
    assert spots.texture_stratum_n_positions.eq(49).all()
    assert nuclei.iloc[0].upp_texture_matched_calibration == "EXACT_FINITE_K_FIXED_STRATA"
    nuclei, spots = _tables(data, texture_null=TextureNullParams(method="knn", knn_k=200))
    assert spots.na_reason_upp_texture_matched_spot.eq("KNN_SPARSE").all()
    nuclei, spots = _tables(data, texture_null=TextureNullParams(method="knn", knn_k=30))
    assert spots.texture_stratum_n_positions.eq(30).all()
    assert nuclei.iloc[0].upp_texture_matched_calibration == "NOMINAL_KNN_OBSERVATION_CENTRED"
    assert nuclei.iloc[0].upp_texture_matched_method == "knn_k30"


def test_no_spots_gives_n0():
    labels = np.zeros((20, 20), np.int32)
    labels[2:18, 2:18] = 1
    flat = np.full((20, 20), 1000.0)
    nuclei, spots = _tables((np.zeros((20, 20)), flat, labels, [], flat, None))
    assert nuclei.iloc[0].na_reason_upp_texture_matched == "N0"
    assert nuclei.iloc[0].na_reason_upp_rotation == "N0"
    assert spots.empty


# ------------------------------------------------------------ construction (F4/F5)
def _brute_force_admissible(fp, region, qki):
    out = []
    for y, x in np.argwhere(region):
        ys, xs = y + fp.dy_px, x + fp.dx_px
        inside = (ys >= 0) & (ys < region.shape[0]) & (xs >= 0) & (xs < region.shape[1])
        if inside.all() and region[ys, xs].all() and np.isfinite(qki[ys, xs]).all():
            out.append((y, x))
    return np.asarray(out)


def test_admissible_centre_coordinates_equal_brute_force_and_placement_null_samples():
    from fishsuite.core.texture_null import admissible_centers
    rng = np.random.default_rng(3)
    nuc = sim.nucleus(rng, "independent")
    region, qki = nuc["eligible"], nuc["qki"]
    for fp in nuc["footprints"][:3]:
        centers = admissible_centers(fp, region, qki)
        assert np.array_equal(centers, _brute_force_admissible(fp, region, qki))
        reference = exact_footprint_position_null(qki, fp, region, n_null=5, rng=rng)
        assert centers.shape[0] == reference.valid_center_count


def test_texture_draws_are_exactly_the_observed_fixed_cell():
    from fishsuite.core.texture_null import (TextureNullParams, admissible_centers, footprint_means_at,
                                             normalized_radial_map, quantile_bins, texture_matched_spot_scores)
    rng = np.random.default_rng(4)
    nuc = sim.nucleus(rng, "independent")
    region, qki, dapi, mask = nuc["eligible"], nuc["qki"], nuc["dapi"], nuc["mask"]
    fp = nuc["footprints"][0]
    params = TextureNullParams(n_dapi_bins=3, n_radial_bins=3)
    obs = np.asarray([qki[fp.y_px, fp.x_px].mean()])
    row = texture_matched_spot_scores(qki, dapi, mask, region, [fp], obs, n_null=20000, params=params,
                                      rng=rng, tie_rng=rng, return_draw_centers=True)[0]
    centers = admissible_centers(fp, region, qki)
    _, dbin = quantile_bins(footprint_means_at(dapi, centers, fp), 3)
    _, rbin = quantile_bins(normalized_radial_map(mask)[centers[:, 0], centers[:, 1]], 3)
    index = np.flatnonzero((centers[:, 0] == fp.center_y_px) & (centers[:, 1] == fp.center_x_px))[0]
    expected = centers[(dbin == dbin[index]) & (rbin == rbin[index])]
    assert np.array_equal(row["_matched_centers"], expected)
    drawn = {tuple(c) for c in row["_draw_centers"]}
    assert drawn == {tuple(c) for c in expected}  # 20000 draws cover every cell position
    assert tuple(fp.center_yx) in drawn


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
@pytest.fixture(scope="module")
def tiled():
    return _tiled("confound", 3, seed=5)


def test_existing_columns_are_value_identical_and_absent_by_default(tiled):
    from fishsuite.core.qki_association import association_tables, NUCLEUS_COLUMNS, SPOT_COLUMNS
    on_nuc, on_spot = _tables(tiled)
    miat, qki, labels, fps, _, eligible = tiled
    off_nuc, off_spot = association_tables(miat, qki, labels, fps, pixel_size_um=0.13, miat_min=50,
                                           qki_min=1100, sensitivity=(1.0,), n_null=K, seed=0, n_costes=5,
                                           eligible_mask=eligible)
    assert list(off_nuc.columns) == NUCLEUS_COLUMNS and list(off_spot.columns) == SPOT_COLUMNS
    pd.testing.assert_frame_equal(on_nuc[NUCLEUS_COLUMNS], off_nuc)
    pd.testing.assert_frame_equal(on_spot[SPOT_COLUMNS], off_spot)
    assert on_nuc.columns[len(NUCLEUS_COLUMNS):].str.contains("texture|rotation").all()
    assert on_spot.upp_texture_matched_qki.notna().mean() > 0.9


def test_texture_null_is_seeded_and_needs_dapi(tiled):
    from fishsuite.core.texture_null import TextureNullParams
    a, b = _tables(tiled)[1], _tables(tiled)[1]
    pd.testing.assert_frame_equal(a, b)
    c = _tables(tiled, seed=1)[1]
    assert not np.allclose(a.upp_texture_matched_qki, c.upp_texture_matched_qki, equal_nan=True)
    with pytest.raises(ValueError, match="dapi"):
        _tables(tiled, dapi=None)
    with pytest.raises(ValueError):
        TextureNullParams(n_dapi_bins=0)
    with pytest.raises(ValueError):
        TextureNullParams(method="kernel")


def test_column_definitions_label_sensitivity_only_and_limit_exact_wording():
    from fishsuite.core.qki_association import COLUMN_DEFINITIONS, TEXTURE_NUCLEUS_COLUMNS, \
        TEXTURE_SPOT_COLUMNS
    from fishsuite.core.texture_null import is_sensitivity_column
    for name in TEXTURE_NUCLEUS_COLUMNS + TEXTURE_SPOT_COLUMNS:
        assert "SENSITIVITY ONLY" in COLUMN_DEFINITIONS[name], name
        assert is_sensitivity_column(name), name
        text = COLUMN_DEFINITIONS[name].lower()
        assert "merge" not in text.replace("unmerged", "").replace("no merg", "").replace("never merged", ""), name
    assert "exact only for method strata" in COLUMN_DEFINITIONS["upp_texture_matched_chance_frac_ge_0p90"]


# ------------------------------------------------------------ coupling exclusion (F2)
def test_no_coupling_output_contains_sensitivity_columns(tmp_path, tiled, monkeypatch):
    import matplotlib.pyplot as plt
    from pathlib import Path
    from fishsuite.report import coupling
    from fishsuite.core.texture_null import is_sensitivity_column
    frames = []
    for arm, wells in (("control", ("C1", "C2")), ("treated", ("T1", "T2"))):
        for well in wells:
            nuclei, _ = _tables(tiled, condition=arm, well=well, image=f"{well}.tif")
            frames.append(nuclei)
    table = pd.concat(frames, ignore_index=True)
    assert any(is_sensitivity_column(c) for c in table.columns)
    source = tmp_path / "assoc.csv"
    table.to_csv(source, index=False)

    def capture_save(fig, destination, stem, *args):
        Path(destination).mkdir(parents=True, exist_ok=True)
        plt.close(fig)
    monkeypatch.setattr(coupling.figlib, "save", capture_save)
    out = coupling.build_coupling(source, "treated", "control", tmp_path / "report", n_boot=10)
    # Column-name tokens (pre-existing Costes README prose says "texture-matched synthetic").
    TOKENS = ("upp_texture_matched", "texture_dapi", "texture_radial", "texture_stratum",
              "texture_null_mean", "texture_match_max", "upp_rotation")
    found = []
    for path in out.rglob("*"):
        if path.suffix == ".xlsx":
            for name, sheet in pd.read_excel(path, sheet_name=None, header=None).items():
                text = sheet.astype(str).to_numpy().ravel().tolist()
                found += [(path.name, name, t) for t in text if any(token in t for token in TOKENS)]
        elif path.suffix in (".csv", ".md", ".txt", ".log"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for token in TOKENS:
                if token in text and not (path.name == "command.log"):
                    found.append((path.name, token))
    assert not found, found[:10]


# ------------------------------------------------------------ adapter / CLI
def test_adapter_writes_sensitivity_tables_and_logs_parameters(tmp_path):
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
            "well_frac_ge_0p90_upp_texture_matched", "texture_na_rate_spots", "balance_matched_dapi_q",
            "balance_uniform_dapi_q", "support_median_positions"} <= set(per_well.columns)
    assert len(per_arm) == 1
    log = (out / "command.log").read_text(encoding="utf-8")
    assert "texture_null_params" in log and "SENSITIVITY ONLY" in log and "--texture-null" in log
    assert "texture_null_merge: none" in log
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
    result = CliRunner().invoke(cli, ["qki-assoc", "--run-dir", str(root), "--miat-min", "10", "--qki-min", "10",
        "--objective-na", "1.5", "--emission-nm-miat", "668", "--emission-nm-qki", "603", "--n-null", "10",
        "--texture-null", "--texture-method", "knn", "--texture-knn-k", "5", "--out", str(tmp_path / "knn")])
    assert result.exit_code == 0, result.output
    assert '"method": "knn"' in (tmp_path / "knn" / "command.log").read_text(encoding="utf-8")


def test_cli_one_radial_bin_sparse_stratum_is_documented_na(tmp_path):
    from click.testing import CliRunner
    from fishsuite.cli import cli
    from test_qki_association import _assoc_cached_run
    root = _assoc_cached_run(tmp_path / "source")
    result = CliRunner().invoke(cli, ["qki-assoc", "--run-dir", str(root), "--miat-min", "10", "--qki-min", "10",
        "--objective-na", "1.5", "--emission-nm-miat", "668", "--emission-nm-qki", "603", "--n-null", "10",
        "--texture-null", "--texture-dapi-bins", "1", "--texture-radial-bins", "1",
        "--texture-min-positions", "1000", "--out", str(tmp_path / "sparse")])
    assert result.exit_code == 0, result.output
    spots = pd.read_csv(tmp_path / "sparse" / "qki_association_per_spot.csv")
    assert spots.na_reason_upp_texture_matched_spot.eq("SPARSE_STRATUM").all()
    assert spots.upp_texture_matched_qki.isna().all()
    columns_md = (tmp_path / "sparse" / "qki_association_columns.md").read_text(encoding="utf-8")
    assert "SPARSE_STRATUM" in columns_md
