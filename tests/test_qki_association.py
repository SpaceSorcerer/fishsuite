from __future__ import annotations

import importlib.util
import inspect

import numpy as np
import pandas as pd
import pytest

from fishsuite.core.footprint_null import MiatFootprint


def _api():
    assert importlib.util.find_spec("fishsuite.core.qki_association") is not None
    from fishsuite.core.qki_association import association_tables
    return association_tables


def _footprint(index, y, x, radius=1):
    dy, dx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    dy, dx = dy.ravel(), dx.ravel()
    return MiatFootprint(index, y, x, y + dy, x + dx, dy, dx,
                         "synthetic", None, True, None, 100.0, 0.0)


def _four():
    labels = np.ones((64, 64), dtype=np.int32)
    miat = np.zeros((64, 64), dtype=np.uint16)
    qki = np.ones((64, 64), dtype=np.uint16)
    footprints = [_footprint(i, y, x) for i, (y, x) in
                  enumerate(((10, 10), (20, 20), (30, 30), (40, 40)))]
    for fp in footprints:
        miat[fp.y_px, fp.x_px] = 100
    qki[9:12, 9:12] = 20
    qki[49:52, 49:52] = 20
    return miat, qki, labels, footprints


def _run(data, **kwargs):
    options = dict(pixel_size_um=0.5, miat_min=50, qki_min=10,
                   sensitivity=(1.0,), n_null=200, seed=7)
    options.update(kwargs)
    return _api()(*data, **options)


def test_four_spots_literal_counts_areas_and_context():
    nuclei, spots = _run(_four())
    row = nuclei.iloc[0]
    assert row.n_miat_spots == 4
    assert row.n_miat_spots_qki_pos == 1
    assert row.frac_miat_spots_qki_pos == 0.25
    assert row.frac_qki_area_on_miat_area == 9 / 18
    assert row.frac_qki_area_on_miat_footprints == 9 / 18
    assert row.qki_pos_area_frac == 18 / 4096
    assert row.miat_pos_area_frac == 36 / 4096
    assert row.miat_footprint_area_frac == 36 / 4096
    assert row.nuclear_area_um2 == 1024.0
    assert row.integrated_nuclear_miat == 3600.0
    assert row.mean_qki_at_miat_spots == 5.75
    assert row.median_footprint_area_px == 9
    assert row.n_miat_spots_per_100um2 == 0.390625
    assert spots.qki_positive.tolist() == [True, False, False, False]
    assert "exp_frac_spots_qki_pos_analytic" not in nuclei


def test_zero_spots_zero_qki_and_empty_eligible_nucleus_remain():
    miat, qki, labels, _ = _four()
    qki[:] = 0
    labels[32:] = 2
    eligible = labels == 1
    nuclei, spots = _run((miat, qki, labels, []), eligible_mask=eligible)
    assert nuclei.nucleus_id.tolist() == [1, 2]
    assert nuclei.n_miat_spots.tolist() == [0, 0]
    assert nuclei.n_miat_spots_qki_pos.tolist() == [0, 0]
    assert nuclei.frac_miat_spots_qki_pos.isna().all()
    assert nuclei.na_reason_frac_miat_spots_qki_pos.tolist() == ["N0", "N0"]
    assert nuclei.na_reason_frac_qki_area_on_miat_area.tolist() == ["Q0", "Q0"]
    assert nuclei.na_reason_frac_qki_area_on_miat_footprints.tolist() == ["Q0", "Q0"]
    assert spots.empty


def test_uniform_qki_has_unit_observed_and_null_spot_fraction():
    data = _four()
    data[1][:] = 20
    nuclei, _ = _run(data)
    row = nuclei.iloc[0]
    assert row.frac_miat_spots_qki_pos == 1.0
    assert row.null_mean_frac_miat_spots_qki_pos == 1.0
    assert row.obs_minus_null_frac_miat_spots_qki_pos == 0.0


def _independent(radius=1, count=15):
    rng = np.random.default_rng(193)
    labels = np.repeat(np.arange(1, 21), 32)[:, None] * np.ones((1, 64), dtype=int)
    miat = np.zeros(labels.shape)
    qki = rng.uniform(0, 20, labels.shape)
    footprints = []
    for nucleus in range(20):
        centers = rng.choice(30 * 62, count, replace=False)
        for center in centers:
            y, x = divmod(int(center), 62)
            fp = _footprint(len(footprints), nucleus * 32 + y + 1, x + 1, radius)
            footprints.append(fp)
            miat[fp.y_px, fp.x_px] = 100
    return miat, qki, labels, footprints


def test_independent_qki_three_hundred_spots_chance_corrected():
    nuclei, _ = _run(_independent())
    assert nuclei.n_miat_spots.sum() == 300
    assert abs(nuclei.obs_minus_null_frac_miat_spots_qki_pos.mean()) < 0.05


def test_signal_only_under_footprints_is_enriched():
    data = _four()
    data[1][:] = 0
    for fp in data[3]:
        data[1][fp.y_px, fp.x_px] = 20
    nuclei, _ = _run(data)
    assert nuclei.iloc[0].obs_minus_null_frac_miat_spots_qki_pos > 0.5
    assert nuclei.iloc[0].obs_minus_null_frac_miat_footprint_area_qki_pos > 0.5


def test_doubled_density_raw_area_rises_correction_stays_near_zero():
    dense = _independent(radius=0, count=30)
    sparse = (dense[0], dense[1], dense[2],
              [fp for fp in dense[3] if fp.spot_index % 30 < 15])
    a, _ = _run(sparse)
    b, _ = _run(dense)
    assert b.n_miat_spots.sum() == 600
    assert a.n_miat_spots.sum() == 300
    assert b.frac_qki_area_on_miat_footprints.mean() > a.frac_qki_area_on_miat_footprints.mean()
    for frame in (a, b):
        assert abs(frame.obs_minus_null_frac_miat_spots_qki_pos.mean()) < 0.05
        assert abs(frame.obs_minus_null_frac_miat_footprint_area_qki_pos.mean()) < 0.05


def test_linear_gradient_uniform_spots_has_near_zero_correction():
    miat, _, labels, _ = _four()
    qki = np.tile(np.arange(64, dtype=float), (64, 1))
    footprints = [_footprint(i, y, x) for i, (y, x) in
                  enumerate((y, x) for y in range(4, 61, 8) for x in range(4, 61, 8))]
    nuclei, _ = _run((miat, qki, labels, footprints), qki_min=32)
    assert abs(nuclei.iloc[0].obs_minus_null_frac_miat_spots_qki_pos) < 0.05


def test_sensitivity_level_one_matches_single_run_and_seed_bytes():
    data = _independent(count=3)
    multi, multi_spots = _run(data, sensitivity=(0.8, 1.0, 1.25))
    single, single_spots = _run(data)
    assert multi.threshold_multiplier.unique().tolist() == [0.8, 1.0, 1.25]
    pd.testing.assert_frame_equal(multi[multi.threshold_multiplier == 1.0].reset_index(drop=True), single)
    pd.testing.assert_frame_equal(multi_spots[multi_spots.threshold_multiplier == 1.0].reset_index(drop=True), single_spots)
    again, again_spots = _run(data)
    assert single.to_csv(index=False).encode() == again.to_csv(index=False).encode()
    assert single_spots.to_csv(index=False).encode() == again_spots.to_csv(index=False).encode()
    changed, changed_spots = _run(data, seed=99)
    null_columns = [c for c in single if c.startswith(("null_", "obs_minus_null_", "na_reason_null_", "na_reason_obs_minus_null_"))]
    assert not changed[null_columns].equals(single[null_columns])
    pd.testing.assert_frame_equal(changed.drop(columns=null_columns), single.drop(columns=null_columns))
    pd.testing.assert_frame_equal(changed_spots, single_spots)


@pytest.mark.parametrize("name", ["miat_min", "qki_min"])
@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf, -np.inf])
def test_invalid_thresholds_rejected(name, value):
    with pytest.raises(ValueError, match="finite.*positive"):
        _run(_four(), **{name: value})


def test_display_settings_absent_from_module_source():
    _api()
    from fishsuite.core import qki_association
    source = inspect.getsource(qki_association)
    assert "manual_rna_max" not in source
    assert "manual_antibody_max" not in source


def test_overlap_union_deduplicated_footprint_mean_and_eligibility():
    miat, qki, labels, footprints = _four()
    footprints = [footprints[0], _footprint(4, 10, 11), _footprint(5, 63, 63)]
    qki[:] = 0
    qki[10, 10] = 20
    nuclei, spots = _run((miat, qki, labels, footprints))
    row = nuclei.iloc[0]
    assert row.n_miat_spots == 2
    assert row.n_miat_spots_qki_pos == 0
    assert row.miat_footprint_area_frac == 12 / 4096
    assert row.frac_qki_area_on_miat_footprints == 1.0
    assert len(spots) == 2


def test_saturation_dtype_and_flat_top():
    data = _four()
    data[0][:] = 0
    data[0][0, 0] = 65535
    data[1][:] = 1
    data[1][0, :5] = 100
    nuclei, _ = _run(data)
    assert nuclei.iloc[0].sat_frac_miat == 1 / 4096
    assert nuclei.iloc[0].sat_frac_qki == 5 / 4096


def test_analytic_area_correction_is_explicit():
    nuclei, _ = _run(_four())
    assert nuclei.iloc[0].obs_minus_exp_frac_qki_area_on_miat_area == 0.4912109375


def test_no_domain_marks_null_only_for_invalid_offsets():
    from dataclasses import replace
    miat, qki, labels, footprints = _four()
    malformed = replace(footprints[0], dy_px=footprints[0].dy_px + 1)
    nuclei, _ = _run((miat, qki, labels, [malformed]))
    row = nuclei.iloc[0]
    assert row.frac_miat_spots_qki_pos == 1.0
    assert np.isnan(row.null_mean_frac_miat_spots_qki_pos)
    assert row.na_reason_null_mean_frac_miat_spots_qki_pos == "NO_DOMAIN"


@pytest.mark.parametrize("option,value", [("n_null", 0), ("n_null", 1.5),
    ("sensitivity", ()), ("sensitivity", (0,)), ("sensitivity", (1, 1)),
    ("sensitivity", (np.inf,)), ("pixel_size_um", 0), ("pixel_size_um", np.nan)])
def test_invalid_sampling_and_scale_parameters(option, value):
    with pytest.raises(ValueError):
        _run(_four(), **{option: value})


def test_excluded_region_shared_by_observed_and_null():
    miat, qki, labels, footprints = _four()
    qki[:] = 20
    eligible = np.ones(labels.shape, dtype=bool)
    eligible[:16] = False
    qki[:16] = 0
    nuclei, _ = _run((miat, qki, labels, footprints), eligible_mask=eligible)
    row = nuclei.iloc[0]
    assert row.n_miat_spots == 3
    assert row.nuclear_area_um2 == 768
    assert row.frac_miat_spots_qki_pos == 1.0
    assert row.null_mean_frac_miat_spots_qki_pos == 1.0


def test_nonfinite_eligible_raw_signal_rejected():
    miat, qki, labels, footprints = _four()
    qki = qki.astype(float)
    qki[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        _run((miat, qki, labels, footprints))


def test_flagged_invalid_footprint_excluded():
    from dataclasses import replace
    miat, qki, labels, footprints = _four()
    footprints[0] = replace(footprints[0], invalid_reason="invalid_saved_footprint")
    nuclei, _ = _run((miat, qki, labels, footprints))
    assert nuclei.iloc[0].n_miat_spots == 3


def test_no_admissible_center_marks_placement_null_no_domain():
    miat, qki, labels, _ = _four()
    eligible = np.zeros(labels.shape, dtype=bool)
    eligible[12, 12] = True
    qki[12, 12] = 20
    footprint = MiatFootprint(0, 10, 10, np.array([12]), np.array([12]),
                             np.array([2]), np.array([2]), "synthetic", None,
                             True, None, 100.0, 20.0)
    nuclei, _ = _run((miat, qki, labels, [footprint]), eligible_mask=eligible)
    row = nuclei.iloc[0]
    assert row.n_miat_spots == 1
    assert row.frac_miat_spots_qki_pos == 1.0
    assert row.frac_qki_area_on_miat_footprints == 1.0
    assert row.na_reason_null_mean_frac_miat_spots_qki_pos == "NO_DOMAIN"
    assert row.na_reason_null_mean_frac_miat_footprint_area_qki_pos == "NO_DOMAIN"
    assert np.isnan(row.null_mean_frac_miat_spots_qki_pos)
import numpy as np
import pandas as pd
import pytest


def _assoc_cached_run(root):
    import h5py
    root.mkdir()
    raw = np.full((16, 16), 20, dtype=np.uint16)
    labels = np.ones((16, 16), dtype=np.int32)
    labels[:, 8:] = 2
    with h5py.File(root / 'selected_planes_and_masks.h5', 'w') as h:
        g = h.create_group('images/a')
        for name, value in dict(image_key='synthetic', complete=True,
                same_plane_all_channels=True, selected_z_1based=1,
                selected_z_0based=0, miat_channel_index=0,
                qki_channel_index=1, dapi_channel_index=2).items():
            g.attrs[name] = value
        for name in ('miat', 'qki', 'dapi'):
            g.create_dataset(name, data=raw)
        g.create_dataset('nucleus_labels', data=labels)
        g.create_dataset('nucleolus_labels', data=np.zeros_like(labels))
    provenance = dict(image_key='synthetic', selected_z_1based=1,
        selected_z_0based=0, miat_channel_index=0, qki_channel_index=1,
        dapi_channel_index=2, quantitation_plane='exact_recorded_single_z', voxel_xy_nm=100)
    pd.DataFrame([{**provenance, 'condition': 'test', 'well': 'A1'}]).to_csv(root / 'image_manifest.csv', index=False)
    pd.DataFrame([{**provenance, 'nucleus_id': n, 'z_mode': 'single_plane',
        'z_range': '1-1', 'n_z_slices': 1} for n in (1, 2)]).to_csv(root / 'nucleus_exact_footprint_metrics.csv', index=False)
    pd.DataFrame([{**provenance, 'nucleus_id': 1, 'spot_id': 'source-spot',
        'spot_uid': 'synthetic::source-spot', 'center_y_px': 3, 'center_x_px': 3,
        'footprint_method': 'half_max', 'footprint_fallback_reason': '',
        'footprint_area_px': 1, 'null_candidate': True}]).to_csv(root / 'spot_exact_footprint_metrics.csv.gz', index=False)
    pd.DataFrame([dict(spot_uid='synthetic::source-spot', pixel_index=0,
        flat_pixel_index=51, y_px=3, x_px=3, dy_px=0, dx_px=0,
        miat_raw=20, qki_raw=20)]).to_csv(root / 'footprint_pixels.csv.gz', index=False)
    return root


def test_assoc_adapter_roundtrip(tmp_path):
    from fishsuite.core.qki_association_postrun import run_qki_association
    root = _assoc_cached_run(tmp_path / 'source')
    first = run_qki_association(root, tmp_path / 'out1', miat_min=10, qki_min=10, n_null=10)
    second = run_qki_association(root, tmp_path / 'out2', miat_min=10, qki_min=10, n_null=10)
    from click.testing import CliRunner
    from fishsuite.cli import cli
    cli_out = tmp_path / 'cli_out'
    result = CliRunner().invoke(cli, ['qki-assoc', '--run-dir', str(root),
        '--miat-min', '10', '--qki-min', '10', '--sensitivity', '0.8,1.0,1.25',
        '--n-null', '10', '--seed', '0', '--out', str(cli_out)])
    assert result.exit_code == 0, result.output
    for name in ('qki_association_per_nucleus.csv', 'qki_association_per_spot.csv'):
        assert (first / name).read_bytes() == (cli_out / name).read_bytes()
    nuclei = pd.read_csv(first / 'qki_association_per_nucleus.csv')
    spots = pd.read_csv(first / 'qki_association_per_spot.csv')
    assert len(nuclei) == 6
    assert len(spots) == 3
    assert set(spots.spot_id) == {'source-spot'}
    assert set(nuclei.loc[nuclei.nucleus_id == 2, 'n_miat_spots']) == {0}
    assert set(nuclei.loc[nuclei.nucleus_id == 2, 'na_reason_frac_miat_spots_qki_pos']) == {'N0'}
    assert (first / 'qki_association_per_nucleus.csv').read_bytes() == (second / 'qki_association_per_nucleus.csv').read_bytes()
    assert (first / 'qki_association_per_spot.csv').read_bytes() == (second / 'qki_association_per_spot.csv').read_bytes()
    for name in ('qki_association_columns.md', 'command.log', 'versions.txt'):
        assert (first / name).stat().st_size > 0


def test_assoc_adapter_missing_and_protection(tmp_path):
    from fishsuite.core.qki_association_postrun import run_qki_association
    with pytest.raises(ValueError, match='missing.*spot_exact_footprint'):
        run_qki_association(tmp_path, tmp_path.parent / 'never_created_assoc', miat_min=10, qki_min=10)
    root = _assoc_cached_run(tmp_path / 'source')
    with pytest.raises(ValueError, match='outside'):
        run_qki_association(root, root / 'out', miat_min=10, qki_min=10)
    manifest = pd.read_csv(root / 'image_manifest.csv').drop(columns='voxel_xy_nm')
    manifest.to_csv(root / 'image_manifest.csv', index=False)
    with pytest.raises(ValueError, match='voxel_xy_nm'):
        run_qki_association(root, tmp_path / 'bad', miat_min=10, qki_min=10)


def test_assoc_cli_help():
    from click.testing import CliRunner
    from fishsuite.cli import cli
    result = CliRunner().invoke(cli, ['qki-assoc', '--help'])
    assert result.exit_code == 0, result.output
    for flag in ('--run-dir', '--miat-min', '--qki-min', '--sensitivity', '--n-null', '--seed', '--out'):
        assert flag in result.output



def test_assoc_adapter_empty_spots_and_missing_scale(tmp_path):
    from fishsuite.core.qki_association_postrun import run_qki_association
    root = _assoc_cached_run(tmp_path / 'source')
    for name in ('spot_exact_footprint_metrics.csv.gz', 'footprint_pixels.csv.gz'):
        pd.read_csv(root / name).iloc[:0].to_csv(root / name, index=False)
    out = run_qki_association(root, tmp_path / 'empty', miat_min=10, qki_min=10, n_null=5)
    table = pd.read_csv(out / 'qki_association_per_nucleus.csv')
    assert len(table) == 6
    assert table.n_miat_spots.eq(0).all()
    assert table.na_reason_frac_miat_spots_qki_pos.eq('N0').all()
    assert pd.read_csv(out / 'qki_association_per_spot.csv').empty
    manifest = pd.read_csv(root / 'image_manifest.csv')
    manifest['voxel_xy_nm'] = np.nan
    manifest.to_csv(root / 'image_manifest.csv', index=False)
    with pytest.raises(ValueError, match='voxel_xy_nm'):
        run_qki_association(root, tmp_path / 'nan_scale', miat_min=10, qki_min=10)


def test_assoc_adapter_parent_specific_exclusion(tmp_path):
    import h5py
    from fishsuite.core.qki_association_postrun import run_qki_association
    root = _assoc_cached_run(tmp_path / 'source')
    with h5py.File(root / 'selected_planes_and_masks.h5', 'r+') as h:
        h['images/a/nucleolus_labels'][3, 3] = 2
    out = run_qki_association(root, tmp_path / 'out', miat_min=10, qki_min=10, n_null=5)
    table = pd.read_csv(out / 'qki_association_per_nucleus.csv')
    assert table.loc[table.nucleus_id == 1, 'n_miat_spots'].eq(1).all()
    assert table.loc[table.nucleus_id == 1, 'nuclear_area_um2'].to_numpy() == pytest.approx([1.28, 1.28, 1.28])


def _balanced_coverage(count):
    labels = np.ones((60, 60), dtype=np.int32)
    miat = np.zeros((60, 60), dtype=np.uint16)
    qki = np.zeros((60, 60), dtype=np.uint16)
    qki[:, :30] = 20
    pairs = [(y, x) for y in range(2, 58, 4) for x in range(2, 27, 4)]
    indices = np.random.default_rng(103).permutation(len(pairs))[:count // 2]
    footprints = []
    for index in indices:
        y, x = pairs[index]
        for col in (x, 59 - x):
            fp = _footprint(len(footprints), y, col)
            footprints.append(fp)
            miat[fp.y_px, fp.x_px] = 100
    return miat, qki, labels, footprints


@pytest.mark.parametrize("count,coverage", [(28, 0.07), (80, 0.20), (196, 0.49)])
def test_half_nucleus_qki_coverage_calibration(count, coverage):
    nuclei, _ = _run(_balanced_coverage(count))
    row = nuclei.iloc[0]
    assert row.miat_footprint_area_frac == coverage
    assert row.frac_miat_footprint_area_qki_pos == 0.5
    assert abs(row.obs_minus_null_frac_miat_footprint_area_qki_pos) < 0.02


def test_dense_uniform_qki_every_area_null_draw_is_one():
    data = _balanced_coverage(196)
    data[1][:] = 20
    nuclei, _ = _run(data)
    row = nuclei.iloc[0]
    assert row.frac_miat_footprint_area_qki_pos == 1.0
    assert row.null_mean_frac_miat_footprint_area_qki_pos == 1.0
    assert row.null_sd_frac_miat_footprint_area_qki_pos == 0.0
    assert row.obs_minus_null_frac_miat_footprint_area_qki_pos == 0.0
    assert row.null_ge_obs_frac_frac_miat_footprint_area_qki_pos == 1.0


def test_old_area_raw_and_analytic_retained_without_placement_correction():
    nuclei, _ = _run(_four())
    assert nuclei.iloc[0].frac_qki_area_on_miat_footprints == 0.5
    assert nuclei.iloc[0].exp_frac_qki_area_on_miat_footprints == 36 / 4096
    for prefix in ("null_mean", "null_sd", "obs_minus_null", "null_ge_obs_frac"):
        assert f"{prefix}_frac_qki_area_on_miat_footprints" not in nuclei
        assert f"na_reason_{prefix}_frac_qki_area_on_miat_footprints" not in nuclei


def test_empty_footprint_union_has_u0_and_zero_support():
    miat, qki, labels, _ = _four()
    nuclei, _ = _run((miat, qki, labels, []))
    row = nuclei.iloc[0]
    assert np.isnan(row.frac_miat_footprint_area_qki_pos)
    assert row.na_reason_frac_miat_footprint_area_qki_pos == "U0"
    assert row.na_reason_null_mean_frac_miat_footprint_area_qki_pos == "U0"
    assert row.null_valid_center_count == 0
    assert row.n_null_effective == 0


@pytest.mark.parametrize("support,reason", [(49, "SPARSE_DOMAIN"), (50, "")])
def test_admissible_center_support_gate_boundary(support, reason):
    miat, qki, labels, _ = _four()
    eligible = np.zeros(labels.shape, dtype=bool)
    eligible.ravel()[:support] = True
    fp = _footprint(0, 0, 0, radius=0)
    qki[:] = 20
    nuclei, _ = _run((miat, qki, labels, [fp]), eligible_mask=eligible)
    row = nuclei.iloc[0]
    assert row.null_valid_center_count == support
    assert row.n_null_effective == 200
    for metric in ("frac_miat_spots_qki_pos", "frac_miat_footprint_area_qki_pos"):
        assert row[metric] == 1.0
        for prefix in ("null_mean", "null_sd", "obs_minus_null", "null_ge_obs_frac"):
            assert row[f"na_reason_{prefix}_{metric}"] == reason
            assert np.isnan(row[f"{prefix}_{metric}"]) if reason else np.isfinite(row[f"{prefix}_{metric}"])


def test_tail_pseudocount_literal_and_descriptive_dictionary(monkeypatch):
    from dataclasses import replace
    from fishsuite.core import qki_association as module
    original = module.exact_footprint_position_null
    def controlled(*args, **kwargs):
        result = original(*args, **kwargs)
        return replace(result, null_qki_raw=np.array([20., 0., 20.]),
                       sampled_centers_yx=np.array([[10, 10], [30, 10], [10, 10]]))
    monkeypatch.setattr(module, "exact_footprint_position_null", controlled)
    miat, qki, labels, footprints = _four()
    nuclei, _ = _run((miat, qki, labels, footprints[:1]), n_null=3)
    row = nuclei.iloc[0]
    for metric in ("frac_miat_spots_qki_pos", "frac_miat_footprint_area_qki_pos"):
        key = f"null_ge_obs_frac_{metric}"
        assert row[key] == 0.75
        assert "per-nucleus descriptive; not a test across nuclei" in module.COLUMN_DEFINITIONS[key]


def test_observed_value_null_is_rejected_by_enrichment_test(monkeypatch):
    from dataclasses import replace
    from fishsuite.core import qki_association as module
    original = module.exact_footprint_position_null
    def observed_only(partner, footprint, *args, **kwargs):
        result = original(partner, footprint, *args, **kwargs)
        return replace(result, null_qki_raw=np.full(kwargs["n_null"], result.spot.observed_qki_raw),
                       sampled_centers_yx=np.tile(footprint.center_yx, (kwargs["n_null"], 1)))
    data = _four()
    data[1][:] = 0
    for fp in data[3]:
        data[1][fp.y_px, fp.x_px] = 20
    def assert_area_enriched():
        nuclei, _ = _run(data)
        assert nuclei.iloc[0].obs_minus_null_frac_miat_footprint_area_qki_pos > 0.5
    assert_area_enriched()
    monkeypatch.setattr(module, "exact_footprint_position_null", observed_only)
    with pytest.raises(AssertionError):
        assert_area_enriched()


def test_heterogeneous_footprint_order_is_exactly_invariant():
    data = _four()
    footprints = [_footprint(0, 10, 10, 0), _footprint(3, 20, 20, 2),
                  _footprint(2, 30, 30, 1), _footprint(1, 40, 40, 3)]
    data = (data[0], np.random.default_rng(123).uniform(0, 20, data[1].shape), data[2], footprints)
    a, sa = _run(data, image="image-one")
    b, sb = _run((*data[:3], footprints[::-1]), image="image-one")
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    pd.testing.assert_frame_equal(sa, sb, check_exact=True)
    assert sa.spot_id.tolist() == [0, 1, 2, 3]


def test_other_nucleus_changes_do_not_reroll_target_nucleus():
    miat, qki, labels, footprints = _independent(count=3)
    a, _ = _run((miat, qki, labels, footprints), image="stable-image")
    b, _ = _run((miat, qki, labels, footprints[3:]), image="stable-image")
    pd.testing.assert_frame_equal(a[a.nucleus_id > 1].reset_index(drop=True),
                                  b[b.nucleus_id > 1].reset_index(drop=True), check_exact=True)


def test_relative_spot_order_preserves_null_when_adapter_ordinals_shift():
    from dataclasses import replace
    data = _independent(count=3)
    a, _ = _run(data, image="stable-image")
    shifted = [replace(fp, spot_index=fp.spot_index + 100) for fp in data[3]]
    b, _ = _run((*data[:3], shifted), image="stable-image")
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_image_identity_seeds_independent_nulls():
    data = _independent(count=3)
    a, _ = _run(data, image="image-one")
    b, _ = _run(data, image="image-two")
    columns = [c for c in a if c.startswith("null_mean_")]
    assert not a[columns].equals(b[columns])
    pd.testing.assert_series_equal(a.frac_miat_spots_qki_pos, b.frac_miat_spots_qki_pos)


def test_support_is_minimum_across_heterogeneous_footprints():
    miat, qki, labels, _ = _four()
    eligible = np.zeros(labels.shape, dtype=bool)
    eligible[:8, :8] = True
    footprints = [_footprint(0, 3, 3, 0), _footprint(1, 4, 4, 1)]
    nuclei, _ = _run((miat, qki, labels, footprints), eligible_mask=eligible)
    row = nuclei.iloc[0]
    assert row.null_valid_center_count == 36
    assert row.n_null_effective == 200
    assert row.na_reason_null_mean_frac_miat_footprint_area_qki_pos == "SPARSE_DOMAIN"
    assert np.isnan(row.null_mean_frac_miat_footprint_area_qki_pos)


def test_assoc_adapter_sorts_source_spot_id_not_uid(tmp_path):
    from fishsuite.core.qki_association_postrun import run_qki_association
    root = _assoc_cached_run(tmp_path / "source")
    spots = pd.read_csv(root / "spot_exact_footprint_metrics.csv.gz")
    pixels = pd.read_csv(root / "footprint_pixels.csv.gz")
    pair = pd.concat([spots, spots], ignore_index=True)
    pair["spot_id"] = ["a", "b"]
    pair["spot_uid"] = ["uid-z", "uid-a"]
    pair.to_csv(root / "spot_exact_footprint_metrics.csv.gz", index=False)
    pixel_pair = pd.concat([pixels, pixels], ignore_index=True)
    pixel_pair["spot_uid"] = ["uid-z", "uid-a"]
    pixel_pair.to_csv(root / "footprint_pixels.csv.gz", index=False)
    out = run_qki_association(root, tmp_path / "out", miat_min=10, qki_min=10, n_null=5)
    result = pd.read_csv(out / "qki_association_per_spot.csv")
    assert result.spot_id.tolist() == ["a", "b", "a", "b", "a", "b"]



@pytest.mark.parametrize("column,denominator_reason", [
    ("frac_miat_spots_qki_pos", "N0"),
    ("frac_qki_area_on_miat_area", "Q0"),
    ("frac_qki_area_on_miat_footprints", "Q0"),
    ("frac_miat_footprint_area_qki_pos", "U0"),
    ("qki_pos_area_frac", "R0"),
    ("miat_pos_area_frac", "R0"),
    ("miat_footprint_area_frac", "R0"),
    ("sat_frac_miat", "R0"),
    ("sat_frac_qki", "R0"),
    ("exp_frac_qki_area_on_miat_area", "R0"),
    ("exp_frac_qki_area_on_miat_footprints", "R0"),
    ("obs_minus_exp_frac_qki_area_on_miat_area", "Q0"),
])
@pytest.mark.parametrize("undefined", [False, True])
def test_fraction_denominator_reason_contract(column, denominator_reason, undefined):
    miat = np.zeros((64, 64), dtype=float)
    qki = np.zeros((64, 64), dtype=float)
    labels = np.ones((64, 64), dtype=np.int32)
    footprints = [_footprint(0, 10, 10)]
    eligible = np.ones((64, 64), dtype=bool)
    if denominator_reason == "Q0":
        qki[:] = 0 if undefined else 20
        footprints = [] if column == "frac_qki_area_on_miat_footprints" and not undefined else footprints
    elif denominator_reason in ("N0", "U0"):
        if undefined:
            footprints = []
    else:
        if column in ("miat_footprint_area_frac", "exp_frac_qki_area_on_miat_footprints"):
            footprints = []
        if column == "sat_frac_miat":
            miat = np.arange(4096, dtype=float).reshape(64, 64)
        if column == "sat_frac_qki":
            qki = np.arange(4096, dtype=float).reshape(64, 64)
        if undefined:
            eligible[:] = False
    nuclei, _ = _run((miat, qki, labels, footprints), eligible_mask=eligible, n_null=3)
    row = nuclei.iloc[0]
    if undefined:
        assert np.isnan(row[column])
        assert row[f"na_reason_{column}"] == denominator_reason
    else:
        assert row[column] == 0.0
        assert row[f"na_reason_{column}"] == ""


@pytest.mark.parametrize("metric,reason", [
    ("frac_miat_spots_qki_pos", "N0"),
    ("frac_miat_footprint_area_qki_pos", "U0"),
])
@pytest.mark.parametrize("prefix", ["null_mean", "null_sd", "obs_minus_null", "null_ge_obs_frac"])
@pytest.mark.parametrize("undefined", [False, True])
def test_null_fraction_denominator_reason_contract(metric, reason, prefix, undefined):
    miat, qki, labels, footprints = _four()
    qki[:] = 0
    if undefined:
        footprints = []
    nuclei, _ = _run((miat, qki, labels, footprints), n_null=3)
    row = nuclei.iloc[0]
    column = f"{prefix}_{metric}"
    if undefined:
        assert np.isnan(row[column])
        assert row[f"na_reason_{column}"] == reason
    else:
        # Q=0 is a defined zero for both N- and U-denominator statistics.
        # All nulls tie the observed zero, so the plus-one tail is 1, not 0.
        assert row[column] == (1.0 if prefix == "null_ge_obs_frac" else 0.0)
        assert row[f"na_reason_{column}"] == ""


def test_generated_dictionary_explains_coverage_denominators_and_null_scope(tmp_path):
    from fishsuite.core.qki_association_postrun import run_qki_association
    root = _assoc_cached_run(tmp_path / "source")
    out = run_qki_association(root, tmp_path / "out", miat_min=10, qki_min=10, n_null=3)
    text = (out / "qki_association_columns.md").read_text()
    header = text.split("- `threshold_multiplier`:", 1)[0]
    entry = next(line for line in text.splitlines() if line.startswith("- `frac_qki_area_on_miat_footprints`:"))
    for block in (header, entry):
        assert "UNCORRECTED for MIAT coverage" in block
        assert "rises with coverage by chance" in block
        assert "obs_minus_null_frac_miat_footprint_area_qki_pos" in block
    assert "## What the placement null does and does not control" in text
    for phrase in ("uniformly over admissible positions", "eligible nuclear region",
                   "MIAT abundance/coverage", "global nuclear QKI level",
                   "same nuclear sub-regions", "nucleolar exclusion zones", "nuclear periphery",
                   "more QKI at MIAT puncta than at random eligible nuclear positions",
                   "not molecular binding", "Single-plane measurement",
                   "undefined denominator", "zero numerator", "empty reason"):
        assert phrase in text
