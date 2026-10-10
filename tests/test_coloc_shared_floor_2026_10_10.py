"""Shared-floor coloc metrics (coloc_pixel_metrics, 2026-10-10) and scripts/coloc_extra_metrics.py on synthetic data."""
import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from fishsuite.core import coloc_pixel_metrics as cpm
from fishsuite.core.exact_footprint_backfill import write_selected_plane_h5_group

ROOT = Path(__file__).resolve().parents[1]


def _script():
    spec = importlib.util.spec_from_file_location("coloc_extra_metrics", ROOT / "scripts" / "coloc_extra_metrics.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_manders_is_one_for_identical_masks_above_floor():
    rng = np.random.default_rng(0)
    on = rng.random((40, 40)) < 0.3
    miat = np.where(on, 1000 + rng.random((40, 40)) * 500, 10.0)
    qki = np.where(on, 2000 + rng.random((40, 40)) * 500, 20.0)
    r = cpm.manders_shared_floor(miat, qki, np.ones_like(on), miat_floor=500, qki_floor=1000)
    assert r["manders_m1_shared_floor"] == pytest.approx(1.0)
    assert r["manders_m2_shared_floor"] == pytest.approx(1.0)


def test_manders_is_bounded_when_qki_positive_area_exceeds_miat_positive_area():
    miat = np.full((10, 10), 100.0); miat[:3] = 1000.0
    qki = np.full((10, 10), 100.0); qki[:8] = 1000.0
    r = cpm.manders_shared_floor(miat, qki, np.ones((10, 10), bool), miat_floor=500, qki_floor=500)
    assert r["manders_m1_shared_floor"] == pytest.approx(1.0)
    assert r["manders_m2_shared_floor"] == pytest.approx(3 / 8)
    assert 0 <= r["manders_m1_shared_floor"] <= 1


def test_manders_floor_is_strict_and_empty_denominator_is_nan_with_reason():
    miat = np.full((10, 10), 500.0)
    r = cpm.manders_shared_floor(miat, miat, np.ones((10, 10), bool), miat_floor=500, qki_floor=500)
    assert np.isnan(r["manders_m1_shared_floor"]) and r["na_reason_manders_m1_shared_floor"] == "NONE_ABOVE_FLOOR"
    with pytest.raises(ValueError):
        cpm.manders_shared_floor(miat, miat, np.ones((10, 10), bool), miat_floor=np.nan, qki_floor=1)


def test_icq_near_zero_for_independent_noise_and_high_for_covarying_channels():
    rng = np.random.default_rng(1)
    mask = np.ones((200, 200), bool)
    a = 1000 + rng.normal(0, 100, mask.shape)
    b = 1000 + rng.normal(0, 100, mask.shape)
    ind = cpm.above_floor_pearson_icq(a, b, mask, miat_floor=0, qki_floor=0)
    assert abs(ind["li_icq_above_floor"]) < 0.02 and abs(ind["pearson_r_above_floor"]) < 0.03
    shared = rng.normal(0, 100, mask.shape)
    co = cpm.above_floor_pearson_icq(1000 + shared + rng.normal(0, 20, mask.shape),
                                     2000 + shared + rng.normal(0, 20, mask.shape), mask, miat_floor=0, qki_floor=0)
    assert co["li_icq_above_floor"] > 0.3 and co["pearson_r_above_floor"] > 0.9


def test_above_floor_set_excludes_pixels_below_either_floor():
    miat = np.arange(100, dtype=float).reshape(10, 10)
    r = cpm.above_floor_pearson_icq(miat, miat, np.ones((10, 10), bool), miat_floor=49, qki_floor=89, min_pixels=5)
    assert r["n_pixels_both_above_floor"] == 10 and r["pearson_r_above_floor"] == pytest.approx(1.0)


def test_object_overlap_fraction_and_nn_distance_on_constructed_case():
    qki = np.zeros((40, 40)); qki[5:8, 5:8] = 1000; qki[30:33, 30:33] = 1000
    mask = np.ones((40, 40), bool)
    centres = np.array([[6, 6], [31, 31], [20, 20], [6, 15]], float)
    fps = [(np.array([c[0], c[0] + 1]), np.array([c[1], c[1]])) for c in centres.astype(int)]
    r = cpm.object_coloc(qki, mask, centres, fps, qki_floor=500, pixel_um=0.065)
    assert r["n_qki_objects"] == 2
    assert r["frac_miat_spots_on_qki_object"] == pytest.approx(0.5)
    d = sorted([0, 0, np.hypot(14, 14), 9])
    assert r["median_nn_dist_um_miat_to_qki_object"] == pytest.approx(np.median(d) * 0.065)


def test_object_components_are_eight_connected_and_masked():
    qki = np.zeros((10, 10)); qki[2, 2] = qki[3, 3] = 1000; qki[8, 8] = 1000
    mask = np.ones((10, 10), bool); mask[8, 8] = False
    r = cpm.object_coloc(qki, mask, np.zeros((0, 2)), [], qki_floor=500, pixel_um=0.1)
    assert r["n_qki_objects"] == 1 and r["na_reason_object_coloc"] == "NO_SPOTS"


def test_local_pearson_window_mean_and_border_clipping():
    rng = np.random.default_rng(2)
    a = rng.normal(0, 1, (30, 30))
    mask = np.ones((30, 30), bool)
    r = cpm.local_pearson_at_spots(a, 3 * a + 5, mask, np.array([[15, 15], [0, 0], [29, 1]]))
    assert r["mean_local_pearson_5x5"] == pytest.approx(1.0) and r["n_spots_local_pearson"] == 3
    r = cpm.local_pearson_at_spots(a, a, mask & False, np.array([[15, 15]]))
    assert np.isnan(r["mean_local_pearson_5x5"]) and r["na_reason_local_pearson"] == "NO_ESTIMABLE_SPOT"


def _mini_backfill(root: Path, voxel_nm=65.0):
    rng = np.random.default_rng(3)
    shape = (60, 60)
    labels = np.zeros(shape, np.int32); labels[5:30, 5:30] = 1; labels[32:58, 32:58] = 2
    nucleoli = np.zeros(shape, np.int32); nucleoli[10:14, 10:14] = 1
    miat = rng.integers(0, 200, shape).astype(np.uint16)
    qki = rng.integers(0, 200, shape).astype(np.uint16)
    for y, x in ((20, 20), (45, 45)):
        miat[y - 1:y + 2, x - 1:x + 2] = 2000; qki[y - 1:y + 2, x - 1:x + 2] = 3000
    miat[12, 12] = 2000
    dapi = rng.integers(0, 200, shape).astype(np.uint16)
    with h5py.File(root / "selected_planes_and_masks.h5", "w") as f:
        write_selected_plane_h5_group(f, image_key="img_a.vsi", selected_z_1based=4,
                                      planes={"miat": miat, "qki": qki, "dapi": dapi}, nucleus_labels=labels,
                                      nucleolus_labels=nucleoli, channel_indices={"miat": 0, "qki": 1, "dapi": 2})
    pd.DataFrame([dict(image="img_a.vsi", image_key="img_a.vsi", condition="VPR noDox", selected_z_1based=4,
                       miat_channel_index=0, qki_channel_index=1, dapi_channel_index=2, voxel_xy_nm=voxel_nm)]
                 ).to_csv(root / "image_manifest.csv", index=False)
    spots = pd.DataFrame(dict(spot_uid=["s1", "s2", "s3", "s4"], image_key="img_a.vsi", nucleus_id=[1, 2, 1, 2],
                              center_y_px=[20, 45, 12, 40], center_x_px=[20, 45, 12, 50]))
    spots.to_csv(root / "spot_exact_footprint_metrics.csv.gz", index=False, compression="gzip")
    pix = [dict(spot_uid=u, pixel_index=i, y_px=y + dy, x_px=x + dx)
           for u, y, x in zip(spots.spot_uid, spots.center_y_px, spots.center_x_px)
           for i, (dy, dx) in enumerate(((0, 0), (0, 1)))]
    pd.DataFrame(pix).to_csv(root / "footprint_pixels.csv.gz", index=False, compression="gzip")
    pd.DataFrame(dict(image_key=["img_a.vsi"], n_eligible_spots=[4])).to_csv(root / "spot_eligibility_summary.csv", index=False)


def test_script_writes_per_nucleus_table_and_provenance(tmp_path):
    _mini_backfill(tmp_path)
    out = tmp_path / "out"
    _script().main(["--backfill", str(tmp_path), "--miat-floor", "500", "--qki-floor", "1000",
                    "--pixel-um", "0.065", "--out", str(out)])
    t = pd.read_csv(out / "coloc_extra_metrics.csv").set_index("nucleus_id")
    assert list(t.index) == [1, 2]
    assert t.loc[1, "n_spots_parent"] == 2 and t.loc[1, "n_spots_nucleoplasm"] == 1  # s3 sits on the nucleolus
    assert t.loc[1, "frac_miat_spots_on_qki_object"] == 1.0
    assert t.loc[2, "frac_miat_spots_on_qki_object"] == 0.5
    assert t.loc[1, "manders_m1_shared_floor"] == pytest.approx(1.0)
    assert (out / "command.log").is_file() and (out / "versions.txt").is_file()
    assert "n_eligible_spots total 4" in (out / "command.log").read_text(encoding="utf-8")


def test_script_refuses_pixel_size_that_disagrees_with_manifest(tmp_path):
    _mini_backfill(tmp_path, voxel_nm=130.0)
    with pytest.raises(ValueError, match="disagrees with manifest"):
        _script().main(["--backfill", str(tmp_path), "--miat-floor", "500", "--qki-floor", "1000",
                        "--pixel-um", "0.065", "--out", str(tmp_path / "out")])
