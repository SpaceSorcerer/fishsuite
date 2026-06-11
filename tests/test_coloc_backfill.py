"""coloc_backfill — CPU re-read QKI coloc backfill (Brian, 2026-06-06).

The trusted MIAT x QKI run was produced BEFORE the native coloc-extra outputs
existed, so it lacks ``coloc_null_draws.csv`` (the 1000 pooled null draws),
``coloc_radial_profile.csv`` and a QKI montage. ``coloc_backfill`` re-reads ONLY
the QKI channel pixels from the source VSI (CPU), reuses the run's SAVED nucleus
masks + MIAT spot coordinates, and emits those artifacts WITHOUT re-segmenting or
re-detecting spots.

The DISSERTATION-CRITICAL contract is that the backfill's pure core reproduces
the engine's already-validated ``protein_pooled_*`` numbers EXACTLY (same disk
r=3 px, n_null=1000, seed=0, in-nucleus, nucleolus-excluded, spot-weighted
pooling). These tests prove that:

  (a) UNIT tests of the pure core ``_compute_coloc_extras_for_image`` on
      synthetic arrays where the answer is known/derivable (bright-at-spots ->
      enrichment > 1; uniform -> enrichment == 1; determinism; nucleolus
      exclusion raises the null floor; radial peak in the matching ring;
      montage crops brighter at spots).
  (b) a REPRODUCTION test: run the engine ``rna_rna.run_one`` on a synthetic
      stack with ``save_partner_null_draws`` + radial ON, pull the SAME
      qki_2d/labels/nucleolus/spots out of ``res.qc``, feed them into the pure
      core, and assert the pooled null VECTOR + pooled obs + empirical-p + the
      radial rows match the engine bit-for-bit (the core IS the engine math,
      single source of truth via the imported ``_partner_null_for_nucleus`` /
      ``_radial_profile_for_nucleus`` helpers).

The real-VSI I/O path (``backfill_run``) is validated at RUN TIME by the
self-validation gate (it reproduces the stored ``per_image_summary`` pooled
columns); it cannot be exercised here without the multi-GB VSI staging tree, so
it is covered by the pure-core + reproduction tests plus an API smoke check.
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

from fishsuite.core.coloc_backfill import (
    _compute_coloc_extras_for_image,
    _montage_crops_for_image,
    _mean_enrichment_patch,
    _matched_random_centers,
    _gate_compare,
    _print_gate,
    backfill_run,
)


# ===========================================================================
# Synthetic-array fixtures for the PURE CORE
# ===========================================================================
def _two_nuclei_labels(h=60, w=60):
    """Two non-touching disk nuclei labelled 1 and 2."""
    from skimage.draw import disk
    labels = np.zeros((h, w), dtype=np.int32)
    rr, cc = disk((20, 18), 12, shape=labels.shape)
    labels[rr, cc] = 1
    rr, cc = disk((20, 42), 12, shape=labels.shape)
    labels[rr, cc] = 2
    return labels


def _spots_for(labels):
    """A handful of spot (x_px, y_px) per nucleus, well inside each disk."""
    spots = {
        1: np.array([[18, 16], [16, 22], [22, 20], [18, 24]], dtype=float),  # (x,y)
        2: np.array([[42, 16], [40, 22], [44, 20], [42, 24]], dtype=float),
    }
    return spots


def _qki_bright_at_spots(labels, spots, base=100.0, bump=600.0, r=3):
    """QKI field that is uniformly ``base`` with a +``bump`` disk of radius ``r``
    painted at every spot centre -> the disk-mean at a spot is much higher than
    a random in-nucleus disk-mean -> enrichment > 1, z > 0."""
    from skimage.draw import disk
    h, w = labels.shape
    qki = np.full((h, w), base, dtype=np.float64)
    for nid, arr in spots.items():
        for (x, y) in arr:
            rr, cc = disk((int(round(y)), int(round(x))), r, shape=qki.shape)
            qki[rr, cc] += bump
    return qki


# ---------------------------------------------------------------------------
# (a) UNIT: pure core — bright-at-spots is enriched
# ---------------------------------------------------------------------------
def test_pure_core_bright_at_spots_is_enriched():
    labels = _two_nuclei_labels()
    spots = _spots_for(labels)
    qki = _qki_bright_at_spots(labels, spots)
    out = _compute_coloc_extras_for_image(
        qki, labels, spots, None,
        disk_px=3.0, n_null=500, seed=0,
        do_null_draws=True, do_radial=False, do_montage=False,
    )
    s = out["null_summary"]
    assert s is not None
    assert s["pooled_null_enrichment"] > 1.2
    assert s["pooled_null_z"] > 0
    # right-tail: observed sits above essentially all null draws
    assert s["pooled_null_p_empirical"] < 0.05
    assert s["n_nuclei_used"] == 2


# ---------------------------------------------------------------------------
# (a) UNIT: pure core — a UNIFORM partner field gives enrichment == 1 exactly
# ---------------------------------------------------------------------------
def test_pure_core_uniform_partner_enrichment_one():
    labels = _two_nuclei_labels()
    spots = _spots_for(labels)
    qki = np.full(labels.shape, 50.0, dtype=np.float64)
    out = _compute_coloc_extras_for_image(
        qki, labels, spots, None,
        disk_px=3.0, n_null=500, seed=0,
        do_null_draws=True, do_radial=False, do_montage=False,
    )
    s = out["null_summary"]
    assert s["pooled_obs"] == pytest.approx(50.0, abs=1e-9)
    assert s["pooled_null_mean"] == pytest.approx(50.0, abs=1e-9)
    assert s["pooled_null_enrichment"] == pytest.approx(1.0, abs=1e-9)
    # every null draw equals the observed -> std 0 -> z NaN, empirical p == 1.0
    assert np.isnan(s["pooled_null_z"])
    assert s["pooled_null_p_empirical"] == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# (a) UNIT: determinism under a fixed seed
# ---------------------------------------------------------------------------
def test_pure_core_deterministic_seed():
    labels = _two_nuclei_labels()
    spots = _spots_for(labels)
    qki = _qki_bright_at_spots(labels, spots)
    o1 = _compute_coloc_extras_for_image(
        qki, labels, spots, None, disk_px=3.0, n_null=400, seed=0,
        do_null_draws=True, do_radial=False, do_montage=False,
    )
    o2 = _compute_coloc_extras_for_image(
        qki, labels, spots, None, disk_px=3.0, n_null=400, seed=0,
        do_null_draws=True, do_radial=False, do_montage=False,
    )
    np.testing.assert_array_equal(
        o1["null_draws_rows"]["pooled_null_value"].to_numpy(),
        o2["null_draws_rows"]["pooled_null_value"].to_numpy(),
    )
    assert o1["null_summary"]["pooled_obs"] == o2["null_summary"]["pooled_obs"]


# ---------------------------------------------------------------------------
# (a) UNIT: null_draws shape + empirical-p reconstruction
# ---------------------------------------------------------------------------
def test_pure_core_null_draws_shape_and_p():
    labels = _two_nuclei_labels()
    spots = _spots_for(labels)
    qki = _qki_bright_at_spots(labels, spots)
    out = _compute_coloc_extras_for_image(
        qki, labels, spots, None, disk_px=3.0, n_null=250, seed=0,
        do_null_draws=True, do_radial=False, do_montage=False,
        image="img.vsi", condition="cond",
    )
    draws = out["null_draws_rows"]
    assert isinstance(draws, pd.DataFrame)
    assert len(draws) == 250
    assert {"image", "condition", "iter", "pooled_null_value", "pooled_obs"}.issubset(draws.columns)
    assert list(draws["iter"]) == list(range(250))
    assert draws["image"].unique().tolist() == ["img.vsi"]
    obs = float(draws["pooled_obs"].iloc[0])
    p_recon = (int((draws["pooled_null_value"] >= obs).sum()) + 1) / (250 + 1)
    assert p_recon == pytest.approx(out["null_summary"]["pooled_null_p_empirical"], abs=1e-12)
    assert float(draws["pooled_null_value"].mean()) == pytest.approx(
        out["null_summary"]["pooled_null_mean"], abs=1e-9
    )


# ---------------------------------------------------------------------------
# (a) UNIT: nucleolus exclusion raises the null floor -> lowers enrichment
# ---------------------------------------------------------------------------
def test_pure_core_nucleolus_exclusion_changes_null():
    """A DIM (partner-poor) nucleolar void inside each nucleus. Excluding it from
    the null sampling positions raises the null floor (random draws can no longer
    fall into the void) -> lower enrichment than the whole-nucleus null. Mirrors
    the engine's exclude_nucleolus_from_partner_null contract."""
    from skimage.draw import disk
    labels = _two_nuclei_labels()
    spots = _spots_for(labels)
    qki = _qki_bright_at_spots(labels, spots, base=100.0, bump=600.0, r=3)
    # Carve a DIM void (the "nucleolus") into a part of each nucleus AWAY from
    # the spots, and build the matching nucleolus label image (parent-id labels).
    nucleolus = np.zeros(labels.shape, dtype=np.int32)
    for nid, (cy, cx) in ((1, (26, 22)), (2, (26, 46))):
        rr, cc = disk((cy, cx), 4, shape=qki.shape)
        inside = labels[rr, cc] == nid
        qki[rr[inside], cc[inside]] = 5.0          # partner avoids the void
        nucleolus[rr[inside], cc[inside]] = nid

    out_whole = _compute_coloc_extras_for_image(
        qki, labels, spots, None, disk_px=3.0, n_null=600, seed=0,
        do_null_draws=True, do_radial=False, do_montage=False,
    )
    out_excl = _compute_coloc_extras_for_image(
        qki, labels, spots, nucleolus, disk_px=3.0, n_null=600, seed=0,
        do_null_draws=True, do_radial=False, do_montage=False,
    )
    e_whole = out_whole["null_summary"]["pooled_null_enrichment"]
    e_excl = out_excl["null_summary"]["pooled_null_enrichment"]
    nm_whole = out_whole["null_summary"]["pooled_null_mean"]
    nm_excl = out_excl["null_summary"]["pooled_null_mean"]
    assert nm_excl > nm_whole          # exclusion removes the dim void -> higher null
    assert e_excl < e_whole            # -> lower enrichment


# ---------------------------------------------------------------------------
# (a) UNIT: radial profile peaks in the ring matching a bright annulus
# ---------------------------------------------------------------------------
def test_pure_core_radial_peak_in_matching_ring():
    from skimage.draw import disk
    labels = _two_nuclei_labels()
    spots = _spots_for(labels)
    qki = np.full(labels.shape, 50.0, dtype=np.float64)
    # Paint a bright ANNULUS at radius ~5 px (in ring (4,6] = index 2) around
    # every spot: ring of disk(r=6) minus disk(r=4).
    for nid, arr in spots.items():
        for (x, y) in arr:
            cy, cx = int(round(y)), int(round(x))
            ro_r, ro_c = disk((cy, cx), 6, shape=qki.shape)
            ri_r, ri_c = disk((cy, cx), 4, shape=qki.shape)
            ann = np.zeros(qki.shape, dtype=bool)
            ann[ro_r, ro_c] = True
            ann[ri_r, ri_c] = False
            qki[ann] = 800.0
    bins_px = [2.0, 4.0, 6.0, 8.0]
    out = _compute_coloc_extras_for_image(
        qki, labels, spots, None, disk_px=3.0, n_null=400, seed=0,
        do_null_draws=False, do_radial=True, radial_bins_px=bins_px,
        do_montage=False,
    )
    rows = out["radial_rows"]
    assert len(rows) == len(bins_px)
    enr = {r["ring_idx"]: r["enrichment"] for r in rows}
    # ring index 2 = outer edge 6 px = the (4,6] annulus = the painted ring.
    assert enr[2] == max(enr.values())
    assert enr[2] > 1.2


# ---------------------------------------------------------------------------
# (a) UNIT: montage crops — obs crops brighter than null crops at bright spots
# ---------------------------------------------------------------------------
def test_montage_crops_obs_brighter_than_null():
    labels = _two_nuclei_labels()
    spots = _spots_for(labels)
    qki = _qki_bright_at_spots(labels, spots, base=50.0, bump=900.0, r=3)
    out = _compute_coloc_extras_for_image(
        qki, labels, spots, None, disk_px=3.0, n_null=50, seed=0,
        do_null_draws=False, do_radial=False, do_montage=True,
        montage_n_nuclei=2, montage_max_spots=4, montage_crop_half=6,
    )
    crops = out["montage_crops"]
    assert len(crops) >= 1
    obs_means, null_means = [], []
    for c in crops:
        assert len(c["obs_crops"]) >= 1
        assert len(c["null_crops"]) == len(c["obs_crops"])
        for oc in c["obs_crops"]:
            obs_means.append(float(np.asarray(oc).mean()))
        for nc in c["null_crops"]:
            null_means.append(float(np.asarray(nc).mean()))
    assert np.mean(obs_means) > np.mean(null_means)
    # crops are square and the configured size
    assert np.asarray(crops[0]["obs_crops"][0]).shape == (13, 13)  # 2*6+1


def test_montage_crops_deterministic():
    labels = _two_nuclei_labels()
    spots = _spots_for(labels)
    qki = _qki_bright_at_spots(labels, spots)
    a = _montage_crops_for_image(
        qki, [(1, np.array([16]), np.array([18]),
               *np.where(labels == 1))],
        crop_half=5, n_nuclei=1, max_spots=3, seed=0,
    )
    b = _montage_crops_for_image(
        qki, [(1, np.array([16]), np.array([18]),
               *np.where(labels == 1))],
        crop_half=5, n_nuclei=1, max_spots=3, seed=0,
    )
    np.testing.assert_array_equal(a[0]["null_crops"][0], b[0]["null_crops"][0])


# ===========================================================================
# (a) UNIT: mean ENRICHMENT patch — the headline of the new montage. Crops are
# per-nucleus normalized (enrichment units) BEFORE averaging, so a MODEST (~10%)
# central enrichment that is invisible in a single raw crop emerges in the mean.
# ===========================================================================
def test_mean_enrichment_patch_center_gt_edge_and_gt_one():
    """A radially-symmetric bright blob centred on every MIAT spot -> the mean
    enrichment patch has a CENTER pixel that (a) exceeds 1 (brighter than the
    nucleus mean) and (b) exceeds its own EDGE; n_used counts every spot."""
    labels = _two_nuclei_labels()
    spots = _spots_for(labels)
    qki = _qki_bright_at_spots(labels, spots, base=100.0, bump=600.0, r=3)
    patch, n_used = _mean_enrichment_patch(qki, labels, None, spots, 8)
    assert n_used == 8                          # 4 spots x 2 nuclei
    assert patch.shape == (17, 17)              # 2*8 + 1
    h = patch.shape[0] // 2
    center = float(patch[h, h])
    edge = float(np.mean([patch[0, h], patch[-1, h], patch[h, 0], patch[h, -1]]))
    assert center > 1.0                         # brighter than nucleus mean
    assert center > edge                        # central enrichment over edge


def test_mean_enrichment_patch_flat_is_one():
    """A FLAT QKI field normalizes to enrichment == 1 everywhere (the crop equals
    the nucleus mean at every pixel)."""
    labels = _two_nuclei_labels()
    spots = _spots_for(labels)
    qki = np.full(labels.shape, 50.0, dtype=np.float64)
    patch, n_used = _mean_enrichment_patch(qki, labels, None, spots, 6)
    assert n_used == 8
    np.testing.assert_allclose(patch, 1.0, atol=1e-9)


def test_mean_enrichment_patch_unsupported_normalize_raises():
    labels = _two_nuclei_labels()
    spots = _spots_for(labels)
    qki = np.full(labels.shape, 50.0, dtype=np.float64)
    with pytest.raises(ValueError):
        _mean_enrichment_patch(qki, labels, None, spots, 6, normalize_by="zscore")


def test_matched_random_centers_deterministic_and_matched_count():
    """The matched-random generator draws the SAME count per nucleus as it has
    MIAT spots, is deterministic for a fixed seed, and varies with the seed."""
    labels = _two_nuclei_labels()
    spots = _spots_for(labels)
    a = _matched_random_centers(labels, None, spots, seed=0)
    b = _matched_random_centers(labels, None, spots, seed=0)
    assert set(a.keys()) == {1, 2}
    for nid in (1, 2):
        np.testing.assert_array_equal(a[nid], b[nid])      # determinism
        assert a[nid].shape == (len(spots[nid]), 2)        # matched count, (x,y)
    c = _matched_random_centers(labels, None, spots, seed=1)
    assert not np.array_equal(a[1], c[1])                   # seed-sensitive


def test_matched_random_centers_stay_in_nucleus():
    """Every drawn random centre must lie inside its own nucleus's pixels."""
    labels = _two_nuclei_labels()
    spots = _spots_for(labels)
    rc = _matched_random_centers(labels, None, spots, seed=0)
    for nid, xy in rc.items():
        for (x, y) in xy:
            assert labels[int(round(y)), int(round(x))] == nid


def test_matched_random_patch_flatter_than_miat():
    """On the blob image the matched-random patch is LOWER-CENTER and FLATTER
    (smaller centre-to-mean contrast) than the MIAT patch — the visual proof that
    the central enrichment is MIAT-specific, not a cropping artifact."""
    labels = _two_nuclei_labels()
    spots = _spots_for(labels)
    qki = _qki_bright_at_spots(labels, spots, base=100.0, bump=600.0, r=3)
    half = 8
    miat_patch, _ = _mean_enrichment_patch(qki, labels, None, spots, half)
    rand_centers = _matched_random_centers(labels, None, spots, seed=0)
    rand_patch, _ = _mean_enrichment_patch(qki, labels, None, rand_centers, half)
    miat_center = float(miat_patch[half, half])
    rand_center = float(rand_patch[half, half])
    assert rand_center < miat_center                                  # lower-center
    miat_contrast = miat_center - float(miat_patch.mean())
    rand_contrast = rand_center - float(rand_patch.mean())
    assert rand_contrast < miat_contrast                             # flatter


def test_mean_enrichment_patch_nucleolus_excluded_from_norm():
    """A DIM nucleolar void inside each nucleus is excluded from the nucleus-mean
    normalization (mirroring the null sampling region), so the void does not drag
    the nucleus mean down and inflate the enrichment. Excluding it -> a HIGHER
    nucleus mean -> a LOWER central enrichment than the whole-nucleus norm."""
    from skimage.draw import disk
    labels = _two_nuclei_labels()
    spots = _spots_for(labels)
    qki = _qki_bright_at_spots(labels, spots, base=100.0, bump=600.0, r=3)
    nucleolus = np.zeros(labels.shape, dtype=np.int32)
    for nid, (cy, cx) in ((1, (26, 22)), (2, (26, 46))):
        rr, cc = disk((cy, cx), 4, shape=qki.shape)
        inside = labels[rr, cc] == nid
        qki[rr[inside], cc[inside]] = 5.0          # dim void
        nucleolus[rr[inside], cc[inside]] = nid
    half = 8
    p_whole, _ = _mean_enrichment_patch(qki, labels, None, spots, half)
    p_excl, _ = _mean_enrichment_patch(qki, labels, nucleolus, spots, half)
    # excluding the dim void raises the denominator -> lower central enrichment
    assert p_excl[half, half] < p_whole[half, half]


def test_render_mean_enrichment_montage_writes_png(tmp_path):
    """Render smoke test: the headline figure renders to a non-empty 600-DPI PNG
    at the canonical filename, with the example raw-crop strip attached."""
    from fishsuite.core.coloc_backfill import (
        _render_mean_enrichment_montage,
        _montage_crops_for_image,
    )
    labels = _two_nuclei_labels()
    spots = _spots_for(labels)
    qki = _qki_bright_at_spots(labels, spots, base=100.0, bump=600.0, r=3)
    half = 8
    miat_patch, n_m = _mean_enrichment_patch(qki, labels, None, spots, half)
    rc = _matched_random_centers(labels, None, spots, seed=0)
    rand_patch, n_r = _mean_enrichment_patch(qki, labels, None, rc, half)
    ex = _montage_crops_for_image(
        qki, [(1, np.array([16, 22]), np.array([18, 20]), *np.where(labels == 1))],
        crop_half=half, n_nuclei=1, max_spots=2, seed=0,
    )
    png = tmp_path / "79_coloc_qki_montage_at_miat_vs_random.png"
    _render_mean_enrichment_montage(
        miat_patch, rand_patch, png,
        n_miat=n_m, n_rand=n_r, example_crops=ex,
        example_vmin=100.0, example_vmax=700.0,
        condition_label="g2-wDox+g2-noDox", pixel_um=0.13, disk_r_px=3.0,
    )
    assert png.exists() and png.stat().st_size > 5_000


# ===========================================================================
# (b) REPRODUCTION through the real engine (rna_rna.run_one)
# ===========================================================================
DAPI_C, RNA_C, PART_C = 0, 1, 2
NZ = 4
H = W = 200


class _FakeBio:
    def __init__(self, czyx):
        self._czyx = czyx

    def get_image_data(self, order, *, T=0, C=0):  # noqa: N803
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
            y = int(cy + 15 * np.sin(ang)); x = int(cx + 15 * np.cos(ang))
            blob[y, x] += float(rng.uniform(3000.0, 6000.0))
    return img + gaussian_filter(blob, 1.1)


def _partner_plane():
    from skimage.draw import disk
    img = np.random.default_rng(44).uniform(2.0, 8.0, (H, W)).astype(np.float32)
    for (cy, cx) in _nuclei_centers():
        rr, cc = disk((cy, cx), 28, shape=img.shape)
        img[rr, cc] = 800.0
        hr, hc = disk((cy, cx), 7, shape=img.shape)
        img[hr, hc] = 20.0
    return img


def _czyx():
    planes = [_dapi_plane(), _rna_spot_plane(), _partner_plane()]
    return np.stack([np.stack([p] * NZ, axis=0) for p in planes], axis=0).astype(np.float32)


@pytest.fixture()
def fake_img():
    return ImageWrapper(
        path="synthetic_backfill.tif", bio=_FakeBio(_czyx()), scene_idx=0,
        shape=(1, 3, NZ, H, W), channel_names=["DAPI", "RNA", "PART"],
        voxel_xy_nm=130.0, voxel_z_nm=300.0, n_channels=3, n_z=NZ,
    )


def _engine_cfg():
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
    # coloc null + radial + draws ON; nucleolus exclusion ON (UD config)
    cfg.foci.compute_partner_intensity = True
    cfg.foci.compute_partner_null_enrichment = True
    cfg.foci.partner_null_n = 300
    cfg.foci.partner_null_seed = 0
    cfg.foci.partner_null_disk_px = 3.0
    cfg.foci.exclude_nucleolus_from_partner_null = True
    cfg.foci.compute_partner_radial_profile = True
    cfg.foci.partner_radial_bins_um = [0.25, 0.5, 0.75, 1.0]
    cfg.foci.save_partner_null_draws = True
    cfg.nucleolus.enabled = True
    return cfg


def _spots_by_nid_from_engine(spots1_df):
    out = {}
    if spots1_df is None or len(spots1_df) == 0:
        return out
    for nid, grp in spots1_df.groupby("nucleus_id"):
        try:
            nid = int(nid)
        except (TypeError, ValueError):
            continue
        if nid < 1:
            continue
        out[nid] = grp[["x_px", "y_px"]].astype(float).to_numpy()
    return out


def test_repro_pure_core_matches_engine_null(fake_img, monkeypatch):
    """Feed the engine's OWN qki_2d / labels / nucleolus / spots into the pure
    core and assert the pooled null VECTOR + pooled obs + empirical p reproduce
    the engine bit-for-bit (single source of truth)."""
    monkeypatch.setattr(_io, "read_image", lambda p: fake_img)
    cfg = _engine_cfg()
    res = _rna_rna.run_one(Path(fake_img.path), condition="cond", sec_only=False, cfg=cfg)

    qki = res.qc["rna2_2d"]
    labels = res.qc["labels"]
    nucleolus = res.qc["nucleolus_labels"]  # used because exclusion is ON
    spots = _spots_by_nid_from_engine(res.qc["spots1"])

    # radial bins (px) exactly as the engine computes them.
    pix_um = float(fake_img.voxel_xy_nm) / 1000.0
    bins_px = [b / pix_um for b in cfg.foci.partner_radial_bins_um if b > 0]

    out = _compute_coloc_extras_for_image(
        qki, labels, spots, nucleolus,
        disk_px=3.0, n_null=300, seed=0, radial_bins_px=bins_px,
        do_null_draws=True, do_radial=True, do_montage=False,
        image=res.image, condition=res.condition,
    )

    # The engine, with save_partner_null_draws ON, surfaced the exact pooled
    # null vector + pooled obs it used internally.
    eng_draws = res.extra["coloc_null_draws"]
    core_draws = out["null_draws_rows"]
    np.testing.assert_allclose(
        core_draws["pooled_null_value"].to_numpy(),
        eng_draws["pooled_null_value"].to_numpy(),
        rtol=0, atol=1e-9,
    )
    assert float(core_draws["pooled_obs"].iloc[0]) == pytest.approx(
        float(eng_draws["pooled_obs"].iloc[0]), abs=1e-9
    )

    s = out["null_summary"]
    # match the per_image rounded values (engine rounds obs/null_mean to 3,
    # enrichment to 4, z to 3; p is unrounded).
    assert s["pooled_obs"] == pytest.approx(
        float(res.per_image["rna2_pooled_obs_at_rna1_spots"]), abs=1e-3)
    assert s["pooled_null_mean"] == pytest.approx(
        float(res.per_image["rna2_pooled_null_mean_at_rna1_spots"]), abs=1e-3)
    assert s["pooled_null_enrichment"] == pytest.approx(
        float(res.per_image["rna2_pooled_enrichment_vs_null_at_rna1_spots"]), abs=1e-4)
    assert s["pooled_null_z"] == pytest.approx(
        float(res.per_image["rna2_pooled_null_z_at_rna1_spots"]), abs=1e-3)
    assert s["pooled_null_p_empirical"] == pytest.approx(
        float(res.per_image["rna2_pooled_null_p_empirical_at_rna1_spots"]), abs=1e-12)
    assert s["n_nuclei_used"] == int(res.per_image["n_nuclei_partner_null"])


def test_repro_pure_core_matches_engine_radial(fake_img, monkeypatch):
    monkeypatch.setattr(_io, "read_image", lambda p: fake_img)
    cfg = _engine_cfg()
    res = _rna_rna.run_one(Path(fake_img.path), condition="cond", sec_only=False, cfg=cfg)

    qki = res.qc["rna2_2d"]
    labels = res.qc["labels"]
    nucleolus = res.qc["nucleolus_labels"]
    spots = _spots_by_nid_from_engine(res.qc["spots1"])
    pix_um = float(fake_img.voxel_xy_nm) / 1000.0
    bins_px = [b / pix_um for b in cfg.foci.partner_radial_bins_um if b > 0]

    out = _compute_coloc_extras_for_image(
        qki, labels, spots, nucleolus,
        disk_px=3.0, n_null=300, seed=0, radial_bins_px=bins_px,
        do_null_draws=False, do_radial=True, do_montage=False,
    )
    eng_rad = res.extra["coloc_radial_profile"].reset_index(drop=True)
    core_rows = out["radial_rows"]
    assert len(core_rows) == len(eng_rad)
    for row in core_rows:
        eng_row = eng_rad.iloc[row["ring_idx"]]
        assert row["obs_mean"] == pytest.approx(float(eng_row["obs_mean"]), abs=1e-7)
        assert row["null_mean"] == pytest.approx(float(eng_row["null_mean"]), abs=1e-7)
        assert row["enrichment"] == pytest.approx(float(eng_row["enrichment"]), abs=1e-7)


# ===========================================================================
# (c) API smoke: backfill_run signature + graceful behaviour on a bogus dir
# ===========================================================================
def test_backfill_run_is_callable_and_validates_inputs(tmp_path):
    """backfill_run exists with the documented signature and raises a clear
    error when the run_dir has no run_config.json (rather than silently
    succeeding)."""
    import inspect
    sig = inspect.signature(backfill_run)
    for p in ("run_dir", "staging_dir", "input_dir", "do_null_draws",
              "do_radial", "do_montage", "seed"):
        assert p in sig.parameters
    with pytest.raises((FileNotFoundError, ValueError)):
        backfill_run(tmp_path)


# ===========================================================================
# (d) REPORTING-PATH bug regressions (2026-06-06)
# ===========================================================================
def _fake_prow_and_summary(match=True):
    """A stored per_image row (protein_pooled_* like the trusted run) + a
    recomputed null_summary. ``match=True`` -> the recomputed stats equal the
    stored ones (gate PASS); ``match=False`` -> grossly different (gate FAIL)."""
    s = {
        "pooled_obs": 1815.057,
        "pooled_null_mean": 1659.617,
        "pooled_null_sd": 21.9,
        "pooled_null_enrichment": 1.0937,
        "pooled_null_z": 7.097,
        "pooled_null_p_empirical": 0.000999,
        "n_nuclei_used": 35,
        "n_null": 1000,
        "disk_px": 3.0,
    }
    prow = {
        "image": "UD-MIAT-FISH-QKI-IF-g2-no Dox_03.vsi",
        "protein_pooled_obs_at_rna1_spots": 1815.057 if match else 999.0,
        "protein_pooled_null_mean_at_rna1_spots": 1659.617 if match else 900.0,
        "protein_pooled_enrichment_vs_null_at_rna1_spots": 1.0937 if match else 1.6,
        "protein_pooled_null_z_at_rna1_spots": 7.097,
        "protein_pooled_null_p_empirical_at_rna1_spots": 0.000999,
    }
    return prow, s


def test_gate_compare_sets_pass_key():
    """BUG 1 regression: ``_gate_compare`` must itself set a boolean ``"pass"``
    key (the per-image verbose block reads ``gate_rows[-1]["pass"]`` BEFORE
    ``_print_gate`` runs). A match -> True; a gross mismatch -> False."""
    prow_ok, s_ok = _fake_prow_and_summary(match=True)
    row_ok = _gate_compare(prow_ok, s_ok)
    assert "pass" in row_ok
    assert isinstance(row_ok["pass"], bool)
    assert row_ok["pass"] is True

    prow_bad, s_bad = _fake_prow_and_summary(match=False)
    row_bad = _gate_compare(prow_bad, s_bad)
    assert row_bad["pass"] is False


def test_print_gate_reuses_pass_single_source_of_truth(capsys):
    """``_print_gate`` must REUSE the per-row ``"pass"`` computed by
    ``_gate_compare`` (single source of truth), not silently recompute its own
    verdict from a different tol."""
    prow, s = _fake_prow_and_summary(match=True)
    row = _gate_compare(prow, s)
    assert row["pass"] is True
    # Flip the authoritative verdict; _print_gate must honor it.
    row["pass"] = False
    res = _print_gate([row])
    assert res["n_fail"] == 1
    assert res["n_pass"] == 0


def test_print_gate_is_ascii_only_and_does_not_raise(capsys):
    """BUG 2 regression (b): the gate report must be ASCII-encodable so it does
    not die on a cp1252 Windows console (no ``Δ`` glyph)."""
    rows = [_gate_compare(*_fake_prow_and_summary(match=True))]
    res = _print_gate(rows)
    assert res["n_fail"] == 0
    out = capsys.readouterr().out
    assert "SELF-VALIDATION GATE" in out
    assert "PASS" in out
    # The whole report must be plain ASCII (the cp1252 console can encode it).
    out.encode("ascii")  # raises UnicodeEncodeError if any non-ASCII glyph leaks
    assert "Δ" not in out  # no Greek delta anywhere


def test_print_gate_writes_to_cp1252_stream_without_error():
    """BUG 2 regression (faithful): redirect stdout to a STRICT cp1252 stream
    (what the real Windows console is) and assert ``_print_gate`` writes the
    full report without a UnicodeEncodeError."""
    import io
    import contextlib

    rows = [
        _gate_compare(*_fake_prow_and_summary(match=True)),
        _gate_compare(*_fake_prow_and_summary(match=False)),
    ]
    buf = io.BytesIO()
    wrapper = io.TextIOWrapper(buf, encoding="cp1252", errors="strict", newline="")
    with contextlib.redirect_stdout(wrapper):
        _print_gate(rows)  # must NOT raise UnicodeEncodeError
        wrapper.flush()
    text = buf.getvalue().decode("cp1252")
    assert "SELF-VALIDATION GATE" in text
    assert "WARNING" in text  # the FAIL row triggers the warning banner
