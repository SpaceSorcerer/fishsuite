"""MIAT x QKI ASSOCIATION metrics — footprint-based per-spot QKI + continuous /
floor-gated association ratios + threshold-free MOC/ICQ (Brian, 2026-07-07).

Tests the additive, floor-robust association metrics from the approved spec
(_SPEC_association_analysis_2026-07-06.md), gated behind
``foci.compute_footprint_enrichment`` (default OFF).

Families:
  (a) HELPER-LEVEL footprint sampling on a synthetic MIAT-spot-on-QKI image:
      constant-QKI footprint == known value; larger spot -> larger footprint
      area; gradient QKI -> footprint mean == value at the spot; flat crop ->
      fitted-radius disk fallback.
  (b) HELPER-LEVEL MOC/ICQ on correlated vs anti-correlated fields.
  (c) END-TO-END through run_one: columns present; enrichment normalization
      exact; continuous association ratio == mean per-spot enrichment; MOC/ICQ
      alias the existing threshold-free cosine/Li columns; floor-gated <=
      continuous; determinism.
  (d) DEFAULTS-OFF byte-equivalence (no new columns; identical output).
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
from fishsuite.core.metrics import compute_coloc_metrics
from fishsuite.core.modes import rna_rna as _rna_rna
from fishsuite.core.modes.rna_rna import _sample_qki_at_miat_footprint


# ===========================================================================
# (a) HELPER-LEVEL footprint sampling
# ===========================================================================
def _gauss(H, W, cy, cx, sigma, amp, bg):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    return bg + amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma ** 2))


def _spots(y, x, diam_um):
    return pd.DataFrame(
        {"y_px": [float(y)], "x_px": [float(x)], "spot_diameter_um": [float(diam_um)]}
    )


def test_footprint_constant_qki_equals_that_value():
    """QKI constant everywhere -> footprint mean == that constant regardless of
    the footprint's shape; area > 1 px and the union mask matches the area."""
    H = W = 41
    rna1 = _gauss(H, W, 20, 20, sigma=2.0, amp=4000.0, bg=100.0)
    qki = np.full((H, W), 500.0)
    fp_qki, fp_area, union = _sample_qki_at_miat_footprint(
        rna1, qki, _spots(20, 20, 0.6), 0.13, default_spot_diameter_um=0.3
    )
    assert fp_qki[0] == pytest.approx(500.0, abs=1e-9)
    assert fp_area[0] > 1
    assert int(union.sum()) == int(fp_area[0])


def test_footprint_area_scales_with_spot_size():
    """A wider MIAT punctum (larger sigma) yields a LARGER half-max footprint —
    the whole point of a size-scaling footprint vs a fixed disk. Both windows
    are sized generously (diam 1.2 um) so neither half-max region is truncated."""
    H = W = 61
    narrow = _gauss(H, W, 30, 30, sigma=1.5, amp=4000.0, bg=100.0)
    wide = _gauss(H, W, 30, 30, sigma=4.0, amp=4000.0, bg=100.0)
    qki = np.full((H, W), 500.0)
    _, area_narrow, _ = _sample_qki_at_miat_footprint(
        narrow, qki, _spots(30, 30, 1.2), 0.13, default_spot_diameter_um=0.3
    )
    _, area_wide, _ = _sample_qki_at_miat_footprint(
        wide, qki, _spots(30, 30, 1.2), 0.13, default_spot_diameter_um=0.3
    )
    assert area_wide[0] > area_narrow[0]
    # sanity: the wide footprint is meaningfully bigger (FWHM area ~ sigma^2).
    assert area_wide[0] > 2.0 * area_narrow[0]


def test_footprint_samples_qki_at_the_spot_location():
    """QKI = a horizontal gradient (value = x*10). A footprint centered at x=20
    is ~symmetric in x, so its mean QKI == the gradient value at the spot."""
    H = W = 41
    rna1 = _gauss(H, W, 20, 20, sigma=2.0, amp=4000.0, bg=100.0)
    xx = np.tile(np.arange(W, dtype=np.float64) * 10.0, (H, 1))  # qki[y, x] = x*10
    fp_qki, _, _ = _sample_qki_at_miat_footprint(
        rna1, xx, _spots(20, 20, 0.6), 0.13, default_spot_diameter_um=0.3
    )
    assert fp_qki[0] == pytest.approx(200.0, abs=15.0)  # ~= 10 * x_center(20)


def test_footprint_flat_crop_uses_disk_fallback():
    """A flat (no-contrast) MIAT crop has no half-max footprint -> the sampler
    falls back to a per-spot fitted-radius disk and still returns a finite QKI
    mean + area>=1 (never NaN for an in-frame spot)."""
    H = W = 41
    rna1_flat = np.full((H, W), 300.0)
    qki = np.full((H, W), 500.0)
    fp_qki, fp_area, _ = _sample_qki_at_miat_footprint(
        rna1_flat, qki, _spots(20, 20, 0.3), 0.13, default_spot_diameter_um=0.3
    )
    assert np.isfinite(fp_qki[0])
    assert fp_qki[0] == pytest.approx(500.0, abs=1e-9)
    assert fp_area[0] >= 1


def test_footprint_empty_spots_returns_empty():
    H = W = 20
    fp_qki, fp_area, union = _sample_qki_at_miat_footprint(
        np.zeros((H, W)), np.zeros((H, W)),
        pd.DataFrame({"y_px": [], "x_px": [], "spot_diameter_um": []}),
        0.13, default_spot_diameter_um=0.3,
    )
    assert fp_qki.size == 0 and fp_area.size == 0
    assert union.shape == (H, W) and not union.any()


# ===========================================================================
# (b) HELPER-LEVEL MOC / ICQ behavior
# ===========================================================================
def test_moc_icq_correlated_vs_anticorrelated():
    """Manders Overlap Coefficient R (= raw-intensity cosine overlap) and Li's
    ICQ both separate correlated from anti-correlated fields: ICQ > 0 for
    correlated, < 0 for anti-correlated; MOC higher for correlated."""
    rng = np.random.default_rng(0)
    r = np.linspace(1.0, 100.0, 600)
    a_corr = r + rng.normal(0, 2.0, r.size)
    a_anti = 101.0 - r  # anti-correlated, still strictly positive
    m_corr = compute_coloc_metrics(r, a_corr)
    m_anti = compute_coloc_metrics(r, a_anti)
    # ICQ (== coloc_icq)
    assert m_corr["li_icq"] > 0.0
    assert m_anti["li_icq"] < 0.0
    # MOC (== coloc_moc == cosine_overlap)
    assert m_corr["cosine_overlap"] > m_anti["cosine_overlap"]


# ===========================================================================
# Synthetic 3-channel stack (DAPI + rna1/MIAT spots + dense-nuclear QKI).
# (mirrors test_partner_null_coloc.py so the end-to-end path is exercised)
# ===========================================================================
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


def _partner_plane_dense_nuclear():
    from skimage.draw import disk
    img = np.random.default_rng(44).uniform(2.0, 8.0, (H, W)).astype(np.float32)
    for (cy, cx) in _nuclei_centers():
        rr, cc = disk((cy, cx), 28, shape=img.shape)
        img[rr, cc] = 800.0
    return img


def _czyx() -> np.ndarray:
    planes = [_dapi_plane(), _rna_spot_plane(), _partner_plane_dense_nuclear()]
    return np.stack([np.stack([p] * NZ, axis=0) for p in planes], axis=0).astype(np.float32)


@pytest.fixture()
def fake_img() -> ImageWrapper:
    czyx = _czyx()
    return ImageWrapper(
        path="synthetic_footprint.tif",
        bio=_FakeBio(czyx),
        scene_idx=0,
        shape=(1, 3, NZ, H, W),
        channel_names=["DAPI", "RNA", "PART"],
        voxel_xy_nm=130.0,
        voxel_z_nm=300.0,
        n_channels=3,
        n_z=NZ,
    )


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


def _run(cfg, img, monkeypatch):
    monkeypatch.setattr(_io, "read_image", lambda p: img)
    return _rna_rna.run_one(Path(img.path), condition="cond", sec_only=False, cfg=cfg)


# ===========================================================================
# (c) END-TO-END through run_one
# ===========================================================================
def test_end_to_end_footprint_columns_present(fake_img, monkeypatch):
    cfg = _base_cfg()
    cfg.foci.compute_partner_intensity = True
    cfg.foci.compute_footprint_enrichment = True
    res = _run(cfg, fake_img, monkeypatch)

    # Per-spot columns.
    for col in ("qki_at_miat_footprint", "miat_footprint_area_px", "qki_footprint_enrichment"):
        assert col in res.spots.columns
    # Per-nucleus columns.
    for col in ("qki_assoc_ratio_continuous", "coloc_moc", "coloc_icq",
                "qki_at_miat_foci_enrichment"):
        assert col in res.nuclei.columns
    # Per-image rollups.
    for key in ("mean_qki_assoc_ratio_continuous", "mean_coloc_moc",
                "mean_coloc_icq", "mean_qki_at_miat_foci_enrichment"):
        assert key in res.per_image

    # rna1 spots carry finite footprint values; rna2 rows are NaN (footprints
    # are defined on MIAT/rna1 only) but the column still exists.
    sp = res.spots
    rna1 = sp[sp["channel"] == "rna1"]
    rna2 = sp[sp["channel"] == "rna2"]
    assert np.isfinite(pd.to_numeric(rna1["qki_at_miat_footprint"], errors="coerce")).any()
    assert (pd.to_numeric(rna1["miat_footprint_area_px"], errors="coerce") > 0).any()
    if len(rna2) > 0:
        assert pd.to_numeric(rna2["qki_at_miat_footprint"], errors="coerce").isna().all()


def test_end_to_end_enrichment_normalization_exact(fake_img, monkeypatch):
    """qki_footprint_enrichment == qki_at_miat_footprint / that spot's-nucleus
    mean QKI (rna2_nuclear_mean), to float precision."""
    cfg = _base_cfg()
    cfg.foci.compute_partner_intensity = True
    cfg.foci.compute_footprint_enrichment = True
    res = _run(cfg, fake_img, monkeypatch)

    sp = res.spots
    rna1 = sp[(sp["channel"] == "rna1") & (sp["nucleus_id"] > 0)].copy()
    assert len(rna1) > 0
    nuc_mean = res.nuclei.set_index("nucleus_id")["rna2_nuclear_mean"].to_dict()
    fp = pd.to_numeric(rna1["qki_at_miat_footprint"], errors="coerce").to_numpy()
    enr = pd.to_numeric(rna1["qki_footprint_enrichment"], errors="coerce").to_numpy()
    denom = rna1["nucleus_id"].map(nuc_mean).to_numpy(dtype=float)
    expected = fp / denom
    finite = np.isfinite(enr) & np.isfinite(expected)
    assert finite.any()
    np.testing.assert_allclose(enr[finite], expected[finite], rtol=1e-9, atol=1e-9)


def test_end_to_end_continuous_ratio_is_mean_enrichment(fake_img, monkeypatch):
    """Per-nucleus qki_assoc_ratio_continuous == mean of the nucleus's per-spot
    qki_footprint_enrichment."""
    cfg = _base_cfg()
    cfg.foci.compute_partner_intensity = True
    cfg.foci.compute_footprint_enrichment = True
    res = _run(cfg, fake_img, monkeypatch)

    sp = res.spots
    rna1 = sp[sp["channel"] == "rna1"].copy()
    per_nuc = (
        pd.to_numeric(rna1["qki_footprint_enrichment"], errors="coerce")
        .groupby(rna1["nucleus_id"]).mean()
    )
    checked = 0
    for _, nr in res.nuclei.iterrows():
        nid = int(nr["nucleus_id"])
        cont = nr["qki_assoc_ratio_continuous"]
        if nid in per_nuc.index and np.isfinite(per_nuc.loc[nid]):
            assert float(cont) == pytest.approx(float(per_nuc.loc[nid]), rel=1e-9, abs=1e-9)
            checked += 1
    assert checked >= 1


def test_moc_icq_alias_existing_threshold_free_columns(fake_img, monkeypatch):
    """coloc_moc / coloc_icq surface the ALREADY-computed threshold-free
    cosine-overlap (Manders R) + Li ICQ, so they equal the legacy columns."""
    cfg = _base_cfg()
    cfg.foci.compute_partner_intensity = True
    cfg.foci.compute_footprint_enrichment = True
    res = _run(cfg, fake_img, monkeypatch)
    nuc = res.nuclei
    np.testing.assert_allclose(
        pd.to_numeric(nuc["coloc_moc"], errors="coerce"),
        pd.to_numeric(nuc["coloc_cosine_overlap_rna1_rna2"], errors="coerce"),
        rtol=1e-12, atol=1e-12,
    )
    np.testing.assert_allclose(
        pd.to_numeric(nuc["coloc_icq"], errors="coerce"),
        pd.to_numeric(nuc["coloc_li_icq_rna1_rna2"], errors="coerce"),
        rtol=1e-12, atol=1e-12,
    )


def test_floor_gated_ratio_le_continuous_and_named(fake_img, monkeypatch):
    """With a QKI floor set, the floor-gated association column is emitted with
    the floor in its name and is <= the continuous ratio per nucleus (below-floor
    spots contribute 0). Without a floor, no gated column exists."""
    # No floor -> no gated column.
    cfg_nofloor = _base_cfg()
    cfg_nofloor.foci.compute_partner_intensity = True
    cfg_nofloor.foci.compute_footprint_enrichment = True
    res_nf = _run(cfg_nofloor, fake_img, monkeypatch)
    assert not any(c.startswith("qki_assoc_ratio_gated_") for c in res_nf.nuclei.columns)

    # Floor set high enough to zero some spots (nucleoplasm QKI ~800).
    cfg = _base_cfg()
    cfg.foci.compute_partner_intensity = True
    cfg.foci.compute_footprint_enrichment = True
    cfg.foci.assoc_qki_floor = 810.0
    res = _run(cfg, fake_img, monkeypatch)
    gated_cols = [c for c in res.nuclei.columns if c.startswith("qki_assoc_ratio_gated_")]
    assert gated_cols == ["qki_assoc_ratio_gated_810"]
    gcol = gated_cols[0]
    cont = pd.to_numeric(res.nuclei["qki_assoc_ratio_continuous"], errors="coerce")
    gated = pd.to_numeric(res.nuclei[gcol], errors="coerce")
    both = cont.notna() & gated.notna()
    assert both.any()
    # gated <= continuous (within fp tolerance) for every nucleus.
    assert (gated[both] <= cont[both] + 1e-9).all()
    # per-image rollup for the gated variant is present + named.
    assert f"mean_{gcol}" in res.per_image


def test_end_to_end_footprint_deterministic(fake_img, monkeypatch):
    """Footprint sampling is RNG-free -> two runs give identical spot metrics."""
    cfg = _base_cfg()
    cfg.foci.compute_partner_intensity = True
    cfg.foci.compute_footprint_enrichment = True
    r1 = _run(cfg, fake_img, monkeypatch)
    r2 = _run(cfg, fake_img, monkeypatch)
    for col in ("qki_at_miat_footprint", "miat_footprint_area_px", "qki_footprint_enrichment"):
        np.testing.assert_array_equal(
            pd.to_numeric(r1.spots[col], errors="coerce").fillna(-999).to_numpy(),
            pd.to_numeric(r2.spots[col], errors="coerce").fillna(-999).to_numpy(),
        )


# ===========================================================================
# (d) DEFAULTS-OFF byte-equivalence
# ===========================================================================
def test_defaults_off_no_footprint_columns(fake_img, monkeypatch):
    """compute_footprint_enrichment OFF (default) -> none of the new columns /
    per-image keys are emitted, even with partner-intensity ON."""
    cfg = _base_cfg()
    cfg.foci.compute_partner_intensity = True  # footprint OFF (default)
    res = _run(cfg, fake_img, monkeypatch)
    for col in ("qki_at_miat_footprint", "miat_footprint_area_px",
                "qki_footprint_enrichment"):
        assert col not in res.spots.columns
    for col in ("qki_assoc_ratio_continuous", "coloc_moc", "coloc_icq",
                "qki_at_miat_foci_enrichment"):
        assert col not in res.nuclei.columns
    for key in ("mean_qki_assoc_ratio_continuous", "mean_coloc_moc",
                "mean_coloc_icq", "mean_qki_at_miat_foci_enrichment"):
        assert key not in res.per_image


def test_defaults_off_byte_equivalent(fake_img, monkeypatch):
    """A footprint-OFF run equals a run on a config that never knew the feature:
    identical per_image keys/values + identical nuclei/spots frames."""
    cfg_ref = _base_cfg()
    cfg_ref.foci.compute_partner_intensity = True
    res_ref = _run(cfg_ref, fake_img, monkeypatch)

    cfg_off = _base_cfg()
    cfg_off.foci.compute_partner_intensity = True
    cfg_off.foci.compute_footprint_enrichment = False
    res_off = _run(cfg_off, fake_img, monkeypatch)

    assert set(res_ref.per_image.keys()) == set(res_off.per_image.keys())
    for k in res_ref.per_image:
        if k == "runtime_s":
            continue
        a, b = res_ref.per_image[k], res_off.per_image[k]
        if isinstance(a, float) and a != a:
            assert isinstance(b, float) and b != b, k
        else:
            assert a == b, f"per_image[{k}] differs: {a!r} != {b!r}"

    assert list(res_ref.nuclei.columns) == list(res_off.nuclei.columns)
    pd.testing.assert_frame_equal(
        res_ref.nuclei.reset_index(drop=True),
        res_off.nuclei.reset_index(drop=True),
        check_dtype=False,
    )
    assert list(res_ref.spots.columns) == list(res_off.spots.columns)
    # No footprint columns leaked.
    assert not any("footprint" in c for c in res_off.spots.columns)
    assert not any("assoc_ratio" in c for c in res_off.nuclei.columns)
