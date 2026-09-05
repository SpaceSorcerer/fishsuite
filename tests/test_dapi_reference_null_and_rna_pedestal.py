"""DAPI-plane reference for the partner-anchored null + per-image rna1 pedestal
normalisation (Brian, 2026-09-04).

Two additive, default-OFF engine features:

(a) ``FociCfg.partner_anchored_null_sample_field`` — the partner-anchored
    rotation null can sample the DAPI plane instead of (or as well as) the rna1
    plane. DAPI carries no RNA-probe signal, so the same estimator run on it is
    a SIGNAL-FREE REFERENCE for what nuclear texture alone produces. Motivated
    by the QKI x BIN1-intron adversarial review finding that the enrichment
    function returned 1.16-1.20 on an antigen-free antibody channel in QKI-KO.

(b) ``FociCfg.rna_pedestal_normalize`` — the rna1 plane is scaled by
    ``batch_reference / this image's in-nucleus median`` BEFORE LoG detection,
    so one absolute ``threshold_override`` means the same thing in every field.
    Only detection sees the scaled plane; every emitted intensity stays raw.

All fixtures are GPU-free synthetic stacks, reusing the harness idiom of
``test_partner_anchored_rotation_null.py``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import gaussian_filter

from fishsuite.config.schema import FishsuiteConfig, FociCfg
from fishsuite.core import io as _io
from fishsuite.core import spots as _spots_mod
from fishsuite.core.io import ImageWrapper
from fishsuite.core.modes import rna_rna as _rna_rna
from fishsuite.core.modes.rna_protein import _relabel_rna2_to_protein
from fishsuite.runner import resolve_rna_pedestal


# ===========================================================================
# SCHEMA
# ===========================================================================
def test_sample_field_defaults_to_rna():
    assert FociCfg().partner_anchored_null_sample_field == "rna"


@pytest.mark.parametrize("val", ["rna", "dapi", "both"])
def test_sample_field_accepts_the_three_options(val):
    assert FociCfg(partner_anchored_null_sample_field=val
                   ).partner_anchored_null_sample_field == val


def test_sample_field_rejects_an_unknown_channel():
    with pytest.raises(Exception):
        FociCfg(partner_anchored_null_sample_field="green")


def test_pedestal_flags_default_off():
    f = FociCfg()
    assert f.rna_pedestal_normalize is False
    assert f.rna_pedestal_stat == "nuclear_median"


def test_pedestal_stat_rejects_an_unimplemented_statistic():
    with pytest.raises(Exception):
        FociCfg(rna_pedestal_stat="nuclear_mean")


def test_new_foci_fields_round_trip_through_a_yaml_dict():
    cfg = FishsuiteConfig.model_validate(
        {
            "foci": {
                "partner_anchored_null_sample_field": "both",
                "rna_pedestal_normalize": True,
                "rna_pedestal_stat": "nuclear_median",
            }
        }
    )
    assert cfg.foci.partner_anchored_null_sample_field == "both"
    assert cfg.foci.rna_pedestal_normalize is True
    assert cfg.foci.rna_pedestal_stat == "nuclear_median"


# ===========================================================================
# RELABEL: the DAPI-family names must become *_at_protein_spots in rna_protein.
# ===========================================================================
DAPI_NUCLEUS_COLS = [
    "dapi_rotation_enrichment_at_rna2_spots",
    "dapi_rotation_null_z_at_rna2_spots",
    "dapi_rotation_null_p_at_rna2_spots",
    "dapi_rotation_assoc_fraction_at_rna2_spots",
    "rotation_null_usable_dapi_at_rna2_spots",
]
DAPI_IMAGE_COLS = [
    "dapi_pooled_rotation_enrichment_at_rna2_spots",
    "dapi_pooled_rotation_null_z_at_rna2_spots",
    "dapi_pooled_rotation_null_p_empirical_at_rna2_spots",
    "dapi_pooled_rotation_obs_at_rna2_spots",
    "dapi_pooled_rotation_null_mean_at_rna2_spots",
    "dapi_mean_rotation_assoc_fraction_at_rna2_spots",
    "n_nuclei_partner_rotation_null_dapi_at_rna2_spots",
]
RNA1_NUCLEUS_COLS = [
    "rna1_rotation_enrichment_at_rna2_spots",
    "rna1_rotation_null_z_at_rna2_spots",
    "rna1_rotation_null_p_at_rna2_spots",
    "rna1_rotation_assoc_fraction_at_rna2_spots",
    "rotation_null_usable_at_rna2_spots",
]
RNA1_IMAGE_COLS = [
    "rna1_pooled_rotation_enrichment_at_rna2_spots",
    "rna1_pooled_rotation_null_z_at_rna2_spots",
    "rna1_pooled_rotation_null_p_empirical_at_rna2_spots",
    "rna1_mean_rotation_assoc_fraction_at_rna2_spots",
    "n_nuclei_partner_rotation_null_at_rna2_spots",
]


@pytest.mark.parametrize("name", DAPI_NUCLEUS_COLS + DAPI_IMAGE_COLS)
def test_dapi_columns_relabel_to_protein_spots(name):
    out = _relabel_rna2_to_protein(name)
    assert out.endswith("_at_protein_spots")
    assert "rna2" not in out
    # The ``dapi`` token names the SAMPLED FIELD and must survive the relabel.
    assert "dapi" in out


def test_dapi_usable_flag_relabels_to_the_documented_name():
    assert (
        _relabel_rna2_to_protein("rotation_null_usable_dapi_at_rna2_spots")
        == "rotation_null_usable_dapi_at_protein_spots"
    )


# ===========================================================================
# Synthetic 3-channel stack (GPU-free).
# ===========================================================================
DAPI_C, RNA_C, PART_C = 0, 1, 2
NZ = 4
EH = EW = 200


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

    img = np.random.default_rng(11).uniform(0.0, 20.0, (EH, EW)).astype(np.float32)
    for (cy, cx) in _nuclei_centers():
        rr, cc = disk((cy, cx), 28, shape=img.shape)
        img[rr, cc] += 3000.0
    return img


def _spot_positions():
    pos = {}
    for i, (cy, cx) in enumerate(_nuclei_centers(), start=1):
        pts = []
        for k in range(10):
            ang = 2 * np.pi * k / 10
            pts.append((int(cy + 14 * np.sin(ang)), int(cx + 14 * np.cos(ang))))
        pos[i] = pts
    return pos


def _spot_plane(seed_bg, seed_amp):
    img = np.random.default_rng(seed_bg).uniform(2.0, 8.0, (EH, EW)).astype(np.float32)
    blob = np.zeros_like(img)
    rng = np.random.default_rng(seed_amp)
    for _nid, pts in _spot_positions().items():
        for (y, x) in pts:
            blob[y, x] += float(rng.uniform(3000.0, 6000.0))
    return img + gaussian_filter(blob, 1.1)


def _czyx() -> np.ndarray:
    planes = [_dapi_plane(), _spot_plane(22, 33), _spot_plane(44, 55)]
    return np.stack(
        [np.stack([p] * NZ, axis=0) for p in planes], axis=0
    ).astype(np.float32)


def _spot_plane_with_outside(seed_bg, seed_amp):
    """rna2 plane whose constellation ALSO carries spots outside every nucleus.

    The default fixture places every spot at radius 14 px inside a 28 px
    nucleus, so nothing there can exercise an out-of-nucleus anchor.
    """
    img = _spot_plane(seed_bg, seed_amp)
    blob = np.zeros_like(img)
    rng = np.random.default_rng(seed_amp + 7)
    for (cy, cx) in _nuclei_centers():
        for k in range(6):
            ang = 2 * np.pi * k / 6
            y = int(np.clip(cy + 42 * np.sin(ang), 1, EH - 2))
            x = int(np.clip(cx + 42 * np.cos(ang), 1, EW - 2))
            blob[y, x] += float(rng.uniform(3000.0, 6000.0))
    return img + gaussian_filter(blob, 1.1)


@pytest.fixture()
def fake_img_outside() -> ImageWrapper:
    planes = [_dapi_plane(), _spot_plane(22, 33), _spot_plane_with_outside(44, 55)]
    czyx = np.stack(
        [np.stack([p] * NZ, axis=0) for p in planes], axis=0
    ).astype(np.float32)
    return ImageWrapper(
        path="synthetic_outside_anchors.tif",
        bio=_FakeBio(czyx),
        scene_idx=0,
        shape=(1, 3, NZ, EH, EW),
        channel_names=["DAPI", "RNA", "PART"],
        voxel_xy_nm=130.0,
        voxel_z_nm=300.0,
        n_channels=3,
        n_z=NZ,
    )


@pytest.fixture()
def fake_img() -> ImageWrapper:
    return ImageWrapper(
        path="synthetic_dapi_ref_pedestal.tif",
        bio=_FakeBio(_czyx()),
        scene_idx=0,
        shape=(1, 3, NZ, EH, EW),
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


def _anchored_cfg(n=200) -> FishsuiteConfig:
    cfg = _base_cfg()
    cfg.foci.compute_partner_intensity = True
    cfg.foci.compute_partner_rotation_null = True
    cfg.foci.partner_rotation_n = n
    cfg.foci.partner_rotation_seed = 0
    cfg.foci.compute_partner_anchored_rotation_null = True
    return cfg


def _run(cfg, img, monkeypatch, **kw):
    monkeypatch.setattr(_io, "read_image", lambda p: img)
    return _rna_rna.run_one(
        Path(img.path), condition="cond", sec_only=False, cfg=cfg, **kw
    )


# ===========================================================================
# (a) DAPI-PLANE REFERENCE — column sets.
# ===========================================================================
def test_default_field_emits_only_the_rna1_family(fake_img, monkeypatch):
    res = _run(_anchored_cfg(), fake_img, monkeypatch)
    for c in RNA1_NUCLEUS_COLS:
        assert c in res.nuclei.columns
    for c in DAPI_NUCLEUS_COLS:
        assert c not in res.nuclei.columns
    for c in RNA1_IMAGE_COLS:
        assert c in res.per_image
    for c in DAPI_IMAGE_COLS:
        assert c not in res.per_image


def test_dapi_field_emits_only_the_dapi_family(fake_img, monkeypatch):
    cfg = _anchored_cfg()
    cfg.foci.partner_anchored_null_sample_field = "dapi"
    res = _run(cfg, fake_img, monkeypatch)
    for c in DAPI_NUCLEUS_COLS:
        assert c in res.nuclei.columns
    for c in RNA1_NUCLEUS_COLS:
        assert c not in res.nuclei.columns
    for c in DAPI_IMAGE_COLS:
        assert c in res.per_image


def test_both_field_emits_both_families(fake_img, monkeypatch):
    cfg = _anchored_cfg()
    cfg.foci.partner_anchored_null_sample_field = "both"
    res = _run(cfg, fake_img, monkeypatch)
    for c in RNA1_NUCLEUS_COLS + DAPI_NUCLEUS_COLS:
        assert c in res.nuclei.columns
    for c in RNA1_IMAGE_COLS + DAPI_IMAGE_COLS:
        assert c in res.per_image


def test_dapi_family_is_populated_not_all_nan(fake_img, monkeypatch):
    cfg = _anchored_cfg()
    cfg.foci.partner_anchored_null_sample_field = "dapi"
    res = _run(cfg, fake_img, monkeypatch)
    n2 = pd.to_numeric(res.nuclei["n_spots_rna2"], errors="coerce").fillna(0)
    enr = pd.to_numeric(
        res.nuclei["dapi_rotation_enrichment_at_rna2_spots"], errors="coerce"
    )
    assert (n2 > 0).any()
    assert np.isfinite(enr[n2 > 0]).all()
    assert np.isfinite(res.per_image["dapi_pooled_rotation_enrichment_at_rna2_spots"])
    assert res.per_image["n_nuclei_partner_rotation_null_dapi_at_rna2_spots"] > 0


# ===========================================================================
# (a) The DAPI family is the SAME function on the DAPI PLANE — proven, not asserted.
# ===========================================================================
def _spy_rotation_fields(monkeypatch):
    """Record the sampled-field array of every ``_rotation_null_for_nucleus`` call."""
    seen = []
    real = _rna_rna._rotation_null_for_nucleus

    def _spy(field, ys, xs, mask, centroid, dy, dx, n, rng, **kw):
        seen.append(np.asarray(field).copy())
        return real(field, ys, xs, mask, centroid, dy, dx, n, rng, **kw)

    monkeypatch.setattr(_rna_rna, "_rotation_null_for_nucleus", _spy)
    return seen


def _matches_any(arrays, ref) -> bool:
    return any(
        a.shape == ref.shape and np.array_equal(a, ref.astype(a.dtype))
        for a in arrays
    )


def test_dapi_field_hands_the_dapi_plane_to_the_rotation_function(
    fake_img, monkeypatch
):
    cfg = _anchored_cfg()
    cfg.foci.partner_anchored_null_sample_field = "dapi"
    seen = _spy_rotation_fields(monkeypatch)
    _run(cfg, fake_img, monkeypatch)

    dapi_ref = _dapi_plane().astype(np.float64)
    rna_ref = _spot_plane(22, 33).astype(np.float64)
    assert _matches_any(seen, dapi_ref), "no call sampled the DAPI plane"
    # The rna1 plane is NOT sampled by the partner-anchored null in this mode.
    assert not _matches_any(seen, rna_ref)


def test_rna_field_never_hands_over_the_dapi_plane(fake_img, monkeypatch):
    cfg = _anchored_cfg()
    seen = _spy_rotation_fields(monkeypatch)
    _run(cfg, fake_img, monkeypatch)
    assert not _matches_any(seen, _dapi_plane().astype(np.float64))


# ===========================================================================
# (a) Turning on the DAPI family must not perturb the rna1 draws.
# ===========================================================================
def test_both_reproduces_the_rna_only_rna1_numbers_bit_for_bit(
    fake_img, monkeypatch
):
    ref = _run(_anchored_cfg(), fake_img, monkeypatch)
    cfg = _anchored_cfg()
    cfg.foci.partner_anchored_null_sample_field = "both"
    new = _run(cfg, fake_img, monkeypatch)

    shared = [c for c in ref.nuclei.columns if c in new.nuclei.columns]
    assert set(ref.nuclei.columns) == set(shared)
    pd.testing.assert_frame_equal(
        ref.nuclei[shared].reset_index(drop=True),
        new.nuclei[shared].reset_index(drop=True),
        check_dtype=False,
    )
    for k, v in ref.per_image.items():
        if k == "runtime_s":
            continue
        nv = new.per_image[k]
        if isinstance(v, float) and v != v:
            assert isinstance(nv, float) and nv != nv, k
        else:
            assert nv == v, k


# ===========================================================================
# (b) PEDESTAL: batch-reference arithmetic.
# ===========================================================================
def test_pedestal_reference_is_the_median_of_the_biological_medians():
    meds = {"a": 400.0, "b": 600.0, "c": 1000.0, "sec": 50.0}
    ref, factors, n_ref, fell_back = resolve_rna_pedestal(
        meds, ["a", "b", "c"]
    )
    assert ref == pytest.approx(600.0)
    assert n_ref == 3
    assert fell_back is False
    # Every image, INCLUDING the secondary-only field, receives a factor.
    assert set(factors) == {"a", "b", "c", "sec"}
    assert factors["a"] == pytest.approx(600.0 / 400.0)
    assert factors["b"] == pytest.approx(1.0)
    assert factors["c"] == pytest.approx(600.0 / 1000.0)
    assert factors["sec"] == pytest.approx(600.0 / 50.0)


def test_pedestal_reference_excludes_secondary_only_images():
    meds = {"a": 400.0, "b": 600.0, "sec1": 10.0, "sec2": 12.0, "sec3": 14.0}
    ref, _f, n_ref, _fb = resolve_rna_pedestal(meds, ["a", "b"])
    assert ref == pytest.approx(500.0)  # median(400, 600), sec-only ignored
    assert n_ref == 2


def test_pedestal_reference_falls_back_and_says_so():
    meds = {"sec1": 10.0, "sec2": 20.0}
    ref, _f, n_ref, fell_back = resolve_rna_pedestal(meds, [])
    assert ref == pytest.approx(15.0)
    assert n_ref == 2
    assert fell_back is True


def test_pedestal_reference_is_none_without_any_median():
    ref, factors, n_ref, fell_back = resolve_rna_pedestal({}, ["a"])
    assert ref is None and factors == {} and n_ref == 0 and fell_back is False


def test_pedestal_skips_a_zero_median_rather_than_dividing_by_it():
    ref, factors, _n, _fb = resolve_rna_pedestal(
        {"a": 400.0, "b": 600.0, "dark": 0.0}, ["a", "b"]
    )
    assert "dark" not in factors
    assert ref == pytest.approx(500.0)


# ===========================================================================
# (b) PEDESTAL: what the detector actually receives.
# ===========================================================================
def _spy_detect(monkeypatch):
    """Record every array handed to ``spots.detect_spots``."""
    seen = []
    real = _spots_mod.detect_spots

    def _spy(rna, **kw):
        seen.append(np.asarray(rna).copy())
        return real(rna, **kw)

    monkeypatch.setattr(_rna_rna._spots, "detect_spots", _spy)
    return seen


def test_pedestal_off_passes_the_raw_plane(fake_img, monkeypatch):
    seen = _spy_detect(monkeypatch)
    _run(_base_cfg(), fake_img, monkeypatch)
    raw = _spot_plane(22, 33)
    assert any(np.array_equal(a, raw) for a in seen)


def test_pedestal_on_scales_only_the_rna1_plane(fake_img, monkeypatch):
    cfg = _base_cfg()
    cfg.foci.rna_pedestal_normalize = True
    seen = _spy_detect(monkeypatch)
    _run(cfg, fake_img, monkeypatch, rna_pedestal_factor=1.5)

    raw1 = _spot_plane(22, 33)
    raw2 = _spot_plane(44, 55)
    # float32 input is NOT an integer dtype, so no rounding/clipping is applied.
    expected1 = (raw1.astype(np.float64) * 1.5).astype(raw1.dtype)
    assert any(np.array_equal(a, expected1) for a in seen), "rna1 plane not scaled"
    assert any(np.array_equal(a, raw2) for a in seen), "rna2 plane must stay raw"
    assert not any(np.array_equal(a, raw1) for a in seen), "raw rna1 still detected on"


def test_pedestal_rounds_and_clips_for_an_integer_dtype(monkeypatch):
    """An integer input plane keeps its dtype, so the LoG quantisation regime is
    unchanged and only the pedestal differs between arms."""
    from fishsuite.core.io import ImageWrapper as _IW

    planes = [
        _dapi_plane().astype(np.uint16),
        (_spot_plane(22, 33) * 6.0).astype(np.uint16),
        _spot_plane(44, 55).astype(np.uint16),
    ]
    czyx = np.stack([np.stack([p] * NZ, axis=0) for p in planes], axis=0)
    img = _IW(
        path="synthetic_uint16.tif",
        bio=_FakeBio(czyx),
        scene_idx=0,
        shape=(1, 3, NZ, EH, EW),
        channel_names=["DAPI", "RNA", "PART"],
        voxel_xy_nm=130.0,
        voxel_z_nm=300.0,
        n_channels=3,
        n_z=NZ,
    )
    cfg = _base_cfg()
    cfg.foci.rna_pedestal_normalize = True
    seen = _spy_detect(monkeypatch)
    res = _run(cfg, img, monkeypatch, rna_pedestal_factor=3.7)

    raw1 = planes[1]
    expected = np.clip(
        np.rint(raw1.astype(np.float64) * 3.7), 0, np.iinfo(np.uint16).max
    ).astype(np.uint16)
    assert any(
        a.dtype == np.uint16 and np.array_equal(a, expected) for a in seen
    )
    assert res.thresholds["rna_pedestal_clipped_px"] == int(
        np.count_nonzero(raw1.astype(np.float64) * 3.7 > np.iinfo(np.uint16).max)
    )


# ===========================================================================
# (b) PEDESTAL: emitted intensities stay RAW.
# ===========================================================================
def test_spot_peak_intensity_stays_raw_and_normalized_is_emitted_alongside(
    fake_img, monkeypatch
):
    cfg = _base_cfg()
    cfg.foci.rna_pedestal_normalize = True
    factor = 2.25
    res = _run(cfg, fake_img, monkeypatch, rna_pedestal_factor=factor)

    raw = _spot_plane(22, 33)
    sp = res.spots[res.spots["channel"] == "rna1"]
    assert len(sp) > 0
    assert "intensity_peak_normalized" in res.spots.columns

    ys = sp["y_px"].astype(int).to_numpy()
    xs = sp["x_px"].astype(int).to_numpy()
    got = sp["spot_peak_intensity"].astype(float).to_numpy()
    assert np.allclose(got, raw[ys, xs].astype(float), rtol=0, atol=1e-4)

    norm = sp["intensity_peak_normalized"].astype(float).to_numpy()
    expected_norm = (raw.astype(np.float64) * factor).astype(raw.dtype)[ys, xs]
    assert np.allclose(norm, expected_norm.astype(float), rtol=0, atol=1e-4)
    # The normalised column is the scaled one, i.e. genuinely different data.
    assert not np.allclose(norm, got)


def test_rna2_rows_carry_nan_in_the_normalized_column(fake_img, monkeypatch):
    cfg = _base_cfg()
    cfg.foci.rna_pedestal_normalize = True
    res = _run(cfg, fake_img, monkeypatch, rna_pedestal_factor=1.5)
    sp2 = res.spots[res.spots["channel"] == "rna2"]
    assert len(sp2) > 0
    assert sp2["intensity_peak_normalized"].isna().all()


# ===========================================================================
# (b) PEDESTAL: provenance and the missing-factor path.
# ===========================================================================
def test_pedestal_off_adds_no_columns_anywhere(fake_img, monkeypatch):
    res = _run(_base_cfg(), fake_img, monkeypatch)
    for k in (
        "rna_pedestal_normalize", "rna_pedestal_stat", "rna_pedestal_applied",
        "rna_pedestal_factor", "rna_pedestal_clipped_px",
    ):
        assert k not in res.thresholds
    assert "intensity_peak_normalized" not in res.spots.columns


def test_pedestal_records_the_factor_it_applied(fake_img, monkeypatch):
    cfg = _base_cfg()
    cfg.foci.rna_pedestal_normalize = True
    res = _run(cfg, fake_img, monkeypatch, rna_pedestal_factor=1.75)
    assert res.thresholds["rna_pedestal_normalize"] is True
    assert res.thresholds["rna_pedestal_stat"] == "nuclear_median"
    assert res.thresholds["rna_pedestal_applied"] is True
    assert res.thresholds["rna_pedestal_factor"] == pytest.approx(1.75)


def test_pedestal_on_without_a_factor_says_so_and_runs_unnormalised(
    fake_img, monkeypatch, capsys
):
    cfg = _base_cfg()
    cfg.foci.rna_pedestal_normalize = True
    seen = _spy_detect(monkeypatch)
    res = _run(cfg, fake_img, monkeypatch)  # no rna_pedestal_factor
    out = capsys.readouterr().out
    assert "rna_pedestal_normalize is ON but no" in out
    assert res.thresholds["rna_pedestal_applied"] is False
    assert res.thresholds["rna_pedestal_factor"] != res.thresholds["rna_pedestal_factor"]
    assert any(np.array_equal(a, _spot_plane(22, 33)) for a in seen)
    # The column set must NOT drift between images just because one lacked a
    # factor, so the normalised column is still present (all NaN).
    assert "intensity_peak_normalized" in res.spots.columns


def test_factor_of_one_reproduces_the_unnormalised_spot_positions(
    fake_img, monkeypatch
):
    """A factor of exactly 1.0 is a mathematical no-op; the spots must match."""
    ref = _run(_base_cfg(), fake_img, monkeypatch)
    cfg = _base_cfg()
    cfg.foci.rna_pedestal_normalize = True
    new = _run(cfg, fake_img, monkeypatch, rna_pedestal_factor=1.0)
    a = ref.spots[ref.spots["channel"] == "rna1"][["x_px", "y_px"]].reset_index(drop=True)
    b = new.spots[new.spots["channel"] == "rna1"][["x_px", "y_px"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


# ===========================================================================
# (c) IN-NUCLEUS ANCHOR RESTRICTION for the partner-anchored null.
# ===========================================================================
def test_nuclear_anchors_flag_defaults_off():
    assert FociCfg().partner_anchored_null_nuclear_anchors_only is False


def test_only_nuclear_spots_does_not_filter_the_spot_table(fake_img_outside, monkeypatch):
    """DOCUMENTS A FOOTGUN, and the reason the new flag exists.

    ``FociChannelOverrideCfg.only_nuclear_spots`` is resolved and written to
    thresholds.csv but never filters the spot table in rna_rna / rna_protein,
    so a preset that sets it asserts a restriction the engine does not apply.
    """
    from fishsuite.config.schema import FociChannelOverrideCfg

    cfg = _base_cfg()
    cfg.foci.rna2_overrides = FociChannelOverrideCfg(only_nuclear_spots=True)
    res = _run(cfg, fake_img_outside, monkeypatch)
    assert res.thresholds["rna2_only_nuclear_spots"] is True
    sp2 = res.spots[res.spots["channel"] == "rna2"]
    assert (sp2["in_nucleus"].astype(int) == 0).any(), (
        "if this ever fails, only_nuclear_spots started filtering and this note "
        "and the new flag should be revisited"
    )


def test_nuclear_anchor_restriction_drops_out_of_nucleus_anchors(
    fake_img, monkeypatch
):
    cfg = _anchored_cfg()
    cfg.foci.partner_anchored_null_nuclear_anchors_only = True
    res = _run(cfg, fake_img, monkeypatch)

    assert "n_partner_anchors_at_rna2_spots" in res.nuclei.columns
    used = pd.to_numeric(
        res.nuclei["n_partner_anchors_at_rna2_spots"], errors="coerce"
    ).fillna(0)
    # Every anchor the null used must be an in-nucleus rna2 spot of that nucleus.
    sp2 = res.spots[res.spots["channel"] == "rna2"]
    nuc_only = (
        sp2[sp2["in_nucleus"].astype(bool)]
        .groupby("nucleus_id").size()
    )
    for _, row in res.nuclei.iterrows():
        nid = int(row["nucleus_id"])
        assert used.loc[row.name] <= int(nuc_only.get(nid, 0))


def test_nuclear_anchor_restriction_changes_the_enrichment_it_is_meant_to(
    fake_img_outside, monkeypatch
):
    """The restriction must actually bite: with out-of-nucleus anchors present,
    restricting the constellation has to move the enrichment."""
    ref = _run(_anchored_cfg(), fake_img_outside, monkeypatch)
    cfg = _anchored_cfg()
    cfg.foci.partner_anchored_null_nuclear_anchors_only = True
    new = _run(cfg, fake_img_outside, monkeypatch)

    sp2 = ref.spots[ref.spots["channel"] == "rna2"]
    n_out = int((~sp2["in_nucleus"].astype(bool)).sum())
    a = pd.to_numeric(
        ref.nuclei["rna1_rotation_enrichment_at_rna2_spots"], errors="coerce"
    )
    b = pd.to_numeric(
        new.nuclei["rna1_rotation_enrichment_at_rna2_spots"], errors="coerce"
    )
    assert n_out > 0, "fixture must carry out-of-nucleus rna2 spots"
    assert not np.allclose(
        a.to_numpy(), b.to_numpy(), rtol=0, atol=0, equal_nan=True
    )


def test_nuclear_anchor_restriction_off_adds_no_column(fake_img, monkeypatch):
    res = _run(_anchored_cfg(), fake_img, monkeypatch)
    assert "n_partner_anchors_at_rna2_spots" not in res.nuclei.columns


@pytest.mark.parametrize(
    "name", ["n_partner_anchors_at_rna2_spots"]
)
def test_anchor_count_column_relabels_to_protein(name):
    assert _relabel_rna2_to_protein(name) == "n_partner_anchors_at_protein_spots"


# ===========================================================================
# (d) OPTIONAL CLAMP on the per-image pedestal factor.
# ===========================================================================
def test_factor_max_defaults_to_none():
    assert FociCfg().rna_pedestal_factor_max is None


def test_factor_max_round_trips_through_a_yaml_dict():
    cfg = FishsuiteConfig.model_validate(
        {"foci": {"rna_pedestal_normalize": True, "rna_pedestal_factor_max": 1.0}}
    )
    assert cfg.foci.rna_pedestal_factor_max == pytest.approx(1.0)


def test_clamp_caps_a_factor_above_the_max(fake_img, monkeypatch):
    cfg = _base_cfg()
    cfg.foci.rna_pedestal_normalize = True
    cfg.foci.rna_pedestal_factor_max = 1.0
    seen = _spy_detect(monkeypatch)
    res = _run(cfg, fake_img, monkeypatch, rna_pedestal_factor=1.8)

    raw = _spot_plane(22, 33)
    # 1.8 is capped to 1.0, so the detector sees the RAW plane.
    assert any(np.array_equal(a, raw) for a in seen)
    assert not any(
        np.array_equal(a, (raw.astype(np.float64) * 1.8).astype(raw.dtype))
        for a in seen
    )
    assert res.thresholds["rna_pedestal_factor"] == pytest.approx(1.0)
    assert res.thresholds["rna_pedestal_factor_raw"] == pytest.approx(1.8)
    assert res.thresholds["rna_pedestal_factor_max"] == pytest.approx(1.0)
    assert res.thresholds["rna_pedestal_factor_clamped"] is True


def test_clamp_leaves_a_factor_below_the_max_alone(fake_img, monkeypatch):
    """The scale-DOWN direction, which is the whole point of the clamp, is kept."""
    cfg = _base_cfg()
    cfg.foci.rna_pedestal_normalize = True
    cfg.foci.rna_pedestal_factor_max = 1.0
    seen = _spy_detect(monkeypatch)
    res = _run(cfg, fake_img, monkeypatch, rna_pedestal_factor=0.6)

    raw = _spot_plane(22, 33)
    expected = (raw.astype(np.float64) * 0.6).astype(raw.dtype)
    assert any(np.array_equal(a, expected) for a in seen)
    assert res.thresholds["rna_pedestal_factor"] == pytest.approx(0.6)
    assert res.thresholds["rna_pedestal_factor_raw"] == pytest.approx(0.6)
    assert res.thresholds["rna_pedestal_factor_clamped"] is False


def test_factor_exactly_at_the_max_is_not_recorded_as_clamped(fake_img, monkeypatch):
    cfg = _base_cfg()
    cfg.foci.rna_pedestal_normalize = True
    cfg.foci.rna_pedestal_factor_max = 1.0
    res = _run(cfg, fake_img, monkeypatch, rna_pedestal_factor=1.0)
    assert res.thresholds["rna_pedestal_factor_clamped"] is False
    assert res.thresholds["rna_pedestal_factor"] == pytest.approx(1.0)


def test_unset_max_never_clamps(fake_img, monkeypatch):
    cfg = _base_cfg()
    cfg.foci.rna_pedestal_normalize = True
    res = _run(cfg, fake_img, monkeypatch, rna_pedestal_factor=1.8)
    assert res.thresholds["rna_pedestal_factor"] == pytest.approx(1.8)
    assert res.thresholds["rna_pedestal_factor_clamped"] is False
    assert res.thresholds["rna_pedestal_factor_max"] != res.thresholds["rna_pedestal_factor_max"]


def test_clamp_columns_absent_when_the_feature_is_off(fake_img, monkeypatch):
    res = _run(_base_cfg(), fake_img, monkeypatch)
    for k in ("rna_pedestal_factor_raw", "rna_pedestal_factor_max",
              "rna_pedestal_factor_clamped"):
        assert k not in res.thresholds


def test_clamped_run_matches_an_unnormalised_run_when_max_is_one(
    fake_img, monkeypatch
):
    """A cap of 1.0 with an up-scaling factor must reproduce the OFF path exactly."""
    ref = _run(_base_cfg(), fake_img, monkeypatch)
    cfg = _base_cfg()
    cfg.foci.rna_pedestal_normalize = True
    cfg.foci.rna_pedestal_factor_max = 1.0
    new = _run(cfg, fake_img, monkeypatch, rna_pedestal_factor=1.9)
    a = ref.spots[ref.spots["channel"] == "rna1"][["x_px", "y_px"]].reset_index(drop=True)
    b = new.spots[new.spots["channel"] == "rna1"][["x_px", "y_px"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
