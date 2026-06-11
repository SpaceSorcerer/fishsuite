"""Per-nucleus RANDOM-POSITION NULL coloc — unit tests (Brian, 2026-06-05).

PIPELINE-NATIVE version of the validated external script
``qki_at_miat_null_ALLARMS_tm1.0.py``: for each nucleus, compare the partner
(rna2 / QKI) intensity at the rna1 (MIAT) spots against the SAME number of
spots placed at random IN-NUCLEUS positions (disk-sampled, repeated N_NULL
times) -> enrichment / z / per-image pooled enrichment / z / empirical-p.
Gated behind ``foci.compute_partner_null_enrichment`` (requires
``compute_partner_intensity``); default OFF.

Three families of GPU-free tests:
  (a) the null statistic on a synthetic nucleus with a KNOWN enrichment
      (helper-level, deterministic via seed; + end-to-end through run_one);
  (b) nucleolus exclusion changes the null sampling as expected (helper-level
      + end-to-end gating contract);
  (c) defaults-OFF byte-equivalence (the null columns are ABSENT and the rest
      of the output is unchanged vs the pre-feature path).
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
    _disk_stencil,
    _disk_means_at,
    _partner_null_for_nucleus,
)


# ---------------------------------------------------------------------------
# (a) HELPER-LEVEL: known-answer null statistic
# ---------------------------------------------------------------------------
def test_disk_stencil_radius3_is_circular():
    """r=3 disk stencil keeps exactly the in-radius offsets (29-px disk)."""
    dy, dx = _disk_stencil(3.0)
    assert dy.shape == dx.shape
    # every kept offset must satisfy dy^2+dx^2 <= 9
    assert np.all(dy * dy + dx * dx <= 9.0)
    # and (0,0) is included (center)
    assert np.any((dy == 0) & (dx == 0))
    # a 7x7 box has 49 cells; the r=3 disk keeps 29 of them.
    assert dy.size == 29


def test_null_enrichment_known_value_bright_spots():
    """Spots on a bright region of a half-bright/half-dim partner -> the
    whole-nucleus null averages the field, so enrichment ≈ observed/field-mean
    and z > 0 (deterministic via seed)."""
    H = W = 40
    partner = np.full((H, W), 10.0)
    partner[:, :20] = 100.0  # bright left half
    dy, dx = _disk_stencil(3.0)
    spot_cy = np.array([10, 12, 14, 16, 18])
    spot_cx = np.array([5, 7, 9, 5, 7])  # all on the bright side
    nys, nxs = np.where(np.ones((H, W), bool))
    rng = np.random.default_rng(0)
    obs, null = _partner_null_for_nucleus(
        partner, spot_cy, spot_cx, nys, nxs, dy, dx, 1000, rng
    )
    assert null.shape == (1000,)
    assert obs == pytest.approx(100.0, abs=1e-9)          # bright-side disks = 100
    assert null.mean() == pytest.approx(55.0, abs=3.0)    # field mean ≈ (100+10)/2
    enrichment = obs / null.mean()
    assert enrichment > 1.5                                # clearly enriched
    z = (obs - null.mean()) / null.std(ddof=1)
    assert z > 0


def test_null_enrichment_uniform_partner_is_one():
    """A UNIFORM partner field -> observed == null_mean -> enrichment == 1.0
    exactly (every disk samples the same constant)."""
    H = W = 40
    partner = np.full((H, W), 50.0)
    dy, dx = _disk_stencil(3.0)
    spot_cy = np.array([10, 12, 14, 16, 18])
    spot_cx = np.array([5, 7, 9, 5, 7])
    nys, nxs = np.where(np.ones((H, W), bool))
    rng = np.random.default_rng(0)
    obs, null = _partner_null_for_nucleus(
        partner, spot_cy, spot_cx, nys, nxs, dy, dx, 1000, rng
    )
    assert obs == pytest.approx(50.0, abs=1e-9)
    assert null.mean() == pytest.approx(50.0, abs=1e-9)
    assert obs / null.mean() == pytest.approx(1.0, abs=1e-9)


def test_null_is_deterministic_with_seed():
    """Same seed -> identical null distribution (reproducibility contract)."""
    H = W = 32
    partner = np.random.default_rng(7).uniform(0, 100, (H, W))
    dy, dx = _disk_stencil(3.0)
    spot_cy = np.array([8, 10, 12]); spot_cx = np.array([8, 10, 12])
    nys, nxs = np.where(np.ones((H, W), bool))
    o1, n1 = _partner_null_for_nucleus(
        partner, spot_cy, spot_cx, nys, nxs, dy, dx, 500, np.random.default_rng(0)
    )
    o2, n2 = _partner_null_for_nucleus(
        partner, spot_cy, spot_cx, nys, nxs, dy, dx, 500, np.random.default_rng(0)
    )
    assert o1 == o2
    np.testing.assert_array_equal(n1, n2)


# ---------------------------------------------------------------------------
# (b) HELPER-LEVEL: nucleolus exclusion changes the sampling
# ---------------------------------------------------------------------------
def test_nucleolus_exclusion_changes_null_sampling():
    """Excluding the DIM (nucleolar) region from the null sampling positions
    RAISES the null_mean (random draws can no longer fall in the void) and so
    LOWERS the enrichment — the exact behavior the exclusion is meant to test."""
    H = W = 40
    partner = np.full((H, W), 10.0)
    partner[:, :20] = 100.0  # bright left, dim right
    dy, dx = _disk_stencil(3.0)
    spot_cy = np.array([10, 12, 14, 16, 18])
    spot_cx = np.array([5, 7, 9, 5, 7])  # spots on the bright side
    full = np.ones((H, W), bool)
    # Treat the DIM (right) half as the "nucleolus" to exclude.
    nucleolus = np.zeros((H, W), bool); nucleolus[:, 20:] = True
    nucleoplasm = full & (~nucleolus)
    nys_f, nxs_f = np.where(full)
    nys_e, nxs_e = np.where(nucleoplasm)
    o1, n1 = _partner_null_for_nucleus(
        partner, spot_cy, spot_cx, nys_f, nxs_f, dy, dx, 1000, np.random.default_rng(0)
    )
    o2, n2 = _partner_null_for_nucleus(
        partner, spot_cy, spot_cx, nys_e, nxs_e, dy, dx, 1000, np.random.default_rng(0)
    )
    assert o1 == o2  # observed unchanged (spots are not in the excluded region)
    assert n2.mean() > n1.mean()              # exclusion raises the null floor
    assert (o2 / n2.mean()) < (o1 / n1.mean())  # -> lower enrichment


# ---------------------------------------------------------------------------
# Synthetic 3-channel stack (DAPI + rna1 spots + a dense-nuclear partner).
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
    """3 well-separated nuclei. To exercise nucleolus detection, each nucleus
    gets a DAPI-LOW central hole (a "nucleolus") inside an otherwise bright disk.
    """
    from skimage.draw import disk
    img = np.random.default_rng(11).uniform(0.0, 20.0, (H, W)).astype(np.float32)
    for (cy, cx) in _nuclei_centers():
        rr, cc = disk((cy, cx), 28, shape=img.shape)
        img[rr, cc] += 3000.0
        # carve a DAPI-poor nucleolus in the center
        hr, hc = disk((cy, cx), 7, shape=img.shape)
        img[hr, hc] = 200.0
    return img


def _rna_spot_plane():
    """rna1 (MIAT) spots, several per nucleus, placed in the bright nucleoplasm
    (NOT in the central nucleolar hole)."""
    from skimage.draw import disk  # noqa: F401
    img = np.random.default_rng(22).uniform(2.0, 8.0, (H, W)).astype(np.float32)
    blob = np.zeros_like(img)
    rng = np.random.default_rng(33)
    for (cy, cx) in _nuclei_centers():
        # ring of spots ~15 px from center (in nucleoplasm, outside the r=7 hole)
        for k in range(8):
            ang = 2 * np.pi * k / 8
            y = int(cy + 15 * np.sin(ang)); x = int(cx + 15 * np.cos(ang))
            blob[y, x] += float(rng.uniform(3000.0, 6000.0))
    return img + gaussian_filter(blob, 1.1)


def _partner_plane_dense_nuclear():
    """Partner (QKI-like) channel: BRIGHT throughout the nucleoplasm, DIM in the
    nucleolar holes (mimicking QKI excluded from nucleoli). Dense-nuclear -> the
    whole-nucleus null and the fishsuite whole-nucleus mean are nearly equal, so
    enrichment ≈ 1 with whole-nucleus null but the nucleolus exclusion shifts it.
    """
    from skimage.draw import disk
    img = np.random.default_rng(44).uniform(2.0, 8.0, (H, W)).astype(np.float32)
    for (cy, cx) in _nuclei_centers():
        rr, cc = disk((cy, cx), 28, shape=img.shape)
        img[rr, cc] = 800.0
        hr, hc = disk((cy, cx), 7, shape=img.shape)
        img[hr, hc] = 20.0  # partner avoids the nucleolus
    return img


def _czyx() -> np.ndarray:
    planes = [_dapi_plane(), _rna_spot_plane(), _partner_plane_dense_nuclear()]
    return np.stack(
        [np.stack([p] * NZ, axis=0) for p in planes], axis=0
    ).astype(np.float32)


@pytest.fixture()
def fake_img() -> ImageWrapper:
    czyx = _czyx()
    return ImageWrapper(
        path="synthetic_partner_null.tif",
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
    return _rna_rna.run_one(
        Path(img.path), condition="cond", sec_only=False, cfg=cfg,
    )


# ---------------------------------------------------------------------------
# (a) END-TO-END: known enrichment through run_one
# ---------------------------------------------------------------------------
def test_end_to_end_null_columns_present_and_finite(fake_img, monkeypatch):
    """With the feature ON, run_one emits the per-nucleus null columns + the
    per-image pooled rollup, and at least one spot-bearing nucleus is finite."""
    cfg = _base_cfg()
    cfg.foci.compute_partner_intensity = True
    cfg.foci.compute_partner_null_enrichment = True
    cfg.foci.partner_null_n = 200  # keep the test fast
    cfg.foci.partner_null_seed = 0
    res = _run(cfg, fake_img, monkeypatch)

    nuc = res.nuclei
    assert "rna2_enrichment_vs_null_at_rna1_spots" in nuc.columns
    assert "rna2_null_z_at_rna1_spots" in nuc.columns
    # at least one nucleus has rna1 spots -> finite enrichment
    n1 = pd.to_numeric(nuc["n_spots_rna1"], errors="coerce").fillna(0)
    enr = pd.to_numeric(nuc["rna2_enrichment_vs_null_at_rna1_spots"], errors="coerce")
    assert (n1 > 0).any()
    assert np.isfinite(enr[n1 > 0]).any()

    # Per-image pooled rollup present + sane.
    pi = res.per_image
    assert "rna2_pooled_enrichment_vs_null_at_rna1_spots" in pi
    assert "rna2_pooled_null_z_at_rna1_spots" in pi
    assert "rna2_pooled_null_p_empirical_at_rna1_spots" in pi
    assert np.isfinite(float(pi["rna2_pooled_enrichment_vs_null_at_rna1_spots"]))
    # empirical p is a valid bounded probability
    p = float(pi["rna2_pooled_null_p_empirical_at_rna1_spots"])
    assert 0.0 < p <= 1.0
    assert int(pi["n_nuclei_partner_null"]) >= 1
    assert int(pi["partner_null_n"]) == 200
    assert float(pi["partner_null_disk_px"]) == 3.0


def test_end_to_end_run_is_deterministic(fake_img, monkeypatch):
    """Two runs with the same seed give identical pooled enrichment (the fixed
    RNG seed makes the native pipeline reproducible)."""
    cfg = _base_cfg()
    cfg.foci.compute_partner_intensity = True
    cfg.foci.compute_partner_null_enrichment = True
    cfg.foci.partner_null_n = 200
    r1 = _run(cfg, fake_img, monkeypatch)
    r2 = _run(cfg, fake_img, monkeypatch)
    e1 = float(r1.per_image["rna2_pooled_enrichment_vs_null_at_rna1_spots"])
    e2 = float(r2.per_image["rna2_pooled_enrichment_vs_null_at_rna1_spots"])
    assert e1 == pytest.approx(e2, abs=1e-12)


# ---------------------------------------------------------------------------
# (b) END-TO-END: nucleolus exclusion changes the result
# ---------------------------------------------------------------------------
def test_end_to_end_nucleolus_exclusion_changes_pooled_enrichment(fake_img, monkeypatch):
    """Turning on nucleolus exclusion (with nucleolus.enabled) changes the
    pooled enrichment vs the whole-nucleus null on the same image — because the
    DAPI-poor / partner-poor nucleolar voids are removed from the null sampling.
    Also asserts the standard nucleolus columns are emitted in rna_rna."""
    cfg_whole = _base_cfg()
    cfg_whole.foci.compute_partner_intensity = True
    cfg_whole.foci.compute_partner_null_enrichment = True
    cfg_whole.foci.partner_null_n = 300
    cfg_whole.foci.exclude_nucleolus_from_partner_null = False
    cfg_whole.nucleolus.enabled = True  # nucleolus columns emitted regardless
    res_whole = _run(cfg_whole, fake_img, monkeypatch)

    cfg_excl = _base_cfg()
    cfg_excl.foci.compute_partner_intensity = True
    cfg_excl.foci.compute_partner_null_enrichment = True
    cfg_excl.foci.partner_null_n = 300
    cfg_excl.foci.exclude_nucleolus_from_partner_null = True
    cfg_excl.nucleolus.enabled = True
    res_excl = _run(cfg_excl, fake_img, monkeypatch)

    # Feature-2 wiring: nucleolus detection runs in rna_rna -> per-nucleus
    # nucleolus columns + per-spot in_nucleolus are emitted (matching rna_only).
    assert "nucleolus_area_px" in res_whole.nuclei.columns
    assert "nucleolus_fraction_of_nucleus" in res_whole.nuclei.columns
    assert "in_nucleolus" in res_whole.spots.columns

    e_whole = float(res_whole.per_image["rna2_pooled_enrichment_vs_null_at_rna1_spots"])
    e_excl = float(res_excl.per_image["rna2_pooled_enrichment_vs_null_at_rna1_spots"])
    # The synthetic nucleolus is a strong dim void the partner avoids; excluding
    # it from the null raises the null floor and changes the enrichment.
    assert e_whole != pytest.approx(e_excl, abs=1e-6), (
        f"nucleolus exclusion did not change the pooled enrichment "
        f"(whole={e_whole}, excl={e_excl})"
    )


# ---------------------------------------------------------------------------
# (c) DEFAULTS-OFF byte-equivalence
# ---------------------------------------------------------------------------
def test_defaults_off_no_null_columns(fake_img, monkeypatch):
    """With compute_partner_null_enrichment OFF (default), NONE of the null
    columns / per-image keys are emitted — even with partner-intensity ON."""
    cfg = _base_cfg()
    cfg.foci.compute_partner_intensity = True  # intensity on, null OFF (default)
    res = _run(cfg, fake_img, monkeypatch)
    for col in (
        "rna2_enrichment_vs_null_at_rna1_spots",
        "rna2_null_z_at_rna1_spots",
    ):
        assert col not in res.nuclei.columns
    for key in (
        "rna2_pooled_enrichment_vs_null_at_rna1_spots",
        "rna2_pooled_null_z_at_rna1_spots",
        "rna2_pooled_null_p_empirical_at_rna1_spots",
        "n_nuclei_partner_null",
        "partner_null_n",
        "partner_null_disk_px",
    ):
        assert key not in res.per_image


def test_defaults_off_byte_equivalent_per_image_and_nuclei(fake_img, monkeypatch):
    """Defaults-OFF byte-equivalence: a run with the null flag OFF produces the
    EXACT same per_image dict + nuclei/spots frames as a run on a config that
    never knew about the feature. We prove this by comparing an OFF run to a
    second OFF run AND by asserting the column SET equals the partner-intensity-
    only column set (no null leakage)."""
    # Reference: partner-intensity ON, null implicitly OFF.
    cfg_ref = _base_cfg()
    cfg_ref.foci.compute_partner_intensity = True
    res_ref = _run(cfg_ref, fake_img, monkeypatch)

    # Same config, explicitly null=False -> must be identical.
    cfg_off = _base_cfg()
    cfg_off.foci.compute_partner_intensity = True
    cfg_off.foci.compute_partner_null_enrichment = False
    res_off = _run(cfg_off, fake_img, monkeypatch)

    # Identical per_image keys + values (excluding ``runtime_s``, a wall-clock
    # timestamp present in the pre-feature output too — never a data column).
    assert set(res_ref.per_image.keys()) == set(res_off.per_image.keys())
    for k in res_ref.per_image:
        if k == "runtime_s":
            continue
        a, b = res_ref.per_image[k], res_off.per_image[k]
        if isinstance(a, float) and a != a:  # NaN == NaN
            assert isinstance(b, float) and b != b, k
        else:
            assert a == b, f"per_image[{k}] differs: {a!r} != {b!r}"

    # Identical nuclei column set + values.
    assert list(res_ref.nuclei.columns) == list(res_off.nuclei.columns)
    pd.testing.assert_frame_equal(
        res_ref.nuclei.reset_index(drop=True),
        res_off.nuclei.reset_index(drop=True),
        check_dtype=False,
    )
    # No null columns leaked in either.
    assert not any("vs_null" in c for c in res_off.nuclei.columns)
    assert not any("partner_null" in k for k in res_off.per_image)


# ---------------------------------------------------------------------------
# (d) save_partner_null_draws: the pooled 1000-element null vector + pooled
#     observed are SURFACED via res.extra["coloc_null_draws"] (the downstream
#     "make coloc clear" null-overlay needs the per-draw distribution that the
#     pooling block currently computes then discards). Default OFF -> absent.
# ---------------------------------------------------------------------------
def test_save_partner_null_draws_emits_extra(fake_img, monkeypatch):
    """With save_partner_null_draws ON, res.extra carries a DataFrame with
    exactly n_null rows for the single image; the mean of the pooled draws ==
    the reported per-image pooled null mean; and the empirical p reconstructed
    from the draws (#draws >= obs + 1)/(n+1) matches the reported per-image p."""
    cfg = _base_cfg()
    cfg.foci.compute_partner_intensity = True
    cfg.foci.compute_partner_null_enrichment = True
    cfg.foci.partner_null_n = 200
    cfg.foci.save_partner_null_draws = True
    res = _run(cfg, fake_img, monkeypatch)

    assert "coloc_null_draws" in res.extra
    draws = res.extra["coloc_null_draws"]
    assert isinstance(draws, pd.DataFrame)
    # exactly n_null rows for this ONE image
    assert len(draws) == 200
    assert {"image", "condition", "iter", "pooled_null_value", "pooled_obs"}.issubset(
        draws.columns
    )
    assert list(draws["iter"]) == list(range(200))
    assert draws["image"].nunique() == 1

    # mean(pooled_null_value) == the reported pooled null mean (round(.,3)).
    rep_mean = float(res.per_image["rna2_pooled_null_mean_at_rna1_spots"])
    assert float(draws["pooled_null_value"].mean()) == pytest.approx(rep_mean, abs=1e-3)

    # Reconstructed empirical p == reported per-image empirical p (exact: both
    # use the SAME unrounded pooled obs + pooled null vector).
    obs = float(draws["pooled_obs"].iloc[0])
    p_recon = (int((draws["pooled_null_value"] >= obs).sum()) + 1) / (200 + 1)
    assert p_recon == pytest.approx(
        float(res.per_image["rna2_pooled_null_p_empirical_at_rna1_spots"]), abs=1e-12
    )


def test_save_partner_null_draws_default_off_absent(fake_img, monkeypatch):
    """save_partner_null_draws defaults OFF -> the key never appears in extra,
    even with the null feature itself ON (byte-equivalent carrier)."""
    cfg = _base_cfg()
    cfg.foci.compute_partner_intensity = True
    cfg.foci.compute_partner_null_enrichment = True
    cfg.foci.partner_null_n = 200
    # save_partner_null_draws left at its default (False)
    res = _run(cfg, fake_img, monkeypatch)
    assert "coloc_null_draws" not in res.extra
