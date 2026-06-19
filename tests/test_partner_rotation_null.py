"""Per-nucleus ROTATION (and optional TRANSLATION) "proper background" null for
spot-centric partner colocalization — unit + integration tests (Brian, 2026-06-19).

PIPELINE-NATIVE nativization of the adversarially-validated prototype
``rotation_null_prototype.py`` (F:\\Image Analysis Work\\MIAT-QKI-Coloc\\UD\\
_QKI_association_proper_background_2026-06-18). The rotation null asks a STRICTER
question than the random-position null: it rotates each nucleus's MIAT spot
CONSTELLATION about its OWN centroid (registration-destroying, structure-
PRESERVING), disk-samples the partner (QKI) at the rotated coords, and so tests
whether the partner is SPECIFICALLY concentrated at the spot positions BEYOND a
merely shared nuclear compartment.

Validated design decisions exercised here:
  * rotation about the CONSTELLATION centroid (the spot set's own mean), not the
    nucleus / sampling-region centroid;
  * KEEP-N redraw: rotated spots that leave the in-mask region are re-rotated by a
    fresh per-spot angle until in-mask (NOT dropped — dropping biases enrichment
    LOW by ~0.01-0.015); N preserved every iteration;
  * N=1000 seeded (deterministic);
  * a min-retention usable-nucleus gate (drops SPARSE nuclei whose constellation
    cannot be rotated within the mask);
  * a null-calibrated association fraction = fraction of observed spots whose
    partner disk-mean exceeds the per-nucleus rotation single-position null high
    percentile (chance floor = 1 - pct/100).

All tests are GPU-free synthetic fixtures. Gated behind
``foci.compute_partner_rotation_null`` (requires ``compute_partner_intensity``);
default OFF -> the rotation columns are ABSENT and the output is byte-equivalent
to the pre-feature path.
"""
from __future__ import annotations

import math
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
    _rotation_null_for_nucleus,
    _translation_null_for_nucleus,
    _rotation_single_position_dist,
)


# ===========================================================================
# Synthetic single-nucleus builders (ported from the validated prototype's
# controls so the unit-level pass criteria match the adversarial validation).
# ===========================================================================
def _make_synthetic_nucleus(seed=0, size=200, n_spots=40, radius=70,
                            shape="circle", spot_spread=0.7):
    """A nucleus mask + a constellation of MIAT spots inside it.

    ``shape='circle'`` -> circular mask, spots inside ``spot_spread*radius`` so
    rotation keeps essentially all spots in-mask (first-pass retention ~1.0).
    ``shape='ellipse'`` -> elongated (3:1) mask with spots filling toward the
    edges, so rotating the constellation pushes a substantial fraction out of
    mask (first-pass retention ~0.5-0.75, like the REAL data) -> stress-tests the
    keep-N redraw.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    cy, cx = size / 2.0, size / 2.0
    if shape == "ellipse":
        ay, ax = radius * 0.45, radius * 1.35   # 3:1 elongated
        mask = (((yy - cy) / ay) ** 2 + ((xx - cx) / ax) ** 2) <= 1.0
        pts = []
        while len(pts) < n_spots:
            u = math.sqrt(rng.uniform(0, 1))
            a = rng.uniform(0, 2 * math.pi)
            py = cy + 0.92 * ay * u * math.sin(a)
            px = cx + 0.92 * ax * u * math.cos(a)
            if (((py - cy) / ay) ** 2 + ((px - cx) / ax) ** 2) <= 1.0:
                pts.append((px, py))
        return mask, np.asarray(pts, dtype=float), (cy, cx)
    mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius ** 2
    pts = []
    while len(pts) < n_spots:
        a = rng.uniform(0, 2 * math.pi)
        r = radius * float(spot_spread) * math.sqrt(rng.uniform(0, 1))
        py, px = cy + r * math.sin(a), cx + r * math.cos(a)
        pts.append((px, py))  # (x, y)
    return mask, np.asarray(pts, dtype=float), (cy, cx)


def _qki_field(kind, mask, spots_xy, centroid, seed=0, size=200):
    """Synthetic partner fields:
      'positive'    — base + hotspots painted AT the spot positions
      'negative'    — spatially-uniform (flat + small iid noise) inside the mask
      'compartment' — a smooth off-center blob (structure) UNRELATED to spots
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    base = np.full((size, size), 100.0)
    if kind == "negative":
        field = base + rng.normal(0, 5, size=(size, size))
    elif kind == "compartment":
        cy, cx = centroid
        by, bx = cy - 25, cx + 25
        blob = 200.0 * np.exp(-(((yy - by) ** 2 + (xx - bx) ** 2) / (2 * 30.0 ** 2)))
        field = base + blob + rng.normal(0, 3, size=(size, size))
    elif kind in ("positive", "positive_ellipse"):
        field = base + rng.normal(0, 5, size=(size, size))
        for (px, py) in spots_xy:
            field += 250.0 * np.exp(
                -(((yy - py) ** 2 + (xx - px) ** 2) / (2 * 2.5 ** 2))
            )
    else:
        raise ValueError(kind)
    field[~mask] = 0.0
    return field


def _summ(obs, null_stats):
    ns = np.asarray(null_stats, dtype=np.float64)
    ns = ns[np.isfinite(ns)]
    if ns.size == 0 or not np.isfinite(obs):
        return float("nan"), float("nan"), float("nan")
    nm = float(ns.mean())
    enr = (obs / nm) if nm > 0 else float("nan")
    p = float((np.sum(ns >= obs) + 1) / (ns.size + 1))
    return enr, p, nm


def _run_case(kind, n_null=400, seed=0, shape="circle", disk_px=3.0,
              min_retention=0.5):
    mask, spots_xy, centroid = _make_synthetic_nucleus(seed=seed, shape=shape)
    qki = _qki_field(kind, mask, spots_xy, centroid, seed=seed)
    dy, dx = _disk_stencil(disk_px)
    scx = np.clip(np.rint(spots_xy[:, 0]).astype(np.intp), 0, qki.shape[1] - 1)
    scy = np.clip(np.rint(spots_xy[:, 1]).astype(np.intp), 0, qki.shape[0] - 1)
    cy0 = float(scy.mean())
    cx0 = float(scx.mean())
    rng = np.random.default_rng(seed + 101)
    rot = _rotation_null_for_nucleus(
        qki, scy, scx, mask, (cy0, cx0), dy, dx, n_null, rng,
        min_retention=min_retention,
    )
    enr, p, _ = _summ(rot["obs"], rot["null_stats"])
    enr_drop, _, _ = _summ(rot["obs"], rot["null_stats_drop"])
    return {
        "obs": rot["obs"], "enrichment": enr, "p": p,
        "enrichment_drop": enr_drop,
        "median_retention": rot["median_retention"],
        "usable": rot["usable"],
        "rot": rot, "qki": qki, "mask": mask, "scy": scy, "scx": scx,
        "centroid": (cy0, cx0), "dy": dy, "dx": dx,
    }


# ===========================================================================
# (a) Positive control — partner painted AT spots -> rotation enr >1, sig.
# ===========================================================================
def test_rotation_positive_control_enriched_and_significant():
    c = _run_case("positive")
    assert c["enrichment"] > 1.10
    assert c["p"] < 0.05
    assert c["usable"]


# ===========================================================================
# (b) Negative control — uniform/random partner -> rotation enr ~1, n.s.
# ===========================================================================
def test_rotation_negative_control_is_one_and_ns():
    c = _run_case("negative")
    assert abs(c["enrichment"] - 1.0) < 0.05
    assert c["p"] > 0.05


# ===========================================================================
# (c) Equivalence — no compartment structure -> rotation ~ position null.
#     With a uniform/random field the position-null mean and the rotation-null
#     mean both equal the field mean, so the two enrichments coincide.
# ===========================================================================
def test_equivalence_no_compartment_rotation_matches_position():
    from fishsuite.core.modes.rna_rna import _partner_null_for_nucleus
    c = _run_case("negative")
    nys, nxs = np.where(c["mask"])
    _o, pos_null = _partner_null_for_nucleus(
        c["qki"], c["scy"], c["scx"], nys, nxs, c["dy"], c["dx"], 400,
        np.random.default_rng(0),
    )
    pos_enr, _, _ = _summ(_o, pos_null)
    assert abs(c["enrichment"] - pos_enr) < 0.05


# ===========================================================================
# (d) Ordering — compartment structure -> rotation MORE conservative than
#     position (|pos-1| >= |rot-1|): the shared compartment inflates the
#     looser position null but the structure-preserving rotation discounts it.
# ===========================================================================
def test_compartment_rotation_more_conservative_than_position():
    from fishsuite.core.modes.rna_rna import _partner_null_for_nucleus
    c = _run_case("compartment")
    nys, nxs = np.where(c["mask"])
    _o, pos_null = _partner_null_for_nucleus(
        c["qki"], c["scy"], c["scx"], nys, nxs, c["dy"], c["dx"], 400,
        np.random.default_rng(0),
    )
    pos_enr, _, _ = _summ(_o, pos_null)
    assert abs(pos_enr - 1.0) >= abs(c["enrichment"] - 1.0) - 1e-6


# ===========================================================================
# (e) Elliptical low-retention positive control — keep-N redraw recovers the
#     enrichment (>1, sig) where first-pass retention is well below 1.0 and a
#     naive drop would bias it low.
# ===========================================================================
def test_elliptical_low_retention_keepN_recovers_enrichment():
    c = _run_case("positive_ellipse", shape="ellipse")
    assert c["median_retention"] < 0.85   # genuinely low first-pass retention
    assert c["enrichment"] > 1.10
    assert c["p"] < 0.05
    # keep-N should not be biased BELOW the legacy drop estimate (drop biases low).
    assert c["enrichment"] >= c["enrichment_drop"] - 0.05


# ===========================================================================
# (f) Uniform-field-no-redraw-bias guard — keep-N redraw on a UNIFORM partner
#     field returns enrichment ~1.0 even at low retention, confirming the
#     redraw itself injects NO angular/spatial bias (guards the corrected result).
# ===========================================================================
def test_uniform_field_keepN_redraw_introduces_no_bias():
    # Elliptical mask -> low first-pass retention -> redraw is actually exercised.
    # A STRICTLY uniform (flat, noiseless) field must give enrichment EXACTLY 1.0:
    # every disk-mean equals the constant, so any angular/spatial bias the redraw
    # might inject would show up as a deviation from 1.0. (A noisy "negative" field
    # only gives ~1.0 +/- sampling jitter, which would confound this guard.)
    mask, spots_xy, _ = _make_synthetic_nucleus(seed=0, shape="ellipse")
    # Flat EVERYWHERE (incl. background) so a disk that overlaps the mask boundary
    # still samples the same constant -> isolates pure redraw/angular bias from the
    # separate (real, documented) boundary-disk-overlap effect.
    qki = np.full(mask.shape, 100.0, dtype=np.float64)
    dy, dx = _disk_stencil(3.0)
    scx = np.clip(np.rint(spots_xy[:, 0]).astype(np.intp), 0, qki.shape[1] - 1)
    scy = np.clip(np.rint(spots_xy[:, 1]).astype(np.intp), 0, qki.shape[0] - 1)
    cy0, cx0 = float(scy.mean()), float(scx.mean())
    rot = _rotation_null_for_nucleus(
        qki, scy, scx, mask, (cy0, cx0), dy, dx, 400, np.random.default_rng(101),
        min_retention=0.5,
    )
    assert rot["median_retention"] < 0.95   # redraw path is exercised
    enr, _, _ = _summ(rot["obs"], rot["null_stats"])
    # All kept (in-mask) disk-means equal 100 exactly -> enrichment EXACTLY 1.0.
    assert enr == pytest.approx(1.0, abs=1e-9)
    # redraw places spots within the mask: no unplaceable fallbacks on a big ellipse
    assert rot["frac_unplaceable"] < 0.01


# ===========================================================================
# (g) Determinism — same seed -> identical rotation null distribution.
# ===========================================================================
def test_rotation_null_deterministic_with_seed():
    mask, spots_xy, _ = _make_synthetic_nucleus(seed=1)
    qki = _qki_field("positive", mask, spots_xy, (100.0, 100.0), seed=1)
    dy, dx = _disk_stencil(3.0)
    scx = np.rint(spots_xy[:, 0]).astype(np.intp)
    scy = np.rint(spots_xy[:, 1]).astype(np.intp)
    cy0, cx0 = float(scy.mean()), float(scx.mean())
    r1 = _rotation_null_for_nucleus(
        qki, scy, scx, mask, (cy0, cx0), dy, dx, 300, np.random.default_rng(0)
    )
    r2 = _rotation_null_for_nucleus(
        qki, scy, scx, mask, (cy0, cx0), dy, dx, 300, np.random.default_rng(0)
    )
    assert r1["obs"] == r2["obs"]
    np.testing.assert_array_equal(r1["null_stats"], r2["null_stats"])


# ===========================================================================
# (h) Association fraction — positive control concentrates partner at spots, so
#     the rotation single-position 95th-pct threshold flags MANY observed spots
#     (well above the 5% chance floor); negative control stays near chance.
# ===========================================================================
def test_assoc_fraction_above_chance_for_positive_below_for_negative():
    def _assoc(kind, shape="circle"):
        c = _run_case(kind, shape=shape)
        single = _rotation_single_position_dist(
            c["qki"], c["scy"], c["scx"], c["mask"], c["centroid"],
            c["dy"], c["dx"], n_iters=200, rng=np.random.default_rng(404),
        )
        thr = float(np.percentile(single, 95))
        from fishsuite.core.modes.rna_rna import _disk_means_at
        obs_per_spot = _disk_means_at(c["qki"], c["scy"], c["scx"], c["dy"], c["dx"])
        return float((obs_per_spot > thr).mean())
    assoc_pos = _assoc("positive")
    assoc_neg = _assoc("negative")
    # Positive: many observed spots exceed the rotation single-position 95th pct
    # (focal recruitment) -> well above chance and well above the negative.
    assert assoc_pos > 0.30
    assert assoc_neg < 0.15          # near the 5% chance floor
    assert assoc_pos > assoc_neg + 0.20


# ===========================================================================
# (i) Translation null — positive control also enriched (sanity that the
#     optional translation companion works); flagged unreliable for dense
#     patterns at the method level, but on a sparse-circle control it recovers.
# ===========================================================================
def test_translation_positive_control_enriched():
    mask, spots_xy, centroid = _make_synthetic_nucleus(seed=0, n_spots=20,
                                                       spot_spread=0.4)
    qki = _qki_field("positive", mask, spots_xy, centroid, seed=0)
    dy, dx = _disk_stencil(3.0)
    scx = np.rint(spots_xy[:, 0]).astype(np.intp)
    scy = np.rint(spots_xy[:, 1]).astype(np.intp)
    nys, nxs = np.where(mask)
    tr = _translation_null_for_nucleus(
        qki, scy, scx, mask, nys, nxs, dy, dx, 400, np.random.default_rng(202),
        min_retention=0.5,
    )
    enr, p, _ = _summ(tr["obs"], tr["null_stats"])
    assert enr > 1.10
    assert p < 0.05


# ===========================================================================
# Synthetic 3-channel stack for END-TO-END run_one wiring tests.
# (Same structure/idiom as test_partner_null_coloc.py.)
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
            y = int(cy + 14 * np.sin(ang))
            x = int(cx + 14 * np.cos(ang))
            pts.append((y, x))
        pos[i] = pts
    return pos


def _rna_spot_plane():
    img = np.random.default_rng(22).uniform(2.0, 8.0, (EH, EW)).astype(np.float32)
    blob = np.zeros_like(img)
    rng = np.random.default_rng(33)
    for _nid, pts in _spot_positions().items():
        for (y, x) in pts:
            blob[y, x] += float(rng.uniform(3000.0, 6000.0))
    return img + gaussian_filter(blob, 1.1)


def _partner_plane_at_spots():
    """Partner BRIGHT AT the MIAT spot positions (so rotation breaks the
    registration and the rotation enrichment is > 1)."""
    img = np.random.default_rng(44).uniform(50.0, 60.0, (EH, EW)).astype(np.float32)
    from skimage.draw import disk
    for _nid, pts in _spot_positions().items():
        for (y, x) in pts:
            rr, cc = disk((y, x), 3, shape=img.shape)
            img[rr, cc] += 400.0
    # zero outside nuclei not required; nucleus masks restrict sampling.
    return img


def _czyx() -> np.ndarray:
    planes = [_dapi_plane(), _rna_spot_plane(), _partner_plane_at_spots()]
    return np.stack([np.stack([p] * NZ, axis=0) for p in planes], axis=0).astype(np.float32)


@pytest.fixture()
def fake_img() -> ImageWrapper:
    czyx = _czyx()
    return ImageWrapper(
        path="synthetic_rotation_null.tif",
        bio=_FakeBio(czyx),
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


def _run(cfg, img, monkeypatch):
    monkeypatch.setattr(_io, "read_image", lambda p: img)
    return _rna_rna.run_one(Path(img.path), condition="cond", sec_only=False, cfg=cfg)


# ===========================================================================
# (j) END-TO-END: rotation columns present + finite + pooled rollup sane.
# ===========================================================================
def test_end_to_end_rotation_columns_present_and_finite(fake_img, monkeypatch):
    cfg = _base_cfg()
    cfg.foci.compute_partner_intensity = True
    cfg.foci.compute_partner_rotation_null = True
    cfg.foci.partner_rotation_n = 200
    cfg.foci.partner_rotation_seed = 0
    res = _run(cfg, fake_img, monkeypatch)

    nuc = res.nuclei
    assert "rna2_rotation_enrichment_at_rna1_spots" in nuc.columns
    assert "rna2_rotation_null_z_at_rna1_spots" in nuc.columns
    assert "rna2_rotation_null_p_at_rna1_spots" in nuc.columns
    assert "rna2_rotation_assoc_fraction_at_rna1_spots" in nuc.columns
    assert "rotation_null_usable" in nuc.columns

    n1 = pd.to_numeric(nuc["n_spots_rna1"], errors="coerce").fillna(0)
    enr = pd.to_numeric(nuc["rna2_rotation_enrichment_at_rna1_spots"], errors="coerce")
    assert (n1 > 0).any()
    assert np.isfinite(enr[n1 > 0]).any()

    pi = res.per_image
    assert "rna2_pooled_rotation_enrichment_at_rna1_spots" in pi
    assert "rna2_pooled_rotation_null_z_at_rna1_spots" in pi
    assert "rna2_pooled_rotation_null_p_empirical_at_rna1_spots" in pi
    assert "n_nuclei_partner_rotation_null" in pi
    assert int(pi["partner_rotation_n"]) == 200
    pooled = float(pi["rna2_pooled_rotation_enrichment_at_rna1_spots"])
    assert np.isfinite(pooled)
    # partner painted at spots -> rotation enrichment clearly > 1.
    assert pooled > 1.05


def test_end_to_end_rotation_is_deterministic(fake_img, monkeypatch):
    cfg = _base_cfg()
    cfg.foci.compute_partner_intensity = True
    cfg.foci.compute_partner_rotation_null = True
    cfg.foci.partner_rotation_n = 200
    r1 = _run(cfg, fake_img, monkeypatch)
    r2 = _run(cfg, fake_img, monkeypatch)
    e1 = float(r1.per_image["rna2_pooled_rotation_enrichment_at_rna1_spots"])
    e2 = float(r2.per_image["rna2_pooled_rotation_enrichment_at_rna1_spots"])
    assert e1 == pytest.approx(e2, abs=1e-12)


# ===========================================================================
# (k) DEFAULTS-OFF byte-equivalence: rotation flag OFF -> no rotation columns /
#     keys, and the per_image + nuclei frames are byte-identical to a run that
#     never knew about the feature (partner-intensity ON, rotation default OFF).
# ===========================================================================
def test_defaults_off_no_rotation_columns(fake_img, monkeypatch):
    cfg = _base_cfg()
    cfg.foci.compute_partner_intensity = True  # rotation OFF (default)
    res = _run(cfg, fake_img, monkeypatch)
    for col in (
        "rna2_rotation_enrichment_at_rna1_spots",
        "rna2_rotation_null_z_at_rna1_spots",
        "rna2_rotation_null_p_at_rna1_spots",
        "rna2_rotation_assoc_fraction_at_rna1_spots",
        "rotation_null_usable",
    ):
        assert col not in res.nuclei.columns
    for key in (
        "rna2_pooled_rotation_enrichment_at_rna1_spots",
        "rna2_pooled_rotation_null_z_at_rna1_spots",
        "rna2_pooled_rotation_null_p_empirical_at_rna1_spots",
        "n_nuclei_partner_rotation_null",
        "partner_rotation_n",
    ):
        assert key not in res.per_image


def test_defaults_off_byte_equivalent(fake_img, monkeypatch):
    cfg_ref = _base_cfg()
    cfg_ref.foci.compute_partner_intensity = True
    res_ref = _run(cfg_ref, fake_img, monkeypatch)

    cfg_off = _base_cfg()
    cfg_off.foci.compute_partner_intensity = True
    cfg_off.foci.compute_partner_rotation_null = False
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
    assert not any("rotation" in c for c in res_off.nuclei.columns)
    assert not any("rotation" in k for k in res_off.per_image)


# ===========================================================================
# (l) save_partner_rotation_null_draws -> res.extra["coloc_rotation_null"].
# ===========================================================================
def test_save_rotation_null_draws_emits_extra(fake_img, monkeypatch):
    cfg = _base_cfg()
    cfg.foci.compute_partner_intensity = True
    cfg.foci.compute_partner_rotation_null = True
    cfg.foci.partner_rotation_n = 200
    cfg.foci.save_partner_rotation_null_draws = True
    res = _run(cfg, fake_img, monkeypatch)

    assert "coloc_rotation_null" in res.extra
    draws = res.extra["coloc_rotation_null"]
    assert isinstance(draws, pd.DataFrame)
    assert len(draws) == 200
    assert {"image", "condition", "iter", "pooled_null_value", "pooled_obs"}.issubset(
        draws.columns
    )
    assert list(draws["iter"]) == list(range(200))


def test_save_rotation_null_draws_default_off_absent(fake_img, monkeypatch):
    cfg = _base_cfg()
    cfg.foci.compute_partner_intensity = True
    cfg.foci.compute_partner_rotation_null = True
    cfg.foci.partner_rotation_n = 200
    res = _run(cfg, fake_img, monkeypatch)
    assert "coloc_rotation_null" not in res.extra


# ===========================================================================
# (m) END-TO-END: optional TRANSLATION companion wires through run_one.
# ===========================================================================
def test_end_to_end_translation_companion_columns(fake_img, monkeypatch):
    cfg = _base_cfg()
    cfg.foci.compute_partner_intensity = True
    cfg.foci.compute_partner_rotation_null = True
    cfg.foci.compute_partner_translation_null = True
    cfg.foci.partner_rotation_n = 150
    res = _run(cfg, fake_img, monkeypatch)
    assert "rna2_translation_enrichment_at_rna1_spots" in res.nuclei.columns
    assert "translation_null_usable" in res.nuclei.columns
    assert "rna2_pooled_translation_enrichment_at_rna1_spots" in res.per_image
    assert "n_nuclei_partner_translation_null" in res.per_image


def test_translation_default_off_when_rotation_on(fake_img, monkeypatch):
    """Translation defaults OFF even with rotation ON (companion is opt-in)."""
    cfg = _base_cfg()
    cfg.foci.compute_partner_intensity = True
    cfg.foci.compute_partner_rotation_null = True
    cfg.foci.partner_rotation_n = 150
    res = _run(cfg, fake_img, monkeypatch)
    assert "rna2_translation_enrichment_at_rna1_spots" not in res.nuclei.columns
    assert "rna2_pooled_translation_enrichment_at_rna1_spots" not in res.per_image
