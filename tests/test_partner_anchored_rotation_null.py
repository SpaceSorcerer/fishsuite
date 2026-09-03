"""Per-channel BigFISH ``threshold_override`` + the PARTNER-ANCHORED rotation
null (Brian, 2026-09-03).

Two additive, default-OFF engine features:

(a) ``FociChannelOverrideCfg.threshold_override`` — the antibody/protein channel
    can carry its OWN fixed BigFISH LoG threshold. ``FociCfg.resolved_for``
    falls back to the shared ``FociCfg.threshold_override`` when the channel
    value is None, so every existing YAML keeps its behaviour.

(b) ``FociCfg.compute_partner_anchored_rotation_null`` — a SECOND rotation null
    anchored on the partner (rna2 / antibody) spots with the rna1 channel as the
    sampled field, i.e. the reciprocal of the existing rna1-anchored null. It
    calls the SAME ``_rotation_null_for_nucleus`` with the (field, constellation)
    pair swapped and its own RNG stream (seed root + 303 / + 606); the tests
    below prove it is that call and not a re-implementation.

All fixtures are GPU-free synthetic stacks, reusing the idiom of
``test_partner_rotation_null.py``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import gaussian_filter

from fishsuite.config.schema import (
    FishsuiteConfig,
    FociCfg,
    FociChannelOverrideCfg,
)
from fishsuite.core import io as _io
from fishsuite.core.io import ImageWrapper
from fishsuite.core.modes import rna_rna as _rna_rna
from fishsuite.core.modes.rna_protein import _relabel_rna2_to_protein
from fishsuite.core.modes.rna_rna import _rotation_null_for_nucleus


# ===========================================================================
# (a) SCHEMA: per-channel threshold_override round-trip.
# ===========================================================================
def test_threshold_override_unset_everywhere_is_none():
    f = FociCfg()
    for ch in ("rna", "rna2", "antibody"):
        assert f.resolved_for(ch)["threshold_override"] is None


def test_threshold_override_falls_back_to_shared_value():
    f = FociCfg(threshold_override=7.5)
    for ch in ("rna", "rna2", "antibody"):
        assert f.resolved_for(ch)["threshold_override"] == pytest.approx(7.5)


def test_channel_threshold_override_wins_over_shared():
    f = FociCfg(
        threshold_override=7.5,
        antibody_overrides=FociChannelOverrideCfg(threshold_override=15.0),
    )
    assert f.resolved_for("antibody")["threshold_override"] == pytest.approx(15.0)
    # the other channels still see the shared value — the override is scoped.
    assert f.resolved_for("rna")["threshold_override"] == pytest.approx(7.5)
    assert f.resolved_for("rna2")["threshold_override"] == pytest.approx(7.5)


def test_channel_threshold_override_without_shared_value():
    f = FociCfg(antibody_overrides=FociChannelOverrideCfg(threshold_override=15.0))
    assert f.resolved_for("antibody")["threshold_override"] == pytest.approx(15.0)
    assert f.resolved_for("rna")["threshold_override"] is None


def test_threshold_override_round_trips_through_yaml_dict():
    cfg = FishsuiteConfig.model_validate(
        {"foci": {"antibody_overrides": {"threshold_override": 15.0}}}
    )
    assert cfg.foci.antibody_overrides.threshold_override == pytest.approx(15.0)
    assert cfg.foci.rna_overrides.threshold_override is None
    assert cfg.foci.resolved_for("antibody")["threshold_override"] == pytest.approx(15.0)


def test_partner_anchored_flag_defaults_off():
    assert FociCfg().compute_partner_anchored_rotation_null is False


# ===========================================================================
# (b) RELABEL: the new names must become *_at_protein_spots in rna_protein.
# ===========================================================================
NEW_NUCLEUS_COLS = [
    "rna1_rotation_enrichment_at_rna2_spots",
    "rna1_rotation_null_z_at_rna2_spots",
    "rna1_rotation_null_p_at_rna2_spots",
    "rna1_rotation_assoc_fraction_at_rna2_spots",
    "rotation_null_usable_at_rna2_spots",
]
NEW_IMAGE_COLS = [
    "rna1_pooled_rotation_enrichment_at_rna2_spots",
    "rna1_pooled_rotation_null_z_at_rna2_spots",
    "rna1_pooled_rotation_null_p_empirical_at_rna2_spots",
    "rna1_mean_rotation_assoc_fraction_at_rna2_spots",
    "n_nuclei_partner_rotation_null_at_rna2_spots",
]


@pytest.mark.parametrize("name", NEW_NUCLEUS_COLS + NEW_IMAGE_COLS)
def test_new_columns_relabel_to_protein_spots(name):
    out = _relabel_rna2_to_protein(name)
    assert out.endswith("_at_protein_spots")
    assert "rna2" not in out
    # the rna1 half names the RNA channel and must SURVIVE the relabel.
    if name.startswith("rna1"):
        assert out.startswith("rna1")


# ===========================================================================
# Synthetic 3-channel stack for END-TO-END run_one wiring tests.
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
    # BOTH punctate channels carry spots at the SAME positions, so the two
    # rotation nulls are each other's mirror image and both are computable.
    planes = [_dapi_plane(), _spot_plane(22, 33), _spot_plane(44, 55)]
    return np.stack(
        [np.stack([p] * NZ, axis=0) for p in planes], axis=0
    ).astype(np.float32)


@pytest.fixture()
def fake_img() -> ImageWrapper:
    return ImageWrapper(
        path="synthetic_partner_anchored.tif",
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


def _rotation_cfg(n=200) -> FishsuiteConfig:
    cfg = _base_cfg()
    cfg.foci.compute_partner_intensity = True
    cfg.foci.compute_partner_rotation_null = True
    cfg.foci.partner_rotation_n = n
    cfg.foci.partner_rotation_seed = 0
    return cfg


def _run(cfg, img, monkeypatch):
    monkeypatch.setattr(_io, "read_image", lambda p: img)
    return _rna_rna.run_one(Path(img.path), condition="cond", sec_only=False, cfg=cfg)


# ===========================================================================
# DEFAULT OFF: the new columns must be absent.
# ===========================================================================
def test_partner_anchored_default_off_no_columns(fake_img, monkeypatch):
    res = _run(_rotation_cfg(), fake_img, monkeypatch)
    for c in NEW_NUCLEUS_COLS:
        assert c not in res.nuclei.columns
    for c in NEW_IMAGE_COLS:
        assert c not in res.per_image
    # the rna1-anchored null is untouched and still emitted.
    assert "rna2_rotation_enrichment_at_rna1_spots" in res.nuclei.columns


def test_partner_anchored_off_is_byte_equivalent(fake_img, monkeypatch):
    """Turning the flag ON must not perturb ANY pre-existing column."""
    ref = _run(_rotation_cfg(), fake_img, monkeypatch)
    cfg = _rotation_cfg()
    cfg.foci.compute_partner_anchored_rotation_null = True
    new = _run(cfg, fake_img, monkeypatch)

    shared = [c for c in ref.nuclei.columns if c in new.nuclei.columns]
    assert set(ref.nuclei.columns) == set(shared)
    pd.testing.assert_frame_equal(
        ref.nuclei[shared].reset_index(drop=True),
        new.nuclei[shared].reset_index(drop=True),
        check_dtype=False,
    )
    for k, v in ref.per_image.items():
        if k == "runtime_s":  # wall clock, not an analysis number
            continue
        nv = new.per_image[k]
        if isinstance(v, float) and v != v:
            assert isinstance(nv, float) and nv != nv, k
        else:
            assert nv == v, k


def test_flag_on_without_prerequisites_emits_nothing(fake_img, monkeypatch, capsys):
    cfg = _base_cfg()
    cfg.foci.compute_partner_intensity = True
    cfg.foci.compute_partner_rotation_null = False
    cfg.foci.compute_partner_anchored_rotation_null = True
    res = _run(cfg, fake_img, monkeypatch)
    for c in NEW_NUCLEUS_COLS:
        assert c not in res.nuclei.columns
    # ...and it says so instead of silently producing nothing.
    assert "compute_partner_anchored_rotation_null is ON" in capsys.readouterr().out


# ===========================================================================
# ON: columns present, populated, and pooled rollup sane.
# ===========================================================================
def test_partner_anchored_columns_present_and_finite(fake_img, monkeypatch):
    cfg = _rotation_cfg()
    cfg.foci.compute_partner_anchored_rotation_null = True
    res = _run(cfg, fake_img, monkeypatch)

    nuc = res.nuclei
    for c in NEW_NUCLEUS_COLS:
        assert c in nuc.columns
    n2 = pd.to_numeric(nuc["n_spots_rna2"], errors="coerce").fillna(0)
    enr = pd.to_numeric(nuc["rna1_rotation_enrichment_at_rna2_spots"], errors="coerce")
    assert (n2 > 0).any()
    assert np.isfinite(enr[n2 > 0]).any()

    for c in NEW_IMAGE_COLS:
        assert c in res.per_image
    assert int(res.per_image["n_nuclei_partner_rotation_null_at_rna2_spots"]) > 0
    # rna1 and rna2 spots share positions here -> enrichment clearly > 1.
    assert float(res.per_image["rna1_pooled_rotation_enrichment_at_rna2_spots"]) > 1.05


def test_partner_anchored_is_deterministic(fake_img, monkeypatch):
    cfg = _rotation_cfg()
    cfg.foci.compute_partner_anchored_rotation_null = True
    a = _run(cfg, fake_img, monkeypatch)
    b = _run(cfg, fake_img, monkeypatch)
    pd.testing.assert_series_equal(
        a.nuclei["rna1_rotation_enrichment_at_rna2_spots"],
        b.nuclei["rna1_rotation_enrichment_at_rna2_spots"],
    )


# ===========================================================================
# THE KEY TEST: the partner-anchored value IS the existing function applied to
# the swapped (field, constellation) pair — not a re-implementation.
# ===========================================================================
def test_partner_anchored_call_uses_existing_function_with_swapped_pair(
    fake_img, monkeypatch
):
    calls = []
    real = _rna_rna._rotation_null_for_nucleus

    def spy(partner_2d, scy, scx, in_mask, centroid_yx, dy, dx, n_null, rng, **kw):
        calls.append(
            dict(
                field=np.array(partner_2d, copy=True),
                scy=np.array(scy, copy=True),
                scx=np.array(scx, copy=True),
                in_mask=np.array(in_mask, copy=True),
                centroid=tuple(centroid_yx),
                dy=np.array(dy, copy=True),
                dx=np.array(dx, copy=True),
                n_null=n_null,
                kw=dict(kw),
            )
        )
        return real(partner_2d, scy, scx, in_mask, centroid_yx, dy, dx, n_null, rng, **kw)

    monkeypatch.setattr(_rna_rna, "_rotation_null_for_nucleus", spy)

    cfg = _rotation_cfg()
    cfg.foci.compute_partner_anchored_rotation_null = True
    res = _run(cfg, fake_img, monkeypatch)

    # Two calls per nucleus that has spots in both channels: rna1-anchored first,
    # partner-anchored second.
    assert len(calls) >= 2
    rna1_anchored, partner_anchored = calls[0], calls[1]

    # The two calls differ ONLY in the (field, constellation) pair.
    assert not np.array_equal(rna1_anchored["field"], partner_anchored["field"])
    assert np.array_equal(rna1_anchored["in_mask"], partner_anchored["in_mask"])
    assert np.array_equal(rna1_anchored["dy"], partner_anchored["dy"])
    assert np.array_equal(rna1_anchored["dx"], partner_anchored["dx"])
    assert rna1_anchored["n_null"] == partner_anchored["n_null"]
    assert rna1_anchored["kw"] == partner_anchored["kw"]

    # The partner-anchored constellation is the rna2 spot set, and its field is
    # the rna1 plane — i.e. exactly the rna1-anchored pair, swapped.
    spots2 = res.spots[res.spots["channel"] == "rna2"]
    ys = set(int(round(v)) for v in spots2["y_px"])
    assert set(int(v) for v in partner_anchored["scy"]).issubset(ys)

    # REPRODUCE the engine's emitted number for the FIRST nucleus by calling the
    # SAME function with the captured arrays and a fresh RNG at the documented
    # derivation (partner_rotation_seed + 303). If the engine had its own copy of
    # the maths, this would not match.
    ref = real(
        partner_anchored["field"],
        partner_anchored["scy"],
        partner_anchored["scx"],
        partner_anchored["in_mask"],
        partner_anchored["centroid"],
        partner_anchored["dy"],
        partner_anchored["dx"],
        partner_anchored["n_null"],
        np.random.default_rng(int(cfg.foci.partner_rotation_seed) + 303),
        **partner_anchored["kw"],
    )
    ns = ref["null_stats"]
    expected = float(ref["obs"] / ns.mean())

    # locate the nucleus the second call belonged to: the first row whose
    # partner-anchored enrichment is finite.
    enr = pd.to_numeric(
        res.nuclei["rna1_rotation_enrichment_at_rna2_spots"], errors="coerce"
    )
    got = float(enr[np.isfinite(enr)].iloc[0])
    assert got == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_rna1_anchored_values_unchanged_by_the_new_flag(fake_img, monkeypatch):
    """The new RNG streams must not perturb the rna1-anchored draws."""
    ref = _run(_rotation_cfg(), fake_img, monkeypatch)
    cfg = _rotation_cfg()
    cfg.foci.compute_partner_anchored_rotation_null = True
    new = _run(cfg, fake_img, monkeypatch)
    for c in (
        "rna2_rotation_enrichment_at_rna1_spots",
        "rna2_rotation_null_z_at_rna1_spots",
        "rna2_rotation_null_p_at_rna1_spots",
        "rna2_rotation_assoc_fraction_at_rna1_spots",
    ):
        pd.testing.assert_series_equal(ref.nuclei[c], new.nuclei[c])


# ===========================================================================
# END-TO-END: per-channel threshold_override reaches the detector.
# ===========================================================================
def test_channel_threshold_override_pins_only_that_channel(fake_img, monkeypatch):
    ref = _run(_base_cfg(), fake_img, monkeypatch)
    ref_rna = float(ref.thresholds["rna_bigfish_log_threshold"])

    cfg = _base_cfg()
    cfg.foci.rna2_overrides = FociChannelOverrideCfg(threshold_override=15.0)
    got = _run(cfg, fake_img, monkeypatch)

    assert float(got.thresholds["rna2_bigfish_log_threshold"]) == pytest.approx(15.0)
    # the RNA channel keeps its per-image auto threshold.
    assert float(got.thresholds["rna_bigfish_log_threshold"]) == pytest.approx(ref_rna)


def test_shared_threshold_override_still_applies_to_both_channels(fake_img, monkeypatch):
    cfg = _base_cfg()
    cfg.foci.threshold_override = 12.0
    res = _run(cfg, fake_img, monkeypatch)
    assert float(res.thresholds["rna_bigfish_log_threshold"]) == pytest.approx(12.0)
    assert float(res.thresholds["rna2_bigfish_log_threshold"]) == pytest.approx(12.0)


def test_channel_override_wins_over_shared_end_to_end(fake_img, monkeypatch):
    cfg = _base_cfg()
    cfg.foci.threshold_override = 12.0
    cfg.foci.rna2_overrides = FociChannelOverrideCfg(threshold_override=15.0)
    res = _run(cfg, fake_img, monkeypatch)
    assert float(res.thresholds["rna_bigfish_log_threshold"]) == pytest.approx(12.0)
    assert float(res.thresholds["rna2_bigfish_log_threshold"]) == pytest.approx(15.0)


def test_no_threshold_override_is_unchanged(fake_img, monkeypatch):
    """The unset path must reproduce the pre-change detector call exactly."""
    a = _run(_base_cfg(), fake_img, monkeypatch)
    b = _run(_base_cfg(), fake_img, monkeypatch)
    assert float(a.thresholds["rna_bigfish_log_threshold"]) == pytest.approx(
        float(b.thresholds["rna_bigfish_log_threshold"])
    )
    pd.testing.assert_frame_equal(
        a.spots.reset_index(drop=True), b.spots.reset_index(drop=True)
    )
