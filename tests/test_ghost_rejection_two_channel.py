"""Ghost rejection in a TWO-channel mode must look at both channels.

``nuclei.reject_ghost_nuclei`` deletes "empty shells" — segmented objects that
carry no detected signal AND are large AND are flat in DAPI. All three
conditions are required, and the zero-spot one is the only clean signal in the
audit that produced the rule.

The defect this file covers: rna_rna built the ghost probe from
``nucleus_pre_pass``'s ``spot_count``, which is RNA1 only, because the pre-pass
was handed rna1's coordinates. ``identify_ghost_nuclei`` then asked for
``spots == 0`` and got "zero rna1 spots". So a large flat nucleus with NO RNA1
and ABUNDANT RNA2 was deleted as debris: its row left ``nuclei_df``, its label
was zeroed, and its RNA2 spots were dropped from ``spot_metrics.csv``.

That nucleus is not debris. For a MIAT knockdown against a QKI partner channel it
is the phenotype — a nucleus with no MIAT signal and plenty of partner signal is
the observation the experiment is for. rna_protein delegates here, so the
antibody channel had the same exposure.

The synthetic field below is that case made literal: every nucleus is large and
flat, the RNA1 channel contains NO spots at all, and RNA2 is full of them. Under
the old rna1-only probe every nucleus in the frame satisfied all three conditions
and the run would have returned an empty table.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fishsuite.config.schema import FishsuiteConfig
from fishsuite.core import io as _io
from fishsuite.core import segmentation as _seg
from fishsuite.core.io import ImageWrapper
from fishsuite.core.modes import rna_rna as _rna_rna

DAPI_C, RNA1_C, RNA2_C = 0, 1, 2
NZ = 3
H = W = 200
_CENTERS = [(60, 60), (60, 140), (140, 100)]
_RADIUS = 28


class _FakeBio:
    def __init__(self, czyx: np.ndarray):
        self._czyx = czyx

    def get_image_data(self, order: str, *, T: int = 0, C: int = 0):  # noqa: N803
        assert order == "ZYX"
        return self._czyx[C]


def _flat_dapi_plane():
    """Large, FLAT nuclei: constant inside each disk, so dapi_cv is ~0.

    Flat + large is two of the three ghost conditions. Deliberate — it puts every
    nucleus one condition away from deletion, so the only thing keeping them is
    the spot count, which is what this file is about.
    """
    from skimage.draw import disk

    img = np.full((H, W), 10.0, dtype=np.float32)
    for (cy, cx) in _CENTERS:
        rr, cc = disk((cy, cx), _RADIUS, shape=img.shape)
        img[rr, cc] = 3000.0
    return img


def _empty_rna1_plane():
    """NO spots. A CONSTANT plane — a LoG detector on uniform noise still finds
    hundreds of maxima (measured: 529 on this field), and a fixture that quietly
    has rna1 signal would make the test below pass for the wrong reason."""
    return np.full((H, W), 3.0, dtype=np.float32)


def _punctate_rna2_plane():
    from scipy.ndimage import gaussian_filter

    img = np.random.default_rng(8).uniform(2.0, 6.0, (H, W)).astype(np.float32)
    blob = np.zeros_like(img)
    rng = np.random.default_rng(9)
    for (cy, cx) in _CENTERS:
        for k in range(8):
            ang = 2 * np.pi * k / 8
            y = int(cy + 14 * np.sin(ang))
            x = int(cx + 14 * np.cos(ang))
            blob[y, x] += float(rng.uniform(4000.0, 7000.0))
    return img + gaussian_filter(blob, 1.1)


def _czyx() -> np.ndarray:
    planes = [_flat_dapi_plane(), _empty_rna1_plane(), _punctate_rna2_plane()]
    return np.stack(
        [np.stack([p] * NZ, axis=0) for p in planes], axis=0
    ).astype(np.float32)


@pytest.fixture()
def fake_img() -> ImageWrapper:
    return ImageWrapper(
        path="synthetic_ghost_two_channel.tif",
        bio=_FakeBio(_czyx()),
        scene_idx=0,
        shape=(1, 3, NZ, H, W),
        channel_names=["DAPI", "RNA1", "RNA2"],
        voxel_xy_nm=130.0,
        voxel_z_nm=300.0,
        n_channels=3,
        n_z=NZ,
    )


def _cfg(*, reject_ghosts: bool) -> FishsuiteConfig:
    cfg = FishsuiteConfig()
    cfg.channels.dapi = DAPI_C
    cfg.channels.rna = RNA1_C
    cfg.channels.rna2 = RNA2_C
    cfg.channels.analysis_mode = "rna_rna"
    cfg.nuclei.backend = "otsu"
    cfg.nuclei.min_area_px = 120
    cfg.nuclei.max_area_px = 10_000_000
    cfg.nuclei.exclude_border = True
    cfg.nuclei.border_margin_px = 3
    cfg.nuclei.reject_ghost_nuclei = reject_ghosts
    # Both set so the two non-spot conditions are SATISFIED by this field: the
    # disks are ~2400 px and constant inside. The spot count is then the only
    # thing standing between these nuclei and deletion.
    cfg.nuclei.reject_ghost_min_area_px = 500
    cfg.nuclei.reject_ghost_max_dapi_cv = 0.5
    cfg.z_stack.mode = "maxproj"
    cfg.cytoplasm.enabled = False
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
# THE PREMISE — assert the fixture really is the hard case
# ---------------------------------------------------------------------------
def test_the_field_has_no_rna1_spots_and_plenty_of_rna2(fake_img, monkeypatch):
    res = _run(_cfg(reject_ghosts=False), fake_img, monkeypatch)
    assert int(res.per_image["total_spots_rna1"]) == 0, (
        "the fixture is meant to have NO rna1 signal; without that the test "
        "below passes for the wrong reason"
    )
    assert int(res.per_image["total_spots_rna2"]) > 0
    assert len(res.nuclei) == len(_CENTERS)


def test_the_nuclei_satisfy_the_other_two_ghost_conditions(fake_img, monkeypatch):
    """Large and flat, so only the spot count can save them."""
    res = _run(_cfg(reject_ghosts=False), fake_img, monkeypatch)
    cfg = _cfg(reject_ghosts=True)
    areas = pd.to_numeric(res.nuclei["nucleus_area_px"], errors="coerce")
    assert (areas >= cfg.nuclei.reject_ghost_min_area_px).all()
    # Feed a ZERO spot count — i.e. reproduce the old rna1-only probe exactly —
    # and every nucleus in the frame is condemned.
    probe = pd.DataFrame({
        "nucleus_id": res.nuclei["nucleus_id"],
        "rna_spot_count": 0,
        "dapi_cv": 0.0,
        "nucleus_area_px": areas,
    })
    condemned = _seg.identify_ghost_nuclei(
        probe,
        max_dapi_cv=cfg.nuclei.reject_ghost_max_dapi_cv,
        min_area_px=cfg.nuclei.reject_ghost_min_area_px,
    )
    assert sorted(condemned) == sorted(res.nuclei["nucleus_id"].astype(int))


# ---------------------------------------------------------------------------
# THE FIX
# ---------------------------------------------------------------------------
def test_a_nucleus_with_no_rna1_but_abundant_rna2_is_not_a_ghost(
    fake_img, monkeypatch
):
    """The regression test. Every nucleus here is large, flat and rna1-empty, so
    the old rna1-only probe deleted the entire frame."""
    res = _run(_cfg(reject_ghosts=True), fake_img, monkeypatch)
    assert len(res.nuclei) == len(_CENTERS), (
        "ghost rejection deleted nuclei that carry RNA2 signal — the probe is "
        "keyed on RNA1 alone again"
    )


def test_the_rna2_spots_of_such_a_nucleus_survive(fake_img, monkeypatch):
    """Deleting the nucleus also dropped its rows from spot_metrics.csv, so the
    measurement disappeared, not just the label."""
    off = _run(_cfg(reject_ghosts=False), fake_img, monkeypatch)
    on = _run(_cfg(reject_ghosts=True), fake_img, monkeypatch)
    n_rna2 = lambda r: int((r.spots["channel"] == "rna2").sum())  # noqa: E731
    assert n_rna2(on) == n_rna2(off) > 0
    assert int(on.per_image["total_spots_rna2"]) == int(
        off.per_image["total_spots_rna2"]
    )


def test_ghost_rejection_off_is_the_baseline(fake_img, monkeypatch):
    """Default OFF must be untouched by any of this."""
    off = _run(_cfg(reject_ghosts=False), fake_img, monkeypatch)
    assert len(off.nuclei) == len(_CENTERS)


def test_the_probe_is_the_total_over_channels_not_channel_one(
    fake_img, monkeypatch
):
    """Bind the CALL SITE, not just the outcome: capture the probe rna_rna hands
    to identify_ghost_nuclei and check both the column it names and its values.
    Without this a future edit could revert to the rna1-only count and only the
    behavioural test above would notice — and only for this fixture's shape."""
    seen = {}
    real = _seg.identify_ghost_nuclei

    def _spy(probe, **kw):
        seen["probe"] = probe.copy()
        seen["kw"] = dict(kw)
        return real(probe, **kw)

    monkeypatch.setattr(_rna_rna._seg, "identify_ghost_nuclei", _spy)
    _run(_cfg(reject_ghosts=True), fake_img, monkeypatch)

    assert seen["kw"].get("spot_count_col") == "total_spot_count"
    probe = seen["probe"]
    assert "total_spot_count" in probe.columns
    # rna1 contributes nothing in this fixture, so any nonzero total came from
    # channel 2 — which is the whole point.
    assert float(probe["total_spot_count"].sum()) > 0


# ---------------------------------------------------------------------------
# PER-IMAGE TOTALS MUST MATCH THE ROWS ACTUALLY WRITTEN
#
# total_spots1/2 were computed from the UNFILTERED spots1_df / spots2_df while
# spots_out_df was ghost-filtered, so per_image_summary could report more RNA2
# spots than spot_metrics.csv contains. Forced here by condemning a nucleus that
# genuinely carries spots — the condition the defect needs, which a real ghost
# (zero spots by definition) can never produce.
# ---------------------------------------------------------------------------
def _in_cell_written(res, channel: str) -> int:
    """Rows in spot_metrics.csv for ``channel`` that lie in a cell.

    `total_spots_rna*` counts in-cell spots only (in_nucleus OR in_cytoplasm) —
    floaters stay in the table for audit — so the comparison has to use the same
    definition or it fails on a difference that is intentional.
    """
    s = res.spots
    sub = s.loc[s["channel"] == channel]
    return int(
        (sub["in_nucleus"].astype(bool) | sub["in_cytoplasm"].astype(bool)).sum()
    )


def test_per_image_totals_match_the_written_spot_rows_after_a_ghost_drop(
    fake_img, monkeypatch
):
    monkeypatch.setattr(
        _rna_rna._seg, "identify_ghost_nuclei", lambda probe, **kw: [1]
    )
    res = _run(_cfg(reject_ghosts=True), fake_img, monkeypatch)

    assert len(res.nuclei) == len(_CENTERS) - 1, "the forced ghost was not dropped"
    assert int(res.per_image["total_spots_rna2"]) == _in_cell_written(res, "rna2"), (
        "per_image_summary reports more RNA2 spots than spot_metrics.csv holds"
    )
    assert int(res.per_image["total_spots_rna1"]) == _in_cell_written(res, "rna1")
    assert int(res.per_image["total_spots_rna2"]) > 0, (
        "the forced ghost carried spots, so some were dropped — the totals had to "
        "follow"
    )


def test_the_totals_drop_by_exactly_the_forced_ghosts_spots(fake_img, monkeypatch):
    """Not merely consistent — consistent at the RIGHT number. An unfiltered
    total and a filtered table can also be reconciled by breaking both."""
    baseline = _run(_cfg(reject_ghosts=False), fake_img, monkeypatch)
    nuc1 = baseline.spots.loc[
        (baseline.spots["channel"] == "rna2")
        & (baseline.spots["nucleus_id"] == 1)
    ]
    n_lost = int(
        (nuc1["in_nucleus"].astype(bool) | nuc1["in_cytoplasm"].astype(bool)).sum()
    )
    assert n_lost > 0, "nucleus 1 must carry rna2 spots for this to test anything"

    monkeypatch.setattr(
        _rna_rna._seg, "identify_ghost_nuclei", lambda probe, **kw: [1]
    )
    after = _run(_cfg(reject_ghosts=True), fake_img, monkeypatch)
    assert int(after.per_image["total_spots_rna2"]) == \
        int(baseline.per_image["total_spots_rna2"]) - n_lost


# ---------------------------------------------------------------------------
# THE ID-SPACE DEFECT UNDERNEATH THE ABOVE (2026-08-10)
#
# The three filters in this section (speck, floater, ghost) each mirrored their
# spots_out_df result back onto spots1_df / spots2_df by intersecting `spot_id`.
# But spots_out_df's spot_id is a 1-based GLOBAL running id assigned at emit
# time, while the source frames carry the detector's per-channel 0-based index.
# Intersecting them is meaningless, and for rna2 the two ranges are normally
# disjoint — so with the DEFAULT drop_floater_spots=True, `total_spots_rna2` in
# per_image_summary.csv was structurally 0 and `total_spots_rna1` was a wrong
# number. Measured on this fixture before the fix: reported rna1=375 / rna2=0
# against 529 / 24 rows actually written.
# ---------------------------------------------------------------------------
def test_totals_are_consistent_under_the_default_floater_filter(
    fake_img, monkeypatch
):
    cfg = _cfg(reject_ghosts=False)
    cfg.foci.drop_floater_spots = True          # the DEFAULT
    res = _run(cfg, fake_img, monkeypatch)
    assert int(res.per_image["total_spots_rna2"]) == _in_cell_written(res, "rna2") > 0
    assert int(res.per_image["total_spots_rna1"]) == _in_cell_written(res, "rna1")


def test_totals_are_consistent_under_the_speck_filter(fake_img, monkeypatch):
    cfg = _cfg(reject_ghosts=False)
    cfg.foci.max_peak_over_p95_ratio = 1.05     # bites, so the mirror has to run
    res = _run(cfg, fake_img, monkeypatch)
    assert int(res.per_image["total_spots_rna2"]) == _in_cell_written(res, "rna2")
    assert int(res.per_image["total_spots_rna1"]) == _in_cell_written(res, "rna1")


def test_a_mirror_never_reads_an_already_mirrored_frame(fake_img, monkeypatch):
    """Two filters in one run. The recorded positions index the ORIGINAL frames,
    so mirroring a mirror would misalign them and silently keep wrong rows."""
    cfg = _cfg(reject_ghosts=False)
    cfg.foci.drop_floater_spots = True
    cfg.foci.max_peak_over_p95_ratio = 1.05
    res = _run(cfg, fake_img, monkeypatch)
    assert int(res.per_image["total_spots_rna2"]) == _in_cell_written(res, "rna2")
    assert int(res.per_image["total_spots_rna1"]) == _in_cell_written(res, "rna1")


# ---------------------------------------------------------------------------
# THE SHARED HELPER
# ---------------------------------------------------------------------------
def test_spot_counts_per_label_counts_a_second_channel_off_the_same_labels():
    labels = np.zeros((20, 20), dtype=np.uint16)
    labels[2:8, 2:8] = 1
    labels[12:18, 12:18] = 2
    spots = pd.DataFrame({"y_px": [4.0, 5.0, 14.0], "x_px": [4.0, 5.0, 14.0]})
    counts = _seg.spot_counts_per_label(labels, *_seg.spot_xy_columns(spots))
    assert counts == {1: 2, 2: 1}


def test_spot_counts_per_label_is_empty_not_wrong_without_coordinates():
    """Returning zeros for every label is what makes the ghost rule fire on real
    nuclei, so 'no coordinates' must be distinguishable from 'no spots'."""
    labels = np.ones((4, 4), dtype=np.uint16)
    assert _seg.spot_counts_per_label(labels, None, None) == {}
    assert _seg.spot_counts_per_label(labels, [], []) == {}
    assert _seg.spot_counts_per_label(None, [1.0], [1.0]) == {}


def test_pre_pass_still_agrees_with_the_extracted_helper():
    """The counting moved out of nucleus_pre_pass; it must not have changed."""
    labels = np.zeros((20, 20), dtype=np.uint16)
    labels[2:8, 2:8] = 1
    dapi = np.random.default_rng(3).uniform(100, 200, (20, 20))
    spots = pd.DataFrame({"y_px": [4.0, 5.0], "x_px": [4.0, 5.0]})
    y, x = _seg.spot_xy_columns(spots)
    pre = _seg.nucleus_pre_pass(labels, dapi, spot_y=y, spot_x=x)
    assert pre[1]["spot_count"] == _seg.spot_counts_per_label(labels, y, x)[1] == 2
