"""Fixed-N nucleus sampling.

Quantifying the SAME number of nuclei in every field of view is what makes two
conditions comparable on an identical denominator. These tests lock the three
properties that make the feature trustworthy:

  * the draw is reproducible and independent of processing order,
  * sampling runs LAST, after every quality filter, and
  * with the feature off, nothing changes at all.

Everything is built from synthetic arrays — no image fixture is committed.

NOTE ON NAMING: this file is about NUCLEUS selection. ``test_focus_window_fixed_n``
is a different feature entirely (the z-focus window).
"""
from __future__ import annotations

import numpy as np
import pytest

from fishsuite.config.schema import FishsuiteConfig, SamplingCfg
from fishsuite.core import segmentation as _seg


# ---------------------------------------------------------------------------
# Synthetic field helpers
# ---------------------------------------------------------------------------

GRID_ROWS, GRID_COLS = 4, 3
FIELD_H = FIELD_W = 200
_ROW_Y = [20 + r * 45 for r in range(GRID_ROWS)]
_COL_X = [20 + c * 55 for c in range(GRID_COLS)]
_SIDE = 20


def _synthetic_field(*, side=_SIDE):
    """A 4x3 grid of square nuclei with KNOWN centroids, labelled row-major.

    Label k sits at row ``(k-1)//3``, column ``(k-1)%3``, so the expected raster
    order is simply 1, 2, 3, ... and the geometry of every ordering rule can be
    asserted exactly rather than approximately.
    """
    labels = np.zeros((FIELD_H, FIELD_W), dtype=np.int32)
    dapi = np.zeros((FIELD_H, FIELD_W), dtype=np.float32)
    centroids = {}
    nid = 0
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            nid += 1
            y0, x0 = _ROW_Y[r], _COL_X[c]
            labels[y0:y0 + side, x0:x0 + side] = nid
            # Distinct mean per nucleus, plus one hot pixel so dapi_cv > 0.
            dapi[y0:y0 + side, x0:x0 + side] = 100.0 + nid * 5.0
            dapi[y0, x0] = 900.0
            centroids[nid] = (y0 + (side - 1) / 2.0, x0 + (side - 1) / 2.0)
    return labels, dapi, centroids


def _centre_distance(centroids, nid):
    cy0 = (FIELD_H - 1) / 2.0
    cx0 = (FIELD_W - 1) / 2.0
    y, x = centroids[nid]
    return (y - cy0) ** 2 + (x - cx0) ** 2


# ---------------------------------------------------------------------------
# Pre-pass
# ---------------------------------------------------------------------------

def test_pre_pass_reports_geometry_dapi_and_spot_counts():
    labels, dapi, centroids = _synthetic_field()
    # Two spots inside nucleus 1, one inside nucleus 7, one on background.
    pre = _seg.nucleus_pre_pass(
        labels, dapi,
        spot_y=[_ROW_Y[0] + 5, _ROW_Y[0] + 6, _ROW_Y[2] + 5, 5],
        spot_x=[_COL_X[0] + 5, _COL_X[0] + 6, _COL_X[0] + 5, 5],
    )
    assert set(pre) == set(range(1, GRID_ROWS * GRID_COLS + 1))
    assert pre[1]["area"] == float(_SIDE * _SIDE)
    assert pre[1]["centroid_y"] == pytest.approx(centroids[1][0])
    assert pre[1]["centroid_x"] == pytest.approx(centroids[1][1])
    assert pre[1]["spot_count"] == 2
    assert pre[7]["spot_count"] == 1
    assert pre[2]["spot_count"] == 0
    # dapi_cv must be finite and positive (the hot pixel guarantees spread).
    assert pre[1]["dapi_cv"] > 0


def test_spot_coordinate_columns_resolve_under_either_naming():
    """``y_px``/``x_px`` is canonical; bare ``y``/``x`` appears downstream.

    Getting this wrong is silent and consequential: finding neither name yields
    zero spots for every nucleus, and the ghost rule (zero spots AND large AND
    flat) would then start firing on real nuclei. Caught exactly that way during
    development, hence this test.
    """
    import pandas as pd

    canonical = pd.DataFrame({"y_px": [1.0, 2.0], "x_px": [3.0, 4.0]})
    y, x = _seg.spot_xy_columns(canonical)
    assert list(y) == [1.0, 2.0] and list(x) == [3.0, 4.0]

    bare = pd.DataFrame({"y": [5.0], "x": [6.0]})
    y, x = _seg.spot_xy_columns(bare)
    assert list(y) == [5.0] and list(x) == [6.0]

    # No coordinates at all -> (None, None), never a silent empty array.
    assert _seg.spot_xy_columns(pd.DataFrame({"spot_id": [1]})) == (None, None)
    assert _seg.spot_xy_columns(pd.DataFrame()) == (None, None)
    assert _seg.spot_xy_columns(None) == (None, None)


def test_ghost_rule_sees_real_spot_counts_through_the_resolver():
    """End-to-end of the bug above: a spot-bearing nucleus is not a ghost."""
    import pandas as pd

    labels, dapi, _ = _synthetic_field()
    # Flat, large nucleus 1 -> a ghost candidate on area+texture alone.
    dapi[labels == 1] = 500.0
    spots = pd.DataFrame({
        "y_px": [_ROW_Y[0] + 5.0], "x_px": [_COL_X[0] + 5.0],
    })
    y, x = _seg.spot_xy_columns(spots)
    pre = _seg.nucleus_pre_pass(labels, dapi, spot_y=y, spot_x=x)
    assert pre[1]["spot_count"] == 1
    ghosts = _seg.identify_ghost_nuclei(
        pd.DataFrame([
            {"nucleus_id": k, "rna_spot_count": v["spot_count"],
             "dapi_cv": v["dapi_cv"], "nucleus_area_px": v["area"]}
            for k, v in sorted(pre.items())
        ]),
        max_dapi_cv=0.12, min_area_px=100,
    )
    # Nucleus 1 carries a spot, so it is NOT a ghost despite being flat + large.
    assert 1 not in ghosts


def test_pre_pass_dapi_cv_matches_chromatin_metrics_definition():
    """The ghost rule is evaluated on both tables, so one definition only.

    ``nucleolus.chromatin_metrics_per_nucleus`` uses std/mean with ddof=0. If the
    pre-pass drifted to ddof=1, a nucleus could be a ghost in one place and not
    the other.
    """
    labels, dapi, _ = _synthetic_field()
    pre = _seg.nucleus_pre_pass(labels, dapi)
    vals = dapi[labels == 1].astype(np.float64)
    assert pre[1]["dapi_cv"] == pytest.approx(float(vals.std()) / float(vals.mean()))


def test_pre_pass_omits_labels_absent_from_the_mask():
    """A label zeroed by the area filter must not even be a candidate."""
    labels, dapi, _ = _synthetic_field()
    labels[labels == 5] = 0
    pre = _seg.nucleus_pre_pass(labels, dapi)
    assert 5 not in pre
    assert len(pre) == GRID_ROWS * GRID_COLS - 1


# ---------------------------------------------------------------------------
# Ordering rules
# ---------------------------------------------------------------------------

def test_raster_order_is_row_major_by_centroid():
    labels, dapi, _ = _synthetic_field()
    pre = _seg.nucleus_pre_pass(labels, dapi)
    res = _seg.resolve_nucleus_sampling(
        pre, n_target=5, unit_key="u", seed=1, order="raster",
        field_shape=labels.shape,
    )
    assert res.ordered_ids == list(range(1, 13))
    # The literal "first 5" are the top-left five — which is exactly the bias:
    # every selected nucleus comes from the top two rows of the frame.
    assert res.selected_ids == [1, 2, 3, 4, 5]


def test_center_out_order_is_ascending_distance_from_field_centre():
    labels, dapi, centroids = _synthetic_field()
    pre = _seg.nucleus_pre_pass(labels, dapi)
    res = _seg.resolve_nucleus_sampling(
        pre, n_target=4, unit_key="u", seed=1, order="center_out",
        field_shape=labels.shape,
    )
    expected = sorted(range(1, 13), key=lambda i: _centre_distance(centroids, i))
    assert res.ordered_ids == expected
    # And it is centre-biased by construction: no corner nucleus is picked.
    corners = {1, 3, 10, 12}
    assert not (set(res.selected_ids) & corners)


def test_random_order_is_a_permutation_and_reproducible_under_a_fixed_seed():
    labels, dapi, _ = _synthetic_field()
    pre = _seg.nucleus_pre_pass(labels, dapi)
    kw = dict(n_target=4, unit_key="plate/A01.tif", order="random",
              field_shape=labels.shape)
    a = _seg.resolve_nucleus_sampling(pre, seed=42, **kw)
    b = _seg.resolve_nucleus_sampling(pre, seed=42, **kw)
    assert sorted(a.ordered_ids) == list(range(1, 13))
    assert a.ordered_ids == b.ordered_ids
    assert a.selected_ids == b.selected_ids
    # A different seed must actually move the draw.
    c = _seg.resolve_nucleus_sampling(pre, seed=43, **kw)
    assert c.ordered_ids != a.ordered_ids


def test_each_order_is_deterministic_under_a_fixed_seed():
    labels, dapi, _ = _synthetic_field()
    pre = _seg.nucleus_pre_pass(labels, dapi)
    for order in ("random", "raster", "center_out"):
        runs = {
            tuple(
                _seg.resolve_nucleus_sampling(
                    pre, n_target=6, unit_key="u", seed=7, order=order,
                    field_shape=labels.shape,
                ).selected_ids
            )
            for _ in range(5)
        }
        assert len(runs) == 1, f"{order} is not deterministic: {runs}"


def test_unknown_order_is_rejected():
    labels, dapi, _ = _synthetic_field()
    pre = _seg.nucleus_pre_pass(labels, dapi)
    with pytest.raises(ValueError, match="unknown sampling order"):
        _seg.order_nuclei(pre, list(pre), order="dapi_rank")


def test_intensity_and_area_orders_are_not_offered():
    """Ordering on DAPI intensity or area would be selection-on-the-outcome.

    Both track cell-cycle stage, which correlates with total RNA — i.e. with the
    reported readout. They are excluded on purpose; this test is the tripwire
    against someone adding them back for convenience.
    """
    allowed = set(SamplingCfg.model_fields["order"].annotation.__args__)
    assert allowed == {"random", "raster", "center_out"}
    assert not {"dapi_rank", "area_rank"} & allowed


# ---------------------------------------------------------------------------
# Selection semantics
# ---------------------------------------------------------------------------

def test_sampling_runs_after_filtering_so_a_failing_nucleus_is_never_selected():
    """area -> border -> ghost -> sample, with no exceptions.

    Nucleus 4 is removed from the mask (as the area/border filters do, by
    zeroing) and nucleus 6 is a ghost. Neither may appear in the sample even
    when N is large enough to take everything else.
    """
    labels, dapi, _ = _synthetic_field()
    labels[labels == 4] = 0                      # failed area/border
    pre = _seg.nucleus_pre_pass(labels, dapi)
    res = _seg.resolve_nucleus_sampling(
        pre, n_target=99, unit_key="u", seed=1, order="raster",
        ghost_ids=[6],                           # failed ghost rejection
        field_shape=labels.shape,
    )
    assert 4 not in res.selected_ids and 4 not in res.eligible_ids
    assert 6 not in res.selected_ids and 6 not in res.eligible_ids
    assert res.n_eligible == GRID_ROWS * GRID_COLS - 2
    assert res.n_ghost_excluded == 1


def test_n_above_eligible_count_takes_everything_and_flags_short():
    labels, dapi, _ = _synthetic_field()
    pre = _seg.nucleus_pre_pass(labels, dapi)
    res = _seg.resolve_nucleus_sampling(
        pre, n_target=500, unit_key="u", seed=1, field_shape=labels.shape,
    )
    assert res.n_sampled == res.n_eligible == GRID_ROWS * GRID_COLS
    assert res.short_of_target is True
    assert res.unit_dropped is False


def test_not_short_when_exactly_on_target():
    labels, dapi, _ = _synthetic_field()
    pre = _seg.nucleus_pre_pass(labels, dapi)
    res = _seg.resolve_nucleus_sampling(
        pre, n_target=GRID_ROWS * GRID_COLS, unit_key="u", seed=1,
        field_shape=labels.shape,
    )
    assert res.short_of_target is False
    assert res.n_sampled == GRID_ROWS * GRID_COLS


def test_on_short_fail_raises():
    labels, dapi, _ = _synthetic_field()
    pre = _seg.nucleus_pre_pass(labels, dapi)
    with pytest.raises(ValueError, match="on_short='fail'"):
        _seg.resolve_nucleus_sampling(
            pre, n_target=99, unit_key="wellA/img1.tif", seed=1,
            on_short="fail", field_shape=labels.shape,
        )


def test_on_short_drop_unit_excludes_the_unit_and_records_it():
    labels, dapi, _ = _synthetic_field()
    pre = _seg.nucleus_pre_pass(labels, dapi)
    res = _seg.resolve_nucleus_sampling(
        pre, n_target=99, unit_key="u", seed=1, on_short="drop_unit",
        field_shape=labels.shape,
    )
    assert res.selected_ids == []
    assert res.n_sampled == 0
    assert res.unit_dropped is True
    assert res.short_of_target is True
    # The nuclei are still ELIGIBLE and still ranked — the unit was dropped, the
    # nuclei were not silently deleted.
    assert res.n_eligible == GRID_ROWS * GRID_COLS
    assert len(res.rank) == GRID_ROWS * GRID_COLS
    cols = _seg.sampling_per_image_cols(res)
    assert cols["sampling_unit_dropped"] is True
    assert cols["n_nuclei_sampled"] == 0


def test_rank_covers_every_eligible_nucleus_so_n_is_revisitable():
    labels, dapi, _ = _synthetic_field()
    pre = _seg.nucleus_pre_pass(labels, dapi)
    res = _seg.resolve_nucleus_sampling(
        pre, n_target=5, unit_key="u", seed=3, field_shape=labels.shape,
    )
    assert sorted(res.rank.values()) == list(range(1, 13))
    # selected == the first n_target ranks, so a different N can be recovered
    # from the recorded ranks without re-running the analysis.
    assert set(res.selected_ids) == {n for n, r in res.rank.items() if r <= 5}


def test_min_eligible_only_annotates_and_never_excludes():
    """`conditions.min_nuclei_for_stats` is step-1 ANNOTATION, not enforcement."""
    labels, dapi, _ = _synthetic_field()
    pre = _seg.nucleus_pre_pass(labels, dapi)
    res = _seg.resolve_nucleus_sampling(
        pre, n_target=4, unit_key="u", seed=1, min_eligible=999,
        field_shape=labels.shape,
    )
    assert res.included_in_stats is False      # annotated as below the bar ...
    assert res.n_sampled == 4                  # ... and still fully quantified
    assert _seg.sampling_per_image_cols(res)["image_included_in_stats"] is False


def test_per_image_cols_itemise_the_whole_filter_chain():
    labels, dapi, _ = _synthetic_field()
    pre = _seg.nucleus_pre_pass(labels, dapi)
    res = _seg.resolve_nucleus_sampling(
        pre, n_target=4, unit_key="w/img.tif", seed=11, order="raster",
        ghost_ids=[2], field_shape=labels.shape,
        n_segmented=20, n_area_excluded=5, n_border_excluded=3,
    )
    cols = _seg.sampling_per_image_cols(res, n_ghost_excluded=1)
    assert cols["n_nuclei_segmented"] == 20
    assert cols["n_nuclei_area_excluded"] == 5
    assert cols["n_nuclei_ghost_excluded"] == 1
    assert cols["n_nuclei_eligible"] == 11
    assert cols["n_nuclei_sampled"] == 4
    assert cols["sampling_unit"] == "w/img.tif"
    assert cols["sampling_order"] == "raster"
    assert cols["sampling_seed_used"] == 11
    # n_nuclei_border_excluded is emitted by the modes and must NOT be
    # duplicated here — two columns could disagree.
    assert "n_nuclei_border_excluded" not in cols


def test_empty_field_selects_nothing_without_raising():
    labels = np.zeros((50, 50), dtype=np.int32)
    dapi = np.zeros((50, 50), dtype=np.float32)
    pre = _seg.nucleus_pre_pass(labels, dapi)
    assert pre == {}
    res = _seg.resolve_nucleus_sampling(
        pre, n_target=20, unit_key="u", seed=1, field_shape=labels.shape,
    )
    assert res.selected_ids == [] and res.n_eligible == 0


def test_unknown_segmentation_counts_are_nan_not_zero():
    """A count we cannot know must not be reported as 0.

    On the batch-threshold path the pre-scan hands the mode labels it already
    segmented and area-filtered, so those two counts are unrecoverable. Writing
    0 would assert that nothing was filtered — a false statement in a
    provenance column.
    """
    import math

    labels, dapi, _ = _synthetic_field()
    pre = _seg.nucleus_pre_pass(labels, dapi)
    res = _seg.resolve_nucleus_sampling(
        pre, n_target=4, unit_key="u", seed=1, field_shape=labels.shape,
        n_segmented=None, n_area_excluded=None, n_border_excluded=0,
    )
    cols = _seg.sampling_per_image_cols(res)
    assert math.isnan(cols["n_nuclei_segmented"])
    assert math.isnan(cols["n_nuclei_area_excluded"])
    # The counts that ARE knowable stay exact.
    assert cols["n_nuclei_eligible"] == 12
    assert cols["n_nuclei_sampled"] == 4


def test_empty_field_still_yields_the_full_per_image_key_set():
    """A zero-nucleus image must not produce a ragged per_image_summary.

    If an empty field emitted no sampling columns while its neighbours did, the
    CSV would carry blanks for some rows and values for others — the same
    image-invariant-schema requirement the gated coloc rollups already observe.
    """
    labels = np.zeros((60, 60), dtype=np.int32)
    dapi = np.zeros((60, 60), dtype=np.float32)
    empty = _seg.resolve_nucleus_sampling(
        _seg.nucleus_pre_pass(labels, dapi), n_target=20, unit_key="e", seed=1,
        field_shape=labels.shape,
    )
    labels2, dapi2, _ = _synthetic_field()
    full = _seg.resolve_nucleus_sampling(
        _seg.nucleus_pre_pass(labels2, dapi2), n_target=4, unit_key="f", seed=1,
        field_shape=labels2.shape,
    )
    assert set(_seg.sampling_per_image_cols(empty)) == \
        set(_seg.sampling_per_image_cols(full))
    cols = _seg.sampling_per_image_cols(empty)
    assert cols["n_nuclei_eligible"] == 0
    assert cols["n_nuclei_sampled"] == 0
    assert cols["sampling_short_of_target"] is True


def test_sampler_does_not_renumber_labels():
    """Downstream code assumes dense IDs 1..N; the sampler must not touch them.

    ``exclude_border_labels`` renumbers contiguously and the per-nucleus loops
    iterate ``range(1, n_after + 1)``, so relabelling in the sampler would
    silently corrupt every ID-keyed join.
    """
    labels, dapi, _ = _synthetic_field()
    before = labels.copy()
    pre = _seg.nucleus_pre_pass(labels, dapi)
    _seg.resolve_nucleus_sampling(
        pre, n_target=4, unit_key="u", seed=1, field_shape=labels.shape,
    )
    assert np.array_equal(labels, before)


# ---------------------------------------------------------------------------
# Seed derivation / order independence
# ---------------------------------------------------------------------------

def test_draw_is_keyed_on_the_unit_not_on_consumption_order():
    """The decisive reproducibility property.

    Images are handed to a worker pool, so completion order is nondeterministic.
    Deriving each unit's generator from a hash of the unit's own key — rather
    than pulling from one shared stream — is what makes the sample independent
    of scheduling. Here: resolving the units in REVERSE order must not change
    any unit's draw.
    """
    labels, dapi, _ = _synthetic_field()
    pre = _seg.nucleus_pre_pass(labels, dapi)
    keys = [f"cond/img{i:02d}.tif" for i in range(8)]

    forward = {
        k: _seg.resolve_nucleus_sampling(
            pre, n_target=4, unit_key=k, seed=5, field_shape=labels.shape,
        ).selected_ids
        for k in keys
    }
    backward = {
        k: _seg.resolve_nucleus_sampling(
            pre, n_target=4, unit_key=k, seed=5, field_shape=labels.shape,
        ).selected_ids
        for k in reversed(keys)
    }
    assert forward == backward
    # Different units must still get genuinely different draws, or "N per FOV"
    # would be the same N nuclei everywhere.
    assert len({tuple(v) for v in forward.values()}) > 1


def test_unit_rng_is_stable_across_processes():
    """Same key + seed must give the same stream in a *different* interpreter.

    A hash-derived SeedSequence is only order-independent if the hash itself is
    stable across processes. Python's ``hash()`` of a str is salted per process
    and would silently fail this; blake2b does not.
    """
    import subprocess
    import sys

    code = (
        "from fishsuite.core.segmentation import derive_unit_rng;"
        "print(derive_unit_rng(5, 'cond/img03.tif').integers(0, 10**9, 4).tolist())"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True,
    ).stdout.strip()
    local = _seg.derive_unit_rng(5, "cond/img03.tif").integers(0, 10 ** 9, 4).tolist()
    assert out == str(local)


def test_per_well_allocation_is_equal_and_order_independent():
    """A well's N is split across its images, never drawn flat from the pool.

    Drawing flat would let one crowded field of view supply most of the well's
    sample — the variable denominator this feature exists to remove.
    """
    keys = ["w/img3.tif", "w/img1.tif", "w/img2.tif"]
    alloc = _seg.allocate_per_unit(keys, 20)
    assert sum(alloc.values()) == 20
    assert sorted(alloc.values()) == [6, 7, 7]
    # Remainder assignment follows SORTED key order, not argument order.
    assert _seg.allocate_per_unit(list(reversed(keys)), 20) == alloc
    # Exact division leaves every image equal.
    assert set(_seg.allocate_per_unit(keys, 21).values()) == {7}
    assert _seg.allocate_per_unit([], 20) == {}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_sampling_defaults_off_so_existing_presets_are_untouched():
    cfg = FishsuiteConfig()
    assert cfg.sampling.enabled is False
    assert cfg.sampling.n_per_unit == 20
    assert cfg.sampling.unit == "per_image"
    assert cfg.sampling.order == "random"          # the unbiased default
    assert cfg.sampling.seed is None
    assert cfg.sampling.on_short == "keep"
    assert cfg.sampling.min_eligible == 0
    assert cfg.sampling.apply_to_rollups is True


def test_legacy_config_without_a_sampling_block_still_loads(tmp_path):
    p = tmp_path / "legacy.yaml"
    p.write_text("experiment:\n  name: legacy\nnuclei:\n  backend: otsu\n",
                 encoding="utf-8")
    cfg = FishsuiteConfig.from_yaml(p)
    assert cfg.sampling.enabled is False
    assert cfg.resolved_sampling()["enabled"] is False


def test_resolved_sampling_substitutes_the_inherited_values():
    cfg = FishsuiteConfig()
    cfg.seed = 1234
    cfg.conditions.min_nuclei_for_stats = 9
    r = cfg.resolved_sampling()
    assert r["seed"] == 1234 and r["seed_inherited_from_top_level"] is True
    assert r["min_eligible"] == 9 and r["min_eligible_inherited"] is True
    # Explicit values win over inheritance.
    cfg.sampling.seed = 7
    cfg.sampling.min_eligible = 3
    r = cfg.resolved_sampling()
    assert r["seed"] == 7 and r["seed_inherited_from_top_level"] is False
    assert r["min_eligible"] == 3 and r["min_eligible_inherited"] is False


def test_order_help_text_names_which_options_are_biased():
    """The bias must be stated where the knob is set, not only in a doc."""
    desc = SamplingCfg.model_fields["order"].description.lower()
    assert "unbiased" in desc          # random is labelled as the unbiased one
    assert "biased" in desc            # raster / center_out are labelled biased
    assert "raster" in desc and "center_out" in desc


def test_apply_to_rollups_help_text_records_the_threshold_exception():
    desc = SamplingCfg.model_fields["apply_to_rollups"].description.lower()
    assert "threshold" in desc and "all pooled pixels" in desc


# ---------------------------------------------------------------------------
# Methods sentence
# ---------------------------------------------------------------------------

def test_methods_text_is_generated_from_the_resolved_config():
    from fishsuite.runner import _sampling_methods_text

    cfg = FishsuiteConfig()
    cfg.nuclei.backend = "otsu"
    cfg.sampling.enabled = True
    cfg.sampling.n_per_unit = 20
    cfg.sampling.seed = 99
    txt = _sampling_methods_text(cfg, cfg.resolved_sampling(), None)
    assert "20" in txt
    assert "otsu" in txt
    assert "without replacement" in txt.lower()
    assert "PCG64" in txt
    assert "seed 99" in txt
    assert "independent of the order in which images were processed" in txt
    # Must state that selection saw none of the readout.
    assert "no information from the analysis channels" in txt.lower()
    # Must NOT attribute the per-FOV rule to a paper.
    for word in ("khong", "parker", "tauber", "molecular cell", "et al"):
        assert word not in txt.lower()


def test_methods_text_labels_the_biased_orders_as_biased():
    from fishsuite.runner import _sampling_methods_text

    cfg = FishsuiteConfig()
    cfg.sampling.enabled = True
    for order in ("raster", "center_out"):
        cfg.sampling.order = order
        txt = _sampling_methods_text(cfg, cfg.resolved_sampling(), None)
        assert "not an unbiased sample" in txt.lower(), order


def test_methods_text_states_what_happened_to_short_units():
    from fishsuite.runner import _sampling_methods_text

    cfg = FishsuiteConfig()
    cfg.sampling.enabled = True
    for on_short, needle in (
        ("keep", "contributed all of"),
        ("drop_unit", "excluded"),
        ("fail", "abort"),
    ):
        cfg.sampling.on_short = on_short
        txt = _sampling_methods_text(cfg, cfg.resolved_sampling(), None)
        assert needle in txt.lower(), on_short


def test_methods_text_records_the_filters_that_ran():
    from fishsuite.runner import _sampling_methods_text

    cfg = FishsuiteConfig()
    cfg.sampling.enabled = True
    cfg.nuclei.min_area_px = 4321
    cfg.nuclei.exclude_border = True
    cfg.nuclei.border_margin_px = 8
    cfg.nuclei.reject_ghost_nuclei = True
    txt = _sampling_methods_text(cfg, cfg.resolved_sampling(), None)
    assert "4321" in txt
    assert "border" in txt.lower() and "8" in txt
    assert "ghost" in txt.lower()


# ---------------------------------------------------------------------------
# Runner plan
# ---------------------------------------------------------------------------

class _FakeImg:
    def __init__(self, path, condition="A", sec_only=False):
        self.path = path
        self.condition = condition
        self.sec_only = sec_only


def test_sampling_kwargs_are_only_sent_to_modes_that_accept_them():
    """Guard against a TypeError on every image of an rna_protein run.

    The runner forwards `sampling_unit_key` / `sampling_n_alloc`, but only
    rna_only and rna_rna declare them. rna_protein.run_one is keyword-only with
    no **kwargs, so forwarding to it would raise TypeError per image. It reaches
    the sampler anyway by delegating to rna_rna.run_one, which supplies its own
    defaults.
    """
    import inspect

    from fishsuite.core.modes import rna_only, rna_protein, rna_rna

    for mod in (rna_only, rna_rna):
        params = inspect.signature(mod.run_one).parameters
        assert "sampling_unit_key" in params, mod.__name__
        assert "sampling_n_alloc" in params, mod.__name__
        # Optional, so every existing caller keeps working untouched.
        assert params["sampling_unit_key"].default is None
        assert params["sampling_n_alloc"].default is None

    rp = inspect.signature(rna_protein.run_one).parameters
    accepts_kwargs = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in rp.values()
    )
    if not accepts_kwargs and "sampling_unit_key" not in rp:
        # Then the runner MUST NOT forward to it. Read the guard back out of the
        # source so this test fails if someone widens the mode list.
        from pathlib import Path

        import fishsuite.runner as _r

        src = Path(_r.__file__).read_text(encoding="utf-8")
        assert 'in ("rna_only", "rna_rna"):\n                    _plan' in src, (
            "runner forwards sampling kwargs to a mode whose run_one cannot "
            "accept them"
        )


def test_plan_keys_are_input_relative_so_the_draw_survives_a_move(tmp_path):
    from fishsuite.runner import _resolve_sampling_plan

    imgs = [_FakeImg(tmp_path / "WellA" / "img01.tif"),
            _FakeImg(tmp_path / "WellA" / "img02.tif")]
    plan = _resolve_sampling_plan(
        imgs, tmp_path, {"unit": "per_image", "n_per_unit": 20})
    keys = {v[0] for v in plan.values()}
    assert keys == {"WellA/img01.tif", "WellA/img02.tif"}
    assert all(v[1] == 20 for v in plan.values())
    # Same dataset under a different root -> same keys -> same nuclei.
    other = tmp_path / "elsewhere"
    imgs2 = [_FakeImg(other / "WellA" / "img01.tif")]
    plan2 = _resolve_sampling_plan(
        imgs2, other, {"unit": "per_image", "n_per_unit": 20})
    assert next(iter(plan2.values()))[0] == "WellA/img01.tif"


def test_per_well_plan_divides_n_across_the_images_beneath_each_well(tmp_path):
    from fishsuite.runner import _resolve_sampling_plan

    imgs = [_FakeImg(tmp_path / "WellA" / f"img{i}.tif") for i in range(3)]
    imgs += [_FakeImg(tmp_path / "WellB" / "img0.tif")]
    plan = _resolve_sampling_plan(
        imgs, tmp_path, {"unit": "per_well", "n_per_unit": 20})
    a = [v[1] for k, v in plan.items() if "WellA" in k]
    b = [v[1] for k, v in plan.items() if "WellB" in k]
    assert sum(a) == 20 and sorted(a) == [6, 7, 7]
    assert b == [20]
    # Every image beneath one well shares that well's RNG stream.
    a_units = {v[0].split("|")[0] for k, v in plan.items() if "WellA" in k}
    assert a_units == {"WellA"}


# ---------------------------------------------------------------------------
# End-to-end: a real run through runner.run_batch on synthetic TIFFs.
#
# The decisive integration checks. Inputs are built at test time (nothing is
# committed) and the `otsu` backend keeps the ML stack out of it.
# ---------------------------------------------------------------------------

_CENTRES = [(35, 35), (35, 90), (35, 145), (90, 35),
            (90, 90), (90, 145), (145, 35), (145, 90)]


_READER_PROBE = r'''
import pathlib, sys, tempfile
import numpy as np
import tifffile
from fishsuite.core.io import read_image

# The same 2-plane layout _write_synthetic_tiffs produces.
d = pathlib.Path(tempfile.mkdtemp())
p = d / "probe.tif"
tifffile.imwrite(str(p), np.stack([
    np.full((180, 180), 200, np.uint16),
    np.full((180, 180), 100, np.uint16),
], axis=0))
img = read_image(p)
print(f"shape={img.shape} n_channels={img.n_channels} n_z={img.n_z}")
sys.exit(0)
'''


def _reader_probe_failure() -> str | None:
    """None when a working reader is present; else a one-line reason to skip.

    ``bioio`` is a dispatcher: it reads nothing itself and delegates to a reader
    plugin. The only plugin this package declares is the optional ``bioformats``
    extra, so in a light environment ``read_image`` raises on every format
    including plain TIFF, and the end-to-end tests below report zero processed
    images rather than skipping.

    The probe OPENS A FILE rather than inspecting the ``bioio.readers`` entry-point
    registry, because presence of a plugin is not the property that matters. It
    deliberately asserts nothing about the axis layout: an earlier version
    required ``n_channels == 2`` and was WRONG — Bio-Formats reads this ambiguous
    2-plane TIFF as ``T=2, C=1, Z=1``, and the pipeline handles that fine (it
    segments all eight synthetic nuclei). That version skipped five tests that
    pass, in the author's own environment, which is worse than not guarding at
    all: a silently-removed test looks like a passing one.

    Not covered here, on purpose: a third-party reader such as ``bioio-tifffile``
    opens the file but resolves the extra axis differently, and DAPI segmentation
    then runs on the wrong plane and finds no nuclei. These tests FAIL rather than
    skip in that case. That is the right outcome — it is a real misconfiguration
    that would also silently corrupt a real run, so it should be loud. See the
    warning in README.md; CI is explicitly instructed not to install one.

    Runs in a SUBPROCESS: the bioformats plugin starts a JVM on first read, and on
    Windows that start can raise a native access violation that no ``except``
    catches and which corrupts JVM state for the rest of the process (see the note
    in ``_run_batch``). A child process contains that.
    """
    import subprocess
    import sys

    try:
        proc = subprocess.run(
            [sys.executable, "-c", _READER_PROBE],
            capture_output=True, text=True, timeout=300,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return f"bioio reader probe could not run ({type(exc).__name__}: {exc})"
    if proc.returncode == 0:
        return None
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()
    return (
        "no working bioio reader plugin — bioio cannot open even a plain TIFF; "
        "install the `bioformats` extra. Underlying error: "
        + (tail[-1][:200] if tail else f"probe exited {proc.returncode}")
    )


@pytest.fixture(scope="session")
def working_image_reader():
    """Skip the requesting test unless a bioio reader can actually open a file.

    A session fixture rather than a module-level ``skipif`` so the probe (which
    spawns a subprocess and may start a JVM) runs only when one of these tests is
    actually selected, instead of on every collection of this file.
    """
    if not hasattr(working_image_reader, "_reason"):
        working_image_reader._reason = _reader_probe_failure()
    if working_image_reader._reason:
        pytest.skip(working_image_reader._reason)


def _write_synthetic_tiffs(root):
    """Two wells, three fields of view, eight round nuclei each."""
    import tifffile

    for well in ("WellA", "WellB"):
        (root / well).mkdir(parents=True, exist_ok=True)

    def make(path, seed):
        r = np.random.default_rng(seed)
        h = w = 180
        dapi = np.full((h, w), 200, np.uint16)
        rna = np.full((h, w), 100, np.uint16)
        yy, xx = np.mgrid[0:h, 0:w]
        for i, (cy, cx) in enumerate(_CENTRES):
            m = (yy - cy) ** 2 + (xx - cx) ** 2 < 16 ** 2
            dapi[m] = (3000 + i * 50 + r.random(int(m.sum())) * 300).astype(np.uint16)
            rna[cy - 4:cy - 2, cx - 4:cx - 2] = 4200
            rna[cy + 2:cy + 5, cx + 2:cx + 5] = 3900
        tifffile.imwrite(str(path), np.stack([dapi, rna], axis=0))

    make(root / "WellA" / "img01.tif", 1)
    make(root / "WellA" / "img02.tif", 2)
    make(root / "WellB" / "img03.tif", 3)
    return root


def _run_batch(root, outdir, *, parallel, seg_workers, sampling=None):
    """Run the real pipeline IN A SUBPROCESS; return (per_image, nuclei, outdir).

    Out-of-process on purpose. ``core.io.read_image`` hands the file to bioio,
    whose plugin probe starts a BioFormats JVM, and on Windows that start can
    raise a native access violation which — as ``io.read_image`` itself
    documents — no Python ``except`` can catch and which corrupts JVM state for
    the REST of the process. Keeping the run in a child process means that can
    never reach the pytest process and destabilise unrelated tests.

    It also makes the worker-count comparison a genuinely cross-process test of
    the seed derivation, which is the property under test.
    """
    import subprocess
    import sys

    import pandas as pd

    cfg = FishsuiteConfig()
    cfg.channels.analysis_mode = "rna_only"
    cfg.channels.dapi = 0
    cfg.channels.rna = 1
    cfg.nuclei.backend = "otsu"
    cfg.nuclei.min_area_px = 120
    cfg.z_stack.mode = "single"
    # 'batch' scope engages the PARALLEL segmentation pre-scan, so the worker
    # count genuinely changes how the work is scheduled.
    cfg.pixel_coloc.threshold_scope = "batch"
    cfg.parallel.seg_workers = seg_workers
    cfg.output.save_qc_overlays = False
    cfg.output.save_publication_images = False
    cfg.output.save_masks = False
    cfg.conditions.mode = "subfolders"
    cfg.conditions.subfolder_conditions = {"WellA": "A", "WellB": "B"}
    for k, v in (sampling or {}).items():
        setattr(cfg.sampling, k, v)
    outdir.parent.mkdir(parents=True, exist_ok=True)
    cfg_path = outdir.parent / f"cfg_{outdir.name}.yaml"
    cfg.dump_yaml(cfg_path)

    code = (
        "from fishsuite.runner import run_batch;"
        f"run_batch(r'{cfg_path}', r'{root}', r'{outdir}',"
        f" parallel={int(parallel)}, dry_run=False)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
    )
    summary = outdir / "per_image_summary.csv"
    if not summary.exists():
        raise AssertionError(
            f"run_batch produced no per_image_summary.csv\n"
            f"stdout tail:\n{proc.stdout[-2000:]}\n"
            f"stderr tail:\n{proc.stderr[-2000:]}"
        )
    return (
        pd.read_csv(summary),
        pd.read_csv(outdir / "nuclei_metrics.csv"),
        outdir,
    )


@pytest.mark.bioformats
def test_end_to_end_selection_is_identical_at_1_and_12_workers(tmp_path, working_image_reader):
    """THE reproducibility test: the draw must not depend on worker count.

    Images are handed to a pool, so completion order varies with the worker
    count. Had the sampler consumed one global RNG in processing order, this
    would fail intermittently — the worst possible failure mode for a published
    N, because the number would look stable until someone changed -p.
    """
    root = _write_synthetic_tiffs(tmp_path / "data")
    samp = dict(enabled=True, n_per_unit=4, seed=123)
    _, n1, _ = _run_batch(root, tmp_path / "out_p1", parallel=1,
                          seg_workers=1, sampling=samp)
    _, n12, _ = _run_batch(root, tmp_path / "out_p12", parallel=12,
                           seg_workers=4, sampling=samp)

    key = ["image", "nucleus_id"]
    a = n1.sort_values(key).reset_index(drop=True)
    b = n12.sort_values(key).reset_index(drop=True)
    assert a["sampled_in_analysis"].tolist() == b["sampled_in_analysis"].tolist()
    assert a["sampling_rank"].tolist() == b["sampling_rank"].tolist()
    # Something was actually selected, so the assertions above are not
    # vacuously comparing two empty columns.
    assert int(a["sampled_in_analysis"].astype(bool).sum()) == 12


@pytest.mark.bioformats
def test_end_to_end_disabled_adds_no_columns_and_no_methods_file(tmp_path, working_image_reader):
    """Sampling off must be invisible: no new columns, no new files.

    Byte-identity against the pre-feature code was verified separately by
    running both versions over one dataset; this locks the invariant going
    forward, since an unconditionally-added column is what would break it.
    """
    root = _write_synthetic_tiffs(tmp_path / "data")
    pi, nu, out = _run_batch(root, tmp_path / "out_off", parallel=1, seg_workers=1)

    for col in ("eligible_for_sampling", "sampled_in_analysis", "sampling_rank"):
        assert col not in nu.columns
    for col in ("n_nuclei_segmented", "n_nuclei_area_excluded",
                "n_nuclei_ghost_excluded", "n_nuclei_eligible",
                "n_nuclei_sampled", "sampling_short_of_target", "sampling_unit",
                "sampling_order", "sampling_seed_used", "image_included_in_stats"):
        assert col not in pi.columns
    # The pre-existing column is still there, and not duplicated.
    assert list(pi.columns).count("n_nuclei_border_excluded") == 1
    assert not (out / "sampling_methods.txt").exists()


@pytest.mark.bioformats
def test_end_to_end_sampled_count_matches_the_per_nucleus_flags(tmp_path, working_image_reader):
    root = _write_synthetic_tiffs(tmp_path / "data")
    pi, nu, out = _run_batch(
        root, tmp_path / "out_on", parallel=1, seg_workers=1,
        sampling=dict(enabled=True, n_per_unit=4, seed=123),
    )
    for _, row in pi.iterrows():
        sub = nu[nu["image"] == row["image"]]
        assert int(sub["sampled_in_analysis"].astype(bool).sum()) == \
            int(row["n_nuclei_sampled"])
        assert int(row["n_nuclei_sampled"]) == 4
        assert int(row["n_nuclei_eligible"]) == 8
        assert bool(row["sampling_short_of_target"]) is False
        # Every eligible nucleus keeps its row — nothing silently discarded.
        assert int(sub["eligible_for_sampling"].astype(bool).sum()) == 8
    assert (nu["sampled_in_analysis"].astype(bool)
            == (nu["sampling_rank"] <= 4)).all()
    assert (out / "sampling_methods.txt").exists()


@pytest.mark.bioformats
def test_end_to_end_n_above_eligible_is_short_and_takes_everything(tmp_path, working_image_reader):
    root = _write_synthetic_tiffs(tmp_path / "data")
    pi, nu, _ = _run_batch(
        root, tmp_path / "out_short", parallel=1, seg_workers=1,
        sampling=dict(enabled=True, n_per_unit=500, seed=1),
    )
    assert (pi["n_nuclei_sampled"] == pi["n_nuclei_eligible"]).all()
    assert pi["sampling_short_of_target"].astype(bool).all()
    assert nu["sampled_in_analysis"].astype(bool).all()


@pytest.mark.bioformats
def test_end_to_end_run_config_records_the_resolved_sampling_block(tmp_path, working_image_reader):
    import json

    root = _write_synthetic_tiffs(tmp_path / "data")
    _, _, out = _run_batch(
        root, tmp_path / "out_rc", parallel=1, seg_workers=1,
        sampling=dict(enabled=True, n_per_unit=4, seed=123, order="raster"),
    )
    rc = json.loads((out / "run_config.json").read_text(encoding="utf-8"))
    assert rc["sampling"]["enabled"] is True
    assert rc["sampling"]["n_per_unit"] == 4
    assert rc["sampling"]["seed"] == 123
    assert rc["sampling"]["order"] == "raster"
    # The Methods text must describe the run that actually happened.
    txt = (out / "sampling_methods.txt").read_text(encoding="utf-8")
    assert "raster" in txt.lower()
    assert "not an unbiased sample" in txt.lower()
