"""Locked figure-panel module (fishsuite.figures, 2026-09-28).

Unit tests (no data): holm(), the marker rule, the LUT guard.
Reproduction tests (skip when the source run is absent): the module rebuilds the saved C stats and the field-15
nucleus-11 linked-set coordinates of FIG1_IMAGING_v2 exactly (values read with round-trip float parsing).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fishsuite.figures import holm, well_mean_style
from fishsuite.figures.style import RGB, Saver, check_luts

BASAL = Path(r"F:\Image Analysis Work\MIAT_QKI_BASAL_Fig1_2026-09-24")
V2 = BASAL / "FIG1_IMAGING_v2_2026-09-28"
MENU = BASAL / "FIG1_IMAGING_MENU_2026-09-28"
BF = Path(r"F:\Image Analysis Work\MIAT-QKI-Coloc\UD\_OEvWT_2026-09\EXACT_FOOTPRINT_BACKFILL_20260917-205946")
RUN = Path(r"F:\Image Analysis Work\MIAT-QKI-Coloc\UD\_OEvWT_2026-09\RUN_PROD_OEvControl_PLAIN_jointAF_diam11um_LoG174_miat500_qki1050_20260917-203051")
HAVE_DATA = all(p.exists() for p in (V2 / "data" / "stats.csv", V2 / "data" / "B_linked_params.json",
                                     MENU / "data" / "menu_per_nucleus.csv", BF / "selected_planes_and_masks.h5"))
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="FIG1_IMAGING_v2 source run not on this machine")


# ---------------------------------------------------------------- holm
def test_holm_hand_computed():
    got = holm([0.01, 0.04, 0.03, 0.005])
    # sorted: .005*4=.02, .01*3=.03, .03*2=.06, .04*1=.04 -> monotone max .06
    np.testing.assert_allclose(got, [0.03, 0.06, 0.06, 0.02], rtol=0, atol=1e-15)


def test_holm_caps_at_one_and_is_monotone():
    got = holm([0.5, 0.6, 0.001])
    np.testing.assert_allclose(got, [1.0, 1.0, 0.003], rtol=0, atol=1e-15)
    order = np.argsort([0.5, 0.6, 0.001])
    assert np.all(np.diff(got[order]) >= 0)


def test_holm_single_and_matches_statsmodels():
    assert holm([0.2]).tolist() == [0.2]
    sm = pytest.importorskip("statsmodels.stats.multitest")
    ps = np.array([2.7e-12, 7.7e-7, 1.8e-17, 1.8e-17, 7.4e-10, 4.5e-11, 0.03, 0.2])
    np.testing.assert_allclose(holm(ps), sm.multipletests(ps, method="holm")[1], rtol=1e-12)


# ---------------------------------------------------------------- marker rule
def test_marker_rule_same_condition_one_marker():
    a, b = well_mean_style("VPR noDox", "14"), well_mean_style("VPR noDox", "15")
    assert a == b and a["marker"] == "o"


def _synthetic_nv(seed=1):
    rng = np.random.default_rng(seed)
    n = 30
    return pd.DataFrame(dict(field=["14"] * n + ["15"] * n,
                             mean_uniform_position_percentile_qki=rng.uniform(.5, .9, 2 * n),
                             frac_miat_spots_qki_pos=rng.uniform(.6, 1, 2 * n),
                             null_mean_frac_miat_spots_qki_pos=rng.uniform(.4, .8, 2 * n)))


def _well_mean_collections(fig):
    return [c for ax in fig.axes for c in ax.collections
            if hasattr(c, "get_sizes") and len(c.get_sizes()) and c.get_sizes()[0] == 110]


def test_marker_rule_in_rendered_superplots():
    import matplotlib.pyplot as plt
    from fishsuite.figures.c_superplots import plot_one_sample, plot_paired
    from fishsuite.figures.stats import wilcoxon_effect
    nv = _synthetic_nv()
    t = wilcoxon_effect(nv.mean_uniform_position_percentile_qki.values - .5) | {"p_holm": 1e-5}
    sv = Saver("unused", keep_open=True, write=False)
    rng = np.random.default_rng(0)
    s1 = plot_one_sample(nv, ["14", "15"], "C1_mean_UPP", "mean_uniform_position_percentile_qki", .5, "UPP", "violin", t, True, rng, sv)
    s2 = plot_paired(nv, ["14", "15"], "C6_x", "frac_miat_spots_qki_pos", "null_mean_frac_miat_spots_qki_pos", "frac", t, True, rng, sv)
    ref = plt.figure().add_subplot().scatter([0], [0], marker="o").get_paths()[0].vertices
    for stem, n_expected in ((s1, 2), (s2, 4)):
        cols = _well_mean_collections(sv.figures[stem])
        assert len(cols) == n_expected, stem
        for c in cols:
            np.testing.assert_array_equal(c.get_paths()[0].vertices, ref)
        obs = [c for c in cols if not np.allclose(c.get_facecolor()[0][:3], [0xDD / 255] * 3)]
        assert len({tuple(c.get_facecolor()[0]) for c in obs}) == 1, "wells of one condition must share one colour"
    plt.close("all")


def test_luts_never_green():
    check_luts()
    assert RGB["miat"] == (1, 1, 0) and RGB["qki"] == (1, 0, 1)
    with pytest.raises(ValueError):
        check_luts({"miat": (0, 1, 0)})


# ---------------------------------------------------------------- reproduction against the saved run
@pytest.fixture(scope="module")
def run(tmp_path_factory):
    from fishsuite.figures.config import PanelRun
    return PanelRun(MENU, BF, RUN, tmp_path_factory.mktemp("figpanels"))


@needs_data
def test_repro_c1_stats_equal_saved(run):
    from fishsuite.figures.stats import run_tests
    nv, _ = run.tables()
    fields = sorted(nv.field.unique())
    assert fields == ["14", "15"]
    T, _, _ = run_tests(nv, fields)
    saved = pd.read_csv(V2 / "data" / "stats.csv", float_precision="round_trip").set_index("panel").loc["C1_mean_UPP"]
    c1 = T["C1_mean_UPP"]
    assert c1["n"] == int(saved.n_nuclei)
    assert c1["W"] == saved.W
    assert c1["p"] == saved.p
    for f in fields:
        assert c1[f"well_mean_{f}"] == saved[f"well_{f}_mean"]
        assert c1[f"well_{f}"]["p"] == saved[f"well_{f}_p"]
        assert c1[f"well_{f}"]["n"] == int(saved[f"well_{f}_n"])


@needs_data
def test_repro_c_holm_table_equal_saved(run):
    from fishsuite.figures.c_superplots import build_c_panels
    nv, _ = run.tables()
    got = build_c_panels(nv, sorted(nv.field.unique()), run.saver, render=False)
    saved = pd.read_csv(V2 / "data" / "stats_round2_C_holm.csv", float_precision="round_trip")
    pd.testing.assert_frame_equal(got, saved, check_exact=True, check_dtype=False)


@needs_data
def test_repro_c1_is_sensitive_to_input(run):
    """Guard against a comparison that cannot fail: dropping one nucleus must change C1 p."""
    from fishsuite.figures.stats import run_tests
    nv, _ = run.tables()
    T, _, _ = run_tests(nv.iloc[1:], ["14", "15"])
    saved_p = pd.read_csv(V2 / "data" / "stats.csv", float_precision="round_trip").set_index("panel").loc["C1_mean_UPP", "p"]
    assert T["C1_mean_UPP"]["p"] != saved_p


def _assert_geometry_matches(G, saved):
    from fishsuite.figures.linked_set import params_dict
    got = params_dict(G, G["key"][-6:-4], saved["nucleus"]["nucleus_id"], "", 2.0, None, None)
    assert got["nucleus"]["image"] == saved["nucleus"]["image"]
    assert got["nucleus"]["box_xywh_px"] == saved["nucleus"]["box_xywh_px"]
    for k in ("spot_id", "y", "x", "z0", "crosshair_yx"):
        assert got["marked_spot"][k] == saved["marked_spot"][k], k
    for k in ("rule", "angle_deg", "p0_yx", "p1_yx", "length_um", "n_puncta_on_line", "n_puncta_in_nucleus"):
        assert got["line"][k] == saved["line"][k], k
    assert got["puncta_on_line"] == saved["puncta_on_line"]
    assert got["pixel_um"] == saved["pixel_um"] and got["z_step_um"] == saved["z_step_um"]


@needs_data
def test_repro_b_field15_nuc11_coordinates_equal_saved(run):
    from fishsuite.figures.linked_set import linked_geometry
    saved = json.loads((V2 / "data" / "B_linked_params.json").read_text())
    G = linked_geometry(run, "15", 11)
    assert (G["line"]["angle_deg"], int(G["mark"].spot_id), len(G["on"])) == (56, 166, 3)
    _assert_geometry_matches(G, saved)


@needs_data
@pytest.mark.parametrize("nid", [27, 14])
def test_repro_b_field14_coordinates_equal_saved(run, nid):
    from fishsuite.figures.linked_set import linked_geometry
    saved = json.loads((V2 / "data" / f"B_linked_params_field14_nuc{nid}.json").read_text())
    _assert_geometry_matches(linked_geometry(run, "14", nid), saved)


@needs_data
def test_repro_selection_equal_saved(run):
    from fishsuite.figures import selection
    nv, _ = run.tables()
    saved = json.loads((V2 / "data" / "round2_selections.json").read_text())
    f14 = selection.field_rep(nv, "14")
    assert f14["nucleus_id"] == saved["field14_rep"]["nucleus_id"] and f14["distance"] == saved["field14_rep"]["distance"]
    cr = selection.coloc_reps(nv)
    assert [(p["field"], p["nucleus_id"], p["d"]) for p in cr["picks"]] == \
           [(p["field"], p["nucleus_id"], p["d"]) for p in saved["coloc_reps"]["picks"]]
    assert cr["scale"] == saved["coloc_reps"]["scale"]


@needs_data
@pytest.mark.lab
@pytest.mark.bioformats
def test_repro_full_linked_render_field15_nuc11(run):
    """Full B1-B4 render from the raw VSI (read-only): params incl. z window equal the saved JSON; 4 panels written."""
    pytest.importorskip("bioio_bioformats")
    from fishsuite.figures.linked_set import build_linked_set
    from fishsuite.figures.style import apply_style
    apply_style()
    saved = json.loads((V2 / "data" / "B_linked_params.json").read_text())
    p = build_linked_set(run, "15", 11, 0.4927148355969213)
    assert p["z_window"] == saved["z_window"]
    assert p["stack_matches_h5"] == saved["stack_matches_h5"]
    assert p["line"]["p0_yx"] == saved["line"]["p0_yx"] and p["line"]["p1_yx"] == saved["line"]["p1_yx"]
    for stem in ("B1_FOV_ortho_VPRnoDox_field15_nuc11_box", "B2_nucleus_ortho_VPRnoDox_field15_nuc11_fishsuite",
                 "B3_nucleus_zoom_line_VPRnoDox_field15_nuc11", "B4_line_profile_VPRnoDox_field15_nuc11"):
        assert (run.pan / f"{stem}.png").stat().st_size > 0
    prof = pd.read_csv(run.data / "B4_line_profile_values_field15_nuc11.csv", float_precision="round_trip")
    pd.testing.assert_frame_equal(prof, pd.read_csv(V2 / "data" / "B4_line_profile_values.csv", float_precision="round_trip"),
                                  check_exact=True, check_dtype=False)
