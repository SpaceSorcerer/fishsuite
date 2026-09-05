"""The post-hoc peak floor, and all-pairs contrasts.

The peak gate reproduces a gated run from an ungated one because fishsuite's own
spot floor is applied AFTER detection. What must not drift: the boundary is at or
above the floor, the nuclear-fraction denominator is in-nucleus plus
in-cytoplasm rather than the row count, and columns a floor makes stale are NAMED
rather than silently carried through at their pre-gate value.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fishsuite.report import aggregate as agg
from fishsuite.report import endpoints as ep
from fishsuite.report import peak_gate as pg


# ------------------------------------------------------------------ parsing


def test_peak_floor_spec_maps_channel_aliases():
    assert pg.parse_peak_floors("rna=1000,rna2=1200") == {"rna1": 1000.0, "rna2": 1200.0}
    assert pg.parse_peak_floors("intron=900, exon=700") == {"rna1": 900.0, "rna2": 700.0}


def test_peak_floor_spec_rejects_a_bad_channel_and_a_bad_number():
    with pytest.raises(pg.PeakGateError, match="not one of"):
        pg.parse_peak_floors("dapi=100")
    with pytest.raises(pg.PeakGateError, match="not a number"):
        pg.parse_peak_floors("rna=high")
    with pytest.raises(pg.PeakGateError, match="CHANNEL=VALUE"):
        pg.parse_peak_floors("rna 1000")


# ------------------------------------------------------------------- gating


def _spots():
    return pd.DataFrame({
        "image": ["a.vsi"] * 6,
        "channel": ["rna1", "rna1", "rna1", "rna2", "rna2", "rna2"],
        "nucleus_id": [1, 1, 1, 1, 1, 1],
        "in_nucleus": [True, True, False, True, False, False],
        "in_cytoplasm": [False, False, True, False, True, True],
        "peak_intensity": [999.0, 1000.0, 1001.0, 1199.0, 1200.0, 1201.0],
    })


def test_a_spot_exactly_at_the_floor_is_kept():
    kept, rec = pg.gate_spots(_spots(), {"rna1": 1000.0, "rna2": 1200.0})
    assert sorted(kept["peak_intensity"]) == [1000.0, 1001.0, 1200.0, 1201.0]
    assert rec["per_channel"]["rna1"]["spots_kept"] == 2
    assert rec["per_channel"]["rna1"]["spots_dropped"] == 1


def test_a_non_positive_floor_leaves_its_channel_untouched():
    kept, rec = pg.gate_spots(_spots(), {"rna1": 0.0, "rna2": float("nan")})
    assert len(kept) == 6
    assert not rec["per_channel"]["rna1"]["applied"]
    assert not rec["per_channel"]["rna2"]["applied"]


def test_a_run_without_a_peak_column_is_refused_not_silently_skipped():
    s = _spots().drop(columns=["peak_intensity"])
    with pytest.raises(pg.PeakGateError, match="none of"):
        pg.gate_spots(s, {"rna1": 1000.0})


def test_the_nuclear_fraction_denominator_is_nucleus_plus_cytoplasm():
    """A spot flagged neither in-nucleus nor in-cytoplasm is excluded from every
    count. Using the row count instead would silently deflate the fraction."""
    spots = pd.DataFrame({
        "image": ["a.vsi"] * 4, "channel": ["rna1"] * 4, "nucleus_id": [1] * 4,
        "in_nucleus": [True, True, False, False],
        "in_cytoplasm": [False, False, True, False],   # the last is neither
        "peak_intensity": [10.0] * 4,
    })
    nuclei = pd.DataFrame({"image": ["a.vsi"], "nucleus_id": [1],
                           "n_spots_rna1": [99.0], "nuclear_spot_count": [99.0],
                           "cyto_spot_count": [99.0], "nuclear_spot_fraction": [0.5],
                           "nucleus_area_px": [1000.0]})
    out, _ = pg.apply_to_nuclei(nuclei, spots, voxel_xy_um=0.1)
    assert float(out["nuclear_spot_count"].iloc[0]) == 2
    assert float(out["cyto_spot_count"].iloc[0]) == 1
    assert float(out["n_spots_rna1"].iloc[0]) == 3          # not 4
    assert float(out["nuclear_spot_fraction"].iloc[0]) == pytest.approx(2 / 3)


def test_columns_a_floor_cannot_re_derive_are_named():
    spots = _spots()
    nuclei = pd.DataFrame({"image": ["a.vsi"], "nucleus_id": [1],
                           "n_spots_rna1": [3.0], "nucleus_area_px": [1000.0],
                           "nuclear_above_floor_intensity_rna1": [5000.0],
                           "manders_rna1_in_rna2": [0.4]})
    _, stale = pg.apply_to_nuclei(nuclei, spots, voxel_xy_um=0.1)
    assert "nuclear_above_floor_intensity_rna1" in stale
    assert "manders_rna1_in_rna2" in stale


# --------------------------------------------------------------- all pairs


def _five_group_run(tmp_path: Path) -> Path:
    run = tmp_path / "RUN_5"
    run.mkdir(parents=True)
    rng = np.random.default_rng(0)
    img, nuc = [], []
    means = {"WT": 2.5, "QKI-KO": 4.2, "Clone16": 0.7, "Clone17": 0.9, "Mix": 1.5}
    for line, mu in means.items():
        for w in (1, 2, 3):
            for f in range(3):
                image = f"X_{line}-{w}_{f:02d}.vsi"
                img.append({"image": image, "condition": line, "secondary_only": False})
                for k in range(5):
                    nuc.append({"image": image, "nucleus_id": k,
                                "n_spots_rna1": float(mu + rng.normal(0, 0.3)),
                                "nuclear_spot_fraction": 0.8,
                                "nucleus_area_px": 16000.0, "voxel_xy_um": 0.13})
    pd.DataFrame(img).to_csv(run / "per_image_summary.csv", index=False)
    pd.DataFrame(nuc).to_csv(run / "nuclei_metrics.csv", index=False)
    (run / "run_config.json").write_text(json.dumps({"config_resolved": {"channels": {}}}),
                                         encoding="utf-8")
    return run


def _contrasts(run: Path, all_pairs: bool):
    data = agg.load_run(run, {}, {},
                        well_from_image=r"_((?:WT|QKI-KO|Clone16|Clone17|Mix)-\d)_")
    order = ["WT", "QKI-KO", "Clone16", "Clone17", "Mix"]
    endpoints, absent = ep.resolve(data["nuclei"], data["per_image"])
    field = agg.per_field_long(data["nuclei"], data["per_image"], endpoints,
                               data["labels"], 5)
    well = agg.per_well_long(field)
    return agg.build_contrasts(well, field, endpoints, absent, order, "WT",
                               ep.channel_labels(data["cfg"]), all_pairs=all_pairs)


def test_five_groups_give_ten_pairs_and_four_against_the_reference(tmp_path):
    run = _five_group_run(tmp_path)
    name = "rna1_spots_per_nucleus"
    ap = _contrasts(run, True)
    ap = ap[ap.endpoint == name]
    vr = _contrasts(run, False)
    vr = vr[vr.endpoint == name]
    assert len(ap) == 10
    assert len(vr) == 4
    pairs = {frozenset((r.test_group, r.reference_group)) for r in ap.itertuples()}
    assert len(pairs) == 10


def test_the_tukey_adjustment_is_over_all_five_groups_for_every_pair(tmp_path):
    run = _five_group_run(tmp_path)
    c = _contrasts(run, True)
    c = c[c.endpoint == "rna1_spots_per_nucleus"]
    assert set(c["tukey_k"].dropna()) == {5.0}
    assert set(c["well_tukey_k"].dropna()) == {5.0}
    # 45 fields minus 5 groups, and 15 wells minus 5 groups.
    assert set(c["tukey_df"].dropna()) == {40.0}
    assert set(c["well_tukey_df"].dropna()) == {10.0}


def test_the_well_level_and_field_level_tukey_are_reported_separately(tmp_path):
    """They are different statistics on different units and must not be conflated:
    the field-level fit treats a technical replicate as independent."""
    run = _five_group_run(tmp_path)
    c = _contrasts(run, True)
    c = c[c.endpoint == "rna1_spots_per_nucleus"].dropna(
        subset=["p_tukey_fov", "well_p_tukey_fov"])
    assert len(c) > 0
    assert not np.allclose(c["p_tukey_fov"], c["well_p_tukey_fov"])
    # Same estimand, so the differences agree even though the p-values do not.
    assert np.allclose(c["tukey_difference"], c["well_tukey_difference"], rtol=1e-9)


# ------------------------------------------- palette scope, LUT slots, pairing repair


def test_the_two_group_imaging_pair_is_not_applied_to_a_multi_group_set():
    """The WT/QKI-KO imaging pair is a TWO-GROUP lock. Applying it inside a
    five-line set recoloured WT and QKI-KO away from that set's own published key
    on 2026-09-04."""
    from fishsuite.report.figures import group_colors

    two = group_colors(["WT", "QKI-KO"])
    assert two["WT"] == "#595959" and two["QKI-KO"] == "#D67AE5"
    five = group_colors(["WT", "QKI-KO", "Clone16", "Clone17", "Mix"])
    assert five["WT"] != "#595959" and five["QKI-KO"] != "#D67AE5"
    assert len(set(five.values())) == len(five), "two groups share a colour"


def test_an_explicit_colour_key_beats_every_lock():
    from fishsuite.report.figures import group_colors

    key = {"WT": "#333333", "QKI-KO": "#CC79A7", "Clone16": "#0072B2",
           "Clone17": "#56B4E9", "Mix": "#E69F00"}
    got = group_colors(list(key), key)
    for g, hx in key.items():
        assert got[g] == hx
    # and it also wins at two groups, where the imaging lock would otherwise apply
    assert group_colors(["WT", "QKI-KO"], {"WT": "#000000"})["WT"] == "#000000"


def test_lut_slots_follow_the_runs_analysis_mode(tmp_path):
    """An rna_rna run must take rna2_lut, not antibody_lut. Both keys are always
    present in run_config, so a fixed slot list silently hands an rna_rna panel
    the antibody colour: that produced a green exon channel on a magenta run."""
    from fishsuite.report.figures import read_luts

    ch = {"rna_label": "introns", "rna_lut": "yellow",
          "rna2_label": "exons", "rna2_lut": "magenta",
          "antibody_label": "Protein", "antibody_lut": "green",
          "dapi_label": "DAPI", "dapi_lut": "blue"}
    for mode, want in (("rna_rna", ["introns", "exons", "DAPI"]),
                       ("rna_protein", ["introns", "Protein", "DAPI"])):
        run = tmp_path / mode
        run.mkdir()
        (run / "run_config.json").write_text(
            json.dumps({"config_resolved": {"channels": dict(ch, analysis_mode=mode)}}),
            encoding="utf-8")
        ent, src = read_luts(run, None)
        assert [n for n, _ in ent] == want, (mode, ent)
        hexes = [h.lower() for _, h in ent]
        if mode == "rna_rna":
            # The antibody slot is green in this config; an rna_rna panel must
            # never reach it, and green is banned by the style in any case.
            assert "#00ff00" not in hexes, "the antibody LUT leaked into an rna_rna key"
            assert "#ff00ff" in hexes, "the rna2 LUT is missing from an rna_rna key"


def test_an_unknown_analysis_mode_draws_no_lut_key_rather_than_guessing(tmp_path):
    from fishsuite.report.figures import read_luts

    run = tmp_path / "weird"
    run.mkdir()
    (run / "run_config.json").write_text(
        json.dumps({"config_resolved": {"channels": {"analysis_mode": "not_a_mode"}}}),
        encoding="utf-8")
    ent, src = read_luts(run, None)
    assert ent == []
    assert "no channel slot map" in src


def test_pairing_is_recomputed_against_the_surviving_partner_set():
    """After a floor, a punctum whose only partner was dropped must read UNPAIRED.
    Reading the run-time flag would keep it paired to a spot that no longer exists."""
    spots = pd.DataFrame({
        "image": ["a.vsi"] * 3,
        "channel": ["rna1", "rna2", "rna2"],
        "nucleus_id": [1, 1, 1],
        "in_nucleus": [True, True, True],
        "in_cytoplasm": [False, False, False],
        "x_px": [0.0, 1.0, 100.0], "y_px": [0.0, 0.0, 0.0], "z_slice": [0, 0, 0],
        "peak_intensity": [5000.0, 100.0, 5000.0],
        "paired_at_0p3um": [1, 1, 0],          # run-time flag: the rna1 spot is paired
        "nn_distance_um": [0.1, 0.1, 10.0],
    })
    kept, _ = pg.gate_spots(spots, {"rna2": 1000.0})   # drops the near rna2 partner
    repaired, rec = pg.repair_pairing(kept, 0.3, voxel_xy_um=0.13, voxel_z_um=0.33)
    assert rec["recomputed"]
    r1 = repaired[repaired.channel == "rna1"].iloc[0]
    assert r1["paired_at_0p3um"] == 0, "stale pairing survived the floor"
    assert r1["nn_distance_um"] > 0.3


def test_pairing_repair_is_a_no_op_when_no_partner_was_dropped():
    spots = pd.DataFrame({
        "image": ["a.vsi"] * 2, "channel": ["rna1", "rna2"], "nucleus_id": [1, 1],
        "in_nucleus": [True, True], "in_cytoplasm": [False, False],
        "x_px": [0.0, 1.0], "y_px": [0.0, 0.0], "z_slice": [0, 0],
        "peak_intensity": [5000.0, 5000.0],
        "paired_at_0p3um": [1, 1], "nn_distance_um": [0.13, 0.13],
    })
    kept, _ = pg.gate_spots(spots, {"rna2": 1000.0})
    repaired, _ = pg.repair_pairing(kept, 0.3, 0.13, 0.33)
    assert list(repaired["paired_at_0p3um"]) == [1, 1]


def test_the_secondary_only_outlier_rule_is_per_channel(tmp_path):
    """A control field can be clean on one channel and hot on the other; the rule
    must catch it on either."""
    from fishsuite.report import aggregate as agg2

    nuc = pd.DataFrame({
        "image": sum([[f"s{i}.vsi"] * 20 for i in range(4)], []),
        "nucleus_id": list(range(20)) * 4,
        "secondary_only": [True] * 80,
        "n_spots_rna1": [1.0] * 60 + [1.0] * 20,
        "n_spots_rna2": [1.0] * 60 + [9.0] * 20,      # s3 is hot on rna2 only
    })
    lab = pd.DataFrame({"image": [f"s{i}.vsi" for i in range(4)],
                        "secondary_only": [True] * 4})
    per_image = pd.DataFrame({"image": [f"s{i}.vsi" for i in range(4)]})
    out = agg2.secondary_only_table(nuc, per_image, lab, {}, min_nuclei=5, outlier_k=3.0)
    hot = out[out["field"] == "s3.vsi"].iloc[0]
    assert bool(hot["excluded"]) and hot["outlier_channel"] == "rna2"
    assert int(out["excluded"].sum()) == 1


def test_a_derived_column_never_shadows_the_engines_own(tmp_path):
    """Merging a recomputed column onto a run that already emits it suffixes both
    to `_x`/`_y`, removing the plain name and making the endpoint resolve ABSENT.
    Two pairing endpoints were silently blanked that way."""
    run = tmp_path / "RUN_DUP"
    run.mkdir(parents=True)
    img = [{"image": "a.vsi", "condition": "WT_1", "secondary_only": False}]
    nuc = [{"image": "a.vsi", "nucleus_id": 1, "n_spots_rna1": 2.0,
            "nucleus_area_px": 16000.0, "voxel_xy_um": 0.065,
            "paired_fraction_rna1_at_0p3um": 0.75,      # the engine's own value
            "median_nn_distance_rna1_um": 0.42}]
    spot = [{"image": "a.vsi", "channel": "rna1", "nucleus_id": 1, "in_nucleus": True,
             "in_cytoplasm": False, "x_px": 0.0, "y_px": 0.0, "z_slice": 0,
             "paired_at_0p3um": 0, "nn_distance_um": 9.0,
             "miat_footprint_area_px": 20.0}]
    pd.DataFrame(img).to_csv(run / "per_image_summary.csv", index=False)
    pd.DataFrame(nuc).to_csv(run / "nuclei_metrics.csv", index=False)
    pd.DataFrame(spot).to_csv(run / "spot_metrics.csv", index=False)
    (run / "run_config.json").write_text(
        json.dumps({"config_resolved": {"channels": {"analysis_mode": "rna_protein"}}}),
        encoding="utf-8")

    data = agg.load_run(run, {}, {})
    cols = list(data["nuclei"].columns)
    for c in ("paired_fraction_rna1_at_0p3um", "median_nn_distance_rna1_um"):
        assert c in cols, f"{c} lost its plain name in the merge"
        assert f"{c}_x" not in cols and f"{c}_y" not in cols, f"{c} was suffixed"
    # The engine's value survives; the recomputation does not shadow it.
    assert float(data["nuclei"]["paired_fraction_rna1_at_0p3um"].iloc[0]) == 0.75
    assert float(data["nuclei"]["median_nn_distance_rna1_um"].iloc[0]) == 0.42
    _, absent = ep.resolve(data["nuclei"], data["per_image"])
    assert "paired_fraction_rna1_at_0p3um" not in absent
    assert "median_nn_distance_rna1_um" not in absent
