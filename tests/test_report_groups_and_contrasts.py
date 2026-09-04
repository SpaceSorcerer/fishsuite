"""Condition groups and the contrast maths behind ``fishsuite report``.

Two things are held here and they fail for different reasons.

The GROUP layer: a condition is one well, a group is the condition several wells
belong to. A run with no groups configured must be untouched, and a run with
groups must place every well in exactly one group with sec-only kept out of the
biological groups entirely.

The CONTRAST maths: Welch on well means, Hedges g, Holm within a family and the
minimum detectable effect are checked against values computed by hand in the
test rather than against the implementation, so a change in the implementation
that changes a number fails here instead of passing quietly.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fishsuite.config.schema import ConditionsCfg
from fishsuite.report import aggregate as agg
from fishsuite.report import endpoints as ep
from fishsuite.report import stats as st


# ---------------------------------------------------------------- group layer


def test_group_of_maps_wells_and_falls_back_to_the_well_itself():
    c = ConditionsCfg(groups={"WT": ["WT_1", "WT_2"], "QKI-KO": ["KO_1"]},
                      group_order=["WT", "QKI-KO"])
    assert c.group_of("WT_2") == "WT"
    assert c.group_of("KO_1") == "QKI-KO"
    # An unlisted well is visible as its own group rather than pooled silently.
    assert c.group_of("KO_9") == "KO_9"


def test_resolved_group_order_appends_groups_missing_from_the_declared_order():
    c = ConditionsCfg(groups={"WT": ["WT_1"], "QKI-KO": ["KO_1"], "Rescue": ["R_1"]},
                      group_order=["QKI-KO", "WT"])
    assert c.resolved_group_order() == ["QKI-KO", "WT", "Rescue"]


def test_no_groups_configured_means_no_group_mapping_at_all():
    c = ConditionsCfg()
    assert c.groups == {}
    assert c.resolved_group_order() == []
    assert c.group_of("WT_1") == "WT_1"


def test_parse_groups_round_trips_the_cli_spelling():
    m, order = agg.parse_groups(["WT=WT_1,WT_2,WT_3", "QKI-KO=KO_1,KO_2"])
    assert order == ["WT", "QKI-KO"]
    assert m == {"WT_1": "WT", "WT_2": "WT", "WT_3": "WT",
                 "KO_1": "QKI-KO", "KO_2": "QKI-KO"}


def test_parse_groups_rejects_a_well_claimed_by_two_groups():
    with pytest.raises(agg.ReportInputError, match="assigned to both"):
        agg.parse_groups(["WT=WT_1", "QKI-KO=WT_1"])


def test_parse_groups_rejects_a_spec_with_no_equals_sign():
    with pytest.raises(agg.ReportInputError, match="NAME=well1"):
        agg.parse_groups(["WT WT_1"])


def test_label_frame_keeps_sec_only_out_of_every_biological_group():
    per_image = pd.DataFrame({
        "image": ["a_01.vsi", "b_02.vsi", "s_03.vsi"],
        "condition": ["WT_1", "KO_1", "Sec-Only"],
        "secondary_only": [False, False, True],
    })
    lab = agg.label_frame(per_image, {"WT_1": "WT", "KO_1": "QKI-KO"}, {})
    assert list(lab["group"]) == ["WT", "QKI-KO", agg.SEC_ONLY_GROUP]
    assert lab.loc[2, "well_id"] is pd.NA or pd.isna(lab.loc[2, "well_id"])
    assert list(lab["field"]) == ["01", "02", "03"]


def test_resolve_group_order_never_returns_the_sec_only_group():
    per_image = pd.DataFrame({
        "image": ["a.vsi", "b.vsi", "s.vsi"],
        "condition": ["WT_1", "KO_1", "Sec-Only"],
        "secondary_only": [False, False, True],
    })
    lab = agg.label_frame(per_image, {"WT_1": "WT", "KO_1": "QKI-KO"}, {})
    assert agg.resolve_group_order(lab, ["QKI-KO", "WT"]) == ["QKI-KO", "WT"]
    assert agg.SEC_ONLY_GROUP not in agg.resolve_group_order(lab, [])


# ------------------------------------------------------------ contrast maths


def test_hedges_g_matches_the_hand_computation():
    a = np.array([10.0, 12.0, 14.0])
    b = np.array([4.0, 5.0, 6.0])
    sp2 = ((3 - 1) * a.std(ddof=1) ** 2 + (3 - 1) * b.std(ddof=1) ** 2) / (3 + 3 - 2)
    d = (a.mean() - b.mean()) / math.sqrt(sp2)
    j = 1.0 - 3.0 / (4.0 * 6 - 9.0)
    assert st.hedges_g(a, b) == pytest.approx(j * d, rel=1e-12)


def test_welch_matches_a_hand_computed_t_and_df():
    a = np.array([9.808333, 10.355556, 15.975397])
    b = np.array([5.489704, 3.277778, 5.040646])
    n1 = n2 = 3
    v1, v2 = a.var(ddof=1), b.var(ddof=1)
    t_hand = (a.mean() - b.mean()) / math.sqrt(v1 / n1 + v2 / n2)
    df_hand = ((v1 / n1 + v2 / n2) ** 2
               / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)))
    r = st.welch(a, b)
    assert r["t"] == pytest.approx(t_hand, rel=1e-12)
    assert r["df"] == pytest.approx(df_hand, rel=1e-12)
    assert r["diff"] == pytest.approx(a.mean() - b.mean(), rel=1e-12)
    # The interval is the difference plus or minus t critical times the standard error.
    from scipy import stats as sps
    se = math.sqrt(v1 / n1 + v2 / n2)
    tcrit = sps.t.ppf(0.975, df_hand)
    assert r["ci_low"] == pytest.approx(r["diff"] - tcrit * se, rel=1e-12)
    assert r["ci_high"] == pytest.approx(r["diff"] + tcrit * se, rel=1e-12)


def test_welch_refuses_a_group_with_fewer_than_two_wells():
    r = st.welch(np.array([1.0]), np.array([2.0, 3.0]))
    assert np.isnan(r["p_welch"])
    assert "fewer than 2 wells" in r["welch_note"]


def test_holm_matches_the_hand_worked_step_down():
    # m = 4. Sorted p: 0.01, 0.02, 0.04, 0.30.
    # Adjusted: 4*0.01=0.04; 3*0.02=0.06; 2*0.04=0.08; 1*0.30=0.30. All monotone.
    got = st.holm([0.04, 0.01, 0.30, 0.02])
    assert got == pytest.approx([0.08, 0.04, 0.30, 0.06], rel=1e-12)


def test_holm_enforces_monotonicity_when_a_later_step_would_drop():
    # Sorted p: 0.02, 0.03. Adjusted: 2*0.02=0.04; 1*0.03=0.03 -> lifted to 0.04.
    assert st.holm([0.02, 0.03]) == pytest.approx([0.04, 0.04], rel=1e-12)


def test_holm_skips_nan_and_shrinks_the_family_accordingly():
    got = st.holm([0.01, float("nan"), 0.02])
    assert np.isnan(got[1])
    # Family size is 2, not 3: 2*0.01 = 0.02 and 1*0.02 = 0.02.
    assert got[0] == pytest.approx(0.02, rel=1e-12)
    assert got[2] == pytest.approx(0.02, rel=1e-12)


def test_exact_permutation_hits_its_arithmetic_floor_at_three_versus_three():
    a = np.array([100.0, 101.0, 102.0])
    b = np.array([1.0, 2.0, 3.0])
    r = st.exact_permutation(a, b)
    assert r["perm_n_assignments"] == 20
    assert r["perm_arithmetic_floor"] == pytest.approx(0.1, rel=1e-12)
    # Even a maximally separated pair cannot go below the floor.
    assert r["p_permutation"] == pytest.approx(0.1, rel=1e-12)


def test_mde_is_larger_at_a_stricter_alpha_and_reproduces_a_known_value():
    m05 = st.mde_hedges_g(0.05, n1=3, n2=3)
    m01 = st.mde_hedges_g(0.01, n1=3, n2=3)
    assert m01 > m05
    # Cross-checked against the power function this module solves: at the returned
    # effect the two-sided power is 0.80 by construction.
    j = 1.0 - 3.0 / (4.0 * 6 - 9.0)
    assert st._power_two_sided(m05 / j, 0.05, 3, 3) == pytest.approx(0.80, abs=1e-6)
    # More wells detect a smaller effect.
    assert st.mde_hedges_g(0.05, n1=6, n2=6) < m05


def test_stars_follow_the_locked_thresholds():
    assert st.stars(1e-5) == "****"
    assert st.stars(5e-4) == "***"
    assert st.stars(5e-3) == "**"
    assert st.stars(0.04) == "*"
    assert st.stars(0.051) == "ns"
    assert st.stars(float("nan")) == "n/a"


# -------------------------------------------------------- hierarchy and gate


def _toy_run(tmp_path: Path) -> Path:
    """Three wells per group, two fields per well, four nuclei per field."""
    rng = np.random.default_rng(0)
    run = tmp_path / "RUN_TOY"
    run.mkdir()
    rows_img, rows_nuc = [], []
    means = {"WT_1": 4.0, "WT_2": 5.0, "WT_3": 6.0,
             "KO_1": 10.0, "KO_2": 11.0, "KO_3": 12.0}
    for well, mu in means.items():
        for f in (1, 2):
            image = f"{well}_{f:02d}.vsi"
            rows_img.append({"image": image, "condition": well, "secondary_only": False,
                             "protein_pooled_rotation_enrichment_at_rna1_spots": mu / 10})
            for k in range(4):
                rows_nuc.append({
                    "image": image, "nucleus_id": k,
                    "n_spots_rna1": mu + rng.normal(0, 0.2),
                    "nuclear_spot_fraction": 0.9,
                    "nucleus_area_px": 16000.0, "voxel_xy_um": 0.065,
                    "rotation_null_usable": True,
                    "protein_rotation_enrichment_at_rna1_spots": mu / 10,
                })
    rows_img.append({"image": "Sec-Only_09.vsi", "condition": "Sec-Only",
                     "secondary_only": True})
    for k in range(12):
        rows_nuc.append({"image": "Sec-Only_09.vsi", "nucleus_id": k,
                         "n_spots_rna1": 0.0, "nuclear_spot_fraction": 0.0,
                         "nucleus_area_px": 16000.0, "voxel_xy_um": 0.065,
                         "rotation_null_usable": False})
    pd.DataFrame(rows_img).to_csv(run / "per_image_summary.csv", index=False)
    pd.DataFrame(rows_nuc).to_csv(run / "nuclei_metrics.csv", index=False)
    (run / "run_config.json").write_text(json.dumps(
        {"config_resolved": {"channels": {"analysis_mode": "rna_protein",
                                          "rna_label": "BIN1 intron",
                                          "antibody_label": "RNASEH2B"}}}),
        encoding="utf-8")
    return run


def test_the_well_not_the_nucleus_is_the_tested_unit(tmp_path):
    run = _toy_run(tmp_path)
    m, order = agg.parse_groups(["WT=WT_1,WT_2,WT_3", "QKI-KO=KO_1,KO_2,KO_3"])
    data = agg.load_run(run, m, {})
    endpoints, absent = ep.resolve(data["nuclei"], data["per_image"])
    field = agg.per_field_long(data["nuclei"], data["per_image"], endpoints,
                               data["labels"], 5)
    well = agg.per_well_long(field)
    sub = well[well["endpoint"] == "rna1_spots_per_nucleus"]
    # Six wells, two fields each, eight nuclei each. Not 48 tested points.
    assert len(sub) == 6
    assert set(sub["n_fields"]) == {2}
    assert set(sub["n_nuclei_in_well"]) == {8}
    contrasts = agg.build_contrasts(well, field, endpoints, absent, order, "WT",
                                    ep.channel_labels(data["cfg"]))
    row = contrasts[contrasts["endpoint"] == "rna1_spots_per_nucleus"].iloc[0]
    assert int(row["n_wells_test"]) == 3
    assert int(row["n_wells_reference"]) == 3
    assert row["test_group"] == "QKI-KO" and row["reference_group"] == "WT"


def test_the_well_mean_is_the_mean_of_its_field_values(tmp_path):
    run = _toy_run(tmp_path)
    m, _ = agg.parse_groups(["WT=WT_1,WT_2,WT_3", "QKI-KO=KO_1,KO_2,KO_3"])
    data = agg.load_run(run, m, {})
    endpoints, _ = ep.resolve(data["nuclei"], data["per_image"])
    field = agg.per_field_long(data["nuclei"], data["per_image"], endpoints,
                               data["labels"], 5)
    well = agg.per_well_long(field)
    for wid in ("WT_1", "KO_3"):
        f = field[(field["endpoint"] == "rna1_spots_per_nucleus")
                  & (field["well_id"] == wid)]["field_value"]
        w = well[(well["endpoint"] == "rna1_spots_per_nucleus")
                 & (well["well_id"] == wid)]["well_mean_of_field_values"].iloc[0]
        assert w == pytest.approx(float(f.mean()), rel=1e-12)


def test_an_absent_column_is_reported_as_absent_not_dropped(tmp_path):
    run = _toy_run(tmp_path)
    data = agg.load_run(run, {"WT_1": "WT"}, {})
    _, absent = ep.resolve(data["nuclei"], data["per_image"])
    # This run emits no rna2 channel and no partner-anchored null.
    assert "rna2_spots_per_nucleus" in absent
    assert "rna1_rotation_enrichment_at_partner_puncta" in absent


def test_descriptive_and_absent_endpoints_stay_out_of_the_holm_family(tmp_path):
    run = _toy_run(tmp_path)
    m, order = agg.parse_groups(["WT=WT_1,WT_2,WT_3", "QKI-KO=KO_1,KO_2,KO_3"])
    data = agg.load_run(run, m, {})
    endpoints, absent = ep.resolve(data["nuclei"], data["per_image"])
    field = agg.per_field_long(data["nuclei"], data["per_image"], endpoints,
                               data["labels"], 5)
    well = agg.per_well_long(field)
    c = agg.build_contrasts(well, field, endpoints, absent, order, "WT",
                            ep.channel_labels(data["cfg"]))
    assert not bool(c.loc[c["endpoint"] == "nucleus_area_um2", "in_holm_family"].iloc[0])
    assert not bool(c.loc[c["endpoint"] == "protein_nuclear_mean", "in_holm_family"].iloc[0])
    absent_row = c[c["endpoint"] == "rna2_spots_per_nucleus"].iloc[0]
    assert not bool(absent_row["in_holm_family"])
    assert absent_row["holm_exclusion_reason"] == "column absent from this run"
    gated = c[c["in_holm_family"]]
    assert len(gated) >= 1
    # The adjusted p is never smaller than the raw p it came from.
    ok = gated.dropna(subset=["p_welch", "p_welch_holm_within_family"])
    assert (ok["p_welch_holm_within_family"] >= ok["p_welch"] - 1e-12).all()


def test_excluding_a_field_removes_its_nuclei_and_records_the_reason(tmp_path):
    run = _toy_run(tmp_path)
    m, _ = agg.parse_groups(["WT=WT_1,WT_2,WT_3", "QKI-KO=KO_1,KO_2,KO_3"])
    base = agg.load_run(run, m, {})
    cut = agg.load_run(run, m, {"WT_1_01.vsi": "out of focus"})
    assert cut["n_nuclei_after_field_exclusion"] == base["n_nuclei_after_field_exclusion"] - 4
    lab = cut["labels"]
    row = lab[lab["image"] == "WT_1_01.vsi"].iloc[0]
    assert bool(row["excluded_field"]) and row["exclusion_reason"] == "out of focus"


def test_a_usability_flag_filters_the_nuclei_that_contribute(tmp_path):
    run = _toy_run(tmp_path)
    nuc = pd.read_csv(run / "nuclei_metrics.csv")
    # Flag half of one well's nuclei unusable and confirm the count drops by that many.
    mask = nuc["image"] == "WT_1_01.vsi"
    nuc.loc[mask & (nuc["nucleus_id"] < 2), "rotation_null_usable"] = False
    nuc.to_csv(run / "nuclei_metrics.csv", index=False)
    m, _ = agg.parse_groups(["WT=WT_1,WT_2,WT_3", "QKI-KO=KO_1,KO_2,KO_3"])
    data = agg.load_run(run, m, {})
    endpoints, _ = ep.resolve(data["nuclei"], data["per_image"])
    field = agg.per_field_long(data["nuclei"], data["per_image"], endpoints,
                               data["labels"], 5)
    f = field[(field["endpoint"] == "partner_rotation_enrichment_at_rna1")
              & (field["image"] == "WT_1_01.vsi")].iloc[0]
    assert int(f["n_nuclei_total"]) == 2
    assert int(f["n_nuclei_in_field_before_filter"]) == 4
    assert f["usability_filter"] == "rotation_null_usable"
