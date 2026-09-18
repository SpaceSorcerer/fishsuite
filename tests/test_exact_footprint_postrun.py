from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _endpoint_fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nuclei = pd.DataFrame(
        {
            "image": ["field.vsi"] * 3,
            "image_key": ["field.vsi"] * 3,
            "nucleus_id": [1, 2, 3],
            "slide": [1, 1, 1],
            "arm": ["NT", "NT", "NT"],
            "replicate": [1, 1, 1],
            "fov": [1, 1, 1],
            "biological_set": ["S1_NT_1"] * 3,
            "is_control": [False, False, False],
            "eligible_for_sampling": [True, True, True],
            "sampled_in_analysis": [True, False, True],
            "whole_nucleus_miat_sum_raw": [100.0, 200.0, 300.0],
            "whole_nucleus_qki_sum_raw": [40.0, 50.0, 60.0],
            "nucleoplasm_qki_mean_raw": [4.0, 5.0, 6.0],
        }
    )
    spots = pd.DataFrame(
        {
            "spot_uid": ["s1", "s2", "s3", "s4", "s5"],
            "image_key": ["field.vsi"] * 5,
            "nucleus_id": [1, 1, 1, 1, 2],
            # Deliberately carries the historical image-level bug for nucleus 2.
            "eligible_for_sampling": [True] * 5,
            "sampled_in_analysis": [True] * 5,
            "passes_miat_floor": [True, True, True, False, True],
            "footprint_full_nucleus_valid": [True] * 5,
            "stored_in_nucleolus": [False, False, False, False, True],
            "null_candidate": [True, True, True, False, False],
            "null_usable": [True, True, False, False, False],
            "null_exclusion_reason": [
                "",
                "",
                "low_first_pass_retention",
                "footprint_not_full_nucleoplasm",
                "nucleolar_spot",
            ],
            "miat_footprint_sum_raw": [11.0, 13.0, 17.0, 19.0, 23.0],
            "qki_footprint_sum_raw": [3.0, 5.0, 7.0, 11.0, 13.0],
            "miat_footprint_mean_raw": [5.5, 6.5, 8.5, 9.5, 11.5],
            "qki_footprint_mean_raw": [1.5, 2.5, 3.5, 5.5, 6.5],
            "miat_footprint_enrichment_vs_nucleus": [1.1, 1.2, 1.3, 1.4, 1.5],
            "qki_footprint_enrichment_vs_nucleus": [0.8, 1.0, 1.2, 1.4, 1.6],
            "qki_footprint_enrichment_vs_nucleoplasm": [0.7, 0.9, 1.1, 1.3, 1.5],
            "population_label_q90": [
                "threshold_positive",
                "threshold_negative",
                "unusable",
                "below_miat_floor",
                "unusable",
            ],
            "population_label_q95": [
                "threshold_positive",
                "threshold_negative",
                "unusable",
                "below_miat_floor",
                "unusable",
            ],
            "population_label_q99": [
                "threshold_negative",
                "threshold_negative",
                "unusable",
                "below_miat_floor",
                "unusable",
            ],
            "qki_threshold_positive_q90": [True, False, pd.NA, pd.NA, pd.NA],
            "qki_threshold_positive_q95": [True, False, pd.NA, pd.NA, pd.NA],
            "qki_threshold_positive_q99": [False, False, pd.NA, pd.NA, pd.NA],
        }
    )
    pixels = pd.DataFrame(
        {
            "spot_uid": ["s1", "s1", "s2", "s2", "s3", "s4", "s5"],
            "flat_pixel_index": [10, 11, 11, 12, 13, 14, 20],
            "miat_raw": [5.0, 6.0, 6.0, 7.0, 17.0, 19.0, 23.0],
            "qki_raw": [1.0, 2.0, 2.0, 3.0, 7.0, 11.0, 13.0],
        }
    )
    return nuclei, spots, pixels


def test_build_nucleus_endpoints_uses_authoritative_keys_and_keeps_zero_spot_nuclei():
    """Catches spot-first aggregation and trusting the historical image-level flag."""

    from fishsuite.core.exact_footprint_postrun import build_nucleus_endpoints

    nuclei, spots, pixels = _endpoint_fixture()
    corrected, audit, result = build_nucleus_endpoints(nuclei, spots, pixels)

    assert corrected.loc[corrected["nucleus_id"].eq(2), "sampled_in_analysis"].eq(False).all()
    assert audit["n_spot_sampling_mismatches_corrected"] == 1
    assert result["nucleus_id"].tolist() == [1, 2, 3]
    zero = result.loc[result["nucleus_id"].eq(3)].iloc[0]
    assert zero["n_spots_all"] == 0
    assert zero["n_spots_floor"] == 0
    assert zero["miat_footprint_mass_all_spot_summed"] == 0.0
    assert np.isnan(zero["association_fraction_among_usable_q95"])


def test_build_nucleus_endpoints_reports_every_threshold_population_and_mass_kind():
    """Catches q95-only summaries and double-counting overlapping union pixels."""

    from fishsuite.core.exact_footprint_postrun import build_nucleus_endpoints

    nuclei, spots, pixels = _endpoint_fixture()
    _, _, result = build_nucleus_endpoints(nuclei, spots, pixels)
    one = result.loc[result["nucleus_id"].eq(1)].iloc[0]

    assert one["n_spots_all"] == 4
    assert one["n_spots_floor"] == 3
    assert one["n_threshold_positive_q95"] == 1
    assert one["n_threshold_negative_q95"] == 1
    assert one["n_unusable_q95"] == 1
    assert one["n_below_floor"] == 1
    assert one["association_fraction_among_usable_q95"] == pytest.approx(0.5)
    assert one["association_fraction_among_all_floor_spots_q95"] == pytest.approx(1 / 3)
    assert one["n_threshold_positive_q99"] == 0
    assert one["n_threshold_negative_q99"] == 2
    assert one["miat_footprint_mass_all_spot_summed"] == pytest.approx(60.0)
    # s1 and s2 overlap at pixel 11: the union is 5 + 6 + 7, not 11 + 13.
    assert one["miat_footprint_mass_q95_usable_union_deduplicated"] == pytest.approx(18.0)
    assert one["qki_footprint_mass_q95_positive_union_deduplicated"] == pytest.approx(3.0)
    assert one["associated_miat_mass_fraction_among_usable_q95"] == pytest.approx(11 / 24)
    assert one["associated_miat_mass_fraction_among_all_floor_spots_q95"] == pytest.approx(11 / 41)
    assert one["qki_footprint_mean_all"] == pytest.approx(3.25)
    assert one["qki_footprint_mean_q95_positive"] == pytest.approx(1.5)
    assert one["qki_enrichment_vs_nucleus_mean_q95_positive"] == pytest.approx(0.8)
    assert one["qki_enrichment_vs_nucleoplasm_mean_q95_positive"] == pytest.approx(0.7)
    assert one["n_excluded_low_first_pass_retention"] == 1
    assert one["n_excluded_footprint_not_full_nucleoplasm"] == 0
    assert bool(one["population_reconciliation_pass_q90"])
    assert bool(one["population_reconciliation_pass_q95"])
    assert bool(one["population_reconciliation_pass_q99"])
    assert one["whole_nucleus_miat_sum_raw"] == pytest.approx(100.0)


def _hierarchy_fixture() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for slide in (1, 2):
        for arm in ("NT", "KD"):
            for replicate in (1, 2, 3):
                biological_set = f"S{slide}_{arm}_{replicate}"
                base = 10.0 * slide + replicate + (2.0 if arm == "KD" else 0.0)
                fov_values = [[base]]
                if biological_set == "S1_NT_1":
                    # Equal-FOV result is mean(mean(0, 10), mean(100)) = 52.5.
                    fov_values = [[0.0, 10.0], [100.0]]
                for fov, values in enumerate(fov_values, start=1):
                    for nucleus_id, value in enumerate(values, start=1):
                        rows.append(
                            {
                                "image": f"{biological_set}_f{fov}.vsi",
                                "image_key": f"{biological_set}_f{fov}.vsi".casefold(),
                                "nucleus_id": nucleus_id,
                                "slide": slide,
                                "arm": arm,
                                "replicate": replicate,
                                "fov": fov,
                                "biological_set": biological_set,
                                "is_control": False,
                                "eligible_for_sampling": True,
                                "sampled_in_analysis": True,
                                "metric": value,
                            }
                        )
    # A control and an unsampled biological nucleus must not enter primary means.
    rows.append(
        {
            "image": "control.vsi",
            "image_key": "control.vsi",
            "nucleus_id": 1,
            "slide": 2,
            "arm": "CONTROL",
            "replicate": 0,
            "fov": 1,
            "biological_set": "",
            "is_control": True,
            "eligible_for_sampling": True,
            "sampled_in_analysis": True,
            "metric": 10000.0,
        }
    )
    rows.append(
        {
            "image": "S2_KD_3_f1.vsi",
            "image_key": "s2_kd_3_f1.vsi",
            "nucleus_id": 99,
            "slide": 2,
            "arm": "KD",
            "replicate": 3,
            "fov": 1,
            "biological_set": "S2_KD_3",
            "is_control": False,
            "eligible_for_sampling": True,
            "sampled_in_analysis": False,
            "metric": 10000.0,
        }
    )
    return pd.DataFrame(rows)


def test_aggregate_endpoint_hierarchy_weights_nuclei_then_fovs_and_returns_12_sets():
    """Catches pseudoreplication and weighting a two-nucleus FOV twice."""

    from fishsuite.core.exact_footprint_postrun import aggregate_endpoint_hierarchy

    fov, sets = aggregate_endpoint_hierarchy(
        _hierarchy_fixture(), endpoints=["metric"], cohorts=["sampled_primary"]
    )

    assert len(sets) == 12
    assert sets.groupby(["cohort", "endpoint"]).size().to_dict() == {
        ("sampled_primary", "metric"): 12
    }
    target = sets.loc[sets["biological_set"].eq("S1_NT_1")].iloc[0]
    assert target["value"] == pytest.approx(52.5)
    assert target["n_fovs_total"] == 2
    assert target["n_nuclei_total"] == 3
    assert not fov["is_control"].any()
    assert fov.loc[fov["biological_set"].eq("S2_KD_3"), "value"].iloc[0] < 100.0


def _set_long_fixture(endpoint: str = "metric") -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for slide in (1, 2):
        for arm in ("NT", "KD"):
            for replicate in (1, 2, 3):
                value = 10.0 * slide + replicate + (2.0 if arm == "KD" else 0.0)
                rows.append(
                    {
                        "cohort": "sampled_primary",
                        "endpoint": endpoint,
                        "slide": slide,
                        "arm": arm,
                        "replicate": replicate,
                        "biological_set": f"S{slide}_{arm}_{replicate}",
                        "n_fovs_total": 1,
                        "n_fovs_finite": 1,
                        "n_nuclei_total": 10,
                        "value": value,
                        "complete_for_inference": True,
                    }
                )
    return pd.DataFrame(rows)


def test_build_endpoint_inference_uses_unpaired_sets_and_exact_400_and_200_designs():
    """Catches NT1-to-KD1 pairing or treating Cartesian contrasts as replicates."""

    from fishsuite.core.exact_footprint_postrun import build_endpoint_inference

    result = build_endpoint_inference(
        _set_long_fixture(),
        endpoint_scales={"metric": "unsigned_positive"},
        include_s1_kd1_exclusion_sensitivity=True,
    )
    primary = result.loc[result["analysis_role"].eq("primary")].iloc[0]
    sensitivity = result.loc[
        result["analysis_role"].eq("s1_kd1_whole_set_exclusion_sensitivity")
    ].iloc[0]

    assert primary["n_nt"] == 6
    assert primary["n_kd"] == 6
    assert primary["difference_kd_minus_nt"] == pytest.approx(2.0)
    assert primary["slide_adjusted_coefficient_kd_minus_nt"] == pytest.approx(2.0)
    assert primary["permutation_n_permutations"] == 400
    assert primary["pairing_semantics"] == "none_independent_biological_sets"
    assert np.isfinite(primary["ratio_kd_over_nt"])
    assert sensitivity["n_nt"] == 6
    assert sensitivity["n_kd"] == 5
    assert sensitivity["permutation_n_permutations"] == 200


def test_inference_suppresses_ratios_for_signed_endpoints_and_gates_incomplete_q99():
    """Catches invalid ratios and silently running a partial q99 analysis."""

    from fishsuite.core.exact_footprint_postrun import build_endpoint_inference

    signed = _set_long_fixture("fisher_z_correlation")
    signed["value"] -= 20.0
    complete = build_endpoint_inference(
        signed,
        endpoint_scales={"fisher_z_correlation": "signed_additive"},
    ).iloc[0]
    assert np.isnan(complete["ratio_kd_over_nt"])
    assert np.isnan(complete["percent_change_kd_vs_nt"])

    q99 = _set_long_fixture("corr_threshold_positive_q99_raw_pearson_fisher_z")
    q99.loc[q99.index[0], ["n_fovs_finite", "complete_for_inference"]] = [0, False]
    q99.loc[q99.index[0], "value"] = np.nan
    gated = build_endpoint_inference(
        q99,
        endpoint_scales={
            "corr_threshold_positive_q99_raw_pearson_fisher_z": "signed_additive"
        },
    ).iloc[0]
    assert gated["inference_status"] == "descriptive_only_incomplete"
    assert gated["n_complete_sets"] == 11
    assert np.isnan(gated["welch_p_two_sided"])
    assert np.isnan(gated["permutation_p_exact_two_sided"])

    invalid_unsigned = _set_long_fixture("count")
    invalid_unsigned.loc[invalid_unsigned.index[0], "value"] = -1.0
    with pytest.raises(ValueError, match="nonnegative"):
        build_endpoint_inference(
            invalid_unsigned,
            endpoint_scales={"count": "unsigned_positive"},
        )


def test_cartesian_contrasts_are_18_descriptive_rows_without_pairing_or_pvalues():
    """Catches replicate-label pairing and inferential use of the 3x3 grid."""

    from fishsuite.core.exact_footprint_postrun import build_cartesian_contrasts

    result = build_cartesian_contrasts(
        _set_long_fixture(),
        endpoint_scales={"metric": "unsigned_positive"},
        include_s1_kd1_exclusion_sensitivity=True,
    )
    primary = result.loc[result["analysis_role"].eq("primary_descriptive_full_cohort")]
    sensitivity = result.loc[result["analysis_role"].eq("declared_incomplete_sensitivity")]
    assert len(primary) == 18
    assert primary.groupby("slide").size().to_dict() == {1: 9, 2: 9}
    assert len(sensitivity) == 6
    assert sensitivity["slide"].eq(1).all()
    assert result["pairing_semantics"].eq("none_cartesian_cross_set_descriptive").all()
    assert result["inferential_test"].eq("none_descriptive_only").all()
    assert not any("p_value" in column or column.startswith("p_") for column in result)


def test_ratio_of_ratios_uses_arm_means_and_within_arm_covariance_without_pairing():
    """Catches forming six arbitrary NT/KD replicate-number pairs."""

    from fishsuite.core.exact_footprint_postrun import ratio_of_ratios

    base = _set_long_fixture("global")
    associated = base.copy()
    associated["endpoint"] = "associated"
    # Global KD/NT is 0.5; associated KD/NT is 0.25; ratio-of-ratios is 0.5.
    base["value"] = np.where(base["arm"].eq("NT"), 100.0, 50.0)
    associated["value"] = np.where(associated["arm"].eq("NT"), 20.0, 5.0)
    result = ratio_of_ratios(
        pd.concat([base, associated], ignore_index=True),
        numerator_endpoint="associated",
        denominator_endpoint="global",
        cohort="sampled_primary",
        endpoint_scales={
            "associated": "unsigned_positive",
            "global": "unsigned_positive",
        },
    )

    assert result["ratio_of_ratios"] == pytest.approx(0.5)
    assert result["log_ratio_of_ratios"] == pytest.approx(np.log(0.5))
    assert result["pairing_semantics"] == "none_arm_mean_ratio_of_ratios"
    assert result["ci_method"] == "multivariate_delta_within_arm_covariance"
    assert result["ratio_of_ratios_ci95_low"] == pytest.approx(0.5)
    assert result["ratio_of_ratios_ci95_high"] == pytest.approx(0.5)

    with pytest.raises(ValueError, match="unsigned_positive"):
        ratio_of_ratios(
            pd.concat([base, associated], ignore_index=True),
            numerator_endpoint="associated",
            denominator_endpoint="global",
            cohort="sampled_primary",
            endpoint_scales={
                "associated": "signed_additive",
                "global": "unsigned_positive",
            },
        )


def _correlation_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    nuclei_rows: list[dict[str, object]] = []
    spot_rows: list[dict[str, object]] = []
    for slide in (1, 2):
        for arm in ("NT", "KD"):
            for replicate in (1, 2, 3):
                biological_set = f"S{slide}_{arm}_{replicate}"
                image_key = f"{biological_set}_f1.vsi".casefold()
                nuclei_rows.append(
                    {
                        "image": image_key,
                        "image_key": image_key,
                        "nucleus_id": 1,
                        "slide": slide,
                        "arm": arm,
                        "replicate": replicate,
                        "fov": 1,
                        "biological_set": biological_set,
                        "is_control": False,
                        "eligible_for_sampling": True,
                        "sampled_in_analysis": True,
                    }
                )
                for spot_id, (miat, qki) in enumerate(
                    [(1.0, 1.2), (2.0, 2.7), (3.0, 2.1), (4.0, 4.8)], start=1
                ):
                    q99_positive = not (
                        biological_set == "S1_NT_1" and spot_id > 2
                    )
                    spot_rows.append(
                        {
                            "spot_uid": f"{image_key}:{spot_id}",
                            "image_key": image_key,
                            "nucleus_id": 1,
                            # Deliberately not used for cohort membership.
                            "eligible_for_sampling": True,
                            "sampled_in_analysis": True,
                            "footprint_full_nucleus_valid": True,
                            "null_usable": True,
                            "miat_footprint_mean_raw": miat,
                            "qki_footprint_mean_raw": qki + (0.1 if arm == "KD" else 0.0),
                            "miat_footprint_enrichment_vs_nucleus": miat / 2,
                            "qki_footprint_enrichment_vs_nucleus": qki / 3,
                            "population_label_q90": "threshold_positive",
                            "population_label_q95": "threshold_positive",
                            "population_label_q99": (
                                "threshold_positive" if q99_positive else "threshold_negative"
                            ),
                        }
                    )
    return pd.DataFrame(nuclei_rows), pd.DataFrame(spot_rows)


def test_correlation_hierarchy_uses_all_exact_and_threshold_populations_with_fisher_z():
    """Catches nucleus-level averaging and omitting threshold-only correlations."""

    from fishsuite.core.exact_footprint_postrun import (
        aggregate_correlation_hierarchy,
        compute_fov_correlations,
    )

    nuclei, spots = _correlation_fixture()
    fov = compute_fov_correlations(spots, nuclei, cohorts=["sampled_primary"])
    sets = aggregate_correlation_hierarchy(fov)

    # 37 is the real design; this synthetic fixture has 12 FOVs and 16 rows/FOV.
    assert len(fov) == 12 * 4 * 2 * 2
    assert set(fov["population"]) == {
        "all_detected_exact_valid",
        "threshold_positive_q90",
        "threshold_positive_q95",
        "threshold_positive_q99",
    }
    all_raw_pearson = sets.loc[
        sets["population"].eq("all_detected_exact_valid")
        & sets["measurement_pair"].eq("raw")
        & sets["correlation_method"].eq("pearson")
    ]
    assert len(all_raw_pearson) == 12
    assert all_raw_pearson["complete_for_inference"].all()
    assert np.allclose(
        all_raw_pearson["correlation_backtransformed"],
        np.tanh(all_raw_pearson["fisher_z_mean"]),
    )
    q99 = sets.loc[
        sets["population"].eq("threshold_positive_q99")
        & sets["measurement_pair"].eq("raw")
        & sets["correlation_method"].eq("pearson")
    ]
    assert q99["conditional_descriptive"].all()
    assert q99["complete_for_inference"].sum() == 11
    assert not bool(q99.loc[q99["biological_set"].eq("S1_NT_1"), "complete_for_inference"].iloc[0])
    threshold_fov = fov.loc[
        fov["population"].eq("threshold_positive_q95")
        & fov["measurement_pair"].eq("raw")
        & fov["correlation_method"].eq("pearson")
    ]
    assert threshold_fov["p_value"].isna().all()


def test_exact_boundary_correlation_is_explicitly_nonestimable_for_fisher_aggregation():
    """Catches clipping r=1 before arctanh, which invents a finite Fisher value."""

    from fishsuite.core.exact_footprint_postrun import compute_fov_correlations

    nuclei, spots = _correlation_fixture()
    target = spots["image_key"].eq("s1_nt_1_f1.vsi")
    spots.loc[target, "qki_footprint_mean_raw"] = spots.loc[
        target, "miat_footprint_mean_raw"
    ].to_numpy()
    result = compute_fov_correlations(spots, nuclei, cohorts=["sampled_primary"])
    row = result.loc[
        result["image_key"].eq("s1_nt_1_f1.vsi")
        & result["population"].eq("all_detected_exact_valid")
        & result["measurement_pair"].eq("raw")
        & result["correlation_method"].eq("pearson")
    ].iloc[0]

    assert row["correlation"] == pytest.approx(1.0)
    assert not bool(row["fisher_estimable"])
    assert np.isnan(row["fisher_z"])
    assert row["nonestimable_reason"] == "exact_boundary_correlation"


def test_fov_correlation_membership_is_authoritative_after_spot_filtering():
    """Catches reusing pre-merge row indices and admitting an unsampled nucleus."""

    from fishsuite.core.exact_footprint_postrun import compute_fov_correlations

    nuclei, spots = _correlation_fixture()
    extra_nucleus = nuclei.iloc[[0]].copy()
    extra_nucleus["nucleus_id"] = 2
    extra_nucleus["sampled_in_analysis"] = False
    nuclei = pd.concat([nuclei, extra_nucleus], ignore_index=True)
    unsampled = spots.iloc[[0]].copy()
    unsampled["spot_uid"] = "unsampled-first"
    unsampled["nucleus_id"] = 2
    unsampled["footprint_full_nucleus_valid"] = False
    spots = pd.concat([unsampled, spots], ignore_index=True)

    result = compute_fov_correlations(spots, nuclei, cohorts=["sampled_primary"])
    row = result.loc[
        result["image_key"].eq("s1_nt_1_f1.vsi")
        & result["population"].eq("all_detected_exact_valid")
        & result["measurement_pair"].eq("raw")
        & result["correlation_method"].eq("pearson")
    ].iloc[0]
    assert row["n_spots"] == 4


def test_correlation_inference_is_primary_only_for_all_spots_and_gates_q99():
    """Catches p-values on conditional threshold-selected spots and partial q99 sets."""

    from fishsuite.core.exact_footprint_postrun import (
        aggregate_correlation_hierarchy,
        build_correlation_inference,
        compute_fov_correlations,
    )

    nuclei, spots = _correlation_fixture()
    sets = aggregate_correlation_hierarchy(
        compute_fov_correlations(spots, nuclei, cohorts=["sampled_primary"])
    )
    result = build_correlation_inference(sets)
    all_spots = result.loc[
        result["population"].eq("all_detected_exact_valid")
        & result["measurement_pair"].eq("raw")
        & result["correlation_method"].eq("pearson")
    ].iloc[0]
    q95 = result.loc[
        result["population"].eq("threshold_positive_q95")
        & result["measurement_pair"].eq("raw")
        & result["correlation_method"].eq("pearson")
    ].iloc[0]
    q99 = result.loc[
        result["population"].eq("threshold_positive_q99")
        & result["measurement_pair"].eq("raw")
        & result["correlation_method"].eq("pearson")
    ].iloc[0]

    assert all_spots["inference_status"] == "complete"
    assert all_spots["permutation_n_permutations"] == 400
    assert np.isfinite(all_spots["welch_p_two_sided"])
    assert q95["inference_status"] == "descriptive_conditional"
    assert np.isnan(q95["welch_p_two_sided"])
    assert q99["inference_status"] == "descriptive_only_incomplete_conditional"
    assert q99["n_complete_sets"] == 11
    assert np.isnan(q99["permutation_p_exact_two_sided"])


def _full_postrun_fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nuclei, spots = _correlation_fixture()
    nuclei["whole_nucleus_miat_sum_raw"] = 100.0
    nuclei["whole_nucleus_qki_sum_raw"] = 50.0
    nuclei["nucleoplasm_qki_mean_raw"] = 5.0
    nuclei["z_mode"] = "autofocus"
    nuclei["z_range"] = "5-5"
    nuclei["n_z_slices"] = 10
    spots["passes_miat_floor"] = True
    spots["stored_in_nucleolus"] = False
    spots["null_candidate"] = True
    spots["null_exclusion_reason"] = ""
    spots["miat_footprint_sum_raw"] = spots["miat_footprint_mean_raw"]
    spots["qki_footprint_sum_raw"] = spots["qki_footprint_mean_raw"]
    spots["qki_footprint_enrichment_vs_nucleoplasm"] = (
        spots["qki_footprint_enrichment_vs_nucleus"]
    )
    spots["selected_z_1based"] = 5
    spots["selected_z_0based"] = 4
    spots["quantitation_plane"] = "exact_recorded_single_z"
    spots["miat_channel_index"] = 0
    spots["qki_channel_index"] = 1
    spots["dapi_channel_index"] = 2
    for percentile in (90, 95, 99):
        spots[f"qki_threshold_positive_q{percentile}"] = spots[
            f"population_label_q{percentile}"
        ].eq("threshold_positive")

    # Add one unsampled nucleus/spot whose historical image-level flag is wrong.
    extra_nucleus = nuclei.iloc[[0]].copy()
    extra_nucleus["nucleus_id"] = 2
    extra_nucleus["sampled_in_analysis"] = False
    nuclei = pd.concat([nuclei, extra_nucleus], ignore_index=True)
    extra_spot = spots.iloc[[0]].copy()
    extra_spot["spot_uid"] = "historically-mislabeled-unsampled"
    extra_spot["nucleus_id"] = 2
    extra_spot["sampled_in_analysis"] = True
    spots = pd.concat([extra_spot, spots], ignore_index=True)

    pixels = pd.DataFrame(
        {
            "spot_uid": spots["spot_uid"].astype(str),
            "flat_pixel_index": np.arange(len(spots), dtype=int),
            "miat_raw": spots["miat_footprint_sum_raw"].to_numpy(float),
            "qki_raw": spots["qki_footprint_sum_raw"].to_numpy(float),
        }
    )
    return nuclei, spots, pixels


def test_validate_quantitation_invariants_accepts_locked_single_plane_evidence():
    """Catches manifest claims that are not derived from retained z/channel fields."""

    from fishsuite.core.exact_footprint_postrun import validate_quantitation_invariants

    nuclei, spots, _ = _full_postrun_fixture()
    audit = validate_quantitation_invariants(spots, nuclei)

    assert audit["validation_status"] == "pass"
    assert audit["n_nucleus_images"] == 12
    assert audit["n_spot_bearing_images"] == 12
    assert audit["selected_z_parity_pass"]
    assert audit["nucleus_singleton_z_range_pass"]
    assert audit["spot_nucleus_z_agreement_pass"]
    assert audit["quantitation_plane_values"] == ["exact_recorded_single_z"]
    assert audit["expected_channel_indices"] == {"miat": 0, "qki": 1, "dapi": 2}


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("multiple_spot_z", "single selected_z"),
        ("z_parity", "selected_z_0based"),
        ("channel_drift", "expected channel indices"),
        ("projection_plane", "single-plane quantitation"),
        ("nucleus_z_mode", "z_mode contradicts"),
        ("nucleus_z_range", "singleton z_range"),
        ("nucleus_spot_disagreement", "disagrees with nucleus z_range"),
    ],
)
def test_validate_quantitation_invariants_fails_closed_on_contradictions(
    mutation: str,
    message: str,
):
    """Catches accepting a projection, inconsistent z, or per-image channel drift."""

    from fishsuite.core.exact_footprint_postrun import validate_quantitation_invariants

    nuclei, spots, _ = _full_postrun_fixture()
    image = "s1_nt_1_f1.vsi"
    spot_gate = spots["image_key"].eq(image)
    nucleus_gate = nuclei["image_key"].eq(image)
    if mutation == "multiple_spot_z":
        spots.loc[spots.index[spot_gate][0], "selected_z_1based"] = 6
        spots.loc[spots.index[spot_gate][0], "selected_z_0based"] = 5
    elif mutation == "z_parity":
        spots.loc[spot_gate, "selected_z_0based"] = 5
    elif mutation == "channel_drift":
        spots.loc[spot_gate, "qki_channel_index"] = 2
    elif mutation == "projection_plane":
        spots.loc[spot_gate, "quantitation_plane"] = "small_maximum_intensity_projection"
    elif mutation == "nucleus_z_mode":
        nuclei.loc[nucleus_gate, "z_mode"] = "maximum_intensity_projection"
    elif mutation == "nucleus_z_range":
        nuclei.loc[nucleus_gate, "z_range"] = "4-5"
    elif mutation == "nucleus_spot_disagreement":
        nuclei.loc[nucleus_gate, "z_range"] = "6-6"
    with pytest.raises(ValueError, match=message):
        validate_quantitation_invariants(spots, nuclei)


def test_run_postrun_directory_writes_corrected_sampling_before_summaries_without_overwrite(
    tmp_path: Path,
):
    """Catches using the source spot flag or modifying the completed source run."""

    from fishsuite.core.exact_footprint_postrun import (
        DEFAULT_RATIO_OF_RATIOS_SPECS,
        run_postrun_directory,
    )

    nuclei, spots, pixels = _full_postrun_fixture()
    source = tmp_path / "completed_source_run"
    source.mkdir()
    spots.to_csv(source / "spot_exact_footprint_metrics.csv.gz", index=False)
    nuclei.to_csv(source / "nucleus_exact_footprint_metrics.csv", index=False)
    pixels.to_csv(source / "footprint_pixels.csv.gz", index=False)
    output = tmp_path / "postrun_revision"
    endpoints = [
        "n_spots_all",
        "n_spots_floor",
        "threshold_positive_spots_per_nucleus_q95",
        "miat_footprint_mass_q95_positive_spot_summed",
        "miat_footprint_mass_floor_spot_summed",
    ]
    endpoint_scales = {endpoint: "unsigned_positive" for endpoint in endpoints}

    with pytest.raises(ValueError, match="outside the completed source run"):
        run_postrun_directory(
            source,
            source / "forbidden_nested_revision",
            endpoints=endpoints,
            endpoint_scales=endpoint_scales,
        )

    result = run_postrun_directory(
        source,
        output,
        endpoints=endpoints,
        endpoint_scales=endpoint_scales,
    )

    expected = {
        "corrected_spot_exact_footprint_metrics.csv.gz",
        "sampling_correction_audit.json",
        "quantitation_invariant_audit.json",
        "nucleus_exact_footprint_endpoints.csv",
        "fov_endpoint_means.csv",
        "biological_set_endpoint_means.csv",
        "endpoint_inference.csv",
        "cartesian_contrasts.csv",
        "fov_correlations.csv",
        "biological_set_correlations.csv",
        "correlation_inference.csv",
        "ratio_of_ratios.csv",
        "postrun_manifest.json",
    }
    assert expected.issubset({path.name for path in output.iterdir()})
    audit = json.loads((output / "sampling_correction_audit.json").read_text())
    assert audit["n_spot_sampling_mismatches_corrected"] == 1
    quantitation_audit = json.loads(
        (output / "quantitation_invariant_audit.json").read_text()
    )
    assert quantitation_audit["validation_status"] == "pass"
    assert result.quantitation_audit == quantitation_audit
    assert len(result.ratio_of_ratios) == 4
    manifest = json.loads((output / "postrun_manifest.json").read_text())
    assert manifest["ratio_of_ratios_specs"] == [
        {
            "numerator_endpoint": numerator,
            "denominator_endpoint": denominator,
            "endpoint_scale": "unsigned_positive",
            "threshold_scope": "q95_primary" if "q95" in numerator else "global",
        }
        for numerator, denominator in DEFAULT_RATIO_OF_RATIOS_SPECS
    ]
    assert "q99" not in json.dumps(manifest["ratio_of_ratios_specs"])
    assert manifest["ratio_of_ratios_spec_source"] == "locked_default_q95_specs"
    assert manifest["ratio_of_ratios_cohorts"] == [
        "sampled_primary",
        "all_eligible",
    ]
    assert manifest["quantitation_invariant_audit"]["validation_status"] == "pass"
    corrected = pd.read_csv(output / "corrected_spot_exact_footprint_metrics.csv.gz")
    mislabeled = corrected.loc[
        corrected["spot_uid"].eq("historically-mislabeled-unsampled")
    ].iloc[0]
    assert not bool(mislabeled["sampled_in_analysis"])
    primary_fov = result.endpoint_fov_means.loc[
        result.endpoint_fov_means["image_key"].eq("s1_nt_1_f1.vsi")
        & result.endpoint_fov_means["endpoint"].eq("n_spots_all")
    ].iloc[0]
    assert primary_fov["value"] == pytest.approx(4.0)
    # The immutable source still contains the deliberately wrong historical value.
    source_spots = pd.read_csv(source / "spot_exact_footprint_metrics.csv.gz")
    assert bool(
        source_spots.loc[
            source_spots["spot_uid"].eq("historically-mislabeled-unsampled"),
            "sampled_in_analysis",
        ].iloc[0]
    )
    with pytest.raises(ValueError, match="already exists and is not empty"):
        run_postrun_directory(
            source,
            output,
            endpoints=endpoints,
            endpoint_scales=endpoint_scales,
        )
