from __future__ import annotations

import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from fishsuite.core.exact_footprint_superplots import (
    DEFAULT_ENDPOINT_FAMILIES,
    FigureDataError,
    build_correlation_source_data,
    build_endpoint_source_data,
    render_continuous_correlation_superplot,
    render_continuous_correlation_package,
    render_exact_footprint_superplots,
    register_correlation_in_superplot_manifest,
    select_primary_correlation_inference,
    select_primary_inference,
)


def _synthetic_tables() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    nuclei_rows: list[dict[str, object]] = []
    fov_rows: list[dict[str, object]] = []
    set_rows: list[dict[str, object]] = []
    inference_rows: list[dict[str, object]] = []
    endpoints = [
        spec.endpoint for family in DEFAULT_ENDPOINT_FAMILIES for spec in family.panels
    ]
    for arm_index, arm in enumerate(("NT", "KD")):
        for slide in (1, 2):
            replicate = slide
            biological_set = f"S{slide}_{arm}_{replicate}"
            image_key = f"{biological_set}_fov1.vsi".casefold()
            for nucleus_id in (1, 2):
                row: dict[str, object] = {
                    "sampled_in_analysis": True,
                    "include": True,
                    "is_control": False,
                    "secondary_only": False,
                    "arm": arm,
                    "slide": slide,
                    "replicate": replicate,
                    "biological_set": biological_set,
                    "image": image_key,
                    "image_key": image_key,
                    "fov": 1,
                    "nucleus_id": nucleus_id,
                }
                for endpoint_index, endpoint in enumerate(endpoints):
                    row[endpoint] = float(10 + endpoint_index + arm_index + nucleus_id)
                nuclei_rows.append(row)
            for endpoint_index, endpoint in enumerate(endpoints):
                fov_rows.append(
                    {
                        "cohort": "sampled_primary",
                        "endpoint": endpoint,
                        "image": image_key,
                        "image_key": image_key,
                        "slide": slide,
                        "arm": arm,
                        "replicate": replicate,
                        "fov": 1,
                        "biological_set": biological_set,
                        "is_control": False,
                        "n_nuclei_total": 2,
                        "n_nuclei_finite": 2,
                        "value": float(11.5 + endpoint_index + arm_index),
                    }
                )
                set_rows.append(
                    {
                        "cohort": "sampled_primary",
                        "endpoint": endpoint,
                        "slide": slide,
                        "arm": arm,
                        "replicate": replicate,
                        "biological_set": biological_set,
                        "n_fovs_total": 1,
                        "n_fovs_finite": 1,
                        "n_nuclei_total": 2,
                        "n_nuclei_finite": 2,
                        "value": float(11.5 + endpoint_index + arm_index),
                        "complete_for_inference": True,
                    }
                )
    for endpoint_index, endpoint in enumerate(endpoints):
        inference_rows.append(
            {
                "cohort": "sampled_primary",
                "endpoint": endpoint,
                "endpoint_scale": "unsigned_positive",
                "analysis_role": "primary",
                "excluded_biological_set": np.nan,
                "pairing_semantics": "none_independent_biological_sets",
                "inference_status": "complete",
                "n_complete_sets": 4,
                "n_nt": 2,
                "n_kd": 2,
                "mean_nt": 11.5 + endpoint_index,
                "mean_kd": 12.5 + endpoint_index,
                "difference_kd_minus_nt": 1.0,
                "difference_ci95_low": -0.5,
                "difference_ci95_high": 2.5,
                "ratio_kd_over_nt": 1.1,
                "percent_change_kd_vs_nt": 10.0,
                "welch_p_two_sided": 0.04,
                "slide_adjusted_p_two_sided": 0.05,
                "permutation_test": "exact_within_slide_label_permutation",
                "permutation_n_permutations": 16,
                "permutation_p_exact_two_sided": 0.03125,
            }
        )
    return (
        pd.DataFrame(nuclei_rows),
        pd.DataFrame(fov_rows),
        pd.DataFrame(set_rows),
        pd.DataFrame(inference_rows),
    )


def test_select_primary_inference_uses_unexcluded_sampled_row() -> None:
    *_, inference = _synthetic_tables()
    endpoint = DEFAULT_ENDPOINT_FAMILIES[0].panels[0].endpoint
    sensitivity = inference.loc[inference["endpoint"].eq(endpoint)].copy()
    sensitivity["analysis_role"] = "s1_kd1_whole_set_exclusion_sensitivity"
    sensitivity["excluded_biological_set"] = "S1_KD_1"
    selected = select_primary_inference(
        pd.concat([inference, sensitivity], ignore_index=True),
        [endpoint],
    )
    assert selected["endpoint"].tolist() == [endpoint]
    assert selected.iloc[0]["analysis_role"] == "primary"
    assert selected.iloc[0]["permutation_p_exact_two_sided"] == pytest.approx(0.03125)


def test_select_primary_inference_rejects_duplicate_primary_row() -> None:
    *_, inference = _synthetic_tables()
    endpoint = DEFAULT_ENDPOINT_FAMILIES[0].panels[0].endpoint
    duplicate = inference.loc[inference["endpoint"].eq(endpoint)]
    with pytest.raises(FigureDataError, match="exactly one primary inference row"):
        select_primary_inference(
            pd.concat([inference, duplicate], ignore_index=True),
            [endpoint],
        )


def test_build_endpoint_source_data_marks_only_sets_as_tested() -> None:
    nuclei, fovs, sets, _ = _synthetic_tables()
    endpoint = DEFAULT_ENDPOINT_FAMILIES[0].panels[0].endpoint
    source = build_endpoint_source_data(nuclei, fovs, sets, endpoint)
    assert source.groupby("tier", observed=True).size().to_dict() == {
        "biological_set_mean": 4,
        "fov_mean": 4,
        "nucleus": 8,
    }
    assert (
        source.loc[source["tested_inference_unit"], "tier"]
        .eq("biological_set_mean")
        .all()
    )
    assert not source.loc[
        source["tier"].ne("biological_set_mean"), "tested_inference_unit"
    ].any()
    assert source["cohort"].eq("sampled_primary").all()
    assert source["pairing_semantics"].eq("none_independent_biological_sets").all()


def test_build_endpoint_source_data_excludes_controls_and_unsampled_nuclei() -> None:
    nuclei, fovs, sets, _ = _synthetic_tables()
    endpoint = DEFAULT_ENDPOINT_FAMILIES[0].panels[0].endpoint
    control = nuclei.iloc[[0]].copy()
    control["is_control"] = True
    unsampled = nuclei.iloc[[1]].copy()
    unsampled["sampled_in_analysis"] = False
    source = build_endpoint_source_data(
        pd.concat([nuclei, control, unsampled], ignore_index=True),
        fovs,
        sets,
        endpoint,
    )
    assert len(source.loc[source["tier"].eq("nucleus")]) == 8


def test_build_correlation_source_data_preserves_three_level_hierarchy(
    tmp_path: Path,
) -> None:
    nuclei, _, _, _ = _synthetic_tables()
    nuclei["corr_all_detected_exact_valid_raw_pearson_r"] = 0.30
    nuclei["corr_all_detected_exact_valid_raw_spearman_rho"] = 0.25
    nuclei["corr_all_detected_exact_valid_raw_estimable"] = pd.Series(
        [True] * len(nuclei), dtype=object
    )
    nuclei["corr_all_detected_exact_valid_raw_n_spots"] = 20
    nuclei.loc[nuclei.index[0], "corr_all_detected_exact_valid_raw_estimable"] = np.nan
    fov_rows: list[dict[str, object]] = []
    set_rows: list[dict[str, object]] = []
    inference_rows: list[dict[str, object]] = []
    for method, value in (("pearson", 0.30), ("spearman", 0.25)):
        for arm in ("NT", "KD"):
            for slide in (1, 2):
                biological_set = f"S{slide}_{arm}_{slide}"
                image_key = f"{biological_set}_fov1.vsi".casefold()
                z_value = float(np.arctanh(value))
                fov_rows.append(
                    {
                        "cohort": "sampled_primary",
                        "image": image_key,
                        "image_key": image_key,
                        "slide": slide,
                        "arm": arm,
                        "replicate": slide,
                        "fov": 1,
                        "biological_set": biological_set,
                        "population": "all_detected_exact_valid",
                        "measurement_pair": "raw",
                        "correlation_method": method,
                        "conditional_descriptive": False,
                        "n_spots": 20,
                        "correlation": value,
                        "estimable": True,
                        "fisher_estimable": True,
                        "fisher_z": z_value,
                    }
                )
                set_rows.append(
                    {
                        "cohort": "sampled_primary",
                        "population": "all_detected_exact_valid",
                        "measurement_pair": "raw",
                        "correlation_method": method,
                        "conditional_descriptive": False,
                        "slide": slide,
                        "arm": arm,
                        "replicate": slide,
                        "biological_set": biological_set,
                        "n_fovs_total": 1,
                        "n_fovs_estimable": 1,
                        "n_spots_total": 20,
                        "fisher_z_mean": z_value,
                        "value": z_value,
                        "correlation_backtransformed": value,
                        "complete_for_inference": True,
                    }
                )
        inference_rows.append(
            {
                "cohort": "sampled_primary",
                "endpoint": f"all_detected_exact_valid__raw__{method}__fisher_z",
                "analysis_role": "primary_continuous_all_spots",
                "excluded_biological_set": np.nan,
                "inference_status": "complete",
                "n_nt": 2,
                "n_kd": 2,
                "mean_nt": float(np.arctanh(value)),
                "mean_kd": float(np.arctanh(value)),
                "difference_kd_minus_nt": 0.0,
                "welch_p_two_sided": 1.0,
                "slide_adjusted_p_two_sided": 1.0,
                "permutation_test": "exact_within_slide_label_permutation",
                "permutation_n_permutations": 16,
                "permutation_p_exact_two_sided": 1.0,
                "population": "all_detected_exact_valid",
                "measurement_pair": "raw",
                "correlation_method": method,
                "conditional_descriptive": False,
            }
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        source = build_correlation_source_data(
            nuclei, pd.DataFrame(fov_rows), pd.DataFrame(set_rows)
        )
    assert source.groupby(
        ["correlation_method", "tier"], observed=True
    ).size().to_dict() == {
        ("pearson", "biological_set_mean"): 4,
        ("pearson", "fov_mean"): 4,
        ("pearson", "nucleus"): 7,
        ("spearman", "biological_set_mean"): 4,
        ("spearman", "fov_mean"): 4,
        ("spearman", "nucleus"): 7,
    }
    selected = select_primary_correlation_inference(pd.DataFrame(inference_rows))
    assert selected["correlation_method"].tolist() == ["pearson", "spearman"]
    assert selected["star_from"].eq("permutation_p_exact_two_sided").all()
    rendered = render_continuous_correlation_superplot(
        source, selected, tmp_path, png_dpi=600
    )
    assert {path.suffix for path in rendered} == {".png", ".svg", ".pdf"}
    with Image.open(next(path for path in rendered if path.suffix == ".png")) as image:
        assert image.info["dpi"][0] == pytest.approx(600, rel=0.01)
    package = render_continuous_correlation_package(
        nuclei,
        pd.DataFrame(fov_rows),
        pd.DataFrame(set_rows),
        pd.DataFrame(inference_rows),
        tmp_path / "package",
        png_dpi=600,
    )
    assert package.source_data_path.is_file()
    assert package.statistics_path.is_file()
    assert package.manifest_path.is_file()
    assert package.png_path.is_file()
    parent_manifest = tmp_path / "parent_manifest.json"
    parent_manifest.write_text(
        json.dumps(
            {
                "families": [],
                "files": {
                    "png": [],
                    "svg": [],
                    "pdf": [],
                    "source_data": [],
                    "statistics": [],
                },
            }
        ),
        encoding="utf-8",
    )
    register_correlation_in_superplot_manifest(parent_manifest, package)
    linked = json.loads(parent_manifest.read_text(encoding="utf-8"))
    assert linked["families"][0]["stem"].startswith("06_continuous")
    assert linked["related_manifests"]["continuous_all_spot_correlation"] == str(
        package.manifest_path.resolve()
    )
    assert str(package.png_path.resolve()) in linked["files"]["png"]


def test_render_writes_vector_raster_sources_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    nuclei, fovs, sets, inference = _synthetic_tables()
    outputs = render_exact_footprint_superplots(
        nuclei,
        fovs,
        sets,
        inference,
        tmp_path / "figures",
        families=DEFAULT_ENDPOINT_FAMILIES[:1],
        png_dpi=600,
    )
    expected_stems = {
        DEFAULT_ENDPOINT_FAMILIES[0].stem,
        *(panel.stem for panel in DEFAULT_ENDPOINT_FAMILIES[0].panels),
    }
    assert {path.stem for path in outputs.png_paths} == expected_stems
    assert len(outputs.svg_paths) == len(expected_stems)
    assert len(outputs.pdf_paths) == len(expected_stems)
    assert outputs.master_source_data_path.is_file()
    assert outputs.master_statistics_path.is_file()
    assert outputs.manifest_path.is_file()
    with Image.open(outputs.png_paths[0]) as rendered:
        assert rendered.info["dpi"][0] == pytest.approx(600, rel=0.01)
        assert rendered.width > 1000
        assert rendered.height > 1000
    with pytest.raises(FileExistsError, match="already exists"):
        render_exact_footprint_superplots(
            nuclei,
            fovs,
            sets,
            inference,
            tmp_path / "figures",
            families=DEFAULT_ENDPOINT_FAMILIES[:1],
            png_dpi=600,
        )
