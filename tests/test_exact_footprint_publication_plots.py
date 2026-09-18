from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from fishsuite.core.exact_footprint_publication_plots import (
    FigureDataError,
    PUBLICATION_FAMILIES,
    render_publication_plot_package,
)


def _publication_tables() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """Hand-checked two-slide, six-independent-set-per-arm fixture."""

    endpoints = [
        panel.endpoint for family in PUBLICATION_FAMILIES for panel in family.panels
    ]
    nuclei_rows: list[dict[str, object]] = []
    fov_rows: list[dict[str, object]] = []
    set_rows: list[dict[str, object]] = []
    inference_rows: list[dict[str, object]] = []
    endpoint_values = {
        "n_spots_all": (100.0, 50.0),
        "miat_footprint_mass_all_union_deduplicated": (200.0, 100.0),
        "association_fraction_among_usable_q95": (0.40, 0.20),
        "association_fraction_among_all_floor_spots_q95": (0.30, 0.15),
        "threshold_positive_spots_per_nucleus_q95": (20.0, 5.0),
        "miat_footprint_mass_q95_positive_union_deduplicated": (40.0, 10.0),
    }
    for arm_index, arm in enumerate(("NT", "KD")):
        for replicate in range(1, 7):
            slide = 1 if replicate <= 3 else 2
            biological_set = f"{arm}_set_{replicate}"
            image_key = f"{biological_set}_fov_1.vsi"
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
                for endpoint, values in endpoint_values.items():
                    row[endpoint] = values[arm_index]
                nuclei_rows.append(row)
            for endpoint, values in endpoint_values.items():
                shared = {
                    "cohort": "sampled_primary",
                    "endpoint": endpoint,
                    "slide": slide,
                    "arm": arm,
                    "replicate": replicate,
                    "biological_set": biological_set,
                    "n_nuclei_total": 2,
                    "n_nuclei_finite": 2,
                    "value": values[arm_index],
                }
                fov_rows.append(
                    {
                        **shared,
                        "image": image_key,
                        "image_key": image_key,
                        "fov": 1,
                        "is_control": False,
                    }
                )
                set_rows.append(
                    {
                        **shared,
                        "n_fovs_total": 1,
                        "n_fovs_finite": 1,
                        "complete_for_inference": True,
                    }
                )
    for endpoint, values in endpoint_values.items():
        inference_rows.append(
            {
                "cohort": "sampled_primary",
                "endpoint": endpoint,
                "analysis_role": "primary",
                "excluded_biological_set": np.nan,
                "inference_status": "complete",
                "n_nt": 6,
                "n_kd": 6,
                "mean_nt": values[0],
                "mean_kd": values[1],
                "difference_kd_minus_nt": values[1] - values[0],
                "percent_change_kd_vs_nt": -50.0,
                "welch_p_two_sided": 0.03,
                "slide_adjusted_p_two_sided": 0.04,
                "permutation_test": "exact_within_slide_label_permutation",
                "permutation_n_permutations": 400,
                "permutation_p_exact_two_sided": 0.0125,
            }
        )
    ratio_rows = []
    for numerator, denominator, effect in (
        ("threshold_positive_spots_per_nucleus_q95", "n_spots_all", 0.50),
        (
            "miat_footprint_mass_q95_positive_union_deduplicated",
            "miat_footprint_mass_all_union_deduplicated",
            0.50,
        ),
    ):
        ratio_rows.append(
            {
                "cohort": "sampled_primary",
                "numerator_endpoint": numerator,
                "denominator_endpoint": denominator,
                "n_nt": 6,
                "n_kd": 6,
                "numerator_ratio_kd_over_nt": 0.25,
                "denominator_ratio_kd_over_nt": 0.50,
                "ratio_of_ratios": effect,
                "ratio_of_ratios_ci95_low": 0.30,
                "ratio_of_ratios_ci95_high": 0.80,
                "p_two_sided": 0.009,
                "ci_method": "multivariate_delta_within_arm_covariance",
                "pairing_semantics": "none_arm_mean_ratio_of_ratios",
            }
        )
    return (
        pd.DataFrame(nuclei_rows),
        pd.DataFrame(fov_rows),
        pd.DataFrame(set_rows),
        pd.DataFrame(inference_rows),
        pd.DataFrame(ratio_rows),
    )


def test_publication_package_uses_clean_formats_and_pooled_slide_encoding(
    tmp_path: Path,
) -> None:
    """Catches reintroducing PDF output or visible slide-specific set symbols."""

    outputs = render_publication_plot_package(
        *_publication_tables(), tmp_path / "publication", png_dpi=600
    )

    assert {path.suffix for path in (*outputs.png_paths, *outputs.svg_paths)} == {
        ".png",
        ".svg",
    }
    assert not list(outputs.output_dir.glob("*.pdf"))
    with Image.open(outputs.png_paths[0]) as image:
        assert image.info["dpi"][0] == pytest.approx(600, rel=0.01)
    manifest = json.loads(outputs.manifest_path.read_text(encoding="utf-8"))
    assert manifest["slide_visual_encoding"] is False
    assert manifest["set_mean_marker"] == "o"
    assert manifest["files"].get("pdf", []) == []
    svg_texts = {
        path.stem: path.read_text(encoding="utf-8") for path in outputs.svg_paths
    }
    for svg_text in svg_texts.values():
        for forbidden in (
            "Slide 1",
            "Slide 2",
            "exact-footprint mass",
            "MIAT-mass",
            "(q95)",
        ):
            assert forbidden not in svg_text
        assert "q95 is QKI greater than 95%" in svg_text
    assert any("MIAT spot-pixel intensity" in text for text in svg_texts.values())
    ror_text = svg_texts["04_global_vs_qki_associated_depletion"]
    assert "Count RoR" in ror_text
    assert "MIAT spot-pixel intensity RoR" in ror_text
    source = pd.read_csv(outputs.master_source_data_path)
    statistics = pd.read_csv(outputs.master_statistics_path)
    assert "slide" in source.columns
    assert statistics["permutation_test"].eq(
        "exact_within_slide_label_permutation"
    ).all()


def test_depletion_summary_uses_hand_checked_remaining_and_retained_inference(
    tmp_path: Path,
) -> None:
    """Catches swapped global/associated values or recomputed ratio inference."""

    outputs = render_publication_plot_package(
        *_publication_tables(), tmp_path / "publication", png_dpi=600
    )

    summary = pd.read_csv(outputs.depletion_summary_path).set_index("measure")
    assert summary.loc["counts_global", "kd_nt_percent_remaining"] == 50.0
    assert summary.loc["counts_q95_associated", "kd_nt_percent_remaining"] == 25.0
    assert summary.loc["intensity_global", "kd_nt_percent_remaining"] == 50.0
    assert summary.loc["intensity_q95_associated", "kd_nt_percent_remaining"] == 25.0
    ratios = pd.read_csv(outputs.ratio_of_ratios_statistics_path)
    assert ratios["ratio_of_ratios"].tolist() == [0.5, 0.5]
    assert ratios["ratio_of_ratios_ci95_low"].tolist() == [0.3, 0.3]
    assert ratios["ratio_of_ratios_ci95_high"].tolist() == [0.8, 0.8]
    assert ratios["p_two_sided"].tolist() == [0.009, 0.009]


def test_publication_package_rejects_non_600_dpi(tmp_path: Path) -> None:
    """Catches silently producing a non-final-resolution PNG package."""

    with pytest.raises(FigureDataError, match="exactly 600 dpi"):
        render_publication_plot_package(
            *_publication_tables(), tmp_path / "publication", png_dpi=72
        )


@pytest.mark.parametrize("failure", ("incomplete", "missing_arm"))
def test_publication_package_requires_six_complete_sets_for_each_endpoint_arm(
    tmp_path: Path, failure: str
) -> None:
    """Catches incomplete sets counting as displayed sets or an absent arm passing."""

    nuclei, fovs, sets, inference, ratios = _publication_tables()
    endpoint = "n_spots_all"
    if failure == "incomplete":
        sets.loc[
            sets["endpoint"].eq(endpoint)
            & sets["arm"].eq("KD")
            & sets["biological_set"].eq("KD_set_6"),
            "complete_for_inference",
        ] = False
    else:
        sets = sets.loc[
            ~(sets["endpoint"].eq(endpoint) & sets["arm"].eq("KD"))
        ].copy()
    with pytest.raises(FigureDataError, match="six complete-for-inference"):
        render_publication_plot_package(
            nuclei, fovs, sets, inference, ratios, tmp_path / failure, png_dpi=600
        )


def test_publication_package_requires_primary_inference_counts_matching_sets(
    tmp_path: Path,
) -> None:
    """Catches a retained primary row whose reported arm count differs from display."""

    nuclei, fovs, sets, inference, ratios = _publication_tables()
    inference.loc[inference["endpoint"].eq("n_spots_all"), "n_kd"] = 5
    with pytest.raises(FigureDataError, match="n_nt=n_kd=6"):
        render_publication_plot_package(
            nuclei,
            fovs,
            sets,
            inference,
            ratios,
            tmp_path / "count_mismatch",
            png_dpi=600,
        )
