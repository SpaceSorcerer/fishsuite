"""Publication-grade hierarchical SuperPlots for exact-footprint endpoints.

The module is deliberately downstream-only: it reads finalized post-run CSVs,
draws nuclei and FOV means as descriptive nested tiers, and marks biological-
set means as the sole inferential tier.  NT and MIAT-KD replicate numbers are
never paired.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


ARM_ORDER = ("NT", "KD")
ARM_LABELS = {"NT": "NT", "KD": "MIAT-KD"}
ARM_COLORS = {"NT": "#0072B2", "KD": "#D55E00"}
STYLE_DIR = Path(
    r"F:\RNA-SEQ-ANALYSIS\MIAT-KD-RNAseq\committee_june_figures"
    r"\_REBUILD_v49_2026-06-12\_style"
)


class FigureDataError(ValueError):
    """Raised when finalized tables violate the figure data contract."""


@dataclass(frozen=True)
class PanelSpec:
    endpoint: str
    stem: str
    title: str
    ylabel: str
    display_unit: str
    scale: float = 1.0
    effect_kind: str = "additive"


@dataclass(frozen=True)
class FamilySpec:
    stem: str
    title: str
    subtitle: str
    filter_label: str
    panels: tuple[PanelSpec, PanelSpec]


@dataclass(frozen=True)
class SuperplotOutputs:
    output_dir: Path
    png_paths: tuple[Path, ...]
    svg_paths: tuple[Path, ...]
    pdf_paths: tuple[Path, ...]
    source_data_paths: tuple[Path, ...]
    statistics_paths: tuple[Path, ...]
    master_source_data_path: Path
    master_statistics_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class CorrelationOutputs:
    output_dir: Path
    png_path: Path
    svg_path: Path
    pdf_path: Path
    source_data_path: Path
    statistics_path: Path
    manifest_path: Path


_COMMON_FILTER = (
    "Filter: sampled_in_analysis=true + biological images only + one reviewed "
    "z/FOV + exact detected-MIAT footprints"
)
_Q95_FILTER = _COMMON_FILTER + "; q95 = own-nucleus KEEP-N spatial null"


DEFAULT_ENDPOINT_FAMILIES: tuple[FamilySpec, ...] = (
    FamilySpec(
        stem="01_miat_spot_counts",
        title="MIAT spot abundance by condition",
        subtitle=(
            "All detected MIAT spots and the subset with local QKI above the "
            "q95 own-nucleus spatial-null threshold"
        ),
        filter_label=_Q95_FILTER,
        panels=(
            PanelSpec(
                endpoint="n_spots_all",
                stem="01A_all_miat_spots_per_nucleus",
                title="All detected MIAT spots",
                ylabel="MIAT spots per nucleus",
                display_unit="spots per nucleus",
            ),
            PanelSpec(
                endpoint="threshold_positive_spots_per_nucleus_q95",
                stem="01B_q95_associated_miat_spots_per_nucleus",
                title="QKI-associated MIAT spots (q95)",
                ylabel="q95-associated MIAT spots per nucleus",
                display_unit="spots per nucleus",
            ),
        ),
    ),
    FamilySpec(
        stem="02_miat_exact_footprint_mass",
        title="MIAT signal within exact detected-spot footprints",
        subtitle=(
            "Union-deduplicated MIAT intensity for all footprints and the "
            "q95 QKI-associated subset"
        ),
        filter_label=_Q95_FILTER + "; overlapping footprint pixels counted once",
        panels=(
            PanelSpec(
                endpoint="miat_footprint_mass_all_union_deduplicated",
                stem="02A_global_miat_union_footprint_mass",
                title="All MIAT footprint signal",
                ylabel="Union-deduplicated MIAT signal (a.u.)",
                display_unit="a.u. per nucleus",
            ),
            PanelSpec(
                endpoint="miat_footprint_mass_q95_positive_union_deduplicated",
                stem="02B_q95_associated_miat_union_footprint_mass",
                title="QKI-associated MIAT footprint signal (q95)",
                ylabel="q95-associated MIAT signal (a.u.)",
                display_unit="a.u. per nucleus",
            ),
        ),
    ),
    FamilySpec(
        stem="03_q95_association_fractions",
        title="Fraction of MIAT spots classified as QKI-associated",
        subtitle=(
            "Primary usable-spot denominator and conservative all-floor-spot "
            "denominator shown side by side"
        ),
        filter_label=_Q95_FILTER
        + "; all-floor denominator treats unusable spots as not associated",
        panels=(
            PanelSpec(
                endpoint="association_fraction_among_usable_q95",
                stem="03A_q95_association_fraction_usable_denominator",
                title="Among spatial-null-usable MIAT spots",
                ylabel="QKI-associated MIAT spots (%)",
                display_unit="percent",
                scale=100.0,
                effect_kind="percentage_points",
            ),
            PanelSpec(
                endpoint="association_fraction_among_all_floor_spots_q95",
                stem="03B_q95_association_fraction_all_floor_denominator",
                title="Among all MIAT spots at the analysis floor",
                ylabel="QKI-associated MIAT spots (%)",
                display_unit="percent",
                scale=100.0,
                effect_kind="percentage_points",
            ),
        ),
    ),
    FamilySpec(
        stem="04_local_qki_exact_footprint_intensity",
        title="Local QKI intensity within exact MIAT footprints",
        subtitle=(
            "QKI sampled only inside detected MIAT pixels; all exact-valid spots "
            "and the q95-associated subset"
        ),
        filter_label=_Q95_FILTER + "; QKI region = exact MIAT footprint (no disk)",
        panels=(
            PanelSpec(
                endpoint="qki_footprint_mean_all",
                stem="04A_local_qki_mean_all_exact_miat_footprints",
                title="All exact-valid MIAT footprints",
                ylabel="Local QKI mean intensity (a.u.)",
                display_unit="a.u.",
            ),
            PanelSpec(
                endpoint="qki_footprint_mean_q95_positive",
                stem="04B_local_qki_mean_q95_associated_footprints",
                title="q95-associated MIAT footprints",
                ylabel="Local QKI mean intensity (a.u.)",
                display_unit="a.u.",
            ),
        ),
    ),
    FamilySpec(
        stem="05_local_qki_enrichment_vs_nucleus",
        title="Local QKI enrichment at exact MIAT footprints",
        subtitle=(
            "Within-nucleus QKI ratio at all exact-valid MIAT spots and the "
            "q95-associated subset"
        ),
        filter_label=_Q95_FILTER
        + "; enrichment = exact-footprint QKI mean / same-nucleus QKI mean",
        panels=(
            PanelSpec(
                endpoint="qki_enrichment_vs_nucleus_mean_all",
                stem="05A_qki_enrichment_all_exact_miat_footprints",
                title="All exact-valid MIAT footprints",
                ylabel="QKI enrichment vs same nucleus (ratio)",
                display_unit="ratio",
            ),
            PanelSpec(
                endpoint="qki_enrichment_vs_nucleus_mean_q95_positive",
                stem="05B_qki_enrichment_q95_associated_footprints",
                title="q95-associated MIAT footprints",
                ylabel="QKI enrichment vs same nucleus (ratio)",
                display_unit="ratio",
            ),
        ),
    ),
)


def _require_columns(
    frame: pd.DataFrame, columns: Iterable[str], *, table: str
) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise FigureDataError(f"{table} is missing required columns: {missing}")


def _parse_bool(series: pd.Series, *, column: str) -> pd.Series:
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }
    result = series.map(
        lambda value: (
            value
            if isinstance(value, (bool, np.bool_))
            else mapping.get(str(value).strip().casefold())
        )
    )
    if result.isna().any():
        bad = sorted(set(series.loc[result.isna()].astype(str)))
        raise FigureDataError(f"{column} contains non-boolean values: {bad}")
    return result.astype(bool)


def _read_table(
    value: pd.DataFrame | str | Path, *, table: str
) -> tuple[pd.DataFrame, dict[str, object]]:
    if isinstance(value, pd.DataFrame):
        if value.empty:
            raise FigureDataError(f"{table} is empty")
        return value.copy(), {"source": "in_memory_dataframe", "sha256": None}
    path = Path(value)
    if not path.is_file():
        raise FigureDataError(f"{table} is missing: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise FigureDataError(f"{table} is empty: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return frame, {"source": str(path.resolve()), "sha256": digest}


def _blank(series: pd.Series) -> pd.Series:
    return series.isna() | series.astype(str).str.strip().eq("")


def _missing_false(series: pd.Series) -> pd.Series:
    """Replace missing boolean-like values without pandas dtype downcasting."""

    return series.map(lambda value: False if pd.isna(value) else value)


def select_primary_inference(
    inference: pd.DataFrame, endpoints: Sequence[str]
) -> pd.DataFrame:
    """Select exactly one unexcluded sampled-primary row per endpoint."""

    _require_columns(
        inference,
        {
            "cohort",
            "endpoint",
            "analysis_role",
            "excluded_biological_set",
            "inference_status",
            "n_nt",
            "n_kd",
            "mean_nt",
            "mean_kd",
            "difference_kd_minus_nt",
            "percent_change_kd_vs_nt",
            "welch_p_two_sided",
            "slide_adjusted_p_two_sided",
            "permutation_test",
            "permutation_n_permutations",
            "permutation_p_exact_two_sided",
        },
        table="endpoint inference",
    )
    requested = list(dict.fromkeys(str(endpoint) for endpoint in endpoints))
    candidate = inference.loc[
        inference["cohort"].astype(str).eq("sampled_primary")
        & inference["analysis_role"].astype(str).eq("primary")
        & _blank(inference["excluded_biological_set"])
        & inference["endpoint"].astype(str).isin(requested)
    ].copy()
    rows: list[pd.DataFrame] = []
    for endpoint in requested:
        selected = candidate.loc[candidate["endpoint"].astype(str).eq(endpoint)]
        if len(selected) != 1:
            raise FigureDataError(
                f"endpoint {endpoint!r} must have exactly one primary inference row; "
                f"found {len(selected)}"
            )
        row = selected.iloc[0]
        if str(row["inference_status"]) != "complete":
            raise FigureDataError(f"endpoint {endpoint!r} inference is not complete")
        if str(row["permutation_test"]) != "exact_within_slide_label_permutation":
            raise FigureDataError(
                f"endpoint {endpoint!r} does not carry the exact within-slide permutation test"
            )
        if not np.isfinite(float(row["permutation_p_exact_two_sided"])):
            raise FigureDataError(
                f"endpoint {endpoint!r} has no finite exact permutation p-value"
            )
        if int(float(row["permutation_n_permutations"])) <= 0:
            raise FigureDataError(
                f"endpoint {endpoint!r} has no permutation assignments"
            )
        rows.append(selected)
    return pd.concat(rows, ignore_index=True)


def select_primary_correlation_inference(
    inference: pd.DataFrame,
) -> pd.DataFrame:
    """Select Pearson and Spearman all-spot Fisher-z inference rows."""

    _require_columns(
        inference,
        {
            "cohort",
            "endpoint",
            "analysis_role",
            "excluded_biological_set",
            "inference_status",
            "n_nt",
            "n_kd",
            "mean_nt",
            "mean_kd",
            "difference_kd_minus_nt",
            "welch_p_two_sided",
            "slide_adjusted_p_two_sided",
            "permutation_test",
            "permutation_n_permutations",
            "permutation_p_exact_two_sided",
            "population",
            "measurement_pair",
            "correlation_method",
            "conditional_descriptive",
        },
        table="correlation inference",
    )
    conditional = _parse_bool(
        _missing_false(inference["conditional_descriptive"]),
        column="correlation inference conditional_descriptive",
    )
    candidate = inference.loc[
        inference["cohort"].astype(str).eq("sampled_primary")
        & inference["analysis_role"].astype(str).eq("primary_continuous_all_spots")
        & _blank(inference["excluded_biological_set"])
        & inference["population"].astype(str).eq("all_detected_exact_valid")
        & inference["measurement_pair"].astype(str).eq("raw")
        & inference["correlation_method"].astype(str).isin(("pearson", "spearman"))
        & ~conditional
    ].copy()
    rows: list[pd.DataFrame] = []
    for method in ("pearson", "spearman"):
        selected = candidate.loc[candidate["correlation_method"].astype(str).eq(method)]
        if len(selected) != 1:
            raise FigureDataError(
                f"correlation method {method!r} must have exactly one primary "
                f"all-spot inference row; found {len(selected)}"
            )
        row = selected.iloc[0]
        if str(row["inference_status"]) != "complete":
            raise FigureDataError(
                f"correlation method {method!r} inference is not complete"
            )
        if str(row["permutation_test"]) != ("exact_within_slide_label_permutation"):
            raise FigureDataError(
                f"correlation method {method!r} lacks exact within-slide inference"
            )
        rows.append(selected)
    out = pd.concat(rows, ignore_index=True)
    out["star_from"] = "permutation_p_exact_two_sided"
    out["primary_p_value_column"] = "permutation_p_exact_two_sided"
    out["inference_unit"] = "biological_set_mean_fisher_z"
    out["paired_across_arms"] = False
    return out


def _standardize_nucleus_rows(nuclei: pd.DataFrame, endpoint: str) -> pd.DataFrame:
    _require_columns(
        nuclei,
        {
            "sampled_in_analysis",
            "include",
            "is_control",
            "secondary_only",
            "arm",
            "slide",
            "replicate",
            "biological_set",
            "image",
            "image_key",
            "fov",
            "nucleus_id",
            endpoint,
        },
        table="nucleus endpoints",
    )
    keep = (
        _parse_bool(nuclei["sampled_in_analysis"], column="sampled_in_analysis")
        & _parse_bool(nuclei["include"], column="include")
        & ~_parse_bool(nuclei["is_control"], column="is_control")
        & ~_parse_bool(nuclei["secondary_only"], column="secondary_only")
        & nuclei["arm"].astype(str).isin(ARM_ORDER)
    )
    out = nuclei.loc[
        keep,
        [
            "arm",
            "slide",
            "replicate",
            "biological_set",
            "image",
            "image_key",
            "fov",
            "nucleus_id",
            endpoint,
        ],
    ].copy()
    out = out.rename(columns={endpoint: "raw_value"})
    out["n_nuclei_total"] = 1
    out["n_nuclei_finite"] = (
        pd.to_numeric(out["raw_value"], errors="coerce").notna().astype(int)
    )
    out["n_fovs_total"] = np.nan
    out["n_fovs_finite"] = np.nan
    return out


def _standardize_fov_rows(fovs: pd.DataFrame, endpoint: str) -> pd.DataFrame:
    _require_columns(
        fovs,
        {
            "cohort",
            "endpoint",
            "image",
            "image_key",
            "slide",
            "arm",
            "replicate",
            "fov",
            "biological_set",
            "is_control",
            "n_nuclei_total",
            "n_nuclei_finite",
            "value",
        },
        table="FOV endpoint means",
    )
    keep = (
        fovs["cohort"].astype(str).eq("sampled_primary")
        & fovs["endpoint"].astype(str).eq(endpoint)
        & ~_parse_bool(fovs["is_control"], column="FOV is_control")
        & fovs["arm"].astype(str).isin(ARM_ORDER)
    )
    out = fovs.loc[
        keep,
        [
            "arm",
            "slide",
            "replicate",
            "biological_set",
            "image",
            "image_key",
            "fov",
            "n_nuclei_total",
            "n_nuclei_finite",
            "value",
        ],
    ].copy()
    out = out.rename(columns={"value": "raw_value"})
    out["nucleus_id"] = np.nan
    out["n_fovs_total"] = 1
    out["n_fovs_finite"] = (
        pd.to_numeric(out["raw_value"], errors="coerce").notna().astype(int)
    )
    return out


def _standardize_set_rows(sets: pd.DataFrame, endpoint: str) -> pd.DataFrame:
    _require_columns(
        sets,
        {
            "cohort",
            "endpoint",
            "slide",
            "arm",
            "replicate",
            "biological_set",
            "n_fovs_total",
            "n_fovs_finite",
            "n_nuclei_total",
            "n_nuclei_finite",
            "value",
            "complete_for_inference",
        },
        table="biological-set endpoint means",
    )
    keep = (
        sets["cohort"].astype(str).eq("sampled_primary")
        & sets["endpoint"].astype(str).eq(endpoint)
        & sets["arm"].astype(str).isin(ARM_ORDER)
        & _parse_bool(sets["complete_for_inference"], column="complete_for_inference")
    )
    out = sets.loc[
        keep,
        [
            "arm",
            "slide",
            "replicate",
            "biological_set",
            "n_fovs_total",
            "n_fovs_finite",
            "n_nuclei_total",
            "n_nuclei_finite",
            "value",
        ],
    ].copy()
    if out["biological_set"].astype(str).duplicated().any():
        raise FigureDataError(
            f"endpoint {endpoint!r} has duplicate biological-set means"
        )
    out = out.rename(columns={"value": "raw_value"})
    out["image"] = pd.NA
    out["image_key"] = pd.NA
    out["fov"] = np.nan
    out["nucleus_id"] = np.nan
    return out


def build_endpoint_source_data(
    nuclei: pd.DataFrame,
    fovs: pd.DataFrame,
    sets: pd.DataFrame,
    endpoint: str,
    *,
    display_scale: float = 1.0,
    display_unit: str = "raw units",
) -> pd.DataFrame:
    """Build the exact three-tier source table rendered for one endpoint."""

    tier_frames = (
        ("nucleus", False, _standardize_nucleus_rows(nuclei, endpoint)),
        ("fov_mean", False, _standardize_fov_rows(fovs, endpoint)),
        ("biological_set_mean", True, _standardize_set_rows(sets, endpoint)),
    )
    standardized: list[pd.DataFrame] = []
    for tier, tested, frame in tier_frames:
        frame = frame.copy()
        frame["endpoint"] = endpoint
        frame["tier"] = tier
        frame["tested_inference_unit"] = tested
        frame["raw_value"] = pd.to_numeric(frame["raw_value"], errors="coerce")
        frame["plot_value"] = frame["raw_value"] * float(display_scale)
        frame["finite_value"] = np.isfinite(frame["plot_value"])
        frame["display_unit"] = display_unit
        frame["cohort"] = "sampled_primary"
        frame["sampling_gate"] = "sampled_in_analysis=true"
        frame["pairing_semantics"] = "none_independent_biological_sets"
        frame["aggregation"] = {
            "nucleus": "descriptive sampled nucleus nested in FOV",
            "fov_mean": "equal-weight mean of sampled nuclei within FOV",
            "biological_set_mean": (
                "equal-weight mean of FOV means; tested inference unit"
            ),
        }[tier]
        standardized.append(frame)
    out = pd.concat(standardized, ignore_index=True, sort=False)
    tier_order = {"nucleus": 0, "fov_mean": 1, "biological_set_mean": 2}
    arm_order = {"NT": 0, "KD": 1}
    out["_tier_order"] = out["tier"].map(tier_order)
    out["_arm_order"] = out["arm"].map(arm_order)
    out = out.sort_values(
        [
            "_tier_order",
            "_arm_order",
            "slide",
            "replicate",
            "biological_set",
            "image_key",
            "nucleus_id",
        ],
        kind="stable",
        na_position="last",
    ).drop(columns=["_tier_order", "_arm_order"])
    return out.reset_index(drop=True)


def build_correlation_source_data(
    nuclei: pd.DataFrame,
    fov_correlations: pd.DataFrame,
    set_correlations: pd.DataFrame,
) -> pd.DataFrame:
    """Build nucleus/FOV/set source rows for all-spot Pearson and Spearman r."""

    nucleus_value_columns = {
        "pearson": "corr_all_detected_exact_valid_raw_pearson_r",
        "spearman": "corr_all_detected_exact_valid_raw_spearman_rho",
    }
    nucleus_estimable = "corr_all_detected_exact_valid_raw_estimable"
    nucleus_n_spots = "corr_all_detected_exact_valid_raw_n_spots"
    _require_columns(
        nuclei,
        {
            "sampled_in_analysis",
            "include",
            "is_control",
            "secondary_only",
            "arm",
            "slide",
            "replicate",
            "biological_set",
            "image",
            "image_key",
            "fov",
            "nucleus_id",
            nucleus_estimable,
            nucleus_n_spots,
            *nucleus_value_columns.values(),
        },
        table="nucleus correlation endpoints",
    )
    _require_columns(
        fov_correlations,
        {
            "cohort",
            "image",
            "image_key",
            "slide",
            "arm",
            "replicate",
            "fov",
            "biological_set",
            "population",
            "measurement_pair",
            "correlation_method",
            "conditional_descriptive",
            "n_spots",
            "correlation",
            "estimable",
            "fisher_estimable",
            "fisher_z",
        },
        table="FOV correlations",
    )
    _require_columns(
        set_correlations,
        {
            "cohort",
            "population",
            "measurement_pair",
            "correlation_method",
            "conditional_descriptive",
            "slide",
            "arm",
            "replicate",
            "biological_set",
            "n_fovs_total",
            "n_fovs_estimable",
            "n_spots_total",
            "value",
            "correlation_backtransformed",
            "complete_for_inference",
        },
        table="biological-set correlations",
    )
    nucleus_keep = (
        _parse_bool(nuclei["sampled_in_analysis"], column="sampled_in_analysis")
        & _parse_bool(nuclei["include"], column="include")
        & ~_parse_bool(nuclei["is_control"], column="is_control")
        & ~_parse_bool(nuclei["secondary_only"], column="secondary_only")
        & nuclei["arm"].astype(str).isin(ARM_ORDER)
        & _parse_bool(
            _missing_false(nuclei[nucleus_estimable]),
            column=nucleus_estimable,
        )
    )
    fov_conditional = _parse_bool(
        _missing_false(fov_correlations["conditional_descriptive"]),
        column="FOV conditional_descriptive",
    )
    fov_estimable = _parse_bool(
        _missing_false(fov_correlations["estimable"]), column="FOV estimable"
    ) & _parse_bool(
        _missing_false(fov_correlations["fisher_estimable"]),
        column="FOV fisher_estimable",
    )
    set_conditional = _parse_bool(
        _missing_false(set_correlations["conditional_descriptive"]),
        column="set conditional_descriptive",
    )
    set_complete = _parse_bool(
        _missing_false(set_correlations["complete_for_inference"]),
        column="set complete_for_inference",
    )
    frames: list[pd.DataFrame] = []
    for method, value_column in nucleus_value_columns.items():
        nucleus = nuclei.loc[
            nucleus_keep,
            [
                "arm",
                "slide",
                "replicate",
                "biological_set",
                "image",
                "image_key",
                "fov",
                "nucleus_id",
                nucleus_n_spots,
                value_column,
            ],
        ].copy()
        nucleus = nucleus.rename(
            columns={nucleus_n_spots: "n_spots", value_column: "raw_value"}
        )
        nucleus["inference_value_fisher_z"] = np.arctanh(
            pd.to_numeric(nucleus["raw_value"], errors="coerce").clip(
                -0.999999, 0.999999
            )
        )
        nucleus["n_fovs_total"] = np.nan
        nucleus["n_fovs_estimable"] = np.nan
        nucleus["tier"] = "nucleus"
        nucleus["tested_inference_unit"] = False

        fov_keep = (
            fov_correlations["cohort"].astype(str).eq("sampled_primary")
            & fov_correlations["population"].astype(str).eq("all_detected_exact_valid")
            & fov_correlations["measurement_pair"].astype(str).eq("raw")
            & fov_correlations["correlation_method"].astype(str).eq(method)
            & ~fov_conditional
            & fov_estimable
            & fov_correlations["arm"].astype(str).isin(ARM_ORDER)
        )
        fov = fov_correlations.loc[
            fov_keep,
            [
                "arm",
                "slide",
                "replicate",
                "biological_set",
                "image",
                "image_key",
                "fov",
                "n_spots",
                "correlation",
                "fisher_z",
            ],
        ].copy()
        fov = fov.rename(
            columns={
                "correlation": "raw_value",
                "fisher_z": "inference_value_fisher_z",
            }
        )
        fov["nucleus_id"] = np.nan
        fov["n_fovs_total"] = 1
        fov["n_fovs_estimable"] = 1
        fov["tier"] = "fov_mean"
        fov["tested_inference_unit"] = False

        set_keep = (
            set_correlations["cohort"].astype(str).eq("sampled_primary")
            & set_correlations["population"].astype(str).eq("all_detected_exact_valid")
            & set_correlations["measurement_pair"].astype(str).eq("raw")
            & set_correlations["correlation_method"].astype(str).eq(method)
            & ~set_conditional
            & set_complete
            & set_correlations["arm"].astype(str).isin(ARM_ORDER)
        )
        set_frame = set_correlations.loc[
            set_keep,
            [
                "arm",
                "slide",
                "replicate",
                "biological_set",
                "n_fovs_total",
                "n_fovs_estimable",
                "n_spots_total",
                "value",
                "correlation_backtransformed",
            ],
        ].copy()
        if set_frame["biological_set"].astype(str).duplicated().any():
            raise FigureDataError(
                f"correlation method {method!r} has duplicate biological-set means"
            )
        set_frame = set_frame.rename(
            columns={
                "n_spots_total": "n_spots",
                "value": "inference_value_fisher_z",
                "correlation_backtransformed": "raw_value",
            }
        )
        set_frame["image"] = pd.NA
        set_frame["image_key"] = pd.NA
        set_frame["fov"] = np.nan
        set_frame["nucleus_id"] = np.nan
        set_frame["tier"] = "biological_set_mean"
        set_frame["tested_inference_unit"] = True
        for frame in (nucleus, fov, set_frame):
            frame["correlation_method"] = method
            frame["population"] = "all_detected_exact_valid"
            frame["measurement_pair"] = "raw"
            frame["cohort"] = "sampled_primary"
            frame["plot_value"] = pd.to_numeric(frame["raw_value"], errors="coerce")
            frame["finite_value"] = np.isfinite(frame["plot_value"])
            frame["pairing_semantics"] = "none_independent_biological_sets"
            frame["aggregation"] = {
                "nucleus": "descriptive per-nucleus spot-level correlation",
                "fov_mean": "FOV spot-level correlation",
                "biological_set_mean": (
                    "equal-weight mean of FOV Fisher z; displayed back-transformed"
                ),
            }[str(frame["tier"].iloc[0])]
            frames.append(frame)
    out = pd.concat(frames, ignore_index=True, sort=False)
    method_order = {"pearson": 0, "spearman": 1}
    tier_order = {"nucleus": 0, "fov_mean": 1, "biological_set_mean": 2}
    arm_order = {"NT": 0, "KD": 1}
    out["_method_order"] = out["correlation_method"].map(method_order)
    out["_tier_order"] = out["tier"].map(tier_order)
    out["_arm_order"] = out["arm"].map(arm_order)
    out = out.sort_values(
        [
            "_method_order",
            "_tier_order",
            "_arm_order",
            "slide",
            "replicate",
            "biological_set",
            "image_key",
            "nucleus_id",
        ],
        kind="stable",
        na_position="last",
    ).drop(columns=["_method_order", "_tier_order", "_arm_order"])
    return out.reset_index(drop=True)


def _stable_jitter(keys: Iterable[object], half_width: float) -> np.ndarray:
    result: list[float] = []
    denominator = float(2**64 - 1)
    for key in keys:
        digest = hashlib.blake2b(str(key).encode("utf-8"), digest_size=8).digest()
        unit = int.from_bytes(digest, "big") / denominator
        result.append((2.0 * unit - 1.0) * half_width)
    return np.asarray(result, dtype=float)


def _stars_for_p(value: object) -> str:
    try:
        p = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(p):
        return "n/a"
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _fmt_p(value: object) -> str:
    p = float(value)
    if p < 1e-4:
        return f"{p:.2e}"
    if p < 0.001:
        return f"{p:.6f}".rstrip("0")
    if p < 0.01:
        return f"{p:.4f}".rstrip("0")
    return f"{p:.3f}".rstrip("0").rstrip(".")


def _fmt_effect(value: float, unit: str) -> str:
    magnitude = abs(float(value))
    sign = "−" if value < 0 else "+"
    if magnitude >= 100_000:
        rendered = f"{magnitude:,.0f}"
    elif magnitude >= 100:
        rendered = f"{magnitude:,.1f}"
    elif magnitude >= 10:
        rendered = f"{magnitude:.2f}"
    else:
        rendered = f"{magnitude:.3f}".rstrip("0").rstrip(".")
    return f"{sign}{rendered} {unit}"


def _load_style():
    if not (STYLE_DIR / "fig_style.py").is_file():
        raise FigureDataError(f"locked MIAT/QKI style module is missing: {STYLE_DIR}")
    import matplotlib

    matplotlib.use("Agg", force=True)
    if "tomllib" not in sys.modules:
        try:
            import tomllib  # type: ignore # noqa: F401
        except ModuleNotFoundError:
            import tomli

            sys.modules["tomllib"] = tomli
    style_dir = str(STYLE_DIR)
    if style_dir not in sys.path:
        sys.path.insert(0, style_dir)
    import fig_style as fs  # type: ignore

    fs.set_rna_style()
    return fs


def _panel_bounds(
    values: np.ndarray, *, fraction: bool
) -> tuple[float, float, float, float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        raise FigureDataError("panel has no finite values")
    low = min(0.0, float(finite.min()))
    high = float(finite.max())
    span = max(high - low, abs(high), 1.0)
    if fraction:
        low = 0.0
    top = high + 0.27 * span
    bracket = high + 0.12 * span
    star = high + 0.15 * span
    return low, top, bracket, star


def _draw_panel(
    ax: Any,
    source: pd.DataFrame,
    stat_row: pd.Series,
    spec: PanelSpec,
    panel_letter: str,
    fs: Any,
) -> None:
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import MaxNLocator

    finite_source = source.loc[source["finite_value"]].copy()
    values = pd.to_numeric(finite_source["plot_value"], errors="coerce").to_numpy(float)
    y_min, y_max, bracket_y, star_y = _panel_bounds(
        values, fraction=spec.effect_kind == "percentage_points"
    )
    centers = {"NT": 0.0, "KD": 1.0}
    for arm in ARM_ORDER:
        center = centers[arm]
        color = ARM_COLORS[arm]
        nuclei = finite_source.loc[
            finite_source["arm"].eq(arm) & finite_source["tier"].eq("nucleus")
        ].copy()
        fovs = finite_source.loc[
            finite_source["arm"].eq(arm) & finite_source["tier"].eq("fov_mean")
        ].copy()
        sets = finite_source.loc[
            finite_source["arm"].eq(arm)
            & finite_source["tier"].eq("biological_set_mean")
        ].sort_values(["slide", "replicate"], kind="stable")
        if sets.empty or fovs.empty or nuclei.empty:
            raise FigureDataError(
                f"endpoint {spec.endpoint!r} has an empty plotted tier for arm {arm}"
            )
        set_values = pd.to_numeric(sets["plot_value"], errors="coerce").to_numpy(float)
        q1, q3 = np.percentile(set_values, [25, 75])
        rgba = (*__import__("matplotlib").colors.to_rgb(color), 0.08)
        ax.add_patch(
            Rectangle(
                (center - 0.40, q1),
                0.80,
                max(float(q3 - q1), np.finfo(float).eps),
                facecolor=rgba,
                edgecolor="0.68",
                linewidth=0.45,
                zorder=1,
            )
        )
        ax.hlines(
            float(set_values.mean()),
            center - 0.40,
            center + 0.40,
            color="black",
            linewidth=1.2,
            zorder=2,
        )
        nucleus_keys = (
            nuclei["image_key"].astype(str) + ":" + nuclei["nucleus_id"].astype(str)
        )
        fs.scatter_pts(
            ax,
            center - 0.22 + _stable_jitter(nucleus_keys, 0.13),
            pd.to_numeric(nuclei["plot_value"]),
            color,
            s=8.0,
            alpha_fill=0.18,
            lw=0.22,
            zorder=5,
        )
        fs.scatter_pts(
            ax,
            center + 0.035 + _stable_jitter(fovs["image_key"], 0.07),
            pd.to_numeric(fovs["plot_value"]),
            color,
            marker="D",
            s=28.0,
            alpha_fill=0.66,
            lw=0.6,
            zorder=4,
        )
        set_x = center + 0.27 + np.linspace(-0.070, 0.070, len(sets))
        for x, (_, row) in zip(set_x, sets.iterrows(), strict=True):
            fs.scatter_pts(
                ax,
                [x],
                [float(row["plot_value"])],
                color,
                marker="o" if int(float(row["slide"])) == 1 else "s",
                s=78.0,
                alpha_fill=0.56,
                lw=0.95,
                zorder=6,
            )
    bracket_drop = max((y_max - y_min) * 0.018, np.finfo(float).eps)
    ax.plot(
        [0.0, 0.0, 1.0, 1.0],
        [bracket_y - bracket_drop, bracket_y, bracket_y, bracket_y - bracket_drop],
        color="black",
        linewidth=0.75,
        zorder=7,
    )
    ax.text(
        0.5,
        star_y,
        _stars_for_p(stat_row["permutation_p_exact_two_sided"]),
        ha="center",
        va="bottom",
        fontsize=7,
        fontweight="bold",
        color="black",
    )
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(-0.52, 1.52)
    ax.set_xticks([0, 1])
    labels: list[str] = []
    for arm in ARM_ORDER:
        arm_rows = finite_source.loc[finite_source["arm"].eq(arm)]
        counts = arm_rows["tier"].value_counts()
        labels.append(
            f"{ARM_LABELS[arm]}\n{int(counts.get('biological_set_mean', 0))} sets · "
            f"{int(counts.get('fov_mean', 0))} FOVs · {int(counts.get('nucleus', 0))} nuclei"
        )
    ax.set_xticklabels(labels)
    for label, arm in zip(ax.get_xticklabels(), ARM_ORDER, strict=True):
        label.set_color(ARM_COLORS[arm])
        label.set_fontweight("bold")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(axis="y", color="0.87", linewidth=0.45, zorder=0)
    ax.set_axisbelow(True)
    fs.black_chrome(ax, ylabel=spec.ylabel, labelsize=7, ticklabelsize=6)
    fs.plain_title(ax, spec.title, fontsize=8, fontweight="bold", pad=20)
    difference = float(stat_row["difference_kd_minus_nt"]) * spec.scale
    if spec.effect_kind == "percentage_points":
        effect = _fmt_effect(difference, "percentage points")
    else:
        effect = _fmt_effect(difference, spec.display_unit)
    percent = float(stat_row["percent_change_kd_vs_nt"])
    ax.text(
        0.5,
        1.026,
        f"KD − NT: {effect} ({percent:+.1f}%)",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=6,
        color="black",
    )
    ax.text(
        -0.12,
        1.13,
        panel_letter,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="top",
        color="black",
    )
    fs.no_box(None, ax)


def _legend_handles():
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    return [
        Patch(facecolor=ARM_COLORS["NT"], edgecolor="none", label="NT"),
        Patch(facecolor=ARM_COLORS["KD"], edgecolor="none", label="MIAT-KD"),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=3.0,
            markerfacecolor="0.72",
            markeredgecolor="0.45",
            markeredgewidth=0.3,
            label="nucleus",
        ),
        Line2D(
            [],
            [],
            marker="D",
            linestyle="none",
            markersize=4.3,
            markerfacecolor="0.72",
            markeredgecolor="black",
            markeredgewidth=0.55,
            label="FOV mean",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=6.4,
            markerfacecolor="0.72",
            markeredgecolor="black",
            markeredgewidth=0.8,
            label="set mean · slide 1",
        ),
        Line2D(
            [],
            [],
            marker="s",
            linestyle="none",
            markersize=6.0,
            markerfacecolor="0.72",
            markeredgecolor="black",
            markeredgewidth=0.8,
            label="set mean · slide 2",
        ),
        Patch(
            facecolor="0.93",
            edgecolor="0.65",
            linewidth=0.5,
            label="set-mean IQR",
        ),
        Line2D(
            [],
            [],
            color="black",
            linewidth=1.2,
            label="mean of set means",
        ),
    ]


def _stats_footer(stats: pd.DataFrame, panels: Sequence[PanelSpec]) -> str:
    indexed = stats.set_index("endpoint")
    details: list[str] = []
    for letter, panel in zip("AB", panels, strict=True):
        row = indexed.loc[panel.endpoint]
        details.append(
            f"{letter}: exact p={_fmt_p(row['permutation_p_exact_two_sided'])} "
            f"({int(float(row['permutation_n_permutations']))} assignments); "
            f"Welch p={_fmt_p(row['welch_p_two_sided'])}; "
            f"slide-adjusted p={_fmt_p(row['slide_adjusted_p_two_sided'])}."
        )
    return (
        "Inference: exact two-sided within-slide label permutation on independent "
        "equal-weight biological-set means (star; primary). "
        + " ".join(details)
        + " **** p<1e-4, *** p<0.001, ** p<0.01, * p<0.05, ns ≥0.05. "
        "Welch and slide-adjusted values are secondary. No NT↔KD pairing or connecting lines."
    )


def _save_figure(
    figure: Any, output_dir: Path, stem: str, *, png_dpi: int
) -> tuple[Path, Path, Path]:
    png = output_dir / f"{stem}.png"
    svg = output_dir / f"{stem}.svg"
    pdf = output_dir / f"{stem}.pdf"
    figure.savefig(png, dpi=png_dpi, facecolor="white")
    figure.savefig(svg, format="svg", facecolor="white")
    figure.savefig(pdf, format="pdf", facecolor="white")
    return png, svg, pdf


def _render_family(
    source: pd.DataFrame,
    stats: pd.DataFrame,
    family: FamilySpec,
    output_dir: Path,
    *,
    png_dpi: int,
    fs: Any,
) -> tuple[list[Path], list[Path], list[Path]]:
    import matplotlib.pyplot as plt

    pngs: list[Path] = []
    svgs: list[Path] = []
    pdfs: list[Path] = []
    stat_index = stats.set_index("endpoint")
    cm = fs.CM
    figure, axes = plt.subplots(1, 2, figsize=(18.0 * cm, 13.2 * cm))
    for letter, axis, panel in zip("AB", axes, family.panels, strict=True):
        _draw_panel(
            axis,
            source.loc[source["endpoint"].eq(panel.endpoint)],
            stat_index.loc[panel.endpoint],
            panel,
            letter,
            fs,
        )
    figure.suptitle(
        family.title,
        x=0.5,
        y=0.973,
        fontsize=10,
        fontweight="bold",
        color="black",
    )
    figure.text(
        0.5,
        0.925,
        family.subtitle,
        ha="center",
        va="center",
        fontsize=6.4,
        color="0.25",
    )
    figure.subplots_adjust(left=0.09, right=0.985, top=0.79, bottom=0.34, wspace=0.31)
    figure.legend(
        handles=_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.205),
        ncol=4,
        frameon=False,
        fontsize=5.4,
        handletextpad=0.5,
        columnspacing=1.1,
    )
    figure.text(
        0.5,
        0.14,
        family.filter_label,
        ha="center",
        va="center",
        fontsize=5.0,
        color="0.25",
    )
    footer = "\n".join(textwrap.wrap(_stats_footer(stats, family.panels), width=166))
    figure.text(
        0.035,
        0.035,
        footer,
        ha="left",
        va="bottom",
        fontsize=4.7,
        color="0.18",
        linespacing=1.25,
    )
    fs.no_box(figure, list(axes))
    png, svg, pdf = _save_figure(figure, output_dir, family.stem, png_dpi=png_dpi)
    plt.close(figure)
    pngs.append(png)
    svgs.append(svg)
    pdfs.append(pdf)

    for panel in family.panels:
        figure, axis = plt.subplots(1, 1, figsize=(9.0 * cm, 13.2 * cm))
        _draw_panel(
            axis,
            source.loc[source["endpoint"].eq(panel.endpoint)],
            stat_index.loc[panel.endpoint],
            panel,
            "",
            fs,
        )
        figure.suptitle(
            family.title,
            x=0.5,
            y=0.973,
            fontsize=9,
            fontweight="bold",
            color="black",
        )
        figure.subplots_adjust(left=0.18, right=0.97, top=0.77, bottom=0.37)
        figure.legend(
            handles=_legend_handles(),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.225),
            ncol=2,
            frameon=False,
            fontsize=5.2,
            handletextpad=0.45,
            columnspacing=0.9,
        )
        figure.text(
            0.5,
            0.145,
            "\n".join(textwrap.wrap(family.filter_label, width=96)),
            ha="center",
            va="center",
            fontsize=4.9,
            color="0.25",
            linespacing=1.15,
        )
        # A single-panel footer names one endpoint once rather than repeating A/B.
        row = stat_index.loc[panel.endpoint]
        footer = (
            "Inference: exact two-sided within-slide label permutation on independent "
            "equal-weight biological-set means (star; primary). "
            f"Exact p={_fmt_p(row['permutation_p_exact_two_sided'])} "
            f"({int(float(row['permutation_n_permutations']))} assignments); "
            f"Welch p={_fmt_p(row['welch_p_two_sided'])}; "
            f"slide-adjusted p={_fmt_p(row['slide_adjusted_p_two_sided'])}. "
            "**** p<1e-4, *** p<0.001, ** p<0.01, * p<0.05, ns ≥0.05. "
            "No NT↔KD pairing or connecting lines."
        )
        figure.text(
            0.04,
            0.035,
            "\n".join(textwrap.wrap(footer, width=94)),
            ha="left",
            va="bottom",
            fontsize=4.6,
            color="0.18",
            linespacing=1.2,
        )
        fs.no_box(figure, axis)
        png, svg, pdf = _save_figure(figure, output_dir, panel.stem, png_dpi=png_dpi)
        plt.close(figure)
        pngs.append(png)
        svgs.append(svg)
        pdfs.append(pdf)
    return pngs, svgs, pdfs


def _draw_correlation_panel(
    ax: Any,
    source: pd.DataFrame,
    stat_row: pd.Series,
    *,
    method: str,
    panel_letter: str,
    fs: Any,
) -> None:
    from matplotlib.patches import Rectangle

    finite = source.loc[
        source["correlation_method"].astype(str).eq(method) & source["finite_value"]
    ].copy()
    for arm, center in zip(ARM_ORDER, (0.0, 1.0), strict=True):
        color = ARM_COLORS[arm]
        nuclei = finite.loc[finite["arm"].eq(arm) & finite["tier"].eq("nucleus")]
        fovs = finite.loc[finite["arm"].eq(arm) & finite["tier"].eq("fov_mean")]
        sets = finite.loc[
            finite["arm"].eq(arm) & finite["tier"].eq("biological_set_mean")
        ].sort_values(["slide", "replicate"], kind="stable")
        if nuclei.empty or fovs.empty or sets.empty:
            raise FigureDataError(
                f"continuous {method} panel has an empty hierarchy tier for {arm}"
            )
        set_values = pd.to_numeric(sets["plot_value"], errors="coerce").to_numpy(float)
        q1, q3 = np.percentile(set_values, [25, 75])
        rgba = (*__import__("matplotlib").colors.to_rgb(color), 0.08)
        ax.add_patch(
            Rectangle(
                (center - 0.40, q1),
                0.80,
                max(float(q3 - q1), np.finfo(float).eps),
                facecolor=rgba,
                edgecolor="0.68",
                linewidth=0.45,
                zorder=1,
            )
        )
        ax.hlines(
            float(set_values.mean()),
            center - 0.40,
            center + 0.40,
            color="black",
            linewidth=1.2,
            zorder=2,
        )
        nucleus_keys = (
            nuclei["image_key"].astype(str) + ":" + nuclei["nucleus_id"].astype(str)
        )
        fs.scatter_pts(
            ax,
            center - 0.22 + _stable_jitter(nucleus_keys, 0.13),
            pd.to_numeric(nuclei["plot_value"]),
            color,
            s=8.0,
            alpha_fill=0.18,
            lw=0.22,
            zorder=5,
        )
        fs.scatter_pts(
            ax,
            center + 0.035 + _stable_jitter(fovs["image_key"], 0.07),
            pd.to_numeric(fovs["plot_value"]),
            color,
            marker="D",
            s=28.0,
            alpha_fill=0.66,
            lw=0.6,
            zorder=4,
        )
        set_x = center + 0.27 + np.linspace(-0.070, 0.070, len(sets))
        for x, (_, row) in zip(set_x, sets.iterrows(), strict=True):
            fs.scatter_pts(
                ax,
                [x],
                [float(row["plot_value"])],
                color,
                marker="o" if int(float(row["slide"])) == 1 else "s",
                s=78.0,
                alpha_fill=0.56,
                lw=0.95,
                zorder=6,
            )
    ax.axhline(0.0, color="0.55", linestyle="--", linewidth=0.65, zorder=0)
    ax.plot(
        [0.0, 0.0, 1.0, 1.0],
        [1.015, 1.05, 1.05, 1.015],
        color="black",
        linewidth=0.75,
        zorder=7,
    )
    ax.text(
        0.5,
        1.075,
        _stars_for_p(stat_row["permutation_p_exact_two_sided"]),
        ha="center",
        va="bottom",
        fontsize=7,
        fontweight="bold",
        color="black",
    )
    ax.set_ylim(-1.06, 1.18)
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_xlim(-0.52, 1.52)
    ax.set_xticks([0, 1])
    labels: list[str] = []
    for arm in ARM_ORDER:
        arm_rows = finite.loc[finite["arm"].eq(arm)]
        counts = arm_rows["tier"].value_counts()
        labels.append(
            f"{ARM_LABELS[arm]}\n{int(counts.get('biological_set_mean', 0))} sets · "
            f"{int(counts.get('fov_mean', 0))} FOVs · {int(counts.get('nucleus', 0))} nuclei"
        )
    ax.set_xticklabels(labels)
    for label, arm in zip(ax.get_xticklabels(), ARM_ORDER, strict=True):
        label.set_color(ARM_COLORS[arm])
        label.set_fontweight("bold")
    ax.grid(axis="y", color="0.87", linewidth=0.45, zorder=0)
    ax.set_axisbelow(True)
    ylabel = (
        "Pearson correlation (r)" if method == "pearson" else "Spearman correlation (ρ)"
    )
    fs.black_chrome(ax, ylabel=ylabel, labelsize=7, ticklabelsize=6)
    fs.plain_title(
        ax,
        "Pearson linear association"
        if method == "pearson"
        else "Spearman rank association",
        fontsize=8,
        fontweight="bold",
        pad=20,
    )
    fisher_difference = float(stat_row["difference_kd_minus_nt"])
    ax.text(
        0.5,
        1.026,
        f"KD − NT in Fisher z: {fisher_difference:+.3f}",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=6,
        color="black",
    )
    ax.text(
        -0.12,
        1.13,
        panel_letter,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="top",
        color="black",
    )
    fs.no_box(None, ax)


def render_continuous_correlation_superplot(
    source: pd.DataFrame,
    statistics: pd.DataFrame,
    output_dir: str | Path,
    *,
    png_dpi: int = 600,
    stem: str = "06_continuous_miat_qki_correlations_all_exact_valid_spots",
) -> tuple[Path, Path, Path]:
    """Render Pearson/Spearman all-spot correlation hierarchy as one composite."""

    _require_columns(
        source,
        {
            "correlation_method",
            "tier",
            "arm",
            "slide",
            "replicate",
            "biological_set",
            "image_key",
            "nucleus_id",
            "plot_value",
            "finite_value",
        },
        table="correlation figure source",
    )
    selected = select_primary_correlation_inference(statistics)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        target = output / f"{stem}.{suffix}"
        if target.exists():
            raise FileExistsError(f"correlation figure already exists: {target}")
    fs = _load_style()
    import matplotlib.pyplot as plt

    cm = fs.CM
    figure, axes = plt.subplots(1, 2, figsize=(18.0 * cm, 13.2 * cm))
    indexed = selected.set_index("correlation_method")
    for letter, axis, method in zip("AB", axes, ("pearson", "spearman"), strict=True):
        _draw_correlation_panel(
            axis,
            source,
            indexed.loc[method],
            method=method,
            panel_letter=letter,
            fs=fs,
        )
    figure.suptitle(
        "Continuous MIAT–QKI signal association across all exact-valid spots",
        x=0.5,
        y=0.973,
        fontsize=10,
        fontweight="bold",
        color="black",
    )
    figure.text(
        0.5,
        0.925,
        "Per-nucleus and per-FOV coefficients are descriptive; biological-set inference uses equal-weight FOV Fisher z means",
        ha="center",
        va="center",
        fontsize=6.2,
        color="0.25",
    )
    figure.subplots_adjust(left=0.09, right=0.985, top=0.79, bottom=0.34, wspace=0.31)
    figure.legend(
        handles=_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.205),
        ncol=4,
        frameon=False,
        fontsize=5.4,
        handletextpad=0.5,
        columnspacing=1.1,
    )
    figure.text(
        0.5,
        0.14,
        _COMMON_FILTER
        + "; all detected exact-valid MIAT spots; raw MIAT vs exact-footprint QKI",
        ha="center",
        va="center",
        fontsize=5.0,
        color="0.25",
    )
    details: list[str] = []
    for letter, method in zip("AB", ("pearson", "spearman"), strict=True):
        row = indexed.loc[method]
        details.append(
            f"{letter}: exact p={_fmt_p(row['permutation_p_exact_two_sided'])} "
            f"({int(float(row['permutation_n_permutations']))} assignments); "
            f"Welch p={_fmt_p(row['welch_p_two_sided'])}; "
            f"slide-adjusted p={_fmt_p(row['slide_adjusted_p_two_sided'])}."
        )
    footer = (
        "Inference: exact two-sided within-slide label permutation of equal-weight "
        "biological-set mean Fisher z (star; primary). Set markers are back-transformed "
        "to r/ρ for display. "
        + " ".join(details)
        + " **** p<1e-4, *** p<0.001, ** p<0.01, * p<0.05, ns ≥0.05. "
        "No NT↔KD pairing or connecting lines."
    )
    figure.text(
        0.035,
        0.035,
        "\n".join(textwrap.wrap(footer, width=166)),
        ha="left",
        va="bottom",
        fontsize=4.7,
        color="0.18",
        linespacing=1.25,
    )
    fs.no_box(figure, list(axes))
    paths = _save_figure(figure, output, stem, png_dpi=int(png_dpi))
    plt.close(figure)
    return paths


def render_continuous_correlation_package(
    nuclei: pd.DataFrame | str | Path,
    fov_correlations: pd.DataFrame | str | Path,
    set_correlations: pd.DataFrame | str | Path,
    correlation_inference: pd.DataFrame | str | Path,
    output_dir: str | Path,
    *,
    png_dpi: int = 600,
    stem: str = "06_continuous_miat_qki_correlations_all_exact_valid_spots",
) -> CorrelationOutputs:
    """Write the all-spot correlation source/stat tables, figure, and manifest."""

    nucleus_table, nucleus_meta = _read_table(
        nuclei, table="nucleus correlation endpoints"
    )
    fov_table, fov_meta = _read_table(fov_correlations, table="FOV correlations")
    set_table, set_meta = _read_table(
        set_correlations, table="biological-set correlations"
    )
    inference_table, inference_meta = _read_table(
        correlation_inference, table="correlation inference"
    )
    source = build_correlation_source_data(nucleus_table, fov_table, set_table)
    statistics = select_primary_correlation_inference(inference_table)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source_path = output / f"{stem}.source_data.csv"
    statistics_path = output / f"{stem}.statistics.csv"
    manifest_path = output / f"{stem}.manifest.json"
    for target in (source_path, statistics_path, manifest_path):
        if target.exists():
            raise FileExistsError(f"correlation package file already exists: {target}")
    source.to_csv(source_path, index=False)
    statistics.to_csv(statistics_path, index=False)
    png, svg, pdf = render_continuous_correlation_superplot(
        source,
        statistics,
        output,
        png_dpi=png_dpi,
        stem=stem,
    )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "renderer": "fishsuite.core.exact_footprint_superplots",
        "figure": "continuous_miat_qki_all_exact_valid_spots",
        "png_dpi": int(png_dpi),
        "inputs": {
            "nucleus_endpoints": nucleus_meta,
            "fov_correlations": fov_meta,
            "biological_set_correlations": set_meta,
            "correlation_inference": inference_meta,
        },
        "scientific_contract": {
            "population": "all_detected_exact_valid",
            "measurement_pair": "raw MIAT vs exact-footprint QKI",
            "display": "per-nucleus/FOV r or rho and back-transformed set mean",
            "inference": "exact within-slide permutation of equal-weight biological-set Fisher-z means",
            "pairing": "none between NT and MIAT-KD",
        },
        "files": {
            "png": str(png.resolve()),
            "svg": str(svg.resolve()),
            "pdf": str(pdf.resolve()),
            "source_data": str(source_path.resolve()),
            "statistics": str(statistics_path.resolve()),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return CorrelationOutputs(
        output_dir=output,
        png_path=png,
        svg_path=svg,
        pdf_path=pdf,
        source_data_path=source_path,
        statistics_path=statistics_path,
        manifest_path=manifest_path,
    )


def register_correlation_in_superplot_manifest(
    superplot_manifest_path: str | Path,
    correlation: CorrelationOutputs,
) -> None:
    """Make the separately provenanced correlation package top-level discoverable."""

    path = Path(superplot_manifest_path)
    if not path.is_file():
        raise FigureDataError(f"top-level superplot manifest is missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest.get("families"), list) or not isinstance(
        manifest.get("files"), dict
    ):
        raise FigureDataError(
            "top-level superplot manifest lacks families/files collections"
        )
    family = {
        "stem": correlation.png_path.stem,
        "title": "Continuous MIAT–QKI signal association across all exact-valid spots",
        "endpoints": [
            "all_detected_exact_valid__raw__pearson__fisher_z",
            "all_detected_exact_valid__raw__spearman__fisher_z",
        ],
        "dedicated_manifest": str(correlation.manifest_path.resolve()),
    }
    if not any(
        isinstance(existing, dict) and existing.get("stem") == family["stem"]
        for existing in manifest["families"]
    ):
        manifest["families"].append(family)
    additions = {
        "png": correlation.png_path,
        "svg": correlation.svg_path,
        "pdf": correlation.pdf_path,
        "source_data": correlation.source_data_path,
        "statistics": correlation.statistics_path,
    }
    for category, file_path in additions.items():
        listing = manifest["files"].setdefault(category, [])
        if not isinstance(listing, list):
            raise FigureDataError(f"top-level manifest files.{category} is not a list")
        resolved = str(file_path.resolve())
        if resolved not in listing:
            listing.append(resolved)
    related = manifest.setdefault("related_manifests", {})
    if not isinstance(related, dict):
        raise FigureDataError("top-level related_manifests is not an object")
    related["continuous_all_spot_correlation"] = str(
        correlation.manifest_path.resolve()
    )
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _write_figure_tables(
    output_dir: Path,
    stem: str,
    source: pd.DataFrame,
    stats: pd.DataFrame,
) -> tuple[Path, Path]:
    source_path = output_dir / f"{stem}.source_data.csv"
    stats_path = output_dir / f"{stem}.statistics.csv"
    source.to_csv(source_path, index=False)
    stats.to_csv(stats_path, index=False)
    return source_path, stats_path


def render_exact_footprint_superplots(
    nuclei: pd.DataFrame | str | Path,
    fovs: pd.DataFrame | str | Path,
    sets: pd.DataFrame | str | Path,
    inference: pd.DataFrame | str | Path,
    output_dir: str | Path,
    *,
    families: Sequence[FamilySpec] = DEFAULT_ENDPOINT_FAMILIES,
    png_dpi: int = 600,
) -> SuperplotOutputs:
    """Render endpoint composites, standalone panels, and exact source tables."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(
            f"figure output directory already exists; use a fresh path: {output}"
        )
    if int(png_dpi) < 300:
        raise FigureDataError("png_dpi must be at least 300")
    if not families:
        raise FigureDataError("at least one figure family is required")
    nucleus_table, nucleus_meta = _read_table(nuclei, table="nucleus endpoints")
    fov_table, fov_meta = _read_table(fovs, table="FOV endpoint means")
    set_table, set_meta = _read_table(sets, table="biological-set endpoint means")
    inference_table, inference_meta = _read_table(inference, table="endpoint inference")
    panels = [panel for family in families for panel in family.panels]
    endpoints = [panel.endpoint for panel in panels]
    if len(endpoints) != len(set(endpoints)):
        raise FigureDataError("figure panel endpoints must be unique")
    selected_stats = select_primary_inference(inference_table, endpoints)
    sources: list[pd.DataFrame] = []
    panel_by_endpoint = {panel.endpoint: panel for panel in panels}
    family_by_endpoint = {
        panel.endpoint: family for family in families for panel in family.panels
    }
    for endpoint in endpoints:
        panel = panel_by_endpoint[endpoint]
        family = family_by_endpoint[endpoint]
        source = build_endpoint_source_data(
            nucleus_table,
            fov_table,
            set_table,
            endpoint,
            display_scale=panel.scale,
            display_unit=panel.display_unit,
        )
        source.insert(0, "figure_family", family.stem)
        source.insert(1, "panel_stem", panel.stem)
        sources.append(source)
    master_source = pd.concat(sources, ignore_index=True, sort=False)
    selected_stats = selected_stats.copy()
    selected_stats.insert(
        0,
        "figure_family",
        selected_stats["endpoint"].map(
            {endpoint: family_by_endpoint[endpoint].stem for endpoint in endpoints}
        ),
    )
    selected_stats.insert(
        1,
        "panel_stem",
        selected_stats["endpoint"].map(
            {endpoint: panel_by_endpoint[endpoint].stem for endpoint in endpoints}
        ),
    )
    selected_stats["primary_p_value_column"] = "permutation_p_exact_two_sided"
    selected_stats["star_from"] = "permutation_p_exact_two_sided"
    selected_stats["inference_unit"] = "biological_set_mean"
    selected_stats["paired_across_arms"] = False
    selected_stats["figure_display_scale"] = selected_stats["endpoint"].map(
        {panel.endpoint: panel.scale for panel in panels}
    )
    selected_stats["figure_display_unit"] = selected_stats["endpoint"].map(
        {panel.endpoint: panel.display_unit for panel in panels}
    )
    output.mkdir(parents=True, exist_ok=False)
    master_source_path = output / "all_superplots.source_data.csv"
    master_statistics_path = output / "all_superplots.statistics.csv"
    master_source.to_csv(master_source_path, index=False)
    selected_stats.to_csv(master_statistics_path, index=False)
    fs = _load_style()
    png_paths: list[Path] = []
    svg_paths: list[Path] = []
    pdf_paths: list[Path] = []
    source_paths: list[Path] = [master_source_path]
    statistics_paths: list[Path] = [master_statistics_path]
    for family in families:
        family_endpoints = [panel.endpoint for panel in family.panels]
        family_source = master_source.loc[
            master_source["endpoint"].isin(family_endpoints)
        ].copy()
        family_stats = selected_stats.loc[
            selected_stats["endpoint"].isin(family_endpoints)
        ].copy()
        source_path, stats_path = _write_figure_tables(
            output, family.stem, family_source, family_stats
        )
        source_paths.append(source_path)
        statistics_paths.append(stats_path)
        for panel in family.panels:
            source_path, stats_path = _write_figure_tables(
                output,
                panel.stem,
                family_source.loc[family_source["endpoint"].eq(panel.endpoint)],
                family_stats.loc[family_stats["endpoint"].eq(panel.endpoint)],
            )
            source_paths.append(source_path)
            statistics_paths.append(stats_path)
        pngs, svgs, pdfs = _render_family(
            family_source,
            family_stats,
            family,
            output,
            png_dpi=int(png_dpi),
            fs=fs,
        )
        png_paths.extend(pngs)
        svg_paths.extend(svgs)
        pdf_paths.extend(pdfs)
    manifest_path = output / "superplot_manifest.json"
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "renderer": "fishsuite.core.exact_footprint_superplots",
        "png_dpi": int(png_dpi),
        "output_dir": str(output.resolve()),
        "inputs": {
            "nucleus_endpoints": nucleus_meta,
            "fov_endpoint_means": fov_meta,
            "biological_set_endpoint_means": set_meta,
            "endpoint_inference": inference_meta,
        },
        "scientific_contract": {
            "quantitation_plane": "one reviewed z per FOV; same z for all channels",
            "projection": "none",
            "qki_sampling_region": "exact detected MIAT footprint; no disk",
            "primary_threshold": "q95 own-nucleus KEEP-N spatial null",
            "hierarchy": "nucleus -> equal-weight FOV mean -> equal-weight biological-set mean",
            "inference_unit": "biological-set mean",
            "pairing": "none between NT and MIAT-KD",
            "primary_p_value": "exact two-sided within-slide label permutation",
        },
        "families": [
            {
                "stem": family.stem,
                "title": family.title,
                "endpoints": [panel.endpoint for panel in family.panels],
            }
            for family in families
        ],
        "files": {
            "png": [str(path.resolve()) for path in png_paths],
            "svg": [str(path.resolve()) for path in svg_paths],
            "pdf": [str(path.resolve()) for path in pdf_paths],
            "source_data": [str(path.resolve()) for path in source_paths],
            "statistics": [str(path.resolve()) for path in statistics_paths],
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return SuperplotOutputs(
        output_dir=output,
        png_paths=tuple(png_paths),
        svg_paths=tuple(svg_paths),
        pdf_paths=tuple(pdf_paths),
        source_data_paths=tuple(source_paths),
        statistics_paths=tuple(statistics_paths),
        master_source_data_path=master_source_path,
        master_statistics_path=master_statistics_path,
        manifest_path=manifest_path,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render exact-footprint hierarchical statistical SuperPlots."
    )
    parser.add_argument("--analysis-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--png-dpi", type=int, default=600)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.analysis_dir
    outputs = render_exact_footprint_superplots(
        root / "nucleus_exact_footprint_endpoints.csv",
        root / "fov_endpoint_means.csv",
        root / "biological_set_endpoint_means.csv",
        root / "endpoint_inference.csv",
        args.output_dir,
        png_dpi=args.png_dpi,
    )
    correlation_outputs = render_continuous_correlation_package(
        root / "nucleus_exact_footprint_endpoints.csv",
        root / "fov_correlations.csv",
        root / "biological_set_correlations.csv",
        root / "correlation_inference.csv",
        args.output_dir,
        png_dpi=args.png_dpi,
    )
    register_correlation_in_superplot_manifest(
        outputs.manifest_path, correlation_outputs
    )
    print(f"Wrote exact-footprint SuperPlots to {outputs.output_dir}")
    print(f"PNG files: {len(outputs.png_paths)}")
    print(f"SVG files: {len(outputs.svg_paths)}")
    print(f"PDF files: {len(outputs.pdf_paths)}")
    print(f"Manifest: {outputs.manifest_path}")
    print(f"Correlation figure: {correlation_outputs.png_path}")
    print(f"Correlation manifest: {correlation_outputs.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
