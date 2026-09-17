"""Clean final-publication renderer for retained MIAT--QKI endpoints.

This renderer is downstream-only. It preserves slide in audit tables and
primary inference while deliberately omitting slide from all visible encodings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fishsuite.core.exact_footprint_superplots import (
    ARM_COLORS,
    ARM_LABELS,
    FigureDataError,
    _parse_bool,
    _read_table,
    build_endpoint_source_data,
    select_primary_inference,
)


@dataclass(frozen=True)
class PublicationPanel:
    endpoint: str


@dataclass(frozen=True)
class PublicationFamily:
    stem: str
    title: str
    subtitle: str
    endpoints: tuple[str, str]
    labels: tuple[str, str]
    units: tuple[str, str]
    scales: tuple[float, float] = (1.0, 1.0)

    @property
    def panels(self) -> tuple[PublicationPanel, PublicationPanel]:
        """Expose endpoint panels for callers constructing source fixtures."""

        return tuple(PublicationPanel(endpoint) for endpoint in self.endpoints)  # type: ignore[return-value]


@dataclass(frozen=True)
class PublicationPlotOutputs:
    output_dir: Path
    png_paths: tuple[Path, ...]
    svg_paths: tuple[Path, ...]
    source_data_paths: tuple[Path, ...]
    statistics_paths: tuple[Path, ...]
    master_source_data_path: Path
    master_statistics_path: Path
    depletion_summary_path: Path
    ratio_of_ratios_statistics_path: Path
    manifest_path: Path


_Q95_DEFINITION = (
    "q95 is QKI greater than 95% of 1,000 same-nucleus KEEP-N randomized "
    "same-shape placements."
)

PUBLICATION_FAMILIES = (
    PublicationFamily(
        "01_total_miat_knockdown",
        "Total MIAT knockdown",
        "All detected MIAT puncta and summed fluorescence over detected spot pixels. "
        + _Q95_DEFINITION,
        ("n_spots_all", "miat_footprint_mass_all_union_deduplicated"),
        ("All MIAT spots per nucleus", "MIAT spot-pixel intensity"),
        ("spots per nucleus", "a.u. per nucleus"),
    ),
    PublicationFamily(
        "02_base_qki_association",
        "Base QKI association",
        "Primary and conservative denominators for spatial QKI association. "
        + _Q95_DEFINITION,
        (
            "association_fraction_among_usable_q95",
            "association_fraction_among_all_floor_spots_q95",
        ),
        (
            "QKI-associated fraction among usable MIAT spots",
            "QKI-associated fraction among all floor-passing MIAT spots",
        ),
        ("percent", "percent"),
        (100.0, 100.0),
    ),
    PublicationFamily(
        "03_qki_associated_miat_depletion",
        "QKI-associated MIAT depletion",
        "Associated detected puncta and associated MIAT fluorescence; overlapping pixels counted once. "
        + _Q95_DEFINITION,
        (
            "threshold_positive_spots_per_nucleus_q95",
            "miat_footprint_mass_q95_positive_union_deduplicated",
        ),
        (
            "QKI-associated MIAT spots per nucleus",
            "QKI-associated MIAT spot-pixel intensity",
        ),
        ("spots per nucleus", "a.u. per nucleus"),
    ),
)

_SUMMARY_MEASURES = (
    ("counts_global", "All MIAT spots", "n_spots_all"),
    ("counts_q95_associated", "QKI-associated MIAT spots", "threshold_positive_spots_per_nucleus_q95"),
    (
        "intensity_global",
        "All MIAT spot-pixel intensity",
        "miat_footprint_mass_all_union_deduplicated",
    ),
    (
        "intensity_q95_associated",
        "QKI-associated MIAT spot-pixel intensity",
        "miat_footprint_mass_q95_positive_union_deduplicated",
    ),
)


def _required_ratio_rows(ratios: pd.DataFrame) -> pd.DataFrame:
    required = {
        "cohort",
        "numerator_endpoint",
        "denominator_endpoint",
        "ratio_of_ratios",
        "ratio_of_ratios_ci95_low",
        "ratio_of_ratios_ci95_high",
        "p_two_sided",
    }
    missing = sorted(required.difference(ratios.columns))
    if missing:
        raise FigureDataError(
            f"ratio-of-ratios inference is missing required columns: {missing}"
        )
    pairs = (
        ("threshold_positive_spots_per_nucleus_q95", "n_spots_all"),
        (
            "miat_footprint_mass_q95_positive_union_deduplicated",
            "miat_footprint_mass_all_union_deduplicated",
        ),
    )
    selected: list[pd.DataFrame] = []
    for numerator, denominator in pairs:
        rows = ratios.loc[
            ratios["cohort"].astype(str).eq("sampled_primary")
            & ratios["numerator_endpoint"].astype(str).eq(numerator)
            & ratios["denominator_endpoint"].astype(str).eq(denominator)
        ].copy()
        if len(rows) != 1:
            raise FigureDataError(
                f"ratio-of-ratios row {numerator!r}/{denominator!r} must occur "
                f"exactly once; found {len(rows)}"
            )
        selected.append(rows)
    return pd.concat(selected, ignore_index=True)


def _draw_endpoint(axis: Any, source: pd.DataFrame, stat: pd.Series, label: str) -> None:
    for arm_index, arm in enumerate(("NT", "KD")):
        subset = source.loc[source["arm"].astype(str).eq(arm)]
        for tier, marker, size, alpha in (
            ("nucleus", "o", 9, 0.32),
            ("fov_mean", "D", 22, 0.65),
            ("biological_set_mean", "o", 38, 1.0),
        ):
            values = subset.loc[subset["tier"].eq(tier), "plot_value"].to_numpy(float)
            if not len(values):
                continue
            offsets = (
                np.linspace(-0.12, 0.12, len(values))
                if len(values) > 1
                else np.array([0.0])
            )
            axis.scatter(
                arm_index + offsets,
                values,
                marker=marker,
                s=size,
                facecolor=ARM_COLORS[arm]
                if tier == "biological_set_mean"
                else "0.62",
                edgecolor="black" if tier != "nucleus" else "none",
                linewidth=0.45,
                alpha=alpha,
                zorder={"nucleus": 1, "fov_mean": 2, "biological_set_mean": 3}[tier],
            )
        set_values = subset.loc[
            subset["tier"].eq("biological_set_mean"), "plot_value"
        ].to_numpy(float)
        axis.hlines(
            np.mean(set_values), arm_index - 0.19, arm_index + 0.19,
            color="black", linewidth=1.2, zorder=4,
        )
    axis.set_xticks((0, 1), (ARM_LABELS["NT"], ARM_LABELS["KD"]))
    axis.set_ylabel(label)
    axis.set_title(
        f"exact p = {float(stat['permutation_p_exact_two_sided']):.4g}",
        fontsize=7,
    )
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=7)


def _save(
    figure: Any, output_dir: Path, stem: str, png_dpi: int
) -> tuple[Path, Path]:
    png = output_dir / f"{stem}.png"
    svg = output_dir / f"{stem}.svg"
    with __import__("matplotlib").rc_context({"svg.fonttype": "none"}):
        figure.savefig(png, dpi=png_dpi, facecolor="white")
        figure.savefig(svg, format="svg", facecolor="white")
    return png, svg


def _render_family(
    source: pd.DataFrame,
    stats: pd.DataFrame,
    family: PublicationFamily,
    output_dir: Path,
    png_dpi: int,
) -> tuple[Path, Path]:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(7.2, 4.2))
    indexed = stats.set_index("endpoint")
    for axis, endpoint, label in zip(
        axes, family.endpoints, family.labels, strict=True
    ):
        _draw_endpoint(
            axis,
            source.loc[source["endpoint"].eq(endpoint)],
            indexed.loc[endpoint],
            label,
        )
    figure.suptitle(family.title, fontweight="bold", fontsize=11)
    figure.text(0.5, 0.885, family.subtitle, ha="center", va="top", fontsize=6.5)
    figure.text(
        0.5,
        0.035,
        "Points: nuclei, equal-weight FOV means, and equal-weight biological-set "
        "means. Primary inference: exact two-sided within-slide label permutation; "
        "no NT-to-KD pairing.",
        ha="center",
        fontsize=5.8,
        wrap=True,
    )
    figure.subplots_adjust(left=0.12, right=0.98, top=0.72, bottom=0.18, wspace=0.43)
    png, svg = _save(figure, output_dir, family.stem, png_dpi)
    plt.close(figure)
    return png, svg


def _depletion_summary(stats: pd.DataFrame) -> pd.DataFrame:
    indexed = stats.set_index("endpoint")
    rows: list[dict[str, object]] = []
    for measure, label, endpoint in _SUMMARY_MEASURES:
        row = indexed.loc[endpoint]
        nt = float(row["mean_nt"])
        kd = float(row["mean_kd"])
        if not np.isfinite(nt) or nt <= 0 or not np.isfinite(kd):
            raise FigureDataError(
                f"depletion summary needs finite positive NT mean for {endpoint!r}"
            )
        rows.append(
            {
                "measure": measure,
                "reader_label": label,
                "endpoint": endpoint,
                "mean_nt": nt,
                "mean_kd": kd,
                "kd_nt_percent_remaining": 100.0 * kd / nt,
                "primary_permutation_p_two_sided": row[
                    "permutation_p_exact_two_sided"
                ],
            }
        )
    return pd.DataFrame(rows)


def _render_summary(
    summary: pd.DataFrame,
    ratios: pd.DataFrame,
    output_dir: Path,
    png_dpi: int,
) -> tuple[Path, Path]:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(7.2, 4.2))
    groups = (
        ("counts_global", "counts_q95_associated"),
        ("intensity_global", "intensity_q95_associated"),
    )
    for axis, measures, title in zip(
        axes, groups, ("MIAT spots", "MIAT spot-pixel intensity"), strict=True
    ):
        frame = summary.set_index("measure").loc[list(measures)]
        axis.bar(
            (0, 1), frame["kd_nt_percent_remaining"],
            color=(ARM_COLORS["NT"], ARM_COLORS["KD"]),
        )
        axis.set_xticks((0, 1), ("All", "QKI-associated"))
        axis.set_ylim(0, max(110, float(frame["kd_nt_percent_remaining"].max()) * 1.2))
        axis.set_ylabel("KD/NT percent remaining")
        axis.set_title(title, fontsize=8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=7)
    figure.suptitle("Global versus QKI-associated depletion", fontweight="bold", fontsize=11)
    figure.text(
        0.5, 0.89, "KD/NT uses retained biological-set arm means. " + _Q95_DEFINITION,
        ha="center", va="top", fontsize=6.5,
    )
    ratio_labels = {
        (
            "threshold_positive_spots_per_nucleus_q95",
            "n_spots_all",
        ): "Count RoR",
        (
            "miat_footprint_mass_q95_positive_union_deduplicated",
            "miat_footprint_mass_all_union_deduplicated",
        ): "MIAT spot-pixel intensity RoR",
    }
    ratio_text = "; ".join(
        f"{ratio_labels[(str(row.numerator_endpoint), str(row.denominator_endpoint))]}: "
        f"ratio-of-ratios={float(row.ratio_of_ratios):.3g}, 95% CI "
        f"{float(row.ratio_of_ratios_ci95_low):.3g}–"
        f"{float(row.ratio_of_ratios_ci95_high):.3g}, p={float(row.p_two_sided):.3g}"
        for row in ratios.itertuples(index=False)
    )
    figure.text(
        0.5, 0.035, "Retained ratio-of-ratios inference: " + ratio_text,
        ha="center", fontsize=5.8, wrap=True,
    )
    figure.subplots_adjust(left=0.12, right=0.98, top=0.73, bottom=0.18, wspace=0.38)
    png, svg = _save(
        figure, output_dir, "04_global_vs_qki_associated_depletion", png_dpi
    )
    plt.close(figure)
    return png, svg


def render_publication_plot_package(
    nuclei: pd.DataFrame | str | Path,
    fovs: pd.DataFrame | str | Path,
    sets: pd.DataFrame | str | Path,
    primary_inference: pd.DataFrame | str | Path,
    ratio_of_ratios_inference: pd.DataFrame | str | Path,
    output_dir: str | Path,
    *,
    png_dpi: int = 600,
) -> PublicationPlotOutputs:
    """Render the four approved final-publication statistical families."""

    if int(png_dpi) != 600:
        raise FigureDataError("publication PNG output must be exactly 600 dpi")
    nucleus_table, nucleus_meta = _read_table(nuclei, table="nucleus endpoints")
    fov_table, fov_meta = _read_table(fovs, table="FOV endpoint means")
    set_table, set_meta = _read_table(sets, table="biological-set endpoint means")
    inference_table, inference_meta = _read_table(
        primary_inference, table="endpoint inference"
    )
    ratio_table, ratio_meta = _read_table(
        ratio_of_ratios_inference, table="ratio-of-ratios inference"
    )
    endpoints = [
        panel.endpoint for family in PUBLICATION_FAMILIES for panel in family.panels
    ]
    selected = select_primary_inference(inference_table, endpoints).copy()
    complete_for_inference = _parse_bool(
        set_table["complete_for_inference"],
        column="biological-set endpoint means complete_for_inference",
    )
    n_sets = (
        set_table.loc[
            set_table["endpoint"].astype(str).isin(endpoints)
            & set_table["cohort"].astype(str).eq("sampled_primary")
            & set_table["arm"].astype(str).isin(("NT", "KD"))
            & complete_for_inference
        ]
        .groupby(["endpoint", "arm"], observed=True)["biological_set"]
        .nunique()
    )
    expected_groups = pd.MultiIndex.from_product(
        [endpoints, ("NT", "KD")], names=("endpoint", "arm")
    )
    n_sets = n_sets.reindex(expected_groups, fill_value=0)
    if not bool(n_sets.eq(6).all()):
        raise FigureDataError(
            "publication renderer requires six complete-for-inference biological "
            "sets per arm for every endpoint"
        )
    for row in selected.itertuples(index=False):
        endpoint = str(row.endpoint)
        expected_nt = int(n_sets.loc[(endpoint, "NT")])
        expected_kd = int(n_sets.loc[(endpoint, "KD")])
        if float(row.n_nt) != expected_nt or float(row.n_kd) != expected_kd:
            raise FigureDataError(
                f"primary inference for {endpoint!r} must report n_nt=n_kd=6 "
                "matching complete displayed biological sets"
            )
    ratios = _required_ratio_rows(ratio_table)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    sources: list[pd.DataFrame] = []
    for family in PUBLICATION_FAMILIES:
        for panel, scale, unit in zip(
            family.panels, family.scales, family.units, strict=True
        ):
            source = build_endpoint_source_data(
                nucleus_table,
                fov_table,
                set_table,
                panel.endpoint,
                display_scale=scale,
                display_unit=unit,
            )
            source.insert(0, "figure_family", family.stem)
            sources.append(source)
    master_source = pd.concat(sources, ignore_index=True, sort=False)
    selected["primary_p_value_column"] = "permutation_p_exact_two_sided"
    selected["inference_unit"] = "biological_set_mean"
    selected["paired_across_arms"] = False
    master_source_path = output / "publication_plots.source_data.csv"
    master_statistics_path = output / "publication_plots.statistics.csv"
    depletion_summary_path = output / "global_vs_associated_depletion.source_data.csv"
    ratio_statistics_path = (
        output / "global_vs_associated_depletion.ratio_of_ratios.statistics.csv"
    )
    master_source.to_csv(master_source_path, index=False)
    selected.to_csv(master_statistics_path, index=False)
    summary = _depletion_summary(selected)
    summary.to_csv(depletion_summary_path, index=False)
    ratios.to_csv(ratio_statistics_path, index=False)
    png_paths: list[Path] = []
    svg_paths: list[Path] = []
    for family in PUBLICATION_FAMILIES:
        png, svg = _render_family(
            master_source.loc[master_source["figure_family"].eq(family.stem)],
            selected.loc[selected["endpoint"].isin(family.endpoints)],
            family,
            output,
            int(png_dpi),
        )
        png_paths.append(png)
        svg_paths.append(svg)
    png, svg = _render_summary(summary, ratios, output, int(png_dpi))
    png_paths.append(png)
    svg_paths.append(svg)
    manifest_path = output / "publication_plot_manifest.json"
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "renderer": "fishsuite.core.exact_footprint_publication_plots",
        "png_dpi": int(png_dpi),
        "slide_visual_encoding": False,
        "set_mean_marker": "o",
        "inputs": {
            "nucleus_endpoints": nucleus_meta,
            "fov_endpoint_means": fov_meta,
            "biological_set_endpoint_means": set_meta,
            "primary_inference": inference_meta,
            "ratio_of_ratios_inference": ratio_meta,
        },
        "scientific_contract": {
            "hierarchy": "nucleus -> equal-weight FOV mean -> equal-weight biological-set mean",
            "slide": "retained only as statistical blocking/audit field",
            "primary_inference": "exact_within_slide_label_permutation",
            "pairing": "none between NT and MIAT-KD",
        },
        "files": {
            "png": [str(path.resolve()) for path in png_paths],
            "svg": [str(path.resolve()) for path in svg_paths],
            "pdf": [],
            "source_data": [
                str(master_source_path.resolve()),
                str(depletion_summary_path.resolve()),
            ],
            "statistics": [
                str(master_statistics_path.resolve()),
                str(ratio_statistics_path.resolve()),
            ],
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return PublicationPlotOutputs(
        output,
        tuple(png_paths),
        tuple(svg_paths),
        (master_source_path, depletion_summary_path),
        (master_statistics_path, ratio_statistics_path),
        master_source_path,
        master_statistics_path,
        depletion_summary_path,
        ratio_statistics_path,
        manifest_path,
    )
