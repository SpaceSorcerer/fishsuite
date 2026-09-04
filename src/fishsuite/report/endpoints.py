"""Endpoint registry for the ``fishsuite report`` layer.

Endpoints are declared once, in ROLE terms (rna1 / rna2 / protein), and resolved
against whatever columns a given run actually emitted. An endpoint whose column
the run did not write is kept, marked absent, and reported as NA with a note; it
is never silently dropped.

Family names are plain language and map one-to-one onto the workbook's by-group
sheets. There are no ``Q1``-style codes anywhere in this layer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

# family key -> workbook sheet name. Plain language, no codes.
FAMILY_SHEET: Dict[str, str] = {
    "detection": "Spots per nucleus by group",
    "localization": "Nuclear fraction by group",
    "partner": "Partner at puncta by group",
}
FAMILY_ORDER: Tuple[str, ...] = ("detection", "localization", "partner")

FAMILY_DESCRIPTION: Dict[str, str] = {
    "detection": (
        "How much signal each nucleus carries: puncta counted per nucleus, punctum "
        "size, and absolute intensity. Absolute-intensity rows are descriptive only "
        "because laser power is retuned per section, so a level is not comparable "
        "across sections."),
    "localization": (
        "Where the signal sits rather than how much of it there is: the nuclear "
        "fraction of each nucleus's puncta, area-normalised density, and the "
        "nuclear-to-cytoplasmic intensity ratio. These are within-nucleus ratios, so "
        "a shifted detection floor moves numerator and denominator together."),
    "partner": (
        "Whether the partner channel is enriched at the puncta of the anchor channel, "
        "against the engine's own per-nucleus nulls, plus punctum-to-punctum pairing "
        "and the reciprocal direction."),
}

NUCLEUS = "nuclei_metrics.csv"
IMAGE = "per_image_summary.csv (image level, pooled over nuclei by the engine)"
SPOT = "spot_metrics.csv, aggregated per nucleus over that nucleus's nuclear rna1 spots"
DERIVED_AREA = "nuclei_metrics.csv, nucleus_area_px times the run voxel area"
DERIVED_DENSITY = "nuclei_metrics.csv, n_spots_rna1 divided by nucleus area"


@dataclass(frozen=True)
class Endpoint:
    """One measurable quantity, declared independently of any particular run."""

    name: str                        # stable machine name
    column: str                      # column in the per-nucleus or per-image frame
    family: str                      # key into FAMILY_SHEET
    unit: str
    label: str                       # plain-language name, {rna1}/{rna2}/{protein}
    level: str = "nucleus"           # 'nucleus' or 'image'
    source: str = NUCLEUS
    primary: bool = False
    descriptive_only: bool = False   # reported, never enters a Holm family
    exploratory: bool = False
    absolute_intensity: bool = False
    excluded_from_holm: str = ""
    usability_flag: str = ""         # per-nucleus boolean that must be True
    note: str = ""

    def pretty(self, labels: Dict[str, str]) -> str:
        out = self.label
        for role, text in labels.items():
            out = out.replace("{" + role + "}", str(text))
        return out


ENDPOINTS: Tuple[Endpoint, ...] = (
    # ------------------------------------------------------------- detection
    Endpoint("rna1_spots_per_nucleus", "n_spots_rna1", "detection",
             "puncta per nucleus", "{rna1} puncta per nucleus", primary=True,
             note="Total puncta detected in that nucleus, nuclear plus cytoplasmic."),
    Endpoint("rna1_nuclear_spots_per_nucleus", "nuclear_spot_count", "detection",
             "nuclear puncta per nucleus", "{rna1} nuclear puncta per nucleus"),
    Endpoint("rna1_spot_fwhm_px", "rna1_spot_fwhm_px", "detection",
             "pixels", "{rna1} punctum width (full width at half maximum)",
             source=SPOT,
             note="Per-nucleus mean over that nucleus's nuclear puncta."),
    Endpoint("rna1_spot_diameter_um", "rna1_spot_diameter_um", "detection",
             "micrometres", "{rna1} punctum diameter", source=SPOT,
             note="Per-nucleus mean over that nucleus's nuclear puncta."),
    Endpoint("rna1_nuclear_above_floor_intensity", "nuclear_above_floor_intensity_rna1",
             "detection", "arbitrary units",
             "{rna1} nuclear intensity above the detection floor",
             absolute_intensity=True,
             note="Run-floor pixel integral in arbitrary units. DESCRIPTIVE ONLY: an "
                  "absolute intensity is not comparable as a level claim across "
                  "sections, so it carries no multiplicity adjustment."),
    Endpoint("protein_nuclear_mean", "protein_nuclear_mean", "detection",
             "arbitrary units", "{protein} mean nuclear intensity",
             descriptive_only=True, absolute_intensity=True,
             note="DESCRIPTIVE ONLY. Report the within-nucleus nuclear-to-cytoplasmic "
                  "ratio instead of this level."),
    Endpoint("protein_spots_per_nucleus", "n_spots_protein", "detection",
             "puncta per nucleus", "{protein} puncta per nucleus", exploratory=True,
             excluded_from_holm="proxy for absolute partner level",
             note="EXPLORATORY. The count is thresholded at one global intensity, so "
                  "for a diffuse partner it tracks absolute level rather than object "
                  "number and carries no multiplicity adjustment."),
    Endpoint("rna2_spots_per_nucleus", "n_spots_rna2", "detection",
             "puncta per nucleus", "{rna2} puncta per nucleus"),

    # ---------------------------------------------------------- localization
    Endpoint("rna1_nuclear_spot_fraction", "nuclear_spot_fraction", "localization",
             "fraction of that nucleus's puncta that are nuclear",
             "{rna1} nuclear fraction", primary=True,
             note="Floor-robust headline: a ratio within one nucleus, so a shifted "
                  "detection floor moves numerator and denominator together."),
    Endpoint("rna1_spots_per_um2", "rna1_spots_per_um2", "localization",
             "puncta per square micrometre of nucleus",
             "{rna1} puncta per square micrometre", source=DERIVED_DENSITY,
             note="SENSITIVITY for the per-nucleus count, normalising for any "
                  "difference in nuclear area between groups."),
    Endpoint("nucleus_area_um2", "nucleus_area_um2", "localization",
             "square micrometres", "Nucleus area", source=DERIVED_AREA,
             descriptive_only=True,
             note="DESIGN DESCRIPTOR, not a result. Reported because a segmentation "
                  "area floor can truncate two groups unequally."),
    Endpoint("protein_nc_ratio", "protein_nc_ratio", "localization",
             "nuclear to cytoplasmic mean intensity ratio",
             "{protein} nuclear-to-cytoplasmic ratio",
             note="A within-nucleus ratio, so it survives the level-claim objection "
                  "that rules out the absolute nuclear mean."),
    Endpoint("rna1_nc_ratio", "rna_nc_ratio", "localization",
             "nuclear to cytoplasmic mean intensity ratio",
             "{rna1} nuclear-to-cytoplasmic ratio"),

    # --------------------------------------------------------------- partner
    Endpoint("partner_rotation_enrichment_at_rna1",
             "protein_rotation_enrichment_at_rna1_spots", "partner",
             "observed divided by the rotation-null mean",
             "{protein} enrichment at {rna1} puncta, rotation null", primary=True,
             usability_flag="rotation_null_usable",
             note="HEADLINE. The rotation null preserves each nucleus's own punctum "
                  "geometry, so it is the stricter of the two nulls. Restricted to "
                  "nuclei the engine flagged usable, which is the filter the engine "
                  "applies in its own aggregation path. Enrichment above 1 in BOTH "
                  "groups is the signature of co-distribution with a nuclear "
                  "sub-compartment, not evidence of a specific association."),
    Endpoint("partner_rotation_enrichment_at_rna1_allnuclei",
             "protein_rotation_enrichment_at_rna1_spots", "partner",
             "observed divided by the rotation-null mean",
             "{protein} enrichment at {rna1} puncta, rotation null, every nucleus",
             descriptive_only=True,
             note="SENSITIVITY: the same endpoint over every nucleus, including those "
                  "whose rotation null the engine flagged unusable, so the effect of "
                  "the usability filter is visible rather than hidden."),
    Endpoint("partner_rotation_null_z_at_rna1", "protein_rotation_null_z_at_rna1_spots",
             "partner", "z against the per-nucleus rotation null",
             "{protein} z at {rna1} puncta, rotation null",
             usability_flag="rotation_null_usable"),
    Endpoint("partner_rotation_null_p_at_rna1", "protein_rotation_null_p_at_rna1_spots",
             "partner", "empirical p against the per-nucleus rotation null",
             "{protein} per-nucleus empirical p at {rna1} puncta",
             descriptive_only=True, usability_flag="rotation_null_usable",
             note="A per-nucleus empirical p. Averaging p-values is not inference; "
                  "shown so the per-nucleus null result is visible, never adjusted."),
    Endpoint("partner_random_null_enrichment_at_rna1",
             "protein_enrichment_vs_null_at_rna1_spots", "partner",
             "observed divided by the random-position-null mean",
             "{protein} enrichment at {rna1} puncta, random-position null",
             note="The weaker of the two nulls; kept for continuity."),
    Endpoint("partner_random_null_z_at_rna1", "protein_null_z_at_rna1_spots", "partner",
             "z against the per-nucleus random-position null",
             "{protein} z at {rna1} puncta, random-position null"),
    Endpoint("fraction_rna1_puncta_above_rotation_null_p95",
             "protein_rotation_assoc_fraction_at_rna1_spots", "partner",
             "fraction of that nucleus's puncta",
             "Fraction of {rna1} puncta above the rotation-null 95th percentile",
             usability_flag="rotation_null_usable"),
    Endpoint("partner_mean_in_exact_rna1_footprint", "partner_mean_in_exact_rna1_footprint",
             "partner", "arbitrary units",
             "{protein} mean inside the exact {rna1} punctum footprint",
             source=SPOT, absolute_intensity=True,
             note="Mean partner intensity inside the exact connected half-maximum "
                  "punctum footprint, never a fixed disk. DESCRIPTIVE ONLY: an "
                  "absolute intensity in arbitrary units carries no adjustment."),
    Endpoint("partner_enrichment_in_exact_rna1_footprint",
             "partner_enrichment_in_exact_rna1_footprint", "partner",
             "footprint mean divided by local background",
             "{protein} enrichment inside the exact {rna1} punctum footprint",
             source=SPOT),
    Endpoint("fraction_rna1_puncta_partner_positive_exact_footprint",
             "fraction_rna1_puncta_partner_positive_exact_footprint", "partner",
             "fraction of that nucleus's nuclear puncta",
             "Fraction of {rna1} puncta with {protein} above threshold in the exact "
             "footprint", source=SPOT,
             note="Positive means the footprint mean is at or above the run's single "
                  "constant partner threshold. An absolute gate reads a group-level "
                  "intensity difference, so read the enrichment ratio rather than this "
                  "fraction as the association measure."),
    Endpoint("partner_radial_enrichment_at_0p25um", "protein_radial_enrichment_at_0p25um",
             "partner", "observed divided by null at 0.25 micrometres",
             "{protein} radial enrichment at 0.25 micrometres"),
    Endpoint("partner_radial_enrichment_at_0p5um", "protein_radial_enrichment_at_0p5um",
             "partner", "observed divided by null at 0.5 micrometres",
             "{protein} radial enrichment at 0.5 micrometres"),
    Endpoint("partner_radial_enrichment_at_0p75um", "protein_radial_enrichment_at_0p75um",
             "partner", "observed divided by null at 0.75 micrometres",
             "{protein} radial enrichment at 0.75 micrometres"),
    Endpoint("partner_radial_enrichment_at_1um", "protein_radial_enrichment_at_1um",
             "partner", "observed divided by null at 1.0 micrometres",
             "{protein} radial enrichment at 1.0 micrometres"),
    Endpoint("partner_pooled_rotation_enrichment_at_rna1",
             "protein_pooled_rotation_enrichment_at_rna1_spots", "partner",
             "observed divided by the rotation-null mean, pooled over the image",
             "{protein} enrichment at {rna1} puncta, pooled per image",
             level="image", source=IMAGE,
             note="The engine's punctum-weighted pooled per-image value. The "
                  "per-nucleus endpoint weights every nucleus equally instead, so the "
                  "two differ by construction and both are shown."),
    Endpoint("partner_pooled_random_null_enrichment_at_rna1",
             "protein_pooled_enrichment_vs_null_at_rna1_spots", "partner",
             "observed divided by the random-position-null mean, pooled over the image",
             "{protein} random-null enrichment at {rna1} puncta, pooled per image",
             level="image", source=IMAGE),
    Endpoint("fraction_rna1_puncta_above_rotation_null_p95_pooled",
             "protein_mean_rotation_assoc_fraction_at_rna1_spots", "partner",
             "fraction of puncta, pooled over the image",
             "Fraction of {rna1} puncta above the rotation-null 95th percentile, "
             "pooled per image", level="image", source=IMAGE),
    Endpoint("partner_pooled_rotation_null_p_empirical",
             "protein_pooled_rotation_null_p_empirical_at_rna1_spots", "partner",
             "empirical p, pooled over the image",
             "{protein} pooled empirical p at {rna1} puncta, rotation null",
             level="image", source=IMAGE, descriptive_only=True),
    Endpoint("partner_pooled_random_null_p_empirical",
             "protein_pooled_null_p_empirical_at_rna1_spots", "partner",
             "empirical p, pooled over the image",
             "{protein} pooled empirical p at {rna1} puncta, random-position null",
             level="image", source=IMAGE, descriptive_only=True),
    Endpoint("paired_fraction_rna1_at_0p3um", "paired_fraction_rna1_at_0p3um", "partner",
             "fraction of {rna1} puncta with a partner punctum within 0.3 micrometres",
             "Fraction of {rna1} puncta paired to {protein} within 0.3 micrometres",
             exploratory=True),
    Endpoint("paired_fraction_partner_at_0p3um", "paired_fraction_protein_at_0p3um",
             "partner",
             "fraction of {protein} puncta with an anchor punctum within 0.3 micrometres",
             "Fraction of {protein} puncta paired to {rna1} within 0.3 micrometres",
             exploratory=True),
    Endpoint("median_nn_distance_rna1_um", "median_nn_distance_rna1_um", "partner",
             "micrometres",
             "Median nearest-neighbour distance from {rna1} to {protein}",
             exploratory=True),
    Endpoint("median_nn_distance_partner_um", "median_nn_distance_protein_um", "partner",
             "micrometres",
             "Median nearest-neighbour distance from {protein} to {rna1}",
             exploratory=True),
    Endpoint("rna1_enrichment_at_partner_puncta", "rna1_enrichment_at_protein_spots",
             "partner", "observed divided by local background",
             "{rna1} enrichment at {protein} puncta, local background only",
             note="Local-background enrichment with no null model. The reciprocal "
                  "headline is the rotation-null endpoint below."),
    Endpoint("rna1_rotation_enrichment_at_partner_puncta",
             "rna1_rotation_enrichment_at_protein_spots", "partner",
             "observed divided by the partner-anchored rotation-null mean",
             "{rna1} enrichment at {protein} puncta, rotation null", primary=True,
             usability_flag="rotation_null_usable_at_protein_spots",
             note="Reciprocal headline. When the partner is diffuse, most nuclei fail "
                  "the usability test at the partner anchor because the anchor tiles "
                  "the nucleus; that makes this direction UNINFORMATIVE, not negative, "
                  "and it is not the reciprocal of the anchor-side result."),
    Endpoint("rna1_rotation_enrichment_at_partner_puncta_allnuclei",
             "rna1_rotation_enrichment_at_protein_spots", "partner",
             "observed divided by the partner-anchored rotation-null mean",
             "{rna1} enrichment at {protein} puncta, rotation null, every nucleus",
             descriptive_only=True,
             note="SENSITIVITY: the same endpoint over every nucleus, including those "
                  "whose partner-anchored null the engine flagged unusable."),
    Endpoint("rna1_rotation_null_z_at_partner_puncta",
             "rna1_rotation_null_z_at_protein_spots", "partner",
             "z against the partner-anchored rotation null",
             "{rna1} z at {protein} puncta, rotation null",
             usability_flag="rotation_null_usable_at_protein_spots"),
    Endpoint("rna1_rotation_null_p_at_partner_puncta",
             "rna1_rotation_null_p_at_protein_spots", "partner",
             "empirical p against the partner-anchored rotation null",
             "{rna1} per-nucleus empirical p at {protein} puncta",
             descriptive_only=True,
             usability_flag="rotation_null_usable_at_protein_spots"),
    Endpoint("manders_rna1_in_partner", "manders_rna1_in_protein", "partner",
             "fraction of anchor signal in partner-positive pixels",
             "Manders fraction of {rna1} signal in {protein}-positive pixels",
             excluded_from_holm="tracks absolute partner level",
             note="CONTEXT ONLY. A pixel-overlap coefficient sensitive to the pixel "
                  "threshold; it is not an enrichment test and it tracks the absolute "
                  "partner level, so it carries no multiplicity adjustment."),
    Endpoint("rna1_pooled_rotation_enrichment_at_partner_puncta",
             "rna1_pooled_rotation_enrichment_at_protein_spots", "partner",
             "observed divided by the partner-anchored rotation-null mean, pooled over "
             "the image",
             "{rna1} enrichment at {protein} puncta, pooled per image",
             level="image", source=IMAGE),
    Endpoint("rna1_pooled_rotation_null_p_empirical",
             "rna1_pooled_rotation_null_p_empirical_at_protein_spots", "partner",
             "empirical p, pooled over the image",
             "{rna1} pooled empirical p at {protein} puncta, rotation null",
             level="image", source=IMAGE, descriptive_only=True),
)


def channel_labels(cfg: dict) -> Dict[str, str]:
    """Plain-language channel names from the run's own resolved config."""
    out = {"rna1": "RNA1", "rna2": "RNA2", "protein": "Protein", "dapi": "DAPI"}
    try:
        ch = cfg["config_resolved"]["channels"]
    except Exception:                                          # noqa: BLE001
        return out
    for role, key in (("rna1", "rna_label"), ("rna2", "rna2_label"),
                      ("protein", "antibody_label"), ("dapi", "dapi_label")):
        val = ch.get(key)
        if val:
            out[role] = str(val)
    return out


def resolve(nuc: pd.DataFrame, per_image: pd.DataFrame
            ) -> Tuple[List[Endpoint], List[str]]:
    """Keep every endpoint. Columns the run did not emit become all-NA and are
    listed, so a missing partner-anchored null degrades to NA with a note instead
    of silently vanishing from the report."""
    absent: List[str] = []
    for ep in ENDPOINTS:
        frame = nuc if ep.level == "nucleus" else per_image
        if ep.column not in frame.columns:
            absent.append(ep.name)
            frame[ep.column] = np.nan
    return list(ENDPOINTS), absent


def usable_endpoints(endpoints: Sequence[Endpoint], absent: Sequence[str]
                     ) -> List[Endpoint]:
    absent = set(absent)
    return [ep for ep in endpoints if ep.name not in absent]
