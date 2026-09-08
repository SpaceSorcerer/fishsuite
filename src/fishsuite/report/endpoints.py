"""Endpoint registry for the ``fishsuite report`` layer.

Endpoints are declared once, in ROLE terms (rna1 / rna2 / protein), and resolved
against whatever columns a given run actually emitted. An endpoint whose column
the run did not write is kept, marked absent, and reported as NA with a note; it
is never silently dropped.

Family names are plain language and map one-to-one onto the workbook's by-group
sheets. There are no ``Q1``-style codes anywhere in this layer.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
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
        "size, and absolute intensity. Acquisition uniform per acquirer 2026-09-07; "
        "staining batch caveat. Absolute intensities enter the detection Holm family."),
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
FOOTPRINT = ("spot_metrics.csv miat_footprint_area_px (legacy column name; it is the "
             "half-maximum footprint of the rna1 punctum), times the run voxel area, "
             "aggregated per nucleus")
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
    # Alternative spellings of ``column``, tried in order when it is absent. The
    # engine relabels the rna2 partner columns to ``protein_*`` in rna_protein
    # mode but leaves them ``rna2_*`` in rna_rna, so one endpoint covers both.
    alt_columns: Tuple[str, ...] = ()
    note: str = ""
    axis_group: str = ""

    def pretty(self, labels: Dict[str, str]) -> str:
        out = self.label
        for role, text in labels.items():
            out = out.replace("{" + role + "}", str(text))
        return out


# Ratio-module scalar estimands: paired pools share a display scale by unit.
RATIO_ENDPOINTS = tuple(
    Endpoint('ratio_' + kind + '_' + pool, pool, 'detection', unit,
             title, axis_group='ratio_' + kind + '_' + group,
             absolute_intensity=kind != 'miat_count')
    for kind, unit, group, titles in (
        ('protein', 'Intensity (AU)', 'intensity',
         ('Total nuclear RNASEH2B', 'RNASEH2B at BIN1 introns')),
        ('miat_count', 'puncta per nucleus', 'count',
         ('Total nuclear MIAT', 'QKI-associated MIAT')),
        ('miat_intensity', 'MIAT footprint intensity per nucleus (a.u.)', 'intensity',
         ('Total nuclear MIAT intensity', 'QKI-associated MIAT intensity')))
    for pool, title in zip(('total', 'associated'), titles)
)


ENDPOINTS: Tuple[Endpoint, ...] = (
    # ------------------------------------------------------------- detection
    Endpoint("rna1_spots_per_nucleus", "n_spots_rna1", "detection",
             "puncta per nucleus", "{rna1} puncta per nucleus", primary=True,
             note="Total puncta detected in that nucleus, nuclear plus cytoplasmic."),
    Endpoint("rna1_nuclear_spots_per_nucleus", "nuclear_spot_count", "detection",
             "nuclear puncta per nucleus", "{rna1} nuclear puncta per nucleus"),
    Endpoint("rna1_cyto_spots_per_nucleus", "cyto_spot_count", "detection",
             "cytoplasmic puncta per nucleus and assigned cell territory",
             "{rna1} assigned-cytoplasmic puncta",
             note="A2 detection-family amendment: one additional count test. Read "
                  "cyto_spot_count without replacing historical aliases. Zero counts "
                  "remain zero. Territory is the run's recorded estimate, not a "
                  "membrane-bounded cell; unassigned spots are excluded. Detection "
                  "Holm and family-alpha MDE are amended, not baseline parity."),
    # ---- punctum size ----
    # The footprint endpoints are the size measurement. The moment-estimator
    # columns below them SATURATE and are kept only for continuity; see the note
    # on each and docs/REPORT_SUBCOMMAND_2026-09-04.md.
    Endpoint("rna1_punctum_footprint_area_um2", "rna1_punctum_footprint_area_um2",
             "detection", "square micrometres", "{rna1} punctum footprint area",
             source=FOOTPRINT, primary=True,
             note="THE punctum-size endpoint. Area of the connected component of "
                  "pixels at or above the local half-maximum that contains the "
                  "punctum, converted from pixels with the run's own voxel area. It "
                  "is segmented per punctum, so unlike the moment estimator it does "
                  "not saturate."),
    Endpoint("rna1_punctum_equivalent_diameter_um",
             "rna1_punctum_equivalent_diameter_um", "detection", "micrometres",
             "{rna1} punctum equivalent diameter", source=FOOTPRINT,
             note="The diameter of a circle with the same area as the punctum's "
                  "half-maximum footprint. A shape-free restatement of the footprint "
                  "area, in the units a reader expects for a punctum size."),
    Endpoint("rna1_spot_fwhm_px", "rna1_spot_fwhm_px", "detection",
             "pixels", "{rna1} punctum width, moment estimator (saturating)",
             source=SPOT, descriptive_only=True,
             note="DESCRIPTIVE ONLY, AND SATURATING. Measured per punctum as the "
                  "second central moment of the background-subtracted signal in a "
                  "FIXED 9 by 9 pixel crop, then converted to a full width at half "
                  "maximum. It is not a kernel constant, but the fixed crop truncates "
                  "the moment: measured on synthetic Gaussians it is accurate below "
                  "about 3 pixels of true width, reads about 14 percent low at 4.7 "
                  "pixels, and asymptotes near 4.7 pixels however wide the punctum "
                  "actually is. Above the linear range it reports local crowding "
                  "rather than punctum size, so two conditions can look identical on "
                  "it while differing in real size. Read the footprint area instead."),
    Endpoint("rna1_spot_diameter_um", "rna1_spot_diameter_um", "detection",
             "micrometres", "{rna1} punctum diameter, moment estimator (saturating)",
             source=SPOT, descriptive_only=True,
             note="DESCRIPTIVE ONLY, AND SATURATING. The same moment estimator as "
                  "the width above, multiplied by the voxel size; it inherits the "
                  "same saturation. Read the footprint area instead."),
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
                  "ratio instead of this level.",
             alt_columns=("rna2_nuclear_mean",)),
    Endpoint("protein_spots_per_nucleus", "n_spots_protein", "detection",
             "puncta per nucleus", "{protein} puncta per nucleus", exploratory=True,
             excluded_from_holm="proxy for absolute partner level",
             note="EXPLORATORY. The count is thresholded at one global intensity, so "
                  "for a diffuse partner it tracks absolute level rather than object "
                  "number and carries no multiplicity adjustment."),
    Endpoint("rna2_spots_per_nucleus", "n_spots_rna2", "detection",
             "puncta per nucleus", "{rna2} puncta per nucleus"),
    Endpoint("rna2_nuclear_spots_per_nucleus", "nuclear_spot_count_rna2", "detection",
             "nuclear puncta per nucleus", "{rna2} nuclear puncta per nucleus"),
    Endpoint("rna2_punctum_footprint_area_um2", "rna2_punctum_footprint_area_um2",
             "detection", "square micrometres", "{rna2} punctum footprint area",
             source=FOOTPRINT,
             note="Punctum size for the SECOND RNA channel, from its own "
                  "half-maximum footprint. Absent unless the run computed the "
                  "footprint for that channel."),
    Endpoint("rna2_nuclear_above_floor_intensity", "nuclear_above_floor_intensity_rna2",
             "detection", "arbitrary units",
             "{rna2} nuclear intensity above the detection floor",
             absolute_intensity=True,
             note="DESCRIPTIVE ONLY: an absolute intensity is not comparable as a "
                  "level claim across sections."),

    # ---------------------------------------------------------- localization
    Endpoint("rna1_nuclear_spot_fraction", "nuclear_spot_fraction", "localization",
             "fraction of that nucleus's puncta that are nuclear",
             "{rna1} nuclear fraction", primary=True,
             note="Floor-robust headline: a ratio within one nucleus, so a shifted "
                  "detection floor moves numerator and denominator together."),
    Endpoint("rna2_nuclear_spot_fraction", "nuclear_spot_fraction_rna2", "localization",
             "fraction of that nucleus's puncta that are nuclear",
             "{rna2} nuclear fraction", primary=True,
             note="PRIMARY for an rna_rna run whose second channel is the mature "
                  "species: a within-nucleus ratio, so a shifted detection floor "
                  "moves numerator and denominator together. The anchor channel's "
                  "nuclear fraction is the matching primary for rna1."),
    Endpoint("rna2_spots_per_um2", "rna2_spots_per_um2", "localization",
             "TOTAL puncta per square micrometre of NUCLEAR area",
             "{rna2} total puncta per square micrometre of nucleus",
             source=DERIVED_DENSITY,
             note="SENSITIVITY for the {rna2} per-nucleus count, normalising for any "
                  "difference in nuclear area between groups. SCOPE MISMATCH: the "
                  "numerator is every punctum in the cell, nuclear and cytoplasmic, "
                  "while the denominator is the NUCLEAR area only, so it is a count "
                  "normalised by nuclear size and not a nuclear concentration. Use "
                  "rna2_nuclear_spot_fraction, or the scope-matched "
                  "protein_spots_per_um2 for the partner, when concentration is "
                  "what is meant."),
    Endpoint("rna1_spots_per_um2", "rna1_spots_per_um2", "localization",
             "TOTAL puncta per square micrometre of NUCLEAR area",
             "{rna1} total puncta per square micrometre of nucleus",
             source=DERIVED_DENSITY,
             note="SENSITIVITY for the per-nucleus count, normalising for any "
                  "difference in nuclear area between groups. SCOPE MISMATCH: the "
                  "numerator is every punctum in the cell, nuclear and cytoplasmic, "
                  "while the denominator is the NUCLEAR area only, so it is a count "
                  "normalised by nuclear size and not a nuclear concentration. "
                  "rna1_nuclear_spots_per_um2 is the scope-matched companion."),
    Endpoint("rna1_nuclear_spots_per_um2", "rna1_nuclear_spots_per_um2",
             "localization", "nuclear puncta per square micrometre of nuclear area",
             "{rna1} nuclear puncta per square micrometre",
             source=DERIVED_DENSITY,
             note="SCOPE-MATCHED density: NUCLEAR puncta over NUCLEAR area, so it is "
                  "a concentration. Read from the engine's own "
                  "nuclear_spot_density_per_um2 when the run emitted it, otherwise "
                  "nuclear_spot_count / nucleus_area_um2. Companion to "
                  "rna1_spots_per_um2, whose numerator is the whole cell."),
    Endpoint("protein_spots_per_um2", "protein_spots_per_um2", "localization",
             "nuclear puncta per square micrometre of nuclear area",
             "{protein} nuclear puncta per square micrometre",
             source=DERIVED_DENSITY, exploratory=True,
             excluded_from_holm="proxy for absolute partner level",
             note="EXPLORATORY, and the per-area companion to "
                  "protein_spots_per_nucleus. NUCLEAR partner puncta over NUCLEAR "
                  "area, from the engine's nuclear_spot_density_per_um2_protein when "
                  "present, otherwise nuclear_spot_count_protein / nucleus_area_um2. "
                  "It inherits its parent's limitation: the count is thresholded at "
                  "one global intensity, so for a diffuse partner it tracks absolute "
                  "level rather than object number, and it carries no multiplicity "
                  "adjustment for the same reason."),
    Endpoint("nucleus_area_um2", "nucleus_area_um2", "localization",
             "square micrometres", "Nucleus area", source=DERIVED_AREA,
             descriptive_only=True,
             note="DESIGN DESCRIPTOR, not a result. Reported because a segmentation "
                  "area floor can truncate two groups unequally."),
    Endpoint("protein_nc_ratio", "protein_nc_ratio", "localization",
             "nuclear to cytoplasmic mean intensity ratio",
             "{protein} nuclear-to-cytoplasmic ratio",
             alt_columns=("rna2_nc_ratio",),
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
             exploratory=True,
             note="DEFINITION, because a similarly named endpoint elsewhere is NOT "
                  "the same quantity. Denominator: EVERY punctum of the anchor "
                  "channel in that nucleus, nuclear and cytoplasmic. Distance: "
                  "three-dimensional centroid separation at or below 0.3 "
                  "micrometres, using the run's own voxel size. Partner set: every "
                  "partner punctum, not restricted to the nucleus. An endpoint that "
                  "restricts either side to nuclear puncta, or uses a different "
                  "pairing distance, will differ from this one and the two must not "
                  "be compared without restating both definitions."),
    Endpoint("paired_fraction_partner_at_0p3um", "paired_fraction_protein_at_0p3um",
             "partner",
             "fraction of {protein} puncta with an anchor punctum within 0.3 micrometres",
             "Fraction of {protein} puncta paired to {rna1} within 0.3 micrometres",
             exploratory=True,
             alt_columns=("paired_fraction_rna2_at_0p3um",),
             note="DEFINITION, the mirror of the anchor-side paired fraction and NOT "
                  "its reciprocal. Denominator: EVERY punctum of the PARTNER channel "
                  "in that nucleus, nuclear and cytoplasmic. Distance: three-"
                  "dimensional centroid separation at or below 0.3 micrometres, using "
                  "the run's own voxel size. Anchor set: every anchor punctum, not "
                  "restricted to the nucleus. It differs from the anchor-side fraction "
                  "whenever the two channels have different punctum counts, because "
                  "only the denominator changes; the two are not interchangeable and "
                  "neither implies the other."),
    Endpoint("median_nn_distance_rna1_um", "median_nn_distance_rna1_um", "partner",
             "micrometres",
             "Median nearest-neighbour distance from {rna1} to {protein}",
             exploratory=True),
    Endpoint("median_nn_distance_partner_um", "median_nn_distance_protein_um", "partner",
             "micrometres",
             "Median nearest-neighbour distance from {protein} to {rna1}",
             exploratory=True,
             alt_columns=("median_nn_distance_rna2_um",)),
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
                  "partner level, so it carries no multiplicity adjustment.",
             alt_columns=("manders_rna1_in_rna2",)),
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


INTENSITY_CAVEAT = 'same acquisition settings (acquirer); staining batch not controlled'
AXIS_GROUPS = {
    'protein_nuclear_mean': 'protein_nuclear_mean',
    'protein_nuclear_mean_seconly_corrected': 'protein_nuclear_mean',
    'paired_frac_rna1_at_partner': 'pairing_fraction',
    'paired_fraction_rna1_at_0p3um': 'pairing_fraction',
    'paired_fraction_partner_at_0p3um': 'pairing_fraction',
    'paired_frac_partner_at_rna1': 'pairing_fraction',
    'paired_frac_rna1_at_partner_minus_shuffle': 'pairing_excess',
    'paired_frac_partner_at_rna1_minus_shuffle': 'pairing_excess',
    'fraction_rna1_puncta_partner_positive_exact_footprint': 'called_fraction',
    'frac_called_coloc_partner_runthr': 'called_fraction',
    'frac_called_coloc_partner_minus_shuffle_runthr': 'called_excess',
    'frac_called_coloc_minus_shuffle_runthr': 'called_excess',
}


def amend_endpoint(endpoint):
    """A20 membership and display groups, declared independently of observed p."""
    group = AXIS_GROUPS.get(endpoint.name, endpoint.axis_group)
    if endpoint.absolute_intensity:
        group = group or endpoint.name.removesuffix('_seconly_corrected')
        return replace(endpoint, descriptive_only=False, excluded_from_holm='',
                       note=INTENSITY_CAVEAT, axis_group=group)
    return replace(endpoint, axis_group=group)


ENDPOINTS = tuple(amend_endpoint(e) for e in ENDPOINTS)

# Opt-in A3 registry. The default registry remains the A0 + A2 proposal.
# Freeze membership before computing any p value; never select by observed p.
A3_PARTNER_ADDITIONS = (
    'paired_frac_rna1_at_partner',
    'paired_frac_rna1_at_partner_minus_shuffle',
    'frac_called_coloc_partner_runthr',
    'frac_called_coloc_partner_minus_shuffle_runthr',
    'paired_frac_partner_at_rna1_minus_shuffle',
)
A3_PANEL_ALIASES = {
    'frac_called_coloc_runthr': 'fraction_rna1_puncta_partner_positive_exact_footprint',
    'paired_frac_partner_at_rna1': 'paired_fraction_partner_at_0p3um',
}


def a3_endpoints(panel_columns: Sequence[str]) -> List[Endpoint]:
    """Additional measurements; observed aliases retain their original test identity."""
    out = [Endpoint(c, c, 'detection', 'integrated arbitrary units', label,
                    descriptive_only=True, absolute_intensity=True,
                    note='Descriptive total IF; acquisition comparability is unverified. '
                         'Assigned cell territory is Voronoi, not anatomical cell.')
           for c, label in (
               ('cell_total_intensity_protein', '{protein} total IF in assigned cell territory'),
               ('nuclear_total_intensity_protein', '{protein} total nuclear IF'),
               ('nuclear_above_floor_intensity_protein', '{protein} nuclear intensity above the display floor'),
               ('cell_total_intensity_rna1', '{rna1} total intensity in assigned cell territory'),
               ('nuclear_total_intensity_rna1', '{rna1} total nuclear intensity'))]
    out.append(Endpoint('rna1_local_mean_at_partner_puncta', 'rna1_local_mean_at_protein_spots',
                        'partner', 'arbitrary units', '{rna1} signal at nuclear {protein} puncta',
                        descriptive_only=True, absolute_intensity=True,
                        note='Reverse continuous signal; distinct from calls and pairing.'))
    for c in panel_columns:
        if c in A3_PANEL_ALIASES:
            continue
        if not c.startswith(('paired_frac_', 'frac_called_coloc', 'pearson_r_csp',
                             'li_icq_csp', 'manders_')) or c.endswith('_sd_across_fov'):
            continue
        anchor = ('nuclear partner puncta' if 'partner_at_rna1' in c or
                  'coloc_partner' in c else 'nuclear RNA1 puncta')
        if c.startswith(('pearson_', 'li_', 'manders_')):
            anchor = 'retained nuclear pixels'
        threshold = ('run batch threshold' if 'runthr' in c else
                     'Costes converged only' if 'costes_only' in c else
                     'Costes with fallback mixture' if 'costes' in c or 'coloc' in c else
                     'persisted pairing distance')
        if c.startswith('paired_frac_'):
            label=('{protein} nuclear pairing to {rna1}' if 'partner_at_rna1' in c else
                   '{rna1} nuclear pairing to {protein}')
        elif c.startswith('frac_called_coloc'):
            label=('{rna1} calls at nuclear {protein}' if 'coloc_partner' in c else
                   '{protein} calls at nuclear {rna1}')
            label+=' (run threshold)' if 'runthr' in c else ' (Costes with fallback)'
        else:
            label=c.replace('_',' ')
        if 'minus_shuffle' in c:
            label+=': excess over shuffle'
        elif 'shuffle' in c:
            label+=': shuffled'
        out.append(Endpoint(c, c, 'partner', 'fraction' if 'frac' in c or 'manders' in c else 'coefficient',
                            label, source='persisted coloc_standard_panel.xlsx:per_nucleus',
                            descriptive_only=c not in A3_PARTNER_ADDITIONS,
                            exploratory=True,
                            note=f'Anchor denominator: {anchor}; {threshold}. '
                                 'Observed, shuffled and excess values are separate measurements. '
                                 'Persisted nulls only. A3 partner family amendment; no pruning on p.'))
    return [amend_endpoint(e) for e in out]


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
    """Bind every endpoint to a column this run actually emitted.

    An endpoint whose primary column is missing falls back to the first of its
    alternative spellings that is present. If none is, the endpoint is KEPT, its
    column is created as all-NA and its name is returned in the absent list, so a
    missing partner-anchored null degrades to a reported NA with a note instead of
    silently vanishing from the report.
    """
    absent: List[str] = []
    out: List[Endpoint] = []
    for ep in ENDPOINTS:
        frame = nuc if ep.level == "nucleus" else per_image
        chosen = ep.column
        if chosen not in frame.columns:
            chosen = next((c for c in ep.alt_columns if c in frame.columns), "")
        if not chosen:
            absent.append(ep.name)
            chosen = ep.column
            frame[chosen] = np.nan
        out.append(ep if chosen == ep.column else replace(ep, column=chosen))
    return out, absent


def usable_endpoints(endpoints: Sequence[Endpoint], absent: Sequence[str]
                     ) -> List[Endpoint]:
    absent = set(absent)
    return [ep for ep in endpoints if ep.name not in absent]


# A17 explicit display titles; data labels and endpoint meanings remain unchanged.
SHORT_TITLES = {
    "rna1_spots_per_nucleus": "{rna1} puncta per nucleus",
    "rna1_nuclear_spots_per_nucleus": "{rna1} nuclear puncta",
    "rna1_cyto_spots_per_nucleus": "{rna1} cytoplasmic puncta",
    "rna1_nuclear_spot_fraction": "{rna1} puncta, % nuclear",
    "rna2_nuclear_spot_fraction": "{rna2} puncta, % nuclear",
    "protein_spots_per_nucleus": "{protein} puncta per nucleus",
    "rna1_punctum_footprint_area_um2": "{rna1} footprint area",
    "rna1_punctum_equivalent_diameter_um": "{rna1} footprint diameter",
    "rna1_spot_fwhm_px": "{rna1} moment width",
    "rna1_spot_diameter_um": "{rna1} moment diameter",
    "rna1_nuclear_above_floor_intensity": "{rna1} above-floor intensity",
    "rna2_nuclear_above_floor_intensity": "{rna2} above-floor intensity",
    "protein_nuclear_mean": "{protein} nuclear intensity",
    "rna1_spots_per_um2": "{rna1} puncta / nuclear area",
    "rna2_spots_per_um2": "{rna2} puncta / nuclear area",
    "rna1_nuclear_spots_per_um2": "{rna1} nuclear puncta density",
    "protein_spots_per_um2": "{protein} nuclear puncta density",
    "protein_nc_ratio": "{protein} N:C intensity",
    "rna1_nc_ratio": "{rna1} N:C intensity",
    "partner_rotation_enrichment_at_rna1": "{protein} rotation enrichment",
    "partner_local_enrichment_at_rna1": "{protein} local enrichment",
}
SHORT_TITLE_TEXT = {
    "Fraction of BIN1 introns puncta with RNASEH2B above threshold in the exact footprint": "BIN1 puncta with RNASEH2B",
    "BIN1 intron puncta, % nuclear": "BIN1 puncta, % nuclear",
    "Spot classification: descriptive, no test": "BIN1 spot classification",
    "Area census: descriptive, no test": "Unretained DAPI object area",
}

SHORT_TITLES.update({
    "rna2_spots_per_nucleus": "{rna2} puncta per nucleus",
    "rna2_nuclear_spots_per_nucleus": "{rna2} nuclear puncta",
    "rna2_punctum_footprint_area_um2": "{rna2} footprint area",
    "nucleus_area_um2": "Nuclear area",
    "partner_rotation_enrichment_at_rna1_allnuclei": "{protein} rotation enrichment, all nuclei",
    "partner_rotation_null_z_at_rna1": "{protein} rotation-null z",
    "partner_rotation_null_p_at_rna1": "{protein} rotation-null p",
    "partner_random_null_enrichment_at_rna1": "{protein} random-null enrichment",
    "partner_random_null_z_at_rna1": "{protein} random-null z",
    "fraction_rna1_puncta_above_rotation_null_p95": "{rna1} puncta above null p95",
    "partner_mean_in_exact_rna1_footprint": "{protein} footprint intensity",
    "partner_enrichment_in_exact_rna1_footprint": "{protein} footprint enrichment",
    "fraction_rna1_puncta_partner_positive_exact_footprint": "{rna1} puncta with {protein}",
    "partner_radial_enrichment_at_0p25um": "{protein} radial enrichment, 0.25 µm",
    "partner_radial_enrichment_at_0p5um": "{protein} radial enrichment, 0.5 µm",
    "partner_radial_enrichment_at_0p75um": "{protein} radial enrichment, 0.75 µm",
    "partner_radial_enrichment_at_1um": "{protein} radial enrichment, 1 µm",
    "partner_pooled_rotation_enrichment_at_rna1": "{protein} pooled rotation enrichment",
    "partner_pooled_random_null_enrichment_at_rna1": "{protein} pooled random enrichment",
    "fraction_rna1_puncta_above_rotation_null_p95_pooled": "{rna1} puncta above null p95, pooled",
    "partner_pooled_rotation_null_p_empirical": "{protein} pooled rotation-null p",
    "partner_pooled_random_null_p_empirical": "{protein} pooled random-null p",
    "paired_fraction_rna1_at_0p3um": "{rna1} pairing within 0.3 µm",
    "paired_fraction_partner_at_0p3um": "{protein} pairing within 0.3 µm",
    "median_nn_distance_rna1_um": "{rna1} nearest-partner distance",
    "median_nn_distance_partner_um": "{protein} nearest-RNA distance",
    "rna1_enrichment_at_partner_puncta": "{rna1} local enrichment",
    "rna1_rotation_enrichment_at_partner_puncta": "{rna1} rotation enrichment",
    "rna1_rotation_enrichment_at_partner_puncta_allnuclei": "{rna1} rotation enrichment, all nuclei",
    "rna1_rotation_null_z_at_partner_puncta": "{rna1} rotation-null z",
    "rna1_rotation_null_p_at_partner_puncta": "{rna1} rotation-null p",
    "manders_rna1_in_partner": "{rna1} Manders fraction",
    "rna1_pooled_rotation_enrichment_at_partner_puncta": "{rna1} pooled rotation enrichment",
    "rna1_pooled_rotation_null_p_empirical": "{rna1} pooled rotation-null p",
})

# Dynamic A3 shuffle endpoints: explicit short titles at the same 11 pt size.
SHORT_TITLES.update({
    'cell_total_intensity_protein': '{protein} assigned-cell total IF',
    'cell_total_intensity_protein_seconly_corrected': '{protein} cell IF, corrected',
    'nuclear_total_intensity_protein': '{protein} total nuclear IF',
    'nuclear_total_intensity_protein_seconly_corrected': '{protein} nuclear IF, corrected',
    'nuclear_above_floor_intensity_protein': '{protein} above-floor intensity',
    'nuclear_above_floor_intensity_protein_seconly_corrected': '{protein} above-floor IF, corrected',
    'rna1_nuclear_above_floor_intensity_seconly_corrected': '{rna1} above-floor IF, corrected',
    'cell_total_intensity_rna1': '{rna1} assigned-cell intensity',
    'cell_total_intensity_rna1_seconly_corrected': '{rna1} cell intensity, corrected',
    'nuclear_total_intensity_rna1': '{rna1} total nuclear intensity',
    'nuclear_total_intensity_rna1_seconly_corrected': '{rna1} nuclear intensity, corrected',
    'protein_nuclear_mean_seconly_corrected': '{protein} nuclear mean, corrected',
    'partner_mean_in_exact_rna1_footprint_seconly_corrected': '{protein} footprint IF, corrected',
    'rna1_local_mean_at_partner_puncta_seconly_corrected': '{rna1} at {protein}, corrected',
    'rna1_local_mean_at_partner_puncta': '{rna1} signal at {protein} puncta',
    'frac_called_coloc_shuffle_runthr': '{protein} calls: shuffled',
    'frac_called_coloc_minus_shuffle_runthr': '{protein} calls: excess over shuffle',
    'frac_called_coloc_partner_minus_shuffle_runthr': '{rna1} calls: excess over shuffle',
    'paired_frac_rna1_at_partner_minus_shuffle': '{rna1} pairing: excess over shuffle',
    'paired_frac_partner_at_rna1_minus_shuffle': '{protein} pairing: excess over shuffle',
})


def short_title(endpoint, title, channel_labels=None):
    """Apply an explicit title map without wrapping or changing font size."""
    labels = channel_labels or {}
    return SHORT_TITLES.get(endpoint, SHORT_TITLE_TEXT.get(title, title)).format(
        rna1=labels.get('rna1','RNA1'), rna2=labels.get('rna2','RNA2'),
        protein=labels.get('protein','Protein'))
