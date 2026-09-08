"""``REPORT.xlsx`` — the condition-versus-condition workbook.

Sheet names are plain language and every sheet opens with a two-to-three sentence
description of what it holds, so the workbook can be read without a key. There
are no ``Q1``-style codes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

SHEET_ORDER: List[str] = [
    "Read me",
    "Spots per nucleus by group",
    "Nuclear fraction by group",
    "Partner at puncta by group",
    "Per well",
    "Per field",
    "Per nucleus",
    "Contrasts",
    "Secondary-only",
    "Run provenance",
]

SHEET_DESCRIPTION: Dict[str, str] = {
    'Micrograph panels': 'Four native publication PNGs per representative arm: all channels, partner plus RNA, partner alone, RNA alone. Absolute paths, SHA256, recorded manual display windows and configured wavelength LUTs are retained. No DAPI-only panel.',
    'FOV outlier sensitivity': 'Pre-specified sensitivity only; never applied to headline. Within each well and endpoint, flag leave-one-out |z| > 2.5 with at least 3 finite FOVs; sample SD of other FOVs; exclude at most one maximum per well. Zero SD yields signed infinity for a differing value, zero otherwise; ties use image name. Flag rows and well-mean Welch contrasts with and without exclusions.', 
    'Ratio intervals': 'Fixed-10 MIAT/QKI: R = (A_KD/A_NT)/(T_KD/T_NT), using six matched well means per arm and within-arm A,T covariance. Marginal normal log-delta 95% intervals are not exact-test inversions or simultaneous intervals. All q95 ratio tests remain exploratory.',
    'Retention sensitivity': 'Policy-specific plug-in normal R-MDE at 80% power, alpha .05 and .05/8, with SE and covariance. Detection thresholds around R=1 are not observed-CI limits, equivalence margins, or exact-permutation power.',
    'Coverage': 'Fixed-10 policy-specific observed, candidate, usable and positive counts, with separate noncandidate, candidate-unusable, usable-negative and usable-positive states. Well coverage ratios have finite n, SD and arm t intervals; unknown calls are never negative biology.',
    'COUNT localization': 'Separate COUNT all-eligible localization values copied from its own persisted report. Its population and detection differ from fixed-10 coloc; these values are forbidden as MIAT/QKI ratio denominators.',
    'Ratio well values': 'Matched A,T well vectors with slide/arm IDs and original CSV row numbers. Nucleus means are averaged within FOV and FOV means equally within well; total T is not multiplied by the number of policies.',
    'Nucleus roster': 'All four policy copies of the 370 fixed-10 nuclei, including four restored zero-spot nuclei per policy. Policy-specific observed/candidate/usable/q95 fields govern reconstruction; historical aliases are retained only as source data.',
    'Assignments': 'All 400 assignments of three KD and three NT labels among six wells on each of two slides. A,T pairs move intact by well; two-sided abs(log R) uses >= observed minus 1e-12 and denominator 400. Original assignment file missing; enumeration reproduces every delivered R p-value.',
    'Scalar contrasts': 'Ten exploratory scalar tests: two total MIAT endpoints and eight associated-pool endpoints, using raw Welch and Hedges g on six wells per arm, with a distinct Holm-ten family. Historical exact scalar p-values remain separately labelled.',
    'Usable pool ratios': 'Coverage-matched sensitivity: positive A and usable-footprint T share support, preserving the full nucleus roster and equal-weight FOV/well hierarchy. Usable-support intensity mass is missing and remains NA with reason.',
    'Historical union sensitivity': 'Separately sourced historical fixed-10 q95 union-deduplicated intensity sensitivity from pooled_Fig3. No union-deduplicated values are invented for the other policies.',
    'Source provenance': 'Absolute named input paths and SHA256 hashes. Source files are read only; the report does not rerun segmentation, detection or spatial nulls.',
    "Localization counts": "Localization per nucleus and assigned cell territory. Authoritative counts are reconciled with spot assignments; zero-total fractions remain NA.",
    "Localization unassigned": "Unassigned spots (source nucleus_id 0), excluded from localization per nucleus and assigned cell territory and tabulated separately by image.",
    "Localization checks": "Source-column and NA-mask checks for localization per nucleus and assigned cell territory.",
    "Localization territory": "Recorded geometry for localization per nucleus and assigned cell territory; an assigned territory is not an anatomical cell boundary.",
    "Localization crops": "Crop selection and provenance for localization per nucleus and assigned cell territory; missing boundary masks are explicit.",
    "Read me": (
        "What this workbook is, which fishsuite run produced it, how the replicate "
        "structure works and what each other sheet holds. Read the replicate-unit row "
        "before reading any p-value: the well is the biological replicate and every "
        "headline uses the nucleus-level mixed model; Welch on well means remains reported."),
    "Spots per nucleus by group": (
        "How much signal each nucleus carries, compared between condition groups: "
        "puncta counted per nucleus, punctum size, and absolute intensity. One row per "
        "endpoint and comparison, with each well's own mean in its own column so the "
        "replicates behind every test are visible. Absolute-intensity rows are "
        "descriptive only and carry no multiplicity adjustment."),
    "Nuclear fraction by group": (
        "When DAPI correction is enabled, mask-only nuclear_spot_fraction is legacy_mask_only; use nuclear_spot_fraction_dapi. "
        "Where the signal sits rather than how much of it there is, compared between "
        "condition groups: the nuclear fraction of each nucleus's puncta, the "
        "area-normalised density, and nuclear-to-cytoplasmic intensity ratios. These "
        "are within-nucleus ratios, so a shifted detection floor moves numerator and "
        "denominator together. Non-detection is not evidence of no effect; MDE is not an exclusion bound."),
    "Partner at puncta by group": (
        "Whether the partner channel is enriched at the anchor channel's puncta, "
        "against the engine's own per-nucleus nulls, plus punctum-to-punctum pairing "
        "and the reciprocal direction. Enrichment above one in every group is "
        "co-distribution with a nuclear sub-compartment, not evidence of a specific "
        "molecular association."),
    "Per well": (
        "One row per endpoint and well. The well mean of that well's field values is "
        "the point the supplementary Welch comparison is run on, so this sheet is the input to the Contrasts "
        "sheet. Nucleus counts before and after any usability filter are carried "
        "alongside, so a filtered endpoint cannot hide how much it dropped."),
    "Per field": (
        "One row per endpoint and field of view. A field is a technical replicate "
        "within a well; direct FOV Welch is a technical-level sensitivity only. Field values are averaged into "
        "the well means on the Per well sheet. The nucleus count per field and the "
        "quality-control floor it was checked against are both recorded."),
    "Per nucleus": (
        "cyto_spot_count and nuclear_spot_fraction are legacy_mask_only when DAPI-corrected columns are present. "
        "The measurement level: one row per segmented nucleus, with its image, well, "
        "condition group and every per-nucleus endpoint column. Nuclei are "
        "measurement units; the mixed-model sensitivity accounts for well and nested FOV clustering. Any well "
        "mean can be traced back to the nuclei that produced it."),
    "Contrasts": (
        "Sensitivity analyses: added after inspection of the data; the pre-specified gate is Welch on well means. Mixed model uses REML with arm fixed, well and FOV-within-well random intercepts; two-sided asymptotic Wald p. Student uses pooled variance on well means; FOV Welch is technical-level only. Every endpoint tested between condition groups, in full. Welch t on well "
        "means with Hedges g and a 95 percent interval, the raw p, the "
        "Holm-adjusted p within its endpoint family, and the minimum detectable "
        "effect at this number of wells. Exact permutation of well labels and Tukey on "
        "field means are sensitivity columns, never the gate."),
    "Secondary-only": (
        "The no-probe, no-primary-antibody control fields, one row each, with the "
        "detection they produced at this run's thresholds. Any field excluded is "
        "excluded by a stated rule or an operator-supplied reason recorded in its own "
        "column, and the sensitivity columns show what including everything would do."),
    "Run provenance": (
        "Which fishsuite run this report was built from, and everything needed to "
        "reproduce it. The run directory, the detection thresholds read back off the "
        "run itself, the segmentation model and version, library versions, the seed, "
        "the group definitions and any excluded field with its reason. Checksums of "
        "the source tables are recorded so a later report can be checked against the "
        "same bytes."),
}


SHEET_DESCRIPTION['Contrasts'] = ('Mixed-model headline decision 2026-09-07 after data inspection; Welch on well means remains reported. p_mixed / p_mixed_holm are nucleus-level mixed p and headline-family adjustment. p_headline / p_headline_holm also include explicitly labeled Welch fallbacks. p_welch retains the well-mean comparison. significant_holm_0p05 retains the legacy Welch Holm flag; significant_headline_holm_0p05 reports the mixed/fallback headline Holm flag. Endpoint thresholds, usability filters, Hedges g, Holm and well-based MDE are recorded in this workbook; MDE is not mixed-model power.')


def _description(name: str) -> str:
    return SHEET_DESCRIPTION.get(name, "")


SHEET_DESCRIPTION.update({
    'Coloc figure sources':'Re-rendered persisted panel figure hashes with exact source workbook and imported report per-well cells. Original values remain unchanged; biological wells only.',
    'Deck specification':'Curated slide titles, readouts and qualifications from the requested deck specification. Measurement claims resolve separately to report data cells.',
    'Endpoint coverage':'Defined nucleus counts and eligible counts are distinct. Summed per-field finite counts retain endpoint-specific missingness and usability masks; no imputation.',
    'Coloc per nucleus':'Persisted standard-panel values for both nuclear anchors. Missing calls are not negatives. Costes-only values require convergence; fallback mixtures remain separate.',
    'Coloc per field':'Persisted field means, retained verbatim. Fields are technical replicates within biological wells.',
    'Coloc per well':'Exact persisted per-well panel numbers. The group column aliases source line; source values and their cells are retained in Coloc source cells.',
    'Coloc contrasts':'Historical panel contrasts, unchanged. Descriptive rows have no p values. New exploratory partner tests appear separately in Contrasts, with amended Holm.',
    'Coloc line profiles':'Persisted profile samples, with image, well, nuclear anchor and calibrated distance. No pixel resampling or new null draws.',
    'Coloc source cells':'Exact source workbook sheet/cell addresses and file SHA256 hashes, plus the frozen figure index. All source files are read only.',
    'Endpoint definitions':'Full endpoint map including total IF (descriptive, acquisition-qualified), nuclear-forward pairing, and distinct reverse signal/calls/pairing denominators. Whole-territory pairing retains its historical identity.',
    'Multiplicity plan':'Frozen original members plus declared additions: detection 4 to 5 (A2); partner 22 to 27 (A3); localization unchanged. Observed call/pairing aliases are counted once. No pruning on p.',
    'Slide values':'Slide values resolved after workbook layout; each value names its source measurement/provenance sheet and cell. MEASURED direction follows the observed contrast.',
    'Figure sources':'Figure paths and SHA256 hashes, endpoints, cohort, filters and tests. Speaker notes resolve numeric values to workbook cells.',
    'Nucleus selection':'Retained labels are the starting population. Pre-area/pre-border candidates and exclusions remain missing. Endpoint usability masks act later.',
    'Selection evidence':'Verbatim A0 nucleus-selection evidence and source hash, kept as provenance rather than an instruction to execute.',
    'Micrographs':'Exact persisted micrograph paths, hashes, image identities, planes and source pixel calibration. The native annotated scale bar is retained; its pixel length is checked against the source calibration. No new segmentation.',
})


def write(path: Path, sheets: Dict[str, pd.DataFrame],
          strike_rows: Dict[str, Sequence[int]] | None = None,
          order: Sequence[str] | None = None,
          descriptions: Dict[str, str] | None = None) -> Path:
    """Write the workbook. Row 1 of every sheet is its description, row 2 the header."""
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    strike_rows = strike_rows or {}
    order = list(order or SHEET_ORDER)
    path = Path(path)
    from .provenance import guard_output
    guard_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        for name in order:
            df = sheets.get(name)
            if df is None or len(df) == 0:
                df = pd.DataFrame({"note": [f"the sheet '{name}' produced no rows"]})
            df.to_excel(xl, sheet_name=name[:31], index=False, startrow=1)
        for name in order:
            ws = xl.book[name[:31]]
            ncol = max(ws.max_column, 1)
            ws.cell(row=1, column=1, value=(descriptions or {}).get(name, _description(name)))
            ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=ncol)
            head = ws.cell(row=1, column=1)
            head.alignment = Alignment(wrap_text=True, vertical="top")
            head.font = Font(italic=True, color="333333")
            head.fill = PatternFill("solid", fgColor="F2F2F2")
            ws.row_dimensions[1].height = 46
            for c in range(1, ncol + 1):
                ws.cell(row=2, column=c).font = Font(bold=True)
            ws.freeze_panes = "A3"
            ws.auto_filter.ref = f"A2:{get_column_letter(ncol)}{max(ws.max_row, 3)}"
            for col in range(1, ncol + 1):
                header = ws.cell(row=2, column=col).value
                ws.column_dimensions[get_column_letter(col)].width = min(
                    60, max(12, len(str(header)) + 2 if header else 12))
            if name == "Read me":
                ws.column_dimensions["A"].width = 22
                ws.column_dimensions["B"].width = 46
                ws.column_dimensions["C"].width = 110
                for row in ws.iter_rows(min_row=3, min_col=3, max_col=3):
                    for cell in row:
                        cell.alignment = Alignment(wrap_text=True, vertical="top")
            for r in strike_rows.get(name, []):
                for cell in ws[int(r) + 3]:   # +1 description, +1 header, +1 to 1-based
                    cell.font = Font(strike=True)
    return path
