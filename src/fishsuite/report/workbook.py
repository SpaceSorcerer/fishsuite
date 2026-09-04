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
    "Read me": (
        "What this workbook is, which fishsuite run produced it, how the replicate "
        "structure works and what each other sheet holds. Read the replicate-unit row "
        "before reading any p-value: the well is the biological replicate and every "
        "test runs on well means."),
    "Spots per nucleus by group": (
        "How much signal each nucleus carries, compared between condition groups: "
        "puncta counted per nucleus, punctum size, and absolute intensity. One row per "
        "endpoint and comparison, with each well's own mean in its own column so the "
        "replicates behind every test are visible. Absolute-intensity rows are "
        "descriptive only and carry no multiplicity adjustment."),
    "Nuclear fraction by group": (
        "Where the signal sits rather than how much of it there is, compared between "
        "condition groups: the nuclear fraction of each nucleus's puncta, the "
        "area-normalised density, and nuclear-to-cytoplasmic intensity ratios. These "
        "are within-nucleus ratios, so a shifted detection floor moves numerator and "
        "denominator together."),
    "Partner at puncta by group": (
        "Whether the partner channel is enriched at the anchor channel's puncta, "
        "against the engine's own per-nucleus nulls, plus punctum-to-punctum pairing "
        "and the reciprocal direction. Enrichment above one in every group is "
        "co-distribution with a nuclear sub-compartment, not evidence of a specific "
        "molecular association."),
    "Per well": (
        "One row per endpoint and well. The well mean of that well's field values is "
        "the point every test is run on, so this sheet is the input to the Contrasts "
        "sheet. Nucleus counts before and after any usability filter are carried "
        "alongside, so a filtered endpoint cannot hide how much it dropped."),
    "Per field": (
        "One row per endpoint and field of view. A field is a technical replicate "
        "within a well and is never tested directly; field values are averaged into "
        "the well means on the Per well sheet. The nucleus count per field and the "
        "quality-control floor it was checked against are both recorded."),
    "Per nucleus": (
        "The measurement level: one row per segmented nucleus, with its image, well, "
        "condition group and every per-nucleus endpoint column. Nuclei are "
        "pseudoreplicates and are never tested directly; they are here so any well "
        "mean can be traced back to the nuclei that produced it."),
    "Contrasts": (
        "Every endpoint tested between condition groups, in full. Welch t on well "
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


def _description(name: str) -> str:
    return SHEET_DESCRIPTION.get(name, "")


def write(path: Path, sheets: Dict[str, pd.DataFrame],
          strike_rows: Dict[str, Sequence[int]] | None = None,
          order: Sequence[str] | None = None) -> Path:
    """Write the workbook. Row 1 of every sheet is its description, row 2 the header."""
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    strike_rows = strike_rows or {}
    order = list(order or SHEET_ORDER)
    path = Path(path)
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
            ws.cell(row=1, column=1, value=_description(name))
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
