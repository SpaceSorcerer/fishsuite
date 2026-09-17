"""Document contracts for the MIAT-QKI publication Methods and terminology."""

from __future__ import annotations

from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs" / "miat_qki"
METHODS_PATH = PROJECT_ROOT / "docs" / "miat_qki" / "MICROSCOPY_ACQUISITION_METHODS.md"
TERMS_PATH = PROJECT_ROOT / "docs" / "miat_qki" / "FIGURE_TERMS_PLAIN_LANGUAGE.md"
DETAILED_METHODS_PATH = PROJECT_ROOT / "docs" / "miat_qki" / "METHODS_EXACT_FOOTPRINT.md"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8").lower()


def _first_non_heading_paragraph(text: str) -> str:
    paragraphs = re.split(r"\n\s*\n", text.strip())
    return next(paragraph.strip() for paragraph in paragraphs if not paragraph.lstrip().startswith("#"))


def test_short_microscopy_manuscript_paragraph_is_first_after_title():
    """Catches prefatory material displacing the manuscript-ready microscopy paragraph."""
    methods = METHODS_PATH.read_text(encoding="utf-8")
    first_paragraph = _first_non_heading_paragraph(methods)
    assert first_paragraph.startswith(
        "MIAT-QKI images from human undifferentiated H9 hESCs in an NT-versus-MIAT-KD comparison"
    )
    assert "No maximum-intensity projection (MIP) or other projection was used" in first_paragraph


def test_all_miat_qki_docs_avoid_forbidden_reader_facing_terminology():
    """Catches deprecated mass terminology leaking into any reader-facing source document."""
    forbidden = ("exact-footprint mass", "miat-mass")
    for path in sorted(DOCS_DIR.rglob("*.md")):
        text = _read(path)
        for phrase in forbidden:
            assert phrase not in text, f"{phrase!r} remains in {path.name}"


def test_detailed_methods_define_all_detected_and_thresholded_intensity_endpoints():
    """Catches conflation of spot-summed, union-deduplicated, global, and thresholded endpoints."""
    methods = _read(DETAILED_METHODS_PATH)
    for required in (
        "miat spot-pixel intensity",
        "spot-summed",
        "shared pixels contribute once for each overlapping spot footprint",
        "union-deduplicated",
        "each shared image pixel contributed once",
        "global intensity used all detected miat spots and their detected pixels",
        "threshold-associated intensity used only q95-positive spots for the primary analysis",
        "q90 and q99 were sensitivity thresholds",
        "strictly greater than",
        "spatial/pseudo-binding measurement, not a direct-binding measurement",
    ):
        assert required in methods
    assert "global floor-passing" not in methods


def test_detailed_methods_preserve_floor_passing_ror_denominators():
    """Catches conflation of global all-detected endpoints with retained q95 RoRs."""
    methods = _read(DETAILED_METHODS_PATH)
    paragraph = next(
        item
        for item in re.split(r"\n\s*\n", methods)
        if item.startswith("ratio-of-ratios comparisons used")
    )
    for required in (
        "associated spot count/floor-passing spot count",
        "associated union-deduplicated miat spot-pixel intensity/floor-passing union-deduplicated miat spot-pixel intensity",
        "associated spot-summed miat spot-pixel intensity/floor-passing spot-summed miat spot-pixel intensity",
    ):
        assert required in paragraph
    assert "all-detected" not in paragraph


def _markdown_table(text: str, headers: tuple[str, ...]) -> dict[str, dict[str, str]]:
    """Return rows from a simple Markdown table keyed by its first column."""
    expected = tuple(header.lower() for header in headers)
    lines = text.splitlines()
    for index, line in enumerate(lines[:-1]):
        cells = tuple(cell.strip().lower() for cell in line.strip().strip("|").split("|"))
        if cells != expected:
            continue
        if not set(lines[index + 1].strip().replace("|", "").replace("-", "").replace(":", "").strip()):
            rows: dict[str, dict[str, str]] = {}
            for row in lines[index + 2 :]:
                if not row.startswith("|"):
                    break
                values = tuple(cell.strip().lower() for cell in row.strip().strip("|").split("|"))
                if len(values) != len(expected):
                    raise AssertionError(f"malformed table row: {row}")
                rows[values[0]] = dict(zip(expected[1:], values[1:], strict=True))
            return rows
    raise AssertionError(f"missing Markdown table with headers: {headers}")


def test_acquisition_methods_preserve_audited_provenance_and_single_z_scope():
    """Catches a methods rewrite that loses audited acquisition facts or permits projections."""
    methods = _read(METHODS_PATH)
    for required in (
        "acquisition date: 2026-08-26", "44 retained stacks", "18 nt", "19 miat-kd",
        "7 dual-omission controls", "hamamatsu orca-fusion #1", "100×/na 1.5",
        "2304×2304 uint16", "1×1 binning", "0.065 µm xy", "39–54 planes",
        "28 stacks at about 0.21 µm z spacing", "16 stacks at about 0.30 µm z spacing",
        "c0 `640 csu` miat-647 at 0.7 s", "c1 `561 csu` qki-568 at 0.7 s",
        "c2 `405 csu` dapi at 0.5 s", "human undifferentiated h9 hescs",
        "nt-versus-miat-kd", "each stack received an independently algorithmically proposed plane, visually reviewed and accepted",
        "zero manual overrides", "one reviewed recorded optical plane per image", "same recorded plane",
        "no maximum-intensity projection (mip) or other projection",
        "narrow physically matched miat projection sensitivity analysis", "separate miat-count sensitivity analysis",
        "seven planes for about 0.21 µm stacks", "five planes for about 0.30 µm stacks",
        "not used for exact-footprint colocalization", "global exact-single-plane endpoints", "representative micrographs",
    ):
        assert required in methods


def test_acquisition_methods_distinguish_confirmed_reported_and_missing_details():
    """Catches invented instrument, laser, or staining detail and incorrect control wording."""
    methods = _read(METHODS_PATH)
    for required in (
        "metadata-confirmed", "experimenter-reported", "missing / not recovered",
        "laser powers were experimenter-reported as held constant", "numeric settings and metadata confirmation are unavailable",
        "miat probe and qki primary antibody were omitted", "missing information checklist",
        "microscope stand/model", "csu scan-head model", "acquisition software", "staining and filter details",
    ):
        assert required in methods
    for forbidden in (
        "universal 210-nm spacing", "metadata-verified laser powers", "both primary antibodies",
        "projection-based colocalization", "exact-footprint mass",
    ):
        assert forbidden not in methods
    for line in methods.splitlines():
        assert not re.search(r"\b(?:metadata[- ]confirmed|metadata[- ]verified)\b.*\blaser powers?\b", line)
        assert not re.search(r"\blaser powers?\b.*\b(?:metadata[- ]confirmed|metadata[- ]verified)\b", line)
        assert not re.search(r"\b(?:known|identified|confirmed)\s+(?:microscope (?:stand|model)|acquisition software)\b", line)
        assert not re.search(r"\b(?:microscope (?:stand|model)|acquisition software)\s+(?:was|is)\s+(?!not recovered|unavailable|missing)\w+", line)
    for sentence in re.split(r"(?<=[.!?])\s+", methods):
        if "|" in sentence:
            continue
        has_projection = bool(re.search(r"\b(?:projection|mip)\b", sentence))
        has_protected_target = bool(re.search(r"\b(?:colocalization|global exact-single-plane endpoints|representative micrographs)\b", sentence))
        positive_use = bool(re.search(r"\b(?:used|entered|included|applied)\b", sentence))
        stated_limit = bool(re.search(r"\b(?:no|not|never)\b.{0,120}\b(?:used|entered|included|applied)\b", sentence))
        assert not (has_projection and has_protected_target and positive_use and not stated_limit)


def test_structured_acquisition_and_analysis_boundaries_are_machine_checkable():
    """Catches a return to free-prose-only acquisition uncertainty or projection boundaries."""
    methods = _read(METHODS_PATH)
    acquisition = _markdown_table(methods, ("acquisition detail", "status", "value / limitation"))
    expected_acquisition_statuses = {
        "microscope stand/model": "not recovered", "csu scan-head model": "not recovered",
        "acquisition software/version": "not recovered", "numeric laser powers": "experimenter-reported only",
    }
    assert {item: acquisition[item]["status"] for item in expected_acquisition_statuses} == expected_acquisition_statuses
    assert acquisition["microscope stand/model"]["value / limitation"] == "no model value recovered"
    assert acquisition["csu scan-head model"]["value / limitation"] == "no model value recovered"
    assert acquisition["acquisition software/version"]["value / limitation"] == "no software or version recovered"
    assert acquisition["numeric laser powers"]["value / limitation"] == "held constant (experimenter-reported); no numeric values recovered"
    boundaries = _markdown_table(methods, ("analysis boundary", "projection used", "definition"))
    assert boundaries["exact-footprint colocalization"]["projection used"] == "no"
    assert boundaries["global exact-single-plane endpoints"]["projection used"] == "no"
    assert boundaries["representative micrographs"]["projection used"] == "no"
    sensitivity = boundaries["miat-count sensitivity analysis"]
    assert sensitivity["projection used"] == "narrow physically matched miat projection"
    assert sensitivity["definition"] == "seven planes for about 0.21 µm stacks; five planes for about 0.30 µm stacks"


def test_plain_language_terms_define_q95_endpoints_and_interpretive_limits():
    """Catches undefined q95 terminology, denominator loss, or an overclaim of direct binding."""
    terms = _read(TERMS_PATH)
    for required in (
        "all miat spots per nucleus", "miat spot-pixel intensity", "overlapping pixels counted once", "q95",
        "greater than 95% of 1,000 same-nucleus keep-n randomized same-shape placements", "among null-usable spots",
        "among all floor-passing spots", "spatial colocalization", "pseudo-binding", "does not establish direct binding",
        "global-versus-associated depletion", "global miat depletion", "q95-associated miat depletion",
    ):
        assert required in terms
    for forbidden in ("exact-footprint mass", "both primary antibodies", "projection-based colocalization"):
        assert forbidden not in terms
    for sentence in re.split(r"(?<=[.!?])\s+", terms):
        states_limit = bool(re.search(r"\b(?:not|does not|cannot|never)\b.{0,32}\b(?:prove|proves|disprove|disproves|establish|establishes|demonstrate|demonstrates|show|shows)\b", sentence))
        positive_binding_claim = bool(re.search(r"\b(?:prove|proves|establish|establishes|demonstrate|demonstrates|show|shows)\b.{0,48}\bdirect binding\b", sentence))
        positive_sponge_claim = bool(re.search(r"\b(?:prove|proves|disprove|disproves|establish|establishes|demonstrate|demonstrates|show|shows)\b.{0,48}\bsponge model\b", sentence))
        assert not (positive_binding_claim and not states_limit)
        assert not (positive_sponge_claim and not states_limit)


def test_structured_interpretation_boundaries_are_machine_checkable():
    """Catches a direct-binding or sponge-model conclusion leaking into reader-facing prose."""
    terms = _read(TERMS_PATH)
    boundaries = _markdown_table(terms, ("interpretive question", "conclusion", "scope"))
    assert boundaries["direct molecular binding"]["conclusion"] == "not established by this assay"
    assert boundaries["miat sponge mechanism"]["conclusion"] == "neither proved nor disproved"
