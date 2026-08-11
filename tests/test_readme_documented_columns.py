"""README column names must name columns that exist, in the table claimed.

The README's "Master CSVs and key columns" section is where a user goes to find
out what a run produced. A name there that no longer exists is worse than an
omission: the reader writes an analysis against it, gets a KeyError or — with a
``.get()`` — a column of NaN, and has no reason to doubt the documentation.

The specific drift caught on 2026-08-10: the per-image nearest-neighbour column
was renamed to ``median_nn_distance_<ch>_um_all_spots_in_frame`` (because the
per-image value pools every in-frame spot rather than averaging nuclei, and the
bare name invited reading it as a per-nucleus mean), while the bare
``median_nn_distance_<ch>_um`` survived per NUCLEUS. The README still listed the
bare name under ``per_image_summary.csv``.

Scoped deliberately: this checks the names the README asserts against the mode
source, not the reverse. A column the README does not mention is not an error.
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

from fishsuite.core import excel_report as _excel_report
from fishsuite.core.modes import rna_only as _rna_only
from fishsuite.core.modes import rna_rna as _rna_rna

_README = Path(__file__).resolve().parents[1] / "README.md"

# Backticked tokens in those bullets that are NOT literal column names. Listed
# explicitly, with the reason, rather than loosening the regex — a looser matcher
# would stop catching the class of drift this file exists for.
_NOT_A_LITERAL_COLUMN = {
    # A mode name in prose ("relabeled protein_* in rna_protein").
    "rna_protein",
    # PARAMETERISED by spot_coloc.pair_distance_um, so the suffix is built at
    # runtime (f"paired_at_{pair_suffix}") and 0p3um is only the default.
    "paired_at_0p3um",
}


@pytest.fixture(scope="module")
def readme() -> str:
    return _README.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def mode_source() -> str:
    """Where a documented column name has to appear as a literal.

    ``excel_report`` is included because it carries the per-column registries
    (name -> type/unit/description) that the workbook builds from. Several
    per-image rollup names are assembled from an f-string prefix in the mode and
    exist as literals only there, so a mode-only corpus would report them missing
    and the sweep would have to be weakened.
    """
    return (
        inspect.getsource(_rna_rna)
        + inspect.getsource(_rna_only)
        + inspect.getsource(_excel_report)
    )


def _bullet(readme: str, csv_name: str) -> str:
    """The README bullet describing one master CSV."""
    for line in readme.splitlines():
        if line.startswith(f"- **`{csv_name}`**"):
            return line
    raise AssertionError(f"README has no bullet for {csv_name}")


def test_the_per_image_bullet_names_the_in_frame_suffix(readme):
    """The rename is the point: the per-image value pools every spot in the frame
    and must not be readable as a per-nucleus mean."""
    line = _bullet(readme, "per_image_summary.csv")
    assert "median_nn_distance_*_um_all_spots_in_frame" in line
    # The bare name may still appear, but only where it is explained as
    # per-nucleus — never as a bare per-image column claim.
    assert "`median_nn_distance_*_um`," not in line


def test_the_documented_nn_columns_exist_in_the_mode_source(mode_source):
    for name in (
        "median_nn_distance_rna1_um_all_spots_in_frame",
        "median_nn_distance_rna2_um_all_spots_in_frame",
        "mean_median_nn_distance_rna1_um_per_nucleus",
    ):
        assert f'"{name}"' in mode_source, f"README documents a missing {name}"


def test_the_bare_name_is_per_nucleus_only(mode_source):
    """It is emitted into the per-NUCLEUS row, and the per-image dict must not
    carry it — otherwise both names would be live and could disagree."""
    assert 'f"median_nn_distance_rna1_um"' in mode_source \
        or '"median_nn_distance_rna1_um"' in mode_source
    per_image_blocks = re.findall(
        r"per_image = \{(.*?)\n {4,8}\}", mode_source, flags=re.S
    )
    assert per_image_blocks, "could not locate a per_image dict to check"
    for block in per_image_blocks:
        assert '"median_nn_distance_rna1_um":' not in block
        assert '"median_nn_distance_rna2_um":' not in block


@pytest.mark.parametrize("csv_name", [
    "per_image_summary.csv",
    "nuclei_metrics.csv",
    "spot_metrics.csv",
])
def test_every_backticked_column_in_the_bullet_exists(readme, mode_source, csv_name):
    """Sweep the whole bullet, not just the one name that drifted.

    Wildcards (`*`), prose and parameterised suffixes are skipped — only literal
    snake_case identifiers are checked, which is the class of claim a reader
    copies verbatim into their own script.
    """
    line = _bullet(readme, csv_name)
    names = {
        n for n in re.findall(r"`([a-z0-9_]+)`", line)
        if "_" in n and not n.endswith("_csv") and n not in _NOT_A_LITERAL_COLUMN
    }
    assert names, f"no literal column names found in the {csv_name} bullet"
    missing = sorted(n for n in names if f'"{n}"' not in mode_source)
    assert not missing, (
        f"README's {csv_name} bullet documents columns absent from the mode "
        f"source: {missing}"
    )
