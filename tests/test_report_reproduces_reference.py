"""Opt-in regression: ``fishsuite report`` reproduces a known reference report.

The one-off builders this subcommand replaces produced a report whose numbers
were reviewed and accepted. This test re-derives them from the same run and
compares, so a future change to the aggregation or the statistics is caught
against a real dataset rather than only against synthetic input.

It is SKIPPED unless both paths are supplied, because the run and the reference
live outside the repository::

    FISHSUITE_TEST_RUN_DIR=<the fishsuite run directory>
    FISHSUITE_TEST_REFERENCE_DIR=<a directory holding per_well.csv and contrasts.csv>

The reference column names come from the builder that produced them, so the
mapping from its endpoint names to the registry's role-based names is spelled
out here rather than guessed.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fishsuite.report.build import build_report

RUN = os.environ.get("FISHSUITE_TEST_RUN_DIR", "")
REF = os.environ.get("FISHSUITE_TEST_REFERENCE_DIR", "")

pytestmark = pytest.mark.skipif(
    not (RUN and REF and Path(RUN).is_dir() and Path(REF).is_dir()),
    reason="set FISHSUITE_TEST_RUN_DIR and FISHSUITE_TEST_REFERENCE_DIR to run")

# reference endpoint name -> registry endpoint name
NAME_MAP = {
    "bin1_spots_per_nucleus": "rna1_spots_per_nucleus",
    "rnaseh2b_rotation_enrichment_at_bin1": "partner_rotation_enrichment_at_rna1",
}
TOLERANCE = 1e-9


@pytest.fixture(scope="module")
def rebuilt(tmp_path_factory):
    out = tmp_path_factory.mktemp("repro")
    return build_report(run_dir=Path(RUN), out_dir=out,
                        groups=["WT=WT_1,WT_2,WT_3", "QKI-KO=KO_1,KO_2,KO_3"],
                        reference="WT", make_figures=False, coloc_panel=False,
                        argv=["fishsuite", "report"])


@pytest.mark.parametrize("old,new", list(NAME_MAP.items()))
def test_well_means_reproduce_the_reference(rebuilt, old, new):
    ref = pd.read_csv(Path(REF) / "per_well.csv")
    got = rebuilt["well"]
    a = (ref[ref["endpoint"] == old].set_index("well_id")["well_mean_of_fov_values"]
         .sort_index())
    b = (got[got["endpoint"] == new].set_index("well_id")["well_mean_of_field_values"]
         .sort_index())
    assert list(a.index) == list(b.index)
    assert np.abs(a.to_numpy() - b.to_numpy()).max() < TOLERANCE


@pytest.mark.parametrize("old,new", list(NAME_MAP.items()))
def test_the_welch_gate_reproduces_the_reference(rebuilt, old, new):
    ref = pd.read_csv(Path(REF) / "contrasts.csv")
    got = rebuilt["contrasts"]
    a = ref[ref["endpoint"] == old].iloc[0]
    b = got[got["endpoint"] == new].iloc[0]
    for col in ("p_welch", "mean_test", "mean_ref", "diff", "hedges_g", "t", "df",
                "p_permutation", "p_tukey_fov"):
        va, vb = float(a[col]), float(b[col])
        assert abs(va - vb) <= TOLERANCE * max(abs(va), 1.0), (
            f"{new}.{col}: reference {va!r} against rebuilt {vb!r}")
