"""Holm membership of absolute intensities, superplot brackets and star provenance text."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from fishsuite.report import figures as fig
from fishsuite.report import endpoints as ep
from fishsuite.report import stats as st
from fishsuite.report.build import build_report
from test_report_end_to_end import _synthetic_run


def test_default_registry_keeps_absolute_intensity_descriptive():
    absolute = [e for e in (*ep.ENDPOINTS, *ep.a3_endpoints([])) if e.absolute_intensity]
    assert absolute
    for e in absolute:
        assert e.descriptive_only or e.excluded_from_holm, e.name
        amended = ep.amend_endpoint(e)
        assert not amended.descriptive_only and not amended.excluded_from_holm


def test_default_build_leaves_absolute_intensity_descriptive(tmp_path):
    run = _synthetic_run(tmp_path)
    r = build_report(run_dir=run, groups=["WT=WT_1,WT_2,WT_3", "QKI-KO=KO_1,KO_2,KO_3"],
                     reference="WT", stamp="t", make_figures=False, coloc_panel=False)
    c = r["contrasts"]
    rows = c[c["absolute_intensity"].astype(bool)]
    assert "protein_nuclear_mean" in set(rows["endpoint"])
    assert not rows["in_holm_family"].any()
    assert rows["p_welch_holm_within_family"].isna().all()
    readme = pd.read_excel(r["xlsx"], sheet_name="Read me", header=1)
    detail = dict(zip(readme["Item"], readme["Detail"]))
    assert "descriptive only" in detail["Absolute intensity"]
    assert "absolute-intensity endpoints" in detail["Multiplicity"]
    assert detail["Stars on figures"].startswith(fig.star_source("superplot"))
    readout = (r["out_dir"] / "READOUT.md").read_text(encoding="utf-8")
    assert fig.star_source("superplot") in readout


def _three_group_inputs():
    order = ["A", "B", "C"]
    well = pd.DataFrame([dict(endpoint="x", group=g, well_id=f"{g}{i}",
                              well_mean_of_field_values=v + i)
                         for g, v in zip(order, (1.0, 5.0, 9.0)) for i in range(3)])
    rows = []
    for test, ref, p in (("B", "A", 0.2), ("C", "A", 0.0005), ("C", "B", 0.04)):
        rows.append(dict(endpoint="x", test_group=test, reference_group=ref, p_welch=p,
                         p_welch_holm_within_family=p, hedges_g=1.0,
                         mde_hedges_g_at_family_alpha=np.nan, descriptive_only=False,
                         usability_filter="none"))
    return order, well, pd.DataFrame(rows)


def _ctx(tmp_path, order, reference):
    return fig.FigureContext(tmp_path, {}, None, order, reference, .05, {}, {})


def test_superplot_bracket_uses_the_contrast_against_the_reference(tmp_path):
    order, well, contrasts = _three_group_inputs()
    # The C-vs-B row is listed first, so a test_group-only lookup would pick it.
    contrasts = contrasts.iloc[[2, 0, 1]].reset_index(drop=True)
    canvas, ax = plt.subplots()
    foot = fig.draw_superplot(ax, _ctx(tmp_path, order, "A"), "x", well, pd.DataFrame(),
                              pd.DataFrame(), contrasts, "x", None)
    plt.close(canvas)
    assert "C vs A: raw Welch p 0.0005 " in foot
    assert "p 0.04" not in foot
    assert sorted(t.get_text() for t in ax.texts if "*" in t.get_text() or t.get_text() == "ns") == ["***", "ns"]


def test_superplot_writes_descriptive_no_test_instead_of_stars(tmp_path):
    order, well, contrasts = _three_group_inputs()
    contrasts["descriptive_only"] = True
    canvas, ax = plt.subplots()
    foot = fig.draw_superplot(ax, _ctx(tmp_path, order, "A"), "x", well, pd.DataFrame(),
                              pd.DataFrame(), contrasts, "x", None)
    plt.close(canvas)
    texts = [t.get_text() for t in ax.texts]
    assert texts.count("descriptive, no test") == 2
    assert not any(t in ("*", "**", "***", "****", "ns") for t in texts)
    assert "descriptive, no test" in foot


def test_star_provenance_follows_plot_style(tmp_path):
    simple = fig.FigureContext(tmp_path, {}, None, ["A", "B"], "A", .05, {}, {},
                               plot_style="replicate-simple")
    plain = _ctx(tmp_path, ["A", "B"], "A")
    assert "mixed-model" in fig.star_source("replicate-simple")
    assert "mixed-model" not in fig.star_source("superplot")
    assert "mixed model" in simple.banner and "mixed model" in simple.filt
    assert "mixed model" in simple.footer()
    assert "mixed model" not in plain.banner and "Welch t on well means" in plain.banner


def test_holm_family_is_keyed_by_comparison(tmp_path):
    from fishsuite.report import aggregate as agg
    order, well, _ = _three_group_inputs()
    well = pd.concat([well.assign(endpoint=name) for name in ("rna1_spots_per_nucleus",
                                                               "rna1_nuclear_spots_per_nucleus")])
    well["n_nuclei_in_well"] = 10
    well["family"] = "detection"
    eps = [e for e in ep.ENDPOINTS if e.name in ("rna1_spots_per_nucleus", "rna1_nuclear_spots_per_nucleus")]
    c = agg.build_contrasts(well, pd.DataFrame(columns=["endpoint", "group", "field_value"]),
                            eps, [], order, "A", {}, all_pairs=True)
    member = c[c["in_holm_family"]]
    assert (member["holm_family_size"] == 2).all()
    assert member.groupby(["test_group", "reference_group"]).size().eq(2).all()


@pytest.mark.parametrize("n1,n2,floor", [(3, 3, 0.1), (2, 3, 0.1), (2, 4, 1 / 15)])
def test_permutation_floor_matches_design(n1, n2, floor):
    out = st.exact_permutation(np.arange(n1, dtype=float), np.arange(n2, dtype=float) + 10)
    assert out["perm_arithmetic_floor"] == pytest.approx(floor)
    assert out["p_permutation"] == pytest.approx(floor)


def test_coupling_permutation_note_uses_observed_wells():
    from fishsuite.report.coupling_stats import permutation_note
    assert permutation_note(2, 2).endswith("6 allocations; smallest two-sided p = 0.333")
    assert permutation_note(3, 3).endswith("20 allocations; smallest two-sided p = 0.1")
    assert permutation_note(2, 3).endswith("10 allocations; smallest two-sided p = 0.1")
