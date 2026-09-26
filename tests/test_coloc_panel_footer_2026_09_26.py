"""coloc_standard_panel figure text on a 2-arm subset (2026-09-26).

Reproduced on 4ea2410 with g2 +Dox vs g2 noDox (4 wells): (1) the footer said
"in 6 wells" (a literal, not a count); (2) p, MDE and 95% CI were still drawn
with 2 v 2 wells; (3) the panel F title ran off the figure's right edge.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest

PANEL_PATH = Path(__file__).resolve().parents[1] / "scripts" / "coloc_standard_panel.py"
REF, TEST, DROPPED = "g2 noDox (control)", "g2 +Dox (MIAT OE)", "g1 noDox"
LONG_WELL = "UD-MIAT-FISH-QKI-IF-g2-no Dox_03"


@pytest.fixture
def panel():
    spec = importlib.util.spec_from_file_location("_panel_footer_2026_09_26", PANEL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _per_nucleus(panel, arms_wells):
    """arms_wells: {arm: [(well_id, n_nuclei), ...]}; plus secondary-only rows
    in their own well, which must never be counted."""
    rng = np.random.default_rng(0)
    cols = [c for c, _ in panel.ALL_ENDPOINTS]
    rows = []
    for arm, wells in arms_wells.items():
        for well, n in wells:
            for fov in range(2):
                for k in range(n // 2 + (n % 2 if fov == 0 else 0)):
                    rows.append(dict(line=arm, condition=arm, well_id=well,
                                     secondary_only=False, image=f"{well}_{fov}.vsi",
                                     nucleus_id=k + 1, n_rna1_nuclear_puncta=3,
                                     **{c: float(rng.uniform(0.02, 0.3)) for c in cols}))
    for k in range(5):
        rows.append(dict(line=REF, condition="sec", well_id="SEC_W", secondary_only=True,
                         image="sec.vsi", nucleus_id=k + 1, n_rna1_nuclear_puncta=0,
                         **{c: 0.1 for c in cols}))
    return pd.DataFrame(rows)


# Full run: 3 arms x 2 wells = 6 wells. The subset keeps REF and TEST only (4 wells).
# The well label "W1" is reused by both kept arms: the composite (arm, well) key counts 2.
FULL = {REF: [("W1", 30), (LONG_WELL, 27)],
        TEST: [("W1", 35), ("W4", 38)],
        DROPPED: [("W5", 20), ("W6", 22)]}


def _subset(panel):
    pn = _per_nucleus(panel, FULL)
    pn = pn[pn["line"] != DROPPED].reset_index(drop=True)
    panel.ARMS[:] = [REF, TEST]
    return pn


def test_plotted_counts_come_from_the_plotted_rows(panel):
    pn = _subset(panel)
    _fov, well = panel.rollup(pn)
    c = panel.plotted_counts(well)
    assert c["n_nuclei"] == 130
    assert c["n_wells"] == 4
    assert c["per_arm"] == {REF: (57, 2), TEST: (73, 2)}
    assert c["inferential"] is False


def test_plotted_counts_three_wells_per_arm_is_inferential(panel):
    pn = _per_nucleus(panel, {REF: [("a", 4), ("b", 4), ("c", 4)],
                              TEST: [("d", 4), ("e", 4), ("f", 4)]})
    panel.ARMS[:] = [REF, TEST]
    _fov, well = panel.rollup(pn)
    c = panel.plotted_counts(well)
    assert (c["n_nuclei"], c["n_wells"], c["inferential"]) == (24, 6, True)


def _ctx(panel, pn):
    return dict(
        seed=0, rna_label="MIAT-640", partner_label="QKI-561", filt="gate: test.",
        costes_fit="tls", costes_pct=84.6, n_bio_nuc=int((~pn["secondary_only"]).sum()),
        arm_n_text="unused", n_sec_nuc=5, n_bio_nuc_with_puncta=130,
        primary_obs="frac_called_coloc_runthr", primary_shuf="frac_called_coloc_shuffle_runthr",
        secondary_obs="frac_called_coloc", secondary_shuf="frac_called_coloc_shuffle",
        primary_thr_label="run batch threshold, every nucleus",
        secondary_thr_label="Costes threshold with fallback", rule_why="for the test",
        fp_area_median=9.0, fp_area_q1=7.0, fp_area_q3=12.0, fp_area_min=3.0,
        fp_area_max=30.0, pair_um=0.3,
        cyto={a: dict(costes_thr_rna1=np.nan, costes_thr_partner=np.nan, pearson_r=0.2,
                      costes_converged=False) for a in panel.ARMS})


def _render_all(panel, monkeypatch, tmp_path, pn):
    figs = {}

    def _capture(fig, out_dir, stem, manifest, *a):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        fb = fig.bbox
        undrawn = set()   # tick labels outside the view interval are never drawn
        for ax in fig.axes:
            for axis in (ax.xaxis, ax.yaxis):
                lo, hi = sorted(axis.get_view_interval())
                for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                    if not lo - 1e-9 <= tick.get_loc() <= hi + 1e-9:
                        undrawn.update({id(tick.label1), id(tick.label2)})
        texts = []
        for t in fig.findobj(matplotlib.text.Text):
            s = t.get_text()
            if not s.strip() or not t.get_visible() or id(t) in undrawn:
                continue
            bb = t.get_window_extent(renderer)
            inside = (bb.x0 >= fb.x0 - 0.5 and bb.x1 <= fb.x1 + 0.5
                      and bb.y0 >= fb.y0 - 0.5 and bb.y1 <= fb.y1 + 0.5)
            texts.append((s, inside))
        figs[stem] = texts
        matplotlib.pyplot.close(fig)

    monkeypatch.setattr(panel, "save", _capture)
    _fov, well = panel.rollup(pn)
    contrasts = panel.contrasts_table(well, pn, panel.mde_hedges_g(n1=2, n2=2),
                                      "frac_called_coloc_runthr", "frac_called_coloc")
    ctx = _ctx(panel, pn)
    rng = np.random.default_rng(1)
    pooled = {a: (rng.uniform(0, 100, 500), rng.uniform(0, 100, 500)) for a in panel.ARMS}
    fields = [dict(zoom_rgb=np.zeros((64, 64, 3)), zoom_puncta=[(10, 10, True), (30, 30, False)],
                   line=REF, well_id=LONG_WELL)]
    panel.set_style()
    panel.fig_pearson_manders(pn, well, ctx, tmp_path, [])
    panel.fig_manders_threshold_sensitivity(pn, well, ctx, tmp_path, [])
    panel.fig_object_fraction(well, contrasts, ctx, tmp_path, [])
    panel.fig_anchor_directions(pn, well, ctx, tmp_path, [])
    panel.fig_composite(pn, well, contrasts, pooled, fields, ctx, tmp_path, [])
    return figs


def test_footers_state_the_plotted_counts(panel, monkeypatch, tmp_path):
    figs = _render_all(panel, monkeypatch, tmp_path, _subset(panel))
    for stem in ("FIG_COLOC_STANDARD", "csp01_pearson_manders_superplot"):
        body = " ".join(s for s, _ in figs[stem]).replace("\n", " ")
        assert "130 biological nuclei in 4 wells" in body, stem
        assert not re.search(r"(in|of) [36] wells|3 vs 3", body), stem
        assert f"57 nuclei / 2 wells {REF}" in body and f"73 nuclei / 2 wells {TEST}" in body, stem


INFERENTIAL = re.compile(r"\bMDE\b|\bCI\b|\bp\s*=|p-value|(?<!\w)\*{1,4}(?!\w)|\bns\b|Welch t")


def test_no_inferential_text_with_fewer_than_three_wells(panel, monkeypatch, tmp_path):
    figs = _render_all(panel, monkeypatch, tmp_path, _subset(panel))
    assert set(figs) >= {"FIG_COLOC_STANDARD", "csp04_object_coloc_fraction",
                         "csp01_pearson_manders_superplot", "csp06_anchor_directions"}
    for stem, texts in figs.items():
        for s, _ in texts:
            hit = INFERENTIAL.search(s.replace("\n", " "))
            assert hit is None, (stem, hit.group(0), s[:200])


def test_every_text_lies_inside_the_figure(panel, monkeypatch, tmp_path):
    figs = _render_all(panel, monkeypatch, tmp_path, _subset(panel))
    for stem, texts in figs.items():
        out = [s[:80] for s, inside in texts if not inside]
        assert not out, (stem, out)
    f_titles = [s for s, _ in figs["FIG_COLOC_STANDARD"] if s.startswith("F  ")]
    assert f_titles and LONG_WELL in f_titles[0].replace("\n", " ")


def test_three_wells_per_arm_keeps_the_test_on_the_figure(panel, monkeypatch, tmp_path):
    pn = _per_nucleus(panel, {REF: [("a", 10), ("b", 10), ("c", 10)],
                              TEST: [("d", 10), ("e", 10), ("f", 10)]})
    panel.ARMS[:] = [REF, TEST]
    figs = _render_all(panel, monkeypatch, tmp_path, pn)
    body = " ".join(s for s, _ in figs["FIG_COLOC_STANDARD"]).replace("\n", " ")
    assert "p = " in body and "MDE" in body and "60 biological nuclei in 6 wells" in body
