"""Punctum size, source-run linkage, and recovering a well from the image name.

The size half is the important one. `spot_fwhm_px` and `spot_diameter_um` ARE
measured per punctum, but by a second moment over a FIXED 9 by 9 pixel crop, so
they saturate: past about 3 pixels of true width the reported value stops
tracking the punctum. The footprint area does not saturate, so it is the size
endpoint. These tests pin both halves of that claim.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fishsuite.core.modes.rna_rna import _measure_spot_diameter_um
from fishsuite.report import aggregate as agg
from fishsuite.report import endpoints as ep
from fishsuite.report import provenance as prov


# ------------------------------------------------- the moment estimator saturates


def _reported_fwhm_px(true_sigma_px: float, voxel_um: float = 0.065) -> float:
    """Run the engine's own estimator on an isolated Gaussian of known width."""
    H = W = 81
    yy, xx = np.indices((H, W)).astype(float)
    img = 1000 * np.exp(-(((yy - 40) ** 2 + (xx - 40) ** 2) / (2 * true_sigma_px ** 2))) + 50
    spots = pd.DataFrame({"y_px": [40.0], "x_px": [40.0]})
    d_um = _measure_spot_diameter_um(img.astype(np.float32), spots, voxel_um)[0]
    return float(d_um) / voxel_um


def test_the_moment_estimator_is_accurate_on_a_narrow_punctum():
    """Below about 3 px true width it is a real measurement, within 10 percent."""
    for sigma in (0.8, 1.2):
        true_fwhm = 2.355 * sigma
        got = _reported_fwhm_px(sigma)
        assert abs(got - true_fwhm) / true_fwhm < 0.10, (sigma, true_fwhm, got)


def test_the_moment_estimator_saturates_on_a_wide_punctum():
    """A 5-fold change in true width must not be reported as one.

    This is the defect behind two arms reporting nearly the same punctum size.
    """
    narrow = _reported_fwhm_px(2.0)     # true full width 4.71 px
    wide = _reported_fwhm_px(10.0)      # true full width 23.55 px, 5 times larger
    assert wide / narrow < 1.30, (
        f"the estimator tracked a 5-fold width change as {wide / narrow:.2f}-fold; "
        "if this now passes at a larger ratio the crop window was widened and the "
        "saturation note in the endpoint registry needs revisiting")
    # And it must be bounded: the fixed 9 by 9 crop caps the attainable moment.
    assert _reported_fwhm_px(20.0) < 6.1


def test_the_saturating_columns_are_marked_descriptive_and_say_why():
    by_name = {e.name: e for e in ep.ENDPOINTS}
    for name in ("rna1_spot_fwhm_px", "rna1_spot_diameter_um"):
        e = by_name[name]
        assert e.descriptive_only, f"{name} must not carry a multiplicity-adjusted gate"
        assert "SATURAT" in e.note.upper()
        assert not e.primary


def test_every_alternative_column_is_the_same_quantity_as_its_primary():
    """An alternative spelling must name the SAME measurement.

    The first draft of these aliases attached an intensity ratio to the nucleus
    area endpoint, which would have reported a ratio as an area on any run that
    lacked the area column. Nothing about that would have looked wrong in the
    output, so it is pinned here.
    """
    STEM = [("nc_ratio", "nc_ratio"), ("nuclear_mean", "nuclear_mean"),
            ("area", "area"), ("paired_fraction", "paired_fraction"),
            ("nn_distance", "nn_distance"), ("manders", "manders"),
            ("rotation", "rotation"), ("enrichment", "enrichment")]
    for e in ep.ENDPOINTS:
        for alt in e.alt_columns:
            for token, _ in STEM:
                assert (token in e.column) == (token in alt), (
                    f"{e.name}: primary {e.column!r} and alternative {alt!r} disagree "
                    f"on {token!r}, so they are not the same measurement")
            # An alternative must differ only in the partner role name.
            assert e.column.replace("protein", "rna2") == alt or \
                   e.column.replace("_protein", "_rna2") == alt, \
                   f"{e.name}: {alt!r} is not the rna_rna spelling of {e.column!r}"


def test_the_footprint_area_is_the_primary_size_endpoint():
    by_name = {e.name: e for e in ep.ENDPOINTS}
    area = by_name["rna1_punctum_footprint_area_um2"]
    assert area.primary and not area.descriptive_only
    assert area.unit == "square micrometres"
    assert "rna1_punctum_equivalent_diameter_um" in by_name


# --------------------------------------------- the footprint conversion is correct


def _run_with_footprints(tmp_path: Path, areas_px, voxel_um=0.065) -> Path:
    run = tmp_path / "RUN_FP"
    run.mkdir(parents=True)
    img, nuc, spot = [], [], []
    for well in ("WT_1", "WT_2", "WT_3", "KO_1", "KO_2", "KO_3"):
        for f in (1, 2):
            image = f"{well}_{f:02d}.vsi"
            img.append({"image": image, "condition": well, "secondary_only": False})
            for k in range(3):
                nuc.append({"image": image, "nucleus_id": k, "n_spots_rna1": 5.0,
                            "nucleus_area_px": 16000.0, "voxel_xy_um": voxel_um})
                for a in areas_px:
                    spot.append({"image": image, "channel": "rna1", "nucleus_id": k,
                                 "in_nucleus": True, "spot_fwhm_px": 4.5,
                                 "spot_diameter_um": 4.5 * voxel_um,
                                 "miat_footprint_area_px": float(a)})
    pd.DataFrame(img).to_csv(run / "per_image_summary.csv", index=False)
    pd.DataFrame(nuc).to_csv(run / "nuclei_metrics.csv", index=False)
    pd.DataFrame(spot).to_csv(run / "spot_metrics.csv", index=False)
    (run / "run_config.json").write_text(json.dumps(
        {"config_resolved": {"channels": {"analysis_mode": "rna_protein"}}}),
        encoding="utf-8")
    return run


def test_footprint_area_converts_with_the_runs_own_voxel_size(tmp_path):
    areas = [16.0, 36.0, 64.0]
    voxel = 0.065
    run = _run_with_footprints(tmp_path, areas, voxel)
    m, _ = agg.parse_groups(["WT=WT_1,WT_2,WT_3", "QKI-KO=KO_1,KO_2,KO_3"])
    data = agg.load_run(run, m, {})
    got_area = float(data["nuclei"]["rna1_punctum_footprint_area_um2"].iloc[0])
    want_area = float(np.mean([a * voxel ** 2 for a in areas]))
    assert got_area == pytest.approx(want_area, rel=1e-12)
    # Equivalent diameter is the mean of the per-punctum diameters, which is NOT
    # the diameter of the mean area.
    got_d = float(data["nuclei"]["rna1_punctum_equivalent_diameter_um"].iloc[0])
    want_d = float(np.mean([2 * math.sqrt(a * voxel ** 2 / math.pi) for a in areas]))
    assert got_d == pytest.approx(want_d, rel=1e-12)
    assert got_d != pytest.approx(2 * math.sqrt(want_area / math.pi), rel=1e-6)


def test_a_run_without_the_footprint_column_reports_it_absent(tmp_path):
    run = _run_with_footprints(tmp_path, [16.0])
    sm = pd.read_csv(run / "spot_metrics.csv").drop(columns=["miat_footprint_area_px"])
    sm.to_csv(run / "spot_metrics.csv", index=False)
    data = agg.load_run(run, {}, {})
    _, absent = ep.resolve(data["nuclei"], data["per_image"])
    assert "rna1_punctum_footprint_area_um2" in absent
    assert "rna1_punctum_equivalent_diameter_um" in absent


# ------------------------------------------------- well recovered from the file name


def _line_conditioned_run(tmp_path: Path) -> Path:
    run = tmp_path / "RUN_LINES"
    run.mkdir()
    img, nuc = [], []
    for line in ("WT", "KO"):
        for well in (1, 2, 3):
            for f in range(3):
                image = f"BIN1_{line}-{well}_{f:02d}.vsi"
                img.append({"image": image, "condition": line, "secondary_only": False})
                nuc.append({"image": image, "nucleus_id": 0, "n_spots_rna1": float(well),
                            "nucleus_area_px": 16000.0, "voxel_xy_um": 0.13})
    img.append({"image": "BIN1_Sec-Only-1_99.vsi", "condition": "Sec-Only",
                "secondary_only": True})
    nuc.append({"image": "BIN1_Sec-Only-1_99.vsi", "nucleus_id": 0, "n_spots_rna1": 0.0,
                "nucleus_area_px": 16000.0, "voxel_xy_um": 0.13})
    pd.DataFrame(img).to_csv(run / "per_image_summary.csv", index=False)
    pd.DataFrame(nuc).to_csv(run / "nuclei_metrics.csv", index=False)
    (run / "run_config.json").write_text(json.dumps({"config_resolved": {"channels": {}}}),
                                         encoding="utf-8")
    return run


def test_without_the_regex_a_line_collapses_into_one_well(tmp_path):
    """The failure this option exists to prevent: three wells become one, and
    there is nothing left to test."""
    run = _line_conditioned_run(tmp_path)
    data = agg.load_run(run, {}, {})
    bio = data["labels"][~data["labels"]["secondary_only"]]
    assert set(bio["well_id"]) == {"WT", "KO"}


def test_the_regex_recovers_three_wells_per_line(tmp_path):
    run = _line_conditioned_run(tmp_path)
    data = agg.load_run(run, {}, {}, well_from_image=r"_((?:WT|KO)-\d)_")
    bio = data["labels"][~data["labels"]["secondary_only"]]
    assert set(bio["well_id"]) == {"WT-1", "WT-2", "WT-3", "KO-1", "KO-2", "KO-3"}
    # The group still comes from the condition, so the lines survive as groups.
    assert set(bio["group"]) == {"WT", "KO"}


def test_a_regex_that_misses_a_biological_image_is_refused(tmp_path):
    run = _line_conditioned_run(tmp_path)
    with pytest.raises(agg.ReportInputError, match="matched no well"):
        agg.load_run(run, {}, {}, well_from_image=r"_(ZZZ-\d)_")


def test_a_regex_with_the_wrong_number_of_groups_is_refused(tmp_path):
    run = _line_conditioned_run(tmp_path)
    with pytest.raises(agg.ReportInputError, match="exactly one capture group"):
        agg.load_run(run, {}, {}, well_from_image=r"_(WT|KO)-(\d)_")


# ----------------------------------------------------------- source-run linkage


def test_source_run_names_the_run_and_copies_its_provenance(tmp_path):
    run = _run_with_footprints(tmp_path / "src", [16.0])
    (run / "thresholds.csv").write_text(
        "image,rna_bigfish_log_threshold\na.vsi,36.0\nb.vsi,36.0\n", encoding="utf-8")
    (run / "versions.txt").write_text("cellpose: 4.2.1.1\nglobal_seed: 0\n",
                                      encoding="utf-8")
    (run / "command.log").write_text("a command\n", encoding="utf-8")
    out = tmp_path / "delivery"
    r = prov.write_source_run(run, out, groups={"WT_1": "WT", "KO_1": "QKI-KO"},
                              group_order=["WT", "QKI-KO"], reference="WT")
    md = Path(r["source_run_md"])
    assert md.name == "SOURCE_RUN.md"
    text = md.read_text(encoding="utf-8")
    assert str(run) in text.splitlines()[2]
    assert "4.2.1.1" in text
    assert "36" in text
    for name in ("run_config.json", "versions.txt", "thresholds.csv", "command.log"):
        assert (out / "provenance" / "source_run" / name).is_file(), name
        assert r["sha256"][name] != "file absent from the run"
    # The link exists in one form or another; a junction may be refused.
    assert (out / prov.JUNCTION_NAME).exists() or (out / f"{prov.JUNCTION_NAME}.txt").is_file()


def test_a_report_inside_the_run_needs_no_junction(tmp_path):
    run = _run_with_footprints(tmp_path, [16.0])
    out = run / "report_x"
    r = prov.write_source_run(run, out)
    assert "inside the run directory" in r["link"]["method"]
    assert not (out / prov.JUNCTION_NAME).exists()


def test_a_missing_provenance_file_is_reported_not_hidden(tmp_path):
    run = _run_with_footprints(tmp_path, [16.0])       # no thresholds.csv written
    out = tmp_path / "delivery2"
    r = prov.write_source_run(run, out)
    assert r["sha256"]["thresholds.csv"] == "file absent from the run"
    assert "thresholds.csv is absent" in Path(r["source_run_md"]).read_text(encoding="utf-8")


# ------------------------------------------------------- Tukey uses the whole design


def test_tukey_is_fitted_over_every_group_not_just_the_pair():
    """Tukey's adjustment is a studentized range over k groups.

    Fitting it on a two-group subset of a five-group design gives a p that is too
    small, so the pair must be read out of a fit over the whole design.
    """
    from fishsuite.report.stats import tukey_two_group

    rng = np.random.default_rng(0)
    full = {n: rng.normal(m, 1.0, 5) for n, m in
            [("WT", 0.0), ("KO", 2.0), ("C16", -1.0), ("C17", -0.5), ("Mix", 0.5)]}
    five = tukey_two_group(full, "KO", "WT")
    two = tukey_two_group({"WT": full["WT"], "KO": full["KO"]}, "KO", "WT")
    assert five["tukey_k"] == 5 and two["tukey_k"] == 2
    assert five["tukey_df"] == 20 and two["tukey_df"] == 8
    # Same difference, stricter p once the other groups are in the fit.
    assert five["tukey_difference"] == pytest.approx(two["tukey_difference"], rel=1e-12)
    assert five["p_tukey_fov"] > two["p_tukey_fov"]


def test_tukey_reads_out_the_requested_pair_not_the_first_one():
    from fishsuite.report.stats import tukey_two_group

    rng = np.random.default_rng(1)
    full = {"A": rng.normal(0, 1, 5), "B": rng.normal(3, 1, 5), "C": rng.normal(9, 1, 5)}
    bc = tukey_two_group(full, "C", "B")
    assert bc["tukey_difference"] == pytest.approx(
        full["C"].mean() - full["B"].mean(), rel=1e-9)


def test_tukey_says_so_when_the_requested_group_has_no_data():
    from fishsuite.report.stats import tukey_two_group

    rng = np.random.default_rng(2)
    r = tukey_two_group({"A": rng.normal(0, 1, 4), "B": rng.normal(1, 1, 4)}, "Z", "A")
    assert np.isnan(r["p_tukey_fov"])
    assert "cannot be read out" in r["tukey_note"]
