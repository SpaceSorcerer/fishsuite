"""The post-hoc peak floor, and all-pairs contrasts.

The peak gate reproduces a gated run from an ungated one because fishsuite's own
spot floor is applied AFTER detection. What must not drift: the boundary is at or
above the floor, the nuclear-fraction denominator is in-nucleus plus
in-cytoplasm rather than the row count, and columns a floor makes stale are NAMED
rather than silently carried through at their pre-gate value.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fishsuite.report import aggregate as agg
from fishsuite.report import endpoints as ep
from fishsuite.report import peak_gate as pg


# ------------------------------------------------------------------ parsing


def test_peak_floor_spec_maps_channel_aliases():
    assert pg.parse_peak_floors("rna=1000,rna2=1200") == {"rna1": 1000.0, "rna2": 1200.0}
    assert pg.parse_peak_floors("intron=900, exon=700") == {"rna1": 900.0, "rna2": 700.0}


def test_peak_floor_spec_rejects_a_bad_channel_and_a_bad_number():
    with pytest.raises(pg.PeakGateError, match="not one of"):
        pg.parse_peak_floors("dapi=100")
    with pytest.raises(pg.PeakGateError, match="not a number"):
        pg.parse_peak_floors("rna=high")
    with pytest.raises(pg.PeakGateError, match="CHANNEL=VALUE"):
        pg.parse_peak_floors("rna 1000")


# ------------------------------------------------------------------- gating


def _spots():
    return pd.DataFrame({
        "image": ["a.vsi"] * 6,
        "channel": ["rna1", "rna1", "rna1", "rna2", "rna2", "rna2"],
        "nucleus_id": [1, 1, 1, 1, 1, 1],
        "in_nucleus": [True, True, False, True, False, False],
        "in_cytoplasm": [False, False, True, False, True, True],
        "peak_intensity": [999.0, 1000.0, 1001.0, 1199.0, 1200.0, 1201.0],
    })


def test_a_spot_exactly_at_the_floor_is_kept():
    kept, rec = pg.gate_spots(_spots(), {"rna1": 1000.0, "rna2": 1200.0})
    assert sorted(kept["peak_intensity"]) == [1000.0, 1001.0, 1200.0, 1201.0]
    assert rec["per_channel"]["rna1"]["spots_kept"] == 2
    assert rec["per_channel"]["rna1"]["spots_dropped"] == 1


def test_a_non_positive_floor_leaves_its_channel_untouched():
    kept, rec = pg.gate_spots(_spots(), {"rna1": 0.0, "rna2": float("nan")})
    assert len(kept) == 6
    assert not rec["per_channel"]["rna1"]["applied"]
    assert not rec["per_channel"]["rna2"]["applied"]


def test_a_run_without_a_peak_column_is_refused_not_silently_skipped():
    s = _spots().drop(columns=["peak_intensity"])
    with pytest.raises(pg.PeakGateError, match="none of"):
        pg.gate_spots(s, {"rna1": 1000.0})


def test_the_nuclear_fraction_denominator_is_nucleus_plus_cytoplasm():
    """A spot flagged neither in-nucleus nor in-cytoplasm is excluded from every
    count. Using the row count instead would silently deflate the fraction."""
    spots = pd.DataFrame({
        "image": ["a.vsi"] * 4, "channel": ["rna1"] * 4, "nucleus_id": [1] * 4,
        "in_nucleus": [True, True, False, False],
        "in_cytoplasm": [False, False, True, False],   # the last is neither
        "peak_intensity": [10.0] * 4,
    })
    nuclei = pd.DataFrame({"image": ["a.vsi"], "nucleus_id": [1],
                           "n_spots_rna1": [99.0], "nuclear_spot_count": [99.0],
                           "cyto_spot_count": [99.0], "nuclear_spot_fraction": [0.5],
                           "nucleus_area_px": [1000.0]})
    out, _ = pg.apply_to_nuclei(nuclei, spots, voxel_xy_um=0.1)
    assert float(out["nuclear_spot_count"].iloc[0]) == 2
    assert float(out["cyto_spot_count"].iloc[0]) == 1
    assert float(out["n_spots_rna1"].iloc[0]) == 3          # not 4
    assert float(out["nuclear_spot_fraction"].iloc[0]) == pytest.approx(2 / 3)


def test_columns_a_floor_cannot_re_derive_are_named():
    spots = _spots()
    nuclei = pd.DataFrame({"image": ["a.vsi"], "nucleus_id": [1],
                           "n_spots_rna1": [3.0], "nucleus_area_px": [1000.0],
                           "nuclear_above_floor_intensity_rna1": [5000.0],
                           "manders_rna1_in_rna2": [0.4]})
    _, stale = pg.apply_to_nuclei(nuclei, spots, voxel_xy_um=0.1)
    assert "nuclear_above_floor_intensity_rna1" in stale
    assert "manders_rna1_in_rna2" in stale


# --------------------------------------------------------------- all pairs


def _five_group_run(tmp_path: Path) -> Path:
    run = tmp_path / "RUN_5"
    run.mkdir(parents=True)
    rng = np.random.default_rng(0)
    img, nuc = [], []
    means = {"WT": 2.5, "QKI-KO": 4.2, "Clone16": 0.7, "Clone17": 0.9, "Mix": 1.5}
    for line, mu in means.items():
        for w in (1, 2, 3):
            for f in range(3):
                image = f"X_{line}-{w}_{f:02d}.vsi"
                img.append({"image": image, "condition": line, "secondary_only": False})
                for k in range(5):
                    nuc.append({"image": image, "nucleus_id": k,
                                "n_spots_rna1": float(mu + rng.normal(0, 0.3)),
                                "nuclear_spot_fraction": 0.8,
                                "nucleus_area_px": 16000.0, "voxel_xy_um": 0.13})
    pd.DataFrame(img).to_csv(run / "per_image_summary.csv", index=False)
    pd.DataFrame(nuc).to_csv(run / "nuclei_metrics.csv", index=False)
    (run / "run_config.json").write_text(json.dumps({"config_resolved": {"channels": {}}}),
                                         encoding="utf-8")
    return run


def _contrasts(run: Path, all_pairs: bool):
    data = agg.load_run(run, {}, {},
                        well_from_image=r"_((?:WT|QKI-KO|Clone16|Clone17|Mix)-\d)_")
    order = ["WT", "QKI-KO", "Clone16", "Clone17", "Mix"]
    endpoints, absent = ep.resolve(data["nuclei"], data["per_image"])
    field = agg.per_field_long(data["nuclei"], data["per_image"], endpoints,
                               data["labels"], 5)
    well = agg.per_well_long(field)
    return agg.build_contrasts(well, field, endpoints, absent, order, "WT",
                               ep.channel_labels(data["cfg"]), all_pairs=all_pairs)


def test_five_groups_give_ten_pairs_and_four_against_the_reference(tmp_path):
    run = _five_group_run(tmp_path)
    name = "rna1_spots_per_nucleus"
    ap = _contrasts(run, True)
    ap = ap[ap.endpoint == name]
    vr = _contrasts(run, False)
    vr = vr[vr.endpoint == name]
    assert len(ap) == 10
    assert len(vr) == 4
    pairs = {frozenset((r.test_group, r.reference_group)) for r in ap.itertuples()}
    assert len(pairs) == 10


def test_the_tukey_adjustment_is_over_all_five_groups_for_every_pair(tmp_path):
    run = _five_group_run(tmp_path)
    c = _contrasts(run, True)
    c = c[c.endpoint == "rna1_spots_per_nucleus"]
    assert set(c["tukey_k"].dropna()) == {5.0}
    assert set(c["well_tukey_k"].dropna()) == {5.0}
    # 45 fields minus 5 groups, and 15 wells minus 5 groups.
    assert set(c["tukey_df"].dropna()) == {40.0}
    assert set(c["well_tukey_df"].dropna()) == {10.0}


def test_the_well_level_and_field_level_tukey_are_reported_separately(tmp_path):
    """They are different statistics on different units and must not be conflated:
    the field-level fit treats a technical replicate as independent."""
    run = _five_group_run(tmp_path)
    c = _contrasts(run, True)
    c = c[c.endpoint == "rna1_spots_per_nucleus"].dropna(
        subset=["p_tukey_fov", "well_p_tukey_fov"])
    assert len(c) > 0
    assert not np.allclose(c["p_tukey_fov"], c["well_p_tukey_fov"])
    # Same estimand, so the differences agree even though the p-values do not.
    assert np.allclose(c["tukey_difference"], c["well_tukey_difference"], rtol=1e-9)
