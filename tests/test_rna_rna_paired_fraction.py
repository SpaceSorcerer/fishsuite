"""paired_fraction must be a fraction of ONE spot set.

The image-level ``paired_fraction_rna{1,2}_at_X`` columns divide paired spots by
in-cell spots. The numerator used to count every detected spot (floaters off
any cell included) while the denominator counted in-cell spots only, so the
"fraction" could exceed 1. Here two floater spots in each channel sit exactly on
each other, so counting them in the numerator alone pushes the old value to
1.5 (rna1) and 3.0 (rna2).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fishsuite.config.schema import FishsuiteConfig
from fishsuite.core import io as _io
from fishsuite.core.io import ImageWrapper
from fishsuite.core.modes import rna_rna as _rna_rna

H = W = 200
NZ = 2
_CENTERS = [(60, 60), (60, 140), (140, 100)]
_RADIUS = 25

# (y, x): two rna1 spots inside nucleus 1, two floaters far from any nucleus.
_RNA1 = [(55, 60), (65, 60), (10, 10), (190, 10)]
# rna2: one on an in-nucleus rna1 spot, one on each floater.
_RNA2 = [(55, 60), (10, 10), (190, 10)]


class _FakeBio:
    def __init__(self, czyx):
        self._czyx = czyx

    def get_image_data(self, order, *, T=0, C=0):  # noqa: N803
        return self._czyx[C]


def _img():
    from skimage.draw import disk

    dapi = np.full((H, W), 10.0, dtype=np.float32)
    for c in _CENTERS:
        rr, cc = disk(c, _RADIUS, shape=dapi.shape)
        dapi[rr, cc] = 3000.0
    rng = np.random.default_rng(0)
    planes = [dapi, rng.uniform(5, 50, (H, W)), rng.uniform(5, 50, (H, W))]
    czyx = np.stack([np.stack([p] * NZ) for p in planes]).astype(np.float32)
    return ImageWrapper(
        path="synthetic_paired_fraction.tif", bio=_FakeBio(czyx), scene_idx=0,
        shape=(1, 3, NZ, H, W), channel_names=["DAPI", "RNA1", "RNA2"],
        voxel_xy_nm=130.0, voxel_z_nm=300.0, n_channels=3, n_z=NZ,
    )


def _spot_frame(coords):
    return pd.DataFrame({
        "spot_id": np.arange(len(coords)),
        "x_px": [float(x) for _, x in coords],
        "y_px": [float(y) for y, _ in coords],
        "z_slice": 0,
        "intensity_peak": 1000.0,
        "threshold_used": 100.0,
    })


def _cfg():
    cfg = FishsuiteConfig()
    cfg.channels.dapi, cfg.channels.rna, cfg.channels.rna2 = 0, 1, 2
    cfg.channels.analysis_mode = "rna_rna"
    cfg.nuclei.backend = "otsu"
    cfg.nuclei.min_area_px = 120
    cfg.nuclei.max_area_px = 10_000_000
    cfg.nuclei.exclude_border = False
    cfg.z_stack.mode = "maxproj"
    cfg.cytoplasm.enabled = False
    cfg.foci.enabled = True
    cfg.foci.drop_floater_spots = True
    cfg.pixel_coloc.threshold_scope = "per_image"
    return cfg


@pytest.fixture()
def result(monkeypatch):
    img = _img()
    frames = iter([_spot_frame(_RNA1), _spot_frame(_RNA2)])
    monkeypatch.setattr(_io, "read_image", lambda p: img)
    monkeypatch.setattr(_rna_rna._spots, "detect_spots", lambda *a, **k: next(frames))
    cfg = _cfg()
    res = _rna_rna.run_one(Path(img.path), condition="c", sec_only=False, cfg=cfg)
    sfx = _rna_rna._format_pair_um(float(cfg.spot_coloc.pair_distance_um))
    return res, sfx


def test_paired_fraction_is_paired_in_cell_over_total_in_cell(result):
    res, sfx = result
    spots = res.spots
    for ch, key in (("rna1", "rna1"), ("rna2", "rna2")):
        rows = spots[spots["channel"] == ch]
        in_cell = rows["in_nucleus"].astype(bool) | rows["in_cytoplasm"].astype(bool)
        total = int(in_cell.sum())
        paired = int(rows.loc[in_cell, f"paired_at_{sfx}"].astype(int).sum())
        frac = res.per_image[f"paired_fraction_{key}_at_{sfx}"]
        assert total > 0
        assert frac <= 1.0
        assert frac == pytest.approx(paired / total)
        assert res.per_image[f"paired_count_{key}_at_{sfx}"] == paired


def test_synthetic_case_values(result):
    res, sfx = result
    # rna1: 2 in-cell spots, 1 of them paired. rna2: 1 in-cell spot, paired.
    assert res.per_image["total_spots_rna1"] == 2
    assert res.per_image["total_spots_rna2"] == 1
    assert res.per_image[f"paired_fraction_rna1_at_{sfx}"] == pytest.approx(0.5)
    assert res.per_image[f"paired_fraction_rna2_at_{sfx}"] == pytest.approx(1.0)


def test_in_cell_paired_counts_helper():
    df = pd.DataFrame({
        "in_nucleus": [True, True, False, False],
        "in_cytoplasm": [False, False, True, False],
        "paired_at_0p3um": [1, 0, 1, 1],
    })
    assert _rna_rna._in_cell_paired_counts(df, "paired_at_0p3um") == (2, 3)
    assert _rna_rna._in_cell_paired_counts(pd.DataFrame(), "paired_at_0p3um") == (0, 0)
