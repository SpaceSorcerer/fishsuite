"""The translation null runs standalone (2026-08-10).

``foci.compute_partner_translation_null`` used to ALSO require
``compute_partner_rotation_null``, so a config that asked for translation alone
silently produced nothing at all — no columns, no warning, no clue. It now has
its own gate.

Un-nesting can only ADD output to a config that previously emitted none, so no
existing run changes: rotation+translation together must emit exactly the key set
it always did. The reliability caveat now travels as an emitted COLUMN rather
than living only in the config docstring, because the number and the warning
about the number belong in the same file.

Synthetic arrays only — no image fixtures.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import gaussian_filter

from fishsuite.config.schema import FishsuiteConfig
from fishsuite.core import io as _io
from fishsuite.core.io import ImageWrapper
from fishsuite.core.modes import rna_rna as _rna_rna

DAPI_C, RNA_C, PART_C = 0, 1, 2
NZ = 4
H = W = 200

_TR_NUC_COLS = (
    "rna2_translation_enrichment_at_rna1_spots",
    "rna2_translation_null_z_at_rna1_spots",
    "translation_null_usable",
)
_TR_IMG_COLS = (
    "rna2_pooled_translation_enrichment_at_rna1_spots",
    "rna2_pooled_translation_null_z_at_rna1_spots",
    "n_nuclei_partner_translation_null",
    "translation_null_caveat",
)
_ROT_NUC_COLS = (
    "rna2_rotation_enrichment_at_rna1_spots",
    "rna2_rotation_null_z_at_rna1_spots",
    "rna2_rotation_null_p_at_rna1_spots",
    "rna2_rotation_assoc_fraction_at_rna1_spots",
    "rotation_median_retention",
    "rotation_null_usable",
)


class _FakeBio:
    def __init__(self, czyx: np.ndarray):
        self._czyx = czyx

    def get_image_data(self, order: str, *, T: int = 0, C: int = 0):  # noqa: N803
        assert order == "ZYX"
        return self._czyx[C]


def _nuclei_centers():
    return [(70, 70), (70, 130), (130, 100)]


def _dapi_plane():
    from skimage.draw import disk

    img = np.random.default_rng(11).uniform(0.0, 20.0, (H, W)).astype(np.float32)
    for (cy, cx) in _nuclei_centers():
        rr, cc = disk((cy, cx), 28, shape=img.shape)
        img[rr, cc] += 3000.0
    return img


def _rna_spot_plane():
    img = np.random.default_rng(22).uniform(2.0, 8.0, (H, W)).astype(np.float32)
    blob = np.zeros_like(img)
    rng = np.random.default_rng(33)
    for (cy, cx) in _nuclei_centers():
        for k in range(6):
            ang = 2 * np.pi * k / 6
            y = int(cy + 12 * np.sin(ang))
            x = int(cx + 12 * np.cos(ang))
            blob[y, x] += float(rng.uniform(3000.0, 6000.0))
    return img + gaussian_filter(blob, 1.1)


def _partner_plane():
    from skimage.draw import disk

    img = np.random.default_rng(44).uniform(2.0, 8.0, (H, W)).astype(np.float32)
    for (cy, cx) in _nuclei_centers():
        rr, cc = disk((cy, cx), 28, shape=img.shape)
        img[rr, cc] = 800.0
    return img


def _czyx() -> np.ndarray:
    planes = [_dapi_plane(), _rna_spot_plane(), _partner_plane()]
    return np.stack(
        [np.stack([p] * NZ, axis=0) for p in planes], axis=0
    ).astype(np.float32)


@pytest.fixture()
def fake_img() -> ImageWrapper:
    return ImageWrapper(
        path="synthetic_translation_standalone.tif",
        bio=_FakeBio(_czyx()),
        scene_idx=0,
        shape=(1, 3, NZ, H, W),
        channel_names=["DAPI", "RNA", "PART"],
        voxel_xy_nm=130.0,
        voxel_z_nm=300.0,
        n_channels=3,
        n_z=NZ,
    )


def _base_cfg() -> FishsuiteConfig:
    cfg = FishsuiteConfig()
    cfg.channels.dapi = DAPI_C
    cfg.channels.rna = RNA_C
    cfg.channels.rna2 = PART_C
    cfg.channels.analysis_mode = "rna_rna"
    cfg.nuclei.backend = "otsu"
    cfg.nuclei.min_area_px = 120
    cfg.nuclei.max_area_px = 10_000_000
    cfg.nuclei.exclude_border = True
    cfg.nuclei.border_margin_px = 3
    cfg.z_stack.mode = "maxproj"
    cfg.cytoplasm.enabled = True
    cfg.foci.enabled = True
    cfg.foci.backend = "bigfish"
    cfg.foci.threshold_multiplier = 1.0
    cfg.foci.drop_floater_spots = False
    cfg.pixel_coloc.threshold_scope = "per_image"
    cfg.foci.compute_partner_intensity = True
    cfg.foci.partner_rotation_n = 40
    cfg.foci.partner_null_n = 40
    return cfg


def _run(cfg, img, monkeypatch):
    monkeypatch.setattr(_io, "read_image", lambda p: img)
    return _rna_rna.run_one(
        Path(img.path), condition="cond", sec_only=False, cfg=cfg,
    )


def test_translation_default_off_emits_nothing(fake_img, monkeypatch):
    res = _run(_base_cfg(), fake_img, monkeypatch)
    for col in _TR_NUC_COLS:
        assert col not in res.nuclei.columns
    for key in _TR_IMG_COLS:
        assert key not in res.per_image


def test_translation_runs_without_rotation(fake_img, monkeypatch):
    """The whole point: translation alone now produces its columns. Before this
    change the same config produced nothing and said nothing."""
    cfg = _base_cfg()
    cfg.foci.compute_partner_translation_null = True
    assert cfg.foci.compute_partner_rotation_null is False
    res = _run(cfg, fake_img, monkeypatch)
    for col in _TR_NUC_COLS:
        assert col in res.nuclei.columns
    for key in _TR_IMG_COLS:
        assert key in res.per_image


def test_translation_standalone_does_not_emit_rotation_columns(fake_img, monkeypatch):
    """Un-nesting must not smuggle rotation in as a side effect."""
    cfg = _base_cfg()
    cfg.foci.compute_partner_translation_null = True
    res = _run(cfg, fake_img, monkeypatch)
    for col in _ROT_NUC_COLS:
        assert col not in res.nuclei.columns
    assert "rna2_pooled_rotation_enrichment_at_rna1_spots" not in res.per_image


def test_rotation_alone_still_emits_no_translation_columns(fake_img, monkeypatch):
    cfg = _base_cfg()
    cfg.foci.compute_partner_rotation_null = True
    res = _run(cfg, fake_img, monkeypatch)
    for col in _ROT_NUC_COLS:
        assert col in res.nuclei.columns
    for col in _TR_NUC_COLS:
        assert col not in res.nuclei.columns


def test_nested_form_keeps_its_full_key_set(fake_img, monkeypatch):
    """Preserving today's behaviour: both flags on emits both families."""
    cfg = _base_cfg()
    cfg.foci.compute_partner_rotation_null = True
    cfg.foci.compute_partner_translation_null = True
    res = _run(cfg, fake_img, monkeypatch)
    for col in _ROT_NUC_COLS + _TR_NUC_COLS:
        assert col in res.nuclei.columns
    for key in _TR_IMG_COLS:
        assert key in res.per_image


def test_nested_form_warns_that_the_gate_is_no_longer_nested(fake_img, monkeypatch, capsys):
    cfg = _base_cfg()
    cfg.foci.compute_partner_rotation_null = True
    cfg.foci.compute_partner_translation_null = True
    _run(cfg, fake_img, monkeypatch)
    out = capsys.readouterr().out
    assert "no longer requires" in out
    assert "UNRELIABLE" in out


def test_standalone_translation_matches_the_nested_values(fake_img, monkeypatch):
    """Translation's RNG stream is derived independently of rotation's, so
    turning rotation off must not perturb a single translation number."""
    cfg_nested = _base_cfg()
    cfg_nested.foci.compute_partner_rotation_null = True
    cfg_nested.foci.compute_partner_translation_null = True
    cfg_alone = _base_cfg()
    cfg_alone.foci.compute_partner_translation_null = True

    nested = _run(cfg_nested, fake_img, monkeypatch)
    alone = _run(cfg_alone, fake_img, monkeypatch)
    col = "rna2_translation_enrichment_at_rna1_spots"
    np.testing.assert_allclose(
        pd.to_numeric(nested.nuclei[col], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(alone.nuclei[col], errors="coerce").to_numpy(dtype=float),
        rtol=0, atol=0, equal_nan=True,
    )


def test_translation_does_not_perturb_the_rotation_draws(fake_img, monkeypatch):
    """The byte-identical pooled-null contract, in the other direction."""
    cfg_rot = _base_cfg()
    cfg_rot.foci.compute_partner_rotation_null = True
    cfg_both = _base_cfg()
    cfg_both.foci.compute_partner_rotation_null = True
    cfg_both.foci.compute_partner_translation_null = True

    col = "rna2_rotation_enrichment_at_rna1_spots"
    np.testing.assert_allclose(
        pd.to_numeric(_run(cfg_rot, fake_img, monkeypatch).nuclei[col],
                      errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(_run(cfg_both, fake_img, monkeypatch).nuclei[col],
                      errors="coerce").to_numpy(dtype=float),
        rtol=0, atol=0, equal_nan=True,
    )


def test_caveat_is_an_emitted_column_naming_its_own_denominator(fake_img, monkeypatch):
    """The reliability limit must be readable from the output, and must point at
    the column that lets a reader check it."""
    cfg = _base_cfg()
    cfg.foci.compute_partner_translation_null = True
    res = _run(cfg, fake_img, monkeypatch)
    caveat = str(res.per_image["translation_null_caveat"])
    assert "UNRELIABLE" in caveat
    assert "dense" in caveat and "space-filling" in caveat
    assert "n_nuclei_partner_translation_null" in caveat
    assert "n_nuclei_partner_translation_null" in res.per_image


def test_caveat_column_is_documented_in_the_excel_legend():
    from fishsuite.core.excel_report import PER_IMAGE_GLOSSARY

    assert "translation_null_caveat" in PER_IMAGE_GLOSSARY
    assert "n_nuclei_partner_translation_null" in (
        PER_IMAGE_GLOSSARY["translation_null_caveat"][2]
    )


def test_translation_still_requires_partner_intensity(fake_img, monkeypatch):
    """It reads the partner-intensity machinery, so the existing prerequisite on
    compute_partner_intensity is unchanged."""
    cfg = _base_cfg()
    cfg.foci.compute_partner_intensity = False
    cfg.foci.compute_partner_translation_null = True
    res = _run(cfg, fake_img, monkeypatch)
    for col in _TR_NUC_COLS:
        assert col not in res.nuclei.columns
