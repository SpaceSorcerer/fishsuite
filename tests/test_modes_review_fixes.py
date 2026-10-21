"""Small contracts for the analysis-mode review fixes."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fishsuite.config.schema import FishsuiteConfig
from fishsuite.core.modes import get_mode, if_intensity, rna_only, rna_rna


@pytest.mark.parametrize("name", ["ab_ab", "protein_only", "pub_images"])
def test_unimplemented_modes_are_registered_but_refuse_to_run(name):
    run = get_mode(name)
    with pytest.raises(NotImplementedError, match=name):
        run("x.tif", condition="c", sec_only=False, cfg=FishsuiteConfig())


def test_radial_helper_returns_per_ring_null_vectors_on_request():
    rng_a, rng_b = np.random.default_rng(0), np.random.default_rng(0)
    partner = np.random.default_rng(1).uniform(0, 100, (40, 40))
    stencils = rna_rna._annulus_stencils([2.0, 4.0])
    nys, nxs = np.where(np.ones((40, 40), bool))
    cy, cx = np.array([10, 20]), np.array([10, 20])
    plain = rna_rna._radial_profile_for_nucleus(partner, cy, cx, nys, nxs, stencils, 50, rng_a)
    rows, nulls = rna_rna._radial_profile_for_nucleus(
        partner, cy, cx, nys, nxs, stencils, 50, rng_b, return_null_stats=True,
    )
    assert rows == plain
    assert len(nulls) == len(stencils)
    for (obs, nmean, nsd, nsp), vec in zip(rows, nulls):
        assert vec.shape == (50,)
        assert nmean == pytest.approx(float(vec.mean()))
        assert nsd == pytest.approx(float(vec.std(ddof=1)))
    empty_rows, empty_nulls = rna_rna._radial_profile_for_nucleus(
        partner, cy[:0], cx[:0], nys, nxs, stencils, 50, rng_b, return_null_stats=True,
    )
    assert all(r[3] == 0 for r in empty_rows)
    assert all(v.size == 0 for v in empty_nulls)


def test_plate_map_rejects_unknown_labels():
    labels = if_intensity._GENOTYPE_LABELS
    assert if_intensity._plate_label(" wt ", labels, "genotype", 3, "p.csv") == "WT"
    assert if_intensity._plate_label("KO", labels, "genotype", 3, "p.csv") == "KO"
    arms = if_intensity._ARM_LABELS
    assert if_intensity._plate_label("Primary+Secondary", arms, "arm", 3, "p.csv") == "primary"
    assert if_intensity._plate_label("Secondary only", arms, "arm", 3, "p.csv") == "seconly"
    with pytest.raises(ValueError, match="well 3"):
        if_intensity._plate_label("Het", labels, "genotype", 3, "p.csv")
    with pytest.raises(ValueError, match="unrecognised staining arm"):
        if_intensity._plate_label("isotype", arms, "staining arm", 3, "p.csv")


def _cfg():
    cfg = FishsuiteConfig()
    cfg.channels.dapi, cfg.channels.rna, cfg.channels.rna2 = 0, 1, 2
    cfg.channels.one_indexed = False
    cfg.z_stack.mode = "maxproj"
    cfg.nuclei.exclude_border = False
    cfg.cytoplasm.enabled = False
    cfg.foci.enabled = True
    cfg.foci.bigfish_voxel_size_nm = 130.0
    return cfg


def _mock_io(monkeypatch, module):
    img = SimpleNamespace(n_channels=3, n_z=1, voxel_xy_nm=130.0, voxel_z_nm=230.0)
    monkeypatch.setattr(module._io, "read_image", lambda path: img)
    monkeypatch.setattr(module._io, "extract_channel",
                        lambda *a, **k: np.ones((16, 16), dtype=np.float32))
    monkeypatch.setattr(module._seg, "segment_nuclei",
                        lambda image, **k: np.zeros_like(image, dtype=np.int32))


class _Stop(Exception):
    pass


def test_rna_only_uses_the_rna_channel_threshold_override(monkeypatch):
    cfg = _cfg()
    cfg.foci.threshold_override = 11.0
    cfg.foci.rna_overrides.threshold_override = 42.0
    _mock_io(monkeypatch, rna_only)
    seen = {}

    def detect(img, **kw):
        seen.update(kw)
        raise _Stop

    monkeypatch.setattr(rna_only._spots, "detect_spots", detect)
    with pytest.raises(_Stop):
        rna_only.run_one(Path("a.tif"), condition="c", sec_only=False, cfg=cfg)
    assert seen["threshold"] == 42.0


@pytest.mark.parametrize("module", [rna_only, rna_rna])
def test_detector_errors_propagate(monkeypatch, module):
    _mock_io(monkeypatch, module)

    def boom(*a, **k):
        raise RuntimeError("detector exploded")

    monkeypatch.setattr(module._spots, "detect_spots", boom)
    with pytest.raises(RuntimeError, match="detector exploded"):
        module.run_one("a.tif", condition="c", sec_only=False, cfg=_cfg())


def test_rna_rna_prescan_masks_dust_specks_like_run_one(monkeypatch):
    cfg = _cfg()
    cfg.foci.mask_dust_specks_min_size_px = 5
    _mock_io(monkeypatch, rna_rna)
    monkeypatch.setattr(rna_rna._seg, "segment_nuclei",
                        lambda image, **k: np.ones_like(image, dtype=np.int32))
    calls = []

    def fake_mask(cfg_, *planes):
        calls.append(len(planes))
        return tuple(p * 0 + 7 for p in planes)

    monkeypatch.setattr(rna_rna._zplanes, "mask_dust_specks_planes", fake_mask)
    v1, v2, _labels = rna_rna.collect_nuclear_rna_pixels(Path("a.tif"), cfg=cfg)
    assert calls == [2]
    assert np.all(v1 == 7) and np.all(v2 == 7)


def test_rna_rna_honours_rna_z_window_override(monkeypatch):
    cfg = _cfg()
    cfg.z_stack.file_overrides = {"a.tif": {"rna_start_slice": 2, "rna_end_slice": 3}}
    img = SimpleNamespace(n_channels=3, n_z=4)
    calls = []

    def extract(image, ch, **kw):
        calls.append((ch, kw))
        return np.ones((4, 4), dtype=np.float32)

    monkeypatch.setattr(rna_rna._io, "extract_channel", extract)
    rna_rna.prepare_segmentation_planes(img, Path("a.tif"), cfg)
    by_ch = dict(calls)
    assert by_ch[0] == {"z_mode": "maxproj", "z_start": None, "z_end": None}
    assert by_ch[1] == {"z_mode": "maxproj", "z_start": 2, "z_end": 3}
    assert by_ch[2] == {"z_mode": "maxproj", "z_start": 2, "z_end": 3}


def test_rna_only_honours_rna_autofocus_anchor(monkeypatch):
    cfg = _cfg()
    cfg.z_stack.mode = "autofocus"
    cfg.z_stack.autofocus_channel = "rna"
    img = SimpleNamespace(n_channels=3, n_z=6)
    monkeypatch.setattr(rna_only._io, "resolve_autofocus_plane",
                        lambda *a, **k: (5, "rna", {}))
    monkeypatch.setattr(rna_only._io, "extract_channel_at_z",
                        lambda image, ch, z_1indexed: np.full((4, 4), ch, np.float32))
    out = rna_only.prepare_segmentation_planes(img, Path("a.tif"), cfg)
    assert out["dapi_autofocus_z"] == 5
    assert out["_z_af_extra"]["z_autofocus_channel_used"] == "rna"
    assert "rna2_2d" not in out


def test_rna_only_detects_on_secondary_only_images(monkeypatch):
    cfg = _cfg()
    _mock_io(monkeypatch, rna_only)
    called = []

    def detect(img, **kw):
        called.append(True)
        raise _Stop

    monkeypatch.setattr(rna_only._spots, "detect_spots", detect)
    with pytest.raises(_Stop):
        rna_only.run_one(Path("a.tif"), condition="c", sec_only=True, cfg=cfg)
    assert called
