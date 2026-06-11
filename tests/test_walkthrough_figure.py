"""walkthrough_figure — publication PIPELINE-WALKTHROUGH composite (Brian, 2026-06-07).

A single labeled 8-panel figure that "shows how the MIAT x QKI RNA-FISH pipeline
works" using the pipeline's OWN per-step microscope PNGs (A-F, H) plus ONE newly
rendered panel (G): the run's thresholded QKI (magenta) intensity field with the
detected MIAT spots overlaid as small yellow open markers.

The figure must REGENERATE for the d4/d8/d15 timepoint runs, so it is a reusable
``build_walkthrough_figure`` + CLI that locates panels by the ``image_key`` prefix
and the known step suffixes, self-skipping (with a warning) any missing panel.

These tests prove, in TDD order:
  (1) the PURE new-panel logic on synthetic arrays — ``threshold_qki_plane`` masks
      sub-threshold pixels and keeps supra ones (deterministic), and
      ``render_panel_g`` places the yellow markers EXACTLY at the supplied
      (x_px, y_px) spot coordinates and masks the sub-threshold QKI field;
  (2) the panel-resolution helpers (default image_key prefers the OE g2-Dox image;
      panel-prefix -> per_image VSI row mapping);
  (3) a SMOKE test of ``build_walkthrough_figure`` against a tiny synthetic run-dir
      (a few step PNGs + minimal spot_metrics / per_image_summary / run_config +
      a monkeypatched ``io.read_image`` for an availability-guarded fake VSI):
      a non-empty 600-DPI PNG with the expected 8 panel axes, panel G rendered;
  (4) a SELF-SKIP test: a missing composable panel AND no VSI staging -> a warning
      per missing panel and NO crash; the PNG is still produced with 8 axes.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from fishsuite.config.schema import FishsuiteConfig
from fishsuite.core import io as _io
from fishsuite.core.io import ImageWrapper

from fishsuite.core.walkthrough_figure import (
    threshold_qki_plane,
    render_panel_g,
    _resolve_image_key,
    _match_image_row,
    _build_walkthrough_fig,
    build_walkthrough_figure,
)


# ===========================================================================
# (1) PURE new-panel logic — synthetic arrays, known answers
# ===========================================================================
def test_threshold_qki_plane_masks_subthreshold_keeps_supra():
    """Sub-threshold pixels -> 0.0; supra-threshold pixels kept verbatim."""
    qki = np.array([[10.0, 100.0, 250.0],
                    [99.9, 100.1, 5000.0],
                    [0.0, 50.0, 100.0]], dtype=np.float64)
    out = threshold_qki_plane(qki, 100.0)
    assert out.shape == qki.shape
    sub = qki < 100.0
    assert np.all(out[sub] == 0.0)                 # below floor -> zeroed
    assert np.all(out[~sub] == qki[~sub])          # at/above floor -> verbatim
    # the exact boundary (== threshold) is KEPT
    assert out[0, 1] == 100.0
    assert out[1, 0] == 0.0                         # 99.9 < 100 -> zeroed


def test_threshold_qki_plane_deterministic_and_nondestructive():
    qki = np.random.default_rng(0).uniform(0, 500, (40, 40))
    qki_before = qki.copy()
    a = threshold_qki_plane(qki, 123.0)
    b = threshold_qki_plane(qki, 123.0)
    np.testing.assert_array_equal(a, b)                 # deterministic
    np.testing.assert_array_equal(qki, qki_before)      # input not mutated (copy)


def test_render_panel_g_places_markers_at_spot_coords():
    """The overlaid MIAT markers sit at EXACTLY the supplied (x_px, y_px)."""
    qki = np.full((64, 64), 200.0)
    spots = np.array([[10.0, 20.0], [30.0, 5.0], [50.5, 40.25]], dtype=float)
    fig, ax = plt.subplots()
    try:
        res = render_panel_g(ax, qki, threshold=100.0, spots_xy=spots,
                             vmin=100.0, vmax=400.0, pixel_um=0.13)
        offsets = np.asarray(res["scatter"].get_offsets())
        np.testing.assert_array_equal(offsets, spots)   # (x, y) preserved exactly
        assert res["n_spots"] == 3
    finally:
        plt.close(fig)


def test_render_panel_g_masks_subthreshold_field():
    """render_panel_g displays QKI as a masked array: sub-threshold pixels are
    masked (not drawn), supra-threshold pixels remain."""
    qki = np.array([[10.0, 300.0], [400.0, 50.0]], dtype=float)
    fig, ax = plt.subplots()
    try:
        res = render_panel_g(ax, qki, threshold=100.0,
                             spots_xy=np.empty((0, 2)), vmin=100.0, vmax=500.0)
        disp = res["masked"]
        assert np.ma.is_masked(disp)
        expected_mask = qki < 100.0
        np.testing.assert_array_equal(np.ma.getmaskarray(disp), expected_mask)
        assert res["n_spots"] == 0
    finally:
        plt.close(fig)


def test_render_panel_g_empty_spots_is_graceful():
    qki = np.full((20, 20), 150.0)
    fig, ax = plt.subplots()
    try:
        res = render_panel_g(ax, qki, threshold=100.0, spots_xy=np.empty((0, 2)))
        assert res["n_spots"] == 0
        assert np.asarray(res["scatter"].get_offsets()).reshape(-1, 2).shape[0] == 0
    finally:
        plt.close(fig)


# ===========================================================================
# (2) panel-resolution helpers
# ===========================================================================
def _write_png(path: Path, h=24, w=24, value=120):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((h, w, 3), value, dtype=np.uint8)
    Image.fromarray(arr).save(path)


def test_resolve_image_key_default_prefers_OE(tmp_path):
    """With no image_key, the OE g2-Dox image (its prefix contains MIAT_OE) is
    chosen over the non-targeting Nog control."""
    wk = tmp_path / "pipeline_walkthrough"
    _write_png(wk / "Nog_noDox__Nog_07__step01_DAPI_raw.png")
    _write_png(wk / "g2_wDox_(MIAT_OE)__g2-Dox_01__step01_DAPI_raw.png")
    key = _resolve_image_key(tmp_path, None)
    assert key == "g2_wDox_(MIAT_OE)__g2-Dox_01"


def test_resolve_image_key_explicit_passthrough(tmp_path):
    (tmp_path / "pipeline_walkthrough").mkdir(parents=True)
    assert _resolve_image_key(tmp_path, "anything__given") == "anything__given"


def test_match_image_row_maps_prefix_to_vsi():
    df = pd.DataFrame({
        "image": ["UD-MIAT-FISH-QKI-IF-g2-no Dox_03.vsi",
                  "UD-MIAT-FISH-QKI-IF-g2-Dox_01.vsi"],
        "condition": ["g2 noDox (control)", "g2 wDox (MIAT-OE)"],
        "protein_threshold_value": [2500.0, 2937.0],
    })
    row = _match_image_row(df, "g2_wDox_(MIAT_OE)__g2-Dox_01")
    assert row is not None
    assert row["image"] == "UD-MIAT-FISH-QKI-IF-g2-Dox_01.vsi"
    assert float(row["protein_threshold_value"]) == 2937.0


# ===========================================================================
# (3) + (4) SMOKE: build_walkthrough_figure on a synthetic run-dir
# ===========================================================================
IMAGE_KEY = "g2_wDox_(MIAT_OE)__g2-Dox_01"
VSI_NAME = "UD-MIAT-FISH-QKI-IF-g2-Dox_01.vsi"
CONDITION = "g2 wDox (MIAT-OE)"
DAPI_C, RNA_C, AB_C = 0, 1, 2
NZ, H, W = 3, 64, 64


class _FakeBio:
    def __init__(self, czyx):
        self._czyx = czyx

    def get_image_data(self, order, *, T=0, C=0):  # noqa: N803
        assert order == "ZYX"
        return self._czyx[C]


def _fake_czyx():
    """3 channels x NZ x H x W. The QKI channel (2) is a horizontal ramp 0..~500
    so a 100 threshold masks the left edge and keeps the right (visible field)."""
    rng = np.random.default_rng(7)
    dapi = rng.uniform(0, 50, (H, W)).astype(np.float32)
    dapi[20:44, 20:44] += 2000.0                      # a bright nucleus for autofocus
    rna = rng.uniform(0, 30, (H, W)).astype(np.float32)
    ramp = np.linspace(0, 500, W, dtype=np.float32)[None, :].repeat(H, axis=0)
    planes = [dapi, rna, ramp]
    return np.stack([np.stack([p] * NZ, axis=0) for p in planes], axis=0).astype(np.float32)


def _fake_img():
    return ImageWrapper(
        path=VSI_NAME, bio=_FakeBio(_fake_czyx()), scene_idx=0,
        shape=(1, 3, NZ, H, W), channel_names=["DAPI", "MIAT-640", "QKI-561"],
        voxel_xy_nm=130.0, voxel_z_nm=300.0, n_channels=3, n_z=NZ,
    )


def _make_run_config(staging_dir: Path) -> dict:
    cfg = FishsuiteConfig()
    cfg.channels.analysis_mode = "rna_protein"
    cfg.channels.dapi = DAPI_C
    cfg.channels.rna = RNA_C
    cfg.channels.antibody = AB_C
    cfg.channels.rna_label = "MIAT-640"
    cfg.channels.antibody_label = "QKI-561"
    cfg.channels.rna_lut = "yellow"
    cfg.channels.antibody_lut = "magenta"
    cfg.z_stack.mode = "autofocus"
    cfg.z_stack.start_slice = 1
    cfg.z_stack.end_slice = NZ
    cfg.z_stack.autofocus_intensity_weighted = True
    cfg.output.manual_antibody_min = 100.0
    cfg.output.manual_antibody_max = 500.0
    return {"config_resolved": cfg.model_dump(mode="json"),
            "input_dir": str(staging_dir)}


def _build_synthetic_run(tmp_path: Path, *, include_step07=True, with_staging=True):
    run = tmp_path / "run"
    wk = run / "pipeline_walkthrough"
    pub = run / "publication_images"
    wk.mkdir(parents=True)
    pub.mkdir(parents=True)

    # composable per-step PNGs (A,B,D,F + C,E,H) + distractors to test disambiguation
    _write_png(wk / f"{IMAGE_KEY}__step01_DAPI_raw.png")                       # A
    _write_png(wk / f"{IMAGE_KEY}__step03_nuclei_outlines_on_DAPI.png")        # B
    if include_step07:
        _write_png(wk / f"{IMAGE_KEY}__step07_MIAT_640_spots_on_DAPI.png")     # D
    _write_png(wk / f"{IMAGE_KEY}__step06_MIAT_640_threshold_on_signal.png")   # distractor
    _write_png(wk / f"{IMAGE_KEY}__step06_QKI_561_threshold_on_signal.png")    # F
    _write_png(wk / f"{IMAGE_KEY}__step11_merge_all.png")                      # H fallback
    _write_png(pub / f"{IMAGE_KEY}__DAPI_blue.png")                            # distractor
    _write_png(pub / f"{IMAGE_KEY}__MIAT_640_yellow.png")                      # C
    _write_png(pub / f"{IMAGE_KEY}__QKI_561_magenta.png")                      # E
    _write_png(pub / f"{IMAGE_KEY}__merge_MIAT_640_QKI_561.png")               # H

    # minimal MIAT (rna1) spot table
    spots = pd.DataFrame({
        "image": [VSI_NAME] * 4,
        "condition": [CONDITION] * 4,
        "channel": ["rna1"] * 4,
        "x_px": [12, 40, 55, 30],
        "y_px": [10, 33, 50, 25],
        "nucleus_id": [1, 1, 2, 1],
    })
    spots.to_csv(run / "spot_metrics.csv", index=False)

    per_image = pd.DataFrame({
        "image": [VSI_NAME],
        "condition": [CONDITION],
        "protein_threshold_value": [150.0],
        "protein_thresh_floor": [100.0],
    })
    per_image.to_csv(run / "per_image_summary.csv", index=False)

    staging = tmp_path / "staging"
    staging.mkdir()
    if with_staging:
        # an availability-guarded fake VSI file (read via monkeypatched read_image)
        (staging / VSI_NAME).write_bytes(b"FAKE-VSI")

    (run / "run_config.json").write_text(json.dumps(_make_run_config(staging)))
    return run, staging


def test_build_walkthrough_figure_full_synthetic(tmp_path, monkeypatch):
    monkeypatch.setattr(_io, "read_image", lambda p, *a, **k: _fake_img())
    run, staging = _build_synthetic_run(tmp_path)

    out = build_walkthrough_figure(run, staging_dir=staging, image_key=IMAGE_KEY)
    out = Path(out)
    # default location: <run>/figures/07_coloc/79_pipeline_walkthrough.png
    assert out == run / "figures" / "07_coloc" / "79_pipeline_walkthrough.png"
    assert out.exists() and out.stat().st_size > 5_000

    # 600-DPI PNG
    with Image.open(out) as im:
        dpi = im.info.get("dpi", (0, 0))
    assert round(dpi[0]) == 600

    # 8 panel axes, panel G rendered (not skipped)
    fig, statuses = _build_walkthrough_fig(run, staging_dir=staging, image_key=IMAGE_KEY)
    try:
        assert len(statuses) == 8
        assert sum(1 for s in statuses if s["letter"] in "ABCDEFGH") == 8
        g = next(s for s in statuses if s["letter"] == "G")
        assert g["status"] == "rendered"
        # every composable panel (A-F except G, plus H) resolved to a file
        non_g_ok = [s for s in statuses if s["letter"] != "G"]
        assert all(s["status"] == "ok" for s in non_g_ok)
        # panel axes count (no colorbar axes added)
        assert len(fig.axes) == 8
    finally:
        plt.close(fig)


def test_build_walkthrough_self_skips_missing_panel(tmp_path, monkeypatch):
    """A missing composable panel (step07/D) AND no VSI staging (panel G) -> a
    warning per skip and NO crash; the PNG is still produced with 8 axes."""
    monkeypatch.setattr(_io, "read_image", lambda p, *a, **k: _fake_img())
    run, staging = _build_synthetic_run(tmp_path, include_step07=False,
                                        with_staging=False)

    with pytest.warns(UserWarning):
        out = build_walkthrough_figure(run, staging_dir=staging, image_key=IMAGE_KEY)
    assert Path(out).exists() and Path(out).stat().st_size > 5_000

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig, statuses = _build_walkthrough_fig(run, staging_dir=staging,
                                               image_key=IMAGE_KEY)
    try:
        assert len(statuses) == 8
        d = next(s for s in statuses if s["letter"] == "D")
        assert d["status"] == "missing"
        g = next(s for s in statuses if s["letter"] == "G")
        assert g["status"] == "skipped"
        assert len(fig.axes) == 8           # placeholder axes still present
    finally:
        plt.close(fig)


def test_cli_main_runs_on_synthetic(tmp_path, monkeypatch):
    monkeypatch.setattr(_io, "read_image", lambda p, *a, **k: _fake_img())
    run, staging = _build_synthetic_run(tmp_path)
    from fishsuite.core import walkthrough_figure as wf
    rc = wf.main(["--run-dir", str(run), "--staging", str(staging),
                  "--image", IMAGE_KEY])
    assert rc == 0
    assert (run / "figures" / "07_coloc" / "79_pipeline_walkthrough.png").exists()
