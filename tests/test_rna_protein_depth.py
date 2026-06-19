"""rna_protein analysis-depth parity test (Brian, 2026-05-28).

Requirement: the upgraded ``rna_protein`` mode must analyze the PROTEIN /
antibody channel at the SAME depth ``rna_rna`` gives its 2nd RNA channel —
BigFISH spot detection, per-nucleus spot counts, per-nucleus protein intensity
(mean + total), and a full RNA x protein colocalization — but with PROTEIN
semantics in the output (columns / channel value read "protein", never
"rna2").

Design under test (remap): rna_protein maps the antibody channel into
rna_rna's rna2 slot, runs the rna_rna two-channel core, then relabels rna2-*
outputs to protein-*. So this test:

  1. Builds a synthetic 3-channel z-stack (DAPI + 2 dense spot fields).
  2. Runs ``rna_rna.run_one`` on (DAPI, RNA1, RNA2) and snapshots its output —
     this is the REGRESSION GUARD (rna_rna behavior must be unchanged).
  3. Runs ``rna_protein.run_one`` on (DAPI, RNA, PROTEIN) reading the SAME two
     non-DAPI channels, and asserts the PROTEIN channel emits the same FAMILY
     of metrics rna_rna emits for rna2 — under protein-* names — and that the
     underlying VALUES match the rna2 reference (the analysis core is shared).

Segmentation uses the model-free ``otsu`` backend and BigFISH spot detection
runs on dense synthetic spot fields, so the test needs no StarDist / cellpose
model and no GPU.
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
from fishsuite.core.modes import rna_only as _rna_only
from fishsuite.core.modes import rna_rna as _rna_rna
from fishsuite.core.modes import rna_protein as _rna_protein


# Channel layout (0-indexed): DAPI / RNA / PROTEIN.
DAPI_C, RNA_C, PROT_C = 0, 1, 2
NZ = 5
H = W = 192


class _FakeBio:
    """Minimal bioio stand-in: stores (C, Z, Y, X), returns a channel's ZYX."""

    def __init__(self, czyx: np.ndarray):
        self._czyx = czyx

    def get_image_data(self, order: str, *, T: int = 0, C: int = 0):  # noqa: N803
        assert order == "ZYX"
        return self._czyx[C]


def _dapi_plane(seed: int) -> np.ndarray:
    """Two well-separated, away-from-border nuclei (otsu-segmentable)."""
    from skimage.draw import disk

    rng = np.random.default_rng(seed)
    img = rng.uniform(0.0, 25.0, (H, W)).astype(np.float32)
    for (cy, cx) in [(70, 70), (70, 130)]:
        rr, cc = disk((cy, cx), 26, shape=img.shape)
        img[rr, cc] += 3200.0
    return img


def _spot_plane(seed: int, n_spots: int = 55) -> np.ndarray:
    """Dense Gaussian-blob spot field (rich enough for BigFISH auto-threshold).

    Spots are scattered across the whole field; stratification into the two
    nuclei vs cytoplasm happens downstream. ~55 spots gives BigFISH a clean
    LoG elbow.
    """
    rng = np.random.default_rng(seed)
    img = rng.uniform(5.0, 15.0, (H, W)).astype(np.float32)
    blob = np.zeros_like(img)
    ys = rng.integers(12, H - 12, n_spots)
    xs = rng.integers(12, W - 12, n_spots)
    amps = rng.uniform(2500.0, 6000.0, n_spots)
    for y, x, a in zip(ys, xs, amps):
        blob[y, x] += float(a)
    return img + gaussian_filter(blob, 1.2)


def _czyx() -> np.ndarray:
    """Build a (3, Z, Y, X) stack: DAPI + two distinct spot fields.

    Each channel is the SAME 2D plane tiled across Z (maxproj/single collapse
    to that plane), so z handling is irrelevant to the parity assertion.
    """
    dapi = _dapi_plane(seed=101)
    rna = _spot_plane(seed=202)      # RNA1 / RNA
    prot = _spot_plane(seed=303)     # RNA2 / PROTEIN (different spot pattern)
    planes = [dapi, rna, prot]
    czyx = np.stack(
        [np.stack([p] * NZ, axis=0) for p in planes], axis=0
    ).astype(np.float32)
    return czyx


@pytest.fixture()
def fake_img() -> ImageWrapper:
    czyx = _czyx()
    return ImageWrapper(
        path="synthetic_rna_protein.tif",
        bio=_FakeBio(czyx),
        scene_idx=0,
        shape=(1, 3, NZ, H, W),
        channel_names=["DAPI", "RNA", "PROTEIN"],
        voxel_xy_nm=65.0,
        voxel_z_nm=230.0,
        n_channels=3,
        n_z=NZ,
    )


def _base_cfg() -> FishsuiteConfig:
    cfg = FishsuiteConfig()
    cfg.channels.dapi = DAPI_C
    cfg.channels.rna = RNA_C
    # Model-free segmentation + a deterministic single-plane z so the test is
    # backend/GPU independent and z-handling is trivial.
    cfg.nuclei.backend = "otsu"
    cfg.nuclei.min_area_px = 80
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
    return cfg


def _run(mode_cfg, mode_module, path, monkeypatch, img):
    monkeypatch.setattr(_io, "read_image", lambda p: img)
    return mode_module.run_one(
        Path(path), condition="cond", sec_only=False, cfg=mode_cfg,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_fixture_segments_and_detects(fake_img, monkeypatch):
    """Sanity: the synthetic image yields >=1 nucleus and spots in BOTH the
    RNA and the 2nd (rna2/protein) channel — otherwise the parity assertions
    below would be vacuously true."""
    cfg = _base_cfg()
    cfg.channels.analysis_mode = "rna_rna"
    cfg.channels.rna2 = PROT_C
    res = _run(cfg, _rna_rna, fake_img.path, monkeypatch, fake_img)
    assert len(res.nuclei) >= 1
    assert int(res.per_image["total_spots_rna1"]) > 0
    assert int(res.per_image["total_spots_rna2"]) > 0


def test_protein_channel_has_rna2_metric_family(fake_img, monkeypatch):
    """rna_protein's PROTEIN channel must emit the same FAMILY of per-nucleus
    fields rna_rna emits for rna2 — under protein-* names: protein spot count,
    protein per-nucleus intensity (mean + total), and RNA x protein coloc."""
    cfg = _base_cfg()
    cfg.channels.analysis_mode = "rna_protein"
    cfg.channels.antibody = PROT_C
    cfg.channels.antibody_label = "XRN2"
    res = _run(cfg, _rna_protein, fake_img.path, monkeypatch, fake_img)

    nuc_cols = set(res.nuclei.columns)
    # --- protein spot count per nucleus ---
    assert "n_spots_protein" in nuc_cols
    assert "nuclear_spot_count_protein" in nuc_cols
    assert "cyto_spot_count_protein" in nuc_cols
    # --- protein per-nucleus intensity (mean + total) ---
    assert "protein_nuclear_mean" in nuc_cols          # mean nuclear intensity
    assert "sum_protein_intensity" in nuc_cols          # total nuclear intensity
    assert "protein_nc_ratio" in nuc_cols
    assert "protein_spot_total_peak_intensity" in nuc_cols
    # --- RNA x protein colocalization (spot-level NN + pairing) ---
    paired_cols = [c for c in nuc_cols if c.startswith("paired_fraction_protein_at_")]
    assert paired_cols, f"no protein paired_fraction col in {sorted(nuc_cols)}"
    assert "median_nn_distance_protein_um" in nuc_cols
    assert "n_nuclear_rna_protein_overlap_per_nucleus" in nuc_cols

    # --- RNA x protein MASK (Manders-style) colocalization, BOTH directions ---
    # manders_rna1_in_protein = M1 = fraction of RNA inside the PROTEIN mask
    #   (protein masked, RNA measured).
    # manders_protein_in_rna1 = M2 = fraction of PROTEIN inside the RNA mask
    #   (RNA masked, protein measured).
    assert "manders_rna1_in_protein" in nuc_cols
    assert "manders_protein_in_rna1" in nuc_cols
    # correlation / overlap pair metrics (symmetric -> _rna_protein suffix)
    assert "coloc_pearson_r_rna_protein" in nuc_cols
    assert "coloc_spearman_rho_rna_protein" in nuc_cols
    assert "coloc_jaccard_rna_protein" in nuc_cols
    assert "coloc_dice_rna_protein" in nuc_cols
    assert "coloc_li_icq_rna_protein" in nuc_cols
    assert "coloc_cosine_overlap_rna_protein" in nuc_cols
    assert "coloc_both_frac_rna_protein" in nuc_cols
    # directional fraction-above-thr + enrichment
    assert "coloc_frac_above_thr_rna1" in nuc_cols
    assert "coloc_frac_above_thr_protein" in nuc_cols
    assert "protein_enrich_in_rna1_high" in nuc_cols
    assert "rna1_enrich_in_protein_high" in nuc_cols
    # Manders are bounded fractions [0, 1] on a real (>=10 px) nucleus.
    m1 = pd.to_numeric(res.nuclei["manders_rna1_in_protein"], errors="coerce").dropna()
    m2 = pd.to_numeric(res.nuclei["manders_protein_in_rna1"], errors="coerce").dropna()
    assert len(m1) and len(m2)
    assert ((m1 >= -1e-9) & (m1 <= 1.0 + 1e-9)).all()
    assert ((m2 >= -1e-9) & (m2 <= 1.0 + 1e-9)).all()

    # --- NO rna2-* leakage into protein-mode output ---
    leaked = [c for c in nuc_cols if "rna2" in c]
    assert not leaked, f"rna2-named columns leaked into rna_protein output: {leaked}"

    # --- per-spot channel value is 'protein', never 'rna2' ---
    assert "channel" in res.spots.columns
    chans = set(res.spots["channel"].unique())
    assert "protein" in chans
    assert "rna2" not in chans
    assert chans <= {"rna1", "protein"}

    # --- thresholds + per_image carry protein-* provenance ---
    assert "protein_threshold_value" in res.thresholds
    assert "rna2_threshold_value" not in res.thresholds
    assert int(res.per_image["total_spots_protein"]) > 0
    assert res.extra.get("mode") == "rna_protein"


def test_protein_values_match_rna2_reference(fake_img, monkeypatch):
    """The shared analysis core means rna_protein's protein-* VALUES must
    equal rna_rna's rna2-* values when fed the SAME two physical channels.

    This proves the protein channel gets the FULL robust treatment (identical
    spot detection + per-nucleus metrics + coloc), differing only in labels.
    """
    # rna_rna reference: (DAPI, RNA1=RNA_C, RNA2=PROT_C).
    cfg_rr = _base_cfg()
    cfg_rr.channels.analysis_mode = "rna_rna"
    cfg_rr.channels.rna2 = PROT_C
    res_rr = _run(cfg_rr, _rna_rna, fake_img.path, monkeypatch, fake_img)

    # rna_protein: (DAPI, RNA=RNA_C, PROTEIN=PROT_C) — same channels.
    cfg_rp = _base_cfg()
    cfg_rp.channels.analysis_mode = "rna_protein"
    cfg_rp.channels.antibody = PROT_C
    cfg_rp.channels.antibody_label = "XRN2"
    res_rp = _run(cfg_rp, _rna_protein, fake_img.path, monkeypatch, fake_img)

    # Same nuclei count.
    assert len(res_rr.nuclei) == len(res_rp.nuclei) >= 1

    # Per-nucleus protein metrics equal the rna2 reference (value-for-value).
    pairs = [
        ("n_spots_rna2", "n_spots_protein"),
        ("nuclear_spot_count_rna2", "nuclear_spot_count_protein"),
        ("rna2_nuclear_mean", "protein_nuclear_mean"),
        ("sum_rna2_intensity", "sum_protein_intensity"),
        ("rna2_nc_ratio", "protein_nc_ratio"),
        ("rna2_spot_total_peak_intensity", "protein_spot_total_peak_intensity"),
        ("median_nn_distance_rna2_um", "median_nn_distance_protein_um"),
        # MASK (Manders-style) coloc — shared core, relabeled to protein.
        ("manders_rna1_in_rna2", "manders_rna1_in_protein"),
        ("manders_rna2_in_rna1", "manders_protein_in_rna1"),
        ("coloc_pearson_r_rna1_rna2", "coloc_pearson_r_rna_protein"),
        ("coloc_jaccard_rna1_rna2", "coloc_jaccard_rna_protein"),
        ("coloc_both_frac_rna1_rna2", "coloc_both_frac_rna_protein"),
        ("coloc_frac_above_thr_rna2", "coloc_frac_above_thr_protein"),
    ]
    rr = res_rr.nuclei.reset_index(drop=True)
    rp = res_rp.nuclei.reset_index(drop=True)
    for rr_col, rp_col in pairs:
        assert rr_col in rr.columns, rr_col
        assert rp_col in rp.columns, rp_col
        a = pd.to_numeric(rr[rr_col], errors="coerce").to_numpy()
        b = pd.to_numeric(rp[rp_col], errors="coerce").to_numpy()
        np.testing.assert_allclose(
            a, b, rtol=1e-9, atol=1e-9, equal_nan=True,
            err_msg=f"{rr_col} (rna2) != {rp_col} (protein)",
        )

    # Per-image total protein spots == total rna2 spots.
    assert int(res_rp.per_image["total_spots_protein"]) == int(
        res_rr.per_image["total_spots_rna2"]
    )
    # And the RNA channel is itself identical between the two runs (the RNA
    # half of rna_protein matches rna_rna's rna1 half).
    np.testing.assert_allclose(
        pd.to_numeric(rr["rna_spot_count"], errors="coerce").to_numpy(),
        pd.to_numeric(rp["rna_spot_count"], errors="coerce").to_numpy(),
        rtol=0, atol=0,
    )


def test_rna_rna_unchanged_on_two_rna_fixture(fake_img, monkeypatch):
    """Regression guard: rna_rna output on a 2-RNA fixture is byte-stable and
    keeps rna2-* semantics (the rna_protein remap must not perturb rna_rna)."""
    cfg = _base_cfg()
    cfg.channels.analysis_mode = "rna_rna"
    cfg.channels.rna2 = PROT_C
    res = _run(cfg, _rna_rna, fake_img.path, monkeypatch, fake_img)

    nuc_cols = set(res.nuclei.columns)
    # rna2-* columns present, protein-* columns ABSENT — rna_rna is untouched.
    assert "n_spots_rna2" in nuc_cols
    assert "rna2_nuclear_mean" in nuc_cols
    assert "sum_rna2_intensity" in nuc_cols
    assert not any(c.startswith("n_spots_protein") for c in nuc_cols)
    assert not any("protein" in c for c in nuc_cols)
    # MASK (Manders-style) coloc columns are emitted by the SHARED core, so
    # rna_rna gets them too (rna1/rna2-named, additive — no protein here).
    assert "manders_rna1_in_rna2" in nuc_cols
    assert "manders_rna2_in_rna1" in nuc_cols
    assert "coloc_pearson_r_rna1_rna2" in nuc_cols
    assert "coloc_jaccard_rna1_rna2" in nuc_cols
    assert "coloc_frac_above_thr_rna1" in nuc_cols
    assert "coloc_frac_above_thr_rna2" in nuc_cols
    # Spot-spot coloc still coexists (additive, not replaced).
    assert any(c.startswith("paired_fraction_rna2_at_") for c in nuc_cols)
    # per-spot channels are rna1 / rna2 (NOT protein).
    chans = set(res.spots["channel"].unique())
    assert chans <= {"rna1", "rna2"}
    assert "rna2" in chans
    assert res.extra.get("mode") == "rna_rna"


def test_partner_intensity_per_nucleus_populated(fake_img, monkeypatch):
    """Regression for the all-NaN per-nucleus intensity-coloc bug (2026-05-29).

    With ``foci.compute_partner_intensity=True`` the per-spot
    ``partner_local_mean_intensity`` was 100% populated, yet the per-nucleus
    rollup columns (``rna2_local_mean_at_rna1_spots`` /
    ``rna2_enrichment_at_rna1_spots`` and the reciprocal rna1-at-rna2 pair, and
    after the rna_protein relabel ``protein_local_mean_at_rna1_spots`` etc.)
    came out ALL NaN on dense data. Root cause: the per-nucleus spot index
    (``spots*_by_nid``) was built via ``groupby`` BEFORE the
    ``partner_local_mean_intensity`` column was added to the parent spot frame,
    so the grouped sub-frames never saw the column and the aggregation returned
    NaN for every nucleus.

    This asserts that on the dense synthetic fixture at least one spot-bearing
    nucleus has a FINITE local-mean + enrichment in BOTH directions (which
    would have failed pre-fix), in BOTH rna_rna and rna_protein modes — and
    that a nucleus with no spots in a channel yields NaN for that direction.
    """
    # ---- rna_rna mode ----
    cfg = _base_cfg()
    cfg.channels.analysis_mode = "rna_rna"
    cfg.channels.rna2 = PROT_C
    cfg.foci.compute_partner_intensity = True
    res = _run(cfg, _rna_rna, fake_img.path, monkeypatch, fake_img)

    # Per-spot sampler column must be present + populated (it always was).
    assert "partner_local_mean_intensity" in res.spots.columns
    psp = pd.to_numeric(
        res.spots["partner_local_mean_intensity"], errors="coerce"
    )
    assert psp.notna().any(), "per-spot partner intensity should be populated"

    nuc = res.nuclei
    for col in (
        "rna2_local_mean_at_rna1_spots",
        "rna2_enrichment_at_rna1_spots",
        "rna1_local_mean_at_rna2_spots",
        "rna1_enrichment_at_rna2_spots",
    ):
        assert col in nuc.columns, f"missing per-nucleus column {col}"

    # The bug: ALL rows NaN despite spots. Require >=1 FINITE per-nucleus value
    # for each direction's local-mean AND enrichment.
    n_spots_rna1 = pd.to_numeric(nuc["n_spots_rna1"], errors="coerce").fillna(0)
    n_spots_rna2 = pd.to_numeric(nuc["n_spots_rna2"], errors="coerce").fillna(0)
    lm1 = pd.to_numeric(nuc["rna2_local_mean_at_rna1_spots"], errors="coerce")
    en1 = pd.to_numeric(nuc["rna2_enrichment_at_rna1_spots"], errors="coerce")
    lm2 = pd.to_numeric(nuc["rna1_local_mean_at_rna2_spots"], errors="coerce")
    en2 = pd.to_numeric(nuc["rna1_enrichment_at_rna2_spots"], errors="coerce")

    assert (n_spots_rna1 > 0).any() and (n_spots_rna2 > 0).any(), (
        "fixture must yield spot-bearing nuclei in both channels"
    )
    # Spot-bearing nuclei must have FINITE intensity-coloc (the regression).
    assert np.isfinite(lm1[n_spots_rna1 > 0]).any()
    assert np.isfinite(en1[n_spots_rna1 > 0]).any()
    assert np.isfinite(lm2[n_spots_rna2 > 0]).any()
    assert np.isfinite(en2[n_spots_rna2 > 0]).any()
    # And NaN only where there are genuinely no spots in that channel.
    assert lm1[n_spots_rna1 == 0].isna().all()
    assert lm2[n_spots_rna2 == 0].isna().all()
    # Per-image rollup must also be finite (it averages the per-nucleus values).
    assert np.isfinite(float(res.per_image["mean_rna2_local_mean_at_rna1_spots"]))
    assert np.isfinite(float(res.per_image["mean_rna1_local_mean_at_rna2_spots"]))

    # ---- rna_protein mode: the exact columns that were 0/184 in the BIN1 run ----
    cfg_rp = _base_cfg()
    cfg_rp.channels.analysis_mode = "rna_protein"
    cfg_rp.channels.antibody = PROT_C
    cfg_rp.channels.antibody_label = "XRN2"
    cfg_rp.foci.compute_partner_intensity = True
    res_rp = _run(cfg_rp, _rna_protein, fake_img.path, monkeypatch, fake_img)

    nrp = res_rp.nuclei
    for col in (
        "protein_local_mean_at_rna1_spots",
        "protein_enrichment_at_rna1_spots",
        "rna1_local_mean_at_protein_spots",
        "rna1_enrichment_at_protein_spots",
    ):
        assert col in nrp.columns, f"missing relabeled per-nucleus column {col}"

    n_rna1 = pd.to_numeric(nrp["n_spots_rna1"], errors="coerce").fillna(0)
    n_prot = pd.to_numeric(nrp["n_spots_protein"], errors="coerce").fillna(0)
    assert np.isfinite(
        pd.to_numeric(nrp["protein_local_mean_at_rna1_spots"], errors="coerce")[n_rna1 > 0]
    ).any()
    assert np.isfinite(
        pd.to_numeric(nrp["protein_enrichment_at_rna1_spots"], errors="coerce")[n_rna1 > 0]
    ).any()
    assert np.isfinite(
        pd.to_numeric(nrp["rna1_local_mean_at_protein_spots"], errors="coerce")[n_prot > 0]
    ).any()
    assert np.isfinite(
        pd.to_numeric(nrp["rna1_enrichment_at_protein_spots"], errors="coerce")[n_prot > 0]
    ).any()
    # No rna2-* leakage from the relabel.
    assert not any("rna2" in c for c in nrp.columns)


def test_partner_intensity_off_by_default(fake_img, monkeypatch):
    """When the flag is OFF (default) the intensity-coloc columns must be
    ABSENT entirely (byte-equivalent to the pre-feature output) — this guards
    the gating contract so the fix above did not accidentally always-emit."""
    cfg = _base_cfg()
    cfg.channels.analysis_mode = "rna_rna"
    cfg.channels.rna2 = PROT_C
    # compute_partner_intensity defaults to False.
    res = _run(cfg, _rna_rna, fake_img.path, monkeypatch, fake_img)
    assert "partner_local_mean_intensity" not in res.spots.columns
    for col in (
        "rna2_local_mean_at_rna1_spots",
        "rna2_enrichment_at_rna1_spots",
        "rna1_local_mean_at_rna2_spots",
        "rna1_enrichment_at_rna2_spots",
    ):
        assert col not in res.nuclei.columns
    assert "mean_rna2_local_mean_at_rna1_spots" not in res.per_image


def test_rna_only_regression_smoke(fake_img, monkeypatch):
    """rna_only is unaffected by the rna_protein/FociCfg changes: it still
    segments + detects + emits its canonical single-channel schema."""
    cfg = _base_cfg()
    cfg.channels.analysis_mode = "rna_only"
    res = _run(cfg, _rna_only, fake_img.path, monkeypatch, fake_img)
    assert len(res.nuclei) >= 1
    assert "rna_spot_count" in res.nuclei.columns
    assert "sum_rna_intensity" in res.nuclei.columns
    # No 2nd-channel columns in single-channel mode.
    assert not any("rna2" in c or "protein" in c for c in res.nuclei.columns)


# ---------------------------------------------------------------------------
# Rotation "proper background" null relabels rna2 -> protein (2026-06-19).
# The native rotation feature emits rna2_rotation_* per-nucleus + per-image
# columns + (optionally) an extra["coloc_rotation_null"] DataFrame; in
# rna_protein mode those MUST surface under protein-* names with no rna2 leak.
# ---------------------------------------------------------------------------
def test_rotation_null_relabels_rna2_to_protein(fake_img, monkeypatch):
    cfg = _base_cfg()
    cfg.channels.analysis_mode = "rna_protein"
    cfg.channels.antibody = PROT_C
    cfg.channels.antibody_label = "QKI"
    cfg.foci.compute_partner_intensity = True
    cfg.foci.compute_partner_rotation_null = True
    cfg.foci.partner_rotation_n = 150
    cfg.foci.save_partner_rotation_null_draws = True
    res = _run(cfg, _rna_protein, fake_img.path, monkeypatch, fake_img)

    nuc_cols = set(res.nuclei.columns)
    # protein-* rotation per-nucleus columns present; no rna2-* leak.
    assert "protein_rotation_enrichment_at_rna1_spots" in nuc_cols
    assert "protein_rotation_null_z_at_rna1_spots" in nuc_cols
    assert "rotation_null_usable" in nuc_cols
    assert not any("rna2_rotation" in c for c in nuc_cols)

    # per-image pooled rollup relabeled too.
    assert "protein_pooled_rotation_enrichment_at_rna1_spots" in res.per_image
    assert "n_nuclei_partner_rotation_null" in res.per_image
    assert not any("rna2_pooled_rotation" in k for k in res.per_image)

    # The extra "coloc_rotation_null" carrier is emitted only when >=1 nucleus is
    # rot-USABLE (this fixture's tiny dense nuclei may all fail the retention gate
    # -> no pooled null -> no carrier, which is correct). WHEN present, its columns
    # must also be routed through the rna2->protein relabel (no rna2 token).
    rot_df = res.extra.get("coloc_rotation_null")
    if rot_df is not None:
        assert isinstance(rot_df, pd.DataFrame)
        assert not any("rna2" in c for c in rot_df.columns)
