"""Costes coloc-threshold wiring (Brian, 2026-07-08).

Regression guard for the wiring gap where selecting ``threshold_mode:"costes"``
silently fell back to MAD because the call sites in ``core/modes/rna_rna.py``
(``_compute_pixel_coloc_thr``) and ``core/runner.py`` (batch pre-scan) called
``coloc_threshold()`` WITHOUT the paired ``vals_other`` argument. With the fix,
``costes`` runs the real Costes regression (``costes_threshold``) per channel;
``mad`` / ``percentile`` are unchanged whether or not a partner channel is
supplied.

  (a) UNIT — ``coloc_threshold(..., mode="costes", vals_other=...)`` runs the
      regression and differs from MAD; without ``vals_other`` it falls back to
      MAD; ``mad``/``percentile`` ignore ``vals_other`` (byte-identical).
  (b) CALL SITE — ``_compute_pixel_coloc_thr`` (the exact per-image site
      ``run_one`` uses) fires real Costes when the partner channel is passed and
      differs from the MAD result on a correlated pair.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from fishsuite.config.schema import FishsuiteConfig
from fishsuite.core import thresholds as _thr
from fishsuite.core.modes import rna_rna as _rna_rna


def _correlated_pair(n=800, seed=0):
    """A strictly-positive, correlated (r, a) pixel pair (n samples)."""
    rng = np.random.default_rng(seed)
    r = rng.uniform(50.0, 5000.0, n)
    a = 0.7 * r + rng.normal(0.0, 30.0, n) + 100.0
    return r, a


# ===========================================================================
# (a) UNIT: coloc_threshold costes branch
# ===========================================================================
def test_costes_with_partner_runs_regression_not_mad():
    """costes + vals_other == the direct Costes regression, and != the MAD
    threshold (proves the branch actually fires)."""
    r, a = _correlated_pair()
    v_costes = _thr.coloc_threshold(r.tolist(), mode="costes", vals_other=a.tolist())
    v_mad = _thr.coloc_threshold(r.tolist(), mode="mad")
    r_thr_direct, _ = _thr.costes_threshold(r.tolist(), a.tolist())

    assert math.isfinite(v_costes)
    assert v_costes == pytest.approx(r_thr_direct)   # it's the regression value
    assert v_costes != pytest.approx(v_mad)          # not the MAD fallback


def test_costes_without_partner_falls_back_to_mad():
    """No paired channel -> single-channel MAD fallback (unchanged behavior)."""
    r, _ = _correlated_pair(seed=1)
    assert _thr.coloc_threshold(r.tolist(), mode="costes") == pytest.approx(
        _thr.coloc_threshold(r.tolist(), mode="mad")
    )


def test_mad_and_percentile_ignore_vals_other():
    """vals_other must NOT perturb mad / percentile (byte-identical)."""
    r, a = _correlated_pair(seed=2)
    assert _thr.coloc_threshold(r.tolist(), mode="mad", vals_other=a.tolist()) == (
        _thr.coloc_threshold(r.tolist(), mode="mad")
    )
    assert _thr.coloc_threshold(r.tolist(), mode="percentile", vals_other=a.tolist()) == (
        _thr.coloc_threshold(r.tolist(), mode="percentile")
    )


def test_costes_mismatched_length_falls_back_to_mad():
    """A length-mismatched partner (would break pixel pairing) -> MAD fallback,
    never a crash or a non-finite value."""
    r, a = _correlated_pair(seed=3)
    v = _thr.coloc_threshold(r.tolist(), mode="costes", vals_other=a[:-5].tolist())
    assert math.isfinite(v)
    assert v == pytest.approx(_thr.coloc_threshold(r.tolist(), mode="mad"))


# ===========================================================================
# (b) CALL SITE: _compute_pixel_coloc_thr (the exact per-image site run_one uses)
# ===========================================================================
def _pc_cfg(mode):
    cfg = FishsuiteConfig()
    pc = cfg.pixel_coloc
    pc.threshold_mode = mode
    pc.threshold_scope = "per_image"
    return pc


def test_compute_pixel_coloc_thr_costes_fires_and_differs_from_mad():
    """The fixed call site: passing the partner channel makes costes run the
    real regression on the nuclear-mask pixels, giving a threshold that differs
    from the MAD fallback and equals the direct Costes value."""
    rng = np.random.default_rng(7)
    H = W = 60
    labels = np.zeros((H, W), dtype=np.int32)
    labels[10:50, 10:50] = 1  # 1600-pixel nucleus (n >= 20)
    base = rng.uniform(100.0, 4000.0, (H, W))
    rna = base.astype(np.float64)
    rna2 = (0.8 * base + rng.normal(0.0, 50.0, (H, W)) + 200.0).astype(np.float64)

    thr_mad = _rna_rna._compute_pixel_coloc_thr(
        rna, labels, pc_cfg=_pc_cfg("mad"),
        precomputed=None, bigfish_auto_thr=0.0, img2d_other=rna2,
    )
    thr_costes = _rna_rna._compute_pixel_coloc_thr(
        rna, labels, pc_cfg=_pc_cfg("costes"),
        precomputed=None, bigfish_auto_thr=0.0, img2d_other=rna2,
    )

    assert math.isfinite(thr_costes)
    assert thr_costes != pytest.approx(thr_mad)  # Costes fired, not MAD

    # It equals the direct regression on exactly the paired nuclear pixels.
    m = labels > 0
    r_thr_direct, _ = _thr.costes_threshold(
        rna[m].astype(np.float64).tolist(), rna2[m].astype(np.float64).tolist()
    )
    assert thr_costes == pytest.approx(r_thr_direct)


def test_compute_pixel_coloc_thr_mad_unchanged_by_partner():
    """MAD result is identical with or without a partner channel supplied
    (the default path is untouched)."""
    rng = np.random.default_rng(8)
    H = W = 50
    labels = np.zeros((H, W), dtype=np.int32)
    labels[5:45, 5:45] = 1
    rna = rng.uniform(0.0, 3000.0, (H, W)).astype(np.float64)
    rna2 = rng.uniform(0.0, 3000.0, (H, W)).astype(np.float64)

    with_partner = _rna_rna._compute_pixel_coloc_thr(
        rna, labels, pc_cfg=_pc_cfg("mad"),
        precomputed=None, bigfish_auto_thr=0.0, img2d_other=rna2,
    )
    without_partner = _rna_rna._compute_pixel_coloc_thr(
        rna, labels, pc_cfg=_pc_cfg("mad"),
        precomputed=None, bigfish_auto_thr=0.0,
    )
    assert with_partner == without_partner
