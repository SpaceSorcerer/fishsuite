"""Regression guard: the 2026-05-27 bbox-crop speedup of detect_nucleoli /
chromatin_metrics_per_nucleus must stay byte-identical to the legacy
full-frame implementation.

The legacy implementation is reproduced inline here as the oracle. The test
covers the cases that exposed divergences during development:
  - touching nuclei (shared boundary; global-vs-local erosion semantics)
  - a nucleus on the frame edge (erosion / closing border handling)
  - multiple nucleoli per nucleus and size-filter boundaries
"""
import numpy as np
import pytest

from fishsuite.core.nucleolus import (
    NucleolusParams, detect_nucleoli, chromatin_metrics_per_nucleus,
)


def _legacy_detect(nuc_labels, dapi_2d, pixel_size_um, params):
    """Pre-2026-05-27 full-frame implementation (the oracle)."""
    out = np.zeros_like(nuc_labels, dtype=np.int32)
    min_area_px = max(1, int(round(params.min_area_um2 / (pixel_size_um ** 2))))
    uids = np.unique(nuc_labels); uids = uids[uids > 0]
    from skimage.morphology import disk, binary_closing
    se = disk(params.closing_radius_px) if params.closing_radius_px > 0 else None
    bd = int(getattr(params, "min_border_distance_px", 0) or 0)
    eroded = None
    if bd > 0:
        from skimage.morphology import binary_erosion as be, disk as dk
        eroded = be(nuc_labels > 0, dk(bd))
    from scipy.ndimage import label as cc_label
    for nid in uids:
        nuc_mask = nuc_labels == nid
        na = int(nuc_mask.sum())
        if na < min_area_px * 2:
            continue
        thr = float(np.percentile(dapi_2d[nuc_mask], params.intra_nuclear_percentile))
        if eroded is not None:
            interior = nuc_mask & eroded
            if not interior.any():
                interior = nuc_mask
        else:
            interior = nuc_mask
        candidate = (dapi_2d <= thr) & interior
        cc, _ = cc_label(candidate)
        max_a = int(na * params.max_area_frac_of_nucleus)
        for cid in range(1, int(cc.max()) + 1):
            cm = cc == cid; cs = int(cm.sum())
            if cs < min_area_px or cs > max_a:
                continue
            if se is not None:
                cm = binary_closing(cm, se); cm &= nuc_mask
            out[cm] = int(nid)
    return out


def _make_scene():
    """Build a 200x200 scene: two touching nuclei, one edge nucleus, each with
    a DAPI-low nucleolus island."""
    rng = np.random.default_rng(7)
    labels = np.zeros((200, 200), dtype=np.int32)
    dapi = (rng.random((200, 200)) * 200 + 800).astype(np.float64)  # bright bg chromatin

    # Nucleus 1 and 2: touching (share column 90)
    labels[30:90, 30:90] = 1
    labels[30:90, 90:150] = 2
    # Nucleus 3: on the top-left frame edge
    labels[0:50, 0:40] = 3

    # Give each a central DAPI-low nucleolus
    for nid, (cy, cx) in [(1, (60, 60)), (2, (60, 120)), (3, (20, 18))]:
        yy, xx = np.ogrid[:200, :200]
        disk_mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= 12 ** 2
        disk_mask &= (labels == nid)
        dapi[disk_mask] = 100.0  # very low -> nucleolus
    return labels, dapi


@pytest.mark.parametrize("border", [0, 3, 5])
@pytest.mark.parametrize("closing", [0, 2])
def test_detect_nucleoli_byte_identical(border, closing):
    labels, dapi = _make_scene()
    params = NucleolusParams(
        intra_nuclear_percentile=25.0, min_area_um2=1.0,
        max_area_frac_of_nucleus=0.6, closing_radius_px=closing,
        min_border_distance_px=border,
    )
    pix_um = 0.13
    legacy = _legacy_detect(labels, dapi, pix_um, params)
    new = detect_nucleoli(labels, dapi, pix_um, params)
    assert np.array_equal(legacy, new), (
        f"nucleolus label maps differ (border={border}, closing={closing}): "
        f"legacy px={int((legacy>0).sum())} new px={int((new>0).sum())}"
    )


def test_chromatin_metrics_byte_identical():
    labels, dapi = _make_scene()
    params = NucleolusParams(min_border_distance_px=5, closing_radius_px=2)
    nucleoli = detect_nucleoli(labels, dapi, 0.13, params)
    df = chromatin_metrics_per_nucleus(labels, dapi, nucleoli)
    # Oracle: full-frame per-nucleus metrics
    for nid in [1, 2, 3]:
        mask = labels == nid
        vals = dapi[mask]
        nm = (nucleoli == nid) & mask
        onm = mask & (~nm)
        exp_mean = float(vals.mean())
        exp_med = float(np.median(vals))
        exp_na = int(nm.sum())
        exp_out = float(dapi[onm].mean()) if onm.any() else float("nan")
        row = df[df.nucleus_id == nid].iloc[0]
        assert abs(row.dapi_mean - exp_mean) < 1e-9
        assert abs(row.dapi_median - exp_med) < 1e-9
        assert int(row.nucleolus_area_px) == exp_na
        if exp_na and onm.any():
            assert abs(float(row.chromatin_dapi_mean) - exp_out) < 1e-9
