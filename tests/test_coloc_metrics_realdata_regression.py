"""Legacy qki-assoc columns are unchanged by the 2026-09-24 additions, on real data.

Recomputes one real field of the VPR noDox arm from the production
exact-footprint backfill (read-only) with the production settings recorded in
RESULTS_PROD_..._211001/qki_assoc/command.log and compares every legacy
per-nucleus and per-spot column with the DELIVERED CSVs (numeric within 1e-9,
text exactly). Skips when the F: data are not mounted.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(r"F:\Image Analysis Work\MIAT-QKI-Coloc")
BACKFILL = ROOT / "UD" / "_OEvWT_2026-09" / "EXACT_FOOTPRINT_BACKFILL_20260917-205946"
DELIVERED = ROOT / "DELIVERY_MIAT_OE_UD_2026-09-18" / "qki_association"
IMAGE_KEY = ("f:/raw images/121925_miat-qki_confocal_ud_miat oe v wt/vpr/no dox/"
             "ud-miat-fish-qki-if-vpr-no dox_14.vsi")

pytestmark = pytest.mark.skipif(
    not (BACKFILL / "selected_planes_and_masks.h5").is_file()
    or not (DELIVERED / "qki_association_per_nucleus.csv").is_file(),
    reason="production backfill / delivery not mounted")


def _compare(new, old, columns, keys):
    new = new.sort_values(keys).reset_index(drop=True)
    old = old.sort_values(keys).reset_index(drop=True)
    assert len(new) == len(old) > 0
    for column in columns:
        a, b = new[column], old[column]
        if pd.api.types.is_numeric_dtype(b) and not pd.api.types.is_bool_dtype(b):
            a, b = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
            assert np.array_equal(np.isnan(a), np.isnan(b)), column
            finite = ~np.isnan(b)
            assert np.max(np.abs(a[finite] - b[finite]), initial=0.0) <= 1e-9, column
        else:
            assert a.fillna("").astype(str).tolist() == b.fillna("").astype(str).tolist(), column


def test_vpr_nodox_field_legacy_columns_match_delivery(tmp_path):
    from fishsuite.core import qki_association as qa
    from fishsuite.core.qki_association_postrun import run_qki_association

    out = run_qki_association(BACKFILL, tmp_path / "assoc", miat_min=500.0, qki_min=1050.0,
                              sensitivity=(0.8, 1.0, 1.25), n_null=200, seed=0,
                              image_keys=(IMAGE_KEY,))
    new_nuc = pd.read_csv(out / "qki_association_per_nucleus.csv", keep_default_na=False,
                          na_values=[""], dtype={"well": str})
    new_spot = pd.read_csv(out / "qki_association_per_spot.csv", dtype={"spot_id": str})
    old_nuc = pd.read_csv(DELIVERED / "qki_association_per_nucleus.csv", keep_default_na=False,
                          na_values=[""], dtype={"well": str})
    old_spot = pd.read_csv(DELIVERED / "qki_association_per_spot.csv", dtype={"spot_id": str})
    old_nuc = old_nuc[old_nuc.image.eq(IMAGE_KEY)]
    old_spot = old_spot[old_spot.image.eq(IMAGE_KEY)]

    assert list(old_nuc.columns) == list(qa.LEGACY_NUCLEUS_COLUMNS)
    assert list(new_nuc.columns[:len(qa.LEGACY_NUCLEUS_COLUMNS)]) == list(qa.LEGACY_NUCLEUS_COLUMNS)
    _compare(new_nuc, old_nuc, qa.LEGACY_NUCLEUS_COLUMNS, ["threshold_multiplier", "nucleus_id"])
    _compare(new_spot, old_spot, qa.LEGACY_SPOT_COLUMNS,
             ["threshold_multiplier", "nucleus_id", "spot_id"])
    assert new_nuc.mean_null_rank_qki_at_miat.notna().any()
    assert new_nuc.pearson_r_nucleoplasm.notna().all()
