"""Legacy qki-assoc column VALUES are preserved by the 2026-09-24 additions.

Recomputes every authorized basal field (VPR noDox + Nog noDox, four fields)
from the production exact-footprint backfill (read-only) with the production
settings recorded in RESULTS_PROD_..._211001/qki_assoc/command.log, then
compares a SHA-256 of the legacy-column subset, serialized as text exactly as
written to disk, with the same subset of the DELIVERED CSVs (per nucleus and
per spot). The complete new CSV is not byte-identical to the delivery because
columns were appended; the claim is "legacy column values preserved".
Optics come from the source VSI OME metadata (the production path).
Skips when the F: data are not mounted.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(r"F:\Image Analysis Work\MIAT-QKI-Coloc")
BACKFILL = ROOT / "UD" / "_OEvWT_2026-09" / "EXACT_FOOTPRINT_BACKFILL_20260917-205946"
DELIVERED = ROOT / "DELIVERY_MIAT_OE_UD_2026-09-18" / "qki_association"
AUTHORIZED_CONDITIONS = ("VPR noDox", "Nog noDox")
N_AUTHORIZED_FIELDS = 4

pytestmark = pytest.mark.skipif(
    not (BACKFILL / "selected_planes_and_masks.h5").is_file()
    or not (DELIVERED / "qki_association_per_nucleus.csv").is_file(),
    reason="production backfill / delivery not mounted")


def _legacy_digest(path, columns, keys, conditions):
    text = pd.read_csv(path, dtype=str, keep_default_na=False)
    text = text[text.condition.isin(conditions)]
    subset = text[list(columns)].sort_values(list(keys)).reset_index(drop=True)  # text sort, same both sides
    payload = subset.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), len(subset), subset.image.nunique()


@pytest.fixture(scope="module")
def recomputed(tmp_path_factory):
    from fishsuite.core.qki_association_postrun import run_qki_association
    return run_qki_association(BACKFILL, tmp_path_factory.mktemp("legacy") / "assoc",
                               miat_min=500.0, qki_min=1050.0, sensitivity=(0.8, 1.0, 1.25),
                               n_null=200, seed=0, conditions=AUTHORIZED_CONDITIONS)


def test_legacy_nucleus_subset_hash_matches_delivery_on_all_authorized_fields(recomputed):
    from fishsuite.core import qki_association as qa
    keys = ("threshold_multiplier", "image", "nucleus_id")
    new = _legacy_digest(recomputed / "qki_association_per_nucleus.csv", qa.LEGACY_NUCLEUS_COLUMNS, keys,
                         AUTHORIZED_CONDITIONS)
    old = _legacy_digest(DELIVERED / "qki_association_per_nucleus.csv", qa.LEGACY_NUCLEUS_COLUMNS, keys,
                         AUTHORIZED_CONDITIONS)
    assert new[2] == old[2] == N_AUTHORIZED_FIELDS
    assert new[1] == old[1] > 0
    assert new[0] == old[0]


def test_legacy_spot_subset_hash_matches_delivery_on_all_authorized_fields(recomputed):
    from fishsuite.core import qki_association as qa
    keys = ("threshold_multiplier", "image", "nucleus_id", "spot_id")
    new = _legacy_digest(recomputed / "qki_association_per_spot.csv", qa.LEGACY_SPOT_COLUMNS, keys,
                         AUTHORIZED_CONDITIONS)
    old = _legacy_digest(DELIVERED / "qki_association_per_spot.csv", qa.LEGACY_SPOT_COLUMNS, keys,
                         AUTHORIZED_CONDITIONS)
    assert new[2] == old[2] == N_AUTHORIZED_FIELDS
    assert new[1] == old[1] > 0
    assert new[0] == old[0]


def test_new_columns_present_and_psf_from_source_metadata(recomputed):
    nuclei = pd.read_csv(recomputed / "qki_association_per_nucleus.csv")
    assert nuclei.pearson_r_nucleoplasm.notna().all()
    assert nuclei.mean_uniform_position_percentile_qki.notna().any()
    assert nuclei.costes_psf_source.str.startswith("source OME metadata").all()
