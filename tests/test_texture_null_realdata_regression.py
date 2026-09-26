"""Adding the SENSITIVITY-ONLY texture-matched / rotation nulls leaves every
existing qki-assoc column value-identical on real data.

Recomputes field VPR noDox_14 from the production exact-footprint backfill
(read-only) with the flags in RESULTS_basal_noDox/qki_assoc/command.log plus
--texture-null --rotation-upp, and compares the existing nucleus and spot
columns, serialized as text, with the delivered basal run. Skips without F:.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

BACKFILL = Path(r"F:\Image Analysis Work\MIAT-QKI-Coloc\UD\_OEvWT_2026-09\EXACT_FOOTPRINT_BACKFILL_20260917-205946")
DELIVERED = Path(r"F:\Image Analysis Work\MIAT_QKI_BASAL_Fig1_2026-09-24\RESULTS_basal_noDox\qki_assoc")
FIELD = ("f:/raw images/121925_miat-qki_confocal_ud_miat oe v wt/vpr/no dox/"
         "ud-miat-fish-qki-if-vpr-no dox_14.vsi")

pytestmark = pytest.mark.skipif(
    not (BACKFILL / "selected_planes_and_masks.h5").is_file()
    or not (DELIVERED / "qki_association_per_nucleus.csv").is_file(),
    reason="production backfill / basal delivery not mounted")


@pytest.fixture(scope="module")
def recomputed(tmp_path_factory):
    from fishsuite.core.qki_association_postrun import run_qki_association
    from fishsuite.core.texture_null import TextureNullParams
    return run_qki_association(BACKFILL, tmp_path_factory.mktemp("texture") / "assoc",
                               miat_min=500.0, qki_min=1050.0, sensitivity=(0.8, 1.0, 1.25),
                               n_null=200, seed=0, n_costes=200, image_keys=(FIELD,),
                               texture_null=TextureNullParams(), rotation_upp=True)


def _text(path, columns, keys):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame = frame[frame.image == FIELD][list(columns)].sort_values(list(keys)).reset_index(drop=True)
    return frame


@pytest.mark.parametrize("table,columns_name,keys", [
    ("qki_association_per_nucleus.csv", "NUCLEUS_COLUMNS", ("threshold_multiplier", "nucleus_id")),
    ("qki_association_per_spot.csv", "SPOT_COLUMNS", ("threshold_multiplier", "nucleus_id", "spot_id")),
])
def test_existing_columns_value_identical_with_sensitivity_nulls_on(recomputed, table, columns_name, keys):
    from fishsuite.core import qki_association as qa
    columns = getattr(qa, columns_name)
    new, old = _text(recomputed / table, columns, keys), _text(DELIVERED / table, columns, keys)
    assert len(new) == len(old) > 0
    pd.testing.assert_frame_equal(new, old)


def test_sensitivity_columns_present_and_defined(recomputed):
    nuclei = pd.read_csv(recomputed / "qki_association_per_nucleus.csv")
    spots = pd.read_csv(recomputed / "qki_association_per_spot.csv")
    assert nuclei.upp_texture_matched_mean_qki.notna().any()
    assert spots.upp_texture_matched_qki.notna().mean() > 0.5
    assert nuclei.upp_rotation_mean_qki.notna().any()
    assert (recomputed / "texture_null_per_well.csv").is_file()
