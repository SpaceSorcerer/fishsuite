from __future__ import annotations

import pandas as pd
import pytest

from fishsuite.core.authoritative_sampling import correct_spot_sampling_flags


def test_correct_spot_sampling_flags_uses_nucleus_keys_and_preserves_zero_spot_nuclei():
    spots = pd.DataFrame(
        {
            "spot_uid": ["a:1", "a:2", "a:3"],
            "image_key": ["Field.VSI", "Field.VSI", "Field.VSI"],
            "nucleus_id": [1, 1, 2],
            "is_control": [False, False, False],
            # Deliberately wrong image-level values.
            "eligible_for_sampling": [True, True, True],
            "sampled_in_analysis": [True, True, True],
        }
    )
    nuclei = pd.DataFrame(
        {
            "image_key": ["field.vsi", "field.vsi", "field.vsi"],
            "nucleus_id": [1, 2, 3],
            "is_control": [False, False, False],
            "eligible_for_sampling": [True, True, True],
            "sampled_in_analysis": [True, False, True],
        }
    )

    corrected, audit = correct_spot_sampling_flags(spots, nuclei)

    assert corrected["sampled_in_analysis"].tolist() == [True, True, False]
    assert corrected["sampled_in_analysis_image_aggregate"].tolist() == [True, True, True]
    assert corrected["sampling_metadata_source"].eq(
        "authoritative_nucleus_key_join"
    ).all()
    assert audit["n_spots"] == 3
    assert audit["n_spot_sampling_mismatches_corrected"] == 1
    assert audit["n_sampled_nuclei"] == 2
    assert audit["n_sampled_zero_spot_nuclei"] == 1
    assert audit["n_sampled_biological_spots"] == 2


def test_correct_spot_sampling_flags_fails_on_missing_or_duplicate_nucleus_keys():
    spots = pd.DataFrame(
        {
            "spot_uid": ["a:1"],
            "image_key": ["field.vsi"],
            "nucleus_id": [2],
            "eligible_for_sampling": [True],
            "sampled_in_analysis": [True],
        }
    )
    nuclei = pd.DataFrame(
        {
            "image_key": ["field.vsi"],
            "nucleus_id": [1],
            "eligible_for_sampling": [True],
            "sampled_in_analysis": [True],
        }
    )
    with pytest.raises(ValueError, match="missing authoritative nucleus keys"):
        correct_spot_sampling_flags(spots, nuclei)

    duplicated = pd.concat([nuclei, nuclei], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate nucleus keys"):
        correct_spot_sampling_flags(spots.iloc[:0], duplicated)
