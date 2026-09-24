"""Replicate-well identity: composite (biological_set, slide, well) or a
validated globally unique well label. Ambiguous reuse fails loudly.

A physical well carries exactly one condition. A well label that appears under
two conditions therefore cannot be a unique physical well: either the labels
restart per slide/set (so the composite components are required) or the
table is wrong. In both cases grouping on the bare label could merge or split
replicates silently, so it is refused.
"""
from __future__ import annotations

import pandas as pd

COMPOSITE = ("biological_set", "slide")


def _present(frame, column):
    return column in frame.columns and not frame[column].isna().all() and \
        not frame[column].astype(str).str.strip().eq("").all()


def resolve_well_ids(frame: pd.DataFrame, *, well_col: str = "well",
                     condition_col: str = "condition") -> tuple[pd.Series, str]:
    """Return (well key per row, key source). Raises ValueError on ambiguity."""
    wells = frame[well_col].astype(str)
    if all(_present(frame, c) for c in COMPOSITE):
        parts = [frame[c].astype(str).str.strip() for c in COMPOSITE]
        if any(p.eq("").any() or p.eq("nan").any() for p in parts):
            raise ValueError("biological_set and slide must be filled on every row when used "
                             "for the composite well key")
        keys = parts[0] + "|" + parts[1] + "|" + wells
        source = "biological_set|slide|" + well_col
    elif any(_present(frame, c) for c in COMPOSITE):
        raise ValueError("composite well key needs BOTH biological_set and slide; only one is present")
    else:
        keys = wells
        source = f"{well_col} (validated globally unique)"
    per_key = pd.DataFrame({"key": keys.to_numpy(), "condition": frame[condition_col].astype(str).to_numpy()})
    conditions = per_key.groupby("key")["condition"].nunique()
    reused = sorted(conditions[conditions > 1].index)
    if reused:
        raise ValueError(
            "ambiguous well identity: well key(s) {} occur under more than one condition. "
            "Supply biological_set and slide columns (composite key) or globally unique "
            "well labels.".format(", ".join(map(repr, reused[:10]))))
    return pd.Series(keys.to_numpy(), index=frame.index, name=well_col), source
