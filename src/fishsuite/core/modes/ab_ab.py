"""ab_ab — pixel coloc on two antibody channels (Phase-2 stub)."""
from __future__ import annotations

from typing import Optional

from . import register_mode, rna_only as _rna_only


@register_mode("ab_ab")
def run(path, *, condition: str, sec_only: bool, cfg,
        sampling_unit_key: Optional[str] = None,
        sampling_n_alloc: Optional[int] = None):
    # The two sampling kwargs are accepted purely to be passed through. Without
    # them in this signature the runner cannot forward the fixed-N plan, so
    # rna_only falls back to a per-IMAGE unit key and `sampling.unit: per_well`
    # degrades to per_image while sampling_methods.txt still describes a per-well
    # equal split. See runner.SAMPLING_SUPPORTED_MODES.
    return _rna_only.run_one(
        path, condition=condition, sec_only=sec_only, cfg=cfg,
        sampling_unit_key=sampling_unit_key,
        sampling_n_alloc=sampling_n_alloc,
    )
