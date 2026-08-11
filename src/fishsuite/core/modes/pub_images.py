"""pub_images — figures-only mode (Phase-2 stub)."""
from __future__ import annotations

from typing import Optional

from . import register_mode, rna_only as _rna_only


@register_mode("pub_images")
def run(path, *, condition: str, sec_only: bool, cfg,
        sampling_unit_key: Optional[str] = None,
        sampling_n_alloc: Optional[int] = None):
    # See the note in ab_ab.run: these are forwarded so `sampling.unit:
    # per_well` is actually honoured rather than silently becoming per_image.
    return _rna_only.run_one(
        path, condition=condition, sec_only=sec_only, cfg=cfg,
        sampling_unit_key=sampling_unit_key,
        sampling_n_alloc=sampling_n_alloc,
    )
