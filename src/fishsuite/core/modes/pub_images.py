"""pub_images — figures-only mode (Phase-2 stub)."""
from __future__ import annotations

from . import register_mode, rna_only as _rna_only


@register_mode("pub_images")
def run(path, *, condition: str, sec_only: bool, cfg):
    return _rna_only.run_one(path, condition=condition, sec_only=sec_only, cfg=cfg)
