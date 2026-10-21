"""pub_images — figures-only mode (Phase-2 stub)."""
from __future__ import annotations

from typing import Optional

from . import register_mode


# Registered so the mode name still validates; not implemented yet.
@register_mode("pub_images")
def run(path, *, condition: str, sec_only: bool, cfg,
        sampling_unit_key: Optional[str] = None,
        sampling_n_alloc: Optional[int] = None):
    raise NotImplementedError("analysis mode 'pub_images' is not implemented yet.")
