"""Parallel batch execution helpers."""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Iterable, List


def auto_n_workers(
    *,
    cap: int = 8,
    per_worker_ram_gb: float = 1.5,
) -> int:
    """Pick a sensible worker count.

    Formula: ``min(physical_cores // 2, by_memory, cap)``.
    On Brian's box: 12 physical -> 6, well under 128 GB, capped to 8.
    """
    try:
        import psutil
        physical = psutil.cpu_count(logical=False)
        avail_gb = psutil.virtual_memory().available / (1024 ** 3)
    except Exception:
        physical = (os.cpu_count() or 4) // 2
        avail_gb = 8.0
    if not physical or physical < 1:
        physical = 2
    by_cpu = max(1, physical // 2)
    by_memory = max(1, int(avail_gb / max(per_worker_ram_gb, 0.5)))
    return int(min(by_cpu, by_memory, cap))


class BatchExecutor:
    """Thin wrapper over ProcessPoolExecutor with progress + error capture."""

    def __init__(self, n_workers: int = 1):
        self.n_workers = max(1, int(n_workers))

    def map(self, fn: Callable, items: Iterable, *, on_done: Callable | None = None) -> List:
        items = list(items)
        results: List = [None] * len(items)
        if self.n_workers == 1:
            for i, it in enumerate(items):
                try:
                    results[i] = fn(it)
                except Exception as e:
                    results[i] = ("ERROR", repr(e))
                if on_done:
                    on_done(i, results[i], it)
            return results

        with ProcessPoolExecutor(max_workers=self.n_workers) as pool:
            future_to_idx = {pool.submit(fn, it): i for i, it in enumerate(items)}
            for fut in as_completed(future_to_idx):
                i = future_to_idx[fut]
                try:
                    results[i] = fut.result()
                except Exception as e:
                    results[i] = ("ERROR", repr(e))
                if on_done:
                    on_done(i, results[i], items[i])
        return results
