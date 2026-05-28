"""Parallel batch execution helpers."""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Iterable, List, Optional


# ---------------------------------------------------------------------------
# Per-worker thread cap (avoid BLAS / OMP oversubscription).
#
# When we run N worker PROCESSES and each one also spins up numpy/MKL/OpenMP/
# torch threads, N x threads_per_worker can blow past the 24 logical cores and
# cause cache thrash that makes things SLOWER than fewer workers. This module
# initializer pins each worker's thread pools. Set the env vars BEFORE numpy /
# torch import inside the worker (process start), so it is reliable.
# ---------------------------------------------------------------------------

def _init_worker_threads(threads_per_worker: int) -> None:
    if threads_per_worker and threads_per_worker > 0:
        n = str(int(threads_per_worker))
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS"):
            os.environ[var] = n
        try:
            import torch  # noqa
            torch.set_num_threads(int(threads_per_worker))
        except Exception:
            pass


def auto_seg_workers(
    *,
    device: str = "cpu",
    per_worker_ram_gb: float = 3.0,
    cap: int = 8,
) -> int:
    """Worker count for the segmentation pre-scan.

    cpsam holds a ~1.2 GB model + working tensors per worker. On directml the
    workers cannot share the 12 GB VRAM, so we force 1 (serial GPU seg — the
    GPU is already ~7x faster per image). On CPU we budget ~3 GB/worker and
    leave 2 cores free, capped at 8 to bound memory + cache contention.
    """
    if str(device).lower() in ("directml", "dml"):
        return 1
    try:
        import psutil
        physical = psutil.cpu_count(logical=False) or 6
        avail_gb = psutil.virtual_memory().available / (1024 ** 3)
    except Exception:
        physical = (os.cpu_count() or 8) // 2
        avail_gb = 32.0
    by_cpu = max(1, (physical or 6) - 2)
    by_mem = max(1, int(avail_gb / max(per_worker_ram_gb, 1.0)))
    return int(min(by_cpu, by_mem, cap))


def auto_main_workers(*, cap: int = 12, per_worker_ram_gb: float = 4.0) -> int:
    """Worker count for the main per-image pass (BigFISH + measure + figures).

    Memory-light vs segmentation but each worker still holds a few full-frame
    float arrays for rendering, so budget ~4 GB/worker and leave 2 cores free.
    """
    try:
        import psutil
        physical = psutil.cpu_count(logical=False) or 6
        avail_gb = psutil.virtual_memory().available / (1024 ** 3)
    except Exception:
        physical = (os.cpu_count() or 8) // 2
        avail_gb = 32.0
    by_cpu = max(1, (physical or 6) - 2)
    by_mem = max(1, int(avail_gb / max(per_worker_ram_gb, 1.0)))
    return int(min(by_cpu, by_mem, cap))


def resolve_workers(value, *, kind: str = "main", device: str = "cpu") -> int:
    """Resolve a workers config value ('auto' | int) to a concrete count."""
    if isinstance(value, str) and value.lower() == "auto":
        return auto_seg_workers(device=device) if kind == "seg" else auto_main_workers()
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 1


def auto_n_workers(
    *,
    cap: int = 12,
    per_worker_ram_gb: float = 2.0,
) -> int:
    """Pick a sensible worker count.

    Formula: ``min(physical_cores - 2, by_memory, cap)`` — leaves 2 cores
    for OS / UI / StarDist internal threading, capped at 12 to avoid
    cache contention on numeric LoG kernels.
    On Brian's box (12 physical / 24 logical / 128 GB):
      by_cpu = 12 - 2 = 10
      by_memory = ~110 / 2 = 55
      cap = 12
      -> 10 workers (was 6 with the prior formula).
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
    # Reserve 2 cores for OS + UI. On small boxes (<= 4 cores) just halve.
    if physical <= 4:
        by_cpu = max(1, physical // 2)
    else:
        by_cpu = max(1, physical - 2)
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
