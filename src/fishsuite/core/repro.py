"""Reproducibility + provenance helpers for fishsuite (ADDITIVE, 2026-06-10).

This module is purely additive QC/reproducibility scaffolding. None of it
changes any existing analysis number, CSV column, or config default — it only
*records* what happened (versions, command line, the global seed) and locks the
global RNG state so stochastic steps are deterministic given a seed.

Three responsibilities:

1. ``set_global_seeds(seed)`` — seed Python's ``random``, NumPy, the
   ``PYTHONHASHSEED`` env var, and (if importable) torch / torch-CUDA, each in
   its OWN try/except so a seeding failure on one backend can NEVER abort a run.
   This is the BROAD global seed, complementary to — and NOT a replacement for —
   the focused ``foci.partner_null_seed`` used by the per-image partner-null
   permutation test.

2. ``write_versions_txt(out_dir, seed)`` — write a ``versions.txt`` capturing
   the fishsuite version, interpreter, platform, the global seed, and the
   installed versions of the scientific stack (numpy/scipy/scikit-image/pandas/
   cellpose/stardist/big-fish/torch/torch-directml/bioio/bioio-bioformats).

3. ``write_command_log(out_dir, config_path, output_dir, seed)`` — write a
   compact ``command.log`` with the exact ``sys.argv``, the resolved config
   path, the output dir, and the global seed.

Every writer is crash-proof: it catches all exceptions and returns ``False``
rather than raising, so a metadata-write failure can never abort a real run.
"""
from __future__ import annotations

import os
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np


# Module-level default RNG exposed for callers that want a seeded Generator.
# Re-seeded by ``set_global_seeds``. Optional convenience; legacy code that uses
# the global ``np.random`` state is unaffected.
default_rng = np.random.default_rng()


def set_global_seeds(seed: int) -> Dict[str, bool]:
    """Seed every available RNG backend DEFENSIVELY.

    Each backend is seeded inside its own try/except so this function can NEVER
    raise — even if ``import torch`` itself raises (e.g. torch monkeypatched
    absent) or torch-directml lacks deterministic-algorithm support.

    Parameters
    ----------
    seed : int
        The global seed. ``0`` is a valid, deterministic seed.

    Returns
    -------
    dict[str, bool]
        Which backends were successfully seeded, e.g.::

            {"python_random": True, "numpy": True, "pythonhashseed": True,
             "torch": False, "torch_cuda": False, "torch_deterministic": False}
    """
    seeded: Dict[str, bool] = {
        "python_random": False,
        "numpy": False,
        "pythonhashseed": False,
        "torch": False,
        "torch_cuda": False,
        "torch_deterministic": False,
    }

    try:
        random.seed(seed)
        seeded["python_random"] = True
    except Exception:
        pass

    try:
        np.random.seed(seed)
        # Also refresh the module-level Generator so callers using it stay
        # deterministic. Failure here must not undo the np.random.seed above.
        try:
            global default_rng
            default_rng = np.random.default_rng(seed)
        except Exception:
            pass
        seeded["numpy"] = True
    except Exception:
        pass

    try:
        os.environ["PYTHONHASHSEED"] = str(seed)
        seeded["pythonhashseed"] = True
    except Exception:
        pass

    # torch is optional. Guard the import itself so a raising/absent torch
    # (including a monkeypatched-absent torch) cannot break seeding.
    try:
        import torch  # type: ignore
    except Exception:
        torch = None  # type: ignore

    if torch is not None:
        try:
            torch.manual_seed(seed)
            seeded["torch"] = True
        except Exception:
            pass
        try:
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                seeded["torch_cuda"] = True
        except Exception:
            pass
        try:
            # Degrades gracefully on torch-directml / backends without
            # deterministic-algorithm support. Catch ALL exceptions.
            torch.use_deterministic_algorithms(True, warn_only=True)
            seeded["torch_deterministic"] = True
        except Exception:
            pass

    return seeded


# Distribution names for importlib.metadata.version. Import name differs from
# the installed-distribution name for several packages (skimage->scikit-image,
# bigfish->big-fish, torch_directml->torch-directml), so we use the DISTRIBUTION
# names here.
_VERSION_PKGS = [
    "numpy",
    "scipy",
    "scikit-image",
    "pandas",
    "cellpose",
    "stardist",
    "big-fish",
    "torch",
    "torch-directml",
    "bioio",
    "bioio-bioformats",
]


def write_versions_txt(out_dir: Path | str, seed: int) -> bool:
    """Write ``versions.txt`` into ``out_dir``. Crash-proof (returns bool).

    Records the fishsuite version, interpreter + executable, platform, the
    global seed, and per-package installed versions. Never raises.
    """
    try:
        from importlib.metadata import (
            PackageNotFoundError,
            version as _pkg_version,
        )

        try:
            from .. import __version__ as _fs_version
        except Exception:
            _fs_version = "unknown"

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        lines = []
        lines.append(f"fishsuite_version: {_fs_version}")
        lines.append(f"global_seed: {seed}")
        lines.append(
            f"written_utc: {datetime.now(tz=timezone.utc).isoformat()}"
        )
        lines.append(f"python: {sys.version.split()[0]}")
        lines.append(f"python_executable: {sys.executable}")
        lines.append(f"platform: {platform.platform()}")
        lines.append("")
        lines.append("# installed package versions (distribution names)")
        for pkg in _VERSION_PKGS:
            try:
                v = _pkg_version(pkg)
            except PackageNotFoundError:
                v = "not installed"
            except Exception as e:  # pragma: no cover - defensive
                v = f"not installed ({e})"
            lines.append(f"{pkg}: {v}")

        (out_dir / "versions.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        return True
    except Exception:
        return False


def write_command_log(
    out_dir: Path | str,
    config_path: Path | str,
    output_dir: Path | str,
    seed: int,
    *,
    extra: Dict[str, Any] | None = None,
) -> bool:
    """Write a compact ``command.log`` into ``out_dir``. Crash-proof.

    Captures the exact ``sys.argv``, the resolved config path, the output dir,
    and the global seed. ``extra`` (optional) appends a few compact key:value
    lines (e.g. analysis_mode, z mode). Never raises.
    """
    try:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        lines = []
        lines.append(f"written_utc: {datetime.now(tz=timezone.utc).isoformat()}")
        lines.append(f"argv: {' '.join(sys.argv)}")
        lines.append(f"config_path: {str(config_path)}")
        lines.append(f"output_dir: {str(output_dir)}")
        lines.append(f"global_seed: {seed}")
        if extra:
            for k, v in extra.items():
                try:
                    lines.append(f"{k}: {v}")
                except Exception:
                    pass
        (out_dir / "command.log").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        return True
    except Exception:
        return False


def write_run_metadata(
    out_dir: Path | str,
    config_path: Path | str,
    output_dir: Path | str,
    seed: int,
    *,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, bool]:
    """Convenience: write both ``versions.txt`` and ``command.log``.

    Returns ``{"versions_txt": bool, "command_log": bool}``. Never raises.
    """
    return {
        "versions_txt": write_versions_txt(out_dir, seed),
        "command_log": write_command_log(
            out_dir, config_path, output_dir, seed, extra=extra
        ),
    }
