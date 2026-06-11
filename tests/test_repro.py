"""Tests for the additive reproducibility/provenance helpers (2026-06-10)."""
from __future__ import annotations

import random

import numpy as np

from fishsuite.core import repro


def test_set_global_seeds_returns_dict():
    seeded = repro.set_global_seeds(123)
    assert isinstance(seeded, dict)
    # Core backends always seed in this env.
    assert seeded["python_random"] is True
    assert seeded["numpy"] is True
    assert seeded["pythonhashseed"] is True
    # torch keys are present regardless of whether torch is importable.
    for k in ("torch", "torch_cuda", "torch_deterministic"):
        assert k in seeded
        assert isinstance(seeded[k], bool)


def test_same_seed_identical_draw_sequences():
    repro.set_global_seeds(7)
    np_a = np.random.rand(8).tolist()
    py_a = [random.random() for _ in range(8)]

    repro.set_global_seeds(7)
    np_b = np.random.rand(8).tolist()
    py_b = [random.random() for _ in range(8)]

    assert np_a == np_b
    assert py_a == py_b

    # And the exposed default_rng is deterministic for a given seed too.
    repro.set_global_seeds(7)
    rng_a = repro.default_rng.random(5).tolist()
    repro.set_global_seeds(7)
    rng_b = repro.default_rng.random(5).tolist()
    assert rng_a == rng_b


def test_set_global_seeds_never_raises_when_torch_import_fails(monkeypatch):
    """Even if `import torch` raises, set_global_seeds must not raise."""
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("simulated torch-absent")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    seeded = repro.set_global_seeds(99)  # must not raise
    assert isinstance(seeded, dict)
    assert seeded["python_random"] is True
    assert seeded["numpy"] is True
    # torch unavailable -> stays False, no crash.
    assert seeded["torch"] is False
    assert seeded["torch_cuda"] is False
    assert seeded["torch_deterministic"] is False


def test_write_versions_txt(tmp_path):
    ok = repro.write_versions_txt(tmp_path, seed=42)
    assert ok is True
    f = tmp_path / "versions.txt"
    assert f.exists()
    text = f.read_text(encoding="utf-8")
    assert text.strip() != ""
    assert "42" in text  # the seed
    assert "python" in text.lower()
    assert "numpy" in text.lower()


def test_write_command_log(tmp_path):
    ok = repro.write_command_log(
        tmp_path,
        config_path="cfg.yaml",
        output_dir=str(tmp_path / "out"),
        seed=314,
        extra={"analysis_mode": "rna_only"},
    )
    assert ok is True
    f = tmp_path / "command.log"
    assert f.exists()
    text = f.read_text(encoding="utf-8")
    assert text.strip() != ""
    assert "314" in text  # the seed
    assert "argv" in text
    assert "rna_only" in text


def test_write_run_metadata_both(tmp_path):
    res = repro.write_run_metadata(
        tmp_path, "cfg.yaml", tmp_path / "out", seed=5
    )
    assert res == {"versions_txt": True, "command_log": True}
    assert (tmp_path / "versions.txt").exists()
    assert (tmp_path / "command.log").exists()
