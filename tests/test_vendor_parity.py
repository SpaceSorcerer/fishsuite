"""Parity between the vendored segmentation/spot code and its original source.

``core/_vendor/segmentation/segment_image.py`` and ``core/_vendor/spots/
detect_spots.py`` were copied verbatim out of the image-analysis-pipeline
repository (see ``_vendor/PROVENANCE.md``). Results already published from this
project were produced by importing those files from that external tree, so the
move is only legitimate if it changed nothing.

Both copies are imported **into the same process** and compared. That matters:
neither cellpose nor StarDist is seeded, so comparing a fresh run against an
archived output cannot distinguish a real difference from run-to-run variation.
Comparing the two implementations side by side on identical input in one process
does isolate the move itself.

Skipped entirely when the original tree is absent, which is the normal state on
CI and on any other machine.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ORIGINAL_TREE = Path(r"F:\Image Analysis Work\image-analysis-pipeline\python")

pytestmark = pytest.mark.skipif(
    not ORIGINAL_TREE.is_dir(),
    reason=f"original source tree not present at {ORIGINAL_TREE}",
)


def _load_by_path(module_name: str, file_path: Path):
    """Import a module from an explicit file path under a private name.

    Safe for these two files specifically because both import only stdlib and
    third-party packages -- they have no intra-repository imports to resolve.
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def originals():
    seg = _load_by_path(
        "_orig_segment_image", ORIGINAL_TREE / "segmentation" / "segment_image.py"
    )
    spots = _load_by_path(
        "_orig_detect_spots", ORIGINAL_TREE / "spots" / "detect_spots.py"
    )
    return seg, spots


@pytest.fixture(scope="module")
def vendored():
    from fishsuite.core._vendor.segmentation import segment_image as seg
    from fishsuite.core._vendor.spots import detect_spots as spots

    return seg, spots


def _synthetic_nuclei(seed: int = 0) -> np.ndarray:
    """Deterministic DAPI-like field: Gaussian blobs on a low noise floor."""
    rng = np.random.default_rng(seed)
    img = rng.normal(120.0, 8.0, size=(256, 256))
    yy, xx = np.mgrid[0:256, 0:256]
    for cy, cx, sigma, amp in [
        (60, 70, 18.0, 1400.0),
        (70, 180, 15.0, 1200.0),
        (180, 60, 20.0, 1600.0),
        (190, 190, 16.0, 1100.0),
    ]:
        img += amp * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2)))
    return np.clip(img, 0, 65535).astype(np.uint16)


def _synthetic_spots(seed: int = 1) -> np.ndarray:
    """Deterministic FISH-like field: sub-diffraction puncta on a noise floor."""
    rng = np.random.default_rng(seed)
    img = rng.normal(200.0, 12.0, size=(256, 256))
    yy, xx = np.mgrid[0:256, 0:256]
    coords = rng.integers(20, 236, size=(40, 2))
    for cy, cx in coords:
        img += 2500.0 * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 1.3**2)))
    return np.clip(img, 0, 65535).astype(np.uint16)


@pytest.mark.parametrize("backend", ["otsu", "stardist", "cellpose"])
def test_run_backend_is_bitwise_identical(originals, vendored, backend):
    orig_seg, _ = originals
    vend_seg, _ = vendored
    img = _synthetic_nuclei()

    kwargs = dict(
        min_area=250,
        max_area=1e12,
        prob_threshold=0.5,
        diameter=30.0,
    )
    if backend == "cellpose":
        kwargs.update(cellpose_model_type="cpsam", cellpose_device="cpu")

    try:
        expected = orig_seg.run_backend(backend, img, **kwargs)
    except Exception as exc:  # backend's own dependency missing in this env
        pytest.skip(f"{backend} unavailable in this environment: {exc}")
    actual = vend_seg.run_backend(backend, img, **kwargs)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert np.array_equal(actual, expected), (
        f"{backend}: vendored labels differ from the original implementation"
    )


def test_detect_spots_bigfish_is_bitwise_identical(originals, vendored):
    _, orig_spots = originals
    _, vend_spots = vendored
    img = _synthetic_spots()
    kw = dict(voxel_size_nm=65.0, spot_radius_nm=150.0)

    try:
        exp_coords, exp_thr = orig_spots.detect_spots_bigfish(img, **kw)
    except Exception as exc:
        pytest.skip(f"big-fish unavailable in this environment: {exc}")
    act_coords, act_thr = vend_spots.detect_spots_bigfish(img, **kw)

    assert act_thr == exp_thr, "auto threshold differs"
    assert np.array_equal(np.asarray(act_coords), np.asarray(exp_coords)), (
        "bigfish spot coordinates differ from the original implementation"
    )


def test_detect_spots_log_is_bitwise_identical(originals, vendored):
    _, orig_spots = originals
    _, vend_spots = vendored
    img = _synthetic_spots()
    kw = dict(threshold=0.05, spot_radius_px=2.5)

    exp_coords, exp_thr = orig_spots.detect_spots_log(img, **kw)
    act_coords, act_thr = vend_spots.detect_spots_log(img, **kw)

    assert act_thr == exp_thr, "LoG threshold differs"
    assert np.array_equal(np.asarray(act_coords), np.asarray(exp_coords)), (
        "LoG spot coordinates differ from the original implementation"
    )


def test_vendored_checksums_match_provenance():
    """Every vendored file still hashes to what PROVENANCE.md records.

    Guards against an accidental edit inside ``_vendor/`` -- the checksum table
    is what lets a reader verify the algorithm is unchanged.
    """
    import hashlib
    import re

    vendor_dir = Path(__file__).resolve().parents[1] / "src" / "fishsuite" / "core" / "_vendor"
    provenance = (vendor_dir / "PROVENANCE.md").read_text(encoding="utf-8")

    rows = re.findall(
        r"^\|\s*`([^`]+)`\s*\|[^|]*\|\s*\d+\s*\|\s*`([0-9a-f]{64})`\s*\|",
        provenance,
        flags=re.MULTILINE,
    )
    assert rows, "no checksum rows parsed out of PROVENANCE.md"

    mismatches = []
    for rel, want in rows:
        path = vendor_dir / rel
        assert path.is_file(), f"PROVENANCE.md lists a missing file: {rel}"
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != want:
            mismatches.append(f"{rel}: recorded {want[:16]}..., found {got[:16]}...")
    assert not mismatches, "vendored files edited without updating PROVENANCE.md:\n" + "\n".join(
        mismatches
    )
