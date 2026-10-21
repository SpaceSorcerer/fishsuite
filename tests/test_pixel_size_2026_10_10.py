"""figures/ pixel size comes from the run (--px-um, else image_manifest.csv voxel_xy_nm), PX_UM only with a WARNING (2026-10-10)."""
import logging
import tokenize
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from fishsuite.figures import fov, style  # noqa: E402
from fishsuite.core.pixel_size import FALLBACK_PIXEL_UM, nm_to_um, resolve_pixel_um  # noqa: E402
from fishsuite.figures.config import PanelRun  # noqa: E402


class _Keep:
    def __init__(self):
        self.figs = {}

    def save(self, fig, stem):
        self.figs[stem] = fig


class _Run:
    arm = "VPR noDox"

    def __init__(self, px=None):
        self.saver = _Keep()
        if px is not None:
            self.pixel_size_um = lambda key=None: px

    def image_key(self, field_id):
        return f"img_{field_id}.vsi"

    def load_plane(self, key):
        z = np.zeros((96, 128), np.uint16)
        return {"miat": z, "qki": z, "dapi": z, "nucleus_labels": z.astype(np.int32), "z0": 0}


def _bar_px(run):
    fov.build_fov(run, "14", channels=("merge",))
    fig = run.saver.figs["A_FOV_VPRnoDox_field14_merge"]
    x = fig.axes[0].lines[0].get_xdata()
    matplotlib.pyplot.close(fig)
    return float(x[1] - x[0])


def test_scale_bar_length_in_px_doubles_at_0p065_um():
    at13, at065 = _bar_px(_Run(0.13)), _bar_px(_Run(0.065))
    assert at13 == pytest.approx(20 / 0.13)
    assert at065 == pytest.approx(2 * at13)


def test_fallback_is_0p13_with_a_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="fishsuite.core.pixel_size"):
        assert _bar_px(_Run()) == pytest.approx(20 / 0.13)
    assert any("falling back" in r.getMessage() for r in caplog.records)


def _panelrun(tmp_path, voxels, px_um=None):
    tmp_path.joinpath("m").mkdir(exist_ok=True)
    if voxels is not None:
        pd.DataFrame(dict(image=[f"img_{i}.vsi" for i in range(len(voxels))],
                          image_key=[f"f:/raw/x/img_{i}.vsi" for i in range(len(voxels))],
                          voxel_xy_nm=voxels)).to_csv(tmp_path / "image_manifest.csv", index=False)
    return PanelRun(tmp_path / "m", tmp_path, tmp_path, tmp_path / "o", px_um=px_um)


def test_panelrun_reads_manifest_and_cli_override_wins(tmp_path):
    run = _panelrun(tmp_path, [65.0, 65.0])
    assert run.pixel_size_um("C:/anything/IMG_1.vsi") == pytest.approx(0.065)
    assert run.pixel_size_um() == pytest.approx(0.065)
    assert _panelrun(tmp_path, [65.0, 65.0], px_um=0.13).pixel_size_um("img_0.vsi") == 0.13


def test_panelrun_mixed_manifest_needs_a_key_and_missing_manifest_gives_none(tmp_path):
    run = _panelrun(tmp_path, [65.0, 130.0])
    assert run.pixel_size_um("img_1.vsi") == pytest.approx(0.13)
    with pytest.raises(ValueError, match="--px-um"):
        run.pixel_size_um()
    (tmp_path / "image_manifest.csv").unlink()
    assert _panelrun(tmp_path, None).pixel_size_um("img_0.vsi") is None
    assert style.run_px(_panelrun(tmp_path, None), "img_0.vsi") == style.PX_UM


SRC = Path(__file__).resolve().parents[1] / "src" / "fishsuite"
# Non-pixel 0.13 literals (jitter width, layout offsets). Any other bare 0.13 number token in src/ fails.
ALLOWED_0P13 = {"core/exact_footprint_superplots.py": 1, "report/figures.py": 4}


def test_no_new_bare_0p13_outside_the_resolver():
    found = {}
    for path in SRC.rglob("*.py"):
        rel = path.relative_to(SRC).as_posix()
        if rel.startswith("core/_vendor/") or rel == "core/pixel_size.py":
            continue
        with tokenize.open(path) as handle:
            n = sum(tok.type == tokenize.NUMBER and tok.string == "0.13" for tok in tokenize.generate_tokens(handle.readline))
        if n:
            found[rel] = n
    assert found == ALLOWED_0P13, "bare 0.13 outside core/pixel_size.py; use resolve_pixel_um / FALLBACK_PIXEL_UM"


def test_resolver_order_and_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="fishsuite.core.pixel_size"):
        assert resolve_pixel_um(cli=0.065, recorded=0.13) == 0.065
        assert resolve_pixel_um(cli=None, recorded=0.065) == 0.065
        assert not caplog.records
        assert resolve_pixel_um(cli=float("nan"), recorded=0, where="x") == FALLBACK_PIXEL_UM
    assert any("falling back" in r.getMessage() for r in caplog.records)
    assert nm_to_um(65.0) == 0.065 and nm_to_um(float("nan")) is None and nm_to_um(None) is None


def test_unmatched_key_does_not_borrow_another_images_pixel_size(tmp_path, caplog):
    run = _panelrun(tmp_path, [130.0])
    assert run.pixel_size_um("img_0.vsi") == pytest.approx(0.13)
    assert run.pixel_size_um("new_image.vsi") is None
    with caplog.at_level(logging.WARNING, logger="fishsuite.core.pixel_size"):
        assert style.run_px(run, "new_image.vsi") == FALLBACK_PIXEL_UM
    assert caplog.records


def test_linked_geometry_prefers_run_pixel_size_over_the_selection_json(tmp_path):
    run = _panelrun(tmp_path, [65.0])
    assert style.run_px(run, "img_0.vsi", recorded=0.13) == pytest.approx(0.065)
    assert style.run_px(_Run(), "img_0.vsi", recorded=0.13) == 0.13


def test_d1_footprint_area_scales_with_per_row_pixel_size(monkeypatch):
    from fishsuite.figures import d_scatters
    seen = {}
    monkeypatch.setattr(d_scatters, "scat", lambda stem, x, *a, **k: seen.setdefault(stem, np.asarray(x)))
    sv = pd.DataFrame(dict(image=["a", "b"], nucleus_id=[1, 1], footprint_area_px=[10.0, 10.0],
                           miat_footprint_mean_raw=[1.0, 2.0], footprint_mean_qki=[1.0, 2.0]))
    nv = pd.DataFrame(dict(integrated_nuclear_miat=[1.0], n_miat_spots=[2], mean_nuclear_qki=[1.0]))
    d_scatters.build_d_round2(nv, sv, None, px_um=np.array([0.13, 0.065]))
    np.testing.assert_allclose(seen["D1_punctum_MIAT_vs_footprint_area_pooled_r2"], [10 * 0.13 ** 2, 10 * 0.065 ** 2])


def test_undeclared_voxel_substitution_warns_without_changing_the_value(caplog):
    from fishsuite.core.pixel_size import warn_undeclared_voxel
    with caplog.at_level(logging.WARNING, logger="fishsuite.core.pixel_size"):
        warn_undeclared_voxel("img.vsi", 65.0)
    assert "img.vsi declares no lateral pixel size; substituting 65 nm/px" in caplog.text
    for rel in ("core/modes/rna_only.py", "core/modes/rna_rna.py", "core/coloc_backfill.py"):
        src = (SRC / rel).read_text(encoding="utf-8")
        assert src.count("warn_undeclared_voxel(") == 1 and "65.0" in src, rel
