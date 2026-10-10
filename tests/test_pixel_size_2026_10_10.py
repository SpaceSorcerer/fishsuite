"""figures/ pixel size comes from the run (--px-um, else image_manifest.csv voxel_xy_nm), PX_UM only with a WARNING (2026-10-10)."""
import logging

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from fishsuite.figures import fov, style  # noqa: E402
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
    with caplog.at_level(logging.WARNING, logger="fishsuite.figures.style"):
        assert _bar_px(_Run()) == pytest.approx(20 / 0.13)
    assert any("falling back to PX_UM" in r.getMessage() for r in caplog.records)


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
