"""Per-set display ranges for figures/ (2026-10-10): the default LV path must render byte-identically."""
import hashlib
import sys

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from fishsuite.figures import fov, style  # noqa: E402

# sha256 of the RGBA buffer of build_fov(..., channels=("merge",)) on the synthetic plane below,
# recorded on main eb4dcca (before display ranges became a parameter); matplotlib 3.10.9, fishproc env.
MAIN_EB4DCCA_MERGE_RGBA_SHA256 = "52042c23981ae0ae3b9eee9a6eda763a4bcf06f1f953fd8d227a1a87d3bec7ef"


class _Saver:
    def __init__(self):
        self.digests = {}

    def save(self, fig, stem):
        fig.canvas.draw()
        self.digests[stem] = hashlib.sha256(np.asarray(fig.canvas.buffer_rgba()).tobytes()).hexdigest()
        matplotlib.pyplot.close(fig)


class _Run:
    arm = "VPR noDox"

    def __init__(self, **kw):
        self.saver = _Saver()
        self.__dict__.update(kw)

    def image_key(self, field_id):
        return f"img_{field_id}.vsi"

    def load_plane(self, key):
        rng = np.random.default_rng(7)
        return {"miat": rng.integers(0, 4000, (96, 128)).astype(np.uint16),
                "qki": rng.integers(0, 5000, (96, 128)).astype(np.uint16),
                "dapi": rng.integers(0, 10000, (96, 128)).astype(np.uint16),
                "nucleus_labels": np.zeros((96, 128), np.int32), "z0": 3}


def _merge_digest(**kw):
    style.apply_style()
    run = _Run(**kw)
    fov.build_fov(run, "14", channels=("merge",))
    return run.saver.digests["A_FOV_VPRnoDox_field14_merge"]


def _recording_environment():
    from matplotlib import font_manager
    try:
        font_manager.findfont("Arial", fallback_to_default=False)
    except ValueError:
        return False
    return sys.platform == "win32" and matplotlib.__version__ == "3.10.9"


@pytest.mark.skipif(not _recording_environment(),
                    reason="digest recorded on eb4dcca in the fishproc env (win32, matplotlib 3.10.9, Arial)")
def test_default_lv_render_is_byte_identical_to_main():
    assert _merge_digest() == MAIN_EB4DCCA_MERGE_RGBA_SHA256


def test_custom_ranges_change_render_and_footer_and_default_note_is_unchanged():
    lv = style.parse_display_ranges("MIAT=250:1700,QKI=420:3000,DAPI=600:5000")
    assert lv == {"miat": (250.0, 1700.0), "qki": (420.0, 3000.0), "dapi": (600.0, 5000.0)}
    assert _merge_digest(display_ranges=lv) != _merge_digest()
    assert style.display_note(style.LV) == "Display: MIAT 500-2250, QKI 1050-3746, DAPI 607-9000 raw a.u."
    assert style.display_note(lv) == "Display: MIAT 250-1700, QKI 420-3000, DAPI 600-5000 raw a.u."


def test_preset_key_and_empty_spec_resolve_to_locked_lv():
    assert style.parse_display_ranges(None) is None
    assert style.parse_display_ranges("fig1_v2") == style.LV
    assert _merge_digest(display_ranges=style.parse_display_ranges("fig1_v2")) == _merge_digest()


@pytest.mark.parametrize("bad", ["MIAT=250:1700,QKI=420:3000", "MIAT=1700:250,QKI=420:3000,DAPI=600:5000",
                                 "MIAT=a:b,QKI=420:3000,DAPI=600:5000", "nope"])
def test_bad_display_ranges_are_refused(bad):
    with pytest.raises(ValueError):
        style.parse_display_ranges(bad)


def test_default_render_unchanged_for_a_run_whose_pixel_size_is_0p13():
    if not _recording_environment():
        pytest.skip("digest recorded on eb4dcca in the fishproc env (win32, matplotlib 3.10.9, Arial)")
    assert _merge_digest(pixel_size_um=lambda key=None: 0.13) == MAIN_EB4DCCA_MERGE_RGBA_SHA256


def test_cli_passes_display_ranges_to_panelrun(monkeypatch, tmp_path):
    from click.testing import CliRunner
    from fishsuite import cli
    from fishsuite.figures import gallery
    seen = {}
    monkeypatch.setattr(gallery, "build_gallery",
                        lambda run, **kw: seen.update(lv=run.display_ranges, px=run.px_um) or {})
    args = ["fig-panels", "--menu", str(tmp_path), "--backfill", str(tmp_path), "--run", str(tmp_path),
            "--out", str(tmp_path / "o"), "--display-ranges", "MIAT=250:1700,QKI=420:3000,DAPI=600:5000"]
    res = CliRunner().invoke(cli.cli, args)
    assert res.exit_code == 0, res.output
    assert seen["lv"]["qki"] == (420.0, 3000.0) and seen["px"] is None
    assert CliRunner().invoke(cli.cli, args + ["--px-um", "0.065"]).exit_code == 0 and seen["px"] == 0.065
    res = CliRunner().invoke(cli.cli, args[:-1] + ["MIAT=1:0"])
    assert res.exit_code == 2 and "--display-ranges" in res.output
