"""Qt-free checks for GUI config round-tripping, readiness modes and the runner."""
from fishsuite.gui import readiness as ready
from fishsuite.gui import state


def test_overlay_keeps_preset_keys_without_widgets():
    preset = {
        "nuclei": {"backend": "cellpose", "cellpose_device": "cuda", "n_tiles": 4},
        "z_stack": {"mode": "autofocus_maxproj", "focus_metric": "tenengrad"},
        "custom_block": {"keep": True},
    }
    cfg = state.merge_config(state.schema_defaults(), preset)
    widgets = {
        "nuclei": {"backend": "stardist", "min_area_px": 123},
        "z_stack": {"mode": "autofocus_maxproj", "start_slice": None},
        "input_file_subset": ["a.tif"],
    }
    out = state.overlay_widget_sections(cfg, widgets)
    assert out["nuclei"]["backend"] == "stardist"
    assert out["nuclei"]["min_area_px"] == 123
    assert out["nuclei"]["cellpose_device"] == "cuda"
    assert out["nuclei"]["n_tiles"] == 4
    assert out["z_stack"]["focus_metric"] == "tenengrad"
    assert out["custom_block"] == {"keep": True}
    assert out["input_file_subset"] == ["a.tif"]
    # Sections the widgets never touch still carry schema defaults.
    assert out["qc"] == state.schema_defaults()["qc"]
    # The source config is not mutated.
    assert cfg["nuclei"]["backend"] == "cellpose"


def test_overlay_fills_missing_sections_from_defaults():
    out = state.overlay_widget_sections({}, {"foci": {"enabled": False}})
    assert out["foci"]["enabled"] is False
    assert "threshold_multiplier" in out["foci"]


def test_overlay_round_trips_every_shipped_preset():
    for path in state.list_presets():
        cfg = state.merge_config(state.schema_defaults(), state.load_preset(path.stem))
        assert state.overlay_widget_sections(cfg, {}) == cfg, path.stem


def test_zstack_readiness_modes():
    z = lambda **kw: {"z_stack": kw}  # noqa: E731
    assert ready.zstack_status(z(mode="autofocus_maxproj")) == "green"
    assert ready.zstack_status(z(mode="autofocus_maxproj", start_slice=10, end_slice=5)) == "red"
    assert ready.zstack_status(z(mode="autofocus_maxproj", start_slice=2, end_slice=30)) == "green"
    assert ready.zstack_status(z(mode="single", single_slice=0)) == "green"
    assert ready.zstack_status(z(mode="single", single_slice=-1)) == "red"
    assert ready.zstack_status(z(mode="range", start_slice=1, end_slice=5)) == "yellow"


def test_if_intensity_readiness():
    cfg = {"channels": {"analysis_mode": "if_intensity"}}
    assert ready.channels_status(cfg) == "green"
    assert ready.foci_status(cfg) == "yellow"
    assert ready.pixel_coloc_status(cfg) == "yellow"


def test_conditions_discovery_is_cached(tmp_path, monkeypatch):
    from fishsuite.core import io as coreio
    for name in ("A/img_01.tif", "B/img_02.tif"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).touch()
    calls = []
    original = coreio.discover_from_conditions

    def counting(*a, **kw):
        calls.append(1)
        return original(*a, **kw)

    monkeypatch.setattr(coreio, "discover_from_conditions", counting)
    ready.clear_discovery_cache()
    cfg = {"conditions": {"mode": "subfolders", "subfolder_conditions": {"A": "A", "B": "B"}}}
    first = ready.conditions_status(cfg, input_dir=str(tmp_path))
    assert ready.conditions_status(cfg, input_dir=str(tmp_path)) == first == "green"
    assert len(calls) == 1
    cfg["conditions"]["subfolder_conditions"] = {"A": "A"}
    ready.conditions_status(cfg, input_dir=str(tmp_path))
    assert len(calls) == 2
    ready.clear_discovery_cache()
    ready.conditions_status(cfg, input_dir=str(tmp_path))
    assert len(calls) == 3


def test_stop_during_downstream_reports_failure():
    from fishsuite.gui.runner_proc import PipelineRunner
    runner = PipelineRunner()
    results = []
    runner.finished.connect(lambda ok, out: results.append(ok))
    runner._phase = "downstream"
    runner._pipeline_ok = True
    runner._stop_requested = True
    runner._on_child_exit(-9)
    assert results == [False]


class _FakeSaver:
    def __init__(self):
        self.texts = []

    def save(self, fig, stem):
        import matplotlib.pyplot as plt
        self.texts.append(" ".join(t.get_text() for t in fig.texts))
        plt.close(fig)


class _FakeRun:
    arm = "Ctrl"
    display_ranges = None

    def __init__(self, cell_line=None):
        import numpy as np
        if cell_line is not None:
            self.cell_line = cell_line
        self.saver = _FakeSaver()
        plane = np.zeros((8, 8), dtype=np.uint16)
        self._plane = {"dapi": plane, "miat": plane, "qki": plane, "z0": 2}

    def image_key(self, field_id):
        return "img"

    def load_plane(self, key):
        return self._plane


def test_fov_footer_cell_line_is_optional():
    from fishsuite.figures.config import PanelRun
    from fishsuite.figures.fov import build_fov
    assert PanelRun.__dataclass_fields__["cell_line"].default == ""
    for cell, expect in ((None, False), ("", False), ("HeLa", True)):
        run = _FakeRun(cell)
        build_fov(run, "1", channels=("DAPI",), px=0.1)
        text = run.saver.texts[0]
        assert "analysed plane z = 3" in text
        assert ("HeLa; " in text) is expect
        assert "hESC" not in text
