"""The raw-intensity legend never covers a trace or bound line (2026-09-28)."""
import numpy as np
import pytest

from test_ortho_physical_layout import _render


@pytest.mark.parametrize('include_dapi', [False, True])
def test_raw_legend_clear_of_data(include_dapi):
    rng = np.random.default_rng(0)
    stack = rng.integers(0, 60, (3, 21, 61, 61)).astype(float)
    stack[:, :, 30, 55] = 5000  # tall peak near the right edge, where the legend sits
    fig, axes = _render(stack, (10, 30, 30), 28, .13, .21, include_dapi=include_dapi,
                        dapi_display_level=(0, 100), line_endpoints=((30, 5), (30, 58)),
                        qki_min=50, miat_min=40, nucleus_mask=np.ones((61, 61), bool))
    raw = axes['raw']
    r = fig.canvas.get_renderer()
    box = raw.get_legend().get_window_extent(r)
    for line in raw.get_lines():
        xy = line.get_transform().transform(line.get_xydata())
        inside = (xy[:, 0] > box.x0) & (xy[:, 0] < box.x1) & (xy[:, 1] > box.y0) & (xy[:, 1] < box.y1)
        assert not inside.any(), line.get_label()
