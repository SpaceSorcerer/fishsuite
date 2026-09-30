"""B1 XZ must use the same z orientation as B2 (render_ortho_figure): row i = plane i, z increasing downward."""
import inspect

import numpy as np

from fishsuite.figures import linked_set


def test_xz_section_row_is_plane():
    sub = np.zeros((3, 6, 4, 5)); sub[0, 1, 2, 3] = 9  # MIAT voxel at plane 1
    xz = linked_set.xz_section(sub, 2)
    assert xz["miat"].shape == (6, 5) and xz["miat"][1, 3] == 9 and xz["miat"][4, 3] == 0


def test_b1_not_flipped_and_top_down_extent():
    src = inspect.getsource(linked_set.build_linked_set)
    assert "xz_section(sub, cy)))[::-1]" not in src and "[::-1]" not in src.split("# B3")[0]
    assert "extent=[0, wum, zum, 0]" in src
