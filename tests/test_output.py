"""Smoke tests for fishsuite.core.output — rendering, scale bar, mask writers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_dummy_image(h=128, w=128, seed=0):
    rng = np.random.default_rng(seed)
    dapi = (rng.normal(800, 100, size=(h, w)).clip(0, 4095)).astype(np.uint16)
    rna = (rng.normal(120, 30, size=(h, w)).clip(0, 4095)).astype(np.uint16)
    # Insert a few bright spots
    for y, x in [(40, 40), (80, 60), (90, 100)]:
        rna[y - 1:y + 2, x - 1:x + 2] = 4000
    # Two nuclei labels
    labels = np.zeros((h, w), dtype=np.uint16)
    yy, xx = np.ogrid[:h, :w]
    labels[(yy - 40) ** 2 + (xx - 40) ** 2 <= 18 ** 2] = 1
    labels[(yy - 80) ** 2 + (xx - 80) ** 2 <= 22 ** 2] = 2
    return dapi, rna, labels


def test_apply_lut_yellow_zero_blue_channel():
    from fishsuite.core.output import apply_lut
    g = np.array([[0, 128, 255]], dtype=np.uint8)
    rgb = apply_lut(g, 1.0, 1.0, 0.0, floor=0, ceil=255)
    assert rgb.shape == (1, 3, 3)
    # Yellow = R + G, no B
    assert rgb[0, 2, 2] == 0  # bright cell, blue channel zero
    assert rgb[0, 2, 0] > 0.99 and rgb[0, 2, 1] > 0.99


def test_apply_lut_blue_zero_red_channel():
    from fishsuite.core.output import apply_lut
    g = np.array([[0, 128, 255]], dtype=np.uint8)
    rgb = apply_lut(g, 0.0, 0.3, 1.0, floor=0, ceil=255)
    # Blue-dominant: red=0, green=0.3*norm, blue=norm
    assert rgb[0, 2, 0] == 0
    assert rgb[0, 2, 2] > 0.99


def test_burn_scale_bar_produces_white_pixels():
    from fishsuite.core.output import burn_scale_bar
    rgb = np.zeros((400, 600, 3), dtype=np.uint8)
    out = burn_scale_bar(rgb, voxel_xy_nm=65.0, bar_um=10.0, label=False)
    # There should now be white pixels somewhere in the bottom strip.
    # The bar sits inside the bottom 'margin_px + height_px' band.
    assert out[-50:, -250:, :].sum() > 0


def test_render_all_in_one_qc_returns_rgb_uint8():
    from fishsuite.core.output import render_all_in_one_qc
    dapi, rna, labels = _make_dummy_image()
    spots = pd.DataFrame([{"x_px": 40, "y_px": 40}, {"x_px": 60, "y_px": 80}])
    rgb = render_all_in_one_qc(dapi, rna, labels, spots, voxel_xy_nm=65.0)
    assert rgb.dtype == np.uint8
    assert rgb.shape == (128, 128, 3)
    # DAPI is blue, so blue channel should be > 0 over nuclei
    nuc_mask = labels > 0
    assert rgb[..., 2][nuc_mask].max() > 0


def test_save_label_tiff_roundtrip(tmp_path):
    from fishsuite.core.output import save_label_tiff
    labels = np.array([[0, 1, 1], [2, 2, 0]], dtype=np.uint16)
    p = tmp_path / "labels.tif"
    save_label_tiff(labels, p)
    assert p.exists()
    import tifffile
    rt = tifffile.imread(str(p))
    assert rt.dtype == np.uint16
    assert np.array_equal(rt, labels)


def test_save_walkthrough_bundle_writes_six(tmp_path):
    from fishsuite.core.output import save_walkthrough_bundle
    dapi, rna, labels = _make_dummy_image()
    dapi_mask = (dapi > dapi.mean()).astype(np.uint8) * 255
    rna_mask = (rna > rna.mean() * 1.5).astype(np.uint8) * 255
    paths = save_walkthrough_bundle(
        tmp_path / "walk", "img01",
        dapi=dapi, rna=rna, dapi_mask=dapi_mask,
        labels=labels, rna_pos_mask=rna_mask,
        voxel_xy_nm=65.0,
    )
    assert len(paths) == 6
    for p in paths:
        assert p.exists() and p.suffix == ".png"


def test_save_publication_images_bundle_writes_six(tmp_path):
    from fishsuite.core.output import save_publication_images_bundle
    dapi, rna, _ = _make_dummy_image()
    paths = save_publication_images_bundle(
        tmp_path, "img01", dapi, rna, voxel_xy_nm=65.0,
    )
    # 3 outputs (DAPI, RNA, merge) x 2 formats (PNG + TIF) = 6
    assert len(paths) == 6
    suffixes = sorted(p.suffix for p in paths)
    assert suffixes.count(".png") == 3
    assert suffixes.count(".tif") == 3


def test_sanitize_condition_for_filename():
    from fishsuite.core.output import sanitize_condition_for_filename as s
    # Spaces -> underscore
    assert s("NT ASO") == "NT_ASO"
    # Hyphens -> underscore
    assert s("Sec-Only") == "Sec_Only"
    assert s("KD ASO") == "KD_ASO"
    # Slashes / quotes stripped
    assert s("MIAT OE/KD") == "MIAT_OE_KD"
    assert s('"NT"') == "NT"
    # Empty / None
    assert s(None) == ""
    assert s("") == ""
    assert s("   ") == ""
    # Collapsed runs + trim
    assert s("--abc---def--") == "abc_def"


def test_sec_only_does_not_pollute_batch_cache(tmp_path):
    """Sec-only images must consult-but-not-update the running-max contrast
    cache, so a dim no-probe control rendered AFTER bright real images
    inherits the real-image (floor, ceil) and looks correctly dim — and
    the sec-only image's own percentiles never get folded back into the
    cache (would lower it relative to the real images)."""
    from fishsuite.core import output as _out
    _out.reset_batch_disp_range_cache()
    # Bright real image first → populates the cache.
    bright = np.full((64, 64), 4000, dtype=np.uint16)
    bright[10:20, 10:20] = 4090  # a brighter patch to push p99.95
    _out.save_publication_images_bundle(
        tmp_path / "real", "real", bright, bright, voxel_xy_nm=65.0,
    )
    floor_after_real, ceil_after_real = _out.get_batch_disp_range("rna")
    assert ceil_after_real is not None and ceil_after_real > 3000
    # Dim sec-only image second — must NOT shrink the cache.
    dim = np.full((64, 64), 50, dtype=np.uint16)
    _out.save_publication_images_bundle(
        tmp_path / "sec", "sec", dim, dim, voxel_xy_nm=65.0, sec_only=True,
    )
    floor_after_sec, ceil_after_sec = _out.get_batch_disp_range("rna")
    # Cache must be UNCHANGED by the sec-only call.
    assert floor_after_sec == floor_after_real
    assert ceil_after_sec == ceil_after_real


def test_scale_bar_text_has_black_outline_pixels():
    """The bold-white-with-black-outline label needs some black pixels in
    the label region to ensure the outline is actually being drawn."""
    from fishsuite.core.output import burn_scale_bar
    # Mid-gray background to make black outline visible
    rgb = np.full((400, 600, 3), 128, dtype=np.uint8)
    out = burn_scale_bar(rgb, voxel_xy_nm=65.0, bar_um=10.0, label=True, font_px=28)
    # Look in the strip above the bar where the label sits — should have
    # white text pixels AND some black/near-black outline pixels (the
    # background is 128, so any pixel <30 is from the outline).
    strip = out[-90:-40, -200:, :]
    near_white = ((strip > 240).all(axis=-1)).sum()
    near_black = ((strip < 30).all(axis=-1)).sum()
    assert near_white > 5, "expected white text pixels above the bar"
    assert near_black > 5, "expected black outline pixels around the text"


def test_lut_name_to_weights():
    """LUT name -> RGB weights lookup."""
    from fishsuite.core.output import lut_name_to_weights
    assert lut_name_to_weights("yellow") == (1.0, 1.0, 0.0)
    assert lut_name_to_weights("Yellow") == (1.0, 1.0, 0.0)  # case-insensitive
    assert lut_name_to_weights("blue") == (0.0, 0.3, 1.0)
    assert lut_name_to_weights("cyan") == (0.0, 1.0, 1.0)
    assert lut_name_to_weights("magenta") == (1.0, 0.0, 1.0)
    assert lut_name_to_weights("green") == (0.0, 1.0, 0.0)
    assert lut_name_to_weights("red") == (1.0, 0.0, 0.0)
    assert lut_name_to_weights("orange") == (1.0, 0.5, 0.0)
    # Unknown name falls back to gray
    assert lut_name_to_weights("nonexistent") == (1.0, 1.0, 1.0)
    # Empty / None returns the fallback param
    assert lut_name_to_weights(None, (0.5, 0.5, 0.5)) == (0.5, 0.5, 0.5)
    assert lut_name_to_weights("") == (1.0, 1.0, 1.0)


def test_publication_image_lut_selection_changes_color(tmp_path):
    """Setting rna_lut='red' should produce a red render. Verified by calling
    apply_lut directly with the weights returned by lut_name_to_weights —
    keeps the test independent of contrast / scale-bar code paths."""
    import numpy as np
    from fishsuite.core.output import apply_lut, lut_name_to_weights
    rna = np.full((32, 32), 4000, dtype=np.uint16)
    yw = lut_name_to_weights("yellow")
    layer_y = apply_lut(rna, yw[0], yw[1], yw[2], floor=0, ceil=4000)
    rw = lut_name_to_weights("red")
    layer_r = apply_lut(rna, rw[0], rw[1], rw[2], floor=0, ceil=4000)
    # apply_lut returns float [0, 1] HxWx3.
    assert layer_y[..., 0].mean() > 0.9   # yellow has R high
    assert layer_y[..., 1].mean() > 0.9   # and G high
    assert layer_y[..., 2].mean() < 0.1   # and no B
    assert layer_r[..., 0].mean() > 0.9   # red has R high
    assert layer_r[..., 1].mean() < 0.05  # and no G
    assert layer_r[..., 2].mean() < 0.05  # and no B


def test_publication_image_lut_changes_output_filename(tmp_path):
    """When rna_lut='red' is passed, the output filename should reflect the
    LUT name (__RNA_red instead of __RNA_yellow)."""
    import numpy as np
    from fishsuite.core import output as _out
    _out.reset_batch_disp_range_cache()
    dapi = np.full((32, 32), 100, dtype=np.uint16)
    rna = np.full((32, 32), 2000, dtype=np.uint16)
    paths = _out.save_publication_images_bundle(
        tmp_path, "img", dapi, rna, voxel_xy_nm=65.0, rna_lut="red",
    )
    names = [p.name for p in paths]
    assert any("__RNA_red.png" in n for n in names), f"no __RNA_red.png in {names}"
    assert not any("__RNA_yellow.png" in n for n in names), (
        f"unexpected __RNA_yellow.png with rna_lut='red': {names}"
    )
