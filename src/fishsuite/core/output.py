"""Per-image output rendering + CSV writing for fishsuite.

Implements the Fiji-pipeline output layout in pure Python:
    output_dir/
      per_image_summary.csv      (master, one row per image)
      nuclei_metrics.csv         (master, one row per nucleus)
      spot_metrics.csv           (master, one row per spot)
      cell_morphology.csv        (master, one row per nucleus)
      thresholds.csv             (master, one row per image)
      run_config.json
      analysis_summary.xlsx
      qc_overlays/               <stem>__qc_dapi_rna_nuclei_spots.png
                                 <stem>__qc_nuclei_on_dapi.png
      publication_images/        <stem>__DAPI_blue.png/.tif
                                 <stem>__RNA_yellow.png/.tif
                                 <stem>__merge_DAPI_RNA.png/.tif
      pipeline_walkthrough/      <stem>__step01_DAPI_raw.png
                                 <stem>__step02_DAPI_mask.png
                                 <stem>__step03_nuclei_outlines_on_DAPI.png
                                 <stem>__step04_RNA_raw_yellow.png
                                 <stem>__step05_RNA_threshold_yellow.png
                                 <stem>__step06_RNA_threshold_on_signal.png
      nuclei_popouts/            <stem>__representative_nuc_NNN_spotsM.png
      masks/                     <stem>__nuclei_label_mask.tif
                                 <stem>__spot_mask.tif
                                 <stem>__thresholds.csv     (per-image)
      per_image_csv/             <stem>__nuclei_metrics.csv
                                 <stem>__spot_metrics.csv

Colors / LUTs match the Fiji pipeline:
    DAPI  = Blue   (0.0, 0.3, 1.0) weights, dapi_floor=p10, dapi_ceil=p99.9
    RNA   = Yellow (1.0, 1.0, 0.0) weights, rna_floor=p95, rna_ceil=p99.95
                                            (per H9 100x preset DISP_*)
    Nuclei outlines = white, 2 px
    Spots = yellow ovals (when on the all-in-one composite), white in popouts

This module is intentionally framework-light: takes numpy arrays + simple
dicts in, writes PNG/TIF/CSV out. No bioio, no skimage in the hot path
beyond contour finding + IO.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional, Sequence, Tuple, List, Dict, Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants — match Fiji-pipeline H9 100x defaults
# ---------------------------------------------------------------------------

DAPI_FLOOR_PCT = 10.0
DAPI_CEIL_PCT = 99.9
# Brian's H9 preset: DISP_FLOOR_PERCENTILE=95.0, DISP_CEIL_PERCENTILE=99.95
RNA_FLOOR_PCT = 95.0
RNA_CEIL_PCT = 99.95
# Magenta/Protein channel (Fiji DISP_*_PERCENTILE for ab channel batch render)
AB_FLOOR_PCT = 80.0
AB_CEIL_PCT = 99.5
# Second RNA channel (rna_rna mode) — cyan (0, 1, 1). Use the same default
# stretch as the primary RNA channel since both are FISH spots with similar
# intensity distributions. Batch-coordinated under batch_key="rna2".
RNA2_FLOOR_PCT = 95.0
RNA2_CEIL_PCT = 99.95

SCALEBAR_UM = 50.0  # all-in-one + publication + walkthrough
POPOUT_SCALEBAR_UM = 5.0

# Scale-bar geometry — matches Fiji's add_scale_bar_50um exactly
# (Coloc_Analysis.py line 1687-1688: "width=50 height=12 font=28 color=White
#  background=None location=[Lower Right] bold overlay").
SCALEBAR_HEIGHT_PX = 12
SCALEBAR_FONT_PX = 28

# Stroke widths
NUC_OUTLINE_WIDTH_PX = 2
SPOT_MARKER_RADIUS_PX = 3  # base size; scales by spot_diameter when available


def sanitize_condition_for_filename(condition: Optional[str]) -> str:
    """Make a condition label safe to embed in an output filename.

    Replaces whitespace / hyphens with single underscores, strips disallowed
    filesystem characters (quotes, slashes, backslashes, colons, asterisks,
    question marks, angle brackets, pipes), collapses repeated underscores,
    and trims leading/trailing underscores. Returns the empty string for
    None / blank / unlabeled inputs (caller decides whether to skip the
    extra ``__<condition>__`` segment).

    Examples:
        "NT ASO"      -> "NT_ASO"
        "Sec-Only"    -> "Sec_Only"
        "KD ASO"      -> "KD_ASO"
        "MIAT OE/KD"  -> "MIAT_OE_KD"
    """
    if condition is None:
        return ""
    s = str(condition).strip()
    if not s:
        return ""
    # Strip disallowed filename chars (quotes etc) entirely
    for bad in ('"', "'", ":", "*", "?", "<", ">", "|"):
        s = s.replace(bad, "")
    # Normalize separators -> underscore. Slashes, backslashes, whitespace,
    # hyphens, dots, and commas all map to "_" so e.g. "MIAT OE/KD" becomes
    # "MIAT_OE_KD" rather than "MIAT_OEKD".
    out_chars: List[str] = []
    for ch in s:
        if ch.isspace() or ch in ("-", ".", ",", "/", "\\"):
            out_chars.append("_")
        else:
            out_chars.append(ch)
    s = "".join(out_chars)
    # Collapse runs of underscores
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


# ---------------------------------------------------------------------------
# Named-LUT → RGB-weights map. Mirrors Fiji's _LUT_NAME_TO_RGB_WEIGHTS
# in Coloc_Analysis.py. Used by the publication-image renderer + GUI
# color-picker so the user can swap RNA1's pseudo-color from yellow to
# (e.g.) orange in YAML config or via a GUI combo.
# ---------------------------------------------------------------------------
_LUT_NAME_TO_RGB_WEIGHTS: Dict[str, Tuple[float, float, float]] = {
    "blue":     (0.0, 0.3, 1.0),
    "yellow":   (1.0, 1.0, 0.0),
    "magenta":  (1.0, 0.0, 1.0),
    "cyan":     (0.0, 1.0, 1.0),
    "green":    (0.0, 1.0, 0.0),
    "red":      (1.0, 0.0, 0.0),
    "orange":   (1.0, 0.5, 0.0),
    "fire":     (1.0, 0.4, 0.0),
    "gray":     (1.0, 1.0, 1.0),
    "grays":    (1.0, 1.0, 1.0),
    "white":    (1.0, 1.0, 1.0),
}

LUT_NAMES: Tuple[str, ...] = (
    "blue", "yellow", "cyan", "magenta", "green",
    "red", "orange", "gray", "fire",
)


def lut_name_to_weights(
    name: Optional[str],
    fallback: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Tuple[float, float, float]:
    """Resolve a LUT color name to (r_w, g_w, b_w) weights.

    Case-insensitive. Unknown names fall back to gray (so a typo still
    renders the channel). Returns the ``fallback`` triple when ``name``
    is None / empty.
    """
    if not name:
        return fallback
    key = str(name).strip().lower()
    return _LUT_NAME_TO_RGB_WEIGHTS.get(key, fallback)


def sanitize_label_for_filename(label: Optional[str], default: str = "") -> str:
    """Sanitize a user-supplied channel label for safe filename embedding.

    Reuses the same character rules as ``sanitize_condition_for_filename``
    so condition-labels and channel-labels look consistent in output
    filenames. If the label is None/empty after sanitization, returns the
    ``default`` (which may itself be the empty string).

    Examples:
        "RNA1"        -> "RNA1"
        "MIAT-Cy5"    -> "MIAT_Cy5"
        "QKI Ab"      -> "QKI_Ab"
        ""            -> default
    """
    s = sanitize_condition_for_filename(label)
    return s if s else default


# ---------------------------------------------------------------------------
# Batch contrast coordination — mirrors Fiji's
#   _BATCH_DISP_CEIL_CACHE / _BATCH_DISP_FLOOR_CACHE in Coloc_Analysis.py.
# Each batch_key ('rna', 'ab', optionally 'dapi') accumulates the running
# max-of-floors and max-of-ceilings across all images. Pipeline runners
# should call reset_batch_disp_range_cache() once at the start of each run
# so state from a prior run does not leak in. This is what makes the
# multi-image publication contrast consistent — dim secondary-only fields
# inherit the brighter real-image ceiling and render correctly dark instead
# of auto-stretching their own background to fill the byte range.
# ---------------------------------------------------------------------------
_BATCH_DISP_FLOOR_CACHE: Dict[str, float] = {}
_BATCH_DISP_CEIL_CACHE: Dict[str, float] = {}


def reset_batch_disp_range_cache() -> None:
    """Clear the batch (floor, ceil) caches. Call once per pipeline run."""
    global _BATCH_DISP_FLOOR_CACHE, _BATCH_DISP_CEIL_CACHE
    _BATCH_DISP_FLOOR_CACHE = {}
    _BATCH_DISP_CEIL_CACHE = {}


def update_batch_disp_range(
    batch_key: str, floor: float, ceil: float
) -> Tuple[float, float]:
    """Update the batch caches with this image's (floor, ceil), taking the
    running max in both directions. Returns the merged (floor, ceil) the
    caller should use for the current render. Mirrors Fiji
    Coloc_Analysis.update_batch_disp_range exactly."""
    cached_floor = _BATCH_DISP_FLOOR_CACHE.get(batch_key)
    cached_ceil = _BATCH_DISP_CEIL_CACHE.get(batch_key)
    new_floor = max(floor, cached_floor) if cached_floor is not None else floor
    new_ceil = max(ceil, cached_ceil) if cached_ceil is not None else ceil
    _BATCH_DISP_FLOOR_CACHE[batch_key] = new_floor
    _BATCH_DISP_CEIL_CACHE[batch_key] = new_ceil
    return (new_floor, new_ceil)


def get_batch_disp_range(batch_key: str) -> Tuple[Optional[float], Optional[float]]:
    """Return the current (floor, ceil) in the batch cache without modifying
    it. Used to render sec-only / no-primary-probe controls at the same
    contrast scale as real images WITHOUT letting their dimmer background
    pollute (or pull down) the running-max."""
    return (_BATCH_DISP_FLOOR_CACHE.get(batch_key),
            _BATCH_DISP_CEIL_CACHE.get(batch_key))


def _resolve_lut_range(
    gray: np.ndarray,
    floor_pct: float,
    ceil_pct: float,
    *,
    batch_key: Optional[str] = None,
    is_sec_only: bool = False,
) -> Tuple[float, float]:
    """Return (floor, ceil) for this image.

    If ``batch_key`` is supplied and ``is_sec_only`` is False, the range is
    merged into the running batch-max cache (Fiji parity).

    If ``is_sec_only`` is True, the cache is READ but NOT UPDATED. The
    returned (floor, ceil) is the current batch-max range (if any image has
    already populated it), so the sec-only image renders at the same
    contrast scale as the real-signal images. Its dim autofluorescence does
    not get auto-stretched up to fill the byte range, so it correctly
    appears DIM in publication renders. This matches Fiji's pub-images
    batch-contrast logic which skips sec-only images entirely
    (``Coloc_Analysis.compute_pub_images_batch_contrast`` lines 3990-3992).
    """
    f = _percentile(gray, floor_pct)
    c = _percentile(gray, ceil_pct)
    if batch_key is not None:
        if is_sec_only:
            # Consult the cache but do not update it. If the cache is empty
            # (this is the first image processed and it happens to be
            # sec-only) fall back to this image's own percentiles so we
            # still produce a valid render — the running max from later
            # real images cannot retroactively re-render past frames anyway.
            cached_floor, cached_ceil = get_batch_disp_range(batch_key)
            if cached_floor is not None and cached_ceil is not None:
                return cached_floor, cached_ceil
            return f, c
        f, c = update_batch_disp_range(batch_key, f, c)
    return f, c


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def _percentile(img: np.ndarray, pct: float) -> float:
    """np.percentile but tolerant of all-zero / NaN images."""
    arr = np.asarray(img)
    if arr.size == 0:
        return 0.0
    finite = arr[np.isfinite(arr)] if np.issubdtype(arr.dtype, np.floating) else arr
    if finite.size == 0:
        return 0.0
    return float(np.percentile(finite, pct))


def apply_lut(
    gray: np.ndarray,
    r_w: float, g_w: float, b_w: float,
    *,
    floor: Optional[float] = None,
    ceil: Optional[float] = None,
) -> np.ndarray:
    """Stretch a grayscale image to [0, 1] using floor/ceil and tint with RGB weights.

    Returns an HxWx3 float32 array in [0, 1].
    """
    g = np.asarray(gray, dtype=np.float32)
    if floor is None:
        floor = float(g.min())
    if ceil is None or ceil <= floor:
        ceil = float(g.max())
        if ceil <= floor:
            ceil = floor + 1.0
    span = ceil - floor
    norm = np.clip((g - floor) / span, 0.0, 1.0)
    rgb = np.zeros((g.shape[0], g.shape[1], 3), dtype=np.float32)
    rgb[..., 0] = norm * float(r_w)
    rgb[..., 1] = norm * float(g_w)
    rgb[..., 2] = norm * float(b_w)
    return rgb


def merge_rgb_additive(layers: Sequence[np.ndarray]) -> np.ndarray:
    """Additively merge a list of HxWx3 RGB float arrays into one, clipped to [0,1]."""
    if not layers:
        raise ValueError("no layers to merge")
    out = np.zeros_like(layers[0], dtype=np.float32)
    for ly in layers:
        out += ly
    return np.clip(out, 0.0, 1.0)


def _to_uint8(rgb_f: np.ndarray) -> np.ndarray:
    """Convert HxWx3 float [0,1] (or HxW grayscale) -> uint8 RGB."""
    if rgb_f.ndim == 2:
        arr = rgb_f
        if arr.dtype != np.float32 and arr.dtype != np.float64:
            arr = arr.astype(np.float32)
        # Stretch to 0..1 if not already there
        lo, hi = float(arr.min()), float(arr.max())
        if hi > 1.5 or lo < -0.01:
            arr = (arr - lo) / (hi - lo + 1e-12)
        arr = np.clip(arr, 0, 1)
        u = (arr * 255.0).astype(np.uint8)
        return np.stack([u, u, u], axis=-1)
    return (np.clip(rgb_f, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


# ---------------------------------------------------------------------------
# Scale bar
# ---------------------------------------------------------------------------

def _load_bold_font(font_px: int):
    """Try a sequence of bold sans-serif fonts and return a PIL ImageFont.

    Order: Windows arialbd.ttf, generic 'arialbd', Liberation Sans Bold,
    DejaVu Sans Bold, then any non-bold Arial as a fallback, then PIL's
    default bitmap font. Cached implicitly by PIL across calls.
    """
    from PIL import ImageFont
    candidates = [
        "arialbd.ttf",                                  # Windows Arial Bold
        r"C:\Windows\Fonts\arialbd.ttf",                # Windows abs path
        "Arial Bold.ttf",
        "LiberationSans-Bold.ttf",                      # Linux
        "DejaVuSans-Bold.ttf",                          # most Linux distros
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "arial.ttf",                                    # last-resort non-bold
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, font_px)
        except Exception:
            continue
    return ImageFont.load_default()


def burn_scale_bar(
    rgb_u8: np.ndarray,
    voxel_xy_nm: float,
    *,
    bar_um: float = 50.0,
    height_px: int = 12,
    margin_px: int = 30,
    color: Tuple[int, int, int] = (255, 255, 255),
    label: bool = True,
    font_px: int = 28,
    text_outline_px: int = 2,
    text_outline_color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Burn a horizontal scale bar in the bottom-right corner of an RGB uint8 image.

    Matches the Fiji pipeline's "Scale Bar..." command output (lower-right,
    white, bold, font 28, height 12, with a numeric label) and adds a thin
    black text outline so the label stays readable over both dark background
    and bright-yellow signal regions (Fiji's renders show the bold label
    sitting just above the bar; the outline replaces the Fiji `bold overlay`
    flag's anti-aliased-on-dark visual).

    Returns a new RGB uint8 array (does not mutate input).
    """
    if rgb_u8.ndim != 3 or rgb_u8.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB uint8, got {rgb_u8.shape} {rgb_u8.dtype}")
    if voxel_xy_nm is None or voxel_xy_nm <= 0:
        voxel_xy_nm = 65.0  # H9 100x fallback
    um_per_px = float(voxel_xy_nm) / 1000.0
    bar_px = int(round(float(bar_um) / um_per_px))
    h, w = rgb_u8.shape[:2]
    bar_px = min(bar_px, w - 2 * margin_px)
    if bar_px <= 0:
        return rgb_u8.copy()
    out = rgb_u8.copy()
    x1 = w - margin_px
    x0 = x1 - bar_px
    y1 = h - margin_px
    y0 = y1 - height_px
    out[y0:y1, x0:x1] = np.array(color, dtype=np.uint8)
    if label:
        try:
            from PIL import Image, ImageDraw
            img = Image.fromarray(out)
            draw = ImageDraw.Draw(img)
            font = _load_bold_font(font_px)
            # Use the micro-mu sign so the label reads "50 µm" like Fiji's
            # output (java prints "50 \xb5m" via the Scale Bar plugin).
            txt = f"{int(round(bar_um))} µm"
            # Place text just above the bar, right-aligned with bar end
            try:
                bbox = draw.textbbox((0, 0), txt, font=font, stroke_width=text_outline_px)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
            except (AttributeError, TypeError):
                try:
                    tw, th = draw.textsize(txt, font=font)
                except AttributeError:
                    tw, th = (len(txt) * font_px // 2, font_px)
            tx = x1 - tw
            ty = y0 - th - 6
            ty = max(text_outline_px, ty)
            # PIL >= 8.0 supports stroke_width / stroke_fill — gives us a
            # crisp black halo around bold white text in one draw call.
            try:
                draw.text(
                    (tx, ty), txt, fill=color, font=font,
                    stroke_width=int(text_outline_px),
                    stroke_fill=text_outline_color,
                )
            except TypeError:
                # Old PIL: emulate stroke by drawing 8 offset black copies
                # then the white text on top.
                ox = int(text_outline_px)
                for dx in range(-ox, ox + 1):
                    for dy in range(-ox, ox + 1):
                        if dx == 0 and dy == 0:
                            continue
                        draw.text((tx + dx, ty + dy), txt,
                                  fill=text_outline_color, font=font)
                draw.text((tx, ty), txt, fill=color, font=font)
            out = np.asarray(img)
        except Exception:
            # PIL/fonts unavailable — bar still present, just no label.
            pass
    return out


# ---------------------------------------------------------------------------
# Channel-label legend (top-left of QC overlays)
# ---------------------------------------------------------------------------

def burn_channel_legend(
    rgb_u8: np.ndarray,
    entries: Sequence[Tuple[str, Tuple[int, int, int]]],
    *,
    margin_px: int = 20,
    font_px: int = 18,
    swatch_radius_px: int = 7,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    text_outline_px: int = 2,
    text_outline_color: Tuple[int, int, int] = (0, 0, 0),
    bg_pad_px: int = 6,
) -> np.ndarray:
    """Burn a small "<color swatch> <label>" legend into the top-left corner.

    Each entry is ``(label, (r, g, b))``. Used by QC overlays so a viewer
    can map the yellow / cyan / magenta markers back to their user-typed
    channel names (e.g. "yellow = MIAT-Cy5"). Returns a new image (does
    not mutate input). If PIL is unavailable, the legend is silently
    skipped — the QC overlay still renders.
    """
    if rgb_u8.ndim != 3 or rgb_u8.shape[2] != 3:
        return rgb_u8.copy()
    if not entries:
        return rgb_u8.copy()
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return rgb_u8.copy()
    img = Image.fromarray(rgb_u8)
    draw = ImageDraw.Draw(img)
    font = _load_bold_font(font_px)

    # Measure the widest text line and total height so we can paint a
    # semi-transparent dark backdrop behind the entries (keeps the labels
    # readable over bright signal regions).
    line_h = font_px + 4
    swatch_box = 2 * swatch_radius_px
    text_x_offset = swatch_box + 8
    line_widths: List[int] = []
    for label, _ in entries:
        try:
            bbox = draw.textbbox((0, 0), label, font=font,
                                  stroke_width=text_outline_px)
            tw = bbox[2] - bbox[0]
        except (AttributeError, TypeError):
            try:
                tw, _ = draw.textsize(label, font=font)
            except AttributeError:
                tw = len(label) * font_px // 2
        line_widths.append(int(tw))
    block_w = max(line_widths) + text_x_offset + 2 * bg_pad_px
    block_h = line_h * len(entries) + 2 * bg_pad_px

    # Backdrop: dim the underlying region to ~40% so the legend reads on
    # any background. Use uint8 multiplication rather than alpha compositing
    # so we don't drag in a fourth channel.
    x0 = margin_px
    y0 = margin_px
    x1 = min(img.width, x0 + block_w)
    y1 = min(img.height, y0 + block_h)
    if x1 > x0 and y1 > y0:
        crop = np.asarray(img.crop((x0, y0, x1, y1)), dtype=np.float32)
        crop = crop * 0.35  # dim
        crop = np.clip(crop, 0, 255).astype(np.uint8)
        img.paste(Image.fromarray(crop), (x0, y0))
        # Re-bind draw to the updated image
        draw = ImageDraw.Draw(img)

    cur_y = y0 + bg_pad_px
    for (label, color) in entries:
        sx = x0 + bg_pad_px
        sy = cur_y + (line_h - swatch_box) // 2
        # Filled color swatch with white outline
        draw.ellipse(
            [sx, sy, sx + swatch_box, sy + swatch_box],
            fill=tuple(int(c) for c in color),
            outline=(255, 255, 255),
            width=1,
        )
        tx = sx + text_x_offset
        ty = cur_y
        try:
            draw.text(
                (tx, ty), label, fill=text_color, font=font,
                stroke_width=int(text_outline_px),
                stroke_fill=text_outline_color,
            )
        except TypeError:
            ox = int(text_outline_px)
            for dx in range(-ox, ox + 1):
                for dy in range(-ox, ox + 1):
                    if dx == 0 and dy == 0:
                        continue
                    draw.text((tx + dx, ty + dy), label,
                              fill=text_outline_color, font=font)
            draw.text((tx, ty), label, fill=text_color, font=font)
        cur_y += line_h
    return np.asarray(img)


# ---------------------------------------------------------------------------
# Overlay rendering: nuclei outlines, spots
# ---------------------------------------------------------------------------

def _find_label_boundaries(labels: np.ndarray, width_px: int = 2) -> np.ndarray:
    """Return a HxW boolean mask of label-region boundaries, dilated to width_px."""
    from skimage.segmentation import find_boundaries
    b = find_boundaries(labels, mode="outer")
    if width_px > 1:
        from scipy.ndimage import binary_dilation
        b = binary_dilation(b, iterations=max(1, width_px - 1))
    return b


def draw_nuclei_outlines(
    rgb_u8: np.ndarray,
    labels: np.ndarray,
    color: Tuple[int, int, int] = (255, 255, 255),
    width_px: int = NUC_OUTLINE_WIDTH_PX,
) -> np.ndarray:
    if int(labels.max()) == 0:
        return rgb_u8.copy()
    out = rgb_u8.copy()
    b = _find_label_boundaries(labels.astype(np.int32), width_px=width_px)
    out[b] = np.array(color, dtype=np.uint8)
    return out


def draw_spot_markers(
    rgb_u8: np.ndarray,
    spots: pd.DataFrame,
    *,
    color: Tuple[int, int, int] = (255, 255, 0),  # yellow
    radius: int = 4,
    thickness: int = 2,
) -> np.ndarray:
    """Draw open circles around each (x_px, y_px) spot. Uses PIL if available,
    otherwise stamps a square cross."""
    if spots is None or len(spots) == 0:
        return rgb_u8.copy()
    try:
        from PIL import Image, ImageDraw
        img = Image.fromarray(rgb_u8)
        draw = ImageDraw.Draw(img)
        for _, row in spots.iterrows():
            try:
                x = int(row["x_px"])
                y = int(row["y_px"])
            except Exception:
                continue
            r = int(radius)
            draw.ellipse([x - r, y - r, x + r, y + r], outline=color, width=int(thickness))
        return np.asarray(img)
    except Exception:
        out = rgb_u8.copy()
        h, w = out.shape[:2]
        for _, row in spots.iterrows():
            try:
                x = int(row["x_px"])
                y = int(row["y_px"])
            except Exception:
                continue
            for r in range(-radius, radius + 1):
                if 0 <= x + r < w and 0 <= y < h:
                    out[y, x + r] = color
                if 0 <= y + r < h and 0 <= x < w:
                    out[y + r, x] = color
        return out


# ---------------------------------------------------------------------------
# PNG / TIF writers
# ---------------------------------------------------------------------------

def save_png(rgb_u8: np.ndarray, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
        Image.fromarray(rgb_u8).save(str(path), format="PNG")
    except Exception:
        # Fallback to imageio if PIL is unavailable
        import imageio.v3 as iio
        iio.imwrite(str(path), rgb_u8)


def save_label_tiff(labels: np.ndarray, path: Path) -> None:
    """Save a 16-bit label TIFF (Fiji-compatible)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = labels.astype(np.uint16)
    try:
        import tifffile
        tifffile.imwrite(str(path), arr, compression="zlib")
    except Exception:
        from PIL import Image
        Image.fromarray(arr).save(str(path), format="TIFF")


def save_mask_tiff(mask: np.ndarray, path: Path) -> None:
    """Save a binary mask as 8-bit TIFF (0/255)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (np.asarray(mask) > 0).astype(np.uint8) * 255
    try:
        import tifffile
        tifffile.imwrite(str(path), arr, compression="zlib")
    except Exception:
        from PIL import Image
        Image.fromarray(arr).save(str(path), format="TIFF")


def save_rgb_tiff(rgb_u8: np.ndarray, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import tifffile
        tifffile.imwrite(str(path), rgb_u8, photometric="rgb", compression="zlib")
    except Exception:
        from PIL import Image
        Image.fromarray(rgb_u8).save(str(path), format="TIFF")


# ---------------------------------------------------------------------------
# Per-image render bundles (the high-level API runner.py calls)
# ---------------------------------------------------------------------------

def render_all_in_one_qc(
    dapi: np.ndarray,
    rna: np.ndarray,
    labels: np.ndarray,
    spots: pd.DataFrame,
    voxel_xy_nm: float,
    *,
    dapi_floor_pct: float = DAPI_FLOOR_PCT,
    dapi_ceil_pct: float = DAPI_CEIL_PCT,
    rna_floor_pct: float = RNA_FLOOR_PCT,
    rna_ceil_pct: float = RNA_CEIL_PCT,
    use_batch_contrast: bool = True,
    sec_only: bool = False,
    rna_label: Optional[str] = None,
    dapi_label: Optional[str] = None,
) -> np.ndarray:
    """The canonical Fiji-style QC overlay: DAPI(blue) + RNA(yellow) +
    white nuclei outlines + yellow spot circles + 50 um scale bar.

    RNA contrast is batch-coordinated (Fiji parity:
    apply_lut_to_rgb(rna2d, ..., batch_key="rna") in
    Coloc_Analysis.save_all_in_one_qc_overlay) so QC overlays use the same
    floor/ceil as the publication renders for the same image batch.

    When ``sec_only`` is True the image's percentiles are NOT folded into the
    batch cache (sec-only autofluorescence would inflate the running floor
    or under-feed the running ceiling), but the current batch range IS used
    to render — so a sec-only no-probe control renders at the same contrast
    scale as real-signal images and therefore appears correctly dim.
    """
    dapi_f = _percentile(dapi, dapi_floor_pct)
    dapi_c = _percentile(dapi, dapi_ceil_pct)
    rna_bk = "rna" if use_batch_contrast else None
    rna_f, rna_c = _resolve_lut_range(
        rna, rna_floor_pct, rna_ceil_pct, batch_key=rna_bk, is_sec_only=sec_only,
    )
    dapi_b = apply_lut(dapi, 0.0, 0.3, 1.0, floor=dapi_f, ceil=dapi_c)
    rna_y = apply_lut(rna, 1.0, 1.0, 0.0, floor=rna_f, ceil=rna_c)
    rgb = merge_rgb_additive([dapi_b, rna_y])
    rgb_u8 = _to_uint8(rgb)
    rgb_u8 = draw_nuclei_outlines(rgb_u8, labels, color=(255, 255, 255), width_px=NUC_OUTLINE_WIDTH_PX)
    rgb_u8 = draw_spot_markers(rgb_u8, spots, color=(255, 255, 0), radius=4, thickness=2)
    rgb_u8 = burn_scale_bar(
        rgb_u8, voxel_xy_nm, bar_um=SCALEBAR_UM,
        height_px=SCALEBAR_HEIGHT_PX, font_px=SCALEBAR_FONT_PX,
    )
    # Channel legend (top-left): "<color swatch> <label>" so a viewer can map
    # the yellow spot markers back to the user-typed channel name (e.g.
    # MIAT-Cy5). Falls back to the role names when labels are not supplied.
    legend_entries: List[Tuple[str, Tuple[int, int, int]]] = [
        (str(dapi_label or "DAPI"), (0, 100, 255)),
        (str(rna_label or "RNA"), (255, 255, 0)),
    ]
    rgb_u8 = burn_channel_legend(rgb_u8, legend_entries)
    return rgb_u8


def render_all_in_one_qc_rna_rna(
    dapi: np.ndarray,
    rna1: np.ndarray,
    rna2: np.ndarray,
    labels: np.ndarray,
    spots1: pd.DataFrame,
    spots2: pd.DataFrame,
    voxel_xy_nm: float,
    *,
    dapi_floor_pct: float = DAPI_FLOOR_PCT,
    dapi_ceil_pct: float = DAPI_CEIL_PCT,
    rna_floor_pct: float = RNA_FLOOR_PCT,
    rna_ceil_pct: float = RNA_CEIL_PCT,
    rna2_floor_pct: float = RNA2_FLOOR_PCT,
    rna2_ceil_pct: float = RNA2_CEIL_PCT,
    use_batch_contrast: bool = True,
    sec_only: bool = False,
    rna_label: Optional[str] = None,
    rna2_label: Optional[str] = None,
    dapi_label: Optional[str] = None,
) -> np.ndarray:
    """rna_rna QC overlay: DAPI(blue) + RNA1(yellow) + RNA2(cyan) +
    white nuclei outlines + per-channel spot markers (yellow for RNA1,
    cyan for RNA2) + 50 um scale bar.

    Uses separate batch_key caches ("rna" for RNA1, "rna2" for RNA2) so
    each channel's running-max contrast is tracked independently.
    """
    dapi_f = _percentile(dapi, dapi_floor_pct)
    dapi_c = _percentile(dapi, dapi_ceil_pct)
    rna_bk = "rna" if use_batch_contrast else None
    rna2_bk = "rna2" if use_batch_contrast else None
    rna_f, rna_c = _resolve_lut_range(
        rna1, rna_floor_pct, rna_ceil_pct, batch_key=rna_bk, is_sec_only=sec_only,
    )
    rna2_f, rna2_c = _resolve_lut_range(
        rna2, rna2_floor_pct, rna2_ceil_pct, batch_key=rna2_bk, is_sec_only=sec_only,
    )
    dapi_b = apply_lut(dapi, 0.0, 0.3, 1.0, floor=dapi_f, ceil=dapi_c)
    rna_y = apply_lut(rna1, 1.0, 1.0, 0.0, floor=rna_f, ceil=rna_c)
    rna2_c_layer = apply_lut(rna2, 0.0, 1.0, 1.0, floor=rna2_f, ceil=rna2_c)
    rgb = merge_rgb_additive([dapi_b, rna_y, rna2_c_layer])
    rgb_u8 = _to_uint8(rgb)
    rgb_u8 = draw_nuclei_outlines(rgb_u8, labels, color=(255, 255, 255), width_px=NUC_OUTLINE_WIDTH_PX)
    # RNA1 spots = yellow circles, RNA2 spots = cyan circles. Different
    # marker colors keep the two channels distinguishable when overlaid.
    rgb_u8 = draw_spot_markers(rgb_u8, spots1, color=(255, 255, 0), radius=4, thickness=2)
    rgb_u8 = draw_spot_markers(rgb_u8, spots2, color=(0, 255, 255), radius=4, thickness=2)
    rgb_u8 = burn_scale_bar(
        rgb_u8, voxel_xy_nm, bar_um=SCALEBAR_UM,
        height_px=SCALEBAR_HEIGHT_PX, font_px=SCALEBAR_FONT_PX,
    )
    # Channel legend in the top-left so the reader knows which marker color
    # belongs to which user-typed channel name.
    legend_entries: List[Tuple[str, Tuple[int, int, int]]] = [
        (str(dapi_label or "DAPI"), (0, 100, 255)),
        (str(rna_label or "RNA1"), (255, 255, 0)),
        (str(rna2_label or "RNA2"), (0, 255, 255)),
    ]
    rgb_u8 = burn_channel_legend(rgb_u8, legend_entries)
    return rgb_u8


def save_walkthrough_bundle_rna_rna(
    walk_dir: Path,
    stem: str,
    *,
    dapi: np.ndarray,
    rna1: np.ndarray,
    rna2: np.ndarray,
    dapi_mask: np.ndarray,
    labels: np.ndarray,
    rna1_pos_mask: np.ndarray,
    rna2_pos_mask: np.ndarray,
    voxel_xy_nm: float,
    use_batch_contrast: bool = True,
    sec_only: bool = False,
) -> List[Path]:
    """Walkthrough bundle for rna_rna mode.

    Produces both channels' threshold visualisations:
      Step 01: Raw DAPI grayscale
      Step 02: DAPI binary mask
      Step 03: Nuclei outlines on DAPI
      Step 04a: Raw RNA1 in yellow + scale bar
      Step 04b: Raw RNA2 in cyan + scale bar
      Step 05a: RNA1 threshold mask in yellow on black
      Step 05b: RNA2 threshold mask in cyan on black
      Step 06a: RNA1 threshold overlay on grayscale RNA1
      Step 06b: RNA2 threshold overlay on grayscale RNA2
    """
    walk_dir = Path(walk_dir)
    walk_dir.mkdir(parents=True, exist_ok=True)
    out: List[Path] = []

    # Step 01: Raw DAPI grayscale
    df = _percentile(dapi, DAPI_FLOOR_PCT)
    dc = _percentile(dapi, DAPI_CEIL_PCT)
    s01 = _to_uint8(apply_lut(dapi, 1.0, 1.0, 1.0, floor=df, ceil=dc))
    s01 = burn_scale_bar(s01, voxel_xy_nm, bar_um=SCALEBAR_UM)
    p = walk_dir / f"{stem}__step01_DAPI_raw.png"
    save_png(s01, p); out.append(p)

    # Step 02: DAPI binary mask
    s02 = (np.asarray(dapi_mask) > 0).astype(np.uint8) * 255
    s02 = np.stack([s02, s02, s02], axis=-1)
    p = walk_dir / f"{stem}__step02_DAPI_mask.png"
    save_png(s02, p); out.append(p)

    # Step 03: Nuclei outlines on DAPI
    s03 = _to_uint8(apply_lut(dapi, 1.0, 1.0, 1.0, floor=df, ceil=dc))
    s03 = draw_nuclei_outlines(s03, labels, color=(255, 255, 255), width_px=NUC_OUTLINE_WIDTH_PX)
    s03 = burn_scale_bar(s03, voxel_xy_nm, bar_um=SCALEBAR_UM)
    p = walk_dir / f"{stem}__step03_nuclei_outlines_on_DAPI.png"
    save_png(s03, p); out.append(p)

    # Step 04a/b + step 05a/b + step 06a/b per channel
    def _channel_steps(rna, mask, suffix, r_w, g_w, b_w, color_tuple):
        bk = ("rna" if suffix == "RNA1" else "rna2") if use_batch_contrast else None
        floor_pct = RNA_FLOOR_PCT if suffix == "RNA1" else RNA2_FLOOR_PCT
        ceil_pct = RNA_CEIL_PCT if suffix == "RNA1" else RNA2_CEIL_PCT
        rf, rc = _resolve_lut_range(
            rna, floor_pct, ceil_pct, batch_key=bk, is_sec_only=sec_only,
        )
        s04 = _to_uint8(apply_lut(rna, r_w, g_w, b_w, floor=rf, ceil=rc))
        s04 = burn_scale_bar(
            s04, voxel_xy_nm, bar_um=SCALEBAR_UM,
            height_px=SCALEBAR_HEIGHT_PX, font_px=SCALEBAR_FONT_PX,
        )
        p4 = walk_dir / f"{stem}__step04_{suffix}_raw.png"
        save_png(s04, p4); out.append(p4)

        pos = np.asarray(mask) > 0
        s05 = np.zeros((rna.shape[0], rna.shape[1], 3), dtype=np.uint8)
        s05[pos] = color_tuple
        p5 = walk_dir / f"{stem}__step05_{suffix}_threshold.png"
        save_png(s05, p5); out.append(p5)

        base = _to_uint8(apply_lut(rna, 1.0, 1.0, 1.0, floor=rf * 0.75, ceil=rc))
        s06 = base.copy()
        s06[pos] = (
            0.5 * np.array(color_tuple, dtype=np.float32) + 0.5 * base[pos].astype(np.float32)
        ).astype(np.uint8)
        s06 = burn_scale_bar(
            s06, voxel_xy_nm, bar_um=SCALEBAR_UM,
            height_px=SCALEBAR_HEIGHT_PX, font_px=SCALEBAR_FONT_PX,
        )
        p6 = walk_dir / f"{stem}__step06_{suffix}_threshold_on_signal.png"
        save_png(s06, p6); out.append(p6)

    _channel_steps(rna1, rna1_pos_mask, "RNA1", 1.0, 1.0, 0.0, (255, 255, 0))
    _channel_steps(rna2, rna2_pos_mask, "RNA2", 0.0, 1.0, 1.0, (0, 255, 255))
    return out


def render_segmentation_qc(
    dapi: np.ndarray,
    labels: np.ndarray,
    voxel_xy_nm: float,
) -> np.ndarray:
    """Grayscale-ish DAPI with white nuclei outlines + 50 um scale bar."""
    dapi_f = _percentile(dapi, DAPI_FLOOR_PCT)
    dapi_c = _percentile(dapi, DAPI_CEIL_PCT)
    g = apply_lut(dapi, 1.0, 1.0, 1.0, floor=dapi_f, ceil=dapi_c)
    rgb_u8 = _to_uint8(g)
    rgb_u8 = draw_nuclei_outlines(rgb_u8, labels, color=(255, 255, 255), width_px=NUC_OUTLINE_WIDTH_PX)
    return burn_scale_bar(
        rgb_u8, voxel_xy_nm, bar_um=SCALEBAR_UM,
        height_px=SCALEBAR_HEIGHT_PX, font_px=SCALEBAR_FONT_PX,
    )


def save_publication_images_bundle(
    pub_dir: Path,
    stem: str,
    dapi: np.ndarray,
    rna: np.ndarray,
    voxel_xy_nm: float,
    *,
    protein: Optional[np.ndarray] = None,
    rna2: Optional[np.ndarray] = None,
    use_batch_contrast: bool = True,
    sec_only: bool = False,
    dapi_floor_pct: float = DAPI_FLOOR_PCT,
    dapi_ceil_pct: float = DAPI_CEIL_PCT,
    rna_floor_pct: float = RNA_FLOOR_PCT,
    rna_ceil_pct: float = RNA_CEIL_PCT,
    ab_floor_pct: float = AB_FLOOR_PCT,
    ab_ceil_pct: float = AB_CEIL_PCT,
    rna2_floor_pct: float = RNA2_FLOOR_PCT,
    rna2_ceil_pct: float = RNA2_CEIL_PCT,
    dapi_label: Optional[str] = None,
    rna_label: Optional[str] = None,
    rna2_label: Optional[str] = None,
    antibody_label: Optional[str] = None,
    dapi_lut: Optional[str] = None,
    rna_lut: Optional[str] = None,
    rna2_lut: Optional[str] = None,
    antibody_lut: Optional[str] = None,
) -> List[Path]:
    """Save per-channel publication PNGs + TIFs + pairwise/triple merges, with
    50 um scale bars burned in.

    Mirrors Fiji's ``save_publication_images()`` in Coloc_Analysis.py:
      - DAPI : blue   weights (0.0, 0.3, 1.0), per-image p10..p99.9 stretch
               (Fiji does NOT batch-coordinate DAPI; uniformly bright).
      - RNA  : yellow weights (1.0, 1.0, 0.0), p95..p99.95 stretch (H9 preset);
               batch-coordinated (running-max floor + ceil across all images
               in the run) when ``use_batch_contrast`` is True. This matches
               Fiji's ``apply_lut_to_rgb(rna2d, ..., batch_key="rna")`` call.
      - Prot : magenta weights (1.0, 0.0, 1.0), p80..p99.5 stretch (Fiji
               default DISP_FLOOR/CEIL_PERCENTILE for AB channel), also
               batch-coordinated under ``batch_key="ab"``.

    Outputs (PNG + TIFF for each):
      <stem>__DAPI_blue
      <stem>__RNA_yellow            (aliased to RNA1 when rna2 is supplied)
      <stem>__RNA2_cyan             (only if rna2 is not None — rna_rna mode)
      <stem>__Protein_magenta       (only if protein is not None)
      <stem>__merge_DAPI_RNA        (rna_only / rna_protein convention)
      <stem>__merge_DAPI_RNA1       (rna_rna convention; same image as DAPI_RNA)
      <stem>__merge_DAPI_RNA2       (only if rna2 is not None)
      <stem>__merge_DAPI_RNA1_RNA2  (only if rna2 is not None)
      <stem>__merge_DAPI_Protein    (only if protein is not None)
      <stem>__merge_RNA_Protein     (only if protein is not None)
      <stem>__merge_all             (only if protein is not None)

    Scale bar: 50 um, height=12 px, font=28, white, lower right (matches
    Fiji's ``add_scale_bar_50um``).
    """
    pub_dir = Path(pub_dir)
    pub_dir.mkdir(parents=True, exist_ok=True)

    # DAPI is per-image (Fiji uses explicit p10/p99.9, no batch_key).
    dapi_f = _percentile(dapi, dapi_floor_pct)
    dapi_c = _percentile(dapi, dapi_ceil_pct)
    # RNA is batch-coordinated to keep contrast consistent across the run
    # (Fiji parity: apply_lut_to_rgb(..., batch_key="rna")). Sec-only
    # images do NOT update the batch cache (their dim background biases the
    # running max — Fiji's compute_pub_images_batch_contrast skips sec-only
    # for the same reason at lines 3990-3992 of Coloc_Analysis.py), but they
    # DO render with the current batch range, so a real no-probe control
    # appears dim at the same contrast scale as the NT/KD images.
    rna_bk = "rna" if use_batch_contrast else None
    rna_f, rna_c = _resolve_lut_range(
        rna, rna_floor_pct, rna_ceil_pct, batch_key=rna_bk, is_sec_only=sec_only,
    )

    # Resolve per-role LUTs (defaults match historical Blue / Yellow / Cyan /
    # Magenta so legacy callers that don't pass *_lut produce identical
    # output filenames + colors).
    _dapi_lut_name = (dapi_lut or "blue").lower()
    _rna_lut_name = (rna_lut or "yellow").lower()
    _rna2_lut_name = (rna2_lut or "cyan").lower()
    _ab_lut_name = (antibody_lut or "magenta").lower()
    _dapi_w = lut_name_to_weights(_dapi_lut_name, (0.0, 0.3, 1.0))
    _rna_w = lut_name_to_weights(_rna_lut_name, (1.0, 1.0, 0.0))
    _rna2_w = lut_name_to_weights(_rna2_lut_name, (0.0, 1.0, 1.0))
    _ab_w = lut_name_to_weights(_ab_lut_name, (1.0, 0.0, 1.0))

    dapi_b = apply_lut(dapi, _dapi_w[0], _dapi_w[1], _dapi_w[2], floor=dapi_f, ceil=dapi_c)
    rna_y = apply_lut(rna, _rna_w[0], _rna_w[1], _rna_w[2], floor=rna_f, ceil=rna_c)

    if protein is not None:
        ab_bk = "ab" if use_batch_contrast else None
        ab_f, ab_c = _resolve_lut_range(
            protein, ab_floor_pct, ab_ceil_pct, batch_key=ab_bk, is_sec_only=sec_only,
        )
        prot_m = apply_lut(protein, _ab_w[0], _ab_w[1], _ab_w[2], floor=ab_f, ceil=ab_c)
    else:
        prot_m = None

    # Second RNA channel (rna_rna mode) — LUT configurable (default cyan).
    # Batch-coordinated under a SEPARATE key ("rna2") so its running-max
    # contrast cache does not interfere with the primary RNA channel ("rna").
    # Sec-only behavior matches the primary RNA path: consult-but-don't-update.
    if rna2 is not None:
        rna2_bk = "rna2" if use_batch_contrast else None
        rna2_f, rna2_c = _resolve_lut_range(
            rna2, rna2_floor_pct, rna2_ceil_pct, batch_key=rna2_bk, is_sec_only=sec_only,
        )
        rna2_c_layer = apply_lut(rna2, _rna2_w[0], _rna2_w[1], _rna2_w[2], floor=rna2_f, ceil=rna2_c)
    else:
        rna2_c_layer = None

    saved: List[Path] = []

    def _save_dual(layer: np.ndarray, suffix: str) -> None:
        u = _to_uint8(layer)
        u = burn_scale_bar(
            u, voxel_xy_nm,
            bar_um=SCALEBAR_UM,
            height_px=SCALEBAR_HEIGHT_PX,
            font_px=SCALEBAR_FONT_PX,
        )
        p_png = pub_dir / f"{stem}__{suffix}.png"
        p_tif = pub_dir / f"{stem}__{suffix}.tif"
        save_png(u, p_png)
        save_rgb_tiff(u, p_tif)
        saved.extend([p_png, p_tif])

    # Resolve per-channel filename-safe labels. Defaults match the legacy
    # generic names ("DAPI", "RNA", "RNA2", "Protein") so a config without
    # any *_label fields produces byte-identical output filenames as before
    # the labels feature shipped. Custom labels ("MIAT-Cy5", "QKI-Ab") get
    # sanitized ("MIAT_Cy5", "QKI_Ab") for safe filesystem use.
    dapi_lbl = sanitize_label_for_filename(dapi_label, default="DAPI")
    rna_lbl = sanitize_label_for_filename(rna_label, default="RNA")
    rna2_lbl = sanitize_label_for_filename(rna2_label, default="RNA2")
    ab_lbl = sanitize_label_for_filename(antibody_label, default="Protein")

    # Per-channel renders — suffix = "<label>_<color>" where <color> reflects
    # the user-chosen LUT (defaults to blue/yellow/cyan/magenta for backward
    # compatibility).
    _save_dual(dapi_b, f"{dapi_lbl}_{_dapi_lut_name}")
    _save_dual(rna_y, f"{rna_lbl}_{_rna_lut_name}")
    if rna2_c_layer is not None:
        _save_dual(rna2_c_layer, f"{rna2_lbl}_{_rna2_lut_name}")
    if prot_m is not None:
        _save_dual(prot_m, f"{ab_lbl}_{_ab_lut_name}")

    # Pairwise merge: DAPI + RNA (always produced)
    _save_dual(merge_rgb_additive([dapi_b, rna_y]), f"merge_{dapi_lbl}_{rna_lbl}")

    # rna_rna multi-channel merges
    if rna2_c_layer is not None:
        _save_dual(merge_rgb_additive([dapi_b, rna2_c_layer]),
                   f"merge_{dapi_lbl}_{rna2_lbl}")
        _save_dual(merge_rgb_additive([dapi_b, rna_y, rna2_c_layer]),
                   f"merge_{dapi_lbl}_{rna_lbl}_{rna2_lbl}")

    # Multi-channel merges (RNA + protein)
    if prot_m is not None:
        _save_dual(merge_rgb_additive([dapi_b, prot_m]),
                   f"merge_{dapi_lbl}_{ab_lbl}")
        _save_dual(merge_rgb_additive([rna_y, prot_m]),
                   f"merge_{rna_lbl}_{ab_lbl}")
        _save_dual(merge_rgb_additive([dapi_b, rna_y, prot_m]), "merge_all")

    return saved


def save_walkthrough_bundle(
    walk_dir: Path,
    stem: str,
    *,
    dapi: np.ndarray,
    rna: np.ndarray,
    dapi_mask: np.ndarray,
    labels: np.ndarray,
    rna_pos_mask: np.ndarray,
    voxel_xy_nm: float,
    use_batch_contrast: bool = True,
    sec_only: bool = False,
) -> List[Path]:
    """Save the 6 pipeline-walkthrough PNGs (matches Fiji step01-step06).

    Step 01: Raw DAPI grayscale + scale bar
    Step 02: DAPI binary mask
    Step 03: Nuclei outlines on DAPI
    Step 04: Raw RNA in yellow + scale bar
    Step 05: RNA threshold mask in yellow on black
    Step 06: RNA threshold overlay on grayscale RNA
    """
    walk_dir = Path(walk_dir)
    walk_dir.mkdir(parents=True, exist_ok=True)
    out: List[Path] = []

    # Step 01: Raw DAPI grayscale
    df = _percentile(dapi, DAPI_FLOOR_PCT)
    dc = _percentile(dapi, DAPI_CEIL_PCT)
    s01 = _to_uint8(apply_lut(dapi, 1.0, 1.0, 1.0, floor=df, ceil=dc))
    s01 = burn_scale_bar(s01, voxel_xy_nm, bar_um=SCALEBAR_UM)
    p = walk_dir / f"{stem}__step01_DAPI_raw.png"
    save_png(s01, p); out.append(p)

    # Step 02: DAPI binary mask (white on black)
    s02 = (np.asarray(dapi_mask) > 0).astype(np.uint8) * 255
    s02 = np.stack([s02, s02, s02], axis=-1)
    p = walk_dir / f"{stem}__step02_DAPI_mask.png"
    save_png(s02, p); out.append(p)

    # Step 03: Nuclei outlines on DAPI
    s03 = _to_uint8(apply_lut(dapi, 1.0, 1.0, 1.0, floor=df, ceil=dc))
    s03 = draw_nuclei_outlines(s03, labels, color=(255, 255, 255), width_px=NUC_OUTLINE_WIDTH_PX)
    s03 = burn_scale_bar(s03, voxel_xy_nm, bar_um=SCALEBAR_UM)
    p = walk_dir / f"{stem}__step03_nuclei_outlines_on_DAPI.png"
    save_png(s03, p); out.append(p)

    # Step 04: Raw RNA in yellow LUT — batch-coordinated contrast matches
    # the publication / QC overlay renderers (Fiji parity). Sec-only images
    # consult-but-do-not-update the cache (see save_publication_images_bundle).
    rna_bk = "rna" if use_batch_contrast else None
    rf, rc = _resolve_lut_range(
        rna, RNA_FLOOR_PCT, RNA_CEIL_PCT, batch_key=rna_bk, is_sec_only=sec_only,
    )
    s04 = _to_uint8(apply_lut(rna, 1.0, 1.0, 0.0, floor=rf, ceil=rc))
    s04 = burn_scale_bar(
        s04, voxel_xy_nm, bar_um=SCALEBAR_UM,
        height_px=SCALEBAR_HEIGHT_PX, font_px=SCALEBAR_FONT_PX,
    )
    p = walk_dir / f"{stem}__step04_RNA_raw_yellow.png"
    save_png(s04, p); out.append(p)

    # Step 05: RNA threshold mask in yellow on black
    rna_pos = np.asarray(rna_pos_mask) > 0
    s05 = np.zeros((rna.shape[0], rna.shape[1], 3), dtype=np.uint8)
    s05[rna_pos] = (255, 255, 0)
    p = walk_dir / f"{stem}__step05_RNA_threshold_yellow.png"
    save_png(s05, p); out.append(p)

    # Step 06: RNA threshold overlay on grayscale RNA (50% alpha-ish blend).
    # Fiji uses overlay_mask_on_gray(rna2d, rna_pos, 1.0, 1.0, 0.0, 0.5,
    # disp_floor=r_thr_img * 0.75) — i.e. the GRAY base contrast floor is
    # tied to the RNA threshold itself (0.75x), not the publication
    # percentile floor. Mirrors Fiji exactly.
    base = _to_uint8(apply_lut(rna, 1.0, 1.0, 1.0, floor=rf * 0.75, ceil=rc))
    s06 = base.copy()
    s06[rna_pos] = (
        0.5 * np.array([255, 255, 0], dtype=np.float32) + 0.5 * base[rna_pos].astype(np.float32)
    ).astype(np.uint8)
    s06 = burn_scale_bar(
        s06, voxel_xy_nm, bar_um=SCALEBAR_UM,
        height_px=SCALEBAR_HEIGHT_PX, font_px=SCALEBAR_FONT_PX,
    )
    p = walk_dir / f"{stem}__step06_RNA_threshold_on_signal.png"
    save_png(s06, p); out.append(p)
    return out


def save_nuclei_popouts(
    popout_dir: Path,
    stem: str,
    *,
    dapi: np.ndarray,
    rna: np.ndarray,
    labels: np.ndarray,
    spots_df: pd.DataFrame,
    per_nuc_rows: List[Dict[str, Any]],
    voxel_xy_nm: float,
    n_per_image: int = 1,
    padding_px: int = 30,
    scalebar_um: float = POPOUT_SCALEBAR_UM,
) -> List[Path]:
    """Pick representative nuclei (closest to median rna_mean_in_nucleus)
    and render close-up DAPI(blue)+RNA(yellow) crops with spots + scale bar."""
    popout_dir = Path(popout_dir)
    popout_dir.mkdir(parents=True, exist_ok=True)
    if not per_nuc_rows:
        return []
    means = []
    for r in per_nuc_rows:
        v = r.get("rna_mean_in_nucleus")
        try:
            vf = float(v)
        except (TypeError, ValueError):
            continue
        if vf == vf:
            means.append(vf)
    if not means:
        return []
    median = sorted(means)[len(means) // 2]
    candidates: List[Tuple[int, float, int]] = []
    for r in per_nuc_rows:
        try:
            nid = int(r.get("nucleus_id", 0))
        except (TypeError, ValueError):
            continue
        try:
            mean_v = float(r.get("rna_mean_in_nucleus"))
        except (TypeError, ValueError):
            continue
        spot_count = int(r.get("rna_spot_count", 0) or 0)
        candidates.append((nid, abs(mean_v - median), spot_count))
    candidates.sort(key=lambda t: t[1])
    candidates = candidates[: max(1, n_per_image)]
    saved: List[Path] = []
    h, w = labels.shape
    # Per-image contrast (computed on parent so contrast is consistent with QC)
    df = _percentile(dapi, DAPI_FLOOR_PCT)
    dc = _percentile(dapi, DAPI_CEIL_PCT)
    rf = _percentile(rna, RNA_FLOOR_PCT)
    rc = _percentile(rna, RNA_CEIL_PCT)
    for nid, _dist, n_spots in candidates:
        mask = labels == nid
        if not mask.any():
            continue
        ys, xs = np.where(mask)
        y0 = max(0, int(ys.min()) - padding_px)
        y1 = min(h, int(ys.max()) + 1 + padding_px)
        x0 = max(0, int(xs.min()) - padding_px)
        x1 = min(w, int(xs.max()) + 1 + padding_px)
        if (y1 - y0) < 8 or (x1 - x0) < 8:
            continue
        d_crop = dapi[y0:y1, x0:x1]
        r_crop = rna[y0:y1, x0:x1]
        lbl_crop = labels[y0:y1, x0:x1]
        dapi_b = apply_lut(d_crop, 0.0, 0.3, 1.0, floor=df, ceil=dc)
        rna_y = apply_lut(r_crop, 1.0, 1.0, 0.0, floor=rf, ceil=rc)
        rgb_u8 = _to_uint8(merge_rgb_additive([dapi_b, rna_y]))
        # Outline only this nucleus
        only_this = (lbl_crop == nid).astype(np.int32)
        rgb_u8 = draw_nuclei_outlines(rgb_u8, only_this, color=(255, 255, 255), width_px=2)
        # Spot markers inside the crop
        if spots_df is not None and len(spots_df) > 0:
            mask_in = (spots_df.get("nucleus_id", pd.Series(dtype=int)) == nid)
            sub = spots_df.loc[mask_in].copy()
            if len(sub) > 0:
                sub["x_px"] = sub["x_px"].astype(int) - x0
                sub["y_px"] = sub["y_px"].astype(int) - y0
                rgb_u8 = draw_spot_markers(rgb_u8, sub, color=(255, 255, 255),
                                           radius=4, thickness=1)
        rgb_u8 = burn_scale_bar(
            rgb_u8, voxel_xy_nm, bar_um=scalebar_um,
            height_px=4, margin_px=10, font_px=14,
        )
        p = popout_dir / f"{stem}__representative_nuc_{nid:03d}_spots{n_spots}.png"
        save_png(rgb_u8, p)
        saved.append(p)
    return saved
