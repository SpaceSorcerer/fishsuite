"""Generate a fishsuite.ico from a stylized FISH-themed fish drawing.

Draws a teal fish silhouette with bright punctate "spots" overlaid on its body
(FISH = fluorescent in situ hybridization → fluorescent dots inside a cell-like
shape, the fish becomes a memorable mascot for the suite). Saves a multi-size
.ico that Windows uses for the Desktop shortcut + taskbar icon.

Run once: produces ``E:\\Claude\\fishsuite\\assets\\fishsuite.ico``.
"""
from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw

ICON_SIZES = [16, 32, 48, 64, 128, 256]
OUT_DIR = Path(r"E:\Claude\fishsuite\assets")
OUT_PNG = OUT_DIR / "fishsuite_256.png"
OUT_ICO = OUT_DIR / "fishsuite.ico"

# Palette — Okabe-Ito-adjacent, matches fishsuite GUI accents.
BG          = (0, 0, 0, 0)                # transparent
TEAL_BODY   = (10, 125, 107, 255)         # main fish color (#0a7d6b)
TEAL_DARK   = (7, 90, 78, 255)            # fin shadow
EYE_WHITE   = (245, 247, 246, 255)
EYE_DARK    = (15, 23, 42, 255)
DAPI_BLUE   = (40, 100, 230, 255)         # spot color 1
RNA_YELLOW  = (240, 200, 60, 255)         # spot color 2
RNA2_CYAN   = (90, 200, 220, 255)         # spot color 3
OUTLINE     = (8, 50, 45, 255)


def draw_fish(size: int) -> Image.Image:
    """Render the fish at ``size``x``size`` px on a transparent canvas."""
    img = Image.new("RGBA", (size, size), BG)
    d = ImageDraw.Draw(img)
    s = size  # alias

    # Body — teardrop, pointing right.
    body_left = int(0.08 * s)
    body_right = int(0.76 * s)
    body_top = int(0.27 * s)
    body_bot = int(0.73 * s)
    d.ellipse([body_left, body_top, body_right, body_bot], fill=TEAL_BODY, outline=OUTLINE, width=max(1, s // 64))

    # Tail — triangle pointing left from body's left edge.
    tail_x = body_left + int(0.04 * s)
    tail_top_y = int(0.18 * s)
    tail_bot_y = int(0.82 * s)
    tail_tip_x = int(0.00 * s)
    tail_tip_y_top = int(0.30 * s)
    tail_tip_y_bot = int(0.70 * s)
    d.polygon(
        [
            (tail_x, int(0.5 * s)),
            (tail_tip_x, tail_top_y),
            (int(0.06 * s), tail_tip_y_top),
            (int(0.06 * s), tail_tip_y_bot),
            (tail_tip_x, tail_bot_y),
        ],
        fill=TEAL_DARK,
        outline=OUTLINE,
        width=max(1, s // 80),
    )

    # Top fin — curved triangle on the upper body.
    fin_top_pts = [
        (int(0.30 * s), body_top + 2),
        (int(0.45 * s), int(0.10 * s)),
        (int(0.58 * s), body_top + 2),
    ]
    d.polygon(fin_top_pts, fill=TEAL_DARK, outline=OUTLINE, width=max(1, s // 80))

    # Bottom fin.
    fin_bot_pts = [
        (int(0.32 * s), body_bot - 2),
        (int(0.42 * s), int(0.92 * s)),
        (int(0.50 * s), body_bot - 2),
    ]
    d.polygon(fin_bot_pts, fill=TEAL_DARK, outline=OUTLINE, width=max(1, s // 80))

    # Eye — white circle with dark pupil, near front of body.
    eye_cx = int(0.66 * s)
    eye_cy = int(0.42 * s)
    eye_r = int(0.055 * s)
    pupil_r = int(0.028 * s)
    d.ellipse([eye_cx - eye_r, eye_cy - eye_r, eye_cx + eye_r, eye_cy + eye_r], fill=EYE_WHITE, outline=OUTLINE, width=max(1, s // 100))
    d.ellipse([eye_cx - pupil_r, eye_cy - pupil_r, eye_cx + pupil_r, eye_cy + pupil_r], fill=EYE_DARK)

    # Mouth — small line in front.
    mouth_x = int(0.74 * s)
    mouth_y = int(0.52 * s)
    d.line([(mouth_x, mouth_y), (mouth_x + int(0.04 * s), mouth_y + int(0.02 * s))], fill=OUTLINE, width=max(1, s // 80))

    # FISH spots — bright punctate dots scattered across the body.
    # Position relative to body ellipse to look like "RNA-FISH puncta in a cell".
    spot_r = max(1, s // 36)
    spots = [
        # (relative_x_fraction, relative_y_fraction, color)
        (0.22, 0.40, RNA_YELLOW),
        (0.26, 0.55, RNA_YELLOW),
        (0.32, 0.42, RNA2_CYAN),
        (0.36, 0.60, DAPI_BLUE),
        (0.42, 0.50, RNA_YELLOW),
        (0.46, 0.40, RNA2_CYAN),
        (0.48, 0.62, RNA_YELLOW),
        (0.52, 0.55, RNA2_CYAN),
        (0.56, 0.48, DAPI_BLUE),
        (0.40, 0.65, RNA2_CYAN),
        (0.30, 0.50, DAPI_BLUE),
        (0.50, 0.42, RNA_YELLOW),
    ]
    for fx, fy, color in spots:
        cx = int(fx * s)
        cy = int(fy * s)
        # Skip tiny spots on very small icons to avoid mush.
        if s >= 24:
            d.ellipse([cx - spot_r, cy - spot_r, cx + spot_r, cy + spot_r], fill=color)

    return img


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Master image at 256px — Pillow downsamples for the smaller sizes.
    master = draw_fish(256)
    master.save(OUT_PNG, format="PNG")
    print(f"Wrote preview PNG: {OUT_PNG}")
    # Save multi-resolution ICO from the master. Pillow handles downscaling.
    master.save(OUT_ICO, format="ICO", sizes=[(s, s) for s in ICON_SIZES])
    print(f"Wrote multi-size ICO: {OUT_ICO}  ({OUT_ICO.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
