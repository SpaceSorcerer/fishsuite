"""Locked style layer for the basal MIAT x QKI figure panels (FIG1_IMAGING_v2, 2026-09-28).

Ported from the run-local ``scripts/v2style.py``. Display ranges, LUTs and palette are the
locked defaults: MIAT yellow, QKI magenta, DAPI blue (never green). rcParams are applied by
``apply_style()`` rather than at import so importing the package does not mutate global state.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RCPARAMS = {'font.family': 'Arial', 'svg.fonttype': 'none', 'font.size': 7,
            'axes.spines.top': False, 'axes.spines.right': False,
            'axes.linewidth': .75, 'pdf.fonttype': 42}

COL = '#0072B2'          # VPR noDox arm colour (delivery ARM_COL)
NULLC = '#DDDDDD'        # null fill
LINE_MIAT, LINE_QKI, LINE_DAPI = '#C9A900', '#CC00CC', '#0072B2'
LV = {"miat": (500.0, 2250.0), "qki": (1050.0, 3746.0), "dapi": (607.0, 9000.0)}
RGB = {"miat": (1, 1, 0), "qki": (1, 0, 1), "dapi": (0.25, 0.45, 1.0)}
AXIAL = 0.878
from ..core.pixel_size import FALLBACK_PIXEL_UM as PX_UM, resolve_pixel_um  # noqa: E402
FILT = "Filter: MIAT min 500 + QKI min 1050 + LoG 174; single analysed plane."

WELL_MEAN_MARKER = dict(s=110, marker='o', edgecolor='black', lw=1.0, zorder=5)

_FORBIDDEN_RGB = {(0, 1, 0), (0.0, 1.0, 0.0)}

DISPLAY_PRESETS = {"fig1_v2": LV}   # per-set display ranges, raw a.u.; LV stays the default


def parse_display_ranges(spec):
    """None/'' -> None (LV); a DISPLAY_PRESETS key; or 'MIAT=lo:hi,QKI=lo:hi,DAPI=lo:hi' naming all three."""
    if not spec:
        return None
    if spec in DISPLAY_PRESETS:
        return dict(DISPLAY_PRESETS[spec])
    out = {}
    for item in spec.split(","):
        ch, _, rng = item.partition("=")
        lo, _, hi = rng.partition(":")
        try:
            out[ch.strip().lower()] = (float(lo), float(hi))
        except ValueError as exc:
            raise ValueError(f"display range {item!r} is not CHANNEL=lo:hi") from exc
    if set(out) != set(LV) or not all(np.isfinite(v).all() and v[0] < v[1] for v in out.values()):
        raise ValueError(f"display ranges must give finite lo < hi for exactly {sorted(LV)}; got {spec!r}")
    return out


def run_lv(run) -> dict:
    return getattr(run, "display_ranges", None) or LV


def run_px(run, key=None, recorded=None) -> float:
    """Pixel size (um) of image ``key``: the run (``--px-um``, then its image manifest), then ``recorded``
    (e.g. the menu selection), then PX_UM with a logged WARNING (core.pixel_size.resolve_pixel_um)."""
    resolve = getattr(run, "pixel_size_um", None)
    px = resolve(key) if callable(resolve) else None
    return resolve_pixel_um(recorded=px if px is not None else recorded, where=f"image {key}")


def display_note(lv) -> str:
    return "Display: " + ", ".join(f"{c.upper()} {lv[c][0]:g}-{lv[c][1]:g}"
                                   for c in ("miat", "qki", "dapi")) + " raw a.u."


def apply_style() -> None:
    plt.rcParams.update(RCPARAMS)


def check_luts(rgb=None) -> None:
    """Raise if a LUT is pure green (lab rule: MIAT yellow, QKI magenta, never green)."""
    for ch, v in (rgb or RGB).items():
        if tuple(float(x) for x in v) in _FORBIDDEN_RGB:
            raise ValueError(f"channel {ch} uses a green LUT {v}; locked LUTs are MIAT yellow, QKI magenta")


def well_mean_style(condition: str, well: str | None = None, color: str = COL) -> dict:
    """Marker rule: every well of one condition gets the SAME marker and colour ('o').

    ``well`` is accepted and deliberately ignored so a caller cannot vary the marker by well.
    """
    return dict(WELL_MEAN_MARKER, facecolor=color)


def colorize(planes: dict, lv=None, rgb=None) -> np.ndarray:
    lv = lv or LV
    rgb = rgb or RGB
    out = sum(np.clip((a.astype(float) - lv[c][0]) / (lv[c][1] - lv[c][0]), 0, 1)[..., None]
              * np.array(rgb[c])[None, None, :] for c, a in planes.items())
    return np.clip(out, 0, 1)


def footer(fig, lines, width=None):
    W = width or int(fig.get_figwidth() * 72 / 2.6)
    txt = "\n".join("\n".join(textwrap.wrap(line, W)) for line in lines)
    fig.text(0.01, 0.012, txt, fontsize=5, va='bottom', ha='left', color='#333333')


def stars(p):
    return '****' if p < 1e-4 else '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < .05 else 'ns'


def fmt_p(p):
    return f"{p:.2g}" if p >= 1e-4 else f"{p:.1e}"


class Saver:
    """Writes <stem>.{png,svg,pdf} at 600 dpi plus a <stem>_native.svg twin for image panels.

    native_only=True writes only the native SVG twin (re-render mode). keep_open=True leaves the
    figure open and records it in ``self.figures`` (used by tests to inspect artists).
    """

    def __init__(self, pan_dir, native_only: bool = False, keep_open: bool = False, write: bool = True):
        self.pan = Path(pan_dir)
        self.native_only = native_only
        self.keep_open = keep_open
        self.write = write
        self.figures: dict = {}
        self.written: list = []

    def _native_svg(self, fig, path):
        old = [(im, im.get_interpolation()) for ax in fig.axes for im in ax.get_images()]
        for im, _ in old:
            im.set_interpolation('none')
        fig.savefig(path, dpi=600, facecolor='white')
        for im, it in old:
            im.set_interpolation(it)

    def save(self, fig, stem, bbox=None):
        if self.write:
            self.pan.mkdir(parents=True, exist_ok=True)
            has_img = any(ax.get_images() for ax in fig.axes)
            if not self.native_only:
                for ext in ("png", "svg", "pdf"):
                    fig.savefig(self.pan / f"{stem}.{ext}", dpi=600, facecolor='white', bbox_inches=bbox)
                    self.written.append(self.pan / f"{stem}.{ext}")
            if has_img:
                self._native_svg(fig, self.pan / f"{stem}_native.svg")
                self.written.append(self.pan / f"{stem}_native.svg")
        if self.keep_open:
            self.figures[stem] = fig
        else:
            plt.close(fig)
