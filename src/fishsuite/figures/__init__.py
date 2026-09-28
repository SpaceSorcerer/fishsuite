"""Locked basal MIAT x QKI figure-panel module (FIG1_IMAGING_v2, locked 2026-09-28).

Linked ortho sets (B1-B4), well-mean superplots with Holm-adjusted nucleus-unit Wilcoxon (C), correlation
scatters (D), representative crops (R), whole-FOV panels (A) and native TIFF export. Locked defaults live in
``style``: display MIAT 500-2250 / QKI 1050-3746 / DAPI 607-9000 raw a.u.; MIAT yellow, QKI magenta, DAPI blue
(never green); one marker and colour per condition; 600 dpi PNG + SVG (text as text) + PDF + native SVG twin.

Entry points: ``fishsuite fig-panels`` or ``python -m fishsuite.figures``; library: ``gallery.build_gallery``.
"""
from .config import PanelRun
from .stats import holm, wilcoxon_effect, run_tests
from .style import well_mean_style

__all__ = ["PanelRun", "holm", "wilcoxon_effect", "run_tests", "well_mean_style"]
