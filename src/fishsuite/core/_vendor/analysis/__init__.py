# -*- coding: utf-8 -*-
"""Single-condition exploratory analysis utilities.

The python/postprocessing/ + visualization/ modules are oriented toward
formal condition comparisons (WT vs KO via Mann-Whitney etc.). This
module is the lighter-weight cousin: takes one Fiji output dir, produces
a multi-panel summary PNG showing distributions per image — useful for
QC, parameter tuning, and single-condition / pilot datasets.

Usage:
    python -m analysis.single_condition_plots --output-dir <run-dir>
"""
