"""Condition-group report layer for fishsuite runs.

``fishsuite report --run <dir> --groups NAME=well,well ...`` builds the
condition-versus-condition workbook, plain-language readout and figures for a
finished run. The run directory is read, never modified.
"""
from __future__ import annotations

from .build import build_report, library_versions
from .aggregate import ReportInputError, parse_groups

__all__ = ["build_report", "library_versions", "ReportInputError", "parse_groups"]
