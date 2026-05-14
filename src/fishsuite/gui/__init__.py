"""fishsuite.gui — PySide6 desktop launcher.

Public entry points:

* :func:`main` -- the no-arg entry used by ``fishsuite gui`` (CLI) and
  ``python -m fishsuite.gui`` (module-as-script). Returns an int exit code.

The actual Qt window lives in :mod:`fishsuite.gui.main`. We re-export
``main`` here so:

    from fishsuite import gui
    rc = gui.main()

still works exactly like the old single-file ``gui.py`` module (which is
preserved as a shim that forwards here).
"""
from __future__ import annotations

from .main import main

__all__ = ["main"]
