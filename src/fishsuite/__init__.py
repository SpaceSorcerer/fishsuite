"""fishsuite — standalone Python RNA-FISH / IF colocalization & quantification suite.

See README.md for a quickstart and docs/IMPLEMENTATION_LOG.md for build status.
"""

import os as _os
# Force a non-interactive matplotlib backend at the earliest possible point so
# ANY entry path (CLI, GUI subprocess, Jupyter, ad-hoc script, future test
# harness) gets the same default. Worker threads on Windows otherwise inherit
# TkAgg, which calls into Tcl from the wrong thread once Bio-Formats' JVM is
# alive → Tcl_AsyncDelete crash that brings the JVM down with it. `setdefault`
# is courteous: anything that already set MPLBACKEND (e.g. the Qt-based GUI)
# is honored.
_os.environ.setdefault("MPLBACKEND", "Agg")

__version__ = "0.1.0"

# Apply the bffile numpy-1 compatibility monkeypatch as soon as the package
# is imported. Bioio depends on bffile, which calls
#     np.asarray(data, dtype=dtype, copy=False)
# — a numpy-2 kwarg that errors under numpy 1.x. We pin numpy<2 to keep
# tensorflow / stardist compatible, so we patch bffile to be numpy-1 safe.
def _apply_bffile_compat_patch() -> None:
    try:
        import numpy as _np
        import bffile._biofile as _bf
    except Exception:
        return  # bioio not installed or different layout — nothing to patch
    _ORIG = getattr(_bf, "_reshape_image_buffer", None)
    if _ORIG is None or getattr(_ORIG, "_fishsuite_patched", False):
        return

    def _reshape_image_buffer(data, *, dtype, height, width, rgb, interleaved):
        if hasattr(data, "dtype") and data.dtype == _np.dtype(dtype):
            arr = _np.asarray(data)
        else:
            arr = _np.asarray(data, dtype=dtype)
        if rgb > 1:
            if interleaved:
                return arr.reshape(height, width, rgb)
            return arr.reshape(rgb, height, width).transpose(1, 2, 0)
        return arr.reshape(height, width)

    _reshape_image_buffer._fishsuite_patched = True  # type: ignore[attr-defined]
    _bf._reshape_image_buffer = _reshape_image_buffer  # type: ignore[assignment]


_apply_bffile_compat_patch()
