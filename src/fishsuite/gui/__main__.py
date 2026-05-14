"""Allow ``python -m fishsuite.gui`` to launch the GUI."""
from __future__ import annotations

import sys

from .main import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
