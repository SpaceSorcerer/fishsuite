"""Analysis modes — one module per mode (rna_only, rna_protein, ...)."""
from typing import Any, Dict

MODE_REGISTRY: Dict[str, Any] = {}


def register_mode(name: str):
    def deco(fn):
        MODE_REGISTRY[name] = fn
        return fn
    return deco


def get_mode(name: str):
    if name not in MODE_REGISTRY:
        raise KeyError(
            f"Unknown analysis mode: {name!r}. Available: {sorted(MODE_REGISTRY)}"
        )
    return MODE_REGISTRY[name]


# Import side effects register modes
from . import rna_only  # noqa: F401, E402
from . import rna_protein  # noqa: F401, E402
from . import rna_rna  # noqa: F401, E402
from . import ab_ab  # noqa: F401, E402
from . import protein_only  # noqa: F401, E402
from . import pub_images  # noqa: F401, E402
from . import if_intensity  # noqa: F401, E402
