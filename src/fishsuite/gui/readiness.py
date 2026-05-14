"""Readiness logic for the fishsuite GUI.

Pure functions: given a config dict + a path string + a tag, decide
whether each tab is in red / yellow / green state. NO Qt / Tk imports
here so the logic is testable headless.

Status codes:
    "red"    - required field missing or invalid; cannot run
    "yellow" - tab is conditionally inactive (not applicable to this mode)
               OR all required fields are valid but optional fields deviate
               from sensible defaults (informational)
    "green"  - all required fields valid AND optional fields at defaults
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple


Status = str  # "red" | "yellow" | "green"


# ---------------------------------------------------------------------------
# Public API: tab -> status
# ---------------------------------------------------------------------------

def evaluate_all(
    cfg: Dict[str, Any],
    *,
    input_dir: str = "",
    output_base: str = "",
    run_tag: str = "",
) -> Dict[str, Status]:
    """Evaluate every tab's readiness.

    Returns a dict keyed by canonical tab name (lower-case underscore form
    matching what main.py uses internally — e.g. "experiment", "conditions",
    "channels", "zstack", "nuclei", "foci", "pixel_coloc", "cytoplasm",
    "output", "run", "yaml").
    """
    out: Dict[str, Status] = {}
    out["experiment"] = experiment_status(cfg, input_dir=input_dir, output_base=output_base, run_tag=run_tag)
    out["conditions"] = conditions_status(cfg, input_dir=input_dir)
    out["channels"] = channels_status(cfg)
    out["zstack"] = zstack_status(cfg)
    out["nuclei"] = nuclei_status(cfg)
    out["foci"] = foci_status(cfg)
    out["pixel_coloc"] = pixel_coloc_status(cfg)
    out["cytoplasm"] = cytoplasm_status(cfg)
    out["output"] = output_status(cfg)
    # Run + YAML inherit from the rest.
    blocking = any(v == "red" for k, v in out.items() if k not in ("run", "yaml"))
    out["run"] = "red" if blocking else "green"
    out["yaml"] = "green"
    return out


def overall_ready(statuses: Dict[str, Status]) -> bool:
    """True iff no tab is red (so the Run button can be enabled)."""
    return all(v != "red" for k, v in statuses.items() if k != "yaml")


# ---------------------------------------------------------------------------
# Per-tab evaluators
# ---------------------------------------------------------------------------

def _g(cfg: Dict[str, Any], *path: str, default: Any = None) -> Any:
    cur: Any = cfg
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def experiment_status(cfg: Dict[str, Any], *, input_dir: str, output_base: str, run_tag: str) -> Status:
    # Required: input_dir exists and is a directory.
    in_dir_ok = bool(input_dir) and Path(input_dir).is_dir()
    if not in_dir_ok:
        return "red"
    # Output base — we always auto-compute, so even if blank we fall back
    # to a default. Yellow if base is custom but parent doesn't exist;
    # green otherwise.
    if output_base:
        p = Path(output_base)
        if not p.parent.exists():
            return "yellow"
    tag = (run_tag or "").strip()
    if not tag:
        return "yellow"  # works, but unnamed run is sub-optimal
    return "green"


def conditions_status(cfg: Dict[str, Any], *, input_dir: str) -> Status:
    mode = _g(cfg, "conditions", "mode", default="subfolders")
    if mode == "subfolders":
        sub = _g(cfg, "conditions", "subfolder_conditions", default={}) or {}
        # Yellow if no mapping at all - pipeline will still run but everything
        # gets a single condition. Most of the time Brian wants explicit labels.
        if not sub:
            return "yellow"
        # Check existence of mapped subfolders (informational only; runner
        # would still proceed). Green if every mapped folder actually exists
        # under input_dir, yellow if some are missing.
        if input_dir and Path(input_dir).is_dir():
            ip = Path(input_dir)
            missing = [k for k in sub.keys() if not (ip / k).is_dir()]
            if missing:
                return "yellow"
        return "green"
    if mode in ("auto", "explicit"):
        return "yellow"  # works but unusual choices
    return "green"


def channels_status(cfg: Dict[str, Any]) -> Status:
    mode = _g(cfg, "channels", "analysis_mode", default="rna_only")
    dapi = _g(cfg, "channels", "dapi", default=-1)
    rna = _g(cfg, "channels", "rna", default=-1)
    rna2 = _g(cfg, "channels", "rna2", default=-1)
    ab = _g(cfg, "channels", "antibody", default=-1)
    ab2 = _g(cfg, "channels", "antibody2", default=-1)

    def _set(v):
        # -1 = auto; we treat auto-detect as YELLOW (works but Brian usually
        # pins explicit indices). Any value >= 0 is "set".
        return isinstance(v, int) and v >= 0

    if mode == "rna_only":
        if not _set(dapi) and not _set(rna):
            return "yellow"  # full auto-detect — works
        if (not _set(dapi)) or (not _set(rna)):
            return "yellow"  # partial auto-detect
        return "green"
    if mode == "rna_protein":
        req = [dapi, rna, ab]
    elif mode == "rna_rna":
        req = [dapi, rna, rna2]
    elif mode == "ab_ab":
        req = [dapi, ab, ab2]
    elif mode == "protein_only":
        req = [dapi, ab]
    elif mode == "pub_images":
        # All channels optional.
        return "green"
    else:
        return "red"
    if all(_set(v) for v in req):
        return "green"
    if any(_set(v) for v in req):
        return "yellow"
    return "yellow"  # full auto-detect


def zstack_status(cfg: Dict[str, Any]) -> Status:
    mode = _g(cfg, "z_stack", "mode", default="autofocus")
    if mode == "autofocus":
        s = _g(cfg, "z_stack", "start_slice")
        e = _g(cfg, "z_stack", "end_slice")
        if s is None or e is None:
            return "yellow"  # works with full-stack autofocus, but Brian
            # gets bitten when the actual in-focus window isn't pinned
        try:
            if int(s) >= int(e):
                return "red"
            if int(s) < 1:
                return "red"
        except (TypeError, ValueError):
            return "red"
        return "green"
    if mode == "range":
        s = _g(cfg, "z_stack", "start_slice")
        e = _g(cfg, "z_stack", "end_slice")
        if s is None or e is None:
            return "red"
        try:
            if int(s) >= int(e) or int(s) < 1:
                return "red"
        except (TypeError, ValueError):
            return "red"
        return "green"
    if mode == "single":
        ss = _g(cfg, "z_stack", "single_slice")
        if ss is None:
            return "red"
        try:
            if int(ss) < 1:
                return "red"
        except (TypeError, ValueError):
            return "red"
        return "green"
    if mode in ("maxproj", "3d"):
        return "green"
    return "yellow"


def nuclei_status(cfg: Dict[str, Any]) -> Status:
    backend = _g(cfg, "nuclei", "backend", default="stardist")
    if backend == "stardist":
        prob = _g(cfg, "nuclei", "prob_threshold", default=0.5)
        try:
            p = float(prob)
            if p <= 0.0 or p >= 1.0:
                return "red"
        except (TypeError, ValueError):
            return "red"
        mins = _g(cfg, "nuclei", "min_area_px", default=10000)
        try:
            if int(mins) <= 0:
                return "red"
        except (TypeError, ValueError):
            return "red"
        return "green"
    if backend == "cellpose":
        return "green"
    if backend == "otsu":
        return "green"
    return "red"


def foci_status(cfg: Dict[str, Any]) -> Status:
    mode = _g(cfg, "channels", "analysis_mode", default="rna_only")
    enabled = _g(cfg, "foci", "enabled", default=True)
    # In modes that have no RNA channel, foci doesn't apply.
    if mode in ("protein_only", "ab_ab", "pub_images"):
        return "yellow"  # not applicable
    if not enabled:
        return "yellow"  # explicitly disabled
    backend = _g(cfg, "foci", "backend", default="bigfish")
    if backend == "bigfish":
        rad = _g(cfg, "foci", "bigfish_spot_radius_nm", default=130.0)
        try:
            if float(rad) <= 0:
                return "red"
        except (TypeError, ValueError):
            return "red"
        tm = _g(cfg, "foci", "threshold_multiplier", default=0.7)
        try:
            if float(tm) <= 0:
                return "red"
        except (TypeError, ValueError):
            return "red"
        return "green"
    if backend == "log":
        return "green"
    return "red"


def pixel_coloc_status(cfg: Dict[str, Any]) -> Status:
    mode = _g(cfg, "channels", "analysis_mode", default="rna_only")
    if mode in ("pub_images", "protein_only"):
        return "yellow"
    tm = _g(cfg, "pixel_coloc", "threshold_mode", default="mad")
    if tm == "mad":
        try:
            k = float(_g(cfg, "pixel_coloc", "k_mad", default=2.0))
            if k <= 0:
                return "red"
        except (TypeError, ValueError):
            return "red"
    elif tm == "percentile":
        try:
            p = float(_g(cfg, "pixel_coloc", "percentile", default=80.0))
            if not (0.0 < p < 100.0):
                return "red"
        except (TypeError, ValueError):
            return "red"
    elif tm == "costes":
        pass
    else:
        return "red"
    return "green"


def cytoplasm_status(cfg: Dict[str, Any]) -> Status:
    enabled = _g(cfg, "cytoplasm", "enabled", default=True)
    if not enabled:
        return "yellow"
    vme = _g(cfg, "cytoplasm", "voronoi_max_expansion_px", default=80)
    try:
        if int(vme) <= 0:
            return "red"
    except (TypeError, ValueError):
        return "red"
    return "green"


def output_status(cfg: Dict[str, Any]) -> Status:
    # All output fields are booleans/strings with safe defaults.
    return "green"


# ---------------------------------------------------------------------------
# Sanity self-checks (run on import in __main__) — also serves as light unit
# coverage for the readiness logic. Not registered in pytest because the GUI
# package is optional-dep.
# ---------------------------------------------------------------------------

def _self_check() -> None:
    # Case 1: empty config + no input dir -> experiment is red.
    s = evaluate_all({}, input_dir="", output_base="", run_tag="")
    assert s["experiment"] == "red", s
    assert s["run"] == "red", s

    # Case 2: minimal sane config with a valid input dir -> several greens.
    sane = {
        "channels": {"analysis_mode": "rna_only", "dapi": 2, "rna": 0},
        "z_stack": {"mode": "autofocus", "start_slice": 5, "end_slice": 15},
        "nuclei": {"backend": "stardist", "prob_threshold": 0.3, "min_area_px": 10000},
        "pixel_coloc": {"threshold_mode": "mad", "k_mad": 2.5},
        "foci": {"enabled": True, "backend": "bigfish",
                 "bigfish_spot_radius_nm": 130.0, "threshold_multiplier": 0.5},
        "cytoplasm": {"enabled": True, "voronoi_max_expansion_px": 80},
        "conditions": {"mode": "subfolders",
                       "subfolder_conditions": {"X": "X"}},
    }
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        s = evaluate_all(sane, input_dir=td, output_base=td, run_tag="t")
        assert s["channels"] == "green", s
        assert s["zstack"] == "green", s
        assert s["nuclei"] == "green", s
        assert s["foci"] == "green", s
        assert s["pixel_coloc"] == "green", s
        # conditions: mapped folder "X" doesn't exist -> yellow
        assert s["conditions"] == "yellow", s
        # experiment is green (has tag + real dir)
        assert s["experiment"] == "green", s

    # Case 3: rna_protein mode missing antibody index -> yellow.
    rp = dict(sane)
    rp["channels"] = {"analysis_mode": "rna_protein", "dapi": 2, "rna": 0, "antibody": -1}
    with tempfile.TemporaryDirectory() as td:
        s = evaluate_all(rp, input_dir=td, output_base=td, run_tag="t")
        assert s["channels"] == "yellow", s

    # Case 4: invalid z_stack range -> red.
    bad = dict(sane)
    bad["z_stack"] = {"mode": "autofocus", "start_slice": 15, "end_slice": 5}
    with tempfile.TemporaryDirectory() as td:
        s = evaluate_all(bad, input_dir=td, output_base=td, run_tag="t")
        assert s["zstack"] == "red", s
        assert s["run"] == "red", s


if __name__ == "__main__":  # pragma: no cover
    _self_check()
    print("readiness self-check OK")
