"""Config + settings persistence for the fishsuite GUI.

Two layers of state:

* **Run config** — a plain nested dict that mirrors the FishsuiteConfig
  schema. The GUI builds this from user input and writes it as YAML.
  Loaded from preset files; merged with widget values at run time.

* **GUI settings** — last-used preset name, last input/output paths,
  run tag, recent run history. Persisted to ``~/.fishsuite_gui.json``
  so the launcher restores its previous state on next open.

This module is intentionally Qt-free — pure data + I/O.
"""
from __future__ import annotations

import copy
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


SETTINGS_PATH = Path.home() / ".fishsuite_gui.json"

DEFAULT_OUTPUT_BASE = Path(r"F:\Image Analysis Work\fishsuite-runs")
DEFAULT_PYTHON_EXE = Path(r"C:\Users\ambur\miniconda3\envs\fishproc\python.exe")
DEFAULT_DOWNSTREAM_CWD = Path(r"F:\Image Analysis Work\image-analysis-pipeline\python")
DEFAULT_DOWNSTREAM_MODULE = "analysis.single_condition_plots"
DEFAULT_PRESET_STEM = "h9_hesc_100x"


# ---------------------------------------------------------------------------
# Preset helpers
# ---------------------------------------------------------------------------

def presets_dir() -> Path:
    """Return the directory containing built-in preset YAMLs."""
    return Path(__file__).resolve().parent.parent / "config" / "presets"


def list_presets() -> List[Path]:
    pd = presets_dir()
    if not pd.is_dir():
        return []
    return sorted(pd.glob("*.yaml"))


def load_preset(stem: str) -> Dict[str, Any]:
    """Load a preset by stem name; returns {} on failure."""
    p = presets_dir() / f"{stem}.yaml"
    try:
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def save_preset_as(stem: str, cfg: Dict[str, Any]) -> Path:
    """Persist a config dict to ``<presets>/<stem>.yaml`` (atomically).

    Raises on filesystem error.
    """
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem.strip()) or "preset"
    out = presets_dir() / f"{safe_stem}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    # Use a tempfile next to the destination for atomic replace.
    tmp = out.with_suffix(out.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    shutil.move(str(tmp), str(out))
    return out


# ---------------------------------------------------------------------------
# Schema defaults — built by introspecting the pydantic model so we never
# drift from the source of truth (config/schema.py).
# ---------------------------------------------------------------------------

def schema_defaults() -> Dict[str, Any]:
    """Return a fresh, fully-defaulted FishsuiteConfig as a plain dict."""
    from ..config.schema import FishsuiteConfig
    return FishsuiteConfig().model_dump(mode="json")


def merge_config(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``over`` into a deep copy of ``base``."""
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_config(out[k], v)
        else:
            out[k] = v
    return out


def cfg_to_yaml_str(cfg: Dict[str, Any]) -> str:
    """Render a config dict to a stable YAML string for previewing."""
    return yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False)


# ---------------------------------------------------------------------------
# Settings (~/.fishsuite_gui.json)
# ---------------------------------------------------------------------------

DEFAULT_SETTINGS: Dict[str, Any] = {
    "last_preset": DEFAULT_PRESET_STEM,
    "last_input_dir": "",
    "last_output_base": str(DEFAULT_OUTPUT_BASE),
    "last_tag": "run",
    "last_overrides": {},  # nested-dict overrides on top of preset
    "skip_downstream": False,
    # ``theme`` is the new tri-state setting: "system" | "light" | "dark".
    # ``dark_mode`` is kept for backwards compatibility with existing
    # ~/.fishsuite_gui.json files — when ``theme`` is absent we fall back to it.
    "theme": "system",
    "dark_mode": False,
    "run_history": [],  # list of {output_dir, ts, success}
    "window": {"w": 1280, "h": 860, "x": None, "y": None},
    # Last-detected channel layout from the "Detect channels" button on the
    # Channels & Mode tab. Persists across sessions so the user still sees
    # "Ch 0 = 640 CSU / Cy5" etc. when reopening the GUI before re-running
    # detect. Keys: ``names`` (list[str]), ``source_file`` (str, optional),
    # ``voxel_xy_nm`` (float | None), ``voxel_z_nm`` (float | None).
    "last_detected_channels": {},
    # Per-input-dir file-selection state (Improvement 2: per-file checkable
    # tree). Mapped {input_dir_absolute: [relative_file, ...]} so different
    # input directories keep independent selections. Empty list (or missing
    # key) = include all discovered files in that dir.
    "file_subset_by_input": {},
}


VALID_THEMES = ("system", "light", "dark")


def resolve_theme(settings: Dict[str, Any]) -> str:
    """Return one of 'system' | 'light' | 'dark' from the settings dict.

    Falls back to the legacy ``dark_mode`` bool if no explicit ``theme`` key
    is present (so existing user setting files keep working unchanged).
    """
    t = settings.get("theme")
    if t in VALID_THEMES:
        return t
    return "dark" if settings.get("dark_mode") else "system"


def load_settings() -> Dict[str, Any]:
    base = copy.deepcopy(DEFAULT_SETTINGS)
    try:
        if SETTINGS_PATH.is_file():
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
            # Shallow-merge so newly-added keys carry sane defaults.
            for k, v in data.items():
                base[k] = v
    except Exception:
        pass
    return base


def save_settings(d: Dict[str, Any]) -> None:
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2, default=str)
    except Exception:
        pass  # best-effort


def append_run_history(settings: Dict[str, Any], entry: Dict[str, Any], cap: int = 5) -> None:
    """Mutate ``settings`` in-place: prepend ``entry`` to run_history, cap at N."""
    hist = list(settings.get("run_history") or [])
    hist.insert(0, entry)
    settings["run_history"] = hist[:cap]


# ---------------------------------------------------------------------------
# Output dir resolution
# ---------------------------------------------------------------------------

_TAG_BAD = re.compile(r"[^A-Za-z0-9_.\-]+")


def sanitize_tag(tag: str) -> str:
    tag = (tag or "").strip()
    if not tag:
        return "run"
    return _TAG_BAD.sub("_", tag)[:64]


def compute_output_dir(output_base: str, run_tag: str, *, ts: Optional[str] = None) -> Path:
    """Build a unique output dir name from base + tag + timestamp."""
    base = Path(output_base or str(DEFAULT_OUTPUT_BASE))
    if ts is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = sanitize_tag(run_tag)
    return base / f"fishsuite-{ts}__{tag}"


# ---------------------------------------------------------------------------
# Channel-name → fluorophore heuristic
#
# Microscope channel names are usually a laser / filter label like "640 CSU"
# or "DAPI 405". We translate those to the user-meaningful fluorophore +
# nominal emission peak (in nm) so the per-role row in the Channels & Mode
# tab can show "→ 640 CSU / Cy5 (em ~668 nm)" rather than just "640 CSU".
# Lookup is keyed by laser wavelength (the leading number) so it works for
# both "640 CSU" and "Cy5 640" forms.
# ---------------------------------------------------------------------------

# (laser_nm_substring, fluorophore, nominal_emission_nm)
_FLUOROPHORE_HEURISTICS = [
    ("405", "DAPI", 442),
    ("488", "GFP/AF488", 525),
    ("561", "Cy3/AF555", 603),
    ("640", "Cy5/AF647", 668),
    ("647", "Cy5/AF647", 668),
]


def fluorophore_for_channel_name(name: str) -> Dict[str, Any]:
    """Map a microscope channel name to a fluorophore + emission peak.

    Returns ``{'fluor': str|None, 'em_nm': int|None, 'laser_nm': int|None}``.
    Best-effort: any field can be None when no match is found. The matcher
    looks for a 3-digit substring in [400, 700] anywhere in ``name`` (so
    "640 CSU", "Cy5 640", "ex640/em668" all work).
    """
    s = str(name or "")
    out: Dict[str, Any] = {"fluor": None, "em_nm": None, "laser_nm": None}
    # Look for any 3-digit token; pick the first one that lives in the
    # plausible visible-spectrum range.
    import re as _re
    for tok in _re.findall(r"\d{3}", s):
        try:
            n = int(tok)
        except ValueError:
            continue
        if 380 <= n <= 750:
            out["laser_nm"] = n
            for lsr, fl, em in _FLUOROPHORE_HEURISTICS:
                if lsr in tok:
                    out["fluor"] = fl
                    out["em_nm"] = em
                    return out
            break
    # No digit match — fall back to fluorophore-name keyword matching.
    sl = s.lower()
    for needle, fluor, em in (
        ("dapi", "DAPI", 442),
        ("hoechst", "Hoechst", 460),
        ("cy5", "Cy5", 668),
        ("cy3", "Cy3", 570),
        ("alexa647", "AF647", 668),
        ("alexa488", "AF488", 525),
        ("alexa555", "AF555", 580),
        ("gfp", "GFP", 510),
        ("rfp", "RFP", 610),
        ("mcherry", "mCherry", 610),
    ):
        if needle in sl:
            out["fluor"] = fluor
            out["em_nm"] = em
            return out
    return out


def lut_for_emission_nm(em_nm: int | None) -> str | None:
    """Brian's standard wavelength → LUT mapping for RNA-FISH outputs.

    Convention (set 2026-05-14):
        - DAPI / 405 nm excitation (em ~440-470) → "blue"
        - 488 nm excitation (em ~510-540) → "green"
        - 561 nm excitation (em ~580-610) → "magenta"  (Cy3 / AF555 / AF568)
        - 640 nm excitation (em ~660-680) → "yellow"   (Cy5 / AF647)
        - Anything outside those bands → None (caller uses its own default)

    Used by the "Detect channels" button to auto-suggest LUT colors for the
    five channel-role dropdowns based on each detected channel's emission
    wavelength. The user can override afterward.
    """
    if em_nm is None:
        return None
    n = int(em_nm)
    if 410 <= n <= 480:
        return "blue"
    if 500 <= n <= 555:
        return "green"
    if 570 <= n <= 625:
        return "magenta"
    if 640 <= n <= 700:
        return "yellow"
    return None


def suggested_lut_for_channel_name(name: str) -> str | None:
    """Combine ``fluorophore_for_channel_name`` + ``lut_for_emission_nm``.

    Returns Brian's standard LUT for a microscope channel name string
    (e.g. "640 CSU" → "yellow", "405 CSU" → "blue"), or None if no
    confident emission match.
    """
    info = fluorophore_for_channel_name(name)
    return lut_for_emission_nm(info.get("em_nm"))


def format_channel_metadata_row(name: str) -> str:
    """One-line human-readable description of a metadata channel name.

    Example:
        "640 CSU" -> "640 CSU / Cy5/AF647 (em ~668 nm)"
        "405 CSU" -> "405 CSU / DAPI (em ~442 nm)"
        "unknown" -> "unknown"
    """
    info = fluorophore_for_channel_name(name)
    if info["fluor"] and info["em_nm"]:
        return f"{name} / {info['fluor']} (em ~{info['em_nm']} nm)"
    if info["fluor"]:
        return f"{name} / {info['fluor']}"
    return str(name)


def detect_channels_from_file(file_path: str) -> Dict[str, Any]:
    """Open ``file_path`` with bioio and return a small metadata summary.

    Returns dict with keys:
        ``source_file`` : str (absolute path)
        ``n_channels``  : int
        ``names``       : list[str] — channel names from the file
        ``voxel_xy_nm`` : float | None
        ``voxel_z_nm``  : float | None
        ``error``       : str | None (set when reading failed; other fields
                          will be empty)

    Pure helper (no Qt) so it's testable from the CLI and survives a future
    threaded-detect implementation.
    """
    p = Path(file_path)
    out: Dict[str, Any] = {
        "source_file": str(p),
        "n_channels": 0,
        "names": [],
        "voxel_xy_nm": None,
        "voxel_z_nm": None,
        "error": None,
    }
    try:
        from bioio import BioImage  # type: ignore
    except Exception as e:
        out["error"] = f"bioio import failed: {e}"
        return out
    if not p.is_file():
        out["error"] = f"file not found: {p}"
        return out
    try:
        img = BioImage(p)
        try:
            img.set_scene(0)
        except Exception:
            pass
        names = [str(n) for n in (img.channel_names or [])]
        out["names"] = names
        out["n_channels"] = int(getattr(img.dims, "C", len(names) or 0))
        try:
            psx = img.physical_pixel_sizes
            out["voxel_xy_nm"] = float(psx.X) * 1000.0 if psx.X else None
            out["voxel_z_nm"] = float(psx.Z) * 1000.0 if psx.Z else None
        except Exception:
            pass
    except Exception as e:
        out["error"] = f"failed to read {p.name}: {e}"
    return out


# ---------------------------------------------------------------------------
# Input-file discovery for the per-file selection tree (Improvement 2).
# ---------------------------------------------------------------------------

# File extensions surfaced in the per-file tree. Mirrors the runner's
# discovery rule plus a couple of extras (lif / ome.tif) Brian's lab
# occasionally hands us.
TREE_EXTENSIONS = (".vsi", ".czi", ".tif", ".tiff", ".ome.tif",
                   ".lif", ".nd2", ".oib", ".oif")


def scan_input_dir_tree(input_dir: str, *, max_depth: int = 2) -> Dict[str, Any]:
    """Walk ``input_dir`` and return a nested mapping for the per-file tree.

    Pure helper (no Qt) so the scan logic is unit-testable. Returns:
        ``{'root': str, 'subfolders': [{'name': str, 'rel': str,
                                        'files': [{'name': str, 'rel': str}, ...]}],
           'root_files': [{'name': str, 'rel': str}, ...]}``

    Subfolders are walked one level deep (matches the runner's
    ``discover_inputs`` flat / subfolder behavior); ``max_depth`` is reserved
    for future expansion. ``rel`` is always a forward-slash relative path
    suitable for use as a config-stable identifier.
    """
    root = Path(input_dir)
    out: Dict[str, Any] = {
        "root": str(root),
        "subfolders": [],
        "root_files": [],
    }
    if not root.is_dir():
        return out

    def _walk_dir(d: Path) -> List[Dict[str, str]]:
        files: List[Dict[str, str]] = []
        try:
            entries = sorted(d.iterdir(), key=lambda p: p.name.lower())
        except Exception:
            return files
        for f in entries:
            if not f.is_file():
                continue
            if f.name.startswith("_"):
                continue
            # Compose extension match: .ome.tif must beat .tif on suffix.
            lname = f.name.lower()
            if not any(lname.endswith(ext) for ext in TREE_EXTENSIONS):
                continue
            rel = f.relative_to(root).as_posix()
            files.append({"name": f.name, "rel": rel})
        return files

    # Top-level files (flat layout).
    out["root_files"] = _walk_dir(root)

    # Subfolders (subfolder layout). Skip underscore-prefixed and hidden.
    try:
        subdirs = sorted(
            (p for p in root.iterdir()
             if p.is_dir() and not p.name.startswith("_") and not p.name.startswith(".")),
            key=lambda p: p.name.lower(),
        )
    except Exception:
        subdirs = []

    for sub in subdirs:
        sub_rel = sub.relative_to(root).as_posix()
        files = _walk_dir(sub)
        if not files:
            continue
        out["subfolders"].append({
            "name": sub.name,
            "rel": sub_rel,
            "files": files,
        })

    return out
