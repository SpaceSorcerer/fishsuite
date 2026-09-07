"""Source-run linkage for a report folder.

Brian could not tell which fishsuite run produced a delivery, so every report
folder carries the answer in three redundant forms:

1. ``SOURCE_RUN.md`` at the top, naming the run and everything needed to find it
   again.
2. A junction ``00_SOURCE_FISHSUITE_RUN`` pointing at the run, when the report is
   written outside the run directory. A ``.lnk`` plus a plain-text path file is
   the fallback when the target refuses a junction.
3. Copies of the run's own provenance files under ``provenance/source_run/``.

Rule of record: the ``deliverable-package`` skill, section "Source-run linkage".
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

PROVENANCE_FILES = ("run_config.json", "versions.txt", "thresholds.csv", "command.log")
JUNCTION_NAME = "00_SOURCE_FISHSUITE_RUN"

# Subfolders a reader should open, and what each holds.
RUN_SUBFOLDERS = [
    ("figures", "the run's OWN plots, one panel per condition (a condition is a well)"),
    ("figures/by_group", "by-group plots, written only when the run had condition groups"),
    ("qc_overlays", "per-image segmentation and detection overlays"),
    ("publication_images", "merged display images at the run's own display window"),
    ("masks", "nuclei label masks, needed to trace any nucleus back to a pixel"),
    ("per_image_csv", "per-image copies of the master tables"),
]
RUN_FILES = [
    ("per_image_summary.csv", "one row per image"),
    ("nuclei_metrics.csv", "one row per segmented nucleus"),
    ("spot_metrics.csv", "one row per detected punctum"),
    ("thresholds.csv", "the detection threshold actually used for each image"),
    ("run_config.json", "the fully resolved configuration"),
    ("versions.txt", "tool versions and the seed"),
]


FREEZE_MARKERS = ("MANIFEST_SHA256.tsv", "CHECKSUMS.sha256", "CHECKSUMS.txt")


def is_frozen_delivery(directory: Path) -> bool:
    """A DELIVERY_* folder is frozen once it carries a release manifest or a SUPERSEDED marker.

    A freshly created, still-empty DELIVERY_* folder is a legitimate build target.
    """
    if not directory.name.upper().startswith("DELIVERY_"):
        return False
    if any((directory / m).is_file() for m in FREEZE_MARKERS):
        return True
    try:
        return any(p.name.upper().startswith("SUPERSEDED_BY") for p in directory.iterdir())
    except OSError:
        return False


def guard_output(path: Path) -> Path:
    """Never write into a frozen delivery, including through a junction/symlink."""
    from .aggregate import ReportInputError
    path = Path(path).absolute()
    for candidate in (path, path.resolve()):
        for ancestor in (candidate, *candidate.parents):
            if is_frozen_delivery(ancestor):
                raise ReportInputError(f'frozen output forbidden: {path}')
    return path


def producing_commit(run_dir: Path) -> str:
    """Only a run-embedded identity is a producing commit; HEAD is never a fallback."""
    import re
    path = Path(run_dir) / 'versions.txt'
    if path.is_file():
        for line in path.read_text(encoding='utf-8').splitlines():
            key, sep, value = line.partition(':')
            if sep and key.strip().lower() in {'producing engine commit', 'engine_commit', 'git_commit'}:
                if re.fullmatch(r'[0-9a-fA-F]{40}', value.strip()):
                    return value.strip()
    return 'missing from run'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_junction(link: Path, target: Path) -> Dict[str, str]:
    """Directory junction via ``cmd /c mklink /J``.

    Never ``ln -s``: under Git Bash on Windows that deep-copies the target, which
    would duplicate an entire run directory. On failure, fall back to a ``.lnk``
    shortcut plus a plain-text path file so the linkage still survives.
    """
    link, target = Path(link), Path(target)
    if link.exists():
        return {"method": "already present", "path": str(link)}
    try:
        r = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                           capture_output=True, text=True, timeout=60)
        if r.returncode == 0 and link.exists():
            return {"method": "junction", "path": str(link)}
        err = (r.stdout or "") + (r.stderr or "")
    except Exception as exc:                                   # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"

    fallback = link.parent / f"{JUNCTION_NAME}.txt"
    fallback.write_text(
        f"The producing fishsuite run is:\n{target}\n\n"
        f"A directory junction could not be created here, so this file and the "
        f"shortcut beside it carry the link instead.\nReason: {err.strip()}\n",
        encoding="utf-8")
    lnk = link.parent / f"{JUNCTION_NAME}.lnk"
    ps = (f"$s=(New-Object -ComObject WScript.Shell).CreateShortcut('{lnk}');"
          f"$s.TargetPath='{target}';$s.Save()")
    try:
        subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                       capture_output=True, text=True, timeout=60)
    except Exception:                                          # noqa: BLE001
        pass
    return {"method": "path file" + (" plus shortcut" if lnk.exists() else ""),
            "path": str(fallback), "junction_error": err.strip()[:300]}


def copy_provenance(run_dir: Path, out_dir: Path) -> List[str]:
    dest = Path(out_dir) / "provenance" / "source_run"
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in PROVENANCE_FILES:
        src = Path(run_dir) / name
        if src.is_file():
            shutil.copy2(src, dest / name)
            copied.append(name)
    return copied


def _threshold_lines(run_dir: Path) -> List[str]:
    path = Path(run_dir) / "thresholds.csv"
    if not path.is_file():
        return ["- thresholds.csv is absent from this run."]
    thr = pd.read_csv(path)
    out = []
    for col in ("rna_bigfish_log_threshold", "protein_bigfish_log_threshold",
                "rna2_bigfish_log_threshold", "protein_threshold_value"):
        if col not in thr.columns:
            continue
        vals = sorted(pd.to_numeric(thr[col], errors="coerce").dropna().unique().tolist())
        if not vals:
            continue
        if len(vals) == 1:
            out.append(f"- `{col}` = {vals[0]:g}, the same for every image (harmonized).")
        else:
            out.append(f"- `{col}` varies per image, {min(vals):g} to {max(vals):g} "
                       f"across {len(vals)} distinct values.")
    return out or ["- no threshold column was found in thresholds.csv."]


def write_source_run(run_dir: Path, out_dir: Path, *, preset: Optional[Path] = None,
                     groups: Optional[Dict[str, str]] = None,
                     group_order: Sequence[str] = (), reference: str = "",
                     label: str = "") -> Dict[str, object]:
    """Write ``SOURCE_RUN.md``, the junction and the provenance copies."""
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    digests: Dict[str, str] = {}
    for name in PROVENANCE_FILES:
        p = run_dir / name
        digests[name] = sha256(p) if p.is_file() else "file absent from the run"

    cfg = {}
    cfg_path = run_dir / "run_config.json"
    if cfg_path.is_file():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:                                      # noqa: BLE001
            cfg = {}
    versions: Dict[str, str] = {}
    vpath = run_dir / "versions.txt"
    if vpath.is_file():
        for line in vpath.read_text(encoding="utf-8", errors="replace").splitlines():
            if ":" in line and not line.strip().startswith("#"):
                k, v = line.split(":", 1)
                versions[k.strip()] = v.strip()

    inside = out_dir.resolve().is_relative_to(run_dir.resolve())
    link = {"method": "not created; the report is inside the run directory",
            "path": str(run_dir)}
    if not inside:
        link = make_junction(out_dir / JUNCTION_NAME, run_dir)
    copied = copy_provenance(run_dir, out_dir)

    nuc = (cfg.get("config_resolved") or {}).get("nuclei") or {}
    backend = str(cfg.get("SEGMENTATION_BACKEND") or nuc.get("backend") or "not recorded")
    model = str(nuc.get(f"{backend}_model_type") or nuc.get("cellpose_model_type") or "")

    lines: List[str] = []
    a = lines.append
    a(f"# Source fishsuite run{f' - {label}' if label else ''}")
    a("")
    a(f"`{run_dir}`")
    a("")
    a("This report was built from that directory and only that directory. The run "
      "itself was read, never modified.")
    a("")
    a("## Identity")
    a("")
    a("| Item | Value |")
    a("|---|---|")
    a(f"| Run directory | `{run_dir}` |")
    a(f"| Report directory | `{out_dir}` |")
    a(f"| Preset | `{preset}` |" if preset else
      f"| Preset | `{cfg.get('config_path', 'not recorded')}` |")
    a(f"| Analysis mode | {(cfg.get('config_resolved') or {}).get('channels', {}).get('analysis_mode', cfg.get('ANALYSIS_MODE', 'not recorded'))} |")
    a(f"| Images in the run | {cfg.get('n_images', 'not recorded')} |")
    a(f"| Segmentation | {backend}{(' ' + model) if model else ''} |")
    a(f"| cellpose | {versions.get('cellpose', 'not recorded in versions.txt')} |")
    a(f"| fishsuite | {versions.get('fishsuite_version', cfg.get('version', 'not recorded'))} |")
    a(f"| Seed | {versions.get('global_seed', 'not recorded')} |")
    a(f"| Run started | {cfg.get('run_start_utc', 'not recorded')} |")
    a(f"| Run finished | {cfg.get('run_end_utc', 'not recorded')} |")
    a(f"| Report built | {datetime.now(timezone.utc).isoformat(timespec='seconds')} |")
    a("")
    a("## Checksums, SHA-256")
    a("")
    a("| File | SHA-256 |")
    a("|---|---|")
    for name, dg in digests.items():
        a(f"| `{name}` | `{dg}` |")
    a("")
    a("## Detection thresholds")
    a("")
    lines.extend(_threshold_lines(run_dir))
    a("")
    a("## Comparison this report makes")
    a("")
    if groups:
        a("| Condition group | Wells |")
        a("|---|---|")
        for g in (group_order or sorted(set(groups.values()))):
            wells = sorted(w for w, gg in groups.items() if gg == g)
            a(f"| {g}{' (reference)' if g == reference else ''} | "
              f"{', '.join(wells) if wells else 'every well not named in another group'} |")
    else:
        a("Groups were not supplied on the command line; each condition in the run "
          "was treated as its own group.")
    a("")
    a("## What to open in the run")
    a("")
    for name, what in RUN_SUBFOLDERS:
        p = run_dir / name
        mark = "" if p.exists() else "  (not present in this run)"
        a(f"- `{name}\\` - {what}.{mark}")
    for name, what in RUN_FILES:
        p = run_dir / name
        mark = "" if p.exists() else "  (not present in this run)"
        a(f"- `{name}` - {what}.{mark}")
    a("")
    a("## Linkage written beside this file")
    a("")
    a(f"- `{JUNCTION_NAME}` - {link['method']}, pointing at the run directory.")
    a(f"- `provenance\\source_run\\` - copies of "
      f"{', '.join(f'`{c}`' for c in copied) if copied else 'nothing; the run had none of the provenance files'}.")
    a("")

    name = f"SOURCE_RUN_{label}.md" if label else "SOURCE_RUN.md"
    path = out_dir / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"source_run_md": str(path), "link": link, "copied": copied,
            "sha256": digests}
