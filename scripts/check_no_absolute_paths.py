"""Fail if a machine-specific absolute path is an ACTIVE value in ``src/``.

Why this exists: a hard-coded ``F:\\Image Analysis Work\\...`` is not merely
unhelpful on someone else's machine. If a path by that name happens to exist
there, the pipeline reads the wrong data and reports a number instead of an
error — the worst failure mode available. Several such paths shipped in the
config presets before the 2026-08-10 release-prep pass.

What counts as an ACTIVE value, and what does not:

* Python — every string literal EXCEPT docstrings and ``#`` comments. Docstrings
  are exempt on purpose: several modules document, for provenance, which external
  script they were ported from, and deleting those references would destroy the
  audit trail without making anything safer. Comments likewise.
* YAML — every scalar reached by loading the document, which is exactly the set
  of values the config model will see. Comments disappear during the load, so
  they need no special handling.
* ``core/_vendor/`` is skipped. It is copied verbatim and checksum-pinned in
  ``_vendor/PROVENANCE.md``; it cannot be edited to satisfy this check.
* DATED RUN-RECORD PRESETS are skipped, because for them the absolute path IS the
  artifact. ``pyproject.toml`` keeps 42 of the 48 presets in the repository
  precisely as the provenance of published figures, and the wheel excludes every
  one of them (deny-all-then-allow), so the hazard this gate exists to prevent —
  an outside user's machine silently reading a same-named directory — cannot
  reach them. Scrubbing their paths to ``<SET_ME>`` destroyed the record without
  making anything safer, and it was reverted.

  The exempt set is DERIVED from the wheel allowlist rather than listed here: a
  preset under ``config/presets/`` is checked if and only if pyproject
  force-includes it into the wheel. That way adding a new PORTABLE preset puts it
  under the gate automatically, and neither list can drift from the other. If the
  allowlist cannot be read the gate checks everything and says so — it never
  silently exempts.

Run it directly (``python scripts/check_no_absolute_paths.py``) or let CI do it.
Exit status 0 = clean, 1 = findings, 2 = the check itself could not run.

A note for whoever maintains this: the first version of this gate was a
``grep -rnE 'F:\\|...'`` one-liner that matched NOTHING because of how the shell
and ERE each treat a backslash — it passed loudly while checking nothing. If you
change the matching here, add a case to ``_self_test`` and confirm it fails on a
known-bad string before trusting a pass.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

# Case-insensitive substrings that mark a path as machine-specific.
BAD_SUBSTRINGS = (
    "f:\\",
    "f:/",
    "c:\\users",
    "c:/users",
    "e:\\claude",
    "e:/claude",
    "d:\\",
)

SKIP_DIR_PARTS = ("_vendor",)

# Presets live here. Only the ones the wheel ships are gated; see the module
# docstring for why the rest are provenance, not template.
PRESET_DIR_PARTS = ("config", "presets")
_FORCE_INCLUDE_HEADER = "[tool.hatch.build.targets.wheel.force-include]"


def shipped_preset_names(pyproject: Path) -> set[str] | None:
    """Basenames of the presets the wheel force-includes, or None if unreadable.

    Deliberately a line scan of the flat ``force-include`` table rather than a
    TOML parse: ``tomllib`` is 3.11+, this gate must run on the 3.10 CI leg, and
    ``tomli`` is not a declared dependency. Returning None means "could not tell",
    which the caller turns into checking everything.
    """
    try:
        text = pyproject.read_text(encoding="utf-8")
    except Exception:
        return None
    if _FORCE_INCLUDE_HEADER not in text:
        return None
    body = text.split(_FORCE_INCLUDE_HEADER, 1)[1]
    names: set[str] = set()
    for raw in body.splitlines():
        line = raw.strip()
        if line.startswith("["):
            break
        if not line or line.startswith("#") or "=" not in line:
            continue
        lhs = line.split("=", 1)[0].strip().strip('"').strip("'")
        if lhs.endswith((".yaml", ".yml")):
            names.add(Path(lhs).name)
    return names or None


def is_gated_preset(path: Path, shipped: set[str] | None) -> bool:
    """Should this preset file be checked? Non-presets are always checked."""
    parts = path.parts
    is_preset = any(
        parts[i:i + len(PRESET_DIR_PARTS)] == PRESET_DIR_PARTS
        for i in range(len(parts))
    )
    if not is_preset:
        return True
    if shipped is None:
        return True  # fail closed: cannot tell what ships, so check it
    return path.name in shipped


def _offending(text: str) -> str | None:
    low = text.lower()
    for bad in BAD_SUBSTRINGS:
        if bad in low:
            return bad
    return None


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """id() of every Constant node that is a module/class/function docstring."""
    out: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            out.add(id(first.value))
    return out


def check_python(path: Path) -> list[str]:
    src = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return [f"{path}:{exc.lineno}: could not parse ({exc.msg})"]
    exempt = _docstring_nodes(tree)
    findings = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        if id(node) in exempt:
            continue
        bad = _offending(node.value)
        if bad:
            findings.append(
                f"{path}:{node.lineno}: string literal contains {bad!r}: "
                f"{node.value[:110]!r}"
            )
    return findings


def _walk_yaml(obj, path: Path, trail: str = "") -> list[str]:
    findings = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            findings += _walk_yaml(v, path, f"{trail}.{k}" if trail else str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            findings += _walk_yaml(v, path, f"{trail}[{i}]")
    elif isinstance(obj, str):
        bad = _offending(obj)
        if bad:
            findings.append(
                f"{path}: active value at {trail or '<root>'} contains "
                f"{bad!r}: {obj[:110]!r}"
            )
    return findings


def check_yaml(path: Path) -> list[str]:
    import yaml

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:  # noqa: BLE001 — report, do not crash the gate
        return [f"{path}: could not parse as YAML ({type(exc).__name__}: {exc})"]
    return _walk_yaml(data, path)


def _self_test() -> None:
    """Prove the matcher fires. A gate that cannot fail is not a gate."""
    assert _offending(r"F:\Image Analysis Work\x.csv") == "f:\\"
    assert _offending("F:/Image Analysis Work/x.csv") == "f:/"
    assert _offending(r"C:\Users\someone\data") == "c:\\users"
    assert _offending(r"E:\Claude\fishsuite\_staging") == "e:\\claude"
    assert _offending("./results/my_run") is None
    assert _offending("<SET_ME>") is None
    assert _offending("") is None

    # And prove the preset exemption cannot swallow a preset that DOES ship.
    shipped = {"generic_100x_0p065.yaml"}
    presets = Path("src/fishsuite/config/presets")
    assert is_gated_preset(presets / "generic_100x_0p065.yaml", shipped)
    assert not is_gated_preset(presets / "panqki_if_wtko_2026-07-03.yaml", shipped)
    # Unreadable allowlist -> check everything, never silently exempt.
    assert is_gated_preset(presets / "panqki_if_wtko_2026-07-03.yaml", None)
    # A non-preset file is never exempt.
    assert is_gated_preset(Path("src/fishsuite/core/_superplot.py"), shipped)


def main(argv: list[str]) -> int:
    _self_test()
    root = Path(argv[1]) if len(argv) > 1 else Path("src")
    if not root.exists():
        print(f"error: {root} does not exist", file=sys.stderr)
        return 2

    shipped = shipped_preset_names(Path("pyproject.toml"))
    if shipped is None:
        print("WARNING: could not read the wheel preset allowlist from "
              "pyproject.toml — checking EVERY preset, including the dated run "
              "records.", file=sys.stderr)

    findings: list[str] = []
    n_py = n_yaml = n_exempt = 0
    for path in sorted(root.rglob("*")):
        if any(part in SKIP_DIR_PARTS for part in path.parts):
            continue
        if path.suffix == ".py":
            n_py += 1
            findings += check_python(path)
        elif path.suffix in (".yaml", ".yml"):
            if not is_gated_preset(path, shipped):
                n_exempt += 1
                continue
            n_yaml += 1
            findings += check_yaml(path)

    print(f"scanned {n_py} .py and {n_yaml} .yaml files under {root} "
          f"(skipped {'/'.join(SKIP_DIR_PARTS)}; exempted {n_exempt} dated "
          f"run-record preset(s) the wheel does not ship)")
    if findings:
        print(f"\nFAIL: {len(findings)} machine-specific absolute path(s) as "
              f"active values:\n", file=sys.stderr)
        for f in findings:
            print("  " + f, file=sys.stderr)
        print("\nReplace with a relative path or the '<SET_ME>' placeholder.",
              file=sys.stderr)
        return 1
    print("OK: no machine-specific absolute paths as active values.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
