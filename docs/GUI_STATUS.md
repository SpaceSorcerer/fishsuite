# Desktop GUI status — 2026-09-08

The GUI exists in this branch: `src/fishsuite/gui/` contains the PySide6 window,
settings, readiness checks and subprocess runner. This is a source review;
the window was not launched or interactively tested for this documentation task.

Launch from PowerShell using the intended environment and checkout:

```powershell
$env:PYTHONPATH='E:/Claude/fishsuite-imaging-closeout-2026-09-07/src'
& 'C:/Users/ambur/miniconda3/envs/fishproc_dml/python.exe' -m fishsuite.cli gui
```

After installation, `fishsuite gui` is equivalent; `python -m fishsuite.gui`
is another entry point. The optional `gui` extra provides `PySide6>=6.6`;
`deck` provides `python-pptx>=1.0`. Installing only `[deck]` does not install Qt.
Use the QUICKSTART installation command with `[deck,gui]` if Qt is missing.

The existing window prepares analysis YAML, chooses images, checks readiness,
and launches `run` with a live log and optional downstream figures.
It does **not** launch `report`. It exposes none of the new report options:
plot style, groups YAML, deck/specification, per-well micrograph slides,
DAPI report localization, secondary-corrected report input, `--miat-qki`,
or `--qc-cyto-calls`. Analysis channel controls are separate from report options.

A small future addition (estimate, not implemented):

- Add a Report tab with completed-run/output pickers, groups YAML picker,
  plot-style dropdown, deck/spec picker and per-well micrograph checkbox:
  about 120–180 lines in `main.py`.
- Persist those choices and validate paths, frozen outputs, and deck prerequisites:
  about 50–90 lines across `state.py` and `readiness.py`.
- Build the `python -m fishsuite.cli report` argument list and reuse log streaming:
  about 40–70 lines across `main.py` and `runner_proc.py`.
- Add command-construction/validation checks in one new test file:
  about 80–120 lines. Total: roughly 290–460 lines across five files.

The first version should require an existing validated deck spec/panel/manifest;
making new cohort specifications and secondary corrections is additional work.
