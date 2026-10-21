# Desktop GUI

## Launching

```bash
pip install "fishsuite[gui]"      # PySide6
fishsuite gui                     # or: python -m fishsuite.gui
```

If PySide6 is missing, the command prints an install hint. The GUI never installs or downloads tools. PowerPoint export from the Report tab also needs the `deck` extra (`pip install "fishsuite[deck,gui]"`).

## Run tab and settings tabs

The window prepares an analysis YAML, lets you choose images, checks readiness, and runs `fishsuite run` with a live log. Each tab carries a readiness dot: green is ready, amber is usable with defaults or guesses, red blocks the run. The live YAML preview can be saved as a reusable preset.

In the Conditions tab, `conditions.groups` accepts a YAML mapping without the outer `groups:` key. It is saved in the analysis YAML and restored when a preset is loaded. The well and group workflow is described in [CONDITIONS.md](CONDITIONS.md).

## Report tab

1. Open **Report** and browse to a completed run, or paste its path and press Tab. The run must contain `per_image_summary.csv`, `nuclei_metrics.csv` and `run_config.json`.
2. Review the editable groups YAML, or load a saved one. Groups recorded in the run are reused; otherwise each well becomes its own group. Replicate wells are never pooled by guesswork, so combine them explicitly, for example `groups: {WT: [WT_1, WT_2], KO: [KO_1, KO_2]}`.
3. Choose the plot style: `replicate-simple` (default) or `superplot`.
4. For a deck or DAPI localization, select an existing validated deck specification that points to its persisted panel and source manifest. Reports from a persisted panel always use `replicate-simple`.
5. Enable **Deck** for PowerPoint. **Micrograph slides per-well** requires Deck.
6. **Sec-only correction** imports an existing matched corrected CSV. It does not compute the subtraction. DAPI must be on. The DAPI and correction choices override those keys in a staged copy of the spec.
7. Output defaults to a new `report_<stamp>` folder beside the run. Choose an unused folder outside the source run; existing destinations are rejected.
8. Click **Build** and follow the log. **Stop** ends the child process. **Native figures by condition** runs `fishsuite native-figures` with the same groups and output; deck, DAPI, correction and plot settings do not apply to it.

Inputs are copied into a sibling `<output>_inputs_<stamp>` folder. Builds use the groups text in the editor. Source runs and original YAML files are never edited.

Report options are described in [REPORT_SUBCOMMAND.md](REPORT_SUBCOMMAND.md) and [REPORT_FEATURES.md](REPORT_FEATURES.md).
