# Desktop report workflow

Launch the updated checkout from PowerShell:
```powershell
$env:PYTHONPATH='E:/Claude/fishsuite-wt-gui-report-tab/src'
& 'C:/Users/ambur/miniconda3/envs/fishproc_dml/python.exe' -m fishsuite.cli gui
```
An installed checkout also supports `fishsuite gui` or `python -m fishsuite.gui`.
Missing PySide6 produces a plain install hint; the GUI never installs or downloads tools.

1. Open **Report**, browse to a completed run, or paste its path and press Tab.
   Required files: `per_image_summary.csv`, `nuclei_metrics.csv`, `run_config.json`.
2. Review the editable groups YAML, or browse to a saved groups YAML.
   Recorded groups are reused; otherwise each persisted well/condition name
   becomes its own group. Combine replicate wells explicitly; pooling is not guessed.
   Example: `groups: {WT: [WT_1, WT_2], KO: [KO_1, KO_2]}`.
3. Select `replicate-simple` (default) or `superplot`.
4. For Deck or DAPI localization, select an existing validated deck spec
   referencing its persisted panel and source manifest. The backend checks provenance.
   Persisted-panel reports force `replicate-simple`, regardless of the dropdown.
5. Enable **Deck** for PowerPoint; **Micrograph slides per-well** requires Deck.
   Deck export also requires the optional python-pptx dependency.
6. **Sec-only correction** imports an existing matched corrected CSV; it does not
   calculate subtraction. Select the CSV, or use its path in the spec. DAPI must be on.
   DAPI and correction selections override those keys in a staged copy of the spec.
7. Output defaults to a new timestamped `report_<stamp>` beside the run.
   Choose an unused folder outside the source run. Existing destinations are rejected.
8. Click **Build** and follow the live log. **Stop** terminates the child process.
   **Native figures by condition** uses the same groups/output and grouping only;
   deck, DAPI, correction and report plot settings do not apply to that command.
   Select a fresh output path before another build if the previous one created it.

Inputs are retained in a unique sibling `<output>_inputs_<stamp>` folder.
The selected groups file is loaded into the editor; builds use the editor's current text.
Source runs and original YAML files are not edited by these controls.
In **Run**, `conditions.groups` accepts a YAML mapping without the outer `groups:` key;
it is saved in the analysis YAML and restored when a preset/config is loaded.
