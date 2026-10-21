# fishsuite: short PowerShell guide
Use one PowerShell window; keep the variables below in that window. Commands are for Brian's workstation.

## Install/update this checkout in fishproc_dml
```powershell
$repo='E:/Claude/fishsuite-imaging-closeout-2026-09-07'
$py='C:/Users/ambur/miniconda3/envs/fishproc_dml/python.exe'
Set-Location -LiteralPath $repo
& $py -m pip install -e '.[deck]'
$env:PYTHONPATH="$repo/src"
```
This selects the conda environment without activating it; editable install uses this checkout's current code (it does not fetch Git updates). `deck` exists; add `gui` as `.[deck,gui]` for the desktop launcher. **Packaging mismatch:** this branch pins Cellpose <4, while the recorded cpsam_v2 workflow needs 4.2.1.1; the install can downgrade it. Resolve that pin/environment mismatch before a production run; installation was not tested here.

## 1. Stage images and prepare the YAML settings
Keep raw acquisitions untouched. Stage images in well subfolders, including secondary-only wells; keep each VSI beside its companion data. Copy/edit a preset in a new working folder: channel identities, pixel size, wells, z settings and image subsets must match your experiment. Templates are in [src/fishsuite/config/presets/](../src/fishsuite/config/presets/); there is no repository-root `presets/`.
This example reuses the recorded BIN1 settings and existing staging (do not edit the record):
```powershell
$lab='F:/Image Analysis Work/RNASEH2B_BIN1introns_2026_08_25'
$base="$lab/13c_FULL_HARMONIZED_T36_MINAREA4000_2026-09-07"
$preset="$base/presets/rnaseh2b_bin1_HARMONIZED_T36_MINAREA4000_2026-09-07.yaml"
$input="$lab/03_FISHSUITE_SETUP_2026-09-01/input_stage"
$record="$base/RUN_T36_minarea4000_2026-09-07_1941"
```
Other run-of-record example: `F:/Image Analysis Work/MIAT_QKI_Coloc_2026_08_25/04_RERUN_CPSAMV2_2026-09-04/control_recordpreset_2026-09-04.yaml`, used by `RUN_control_recordpreset_FIXEDNUCLEAR_2026-09-05_1754` in that same folder. Use its MIAT/QKI channel settings only with its matching images.

## 2. Check the plan, then run
Make a **new timestamped output folder every time**; run **one GPU job at a time**. Check the dry-run's image list before executing the next line.
```powershell
$stamp=Get-Date -Format 'yyyy-MM-dd_HHmmss_fff'
$run="$base/RUN_T36_minarea4000_$stamp"
& $py -m fishsuite.cli run -c "$preset" -i "$input" -o "$run" -p 1 --dry-run
& $py -m fishsuite.cli run -c "$preset" -i "$input" -o "$run" -p 1 -v
```

## 3. Make the report, figures and PowerPoint
**Example: rebuild the completed recorded BIN1 cohort**, not `$run`. The saved v5 deck specification supplies the panel, source hashes, `dapi_localization: true`, and `secondary_corrected_csv` (matched secondary-only subtraction). It references existing worktree evidence; keep those sources available. There are no standalone DAPI/correction CLI switches.
```powershell
$delivery="$lab/DELIVERY_RNASEH2B_BIN1intron_2026-09-07_v5"
$groups="$repo/report_groups_$stamp.yaml"
Set-Content -LiteralPath $groups -Encoding utf8 -Value "groups: {WT: [WT_1, WT_2, WT_3], QKI-KO: [KO_1, KO_2, KO_3]}`ngroup_order: [WT, QKI-KO]`nreference: WT"
$report="$lab/REPORT_BIN1_$stamp"
& $py -m fishsuite.cli report --run "$record" --groups-file "$groups" --plot-style replicate-simple --out "$report" --deck-spec "$delivery/data/deck_spec.yaml" --deck --micrograph-slides per-well
```
For a new run, replace `--run "$record"` with `--run "$run"` and first make its basic report without `--deck-spec`, `--deck` or `--micrograph-slides`, then do step 4; a new cohort needs its own validated panel/manifest/deck specification and matched correction table before deck export. Those inputs are not created by `--deck`. `--qc-cyto-calls` optionally exports persisted cytoplasmic-call QC. `--miat-qki` is a separate fixed-cohort ratio report; it rejects `--deck` and regrouping.

## 4. Add the standard colocalization panel
The September 7 BIN1 arguments below are copied from its `command.log`; only `--out` is redirected to a fresh folder to protect the recorded panel. For a new cohort, use its run/report paths and review the display ranges. This CPU step reads pixels and computes shuffle controls; it can take time. A basic report normally invokes a panel already; this standalone form reproduces the recorded custom settings.
```powershell
& $py "$repo/scripts/coloc_standard_panel.py" --run "$record" --out "$lab/COLOC_BIN1_$stamp" --report-per-well "$delivery/per_well.csv" --arm-map 'WT_*=WT' --arm-map 'KO_*=QKI-KO' --arm-order 'WT,QKI-KO' --anchor both --rna-min 400 --rna-max 2750 --partner-min 1500 --partner-max 14141 --dapi-min 334 --dapi-max 8000 --rna-lut magenta --partner-lut yellow --costes-fit tls --call-threshold auto --fields 1 --seed 0
```

## 5. Open the outputs and prepare a delivery
Example: open `$report/REPORT.xlsx`, then `$report/Sam_RNASEH2B_BIN1.pptx`. Raw run tables, masks and `versions.txt` remain under `$run`; report figures are in `$report/figures/` as PNG + editable SVG. `slide_sources.csv` traces slide values to workbook cells; `READOUT.md` explains results. The panel has its own workbook/CSVs, figures and overlays in its `--out` folder.
A released `DELIVERY_*` folder contains `REPORT.xlsx`, `figures/`, the deck, `slide_sources.csv`, `SOURCE_RUN.md`, a `00_SOURCE_FISHSUITE_RUN` junction and `MANIFEST_SHA256.tsv`. The manifest/sealing is a separate reviewed packaging step, not automatic `report` output. A junction points to the original run; it is not a portable copy. Keep the run accessible or package the source files for transfer.
> **What each number means:** one well = one biological replicate; nuclei and fields are subsamples. Headline p = nucleus-level mixed model with well and nested-field effects (explicit Welch fallback if unavailable). Welch on well means stays in the footer. “Not detected” ≠ “no effect”; read effect size, uncertainty and multiplicity. Secondary-corrected absolute intensity remains descriptive; baseline uncertainty is not propagated.

## Troubleshooting (six checks)

- Cellpose 4.1.1 substitutes `cpsam` for `cpsam_v2`: inspect `versions.txt` and the run log; resolve the install pin mismatch above before running.
- Frozen-delivery guard: choose a new `--out` folder; do not remove manifests or write through source junctions.
- Missing `python-pptx`: install this checkout with `[deck]` in the same environment; a validated `--deck-spec` is also required.
- `--resume` is dead: accepted by the CLI but unused by the runner; always start with a new output folder.
- Recursive greps/searches on `F:/Image Analysis Work` can freeze the machine; inspect only named files and immediate folders.
- Missing/zero images: check channel mapping, preset subsets, staged VSI companions and reader dependencies; `preview` currently processes the image's parent folder, so do not assume single-image isolation.

## Appendix: exact report help (500-column capture; scroll horizontally)
The help text still describes the older Welch headline; the box above reflects current code. Help checks validate option availability, not a complete run or environment compatibility. Capture command (same CLI entry point):
```powershell
& $py -c 'from fishsuite.cli import cli; cli(prog_name="fishsuite", terminal_width=500, max_content_width=500)' report --help
```
```text
Usage: fishsuite report [OPTIONS]

  Build the condition-versus-condition report for a finished run.

  Wells are the biological replicates and the condition GROUP is what gets compared. Every gate is a Welch t on well means with Hedges g, a Holm adjustment within its endpoint family, and the minimum detectable effect at that number of wells. Figures carry the star from the RAW p and print the adjusted p and the minimum detectable effect in the footnote.

  Writes REPORT.xlsx with plain sheet names, READOUT.md, figures/, per_well.csv, contrasts.csv, versions.txt and command.log. The run directory is read only.

Options:
  --miat-qki DIRECTORY            Report fixed-10 MIAT/QKI ratios from a persisted delivery data directory.
  --miat-qki-count-report DIRECTORY
                                  Separate COUNT localization report; never used as a ratio denominator.
  --run DIRECTORY                 Finished fishsuite output directory to report on. It is READ, never modified.
  --groups NAME=well1,well2,...   Condition GROUP definition, repeatable. A condition is one WELL; a group is the condition several wells belong to and is what gets compared. e.g. --groups WT=WT_1,WT_2,WT_3 --groups QKI-KO=KO_1,KO_2,KO_3. The FIRST group given is the reference unless --reference says otherwise. Omit this and the groups recorded in the run's own config are used; failing that, every well is its own group.
  --groups-file FILE              A report_groups.yaml staged beside the run, carrying `groups`, `group_order`, `reference`, `well_from_image`, `exclude_fields` and a free-text `note`. Command-line flags override any key it sets, so one file makes a run's report reproducible.
  --reference NAME                Group every other group is compared against. Default: the first group.
  --group-order A,B,C             Plotting and reporting order of the groups. Groups present in the data but missing here are appended in sorted order.
  --well-from-image REGEX         Recover the WELL from the image name with a regular expression carrying exactly one capture group. Use this when the run recorded the LINE as its condition and the well only in the file name; without it every field of a line collapses into one well and no test is possible. Every biological image must match or the command refuses to run.
  --out DIRECTORY                 Where to write the report. Default: <run>/report_<timestamp>/.
  --exclude-field NAME            Drop one field of view (the image name as it appears in per_image_summary.csv), repeatable. Each --exclude-field must be followed by its own --reason; the pair is recorded in the workbook and in every figure's filter line.
  --reason TEXT                   Why the preceding --exclude-field was dropped. Given in the same order as the --exclude-field flags, one each.
  --all-pairs / --vs-reference    Report EVERY unordered pair of condition groups, or only each group against the reference. Default: all pairs when there are more than two groups, reference-only when there are two, which are the same thing at two groups.
  --primary-endpoint NAME         Endpoint this report answers on. Named on line 2 of READOUT.md and listed first among the headline endpoints, so a reader who stops after two lines still knows which result is the primary one.
  --sec-outlier-k FLOAT           Drop a secondary-only control field whose puncta per nucleus on ANY channel exceed this multiple of the median across the control fields on that same channel. 0 disables the rule. The rule, its per-channel median and cutoff, and which channel triggered it are all recorded in the workbook.  [default: 0.0]
  --nucleus-filter [all|sampled]  Which nuclei enter the report. 'all' uses every segmented nucleus. 'sampled' keeps only those the run flagged sampled_in_analysis, which is the fixed-N balanced set; use it to match an analysis built on that set. The run must carry the column or the command refuses.  [default: all]
  --peak-floor rna=1000,rna2=1200
                                  Apply a peak-intensity floor to the run's spots AFTER detection, then re-derive the per-nucleus counts. fishsuite's own floor gate is post-detection too, so this reproduces a gated run without re-detecting. Boundary is at or above the floor. Columns that cannot be re-derived post hoc are named in the workbook rather than silently carrying a pre-gate value.
  --caveat-file FILE              A text or markdown file whose contents are inserted into READOUT.md immediately after the run path, and recorded in the Read me sheet. Use it to carry a calibration or interpretation caveat from the analysis record into the deliverable.
  --style [brian|plain]           Figure style. 'brian' is the locked lab style: Okabe-Ito colours, 600-dpi PNG plus editable-text SVG, filter line and single croppable footnote on every chart.  [default: brian]
  --plot-style [superplot|replicate-simple]
                                  Point layout; CLI overrides YAML plot_style. Default: superplot.
  --technical-layer [none|fov]    Optional muted FOVs in replicate-simple; CLI overrides YAML technical_layer. Default: none. Superplot retains its established layers.
  --alpha FLOAT                   Significance level for the Welch gate and the Holm family.  [default: 0.05]
  --qc-min-nuclei INTEGER         Fields with fewer nuclei than this are FLAGGED in the Per field sheet. Nothing is dropped by this flag.  [default: 5]
  --sec-min-nuclei INTEGER        Secondary-only control fields with fewer nuclei than this are excluded by rule, with the rule recorded in the workbook.  [default: 10]
  --engine-repo DIRECTORY         fishsuite checkout whose git HEAD is recorded in Run provenance.
  --preset FILE                   Preset YAML the run used; its md5 is recorded in Run provenance.
  --no-figures                    Skip standard report/panel figures. An explicit --deck-spec still prepares its slide assets.
  --no-coloc-panel                Skip the standard colocalization panel on an rna_rna or rna_protein run.
  --existing-coloc FILE           Import a frozen standard-panel workbook; never generate new nulls.
  --baseline-manifest FILE        A0 source hashes and frozen registry for persisted panel validation.
  --deck-spec FILE                Prepare workbook-traced deck assets and resolve the slide specification.
  --deck                          Export the prepared deck using optional python-pptx.
  --micrograph-slides [per-well]  Matched per-well FOV slides with four independent native publication panels per arm.
  --qc-cyto-calls                 Export every persisted cytoplasmic RNA1 call as calibrated QC crops and CSV.
  --stamp TEXT                    Timestamp used in the default output directory name. Defaults to now.
  --help                          Show this message and exit.
```
