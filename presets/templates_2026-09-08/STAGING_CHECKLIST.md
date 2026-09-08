# Staging checklist — new imaging datasets (2026-09-08)
- These are pilot templates, not production settings. Replace every placeholder; well counts and acquisition dates are missing.
- MIAT source YAML is missing; its template recovers `config_resolved` from the authorized run JSON. Fixed MIAT LoG threshold is missing (source uses auto).
- Use separate staging roots for MIAT replicate 2 and QKI/BIN1. Keep raw acquisitions untouched; preserve VSI companion data beside each VSI.
- Expected layout (one immediate folder per well; FOV files inside; no extra condition-level folder):
```text
MIAT_R2_STAGE/
  R2_NT_WELL_A/       <FOV files + companion data>
  R2_KD_WELL_A/       <FOV files + companion data>
  R2_SecOnly_NT/      <all matched new-slide secondary-only FOVs>
  R2_SecOnly_KD/      <all matched new-slide secondary-only FOVs>
QKI_BIN1_STAGE/
  NEW_WT_WELL_A/      <FOV files + companion data>
  NEW_KO_WELL_A/      <FOV files + companion data>
  Shared_SecOnly_WT/  <all matched shared-design secondary-only FOVs>
  Shared_SecOnly_KO/  <all matched shared-design secondary-only FOVs>
```
- Rename/extend `subfolder_conditions`, `condition_order`, and `groups` together to actual well names; keep control folders only in `sec_only_folders`, outside biological groups.
- MIAT R2 is a NEW slide/acquisition: use its OWN sec-only and run/report; never pool with the 2026-08-28 acquisition without an explicit batch term. Group names do not fit a batch model.
- QKI/BIN1 shares the RNASEH2B sec-only design; record control provenance and verify staining, channel, exposure and acquisition compatibility before reusing any control.
- Verify channel indices from metadata: QKI/BIN1 640=QKI (yellow), 561=BIN1 intron (magenta), 405=DAPI (blue); MIAT indices are inherited, not newly verified.
- Confirm pixel size and stack depths; inspect new autofocus planes. Old filename-specific z overrides/allowlists were removed. Do not transfer old reviewed planes.
- Require fishproc_dml + Cellpose 4.2.1.1 + cpsam_v2 weights; check logs for fallback. QUICKSTART notes a dependency pin mismatch; resolve before production, do not blindly reinstall.
- Keep min_area_px=4000: September 7 census found disproportionate KO loss at 16000; verify segmentation at the new pixel size. MIAT retains fixed-10/image sampling; review pilot eligibility.
- Edit a COPY of the template; set these PowerShell variables to actual absolute paths (tokens below are placeholders, not runnable paths):
```powershell
$repo='E:/Claude/fishsuite-wt-presets-new-datasets'
$py='C:/Users/ambur/miniconda3/envs/fishproc_dml/python.exe'
$env:PYTHONPATH="$repo/src"
$env:PYTHONDONTWRITEBYTECODE='1'
$preset='<ABSOLUTE_PATH_TO_EDITED_DATASET_YAML>'
$input='<ABSOLUTE_PATH_TO_DATASET_STAGING_ROOT>'
$base='<ABSOLUTE_PATH_TO_NEW_DATASET_OUTPUT_ROOT>'
$stamp=Get-Date -Format 'yyyy-MM-dd_HHmmss_fff'
$run="$base/RUN_$stamp"
& $py -m fishsuite.cli run -c "$preset" -i "$input" -o "$run" -p 1 --dry-run
```
- Inspect roster: intended FOVs and well labels, every matched sec-only, no old slide, no duplicate raw/decon acquisitions. Dry-run validates discovery; it does not validate thresholds or segmentation.
- Pilot on 2 FOVs per biological well + ALL matched sec-only (separate pilot staging or an explicit new-image subset); fewer available FOVs is missing input, not permission to invent them.
- Recheck carried LoG thresholds AND raw-intensity floors on the new slide against sec-only (lab 3× rule). The exact 3× statistic is missing in supplied sources: resolve before production; do not multiply a LoG threshold blindly.
- Sweep candidates; choose the LOWEST threshold with signal-to-control >=2 AND sec-only <10% of biological. Record per-channel spots/nucleus rates and matched-control denominators; inspect both arms. Zero-control ratios need explicit handling, not an automatic pass.
- Keep `detect_in_sec_only: true`; retain all controls for pilot QC. The >=2 config diagnostic only warns and does not enforce the <10% criterion. Recheck QKI spot callability; diffuse protein is not a puncta count endpoint.
- Same floors for analysis and micrographs: templates enable both publication-floor gates; use one per-channel floor across biological and sec-only images. Synchronize peak/intensity overrides and antibody/rna2 aliases when tuning. LoG response thresholds are separate units.
- Freeze accepted thresholds/floors across the new slide; record the pilot decision, metadata and control provenance. Use a fresh timestamped production output, remove pilot subsets, dry-run again; run ONE GPU job at a time.
- Production command after those checks: `& $py -m fishsuite.cli run -c "$preset" -i "$input" -o "$run" -p 1 -v`.
- Report/deck commands (ordinary report reads the template's recorded groups; each report output must be fresh):
```powershell
$report="$base/REPORT_$stamp"
& $py -m fishsuite.cli report --run "$run" --plot-style replicate-simple --out "$report"
$deckSpec='<ABSOLUTE_PATH_TO_NEW_COHORT_VALIDATED_DECK_SPEC_YAML>'
& $py -m fishsuite.cli report --run "$run" --plot-style replicate-simple --out "$base/REPORT_DECK_$stamp" --deck-spec "$deckSpec" --deck --micrograph-slides per-well
```
- Before deck export, prepare this cohort's validated panel/manifest/deck specification and matched correction table; `--deck` does not create them. Do not use fixed-cohort `--miat-qki` or old cohort assets for R2.
- Inspect REPORT.xlsx, READOUT.md, per_well.csv, figures, versions and source tracing. Keep wells distinct; nuclei/FOVs are subsamples. Joint slide analysis needs a separately specified batch-aware model.
