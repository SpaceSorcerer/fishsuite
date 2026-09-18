# FishSuite single-z review utility

This utility turns an inventory CSV into a manual z-plane review packet. It is
appropriate before a FishSuite run when every field of view needs its own
reviewed optical section.

It does **not** finalize a z plane. The numerical scorer only nominates likely
planes. A reviewer must inspect the contact sheet and fill `selected_z` in
`selected_z_review.csv`. Re-running the utility preserves completed review
fields.

## Locked behavior

- Input rows must contain `source_vsi`; every other inventory column is carried
  into `review_manifest.csv` for traceability.
- Every diagnostic row and every contact-sheet tile is one exact optical plane.
  No MIP or other z projection is computed or rendered.
- Every channel shown inside one z card comes from that same z index.
- Candidate focus scoring uses FishSuite's intensity-weighted variance of the
  Laplacian by default. If nuclear masking is enabled, the mask is derived from
  one provisional DAPI plane, not a projection.
- The output directory is rejected if it is inside the inferred raw-image tree.
- The biological channel map is explicit; the utility never guesses it.

## DAPI-only review before the biological map is confirmed

Run from the isolated FishSuite worktree. Replace the display bounds with the
review window chosen for the new dataset:

```powershell
& "C:\Users\ambur\miniconda3\envs\fishproc_dml\python.exe" -m tools.single_z_review `
  "F:\Image Analysis Work\PROJECT\00_INVENTORY\image_inventory.csv" `
  "F:\Image Analysis Work\PROJECT\01_Z_REVIEW_DAPI" `
  --channel "dapi=2" `
  --label "dapi=DAPI-405" `
  --window "dapi=<LOW>,<HIGH>" `
  --focus-role dapi
```

This reads only channel 2 and is safe to prepare before deciding which
biological target is on channel 0 versus channel 1.

## Three-channel review after the map is confirmed

Roles and labels are free text. This role-neutral example deliberately avoids
asserting the biological identity of channels 0 and 1:

```powershell
& "C:\Users\ambur\miniconda3\envs\fishproc_dml\python.exe" -m tools.single_z_review `
  "F:\Image Analysis Work\PROJECT\00_INVENTORY\image_inventory.csv" `
  "F:\Image Analysis Work\PROJECT\01_Z_REVIEW_3CH" `
  --channel "channel0=0" --label "channel0=640 channel" --window "channel0=<LOW>,<HIGH>" `
  --channel "channel1=1" --label "channel1=561 channel" --window "channel1=<LOW>,<HIGH>" `
  --channel "dapi=2"     --label "dapi=DAPI-405"          --window "dapi=<LOW>,<HIGH>" `
  --focus-role dapi --focus-role channel0 --focus-role channel1
```

The same candidate plane is displayed across all three channels. To display all
three channels while letting only DAPI nominate candidates, keep only
`--focus-role dapi`.

## Outputs

- `review_manifest.csv`: original inventory metadata, processing status,
  provisional candidate z values, and artifact paths.
- `selected_z_review.csv`: editable review ledger. Fill `review_status`,
  `selected_z`, `reviewer`, `reviewed_at`, and `review_notes`.
- `z_diagnostics.csv`: long table with one row per image × z × channel,
  including focus, intensity, percentile, and display-clipping diagnostics.
- `images/<review_id>/all_z_contact_sheet.png`: all planes, grouped by z, with
  configured channels shown at the same plane.
- `images/<review_id>/z_diagnostics.csv`: per-image copy of the numerical table.
- `review_config.json`: complete settings and the explicit
  `manual_review_required` / `no_projection` provenance contract.

The command exits with code 2 if any inventory image fails, while retaining
successful review packets and recording each error in the manifest.
