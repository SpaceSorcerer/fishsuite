# MIAT×QKI postrun workbook utility

This isolated utility converts completed postrun CSV tables into one reviewable Excel workbook. Workbook authoring uses the loader-provided Node runtime and `@oai/artifact-tool`; it does not use Python spreadsheet libraries.

## Run

```powershell
<bundled-node> build_workbook.mjs `
  --input-dir <postrun-csv-directory> `
  --config schema.default.json `
  --output <output.xlsx> `
  --preview-dir <preview-directory> `
  --verification-json <verification.json>
```

The default schema requires `summary.csv`, `nucleus.csv`, `fov.csv`, `set.csv`, `inference.csv`, `correlation.csv`, `representative.csv`, `control.csv`, and `qc.csv`. Copy and edit the JSON configuration when filenames, tab names, minimum required columns, column definitions, number formats, or superplots change.

Additional CSV columns are preserved automatically and added to the generated Data Dictionary. Missing required tables or columns stop the build with a stable, explicit error code. Optional table files are omitted and recorded in the README sheet.

The workbook contains README, Data Dictionary, Summary, Nucleus, FOV, Set, Inference, Correlation, Representative, Control, and QC sheets. Every data tab is a filterable Excel table with a frozen header, capped readable widths, semantic number formats, and conditional highlights. The Summary sheet can contain formula-linked native scatter superplots driven by the Set sheet.

## Tests

Run `npm test` with the loader-provided Node executable after creating the required `node_modules` junction. Tests build a synthetic workbook, re-import it, inspect its sheets and formulas, verify every preview, and exercise missing-table and missing-column failures.
