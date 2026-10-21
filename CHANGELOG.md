# Changelog

All notable changes to fishsuite. A change that moves a computed number is called out explicitly; see [CONTRIBUTING.md](CONTRIBUTING.md).

## Unreleased

### Documentation

- README rewritten for outside users: generic install, organising images, pixel size and channel detection, the `if_intensity` plate map, one outputs table, and every subcommand grouped as core, post-run and dataset-specific.
- New [docs/QUICKSTART.md](docs/QUICKSTART.md). Guides renamed without dates: [docs/REPORT_SUBCOMMAND.md](docs/REPORT_SUBCOMMAND.md), [docs/REPORT_FEATURES.md](docs/REPORT_FEATURES.md), [docs/SPOT_SIZE_FIT.md](docs/SPOT_SIZE_FIT.md), [docs/GUI.md](docs/GUI.md), [docs/CONDITIONS.md](docs/CONDITIONS.md). Footprint association metrics moved to [docs/FOOTPRINT_ASSOCIATION.md](docs/FOOTPRINT_ASSOCIATION.md).
- Internal change notes, audits and environment records moved to `docs/dev/`.

## 1.0.0

### Report

- Replicate-simple plot style: well means as tested points, mean ± SD columns, shared axes, full and focus PNG plus editable SVG.
- Nucleus-level mixed-model headline with a labelled Welch-on-well-means fallback; equal-variance t, Welch on fields and field-outlier sensitivities.
- DAPI-based localization endpoints, unretained-object census and cytoplasmic-call QC.
- Import of a persisted colocalization panel; simple Pearson, ICQ and Costes-converged-only Manders figures with cytofluorograms.
- Preferential-association ratios with covariance CI, exact tests and minimum detectable ratio.
- Workbook-traced deck export with speaker notes, `slide_sources.csv` and per-well micrograph panels.
- `conditions.groups` collapses native run figures by group; `fishsuite native-figures` regenerates them from finished CSVs.
- Output guards for released deliveries; reporter and producing commits recorded separately.
- `fishsuite report` subcommand and condition groups: wells as replicates, Holm within endpoint families, minimum detectable effect beside every p, and source-run linkage.

### Engine

- Per-spot Gaussian size fit (`size_fit_*`, on by default) and `fishsuite sizefit` to add it to finished runs.
- Fixed-N nucleus sampling (`sampling` block, off by default), selecting last and from DAPI and geometry only.
- Per-image rollups of all nine pixel-coloc coefficients.
- Radial profile per nucleus and per image, with a mean and 95% CI figure; refuses to run without a trustworthy pixel size.
- Spot-callability diagnostic: sample against secondary-only spot rate per punctate channel.
- Rotation null, partner-anchored rotation null and translation null (off by default).
- Footprint association metrics (`compute_footprint_enrichment`, off by default).
- Autofocus channel choice (`autofocus_channel`: dapi, rna, auto, joint) and RNA1 over-detection QC flags.
- Thresholded compartment intensity (`rna_thresh_*`, `rna2_thresh_*`).
- Diffuse-antibody handling (`detect_antibody_spots: false`).
- CUDA support for Cellpose alongside DirectML.
- Global seed, `versions.txt` and `command.log` written at run start; advisory `qc_*` columns that never drop an image.

### Packaging

- Segmentation and spot-detection code vendored at `core/_vendor/` and checksum-pinned.
- `bioio-bioformats` moved to the `bioformats` extra so the base install resolves; new `all` and `deck` extras.
- Portable presets, including `generic_rna_rna_100x.yaml` and `generic_rna_protein_100x.yaml`; run-record presets excluded from the wheel.
- Continuous integration on Linux and Windows with a wheel install check and a dependency-resolution gate.

### Post-run utilities

- `backfill`, `walkthrough`, `postrun`, `singlecell`, `pixelpattern` and `if-pub-images`, with plain-language errors.
