# MIAT–QKI Final Publication Outputs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a clean, publication-ready MIAT–QKI statistical-figure set, six exact-single-z representative micrographs, a plain-language microscopy/acquisition Methods file, and a verified delivery package.

**Architecture:** Add focused publication renderers that consume only retained post-run tables and exact selected-plane records, leaving detection and quantitation untouched. Generate every deliverable into a fresh timestamped output directory, retain complete source-data/selection manifests, and stage only PNG/SVG figures plus verified workbooks and documentation in the final delivery package.

**Tech Stack:** Python 3.10, pandas, NumPy, matplotlib, Pillow, existing FishSuite exact-footprint helpers, pytest, Ruff, PowerShell 7.

**Spec:** `docs/superpowers/specs/2026-08-28-miat-qki-publication-final-spec.md`

## Global Constraints

- Work only in `E:\Claude\fishsuite-codex-miat-qki-footprint` and the named MIAT–QKI output/delivery directories under `F:\Image Analysis Work\MIAT_QKI_Coloc_2026_08_25`.
- Raw images and prior output directories are read-only; every production render uses a new timestamped directory.
- Exact one-z, same-z-across-channels colocalization is mandatory; no projection enters colocalization or representative micrographs.
- Do not modify Claude memory/core/configuration or any `file_map.md`.
- New final figures are 600-dpi PNG and editable SVG only; no new PDF output.
- Slide remains in statistical source data and within-slide permutation inference, but is absent from visible figure encoding.
- Use TDD for every production-code behavior change: failing test, observed failure, minimal implementation, passing test.
- Do not commit, merge, push, or stage changes. The worktree diff and generated patch are the proposal handoff.

---

### Task 1: Clean publication statistical renderer

**Files:**
- Create: `src/fishsuite/core/exact_footprint_publication_plots.py`
- Create: `tests/test_exact_footprint_publication_plots.py`
- Reuse: `src/fishsuite/core/exact_footprint_superplots.py`

**Interfaces:**
- Consumes retained nucleus/FOV/set endpoint tables, primary inference, ratio-of-ratios inference, and existing source-data builders.
- Produces `PublicationPlotOutputs` with PNG paths, SVG paths, source tables, statistics tables, and manifest; `render_publication_plot_package(...)` is the public entry point.

- [ ] **Step 1: Write failing tests for output format and visible pooling**

Add tests that call the public renderer on synthetic two-slide/six-set-per-arm tables and assert: output suffixes are exactly `{.png, .svg}`; no PDF exists; the manifest says `slide_visual_encoding=false`; all set means share one marker; the figure-facing labels and SVG text contain neither `Slide 1`, `Slide 2`, `exact-footprint mass`, nor an unexplained q95 label; source/statistics tables retain slide and `exact_within_slide_label_permutation`.

- [ ] **Step 2: Verify the tests fail for the missing module**

Run:
`$env:PYTHONPATH='E:\Claude\fishsuite-codex-miat-qki-footprint\src'; & 'C:\Users\ambur\miniconda3\envs\fishproc_dml\python.exe' -m pytest tests\test_exact_footprint_publication_plots.py -q`

Expected: collection/import failure because `exact_footprint_publication_plots` does not yet exist.

- [ ] **Step 3: Implement the four approved statistical families**

Implement the total MIAT, base q95 association, q95-associated MIAT depletion, and global-versus-associated depletion families. Reuse existing hierarchy/source-data validation; use one common set-mean marker; omit slide from visible encodings; define q95 in adjacent text; use plain `MIAT spot-pixel intensity` wording; save only PNG/SVG; write exact source-data/statistics tables and a provenance manifest.

- [ ] **Step 4: Add known-value tests for depletion and ratio-of-ratios summaries**

Test that KD/NT percent remaining is `100 * KD / NT`, global and q95 values are not swapped, and ratio-of-ratios inputs/intervals/p-values are copied from the retained inference table rather than recomputed from display values.

- [ ] **Step 5: Run focused and regression tests**

Run the new test file and `tests/test_exact_footprint_superplots.py`; require zero failures.

- [ ] **Step 6: Capture the uncommitted diff and test evidence**

Record modified paths and verbatim test output in the task report. Do not stage or commit.

### Task 2: Multi-example exact-single-z micrograph renderer

**Files:**
- Create: `src/fishsuite/core/exact_footprint_publication_micrographs.py`
- Create: `tests/test_exact_footprint_publication_micrographs.py`
- Reuse: `src/fishsuite/core/exact_footprint_figures.py`

**Interfaces:**
- Consumes exact selected-plane records, image/nucleus/spot QC tables, footprint-pixel records, and existing display/crop/overlay helpers.
- Produces `PublicationMicrographOutputs`; public entry point `render_publication_micrograph_package(..., representatives_per_arm=3)`.

- [ ] **Step 1: Write failing selection tests**

Create synthetic candidates spanning slides and biological sets. Assert deterministic selection of three NT and three MIAT-KD examples, distinct biological sets within each arm where available, exclusion of controls and failed parity/QC, and audit retention of hidden slide/set/source identifiers.

- [ ] **Step 2: Verify the tests fail for the missing module**

Run the new test file and confirm the missing-module failure.

- [ ] **Step 3: Implement deterministic condition-level selection**

Rank valid candidates by condition-relative median/MAD centrality using q95 association and spot-count/QKI summary variables already retained. Select distinct sets first, then deterministic fallbacks only if fewer than three sets contain valid candidates. Do not select by extreme effect size.

- [ ] **Step 4: Write failing render-contract tests**

Assert that individual examples and contact sheets contain DAPI, MIAT, QKI, merge, and q95 exact-footprint overlay panels; use only one recorded z for all channels; output only PNG/SVG; row labels contain condition/example number but no slide/set label; the manifest records `quantitation_plane=exact_recorded_single_z` and `projection_used=false`.

- [ ] **Step 5: Implement individual and contact-sheet rendering**

Reuse `load_selected_plane`, display windows, crop helpers, and exact footprint outline generation. Render three NT and three MIAT-KD examples, with exact q95-positive/q95-negative/unusable footprint outlines and a concise on-figure q95 definition. Save display-only source/audit tables and hashes.

- [ ] **Step 6: Run focused and existing figure regression tests**

Run the new test file plus `tests/test_exact_footprint_figures.py`; require zero failures.

- [ ] **Step 7: Capture the uncommitted diff and test evidence**

Record modified paths and verbatim test output in the task report. Do not stage or commit.

### Task 3: Acquisition methods and plain-language terminology

**Files:**
- Create: `docs/miat_qki/MICROSCOPY_ACQUISITION_METHODS.md`
- Create: `docs/miat_qki/FIGURE_TERMS_PLAIN_LANGUAGE.md`
- Create: `tests/test_miat_qki_publication_docs.py`

**Interfaces:**
- Consumes the approved spec and independently audited VSI/OEX provenance.
- Produces manuscript-ready Markdown copied verbatim into the final report/delivery directories.

- [ ] **Step 1: Write failing document-contract tests**

Assert required acquisition values, mixed z spacing, exact-single-z language, q95 definition, correct dual-omission wording, and missing-detail disclosures. Assert forbidden claims/phrases are absent: universal 210-nm spacing, metadata-verified laser powers, known microscope model/software, `both primary antibodies`, `exact-footprint mass`, and projection-based colocalization.

- [ ] **Step 2: Verify the tests fail because the documents do not exist**

Run `tests/test_miat_qki_publication_docs.py` and observe the expected missing-file failure.

- [ ] **Step 3: Write the methods and terminology documents**

Put a short manuscript-ready paragraph first, then detailed provenance, exact z-selection wording, the separate narrow-projection sensitivity statement, metadata/experimenter/missing evidence table, and missing-information checklist. Keep the language concise and never invent absent acquisition or staining details.

- [ ] **Step 4: Run the document tests**

Require zero failures and record output. Do not stage or commit.

### Task 4: Production generation and visual/numerical QA

**Files:**
- Create: `tools/miat_qki_publication/generate_final_publication.py`
- Create: `tests/test_generate_final_publication.py`
- Create at runtime: fresh timestamped directory below `F:\Image Analysis Work\MIAT_QKI_Coloc_2026_08_25\06_ANALYSIS\EXACT_FOOTPRINT_POSTRUN_20260828-155508`

**Interfaces:**
- Consumes Task 1 and Task 2 renderers plus retained canonical post-run paths.
- Produces a new `FINAL_PUBLICATION_<timestamp>` tree with `STATISTICS`, `MICROGRAPHS`, `METHODS`, `SOURCE_DATA`, and `QA` subdirectories.

- [ ] **Step 1: Write a failing orchestration test**

Assert refusal to overwrite a non-empty output path, exact input-file requirements, PNG/SVG-only manifest output, copied methods hashes, and an explicit six-example micrograph inventory.

- [ ] **Step 2: Verify the test fails for the missing orchestrator**

Run the new orchestration test and observe the expected import failure.

- [ ] **Step 3: Implement the orchestrator**

Add explicit arguments for the retained post-run, backfill/run records, catalog input, output directory, and PNG DPI. Never discover unrelated projects. Validate all sources before creating output; create output once; call the two renderers; copy the two approved Markdown documents; write a source-hash manifest and a plain-language README.

- [ ] **Step 4: Run a dry fixture build and tests**

Require focused tests to pass and assert no PDF in the fixture tree.

- [ ] **Step 5: Run the production build once**

Generate one fresh timestamped production directory from the retained MIAT–QKI sources. Do not rerun into it.

- [ ] **Step 6: Perform numerical and visual QA**

Verify source-table row counts, statistics against retained values, six distinct representative examples, exact z parity, PNG dimensions/DPI, editable SVG structure/text, absence of PDFs, and absence of slide labels. Render contact sheets for visual inspection and log every inspected file/result.

### Task 5: Delivery-package finalization and handoff

**Files:**
- Modify: `F:\Image Analysis Work\MIAT_QKI_Coloc_2026_08_25\DELIVERY_MIAT_QKI_EXACT_FOOTPRINT_2026-08-28\README.md`
- Modify: `...\METHODS.md`
- Modify: `...\PROVENANCE.md`
- Modify: `...\docs\CODEX_CLAUDE_HANDOFF.md`
- Modify: retained standalone `REPORTS_FINAL_20260828\CODEX_CLAUDE_HANDOFF.md`
- Regenerate: delivery code patch, source manifest, package inventory, and SHA-256 checksums.

**Interfaces:**
- Consumes the verified production directory, three companion workbooks, retained canonical large-table references, and the complete worktree diff.
- Produces the final self-contained delivery folder and Claude-discoverable handoff.

- [ ] **Step 1: Copy only the final selected outputs**

Copy Task 4 PNG/SVG/statistics/source-data/methods/QA outputs and the three verified workbooks. Remove only previously staged PDF copies from the delivery folder, leaving canonical source PDFs untouched and recoverable; final delivery is PNG/SVG-only. Do not copy newly generated PDFs, raw VSI/ETS/OEX, BIN1 project files, HDF5, or the 707,446-row footprint-pixel table.

- [ ] **Step 2: Update reader-facing documentation**

Point README/Methods/Provenance/handoffs to the final figures and workbooks, define q95 and MIAT spot-pixel intensity, state the scientific conclusion conservatively, and mark earlier V2/V3 directories as supporting/superseded figure sets rather than current publication figures.

- [ ] **Step 3: Regenerate the complete uncommitted patch**

Include tracked and untracked FishSuite source/tests/tools/docs without staging or committing. Exclude caches, test outputs, temporary previews, and raw data.

- [ ] **Step 4: Refresh copy verification, inventory, and checksums**

Require every copied file to match its canonical source by size and SHA-256. Freeze final manifests only after all copies and docs are stable.

- [ ] **Step 5: Run full verification and independent code review**

Run all exact-footprint/publication/workbook tests, Ruff, `git diff --check`, package/content audits, workbook reload checks, figure inventory/hash checks, and an independent whole-worktree code review. Fix Critical/Important findings and re-review once.

- [ ] **Step 6: Hand off without merging or pushing**

Report absolute paths, actual test outputs, remaining missing acquisition details, scientific interpretation, and every ruling. Leave the branch/worktree and proposal patch for Claude/user review.

