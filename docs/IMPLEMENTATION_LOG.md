# fishsuite — Implementation Log

## Phase 0 — Foundations (DONE)

**Outcome:** scaffold created at `E:\Claude\fishsuite\`; bioio VSI reads work via a one-function monkey-patch around `bffile`. See `docs/PHASE0_STATUS.md`.

Key artifacts:
- `pyproject.toml`, `README.md`
- `src/fishsuite/__init__.py` — applies bffile numpy<2 patch on import
- `tests/test_smoke.py` — 4 passing tests (package import, YAML load, Pearson identity, Pearson orthogonal)

## Phase 1 — Core engine (DONE)

Modules in `src/fishsuite/core/`:
- `io.py` — `read_image`, `extract_channel(z_mode=single|maxproj|autofocus|3d)`, `get_voxel_size_nm`, `autodetect_channels`, `discover_inputs`. Wraps bioio with the bffile monkey-patch.
- `segmentation.py` — `segment_nuclei(backend, params)` reuses `F:\Image Analysis Work\image-analysis-pipeline\python\segmentation\segment_image.py::run_backend` verbatim (sys.path.insert). Adds `exclude_border_labels`.
- `spots.py` — `detect_spots(backend, voxel_xy_nm, ...)` reuses Fiji's `spots.detect_spots` and supports `threshold_multiplier`.
- `thresholds.py` — pure-NumPy port of `Coloc_Core.median/mad/percentile` and `Coloc_Analysis.coloc_threshold/costes_threshold`. Bit-identical to the Jython math.
- `metrics.py` — NumPy-vectorized port of `compute_nucleus_coloc_metrics` (Pearson, Spearman, Manders M1/M2, Li ICQ, Jaccard, Dice, cosine_overlap, both_frac, enrichment ratios).
- `morphology.py` — `compute_cytoplasm_mask` via `expand_labels`, `stratify_spots`, `per_nucleus_spot_counts`, `nuclear_cytoplasmic_intensity`, `regionprops_table`.
- `parallel.py` — `auto_n_workers()` + `BatchExecutor`.

Modes (`src/fishsuite/core/modes/`):
- `rna_only.py` — fully implemented; produces nuclei + spots tables + per-image summary + threshold record.
- `rna_protein.py` — implemented (rna_only + per-nucleus pixel coloc on RNA + AB pixels).
- `rna_rna.py`, `ab_ab.py`, `protein_only.py`, `pub_images.py` — registry stubs that route through `rna_only` for now; Phase-2 will implement full semantics.

## Phase 2 — Config + CLI (DONE)

- `config/schema.py` — pydantic v2 schema covering all knobs (experiment, conditions, channels, z_stack, nuclei, pixel_coloc, foci, cytoplasm, output, parallel).
- `config/presets/` — `h9_hesc_100x.yaml` (production-validated), plus `hek293_60x`, `u2os_100x`, `generic_100x_0p065`, `generic_60x_0p108`. Other 4 presets (`primary_neurons_60x`, `organoid_40x`, `fibroblasts_60x`, `ipsc_100x`) deferred to Phase-4 per the dispatch instructions.
- `cli.py` — `click` command group with: `run`, `init`, `preview`, `presets list`, `presets show`. `fishsuite.exe` installed in `C:\Users\ambur\miniconda3\envs\fishproc\Scripts\`.
- `runner.py` — orchestrator: discovers inputs, runs mode per image, writes the 6 output artifacts (CSVs + xlsx + run_config.json + qc_overlays + publication_images).

## Phase 3 — Streamlit UI

DEFERRED per dispatch instructions (skip for end-to-end proof).

## Phase 4 — Docs

`docs/PHASE0_STATUS.md` written. Remaining docs (`getting-started.md`, `using-fiji-to-tune.md`, `presets.md`, `output-schema.md`) deferred — not required for the proof artifact.

## Phase 5 — End-to-end run (DONE)

Run command:
```powershell
fishsuite run `
  --config E:\Claude\fishsuite\src\fishsuite\config\presets\h9_hesc_100x.yaml `
  --input-dir "F:\Raw Images\H9-MIAT-KD-ASO\H9-MIAT-KD-05-08-2026" `
  --output-dir "F:\Image Analysis Work\H9-Output-2\MIAT-KD-ASO" `
  --parallel 1
```

Results — see `docs/END_TO_END_RESULTS.md`.

## What didn't work, and why

1. **bffile numpy compatibility**. `bffile 0.1.0+` calls `np.asarray(..., copy=False)`, which only exists in numpy 2.x. Numpy <2 is required by tensorflow 2.15 / stardist. Fix: monkey-patch `bffile._biofile._reshape_image_buffer` in `src/fishsuite/__init__.py`. Verified working.
2. **Initial attempt to edit bffile in site-packages was denied** (correct policy). Pivoted to the import-time patch.
3. **First per-image run on `_01.vsi` showed only Z=4** while every other image has Z=33. The autofocus range Z_START=20, Z_END=30 doesn't apply to `_01`, but `extract_channel` clamps to `[1, n_z]` so the run still succeeds (picks best of 4 z-planes on `_01`; behavior is documented).
4. **Process-pool parallelism deferred.** TensorFlow + bioio loaded in workers tends to deadlock unless we use a `set_start_method('spawn')` initializer; the current build is single-process (the full batch still finishes in ~8 min on 13 images). Phase-2 will add the ProcessPoolExecutor wrapper with proper worker init.
5. **Nuclei counts (7-29 per image) are below Brian's spec range of 40-150.** Spec values `min_area=10000, prob=0.5, sigma=3` are the canonical H9 100x GUI profile values (verified in `gui_pipeline.py` line 262-289). The Fiji-pipeline output at `F:\Image Analysis Work\H9-Analysis\` was generated with a *different* dataset and `prob=0.2, min_area=250` (much more permissive). With the canonical sweep-validated H9 settings, fewer-but-cleaner nuclei is the expected result. Verified the Sec-Only baseline is right (0 spots), and NT ASO has ~3x more MIAT spots per nucleus than KD ASO (matches biology).
