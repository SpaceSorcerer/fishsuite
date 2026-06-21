# fishsuite

**Standalone, Fiji-free Python pipeline for RNA-FISH / immunofluorescence (IF) image quantification and colocalization.**

`fishsuite` segments nuclei (Cellpose / StarDist / Otsu), detects RNA-FISH spots (BigFISH LoG or plain LoG), measures per-nucleus spot counts, intensities and nuclear-vs-cytoplasmic distribution, and quantifies colocalization between two channels — including a literature-grounded **rotation "proper-background" null** for spot-vs-diffuse-protein association. It is built for *Homo sapiens* fluorescence microscopy (hESC and cardiomyocyte RNA-FISH / IF), runs from a single `fishsuite` command-line tool or a PySide6 GUI, and writes Excel-explorable result workbooks plus publication-ready figures.

- **Version:** 0.1.0
- **License:** MIT
- **Scope:** *Homo sapiens* only. Tooling, presets and conventions assume human hESC / d8-cardiomyocyte imaging.
- **Status:** `rna_only`, `rna_rna` and `rna_protein` are the production modes (validated end-to-end on H9 hESC, BIN1 d8-cardiomyocyte and MIAT×QKI colocalization data). `ab_ab`, `protein_only` and `pub_images` are Phase-2 stubs (see [Analysis modes](#analysis-modes)).

> This README documents the actual source of this branch. Where a feature is opt-in, gated, or a stub, that is stated explicitly.

---

## Table of contents

1. [Overview](#overview)
2. [Highlights](#highlights)
3. [Installation and environment](#installation-and-environment)
4. [Quickstart](#quickstart)
5. [Analysis modes](#analysis-modes)
6. [The pipeline in depth](#the-pipeline-in-depth)
7. [Colocalization](#colocalization)
8. [CLI reference](#cli-reference)
9. [Configuration and presets](#configuration-and-presets)
10. [Outputs and metrics](#outputs-and-metrics)
11. [Statistics conventions](#statistics-conventions)
12. [Reproducibility](#reproducibility)
13. [Testing](#testing)
14. [Repository layout](#repository-layout)
15. [Citations and methods grounding](#citations-and-methods-grounding)
16. [Scope and limitations](#scope-and-limitations)
17. [Changelog / recent additions](#changelog--recent-additions)

---

## Overview

`fishsuite` is a standalone re-implementation of a Fiji image-analysis pipeline, written in pure Python so it can run headless, in parallel, and without ImageJ. It takes a folder of microscope images (`.vsi`, `.czi`, `.tif`/`.tiff`, `.lif`, `.nd2`, `.oib`/`.oif` — read through `bioio` + Bio-Formats) and produces a complete, condition-aware quantification of an RNA-FISH / IF experiment.

The pipeline is organized around four building blocks:

- **Nucleus segmentation** — Cellpose (incl. AMD-GPU via DirectML), StarDist, or Otsu, with border exclusion, label smoothing, optional Voronoi-expanded cytoplasm, and an opt-in ghost-nucleus rejection rule.
- **Spot detection** — BigFISH Laplacian-of-Gaussian (default) or a plain scikit-image LoG, with physically-sized spot kernels and an auto-threshold scaled by a user multiplier.
- **Colocalization** — whole-nucleus pixel coefficients (Pearson, Spearman, Manders M1/M2, Li ICQ, cosine, Jaccard, Dice), nearest-neighbor spot-to-spot pairing, and a spot-centric partner-intensity statistic tested against **three nulls** (position-randomization, rotation, translation).
- **Nuclear retention / N:C** — per-nucleus nuclear-vs-cytoplasmic spot fractions and intensity ratios — the floor-robust readout for RNA nuclear-retention experiments.

It is for wet-and-dry-lab biologists who acquire RNA-FISH / IF stacks and want reproducible, committee-defensible quantification with explorable Excel deliverables — without manual ImageJ work.

## Highlights

- **Single CLI** (`fishsuite`) with `run`, `preview`, `presets`, `init`, `gui` and three CPU-only post-run utilities (`backfill`, `walkthrough`, `postrun`).
- **PySide6 desktop GUI** (`fishsuite gui`) with channel auto-detection, per-file selection, live YAML preview and readiness checks.
- **Three production analysis modes**: `rna_only`, `rna_rna`, `rna_protein` (the last routes through the two-channel core with antibody-aware handling).
- **Diffuse-antibody handling** — treat a dense nuclear IF channel (e.g. QKI) as an intensity layer instead of spot-detecting it (`detect_antibody_spots: false`).
- **Rotation "proper-background" null** — a registration-destroying, structure-preserving control for whether a partner protein is concentrated at RNA spots *beyond* shared sub-nuclear compartmentalization.
- **Locked, defensible z-handling** — intensity-weighted autofocus with a central-fraction peak guard, and a fixed-N objective-window max-projection.
- **Excel-explorable deliverables** — `analysis_summary.xlsx` (PI report, column glossary, group comparison with Mann-Whitney U + Cliff's delta) and `analysis_raw_data.xlsx`.
- **Reproducibility built in** — global seed, deterministic nulls, `versions.txt` + `command.log` written at run start.
- **Parallel + GPU-aware** — memory/core-aware worker counts; DirectML segmentation forced single-GPU.
- **184 tests** (pytest).

---

## Installation and environment

`fishsuite` is an editable-installed Python package. Two conda environments exist on the lab workstation:

| Env | Purpose |
|---|---|
| `fishproc_dml` | **DirectML / AMD GPU.** Required when a preset sets `nuclei.cellpose_device: directml`. |
| `fishproc` | **CPU fallback.** Use when a preset sets `cellpose_device: cpu`. The post-run utilities (`backfill`, `walkthrough`, `postrun`) are CPU-only and run in either env. |

> GPU is used **once**, during `fishsuite run`, for Cellpose nucleus segmentation (and accelerates the run overall). All post-run utilities are CPU-only. DirectML targets a single AMD GPU — run **one** GPU job at a time.

### Editable install

```powershell
# DirectML / AMD GPU env:
"C:\Users\ambur\miniconda3\envs\fishproc_dml\python.exe" -m pip install -e E:\Claude\fishsuite

# or the CPU env:
"C:\Users\ambur\miniconda3\envs\fishproc\python.exe"     -m pip install -e E:\Claude\fishsuite
```

This installs the `fishsuite` console script (entry point `fishsuite.cli:cli`).

### Dependencies

Runtime dependencies (from `pyproject.toml`):

`numpy>=1.24,<2.0`, `scipy>=1.10`, `scikit-image>=0.22`, `tifffile>=2024.1`, `stardist>=0.9`, `cellpose>=3.0`, `big-fish>=0.6`, `bioio>=3.0`, `bioio-bioformats>=2.0`, `pydantic>=2.5`, `click>=8.1`, `rich>=13.0`, `psutil>=5.9`, `pyyaml>=6.0`, `openpyxl>=3.1`, `pandas>=2.0`, `matplotlib>=3.7`.

Optional extras:

- `dev` — `pytest>=8.0`, `hypothesis>=6.0`
- `gui` — `PySide6>=6.6` (needed for `fishsuite gui`)

Python `>=3.10`.

> **Notes on numpy/Bio-Formats:** numpy is pinned `<2.0` for TensorFlow/StarDist compatibility. On import, `fishsuite` forces a headless matplotlib backend (`MPLBACKEND=Agg`) and applies a small `bffile` numpy-1 compatibility monkeypatch so `bioio` works under numpy 1.x. Bio-Formats runs under a JVM; truncated/0-byte image files (`<512` bytes) are rejected before reaching it (a guard against native JVM crashes).

---

## Quickstart

Pick the preset closest to your experiment, dry-run it to verify the discovered roster, then run for real and check the outputs.

```powershell
# 1) List the built-in presets
fishsuite presets list

# 2) Print one to inspect/clone
fishsuite presets show h9_miat_kd_rerun_iwfocus_2026-05-31

# 3) ALWAYS dry-run first: discover inputs and print the plan, do NOT process
fishsuite run `
  -c "E:\Claude\fishsuite\src\fishsuite\config\presets\h9_miat_kd_rerun_iwfocus_2026-05-31.yaml" `
  -i "F:\Raw Images\H9-MIAT-KD-ASO\<dataset>" `
  -o "F:\Image Analysis Work\H9-Output\RUN_<descriptor>_<timestamp>" `
  --dry-run

# 4) Real run (new, timestamped output dir; raw input dirs are read-only)
fishsuite run `
  -c "E:\Claude\fishsuite\src\fishsuite\config\presets\h9_miat_kd_rerun_iwfocus_2026-05-31.yaml" `
  -i "F:\Raw Images\H9-MIAT-KD-ASO\<dataset>" `
  -o "F:\Image Analysis Work\H9-Output\RUN_<descriptor>_<timestamp>"

# 5) Check outputs (run-root master tables + workbook)
#    per_image_summary.csv, nuclei_metrics.csv, analysis_summary.xlsx,
#    qc_overlays/, publication_images/, masks/
```

The dry-run flag is exactly `--dry-run` (it exists only on `run`). Use a **new, descriptively-named, timestamped output directory for every run** — never overwrite a prior run's folder, and never write into a raw-image directory.

Single-image preview (debug):

```powershell
fishsuite preview "F:\Raw Images\...\image01.vsi" -c <preset>.yaml -o "F:\...\preview01"
```

Or the GUI:

```powershell
fishsuite gui
```

---

## Analysis modes

The mode is set by `channels.analysis_mode`. The dispatcher (`core/modes/__init__.py`) maps each mode name to its implementation:

| `analysis_mode` | Status | Channel roles | What it does |
|---|---|---|---|
| `rna_only` | Production | `dapi`, `rna` | One FISH target. Per-nucleus spot counts (nuclear/cyto/total), `nuclear_spot_fraction`, measured spot sizes, spot/peak intensities, nuclear-vs-cyto intensity (N:C), and thresholded compartment intensity. (Single channel → no pixel-pixel coloc.) |
| `rna_rna` | Production | `dapi`, `rna`, `rna2` | Two distinct FISH targets. Everything in `rna_only` per channel, **plus** spot-to-spot nearest-neighbor pairing, whole-nucleus pixel colocalization (Pearson/Spearman/Manders/Li ICQ/cosine/Jaccard/Dice), active-TS and mature-mRNA proxies, and (gated) partner-intensity + nulls. `rna2` is required. |
| `rna_protein` | Production | `dapi`, `rna`, `antibody` | FISH + IF. **Routes through the `rna_rna` core**: the antibody channel is mapped into the `rna2` slot, the full two-channel analysis runs, then every `rna2_*` output is relabeled `protein_*`. Supports diffuse-antibody handling. |
| `ab_ab` | **Stub** | — | Phase-2 stub. Delegates verbatim to `rna_only.run_one`; does **not** perform two-antibody coloc. |
| `protein_only` | **Stub** | — | Phase-2 stub. Delegates to `rna_only` (treats the configured channel as the single channel). |
| `pub_images` | **Stub** | — | Phase-2 stub. Delegates to `rna_only`. |

### Channel roles and LUT-by-wavelength

Channel indices are configured per dataset (`channels.dapi/rna/rna2/antibody`, `-1` = auto-detect; 0-indexed unless `one_indexed: true`). The lab convention assigns pseudo-color **by emission wavelength, not by probe** — e.g. 640/647 → yellow, 561/568/594 → magenta, 488 → green, DAPI/405 → blue — via `*_lut` fields. Channel `*_label` fields name each channel in filenames and burned-in legends.

### Diffuse-antibody handling

In `rna_protein` mode, `rna_protein.run_one` calls the `rna_rna` core with `rna2_is_antibody=True`. When the antibody is a **diffuse, abundant nuclear protein** (e.g. QKI IF that fills the nucleoplasm rather than forming sparse puncta), spot-detecting it carpets every nucleus with meaningless "spots." Set:

```yaml
foci:
  detect_antibody_spots: false   # rna_protein only
```

With `detect_antibody_spots: false` **and** `rna2_is_antibody` (i.e. in `rna_protein`), the antibody channel is **not** spot-detected (empty spot set). The antibody **pixel** plane is still loaded, so pixel colocalization and the partner-intensity nulls — which sample antibody **pixels** at the RNA1 spots, never antibody spots — are unaffected. Plain `rna_rna` (two real FISH targets) always detects both channels regardless of this flag.

---

## The pipeline in depth

### Segmentation

`segment_nuclei(dapi_2d, backend, params)` dispatches to one of three backends:

- **StarDist** (`backend: stardist`, default model `2D_versatile_fluo`) — knobs: `prob_threshold`, `nms_threshold`, `n_tiles`, `stardist_gauss_sigma`, and an optional post-process (`stardist_postprocess` ∈ `none`/`dilate`/`watershed_otsu`/`watershed_triangle`). StarDist ignores diameter.
- **Cellpose** (`backend: cellpose`, default model `cpsam`) — knobs: `cellpose_diameter_px` (0 = auto), `cellpose_flow_threshold`, `cellpose_cellprob_threshold`. `cellpose_device: directml` enables the torch-DirectML GPU path; `cpu` is the legacy path.
- **Otsu** (`backend: otsu`) — pure thresholding.

> `stardist_model` and `cellpose_model_type` are plain strings in the schema (not restricted enums) — any model name is accepted; the defaults are `2D_versatile_fluo` / `cpsam`.

Common post-segmentation steps: an authoritative `[min_area_px, max_area_px]` area filter applied after smoothing; optional **label-boundary smoothing** (`label_smoothing_radius_px`, morphological close-then-open with a disk, to round StarDist star-convex corners); a **downsample speed lever** (`cellpose_downsample_factor`, applies to any backend); and **border exclusion** (`exclude_border` / `border_margin_px`).

**Ghost-nucleus rejection** (`reject_ghost_nuclei`, opt-in, default off) — a post-detection composite rule that flags a nucleus as an out-of-focus "ghost shell" **only if all three** hold: spot count `== 0`, area `>= reject_ghost_min_area_px` (default 6000 px), and nuclear DAPI CV `<= reject_ghost_max_dapi_cv` (default 0.12). Each condition alone is intentionally insufficient.

### Z-handling

The z mode is `z_stack.mode` ∈ `single`, `maxproj`, `autofocus`, `autofocus_maxproj`, `3d`. Per-slice focus is scored by `focus_metric` (default `variance_of_laplacian` = `var(laplace(plane / mean))`; the plane is mean-normalized so the score depends on gradient structure, not absolute brightness; also `tenengrad`, `normalized_variance`).

- **`autofocus`** — pick one in-focus DAPI plane; RNA/antibody channels are **locked to that same plane** (the nuclear mask and spot xy come from one physical plane, so disk-sampling stays co-registered).
- **`autofocus_maxproj`** — detect a DAPI focus *window*, then max-project that same window for DAPI and the RNA channel(s).

Two locked guards make the focus pick robust on thick / bright-throughout stacks:

- **`autofocus_intensity_weighted: true`** multiplies each slice's score by its mean (`var(laplace(plane/mean)) * mean`), pulling the peak to the bright **and** sharp nuclear plane instead of a dim/noisy stack edge. The plain (unweighted) metric can climb toward dim edge slices and pick garbage.
- **`focus_central_fraction`** (e.g. `0.6`) restricts the **peak search** to the central fraction of the stack, so the window anchor can never be a true edge plane.

Window selection is FWHM-based by default (walk outward while score `>= focus_threshold_frac * peak`, enforce `focus_window_min_slices` / `focus_window_max_slices`), or a **fixed-N centered window** when `focus_window_fixed_n_slices > 0` (constant integration depth across the batch; the window slides rather than shrinks at stack bounds). Per-image z windows can be pinned via `z_stack.file_overrides`.

### Spot detection

`detect_spots(rna, backend, ...)` returns one row per spot.

- **BigFISH LoG** (`backend: bigfish`, default) — auto-threshold from BigFISH, then re-run scaled by `threshold_multiplier` (`threshold = max(1, auto * multiplier)`), or use an explicit `threshold_override`. Spot size is physical: `bigfish_spot_radius_nm` (default 130), `bigfish_spot_radius_z_nm` (default 300), with voxel sizes feeding the LoG sigma and built-in local-max separation.
- **Plain LoG** (`backend: log`) — `log_spot_radius_px` (default 2.5) in pixels, threshold `log_threshold` (default 0.05) scaled by the multiplier.

Per-spot diameters are **measured** (moment-based 2D Gaussian FWHM), not assumed constant. Spot-to-compartment assignment (`in_nucleus`/`in_cytoplasm`, parent `nucleus_id`) is done downstream during stratification; "nuclear-only" analyses then filter on `in_nucleus`. An optional post-detection **floor filter** (`apply_pub_contrast_floor_to_spots`) drops spots whose peak intensity is below the channel's resolved floor.

### Thresholds

`thresholds.py` is a bit-identical port of the Fiji coloc threshold math:

- **MAD** (default): `median + k_mad * MAD` over raw nuclear pixels (**unscaled** MAD; `k_mad` default 2.0).
- **percentile**: a chosen percentile (default 80th).
- **Costes**: the genuine automatic Costes threshold (requires `>=20` pixels; scans descending thresholds until the below-threshold Pearson drops `<=0`). Its fallback uses `1.4826 * MAD` — note this differs from the plain (unscaled) MAD threshold.

Scope is `threshold_scope` ∈ `batch` (one pooled threshold over all images, computed in a pre-pass) or `per_image`. This pixel-coloc threshold is the **internal coloc cut**, distinct from (and usually much lower than) the spot-detection floor.

### Floors and the floor-robust readout

Display/analysis floors live in `output` (`pub_contrast_mode: manual` + `manual_<channel>_min/_max`; "Sam's method" tunes the retention-channel floor on the strongest-retention condition and applies the same floor everywhere). `apply_pub_contrast_floor_to_spots` gates spots below the floor; `apply_pub_contrast_floor_to_analysis` adds above-floor intensity columns.

Because absolute spot counts and intensities are floor-sensitive in **magnitude**, the headline nuclear-retention readout is **`nuclear_spot_fraction`** (% of a nucleus's spots that are nuclear) and the N:C ratios — these are floor-robust in direction. A third, spot-caller-independent view, **thresholded compartment intensity** (`compute_thresholded_compartment_intensity`), integrates the raw intensity of all pixels `>=` a settable floor separately in nucleus and cytoplasm, capturing diffuse + punctate above-floor signal. (See `THRESHOLD_INTENSITY_FEATURE.md`.)

### Parallelism

Worker counts are memory- and core-aware (`min(physical_cores - 2, available_RAM / per_worker_GB, cap)`), with per-worker BLAS/OMP thread caps to avoid oversubscription. **DirectML segmentation is forced to a single worker** (one GPU). `--parallel`/`-p` accepts `auto` (default) or an integer.

---

## Colocalization

`fishsuite` measures colocalization at three levels (in `rna_rna` / `rna_protein`).

### 1. Pixel colocalization (whole-nucleus)

`compute_coloc_metrics` operates on the two channels' pixels inside each nuclear mask, thresholded at the run's pixel-coloc thresholds, and returns: **Pearson** `r`, **Spearman** `rho`, **Li ICQ** (fraction of pixels with co-varying intensity, minus 0.5), **cosine overlap**, **Manders M1/M2**, **Jaccard**, **Dice**, plus reciprocal enrichment ratios and overlap fractions. For a diffuse, abundant partner these whole-nucleus coefficients wash out (the partner fills the nucleus), which is why the spot-centric nulls below are the headline for spot-vs-diffuse cases.

### 2. Spot-to-spot pairing

A `scipy.spatial.cKDTree` nearest-neighbor search pairs spots across channels in 3D: per-spot `nn_distance_um` and `paired_at_<X>um` (X = `spot_coloc.pair_distance_um`, default 0.3 µm), aggregated to per-nucleus / per-image paired fractions and median NN distances. Nuclear + paired spots serve as an active-transcription-site proxy.

### 3. Partner-intensity statistic and nulls (spot-centric, floor-robust)

For each RNA1 spot, the partner channel's mean intensity is sampled in a small disk (`partner_null_disk_px`, default 3.0 px) on the **same z-locked plane**, using **raw** intensity (so the metric does not move when the display/spot floor moves). The per-nucleus observed statistic is the mean of those disk-means over the nucleus's RNA1 spots. It is tested against three nulls (all opt-in; all require `compute_partner_intensity: true`):

**(a) Position-randomization null** (`compute_partner_null_enrichment`) — re-place the same number of spots uniformly within the nucleus and re-sample. Controls for spot count and nuclear geometry. *Limitation (stated in code):* it does **not** control for co-distribution — if both channels prefer the same sub-nuclear regions, enrichment is inflated.

**(b) Rotation "proper-background" null** (`compute_partner_rotation_null`) — the headline control. Instead of randomizing positions, it **rotates the entire RNA1 spot constellation rigidly about its own centroid**, preserving the spot pattern's internal geometry while destroying its registration to the (fixed) partner field. `observed > rotation-null` therefore means the partner is concentrated at the spots **beyond shared sub-nuclear compartmentalization**. Implementation details (function `_rotation_null_for_nucleus` in `core/modes/rna_rna.py`):

- First three rotations are exactly **90°, 180°, 270°**; the remaining `partner_rotation_n - 3` (default 1000 total) are uniform on `[0, 360)`.
- **Keep-N redraw:** any spot rotated out of the nuclear mask is **redrawn** (fresh per-spot angle, up to 40 retries) rather than dropped — dropping shrinks the active spot count and biases enrichment low. Spots that remain unplaceable fall back to their observed position (rare).
- **Usability gate:** a nucleus is usable only if median first-pass in-mask retention `>= partner_rotation_min_retention` (default 0.5) and at least 2 valid draws exist (`rotation_null_usable`). Only usable nuclei contribute to the pooled rotation null.
- **Association fraction** (`partner_rotation_assoc_percentile`, default 95.0) — the fraction of observed spots whose own disk-mean partner exceeds the high-percentile threshold of a single-spot rotation null (chance level `= 1 - pct/100`, i.e. 0.05 at the 95th percentile). Reads as "X% of RNA spots sit in partner-rich neighborhoods beyond the rotation-chance level."

**(c) Translation null** (`compute_partner_translation_null`) — a rigid-shift companion. **Flagged unreliable for dense / space-filling spot patterns** (most shifts push too many points out of the mask, biasing enrichment low). Use rotation as the headline; translation is supplementary at best.

Supporting controls: a **radial profile** (`compute_partner_radial_profile`) reports partner enrichment in concentric rings (`partner_radial_bins_um`, default `[0.25, 0.5, 0.75, 1.0]`); **nucleolus exclusion** (`exclude_nucleolus_from_partner_null`, with `nucleolus.enabled`) removes DAPI-poor nucleolar voids — which an abundant nuclear protein also avoids — from **both** the null positions and the observed spots, so mutual nucleolar avoidance cannot inflate enrichment.

All nulls use fixed seeds with separate RNG streams (position `partner_null_seed`; rotation offset +101; association +404), so toggling one never perturbs another, and the post-run `backfill` reproduces the engine's draws bit-for-bit.

These methods follow the colocalization-with-an-explicit-null tradition: pixel coefficients (Manders 1993; Pearson) require a chance model (Costes 2004; van Steensel 1996), object/spot association is tested against a mask-constrained random placement (Lagache/SODA 2018), and the defensible null must **destroy registration while preserving each channel's own structure** (Dunn 2011; Aaron 2018). See [Citations](#citations-and-methods-grounding). The rotation "proper-background" null is **our own construction** in the registration-destroying tradition — not attributable to a single methods paper.

---

## CLI reference

The console script is **`fishsuite`** (entry point `fishsuite.cli:cli`). It exposes `--version` and the subcommands below. Quoting paths with spaces is required on Windows.

### `fishsuite run`

Run the full pipeline on a folder of images.

| Option | Required | Default | Meaning |
|---|---|---|---|
| `-c`, `--config` | yes | — | Path to a fishsuite YAML config / preset. |
| `-i`, `--input-dir` | yes | — | Folder of images (or folder of subfolders). |
| `-o`, `--output-dir` | yes | — | Where to write outputs. |
| `-p`, `--parallel` | no | `auto` | Worker count: `auto` or an integer (string, resolved downstream). |
| `--resume` | no | off | Skip images that already have outputs. |
| `--dry-run` | no | off | Discover inputs and print the plan; do **not** process. |
| `-v`, `--verbose` | no | off | Print full tracebacks on per-image failures. |

```powershell
fishsuite run -c preset.yaml -i "F:\Raw Images\UD" -o "F:\out\UD_run" --dry-run
```

### `fishsuite preview`

Run the pipeline on a single image (preview / debug); processes the image's parent folder with `parallel=1`.

```powershell
fishsuite preview "F:\Raw Images\UD\img01.vsi" -c preset.yaml -o "F:\out\preview01"
```

Required options: `-c/--config`, `-o/--output-dir`. (No `--dry-run`/`--parallel`/`--resume`.)

### `fishsuite presets`

Manage built-in presets.

```powershell
fishsuite presets list                  # print "<stem>\t<path>" for every shipped *.yaml
fishsuite presets show <name>           # print the named preset's YAML (exit 2 if not found)
```

### `fishsuite init`

Placeholder setup command. Prints info and lists the shipped preset YAMLs; it does not (yet) run an interactive wizard.

### `fishsuite gui`

Launch the PySide6 desktop launcher (requires the `gui` extra / `PySide6`).

### Post-run utilities (CPU-only)

These operate on a **completed run directory** (one containing `run_config.json`, `per_image_summary.csv`, `nuclei_metrics.csv`, `masks/`). They reuse saved masks + spots and never re-segment, re-detect, or touch the GPU. Errors are plain-English (exit 2 for user-fixable issues). Source VSIs are auto-detected from the run's recorded `input_dir`; pass `--staging` if auto-detection fails.

#### `fishsuite backfill`

Retrofit colocalization products onto an existing run.

| Option | Default | Meaning |
|---|---|---|
| `--run` | (required) | Completed run output directory. |
| `--staging` | auto-detect | Folder holding the source VSIs. |
| `--input` | auto-detect | Alternate source folder (rarely needed). |
| `--seed` | 0 | RNG seed for the null/montage (deterministic). |
| `--no-null-draws` | off | Skip writing the null-draw CSV(s). |
| `--no-radial` | off | Skip writing `coloc_radial_profile.csv`. |
| `--no-montage` | off | Skip the partner-enrichment montage PNG. |
| `--rotation` | off | **Also** compute the rotation "proper-background" null (writes `coloc_rotation_null_summary.csv` + `coloc_rotation_null_draws.csv`). Opt-in. |

The three `--no-*` flags are negative-only (products are on by default); `--rotation` is positive opt-in. `backfill` self-validates pooled numbers against the run's stored records and warns (exit 1) on mismatch.

```powershell
fishsuite backfill --run "F:\Image Analysis Work\MIAT-QKI-Coloc\my_run"
fishsuite backfill --run "F:\...\my_run" --rotation --seed 0
fishsuite backfill --run "F:\...\my_run" --no-null-draws --no-radial
```

#### `fishsuite walkthrough`

Build the 8-panel pipeline-walkthrough figure for one representative image.

| Option | Default | Meaning |
|---|---|---|
| `--run` | (required) | Completed run output directory. |
| `--image` | auto-pick | Panel-prefix image key. |
| `--out` | `<run>/figures/07_coloc/79_pipeline_walkthrough.png` | Output PNG path (created). |
| `--staging` / `--input` | auto-detect | VSI source for the rendered panel. |

The figure is a 2×4 grid (600 DPI): (A) DAPI, (B) nucleus segmentation, (C) RNA FISH, (D) RNA spot detection, (E) protein IF, (F) thresholded protein, (G) RNA spots on thresholded protein (freshly rendered at the DAPI autofocus z), (H) merge. Missing panels self-skip with a gray placeholder.

```powershell
fishsuite walkthrough --run "F:\...\my_run"
fishsuite walkthrough --run "F:\...\my_run" --image "g2_wDox_(MIAT_OE)__g2-Dox_01" --out "F:\figures\wt.png"
```

#### `fishsuite postrun`

One-shot "make my figures": runs `backfill` then `walkthrough`, prints a per-step progress line, continues past a failed step, and exits non-zero if any step failed. (No `--no-*`, no `--rotation`, no `--out`; its backfill step writes null-draws + radial + montage but **not** the rotation null.)

```powershell
fishsuite postrun --run "F:\Image Analysis Work\MIAT-QKI-Coloc\my_run"
```

> The legacy module entry points still work: `python -m fishsuite.core.coloc_backfill` and `python -m fishsuite.core.walkthrough_figure`. The subcommands are friendlier wrappers. See `POSTRUN_UTILITIES.md`.

---

## Configuration and presets

Configuration is a Pydantic v2 YAML model (`config/schema.py`). `FishsuiteConfig.from_yaml(path)` loads and validates; omitted blocks fall back to defaults. The config is grouped into blocks: `experiment`, `conditions`, `channels`, `z_stack`, `nuclei`, `pixel_coloc`, `spot_coloc`, `foci`, `cytoplasm`, `nucleolus`, `output`, `parallel`, `qc`, plus top-level `seed` (default 0) and `input_file_subset` (default `[]`).

### Selected fields and real defaults

**`channels`** — `analysis_mode` (default `rna_only`; one of `rna_only`/`rna_protein`/`rna_rna`/`ab_ab`/`protein_only`/`pub_images`); indices `dapi`/`rna`/`rna2`/`antibody`/`antibody2` (default `-1` = auto-detect; `one_indexed: false`); LUTs `dapi_lut`=`blue`, `rna_lut`=`yellow`, `rna2_lut`=`magenta`, `antibody_lut`=`green`, `ab2_lut`=`magenta`; labels default `DAPI`/`RNA1`/`RNA2`/`Protein`/`Protein2`.

**`z_stack`** — `mode` (default `autofocus`; `single`/`maxproj`/`autofocus`/`autofocus_maxproj`/`3d`); `start_slice`/`end_slice` (None); `autofocus_intensity_weighted` (False); `focus_central_fraction` (0.0 = off); `focus_metric` (`variance_of_laplacian`); `focus_threshold_frac` (0.5); `focus_window_min_slices` (3); `focus_window_max_slices` (0); `focus_window_fixed_n_slices` (0); `focus_min_intensity_frac_of_peak` (0.0); `file_overrides` (`{}`).

**`nuclei`** — `backend` (`stardist`; or `cellpose`/`otsu`); `prob_threshold` (0.5); `nms_threshold` (0.5); `stardist_model` (`2D_versatile_fluo`, free string); `cellpose_model_type` (`cpsam`, free string); `cellpose_diameter_px` (0.0 = auto); `cellpose_downsample_factor` (1.0); `cellpose_device` (`cpu`; or `directml`); `min_area_px` (10000); `max_area_px` (1e12); `label_smoothing_radius_px` (0); `exclude_border` (True) / `border_margin_px` (5); `reject_ghost_nuclei` (False) / `reject_ghost_max_dapi_cv` (0.12) / `reject_ghost_min_area_px` (6000).

**`pixel_coloc`** — `threshold_mode` (`mad`; or `percentile`/`costes`); `threshold_scope` (`batch`; or `per_image`); `k_mad` (2.0); `percentile` (80.0).

**`spot_coloc`** — `pair_distance_um` (0.3); `report_nn_distance` (True).

**`foci`** — `backend` (`bigfish`; or `log`); `bigfish_spot_radius_nm` (130.0); `bigfish_spot_radius_z_nm` (300.0); `threshold_multiplier` (0.7); `only_nuclear_spots` (False); `min_sep_px` (1); `log_spot_radius_px` (2.5) / `log_threshold` (0.05); per-channel `rna_overrides`/`rna2_overrides`/`antibody_overrides`; `compute_partner_intensity` (False); `detect_antibody_spots` (True). Null/coloc cluster (all default off unless noted): `compute_partner_null_enrichment` (False), `partner_null_n` (1000), `partner_null_disk_px` (3.0), `partner_null_seed` (0), `exclude_nucleolus_from_partner_null` (False), `save_partner_null_draws` (False), `compute_partner_radial_profile` (False), `partner_radial_bins_um` (`[0.25, 0.5, 0.75, 1.0]`), `compute_partner_rotation_null` (False), `partner_rotation_n` (1000), `partner_rotation_seed` (0), `partner_rotation_min_retention` (0.5), `partner_rotation_assoc_percentile` (95.0), `compute_partner_translation_null` (False), `save_partner_rotation_null_draws` (False).

**`cytoplasm`** — `enabled` (True); `voronoi_max_expansion_px` (80); `measure_nc_ratio` (True).

**`nucleolus`** — `enabled` (False); `intra_nuclear_percentile` (25.0); `min_area_um2` (1.0); `max_area_frac_of_nucleus` (0.6); `closing_radius_px` (2); `min_border_distance_px` (5).

**`output`** — save toggles `save_qc_overlays`/`save_per_image_csv`/`save_masks`/`save_publication_images` (True), `save_publication_tifs` (False); `pub_contrast_mode` (`auto_batch`; or `auto_per_image`/`manual`/`reference_image`) with percentile knobs (`pub_contrast_floor_pct` 98.0, `pub_contrast_ceil_pct` 99.9, `pub_contrast_dapi_floor_pct` 40.0, `pub_contrast_dapi_ceil_pct` 99.9, `pub_contrast_rna_floor_bump_pct` 10.0); manual floors `manual_<dapi|rna|rna2|antibody>_min/_max` (None); `apply_pub_contrast_floor_to_analysis` (False) / `apply_pub_contrast_floor_to_spots` (False); `rna_intensity_threshold` / `rna2_intensity_threshold` (None); `scalebar_um` (50.0).

**`qc`** — `qc_min_nuclei` (5); `qc_saturated_frac` (0.01); `qc_min_focus_score` (0.0 = focus never flags). QC columns are advisory only; no image is ever dropped.

**`parallel`** — `workers`/`seg_workers` (`auto`); `main_workers` (1); `threads_per_worker` (4).

### Cloning a preset

Clone the closest shipped preset and retarget channels/floors per dataset — never write a config from scratch, and don't improvise z-handling, floor, or output-dir naming.

```powershell
fishsuite presets show bin1_d8cmyo_100x > my_new_preset.yaml
# edit channel indices/labels/LUTs, z-handling, and floors, then run --dry-run
```

### Built-in presets

Shipped presets live in `src/fishsuite/config/presets/`. Representative ones:

| Preset | Mode | Purpose |
|---|---|---|
| `h9_hesc_100x.yaml` | rna_only | H9 hESC 100x baseline (DAPI + RNA). |
| `h9_miat_kd_rerun_iwfocus_2026-05-31.yaml` | rna_only | Committee-grade H9 MIAT NT-vs-KD with the objective windowed-MIP z-handling. |
| `h9_miat_kd_aso_cellpose.yaml`, `..._stardist_ds3.yaml`, `h9_miat_kd_0505_rerun_*`, `h9_miat_kd_0506_descriptive_*`, `h9_miat_kd_aso_DECONV_*` | rna_only | H9 MIAT-KD variants (backend / dataset / deconvolution-specific). |
| `miat_oe_ud_g2_rna_only_2026-06-03.yaml` | rna_only | Undifferentiated hESC g2 MIAT overexpression (Dox-inducible) FISH. |
| `bin1_d8cmyo_100x.yaml` | rna_rna | BIN1 d8 cardiomyocyte exon/intron retention (KO vs WT). |
| `bin1_d8cmyo_*` (XRN2, QKIKO4-2, RNaseTreat variants) | rna_rna | BIN1 d8-cMyo follow-ups. |
| `h9_rna_rna_test.yaml`, `..._labeled.yaml` | rna_rna | Two-channel infrastructure validation on H9 data (the 561 channel is an unused stand-in — not a real two-probe experiment). |
| `miat_qki_coloc_ud_g2_PLAIN_strictMIAT_2026-06-05.yaml` | rna_protein | MIAT × QKI coloc, g2 control vs MIAT-OE, diffuse-partner; rotation/null products on. |
| `miat_qki_coloc_ud_ALLARMS_PLAIN_strictMIAT_2026-06-05.yaml` | rna_protein | MIAT × QKI coloc, all three dCas9-VPR arms × ±Dox in one run. |
| `miat_qki_coloc_ud_g2_rna_protein_2026-06-04.yaml` | rna_protein | Earlier g2 MIAT × QKI coloc pilot. |
| `miat_qki_coloc_d4CM_decon_2026-06-20.yaml`, `..._d8CM_...`, `..._d15CM_...` | rna_protein | MIAT × QKI coloc on d4/d8/d15 cardiomyocytes (deconvolved). |
| `miat_qki_EXPLORATORY_qkifoci_*` | rna_protein | Exploratory QKI-foci tuning variants. |
| `generic_60x_0p108.yaml`, `generic_100x_0p065.yaml` | rna_only | Generic single-FISH starting points at the named pixel sizes. |
| `hek293_60x.yaml`, `u2os_100x.yaml` | rna_only | Generic cell-line single-FISH templates. |

> `presets list` prints every `*.yaml` in the presets folder, which may include local scratch presets (e.g. `_tmp_*`); those are not official shipped presets.

---

## Outputs and metrics

A run writes a complete, condition-aware output tree. Per-image files are condition-prefixed (`<condition>__<stem>__<suffix>`); an optional `output.prefix` is prepended to all names.

### Directory layout

```
<output_dir>/
  per_image_summary.csv      # master, one row per image
  nuclei_metrics.csv         # master, one row per nucleus
  spot_metrics.csv           # master, one row per spot (has a `channel` column)
  cell_morphology.csv        # master, one row per nucleus (shape)
  thresholds.csv             # master, per-image threshold record
  coloc_null_draws.csv       # only when save_partner_null_draws is on
  coloc_radial_profile.csv   # only when the radial-profile feature is on
  coloc_rotation_null.csv    # only when save_partner_rotation_null_draws is on
  analysis_summary.xlsx      # PI report workbook
  analysis_raw_data.xlsx     # raw-data workbook
  run_config.json            # full resolved config + provenance
  versions.txt               # tool versions + seed (written at run start)
  command.log                # argv + config + seed (written at run start)
  qc_overlays/               # QC composite + segmentation-on-DAPI PNGs
  per_image_csv/             # per-image nuclei + spot CSVs (+ optional channel-split spot CSVs)
  masks/                     # per-image label/mask TIFFs + per-image thresholds.csv
  publication_images/        # per-channel pseudo-colored PNGs (+ optional 16-bit TIFs) and merges
  pipeline_walkthrough/      # step01..stepNN methods micrographs
  nuclei_popouts/            # representative single-nucleus crops
  nucleolus_overlay/         # nucleolus overlays (populated only in nucleolus-aware runs)
  _downstream_plots.log      # log of the optional downstream figure step
```

> **`figures/` is produced by an optional bundled downstream step, not by fishsuite itself.** At the end of a run, the runner shells out to an external script (`python -m analysis.single_condition_plots`) that creates `figures/` (and subfolders such as `figures/07_coloc/`). The `walkthrough` utility also writes its figure to `figures/07_coloc/79_pipeline_walkthrough.png` by default (creating that path).

### Master CSVs and key columns

- **`per_image_summary.csv`** (the replicate-level table): per-image spot totals and per-nucleus rollups — `total_spots`, `total_spots_rna1/rna2`, `mean/median/cv_spots_per_nucleus(_rna1/_rna2)`, `frac_nuclear_rna1/rna2`, `frac_nuclei_with_ge_{1,5,10}_spot(s)`, intensity rollups, pairing (`paired_fraction_*_at_0p3um`, `median_nn_distance_*_um`), and — when the partner-null features are on — the **pooled null summary columns** (e.g. `rna2_pooled_enrichment_vs_null_at_rna1_spots`, `rna2_pooled_rotation_enrichment_at_rna1_spots`, with their pooled null mean / z / empirical p; relabeled `protein_*` in `rna_protein`).
- **`nuclei_metrics.csv`** (per nucleus): `rna_spot_count`, `nuclear_spot_count`, `cyto_spot_count`, **`nuclear_spot_fraction`**, `nuclear_spot_density_per_um2`, raw intensities and `rna_nc_ratio` / `nc_ratio_total_intensity_*`, spot peak-intensity aggregates, the thresholded-compartment columns (`rna_thresh_total/mean_intensity_*`, `_pos_area_px_*`, `_pos_fraction_*`, `rna_thresh_floor`, plus `rna2_thresh_*`/`protein_thresh_*`), pairing, the Manders/Pearson/etc. per-nucleus coloc columns, and — when on — the per-nucleus partner columns (`rna2_enrichment_vs_null_at_rna1_spots`, `rna2_rotation_enrichment_at_rna1_spots`, `rna2_rotation_assoc_fraction_at_rna1_spots`, `rotation_null_usable`).
- **`spot_metrics.csv`** (per spot): `channel` (`rna1`/`rna2`/`protein`), `spot_id`, `nucleus_id`, `in_nucleus`, `in_cytoplasm`, `x_px`/`y_px`/`z_slice`, `spot_peak_intensity`, measured `spot_fwhm_px`/`spot_diameter_um`/`spot_area_px`, `nn_distance_um`, `paired_at_0p3um`.
- **`cell_morphology.csv`**: per-nucleus `area_um2`, `perimeter_um`, `circularity`, `aspect_ratio`, `roundness`, `elongation`, `solidity`, `feret_max_um`/`feret_min_um`.
- **`thresholds.csv`**: per-image per-channel threshold provenance (method/mode/`k_mad`/scope/value, BigFISH params, channel labels).
- **Coloc CSVs**: `coloc_null_draws.csv` and `coloc_rotation_null.csv` carry per-iteration pooled draws (`image, condition, iter, pooled_null_value, pooled_obs`); `coloc_radial_profile.csv` carries per-ring stats (`image, condition, ring_um, obs_mean, null_mean, null_sd, enrichment, z, n_spots`).

> The full, authoritative per-column glossary (name / type / unit / description) is embedded in the workbook README sheet and in `core/excel_report.py`.

### Excel workbooks

- **`analysis_summary.xlsx`** — 10 sheets: `README` (provenance + per-column glossary), `Executive_Summary`, `PI_Focus`, `Comparison_Table` (group comparison with Mann-Whitney U p-value + Cliff's delta), `Per_Image_Summary`, `Per_Nucleus_Metrics`, `Per_Spot_Metrics`, `Cell_Morphology`, `Thresholds`, `Run_Config`. Data sheets have a bold header, frozen header row, auto-fit widths, numeric formats, and condition color-coding; generic channel tokens are substituted with the preset's channel labels.
- **`analysis_raw_data.xlsx`** — 5 sheets: `Raw_README` + the 4 data sheets (`Per_Image_Summary`, `Per_Nucleus_Metrics`, `Per_Spot_Metrics`, `Cell_Morphology`).

### `run_config.json`

Records identity/provenance (`package`, `version`, `python_version`, `platform`, `run_start_utc`/`run_end_utc`, `runtime_s`, `n_workers`, `config_path`, `input_dir`, `output_dir`, `n_images`, `failures`), the full resolved config (`config_resolved`), Fiji-parity uppercase keys, output toggles, the resolved publication-contrast (`batch_contrast`), and top-level channel labels.

---

## Statistics conventions

- **The per-image mean is the replicate unit.** Per-nucleus values are pseudoreplicated (Lord 2020); inference is at the image/replicate level. SuperPlots show per-nucleus points shaded by image, with image-means as the tested replicates.
- **Report `nuclear_spot_fraction` / N:C as the headline** for nuclear-retention experiments ("at floor N"); absolute counts/intensities are floor-sensitive support, robust in direction but not magnitude.
- **Never compare absolute antibody/RNA intensity across conditions or sections** when laser power was re-tuned per section. Report counts, fractions, and within-nucleus ratios only.
- **Colocalization is reported with its null** — effect size (observed vs null) plus an empirical p, never a bare coefficient; for diffuse-partner cases the rotation-null columns are the headline.

---

## Reproducibility

- A global `seed` (default 0) seeds Python `random`, NumPy, `PYTHONHASHSEED`, and torch (via `core/repro.py`) at the very start of a run.
- Every stochastic null uses `numpy.random.default_rng` with a fixed seed and a separate RNG stream per null family, so results are deterministic and `backfill` reproduces the engine's draws bit-for-bit.
- Z-window selection is deterministic.
- `versions.txt` (fishsuite version, seed, Python, platform, and installed versions of numpy/scipy/scikit-image/pandas/cellpose/stardist/big-fish/torch/torch-directml/bioio/bioio-bioformats) and `command.log` (full `sys.argv`, config path, output dir, seed, mode, z-mode) are written at run start, so provenance survives even if a run later crashes.
- Use a new, descriptively-named, timestamped output directory per run; raw input directories are read-only.

---

## Testing

Run the suite with pytest from the repo root (in either env):

```powershell
"C:\Users\ambur\miniconda3\envs\fishproc_dml\python.exe" -m pytest -q
```

**184 tests collected.** Coverage areas (one test file each): autofocus z-lock, fixed-N focus window, partner-intensity performance, position-randomization null coloc, partner radial profile, rotation null, threshold-intensity feature, output/Excel schema, reproducibility (`repro`), QC flags, nucleolus performance-equivalence, rna_protein depth, coloc backfill, CLI post-run subcommands, walkthrough figure, and a general smoke test.

---

## Repository layout

```
src/fishsuite/
  __init__.py            # version, headless matplotlib, bffile numpy-1 patch
  cli.py                 # Click CLI (run/preview/presets/init/gui/backfill/walkthrough/postrun)
  config/
    schema.py            # Pydantic v2 config model
    presets/             # shipped *.yaml presets
  core/
    io.py                # bioio reader, channel autodetect, z-window logic
    segmentation.py      # cellpose / stardist / otsu; ghost-nucleus rule
    spots.py             # BigFISH / LoG spot detection
    thresholds.py        # MAD / Costes thresholds (Fiji-bit-compatible)
    metrics.py           # Pearson/Manders/Li-ICQ/Jaccard/Dice; thresholded compartment intensity
    morphology.py        # Voronoi cytoplasm, N/C stratification, regionprops
    nucleolus.py         # DAPI-low nucleolus detection + chromatin texture
    parallel.py          # ProcessPool worker count + thread caps
    repro.py             # seeds, versions.txt, command.log
    qc.py                # advisory per-image QC flags
    output.py            # per-image PNG/TIF/CSV/mask writers
    excel_report.py      # the two Excel workbooks + column glossaries
    coloc_backfill.py    # CPU post-run coloc products (orchestrates rna_rna null helpers)
    walkthrough_figure.py# the 8-panel pipeline-walkthrough figure
    modes/
      __init__.py        # mode registry / dispatch
      rna_only.py        # single-channel mode (+ the floor resolver helpers)
      rna_rna.py         # two-channel core (partner-intensity + ALL nulls live here)
      rna_protein.py     # antibody->rna2 remap wrapper over rna_rna
      ab_ab.py           # Phase-2 stub -> rna_only
      protein_only.py    # Phase-2 stub -> rna_only
      pub_images.py      # Phase-2 stub -> rna_only
  gui/                   # PySide6 desktop launcher (main, state, widgets, readiness, runner_proc)
tests/                   # pytest suite (184 tests)
file_map.md              # per-file orientation index
POSTRUN_UTILITIES.md     # beginner guide to backfill/walkthrough/postrun
THRESHOLD_INTENSITY_FEATURE.md  # the thresholded-compartment-intensity feature
```

---

## Citations and methods grounding

The colocalization design follows the "coefficient-with-an-explicit-null" tradition. All references below were verified against PubMed (PMID + DOI):

- **Manders, Verbeek & Aten 1993** — co-occurrence coefficients M1/M2. *J Microsc* 169(3):375-382. PMID 33930978 · DOI 10.1111/j.1365-2818.1993.tb03313.x
- **van Steensel et al. 1996** — cross-correlation / lateral-shift control for nuclear-compartment coloc. *J Cell Sci* 109(4):787-792. PMID 8718670 · DOI 10.1242/jcs.109.4.787
- **Costes et al. 2004** — automatic threshold + randomization significance test. *Biophys J* 86(6):3993-4003. PMID 15189895 · DOI 10.1529/biophysj.103.038422
- **Dunn, Kamocka & McDonald 2011** — practical guide to evaluating colocalization (registration-destroying controls). *Am J Physiol Cell Physiol* 300(4):C723-C742. PMID 21209361 · DOI 10.1152/ajpcell.00462.2010
- **Aaron, Taylor & Chew 2018** — co-occurrence vs correlation; pixel-coloc / resolution limits. *J Cell Sci* 131(3):jcs211847. PMID 29439158 · DOI 10.1242/jcs.211847
- **Lagache et al. 2018 (SODA)** — object-based spatial statistics with a mask-constrained random-placement null. *Nat Commun* 9(1):698. PMID 29449608 · DOI 10.1038/s41467-018-03053-x
- **Lord et al. 2020 (SuperPlots)** — replicate-level reporting, anti-pseudoreplication. *J Cell Biol* 219(6):e202001064. PMID 32346721 · DOI 10.1083/jcb.202001064

> **On the rotation "proper-background" null:** it is **not attributable to a single paper.** It is a registration-destroying, structure-preserving control built for this pipeline, in the spirit of the randomization-null tradition (Costes 2004; van Steensel 1996) and the registration-destroying principle articulated by Dunn 2011 and Aaron 2018. Cite it as our own construction, framed against those verified principles — not as a named published method.

---

## Scope and limitations

- **Homo sapiens only.** Presets, channel conventions and validation data are human hESC / cardiomyocyte RNA-FISH / IF. There is no multi-species mode.
- **Imaging is a lower bound on co-occupation for a diffuse, abundant partner.** In any diffraction-limited voxel, the *bound* fraction of an abundant nuclear protein is small relative to the diffuse pool, so a sparse RNA target yields low *apparent* colocalization even when association is real. A modest or null coloc result is sensitivity-limited and does not exclude interaction; report effect size + nulls and state the diffraction/abundance caveat.
- **`ab_ab`, `protein_only`, `pub_images` are stubs** that currently delegate to `rna_only`; they do not implement their named behaviors.
- **`figures/` depends on an external downstream script** (`analysis.single_condition_plots`); the core `fishsuite` package produces the CSVs, masks, QC overlays, publication images, walkthrough steps, and Excel workbooks.
- **Splicing tools disagree by design** is not relevant here — but, analogously, colocalization coefficients answer different questions (co-occurrence vs correlation); never present one coefficient as "the answer" without its null.

---

## Changelog / recent additions

- **Rotation "proper-background" null** — native, default-off rotation/translation nulls (`compute_partner_rotation_null` / `partner_rotation_*`), validated against an adversarial prototype; the headline control for spot-vs-diffuse-protein association beyond shared compartmentalization.
- **Self-sufficient coloc outputs** — the canonical MIAT/QKI presets emit `coloc_null_draws.csv` + `coloc_radial_profile.csv` themselves, so `backfill` is only needed to retrofit older runs.
- **Post-run utilities** — `backfill`, `walkthrough`, `postrun` as friendly CPU-only wrappers over the standalone modules, with plain-English errors.
- **Reproducibility / QC hardening** — global seed + `versions.txt` + `command.log` at run start, and additive advisory `qc_*` columns (never drop an image).
- **Thresholded compartment intensity** — a spot-caller-independent third intensity readout (`rna_thresh_*` / `rna2_thresh_*`).
- **Diffuse-antibody handling** — `detect_antibody_spots: false` replaces the old threshold-multiplier hack for dense nuclear IF channels.

---

*fishsuite is dissertation-adjacent research tooling. For workflow conventions (z-handling, floors, output-dir naming, one-GPU-at-a-time), see `POSTRUN_UTILITIES.md`, `THRESHOLD_INTENSITY_FEATURE.md`, and `file_map.md`.*
