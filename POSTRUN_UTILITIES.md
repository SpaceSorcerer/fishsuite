# fishsuite post-run utilities

A short, friendly guide to the "post-run" commands - the ones you run **after** a
pipeline run has finished, to make extra figures and tables from the results you
already have. You do not need to know the internals to use these.

---

## What is a "post-run utility"?

When you run the main pipeline (`fishsuite run ...`) on a folder of microscope
images, it produces an **output folder** (a "run directory"). That folder holds
everything the analysis found: the per-image table, the per-spot table, the
saved nucleus masks, and a `figures/` folder.

A **post-run utility** takes one of those finished run folders and produces
*additional* products from it - without re-doing the slow, GPU-heavy analysis.
They **reuse** the nucleus masks and the detected MIAT spots that the run already
saved, so they are fast and safe to run as many times as you like.

There are several, plus a one-shot "do everything" button:

| Command | What it makes |
|---|---|
| `fishsuite backfill`    | The extra colocalization tables + a QKI enrichment montage |
| `fishsuite walkthrough` | The 8-panel "how the pipeline works" figure |
| `fishsuite postrun`     | Runs backfill + walkthrough in one go (the "just make my figures" button) |
| `fishsuite singlecell`  | Single-cell (per-nucleus) treatment analysis: dose-response, matched-abundance, heterogeneity, saturation (Excel + SuperPlots) |
| `fishsuite pixelpattern`| Per-nucleus pixel-pattern metrics + a stain-QC panel (Gini, perinuclear/radial index, foci-band, intensity sweep) |

When would you use them?

- You have an **older run** that was made before these extra outputs existed, and
  you want to add them now (this is exactly what `backfill` is for).
- You want to regenerate or tweak a publication figure from an existing run.
- You just finished a run and want all the figures in one command (`postrun`).

---

## Do I need the GPU? (the short answer: no, not for these)

The GPU is used **once**: during the main `fishsuite run`, for the heavy work -
Cellpose nucleus segmentation and spot detection. That is the only step that
needs it.

- **`backfill` is CPU-only.** It does not touch the GPU at all. It re-reads only
  the QKI/protein channel pixels and recomputes the QKI-at-MIAT statistics,
  **reusing** the nucleus masks and MIAT spots the run already saved. It never
  re-segments or re-detects.
- **`walkthrough`** is also CPU-only - it stitches the run's own per-step images
  together and re-renders a single panel from pixels.
- **`singlecell` is CPU-only and reads NO images at all.** It only reads the
  run's `nuclei_metrics.csv` (the per-nucleus table the run already wrote).
- **`pixelpattern` is CPU-only.** Exactly like `backfill`, it reuses the saved
  nucleus masks and re-reads only the raw channel pixels; it never re-segments.

And going forward, you usually **won't even need `backfill`**: the canonical
MIAT x QKI colocalization presets now turn the extra-output flags on, so a new
run writes `coloc_null_draws.csv` and `coloc_radial_profile.csv` itself. Backfill
exists to *retrofit* runs made before that was the default.

---

## Where do the outputs go?

Everything is written **inside the run directory you point at**.

`backfill` writes:

| File | What it is |
|---|---|
| `coloc_null_draws.csv`     | The 1000 pooled random-null draws (the null distribution) |
| `coloc_null_summary.csv`   | Pooled enrichment / z-score / empirical p-value, per image |
| `coloc_radial_profile.csv` | QKI enrichment in concentric rings around each MIAT spot |
| `figures/07_coloc/79_coloc_qki_montage_at_miat_vs_random.png` | The mean QKI-enrichment montage (MIAT vs matched-random positions) |

`backfill` also prints a **self-validation table**: it re-derives the pooled
numbers and checks they reproduce the values the original run stored. If any
image fails, it says so loudly - inspect before trusting the output.

`walkthrough` writes (by default):

| File | What it is |
|---|---|
| `figures/07_coloc/79_pipeline_walkthrough.png` | The labeled 8-panel pipeline-walkthrough figure |

`singlecell` writes into `deliverables/singlecell/`:

| File | What it is |
|---|---|
| `single_cell_analysis.xlsx` | Explorable workbook: How_to_read, Dose_response, Dose_binned, Matched_abund, Heterogeneity, Distribution, Saturation |
| `figures/*.png`             | NT-vs-perturbation SuperPlots + dose-response scatters + a saturation composite |
| `SINGLE_CELL_FINDINGS.md`   | Plain-language readout (top dose-responses + the saturation headline) |

`pixelpattern` writes into `deliverables/pixelpattern/`:

| File | What it is |
|---|---|
| `pixel_pattern_metrics.csv`             | One row per nucleus: Gini, top-5/10%, perinuclear index, foci-band counts, etc. |
| `_radial_per_nucleus.csv` / `_sweep_per_nucleus.csv` | Radial (edge->center) profiles and decile intensity sweeps |
| `foci_band_summary.csv`                 | Partner spots/nucleus above intensity floors, with the secondary-only anchor |
| `nt_vs_condition_wellmeans_welch.csv`   | Control-vs-perturbation Welch on per-well means |
| `pixel_pattern.xlsx`                     | Explorable workbook (How_to_read first) |
| `figures/*.png`                         | SuperPlots + a stain-QC panel (perinuclear index vs secondary-only) + radial / sweep plots |
| `PIXEL_PATTERN_FINDINGS.md`             | Plain-language readout incl. the stain QC |

### A note on the canonical coloc metric

The **rotation "proper background" null** is the *documented canonical*
colocalization statistic for this pipeline: it keeps each nucleus's own spot
constellation and rotates it about the nucleus centroid, so it corrects for spot
density and is the density-robust measure of QKI-at-MIAT association. The
`singlecell` saturation headline and `pixelpattern` are read *alongside* it; they
do not replace it. (`backfill --rotation` retrofits it onto older runs; the
canonical presets now write it during the run.)

---

## Copy-paste examples

Replace the run path with your own. (Windows paths with spaces need quotes.)

### Make everything in one go (recommended)

```
fishsuite postrun --run "F:\Image Analysis Work\MIAT-QKI-Coloc\my_run_20260605"
```

This runs `backfill` then `walkthrough`, prints a progress line per step, and
ends with a list of every file it produced. In the common case this is all you
need - the source images are found automatically from what the run recorded.

### Just the colocalization tables + montage (CPU-only)

```
fishsuite backfill --run "F:\Image Analysis Work\MIAT-QKI-Coloc\my_run_20260605"
```

Only the montage (skip the CSV tables):

```
fishsuite backfill --run "F:\...\my_run_20260605" --no-null-draws --no-radial
```

If the source images can't be found automatically, point at the VSI folder:

```
fishsuite backfill --run "F:\...\my_run" --staging "E:\Claude\fishsuite\_staging_UD_ALLARMS"
```

### Just the walkthrough figure

```
fishsuite walkthrough --run "F:\Image Analysis Work\MIAT-QKI-Coloc\my_run_20260605"
```

Choose a specific image, or a custom output path:

```
fishsuite walkthrough --run "F:\...\my_run" --image "g2_wDox_(MIAT_OE)__g2-Dox_01"
fishsuite walkthrough --run "F:\...\my_run" --out  "F:\figures\walkthrough.png"
```

### Single-cell (per-nucleus) treatment analysis (CPU; no images)

```
fishsuite singlecell --run "F:\Image Analysis Work\MIAT-QKI-Coloc\my_run"
```

The abundance axis and the two groups are auto-detected; override them if needed:

```
fishsuite singlecell --run "F:\...\my_run" --abundance-col nuclear_spot_count --group-a NT --group-b KD
```

### Per-nucleus pixel-pattern metrics + stain QC (CPU; reuses masks)

```
fishsuite pixelpattern --run "F:\Image Analysis Work\MIAT-QKI-Coloc\my_run"
```

Pick the CLEAN secondary-only well for the stain-QC control (some secondary-only
wells can be contaminated - designate the good one by a substring of its name):

```
fishsuite pixelpattern --run "F:\...\my_run" --secondary-match well12
```

---

## Common options

- `--run <folder>` (required) - the **output folder a completed run produced**
  (the one that contains `run_config.json`, `per_image_summary.csv`, `masks/`,
  `figures/`). Not the raw-image folder.
- `--staging <folder>` - where the source `.vsi` images live. **Auto-detected**
  from the run if you leave it off; pass it only if auto-detection fails.
- `--input <folder>` - an alternate source folder for the images (rarely needed).
- `--seed <n>` - random seed for the null/montage (kept deterministic; default 0).
- `backfill` toggles: `--no-null-draws`, `--no-radial`, `--no-montage` to skip
  any of the three products.

Every command has friendly `--help` with examples:

```
fishsuite --help
fishsuite backfill --help
fishsuite walkthrough --help
fishsuite postrun --help
fishsuite singlecell --help
fishsuite pixelpattern --help
```

`singlecell` / `pixelpattern` options worth knowing:

- `--abundance-col <col>` (singlecell) - the per-nucleus "dose" axis
  (default `nuclear_spot_count`).
- `--group-a <label>` / `--group-b <label>` (singlecell) - the control and
  perturbation groups to compare (default: auto-detected, control-like first).
- `--secondary-match <substr>` (pixelpattern) - restrict the secondary-only
  stain-QC control to the wells whose name contains the substring (e.g.
  `well12`). Use this to pick the CLEAN secondary-only well.
- `--out-subdir <name>` - the sub-folder under `deliverables/` to write into.
- `--no-figures` / `--no-excel` - skip the figures / the workbook.

If something is wrong (e.g. the run isn't finished, or the images can't be
found), the command prints a plain-English message telling you what to do -
not a Python error dump.

---

## Power-user note (back-compat)

The original module entry points still work and behave identically:

```
python -m fishsuite.core.coloc_backfill    --run-dir <run> --staging <staging>
python -m fishsuite.core.walkthrough_figure --run-dir <run> --staging <staging>
python -m fishsuite.core.singlecell         --run-dir <run>
python -m fishsuite.core.pixel_pattern      --run-dir <run> [--staging <raw>]
```

The `fishsuite backfill` / `walkthrough` / `postrun` / `singlecell` /
`pixelpattern` subcommands are just friendlier wrappers around these.
