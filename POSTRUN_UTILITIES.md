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

There are two of them, plus a one-shot "do everything" button:

| Command | What it makes |
|---|---|
| `fishsuite backfill`    | The extra colocalization tables + a QKI enrichment montage |
| `fishsuite walkthrough` | The 8-panel "how the pipeline works" figure |
| `fishsuite postrun`     | Runs **both** of the above in one go (the "just make my figures" button) |

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
```

If something is wrong (e.g. the run isn't finished, or the images can't be
found), the command prints a plain-English message telling you what to do -
not a Python error dump.

---

## Power-user note (back-compat)

The original module entry points still work and behave identically:

```
python -m fishsuite.core.coloc_backfill   --run-dir <run> --staging <staging>
python -m fishsuite.core.walkthrough_figure --run-dir <run> --staging <staging>
```

The `fishsuite backfill` / `fishsuite walkthrough` / `fishsuite postrun`
subcommands are just friendlier wrappers around these.
