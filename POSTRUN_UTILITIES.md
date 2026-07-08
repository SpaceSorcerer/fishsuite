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

---

## Detection & z-stack robustness (config-level, 2026-07-05)

These are **not** post-run utilities — they are options set in the preset YAML
that make a *run* more robust. Documented here because this is the field guide
people already reach for. All three are **additive and default-off**: leaving
them at their defaults reproduces the previous behaviour byte-for-byte.

### 1. RNA-anchored autofocus (`z_stack.autofocus_channel`)

In single-plane `autofocus` mode the pipeline picks **one** optical section and
reads every channel (DAPI segmentation + RNA + antibody) at that same z — this
one-plane rule is what keeps colocalization honest.

By default the plane is chosen by focusing on **DAPI**. That is usually right,
but on some z-stacks the DAPI-sharpest plane is **not** where the dim
single-molecule RNA (e.g. MIAT) is in focus. Reading MIAT out of focus makes its
BigFISH auto-threshold collapse and carpets the field with **thousands of noise
"spots"** (the well11 MIAT-KD ~1500-spots/nucleus artifact). QKI, being a bright
diffuse antibody signal, is unaffected — so you see a MIAT-only blow-up.

The fix is to let the **RNA** channel choose the plane:

```yaml
z_stack:
  mode: autofocus
  autofocus_channel: rna         # dapi (default) | rna | auto
  autofocus_intensity_weighted: true   # keep your existing thick-stack setting
  # start_slice / end_slice: the same +/-10% edge guard as before, still honored
```

- `autofocus_channel: dapi` — **default, unchanged.** Pick the sharpest DAPI
  plane, lock all channels to it.
- `autofocus_channel: rna` — pick the sharpest **RNA1** plane instead, then lock
  DAPI + RNA + antibody to it. Use this when the RNA target focuses on a
  different plane than DAPI.
- `autofocus_channel: auto` — compute a per-image RNA signal-quality score and
  RNA-anchor **only when it clears a threshold**, else fall back to DAPI-anchor.
  Per-image reporting tells you which channel was used for each FOV:

  ```yaml
  z_stack:
    mode: autofocus
    autofocus_channel: auto
    autofocus_auto_rna_quality_min: 3.0   # RNA dynamic-range gate (default 3.0)
  ```

**What gets reported (only when `autofocus_channel` is `rna` or `auto`).** New
columns appear in `per_image_summary.csv` so the choice is auditable:

| Column | Meaning |
|---|---|
| `z_autofocus_mode` | the config value (`rna` / `auto`) |
| `z_autofocus_channel_used` | which channel actually chose the plane (`rna` / `dapi`) |
| `z_plane` | the 1-indexed absolute plane used |
| `rna_focus_score` | RNA sharpness at that plane (variance of Laplacian) |
| `rna_dynamic_range` | RNA spot-callability / SNR, `(p99.9 - median) / (1.4826*MAD)` |
| `rna_n_confident_spots` | number of RNA1 spots detected |
| `z_autofocus_rna_quality_score` / `_min` | (auto only) the decision score + threshold |

With `autofocus_channel: dapi` (the default) **none** of these columns are
emitted and the CSV is byte-identical to older runs. Currently wired for the
`rna_rna` / `rna_protein` modes (which is what the MIAT x QKI coloc presets use).

### 2. RNA1 over-detection QC flag (bulletproofing)

Independently of the autofocus choice, every run now **flags** — never drops —
images whose RNA1 spot count is implausibly high, so you can exclude them by
hand. This is the safety net for the out-of-focus-collapse symptom above.

`per_image_summary.csv` gains:

- `qc_rna1_spots_per_nucleus` — the RNA1 spots-per-nucleus for the image.
- `qc_overdetect_rna1` (bool) — fires when that exceeds an **absolute cap**
  (`qc.qc_overdetect_rna1_max_per_nucleus`, default **300**; a "few hundred per
  nucleus" is already far above any real single-molecule count here).
- `qc_overdetect_rna1_run_outlier` (bool) — fires when the image is a **robust
  outlier** vs the run median (`> median + k*1.4826*MAD`,
  `qc.qc_overdetect_robust_mad_k` default 5) **and** above the small-signal
  floor (`qc.qc_overdetect_min_per_nucleus_for_outlier`, default 50).

Either trigger adds `overdetect_rna1` / `overdetect_rna1_outlier` to `qc_flags`
and sets `qc_pass = False`. **Detection is never altered** — these are advisory.
Set the relevant threshold to `0` to disable a trigger.

```yaml
qc:
  qc_overdetect_rna1_max_per_nucleus: 300.0   # absolute cap; 0 disables
  qc_overdetect_robust_mad_k: 5.0             # run-level outlier k; 0 disables
  qc_overdetect_min_per_nucleus_for_outlier: 50.0
```

### 3. QKI foci detection specificity — the recommended pattern

For a **textured antibody stain** like QKI, the LoG `threshold_multiplier` alone
does **not** buy you specificity: loosening it to detect real foci also lights up
the textured cytoplasmic/nucleoplasmic background, and tightening it to suppress
background also kills real foci. The multiplier controls *relative* contrast, not
an *absolute* brightness floor, so background texture rides along with signal.

The pattern that **does** work is a **loose LoG detector plus an absolute
peak-intensity floor calibrated on the clean secondary-only control**:

```yaml
foci:
  antibody_overrides:            # (or rna2_overrides, whichever slot QKI is in)
    threshold_multiplier: 1.0    # LOOSE LoG — catch candidate foci
    only_nuclear_spots: true
    min_sep_px: 2
    min_spot_peak_intensity: 5000.0   # ABSOLUTE floor, sec-only-calibrated
```

Calibration: measure the **maximum spot peak intensity in the secondary-only
well** (no primary antibody → any "spots" there are non-specific). Set
`min_spot_peak_intensity` just above it. Example from the MIAT x QKI presets: the
sec-only max peak was ~4974, so a floor of **5000** drives sec-only spot counts to
~0 while keeping true QKI foci in the real wells. Keep `detect_in_sec_only: true`
so the sec-only spot rate is quantified and the floor stays honest.

Because the floor is an absolute intensity gate (not a relative multiplier), it
rejects textured background of any shape while preserving genuinely bright foci —
which the multiplier cannot do on its own.

---

## MIAT × QKI association metrics (continuous, floor-robust; 2026-07-07)

These are **new per-run analysis columns** (not a post-run utility): they are
written *during* the pipeline run, in `rna_protein` / `rna_rna` mode, when the
config flag `foci.compute_footprint_enrichment: true` is set. Default **OFF** →
byte-identical to older runs. They implement the approved spec
`_SPEC_association_analysis_2026-07-06.md`, replacing the old **binary**
"QKI-associated MIAT spots per nucleus" count (which conflated MIAT *abundance*
with per-molecule *association propensity*) with **continuous, at-the-punctum,
floor-robust** measures. No CLIP terminology — this is *spatial association*, the
microscopy analog of a pull-down-normalized-to-input.

### The footprint definition (the crux)

For each MIAT (rna1) spot, the QKI (rna2/protein) intensity is sampled over the
spot's **exact half-max (FWHM) footprint**, so it **scales with the spot's real
size** (a 0.3 µm punctum covers fewer pixels than a 1 µm focus) — **not** a fixed
disk:

1. A small per-spot window is taken around the spot centroid, sized from the
   spot's **measured** FWHM (`spot_diameter_um`) so a large focus is not clipped.
2. The **local peak** is the brightest MIAT pixel within ±1 px of the centroid;
   the **local background** is the 10th-percentile of the window.
3. The **half-max threshold** is `background + 0.5 × (peak − background)` (a true
   FWHM cut on the background-subtracted profile).
4. The footprint is the **connected component** (8-connectivity) of MIAT pixels
   ≥ that threshold **containing the seed** — so a neighbouring spot's separate
   blob is excluded.
5. QKI is averaged over **exactly those footprint pixels** (RAW, never floored).

If the half-max footprint is impractical (edge-clipped window, flat/no-contrast
crop, or the seed falls below threshold) the spot falls back to a per-spot
**fitted-radius disk** (radius = FWHM/2) — still size-scaling.

### New columns

**Per spot** (`spot_metrics.csv`, MIAT/`rna1` rows; `rna2` rows carry NaN):

| Column | Meaning |
|---|---|
| `qki_at_miat_footprint`   | RAW mean QKI over the spot's exact footprint pixels |
| `miat_footprint_area_px`  | Footprint size in pixels (scales with spot size) |
| `qki_footprint_enrichment`| `qki_at_miat_footprint ÷` the spot's-nucleus mean QKI — **floor-robust, PRIMARY per-spot metric** |

**Per nucleus** (`nuclei_metrics.csv`):

| Column | Meaning |
|---|---|
| `qki_assoc_ratio_continuous`        | Mean of `qki_footprint_enrichment` over the nucleus's MIAT spots (avg QKI fold-enrichment at MIAT; "input" = all MIAT spots → pull-down-over-input in spirit). **PRIMARY.** |
| `qki_assoc_ratio_gated_<floor>`     | Same, but MIAT spots whose footprint QKI < `foci.assoc_qki_floor` contribute 0. **SECONDARY** — reintroduces floor-sensitivity, shown for contrast. Emitted only when a floor is set (`<floor>` = the floor value in the name). |
| `coloc_moc`                         | Manders' Overlap Coefficient **R** = Σ(MIAT·QKI)/√(ΣMIAT²·ΣQKI²) on nuclear pixels — **threshold-free** (aliases the existing `coloc_cosine_overlap_rna1_rna2`). |
| `coloc_icq`                         | Li's ICQ on nuclear pixels — **threshold-free** (aliases `coloc_li_icq_rna1_rna2`). |
| `qki_at_miat_foci_enrichment`       | Mean QKI over the **union** of all this nucleus's MIAT-footprint pixels ÷ nuclear-mean QKI. |
| `qki_held_by_miat`                  | **CAPACITY (extensive).** Σ over the nucleus's **nuclear** MIAT spots of `qki_at_miat_footprint × miat_footprint_area_px` — the **total (integrated) QKI intensity inside MIAT-spot footprint pixels** (mean × area). |
| `miat_mass_nuclear`                 | **CAPACITY (extensive).** Σ over the same spots of `spot_peak_intensity × spot_area_px` — the **integrated MIAT signal** (peak intensity × nominal spot area, `π(FWHM/2)²`). |
| `capacity_qki_at_miat`              | **CAPACITY (the sponge test).** `qki_held_by_miat ÷` the **total nuclear QKI intensity** (`nuclear_total_intensity_rna2` → `nuclear_total_intensity_protein`) — the **fraction of the nucleus's QKI captured by MIAT footprints**. NaN when the denominator ≤ 0. |

The three **CAPACITY** columns are the **extensive / mass-based sponge test** — they
ask *how much* QKI this nucleus's MIAT actually holds, and what fraction of the
nuclear QKI pool that is, over the nucleus's **nuclear** MIAT spots (so numerator and
denominator share the same compartment). They complement the *intensive* per-spot
`qki_footprint_enrichment` / `qki_assoc_ratio_continuous` (fold-enrichment, not amount).

These sit **alongside** (never replace) the existing MAD-thresholded Manders
`manders_rna1_in_rna2` / `manders_rna2_in_rna1`, so a **gated-vs-ungated**
comparison is possible (the point of the spec). The rotation "proper background"
null remains the compartment-controlled backbone.

**Per image** (`per_image_summary.csv`): nucleus-mean rollups
`mean_qki_assoc_ratio_continuous`, `mean_coloc_moc`, `mean_coloc_icq`,
`mean_qki_at_miat_foci_enrichment`, `mean_qki_held_by_miat`,
`mean_miat_mass_nuclear`, `mean_capacity_qki_at_miat` (+
`mean_qki_assoc_ratio_gated_<floor>` when a floor is set).

### Enabling it

```yaml
foci:
  compute_partner_intensity: true      # (already on in the MIAT/QKI presets)
  compute_footprint_enrichment: true   # turn ON the association metrics
  assoc_qki_floor: null                # optional QKI floor for the SECONDARY gated ratio (null = off)
```

Runs the per-spot footprint loop **in-process** (like `compute_partner_intensity`)
— use `-p 1`. Single-plane only for QKI quantification (z-stacks have confirmed
16-bit QKI saturation).
