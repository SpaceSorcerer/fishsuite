# Post-run utilities

Commands you run after `fishsuite run` has finished, to add tables and figures from the results you already have. They reuse the run's saved nucleus masks and spots, never re-segment or re-detect, and run on the CPU. They are safe to run as often as you like.

Point `--run` at the output folder a run produced (it holds `run_config.json`, `per_image_summary.csv`, `nuclei_metrics.csv` and `masks/`), not at the raw images. Source images are found from the `input_dir` the run recorded; pass `--staging` only if that fails. If something is wrong, the command prints a plain message and exits with code 2.

| Command | What it makes |
|---|---|
| `backfill` | Colocalization null draws, null summary, radial profile and an enrichment montage. `--rotation` adds the rotation null. |
| `walkthrough` | The 8-panel pipeline-walkthrough figure. |
| `postrun` | `backfill` then `walkthrough` in one command. |
| `sizefit` | Per-spot Gaussian size fit. |
| `singlecell` | Per-nucleus dose-response, matched-abundance and heterogeneity analysis. |
| `pixelpattern` | Per-nucleus pixel-pattern metrics and a stain-QC panel. |
| `if-pub-images` | Publication images for an `if_intensity` run. |
| `native-figures` | Group-level figures rebuilt from a run's CSVs. |
| `figure` | A plot of workbook values in the fishsuite figure style. |
| `ortho` | Orthogonal sections and line profiles of median-selected nuclei. |

New runs that turn on the matching `foci` options (`save_partner_null_draws`, `compute_partner_radial_profile`, `save_partner_rotation_null_draws`) already write these colocalization files. `backfill` adds them to runs made without those options.

---

## `backfill`

```bash
fishsuite backfill --run ./results/my_run
fishsuite backfill --run ./results/my_run --rotation
```

| Option | Default | Meaning |
|---|---|---|
| `--run` | required | Finished run folder. |
| `--staging` | auto | Folder holding the source images. |
| `--input` | auto | Alternate source folder (rarely needed). |
| `--seed` | 0 | Seed for the null and montage. |
| `--no-null-draws` | off | Skip `coloc_null_draws.csv` and `coloc_null_summary.csv`. |
| `--no-radial` | off | Skip `coloc_radial_profile.csv`. |
| `--no-montage` | off | Skip the montage PNG. |
| `--rotation` | off | Also compute the rotation null. |

Writes into the run folder:

| File | Contents |
|---|---|
| `coloc_null_draws.csv` | The pooled position-null draws. |
| `coloc_null_summary.csv` | Pooled enrichment, z and empirical p per image. |
| `coloc_radial_profile.csv` | Partner enrichment in rings around each RNA1 spot. |
| `figures/07_coloc/79_coloc_qki_montage_at_miat_vs_random.png` | Mean partner enrichment at RNA1 spots against matched random positions. The filename keeps legacy tokens (RNA1 = `miat`, partner = `qki`). |
| `coloc_rotation_null_summary.csv`, `coloc_rotation_null_draws.csv` | With `--rotation` only. |

`backfill` re-derives the pooled numbers and checks them against the values the run stored. It prints a validation table and exits 1 on a mismatch; inspect before trusting the output.

## `walkthrough`

```bash
fishsuite walkthrough --run ./results/my_run
fishsuite walkthrough --run ./results/my_run --image "cond_A__field_01" --out ./figures/walkthrough.png
```

| Option | Default | Meaning |
|---|---|---|
| `--run` | required | Finished run folder. |
| `--image` | auto | Image key (the per-image file prefix). |
| `--out` | `<run>/figures/07_coloc/79_pipeline_walkthrough.png` | Output PNG. |
| `--staging`, `--input` | auto | Source images for the re-rendered panel. |

A 2 by 4 grid at 600 dpi: (A) DAPI, (B) nucleus segmentation, (C) RNA FISH, (D) RNA spots, (E) protein IF, (F) thresholded protein, (G) RNA spots on thresholded protein, re-rendered at the analysed plane, (H) merge. Missing panels get a grey placeholder.

## `postrun`

```bash
fishsuite postrun --run ./results/my_run
```

Runs `backfill` (null draws, radial profile, montage; no rotation null) then `walkthrough`. It prints a line per step, continues past a failed step, lists every file produced, and exits non-zero if any step failed. Options: `--run`, `--staging`, `--input`, `--image`, `--seed`.

## `sizefit`

```bash
fishsuite sizefit --run ./results/my_run
```

| Option | Default | Meaning |
|---|---|---|
| `--run` | required | Run folder with `run_config.json`, `per_image_summary.csv` and `spot_metrics.csv`. |
| `--input-dir` | the run's `input_dir` | Source image tree. |
| `--window-px` | the run's `foci.size_fit_window_px` | Odd fit-window side in px. |
| `--limit-images` | all | Fit only the first N images. |

Outputs and columns are described in [docs/SPOT_SIZE_FIT.md](docs/SPOT_SIZE_FIT.md).

## `singlecell`

Reads only `nuclei_metrics.csv`; no images are opened.

```bash
fishsuite singlecell --run ./results/my_run
fishsuite singlecell --run ./results/my_run --group-a Control --group-b Treated --abundance-col nuclear_spot_count
```

| Option | Default | Meaning |
|---|---|---|
| `--run` | required | Run folder with `nuclei_metrics.csv`. |
| `--abundance-col` | `nuclear_spot_count` | Per-nucleus abundance axis. |
| `--group-a`, `--group-b` | auto | Control and perturbation labels. |
| `--include-secondary` | off | Keep secondary-only nuclei. |
| `--out-subdir` | `singlecell` | Folder under `<run>/deliverables/`. |
| `--seed` | 0 | Seed. |
| `--no-figures`, `--no-excel` | off | Skip figures or the workbook. |

Writes into `deliverables/singlecell/`:

| File | Contents |
|---|---|
| `single_cell_analysis.xlsx` | How_to_read, Dose_response, Dose_binned, Matched_abund, Heterogeneity, Distribution, Saturation. |
| `figures/*.png` | Control-versus-perturbation SuperPlots, dose-response scatters, a saturation composite. |
| `SINGLE_CELL_FINDINGS.md` | Plain-language readout. |

## `pixelpattern`

Reuses masks and re-reads the raw channel pixels at the analysed plane.

```bash
fishsuite pixelpattern --run ./results/my_run
fishsuite pixelpattern --run ./results/my_run --secondary-match well12
```

| Option | Default | Meaning |
|---|---|---|
| `--run` | required | Run folder with `masks/` and `nuclei_metrics.csv`. |
| `--staging`, `--input` | auto | Source images. |
| `--secondary-match` | all secondary-only | Substring choosing the clean secondary-only well for the stain-QC control. |
| `--out-subdir` | `pixelpattern` | Folder under `<run>/deliverables/`. |
| `--seed` | 0 | Seed. |
| `--no-figures`, `--no-excel` | off | Skip figures or the workbook. |

Writes into `deliverables/pixelpattern/`:

| File | Contents |
|---|---|
| `pixel_pattern_metrics.csv` | One row per nucleus: Gini, top 5% and 10% concentration, perinuclear index, foci-band counts. |
| `_radial_per_nucleus.csv`, `_sweep_per_nucleus.csv` | Edge-to-centre profiles and decile intensity sweeps. |
| `foci_band_summary.csv` | Partner spots per nucleus above intensity floors, with the secondary-only anchor. |
| `nt_vs_condition_wellmeans_welch.csv` | Control against perturbation, Welch on well means. |
| `pixel_pattern.xlsx` | Workbook, How_to_read first. |
| `figures/*.png` | SuperPlots, the stain-QC panel, radial and sweep plots. |
| `PIXEL_PATTERN_FINDINGS.md` | Plain-language readout including the stain QC. |

The `singlecell` saturation headline and the `pixelpattern` metrics are read alongside the rotation null, not instead of it.

## `if-pub-images`

Regenerates publication images for an `if_intensity` run from its `per_well.csv` and `per_fov.csv`. It re-reads only DAPI and each well's signal channel. Defaults for folders, floors, ceilings and the label come from `if_run_context.json`, written by the run.

```bash
fishsuite if-pub-images --run ./results/my_if_run
fishsuite if-pub-images --run ./results/my_if_run --floor 647:single_plane=5000 --dapi-ceiling 8000 --label MyProtein
```

| Option | Default | Meaning |
|---|---|---|
| `--run` | required | Finished `if_intensity` run folder. |
| `--staging` | recorded | Single-plane raw folder, one subfolder per well. |
| `--zstack` | recorded | Z-stack raw folder for the `picked_z` source. |
| `--source` | both | `single_plane` and/or `picked_z` (best-focus plane, no projection). Repeatable. |
| `--floor` | recorded | Signal floor as `SEC=VALUE` or `SEC:source=VALUE`. Repeatable. |
| `--ceiling` | none | Explicit signal ceiling, same syntax. Overrides `--ceiling-pct`. |
| `--ceiling-pct` | 99.5 | Ceiling percentile of the WT primary signal. |
| `--dapi-floor`, `--dapi-ceiling` | per image | Fixed DAPI display range across all panels. |
| `--per-image-ceiling` | off | Each panel's ceiling from its own percentile. Breaks cross-panel comparability. |
| `--scalebar` | 20 | Scale bar in um. |
| `--label` | recorded | Signal label in panels. |

The same display range is applied to WT, KO and secondary-only panels. Output goes to `<run>/publication_images/`, with the applied ranges logged in `qc_nuclear_dominance.txt` and channel routing in `channel_rule_log.txt`.

## `native-figures`

```bash
fishsuite native-figures --run ./results/my_run --groups groups.yaml --out ./results/my_run_by_group
```

Rebuilds group-level figures, summaries and workbook sheets from a run's CSVs into a new folder, using a groups YAML (a `conditions` block or a `groups` block; see [Declaring groups](README.md#declaring-groups)). Grouping only: no detection, segmentation, peak floors or nucleus filters.

## `figure`

```bash
fishsuite figure --from-xlsx values.xlsx --sheet Sheet1 --out ./figures/
```

Plots the values in a workbook sheet in the fishsuite figure style. No image analysis.

## `ortho`

```bash
fishsuite ortho --run-dir ./results/my_run --k 3 --metric nuclear_spot_count --out ./figures/ortho
```

Renders the `--k` nuclei nearest the median of `--metric` with XY, XZ and YZ sections and linked line profiles, using the run's saved labels and source stacks. It needs fixed manual display bounds for the RNA and antibody channels. Run `fishsuite ortho --help` for the display options (`--include-dapi`, `--z-crop`, `--axial-scale-factor` and others).

---

## Module entry points

The original modules still run directly and behave the same:

```
python -m fishsuite.core.coloc_backfill     --run-dir <run> --staging <staging>
python -m fishsuite.core.walkthrough_figure --run-dir <run> --staging <staging>
python -m fishsuite.core.singlecell         --run-dir <run>
python -m fishsuite.core.pixel_pattern      --run-dir <run> [--staging <raw>]
```
