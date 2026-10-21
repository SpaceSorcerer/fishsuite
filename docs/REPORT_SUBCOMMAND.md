# `fishsuite report`

`fishsuite report` builds a condition-versus-condition workbook, a short readout and figures for a finished run. It reads the run and never writes into it. Plot styles, DAPI localization, deck export and the other optional features are in [REPORT_FEATURES.md](REPORT_FEATURES.md). Run `fishsuite report --help` for every option.

## Groups

A condition is one well. A group is the set of wells being compared. Declare groups in the run config (see [Declaring groups](../README.md#declaring-groups)), or at report time:

```bash
fishsuite report --run <run dir> \
    --groups WT=WT_1,WT_2,WT_3 --groups KO=KO_1,KO_2,KO_3 \
    --reference WT \
    --exclude-field <image name> --reason "why it was dropped"
```

Groups come from `--groups`, then `--groups-file`, then the run's own config. Without any, every well is its own group. The first group is the reference unless `--reference` says otherwise. Each `--exclude-field` needs its own `--reason`, in the same order; the pair is recorded in the workbook and in every figure's filter line.

### `report_groups.yaml`

A groups file beside a run makes its report reproducible with one flag. It takes the keys `groups`, `group_order`, `reference`, `well_from_image`, `exclude_fields` (image name to reason) and `note` (free text recorded with the report):

```bash
fishsuite report --run <run> --groups-file <run>/report_groups.yaml
```

Command-line flags override keys set in the file.

Use `well_from_image` when a run recorded the line as its condition and the well only in the filename (for example `..._WT-2_22.tif`). Without it every field of a line collapses into one well and no test is possible. The regex needs exactly one capture group and must match every biological image, or the command stops.

## Outputs

Under `<run>/report_<stamp>/` by default, or `--out`:

| File | Contents |
|---|---|
| `REPORT.xlsx` | Read me, Spots per nucleus by group, Nuclear fraction by group, Partner at puncta by group, Per well, Per field, Per nucleus, Contrasts, Secondary-only, Run provenance. Each sheet opens with a short description row. |
| `READOUT.md` | Ten plain lines; the first is the run path. |
| `figures/` | One plot per endpoint, representative micrographs, `FIG_MAIN` and `FIGURE_INDEX.md`. 600 dpi PNG plus editable-text SVG. |
| `per_well.csv`, `contrasts.csv` | The tested points and every statistic. |
| `SOURCE_RUN.md`, `provenance/source_run/` | Which run produced the report, with file hashes and copies of its provenance files. |
| `versions.txt`, `command.log` | Library versions, seed and the exact command. |
| `coloc_standard_panel/` | `rna_rna` and `rna_protein` runs only. Pearson, Manders with Costes thresholds, Li ICQ, a cytofluorogram, line profiles and an object-based called-coloc fraction. Skip with `--no-coloc-panel`. |

`--no-figures` writes the workbook and readout only.

## Statistics

- **Replicate unit: the well.** Fields of view are technical replicates; nuclei are the measurement unit.
- **Headline:** a nucleus-level linear mixed model (REML, fixed group, random well intercept, field-within-well variance component). Its two-sided Wald p is the headline when the fit succeeds. When the fit fails or is unavailable, the report falls back to Welch's t on well means and labels the row as a fallback. Figure brackets use the headline p.
- **Always reported:** Welch's t on well means with Hedges g and a 95% interval, Holm adjustment within an endpoint family (one by-group sheet), and the minimum detectable effect in Hedges g units at the number of wells you have. `--alpha` sets the level (default 0.05).
- **Sensitivities, never the headline:** exact permutation of well labels (its two-sided p cannot go below 0.10 at three wells against three), equal-variance t on well means, Welch on field means, Tukey on field means, and a field-outlier sensitivity. See [REPORT_FEATURES.md](REPORT_FEATURES.md#statistical-sensitivities).
- Descriptive-only, absolute-intensity and absent endpoints are left out of the Holm family; `holm_exclusion_reason` says why.

With more than two groups, every unordered pair is compared by default. `--vs-reference` compares each group only with the reference. Tukey is fitted over the whole design; `tukey_k` records how many groups it covered.

## Endpoints

Endpoints are declared by role (rna1, rna2, protein) and matched to whatever columns the run wrote. An endpoint whose column is absent is listed as not available, never dropped. In `rna_protein` runs partner columns are named `protein_*` and in `rna_rna` runs `rna2_*`; each endpoint lists both spellings.

Punctum size uses the exact half-maximum footprint (`rna1_punctum_footprint_area_um2` and `rna1_punctum_equivalent_diameter_um`). The moment-based `rna1_spot_fwhm_px` and `rna1_spot_diameter_um` are kept as descriptive only because they saturate; see [SPOT_SIZE_FIT.md](SPOT_SIZE_FIT.md). A run with no footprint column reports both footprint endpoints as not available.

## Post-hoc options

- `--peak-floor rna=1000,rna2=1200` applies a spot peak-intensity floor to a finished run and re-derives per-nucleus counts. A spot at or above the floor is kept. A floor that is empty or not positive leaves its channel untouched. Columns the floor makes stale and that cannot be recomputed from `spot_metrics.csv` are named in the Read me sheet.
- `--nucleus-filter sampled` restricts the report to nuclei flagged `sampled_in_analysis` by fixed-N sampling. The run must carry that column.
- `per_well.csv` reports two well statistics: `well_mean_of_field_values` (each field weighs equally; used for the Welch test) and `well_pooled_mean_of_nuclei` (each nucleus weighs equally).

## Source-run linkage

Every report folder names its run three ways:

- `SOURCE_RUN.md`: run path, preset, SHA-256 of `run_config.json`, `versions.txt`, `thresholds.csv` and `command.log`, segmentation model and version, detection thresholds, group definitions, and what to open in the run.
- `00_SOURCE_FISHSUITE_RUN`: a directory junction to the run, created only when the report is outside the run. If a junction cannot be made, a shortcut and a text file with the path are written instead.
- `provenance/source_run/`: copies of the run's own provenance files.

Line one of `READOUT.md` and every figure footer name the run.

## Colours

Groups take colours from the Okabe-Ito palette in order, so a two-group figure is never red with green. Set `conditions.group_colors` in the run config, or `group_colors` in `report_groups.yaml`, to choose your own.
