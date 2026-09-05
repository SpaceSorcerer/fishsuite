# `fishsuite report` and condition groups (2026-09-04)

Two features, added together because they are the same idea at two levels.

1. **Condition groups** in the engine. A fishsuite `condition` has always been one
   WELL. A **group** is the condition several wells belong to, and is what gets
   compared. Configure `conditions.groups` and every master CSV gains a `group`
   column, and the run writes by-group SuperPlots alongside its native figures.
2. **`fishsuite report`**, a subcommand that builds the condition-versus-condition
   workbook, readout and figures for a finished run. It replaces the one-off
   `build_report_*.py` / `make_figures_*.py` scripts that were being copied per
   dataset.

## What changed, and why

| Complaint (Brian, 2026-09-04) | What now happens |
|---|---|
| Figures not in his style, wrong colours | Okabe-Ito only, never red plus green; 600-dpi PNG plus editable-text SVG; filter line, gate, n and test on every chart; one croppable footnote |
| Conditions not collapsed, each well is its own well within a condition | Wells sit INSIDE a group. Nuclei are shaded by well; well means are the tested diamonds; the heavy line is the mean of the well means |
| Raw significance hidden | The star is the **RAW** Welch p. The Holm-adjusted p and the minimum detectable effect are printed in the same footnote |
| Workbook unreadable, impossible to understand what Q1 means | Plain sheet names, and every sheet opens with a two-to-three sentence description row. No Q-code survives anywhere |
| Provenance unclear, which fishsuite run | Every figure footer, and the Read me and Run provenance sheets, name the producing run directory, the detection thresholds, the segmentation model and version, the replicate unit and the test |
| One-off script sprawl | The algorithms live in `src/fishsuite/report/`. Nothing is imported from a dataset folder |

## Condition groups

```yaml
conditions:
  mode: subfolders
  subfolder_conditions: {WT_1: WT_1, WT_2: WT_2, WT_3: WT_3,
                         KO_1: KO_1, KO_2: KO_2, KO_3: KO_3}
  sec_only_folders: ["Sec-Only"]
  groups:
    WT:     ["WT_1", "WT_2", "WT_3"]
    QKI-KO: ["KO_1", "KO_2", "KO_3"]
  group_order: ["WT", "QKI-KO"]
```

Rules:

* A well named in no group keeps its own label as its group, so it stays visible
  rather than being pooled silently. This is deliberate: the folder-to-condition
  table already has a failure mode where an unlisted folder is binned as junk.
* Secondary-only images are never in a biological group. They are labelled
  `Secondary-only`.
* With `groups` empty the column is not written and behaviour is unchanged.
* The FIRST entry of `group_order` is the default reference group.

The run's native figure step splits on `condition`, so with groups configured it
draws one panel per WELL. It is left alone. The runner additionally writes
`figures/by_group/` through the same code path `fishsuite report` uses, so a run
figure and a report figure cannot drift apart.

## `fishsuite report`

```
fishsuite report --run <run dir> \
    --groups WT=WT_1,WT_2,WT_3 --groups QKI-KO=KO_1,KO_2,KO_3 \
    --reference WT \
    --exclude-field <image name> --reason "why it was dropped" \
    --engine-repo <fishsuite checkout> --preset <preset yaml>
```

Omit `--groups` and the groups recorded in the run's own config are used.
`--exclude-field` and `--reason` are positional to each other: one reason per
excluded field, in the same order, and the pair is recorded in the workbook and
in every figure's filter line. `fishsuite report --help` documents the rest.

Output, under `<run>/report_<stamp>/` by default:

| File | Contents |
|---|---|
| `REPORT.xlsx` | Read me / Spots per nucleus by group / Nuclear fraction by group / Partner at puncta by group / Per well / Per field / Per nucleus / Contrasts / Secondary-only / Run provenance |
| `READOUT.md` | Ten plain lines, agnostic framing, first line the run path |
| `figures/` | One SuperPlot per endpoint, representative micrographs, `FIG_MAIN`, `FIGURE_INDEX.md`; PNG at 600 dpi plus editable-text SVG |
| `per_well.csv`, `contrasts.csv` | The tested points and every statistic |
| `versions.txt`, `command.log` | Library versions, seed, and the exact command |

The run directory is read, never written.

## Statistics

* **Replicate unit: the WELL.** Fields of view are technical replicates; nuclei
  are the measurement unit and are pseudoreplicates.
* **Gate:** Welch t on well means, with Hedges g and a 95 percent interval.
* **Multiplicity:** Holm-Bonferroni within an endpoint family, where a family is
  one by-group sheet. Descriptive-only, absolute-intensity and absent endpoints
  are excluded from the family and say why in `holm_exclusion_reason`.
* **Minimum detectable effect** is carried next to every p, in Hedges g units, so
  a null result can be read for what it is worth. It is solved by quadrature over
  the chi-square mixing variable, because `scipy.stats.nct` and
  `statsmodels.TTestIndPower` are both non-monotone in alpha at 4 degrees of
  freedom below about alpha 0.004, which is exactly where a Holm-adjusted family
  lands.
* **Sensitivity, never the gate:** exact permutation of well labels, whose
  two-sided p has an arithmetic floor of 0.10 at three versus three wells and
  reports that floor, and Tukey on field means, which treats a technical
  replicate as independent.
* **Stars come from the raw p.** The adjusted p is in the footnote next to it.
  This is a deliberate change from the builders this replaces, which starred the
  adjusted p and left the raw one only in the workbook.

## Where the algorithms came from

Ported in behaviour, not imported, from three one-off builders under
`F:\Image Analysis Work`:

* `BIN1_Introns_Exons_RNA-FISH_Analysisi\05_REPORT_2026-09-03\build_report.py` and
  `make_figures_v2.py` gave the Holm, Hedges, Welch, permutation and Tukey
  routines, and the SuperPlot layout.
* `RNASEH2B_BIN1introns_2026_08_25\14_REPORT_HARMONIZED_2026-09-03\build_report_rnaseh2b.py`
  and `make_figures_rnaseh2b.py` gave the nucleus to field to well hierarchy, the
  usability-flag filter, the minimum-detectable-effect solver, the exact-footprint
  per-nucleus aggregation, and the micrograph selection rule.
* `QKI_BIN1introns_2026_08_25\06_REPORT_2026-09-04\build_report_qki_bin1.py` gave
  the arm switch, generalised here into `--groups` plus `--reference`.

Two things were generalised rather than copied. Endpoints are declared in ROLE
terms (rna1 / rna2 / protein) and resolved against whatever a run emitted, so the
registry is not specific to one probe pair. Families are plain-language names
that map one-to-one onto the by-group sheets, replacing the Q1 to Q4 codes.

**Family membership therefore differs from the Q1 to Q4 grouping.** The raw Welch
p, the effect size and the interval are unchanged, because none of them depends
on the family. The Holm-adjusted p does depend on it, so an adjusted p from this
subcommand is not comparable with an adjusted p from the earlier builders.

## Reproduction check

Run `RUN_T36_2026-09-03_2350` was re-reported and compared against
`14_REPORT_HARMONIZED_2026-09-03\T36_2026-09-04_0127_rev2`. All 42 endpoints
agreed to better than 1e-9 relative on the well means and on the mean, the
difference, Hedges g, t, degrees of freedom, the Welch p, the permutation p and
the Tukey p.

The comparison is available as an opt-in test:

```
FISHSUITE_TEST_RUN_DIR=<run dir> FISHSUITE_TEST_REFERENCE_DIR=<rev2 dir> \
  pytest tests/test_report_reproduces_reference.py
```

It skips when the variables are unset, because both directories live outside the
repository.

## Standard colocalization panel

On an `rna_rna` or `rna_protein` run, `fishsuite report` calls
`scripts/coloc_standard_panel.py` and writes `coloc_standard_panel/` beside the
workbook, per the lab rule that every colocalization report carries the standard
panel alongside any null-based enrichment. Pass `--no-coloc-panel` to skip it.

## Punctum size: the moment estimator saturates, the footprint does not

Brian asked on 2026-09-04 whether every BIN1 punctum was being called the same
size, and whether `spot_fwhm_px` and `spot_diameter_um` were kernel constants
derived from `bigfish_spot_radius_nm`. They are not. Both are MEASURED per
punctum by `_measure_spot_diameter_um` (`src/fishsuite/core/modes/rna_rna.py`
lines 224 to 283, duplicated at `src/fishsuite/core/modes/rna_only.py` line 95):
a small crop is taken around the punctum centre, a tenth-percentile local
background is subtracted, negatives are clipped, and the second central moment of
the remaining signal gives a sigma that is converted to a full width at half
maximum. Reporting the configured BigFISH radius doubled was the OLD behaviour and
was fixed earlier; the comment recording that fix is at `rna_only.py` lines 481 to
487. On the RNASEH2B run the column holds 2,700 distinct values across 2,701
nuclear puncta, so it is not a constant.

The problem is different, and it is real. The crop is FIXED at `crop_half = 4`,
that is 9 by 9 pixels, so the second moment is truncated by the window. Measured
on isolated synthetic Gaussians of known width, the estimator tracks true width
below about 3 pixels and then saturates:

| True full width at half maximum (px) | Reported (px) |
|---|---|
| 1.88 | 1.88 |
| 2.83 | 2.82 |
| 4.71 | 4.03 |
| 7.06 | 4.47 |
| 9.42 | 4.60 |
| 23.55 | 4.74 |

A five-fold change in true width is reported as a 1.17-fold change. On the
RNASEH2B run the observed nuclear-punctum distribution has a median of 4.50
pixels, which is 95 percent of that asymptote, so essentially every punctum sits
where the estimator has lost its sensitivity. Above the linear range the value is
driven by neighbouring signal inside the crop rather than by the punctum, which is
why two conditions differ by under 1 percent on it. This is exactly the
observation Brian made.

What the report does about it:

* `rna1_punctum_footprint_area_um2` is the PRIMARY size endpoint. It comes from
  the engine's exact half-maximum footprint (`miat_footprint_area_px`, a legacy
  column name for the rna1 punctum's own footprint) times the run's voxel area.
  On the RNASEH2B run that column spans 12 to 110 pixels with a coefficient of
  variation of 0.45, against 0.053 for the moment estimator, so it has roughly
  eight times the relative dynamic range and does not saturate.
* `rna1_punctum_equivalent_diameter_um` restates the same footprint as the
  diameter of a circle of equal area, averaged per nucleus AFTER the per-punctum
  conversion. That is the mean punctum diameter, which is not the same number as
  the diameter of the mean area.
* `rna1_spot_fwhm_px` and `rna1_spot_diameter_um` are kept but marked descriptive
  only, so they carry no multiplicity-adjusted gate, and their workbook note
  states the saturation and its measured shape.
* A run that emitted no footprint column reports both footprint endpoints as not
  available rather than substituting the saturating one.

**Open, not implemented here:** a genuine per-punctum size would come from fitting
a two-dimensional Gaussian per punctum at detection time, with a crop that scales
with the fitted width instead of a fixed 9 by 9, and with the point spread
function deconvolved so the reported size is the object and not the optics. That
belongs in the detection layer, not in the report layer, and it would change every
existing run's size column, so it is left as a flagged item rather than a change
made in passing.

## Source-run linkage

Every report folder answers "which fishsuite run produced this" three times over,
per the `deliverable-package` skill:

* `SOURCE_RUN.md` at the top of the folder, with the run path on its third line,
  the preset, SHA-256 of `run_config.json`, `versions.txt`, `thresholds.csv` and
  `command.log`, the segmentation model and cellpose version, the detection
  thresholds read back off the run, the group definitions, and the list of run
  subfolders and files a reader should open.
* `00_SOURCE_FISHSUITE_RUN`, a directory junction created with
  `cmd /c mklink /J`, written only when the report is OUTSIDE the run directory.
  Never `ln -s`, which deep-copies under Git Bash on Windows and would duplicate
  an entire run. If the target refuses a junction, a `.lnk` shortcut and a
  plain-text path file are written instead and the reason is recorded.
* `provenance/source_run/` holding copies, never moves, of the run's own
  `run_config.json`, `versions.txt`, `thresholds.csv` and `command.log`.

Line one of `READOUT.md` is the run path, and every figure footer names the run
directory.

## Staging a run's groups: report_groups.yaml

A `report_groups.yaml` beside a run makes its report reproducible from one flag:

```yaml
groups:
  WT:     ["WT_1", "WT_2", "WT_3"]
  QKI-KO: ["KO_1", "KO_2", "KO_3"]
group_order: ["WT", "QKI-KO"]
reference: WT
well_from_image: '_((?:WT|KO|Mix|16|17)-\d)_'   # only when needed, see below
exclude_fields: {"WT_1_01.vsi": "out of focus"}
note: free text recorded with the report
```

```
fishsuite report --run <run> --groups-file <run>/report_groups.yaml
```

Command-line flags override any key the file sets, so a staged file is a default
rather than something that silently wins over what was just typed.

`well_from_image` exists for a specific failure. Some runs record the LINE as the
condition and carry the well only in the file name, for example
`..._WT-2_22.vsi`. Without the regex all nine fields of a line collapse into a
single well, leaving one replicate per group and no possible test. The expression
must have exactly one capture group and must match every biological image, or the
command refuses to run rather than proceeding with a partly recovered design.

`report_groups.yaml` files are staged beside these runs:

| Dataset | Run |
|---|---|
| RNASEH2B by BIN1 intron | `13_FULL_HARMONIZED_2026-09-03\RUN_T36_2026-09-03_2350` |
| BIN1 exons and introns, five lines | `04_FULL\RUN_full51_harmonizedT220_ungated_20260903-061535` |
| MIAT by QKI colocalization | `RAW_RUNS\MIAT_QKI_COLOC_2026-08-28_0214` |
| QKI by BIN1 intron, arm 1, diffuse | `05_FULL_HARMONIZED_T36_diffuseQKI_2026-09-04\RUN_2026-09-04_0121` |
| QKI by BIN1 intron, arm 2, strict spots | `05b_SENSITIVITY_AND_ARM2_2026-09-04\arm2_strictQKIspots_T36_2026-09-04_0546` |

Each file's `note` records the caveat that report must not be read without: the
WT-only readout on the QKI arms, the slide stratum on MIAT by QKI, the
batch-confounded Mix line on BIN1 exons and introns, and that arm 2 is a
sensitivity view on the same nuclei as arm 1 rather than independent evidence.

## One endpoint, two column spellings

The engine relabels the rna2 partner columns to `protein_*` in `rna_protein` mode
but leaves them `rna2_*` in `rna_rna`. An endpoint therefore carries
`alt_columns`, alternative spellings tried in order when its primary column is
absent, so one registry entry covers both modes instead of two near-duplicates
that could drift apart.


## Post-hoc peak floor, and two well statistics

`--peak-floor rna=1000,rna2=1200` applies a peak-intensity floor to a finished
run's spots and re-derives the per-nucleus counts. This reproduces a gated run
because fishsuite's own spot floor is applied AFTER detection. The boundary is at
or above the floor; a floor that is None, NaN or non-positive leaves its channel
untouched; and the nuclear-fraction denominator is in-nucleus plus in-cytoplasm,
not the row count, so a spot flagged as neither is excluded from every count.
Columns a floor makes stale and that cannot be re-derived from `spot_metrics.csv`
(above-floor intensities, Manders coefficients, the rotation-null enrichments) are
NAMED in the Read me sheet rather than silently carried at their pre-gate value.

`per_well.csv` reports two well statistics side by side because different
analyses have used each and they are different numbers:

* `well_mean_of_field_values` weights every FIELD equally. This is what the Welch
  gate runs on.
* `well_pooled_mean_of_nuclei` weights every NUCLEUS equally, so a field with more
  nuclei counts for more. This is what a fixed-N sampled design reports.

`--nucleus-filter sampled` restricts the report to the nuclei a run flagged
`sampled_in_analysis`, which is its fixed-N balanced set. The run must carry the
column or the command refuses.

## All-pairs contrasts

With more than two groups the report gives EVERY unordered pair by default, not
only each group against the reference; `--vs-reference` forces the two-group
behaviour. Tukey is fitted over the whole design and the pair of interest is read
out afterwards, so `tukey_k` records how many groups the adjustment covered. Both
a FIELD-level and a WELL-level Tukey are reported (`p_tukey_fov` and
`well_p_tukey_fov`): the field-level fit treats a technical replicate as
independent, so the two are not interchangeable.

## Reproduction against the reports this replaces

| Dataset | Reference | Result |
|---|---|---|
| RNASEH2B by BIN1 intron | `T36_2026-09-04_0127_rev2` | 42 endpoints, well means and 9 statistics each, to better than 1e-9 relative |
| BIN1 exons and introns | `BIN1_ExIn_report_20260903-104740.xlsx`, `per_line_summary` | 20 line-endpoint cells to 3.2e-16 relative |
| BIN1 exons and introns | the same workbook, `contrasts_all_pairs_wells` | 40 pairwise cells to 2.7e-13 relative, over difference, Hedges g, Tukey adjusted p, both interval bounds, k, degrees of freedom and the studentized range |
| MIAT by QKI | `DELIVERY_MIAT_QKI_REVISED_2026-09-03`, `biological_endpoint_values.csv` | 12 per-well values to 2.5e-16 relative on `well_pooled_mean_of_nuclei` over the sampled nucleus set |

The Holm-adjusted p is the one number that does NOT reproduce, by design: families
are the three by-group sheets rather than the earlier `Q1` to `Q4` grouping.

## Condition colours

The imaging WT-versus-QKI-KO pair is Brian's per-paper BIN1 palette, WT `#595959`
and QKI-KO `#D67AE5`. The hESC knockdown conditions take the shared condition map,
MIAT-KD `#E69F00` and MIAT-OE `#56B4E9`. Anything without a locked colour takes the
next unused Okabe-Ito entry, ordered so a two-group figure can never be red plus
green. `group_colors(order, overrides)` takes a group-name to hex mapping, so a
paper with its own palette needs no code change.
