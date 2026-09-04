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
