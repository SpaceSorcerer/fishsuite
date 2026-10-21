# On-figure filter line: defect and report survey (2026-09-05)

**Verdict: the defect is real in code, and no artifact on disk is affected.
Every `fishsuite report` ever built used `nucleus_filter: all`, for which the
hardcoded line happened to be true. The fix is preventive.**

## The defect

`src/fishsuite/report/figures.py:290` (pre-fix) built the line from nothing:

```python
def _filter_text(self) -> str:
    ...
    return (f"Filter: every segmented nucleus, no post-hoc nucleus filter, except "
            f"where an endpoint states the engine usability flag it applies"
            ...)
```

`FigureContext.__init__` never received `nucleus_filter`, so the sentence was a
constant. `aggregate.py:283-295` does apply the filter and records
`nucleus_filter`, `n_nuclei_all` and `n_nuclei_after_nucleus_filter`, and
`build.py:283` writes the value into the provenance table, but none of that
reached the figure. A report built with `--nucleus-filter sampled` would have
printed "every segmented nucleus, no post-hoc nucleus filter" on every figure
and in every footer while analysing a strict subset.

**Fix:** `FigureContext` takes `nucleus_filter`, `n_nuclei_all` and
`n_nuclei_after_nucleus_filter`; `build.py` passes them from the aggregate. For
`sampled` the line now reads:

```
Filter: post-hoc nucleus filter: the 10 nuclei per image the run sampled,
120 of 366 segmented, except where an endpoint states ...
```

`all` and `""` keep the historical wording. An unrecognised value is printed
verbatim rather than silently reported as `all`.

## Survey: every `report_*` directory under `F:\Image Analysis Work\`

Glob `**\report_*\`, `00_SOURCE_FISHSUITE_RUN` pruned, targets deduplicated by
`os.path.realpath`. **13 directories.** The filter value is read from each
report's own `command.log`, which records the full CLI invocation; absence of
`--nucleus-filter` means the CLI default `all`.

| `nucleus_filter` | evidence | png | svg | svg with the line | report dir |
|---|---|---|---|---|---|
| all (flag absent) | command.log | 23 | 23 | 23 | `BIN1_Introns_Exons_RNA-FISH_Analysisi\04_FULL\RUN_full51_harmonizedT220_ungated_20260903-061535\report_2026-09-04_report` |
| all (flag absent) | command.log | 6 | 6 | 6 | `MIAT_QKI_Coloc_2026_08_25\RAW_RUNS\MIAT_COUNT_2026-08-28_0020\report_2026-09-05_nuclearfraction_count` |
| all (flag absent) | command.log | 6 | 6 | 6 | `...\MIAT_COUNT_2026-08-28_0020\report_2026-09-05_nuclearfraction_count_allnuclei` |
| all (flag absent) | command.log | 35 | 35 | 35 | `MIAT_QKI_Coloc_2026_08_25\RAW_RUNS\MIAT_QKI_COLOC_2026-08-28_0214\report_2026-09-04_report` |
| all (flag absent) | command.log | 57 | 45 | 36 | `QKI_BIN1introns_2026_08_25\05_FULL_HARMONIZED_T36_diffuseQKI_2026-09-04\RUN_2026-09-04_0121\report_2026-09-04_report` |
| all (inferred) | run has no sampling | 36 | 36 | 36 | `QKI_BIN1introns_2026_08_25\05b_...\_superseded_fs-calib_arm3_report_2026-09-05\report_arm3_2026-09-05` |
| all (flag absent) | command.log | 66 | 54 | 45 | `QKI_BIN1introns_2026_08_25\05b_...\arm2_strictQKIspots_T36_2026-09-04_0546\report_2026-09-04_report` |
| all (inferred) | run has no sampling | 45 | 45 | 45 | `QKI_BIN1introns_2026_08_25\05b_...\arm2b_2026-09-04_2035\report_arm2b_2026-09-04` |
| all (flag absent) | command.log | 57 | 45 | 36 | `QKI_BIN1introns_2026_08_25\05b_...\arm3_pedestalNorm_2026-09-04_2213\report_arm3_2026-09-05` |
| all (flag absent) | command.log | 56 | 44 | 35 | `QKI_BIN1introns_2026_08_25\05b_...\arm4_pedestalClamp1_2026-09-05_0758\report_arm4_2026-09-05` |
| all (flag absent) | command.log | 45 | 45 | 45 | `RNASEH2B_BIN1introns_2026_08_25\13_FULL_HARMONIZED_2026-09-03\RUN_T36_2026-09-03_2350\report_2026-09-04_proof` |
| all (flag absent) | command.log | 45 | 45 | 45 | `RNASEH2B_BIN1introns_2026_08_25\13b_FULL_HARMONIZED_T36_FIXEDNUCLEAR_2026-09-05\RUN_T36_fixed_2026-09-05_0915\report_2026-09-05` |
| all (flag absent) | command.log | 0 | 0 | 0 | `RNASEH2B_BIN1introns_2026_08_25\DELIVERY_RNASEH2B_BIN1intron_2026-09-04\report_source` |

**Totals: 13 report dirs, `sampled` = 0, `all` = 13, 393 SVGs carry the line and
all 393 are correct.**

### The two reports without a `--nucleus-filter` record

`report_arm3_2026-09-05` and `report_arm2b_2026-09-04` were built by a custom
builder rather than the plain CLI, so no flag is recorded. `sampled` was
nevertheless **impossible** for them: `aggregate.py:284-289` raises
`ReportInputError` unless `nuclei_metrics.csv` carries `sampled_in_analysis`,
which the engine writes only under fixed-N sampling.

| source run | `sampled_in_analysis` column | `sampling.enabled` |
|---|---|---|
| `arm3_pedestalNorm_2026-09-04_2213` | absent | `False` |
| `arm2b_2026-09-04_2035` | absent | `False` |

A `sampled` report would have raised rather than produced figures.

## The three deliveries

| Delivery | SVGs | carrying the line | report invocations found | Verdict |
|---|---|---|---|---|
| `DELIVERY_RNASEH2B_BIN1intron_2026-09-04_v2` | 100 | 90 | `all (flag absent)` | **NOT AFFECTED** |
| `DELIVERY_BIN1_ExIn_2026-09-05` | 72 | 46 | `all (flag absent)` | **NOT AFFECTED** |
| `DELIVERY_MIAT_QKI_REVISED_2026-09-03` | 125 | 48 | `all (flag absent)` | **NOT AFFECTED** |

Every `command.log` under each delivery that invokes `report` lacks
`--nucleus-filter`, so all three ran the default `all` and their filter lines
state the truth.

This is a different question from the second defect.
`DELIVERY_MIAT_QKI_REVISED_2026-09-03` is affected by the per-image
nuclear-column bug because its fishsuite RUN used sampling
(`sampling.enabled: True, n_per_unit: 10`); its REPORT did not use
`--nucleus-filter sampled`. A run can sample while its report does not
re-filter, and that is what happened here.

## Why this was worth fixing anyway

The report layer already held the value and discarded it one function short of
the figure. That line is the only place a reader learns which nuclei a figure
covers, so the first `--nucleus-filter sampled` report would have shipped a
false provenance statement on every panel with nothing in the output to
contradict it. Same shape as the other two 2026-09-05 defects: a claim asserted
rather than derived.

## Tests

`tests/test_report_filter_line_2026_09_05.py`, 10 tests: one per filter value
(`all`, `""`, `sampled`, an unknown value); that `sampled` names the per-unit N
and the retained/segmented counts; that `sampled` still works when the config
carries no sampling block; `per_well` wording; footer consistency; and that the
default argument preserves the historical wording for callers passing nothing.
Fails 8 of 10 on the pre-fix engine, passes 10 of 10 after.
