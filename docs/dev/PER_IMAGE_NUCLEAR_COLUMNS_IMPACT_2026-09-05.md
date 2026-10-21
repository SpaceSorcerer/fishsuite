# Per-image nuclear columns: defect and consumer trace (2026-09-05)

## The defect

`rna_rna.py` rebinds `nuclei_df` to the **sampled** subset, keeping the full set
in `nuclei_df_all`:

```
src/fishsuite/core/modes/rna_rna.py:3855   nuclei_df_all = nuclei_df
src/fishsuite/core/modes/rna_rna.py:3856   if _restrict_rollups and "sampled_in_analysis" in nuclei_df.columns:
src/fishsuite/core/modes/rna_rna.py:3857       nuclei_df = nuclei_df[nuclei_df["sampled_in_analysis"].astype(bool)]...
```

The per-image nuclear counters then summed the **sampled** frame while dividing
by the **whole-image** spot total:

```
src/fishsuite/core/modes/rna_rna.py:3977-3981 (pre-fix)
    nuclear_spots_1 = ... nuclei_df.get("nuclear_spot_count") ... .sum()
    cyto_spots_1    = ... nuclei_df.get("cyto_spot_count")    ... .sum()
    frac_nuclear_image_1 = nuclear_spots_1 / float(total_spots1)
```

`total_spots1` comes from the spot table (`_in_cell_count(spots1_df)`, line
3848), which is image-wide. So `nuclear_spots_rna1` and
`cytoplasmic_spots_rna1` under-count by the sampling ratio,
`nuclear + cytoplasmic != total`, and `frac_nuclear_rna1` is low by the same
factor.

**Only bites when fixed-N sampling is enabled with `apply_to_rollups`.** With
sampling off, `_restrict_rollups` is False and `nuclei_df is nuclei_df_all`, so
the columns are already correct.

**Fix:** the numerator reads `nuclei_df_all`. These columns sit beside
`total_spots_rna1` and are image-level by definition; the sample-restricted
rollups are the `mean_*` / `median_*` columns, which still read `nuclei_df`.

**Scope.** `rna_rna.py` only, which `rna_protein` inherits through the
`rna2` -> `protein` relabel. `rna_only.py` has no such column: grep for
`nuclear_spots|frac_nuclear` in `src/fishsuite/core/modes/rna_only.py` returns
**0 hits**.

## Consumers in `E:\Claude\fishsuite`

Pattern: `nuclear_spots_(rna1|rna2|protein)`, `cytoplasmic_spots_*`,
`frac_nuclear_*`. Scanned **62** `.py` files under `src/` and **6** files under
`scripts/`.

| Consumer | Reads per-image columns? | Verdict |
|---|---|---|
| `src/fishsuite/report/` (`fishsuite report`) | **No.** Every hit is a per-NUCLEUS column: `endpoints.py:89,149,165,170` and `aggregate.py:600` use `nuclear_spot_count`, `nuclear_spot_count_rna2`, `nuclear_spot_fraction`, `nuclear_spot_fraction_rna2`; `peak_gate.py:209-219` likewise. **0 hits** for any `nuclear_spots_*` / `frac_nuclear_*` per-image name. | **NOT AFFECTED** |
| `src/fishsuite/core/excel_report.py` | **Yes.** `:78-79` lists them in the per-image sheet column order; `:241-252` glossary; `:1797-1798` `COMPARISON_METRICS` uses `frac_nuclear_rna1` / `frac_nuclear_rna2` as the per-image column for the mean/SEM display. | **AFFECTED** (workbook writes the columns verbatim and displays their means) |
| `src/fishsuite/core/excel_report.py:1699-1702` | Prefers the per-nucleus column and falls back to `frac_nuclear_rna1` only when `nuclear_spot_fraction` is missing. | Not affected in practice; the fallback is affected |
| `src/fishsuite/core/_vendor/analysis/single_condition_plots.py` | **Yes.** `:5865,5881` `top_col="frac_nuclear_rna1"` / `_rna2`; `:8029-8030` weights `frac_nuclear_rna1` by `total_spots_rna1`; `:8171` RNA1−RNA2 nuclear-fraction difference. | **AFFECTED** |
| `scripts/` | **0 hits** in 6 files. | **NOT AFFECTED** |

## Delivery verdicts

The decisive question per delivery is whether its source run had sampling on.
Read from the delivery's own `run_config.json`.

### `DELIVERY_RNASEH2B_BIN1intron_2026-09-04_v2` — **NOT AFFECTED**

- `00_SOURCE_FISHSUITE_RUN/run_config.json` -> `sampling {'enabled': False, ...}`;
  `preset_rnaseh2b_bin1_HARMONIZED_T36_PRODUCTION_2026-09-03.yaml` -> `sampling: enabled: false`.
- `nuclei_metrics.csv` carries **no** `sampled_in_analysis` column, so
  `_restrict_rollups` never fired.
- Numeric check on `00_SOURCE_FISHSUITE_RUN/per_image_summary.csv`:
  `nuclear + cytoplasmic != total` on **0 of 26** images (rna1 and protein), and
  `nuclear_spots_rna1` equals the sum over all nuclei on **0 of 26** mismatching.
- Only textual hit is `07_reviews_and_thresholds/THRESHOLD_PROPOSAL.md:329`,
  which merely *describes* a column present in `sweep_per_field.csv`. That sweep
  also ran unsampled.

### `DELIVERY_BIN1_ExIn_2026-09-05` — **NOT AFFECTED**

- `00_SOURCE_FISHSUITE_RUN/run_config.json` -> `sampling {'enabled': False, ...}`;
  `PRESET_USED_bin1_exin_HARMONIZED_T220_floors1000_1200_2026-09-03.yaml` has no
  `sampling:` block, so the default off applies.
- Numeric check: `nuclear + cytoplasmic != total` on **0 of 51** images for both
  `rna1` and `rna2`; `nuclear_spots_rna1` matches the all-nuclei sum on all 51.
- Grep over **75** text/code/csv files: the only hit is
  `data/per_image_summary.csv` itself, i.e. the column exists and is correct.
  **0 hits** in any README, READOUT, figure script or xlsx-producing script.

### `DELIVERY_MIAT_QKI_REVISED_2026-09-03` — **AFFECTED**

- `00_SOURCE_FISHSUITE_RUN/run_config.json` ->
  `sampling {'enabled': True, 'n_per_unit': 10, 'unit': 'per_image', ...}`.
  `nuclei_metrics.csv` carries `eligible_for_sampling`, `sampled_in_analysis`,
  `sampling_rank`.
- Numeric check on `00_SOURCE_FISHSUITE_RUN/per_image_summary.csv`:
  `nuclear_spots_rna1 + cytoplasmic_spots_rna1 != total_spots_rna1` on
  **42 of 44** images, and `nuclear_spots_rna1` disagrees with the sum over all
  nuclei on **42 of 44**:

  | image | sum over all nuclei | `nuclear_spots_rna1` |
  |---|---|---|
  | `MIAT_647_QKI_565__KD_1_17.vsi` | 594 | 157 |
  | `MIAT_647_QKI_565__KD_1_18.vsi` | 341 | 94 |
  | `MIAT_647_QKI_565__KD_1_19.vsi` | 805 | 174 |

  The `protein` channel shows 0 of 44 because it is not spot-detected in this
  preset.
- **Delivered figure consumes it:**
  `figures_v2_2026-09-03/make_figures_v2.py:187`

  ```python
  return {w: float(g.frac_nuclear_rna1.mean() * 100) for w, g in bio_img.groupby("condition")}
  ```

  This is `fig02_MIAT_percent_nuclear`. `figures_v2_2026-09-03/FIGURE_INDEX.md:19`
  reports its well means as 24.61 % -> 35.96 % and names both
  `nuclei_metrics.csv (nuclear_spot_fraction)` and
  `per_image_summary.csv (frac_nuclear_rna1)` as sources; the script uses the
  per-image one, so those two numbers are computed from the sampled-numerator
  column.
- `FIGURE_REVIEW_2026-09-03.md:43` refers to `frac_nuclear_rna1` as
  `per_image_summary.csv` column 33 while noting the endpoint is absent from the
  delivered inferential families.

**Note on direction.** `fig02` reports a *fraction*, and both its numerator and
denominator are wrong in the same direction but not by the same factor: the
numerator counts only sampled nuclei while the denominator is the whole image.
The reported percentages are therefore biased **low**, but the size of the bias
varies per image with the sampled-to-total nucleus ratio, so no single
correction factor applies and the figure needs regenerating rather than
rescaling. No corrected value is asserted here.

## Fix status

Branch `fix/per-image-nuclear-columns`, cut from `fix/only-nuclear-spots` (not
from `main`, and no commits added to `fix/only-nuclear-spots`, which a detached
chain fast-forwards). Regression test
`tests/test_per_image_nuclear_columns_2026_09_05.py` asserts, for both channels
and with sampling on and off, that the per-image counts equal the sum over
**all** nuclei, that `nuclear + cytoplasmic == total`, that `frac_nuclear`
matches its own numerator and denominator, and that enabling sampling does not
move the image-level counts. A guard test asserts the fixture really does sample
a strict subset, so the sampled cases cannot pass vacuously. The test fails
6 of 15 on the pre-fix engine and passes 15 of 15 on the fixed one.
