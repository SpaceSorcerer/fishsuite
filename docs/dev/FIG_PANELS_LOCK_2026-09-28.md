# Locked basal figure-panel module: `fishsuite.figures` (2026-09-28)

Approved by Brian 2026-09-28 ("YES please, and merge and everything"). Human H9 imaging; no genome reference.

## What
- Source: `F:\Image Analysis Work\MIAT_QKI_BASAL_Fig1_2026-09-24\FIG1_IMAGING_v2_2026-09-28\scripts\` (`fig1lib\*.py`, `v2style.py`, drivers `04_select_r2.py`, `06_round2_build.py`, `07_mix_native_tiff.py`). The source was read only; nothing under F: was edited.
- Target: `src\fishsuite\figures\`.

| Module | Port of | Change |
|---|---|---|
| `style.py` | `v2style.py` | rcParams applied by `apply_style()`, not at import. Adds `check_luts()` (raises on a green LUT), `well_mean_style()` (the marker rule) and a `Saver` class instead of module-global paths. |
| `config.py` | path constants in `v2style.py` | `PanelRun(menu_dir, backfill_dir, run_dir, out_dir, arm, raw_dir, native_only)`. No absolute paths in `src/` (`scripts/check_no_absolute_paths.py` returns OK). |
| `stats.py` | `c_superplots.test/holm/run_tests` | Same arithmetic. `test` is renamed `wilcoxon_effect`, and the redundant second Holm pass is removed. |
| `c_superplots.py`, `d_scatters.py`, `rep_crops.py`, `fov.py` | same-named `fig1lib` modules | Take a `PanelRun` and a `Saver`. |
| `linked_set.py` | `fig1lib.linked_set` | The line rule is split into the pure functions `select_line` and `linked_geometry`, so the coordinates can be tested without the raw stack. |
| `selection.py` | `04_select_r2.py` | Unchanged apart from being turned into functions. |
| `gallery.py`, `__main__.py`, `cli.py` (`fig-panels`) | `06`/`07` drivers + `09` index | Step-selectable build. Writes `data\*.csv`, `stats.xlsx` and `index.html`. |

- Locked defaults:
  - Display ranges: MIAT 500–2250, QKI 1050–3746, DAPI 607–9000 raw a.u.
  - Colours: MIAT yellow, QKI magenta, DAPI blue, never green.
  - Axial factor 0.878; 0.13 µm/px.
  - Markers: one `'o'` marker and one colour per condition, across all wells.
  - Stats: C tests are two-sided Wilcoxon with nucleus as the unit, with per-well tests alongside. Holm runs across the six C tests; the effect size is the rank-biserial r.
  - D panels label pooled-puncta ρ/r as "pooled".
  - Every panel is written as 600 dpi PNG, SVG (text kept as text) and PDF, plus a `_native.svg` twin.
- Not ported:
  - Round-1-only panels: the raw-p C panels, D3/D6/D7/D8 from `02_quant.py`, and the `08`/`09` doc drivers (`STATS.md` prose).
  - The locked C family is the Holm version.
  - The round-1 field-15 B1 stem was `..._field15_box`; the module writes `..._field15_nuc11_box`.

## Commands
```
fishsuite fig-panels --menu "<...>\FIG1_IMAGING_MENU_2026-09-28" --backfill "<...>\EXACT_FOOTPRINT_BACKFILL_20260917-205946" \
  --run "<...>\RUN_PROD_OEvControl_PLAIN_jointAF_diam11um_LoG174_miat500_qki1050_20260917-203051" --out <NEW dir> \
  --linked "14:14:delivery Fig1A nucleus"          # optional: --arm, --raw-dir, --steps select,fov,linked,reps,c,d,mix,tiff,index, --native-only, --seed
python -m fishsuite.figures <same options>
```
Seeds: `np.random.seed`, `random.seed` and the jitter `default_rng` are all set from `--seed` (default 0).

## Tests (`tests\test_fig_panels_2026_09_28.py`, env `fishproc_dml`)
- Unit tests (no data needed):
  - `holm()`: hand-computed values, capping at 1 and monotonicity, and agreement with statsmodels.
  - Marker rule: checked by function, and on the rendered C panels (every well-mean marker has the same circle path; observed wells share one colour).
  - LUT guard: a green LUT raises.
- Reproduction tests (skip when the F: run is absent). Each compares exactly, with round-trip float parsing:
  - C1 stats against `data\stats.csv`: n, W, p, well means, well p and well n.
  - The full C Holm table against `stats_round2_C_holm.csv`.
  - The selections against `round2_selections.json`.
  - B-set geometry against `B_linked_params.json` for field 15 nucleus 11 (box, spot 166, angle 56, P0/P1, length, puncta on the line), and against the field-14 nucleus 27 and nucleus 14 JSONs.
- Sensitivity guard: dropping one nucleus changes C1 p, so the comparison is able to fail.
- `lab` + `bioformats` test: full B1–B4 render from the raw VSI. `z_window`, `stack_matches_h5`, P0/P1 and the B4 profile CSV all equal the saved values.
- Result, observed 2026-09-28: 14 passed (13 in 7 s, plus the render test in 52 s). The render test prints JVM "Windows fatal exception: access violation" faulthandler dumps and still passes. This is JPype/bioformats noise.
- End-to-end CLI rebuild to scratch: exit 0, 77 PNGs indexed.
  - `C_holm_effect`, `D_correlations`, `coloc_representatives`, the 3 B4 profile CSVs, the selections and the 24-row TIFF manifest are all exactly equal to the saved files. The B params match on every shared key; the round-1 JSON lacks `nucleus.field`.
  - 75 of the 76 PNGs that have a saved twin are pixel-identical. The field-15 B3 differs only in the title rows (y 42–111), because it uses the round-2 title placement (`va='center'`) rather than the round-1 placement.
- Full existing suite:
  - Branch: 10 failed, 1693 passed, 4 skipped (66 min).
  - The same 10 fail on unmodified main 4ea2410. Six tests in `test_report_coloc_existing` and `test_report_localization` fail on a missing `DELIVERY_RNASEH2B_BIN1intron_2026-09-05_v3\02_colocalization_panel` file. The four `test_report_slides` tests fail on a missing `_closeout_evidence\A\deck_spec.yaml`. None of these is a regression.
  - `ruff check` is clean on the new files; the repo has 21 existing findings in other files.

## Commits
- `9b390db`: the `fishsuite.figures` lock, on `feat/fig-panels-linked-ortho-superplot-2026-09-28` (worktree `E:\Claude\fishsuite-wt\fig-panels-2026-09-28`).
- main was fast-forwarded 4ea2410 → `9b390db`.
- `37423f9`: merge of `fix/coloc-panel-footer-2026-09-26` (9c3f33b) into main. The merge was clean (`merge-tree` gave no conflicts). The footer, figure, MDE and provenance tests pass on the merged tree (48 passed).
- Full suite on the merged main (37423f9): 10 failed, 1701 passed, 4 skipped (75 min). These are the same 10 missing-evidence failures as on 4ea2410, so there are no regressions.
- **Not pushed.** The dispatch pushes only if the full suite passes, and it does not: there are 10 failures, all of which predate this work. Local main is also 59 commits ahead of `origin/main` (5c03381, 2026-09-08). A push would publish all 59 of those commits, so that decision is Brian's.
- `stash@{0}` (pre-ff 2026-09-26) was left untouched. No branches or worktrees were deleted.
