# fishsuite consolidation, 2026-09-29

Brian's request (2026-09-29): apply all genuine fixes, merge them, point imaging at the correct settings, and update the skills. Human H9 imaging; no genome reference involved. Nothing was pushed.

## Merged into main
| Branch | Commits | Merge | Why |
|---|---|---|---|
| `feat/fig-panels-twoarm-2026-09-28` | 65a3321, 46a433e (base 5569454) | `2b146d4` (clean, ort) | Genuine feature. It adds a two-condition renderer (`figures/twoarm.py`) and dataset kwargs for `linked_set.build_linked_set` (channel labels, MIAT/QKI minima, arm colour, B4 floor lines). The locked basal defaults are unchanged, and a test asserts this. |

Already on main before this pass: 160daeb (image_key docs), 64c00be (image_key fix), 5569454 (legend headroom), f48ff8c (lock doc), 37423f9 (footer merge), 9b390db (figure module).

## Not merged (listed for Brian; all predate 2026-09-26)
| Branch | Tip / date | Commits ahead | Note |
|---|---|---|---|
| `codex/fix-only-nuclear-spots` | da5b350, 2026-08-29 | 1 | only_nuclear_spots gate-ordering fix. Could be a real engine fix, but it is old. Needs review before it is merged. |
| `codex/reusable-imaging-closeout-20260915` | f9ee32a, 2026-09-15 | 5 | postrun/report helpers + CI portability. It also carries fb771df (condition/well/FOV hierarchy, 2026-09-10). Worktree on F: (CLOSEOUT `fishsuite-public`). |
| `feat/coupling-report` | 7b59cd2, 2026-09-17 | 1 | coupling: aggregates every numeric per-nucleus metric. |
| `miat-native-mask-reuse-20260915` | df8bc95, 2026-09-15 | 9 | Contains `miat-qki-code-of-record-2026-09-14` plus the reviewed-mask reuse. Worktree on F: (CLOSEOUT `fishsuite-native-outputs`). |
| `miat-qki-code-of-record-2026-09-14` | 51b4fa2, 2026-09-14 | 8 | MIAT×QKI three-slide code of record (Run B adapter, footprint_union_mass, notes_trailer). |

No other branch has unmerged commits (`git branch --no-merged main`). `stash@{0}` (pre-ff 2026-09-26) was left untouched.

## Worktrees (none deleted; candidates for cleanup, Brian's call)
- Fully merged, so safe to remove later: `fishsuite-wt/fig-panels-twoarm-2026-09-28`, `fig-panels-2026-09-28`, `fix-image-key-2026-09-29`, `fix-coloc-panel-footer`, `basal-*`, and the `feat/*`/`fix/*`/`integration/*` worktrees from 2026-09-17.
- The worktrees sitting at 5c03381 (old origin/main) are also merged.
- Holding unmerged work: `fishsuite-codex-only-nuclear-spots`, `fishsuite-code-of-record-2026-09-14`, and the F: CLOSEOUT worktrees `fishsuite-native-outputs` and `fishsuite-public`.

## Tests
- Targeted, on merged main: `test_fig_panels_twoarm_2026_09_28.py`, `test_panelrun_image_key_2026_09_29.py` and `test_fig_panels_2026_09_28.py` gave `23 passed in 46.26s`. The JPype/bioformats faulthandler dumps are the known noise.
- Full suite on `2b146d4`, run from `fishproc_dml` against the E: editable install: `10 failed, 1712 passed, 4 skipped, 282 warnings in 4418.02s (1:13:38)`.
- The 10 failures are the same IDs as the missing-evidence baseline in `IMAGE_KEY_BUG_2026-09-29.md`:
  - 4 in `test_report_coloc_existing` (import_exact_persisted_cells_and_anchors, missing_stale_and_frozen_guard, report_integration_no_nulls_raw_parity, line_group_conflict_and_cohort_rejected);
  - 1 in `test_report_localization` (recorded_reconciliation_baseline_parity_and_proposal);
  - 5 in `test_report_slides` (recorded_deck_semantic_assets_and_localization, frozen_writers_before_side_effect[save/localization/workbook/deck-True]).
- **No new failures.**

## Env repoint
- `fishproc_dml`: `pip install -e E:\Claude\fishsuite --no-deps` uninstalled the F: editable install and reinstalled from E:.
- `_editable_impl_fishsuite.pth` now contains `E:\Claude\fishsuite\src`, and no `.pth` or dist-info file mentions `fishsuite-main`. The F: code folder was not touched.
- `import fishsuite` → `E:\Claude\fishsuite\src\fishsuite\__init__.py`; `fishsuite.figures.twoarm` → `E:\...\figures\twoarm.py`.
- `fishsuite --help` lists the subcommands, including `fig-panels`, and `fishsuite fig-panels --help` shows the module options.
- `site-packages\fishsuite\config\presets\*.yaml` is package data listed in the new RECORD. It is not a pointer to F:.
- Other envs:
  - `fishproc` already imports `E:\Claude\fishsuite\src\fishsuite\__init__.py`, so it was not changed.
  - `codex_py`, `dml_test` and base miniconda have no fishsuite.
- Callers no longer need `PYTHONPATH=E:\Claude\fishsuite\src`. The S1d monkeypatch script (`render_s1d_MIATmax1200.py`) is now redundant. It was not edited.

## Skill
- File: `C:\Users\ambur\.claude\skills\fishsuite-rna-fish\SKILL.md`. Backup: `SKILL.md.bak_2026-09-29b`.
- Only the "Locked figure module" section was rewritten. It now records main `2b146d4`, the E: editable install, the basal `fig-panels` command, the two-condition library mode, and states there are no size options. It also records the image_key fix and the rules:
  - one circle per condition;
  - well means over a violin;
  - significance printed with its unit;
  - display ranges set per imaging set/slide;
  - the QKI/MIAT floor comes from the dataset's run and is drawn on B4;
  - near-centre representative nuclei, with the count as a separate text object;
  - MIAT yellow / QKI magenta, never green;
  - lossless native-resolution micrographs.
- Flag: Brian's wording for representative nuclei is "near-mean", but the basal `selection.py` uses the closest-to-MEDIAN robust distance. The skill records both. The code was not changed.

## Push decision (Brian)
- Local main `2b146d4` is 64 commits ahead of `origin/main` (5c03381, 2026-09-08). A push publishes all of them. Not pushed.
