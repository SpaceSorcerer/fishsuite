# fishsuite main sync + push — 2026-10-04

**Verdict: STOPPED at step 3 — `git merge --no-ff origin/main` conflicted in 5 files; merge aborted, nothing pushed, nothing committed.**

## State
| Item | Value |
|---|---|
| Branch | `main` |
| HEAD before / after | `b89a7176af914923667a4bff6e7deaf3c99e82a5` / unchanged (merge aborted) |
| Remote | `https://github.com/SpaceSorcerer/fishsuite.git` |
| origin/main after fetch | `e28ad11` (Merge pull request #1 from SpaceSorcerer/codex/reusable-imaging-closeout-20260915) |
| Merge base | `5c03381f0858a38cfea605df96b0480b4499a4ba` |
| Ahead / behind origin/main | 91 / 6 (dispatch said 89 / 6 on 2026-10-01; 2 local commits since) |
| Tracked modifications (step 1) | none |
| Untracked (left alone) | `.agent-handoff/`, `.codex/`, `AGENTS.md`, `CLAUDE.md`, `ORTHO_RENDERER_IMPLEMENTATION_REPORT_2026-09-20.md`, `_micrograph_zoom_builder_2026-06-20.py` |

## Incoming commits (main..origin/main)
| sha | subject | files |
|---|---|---|
| e28ad11 | Merge pull request #1 (codex/reusable-imaging-closeout-20260915) | merge |
| f9ee32a | ci: make light test matrix portable | `.github/workflows/test.yml`, `pyproject.toml`, `core/native_by_condition.py`, `report/cyto_calls.py`, tests: `test_condition_hierarchy`, `test_native_by_condition`, `test_report_coloc_existing`, `test_report_slides` |
| 016dd57 | postrun: reject ambiguous inputs and deduplicate footprint pixels | `postrun.py`, `tests/test_postrun.py` |
| 2b16039 | postrun: add generic metadata and footprint helpers | `postrun.py`, `tests/test_postrun.py` |
| c6fdcc7 | report: add reusable count and presentation options | `report/endpoints.py`, `report/figures.py`, `report/slides.py`, tests: `test_report_count_alias`, `test_report_rendering_options`, `test_report_size_provenance_wells` |
| fb771df | Preserve condition, well and FOV hierarchy throughout fishsuite | 14 files incl. `config/hierarchy.py`, `config/schema.py`, `core/native_by_condition.py`, `gui/main.py`, `gui/readiness.py`, `report/*`, `runner.py`, `tests/test_condition_hierarchy.py` |

## Conflicts (merge aborted with `git merge --abort`)
| File | Local side (main, newest first) | Remote side (origin/main) |
|---|---|---|
| `src/fishsuite/config/schema.py` | 741c3e8 Add UD MIAT-OE vs control preset and seg-sweep nuclear-size tuning; 12fd796 Merge feat/scale-aware-nuclei; 7653de0 Run guardrails | fb771df Preserve condition, well and FOV hierarchy |
| `src/fishsuite/gui/main.py` | 2e42076 conditions: one discovery call for runner, GUI preview and readiness; 741c3e8; 0f34473 Preserve condition, well and FOV hierarchy | fb771df |
| `src/fishsuite/gui/readiness.py` | 132a1a1 ortho + readiness: round-2 fixes (Codex Astra review); 2e42076; 741c3e8 | fb771df |
| `src/fishsuite/runner.py` | 2e42076; c80f511 Provenance fixes from 2026-09-17 three-lens review; 741c3e8 | fb771df |
| `tests/test_condition_hierarchy.py` (add/add) | 0f34473 | f9ee32a ci: make light test matrix portable; fb771df |

Auto-merged cleanly (not committed): `test.yml`, `pyproject.toml`, `core/native_by_condition.py`, `postrun.py`, `report/{cyto_calls,endpoints,figures,slides}.py`, 6 test files.

## Diagnosis (observed, not acted on)
- Local `0f34473` and remote `fb771df` have the **identical stable patch-id** (`1cabccac18d0`, both 2026-09-10, Brian): the same hierarchy commit landed on both sides under different shas.
- Local then rewrote those regions further (2e42076, 132a1a1, 741c3e8, c80f511); remote `f9ee32a` edited `test_condition_hierarchy.py` on top of its copy. The conflicts are local-evolution vs the shared base commit, plus f9ee32a's CI-portability edits.
- Resolving needs a per-hunk judgment (likely local side for the hierarchy code, then re-apply f9ee32a's portability edits). That is a source edit — out of scope for this dispatch.

## Tests / push
- Pre/post suite: **not run** — the merge never produced a post-merge tree. Env check: `fishproc_dml` imports `E:\Claude\fishsuite\src\fishsuite\__init__.py`; `pytest-xdist` is **not installed** there, so the suite runs serially (~74 min per run per `docs/CONSOLIDATION_2026-09-29.md`).
- Push: **not attempted**. origin/main (`e28ad11`) ≠ local HEAD (`b89a717`).
- Worktree `E:\Claude\_wt_fishsuite_premerge` was created at b89a717 and removed with `git worktree remove`; `git worktree list` no longer shows it.

## file_map.md
Not edited (no commit/push happened; editing it would dirty the tracked tree). Proposed entry:
`| E:\Claude\fishsuite\_sync_reports\PUSH_REPORT_2026-10-04.md | 2026-10-04 attempt to merge origin/main into main and push: stopped on 5 merge conflicts, with incoming commits, conflict sides and the duplicate-commit diagnosis. |`

---

# Round 2 — conflict resolution, tests, push (2026-10-04 → 2026-10-05)

**Verdict: merged and pushed. origin/main == HEAD == `3e17a79e992d4def293c57659801ca3d11ac5bec` (normal push, `e28ad11..3e17a79`). There are no new test failures relative to pre-merge.**

## Gate
- All 5 conflicts were mechanical. In each one the remote side is the fb771df text, and local already carried that same change as 0f34473 (identical stable patch-id), then revised it.
- Of f9ee32a's hunks, only one sits inside a conflict region.

## Resolutions
| File | Resolution | Local commits that superseded fb771df |
|---|---|---|
| `src/fishsuite/config/schema.py` | local | keeps `strict_filenames` (7653de0 / 741c3e8) |
| `src/fishsuite/gui/main.py` | local | `discover_from_conditions` (2e42076) |
| `src/fishsuite/gui/readiness.py` | local — **behaviour differs**: preflight runs for every preset, not only grouped | 2e42076, 132a1a1 (review C1) |
| `src/fishsuite/runner.py` | local — **behaviour differs**: when image failures exist, incomplete condition figures are printed instead of raised; adds `_sha256_file` and nuclear-size resolve | c80f511, 741c3e8 |
| `tests/test_condition_hierarchy.py` | local + f9ee32a hunk **adapted** (re-applied by hand at line 200: `pytest.importorskip('PySide6.QtWidgets')`) | — |

## f9ee32a hunk audit (staged diff vs HEAD compared line-for-line with `git show f9ee32a`)
| File | Status |
|---|---|
| `.github/workflows/test.yml` | applied, identical |
| `pyproject.toml` | applied, identical (`statsmodels` appears once) |
| `src/fishsuite/core/native_by_condition.py` | applied, identical |
| `src/fishsuite/report/cyto_calls.py` | applied, identical (one `import os`, one `_annotation_font` definition) |
| `tests/test_native_by_condition.py`, `tests/test_report_coloc_existing.py`, `tests/test_report_slides.py` | applied, identical |
| `tests/test_condition_hierarchy.py` | adapted (see above); result is identical to f9ee32a's hunk |

- `report/figures.py` diff vs HEAD is identical to c6fdcc7's.
- Re-introduction check: none of the other 10 files touched by fb771df differs from HEAD (0 changed lines each).

## Checks
| Check | Result |
|---|---|
| `compileall` on the 5 files, fishproc_dml Python 3.10.20 | exit 0 |
| Targeted: test_condition_hierarchy, test_conditions_discovery_roster_parity, test_gui_report_tab, test_native_by_condition, test_postrun, test_report_count_alias, test_report_rendering_options, test_report_size_provenance_wells, test_report_groups_and_contrasts | 109 passed in 88 s |
| Post-merge full suite, run 1 | **crashed** at ~76 %, exit 139 (Windows access violation in `scipy.sparse._compressed._getnnz`); no junit was written. Up to the crash there was 1 failure, the baseline id below. Crash site maps to `test_report_miat_qki.py::test_actual_figure_wells_and_extents`. That file alone gave 22 passed. No WHEA-Logger events in the System log between 2026-10-04 20:30 and 21:15. |
| Post-merge full suite, run 2 (serial, `-v`, junit) | **1 failed, 1791 passed, 9 skipped** in 1:21:59 |
| Pre-merge baseline (b89a717 worktree, `PYTHONPATH` = worktree src, confirmed by `fishsuite.__file__`) | **10 failed, 1759 passed, 4 skipped** in 1:52:11. These are the same 10 ids as `docs/CONSOLIDATION_2026-09-29.md`. |

- The only post-merge failure, `test_report_localization.py::test_recorded_reconciliation_baseline_parity_and_proposal`, is in the pre-merge failure set.
- The other 9 baseline failures now pass or skip. f9ee32a moved the recorded-fixture tests behind environment variables, and the junction test now uses `tmp_path`. That is a change in test design; nothing in fishsuite's own behaviour was fixed.
- Pre-merge worktree `E:\Claude\_wt_fishsuite_premerge`: created, then removed with `git worktree remove --force`. `--force` was needed because test artifacts were left in the tree; the removal exit code was 0.
- CI on origin/main (`gh run list`): the push run for e28ad11 succeeded on 2026-09-15, and the latest scheduled run succeeded on 2026-09-28. CI after this push was not checked.

## Push
- Merge commit `3e17a79` (parents b89a717 and e28ad11), with ahead/behind 92/0 before the push.
- `git push origin main` (no force) gave `e28ad11..3e17a79`.
- After `git fetch`, `git rev-parse origin/main` = `3e17a79e992d4def293c57659801ca3d11ac5bec` = HEAD.
- The `file_map.md` line was added in a separate follow-up commit.
