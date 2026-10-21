# `PanelRun.image_key` bug: fix and audit (2026-09-29)

**Verdict:** fixed upstream. Among current (non-superseded) outputs in the 9 audited folders, 0 cross-field pixel collisions were found. The only affected renders are the first Fig S1d Nog/g2 run, which had already been moved to `_superseded_imagekey_bug\`. No re-render is needed. Scope: human H9 imaging; no genome reference.

## Bug
- `src\fishsuite\figures\config.py` (main 5569454): the old code was
  ```python
  image_key(field_id) = reps[arm]["image"].replace("dox_15.vsi", f"dox_{field_id}.vsi")
  ```
- The substitution fires only when the arm's representative image is field 15, so it works only for VPR noDox (rep 15).
- For Nog noDox (rep 07) and g2 noDox (control) (rep 04), every field resolved to the representative image. The FOV, linked-set and native-TIFF steps (`fov.py`, `linked_set.py`) then rendered one image under several field labels, and nothing raised.
- The predecessor `fig1lib\fov.py` / `linked_set.py` (under `FIG1_IMAGING_v2_2026-09-28\scripts\`) has the same `.replace`. It only ever ran with `ARM = "VPR noDox"` (`v2style.py:20`), so it was never triggered.

## Fix (branch `fix/panelrun-image-key-2026-09-29`, worktree `E:\Claude\fishsuite-wt\fix-image-key-2026-09-29`)
- `image_key` now reads the menu's `data\menu_per_nucleus.csv` `image` column for rows where `condition == arm` and the well suffix equals the field. It also accepts an int field id with zero-padding.
- It raises `KeyError` unless exactly one image matches. It never falls back to the representative image.
- On the real basal menu, the new key equals the S1d wrapper's regex key for all 6 fields: VPR 14/15, Nog 07/08, g2 03/04.
- Regression test: `tests\test_panelrun_image_key_2026_09_29.py`, using a synthetic menu and h5, with no F: data. For each arm, two fields must give two keys and two different pixel hashes. An unknown field must raise.
  - On main 5569454 src: **3 failed, 1 passed** (Nog, g2 and unknown-field fail; VPR passes, which is correct since VPR worked before).
  - On the fix: **4 passed**. Together with `test_fig_panels_2026_09_28.py`, 18 passed, including the raw-VSI render test.
- Full suite: see "Suite" below.
- S1d workaround: `F:\Publication Work\05_FINALIZATION_2026-09-23\fig_drafts_2026-09-29\Fig1_supp\render_s1d_MIATmax1200.py` monkeypatches `PanelRun.image_key` with a trailing-field regex. Its keys are identical to the fix's, so it is now redundant but harmless. Remove it at the next edit; that file was not touched here.
- Env caveat: `fishproc_dml` imports an installed fishsuite from `F:\Image Analysis Work\MIAT_QKI_Coloc_2026_08_25\CLOSEOUT_2026-09-15\code\fishsuite-main\src`, not `E:\Claude\fishsuite`. Callers must set `PYTHONPATH=E:\Claude\fishsuite\src`, as the S1d `command.log` does.

## Audit method
- Script: `docs\IMAGE_KEY_BUG_2026-09-29_audit.py`. Full per-tile table: `docs\IMAGE_KEY_BUG_2026-09-29_tile_hashes.csv.gz`.
- **Hashed unit:** sha256 over (shape, dtype, bytes) of each decoded raster. This covers every raster embedded in every `.svg` (so both plain and `_native` twins), every `.tif` array, and standalone `.png` files that have no svg twin. Rasters under 32 px or with fewer than 16 distinct values are skipped (colourbars, blanks).
- **Labels:** the field and nucleus are parsed from the file name. Basal g2 fields 03/04 are canonicalised to `g2-noDox_03/04`, because they are the same raw files as OE `g2-noDox_03/04`. Their DAPI tiles do in fact hash-match the gallery `FOV_atlas\OE` tiles.
- **Flag:** one pixel hash shared by two or more different field labels. Current and `_superseded*` outputs are tested separately.
- **Enumeration:** a bounded `os.scandir` walk (depth ≤5) of each listed output folder, with no content grep. Every file read succeeded (0 read errors).
- **Positive control:** the known-bad `_superseded_imagekey_bug\` set must flag. My first pass had a field regex that missed `field08_…`, so 0 was flagged and the control failed. I fixed the regex, re-ran labelling on the same hash table, and the control now flags.
- **Limitation:** a nucleus crop cut from the wrong image at another field's coordinates is not byte-equal to anything, so hashing catches the bug at FOV level only. Crops are covered by the code-path check below.

## Audit table (hashes from the 2026-09-29 run; tile counts are from that run's log)
| Folder | Tiles hashed | Labelled | Superseded tiles | Cross-field collisions (current) |
|---|---|---|---|---|
| `F:\Image Analysis Work\MIAT_QKI_BASAL_Fig1_2026-09-24\FIG1_IMAGING_v2_2026-09-28` | 284 | 284 | 66 | 0 |
| `F:\Publication Work\PRESENTATION_MIAT_KD_OE_2026-09-26\CONFOCAL_GALLERY_v2_2026-09-28` (incl. `_20260929`, linked_ortho*, representatives*, FOV_atlas*, FOV_ortho*) | 2710 | 2639 | 0 | 0 |
| `...\PRESENTATION_MIAT_KD_OE_2026-09-26\LINKED_ORTHO_2026-09-28` | 464 | 464 | 276 | 0 |
| `F:\Publication Work\05_FINALIZATION_2026-09-23\G_figure_rebuild_2026-09-28\confocal_selection_2026-09-29` | 3168 | 2902 | 298 | 0 |
| `...\G_figure_rebuild_2026-09-28\confocal_panels_draft3` | 222 | 170 | 0 | 0 |
| `...\G_figure_rebuild_2026-09-28\confocal_panels_draft3_20260929` | 220 | 170 | 0 | 0 |
| `F:\Publication Work\05_FINALIZATION_2026-09-23\fig_drafts_2026-09-29\Fig1` | 33 | 0 (composites only) | 0 | 0 |
| `...\fig_drafts_2026-09-29\Fig1_supp` | 431 | 344 | 118 | 0 |
| `...\fig_drafts_2026-09-29\Fig4` (Fig4_supp lives inside `Fig4\`; there is no top-level `Fig4_supp`) | 302 | 78 | 48 | 0 |

- **Superseded set: 21 hashes flagged across 45 files, all in `Fig1_supp\S1d_renders_MIATmax1200\_superseded_imagekey_bug\`.** These are the Nog field 07 = 08 and g2 field 03 = 04 A_FOV DAPI/MIAT/QKI/merge/merge_MIATxQKI tiles (plain and `_native`), plus the B1 FOV-ortho merge tiles. This is the known first run and the positive control.
  - Figures using them: none. No current composite embeds a raster that exists only in the superseded set.
  - Current `FigS1d_p3.svg` embeds distinct UD_07 and UD_08 rasters, and `FigS1d_p4.svg` embeds distinct g2-noDox_03 and g2-noDox_04 rasters.
- **Same field, different nucleus label, identical pixels:** 29 hashes. All are whole-field sizes (1152², 2040², 2304²), for example B1 FOV-ortho XY shared between the nuc14 and nuc27 panels of field 14. This is the same field legitimately reused, not the bug.
- **Code-path check:** every other builder in these folders overrides `image_key` with its own explicit field-to-image map, so none uses the buggy base method:
  - `LINKED_ORTHO_2026-09-28\scripts\build_linked_ortho.py` `DatasetRun`, which the gallery `30_picks_ortho*` scripts reuse.
  - `confocal_selection_2026-09-29\scripts\s1_score_export.py` `BasalRun`.
  - `FIG1_IMAGING_v2` runs the VPR arm only.

## Affected outputs / re-render
- Affected and current: **none**.
- Affected but superseded, and already replaced: `F:\Publication Work\05_FINALIZATION_2026-09-23\fig_drafts_2026-09-29\Fig1_supp\S1d_renders_MIATmax1200\_superseded_imagekey_bug\{Nog,g2}\`.
- Re-render: not required. If wanted with the fixed code alone (fishproc_dml, `PYTHONPATH=E:\Claude\fishsuite\src`, after removing the wrapper lambda), re-run the existing commands from that folder's `command.log`:
  - `python render_s1d_MIATmax1200.py "Nog noDox" Nog select,fov,mix 07`
  - `python render_s1d_MIATmax1200.py "g2 noDox (control)" g2 select,fov,mix 03`

## Suite
- Full suite on 64c00be (fishproc_dml, `PYTHONPATH` = worktree src, 2026-09-29): **10 failed, 1707 passed, 4 skipped** (62 min).
- The 10 failures are the known missing-evidence baseline: 4 in `test_report_coloc_existing`, 1 in `test_report_localization`, 5 in `test_report_slides`.
- Re-running those three files against main 5569454 src gives the same 10 test IDs failing (10 failed, 52 passed). There are no new failures.
- The lock doc splits them as 6 + 4 by cause. The IDs are identical, so the difference is only in how they were grouped.
