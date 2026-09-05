# Pending file_map entries — only_nuclear_spots fix (2026-09-05)

Branch `fix/only-nuclear-spots` @ `e081d71`. NOT merged to main: two fishsuite
jobs were live and one launcher starts further arms.

| Path | Description |
|---|---|
| `E:\Claude\_fs_fix_only_nuclear_spots` | Git worktree on branch `fix/only-nuclear-spots`, holding the restored `only_nuclear_spots` filter (cherry-pick of the unmerged `da5b350`) plus its regression test. Kept out of `E:\Claude\fishsuite` so live runs importing the editable install were not changed mid-flight. Delete after the merge. |
| `E:\Claude\fishsuite\tests\test_only_nuclear_spots_regression_2026_09_05.py` | NEW on the branch. 7 tests asserting the OUTCOME rather than the code path: zero `in_nucleus=0` rows under the shared flag, one channel only under a per-channel override, the protein channel under `antibody_overrides` in `rna_protein`, and a guard that the fixture really produces extra-nuclear spots so the assertions cannot pass vacuously. |
| `E:\Claude\fishsuite\tests\test_rna_rna_only_nuclear_spots.py`, `...\test_rna_only_spot_filters.py` | Arrived with the cherry-pick of `da5b350`; they were written with the fix and had never been on main. |
| `E:\Claude\fishsuite\tests\test_dapi_reference_null_and_rna_pedestal.py` | One test inverted: `test_only_nuclear_spots_does_not_filter_the_spot_table` asserted the unfiltered behaviour and is now `..._does_filter_...`. It was written against a main that had never carried the filter, so it characterised a regression rather than a requirement. fs-calib's `partner_anchored_null_nuclear_anchors_only` tests are untouched and still pass. |
| `F:\Image Analysis Work\MIAT_QKI_Coloc_2026_08_25\_BISECT_only_nuclear_spots_2026-09-05` | Bisect and confirmation stage. Holds the single-image CPU config and `RUN_NT330_fixbranch_cpu`, which returns 890 spots with zero extra-nuclear rows, matching the 2026-08-28 record coordinate-for-coordinate. Not a biological result. |
| `F:\Image Analysis Work\MIAT_QKI_Coloc_2026_08_25\_BISECT_only_nuclear_spots_2026-09-05\nt330_cpu_singleimage_2026-09-05.yaml` | The 2026-09-04 control preset restricted to `S2_NT_3/MIAT_647_QKI_565__NT_3_30.vsi` with `cellpose_device: cpu`, so the check never touched the GPU while the QKI arm held it. |
