# Fiji vs fishsuite audit — H9 MIAT-KD-ASO run (2026-05-12)

Compares the canonical Fiji pipeline (`F:\Image Analysis Work\image-analysis-pipeline\fiji_scripts\`) to fishsuite (`E:\Claude\fishsuite\`) on the same 12 .vsi files. Fiji output: `H9-Output-2\MIAT-KD-ASO-2\`. Fishsuite output: `H9-Output-2\MIAT-KD-ASO-claude-20260512_100104__tm_0_5__BATCH_FIXED\`.

## 1. TL;DR

- **Z-stack mode**: BOTH pipelines run "autofocus" = pick ONE best-focus z-slice (normalized Laplacian/Sobel variance over the DAPI substack) and do everything (segmentation, spot detection, pixel-coloc threshold) on that single 2D plane. Neither does 3D spot detection. `bigfish_voxel_z_nm: 210.0` in the run_config is metadata only; BigFISH is invoked with 2D `voxel_size=(xy, xy)` and a 2D image.
- **Z-range mismatch is the dominant divergence**: Fiji's autofocus searches **slices 5-15** (per Fiji's run_config); fishsuite's H9 preset searches **slices 20-30**. These are non-overlapping z windows in a 33-slice stack. Brian's DAPI plane of focus apparently lives in the lower half, so fishsuite's autofocus picks an out-of-focus DAPI plane → StarDist finds far fewer nuclei (8 shared images: Fiji 858 nuclei vs fishsuite 314, a 2.7x deficit).
- **Spot total is close** (Fiji 14,786 vs fishsuite 13,404 across the 8 shared images = 90.6% of Fiji), but fishsuite gets there with one-third the nuclei → `mean_spots_per_nucleus` is inflated 2–3x (e.g. _02: 17.4 → 48.9; _03: 13.3 → 42.0).
- **Spot-detection preprocessing differs**: Fiji applies `RNA_DETECT_ROLLINGBALL=40` + Gaussian `sigma=1.0` to the 2D RNA image BEFORE BigFISH (`Coloc_Analysis.py:3263-3269`). Fishsuite passes RAW `rna_2d` to BigFISH (`rna_only.py:174`, `spots.py:44-51`). This shifts the auto-Otsu LoG threshold.
- **Intensity gap**: Fiji `mean_cell_total_intensity_fit` runs 5–11x higher than fishsuite (e.g. _02: 461,055 vs 40,790). Fishsuite stores `intensity_peak = rna_2d[y,x]` (single pixel value at the centroid; `spots.py:97-110`), then aggregates `sum(ipeaks)` per nucleus. Fiji's "fit" intensity is the *integrated* Gaussian-fit volume (amplitude × 2π σ² + background), not a single-pixel peak. The "fit" column in fishsuite is a misnomer.

## 2. Z-stack handling

### Fishsuite behavior
- Config keys: `z_stack.mode`, `z_stack.start_slice`, `z_stack.end_slice` (`config/schema.py:46-50`).
- For mode=`autofocus`, fishsuite extracts the full ZYX stack for that channel, slices it to `zyx[zs0:ze0]`, then runs normalized Laplacian variance over the substack and returns the single sharpest 2D plane (`core/io.py:155-193`). Score = `Var(Laplacian(plane / mean))`.
- `rna_only.run_one` calls `extract_channel` for DAPI AND RNA with the SAME `z_mode/z_start/z_end` (`modes/rna_only.py:108-113`). The autofocus z is computed INDEPENDENTLY per channel — DAPI picks its own sharpest plane, RNA picks its own. (Fiji picks the DAPI-best plane and uses that same z for RNA — see Fiji section.)
- Both segmentation (line 149) and the pixel-coloc threshold (lines 248-255) operate on the 2D DAPI/RNA autofocus planes.
- BigFISH is called with `rna_2d` (2D ndarray) — `core/spots.py:91-110` detects `is_3d = rna.ndim == 3`; here rna is 2D, so detection is 2D. The `bigfish_voxel_z_nm=210.0` in the run_config is unused by BigFISH on this path (verified by reading `python/spots/detect_spots.py:148-167`).

### Fiji behavior
- `Z_MODE="autofocus"`, `Z_START=5`, `Z_END=15` per `MIAT-KD-ASO-2/run_config.json`.
- `Coloc_Pipeline.py:476-499`: extracts the DAPI substack via `extract_channel_3d`, calls `select_best_focus_slice(dapi_3d_tmp, z_offset=Z_START-1)` (defined `Coloc_Core.py:1172-1217`) — Sobel-edge stdDev² / mean, returns substack-relative 1-based index → converts to global z via `best_z = _substack_z + Z_START - 1`.
- That ONE global z-slice is then used to extract BOTH DAPI and RNA 2D planes via `Duplicator().run(imp, CH, CH, bz, bz, 1, 1)` (lines 491-498). Same z for both channels.
- Segmentation runs on that 2D DAPI plane (`segment_nuclei(dapi2d)` at line 553). RNA spot detection: `detect_spot_candidates` is called on the 2D RNA plane after rolling-ball+Gaussian preprocessing (`Coloc_Analysis.py:3262-3272`); BigFISH receives a 2D image.

### Divergence + recommendation
| Aspect | Fiji | Fishsuite |
|---|---|---|
| z-search range (this run) | slices 5-15 | slices 20-30 |
| autofocus metric | Sobel edges, var/mean | Laplacian, Var(Lap/mean) |
| DAPI vs RNA share the z? | yes (DAPI's best z used for RNA) | no (each channel picks its own best z) |
| Spot detection dimensionality | 2D | 2D |
| Voxel-z used by BigFISH | n/a (2D) | n/a (2D — `voxel_z_nm=210` is recorded but unused on the 2D code path) |

**Brian's stated intent ("autofocus → pick the BEST z slice per image for 2D segmentation, with spot detection in 3D")**: BOTH pipelines violate the "spot detection in 3D" half. Both pick one z and run BigFISH in 2D. If Brian wants true 3D spot detection while keeping 2D segmentation, that is currently NOT happening in either pipeline.

## 3. Pipeline divergence — 8 shared images

### Side-by-side per-image table
| Image | Fiji nuclei | FS nuclei | Δnuc % | Fiji spots | FS spots | Δspot % | Fiji mean/nuc | FS mean/nuc | Fiji tot_int | FS tot_int |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| _02 | 97 | 17 | -82% | 2263 | 2052 | -9% | 17.35 | 48.88 | 461,055 | 40,790 |
| _03 | 133 | 54 | -59% | 2191 | 2584 | +18% | 13.29 | 41.96 | 395,873 | 40,617 |
| _04 | 106 | 41 | -61% | 1829 | 2034 | +11% | 13.16 | 42.00 | 369,718 | 26,594 |
| _05 | 111 | 48 | -57% | 2420 | 2042 | -16% | 16.26 | 35.50 | 368,516 | 29,989 |
| _07 | 77 | 16 | -79% | 1666 | 310 | -81% | 14.52 | 10.44 | 328,211 | 39,407 |
| _08 | 100 | 59 | -41% | 1243 | 1113 | -10% | 8.50 | 14.90 | 181,477 | 13,327 |
| _09 | 104 | 55 | -47% | 1284 | 1749 | +36% | 8.21 | 27.73 | 168,981 | 21,431 |
| _13 | 130 | 24 | -82% | 1890 | 1520 | -20% | 10.45 | 30.96 | 223,819 | 25,654 |
| **Total** | **858** | **314** | **-63%** | **14,786** | **13,404** | **-9%** | — | — | — | — |

### Top causes ranked

1. **Z-range mismatch** (dominant driver of nuclei deficit).
   - Fiji: `Z_START=5, Z_END=15` (run_config.json).
   - Fishsuite: `start_slice=20, end_slice=30` (`presets/h9_hesc_100x.yaml` via `config_resolved.z_stack`).
   - Fishsuite scans the WRONG half of the 33-slice stack → autofocus picks an out-of-focus DAPI plane on most images → StarDist finds fewer/dimmer nuclei.
   - Evidence: _07 (worst case) has 77 → 16 nuclei AND 1666 → 310 spots (spot count tracks the bad RNA plane too).
   - Code: fishsuite `core/io.py:142-168` honors start/end exactly; Fiji `Coloc_Pipeline.py:485-486` uses `Z_START-1` offset identically. So the issue is the YAML value, not the code.

2. **Spot-detection preprocessing differs** (sub-dominant, but explains the ~10% spot delta and likely some of the spot-localization differences).
   - Fiji: rolling-ball radius=40 px + Gaussian sigma=1.0 applied to rna_2d BEFORE BigFISH (`Coloc_Analysis.py:3263-3269`). This flattens uneven illumination, raising auto-Otsu sensitivity in dim regions.
   - Fishsuite: passes raw `rna_2d` directly to BigFISH (`modes/rna_only.py:174-185`, then `core/spots.py:44-51`). No background subtraction, no smoothing.
   - Net effect: similar total spot counts on most images, but spatial distribution will differ; auto-Otsu threshold lands at a different LoG value.

3. **StarDist parameter differences** (small but real).
   - Fiji: `STARDIST_PROB_THRESHOLD=0.35`, `NUC_MIN_AREA_PX=8000` (run_config.json).
   - Fishsuite: `prob_threshold=0.3`, `min_area_px=10000` (resolved config).
   - Fishsuite is MORE permissive on probability (0.30 vs 0.35) but STRICTER on min area (10000 vs 8000). Net direction is image-dependent; this is a minor confound compared to z-range.
   - `stardist_gauss_sigma=3.0` is identical in both (Fiji launcher; fishsuite preset).
   - Watershed postprocess + mask_closing=5 are identical.

4. **"fit" intensity is not actually a Gaussian fit in fishsuite** (cosmetic-but-confusing; explains the order-of-magnitude `mean_cell_total_intensity_fit` gap).
   - Fishsuite: `rna_spot_total_intensity_fit = sum(intensity_peak)` where `intensity_peak = rna_2d[y_px, x_px]` — a single raw pixel value (`modes/rna_only.py:360-364`, populated from `spots.py:97-110`).
   - Fiji: integrated Gaussian-fit volume (amplitude × area, see `python/spots/detect_spots.py:175-274` — the `_fit_gaussian_2d`/`_fit_gaussian_3d` helpers) — units are pixel-intensity × pixel-count, ~10²-10³ larger than a single pixel peak.
   - This is a column-meaning mismatch, not a real biology gap. Fishsuite should either rename or actually run the Gaussian fit.

### File:line evidence summary
| Cause | Fishsuite | Fiji |
|---|---|---|
| Z-range | `presets/h9_hesc_100x.yaml` (start=20, end=30); `core/io.py:142-168` slicing | `_gui_launcher.py` (Z_START=5, Z_END=15); `Coloc_Pipeline.py:449-499` |
| Autofocus metric | `core/io.py:172-193` (Laplacian) | `Coloc_Core.py:1172-1217` (Sobel `Find Edges`) |
| RNA preprocessing before BigFISH | none — `modes/rna_only.py:163-189` passes raw rna_2d | `Coloc_Analysis.py:3263-3273` (rolling=40, sigma=1.0) |
| StarDist params | `presets/h9_hesc_100x.yaml`: prob=0.3, min_area=10000 | run_config: prob=0.35, min_area=8000 |
| "fit" intensity | `modes/rna_only.py:362-365` (sum of `intensity_peak` = raw pixel) | `python/spots/detect_spots.py:175-274` (actual Gaussian fit) |
| Voxel-z noise | `core/spots.py:44-71` (2D branch ignores voxel_z) | n/a |

## 4. Recommendation

**Verdict**: fishsuite has *significant* divergence from Fiji driven primarily by a config (z-range) mistake, plus three real algorithmic gaps. Spot totals agree to within 10% on the shared images, but nuclei counts disagree by ~63% so per-cell metrics are not comparable.

To close the gap (in priority order, no code yet — direction only):

1. **Fix the H9 preset z-range**. Change `z_stack.start_slice: 20 → 5` and `end_slice: 30 → 15` to match Fiji's run_config. This alone should recover ~2-3x more nuclei. Verify by re-running and checking nuclei totals against Fiji.
2. **Share the autofocus z across DAPI and RNA**. Either pass the DAPI-derived best z to RNA extraction (fishsuite currently lets each channel pick its own), or use Fiji's convention of DAPI-first → reuse for RNA. Same-z is what Fiji does and what the pixel-coloc semantics implicitly assume.
3. **Add the rolling-ball + sigma=1.0 RNA preprocessing before BigFISH** to match Fiji's auto-Otsu calibration on the same signal distribution. Expose as `foci.rna_detect_rollingball` and `foci.rna_detect_blur_sigma` in the schema (defaults: 40, 1.0).
4. **Align StarDist params**: bump `prob_threshold` 0.3 → 0.35 (Fiji); lower `min_area_px` 10000 → 8000. These are minor compared to (1) but free to land.
5. **Either rename `rna_spot_*_intensity_fit` columns or run the actual 2D Gaussian fit**. Right now downstream comparisons of cell intensity totals between the two pipelines silently differ by 10x. Easiest fix: drop the `_fit` suffix until a real fit is wired in.
   - **RESOLVED 2026-06-10**: the rename was done — `rna_spot_*_intensity_fit` → `rna_spot_*_peak_intensity`, per-spot `integrated_intensity_fit` → `peak_intensity`, per-image `*_cell_total_intensity_fit*` → `*_cell_total_peak_intensity*`; the meaningless `spot_fit_success_*` columns were removed. Honest names now reflect that these are summed per-spot PEAK-pixel intensities, NOT a Gaussian fit. (A real Gaussian fitter is still NOT wired in.) The geometry `*_px_fit` / `fit_ok` columns were intentionally left untouched.
6. **Optional, separate**: if Brian wants true 3D spot detection with 2D-autofocus segmentation, that's a new code path in BOTH pipelines — neither currently supports it.
