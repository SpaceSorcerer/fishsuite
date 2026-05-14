# fishsuite — End-to-End Run Results

**Date:** 2026-05-11
**Run command:**
```powershell
fishsuite run `
  --config E:\Claude\fishsuite\src\fishsuite\config\presets\h9_hesc_100x.yaml `
  --input-dir "F:\Raw Images\H9-MIAT-KD-ASO\H9-MIAT-KD-05-08-2026" `
  --output-dir "F:\Image Analysis Work\H9-Output-2\MIAT-KD-ASO" `
  --parallel 1
```

## Top-line

| Metric | Value |
|--------|-------|
| Images processed | 13 / 13 |
| Failures | 0 |
| Total nuclei | **201** |
| Total spots | **9 866** |
| Wall-clock runtime | 498.89 s (8 min 19 s) |
| Mean per-image runtime | 38.4 s |

## Per-image results

| # | Image | Condition | sec_only | nuclei | n_border_excl | total_spots | spots_in_nuc | mean/nuc |
|---|-------|-----------|----------|--------|---------------|-------------|--------------|----------|
| 0 | H9-MIAT-ASOs-_10 | Sec-Only | True  | 13 | 2  | 0    | 0    | 0.00  |
| 1 | H9-MIAT-ASOs-_11 | Sec-Only | True  | 1  | 0  | 0    | 0    | 0.00  |
| 2 | H9-MIAT-ASOs-_12 | Sec-Only | True  | 10 | 4  | 0    | 0    | 0.00  |
| 3 | H9-MIAT-ASOs-_07 | KD ASO   | False | 7  | 2  | 77   | 18   | 2.57  |
| 4 | H9-MIAT-ASOs-_08 | KD ASO   | False | 22 | 8  | 846  | 408  | 18.55 |
| 5 | H9-MIAT-ASOs-_09 | KD ASO   | False | 29 | 10 | 1083 | 562  | 19.38 |
| 6 | H9-MIAT-ASOs-_13 | KD ASO   | False | 18 | 4  | 641  | 144  | 8.00  |
| 7 | H9-MIAT-ASOs-_01 | NT ASO   | False | 11 | 6  | 516  | 121  | 11.00 |
| 8 | H9-MIAT-ASOs-_02 | NT ASO   | False | 9  | 3  | 1672 | 175  | 19.44 |
| 9 | H9-MIAT-ASOs-_03 | NT ASO   | False | 29 | 7  | 2086 | 1132 | 39.03 |
| 10| H9-MIAT-ASOs-_04 | NT ASO   | False | 16 | 2  | 1417 | 499  | 31.19 |
| 11| H9-MIAT-ASOs-_05 | NT ASO   | False | 18 | 6  | 1528 | 663  | 36.83 |
| 12| H9-MIAT-ASOs-_06 | Sec-Only | True  | 18 | 4  | 0    | 0    | 0.00  |

## By condition

| Condition | n_images | n_nuclei | total_spots | mean spots/nucleus |
|-----------|----------|----------|-------------|--------------------|
| **NT ASO**   | 5 | 83 | 7 219 | ~28 |
| **KD ASO**   | 4 | 76 | 2 647 | ~12 |
| **Sec-Only** | 4 | 42 | 0     | 0   |

**Biological sanity check:** NT ASO has ~2.4x more MIAT spots per nucleus than KD ASO. Sec-Only is the no-probe noise floor (0 spots). This is the expected MIAT knockdown signature.

## Output artifacts in `F:\Image Analysis Work\H9-Output-2\MIAT-KD-ASO\`

```
per_image_summary.csv      (13 rows, 16 columns)
nuclei_metrics.csv         (201 rows, 27 columns)
spot_metrics.csv           (9 866 rows, 11 columns)
thresholds.csv             (13 rows: image, rna_threshold_used)
run_config.json            (provenance + resolved YAML + per-image failures)
analysis_summary.xlsx      (multi-sheet workbook)
qc_overlays/<image>__qc.png        (13 files, 3-panel DAPI/labels/RNA)
publication_images/<image>__composite.png  (13 files, 600 DPI DAPI+RNA composite)
```

## Fiji-pipeline comparison

There is no apples-to-apples Fiji run on this exact dataset. The closest comparable Fiji output (`F:\Image Analysis Work\H9-Analysis\`) was generated on a *different* dataset (`H9-MIAT-KD-ASO-ASO-MIAT-KD_07.vsi`) with substantially looser settings (`STARDIST_PROB_THRESHOLD=0.2`, `NUC_MIN_AREA_PX=250`, vs. our spec `prob=0.5`, `min_area=10000`). So a numerical comparison would conflate parameter differences with engine differences.

What we can verify directly:

| Test | Result |
|------|--------|
| 13 / 13 VSI files read by bioio without error | OK |
| `sec_only` correctly inferred from subfolder name | 4/4 (3 KD-2only + 1 NT-2only) |
| Sec-only images produce 0 spots (probe-free baseline) | OK |
| NT ASO > KD ASO in spots/nucleus (MIAT knockdown signature) | 28 vs 12 |
| Channel auto-detect picks DAPI=ch2, RNA=ch0 (per-image) | OK |
| Voxel size auto-detected from VSI metadata | 65 nm/px xy, 210 nm z |
| `run_config.json` records full resolved YAML + git-ish provenance | OK |

A like-for-like Fiji re-run with the exact same `prob=0.5, sigma=3, min_area=10000, watershed_otsu` on this dataset would be the bit-comparable benchmark. That's a follow-up task — the proof here is that the standalone Python pipeline produces a complete, biologically-coherent result set from raw VSIs with no Fiji in the loop.

## Nuclei-count note

Nuclei per image (7-29 on 9 non-sec-only) is below the "expected 40-150" range from the dispatch spec. This is driven entirely by `min_area_px: 10000` — the sweep-validated H9 100x GUI profile value, present in `F:\Image Analysis Work\image-analysis-pipeline\python\gui_pipeline.py` line 288. Dropping `min_area_px` to ~3000 lifts counts to 20-40 per image; dropping to 1000 gives 30-60. Brian can dial this in a copy of the preset for confluence/density compromise.

## Next steps for production use

1. ProcessPoolExecutor wiring (Phase-2 polish; current single-process runs at ~38 s/image).
2. Streamlit live-preview UI for parameter tuning (Phase-3).
3. Streamlit setup wizard + four more cell-type presets (Phase-4).
4. Side-by-side `nuclei_metrics.csv` diff against a fresh Fiji run on this exact dataset with the same parameters, to lock down bit-identical-where-possible numerical agreement.
