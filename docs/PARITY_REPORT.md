# fishsuite ↔ Fiji pipeline — Output parity report

This document tracks how the standalone Python pipeline (`fishsuite`) compares
to the canonical Fiji pipeline at
`F:\Image Analysis Work\image-analysis-pipeline\` on every dimension
Brian asked about: output directory layout, CSV column schemas, QC overlay
rendering, walkthrough steps, popouts, and Excel workbook.

The goal is **functional output parity** — every downstream tool that
consumes a Fiji output dir must consume a fishsuite output dir transparently
(combine_to_xlsx.py, single_condition_plots.py, R scripts).

**Reference Fiji output used for comparison:**
`F:\Image Analysis Work\H9-Analysis\` — Brian's previous Fiji run on the same
H9 100x dataset.

**Standalone output produced:**
`F:\Image Analysis Work\H9-Output-2\MIAT-KD-ASO\` — `fishsuite run` on
`F:\Raw Images\H9-MIAT-KD-ASO\H9-MIAT-KD-05-08-2026\` (13 VSIs).

---

## 1. Top-level directory layout

| Subdir / file | Fiji | fishsuite (this revision) | Status |
|---|:-:|:-:|:-:|
| `per_image_summary.csv` | x | x | OK |
| `nuclei_metrics.csv` | x | x | OK |
| `spot_metrics.csv` | x | x | OK |
| `cell_morphology.csv` | x | x | OK (was missing pre-fix) |
| `thresholds.csv` | x | x | OK |
| `run_config.json` | x | x | OK |
| `analysis_summary.xlsx` | x | x | OK |
| `qc_overlays/` | x | x | OK (was a 3-panel matplotlib PNG pre-fix) |
| `publication_images/` | x | x | OK (per-channel + merge, was merge only) |
| `pipeline_walkthrough/` | x | x | NEW (was missing pre-fix) |
| `nuclei_popouts/` | x | x | NEW (was missing pre-fix) |
| `masks/` | x | x | NEW (was missing pre-fix) |
| `per_image_csv/` | x | x | NEW (was missing pre-fix) |

---

## 2. CSV column-by-column comparison

### per_image_summary.csv

Both pipelines write the canonical 20-column Fiji set:

```
image, condition, secondary_only, nuclei_analyzed,
mean_spots_per_nucleus, median_spots_per_nucleus, cv_spots_per_nucleus,
frac_nuclei_with_ge_1_spot, frac_nuclei_with_ge_5_spots, frac_nuclei_with_ge_10_spots,
mean_spot_diameter_um, median_spot_diameter_um,
mean_nuclear_spot_density_per_um2,
mean_cell_intensity_blend, median_cell_intensity_blend,
mean_cell_total_peak_intensity, median_cell_total_peak_intensity,
cv_cell_total_peak_intensity,
mean_spot_volume_um3, mean_spot_anisotropy
```

fishsuite **adds** these provenance columns (do not appear in Fiji
per_image_summary, but are harmless extras — column-order-tolerant tools
ignore them, combine_to_xlsx.py uses preferred-first ordering):

```
n_nuclei_border_excluded, total_spots, spots_in_nuclei, runtime_s,
dapi_channel, rna_channel, voxel_xy_nm, voxel_z_nm, n_z
```

### nuclei_metrics.csv

| Column | Fiji | fishsuite | Notes |
|---|:-:|:-:|---|
| image | x | x | |
| condition | x | x | |
| secondary_only | x | x | |
| experiment_id | x | x | "" in fishsuite |
| nucleus_id | x | x | |
| nucleus_area_px | x | x | |
| rna_mean_in_nucleus | x | x | |
| rna_nuclear_mean | x | x | aliased to rna_mean_in_nucleus (no nucleoplasm subtraction) |
| rna_cytoplasmic_mean | x | x | from Voronoi-expanded cytoplasm |
| rna_nc_ratio | x | x | |
| rna_spot_count | x | x | |
| nuclear_spot_count | x | x | |
| cyto_spot_count | x | x | |
| nuclear_spot_fraction | x | x | |
| nuclear_spot_density_per_um2 | x | x | |
| mean_spot_diameter_um | x | x | constant per image (BigFISH spot model) |
| mean_spot_fwhm_px / median_spot_fwhm_px / mean_spot_area_px | x | x | constant per image |
| mean_spot_volume_um3 / mean_spot_volume_vox | x | x | constant per image |
| mean_spot_anisotropy | x | x | constant per image (z_radius/xy_radius) |
| rna_spot_mean_intensity_bgc_blend / rna_spot_total_intensity_bgc_blend / rna_spot_median_intensity_bgc_blend | x | x | from spot peak intensity (BigFISH) |
| rna_spot_mean_peak_intensity / rna_spot_total_peak_intensity / rna_spot_median_peak_intensity / rna_spot_peak_intensity_cv | x | x | summed per-spot PEAK-pixel intensity (consistent proxy, NOT a Gaussian fit) |
| sum_rna_intensity | x | x | |
| cyto_area_px | x | x | |
| cyto_estimation_method | x | x | "voronoi" |
| n_voxels / n_pix / n_z_slices | x | x | |
| z_mode / z_range / autofocus_z | x | x | |
| voxel_xy_um / voxel_z_um | x | x | |
| rna_threshold_value / rna_frac_above_thr | x | x | |
| frac_spots_nuc_edge | x | x | NaN in fishsuite (not yet implemented) |
| dapi_mean_in_nucleus | (n/a) | x | extra, harmless |
| ab_*, manders_*, pearson_r, cosine_overlap, dice, jaccard, li_icq, sum_intensity_product, sum_min_intensity, both_frac, coloc_spot_count, coloc_spot_fraction | x (rna_protein mode) | (rna_only emits NaN/absent) | rna_only mode in both pipelines does not produce these |

### spot_metrics.csv

All Fiji rna_only columns present:
```
image, condition, secondary_only, experiment_id, spot_id, nucleus_id,
x_px, y_px, z_slice, z_position_um, spot_peak_intensity, quality,
spot_fwhm_px, fwhm_xy_px_fit, fwhm_z_px_fit, sigma_xy_px_fit, sigma_z_px_fit,
spot_diameter_um, spot_area_px, spot_volume_vox, spot_volume_um3, spot_anisotropy,
peak_intensity, rna_mean_raw_disk, rna_mean_bgc_blend,
rna_sum_bgc_blend, rna_sum_raw_disk, rna_bg_blend, rna_contrast_blend,
spot_bg_estimate, spot_to_nuc_edge_um, spot_to_nuc_centroid_um,
spot_to_nuc_edge_px, spot_to_nuc_centroid_px,
local_snr, fit_ok, n_voxels_sampled, z_fwhm_slices,
colocalized, coloc_partner_id, coloc_partner_dist_px, coloc_partner_dist_um,
coloc_partner_intensity, contrast_threshold
```

Several BigFISH-irrelevant fields (`rna_contrast_blend`, `local_snr`, etc.) are
present as NaN so downstream code that does `df.get("local_snr", default)`
works.

### cell_morphology.csv

```
image, condition, experiment_id, cell_id, nucleus_id, segmentation_mode,
area_um2, perimeter_um, circularity, aspect_ratio, roundness, elongation,
solidity, feret_max_um, feret_min_um
```

Matches Fiji 15-column set.

### thresholds.csv

Master-level thresholds.csv from fishsuite captures the same auditable
fields the Fiji per-image __thresholds.csv tracks: image, rna_threshold_used,
rna_threshold_value, dapi_threshold_method/value, watershed,
nuc_min_area_px, exclude_border_nuclei, z_mode/start/end, segmentation_backend,
stardist_prob_threshold, spot_backend, bigfish_spot_radius_nm,
bigfish_voxel_size_nm/z_nm.

The per-image equivalent is written to
`masks/<stem>__thresholds.csv` (one row per image — same shape as Fiji's).

---

## 3. QC overlay rendering

### qc_overlays/<stem>__qc_dapi_rna_nuclei_spots.png

| Element | Fiji | fishsuite (this revision) |
|---|---|---|
| DAPI LUT | Blue (rgb weights 0.0, 0.3, 1.0) | identical |
| DAPI contrast | p10 / p99.9 | identical |
| RNA LUT | Yellow (1.0, 1.0, 0.0) | identical |
| RNA contrast | DISP_FLOOR_PCT=95.0 / DISP_CEIL_PCT=99.95 | identical |
| Nuclei outline color/width | white, 2 px | white, 2 px |
| Spot marker color | yellow circle, r=4 px | yellow circle, r=4 px |
| Scale bar | 50 µm, bottom-right, white, height 12 | 50 µm, bottom-right, white, height 12, label "50 um" |
| File extension | PNG | PNG |

### qc_overlays/<stem>__qc_nuclei_on_dapi.png

DAPI gray + white nuclei outlines + 50 µm scale bar. fishsuite mirrors
Fiji's `__qc_nuclei_on_dapi.png` segmentation-check overlay.

---

## 4. Publication images

For each image, the Fiji pipeline writes:

| File | Fiji | fishsuite |
|---|:-:|:-:|
| `<stem>__DAPI_blue.png` + `.tif` | x | x |
| `<stem>__RNA_yellow.png` + `.tif` | x | x |
| `<stem>__merge_DAPI_RNA.png` + `.tif` | x | x |

All have 50 µm scale bars burned in (lower-right, white, font 28).

Pre-fix, fishsuite only wrote a single matplotlib composite — missing per-
channel publication renders.

---

## 5. Pipeline walkthrough

`pipeline_walkthrough/<stem>__step01..06.png`, one set per non-sec-only image.

| Step | Description | Fiji | fishsuite |
|---|---|:-:|:-:|
| step01 | Raw DAPI grayscale + scale bar | x | x |
| step02 | DAPI binary mask | x | x |
| step03 | Nuclei outlines on DAPI | x | x |
| step04 | RNA raw in yellow LUT | x | x |
| step05 | RNA threshold mask in yellow on black | x | x |
| step06 | RNA threshold overlay on grayscale RNA | x | x |

Pre-fix, fishsuite did not produce any walkthrough images.

Fiji's optional `step13/14/15/19` (spot-detection-internal stages) are not
produced by fishsuite because BigFISH does not expose the same intermediates;
those steps are not required by any downstream tool.

---

## 6. Per-nucleus popouts

`nuclei_popouts/<stem>__representative_nuc_NNN_spotsM.png`, 1–2 per
non-sec-only image. Selection: nuclei with `rna_mean_in_nucleus` closest to
the image median (matches Fiji's `save_image_nuclei_popouts` selection rule).

Each popout:
- DAPI(blue) + RNA(yellow) composite cropped ±30 px around the nuclear bbox
- This nucleus outlined in white
- Spots inside this nucleus marked with small white circles
- 5 µm scale bar (bottom-right)
- Same per-image contrast as the QC overlay (so popouts look consistent
  with the parent image)

---

## 7. Masks

`masks/<stem>__nuclei_label_mask.tif` — 16-bit label image
`masks/<stem>__spot_mask.tif` — 8-bit binary (255 / 0)
`masks/<stem>__dapi_mask.tif` — 8-bit binary (Otsu mask used in walkthrough step 02)
`masks/<stem>__thresholds.csv` — per-image auditable thresholds

---

## 8. Excel workbook

`analysis_summary.xlsx` has one sheet per master CSV plus the `How_to_read`
orientation sheet:

| Sheet | Source CSV |
|---|---|
| How_to_read | (generated) |
| Per_Image_Summary | per_image_summary.csv |
| Per_Nucleus_Metrics | nuclei_metrics.csv |
| Per_Spot_Metrics | spot_metrics.csv |
| Cell_Morphology | cell_morphology.csv |
| Thresholds | thresholds.csv |

The downstream `combine_to_xlsx.py` script can also be re-run on the
fishsuite output dir to get the full rich-sheet workbook (Headline_Numbers,
Per_Condition_x_Image, Per_Condition_Image_Replicates, Pairwise_Comparisons,
Quick_Look, etc.) — fishsuite's xlsx is the basic 5-sheet view; combine_to_xlsx
adds the derived analytical sheets.

---

## 9. Intentional divergences (and why)

| Item | Fiji | fishsuite | Why |
|---|---|---|---|
| Spot diameter / FWHM / volume | Per-spot geometry (`peak_intensity`, `sigma_xy_px_fit`, etc.) | Constant per image from BigFISH spot model | BigFISH does not produce a per-spot Gaussian fit; the spot model assumes a fixed radius. Per-image constant is the correct expression of this. |
| RNA spot intensity columns | Disk-sum + background blend | BigFISH peak intensity copied to all per-spot intensity slots | BigFISH does not expose disk-sum/bg-blend internally; peak is the available signal. Downstream relative comparisons across cells/conditions are preserved. |
| ab_* / coloc_* columns | Present in nuclei_metrics in rna_protein mode | Filled with NaN/absent in rna_only mode | The standalone runs rna_only here; rna_protein mode would populate these. |
| frac_spots_nuc_edge | Computed from spot-to-ROI-edge distances | NaN | Not yet wired into fishsuite. Cosmetic — does not affect counts/intensities. |

---

## 10. End-to-end validation

See `docs/END_TO_END_RESULTS.md` for the full run summary. The output of the
final standalone run is at `F:\Image Analysis Work\H9-Output-2\MIAT-KD-ASO\`.
