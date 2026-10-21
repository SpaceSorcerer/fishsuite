# Thresholded RNA intensity in compartments

A third RNA-intensity measurement, beside spot intensities and raw pixel sums.
For each nucleus it integrates the RNA-channel intensity of all pixels at or
above a settable floor, separately in the nucleus and in the cytoplasm. It is
pixel thresholding, not spot detection, so it captures all above-floor signal,
diffuse and punctate, independent of the spot caller.

---

## How it compares

| Metric family | Column(s) | What it measures | Floor? |
|---|---|---|---|
| **Spot-based** | `rna_spot_total_peak_intensity`, `rna_spot_mean_intensity_bgc_blend` and others | Sum/mean of **called-spot** peak-pixel intensities (BigFISH/LoG detections only; a consistent intensity proxy, NOT a Gaussian-integrated value) | spot-detection floor, but only counts *detected spots* |
| **Raw pixel sum** | `sum_rna_intensity` (nucleus), `sum_rna_intensity_cyto` (cyto) | Sum of **every** pixel in the compartment, no threshold | **none**; includes background |
| **Thresholded** | `rna_thresh_*` (and `rna2_thresh_*`) | Sum/mean/area/fraction of **all pixels at or above the floor** in the compartment | **settable floor** (default = spot floor) |

It removes background like the spot metric, but does not require a pixel to
be part of a called punctum. Diffuse but real signal above the floor is counted.

### Relationship to the existing `*_above_floor_*` columns

The `nuclear_above_floor_intensity_rna1` family (gated by
`output.apply_pub_contrast_floor_to_analysis`) sums `clip(value - floor, 0, None)`,
a background-subtracted intensity. The `*_thresh_*` columns sum the raw
intensities of the pixels at or above the floor; the floor selects pixels and
is not subtracted. Both can be written at once.

---

## Columns (per nucleus, in `nuclei_metrics.csv`)

For each compartment (`nuclear`, `cyto`), and in two-channel modes for each
channel (`rna`, `rna2` / `protein`):

| Column | Definition |
|---|---|
| `rna_thresh_total_intensity_nuclear` | sum of RAW intensities of pixels with value at or above the floor, within the nucleus |
| `rna_thresh_mean_intensity_nuclear` | mean intensity of those pixels |
| `rna_thresh_pos_area_px_nuclear` | count of those pixels in the nucleus |
| `rna_thresh_pos_fraction_nuclear` | `rna_thresh_pos_area_px_nuclear` / (nucleus area in px) |
| `rna_thresh_total_intensity_cyto` | the same, within the Voronoi cytoplasm |
| `rna_thresh_mean_intensity_cyto` | |
| `rna_thresh_pos_area_px_cyto` | |
| `rna_thresh_pos_fraction_cyto` | `rna_thresh_pos_area_px_cyto` / (cytoplasm area in px) |
| `rna_thresh_floor` | the floor value actually used for this image (provenance) |

In `rna_rna` and `rna_protein` modes the second channel adds the parallel
`rna2_thresh_*` set (renamed to `protein_thresh_*` in `rna_protein` output by
the standard rna2toprotein relabeling), plus `rna2_thresh_floor`.

### Per-image aggregates (`per_image_summary.csv`)

Per-image **mean of the per-nucleus values** is reported for each of the above
(prefixed `mean_`), e.g. `mean_rna_thresh_total_intensity_nuclear`,
`mean_rna_thresh_pos_fraction_cyto`, and the `rna2_` equivalents in two-channel
modes. The floor used is recorded as `rna_thresh_floor` (and `rna2_thresh_floor`).

### Edge-case semantics

* **No floor resolvable** (setting unset and no spot floor available): value
  columns are `NaN`, `*_pos_area_px` stays `0` (never NaN, so the column stays
  integer), `*_pos_fraction` is `0.0` when the compartment has pixels.
* **No pixels above floor**: total `= 0.0`, mean `= NaN`, area `= 0`, fraction
  `= 0.0`.
* **Empty compartment** (e.g. no cytoplasm mask for a nucleus): all value
  columns `NaN`, area `0`, fraction `NaN`.
* **The floor is inclusive**: a pixel whose value exactly equals the floor is counted.

---

## Config: the settable floor

Two fields on `output` (`config/schema.py :: OutputCfg`):

```yaml
output:
  rna_intensity_threshold:  null    # channel 1 (RNA1) floor; null/0 => default
  rna2_intensity_threshold: null    # channel 2 (RNA2 / antibody) floor
```

**Default behavior when the field is `null` or `≤ 0`** (resolved per channel,
in this order):

1. The **resolved spot floor** the runner forwards via `analysis_floors`
   (the publication-contrast RNA floor: `manual_rna_min` when
   `output.apply_pub_contrast_floor_to_spots` is on, or the value the
   auto-batch or reference-image contrast pre-scan resolved). This is the same
   floor the spot filter and the display use.
2. Else `output.manual_rna_min`.
3. Else `NaN`: the columns are present but empty.

The rna2 channel uses `rna2_intensity_threshold` / `analysis_floors["rna2"]` /
`manual_rna2_min` (the antibody floor in `rna_protein`, mapped into the rna2
slot by the `rna_protein` wrapper).

Set the field to a positive number to **pin** the threshold explicitly,
independent of the spot floor.

> The `*_thresh_*` columns are always written, independent of the
> `apply_pub_contrast_floor_to_analysis` and `apply_pub_contrast_floor_to_spots`
> toggles. The spot-floor default works with both toggles off.

The floor is applied to the **same RNA image plane used for spot detection**
(`rna_2d` / `rna2_2d` after z handling).

---

## Modes and channels

| Mode | Channel 1 | Channel 2 |
|---|---|---|
| `rna_only` | `rna_thresh_*` | none |
| `rna_rna` | `rna_thresh_*` | `rna2_thresh_*` |
| `rna_protein` | `rna_thresh_*` | `protein_thresh_*` |

`rna_protein` routes through the `rna_rna` core with the antibody channel in the
rna2 slot, so the antibody channel gets the same threshold-and-integrate logic
and its columns are renamed from `rna2_*` to `protein_*` on output.

---

## Implementation map

| Piece | Location |
|---|---|
| Core helper | `src/fishsuite/core/metrics.py :: compute_thresholded_compartment_intensity(image_2d, mask, floor)` |
| Floor resolver | `src/fishsuite/core/modes/rna_only.py :: _resolve_thresh_intensity_floor(cfg, analysis_floors, channel)` (imported by `rna_rna`) |
| Per-nucleus wiring (1 ch) | `src/fishsuite/core/modes/rna_only.py` (per-nucleus loop + per-image means) |
| Per-nucleus wiring (2 ch) | `src/fishsuite/core/modes/rna_rna.py` (both channels; relabeled for `rna_protein`) |
| Config fields | `src/fishsuite/config/schema.py :: OutputCfg.rna_intensity_threshold / rna2_intensity_threshold` |
| Runner forwarding | `src/fishsuite/runner.py` (forwards `analysis_floors` for supported modes) |
| Tests | `tests/test_threshold_intensity.py` |

The master CSVs are written from the union of per-nucleus and per-image columns,
so these columns reach `nuclei_metrics.csv` and `per_image_summary.csv` without
extra wiring.
