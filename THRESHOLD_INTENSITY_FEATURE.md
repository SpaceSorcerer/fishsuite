# Thresholded RNA intensity in compartments

**Added 2026-06-02 (Brian).** A THIRD RNA-intensity measurement for the
fishsuite RNA-FISH pipeline, alongside the two that already existed. It mirrors
a protein "threshold-and-integrate" approach: for each nucleus, integrate the
RNA-channel intensity of **all pixels whose raw intensity is at or above a
settable floor** — measured **separately within the nucleus and within the
cytoplasm**. This is pure **pixel-thresholding**, not spot detection, so it
captures all above-floor MIAT signal (diffuse + punctate) independent of the
spot-caller.

---

## Why a third intensity metric

| Metric family | Column(s) | What it measures | Floor? |
|---|---|---|---|
| **Spot-based** | `rna_spot_total_intensity_fit`, `rna_spot_mean_intensity_bgc_blend`, … | Sum/mean of **called-spot** peak intensities (BigFISH/LoG detections only) | spot-detection floor, but only counts *detected spots* |
| **Raw pixel sum** | `sum_rna_intensity` (nucleus), `sum_rna_intensity_cyto` (cyto) | Sum of **every** pixel in the compartment, no threshold | **none** — includes background |
| **Thresholded (NEW)** | `rna_thresh_*` (and `rna2_thresh_*`) | Sum/mean/area/fraction of **all pixels ≥ floor** in the compartment | **settable floor** (default = spot floor) |

The new metric fills the gap between the other two: it floors out background
like the spot metric, but — unlike spot detection — it does not require a pixel
to be part of a discrete called punctum. Diffuse-but-real signal above the
floor is counted.

### Relationship to the existing `*_above_floor_*` columns

The pre-existing `nuclear_above_floor_intensity_rna1` family (gated by
`output.apply_pub_contrast_floor_to_analysis`) sums `clip(value − floor, 0,
None)` — i.e. background-subtracted intensity. The **new** `*_thresh_*` columns
sum the **RAW** intensities of the ≥floor pixels (the floor only *selects which
pixels* contribute; it is not subtracted). They are deliberately distinct and
both can be emitted at once.

---

## New columns (per nucleus, in `nuclei_metrics.csv`)

For each compartment (`nuclear`, `cyto`), and — in two-channel modes — for each
channel (`rna`, `rna2` / `protein`):

| Column | Definition |
|---|---|
| `rna_thresh_total_intensity_nuclear` | Σ of RAW intensities of pixels with value ≥ floor, within the nucleus |
| `rna_thresh_mean_intensity_nuclear` | mean of those ≥floor pixel intensities |
| `rna_thresh_pos_area_px_nuclear` | count of ≥floor pixels in the nucleus |
| `rna_thresh_pos_fraction_nuclear` | `rna_thresh_pos_area_px_nuclear` / (nucleus area in px) |
| `rna_thresh_total_intensity_cyto` | …same, within the Voronoi cytoplasm |
| `rna_thresh_mean_intensity_cyto` | |
| `rna_thresh_pos_area_px_cyto` | |
| `rna_thresh_pos_fraction_cyto` | `…_cyto_area_px` / (cytoplasm area in px) |
| `rna_thresh_floor` | the floor value actually used for this image (provenance) |

In `rna_rna` and `rna_protein` modes the second channel adds the parallel
`rna2_thresh_*` set (renamed to `protein_thresh_*` in `rna_protein` output by
the standard rna2→protein relabeling), plus `rna2_thresh_floor`.

### Per-image aggregates (`per_image_summary.csv`)

Per-image **mean of the per-nucleus values** is reported for each of the above
(prefixed `mean_`), e.g. `mean_rna_thresh_total_intensity_nuclear`,
`mean_rna_thresh_pos_fraction_cyto`, and the `rna2_` equivalents in two-channel
modes. The floor used is recorded as `rna_thresh_floor` (and `rna2_thresh_floor`).

### Edge-case semantics

* **No floor resolvable** (knob unset *and* no spot floor available): value
  columns are `NaN`, `*_pos_area_px` stays `0` (never NaN — keeps the integer
  column schema-stable), `*_pos_fraction` is `0.0` when the compartment has
  pixels.
* **No pixels above floor**: total `= 0.0`, mean `= NaN`, area `= 0`, fraction
  `= 0.0`.
* **Empty compartment** (e.g. no cytoplasm mask for a nucleus): all value
  columns `NaN`, area `0`, fraction `NaN`.
* **`≥` is inclusive**: a pixel whose value exactly equals the floor is counted.

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
   (the publication-contrast RNA floor — i.e. `manual_rna_min` when
   `output.apply_pub_contrast_floor_to_spots` is on, or whatever the
   auto-batch / reference-image contrast pre-scan resolved). This is the SAME
   floor the spot-detection filter and the viewer's eye use.
2. Else `output.manual_rna_min` (read directly from config — so the feature
   works without the runner, e.g. in unit tests).
3. Else `NaN` → columns present but empty.

The rna2 channel uses `rna2_intensity_threshold` / `analysis_floors["rna2"]` /
`manual_rna2_min` (the antibody floor in `rna_protein`, mapped into the rna2
slot by the `rna_protein` wrapper).

Set the field to a positive number to **pin** the threshold explicitly,
independent of the spot floor.

> The feature is **independent** of the `apply_pub_contrast_floor_to_analysis`
> and `apply_pub_contrast_floor_to_spots` toggles. The `*_thresh_*` columns are
> always emitted; the runner now always forwards `analysis_floors` for the
> supported modes so the default-to-spot-floor behavior works even with both
> toggles off.

The floor is applied to the **same RNA image plane used for spot detection**
(the objective-window MIP — `rna_2d` / `rna2_2d` after z handling).

---

## Modes & channels covered

| Mode | Channel 1 | Channel 2 |
|---|---|---|
| `rna_only` (H9 MIAT — primary) | ✅ `rna_thresh_*` | — |
| `rna_rna` (BIN1 etc.) | ✅ `rna_thresh_*` | ✅ `rna2_thresh_*` |
| `rna_protein` (protein mirror) | ✅ `rna_thresh_*` | ✅ `protein_thresh_*` |

`rna_protein` routes through the `rna_rna` core with the antibody channel in the
rna2 slot, so the antibody channel gets the SAME threshold-integrate logic
(this is the protein-pipeline mirror Brian referenced) and its columns are
relabeled `rna2_*` → `protein_*` on output.

---

## Implementation map

| Piece | Location |
|---|---|
| Core helper | `src/fishsuite/core/metrics.py :: compute_thresholded_compartment_intensity(image_2d, mask, floor)` |
| Floor resolver | `src/fishsuite/core/modes/rna_only.py :: _resolve_thresh_intensity_floor(cfg, analysis_floors, channel)` (imported by `rna_rna`) |
| Per-nucleus wiring (1 ch) | `src/fishsuite/core/modes/rna_only.py` (per-nucleus loop + per-image means) |
| Per-nucleus wiring (2 ch) | `src/fishsuite/core/modes/rna_rna.py` (both channels; relabeled for `rna_protein`) |
| Config fields | `src/fishsuite/config/schema.py :: OutputCfg.rna_intensity_threshold / rna2_intensity_threshold` |
| Runner forwarding | `src/fishsuite/runner.py` (always forwards `analysis_floors` for supported modes) |
| Tests | `tests/test_threshold_intensity.py` (10 tests) |

CSV emission is DataFrame-driven (`nuclei_metrics.csv` and `per_image_summary.csv`
are written via `to_csv` over a column-union of the per-nucleus / per-image
dicts), so the new columns flow through to the master + per-image CSVs
automatically.
