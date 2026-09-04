# Per-spot Gaussian size fit (`size_fit_*`) — 2026-09-04

Closes the open item named at the end of the README section "Punctum size: read
the footprint, not the moment estimator": a genuine per-punctum Gaussian fit now
runs at detection time and can be backfilled onto finished runs.

Additive only. `spot_diameter_um`, `spot_fwhm_px` and `spot_area_px` keep their
existing values and their existing meaning, so older runs stay comparable.

## Why the legacy columns are near-constant

`spot_diameter_um` IS computed per spot — it is not the configured BigFISH
radius — but the estimator cannot resolve a real size range.

| Where | What |
|---|---|
| `src/fishsuite/core/modes/rna_rna.py:224` | `_measure_spot_diameter_um`, a second central moment on a background-subtracted crop |
| `src/fishsuite/core/modes/rna_rna.py:228` | `crop_half: int = 4`, so the crop is FIXED at 9x9 px |
| `src/fishsuite/core/modes/rna_rna.py:275-278` | the moment is summed over the WHOLE crop, so a uniformly filled 9x9 window caps it: `var_y = var_x = (81-1)/12 = 6.667`, `sigma = sqrt(13.33/2) = 2.58` px, `FWHM = 6.08` px |
| `src/fishsuite/core/modes/rna_rna.py:279` | `max(var / 2.0, 0.25)` floors sigma at 0.5 px, so `FWHM >= 1.18` px |
| `src/fishsuite/core/modes/rna_rna.py:3347` | `spot_fwhm_px = spot_diameter_um / voxel_xy_um`, a unit conversion of the same quantity |
| `src/fishsuite/core/modes/rna_rna.py:3348` | `spot_area_px = pi * (FWHM/2)**2`, which inherits it again |

Between that floor and that ceiling the background pedestal left after the
10th-percentile subtraction dominates the `r**2`-weighted sum, so the estimate
compresses towards the middle of the window's range.

Measured on synthetic Gaussians with Poisson noise
(`tests/test_spot_size_fit.py::test_moment_estimator_saturates_while_the_fit_does_not`,
seed 0, amplitude 2000 over background 300):

| true sigma (px) | true FWHM (px) | moment estimator FWHM (px) |
|---|---|---|
| 0.8 | 1.88 | 2.99 |
| 1.2 | 2.83 | 3.07 |
| 1.8 | 4.24 | 3.83 |
| 2.5 | 5.89 | 4.32 |
| 3.5 | 8.24 | 4.55 |

A 4.4-fold range in true FWHM is reported as a 1.5-fold range, and the estimator
is flat above sigma about 2.5 px.

## What the fit does

Model, on the plane the spot was DETECTED on:

    z(x, y) = A * exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2)) + B

Five parameters, solved by batched Levenberg-Marquardt in
`src/fishsuite/core/spot_size.py`. All spots advance together in numpy, so a
138k-spot run costs seconds. Agreement with `scipy.optimize.curve_fit` on the
same crops is asserted in `test_matches_scipy_curve_fit_on_a_subset`
(max absolute sigma difference below 0.02 px).

Window: `foci.size_fit_window_px`, default 9 px. Chosen from a
window-convergence scan on real 65 nm/px data, not from theory. On the
RNASEH2B / BIN1-intron 561 channel the fitted sigma settles at about 2.42 px
from window 11 upward, window 9 returns 2.37 px (2 % low), and windows above 11
lose spots to neighbours entering the crop.

A spot whose fitted sigma exceeds half its window is extrapolated rather than
measured, so it is refitted in +4 px steps up to `max_window_px` (15). The wider
fit replaces the narrower one ONLY if it is itself sound, so escalation can
never degrade a spot that was already measured.

## Columns

Per spot, `spot_metrics.csv` (both channels), and `spot_metrics_sizefit.csv` for
backfilled runs:

| Column | Meaning |
|---|---|
| `size_fit_sigma_px` | fitted Gaussian standard deviation, px |
| `size_fit_fwhm_px` | `2*sqrt(2*ln2) * sigma` |
| `size_fit_fwhm_um` | the same, times the run's `voxel_xy_nm / 1000` |
| `size_fit_amplitude` | fitted `A`, above the fitted background |
| `size_fit_background` | fitted `B` |
| `size_fit_r2` | fraction of crop variance explained |
| `size_fit_window_used_px` | the window this spot was finally fitted on |
| `size_fit_ok` | 1 only when every acceptance check passed |
| `size_fit_flag` | why it was rejected, see below |

Filter on `size_fit_ok == 1` before any summary. Rejected fits keep their
numbers because they are still informative, but they are not measurements.

| `size_fit_flag` | Meaning |
|---|---|
| `ok` | converged inside its own window, `A > 0`, `0.3 <= sigma <= window/2`, centre moved at most 1.5 px, `r2 >= 0.5` |
| `wider_than_window` | sigma exceeded half the widest window tried; the number is an extrapolation |
| `center_drift` | the fitted centre moved more than 1.5 px, usually a brighter neighbour inside the crop |
| `low_r2` | a Gaussian plus a constant does not describe the crop, e.g. a diffuse channel with no isolated object |
| `bad_amplitude` | non-positive amplitude, i.e. the crop centre sits in a trough |
| `edge` | the window did not fit inside the frame |

Footprint columns, emitted where the run already stored
`miat_footprint_area_px`:

| Column | Meaning |
|---|---|
| `footprint_area_um2` | `miat_footprint_area_px * (voxel_xy_um)**2` |
| `footprint_equiv_diameter_um` | diameter of the disk of that area |

Rollups, per nucleus in `nuclei_metrics.csv` and per image in
`per_image_summary.csv`, prefixed `rna1_` / `rna2_`:
`median_size_fit_fwhm_um`, `median_footprint_area_um2`, `n_size_fit_ok`,
`frac_size_fit_ok`. The per-image rollup uses nuclear spots only.

## Backfilling a finished run

    fishsuite sizefit --run <run_dir> [--input-dir <image_tree>] [--window-px 9]

Re-opens each source image, re-extracts the z-plane the run recorded in
`per_image_summary.z_plane`, and fits every spot in `spot_metrics.csv` on its own
detection plane. Size is invariant to a multiplicative rescale of the plane, so a
run that pedestal-normalised rna1 before detection needs no special handling.

Writes beside the run's tables and modifies nothing already there:
`spot_metrics_sizefit.csv`, `sizefit_per_nucleus.csv`, `sizefit_per_image.csv`,
`sizefit_command.log`, `sizefit_versions.txt`.

## What this does and does not settle

The fit measures the width of the intensity profile at one z-plane. It is not a
count of molecules, not a 3-D volume, and not a claim that a punctum is Gaussian.
On a channel with no isolated compact objects the model has no scale of its own:
the fitted sigma then grows with whatever window it is given, which shows up as a
low `frac_size_fit_ok` dominated by `center_drift` and `low_r2` rather than as a
plausible-looking number. Read the flag breakdown before reading a median.
