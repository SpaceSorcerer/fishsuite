# Spot size: the Gaussian fit (`size_fit_*`)

Every detected spot gets a 2D Gaussian fit at detection time. Use the `size_fit_*` columns, or the exact footprint, to compare punctum sizes. The older `spot_diameter_um`, `spot_fwhm_px` and `spot_area_px` columns keep their values and meaning, but they saturate and should be treated as descriptive only.

## Why the moment-based columns saturate

`spot_diameter_um` is measured per spot, not copied from the configured BigFISH radius. It is the second central moment of a background-subtracted crop around the spot (`_measure_spot_diameter_um` in `core/modes/rna_rna.py`). The crop is fixed at 9 by 9 pixels, so the moment is truncated: the estimate can never exceed about 6.1 px FWHM (a uniformly filled window) and is floored at about 1.2 px. Between those limits, the leftover background inside the crop pulls the estimate toward the middle. `spot_fwhm_px` and `spot_area_px` are derived from the same number.

Measured on synthetic Gaussians with Poisson noise (`tests/test_spot_size_fit.py::test_moment_estimator_saturates_while_the_fit_does_not`):

| true sigma (px) | true FWHM (px) | moment estimate FWHM (px) |
|---|---|---|
| 0.8 | 1.88 | 2.99 |
| 1.2 | 2.83 | 3.07 |
| 1.8 | 4.24 | 3.83 |
| 2.5 | 5.89 | 4.32 |
| 3.5 | 8.24 | 4.55 |

A 4.4-fold range in true width reads as a 1.5-fold range, and the estimate is flat above a sigma of about 2.5 px. Two conditions can look identical on these columns while their puncta differ in size. `fishsuite report` therefore uses the exact half-maximum footprint as its size endpoint and marks the moment columns as descriptive.

## What the fit does

Model, on the plane where the spot was detected:

    z(x, y) = A * exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2)) + B

Five parameters are solved by batched Levenberg-Marquardt in `core/spot_size.py`, so large runs take seconds. A test checks agreement with `scipy.optimize.curve_fit`.

The window is `foci.size_fit_window_px` (default 9 px). It was chosen from a window-convergence scan on 65 nm/px data; retune it for other pixel sizes. A spot whose fitted sigma exceeds half its window is refitted on wider windows in 4 px steps up to 15 px. A wider fit replaces the narrower one only if it passes every check.

Set `foci.compute_size_fit: false` to turn the fit off.

## Columns

Per spot, in `spot_metrics.csv` (and in `spot_metrics_sizefit.csv` for backfilled runs):

| Column | Meaning |
|---|---|
| `size_fit_sigma_px` | fitted Gaussian standard deviation, px |
| `size_fit_fwhm_px` | `2*sqrt(2*ln2) * sigma` |
| `size_fit_fwhm_um` | the same, times the run's pixel size |
| `size_fit_amplitude` | fitted `A`, above background |
| `size_fit_background` | fitted `B` |
| `size_fit_r2` | fraction of crop variance explained |
| `size_fit_window_used_px` | the window of the final fit |
| `size_fit_ok` | 1 only when every check passed |
| `size_fit_flag` | why the fit was rejected |

Filter on `size_fit_ok == 1` before summarising. Rejected fits keep their numbers but are not measurements.

| `size_fit_flag` | Meaning |
|---|---|
| `ok` | converged, `A > 0`, `0.3 <= sigma <= window/2`, centre moved at most 1.5 px, `r2 >= 0.5` |
| `wider_than_window` | sigma exceeded half the widest window; the value is an extrapolation |
| `center_drift` | the centre moved more than 1.5 px, usually toward a brighter neighbour |
| `low_r2` | a Gaussian plus a constant does not describe the crop, for example a diffuse channel |
| `bad_amplitude` | non-positive amplitude |
| `edge` | the window did not fit inside the frame |

Where the run stored `miat_footprint_area_px` (the RNA1 punctum's exact footprint; the name is a legacy token), two more columns are written:

| Column | Meaning |
|---|---|
| `footprint_area_um2` | footprint area times pixel area |
| `footprint_equiv_diameter_um` | diameter of a disk with that area |

Rollups, per nucleus in `nuclei_metrics.csv` and per image in `per_image_summary.csv`, prefixed `rna1_` or `rna2_`: `median_size_fit_fwhm_um`, `median_footprint_area_um2`, `n_size_fit_ok`, `frac_size_fit_ok`. The per-image rollup uses nuclear spots only.

## Adding the fit to a finished run

    fishsuite sizefit --run <run_dir> [--input-dir <image_tree>] [--window-px 9]

This re-opens each image, takes the plane recorded in `per_image_summary.z_plane`, and fits every spot in `spot_metrics.csv`. It writes `spot_metrics_sizefit.csv`, `sizefit_per_nucleus.csv`, `sizefit_per_image.csv`, `sizefit_command.log` and `sizefit_versions.txt` beside the run's tables, and changes nothing already there.

## What the fit does not tell you

The fit measures the width of the intensity profile on one plane. It is not a molecule count, not a 3D volume, and not a claim that a punctum is Gaussian. On a channel with no isolated compact objects the fit has no scale of its own; this shows up as a low `frac_size_fit_ok` dominated by `center_drift` and `low_r2`. Read the flag breakdown before reading a median.
