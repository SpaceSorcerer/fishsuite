# fishsuite file map

One line per directory or file worth knowing about. Paths are relative to the
repository root. Update this file when you add, move or delete a file.

## Top level

| Path | What it is |
|---|---|
| `README.md` | User documentation: install, modes, config keys, outputs. |
| `CHANGELOG.md` | Release notes. |
| `CONTRIBUTING.md` | Development setup, test markers, documentation rules. |
| `POSTRUN_UTILITIES.md` | CPU post-run subcommands that read a finished run (backfill, singlecell, pixelpattern, sizefit, ...). |
| `THRESHOLD_INTENSITY_FEATURE.md` | Thresholded RNA intensity per compartment. |
| `CITATION.cff`, `.zenodo.json` | Citation and archive metadata. |
| `LICENSE` | License. |
| `pyproject.toml` | Package metadata, dependencies, `fishsuite` console script, wheel preset allow-list, pytest markers, ruff. |
| `fishsuite.bat` | Windows double-click launcher for the GUI (`pythonw -m fishsuite.gui`). |
| `.github/workflows/test.yml` | CI: light tests on a matrix, ruff, build, wheel smoke test, absolute-path gate, preset allow-list gate. |
| `.gitattributes` | LF normalisation; vendored files are stored byte-for-byte. |
| `assets/` | Icon files and the script that draws them. |

## Source: `src/fishsuite/`

| Path | What it is |
|---|---|
| `cli.py` | Click entry point. Subcommands: `run`, `init`, `preview`, `gui`, `presets list/show`, `seg-sweep`, `report`, `figure`, `native-figures`, `fig-panels`, `backfill`, `sizefit`, `footprint-backfill`, `walkthrough`, `singlecell`, `pixelpattern`, `if-pub-images`, `postrun`, `qki-assoc`, `ortho`, `coupling`. |
| `runner.py` | Batch runner: discovers inputs, runs the chosen mode per image, writes the output directory. |
| `postrun.py` | Helpers for measurements derived from completed runs. |
| `config/schema.py` | Pydantic v2 schema for the YAML config. |
| `config/hierarchy.py` | Resolves condition, well and field-of-view metadata. |
| `config/presets/` | Shipped presets. Only the portable ones listed in `pyproject.toml` go into the wheel. |
| `core/io.py` | Image reading (bioio), channel autodetect, input discovery. |
| `core/pixel_size.py` | The single lateral pixel-size resolver. |
| `core/image_preprocess.py` | Bright-artifact removal. |
| `core/segmentation.py` | Nuclear segmentation wrapper (StarDist, Cellpose, Otsu). |
| `core/spots.py`, `core/spot_size.py`, `core/thresholds.py` | Spot detection, Gaussian size fit, threshold computation. |
| `core/morphology.py`, `core/nucleolus.py`, `core/nuclear_size.py` | Masks, spot stratification, nucleolus and chromatin metrics, physical nuclear size. |
| `core/metrics.py`, `core/coloc_pixel_metrics.py` | Per-nucleus pixel-colocalization metrics. |
| `core/footprint_null.py`, `core/texture_null.py`, `core/qki_association.py` | Exact-footprint construction, rotation and texture nulls, partner association. |
| `core/exact_footprint_*.py` | Exact single-plane backfill, post-run statistics, figures, superplots and publication renderers. |
| `core/coloc_backfill.py`, `core/sizefit_backfill.py` | CPU backfills onto a finished run. |
| `core/singlecell.py`, `core/pixel_pattern.py`, `core/_superplot.py` | Per-nucleus post-run analyses and the shared SuperPlot drawer. |
| `core/ortho_profile.py`, `core/radial_profile_figure.py`, `core/walkthrough_figure.py` | Orthogonal sections, radial profile and pipeline-walkthrough figures. |
| `core/output.py`, `core/excel_report.py` | Per-image renders, CSV writing, Excel deliverable. |
| `core/qc.py`, `core/repro.py`, `core/parallel.py` | QC flags, seeds and provenance, worker-count resolution. |
| `core/well_key.py`, `core/authoritative_sampling.py`, `core/native_by_condition.py`, `core/seg_sweep.py` | Replicate-well identity, sampling-label repair, condition figures, nuclear-diameter sweep. |
| `core/modes/` | One module per analysis mode: `rna_only`, `rna_rna`, `rna_protein`, `if_intensity` (with `if_report`, `if_pub_images`); `ab_ab`, `protein_only`, `pub_images` are stubs. `_zplanes.py` holds shared z-plane extraction. |
| `core/_vendor/` | Verbatim copies from the original pipeline; checksums in `PROVENANCE.md`, enforced by `tests/test_vendor_parity.py`. Do not edit. |
| `report/` | `fishsuite report`: run loading and hierarchy (`aggregate.py`), endpoints, replicate statistics, figures, `REPORT.xlsx` (`workbook.py`), slides, delivery packagers. |
| `figures/` | Locked figure-panel module behind `fig-panels`: FOV panels, linked ortho sets, representative crops, superplots, scatters, stats, style. |
| `gui/` | PySide6 launcher: main window, config state, readiness checks, subprocess runner, report tab, widgets. |

## Tests: `tests/`

`pytest` runs from the repository root. Markers `lab`, `gpu`, `heavy` and
`bioformats` gate tests that need local data, a GPU, the full ML stack or a
JVM; CI runs `-m "not heavy"`. `_texture_null_sim.py` builds the synthetic
nuclei used by the texture-null tests and calibration script.

## Scripts and tools

| Path | What it is |
|---|---|
| `scripts/check_no_absolute_paths.py` | CI gate: no machine-specific absolute path as an active value in `src/`. |
| `scripts/coloc_standard_panel.py`, `scripts/coloc_extra_metrics.py` | Field-standard and shared-floor colocalization readouts for a completed run. |
| `scripts/render_pub_images.py` | Re-render publication images of a completed run at a new display window. |
| `scripts/texture_null_calibration.py`, `scripts/texture_null_sensitivity_report.py` | Texture-null calibration on synthetic fields and its workbook. |
| `tools/single_z_review/` | Manual z-plane review packet from an inventory CSV (`python -m tools.single_z_review`). |
| `tools/miat_qki_publication/` | Downstream-only orchestrator that builds a final-publication directory from retained sources. |

## Documentation: `docs/`

| Path | What it is |
|---|---|
| `QUICKSTART.md`, `GUI.md` | Getting started on the command line and in the GUI. |
| `CONDITIONS.md` | Conditions, wells and fields of view. |
| `REPORT_SUBCOMMAND.md`, `REPORT_FEATURES.md` | The `report` subcommand and what it produces. |
| `FOOTPRINT_ASSOCIATION.md`, `SPOT_SIZE_FIT.md`, `seg_sweep.md` | Footprint association metrics, Gaussian size fit, nuclear-diameter sweep. |
| `dev/VENDOR_PARITY.md` | Vendoring parity record. |
| `gui_light.png`, `gui_dark.png` | GUI screenshots used by the README. |
