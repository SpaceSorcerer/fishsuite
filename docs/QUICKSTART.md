# Quickstart

From raw images to a condition report in six steps. The [README](../README.md) explains every option named here.

## 1. Install

```bash
python -m venv fishsuite-env
source fishsuite-env/bin/activate          # Windows: fishsuite-env\Scripts\activate
pip install "fishsuite[all]"               # or, from a clone: pip install -e ".[all]"
pip install tensorflow                     # only if you will use the StarDist backend
```

`[all]` adds the Bio-Formats reader (needs a JDK, for `.vsi`, `.czi`, `.lif`, `.nd2`) and the desktop GUI. A base install reads TIFF and OME-TIFF. Do not install other bioio reader plugins. See [Installation](../README.md#installation).

## 2. Stage your images, one folder per well

```
raw/my_experiment/
  Control_1/   field01.tif  field02.tif ...
  Control_2/
  Treated_1/
  Treated_2/
  Sec-Only/
```

Keep the raw data read-only; copy or link it into a staging folder if you need to rename anything. Files at the top level are ignored when subfolders exist. Other layouts (condition in the filename, nested folders) are in [Organising your images](../README.md#organising-your-images).

## 3. Copy a portable preset

```bash
fishsuite presets list
fishsuite presets show generic_100x_0p065 > my_experiment.yaml
```

Pick the preset closest to your objective and mode (`generic_60x_0p108`, `generic_rna_rna_100x`, `generic_rna_protein_100x` and others). Then edit `my_experiment.yaml`:

- `channels`: set `dapi`, `rna` (and `rna2` or `antibody` for two-channel modes) to your channel indices.
- `conditions`: list secondary-only folders in `sec_only_folders`, and add `groups` and `group_order` so wells are compared as replicates. The YAML is shown in [Declaring groups](../README.md#declaring-groups).
- `nuclei.min_area_px` and other pixel-based sizes: scale them to your pixel size. Set `foci.bigfish_voxel_size_nm` if your images carry no pixel-size metadata.
- `nuclei.backend: cellpose` if you did not install TensorFlow.

## 4. Dry run

```bash
fishsuite run -c my_experiment.yaml -i raw/my_experiment -o results/my_experiment_run1 --dry-run
```

Check that every image is listed, each has the right condition, and the controls are flagged secondary-only. To debug settings on one image first:

```bash
fishsuite preview raw/my_experiment/Control_1/field01.tif -c my_experiment.yaml -o results/preview01
```

## 5. Run

```bash
fishsuite run -c my_experiment.yaml -i raw/my_experiment -o results/my_experiment_run1
```

Use a new output folder every time. Open `qc_overlays/` first to confirm segmentation and spot calls, then `analysis_summary.xlsx`. The full list of outputs is in [Outputs](../README.md#outputs).

## 6. Report

```bash
fishsuite report --run results/my_experiment_run1
```

This writes `results/my_experiment_run1/report_<stamp>/` with `REPORT.xlsx`, `READOUT.md` and figures. Groups come from the run's config, or pass `--groups Control=Control_1,Control_2 --groups Treated=Treated_1,Treated_2`. See [docs/REPORT_SUBCOMMAND.md](REPORT_SUBCOMMAND.md).
