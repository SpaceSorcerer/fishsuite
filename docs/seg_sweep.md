# Nuclear diameter sweep

Run `fishsuite seg-sweep -c <preset.yaml> -i <input directory> --diameters-um 9,11,13 --max-images-per-condition 1 --seed 0 --out <new directory>`.

This QC command supports Cellpose in `rna_only`, `rna_rna` and `rna_protein` modes. It uses the run's discovery, explicit input subset, channel mapping, plane preparation, segmentation wrapper and border filter. Nested traversal is opt-in via `conditions.recursive_discovery: true`, enabled only in the new UD OE preset. The default preserves the legacy flat-or-one-level discovery, including omission of root-level files when subfolders exist. Recursive folder keys use forward slashes on every platform, and excluded/underscore-prefixed directories are pruned. Unmatched filenames raise when `strict_filenames` is enabled, before subset selection.

Within each condition, names and paths are sorted, then permuted by the seeded NumPy generator; the first N images are selected. The same images are used for all diameters. Only `expected_diameter_um` is changed in each copied nuclei configuration. The model receives `diameter_um / (pixel_size_um * downsample_factor)`; the wrapper receives native diameter and performs the downsampling division once. The preset's Cellpose device and all remaining segmentation settings are preserved.

XY scale uses the same positive configured voxel override, otherwise image metadata, as the analysis path. Missing or nonpositive scale is an error. One image is opened at a time. Only rendered row rasters and scalar area measurements persist between images.

Outputs:

- `seg_sweep_contact_sheet.png`: 600 dpi, images in rows and diameters in columns. DAPI is grey with one shared range, using valid preset manual DAPI levels or the first selected plane's range. The actual range is recorded. Blue outlines are retained nuclei, orange are area exclusions and magenta are border exclusions. Every panel has a physical scale bar and model-input diameter.
- `seg_sweep_summary.csv`: image-relative path, condition, diameter, model-input pixels, retained/small/large/border counts, retained median area, area IQR and median equivalent diameter. Empty populations have blank size statistics. Areas are native-mask pixels multiplied by XY scale squared.
- `seg_sweep_area_hist.png`: retained-area distributions with common bins and axes, at 600 dpi.
- `command.log` and `versions.txt`: invocation, effective arguments, seed, fixed display range and software versions.

Area exclusions include the vendor wrapper's post-model minimum-area filter and the final native-mask area filter. Optional diagnostics observe model outputs without altering model inputs or returned masks; the vendor source remains unchanged. Objects rejected internally by the Cellpose model before it returns a label mask cannot be displayed. Ghost rejection and fixed-number nucleus sampling occur later in the full analysis and are not part of this segmentation QC.

The UD OE preset's diameter and display levels remain provisional. Set the production MIAT fixed LoG threshold from the QC threshold table, then use the declared analysis and exact-footprint downstream processing chain. This command does not select the final diameter or run association analysis.
