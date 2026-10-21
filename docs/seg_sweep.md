# Nuclear diameter sweep

Use `fishsuite seg-sweep` before a run to choose the nuclear diameter for Cellpose:

```bash
fishsuite seg-sweep -c my_experiment.yaml -i raw/my_experiment --diameters-um 9,11,13 \
    --max-images-per-condition 1 --seed 0 --out results/seg_sweep_01
```

It supports Cellpose in `rna_only`, `rna_rna` and `rna_protein` modes, and uses the same discovery, input subset, channel mapping, plane preparation, segmentation wrapper and border filter as a run. Discovery follows the `conditions` settings described in [Organising your images](../README.md#organising-your-images).

Within each condition, images are sorted and then shuffled with the seed; the first N are used, and the same images are used for every diameter. Only `nuclei.expected_diameter_um` changes between columns. The model receives `diameter_um / (pixel_size_um * downsample_factor)`. The preset's Cellpose device and other segmentation settings are kept.

The pixel size comes from `foci.bigfish_voxel_size_nm` when it is positive, otherwise from image metadata. A missing pixel size is an error. Images are opened one at a time.

## Outputs

- `seg_sweep_contact_sheet.png`: 600 dpi, images in rows and diameters in columns. DAPI is grey with one shared display range (the preset's manual DAPI levels if valid, otherwise the first plane's range; the range used is recorded). Blue outlines are kept nuclei, orange are area exclusions, magenta are border exclusions. Every panel has a scale bar and the model-input diameter.
- `seg_sweep_summary.csv`: image path, condition, diameter, model-input pixels, kept, small, large and border counts, median kept area, area IQR and median equivalent diameter. Areas are mask pixels times pixel area.
- `seg_sweep_area_hist.png`: kept-area distributions on common bins and axes.
- `command.log` and `versions.txt`: the command, arguments, seed, display range and software versions.

Area exclusions include the segmentation wrapper's minimum-area filter and the final area filter. Objects the Cellpose model rejects internally before returning a mask cannot be shown. Ghost rejection and fixed-N sampling happen later in a full run and are not part of this check.

## Using the result

Pick the diameter whose kept nuclei match real nuclei without splits or merges, set `nuclei.expected_diameter_um` in your preset, and run. The sweep does not choose the diameter for you.
