# Conditions, wells and fields of view

`fishsuite` uses a three-level hierarchy:

- **condition group**: what you compare, for example WT against KO;
- **well**: one biological replicate, called a condition in the config and output columns;
- **field of view (FOV)**: one image, a technical replicate inside a well.

Statistics run nucleus, then FOV mean, then well mean, then group mean. FOV means weigh equally inside a well, and well means weigh equally inside a group. Biological n is the number of wells, not images or nuclei.

## Folder layout

Put each well in its own subfolder of the input folder. Secondary-only controls get their own folder.

```
raw/my_experiment/
  WT-1/   field01.tif field02.tif ...
  WT-2/
  KO-1/
  KO-2/
  Sec-Only/
```

Then declare the groups in the config. The full `conditions` reference, including `groups`, `group_order` and `group_colors`, is in the README under [Declaring groups](../README.md#declaring-groups). For this layout:

```yaml
conditions:
  sec_only_folders: ["Sec-Only"]
  groups:
    WT: [WT-1, WT-2]
    KO: [KO-1, KO-2]
  group_order: [WT, KO]
```

If your folders name the line or treatment and the well is only in the filename (for example `sample_WT-1_00.tif` inside a `WT/` folder), add `well_from_image`, a regex with exactly one capture group that recovers the well from every biological filename:

```yaml
conditions:
  well_from_image: '_((?:WT|KO)-\d+)_'
```

Other layouts (flat folder with `filename_conditions`, nested folders with `recursive_discovery`) are in [Organising your images](../README.md#organising-your-images). Use your real well roster; these examples do not prescribe replicate counts.

## Checking the hierarchy

`fishsuite run --dry-run` lists every image with its condition. During a real run, the console prints the number of wells and FOVs per group, and the run writes:

- `resolved_experiment_hierarchy.csv`: every discovered image with source path, condition, well, group and output stem;
- `condition_report_hierarchy.csv`: the assignment used for grouped figures;
- `condition_output_status.json`: whether grouped figures completed. Incomplete grouped figures make the command fail visibly while keeping the measurement files.

Grouped runs also write `per_condition_summary.csv`, `per_well_condition_summary.csv`, `per_field_condition_summary.csv` and matching workbook sheets.

Without `groups`, each well is reported on its own. That mode is descriptive and is not the route for condition-level inference.

## Rules worth knowing

- A source-relative FOV ID tells apart duplicate filenames in different wells. Ambiguous output names fail before any image is processed.
- `reference_image` contrast mode needs a reference filename that is unique across wells.
- Discovery reads a flat folder or one level of subfolders unless `recursive_discovery: true` is set.
- `fishsuite native-figures` regroups a finished run from its CSVs. It changes grouping only, not spot detection, thresholds or footprints.

## In the GUI

Open the **Conditions** tab, section **Conditions / biological wells / technical FOVs**.

1. Map source folders to labels. With one folder per well, use distinct well IDs such as WT-1 and WT-2.
2. If folders name conditions instead, enter a filename regex with exactly one capture group in **Recover biological well from filename**.
3. Click **Load discovered wells into assignment table**, enter the group name beside each well, and click **Apply well assignments**.
4. Click **Preview condition / well / FOV hierarchy** and check that each group has the intended wells and each well its FOVs. Controls appear separately. The same check runs before processing.
5. Save the YAML. Filename mappings, group order, colours and the well regex are kept. The advanced groups YAML box is also editable.
