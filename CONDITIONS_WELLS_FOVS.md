# Conditions, biological wells and technical FOVs

Open **Conditions** in the GUI. The section **Conditions / biological wells / technical FOVs** controls biological grouping.

1. Map source folders to their original labels. With one folder per well, use distinct well IDs such as WT-1 and WT-2.
2. If source folders identify conditions instead, enter a filename regex with exactly one capture group in **Recover biological well from filename**. This must recover a unique biological well ID from every biological FOV.
3. Click **Load discovered wells into assignment table**. Enter WT, KO or your condition name beside each well. Click **Apply well assignments**.
4. Click **Preview condition / well / FOV hierarchy**. Confirm each condition has the intended biological wells and each well contains its technical FOVs. Controls appear separately. The same validation runs before processing.
5. Save the YAML normally. Loaded filename mappings, group order, colors and the well regex are preserved. The advanced groups YAML box is also editable.

For example, when the original source folders are `WT-1`, `WT-2`, `KO-1` and `KO-2`:

```yaml
conditions:
  groups:
    WT: [WT-1, WT-2]
    KO: [KO-1, KO-2]
  group_order: [WT, KO]
```

For filenames such as `sample_WT-1_00.vsi` inside a condition-labelled folder, add:

```yaml
conditions:
  well_from_image: '_((?:WT|KO)-\d+)_'
  groups:
    WT: [WT-1, WT-2]
    KO: [KO-1, KO-2]
```

Use the actual experimental well roster; these examples do not prescribe replicate counts. The statistical hierarchy is nucleus → FOV mean → well mean → condition mean. FOV means have equal weight inside a well; well means have equal weight inside a condition. Biological n is the number of wells, not images or nuclei. Existing endpoint definitions and statistical tests are unchanged.

Configured runs automatically produce condition figures, `per_condition_summary.csv`, `per_well_condition_summary.csv`, `per_field_condition_summary.csv`, and matching workbook sheets. `resolved_experiment_hierarchy.csv` records the complete discovered roster, source paths and exact output stems. `condition_report_hierarchy.csv` records the report assignment separately. `condition_output_status.json` distinguishes current-attempt completion from measurement output; incomplete condition figures make the command fail visibly while preserving measurement files.

No-group mode remains available as explicitly unspecified biological grouping with legacy output. It is not the condition-level inference route. Existing legacy per-well figures remain in their supplementary directory when grouped figures are generated.

A source-relative FOV ID distinguishes duplicate basenames across wells; original paths and output stems remain recorded. A selected relative path selects exactly that field. Ambiguous per-image artifact names fail before image processing. Reference-image contrast mode requires an unambiguous reference basename. Ambiguous historical publication-image suffix matches fail instead of selecting another well's image.

Discovery currently accepts a flat directory or one level of source folders. Literal nested condition/well directories are not recursively scanned; the explicit metadata above supplies the statistical hierarchy without moving raw files. CSV-only regrouping changes grouping, not spot detection, thresholds, or footprint calculations. Use measurements that already embody the intended scientific filtering.
