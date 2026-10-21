# Footprint association metrics

Continuous, floor-robust measures of how much partner signal sits on each RNA1 punctum. They are written during `fishsuite run` in `rna_rna` and `rna_protein` mode when `foci.compute_footprint_enrichment: true`. The default is off, which leaves outputs unchanged.

Column names carry legacy tokens from the dataset they were built for: read `miat` as RNA1 (the punctate RNA) and `qki` as the partner channel (RNA2 or protein).

## The footprint

For each RNA1 spot, the partner intensity is averaged over the spot's exact half-maximum footprint, so the sampled area grows with the spot's real size instead of using a fixed disk:

1. A window is taken around the spot, sized from its measured `spot_diameter_um`.
2. The local peak is the brightest RNA1 pixel within 1 px of the centre; the local background is the window's 10th percentile.
3. The threshold is `background + 0.5 × (peak − background)`.
4. The footprint is the 8-connected region of RNA1 pixels at or above that threshold that contains the seed, so a neighbouring spot is excluded.
5. Raw partner intensity is averaged over exactly those pixels.

If the footprint cannot be formed (window clipped at the edge, flat crop, seed below threshold), the spot falls back to a disk of radius FWHM/2.

## Columns

Per spot (`spot_metrics.csv`, RNA1 rows; other rows are NaN):

| Column | Meaning |
|---|---|
| `qki_at_miat_footprint` | raw mean partner intensity over the footprint |
| `miat_footprint_area_px` | footprint size in pixels |
| `qki_footprint_enrichment` | `qki_at_miat_footprint` divided by the nucleus's mean partner intensity. The main per-spot metric. |

Per nucleus (`nuclei_metrics.csv`):

| Column | Meaning |
|---|---|
| `qki_assoc_ratio_continuous` | mean `qki_footprint_enrichment` over the nucleus's RNA1 spots. The main per-nucleus metric. |
| `qki_assoc_ratio_gated_<floor>` | the same, with spots below `foci.assoc_qki_floor` counted as 0. Written only when a floor is set; floor-sensitive by design. |
| `coloc_moc` | Manders' overlap coefficient on nuclear pixels, threshold-free (same as `coloc_cosine_overlap_rna1_rna2`) |
| `coloc_icq` | Li's ICQ on nuclear pixels, threshold-free (same as `coloc_li_icq_rna1_rna2`) |
| `qki_at_miat_foci_enrichment` | mean partner over the union of all footprints in the nucleus, divided by the nuclear mean |
| `qki_associated_with_miat` | integrated partner intensity inside nuclear RNA1 footprints (mean × area, summed) |
| `miat_mass_nuclear` | integrated RNA1 signal of the same spots (peak × nominal spot area) |
| `capacity_qki_at_miat` | `qki_associated_with_miat` divided by the nucleus's total partner intensity: the fraction of nuclear partner held in RNA1 footprints. NaN when the denominator is not positive. |

The last three are extensive (amounts); the enrichment columns are intensive (fold changes). They sit beside the thresholded Manders columns, so gated and ungated views can be compared. Report them with the rotation null and the standard colocalization panel.

Per image (`per_image_summary.csv`): nucleus means `mean_qki_assoc_ratio_continuous`, `mean_coloc_moc`, `mean_coloc_icq`, `mean_qki_at_miat_foci_enrichment`, `mean_qki_associated_with_miat`, `mean_miat_mass_nuclear`, `mean_capacity_qki_at_miat`, and `mean_qki_assoc_ratio_gated_<floor>` when a floor is set.

## Enabling

```yaml
foci:
  compute_partner_intensity: true
  compute_footprint_enrichment: true
  assoc_qki_floor: null        # optional floor for the gated ratio
```

The footprint loop runs in the main process. Quantify the partner on single planes; max projections of z-stacks can saturate a bright partner channel.

`fishsuite footprint-backfill` reconstructs exact footprints on a finished run; it is dataset-specific (see the [command reference](../README.md#command-reference)).
