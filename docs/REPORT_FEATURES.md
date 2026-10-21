# Report features

Optional features of `fishsuite report`. The basics (groups, outputs, statistics, provenance) are in [REPORT_SUBCOMMAND.md](REPORT_SUBCOMMAND.md). In the examples, set `$run`, `$out`, `$groups`, `$spec`, `$panel`, `$manifest`, `$preset` and `$repo` to your own paths.

## Plot styles

Flags: `--plot-style superplot|replicate-simple` (default `superplot`), `--technical-layer none|fov`. Both can also be set in the groups YAML; the command line wins.

`replicate-simple` draws well means as coloured points. The column is the mean of well means (width 0.6, 35% fill, 1 pt edge) with capped sample SD across wells. Columns and SD bars need at least two wells. `--technical-layer fov` adds small muted field points; nuclei are not drawn. The footer reads "bar = mean of well means, error bar = ± SD".

Every plot and composite is exported as a `_full` and a `_focus` pair, each as 600 dpi PNG and editable-text SVG, with identical values and statistics. Full axes start at zero; percentages span 0 to 100. Focus axes for percentages use 50 to 100 only when every well mean exceeds 50; other focus axes round outward around the data and never go below zero for non-negative values. Workbook figure indexes and decks default to `_focus` and list `_full` beside it.

Compare groups on the same endpoint with the same axis variant. Do not zoom each group separately.

The bracket shows the headline p (mixed model, or the labelled Welch fallback). Descriptive and absolute-intensity endpoints say "descriptive, no test".

```bash
fishsuite report --run "$run" --groups-file "$groups" --out "$out" --plot-style replicate-simple --technical-layer fov
```

## DAPI localization endpoints

Set `dapi_localization: true` in a deck specification passed with `--deck-spec`. There is no separate switch. This needs a validated persisted panel and pinned selection and micrograph inputs.

Report-time DAPI QC leaves the run's segmentation unchanged. Each spot is classed as `in_retained_nucleus` (inside DAPI and a retained nucleus), `in_unretained_dapi_object` (inside DAPI but outside every retained nucleus) or `extranuclear` (outside DAPI, even if inside a retained mask). The second class can include parts of a DAPI object that overlaps a retained nucleus; object-overlap columns make that explicit.

Endpoints: `nuclear_spot_count_dapi`, `extranuclear_spot_count_per_cell_territory`, and `nuclear_spot_fraction_dapi` = nuclear / (nuclear + extranuclear in that cell's territory). Spots in unretained DAPI objects are left out of the denominator. Sheets `DAPI census`, `DAPI census by arm`, `DAPI objects`, `DAPI spot classes` and `DAPI class summary` record the details.

`--qc-cyto-calls` exports every RNA1 cytoplasmic call as calibrated QC crops and a CSV. It is separate from the DAPI reclassification.

```bash
fishsuite report --run "$run" --groups-file "$groups" --out "$out" --deck-spec "$spec" --qc-cyto-calls
```

## Statistical sensitivities

These are always computed; there is nothing to switch on. The mixed model uses REML with a fixed group effect, a random well intercept and a field-within-well variance component. Fit status, warnings and nucleus, field and well counts are kept.

Contrasts also carry an equal-variance t on well means and Welch on field means. The field-level test treats technical replicates as independent and is a sensitivity only.

`FOV outlier sensitivity` flags at most one field per well and endpoint: the field with the largest absolute leave-one-out z, only if the well has at least three fields and |z| > 2.5 against the other fields' sample SD. The sheet recomputes well means and Welch without flagged fields and shows before and after. The main analysis is not changed.

## Persisted colocalization panel and simple pixel metrics

Flags: `--existing-coloc FILE` and `--baseline-manifest FILE` import a frozen standard-panel workbook instead of generating one. Import checks source hashes and cohort, never generates new nulls, and rejects new peak floors, sampled filtering or field exclusions. `--no-coloc-panel` skips the panel.

Simple metrics are enabled by the deck-spec keys `simple_coloc_csv` and `simple_coloc_image_paths`. Pearson r and Li ICQ are threshold-free. Manders M1/M2 use only nuclei where Costes converged; failed fits stay missing. Older Costes-with-fallback columns mix two methods and must not be labelled Costes-only. Values aggregate nucleus, then field, then well.

The standard-panel script's `--call-threshold auto` picks its primary thresholded panel by a 90% Costes-convergence floor across biological nuclei. This does not change the converged-only Manders figures.

`--peak-floor` is post-detection gating, not a pixel-metric threshold, and cannot be combined with a frozen panel import.

```bash
fishsuite report --run "$run" --groups-file "$groups" --out "$out" --existing-coloc "$panel" --baseline-manifest "$manifest" --deck-spec "$spec"
```

## Preferential-association ratio

`--miat-qki DIRECTORY` (with optional `--miat-qki-count-report DIRECTORY`) reads a persisted delivery folder from one published dataset and reports its preferential ratio. It rejects deck export and regrouping. The flag name keeps the dataset's tokens: read them as RNA1 and partner.

With T the total and A the associated measurement in matched wells: `rT = mean(T_treated)/mean(T_control)`, `rA = mean(A_treated)/mean(A_control)`, and `R = rA/rT`. R is a ratio of group means, not a mean of per-well ratios. R > 1 means the associated fraction is preferentially retained. The 95% CI is `exp(log(R) ± 1.96 × SE_logR)` by the multivariate delta method with within-group A/T covariance. Exact tests enumerate all balanced well assignments. `R_MDE_05` is the plug-in 80%-power detectable ratio at alpha 0.05.

```bash
fishsuite report --miat-qki "$data" --out "$out"
```

## Deck export

Flags: `--deck-spec FILE` prepares assets, `--deck` exports PowerPoint (needs the `deck` extra), and `--micrograph-slides per-well` replaces the overview micrograph slide with matched per-well slides. `--deck` needs a spec, and a spec needs a persisted panel. `--no-figures` still allows requested deck assets.

The spec supplies an ordered `slides` list. `deck_spec.resolved.yaml` is the resolved map, with workbook-traced values and hashed assets. The `Deck specification`, `Slide values` and `Figure sources` sheets keep each value's workbook source, and `slide_sources.csv` records workbook cells and figure hashes. Export rejects untraced numbers and stale sources.

Per-well micrographs reuse four native publication panels per group with their saved LUTs, display windows and calibration. The field nearest the well's median retained-nucleus count is chosen, with ties broken by image name. Chart assets are written under `figures/` as separate full and focus PNG and SVG files.

```bash
fishsuite report --run "$run" --groups-file "$groups" --out "$out" --deck-spec "$spec" --deck --micrograph-slides per-well
```

## Grouped figures from a finished run

`fishsuite native-figures --run DIRECTORY --groups FILE --out DIRECTORY` rebuilds the run's group-level figures from its CSVs, without detection or segmentation. Here `--groups` is a YAML file holding a `conditions` block or a `groups` block (see [Declaring groups](../README.md#declaring-groups)). Only grouping applies; peak floors and nucleus filters are not. The default output is `<run>/figures_by_condition_<timestamp>/`. Existing destinations are refused, and a frozen run needs an `--out` outside it.

## Frozen outputs

The report refuses to write into a released delivery. A `DELIVERY_*` folder counts as frozen when it holds `MANIFEST_SHA256.tsv`, `CHECKSUMS.sha256`, `CHECKSUMS.txt`, or an entry starting with `SUPERSEDED_BY`, `CHECKSUMS` or `MANIFEST_SHA256`. The check covers the destination and its parents, by path and after resolving links. Use a fresh `--out` for a released run.

## Provenance

`--engine-repo DIRECTORY` records that checkout's git HEAD, labelled "not producing identity". `--preset FILE` records the preset path and MD5. The producing commit is read only from the run's own `versions.txt`; if absent it is reported as "missing from run", never guessed.
