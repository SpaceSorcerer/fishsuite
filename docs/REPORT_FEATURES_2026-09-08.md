# Report features — 2026-09-07/08

Source scope: `19495c6..fca3a12`, including the September 8 native-condition update.
This page supersedes the older Welch-headline and per-well-native descriptions in [the September 4 guide](REPORT_SUBCOMMAND_2026-09-04.md); see also [QUICKSTART](QUICKSTART.md).
Examples are templates, not executed analyses: set `$run`, `$out`, `$groups`, `$spec`, `$panel`, `$manifest`, `$data`, `$prior`, `$preset`, and `$repo` to existing absolute input paths or fresh absolute output paths as appropriate. No dataset or deck specification is supplied with this documentation task.

## Replicate-simple locked style

Flags: `--plot-style replicate-simple`, `--technical-layer none|fov`, `--style brian`.
Well means are coloured points; the column is the mean of well means (width 0.6, 35% fill, solid 1 pt edge), with capped ± sample SD across wells (`ddof=1`). Columns/SD require at least two finite wells. Optional small muted FOV points are technical replicates; nuclei are not drawn in this layout.
Each supported plot/composite has `_full` and `_focus` 600-dpi PNG plus editable-text SVG. Both variants contain identical values and statistics. Full axes include zero (and negative data when present); percentage ticks span 0–100. Focus percentages use 50–100 only when every well mean exceeds 50; other focus limits round outward around displayed values. Bracket headroom can extend above 100.
Shared-axes rule: compare arms on the same endpoint axis and use the same variant; do not independently zoom each arm. Different endpoint units have their own axes, not a universal shared scale. Focus columns begin at the displayed axis floor.
The bracket uses the successful nucleus-level mixed-model p; unavailable/failed fits explicitly fall back to Welch on well means. Welch remains in the footer, with the run name and replicate counts. Descriptive/absolute-intensity endpoints say “descriptive, no test”. This updates the older raw-Welch headline convention; the Welch/Holm results remain in the workbook.
Colour key: two-arm WT `#595959`, QKI-KO `#D67AE5`; MIAT-KD `#E69F00`, MIAT-OE `#56B4E9` in the general condition map. The dedicated MIAT NT/KD ratio renderer uses `#595959`/`#CC79A7`. `group_colors` YAML overrides are supported; unassigned colours use the Okabe-Ito fallback.

Example: `fishsuite report --run "$run" --groups-file "$groups" --out "$out" --plot-style replicate-simple --technical-layer fov`

## Localization endpoints and DAPI census

Flag: `--deck-spec`; its YAML must set `dapi_localization: true`. There is no `--dapi-localization` switch. Deck preparation also requires a validated persisted panel and pinned selection/micrograph inputs.
Report-time DAPI QC leaves the segmentation of record unchanged. Classes are `in_retained_nucleus` (inside DAPI and retained mask), `in_unretained_dapi_object` (inside DAPI but outside retained mask), and `extranuclear` (outside DAPI, even if inside a retained mask).
The second class can include extensions of a DAPI object overlapping a retained nucleus; it does not mean every such spot belongs to a wholly dropped nucleus. Object-overlap columns make that distinction explicit.
Endpoints: `nuclear_spot_count_dapi`, `extranuclear_spot_count_per_cell_territory`, and `nuclear_spot_fraction_dapi` = nuclear / (nuclear + extranuclear assigned to that retained cell territory). Unretained-DAPI spots are excluded from this denominator; undefined denominators stay missing.
`DAPI census`, `DAPI census by arm`, `DAPI objects`, `DAPI spot classes`, and `DAPI class summary` record retained labels, unretained objects, area/overlap and puncta counts. DAPI-object reassignment sensitivity is recorded separately. Legacy mask-only localization endpoints become descriptive when correction is active.
`--qc-cyto-calls` additionally exports persisted RNA1 cytoplasmic calls as calibrated QC crops and CSV; this audit is distinct from the DAPI reclassification.

Example: `fishsuite report --run "$run" --groups-file "$groups" --out "$out" --deck-spec "$spec" --qc-cyto-calls`

## Statistical sensitivities

Flag: `--run` invokes the normal report statistics; there is no mixed-model, Student or FOV-outlier enable switch. `--alpha` sets the Welch/Holm significance level (default 0.05).
The mixed model uses REML, fixed arm, random well intercept and FOV-within-well variance component. Its two-sided asymptotic Wald-normal p is the headline when a fit succeeds; it is not a small-sample well t test. Fit status/warnings and defined nuclei/FOV/well counts remain available. The decision is labelled “2026-09-07 after data inspection”.
Contrasts also carry equal-variance Student t on well means and Welch on FOV means; FOV inference treats technical replicates as independent and is a sensitivity only.
`FOV outlier sensitivity` flags at most one maximum absolute leave-one-out z per well/endpoint, requiring at least three FOVs and strictly `|z| > 2.5` using the other FOVs' sample SD. Image-name ordering breaks ties; zero SD yields zero for identical values or signed infinity otherwise.
The sheet recomputes well means and Welch with flagged FOVs removed and shows before/after results. It does not change the main analysis or silently exclude fields.

Example: `fishsuite report --run "$run" --groups-file "$groups" --out "$out" --alpha 0.05`

## Persisted coloc panel and simple pixel metrics

Flags: `--existing-coloc FILE`, `--baseline-manifest FILE`; `--no-coloc-panel` suppresses automatic generation. Import takes precedence over automatic generation and never generates new nulls. It checks source hashes/cohort and rejects new peak gates, sampled filtering or field exclusions.
Simple metrics are enabled through `--deck-spec` YAML keys `simple_coloc_csv` and `simple_coloc_image_paths` (optional channel-label keys); there is no standalone simple-coloc flag. The imported workbook alone does not enable these extra slides.
Pearson r and Li ICQ are threshold-free nuclear-pixel measurements. Manders M1/M2 use only Costes-converged nuclei; failed fits remain missing. Historical Costes-with-run-threshold-fallback columns are mixtures and must not be labelled Costes-only. Values aggregate nucleus → FOV → well, with mixed-model and well-Welch statistics.
Lab floor distinction: the standard-panel script's `--call-threshold auto` uses a **90% biological-nucleus Costes-convergence floor** to choose its primary thresholded panel (batch below the floor, Costes-with-fallback otherwise). This does not change the converged-only definition of the simple Manders figures.
Acquisition/spot floors stay those recorded by the producing run; `--peak-floor rna=1000,rna2=1200` is post-detection gating, not a pixel-metric threshold, and is incompatible with frozen-panel import. The quickstart's recorded `--rna-min 400 --partner-min 1500 --dapi-min 334` are display floors, not universal analytical cutoffs.
Simple figures export independent Pearson/M1/M2/ICQ assets and full/focus pairs; representative cytofluorograms display persisted Costes thresholds and verify pixel Pearson against the stored value. They supplement spatial-null enrichment.

Example: `fishsuite report --run "$run" --groups-file "$groups" --out "$out" --existing-coloc "$panel" --baseline-manifest "$manifest" --deck-spec "$spec"`

## Preferential-colocalization ratio

Flag: `--miat-qki DIRECTORY` selects the persisted fixed-10 MIAT/QKI ratio report; optional `--miat-qki-count-report DIRECTORY` supplies separate COUNT localization, never a ratio denominator. This mode rejects deck export and regrouping.
Let T be total and A the associated measurement in matched wells: `rT = mean(T_KD)/mean(T_NT)`, `rA = mean(A_KD)/mean(A_NT)`, and **`R = rA/rT`**. It is a ratio of arm means, not a mean of per-well ratios or a loss ratio. R > 1 is the preferential-retention prediction; association is an operational spatial category, not bound molecules.
The marginal 95% CI is `exp(log(R) ± z_0.975 × SE_logR)` using the multivariate delta method with within-arm A/T covariance. It is not simultaneous and not inversion of the exact test. Invalid/nonpositive inputs produce missing inference with a reason.
MIAT tests enumerate all 400 within-slide balanced assignments (two slides, three NT/three KD wells each), using two-sided `abs(log(R))`, including ties, without a plus-one correction. Four null policies × count/intensity form eight exploratory tests with Holm8, distinct from the historical gate-nine.
`R_MDE_05` and `R_MDE_05_8` are plug-in normal 80%-power thresholds around R=1: `exp((z_(1-alpha/2) + z_0.8) × SE_logR)`, at alpha 0.05 and 0.05/8. Report `100 × (R_MDE-1)%`; MDE is neither an exclusion bound nor exact-permutation power.
Wording template: “Total retained rT=[value]; associated retained rA=[value]. Preferential spatial retention was [detected/not detected]; R=[value], marginal 95% log-delta CI [low, high], exact p=[value], exploratory Holm p=[value]. Minimum detectable retention at 80% power=[value]%. This is not evidence of equivalence, absence of binding, or molecular protection.” Select “detected” only for R>1 and Holm8 p<0.05; disclose any CI/exact-test disagreement.

Example: `fishsuite report --miat-qki "$data" --out "$out"`

RNASEH2B/BIN1 uses the same ratio machinery with KO/WT: T is matched-secondary-corrected nuclear mean; A is mean stored RNASEH2B signal per exact nuclear BIN1 footprint, without secondary subtraction on A. It uses three wells per arm and complete unblocked enumeration of 20 assignments (two-sided and R>1 tails), not MIAT's blocked/Holm8 design.
There is no dedicated `fishsuite report` RNASEH2B-ratio flag. The specialized module assembler exposes `--only rn`, `--run`, `--rn-prior`, `--rn-out`, `--miat-prior`, `--miat-out`; even unused MIAT paths are parser-required. It reads the prior validated delivery and its correction inputs.
Example: `python -m fishsuite.report.object_delivery --only rn --run "$run" --rn-prior "$prior" --rn-out "$out" --miat-prior "$prior" --miat-out "$out"`

## Deck export and slide assets

Flags: `--deck-spec FILE` prepares assets; `--deck` exports using optional python-pptx; `--micrograph-slides per-well` replaces the overview micrograph slide with matched well/FOV slides. `--deck` requires a spec, and the spec requires a persisted panel. `--no-figures` still permits explicitly requested deck assets.
A spec supplies an ordered `slides` list; there is no hard-coded universal numbered slide map. `deck_spec.resolved.yaml` is the resolved map, with workbook-traced values and hashed assets; `Deck specification`, `Slide values`, and `Figure sources` retain workbook lineage.
The RNASEH2B assembly maps count/size, localization, total IF, partner intensity at BIN1, partner puncta at BIN1, reverse anchor, pixel metrics/cytofluorograms, selection and micrographs. Added context slides cover preferential ratio, level-versus-enrichment scatter, radial profile, nearest-neighbour distances, secondary-only checks and cytoplasmic-call QC. Missing null-distance draws or acquisition metadata are stated, not synthesized.
Speaker notes start with the absolute workbook path, summarize the question/readout, arm means, mixed/Welch p and available adjustment/MDE, then “Levels of comparison” with nuclei/FOV/well counts and the post-inspection headline decision. Cell references belong in `slide_sources.csv`, not spoken prose.
`slide_sources.csv` records workbook/cell values and figure paths/hashes. Export validates workbook hashes, cell lineage, cohort and assets, rejecting untraced literal numbers or stale sources. Standard output deck name is `Sam_RNASEH2B_BIN1.pptx`.
Per-well micrographs reuse four independent native publication panels per arm, retaining saved LUTs/display windows and calibration; each panel is movable. FOV selection is nearest the within-well median retained-nucleus count, with image-name tie-breaking; wells pair by ordinal suffix. Selection records are saved in `Per-well micrograph selection`. Quantitative object-delivery charts are also separate assets, not one flattened whole-slide image.
**`deck_figures` limitation:** the current source guards this output name but does not populate it. Current slide preparation/assembly writes chart assets under `figures/`, with full/focus PNG/SVG alternatives. No `--deck-figures` flag exists; do not promise that directory as an automatic product.

Example: `fishsuite report --run "$run" --groups-file "$groups" --out "$out" --deck-spec "$spec" --deck --micrograph-slides per-well`

## conditions.groups and native-figures

Settings: `conditions.groups`, `conditions.group_order`, optional `conditions.group_colors` in run YAML. Example mapping: `groups: {WT: [WT_1, WT_2, WT_3], QKI-KO: [KO_1, KO_2, KO_3]}`. Well identities remain in source CSVs; secondary-only controls remain visible and never enter biological contrasts. Unlisted wells retain their own group labels.
New grouped runs now collapse native outputs by condition with equal weight per well: nucleus → FOV mean → well mean → condition mean, SD across wells. `figures/per_condition/` holds endpoint plots; `figures/00_overview/` holds combined panels; legacy well figures move to `figures/per_well_supplementary/`, while per-image figures stay available.
`analysis_summary.xlsx` gains `Per condition` and `Condition contrasts`, with per-well/per-image sheets retained. `per_condition_summary.csv` and `native_by_condition.json` record summaries, group settings, input hashes and figure inventory.
Post-hoc flags: `native-figures --run DIRECTORY --groups FILE --out DIRECTORY`. Here `--groups` is a YAML **file**, unlike report's repeatable `--groups NAME=wells`. It accepts a top-level conditions block or groups block; only grouping applies, not YAML peak floors or nucleus filters.
The regenerator reads finished CSVs without detection/segmentation and copies legacy figures into a fresh output. Default is `<run>/figures_by_condition_<timestamp>/`; existing destinations and overwriting original run/figures are refused. Frozen runs require a new external `--out`.

Example: `fishsuite native-figures --run "$run" --groups "$groups" --out "$out"`

## Frozen-delivery guard and release markers

Flag: `--out DIRECTORY` selects a fresh destination; there is no bypass flag.
The report guard checks the destination and ancestors, both lexical and resolved paths (including junctions). A `DELIVERY_*` directory is frozen when it contains `MANIFEST_SHA256.tsv`, `CHECKSUMS.sha256`, `CHECKSUMS.txt`, or an entry prefixed `SUPERSEDED_BY`, `CHECKSUMS`, or `MANIFEST_SHA256` (including `CHECKSUMS_*` ledgers).
The `DELIVERY_` name alone does not freeze an empty staging folder. A released directory cannot be overwritten through a child or a source-run junction. Release manifests/sealing are a separate packaging step, not automatically created by `report`.
The native regenerator has a separate run/ancestor check for files matching `*manifest*`, `*ledger*`, `REPORT_LOCK.json`, `*frozen*`; do not assume its predicate is identical to the report delivery guard. Use a new external output for released runs.

Example: `fishsuite report --run "$run" --groups-file "$groups" --out "$out"`

## Provenance: reporter versus producing commit

Flags: `--engine-repo DIRECTORY`, `--preset FILE`; provenance is otherwise automatic.
`reporter commit` is the checkout executing the report. `producing engine commit` comes only from a valid 40-hex commit recorded in the source run's `versions.txt` under `producing engine commit`, `engine_commit` or `git_commit`; absent identity is explicitly “missing from run”. Current HEAD is never substituted.
`--engine-repo` records inspected HEAD as “not producing identity”; `--preset` records its path and MD5. `Run provenance` and report `versions.txt` distinguish reporter/producer. `SOURCE_RUN.md`, hashes and copied records under `provenance/source_run/` trace the inputs; external reports also create a source-run junction (shortcut/text fallback).

Example: `fishsuite report --run "$run" --groups-file "$groups" --out "$out" --engine-repo "$repo" --preset "$preset"`
