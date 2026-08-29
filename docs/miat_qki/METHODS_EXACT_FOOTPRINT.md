# Methods: MIAT × QKI single-plane exact-footprint analysis

## Experimental comparison and analysis unit

The analyzed material was undifferentiated H9 hESCs imaged after non-targeting control (NT) or MIAT knockdown (KD). The retained biological roster comprised 37 fields of view (FOVs) in 12 independent biological sets: six NT and six KD. Each slide contained three NT sets and three KD sets. Within-arm replicate numbers were identifiers only and did not define NT–KD pairs. S1_KD_2 contained four FOVs; every other biological set contained three FOVs. Seven dual-omission FOVs, in which the MIAT probe and QKI primary antibody were omitted, were processed as QC controls and excluded from biological inference.

The primary inferential unit was the biological set. Nuclei and FOVs were nested technical observations and were not treated as independent biological replicates.

## Channels and image geometry

Resolved channel indices were:

- channel 0: MIAT-647 RNA-FISH;
- channel 1: QKI-568 immunofluorescence;
- channel 2: DAPI.

The 44 retained stacks were acquired on 2026-08-26 with a Hamamatsu ORCA-Fusion #1 detector and a recorded 100×/NA 1.5 objective. Images were 2,304 × 2,304 uint16 with 1×1 binning and 65 nm/pixel XY sampling. Stacks contained 39–54 planes; 28 had approximately 0.21 µm z spacing and 16 had approximately 0.30 µm spacing. Recorded channel names and exposures were C0 `640 CSU` MIAT-647 at 0.7 s, C1 `561 CSU` QKI-568 at 0.7 s, and C2 `405 CSU` DAPI at 0.5 s. These strings are recorded channel names, not recovered excitation wavelengths. All quantitative intensity values are raw detector units unless expressed as a within-nucleus ratio.

Laser powers were experimenter-reported as held constant, but numeric settings and metadata confirmation were not recovered. The microscope stand/model, CSU scan-head model, acquisition software/version, and several staining and filter details also remain missing. QKI enrichment relative to the parent-nucleus mean provides the within-image normalized companion to raw QKI intensity.

## Single-plane selection: no projection

Each stack received an independently algorithmically proposed plane that was visually reviewed and accepted; the retained audit records zero manual overrides. Each selected plane was encoded as a file-specific override with identical start and end slices. DAPI, MIAT, and QKI were all read from that same selected plane. No maximum-intensity projection, small projection, or multi-plane collapse was used for segmentation, spot detection, exact-footprint quantitation, colocalization, or the final publication micrographs described here.

The retained quantitation-invariant audit passed all of the following checks across the 44-image roster:

- one selected Z per image;
- one selected Z per nucleus;
- spot and nucleus Z agreement;
- channel lock within image;
- expected channel indices (MIAT 0, QKI 1, DAPI 2);
- `selected_z_0based = selected_z_1based − 1`;
- quantitation plane labeled `exact_recorded_single_z`;
- no projection/MIP contradiction.

The six final publication examples were rendered from the exact selected planes used for their quantitative rows. Their selected 1-based Z planes were NT examples 1–3: Z22, Z12, and Z16; MIAT-KD examples 1–3: Z13, Z24, and Z22. Exact image, biological-set, slide, and crop identities remain in the micrograph manifest and source-data audit rather than in visible figure labels.

## Nucleus segmentation and equalized sampling

DAPI nuclei were segmented with Cellpose 4.1.1 using the `cpsam` model on DirectML. The resolved configuration used a 200-pixel diameter, four-fold downsampling, flow threshold 0.4, cell-probability threshold 0.0, a minimum nuclear area of 16,000 pixels, and exclusion of objects within 20 pixels of the image border.

After geometry-based QC, 10 eligible nuclei per FOV were selected at random without replacement. Sampling used NumPy PCG64 with global seed 0 and a separate deterministic generator per image keyed to the image path. Selection used DAPI and nuclear geometry only; MIAT and QKI values did not enter the selection. The run was configured to fail if an FOV contained fewer than 10 eligible nuclei.

The primary sample therefore contained 370 biological nuclei (10 from each of 37 biological FOVs) and 70 QC nuclei (10 from each of seven controls). The complete nucleus roster retained all 1,682 segmented nuclei with eligibility and sampling flags. A post-run authoritative nucleus-key join corrected 17,559 stale spot-level sampling flags; no spot keys were missing from the nucleus authority and no eligibility mismatches remained. The corrected primary biological sample contained 6,267 MIAT spots.

## MIAT spot detection

MIAT spots were detected on the selected channel-0 plane with Big-FISH 0.6.2/LoG using the resolved FishSuite `rna_protein` configuration. The voxel size was 65 nm in XY, the nominal spot radius was 120 nm, only nuclear spots were retained, and the minimum MIAT peak-intensity floor was 364 raw units. QKI was treated as a diffuse intensity field (`detect_antibody_spots: false`); QKI puncta were not detected or paired.

The exact-footprint post-run tables contain 23,829 detected MIAT spots. All 23,829 passed footprint construction with `half_max_component`; no fitted-disk fallback was used in the retained dataset.

## Exact detected MIAT footprint

For each detected MIAT spot, the stored center was rounded to the nearest pixel. Within a local window, the brightest pixel in a 3 × 3 center search was used as the seed. Local background was the 10th percentile of the MIAT window. The footprint threshold was

`background + 0.5 × (peak − background)`.

An 8-connected component was grown from the seed through pixels meeting that threshold. The resulting irregular raster component was the exact detected MIAT footprint. Its original pixel offsets relative to the spot center were persisted. QKI mean, sum, median, standard deviation, minimum, and maximum were measured over those exact pixels; the primary local-QKI metric did not use a disk around the MIAT center.

If a half-maximum connected component cannot be constructed, the implementation fails closed by recording an explicitly invalid empty footprint with zero exact pixels. Such a spot remains part of the all-detected MIAT spot count but contributes no exact-footprint pixels or MIAT spot-pixel intensity and is excluded from exact-footprint nulls. No disk or halo is synthesized. All 23,829 retained production spots used the half-maximum connected component. Radius-based QKI disk and annulus values were saved only as non-primary sensitivity fields.

## Same-nucleus KEEP-N exact-footprint null

The association null preserved the number of detected MIAT spots in each nucleus (“KEEP-N”) and kept every spot's exact raster footprint unchanged. For each requested null iteration, one initial angle was applied to the full MIAT spot-center constellation around its centroid. Each footprint was translated to its rotated center using its stored pixel offsets; the irregular raster footprint itself was not rotated, resized, or resampled.

Placements were restricted to finite QKI pixels in the same parent nucleus and outside the recomputed nucleolus. A placement was valid only if every pixel in the exact footprint remained inside the valid nucleoplasm with unchanged unique-pixel cardinality. Invalid spot placements were redrawn independently, up to 1,000 attempts, rather than clipped or replaced with observed pixels.

The production parameters were:

- 1,000 null draws per usable spot;
- NumPy random seed 0;
- median first-pass constellation retention gate ≥ 0.5;
- all 1,000 requested valid draws required for a spot by the primary `min_valid_draw_fraction = 1.0` gate;
- at least two MIAT spots required in a nucleus for rotation;
- maximum 1,000 redraw attempts for an invalid placement.

Spots failing the geometry, nucleolar, minimum-spot, first-pass-retention, finite-signal, or complete-draw requirements were labeled unusable and kept as a separate population. They were not silently reclassified as negative.

## q90/q95/q99 association calls

For each null-usable MIAT spot, the linear q90, q95, and q99 quantiles were calculated from that spot's own 1,000 QKI-null values. A spot was called positive only when its observed mean QKI intensity over the exact MIAT footprint was **strictly greater than** its own null quantile. q95 was prespecified as primary; q90 and q99 were sensitivity thresholds.

The two association-fraction denominators were defined separately:

- `association_fraction_among_usable_q95` = q95-positive spots / (q95-positive + q95-negative spots);
- `association_fraction_among_all_floor_spots_q95` = q95-positive spots / all MIAT floor-passing spots, including null-unusable spots in the denominator.

Null usability itself was reported as (positive + negative) / all floor-passing spots. This explicit reconciliation ensured that every floor-passing spot was positive, negative, or unusable at each threshold.

## Counts and MIAT spot-pixel intensity

Spot count endpoints were calculated per nucleus, retaining sampled or eligible zero-spot nuclei through a nucleus-roster left join.

MIAT and QKI spot-pixel intensity was summarized per nucleus over the exact detected MIAT pixels in two forms:

- `spot_summed`: the footprint-pixel sum was calculated for each spot and then summed; shared pixels contribute once for each overlapping spot footprint;
- `union_deduplicated`: the union of selected footprint pixels was formed within each nucleus and each shared image pixel contributed once.

Global intensity used all detected MIAT spots and their detected pixels. Threshold-associated intensity used only q95-positive spots for the primary analysis; q90 and q99 were sensitivity thresholds. Whole-nucleus and nucleoplasm raw sums/means were retained as separate diffuse-signal endpoints and were not substituted for puncta-specific spot-pixel intensity.

QKI enrichment at a MIAT footprint was the QKI mean over the exact footprint divided by the mean QKI intensity of the parent nucleus (or, in separately named columns, the parent nucleoplasm).

## Aggregation hierarchy

For every endpoint and cohort:

1. nuclei contributed equally to the FOV mean;
2. finite FOV means contributed equally to the biological-set mean;
3. the 12 independent biological-set means (six NT, six KD) entered inference.

This prevents the fourth S1_KD_2 FOV from giving that biological set extra weight and avoids nucleus-level pseudoreplication. Controls were excluded before biological aggregation.

The equalized random sample (`sampled_primary`) was primary. Recalculation across all DAPI-eligible nuclei (`all_eligible`) was retained as a sampling sensitivity analysis.

## Statistical inference

The primary two-sided test was an exact within-slide label permutation of independent biological-set means. Within each slide, the observed number of KD labels was retained. With three NT and three KD sets on each of two slides, the full primary space contained `choose(6,3) × choose(6,3) = 400` labelings. The exact p-value was the fraction of all labelings whose absolute KD-minus-NT mean difference was at least as large as the observed absolute difference.

The retained inference table also reports:

- an unpaired Welch t-test on biological-set means;
- an OLS model with condition plus slide as a fixed blocking factor;
- the exact within-slide permutation result, used as the primary reported p-value.

The S1_KD_1 sensitivity excluded that entire biological set. It left three NT and two KD sets on slide 1 and three NT and three KD sets on slide 2, yielding exactly 200 permitted within-slide labelings. No individual FOV, nucleus, or spot was selectively removed.

Within-slide 3 × 3 NT-by-KD Cartesian contrasts were saved for descriptive inspection only. They do not define pairs and contain no inferential p-values.

Ratio-of-ratios comparisons used independent arm means with a multivariate delta method that retained the within-arm covariance between numerator and denominator. q95 comparisons were prespecified for associated spot count/floor-passing spot count, associated union-deduplicated MIAT spot-pixel intensity/floor-passing union-deduplicated MIAT spot-pixel intensity, and associated spot-summed MIAT spot-pixel intensity/floor-passing spot-summed MIAT spot-pixel intensity. Equivalent q90 comparisons were sensitivity analyses. q99 RoRs were not forced in the production specification. These retained RoRs use the prespecified floor-passing denominator; the separate Global MIAT summaries use all detected MIAT spots.

## Continuous MIAT–QKI correlations

Continuous correlations were computed within each biological FOV across individual MIAT spot rows with a fully valid exact footprint. Two measurement pairs were retained:

- raw MIAT exact-footprint mean versus raw QKI mean over the same pixels;
- MIAT footprint enrichment versus its parent-nucleus mean versus QKI footprint enrichment versus its parent-nucleus mean.

Pearson and Spearman correlations were calculated per FOV, transformed with Fisher's z, and averaged equally across finite FOVs within each biological set. All-exact-valid-spot correlations entered the same independent-set inference workflow described above. Correlations restricted to q90-, q95-, or q99-positive spots were retained as conditional/descriptive only because QKI intensity defines the threshold-selected population; no inferential p-values were assigned to these conditional subsets.

## Publication rendering

Publication images were generated from the same selected single Z planes used for quantitation. Rendering was display-only and did not feed any measurement. The final primary display windows in `FINAL_PUBLICATION_20260828-231545` were DAPI 334–8,000, MIAT 400–4,000, and QKI 555–3,000. There was no projection in any final publication panel.

## Validation and provenance

The exact-footprint backfill completed 44/44 images (37 biological and seven controls), with 23,829 spots. The full historical parity gate passed for spot area and QKI values before null calculation. Population reconciliation, input immutability, source checksums, selected-plane parity, channel lock, and quantitation-plane invariants all passed.

The source FishSuite run was:

`F:\Image Analysis Work\MIAT_QKI_Coloc_2026_08_25\05_RUNS\MIAT_QKI_COLOC_2026-08-28_0214`

The exact-footprint backfill was:

`F:\Image Analysis Work\MIAT_QKI_Coloc_2026_08_25\06_ANALYSIS\EXACT_FOOTPRINT_BACKFILL_20260828-134838`

The retained post-run statistics were:

`F:\Image Analysis Work\MIAT_QKI_Coloc_2026_08_25\06_ANALYSIS\EXACT_FOOTPRINT_POSTRUN_20260828-155508`

The analysis code was generated in the isolated worktree `E:\Claude\fishsuite-codex-miat-qki-footprint` on branch `codex/miat-qki-exact-footprint`, based on commit `5d9870bc717741be7578090e27caa0af6ccd8195`. Production module hashes are recorded in `software_provenance.json`.

Recorded software versions were FishSuite 1.0.0, Python 3.10.20, NumPy 1.26.4, SciPy 1.15.3, pandas 2.3.3, scikit-image 0.25.2, Cellpose 4.1.1, Big-FISH 0.6.2, bioio 3.3.0, bioio-bioformats 2.0.0, h5py 3.16.0, and tifffile 2025.5.10. The exact-footprint backfill used Java 11 through the recorded Zulu JRE path.

## Assay boundary

This method quantifies spatial co-occurrence of QKI intensity with exact MIAT spot pixels on a single optical plane. It is a **spatial/pseudo-binding measurement, not a direct-binding measurement**. It does not measure molecular contact, occupancy, affinity, or stoichiometry.
