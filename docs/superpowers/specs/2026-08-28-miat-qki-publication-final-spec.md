# MIAT–QKI Final Publication Outputs Specification

## Scope

Build a fresh, non-overwriting final-publication output set from the retained MIAT–QKI exact-footprint post-run tables and exact selected optical planes. Do not re-detect spots, re-segment nuclei, change z selections, edit raw data, or modify Claude memory/configuration or any `file_map.md`.

## Scientific invariants

- Human H9 hESC NT and MIAT-KD are independent groups with six biological sets per arm; there is no NT1-to-KD1 pairing.
- One reviewed optical plane is used per image. DAPI, MIAT, and QKI must all come from that same recorded plane for segmentation, quantification, overlays, and representative micrographs.
- No MIP or other projection may enter any MIAT–QKI colocalization endpoint or representative colocalization micrograph.
- QKI is measured only inside the actual detected MIAT spot pixels. No disk, halo, radius expansion, or antibody spot detection is allowed.
- The primary q95 call means that QKI within a spot's exact MIAT footprint is greater than 95% of 1,000 same-nucleus KEEP-N randomized same-shape placements.
- Continuous QKI-at-MIAT measurements include all exact-valid spots; thresholded measurements include the q95-positive subset.
- Primary inference remains the retained exact two-sided within-slide label permutation on equal-weight biological-set values. Slide is a statistical blocking factor only; it is not a visible figure grouping.
- Report both q95 denominators: among null-usable spots and conservatively among all floor-passing spots.
- Describe the assay as spatial colocalization/QKI association or a pseudo-binding assay. Do not claim direct binding and do not claim the results prove or disprove the sponge model.

## Reader-facing terminology

- Use **All MIAT spots per nucleus** for the total detected MIAT puncta endpoint.
- Use **MIAT spot-pixel intensity** for summed raw MIAT fluorescence over detected MIAT spot pixels.
- When overlapping spot footprints are deduplicated, say **overlapping pixels counted once**.
- Do not use reader-facing phrases `exact-footprint mass`, `global NEAT spots`, or unexplained `q95`.
- Every q95 statistical figure must define q95 in its subtitle, caption, footer, or adjacent note.

## Statistical figures

Create a clean final family containing:

1. Total MIAT knockdown: all MIAT spots per nucleus and MIAT spot-pixel intensity across all detected spot pixels.
2. Base QKI association: q95-associated fraction in NT versus MIAT-KD for both usable and all-floor denominators.
3. QKI-associated MIAT depletion: q95-associated MIAT spots per nucleus and q95-associated MIAT spot-pixel intensity.
4. Global versus QKI-associated depletion summary: KD/NT percent remaining for counts and union-deduplicated spot-pixel intensity, plus the retained ratio-of-ratios estimates and uncertainty/inference.

Plots must retain the nucleus → equal-weight FOV → equal-weight biological-set hierarchy. All six set means per arm must use one common marker and no slide-coded shapes, slide legend, slide name, or slide-specific color. The source-data/statistics tables and manifest must retain slide fields and identify the within-slide inference. Do not draw NT-to-KD connecting lines.

New final figures must be 600-dpi PNG and editable SVG only. Do not emit PDF files or list PDF paths in the final manifest.

## Representative micrographs

Produce several representative micrographs: three NT and three MIAT-KD examples, deterministically selected from distinct biological sets where sufficient valid candidates exist. Selection must use recorded QC/parity evidence and condition-relative centrality, not hand-picked biological effect size. Controls are ineligible.

For each example, show the exact analyzed single-z plane with DAPI, MIAT, QKI, merge, and q95 association overlay. The q95 overlay must trace exact detected MIAT footprint pixels and distinguish q95-positive, q95-negative, and unusable spots without substituting disks. Visible row labels may identify only condition and example number; slide names and biological-set identifiers belong in the audit/manifest, not the image.

Create individual representative panels plus clean multi-example contact sheets. New publication outputs are PNG and SVG only. Exact z, image, nucleus/crop, biological set, slide, source hashes, display windows, and selection rationale must remain in source-data/audit files.

Primary display windows are DAPI 334–8000, MIAT 400–4000, and QKI 555–3000. These are display settings only and must never alter quantitation.

## Microscopy and acquisition methods

Create `MICROSCOPY_ACQUISITION_METHODS.md` with:

- a short manuscript-ready paragraph first;
- a concise detailed acquisition/provenance section;
- an explicit single-z colocalization versus narrow-projection sensitivity distinction;
- an evidence table separating metadata-confirmed, experimenter-reported, and missing details;
- a short list of manuscript-critical missing acquisition/staining details for later user completion.

Use the all-44 VSI/OEX audit evidence: acquisition 2026-08-26; 44 retained stacks (18 NT, 19 KD, 7 dual-omission controls); Hamamatsu ORCA-Fusion #1 detector; recorded 100×/NA 1.5 objective; 2304×2304 uint16, 1×1 binning, 0.065 µm XY; 39–54 planes; 28 stacks at about 0.21 µm z spacing and 16 at about 0.30 µm; C0 `640 CSU` MIAT-647 at 0.7 s, C1 `561 CSU` QKI-568 at 0.7 s, C2 `405 CSU` DAPI at 0.5 s. Treat channel numbers as recorded channel names, not recovered excitation wavelengths.

Laser powers, microscope stand/model, CSU scan-head model, acquisition software, and several staining/filter details are not recovered. State that laser powers were experimenter-reported as held constant, while numeric settings and metadata confirmation are unavailable. Do not invent missing details. Controls omitted the MIAT probe and QKI primary antibody; do not call both omissions antibodies.

## Output and packaging

- Create a new timestamped final-publication directory under the retained exact-footprint post-run directory. Never overwrite V2/V3 or earlier workbook/output directories.
- Copy the three independently verified companion workbooks into the final delivery package.
- Preserve canonical large CSV/HDF5 tables by reference and hash rather than duplicating them unnecessarily.
- Update the standalone Claude-discoverable handoff file, but do not edit Claude memory/core/configuration.
- Final delivery selection contains PNG and SVG figures, no newly copied PDFs.
- Run numerical, structural, visual, source-hash, workbook, manifest, and code tests before any completion claim.

