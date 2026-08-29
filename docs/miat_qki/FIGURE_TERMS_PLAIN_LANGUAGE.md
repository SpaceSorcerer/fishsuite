# MIAT-QKI figure terms in plain language

These definitions are the reader-facing terms for the MIAT-QKI figures and source-data summaries. They keep the displayed endpoints close to what was measured while limiting interpretation to spatial association.

## MIAT endpoints

- **All MIAT spots per nucleus**: the total number of detected MIAT puncta in each nucleus.
- **MIAT spot-pixel intensity**: the summed raw MIAT fluorescence over detected MIAT spot pixels.
- **Overlapping pixels counted once**: when detected MIAT spot footprints overlap in an aggregate intensity endpoint, the overlapping pixels are counted once by taking their union rather than counting them again for each spot.

## q95 QKI association

**q95** labels a MIAT spot as QKI-associated when QKI within that spot's detected MIAT pixels is greater than 95% of 1,000 same-nucleus KEEP-N randomized same-shape placements. It is a spatial colocalization or pseudo-binding measure, not evidence that MIAT and QKI bind directly.

Every q95-associated fraction is reported with both denominators:

- **Among null-usable spots**: the number of q95-positive spots divided by spots with a usable same-nucleus randomized-placement comparison.
- **Among all floor-passing spots**: the conservative number of q95-positive spots divided by every MIAT spot that passed the fluorescence floor, including spots without a usable randomized-placement comparison.

## What the association does and does not mean

Spatial colocalization here means that measured QKI signal is enriched at the detected MIAT spot pixels relative to the same-nucleus randomized placements. It is a pseudo-binding assay that does not establish direct binding. It also does not prove or disprove the sponge model.

| Interpretive question | Conclusion | Scope |
| --- | --- | --- |
| Direct molecular binding | Not established by this assay | Spatial colocalization or pseudo-binding only |
| MIAT sponge mechanism | Neither proved nor disproved | This assay does not resolve the mechanism |

## Global-versus-associated depletion

**Global-versus-associated depletion** compares the MIAT decrease across all detected MIAT spots with the decrease in the q95-associated subset. **Global MIAT depletion** describes the all-MIAT endpoint. **q95-associated MIAT depletion** describes the subset that meets the q95 spatial-association criterion. Comparing the two asks whether the QKI-associated subset changes differently from the overall detected MIAT population; it does not convert spatial association into a direct-binding claim.
