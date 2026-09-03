# Engine change 2026-09-03 — per-channel `threshold_override` + partner-anchored rotation null

Two additive, default-off changes. With both unset the engine reproduces the
pre-change output exactly; verified on real data, not asserted (see
`F:\Image Analysis Work\RNASEH2B_BIN1introns_2026_08_25\12a_ENGINE_REPRO_2026-09-03\REPRO_RESULT.md`).

---

## (a) Per-channel fixed BigFISH LoG threshold

**Problem.** `foci.threshold_override` was shared across channels, and
`_detect` read it directly from the shared config for both. Pinning a threshold
for the antibody channel therefore also pinned the RNA channel.

**Change.** `FociChannelOverrideCfg` gains `threshold_override: Optional[float] = None`.
`FociCfg.resolved_for(channel)` returns it as `params["threshold_override"]`,
falling back to the shared `FociCfg.threshold_override` when the channel value
is None. `rna_rna._detect` reads `params["threshold_override"]` instead of the
shared value.

Resolution order per channel: channel override -> shared value -> `None`
(per-image auto threshold). An unset channel override reproduces the previous
behaviour exactly, which is why the shared-only path is untouched.

`rna_protein` needs no change: its config shim already copies
`foci.antibody_overrides` into `foci.rna2_overrides`, so an antibody override
reaches the detector through `resolved_for("rna2")`.

**Motivation.** The per-image BigFISH LoG threshold collapses on a blank
secondary-only field, so a fixed antibody threshold is sometimes required while
the RNA channel must keep its per-image auto threshold.

## (b) Partner-anchored rotation null

**Change.** New master gate `FociCfg.compute_partner_anchored_rotation_null:
bool = False`, requiring `compute_partner_intensity` **and**
`compute_partner_rotation_null`. When the flag is on but a prerequisite is off
the engine prints an explicit NOTE and emits nothing, rather than silently
producing no columns.

It calls the **existing** `_rotation_null_for_nucleus` with the (field,
constellation) pair swapped: constellation = this nucleus's rna2 / antibody
spots, sampled field = the rna1 plane. Disk radius, `partner_rotation_n`, seed
root, `partner_rotation_min_retention` and `partner_rotation_assoc_percentile`
are shared with the rna1-anchored null; only the RNG stream differs
(`partner_rotation_seed + 303` for rotation, `+ 606` for the association
distribution), so turning it on leaves the rna1-anchored draws bit-for-bit
unchanged. Nucleolus exclusion is applied to the partner constellation by the
same rule the rna1-anchored null applies to its own.

The block sits as a **sibling** of the rna1-spot branch, not nested inside it,
so a nucleus with partner spots but no rna1 spots still yields a value. This
required hoisting the per-nucleus sampling-mask computation (`_samp_mask`,
`_nucleolus_in_nuc`, `_nys`, `_nxs`) up one level. That computation depends only
on `nuc_mask`, `nid` and the precomputed nucleolus labels, so the values the
rna1-anchored path sees are unchanged; the real-data comparison confirms it.

Values are NaN for any nucleus with no partner spots, which is every nucleus
when `detect_antibody_spots: false`.

### Emitted columns

Per nucleus (`nuclei_metrics.csv`):

- `rna1_rotation_enrichment_at_rna2_spots`
- `rna1_rotation_null_z_at_rna2_spots`
- `rna1_rotation_null_p_at_rna2_spots`
- `rna1_rotation_assoc_fraction_at_rna2_spots`
- `rotation_null_usable_at_rna2_spots`

Per image (`per_image_summary.csv`), spot-count-weighted over usable nuclei:

- `rna1_pooled_rotation_enrichment_at_rna2_spots`
- `rna1_pooled_rotation_null_z_at_rna2_spots`
- `rna1_pooled_rotation_null_p_empirical_at_rna2_spots`
- `rna1_pooled_rotation_obs_at_rna2_spots`
- `rna1_pooled_rotation_null_mean_at_rna2_spots`
- `rna1_mean_rotation_assoc_fraction_at_rna2_spots`
- `n_nuclei_partner_rotation_null_at_rna2_spots`

Every name carries the `rna2` token only where it refers to the partner, so
`rna_protein`'s substring relabel turns them into `..._at_protein_spots` while
the leading `rna1` survives. Verified in the run output, not only in a unit test.

### Reading the numbers

`rna1_pooled_rotation_null_p_empirical_at_rna2_spots` has an arithmetic floor of
`1 / (partner_rotation_n + 1)`; at the default 1000 that is 0.000999 and it
cannot go lower. `rna1_rotation_assoc_fraction_at_rna2_spots` has a chance floor
of `1 - partner_rotation_assoc_percentile/100`, i.e. 0.05 at the default 95.0.

---

## Verification

| check | result |
|---|---|
| `tests/test_partner_anchored_rotation_null.py`, 27 tests | 27 passed, 7.53 s |
| existing suite, 34 files | 471 passed, 0 failed, 2211 s |
| unset path vs pristine `HEAD` engine, 2 real images | 484 columns compared, **0 differ** |
| new flags on, per-channel threshold scoping | antibody 17.0/16.0 -> 15.0, RNA 37.0/30.0 unchanged |
| new columns populated | 41/41 nuclei, both images |

The key unit test spies on `_rotation_null_for_nucleus`, captures the arguments
of the partner-anchored call, and reproduces the engine's emitted per-nucleus
enrichment by re-invoking that same function with the captured arrays and a
fresh RNG at `partner_rotation_seed + 303`. That is what makes "the new call is
the old function, not a re-implementation" a tested claim.

## Known issue found while verifying, NOT caused by this change

Run 11 (`11_FULL_CPSAMV2_OTSU_2026-09-01`) does **not** reproduce from this
repository: it gives 36 nuclei for the two test images where both the modified
and the pristine `HEAD` engine give 41. Its preset sets
`nuclei.cellpose_preclip_dapi_otsu: true`, a per-image DAPI Otsu floor applied
before Cellpose inference. That key exists nowhere in this repository's `src/`
or in `git log -S` history, and `NucleiCfg` silently ignores unknown keys
(pydantic default `extra='ignore'`), so the preset runs without the feature and
without a warning.

The engine that produced run 11 is the stage-09 proposal worktree,
`F:\Image Analysis Work\RNASEH2B_BIN1introns_2026_08_25\09_CODE_PROPOSAL_FISHSUITE_DAPI_FLOOR_2026-09-01\src\fishsuite\\`
(`config\schema.py:350`). It is unmerged and small: 12 lines in
`core\modes\rna_rna.py` and 4 in `config\schema.py` against `HEAD`. Merging it
is out of scope for this change and was not attempted, but a run needing both
the pre-clip and the two features added here will require it. Spot detection,
the path this change touches, matches run 11 exactly. Details in
`REPRO_RESULT.md`.

## Files changed

| path | change |
|---|---|
| `E:\Claude\fishsuite\src\fishsuite\config\schema.py` | `FociChannelOverrideCfg.threshold_override`; `resolved_for` returns it with shared-value fallback; new `FociCfg.compute_partner_anchored_rotation_null`. |
| `E:\Claude\fishsuite\src\fishsuite\core\modes\rna_rna.py` | `_detect` reads the per-channel threshold; new gate + RNG streams + accumulators; hoisted sampling mask; partner-anchored null block; 5 per-nucleus and 7 per-image columns in both the populated and empty-image branches. |
| `E:\Claude\fishsuite\tests\test_partner_anchored_rotation_null.py` | NEW. 27 tests: schema round-trip, relabel, default-off byte-equivalence, determinism, same-function proof, end-to-end threshold scoping. |
| `E:\Claude\fishsuite\README.md` | Documents both knobs. **`PRESETS.md` does not exist in this repository**; the FociCfg field reference and the null documentation live in `README.md`, so the documentation went there. Flagged for the orchestrator. |
| `E:\Claude\fishsuite\docs\ENGINE_CHANGE_2026-09-03_...md` | NEW. This note. |

`E:\Claude\fishsuite\file_map.md` was deliberately **not** edited. Proposed
entries for whoever owns it:

- `src/fishsuite/config/schema.py` — append: *2026-09-03 per-channel LoG
  threshold + partner-anchored rotation gate: `FociChannelOverrideCfg` gains
  `threshold_override` (None inherits the shared `FociCfg.threshold_override`),
  surfaced through `resolved_for`; `FociCfg` gains default-off
  `compute_partner_anchored_rotation_null`.*
- `src/fishsuite/core/modes/rna_rna.py` — append: *2026-09-03 `_detect` reads the
  per-channel `threshold_override`; new default-off partner-anchored rotation
  null (constellation = rna2/antibody spots, field = rna1 plane) reusing
  `_rotation_null_for_nucleus` on RNG streams seed+303 / seed+606, emitting 5
  per-nucleus and 7 per-image `*_at_rna2_spots` columns that relabel to
  `*_at_protein_spots` in `rna_protein`.*
- `tests/test_partner_anchored_rotation_null.py` — NEW: *27 GPU-free tests for
  the per-channel `threshold_override` and the partner-anchored rotation null,
  including a spy test proving the new path calls the existing
  `_rotation_null_for_nucleus` with the (field, constellation) pair swapped.*
- `docs/ENGINE_CHANGE_2026-09-03_per_channel_threshold_and_partner_anchored_rotation_null.md`
  — NEW: *this change note, with the verification table and the run-11
  non-reproducibility finding.*
