# Engine change 2026-09-04 — DAPI-plane reference for the partner-anchored null, in-nucleus anchors, per-image RNA pedestal normalisation

Three additive, default-off changes. With all three unset the engine reproduces
the pre-change output exactly; verified on real data, not asserted (see
[Verification](#verification)).

They exist because the QKI × BIN1-intron arm-2 adversarial review
(`F:\Image Analysis Work\QKI_BIN1introns_2026_08_25\06_REPORT_2026-09-04\arm2_strictQKIspots_2026-09-04_0648\ADVERSARIAL_REVIEW.md`)
and the T36/T45/T60 sensitivity analysis
(`...\06_REPORT_2026-09-04\SENSITIVITY_T36_T45_T60.md`) each identified a
specific missing capability rather than a bug.

---

## (a) `foci.partner_anchored_null_sample_field` — a signal-free reference

**Problem.** The partner-anchored rotation null returned WT enrichment 1.102 at
QKI puncta and was tested against a fixed null value of 1.0. The same function,
on the same nuclei, with the same sampling mask and disk, returns 1.16–1.20 on
the antigen-free 640 channel in QKI-KO. A value of ~1.10 therefore sits inside
the range this machinery produces from a channel with no specific signal, and
1.0 is not what it returns under the null.

**Change.** `FociCfg.partner_anchored_null_sample_field: Literal["rna", "dapi",
"both"] = "rna"`. `"dapi"` runs the same `_rotation_null_for_nucleus` on the
same partner constellation but samples the **DAPI plane**. DAPI carries no
RNA-probe signal, so what comes back is what intranuclear texture alone
produces — the empirical null the fixed 1.0 was standing in for. `"both"` emits
both families in one pass.

The implementation is a loop over `_pa_prefixes`; every per-nucleus, pooled and
per-image block is written once and indexed by prefix, so adding a field is one
list entry rather than a second copy of the computation.

RNG streams are disjoint by construction: rna1-anchored `seed+101 / +404`,
partner-anchored rna1-sampled `seed+303 / +606`, partner-anchored DAPI-sampled
`seed+909 / +1212`. A `"both"` run therefore reproduces a `"rna"`-only run's
rna1 numbers bit-for-bit, which is a test, not a claim.

### Emitted columns

Per nucleus (`nuclei_metrics.csv`), `<field>` ∈ {`rna1`, `dapi`}:

- `<field>_rotation_enrichment_at_rna2_spots`
- `<field>_rotation_null_z_at_rna2_spots`
- `<field>_rotation_null_p_at_rna2_spots`
- `<field>_rotation_assoc_fraction_at_rna2_spots`
- `rotation_null_usable_at_rna2_spots` (rna1) / `rotation_null_usable_dapi_at_rna2_spots` (dapi)

Per image (`per_image_summary.csv`), spot-count-weighted over usable nuclei:

- `<field>_pooled_rotation_enrichment_at_rna2_spots`
- `<field>_pooled_rotation_null_z_at_rna2_spots`
- `<field>_pooled_rotation_null_p_empirical_at_rna2_spots`
- `<field>_pooled_rotation_obs_at_rna2_spots`
- `<field>_pooled_rotation_null_mean_at_rna2_spots`
- `<field>_mean_rotation_assoc_fraction_at_rna2_spots`
- `n_nuclei_partner_rotation_null_at_rna2_spots` (rna1) / `n_nuclei_partner_rotation_null_dapi_at_rna2_spots` (dapi)

The `rna1` names are byte-for-byte the 2026-09-03 names, so a `"rna"`-only run
is unchanged. `rna_protein`'s `rna2` → `protein` relabel turns every one of
these into `..._at_protein_spots` while the leading `rna1` / `dapi` token, which
names the SAMPLED FIELD, survives.

### Reading the numbers

The DAPI value is a reference for the estimator, not a biological control. It
shows what the metric returns when the sampled field cannot carry the signal;
it does not show what the metric would return from a non-specific *probe*. A WT
rna1 value that merely exceeds 1.0 is uninformative if the DAPI value also
does; the comparison to make is rna1 against dapi on the same nuclei.

## (b) `foci.partner_anchored_null_nuclear_anchors_only` — one spatial support

**Problem.** The observed statistic is computed over every anchor wherever it
lies. The keep-N rotation redraw forces every null anchor into
`nucleus minus nucleolus`. In the arm-2 run, 2 304 of 11 228 WT QKI anchors —
20.5 % spot-weighted, 13.2 % per-nucleus mean — lay outside the nucleus mask, so
the observed value drew on a support the null never sampled.

**Change.** `FociCfg.partner_anchored_null_nuclear_anchors_only: bool = False`.
When on, the partner constellation is restricted to anchors with
`in_nucleus == True` before the existing nucleolus drop, which leaves exactly
`nucleus minus nucleolus` — the null's own support. The anchor count the null
actually used is emitted per nucleus as `n_partner_anchors_at_rna2_spots`, so
the set the statistic rests on is on disk rather than inferred.

### FOOTGUN this exists because of

**`FociChannelOverrideCfg.only_nuclear_spots` does not filter anything in
`rna_rna` / `rna_protein`.** It is resolved by `FociCfg.resolved_for` and
written to `thresholds.csv` as `rna_only_nuclear_spots` /
`rna2_only_nuclear_spots`, and that is all it does — `grep` finds no other
consumer in `src/`. `stratify_spots` annotates `in_nucleus` but never drops a
row on the strength of this flag.

The arm-2 preset set `antibody_overrides.only_nuclear_spots: true` and still
produced 20.5 % out-of-nucleus anchors. A run asserted a restriction it never
applied and exited 0, with `thresholds.csv` recording `True` as evidence for it.
Same shape as the other traps in this project: a check that cannot fire, so it
passes confidently.

This change does **not** repair `only_nuclear_spots` — doing so would silently
alter every preset that sets it. The new flag is scoped to the partner-anchored
null only. Repairing or removing `only_nuclear_spots` is a separate decision.

## (c) `foci.rna_pedestal_normalize` — per-image pedestal, before detection

**Problem.** A fixed absolute `threshold_override` is only comparable across
images when their background pedestals are. In the QKI × BIN1-intron
acquisition the in-nucleus 561 median spans 403–1056 across fields (2.6×) while
p99/p50 stays 1.77–2.10 in WT and 2.09–2.28 in KO. The field-to-field difference
is **multiplicative**, so no single threshold calibrates the set, and a higher
global `min_spot_peak_intensity` makes it worse: KO cytoplasmic calls have
higher peak intensity than WT nuclear calls (medians 1653 vs 1277), so every
floor removes WT signal faster than KO pedestal calls.

**Change.** `FociCfg.rna_pedestal_normalize: bool = False` plus
`FociCfg.rna_pedestal_stat: Literal["nuclear_median"] = "nuclear_median"`. When
on, the rna1 plane handed to LoG detection is multiplied by

```
factor = batch_reference / this image's in-nucleus median
```

where `batch_reference` is the median of the per-image in-nucleus medians over
the **biological** (non secondary-only) images, so absolute thresholds keep
their original scale. Secondary-only fields still receive a factor — they are
detected on the same pedestal — but do not set it, because they exist to measure
background.

The medians are harvested from the existing batch pre-pass, which already pools
nuclear rna1 pixels per image, so the feature costs no extra segmentation. It
therefore **requires `pixel_coloc.threshold_scope: batch`**; without it the
runner prints an explicit NOTE and detection runs un-normalised rather than
inventing a reference. `runner.resolve_rna_pedestal` is a pure function and is
unit-tested directly.

**Scope is deliberately narrow.** Only the array handed to spot detection is
scaled. Partner intensity, pixel coloc, every rotation null, the footprint
metrics and the publication images all still read raw counts.

The scaled plane is cast **back to the input dtype**, so the BigFISH LoG
response stays in the same quantisation regime as an un-normalised run.
Otherwise an arm-to-arm comparison would confound the pedestal with a
float-vs-uint16 filter change. Clipped pixels are counted, not hidden.

### Emitted columns

`thresholds.csv`, only when the feature is requested:

| column | meaning |
|---|---|
| `rna_pedestal_normalize` | the feature was requested |
| `rna_pedestal_stat` | which statistic defined the pedestal |
| `rna_pedestal_applied` | a factor actually reached this image |
| `rna_pedestal_factor` | the multiplier applied (NaN if not applied) |
| `rna_pedestal_clipped_px` | pixels clipped at the dtype ceiling |

`rna_pedestal_applied` separates "requested and applied" from "requested but no
factor reached this image". Those two produce the same spots and must not look
the same in the provenance table.

`spot_metrics.csv` gains `intensity_peak_normalized`: the detection-time value.
`spot_peak_intensity` / `peak_intensity` stay RAW. The column is emitted for
both channels whenever the feature is requested, so the column set does not
drift between images when one lacks a factor; rna2 rows are NaN.

---

## Verification

| check | result |
|---|---|
| `tests/test_dapi_reference_null_and_rna_pedestal.py` | 55 tests |
| `tests/test_partner_anchored_rotation_null.py` (pre-existing) | 44 passed after the change, before an unrelated concurrent edit landed |
| unset path vs the arm-2 run of record, 2 real images | see below |

### Real-data no-op proof

Arm-2 preset **unchanged**, current engine, all three new fields unset, two
images (`BIN1_introns_QKI_QKI-BIN1in_WT-1_00.vsi`,
`BIN1_introns_QKI_QKi-BIN1-KO-1_09.vsi`), exit 0 in 248 s. Compared against
`F:\Image Analysis Work\QKI_BIN1introns_2026_08_25\05b_SENSITIVITY_AND_ARM2_2026-09-04\arm2_strictQKIspots_T36_2026-09-04_0546`
restricted to those two images.

| check | result |
|---|---|
| `spot_metrics.csv` counts per (image, channel) | identical: 1775 / 152 / 1 / 568 |
| spot `x_px`, `y_px`, `spot_peak_intensity` | identical, max abs diff **0** |
| nuclei | 35 vs 35 |
| all 11 rotation columns | identical, max abs diff **0** |
| pedestal columns in the OFF run | absent |
| `intensity_peak_normalized` in the OFF run | absent |

Full output:
`F:\Image Analysis Work\QKI_BIN1introns_2026_08_25\05b_SENSITIVITY_AND_ARM2_2026-09-04\_noop_proof_2026-09-04\NOOP_PROOF_RESULT.txt`.

**15 of 169 shared per-nucleus columns differ, and none of them is caused by
this change.** All 15 are downstream of the batch-pooled pixel-coloc threshold
(`coloc_mask_thr_*`, `rna_threshold_value`, `protein_threshold_value`, the
Manders / Jaccard / Dice / `frac_above_thr` family). The proof run pools 2
images where the reference pooled 26, so the pooled threshold differs by
construction. This is the same class of difference the 2026-09-03 repro
documented.

Six columns are present only in the proof run
(`rna1_/protein_median_footprint_area_um2`, `*_median_size_fit_fwhm_um`,
`*_n_size_fit_ok`). They come from a **concurrent, unrelated** spot-size feature
added to the working tree by another agent between the arm-2 run and this proof,
not from this change.

---

## Files changed

| path | change |
|---|---|
| `E:\Claude\fishsuite\src\fishsuite\config\schema.py` | `FociCfg` gains `partner_anchored_null_sample_field`, `partner_anchored_null_nuclear_anchors_only`, `rna_pedestal_normalize`, `rna_pedestal_stat`. |
| `E:\Claude\fishsuite\src\fishsuite\core\modes\rna_rna.py` | `run_one` gains `rna_pedestal_factor`; pedestal scaling before `_detect` with raw-intensity restoration; partner-anchored null generalised to a per-sampled-field loop with disjoint RNG streams; in-nucleus anchor restriction and anchor count; gated pedestal provenance in `thresholds`. |
| `E:\Claude\fishsuite\src\fishsuite\core\modes\rna_protein.py` | forwards `rna_pedestal_factor`. |
| `E:\Claude\fishsuite\src\fishsuite\runner.py` | new pure `resolve_rna_pedestal`; per-image in-nucleus medians harvested in the batch pre-pass; reference + per-image factors resolved and forwarded. |
| `E:\Claude\fishsuite\tests\test_dapi_reference_null_and_rna_pedestal.py` | NEW. 55 GPU-free tests. |
| `E:\Claude\fishsuite\docs\ENGINE_CHANGE_2026-09-04_...md` | NEW. This note. |

Presets built from these options, with their derivations:

- `F:\Image Analysis Work\QKI_BIN1introns_2026_08_25\04_FISHSUITE_SETUP_2026-09-04\make_arm2b_arm3_presets.py`
- `...\qki_bin1_arm2b_strictQKIspots_nucAnchors_dapiRef_T36_2026-09-04.yaml`
- `...\qki_bin1_arm3_pedestalNorm_T36_diffuseQKI_2026-09-04.yaml`
