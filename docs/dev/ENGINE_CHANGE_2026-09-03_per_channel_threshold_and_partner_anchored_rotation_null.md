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


---

# ADDENDUM 2026-09-03 - stage-09 DAPI Otsu pre-clip merged

Ported verbatim from the stage-09 proposal worktree
`F:\Image Analysis Work\RNASEH2B_BIN1introns_2026_08_25\09_CODE_PROPOSAL_FISHSUITE_DAPI_FLOOR_2026-09-01\src\fishsuite`,
the engine that produced run 11. Credit for the implementation belongs to that
proposal; this change only merges it and adds tests.

`nuclei.cellpose_preclip_dapi_otsu` (default `False`) zeroes DAPI pixels below a
per-image Otsu threshold **before** Cellpose inference. It changes the model
input, not the returned labels, and applies only to the cellpose backend. The
floor is recorded per image as `cellpose_dapi_otsu_floor` in `thresholds.csv`.

**Scope correction.** The feature spans **7 files**, not the 2 estimated
earlier: `core/segmentation.py` is the consumer, so porting only `rna_rna.py`
and `schema.py` would have raised `TypeError` on an unexpected kwarg. Ported:
`core/segmentation.py` (+12/-2), `config/schema.py` (+4), `core/modes/rna_rna.py`
(+12), `core/modes/rna_only.py` (+12), `core/modes/if_intensity.py` (+1),
`runner.py` (+1), `core/excel_report.py` (+4). A guard asserted that each file's
only difference from `HEAD` was this feature before it was copied.

## NucleiCfg rejects unknown keys

`model_config = ConfigDict(extra="forbid")`. It previously inherited
`extra="ignore"`, which silently discarded `cellpose_preclip_dapi_otsu` from
every preset naming it while the feature was unmerged, so a run asserted a
segmentation step it never performed and exited 0.

Rejection, not a warning: a warning in several hundred lines of run output is
missable, and this failure mode produced wrong science silently. Verified safe
against all 63 configs in this repository and the RNASEH2B analysis tree - zero
carry unknown keys. The error names the offending key.

Only `NucleiCfg` was in scope. Every other block still inherits
`extra="ignore"` and remains exposed to the same failure.

## Verification

| check | result |
|---|---|
| `tests/test_cellpose_preclip_dapi_otsu.py`, 9 tests | 9 passed, 4.64 s |
| pre-clip OFF passes raw DAPI to the backend | asserted array-equal |
| pre-clip ON zeroes exactly the sub-Otsu pixels | asserted, non-Otsu pixels untouched |
| ignored for non-cellpose backends | asserted |
| unknown `nuclei` key rejected and named in the message | asserted |
| run-11 preset, pre-clip ON, 2 real images | 37 nuclei vs run 11's 36 - closer than 41, still not equal |

The residual is a **cellpose downgrade**, not this code: run 11 used cellpose
4.2.1.1, the current `fishproc_dml` environment has 4.1.1, and every other
package version matches. `nucleus_area_px` differs for every nucleus by at most
3.05 %, spot assignment is unchanged, and the preset asks for the 4.2-era model
`cpsam_v2` which 4.1.1 cannot resolve. Restoring 4.2.1.1 is a system-state
decision for Brian. Full comparison in `REPRO_RESULT.md`.


---

# ADDENDUM 2 (2026-09-03) - run 11 reproduced exactly; the earlier version claim was wrong

**The merged engine reproduces run 11 exactly**: 36 nuclei (16 WT + 20 KO),
identical spot counts, and `nucleus_area_px`, `nuclear_spot_fraction`,
`rna_spot_count`, `nuclear_spot_count`, `protein_nuclear_mean` and all three
`protein_rotation_*_at_rna1_spots` columns equal for all 36 nuclei with maximum
absolute difference 0. 148 of 163 shared per-nucleus columns are bit-identical;
the 15 that differ are all downstream of the batch-pooled pixel-coloc threshold,
which pools 2 images instead of run 11's 26. Full tables in `REPRO_RESULT.md`.

No engine code changed for this result. It required only running the merged
engine in run 11's environment.

## Correction: there was no cellpose downgrade

The earlier addendum blamed a cellpose 4.2.1.1 -> 4.1.1 downgrade of
`fishproc_dml`. **That was wrong.** `fishproc_dml` has never held 4.2.1.1:

| evidence | finding |
|---|---|
| `site-packages\cellpose-*.dist-info` | exactly one, `4.1.1`, dated 2026-05-27 |
| `conda-meta\history` | one transaction, 2026-05-27 14:25:51, no cellpose entry (pip-installed) |
| `importlib.metadata` + `cellpose.version` today | both 4.1.1, one distribution only |
| every cellpose install on this machine | 4.1.1 in `fishproc`, `fishproc_dml`, `dml_test`, all May 2026 |
| every other fishsuite run 2026-08-13 to 2026-09-03 | records `cellpose: 4.1.1` |

A pip downgrade would have stamped a new dist-info mtime. It did not.

## Where run 11's 4.2.1.1 actually came from

It is **vendored inside the stage-09 proposal**, alongside its weights:

| path | contents |
|---|---|
| `09_CODE_PROPOSAL_FISHSUITE_DAPI_FLOOR_2026-09-01\_deps\cellpose_4_2_1_1\` | the cellpose 4.2.1.1 package + dist-info |
| `09_CODE_PROPOSAL_FISHSUITE_DAPI_FLOOR_2026-09-01\_models\cellpose_4_2_1_1\cpsam_v2` | the cpsam_v2 weights, 1,233,586,851 bytes |

Injected per process with `PYTHONPATH` and `CELLPOSE_LOCAL_MODELS_PATH`, which is
why `sys.executable` still recorded `fishproc_dml`. Corroborated within one day
and one folder: stage 10's `CP411_CPSAM_OTSU` arm records 4.1.1 while its
`CP421_CPSAMV2` and `CP421_CPSAMV2_OTSU` arms and run 11 record 4.2.1.1.

**No environment was created and `fishproc_dml` was not modified.** Cloning it
and pip-installing 4.2.1.1 was unnecessary once the vendored copy was found, and
would have pulled ~1.2 GB of weights over the network for a model already on
disk.

## FOOTGUN: cellpose 4.1.1 silently substitutes a different model for cpsam_v2

`cellpose/models.py` in 4.1.1 defines `MODEL_NAMES = ["cpsam"]`. `cpsam_v2` is
not in that list and is not a path, so it falls to the `else` branch:

```python
pretrained_model = os.path.join(MODEL_DIR, "cpsam")
models_logger.warning(
    f"pretrained model {pretrained_model} not found, using default model")
```

Observed verbatim in the merged-pre-clip repro run:

```
pretrained model C:\Users\ambur\.cellpose\models\cpsam not found, using default model
```

Three ways that line misleads:

1. It names `...\cpsam`, the **substitute**, not `cpsam_v2`, the model actually
   requested. The requested name never appears.
2. It says "not found" about a path that **does exist** (1,233,587,898 bytes,
   present since 2026-05-09), so the message is false as written.
3. It is a WARNING. Segmentation proceeds with the wrong model and the run exits
   0 with a complete, plausible output tree.

4.2.1.1 has `MODEL_NAMES = ["cpsam_v2", "cpdino", "cpdino-vitb", "cpsam"]` and
resolves `cpsam_v2` through `cache_model_path`, which returns the cached file
under `MODEL_DIR` and downloads only if absent. With
`CELLPOSE_LOCAL_MODELS_PATH` pointed at the vendored `_models` directory it
resolves locally with no network access, verified before running.

Same shape as the other traps in this project: a check that cannot fire, so it
passes confidently. Anyone reading `versions.txt` alone cannot tell which model
ran; only the warning line distinguishes them, and it names the wrong model.


---

# ADDENDUM 3 (2026-09-03) - fishproc_dml upgraded to cellpose 4.2.1.1 (standing stack)

Brian's decision: cellpose 4.2.1.1 + `cpsam_v2` is the standing model going
forward, permanently. `fishproc_dml` itself was upgraded. **No clone was ever
created**, so there is no rollback environment; the rollback record is the
pre-upgrade freeze.

| artifact | path |
|---|---|
| pre-upgrade freeze | `E:\Claude\fishsuite\docs\env_freeze_fishproc_dml_pre_cellpose_4.2.1.1_2026-09-03.txt` |
| post-upgrade freeze | `E:\Claude\fishsuite\docs\env_freeze_fishproc_dml_post_cellpose_4.2.1.1_2026-09-03.txt` |

## Deviation: `--no-deps`, because a plain install would have moved numpy

`pip install cellpose==4.2.1.1 --dry-run` reported it would also install
**numpy 2.2.6**, replacing numpy 1.26.4:

```
Would install cellpose-4.2.1.1 numpy-2.2.6
```

That is far outside the requested change. numpy 1.26.4 is what every run of
record used, fishsuite's package init carries a numpy<2 compatibility patch, and
bioio-bioformats / stardist / big-fish sit on that pin. Moving the array library
under the whole stack to obtain a segmentation-model upgrade is not a trade worth
making silently, so the install was `--no-deps`. Every other cellpose 4.2.1.1
requirement was already satisfied, and 4.2.1.1 had already been proven to run
correctly against numpy 1.26.4 in the vendored-environment reproduction.

`pip freeze` before and after differ by **exactly one line** across 142 packages:

```
9c9
< cellpose==4.1.1
---
> cellpose==4.2.1.1
```

torch 2.4.1, torch-directml 0.2.5.dev240914, numpy 1.26.4 all unchanged.

## Weights

`cpsam_v2` was not in the default cache, so a plain run would have downloaded
~1.2 GB from HuggingFace. Instead the vendored copy already on disk was placed in
the default model directory and verified byte-identical:

| item | value |
|---|---|
| destination | `C:\Users\ambur\.cellpose\models\cpsam_v2` |
| bytes | 1,233,586,851 |
| SHA-256 | `0f1cc3f7ecdd8a037a57c6c48d9d8921391be4cbce3fa9f13c3e3a2e1253c667` |
| source | stage 09 `_models\cellpose_4_2_1_1\cpsam_v2`, same hash |

`cache_model_path('cpsam_v2')` now resolves there with **no environment
variables set and no network access**. `PowerShell Get-FileHash` is unavailable
in this shell and returned empty strings, so the hashes were taken with
`sha256sum`; the earlier `MATCH: True` it printed was comparing two empty
strings and proved nothing.

## Reproduction in the upgraded environment

`RUN_repro_preclip_on_envupgrade_2026-09-03_2110\`, run with **no** `PYTHONPATH`
or `CELLPOSE_LOCAL_MODELS_PATH`, run-11 preset unchanged, pre-clip ON, both
2026-09-03 fields unset.

**36 nuclei (16 WT + 20 KO), equal to run 11.** Spot counts identical
(83 / 7219, 324 / 5362). All nine compared per-nucleus columns equal for all 36
nuclei at max absolute difference 0: `nucleus_area_px`, `nuclear_spot_fraction`,
`rna_spot_count`, `nuclear_spot_count`, `protein_nuclear_mean`,
`protein_enrichment_at_rna1_spots`, `protein_rotation_enrichment_at_rna1_spots`,
`protein_rotation_null_z_at_rna1_spots`,
`protein_rotation_assoc_fraction_at_rna1_spots`.

Against the vendored-environment run: **163 of 163 columns identical, 0 differ.**
The upgraded environment and the vendored one are the same stack.

## Launcher

`run_sweep.ps1` now calls the plain `fishproc_dml` `fishsuite.exe` with no
environment injection. It still asserts `cellpose.version == 4.2.1.1` and that
the `cpsam_v2` weights exist before any arm, because a downgrade would segment
with the wrong model and still exit 0. Pre-flight re-run, all five arms:

```
cellpose: 4.2.1.1 (fishproc_dml, standing stack); model cpsam_v2 at C:\Users\ambur\.cellpose\models\cpsam_v2
[T30] roster: 12 biological + 7 secondary-only (expect 12 + 7)
[T33] roster: 12 biological + 7 secondary-only (expect 12 + 7)
[T36] roster: 12 biological + 7 secondary-only (expect 12 + 7)
[T40] roster: 12 biological + 7 secondary-only (expect 12 + 7)
[T45] roster: 12 biological + 7 secondary-only (expect 12 + 7)
PRE-FLIGHT OK - all 5 arms passed the environment + roster gate; nothing was run
```

The sweep has still **not** been launched.

## Consequence for older runs

Every fishsuite run before 2026-09-03 that used `cellpose_model_type: cpsam_v2`
outside the vendored environment segmented with `cpsam`, not `cpsam_v2`, and
recorded `cellpose: 4.1.1`. Those runs are internally consistent but are not
comparable to runs on the standing stack. `versions.txt` distinguishes them.


---

# INCIDENT 2026-09-03 - a live sweep arm was disturbed mid-run

## What happened

| time | event |
|---|---|
| 19:03:31 | orchestrator launched `run_sweep.ps1` (pid 74964) |
| 19:03:39 | T30 real run started (pid 56260), `-o ...\T30_2026-09-03_1903` |
| ~19:15 | I moved `T30_2026-09-03_1903` into `_preflight_stubs_2026-09-03\` **while that arm was running**, having misread it as a leftover dry-run stub |
| same window | I also ran `pip install cellpose==4.2.1.1` into `fishproc_dml` and edited `run_sweep.ps1`, both under the live job |
| - | orchestrator killed the sweep |
| ~19:25 | relaunched with the corrected launcher |

**The 19:03 arm is COMPROMISED and DISCARDED. The ~19:25 relaunch is the sweep
of record.** Nothing from the 19:03 arm may be compared, pooled or cited.

## Why the check I did make was not a check

The directory held no CSVs, so I classified it as a dry-run stub. That is
consistent with a stub **and** with an arm 11 minutes into a run that writes its
master CSVs only at the end. Absence of output does not distinguish "never ran"
from "still running"; only the process table does. I had also just written the
launcher and assumed it had not been launched yet.

## Rule (2026-09-03)

Before touching any file, directory or environment that a run may be using:

1. **Check for a live run first.** `Get-CimInstance Win32_Process` filtered for
   `fishsuite run`, and read its `-o` argument out of the command line.
2. **Never move or rename a directory that a live process was given as `-o`**,
   however empty it looks. An in-progress run dir and an abandoned stub are
   indistinguishable by content.
3. **Never `pip install` into an environment with a live run.** Packages are
   replaced under the running interpreter, and the failure surfaces much later
   as something unrelated.
4. **Never edit a script a live job is executing** - the shell resumes by byte
   offset.
5. **Treat "the orchestrator will launch it" as "may already have launched it."**
   Absence of a completion message is not evidence of an idle machine.

One command that answers 1 and 2 together:

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='fishsuite.exe'" |
  Where-Object { $_.CommandLine -match 'fishsuite.*\brun\b' } |
  Select-Object ProcessId, CreationDate, CommandLine | Format-List
```
