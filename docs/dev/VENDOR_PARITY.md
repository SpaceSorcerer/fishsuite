# Vendoring parity record — 2026-08-09

Internal development record. Evidence that moving the segmentation and
spot-detection code from the external `image-analysis-pipeline` tree into
`src/fishsuite/core/_vendor/` changed no computed value.

Source: `image-analysis-pipeline` @ `4d8c8a74074b3820d3861418e829f2a15ad1b780`.
Provenance and per-file checksums: `src/fishsuite/core/_vendor/PROVENANCE.md`.
Test: `tests/test_vendor_parity.py`.

## Why comparison is in-process, not against an archived run

Neither cellpose nor StarDist is seeded anywhere in this package — no
`tf.random.set_seed`, and `core/segmentation.py` passes no seed through to
either backend. Comparing a fresh run against an archived output therefore
cannot distinguish a real difference caused by the move from ordinary
run-to-run variation.

Both copies are instead imported into the **same process** and called on
identical synthetic input. The original is loaded by explicit file path under a
private module name, which is safe for these two files specifically because
neither has any intra-repository import to resolve.

## Results

Environment: `fishproc_dml` (Python 3.10.20), Windows 11, AMD RX 6750 XT.
Assertions are exact equality (`np.array_equal`), not `allclose`: a file move
must change nothing at all.

| Component | Backend / detector | Result |
|---|---|---|
| `run_backend` | `otsu` | **PASS — bitwise identical** |
| `run_backend` | `stardist` (`2D_versatile_fluo`) | **PASS — bitwise identical** |
| `run_backend` | `cellpose` (`cpsam`, CPU) | **PASS — bitwise identical** (30 min 41 s) |
| `detect_spots_bigfish` | Big-FISH LoG | **PASS — identical coordinates and identical auto threshold** |
| `detect_spots_log` | scikit-image LoG | **PASS — identical coordinates and identical threshold** |
| vendored checksums | all 8 files | **PASS — every file matches `PROVENANCE.md`** |

## Self-containment, which is the point of the exercise

Measured with a `sys.meta_path` finder that raises `ModuleNotFoundError` for any
top-level import resolving under `F:\Image Analysis Work`, i.e. a simulated
machine with no lab tree:

| | Before vendoring | After vendoring |
|---|---|---|
| Full suite, lab tree blocked | **224 passed, 40 failed** | **268 passed, 0 failed** |
| Lab-tree modules demanded | `segmentation`, `spots` (40 tests dual-gated on them) | **none** |

The 40 previous failures were all one root cause: `core/spots.py` and
`core/segmentation.py` hard-coded
`sys.path.insert(0, r"F:\Image Analysis Work\image-analysis-pipeline\python")`,
which no other machine has.

Also verified directly: after `import fishsuite`, `sys.path` contains no
`image-analysis-pipeline` entry, and `core/_superplot.get_locked_drawer()`
resolves to `fishsuite.core._vendor.analysis.single_condition_plots` rather than
falling back.

## The two silent-failure modes that were closed

1. **`runner.py` figure subprocess.** It shells out to the figure module and
   deliberately ignores the exit code, so a wrong path produced zero figures and
   a green run. It now targets the in-package module with no `cwd` override, and
   warns loudly when the run finishes with no PNGs under `figures/`. An external
   checkout can still be selected with `$FISHSUITE_DOWNSTREAM_PATH`.
2. **`core/_superplot.get_locked_drawer()`.** It swallowed every exception and
   fell back to a built-in drawer, so a broken vendor path would still have
   rendered figures — with a different recipe. It now tries the in-package copy
   first and emits a `RuntimeWarning` naming the consequence if it has to fall
   back. Confirmed under `-W error::RuntimeWarning` that no fallback occurs.

Related: the one edited vendored line restored the locked per-condition colour
mapping. The import sits inside a `try/except` whose fallback sets
`CONDITION_COLORS = {}`, so left broken it would have silently drawn figures
with a different palette. Verified populated after the fix.

## Caveats

- Parity was measured on synthetic images, not on a real `.vsi`. The synthetic
  fields exercise the same code path; an end-to-end run on real data is worth
  doing before tagging a release.
- All five arms have now reported and all five are bitwise identical, so the
  parity claim is unqualified. `cellpose` took 30 minutes on CPU because `cpsam`
  inference is slow; run it with `-k cellpose` separately rather than expecting
  the default suite to cover it.
- Both copies were compared under one numpy/scikit-image/cellpose/StarDist
  version set. The comparison isolates the file move, not behaviour across
  dependency upgrades.
