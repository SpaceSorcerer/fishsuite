# Phase 0 status — fishsuite scaffolding + bioio VSI smoke test

**Date:** 2026-05-11
**Build target:** `E:\Claude\fishsuite\`
**Python env:** `C:\Users\ambur\miniconda3\envs\fishproc\python.exe` (Python 3.10.20)

## Package install

Project scaffold created at `E:\Claude\fishsuite\` per the design doc. Layout:

```
E:\Claude\fishsuite\
  pyproject.toml
  README.md
  src/fishsuite/__init__.py
  src/fishsuite/cli.py
  src/fishsuite/runner.py
  src/fishsuite/core/{io,segmentation,spots,thresholds,metrics,morphology,parallel}.py
  src/fishsuite/core/modes/{rna_only,rna_protein,rna_rna,ab_ab,protein_only,pub_images}.py
  src/fishsuite/config/schema.py
  src/fishsuite/config/presets/{h9_hesc_100x,hek293_60x,u2os_100x,generic_100x_0p065,generic_60x_0p108}.yaml
  tests/test_smoke.py
  docs/PHASE0_STATUS.md
  docs/bioio_smoke.py
```

Pip-installed deps into the `fishproc` env: `bioio`, `bioio-bioformats`, `pydantic`, `click`, `rich`, `psutil`, `hypothesis`. Numpy stayed at `<2.0` (required by tensorflow / stardist).

## bioio VSI-read success

`bioio.BioImage(...)` loads Brian's VSIs successfully via the bioio-bioformats backend (cjdk auto-downloaded `zulu-jre 11.0.31`).

Survey of the 13 input VSIs at `F:\Raw Images\H9-MIAT-KD-ASO\H9-MIAT-KD-05-08-2026\`:

| Filename | Scenes | Shape (TCZYX) | Channels |
|----------|--------|---------------|----------|
| H9-MIAT-ASOs-_01.vsi | 2 | (1, 3, 4, 2304, 2304)  | 640 CSU / 561 CSU / 405 CSU |
| H9-MIAT-ASOs-_02..05 (Folder_NT) | 2 | (1, 3, 33, 2304, 2304) | 640 CSU / 561 CSU / 405 CSU |
| H9-MIAT-ASOs-_06 (Folder_NT-2only) | 2 | (1, 3, 33, 2304, 2304) | 640 CSU / 561 CSU / 405 CSU |
| H9-MIAT-ASOs-_07..09, _13 (Folder_MIATKD) | 2 | (1, 3, 33, 2304, 2304) | 640 CSU / 561 CSU / 405 CSU |
| H9-MIAT-ASOs-_10..12 (Folder_KD-2only) | 2 | (1, 3, 33, 2304, 2304) | 640 CSU / 561 CSU / 405 CSU |

Physical pixel sizes: X = Y = 0.065 µm = **65 nm/px** in xy, Z = 0.21 µm = 210 nm.

Channels are 0-indexed `[0=640 CSU (Cy5/MIAT), 1=561 CSU (Cy3), 2=405 CSU (DAPI)]`. Brian's spec mentioned "channel 3 (DAPI)" — that is Fiji's 1-indexed channel 3, which maps to **0-indexed `2`**. The preset writes `dapi: 2` explicitly and falls back to `autodetect_channels()` (which finds `dapi=2` for these images) if `-1` is given.

Smoke test passed: max projection of channel 2 written to `E:\Claude\fishsuite\docs\bioio_smoke_dapi_maxproj.tif` (2304 × 2304 uint16).

## Blocker fixed: bffile expects numpy >= 2.0

`bffile` (bioio's bioformats data shim) calls `np.asarray(data, dtype=dtype, copy=False)`. The `copy=False` kwarg is a numpy-2 addition; on numpy<2 the call raises `TypeError`.

We can't downgrade numpy further (TF/stardist need 1.26.x). We monkey-patch `bffile._biofile._reshape_image_buffer` inside `src/fishsuite/__init__.py::_apply_bffile_compat_patch()` at import time. The patch is one function, ~10 lines, fully reversible by reinstalling bffile.

Verified end-to-end on `H9-MIAT-ASOs-_01.vsi`: `BioImage.get_image_data("ZYX", T=0, C=2)` returns a `(4, 2304, 2304) uint16` array. Issue is permanently solved as long as the package `fishsuite` is imported before bioio.

## Z-stack note

The image `H9-MIAT-ASOs-_01.vsi` has Z=4, while the other 12 VSIs have Z=33. Brian's spec says `Z_START=20, Z_END=30` for autofocus — that range is valid for the 12 normal images, but exceeds the 4-slice stack on `_01`. `core/io.py::extract_channel` clamps the autofocus search range to `[1, n_z]` automatically; for the 4-slice image, autofocus picks the sharpest of the 4 available planes. No data loss; no exception.

## Status

Phase 0 complete. Proceeding to Phase 1 (core engine end-to-end on H9 data).

## Known caveats deferred to later phases

- `--parallel` worker count uses single-process for now; ProcessPoolExecutor wiring is Phase-2.
- Streamlit UI is Phase-3.
- The `rna_protein`, `rna_rna`, `ab_ab`, `protein_only`, `pub_images` modes are present as registry stubs that route through `rna_only` (Phase-2 will flesh them out).
