# fishsuite

Standalone, Fiji-free Python pipeline for RNA-FISH / IF colocalization & quantification. Companion to (not replacement for) the Fiji pipeline at `F:\Image Analysis Work\image-analysis-pipeline\`.

**Status:** v0.1.0 — Phase 0/1 build complete (rna_only mode validated end-to-end on H9 hESC data).

## Quickstart

```powershell
# 1. Install (uses the existing fishproc conda env)
"C:\Users\ambur\miniconda3\envs\fishproc\python.exe" -m pip install -e E:\Claude\fishsuite

# 2. Run on a folder of VSI files (or TIFFs / CZIs)
fishsuite run --config src\fishsuite\config\presets\h9_hesc_100x.yaml `
              --input-dir  "F:\Raw Images\H9-MIAT-KD-ASO\H9-MIAT-KD-05-08-2026" `
              --output-dir "F:\Image Analysis Work\H9-Output-2\MIAT-KD-ASO" `
              --parallel auto

# 3. List built-in presets
fishsuite presets list

# 4. Preview a single image
fishsuite preview "F:\Raw Images\H9-MIAT-KD-ASO\H9-MIAT-KD-05-08-2026\Folder_NT\H9-MIAT-ASOs-_01.vsi"
```

## Post-run figures & tables (after a run)

Make the extra colocalization tables + figures from a **finished** run folder.
These are CPU-only (no GPU) and reuse the run's saved masks + spots:

```powershell
fishsuite postrun     --run "<run output folder>"   # one-shot: backfill + walkthrough
fishsuite backfill    --run "<run output folder>"   # coloc null draws + radial + QKI montage (CPU)
fishsuite walkthrough --run "<run output folder>"   # the 8-panel pipeline-walkthrough figure
```

See **[POSTRUN_UTILITIES.md](POSTRUN_UTILITIES.md)** for a beginner-friendly walkthrough.

## Modes

- `rna_only` — per-nucleus FISH spot count, intensity, N/C ratio
- `rna_protein` — RNA + protein pixel coloc + spot-vs-protein metrics
- `rna_rna` — spot-on-spot coloc (NN distance, paired fraction)
- `ab_ab` — pixel coloc on two antibody channels
- `protein_only` — per-nucleus protein quantification
- `pub_images` — figures only, no quantification

## Output schema

Column-for-column compatible with the existing Fiji pipeline. See `docs/output-schema.md`.

## Architecture

```
src/fishsuite/
  cli.py            -- click CLI
  core/
    io.py           -- VSI / CZI / TIFF reader (bioio + monkeypatch)
    segmentation.py -- stardist / cellpose / otsu wrapper
    spots.py        -- bigfish / LoG wrapper
    thresholds.py   -- MAD / Costes thresholds (bit-compat with Fiji)
    metrics.py      -- Pearson / Manders / Li ICQ / Jaccard / Dice
    morphology.py   -- N/C stratification
    parallel.py     -- ProcessPoolExecutor wrapper + auto worker count
    modes/          -- one module per analysis mode
  config/
    schema.py       -- pydantic v2 schema
    presets/        -- shipped YAMLs
```

## License

MIT.
