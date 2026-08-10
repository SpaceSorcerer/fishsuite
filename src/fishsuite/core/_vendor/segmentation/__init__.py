# -*- coding: utf-8 -*-
"""Pluggable nuclei-segmentation backends for the H9 / coloc pipeline.

Three backends, same input / output contract:
    Input:  2D DAPI image (TIFF, any bit depth)
    Output: 2D label image (16-bit TIFF, 0=background, 1..N=nucleus IDs)

Backends:
    - otsu      : the legacy threshold + watershed pipeline (sanity baseline)
    - stardist  : pretrained 2D fluorescent-nuclei model. Best on confluent
                  monolayers of round/ovoid nuclei. Requires `stardist` +
                  `tensorflow` (CPU is fine, ~500 MB install).
    - cellpose  : `nuclei` model. More flexible on irregular shapes. Requires
                  `cellpose` + `torch` (CPU works, ~2 GB install).

Use from the command line:
    python -m segmentation.segment_image --input dapi.tif --backend stardist --output labels.tif

Or from Jython via subprocess (see fiji_scripts/Coloc_Core.py adapter).
Or compare all three side-by-side on one image:
    python -m segmentation.compare_backends --input dapi.tif --output-dir compare/
"""
