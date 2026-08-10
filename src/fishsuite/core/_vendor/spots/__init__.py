# -*- coding: utf-8 -*-
"""Pluggable spot-detection backends for the H9 / coloc pipeline.

Two backends, same input / output contract:
    Input:  2D (or 3D) RNA-channel image (TIFF)
    Output: CSV of detected spots with x_px, y_px, [z_slice,] intensity, etc.
            + summary metadata (threshold used, n spots, etc.)

Backends:
    - log         : Laplacian-of-Gaussian via skimage.feature.blob_log.
                    Same family as TrackMate's LoG detector. Fast, reliable,
                    needs a hand-tuned threshold (or auto via percentile).
    - bigfish     : `bigfish.detection.detect_spots` with auto Otsu-on-LoG
                    threshold. Built specifically for RNA FISH; produces
                    the same kind of spot list but the threshold is
                    self-calibrating and the algorithm handles dense
                    clusters more gracefully.

Use from the command line:
    python -m spots.detect_spots --input rna.tif --backend bigfish --output spots.csv

Or compare both side-by-side on one image:
    python -m spots.compare_backends --input rna.tif --output-dir compare/
"""
