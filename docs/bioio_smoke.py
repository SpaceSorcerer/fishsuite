"""Smoke test: read a VSI via bioio and confirm shape + channels."""
from pathlib import Path
import sys
import numpy as np
import tifffile

VSI = Path(r"F:\Raw Images\H9-MIAT-KD-ASO\H9-MIAT-KD-05-08-2026\Folder_NT\H9-MIAT-ASOs-_01.vsi")
OUT_TIFF = Path(r"E:\Claude\fishsuite\docs\bioio_smoke_dapi_maxproj.tif")

print(f"Reading: {VSI}")
print(f"Exists: {VSI.exists()}")

try:
    from bioio import BioImage
    img = BioImage(VSI)
    print(f"Backend: {type(img.reader).__name__}")
    print(f"Scenes: {list(img.scenes)}")
    print(f"Dims order: {img.dims.order}")
    print(f"Shape: {img.shape}  (full TCZYX or similar)")
    print(f"Channel names: {img.channel_names}")
    try:
        psx = img.physical_pixel_sizes
        print(f"Physical pixel sizes: X={psx.X} Y={psx.Y} Z={psx.Z}")
    except Exception as e:
        print(f"Physical pixel sizes error: {e}")
    # Pick channel index 3 (DAPI per Brian) — note 0-indexed
    # Brian's pipeline uses 1-indexed channel "3" => python index 2
    # But spec says "channel 3 (DAPI per Brian's setup)". Both Fiji 1-indexed and 0-indexed
    # are plausible. Examine all channels first.
    for c in range(img.shape[img.dims.order.index('C')]):
        plane = img.get_image_data("ZYX", T=0, C=c)
        print(f"  C={c}: shape={plane.shape}, dtype={plane.dtype}, min={plane.min()}, max={plane.max()}, mean={plane.mean():.1f}")
    # Use 0-indexed C=2 (Fiji 3rd channel) for max projection
    dapi_idx = 2
    dapi_zstack = img.get_image_data("ZYX", T=0, C=dapi_idx)
    print(f"DAPI (C={dapi_idx}) zstack shape: {dapi_zstack.shape}")
    dapi_max = dapi_zstack.max(axis=0)
    print(f"DAPI max projection shape: {dapi_max.shape}, dtype: {dapi_max.dtype}")
    tifffile.imwrite(OUT_TIFF, dapi_max)
    print(f"Saved -> {OUT_TIFF}")
    print("SUCCESS")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"FAIL: {type(e).__name__}: {e}")
    sys.exit(1)
