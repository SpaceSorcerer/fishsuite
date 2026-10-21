import numpy as np
from fishsuite.config.schema import FishsuiteConfig
from fishsuite.core.modes import _zplanes

class _Bio:
    def __init__(self, zyx): self._z = zyx
    def get_image_data(self, order, T=0, C=0): return self._z[C]

class _Img:
    def __init__(self):
        z = np.arange(8, dtype=np.uint16)[:, None, None] * np.ones((8, 4, 4), np.uint16)
        self.bio = _Bio([z, z + 100])
        self.n_z = 8; self.n_channels = 2

def _cfg(**z):
    c = FishsuiteConfig.model_validate({"z_stack": {"mode": "single", **z}})
    return c

def test_single_slice_is_honoured_and_default_unchanged():
    img = _Img()
    out = _zplanes.extract_planes(img, "a.tif", _cfg(single_slice=6), dapi_idx=0, rna_idx=1, rna2_idx=None)
    assert out["dapi_2d"][0, 0] == 5 and out["rna_2d"][0, 0] == 105
    out = _zplanes.extract_planes(img, "a.tif", _cfg(), dapi_idx=0, rna_idx=1, rna2_idx=None)
    assert out["dapi_2d"][0, 0] == 4  # mid-stack rule unchanged
    out = _zplanes.extract_planes(img, "a.tif", _cfg(single_slice=2, file_overrides={"a.tif": {"single_slice": 3}}), dapi_idx=0, rna_idx=1, rna2_idx=None)
    assert out["dapi_2d"][0, 0] == 2
