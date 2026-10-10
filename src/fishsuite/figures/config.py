"""Inputs of one figure-panel build: the menu tables, the exact-footprint backfill h5, the source run, the output dir."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .style import Saver


@dataclass
class PanelRun:
    """menu_dir holds data/menu_per_nucleus.csv, data/menu_per_punctum.csv, data/representative_selection.json.
    backfill_dir holds selected_planes_and_masks.h5. run_dir is the source fishsuite run (named on B2 footers).
    raw_dir, if set, replaces the directory part of each image key when the raw VSI is read (read-only)."""
    menu_dir: Path
    backfill_dir: Path
    run_dir: Path
    out_dir: Path
    arm: str = "VPR noDox"
    raw_dir: Path | None = None
    native_only: bool = False
    scratch_dir: Path | None = None
    display_ranges: dict | None = None   # {"miat"|"qki"|"dapi": (lo, hi)}; None = style.LV
    px_um: float | None = None   # --px-um override; else image_manifest.csv voxel_xy_nm; else style.PX_UM + WARNING
    _saver: Saver | None = field(default=None, repr=False)

    def __post_init__(self):
        for k in ("menu_dir", "backfill_dir", "run_dir", "out_dir"):
            setattr(self, k, Path(getattr(self, k)))
        if self.raw_dir is not None:
            self.raw_dir = Path(self.raw_dir)

    @property
    def pan(self) -> Path:
        return self.out_dir / "panels"

    @property
    def data(self) -> Path:
        if self.native_only:
            return Path(self.scratch_dir) if self.scratch_dir else self.out_dir / "_native_rerender_scratch"
        return self.out_dir / "data"

    @property
    def h5(self) -> Path:
        return self.backfill_dir / "selected_planes_and_masks.h5"

    @property
    def saver(self) -> Saver:
        if self._saver is None:
            self._saver = Saver(self.pan, native_only=self.native_only)
        return self._saver

    @saver.setter
    def saver(self, s: Saver):
        self._saver = s

    def selection(self) -> dict:
        return json.loads((self.menu_dir / "data" / "representative_selection.json").read_text())

    def image_key(self, field_id) -> str:
        """Image key of field ``field_id`` of ``arm``, read from the menu's per-nucleus ``image`` column.
        Raises KeyError unless exactly one image matches (never falls back to the representative image)."""
        nuc = pd.read_csv(self.menu_dir / "data" / "menu_per_nucleus.csv", usecols=["image", "condition", "well"])
        nuc = nuc[nuc.condition == self.arm]
        fld = nuc.well.str.split("_").str[-1]
        want = str(field_id)
        hit = fld == want
        if not hit.any() and want.isdigit():
            hit = fld.str.isdigit() & (fld.where(fld.str.isdigit(), "-1").astype(int) == int(want))
        keys = sorted(set(nuc.image[hit]))
        if len(keys) != 1:
            raise KeyError(f"arm {self.arm!r} field {field_id!r}: {len(keys)} images in menu_per_nucleus.csv ({keys})")
        return keys[0]

    def pixel_size_um(self, key=None) -> float | None:
        """--px-um, else voxel_xy_nm / 1000 from the backfill image_manifest.csv (row of ``key`` by basename when
        given, else the manifest's single value); None when the manifest records none or has no row for ``key``. Raises on disagreeing values."""
        if self.px_um is not None:
            return float(self.px_um)
        path = self.backfill_dir / "image_manifest.csv"
        if not path.is_file():
            return None
        m = pd.read_csv(path, usecols=lambda c: c in ("image", "image_key", "voxel_xy_nm"))
        if "voxel_xy_nm" not in m:
            return None
        if key is not None:
            name = Path(str(key)).name.casefold()
            hit = np.zeros(len(m), dtype=bool)
            for col in ("image", "image_key"):
                if col in m:
                    hit |= m[col].astype(str).map(lambda v: Path(v).name.casefold() == name).to_numpy()
            if not hit.any():
                return None
            m = m[hit]
        values = set(pd.to_numeric(m.voxel_xy_nm, errors="coerce").dropna())
        if not values:
            return None
        if len(values) > 1:
            raise ValueError(f"image_manifest.csv records several voxel_xy_nm {sorted(values)} for {key!r}; pass --px-um")
        return values.pop() / 1000.0

    def raw_path(self, key: str) -> Path:
        return Path(key) if self.raw_dir is None else self.raw_dir / Path(key).name

    def tables(self):
        """(nv, sv): per-nucleus and per-punctum rows of ``arm`` with a ``field`` column (last token of well)."""
        nuc = pd.read_csv(self.menu_dir / "data" / "menu_per_nucleus.csv")
        spt = pd.read_csv(self.menu_dir / "data" / "menu_per_punctum.csv")
        nv = nuc[nuc.condition == self.arm].copy()
        sv = spt[spt.condition == self.arm].copy()
        nv["field"] = nv.well.str.split("_").str[-1]
        sv["field"] = sv.well.str.split("_").str[-1]
        return nv, sv

    def spots_all(self) -> pd.DataFrame:
        return pd.read_csv(self.menu_dir / "data" / "menu_per_punctum.csv")

    def load_plane(self, key: str) -> dict:
        import h5py
        with h5py.File(self.h5, "r") as f:
            for k in f["images"]:
                g = f["images"][k]
                if g.attrs["image_key"] == key:
                    return ({c: g[c][()] for c in ("dapi", "miat", "qki", "nucleus_labels")}
                            | {"z0": int(g.attrs["selected_z_0based"])})
        raise KeyError(key)
