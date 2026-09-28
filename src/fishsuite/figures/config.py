"""Inputs of one figure-panel build: the menu tables, the exact-footprint backfill h5, the source run, the output dir."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

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
        return self.selection()["reps"][self.arm]["image"].replace("dox_15.vsi", f"dox_{field_id}.vsi")

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
