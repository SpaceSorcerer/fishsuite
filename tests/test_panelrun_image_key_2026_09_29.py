"""Regression (2026-09-29): PanelRun.image_key must resolve each field's own image for every arm.

The old key substituted only 'dox_15.vsi', so for arms whose representative image is not field 15 (Nog noDox rep = 07,
g2 noDox rep = 04) every field loaded the representative image and the renders were byte-identical across fields.
Synthetic menu + h5 only; no F: data.
"""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from fishsuite.figures.config import PanelRun

h5py = pytest.importorskip("h5py")

ROOT = "f:/raw images/x"
ARMS = {  # arm -> (rep field, fields, path stem)
    "VPR noDox": ("15", ["14", "15"], f"{ROOT}/vpr/no dox/ud-vpr-no dox_{{}}.vsi"),
    "Nog noDox": ("07", ["07", "08"], f"{ROOT}/nog/no dox/ud-nog-no dox_{{}}.vsi"),
    "g2 noDox (control)": ("04", ["03", "04"], f"{ROOT}/g2/no dox/ud-g2-no dox_{{}}.vsi"),
}


@pytest.fixture
def menu(tmp_path):
    (tmp_path / "data").mkdir()
    rows, reps = [], {}
    for arm, (rep, fields, stem) in ARMS.items():
        reps[arm] = {"image": stem.format(rep), "nucleus_id": 1}
        for f in fields:
            well = f"UD-{arm.split()[0]}-no Dox_{f}"
            rows += [dict(image=stem.format(f), condition=arm, well=well, nucleus_id=i) for i in (1, 2)]
    pd.DataFrame(rows).to_csv(tmp_path / "data" / "menu_per_nucleus.csv", index=False)
    (tmp_path / "data" / "representative_selection.json").write_text(json.dumps({"reps": reps}))
    rng = np.random.default_rng(0)
    with h5py.File(tmp_path / "selected_planes_and_masks.h5", "w") as h:
        g0 = h.create_group("images")
        for i, key in enumerate(sorted({r["image"] for r in rows})):
            g = g0.create_group(f"img{i:02d}")
            g.attrs["image_key"] = key
            g.attrs["selected_z_0based"] = 3
            for c in ("dapi", "miat", "qki"):
                g[c] = rng.integers(0, 4096, (16, 16), dtype=np.uint16)
            g["nucleus_labels"] = np.zeros((16, 16), np.int32)
    return tmp_path


def _hash(P):
    return hashlib.sha256(b"".join(np.ascontiguousarray(P[c]).tobytes() for c in ("dapi", "miat", "qki"))).hexdigest()


@pytest.mark.parametrize("arm", list(ARMS))
def test_two_fields_resolve_to_two_images_with_different_pixels(menu, arm):
    run = PanelRun(menu, menu, menu, menu / "out", arm=arm)
    _, fields, stem = ARMS[arm]
    keys = [run.image_key(f) for f in fields]
    assert keys == [stem.format(f) for f in fields]
    assert len(set(keys)) == 2
    hashes = [_hash(run.load_plane(k)) for k in keys]
    assert hashes[0] != hashes[1]


def test_int_field_id_and_unknown_field(menu):
    run = PanelRun(menu, menu, menu, menu / "out", arm="Nog noDox")
    assert run.image_key(8) == ARMS["Nog noDox"][2].format("08")
    with pytest.raises(KeyError):
        run.image_key("15")  # not a Nog field: must raise, never fall back to the representative image
