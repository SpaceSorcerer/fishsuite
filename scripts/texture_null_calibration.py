"""Nucleus-level calibration of the SENSITIVITY-ONLY texture null on realistic
synthetic fields (tests/_texture_null_sim.py). Writes per-nucleus summaries,
the per-(scenario, method) empirical rejection table, command.log and versions.txt."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import _texture_null_sim as sim  # noqa: E402
from fishsuite.core import repro  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-fields", type=int, default=100)
    parser.add_argument("--n-nuclei", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    repro.set_global_seeds(args.seed)
    frames = [sim.nucleus_summaries(s, args.n_fields, args.n_nuclei, args.seed + i)
              for i, s in enumerate(("confound", "independent", "coloc"))]
    nuclei = pd.concat(frames, ignore_index=True)
    table = sim.field_rejections(nuclei)
    nuclei.to_csv(args.out / "calibration_per_nucleus.csv", index=False, lineterminator="\n")
    table.to_csv(args.out / "calibration_rejection_rates.csv", index=False, lineterminator="\n")
    ok = repro.write_command_log(args.out, Path(sim.__file__), args.out, args.seed, extra={
        "design": "fields of independent nuclei; per-field two-sided one-sample t of nucleus means vs 0.5 at alpha 0.05",
        "n_fields": args.n_fields, "n_nuclei_per_field": args.n_nuclei, "K": sim.K,
        "methods": "; ".join(f"{k}={v.as_dict()}" for k, v in sim.DEFAULT_METHODS.items())})
    ok &= repro.write_versions_txt(args.out, args.seed)
    if not ok:
        raise OSError("provenance writer failed")
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
