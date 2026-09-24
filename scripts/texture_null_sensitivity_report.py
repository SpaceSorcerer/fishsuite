"""Workbook + HTML index for the SENSITIVITY-ONLY texture-matched / rotation UPP run.

Reads a ``qki-assoc --texture-null --rotation-upp`` output directory, checks that
every pre-existing column is value-identical to the delivered qki-assoc runs
(text serialization, all fields present in both), and writes
``texture_null_sensitivity_summary.xlsx``, ``regression_existing_columns.csv``,
``index.html``, ``command.log`` and ``versions.txt`` into ``--out``.
Descriptive only: wells are the replicates; no test is computed.
"""
from __future__ import annotations

import argparse
import html
from pathlib import Path

import pandas as pd

from fishsuite.core import qki_association as qa
from fishsuite.core import repro
from fishsuite.core.texture_null import SENSITIVITY_LABEL

NUC_KEYS = ["threshold_multiplier", "image", "nucleus_id"]
SPOT_KEYS = ["threshold_multiplier", "image", "nucleus_id", "spot_id"]


def _text(paths, columns, keys, images):
    frame = pd.concat([pd.read_csv(p, dtype=str, keep_default_na=False) for p in paths], ignore_index=True)
    frame = frame[frame.image.isin(images)]
    return frame[list(columns)].sort_values(keys).reset_index(drop=True)


def regression(new_dir, delivered_dirs):
    rows = []
    for table, columns, keys in (("qki_association_per_nucleus.csv", qa.NUCLEUS_COLUMNS, NUC_KEYS),
                                 ("qki_association_per_spot.csv", qa.SPOT_COLUMNS, SPOT_KEYS)):
        new_all = pd.read_csv(new_dir / table, dtype=str, keep_default_na=False)
        images = sorted(set(new_all.image))
        new = _text([new_dir / table], columns, keys, images)
        old = _text([d / table for d in delivered_dirs], columns, keys, images)
        same_shape = new.shape == old.shape
        mismatched = ([c for c in columns if not new[c].equals(old[c])] if same_shape else list(columns))
        rows.append(dict(table=table, n_fields=len(images), n_rows_new=len(new), n_rows_delivered=len(old),
                         n_existing_columns=len(columns), n_columns_with_any_text_difference=len(mismatched),
                         columns_with_difference=";".join(mismatched),
                         value_identical=bool(same_shape and not mismatched)))
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assoc-dir", required=True, type=Path)
    parser.add_argument("--delivered", required=True, action="append", type=Path,
                        help="Delivered qki-assoc directory (repeatable).")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    assoc, out = args.assoc_dir.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    per_arm = pd.read_csv(assoc / "texture_null_per_arm.csv")
    per_well = pd.read_csv(assoc / "texture_null_per_well.csv")
    nuclei = pd.read_csv(assoc / "qki_association_per_nucleus.csv", keep_default_na=False, na_values=[""])
    spots = pd.read_csv(assoc / "qki_association_per_spot.csv", keep_default_na=False, na_values=[""])
    nuclei = nuclei[nuclei.threshold_multiplier == 1.0]
    spots = spots[spots.threshold_multiplier == 1.0]
    nuc_cols = ["condition", "well", "image", "nucleus_id", "n_miat_spots", "n_null_effective",
                "mean_uniform_position_percentile_qki", "frac_spots_upp_ge_0p90", "na_reason_mean_uniform_position_percentile_qki",
                *qa.TEXTURE_NUCLEUS_COLUMNS]
    spot_cols = ["condition", "well", "image", "nucleus_id", "spot_id", "footprint_area_px", "footprint_mean_qki",
                 "uniform_position_percentile_qki", *qa.TEXTURE_SPOT_COLUMNS]
    reg = regression(assoc, [d.resolve() for d in args.delivered])
    reg.to_csv(out / "regression_existing_columns.csv", index=False, lineterminator="\n")

    readme = pd.DataFrame({"item": [
        "STATUS", "What", "Texture-matched null", "Rotation null", "Aggregation", "Chance references",
        "Existing columns", "Source run", "Column definitions"], "text": [
        SENSITIVITY_LABEL + ". Descriptive; no test; never a replacement for the uniform-null headline.",
        "Uniform-position percentile score (UPP) of footprint-mean raw QKI at each MIAT punctum under three placement references: uniform (existing headline), texture-matched (new), rotation (existing KEEP-N null).",
        "K placements drawn only from admissible positions (eligible, nucleolus-excluded, whole footprint inside) in the observed placement's within-nucleus stratum of footprint-mean DAPI quantile bin x normalized radial position quantile bin; one merge with the adjacent radial bin below the admissible-position gate, else NA. Parameters: see command.log texture_null_params in the assoc directory.",
        "Existing keep_n_footprint_rotation_null: the spot constellation is rotated about its OWN centroid (not the nuclear centroid), per-spot redraws for placements leaving the mask; 0.5 is nominal only.",
        "Single-plane. Per nucleus mean of spot scores; well = equal-weight mean of nuclei; arm = equal-weight mean of wells. Threshold-free metrics; rows at threshold multiplier 1.0 shown.",
        "Mean 0.5; P(score >= 0.90) = 21/201 at K = 200 (finite-K, under exchangeability with the respective null).",
        "regression sheet: every pre-existing qki-assoc column compared as text with the delivered runs.",
        str(assoc), str(assoc / "qki_association_columns.md")]})
    xlsx = out / "texture_null_sensitivity_summary.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        for name, frame in (("README", readme), ("per_arm", per_arm), ("per_well", per_well),
                            ("per_nucleus", nuclei[nuc_cols]), ("per_spot", spots[spot_cols]),
                            ("regression", reg)):
            frame.to_excel(writer, sheet_name=name, index=False)
            sheet = writer.sheets[name]
            sheet.auto_filter.ref = sheet.dimensions
            sheet.freeze_panes = "A2"

    show = [c for c in per_arm.columns if c in (
        "condition", "n_wells", "n_nuclei", "n_spots", "mean_of_well_mean_upp_uniform",
        "mean_of_well_mean_upp_texture_matched", "mean_of_well_mean_upp_rotation",
        "mean_of_well_frac_ge_0p90_upp_uniform", "mean_of_well_frac_ge_0p90_upp_texture_matched",
        "mean_of_well_frac_ge_0p90_upp_rotation", "texture_na_rate_spots_pooled")]
    files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    body = [f"<h1>MIAT x QKI basal UPP: texture-matched null</h1><p><b>{html.escape(SENSITIVITY_LABEL)}</b>. "
            "Descriptive; wells are the replicates; no test.</p>",
            "<h2>Per arm (equal-weight wells)</h2>", per_arm[show].to_html(index=False, float_format="%.4f"),
            "<h2>Per well</h2>", per_well.to_html(index=False, float_format="%.4f"),
            "<h2>Existing-column regression</h2>", reg.to_html(index=False),
            "<h2>Files</h2><ul>" + "".join(f'<li><a href="{html.escape(f)}">{html.escape(f)}</a></li>'
                                           for f in files + ["index.html"]) + "</ul>"]
    (out / "index.html").write_text("<html><head><meta charset='utf-8'><style>body{font-family:sans-serif}"
                                    "td,th{padding:2px 6px;border:1px solid #ccc}table{border-collapse:collapse}"
                                    "</style></head><body>" + "\n".join(body) + "</body></html>", encoding="utf-8")
    repro.write_command_log(out, assoc / "command.log", out, 0, extra={
        "step": "texture_null_sensitivity_report (workbook + regression; deterministic, no random draws)",
        "status": SENSITIVITY_LABEL,
        "delivered_compared": "; ".join(str(d) for d in args.delivered)})
    repro.write_versions_txt(out, 0)
    print(reg.to_string(index=False))
    return 0 if reg.value_identical.all() else 1


if __name__ == "__main__":
    raise SystemExit(main())
