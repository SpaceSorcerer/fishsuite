"""Workbook + HTML index for the SENSITIVITY-ONLY texture-matched / rotation UPP runs.

Reads one or more ``qki-assoc --texture-null`` output directories (one per
matching variant, ``--variant LABEL=DIR``), checks that every pre-existing
column is value-identical to the delivered qki-assoc runs, and writes
``texture_null_sensitivity_summary.xlsx``, ``regression_existing_columns.csv``,
``command.log``, ``versions.txt`` and finally ``index.html`` (which lists
every file present after provenance was written) into ``--out``.
Descriptive only: wells are the replicates; no test is computed.
"""
from __future__ import annotations

import argparse
import html
from pathlib import Path

import pandas as pd

from fishsuite.core import qki_association as qa
from fishsuite.core import repro
from fishsuite.core.texture_null import SENSITIVITY_LABEL, is_sensitivity_column

NUC_KEYS = ["threshold_multiplier", "image", "nucleus_id"]
SPOT_KEYS = ["threshold_multiplier", "image", "nucleus_id", "spot_id"]


def _read(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def regression(new_dir, delivered_dirs, variant):
    """Compare every delivered column (the delivered header is the baseline,
    validated against the current schema) as text on the images in the new run."""
    rows = []
    for table, schema, keys in (("qki_association_per_nucleus.csv", qa.NUCLEUS_COLUMNS, NUC_KEYS),
                                ("qki_association_per_spot.csv", qa.SPOT_COLUMNS, SPOT_KEYS)):
        new_all = _read(new_dir / table)
        delivered = [_read(d / table) for d in delivered_dirs]
        headers = {tuple(frame.columns) for frame in delivered}
        if len(headers) != 1:
            raise ValueError(f"delivered {table} headers differ between runs")
        baseline = list(next(iter(headers)))
        if baseline != list(schema):
            raise ValueError(f"delivered {table} header != current schema; "
                             f"only in delivered: {sorted(set(baseline) - set(schema))}, "
                             f"only in schema: {sorted(set(schema) - set(baseline))}")
        if list(new_all.columns[:len(baseline)]) != baseline:
            raise ValueError(f"new {table} does not start with the delivered header")
        if not all(is_sensitivity_column(c) for c in new_all.columns[len(baseline):]):
            raise ValueError(f"new {table} appends non-sensitivity columns")
        old_all = pd.concat(delivered, ignore_index=True)
        images = sorted(set(new_all.image))
        if not set(images) <= set(old_all.image):
            raise ValueError(f"{table}: images absent from the delivered runs")
        for frame in (new_all, old_all):
            if frame.duplicated(keys).any():
                raise ValueError(f"{table}: duplicate keys {keys}")
        new = new_all[new_all.image.isin(images)][baseline].sort_values(keys).reset_index(drop=True)
        old = old_all[old_all.image.isin(images)][baseline].sort_values(keys).reset_index(drop=True)
        same_keys = new[keys].equals(old[keys])
        mismatched = [c for c in baseline if not new[c].equals(old[c])] if same_keys else list(baseline)
        rows.append(dict(variant=variant, table=table, n_fields=len(images), n_rows_new=len(new),
                         n_rows_delivered=len(old), n_delivered_columns=len(baseline),
                         keys_identical=bool(same_keys), n_columns_with_any_text_difference=len(mismatched),
                         columns_with_difference=";".join(mismatched),
                         value_identical=bool(same_keys and not mismatched)))
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, action="append",
                        help="LABEL=qki-assoc output dir run with --texture-null (repeatable).")
    parser.add_argument("--delivered", required=True, action="append", type=Path,
                        help="Delivered qki-assoc directory (repeatable).")
    parser.add_argument("--calibration", type=Path, default=None,
                        help="Directory with calibration_rejection_rates.csv (synthetic calibration).")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    variants = [(v.split("=", 1)[0], Path(v.split("=", 1)[1]).resolve()) for v in args.variant]

    arms, wells, regs, nuclei_frames, spot_frames = [], [], [], [], []
    for label, directory in variants:
        arm = pd.read_csv(directory / "texture_null_per_arm.csv")
        well = pd.read_csv(directory / "texture_null_per_well.csv")
        arms.append(arm.assign(variant=label))
        wells.append(well.assign(variant=label))
        regs.append(regression(directory, [d.resolve() for d in args.delivered], label))
        nuc = pd.read_csv(directory / "qki_association_per_nucleus.csv", keep_default_na=False, na_values=[""])
        spt = pd.read_csv(directory / "qki_association_per_spot.csv", keep_default_na=False, na_values=[""])
        nuc_cols = ["condition", "well", "image", "nucleus_id", "n_miat_spots",
                    "mean_uniform_position_percentile_qki", "frac_spots_upp_ge_0p90",
                    *[c for c in nuc.columns if is_sensitivity_column(c)]]
        spot_cols = ["condition", "well", "image", "nucleus_id", "spot_id", "footprint_mean_qki",
                     "uniform_position_percentile_qki", *[c for c in spt.columns if is_sensitivity_column(c)]]
        nuclei_frames.append(nuc[nuc.threshold_multiplier == 1.0][nuc_cols].assign(variant=label))
        spot_frames.append(spt[spt.threshold_multiplier == 1.0][spot_cols].assign(variant=label))
    per_arm = pd.concat(arms, ignore_index=True)
    per_well = pd.concat(wells, ignore_index=True)
    reg = pd.concat(regs, ignore_index=True)
    reg.to_csv(out / "regression_existing_columns.csv", index=False, lineterminator="\n")
    compact = per_arm[["variant", "condition", "calibration", "n_wells", "n_nuclei", "n_spots",
                       "mean_of_well_mean_upp_uniform", "mean_of_well_mean_upp_texture_matched",
                       "mean_of_well_frac_ge_0p90_upp_uniform", "mean_of_well_frac_ge_0p90_upp_texture_matched",
                       "texture_na_rate_spots_pooled", "mean_of_support_median_positions",
                       "mean_of_balance_uniform_dapi_q", "mean_of_balance_matched_dapi_q",
                       "mean_of_balance_uniform_radial_q", "mean_of_balance_matched_radial_q"]]
    compact.to_csv(out / "texture_null_variants_per_arm.csv", index=False, lineterminator="\n")

    readme = pd.DataFrame({"item": [
        "STATUS", "What", "Variants", "strata", "knn", "Rotation null", "Balance", "Support", "Aggregation",
        "Chance references", "Existing columns", "Calibration"], "text": [
        SENSITIVITY_LABEL + ". Descriptive; no test; never a replacement for the uniform-null headline.",
        "Uniform-position percentile score (UPP) of footprint-mean raw QKI at each MIAT punctum under the uniform placement null (existing headline) and texture-matched placement nulls (new).",
        "; ".join(f"{label} = {directory}" for label, directory in variants),
        "Fixed within-nucleus quantile cells of footprint-mean DAPI x normalized radial position, defined from the admissible positions alone; a cell below the admissible-position gate is NA (SPARSE_STRATUM), never merged. Finite-K references exact under uniform placement within the cell.",
        "The k admissible positions nearest the observed one in (DAPI quantile, radial quantile) space; centred on the observation, so references are nominal.",
        "Existing keep_n_footprint_rotation_null (pivot = spot-constellation centroid, per-spot redraws); in the strata_5x5 variant only; nominal references.",
        "balance_uniform_* = mean(observed quantile - 0.5) (the uniform null's expected quantile); balance_matched_* = mean(observed quantile - mean quantile of the K matched draws). abs_* = mean absolute value.",
        "support_* = admissible positions in the matched set of scored spots (min / median per well).",
        "Single-plane. Per nucleus mean of spot scores; well = equal-weight mean of nuclei; arm = equal-weight mean of wells. Rows at threshold multiplier 1.0.",
        "Mean 0.5; P(score >= 0.90) = 21/201 at K = 200.",
        "regression sheet: every delivered qki-assoc column compared as text with the delivered runs (delivered header validated against the schema).",
        str(args.calibration) if args.calibration else "not supplied"]})
    sheets = [("README", readme), ("per_arm_compact", compact), ("per_arm", per_arm), ("per_well", per_well),
              ("per_nucleus", pd.concat(nuclei_frames, ignore_index=True)),
              ("per_spot", pd.concat(spot_frames, ignore_index=True)), ("regression", reg)]
    if args.calibration:
        sheets.append(("synthetic_calibration", pd.read_csv(args.calibration / "calibration_rejection_rates.csv")))
    with pd.ExcelWriter(out / "texture_null_sensitivity_summary.xlsx", engine="openpyxl") as writer:
        for name, frame in sheets:
            frame.to_excel(writer, sheet_name=name, index=False)
            sheet = writer.sheets[name]
            sheet.auto_filter.ref = sheet.dimensions
            sheet.freeze_panes = "A2"
    ok = repro.write_command_log(out, variants[0][1] / "command.log", out, 0, extra={
        "step": "texture_null_sensitivity_report (workbook + regression; deterministic, no random draws)",
        "status": SENSITIVITY_LABEL,
        "variants": "; ".join(f"{label}={directory}" for label, directory in variants),
        "delivered_compared": "; ".join(str(d) for d in args.delivered)})
    ok &= repro.write_versions_txt(out, 0)
    if not ok:
        raise OSError("provenance writer failed (command.log / versions.txt)")
    files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    body = [f"<h1>MIAT x QKI basal UPP: texture-matched nulls</h1><p><b>{html.escape(SENSITIVITY_LABEL)}</b>. "
            "Descriptive; wells are the replicates; no test.</p>",
            "<h2>Per arm and variant (equal-weight wells)</h2>", compact.to_html(index=False, float_format="%.4f"),
            "<h2>Per well</h2>", per_well.to_html(index=False, float_format="%.4f"),
            "<h2>Existing-column regression</h2>", reg.to_html(index=False)]
    if args.calibration:
        body += ["<h2>Synthetic nucleus-level calibration</h2>",
                 pd.read_csv(args.calibration / "calibration_rejection_rates.csv").to_html(index=False,
                                                                                         float_format="%.4f")]
    body.append("<h2>Files</h2><ul>" + "".join(f'<li><a href="{html.escape(f)}">{html.escape(f)}</a></li>'
                                               for f in files + ["index.html"]) + "</ul>")
    (out / "index.html").write_text("<html><head><meta charset='utf-8'><style>body{font-family:sans-serif}"
                                    "td,th{padding:2px 6px;border:1px solid #ccc}table{border-collapse:collapse}"
                                    "</style></head><body>" + "\n".join(body) + "</body></html>", encoding="utf-8")
    print(reg.to_string(index=False))
    return 0 if reg.value_identical.all() else 1


if __name__ == "__main__":
    raise SystemExit(main())
