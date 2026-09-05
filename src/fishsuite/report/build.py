"""``fishsuite report`` — build the condition-versus-condition report for one run.

Produces, under ``<run>/report_<stamp>/`` by default:

* ``REPORT.xlsx``   plain-named sheets, each opening with its own description row
* ``READOUT.md``    ten plain lines, agnostic framing, first line the run path
* ``figures/``      SuperPlots, representative micrographs, and a composite
* ``per_well.csv`` / ``contrasts.csv`` / ``versions.txt`` / ``command.log``

Nothing in the run directory is modified.
"""
from __future__ import annotations

import hashlib
import json
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import aggregate as _agg
from . import endpoints as _ep
from . import figures as _fig
from . import peak_gate as _gate
from . import provenance as _prov
from . import workbook as _wb
from .stats import ALPHA, SEED, fmt_p

random.seed(SEED)
np.random.seed(SEED)


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def library_versions() -> Dict[str, str]:
    import matplotlib
    import scipy

    out = {"python": sys.version.split()[0], "platform": platform.platform(),
           "numpy": np.__version__, "pandas": pd.__version__,
           "scipy": scipy.__version__, "matplotlib": matplotlib.__version__}
    try:
        import statsmodels
        out["statsmodels"] = statsmodels.__version__
    except Exception:                                          # noqa: BLE001
        out["statsmodels"] = "not installed"
    try:
        import openpyxl
        out["openpyxl"] = openpyxl.__version__
    except Exception:                                          # noqa: BLE001
        out["openpyxl"] = "not installed"
    try:
        from .. import __version__ as fsver
        out["fishsuite"] = fsver
    except Exception:                                          # noqa: BLE001
        out["fishsuite"] = "unknown"
    return out


def _engine_head(repo: Optional[Path]) -> str:
    if repo is None:
        return "not supplied"
    try:
        r = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=20)
        return r.stdout.strip() or f"git returned nothing ({r.stderr.strip()})"
    except Exception as exc:                                   # noqa: BLE001
        return f"could not read git HEAD: {type(exc).__name__}: {exc}"



def load_groups_file(path: Path) -> dict:
    """Read a ``report_groups.yaml`` staged beside a run.

    Recognised keys, all optional except ``groups``::

        groups:            {GROUP: [well, well, ...]}
        group_order:       [GROUP, ...]          # first entry is the reference
        reference:         GROUP
        well_from_image:   regex with one capture group
        exclude_fields:    {image name: reason}
        note:              free text, copied into the workbook
    """
    import yaml

    path = Path(path)
    if not path.is_file():
        raise _agg.ReportInputError(f"--groups-file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise _agg.ReportInputError(f"{path} must be a mapping, not {type(raw).__name__}")
    groups = raw.get("groups") or {}
    specs = [f"{g}=" + ",".join(str(w) for w in (wells or []))
             for g, wells in groups.items() if wells]
    order = [str(g) for g in (raw.get("group_order") or list(groups))]
    # A group named in group_order but with no wells listed still needs to appear,
    # because its wells may be recovered from the image name instead.
    for g in order:
        if g not in groups and not any(sp.startswith(f"{g}=") for sp in specs):
            specs.append(f"{g}={g}")
    return {"specs": specs, "group_order": order,
            "reference": raw.get("reference") or (order[0] if order else None),
            "well_from_image": raw.get("well_from_image") or None,
            "exclude_fields": {str(k): str(v) for k, v in
                               (raw.get("exclude_fields") or {}).items()},
            "peak_floors": raw.get("peak_floors") or {},
            "caveat_file": raw.get("caveat_file") or "",
            "nucleus_filter": str(raw.get("nucleus_filter") or "all"),
            "note": str(raw.get("note") or ""), "path": str(path)}

# ------------------------------------------------------------------ sheets


def readme_sheet(run_dir: Path, group_order: Sequence[str], reference: str,
                 well_to_group: Dict[str, str], endpoints: Sequence[_ep.Endpoint],
                 absent: Sequence[str], exclude_fields: Dict[str, str],
                 mde_note: str, native_figures: str, alpha: float,
                 gate_record: Optional[Dict[str, object]] = None,
                 caveat: str = "") -> pd.DataFrame:
    rows: List[Tuple[str, str, str]] = []

    def add(section, item, text):
        rows.append((section, item, text))

    add("What this is", "Report",
        "A condition-versus-condition comparison of one fishsuite run. Each sheet name "
        "says what it holds and each sheet opens with a description row, so nothing "
        "here needs a separate key.")
    add("What this is", "Producing run", str(run_dir))
    add("What this is", "Built", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    add("Design", "Replicate unit",
        "The WELL is the biological replicate. A field of view is a technical replicate "
        "within a well and is never tested on its own. A nucleus is the measurement "
        "unit and is a pseudoreplicate, so nuclei are never tested on their own either.")
    add("Design", "Condition groups",
        "; ".join(f"{g}: " + ", ".join(sorted(w for w, gg in well_to_group.items() if gg == g))
                  for g in group_order) or "no groups were supplied")
    add("Design", "Reference group", reference)
    add("Design", "Comparisons",
        "; ".join(f"{g} minus {reference}" for g in group_order if g != reference)
        or "no comparison is possible with one group")
    add("Statistics", "Gate",
        f"Welch t on well means at alpha {alpha:g}, with Hedges g and a 95 percent "
        "confidence interval on the difference.")
    add("Statistics", "Stars on figures",
        "The star on every figure is the RAW Welch p. The Holm-adjusted p within the "
        "endpoint family is printed in that figure's footnote and in the Contrasts "
        "sheet, so the unadjusted result is never hidden and the adjusted one is never "
        "omitted.")
    add("Statistics", "Multiplicity",
        "Holm-Bonferroni step-down within an endpoint family, where a family is one "
        "by-group sheet. Descriptive-only endpoints, absolute-intensity endpoints and "
        "endpoints absent from the run are excluded from the family and say so in the "
        "holm_exclusion_reason column.")
    add("Statistics", "Minimum detectable effect", mde_note)
    add("Statistics", "Sensitivity, never the gate",
        "Exact permutation of well labels, whose two-sided p has an arithmetic floor "
        "reported alongside it, and Tukey on field means, which treats a technical "
        "replicate as independent and therefore overstates confidence.")
    add("Sheets", "Spots per nucleus by group", _wb.SHEET_DESCRIPTION["Spots per nucleus by group"])
    add("Sheets", "Nuclear fraction by group", _wb.SHEET_DESCRIPTION["Nuclear fraction by group"])
    add("Sheets", "Partner at puncta by group", _wb.SHEET_DESCRIPTION["Partner at puncta by group"])
    add("Sheets", "Per well", _wb.SHEET_DESCRIPTION["Per well"])
    add("Sheets", "Per field", _wb.SHEET_DESCRIPTION["Per field"])
    add("Sheets", "Per nucleus", _wb.SHEET_DESCRIPTION["Per nucleus"])
    add("Sheets", "Contrasts", _wb.SHEET_DESCRIPTION["Contrasts"])
    add("Sheets", "Secondary-only", _wb.SHEET_DESCRIPTION["Secondary-only"])
    add("Sheets", "Run provenance", _wb.SHEET_DESCRIPTION["Run provenance"])
    add("Figures", "Where they are",
        "In the figures folder beside this workbook. Every figure carries the "
        "producing run, the detection thresholds, the segmentation model, the "
        "replicate unit and the test in its footer.")
    add("Figures", "The run's own figures", native_figures)
    add("Data handling", "Endpoints absent from this run",
        ", ".join(absent) if absent else
        "none; every declared endpoint had its column in this run")
    add("Data handling", "Fields excluded",
        "; ".join(f"{k}: {v}" for k, v in sorted(exclude_fields.items()))
        if exclude_fields else "none")
    gate_record = gate_record or {}
    if gate_record.get("floors"):
        add("Data handling", "Post-hoc peak floor",
            "A peak-intensity floor was applied to this run's spots AFTER detection, "
            "which reproduces a gated run because fishsuite's own floor gate is also "
            "post-detection. Boundary is at or above the floor, so a spot exactly at "
            "it is kept. Floors: "
            + "; ".join(f"{k} at {v:g}" for k, v in gate_record["floors"].items())
            + ". Compared against the column " + str(gate_record.get("peak_column", "")) + ".")
        for ch, rec in (gate_record.get("per_channel") or {}).items():
            if rec.get("applied"):
                add("Data handling", f"Peak floor, {ch}",
                    f"{rec['spots_kept']} of {rec['spots_before']} spots kept at floor "
                    f"{rec['floor']:g}; {rec['spots_dropped']} dropped.")
            else:
                add("Data handling", f"Peak floor, {ch}",
                    f"NOT applied: {rec.get('reason', 'no reason recorded')}.")
        if gate_record.get("stale_columns"):
            add("Data handling", "Made stale by the peak floor",
                "These columns cannot be re-derived from spot_metrics.csv after a "
                "floor is applied, so any endpoint reading them reflects the run's "
                "own floor and NOT the one applied here: "
                + ", ".join(gate_record["stale_columns"]) + ".")
    if caveat:
        add("What this is", "Caveat carried from the analysis record", caveat)
    add("Data handling", "Absolute intensity",
        "Absolute intensity in arbitrary units is never comparable as a level claim "
        "across sections, because laser power is retuned per section. Those endpoints "
        "are reported as descriptive only and carry no multiplicity adjustment.")
    for ep in endpoints:
        if ep.note:
            add("Endpoint notes", ep.name, ep.note)
    return pd.DataFrame(rows, columns=["Section", "Item", "Detail"])


def provenance_sheet(data: dict, run_dir: Path, out_dir: Path,
                     group_order: Sequence[str], reference: str,
                     exclude_fields: Dict[str, str], absent: Sequence[str],
                     engine_repo: Optional[Path], preset: Optional[Path],
                     alpha: float) -> pd.DataFrame:
    rows: List[Tuple[str, str]] = []

    def add(k, v):
        rows.append((k, str(v)))

    add("run directory", run_dir)
    add("report directory", out_dir)
    add("built (UTC)", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    add("seed", SEED)
    add("alpha", alpha)
    add("reference group", reference)
    add("group order", ", ".join(group_order))
    add("well to group", "; ".join(f"{w} -> {g}" for w, g in
                                   sorted(data["well_to_group"].items())) or "none supplied")
    add("nucleus filter", data.get("nucleus_filter", "all"))
    add("nuclei segmented in the run", data.get("n_nuclei_all", ""))
    add("nuclei after the nucleus filter", data.get("n_nuclei_after_nucleus_filter", ""))
    add("nuclei before field exclusion", data["n_nuclei_before_field_exclusion"])
    add("nuclei after field exclusion", data["n_nuclei_after_field_exclusion"])
    add("fields excluded", "; ".join(f"{k}: {v}" for k, v in sorted(exclude_fields.items()))
        or "none")
    add("endpoints absent from this run", ", ".join(absent) or "none")

    thr = data.get("thresholds")
    if thr is not None:
        for col in ("rna_bigfish_log_threshold", "protein_bigfish_log_threshold",
                    "protein_threshold_value"):
            if col in thr.columns:
                vals = sorted(pd.to_numeric(thr[col], errors="coerce").dropna().unique().tolist())
                add(f"{col} distinct values", vals)
                add(f"{col} is harmonized across images", len(vals) == 1)
    else:
        add("thresholds.csv", "absent from this run")

    cfg = data.get("cfg") or {}
    res = (cfg.get("config_resolved") or {})
    add("analysis mode", (res.get("channels") or {}).get("analysis_mode", "not recorded"))
    for k, v in (res.get("nuclei") or {}).items():
        if k in ("model", "diameter", "flow_threshold", "cellprob_threshold",
                 "min_area_px", "use_gpu"):
            add(f"segmentation {k}", v)
    prov = cfg.get("provenance") or {}
    for k in ("cellpose_version", "fishsuite_version", "python_version", "started_utc",
              "finished_utc", "command"):
        if k in prov:
            add(f"run {k}", prov[k])

    for name, p in (("engine repo", engine_repo), ("preset", preset)):
        if p is None:
            add(name, "not supplied")
            continue
        p = Path(p)
        add(f"{name} path", p)
        if name == "engine repo":
            add("engine git HEAD", _engine_head(p))
        elif p.is_file():
            add("preset md5", _md5(p))

    for f in ("per_image_summary.csv", "nuclei_metrics.csv", "spot_metrics.csv",
              "thresholds.csv", "run_config.json"):
        src = run_dir / f
        add(f"md5 {f}", _md5(src) if src.is_file() else "file absent")

    for k, v in library_versions().items():
        add(f"version {k}", v)
    return pd.DataFrame(rows, columns=["Item", "Value"])


# ------------------------------------------------------------------ readout


def build_readout(run_dir: Path, out_dir: Path, contrasts: pd.DataFrame,
                  well: pd.DataFrame, group_order: Sequence[str], reference: str,
                  sec: pd.DataFrame, absent: Sequence[str],
                  labels: Dict[str, str], alpha: float, caveat: str = "") -> str:
    """Ten plain lines, agnostic framing, first line the run path."""
    lines: List[str] = []
    lines.append(f"Run reported: {run_dir}")
    if caveat:
        lines.append(caveat.strip().replace("\n", " "))
    lines.append(
        f"Design: {len(group_order)} condition groups ("
        + ", ".join(f"{g}, {int(well.loc[well['group'] == g, 'well_id'].nunique())} wells"
                    for g in group_order)
        + f"); the well is the biological replicate and {reference} is the reference.")
    prim = contrasts[contrasts["primary"] & ~contrasts["endpoint_absent_in_run"]] \
        if len(contrasts) else contrasts
    if len(prim):
        parts = []
        for _, r in prim.iterrows():
            direction = ("higher in" if pd.to_numeric(r["diff"], errors="coerce") > 0
                         else "lower in")
            parts.append(
                f"{r['endpoint_plain']}, mean {float(r['mean_test']):.4g} in "
                f"{r['test_group']} against {float(r['mean_ref']):.4g} in {reference}, "
                f"{direction} {r['test_group']}, raw Welch p {fmt_p(r['p_welch'])} and "
                f"Holm-adjusted {fmt_p(r['p_welch_holm_within_family'])}")
        lines.append("Headline endpoints, on well means: " + "; ".join(parts) + ".")
    else:
        lines.append("No primary endpoint had a column in this run, so no headline "
                     "comparison is available.")
    lines.append(
        "Every star on a figure is the RAW Welch p; the Holm-adjusted p and the "
        "minimum detectable effect sit in that figure's footnote and in the Contrasts "
        "sheet, so neither the unadjusted nor the adjusted result is hidden.")
    gated = contrasts[contrasts["in_holm_family"]] if len(contrasts) else contrasts
    n_raw = int((pd.to_numeric(gated.get("p_welch"), errors="coerce") < alpha).sum()) \
        if len(gated) else 0
    n_holm = int((pd.to_numeric(gated.get("p_welch_holm_within_family"),
                                errors="coerce") < alpha).sum()) if len(gated) else 0
    lines.append(
        f"Across the {len(gated)} gated endpoint comparisons, {n_raw} reach raw "
        f"p below {alpha:g} and {n_holm} survive Holm adjustment within their family.")
    mde = pd.to_numeric(gated.get("mde_hedges_g_at_family_alpha"),
                        errors="coerce").dropna() if len(gated) else pd.Series(dtype=float)
    if len(mde):
        lines.append(
            f"At this number of wells the smallest Hedges g detectable at 80 percent "
            f"power and the family-adjusted alpha ranges from {mde.min():.3g} to "
            f"{mde.max():.3g}, so a null result here is only informative about effects "
            f"larger than that.")
    else:
        lines.append("No minimum detectable effect could be computed, which means no "
                     "comparison had at least two wells in both groups.")
    if len(sec) and "excluded" in sec.columns:
        lines.append(
            f"Secondary-only control fields: {int(len(sec))} present, "
            f"{int(sec['excluded'].sum())} excluded by a stated rule or an operator "
            f"reason recorded in the workbook.")
    else:
        lines.append("No secondary-only control field was present in this run.")
    lines.append(
        "Endpoints whose column this run did not emit: "
        + (", ".join(absent) if absent else "none") +
        ". They are reported as not available rather than dropped.")
    lines.append(
        "Framing: these are measured differences between condition groups on well "
        "means. An enrichment above one in every group is co-distribution, not "
        "evidence of a specific molecular association, and an absolute intensity is "
        "not comparable as a level claim across sections.")
    lines.append(f"Workbook, figures and the per-well and contrast tables are in {out_dir}.")
    return "\n".join(f"{i+1}. {t}" for i, t in enumerate(lines)) + "\n"


# -------------------------------------------------------------------- build


def _nuc_column(ep: _ep.Endpoint) -> Optional[str]:
    return ep.column if ep.level == "nucleus" else None


def build_report(run_dir: Path, out_dir: Optional[Path] = None,
                 groups: Sequence[str] = (), reference: Optional[str] = None,
                 exclude_fields: Optional[Dict[str, str]] = None,
                 well_from_image: Optional[str] = None,
                 group_order: Sequence[str] = (),
                 peak_floors: Optional[Dict[str, float]] = None,
                 caveat: str = "",
                 all_pairs: bool = False,
                 nucleus_filter: str = "all",
                 qc_min_nuclei: int = 5, sec_min_nuclei: int = 10,
                 alpha: float = ALPHA, engine_repo: Optional[Path] = None,
                 preset: Optional[Path] = None, style: str = "brian",
                 make_figures: bool = True, coloc_panel: bool = True,
                 stamp: str = "", argv: Optional[Sequence[str]] = None) -> dict:
    run_dir = Path(run_dir)
    stamp = stamp or datetime.now().strftime("%Y-%m-%d_%H%M")
    out_dir = Path(out_dir) if out_dir else (run_dir / f"report_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    exclude_fields = dict(exclude_fields or {})

    well_to_group, parsed_order = _agg.parse_groups(groups)
    declared_order = list(group_order) or parsed_order
    data = _agg.load_run(run_dir, well_to_group, exclude_fields, well_from_image,
                         nucleus_filter=nucleus_filter)
    labels = _ep.channel_labels(data["cfg"])
    if not declared_order:
        _, declared_order = _agg.groups_from_run_config(data["cfg"])
    group_order_resolved = _agg.resolve_group_order(data["labels"], declared_order)
    if not group_order_resolved:
        raise _agg.ReportInputError("no biological condition group was resolved")
    reference = reference or group_order_resolved[0]
    if reference not in group_order_resolved:
        raise _agg.ReportInputError(
            f"reference group {reference!r} is not one of {group_order_resolved}")

    gate_record: Dict[str, object] = {}
    stale_after_gate: List[str] = []
    if peak_floors:
        spots_path = run_dir / "spot_metrics.csv"
        if not spots_path.is_file():
            raise _agg.ReportInputError(
                f"--peak-floor was given but {spots_path} does not exist, so the "
                "floor cannot be applied post hoc")
        spots = pd.read_csv(spots_path)
        gated, gate_record = _gate.gate_spots(spots, peak_floors)
        vox = pd.to_numeric(data["nuclei"].get("voxel_xy_um"), errors="coerce")
        vox = float(vox.dropna().median()) if vox is not None and vox.notna().any() else None
        data["nuclei"], stale_after_gate = _gate.apply_to_nuclei(
            data["nuclei"], gated, vox)
        # The spot-derived per-nucleus columns must come from the GATED spots too.
        derived = _agg.spot_derived_from_frame(gated, data.get("thresholds"), vox)
        if len(derived):
            drop = [c for c in derived.columns if c in data["nuclei"].columns
                    and c not in ("image", "nucleus_id")]
            data["nuclei"] = data["nuclei"].drop(columns=drop).merge(
                derived, on=["image", "nucleus_id"], how="left")
        gate_record["floors"] = dict(peak_floors)
        gate_record["stale_columns"] = list(stale_after_gate)

    endpoints, absent = _ep.resolve(data["nuclei"], data["per_image"])
    field = _agg.per_field_long(data["nuclei"], data["per_image"], endpoints,
                                data["labels"], qc_min_nuclei)
    well = _agg.per_well_long(field)
    contrasts = _agg.build_contrasts(well, field, endpoints, absent, group_order_resolved,
                                     reference, labels, alpha,
                                     all_pairs=all_pairs or len(group_order_resolved) > 2)
    sec = _agg.secondary_only_table(data["nuclei"], data["per_image"], data["labels"],
                                    exclude_fields, sec_min_nuclei)

    nuc_cols = (["image", "condition", "group", "well_id", "field", "secondary_only",
                 "nucleus_id", "nucleus_area_px"]
                + [ep.column for ep in endpoints if ep.level == "nucleus"])
    nuc_cols = [c for c in dict.fromkeys(nuc_cols) if c in data["nuclei"].columns]
    per_nucleus = data["nuclei"][nuc_cols].copy()
    bio_nucleus = per_nucleus[~per_nucleus["secondary_only"]].copy() \
        if "secondary_only" in per_nucleus.columns else per_nucleus.copy()

    mde_all = pd.to_numeric(contrasts.get("mde_hedges_g_at_family_alpha"),
                            errors="coerce").dropna() if len(contrasts) else pd.Series(dtype=float)
    mde_note = (
        f"The smallest Hedges g detectable at 80 percent power, two-sided, at the "
        f"family-adjusted alpha ranges from {mde_all.min():.3g} to {mde_all.max():.3g} "
        f"across the gated comparisons. A comparison that does not reach significance "
        f"is only informative about effects at least that large."
        if len(mde_all) else
        "No minimum detectable effect could be computed; no comparison had at least "
        "two wells in both groups.")
    native = (f"The run's own figures are in {run_dir / 'figures'} and are not "
              "reproduced here; this report adds the by-group comparison on top of "
              "them.") if (run_dir / "figures").is_dir() else \
             "This run wrote no figures folder of its own."

    sheets = {
        "Read me": readme_sheet(run_dir, group_order_resolved, reference, data["well_to_group"],
                                endpoints, absent, exclude_fields, mde_note, native, alpha),
        "Spots per nucleus by group": _agg.family_sheet(contrasts, well, "detection",
                                                        group_order_resolved),
        "Nuclear fraction by group": _agg.family_sheet(contrasts, well, "localization",
                                                       group_order_resolved),
        "Partner at puncta by group": _agg.family_sheet(contrasts, well, "partner",
                                                        group_order_resolved),
        "Per well": well,
        "Per field": field,
        "Per nucleus": per_nucleus,
        "Contrasts": contrasts,
        "Secondary-only": sec,
        "Run provenance": provenance_sheet(data, run_dir, out_dir, group_order_resolved,
                                           reference, exclude_fields, absent,
                                           engine_repo, preset, alpha),
    }
    strike = {"Secondary-only": list(sec.index[sec["excluded"]])
              if "excluded" in sec.columns else []}
    xlsx = _wb.write(out_dir / "REPORT.xlsx", sheets, strike)

    well.to_csv(out_dir / "per_well.csv", index=False)
    contrasts.to_csv(out_dir / "contrasts.csv", index=False)
    (out_dir / "READOUT.md").write_text(
        build_readout(run_dir, out_dir, contrasts, well, group_order_resolved, reference, sec,
                      absent, labels, alpha, caveat), encoding="utf-8")

    source = _prov.write_source_run(
        run_dir, out_dir, preset=preset,
        groups={w: g for w, g in data["well_to_group"].items()},
        group_order=group_order_resolved, reference=reference)

    figs: List[dict] = []
    if make_figures:
        figs = render_figures(out_dir / "figures", run_dir, data, endpoints, absent,
                              well, field, bio_nucleus, contrasts, group_order_resolved,
                              reference, exclude_fields, labels, alpha)

    panel = None
    if coloc_panel:
        panel = run_coloc_standard_panel(run_dir, out_dir)

    (out_dir / "versions.txt").write_text(
        "\n".join(f"{k}: {v}" for k, v in library_versions().items())
        + f"\nseed: {SEED}\nbuilt_utc: "
        + datetime.now(timezone.utc).isoformat(timespec="seconds") + "\n",
        encoding="utf-8")
    with open(out_dir / "command.log", "a", encoding="utf-8") as fh:
        fh.write(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}\t"
                 f"{sys.executable} {' '.join(argv or sys.argv)}\n")

    return dict(out_dir=out_dir, xlsx=xlsx, contrasts=contrasts, well=well, field=field,
                sec=sec, absent=absent, group_order=group_order_resolved, reference=reference,
                figures=figs, coloc_panel=panel, endpoints=endpoints, labels=labels,
                source_run=source, peak_gate=gate_record)


def render_figures(fig_dir: Path, run_dir: Path, data: dict,
                   endpoints: Sequence[_ep.Endpoint], absent: Sequence[str],
                   well: pd.DataFrame, field: pd.DataFrame, per_nucleus: pd.DataFrame,
                   contrasts: pd.DataFrame, group_order: Sequence[str],
                   reference: str, exclude_fields: Dict[str, str],
                   labels: Dict[str, str], alpha: float) -> List[dict]:
    _fig.set_style()
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    ctx = _fig.FigureContext(run_dir, data["cfg"], data.get("thresholds"), group_order,
                             reference, alpha, exclude_fields, labels)
    manifest: List[dict] = []
    absent_set = set(absent)
    drawn = [ep for ep in endpoints
             if ep.name not in absent_set and not ep.name.endswith("_allnuclei")]
    # Only endpoints with at least one finite well mean are worth a figure.
    have = set(well.loc[np.isfinite(pd.to_numeric(well["well_mean_of_field_values"],
                                                  errors="coerce")), "endpoint"]) \
        if len(well) else set()
    drawn = [ep for ep in drawn if ep.name in have]

    for i, ep in enumerate(drawn, start=1):
        hline = 1.0 if "enrichment" in ep.name and "null" in (ep.unit + ep.name) else None
        _fig.superplot_standalone(
            ctx, ep.name, ep.pretty(labels), f"{ep.pretty(labels)}\n({ep.unit})",
            well, field, per_nucleus, contrasts, _nuc_column(ep), fig_dir,
            f"fig{i:02d}_{ep.name}", manifest,
            hline_at=hline, hline_label="no enrichment" if hline else "")

    pub_dir = _fig.publication_image_dir(run_dir)
    luts, lut_source = _fig.read_luts(run_dir, pub_dir)
    anchor = next((ep for ep in drawn if ep.primary and ep.level == "nucleus"), None)
    um_per_px = float(pd.to_numeric(data["nuclei"].get("voxel_xy_um"),
                                    errors="coerce").dropna().median()) \
        if "voxel_xy_um" in data["nuclei"].columns else float("nan")
    panels: List[dict] = []
    if anchor is not None and pub_dir is not None:
        panels = _fig.collect_panels(ctx, field, per_nucleus, anchor.name,
                                     _nuc_column(anchor), run_dir, pub_dir, um_per_px)
        _fig.micrograph_standalone(ctx, panels, luts, lut_source, fig_dir,
                                   "fig_representative_micrographs", manifest,
                                   anchor.pretty(labels))

    specs = [dict(endpoint=ep.name, ylabel=f"{ep.pretty(labels)}\n({ep.unit})",
                  nuc_column=_nuc_column(ep),
                  hline_at=1.0 if ("enrichment" in ep.name and "null" in ep.name) else None,
                  hline_label="no enrichment" if ("enrichment" in ep.name
                                                  and "null" in ep.name) else "")
             for ep in drawn if ep.primary][:4]
    if specs:
        _fig.composite_main(ctx, specs, panels, luts, well, field, per_nucleus,
                            contrasts, fig_dir, manifest, stem="FIG_MAIN")

    index = ["# Figure index", "",
             f"Produced by `fishsuite report` from run `{run_dir}`.", ""]
    index += [f"- `{m['png']}` / `{m['svg']}` — {m['description']} Source: {m['source']}"
              for m in manifest]
    (fig_dir / "FIGURE_INDEX.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    return manifest


def run_coloc_standard_panel(run_dir: Path, out_dir: Path) -> Optional[dict]:
    """Call the standard colocalization panel when the run is a coloc run.

    Brian's rule: every rna_rna or rna_protein report carries the standard panel
    alongside any null-based enrichment.
    """
    cfg_path = Path(run_dir) / "run_config.json"
    try:
        mode = json.loads(cfg_path.read_text(encoding="utf-8"))[
            "config_resolved"]["channels"]["analysis_mode"]
    except Exception:                                          # noqa: BLE001
        return None
    if mode not in ("rna_rna", "rna_protein"):
        return {"skipped": f"analysis mode {mode} is not a colocalization mode"}
    script = Path(__file__).resolve().parents[3] / "scripts" / "coloc_standard_panel.py"
    if not script.is_file():
        return {"skipped": f"coloc_standard_panel.py not found at {script}"}
    dest = Path(out_dir) / "coloc_standard_panel"
    log = Path(out_dir) / "coloc_standard_panel.log"
    cmd = [sys.executable, str(script), "--run", str(run_dir), "--out", str(dest),
           "--report-per-well", str(Path(out_dir) / "per_well.csv"), "--seed", str(SEED)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        log.write_text((r.stdout or "") + "\n" + (r.stderr or ""), encoding="utf-8")
        return {"cmd": " ".join(cmd), "returncode": r.returncode, "out_dir": str(dest),
                "log": str(log)}
    except Exception as exc:                                   # noqa: BLE001
        log.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        return {"cmd": " ".join(cmd), "error": f"{type(exc).__name__}: {exc}",
                "log": str(log)}
