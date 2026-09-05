"""Run loading and the nucleus -> field -> well -> group hierarchy.

The replicate structure this layer encodes:

* the NUCLEUS is the measurement unit,
* the FIELD of view (one image) is a technical replicate,
* the WELL (fishsuite's ``condition``) is the biological replicate,
* the GROUP is the condition several wells belong to, and is what gets compared.

Every gate runs on well means. Nothing here pools nuclei across wells.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import endpoints as _ep
from .stats import (ALPHA, exact_permutation, holm, mde_hedges_g, tukey_two_group,
                    welch)

SEC_ONLY_GROUP = "Secondary-only"
FIELD_RE = re.compile(r"_(\d{2,3})(?:\.[A-Za-z0-9]+)?$")

REQUIRED_FILES = ("per_image_summary.csv", "nuclei_metrics.csv", "run_config.json")


class ReportInputError(RuntimeError):
    """Raised when a run directory cannot support a report."""


# --------------------------------------------------------------------- groups


def parse_groups(specs: Sequence[str]) -> Tuple[Dict[str, str], List[str]]:
    """``["WT=WT_1,WT_2", "QKI-KO=KO_1,KO_2"]`` -> (well -> group, group order)."""
    well_to_group: Dict[str, str] = {}
    order: List[str] = []
    for spec in specs or ():
        if "=" not in spec:
            raise ReportInputError(
                f"--groups entry {spec!r} is not NAME=well1,well2; the group name and "
                "its wells must be separated by one '='")
        name, wells = spec.split("=", 1)
        name = name.strip()
        members = [w.strip() for w in wells.split(",") if w.strip()]
        if not name or not members:
            raise ReportInputError(f"--groups entry {spec!r} has an empty name or well list")
        if name not in order:
            order.append(name)
        for w in members:
            if w in well_to_group and well_to_group[w] != name:
                raise ReportInputError(
                    f"well {w!r} is assigned to both {well_to_group[w]!r} and {name!r}")
            well_to_group[w] = name
    return well_to_group, order


def groups_from_run_config(cfg: dict) -> Tuple[Dict[str, str], List[str]]:
    """Group map recorded by the run itself, if the config carried one."""
    well_to_group: Dict[str, str] = {}
    order: List[str] = []
    try:
        block = cfg["config_resolved"]["conditions"]
    except Exception:                                          # noqa: BLE001
        block = {}
    raw = (block or {}).get("groups") or {}
    for name, wells in raw.items():
        if name not in order:
            order.append(name)
        for w in wells or ():
            well_to_group[str(w)] = str(name)
    declared = (block or {}).get("group_order") or []
    if declared:
        order = [g for g in declared if g in order] + [g for g in order if g not in declared]
    return well_to_group, order


# ---------------------------------------------------------------- run loading


def label_frame(per_image: pd.DataFrame, well_to_group: Dict[str, str],
                exclude_fields: Dict[str, str],
                well_from_image: Optional[str] = None) -> pd.DataFrame:
    """One row per image: its well, its group, whether it is a control, its field id.

    By default the WELL is the run's ``condition``. Some runs record the LINE as
    the condition and carry the well only in the file name; ``well_from_image`` is
    a regular expression with ONE capture group, matched against the image name,
    that recovers the well in that case. Every non-control image must match, or the
    grouping is refused rather than silently collapsing a line into one well.
    """
    need = [c for c in ("image", "condition", "secondary_only") if c not in per_image.columns]
    if need:
        raise ReportInputError(
            f"per_image_summary.csv is missing {need}; this run cannot be grouped")
    lab = per_image[["image", "condition", "secondary_only"]].copy()
    lab["secondary_only"] = lab["secondary_only"].astype(bool)
    if well_from_image:
        rx = re.compile(well_from_image)
        if rx.groups != 1:
            raise ReportInputError(
                f"--well-from-image must have exactly one capture group; "
                f"{well_from_image!r} has {rx.groups}")
        found = lab["image"].astype(str).str.extract(rx, expand=False)
        missed = lab.loc[(~lab["secondary_only"]) & found.isna(), "image"].tolist()
        if missed:
            raise ReportInputError(
                f"--well-from-image {well_from_image!r} matched no well in "
                f"{len(missed)} biological image(s), first: {missed[:3]}")
        lab["well_id"] = np.where(lab["secondary_only"], pd.NA, found)
    else:
        lab["well_id"] = np.where(lab["secondary_only"], pd.NA, lab["condition"])
    lab["group"] = np.where(
        lab["secondary_only"], SEC_ONLY_GROUP,
        lab["condition"].map(lambda c: well_to_group.get(str(c), str(c))))
    lab["field"] = lab["image"].astype(str).str.extract(FIELD_RE, expand=False)
    lab["sec_field"] = np.where(lab["secondary_only"], lab["image"].astype(str), pd.NA)
    lab["excluded_field"] = lab["image"].astype(str).isin(exclude_fields)
    lab["exclusion_reason"] = lab["image"].astype(str).map(
        lambda i: exclude_fields.get(i, ""))
    return lab


def spot_derived_from_frame(rna_or_all: pd.DataFrame,
                            thresholds: Optional[pd.DataFrame],
                            voxel_xy_um: Optional[float] = None) -> pd.DataFrame:
    """The per-nucleus spot aggregates, from a spot frame already in memory.

    Used when a post-hoc peak floor has produced a GATED spot frame: the derived
    columns must come from the surviving spots, not from the file on disk.
    """
    need = {"image", "channel", "nucleus_id", "in_nucleus"}
    if rna_or_all is None or not len(rna_or_all) or not need <= set(rna_or_all.columns):
        return pd.DataFrame(columns=["image", "nucleus_id"])
    return _spot_aggregates(rna_or_all, thresholds, voxel_xy_um)


def spot_derived_per_nucleus(run_dir: Path, thresholds: Optional[pd.DataFrame],
                             voxel_xy_um: Optional[float] = None) -> pd.DataFrame:
    """Per-nucleus aggregates over that nucleus's NUCLEAR anchor (rna1) puncta.

    The exact-footprint columns the engine writes per spot are named for the
    original MIAT/QKI pair (``qki_at_miat_footprint``, ``miat_footprint_area_px``);
    they are role columns, so they are read under those names and emitted under
    role-neutral ones.

    ``miat_footprint_area_px`` is the punctum's own half-maximum footprint and is
    the size measurement. It is converted to square micrometres with the run's own
    voxel area, and restated as the diameter of a circle of the same area.
    """
    path = run_dir / "spot_metrics.csv"
    if not path.is_file():
        return pd.DataFrame(columns=["image", "nucleus_id"])
    head = pd.read_csv(path, nrows=0)
    want = ["image", "channel", "nucleus_id", "in_nucleus", "spot_fwhm_px",
            "spot_diameter_um", "qki_at_miat_footprint", "qki_footprint_enrichment",
            "miat_footprint_area_px", "in_cytoplasm", "nn_distance_um",
            "paired_at_0p3um"]
    cols = [c for c in want if c in head.columns]
    if not {"image", "channel", "nucleus_id", "in_nucleus"} <= set(cols):
        return pd.DataFrame(columns=["image", "nucleus_id"])
    spots = pd.read_csv(path, usecols=cols)
    return _spot_aggregates(spots, thresholds, voxel_xy_um)


def _spot_aggregates(spots: pd.DataFrame, thresholds: Optional[pd.DataFrame],
                     voxel_xy_um: Optional[float]) -> pd.DataFrame:
    """Per-nucleus aggregates over the rna1 puncta of ``spots``.

    Size and footprint use the NUCLEAR puncta only, which is the set those
    measurements are defined on. The pairing endpoints use every punctum of the
    nucleus, nuclear and cytoplasmic, because that is the denominator the engine
    uses and the one the report's definition row states.
    """
    rna = spots[spots["channel"].eq("rna1") & spots["in_nucleus"].astype(bool)].copy()
    if rna.empty:
        return pd.DataFrame(columns=["image", "nucleus_id"])

    agg: Dict[str, tuple] = {}
    # Pairing endpoints, recomputed per nucleus over the SAME spot set the counts
    # use. The denominator is every anchor punctum of that nucleus, nuclear and
    # cytoplasmic, matching the engine's own aggregation.
    if "spot_fwhm_px" in rna.columns:
        agg["rna1_spot_fwhm_px"] = ("spot_fwhm_px", "mean")
    if "spot_diameter_um" in rna.columns:
        agg["rna1_spot_diameter_um"] = ("spot_diameter_um", "mean")
    if "qki_at_miat_footprint" in rna.columns:
        agg["partner_mean_in_exact_rna1_footprint"] = ("qki_at_miat_footprint", "mean")
        agg["rna1_nuclear_puncta_with_footprint"] = ("qki_at_miat_footprint", "size")
        if thresholds is not None and "protein_threshold_value" in thresholds.columns:
            thr = thresholds[["image", "protein_threshold_value"]]
            rna = rna.merge(thr, on="image", how="left", validate="many_to_one")
            rna["_positive"] = (
                rna["qki_at_miat_footprint"] >= rna["protein_threshold_value"]).astype(float)
            agg["fraction_rna1_puncta_partner_positive_exact_footprint"] = ("_positive", "mean")
    if "qki_footprint_enrichment" in rna.columns:
        agg["partner_enrichment_in_exact_rna1_footprint"] = ("qki_footprint_enrichment", "mean")
    if "miat_footprint_area_px" in rna.columns and voxel_xy_um and voxel_xy_um > 0:
        area_um2 = (pd.to_numeric(rna["miat_footprint_area_px"], errors="coerce")
                    * float(voxel_xy_um) ** 2)
        rna["_fp_area_um2"] = area_um2
        # Equivalent diameter of a circle with the same area. Averaged per nucleus
        # AFTER the per-punctum conversion, so it is the mean punctum diameter and
        # not the diameter of the mean area, which are not the same number.
        rna["_fp_eqdiam_um"] = 2.0 * np.sqrt(area_um2 / np.pi)
        agg["rna1_punctum_footprint_area_um2"] = ("_fp_area_um2", "mean")
        agg["rna1_punctum_equivalent_diameter_um"] = ("_fp_eqdiam_um", "mean")
    out = (rna.groupby(["image", "nucleus_id"], sort=False).agg(**agg).reset_index()
           if agg else pd.DataFrame(columns=["image", "nucleus_id"]))

    # Pairing is aggregated over EVERY anchor punctum of the nucleus, nuclear and
    # cytoplasmic, which is the denominator the engine uses and the one the
    # workbook's definition row states. Restricting it to nuclear puncta gives a
    # different number for the same name.
    allr = spots[spots["channel"].eq("rna1")].copy()
    pair_col = next((c for c in allr.columns if c.startswith("paired_at_")), None)
    pagg: Dict[str, tuple] = {}
    if pair_col:
        allr["_paired"] = pd.to_numeric(allr[pair_col], errors="coerce")
        pagg["paired_fraction_rna1_at_0p3um"] = ("_paired", "mean")
    if "nn_distance_um" in allr.columns:
        allr["_nn"] = pd.to_numeric(allr["nn_distance_um"], errors="coerce").replace(
            [np.inf, -np.inf], np.nan)
        pagg["median_nn_distance_rna1_um"] = ("_nn", "median")
    if pagg and len(allr):
        pair = (allr.groupby(["image", "nucleus_id"], sort=False).agg(**pagg)
                .reset_index())
        out = (pair if not len(out)
               else out.merge(pair, on=["image", "nucleus_id"], how="outer"))
    return out


def load_run(run_dir: Path, well_to_group: Dict[str, str],
             exclude_fields: Dict[str, str],
             well_from_image: Optional[str] = None,
             nucleus_filter: str = "all") -> dict:
    run_dir = Path(run_dir)
    missing = [f for f in REQUIRED_FILES if not (run_dir / f).is_file()]
    if missing:
        raise ReportInputError(f"run dir {run_dir} is missing: {missing}")
    per_image = pd.read_csv(run_dir / "per_image_summary.csv")
    nuclei = pd.read_csv(run_dir / "nuclei_metrics.csv")
    thresholds = (pd.read_csv(run_dir / "thresholds.csv")
                  if (run_dir / "thresholds.csv").is_file() else None)
    cfg = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))

    if not well_to_group:
        well_to_group, _ = groups_from_run_config(cfg)
    labels = label_frame(per_image, well_to_group, exclude_fields, well_from_image)
    _vox = pd.to_numeric(nuclei.get("voxel_xy_um"), errors="coerce")
    voxel_xy_um = (float(_vox.dropna().median())
                   if _vox is not None and _vox.notna().any() else None)
    derived = spot_derived_per_nucleus(run_dir, thresholds, voxel_xy_um)

    nuc = nuclei.drop(columns=[c for c in ("condition", "secondary_only", "group")
                               if c in nuclei.columns], errors="ignore")
    nuc = nuc.merge(labels, on="image", how="left", validate="many_to_one")
    if len(derived):
        nuc = nuc.merge(derived, on=["image", "nucleus_id"], how="left",
                        validate="one_to_one")

    vox = pd.to_numeric(nuc.get("voxel_xy_um"), errors="coerce")
    if vox is not None and not vox.isna().all() and "nucleus_area_px" in nuc.columns:
        nuc["nucleus_area_um2"] = pd.to_numeric(nuc["nucleus_area_px"],
                                                errors="coerce") * vox ** 2
        for src_col, out_col in (("n_spots_rna1", "rna1_spots_per_um2"),
                                 ("n_spots_rna2", "rna2_spots_per_um2")):
            if src_col in nuc.columns:
                nuc[out_col] = (pd.to_numeric(nuc[src_col], errors="coerce")
                                / nuc["nucleus_area_um2"])
    for flag in ("rotation_null_usable", "rotation_null_usable_at_protein_spots"):
        if flag in nuc.columns:
            nuc[flag] = nuc[flag].astype(str).str.strip().str.lower().isin(
                ["true", "1", "1.0", "yes"])

    n_nuclei_all = int(len(nuc))
    n_after_nucleus_filter = n_nuclei_all
    if nucleus_filter == "sampled":
        if "sampled_in_analysis" not in nuc.columns:
            raise ReportInputError(
                "--nucleus-filter sampled needs the sampled_in_analysis column, "
                "which this run did not emit; it is written only when the run used "
                "fixed-N nucleus sampling")
        flag = nuc["sampled_in_analysis"].astype(str).str.strip().str.lower().isin(
            ["true", "1", "1.0", "yes"])
        nuc = nuc[flag | nuc["secondary_only"].astype(bool)].copy()
        n_after_nucleus_filter = int(len(nuc))
    elif nucleus_filter not in ("all", ""):
        raise ReportInputError(
            f"--nucleus-filter {nucleus_filter!r} is not one of all, sampled")

    n_before = int(len(nuc))
    if exclude_fields:
        nuc = nuc[~nuc["excluded_field"]].copy()
        keep = labels.loc[~labels["excluded_field"], "image"]
        per_image = per_image[per_image["image"].isin(keep)].copy()
    return dict(per_image=per_image, nuclei=nuc, thresholds=thresholds, cfg=cfg,
                labels=labels, run_dir=run_dir, well_to_group=dict(well_to_group),
                n_nuclei_before_field_exclusion=n_before,
                n_nuclei_after_field_exclusion=int(len(nuc)),
                nucleus_filter=nucleus_filter,
                n_nuclei_all=n_nuclei_all,
                n_nuclei_after_nucleus_filter=n_after_nucleus_filter)


def resolve_group_order(labels: pd.DataFrame, declared: Sequence[str]) -> List[str]:
    """Biological groups in plotting order; the secondary-only group is never one."""
    present = [g for g in labels.loc[~labels["secondary_only"], "group"].dropna().unique()]
    order = [g for g in declared if g in present]
    order += sorted(g for g in present if g not in order)
    return order


# ------------------------------------------------------------------ hierarchy


def per_field_long(nuc: pd.DataFrame, per_image: pd.DataFrame,
                   endpoints: Sequence[_ep.Endpoint], labels: pd.DataFrame,
                   qc_min_nuclei: int) -> pd.DataFrame:
    """One row per (endpoint, field). Nucleus-level endpoints are averaged over
    the nuclei of that field; image-level endpoints are read straight off the
    engine's own per-image row."""
    bio = nuc[~nuc["secondary_only"]].copy()
    lab = labels[["image", "group", "well_id", "secondary_only"]]
    img = (per_image.drop(columns=["condition", "secondary_only", "group"], errors="ignore")
           .merge(lab, on="image", how="left", validate="one_to_one"))
    img = img[~img["secondary_only"].astype(bool)]
    rows = []
    for ep in endpoints:
        if ep.level == "nucleus":
            src = bio
            if ep.usability_flag:
                if ep.usability_flag not in bio.columns:
                    src = bio.iloc[0:0]
                else:
                    src = bio[bio[ep.usability_flag].astype(bool)]
            for (group, well, image), grp in src.groupby(["group", "well_id", "image"],
                                                         dropna=False, sort=True):
                v = pd.to_numeric(grp[ep.column], errors="coerce").dropna()
                rows.append(dict(
                    group=group, well_id=well, image=image, endpoint=ep.name,
                    family=ep.family, unit=ep.unit, level=ep.level, source=ep.source,
                    usability_filter=ep.usability_flag or "none",
                    n_nuclei_total=int(len(grp)), n_nuclei_nonmissing=int(len(v)),
                    n_nuclei_in_field_before_filter=int((bio["image"] == image).sum()),
                    field_value=float(v.mean()) if len(v) else np.nan,
                    field_sd=float(v.std(ddof=1)) if len(v) > 1 else np.nan,
                    below_qc_min_nuclei=bool(len(grp) < qc_min_nuclei),
                    qc_min_nuclei=qc_min_nuclei))
        else:
            n_by_image = bio.groupby("image").size()
            for _, r in img.iterrows():
                val = pd.to_numeric(pd.Series([r.get(ep.column, np.nan)]),
                                    errors="coerce").iloc[0]
                n_tot = int(n_by_image.get(r["image"], 0))
                rows.append(dict(
                    group=r["group"], well_id=r["well_id"], image=r["image"],
                    endpoint=ep.name, family=ep.family, unit=ep.unit, level=ep.level,
                    source=ep.source, usability_filter=ep.usability_flag or "none",
                    n_nuclei_total=n_tot, n_nuclei_nonmissing=int(n_tot if np.isfinite(val) else 0),
                    n_nuclei_in_field_before_filter=n_tot,
                    field_value=float(val) if np.isfinite(val) else np.nan,
                    field_sd=np.nan,
                    below_qc_min_nuclei=bool(n_tot < qc_min_nuclei),
                    qc_min_nuclei=qc_min_nuclei))
    if not rows:
        return pd.DataFrame(columns=["group", "well_id", "image", "endpoint"])
    return pd.DataFrame(rows).sort_values(["family", "endpoint", "group", "well_id", "image"])


def per_well_long(field: pd.DataFrame) -> pd.DataFrame:
    """One row per (endpoint, well). ``well_mean_of_field_values`` is the tested point."""
    keys = ["group", "well_id", "endpoint", "family", "unit", "level", "source",
            "usability_filter"]
    if field.empty:
        return pd.DataFrame(columns=keys)
    rows = []
    for key, grp in field.groupby(keys, dropna=False, sort=True):
        base = dict(zip(keys, key))
        v = pd.to_numeric(grp["field_value"], errors="coerce").dropna()
        # Two well statistics, both reported, because they are different numbers
        # and different analyses have used each. The MEAN OF FIELD MEANS weights
        # every field equally and is what the Welch gate runs on. The POOLED mean
        # weights every nucleus equally, so a field with more nuclei counts for
        # more; it is what a fixed-N sampled design reports.
        fv = pd.to_numeric(grp["field_value"], errors="coerce")
        nn = pd.to_numeric(grp["n_nuclei_nonmissing"], errors="coerce")
        ok = fv.notna() & nn.notna() & (nn > 0)
        pooled = (float((fv[ok] * nn[ok]).sum() / nn[ok].sum())
                  if ok.any() and nn[ok].sum() > 0 else float("nan"))
        rows.append(dict(base,
                         well_pooled_mean_of_nuclei=pooled,
                         n_fields=int(len(grp)),
                         n_fields_nonmissing=int(len(v)),
                         n_nuclei_in_well=int(grp["n_nuclei_total"].sum()),
                         n_nuclei_in_well_before_filter=int(
                             grp["n_nuclei_in_field_before_filter"].sum()),
                         well_mean_of_field_values=float(v.mean()) if len(v) else np.nan,
                         well_sd_of_field_values=float(v.std(ddof=1)) if len(v) > 1 else np.nan,
                         any_field_below_qc_min_nuclei=bool(grp["below_qc_min_nuclei"].any())))
    return pd.DataFrame(rows).sort_values(["family", "endpoint", "group", "well_id"])


# ------------------------------------------------------------------ contrasts


def build_contrasts(well: pd.DataFrame, field: pd.DataFrame,
                    endpoints: Sequence[_ep.Endpoint], absent: Sequence[str],
                    group_order: Sequence[str], reference: str,
                    labels: Dict[str, str], alpha: float = ALPHA,
                    all_pairs: bool = False) -> pd.DataFrame:
    """Every non-reference group against the reference, on well means.

    The star on a figure comes from ``p_welch``. ``p_welch_holm_within_family`` is
    the multiplicity-adjusted value and is printed in the footnote next to the
    minimum detectable effect, so a null result can be read for what it is worth.
    """
    absent = set(absent)
    # Default: every group against the reference. ``all_pairs`` reports every
    # unordered pair instead, which is what a multi-line design needs; the Tukey
    # adjustment is over the whole design either way.
    if all_pairs:
        pairs = [(b, a) for i, a in enumerate(group_order) for b in group_order[i + 1:]]
    else:
        pairs = [(g, reference) for g in group_order if g != reference]
    rows = []
    for ep in endpoints:
        sub = well[well["endpoint"] == ep.name] if len(well) else well
        for test, ref_group in pairs:
            ref_v = (sub.loc[sub["group"] == ref_group, "well_mean_of_field_values"]
                     .astype(float).dropna().to_numpy() if len(sub) else np.array([]))
            test_v = (sub.loc[sub["group"] == test, "well_mean_of_field_values"]
                      .astype(float).dropna().to_numpy() if len(sub) else np.array([]))
            r = dict(endpoint=ep.name, endpoint_plain=ep.pretty(labels),
                     family=ep.family, sheet=_ep.FAMILY_SHEET[ep.family],
                     unit=ep.unit, level=ep.level, source=ep.source,
                     test_group=test, reference_group=ref_group,
                     comparison=f"{test} minus {ref_group}, on well means",
                     primary=ep.primary, descriptive_only=ep.descriptive_only,
                     exploratory=ep.exploratory,
                     absolute_intensity=ep.absolute_intensity,
                     excluded_from_holm=ep.excluded_from_holm,
                     usability_filter=ep.usability_flag or "none",
                     n_nuclei_test=int(sub.loc[sub["group"] == test,
                                               "n_nuclei_in_well"].sum()) if len(sub) else 0,
                     n_nuclei_reference=int(sub.loc[sub["group"] == ref_group,
                                                    "n_nuclei_in_well"].sum()) if len(sub) else 0,
                     n_wells_test=int(len(test_v)), n_wells_reference=int(len(ref_v)))
            r.update({k: v for k, v in welch(test_v, ref_v, alpha).items()
                      if k not in ("n_test", "n_ref")})
            r.update(exact_permutation(test_v, ref_v))
            fsub = field[field["endpoint"] == ep.name] if len(field) else field
            # Tukey is fitted over EVERY group in the design, not just this pair;
            # its adjustment is a studentized range over k groups, so a two-group
            # fit inside a five-group design would understate the p-value.
            groups = {}
            for name in group_order:
                g = (pd.to_numeric(fsub.loc[fsub["group"] == name, "field_value"],
                                   errors="coerce").dropna().to_numpy()
                     if len(fsub) else np.array([]))
                if len(g):
                    groups[name] = g
            r.update(tukey_two_group(groups, test=test, reference=ref_group, alpha=alpha))
            # Tukey on WELL means as well. The field-level fit treats a technical
            # replicate as independent; the well-level fit is on the same unit the
            # Welch gate uses, so the two are not interchangeable and both are shown.
            wgroups = {}
            for name in group_order:
                g = (sub.loc[sub["group"] == name, "well_mean_of_field_values"]
                     .astype(float).dropna().to_numpy() if len(sub) else np.array([]))
                if len(g):
                    wgroups[name] = g
            wt = tukey_two_group(wgroups, test=test, reference=ref_group, alpha=alpha)
            r.update({f"well_{k}": v for k, v in wt.items()})
            r["ratio_test_over_reference"] = (
                float(test_v.mean() / ref_v.mean())
                if len(test_v) and len(ref_v) and ref_v.mean() != 0 else np.nan)
            r["endpoint_absent_in_run"] = ep.name in absent
            r["endpoint_note"] = ep.note
            rows.append(r)
    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["in_holm_family"] = ~(out["descriptive_only"] | out["endpoint_absent_in_run"]
                              | out["absolute_intensity"]
                              | out["excluded_from_holm"].astype(bool))
    out["holm_exclusion_reason"] = np.where(
        out["endpoint_absent_in_run"], "column absent from this run",
        np.where(out["absolute_intensity"],
                 "absolute intensity, not comparable as a level claim across sections",
                 np.where(out["excluded_from_holm"].astype(bool), out["excluded_from_holm"],
                          np.where(out["descriptive_only"], "descriptive only by design", ""))))
    out["p_welch_holm_within_family"] = np.nan
    out["holm_family_size"] = np.nan
    for _, idx in out.groupby(["family", "test_group"]).groups.items():
        idx = list(idx)
        member = [i for i in idx if out.at[i, "in_holm_family"]]
        adj = holm([out.at[i, "p_welch"] for i in member])
        for i, a in zip(member, adj):
            out.at[i, "p_welch_holm_within_family"] = a
        size = int(np.isfinite([out.at[i, "p_welch"] for i in member]).sum()) if member else 0
        for i in member:
            out.at[i, "holm_family_size"] = size
    out["significant_raw_0p05"] = (out["p_welch"] < alpha).astype(object)
    out.loc[~np.isfinite(out["p_welch"]), "significant_raw_0p05"] = pd.NA
    out["significant_holm_0p05"] = (out["p_welch_holm_within_family"] < alpha).astype(object)
    out.loc[~out["in_holm_family"], "significant_holm_0p05"] = pd.NA

    out["mde_hedges_g_alpha_0p05"] = np.nan
    out["mde_hedges_g_at_family_alpha"] = np.nan
    out["observed_g_reaches_mde"] = pd.NA
    cache: Dict[Tuple[float, int, int], float] = {}

    def _mde(a: float, n1: int, n2: int) -> float:
        key = (round(a, 12), n1, n2)
        if key not in cache:
            cache[key] = mde_hedges_g(a, n1=n1, n2=n2)
        return cache[key]

    for i in out.index:
        if not out.at[i, "in_holm_family"]:
            continue
        n1 = int(out.at[i, "n_wells_test"])
        n2 = int(out.at[i, "n_wells_reference"])
        if n1 < 2 or n2 < 2:
            continue
        out.at[i, "mde_hedges_g_alpha_0p05"] = _mde(alpha, n1, n2)
        k = out.at[i, "holm_family_size"]
        if not np.isfinite(k) or k < 1:
            continue
        out.at[i, "mde_hedges_g_at_family_alpha"] = _mde(alpha / float(k), n1, n2)
        g = out.at[i, "hedges_g"]
        if np.isfinite(g):
            out.at[i, "observed_g_reaches_mde"] = bool(
                abs(g) >= out.at[i, "mde_hedges_g_at_family_alpha"])
    fam_rank = {f: i for i, f in enumerate(_ep.FAMILY_ORDER)}
    out["_f"] = out["family"].map(fam_rank).fillna(99)
    out = out.sort_values(["_f", "endpoint", "reference_group", "test_group"]).drop(columns="_f")
    return out.reset_index(drop=True)


def family_sheet(contrasts: pd.DataFrame, well: pd.DataFrame, family: str,
                 group_order: Sequence[str]) -> pd.DataFrame:
    """One by-group sheet: the contrast rows of a family, with each well's own mean
    spread across columns so the replicates behind every test are visible."""
    c = contrasts[contrasts["family"] == family].copy() if len(contrasts) else contrasts
    if not len(c):
        return pd.DataFrame({"note": [f"no endpoints in the {family} family"]})
    w = well[well["family"] == family]
    if len(w):
        wide = w.pivot_table(index="endpoint", columns=["group", "well_id"],
                             values="well_mean_of_field_values", aggfunc="first")
        rank = {g: i for i, g in enumerate(group_order)}
        cols = sorted(wide.columns, key=lambda gw: (rank.get(gw[0], 99), str(gw[1])))
        wide = wide[cols]
        wide.columns = [f"{g} well {wl}" for g, wl in cols]
        c = c.merge(wide.reset_index(), on="endpoint", how="left")
    front = ["endpoint_plain", "endpoint", "unit", "test_group", "reference_group",
             "primary", "descriptive_only", "exploratory", "endpoint_absent_in_run",
             "n_wells_test", "n_wells_reference", "mean_reference", "mean_test",
             "diff", "ratio_test_over_reference", "ci_low", "ci_high", "hedges_g",
             "p_welch", "significant_raw_0p05", "p_welch_holm_within_family",
             "significant_holm_0p05", "mde_hedges_g_at_family_alpha",
             "observed_g_reaches_mde", "p_permutation", "perm_arithmetic_floor",
             "p_tukey_fov"]
    c = c.rename(columns={"mean_ref": "mean_reference"})
    front = [x for x in front if x in c.columns]
    return c[front + [x for x in c.columns if x not in front]]


# ---------------------------------------------------------------- sec-only


def secondary_only_table(nuc: pd.DataFrame, per_image: pd.DataFrame,
                         labels: pd.DataFrame, exclude_fields: Dict[str, str],
                         min_nuclei: int, outlier_k: float = 0.0) -> pd.DataFrame:
    """One row per secondary-only field, with the rule that excluded it, if any.

    Exclusions are RULES plus operator-supplied reasons, never a remembered list:
    rule 1 is the nucleus-count floor; rule 2 is the per-channel detection-rate
    outlier cut at ``outlier_k`` times the control median; anything further arrives
    as an explicit ``--exclude-field`` with its own recorded reason.
    """
    sec = nuc[nuc["secondary_only"]].copy() if "secondary_only" in nuc.columns else nuc.iloc[0:0]
    sec_labels = labels[labels["secondary_only"]]
    if not len(sec_labels):
        return pd.DataFrame({"note": ["no secondary-only images in this run"]})
    rows = []
    for image in sorted(sec_labels["image"].astype(str).unique()):
        grp = sec[sec["image"].astype(str) == image]
        pim = per_image.loc[per_image["image"].astype(str) == image]
        row = dict(field=image, n_nuclei=int(len(grp)))
        for out_name, col in (("rna1_spots_per_nucleus", "n_spots_rna1"),
                              ("rna1_nuclear_spots_per_nucleus", "nuclear_spot_count"),
                              ("rna2_spots_per_nucleus", "n_spots_rna2"),
                              ("partner_spots_per_nucleus", "n_spots_protein")):
            row[out_name] = (float(pd.to_numeric(grp[col], errors="coerce").mean())
                             if col in grp.columns and len(grp) else np.nan)
        for out_name, col in (("total_spots_rna1", "total_spots_rna1"),
                              ("total_spots_partner", "total_spots_protein")):
            row[out_name] = (float(pd.to_numeric(pim[col], errors="coerce").sum())
                             if col in pim.columns and len(pim) else np.nan)
        for col in ("rna_bigfish_log_threshold", "protein_bigfish_log_threshold"):
            row[col] = (float(pd.to_numeric(pim[col], errors="coerce").iloc[0])
                        if col in pim.columns and len(pim) else np.nan)
        rows.append(row)
    df = pd.DataFrame(rows)
    df["rule_min_nuclei"] = min_nuclei
    df["excluded_by_nucleus_floor"] = df["n_nuclei"] < min_nuclei

    # Detection-rate outlier rule. A control field whose puncta per nucleus on ANY
    # channel exceeds k times the median across the control fields on that same
    # channel is dropped from the background estimate. Per channel, because a field
    # can be clean on one channel and hot on the other, and a background estimate
    # that keeps a hot field understates how much of the biological signal is floor.
    df["rule_outlier_k"] = float(outlier_k) if outlier_k else float("nan")
    df["excluded_by_detection_outlier"] = False
    df["outlier_channel"] = ""
    for out_col, chan in (("rna1_spots_per_nucleus", "rna1"),
                          ("rna2_spots_per_nucleus", "rna2"),
                          ("partner_spots_per_nucleus", "partner")):
        if out_col not in df.columns:
            continue
        v = pd.to_numeric(df[out_col], errors="coerce")
        base = v[~df["excluded_by_nucleus_floor"]].dropna()
        med = float(base.median()) if len(base) else float("nan")
        cut = med * float(outlier_k) if outlier_k and np.isfinite(med) else float("nan")
        df[f"outlier_median_{chan}"] = med
        df[f"outlier_cutoff_{chan}"] = cut
        if np.isfinite(cut):
            hit = v > cut
            df["outlier_channel"] = np.where(
                hit & ~df["excluded_by_detection_outlier"], chan, df["outlier_channel"])
            df["excluded_by_detection_outlier"] = df["excluded_by_detection_outlier"] | hit

    df["excluded_by_operator"] = df["field"].isin(exclude_fields)
    df["operator_reason"] = df["field"].map(lambda f: exclude_fields.get(f, ""))
    df["excluded"] = (df["excluded_by_nucleus_floor"] | df["excluded_by_detection_outlier"]
                      | df["excluded_by_operator"])
    df["exclusion_reason"] = np.where(
        df["excluded_by_nucleus_floor"],
        f"rule: fewer than {min_nuclei} nuclei in the field",
        np.where(df["excluded_by_detection_outlier"],
                 ("rule: puncta per nucleus on the " + df["outlier_channel"]
                  + f" channel exceed {outlier_k:g} times the median across the "
                    "control fields on that channel"),
                 np.where(df["excluded_by_operator"], df["operator_reason"], "")))
    kept = df.loc[~df["excluded"], "rna1_spots_per_nucleus"]
    allf = df["rna1_spots_per_nucleus"]
    df["sensitivity_mean_kept_fields"] = float(kept.mean()) if len(kept.dropna()) else np.nan
    df["sensitivity_mean_all_fields"] = float(allf.mean()) if len(allf.dropna()) else np.nan
    df["sensitivity_n_kept_fields"] = int(kept.notna().sum())
    df["sensitivity_n_all_fields"] = int(allf.notna().sum())
    df["role"] = ("SECONDARY check. Inference is across the biological groups; these "
                  "wells received no probe and no primary antibody, so they bound the "
                  "detection floor and are never a comparison arm.")
    return df.sort_values("field").reset_index(drop=True)
