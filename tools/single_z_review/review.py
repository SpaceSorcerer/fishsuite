"""Build manual-review packets for choosing one optical plane per image.

This module deliberately stops before analysis.  Its numerical output nominates
candidate planes, while ``selected_z`` remains blank until a human reviews the
single-plane contact sheet.  All configured channels are displayed at the same
z; no projection is rendered or used as a selected analysis plane.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class ChannelSpec:
    """One channel to load, diagnose, and show in the contact sheet."""

    role: str
    index: int
    label: str
    display_low: float
    display_high: float

    def __post_init__(self) -> None:
        role = self.role.strip()
        if not role:
            raise ValueError("channel role cannot be empty")
        if int(self.index) < 0:
            raise ValueError(f"channel index must be non-negative: {self.index}")
        if not np.isfinite(self.display_low) or not np.isfinite(self.display_high):
            raise ValueError(f"display window for {role!r} must be finite")
        if float(self.display_high) <= float(self.display_low):
            raise ValueError(
                f"display window for {role!r} must satisfy low < high"
            )


@dataclass(frozen=True)
class ReviewConfig:
    """Channel map and candidate-nomination settings for a review run."""

    channels: tuple[ChannelSpec, ...]
    focus_roles: tuple[str, ...] = ()
    candidate_count: int = 3
    central_fraction: float = 0.8
    intensity_weighted: bool = True
    min_intensity_fraction: float = 0.3
    nuclear_mask: bool = True
    tile_size: int = 128
    contact_sheet_columns: int = 5

    def __post_init__(self) -> None:
        if not self.channels:
            raise ValueError("at least one channel must be configured")
        roles = [channel.role for channel in self.channels]
        indices = [int(channel.index) for channel in self.channels]
        if len(set(roles)) != len(roles):
            raise ValueError(f"channel roles must be unique: {roles}")
        if len(set(indices)) != len(indices):
            raise ValueError(f"channel indices must be unique: {indices}")
        focus_roles = self.focus_roles or tuple(roles)
        missing = sorted(set(focus_roles).difference(roles))
        if missing:
            raise ValueError(f"focus roles are not configured channels: {missing}")
        object.__setattr__(self, "focus_roles", tuple(focus_roles))
        if int(self.candidate_count) < 1:
            raise ValueError("candidate_count must be at least 1")
        if not 0.0 < float(self.central_fraction) <= 1.0:
            raise ValueError("central_fraction must be in (0, 1]")
        if not 0.0 <= float(self.min_intensity_fraction) <= 1.0:
            raise ValueError("min_intensity_fraction must be in [0, 1]")
        if int(self.tile_size) < 24:
            raise ValueError("tile_size must be at least 24 pixels")
        if int(self.contact_sheet_columns) < 1:
            raise ValueError("contact_sheet_columns must be at least 1")
        if self.nuclear_mask and "dapi" not in roles:
            raise ValueError("nuclear-masked focus diagnostics require role 'dapi'")


@dataclass(frozen=True)
class ReviewAnalysis:
    """Per-stack diagnostics and explicitly provisional candidate planes."""

    candidate_z: tuple[int, ...]
    diagnostics: tuple[dict[str, object], ...]
    combined_focus_scores: np.ndarray
    nuclear_mask_used: bool
    nuclear_mask_coverage: float
    nuclear_mask_reference_z: int | None


@dataclass(frozen=True)
class RunSummary:
    """Paths and completion counts from one inventory review build."""

    images_total: int
    images_succeeded: int
    images_failed: int
    manifest_csv: Path
    selection_template_csv: Path
    z_diagnostics_csv: Path
    config_json: Path


def _validate_stacks(
    stacks: Mapping[str, np.ndarray], config: ReviewConfig
) -> dict[str, np.ndarray]:
    configured_roles = [spec.role for spec in config.channels]
    if set(stacks) != set(configured_roles):
        raise ValueError(
            "stack roles must exactly match configured channels: "
            f"got={sorted(stacks)}, expected={sorted(configured_roles)}"
        )
    arrays: dict[str, np.ndarray] = {}
    for role in configured_roles:
        array = np.asarray(stacks[role])
        if array.ndim == 2:
            array = array[None, :, :]
        if array.ndim != 3 or array.shape[0] < 1:
            raise ValueError(f"channel {role!r} must be a non-empty ZYX stack")
        arrays[role] = array
    shapes = {array.shape for array in arrays.values()}
    if len(shapes) != 1:
        raise ValueError(f"all configured channels must share one ZYX shape: {shapes}")
    return arrays


def _candidate_indices(
    scores: np.ndarray, *, count: int, central_fraction: float
) -> tuple[int, ...]:
    n_z = int(scores.size)
    keep = max(1, int(round(n_z * float(central_fraction))))
    margin_before = (n_z - keep) // 2
    eligible = list(range(margin_before, margin_before + keep))
    center = (n_z - 1) / 2.0
    ordered = sorted(
        eligible,
        key=lambda z0: (-float(scores[z0]), abs(z0 - center), z0),
    )
    return tuple(z0 + 1 for z0 in ordered[: min(int(count), len(ordered))])


def analyze_stacks(
    stacks: Mapping[str, np.ndarray], config: ReviewConfig
) -> ReviewAnalysis:
    """Compute per-z diagnostics and nominate planes without selecting one."""
    arrays = _validate_stacks(stacks, config)
    from fishsuite.core.io import _derive_nuclear_mask, _per_slice_focus_scores

    mask: np.ndarray | None = None
    mask_coverage = 0.0
    mask_reference_z: int | None = None
    if config.nuclear_mask:
        # First nominate one DAPI plane without a mask, then derive the fixed XY
        # scoring mask from THAT exact optical section.  This keeps even the
        # focus-mask reference projection-free.
        preliminary_scores, _ = _per_slice_focus_scores(
            arrays["dapi"],
            intensity_weighted=config.intensity_weighted,
            min_intensity_frac_of_peak=config.min_intensity_fraction,
            mask=None,
        )
        mask_reference_z = _candidate_indices(
            np.asarray(preliminary_scores, dtype=float),
            count=1,
            central_fraction=config.central_fraction,
        )[0]
        mask, mask_coverage = _derive_nuclear_mask(
            arrays["dapi"][mask_reference_z - 1]
        )

    raw_scores: dict[str, np.ndarray] = {}
    normalized_scores: dict[str, np.ndarray] = {}
    focus_means: dict[str, np.ndarray] = {}
    for role, stack in arrays.items():
        raw, means = _per_slice_focus_scores(
            stack,
            intensity_weighted=config.intensity_weighted,
            min_intensity_frac_of_peak=config.min_intensity_fraction,
            mask=mask,
        )
        raw_scores[role] = np.asarray(raw, dtype=float)
        focus_means[role] = np.asarray(means, dtype=float)
        peak = float(np.max(raw)) if raw.size else 0.0
        normalized_scores[role] = (
            np.asarray(raw, dtype=float) / peak
            if np.isfinite(peak) and peak > 0.0
            else np.ones_like(raw, dtype=float)
        )

    focus_matrix = np.vstack(
        [normalized_scores[role] for role in config.focus_roles]
    )
    combined = np.prod(focus_matrix, axis=0)
    candidate_z = _candidate_indices(
        combined,
        count=config.candidate_count,
        central_fraction=config.central_fraction,
    )
    candidate_ranks = {z: rank for rank, z in enumerate(candidate_z, start=1)}

    specs = {spec.role: spec for spec in config.channels}
    diagnostics: list[dict[str, object]] = []
    n_z = int(next(iter(arrays.values())).shape[0])
    for z0 in range(n_z):
        z = z0 + 1
        for role in [spec.role for spec in config.channels]:
            plane = np.asarray(arrays[role][z0], dtype=float)
            spec = specs[role]
            diagnostics.append(
                {
                    "z": z,
                    "role": role,
                    "channel_index": int(spec.index),
                    "channel_label": spec.label,
                    "candidate_rank": candidate_ranks.get(z, ""),
                    "combined_focus_score": float(combined[z0]),
                    "focus_raw": float(raw_scores[role][z0]),
                    "focus_normalized": float(normalized_scores[role][z0]),
                    "focus_region_mean": float(focus_means[role][z0]),
                    "slice_mean": float(np.mean(plane)),
                    "slice_median": float(np.median(plane)),
                    "slice_p99": float(np.percentile(plane, 99.0)),
                    "display_clip_low_fraction": float(
                        np.mean(plane <= float(spec.display_low))
                    ),
                    "display_clip_high_fraction": float(
                        np.mean(plane >= float(spec.display_high))
                    ),
                }
            )
    return ReviewAnalysis(
        candidate_z=candidate_z,
        diagnostics=tuple(diagnostics),
        combined_focus_scores=combined,
        nuclear_mask_used=mask is not None,
        nuclear_mask_coverage=float(mask_coverage),
        nuclear_mask_reference_z=mask_reference_z,
    )


def _font(size: int) -> ImageFont.ImageFont:
    font_path = Path(r"C:\Windows\Fonts\arial.ttf")
    if font_path.is_file():
        return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def _thumbnail(plane: np.ndarray, spec: ChannelSpec, size: int) -> Image.Image:
    values = np.asarray(plane, dtype=np.float32)
    scaled = np.clip(
        (values - float(spec.display_low))
        / (float(spec.display_high) - float(spec.display_low)),
        0.0,
        1.0,
    )
    gray = Image.fromarray(np.round(scaled * 255.0).astype(np.uint8), mode="L")
    return gray.resize((size, size), resample=Image.Resampling.BILINEAR).convert("RGB")


def _render_contact_sheet(
    stacks: Mapping[str, np.ndarray],
    config: ReviewConfig,
    analysis: ReviewAnalysis,
    destination: Path,
    *,
    title: str,
) -> None:
    arrays = _validate_stacks(stacks, config)
    n_z = int(next(iter(arrays.values())).shape[0])
    columns = min(config.contact_sheet_columns, n_z)
    rows = int(math.ceil(n_z / columns))
    gap = 8
    header = 82
    plane_label_height = 24
    role_label_height = 18
    tile = int(config.tile_size)
    card_width = tile + 2 * gap
    card_height = plane_label_height + len(config.channels) * (
        tile + role_label_height + gap
    )
    width = gap + columns * (card_width + gap)
    height = header + rows * (card_height + gap) + gap
    canvas = Image.new("RGB", (width, height), color=(13, 16, 20))
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 7), title, fill=(255, 255, 255), font=_font(18))
    draw.text(
        (gap, 35),
        "Single optical planes only; nominees are provisional; fill selected_z after manual review.",
        fill=(220, 224, 228),
        font=_font(13),
    )
    draw.text(
        (gap, 56),
        "Every channel inside a z card is the exact same plane (no MIP/projection).",
        fill=(220, 224, 228),
        font=_font(13),
    )
    ranks = {z: rank for rank, z in enumerate(analysis.candidate_z, start=1)}
    border_palette = {1: (255, 215, 0), 2: (230, 159, 0), 3: (190, 190, 190)}
    for z0 in range(n_z):
        z = z0 + 1
        grid_row, grid_col = divmod(z0, columns)
        x0 = gap + grid_col * (card_width + gap)
        y0 = header + grid_row * (card_height + gap)
        rank = ranks.get(z)
        z_label = f"z{z:02d}" + (f"  NOMINEE {rank}" if rank else "")
        draw.text(
            (x0 + 4, y0 + 2),
            z_label,
            fill=border_palette.get(rank, (235, 235, 235)),
            font=_font(13),
        )
        for role_row, spec in enumerate(config.channels):
            y = y0 + plane_label_height + role_row * (
                tile + role_label_height + gap
            )
            draw.text(
                (x0 + 3, y),
                spec.label,
                fill=(220, 224, 228),
                font=_font(11),
            )
            image_y = y + role_label_height
            canvas.paste(_thumbnail(arrays[spec.role][z0], spec, tile), (x0 + gap, image_y))
        border = border_palette.get(rank, (72, 78, 86))
        border_width = 4 if rank else 1
        for inset in range(border_width):
            draw.rectangle(
                (
                    x0 - inset,
                    y0 - inset,
                    x0 + card_width - 1 + inset,
                    y0 + card_height - 1 + inset,
                ),
                outline=border,
                width=1,
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG", optimize=True)


def _read_inventory(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    if "source_vsi" not in fields:
        raise ValueError("inventory CSV must contain a source_vsi column")
    if not rows:
        raise ValueError("inventory CSV contains no images")
    for number, row in enumerate(rows, start=2):
        source_text = str(row.get("source_vsi", "")).strip()
        if not source_text:
            raise ValueError(f"inventory row {number} has an empty source_vsi")
        source = Path(source_text)
        if not source.is_absolute():
            source = Path(path).parent / source
        row["source_vsi"] = str(source.resolve(strict=False))
    return fields, rows


def _raw_root_for_source(source: Path) -> Path:
    candidates = [
        parent
        for parent in source.parents
        if parent.name.casefold() in {"raw", "raw images", "raw_images"}
    ]
    return candidates[-1] if candidates else source.parent


def _is_nested(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _guard_output_outside_sources(output_dir: Path, sources: Sequence[Path]) -> None:
    output = output_dir.resolve(strict=False)
    roots = {_raw_root_for_source(source.resolve(strict=False)) for source in sources}
    for root in roots:
        if _is_nested(output, root.resolve(strict=False)):
            raise ValueError(
                f"output directory must be outside the raw source tree {root}: {output}"
            )


def _review_id(source: Path, row: Mapping[str, str]) -> str:
    preferred = str(row.get("image_key", "")).strip() or source.stem
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", preferred).strip("._-") or "image"
    digest = hashlib.sha256(str(source).casefold().encode("utf-8")).hexdigest()[:10]
    return f"{slug}__{digest}"


def _write_csv(
    destination: Path, rows: Iterable[Mapping[str, object]], fieldnames: Sequence[str]
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, destination)


def _close_image(image: object) -> None:
    try:
        reader = getattr(getattr(image, "bio", None), "reader", None)
        close = getattr(reader, "close", None)
        if callable(close):
            close()
    except Exception:
        pass


def run_review(
    inventory_csv: Path,
    output_dir: Path,
    config: ReviewConfig,
    *,
    reader: Callable[[Path], object] | None = None,
) -> RunSummary:
    """Build review artifacts from an inventory without writing to raw data."""
    inventory_csv = Path(inventory_csv)
    output_dir = Path(output_dir)
    inventory_fields, inventory_rows = _read_inventory(inventory_csv)
    sources = [Path(row["source_vsi"]) for row in inventory_rows]
    _guard_output_outside_sources(output_dir, sources)
    output_dir.mkdir(parents=True, exist_ok=True)
    if reader is None:
        from fishsuite.core.io import read_image

        reader = read_image

    manifest_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    successes = 0
    failures = 0
    max_candidates = int(config.candidate_count)
    for row in inventory_rows:
        source = Path(row["source_vsi"])
        review_id = _review_id(source, row)
        image_dir = output_dir / "images" / review_id
        contact_sheet = image_dir / "all_z_contact_sheet.png"
        per_image_diagnostics = image_dir / "z_diagnostics.csv"
        base: dict[str, object] = dict(row)
        base.update(
            {
                "review_id": review_id,
                "source_vsi": str(source),
                "contact_sheet": "",
                "per_image_z_diagnostics": "",
                "processing_status": "ERROR",
                "processing_error": "",
                "candidate_method": "provisional_intensity_weighted_focus",
                "candidate_z_count": 0,
                "nuclear_mask_used": "",
                "nuclear_mask_coverage": "",
                "nuclear_mask_reference_z": "",
            }
        )
        for rank in range(1, max_candidates + 1):
            base[f"candidate_z_{rank}"] = ""
        image = None
        try:
            image = reader(source)
            n_channels = int(getattr(image, "n_channels"))
            stacks: dict[str, np.ndarray] = {}
            for spec in config.channels:
                if spec.index >= n_channels:
                    raise IndexError(
                        f"configured channel {spec.index} ({spec.role}) is out of "
                        f"range for {source.name} with {n_channels} channels"
                    )
                stack = np.asarray(
                    image.bio.get_image_data("ZYX", T=0, C=spec.index)
                )
                stacks[spec.role] = stack[None, :, :] if stack.ndim == 2 else stack
            analysis = analyze_stacks(stacks, config)
            _render_contact_sheet(
                stacks,
                config,
                analysis,
                contact_sheet,
                title=f"{source.parent.name} | {source.name}",
            )
            image_diag_rows: list[dict[str, object]] = []
            for diag in analysis.diagnostics:
                enriched = {
                    "review_id": review_id,
                    "source_vsi": str(source),
                    **diag,
                }
                image_diag_rows.append(enriched)
                diagnostic_rows.append(enriched)
            diagnostic_fields = [
                "review_id",
                "source_vsi",
                "z",
                "role",
                "channel_index",
                "channel_label",
                "candidate_rank",
                "combined_focus_score",
                "focus_raw",
                "focus_normalized",
                "focus_region_mean",
                "slice_mean",
                "slice_median",
                "slice_p99",
                "display_clip_low_fraction",
                "display_clip_high_fraction",
            ]
            _write_csv(per_image_diagnostics, image_diag_rows, diagnostic_fields)
            base.update(
                {
                    "processing_status": "READY_FOR_MANUAL_REVIEW",
                    "contact_sheet": str(contact_sheet.resolve()),
                    "per_image_z_diagnostics": str(per_image_diagnostics.resolve()),
                    "candidate_z_count": len(analysis.candidate_z),
                    "nuclear_mask_used": analysis.nuclear_mask_used,
                    "nuclear_mask_coverage": analysis.nuclear_mask_coverage,
                    "nuclear_mask_reference_z": (
                        analysis.nuclear_mask_reference_z or ""
                    ),
                    "n_z_read": int(next(iter(stacks.values())).shape[0]),
                    "voxel_xy_nm_read": getattr(image, "voxel_xy_nm", ""),
                    "voxel_z_nm_read": getattr(image, "voxel_z_nm", ""),
                    "channel_names_read": " | ".join(
                        str(name) for name in getattr(image, "channel_names", [])
                    ),
                }
            )
            for rank, z in enumerate(analysis.candidate_z, start=1):
                base[f"candidate_z_{rank}"] = z
            successes += 1
        except Exception as exc:
            base["processing_error"] = f"{type(exc).__name__}: {exc}"
            failures += 1
        finally:
            if image is not None:
                _close_image(image)
        manifest_rows.append(base)
        status = (
            "PENDING_MANUAL_REVIEW"
            if base["processing_status"] == "READY_FOR_MANUAL_REVIEW"
            else "BLOCKED_PROCESSING_ERROR"
        )
        selection = {
            "review_id": review_id,
            "source_vsi": str(source),
            "processing_status": base["processing_status"],
            "review_status": status,
            "selected_z": "",
            "reviewer": "",
            "reviewed_at": "",
            "review_notes": "",
            "contact_sheet": base["contact_sheet"],
        }
        for rank in range(1, max_candidates + 1):
            selection[f"candidate_z_{rank}"] = base[f"candidate_z_{rank}"]
        selection_rows.append(selection)

    manifest_csv = output_dir / "review_manifest.csv"
    selection_csv = output_dir / "selected_z_review.csv"
    diagnostics_csv = output_dir / "z_diagnostics.csv"
    config_json = output_dir / "review_config.json"
    candidate_fields = [
        f"candidate_z_{rank}" for rank in range(1, max_candidates + 1)
    ]
    manifest_extra = [
        "review_id",
        "processing_status",
        "processing_error",
        "candidate_method",
        "candidate_z_count",
        *candidate_fields,
        "nuclear_mask_used",
        "nuclear_mask_coverage",
        "nuclear_mask_reference_z",
        "n_z_read",
        "voxel_xy_nm_read",
        "voxel_z_nm_read",
        "channel_names_read",
        "contact_sheet",
        "per_image_z_diagnostics",
    ]
    manifest_fields = list(dict.fromkeys([*inventory_fields, *manifest_extra]))
    _write_csv(manifest_csv, manifest_rows, manifest_fields)
    diagnostic_fields = [
        "review_id",
        "source_vsi",
        "z",
        "role",
        "channel_index",
        "channel_label",
        "candidate_rank",
        "combined_focus_score",
        "focus_raw",
        "focus_normalized",
        "focus_region_mean",
        "slice_mean",
        "slice_median",
        "slice_p99",
        "display_clip_low_fraction",
        "display_clip_high_fraction",
    ]
    _write_csv(diagnostics_csv, diagnostic_rows, diagnostic_fields)

    selection_fields = [
        "review_id",
        "source_vsi",
        "processing_status",
        *candidate_fields,
        "review_status",
        "selected_z",
        "reviewer",
        "reviewed_at",
        "review_notes",
        "contact_sheet",
    ]
    existing: dict[str, dict[str, str]] = {}
    if selection_csv.is_file():
        with selection_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            existing = {
                str(item.get("review_id", "")): dict(item)
                for item in csv.DictReader(handle)
            }
    manual_fields = [
        "review_status",
        "selected_z",
        "reviewer",
        "reviewed_at",
        "review_notes",
    ]
    for selection in selection_rows:
        previous = existing.get(str(selection["review_id"]))
        if previous:
            for field in manual_fields:
                selection[field] = previous.get(field, selection[field])
    _write_csv(selection_csv, selection_rows, selection_fields)

    config_payload = {
        "schema_version": 1,
        "inventory_csv": str(inventory_csv.resolve()),
        "output_dir": str(output_dir.resolve()),
        "selected_z_policy": "manual_review_required",
        "projection_policy": "no_projection; every diagnostic/rendered tile is one z",
        "candidate_only": True,
        "channels": [asdict(channel) for channel in config.channels],
        "focus_roles": list(config.focus_roles),
        "candidate_count": config.candidate_count,
        "central_fraction": config.central_fraction,
        "intensity_weighted": config.intensity_weighted,
        "min_intensity_fraction": config.min_intensity_fraction,
        "nuclear_mask": config.nuclear_mask,
        "nuclear_mask_note": (
            "One provisional DAPI optical plane is used to derive a fixed XY "
            "focus-scoring mask; no projection is computed, analyzed, or rendered."
            if config.nuclear_mask
            else "disabled"
        ),
        "tile_size": config.tile_size,
        "contact_sheet_columns": config.contact_sheet_columns,
    }
    temporary_config = config_json.with_suffix(".json.partial")
    temporary_config.write_text(
        json.dumps(config_payload, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary_config, config_json)
    return RunSummary(
        images_total=len(inventory_rows),
        images_succeeded=successes,
        images_failed=failures,
        manifest_csv=manifest_csv,
        selection_template_csv=selection_csv,
        z_diagnostics_csv=diagnostics_csv,
        config_json=config_json,
    )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Nominate and render single-z candidates from a FishSuite image "
            "inventory; final selected_z always requires manual review."
        )
    )
    parser.add_argument("inventory_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--channel",
        action="append",
        required=True,
        metavar="ROLE=INDEX",
        help="Configured channel; repeat for DAPI-only or multi-channel review.",
    )
    parser.add_argument(
        "--window",
        action="append",
        required=True,
        metavar="ROLE=LOW,HIGH",
        help="Fixed display window for a configured role; repeat per channel.",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        metavar="ROLE=TEXT",
        help="Human-readable channel label; defaults to the role.",
    )
    parser.add_argument(
        "--focus-role",
        action="append",
        default=[],
        metavar="ROLE",
        help="Role contributing to candidate nomination; default is every channel.",
    )
    parser.add_argument("--candidate-count", type=int, default=3)
    parser.add_argument("--central-fraction", type=float, default=0.8)
    parser.add_argument("--min-intensity-fraction", type=float, default=0.3)
    parser.add_argument(
        "--unweighted-focus",
        action="store_true",
        help="Disable FishSuite's intensity weighting (normally keep it enabled).",
    )
    parser.add_argument(
        "--no-nuclear-mask",
        action="store_true",
        help="Score focus over the whole frame instead of a DAPI-derived XY mask.",
    )
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--columns", type=int, default=5)
    return parser


def _assignment(value: str, *, option: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"{option} expects ROLE=VALUE, got {value!r}")
    role, payload = value.split("=", 1)
    role = role.strip()
    payload = payload.strip()
    if not role or not payload:
        raise ValueError(f"{option} expects non-empty ROLE=VALUE, got {value!r}")
    return role, payload


def _config_from_args(args: argparse.Namespace) -> ReviewConfig:
    channel_pairs = [_assignment(value, option="--channel") for value in args.channel]
    window_pairs = [_assignment(value, option="--window") for value in args.window]
    label_pairs = [_assignment(value, option="--label") for value in args.label]
    channel_map: dict[str, int] = {}
    for role, value in channel_pairs:
        if role in channel_map:
            raise ValueError(f"duplicate --channel role: {role}")
        channel_map[role] = int(value)
    windows: dict[str, tuple[float, float]] = {}
    for role, value in window_pairs:
        pieces = [piece.strip() for piece in value.split(",")]
        if len(pieces) != 2:
            raise ValueError(f"--window expects ROLE=LOW,HIGH, got {role}={value}")
        if role in windows:
            raise ValueError(f"duplicate --window role: {role}")
        windows[role] = (float(pieces[0]), float(pieces[1]))
    labels: dict[str, str] = {}
    for role, label in label_pairs:
        if role in labels:
            raise ValueError(f"duplicate --label role: {role}")
        labels[role] = label
    missing_windows = sorted(set(channel_map).difference(windows))
    unknown_windows = sorted(set(windows).difference(channel_map))
    unknown_labels = sorted(set(labels).difference(channel_map))
    if missing_windows:
        raise ValueError(f"missing --window for roles: {missing_windows}")
    if unknown_windows or unknown_labels:
        raise ValueError(
            "window/label roles must also be configured by --channel: "
            f"windows={unknown_windows}, labels={unknown_labels}"
        )
    channels = tuple(
        ChannelSpec(
            role=role,
            index=index,
            label=labels.get(role, role),
            display_low=windows[role][0],
            display_high=windows[role][1],
        )
        for role, index in channel_map.items()
    )
    return ReviewConfig(
        channels=channels,
        focus_roles=tuple(args.focus_role),
        candidate_count=args.candidate_count,
        central_fraction=args.central_fraction,
        intensity_weighted=not args.unweighted_focus,
        min_intensity_fraction=args.min_intensity_fraction,
        nuclear_mask=not args.no_nuclear_mask,
        tile_size=args.tile_size,
        contact_sheet_columns=args.columns,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    reader: Callable[[Path], object] | None = None,
) -> int:
    """Command-line entry point; reader injection keeps synthetic tests local."""
    parser = _argument_parser()
    args = parser.parse_args(argv)
    try:
        config = _config_from_args(args)
        summary = run_review(
            args.inventory_csv,
            args.output_dir,
            config,
            reader=reader,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "images_total": summary.images_total,
                "images_succeeded": summary.images_succeeded,
                "images_failed": summary.images_failed,
                "manifest_csv": str(summary.manifest_csv.resolve()),
                "selection_template_csv": str(
                    summary.selection_template_csv.resolve()
                ),
                "z_diagnostics_csv": str(summary.z_diagnostics_csv.resolve()),
            }
        )
    )
    return 0 if summary.images_failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
