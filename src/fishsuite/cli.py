"""Click-based CLI for fishsuite."""
from __future__ import annotations

import os
# Force a non-interactive matplotlib backend BEFORE any module imports pyplot.
# Worker threads on Windows otherwise inherit TkAgg, which calls into Tcl from
# the wrong thread once Bio-Formats' JVM is alive -> Tcl_AsyncDelete crash that
# brings the JVM down with it. The CLI is always headless; the GUI sets its
# own backend via Qt before importing fishsuite, so this is safe.
os.environ.setdefault("MPLBACKEND", "Agg")

import sys
from pathlib import Path

import click

from . import __version__


@click.group()
@click.version_option(__version__)
def cli():
    """fishsuite - standalone RNA-FISH / IF analysis pipeline."""


@cli.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True, dir_okay=False),
              help="Path to a fishsuite YAML config (or built-in preset).")
@click.option("--input-dir", "-i", required=True, type=click.Path(exists=True, file_okay=False),
              help="Folder of images (or folder of subfolders).")
@click.option("--output-dir", "-o", required=True, type=click.Path(file_okay=False),
              help="Where to write outputs.")
@click.option("--parallel", "-p", default="auto",
              help="Worker count: 'auto' (default) or an integer.")
@click.option("--resume", is_flag=True, help="Skip images that already have outputs.")
@click.option("--dry-run", is_flag=True, help="Discover inputs and print plan; do not process.")
@click.option("--verbose", "-v", is_flag=True, help="Print full tracebacks on per-image failures.")
def run(config, input_dir, output_dir, parallel, resume, dry_run, verbose):
    """Run the full pipeline on a folder of images."""
    from .runner import run_batch
    summary = run_batch(
        config_path=Path(config),
        input_dir=Path(input_dir),
        output_dir=Path(output_dir),
        parallel=parallel,
        resume=resume,
        dry_run=dry_run,
        verbose=verbose,
    )
    click.echo(f"Summary: {summary}")


@cli.command()
def init():
    """Interactive setup wizard (Phase-3 placeholder)."""
    click.echo("fishsuite init: interactive setup wizard is a Phase-3 deliverable.")
    click.echo("For now, copy a preset and edit it:")
    from .config import schema as _s
    preset_dir = Path(_s.__file__).parent / "presets"
    for p in sorted(preset_dir.glob("*.yaml")):
        click.echo(f"  {p}")


@cli.command()
@click.argument("image_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--config", "-c", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--output-dir", "-o", required=True, type=click.Path(file_okay=False))
def preview(image_path, config, output_dir):
    """Run the pipeline on a single image (preview / debug)."""
    from .runner import run_batch
    p = Path(image_path)
    tmp_in = p.parent
    summary = run_batch(
        config_path=Path(config),
        input_dir=tmp_in,
        output_dir=Path(output_dir),
        parallel=1,
    )
    click.echo(f"Summary: {summary}")


@cli.group()
def presets():
    """Manage built-in presets."""


@presets.command("list")
def presets_list():
    """List built-in presets."""
    from .config import schema as _s
    preset_dir = Path(_s.__file__).parent / "presets"
    for p in sorted(preset_dir.glob("*.yaml")):
        click.echo(f"{p.stem}\t{p}")


@presets.command("show")
@click.argument("name")
def presets_show(name):
    """Print a preset YAML."""
    from .config import schema as _s
    preset_dir = Path(_s.__file__).parent / "presets"
    p = preset_dir / f"{name}.yaml"
    if not p.exists():
        click.echo(f"Preset not found: {p}", err=True)
        sys.exit(2)
    click.echo(p.read_text(encoding="utf-8"))


@cli.command()
def gui():
    """Launch the PySide6 desktop launcher."""
    from . import gui as _gui
    rc = _gui.main()
    sys.exit(rc)


# ===========================================================================
# POST-RUN UTILITIES - friendly wrappers over the standalone utility modules.
#
# These operate on a COMPLETED run output directory (the folder `run` wrote,
# containing run_config.json / per_image_summary.csv / spot_metrics.csv /
# masks/ / figures/). They REUSE the saved nuclei masks + MIAT spots from that
# run - they never re-segment or re-detect and they do NOT touch the GPU.
#
# The underlying functions are imported lazily inside thin module-level
# indirection helpers (``_backfill_run`` / ``_build_walkthrough_figure``) so
# (a) importing the CLI stays cheap and (b) the tests can monkeypatch the
# dispatch target without loading the heavy image stack. The original
# ``python -m fishsuite.core.coloc_backfill`` / ``...walkthrough_figure``
# entry points keep working unchanged (back-compat).
# ===========================================================================
def _backfill_run(run_dir, staging_dir=None, input_dir=None, **kwargs):
    """Indirection to :func:`fishsuite.core.coloc_backfill.backfill_run`
    (lazy import; monkeypatch target for the CLI tests)."""
    from .core.coloc_backfill import backfill_run
    return backfill_run(run_dir, staging_dir=staging_dir, input_dir=input_dir,
                        **kwargs)


def _build_walkthrough_figure(run_dir, staging_dir=None, input_dir=None,
                              image_key=None, out_path=None):
    """Indirection to
    :func:`fishsuite.core.walkthrough_figure.build_walkthrough_figure`
    (lazy import; monkeypatch target for the CLI tests)."""
    from .core.walkthrough_figure import build_walkthrough_figure
    return build_walkthrough_figure(
        run_dir, staging_dir=staging_dir, input_dir=input_dir,
        image_key=image_key, out_path=out_path,
    )


def _singlecell_run(run_dir, **kwargs):
    """Indirection to :func:`fishsuite.core.singlecell.singlecell_run`
    (lazy import; monkeypatch target for the CLI tests)."""
    from .core.singlecell import singlecell_run
    return singlecell_run(run_dir, **kwargs)


def _pixelpattern_run(run_dir, staging_dir=None, input_dir=None, **kwargs):
    """Indirection to :func:`fishsuite.core.pixel_pattern.pixelpattern_run`
    (lazy import; monkeypatch target for the CLI tests)."""
    from .core.pixel_pattern import pixelpattern_run
    return pixelpattern_run(run_dir, staging_dir=staging_dir,
                            input_dir=input_dir, **kwargs)


def _if_pub_images_run(run_dir, **kwargs):
    """Indirection to
    :func:`fishsuite.core.modes.if_pub_images.regenerate_pub_images`
    (lazy import; monkeypatch target for the CLI tests)."""
    from .core.modes.if_pub_images import regenerate_pub_images
    return regenerate_pub_images(run_dir, **kwargs)


def _friendly_postrun_error(exc: Exception, run: Path) -> str:
    """Translate an expected/user-fixable backfill/walkthrough exception into a
    plain-English, actionable message (no raw traceback). Falls back to the
    exception text for anything we did not anticipate."""
    msg = str(exc)
    low = msg.lower()
    if isinstance(exc, ValueError) and "input_dir" in low:
        # the run didn't record where its VSIs live and none was given
        return (
            "Could not find the source images for this run.\n"
            f"  The run at {run} does not record an input/staging folder, and you "
            "did not pass one.\n"
            "  Re-run with the path to the VSI staging folder, e.g.:\n"
            f"      fishsuite backfill --run \"{run}\" --staging <path-to-VSI-staging>"
        )
    if isinstance(exc, FileNotFoundError):
        if "run_config.json" in low:
            return (
                f"This does not look like a finished fishsuite run: no run_config.json "
                f"in {run}.\n"
                "  Point --run at the OUTPUT folder a completed run produced "
                "(it contains run_config.json, per_image_summary.csv, masks/, figures/)."
            )
        if "per_well" in low:
            return (
                f"This does not look like a finished if_intensity run: {msg}\n"
                f"  Looking inside {run}. `if-pub-images` rebuilds the plate map from "
                "per_well.csv (written by an if_intensity run). Point --run at the "
                "OUTPUT folder such a run produced."
            )
        if "nuclei_metrics" in low:
            return (
                f"This run looks incomplete: {msg}\n"
                f"  Looking inside {run}. The single-cell / pixel-pattern utilities "
                "need nuclei_metrics.csv (the per-nucleus table) from a completed run."
            )
        if "per_image_summary" in low or "spot_metrics" in low:
            return (
                f"This run looks incomplete: {msg}\n"
                f"  Looking inside {run}. Has the run finished? The post-run "
                "utilities need per_image_summary.csv + spot_metrics.csv + the saved "
                "masks/ from a completed run."
            )
        if "mask" in low:
            return (
                f"No saved nuclei masks found for this run: {msg}\n"
                f"  Looking in {run / 'masks'}. Has the run finished, and was it run "
                "with save_masks on? If the run is complete, also pass the VSI "
                "source via --staging <path>."
            )
        if "step01" in low or "panel" in low or ".png" in low:
            return (
                f"Could not build the walkthrough figure: {msg}\n"
                f"  Looking under {run / 'pipeline_walkthrough'}. This figure is built "
                "from a run's own per-step PNGs; make sure the run produced them "
                "(save_qc_overlays / save_publication_images on)."
            )
        return f"A required file was not found: {msg}\n  (run dir: {run})"
    # unexpected - still avoid dumping a traceback at the user
    return f"{type(exc).__name__}: {msg}"


_BACKFILL_HELP = """\
Backfill the extra colocalization products onto a COMPLETED run (CPU-only).

\b
CPU-only - does NOT use the GPU. Reuses the nuclei masks and the detected MIAT
spots from a completed run; it re-reads only the QKI/protein channel pixels and
recomputes the QKI-at-MIAT null, so it never re-segments or re-detects anything.

\b
It emits the products that older runs are missing:
  - coloc_null_draws.csv        (the 1000 pooled random-null draws)
  - coloc_null_summary.csv      (pooled enrichment / z / empirical-p per image)
  - coloc_radial_profile.csv    (QKI enrichment in concentric rings around MIAT)
  - a QKI enrichment montage PNG (figures/07_coloc/79_...png)

The source VSIs are found automatically from the folder the run recorded; pass
--staging only if that is wrong or unavailable.

\b
Examples:
  # the common case - everything auto-detected:
  fishsuite backfill --run ./results/my_run_20260605

  # point at the image staging folder explicitly:
  fishsuite backfill --run ./results/my_run --staging ./raw/my_experiment

  # montage only (skip the CSV products):
  fishsuite backfill --run ./results/my_run --no-null-draws --no-radial
"""


@cli.command(help=_BACKFILL_HELP, short_help="CPU coloc backfill onto a finished run.")
@click.option("--run", "run_dir", required=True,
              type=click.Path(exists=True, file_okay=False),
              help="Completed run output directory (the folder a run produced).")
@click.option("--staging", "staging", default=None,
              type=click.Path(exists=True, file_okay=False),
              help="Folder holding the source VSIs. Auto-detected from the run if "
                   "omitted; pass it if auto-detection fails.")
@click.option("--input", "input_dir", default=None,
              type=click.Path(exists=True, file_okay=False),
              help="Alternate source folder for the VSIs (rarely needed).")
@click.option("--seed", default=0, show_default=True, type=int,
              help="Random seed for the null/montage (kept deterministic).")
@click.option("--no-null-draws", is_flag=True,
              help="Skip writing coloc_null_draws.csv / coloc_null_summary.csv.")
@click.option("--no-radial", is_flag=True,
              help="Skip writing coloc_radial_profile.csv.")
@click.option("--no-montage", is_flag=True,
              help="Skip rendering the QKI enrichment montage PNG.")
@click.option("--rotation", is_flag=True,
              help="ALSO compute the rotation 'proper background' null (keep-N "
                   "constellation redraw) -> coloc_rotation_null_summary.csv + "
                   "coloc_rotation_null_draws.csv. OFF by default (opt-in).")
def backfill(run_dir, staging, input_dir, seed, no_null_draws, no_radial,
             no_montage, rotation):
    run = Path(run_dir)
    click.echo(f"[backfill] CPU-only - reusing saved masks + MIAT spots in {run}")
    try:
        res = _backfill_run(
            run,
            staging_dir=staging,
            input_dir=input_dir,
            do_null_draws=not no_null_draws,
            do_radial=not no_radial,
            do_montage=not no_montage,
            do_rotation=rotation,
            seed=seed,
        )
    except (FileNotFoundError, ValueError) as exc:
        click.echo(_friendly_postrun_error(exc, run), err=True)
        sys.exit(2)
    written = res.get("written", {}) if isinstance(res, dict) else {}
    if written:
        click.echo("[backfill] wrote:")
        for k, v in written.items():
            click.echo(f"    {k}: {v}")
    else:
        click.echo("[backfill] no products written (nothing to do / all skipped).")
    n_fail = (res.get("gate", {}) or {}).get("n_fail", 0) if isinstance(res, dict) else 0
    if n_fail:
        click.echo(f"[backfill] WARNING: {n_fail} image(s) failed the "
                   f"self-validation gate - inspect before trusting the output.",
                   err=True)
        sys.exit(1)


_WALKTHROUGH_HELP = """\
Build the 8-panel publication "pipeline walkthrough" figure for a finished run.

\b
Assembles one labeled micrograph figure (DAPI -> segmentation -> MIAT FISH ->
spot detection -> QKI IF -> QKI threshold -> MIAT-on-QKI -> merge) from the run's
OWN per-step images. One panel (MIAT spots on the thresholded QKI field) is
re-rendered from the QKI pixels (CPU; reuses the run's saved spots). A
representative image and the output path are chosen automatically.

\b
Defaults:
  --image   a representative image is auto-picked (the MIAT-OE image if present)
  --out     <run>/figures/07_coloc/79_pipeline_walkthrough.png

\b
Examples:
  fishsuite walkthrough --run ./results/my_run
  fishsuite walkthrough --run ./results/my_run --image "cond_A__field_01"
  fishsuite walkthrough --run ./results/my_run --out ./figures/walkthrough.png
"""


@cli.command(help=_WALKTHROUGH_HELP,
             short_help="Build the 8-panel pipeline-walkthrough figure.")
@click.option("--run", "run_dir", required=True,
              type=click.Path(exists=True, file_okay=False),
              help="Completed run output directory.")
@click.option("--image", "image_key", default=None,
              help="Panel-prefix image key (default: a representative image).")
@click.option("--out", "out_path", default=None, type=click.Path(dir_okay=False),
              help="Output PNG path (default: "
                   "<run>/figures/07_coloc/79_pipeline_walkthrough.png).")
@click.option("--staging", "staging", default=None,
              type=click.Path(exists=True, file_okay=False),
              help="Folder holding the source VSIs (for the rendered panel). "
                   "Auto-detected from the run if omitted.")
@click.option("--input", "input_dir", default=None,
              type=click.Path(exists=True, file_okay=False),
              help="Alternate source folder for the VSIs (rarely needed).")
def walkthrough(run_dir, image_key, out_path, staging, input_dir):
    run = Path(run_dir)
    click.echo(f"[walkthrough] building pipeline-walkthrough figure for {run}")
    try:
        out = _build_walkthrough_figure(
            run, staging_dir=staging, input_dir=input_dir,
            image_key=image_key, out_path=out_path,
        )
    except (FileNotFoundError, ValueError) as exc:
        click.echo(_friendly_postrun_error(exc, run), err=True)
        sys.exit(2)
    click.echo(f"[walkthrough] wrote: {out}")


_SINGLECELL_HELP = """\
Single-cell (per-nucleus) treatment analysis of a COMPLETED run (CPU-only).

\b
CPU-only - does NOT use the GPU and re-reads NO images. It reads the run's
nuclei_metrics.csv (the per-nucleus table) and, for every meaningful per-nucleus
metric, computes:
  - DOSE-RESPONSE vs a per-nucleus abundance axis (default: nuclear_spot_count)
  - MATCHED-ABUNDANCE control-vs-perturbation comparison within each abundance bin
  - GROUP (condition-depth) HETEROGENEITY (perturbation split by its own tertile)
  - DISTRIBUTION + per-replicate Welch t
  - a SATURATION headline when the run carries the rotation-null association columns

\b
It writes an explorable Excel + locked-style NT-vs-perturbation SuperPlots + a
plain-language findings file under <run>/deliverables/singlecell/.

GENERIC: the abundance axis and the two groups to compare are auto-detected and
overridable (--abundance-col / --group-a / --group-b); nothing is hardcoded to a
particular gene.

\b
Examples:
  fishsuite singlecell --run ./results/my_run
  fishsuite singlecell --run ./results/my_run --abundance-col nuclear_spot_count
  fishsuite singlecell --run ./results/my_run --group-a NT --group-b KD --no-figures
"""


@cli.command(help=_SINGLECELL_HELP,
             short_help="CPU single-cell (per-nucleus) treatment analysis.")
@click.option("--run", "run_dir", required=True,
              type=click.Path(exists=True, file_okay=False),
              help="Completed run output directory (contains nuclei_metrics.csv).")
@click.option("--abundance-col", default=None,
              help="Per-nucleus abundance axis (default: nuclear_spot_count).")
@click.option("--group-a", default=None, help="Control group label (default: auto).")
@click.option("--group-b", default=None, help="Perturbation group label (default: auto).")
@click.option("--include-secondary", is_flag=True,
              help="Do NOT drop secondary-only nuclei (default: dropped).")
@click.option("--out-subdir", default="singlecell", show_default=True,
              help="Sub-folder under <run>/deliverables/ for the outputs.")
@click.option("--seed", default=0, show_default=True, type=int)
@click.option("--no-figures", is_flag=True, help="Skip the SuperPlots.")
@click.option("--no-excel", is_flag=True, help="Skip the Excel workbook.")
def singlecell(run_dir, abundance_col, group_a, group_b, include_secondary,
               out_subdir, seed, no_figures, no_excel):
    run = Path(run_dir)
    click.echo(f"[singlecell] CPU-only - reading nuclei_metrics.csv in {run}")
    try:
        res = _singlecell_run(
            run, abundance_col=abundance_col, group_a=group_a, group_b=group_b,
            exclude_secondary=not include_secondary, out_subdir=out_subdir,
            seed=seed, do_figures=not no_figures, do_excel=not no_excel,
        )
    except (FileNotFoundError, ValueError) as exc:
        click.echo(_friendly_postrun_error(exc, run), err=True)
        sys.exit(2)
    written = res.get("written", {}) if isinstance(res, dict) else {}
    click.echo(f"[singlecell] {res.get('n_metrics', 0)} metrics, "
               f"{res.get('n_figures', 0)} figures -> {res.get('out_dir', run)}")
    for k, v in written.items():
        click.echo(f"    {k}: {v}")


_PIXELPATTERN_HELP = """\
Per-nucleus PIXEL-pattern metrics for a COMPLETED run (CPU-only).

\b
CPU-only - does NOT use the GPU. Like `backfill`, it REUSES the run's saved
nucleus masks and re-reads only the raw channel pixels (at the recomputed
analysis z-plane), then computes per-nucleus pixel metrics that need no spot
calling:
  - PERINUCLEAR / RADIAL index (RNA1 + partner + DAPI + a secondary-only control
    = a one-number stain QC of antibody localization)
  - GINI + top-5% / top-10% concentration per channel
  - FOCI-BAND counts (partner spots per nucleus above intensity floors)
  - DECILE intensity-sweep (partner vs RNA1 intensity)

\b
It appends the metrics to pixel_pattern_metrics.csv and writes NT-vs-condition
SuperPlots, a stain-QC panel, an Excel and a findings file under
<run>/deliverables/pixelpattern/.

The source raw images are found automatically from the folder the run recorded;
pass --staging only if that is wrong or unavailable.

\b
--secondary-match designates WHICH secondary-only well is the CLEAN control (some
secondary-only wells can be contaminated); a substring of the image/condition
(e.g. "well12") restricts the control to that well. Default = all secondary-only.

\b
Examples:
  fishsuite pixelpattern --run ./results/my_run
  fishsuite pixelpattern --run ./results/my_run --staging ./raw/my_experiment
  fishsuite pixelpattern --run ./results/my_run --secondary-match well12
"""


@cli.command(help=_PIXELPATTERN_HELP,
             short_help="CPU per-nucleus pixel-pattern metrics + stain QC.")
@click.option("--run", "run_dir", required=True,
              type=click.Path(exists=True, file_okay=False),
              help="Completed run output directory (contains masks/ + nuclei_metrics.csv).")
@click.option("--staging", "staging", default=None,
              type=click.Path(exists=True, file_okay=False),
              help="Folder holding the source raw images. Auto-detected from the "
                   "run if omitted; pass it if auto-detection fails.")
@click.option("--input", "input_dir", default=None,
              type=click.Path(exists=True, file_okay=False),
              help="Alternate source folder for the raw images (rarely needed).")
@click.option("--secondary-match", default=None,
              help="Substring picking the CLEAN secondary-only well/condition "
                   "(e.g. 'well12'); default = all secondary-only nuclei.")
@click.option("--out-subdir", default="pixelpattern", show_default=True,
              help="Sub-folder under <run>/deliverables/ for the outputs.")
@click.option("--seed", default=0, show_default=True, type=int)
@click.option("--no-figures", is_flag=True, help="Skip the figures.")
@click.option("--no-excel", is_flag=True, help="Skip the Excel workbook.")
def pixelpattern(run_dir, staging, input_dir, secondary_match, out_subdir, seed,
                 no_figures, no_excel):
    run = Path(run_dir)
    click.echo(f"[pixelpattern] CPU-only - reusing saved masks + re-reading raw in {run}")
    try:
        res = _pixelpattern_run(
            run, staging_dir=staging, input_dir=input_dir,
            secondary_match=secondary_match, out_subdir=out_subdir, seed=seed,
            do_figures=not no_figures, do_excel=not no_excel,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        click.echo(_friendly_postrun_error(exc, run), err=True)
        sys.exit(2)
    written = res.get("written", {}) if isinstance(res, dict) else {}
    click.echo(f"[pixelpattern] {res.get('n_images', 0)} images, "
               f"{res.get('n_nuclei', 0)} nuclei, {res.get('n_figures', 0)} figures "
               f"-> {res.get('out_dir', run)}")
    for k, v in written.items():
        click.echo(f"    {k}: {v}")
    sq = res.get("stain_qc") if isinstance(res, dict) else None
    if sq:
        click.echo(f"[pixelpattern] stain QC (partner perinuclear index): {sq}")


_IFPUB_HELP = """\
Regenerate the IF publication images from a COMPLETED if_intensity run (CPU-only).

\b
CPU-only - does NOT use the GPU and NEVER re-segments. It rebuilds the plate map
from the run's own per_well.csv, picks the representative FOV per well from
per_fov.csv, re-reads only DAPI + each well's routed signal channel from the raw
VSIs, and renders (per representative well per secondary, for both sources):
  - a signal|DAPI|Merge channel panel (each panel its own scalebar)
  - a standalone Merge micrograph
  - one WT/KO/secondary-only composite per secondary/source

\b
Two SOURCES: single_plane (representative FOV of the quantification set) and
picked_z (the SINGLE best-focus z-plane - var(laplace)*mean on DAPI, central
band, NO max-projection). Uses the RAISED per-secondary display floors so WT
cytoplasm reads near-zero; ceiling = the WT-primary signal percentile. The same
(vmin,vmax) is applied to WT/KO/secondary-only (never per-image auto-contrast).

\b
DISPLAY-RANGE CONTROL (all optional; defaults reproduce prior behaviour):
  --floor SEC=VALUE / SEC:source=VALUE   signal FLOOR (vmin). Bare SEC applies to
                                         BOTH sources; SEC:source scopes to one
                                         (e.g. 647:single_plane=5000). More-
                                         specific source key wins.
  --ceiling SEC=VALUE / SEC:source=VALUE explicit signal CEILING (vmax); overrides
                                         --ceiling-pct for that secondary/source.
  --ceiling-pct PCT                      WT-primary percentile ceiling where no
                                         explicit --ceiling is set (default 99.5).
  --dapi-floor / --dapi-ceiling VALUE    FIXED DAPI vmin/vmax across all panels
                                         (e.g. --dapi-ceiling 8000 so DAPI is not
                                         over-exposed). Unset = per-image DAPI.
  --per-image-ceiling                    each panel's signal ceiling from ITS OWN
                                         percentile (slightly non-rigorous;
                                         breaks cross-panel comparability). An
                                         explicit --ceiling still wins.
The applied (vmin,vmax) per source/secondary/channel is logged to
publication_images/qc_nuclear_dominance.txt.

\b
Source dirs + floors + ceilings + DAPI range + label default from
if_run_context.json (written by the run); pass the flags above to override.
Output goes to <run>/publication_images/.

\b
CRITICAL CHANNEL RULE: 647 wells render only the 640-channel signal + DAPI;
568/565 wells only the 561-channel signal + DAPI. Logged per read to
publication_images/channel_rule_log.txt.

\b
Examples:
  fishsuite if-pub-images --run ./results/my_run
  fishsuite if-pub-images --run ./results/my_run --staging ./raw/single-plane \\
      --zstack ./raw/z-stacks --source single_plane --source picked_z \\
      --floor 647:single_plane=5000 --ceiling 647:single_plane=55000 \\
      --floor 647:picked_z=5500 --dapi-ceiling 8000 --label QKI
"""


@cli.command("if-pub-images", help=_IFPUB_HELP,
             short_help="CPU regenerate IF publication images on a finished run.")
@click.option("--run", "run_dir", required=True,
              type=click.Path(exists=True, file_okay=False),
              help="Completed if_intensity run output dir (contains per_well.csv).")
@click.option("--staging", "staging", default=None,
              type=click.Path(exists=True, file_okay=False),
              help="Single-plane raw dir (subfolders per well). Auto-detected "
                   "from if_run_context.json if omitted.")
@click.option("--zstack", "zstack", default=None,
              type=click.Path(exists=True, file_okay=False),
              help="Z-stack raw dir for the picked_z source (subfolders per well). "
                   "Auto-detected from if_run_context.json if omitted.")
@click.option("--source", "sources", multiple=True,
              type=click.Choice(["single_plane", "picked_z"]),
              help="Source(s) to render (repeatable). Default: both / as recorded.")
@click.option("--floor", "floors", multiple=True,
              help="Signal display floor (vmin) as SEC=VALUE (both sources) or "
                   "SEC:source=VALUE (that source only), e.g. 647=5000 or "
                   "647:single_plane=5000. Repeatable. Default: 647=5000, "
                   "568=3500 / as recorded.")
@click.option("--ceiling", "ceilings", multiple=True,
              help="EXPLICIT signal ceiling (vmax) as SEC=VALUE or "
                   "SEC:source=VALUE (e.g. 647:single_plane=55000). Overrides "
                   "--ceiling-pct for that secondary/source; unset pairs use the "
                   "WT-primary percentile. Repeatable.")
@click.option("--ceiling-pct", default=None, type=float,
              help="Ceiling percentile of the WT-primary signal (default 99.5). "
                   "Used only where no explicit --ceiling is given.")
@click.option("--dapi-floor", "dapi_floor", default=None, type=float,
              help="Fixed DAPI display floor (vmin) across ALL panels. "
                   "Unset = per-image DAPI normalization (legacy).")
@click.option("--dapi-ceiling", "dapi_ceiling", default=None, type=float,
              help="Fixed DAPI display ceiling (vmax) across ALL panels, e.g. "
                   "8000 to keep DAPI from over-exposing. Unset = per-image.")
@click.option("--per-image-ceiling", "per_image_ceiling", is_flag=True, default=False,
              help="Compute each panel's signal ceiling from THAT image's own "
                   "percentile instead of the shared WT-primary ceiling. "
                   "Slightly non-rigorous (breaks cross-panel comparability) - a "
                   "fallback. An explicit --ceiling still wins. Default OFF "
                   "(shared display).")
@click.option("--scalebar", "scalebar_um", default=None, type=float,
              help="Scalebar length in microns (default 20).")
@click.option("--label", default=None, help="Signal label in panels (e.g. QKI).")
def if_pub_images_cmd(run_dir, staging, zstack, sources, floors, ceilings,
                      ceiling_pct, dapi_floor, dapi_ceiling, per_image_ceiling,
                      scalebar_um, label):
    run = Path(run_dir)
    click.echo(f"[if-pub-images] CPU-only - reusing per_well.csv + re-reading raw "
               f"(no GPU / no re-segmentation) in {run}")

    _VALID_SOURCES = ("single_plane", "picked_z")

    def _parse_range_opt(items, flag):
        """Parse a repeatable SEC=VALUE / SEC:source=VALUE option into a dict.

        Keys are kept as ``"647"`` or ``"647:single_plane"``; the renderer
        resolves the more-specific source-scoped key first. Exits(2) with a
        plain-English message on any malformed token."""
        if not items:
            return None
        out = {}
        for it in items:
            s = str(it)
            if "=" not in s:
                click.echo(f"Bad {flag} '{it}' (expected SEC=VALUE or "
                           f"SEC:source=VALUE, e.g. 647=5000 or "
                           f"647:single_plane=5000).", err=True)
                sys.exit(2)
            k, v = s.split("=", 1)
            k = k.strip()
            if ":" in k:
                sec, src = k.split(":", 1)
                src = src.strip()
                if src not in _VALID_SOURCES:
                    click.echo(f"Bad {flag} '{it}': source must be one of "
                               f"{list(_VALID_SOURCES)}, got '{src}'.", err=True)
                    sys.exit(2)
                k = f"{sec.strip()}:{src}"
            try:
                out[k] = float(v)
            except ValueError:
                click.echo(f"Bad {flag} '{it}': value '{v}' is not a number.",
                           err=True)
                sys.exit(2)
        return out

    floor_map = _parse_range_opt(floors, "--floor")
    ceiling_map = _parse_range_opt(ceilings, "--ceiling")
    try:
        res = _if_pub_images_run(
            run, staging_dir=staging, zstack_dir=zstack,
            sources=(list(sources) or None), floors=floor_map,
            ceilings=ceiling_map, ceiling_pct=ceiling_pct,
            dapi_floor=dapi_floor, dapi_ceiling=dapi_ceiling,
            per_image_ceiling=(True if per_image_ceiling else None),
            scalebar_um=scalebar_um, label=label,
        )
    except (FileNotFoundError, ValueError) as exc:
        click.echo(_friendly_postrun_error(exc, run), err=True)
        sys.exit(2)
    if isinstance(res, dict):
        click.echo(f"[if-pub-images] {res.get('channels', 0)} channel panels, "
                   f"{res.get('merge', 0)} merges, {res.get('composite', 0)} composites, "
                   f"sources={res.get('sources', [])} -> {res.get('out_dir', run)}")


_POSTRUN_HELP = """\
One-shot "just make my figures" - run ALL post-run utilities on a finished run.

\b
Runs, in order, on the given run directory:
  1. backfill     (CPU; the coloc null draws + radial profile + QKI montage)
  2. walkthrough  (the 8-panel pipeline-walkthrough figure)

CPU-only - does not use the GPU. Each step prints a progress line, and a final
summary lists every file produced. If one step fails it is reported plainly and
the others are still attempted (the command then exits non-zero).

\b
Example (the common case - nothing else needed):
  fishsuite postrun --run ./results/my_run_20260605
"""


@cli.command(help=_POSTRUN_HELP,
             short_help="One-shot: run ALL post-run utilities on a run.")
@click.option("--run", "run_dir", required=True,
              type=click.Path(exists=True, file_okay=False),
              help="Completed run output directory.")
@click.option("--staging", "staging", default=None,
              type=click.Path(exists=True, file_okay=False),
              help="Folder holding the source VSIs. Auto-detected if omitted.")
@click.option("--input", "input_dir", default=None,
              type=click.Path(exists=True, file_okay=False),
              help="Alternate source folder for the VSIs (rarely needed).")
@click.option("--image", "image_key", default=None,
              help="Walkthrough panel-prefix image key (default: auto-picked).")
@click.option("--seed", default=0, show_default=True, type=int,
              help="Random seed for the backfill null/montage.")
def postrun(run_dir, staging, input_dir, image_key, seed):
    run = Path(run_dir)
    click.echo("=" * 70)
    click.echo(f"[postrun] one-shot post-run utilities (CPU-only) for:\n    {run}")
    click.echo("=" * 70)

    produced: list[str] = []
    failures: list[str] = []

    # ---- step 1: backfill --------------------------------------------------
    click.echo("\n[postrun] step 1/2: backfill (coloc null draws + radial + montage)")
    try:
        res = _backfill_run(
            run, staging_dir=staging, input_dir=input_dir,
            do_null_draws=True, do_radial=True, do_montage=True, seed=seed,
        )
        for v in (res.get("written", {}) or {}).values():
            produced.append(str(v))
        n_fail = (res.get("gate", {}) or {}).get("n_fail", 0)
        if n_fail:
            click.echo(f"    WARNING: {n_fail} image(s) failed the self-validation "
                       f"gate - inspect before trusting the backfilled output.",
                       err=True)
            failures.append(f"backfill: {n_fail} image(s) failed the gate")
        else:
            click.echo("    backfill OK")
    except Exception as exc:  # noqa: BLE001 - report, don't crash; try the next step
        click.echo("    backfill FAILED:", err=True)
        click.echo("    " + _friendly_postrun_error(exc, run).replace("\n", "\n    "),
                   err=True)
        failures.append(f"backfill: {type(exc).__name__}")

    # ---- step 2: walkthrough ----------------------------------------------
    click.echo("\n[postrun] step 2/2: walkthrough (8-panel pipeline figure)")
    try:
        out = _build_walkthrough_figure(
            run, staging_dir=staging, input_dir=input_dir, image_key=image_key,
        )
        produced.append(str(out))
        click.echo(f"    walkthrough OK: {out}")
    except Exception as exc:  # noqa: BLE001
        click.echo("    walkthrough FAILED:", err=True)
        click.echo("    " + _friendly_postrun_error(exc, run).replace("\n", "\n    "),
                   err=True)
        failures.append(f"walkthrough: {type(exc).__name__}")

    # ---- final summary -----------------------------------------------------
    click.echo("\n" + "=" * 70)
    click.echo("[postrun] summary")
    click.echo("=" * 70)
    if produced:
        click.echo(f"produced {len(produced)} file(s):")
        for p in produced:
            click.echo(f"    {p}")
    else:
        click.echo("produced no files.")
    if failures:
        click.echo("\nthe following step(s) had problems:", err=True)
        for f in failures:
            click.echo(f"    - {f}", err=True)
        sys.exit(1)
    click.echo("\n[postrun] all post-run utilities completed.")


@cli.command()
@click.option("--run", "run_dir", required=True,
              type=click.Path(exists=True, file_okay=False),
              help="Finished fishsuite output directory to report on. It is READ, "
                   "never modified.")
@click.option("--groups", "groups", multiple=True, metavar="NAME=well1,well2,...",
              help="Condition GROUP definition, repeatable. A condition is one WELL; "
                   "a group is the condition several wells belong to and is what gets "
                   "compared. e.g. --groups WT=WT_1,WT_2,WT_3 "
                   "--groups QKI-KO=KO_1,KO_2,KO_3. The FIRST group given is the "
                   "reference unless --reference says otherwise. Omit this and the "
                   "groups recorded in the run's own config are used; failing that, "
                   "every well is its own group.")
@click.option("--groups-file", "groups_file", default=None,
              type=click.Path(exists=True, dir_okay=False),
              help="A report_groups.yaml staged beside the run, carrying `groups`, "
                   "`group_order`, `reference`, `well_from_image`, `exclude_fields` "
                   "and a free-text `note`. Command-line flags override any key it "
                   "sets, so one file makes a run's report reproducible.")
@click.option("--reference", default=None, metavar="NAME",
              help="Group every other group is compared against. Default: the first "
                   "group.")
@click.option("--group-order", "group_order", default=None, metavar="A,B,C",
              help="Plotting and reporting order of the groups. Groups present in the "
                   "data but missing here are appended in sorted order.")
@click.option("--well-from-image", "well_from_image", default=None, metavar="REGEX",
              help="Recover the WELL from the image name with a regular expression "
                   "carrying exactly one capture group. Use this when the run recorded "
                   "the LINE as its condition and the well only in the file name; "
                   "without it every field of a line collapses into one well and no "
                   "test is possible. Every biological image must match or the command "
                   "refuses to run.")
@click.option("--out", "out_dir", default=None, type=click.Path(file_okay=False),
              help="Where to write the report. Default: <run>/report_<timestamp>/.")
@click.option("--exclude-field", "exclude_field", multiple=True, metavar="NAME",
              help="Drop one field of view (the image name as it appears in "
                   "per_image_summary.csv), repeatable. Each --exclude-field must be "
                   "followed by its own --reason; the pair is recorded in the workbook "
                   "and in every figure's filter line.")
@click.option("--reason", "reason", multiple=True, metavar="TEXT",
              help="Why the preceding --exclude-field was dropped. Given in the same "
                   "order as the --exclude-field flags, one each.")
@click.option("--all-pairs/--vs-reference", "all_pairs", default=None,
              help="Report EVERY unordered pair of condition groups, or only each "
                   "group against the reference. Default: all pairs when there are "
                   "more than two groups, reference-only when there are two, which "
                   "are the same thing at two groups.")
@click.option("--nucleus-filter", type=click.Choice(["all", "sampled"]), default="all",
              show_default=True,
              help="Which nuclei enter the report. 'all' uses every segmented "
                   "nucleus. 'sampled' keeps only those the run flagged "
                   "sampled_in_analysis, which is the fixed-N balanced set; use it "
                   "to match an analysis built on that set. The run must carry the "
                   "column or the command refuses.")
@click.option("--peak-floor", "peak_floor", default=None, metavar="rna=1000,rna2=1200",
              help="Apply a peak-intensity floor to the run's spots AFTER detection, "
                   "then re-derive the per-nucleus counts. fishsuite's own floor gate "
                   "is post-detection too, so this reproduces a gated run without "
                   "re-detecting. Boundary is at or above the floor. Columns that "
                   "cannot be re-derived post hoc are named in the workbook rather "
                   "than silently carrying a pre-gate value.")
@click.option("--caveat-file", "caveat_file", default=None,
              type=click.Path(exists=True, dir_okay=False),
              help="A text or markdown file whose contents are inserted into "
                   "READOUT.md immediately after the run path, and recorded in the "
                   "Read me sheet. Use it to carry a calibration or interpretation "
                   "caveat from the analysis record into the deliverable.")
@click.option("--style", type=click.Choice(["brian", "plain"]), default="brian",
              show_default=True,
              help="Figure style. 'brian' is the locked lab style: Okabe-Ito colours, "
                   "600-dpi PNG plus editable-text SVG, filter line and single "
                   "croppable footnote on every chart.")
@click.option("--alpha", default=0.05, show_default=True, type=float,
              help="Significance level for the Welch gate and the Holm family.")
@click.option("--qc-min-nuclei", default=5, show_default=True, type=int,
              help="Fields with fewer nuclei than this are FLAGGED in the Per field "
                   "sheet. Nothing is dropped by this flag.")
@click.option("--sec-min-nuclei", default=10, show_default=True, type=int,
              help="Secondary-only control fields with fewer nuclei than this are "
                   "excluded by rule, with the rule recorded in the workbook.")
@click.option("--engine-repo", default=None, type=click.Path(exists=True, file_okay=False),
              help="fishsuite checkout whose git HEAD is recorded in Run provenance.")
@click.option("--preset", default=None, type=click.Path(exists=True, dir_okay=False),
              help="Preset YAML the run used; its md5 is recorded in Run provenance.")
@click.option("--no-figures", is_flag=True,
              help="Write the workbook and readout only; skip every figure.")
@click.option("--no-coloc-panel", is_flag=True,
              help="Skip the standard colocalization panel on an rna_rna or "
                   "rna_protein run.")
@click.option("--stamp", default="", metavar="TEXT",
              help="Timestamp used in the default output directory name. Defaults to "
                   "now.")
def report(run_dir, groups, groups_file, reference, group_order, well_from_image,
           out_dir, exclude_field, reason, all_pairs, nucleus_filter, peak_floor,
           caveat_file, style,
           alpha, qc_min_nuclei, sec_min_nuclei, engine_repo, preset,
           no_figures, no_coloc_panel, stamp):
    """Build the condition-versus-condition report for a finished run.

    Wells are the biological replicates and the condition GROUP is what gets
    compared. Every gate is a Welch t on well means with Hedges g, a Holm
    adjustment within its endpoint family, and the minimum detectable effect at
    that number of wells. Figures carry the star from the RAW p and print the
    adjusted p and the minimum detectable effect in the footnote.

    Writes REPORT.xlsx with plain sheet names, READOUT.md, figures/, per_well.csv,
    contrasts.csv, versions.txt and command.log. The run directory is read only.
    """
    from .report.build import build_report, load_groups_file
    from .report.aggregate import ReportInputError
    from .report.peak_gate import PeakGateError, parse_peak_floors

    if len(reason) != len(exclude_field):
        click.echo(
            f"--exclude-field was given {len(exclude_field)} time(s) but --reason "
            f"{len(reason)} time(s). Every excluded field needs its own reason, in "
            f"the same order.", err=True)
        sys.exit(2)
    excludes = dict(zip(exclude_field, reason))
    specs = list(groups)
    order = [g.strip() for g in group_order.split(",")] if group_order else []
    floors = {}
    caveat = ""
    try:
        if peak_floor:
            floors = parse_peak_floors(peak_floor)
        if caveat_file:
            caveat = Path(caveat_file).read_text(encoding="utf-8").strip()
        if groups_file:
            cfg = load_groups_file(Path(groups_file))
            # Command-line flags override the file, so a staged file is a default
            # rather than something that silently wins over what was just typed.
            specs = specs or cfg["specs"]
            order = order or cfg["group_order"]
            reference = reference or cfg["reference"]
            well_from_image = well_from_image or cfg["well_from_image"]
            excludes = excludes or cfg["exclude_fields"]
            if not floors and cfg.get("peak_floors"):
                floors = {str(k): float(v) for k, v in cfg["peak_floors"].items()}
            if nucleus_filter == "all" and cfg.get("nucleus_filter"):
                nucleus_filter = cfg["nucleus_filter"]
            if not caveat and cfg.get("caveat_file"):
                cav = Path(cfg["caveat_file"])
                if not cav.is_absolute():
                    cav = Path(groups_file).parent / cav
                if cav.is_file():
                    caveat = cav.read_text(encoding="utf-8").strip()
                else:
                    click.echo(f"caveat_file named in the groups file does not exist: "
                               f"{cav}", err=True)
            click.echo(f"groups file : {cfg['path']}")
        r = build_report(
            run_dir=Path(run_dir),
            out_dir=Path(out_dir) if out_dir else None,
            groups=specs,
            reference=reference,
            group_order=order,
            well_from_image=well_from_image,
            exclude_fields=excludes,
            peak_floors=floors,
            caveat=caveat,
            all_pairs=bool(all_pairs),
            nucleus_filter=nucleus_filter,
            qc_min_nuclei=qc_min_nuclei,
            sec_min_nuclei=sec_min_nuclei,
            alpha=alpha,
            engine_repo=Path(engine_repo) if engine_repo else None,
            preset=Path(preset) if preset else None,
            style=style,
            make_figures=not no_figures,
            coloc_panel=not no_coloc_panel,
            stamp=stamp,
        )
    except (ReportInputError, PeakGateError) as exc:
        click.echo(f"fishsuite report: {exc}", err=True)
        sys.exit(2)
    click.echo(f"report      : {r['out_dir']}")
    click.echo(f"workbook    : {r['xlsx']}")
    click.echo(f"readout     : {r['out_dir'] / 'READOUT.md'}")
    click.echo(f"figures     : {len(r['figures'])} figure(s) in {r['out_dir'] / 'figures'}")
    click.echo(f"groups      : {', '.join(r['group_order'])} "
               f"(reference {r['reference']})")
    gate = r.get("peak_gate") or {}
    if gate.get("floors"):
        for ch, rec in (gate.get("per_channel") or {}).items():
            if rec.get("applied"):
                click.echo(f"peak floor  : {ch} at {rec['floor']:g} -> "
                           f"{rec['spots_kept']}/{rec['spots_before']} spots kept")
    src = r.get("source_run") or {}
    if src:
        click.echo(f"source run  : {src.get('source_run_md')} "
                   f"[link: {(src.get('link') or {}).get('method')}]")
    if r["absent"]:
        click.echo(f"absent from this run (reported as NA): {', '.join(r['absent'])}")
    if r.get("coloc_panel"):
        click.echo(f"coloc panel : {r['coloc_panel']}")


@cli.command(short_help="CPU backfill: per-spot Gaussian size fit onto a finished run.")
@click.option("--run", "run_dir", required=True,
              type=click.Path(exists=True, file_okay=False),
              help="Finished run directory (must hold run_config.json, "
                   "per_image_summary.csv and spot_metrics.csv).")
@click.option("--input-dir", "input_dir", default=None,
              type=click.Path(exists=True, file_okay=False),
              help="Source image tree. Default: the run's recorded input_dir.")
@click.option("--window-px", type=int, default=None,
              help="Odd fit-window side in px. Default: the run's "
                   "foci.size_fit_window_px (7).")
@click.option("--limit-images", type=int, default=None,
              help="Fit only the first N images (smoke test).")
def sizefit(run_dir, input_dir, window_px, limit_images):
    """Fit a 2-D Gaussian to every detected spot and write size columns.

    Writes spot_metrics_sizefit.csv, sizefit_per_nucleus.csv and
    sizefit_per_image.csv beside the run's tables. The run's own tables are
    never modified.
    """
    from .core.sizefit_backfill import sizefit_run
    res = sizefit_run(run_dir, input_dir=input_dir, window_px=window_px,
                      limit_images=limit_images)
    click.echo(f"images fitted : {res['images_fitted']}")
    click.echo(f"spots fitted  : {res['spots']}")
    click.echo(f"size_fit_ok   : {100.0 * res['frac_size_fit_ok']:.2f}%")
    for s in res["skipped"][:10]:
        click.echo(f"  SKIP {s}")


if __name__ == "__main__":
    cli()
