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
  fishsuite backfill --run "F:\\Image Analysis Work\\MIAT-QKI-Coloc\\my_run_20260605"

  # point at the VSI staging folder explicitly:
  fishsuite backfill --run "F:\\...\\my_run" --staging "E:\\Claude\\fishsuite\\_staging_UD_ALLARMS"

  # montage only (skip the CSV products):
  fishsuite backfill --run "F:\\...\\my_run" --no-null-draws --no-radial
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
def backfill(run_dir, staging, input_dir, seed, no_null_draws, no_radial, no_montage):
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
  fishsuite walkthrough --run "F:\\Image Analysis Work\\MIAT-QKI-Coloc\\my_run"
  fishsuite walkthrough --run "F:\\...\\my_run" --image "g2_wDox_(MIAT_OE)__g2-Dox_01"
  fishsuite walkthrough --run "F:\\...\\my_run" --out "F:\\figures\\walkthrough.png"
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
  fishsuite postrun --run "F:\\Image Analysis Work\\MIAT-QKI-Coloc\\my_run_20260605"
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


if __name__ == "__main__":
    cli()
