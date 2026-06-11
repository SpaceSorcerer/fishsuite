"""Click-based CLI for fishsuite."""
from __future__ import annotations

import os
# Force a non-interactive matplotlib backend BEFORE any module imports pyplot.
# Worker threads on Windows otherwise inherit TkAgg, which calls into Tcl from
# the wrong thread once Bio-Formats' JVM is alive → Tcl_AsyncDelete crash that
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
    """fishsuite — standalone RNA-FISH / IF analysis pipeline."""


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


if __name__ == "__main__":
    cli()
