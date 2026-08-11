"""Which module draws the locked SuperPlot, and how it is allowed to be found.

``analysis`` is a generic top-level package name. The resolver used to try a bare
``from analysis.single_condition_plots import _superplot_into_axes`` as its second
candidate, so on any machine with an unrelated importable ``analysis`` — an
installed package, or a stray directory on ``sys.path`` — a third party's function
would have been used to draw Brian's locked figures. Silently: the fallback
warning only fires when NOTHING resolves, and this candidate resolving IS
something.

That is the same hazard that got the hard-coded ``F:\\Image Analysis Work\\...``
fallback deleted from this module, cited in its own comment as the reason. The
implicit candidate was left behind. It is gone now: the vendored copy, then an
EXPLICIT ``$FISHSUITE_SUPERPLOT_PATH``, and the env path verifies that the module
it got actually came from where it was pointed.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from fishsuite.core import _superplot


def _write_fake_analysis(root: Path, marker: str) -> Path:
    """A minimal importable ``analysis`` package with a recognisable drawer."""
    pkg = root / "analysis"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "single_condition_plots.py").write_text(
        textwrap.dedent(
            f"""
            MARKER = {marker!r}

            def _superplot_into_axes(*a, **k):
                return {marker!r}
            """
        ),
        encoding="utf-8",
    )
    return pkg


_VENDORED = "fishsuite.core._vendor.analysis.single_condition_plots"


def _block_vendored(monkeypatch):
    """Make candidate (1) fail, so a later candidate is the one under test.

    ``None`` in ``sys.modules`` makes Python raise ImportError for that name.
    Patching ``__package__`` does NOT work: the relative import falls back to
    ``__spec__.parent`` and succeeds anyway (with only an ImportWarning), which
    would silently make these tests exercise candidate (1) instead.
    """
    monkeypatch.setitem(sys.modules, _VENDORED, None)


@pytest.fixture(autouse=True)
def _clean_analysis_modules():
    """Keep a fake ``analysis`` from leaking into other tests.

    ``sys.path`` is restored too: the resolver does its own ``sys.path.insert``,
    which monkeypatch cannot undo, so a tmp_path would otherwise stay on the
    import path after the directory is deleted.
    """
    mods = {k: v for k, v in sys.modules.items() if k.split(".")[0] == "analysis"}
    path = list(sys.path)
    yield
    for key in [k for k in sys.modules if k.split(".")[0] == "analysis"]:
        del sys.modules[key]
    sys.modules.update(mods)
    sys.path[:] = path


# ---------------------------------------------------------------------------
# THE NORMAL PATH
# ---------------------------------------------------------------------------
def test_the_vendored_copy_is_what_resolves():
    drawer = _superplot.get_locked_drawer()
    assert drawer is not None
    assert "_vendor" in Path(sys.modules[drawer.__module__].__file__).parts


def test_a_stray_importable_analysis_package_is_not_used(tmp_path, monkeypatch):
    """THE REGRESSION TEST. An unrelated `analysis` on sys.path must not be able
    to supply the drawer for the locked figures."""
    _write_fake_analysis(tmp_path, "IMPOSTOR")
    monkeypatch.syspath_prepend(str(tmp_path))
    # Prove the impostor really is importable, or this test proves nothing.
    import analysis.single_condition_plots as impostor

    assert impostor.MARKER == "IMPOSTOR"

    drawer = _superplot.get_locked_drawer()
    assert drawer is not None
    origin = Path(sys.modules[drawer.__module__].__file__)
    assert "_vendor" in origin.parts, f"drawer came from {origin}"
    assert getattr(sys.modules[drawer.__module__], "MARKER", None) != "IMPOSTOR"


def test_no_implicit_analysis_candidate_remains_in_the_source():
    """The candidate was a bare top-level import with nothing naming it, so it is
    easy to reintroduce while 'just restoring a fallback'."""
    import inspect

    src = inspect.getsource(_superplot.get_locked_drawer)
    body = src.split('"""', 2)[-1]          # ignore the docstring
    assert "from analysis" not in body, (
        "a bare top-level 'analysis' import is back; it binds to whatever "
        "package of that name happens to be importable"
    )
    # The only remaining reference must be the explicit, verified env-var branch.
    assert 'import_module("analysis.single_condition_plots")' in body
    assert "FISHSUITE_SUPERPLOT_PATH" in body


# ---------------------------------------------------------------------------
# THE EXPLICIT ENV OVERRIDE
# ---------------------------------------------------------------------------
def test_the_env_path_is_honoured_when_the_vendored_copy_is_gone(
    tmp_path, monkeypatch
):
    _write_fake_analysis(tmp_path, "EXPLICIT")
    monkeypatch.setenv("FISHSUITE_SUPERPLOT_PATH", str(tmp_path))
    _block_vendored(monkeypatch)
    drawer = _superplot.get_locked_drawer()
    assert drawer is not None and drawer() == "EXPLICIT"


def test_the_env_path_warns_when_something_else_shadows_it(tmp_path, monkeypatch):
    """Putting a directory on sys.path does not guarantee the import came FROM it.

    An ``analysis`` ALREADY IN ``sys.modules`` wins outright — ``import_module``
    returns the cached module and never consults ``sys.path`` at all. Without the
    origin check the user is told nothing and their locked figures are drawn by
    whatever got imported first.
    """
    requested = tmp_path / "requested"
    shadow = tmp_path / "shadow"
    _write_fake_analysis(requested, "REQUESTED")
    _write_fake_analysis(shadow, "SHADOW")

    # Import the shadow FIRST so it is cached. This is the real-world case: some
    # earlier import in the process already bound the name.
    monkeypatch.syspath_prepend(str(shadow))
    import analysis.single_condition_plots  # noqa: F401

    monkeypatch.setenv("FISHSUITE_SUPERPLOT_PATH", str(requested))
    _block_vendored(monkeypatch)

    with pytest.warns(RuntimeWarning, match="shadows it"):
        drawer = _superplot.get_locked_drawer()
    assert drawer() == "SHADOW", "the warning must name what actually happened"


def test_no_env_var_and_no_vendored_copy_warns_and_returns_none(monkeypatch):
    """None means the caller falls back to a drawer that produces a DIFFERENT
    figure, so it is warned about rather than taken silently."""
    monkeypatch.delenv("FISHSUITE_SUPERPLOT_PATH", raising=False)
    _block_vendored(monkeypatch)
    with pytest.warns(RuntimeWarning, match="does NOT reproduce the"):
        assert _superplot.get_locked_drawer() is None
