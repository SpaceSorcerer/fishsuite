# Contributing to fishsuite

fishsuite quantifies microscopy images and the numbers it produces end up in
figures and manuscripts. That single fact drives most of the rules below: a change
that silently moves a measured value is far more expensive than a change that
breaks a build, because the build failure is visible and the moved number is not.

---

## Development install

```bash
git clone <repo-url> fishsuite
cd fishsuite
python -m pip install -e ".[dev]"
```

`requires-python` is `>=3.10,<3.13`. The upper bound is not arbitrary — it comes
from `numpy<2.0`, which is a TensorFlow/StarDist transitive constraint. There are
no numpy 1.x wheels for CPython 3.13, so installing there tries to build numpy
from source and fails. See the comment on `requires-python` in `pyproject.toml`.

Optional extras:

| Extra | Installs | When you need it |
|---|---|---|
| `dev` | pytest, hypothesis, ruff, build, twine | always, for development |
| `gui` | PySide6 | to run `fishsuite gui` |
| `directml` | torch-directml (Windows only) | AMD GPU cellpose acceleration |

For an NVIDIA GPU install a CUDA build of torch from pytorch.org and set
`nuclei.cellpose_device: cuda`. No extra is needed — CUDA support rides on the
torch build, not on a fishsuite dependency.

---

## Running the tests

```bash
python -m pytest -q
```

The full suite needs the heavy ML stack (StarDist/TensorFlow and cellpose/torch).
To run only what a machine without GPUs and without the lab's image tree can do —
which is what CI runs:

```bash
python -m pytest -q -k "not stardist and not cellpose"
```

### Markers

Declared in `pyproject.toml` under `[tool.pytest.ini_options]`:

| Marker | Meaning | Skipped when |
|---|---|---|
| `lab` | Needs the original `image-analysis-pipeline` tree on the author's machine. | That tree is absent. Never satisfiable on CI or on an external clone. |
| `gpu` | Needs a real GPU (DirectML or CUDA). | No GPU present. |
| `heavy` | Slow; needs the full ML stack (StarDist/TensorFlow, cellpose/torch). | Those packages are not installed. |
| `bioformats` | Needs `bioio-bioformats` and a working JVM. | Either is missing. |

A test that needs something the machine does not have must **skip**, not fail.
If you add a test with an external requirement, give it the right marker and a
`pytest.importorskip` / `skipif` guard so a light environment stays green.

---

## Linting and formatting

```bash
ruff check .            # blocking in CI
ruff format --check .   # advisory today; see below
```

`ruff check` is configured in `pyproject.toml` under `[tool.ruff.lint]` to a
deliberately narrow rule set: every selected rule flags code that is *wrong* —
an undefined name, a redefinition that discards an earlier definition, a format
string that cannot render — rather than code that is merely unfashionable. It
passes clean, so a new finding means you introduced one. Please keep it that way.

Two categories are knowingly excluded, both documented inline in `pyproject.toml`:

- **Cosmetic pyflakes** (`F401` unused import, `F841` unused variable, `F541`
  f-string with no placeholder) — 72 findings as of 2026-08-10. Clearing these
  means deleting imports and locals from numerical code, so it wants a dedicated
  change with a full test run behind it.
- **`E712` (`== True`)** — actively wrong advice in this codebase. On a pandas
  Series, `series == True` is the idiomatic elementwise mask and `if series:`
  raises `ValueError`. Following the rule would introduce a real bug.

`ruff format` would currently rewrite 72 files. That reformat has not been done,
because a whole-repo restyle in the same commits as substantive changes makes the
substantive part unreviewable. If you take it on, do it as its own commit,
touching nothing else.

---

## Never edit `src/fishsuite/core/_vendor/`

The segmentation and spot-detection routines are the lab's own implementations,
copied **verbatim** into the package so that fishsuite is self-contained and
citable. Every vendored file's SHA-256 is recorded in
`src/fishsuite/core/_vendor/PROVENANCE.md` and enforced by

```
tests/test_vendor_parity.py::test_vendored_checksums_match_provenance
```

Editing a vendored file breaks that test and destroys the provenance record that
lets a reader confirm the shipped code is the code that produced the published
numbers.

To change vendored behaviour, **wrap it** in the corresponding module one level
up — `core/segmentation.py` and `core/spots.py` exist for exactly this. A worked
example is `_install_cuda_cellpose_route` in `core/segmentation.py`: the vendored
cellpose constructor knows only `cpu` and `directml`, so the CUDA branch is
installed from the wrapper instead of being edited in.

If you genuinely must re-vendor from upstream, update the source commit **and**
the checksums in `PROVENANCE.md` in the same change, and record the parity
evidence in `docs/dev/VENDOR_PARITY.md`.

---

## Any change that alters a computed number is a MAJOR version bump

This is the rule that matters most.

If a change moves any value that lands in a results file — spot counts, per-nucleus
intensities, colocalization coefficients, nuclear fractions, null-distribution
p-values, thresholds, segmentation labels — then:

1. It is a **MAJOR** version bump, not a minor and not a patch. Published figures
   are traceable to a version; two versions that disagree numerically must not
   share a major.
2. It requires a **parity run**: execute the affected mode on a dataset that was
   already analysed with the previous version and record, in
   `docs/dev/VENDOR_PARITY.md`, what changed, by how much, and why the new value
   is the correct one. "The tests still pass" is not a parity run — the tests
   check invariants and shapes, not that a coefficient is unchanged to 6 decimals.
3. The changelog entry must say plainly that numbers moved, so anyone re-running
   an old analysis knows not to mix outputs across the boundary.

Additive changes are cheap and welcome: a new opt-in metric, a new figure, a new
config field that defaults to off. Anything guarded so that an existing config
produces byte-identical output is a MINOR bump. Much of this codebase is built
that way on purpose — most feature flags default to `false` precisely so that
adding them cannot disturb an existing run. Preserve that property.

---

## Configuration and presets

- Config is a Pydantic v2 model in `src/fishsuite/config/schema.py`. Add fields
  with defaults that reproduce current behaviour.
- Presets in `src/fishsuite/config/presets/` are mostly **run records** — the
  provenance of specific published figures. Do not retune one to fix an unrelated
  problem; clone it under a new dated name.
- Only presets marked `# portable: true` are shipped in the built wheel and are
  meant as starting points. Anything carrying `input_file_subset`,
  `z_stack.file_overrides`, or a machine-specific absolute path stays in the
  repository but is excluded from the wheel (see
  `[tool.hatch.build.targets.wheel] exclude` in `pyproject.toml`). If you add a
  portable preset, mark it and add it to the include side of that list.
- Never commit an absolute path as an active config value. `<SET_ME>` is the
  placeholder convention.

---

## Continuous integration

`.github/workflows/test.yml` runs on Ubuntu and Windows across Python 3.10 and
3.12. Windows is not optional: it is the platform this package is developed on and
where the absolute-path defects lived.

The blocking gates are: the light test suite, `ruff check`, `python -m build` plus
`twine check`, a clean-venv install of the built wheel followed by
`import fishsuite` and `fishsuite --help`, a grep gate that fails on absolute
`F:\` / `C:\Users` paths in `src/`, and a wheel-contents gate asserting no
non-portable presets shipped.

A separate weekly job installs the full stack and runs the heavy tests. It is
non-blocking by design — its purpose is to surface upstream dependency drift, not
to gate a pull request on someone else's release.
