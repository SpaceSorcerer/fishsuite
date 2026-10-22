# Contributing to fishsuite

fishsuite produces numbers that end up in figures and manuscripts. A change that silently moves a measured value costs far more than one that breaks a build, so most of the rules below protect computed values.

---

## Development install

```bash
git clone <repo-url> fishsuite
cd fishsuite
python -m pip install -e ".[dev]"
```

The supported Python range is `requires-python` in `pyproject.toml`. Its upper bound follows the numpy pin that StarDist and TensorFlow need; there are no numpy 1.x wheels for newer CPython.

Extras:

| Extra | Installs | When you need it |
|---|---|---|
| `dev` | pytest, hypothesis, ruff, build, twine | always, for development |
| `bioformats` | bioio-bioformats | to read `.vsi`, `.czi`, `.lif`, `.nd2`; needs a JDK |
| `gui` | PySide6 | `fishsuite gui` |
| `deck` | python-pptx | `fishsuite report --deck` |
| `directml` | torch-directml (Windows only) | AMD GPU Cellpose |
| `all` | `bioformats` + `gui` | a normal workstation |

For an NVIDIA GPU, install a CUDA build of torch from pytorch.org and set `nuclei.cellpose_device: cuda`. No extra is needed.

Two dependency rules:

- **Keep `bioio-bioformats` out of the base dependencies.** Its `bffile` dependency declares a numpy floor that conflicts with the numpy pin, which makes the base install unresolvable. CI checks resolution with `pip install --dry-run` for the base set, `[bioformats]` and `[all]`.
- **Do not add `bioio-tifffile` or another third-party bioio reader.** It reads the channel axis of ambiguous multi-plane TIFFs as Z or T, so channel indices point at the wrong planes and segmentation silently finds no nuclei.

---

## Running the tests

```bash
python -m pytest -q                  # full suite; needs StarDist/TensorFlow and cellpose/torch
python -m pytest -q -m "not heavy"   # what CI runs
```

### Markers

Declared in `pyproject.toml` under `[tool.pytest.ini_options]`:

| Marker | Meaning | Skipped or deselected when |
|---|---|---|
| `lab` | Needs private image data not in the repository. | The data is absent. Never satisfiable on CI or an external clone. |
| `gpu` | Needs a real GPU (DirectML or CUDA). | No GPU present. |
| `heavy` | Slow; needs the full ML stack. | Deselected by `-m "not heavy"` in CI. |
| `bioformats` | Needs a bioio reader that resolves channels correctly. | The reader probe finds none. |

When you add a marker, run `pytest --collect-only` with and without it and confirm the count changes. A test that needs something the machine lacks must skip, not fail: give it the marker and an `importorskip` or `skipif` guard.

The `test_nucleus_sampling.py::test_end_to_end_*` tests are guarded by a probe that reads a file in a subprocess. Keep it that way: a mis-inferring reader is present but wrong, and a JVM start on Windows can crash the process.

---

## Linting and formatting

```bash
ruff check .            # blocking in CI
ruff format --check .   # advisory
```

`ruff check` uses a narrow rule set in `pyproject.toml` (`[tool.ruff.lint]`) that flags code that is wrong, not code that is unfashionable. It passes clean, so any new finding is yours.

Two categories are excluded on purpose, with reasons inline in `pyproject.toml`: cosmetic pyflakes rules (`F401`, `F841`, `F541`), which would mean deleting imports and locals from numerical code, and `E712`, whose advice (`if series:`) is a bug on a pandas Series.

`ruff format` would restyle many files. If you do it, do it as its own commit that touches nothing else.

---

## Never edit `src/fishsuite/core/_vendor/`

The segmentation and spot-detection routines are vendored verbatim so fishsuite is self-contained and citable. Every vendored file's SHA-256 is in `src/fishsuite/core/_vendor/PROVENANCE.md` and checked by `tests/test_vendor_parity.py::test_vendored_checksums_match_provenance`.

To change vendored behaviour, wrap it one level up in `core/segmentation.py` or `core/spots.py`. `_install_cuda_cellpose_route` in `core/segmentation.py` is an example: CUDA support is added from the wrapper, not edited into the vendored code.

To re-vendor from upstream, update the source commit and the checksums in `PROVENANCE.md` in the same change, and record the parity evidence in `docs/dev/VENDOR_PARITY.md`.

---

## A change that alters a computed number is a MAJOR version bump

If a change moves any value that lands in a results file (spot counts, intensities, colocalization coefficients, nuclear fractions, null p-values, thresholds, segmentation labels):

1. It is a **major** version bump. Two versions that disagree numerically must not share a major version.
2. It needs a **parity run**: run the affected mode on a dataset analysed with the previous version and record in `docs/dev/VENDOR_PARITY.md` what changed, by how much, and why the new value is correct. Passing tests is not a parity run.
3. The `CHANGELOG.md` entry must say that numbers moved.

Additive changes are welcome: a new opt-in metric, a new figure, a config field that defaults to off. A change that leaves every existing config's output byte-identical is a minor bump. Keep new feature flags off by default.

---

## Configuration and presets

- The config is a Pydantic v2 model in `src/fishsuite/config/schema.py`. New fields need defaults that reproduce current behaviour.
- `src/fishsuite/config/presets/` holds only portable presets that ship in the wheel. Run records (the exact settings behind a past result) live in the private records repository; copy one under a new name rather than retuning it.
- Only presets marked `# portable: true` ship in the wheel. To add one, mark it, add it to `[tool.hatch.build.targets.wheel.force-include]` in `pyproject.toml`, and add it to the expected set in the CI wheel-contents gate.
- Portable is a judgement, not a grep. A preset with no absolute paths can still be tied to one dataset's folder names or channels. Read it before promoting it.
- Never commit an absolute path as an active config value. Use `<SET_ME>` as the placeholder.

---

## Documentation

User documentation is `README.md`, the guides in `docs/`, `POSTRUN_UTILITIES.md` and `THRESHOLD_INTENSITY_FEATURE.md`. The vendored-code parity record is `docs/dev/VENDOR_PARITY.md`; other internal records (change notes, audits, runbooks) go in the private records repository. Do not put test counts, dependency versions or dates in user documentation; point to `pyproject.toml` or the CI badge instead. Add changes to `CHANGELOG.md`.

---

## Continuous integration

`.github/workflows/test.yml` runs on Ubuntu and Windows across the supported Python range. Blocking gates: the light test suite (`-m "not heavy"`), `ruff check`, `python -m build` plus `twine check`, a clean-venv install of the built wheel followed by `import fishsuite` and `fishsuite --help`, `scripts/check_no_absolute_paths.py`, a wheel-contents gate for the portable presets, and a `deps-resolve` job running `pip install --dry-run` on the base set, `[bioformats]` and `[all]`.

A weekly non-blocking job installs the full stack and runs the heavy tests to catch upstream dependency drift.
