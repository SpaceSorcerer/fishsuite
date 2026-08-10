# Vendored code — provenance

Files in this directory are copied from a separate repository, **verbatim except for
the single deviation recorded below**. Do not modify them here. To change
behaviour, wrap or subclass from `fishsuite/core/`.

Results previously published from this project were produced by importing these
same files from that external tree, so the checksum table below is what lets a
reader confirm the algorithms are unchanged by the move.

| Field | Value |
|---|---|
| Source repository | `https://github.com/SpaceSorcerer/image-analysis-pipeline.git` (private) |
| Source local path | `F:\Image Analysis Work\image-analysis-pipeline\` |
| Source ref | branch `feature/foci-spatial-metrics-and-batch-contrast` |
| Source HEAD SHA | `4d8c8a74074b3820d3861418e829f2a15ad1b780` |
| Source worktree | clean for every copied file (one unrelated untracked file elsewhere in the repo) |
| Date copied | 2026-08-09 |
| Method | selective file copy. **Not** `git subtree`, **not** `git submodule`. No git history was imported. |
| License | MIT, Copyright (c) 2024-2026 Brian Amburn, University of Texas Medical Branch (repo-root `LICENSE`) |

Git history was deliberately not imported: the source repository is private and
this repository is public, so imported history would become permanently public.

## Why this ref, and not `publication-ready`

The source repository also has an `origin/publication-ready` branch carrying two
commits absent from the vendored ref, one of which touches
`visualization/publication_figures.py`. It was checked rather than assumed:

- `f62621a` ("memory leak, CSV ordering, effect size, brackets") fixes a real
  figure-annotation defect — an omnibus Kruskal-Wallis p-value was drawn as a
  bracket spanning only the first two groups, which reads as a pairwise
  comparison. **That fix is already present in the vendored file** via the
  equivalent commit `81234da` on this lineage; verified directly in the blob
  (`is_omnibus` / `bracket_x2` / `x1, x2` parameters), not inferred from the
  commit message.
- The vendored ref is additionally **359 insertions ahead** of
  `publication-ready` for that file, including the Okabe-Ito colour policy and
  the 600 DPI default.
- The other commit, `77a7f8b`, changes only `README.md`, `CLAUDE.md` and
  `file_map.md`, none of which is vendored.

So the vendored ref is the newer and more complete source, and no cherry-pick
was required.

## SHA-256 of every vendored file, as copied

| Path in this directory | Source path | Bytes | sha256 |
|---|---|---|---|
| `segmentation/__init__.py` | `python/segmentation/__init__.py` | 1065 | `5b619744944c4db2bea1cbb96000d23f6ea0cd6262f3343f3f482c08e2e1f407` |
| `segmentation/segment_image.py` | `python/segmentation/segment_image.py` | 37237 | `e7a143ee0893992bd8506652dc0f7622c943e36442888ea405e8446930e37c22` |
| `spots/__init__.py` | `python/spots/__init__.py` | 1151 | `bc405b3b143beca89009394b15c58e622940de7f9e6cd47ce16832ab182fdbd3` |
| `spots/detect_spots.py` | `python/spots/detect_spots.py` | 29440 | `f20166a572b117b871d6bb9d86dd28a59c950bab7968bf27dd574001064e8f86` |
| `analysis/__init__.py` | `python/analysis/__init__.py` | 517 | `9fc916f32a82825ab586ec1a5ec1f3613a153c6d27ea5280aa2181f48e643287` |
| `analysis/single_condition_plots.py` | `python/analysis/single_condition_plots.py` | 504542 | `fd6a40aeb2dcf64d46156bd53fbfe561357bfd4a5a6197a55286a0f21d870fbf` |
| `visualization/__init__.py` | `python/visualization/__init__.py` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `visualization/publication_figures.py` | `python/visualization/publication_figures.py` | 23821 | `acb70e538ba43de974f53eef7329852244e9c15f044c4924f3f235fdb8faa5aa` |

`tests/test_vendor_parity.py::test_vendored_checksums_match_provenance` parses
this table and fails if any file here no longer matches it.

## The one deviation from verbatim

`analysis/single_condition_plots.py` — **one line changed**, at the import on
line 236:

```diff
-    from visualization.publication_figures import (
+    from ..visualization.publication_figures import (
```

| | |
|---|---|
| sha256 at source (`4d8c8a7`) | `7d3f994d119edb3ebd5a4426ad1c9540baeb2b69ae45da04a4cf15ffc9565415` (504540 bytes) |
| sha256 as vendored | `fd6a40aeb2dcf64d46156bd53fbfe561357bfd4a5a6197a55286a0f21d870fbf` (504542 bytes) |

Why it was necessary: unlike its siblings, this file performs no
`sys.path` manipulation of its own — it relied on being executed with the
source `python/` directory as the working directory. The absolute
`from visualization...` import therefore cannot resolve once the file lives
inside a package.

Why it matters that it was fixed rather than left to fail: the import sits
inside a `try/except`, and the fallback sets `CONDITION_COLORS = {}`. Left
broken it would not have raised — it would have silently dropped the locked
per-condition colour mapping and drawn figures with a different palette.

Every other vendored file is byte-identical to its source, verified by the table
above.

## Deliberately not vendored

Five sibling files were present in the source directories and were **not**
copied: `segmentation/batch_segment.py`, `segmentation/compare_backends.py`,
`segmentation/tune_segmentation.py`, `spots/batch_detect_spots.py`,
`spots/compare_backends.py`.

They are standalone batch/CLI drivers that this package never calls, and each
carries an absolute intra-repository import plus a `sys.path.insert` of its own
parent directory. Copying them would have put a directory containing top-level
`segmentation`, `spots`, `analysis` and `visualization` packages onto the global
`sys.path` of any process importing fishsuite — a name-shadowing hazard for
users who have unrelated packages by those names installed. Omitting them
removes four of the five absolute-import sites outright.
`tune_segmentation.py` additionally emits a Jython script and shells out to
Fiji, which does not belong in a package documented as Fiji-free.
