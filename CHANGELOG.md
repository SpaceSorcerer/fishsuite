# Changelog

## 2026-09-08

Document report and native-condition features added September 7–8 (history `19495c6..fca3a12`):

- Replicate-simple well means and mean ± SD columns; full/focus editable figures and consistent arm axes.
- Mixed-model headline statistics with retained Welch, Student and FOV-outlier sensitivity results.
- DAPI localization classes, corrected fractions, object census and cytoplasmic-call QC.
- Frozen coloc import, simple Pearson/ICQ/Costes-only Manders figures and cytofluorograms.
- MIAT/QKI and RNASEH2B preferential ratios, covariance CI, exact enumeration and R-MDE.
- Traced deck specifications, speaker notes, slide_sources.csv and independent per-well micrograph panels.
- Native figures collapsed through conditions.groups, per-well supplementary figures and CSV-only native-figures regeneration.
- Release-marker output guards and distinct reporter versus producing commit identities.

Documentation-only entry; no analysis code or source measurements changed. See [the feature reference](docs/REPORT_FEATURES_2026-09-08.md) for exact CLI/YAML entry points and current limitations (including no automatic deck_figures directory).
