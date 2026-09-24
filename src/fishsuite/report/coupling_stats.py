"""Descriptive, well-weighted single-plane MIAT/QKI coupling summaries."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.qki_association import COLUMN_DEFINITIONS, DESCRIPTIVE_ONLY, DIFFERENCE_SCALE_NEW

METRICS = [
    'n_miat_spots', 'integrated_nuclear_miat', 'nuclear_area_um2',
    'mean_nuclear_qki', 'n_miat_spots_qki_pos', 'frac_miat_spots_qki_pos',
    'obs_minus_null_frac_miat_spots_qki_pos', 'frac_miat_footprint_area_qki_pos',
    'obs_minus_null_frac_miat_footprint_area_qki_pos', 'mean_qki_at_miat_spots',
    'qki_at_spots_minus_nuclear', 'sat_frac_miat', 'sat_frac_qki',
]
ABUNDANCE = ['n_miat_spots', 'integrated_nuclear_miat']
ASSOCIATION = ['obs_minus_null_frac_miat_spots_qki_pos', 'qki_at_spots_minus_nuclear']
DIFFERENCE_SCALE_METRICS = {
    metric for metric in COLUMN_DEFINITIONS if metric.startswith('obs_minus_null_')
} | {'qki_at_spots_minus_nuclear'} | set(DIFFERENCE_SCALE_NEW)
PERMUTATION_NOTE = '2 v 2 wells: exact permutation has 6 allocations; smallest two-sided p = 0.333'
NON_METRIC_COLUMNS = {
    'condition', 'well', 'image', 'nucleus_id', 'measurement_plane',
    'threshold_multiplier', 'well_from_image', 'n_null_effective', 'biological_set', 'slide',
}


def _metrics(data):
    """Every numeric per-nucleus measurement; identifiers and bookkeeping are not endpoints."""
    discovered = [c for c in data if c not in METRICS and c not in NON_METRIC_COLUMNS
                  and c not in DESCRIPTIVE_ONLY
                  and not c.endswith(('_id', '_min_used', '_na_reason'))
                  and not c.startswith('na_reason_')
                  and pd.api.types.is_numeric_dtype(data[c])
                  and not pd.api.types.is_bool_dtype(data[c])]
    return METRICS + discovered


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    required = ['condition', 'well', 'image', 'nucleus_id', 'threshold_multiplier'] + METRICS
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError('Missing association columns: ' + ', '.join(missing))
    if df.empty:
        raise ValueError('Association table is empty')
    data = df.copy()
    for column in ['condition', 'image']:
        if data[column].isna().any() or data[column].astype(str).str.strip().eq('').any():
            raise ValueError(f'Missing {column} identifier')
        data[column] = data[column].astype(str)
    blank = data.well.isna() | data.well.astype(str).str.strip().eq('')
    previous = data.get('well_from_image', pd.Series(False, index=data.index))
    data['well_from_image'] = previous.astype(bool) | blank
    data['well'] = data.well.astype(str)
    data.loc[blank, 'well'] = data.loc[blank, 'image']
    # Astra F9 (2026-09-24): composite or validated globally unique well key.
    from ..core.well_key import resolve_well_ids
    data['well'], data.attrs['well_key_source'] = resolve_well_ids(data)
    metrics = _metrics(data)
    for column in ['threshold_multiplier', 'nucleus_id'] + metrics:
        data[column] = pd.to_numeric(data[column], errors='raise').astype(float)
        if np.isinf(data[column]).any():
            raise ValueError(f'Infinite values in {column}')
    for column in ['threshold_multiplier', 'nucleus_id']:
        if data[column].isna().any() or data[column].le(0).any():
            raise ValueError(f'{column} must be finite and positive')
    if data.nucleus_id.mod(1).ne(0).any():
        raise ValueError('nucleus_id must contain integer labels')
    if data.duplicated(['threshold_multiplier', 'condition', 'image', 'nucleus_id']).any():
        raise ValueError('duplicate nucleus identity at a threshold multiplier')
    if 'measurement_plane' in data:
        if not data.measurement_plane.eq('single-plane').all():
            raise ValueError('All measurements must be single-plane')
    return data


def _per_well(data, metrics):
    rows = []
    reasons = [c for c in data if c.startswith('na_reason_')]
    for (condition, well, level), group in data.groupby(
            ['condition', 'well', 'threshold_multiplier'], sort=True):
        row = dict(condition=condition, well=well, threshold_multiplier=level,
                   n_nuclei=len(group), n_nuclei_with_spots=int(group.n_miat_spots.gt(0).sum()),
                   n_images=int(group.image.nunique()), well_from_image=bool(group.well_from_image.any()))
        for metric in metrics:
            finite = group[metric].dropna()
            row[f'n_finite_{metric}'] = len(finite)
            row[f'n_excluded_{metric}'] = len(group) - len(finite)
            row[f'mean_{metric}'] = float(finite.mean()) if len(finite) else np.nan
            row[f'median_{metric}'] = float(finite.median()) if len(finite) else np.nan
        for reason in reasons:
            counts = group[reason].fillna('').astype(str).value_counts()
            row[reason] = '; '.join(f'{key}: {value}' for key, value in sorted(counts.items()) if key)
        row.update(_spot_pooled(group))
        rows.append(row)
    result = pd.DataFrame(rows)
    roster = data[['condition', 'well']].drop_duplicates()
    expected = pd.MultiIndex.from_tuples(
        [(r.condition, r.well, level) for r in roster.itertuples()
         for level in sorted(data.threshold_multiplier.unique())],
        names=['condition', 'well', 'threshold_multiplier'])
    result = result.set_index(['condition', 'well', 'threshold_multiplier']).reindex(expected)
    absent = result.n_nuclei.isna()
    counts = ['n_nuclei', 'n_nuclei_with_spots', 'n_images'] + [
        f'{prefix}_{metric}' for metric in metrics for prefix in ('n_finite', 'n_excluded')]
    for column in counts:
        result[column] = result[column].fillna(0).astype(int)
    result['well_from_image'] = result.well_from_image.eq(True)
    for reason in reasons:
        result[reason] = result[reason].fillna('')
    result = result.copy()  # Per-column assignment over every metric fragments the frame.
    result['na_reason_threshold'] = np.where(absent, 'missing well at threshold multiplier', '')
    return result.reset_index().sort_values(['condition', 'well', 'threshold_multiplier']).reset_index(drop=True)


POOLED_UPP = ('mean_uniform_position_percentile_qki', 'frac_spots_upp_ge_0p90', 'frac_spots_upp_ge_0p75')


def _spot_pooled(group):
    """Astra F4 sensitivity: every scored spot of the well pooled (a 20-spot
    nucleus weighs 20x a 1-spot nucleus), from per-nucleus value x n_spots_upp."""
    if 'n_spots_upp' not in group:
        return {}
    weights = pd.to_numeric(group['n_spots_upp'], errors='coerce').fillna(0)
    out = {'n_spots_upp_pooled': int(weights.sum())}
    for column in POOLED_UPP:
        if column not in group:
            continue
        values = pd.to_numeric(group[column], errors='coerce')
        keep = values.notna() & weights.gt(0)
        total = float(weights[keep].sum())
        out[f'spot_pooled_{column}'] = float((values[keep] * weights[keep]).sum() / total) if total else np.nan
    return out


def _descriptive_qc(data):
    """Per condition/well/threshold distribution of descriptive-only columns:
    Costes p (with the 1/(1+n_draws) floor fraction), block parameters and
    chance references. Never contrasted, averaged into arms, or ratioed."""
    present = [c for c in sorted(DESCRIPTIVE_ONLY) if c in data]
    rows = []
    for (condition, well, level), group in data.groupby(['condition', 'well', 'threshold_multiplier'], sort=True):
        row = dict(condition=condition, well=well, threshold_multiplier=level, n_nuclei=len(group))
        for column in present:
            values = pd.to_numeric(group[column], errors='coerce').dropna()
            row[f'n_finite_{column}'] = len(values)
            row[f'median_{column}'] = float(values.median()) if len(values) else np.nan
            if column == 'costes_rand_p':
                for name, q in (('min', 0), ('q25', .25), ('q75', .75), ('max', 1)):
                    row[f'{name}_{column}'] = float(values.quantile(q)) if len(values) else np.nan
                draws = pd.to_numeric(group.get('costes_rand_n_draws', pd.Series(np.nan, index=group.index)),
                                      errors='coerce').loc[values.index]
                floor = 1.0 / (1.0 + draws)
                row['frac_costes_rand_p_at_floor'] = float((values <= floor + 1e-12).mean()) if len(values) else np.nan
        if 'na_reason_costes_rand' in group:
            counts = group['na_reason_costes_rand'].fillna('').astype(str).value_counts()
            row['costes_na_reasons'] = '; '.join(f'{k}: {v}' for k, v in sorted(counts.items()) if k)
            row['frac_costes_na'] = float(group['na_reason_costes_rand'].fillna('').astype(str).ne('').mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _per_arm(per_well, metrics):
    """Every arm at every multiplier: equal-weight mean of its well means."""
    rows = []
    for (condition, level), group in per_well.groupby(['condition', 'threshold_multiplier'], sort=True):
        present = group[group.n_nuclei.gt(0)]
        row = dict(condition=condition, threshold_multiplier=level, n_wells=len(present),
                   n_nuclei=int(present.n_nuclei.sum()), wells='; '.join(sorted(present.well.astype(str))))
        for metric in metrics:
            values = present[f'mean_{metric}']
            finite = values.dropna()
            row[f'n_wells_finite_{metric}'] = len(finite)
            row[f'mean_of_well_means_{metric}'] = _arm_mean(values) if len(present) else np.nan
        for column in [c for c in present if c.startswith('spot_pooled_')]:
            row[f'mean_of_well_{column}'] = _arm_mean(present[column]) if len(present) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _ratio(numerator, denominator, label='denominator'):
    if not np.isfinite(denominator):
        return np.nan, f'missing {label}'
    if denominator == 0:
        return np.nan, f'zero {label}; no pseudocount'
    if not np.isfinite(numerator):
        return np.nan, 'missing numerator'
    return float(numerator / denominator), ''


def _arm_mean(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()) if len(values) and np.isfinite(values).all() else np.nan


def _contrast(per_well, metrics, treated, control, level):
    rows = []
    level_data = per_well[per_well.threshold_multiplier.eq(level)]
    difference_scale_metrics = DIFFERENCE_SCALE_METRICS | {
        metric for metric in metrics if metric.startswith('obs_minus_null_')}
    for metric in metrics:
        row = dict(metric=metric, threshold_multiplier=level, treated=treated, control=control)
        for name, condition in [('treated', treated), ('control', control)]:
            wells = level_data[level_data.condition.eq(condition)].sort_values('well')
            row[f'{name}_n_wells'] = len(wells)
            row[f'{name}_mean_of_well_means'] = _arm_mean(wells[f'mean_{metric}'])
            # Metric names are arbitrary column labels; itertuples would mangle them.
            for i, (_, well) in enumerate(wells.iterrows(), 1):
                row[f'{name}_well_{i}'] = well['well']
                row[f'{name}_well_{i}_mean'] = well[f'mean_{metric}']
        treated_mean, control_mean = row['treated_mean_of_well_means'], row['control_mean_of_well_means']
        row['difference'] = treated_mean - control_mean
        row['na_reason'] = '' if np.isfinite(row['difference']) else 'missing well or finite well mean in a contrast arm'
        if metric in difference_scale_metrics:
            row['ratio'], row['ratio_na_reason'] = np.nan, 'DIFFERENCE_SCALE'
        elif np.isfinite(control_mean) and (control_mean <= 0 or abs(control_mean) < 1e-12):
            row['ratio'], row['ratio_na_reason'] = np.nan, 'CONTROL_NONPOSITIVE'
        else:
            row['ratio'], row['ratio_na_reason'] = _ratio(treated_mean, control_mean, 'control arm mean')
        rows.append(row)
    return pd.DataFrame(rows)


def _finite_column_means(values):
    valid = np.isfinite(values)
    counts = valid.sum(axis=0)
    return np.divide(np.where(valid, values, 0).sum(axis=0), counts,
                     out=np.full(values.shape[1], np.nan), where=counts > 0)


def _bootstrap(data, metrics, treated, control, seed, n_boot):
    rng = np.random.default_rng(seed)
    arms = [[group[metrics].to_numpy(dtype=float) for _, group in
             data[data.condition.eq(arm)].groupby('well', sort=True)] for arm in (treated, control)]
    draws = np.full((n_boot, len(metrics)), np.nan)
    for iteration in range(n_boot):
        arm_means = []
        for wells in arms:
            selected = rng.integers(len(wells), size=len(wells))
            means = []
            for index in selected:
                values = wells[index]
                sampled = values[rng.integers(len(values), size=len(values))]
                means.append(_finite_column_means(sampled))
            arm_means.append(np.mean(means, axis=0))
        draws[iteration] = arm_means[0] - arm_means[1]
    return draws


def _correlations(data):
    rows = []
    for (condition, well, level), group in data.groupby(['condition', 'well', 'threshold_multiplier'], sort=True):
        eligible = group[group.n_miat_spots.gt(0)]
        for abundance in ABUNDANCE:
            for association in ASSOCIATION:
                pairs = eligible[[abundance, association]].dropna()
                reason = ''
                if len(pairs) < 10:
                    reason = 'fewer than 10 finite paired nuclei with N>0'
                elif pairs.nunique().min() < 5:
                    reason = 'fewer than 5 distinct values in at least one variable'
                pearson = np.nan if reason else float(pairs.corr(method='pearson').iloc[0, 1])
                spearman = np.nan if reason else float(pairs.rank(method='average').corr(method='pearson').iloc[0, 1])
                rows.append(dict(condition=condition, well=well, threshold_multiplier=level,
                                 abundance_metric=abundance, association_metric=association,
                                 n_pairs=len(pairs), pearson_r=pearson, spearman_r=spearman,
                                 na_reason=reason))
    return pd.DataFrame(rows)


def _ratio_of_ratios(contrast):
    indexed = contrast.set_index('metric')
    count = indexed.loc['n_miat_spots_qki_pos']
    rows = []
    for name, abundance in [('ror_count', 'n_miat_spots'), ('ror_intensity', 'integrated_nuclear_miat')]:
        amount = indexed.loc[abundance]
        reason = count.ratio_na_reason or amount.ratio_na_reason
        value, outer_reason = _ratio(count.ratio, amount.ratio, f'{abundance} treated/control ratio')
        reason = reason or outer_reason
        rows.append(dict(metric=name, value=np.nan if reason else value, na_reason=reason,
                         definition=f'SECONDARY descriptive: (mean C treated / control) / (mean {abundance} treated / control); arm means of well means',
                         count_treated_control_ratio=count.ratio,
                         abundance_treated_control_ratio=amount.ratio))
    corrected = indexed.loc['obs_minus_null_frac_miat_spots_qki_pos']
    rows.append(dict(metric='ror_corrected', value=corrected.difference, na_reason=corrected.na_reason,
                     definition='SECONDARY descriptive treated minus control difference; ratio undefined for difference-scale statistic',
                     count_treated_control_ratio=np.nan, abundance_treated_control_ratio=np.nan))
    return pd.DataFrame(rows)


def _readme(data, per_well, metrics, treated, control, seed, n_boot):
    levels = ', '.join(f'{level:g}' for level in sorted(data.threshold_multiplier.unique()))
    fallback_count = int(data.well_from_image.sum())
    entries = [
        ('measurement', 'All metrics are single-plane measurements; no 3D or MIP interpretation.'),
        ('replicate_structure', 'Design: two wells per arm, one field per well. Wells are replicates; nuclei are not independent.'),
        ('contrast', f'Treated={treated}; control={control}; threshold_multiplier=1.0. Each well has equal weight, regardless of nucleus count.'),
        ('threshold_multipliers', levels),
        ('filters', 'Finite values per metric; N=0 retained for abundance and defined metrics. Undefined fractions at N=0 are excluded as NaN. No saturation exclusion. Missing values are not imputed; n_finite_<metric> and n_excluded_<metric> show retained and excluded nucleus counts.'),
        ('missing_well_mean', 'The observed condition/well roster is retained at every multiplier, including absent wells as zero-nucleus rows. If any well lacks a finite metric mean, its arm mean and contrast are missing; the well is never silently discarded.'),
        ('well_fallback', f'Image fallback used for {fallback_count} input rows with empty well; image is the replicate unit for these rows.'),
        ('bootstrap', f'Exploratory only, requiring at least 3 observed wells per arm; seed={seed}; n_boot={n_boot}. Resample wells within each arm, then nuclei within each sampled well, with replacement. 2.5 and 97.5 percentiles of treated minus control differences; not confirmatory confidence intervals.'),
        ('bootstrap_minimum_wells', 'with fewer than 3 wells per arm no interval is reported; read the individual well means'),
        ('bootstrap_missing', 'Intervals exclude nonfinite bootstrap differences; valid draw counts are shown. No interval when the original contrast is missing.'),
        ('permutation_design_note', PERMUTATION_NOTE),
        ('inference', 'No nucleus-level p-values, pooled correlations, or intervals on differences in r are calculated.'),
        ('ror_count', 'SECONDARY descriptive: (C treated/control)/(N treated/control), using arm means of well means of per-nucleus values. For pooled counts the identity is the ratio of QKI-positive fractions; this is not generally the ratio of means of per-nucleus fractions.'),
        ('ror_intensity', 'SECONDARY descriptive: (C treated/control)/(integrated_nuclear_miat treated/control), using arm means of well means.'),
        ('ror_corrected', 'A ratio is undefined for a difference-scale statistic; report treated minus control difference in obs_minus_null_frac_miat_spots_qki_pos instead.'),
        ('ratio_reasons', 'DIFFERENCE_SCALE: ratios are undefined for all obs_minus_null_* metrics and qki_at_spots_minus_nuclear; report differences only. CONTROL_NONPOSITIVE: the control arm mean for a ratio-scale metric is <=0 or has abs(value)<1e-12. Both yield NaN, without pseudocounts. The outer ratio-of-ratios uses an exact-zero denominator guard and reports its denominator by name.'),
        ('README', 'Definitions, threshold values, replicate structure, filtering, and sheet guide.'),
        ('per_well', 'Each condition/well/threshold: nucleus counts, metric finite and excluded counts, means and medians; source na_reason_* counts retained where supplied.'),
        ('per_arm', "Every arm (not only treated/control) at every threshold multiplier: n_wells, wells, n_nuclei, and mean_of_well_means_<metric> = equal-weight mean of the arm's well means (NaN when any well mean is missing, never silently dropped); n_wells_finite_<metric> counts wells with a finite mean. Descriptive."),
        ('descriptive_qc', 'Astra F3: Costes p, Costes block size / count / coverage / ACF and PSF widths / draws, and the percentile-score chance references are DESCRIPTIVE QC only. They are excluded from per_well means, per_arm, contrast, ratios and sensitivity; descriptive_qc gives per-well n_finite, median, and for costes_rand_p min/q25/q75/max plus frac_costes_rand_p_at_floor (p = 1/(1+n_draws)) and the Costes NA reasons. A mean of per-nucleus randomization p-values is not a well- or arm-level p-value.'),
        ('spot_pooled_upp', 'Astra F4 sensitivity in per_well: spot_pooled_<percentile metric> pools every scored spot of the well (weights n_spots_upp), beside the per-nucleus mean that weights each nucleus equally; per_arm carries mean_of_well_spot_pooled_<metric>.'),
        ('contrast_sheet', 'At multiplier 1.0: every well identifier and mean, equal-weight arm mean, treated-control difference and ratio. exploratory_boot_lo/hi apply to difference only.'),
        ('ratio_of_ratios', 'Secondary descriptive values only; no significance tests.'),
        ('within_well_correlation', 'Pearson and Spearman separately per condition/well/threshold; require >=10 finite paired nuclei with N>0 and >=5 distinct values in each variable; otherwise NaN with reason.'),
        ('sensitivity', 'One row per metric; contrast difference, ratio, and missingness reasons side by side for every multiplier.'),
        ('all_arms', 'Per-well summaries for arms outside the selected treated/control comparison.'),
    ]
    for column in ['miat_min_used', 'qki_min_used']:
        if column in data:
            for level, group in data.groupby('threshold_multiplier', sort=True):
                values = group[column].dropna().unique()
                entries.append((f'{column}_multiplier_{level:g}', ', '.join(map(str, sorted(values))) if len(values) else 'missing'))
        else:
            entries.append((column, 'missing from input'))
    for condition, group in per_well[per_well.threshold_multiplier.eq(1)].groupby('condition', sort=True):
        entries.append((f'observed_replicates_{condition}', f'{len(group)} wells; images per well: ' + ', '.join(f'{r.well}={r.n_images}' for r in group.itertuples())))
    for metric in metrics:
        if metric.startswith('obs_minus_null_'):
            fallback = 'Additional supplied observed minus null statistic; difference-scale; exact upstream definition missing; single-plane'
        elif metric.startswith('sat_frac_'):
            fallback = 'Additional supplied saturation fraction; single-plane'
        else:
            fallback = metric
        entries.append((metric, COLUMN_DEFINITIONS.get(metric, fallback)))
    return pd.DataFrame(entries, columns=['topic', 'description'])


def summarize(df, treated, control, seed=0, n_boot=2000):
    if treated == control:
        raise ValueError('Treated and control labels must differ')
    if not isinstance(n_boot, (int, np.integer)) or n_boot < 1:
        raise ValueError('n_boot must be a positive integer')
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    data = prepare_data(df)
    baseline = data[data.threshold_multiplier.eq(1.0)]
    if not {treated, control}.issubset(set(baseline.condition)):
        raise ValueError('Treated and control must both occur at threshold multiplier 1.0')
    metrics = _metrics(data)
    per_well = _per_well(data, metrics)
    contrast = _contrast(per_well, metrics, treated, control, 1.0)
    too_few_wells = any(baseline.loc[baseline.condition.eq(arm), 'well'].nunique() < 3
                        for arm in (treated, control))
    draws = (np.empty((0, len(metrics))) if too_few_wells
             else _bootstrap(baseline, metrics, treated, control, seed, n_boot))
    lower, upper, counts, reasons = [], [], [], []
    for i, row in enumerate(contrast.itertuples()):
        valid = draws[np.isfinite(draws[:, i]), i]
        counts.append(len(valid))
        reason = ('TOO_FEW_WELLS' if too_few_wells else 'MISSING_CONTRAST' if not np.isfinite(row.difference)
                  else 'NO_VALID_DRAWS' if not len(valid) else '')
        reasons.append(reason)
        bounds = np.percentile(valid, [2.5, 97.5]) if not reason else (np.nan, np.nan)
        lower.append(bounds[0])
        upper.append(bounds[1])
    contrast['exploratory_boot_lo'], contrast['exploratory_boot_hi'] = lower, upper
    contrast['exploratory_boot_valid_draws'] = counts
    contrast['exploratory_boot_na_reason'] = reasons
    contrast['permutation_design_note'] = PERMUTATION_NOTE
    sensitivity = pd.DataFrame({'metric': metrics})
    for level in sorted(data.threshold_multiplier.unique()):
        level_contrast = _contrast(per_well, metrics, treated, control, level)
        for column in ['difference', 'ratio', 'na_reason', 'ratio_na_reason']:
            sensitivity[f'{column}_multiplier_{level:g}'] = level_contrast[column].to_numpy()
    return dict(README=_readme(data, per_well, metrics, treated, control, seed, n_boot),
                per_well=per_well, per_arm=_per_arm(per_well, metrics), descriptive_qc=_descriptive_qc(data),
                contrast=contrast, ratio_of_ratios=_ratio_of_ratios(contrast),
                within_well_correlation=_correlations(data), sensitivity=sensitivity,
                all_arms=per_well[~per_well.condition.isin([treated, control])].reset_index(drop=True))
