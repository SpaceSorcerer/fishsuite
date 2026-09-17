from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def toy_table():
    rows = []
    for condition, well, n, c, intensity in (
        ('control', 'C1', 2, 1, 10), ('control', 'C2', 6, 3, 30),
        ('treated', 'T1', 8, 6, 40), ('treated', 'T2', 16, 12, 80),
    ):
        for nucleus in range(5):
            rows.append(dict(condition=condition, well=well, image=well + '.tif',
                nucleus_id=nucleus + 1, threshold_multiplier=1.0,
                miat_min_used=10, qki_min_used=20, measurement_plane='single-plane',
                n_miat_spots=n, n_miat_spots_qki_pos=c, integrated_nuclear_miat=intensity,
                nuclear_area_um2=100, mean_nuclear_qki=10, mean_qki_at_miat_spots=12,
                qki_at_spots_minus_nuclear=2, frac_miat_spots_qki_pos=c / n,
                obs_minus_null_frac_miat_spots_qki_pos=c / n - .25,
                frac_miat_footprint_area_qki_pos=c / n,
                obs_minus_null_frac_miat_footprint_area_qki_pos=c / n - .25,
                sat_frac_miat=0, sat_frac_qki=0))
    return pd.DataFrame(rows)


def summary(df=None, **kwargs):
    from fishsuite.report.coupling_stats import summarize
    return summarize(toy_table() if df is None else df, 'treated', 'control', n_boot=kwargs.pop('n_boot', 30), **kwargs)


def three_well_table(inner_variation=False):
    rows = []
    template = toy_table().iloc[0].to_dict()
    for condition in ('control', 'treated'):
        for index, (constant, count) in enumerate(zip((3, 33, 63), (5, 5, 10)), 1):
            well = f'{condition}_{index}'
            values = (3, 9) if inner_variation else [constant] * count
            for nucleus, count_value in enumerate(values, 1):
                rows.append(dict(template, condition=condition, well=well, image=well + '.tif',
                                 nucleus_id=nucleus, n_miat_spots=count_value,
                                 integrated_nuclear_miat=count_value * 5))
    return pd.DataFrame(rows)


def test_literal_summary_and_secondary_ratios():
    result = summary()
    wells = result['per_well'].set_index('well')
    assert wells.loc['C1', 'mean_n_miat_spots'] == 2
    assert wells.loc['C2', 'median_n_miat_spots'] == 6
    assert wells.loc['T1', 'n_nuclei'] == 5
    assert wells.loc['T1', 'n_nuclei_with_spots'] == 5
    row = result['contrast'].set_index('metric').loc['n_miat_spots']
    assert row.treated_mean_of_well_means == 12
    assert row.control_mean_of_well_means == 4
    assert row.difference == 8
    assert row.ratio == 3
    assert (row.control_well_1_mean, row.control_well_2_mean) == (2, 6)
    assert (row.treated_well_1_mean, row.treated_well_2_mean) == (8, 16)
    ratios = result['ratio_of_ratios'].set_index('metric')
    assert ratios.loc['ror_count', 'value'] == 1.5
    assert ratios.loc['ror_intensity', 'value'] == 1.5
    assert ratios.loc['ror_corrected', 'value'] == .25
    assert 'difference' in ratios.loc['ror_corrected', 'definition']


def test_unequal_nuclei_keep_equal_well_weights():
    df = toy_table()
    df = pd.concat([df[df.well == w].iloc[[0] * size] for w, size in
                    [('C1', 4), ('C2', 12), ('T1', 4), ('T2', 12)]], ignore_index=True)
    df['nucleus_id'] = df.groupby('well').cumcount() + 1
    row = summary(df)['contrast'].set_index('metric').loc['n_miat_spots']
    assert row.control_mean_of_well_means == 4
    assert row.treated_mean_of_well_means == 12
    assert row.ratio == 3


def test_zero_control_abundance_is_nan_with_reason():
    df = toy_table()
    df.loc[df.condition == 'control', 'n_miat_spots'] = 0
    ratio = summary(df)['ratio_of_ratios'].set_index('metric').loc['ror_count']
    assert pd.isna(ratio.value)
    assert ratio.na_reason == 'CONTROL_NONPOSITIVE'


@pytest.mark.parametrize('count,distinct,expected', [(9, 9, None), (12, 12, 1.0), (12, 4, None)])
def test_correlation_gate(count, distinct, expected):
    df = pd.concat([toy_table().iloc[[0] * count], toy_table().iloc[[10] * count]], ignore_index=True)
    df['nucleus_id'] = df.groupby('well').cumcount() + 1
    vals = np.arange(count) % distinct + 1
    df['n_miat_spots'] = np.tile(vals, 2)
    df['integrated_nuclear_miat'] = df.n_miat_spots * 10
    df['obs_minus_null_frac_miat_spots_qki_pos'] = df.n_miat_spots / 100
    df['qki_at_spots_minus_nuclear'] = df.n_miat_spots * 2
    rows = summary(df)['within_well_correlation']
    if expected is None:
        assert rows.spearman_r.isna().all()
        assert rows.na_reason.str.len().gt(0).all()
    else:
        assert np.allclose(rows.spearman_r, expected)
        assert np.allclose(rows.pearson_r, 1.0)
        assert rows.na_reason.eq('').all()


def test_bootstrap_seed_and_exploratory_columns():
    first = summary(three_well_table(), seed=42)['contrast']
    second = summary(three_well_table(), seed=42)['contrast']
    pd.testing.assert_frame_equal(first, second)
    assert {'exploratory_boot_lo', 'exploratory_boot_hi'} <= set(first)
    assert 'boot_lo' not in first and 'boot_hi' not in first
    row = first.set_index('metric').loc['n_miat_spots']
    assert np.isfinite(row.exploratory_boot_lo)
    assert row.exploratory_boot_na_reason == ''


def test_bootstrap_resamples_wells_with_hand_derived_support():
    result = summary(three_well_table(), seed=7, n_boot=5000)['contrast'].set_index('metric')
    row = result.loc['n_miat_spots']
    # Six independent ternary draws give 729 allocations. CDF(-50)=7/729;
    # CDF(-40)=28/729, so the 2.5/97.5 percentile endpoints are -40 and 40.
    assert (row.exploratory_boot_lo, row.exploratory_boot_hi) == (-40, 40)
    hierarchical_support = set(range(-60, 61, 10))
    # A pooled nucleus-only resample of 20 nuclei per arm has a 30/20=1.5 lattice.
    pooled_nucleus_only_support = {1.5 * k for k in range(-40, 41)}
    fixed_well_nucleus_only_support = {0}
    for bound in (row.exploratory_boot_lo, row.exploratory_boot_hi):
        assert bound in hierarchical_support
        assert bound not in pooled_nucleus_only_support
        assert bound not in fixed_well_nucleus_only_support
    intensity = result.loc['integrated_nuclear_miat']
    assert (intensity.exploratory_boot_lo, intensity.exploratory_boot_hi) == (-200, 200)


def test_bootstrap_resamples_nuclei_inside_selected_wells():
    result = summary(three_well_table(inner_variation=True), seed=7, n_boot=5000)['contrast'].set_index('metric')
    row = result.loc['n_miat_spots']
    # Every fixed well mean is 6; wells-only support is {0}. Twelve Bernoulli
    # draws give CDF(-4)=79/4096 and CDF(-3)=299/4096, hence bounds -3 and 3.
    assert (row.exploratory_boot_lo, row.exploratory_boot_hi) == (-3, 3)


def test_two_wells_skip_bootstrap_and_explain_why(monkeypatch):
    from fishsuite.report import coupling_stats
    def forbidden(*args, **kwargs):
        raise AssertionError('bootstrap must not run with fewer than three wells per arm')
    monkeypatch.setattr(coupling_stats, '_bootstrap', forbidden)
    result = summary()
    contrast = result['contrast']
    assert contrast.exploratory_boot_lo.isna().all()
    assert contrast.exploratory_boot_hi.isna().all()
    assert contrast.exploratory_boot_valid_draws.eq(0).all()
    assert contrast.exploratory_boot_na_reason.eq('TOO_FEW_WELLS').all()
    assert 'with fewer than 3 wells per arm no interval is reported; read the individual well means' in '\n'.join(result['README'].description)


@pytest.mark.parametrize('metric', [
    'obs_minus_null_frac_miat_spots_qki_pos', 'obs_minus_null_frac_miat_footprint_area_qki_pos',
    'qki_at_spots_minus_nuclear', 'obs_minus_null_extra_statistic'])
def test_difference_scale_ratios_are_suppressed(metric):
    df = toy_table()
    df[metric] = np.where(df.condition.eq('control'), -3.5e-18, .2)
    level = df.copy()
    level['threshold_multiplier'] = .8
    result = summary(pd.concat([df, level], ignore_index=True))
    row = result['contrast'].set_index('metric').loc[metric]
    assert row.difference == pytest.approx(.2)
    assert pd.isna(row.ratio)
    assert row.ratio_na_reason == 'DIFFERENCE_SCALE'
    sensitivity = result['sensitivity'].set_index('metric').loc[metric]
    for level in ('0.8', '1'):
        assert pd.isna(sensitivity[f'ratio_multiplier_{level}'])
        assert sensitivity[f'ratio_na_reason_multiplier_{level}'] == 'DIFFERENCE_SCALE'


@pytest.mark.parametrize('control_mean', [0, -1, 1e-13, -1e-13])
def test_ratio_scale_control_nonpositive_or_near_zero(control_mean):
    df = toy_table()
    df['mean_nuclear_qki'] = df.mean_nuclear_qki.astype(float)
    df.loc[df.condition.eq('control'), 'mean_nuclear_qki'] = control_mean
    row = summary(df)['contrast'].set_index('metric').loc['mean_nuclear_qki']
    assert pd.isna(row.ratio)
    assert row.ratio_na_reason == 'CONTROL_NONPOSITIVE'


def test_zero_spot_nucleus_retained_for_counts_excluded_for_fraction():
    df = toy_table()
    rows = df.index[df.well.eq('C1')]
    df.loc[rows, 'n_miat_spots'] = [0, 4, 6, 10, 10]
    df.loc[rows, 'n_miat_spots_qki_pos'] = [0, 1, 2, 3, 3]
    fractions = np.array([np.nan, .25, 1/3, .3, .3])
    for metric in ('frac_miat_spots_qki_pos', 'frac_miat_footprint_area_qki_pos'):
        df.loc[rows, metric] = fractions
        df.loc[rows, 'obs_minus_null_' + metric] = fractions - .25
    row = summary(df)['per_well'].set_index('well').loc['C1']
    assert row.n_nuclei == 5
    assert row.n_nuclei_with_spots == 4
    assert row.n_finite_n_miat_spots == 5
    assert row.n_excluded_n_miat_spots == 0
    assert row.mean_n_miat_spots == 6
    assert row.mean_n_miat_spots_qki_pos == 1.8
    for metric, mean in (
        ('frac_miat_spots_qki_pos', 71/240), ('frac_miat_footprint_area_qki_pos', 71/240),
        ('obs_minus_null_frac_miat_spots_qki_pos', 11/240),
        ('obs_minus_null_frac_miat_footprint_area_qki_pos', 11/240),
    ):
        assert row[f'n_finite_{metric}'] == 4
        assert row[f'n_excluded_{metric}'] == 1
        assert row[f'mean_{metric}'] == pytest.approx(mean)
    assert row.n_finite_sat_frac_miat == 5
    assert row.n_excluded_sat_frac_miat == 0


def test_ratio_of_ratios_allows_tiny_positive_abundance_ratio():
    df = toy_table()
    df['n_miat_spots'] = np.where(df.condition.eq('control'), 1e14, 1)
    df['n_miat_spots_qki_pos'] = 1
    row = summary(df)['ratio_of_ratios'].set_index('metric').loc['ror_count']
    assert row.value == 1e14
    assert row.na_reason == ''


def test_ratio_of_ratios_zero_treated_abundance_has_denominator_reason():
    df = toy_table()
    df.loc[df.condition.eq('treated'), ['n_miat_spots', 'n_miat_spots_qki_pos']] = 0
    row = summary(df)['ratio_of_ratios'].set_index('metric').loc['ror_count']
    assert pd.isna(row.value)
    assert 'zero n_miat_spots treated/control ratio' in row.na_reason
    assert 'CONTROL_NONPOSITIVE' not in row.na_reason


def test_empty_well_falls_back_to_image_and_readme_records_it():
    df = toy_table()
    df['well'] = ''
    result = summary(df)
    assert set(result['per_well'].well) == {'C1.tif', 'C2.tif', 'T1.tif', 'T2.tif'}
    text = result['README'].to_string(index=False).lower()
    assert 'fallback' in text and 'image' in text and '20' in text
    assert 'single-plane' in text


def test_sensitivity_all_arms_and_missing_metric_values():
    df = toy_table()
    extra = df[df.well == 'C1'].copy()
    extra['condition'], extra['well'], extra['image'] = 'other', 'O1', 'O1.tif'
    df = pd.concat([df, extra], ignore_index=True)
    level = df.copy()
    level['threshold_multiplier'] = .8
    df = pd.concat([df, level], ignore_index=True)
    df.loc[df.well == 'C1', 'mean_nuclear_qki'] = np.nan
    result = summary(df)
    assert set(result['all_arms'].condition) == {'other'}
    assert len(result['all_arms']) == 2
    assert {'difference_multiplier_0.8', 'ratio_multiplier_0.8',
            'difference_multiplier_1', 'ratio_multiplier_1'} <= set(result['sensitivity'])
    row = result['contrast'].set_index('metric').loc['mean_nuclear_qki']
    assert pd.isna(row.control_mean_of_well_means)
    assert 'missing' in row.na_reason.lower()


def test_duplicate_nuclei_rejected():
    df = toy_table()
    with pytest.raises(ValueError, match='duplicate'):
        summary(pd.concat([df, df.iloc[[0]]], ignore_index=True))


def test_missing_well_at_sensitivity_threshold_is_not_silently_dropped():
    baseline = toy_table()
    level = baseline[baseline.well.ne('C2')].copy()
    level['threshold_multiplier'] = .8
    result = summary(pd.concat([baseline, level], ignore_index=True))
    row = result['sensitivity'].set_index('metric').loc['n_miat_spots']
    assert pd.isna(row['difference_multiplier_0.8'])
    assert 'missing' in row['na_reason_multiplier_0.8']
    absent = result['per_well'].query("well == 'C2' and threshold_multiplier == 0.8")
    assert len(absent) == 1
    assert absent.iloc[0].n_nuclei == 0
    assert pd.isna(absent.iloc[0].mean_n_miat_spots)


def test_report_artifacts(tmp_path):
    from click.testing import CliRunner
    from fishsuite.cli import cli
    from fishsuite.report.coupling_stats import METRICS
    from openpyxl import load_workbook
    source = tmp_path / 'association.csv'
    baseline = toy_table()
    sensitivity = baseline.copy()
    sensitivity['threshold_multiplier'] = .8
    sensitivity['miat_min_used'], sensitivity['qki_min_used'] = 8, 16
    pd.concat([baseline[baseline.well.ne('C2')], sensitivity], ignore_index=True).to_csv(source, index=False)
    out = tmp_path / 'report'
    config = tmp_path / 'colors.yaml'
    config.write_text('condition_order: [treated, control]\ngroup_colors:\n  treated: "#0072B2"\n  control: "#7F7F7F"\n', encoding='utf-8')
    result = CliRunner().invoke(cli, ['coupling', '--assoc-csv', str(source),
        '--treated', 'treated', '--control', 'control', '--out', str(out),
        '--config', str(config), '--seed', '4', '--n-boot', '10'])
    assert result.exit_code == 0, result.output + repr(result.exception)
    for name in ('coupling_summary.xlsx', 'command.log', 'versions.txt'):
        assert (out / name).stat().st_size > 0
    book = load_workbook(out / 'coupling_summary.xlsx', read_only=False)
    assert book.sheetnames == ['README', 'per_well', 'contrast', 'ratio_of_ratios',
                              'within_well_correlation', 'sensitivity', 'all_arms']
    assert book['per_well'].auto_filter.ref
    assert book['per_well'].freeze_panes == 'A3'
    readme_text = '\n'.join(str(cell.value) for row in book['README'] for cell in row if cell.value is not None)
    assert 'with fewer than 3 wells per arm no interval is reported; read the individual well means' in readme_text
    figure_files = list(out.rglob('*.png')) + list(out.rglob('*.svg'))
    assert figure_files
    for path in figure_files:
        assert path.stat().st_size > 0
        if path.suffix == '.svg':
            svg = path.read_text(encoding='utf-8')
            assert 'single-plane' in svg
            assert 'Filter:' in svg
            assert 'threshold_multiplier=' in svg
            assert str(source) in svg
            import re
            for color in re.findall(r'#[0-9a-fA-F]{6}\b', svg):
                color = color.lower()
                assert color in {'#0072b2', '#7f7f7f'} or color[1:3] == color[3:5] == color[5:7]
    distribution_svg = next(p for p in figure_files if p.name == 'distribution_n_miat_spots_full.svg').read_text(encoding='utf-8')
    assert '#0072b2' in distribution_svg.lower()
    assert '#7f7f7f' in distribution_svg.lower()
    import xml.etree.ElementTree as ET
    label_x = {node.text: float(node.attrib['x']) for node in
               ET.fromstring(distribution_svg).iter('{http://www.w3.org/2000/svg}text')
               if node.text in ('treated', 'control')}
    assert label_x['treated'] < label_x['control']
    assert 'control: C2 (NA)' in distribution_svg
    sensitivity_svg = next(p for p in figure_files if p.name == 'sensitivity_n_miat_spots_full.svg').read_text(encoding='utf-8')
    sensitivity_root = ET.fromstring(sensitivity_svg)
    x_axis = next(node for node in sensitivity_root.iter() if node.attrib.get('id') == 'matplotlib.axis_1')
    x_labels = [''.join(node.itertext()).strip() for node in x_axis.iter('{http://www.w3.org/2000/svg}text')]
    assert '0.8' in x_labels and '1' in x_labels
    sensitivity_text = ' '.join(' '.join(node.itertext()) for node in sensitivity_root.iter('{http://www.w3.org/2000/svg}text'))
    assert 'NA' in sensitivity_text and 'missing well' in sensitivity_text
    from PIL import Image
    with Image.open(next(p for p in figure_files if p.suffix == '.png')) as image:
        assert image.info['dpi'][0] == pytest.approx(600, abs=1)
    for kind in ('distribution', 'well_means', 'sensitivity'):
        for metric in METRICS:
            for view in ('full', 'focus'):
                for ext in ('png', 'svg'):
                    assert any(p.name == f'{kind}_{metric}_{view}.{ext}' for p in figure_files)
    for stem in ('precision_funnel',):
        for view in ('full', 'focus'):
            for ext in ('png', 'svg'):
                assert any(p.name == f'{stem}_{view}.{ext}' for p in figure_files)
    for abundance in ('n_miat_spots', 'integrated_nuclear_miat'):
        for association in ('obs_minus_null_frac_miat_spots_qki_pos', 'qki_at_spots_minus_nuclear'):
            for view in ('full', 'focus'):
                for ext in ('png', 'svg'):
                    assert any(p.name == f'scatter_{abundance}_vs_{association}_{view}.{ext}' for p in figure_files)
    book.close()


def test_cli_registration():
    from click.testing import CliRunner
    from fishsuite.cli import cli
    result = CliRunner().invoke(cli, ['coupling', '--help'])
    assert result.exit_code == 0, result.output
    assert '--assoc-csv' in result.output
    assert '--treated' in result.output
    assert '--control' in result.output


def test_well_markers_are_unique_across_arms():
    from fishsuite.report.coupling import _well_markers
    mapping = _well_markers(toy_table(), ['control', 'treated'])
    assert set(mapping) == {('control', 'C1'), ('control', 'C2'), ('treated', 'T1'), ('treated', 'T2')}
    assert len(set(mapping.values())) == 4
    assert mapping == _well_markers(toy_table().sample(frac=1, random_state=3), ['control', 'treated'])


def test_red_green_guard_and_resolved_colors(tmp_path):
    from fishsuite.report.coupling import _configuration
    config = tmp_path / 'forbidden.yaml'
    config.write_text('group_colors:\n  treated: "#D55E00"\n  control: "#009E73"\n', encoding='utf-8')
    with pytest.raises(ValueError, match='red and green'):
        _configuration(config, ['control', 'treated'], 'treated')
    order, colors = _configuration(None, ['control', 'treated', 'other'], 'treated')
    assert order == ['control', 'treated', 'other']
    assert {key: colors[key].upper() for key in order} == {
        'control': '#7F7F7F', 'treated': '#56B4E9', 'other': '#7F7F7F'}
