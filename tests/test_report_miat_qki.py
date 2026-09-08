from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_three_well_ratio_covariance_and_complete_exact_assignments():
    from fishsuite.report.miat_qki import ratio_interval, unblocked_exact_ratio
    pairs = pd.DataFrame(dict(biological_set=['WT1', 'WT2', 'WT3', 'KO1', 'KO2', 'KO3'],
                              slide=['one']*6, arm=['NT']*3+['KD']*3,
                              A=[2., 4., 6., 3., 6., 9.], T=[2., 4., 6., 2., 4., 6.]))
    result = ratio_interval(pairs, 'run_retained', expected_wells=3)
    assert result['R'] == pytest.approx(1.5)
    # A and T are perfectly proportional within each arm: covariance cancels.
    assert result['variance_log_R'] == pytest.approx(0, abs=1e-15)
    exact, assignments = unblocked_exact_ratio(pairs)
    assert exact['n_assignments'] == 20
    assert assignments.assigned_KD.nunique() == 20
    assert exact['p_exact'] == pytest.approx(.1)
    assert exact['p_exact_greater'] == pytest.approx(.05)

from fishsuite.report import miat_qki as mq

DATA = Path('F:/Image Analysis Work/MIAT_QKI_Coloc_2026_08_25/DELIVERIES/DELIVERY_MIAT_QKI_REVISED_2026-09-03/data')
COUNT = Path('F:/Image Analysis Work/MIAT_QKI_Coloc_2026_08_25/RAW_RUNS/MIAT_COUNT_2026-08-28_0020/report_2026-09-05_nuclearfraction_count_allnuclei')


def fixture_pairs():
    return pd.DataFrame({'biological_set': [f'S{s}_{a}_{i}' for s in (1, 2) for a in ('NT', 'KD') for i in (1, 2, 3)],
                         'slide': np.repeat([1, 2], 6), 'arm': ['NT']*3+['KD']*3+['NT']*3+['KD']*3,
                         'T': [10., 20., 30., 5., 10., 15.]*2,
                         'A': [2., 4., 6., 1., 2., 3.]*2})


def test_proportional_preferential_and_scaling():
    pairs = fixture_pairs()
    assert mq.ratio_interval(pairs)['R'] == pytest.approx(1)
    pairs.loc[pairs.arm == 'KD', 'A'] *= 1.2
    first = mq.ratio_interval(pairs)
    assert first['R'] == pytest.approx(1.2)
    pairs['A'] *= 17
    pairs['T'] *= 3
    for key in ('R', 'ci_low', 'ci_high'):
        assert mq.ratio_interval(pairs)[key] == pytest.approx(first[key])


def test_hand_covariance():
    pairs = fixture_pairs()
    pairs['A'] = [2, 3, 7, 1, 3, 2]*2
    # Six values in each arm: mean A=4/2, T=20/10;
    # sample variances A=5.6/.8, T=80/20; covariances=20/2.
    expected = 5.6/(6*16)+80/(6*400)-40/(6*80) + .8/(6*4)+20/(6*100)-4/(6*20)
    assert mq.ratio_interval(pairs)['variance_log_R'] == pytest.approx(expected)


@pytest.mark.parametrize('bad', [0, -1, np.nan, np.inf])
@pytest.mark.parametrize('column', ['T', 'A'])
def test_invalid_denominator(bad, column):
    pairs = fixture_pairs()
    pairs.loc[pairs.arm == 'NT', column] = bad
    result = mq.ratio_interval(pairs)
    assert np.isnan(result['R']) and result['reason']


def test_missing_pairs_and_count_rejected():
    assert mq.ratio_interval(fixture_pairs().iloc[:-1])['reason']
    with pytest.raises(ValueError, match='COUNT'):
        mq.ratio_interval(fixture_pairs(), denominator_source='COUNT localization')


@pytest.fixture(scope='module')
def analysis():
    if not DATA.is_dir():
        pytest.skip('named MIAT delivery missing')
    return mq.analyze(DATA)


def test_historical_reproduction(analysis):
    ratios = analysis['Ratio intervals']
    h = ratios[ratios.null_policy == mq.POLICIES[0]].set_index('measurement')
    for measurement, values in [('spot_count', (1.0798527401191855, .9003757874954351, 1.29510583973675)),
                                ('miat_intensity', (1.0801313437310476, .789076124111532, 1.4785439377270189))]:
        np.testing.assert_allclose(h.loc[measurement, ['R', 'ci_low', 'ci_high']].to_numpy(float), values, rtol=0, atol=1e-9)
    assert len(ratios) == 8
    np.testing.assert_allclose(ratios.p_exact, ratios.p_exact_delivered, rtol=0, atol=1e-9)
    assert (ratios.n_assignments == 400).all()


def test_roster_and_well_points(analysis):
    nuclei = analysis['Nucleus roster']
    assert nuclei.groupby('null_policy').size().tolist() == [370]*4
    assert nuclei.groupby(['null_policy', 'image_key']).size().eq(10).all()
    assert nuclei[nuclei.observed_spot_count == 0].groupby('null_policy').size().tolist() == [4]*4
    assert set(nuclei.slide) == {1, 2}
    wells = analysis['Ratio well values']
    assert wells.groupby(['null_policy', 'measurement', 'arm']).size().eq(6).all()
    assert analysis['Assignments'].assignment.nunique() == 400


def test_zero_well_preserved():
    pairs = fixture_pairs()
    pairs.loc[0, ['A', 'T']] = 0
    assert np.isfinite(mq.ratio_interval(pairs)['R'])


@pytest.mark.parametrize('measurement', mq.MEASUREMENTS)
@pytest.mark.parametrize('variant', ['full', 'focus'])
def test_actual_figure_wells_and_extents(analysis, variant, tmp_path, measurement):
    import matplotlib.pyplot as plt
    fig = mq.make_figure(analysis, DATA, measurement=measurement, variant=variant)
    for ax in fig.axes[:5]:
        wells = [c for c in ax.collections if (c.get_gid() or '').startswith('well:')]
        assert len(wells) == 2
        assert [len(c.get_offsets()) for c in wells] == [6, 6]
        low, high = ax.get_ylim()
        for c in wells:
            assert np.all((c.get_offsets()[:, 1] >= low) & (c.get_offsets()[:, 1] <= high))
        if variant == 'full':
            assert low == 0
    assert len({ax.get_ylim() for ax in fig.axes[:5]}) == 1
    kind = 'miat_count' if measurement == 'spot_count' else 'miat_intensity'
    for ax in fig.axes[:5]:
        assert ax._figure_data['endpoint'] in {'ratio_'+kind+'_total', 'ratio_'+kind+'_associated'}
        assert ax._axis_group == 'ratio_'+kind+('_count' if measurement == 'spot_count' else '_intensity')
        assert any('p =' in t.get_text() for t in ax.texts)
        assert len(ax.patches) == 2
    assert any('mixed model p' in t.get_text() and '; run ' in t.get_text() for t in fig.texts)
    for ext in ('png', 'svg'):
        path = tmp_path / ('ratio_' + variant + '.' + ext)
        fig.savefig(path, dpi=72)
        assert path.is_file()
    assert 'mixed model p' in (tmp_path / ('ratio_' + variant + '.svg')).read_text(encoding='utf-8')
    r = analysis['Ratio intervals'].loc[lambda x:x.measurement.eq(measurement)]
    low, high = fig.axes[5].get_xlim()
    assert low < min(1, r.ci_low.min())
    assert high > max(r.ci_high.max(), r.R_MDE_05_8.max())
    plt.close(fig)


def test_workbook_cells_and_cli(analysis, tmp_path):
    from click.testing import CliRunner
    from fishsuite.cli import cli
    import openpyxl
    result = CliRunner().invoke(cli, ['report', '--miat-qki', str(DATA), '--miat-qki-count-report', str(COUNT),
                                    '--out', str(tmp_path), '--no-figures'])
    assert result.exit_code == 0, result.output
    wb = openpyxl.load_workbook(tmp_path/'REPORT.xlsx', data_only=True, read_only=True)
    for name in ('Ratio intervals', 'Retention sensitivity', 'Coverage', 'COUNT localization', 'Multiplicity plan', 'Figure sources', 'Slide values'):
        assert wb[name]['A1'].value
    # Snapshot cells once: read-only worksheet lookup reparses XML on every access.
    snapshots = {ws.title: {c.coordinate: c.value for row in ws for c in row if c.value is not None} for ws in wb}
    rows = list(wb['Slide values'].values)
    headers = rows[1]
    for values in rows[2:]:
        row = dict(zip(headers, values))
        assert snapshots[row['sheet']][row['cell']] == pytest.approx(row['value'])
    wb.close()


def test_usable_count_and_separate_families(analysis):
    usable = analysis['Usable pool ratios']
    historical = usable[(usable.null_policy == mq.POLICIES[0]) & (usable.measurement == 'spot_count')].iloc[0]
    assert historical.R == pytest.approx(.916506257, abs=1e-9)
    assert usable[usable.measurement == 'miat_intensity'].R.isna().all()
    families = analysis['Multiplicity plan'].groupby('family').size().to_dict()
    assert families == {'exploratory_R_8':8, 'exploratory_scalar_10':10, 'frozen_historical_9':9, 'original_primary_gate':1}


def test_ratio_scalars_use_registered_locked_drawer(tmp_path, monkeypatch):
    from fishsuite.report import figures, rnaseh2b_analysis as rn
    calls = []
    original = figures.draw_replicate_simple
    def spy(*args, **kwargs):
        foot = original(*args, **kwargs)
        calls.append((args[2], args[0], foot))
        return foot
    monkeypatch.setattr(figures, 'draw_replicate_simple', spy)
    wells = fixture_pairs().iloc[:6].rename(columns={'biological_set': 'condition'})
    wells['arm'] = wells.arm.map({'NT': 'WT', 'KD': 'KO'})
    result = mq.ratio_interval(fixture_pairs())
    result['p_exact'] = .5
    records = rn.render_ratio(wells, pd.DataFrame(), result, tmp_path, tmp_path)
    assert len(calls) == 2
    assert {c[0] for c in calls} == {'ratio_protein_total', 'ratio_protein_associated'}
    for endpoint, ax, foot in calls:
        assert ax._axis_group == 'ratio_protein_intensity'
        assert any('p =' in t.get_text() for t in ax.texts)
        assert 'Welch (wells) p' in foot
        assert len(ax.patches) == 2
        assert ax._figure_data['nuclei'].empty
    assert calls[0][1].get_ylim() == calls[1][1].get_ylim()
    for record in records[:2]:
        for variant in ('full', 'focus'):
            assert Path(record[variant+'_png']).is_file()
            svg = Path(record[variant+'_svg']).read_text(encoding='utf-8')
            assert 'mixed model p' in svg and '; run ' in svg
            assert '#d67ae5' in svg.lower()
