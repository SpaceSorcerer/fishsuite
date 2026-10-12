"""One mechanism for persisted statistics: writers stamp, readers require generation >= 2 (2026-10-12)."""
import numpy as np
import pandas as pd
import pytest

from fishsuite.report import sensitivities as S

REBUILD = 'legacy statistics: rebuild from the source run with fishsuite >= '


def _validated_contrasts():
    # ba24827-style table: validated mixed inference, correct-looking Holm, but built before the stamp existed.
    return pd.DataFrame(dict(endpoint=['x', 'y'], p_welch=[.4, .3], p_mixed=[.03, .04], sensitivity_mixed_p=[.03, .04],
                             p_headline=[.03, .04], headline_test=['mixed model'] * 2, p_headline_holm=[.03, .04],
                             holm_family_size=[1, 1], significant_headline_holm_0p05=[True, True],
                             sensitivity_mixed_validation=['validated'] * 2))


@pytest.mark.parametrize('case', ['pre_stamp_ba24827', 'generation_1', 'nan_generation', 'headline_schema_only',
                                  'ci_only', 'slide_cache_without_stamp'])
def test_reader_refuses_everything_without_a_current_stamp(case):
    table = _validated_contrasts()
    if case == 'generation_1':
        table = table.assign(fishsuite_stats_generation=1)
    elif case == 'nan_generation':
        table = table.assign(fishsuite_stats_generation=[2, np.nan])
    elif case == 'headline_schema_only':  # Codex (b): incomplete schema, invalid marker, no headline type
        table = pd.DataFrame(dict(p_headline=[1e-5], p_headline_holm=[2e-5], sensitivity_mixed_validation=['x']))
    elif case == 'ci_only':  # Codex (e): DATA_FOR_FIGURES carried a persisted CI
        table = pd.DataFrame(dict(condition=['a'], ci_low=[1.1], ci_high=[1.3]))
    elif case == 'slide_cache_without_stamp':
        table = pd.DataFrame(dict(slide=[1], source_sheet=['Contrasts'], source_cell=['B3'], label=['x: p_mixed'], value=[1e-5]))
    with pytest.raises(S.LegacyStatisticsError, match=REBUILD):
        S.require_current_statistics(table, case)


def test_stamped_current_table_passes_and_non_statistical_tables_need_no_stamp():
    stamped = S.stamp_statistics(_validated_contrasts(), fresh=True)
    assert stamped[S.STATS_GENERATION_COLUMN].eq(S.STATS_GENERATION).all() and S.STATS_GENERATION >= 2
    assert stamped[S.STATS_COMMIT_COLUMN].eq(S.STATS_COMMIT).all()
    assert S.require_current_statistics(stamped) is stamped
    plain = pd.DataFrame(dict(image=['a'], well_id=['w1'], value=[3.]))
    assert S.require_current_statistics(plain) is plain


def test_writer_refuses_partial_or_stale_stamps_and_keeps_an_existing_commit():
    stamped = S.stamp_statistics(_validated_contrasts(), fresh=True).assign(fishsuite_stats_commit='abc1234')
    assert S.stamp_statistics(stamped)['fishsuite_stats_commit'].eq('abc1234').all()
    mixed = pd.concat([stamped, _validated_contrasts()], ignore_index=True)  # stamped + unstamped rows
    with pytest.raises(S.LegacyStatisticsError):
        S.stamp_statistics(mixed)
    assert S.STATS_GENERATION_COLUMN in S.stamp_statistics(S.unstamp(mixed), fresh=True)


def test_workbook_write_stamps_and_reader_round_trips(tmp_path):
    from fishsuite.report import workbook
    path = workbook.write(tmp_path / 'r.xlsx', {'Contrasts': _validated_contrasts(),
                                                'Per well': pd.DataFrame(dict(well_id=['w'], value=[1.]))},
                          {}, order=['Contrasts', 'Per well'], fresh=True)
    sheets = S.read_stats_excel(path)
    assert sheets['Contrasts'][S.STATS_GENERATION_COLUMN].eq(S.STATS_GENERATION).all()
    assert S.STATS_GENERATION_COLUMN not in sheets['Per well']


def test_slide_cache_lineage_is_enforced_without_a_deck_schema(tmp_path):
    from fishsuite.report import workbook, slides
    contrasts = _validated_contrasts()
    cache = pd.DataFrame(dict(slide=[1], source_sheet=['Contrasts'], source_cell=['C3'], label=['x: p_mixed'],
                              value=[1e-5]))  # stale: the source cell holds .03
    path = workbook.write(tmp_path / 'r.xlsx', {'Contrasts': contrasts, 'Slide values': cache}, {},
                          order=['Contrasts', 'Slide values'], fresh=True)
    spec = {'workbook_sha256': slides.sha256(path), 'slides': [{'title': 't', 'values': []}]}
    with pytest.raises(Exception, match='stale Slide values lineage'):
        slides.validate_deck(path, spec)


def test_finish_refuses_before_writing_anything(tmp_path):
    from fishsuite.report.object_delivery import finish
    legacy = _validated_contrasts().assign(sensitivity_mixed_validation=['validated', 'not_validated:x'])
    out = tmp_path / 'delivery'
    with pytest.raises(S.LegacyStatisticsError, match=REBUILD):
        finish(out, {'Contrasts': legacy}, {'slides': [], 'a20': True}, 'REPORT.xlsx', 'deck.pptx', [])
    assert not out.exists()


def test_figure_workbook_render_refuses_unstamped_and_computes_no_test(tmp_path):
    from fishsuite.report.figure_workbook import render
    values = pd.DataFrame(dict(condition=['WT'] * 3 + ['KO'] * 3, well=list('abcdef'), value=[1., 2, 4, 7, 8, 11]))
    raw = tmp_path / 'raw.xlsx'
    values.to_excel(raw, sheet_name='Edit', index=False)
    with pytest.raises(S.LegacyStatisticsError, match=REBUILD):
        render(raw, 'Edit', tmp_path / 'r1')
    stamped = tmp_path / 'stamped.xlsx'
    S.stamp_statistics(values, force=True, fresh=True).to_excel(stamped, sheet_name='Edit', index=False)
    render(stamped, 'Edit', tmp_path / 'r2')
    svg = (tmp_path / 'r2' / 'Edit_focus.svg').read_text(encoding='utf-8')
    assert 'Welch' not in svg and 'p = ' not in svg and 'descriptive only' in svg


def test_native_regeneration_never_republishes_source_workbook_sheets(tmp_path):
    from test_native_by_condition import tiny_run
    from fishsuite.core.native_by_condition import regenerate
    from fishsuite.config.schema import ConditionsCfg
    run = tiny_run(tmp_path)
    pd.DataFrame(dict(endpoint=['x'], p_mixed=[.001])).to_excel(run / 'analysis_summary.xlsx', sheet_name='Legacy stats', index=False)
    out = tmp_path / 'new'
    regenerate(run, ConditionsCfg(groups={'WT': ['a', 'b'], 'KO': ['c', 'd']}), out)
    book = pd.ExcelFile(out / 'analysis_summary.xlsx')
    assert 'Legacy stats' not in book.sheet_names
    contrasts = pd.read_excel(book, sheet_name='Condition contrasts')
    assert contrasts[S.STATS_GENERATION_COLUMN].eq(S.STATS_GENERATION).all()


def test_slide_values_sheet_without_lineage_schema_is_refused(tmp_path):
    from fishsuite.report import workbook, slides
    path = workbook.write(tmp_path / 'r.xlsx', {'Slide values': pd.DataFrame(dict(label=['x: p_mixed'], value=[1e-5]))},
                          {}, order=['Slide values'], fresh=True)
    with pytest.raises(S.LegacyStatisticsError, match=REBUILD):
        slides.validate_deck(path, {'workbook_sha256': slides.sha256(path), 'slides': [{'title': 't', 'values': []}]})


def test_floor_table_reads_stored_flags_and_refuses_missing_ones():
    row = pd.Series(dict(endpoint='x', p_welch=.03, significant_raw_0p05=False, mixed_p=.2,
                         significant_mixed_raw_0p05='True'))
    assert S.stored_flag(row, 'p_welch') is False  # stored decision at the run's alpha, not p < .05 now
    assert S.stored_flag(row, 'mixed_p') is True
    with pytest.raises(S.LegacyStatisticsError, match=REBUILD):
        S.stored_flag(row.drop('significant_raw_0p05'), 'p_welch')
    with pytest.raises(S.LegacyStatisticsError, match=REBUILD):
        S.stored_flag(pd.concat([row.drop('significant_mixed_raw_0p05'), pd.Series({'significant_mixed_raw_0p05': np.nan})]), 'mixed_p')


def test_build_time_holm_stores_the_raw_mixed_flag_at_the_run_alpha():
    fresh = pd.DataFrame(dict(endpoint=['x', 'y'], sensitivity_mixed_p=[np.nan, .02], p_welch=[.8, .4],
                              sensitivity_mixed_validation=['not_validated:x', 'validated'],
                              family=['f', 'f'], test_group=['t', 't'], in_holm_family=[True, True]))
    out = S.add_headline_holm(S.add_mixed_columns(fresh), .01, ['family', 'test_group'])
    assert out.significant_mixed_raw_0p05.iloc[1] is False and pd.isna(out.significant_mixed_raw_0p05.iloc[0])


def test_pixel_pattern_welch_sheet_is_stamped(tmp_path):
    from fishsuite.core.pixel_pattern import _write_excel
    xlsx = tmp_path / 'pixel_pattern.xlsx'
    _write_excel(pd.DataFrame(dict(a=[1.])), pd.DataFrame(dict(metric=['gini'], welch_p=[.2])),
                 pd.DataFrame(), pd.DataFrame(), None, xlsx, chan_labels={})
    welch = pd.read_excel(xlsx, sheet_name='nt_vs_condition_welch')
    assert S.require_current_statistics(welch) is not None and welch[S.STATS_GENERATION_COLUMN].eq(S.STATS_GENERATION).all()
    assert S.STATS_GENERATION_COLUMN not in pd.read_excel(xlsx, sheet_name='per_nucleus')


def test_engine_inferential_sheets_declare_the_stamp_columns():
    import inspect
    from fishsuite.core import excel_report, singlecell
    from fishsuite.figures import gallery
    assert excel_report._STATS_STAMP_HEADERS == (S.STATS_GENERATION_COLUMN, S.STATS_COMMIT_COLUMN)
    for fn in (excel_report._build_comparison_table, excel_report._build_pi_focus):
        assert '_STATS_STAMP_HEADERS' in inspect.getsource(fn) and '_stats_stamp_values' in inspect.getsource(fn)
    assert 'stamp_statistics' in inspect.getsource(singlecell._write_excel)
    assert inspect.getsource(gallery).count('stamp_statistics(df, name, fresh=True)') == 2


# ---------------------------------------------------------------- round 9 (bounded scope, final2 review)
def _stamped(table, generation=None, commit=None):
    out = table.copy()
    out[S.STATS_GENERATION_COLUMN] = S.STATS_GENERATION if generation is None else generation
    out[S.STATS_COMMIT_COLUMN] = S.STATS_COMMIT if commit is None else commit
    return out


@pytest.mark.parametrize('generation,commit', [(2, None), (' 3 ', None), (3.5, None), (np.inf, None), (4, None),
                                               (None, ''), (None, ' abc1234'), (None, 'abc')])
def test_pre_floor_unsupported_or_malformed_stamps_are_refused(generation, commit):
    table = _stamped(_validated_contrasts(), generation, commit)
    with pytest.raises(S.LegacyStatisticsError, match=REBUILD):
        S.require_current_statistics(table)


def test_stamped_headline_without_provenance_is_refused():
    table = _stamped(pd.DataFrame(dict(endpoint=['x'], p_headline=[1e-5], p_headline_holm=[2e-5],
                                       sensitivity_mixed_validation=['not_validated:x'])))
    with pytest.raises(S.LegacyStatisticsError, match=REBUILD):
        S.require_current_statistics(table)
    welch = table.assign(headline_test='Welch (well means)')
    assert S.require_current_statistics(welch) is welch


@pytest.mark.parametrize('rows', [
    [('endpoint', 'p_mixed', 'sensitivity_mixed_validation'), ('a', 1e-4, 'not_validated:x')],  # row-1 header
    [('Metric', 'MWU p-value', "Cliff's delta"), ('spots', .001, .5)],                             # engine header
    [('Mann-Whitney comparison',), ('Metric', 'Mann-Whitney p', 'Direction'), ('spots', .001, 'up')]])
def test_row_one_and_engine_headers_are_gated(rows):
    with pytest.raises(S.LegacyStatisticsError, match=REBUILD):
        S.require_current_rows(rows, 'probe')


def test_description_row_is_not_mistaken_for_a_header():
    stamped = _stamped(_validated_contrasts())
    rows = [('Welch p and mixed p per endpoint',), tuple(stamped.columns)] + [tuple(r) for r in stamped.itertuples(index=False)]
    assert S.require_current_rows(rows, 'ok') == [list(r) for r in rows]


def test_writers_refuse_unstamped_tables_not_declared_fresh(tmp_path):
    from fishsuite.report import workbook
    from fishsuite.report.object_delivery import finish
    with pytest.raises(S.LegacyStatisticsError, match='not declared as built here'):
        workbook.write(tmp_path / 'r.xlsx', {'Contrasts': _validated_contrasts()}, {}, order=['Contrasts'])
    metadata = pd.DataFrame(dict(endpoint=['x'], holm_family_size=[2], mde_hedges_g_at_family_alpha=[9.],
                                 significant_headline_holm_0p05=[True]))
    with pytest.raises(S.LegacyStatisticsError):  # Codex: metadata-only export must not be stamped as current
        workbook.write(tmp_path / 'm.xlsx', {'Plan': metadata}, {}, order=['Plan'])
    out = tmp_path / 'delivery'
    with pytest.raises(S.LegacyStatisticsError, match='not declared as built here'):  # direct finish call
        finish(out, {'Contrasts': _validated_contrasts()}, {'slides': [], 'a20': True}, 'REPORT.xlsx', 'deck.pptx', [])
    assert not out.exists() and not (tmp_path / 'r.xlsx').exists()


def test_figure_render_gates_auxiliary_sheets_before_any_output(tmp_path):
    from fishsuite.report.figure_workbook import render
    values = pd.DataFrame(dict(condition=['WT'] * 3 + ['KO'] * 3, well=list('abcdef'), value=[1., 2, 4, 7, 8, 11]))
    book = tmp_path / 'DATA_FOR_FIGURES.xlsx'
    with pd.ExcelWriter(book) as xw:
        S.stamp_statistics(values, force=True, fresh=True).to_excel(xw, sheet_name='Main', index=False)
        pd.DataFrame(dict(condition=['WT'], ci_low=[1.1], ci_high=[1.3])).to_excel(xw, sheet_name='Main_nuc', index=False)
    out = tmp_path / 'render'
    with pytest.raises(S.LegacyStatisticsError, match=REBUILD):
        render(book, 'Main', out)
    assert not any(out.glob('*'))


def test_narrative_reads_the_stored_mixed_decision():
    from fishsuite.report import final_delivery
    row = dict(endpoint='nuclear_spot_fraction_dapi', diff=1., p_mixed=.03, sensitivity_mixed_p=.03,
               sensitivity_mixed_status='ok', sensitivity_mixed_validation='validated', significant_mixed_raw_0p05=False)
    spec = {'slides': [{'identity': 'q1_localization'}]}
    body = final_delivery.narrative(spec, pd.DataFrame([row]))['slides'][0]['body']
    assert 'does not detect' in body  # stored False at the run's alpha, although .03 < .05
    with pytest.raises(S.LegacyStatisticsError, match=REBUILD):
        final_delivery.narrative(spec, pd.DataFrame([{k: v for k, v in row.items() if k != 'significant_mixed_raw_0p05'}]))


def test_native_regeneration_does_not_copy_inferential_legacy_assets(tmp_path):
    from test_native_by_condition import tiny_run
    from fishsuite.core.native_by_condition import regenerate
    from fishsuite.config.schema import ConditionsCfg
    run = tiny_run(tmp_path)
    legacy = run / 'figures' / 'per_condition'
    pd.DataFrame(dict(endpoint=['x'], p_mixed=[.001])).to_csv(legacy / 'stats.csv', index=False)
    (legacy / 'panel.svg').write_text('<svg><text>mixed model p 0.001</text></svg>', encoding='utf-8')
    (legacy / 'panel.png').write_bytes(b'png')
    out = tmp_path / 'new'
    regenerate(run, ConditionsCfg(groups={'WT': ['a', 'b'], 'KO': ['c', 'd']}), out)
    copied = out / 'figures/per_well_supplementary/per_condition'
    assert not (copied / 'original.png').exists()  # fail-closed: legacy charts are regenerated, never copied
    assert not (copied / 'stats.csv').exists() and not (copied / 'panel.svg').exists() and not (copied / 'panel.png').exists()
