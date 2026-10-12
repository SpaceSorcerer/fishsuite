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
    stamped = S.stamp_statistics(_validated_contrasts())
    assert stamped[S.STATS_GENERATION_COLUMN].eq(S.STATS_GENERATION).all() and S.STATS_GENERATION >= 2
    assert stamped[S.STATS_COMMIT_COLUMN].eq(S.STATS_COMMIT).all()
    assert S.require_current_statistics(stamped) is stamped
    plain = pd.DataFrame(dict(image=['a'], well_id=['w1'], value=[3.]))
    assert S.require_current_statistics(plain) is plain


def test_writer_refuses_partial_or_stale_stamps_and_keeps_an_existing_commit():
    stamped = S.stamp_statistics(_validated_contrasts()).assign(fishsuite_stats_commit='abc1234')
    assert S.stamp_statistics(stamped)['fishsuite_stats_commit'].eq('abc1234').all()
    mixed = pd.concat([stamped, _validated_contrasts()], ignore_index=True)  # stamped + unstamped rows
    with pytest.raises(S.LegacyStatisticsError):
        S.stamp_statistics(mixed)
    assert S.STATS_GENERATION_COLUMN in S.stamp_statistics(S.unstamp(mixed))


def test_workbook_write_stamps_and_reader_round_trips(tmp_path):
    from fishsuite.report import workbook
    path = workbook.write(tmp_path / 'r.xlsx', {'Contrasts': _validated_contrasts(),
                                                'Per well': pd.DataFrame(dict(well_id=['w'], value=[1.]))},
                          {}, order=['Contrasts', 'Per well'])
    sheets = S.read_stats_excel(path)
    assert sheets['Contrasts'][S.STATS_GENERATION_COLUMN].eq(2).all()
    assert S.STATS_GENERATION_COLUMN not in sheets['Per well']


def test_slide_cache_lineage_is_enforced_without_a_deck_schema(tmp_path):
    from fishsuite.report import workbook, slides
    contrasts = _validated_contrasts()
    cache = pd.DataFrame(dict(slide=[1], source_sheet=['Contrasts'], source_cell=['C3'], label=['x: p_mixed'],
                              value=[1e-5]))  # stale: the source cell holds .03
    path = workbook.write(tmp_path / 'r.xlsx', {'Contrasts': contrasts, 'Slide values': cache}, {},
                          order=['Contrasts', 'Slide values'])
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
    S.stamp_statistics(values, force=True).to_excel(stamped, sheet_name='Edit', index=False)
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
    assert contrasts[S.STATS_GENERATION_COLUMN].eq(2).all()


def test_slide_values_sheet_without_lineage_schema_is_refused(tmp_path):
    from fishsuite.report import workbook, slides
    path = workbook.write(tmp_path / 'r.xlsx', {'Slide values': pd.DataFrame(dict(label=['x: p_mixed'], value=[1e-5]))},
                          {}, order=['Slide values'])
    with pytest.raises(S.LegacyStatisticsError, match=REBUILD):
        slides.validate_deck(path, {'workbook_sha256': slides.sha256(path), 'slides': [{'title': 't', 'values': []}]})
