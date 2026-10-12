"""Follow-up to the scoped final review (2026-10-12): flag round-trips, native copies, staged publication,
slide cells outside the validated block."""
import numpy as np
import pandas as pd
import pytest
from openpyxl import Workbook

from fishsuite.report import sensitivities as S

REBUILD = 'legacy statistics: rebuild from the source run with fishsuite >= '


def _stamped(table):
    out = table.copy()
    out[S.STATS_GENERATION_COLUMN] = S.STATS_GENERATION
    out[S.STATS_COMMIT_COLUMN] = S.STATS_COMMIT
    return out


def _contrasts(p=.02, flag=True):
    return pd.DataFrame(dict(endpoint=['nuclear_spot_fraction_dapi', 'b'], diff=[1., -1.], p_welch=[.3, .4],
                             p_mixed=[p, .5], sensitivity_mixed_p=[p, .5], sensitivity_mixed_status=['ok', 'ok'],
                             sensitivity_mixed_validation=['validated', 'validated'],
                             significant_mixed_raw_0p05=[flag, np.nan]))


@pytest.mark.parametrize('value,expected', [(True, True), (False, False), (1, True), (0, False), (1.0, True),
                                            (0.0, False), ('TRUE', True), ('false', False), (np.bool_(True), True)])
def test_stored_flag_accepts_excel_representations(value, expected):
    assert S.stored_mixed_flag(dict(endpoint='x', significant_mixed_raw_0p05=value)) is expected


@pytest.mark.parametrize('value', ['yes', 2, 0.5, np.inf, 'T'])
def test_stored_flag_refuses_malformed_values(value):
    with pytest.raises(S.LegacyStatisticsError, match=REBUILD):
        S.stored_mixed_flag(dict(endpoint='x', significant_mixed_raw_0p05=value))


def test_excel_round_trip_keeps_the_stored_decision(tmp_path):
    from fishsuite.report import workbook, final_delivery
    path = workbook.write(tmp_path / 'r.xlsx', {'Contrasts': _contrasts()}, {}, order=['Contrasts'], fresh=True)
    back = S.read_stats_excel(path)['Contrasts']
    assert S.stored_mixed_flag(back.iloc[0]) is True  # Codex: 1.0 was read as False
    body = final_delivery.narrative({'slides': [{'identity': 'q1_localization'}]}, back)['slides'][0]['body']
    assert 'detects a difference' in body and 'does not detect' not in body


def test_native_regeneration_never_copies_legacy_inference_even_when_stamped(tmp_path):
    from test_native_by_condition import tiny_run
    from fishsuite.core.native_by_condition import regenerate
    from fishsuite.config.schema import ConditionsCfg
    run = tiny_run(tmp_path)
    legacy = run / 'figures' / 'per_condition'
    gen2 = pd.DataFrame(dict(endpoint=['x'], p_mixed=[.001], sensitivity_mixed_validation=['validated'],
                             fishsuite_stats_generation=[2], fishsuite_stats_commit=['abcdef1']))
    gen2.to_csv(legacy / 'gen2.csv', index=False)
    invalid = _stamped(pd.DataFrame(dict(endpoint=['x'], p_mixed=[.001], sensitivity_mixed_validation=['not_validated:x'])))
    invalid.to_csv(legacy / 'invalid.csv', index=False)
    pd.DataFrame(dict(well=['a'], value=[1.])).to_csv(legacy / 'descriptive.csv', index=False)
    out = tmp_path / 'new'
    regenerate(run, ConditionsCfg(groups={'WT': ['a', 'b'], 'KO': ['c', 'd']}), out)
    copied = out / 'figures/per_well_supplementary/per_condition'
    assert not (copied / 'gen2.csv').exists() and not (copied / 'invalid.csv').exists()
    assert (copied / 'descriptive.csv').exists()


def test_finish_validates_the_whole_bundle_before_any_write(tmp_path):
    from fishsuite.report.object_delivery import finish
    contrasts = _stamped(_contrasts())
    cache = _stamped(pd.DataFrame(dict(slide=[1], source_sheet=['Contrasts'], source_cell=['D3'], label=['x: p_mixed'],
                                       value=[.001])))  # stale: Contrasts!D3 holds .02
    out = tmp_path / 'delivery'
    spec = {'slides': [{'title': 't', 'values': [], 'figures': []}]}
    with pytest.raises(Exception, match='stale Slide values lineage'):
        finish(out, {'Contrasts': contrasts, 'Slide values': cache}, spec, 'REPORT.xlsx', 'deck.pptx', [])
    assert not out.exists()


def _book(path, rows_by_sheet):
    wb = Workbook()
    wb.remove(wb.active)
    for name, rows in rows_by_sheet.items():
        ws = wb.create_sheet(name)
        for r in rows:
            ws.append(list(r))
    wb.save(path)
    return path


def test_slide_cells_outside_the_validated_block_are_refused(tmp_path):
    from fishsuite.report import slides
    good = _stamped(_contrasts())
    valid_rows = [('desc',), tuple(good.columns)] + [tuple(None if pd.isna(v) else v for v in r)
                                                     for r in good.itertuples(index=False)]
    bad_row = ('bad', 0., .3, 1e-4, 1e-4, 'ok', 'not_validated:x', True, S.STATS_GENERATION, S.STATS_COMMIT)
    cases = [('single.xlsx', {'P': [('p_mixed',), (1e-4,)]}, 'A2'),
             ('postblank.xlsx', {'P': valid_rows + [(), bad_row]}, 'D6')]
    for name, sheets, cell in cases:
        path = _book(tmp_path / name, sheets)
        spec = {'workbook_sha256': slides.sha256(path), 'slides': [{'title': 't', 'values': [{'sheet': 'P', 'cell': cell}]}]}
        with pytest.raises(S.LegacyStatisticsError, match=REBUILD):
            slides.validate_deck(path, spec)
    path = _book(tmp_path / 'notes.xlsx', {'P': valid_rows, 'Notes': [('free text only',), (), ('p_mixed 1e-4',)]})
    spec = {'workbook_sha256': slides.sha256(path), 'slides': [{'title': 't', 'values': [{'sheet': 'Notes', 'cell': 'A3'}]}]}
    with pytest.raises(S.LegacyStatisticsError, match='outside the validated table block'):
        slides.validate_deck(path, spec)
    spec = {'workbook_sha256': slides.sha256(path), 'slides': [{'title': 't', 'values': [{'sheet': 'P', 'cell': 'D3'}]}]}
    assert slides.validate_deck(path, spec)[0]['values'][0]['value'] == .02


def test_a17_json_and_dapi_sensitivity_csv_writers_stamp():
    import inspect
    from fishsuite.report import rnaseh2b_analysis, build
    src = inspect.getsource(rnaseh2b_analysis)
    assert src.count('**stamp)') == 3 and 'fishsuite_stats_generation=STATS_GENERATION' in src
    assert "stamp_statistics(dapi[key], f'dapi_{key}', fresh=True)" in inspect.getsource(build)


def test_native_excludes_compressed_and_extensionless_legacy_statistics(tmp_path):
    import gzip, json
    from test_native_by_condition import tiny_run
    from fishsuite.core.native_by_condition import regenerate
    from fishsuite.config.schema import ConditionsCfg
    run = tiny_run(tmp_path)
    legacy = run / 'figures' / 'per_condition'
    table = pd.DataFrame(dict(endpoint=['x'], p_mixed=[.001], sensitivity_mixed_validation=['not_validated:x']))
    table.to_csv(legacy / 'stats.csv.gz', index=False, compression='gzip')
    table.to_csv(legacy / 'stats.tsv.gz', index=False, sep='\t', compression='gzip')
    table.to_csv(legacy / 'stats_noext', index=False)
    (legacy / 'stats.json').write_text(json.dumps({'rows': [{'endpoint': 'x', 'p_mixed': .001}]}), encoding='utf-8')
    with gzip.open(legacy / 'renamed.dat', 'wt') as handle:
        handle.write('endpoint\tmde_hedges_g\nx\t1.2\n')
    pd.DataFrame(dict(well=['a'], value=[1.])).to_csv(legacy / 'descriptive.csv.gz', index=False, compression='gzip')
    out = tmp_path / 'new'
    regenerate(run, ConditionsCfg(groups={'WT': ['a', 'b'], 'KO': ['c', 'd']}), out)
    copied = out / 'figures/per_well_supplementary/per_condition'
    for name in ('stats.csv.gz', 'stats.tsv.gz', 'stats_noext', 'stats.json', 'renamed.dat'):
        assert not (copied / name).exists(), name
    assert (copied / 'descriptive.csv.gz').exists() and (copied / 'original.png').exists()


def test_finish_checks_full_scale_assets_before_publication(tmp_path):
    from fishsuite.report.object_delivery import finish
    from fishsuite.report.provenance import sha256
    figure = tmp_path / 'fig.png'
    figure.write_bytes(b'focus')
    asset = dict(path=str(figure), sha256=sha256(figure), cohort=None,
                 full_path=str(tmp_path / 'missing_full.png'), full_sha256='0' * 64)
    spec = {'slides': [{'title': 'Figure', 'values': [], 'figures': [asset]}]}
    out = tmp_path / 'delivery'
    with pytest.raises(Exception, match='missing figure'):
        finish(out, {'Contrasts': _stamped(_contrasts())}, spec, 'REPORT.xlsx', 'deck.pptx', [])
    assert not out.exists()
