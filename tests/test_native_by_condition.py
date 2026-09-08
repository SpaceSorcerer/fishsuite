import json
from pathlib import Path

import pandas as pd
import pytest

from fishsuite.config.schema import ConditionsCfg
from fishsuite.core.native_by_condition import collapse_conditions, regenerate, read_groups


def test_groups_and_old_defaults(tmp_path):
    assert ConditionsCfg().groups == {}
    assert ConditionsCfg().resolved_group_order() == []
    cfg = ConditionsCfg(groups={'WT': ['a', 'b'], 'KO': ['c']},
                        order=['KO', 'WT'], colors={'KO': '#CC79A7'})
    assert cfg.resolved_group_order() == ['KO', 'WT']
    fallback = ConditionsCfg(groups={'WT':['a','b']}, group_order=['WT','c'])
    assert fallback.group_of('c') == 'c'
    assert fallback.resolved_group_order() == ['WT','c']
    p = tmp_path / 'groups.yaml'
    p.write_text('conditions:\n  groups: {WT: [a, b], KO: [c]}\n  order: [KO, WT]\n')
    assert read_groups(p).resolved_group_order() == ['KO', 'WT']
    with pytest.raises(ValueError, match='both'):
        ConditionsCfg(groups={'WT': ['a'], 'KO': ['a']})


def test_collapse_is_mean_of_well_means():
    wells = pd.DataFrame(dict(group=['WT']*2, endpoint=['x']*2,
        well_mean_of_field_values=[2., 10.], n_fields=[1, 9],
        n_nuclei_in_well=[2, 90]))
    row = collapse_conditions(wells).iloc[0]
    assert row['mean_of_well_means'] == 6
    assert row['sd_across_wells'] == pytest.approx(32**.5)
    assert (row.n_wells, row.n_FOVs, row.n_nuclei) == (2, 10, 92)


def tiny_run(tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    rows = [dict(image=f'{w}_{f}', condition=w, secondary_only=w=='Sec-Only',
                 nuclear_spot_count=v, nucleus_id=1)
            for w, f, v in [('a',1,2), ('b',1,8), ('b',2,12), ('c',1,15), ('d',1,20), ('Sec-Only',1,0)]]
    pd.DataFrame(rows).to_csv(run/'nuclei_metrics.csv', index=False)
    pd.DataFrame(rows).drop(columns=['nucleus_id']).to_csv(run/'per_image_summary.csv', index=False)
    (run/'run_config.json').write_text(json.dumps({'config_resolved': {}}))
    for folder in ['per_condition', '00_overview', 'per_image']:
        p = run/'figures'/folder
        p.mkdir(parents=True)
        (p/'original.png').write_bytes(b'original')
    return run


def test_regeneration_preserves_source_and_supplementary(tmp_path):
    run = tiny_run(tmp_path)
    groups = ConditionsCfg(groups={'WT':['a','b'], 'KO':['c','d']})
    before = {p.relative_to(run): p.read_bytes() for p in run.rglob('*') if p.is_file()}
    out = tmp_path/'new'
    result = regenerate(run, groups, out)
    assert (out/'figures/per_well_supplementary/per_condition/original.png').read_bytes() == b'original'
    assert (out/'figures/per_image/original.png').read_bytes() == b'original'
    assert (out/'figures/00_overview/combined_output_panel_focus.png').is_file()
    assert 'Per condition' in pd.ExcelFile(out/'analysis_summary.xlsx').sheet_names
    assert set(result['conditions'].condition) == {'WT','KO','Secondary-only'}
    assert not result['contrasts'].test_group.eq('Secondary-only').any()
    assert before == {p.relative_to(run): p.read_bytes() for p in run.rglob('*') if p.is_file()}


@pytest.mark.parametrize('target', ['', 'figures', 'figures/per_condition'])
def test_refuses_original_destinations(tmp_path, target):
    run = tiny_run(tmp_path)
    with pytest.raises(ValueError, match='original|new'):
        regenerate(run, ConditionsCfg(groups={'WT':['a','b']}), run/target)


def test_frozen_run_requires_external_output(tmp_path):
    run = tiny_run(tmp_path)
    (run/'MANIFEST_SHA256.tsv').write_text('frozen')
    with pytest.raises(ValueError, match='frozen'):
        regenerate(run, ConditionsCfg(groups={'WT':['a','b']}))


def test_native_hook_archives_well_output_and_legacy_is_noop(tmp_path, monkeypatch):
    from fishsuite.core import native_by_condition as native
    run = tiny_run(tmp_path)
    before = {p.relative_to(run):p.read_bytes() for p in run.rglob('*') if p.is_file()}
    assert native.finalize_native(run) is None
    assert before == {p.relative_to(run):p.read_bytes() for p in run.rglob('*') if p.is_file()}
    (run/'run_config.json').write_text(json.dumps({'config_resolved':{
        'conditions':{'groups':{'WT':['a','b'],'KO':['c','d']}}}}))
    seen = []
    monkeypatch.setattr(native,'_render',lambda source,out,groups: seen.append((source,out,groups.groups)))
    native.finalize_native(run)
    assert (run/'figures/per_well_supplementary/00_overview/original.png').is_file()
    assert (run/'figures/per_image/original.png').is_file()
    assert not (run/'figures/per_condition').exists()
    assert seen == [(run,run,{'WT':['a','b'],'KO':['c','d']})]
    # Simulate a resumed native downstream pass producing another well panel.
    (run/'figures/per_condition').mkdir()
    (run/'figures/per_condition/new.png').write_bytes(b'resumed')
    native.finalize_native(run)
    assert (run/'figures/per_well_supplementary/per_condition/original.png').read_bytes() == b'original'
    assert list((run/'figures/per_well_supplementary').glob('generation_*/per_condition/new.png'))


def test_cli_exposes_native_figures_and_refuses_overwrite(tmp_path):
    from click.testing import CliRunner
    from fishsuite.cli import cli
    run = tiny_run(tmp_path)
    groups = tmp_path/'groups.yaml'
    groups.write_text('groups: {WT: [a, b], KO: [c, d]}')
    result = CliRunner().invoke(cli,['native-figures','--run',str(run),'--groups',str(groups),'--out',str(run/'figures')])
    assert result.exit_code != 0
    assert 'refusing original' in result.output


def test_fraction_axis_uses_run_channel_label(tmp_path):
    from fishsuite.report import figures
    ctx = figures.FigureContext(tmp_path,{},None,['NT','KD'],'NT',.05,{},
        {'rna1':'MIAT'},plot_style='replicate-simple')
    wells = pd.DataFrame(dict(group=['NT','KD'],well_id=['a','b'],
        endpoint=['rna1_nuclear_spot_fraction']*2,well_mean_of_field_values=[.5,.8]))
    canvas, ax = figures.plt.subplots()
    figures.draw_plot(ax,ctx,'rna1_nuclear_spot_fraction',wells,pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),'fraction',None)
    assert ax.get_ylabel() == 'MIAT puncta: % nuclear (per nucleus)'
    figures.plt.close(canvas)
