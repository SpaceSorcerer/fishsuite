import json
from types import SimpleNamespace
from pathlib import Path

import pandas as pd
import pytest

from fishsuite.config.schema import ConditionsCfg
from fishsuite.config.hierarchy import (resolve_hierarchy, group_mapping,
    discovery_roster, annotate_frame, set_result_identity)
from fishsuite.core.io import DiscoveredImage
from fishsuite.core.native_by_condition import _load, collapse_conditions
from fishsuite.report.aggregate import label_frame, per_field_long, per_well_long
from fishsuite.report.endpoints import Endpoint

PATTERN = r'_((?:WT|KO|Sec-Only)-\d+)_'


def images():
    return pd.DataFrame(dict(image=['p_WT-1_00.vsi','p_WT-2_00.vsi','p_KO-1_00.vsi','p_Sec-Only-2_00.vsi'],
        condition=['sourceA','sourceA','sourceB','control'], secondary_only=[False,False,False,True]))


def test_schema_and_well_first_report_assignment():
    cfg = ConditionsCfg(groups={'WT':['WT-1','WT-2'],'KO':['KO-1']}, well_from_image=PATTERN)
    assert cfg.model_dump()['well_from_image'] == PATTERN
    rows = label_frame(images(), group_mapping(cfg.groups), {}, cfg.well_from_image)
    assert list(rows.group) == ['WT','WT','KO','Secondary-only']
    assert list(rows.well_id) == ['WT-1','WT-2','KO-1','Sec-Only-2']
    assert list(rows.condition) == list(images().condition)


def test_line_label_fallback_and_string_control_flags():
    frame = images().iloc[:3].copy()
    frame.condition = ['WT','WT','KO']
    frame.secondary_only = 'False'
    rows = resolve_hierarchy(frame, {'WT':'WT','KO':'KO'}, PATTERN, strict=True)
    assert list(rows.group) == ['WT','WT','KO']
    assert not rows.secondary_only.any()


@pytest.mark.parametrize('mapping,pattern,error', [
    ({'WT-1':'WT','WT-2':'WT'}, PATTERN, 'Unassigned'),
    ({'WT-1':'WT','WT-2':'WT','KO-1':'KO','missing':'WT'}, PATTERN, 'missing'),
    ({'WT-1':'WT','sourceA':'KO'}, PATTERN, 'Conflicting'),
    ({}, r'(unmatched)', 'matched no well'),
    ({}, r'(a)(b)', 'exactly one'),
])
def test_fail_closed(mapping, pattern, error):
    with pytest.raises(ValueError, match=error):
        resolve_hierarchy(images(), mapping, pattern, strict=True)


def test_duplicates_and_controls_cannot_be_biological_members():
    with pytest.raises(ValueError, match='both'):
        group_mapping({'WT':['a','a']})
    with pytest.raises(ValueError, match='Duplicate image'):
        resolve_hierarchy(pd.concat([images(), images()]))
    with pytest.raises(ValueError, match='missing'):
        resolve_hierarchy(images(), {'WT-1':'WT','WT-2':'WT','KO-1':'KO','Sec-Only-2':'WT'}, PATTERN, strict=True)


def test_duplicate_basename_roster_and_all_result_carriers(tmp_path):
    discovered = [DiscoveredImage(tmp_path/w/'same.vsi', w, False, w) for w in ['a','b']]
    cfg = ConditionsCfg(groups={'WT':['a','b']})
    roster = discovery_roster(discovered, tmp_path, cfg)
    assert list(roster.image) == ['a/same.vsi','b/same.vsi']
    assert list(roster.source_path) == [str(im.path) for im in discovered]
    tables = [pd.DataFrame({'image':['same.vsi'], 'x':[3.]}) for _ in range(4)]
    result = SimpleNamespace(image='same.vsi', per_image={'image':'same.vsi'}, thresholds={'image':'same.vsi'},
        nuclei=tables[0], spots=tables[1], morphology=tables[2], extra={'null':tables[3]})
    set_result_identity(result, roster.image.iloc[0])
    assert all(t.image.iloc[0] == 'a/same.vsi' for t in tables)
    annotated = annotate_frame(result.nuclei, roster)
    assert annotated.well_id.iloc[0] == 'a'
    assert annotated.x.iloc[0] == 3


def test_unbalanced_hierarchy_actual_production_functions():
    # a: FOV means 0 and 10 despite 1 vs 9 cells; b: one FOV mean 20.
    rows = [dict(image='a/00.vsi', condition='a', secondary_only=False, x=0)]
    rows += [dict(image='a/01.vsi', condition='a', secondary_only=False, x=10)]*9
    rows += [dict(image='b/00.vsi', condition='b', secondary_only=False, x=20)]*2
    nuc = pd.DataFrame(rows)
    image = nuc.drop_duplicates('image').drop(columns='x')
    labels = label_frame(image, {'a':'WT','b':'WT'}, {})
    nuc = nuc.drop(columns=['condition','secondary_only']).merge(labels,on='image',validate='many_to_one')
    ep = Endpoint('x','x','rna1','count','x')
    fields = per_field_long(nuc, image, [ep], labels, 0)
    wells = per_well_long(fields)
    summary = collapse_conditions(wells).iloc[0]
    assert list(wells.well_mean_of_field_values) == [5,20]
    assert summary.mean_of_well_means == 12.5
    assert summary.n_wells == 2 and summary.n_FOVs == 3 and summary.n_nuclei == 12


def test_native_filename_wells_and_presaved_ids(tmp_path):
    frame = images()
    frame['nucleus_id'] = 1
    frame['nuclear_spot_count'] = [2,6,9,0]
    frame.to_csv(tmp_path/'nuclei_metrics.csv',index=False)
    frame.drop(columns='nucleus_id').to_csv(tmp_path/'per_image_summary.csv',index=False)
    (tmp_path/'run_config.json').write_text(json.dumps({'config_resolved':{}}))
    cfg = ConditionsCfg(groups={'WT':['WT-1','WT-2'],'KO':['KO-1']}, well_from_image=PATTERN)
    _, nuclei, fields, wells, contrasts, endpoints, order = _load(tmp_path,cfg)
    assert set(wells.well_id) == {'WT-1','WT-2','KO-1','Sec-Only-2'}
    assert order == ['WT','KO','Secondary-only']
    assert contrasts.reference_group.eq('WT').all()
    assert nuclei.loc[nuclei.secondary_only,'group'].eq('Secondary-only').all()


def test_strict_unlisted_well_and_global_well_ownership():
    frame = pd.DataFrame(dict(image=['WT-9'], condition=['WT'], secondary_only=[False], well_id=['WT-9']))
    with pytest.raises(ValueError, match='Unassigned'):
        resolve_hierarchy(frame, {'WT-1':'WT'}, strict=True)
    frame = pd.DataFrame(dict(image=['a','b'], condition=['WT','KO'], secondary_only=False, well_id=['1','1']))
    with pytest.raises(ValueError, match='globally unique'):
        resolve_hierarchy(frame, {'WT':'WT','KO':'KO'}, strict=True)


def test_exact_selected_path_and_output_collision(tmp_path):
    from fishsuite.config.hierarchy import select_inputs
    from fishsuite.runner import _validate_output_stems
    discovered = [DiscoveredImage(tmp_path/w/'same.vsi', w, False, w) for w in ['a','b']]
    assert len(select_inputs(discovered,tmp_path,['a/same.vsi'])) == 1
    assert len(select_inputs(discovered,tmp_path,[str(discovered[0].path)])) == 1
    assert len(select_inputs(discovered,tmp_path,['same.vsi'])) == 2
    assert len(_validate_output_stems(discovered)) == 2
    for im in discovered:
        im.condition = 'WT'
    with pytest.raises(ValueError, match='output filename collision'):
        _validate_output_stems(discovered)


def test_collision_and_bad_regex_block_runner_before_pixels(tmp_path, monkeypatch):
    from fishsuite.config.schema import FishsuiteConfig
    from fishsuite import runner
    from fishsuite.core import repro
    monkeypatch.setattr(repro,'set_global_seeds',lambda *_: {})
    monkeypatch.setattr(repro,'write_run_metadata',lambda *_a,**_k: None)
    def no_pixels(*args, **kwargs):
        raise AssertionError('image pixels must not be read')
    monkeypatch.setattr(runner._io,'read_image',no_pixels)
    root = tmp_path/'input'
    for well in ['a','b']:
        (root/well).mkdir(parents=True)
        (root/well/'same.vsi').write_bytes(b'')
    cfg = FishsuiteConfig(conditions={'subfolder_conditions':{'a':'WT','b':'WT'}})
    config = tmp_path/'config.yaml'
    cfg.dump_yaml(config)
    with pytest.raises(ValueError, match='output filename collision'):
        runner.run_batch(config,root,tmp_path/'out',dry_run=True)
    cfg.conditions.subfolder_conditions = {'a':'a','b':'b'}
    cfg.output.pub_contrast_mode = 'reference_image'
    cfg.output.manual_rna_reference_image = 'same.vsi'
    cfg.dump_yaml(config)
    with pytest.raises(ValueError, match='Ambiguous reference-image'):
        runner.run_batch(config,root,tmp_path/'out_ref',dry_run=True)
    cfg.output.pub_contrast_mode = 'manual'
    cfg.conditions.well_from_image = '(missing)'
    cfg.dump_yaml(config)
    with pytest.raises(ValueError, match='matched no well'):
        runner.run_batch(config,root,tmp_path/'out2',dry_run=True)


def test_resumed_completion_requires_this_attempt(tmp_path):
    import hashlib
    from fishsuite.core.native_by_condition import completed_attempt
    (tmp_path/'data.csv').write_text('x')
    source = {'data.csv':hashlib.sha256(b'x').hexdigest()}
    path = tmp_path/'native_by_condition.json'
    path.write_text(json.dumps({'attempt_id':'old','sources':source}))
    assert not completed_attempt(tmp_path,'new')
    path.write_text(json.dumps({'attempt_id':'new','sources':source}))
    assert completed_attempt(tmp_path,'new')
    (tmp_path/'data.csv').write_text('changed')
    assert not completed_attempt(tmp_path,'new')


def test_source_acquisition_and_original_condition_stem(tmp_path):
    from fishsuite.report.dapi_mask import source_acquisition
    from fishsuite.report.micrograph_slides import _native_stem
    root=tmp_path/'raw'
    (root/'a').mkdir(parents=True)
    path=root/'a'/'same.vsi'
    path.write_bytes(b'')
    row=SimpleNamespace(image='a/same.vsi',condition='source A',secondary_only=False)
    assert source_acquisition(root,row) == path
    row.source_path=str(path)
    assert source_acquisition(tmp_path/'different',row) == path
    pub=tmp_path/'publication_images'
    pub.mkdir()
    (pub/'source_A__same__merge_all.png').write_bytes(b'')
    assert _native_stem(pub,row.image,row.condition) == 'source_A__same'


def test_gui_metadata_assignment_preview_and_report_reload(tmp_path, monkeypatch):
    import importlib
    import yaml
    QApplication = pytest.importorskip('PySide6.QtWidgets').QApplication
    main=importlib.import_module('fishsuite.gui.main')
    monkeypatch.setattr(main._state,'load_settings',lambda: {})
    monkeypatch.setattr(main._state,'save_settings',lambda *_: None)
    app=QApplication.instance() or QApplication([])
    window=main.FishsuiteWindow()
    root=tmp_path/'images'
    root.mkdir()
    for name in ['p_WT-1_00.vsi','p_WT-1_01.vsi','p_WT-2_00.vsi','p_KO-1_00.vsi']:
        (root/name).write_bytes(b'')
    metadata=dict(groups={'WT':['WT-1','WT-2'],'KO':['KO-1']},well_from_image=PATTERN,
        filename_conditions=[['_WT-','WT'],['_KO-','KO']],group_order=['WT','KO'],
        group_colors={'WT':'#595959','KO':'#D67AE5'})
    window._cfg['conditions'].update(metadata)
    window._cfg_to_widgets()
    window.input_edit.setText(str(root))
    saved=window._read_widgets_into_cfg()['conditions']
    assert all(saved[k] == v for k,v in metadata.items())
    assert window._preview_hierarchy()
    assert window.hierarchy_tree.topLevelItemCount() == 2
    wt=window.hierarchy_tree.topLevelItem(0)
    assert wt.text(0) == 'WT' and wt.childCount() == 2
    assert wt.child(0).childCount() == 2
    window._load_discovered_wells()
    window._apply_well_assignments()
    assert window._read_widgets_into_cfg()['conditions']['groups'] == metadata['groups']
    run=tmp_path/'run'
    run.mkdir()
    images().to_csv(run/'per_image_summary.csv',index=False)
    (run/'nuclei_metrics.csv').write_text('image,nucleus_id')
    (run/'run_config.json').write_text(json.dumps({'config_resolved':{'conditions':saved}}))
    window.report_tab.run_path.setText(str(run))
    window.report_tab.load_run()
    imported=yaml.safe_load(window.report_tab.groups_editor.toPlainText())
    assert all(imported[k] == v for k,v in metadata.items())
    window.close()
    app.processEvents()


def test_publication_lookup_uses_recorded_stem_not_suffix(tmp_path):
    from fishsuite.report.figures import merge_png_for
    pub=tmp_path/'publication_images'
    pub.mkdir()
    for stem in ['a__same','b__same']:
        (pub/(stem+'__merge_all.png')).write_bytes(b'')
    with pytest.raises(ValueError,match='Ambiguous publication'):
        merge_png_for(pub,'b/same.vsi')
    pd.DataFrame({'image':['a/same.vsi','b/same.vsi'], 'output_stem':['a__same','b__same']}).to_csv(
        tmp_path/'resolved_experiment_hierarchy.csv',index=False)
    path, stem=merge_png_for(pub,'b/same.vsi')
    assert Path(path).name == 'b__same__merge_all.png'
    assert stem == 'b__same__'
