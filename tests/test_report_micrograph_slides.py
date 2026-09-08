"""Per-well native publication panels: selection and slide-object contract."""
import pandas as pd
import pytest

from fishsuite.report.micrograph_slides import select_per_well_fovs, pair_wells, prepare_per_well_micrographs


def test_fov_closest_to_well_median_and_deterministic_tie():
    fields = pd.DataFrame([
        dict(condition='WT_1', image='c.vsi', nuclei_analyzed=30),
        dict(condition='WT_1', image='b.vsi', nuclei_analyzed=12),
        dict(condition='WT_1', image='a.vsi', nuclei_analyzed=10),
        dict(condition='WT_1', image='d.vsi', nuclei_analyzed=2),
        dict(condition='KO_1', image='k.vsi', nuclei_analyzed=15),
        dict(condition='Sec-Only', image='sec.vsi', nuclei_analyzed=99),
    ])
    chosen = select_per_well_fovs(fields)
    assert chosen.set_index('well_id').loc['WT_1', 'image'] == 'a.vsi'
    assert chosen.set_index('well_id').loc['WT_1', 'well_median_nuclei'] == 11
    assert len(chosen) == 2
    assert len(pair_wells(chosen)) == 1


def test_missing_partner_well_is_explicit():
    fields = pd.DataFrame([dict(condition='WT_1', image='a.vsi', nuclei_analyzed=12)])
    with pytest.raises(ValueError, match='unpaired'):
        pair_wells(select_per_well_fovs(fields))


def test_four_independent_panels_and_no_dapi_only(tmp_path):
    import json
    from PIL import Image, ImageDraw
    run, out = tmp_path/'run', tmp_path/'out'
    (run/'publication_images').mkdir(parents=True)
    cfg = dict(channels=dict(analysis_mode='rna_protein', rna_label='BIN1 introns',
                            antibody_label='RNASEH2B',rna_lut='yellow',antibody_lut='magenta',dapi_lut='blue'),
               output={f'manual_{role}_{bound}': value for role in ('dapi','rna','antibody')
                       for bound,value in [('min',1),('max',100)]})
    (run/'run_config.json').write_text(json.dumps({'config_resolved':cfg}))
    fields = []
    for well in ['WT_1','KO_1']:
        fields.append(dict(condition=well,image='prefix_'+well+'.vsi',nuclei_analyzed=10,voxel_xy_nm=1000))
        for suffix in ['merge_all','merge_BIN1_introns_RNASEH2B','RNASEH2B_magenta','BIN1_introns_yellow']:
            im = Image.new('RGB',(200,200)); ImageDraw.Draw(im).line((165,185,184,185),fill='white',width=2)
            im.save(run/'publication_images'/f'{well}__{well}__{suffix}.png')
    pd.DataFrame(fields).to_csv(run/'per_image_summary.csv',index=False)
    result = prepare_per_well_micrographs(run,out)
    rows = result['slides'][0]['rows']
    assert len(rows) == 2
    assert all(len(row['panels']) == 4 for row in rows)
    paths = [p['path'] for row in rows for p in row['panels']]
    assert len(set(paths)) == 8
    assert all('DAPI_blue' not in p for p in paths)
    assert rows[0]['panels'][0]['bar_um'] == 20
    assert rows[0]['panels'][2]['panel'] == 'RNASEH2B'
