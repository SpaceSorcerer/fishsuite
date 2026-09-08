"""Assemble a report from a completed run, same-cohort panel and verified DAPI cache.

No segmentation, detection or spatial-null computation is performed here.
"""
from pathlib import Path
import argparse
import json
import shutil
from dataclasses import asdict
import pandas as pd
import yaml
from PIL import Image
from . import aggregate, endpoints, build
from .provenance import sha256

DECISION = ('Mixed-model headline decision 2026-09-07 after data inspection; '
            'Welch on well means remains reported. Nucleus-level linear mixed model: '
            'fixed arm, random well and FOV within well, REML, two-sided Wald-normal p. '
            'Field-only or failed mixed fits use Welch (well means) on the bracket. '
            'Nonconverged fits remain missing and are explicitly labeled. '
            'p_mixed and p_mixed_holm contain mixed inference only; p_headline and '
            'p_headline_holm include the declared Welch fallbacks. Holm uses the '
            'headline family; p_welch and its separate Holm values remain available. '
            'significant_holm_0p05 retains legacy Welch Holm; significant_headline_holm_0p05 reports headline Holm. '
            'Absolute intensity and declared descriptive endpoints remain descriptive. '
            'Well dots, means and SD retain equal FOV weighting. Hedges g and MDE '
            'describe the well-means comparison, not mixed-model power.')


def narrative(spec, contrasts=None):
    bodies=dict(
        overview='WT and QKI-KO are compared using the recorded nucleus cohort. Signal intensity, localization and punctum pairing answer distinct questions.',
        q1_total_if='Matched-secondary-corrected nuclear intensity remains descriptive. Nuclear-to-cytoplasmic ratio and puncta counts are shown separately; staining batch remains a caveat.',
        q2='Continuous RNASEH2B signal, relative enrichment and threshold calls measure different properties and are reported separately.',
        nucleus_selection='The census counts DAPI objects with no retained-mask overlap. It does not reconstruct the reason each object lacked a retained mask.',
        methods_provenance='Nucleus-level mixed models test arm with random well and nested field effects. Welch on well means remains reported. The headline choice was made after data inspection; Holm adjusts the declared headline families.',
        micrographs='Persisted fields show composite, paired channels, RNASEH2B and BIN1. Images document the recorded display and do not determine the statistical comparisons.',
        standard_coloc='The standard colocalization panel uses the same retained nuclei as the current run. Pixel measures are descriptive; line profiles are recorded examples.')
    selected=dict(q1_count_size=('rna1_spots_per_nucleus','BIN1 intron puncta per nucleus'),
                  q1_localization=('nuclear_spot_fraction_dapi','DAPI-corrected BIN1 nuclear fraction'),
                  q3=('paired_frac_rna1_at_partner_minus_shuffle','BIN1-anchored pairing excess'),
                  q4_reverse_anchor=('paired_frac_partner_at_rna1_minus_shuffle','RNASEH2B-anchored pairing excess'))
    for item in spec['slides']:
        identity=item['identity']
        body=bodies.get(identity,'Well means and the nucleus-level mixed-model comparison are shown for this endpoint.')
        if contrasts is not None and identity in selected:
            endpoint,label=selected[identity]
            row=contrasts.loc[contrasts.endpoint.eq(endpoint)].iloc[0]
            direction='higher' if row['diff']>0 else 'lower' if row['diff']<0 else 'unchanged'
            body=f'{label} is {direction} in KO. '
            if row.sensitivity_mixed_status=='ok':
                body+='The raw mixed-model comparison '+('detects' if row.p_mixed<.05 else 'does not detect')+' a difference; family adjustment remains in the workbook and notes.'
            else:
                body+='The mixed fit is unavailable; the bracket explicitly reports Welch on well means.'
        item['body']=body
    return spec


def assemble(run, panel, prior, spec_path, out):
    from . import workbook, slides
    out.mkdir(parents=True, exist_ok=True)
    (out/'data').mkdir(exist_ok=True)
    shutil.copytree(prior/'dapi', out/'dapi', dirs_exist_ok=True)
    shutil.copytree(prior/'micrographs', out/'micrographs', dirs_exist_ok=True)
    shutil.copyfile(prior/'SOURCE_RUN.md', out/'SOURCE_RUN.md')
    correction = out/'data/RNASEH2B_TOTAL_CORRECTED.csv'
    shutil.copyfile(prior/'data/RNASEH2B_TOTAL_CORRECTED.csv', correction)
    spec = yaml.safe_load(spec_path.read_text(encoding='utf-8'))
    spec.update(current_cohort_narrative=True, existing_coloc=str(panel), cohort=run.name,
                secondary_corrected_csv=str(correction), micrographs=str(out/'micrographs'),
                figure_index_sha256=sha256(panel.with_name('FIGURE_INDEX.md')))
    titles = dict(q1_count_size='Q1 · BIN1 intron puncta per nucleus',
                  q1_localization='Q1 · BIN1 intron localization',
                  q1_total_if='Q1 · RNASEH2B level',
                  q2='Q2 · RNASEH2B signal at nuclear BIN1 puncta',
                  q3='Q3 · RNASEH2B puncta at nuclear BIN1 puncta',
                  q4_reverse_anchor='Q4 · BIN1 at RNASEH2B puncta (reverse anchor)',
                  nucleus_selection='Nucleus selection and size floor')
    for item in spec['slides']:
        item['question'] = item['title']
        item['title'] = titles.get(item['identity'], item['title'])
        # Historical narrative claims must not be carried into a new cohort.
        item['body'] = 'Measurements and comparisons are recorded in the workbook.'
    overlays=pd.read_excel(panel,sheet_name='overlays_index')
    for group in ['WT','QKI-KO']:
        row=overlays.loc[overlays.line.eq(group)].sort_values('well_id').iloc[0]
        for image in (run/'publication_images').glob(row.stem+'__*.png'):
            shutil.copyfile(image,out/'micrographs'/image.name)
        spec['micrograph_bar_um'][row.stem+'__merge_all.png']=json.loads((run/'run_config.json').read_text(encoding='utf-8'))['config_resolved']['output']['scalebar_um']
    spec['micrograph_hashes'] = {f.name:sha256(f) for f in (out/'micrographs').glob('*__merge_all.png')}
    selection = out/'data/NUCLEUS_SELECTION.md'
    selection.write_text('Retained labels from '+str(run)+'. Minimum nucleus area: 4000 pixels. '
                         'DAPI census counts zero-overlap components; rejection reasons are not reconstructed.\n')
    spec.update(nucleus_selection=str(selection), nucleus_selection_sha256=sha256(selection))
    groups=['WT=WT_1,WT_2,WT_3','QKI-KO=KO_1,KO_2,KO_3']
    data=aggregate.load_run(run, {'WT_1':'WT','WT_2':'WT','WT_3':'WT','KO_1':'QKI-KO','KO_2':'QKI-KO','KO_3':'QKI-KO'}, {})
    eps,_=endpoints.resolve(data['nuclei'],data['per_image'])
    manifest={'named_files':[dict(path=str(f),sha256=sha256(f)) for f in
              [panel,*[run/n for n in ['nuclei_metrics.csv','spot_metrics.csv','per_image_summary.csv','run_config.json','versions.txt']]]],
              'endpoint_registry':[asdict(e) for e in eps], 'persisted_masks':[]}
    provenance=json.loads((out/'dapi/dapi_provenance.json').read_text(encoding='utf-8'))
    for row in provenance['sources']:
        path=Path(row['mask_path'])
        with Image.open(path) as im: shape=[im.height,im.width]
        manifest['persisted_masks'].append(dict(image=row['image'],path=str(path),sha256=sha256(path),shape=shape))
    pinned=out/'data/source_manifest.json'
    pinned.write_text(json.dumps(manifest,indent=2))
    spec['baseline_manifest']=str(pinned)
    resolved_input=out/'data/deck_spec.yaml'
    resolved_input.write_text(yaml.safe_dump(spec,sort_keys=False),encoding='utf-8')
    result=build.build_report(run,out,groups=groups,reference='WT',plot_style='replicate-simple',
                              existing_coloc=panel,baseline_manifest=pinned,deck_spec=resolved_input,
                              deck=True,coloc_panel=False)
    result['field'].to_csv(out/'per_field.csv',index=False)
    # Retain the source declaration including both source-run junction destinations.
    shutil.copyfile(prior/'SOURCE_RUN.md',out/'SOURCE_RUN.md')
    finalize(out,prior,run,panel)


def finalize(out,prior,run,panel):
    """Add floor comparison, amend documentation, validate deck, seal last."""
    import openpyxl
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from . import figures, slides
    from .dapi_delivery import preview_deck
    book=openpyxl.load_workbook(out/'REPORT.xlsx')
    contrasts=pd.read_csv(out/'contrasts.csv')
    old=pd.read_csv(prior/'nucleus_floor_comparison.csv')
    old=old.loc[old.floor_px.eq(16000)].copy()
    new=contrasts.copy().assign(floor_px=4000)
    new['mixed_p']=new.p_mixed
    new['mixed_status']=new.sensitivity_mixed_status
    comp=pd.concat([old,new],ignore_index=True)
    comp.to_csv(out/'nucleus_floor_comparison.csv',index=False)
    if 'Nucleus floor comparison' in book:
        del book['Nucleus floor comparison']
    ws=book.create_sheet('Nucleus floor comparison')
    ws.append(['Each floor retains its own cohort. Old-floor values copied from prior comparison; new-floor values rebuilt from the same-floor panel.'])
    ws.append(comp.columns.tolist())
    for row in comp.itertuples(index=False,name=None):
        ws.append([None if pd.isna(v) else v for v in row])
    changes=[]
    for endpoint,rows in comp.groupby('endpoint'):
        if set(rows.floor_px)!={4000,16000} or len(rows)!=2:
            continue
        a=rows.loc[rows.floor_px.eq(16000)].iloc[0]
        b=rows.loc[rows.floor_px.eq(4000)].iloc[0]
        for column in ['p_welch','mixed_p']:
            if pd.notna(a[column]) and pd.notna(b[column]) and (a[column]<.05)!=(b[column]<.05):
                changes.append(dict(endpoint=endpoint,test=column,old_p=a[column],new_p=b[column]))
    changed=pd.DataFrame(changes,columns=['endpoint','test','old_p','new_p'])
    changed.to_csv(out/'floor_conclusion_changes.csv',index=False)
    if 'Floor conclusion changes' in book:
        del book['Floor conclusion changes']
    change_sheet=book.create_sheet('Floor conclusion changes')
    change_sheet.append(['Raw p threshold transitions across floors; descriptive comparison, not an equivalence test.'])
    change_sheet.append(changed.columns.tolist())
    for row in changed.itertuples(index=False,name=None): change_sheet.append(row)
    for ws in book:
        if ws.title in ['Contrasts','Read me','Multiplicity plan'] or 'by group' in ws.title:
            ws.cell(1,1).value=DECISION+' Thresholds, filters, Holm, effect sizes and MDE are recorded in endpoint/provenance rows.'
    book.save(out/'REPORT.xlsx')
    # Workbook bytes changed: revalidate typed cells and rebuild using the new hash.
    spec=yaml.safe_load((out/'deck_spec.resolved.yaml').read_text(encoding='utf-8'))
    spec['workbook_sha256']=sha256(out/'REPORT.xlsx')
    (out/'deck_spec.resolved.yaml').write_text(yaml.safe_dump(spec,sort_keys=False),encoding='utf-8')
    slides.build_deck(out/'REPORT.xlsx',spec,out/'Sam_RNASEH2B_BIN1.pptx')
    ppt=Presentation(out/'Sam_RNASEH2B_BIN1.pptx')
    slide=ppt.slides.add_slide(ppt.slide_layouts[6])
    box=slide.shapes.add_textbox(Inches(.4), Inches(.2), Inches(12.4), Inches(.8))
    box.text_frame.text='Nucleus selection and size floor'
    box.text_frame.paragraphs[0].font.size=Pt(23)
    focus=comp.loc[comp.endpoint.isin(['rna1_spots_per_nucleus','nuclear_spot_fraction_dapi'])]
    lines=[]
    for row in focus.itertuples():
        scale=100 if row.endpoint=='nuclear_spot_fraction_dapi' else 1
        lines.append(f'{row.endpoint}, floor {row.floor_px:g}: WT {row.mean_ref*scale:.4g}; KO {row.mean_test*scale:.4g}; mixed p {row.mixed_p:.4g}; Welch p {row.p_welch:.4g}')
    body=slide.shapes.add_textbox(Inches(.5), Inches(1.4), Inches(12), Inches(5))
    body.text_frame.text='\n\n'.join(lines)
    for para in body.text_frame.paragraphs: para.font.size=Pt(18)
    slide.notes_slide.notes_text_frame.text='Workbook: '+str(out/'REPORT.xlsx')+'\nNucleus selection and size floor\n'+DECISION+'\nSource: Nucleus floor comparison sheet.\n'+'\n'.join(lines)
    source_rows=pd.read_csv(out/'slide_sources.csv').to_dict('records')
    for index,row in focus.iterrows():
        for column in ['endpoint','floor_px','mean_ref','mean_test','mixed_p','p_welch']:
            source_rows.append(dict(slide=len(ppt.slides),workbook=str(out/'REPORT.xlsx'),
                sheet='Nucleus floor comparison',cell=f'{openpyxl.utils.get_column_letter(comp.columns.get_loc(column)+1)}{index+3}',value=row[column]))
    pd.DataFrame(source_rows).to_csv(out/'slide_sources.csv',index=False)
    ppt.save(out/'Sam_RNASEH2B_BIN1.pptx')
    # Both focus and full-scale deck variants share the same inference and notes.
    full=Presentation(out/'Sam_RNASEH2B_BIN1.pptx')
    from io import BytesIO
    replacements={Path(a['path']).read_bytes():Path(a['path']).with_name(Path(a['path']).name.replace('_focus.png','_full.png'))
                  for sl in spec['slides'] for a in sl.get('figures',[]) if '_focus.png' in a['path']}
    for sl in full.slides:
        for shape in list(sl.shapes):
            if shape.shape_type==13 and shape.image.blob in replacements:
                replacement=replacements[shape.image.blob]
                if replacement.is_file():
                    sl.shapes.add_picture(str(replacement),shape.left,shape.top,shape.width,shape.height)
                    shape._element.getparent().remove(shape._element)
    full.save(out/'Sam_RNASEH2B_BIN1_full.pptx')
    for name in ['METHODS.md','FAMILY_AMENDMENTS.md']:
        text=(out/name).read_text(encoding='utf-8') if (out/name).is_file() else ''
        while text.startswith(DECISION):
            text=text[len(DECISION):].lstrip()
        text=text.replace('two-sided Welch\ntests use three WT and three KO wells','mixed models provide the headline; Welch\ntests remain reported for three WT and three KO wells')
        (out/name).write_text(DECISION+'\n\n'+text,encoding='utf-8')
    lineage=f'Run: {run}\nPanel: {panel}\nPrior floor comparison: {prior}/nucleus_floor_comparison.csv\n'
    shutil.copyfile(panel,out/'data/coloc_standard_panel_floor4000.xlsx')
    (out/'README.md').write_text('v5: floor-4000 report and same-floor standard colocalization panel.\nStart with REPORT.xlsx and Sam_RNASEH2B_BIN1.pptx; full-scale variant: Sam_RNASEH2B_BIN1_full.pptx.\n'+DECISION+'\n'+lineage,encoding='utf-8')
    (out/'READOUT.md').write_text(DECISION+'\n\n'+'\n'.join(lines)+'\n'+lineage,encoding='utf-8')
    (out/'versions.txt').write_text((run/'versions.txt').read_text(encoding='utf-8')+'\nreporter: uncommitted worktree, Claude commits\n'+lineage,encoding='utf-8')
    import sys
    command="& '"+sys.executable+"' -m fishsuite.report.final_delivery "+' '.join("'"+v.replace("'","''")+"'" for v in sys.argv[1:])
    (out/'BUILD_COMMANDS.md').write_text("Reporter module: fishsuite.report.final_delivery. No one-off delivery scripts.\n\n"
        "$env:PYTHONPATH='"+str(Path(__file__).resolve().parents[2])+"'\n"+command+'\n\n'
        'Initial build omits --refresh. --refresh reuses fitted workbook results for presentation-only changes. '
        'Run --seal only after visual review and all files are final.\n',encoding='utf-8')
    preview_deck(out)
    for sl in ppt.slides:
        assert sl.notes_slide.notes_text_frame.text.startswith('Workbook: ')
    print(f'VERIFIED: {len(ppt.slides)} focus slides and {len(full.slides)} full slides open; notes start with workbook paths.',flush=True)
    print('FINAL DELIVERY: '+str(out),flush=True)


def seal(out, run, panel):
    """Write the report lock last; source junctions are recorded without traversal."""
    import os
    files=[]
    for root, dirs, names in os.walk(out,followlinks=False):
        dirs[:]=[d for d in dirs if not (getattr((Path(root)/d).lstat(),'st_file_attributes',0) & 0x400)]
        for name in names:
            path=Path(root)/name
            if name!='REPORT_LOCK.json':
                files.append(dict(path=str(path.resolve()),sha256=sha256(path)))
    code=Path(__file__).parent
    record=dict(reporter='uncommitted worktree, Claude commits',run=str(run),panel=str(panel),
                source_panel_sha256=sha256(panel),files=files,
                reporter_files=[dict(path=str(p),sha256=sha256(p)) for p in sorted(code.glob('*.py'))])
    (out/'REPORT_LOCK.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(f'SEALED: {len(files)} delivery files; REPORT_LOCK.json written last.',flush=True)


def refresh(out, prior, run, panel):
    """Re-render persisted statistics after presentation edits, without refitting."""
    from dataclasses import replace
    from . import figures, slides, workbook, dapi_mask
    from .coloc_existing import load_existing_panel, integrate_panel, render_existing
    from .stats import holm
    sheets=pd.read_excel(out/'REPORT.xlsx',sheet_name=None,header=1)
    spec=yaml.safe_load((out/'data/deck_spec.yaml').read_text(encoding='utf-8'))
    data=aggregate.load_run(run,{'WT_1':'WT','WT_2':'WT','WT_3':'WT','KO_1':'QKI-KO','KO_2':'QKI-KO','KO_3':'QKI-KO'}, {})
    pn=sheets['Per nucleus']
    cols=['image','nucleus_id']+[c for c in pn if c not in data['nuclei']]
    data['nuclei']=data['nuclei'].merge(pn[cols],on=['image','nucleus_id'],validate='one_to_one')
    data['dapi']=dapi_mask.build_dapi(run,out/'dapi',data)
    eps,absent=endpoints.resolve(data['nuclei'],data['per_image'])
    persisted=load_existing_panel(panel,panel.with_name('FIGURE_INDEX.md'),Path(spec['baseline_manifest']))
    eps,_=integrate_panel(data,persisted,eps)
    eps=[replace(e,descriptive_only=True,excluded_from_holm='legacy_mask_only') if e.name in
         {'rna1_cyto_spots_per_nucleus','rna1_nuclear_spot_fraction'} else e for e in eps]+dapi_mask.endpoints()
    eps.append(endpoints.Endpoint('protein_nuclear_mean_seconly_corrected','protein_nuclear_mean_seconly_corrected',
        'detection','AU','{protein} nuclear mean, matched secondary subtracted',descriptive_only=True,
        absolute_intensity=True,source=str(out/'data/RNASEH2B_TOTAL_CORRECTED.csv')))
    contrasts=sheets['Contrasts']
    narrative(spec, contrasts)
    for _,idx in contrasts.groupby(['family','test_group']).groups.items():
        member=[i for i in idx if contrasts.at[i,'in_holm_family']]
        adj=holm(contrasts.loc[member,'p_headline'].tolist())
        contrasts.loc[member,'p_headline_holm']=adj
        contrasts.loc[member,'p_mixed_holm']=adj
    contrasts.loc[contrasts.p_mixed.isna(),'p_mixed_holm']=float('nan')
    contrasts['significant_holm_0p05']=contrasts.p_welch_holm_within_family.lt(.05).where(contrasts.in_holm_family)
    contrasts['significant_headline_holm_0p05']=contrasts.p_headline_holm.lt(.05).where(contrasts.in_holm_family)
    contrasts.to_csv(out/'contrasts.csv',index=False)
    # Family summaries use the same rebuilt headline columns.
    for family,name in [('detection','Spots per nucleus by group'),('localization','Nuclear fraction by group'),('partner','Partner at puncta by group')]:
        sheets[name]=aggregate.family_sheet(contrasts,sheets['Per well'],family,['WT','QKI-KO'])
    for name in ['Read me','Run provenance']:
        table=sheets[name]
        for col in table:
            table[col]=table[col].map(lambda v: DECISION if isinstance(v,str) and ('RAW Welch' in v or 'Welch t on well means at alpha' in v) else v)
    ctx=figures.FigureContext(run,data['cfg'],data.get('thresholds'),['WT','QKI-KO'],'WT',.05,{},endpoints.channel_labels(data['cfg']),plot_style='replicate-simple')
    dapi_mask.render_dapi(ctx,data['dapi'],data['nuclei'],sheets['Per well'],sheets['Per field'],contrasts,out/'localization')
    render_existing(persisted,ctx,contrasts,out/'coloc_existing')
    resolved=slides.prepare_deck(spec,sheets,out,data,persisted,ctx,sheets['Per well'],sheets['Per field'],contrasts,eps)
    build.render_figures(out/'figures',run,data,eps,absent,sheets['Per well'],sheets['Per field'],pn,contrasts,
                         ['WT','QKI-KO'],'WT',{},ctx.channel_labels,.05,plot_style='replicate-simple')
    workbook.write(out/'REPORT.xlsx',sheets,{},order=list(sheets))
    resolved['workbook_sha256']=sha256(out/'REPORT.xlsx')
    (out/'deck_spec.resolved.yaml').write_text(yaml.safe_dump(resolved,sort_keys=False),encoding='utf-8')
    slides.build_deck(out/'REPORT.xlsx',resolved,out/'Sam_RNASEH2B_BIN1.pptx')
    finalize(out,prior,run,panel)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh',action='store_true',help='Reuse workbook statistics and refresh figures/decks')
    parser.add_argument('--seal',action='store_true',help='Write lock after final review')
    for key in ['run','panel','prior','spec','out']: parser.add_argument('--'+key,required=True,type=Path)
    args=parser.parse_args()
    if args.seal:
        seal(args.out,args.run,args.panel)
    elif args.refresh:
        refresh(args.out,args.prior,args.run,args.panel)
    else:
        assemble(args.run,args.panel,args.prior,args.spec,args.out)
