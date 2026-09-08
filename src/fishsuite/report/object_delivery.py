"""Report-only A17 assembly: one chart per asset, several movable assets per slide.

Reads persisted measurements and inference. Never runs segmentation or nulls.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import shutil
import numpy as np
import pandas as pd
import yaml
from openpyxl.utils import get_column_letter
from . import figures, endpoints, slides, workbook
from .provenance import sha256, guard_output


def refs(sheets, names):
    result=[]
    for name in names:
        if name not in sheets: continue
        for i,row in enumerate(sheets[name].itertuples(index=False,name=None),3):
            for j,_ in enumerate(row,1):
                result.append(dict(sheet=name,cell=f'{get_column_letter(j)}{i}',
                    label=sheets[name].columns[j-1],allow_missing=True,display=False))
    return result


def asset(path, cohort, caption=''):
    path=Path(path).resolve()
    full=path.with_name(path.name.replace('_focus','_full'))
    result=dict(path=str(path),sha256=sha256(path),cohort=cohort,caption=caption)
    if full.is_file(): result.update(full_path=str(full),full_sha256=sha256(full))
    return result


def add(spec,sheets,title,paths=(),source=(),identity='',body=''):
    item=dict(identity=identity,title=title,body=body,figures=[asset(p,spec['cohort']) for p in paths],
              values=refs(sheets,source),layout='grid')
    spec['slides'].append(item)
    return item


def chart(ctx,out,stem,title,w,c,endpoint,unit='',scale=1):
    records=[]
    column=getattr(ctx,'endpoint_columns',{}).get(endpoint)
    record=figures.superplot_standalone(ctx,endpoint,title,unit,w,getattr(ctx,'report_fields',pd.DataFrame()),getattr(ctx,'report_nuclei',pd.DataFrame()),
        c,column,out/'figures',stem,records,scale=scale)
    return out/'figures'/record['png']


def finish(out,sheets,spec,book_name,deck_name,missing):
    out=guard_output(out)
    expanded=[]
    for item in spec['slides']:
        if len(item['figures'])>4 and item.get('layout')!='micrographs':
            for start in range(0,len(item['figures']),4):
                expanded.append(dict(item,figures=item['figures'][start:start+4],title=item['title']+(' · continued' if start else '')))
        else:expanded.append(item)
    spec['slides']=expanded
    import xml.etree.ElementTree as ET
    for item in spec['slides']:
        if item.get('layout')=='micrographs':continue
        for a in item['figures']:
            svg=Path(a['path']).with_suffix('.svg')
            if not a.get('caption') and svg.is_file():
                texts=[t for t in ET.parse(svg).iter() if t.tag.endswith('}text') and 'font-size: 11px' in t.attrib.get('style','')]
                if texts:a['caption']=''.join(texts[0].itertext())
    a20_index = None
    if spec.get('a20'):
        from .a20_delivery import organize, export_figure_workbook, remap_local_assets
        captured,a20_index = organize(out,spec,'MIAT' in book_name)
        remap_local_assets(out,sheets,spec)
        export_figure_workbook(out,sheets,spec,captured)
    index=[]
    for i,item in enumerate(spec['slides'],1):
        for a in item['figures']:
            index.append(dict(slide=i,title=item['title'],**a))
    sheets['A17 Figure sources']=pd.DataFrame(index)
    sheets['A17 Slide content']=pd.DataFrame([dict(slide=i,title=s['title'],body=s.get('body',''))
                                            for i,s in enumerate(spec['slides'],1)])
    book=workbook.write(out/book_name,sheets,{},order=list(sheets))
    spec['workbook_sha256']=sha256(book)
    (out/'deck_spec.resolved.yaml').write_text(yaml.safe_dump(spec,sort_keys=False),encoding='utf-8')
    slides.build_deck(book,spec,out/deck_name)
    focus_sources=(out/'slide_sources.csv').read_bytes()
    full=json.loads(json.dumps(spec))
    for s in full['slides']:
        for a in s['figures']:
            if a.get('full_path'): a.update(path=a['full_path'],sha256=a['full_sha256'])
    slides.build_deck(book,full,out/(Path(deck_name).stem+'_full.pptx'))
    # Focus deck source manifest remains the primary index.
    (out/'slide_sources.csv').write_bytes(focus_sources)
    (out/'slides_focus.txt').write_text('\n'.join(f'{i}. {s["title"]}' for i,s in enumerate(spec['slides'],1)),encoding='utf-8')
    (out/'slides_full.txt').write_text((out/'slides_focus.txt').read_text(encoding='utf-8'),encoding='utf-8')
    (out/'MISSING.md').write_text('\n'.join(missing)+'\n',encoding='utf-8')
    (out/'TITLE_MAP.json').write_text(json.dumps({'endpoint_registry':endpoints.SHORT_TITLES,'figure_titles':{Path(a['path']).stem:a.get('caption','') for item in spec['slides'] for a in item['figures']}},indent=2),encoding='utf-8')
    if a20_index is None:
        (out/'FIGURE_INDEX.md').write_text('\n'.join(f"- Slide {r['slide']}: {r['path']}" for r in index)+'\n',encoding='utf-8')
    print(f'BUILT {out}: {len(spec["slides"])} slides, {len(index)} independently embedded images.',flush=True)
    return spec


def rnase(run,prior,out,a20=False):
    from . import aggregate, dapi_mask
    from .coloc_existing import render_simple_coloc, render_cytofluorogram
    from .micrograph_slides import prepare_per_well_micrographs
    from .rnaseh2b_analysis import build_analysis
    out.mkdir(parents=True,exist_ok=True)
    (out/'figures').mkdir(exist_ok=True)
    sheets=pd.read_excel(prior/'REPORT.xlsx',sheet_name=None,header=1)
    if a20:
        from .a20_delivery import amend_rnase
        definitions = amend_rnase(sheets,run,out)
    cfg=json.loads((run/'run_config.json').read_text(encoding='utf-8'))
    labels=endpoints.channel_labels(cfg)
    ctx=figures.FigureContext(run,cfg,None,['WT','QKI-KO'],'WT',.05,{},labels,plot_style='replicate-simple')
    w,c,n=sheets['Per well'],sheets['Contrasts'],sheets['Per nucleus']
    ctx.report_fields=sheets['Per field']
    ctx.report_nuclei=n
    ctx.endpoint_columns=dict(zip(sheets['Endpoint definitions'].name,sheets['Endpoint definitions'].column))
    if a20:
        figures.prepare_axis_groups(ctx,definitions,w,ctx.report_fields)
    epdefs=sheets['Endpoint definitions'].set_index('name')
    template=yaml.safe_load((prior/'data/deck_spec.yaml').read_text(encoding='utf-8'))
    spec=dict(cohort=run.name,slides=[],a20=a20)
    titles=dict(q1_count_size='Q1 · BIN1 intron puncta',q1_localization='Q1 · BIN1 intron localization',
        q1_total_if='Q1b · RNASEH2B level',q2='Q2 · RNASEH2B signal at nuclear BIN1 puncta',
        q3='Q3 · RNASEH2B puncta at nuclear BIN1 puncta',q4_reverse_anchor='Q4 · BIN1 at RNASEH2B puncta')
    figures.set_style()
    for item in template['slides']:
        if item['identity']=='micrographs': continue
        if a20 and item['identity']=='standard_coloc': continue
        paths=[]
        names=list(item.get('endpoints',[]))
        if a20:
            if item['identity']=='q1_total_if':
                names=['protein_nuclear_mean','nuclear_total_intensity_protein','cell_total_intensity_protein','nuclear_above_floor_intensity_protein','protein_nc_ratio','protein_spots_per_nucleus','cell_total_intensity_rna1','nuclear_total_intensity_rna1','rna1_nuclear_above_floor_intensity']
            for name in list(names):
                corrected=name+'_seconly_corrected'
                if corrected in epdefs.index and corrected not in names:
                    names.insert(names.index(name)+1,corrected)
        for name in names:
            e=epdefs.loc[name]
            row=c.loc[c.endpoint.eq(name)].iloc[0]
            title=endpoints.short_title(name,str(row.get('endpoint_plain',name)),labels)
            scale=100 if name=='nuclear_spot_fraction_dapi' else figures.fraction_scale(name)
            paths.append(chart(ctx,out,'FIG_'+name.upper(),title,w,c,name,
                               'percent (%)' if scale==100 else str(e.unit),scale))
        add(spec,sheets,titles.get(item['identity'],item['title']),paths,
            ['Contrasts','Per well'] if paths else item.get('sheets',[]),item['identity'],
            item.get('body','') if not paths else '')
    # Recorded localization classification and object-area distributions.
    dapi={'census':sheets['DAPI census'],'arms':sheets['DAPI census by arm'],'objects':sheets['DAPI objects'],
          'spots':sheets['DAPI spot classes']}
    result=dapi_mask.render_dapi(ctx,dapi,n,w,sheets['Per field'],c,out/'localization')
    loc=next(s for s in spec['slides'] if s['identity']=='q1_localization')
    loc['figures']=[asset(out/'localization'/r['png'],run.name,r.get('caption','')) for r in result['figures']]
    simple=render_simple_coloc(sheets['Simple coloc nuclei'],sheets['Simple coloc FOVs'],sheets['Simple coloc wells'],
        sheets['Simple coloc metrics'],out/'figures','BIN1 intron','RNASEH2B','WT',run.name)
    add(spec,sheets,'Pixel colocalization',[out/'figures'/r['png'] for r in simple],['Simple coloc metrics'],'simple_coloc')
    if a20:
        spec['slides'][-1]['body']='Pearson r and ICQ are threshold-free and unchanged by KO; Manders M1/M2 track the RNASEH2B intensity threshold (lower RNASEH2B in KO), so they are not used as colocalization readouts here. Manders now uses the fixed lab display floors; Costes and MAD versions are sensitivity only.'
    selected=sheets['Simple coloc representatives']
    stored=np.load(prior/'data/representative_pixels.npz')
    sources=pd.read_csv(prior/'data/representative_pixel_sources.csv')
    pixels={}
    for i,row in sources.iterrows():
        pixels[(row.well_id,row.image,row.nucleus_id)]=(stored[f'{i}_rna'],stored[f'{i}_partner'])
    cyto=render_cytofluorogram(selected,pixels,out/'figures','BIN1 intron','RNASEH2B')
    add(spec,sheets,'Cytofluorograms',[out/'figures'/r['png'] for r in cyto],['Simple coloc representatives'],'cytofluorogram')
    floor=pd.read_csv(prior/'nucleus_floor_comparison.csv')
    sheets['A17 Floor comparison']=floor
    paths=[]
    for name in ['rna1_spots_per_nucleus','nuclear_spot_fraction_dapi']:
        f,ax=figures.plt.subplots(figsize=(4.8,3.6))
        rows=floor[floor.endpoint.eq(name)].sort_values('floor_px')
        scale=100 if name=='nuclear_spot_fraction_dapi' else 1
        for arm,col in [('WT','mean_ref'),('QKI-KO','mean_test')]:
            ax.plot(rows.floor_px,rows[col]*scale,'o-',label=arm,color=ctx.colors[arm])
        ax.set_xlabel('Minimum nucleus area (pixels)');ax.set_ylabel('Percent nuclear' if scale==100 else 'Puncta / nucleus')
        ax.legend(frameon=False,fontsize=8)
        floor_window=ax.get_ylim()
        ax._axis_group='nucleus_floor_'+name
        ax._replicate_simple_axis=lambda variant,ax=ax,window=floor_window: ax.set_ylim(0 if variant=='full' else window[0],window[1])
        figures.layout_replicate_simple(f,ax,ctx,'Nucleus floor: '+('localization' if scale==100 else 'BIN1 count'),'Descriptive floor comparison; each floor has its own cohort')
        rec=figures.save(f,out/'figures','FIG_FLOOR_'+name,[],'Nucleus area floor comparison','A17 Floor comparison')
        paths.append(out/'figures'/rec['png'])
    add(spec,sheets,'Nucleus floor comparison',paths,['A17 Floor comparison'],'floor_comparison')
    analysis=build_analysis(run,out,prior/'data/RNASEH2B_TOTAL_CORRECTED.csv')
    sheets.update(analysis['tables'])
    amap={f['key']:f for f in analysis['figures']}
    control=sheets['A17 Secondary fields']
    sheets['A17 Secondary rule result']=pd.DataFrame([dict(BIN1_flagged=int(control.BIN1_3x_exceeds.sum()), RNASEH2B_flagged=int(control.RNASEH2B_3x_exceeds.sum()), control_fields=len(control))])
    analysis_sources=[['A17 Ratio','A17 Ratio wells'],['A17 Spearman','A17 Scatter pairs'],
                      ['A17 Radial summary'],['A17 Nearest observed'],['A17 Secondary fields','A17 Secondary intensity','A17 Secondary rule result']]
    for i,definition in enumerate(analysis['slides']):
        paths=[amap[key]['focus_png'] for key in definition['figures']]
        item=add(spec,sheets,definition['title'],paths,analysis_sources[i],identity='analysis_'+str(i))
        if i==0:
            item['readout_template']='Tested prediction R > 1: preferential retention not detected. R={R:.3f}, 95% CI [{ci_low:.3f}, {ci_high:.3f}], exact p={p_exact:.3g}; minimum detectable retention R={R_MDE_05:.3f} (80% normal-model power).'
        elif i==1:
            item['body']='Per-arm Spearman correlations and nucleus counts are shown in the plots.'
        elif i==3:
            item['body']='Observed nuclear BIN1 puncta. Shuffle-distance draws are missing.'
        elif i==4:
            item['readout_template']='Three-times-median spot audit: BIN1 {BIN1_flagged}/{control_fields} control FOVs flagged; RNASEH2B {RNASEH2B_flagged}/{control_fields}. Nuclear secondary intensity shown by arm. Named acquisition-method document missing.'
    micro=prepare_per_well_micrographs(run,out/'micrographs_per_well')
    slides.append_per_well_micrographs(spec,sheets,micro)
    if a20:
        from .coloc_existing import lab_floor_config
        ranges,partner_role=lab_floor_config(run)
        text='Display ranges (AU): '+ '; '.join(f'{label} {ranges[role][0]:g}–{ranges[role][1]:g}' for role,label in [('dapi','DAPI'),('rna','BIN1 intron'),(partner_role,'RNASEH2B')])
        for item in spec['slides']:
            if item.get('layout')=='micrographs': item['display_ranges']=text
    qc=[r['path'] for r in micro.get('qc',[]) if r.get('path')]
    if qc: add(spec,sheets,'QC walkthrough: segmentation and detections',qc,['Per-well micrograph selection'],'qc_walkthrough')
    if a20:
        names=['rna1_local_mean_at_partner_puncta','rna1_local_mean_at_partner_puncta_seconly_corrected']
        paths=[chart(ctx,out,'FIG_'+name.upper(),endpoints.short_title(name,name,labels),w,c,name,'Intensity (AU)') for name in names]
        add(spec,sheets,'BIN1 signal at RNASEH2B puncta',paths,['Contrasts','Per well'],'reverse_absolute_intensity')
    missing=analysis['missing']+micro.get('missing',[])
    (out/'METHODS.md').write_text((prior/'METHODS.md').read_text(encoding='utf-8')+'\n\n'+(out/'A17_ANALYSIS_METHODS.md').read_text(encoding='utf-8')+'\nPer-well micrographs use native publication images with recorded windows, wavelength LUTs and scale bars.\nFootprint qualification: A uses stored punctum-footprint means with equal punctum weighting within each nucleus. The engine can fall back to fitted-radius footprints; per-spot fallback flags are not persisted. No footprints were reconstructed during reporting.\n',encoding='utf-8')
    (out/'READOUT.md').write_text(rn_readout(c,analysis['summary']),encoding='utf-8')
    if a20:
        methods=(out/'METHODS.md').read_text(encoding='utf-8')
        methods=methods.replace('Absolute intensity and declared descriptive endpoints remain descriptive.',
            'Absolute intensity is tested in its declared Holm family; genuinely test-free endpoints remain descriptive.')
        methods=methods.replace('Absolute IF remains descriptive; N:C is shown beside corrected nuclear mean and raw total is\nsecondary.',
            'Absolute IF uses the mixed-model headline and Welch footer; raw and matched-secondary variants appear together.')
        methods+='\nA20: '+(out/'FAMILY_AMENDMENTS.md').read_text(encoding='utf-8')
        methods+='\nManders of record uses per-channel intensity-weighted signal at or above the fixed manual display floors. The numerator requires both channels to meet their floors; each denominator contains its own above-floor signal. A coefficient with no qualifying denominator remains missing. Costes and median+2.5 MAD coefficients are sensitivity only.\n'
        (out/'METHODS.md').write_text(methods,encoding='utf-8')
    return finish(out,sheets,spec,'REPORT.xlsx','Sam_RNASEH2B_BIN1.pptx',missing)


def rn_readout(contrasts,summary):
    c=contrasts.set_index('endpoint')
    def values(name,scale=1):
        r=c.loc[name]
        return f'WT {scale*r.mean_ref:.6g}, KO {scale*r.mean_test:.6g}'
    def p(name):
        value=c.loc[name,'p_headline']
        return f'{value:.6g}' if np.isfinite(value) else 'missing (descriptive)'
    r=summary['ratio']
    lines=[
        '1: BIN1 intron puncta/nucleus '+values('rna1_spots_per_nucleus')+' (p='+p('rna1_spots_per_nucleus')+'); footprint area (square micrometres) '+values('rna1_punctum_footprint_area_um2')+' (p='+p('rna1_punctum_footprint_area_um2')+').',
        '1b RNASEH2B level: corrected nuclear mean (AU) '+values('protein_nuclear_mean_seconly_corrected')+' (headline p='+p('protein_nuclear_mean_seconly_corrected')+'); assigned-cell total IF '+values('cell_total_intensity_protein')+' (headline p='+p('cell_total_intensity_protein')+').',
        '2: RNASEH2B mean in BIN1 footprints (AU) '+values('partner_mean_in_exact_rna1_footprint')+' (headline p='+p('partner_mean_in_exact_rna1_footprint')+'); rotation enrichment '+values('partner_rotation_enrichment_at_rna1')+' (p='+p('partner_rotation_enrichment_at_rna1')+').',
        '3: BIN1-anchored pairing excess (percentage points) '+values('paired_frac_rna1_at_partner_minus_shuffle',100)+'; headline p='+p('paired_frac_rna1_at_partner_minus_shuffle')+'.',
        '4: RNASEH2B-anchored pairing excess (percentage points) '+values('paired_frac_partner_at_rna1_minus_shuffle',100)+'; headline p='+p('paired_frac_partner_at_rna1_minus_shuffle')+'.',
        f"R: preferential retention not detected; R={r['R']:.6g}, 95% CI [{r['ci_low']:.6g}, {r['ci_high']:.6g}], exact p={r['p_exact']:.6g}; minimum detectable retention {100*(r['R_MDE_05']-1):.6g}%.",
        'Localization: DAPI-corrected BIN1 nuclear fraction (%) '+values('nuclear_spot_fraction_dapi',100)+'; headline p='+p('nuclear_spot_fraction_dapi')+'.']
    return '\n'.join(lines)+'\n'


def miat(prior,out,a20=False):
    from . import miat_qki
    from .coloc_existing import render_simple_coloc
    out.mkdir(parents=True,exist_ok=True);(out/'figures').mkdir(exist_ok=True)
    sheets=pd.read_excel(prior/'MIAT_QKI_RESULTS_v2.xlsx',sheet_name=None,header=1)
    spec=dict(cohort='mixed, explicitly labelled per slide',slides=[],a20=a20)
    ctx=miat_qki._context(prior/'data',miat_qki.COLORS)
    ctx.plot_style='replicate-simple'
    ctx.run_name='MIAT fixed-10 persisted report'
    figures.set_style()
    add(spec,sheets,'MIAT and QKI: study overview',source=['About v2','Deck context'],body='Fixed-colocalization and COUNT localization retain their recorded cohorts and denominators.')
    for measurement in miat_qki.MEASUREMENTS:
        paths_total=[];paths_associated=[];paths_ratio=[]
        for pool,policies in [('T',[miat_qki.POLICIES[0]]),('A',miat_qki.POLICIES)]:
            if a20:
                union=sheets['Ratio well values'].loc[lambda x:x.measurement.eq(measurement)].rename(columns={'arm':'group','biological_set':'well_id',pool:'well_mean_of_field_values'}).assign(endpoint='miat_scalar',axis_series=lambda x:x.null_policy)
                figures.prepare_axis_groups(ctx,[endpoints.Endpoint('miat_scalar','miat_scalar','detection','AU','MIAT',axis_group=f'miat_{pool}_{measurement}')],union)
            for policy in policies:
                w=sheets['Ratio well values'].loc[lambda x:x.measurement.eq(measurement)&x.null_policy.eq(policy)].copy()
                w=w.rename(columns={'arm':'group','biological_set':'well_id',pool:'well_mean_of_field_values'}).assign(endpoint='miat_scalar')
                c=sheets['Scalar contrasts'].loc[lambda x:x.measurement.eq(measurement)&x.pool.eq(pool)&(x.null_policy.eq(policy) if pool=='A' else True)].copy()
                c=c.assign(endpoint='miat_scalar',test_group='KD',reference_group='NT',p_welch_holm_within_family=c.p_welch_holm10)
                revised=sheets['A15 MIAT contrasts'].loc[lambda x:x.endpoint.eq(f'A15:{pool}:{measurement}:{policy}')]
                if len(revised):c=revised.assign(endpoint='miat_scalar')
                label=miat_qki.LABELS[miat_qki.POLICIES.index(policy)]
                title=('Total MIAT' if pool=='T' else label)+(' puncta' if measurement=='spot_count' else ' intensity')
                ctx.report_nuclei=pd.DataFrame();ctx.endpoint_columns={}
                if a20 and (pool=='A' or measurement=='spot_count'):
                    column=('observed_spot_count' if pool=='T' else 'q95_positive_spot_count' if measurement=='spot_count' else 'q95_associated_miat_intensity')
                    ctx.report_nuclei=sheets['Nucleus roster'].loc[lambda x:x.null_policy.eq(policy)].rename(columns={'arm':'group','biological_set':'well_id','image_key':'image','nucleus_uid':'nucleus_id',column:'value'})
                    ctx.endpoint_columns={'miat_scalar':'value'}
                path=chart(ctx,out,f'FIG_MIAT_{pool}_{measurement}_{policy}',title,w,c,'miat_scalar',
                           'Puncta / nucleus' if measurement=='spot_count' else 'MIAT footprint intensity / nucleus (AU)')
                (paths_total if pool=='T' else paths_associated).append(path)
        for policy,label in zip(miat_qki.POLICIES,miat_qki.LABELS):
            row=sheets['Ratio intervals'].loc[lambda x:x.measurement.eq(measurement)&x.null_policy.eq(policy)].iloc[0]
            f,ax=figures.plt.subplots(figsize=(4.8,3.6))
            ax.errorbar([0],[row.R],yerr=[[row.R-row.ci_low],[row.ci_high-row.R]],fmt='o',capsize=5,color=ctx.colors['KD'])
            ax.axhline(1,color='#555555',ls=':');ax.set_xticks([0],['KD / NT relative retention'])
            ax.set_ylabel('R = rA / rT');ax.set_ylim(min(.9,row.ci_low*.9),max(1.1,row.ci_high*1.15))
            if a20:
                union=sheets['Ratio intervals'].loc[lambda x:x.measurement.eq(measurement)]
                lo,hi=min(.9,union.ci_low.min()*.9),max(1.1,union.ci_high.max()*1.15)
                ax._replicate_simple_axis=lambda variant,ax=ax,lo=lo,hi=hi: ax.set_ylim(0 if variant=='full' else lo,hi)
                ax._axis_group='miat_retention_'+measurement
            else: ax._replicate_simple_axis=lambda variant:None
            figures.layout_replicate_simple(f,ax,ctx,label,f'R={row.R:.4g}; 95% CI [{row.ci_low:.4g}, {row.ci_high:.4g}]; exact p={row.p_exact:.4g}; R-MDE={row.R_MDE_05:.4g}')
            rec=figures.save(f,out/'figures',f'FIG_MIAT_R_{measurement}_{policy}',[],label,'Ratio intervals')
            paths_ratio.append(out/'figures'/rec['png'])
        label='puncta' if measurement=='spot_count' else 'footprint intensity'
        add(spec,sheets,'Total MIAT '+label,paths_total,['Ratio well values','Scalar contrasts','A15 MIAT contrasts'])
        add(spec,sheets,'Associated MIAT '+label,paths_associated,['Ratio well values','Scalar contrasts','A15 MIAT contrasts'])
        add(spec,sheets,'Preferential retention: MIAT '+label,paths_ratio,['Ratio intervals'],body='The tested prediction is preferential spatial retention. Exact-test results and model-based intervals are shown per policy.')
    ctx.report_nuclei=pd.DataFrame();ctx.endpoint_columns={}
    qpaths=[]
    for name,title,source,col in [('protein_nuclear_mean','QKI nuclear pixel mean','QKI pixel mean','value'),
            ('mean_nc_ratio_total_intensity_protein','QKI nuclear:cytoplasmic signal','QKI per well','mean_nc_ratio_total_intensity_protein'),
            ('mean_nuc_total_intensity_protein','QKI total nuclear signal','QKI per well','mean_nuc_total_intensity_protein'),
            ('protein_pooled_rotation_enrichment_at_rna1_spots','QKI rotation enrichment at MIAT','QKI per well','protein_pooled_rotation_enrichment_at_rna1_spots')]:
        w=sheets[source].rename(columns={'arm':'group',col:'well_mean_of_field_values'}).assign(endpoint=name)
        c=sheets['QKI contrasts'].loc[lambda x:x.endpoint.eq(name)].copy()
        if not c.empty:c=c.assign(test_group='KD',reference_group='NT')
        qpaths.append(chart(ctx,out,'FIG_MIAT_'+name.upper(),title,w,c,name))
    add(spec,sheets,'QKI nuclear signal and spatial enrichment',qpaths,['QKI per well','QKI pixel mean','QKI contrasts'])
    w=sheets['COUNT localization'];name=w.endpoint.iloc[0]
    c=sheets['COUNT contrast'].assign(endpoint=name,test_group='KD',reference_group='NT')
    path=chart(ctx,out,'FIG_MIAT_LOCALIZATION','MIAT puncta, nuclear fraction',w,c,name,'Percent nuclear',100)
    add(spec,sheets,'MIAT nuclear localization',[path],['COUNT localization','COUNT contrast'])
    simple=render_simple_coloc(sheets['Simple coloc nuclei'],sheets['Simple coloc FOVs'],sheets['Simple coloc wells'],sheets['Simple coloc metrics'],out/'figures','MIAT','QKI','NT','MIAT fixed-colocalization')
    add(spec,sheets,'Pixel colocalization',[out/'figures'/r['png'] for r in simple],['Simple coloc metrics'],'simple_coloc')
    for endpoint in ['candidate/observed','usable/candidate','usable/observed']:
        paths=[]
        if a20:
            union=sheets['Coverage'].loc[lambda x:x.endpoint.eq(endpoint)&x.tier.eq('well')].rename(columns={'arm':'group','biological_set':'well_id','value':'well_mean_of_field_values'}).assign(endpoint='coverage',axis_series=lambda x:x.null_policy)
            figures.prepare_axis_groups(ctx,[endpoints.Endpoint('coverage','coverage','partner','fraction','Coverage',axis_group='coverage_'+endpoint)],union)
        for policy,label in zip(miat_qki.POLICIES,miat_qki.LABELS):
            w=sheets['Coverage'].loc[lambda x:x.endpoint.eq(endpoint)&x.null_policy.eq(policy)&x.tier.eq('well')].rename(columns={'arm':'group','biological_set':'well_id','value':'well_mean_of_field_values'}).assign(endpoint='coverage')
            paths.append(chart(ctx,out,'FIG_MIAT_COVERAGE_'+endpoint.replace('/','_')+'_'+policy,label,w,pd.DataFrame(),'coverage',endpoint))
        add(spec,sheets,'Null usability: '+endpoint,paths,['Coverage'])
    # Retain the prior calibrated micrograph slide if original planes are missing.
    old=yaml.safe_load((prior/'deck_spec.resolved.yaml').read_text(encoding='utf-8'))
    micro=next(s for s in old['slides'] if s.get('identity')=='micrographs')
    destination=out/'micrographs_per_well';destination.mkdir(exist_ok=True)
    paths=[]
    for a in micro['figures']:
        src=Path(a['path']);dst=destination/src.name;shutil.copyfile(src,dst);paths.append(dst)
    add(spec,sheets,'MIAT and QKI micrographs',paths,['A15 Micrographs'],'micrographs')
    missing=['MIAT per-well micrograph planes missing at the recorded acquisition paths; the existing calibrated micrograph slide is retained.']
    for title,body in [('Method review','Associated pools are operational spatial categories. Null-unusable calls remain unknown.'),
            ('Current limits','Spatial association alone does not establish molecular protection. Recorded validation limits remain in the source workbook.'),
            ('Provenance','All measurements and inference derive from the reviewed source workbook. Nuclei, fields and wells retain their recorded hierarchy.')]:
        add(spec,sheets,title,source=['Deck context'],body=body)
    (out/'METHODS.md').write_text((prior/'METHODS.md').read_text(encoding='utf-8'),encoding='utf-8')
    (out/'READOUT.md').write_text((prior/'READOUT.md').read_text(encoding='utf-8'),encoding='utf-8')
    add_miat_context(sheets,spec,prior,out)
    return finish(out,sheets,spec,'MIAT_QKI_RESULTS_v2.xlsx','MIAT_QKI_colocalization_final.pptx',missing)


def add_miat_context(sheets,spec,prior,out):
    """Retain the original basal-NT composite as four distinct chart objects."""
    from .miat_qki import _context,COLORS
    ctx=_context(prior/'data',COLORS)
    ctx.plot_style='replicate-simple';ctx.run_name='MIAT persisted basal-NT cohort';ctx.group_order=['NT']
    paths=[]
    for column,title,unit in [('pearson_r','Basal NT: Pearson r','Pearson r'),
            ('manders_m1_costes','Basal NT: Manders M1 mixture','Signal fraction'),
            ('manders_m2_costes','Basal NT: Manders M2 mixture','Signal fraction'),
            ('rotation_null_enrichment','Basal NT: native rotation','Enrichment ratio')]:
        w=sheets['Basal NT'].rename(columns={'well':'well_id',column:'well_mean_of_field_values'}).assign(group='NT',endpoint='miat_basal')
        paths.append(chart(ctx,out,'FIG_MIAT_BASAL_'+column,title,w,pd.DataFrame(),'miat_basal',unit))
    add(spec,sheets,'Basal NT colocalization',paths,['Basal NT'],identity='basal_nt',
        body='Historical Costes-mixture values and native rotation support remain distinct from converged-only pixel metrics.')
    add(spec,sheets,'Experimental design',source=['Deck context','About v2'],identity='design',
        body='NT and KD wells are distributed across the recorded slides. Nuclei are averaged within FOV, then FOVs within wells. COUNT localization uses its own cohort.')
    design=spec['slides'].pop();spec['slides'].insert(1,design)
    add(spec,sheets,'Validation priorities',source=['Deck context'],identity='validation_priorities',
        body='Specificity, registration and optical-resolution validation remain relevant to spatial interpretation. Orthogonal perturbation evidence is required to test molecular protection.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['run','rn-prior','rn-out','miat-prior','miat-out']:parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--only',choices=['rn','miat'])
    args=parser.parse_args()
    if args.only!='miat':rnase(args.run,args.rn_prior,args.rn_out)
    if args.only!='rn':miat(args.miat_prior,args.miat_out)
