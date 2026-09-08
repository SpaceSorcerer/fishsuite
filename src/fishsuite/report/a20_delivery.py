"""A20 report-only amendments and handoff; source runs remain read-only."""
from pathlib import Path
from dataclasses import fields, replace
import json
import shutil
import numpy as np
import pandas as pd
from . import endpoints as ep, aggregate, figures, workbook
from .stats import holm


def cache_signature(run,frame,definitions=()):
    """Content attestations, not cache existence, govern reuse."""
    import hashlib
    from .provenance import sha256
    return dict(version='A20-1',config=sha256(run/'run_config.json'),nuclei=sha256(run/'nuclei_metrics.csv'),
                values=hashlib.sha256(pd.util.hash_pandas_object(frame,index=True).values.tobytes()).hexdigest(),
                endpoints=[dict(name=e.name,column=e.column,family=e.family,usability=e.usability_flag) for e in definitions])


def cache_matches(path,signature):
    meta=path.with_suffix('.provenance.json')
    if not path.is_file() or not meta.is_file(): return False
    from .provenance import sha256
    record=json.loads(meta.read_text())
    return record.get('inputs')==signature and record.get('output_sha256')==sha256(path)


def seal_cache(path,signature):
    from .provenance import sha256
    path.with_suffix('.provenance.json').write_text(json.dumps(dict(inputs=signature,output_sha256=sha256(path)),indent=2),encoding='utf-8')


def amend_rnase(sheets, run, out):
    """Reuse existing fitted contrasts, fit newly added measurements only."""
    from .coloc_existing import recompute_lab_manders, simple_statistics
    run, out = Path(run), Path(out)
    raw = pd.read_csv(run/'nuclei_metrics.csv')
    n = sheets['Per nucleus'].copy()
    extra = [c for c in raw if c not in n]
    n = n.merge(raw[['image','nucleus_id']+extra], on=['image','nucleus_id'], how='left', validate='one_to_one')
    defs = []
    allowed = {f.name for f in fields(ep.Endpoint)}
    for row in sheets['Endpoint definitions'].to_dict('records'):
        data = {k:v for k,v in row.items() if k in allowed and not (isinstance(v,float) and np.isnan(v))}
        if isinstance(data.get('alt_columns'),str): data['alt_columns'] = ()
        defs.append(ep.amend_endpoint(ep.Endpoint(**data)))
    existing = {e.name for e in defs}
    additions = [e for e in ep.a3_endpoints([]) if e.name not in existing]
    # Equal FOV-weighted matched-arm secondary nuclear means, as in A17.
    sec = raw.loc[raw.secondary_only.astype(str).str.lower().eq('true')].copy()
    sec['group'] = np.where(sec.image.str.contains('ko',case=False),'QKI-KO','WT')
    sec['rna_floor_residual_mean']=sec.nuclear_above_floor_intensity_rna1/sec.nucleus_area_px
    sec['protein_floor_residual_mean']=sec.nuclear_above_floor_intensity_protein/sec.nucleus_area_px
    baselines = sec.groupby(['group','image'])[['protein_nuclear_mean','rna_nuclear_mean','rna_floor_residual_mean','protein_floor_residual_mean']].mean().groupby('group').mean()
    mappings = {
        'protein_nuclear_mean': ('protein_nuclear_mean', None),
        'cell_total_intensity_protein': ('protein_nuclear_mean','cell_area_px'),
        'nuclear_total_intensity_protein': ('protein_nuclear_mean','nucleus_area_px'),
        'cell_total_intensity_rna1': ('rna_nuclear_mean','cell_area_px'),
        'nuclear_total_intensity_rna1': ('rna_nuclear_mean','nucleus_area_px'),
        'partner_mean_in_exact_rna1_footprint': ('protein_nuclear_mean',None),
        'rna1_local_mean_at_partner_puncta': ('rna_nuclear_mean',None),
        'rna1_nuclear_above_floor_intensity': ('rna_floor_residual_mean','nucleus_area_px'),
        'nuclear_above_floor_intensity_protein': ('protein_floor_residual_mean','nucleus_area_px'),
    }
    for e in defs+list(additions):
        if e.name not in mappings: continue
        name = e.name+'_seconly_corrected'
        if name in existing: continue
        baseline, area = mappings[e.name]
        n[name] = n[e.column]-n.group.map(baselines[baseline])*(n[area] if area else 1)
        additions.append(replace(e,name=name,column=name,axis_group=e.axis_group or e.name,
                                 label=e.label+', matched secondary subtracted',
                                 note=ep.INTENSITY_CAVEAT+'; no clipping; baseline uncertainty not propagated'))
    # Above-floor integrals are sum(max(pixel-floor,0)) across the full nucleus
    # (core/modes/rna_rna.py). Subtract the control mean of that SAME transformed
    # quantity, per nuclear pixel, times the primary nucleus area.
    if additions:
        labels = n[['image','group','well_id','secondary_only']].drop_duplicates()
        f = aggregate.per_field_long(n,pd.DataFrame({'image':[]}),additions,labels,0)
        w = aggregate.per_well_long(f)
        fit_cache=out/'data'/'A20_added_contrasts.csv'
        fit_cache.parent.mkdir(parents=True,exist_ok=True)
        fit_signature=cache_signature(run,n,additions)
        if cache_matches(fit_cache,fit_signature):
            c=pd.read_csv(fit_cache,float_precision='round_trip')
        else:
            c = aggregate.build_contrasts(w,f,additions,[],['WT','QKI-KO'],'WT',ep.channel_labels(json.loads((run/'run_config.json').read_text())),nuclei=n)
            c.to_csv(fit_cache,index=False)
            seal_cache(fit_cache,fit_signature)
        sheets['Per field'] = pd.concat([sheets['Per field'],f],ignore_index=True)
        sheets['Per well'] = pd.concat([sheets['Per well'],w],ignore_index=True)
        sheets['Contrasts'] = pd.concat([sheets['Contrasts'],c],ignore_index=True)
        defs += additions
    c = sheets['Contrasts'].copy()
    absolute = c.absolute_intensity.fillna(False).astype(bool)
    c.loc[absolute,['descriptive_only','excluded_from_holm','holm_exclusion_reason','endpoint_note']] = [False,'','',ep.INTENSITY_CAVEAT]
    c['in_holm_family'] = ~(c.descriptive_only.fillna(False).astype(bool) | c.endpoint_absent_in_run.fillna(False).astype(bool) | c.excluded_from_holm.fillna('').astype(bool))
    for _, indexes in c.groupby(['family','test_group','reference_group']).groups.items():
        ids = [i for i in indexes if c.at[i,'in_holm_family']]
        for dest,source in [('p_welch_holm_within_family','p_welch'),('p_headline_holm','p_headline'),('p_mixed_holm','p_headline')]:
            c.loc[ids,dest] = holm(c.loc[ids,source].tolist())
        c.loc[ids,'holm_family_size'] = np.isfinite(c.loc[ids,'p_headline']).sum()
    c.loc[c.p_mixed.isna(),'p_mixed_holm'] = np.nan
    c['significant_headline_holm_0p05'] = (c.p_headline_holm < .05).where(c.in_holm_family)
    c['significant_holm_0p05'] = (c.p_welch_holm_within_family < .05).where(c.in_holm_family)
    c['observed_g_reaches_mde']=c.observed_g_reaches_mde.astype(object)
    from functools import lru_cache
    from .stats import mde_hedges_g
    @lru_cache(None)
    def mde(alpha,n1,n2): return mde_hedges_g(alpha,n1=n1,n2=n2)
    for i in c.index[c.in_holm_family]:
        n1,n2,k=int(c.at[i,'n_wells_test']),int(c.at[i,'n_wells_reference']),c.at[i,'holm_family_size']
        if min(n1,n2)>=2 and k>0:
            c.at[i,'mde_hedges_g_alpha_0p05']=mde(.05,n1,n2)
            c.at[i,'mde_hedges_g_at_family_alpha']=mde(.05/float(k),n1,n2)
            c.at[i,'observed_g_reaches_mde']=abs(c.at[i,'hedges_g'])>=c.at[i,'mde_hedges_g_at_family_alpha']
    sheets['Contrasts'],sheets['Per nucleus'] = c,n
    sheets['Endpoint definitions'] = pd.DataFrame([e.__dict__ for e in defs])
    sheets['Multiplicity plan']=c[['endpoint','family','in_holm_family','holm_exclusion_reason','holm_family_size']]
    for family,name in ep.FAMILY_SHEET.items():
        sheets[name] = aggregate.family_sheet(c,sheets['Per well'],family,['WT','QKI-KO'])
    # Cache the requested read-only pixel measurement, so figure retries never
    # need another image read. The cache is in the new package, never the run.
    cache = out/'data'/'A20_lab_manders.csv'
    cache.parent.mkdir(parents=True,exist_ok=True)
    lab_signature=cache_signature(run,sheets['Simple coloc nuclei'])
    cfg=json.loads((run/'run_config.json').read_text())
    table=sheets['Simple coloc nuclei']
    image_paths={row.image:Path(cfg['input_dir'])/('SecOnly_KO' if 'Sec-Only-KO' in row.image else 'SecOnly_WT' if 'Sec-Only-WT' in row.image else str(row.condition))/row.image for row in table.itertuples()}
    # Per-FOV caches attest the mask, VSI header and ETS pixel payload too.
    lab = recompute_lab_manders(table,run,image_paths,out/'data'/'lab_manders_fovs')
    lab.to_csv(cache,index=False)
    seal_cache(cache,lab_signature)
    metrics=('pearson_r_csp','manders_m1_labfloor','manders_m2_labfloor','li_icq_csp')
    f,w,c=simple_statistics(lab,'WT',metrics)
    sheets['Manders Costes sensitivity'] = sheets['Simple coloc nuclei'][['group','well_id','image','nucleus_id','manders_m1_costes_only','manders_m2_costes_only']]
    sheets['Manders MAD sensitivity'] = sheets['Simple coloc nuclei'][['group','well_id','image','nucleus_id','manders_m1_runthr','manders_m2_runthr']]
    workbook.SHEET_DESCRIPTION['Manders Costes sensitivity']='Sensitivity only: per-nucleus converged Costes thresholds; failed fits remain missing.'
    workbook.SHEET_DESCRIPTION['Manders MAD sensitivity']='Sensitivity only: recorded run-wide median + 2.5 MAD thresholds, identical across nuclei and arms.'
    workbook.SHEET_DESCRIPTION['Simple coloc nuclei']='Manders of record uses fixed manual display floors in lab_floor_rna and lab_floor_partner; Costes and runthr columns are historical sensitivity only. Rows retain exact plane/mask nucleus identity.'
    workbook.SHEET_DESCRIPTION['Simple coloc metrics']='Mixed model and Welch well-mean comparisons; fixed manual-floor Manders of record, plus threshold-free Pearson and ICQ.'
    for name in ('Coloc per nucleus','Coloc per field','Coloc per well','Coloc contrasts'):
        workbook.SHEET_DESCRIPTION[name]='Historical panel sensitivity: Costes and MAD Manders are not the A20 coefficients of record; use Simple coloc sheets for fixed lab-floor Manders.'
    sheets.update({'Simple coloc nuclei':lab,'Simple coloc FOVs':f,'Simple coloc wells':w,'Simple coloc metrics':c})
    text='acquisition uniform per acquirer 2026-09-07; staining batch caveat\nAbsolute-intensity endpoints and matched-secondary variants enter their declared Holm families; mixed model headline, Welch on wells in footer.\nCorrection subtracts matched secondary nuclear mean times measured area for integrals; mean endpoints subtract the baseline directly. No clipping or propagated baseline uncertainty.\nAbove-floor integrals are sum(max(pixel-display_floor,0)) over the full nucleus; their correction uses the matching secondary mean of this same transformed signal per nuclear pixel times the primary nucleus area. Native definition verified in core/modes/rna_rna.py.\n'
    (out/'FAMILY_AMENDMENTS.md').write_text(text,encoding='utf-8')
    return defs


def topic(title, stem, miat=False):
    text=(title+' '+stem).lower()
    if 'qc' in text or 'walkthrough' in text: return 'qc'
    if 'micrograph' in text: return 'micrographs'
    if miat:
        for terms,name in [(('basal',),'basal_colocalization'),(('coverage','usability'),'coverage'),(('pixel','cytofluorogram','simple_coloc'),'pixel_colocalization'),(('localization',),'miat_localization'),(('retention',),'retention_ratio'),(('associated',),'associated_pool'),(('qki nuclear','qki total','qki rotation'),'qki_level')]:
            if any(t in text for t in terms): return name
        return 'miat_level'
    if 'total_intensity_rna1' in text or 'rna1_nuclear_above_floor' in text or 'punctum_footprint' in text: return 'bin1_puncta_count_and_size'
    if 'rna1_local_mean_at_partner' in text: return 'rnaseh2b_at_bin1_puncta'
    if stem.lower().startswith('fig_seconly') or 'secondary-only controls' in title.lower(): return 'secondary_only'
    if 'object_area' in text or stem.lower().startswith('fig_floor'): return 'nucleus_selection_and_floor'
    for terms,name in [(('nucleus floor','selection','census','area distribution'),'nucleus_selection_and_floor'),(('pixel','cytofluorogram','simple_coloc'),'pixel_colocalization'),(('radial','around nuclear'),'radial_profile'),(('preferential',),'preferential_coloc_ratio'),(('paired','pairing','nearest'),'puncta_pairing'),(('localization','dapi fraction','nuclear fraction'),'bin1_localization'),(('enrichment','called_coloc','footprint'),'rnaseh2b_at_bin1_puncta'),(('level','total_if','total_intensity_protein','protein_nuclear_mean'),'rnaseh2b_level')]:
        if any(t in text for t in terms): return name
    return 'bin1_puncta_count_and_size'


def organize(out,spec,miat=False):
    """Move explicitly indexed assets; no recursive file discovery or raw writes."""
    import base64
    from PIL import Image
    topics=('miat_level miat_localization qki_level basal_colocalization associated_pool retention_ratio coverage pixel_colocalization micrographs qc' if miat else 'bin1_puncta_count_and_size bin1_localization rnaseh2b_level rnaseh2b_at_bin1_puncta puncta_pairing pixel_colocalization preferential_coloc_ratio radial_profile secondary_only nucleus_selection_and_floor micrographs qc').split()
    for t in topics:
        for kind in ('png','svg'):
            (out/'figures'/t/kind).mkdir(parents=True,exist_ok=True)
            (out/'figures'/'full_axis'/t/kind).mkdir(parents=True,exist_ok=True)
    deck=out/'deck_figures';deck.mkdir(exist_ok=True)
    moved={}; index=[]; captured={};axes=[]
    from .provenance import sha256
    for slide,item in enumerate(spec['slides'],1):
        for number,a in enumerate(item['figures'],1):
            source=Path(a['path']);stem=source.stem.removesuffix('_focus')
            t=topic(item['title'],stem,miat)
            if str(source.resolve()) in figures.FIGURE_DATA:
                captured[(slide,number)]=dict(figures.FIGURE_DATA[str(source.resolve())],axis_group=figures.FIGURE_AXES.get(str(source.resolve()),{}).get('axis_group',''))
            for full in (False,True):
                key='full_path' if full else 'path'
                if key not in a: continue
                old=Path(a[key]);root=out/'figures'/('full_axis' if full else '')/t
                if str(old.resolve()) in figures.FIGURE_AXES:
                    axes.append(dict(slide=slide,figure=old.name,**figures.FIGURE_AXES[str(old.resolve())]))
                for ext in ('png','svg'):
                    src=old.with_suffix('.'+ext)
                    dst=root/ext/src.name
                    if str(src) in moved:
                        dst=moved[str(src)]
                    elif src.exists():
                        if src.resolve()!=dst.resolve(): shutil.move(str(src),str(dst))
                        moved[str(src)]=dst
                    elif ext=='svg':
                        png=root/'png'/old.name
                        with Image.open(png) as im: width,height=im.size
                        data=base64.b64encode(png.read_bytes()).decode()
                        dst.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{width}" height="{height}"><image width="{width}" height="{height}" xlink:href="data:image/png;base64,{data}"/></svg>',encoding='utf-8')
                    moved[str(src)]=dst
                    if not full:
                        deckpath=deck/f'slide{slide:02d}_{number:02d}_{stem}.{ext}'
                        shutil.copy2(dst,deckpath)
                        index.append(dict(topic=t,file=str(dst.relative_to(out)),slide=slide,deck_file=str(deckpath.relative_to(out))))
                    else:
                        index.append(dict(topic=t,file=str(dst.relative_to(out)),slide=slide,deck_file=''))
                    if ext=='png':
                        a[key]=str(dst.resolve());a['full_sha256' if full else 'sha256']=sha256(dst)
    # All other generated standalone plots have a known flat figures directory.
    for src in list((out/'figures').glob('*.png'))+list((out/'figures').glob('*.svg')):
        full='_full' in src.stem
        root=out/'figures'/('full_axis' if full else '')/topic('',src.stem,miat)/src.suffix[1:]
        dst=root/src.name
        shutil.move(str(src),str(dst))
        index.append(dict(topic=root.parent.name,file=str(dst.relative_to(out)),slide='',deck_file=''))
    (out/'FIGURE_INDEX.md').write_text('\n'.join(f"- {r['topic']} → {r['file']} → slide {r['slide'] or 'supplementary'}" for r in index)+'\n',encoding='utf-8')
    (out/'AXIS_GROUPS.json').write_text(json.dumps(axes,indent=2),encoding='utf-8')
    return captured,index


def export_figure_workbook(out,sheets,spec,captured):
    from .figure_workbook import write_tables
    return write_tables(out,sheets,spec,captured)


def remap_local_assets(out,sheets,spec):
    """Update local report pointers after moving assets; source-run paths persist."""
    out=Path(out).resolve()
    destinations={Path(a[key]).name:str(Path(a[key]).resolve()) for s in spec['slides'] for a in s['figures'] for key in ('path','full_path') if key in a}
    def rewrite(value):
        if not isinstance(value,str): return value
        if not value.lower().startswith(str(out).lower()): return value
        return destinations.get(Path(value).name,value)
    for frame in sheets.values():
        for column in frame.select_dtypes(include='object'):
            frame[column]=frame[column].map(rewrite)
    path=out/'micrographs_per_well'/'micrograph_slides.json'
    if path.is_file():
        def nested(value):
            if isinstance(value,dict): return {k:nested(v) for k,v in value.items()}
            if isinstance(value,list): return [nested(v) for v in value]
            return rewrite(value)
        path.write_text(json.dumps(nested(json.loads(path.read_text())),indent=2),encoding='utf-8')
