"""Values-only figure tables and the fishsuite figure --from-xlsx reader."""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pandas as pd
from . import figures
from .provenance import guard_output


def well_table(well):
    result=well.rename(columns={'group':'condition','well_id':'well','well_mean_of_field_values':'value'})[['condition','well','value']].copy()
    summary=result.groupby('condition').value.agg(SD='std',n='count')
    return result.join(summary,on='condition')


def write_tables(out,sheets,spec,captured):
    """One primary values table per embedded figure; nucleus tables when available."""
    import re
    metadata={}; missing=[]
    with pd.ExcelWriter(out/'DATA_FOR_FIGURES.xlsx',engine='openpyxl') as writer:
        for slide,item in enumerate(spec['slides'],1):
            for number,asset in enumerate(item['figures'],1):
                path=Path(asset['path']);stem=path.stem.removesuffix('_focus')
                name=f's{slide:02d}_{number:02d}_'+re.sub('[^A-Za-z0-9_]','',stem.removeprefix('FIG_'))[:15]
                data=captured.get((slide,number))
                meta=dict(title=asset.get('caption') or item['title'],image=str(path.relative_to(out)),kind='image',ylabel='')
                values=pd.DataFrame(columns=['condition','well','value','SD','n'])
                nuclei=pd.DataFrame()
                if data is not None:
                    values=well_table(data['well'])
                    values[['value','SD']]*=data['scale']
                    meta.update(kind='well',ylabel=data['ylabel'],endpoint=data['endpoint'],axis_group=data.get('axis_group',''))
                    if len(data['nuclei']):
                        raw=data['nuclei']
                        nuclei=raw.rename(columns={'group':'condition','well_id':'well',data['nuc_column']:'value'})
                        nuclei=nuclei[[c for c in ['condition','well','image','nucleus_id','value'] if c in nuclei]].copy()
                        nuclei['value']*=data['scale']
                else:
                    values,nuclei,extra=custom_values(stem,item,sheets,out)
                    meta.update(extra)
                    if values.empty and meta['kind']!='image': missing.append(name+': no numeric source')
                primary=['condition','well','value','SD','n']
                values=values[primary+[c for c in values if c not in primary]]
                values.to_excel(writer,sheet_name=name,index=False)
                if not nuclei.empty:
                    keep=['condition','well','image','nucleus_id','nucleus_uid','object_id','spot_id','class','x','value']
                    nuclei=nuclei[[c for c in keep if c in nuclei]]
                    suffix={'hexbin':'pixels','ecdf':'spots','stacked':'spots','hist':'objects','fieldpoints':'fovs'}.get(meta['kind'],'nuclei')
                    nuclei.to_excel(writer,sheet_name=name+'_'+suffix,index=False)
                    meta['nuclei_sheet']=name+'_'+suffix
                metadata[name]=meta
    (out/'DATA_FOR_FIGURES.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    (out/'regenerate_figures.py').write_text(
        'from pathlib import Path\nimport argparse\nfrom fishsuite.report.figure_workbook import render\n'
        'p=argparse.ArgumentParser()\np.add_argument("--sheet")\np.add_argument("--out",type=Path)\na=p.parse_args()\n'
        'root=Path(__file__).resolve().parent\nrender(root/"DATA_FOR_FIGURES.xlsx",a.sheet,a.out or root/"regenerated_figures")\n',encoding='utf-8')
    return missing


def custom_values(stem,item,sheets,out):
    """Explicit mappings; never infer a biological quantity from a column position."""
    empty=pd.DataFrame(columns=['condition','well','value','SD','n'])
    meta={'kind':'image'}
    nuclei=pd.DataFrame()
    values=empty
    def normal(frame,value,well='condition',arm='arm'):
        frame=frame.copy()
        if well==arm:
            frame['well']=frame[well];well='well'
        frame=frame.rename(columns={arm:'condition',well:'well',value:'value'})
        return frame
    if stem.startswith('FIG_RNASEH2B_PREFERENTIAL_COLOC_'):
        suffix=stem.rsplit('_',1)[-1]
        if suffix=='R':
            values=sheets['A17 Ratio'][['R','ci_low','ci_high']].rename(columns={'R':'value'}).assign(condition='KO / WT',well='')
            meta={'kind':'interval','ylabel':'R = rA / rT'}
        else:
            column='T' if suffix=='TOTAL' else 'A'
            raw=sheets['A17 Ratio wells']
            values=well_table(raw.rename(columns={'arm':'group','biological_set':'well_id',column:'well_mean_of_field_values'}))
            nuclei=normal(sheets['A17 Ratio nuclei'],column)
            meta={'kind':'well','ylabel':'Intensity (AU)'}
    elif 'LEVEL_ENRICHMENT' in stem:
        suffix=stem.rsplit('_',1)[-1]
        nuclei=sheets['A17 Scatter pairs'].copy()
        if suffix!='OVERLAY': nuclei=nuclei.loc[nuclei.arm.eq(suffix)]
        nuclei=normal(nuclei,'protein_rotation_enrichment_at_rna1_spots').rename(columns={'protein_nuclear_mean':'x'})
        values=nuclei.groupby(['condition','well']).value.agg(value='mean',SD='std',n='count').reset_index()
        meta={'kind':'scatter','xlabel':'Nuclear RNASEH2B mean (AU)','ylabel':'Rotation-null enrichment'}
    elif stem=='FIG_RNASEH2B_RADIAL_PROFILE':
        values=normal(sheets['A17 Radial wells'],'enrichment').rename(columns={'ring_um':'x'})
        meta={'kind':'line','xlabel':'Radial bin (um)','ylabel':'Enrichment vs position null'}
    elif 'NEAREST_NEIGHBOUR' in stem:
        nuclei=normal(sheets['A17 Nearest observed'],'nn_distance_um')
        values=nuclei.groupby(['condition','well']).value.agg(value='mean',SD='std',n='count').reset_index()
        meta={'kind':'ecdf','xlabel':'Nearest RNASEH2B distance (um)','ylabel':'Cumulative fraction'}
    elif stem.startswith('FIG_FLOOR_'):
        endpoint=stem.removeprefix('FIG_FLOOR_');source=sheets['A17 Floor comparison']
        source=source.loc[source.endpoint.eq(endpoint)]
        values=pd.concat([source[['floor_px',col]].rename(columns={'floor_px':'x',col:'value'}).assign(condition=arm,well='') for arm,col in [('WT','mean_ref'),('KO','mean_test')]],ignore_index=True)
        if endpoint=='nuclear_spot_fraction_dapi': values.value*=100
        meta={'kind':'line','xlabel':'Minimum nucleus area (pixels)','ylabel':endpoint}
    elif stem.startswith('FIG_MIAT_R_'):
        rows=sheets['Ratio intervals']
        selected=[r for r in rows.to_dict('records') if stem==f"FIG_MIAT_R_{r['measurement']}_{r['null_policy']}"]
        if selected:
            values=pd.DataFrame(selected)[['R','ci_low','ci_high']].rename(columns={'R':'value'}).assign(condition='KD / NT',well='')
            meta={'kind':'interval','ylabel':'R = rA / rT'}
    elif stem=='FIG_LOCALIZATION_spot_classes':
        source=sheets['DAPI spot classes'].loc[lambda x:~x.secondary_only].copy()
        counts=source.groupby(['group','condition','class']).size().rename('count').reset_index()
        counts['value']=100*counts['count']/counts.groupby(['group','condition'])['count'].transform('sum')
        values=counts.rename(columns={'condition':'well','group':'condition'})
        nuclei=source.rename(columns={'condition':'well','group':'condition'})
        meta={'kind':'stacked','ylabel':'BIN1 spots (%)'}
    elif stem=='FIG_LOCALIZATION_object_area':
        nuclei=sheets['DAPI objects'].loc[lambda x:~x.secondary_only & x.unretained.astype(bool)].copy()
        nuclei=nuclei.rename(columns={'well_id':'well','group':'condition','area_px':'value','dapi_object_id':'object_id'})
        values=nuclei.groupby(['condition','well']).value.agg(value='mean',SD='std',n='count').reset_index()
        meta={'kind':'hist','xlabel':'Unretained DAPI object area (pixels)','ylabel':'Objects'}
    elif stem.startswith('FIG_SECONLY_'):
        channel='BIN1' if 'BIN1' in stem else 'RNASEH2B'
        column=channel+('_intensity' if stem.endswith('INTENSITY') else '_spots')
        frames=[]
        for source,label in [('A17 Secondary fields','Sec-Only'),('A17 Biological spot FOVs','Biological')]:
            table=sheets[source].copy()
            if column not in table: continue
            table=table.rename(columns={'arm':'condition',column:'value'})
            table['condition']=label+' '+table.condition
            table=table.merge(sheets['Per nucleus'][['image','well_id']].drop_duplicates(),on='image',how='left',validate='one_to_one').rename(columns={'well_id':'well'})
            frames.append(table)
        nuclei=pd.concat(frames,ignore_index=True)
        values=nuclei.groupby(['condition','well'],dropna=False).value.agg(value='mean',SD='std',n='count').reset_index()
        meta={'kind':'fieldpoints','ylabel':column,'grain':'FOV'}
    elif stem.startswith('FIG_CYTOFLUOROGRAM_') and 'Simple coloc representatives' in sheets:
        sources=sheets['Simple coloc representatives']
        selected=sources.loc[sources.group.map(lambda g:re_safe(g)).eq(stem.removeprefix('FIG_CYTOFLUOROGRAM_'))]
        store=out/'data'/'representative_pixels.npz'
        source_csv=out/'data'/'representative_pixel_sources.csv'
        if len(selected) and store.exists() and source_csv.exists():
            pixels=np.load(store);rows=pd.read_csv(source_csv)
            selected=selected.iloc[0]
            i=rows.index[(rows.image==selected.image)&(rows.nucleus_id==selected.nucleus_id)][0]
            nuclei=pd.DataFrame({'x':pixels[f'{i}_rna'],'value':pixels[f'{i}_partner']}).assign(condition=selected.group,well=selected.well_id,nucleus_id=selected.nucleus_id)
            values=nuclei.groupby(['condition','well']).value.agg(value='mean',SD='std',n='count').reset_index()
            meta={'kind':'hexbin','xlabel':'RNA intensity (raw units)','ylabel':'Partner intensity (raw units)'}
    for column in ['condition','well','value','SD','n']:
        if column not in values: values[column]=np.nan
    return values,nuclei,meta


def re_safe(text):
    import re
    return re.sub(r'[^A-Za-z0-9]+','_',str(text)).strip('_')


def render(path,sheet,out):
    """Render edited values; well-only edits receive Welch, never stale mixed p."""
    from .stats import welch
    import shutil
    path=Path(path).resolve();out=guard_output(Path(out));out.mkdir(parents=True,exist_ok=True)
    metadata=json.loads(path.with_suffix('.json').read_text()) if path.with_suffix('.json').exists() else {}
    with pd.ExcelFile(path) as book:
        names=[sheet] if sheet else list(metadata) if metadata else [n for n in book.sheet_names if not n.endswith('_nuclei')]
        from .endpoints import Endpoint
        definitions=[]; grouped=[]
        for key,entry in metadata.items():
            if entry.get('axis_group') and entry['kind']=='well':
                definitions.append(Endpoint(key,key,'detection','',key,axis_group=entry['axis_group']))
                grouped.append(pd.read_excel(book,sheet_name=key).rename(columns={'condition':'group','well':'well_id','value':'well_mean_of_field_values'}).assign(endpoint=key,axis_series=key))
        union=pd.concat(grouped,ignore_index=True) if grouped else pd.DataFrame()
        for name in names:
            df=pd.read_excel(book,sheet_name=name)
            if not {'condition','well','value'} <= set(df): raise ValueError('required columns: condition, well, value')
            meta=metadata.get(name,{'kind':'well','title':name,'ylabel':'Value'})
            if meta['kind']=='image':
                source=path.parent/meta['image']
                for suffix in ('.png','.svg'):
                    native=source.with_suffix(suffix)
                    if suffix=='.svg' and not native.is_file(): native=source.parent.parent/'svg'/native.name
                    if native.is_file(): shutil.copy2(native,out/(name+suffix))
                continue
            figures.set_style();canvas,ax=figures.plt.subplots(figsize=(4.8,3.6))
            order=df.condition.dropna().astype(str).unique().tolist()
            ctx=SimpleNamespace(group_order=order,reference=order[0],colors=figures.group_colors(order),technical_layer='none',footer=lambda x:x,run_name=path.name)
            if definitions: figures.prepare_axis_groups(ctx,definitions,union)
            kind=meta['kind'];foot='Edited workbook values'
            if kind=='well':
                w=df.rename(columns={'condition':'group','well':'well_id','value':'well_mean_of_field_values'}).assign(endpoint=name)
                contrasts=[]
                for group in order[1:]:
                    result=welch(df.loc[df.condition.eq(group),'value'].dropna().to_numpy(),df.loc[df.condition.eq(order[0]),'value'].dropna().to_numpy())
                    contrasts.append(dict(endpoint=name,test_group=group,reference_group=order[0],**result))
                foot=figures.draw_replicate_simple(ax,ctx,name,w,pd.DataFrame(),pd.DataFrame(),pd.DataFrame(contrasts),meta.get('ylabel','Value'),None)
            else:
                source=pd.read_excel(book,sheet_name=meta['nuclei_sheet']) if meta.get('nuclei_sheet') else df
                for group,rows in source.groupby('condition',sort=False):
                    color=ctx.colors.get(str(group),'#595959')
                    if kind=='scatter': ax.scatter(rows.x,rows.value,s=9,alpha=.45,color=color,label=group)
                    elif kind=='hexbin': ax.hexbin(rows.x,rows.value,gridsize=90,mincnt=1,bins='log',cmap='cividis')
                    elif kind=='fieldpoints': ax.scatter([order.index(str(group))]*len(rows),rows.value,color=color)
                    elif kind=='hist': ax.hist(rows.value.dropna(),bins=24,histtype='step',label=group,color=color)
                    elif kind=='stacked':
                        from .dapi_mask import CLASSES
                        palette=dict(zip(CLASSES,['#595959','#D67AE5','#E69F00']))
                        counts=rows['class'].value_counts(normalize=True)*100
                        bottom=0
                        for label,value in counts.items():
                            ax.bar([order.index(str(group))],[value],bottom=bottom,label=label,color=palette.get(label,'#595959'));bottom+=value
                    elif kind=='ecdf':
                        x=np.sort(rows.value.dropna());ax.step(x,np.arange(1,len(x)+1)/len(x),where='post',label=group,color=color)
                    elif kind=='line':
                        xy=rows.groupby('x').value.mean();ax.plot(xy.index,xy.values,'o-',label=group,color=color)
                    elif kind=='interval':
                        ax.errorbar([order.index(str(group))],rows.value,yerr=[rows.value-rows.ci_low,rows.ci_high-rows.value],fmt='o',capsize=4,color=color)
                ax.set(xlabel=meta.get('xlabel',''),ylabel=meta.get('ylabel',''))
                if kind in ('fieldpoints','stacked','interval'): ax.set_xticks(range(len(order)),order)
                if kind in ('scatter','hist','ecdf','line'): ax.legend(frameon=False,fontsize=7)
                original=ax.get_ylim()
                ax._replicate_simple_axis=lambda variant,ax=ax,original=original:ax.set_ylim(min(0,original[0]) if variant=='full' else original[0],original[1])
            figures.no_box(canvas,ax)
            figures.layout_replicate_simple(canvas,ax,ctx,meta.get('title',name),foot)
            figures.save(canvas,out,name,[],'Workbook values',str(path))
            print(f'Rendered {name}',flush=True)
