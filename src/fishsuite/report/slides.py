"""Workbook-sourced slide preparation and optional python-pptx export."""
from __future__ import annotations

from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter, coordinate_to_tuple

from .aggregate import ReportInputError
from . import figures
from .provenance import guard_output, sha256


def _literal_number(text, identifiers=()):
    # Gene labels such as BIN1 and RNASEH2B are identities, not numeric claims.
    text = str(text)
    # Sam's question identifiers are ordering labels, not measured quantities.
    text = re.sub(r'^(?:1b|[1-4])\.\s+', '', text)
    for identifier in sorted(set(identifiers), key=len, reverse=True):
        text = re.sub(r'(?<!\w)' + re.escape(identifier) + r'(?!\w)', '', text)
    return re.search(r'(?<![\w])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(?:[a-zA-Zµμ%]+)?', text)


def validate_deck(workbook: Path, spec: dict) -> list:
    """Fail closed on untraced literal numbers, stale assets and absent cells."""
    workbook = Path(workbook)
    if not workbook.is_file():
        raise ReportInputError(f'missing workbook: {workbook}')
    if spec.get('workbook_sha256') != sha256(workbook):
        raise ReportInputError('stale workbook hash')
    book = load_workbook(workbook, data_only=True, read_only=True)
    cache={}
    def cell_value(sheet,cell):
        if sheet not in book:
            raise ReportInputError(f'missing workbook sheet: {sheet}')
        if sheet not in cache:
            cache[sheet]=list(book[sheet].values)
        row,col=coordinate_to_tuple(cell)
        values=cache[sheet]
        if row>len(values) or col>len(values[row-1]):
            raise ReportInputError(f'missing workbook cell: {sheet}!{cell}')
        return values[row-1][col-1]
    identifiers = set(book.sheetnames)
    for sheet in ('Per nucleus', 'Per well', 'Micrographs'):
        if sheet in book:
            rows = list(book[sheet].values)
            if len(rows) > 1:
                for index, column in enumerate(rows[1]):
                    if column in ('image', 'stem', 'group', 'condition', 'well_id'):
                        identifiers.update(str(row[index]) for row in rows[2:] if row[index] is not None)
    resolved = []
    try:
        if 'Slide values' in book and spec.get('schema') == 'resolved-deck-1':
            table=list(book['Slide values'].values)
            columns=list(table[1])
            for row in table[2:]:
                entry=dict(zip(columns,row))
                source_sheet,source_cell=entry['source_sheet'],entry['source_cell']
                if source_sheet not in book or not source_cell:
                    raise ReportInputError('missing Slide values source')
                source_value=cell_value(source_sheet,source_cell)
                expected=source_value
                if entry['label']=='MEASURED direction':
                    expected=('higher' if source_value>0 else 'lower' if source_value<0 else 'unchanged') if source_value is not None else 'missing'
                elif entry['label']=='display multiplier':
                    expected=figures.fraction_scale(source_value)
                if entry['value'] != expected:
                    raise ReportInputError(f'stale Slide values lineage: {source_sheet}!{source_cell}')
        for slide in spec.get('slides', []):
            for text in [slide.get('title',''), slide.get('body',''), slide.get('qualification','')]:
                if _literal_number(text, identifiers):
                    raise ReportInputError(f'literal numeric claim must use workbook cells: {text}')
            values = []
            for item in slide.get('values', []):
                if _literal_number(item.get('label', ''), identifiers):
                    raise ReportInputError('literal numeric claim must use workbook cells: ' + item['label'])
                sheet, cell = item['sheet'], item['cell']
                if sheet not in book.sheetnames or not re.fullmatch(r'[A-Z]+[1-9][0-9]*',cell):
                    raise ReportInputError(f'missing workbook cell: {sheet}!{cell}')
                value = cell_value(sheet,cell)
                if value is None and not item.get('allow_missing',False):
                    raise ReportInputError(f'missing workbook value: {sheet}!{cell}')
                if isinstance(value, str) and _literal_number(value, identifiers) and item.get('display', True):
                    raise ReportInputError(f'literal numeric claim must use typed workbook cells: {sheet}!{cell}')
                values.append(dict(item, value=value))
            if slide.get('title_cell'):
                title_ref=slide['title_cell']
                if cell_value(title_ref['sheet'],title_ref['cell']) != slide['title']:
                    raise ReportInputError('stale title claim')
            resolved_assets = []
            for asset in slide.get('figures', []):
                path = Path(asset['path'])
                if not path.is_file():
                    raise ReportInputError(f'missing figure: {path}')
                if sha256(path) != asset.get('sha256'):
                    raise ReportInputError(f'stale figure: {path}')
                if asset.get('cohort') != spec.get('cohort'):
                    raise ReportInputError(f'figure cohort mismatch: {path}')
                caption = ''
                if asset.get('caption_ref'):
                    ref = asset['caption_ref']
                    caption = cell_value(ref['sheet'], ref['cell'])
                    if not isinstance(caption, str) or _literal_number(caption, identifiers):
                        raise ReportInputError('literal numeric or missing micrograph caption; use typed workbook cells')
                resolved_assets.append(dict(asset, caption_value=caption))
            resolved.append(dict(slide, values=values, figures=resolved_assets))
    finally:
        book.close()
    if not resolved:
        raise ReportInputError('missing slide definitions')
    return resolved


def build_deck(workbook: Path, spec: dict, destination: Path) -> Path:
    destination = guard_output(destination)
    sources_path = guard_output(destination.with_name('slide_sources.csv'))
    slides = validate_deck(workbook, spec)
    try:
        from pptx import Presentation
        from pptx.util import Inches, Pt
    except ImportError as exc:
        raise ReportInputError('python-pptx missing; installation prohibited by dispatch') from exc
    ppt = Presentation()
    ppt.slide_width, ppt.slide_height = Inches(13.333333), Inches(7.5)
    sources = []
    for number, definition in enumerate(slides,1):
        slide = ppt.slides.add_slide(ppt.slide_layouts[6])
        box = slide.shapes.add_textbox(Inches(.4), Inches(.15), Inches(12.5), Inches(.95))
        box.text_frame.text = definition['title']
        box.text_frame.word_wrap = True
        for paragraph in box.text_frame.paragraphs:
            paragraph.font.name, paragraph.font.size = 'Arial', Pt(23)
        notes = [str(Path(workbook).resolve()), 'workbook SHA256: '+spec['workbook_sha256']]
        for item in definition['values']:
            location = f'{item["sheet"]}!{item["cell"]}'
            notes.append(f'{item.get("label", "value")}: {item["value"]} | {location}')
            sources.append(dict(slide=number, workbook=str(Path(workbook).resolve()),
                                sheet=item['sheet'], cell=item['cell'], value=item['value']))
        assets = definition.get('figures', [])
        for asset_no, asset in enumerate(assets):
            from PIL import Image
            with Image.open(asset['path']) as img:
                ratio = img.width / img.height
            max_width, max_height = ((9.3, 4.9) if asset_no == 0 else (2.7, 2.2)) if len(assets)>1 else (12.4, 4.9)
            width = min(max_width, max_height*ratio)
            height = width / ratio
            left = .3+(9.3-width)/2 if len(assets)>1 and asset_no==0 else 10+(2.7-width)/2 if len(assets)>1 else (13.333333-width)/2
            top = 1.15+(4.9-height)/2 if asset_no==0 else 1.25+(asset_no-1)*2.5+(2.2-height)/2
            slide.shapes.add_picture(asset['path'], Inches(left), Inches(top), Inches(width), Inches(height))
            if asset.get('caption_value'):
                caption = slide.shapes.add_textbox(Inches(left), Inches(top-.3), Inches(width), Inches(.3))
                caption.text_frame.margin_top = caption.text_frame.margin_bottom = 0
                caption.text_frame.text = asset['caption_value']
                for paragraph in caption.text_frame.paragraphs:
                    paragraph.font.name, paragraph.font.size = 'Arial', Pt(14)
            notes.append(f'Figure: {asset["path"]} | SHA256 {asset["sha256"]}')
            sources.append(dict(slide=number, figure=asset['path'], sha256=asset['sha256'],
                                full_figure=asset.get('full_path', ''), full_sha256=asset.get('full_sha256', '')))
        if not definition.get('figures'):
            content = slide.shapes.add_textbox(Inches(.6), Inches(1.2), Inches(12), Inches(5.5))
            content.text_frame.text = definition.get('body','') + '\n' + '\n'.join(
                f'{v.get("label", "value")}: {v["value"]}' for v in definition['values'] if v.get('display',True))
            content.text_frame.word_wrap = True
            for p in content.text_frame.paragraphs:
                p.font.name, p.font.size = 'Arial', Pt(18)
        readouts = [v['value'] for v in definition['values'] if v.get('label') == 'readout']
        if readouts and assets:
            box = slide.shapes.add_textbox(Inches(.5), Inches(6.25), Inches(12.3), Inches(1.0))
            box.text_frame.word_wrap = True
            box.text_frame.text = str(readouts[0])
            for paragraph in box.text_frame.paragraphs:
                paragraph.font.name, paragraph.font.size = 'Arial', Pt(18)
        slide.notes_slide.notes_text_frame.text = '\n'.join(notes)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ppt.save(destination)
    pd.DataFrame(sources).to_csv(sources_path, index=False)
    return destination


def draw_micrograph_panel(ax, row, color="black"):
    """Display crop with a padded annotation band; source raster is unchanged.

    Omit the bottom six percent containing the verified native annotation, then
    redraw the calibrated bar in padding. Coordinates retain source-pixel scale.
    At the full-width deck placement, all annotation text is at least 18 pt.
    """
    pixels = figures.plt.imread(row['path'])
    height, width = pixels.shape[:2]
    stop = int(height * .94)
    ax.imshow(pixels[:stop], extent=(-.5, width-.5, stop-.5, -.5))
    ax.set_facecolor('black')
    ax.set_xlim(-.5, width-.5)
    ax.set_ylim(height * 1.18, -.5)
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((-.5, stop-.5), width, height * .25,
                           color='black', zorder=2))
    bar_px = float(row['bar_um']) / float(row['voxel_xy_um'])
    right = width * .94
    if bar_px > width * .88:
        raise ReportInputError('calibrated scale bar does not fit micrograph panel')
    ax.plot([right-bar_px, right], [height*1.10]*2, color='white', lw=3,
            solid_capstyle='butt', zorder=3)
    ax.text(right-bar_px/2, height*.99, f"{row['bar_um']:g} µm",
            ha='center', va='center', fontsize=18, color='white', zorder=3)
    ax.set_title(row['group'], fontsize=20, color=color)
    ax.axis('off')


def prepare_deck(template: dict, sheets: dict, out_dir: Path, data: dict,
                 panel, ctx, well: pd.DataFrame, field: pd.DataFrame,
                 contrasts: pd.DataFrame, endpoints: list) -> dict:
    """Resolve semantic endpoint requests only after report rows are available.

    Values copied to Slide values retain their exact origin cell. All contrast
    cells, including missing statistics, enter speaker notes, not just the p.
    """
    from . import figures as fig
    out_dir = guard_output(out_dir)
    for item in template['slides']:
        for key in ('title', 'body', 'qualification'):
            if _literal_number(item.get(key, '')):
                raise ReportInputError('literal numeric claim must use workbook cells: ' + item[key])
    selection = Path(template['nucleus_selection'])
    if not selection.is_file() or sha256(selection) != template['nucleus_selection_sha256']:
        raise ReportInputError('missing or stale nucleus selection source')
    micro_root = Path(template['micrographs'])
    if not micro_root.is_dir():
        raise ReportInputError('missing calibrated micrograph directory')
    selection_text = selection.read_text(encoding='utf-8')
    sheets['Deck specification']=pd.DataFrame([
        dict(title=item['title'],body=item.get('body',''),qualification=item.get('qualification',''))
        for item in template['slides']])
    counts = data['nuclei'].groupby('group',dropna=False).size()
    sel = [dict(stage='retained labels', value=len(data['nuclei']), source='Per nucleus; A0 retained roster'),
           dict(stage='pre-area candidates', value='missing', source=str(selection)),
           dict(stage='pre-border exclusions', value='missing', source=str(selection))]
    sel += [dict(stage=str(group), value=int(count), source='Per nucleus / group') for group,count in counts.items()]
    sheets['Nucleus selection'] = pd.DataFrame(sel)
    sheets['Selection evidence'] = pd.DataFrame([dict(file=str(selection),sha256=sha256(selection),text=selection_text)])
    micro = []
    overlays = panel.tables['overlays_index']
    for group in ctx.group_order:
        choices = overlays.loc[overlays.group==group].sort_values('well_id')
        if choices.empty:
            raise ReportInputError(f'missing micrograph for {group}')
        row = choices.iloc[0]
        path = micro_root / (row['stem']+'__merge_all.png')
        if not path.is_file():
            raise ReportInputError(f'missing micrograph: {path}')
        if sha256(path) != template.get('micrograph_hashes',{}).get(path.name):
            raise ReportInputError(f'missing or stale pinned micrograph: {path}')
        nuc = data['nuclei'].loc[data['nuclei'].image==row['image']]
        vox = pd.to_numeric(nuc.voxel_xy_um).dropna().unique()
        if len(vox)!=1 or not np.isfinite(vox[0]) or vox[0]<=0:
            raise ReportInputError('missing or ambiguous micrograph calibration')
        planes=panel.tables['per_nucleus'].loc[lambda x:x.image==row['image'],'z_plane'].dropna().unique()
        if len(planes)!=1 or planes[0]!=row['z_plane']:
            raise ReportInputError('micrograph plane differs from persisted nuclear plane')
        from PIL import Image
        with Image.open(path) as image:
            width,height = image.size
            pixels=np.asarray(image.convert('RGB'))
        bar_um=template.get('micrograph_bar_um',{}).get(path.name)
        if bar_um is None or not np.isfinite(bar_um) or bar_um<=0:
            raise ReportInputError('missing native micrograph scale-bar annotation')
        # The named native PNGs already carry a lower-right white scale bar.
        # Verify its raster length against the source voxel calibration, and
        # retain it instead of drawing a second, differently sized bar.
        bright=(pixels[int(height*.9):,int(width*.7):,:]>240).all(axis=2)
        runs=[]
        for scan in bright:
            transitions=np.diff(np.r_[False,scan,False].astype(int))
            runs.extend(np.flatnonzero(transitions==-1)-np.flatnonzero(transitions==1))
        native_bar_px=int(max(runs,default=0))
        if abs(native_bar_px-bar_um/float(vox[0]))>1:
            raise ReportInputError('native micrograph scale-bar length disagrees with source calibration')
        # Native image dimensions must equal the persisted mask geometry; a scaled
        # preview cannot inherit source micrometres-per-pixel without a transform.
        masks = [m for m in panel.manifest.get('persisted_masks',[]) if m['image']==row['image']]
        if len(masks)!=1 or masks[0]['shape'] != [height,width]:
            raise ReportInputError('micrograph calibration mismatch: image dimensions differ from retained mask')
        if sha256(Path(masks[0]['path'])) != masks[0]['sha256']:
            raise ReportInputError('stale micrograph calibration mask')
        micro.append(dict(group=group,image=row['image'],path=str(path.resolve()),sha256=sha256(path),
                          voxel_xy_um=float(vox[0]),bar_um=float(bar_um),native_bar_px=native_bar_px,z_plane=int(row['z_plane']),
                          width_px=width,height_px=height,calibration_source='nuclei_metrics.csv:voxel_xy_um',
                          plane_source='coloc_standard_panel.xlsx:overlays_index/z_plane',
                          selection='first persisted representative well in group; presentation only'))
    sheets['Micrographs'] = pd.DataFrame(micro)
    from .aggregate import reconcile_localization
    spots = pd.read_csv(ctx.run_dir/'spot_metrics.csv')
    audit = reconcile_localization(data['nuclei'], spots)
    if audit['status'] != 'ok':
        raise ReportInputError('Localization reconciliation blocked: ' + '; '.join(audit['errors']))
    for key, name in [('per_nucleus','Localization counts'), ('unassigned','Localization unassigned'),
                      ('checks','Localization checks'), ('territory','Localization territory')]:
        sheets[name] = audit[key]
    localization = fig.render_localization(ctx, well, field, data['nuclei'], spots, contrasts,
                                            out_dir/'localization', pub_dir=micro_root)
    sheets['Localization crops'] = pd.DataFrame(localization['crops'])
    sheets['Endpoint coverage'] = (field.groupby(['endpoint','group'],as_index=False)
        .agg(defined_nuclei=('n_nuclei_nonmissing','sum'),
             eligible_nuclei=('n_nuclei_total','sum'),fovs=('image','nunique')))
    by_endpoint = {e.name:e for e in endpoints}
    values, figure_rows, resolved = [], [], []
    figure_dir = guard_output(out_dir/'figures')
    figure_dir.mkdir(parents=True,exist_ok=True)
    fig.set_style()
    for slide_no, item in enumerate(template['slides'],1):
        refs = []
        values.append(dict(slide=slide_no,endpoint='',label='title',value=item['title'],
                           source_sheet='Deck specification',source_cell=f'A{slide_no+2}'))
        title_ref=dict(sheet='Slide values',cell=f'D{len(values)+2}')
        refs.append(dict(**title_ref,label='title',display=False))
        refs.append(dict(sheet='Deck specification',cell=f'A{slide_no+2}',label='title source',display=False))
        names = item.get('endpoints',[])
        for name in names:
            match = contrasts.index[contrasts.endpoint==name].tolist()
            if len(match)!=1:
                raise ReportInputError(f'missing or ambiguous slide endpoint: {name}')
            i = match[0]
            for col_index, col in enumerate(contrasts.columns,1):
                value = contrasts.loc[i,col]
                values.append(dict(slide=slide_no,endpoint=name,label=col,value=value,
                                   source_sheet='Contrasts',source_cell=f'{get_column_letter(col_index)}{i+3}'))
                refs.append(dict(sheet='Slide values',cell=f'D{len(values)+2}',label=f'{name}: {col}',allow_missing=True,display=False))
                refs.append(dict(sheet='Contrasts',cell=f'{get_column_letter(col_index)}{i+3}',label=f'{name}: {col}',allow_missing=True,display=False))
            direction = 'higher' if contrasts.loc[i,'diff']>0 else 'lower' if contrasts.loc[i,'diff']<0 else 'unchanged' if contrasts.loc[i,'diff']==0 else 'missing'
            values.append(dict(slide=slide_no,endpoint=name,label='MEASURED direction',value=direction,
                               source_sheet='Contrasts',source_cell=f'{get_column_letter(contrasts.columns.get_loc("diff")+1)}{i+3}'))
            refs.append(dict(sheet='Slide values',cell=f'D{len(values)+2}',label=f'{name}: MEASURED direction',display=True))
            for wi,row in well.loc[well.endpoint==name].iterrows():
                # iterrows retains the aggregation index, while the workbook uses
                # positional row order. Resolve position explicitly.
                position=well.index.get_loc(wi)
                for column in ('group','well_id','well_mean_of_field_values'):
                    cell=f'{get_column_letter(well.columns.get_loc(column)+1)}{position+3}'
                    refs.append(dict(sheet='Per well',cell=cell,label=f'{name}: {column}',allow_missing=True,display=False))
            values.append(dict(slide=slide_no,endpoint=name,label='display multiplier',
                               value=fig.fraction_scale(name),
                               source_sheet='Endpoint definitions',source_cell=f'A{list(by_endpoint).index(name)+3}'))
            refs.append(dict(sheet='Slide values',cell=f'D{len(values)+2}',label=f'{name}: display multiplier',display=False))
        extra_sheets = item.get('sheets',[])
        if names:
            coverage = sheets['Endpoint coverage']
            for ci,row in coverage.loc[coverage.endpoint.isin(names)].iterrows():
                for column_index,column in enumerate(coverage.columns,1):
                    refs.append(dict(sheet='Endpoint coverage',cell=f'{get_column_letter(column_index)}{ci+3}',
                                     label=f'{row.endpoint}: {row.group}: {column}',display=False))
        for sheet_name in extra_sheets:
            if sheet_name not in sheets:
                raise ReportInputError(f'missing requested slide sheet: {sheet_name}')
            for row_i, row in enumerate(sheets[sheet_name].itertuples(index=False,name=None),3):
                for col_i,value in enumerate(row,1):
                    refs.append(dict(sheet=sheet_name,cell=f'{get_column_letter(col_i)}{row_i}',
                                     label=f'{sheet_name}: {sheets[sheet_name].columns[col_i-1]}',allow_missing=True,display=False))
        assets = []
        stem = item['asset']
        if item.get('kind') == 'localization':
            import shutil
            for variant in ('focus', 'full'):
                for suffix in ('.png', '.svg'):
                    shutil.copyfile(out_dir/'localization'/('FIG_LOCALIZATION_'+variant+suffix), guard_output(figure_dir/(stem+'_'+variant+suffix)))
            path = figure_dir/(stem+'_focus.png')
            assets = [dict(path=str(path.resolve()), sha256=sha256(path), cohort=template['cohort'])]
        elif names:
            count = len(names)
            cols = min(count,3)
            rows = (count+cols-1)//cols
            if item.get('headline'):
                f = fig.plt.figure(figsize=(12.4,5.6))
                secondary_cols = (count - 1 + 1) // 2
                grid = f.add_gridspec(2, secondary_cols + 1, width_ratios=[1.7] + [1] * secondary_cols)
                axes = np.array([f.add_subplot(grid[:, 0])] +
                                [f.add_subplot(grid[i // secondary_cols, 1 + i % secondary_cols])
                                 for i in range(count - 1)])
            else:
                f, axes = fig.plt.subplots(rows,cols,figsize=(12.4,5.6),squeeze=False)
            f.subplots_adjust(left=.08,right=.98,top=.83,bottom=.17,hspace=.7,wspace=.65)
            for ax,name in zip(axes.flat,names):
                e = by_endpoint[name]
                scale = fig.fraction_scale(name)
                unit=e.unit
                if 'observed divided' in unit or 'enrichment' in name:
                    unit='enrichment ratio'
                elif 'fraction' in unit:
                    unit='fraction'
                elif 'assigned cell territory' in unit:
                    unit='puncta per nucleus and assigned cell territory'
                elif 'puncta per nucleus' in unit:
                    unit='puncta / nucleus'
                elif 'square micrometre of nuclear area' in unit:
                    unit='puncta / square micrometre'
                if scale==100:
                    unit='percent (%)'
                if 'minus_shuffle' in name:
                    unit='excess over shuffle (percentage points)'
                fig.draw_replicate_simple(ax,ctx,name,well,field,data['nuclei'],contrasts,
                                          unit,e.column,scale=scale,compact=True)
                title = e.pretty(ctx.channel_labels)
                if item.get('headline'):
                    title = {
                        'rna1_enrichment_at_partner_puncta': 'BIN1 signal: relative enrichment',
                        'rna1_rotation_enrichment_at_partner_puncta': 'BIN1 signal: rotation enrichment',
                        'frac_called_coloc_partner_runthr': 'BIN1 threshold calls',
                        'frac_called_coloc_partner_minus_shuffle_runthr': 'BIN1 calls: excess over shuffle',
                        'paired_fraction_partner_at_0p3um': 'RNASEH2B puncta paired with BIN1',
                        'paired_frac_partner_at_rna1_shuffle': 'RNASEH2B pairing: shuffle baseline',
                        'paired_frac_rna1_at_partner': 'BIN1 puncta paired with RNASEH2B',
                        'paired_frac_rna1_at_partner_shuffle': 'BIN1 pairing: shuffle baseline',
                    }.get(name, title)
                if name == 'paired_frac_rna1_at_partner_minus_shuffle':
                    title = 'BIN1 puncta paired with an RNASEH2B punctum: excess over shuffle'
                elif name == 'paired_frac_partner_at_rna1_minus_shuffle':
                    title = 'RNASEH2B puncta paired with a BIN1 punctum: excess over shuffle'
                if name == 'rna1_nuclear_spot_fraction':
                    ax.set_ylabel('BIN1 intron puncta: % nuclear (per nucleus)')
                    title = 'BIN1 intron puncta, % nuclear'
                import textwrap
                ax.set_title('\n'.join(textwrap.wrap(title,28 if item.get('headline') else 38)),fontsize=9,pad=22)
                fig.no_box(f,ax)
            spare_axes = list(axes.flat)[count:]
            if item.get('kind') == 'standard':
                profiles = sheets['Coloc line profiles']
                channel_colors = dict(fig.read_luts(ctx.run_dir, micro_root)[0])
                for ax, group in zip(spare_axes, ctx.group_order):
                    samples = profiles.loc[profiles.group.eq(group)]
                    first = samples[['image','nucleus_id']].drop_duplicates().iloc[0]
                    samples = samples.loc[samples.image.eq(first.image) & samples.nucleus_id.eq(first.nucleus_id)].sort_values('sample_index')
                    for channel, label in [('rna1_norm',ctx.channel_labels['rna1']),
                                            ('partner_norm',ctx.channel_labels['protein'])]:
                        color = channel_colors[label]
                        ax.plot(samples.distance_um,samples[channel],label=label,color=color)
                    ax.set(xlabel='Distance (micrometres)', ylabel='Normalized intensity', title=group+' recorded profile')
                    ax.legend(fontsize=6)
                    fig.no_box(f,ax)
            else:
                for ax in spare_axes:
                    ax.set_visible(False)
            f.text(.5,.04,'Persisted measurements; defined-value/usability masks; well means; raw Welch p.\n'
                   'Holm, Hedges g, MDE and endpoint n are traced in speaker notes. MDE is not an exclusion bound.',
                   ha='center',fontsize=8,color='#595959')
            if item.get('qualification'):
                f.text(.5,.95,item['qualification'],ha='center',fontsize=9,color='#595959')
            records=[]
            record = fig.save(f,figure_dir,stem,records,item['title'],'Per well / Contrasts')
            path=figure_dir/record['png']
            assets=[dict(path=str(path.resolve()),sha256=sha256(path),cohort=template['cohort'])]
        elif item.get('kind')=='micrographs':
            f,axes=fig.plt.subplots(1,len(micro),figsize=(12.4,5.6),squeeze=False)
            for ax,row in zip(axes.flat,micro):
                draw_micrograph_panel(ax, row, ctx.colors[row['group']])
            f.subplots_adjust(left=.02,right=.98,top=.92,bottom=.04)
            records=[]
            fig.save(f,figure_dir,stem,records,item['title'],'Micrographs')
            path=figure_dir/f'{stem}.png'
            assets=[dict(path=str(path.resolve()),sha256=sha256(path),cohort=template['cohort'])]
        else:
            import textwrap
            f = fig.plt.figure(figsize=(12.4,5.6))
            f.text(.06,.88,item['title'],fontsize=18)
            f.text(.06,.72,'\n'.join(textwrap.wrap(item.get('body',''),95)),fontsize=13,va='top',linespacing=1.6)
            if item.get('kind') == 'selection':
                rows = sheets['Nucleus selection']
                f.text(.06,.32,'\n'.join(f'{r.stage}: {r.value}' for r in rows.itertuples()),fontsize=12,va='top')
            fig.save(f,figure_dir,stem,[],item['title'],'; '.join(extra_sheets))
            path=figure_dir/f'{stem}.png'
            assets=[dict(path=str(path.resolve()),sha256=sha256(path),cohort=template['cohort'])]
        if assets and Path(assets[0]['path']).stem.endswith('_focus'):
            full_path = Path(assets[0]['path']).with_name(stem+'_full.png')
            assets[0].update(full_path=str(full_path.resolve()), full_sha256=sha256(full_path))
        if item.get('micrographs'):
            assets.extend(dict(path=row['path'],sha256=row['sha256'],cohort=template['cohort'],
                               caption_ref=dict(sheet='Micrographs',cell=f'A{i+3}')) for i,row in enumerate(micro))
        for asset in assets:
            figure_rows.append(dict(slide=slide_no,**asset,source_sheet='Contrasts' if names else 'Micrographs',
                                    source_cells=';'.join(f'{r["sheet"]}!{r["cell"]}' for r in refs),
                                    endpoints=';'.join(names),filter='all retained; endpoint finite/usability masks',
                                    test='two-sided Welch on well means'))
        # Non-figure text is workbook material too: no untraced scientific literals.
        if item.get('body'):
            values.append(dict(slide=slide_no,endpoint='',label='readout',value=item['body'],
                               source_sheet='Deck specification',source_cell=f'B{slide_no+2}'))
            refs.append(dict(sheet='Slide values',cell=f'D{len(values)+2}',label='readout',display=True))
        resolved.append(dict(identity=item['identity'],title=item['title'],title_cell=title_ref,values=refs,figures=assets))
    sheets['Slide values']=pd.DataFrame(values)
    sheets['Figure sources']=pd.DataFrame(figure_rows)
    return dict(schema='resolved-deck-1',cohort=template['cohort'],slides=resolved)
