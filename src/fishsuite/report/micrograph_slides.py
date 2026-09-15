"""Select per-well FOVs and reuse exact native publication-image panels.

No image acquisition, segmentation, detection or null analysis runs here. Saved
panels retain the publication renderer's recorded display windows and LUTs.
"""
from pathlib import Path
import json
import re
import shutil

import numpy as np
import pandas as pd

from .provenance import guard_output, sha256


def select_per_well_fovs(fields, well_groups=None):
    """Nearest median nucleus count; ties resolve lexically by full image ID."""
    fields = fields.copy()
    if 'well_id' not in fields:
        fields['well_id'] = fields['condition']
    if well_groups is None:
        well_groups = {}
        for well in fields.well_id.astype(str).unique():
            match = re.fullmatch(r'(WT|KO|NT|KD)_(\d+)', well, re.I)
            if match:
                well_groups[well] = {'WT':'WT','KO':'QKI-KO','NT':'NT','KD':'KD'}[match[1].upper()]
    fields = fields.loc[fields.well_id.isin(well_groups)].copy()
    fields['group'] = fields.well_id.map(well_groups)
    columns = ['image','condition','source_condition','source_path','output_stem','well_id','group','nuclei_analyzed',
               'voxel_xy_nm','z_plane','z_autofocus_mode','z_autofocus_channel_used']
    fields = fields[[c for c in columns if c in fields]]
    fields['nuclei_analyzed'] = pd.to_numeric(fields.nuclei_analyzed, errors='raise')
    if fields.empty or not np.isfinite(fields.nuclei_analyzed).all():
        raise ValueError('missing valid biological FOV nucleus counts')
    selected = []
    for well, rows in fields.groupby('well_id',sort=True):
        median = float(rows.nuclei_analyzed.median())
        rows = rows.assign(median_distance=(rows.nuclei_analyzed-median).abs())
        row = rows.sort_values(['median_distance','image'],kind='stable').iloc[0].to_dict()
        row.update(well_median_nuclei=median,
                   selection='closest well-median retained nucleus count; full image ID lexical tie-break')
        selected.append(row)
    return pd.DataFrame(selected)


def pair_wells(selection, group_order=None):
    """Return matched ordinal well pairs, preserving the requested arm order."""
    groups = selection.group.unique().tolist()
    if group_order is None:
        group_order = [g for g in ('WT','QKI-KO','NT','KD') if g in groups]
    group_order = list(group_order)
    if len(group_order) != 2 or set(groups) != set(group_order):
        raise ValueError('unpaired biological wells: two complete arms required')
    by_pair = {}
    for row in selection.to_dict('records'):
        match = re.search(r'(\d+)$',str(row['well_id']))
        if not match:
            raise ValueError('missing numeric well pairing suffix: '+str(row['well_id']))
        pair = int(match[1])
        if row['group'] in by_pair.setdefault(pair,{}):
            raise ValueError('ambiguous biological well pair')
        by_pair[pair][row['group']] = row
    result = []
    for pair, rows in sorted(by_pair.items()):
        if set(rows) != set(group_order):
            raise ValueError(f'unpaired biological well: {pair}')
        result.append([rows[g] for g in group_order])
    return result


def _native_stem(pub_dir, image, well):
    """Resolve one flat named publication directory; never search raw trees."""
    source_stem = Path(image).stem.replace(' ','_')
    candidates = []
    from fishsuite.core.output import sanitize_condition_for_filename
    prefix = sanitize_condition_for_filename(str(well))
    for path in pub_dir.glob(prefix+'__*__merge_all.png'):
        stem = path.name.removesuffix('__merge_all.png')
        short = stem.split('__',1)[1]
        if source_stem.endswith(short):
            candidates.append(stem)
    if len(candidates) != 1:
        raise FileNotFoundError(f'missing or ambiguous exact publication image: {well} / {image}')
    return candidates[0]


def _scale_bar(path, voxel_xy_um):
    """Measure the existing lower-right native white bar, in calibrated pixels."""
    from PIL import Image
    with Image.open(path) as im:
        pixels = np.asarray(im.convert('RGB'))
    height,width = pixels.shape[:2]
    bright = (pixels[int(height*.9):,int(width*.7):,:]>240).all(axis=2)
    runs = []
    for scan in bright:
        transitions = np.diff(np.r_[False,scan,False].astype(int))
        runs.extend(np.flatnonzero(transitions==-1)-np.flatnonzero(transitions==1))
    bar_px = int(max(runs,default=0))
    if bar_px < 1 or not np.isfinite(voxel_xy_um) or voxel_xy_um <= 0:
        raise ValueError('missing native calibrated scale bar: '+str(path))
    return dict(width_px=width,height_px=height,native_bar_px=bar_px,
                voxel_xy_um=voxel_xy_um,bar_um=bar_px*voxel_xy_um)


def prepare_per_well_micrographs(run_dir, out_dir, nuclei=None, group_order=None,
                                well_groups=None):
    """Return slide rows containing four independent image objects plus QC.

    ``nuclei`` optionally supplies the report's retained nucleus roster and its
    group/well assignments. Missing saved panels raise explicitly; the MIAT
    caller may retain its prior micrograph slide on this exception.
    """
    from .figures import publication_panel_paths
    run_dir = Path(run_dir)
    out_dir = guard_output(Path(out_dir))
    cfg = json.loads((run_dir/'run_config.json').read_text(encoding='utf-8'))['config_resolved']
    fields = pd.read_csv(run_dir/'per_image_summary.csv')
    if nuclei is not None:
        counts = nuclei.groupby('image').size().rename('nuclei_analyzed')
        fields = fields.drop(columns='nuclei_analyzed').merge(counts,on='image',how='inner')
        if {'well_id','group'}.issubset(nuclei.columns):
            assignments = nuclei[['image','well_id','group']].drop_duplicates()
            if assignments.image.duplicated().any():
                raise ValueError('ambiguous report well assignment')
            fields = fields.drop(columns=['well_id','group'], errors='ignore').merge(assignments,on='image',validate='one_to_one')
            well_groups = dict(zip(assignments.well_id,assignments.group))
    selection = select_per_well_fovs(fields,well_groups)
    pairs = pair_wells(selection,group_order)
    pub_dir = run_dir/'publication_images'
    planned = []
    for rows in pairs:
        slide = dict(title=rows[0]['well_id']+' vs '+rows[1]['well_id'],rows=[])
        for row in rows:
            stem = row.get('output_stem') or _native_stem(pub_dir,row['image'],row.get('source_condition', row['condition']))
            panels = publication_panel_paths(pub_dir,stem,cfg)
            for panel in panels:
                panel['display_mode']=cfg['output'].get('pub_contrast_mode')
                for role in ('dapi','rna','rna2','antibody'):
                    for bound in ('min','max'):
                        key=f'manual_{role}_{bound}'
                        if key in cfg['output']: panel[key]=cfg['output'][key]
            voxel = float(row['voxel_xy_nm'])/1000 if 'voxel_xy_nm' in row else float('nan')
            if not np.isfinite(voxel) and nuclei is not None:
                values = nuclei.loc[nuclei.image==row['image'],'voxel_xy_um'].dropna().unique()
                if len(values)==1:
                    voxel = float(values[0])
            calibration = _scale_bar(panels[0]['path'],voxel)
            slide['rows'].append(dict(row,stem=stem,panels=[dict(p,**calibration) for p in panels]))
        planned.append(slide)
    out_dir.mkdir(parents=True,exist_ok=True)
    for slide in planned:
        for row in slide['rows']:
            for panel in row['panels']:
                source = Path(panel['path']); dest = out_dir/source.name
                shutil.copy2(source,dest)
                panel.update(source_path=str(source),path=str(dest.resolve()))
    qc, missing = [], []
    for row in planned[0]['rows']:
        paths = [run_dir/'qc_overlays'/(row['stem']+suffix) for suffix in
                 ('__qc_dapi_rna1_rna2_nuclei_spots.png','__qc_dapi_rna_nuclei_spots.png')]
        source = next((p for p in paths if p.is_file()),None)
        if source is None:
            missing.append('missing segmentation and detection QC overlay: '+row['image'])
            continue
        dest = out_dir/source.name
        shutil.copy2(source,dest)
        qc.append(dict(group=row['group'],well_id=row['well_id'],image=row['image'],
                       path=str(dest.resolve()),source_path=str(source),sha256=sha256(dest)))
    selection.to_csv(out_dir/'per_well_fov_selection.csv',index=False)
    result = dict(slides=planned,selection=selection.to_dict('records'),qc=qc,missing=missing,
                  publication_source=str(pub_dir),rendering='exact saved publication-image PNGs; native recorded windows/LUTs')
    (out_dir/'micrograph_slides.json').write_text(json.dumps(result,indent=2,default=str),encoding='utf-8')
    return result
