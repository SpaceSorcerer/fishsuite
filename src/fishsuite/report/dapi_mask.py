"""DAPI-positive report-time QC; never changes the segmentation of record.

DAPI membership has priority. DAPI-positive spots outside retained masks
are excluded from retained-cell denominators and annotated with their object's
overlap status, including extensions of objects that overlap retained masks.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.measure import regionprops
from .endpoints import Endpoint
from .provenance import guard_output, sha256

COUNT = 'extranuclear_spot_count_per_cell_territory'
FRACTION = 'nuclear_spot_fraction_dapi'
CLASSES = ('in_retained_nucleus', 'in_unretained_dapi_object', 'extranuclear')
CACHE_TABLES = dict(spots='spot_localization_dapi.csv', objects='dapi_objects.csv',
    census='dapi_object_census.csv', nuclei='dapi_per_nucleus.csv',
    sensitivity='dapi_sensitivity_per_object.csv', arms='dapi_census_by_arm.csv')


def seal_dapi_cache(out_dir, run_dir):
    """Seal completed QC tables for resumable figure/report builds."""
    out_dir, run_dir = Path(out_dir), Path(run_dir)
    prov = json.loads((out_dir/'dapi_provenance.json').read_text())
    assert sha256(run_dir/'spot_metrics.csv') == prov['source_spots_sha256']
    for row in prov['sources']:
        assert sha256(Path(row['mask_path'])) == row['mask_sha256']
    source = pd.read_csv(run_dir/'spot_metrics.csv')
    cached = pd.read_csv(out_dir/CACHE_TABLES['spots'])
    positive = cached.dapi_object_id.gt(0)
    expected_classes = np.select([positive & cached.retained_mask_id_at_spot.gt(0),positive],CLASSES[:2],default=CLASSES[2])
    assert np.array_equal(cached['class'],expected_classes), 'stale DAPI-boundary classification'
    expected = source.loc[source.channel.eq('rna1'),['image','spot_id']].sort_values(['image','spot_id']).reset_index(drop=True)
    pd.testing.assert_frame_equal(cached[['image','spot_id']].sort_values(['image','spot_id']).reset_index(drop=True), expected)
    record = dict(run_dir=str(run_dir.resolve()), contract='gaussian1_otsu_fill_4connected_area200_v1',
        files={name:sha256(out_dir/name) for name in [*CACHE_TABLES.values(),'dapi_provenance.json']},
        source_files={name:sha256(run_dir/name) for name in ['run_config.json','spot_metrics.csv','nuclei_metrics.csv','per_image_summary.csv']})
    (out_dir/'dapi_cache_manifest.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print('DAPI cache sealed: complete RNA1 identity roster and persisted mask hashes verified.',flush=True)


def endpoints():
    return [Endpoint(COUNT, COUNT, 'localization', 'puncta per assigned cell territory',
                     '{rna1} extranuclear puncta per cell territory', source='report-time DAPI QC'),
            Endpoint(FRACTION, FRACTION, 'localization', 'fraction',
                     '{rna1} nuclear fraction, DAPI corrected', source='report-time DAPI QC')]


def enforce_dapi_boundary(out_dir, run_dir):
    """Migrate complete QC tables to strict outside-DAPI=extranuclear membership."""
    out_dir = Path(out_dir)
    sp = pd.read_csv(out_dir/CACHE_TABLES['spots'])
    positive = sp.dapi_object_id.gt(0)
    sp['class'] = np.select([positive & sp.retained_mask_id_at_spot.gt(0),positive],CLASSES[:2],default=CLASSES[2])
    n = pd.read_csv(out_dir/CACHE_TABLES['nuclei'])
    census = pd.read_csv(out_dir/CACHE_TABLES['census']).set_index('image')
    for i,r in n.iterrows():
        rows = sp.loc[sp.image.eq(r.image)]
        nuclear = int((rows.retained_mask_id_at_spot.eq(r.nucleus_id) & rows['class'].eq(CLASSES[0])).sum())
        extra = int((rows.nucleus_id.eq(r.nucleus_id) & rows['class'].eq(CLASSES[2])).sum())
        defined = census.loc[r.image,'dapi_objects'] > 0
        n.loc[i,'nuclear_spot_count_dapi'] = nuclear
        n.loc[i,COUNT] = extra if defined else np.nan
        n.loc[i,FRACTION] = nuclear/(nuclear+extra) if defined and nuclear+extra else np.nan
    sp.to_csv(out_dir/CACHE_TABLES['spots'],index=False)
    n.to_csv(out_dir/CACHE_TABLES['nuclei'],index=False)
    prov_path = out_dir/'dapi_provenance.json'
    prov = json.loads(prov_path.read_text())
    prov['classification'] = 'Outside DAPI is extranuclear; inside DAPI is retained only when also inside a retained mask.'
    prov_path.write_text(json.dumps(prov,indent=2),encoding='utf-8')
    seal_dapi_cache(out_dir,run_dir)


def classify_plane(dapi, retained, spots, nucleus_ids, voxel_um, min_area_px=16000):
    """Classify centroids; zero-total or absent-DAPI endpoints are NA.

    Sensitivity assigns extranuclear spots to nearest DAPI object (Euclidean
    nearest positive pixel, unlimited radius); it is not a cell segmentation.
    Distances use the DAPI binary 0.5 contour, in calibrated micrometres.
    """
    from skimage.measure import find_contours
    dapi, retained = np.asarray(dapi), np.asarray(retained)
    if dapi.ndim != 2 or retained.shape != dapi.shape:
        raise ValueError('DAPI and retained label geometry must be identical 2D planes')
    if not np.isfinite(dapi).all() or not np.isfinite(voxel_um) or voxel_um <= 0:
        raise ValueError('nonfinite DAPI or missing calibration')
    if spots.spot_id.duplicated().any():
        raise ValueError('duplicate spot identities')
    smooth = ndi.gaussian_filter(dapi.astype(float), sigma=1.)
    threshold = float(threshold_otsu(smooth)) if smooth.max() > smooth.min() else np.nan
    positive = ndi.binary_fill_holes(smooth > threshold) if np.isfinite(threshold) else np.zeros(dapi.shape, bool)
    lab, _ = ndi.label(positive)
    sizes = np.bincount(lab.ravel())
    positive &= sizes[lab] >= 200
    objects, number = ndi.label(positive)
    rows = spots[[c for c in ['image','condition','spot_id','nucleus_id','x_px','y_px',
                             'in_nucleus','in_cytoplasm'] if c in spots]].copy()
    xy = rows[['y_px', 'x_px']].to_numpy(float)
    if not np.isfinite(xy).all() or (xy < 0).any() or (xy >= np.array(dapi.shape)).any():
        raise ValueError('spot outside image or missing coordinates')
    ij = np.floor(xy + .5).astype(int)
    ij = np.minimum(ij, np.array(dapi.shape)-1)
    yy, xx = ij.T
    mask_ids = retained[yy, xx].astype(int)
    object_ids = objects[yy, xx].astype(int)
    rows['retained_mask_id_at_spot'] = mask_ids
    rows['dapi_object_id'] = object_ids
    rows['class'] = np.select([(mask_ids > 0) & (object_ids > 0), object_ids > 0], CLASSES[:2], default=CLASSES[2])
    distances = []
    if number:
        curves = find_contours(np.pad(positive.astype(float), 1), .5)
        a = np.concatenate([c[:-1] for c in curves]) - 1
        b = np.concatenate([c[1:] for c in curves]) - 1
        v = b-a
        lengths = np.sum(v*v, axis=1)
        for point in xy:
            t = np.clip(np.sum((point-a)*v, axis=1)/lengths, 0, 1)
            distances.append(float(np.linalg.norm(point-(a+t[:,None]*v), axis=1).min())*voxel_um)
    else:
        distances = [np.nan]*len(rows)
    rows['distance_to_nearest_dapi_edge_um'] = distances
    records = []
    for obj in regionprops(objects):
        overlap, counts = np.unique(retained[objects == obj.label], return_counts=True)
        overlap_counts = {str(int(i)): int(n) for i, n in zip(overlap, counts) if i > 0}
        records.append(dict(dapi_object_id=obj.label, area_px=int(obj.area), area_um2=obj.area*voxel_um**2,
            retained_nuclei_count=len(overlap_counts), retained_label_overlap_px=json.dumps(overlap_counts),
            unretained=not overlap_counts, below_min_area_px=obj.area < min_area_px,
            bin1_spots=int((object_ids == obj.label).sum())))
    census = pd.DataFrame(records, columns=['dapi_object_id','area_px','area_um2','retained_nuclei_count',
        'retained_label_overlap_px','unretained','below_min_area_px','bin1_spots'])
    rows['dapi_object_has_retained_overlap'] = rows.dapi_object_id.map(
        census.set_index('dapi_object_id').retained_nuclei_count.gt(0)).fillna(False)
    ns = []
    for nid in nucleus_ids:
        nuclear = int(((mask_ids == nid) & rows['class'].eq(CLASSES[0])).sum())
        extra = int((rows.nucleus_id.eq(nid) & rows['class'].eq(CLASSES[2])).sum())
        ns.append(dict(nucleus_id=nid, nuclear_spot_count_dapi=nuclear,
                       **{COUNT: extra if number else np.nan,
                          FRACTION: nuclear/(nuclear+extra) if number and nuclear+extra else np.nan}))
    sensitivity = []
    if number:
        nearest = ndi.distance_transform_edt(~positive, return_distances=False, return_indices=True)
        assigned = objects[nearest[0, yy, xx], nearest[1, yy, xx]]
        for oid in range(1, number+1):
            nuclear = int((object_ids == oid).sum())
            extra = int(((object_ids == 0) & (assigned == oid)).sum())
            sensitivity.append(dict(nucleus_id=oid, **{COUNT: extra,
                FRACTION: nuclear/(nuclear+extra) if nuclear+extra else np.nan}))
    return dict(spots=rows, objects=census, nuclei=pd.DataFrame(ns),
                sensitivity=pd.DataFrame(sensitivity, columns=['nucleus_id', COUNT, FRACTION]),
                mask=objects, threshold=threshold)


def source_acquisition(staging, row):
    """Follow recorded paths, or a source-relative field ID, before legacy folders."""
    recorded = getattr(row, 'source_path', None)
    if recorded and pd.notna(recorded):
        source = Path(recorded)
    elif Path(str(row.image)).parent != Path('.'):
        source = Path(staging) / str(row.image)
    else:
        name = str(row.image)
        folder = row.condition if not row.secondary_only else ('SecOnly_WT' if 'WT' in name else 'SecOnly_KO')
        source = Path(staging) / folder / name
        if not source.is_file() and (Path(staging)/name).is_file():
            source = Path(staging)/name
    if not source.is_file():
        raise FileNotFoundError(f'missing explicitly rostered acquisition: {source}')
    return source


def build_dapi(run_dir, out_dir, data):
    """Read only named staged acquisitions and persisted masks from the roster."""
    import tifffile
    from fishsuite.core.io import read_image, extract_channel_at_z
    run_dir, out_dir = Path(run_dir), guard_output(Path(out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir/'dapi_cache_manifest.json'
    if cache.is_file():
        record = json.loads(cache.read_text())
        assert record['run_dir'] == str(run_dir.resolve())
        assert record['contract'] == 'gaussian1_otsu_fill_4connected_area200_v1'
        for name, digest in record['files'].items():
            assert sha256(out_dir/name) == digest, f'stale DAPI cache: {name}'
        for name, digest in record['source_files'].items():
            assert sha256(run_dir/name) == digest, f'changed DAPI source: {name}'
        prov = json.loads((out_dir/'dapi_provenance.json').read_text())
        for row in prov['sources']:
            assert sha256(Path(row['mask_path'])) == row['mask_sha256']
        print('DAPI cache reused after table/source/mask SHA256 verification.',flush=True)
        return {key:pd.read_csv(out_dir/name) for key,name in CACHE_TABLES.items()}
    rc = json.loads((run_dir/'run_config.json').read_text())
    cfg, staging = rc['config_resolved'], Path(rc['input_dir'])
    channel = cfg['channels']['dapi'] - int(cfg['channels'].get('one_indexed', False))
    all_spots = pd.read_csv(run_dir/'spot_metrics.csv')
    tables = {key: [] for key in ['spots','objects','nuclei','sensitivity']}
    sources, fields = [], []
    for row in data['per_image'].itertuples():
        name = row.image
        source = source_acquisition(staging, row)
        stem = getattr(row, 'output_stem', None)
        if not stem or pd.isna(stem):
            from .micrograph_slides import _native_stem
            stem = _native_stem(run_dir/'publication_images', name, getattr(row, 'source_condition', row.condition))
        mask_path = run_dir/'masks'/(stem+'__nuclei_label_mask.tif')
        retained = tifffile.imread(mask_path)
        image = read_image(source)
        plane = extract_channel_at_z(image, channel, z_1indexed=int(row.z_plane))
        if not np.isclose(image.voxel_xy_nm, row.voxel_xy_nm, rtol=0, atol=1e-6):
            raise ValueError(f'calibration mismatch: {name}')
        roster = data['nuclei'].loc[data['nuclei'].image.eq(name)]
        ids = roster.nucleus_id.tolist()
        if set(np.unique(retained)) - {0} != set(ids):
            raise ValueError(f'retained label IDs differ from nucleus roster: {name}')
        sp = all_spots.loc[all_spots.image.eq(name) & all_spots.channel.eq('rna1')]
        r = classify_plane(plane, retained, sp, ids, row.voxel_xy_nm/1000.)
        label = data['labels'].loc[data['labels'].image.eq(name)].iloc[0]
        for key in tables:
            table = r[key].copy()
            for col in ['image','group','well_id','secondary_only']:
                table[col] = label[col]
            tables[key].append(table)
        objects = r['objects']
        dropped = objects.loc[objects.unretained.astype(bool)]
        fields.append(dict(image=name, group=label.group, well_id=label.well_id,
            secondary_only=label.secondary_only, dapi_objects=len(objects), retained_nuclei=len(ids),
            unretained_dapi_objects=len(dropped), below_min_area_px=int(dropped.below_min_area_px.sum()),
            unretained_area_median_px=dropped.area_px.median(), unretained_area_q25_px=dropped.area_px.quantile(.25),
            unretained_area_q75_px=dropped.area_px.quantile(.75),
            bin1_in_unretained_objects=int(dropped.bin1_spots.sum())))
        sources.append(dict(image=name, source=str(source), z_plane=int(row.z_plane), channel_index=channel,
            voxel_xy_um=row.voxel_xy_nm/1000., mask_path=str(mask_path), mask_sha256=sha256(mask_path),
            otsu_threshold=r['threshold']))
        print(f'DAPI {name}: {len(objects)} objects, {len(dropped)} unretained', flush=True)
    result = {key: pd.concat(value, ignore_index=True) for key, value in tables.items()}
    result['census'] = pd.DataFrame(fields)
    arm_rows = []
    for group, sub in result['census'].loc[~result['census'].secondary_only].groupby('group'):
        objects = result['objects'].loc[result['objects'].group.eq(group) & result['objects'].unretained.astype(bool)]
        arm_rows.append(dict(group=group, dapi_objects=int(sub.dapi_objects.sum()),
            retained_nuclei=int(sub.retained_nuclei.sum()), unretained_dapi_objects=len(objects),
            below_min_area_px=int(objects.below_min_area_px.sum()),
            unretained_area_median_px=objects.area_px.median(),
            unretained_area_q25_px=objects.area_px.quantile(.25), unretained_area_q75_px=objects.area_px.quantile(.75),
            unretained_area_median_um2=objects.area_um2.median(),
            unretained_area_q25_um2=objects.area_um2.quantile(.25), unretained_area_q75_um2=objects.area_um2.quantile(.75),
            bin1_in_unretained_objects=int(objects.bin1_spots.sum())))
    result['arms'] = pd.DataFrame(arm_rows)
    for key, filename in [('spots','spot_localization_dapi.csv'),('objects','dapi_objects.csv'),
                          ('census','dapi_object_census.csv'),('nuclei','dapi_per_nucleus.csv'),
                          ('sensitivity','dapi_sensitivity_per_object.csv'), ('arms','dapi_census_by_arm.csv')]:
        result[key].to_csv(out_dir/filename, index=False)
    (out_dir/'dapi_provenance.json').write_text(json.dumps(dict(sources=sources,
        method='Gaussian sigma 1 px; Otsu; fill holes; 4-connected objects >=200 px; report-time QC only',
        source_spots_sha256=sha256(run_dir/'spot_metrics.csv')), indent=2), encoding='utf-8')
    seal_dapi_cache(out_dir, run_dir)
    return result


def draw_qc(axes, data, ctx):
    """Pooled class fractions and object areas are descriptive QC only."""
    left, right = axes
    sp = data['spots'].loc[~data['spots'].secondary_only]
    palette = ['#595959', '#D67AE5', '#E69F00']
    bottom = np.zeros(len(ctx.group_order))
    names = ['retained mask', 'DAPI outside retained mask', 'extranuclear']
    for cls, color, label in zip(CLASSES, palette, names):
        values = np.array([sp.loc[sp.group.eq(g), 'class'].eq(cls).mean()*100 for g in ctx.group_order])
        left.bar(ctx.group_order, values, bottom=bottom, color=color, label=label)
        bottom += values
    left.set(ylabel='BIN1 spots (%)', ylim=(0, 100), title='Spot classification: descriptive, no test')
    left.legend(loc='upper left', bbox_to_anchor=(0, -.2), fontsize=7, frameon=False)
    objs = data['objects'].loc[~data['objects'].secondary_only & data['objects'].unretained.astype(bool)]
    maximum = max(20000., float(objs.area_px.max())) if len(objs) else 20000.
    bins = np.linspace(0, maximum*1.02, 25)
    for group in ctx.group_order:
        right.hist(objs.loc[objs.group.eq(group), 'area_px'], bins=bins, histtype='step',
                   linewidth=1.5, color=ctx.colors[group], label=group)
    right.axvline(float(data['census'].nucleus_min_area_px.iloc[0]), color='black', linestyle='--', linewidth=1)
    right.set(xlabel='Unretained DAPI object area (px)', ylabel='Objects',
              title='Area census: descriptive, no test')
    right.legend(loc='upper left', bbox_to_anchor=(0, -.2), fontsize=7, frameon=False)
    from .figures import no_box
    for ax in axes:
        no_box(ax.figure, ax)
        ax.tick_params(labelsize=9)
        ax.xaxis.label.set_size(9)
        ax.yaxis.label.set_size(9)
        ax.title.set_size(11)


def render_dapi(ctx, data, nuclei, well, field, contrasts, out_dir):
    """Independent localization plots and independent descriptive QC figures."""
    from . import figures as fig
    out_dir = guard_output(Path(out_dir))
    fig.set_style(); records=[]
    specs=[(FRACTION,'BIN1 puncta, % nuclear','BIN1 nuclear fraction, DAPI corrected (%)',100.,'fraction'),
           (COUNT,'Extranuclear BIN1 puncta','Extranuclear BIN1 spots / cell territory',1.,'extranuclear')]
    for name,title,ylabel,scale,suffix in specs:
        canvas,ax=fig.plt.subplots(figsize=(4.8,3.6))
        foot=fig.draw_replicate_simple(ax,ctx,name,well,field,nuclei,contrasts,ylabel,name,scale=scale)
        fig.no_box(canvas,ax)
        fig.layout_replicate_simple(canvas,ax,ctx,title,foot)
        rec=fig.save(canvas,out_dir,'FIG_LOCALIZATION_'+suffix,records,title,'DAPI tables / Per well / Contrasts')
        rec.update(endpoint=name,caption=title,is_composite=False)
    for index,suffix,title in [(0,'spot_classes','BIN1 spot classification'),(1,'object_area','Unretained DAPI object area')]:
        canvas,axes=fig.plt.subplots(1,2,figsize=(4.8,3.6))
        draw_qc(axes,data,ctx)
        canvas.delaxes(axes[1-index]); ax=axes[index]
        ax.set_position([.19,.34,.77,.50]); ax.set_title(title,fontsize=11)
        ax._replicate_simple_axis=lambda variant: None
        canvas.text(.02,.02,'Descriptive QC, no test; run '+ctx.run_name,fontsize=6)
        rec=fig.save(canvas,out_dir,'FIG_LOCALIZATION_'+suffix,records,title,'DAPI tables')
        rec.update(caption=title,is_composite=False)
    return dict(crops=[],figures=records)


def write_documents(out_dir, data, contrasts):
    """Write the correction's definitions and measured results beside the report."""
    out_dir = guard_output(Path(out_dir))
    result = contrasts.loc[contrasts.endpoint.eq(FRACTION)].iloc[0]
    sp = data['spots'].loc[~data['spots'].secondary_only]
    summary = sp.groupby(['group','class']).size().unstack(fill_value=0).reindex(columns=CLASSES, fill_value=0)
    summary.to_csv(out_dir/'dapi'/'spot_classes_by_arm.csv')
    measured = (f"DAPI-corrected well-mean nuclear fraction: WT {100*result.mean_ref:.6g}%; "
        f"QKI-KO {100*result.mean_test:.6g}%; raw Welch p={result.p_welch:.9g}; "
        f"Holm p={result.p_welch_holm_within_family:.9g}; Hedges g={result.hedges_g:.6g}.\n")
    methods = '''DAPI correction is a report-time classification, not a segmentation of record.
The original DAPI channel at each recorded z_plane is read using fishsuite.core.io.read_image and
extract_channel_at_z. No detection, retained-nucleus segmentation or null analysis is rerun.
Gaussian sigma 1 px precedes Otsu thresholding, hole filling and removal of connected objects
smaller than 200 px. Components use four-connectivity. Areas use the recorded XY calibration.
Retained label overlap is recorded in pixels for every DAPI object. An unretained object has
zero overlap with any retained label. Touching nuclei can merge into one DAPI object; these are
objects, not a proven census of dropped nuclei or recovered pre-filter candidates.

Spot centroids use nearest-pixel membership (half-pixel rounded upward). Outside all DAPI-positive
area means extranuclear, including retained-mask boundary pixels. Within DAPI-positive area,
retained-mask membership means in_retained_nucleus; otherwise in_unretained_dapi_object.
The second class denotes unretained DAPI-positive area and can include
extensions of components that overlap a retained mask. dapi_object_has_retained_overlap explicitly
identifies this boundary case; the object census counts only zero-overlap components as unretained.
Distance is unsigned Euclidean distance to the filled DAPI mask's 0.5 contour in micrometres.
When no DAPI object is present, edge distance and corrected retained-cell endpoints are NA.

For each retained nucleus, numerator is the number of BIN1 spots in its retained label AND DAPI-positive area.
Extranuclear count includes only extranuclear spots assigned to its original nucleus_id/territory.
nuclear_spot_fraction_dapi = numerator / (numerator + assigned extranuclear count).
DAPI-positive spots outside retained masks and unassigned extranuclear spots do not enter this
retained-cell denominator. Zero-total fractions are NA; valid count zeros are retained.
Nucleus values are averaged within FOV, then FOV means equally within each well; two-sided Welch
tests use three WT and three KO wells. Hedges g is KO minus WT, and the existing report MDE
implementation uses 80% power and family-adjusted alpha. MDE is not an exclusion bound.

SENSITIVITY ONLY: every DAPI component is a nucleus, including zero-retained-overlap components.
Each outside-DAPI spot is assigned to the nearest DAPI component's positive pixel, unlimited radius
and deterministic distance-transform tie handling. This Voronoi-like territory is not the retained
cell territory and has no cell-boundary evidence. Object values then FOV then well means are used.
Sensitivity tests have their own two-endpoint Holm family and are never headline inference.

RNASEH2B nuclear mean: subtract the matching-arm arithmetic mean of all secondary-control FOV
means from each primary nucleus, without clipping; average nucleus -> FOV -> well. Four WT and
three KO secondary fields are included. No DAPI division. The supplied correction CSV is the
source; control-baseline uncertainty is not propagated. Exposure uniformity is reported by the
acquirer in the A11 dispatch, not established from metadata. Staining batch remains a caveat.
Absolute IF remains descriptive; N:C is shown beside corrected nuclear mean and raw total is
secondary. The earlier acquisition note predates the acquirer's exposure clarification.
Species: human / Homo sapiens. No genome reference file is used by this image-report workflow.
'''
    amendments = '''A11 replaces mask-only localization headline inference with DAPI correction.
cyto_spot_count / rna1_cyto_spots_per_nucleus and nuclear_spot_fraction /
rna1_nuclear_spot_fraction remain present and are labelled legacy_mask_only, descriptive and
excluded from multiplicity families. Other source columns and nonlocalization raw tests are preserved.
Detection family loses the legacy cytoplasmic-count test. Localization loses the legacy
nuclear-fraction test and gains extranuclear_spot_count_per_cell_territory and
nuclear_spot_fraction_dapi. Holm and family-alpha MDE are recomputed within amended families.
All-DAPI-object sensitivity has a separate two-endpoint localization family. Partner inference
and the existing exploratory-versus-descriptive pairing policy are unchanged.
Actual membership and finite-test counts are in REPORT.xlsx / Multiplicity plan and contrasts.csv.
'''
    for name, text in [('METHODS.md', methods), ('FAMILY_AMENDMENTS.md', amendments),
        ('CHANGES_v4_to_v4b.md', 'Added DAPI spot reclassification, zero-overlap object census, corrected localization and separate all-object sensitivity.\nRebuilt localization and selection slides; matched-secondary-corrected IF beside N:C; Sam question titles, excess-over-shuffle headlines and bracket/descriptive annotations retained.\n'+measured),
        ('README.md', 'Start with REPORT.xlsx, Sam_RNASEH2B_BIN1.pptx and READOUT.md.\nThe dapi/ folder contains spot identities, object overlaps/areas, census, sensitivity tables and plane provenance.\nThe localization/ folder contains corrected full/focus figures and DAPI_QC.\nThis v4b is built in the authorized E: fallback because F: is outside the writable sandbox.\n'+measured)]:
        (out_dir/name).write_text(text, encoding='utf-8')
    readout = out_dir/'READOUT.md'
    readout.write_text(measured+'\n'+methods+'\n'+readout.read_text(encoding='utf-8'), encoding='utf-8')
