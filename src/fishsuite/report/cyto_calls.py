"""QC of persisted cytoplasmic RNA1 calls; never opens acquisition files."""
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from skimage.measure import find_contours
from skimage.segmentation import find_boundaries
import tifffile
from .provenance import guard_output, sha256
from .figures import merge_png_for
from .aggregate import ReportInputError


def mask_edge_distance(mask, x, y, voxel):
    """Distance to the assigned label's 0.5 contour, in calibrated micrometres."""
    curves = find_contours(np.pad(mask.astype(float), 1), .5)
    best = np.inf
    point = np.array([y + 1, x + 1])
    for curve in curves:
        a, b = curve[:-1], curve[1:]
        v = b - a
        t = np.clip(np.sum((point-a)*v, axis=1) / np.sum(v*v, axis=1), 0, 1)
        best = min(best, float(np.linalg.norm(point - (a + t[:, None]*v), axis=1).min()))
    return best * voxel


def build_cyto_calls(run_dir, out_dir, nuclei):
    run_dir, out_dir = Path(run_dir), guard_output(Path(out_dir))
    spots = pd.read_csv(run_dir/'spot_metrics.csv')
    calls = spots.loc[spots.channel.eq('rna1') & spots.in_cytoplasm.eq(1) & spots.nucleus_id.gt(0)].copy()
    keys = ['image', 'channel', 'spot_id']
    if calls.duplicated(keys).any():
        raise ReportInputError('duplicate cytoplasmic spot identities; QC requires unique persisted calls')
    roster = nuclei[['image', 'nucleus_id', 'group', 'voxel_xy_um']].drop_duplicates()
    calls = calls.merge(roster, on=['image', 'nucleus_id'], how='left', validate='many_to_one')
    if calls.group.isna().any():
        raise ReportInputError('cytoplasmic call lacks group/calibration roster entry')
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles, records, sources = {}, [], []
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 12)
    for field, rows in calls.groupby('image', sort=True):
        hit = merge_png_for(run_dir/'publication_images', field)
        if hit is None:
            # Only persisted, full-frame QC overlays can be used as fallback.
            core = Path(field).stem
            matches = [p for p in (run_dir/'qc_overlays').glob('*__qc_dapi_rna1_rna2_nuclei_spots.png')
                       if core.endswith(p.name.split('__')[1])]
            hit = (str(matches[0]), '__'.join(matches[0].name.split('__')[:2])+'__') if len(matches)==1 else None
        mask_path = run_dir/'masks'/(hit[1]+'nuclei_label_mask.tif') if hit else None
        pixels = labels = None
        status = 'missing overlay' if hit is None else 'missing mask' if not mask_path.is_file() else 'available'
        if status == 'available':
            pixels = Image.open(hit[0]).convert('RGB')
            labels = tifffile.imread(mask_path)
            if labels.shape != (pixels.height, pixels.width):
                status = 'image/mask geometry mismatch'
            sources.append(dict(image=field, image_path=hit[0], image_sha256=sha256(Path(hit[0])),
                                mask_path=str(mask_path), mask_sha256=sha256(mask_path)))
        for row in rows.itertuples():
            record = dict(image=field, condition=row.condition, group=row.group,
                          nucleus_id=int(row.nucleus_id), spot_id=int(row.spot_id),
                          x_px=float(row.x_px), y_px=float(row.y_px), voxel_xy_um=float(row.voxel_xy_um),
                          distance_to_mask_edge_um=np.nan, crop_um=6., status=status,
                          source_image=hit[0] if hit else '', mask_path=str(mask_path) if mask_path else '')
            tile = Image.new('RGB', (300, 380), 'white')
            draw = ImageDraw.Draw(tile)
            if status == 'available':
                mask = labels == row.nucleus_id
                if not mask.any() or not np.isfinite(row.voxel_xy_um) or row.voxel_xy_um <= 0:
                    record['status'] = 'missing label or calibration'
                else:
                    d = mask_edge_distance(mask, row.x_px, row.y_px, row.voxel_xy_um)
                    record['distance_to_mask_edge_um'] = d
                    side = int(round(6. / row.voxel_xy_um))
                    x0 = min(max(int(round(row.x_px)) - side//2, 0), max(pixels.width-side, 0))
                    y0 = min(max(int(round(row.y_px)) - side//2, 0), max(pixels.height-side, 0))
                    record.update(x0=x0, y0=y0, crop_px=side, actual_crop_um=side*row.voxel_xy_um)
                    crop = np.array(pixels.crop((x0,y0,x0+side,y0+side)))
                    edge = find_boundaries(labels[y0:y0+side,x0:x0+side], mode='inner')
                    crop[edge] = [0,255,255]
                    tile.paste(Image.fromarray(crop).resize((300,300), Image.Resampling.NEAREST), (0,0))
                    draw = ImageDraw.Draw(tile)
                    x, y = (row.x_px-x0)*300/side, (row.y_px-y0)*300/side
                    draw.ellipse((x-7,y-7,x+7,y+7), outline='white', width=2)
            label = field.replace('.vsi','')
            label_lines, line = [], ''
            for char in label:
                if draw.textlength(line+char, font=font) > 294:
                    label_lines.append(line)
                    line = ''
                line += char
            label_lines.append(line)
            for i, line in enumerate(label_lines):
                draw.text((3,301+14*i), line, font=font, fill='black')
            d = record['distance_to_mask_edge_um']
            draw.text((3,358), f'nucleus {row.nucleus_id} / spot {row.spot_id} / '+(f'{d:.3f} µm' if np.isfinite(d) else record['status']), font=font, fill='black')
            tiles.setdefault(str(row.group), []).append(tile)
            records.append(record)
    table = pd.DataFrame(records, columns=None if records else ['group','distance_to_mask_edge_um','status'])
    table.to_csv(guard_output(out_dir/'cytoplasmic_calls.csv'), index=False)
    summary = []
    for group, group_tiles in tiles.items():
        # Pagination retains every call without producing an unopenably tall raster.
        for page, start in enumerate(range(0, len(group_tiles), 48), 1):
            subset = group_tiles[start:start+48]
            sheet = Image.new('RGB', (6*300, math.ceil(len(subset)/6)*380+40), 'white')
            ImageDraw.Draw(sheet).text((10,10), f'{group}: cytoplasmic BIN1 calls / 6 µm crops / cyan nuclear outlines / white marked call', font=font, fill='black')
            for i, tile in enumerate(subset):
                sheet.paste(tile, ((i%6)*300, 40+(i//6)*380))
            safe = ''.join(c if c.isalnum() or c in '-_' else '_' for c in group)
            sheet.save(guard_output(out_dir/f'{safe}_sheet_{page:03d}.png'))
        sub = table.loc[table.group.eq(group)]
        summary.append(dict(group=group, calls=len(sub), median_distance_um=sub.distance_to_mask_edge_um.median(),
                            missing=int(sub.status.ne('available').sum())))
    pd.DataFrame(summary).to_csv(guard_output(out_dir/'summary.csv'), index=False)
    table.loc[table.status.ne('available')].to_csv(guard_output(out_dir/'missing.csv'), index=False)
    (out_dir/'provenance.json').write_text(json.dumps(dict(source_run=str(run_dir),
        spot_metrics_sha256=sha256(run_dir/'spot_metrics.csv'), sources=sources,
        distance_definition='Euclidean distance from spot centroid to assigned nuclear label 0.5 contour; linear contour segments; single plane',
        crop_definition='6 µm requested; rounded to nearest pixel; shifted inward at field boundary; source images only'), indent=2), encoding='utf-8')
    return table
