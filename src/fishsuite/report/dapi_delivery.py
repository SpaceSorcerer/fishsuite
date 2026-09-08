"""Validate and package a completed DAPI-corrected report without reading pixels."""
from pathlib import Path
import argparse
import io
import json
import math
import shutil
import numpy as np
import pandas as pd
from .provenance import guard_output, sha256
from .dapi_mask import COUNT, FRACTION, CLASSES


def validate_report(out):
    from scipy.stats import ttest_ind
    from .stats import holm
    out = Path(out)
    sp = pd.read_csv(out/'dapi/spot_localization_dapi.csv')
    n = pd.read_csv(out/'dapi/dapi_per_nucleus.csv')
    obj = pd.read_csv(out/'dapi/dapi_objects.csv')
    census = pd.read_csv(out/'dapi/dapi_object_census.csv')
    well = pd.read_csv(out/'per_well.csv')
    contrasts = pd.read_csv(out/'contrasts.csv')
    assert not sp.duplicated(['image','spot_id']).any()
    assert sp['class'].isin(CLASSES).all()
    assert not n.duplicated(['image','nucleus_id']).any()
    for field, rows in n.groupby('image'):
        spots = sp.loc[sp.image.eq(field)]
        for r in rows.itertuples():
            nuclear = int((spots.retained_mask_id_at_spot.eq(r.nucleus_id) & spots['class'].eq('in_retained_nucleus')).sum())
            extra = int((spots.nucleus_id.eq(r.nucleus_id) & spots['class'].eq('extranuclear')).sum())
            assert r.nuclear_spot_count_dapi == nuclear
            if census.loc[census.image.eq(field),'dapi_objects'].item():
                assert getattr(r,COUNT) == extra
                expected = nuclear/(nuclear+extra) if nuclear+extra else np.nan
                assert np.isclose(getattr(r,FRACTION), expected, equal_nan=True)
        objects = obj.loc[obj.image.eq(field)]
        assert len(objects) == census.loc[census.image.eq(field),'dapi_objects'].item()
        assert int(objects.unretained.sum()) == census.loc[census.image.eq(field),'unretained_dapi_objects'].item()
    bio = n.loc[~n.secondary_only]
    for endpoint in [COUNT,FRACTION]:
        expected = bio.groupby(['group','well_id','image'])[endpoint].mean().groupby(['group','well_id']).mean()
        actual = well.loc[well.endpoint.eq(endpoint)].set_index(['group','well_id']).well_mean_of_field_values
        np.testing.assert_allclose(actual.sort_index(), expected.sort_index(), equal_nan=True, atol=1e-12)
        c = contrasts.loc[contrasts.endpoint.eq(endpoint)].iloc[0]
        a = actual.loc[c.test_group].to_numpy()
        b = actual.loc[c.reference_group].to_numpy()
        assert len(a) == len(b) == 3
        np.testing.assert_allclose(c.p_welch, ttest_ind(a,b,equal_var=False).pvalue, atol=1e-12)
    members = contrasts.loc[contrasts.family.eq('localization') & contrasts.in_holm_family]
    np.testing.assert_allclose(members.p_welch_holm_within_family, holm(members.p_welch.to_list()), equal_nan=True)
    from .stats import _power_two_sided
    for c in contrasts.loc[contrasts.endpoint.isin([COUNT,FRACTION])].itertuples():
        correction = 1-3/(4*(c.n_wells_test+c.n_wells_reference)-9)
        power = _power_two_sided(c.mde_hedges_g_at_family_alpha/correction,
                                 .05/c.holm_family_size,c.n_wells_test,c.n_wells_reference)
        assert np.isclose(power,.8,atol=1e-7), (c.endpoint,power)
    for name in ['rna1_nuclear_spot_fraction','rna1_cyto_spots_per_nucleus']:
        c = contrasts.loc[contrasts.endpoint.eq(name)].iloc[0]
        assert not c.in_holm_family and c.excluded_from_holm == 'legacy_mask_only'
    old_cyto = sp.loc[~sp.secondary_only & sp.in_cytoplasm.eq(1) & sp.nucleus_id.gt(0)]
    old_cyto.groupby(['group','class','dapi_object_has_retained_overlap']).size().rename('spots').reset_index().to_csv(
        out/'dapi/legacy_cyto_reclassification_by_arm.csv',index=False)
    print('VALIDATED: unique spot identities; class accounting; retained-cell counts/fractions; object census; nucleus -> FOV -> well hierarchy; Welch 3 vs 3; localization Holm; legacy exclusions.')


def preview_deck(out, filename='Sam_RNASEH2B_BIN1.pptx'):
    """Approximate Arial previews; never claim these are native Office renders."""
    from pptx import Presentation
    from PIL import Image, ImageDraw, ImageFont
    out = Path(out)
    target = guard_output(out/'validation')
    target.mkdir(parents=True,exist_ok=True)
    ppt = Presentation(out/filename)
    contact = Image.new('RGB',(1600,math.ceil(len(ppt.slides)/2)*470),'#dddddd')
    for i,slide in enumerate(ppt.slides):
        im = Image.new('RGB',(1280,720),'white')
        draw = ImageDraw.Draw(im)
        for shape in slide.shapes:
            x,y,w,h = [round(v/914400*96) for v in [shape.left,shape.top,shape.width,shape.height]]
            if shape.shape_type == 13:
                pic = Image.open(io.BytesIO(shape.image.blob)).convert('RGB').resize((w,h))
                im.paste(pic,(x,y))
            elif shape.has_text_frame:
                frame = shape.text_frame
                tx,ty = x+round(frame.margin_left/914400*96), y+round(frame.margin_top/914400*96)
                width = w-round((frame.margin_left+frame.margin_right)/914400*96)
                for para in frame.paragraphs:
                    fs = round((para.font.size.pt if para.font.size else 18)*96/72)
                    font = ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf' if para.font.bold else 'C:/Windows/Fonts/arial.ttf',fs)
                    lines, line = [], ''
                    for word in para.text.split():
                        candidate = (line+' '+word).strip()
                        if draw.textlength(candidate,font=font) > width and line:
                            lines.append(line)
                            line = word
                        else:
                            line = candidate
                    lines.append(line)
                    for line in lines:
                        draw.text((tx,ty),line,font=font,fill='#111111')
                        ty += round(fs*1.2)
                    assert ty <= y+h+2, (i+1,para.text,ty,y+h)
        im.save(target/f'slide_preview_{i+1:02d}.png')
        im.thumbnail((790,445))
        contact.paste(im,((i%2)*800,(i//2)*470+22))
    contact.save(target/'slide_contact_sheet.jpg')
    print(f'PREVIEW: {len(ppt.slides)} slides; Arial text fits. Approximate previews, not native PowerPoint renders.')


def finalize(out, run, dispatch):
    out, run, dispatch = guard_output(Path(out)), Path(run), Path(dispatch)
    validate_report(out)
    qc = out/'qc_cyto_calls'
    if not (qc/'summary.csv').is_file():
        from .cyto_calls import build_cyto_calls
        nuclei = pd.read_csv(run/'nuclei_metrics.csv')
        labels = pd.read_csv(out/'dapi/dapi_per_nucleus.csv')[['image','group']].drop_duplicates()
        nuclei = nuclei.drop(columns=['group'],errors='ignore').merge(labels,on='image',validate='many_to_one')
        build_cyto_calls(run,qc,nuclei)
    (qc/'README.md').write_text('legacy_mask_only contact sheets from A9: these retain original cytoplasmic calls for audit. Corrected classification and endpoints are in ../dapi and ../localization.\n',encoding='utf-8')
    # Amend generated metadata in the report module and regenerate the deck's
    # workbook attestation, preserving all measurement/statistical cells.
    import openpyxl
    import yaml
    book = openpyxl.load_workbook(out/'REPORT.xlsx')
    slide_values = book['Slide values']
    columns = {cell.value:cell.column for cell in slide_values[2]}
    for i in range(3,slide_values.max_row+1):
        if (slide_values.cell(i,columns['endpoint']).value == FRACTION and
                slide_values.cell(i,columns['label']).value == 'display multiplier'):
            slide_values.cell(i,columns['value'],100.)
    objects = pd.read_csv(out/'dapi/dapi_objects.csv')
    census_path = out/'dapi/dapi_object_census.csv'
    census = pd.read_csv(census_path)
    dropped = objects.loc[objects.unretained].groupby('image').area_um2
    census['unretained_area_iqr_px'] = census.unretained_area_q75_px-census.unretained_area_q25_px
    census['unretained_area_median_um2'] = census.image.map(dropped.median())
    census['unretained_area_q25_um2'] = census.image.map(dropped.quantile(.25))
    census['unretained_area_q75_um2'] = census.image.map(dropped.quantile(.75))
    census['unretained_area_iqr_um2'] = census.unretained_area_q75_um2-census.unretained_area_q25_um2
    census.to_csv(census_path,index=False)
    from .dapi_mask import seal_dapi_cache
    seal_dapi_cache(out/'dapi', run)
    sheet = book['DAPI census']
    from copy import copy
    old_last = sheet.max_column
    for j,col in enumerate(census.columns,1):
        sheet.cell(2,j,col)
        for i,value in enumerate(census[col],3):
            sheet.cell(i,j,None if pd.isna(value) else value)
        if j > old_last:
            from openpyxl.utils import get_column_letter
            sheet.column_dimensions[get_column_letter(j)].width = 22
            for i in range(2,len(census)+3):
                sheet.cell(i,j)._style = copy(sheet.cell(i,old_last)._style)
    sheet['A1'] = 'Report-time DAPI objects, retained nuclei and zero-retained-overlap objects by FOV. Area median, quartiles and IQR width are reported in px and square micrometres. These are not reconstructed segmentation exclusions.'
    for sheet in book:
        for row in sheet:
            for cell in row:
                if isinstance(cell.value,str) and '\u00c3\u00b7' in cell.value:
                    cell.value = cell.value.replace('\u00c3\u00b7','divided by')
                if isinstance(cell.value,str) and 'because laser power is retuned per section' in cell.value:
                    cell.value = cell.value.replace('because laser power is retuned per section',
                        'exposure is uniform per acquirer but staining batch remains a caveat')
    book.save(out/'REPORT.xlsx')
    spec_path = out/'deck_spec.resolved.yaml'
    resolved = yaml.safe_load(spec_path.read_text())
    resolved['workbook_sha256'] = sha256(out/'REPORT.xlsx')
    spec_path.write_text(yaml.safe_dump(resolved,sort_keys=False),encoding='utf-8')
    from .slides import build_deck
    build_deck(out/'REPORT.xlsx',resolved,out/'Sam_RNASEH2B_BIN1.pptx')
    # Copy only explicitly named source tables; no discovery below the run/raw tree.
    dest = out/'data'
    dest.mkdir(exist_ok=True)
    for name in ['nuclei_metrics.csv','spot_metrics.csv','per_image_summary.csv','run_config.json','versions.txt']:
        shutil.copyfile(run/name, guard_output(dest/name))
        assert sha256(run/name) == sha256(dest/name)
    # Persist secondary correction as provenance, including its original caveat labels.
    template = yaml.safe_load((dispatch/'deck_spec_A11.yaml').read_text())
    correction = Path(template['secondary_corrected_csv'])
    shutil.copyfile(correction, dest/correction.name)
    for name in ['BUILD.ps1','FINALIZE.ps1','TEST.ps1','deck_spec_A11.yaml','build.log','tests_final.log']:
        if (dispatch/name).is_file():
            shutil.copyfile(dispatch/name, out/name)
    (out/'BUILD_COMMANDS.md').write_text(
        'Execute BUILD.ps1, then FINALIZE.ps1 with PowerShell. BUILD.ps1 bootstraps the existing Java runtime and cached Bio-Formats jars, then invokes fishsuite.cli report. FINALIZE.ps1 verifies accounting, completes census units and acquisition notes, copies named sources and regenerates the deck attestation.\n'
        'No installations or detection/segmentation/null reruns. Exact build output is build.log.\n'
        'The secondary correction source retains its earlier acquisition caveat labels; the A11 dispatch supplies the later acquirer-reported exposure clarification.\n', encoding='utf-8')
    preview_deck(out)
    sp = pd.read_csv(out/'dapi/spot_localization_dapi.csv')
    legacy = sp.loc[~sp.secondary_only & sp.in_cytoplasm.eq(1) & sp.nucleus_id.gt(0)]
    lines = ['Measured legacy-call QC (DAPI-positive area is not an individually segmented nucleus):']
    for group, rows in legacy.groupby('group'):
        inside = rows['class'].eq('in_unretained_dapi_object')
        zero_overlap = inside & ~rows.dapi_object_has_retained_overlap
        lines.append(f'{group}: {int(inside.sum())}/{len(rows)} legacy cytoplasmic calls lie in DAPI-positive area; '
                     f'{int(rows["class"].eq("extranuclear").sum())} remain extranuclear. '
                     f'{int(zero_overlap.sum())} lie in zero-retained-overlap objects and '
                     f'{int((inside & rows.dapi_object_has_retained_overlap).sum())} in DAPI components with retained overlap.')
    note = '\n'.join(lines)+'\n'
    (out/'dapi/LEGACY_CALL_QC.md').write_text(note,encoding='utf-8')
    readout = out/'READOUT.md'
    text = readout.read_text(encoding='utf-8')
    if lines[0] not in text:
        readout.write_text(note+'\n'+text,encoding='utf-8')
    figures = sorted((out/'figures').glob('*.png'))
    (out/'FIGURE_INDEX.md').write_text('\n'.join(f'- figures/{p.name}' for p in figures)+'\n',encoding='utf-8')
    source = json.loads((out/'dapi/dapi_provenance.json').read_text())
    for row in source['sources']:
        assert sha256(Path(row['mask_path'])) == row['mask_sha256']
    assert sha256(run/'spot_metrics.csv') == source['source_spots_sha256']
    print('SOURCE INTEGRITY: persisted mask hashes and spot-table hash unchanged; copied source tables match byte-for-byte.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--dispatch', type=Path, required=True)
    args = parser.parse_args()
    finalize(args.out,args.run,args.dispatch)
