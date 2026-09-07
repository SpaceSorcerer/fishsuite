from pathlib import Path
import importlib.util
import pytest
import pandas as pd


def test_recorded_deck_semantic_assets_and_localization(tmp_path):
    import yaml
    from fishsuite.report.build import build_report
    from pptx import Presentation
    root = Path(__file__).resolve().parents[1]
    spec = root/'_closeout_evidence/A/deck_spec.yaml'
    run = Path('F:/Image Analysis Work/RNASEH2B_BIN1introns_2026_08_25/13b_FULL_HARMONIZED_T36_FIXEDNUCLEAR_2026-09-05/RUN_T36_fixed_2026-09-05_0915')
    result = build_report(run, tmp_path/'report', groups=['WT=WT_1,WT_2,WT_3', 'QKI-KO=KO_1,KO_2,KO_3'],
                          reference='WT', make_figures=False, deck_spec=spec, deck=True)
    resolved = yaml.safe_load((result['out_dir']/'deck_spec.resolved.yaml').read_text())
    expected = {'overview':'A01_overview', 'q1_count_size':'A04_question1',
                'q1_localization':'A05_BIN1_localization', 'q1_total_if':'A04_RNASEH2B_total_IF',
                'q2':'A06_question2', 'q3':'A07_question3', 'q4_reverse_anchor':'A08_question4',
                'nucleus_selection':'A03_nucleus_selection', 'methods_provenance':'A02_methods',
                'standard_coloc':'A09_standard_coloc'}
    assert [s['identity'] for s in resolved['slides']] == list(expected)
    for slide in resolved['slides']:
        assert Path(slide['figures'][0]['path']).name == expected[slide['identity']]+'.png'
        assert Path(slide['figures'][0]['path']).with_suffix('.svg').is_file()
        if slide['identity'] in {'q1_count_size','q1_total_if','q2','q3','q4_reverse_anchor'}:
            assert len(slide['figures']) == 3
    crops = pd.read_excel(result['xlsx'], sheet_name='Localization crops', header=1)
    assert crops.crop_status.eq('available').all()
    assert crops.territory_boundary.eq('missing').all()
    svg = (result['out_dir']/'figures/A05_BIN1_localization.svg').read_text()
    assert 'per nucleus and assigned cell territory' in svg.lower()
    assert 'Assigned-cytoplasmic puncta count' in svg
    ppt = Presentation(result['out_dir']/'Sam_RNASEH2B_BIN1.pptx')
    assert [s.shapes[0].text for s in ppt.slides] == [s['title'] for s in resolved['slides']]
    for slide, definition in zip(ppt.slides, resolved['slides']):
        if len(definition['figures']) == 3:
            text = [s.text for s in slide.shapes if s.has_text_frame]
            assert 'WT' in text and 'QKI-KO' in text


@pytest.mark.parametrize('claim', ['999 nuclei', '1e9 nuclei', '25um scale bar'])
@pytest.mark.parametrize('field', ['title', 'body', 'label'])
def test_export_rejects_literal_numeric_text(tmp_path, field, claim):
    from fishsuite.report.slides import build_deck
    from fishsuite.report.workbook import write
    from fishsuite.report.provenance import sha256
    book = write(tmp_path/'REPORT.xlsx', {'Values': pd.DataFrame({'value': [1]})}, order=['Values'])
    slide = dict(title='Measured value', values=[dict(sheet='Values', cell='A3', label='nuclei')])
    (slide['values'][0] if field == 'label' else slide)[field] = claim
    with pytest.raises(RuntimeError, match='literal numeric'):
        build_deck(book, dict(workbook_sha256=sha256(book), slides=[slide]), tmp_path/'bad.pptx')
    assert not (tmp_path/'bad.pptx').exists()


@pytest.mark.parametrize('via_junction', [False, True])
@pytest.mark.parametrize('writer', ['save', 'localization', 'workbook', 'deck'])
def test_frozen_writers_before_side_effect(tmp_path, monkeypatch, writer, via_junction):
    from unittest.mock import Mock
    from fishsuite.report import figures, workbook, slides
    target = tmp_path/'DELIVERY_frozen'/'child'
    if via_junction:
        import subprocess
        frozen = Path('F:/Image Analysis Work/RNASEH2B_BIN1introns_2026_08_25/DELIVERY_RNASEH2B_BIN1intron_2026-09-05_v3')
        link = tmp_path/'alias'
        subprocess.run(['powershell', '-NoProfile', '-Command',
                        f"New-Item -ItemType Junction -Path '{link}' -Target '{frozen}' | Out-Null"], check=True)
        assert link.resolve() == frozen.resolve()
        target = link/'child'
    mkdir = Mock(side_effect=AssertionError('mkdir side effect'))
    opened = Mock(side_effect=AssertionError('open side effect'))
    monkeypatch.setattr(Path, 'mkdir', mkdir)
    monkeypatch.setattr('builtins.open', opened)
    with pytest.raises(RuntimeError, match='frozen output'):
        if writer == 'save':
            figures.save(Mock(), target, 'test', [], '', '')
        elif writer == 'localization':
            figures.render_localization(None, None, None, None, None, None, target)
        elif writer == 'workbook':
            workbook.write(target/'test.xlsx', {})
        else:
            slides.build_deck(tmp_path/'absent.xlsx', {}, target/'test.pptx')
    mkdir.assert_not_called()
    opened.assert_not_called()


def test_save_nonfrozen_and_descendant_guard(tmp_path, monkeypatch):
    from unittest.mock import Mock
    from fishsuite.report import figures
    figure, ax = figures.plt.subplots()
    ax.plot([0, 1])
    manifest = []
    figures.save(figure, tmp_path, 'allowed', manifest, '', '')
    assert (tmp_path/'allowed.png').is_file() and (tmp_path/'allowed.svg').is_file()
    mkdir = Mock(side_effect=AssertionError('mkdir side effect'))
    monkeypatch.setattr(Path, 'mkdir', mkdir)
    with pytest.raises(RuntimeError, match='frozen output'):
        figures.save(Mock(), tmp_path, 'DELIVERY_frozen/nested', [], '', '')
    mkdir.assert_not_called()


def test_slide_validation_missing_and_stale(tmp_path):
    from fishsuite.report.slides import validate_deck
    from fishsuite.report.provenance import sha256
    from fishsuite.report.workbook import write
    book = write(tmp_path/'REPORT.xlsx', {'Slide values':pd.DataFrame({'value':[1]})}, order=['Slide values'])
    spec = {'workbook_sha256':sha256(book), 'slides':[{'title':'Measured value', 'values':[{'sheet':'Slide values','cell':'A3'}], 'figures':[]}]}
    validate_deck(book, spec)
    spec['slides'][0]['values'][0]['cell'] = 'A300'
    with pytest.raises(RuntimeError, match='missing'):
        validate_deck(book, spec)
    spec['slides'][0]['values'][0]['allow_missing'] = True
    with pytest.raises(RuntimeError, match='missing workbook cell'):
        validate_deck(book, spec)
    spec['workbook_sha256'] = '0'*64
    with pytest.raises(RuntimeError, match='stale'):
        validate_deck(book, spec)


def test_slide_validation_rejects_untraced_numbers(tmp_path):
    from fishsuite.report.slides import validate_deck
    from fishsuite.report.workbook import write
    from fishsuite.report.provenance import sha256
    book = write(tmp_path/'REPORT.xlsx', {'Slide values':pd.DataFrame({'value':[1]})}, order=['Slide values'])
    with pytest.raises(RuntimeError, match='literal numeric'):
        validate_deck(book, {'workbook_sha256':sha256(book), 'slides':[{'title':'393 nuclei', 'values':[], 'figures':[]}]})


def test_missing_stale_figure_and_cohort(tmp_path):
    from fishsuite.report.slides import validate_deck
    from fishsuite.report.workbook import write
    from fishsuite.report.provenance import sha256
    book=write(tmp_path/'REPORT.xlsx',{'Slide values':pd.DataFrame({'value':[1]})},order=['Slide values'])
    asset=tmp_path/'figure.png'
    spec={'workbook_sha256':sha256(book),'cohort':'retained',
          'slides':[{'title':'Measured endpoint','values':[],
                     'figures':[{'path':str(asset),'sha256':'0'*64,'cohort':'retained'}]}]}
    with pytest.raises(RuntimeError,match='missing figure'):
        validate_deck(book,spec)
    asset.write_bytes(b'persisted test asset')
    with pytest.raises(RuntimeError,match='stale figure'):
        validate_deck(book,spec)
    spec['slides'][0]['figures'][0]['sha256']=sha256(asset)
    spec['slides'][0]['figures'][0]['cohort']='filtered'
    with pytest.raises(RuntimeError,match='cohort mismatch'):
        validate_deck(book,spec)


def test_stale_slide_value_lineage_rejected(tmp_path):
    from fishsuite.report.slides import validate_deck
    from fishsuite.report.workbook import write
    from fishsuite.report.provenance import sha256
    sheets={'Contrasts':pd.DataFrame({'diff':[-2.]}),
            'Slide values':pd.DataFrame([dict(label='MEASURED direction',value='higher',source_sheet='Contrasts',source_cell='A3')])}
    book=write(tmp_path/'REPORT.xlsx',sheets,order=list(sheets))
    with pytest.raises(RuntimeError,match='stale Slide values lineage'):
        validate_deck(book,{'schema':'resolved-deck-1','workbook_sha256':sha256(book),
                            'slides':[{'title':'Measured direction','values':[],'figures':[]}]})


@pytest.mark.skipif(importlib.util.find_spec('pptx') is None,
                    reason='python-pptx missing from fishproc_dml; dispatch prohibits installation')
def test_pptx_notes_media_and_count(tmp_path):
    from fishsuite.report.slides import build_deck
    from fishsuite.report.workbook import write
    from fishsuite.report.provenance import sha256
    from pptx import Presentation
    book = write(tmp_path/'REPORT.xlsx', {'Slide values':pd.DataFrame({'value':[1]})}, order=['Slide values'])
    spec = {'workbook_sha256':sha256(book), 'slides':[{'title':'Measured value','values':[{'sheet':'Slide values','cell':'A3'}], 'figures':[]}]}
    path = build_deck(book, spec, tmp_path/'deck.pptx')
    ppt = Presentation(path)
    assert len(ppt.slides) == 1
    assert 'Slide values!A3' in ppt.slides[0].notes_slide.notes_text_frame.text


def test_export_allows_explicit_source_identifiers(tmp_path):
    from fishsuite.report.slides import build_deck
    from fishsuite.report.workbook import write
    from fishsuite.report.provenance import sha256
    from pptx import Presentation
    table = pd.DataFrame(dict(stem=['2026_09_field'], group=['25KO']))
    book = write(tmp_path/'REPORT.xlsx', {'Micrographs': table, '25 assays': table}, order=['Micrographs', '25 assays'])
    slide = dict(title='2026_09_field / 25KO / 25 assays',
                 values=[dict(sheet='Micrographs', cell='A3', label='file stem')])
    path = build_deck(book, dict(workbook_sha256=sha256(book), slides=[slide]), tmp_path/'ids.pptx')
    assert Presentation(path).slides[0].shapes[0].text == slide['title']
