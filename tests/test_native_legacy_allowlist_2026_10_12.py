"""Native regeneration copies legacy assets fail-closed: only valid raw micrograph PNG/TIFF under per_image."""
import gzip
import io
import json
import zlib

import numpy as np
import pandas as pd
import pytest
from PIL import Image, PngImagePlugin

from fishsuite.config.schema import ConditionsCfg
from fishsuite.core.native_by_condition import regenerate, _legacy_decision


def _png(text=None):
    buffer = io.BytesIO()
    info = PngImagePlugin.PngInfo()
    if text:
        info.add_text('Description', text)
    Image.fromarray(np.arange(16, dtype=np.uint8).reshape(4, 4)).save(buffer, format='PNG', pnginfo=info)
    return buffer.getvalue()


def _tiff(description=None):
    buffer = io.BytesIO()
    kwargs = {'description': description} if description else {}
    Image.fromarray(np.arange(16, dtype=np.uint8).reshape(4, 4)).save(buffer, format='TIFF', **kwargs)
    return buffer.getvalue()


@pytest.fixture
def run_with_probes(tmp_path):
    from test_native_by_condition import tiny_run
    run = tiny_run(tmp_path)
    micro = run / 'figures' / 'per_image'
    elsewhere = run / 'figures' / 'per_condition'
    probes = {}

    def put(folder, name, data):
        (folder / name).write_bytes(data if isinstance(data, bytes) else data.encode('utf-8'))
        probes[f'{folder.name}/{name}'] = data

    # allowlisted raw micrographs
    put(micro, 'field01.png', _png())
    put(micro, 'field02.tif', _tiff())
    put(micro, 'field03.png', _png('Software: Pillow; exposure 700 ms'))
    # Codex confirm-review probes and fail-closed cases (all must be excluded)
    put(micro, 'late.svg', '<svg>' + ' ' * 70000 + '<text>p=0.001</text></svg>')            # p after 64 KB
    put(micro, 'late.png', _png())                                                           # its PNG twin
    put(micro, 'wide.csv', 'x' * 70000 + ',p_mixed\n1,0.001\n')                              # long first column
    put(micro, 'single.csv', 'p_mixed\n0.001\n')                                             # single column
    put(micro, 'huge.json', json.dumps({'rows': [{'v': i} for i in range(200000)] + [{'p_mixed': .001}]}))
    put(micro, 'late.json', json.dumps([{'v': i} for i in range(1500)] + [{'p_mixed': .001}]))
    put(micro, 'renamed.png', 'endpoint,p_mixed\nx,0.001\n')                                 # CSV renamed .png
    put(micro, 'corrupt.xlsx', b'PK\x03\x04not really a workbook')
    put(micro, 'malformed.json', '{"p_mixed": 0.001,')
    put(micro, 'stats.csv.gz', gzip.compress(b'endpoint,p_mixed\nx,0.001\n'))
    put(micro, 'notes.txt', 'descriptive notes')
    put(micro, 'noext', 'endpoint\tp\nx\t0.01\n')
    put(micro, 'inference.png', _png('Welch p = 0.003; mixed model p 0.001'))               # text chunk names p
    put(micro, 'truncated.png', _png()[:-20])                                                 # corrupt PNG
    put(micro, 'labelled.tif', _tiff('Holm-adjusted p 0.02'))                                # TIFF tag names Holm
    put(micro, 'fake.tif', 'endpoint,p_mixed\nx,0.001\n')                                    # not a TIFF
    put(elsewhere, 'chart.png', _png())                                                       # valid PNG, chart folder
    return run, probes


def test_only_allowlisted_raw_micrographs_are_copied(run_with_probes, tmp_path):
    run, probes = run_with_probes
    out = tmp_path / 'new'
    regenerate(run, ConditionsCfg(groups={'WT': ['a', 'b'], 'KO': ['c', 'd']}), out)
    copied = {p.relative_to(out / 'figures').as_posix() for p in (out / 'figures').rglob('*')
              if p.is_file() and 'per_image' in p.parts}
    assert copied == {'per_image/field01.png', 'per_image/field02.tif', 'per_image/field03.png'}
    for name in ('field01.png', 'field02.tif', 'field03.png'):
        assert (out / 'figures/per_image' / name).read_bytes() == (run / 'figures/per_image' / name).read_bytes()
    assert not any((out / 'figures/per_well_supplementary').rglob('*.png'))
    log = pd.read_csv(out / 'figures/per_well_supplementary/LEGACY_COPY_LOG.csv').set_index('legacy_file')
    assert len(log) == len(list((run / 'figures').rglob('*.*'))) + 1  # every legacy file logged (+ 'noext')
    assert (log.decision == 'copied').sum() == 3
    assert 'same-stem sibling' in log.loc['per_image/late.png', 'reason']
    assert 'not a valid PNG' in log.loc['per_image/renamed.png', 'reason']
    assert 'embedded text names inference' in log.loc['per_image/inference.png', 'reason']
    assert 'not a valid PNG' in log.loc['per_image/truncated.png', 'reason']
    assert 'embedded text names inference' in log.loc['per_image/labelled.tif', 'reason']
    assert 'not a valid TIFF' in log.loc['per_image/fake.tif', 'reason']
    assert log.loc['per_condition/chart.png', 'reason'].startswith('outside figures/per_image')
    for name in ('late.svg', 'wide.csv', 'single.csv', 'huge.json', 'late.json', 'corrupt.xlsx', 'malformed.json',
                 'stats.csv.gz', 'notes.txt', 'noext'):
        assert log.loc[f'per_image/{name}', 'decision'] == 'excluded', name


def test_png_with_bad_crc_is_excluded(tmp_path):
    legacy = tmp_path / 'figures'
    (legacy / 'per_image').mkdir(parents=True)
    data = bytearray(_png())
    data[-5] ^= 0xFF  # corrupt the IEND CRC
    path = legacy / 'per_image' / 'f.png'
    path.write_bytes(bytes(data))
    copy, reason = _legacy_decision(path, legacy)
    assert not copy and 'not a valid PNG' in reason
