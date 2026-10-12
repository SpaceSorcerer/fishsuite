"""Native regeneration copies NOTHING from the legacy figures folder, images included (2026-10-12).

A rendered statistics figure is a valid PNG/TIFF; no content check can tell drawn p-values from microscopy.
"""
import gzip
import io
import json

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from fishsuite.config.schema import ConditionsCfg
from fishsuite.core.native_by_condition import regenerate, LEGACY_OMISSION_REASON


def _image(fmt, text=None):
    image = Image.fromarray(np.full((40, 120), 255, dtype=np.uint8))
    if text:
        ImageDraw.Draw(image).text((2, 10), text, fill=0)  # p-value drawn as pixels, no metadata
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


def test_regeneration_copies_no_legacy_file_and_logs_every_one(tmp_path):
    from test_native_by_condition import tiny_run
    run = tiny_run(tmp_path)
    micro, charts = run / 'figures' / 'per_image', run / 'figures' / 'per_condition'
    files = {
        micro / 'field01.png': _image('PNG'),                                  # valid raw-looking micrograph
        micro / 'field02.tif': _image('TIFF'),                                 # valid TIFF
        micro / 'drawn_p.png': _image('PNG', 'p=0.001'),                       # statistics drawn as pixels
        micro / 'late.svg': ('<svg>' + ' ' * 70000 + '<text>p=0.001</text></svg>').encode(),
        micro / 'wide.csv': ('x' * 70000 + ',p_mixed\n1,0.001\n').encode(),
        micro / 'single.csv': b'p_mixed\n0.001\n',
        micro / 'late.json': json.dumps([{'v': i} for i in range(1500)] + [{'p_mixed': .001}]).encode(),
        micro / 'renamed.png': b'endpoint,p_mixed\nx,0.001\n',
        micro / 'corrupt.xlsx': b'PK\x03\x04not a workbook',
        micro / 'malformed.json': b'{"p_mixed": 0.001,',
        micro / 'stats.csv.gz': gzip.compress(b'endpoint,p_mixed\nx,0.001\n'),
        charts / 'chart.png': _image('PNG', 'mixed model p 0.04'),
    }
    for path, data in files.items():
        path.write_bytes(data)
    out = tmp_path / 'new'
    regenerate(run, ConditionsCfg(groups={'WT': ['a', 'b'], 'KO': ['c', 'd']}), out)

    legacy_files = sorted(p.relative_to(run / 'figures').as_posix() for p in (run / 'figures').rglob('*') if p.is_file())
    assert not (out / 'figures' / 'per_image').exists()
    assert not [p for p in (out / 'figures' / 'per_well_supplementary').rglob('*') if p.name != 'LEGACY_COPY_LOG.csv']
    log = pd.read_csv(out / 'figures/per_well_supplementary/LEGACY_COPY_LOG.csv')
    assert sorted(log.legacy_file) == legacy_files
    assert set(log.decision) == {'excluded'} and set(log.reason) == {LEGACY_OMISSION_REASON}
    assert {'per_image/field01.png', 'per_image/field02.tif', 'per_image/drawn_p.png'} <= set(log.legacy_file)


def test_no_legacy_folder_still_writes_an_empty_log(tmp_path):
    from test_native_by_condition import tiny_run
    import shutil
    run = tiny_run(tmp_path)
    shutil.move(str(run / 'figures'), str(tmp_path / 'moved_figures'))
    out = tmp_path / 'new'
    regenerate(run, ConditionsCfg(groups={'WT': ['a', 'b'], 'KO': ['c', 'd']}), out)
    assert pd.read_csv(out / 'figures/per_well_supplementary/LEGACY_COPY_LOG.csv').empty
