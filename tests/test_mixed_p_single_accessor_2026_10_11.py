"""Mixed-model inference is read or written only through the accessor in report/sensitivities.py (2026-10-11).

A direct reference to a mixed or headline p key anywhere else in src/ or scripts/ fails this test, so a new
table writer, readout or slide path cannot publish an unvalidated mixed p by bypassing the accessor.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fishsuite.report import sensitivities as S

ROOT = Path(__file__).resolve().parents[1]
ACCESSOR = ROOT / 'src' / 'fishsuite' / 'report' / 'sensitivities.py'
KEYS = S.MIXED_ONLY_KEYS + S.HEADLINE_KEYS
# A quoted exact key ('p_mixed', "p_headline_holm") or an attribute read (.p_mixed). Prose that merely
# mentions a key inside a longer string (e.g. 'p_mixed and p_mixed_holm contain...') is not a read.
_ALT = '|'.join(KEYS)
# Also keyword construction such as dict(p_mixed=...) or f(p_headline=...), but not comparisons (==).
DIRECT = re.compile(r"""(['"])(%s)\1|\.(%s)\b|\b(%s)\s*=(?!=)""" % (_ALT, _ALT, _ALT))


def _sources():
    for base in (ROOT / 'src', ROOT / 'scripts'):
        for path in sorted(base.rglob('*.py')):
            if path.resolve() != ACCESSOR.resolve() and '__pycache__' not in path.parts:
                yield path


def test_no_direct_mixed_key_access_outside_the_accessor():
    hits = []
    for path in _sources():
        for number, line in enumerate(path.read_text(encoding='utf-8', errors='replace').splitlines(), 1):
            if DIRECT.search(line):
                hits.append(f'{path.relative_to(ROOT)}:{number}: {line.strip()[:120]}')
    assert not hits, 'direct mixed/headline p access outside sensitivities.py:\n' + '\n'.join(hits)


def test_the_pattern_catches_reads_and_writes_and_ignores_prose():
    for line in ["x = row['p_mixed']", 'c.p_headline.lt(.05)', "df.get('sensitivity_mixed_p')",
                 "out['p_headline_holm'] = 1", "for k in ['p_welch','mixed_p']:",
                 "row = dict(p_mixed=1e-5, p_welch=.8)", "f(p_headline =x)", "dict(endpoint='a', sensitivity_mixed_p=p)"]:
        assert DIRECT.search(line), line
    for line in ["'p_mixed and p_mixed_holm contain mixed inference only'", "p_welch_holm_within_family",
                 "if x == p_mixed_threshold:", "p_mixed_holm_note = 1"]:
        assert not DIRECT.search(line), line
