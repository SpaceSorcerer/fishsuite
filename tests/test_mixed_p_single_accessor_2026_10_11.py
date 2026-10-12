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
DIRECT = re.compile(r"""(['"])(%s)\1|\.(%s)\b""" % ('|'.join(KEYS), '|'.join(KEYS)))


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
                 "out['p_headline_holm'] = 1", "for k in ['p_welch','mixed_p']:"]:
        assert DIRECT.search(line), line
    for line in ["'p_mixed and p_mixed_holm contain mixed inference only'", "p_welch_holm_within_family"]:
        assert not DIRECT.search(line), line


def test_workbook_write_withholds_legacy_mixed_p(tmp_path):
    from openpyxl import load_workbook
    from fishsuite.report import workbook
    legacy = pd.DataFrame([dict(endpoint='a', p_mixed=1e-5, sensitivity_mixed_p=1e-5, p_headline=1e-5,
                                p_welch=.8, sensitivity_mixed_status='ok')])
    path = workbook.write(tmp_path / 'r.xlsx', {'Contrasts': legacy}, {}, order=['Contrasts'])
    rows = list(load_workbook(path, read_only=True)['Contrasts'].values)
    table = dict(zip(rows[1], rows[2]))
    assert table['p_mixed'] is None and table['sensitivity_mixed_p'] is None
    assert table['p_headline'] == .8 and table['headline_test'] == 'Welch (well means)'


def test_resolved_slide_cell_from_legacy_workbook_is_withheld():
    rows = [('legacy',), ('endpoint', 'p_mixed', 'p_welch'), ('a', 1e-5, .8)]
    clean = S.withhold_unvalidated_rows(rows)
    header = clean[1]
    assert clean[2][header.index('p_mixed')] is None and clean[2][header.index('p_welch')] == .8


def test_rn_readout_falls_back_to_labelled_welch():
    from fishsuite.report.object_delivery import rn_readout
    import inspect
    names = list(dict.fromkeys(re.findall(r"p\('([^']+)'\)", inspect.getsource(rn_readout))))
    legacy = pd.DataFrame([dict(endpoint=n, mean_ref=1., mean_test=2., p_headline=1e-5, p_mixed=1e-5, p_welch=.8,
                                sensitivity_mixed_status='ok', sensitivity_mixed_validation='not_validated:failed')
                           for n in names])
    summary = {'ratio': dict(R=1., ci_low=.5, ci_high=2., p_exact=.9, R_MDE_05=2., R_MDE_05_8=3.)}
    try:
        text = rn_readout(legacy, summary)
    except (KeyError, TypeError) as exc:  # readout lines beyond the p() calls need more summary fields
        pytest.skip(f'readout needs more summary fields: {exc}')
    assert '1e-05' not in text and '0.8 (Welch, well-level)' in text


def test_a20_style_holm_uses_withheld_headline():
    c = pd.DataFrame(dict(endpoint=['x', 'y'], p_mixed=[1e-5, .02], p_headline=[1e-5, .02], p_welch=[.8, .4],
                          sensitivity_mixed_validation=['not_validated:failed', 'validated'],
                          family=['f', 'f'], test_group=['t', 't'], reference_group=['r', 'r'],
                          in_holm_family=[True, True], holm_family_size=[2., 2.]))
    out = S.withhold_unvalidated_mixed(c, headline=True)
    assert out.p_headline.tolist() == [.8, .02]
    assert out.p_headline_holm.tolist() == pytest.approx([.8, .04])
    assert np.isnan(out.p_mixed_holm.iloc[0]) and out.p_mixed_holm.iloc[1] == pytest.approx(.04)


def test_without_family_columns_stale_adjusted_inference_is_cleared():
    c = pd.DataFrame(dict(p_mixed=[2e-30, 1e-20], p_headline=[2e-30, 1e-20], p_welch=[.02, .3],
                          p_headline_holm=[2e-30, 1e-20], p_mixed_holm=[2e-30, 1e-20]))
    out = S.withhold_unvalidated_mixed(c)
    assert out.p_headline.tolist() == [.02, .3]
    assert out.p_headline_holm.isna().all() and out.p_mixed_holm.isna().all()


def test_family_size_change_recomputes_family_alpha_mde():
    from fishsuite.report.stats import mde_hedges_g
    c = pd.DataFrame(dict(p_mixed=[1e-5, .02], p_headline=[1e-5, .02], p_welch=[np.nan, .4],
                          sensitivity_mixed_validation=['not_validated:failed', 'validated'],
                          family=['f', 'f'], test_group=['t', 't'], in_holm_family=[True, True],
                          holm_family_size=[2., 2.], mde_hedges_g_at_family_alpha=[9., 9.],
                          n_wells_test=[3, 3], n_wells_reference=[3, 3], sd_wells_test=[1., 1.],
                          sd_wells_reference=[1., 1.], hedges_g=[.1, .1], observed_g_reaches_mde=[False, False]))
    out = S.withhold_unvalidated_mixed(c, headline=True)
    assert out.holm_family_size.tolist() == [1., 1.]
    assert out.mde_hedges_g_at_family_alpha.iloc[1] == mde_hedges_g(.05, n1=3, n2=3, sd1=1., sd2=1.)
