"""Mixed-model headline and supplementary analyses; decision 2026-09-07."""
from __future__ import annotations
import re
import warnings
import numpy as np
import pandas as pd
from .stats import welch

DESCRIPTION = "decision 2026-09-07 after data inspection; Welch on well means remains reported"
METHODS = (['lbfgs', 'bfgs'], ['powell'], ['nm'])
# Iteration caps bound one fit to about 10 s at about 100 nuclei (measured 2026-10-11: lbfgs ~36 ms,
# Powell/NM ~2.2 ms per likelihood evaluation; Powell ~85 and NM ~4 evaluations per iteration).
# statsmodels reports a fit that hits its cap as not converged, so it is never validated.
MAXITER = {'lbfgs': 100, 'powell': 50, 'nm': 1000}
# Validation gate (2026-10-11), invariant to response units: I = -(H + H')/2 at the fit; every diagonal
# element of I > 0; the smallest eigenvalue of the diagonal-standardised matrix D^-1/2 I D^-1/2 > PD_TOL;
# and the residual variance (fit.scale) / pooled within-well variance of the analysed response > RESID_TOL.
# The within-well reference (y minus its well mean, df = n - wells) is unchanged by arm offsets and by
# between-well shifts; FOV variation inside a well stays in it, so a residual that collapses to ~0
# when values are constant inside each FOV is still caught (Codex delta review 2026-10-11).
PD_TOL = 1e-6
RESID_TOL = 1e-6
LLF_TIE = 1e-6  # valid fits within this REML llf of the best keep the METHODS order


def _within_well_variance(model):
    """Pooled variance of the response about its well means (df = n - number of wells)."""
    groups = [np.asarray(e, dtype=float) for e in model.endog_li]
    df = sum(len(g) for g in groups) - len(groups)
    if df < 1:
        return np.nan
    return float(sum(((g - g.mean()) ** 2).sum() for g in groups) / df)


def _fit_record(model, method):
    """One optimizer; returns the fit summary and the reason it is not valid, if any."""
    fit = model.fit(reml=True, method=method, maxiter=MAXITER[method[0]], disp=False)
    rec = dict(optimizer='/'.join(method), est=float(fit.fe_params['arm']), se=float(fit.bse_fe['arm']),
               p=float(fit.pvalues['arm']), llf=float(fit.llf), min_eig=np.nan, reason='',
               resid_frac=np.nan)
    within = _within_well_variance(model)
    if np.isfinite(within) and within > 0:
        rec['resid_frac'] = float(fit.scale) / within
    pd_reason = ''
    try:
        h, _ = fit.model.hessian(fit.params_object)
        info = -0.5 * (h + h.T)
        diag = np.diag(info)
        if not (diag > 0).all():
            pd_reason = 'information matrix has a non-positive diagonal element'
        else:
            rec['min_eig'] = float(np.linalg.eigvalsh(info / np.sqrt(np.outer(diag, diag))).min())
            if not rec['min_eig'] > PD_TOL:
                pd_reason = f"information matrix not positive definite (standardised min eigenvalue {rec['min_eig']:.3g})"
    except Exception as exc:
        pd_reason = f'information matrix not computable ({type(exc).__name__})'
    if not fit.converged:
        rec['reason'] = 'did not converge'
    elif not np.isfinite([rec['est'], rec['se'], rec['p'], rec['llf']]).all() or rec['se'] <= 0:
        rec['reason'] = 'non-finite REML log-likelihood, estimate or SE'
    elif not (np.isfinite(np.asarray(fit.params, dtype=float)).all() and np.isfinite(np.asarray(fit.bse_fe, dtype=float)).all()):
        rec['reason'] = 'non-finite parameters or fixed-effect SEs'
    elif pd_reason:
        rec['reason'] = pd_reason
    elif not (np.isfinite(within) and within > 0):
        rec['reason'] = 'no within-well variation in the response'
    elif not rec['resid_frac'] > RESID_TOL:
        rec['reason'] = f"residual variance collapsed ({rec['resid_frac']:.3g} of the within-well variance)"
    return rec


def mixed_model(data, test, reference):
    """REML arm effect, well random intercept and nested FOV variance component.

    P is a two-sided asymptotic Wald normal p, not a small-sample well t test.
    Every optimizer in METHODS is tried; the highest-REML fit that converged, is
    finite, passes the unit-invariant information gate above and has a non-collapsed residual is kept.
    Anything else is 'not_validated:<reason>' with missing inference.
    """
    out = dict(sensitivity_mixed_estimate=np.nan, sensitivity_mixed_se=np.nan,
               sensitivity_mixed_p=np.nan, sensitivity_mixed_status='missing nucleus-level values',
               sensitivity_mixed_warnings='', sensitivity_mixed_validation='not_validated:missing nucleus-level values',
               sensitivity_mixed_min_eig=np.nan, sensitivity_mixed_resid_frac=np.nan, sensitivity_mixed_optimizer='')
    data = data.copy()
    data['value'] = pd.to_numeric(data['value'], errors='coerce')
    data = data.loc[data.group.isin([test, reference]) & np.isfinite(data.value)].dropna(subset=['well_id','image'])
    for arm, label in [(test,'test'), (reference,'reference')]:
        sub = data.loc[data.group.eq(arm)]
        out.update({f'sensitivity_n_nuclei_{label}':len(sub),
                    f'sensitivity_n_fovs_{label}':len(sub[['well_id','image']].drop_duplicates()),
                    f'sensitivity_n_wells_{label}':sub.well_id.nunique(),
                    f'sensitivity_nucleus_median_{label}':float(sub.value.median()) if len(sub) else np.nan})
    def blocked(reason):
        out.update(sensitivity_mixed_status=f'blocked: {reason}', sensitivity_mixed_validation=f'not_validated:{reason}')
        return out
    if any(out[f'sensitivity_n_wells_{label}'] < 2 for label in ['test','reference']):
        return blocked('fewer than two wells per arm with finite nucleus values')
    try:
        from statsmodels.regression.mixed_linear_model import MixedLM
    except ImportError:
        return blocked('statsmodels missing; installation prohibited')
    data['arm'] = data.group.eq(test).astype(float)
    data['well'] = data.groupby(['group','well_id'], sort=True).ngroup().astype(str)
    # Formula variance components are constructed separately inside each well.
    fits, caught_all = [], []
    for method in METHODS:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                # Sparse Nelder-Mead hung inside scipy.sparse on zero-variance FOVs (2026-10-11);
                # the fallback optimizers use the identical model in dense storage.
                model = MixedLM.from_formula('value ~ arm', data=data, groups='well',
                            re_formula='1', vc_formula={'fov':'0 + C(image)'}, use_sparse=method[0] == 'lbfgs')
                rec = _fit_record(model, method)
            rec['warnings'] = '; '.join(dict.fromkeys(str(w.message) for w in caught))
        except Exception as exc:
            rec = dict(optimizer='/'.join(method), reason=f'{type(exc).__name__}: {exc}', min_eig=np.nan,
                       resid_frac=np.nan, warnings='')
        fits.append(rec)
        caught_all.append(rec['warnings'])
    valid = [f for f in fits if not f['reason']]
    if not valid:
        out['sensitivity_mixed_warnings'] = '; '.join(dict.fromkeys(w for w in caught_all if w))
        eigs = [f['min_eig'] for f in fits if np.isfinite(f['min_eig'])]
        out['sensitivity_mixed_min_eig'] = float(min(eigs)) if eigs else np.nan
        reason = 'mixed model did not converge' if all(f['reason'] == 'did not converge' for f in fits) else \
            'mixed model not validated (' + '; '.join(f"{f['optimizer']}: {f['reason']}" for f in fits) + ')'
        return blocked(reason)
    top = max(f['llf'] for f in valid)
    best = next(f for f in valid if f['llf'] >= top - LLF_TIE)
    out.update(sensitivity_mixed_estimate=best['est'], sensitivity_mixed_se=best['se'],
               sensitivity_mixed_p=best['p'], sensitivity_mixed_status='ok',
               sensitivity_mixed_warnings=best['warnings'], sensitivity_mixed_validation='validated',
               sensitivity_mixed_min_eig=best['min_eig'], sensitivity_mixed_resid_frac=best['resid_frac'],
               sensitivity_mixed_optimizer=best['optimizer'])
    return out


def scalar_sensitivities(ep, nuclei, field, test_values, ref_values, test, reference):
    from scipy.stats import ttest_ind
    out = dict(sensitivity_description=DESCRIPTION,
               sensitivity_student_well_p=np.nan, sensitivity_student_well_t=np.nan)
    if len(test_values) >= 2 and len(ref_values) >= 2:
        t, p = ttest_ind(test_values, ref_values, equal_var=True)
        out.update(sensitivity_student_well_p=float(p), sensitivity_student_well_t=float(t))
    fvals = {}
    for group, arm in [(test,'test'), (reference,'reference')]:
        fvals[arm] = pd.to_numeric(field.loc[field.group.eq(group),'field_value'],errors='coerce').dropna().to_numpy() if len(field) else np.array([])
    out['sensitivity_welch_fov_p'] = welch(fvals['test'], fvals['reference'])['p_welch']
    data = pd.DataFrame(columns=['group','well_id','image','value'])
    if nuclei is not None and ep.level == 'nucleus' and ep.column in nuclei:
        data = nuclei.copy()
        if 'secondary_only' in data:
            data = data.loc[~data.secondary_only.astype(bool)]
        if ep.usability_flag:
            data = data.loc[data[ep.usability_flag].fillna(False).astype(bool)] if ep.usability_flag in data else data.iloc[:0]
        data = data[['group','well_id','image',ep.column]].rename(columns={ep.column:'value'})
    out.update(mixed_model(data, test, reference))
    if ep.level != 'nucleus':
        out['sensitivity_mixed_status'] = 'missing: endpoint has only field-level measurements'
        out['sensitivity_mixed_validation'] = 'not_validated:not fitted, endpoint has only field-level measurements'
    # FOV n describes finite field values, including field-only endpoints.
    for arm in ['test','reference']:
        out[f'sensitivity_n_fovs_{arm}'] = len(fvals[arm])
    return out


def flag_fov_outliers(field):
    """Leave-one-out sample-SD z; strict 2.5 threshold; one maximum per well/endpoint.

    Zero SD of the others gives signed infinity for a different value, zero for
    identical values. Ties are resolved by image name, identically in both arms.
    """
    rows = []
    for (endpoint, group, well), sub in field.groupby(['endpoint','group','well_id'],sort=True):
        sub = sub.loc[np.isfinite(pd.to_numeric(sub.field_value,errors='coerce'))].sort_values('image')
        if len(sub) < 3:
            continue
        candidates = []
        values = sub.field_value.to_numpy(dtype=float)
        for i, (_, row) in enumerate(sub.iterrows()):
            others = np.delete(values,i)
            delta, sd = values[i]-others.mean(), others.std(ddof=1)
            z = delta/sd if sd > 0 else (np.copysign(np.inf,delta) if delta != 0 else 0.)
            if abs(z) > 2.5:
                candidates.append(dict(endpoint=endpoint,arm=group,well=well,image=row.image,z=float(z),n_nuclei=int(row.n_nuclei_nonmissing)))
        if candidates:
            rows.append(max(candidates,key=lambda r:abs(r['z'])))
    return pd.DataFrame(rows,columns=['endpoint','arm','well','image','z','n_nuclei'])


def outlier_sensitivity(field, contrasts, alpha=.05):
    flags = flag_fov_outliers(field)
    rows = [dict(record_type='flagged FOV', **row) for row in flags.to_dict('records')]
    for _, contrast in contrasts.iterrows():
        sub = field.loc[field.endpoint.eq(contrast.endpoint)].copy()
        flagged = flags.loc[flags.endpoint.eq(contrast.endpoint)]
        excluded = set(zip(flagged.arm,flagged.well,flagged.image))
        keep = [key not in excluded for key in zip(sub.group,sub.well_id,sub.image)]
        after = sub.loc[keep].groupby(['group','well_id']).field_value.mean()
        def values(arm):
            return after.xs(arm,level='group').dropna().to_numpy() if arm in after.index.get_level_values('group') else np.array([])
        result = welch(values(contrast.test_group),values(contrast.reference_group),alpha)
        rows.append(dict(record_type='contrast',endpoint=contrast.endpoint,arm=contrast.test_group,
             reference_arm=contrast.reference_group,without_exclusion_welch_p=contrast.p_welch,
             without_exclusion_difference=contrast['diff'],with_exclusion_welch_p=result['p_welch'],
             with_exclusion_difference=result['diff'],with_exclusion_mean_test=result['mean_test'],
             with_exclusion_mean_reference=result['mean_ref'],with_exclusion_n_wells_test=result['n_test'],
             with_exclusion_n_wells_reference=result['n_ref']))
    return pd.DataFrame(rows)


LEGACY_VALIDATION_COLUMN = 'sensitivity_mixed_validation'
# The single accessor for mixed-model inference (2026-10-11). Every other module reads or writes these
# keys only through the functions below; tests/test_mixed_p_single_accessor_2026_10_11.py enforces it.
# Consumers never recompute or substitute statistics at read time: a table, workbook or slide cache whose
# mixed inference is not explicitly validated is refused and must be rebuilt from its source run.
MIXED_ONLY_KEYS = ('p_mixed', 'sensitivity_mixed_p', 'p_mixed_holm', 'mixed_p')
HEADLINE_KEYS = ('p_headline', 'p_headline_holm')
MIXED_ALIAS = 'mixed_p'
WELCH_HEADLINE = 'Welch (well means)'
MIXED_HEADLINE = 'mixed model'
# First fishsuite commit whose outputs carry validated mixed inference under the final rules.
MIXED_VALIDATION_COMMIT = '85df5d1'
REBUILD_MESSAGE = f'legacy statistics: rebuild from the source run with fishsuite >= {MIXED_VALIDATION_COMMIT}'


class LegacyStatisticsError(ValueError):
    """Persisted mixed-model inference without a validated marker; the consumer refuses to proceed."""


def _validated(row) -> bool:
    return row.get(LEGACY_VALIDATION_COLUMN, None) == 'validated'


def mixed_value(row, key='p_mixed') -> float:
    """A mixed-only quantity of one row, or NaN unless that row's fit is validated."""
    raw = row.get(key, np.nan)
    if key == 'p_mixed' and pd.isna(raw):
        raw = row.get('sensitivity_mixed_p', np.nan)
    value = pd.to_numeric(raw, errors='coerce')
    return float(value) if _validated(row) and np.isfinite(value) else np.nan


def mixed_validated(row) -> bool:
    """A mixed p may be shown only for an explicitly validated fit with a finite p."""
    return bool(np.isfinite(mixed_value(row)))


def headline_value(row, key='p_headline') -> float:
    """Persisted headline p (or its Holm adjustment); NaN unless validated or a build-time Welch headline."""
    value = pd.to_numeric(row.get(key, np.nan), errors='coerce')
    if not np.isfinite(value):
        return np.nan
    return float(value) if _validated(row) or row.get('headline_test', None) == WELCH_HEADLINE else np.nan


def headline_holm_value(row) -> float:
    """Holm-adjusted headline p of one row; NaN under the same rule as ``headline_value``."""
    return headline_value(row, 'p_headline_holm')


def mixed_p_text(row, sep: str = " ") -> str:
    """Footer/notes text for the mixed model; never a p from a fit that is not validated."""
    from .stats import fmt_p
    status = str(row.get('sensitivity_mixed_status', ''))
    if mixed_validated(row):
        return f"mixed model p{sep}{fmt_p(mixed_value(row))}"
    validation = row.get(LEGACY_VALIDATION_COLUMN, None)
    if validation is None or (isinstance(validation, float) and np.isnan(validation)):
        if status == 'ok':
            return "mixed model not validated (legacy result)"
    if 'field-level' in status:
        return "mixed model not fitted (field-level endpoint)"
    if 'converg' in status and 'not validated' not in status:
        return "mixed model did not converge"
    return "mixed model not validated"


def require_validated_mixed(table, where='table'):
    """Return ``table`` unchanged, or raise ``LegacyStatisticsError``.

    A table carrying any mixed or headline p column must carry ``sensitivity_mixed_validation``; every row
    with a finite mixed-only value, or a mixed-model headline, must be 'validated'. Nothing is recomputed.
    """
    present = [k for k in MIXED_ONLY_KEYS + HEADLINE_KEYS if k in table.columns]
    if not present:
        return table
    if LEGACY_VALIDATION_COLUMN not in table.columns:
        raise LegacyStatisticsError(f'{REBUILD_MESSAGE} ({where}: no {LEGACY_VALIDATION_COLUMN} column)')
    ok = table[LEGACY_VALIDATION_COLUMN].eq('validated')
    bad = pd.Series(False, index=table.index)
    for key in MIXED_ONLY_KEYS:
        if key in table.columns:
            bad |= pd.to_numeric(table[key], errors='coerce').notna() & ~ok
    if 'headline_test' in table.columns:
        bad |= table['headline_test'].eq(MIXED_HEADLINE) & ~ok
    if bad.any():
        first = table.loc[bad].iloc[0]
        raise LegacyStatisticsError(f"{REBUILD_MESSAGE} ({where}: endpoint {first.get('endpoint', '?')} has a mixed p "
                                    f"whose fit is not validated)")
    return table


def require_validated_rows(rows, where='sheet'):
    """Worksheet rows (description, header, data...): refuse unvalidated mixed inference; rows unchanged."""
    rows = list(rows)
    if len(rows) >= 2 and rows[1] is not None:
        header = [str(h) for h in rows[1]]
        if set(MIXED_ONLY_KEYS + HEADLINE_KEYS) & set(header):
            data = [list(r) + [None] * (len(header) - len(r)) for r in rows[2:]]
            require_validated_mixed(pd.DataFrame([r[:len(header)] for r in data], columns=header), where)
    return rows


def require_validated_slide_cache(slide_rows, sheet_rows):
    """Refuse a 'Slide values' cache whose mixed-derived entries do not trace to a validated source row."""
    if len(slide_rows) < 3:
        return
    columns = [str(c) for c in slide_rows[1]]
    keys = MIXED_ONLY_KEYS + HEADLINE_KEYS
    for raw in slide_rows[2:]:
        entry = dict(zip(columns, raw))
        label = str(entry.get('label', ''))
        if not any(label == k or label.endswith(': ' + k) for k in keys) or entry.get('value') is None:
            continue
        source = sheet_rows.get(entry.get('source_sheet'))
        header = [str(h) for h in source[1]] if source and len(source) > 1 else []
        if LEGACY_VALIDATION_COLUMN not in header:
            raise LegacyStatisticsError(f'{REBUILD_MESSAGE} (Slide values: {label} has no validated source)')
        from openpyxl.utils import coordinate_to_tuple
        row_index, _ = coordinate_to_tuple(str(entry.get('source_cell')))
        record = dict(zip(header, source[row_index - 1])) if row_index - 1 < len(source) else {}
        if record.get(LEGACY_VALIDATION_COLUMN) != 'validated' and not (
                label.endswith(HEADLINE_KEYS) and record.get('headline_test') == WELCH_HEADLINE):
            raise LegacyStatisticsError(f'{REBUILD_MESSAGE} (Slide values: {label} traces to an unvalidated fit)')


def add_mixed_columns(table, headline=True):
    """Build time only, on fresh ``mixed_model`` output: ``p_mixed`` from validated fits; optionally the
    headline (validated mixed p, else the well-level Welch p) and its ``headline_test`` label."""
    out = table.copy()
    if LEGACY_VALIDATION_COLUMN not in out.columns:
        raise LegacyStatisticsError(f'{REBUILD_MESSAGE} (fresh contrasts without {LEGACY_VALIDATION_COLUMN})')
    ok = out[LEGACY_VALIDATION_COLUMN].eq('validated')
    out['p_mixed'] = pd.to_numeric(out['sensitivity_mixed_p'], errors='coerce').where(ok)
    if headline:
        out['p_headline'] = out['p_mixed'].fillna(out['p_welch'])
        out['headline_test'] = np.where(out['p_mixed'].notna(), MIXED_HEADLINE, WELCH_HEADLINE)
    return out


def add_headline_holm(table, alpha, family_cols):
    """Build time only: Holm over each declared family's headline p (members = ``in_holm_family``), the
    finite-p family size, the mixed-only Holm column and the headline significance flag."""
    from .stats import holm
    out = require_validated_mixed(table, 'headline Holm input').copy()
    member = out['in_holm_family'].fillna(False).astype(bool)
    out['p_headline_holm'] = np.nan
    out['p_mixed_holm'] = np.nan
    out['holm_family_size'] = np.nan
    for _, idx in out.groupby(list(family_cols)).groups.items():
        ids = [i for i in idx if member.at[i]]
        if not ids:
            continue
        adjusted = holm(out.loc[ids, 'p_headline'].tolist())
        out.loc[ids, 'p_headline_holm'] = adjusted
        out.loc[ids, 'p_mixed_holm'] = adjusted
        out.loc[ids, 'holm_family_size'] = int(np.isfinite(out.loc[ids, 'p_headline'].to_numpy(float)).sum())
    out.loc[out['p_mixed'].isna(), 'p_mixed_holm'] = np.nan
    flag = (out['p_headline_holm'] < alpha).astype(object)
    flag[~member] = pd.NA
    out['significant_headline_holm_0p05'] = flag
    raw = (out['p_mixed'] < alpha).astype(object)
    raw[out['p_mixed'].isna()] = pd.NA
    out['significant_mixed_raw_0p05'] = raw
    return out


# Stored build-time decision behind each p column used in the floor comparison.
FLOOR_FLAGS = {'p_welch': 'significant_raw_0p05', 'mixed_p': 'significant_mixed_raw_0p05'}


def stored_flag(row, p_column):
    """The stored build-time significance flag for one row's p; refuse when the flag is absent."""
    flag = FLOOR_FLAGS[p_column]
    if flag not in row or pd.isna(row[flag]):
        raise LegacyStatisticsError(f'{REBUILD_MESSAGE} (no stored {flag} for {row.get("endpoint", "?")})')
    value = row[flag]
    return value if isinstance(value, bool) else str(value).strip().lower() in ('true', '1')


def with_mixed_alias(table):
    """Validated contrasts with the short ``mixed_p`` alias column (a copy of ``p_mixed``)."""
    out = require_validated_mixed(table, 'mixed_p alias input').copy()
    out[MIXED_ALIAS] = out['p_mixed'] if 'p_mixed' in out else np.nan
    return out


# ---------------------------------------------------------------- statistics generation stamp (2026-10-12)
# One mechanism for every persisted artifact that carries statistics. Writers in fixed code stamp each
# statistical table with the generation and commit; every reader of persisted p, Holm, significance or MDE
# refuses a table without a stamp >= STATS_GENERATION. A missing schema, missing columns, an absent slide
# cache schema and every pre-stamp build (ba24827 included) all fail the same check.
STATS_GENERATION = 2
STATS_GENERATION_COLUMN = 'fishsuite_stats_generation'
STATS_COMMIT_COLUMN = 'fishsuite_stats_commit'
STATS_COMMIT = MIXED_VALIDATION_COMMIT
_STAT_COLUMN = re.compile(r'(^p$|^p_|_p$|_p_|holm|^significant|mde|^ci_|_ci_|ci_low|ci_high|^ci95)', re.I)
_SLIDE_CACHE = {'source_sheet', 'source_cell', 'value'}


def statistical_columns(columns):
    """Column names that carry p, Holm, significance, MDE or confidence-interval values."""
    return [str(c) for c in columns if _STAT_COLUMN.search(str(c))]


def is_statistical(columns) -> bool:
    names = {str(c) for c in columns}
    return bool(statistical_columns(names)) or _SLIDE_CACHE <= names


def _check_stamp(table, where):
    if STATS_GENERATION_COLUMN not in table.columns:
        raise LegacyStatisticsError(f'{REBUILD_MESSAGE} ({where}: no {STATS_GENERATION_COLUMN} stamp)')
    generation = pd.to_numeric(table[STATS_GENERATION_COLUMN], errors='coerce')
    if len(table) and not (generation >= STATS_GENERATION).all():
        raise LegacyStatisticsError(f'{REBUILD_MESSAGE} ({where}: {STATS_GENERATION_COLUMN} below {STATS_GENERATION})')


def require_current_statistics(table, where='table', force=False):
    """Reader gate: a persisted statistical table must carry a current stamp and validated mixed inference."""
    if force or is_statistical(table.columns):
        _check_stamp(table, where)
    return require_validated_mixed(table, where)


def check_writable(table, where='table'):
    """Writer pre-flight: validated mixed inference, and any stamp it already carries must be current."""
    require_validated_mixed(table, where)
    if STATS_GENERATION_COLUMN in table.columns:
        _check_stamp(table, where)
    return table


def stamp_statistics(table, where='table', force=False):
    """Writer: a copy stamped with the current generation and commit when the table is statistical."""
    check_writable(table, where)
    if STATS_GENERATION_COLUMN in table.columns or not (force or is_statistical(table.columns)):
        return table  # an already-current stamp keeps the commit that built it
    out = table.copy()
    out[STATS_GENERATION_COLUMN] = STATS_GENERATION
    out[STATS_COMMIT_COLUMN] = STATS_COMMIT
    return out


def unstamp(table):
    """Build step that combines gated inputs into a new table: drop inherited stamps so the writer stamps the
    combined table with the code that built it (a partially stamped table is refused by the writer)."""
    return table.drop(columns=[c for c in (STATS_GENERATION_COLUMN, STATS_COMMIT_COLUMN) if c in table.columns])


def require_current_rows(rows, where='sheet', force=False):
    """Reader gate on worksheet rows (description row, header row, data rows); rows returned unchanged."""
    rows = list(rows)
    if len(rows) >= 2 and rows[1] is not None:
        header = [str(h) for h in rows[1]]
        data = [list(r)[:len(header)] + [None] * max(0, len(header) - len(r)) for r in rows[2:]]
        require_current_statistics(pd.DataFrame(data, columns=header), where, force)
    return rows


def require_current_book(book):
    """Reader gate on every worksheet of an openpyxl workbook written with a description row."""
    for ws in book.worksheets:
        require_current_rows([list(r) for r in ws.iter_rows(values_only=True)], ws.title)
    return book


def read_stats_csv(path, where=None, **kwargs):
    return require_current_statistics(pd.read_csv(path, **kwargs), where or str(path))


def read_stats_excel(path, sheet_name=None, header=1, force=False):
    """Read persisted workbook sheet(s); every sheet passes the reader gate."""
    tables = pd.read_excel(path, sheet_name=sheet_name, header=header)
    if isinstance(tables, dict):
        return {name: require_current_statistics(t, f'{path}:{name}', force) for name, t in tables.items()}
    return require_current_statistics(tables, f'{path}:{sheet_name}', force)
