"""Mixed-model headline and supplementary analyses; decision 2026-09-07."""
from __future__ import annotations
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


LEGACY_VALIDATION = 'not_validated:legacy result (no validation marker; built before 70500c0)'
# The single accessor for mixed-model inference (2026-10-11). Every other module reads or writes these
# keys only through the functions below; tests/test_mixed_p_single_accessor_2026_10_11.py enforces it.
MIXED_ONLY_KEYS = ('p_mixed', 'sensitivity_mixed_p', 'p_mixed_holm', 'mixed_p')
HEADLINE_KEYS = ('p_headline', 'p_headline_holm')
MIXED_ALIAS = 'mixed_p'
WELCH_HEADLINE = 'Welch (well means)'


def _validated(row) -> bool:
    return row.get('sensitivity_mixed_validation', None) == 'validated'


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
    """Headline p (or its Holm adjustment) of one row; NaN when it would carry unvalidated mixed inference."""
    value = pd.to_numeric(row.get(key, np.nan), errors='coerce')
    if not np.isfinite(value):
        return np.nan
    return float(value) if _validated(row) or row.get('headline_test', None) == WELCH_HEADLINE else np.nan


def headline_holm_value(row) -> float:
    """Holm-adjusted headline p of one row; NaN when it would carry unvalidated mixed inference."""
    return headline_value(row, 'p_headline_holm')


def mixed_p_text(row, sep: str = " ") -> str:
    """Footer/notes text for the mixed model; never a p from a fit that is not validated."""
    from .stats import fmt_p
    status = str(row.get('sensitivity_mixed_status', ''))
    if mixed_validated(row):
        return f"mixed model p{sep}{fmt_p(mixed_value(row))}"
    validation = row.get('sensitivity_mixed_validation', None)
    if validation is None or (isinstance(validation, float) and np.isnan(validation)) or 'legacy' in str(validation):
        if status == 'ok':
            return "mixed model not validated (legacy result)"
    if 'field-level' in status:
        return "mixed model not fitted (field-level endpoint)"
    if 'converg' in status and 'not validated' not in status:
        return "mixed model did not converge"
    return "mixed model not validated"


def withhold_unvalidated_mixed(contrasts, alpha=0.05, headline=False):
    """Produce or sanitise the mixed-model columns of a contrasts table.

    Mixed-only columns keep a value only where ``sensitivity_mixed_validation == 'validated'``; a table
    without that column was built before 70500c0 and loses all of them. ``p_mixed`` is created from
    ``sensitivity_mixed_p`` when absent. When the table has, or ``headline`` asks for, a headline, it is the
    validated mixed p else the well-level Welch p, labelled in ``headline_test``. Headline Holm, family
    size, significance and the family-alpha MDE are recomputed per family; if the family cannot be
    reconstructed, adjusted inference that a withheld row could have influenced is cleared, not kept.
    A table carrying none of the mixed or headline keys is returned unchanged.
    """
    from .stats import holm, mde_hedges_g
    c = contrasts.copy()
    if not (set(MIXED_ONLY_KEYS + HEADLINE_KEYS) & set(c.columns)):
        return c
    if 'sensitivity_mixed_validation' not in c:
        c['sensitivity_mixed_validation'] = LEGACY_VALIDATION
    c['sensitivity_mixed_validation'] = c['sensitivity_mixed_validation'].fillna(LEGACY_VALIDATION)
    ok = c['sensitivity_mixed_validation'].eq('validated')
    if 'p_mixed' not in c and 'sensitivity_mixed_p' in c:
        c['p_mixed'] = c['sensitivity_mixed_p']
    for col in MIXED_ONLY_KEYS:
        if col in c:
            c[col] = pd.to_numeric(c[col], errors='coerce').where(ok)
    if 'p_welch' not in c or not (headline or 'p_headline' in c):
        return c
    mixed = c['p_mixed'] if 'p_mixed' in c else pd.Series(np.nan, index=c.index)
    before = pd.to_numeric(c['p_headline'], errors='coerce') if 'p_headline' in c else None
    c['p_headline'] = mixed.fillna(pd.to_numeric(c['p_welch'], errors='coerce'))
    c['headline_test'] = np.where(mixed.notna(), 'mixed model', WELCH_HEADLINE)
    family_cols = ['family', 'test_group'] + (['reference_group'] if 'reference_group' in c else [])
    old_size = c['holm_family_size'].copy() if 'holm_family_size' in c else None
    if {'family', 'test_group', 'in_holm_family'} <= set(c.columns):
        member = c['in_holm_family'].fillna(False).astype(bool)
        c['p_headline_holm'] = np.nan
        c['holm_family_size'] = np.nan
        for _, idx in c.groupby(family_cols, dropna=False).groups.items():
            ids = [i for i in idx if member.at[i]]
            if ids:
                c.loc[ids, 'p_headline_holm'] = holm(c.loc[ids, 'p_headline'].tolist())
                c.loc[ids, 'holm_family_size'] = float(np.isfinite(c.loc[ids, 'p_headline'].to_numpy(float)).sum())
        c['p_mixed_holm'] = c['p_headline_holm'].where(mixed.notna())
        flag = (c['p_headline_holm'] < alpha).astype(object)
        flag[~member | c['p_headline_holm'].isna()] = pd.NA
        c['significant_headline_holm_0p05'] = flag
    elif before is not None and not np.allclose(before.to_numpy(float), c['p_headline'].to_numpy(float), equal_nan=True):
        for col in ('p_headline_holm', 'p_mixed_holm', 'significant_headline_holm_0p05'):
            if col in c:
                c[col] = np.nan
    mde_cols = {'mde_hedges_g_at_family_alpha', 'n_wells_test', 'n_wells_reference', 'sd_wells_test', 'sd_wells_reference'}
    if old_size is not None and mde_cols <= set(c.columns):
        changed = ~np.isclose(pd.to_numeric(old_size, errors='coerce').to_numpy(float),
                              c['holm_family_size'].to_numpy(float), equal_nan=True)
        for i in c.index[changed]:
            n1, n2, k = (pd.to_numeric(c.at[i, col], errors='coerce')
                         for col in ('n_wells_test', 'n_wells_reference', 'holm_family_size'))
            usable = bool(np.isfinite([n1, n2, k]).all() and min(n1, n2) >= 2 and k >= 1)
            c.at[i, 'mde_hedges_g_at_family_alpha'] = mde_hedges_g(
                alpha / float(k), n1=int(n1), n2=int(n2), sd1=float(c.at[i, 'sd_wells_test']),
                sd2=float(c.at[i, 'sd_wells_reference'])) if usable else np.nan
            if 'observed_g_reaches_mde' in c:
                g = pd.to_numeric(c.at[i, 'hedges_g'], errors='coerce') if 'hedges_g' in c else np.nan
                m = c.at[i, 'mde_hedges_g_at_family_alpha']
                c.at[i, 'observed_g_reaches_mde'] = bool(abs(g) >= m) if np.isfinite([g, m]).all() else pd.NA
    return c


def withhold_unvalidated_rows(rows):
    """Worksheet rows (description, header, data...) with mixed inference withheld, header extended as needed."""
    rows = [list(r) for r in rows]
    if len(rows) < 2 or not (set(MIXED_ONLY_KEYS + HEADLINE_KEYS) & {str(h) for h in rows[1]}):
        return rows
    header = [str(h) for h in rows[1]]
    frame = pd.DataFrame([r + [None] * (len(header) - len(r)) for r in rows[2:]], columns=header)
    clean = withhold_unvalidated_mixed(frame)
    clean = clean[header + [col for col in clean.columns if col not in header]]
    out = [rows[0], list(clean.columns)]
    for record in clean.itertuples(index=False, name=None):
        out.append([None if (isinstance(v, float) and np.isnan(v)) or v is pd.NA else v for v in record])
    return out


def with_mixed_alias(contrasts):
    """Contrasts with the short ``mixed_p`` alias column added, withheld exactly like ``p_mixed``."""
    c = contrasts.copy()
    c[MIXED_ALIAS] = c['p_mixed'] if 'p_mixed' in c else c.get('sensitivity_mixed_p', np.nan)
    return withhold_unvalidated_mixed(c)
