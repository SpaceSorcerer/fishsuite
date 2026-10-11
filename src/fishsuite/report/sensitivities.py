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
PD_TOL = 1e-8  # all eigenvalues of symmetrised -H > PD_TOL * max(1, max|eigenvalue|)
LLF_TIE = 1e-6  # valid fits within this REML llf of the best keep the METHODS order


def _fit_record(model, method):
    """One optimizer; returns the fit summary and the reason it is not valid, if any."""
    fit = model.fit(reml=True, method=method, maxiter=MAXITER[method[0]], disp=False)
    rec = dict(optimizer='/'.join(method), est=float(fit.fe_params['arm']), se=float(fit.bse_fe['arm']),
               p=float(fit.pvalues['arm']), llf=float(fit.llf), min_eig=np.nan, reason='')
    try:
        h, _ = fit.model.hessian(fit.params_object)
        ev = np.linalg.eigvalsh(-0.5 * (h + h.T))
        rec['min_eig'], pd_ok = float(ev.min()), bool(ev.min() > PD_TOL * max(1.0, float(np.abs(ev).max())))
    except Exception as exc:
        pd_ok, rec['reason'] = False, f'information matrix not computable ({type(exc).__name__})'
    if not fit.converged:
        rec['reason'] = 'did not converge'
    elif not np.isfinite([rec['est'], rec['se'], rec['p'], rec['llf']]).all() or rec['se'] <= 0:
        rec['reason'] = 'non-finite REML log-likelihood, estimate or SE'
    elif not (np.isfinite(np.asarray(fit.params, dtype=float)).all() and np.isfinite(np.asarray(fit.bse_fe, dtype=float)).all()):
        rec['reason'] = 'non-finite parameters or fixed-effect SEs'
    elif not pd_ok:
        rec['reason'] = rec['reason'] or f"information matrix not positive definite (min eigenvalue {rec['min_eig']:.3g})"
    return rec


def mixed_model(data, test, reference):
    """REML arm effect, well random intercept and nested FOV variance component.

    P is a two-sided asymptotic Wald normal p, not a small-sample well t test.
    Every optimizer in METHODS is tried; the highest-REML fit that converged, is
    finite and has a positive-definite observed information matrix is kept.
    Anything else is 'not_validated:<reason>' with missing inference.
    """
    out = dict(sensitivity_mixed_estimate=np.nan, sensitivity_mixed_se=np.nan,
               sensitivity_mixed_p=np.nan, sensitivity_mixed_status='missing nucleus-level values',
               sensitivity_mixed_warnings='', sensitivity_mixed_validation='not_validated:missing nucleus-level values',
               sensitivity_mixed_min_eig=np.nan, sensitivity_mixed_optimizer='')
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
            rec = dict(optimizer='/'.join(method), reason=f'{type(exc).__name__}: {exc}', min_eig=np.nan, warnings='')
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
               sensitivity_mixed_min_eig=best['min_eig'], sensitivity_mixed_optimizer=best['optimizer'])
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
