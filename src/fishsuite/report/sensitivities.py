"""Sensitivity analyses only: the pre-specified gate remains well-mean Welch."""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from .stats import welch

DESCRIPTION = "added after inspection of the data; the pre-specified gate is Welch on well means"


def mixed_model(data, test, reference):
    """REML arm effect, well random intercept and nested FOV variance component.

    P is a two-sided asymptotic Wald normal p, not a small-sample well t test.
    Failed/nonconverged fits explicitly retain missing inference.
    """
    out = dict(sensitivity_mixed_estimate=np.nan, sensitivity_mixed_se=np.nan,
               sensitivity_mixed_p=np.nan, sensitivity_mixed_status='missing nucleus-level values',
               sensitivity_mixed_warnings='')
    data = data.copy()
    data['value'] = pd.to_numeric(data['value'], errors='coerce')
    data = data.loc[data.group.isin([test, reference]) & np.isfinite(data.value)].dropna(subset=['well_id','image'])
    for arm, label in [(test,'test'), (reference,'reference')]:
        sub = data.loc[data.group.eq(arm)]
        out.update({f'sensitivity_n_nuclei_{label}':len(sub),
                    f'sensitivity_n_fovs_{label}':len(sub[['well_id','image']].drop_duplicates()),
                    f'sensitivity_n_wells_{label}':sub.well_id.nunique(),
                    f'sensitivity_nucleus_median_{label}':float(sub.value.median()) if len(sub) else np.nan})
    if any(out[f'sensitivity_n_wells_{label}'] < 2 for label in ['test','reference']):
        out['sensitivity_mixed_status'] = 'blocked: fewer than two wells per arm with finite nucleus values'
        return out
    try:
        from statsmodels.regression.mixed_linear_model import MixedLM
    except ImportError:
        out['sensitivity_mixed_status'] = 'blocked: statsmodels missing; installation prohibited'
        return out
    data['arm'] = data.group.eq(test).astype(float)
    data['well'] = data.groupby(['group','well_id'], sort=True).ngroup().astype(str)
    # Formula variance components are constructed separately inside each well.
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model = MixedLM.from_formula('value ~ arm', data=data, groups='well',
                        re_formula='1', vc_formula={'fov':'0 + C(image)'}, use_sparse=True)
            fit = model.fit(reml=True, method=['lbfgs', 'bfgs'], maxiter=300, disp=False)
        out['sensitivity_mixed_warnings'] = '; '.join(dict.fromkeys(str(w.message) for w in caught))
        if not fit.converged:
            out['sensitivity_mixed_status'] = 'blocked: mixed model did not converge'
        elif not np.isfinite([fit.fe_params['arm'], fit.bse_fe['arm'], fit.pvalues['arm']]).all() or fit.bse_fe['arm'] <= 0:
            out['sensitivity_mixed_status'] = 'blocked: invalid mixed model covariance'
        else:
            out.update(sensitivity_mixed_estimate=float(fit.fe_params['arm']),
                       sensitivity_mixed_se=float(fit.bse_fe['arm']),
                       sensitivity_mixed_p=float(fit.pvalues['arm']), sensitivity_mixed_status='ok')
    except Exception as exc:
        out['sensitivity_mixed_status'] = f'blocked: {type(exc).__name__}: {exc}'
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
