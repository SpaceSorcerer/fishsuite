"""Report-only RNASEH2B/BIN1 analyses from explicitly named persisted tables.

No raw discovery, segmentation, detection, or null generation occurs here.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .miat_qki import ratio_interval, unblocked_exact_ratio

COLORS = {'WT': '#595959', 'KO': '#CC79A7'}
TITLE_MAP = {
    'FIG_RNASEH2B_PREFERENTIAL_COLOC_TOTAL': 'Total nuclear RNASEH2B',
    'FIG_RNASEH2B_PREFERENTIAL_COLOC_AT_FOOTPRINT': 'RNASEH2B at BIN1 introns',
    'FIG_RNASEH2B_PREFERENTIAL_COLOC_R': 'Preferential retention ratio',
    'FIG_RNASEH2B_LEVEL_ENRICHMENT_WT': 'WT: RNASEH2B vs enrichment',
    'FIG_RNASEH2B_LEVEL_ENRICHMENT_KO': 'KO: RNASEH2B vs enrichment',
    'FIG_RNASEH2B_LEVEL_ENRICHMENT_OVERLAY': 'RNASEH2B vs enrichment',
    'FIG_RNASEH2B_RADIAL_PROFILE': 'RNASEH2B around BIN1 introns',
    'FIG_BIN1_RNASEH2B_NEAREST_NEIGHBOUR': 'BIN1 to nearest RNASEH2B',
    'FIG_SECONLY_BIN1_SPOTS': 'BIN1 secondary-only check',
    'FIG_SECONLY_RNASEH2B_SPOTS': 'RNASEH2B secondary-only check',
    'FIG_SECONLY_RNASEH2B_INTENSITY': 'Secondary RNASEH2B intensity',
}


def _bool(values):
    return values.astype(str).str.lower().isin(['true', '1', '1.0'])


def _arm(values):
    return values.astype(str).str.extract(r'(?i)(WT|KO)', expand=False).str.upper()


def _figure():
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    fig.subplots_adjust(left=.19, right=.96, bottom=.23, top=.84)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=9)
    return fig, ax


def _export(fig, ax, key, out, run, caption, *, nonnegative=False):
    ax.set_title(TITLE_MAP[key], fontsize=11, wrap=False)
    ax.margins(y=.25)
    fig.text(.19, .035, run.name, fontsize=5.5)
    focus = ax.get_ylim()
    paths = {}
    for variant in ('focus', 'full'):
        if variant == 'full' and nonnegative:
            ax.set_ylim(0, max(focus[1], .01))
        else:
            ax.set_ylim(focus)
        from .figures import FIGURE_AXES
        FIGURE_AXES[str((out/'figures'/f'{key}_{variant}.png').resolve())]=dict(variant=variant,ylim=list(ax.get_ylim()),xlim=list(ax.get_xlim()),axis_group='rnaseh2b_level_enrichment' if 'LEVEL_ENRICHMENT' in key else '')
        for ext in ('png', 'svg'):
            path = out/'figures'/f'{key}_{variant}.{ext}'
            fig.savefig(path, dpi=180)
            paths[f'{variant}_{ext}'] = str(path)
    plt.close(fig)
    return dict(key=key, title=TITLE_MAP[key], caption=caption, **paths)


def build_analysis(run_dir, output_dir, correction_csv, acquisition_method=None):
    """Return workbook tables, independent figure records, slides and results.

    ``correction_csv`` supplies the frozen matched-secondary arm baselines only;
    the retained nuclei and all footprint measurements come from ``run_dir``.
    """
    run, out = Path(run_dir).resolve(), Path(output_dir).resolve()
    (out/'figures').mkdir(parents=True, exist_ok=True)
    (out/'data').mkdir(parents=True, exist_ok=True)
    nuclei = pd.read_csv(run/'nuclei_metrics.csv')
    spots = pd.read_csv(run/'spot_metrics.csv')
    correction = pd.read_csv(correction_csv)
    correction['arm_A17'] = _arm(correction.arm)
    baseline = correction.groupby('arm_A17').secondary_arm_mean.agg(['min', 'max'])
    if set(baseline.index) != {'WT', 'KO'} or not np.allclose(baseline['min'], baseline['max']):
        raise ValueError('missing or inconsistent arm-matched secondary baseline')
    baseline = baseline['min']
    nuclei['arm'] = _arm(nuclei.condition)
    bio = nuclei.loc[~_bool(nuclei.secondary_only)].copy()
    if bio.arm.isna().any():
        raise ValueError('unrecognized biological arm')
    bio['T'] = bio.protein_nuclear_mean - bio.arm.map(baseline)
    anchors = spots.loc[spots.channel.eq('rna1') & _bool(spots.in_nucleus)
                        & ~_bool(spots.secondary_only) & spots.nucleus_id.gt(0)].copy()
    # Each footprint supplies its own mean; equal punctum weight within nucleus.
    footprint = anchors.groupby(['image', 'nucleus_id']).agg(
        A=('qki_at_miat_footprint', lambda x: x.mean(skipna=False)),
        n_footprints=('qki_at_miat_footprint', 'size')).reset_index()
    bio = bio.merge(footprint, on=['image', 'nucleus_id'], how='left', validate='one_to_one')
    fovs = bio.groupby(['condition', 'arm', 'image'])[['A', 'T']].mean().reset_index()
    wells = fovs.groupby(['condition', 'arm'])[['A', 'T']].mean().reset_index()
    pairs = wells.rename(columns={'condition': 'biological_set'}).copy()
    pairs['slide'] = 'run_retained'
    pairs['arm'] = pairs.arm.map({'WT': 'NT', 'KO': 'KD'})
    result = ratio_interval(pairs, 'run_retained', expected_wells=3)
    exact, registry = unblocked_exact_ratio(pairs, expected_wells=3)
    result.update(exact)
    result.update(A_definition='Mean stored RNASEH2B signal per exact nuclear BIN1 footprint, then nucleus/FOV/well means; no secondary subtraction on A.',
                  T_definition='Matched-secondary-corrected RNASEH2B nuclear mean; all retained nuclei, equal FOV weight per well.',
                  n_nuclei=len(bio), n_nuclei_A_defined=int(bio.A.notna().sum()))
    if result['reason'] or result.get('exact_reason'):
        raise ValueError(f'Ratio unavailable: {result}')
    # Explicit defined-pair filters, and persisted rotation usability only.
    rotation = 'protein_rotation_enrichment_at_rna1_spots'
    scatter = bio.loc[_bool(bio.rotation_null_usable)
                      & np.isfinite(bio.protein_nuclear_mean)
                      & np.isfinite(bio[rotation])].copy()
    spearman = []
    for arm in ('WT', 'KO'):
        group = scatter.loc[scatter.arm.eq(arm)]
        rho, p = stats.spearmanr(group.protein_nuclear_mean, group[rotation])
        spearman.append(dict(group=arm, n=len(group), rho=rho, p_descriptive=p))
    spearman = pd.DataFrame(spearman)
    radial = pd.read_csv(run/'coloc_radial_profile.csv')
    radial['arm'] = _arm(radial.condition)
    radial = radial.loc[radial.arm.notna()].copy()
    radial_wells = radial.groupby(['condition', 'arm', 'ring_um']).enrichment.mean().reset_index()
    radial_summary = radial_wells.groupby(['arm', 'ring_um']).enrichment.agg(['mean', 'std', 'count']).reset_index()
    radial_summary['ci_half'] = radial_summary['std']/np.sqrt(radial_summary['count'])*stats.t.ppf(.975, radial_summary['count']-1)
    radial_summary['ci_low'] = radial_summary['mean']-radial_summary.ci_half
    radial_summary['ci_high'] = radial_summary['mean']+radial_summary.ci_half
    nearest = anchors[['image', 'condition', 'nucleus_id', 'spot_id', 'nn_distance_um']].copy()
    nearest['arm'] = _arm(nearest.condition)
    nearest = nearest.loc[np.isfinite(nearest.nn_distance_um)]
    sec = nuclei.loc[_bool(nuclei.secondary_only)].copy()
    sec['arm'] = _arm(sec.image)
    sec_fields = sec.groupby(['image', 'arm']).agg(n_nuclei=('nucleus_id', 'size'),
        BIN1_spots=('n_spots_rna1', 'mean'), RNASEH2B_spots=('n_spots_protein', 'mean'),
        RNASEH2B_intensity=('protein_nuclear_mean', 'mean')).reset_index()
    for channel in ('BIN1', 'RNASEH2B'):
        med = sec_fields[f'{channel}_spots'].median()
        sec_fields[f'{channel}_median'] = med
        sec_fields[f'{channel}_3x_cut'] = 3*med
        sec_fields[f'{channel}_3x_exceeds'] = sec_fields[f'{channel}_spots'] > 3*med
    sec_means = sec_fields.groupby('arm').RNASEH2B_intensity.mean().reset_index()
    sec_means['frozen_baseline'] = sec_means.arm.map(baseline)
    np.testing.assert_allclose(sec_means.RNASEH2B_intensity, sec_means.frozen_baseline, rtol=1e-10, atol=1e-8)
    bio_fields = bio.groupby(['image', 'arm']).agg(BIN1_spots=('n_spots_rna1', 'mean'),
        RNASEH2B_spots=('n_spots_protein', 'mean')).reset_index()
    missing = ['Shuffle nearest-neighbour distance distribution missing: spot_metrics.csv contains observed distances; coloc_rotation_null.csv contains intensity-null values, not distance draws. No null rerun performed.']
    if acquisition_method is None or not Path(acquisition_method).is_file():
        missing.append('ACQUISITION_AND_SECONLY_CHECK method document missing: no verified document path supplied. Matched-secondary intensity method is reproduced from the frozen correction CSV and validated against current stored nucleus/FOV values.')
    tables = {'A17 Ratio': pd.DataFrame([result]), 'A17 Ratio wells': pairs,
              'A17 Ratio FOVs': fovs, 'A17 Ratio nuclei': bio,
              'A17 Ratio permutations': registry, 'A17 Spearman': spearman,
              'A17 Scatter pairs': scatter[['image', 'condition', 'arm', 'nucleus_id', 'protein_nuclear_mean', rotation]],
              'A17 Radial source': radial, 'A17 Radial wells': radial_wells,
              'A17 Radial summary': radial_summary, 'A17 Nearest observed': nearest,
              'A17 Secondary fields': sec_fields, 'A17 Secondary intensity': sec_means,
              'A17 Biological spot FOVs': bio_fields,
              'A17 Missing': pd.DataFrame({'missing': missing})}
    figures = []
    def save(fig, ax, key, caption, **kw):
        figures.append(_export(fig, ax, key, out, run, caption, **kw))
    for column, key, ylabel in [('T', 'TOTAL', 'Secondary-corrected nuclear mean (AU)'),
                                ('A', 'AT_FOOTPRINT', 'Mean at exact BIN1 footprints (AU)')]:
        fig, ax = _figure()
        for index, arm in enumerate(('WT', 'KO')):
            v = wells.loc[wells.arm.eq(arm), column].to_numpy()
            ax.scatter(index+np.linspace(-.07, .07, len(v)), v, c=COLORS[arm], s=35)
            ax.plot([index-.18, index+.18], [v.mean()]*2, color='black', lw=1.5)
        ax.set_xticks([0, 1], ['WT', 'KO']); ax.set_xlim(-.5, 1.5); ax.set_ylabel(ylabel, fontsize=9)
        save(fig, ax, 'FIG_RNASEH2B_PREFERENTIAL_COLOC_'+key, 'One point per well; equal FOV weight; three wells per arm.', nonnegative=True)
    fig, ax = _figure()
    ax.errorbar([0], [result['R']], yerr=[[result['R']-result['ci_low']], [result['ci_high']-result['R']]], fmt='o', color=COLORS['KO'], capsize=5)
    ax.axhline(1, color='gray', linestyle='--', lw=1)
    ax.set_xticks([0], ['R = rA / rT']); ax.set_xlim(-.5, .5); ax.set_ylabel('Ratio with 95% CI', fontsize=9)
    save(fig, ax, 'FIG_RNASEH2B_PREFERENTIAL_COLOC_R', f"Normal log-delta CI; exact two-sided p={result['p_exact']:.3g}; 20 well-label assignments.", nonnegative=True)
    for arms, suffix in [(('WT',), 'WT'), (('KO',), 'KO'), (('WT', 'KO'), 'OVERLAY')]:
        fig, ax = _figure()
        for arm in arms:
            g = scatter.loc[scatter.arm.eq(arm)]
            rho = spearman.loc[spearman.group.eq(arm)].iloc[0]
            ax.scatter(g.protein_nuclear_mean, g[rotation], s=9, alpha=.45, color=COLORS[arm], label=f'{arm}: rho={rho.rho:.3f}, n={rho.n}')
        ax.set_xlabel('Nuclear RNASEH2B mean (AU)', fontsize=9); ax.set_ylabel('Rotation-null enrichment', fontsize=9)
        # Both per-arm panels and their overlay use the union data window.
        for column,setter in [('protein_nuclear_mean',ax.set_xlim),(rotation,ax.set_ylim)]:
            lo,hi=scatter[column].min(),scatter[column].max()
            pad=max((hi-lo)*.05,abs(hi)*.01,1e-6)
            setter(max(0,lo-pad),hi+pad)
        ax.legend(fontsize=7, frameon=False)
        save(fig, ax, 'FIG_RNASEH2B_LEVEL_ENRICHMENT_'+suffix, 'Descriptive per-nucleus Spearman; usable persisted rotation nulls.', nonnegative=True)
    fig, ax = _figure()
    for arm in ('WT', 'KO'):
        g = radial_summary.loc[radial_summary.arm.eq(arm)].sort_values('ring_um')
        ax.plot(g.ring_um, g['mean'], marker='o', color=COLORS[arm], label=arm)
        ax.fill_between(g.ring_um.to_numpy(), g.ci_low.to_numpy(), g.ci_high.to_numpy(), color=COLORS[arm], alpha=.18)
    ax.axhline(1, color='gray', linestyle='--', lw=1); ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel('Radial bin (um)', fontsize=9); ax.set_ylabel('Enrichment vs position null', fontsize=9)
    save(fig, ax, 'FIG_RNASEH2B_RADIAL_PROFILE', 'Persisted radial data; 95% t CI across three well means per arm.', nonnegative=True)
    fig, ax = _figure()
    for arm in ('WT', 'KO'):
        x = np.sort(nearest.loc[nearest.arm.eq(arm), 'nn_distance_um'].to_numpy())
        ax.step(x, np.arange(1, len(x)+1)/len(x), where='post', color=COLORS[arm], label=f'{arm}, n={len(x)}')
    ax.legend(fontsize=8, frameon=False); ax.set_xlabel('Nearest RNASEH2B distance (um)', fontsize=9); ax.set_ylabel('Cumulative fraction of BIN1 puncta', fontsize=9)
    save(fig, ax, 'FIG_BIN1_RNASEH2B_NEAREST_NEIGHBOUR', 'Observed nuclear BIN1 puncta only. Shuffle-distance draws missing.', nonnegative=True)
    for channel in ('BIN1', 'RNASEH2B'):
        fig, ax = _figure()
        for index, (arm, control) in enumerate([('WT', False), ('KO', False), ('WT', True), ('KO', True)]):
            frame = sec_fields if control else bio_fields
            values = frame.loc[frame.arm.eq(arm), channel+'_spots'].to_numpy()
            ax.scatter(index+np.linspace(-.09, .09, len(values)), values, color=COLORS[arm], marker='x' if control else 'o', s=25)
        cut = float(sec_fields[channel+'_3x_cut'].iloc[0]); nflag = int(sec_fields[channel+'_3x_exceeds'].sum())
        ax.axhline(cut, color='gray', ls='--', label=f'3x control median = {cut:.2g}')
        ax.set_xticks(range(4), ['WT', 'KO', 'WT sec', 'KO sec']); ax.set_ylabel('Spots per nucleus (FOV means)', fontsize=9); ax.legend(fontsize=7, frameon=False)
        save(fig, ax, 'FIG_SECONLY_'+channel+'_SPOTS', f'3x-median audit: {nflag}/{len(sec_fields)} control FOVs exceed cut; prior exclusions unchanged.', nonnegative=True)
    fig, ax = _figure()
    for index, arm in enumerate(('WT', 'KO')):
        values = sec_fields.loc[sec_fields.arm.eq(arm), 'RNASEH2B_intensity'].to_numpy()
        ax.scatter(index+np.linspace(-.07, .07, len(values)), values, color=COLORS[arm], s=30)
        ax.plot([index-.18, index+.18], [values.mean()]*2, color='black')
    ax.set_xticks([0, 1], ['WT sec', 'KO sec']); ax.set_xlim(-.5, 1.5); ax.set_ylabel('Nuclear RNASEH2B mean (AU)', fontsize=9)
    save(fig, ax, 'FIG_SECONLY_RNASEH2B_INTENSITY', 'Equal FOV weighting; baseline matches frozen correction CSV.', nonnegative=True)
    detected = result['R'] > 1 and result['p_exact'] < .05
    ratio_line = (f"Tested prediction R > 1: preferential retention {'detected' if detected else 'not detected'}; "
                  f"R={result['R']:.3f}, 95% CI {result['ci_low']:.3f}-{result['ci_high']:.3f}, "
                  f"exact two-sided p={result['p_exact']:.3g}; minimum detectable retention R={result['R_MDE_05']:.3f} (normal model, 80% power).")
    slides = [dict(title='Is the remaining RNASEH2B preferentially at BIN1 introns?',
                   figures=[f['key'] for f in figures[:3]], body=ratio_line),
              dict(title='Does enrichment depend on RNASEH2B level?', figures=[f['key'] for f in figures[3:6]], body='; '.join(f"{r.group}: rho={r.rho:.3f}, n={r.n}" for r in spearman.itertuples())),
              dict(title='RNASEH2B around nuclear BIN1 puncta', figures=[figures[6]['key']], body=figures[6]['caption']),
              dict(title='BIN1 to nearest RNASEH2B punctum', figures=[figures[7]['key']], body=figures[7]['caption']),
              dict(title='Secondary-only controls', figures=[f['key'] for f in figures[8:]], body='; '.join(f['caption'] for f in figures[8:]))]
    for name, table in tables.items():
        table.to_csv(out/'data'/(name.replace(' ', '_')+'.csv'), index=False)
    summary = dict(ratio=result, spearman=spearman.to_dict('records'), ratio_line=ratio_line)
    (out/'A17_RESULTS.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (out/'A17_ANALYSIS_FIGURES.json').write_text(json.dumps(figures, indent=2), encoding='utf-8')
    (out/'A17_ANALYSIS_SLIDES.json').write_text(json.dumps(slides, indent=2), encoding='utf-8')
    (out/'A17_ANALYSIS_MISSING.md').write_text('\n'.join(missing)+'\n', encoding='utf-8')
    (out/'A17_ANALYSIS_TITLE_MAP.json').write_text(json.dumps(TITLE_MAP, indent=2), encoding='utf-8')
    (out/'A17_ANALYSIS_METHODS.md').write_text(
        'Human imaging; no genome reference used. T subtracts the matching-arm arithmetic mean of all secondary-control FOV means from each retained primary nucleus, without clipping. Four WT and three KO secondary FOVs contribute; the baseline matches the supplied frozen correction CSV. Baseline uncertainty is not propagated. Same acquisition settings (acquirer); staining batch not controlled.\n'
        'A averages the stored RNASEH2B mean inside each exact half-maximum nuclear BIN1 punctum footprint, equally across puncta within a nucleus; A is not secondary-subtracted. Nuclei without a defined nuclear footprint have missing A, never zero. T uses all retained nuclei. Both endpoints average nucleus to FOV to well, with equal FOV weights. Therefore A and T share wells but have different defined nucleus denominators, explicitly exported.\n'
        'R=(mean(A_KO)/mean(A_WT))/(mean(T_KO)/mean(T_WT)); three wells per arm. The log-delta normal 95% CI includes within-arm A,T covariance across wells. The two-sided exact test enumerates every balanced assignment of the six whole-well paired A,T values (20 assignments), using abs(log(R)); one-sided R>1 p is separately reported. Numeric well suffixes do not establish paired randomization, so permutations are unblocked. With 20 assignments the smallest two-sided p is 0.10. The normal CI is not an inversion of this exact test; normal-model 80% R-MDE at two-sided alpha .05 is not attainable power of the discrete exact test.\n'
        'Spearman uses finite nucleus-level raw nuclear intensity and rotation-enrichment pairs with a usable persisted rotation null. Nucleus-level p values are descriptive and do not substitute nuclei for biological replicates. Radial curves use the persisted per-FOV radial enrichment values and equal FOV-to-well means; bands are t-based 95% CI across three wells. The source colloquial radial filename _ci is a PNG rather than a numeric CI table; numeric bands are derived transparently from the radial CSV, without null generation.\n'
        'Nearest-neighbour ECDF uses finite stored distances for nuclear BIN1 puncta; shuffle distance draws are missing. Secondary 3x-median audit compares each control FOV spots-per-nucleus mean to three times the median of all seven control FOVs, separately by channel; it does not change existing exclusions or fitted inference. Acquisition method document remains missing unless explicitly supplied.\n', encoding='utf-8')
    return dict(tables=tables, figures=figures, slides=slides, summary=summary, missing=missing)
