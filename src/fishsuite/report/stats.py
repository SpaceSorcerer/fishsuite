"""Replicate-level statistics for the ``fishsuite report`` layer.

Ported in behaviour from the one-off report builders this module replaces;
see ``docs/REPORT_SUBCOMMAND.md``.

The replicate unit is the WELL. Every gate runs on well means; the field of view
is a technical replicate and the nucleus is the measurement unit.
"""
from __future__ import annotations

import itertools
import math
from typing import Dict, List, Sequence

import numpy as np

ALPHA = 0.05
SEED = 0


def holm(pvals: Sequence[float]) -> List[float]:
    """Holm-Bonferroni step-down over one family.

    NaN entries stay NaN and do not consume a step, so the family size is the
    number of finite p-values.
    """
    idx = [i for i, p in enumerate(pvals) if p is not None and np.isfinite(p)]
    out: List[float] = [np.nan] * len(pvals)
    m = len(idx)
    if m == 0:
        return out
    order = sorted(idx, key=lambda i: pvals[i])
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * float(pvals[i]))
        running = max(running, adj)
        out[i] = running
    return out


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    """Bias-corrected standardised mean difference, mean(a) minus mean(b)."""
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float("nan")
    s1, s2 = a.std(ddof=1), b.std(ddof=1)
    sp2 = ((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2)
    if not np.isfinite(sp2) or sp2 <= 0:
        return float("nan")
    d = (a.mean() - b.mean()) / math.sqrt(sp2)
    j = 1.0 - 3.0 / (4.0 * (n1 + n2) - 9.0)
    return float(j * d)


def welch(a: np.ndarray, b: np.ndarray, alpha: float = ALPHA) -> dict:
    """Welch t on group a against group b. ``diff`` is mean(a) minus mean(b)."""
    from scipy import stats

    n1, n2 = len(a), len(b)
    out = dict(n_test=n1, n_ref=n2,
               mean_test=float(a.mean()) if n1 else np.nan,
               mean_ref=float(b.mean()) if n2 else np.nan,
               diff=np.nan, ci_low=np.nan, ci_high=np.nan, hedges_g=np.nan,
               t=np.nan, df=np.nan, p_welch=np.nan, welch_note="",
               sd_wells_test=float(a.std(ddof=1)) if n1 > 1 else np.nan,
               sd_wells_reference=float(b.std(ddof=1)) if n2 > 1 else np.nan)
    if n1 < 2 or n2 < 2:
        out["welch_note"] = "fewer than 2 wells with a finite value in one group"
        return out
    v1, v2 = a.var(ddof=1), b.var(ddof=1)
    se = math.sqrt(v1 / n1 + v2 / n2)
    out["diff"] = float(a.mean() - b.mean())
    out["hedges_g"] = hedges_g(a, b)
    if se == 0:
        out["welch_note"] = "zero within-group variance in both groups; Welch t undefined"
        return out
    df = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    tcrit = stats.t.ppf(1 - alpha / 2, df)
    t, p = stats.ttest_ind(a, b, equal_var=False)
    out.update(ci_low=out["diff"] - tcrit * se, ci_high=out["diff"] + tcrit * se,
               t=float(t), df=float(df), p_welch=float(p))
    return out


def exact_permutation(a: np.ndarray, b: np.ndarray) -> dict:
    """Exact permutation of well labels; statistic is the absolute mean difference.

    With 3 versus 3 wells there are 20 assignments and each is matched by its
    complement, so the smallest attainable two-sided p is 0.10. With unequal
    groups an assignment has no same-size complement, so the floor is
    1/n_assignments. That floor is arithmetic, not a property of the data, and is
    reported alongside the p.
    """
    n1, n2 = len(a), len(b)
    out = dict(perm_obs_abs_diff=np.nan, perm_n_assignments=0,
               p_permutation=np.nan, perm_arithmetic_floor=np.nan, perm_note="")
    if n1 < 1 or n2 < 1:
        out["perm_note"] = "empty group"
        return out
    pooled = np.concatenate([a, b])
    obs = abs(a.mean() - b.mean())
    out["perm_obs_abs_diff"] = float(obs)
    hits = total = 0
    all_idx = set(range(len(pooled)))
    for sel in itertools.combinations(range(len(pooled)), n1):
        s = np.array(sel)
        r = np.array(sorted(all_idx - set(sel)))
        if abs(pooled[s].mean() - pooled[r].mean()) >= obs - 1e-12:
            hits += 1
        total += 1
    out.update(perm_n_assignments=total, p_permutation=hits / total,
               perm_arithmetic_floor=(2.0 if n1 == n2 else 1.0) / total,
               perm_note="exact; two-sided floor is %d/n_assignments by arithmetic"
                         % (2 if n1 == n2 else 1))
    return out


def tukey_two_group(groups: Dict[str, np.ndarray], test: str, reference: str,
                    alpha: float = ALPHA) -> dict:
    """Tukey HSD on per-field means, normalised to (test minus reference).

    ``groups`` must hold EVERY group in the design, not just the two being
    reported. Tukey's adjustment is a studentized range over k groups, so fitting
    it on a two-group subset of a five-group design would understate the p-value.
    The pair of interest is extracted after the full fit; ``tukey_k`` records how
    many groups the adjustment was made over.

    Differences and intervals come from the NUMERIC attributes of statsmodels
    ``pairwise_tukeyhsd``; its ``summary()`` rounds to 4 decimals. The adjusted p
    is recomputed from the studentized range so the statistic is visible.
    """
    out = dict(tukey_difference=np.nan, tukey_ci95_low=np.nan, tukey_ci95_high=np.nan,
               p_tukey_fov=np.nan, tukey_q=np.nan, tukey_k=np.nan, tukey_df=np.nan,
               tukey_pooled_sd=np.nan, tukey_note="")
    usable = {k: v for k, v in groups.items() if len(v) >= 1}
    if len(usable) < 2 or sum(len(v) for v in usable.values()) - len(usable) < 1:
        out["tukey_note"] = "too few fields or residual df for Tukey"
        return out
    if test not in usable or reference not in usable:
        out["tukey_note"] = (f"{test!r} or {reference!r} has no field value, so the "
                             "pair cannot be read out of the Tukey fit")
        return out
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    names = sorted(usable)
    endog = np.concatenate([usable[n] for n in names])
    labels = np.concatenate([[n] * len(usable[n]) for n in names])
    if not np.all(np.isfinite(endog)):
        out["tukey_note"] = "non-finite values reached Tukey"
        return out
    try:
        res = pairwise_tukeyhsd(endog, labels, alpha=alpha)
    except Exception as exc:                                   # noqa: BLE001
        out["tukey_note"] = f"pairwise_tukeyhsd failed: {type(exc).__name__}: {exc}"
        return out
    order = list(res.groupsunique)
    i_idx, j_idx = np.triu_indices(len(order), 1)
    diffs = np.asarray(res.meandiffs, dtype=float)
    ci = np.asarray(res.confint, dtype=float)
    pv_sm = np.asarray(res.pvalues, dtype=float)

    vals = [usable[n] for n in order]
    k = len(vals)
    df_w = sum(len(g) for g in vals) - k
    ss_w = sum(((g - g.mean()) ** 2).sum() for g in vals)
    sp = math.sqrt(ss_w / df_w) if df_w > 0 else np.nan

    wanted = {reference, test}
    m = next((idx for idx in range(len(diffs))
              if {str(order[i_idx[idx]]), str(order[j_idx[idx]])} == wanted), None)
    if m is None:
        out["tukey_note"] = (f"the pair ({reference}, {test}) is not among the "
                             f"{len(diffs)} pairs statsmodels returned")
        return out
    g1, g2 = str(order[i_idx[m]]), str(order[j_idx[m]])
    diff = float(diffs[m])
    lo, hi = float(ci[m, 0]), float(ci[m, 1])
    # statsmodels reports mean(g2) - mean(g1); normalise to test - reference.
    if (g1, g2) == (test, reference):
        diff, lo, hi = -diff, -hi, -lo
    n1, n2 = len(usable[g1]), len(usable[g2])
    se = sp * math.sqrt((1.0 / n1 + 1.0 / n2) / 2.0) if np.isfinite(sp) else np.nan
    q = abs(diffs[m]) / se if np.isfinite(se) and se > 0 else np.nan
    try:
        from scipy.stats import studentized_range

        padj = (float(studentized_range.sf(q, k, df_w))
                if np.isfinite(q) and df_w > 0 else float(pv_sm[m]))
        if np.isfinite(q) and df_w > 0 and padj == 0.0:
            out["tukey_note"] = (
                out["tukey_note"] + "; the exact studentized-range survival function "
                f"UNDERFLOWED to 0 at q={q:.4g}; the true p is smaller than about "
                "1e-16, not zero").strip("; ")
    except Exception:                                          # noqa: BLE001
        padj = float(pv_sm[m])
        out["tukey_note"] = (out["tukey_note"] + "; adjusted p fell back to the "
                             "statsmodels value").strip("; ")
    out.update(tukey_difference=diff, tukey_ci95_low=lo, tukey_ci95_high=hi,
               p_tukey_fov=padj, tukey_q=q, tukey_k=k, tukey_df=df_w,
               tukey_pooled_sd=sp)
    return out


MDE_SIM_REPS = 20_000
MDE_SIM_SEED = 20261010


def welch_power_simulated(diff: float, alpha: float, n1: int, n2: int, sd1: float, sd2: float,
                          reps: int = MDE_SIM_REPS, seed: int = MDE_SIM_SEED) -> float:
    """Monte Carlo power of the two-sided Welch test for a true difference ``diff`` (raw units):
    ``reps`` normal samples of n1 and n2 with SDs sd1, sd2, the same draws (seeded) at every ``diff``."""
    return float(np.mean(_welch_rejects(diff, *_welch_draws(alpha, n1, n2, sd1, sd2, reps, seed))))


def _welch_draws(alpha, n1, n2, sd1, sd2, reps, seed):
    from scipy import stats

    rng = np.random.default_rng(seed)
    a = rng.standard_normal((reps, n1)) * float(sd1)
    b = rng.standard_normal((reps, n2)) * float(sd2)
    v1, v2 = a.var(axis=1, ddof=1) / n1, b.var(axis=1, ddof=1) / n2
    se = np.sqrt(v1 + v2)
    with np.errstate(divide="ignore", invalid="ignore"):
        df = (v1 + v2) ** 2 / (v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1))
    return a.mean(axis=1) - b.mean(axis=1), se, stats.t.ppf(1 - alpha / 2, df)


def _welch_rejects(diff, noise, se, crit):
    # The shift moves only the mean difference; each draw's SE and Welch df do not depend on it.
    return np.abs(diff + noise) > crit * se


def mde_hedges_g(alpha: float, power: float = 0.80, n1: int = 3, n2: int = 3,
                 sd1: float | None = None, sd2: float | None = None,
                 reps: int = MDE_SIM_REPS, seed: int = MDE_SIM_SEED) -> float:
    """Smallest standardized difference at which the two-sided Welch test reaches ``power`` at ``alpha``.

    Simulation-based (2026-10-10): the Welch test is run by Monte Carlo (``reps`` draws, fixed ``seed``,
    the same draws at every candidate difference) on normal arms with the per-arm SDs ``sd1``, ``sd2``
    (the observed well SDs; equal SDs when both are None), and the smallest |difference| with simulated
    power >= ``power`` is found by bisection. This makes the MDE depend on the observed SDs (post-hoc).
    Returned in pooled-SD units, the scale ``hedges_g`` estimates; no Hedges j (j corrects an estimator
    and understated the MDE by 0.571x/0.80x/0.870x at 2/3/4 per arm). It replaces the noncentral-t
    approximation, which overstated the realised Welch power at 2-3 wells per arm.
    NaN below 2 per arm, for an invalid alpha, or for non-finite / all-zero SDs.
    """
    if min(int(n1), int(n2)) < 2:
        return float("nan")
    if not np.isfinite(alpha) or alpha <= 0 or alpha >= 1:
        return float("nan")
    if sd1 is None and sd2 is None:
        sd1 = sd2 = 1.0
    sd1, sd2 = float(sd1), float(sd2)
    sp2 = ((n1 - 1) * sd1 ** 2 + (n2 - 1) * sd2 ** 2) / (n1 + n2 - 2)
    if not (np.isfinite(sd1) and np.isfinite(sd2)) or sd1 < 0 or sd2 < 0 or sp2 <= 0:
        return float("nan")
    draws = _welch_draws(alpha, int(n1), int(n2), sd1, sd2, int(reps), seed)

    def pw(diff):
        return float(np.mean(_welch_rejects(diff, *draws)))

    sp = math.sqrt(sp2)
    lo, hi = 0.0, sp
    while pw(hi) < power:
        hi *= 2
        if hi > 1e6 * sp:
            return float("nan")
    for _ in range(60):
        mid = (lo + hi) / 2
        if pw(mid) < power:
            lo = mid
        else:
            hi = mid
    return float(hi / sp)


def stars(p) -> str:
    """Star glyph for a p-value, per the locked SuperPlot convention."""
    if p is None or not np.isfinite(p):
        return "n/a"
    p = float(p)
    if p < 1e-4:
        return "****"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def fmt_p(p) -> str:
    """Decimal until the leading zeros pile up, then scientific."""
    if p is None or not np.isfinite(p):
        return "NA"
    p = float(p)
    return f"{p:.4g}" if p >= 1e-4 else f"{p:.2e}"
