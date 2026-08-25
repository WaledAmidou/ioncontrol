"""
Statistical validation.

The original manuscript reported a single one-way ANOVA F statistic.  A
significance test among the outputs of an assumed simulator establishes very
little on its own, so the analysis here (i) reports effect sizes and their
confidence intervals alongside every p value, (ii) uses the patient as the
unit of analysis with the Monte-Carlo replicates averaged within patient, so
that the degrees of freedom are not inflated by the number of random seeds,
(iii) uses a repeated-measures design, because every strategy is applied to
the *same* virtual patients, and (iv) applies a non-parametric confirmatory
test with multiplicity control.
"""
import numpy as np
import pandas as pd
from scipy import stats

from .config import N_SEEDS


def per_patient(x, n_seeds=N_SEEDS):
    """Average Monte-Carlo replicates within patient (correct unit of analysis)."""
    x = np.asarray(x, float)
    return x.reshape(-1, n_seeds).mean(axis=1)


def cohens_d_paired(x, y):
    d = x - y
    return float(d.mean() / d.std(ddof=1))


def hedges_g(x, y):
    nx, ny = len(x), len(y)
    sp = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    d = (x.mean() - y.mean()) / sp
    J = 1 - 3 / (4 * (nx + ny) - 9)
    return float(d * J)


def rm_anova(mat):
    """
    One-way repeated-measures ANOVA on a (patients x strategies) matrix.
    Returns F, p, partial eta squared and Greenhouse-Geisser epsilon.
    """
    n, k = mat.shape
    grand = mat.mean()
    ss_cond = n * ((mat.mean(axis=0) - grand) ** 2).sum()
    ss_subj = k * ((mat.mean(axis=1) - grand) ** 2).sum()
    ss_tot = ((mat - grand) ** 2).sum()
    ss_err = ss_tot - ss_cond - ss_subj
    df_c, df_e = k - 1, (n - 1) * (k - 1)
    F = (ss_cond / df_c) / (ss_err / df_e)
    p = stats.f.sf(F, df_c, df_e)
    eta2p = ss_cond / (ss_cond + ss_err)

    # Greenhouse-Geisser correction
    S = np.cov(mat, rowvar=False)
    Sb = S - S.mean(axis=0)[None, :] - S.mean(axis=1)[:, None] + S.mean()
    eps = (np.trace(Sb) ** 2) / ((k - 1) * np.sum(Sb ** 2))
    eps = float(np.clip(eps, 1.0 / (k - 1), 1.0))
    p_gg = stats.f.sf(F, df_c * eps, df_e * eps)
    return dict(F=float(F), df1=df_c, df2=df_e, p=float(p),
                eta2_partial=float(eta2p), gg_epsilon=eps, p_gg=float(p_gg))


def posthoc(mat, labels, ref="Proposed adaptive AI"):
    """Paired comparisons against the proposed strategy, Holm-corrected,
    with Tukey HSD reported alongside for continuity with the original text."""
    j = labels.index(ref)
    rows = []
    for i, lab in enumerate(labels):
        if i == j:
            continue
        t = stats.ttest_rel(mat[:, j], mat[:, i])
        w = stats.wilcoxon(mat[:, j], mat[:, i])
        d = mat[:, j] - mat[:, i]
        ci = np.percentile(
            [np.mean(np.random.default_rng(s).choice(d, d.size)) for s in range(2000)],
            [2.5, 97.5])
        rows.append(dict(comparison=f"{ref} vs {lab}",
                         mean_diff=float(d.mean()),
                         ci_lo=float(ci[0]), ci_hi=float(ci[1]),
                         t=float(t.statistic), p_t=float(t.pvalue),
                         W=float(w.statistic), p_wilcoxon=float(w.pvalue),
                         cohens_dz=cohens_d_paired(mat[:, j], mat[:, i])))
    df = pd.DataFrame(rows)
    for col, out in (("p_t", "p_t_holm"), ("p_wilcoxon", "p_wilcoxon_holm")):
        order = np.argsort(df[col].values)
        m = len(df)
        adj = np.empty(m)
        run = 0.0
        for r, idx in enumerate(order):
            val = (m - r) * df[col].values[idx]
            run = max(run, val)
            adj[idx] = min(run, 1.0)
        df[out] = adj

    tk = stats.tukey_hsd(*[mat[:, i] for i in range(mat.shape[1])])
    tukey = pd.DataFrame(
        [dict(a=labels[i], b=labels[j2], p=float(tk.pvalue[i, j2]))
         for i in range(len(labels)) for j2 in range(i + 1, len(labels))])
    return df, tukey


def analyse(res, metric="sse_z", ion=0, n_seeds=N_SEEDS):
    labels = list(res["results"].keys())
    cols = []
    for s in labels:
        v = res["results"][s][metric]
        v = v[:, ion] if v.ndim == 2 else v
        cols.append(per_patient(v, n_seeds))
    mat = np.column_stack(cols)
    an = rm_anova(mat)
    ph, tk = posthoc(mat, labels)
    return dict(matrix=mat, labels=labels, anova=an, posthoc=ph, tukey=tk)
