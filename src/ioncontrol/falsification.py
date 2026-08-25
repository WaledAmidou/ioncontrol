"""
The falsification analysis.

This module contains the three checks that successively removed the apparent
physiological signal reported in earlier drafts of the manuscript. They are
kept together, in order, because the sequence is the finding: each stage is a
standard, uncontroversial piece of epidemiological hygiene, and each one
eliminated an association that had looked convincing at the previous stage.

  Stage 1  ENDPOINT DECOMPOSITION.  HOMA-B alone confounds beta-cell
           secretory capacity with compensatory hyperinsulinaemia. Fitting
           four endpoints separates them.
  Stage 2  PROXY VALIDATION.  Cycle L measures both dietary magnesium and
           serum magnesium in the same people, so the surrogacy assumption
           that cycle I rests on can be tested rather than asserted.
  Stage 3  ADIPOSITY ADJUSTMENT.  BMI is the dominant confounder of HOMA-IR
           and is associated with both magnesium measures.

Run `scripts/04_falsification.py` to reproduce the whole sequence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .config import COVARS, ENDPOINTS, ADIPOSITY, SEED


# ----------------------------------------------------------------------
def ols(y, X, names):
    """Least squares with analytic standard errors and the parameter covariance."""
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ c
    dof = len(y) - X.shape[1]
    cov = (r @ r / dof) * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t = c / se
    tab = pd.DataFrame(dict(term=names, coef=c, se=se,
                            p=2 * stats.t.sf(np.abs(t), dof),
                            ci_lo=c - 1.96 * se, ci_hi=c + 1.96 * se))
    return tab, cov, float(1 - r.var() / y.var())


def z(s):
    s = np.asarray(s, float)
    return (s - s.mean()) / s.std(ddof=1)


def _endpoint_vector(d, endpoint):
    if endpoint == "DI":
        return np.log((d["HOMA_B"] / d["HOMA_IR"]).values)
    if endpoint == "HOMA_B_adj":
        return np.log(d["HOMA_B"].values)
    return np.log(d[endpoint].values)


# ----------------------------------------------------------------------
def fit_endpoints(d, ions, adjust_adiposity=False, extra=()):
    """
    Fit every endpoint in `config.ENDPOINTS` on the standardised ion
    concentrations, adjusted for age, sex and the income-to-poverty ratio,
    optionally for adiposity, and optionally for additional columns.

    Returns one long table (one row per ion per endpoint) and the parameter
    covariance of the ionic block per endpoint, which the closed-loop code
    needs to build correct confidence intervals.
    """
    need = list(ions) + list(extra) + COVARS + ["HOMA_B", "HOMA_IR"]
    if adjust_adiposity:
        need.append(ADIPOSITY)
    d = d[d["dm_dx"] == 0].dropna(subset=need)
    d = d[(d["HOMA_B"] > 0) & (d["HOMA_IR"] > 0)]

    Z = [z(d[i]) for i in ions] + [z(d[e]) for e in extra]
    names_ion = list(ions) + list(extra)
    cols = [np.ones(len(d))] + Z
    names = ["intercept"] + names_ion
    if adjust_adiposity:
        cols.append(z(d[ADIPOSITY]))
        names.append(ADIPOSITY)
    cols += [d[c].values for c in COVARS]
    names += COVARS
    X = np.column_stack(cols)

    rows, covs = [], {}
    k = len(names_ion)
    for ep in ENDPOINTS:
        Xi, ni = X, names
        if ep == "HOMA_B_adj":
            Xi = np.column_stack([X, np.log(d["HOMA_IR"].values)])
            ni = names + ["log_HOMA_IR"]
        tab, cov, r2 = ols(_endpoint_vector(d, ep), Xi, ni)
        covs[ep] = cov[1:1 + k, 1:1 + k]
        t = tab[tab.term.isin(names_ion)].copy()
        t["endpoint"], t["r2"], t["n"] = ep, r2, len(d)
        t["adiposity_adjusted"] = adjust_adiposity
        rows.append(t)
    return pd.concat(rows, ignore_index=True), covs, len(d)


# ----------------------------------------------------------------------
def stage1_endpoint_decomposition(a_I, ions=("Zn", "Ca", "Mg")):
    """Cycle I, unadjusted for adiposity: the state of the analysis before
    any of the checks below were run."""
    tab, covs, n = fit_endpoints(a_I, ions, adjust_adiposity=False)
    tab["stage"] = "1: endpoint decomposition"
    tab["cycle"] = "I"
    return tab, covs


def stage2_proxy_validation(a_L, n_boot=2000, seed=SEED):
    """
    Cycle L measures dietary and serum magnesium in the same participants.

    lambda is the standardised slope of serum on dietary magnesium: the
    attenuation factor that a regression-calibration correction would divide
    by. It is reported together with R^2, because a lambda near zero means the
    correction is not merely imprecise but inapplicable.
    """
    rng = np.random.default_rng(seed)
    rows = []
    variants = [("mean of available recalls", "Mg_diet", None),
                ("two-day recalls only", "Mg_diet", "two"),
                ("energy-adjusted density", "Mg_dens", None)]
    for label, col, filt in variants:
        d = a_L.dropna(subset=[col, "Mg_serum"])
        if filt == "two":
            d = d[d["Mg_ndays"] == 2]
        if len(d) < 50:
            continue
        x, y = z(d[col]), z(d["Mg_serum"])
        r = stats.linregress(x, y)
        b = [stats.linregress(x[i], y[i]).slope
             for i in (rng.integers(0, len(d), len(d)) for _ in range(n_boot))]
        rows.append(dict(variant=label, n=len(d), lambda_=r.slope,
                         ci_lo=np.percentile(b, 2.5),
                         ci_hi=np.percentile(b, 97.5),
                         pearson_r=r.rvalue, r2_pct=100 * r.rvalue ** 2,
                         p=r.pvalue))
    return pd.DataFrame(rows)


def stage2b_independence(a_L):
    """
    Are dietary and serum magnesium the same exposure measured twice, or two
    different exposures?

    If the dietary variable retains a large coefficient conditional on serum
    magnesium, it is not a surrogate for it, and the classical
    measurement-error correction of stage 2 is invalid in principle.
    """
    tab, _, n = fit_endpoints(a_L, ("Ca", "Mg_serum"), extra=("Mg_diet",),
                              adjust_adiposity=False)
    tab["stage"] = "2b: are the two magnesium measures the same exposure?"
    tab["cycle"] = "L"
    return tab


def stage3_adiposity(a, cycle, ions):
    """The same models with and without adiposity, side by side."""
    out = []
    for adj in (False, True):
        t, covs, n = fit_endpoints(a, ions, adjust_adiposity=adj)
        t["stage"] = "3: adiposity adjustment"
        t["cycle"] = cycle
        out.append(t)
    return pd.concat(out, ignore_index=True)


# ----------------------------------------------------------------------
def coupling_vector(tab_I, tab_L, endpoint, cov_I=None, cov_L=None,
                    adiposity_adjusted=True):
    """
    Two-sample coupling vector for the closed-loop simulation.

    No NHANES cycle measures serum Zn, Ca and Mg together, so alpha_Zn comes
    from cycle I (the only cycle with serum zinc) and gamma_Ca, beta_Mg from
    cycle L (both serum, same model, same people). The two blocks come from
    independent samples, so their covariance is block diagonal. Everything is
    in standardised units, which is what makes the two samples commensurable;
    transportability across cycles is an assumption and is stated as such in
    the manuscript.
    """
    def pick(tab, term):
        m = ((tab.endpoint == endpoint) & (tab.term == term)
             & (tab.adiposity_adjusted == adiposity_adjusted))
        return tab[m].iloc[0]

    zn = pick(tab_I, "Zn")
    ca, mg = pick(tab_L, "Ca"), pick(tab_L, "Mg")
    mu = np.array([zn.coef, ca.coef, mg.coef])
    V = np.zeros((3, 3))
    V[0, 0] = zn.se ** 2
    if cov_L is not None:
        V[1:, 1:] = cov_L
    else:
        V[1, 1], V[2, 2] = ca.se ** 2, mg.se ** 2
    table = pd.DataFrame(dict(term=["beta_Zn", "beta_Ca", "beta_Mg"],
                              coef=mu, se=np.sqrt(np.diag(V))))
    return dict(table=table, icov=V, beta_full=mu)


def summarise(tab):
    """One-line verdict per (cycle, endpoint, ion, adjustment)."""
    t = tab.copy()
    t["signif"] = np.where(t.p < 0.05, "*", "")
    return t[["cycle", "stage", "endpoint", "term", "adiposity_adjusted",
              "coef", "ci_lo", "ci_hi", "p", "signif", "n"]]
