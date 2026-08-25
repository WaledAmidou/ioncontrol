"""
Diagnostic evaluation.

The title of the manuscript advertises a diagnostic strategy, but the
submitted version contained no cohort, no reference standard, no screening
task and no measure of diagnostic accuracy.  This module supplies all four,
using the NHANES analytic cohort, and reports the result honestly, including
where the ionic biomarkers fail.

Tasks
-----
D1  Diabetes (self-reported physician diagnosis or FPG >= 126 mg/dL)
    versus normoglycaemia (no diagnosis and FPG < 100 mg/dL).
D2  Dysglycaemia (diagnosis or FPG >= 100 mg/dL) versus normoglycaemia.
D3  Impaired beta-cell secretory capacity, defined as the lowest quartile of
    HOMA-B among participants without diagnosed diabetes.  The predictors
    exclude glucose and insulin, so the task is not circular.
D4  Insulin-treated early-onset diabetes (the standard NHANES surrogate for
    type 1 diabetes).  Reported for completeness and declared non-evaluable.

Predictor sets
--------------
IONIC        Zn, Ca (the ions the framework actuates) (+ Cu, Se, Cu/Zn as
             the other measured trace metals)
IONIC+DEMO   the above plus age, sex and the income-to-poverty ratio
DEMO         demographics only -- the clinically free baseline that any new
             biomarker must beat
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_curve, roc_auc_score, brier_score_loss

from .config import SEED

PANELS = {
    "Regulated ions (Zn, Ca, Mg)": ["Zn", "Ca", "Mg"],
    "All measured trace metals": ["Zn", "Ca", "Mg", "Mg_dens", "Cu", "Se", "CuZn"],
    "Trace metals + demographics": ["Zn", "Ca", "Mg", "Mg_dens", "Cu", "Se",
                                    "CuZn", "age", "male", "INDFMPIR"],
    "Demographics only (baseline)": ["age", "male", "INDFMPIR"],
}


def _delong_ci(y, s, alpha=0.05):
    """Non-parametric AUC standard error (Hanley-McNeil) and Wald CI."""
    y = np.asarray(y).astype(bool)
    auc = roc_auc_score(y, s)
    n1, n0 = y.sum(), (~y).sum()
    q1 = auc / (2 - auc)
    q2 = 2 * auc ** 2 / (1 + auc)
    se = np.sqrt((auc * (1 - auc) + (n1 - 1) * (q1 - auc ** 2)
                  + (n0 - 1) * (q2 - auc ** 2)) / (n1 * n0))
    z = stats.norm.ppf(1 - alpha / 2)
    return float(auc), float(max(0, auc - z * se)), float(min(1, auc + z * se)), float(se)


def _operating_point(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    j = np.argmax(tpr - fpr)
    sens, spec = tpr[j], 1 - fpr[j]
    prev = np.mean(y)
    ppv = sens * prev / max(sens * prev + (1 - spec) * (1 - prev), 1e-12)
    npv = spec * (1 - prev) / max(spec * (1 - prev) + (1 - sens) * prev, 1e-12)
    return dict(threshold=float(thr[j]), sensitivity=float(sens),
                specificity=float(spec), youden=float(sens + spec - 1),
                ppv=float(ppv), npv=float(npv), prevalence=float(prev))


def _task_frame(a, task):
    d = a.dropna(subset=["Zn", "Ca", "Mg", "Mg_dens", "Cu", "Se", "CuZn",
                         "age", "male", "INDFMPIR"]).copy()
    if task == "D1":
        m = (d["diabetes"] == 1) | (d["normoglycaemic"] == 1)
        d = d[m]
        y = (d["diabetes"] == 1).astype(int).values
    elif task == "D2":
        m = (d["dysglycaemia"] == 1) | (d["normoglycaemic"] == 1)
        d = d[m]
        y = (d["dysglycaemia"] == 1).astype(int).values
    elif task == "D3":
        d = d[(d["dm_dx"] == 0) & d["HOMA_B"].notna() & (d["HOMA_B"] > 0)]
        cut = np.percentile(d["HOMA_B"], 25)
        y = (d["HOMA_B"] <= cut).astype(int).values
    elif task == "D4":
        m = (d["t1d_like"] == 1) | (d["normoglycaemic"] == 1)
        d = d[m]
        y = (d["t1d_like"] == 1).astype(int).values
    else:
        raise ValueError(task)
    return d, y


def evaluate(a, tasks=("D1", "D2", "D3", "D4"), seed=SEED, n_splits=10):
    rows, curves = [], {}
    for task in tasks:
        d, y = _task_frame(a, task)
        n_pos = int(y.sum())
        if n_pos < 10 or len(y) - n_pos < 10:
            rows.append(dict(task=task, panel="-", n=len(y), n_pos=n_pos,
                             auc=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                             sensitivity=np.nan, specificity=np.nan,
                             ppv=np.nan, npv=np.nan, brier=np.nan,
                             note="not evaluable: fewer than 10 positive cases"))
            continue
        for pname, feats in PANELS.items():
            X = d[feats].values
            mdl = Pipeline([("s", StandardScaler()),
                            ("m", LogisticRegression(max_iter=5000, C=1.0))])
            cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
            s = cross_val_predict(mdl, X, y, cv=cv, method="predict_proba")[:, 1]
            auc, lo, hi, se = _delong_ci(y, s)
            op = _operating_point(y, s)
            rows.append(dict(task=task, panel=pname, n=len(y), n_pos=n_pos,
                             auc=auc, ci_lo=lo, ci_hi=hi, se=se,
                             brier=float(brier_score_loss(y, s)),
                             note="", **op))
            curves[(task, pname)] = (y, s)
    return pd.DataFrame(rows), curves


def delong_compare(y, s1, s2, n_boot=4000, seed=SEED):
    """Bootstrap test for the difference between two correlated AUCs."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    diffs = []
    idx_pos, idx_neg = np.where(y == 1)[0], np.where(y == 0)[0]
    for _ in range(n_boot):
        ip = rng.choice(idx_pos, idx_pos.size, replace=True)
        ineg = rng.choice(idx_neg, idx_neg.size, replace=True)
        ii = np.concatenate([ip, ineg])
        diffs.append(roc_auc_score(y[ii], s1[ii]) - roc_auc_score(y[ii], s2[ii]))
    diffs = np.asarray(diffs)
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return dict(delta=float(roc_auc_score(y, s1) - roc_auc_score(y, s2)),
                ci_lo=float(np.percentile(diffs, 2.5)),
                ci_hi=float(np.percentile(diffs, 97.5)), p=float(min(p, 1.0)))
