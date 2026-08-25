"""
Sensitivity and robustness analysis.

The submitted manuscript stated that robustness "should be quantified" and
then gave illustrative numbers that "would make" the argument stronger.  This
module actually performs the analysis and returns measured numbers:

  * a variance-based global sensitivity analysis (Sobol first-order and total
    indices, Saltelli estimator) over every uncertain parameter, including
    the ones the reviewer singled out (unmeasured Mg coupling, unvalidated
    homeostatic rates, actuator gains, sensor noise);
  * a local one-at-a-time sweep of the actuator gain over +/- 20 %;
  * a nanosensor SNR sweep from 40 dB down to 10 dB.
"""
import numpy as np
import pandas as pd

from .config import IONS, DT, SEED
from . import simulate


PARAMS = [
    ("kappa_Zn", 0.5, 1.5),      # multiplier on the literature half-life
    ("kappa_Ca", 0.5, 1.5),
    ("kappa_Mg", 0.5, 1.5),
    ("actuator_gain", 0.6, 1.4),
    ("snr_db", 10.0, 40.0),
    ("rho", 0.5, 8.0),
    ("beta_Zn", -0.089, 0.020),  # empirical 95 % bootstrap intervals,
    ("beta_Ca", -0.095, 0.021),  # all three now measured (Mg from the
    ("beta_Mg", -0.127, -0.005), # NHANES dietary recalls)
    ("u_max", 0.30, 1.00),
]


def _one_run(theta, a, ident, z_h, n_seeds, strategy="Proposed adaptive AI"):
    """Evaluate the model for one parameter vector; return (SSE_Zn, eta)."""
    d = dict(zip([p[0] for p in PARAMS], theta))
    ident2 = dict(ident)
    A = ident["A"].copy()
    mult = np.array([d["kappa_Zn"], d["kappa_Ca"], d["kappa_Mg"]])
    ident2["A"] = np.diag(np.diag(A) * mult)
    ident2["W"] = ident["W"] * ((mult[:, None] + mult[None, :]) / 2.0)
    ident2["beta"] = dict(ident["beta"])
    ident2["beta"]["beta_full"] = np.array([d["beta_Zn"], d["beta_Ca"], d["beta_Mg"]])

    r = simulate.run_strategy(
        strategy, z_h, ident2, snr_db=d["snr_db"], rho=d["rho"],
        u_max=d["u_max"], eff_scale=(d["actuator_gain"],) * 3,
        n_seeds_hint=n_seeds, fault_time=120.0)
    return float(r["sse_phys"][:, 0].mean()), float(r["eta_pt_ss"].mean())


def sobol(a, ident, n_base=32, n_patients=30, n_seeds=3, seed=SEED):
    """Saltelli sampling; first-order (S1) and total-order (ST) indices."""
    rng = np.random.default_rng(seed)
    k = len(PARAMS)
    lo = np.array([p[1] for p in PARAMS])
    hi = np.array([p[2] for p in PARAMS])

    z_h, _, _, _ = simulate.build_cohort(
        a, ident, n_patients=n_patients, n_seeds=n_seeds,
        held_out_idx=ident["learning"]["idx_test"])

    A_ = lo + (hi - lo) * rng.random((n_base, k))
    B_ = lo + (hi - lo) * rng.random((n_base, k))

    def ev(M):
        return np.array([_one_run(t, a, ident, z_h, n_seeds) for t in M])

    fA, fB = ev(A_), ev(B_)
    S1 = np.zeros((k, 2))
    ST = np.zeros((k, 2))
    varY = fA.var(axis=0)
    for i in range(k):
        AB = A_.copy()
        AB[:, i] = B_[:, i]
        fAB = ev(AB)
        S1[i] = np.mean(fB * (fAB - fA), axis=0) / varY
        ST[i] = 0.5 * np.mean((fA - fAB) ** 2, axis=0) / varY
    names = [p[0] for p in PARAMS]
    return pd.DataFrame(dict(parameter=names,
                             S1_sse=S1[:, 0], ST_sse=ST[:, 0],
                             S1_eta=S1[:, 1], ST_eta=ST[:, 1]))


def gain_sweep(a, ident, deltas=np.linspace(-0.20, 0.20, 9),
               n_patients=60, n_seeds=5):
    z_h, _, _, _ = simulate.build_cohort(
        a, ident, n_patients=n_patients, n_seeds=n_seeds,
        held_out_idx=ident["learning"]["idx_test"])
    rows = []
    for dl in deltas:
        r = simulate.run_strategy("Proposed adaptive AI", z_h, ident,
                                  eff_scale=(1 + dl,) * 3, n_seeds_hint=n_seeds,
                                  fault_time=120.0)
        rows.append(dict(delta_gain=float(dl),
                         sse_Zn=float(r["sse_phys"][:, 0].mean()),
                         eta=float(r["eta_pt_ss"].mean()),
                         overshoot=float(np.nanmean(r["overshoot"])),
                         effort=float(r["effort"].mean())))
    return pd.DataFrame(rows)


def snr_sweep(a, ident, snrs=(40, 35, 30, 25, 20, 15, 10),
              n_patients=60, n_seeds=5):
    z_h, _, _, _ = simulate.build_cohort(
        a, ident, n_patients=n_patients, n_seeds=n_seeds,
        held_out_idx=ident["learning"]["idx_test"])
    rows = []
    for s in snrs:
        r = simulate.run_strategy("Proposed adaptive AI", z_h, ident,
                                  snr_db=float(s), n_seeds_hint=n_seeds,
                                  fault_time=120.0)
        rows.append(dict(snr_db=float(s),
                         sse_Zn=float(r["sse_phys"][:, 0].mean()),
                         eta=float(r["eta_pt_ss"].mean()),
                         effort=float(r["effort"].mean())))
    return pd.DataFrame(rows)


def relative_variation(df, col, ref_col, ref_val):
    base = df.loc[np.isclose(df[ref_col], ref_val), col].values[0]
    return 100.0 * (df[col].values - base) / base
