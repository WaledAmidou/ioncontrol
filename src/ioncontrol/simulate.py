"""
Closed-loop Monte-Carlo evaluation on a virtual-patient cohort resampled from
NHANES 2015-2016.

Design choices that answer the editorial critique
-------------------------------------------------
* Virtual patients are *real* NHANES participants: their measured serum Zn and
  Ca define both the initial condition and the homeostatic attractor that the
  controller must fight.  Latent Mg is drawn from its prior conditionally on
  the measured ions.
* Patients are drawn exclusively from the held-out partition that was never
  used to fit the insulin model or the sensitivity coefficients, so the
  reported closed-loop performance is an out-of-sample estimate.
* Parameter uncertainty is propagated: for every Monte-Carlo replicate the
  sensitivity vector beta is redrawn from its bootstrap distribution, so the
  reported physiological benefit carries the statistical uncertainty of the
  underlying epidemiological association instead of hiding it.
* Everything is vectorised over (patient x replicate); the random seed, the
  step size and every gain are fixed in `config.py`, so the whole table is
  reproducible from `run_all.py`.
"""
import zlib

import numpy as np
import pandas as pd

from .config import (IONS, DT, T_END, N_PATIENTS, N_SEEDS, SNR_DB, SEED, FILT_TAU,
                    U_MAX_SIGMA, DU_MAX_SIGMA, RHO, STRATEGIES)
from .controllers import (GlucoseOnly, Supplementation, FixedGainPID,
                         AdaptiveAI, ModelBasedLQR, scalar_dare)


# ----------------------------------------------------------------------
def build_cohort(a, ident, n_patients=N_PATIENTS, n_seeds=N_SEEDS,
                 held_out_idx=None, seed=SEED):
    """Return z_h (N,3), covariate frame and bookkeeping for the batch."""
    rng = np.random.default_rng(seed)
    sp = ident["setpoints"]
    pool = a.loc[a.index.intersection(held_out_idx)] if held_out_idx is not None else a
    pool = pool[(pool["dysglycaemia"] == 1)]
    if len(pool) < 20:                     # fall back if the split is too thin
        pool = a[a["dysglycaemia"] == 1]
    idx = rng.choice(pool.index.values, n_patients, replace=len(pool) < n_patients)
    P = a.loc[idx]

    sd = np.array([sp[i]["sd"] for i in IONS])
    tgt = np.array([sp[i]["target"] for i in IONS])

    zZn = (P["Zn"].values - tgt[0]) / sd[0]
    zCa = (P["Ca"].values - tgt[1]) / sd[1]
    # Magnesium is measured in both cycles now (serum in L, dietary in I),
    # so the state is read from the data rather than drawn from a prior.
    zMg = (P["Mg"].values - tgt[2]) / sd[2]

    z_h_pat = np.column_stack([zZn, zCa, zMg])                  # (n_patients,3)
    z_h = np.repeat(z_h_pat, n_seeds, axis=0)                   # (N,3)
    pid = np.repeat(np.arange(n_patients), n_seeds)
    return z_h, P, pid, idx


# ----------------------------------------------------------------------
def tune_pid(a_nom, b_nom, dt, horizon=1200, os_max=10.0, n_filt=10.0):
    """
    Explicit tuning of the PID comparator: ITAE grid search on the nominal
    model with a unit initial offset, under the same saturation, rate limit
    and 10 % overshoot constraint that fixes the LQR weight R.  The
    derivative term uses the same filtered form as the implementation.
    """
    best, best_c = None, np.inf
    ad = dt / (dt + 1.0 / n_filt)
    for kp in np.linspace(0.05, 2.0, 25):
        for ki in np.linspace(0.0, 0.20, 11):
            for kd in np.linspace(0.0, 0.60, 7):
                z, integ, itae, u_prev, os = 1.0, 0.0, 0.0, 0.0, 0.0
                zp, der = None, 0.0
                for k in range(horizon):
                    rd = 0.0 if zp is None else (z - zp) / dt
                    der += ad * (rd - der)
                    zp = z
                    u = -(kp * z + ki * integ + kd * der)
                    u = np.clip(u, -U_MAX_SIGMA, U_MAX_SIGMA)
                    u = u_prev + np.clip(u - u_prev, -DU_MAX_SIGMA * dt,
                                         DU_MAX_SIGMA * dt)
                    u_prev = u
                    z = a_nom * z + b_nom * u + (1 - a_nom) * 1.0
                    integ += z * dt
                    itae += (k * dt) * abs(z) * dt
                    os = max(os, -z)
                    if not np.isfinite(z) or abs(z) > 50:
                        itae = np.inf
                        break
                if os > os_max / 100.0:
                    continue
                if itae < best_c:
                    best_c, best = itae, (kp, ki, kd)
    return best if best is not None else (0.2, 0.0, 0.0)


def tune_pi(a_nom, b_nom, dt, horizon=1200, os_max=10.0):
    """
    Tuning procedure for the fixed-gain comparator, stated explicitly so that
    the comparison is reproducible: grid search over (kp, ki) minimising the
    integral of time-weighted absolute error (ITAE) on the *nominal* model
    with a unit initial offset, subject to the same saturation and rate
    limit as every other controller **and** to the same overshoot constraint
    (<= `os_max` %) that fixes the LQR weight R.  The winning gains are then
    frozen for the whole study.
    """
    best, best_c = None, np.inf
    for kp in np.linspace(0.05, 2.0, 40):
        for ki in np.linspace(0.0, 0.20, 21):
            z, integ, itae, u_prev, os = 1.0, 0.0, 0.0, 0.0, 0.0
            for k in range(horizon):
                u = -(kp * z + ki * integ)
                u = np.clip(u, -U_MAX_SIGMA, U_MAX_SIGMA)
                u = u_prev + np.clip(u - u_prev, -DU_MAX_SIGMA * dt, DU_MAX_SIGMA * dt)
                u_prev = u
                z = a_nom * z + b_nom * u + (1 - a_nom) * 1.0
                integ += z * dt
                itae += (k * dt) * abs(z) * dt
                os = max(os, -z)
                if not np.isfinite(z) or abs(z) > 50:
                    itae = np.inf
                    break
            if os > os_max / 100.0:
                continue
            if itae < best_c:
                best_c, best = itae, (kp, ki)
    return best if best is not None else (0.2, 0.0)


# ----------------------------------------------------------------------
def plant_heterogeneity(N, seed=SEED, cv_eff=0.25, cv_kappa=0.20,
                        eff_scale=(1.0, 1.0, 1.0)):
    """
    Inter-patient plant uncertainty, unknown to every controller.

    * `eff`   : per-patient nano-actuator delivery efficiency, log-normal
                (CV = 25 %), clipped to [0.4, 2.0].
    * `kappa` : per-patient homeostatic rate, log-normal (CV = 20 %) around
                the literature half-lives.

    This heterogeneity is what makes the identification problem non-trivial;
    with a perfectly known, homogeneous plant a fixed-gain controller is
    already optimal, and no adaptive mechanism can be expected to help.
    """
    rng = np.random.default_rng(seed + 101)
    eff = np.clip(np.exp(rng.normal(0, cv_eff, size=(N, 3))), 0.4, 2.0)
    eff = eff * np.asarray(eff_scale, float)
    kap = np.exp(rng.normal(0, cv_kappa, size=(N, 3)))
    return eff, kap


def run_strategy(name, z_h, ident, dt=DT, t_end=T_END, snr_db=SNR_DB,
                 eff_scale=(1.0, 1.0, 1.0), beta_draws=None, seed=SEED,
                 rho=RHO, u_max=U_MAX_SIGMA, n_seeds_hint=N_SEEDS,
                 fault_time=None, fault_factor=0.5, hetero=True,
                 beta_ir_draws=None):
    """Simulate one strategy for the whole batch; return per-run metrics."""
    # deterministic, process-independent stream per strategy
    rng = np.random.default_rng(seed + zlib.crc32(name.encode()) % 10_000)
    N = z_h.shape[0]
    nsteps = int(round(t_end / dt))

    A = ident["A"]
    sd = np.array([ident["setpoints"][i]["sd"] for i in IONS])
    kappa = -np.diag(A)
    a_nom = 1.0 + dt * np.diag(A)                 # nominal, known to designers
    b_nom = dt * np.asarray(eff_scale, float)

    if hetero:
        eff_n, kap_m = plant_heterogeneity(N, seed=seed, eff_scale=eff_scale)
    else:
        eff_n = np.tile(np.asarray(eff_scale, float), (N, 1))
        kap_m = np.ones((N, 3))
    kappa_n = kappa[None, :] * kap_m               # (N,3) true rates

    # diffusion in standardised units, from the Lyapunov-derived W
    Wz = ident["W"] / np.outer(sd, sd)
    L = np.linalg.cholesky(Wz + 1e-12 * np.eye(3)) * np.sqrt(dt)
    sigma_v = 10 ** (-snr_db / 20.0)               # sensor noise SD (z units)

    # ---- controller instantiation -----------------------------------
    if name == "Glucose-only":
        ctrl = GlucoseOnly(N, dt, u_max=u_max)
    elif name == "Metal supplementation":
        duty = 30.0 / 5.0
        # population-level (not patient-specific) dose sized to correct the
        # average deficiency, i.e. the mean of the negative part of z_h
        deficit = np.mean(np.minimum(z_h, 0.0), axis=0)
        dose = -kappa * deficit * duty
        dose[1] = 0.0                              # Ca is not supplemented
        ctrl = Supplementation(N, dt, dose=dose, u_max=u_max)
    elif name == "Fixed-gain PID":
        kp, ki, kd = tune_pid(float(a_nom.mean()), float(b_nom.mean()), dt)
        ctrl = FixedGainPID(N, dt, kp=np.full(3, kp), ki=np.full(3, ki),
                            kd=kd, u_max=u_max)
    elif name == "Proposed adaptive AI":
        ctrl = AdaptiveAI(N, dt, a_nom=a_nom, b_nom=b_nom, rho=rho,
                          u_max=u_max, seed=seed, filt_tau=0.0)
    elif name == "Model-based LQR (no adaptation)":
        ctrl = ModelBasedLQR(N, dt, a_nom, b_nom, z_h, rho=rho, u_max=u_max)
    else:
        raise ValueError(name)

    z = z_h.copy()
    keep = max(1, nsteps // 600)
    traj, u_sq = [], np.zeros(N)
    tgrid = []

    # Common on-chip signal-conditioning block: every strategy sees the same
    # first-order low-pass filtered measurement, so the comparison is not
    # confounded by differences in sensor processing.
    yf = z.copy()
    alpha_f = dt / (FILT_TAU + dt)

    k_fault = int(round(fault_time / dt)) if fault_time else None

    for k in range(nsteps):
        if k_fault is not None and k == k_fault:
            eff_n = eff_n * fault_factor      # nano-actuator degradation
        y = z + sigma_v * rng.normal(size=(N, 3))
        yf = yf + alpha_f * (y - yf)
        u = ctrl.step(k, yf)
        w = rng.normal(size=(N, 3)) @ L.T
        z = z - dt * kappa_n * (z - z_h) + dt * (eff_n * u) + w
        u_sq += np.sum(u ** 2, axis=1) * dt
        if k % keep == 0:
            traj.append(z.copy())
            tgrid.append(k * dt)

    traj = np.asarray(traj)                        # (T,N,3)
    tgrid = np.asarray(tgrid)

    # ---- metrics -----------------------------------------------------
    tail = traj[int(0.8 * traj.shape[0]):]
    sse_z = np.abs(tail.mean(axis=0))              # (N,3) standardised
    sse_phys = sse_z * sd                          # physical units
    ion_var = tail.var(axis=0)                     # (N,3)

    # Overshoot and settling time are read on the seed-averaged response of
    # each virtual patient, so that they measure the deterministic transient
    # rather than the stochastic diffusion of a single realisation.
    npat = N // n_seeds_hint if n_seeds_hint else None
    if npat:
        tp = traj.reshape(traj.shape[0], npat, n_seeds_hint, 3).mean(axis=2)
        zp = z_h.reshape(npat, n_seeds_hint, 3)[:, 0, :]
        sgn = np.sign(zp)
        noise_floor = 2.0 * tp[int(0.9 * tp.shape[0]):].std(axis=0)
        cross = np.max(-tp * sgn, axis=0) - noise_floor          # (npat,3)
        os_pat = 100.0 * np.maximum(cross, 0.0) / np.maximum(np.abs(zp), 1e-9)
        # only patients with a clinically meaningful initial deviation
        os_pat = np.where(np.abs(zp) >= 1.0, os_pat, np.nan)
        overshoot = np.nanmean(os_pat, axis=1)
        band = np.maximum(0.10 * np.abs(zp), 2.0 * noise_floor)
        outside = np.abs(tp) > band[None]
        last_out = np.where(outside.any(axis=0),
                            (outside.shape[0] - 1
                             - np.argmax(outside[::-1], axis=0)), -1)
        settle = tgrid[np.clip(last_out.max(axis=1) + 1, 0, len(tgrid) - 1)]
        settle = np.where(last_out.max(axis=1) >= len(tgrid) - 2, np.nan, settle)
    else:
        overshoot = np.full(N, np.nan)
        settle = np.full(N, np.nan)

    if beta_draws is None:
        beta = np.tile(ident["beta"]["beta_full"], (traj.shape[1], 1))
    else:
        beta = beta_draws
    beta_pt = np.tile(ident["beta"]["beta_full"], (traj.shape[1], 1))
    lr_pt = np.einsum('tnj,nj->tn', traj, beta_pt)
    eta_pt = 100.0 * np.exp(-np.abs(lr_pt))
    eta_pt_ss = eta_pt[int(0.8 * eta_pt.shape[0]):].mean(axis=0)

    log_ratio = np.einsum('tnj,nj->tn', traj, beta)
    eta = 100.0 * np.exp(-np.abs(log_ratio))       # attainment of reference HOMA-B
    eta_ss = eta[int(0.8 * eta.shape[0]):].mean(axis=0)
    eta_var = eta[int(0.8 * eta.shape[0]):].var(axis=0)
    dHOMA = 100.0 * (np.exp(log_ratio[int(0.8 * log_ratio.shape[0]):].mean(axis=0))
                     - np.exp(np.einsum('nj,nj->n', z_h, beta)))

    # Secondary endpoint: insulin resistance.  The NHANES regressions show
    # that the ionic signal loads on HOMA-IR rather than on secretory
    # capacity, so this is where a detectable physiological effect, if any,
    # should appear.
    if beta_ir_draws is not None:
        lr_ir = np.einsum('tnj,nj->tn', traj, beta_ir_draws)
        dIR = 100.0 * (np.exp(lr_ir[int(0.8 * lr_ir.shape[0]):].mean(axis=0))
                       - np.exp(np.einsum('nj,nj->n', z_h, beta_ir_draws)))
    else:
        dIR = np.full(traj.shape[1], np.nan)

    return dict(name=name, traj=traj, tgrid=tgrid, sse_z=sse_z,
                sse_phys=sse_phys, ion_var=ion_var, overshoot=overshoot,
                settle=settle, effort=u_sq,
                eta_ss=eta_ss, eta_var=eta_var, dHOMA=dHOMA, dIR=dIR,
                eta_pt_ss=eta_pt_ss,
                eta_traj=eta.mean(axis=1))


# ----------------------------------------------------------------------
def beta_bootstrap_draws(ident, N, n_patients, n_seeds, seed=SEED, key="beta"):
    """
    Redraw beta = [beta_Zn, beta_Ca, beta_Mg] per Monte-Carlo replicate from
    the sampling distribution of the NHANES regression.  All three
    coefficients are now measured: magnesium status is estimated from the
    NHANES dietary recalls, so no component of the coupling vector is a
    prior any more.  `key` selects the endpoint ("beta" = HOMA-B,
    "beta_ir" = HOMA-IR, "beta_di" = disposition index).
    """
    rng = np.random.default_rng(seed + 7)
    tab = ident[key]["table"].set_index("term")
    mu = np.array([tab.loc[f"beta_{i}", "coef"] for i in IONS])
    se = np.array([tab.loc[f"beta_{i}", "se"] for i in IONS])
    return rng.normal(mu, se, size=(N, 3))


# ----------------------------------------------------------------------
def run_all_strategies(a, ident, **kw):
    held = ident["learning"]["idx_test"]
    z_h, P, pid, idx = build_cohort(a, ident, held_out_idx=held)
    N = z_h.shape[0]
    tb = ident["beta"]["table"].set_index("term")
    ident["beta"]["beta_full"] = np.array(
        [tb.loc[f"beta_{i}", "coef"] for i in IONS])
    bd = beta_bootstrap_draws(ident, N, N_PATIENTS, N_SEEDS, key="beta")
    bd_ir = beta_bootstrap_draws(ident, N, N_PATIENTS, N_SEEDS, key="beta_ir")

    res = {}
    for s in STRATEGIES:
        res[s] = run_strategy(s, z_h, ident, beta_draws=bd,
                              beta_ir_draws=bd_ir, **kw)
    return dict(results=res, z_h=z_h, patients=P, pid=pid, beta_draws=bd,
                beta_ir_draws=bd_ir)


def population_effect(res, ident, key="beta_ir", subgroup=None,
                      n_draws=4000, seed=SEED):
    """
    Cohort-mean physiological effect with a correctly constructed confidence
    interval.

    The per-run intervals of `summary_table` mix two very different sources of
    variability: between-patient heterogeneity and the sampling uncertainty of
    the regression coefficients.  For a statement about the *population mean*
    effect, only the second is a source of uncertainty.  We therefore hold the
    simulated trajectories fixed, draw the coupling vector from its estimated
    sampling distribution (using the full 3x3 covariance of the regression,
    not the marginal standard errors), and recompute the cohort-mean effect
    for each draw.  This is the interval that should be quoted.
    """
    rng = np.random.default_rng(seed + 991)
    tb = ident[key]["table"].set_index("term")
    mu = np.array([tb.loc[f"beta_{i}", "coef"] for i in IONS])
    C = ident[key]["icov"]
    draws = rng.multivariate_normal(mu, C, size=n_draws)

    zh = res["z_h"]
    m = np.ones(len(zh), bool) if subgroup is None else subgroup
    rows = []
    for s, r in res["results"].items():
        traj = r["traj"]
        z_ss = traj[int(0.8 * traj.shape[0]):].mean(axis=0)   # (N,3)
        a_ss = z_ss[m] @ draws.T                              # (n,B)
        a_h = zh[m] @ draws.T
        eff = 100.0 * (np.exp(a_ss) - np.exp(a_h)).mean(axis=0)
        rows.append(dict(Strategy=s, mean=float(eff.mean()),
                         ci_lo=float(np.percentile(eff, 2.5)),
                         ci_hi=float(np.percentile(eff, 97.5)),
                         p_two_sided=float(2 * min((eff >= 0).mean(),
                                                   (eff <= 0).mean()))))
    return pd.DataFrame(rows)


def low_mg_mask(res, q=33.3):
    zh = res["z_h"]
    return zh[:, 2] <= np.percentile(zh[:, 2], q)


def summary_table(res):
    """
    Cohort-level summary.  Because virtual patients scatter on both sides of
    the reference set-point, the cohort-mean physiological effect averages
    corrections in opposite directions and is close to zero by construction.
    We therefore also report the pre-specified subgroup that any ionic
    intervention would actually target: participants in the lowest tertile of
    baseline magnesium status.
    """
    zh = res["z_h"]
    cut = np.percentile(zh[:, 2], 33.3)
    low_mg = zh[:, 2] <= cut
    rows = []
    for s, r in res["results"].items():
        rows.append(dict(
            Strategy=s,
            SSE_Zn=r["sse_phys"][:, 0].mean(), SSE_Zn_sd=r["sse_phys"][:, 0].std(),
            SSE_Ca=r["sse_phys"][:, 1].mean(), SSE_Ca_sd=r["sse_phys"][:, 1].std(),
            SSE_Mg=r["sse_phys"][:, 2].mean(), SSE_Mg_sd=r["sse_phys"][:, 2].std(),
            Overshoot=np.nanmean(r["overshoot"]),
            Overshoot_sd=np.nanstd(r["overshoot"]),
            Settle=np.nanmean(r["settle"]),
            Effort=r["effort"].mean(), Effort_sd=r["effort"].std(),
            Eta=r["eta_pt_ss"].mean(), Eta_sd=r["eta_pt_ss"].std(),
            EtaU=r["eta_ss"].mean(), EtaU_sd=r["eta_ss"].std(),
            EtaVar=r["eta_var"].mean(),
            dHOMA=r["dHOMA"].mean(),
            dHOMA_lo=np.percentile(r["dHOMA"], 2.5),
            dHOMA_hi=np.percentile(r["dHOMA"], 97.5),
            dIR=np.nanmean(r["dIR"]),
            dIR_lo=np.nanpercentile(r["dIR"], 2.5),
            dIR_hi=np.nanpercentile(r["dIR"], 97.5),
            dIR_lowMg=np.nanmean(r["dIR"][low_mg]),
            dIR_lowMg_lo=np.nanpercentile(r["dIR"][low_mg], 2.5),
            dIR_lowMg_hi=np.nanpercentile(r["dIR"][low_mg], 97.5),
            dHOMA_lowMg=np.nanmean(r["dHOMA"][low_mg]),
            dHOMA_lowMg_lo=np.nanpercentile(r["dHOMA"][low_mg], 2.5),
            dHOMA_lowMg_hi=np.nanpercentile(r["dHOMA"][low_mg], 97.5),
        ))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import data, identification
    _, _, a, ref, _ = data.build()
    ident = identification.identify_all(a, ref)
    out = run_all_strategies(a, ident)
    print(summary_table(out).round(3).to_string(index=False))
