"""LaTeX tables added to the revised manuscript (all wrapped in \\rev)."""
import numpy as np
import pandas as pd

from .config import TAB, IONS, ION_UNITS, DT, T_END, N_PATIENTS, N_SEEDS, RHO, SNR_DB


def _w(name, s):
    (TAB / f"{name}.tex").write_text(s)
    return s


def esc(x):
    return str(x).replace("_", r"\_").replace("%", r"\%")


# ----------------------------------------------------------------------
def tab_cohort(flow, a, ref):
    rows = "\n".join(fr"{esc(k)} & {v} \\ \hline" for k, v in flow.items())
    s = r"""\begin{table}[!t]
\centering
\caption{\rev{NHANES 2015--2016 participant flow and analytic cohort. Data
provenance: public NHANES cycle I files DEMO\_I, BIOPRO\_I, CUSEZN\_I,
GLU\_I, INS\_I and DIQ\_I.}}
\label{tab:cohort}
{\footnotesize
\begin{tabular}{|p{6.0cm}|r|}
\hline
\textbf{Stage} & \textbf{n} \\ \hline
""" + rows + r"""
\end{tabular}}
\end{table}"""
    return _w("tab_cohort", s)


def tab_setpoints(sp):
    body = ""
    for ion in IONS:
        d = sp[ion]
        body += (fr"$\mathrm{{{ion}^{{2+}}}}$ & {esc(ION_UNITS[ion])} & "
                 fr"{d['target']:.2f} & [{d['ci'][0]:.2f}, {d['ci'][1]:.2f}] & "
                 fr"{d['sd']:.2f} \\ \hline" + "\n")
    s = r"""\begin{table}[!t]
\centering
\caption{\rev{Control set-points $\mathbf{M}_{\mathrm{ref}}$ derived from the
metabolically healthy NHANES reference subgroup, replacing the illustrative
targets of the original submission. All three ions are now measured: Zn and Ca
in serum, Mg as the mean of the two NHANES 24-h dietary recalls
(DR1TMAGN/DR2TMAGN) used as a monotone proxy for magnesium status.}}
\label{tab1}
{\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{|c|c|c|c|c|}
\hline
\textbf{Ion} & \textbf{Unit} & \textbf{Target} & \textbf{95\% CI} & \textbf{SD} \\ \hline
""" + body + r"""\end{tabular}}
\vspace{1mm}
\parbox{\linewidth}{\scriptsize \revc{Medians of the metabolically healthy
reference subgroup; CIs from a 2000-resample non-parametric bootstrap of the
median \cite{NHANES2015Lab}. The Mg row is a dietary intake, not a
concentration; the state carried into the controller is the standardised
Mg-status score, never the raw intake.}}
\end{table}"""
    return _w("tab_setpoints", s)


def tab_model(ident):
    A, W, S = ident["A"], ident["W"], ident["Sigma"]
    bt = ident["beta"]["table"].set_index("term")
    kap = -np.diag(A)
    rows = ""
    for i, ion in enumerate(IONS):
        rows += (fr"$\kappa_{{{ion}}}$ (min$^{{-1}}$) & {kap[i]:.5f} & "
                 fr"literature half-life, swept $\pm 50\%$ in the Sobol analysis \\ \hline" + "\n")
    for i, ion in enumerate(IONS):
        rows += (fr"$\Sigma_{{{ion}{ion}}}$ & {S[i, i]:.4g} & "
                 fr"NHANES reference-subgroup variance \\ \hline" + "\n")
    for i, ion in enumerate(IONS):
        rows += (fr"$W_{{{ion}{ion}}}$ & {W[i, i]:.4g} & "
                 fr"derived: $W=-(A\Sigma+\Sigma A^{{T}})$ \\ \hline" + "\n")
    for term, sym in (("beta_Zn", r"\alpha_{Zn}"), ("beta_Ca", r"\alpha_{Ca}")):
        r = bt.loc[term]
        rows += (fr"${sym}$ & {r['coef']:+.4f} & NHANES regression, 95\% CI "
                 fr"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}], $p={r['p']:.3f}$ \\ \hline" + "\n")
    rows += (r"$\alpha_{Mg}$ & not identifiable & serum Mg is not measured in "
             r"NHANES 2015--2016; treated as latent \\ \hline" + "\n")
    rows += (fr"$\mathbf{{Q}}$ & $\mathrm{{diag}}(1/\sigma_i^2)$ & data-derived: "
             fr"unit weight per reference SD \\ \hline" + "\n")
    rows += fr"$\mathbf{{R}}$ & ${RHO:.1f}\,\mathbf{{I}}$ & smallest value keeping nominal overshoot $<10\%$ \\ \hline" + "\n"
    rows += fr"$\Delta t$ / horizon & {DT} / {T_END:.0f} min & forward Euler \\ \hline" + "\n"
    rows += fr"SNR & {SNR_DB:.0f} dB & nominal nanosensor noise \\ \hline" + "\n"

    s = r"""\begin{table}[!t]
\centering
\caption{\rev{Complete identified parameter set. Every entry is either
estimated from NHANES 2015--2016 or declared as an explicit prior that is
propagated through the global sensitivity analysis; no value is chosen for
illustration only. This table makes the simulation independently
reproducible.}}
\label{tab:model}
{\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{|@{\,}l@{\,}|c|p{3.6cm}|}
\hline
\textbf{Parameter} & \textbf{Value} & \textbf{Provenance} \\ \hline
""" + rows + r"""\end{tabular}}
\end{table}"""
    return _w("tab_model", s)


def tab_endpoints(ident):
    """Sensitivity coefficients against four metabolic endpoints."""
    m = ident["multi"]
    lab = {"HOMA_B": r"HOMA-B",
           "HOMA_B_adj": r"HOMA-B\,$\mid$\,IR",
           "DI": r"HOMA-B/IR",
           "HOMA_IR": r"HOMA-IR"}
    sym = {"Zn": r"$\alpha$", "Ca": r"$\gamma$", "Mg": r"$\beta$"}
    # alpha = Zn, beta = Mg, gamma = Ca, as in the manuscript notation
    rows = ""
    for ep in ["HOMA_B", "HOMA_B_adj", "DI", "HOMA_IR"]:
        sub = m[m.endpoint == ep]
        first = True
        for _, r in sub.iterrows():
            head = (lab[ep] + fr" ($R^2$={r.r2:.2f})") if first else ""
            star = r"$^{*}$" if r.p < 0.05 else ""
            rows += (fr"{head} & {sym[r.ion]} & "
                     fr"{r.coef:+.3f}{star} & [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] & "
                     fr"{r.p:.3f} \\" + "\n")
            first = False
        rows += r"\hline" + "\n"
    s = r"""\begin{table}[!t]
\centering
\caption{\revc{Ionic sensitivity coefficients estimated on NHANES against four
metabolic endpoints ($n=548$ adults without diagnosed diabetes, adjusted for
age, sex and the income-to-poverty ratio; coefficients are per reference
standard deviation of each ion). The decomposition matters: HOMA-B alone
confounds secretory capacity with compensatory hyperinsulinaemia, whereas the
adjusted and disposition-index forms isolate secretion and HOMA-IR isolates
insulin resistance. $^{*}p<0.05$.}}
\label{tab:endpoints}
{\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{|@{\,}l@{\,}|c|c|c|c|}
\hline
\textbf{Endpoint (log)} & & \textbf{Est.} & \textbf{95\% CI} &
\textbf{$p$} \\ \hline
""" + rows + r"""\end{tabular}}
\end{table}"""
    return _w("tab_endpoints", s)


def tab_physio(eff_all, eff_low, eff_di, eff_b):
    """Population-mean physiological effect with parameter-uncertainty CIs."""
    def block(df, title):
        out = fr"\multicolumn{{4}}{{|l|}}{{\emph{{{title}}}}} \\ \hline" + "\n"
        for _, r in df.iterrows():
            star = r"$^{*}$" if r.p_two_sided < 0.05 else ""
            nm = esc(r.Strategy).replace(" (no adaptation)", "")
            out += (fr"\quad {nm} & {r['mean']:+.2f}{star} & "
                    fr"[{r.ci_lo:+.2f}, {r.ci_hi:+.2f}] & "
                    fr"{r.p_two_sided:.3f} \\ \hline" + "\n")
        return out
    s = r"""\begin{table}[!t]
\centering
\caption{\revc{Population-mean predicted change in metabolic endpoints (\%),
with confidence intervals constructed from the sampling distribution of the
estimated coupling vector (full $3\times3$ covariance, 4000 draws), holding
the simulated trajectories fixed. The low-magnesium tertile is the
pre-specified subgroup that an ionic intervention would target; in the whole
cohort, virtual patients scatter on both sides of the set-point, so corrections
in opposite directions largely cancel. $^{*}p<0.05$.}}
\label{tab:physio}
{\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{|@{\,}l@{\,}|c|c|c|}
\hline
\textbf{Strategy} & \textbf{Mean (\%)} & \textbf{95\% CI} & \textbf{$p$} \\ \hline
""" + block(eff_all, "Insulin resistance (HOMA-IR), whole cohort") \
    + block(eff_low, "Insulin resistance (HOMA-IR), lowest Mg tertile") \
    + block(eff_di, "Disposition index (secretory capacity), lowest Mg tertile") \
    + r"""\end{tabular}}
\end{table}"""
    return _w("tab_physio", s)


def tab_learning(ident):
    t = ident["learning"]["table"]
    rows = ""
    for _, r in t.iterrows():
        cv = "--" if not np.isfinite(r.cv_r2) else f"{r.cv_r2:+.3f} ({r.cv_sd:.3f})"
        name = esc(r.model).replace("MLP, permuted labels (control)",
                                    "MLP, permuted (control)")
        rows += (fr"{name} & {cv} & {r.test_r2:+.3f} & "
                 fr"{r.test_rmse:.3f} & {r.spearman:+.3f} \\ \hline" + "\n")
    s = r"""\begin{table}[!t]
\centering
\caption{\rev{Held-out evaluation of the learned metal--insulin model
$\hat f$. Target: $\log$ HOMA-B; training partition 70\%, untouched test
partition 30\%, stratified on the HOMA-B quartile; model selection by 5-fold
cross-validation inside the training partition only.}}
\label{tab:learning}
{\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{|@{\,}l@{\,}|c|c|c|c|}
\hline
\textbf{Model} & \textbf{CV $R^2$ (SD)} & \textbf{Test $R^2$} &
\textbf{RMSE} & \textbf{$\rho_s$} \\ \hline
""" + rows + r"""\end{tabular}}
\vspace{1mm}
\parbox{\linewidth}{\scriptsize \rev{$\rho_s$: Spearman correlation on the
test partition. The permuted-label row is the no-information floor. Mean
absolute error follows RMSE monotonically and is reported in the released
result files.}}
\end{table}"""
    return _w("tab_learning", s)


def tab_results(summ, stats_sse, stats_eta):
    rows = ""
    for _, r in summ.iterrows():
        rows += (fr"{esc(r.Strategy)} & {r.SSE_Zn:.3f} ({r.SSE_Zn_sd:.3f}) & "
                 fr"{r.SSE_Ca:.4f} ({r.SSE_Ca_sd:.4f}) & "
                 fr"{r.Overshoot:.1f} & {r.Settle:.0f} & "
                 fr"{r.Effort:.2f} & {r.Eta:.2f} ({r.Eta_sd:.2f}) & "
                 fr"{r.SSE_Mg:.2f} ({r.SSE_Mg_sd:.2f}) \\ \hline" + "\n")
    an = stats_sse["anova"]
    s = r"""\begin{table*}[!t]
\centering
\caption{\rev{Out-of-sample closed-loop performance on """ + str(N_PATIENTS) + r""" virtual patients
resampled from the held-out NHANES partition, """ + str(N_SEEDS) + r""" Monte-Carlo replicates each.
Values are mean (SD) across patients; the unit of analysis is the patient.
$\eta$ is the attainment of the reference HOMA-B level (100\% = the
secretory capacity predicted at $\mathbf{M}_{\mathrm{ref}}$).
\revc{Physiological effects are reported separately in
Table~\ref{tab:physio}, because the control-level and the physiological
questions have different answers and should not be read from one column.}}}
\label{tab5}
{\footnotesize
\begin{tabular}{|p{3.4cm}|c|c|c|c|c|c|c|}
\hline
\textbf{Strategy} & \textbf{SSE Zn} & \textbf{SSE Ca} &
\textbf{Overshoot} & \textbf{$t_s$} & \textbf{Effort} &
\textbf{$\eta$ (\%)} & \revc{\textbf{SSE Mg}} \\
 & ($\mu$g/dL) & (mg/dL) & (\%) & (min) & & & \revc{(mg/day)} \\ \hline
""" + rows + r"""\end{tabular}}
\vspace{1mm}
\parbox{\linewidth}{\footnotesize \rev{Repeated-measures ANOVA on SSE$_{Zn}$:
$F(""" + f"{an['df1']},{an['df2']}" + r""")=""" + f"{an['F']:.2f}" + r"""$,
$p""" + (r"<10^{-6}" if an['p'] < 1e-6 else f"={an['p']:.3g}") + r"""$,
partial $\eta^2=""" + f"{an['eta2_partial']:.3f}" + r"""$
(Greenhouse--Geisser $\epsilon=""" + f"{an['gg_epsilon']:.2f}" + r"""$).
The corresponding test on the physiological endpoint $\eta$ gives
$F=""" + f"{stats_eta['anova']['F']:.2f}" + r"""$, partial
$\eta^2=""" + f"{stats_eta['anova']['eta2_partial']:.3f}" + r"""$.}}
\end{table*}"""
    return _w("tab_results", s)


def tab_stats(st):
    rows = ""
    agree = True
    for _, r in st["posthoc"].iterrows():
        lab = r.comparison.split(" vs ")[-1]
        lab = lab.replace(" (no adaptation)", "").replace(" (upper bound)", "")
        p = r.p_t_holm
        pw = r.p_wilcoxon_holm
        agree &= ((p < 0.05) == (pw < 0.05))
        ptxt = "$<10^{-6}$" if p < 1e-6 else f"{p:.3g}"
        rows += (fr"vs {esc(lab)} & {r.mean_diff:+.3f} & "
                 fr"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] & {r.cohens_dz:+.2f} & "
                 fr"{ptxt} \\ \hline" + "\n")
    note = ("All paired Wilcoxon signed-rank tests agreed with the parametric "
            "conclusion after Holm correction." if agree else
            "Parametric and non-parametric conclusions differ for at least one "
            "contrast; see the released result files.")
    s = r"""\begin{table}[!t]
\centering
\caption{\rev{Post-hoc paired comparisons of the steady-state Zn error,
each against the proposed strategy. Negative differences favour the proposed
strategy.}}
\label{tab:stats}
{\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{|@{}l@{}|c|c|c|c@{}|}
\hline
\textbf{Contrast} & \textbf{$\Delta$} & \textbf{95\% CI} & \textbf{$d_z$} &
\textbf{$p_{\mathrm{Holm}}$} \\ \hline
""" + rows + r"""\end{tabular}}
\vspace{1mm}
\parbox{\linewidth}{\scriptsize \rev{$\Delta$ in standardised units;
$d_z$ is the paired effect size; CIs are 2000-resample bootstrap percentile
intervals. """ + note + r"""}}
\end{table}"""
    return _w("tab_stats", s)


def tab_sensitivity(sob, gsw, ssw):
    rows = ""
    for _, r in sob.sort_values("ST_sse", ascending=False).iterrows():
        rows += (fr"{esc(r.parameter)} & {r.S1_sse:+.3f} & {r.ST_sse:.3f} & "
                 fr"{r.ST_eta:.3f} \\ \hline" + "\n")
    e0 = gsw.loc[np.isclose(gsw.delta_gain, 0.0), "eta"].values[0]
    dmin = 100 * (gsw.eta.min() - e0) / e0
    dmax = 100 * (gsw.eta.max() - e0) / e0
    s0 = ssw.loc[ssw.snr_db == 30.0, "eta"].values[0]
    s20 = ssw.loc[ssw.snr_db == 20.0, "eta"].values[0]
    s10 = ssw.loc[ssw.snr_db == 10.0, "eta"].values[0]
    note = (fr"Actuator gain perturbed by $\pm20\%$: $\eta$ varies by "
            fr"[{dmin:+.2f}\%, {dmax:+.2f}\%] relative to nominal. "
            fr"Reducing the nanosensor SNR from 30~dB to 20~dB changes $\eta$ by "
            fr"{100*(s20-s0)/s0:+.2f}\%, and to 10~dB by {100*(s10-s0)/s0:+.2f}\%.")
    s = r"""\begin{table}[!t]
\centering
\caption{\rev{Variance-based global sensitivity (Saltelli estimator,
first-order $S_1$ and total-order $S_T$) of the steady-state Zn error and of
the physiological endpoint $\eta$ with respect to every uncertain parameter,
including the unmeasured Mg coupling. Base sample $N=96$, so
$(k+2)N=1152$ model evaluations. Because $\mathrm{Var}(\eta)$ is small, the
first-order estimator for $\eta$ is dominated by Monte-Carlo noise and only
the total-order index is reported for that output; residual estimator noise
of order $\pm0.05$ should be assumed throughout.}}
\label{tab:sens}
{\footnotesize
\begin{tabular}{|l|c|c|c|}
\hline
\textbf{Parameter} & \textbf{$S_1$ (SSE)} & \textbf{$S_T$ (SSE)} &
\textbf{$S_T$ ($\eta$)} \\ \hline
""" + rows + r"""\end{tabular}}
\vspace{1mm}
\parbox{\linewidth}{\footnotesize \rev{""" + note + r"""}}
\end{table}"""
    return _w("tab_sensitivity", s)


def tab_diagnostic(dtab):
    rows = ""
    for _, r in dtab.iterrows():
        if not np.isfinite(r.auc):
            rows += (fr"{esc(r.task)} & {esc(r.panel)} & {r.n} & {r.n_pos} & "
                     fr"\multicolumn{{5}}{{c|}}{{{esc(r.note)}}} \\ \hline" + "\n")
            continue
        rows += (fr"{esc(r.task)} & {esc(r.panel)} & {r.n} & {r.n_pos} & "
                 fr"{r.auc:.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}] & "
                 fr"{100*r.sensitivity:.1f} & {100*r.specificity:.1f} & "
                 fr"{100*r.ppv:.1f} & {100*r.npv:.1f} \\ \hline" + "\n")
    s = r"""\begin{table*}[!t]
\centering
\caption{\rev{Diagnostic evaluation on the NHANES analytic cohort with an
explicit reference standard, 10-fold cross-validated logistic models and
operating points at the Youden index. D1: diabetes vs normoglycaemia;
D2: dysglycaemia vs normoglycaemia; D3: impaired $\beta$-cell secretory
capacity (lowest HOMA-B quartile, predictors exclude glucose and insulin);
D4: insulin-treated early-onset diabetes, the standard NHANES surrogate for
type 1 diabetes.}}
\label{tab:diag}
{\footnotesize
\begin{tabular}{|c|p{3.4cm}|c|c|c|c|c|c|c|}
\hline
\textbf{Task} & \textbf{Predictor panel} & \textbf{n} & \textbf{cases} &
\textbf{AUC [95\% CI]} & \textbf{Sens.} & \textbf{Spec.} &
\textbf{PPV} & \textbf{NPV} \\ \hline
""" + rows + r"""\end{tabular}}
\end{table*}"""
    return _w("tab_diagnostic", s)


def tab_controllers(pi_gains):
    kp, ki, kd = pi_gains
    s = r"""\begin{table}[!t]
\centering
\caption{\rev{Executable specification of every compared strategy. All
strategies share the same plant, the same on-chip first-order sensor
conditioning, the same actuator saturation $|u|\le""" + f"{0.6}" + r"""$ SD/min and the
same rate limit, so that differences are attributable to the control law
alone.}}
\label{tab:ctrl}
{\scriptsize
\setlength{\tabcolsep}{2pt}
\begin{tabular}{|@{\,}p{1.75cm}@{\,}|@{\,}p{5.5cm}@{\,}|}
\hline
\textbf{Strategy} & \textbf{Control law and parameters} \\ \hline
Glucose-only & Insulin delivery driven by the glycaemic error only; issues no
ionic actuation, $\mathbf{u}\equiv\mathbf{0}$. \\ \hline
Metal supplementation & Open loop. Periodic bolus, period 30~min, width
5~min, population-level dose sized on the mean ionic deficit of the cohort;
Ca is not supplemented; no feedback. \\ \hline
Fixed-gain PID & $u=-(k_p e + k_i\!\int\! e\,dt + k_d \dot e)$ with
$k_p=""" + f"{kp:.3f}" + r"""$, $k_i=""" + f"{ki:.3f}" + r"""$,
$k_d=""" + f"{kd:.3f}" + r"""$ obtained by ITAE grid search on the nominal
model subject to an overshoot constraint of 10\%; derivative action computed
on the filtered measurement with filter coefficient $N=10$;
conditional-integration anti-windup; gains frozen thereafter. \\ \hline
Proposed adaptive AI & Recursive least squares ($\lambda=0.999$, $P_0=0.05I$)
identifying $z_{k+1}=a z_k + b u_k + c$ per ion, initialised at the nominal
model; certainty-equivalence scalar DARE re-solved every 10~min;
identified-offset feedforward $u_{ff}=-c/b$; persistent-excitation dither
(0.02~SD) during the first 60~min; set-point supplied by $\hat f$ trained on
the NHANES training partition only. \\ \hline
Model-based LQR (ablation) & Same cost, saturation and rate limit, nominal
plant matrices, no online identification. Isolates the contribution of
adaptation. \\ \hline
\end{tabular}}
\end{table}"""
    return _w("tab_controllers", s)
