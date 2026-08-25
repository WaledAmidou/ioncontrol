"""All figures added to the revised manuscript."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_curve

from .config import FIG, IONS, ION_UNITS, N_SEEDS

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 160, "savefig.bbox": "tight", "axes.grid": True,
    "grid.alpha": 0.25, "axes.axisbelow": True, "lines.linewidth": 1.3,
})
COL = {"Glucose-only": "#8c8c8c", "Metal supplementation": "#d1913c",
       "Fixed-gain PID": "#3d7ab8", "Proposed adaptive AI": "#c0392b",
       "Model-based LQR (no adaptation)": "#4f9e6a"}


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)
    return FIG / f"{name}.pdf"


# ----------------------------------------------------------------------
def fig_cohort(a, ref, sp, name="fig_nhanes_cohort"):
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.4))
    panels = [("Zn", r"Serum Zn ($\mu$g/dL)"), ("Ca", "Serum Ca (mg/dL)"),
              ("Mg", "Dietary Mg (mg/day)"), ("Se", r"Serum Se ($\mu$g/L)")]
    g0 = a[a["normoglycaemic"] == 1]
    g1 = a[a["dysglycaemia"] == 1]
    g2 = a[a["diabetes"] == 1]
    for ax, (v, lab) in zip(axes, panels):
        dat = [g0[v].dropna(), g1[v].dropna(), g2[v].dropna()]
        bp = ax.violinplot(dat, showmedians=True, widths=0.8)
        for b, c in zip(bp["bodies"], ["#5b8db8", "#d1913c", "#c0392b"]):
            b.set_facecolor(c); b.set_alpha(0.55)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["Normo", "Dysgly", "DM"], rotation=0)
        ax.set_ylabel(lab)
        p = stats.mannwhitneyu(dat[0], dat[2]).pvalue
        d = ((dat[2].mean() - dat[0].mean())
             / np.sqrt((dat[0].var() + dat[2].var()) / 2))
        ax.set_title(f"$p$={p:.3g}, $d$={d:+.2f}")
        if v in sp:
            ax.axhline(sp[v]["target"], color="k", ls="--", lw=0.9)
    fig.tight_layout()
    _save(fig, name)
    return name


def fig_identification(ident, a, name="fig_identification"):
    beta = ident["beta"]["table"]
    lrn = ident["learning"]
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.4))

    ax = axes[0]
    sub = beta[beta.term.isin(["beta_Zn", "beta_Ca", "age", "male", "INDFMPIR"])]
    ypos = np.arange(len(sub))
    ax.errorbar(sub.coef, ypos,
                xerr=[sub.coef - sub.ci_lo, sub.ci_hi - sub.coef],
                fmt="o", color="#c0392b", ms=3.5, capsize=2)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels([r"$\beta_{Zn}$", r"$\beta_{Ca}$", "age", "sex", "PIR"])
    ax.set_xlabel(r"$\Delta \log$ HOMA-B per SD")
    ax.set_title(f"Sensitivity coefficients\n(n={ident['beta']['n']})")

    ax = axes[1]
    t = lrn["table"].dropna(subset=["cv_r2"])
    x = np.arange(len(t))
    ax.bar(x - 0.2, t.cv_r2, 0.4, label="5-fold CV (train)", color="#5b8db8")
    ax.bar(x + 0.2, t.test_r2, 0.4, label="held-out test", color="#c0392b")
    perm = lrn["table"].iloc[-1]["test_r2"]
    ax.axhline(perm, color="k", ls=":", lw=1, label="permuted labels")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([s.split(" (")[0] for s in t.model], rotation=30, ha="right")
    ax.set_ylabel(r"$R^2$")
    ax.legend(frameon=False, loc="lower left")
    ax.set_title("Learned insulin model")

    ax = axes[2]
    best = lrn["models"][lrn["best"]]
    pred = best.predict(lrn["Xte"])
    ax.plot(lrn["yte"], pred, ".", ms=2.5, color="#3d7ab8", alpha=0.7)
    lims = [min(lrn["yte"].min(), pred.min()), max(lrn["yte"].max(), pred.max())]
    ax.plot(lims, lims, "k--", lw=0.9)
    r2 = float(lrn["table"].set_index("model").loc[lrn["best"], "test_r2"])
    ax.set_xlabel("observed log HOMA-B")
    ax.set_ylabel("predicted")
    ax.set_title(f"{lrn['best']}\nheld-out $R^2$={r2:.3f}")

    ax = axes[3]
    S = ident["Sigma"]
    R = S / np.sqrt(np.outer(np.diag(S), np.diag(S)))
    im = ax.imshow(R, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(3)); ax.set_xticklabels(IONS)
    ax.set_yticks(range(3)); ax.set_yticklabels(IONS)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{R[i, j]:.2f}", ha="center", va="center", fontsize=6.5)
    ax.set_title("NHANES ionic correlation\n(used in the Lyapunov calibration)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    _save(fig, name)
    return name


def fig_closedloop(out, ident, name="fig_closedloop"):
    res = out["results"]
    sd = np.array([ident["setpoints"][i]["sd"] for i in IONS])
    tgt = np.array([ident["setpoints"][i]["target"] for i in IONS])
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.2))

    # (a) Zn trajectory, cohort mean +/- IQR
    ax = axes[0, 0]
    for s, r in res.items():
        tr = r["traj"][:, :, 0] * sd[0] + tgt[0]
        ax.plot(r["tgrid"], tr.mean(axis=1), color=COL[s], label=s)
    ax.axhline(tgt[0], color="k", ls="--", lw=0.9)
    ax.axvline(120, color="k", ls=":", lw=0.9)
    ax.set_xlabel("time (min)"); ax.set_ylabel(r"[Zn$^{2+}$] ($\mu$g/dL)")
    ax.set_title("(a) Ionic regulation (Zn)")
    ax.legend(frameon=False, fontsize=5.4, loc="upper left",
              ncol=1, handlelength=1.2, borderaxespad=0.2)

    # (b) Ca trajectory
    ax = axes[0, 1]
    for s, r in res.items():
        tr = r["traj"][:, :, 1] * sd[1] + tgt[1]
        ax.plot(r["tgrid"], tr.mean(axis=1), color=COL[s])
    ax.axhline(tgt[1], color="k", ls="--", lw=0.9)
    ax.axvline(120, color="k", ls=":", lw=0.9)
    ax.set_xlabel("time (min)"); ax.set_ylabel(r"[Ca$^{2+}$] (mg/dL)")
    ax.set_title("(b) Ionic regulation (Ca)")

    # (c) steady-state error per ion
    ax = axes[0, 2]
    labs = list(res.keys())
    x = np.arange(3)
    w = 0.15
    for i, s in enumerate(labs):
        m = res[s]["sse_phys"].mean(axis=0)
        e = res[s]["sse_phys"].std(axis=0) / np.sqrt(res[s]["sse_phys"].shape[0])
        ax.bar(x + (i - 2) * w, m, w, yerr=e, color=COL[s], label=s, capsize=1.5)
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels([f"{i}\n({ION_UNITS[i]})" for i in IONS])
    ax.set_ylabel("steady-state error")
    ax.set_title("(c) Steady-state error")

    # (d) insulin attainment
    ax = axes[1, 0]
    data = [res[s]["eta_pt_ss"] for s in labs]
    bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.6)
    for p, s in zip(bp["boxes"], labs):
        p.set_facecolor(COL[s]); p.set_alpha(0.7)
    ax.set_xticks(range(1, len(labs) + 1))
    ax.set_xticklabels([s.split(" (")[0] for s in labs], rotation=30, ha="right",
                       fontsize=5.6)
    ax.set_ylabel("HOMA-B attainment (%)")
    ax.set_title("(d) Physiological endpoint")

    # (e) overshoot, with settling time annotated
    ax = axes[1, 1]
    xs = np.arange(len(labs))
    ov = [np.nanmean(res[s]["overshoot"]) for s in labs]
    ose = [np.nanstd(res[s]["overshoot"]) / np.sqrt(len(res[s]["overshoot"]))
           for s in labs]
    st = [np.nanmean(res[s]["settle"]) for s in labs]
    ax.bar(xs, ov, 0.6, yerr=ose, capsize=2,
           color=[COL[s] for s in labs])
    for x, o, t in zip(xs, ov, st):
        lab_t = "not reached" if not np.isfinite(t) else f"$t_s$={t:.0f} min"
        ax.text(x, o + 0.6, lab_t, ha="center", fontsize=5.2)
    ax.set_xticks(xs)
    ax.set_xticklabels([s.split(" (")[0] for s in labs], rotation=30, ha="right",
                       fontsize=5.6)
    ax.set_ylabel("overshoot (%)")
    ax.set_ylim(0, max(ov) * 1.45)
    ax.set_title("(e) Transient quality")

    # (f) effort vs variance trade-off
    ax = axes[1, 2]
    for s in labs:
        ax.scatter(res[s]["effort"].mean(), res[s]["eta_var"].mean(),
                   s=38, color=COL[s], label=s)
    ax.set_xlabel(r"control effort $\int\|u\|^2 dt$")
    ax.set_ylabel("insulin-attainment variance")
    ax.set_title("(f) Effort / stability trade-off")
    ax.legend(frameon=False, fontsize=5.2, loc="upper right")
    fig.tight_layout()
    _save(fig, name)
    return name


def fig_sensitivity(sob, gsw, ssw, name="fig_sensitivity"):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.2))
    ax = axes[0]
    o = sob.sort_values("ST_sse")
    y = np.arange(len(o))
    ax.barh(y - 0.2, o.S1_sse, 0.4, color="#5b8db8", label="$S_1$")
    ax.barh(y + 0.2, o.ST_sse, 0.4, color="#c0392b", label="$S_T$")
    ax.set_yticks(y); ax.set_yticklabels(o.parameter, fontsize=6)
    ax.set_xlabel("Sobol index (SSE$_{Zn}$)")
    ax.legend(frameon=False)
    ax.set_title("(a) Global sensitivity")

    ax = axes[1]
    ax.plot(100 * gsw.delta_gain, gsw.eta, "o-", color="#c0392b")
    ax.set_xlabel("actuator-gain perturbation (%)")
    ax.set_ylabel("HOMA-B attainment (%)")
    ax.set_title("(b) Actuator-gain robustness")

    ax = axes[2]
    ax.plot(ssw.snr_db, ssw.eta, "o-", color="#3d7ab8")
    ax.invert_xaxis()
    ax.set_xlabel("nanosensor SNR (dB)")
    ax.set_ylabel("HOMA-B attainment (%)")
    ax.set_title("(c) Sensor-noise robustness")
    fig.tight_layout()
    _save(fig, name)
    return name


def fig_diagnostic(tab, curves, name="fig_diagnostic"):
    tasks = ["D1", "D2", "D3"]
    titles = {"D1": "D1: diabetes vs normoglycaemia",
              "D2": "D2: dysglycaemia vs normoglycaemia",
              "D3": "D3: impaired $\\beta$-cell function"}
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.6))
    cols = ["#c0392b", "#d1913c", "#3d7ab8", "#8c8c8c"]
    for ax, t in zip(axes[:3], tasks):
        for (task, panel), c in zip([k for k in curves if k[0] == t], cols):
            y, s = curves[(task, panel)]
            fpr, tpr, _ = roc_curve(y, s)
            auc = tab[(tab.task == t) & (tab.panel == panel)].auc.values[0]
            ax.plot(fpr, tpr, color=c, label=f"{panel.split(' (')[0]} ({auc:.2f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlabel("1 - specificity"); ax.set_ylabel("sensitivity")
        ax.set_title(titles[t], fontsize=7)
        ax.legend(frameon=False, fontsize=5.2, loc="lower right")

    ax = axes[3]
    sub = tab.dropna(subset=["auc"])
    labs = sub.task + "\n" + sub.panel.str.split(" \\(").str[0]
    y = np.arange(len(sub))
    ax.errorbar(sub.auc, y, xerr=[sub.auc - sub.ci_lo, sub.ci_hi - sub.auc],
                fmt="o", ms=3, color="#c0392b", capsize=2)
    ax.axvline(0.5, color="k", ls="--", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=4.2)
    ax.set_xlabel("AUC (95 % CI)")
    ax.set_title("(d) Diagnostic accuracy", fontsize=7)
    fig.tight_layout()
    _save(fig, name)
    return name


def fig_physio(ident, eff_all, eff_low, eff_di, name="fig_physio"):
    """Coefficients by endpoint, and the resulting population-mean effects."""
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    m = ident["multi"]
    order = ["HOMA_B", "HOMA_B_adj", "DI", "HOMA_IR"]
    lab = {"HOMA_B": "HOMA-B", "HOMA_B_adj": "HOMA-B | IR",
           "DI": "HOMA-B/IR", "HOMA_IR": "HOMA-IR"}
    cols = {"Zn": "#c0392b", "Ca": "#3d7ab8", "Mg": "#4f9e6a"}

    ax = axes[0]
    yt, ylab = [], []
    y = 0
    for ep in order:
        for ion in ("Zn", "Ca", "Mg"):
            r = m[(m.endpoint == ep) & (m.ion == ion)].iloc[0]
            ax.errorbar(r.coef, y, xerr=[[r.coef - r.ci_lo], [r.ci_hi - r.coef]],
                        fmt="o", ms=3, color=cols[ion], capsize=1.8)
            yt.append(y); ylab.append(f"{ion} / {lab[ep]}")
            y += 1
        y += 0.6
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(yt); ax.set_yticklabels(ylab, fontsize=4.6)
    ax.set_xlabel(r"$\Delta\log$ endpoint per SD")
    ax.set_title("(a) Coefficients by endpoint", fontsize=7.5)

    for ax, df, ttl in ((axes[1], eff_low, "(b) $\\Delta$HOMA-IR, low-Mg tertile"),
                        (axes[2], eff_di, "(c) $\\Delta$disposition index")):
        yy = np.arange(len(df))
        ax.errorbar(df["mean"], yy,
                    xerr=[df["mean"] - df.ci_lo, df.ci_hi - df["mean"]],
                    fmt="o", ms=3.5, capsize=2,
                    color="#c0392b", ls="none")
        ax.axvline(0, color="k", ls="--", lw=0.9)
        ax.set_yticks(yy)
        ax.set_yticklabels([s.split(" (")[0] for s in df.Strategy], fontsize=5)
        ax.set_xlabel("predicted change (%)")
        ax.set_title(ttl, fontsize=7.5)
    fig.tight_layout()
    _save(fig, name)
    return name
