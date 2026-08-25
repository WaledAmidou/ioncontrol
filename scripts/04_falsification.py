#!/usr/bin/env python
"""
Reproduce the falsification sequence in full.

This is the script a reviewer should run. It takes about a minute, needs no
arguments, and prints the three stages that successively removed the apparent
physiological signal, followed by a verdict table.

    python scripts/04_falsification.py

Everything it prints is also written to results/csv/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ioncontrol import cohorts, falsification as fx           # noqa: E402
from ioncontrol.config import RES                             # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)
RULE = "=" * 78


def banner(txt):
    print(f"\n{RULE}\n{txt}\n{RULE}")


def main():
    banner("STAGE 0  The two cycles cannot be merged")
    print("SEQN is a within-cycle identifier. Reviewers sometimes ask why the\n"
          "cycles were not pooled to obtain serum Zn and serum Mg together.\n")
    n_overlap = cohorts.seqn_overlap("I", "L")
    assert n_overlap == 0, "unexpected SEQN overlap"
    print("\nNo NHANES cycle measures all three ions in serum:")
    print("  cycle I (2015-2016): serum Zn, serum Ca, NO serum Mg")
    print("  cycle L (2021-2023): serum Mg, serum Ca, NO serum Zn "
          "(no CUSEZN_L file exists)")

    *_, a_I, dm_I, ref_I, flow_I = (None,) + cohorts.build("I")[1:]
    *_, a_L, dm_L, ref_L, flow_L = (None,) + cohorts.build("L")[1:]
    print(f"\nprimary analytic cohorts: I n={len(a_I)}, L n={len(a_L)}")

    # ------------------------------------------------------------------
    banner("STAGE 1  Endpoint decomposition (cycle I, dietary Mg, unadjusted)")
    print("HOMA-B alone confounds beta-cell secretory capacity with\n"
          "compensatory hyperinsulinaemia. Fitting four endpoints separates\n"
          "them: HOMA-B|HOMA-IR and the disposition index isolate secretion,\n"
          "HOMA-IR isolates insulin resistance.\n")
    t1, cov1 = fx.stage1_endpoint_decomposition(a_I)
    print(fx.summarise(t1).round(4).to_string(index=False))
    print("\nRead: the magnesium coefficient on HOMA-B is negative and\n"
          "significant, which taken alone would say magnesium harms the\n"
          "beta-cell. It does not: the association is with insulin\n"
          "resistance, and it vanishes on the secretion-specific endpoints.")

    # ------------------------------------------------------------------
    banner("STAGE 2  Is dietary magnesium a valid proxy for magnesium status?")
    print("Cycle L measures BOTH dietary and serum magnesium in the same\n"
          "participants, so the surrogacy assumption behind cycle I can be\n"
          "tested rather than asserted.\n")
    prox = fx.stage2_proxy_validation(a_L)
    print(prox.round(4).to_string(index=False))
    lam = float(prox.loc[0, "lambda_"])
    print(f"\nDietary intake explains {prox.loc[0, 'r2_pct']:.1f}% of the "
          f"variance in serum magnesium (lambda = {lam:.3f}).")
    print("A regression-calibration correction would divide by lambda:")
    print(f"  beta_corrected = -0.081 / {lam:.3f} = {-0.081 / lam:+.2f} per SD,")
    print("  i.e. a ~50% change in HOMA-IR per SD. Not credible. Dividing by\n"
          "  0.1 amplifies every bias tenfold along with the signal.")

    print("\n--- 2b: are the two magnesium measures the same exposure? ---\n")
    t2b = fx.stage2b_independence(a_L)
    print(fx.summarise(t2b).round(4).to_string(index=False))
    print("\nThe dietary coefficient stays large CONDITIONAL on serum\n"
          "magnesium, and the two measures correlate at r ~ 0.11. They are\n"
          "not one exposure measured twice. The classical measurement-error\n"
          "correction is therefore invalid in principle, not merely\n"
          "imprecise -- and the dietary variable must be relabelled as a\n"
          "dietary-pattern marker, not a magnesium-status marker.")

    # ------------------------------------------------------------------
    banner("STAGE 3  Adiposity adjustment")
    print("BMI is the dominant confounder of HOMA-IR and is associated with\n"
          "both magnesium measures. Same models, with and without.\n")
    t3_I = fx.stage3_adiposity(a_I, "I", ("Zn", "Ca", "Mg"))
    t3_L = fx.stage3_adiposity(a_L, "L", ("Ca", "Mg"))
    t3 = pd.concat([t3_I, t3_L], ignore_index=True)
    key = t3[t3.endpoint.isin(["HOMA_IR", "DI"])]
    print(fx.summarise(key).round(4).to_string(index=False))

    # ------------------------------------------------------------------
    banner("VERDICT")
    piv = (t3[t3.term.isin(["Zn", "Ca", "Mg"])
              & t3.endpoint.isin(["HOMA_IR", "DI"])]
           .pivot_table(index=["cycle", "endpoint", "term"],
                        columns="adiposity_adjusted",
                        values=["coef", "p"]))
    print(piv.round(4).to_string())
    surv = t3[(t3.adiposity_adjusted) & (t3.p < 0.05)
              & t3.endpoint.isin(["HOMA_IR", "DI"])]
    print("\nIonic coefficients surviving adiposity adjustment at p < 0.05:")
    if len(surv) == 0:
        print("  none")
    else:
        print(fx.summarise(surv).round(4).to_string(index=False))
    print("""
Every magnesium association loses significance once adiposity is accounted
for, in both cycles, on both endpoints, with both exposure measures. The
disposition-index signal -- the endpoint the framework actually targets --
does not survive either.

Adjustment also UNMASKS a calcium association of the opposite sign to the
one the framework assumes: higher serum calcium goes with higher insulin
resistance. It is reported for completeness and labelled exploratory: it
emerged after three prior analyses and would not survive a correction for
the multiplicity of everything tested here.

Conclusion: NHANES does not support a robust association between serum
ionic status and beta-cell function. The control problem posed in the
manuscript is solved rigorously; its physiological premise is not
supported by the data used to parameterise it.""")

    # ------------------------------------------------------------------
    t1.to_csv(RES / "falsification_stage1_endpoints.csv", index=False)
    prox.to_csv(RES / "falsification_stage2_proxy.csv", index=False)
    t2b.to_csv(RES / "falsification_stage2b_independence.csv", index=False)
    t3.to_csv(RES / "falsification_stage3_adiposity.csv", index=False)
    print(f"\nTables written to {RES}")


if __name__ == "__main__":
    main()
