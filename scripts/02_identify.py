#!/usr/bin/env python
"""
Identify every model parameter from cycle I and report the held-out
evaluation of the learned insulin model.

Set-points come from the metabolically healthy reference subgroup; the
diffusion matrix W is DERIVED from the empirical ionic covariance through the
stationary Lyapunov equation rather than assumed; the learned model is
benchmarked against linear baselines and a label-permutation control on an
untouched test partition.
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ioncontrol import cohorts, identification as idn, tables
from ioncontrol.config import IONS, RES

if __name__ == "__main__":
    _, _, a, dm, ref, flow = cohorts.build("I", export=False)
    ident = idn.identify_all(a, ref)
    sp = ident["setpoints"]
    print("set-points (median of the healthy reference subgroup):")
    for i in IONS:
        print(f"  {i:3s} {sp[i]['target']:9.2f}  "
              f"95% CI [{sp[i]['ci'][0]:.2f}, {sp[i]['ci'][1]:.2f}]  "
              f"SD {sp[i]['sd']:.2f}   <- {sp[i]['source']}")
    print("\nSigma (NHANES inter-individual covariance):\n",
          np.round(ident["Sigma"], 5))
    print("\nA = -diag(kappa), kappa =", np.round(ident["kappa"], 5))
    print("W = -(A.Sigma + Sigma.A^T)  [Lyapunov-derived, not assumed]:\n",
          np.round(ident["W"], 5))
    print("\nheld-out evaluation of the learned insulin model:")
    print(ident["learning"]["table"].round(4).to_string(index=False))
    ident["learning"]["table"].to_csv(RES / "learned_model_heldout.csv",
                                      index=False)
    tables.tab_setpoints(sp); tables.tab_model(ident); tables.tab_learning(ident)
    print(f"\nLaTeX tables written; CSV in {RES}")
