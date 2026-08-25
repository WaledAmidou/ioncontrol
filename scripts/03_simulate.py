#!/usr/bin/env python
"""
Closed-loop evaluation with the falsification-corrected coupling vector.

Two questions are answered separately, because they have different answers:

  CONTROL LEVEL       does the adaptive architecture regulate the ionic state?
                      Yes, decisively, and that result is robust.
  PHYSIOLOGICAL LEVEL does regulating it change a validated metabolic
                      endpoint? Not once adiposity is accounted for.

Reporting these in one table is what produced the over-claims in the earlier
drafts, so this script keeps them apart on purpose.

    python scripts/03_simulate.py [--quick]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ioncontrol import (cohorts, identification as idn, simulate,        # noqa
                        stats_tests, falsification as fx, figures, tables)
from ioncontrol.config import RES, DT, N_SEEDS                            # noqa

pd.set_option("display.width", 220)


def build_coupling(a_I, a_L, adjusted=True):
    """
    Two-sample coupling vector, in standardised units.

    alpha_Zn from cycle I (the only cycle with serum zinc); gamma_Ca and
    beta_Mg from cycle L (both serum, same model, same people). The two blocks
    come from independent samples, so the joint covariance is block diagonal.
    """
    tI, covI, _ = fx.fit_endpoints(a_I, ("Zn", "Ca", "Mg"),
                                   adjust_adiposity=adjusted)
    tI["adiposity_adjusted"] = adjusted
    tL, covL, _ = fx.fit_endpoints(a_L, ("Ca", "Mg"),
                                   adjust_adiposity=adjusted)
    tL["adiposity_adjusted"] = adjusted
    out = {}
    for key, ep in (("beta", "HOMA_B"), ("beta_ir", "HOMA_IR"),
                    ("beta_di", "DI")):
        out[key] = fx.coupling_vector(tI, tL, ep, cov_L=covL[ep],
                                      adiposity_adjusted=adjusted)
    return out, tI, tL


def main(quick=False):
    t0 = time.time()
    _, _, a_I, dm_I, ref_I, flow_I = cohorts.build("I", export=True)
    _, _, a_L, dm_L, ref_L, flow_L = cohorts.build("L", export=True)

    ident = idn.identify_all(a_I, ref_I)
    print("[1] model identified from cycle I "
          f"(set-points, Lyapunov-calibrated W, learned f-hat)")

    rows = []
    for adjusted in (False, True):
        coup, tI, tL = build_coupling(a_I, a_L, adjusted=adjusted)
        for k in ("beta", "beta_ir", "beta_di"):
            ident[k] = {**ident.get(k, {}), **coup[k]}
        ident["beta"]["beta_full"] = coup["beta"]["beta_full"]
        tag = "BMI-adjusted" if adjusted else "unadjusted"
        print(f"[2] coupling ({tag}): "
              f"HOMA-IR {np.round(coup['beta_ir']['beta_full'], 4)}  "
              f"DI {np.round(coup['beta_di']['beta_full'], 4)}")

        res = simulate.run_all_strategies(a_I, ident, fault_time=120.0)
        summ = simulate.summary_table(res)
        low = simulate.low_mg_mask(res)

        if not adjusted:
            control = summ
            st_sse = stats_tests.analyse(res, metric="sse_z", ion=0)
            st_eta = stats_tests.analyse(res, metric="eta_pt_ss")

        for key, lab in (("beta_ir", "HOMA-IR"), ("beta_di", "Disposition index")):
            for sg, sl in ((None, "whole cohort"), (low, "lowest Mg tertile")):
                e = simulate.population_effect(res, ident, key=key, subgroup=sg)
                e["endpoint"], e["subgroup"], e["model"] = lab, sl, tag
                rows.append(e)

    eff = pd.concat(rows, ignore_index=True)

    print("\n" + "=" * 78)
    print("CONTROL LEVEL -- ionic regulation (robust)")
    print("=" * 78)
    print(control[["Strategy", "SSE_Zn", "SSE_Ca", "SSE_Mg", "Overshoot",
                   "Settle", "Effort", "Eta"]].round(3).to_string(index=False))
    an = st_sse["anova"]
    print(f"\nrepeated-measures ANOVA on the steady-state Zn error: "
          f"F({an['df1']},{an['df2']})={an['F']:.1f}, "
          f"partial eta^2={an['eta2_partial']:.3f}")

    print("\n" + "=" * 78)
    print("PHYSIOLOGICAL LEVEL -- predicted change in validated endpoints (%)")
    print("=" * 78)
    prop = eff[eff.Strategy == "Proposed adaptive AI"]
    print(prop[["model", "endpoint", "subgroup", "mean", "ci_lo", "ci_hi",
                "p_two_sided"]].round(3).to_string(index=False))
    print("""
The unadjusted rows are what earlier drafts reported. The BMI-adjusted rows
are what survives. Nothing in the targeted subgroup remains significant.""")

    control.to_csv(RES / "closed_loop_control_level.csv", index=False)
    eff.to_csv(RES / "closed_loop_physiological_level.csv", index=False)
    st_sse["posthoc"].to_csv(RES / "posthoc_sse.csv", index=False)
    json.dump(dict(anova_sse=st_sse["anova"], anova_eta=st_eta["anova"],
                   runtime_s=round(time.time() - t0, 1)),
              open(RES / "simulation_metadata.json", "w"), indent=2)
    print(f"\nwritten to {RES}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(**vars(ap.parse_args()))
