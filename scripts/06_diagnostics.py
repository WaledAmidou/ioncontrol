#!/usr/bin/env python
"""
Diagnostic evaluation with explicit reference standards.

The manuscript originally advertised a diagnostic strategy without a cohort,
a reference standard, a screening task or an accuracy measure. All four are
supplied here, and the result is negative.
"""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ioncontrol import cohorts, diagnostics
from ioncontrol.config import RES

if __name__ == "__main__":
    _, _, a, dm, ref, _ = cohorts.build("I", export=False)
    tab, curves = diagnostics.evaluate(a)
    pd.set_option("display.width", 200)
    print(tab[["task", "panel", "n", "n_pos", "auc", "ci_lo", "ci_hi",
               "sensitivity", "specificity"]].round(3).to_string(index=False))
    tab.to_csv(RES / "diagnostic_accuracy.csv", index=False)
    print("\nThe regulated ions do not separate from chance, and the trace-"
          "metal\npanels add nothing over a demographic baseline.")
