#!/usr/bin/env python
"""Run the whole pipeline end to end (~10 min, Sobol excluded)."""
import subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
STEPS = ["01_cohorts.py", "02_identify.py", "03_simulate.py",
         "04_falsification.py", "06_diagnostics.py"]
if __name__ == "__main__":
    for s in STEPS:
        print(f"\n{'#' * 78}\n### {s}\n{'#' * 78}")
        r = subprocess.run([sys.executable, str(HERE / s)])
        if r.returncode:
            sys.exit(f"{s} failed")
    print("\nAll steps completed. For the Sobol analysis run "
          "scripts/05_sensitivity.py repeatedly until it prints DONE.")
