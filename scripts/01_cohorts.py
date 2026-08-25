#!/usr/bin/env python
"""Build, describe and export the analytic cohorts of both cycles."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ioncontrol import cohorts
from ioncontrol.config import CYCLES

if __name__ == "__main__":
    cohorts.seqn_overlap("I", "L")
    for cyc in ("I", "L"):
        print(f"\n===== {CYCLES[cyc]['label']} =====")
        *_, flow = cohorts.build(cyc, export=True)
        for k, v in flow.items():
            print(f"  {k:<62s} {v}")
