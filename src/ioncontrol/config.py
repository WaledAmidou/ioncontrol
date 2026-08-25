"""
Global configuration.

Every quantity a reviewer might want to change lives here: paths, the random
seed, the discretisation, the controller weights, and the definition of the
reference subgroup. Nothing is configured anywhere else.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------
# Paths.  Override with the IONCONTROL_DATA / IONCONTROL_OUT environment
# variables rather than editing this file.
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("IONCONTROL_DATA", ROOT / "data"))
OUT = Path(os.environ.get("IONCONTROL_OUT", ROOT / "results"))
UPLOADS = DATA                      # backward-compatible alias
FIG, TAB, RES = OUT / "figures", OUT / "tables", OUT / "csv"
for _p in (OUT, FIG, TAB, RES):
    _p.mkdir(parents=True, exist_ok=True)

SEED = 20260820
RNG = np.random.default_rng(SEED)

# ----------------------------------------------------------------------
# NHANES cycles
# ----------------------------------------------------------------------
# Cycle I = 2015-2016, cycle L = 2021-2023.  They are analysed separately and
# are NEVER merged: SEQN is a within-cycle identifier and the two cycles
# contain entirely different people (see cohorts.seqn_overlap).
#
# Neither cycle measures all three ions in serum:
#   cycle I : serum Zn, serum Ca, no serum Mg  -> dietary Mg used as a proxy
#   cycle L : serum Mg, serum Ca, no serum Zn  (no CUSEZN_L file exists)
CYCLES = {
    "I": dict(
        label="NHANES 2015-2016",
        suffix="_I",
        files=["DEMO", "BIOPRO", "CUSEZN", "GLU", "INS", "DIQ",
               "DR1TOT", "DR2TOT", "BMX"],
        zn="LBXSZN", ca="LBXSCA", mg=None,
        mg_source="dietary recalls DR1TMAGN/DR2TMAGN (mg/day), proxy",
        seqn_range=(83732, 93702),
    ),
    "L": dict(
        label="NHANES 2021-2023",
        suffix="_L",
        files=["DEMO", "BIOPRO", "GLU", "INS", "DIQ",
               "DR1TOT", "DR2TOT", "BMX"],
        zn=None,
        ca="LBXSCA", mg="LBXMAGN",
        mg_source="serum magnesium LBXMAGN (mg/dL)",
        seqn_range=(130378, 142310),
    ),
}

ZERO_IS_MISSING = ["LBXSCA", "LBXSZN", "LBXSCU", "LBXSSE", "LBXGLU", "LBXIN",
                   "LBXSPH", "LBXSKSI", "LBXSAL", "LBXSCR", "LBXMAGN"]

# ----------------------------------------------------------------------
# State vector
# ----------------------------------------------------------------------
IONS = ["Zn", "Ca", "Mg"]
ION_UNITS = {"Zn": r"$\mu$g/dL", "Ca": "mg/dL", "Mg": "mg/day"}
ION_VARS = {"Zn": "LBXSZN", "Ca": "LBXSCA", "Mg": "MG"}

# Retained for the archived cycle-I-with-prior configuration only; the
# current analysis measures magnesium and does not use these.
MG_PRIOR_MEAN, MG_PRIOR_SD = 0.85, 0.075
MG_CORR_PRIOR = {"Zn": 0.10, "Ca": 0.15}

# Literature-informed homeostatic half-lives (min), swept +/-50 % in the
# Sobol analysis.  These are the only structural parameters not identified
# from data: cross-sectional surveys cannot identify dynamics.
HALF_LIFE_MIN = {"Zn": 90.0, "Ca": 30.0, "Mg": 120.0}

# ----------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------
DT = 0.1
T_END = 240.0
N_PATIENTS = 200
N_SEEDS = 30
SNR_DB = 30.0
U_MAX_SIGMA = 0.60
DU_MAX_SIGMA = 0.15
FILT_TAU = 0.1
RHO = 2.0            # smallest R keeping nominal overshoot below 10 %

STRATEGIES = [
    "Glucose-only",
    "Metal supplementation",
    "Fixed-gain PID",
    "Proposed adaptive AI",
    "Model-based LQR (no adaptation)",
]

# ----------------------------------------------------------------------
# Cohorts and models
# ----------------------------------------------------------------------
AGE_MIN = 20
REF_COHORT = dict(fpg_max=100.0, homa_ir_max=2.5, age_range=(20, 65))
COVARS = ["age", "male", "INDFMPIR"]
ADIPOSITY = "log_BMI"
ENDPOINTS = ["HOMA_B", "HOMA_B_adj", "DI", "HOMA_IR"]
