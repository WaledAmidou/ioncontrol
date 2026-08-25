"""
Unified NHANES ingestion and cohort construction for cycles I and L.

The same code path serves both cycles; the only differences are declared in
`config.CYCLES`. Everything downstream works on the harmonised column names
Zn / Ca / Mg / FPG / INS / BMI, so no analysis module needs to know which
cycle it is looking at.

Processing chain
----------------
  1. read every `.XPT` with `pd.read_sas()`
  2. merge on SEQN, WITHIN a cycle only
  3. magnesium: serum LBXMAGN (cycle L) or the mean of the two 24-h dietary
     recalls (cycle I), the latter kept only when the recall status variable
     indicates a reliable recall and total energy is non-zero
  4. clean: impossible zeros, 3-IQR Tukey fence on the log scale
  5. derive HOMA-IR, HOMA-B, the disposition index, and phenotype flags
  6. export the analytic frames to CSV
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (CYCLES, ZERO_IS_MISSING, AGE_MIN, REF_COHORT, DATA, RES)

CORE = ["Ca", "Mg", "FPG", "INS"]          # Zn added for cycle I


# ----------------------------------------------------------------------
def _read(cycle: str, name: str, cols=None) -> pd.DataFrame:
    path = DATA / f"{name}{CYCLES[cycle]['suffix']}.xpt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. See data/README.md for the download URLs.")
    d = pd.read_sas(path)
    return d[[c for c in cols if c in d.columns]] if cols else d


def load_raw(cycle: str) -> pd.DataFrame:
    """Merge every file of one cycle on SEQN. Never merges across cycles."""
    spec = CYCLES[cycle]
    demo = _read(cycle, "DEMO", ["SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH3",
                                 "RIDEXPRG", "INDFMPIR", "WTMEC2YR",
                                 "SDMVPSU", "SDMVSTRA"])
    frames = [
        _read(cycle, "BIOPRO"),
        _read(cycle, "GLU", ["SEQN", "LBXGLU", "WTSAF2YR"]),
        _read(cycle, "INS", ["SEQN", "LBXIN"]),
        _read(cycle, "DIQ", ["SEQN", "DIQ010", "DID040", "DIQ050", "DIQ070"]),
        _read(cycle, "DR1TOT", ["SEQN", "DR1TMAGN", "DR1TKCAL", "DR1TCALC",
                                "DR1TZINC", "DR1DRSTZ", "WTDRD1"]),
        _read(cycle, "DR2TOT", ["SEQN", "DR2TMAGN", "DR2TKCAL", "DR2TCALC",
                                "DR2TZINC", "DR2DRSTZ"]),
        _read(cycle, "BMX", ["SEQN", "BMXBMI", "BMXWAIST", "BMXWT", "BMXHT"]),
    ]
    if "CUSEZN" in spec["files"]:
        frames.insert(1, _read(cycle, "CUSEZN", ["SEQN", "LBXSZN", "LBXSCU",
                                                 "LBXSSE"]))
    df = demo
    for f in frames:
        df = df.merge(f, on="SEQN", how="left")
    return df


def seqn_overlap(cycle_a="I", cycle_b="L") -> int:
    """
    Empirical proof that the cycles share no participant.

    Reviewers occasionally ask why the cycles were not pooled to obtain serum
    Zn and serum Mg together. They cannot be: SEQN is a within-cycle sequence
    number. The empty intersection is also a safety net -- had the numeric
    ranges overlapped, a naive merge would have paired unrelated people
    silently instead of returning nothing.
    """
    a = _read(cycle_a, "DEMO", ["SEQN"])["SEQN"].values
    b = _read(cycle_b, "DEMO", ["SEQN"])["SEQN"].values
    n = len(np.intersect1d(a, b))
    print(f"cycle {cycle_a}: SEQN {a.min():.0f}-{a.max():.0f} (n={len(a)})")
    print(f"cycle {cycle_b}: SEQN {b.min():.0f}-{b.max():.0f} (n={len(b)})")
    print(f"overlapping identifiers: {n}")
    return n


# ----------------------------------------------------------------------
def clean(df: pd.DataFrame, cycle: str) -> pd.DataFrame:
    d = df.copy()
    for c in ZERO_IS_MISSING:
        if c in d.columns:
            d.loc[d[c] == 0, c] = np.nan

    for k in (1, 2):
        st, mg, kc = f"DR{k}DRSTZ", f"DR{k}TMAGN", f"DR{k}TKCAL"
        if st in d.columns:
            d.loc[(d[st] != 1) | (d[kc].fillna(0) <= 0), mg] = np.nan

    d["Mg_diet"] = d[["DR1TMAGN", "DR2TMAGN"]].mean(axis=1, skipna=True)
    d["Mg_ndays"] = d[["DR1TMAGN", "DR2TMAGN"]].notna().sum(axis=1)
    d["KCAL"] = d[["DR1TKCAL", "DR2TKCAL"]].mean(axis=1, skipna=True)
    d["Mg_dens"] = 1000.0 * d["Mg_diet"] / d["KCAL"]

    mgv = CYCLES[cycle]["mg"]
    d["Mg_serum"] = d[mgv] if mgv and mgv in d.columns else np.nan
    # the harmonised magnesium column: serum where available, else dietary
    d["Mg"] = d["Mg_serum"] if mgv else d["Mg_diet"]

    znv = CYCLES[cycle]["zn"]
    d["Zn"] = d[znv] if znv and znv in d.columns else np.nan
    d["Ca"] = d[CYCLES[cycle]["ca"]]

    for c in ("Mg_diet", "Mg", "LBXIN", "Zn", "LBXSCU"):
        if c in d.columns and d[c].notna().any():
            x = np.log(d[c].where(d[c] > 0))
            q1, q3 = x.quantile([0.25, 0.75])
            iqr = q3 - q1
            d.loc[(x < q1 - 3 * iqr) | (x > q3 + 3 * iqr), c] = np.nan
    return d


def derive(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["age"] = d["RIDAGEYR"]
    d["male"] = (d["RIAGENDR"] == 1).astype(float)
    d["Cu"] = d.get("LBXSCU", np.nan)
    d["Se"] = d.get("LBXSSE", np.nan)
    d["CuZn"] = d["Cu"] / d["Zn"]
    d["FPG"] = d["LBXGLU"]
    d["INS"] = d["LBXIN"]
    d["BMI"] = d.get("BMXBMI", np.nan)
    d["waist"] = d.get("BMXWAIST", np.nan)
    d["log_BMI"] = np.log(d["BMI"].where(d["BMI"] > 0))

    d["HOMA_IR"] = d["FPG"] * d["INS"] / 405.0
    d["HOMA_B"] = np.where(d["FPG"] > 63,
                           360.0 * d["INS"] / (d["FPG"] - 63.0), np.nan)
    d["DI"] = d["HOMA_B"] / d["HOMA_IR"]

    dx = d["DIQ010"] == 1
    d["dm_dx"] = dx.astype(float)
    d["diabetes"] = (dx | (d["FPG"] >= 126)).astype(float)
    d["dysglycaemia"] = (dx | (d["FPG"] >= 100)).astype(float)
    d["normoglycaemic"] = ((d["DIQ010"] == 2) & (d["FPG"] < 100)).astype(float)

    age_dx = d["DID040"].replace({666: 0.5, 777: np.nan, 999: np.nan})
    d["t1d_like"] = (dx & (age_dx < 30) & (d["DIQ050"] == 1)
                     & (d["DIQ070"] == 2)).astype(float)
    return d


# ----------------------------------------------------------------------
def _core(cycle):
    return (["Zn"] if CYCLES[cycle]["zn"] else []) + CORE


def analytic_cohort(d, cycle):
    m = (d["age"] >= AGE_MIN) & (d["RIDEXPRG"] != 1)
    return d[m].dropna(subset=_core(cycle)).reset_index(drop=True)


def diabetic_subgroup(d, cycle):
    """DIQ010 == 1. Reported as a secondary analysis: it has no control group,
    so no diagnostic accuracy is computable, and coefficients estimated in
    treated diabetics describe treatment as much as physiology."""
    return d[d["DIQ010"] == 1].dropna(subset=_core(cycle)).reset_index(drop=True)


def reference_cohort(a):
    lo, hi = REF_COHORT["age_range"]
    return a[(a["FPG"] < REF_COHORT["fpg_max"]) & (a["DIQ010"] == 2)
             & (a["HOMA_IR"] < REF_COHORT["homa_ir_max"])
             & a["age"].between(lo, hi)].copy()


def cohort_flow(cycle, raw, der, a, dm, ref):
    f = {f"{CYCLES[cycle]['label']} participants (DEMO)": int(len(raw))}
    if CYCLES[cycle]["zn"]:
        f["With serum Zn"] = int(der["Zn"].notna().sum())
    f.update({
        "With serum total Ca": int(der["Ca"].notna().sum()),
        f"With magnesium ({CYCLES[cycle]['mg_source']})":
            int(der["Mg"].notna().sum()),
        "With fasting glucose and insulin":
            int((der["FPG"].notna() & der["INS"].notna()).sum()),
        "With BMI": int(der["BMI"].notna().sum()),
        "Primary analytic cohort": int(len(a)),
        "  normoglycaemic": int(a["normoglycaemic"].sum()),
        "  dysglycaemic": int(a["dysglycaemia"].sum()),
        "  diabetes": int(a["diabetes"].sum()),
        "Secondary: diagnosed diabetes (DIQ010==1)": int(len(dm)),
        "  insulin-treated early-onset (T1D surrogate)":
            int(dm["t1d_like"].sum()) if len(dm) else 0,
        "Metabolically healthy reference subgroup": int(len(ref)),
    })
    return f


def build(cycle="I", export=True):
    raw = load_raw(cycle)
    der = derive(clean(raw, cycle))
    a = analytic_cohort(der, cycle)
    dm = diabetic_subgroup(der, cycle)
    ref = reference_cohort(a)
    flow = cohort_flow(cycle, raw, der, a, dm, ref)
    if export:
        cols = [c for c in
                ["SEQN", "age", "male", "INDFMPIR", "BMI", "waist", "log_BMI",
                 "Zn", "Ca", "Mg", "Mg_serum", "Mg_diet", "Mg_dens",
                 "Mg_ndays", "KCAL", "Cu", "Se", "CuZn", "FPG", "INS",
                 "HOMA_IR", "HOMA_B", "DI", "dm_dx", "diabetes",
                 "dysglycaemia", "normoglycaemic", "t1d_like"] if c in a.columns]
        a[cols].to_csv(RES / f"cohort_{cycle}_analytic.csv", index=False)
        dm[cols].to_csv(RES / f"cohort_{cycle}_diabetic.csv", index=False)
        ref[cols].to_csv(RES / f"cohort_{cycle}_reference.csv", index=False)
        pd.Series(flow).to_csv(RES / f"cohort_{cycle}_flow.csv")
    return raw, der, a, dm, ref, flow


if __name__ == "__main__":
    seqn_overlap()
    for cyc in ("I", "L"):
        print(f"\n===== {CYCLES[cyc]['label']} =====")
        *_, flow = build(cyc)
        for k, v in flow.items():
            print(f"  {k:<60s} {v}")
