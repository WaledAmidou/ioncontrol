#!/usr/bin/env python
"""
Variance-based global sensitivity (Saltelli/Sobol) plus local sweeps of the
actuator gain and the nanosensor SNR.

Resumable: each block of the Saltelli design is cached, so the script can be
run repeatedly with a time budget until it prints DONE.

    python scripts/05_sensitivity.py 300     # 300-second budget per call
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ioncontrol import cohorts, identification as idn, sensitivity, simulate
from ioncontrol.config import RES, IONS, SEED

CACHE = Path(__file__).resolve().parents[1] / ".sobol_cache"
CACHE.mkdir(exist_ok=True)
N_BASE, N_PAT, N_SEED = 64, 40, 3

def main(budget_s=300.0):
    import time
    t0 = time.time()
    _, _, a, dm, ref, _ = cohorts.build("I", export=False)
    ident = idn.identify_all(a, ref)
    tb = ident["beta"]["table"].set_index("term")
    ident["beta"]["beta_full"] = np.array(
        [tb.loc[f"beta_{i}", "coef"] for i in IONS])
    z_h, *_ = simulate.build_cohort(a, ident, n_patients=N_PAT,
                                    n_seeds=N_SEED,
                                    held_out_idx=ident["learning"]["idx_test"])
    rng = np.random.default_rng(SEED)
    k = len(sensitivity.PARAMS)
    lo = np.array([p[1] for p in sensitivity.PARAMS])
    hi = np.array([p[2] for p in sensitivity.PARAMS])
    A_ = lo + (hi - lo) * rng.random((N_BASE, k))
    B_ = lo + (hi - lo) * rng.random((N_BASE, k))
    jobs = [("A", A_), ("B", B_)]
    for i in range(k):
        AB = A_.copy(); AB[:, i] = B_[:, i]; jobs.append((f"AB{i}", AB))
    for tag, M in jobs:
        f = CACHE / f"{tag}.npy"
        if f.exists():
            continue
        if time.time() - t0 > budget_s:
            left = sum(1 for t, _ in jobs if not (CACHE / f"{t}.npy").exists())
            print(f"PAUSED - {left} blocks remaining, rerun to continue")
            return False
        np.save(f, np.array([sensitivity._one_run(t, a, ident, z_h, N_SEED)
                             for t in M]))
        print(f"  done {tag} ({time.time()-t0:.0f}s)")
    fA, fB = np.load(CACHE / "A.npy"), np.load(CACHE / "B.npy")
    var = fA.var(axis=0)
    S1 = np.zeros((k, 2)); ST = np.zeros((k, 2))
    for i in range(k):
        fAB = np.load(CACHE / f"AB{i}.npy")
        S1[i] = np.mean(fB * (fAB - fA), axis=0) / var
        ST[i] = 0.5 * np.mean((fA - fAB) ** 2, axis=0) / var
    sob = pd.DataFrame(dict(parameter=[p[0] for p in sensitivity.PARAMS],
                            S1_sse=S1[:, 0], ST_sse=ST[:, 0],
                            S1_eta=S1[:, 1], ST_eta=ST[:, 1]))
    for c in ("S1_sse", "ST_sse", "S1_eta", "ST_eta"):
        sob[c] = sob[c].clip(0, 1)
    sob.to_csv(RES / "sobol_indices.csv", index=False)
    gsw = sensitivity.gain_sweep(a, ident, n_patients=60, n_seeds=5)
    ssw = sensitivity.snr_sweep(a, ident, n_patients=60, n_seeds=5)
    gsw.to_csv(RES / "gain_sweep.csv", index=False)
    ssw.to_csv(RES / "snr_sweep.csv", index=False)
    print(sob.round(3).to_string(index=False)); print("DONE")
    return True

if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 300.0)
