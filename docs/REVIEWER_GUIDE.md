# Reviewer guide

You have limited time. Here is the shortest path to checking each claim.

## If you check only one thing

```bash
pip install -r requirements.txt
python scripts/download_data.py
python scripts/04_falsification.py
```

About a minute after the download. It prints the three-stage sequence and the
verdict table, and writes them to `results/csv/`.

## Claim-by-claim

| Claim in the manuscript | Where to check | Runtime |
|---|---|---|
| The cycles cannot be pooled | `scripts/01_cohorts.py`, first three lines | 20 s |
| Cohort flow numbers | `scripts/01_cohorts.py` | 20 s |
| Set-points are data-derived, not illustrative | `scripts/02_identify.py` | 30 s |
| W is Lyapunov-derived, not assumed | `scripts/02_identify.py`, prints A, Σ, W | 30 s |
| The learned model is properly held out | `scripts/02_identify.py`, last table | 30 s |
| Control-level performance | `scripts/03_simulate.py`, first table | 100 s |
| Physiological effect does not survive | `scripts/03_simulate.py`, second table | 100 s |
| Endpoint decomposition (stage 1) | `scripts/04_falsification.py` | 60 s |
| Proxy validation (stage 2) | `scripts/04_falsification.py` | 60 s |
| Adiposity adjustment (stage 3) | `scripts/04_falsification.py` | 60 s |
| Diagnostic claim is unsupported | `scripts/06_diagnostics.py` | 40 s |
| Robustness / Sobol indices | `scripts/05_sensitivity.py 300`, repeat | ~20 min |

## Adversarial checks we suggest

**Is the negative result an artefact of over-adjustment?** Plausible, and we
say so. Set `ADIPOSITY = "waist"` in `config.py` and rerun stage 3: the
conclusion is unchanged. If magnesium acts on insulin resistance *through*
adiposity, BMI is a mediator and the unadjusted estimate is the right one;
cross-sectional data cannot settle this, and both estimates are reported.

**Is the control result flattered by a weak baseline?** The PID comparator is
tuned by explicit ITAE grid search under the same 10 % overshoot constraint
that fixes the LQR weight (`simulate.tune_pid`). The ablation — same cost,
same limits, nominal plant, no online identification — isolates what
adaptation contributes.

**Is anything hard-coded?** `grep -rn "0\.0" src/ioncontrol/config.py` shows
every tunable in one file. No analysis module contains a magic number.

**Does the seed matter?** Change `config.SEED` and rerun. Strategy-level
streams are keyed by a CRC of the strategy name, so runs are reproducible
across processes and machines, but the substantive conclusions do not depend
on the particular seed.

## What we could not do, and would want done

- Serum-to-intracellular calibration. Nothing in NHANES bears on it.
- A cycle with serum Zn, Ca and Mg together. None exists.
- Longitudinal identification of the ionic dynamics. The relaxation rates
  remain literature priors and are swept ±50 %.
- A properly powered type 1 diabetes cohort. Cycle I contains 3 eligible
  participants, cycle L contains 8.
