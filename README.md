# ioncontrol — identification, simulation and falsification of an AI-enhanced nanosensor framework for pancreatic metal-ion regulation

Reproducibility package for the manuscript *Falsifying Cyber-Physiological Control
Concepts: A Reproducible Framework Applied to Metal-Ion Regulation in Type 1
Diabetes*.

Everything the manuscript reports is produced by this code from the public
NHANES survey files. No figure, table or number in the paper is hand-entered.

**The headline result is a negative one, and reproducing it is the point of
this repository.** The control architecture works; the physiological premise
it rests on does not survive standard epidemiological adjustment. Run
`scripts/04_falsification.py` — it takes about a minute — and you will see the
whole sequence.

---

## Quick start

```bash
pip install -r requirements.txt
python scripts/download_data.py      # public NHANES files, ~120 MB
python scripts/04_falsification.py   # the central result, ~60 s
python scripts/run_all.py            # everything else, ~10 min
pytest -q                            # 7 tests, including the falsification
```

Data location defaults to `data/` and results to `results/`; override with
`IONCONTROL_DATA` and `IONCONTROL_OUT`.

---

## What the analysis found

### Control level — robust

On 200 virtual patients resampled from NHANES, with 30 Monte-Carlo replicates
each, inter-patient plant heterogeneity and a 50 % actuator failure at
mid-experiment:

| Strategy | SSE Zn (µg/dL) | Overshoot | Settling (min) |
|---|---|---|---|
| Glucose-only | 13.17 | — | not reached |
| Metal supplementation | 13.29 | — | 227 |
| Fixed-gain PID | 0.292 | 8.1 % | 66 |
| **Proposed adaptive AI** | **0.314** | **4.8 %** | 76 |
| Model-based LQR (ablation) | 0.524 | 5.8 % | 96 |

Repeated-measures ANOVA on the steady-state error: F(4, 796) = 1050,
partial η² = 0.84. This part of the paper stands.

### Physiological level — does not survive

| Model | Endpoint | Subgroup | Effect | 95 % CI | p |
|---|---|---|---|---|---|
| unadjusted | HOMA-IR | lowest Mg tertile | −3.19 % | [−5.75, −0.82] | 0.006 |
| unadjusted | disposition index | lowest Mg tertile | +1.76 % | [+0.34, +3.10] | 0.013 |
| **BMI-adjusted** | HOMA-IR | lowest Mg tertile | −0.40 % | [−2.43, +1.54] | 0.70 |
| **BMI-adjusted** | disposition index | lowest Mg tertile | +0.85 % | [−0.54, +2.17] | 0.22 |

---

## The falsification sequence

Three checks were applied in order. Each is routine. Each removed an
association that had looked convincing at the previous stage.

**Stage 1 — endpoint decomposition.** HOMA-B alone confounds β-cell secretory
capacity with compensatory hyperinsulinaemia: in participants without
diabetes, a high HOMA-B usually means insulin resistance, not a healthier
β-cell. The magnesium coefficient on log HOMA-B is −0.065 (p = 0.025), which
read naively says magnesium harms the β-cell. Fitting four endpoints shows the
association is with insulin resistance and vanishes on the secretion-specific
endpoints.

**Stage 2 — proxy validation.** Cycle I has no serum magnesium, so dietary
intake was used as a proxy. Cycle L measures both in the same people, so the
surrogacy assumption can be tested:

| Proxy variant | n | λ | R² |
|---|---|---|---|
| Mean of available recalls | 2 386 | 0.124 | 1.5 % |
| Two-day recalls only | 2 138 | 0.139 | 1.9 % |
| Energy-adjusted density | 2 388 | 0.076 | 0.6 % |

Dietary intake explains about 1 % of the variance in serum magnesium. A
regression-calibration correction would divide by λ = 0.124, giving −0.65 per
SD — a ~50 % change in HOMA-IR per SD, which is not credible. Worse, the
correction is invalid *in principle*: with both variables in the same model
the dietary coefficient stays large conditional on serum magnesium
(−0.140, p < 10⁻⁴ versus −0.060, p = 0.001), and the two correlate at r = 0.11.
They are not one exposure measured twice. The dietary variable is a
dietary-pattern marker, not a magnesium-status marker.

**Stage 3 — adiposity adjustment.** BMI is the dominant confounder of HOMA-IR
and is associated with both magnesium measures.

| Cycle | Endpoint | Unadjusted | + log BMI |
|---|---|---|---|
| I (dietary Mg) | log HOMA-IR | −0.081 (p = 0.018) | −0.031 (p = 0.26) |
| I | log DI | +0.016 (p = 0.47) | +0.004 (p = 0.85) |
| L (serum Mg) | log HOMA-IR | −0.042 (p = 0.011) | +0.011 (p = 0.41) |
| L | log DI | +0.023 (p = 0.017) | +0.007 (p = 0.45) |

Every magnesium association loses significance, in both cycles, on both
endpoints, with both exposure measures. Waist circumference gives the same
answer.

Adjustment also **unmasks** a calcium association of the opposite sign to the
one the framework assumes — higher serum calcium with higher insulin
resistance (cycle L: +0.054, p = 0.0001; disposition index −0.021, p = 0.028).
It is reported and labelled exploratory: it emerged after three prior
analyses and would not survive a correction for the multiplicity of everything
tested here.

---

## Why the two survey cycles are never merged

`scripts/01_cohorts.py` prints the proof:

```
cycle I: SEQN  83 732 –  93 702 (n =  9 971)
cycle L: SEQN 130 378 – 142 310 (n = 11 933)
overlapping identifiers: 0
```

`SEQN` is a within-cycle sequence number; the cycles contain entirely
different people. The empty intersection is also a safety net — had the ranges
overlapped, a naive merge would have paired unrelated participants silently.

This matters because **no NHANES cycle measures all three ions in serum**:

| | serum Zn | serum Ca | serum Mg |
|---|---|---|---|
| 2015–2016 (I) | ✓ | ✓ | ✗ (dietary proxy) |
| 2021–2023 (L) | ✗ (no `CUSEZN_L`) | ✓ | ✓ (`LBXMAGN`) |

The closed-loop analysis therefore uses a **two-sample coupling vector**:
α_Zn from cycle I, γ_Ca and β_Mg from cycle L, in standardised units, with a
block-diagonal covariance. Transportability across cycles is an assumption and
is declared as one.

---

## Repository layout

```
src/ioncontrol/
  config.py          all parameters, paths, seed — the single source of truth
  cohorts.py         unified .XPT ingestion for both cycles, cleaning, export
  identification.py  set-points, Lyapunov-derived W, learned insulin model
  falsification.py   the three-stage sequence above
  controllers.py     the five control laws, fully specified
  simulate.py        virtual patients, plant heterogeneity, actuator fault
  stats_tests.py     repeated-measures ANOVA, Holm-corrected post-hoc
  sensitivity.py     Saltelli/Sobol, gain and SNR sweeps
  diagnostics.py     four screening tasks with explicit reference standards
  figures.py         all figures
  tables.py          all LaTeX tables

scripts/
  download_data.py   fetches the public files, writes a SHA-256 manifest
  01_cohorts.py      cohort flow, SEQN non-overlap proof
  02_identify.py     model identification and held-out evaluation
  03_simulate.py     closed loop, control level and physiological level
  04_falsification.py  ** the central result **
  05_sensitivity.py  resumable Sobol analysis
  06_diagnostics.py  ROC/AUC with confidence intervals
  run_all.py         everything in order

tests/test_smoke.py  7 tests; the falsification conclusion is asserted, not
                     merely reported
```

---

## Design decisions a reviewer may want to check

**The `DIQ010 == 1` filter is secondary, not primary.** Restricting to
diagnosed diabetes gives 117 complete participants in cycle I and no
comparison group: diagnostic accuracy is not computable without controls, and
coefficients estimated in treated diabetics describe treatment as much as
physiology. The primary sample is all non-pregnant adults ≥ 20 y with complete
core variables. Both are built and exported by `cohorts.build()`.

**W is derived, not assumed.** Modelling the ionic state as an
Ornstein–Uhlenbeck process, the diffusion matrix is obtained from the
stationary Lyapunov equation `AΣ + ΣAᵀ + W = 0` with Σ set to the empirical
inter-individual covariance. The simulator therefore reproduces the ionic
dispersion of a national sample by construction.

**The relaxation rates are literature priors, and that is unavoidable.**
Cross-sectional surveys cannot identify dynamics. The rates are swept ±50 % in
the Sobol analysis, and `config.HALF_LIFE_MIN` is the only structural
parameter not taken from data.

**Determinism.** Strategy-level random streams are keyed by a CRC of the
strategy name rather than Python's `hash()`, which is salted per process.
Results are identical across machines and runs.

---

## Limitations we state ourselves

- **Serum, not intracellular.** The controller regulates an intracellular
  state; the data are serum. That mapping is an assumption and is not fixable
  with survey data of any size.
- **Cross-sectional.** No causal direction is established, and reverse
  causation is plausible: insulin resistance alters renal magnesium handling
  and dietary patterns alike.
- **Mediation versus confounding.** If magnesium acts on insulin resistance
  *through* adiposity, BMI is a mediator and adjusting for it is wrong. Both
  estimates are reported side by side; cross-sectional data cannot settle it.
- **Unweighted.** The analytic samples are intersections of subsamples for
  which no single official survey weight exists.
- **Multiplicity.** Three ions × four endpoints × two cycles × two adjustment
  sets were examined. The surviving calcium result would not withstand a
  correction for that search.

---

## Data availability

NHANES is public and is not redistributed here. `scripts/download_data.py`
fetches the files from NCHS and writes `data/MANIFEST.sha256`, which lets a
reviewer confirm they analysed the same bytes.

## Licence and citation

Code released under the MIT licence (`LICENSE`). If you use it, please cite
the manuscript and this repository (`CITATION.cff`).
