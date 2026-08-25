# Data

NHANES is public and is **not** redistributed in this repository. Fetch it:

```bash
python scripts/download_data.py          # both cycles
python scripts/download_data.py --list   # just print the URLs
```

This writes `MANIFEST.sha256`, which a reviewer can use to confirm that the
files analysed here are byte-identical to the ones NCHS distributes.

## Files

**Cycle I — NHANES 2015-2016** (`https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/`)

| File | Provides |
|---|---|
| `DEMO_I` | age, sex, income-to-poverty ratio, survey design |
| `BIOPRO_I` | serum total calcium (`LBXSCA`) |
| `CUSEZN_I` | serum zinc (`LBXSZN`), copper, selenium |
| `GLU_I` / `INS_I` | fasting glucose, fasting insulin |
| `DIQ_I` | diabetes questionnaire |
| `DR1TOT_I` / `DR2TOT_I` | dietary magnesium (`DR1TMAGN`, `DR2TMAGN`) |
| `BMX_I` | BMI, waist circumference |

**Cycle L — NHANES 2021-2023**
(`https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/`)

| File | Provides |
|---|---|
| `DEMO_L` | as above |
| `BIOPRO_L` | serum calcium **and serum magnesium (`LBXMAGN`, mg/dL)** |
| `GLU_L` / `INS_L` | fasting glucose, fasting insulin |
| `DIQ_L` | diabetes questionnaire |
| `DR1TOT_L` / `DR2TOT_L` | dietary magnesium |
| `BMX_L` | BMI, waist circumference |

## Two things to know before reading the code

**The serum magnesium variable is `LBXMAGN`, not `LBXSMG`.** Confirm against
`BIOPRO_L.htm` if NCHS revises the file.

**There is no `CUSEZN_L`.** Serum zinc is not measured in 2021-2023; selenium
appears only in whole blood (`PBCD_L`). No NHANES cycle measures serum Zn, Ca
and Mg simultaneously, which is why the analysis uses a two-sample coupling
vector rather than one cycle.

## Why the cycles are never merged

`SEQN` is a within-cycle sequence number. Cycle I spans 83 732–93 702 and
cycle L spans 130 378–142 310; the intersection is empty because the two
cycles enrolled different people. `scripts/01_cohorts.py` asserts this.
