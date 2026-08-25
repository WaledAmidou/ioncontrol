"""
Smoke tests. They check that the pipeline is internally consistent and that
the falsification conclusion is reproduced, not that the numbers are pretty.

    pytest -q
"""
import sys
from pathlib import Path

import numpy as np

try:
    import pytest
except ModuleNotFoundError:          # allows tests/run_tests.py to work
    pytest = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ioncontrol import cohorts, falsification as fx, controllers, simulate
from ioncontrol.config import DATA, IONS

HAVE_DATA = (DATA / "DEMO_I.xpt").exists() and (DATA / "BIOPRO_L.xpt").exists()


def needs_data(fn):
    """Skip when the survey files are absent, with or without pytest."""
    fn._needs_data = True
    if pytest is None:
        return fn
    return pytest.mark.skipif(
        not HAVE_DATA,
        reason="NHANES .xpt files absent; see data/README.md")(fn)


# -- unit tests that need no data ---------------------------------------
def test_scalar_dare_is_stabilising():
    a = np.full((5, 3), 0.999)
    b = np.full((5, 3), 0.1)
    k, p = controllers.scalar_dare(a, b, np.ones((5, 3)), 2.0)
    assert np.all(np.abs(a - b * k) < 1.0), "closed loop must be stable"
    assert np.all(p > 0)


def test_saturation_and_rate_limit():
    c = controllers.GlucoseOnly(4, 0.1, u_max=0.6)
    u = c.finalise(np.full((4, 3), 10.0))
    assert np.all(np.abs(u) <= 0.6 + 1e-9)


def test_ols_matches_numpy():
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(200), rng.normal(size=(200, 2))])
    beta = np.array([1.0, -0.5, 0.25])
    y = X @ beta + rng.normal(scale=0.1, size=200)
    tab, cov, r2 = fx.ols(y, X, ["a", "b", "c"])
    assert np.allclose(tab.coef.values, beta, atol=0.05)
    assert r2 > 0.9


# -- tests that need the survey files -----------------------------------
@needs_data
def test_cycles_share_no_participant():
    assert cohorts.seqn_overlap("I", "L") == 0


@needs_data
def test_cycle_specific_ion_availability():
    _, der_I, a_I, *_ = cohorts.build("I", export=False)
    _, der_L, a_L, *_ = cohorts.build("L", export=False)
    assert a_I["Zn"].notna().all(), "cycle I must have serum zinc"
    assert der_L["Zn"].isna().all(), "cycle L has no CUSEZN file"
    assert der_L["Mg_serum"].notna().any(), "cycle L must have serum magnesium"
    assert der_I["Mg_serum"].isna().all(), "cycle I has no serum magnesium"


@needs_data
def test_proxy_is_weak():
    """The dietary proxy must be shown to be weak; this is a load-bearing
    result, so it is asserted rather than merely reported."""
    _, _, a_L, *_ = cohorts.build("L", export=False)
    prox = fx.stage2_proxy_validation(a_L, n_boot=200)
    lam = float(prox.loc[0, "lambda_"])
    assert 0.0 < lam < 0.25, f"unexpected attenuation factor {lam}"
    assert float(prox.loc[0, "r2_pct"]) < 5.0


@needs_data
def test_magnesium_does_not_survive_adiposity_adjustment():
    """The central claim of the paper's falsification section."""
    _, _, a_L, *_ = cohorts.build("L", export=False)
    t = fx.stage3_adiposity(a_L, "L", ("Ca", "Mg"))
    adj = t[(t.adiposity_adjusted) & (t.term == "Mg")
            & (t.endpoint.isin(["HOMA_IR", "DI"]))]
    assert (adj.p > 0.05).all(), "magnesium unexpectedly survived adjustment"
    una = t[(~t.adiposity_adjusted) & (t.term == "Mg")
            & (t.endpoint == "HOMA_IR")]
    assert (una.p < 0.05).all(), "unadjusted association should be significant"


@needs_data
def test_control_level_result_is_robust():
    """Ionic regulation works; only the physiological claim fails."""
    from ioncontrol import identification as idn
    _, _, a, dm, ref, _ = cohorts.build("I", export=False)
    ident = idn.identify_all(a, ref)
    tb = ident["beta"]["table"].set_index("term")
    ident["beta"]["beta_full"] = np.array(
        [tb.loc[f"beta_{i}", "coef"] for i in IONS])
    res = simulate.run_all_strategies(a, ident, fault_time=120.0)
    s = simulate.summary_table(res).set_index("Strategy")
    assert (s.loc["Proposed adaptive AI", "SSE_Zn"]
            < 0.1 * s.loc["Glucose-only", "SSE_Zn"])
