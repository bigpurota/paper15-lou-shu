"""Fast known-answer simulations through the real pipeline (task.md 4.4) and placebo checks.

Small N and T, one replication, a large premium (3%/month per s.d.) so that the expected
sign and significance are stable; the full-size study is `python -I src/simulate.py`.
"""
from pathlib import Path

import numpy as np
import pytest
import yaml

from analysis import build_panel
from placebo import lag_panel, run_placebos, shuffle_signals
from simulate import _rng, simulate_daily, run_world

CFG = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml").read_text())
FAST = dict(premium=0.03, n_assets=150, n_months=48)


def test_daily_format_and_realism():
    d = simulate_daily("P", CFG, _rng(CFG, "P", 0, None), n_assets=20, n_months=6)
    assert list(d.columns) == ["symbol", "date", "close", "quote_volume"]
    assert (d["close"] > 0).all() and (d["quote_volume"] > 0).all()
    r = d.groupby("symbol")["close"].pct_change().dropna()
    assert 0.02 < r.std() < 0.08                       # crypto-like daily volatility


@pytest.mark.parametrize("rep", [0])
def test_world_V_prices_lnA_C_not_ResLnA(rep):
    s = run_world("V", CFG, rep, **FAST)
    assert s["fm2_lnA_C_coef"] > 0 and s["fm2_lnA_C_t"] > 2
    assert s["fm4_lnA_C_t"] > 2
    assert abs(s["fm3_ResLnA_t"]) < 2.5                 # unpriced (loose bound: one draw)
    assert s["AC_q75_q25"] > 10 and s["corr_A_AC"] > 0.8


@pytest.mark.parametrize("rep", [0])
def test_world_P_prices_ResLnA_not_lnA_C(rep):
    s = run_world("P", CFG, rep, **FAST)
    assert s["fm3_ResLnA_coef"] > 0 and s["fm3_ResLnA_t"] > 2
    assert s["fm4_ResLnA_t"] > 2
    assert abs(s["fm2_lnA_C_t"]) < 2.5


def test_world_0_runs():
    s = run_world("0", CFG, 0, **FAST)
    assert s["premium"] == 0.0
    assert all(np.isfinite(s[f"{k}_t"]) for k in ["fm2_lnA_C", "fm3_ResLnA", "hl_A", "hl_A_C", "hl_ResA"])


def test_placebos_kill_premium_and_run_leak():
    d = simulate_daily("V", CFG, _rng(CFG, "V", 7, 0.03), premium=0.03, n_assets=150, n_months=48)
    m = d["date"].dt.to_period("M")
    start, end = str(m.min() + 14), str(m.max())
    panel, factors = build_panel(d, CFG, start, end)
    sh = shuffle_signals(panel, np.random.default_rng(0))
    # block shuffle: each month holds the same signal ROWS (joint structure kept), reassigned to coins
    cols = ["lnA", "lnA_C", "ResLnA"]
    for (_, a), (_, b) in zip(panel.groupby("month"), sh.groupby("month")):
        ka = a[cols].round(12).fillna(-999).sort_values(cols).to_numpy()
        kb = b[cols].round(12).fillna(-999).sort_values(cols).to_numpy()
        assert np.array_equal(ka, kb)
    assert (sh["ret_adj"].fillna(0) == panel["ret_adj"].fillna(0)).all()
    assert (sh["lnA_C"] != panel["lnA_C"]).mean() > 0.9
    pl = run_placebos(panel, factors, CFG, seed=1, panel_lag0=lag_panel(d, CFG, start, end, 0), n_perm=5)
    assert set(pl["test"]) == {"baseline (lag 2)", "shuffle within month", "leak (lag 0)"}
    base = pl[(pl["test"] == "baseline (lag 2)") & (pl["var"] == "FM(2) lnA_C")].iloc[0]
    shuf = pl[(pl["test"] == "shuffle within month") & (pl["var"] == "FM(2) lnA_C")].iloc[0]
    assert base["t"] > 2
    assert abs(shuf["coef"]) < 0.5 * abs(base["coef"]) and abs(shuf["t"]) < 2
