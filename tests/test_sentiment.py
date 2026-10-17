"""Known-answer tests for src/sentiment.py (upgrades 4-5)."""
import numpy as np
import pandas as pd
import pytest

import sentiment as S


def _months(n, start="2019-01"):
    return pd.period_range(start, periods=n, freq="M")


def test_parse_fng_and_monthly():
    days = pd.date_range("2020-01-01", "2020-02-29", freq="D")
    vals = list(range(len(days)))
    raw = {"data": [{"value": str(v % 101), "timestamp": str(int(d.timestamp()))} for d, v in zip(days, vals)]}
    raw["data"].reverse()
    d = S.parse_fng(raw)
    assert d["date"].is_monotonic_increasing and len(d) == len(days)
    m = S.fng_monthly(d)
    jan = m.loc[pd.Period("2020-01", "M")]
    assert jan["sent_mean"] == pytest.approx(np.mean(range(31)))
    assert jan["sent_eom"] == 30 and jan["n_days"] == 31
    assert m.loc[pd.Period("2020-02", "M"), "sent_eom"] == 59


def test_fng_short_month_is_nan_and_bad_values_rejected():
    days = pd.date_range("2020-01-01", periods=10, freq="D")
    d = pd.DataFrame({"date": days, "value": 50.0})
    assert S.fng_monthly(d)["sent_mean"].isna().all()
    with pytest.raises(ValueError):
        S.parse_fng({"data": [{"value": "150", "timestamp": "1577836800"}]})
    with pytest.raises(ValueError):
        S.parse_fng({"nope": []})


def test_ar2_recovers_coefficients():
    rng = np.random.default_rng(0)
    n, c, p1, p2 = 4000, 0.5, 0.6, 0.2
    x = np.zeros(n)
    for t in range(2, n):
        x[t] = c + p1 * x[t - 1] + p2 * x[t - 2] + rng.standard_normal()
    s = pd.Series(x, index=_months(n, "1700-01"), name="x")
    e, p = S.ar2_residual(s)
    assert p["phi1"] == pytest.approx(p1, abs=0.05) and p["phi2"] == pytest.approx(p2, abs=0.05)
    assert len(e) == n - 2 and abs(e.mean()) < 1e-10
    assert abs(np.corrcoef(e.values, x[1:-1])[0, 1]) < 1e-8


def test_illiq_states_start_and_gap_check():
    m = _months(40, "2018-01")
    rng = np.random.default_rng(1)
    il = pd.DataFrame({"n_coins": [5] * 4 + [50] * 36, "ew_A": np.exp(rng.standard_normal(40))}, index=m)
    il["ln_ew_A"] = np.log(il["ew_A"])
    st, fits = S.illiq_states(il)
    assert st.index.min() == m[4]
    assert st["illiq_ar2"].notna().sum() == 34 and fits[0]["n"] == 34
    with pytest.raises(ValueError):
        S.illiq_states(il.drop(index=m[10]))


def test_lag_state_alignment():
    st = pd.Series([1.0, 2.0, 3.0], index=_months(3))
    lagged = S.lag_state(st, _months(3))
    assert np.isnan(lagged.iloc[0]) and lagged.iloc[1] == 1.0 and lagged.iloc[2] == 2.0


def test_state_regression_known_answer_and_timing():
    rng = np.random.default_rng(2)
    n = 600
    m = _months(n, "1900-01")
    sent = pd.Series(rng.standard_normal(n), index=m)
    cmkt = pd.Series(rng.standard_normal(n), index=m)
    lag = S.lag_state(sent, m)
    y = 0.5 * lag + 1.0 * cmkt + 0.3 * rng.standard_normal(n)
    r = S.state_regression(y, lag, cmkt.to_frame("CMKT"), 6)
    assert r["b"] == pytest.approx(0.5, abs=0.05) and r["t"] > 10
    r0 = S.state_regression(y, sent, cmkt.to_frame("CMKT"), 6)
    assert abs(r0["t"]) < 3


def test_regime_table_known_difference():
    m = _months(200, "1900-01")
    st = pd.Series(np.tile([1.0, -1.0], 100), index=m)
    prem = pd.DataFrame({c: np.where(st > 0, 2.0, 1.0) for c in S.PORTS}, index=m)
    prem += np.random.default_rng(3).normal(0, 0.1, prem.shape)
    t = S.regime_table(prem, st, 6).set_index("regime")
    assert t.loc["High", "Long"] == pytest.approx(2.0, abs=0.05)
    assert t.loc["Low", "Long"] == pytest.approx(1.0, abs=0.05)
    assert t.loc["High-Low", "Long-Short"] == pytest.approx(1.0, abs=0.05)
    assert t.loc["High", "n_months"] == 100


def _toy_panel(n_months=60, n_coins=60, seed=4):
    rng = np.random.default_rng(seed)
    rows = []
    for m in _months(n_months):
        A_C = np.exp(rng.standard_normal(n_coins))
        r = 0.5 * np.log(A_C) + rng.standard_normal(n_coins)
        rows.append(pd.DataFrame({"symbol": [f"C{i}" for i in range(n_coins)], "month": m,
                                  "A_C": A_C, "lnA_C": np.log(A_C), "ret": r, "ret_adj": r,
                                  "mom": rng.standard_normal(n_coins), "ret_1": rng.standard_normal(n_coins)}))
    return pd.concat(rows, ignore_index=True)


def test_premium_series_signs():
    prem = S.premium_series(_toy_panel(), 5)
    assert list(prem.columns) == S.PORTS
    assert prem["Long-Short"].mean() > 0 and prem["FM_lnA_C"].mean() == pytest.approx(0.5, abs=0.1)
    assert np.allclose(prem["Long-Short"], prem["Long"] - prem["Short"])


def test_run_state_tests_end_to_end():
    panel = _toy_panel()
    prem = S.premium_series(panel, 5)
    m = prem.index
    rng = np.random.default_rng(5)
    states = pd.DataFrame({k: rng.standard_normal(len(m) + 1) for k in S.STATE_VARS},
                          index=pd.period_range(m[0] - 1, m[-1], freq="M"))
    fac = pd.DataFrame({"CMKT": rng.standard_normal(len(m)), "CMOM": rng.standard_normal(len(m))}, index=m)
    pb, pa = S.run_state_tests(prem, states, fac, 6)
    assert len(pb) == len(S.STATE_VARS) * len(S.CONTROLS) * len(S.PORTS)
    assert pb["main"].sum() == 2 * len(S.PORTS)
    assert set(pa["regime"]) == {"High", "Low", "High-Low"}
    pvu = S.paper_vs_us(pb, pa)
    assert len(pvu) == len(S.PAPER)
    assert isinstance(S.verdict_sentiment(pb), str) and isinstance(S.verdict_illiq(pb), str)


def _pb(state, b_short, t_short, b_ls, t_ls):
    return pd.DataFrame([{"state": state, "controls": S.MAIN_CONTROLS, "portfolio": "Short", "b": b_short, "t": t_short},
                         {"state": state, "controls": S.MAIN_CONTROLS, "portfolio": "Long-Short", "b": b_ls, "t": t_ls}])


def test_verdict_rules():
    assert S.verdict_sentiment(_pb("sent_mean", -1, -2.5, 1, 2.5)).startswith("the mispricing prediction is confirmed")
    assert S.verdict_sentiment(_pb("sent_mean", -1, -1.0, 1, 2.5)).startswith("the signs match")
    assert S.verdict_sentiment(_pb("sent_mean", 1, 2.5, 1, 2.5)).startswith("the mispricing prediction is not confirmed")
    assert "evidence for the liquidity explanation" in S.verdict_illiq(_pb("illiq_ar2", 0, 0, 1, 2.5))
    assert "against the liquidity explanation" in S.verdict_illiq(_pb("illiq_ar2", 0, 0, -1, -2.5))
    assert S.verdict_illiq(_pb("illiq_ar2", 0, 0, 1, 1.0)).startswith("no significant link")


def test_add_lagged_market_alignment_and_gap():
    m = pd.period_range("2020-01", "2020-04", freq="M")
    fac = pd.DataFrame({"CMKT": [1.0, 2.0, 3.0, 4.0], "CMOM": 0.0}, index=m).drop(m[2])
    f = S.add_lagged_market(fac)
    assert np.isnan(f.loc[m[0], "CMKT_l1"]) and f.loc[m[1], "CMKT_l1"] == 1.0
    assert np.isnan(f.loc[m[3], "CMKT_l1"])
