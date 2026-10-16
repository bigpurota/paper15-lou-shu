"""EDGE spread estimator: reference match with `bidask`, known-answer simulation, costs bookkeeping."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import spread_edge as se


def _sim_ohlc(n_days: int, spread: float, sigma_day: float, steps: int, rng) -> pd.DataFrame:
    """Daily OHLC from trades at bid/ask around a random-walk efficient price (known spread)."""
    eff = np.cumsum(rng.normal(0.0, sigma_day / np.sqrt(steps), n_days * steps))
    side = rng.choice([-1.0, 1.0], n_days * steps)
    px = np.exp(eff + side * spread / 2).reshape(n_days, steps)
    return pd.DataFrame({"open": px[:, 0], "high": px.max(1), "low": px.min(1), "close": px[:, -1]})


def _ref(df: pd.DataFrame) -> float:
    bidask = pytest.importorskip("bidask")
    return bidask.edge(df["open"].to_numpy(), df["high"].to_numpy(), df["low"].to_numpy(),
                       df["close"].to_numpy(), sign=True)


# ------------------------------------------------------------------ reference implementation
def test_matches_bidask_on_simulated_months():
    rng = np.random.default_rng(1)
    for k in range(50):
        df = _sim_ohlc(31, rng.uniform(0, 0.03), rng.uniform(0.01, 0.08), 50, rng)
        ours = se.edge_from_ohlc(df["open"], df["high"], df["low"], df["close"])
        assert ours["edge"] == pytest.approx(_ref(df), abs=1e-10)


def test_gap_handling_equals_bidask_with_nan_rows():
    """A missing day: our pair mask must equal bidask fed with a NaN row for that day."""
    rng = np.random.default_rng(2)
    df = _sim_ohlc(31, 0.01, 0.04, 50, rng)
    df["date"] = pd.date_range("2024-01-01", periods=31, freq="D")
    df.loc[[5, 17], ["open", "high", "low", "close"]] = np.nan          # reference: NaN rows
    obs = df.dropna()                                                    # rows absent in the data
    full = se.calendar_month(obs)
    assert full["open"].isna().sum() == 2
    ours = se.edge_from_ohlc(full["open"], full["high"], full["low"], full["close"])
    assert ours["n_pairs"] == 30 - 4
    assert ours["edge"] == pytest.approx(_ref(df), abs=1e-10)
    em = se.edge_monthly(obs.assign(symbol="X"), 10)
    assert em.loc[0, "edge"] == pytest.approx(ours["edge"], abs=1e-14) and em.loc[0, "n_days"] == 29


def test_flat_bars_and_degenerate_inputs():
    flat = pd.DataFrame({c: np.full(20, 5.0) for c in ["open", "high", "low", "close"]})
    assert np.isnan(se.edge_from_ohlc(flat["open"], flat["high"], flat["low"], flat["close"])["edge"])
    assert np.isnan(_ref(flat))
    two = flat.iloc[:2]
    assert np.isnan(se.edge_from_ohlc(two["open"], two["high"], two["low"], two["close"])["edge"])


def test_matches_bidask_on_real_coin_months():
    if not se.OHLC_PATH.exists():
        pytest.skip("data/cache/ohlc_daily.parquet not built yet")
    pytest.importorskip("bidask")
    ohlc = pd.read_parquet(se.OHLC_PATH)
    syms = pd.Series(ohlc["symbol"].unique()).sample(25, random_state=0)
    sub = ohlc[ohlc["symbol"].isin(syms)]
    em = se.edge_monthly(sub, 10)
    chk = se.bidask_check(sub, em, 200, seed=0)
    assert len(chk) >= 100
    assert chk["abs_diff"].max() < 1e-8


# ------------------------------------------------------------------ known answer
@pytest.mark.parametrize("spread", [0.0, 0.005, 0.02])
def test_recovers_known_spread(spread):
    """Mean s2 over many simulated coin-months recovers spread^2 (EDGE is unbiased in s2)."""
    rng = np.random.default_rng(3)
    s2 = []
    for _ in range(400):
        df = _sim_ohlc(30, spread, 0.03, 200, rng)
        s2.append(se.edge_from_ohlc(df["open"], df["high"], df["low"], df["close"])["s2"])
    s2 = np.array(s2)
    se_mean = s2.std(ddof=1) / np.sqrt(len(s2))
    assert abs(s2.mean() - spread ** 2) < 3.5 * se_mean + 0.1 * spread ** 2
    if spread == 0.0:
        assert 0.25 < (s2 < 0).mean() < 0.75                       # negative estimates are pure noise
    else:
        assert np.sqrt(max(s2.mean(), 0)) == pytest.approx(spread, rel=0.15)


def test_edge_monthly_uses_only_within_month_consecutive_pairs():
    rng = np.random.default_rng(4)
    df = _sim_ohlc(62, 0.01, 0.03, 50, rng)
    df["date"] = pd.date_range("2024-01-01", periods=62, freq="D")      # Jan (31) + Feb (29) + 2 Mar
    df["symbol"] = "XUSDT"
    em = se.edge_monthly(df, min_pairs=10).set_index("month")
    assert em.loc["2024-01", "n_pairs"] == 30 and em.loc["2024-02", "n_pairs"] == 28
    assert np.isnan(em.loc["2024-03", "edge"]) and em.loc["2024-03", "n_pairs"] < 10
    jan = df[df["date"].dt.month == 1]
    assert em.loc["2024-01", "edge"] == pytest.approx(
        se.edge_from_ohlc(jan["open"], jan["high"], jan["low"], jan["close"])["edge"], abs=1e-14)


def test_edge_window_mean_s2():
    em = pd.DataFrame({"symbol": "X", "month": [f"2024-{m:02d}" for m in range(1, 13)],
                       "s2": np.r_[np.full(6, 4e-4), np.full(6, -2e-4)]})
    em["edge"] = np.sign(em["s2"]) * np.sqrt(em["s2"].abs())
    w = se.edge_window(em, 12).set_index("month")
    assert np.isnan(w.loc["2024-05", "edge"])                            # < 6 months
    assert w.loc["2024-12", "s2"] == pytest.approx(1e-4)
    assert w.loc["2024-12", "edge"] == pytest.approx(0.01)


# ------------------------------------------------------------------ rules and costs
def test_nonpositive_rules():
    e = pd.Series([0.02, -0.01, 0.0, 0.005, np.nan, 0.03])
    m = pd.Series(["a", "a", "a", "a", "a", "b"])
    floor = se.edge_signal(e, m, "floor")
    assert floor.tolist()[:4] == [0.02, 0.005, 0.005, 0.005] and np.isnan(floor[4]) and floor[5] == 0.03
    assert se.edge_signal(e, m, "drop").isna().tolist() == [False, True, True, False, True, False]
    assert se.edge_signal(e, m, "abs").tolist()[:2] == [0.02, 0.01]


def _toy_panel(replace_all: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """10 coins, 5 per month in the signal; with replace_all the extreme quintiles swap each month."""
    rows, em = [], []
    for k, m in enumerate(["2024-01", "2024-02", "2024-03", "2024-04"]):
        for i in range(10):
            sig = (i if (k % 2 == 0 or not replace_all) else 9 - i)
            rows.append({"month": m, "symbol": f"C{i}", "A_C": float(sig), "ret": 0.0})
    for m in ["2023-11", "2023-12", "2024-01", "2024-02", "2024-03"]:
        for i in range(10):
            em.append({"symbol": f"C{i}", "month": m, "edge": 0.01 * (1 + i), "s2": (0.01 * (1 + i)) ** 2})
    return pd.DataFrame(rows), pd.DataFrame(em)


def test_costs_no_change_means_no_turnover():
    p, em = _toy_panel(replace_all=False)
    ts, row = se.costs(p, em, "A_C", 5, "floor", 1, None)
    assert ts.loc[0, "turnover_long"] == pytest.approx(1.0)            # first month: build
    assert ts.loc[1:, ["turnover_long", "turnover_short"]].abs().max().max() < 1e-12
    assert row.iloc[0]["cost_ls_pct"] == pytest.approx(0.0)


def test_costs_full_replacement():
    p, em = _toy_panel(replace_all=True)
    ts, row = se.costs(p, em, "A_C", 5, "floor", 1, None)
    s = ts.iloc[1:]
    assert np.allclose(s["turnover_long"], 2.0) and np.allclose(s["turnover_short"], 2.0)
    # each leg sells 2 coins and buys 2 others (1/2 each); half-spread = edge/2 of the coin traded
    # long leg holds {C8,C9} or {C0,C1}; short leg the other pair -> both legs trade all four coins
    hs = np.array([0.01, 0.02, 0.09, 0.10]) / 2
    assert np.allclose(s["cost_long"], 100 * 0.5 * hs.sum())
    assert np.allclose(s["cost_ls"], 2 * 100 * 0.5 * hs.sum())
    assert np.allclose(s["hl_net"], -s["cost_ls"])


def test_costs_pooled_rule_uses_root_mean_s2():
    p, em = _toy_panel(replace_all=True)
    ts, _ = se.costs(p, em, "A_C", 5, "pooled", 1, None)
    s = ts.iloc[1:]
    # month t uses EDGE of t-1; e.g. long leg {C8,C9}: sqrt(mean(0.09^2, 0.10^2))/2
    hi = np.sqrt((0.09 ** 2 + 0.10 ** 2) / 2) / 2 * 100
    lo = np.sqrt((0.01 ** 2 + 0.02 ** 2) / 2) / 2 * 100
    assert set(np.round(s["hs_long"], 10)) <= {round(hi, 10), round(lo, 10)}


def test_drift_of_weights_creates_turnover():
    """Same constituents, unequal returns: rebalancing to 1/N costs exactly sum |w~ - 1/N|."""
    rows = []
    for m in ["2024-01", "2024-02"]:
        for i in range(10):
            rows.append({"month": m, "symbol": f"C{i}", "A_C": float(i),
                         "ret": (100.0 if i == 9 else 0.0) if m == "2024-01" else 0.0})
    p = pd.DataFrame(rows)
    em = pd.DataFrame([{"symbol": f"C{i}", "month": m, "edge": 0.02, "s2": 4e-4}
                       for i in range(10) for m in ["2023-11", "2023-12", "2024-01"]])
    ts, _ = se.costs(p, em, "A_C", 5, "floor", 1, None)
    # long leg {C8, C9}: after +100% on C9 weights are 1/3, 2/3 -> back to 1/2, 1/2: turnover 1/3
    assert ts.loc[1, "turnover_long"] == pytest.approx(1 / 3)
    assert ts.loc[1, "turnover_short"] == pytest.approx(0.0)
    assert ts.loc[1, "cost_long"] == pytest.approx(100 * (1 / 3) * 0.01)


def test_fee_and_breakeven_columns():
    """Fee = (turnover_long + turnover_short) x fee per side; break-even per-side cost = gross / turnover."""
    p, em = _toy_panel(replace_all=True)
    p["ret"] = np.where(p["A_C"] >= 8, 3.0, 0.0)                         # long leg earns 3%/month
    ts, row = se.costs(p, em, "A_C", 5, "floor", 1, None)
    s = ts.iloc[1:]
    assert np.allclose(s["fee_ls"], 4.0 * se.FEE_PER_SIDE * 100)        # turnover 2 + 2
    assert np.allclose(s["hl_net_fee"], s["hl_net"] - s["fee_ls"])
    r = row.iloc[0]
    assert r["breakeven_cost_per_side_pct"] == pytest.approx(r["hl_gross_pct"] / 4.0)
    assert r["share_months_hs_zero_long"] == 0.0
