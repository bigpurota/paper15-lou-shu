"""Unit tests for src/measures.py on toy data with hand-computable answers."""
import numpy as np
import pandas as pd
import pytest

from measures import cs_residual, daily_returns, mean_cs_corr, monthly_measures, winsorize_cs


def _daily(symbol, dates, close, qv):
    return pd.DataFrame({"symbol": symbol, "date": pd.to_datetime(dates), "close": np.asarray(close, float),
                         "quote_volume": np.asarray(qv, float)})


def test_daily_returns_gap_and_volume():
    dates = ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-05", "2020-01-06", "2020-01-07"]
    d = daily_returns(_daily("X", dates, [100, 110, 99, 120, 132, 66], [1, 2, 0, 5, 6, np.nan]))
    r = d["r"].to_numpy()
    assert np.isnan(r[0])
    assert r[1] == pytest.approx(0.10)
    assert r[2] == pytest.approx(-0.10)
    assert np.isnan(r[3])
    assert r[4] == pytest.approx(0.10)
    assert r[5] == pytest.approx(-0.50)
    assert d["valid"].tolist() == [False, True, False, False, True, False]
    assert (d["month"] == pd.Period("2020-01", "M")).all()


def test_daily_returns_sorts_and_separates_symbols():
    a = _daily("A", ["2020-01-02", "2020-01-01"], [110, 100], [1, 1])
    b = _daily("B", ["2020-01-01", "2020-01-02"], [50, 25], [1, 1])
    d = daily_returns(pd.concat([b, a], ignore_index=True))
    assert d["symbol"].tolist() == ["A", "A", "B", "B"]
    assert np.isnan(d["r"].iloc[0]) and np.isnan(d["r"].iloc[2])
    assert d["r"].iloc[1] == pytest.approx(0.10)
    assert d["r"].iloc[3] == pytest.approx(-0.50)


@pytest.fixture(scope="module")
def toy():
    """Two symbols, Jan-Mar 2020 (X) and Jan + Mar 2020 (Y, February missing entirely)."""
    rng = np.random.default_rng(0)
    frames = []
    for sym, months in [("X", ["2020-01", "2020-02", "2020-03"]), ("Y", ["2020-01", "2020-03"])]:
        for m in months:
            dates = pd.date_range(pd.Period(m).start_time, pd.Period(m).end_time.normalize(), freq="D")
            close = 10 * np.exp(np.cumsum(rng.normal(0, 0.05, len(dates))))
            qv = rng.lognormal(12, 1, len(dates))
            frames.append(_daily(sym, dates, close, qv))
    daily = pd.concat(frames, ignore_index=True)
    daily.loc[(daily.symbol == "X") & (daily.date == "2020-01-10"), "quote_volume"] = 0.0
    daily = daily[~((daily.symbol == "X") & (daily.date == "2020-01-20"))].reset_index(drop=True)
    d = daily_returns(daily)
    return d, monthly_measures(d, scale=1e6)


def _valid_days(d, sym, month):
    g = d[(d.symbol == sym) & (d.month == pd.Period(month, "M")) & d.valid]
    return g["r"].to_numpy(), g["quote_volume"].to_numpy()


@pytest.mark.parametrize("sym,month", [("X", "2020-01"), ("X", "2020-02"), ("X", "2020-03"),
                                       ("Y", "2020-01"), ("Y", "2020-03")])
def test_monthly_measures_match_numpy(toy, sym, month):
    d, m = toy
    r, dv = _valid_days(d, sym, month)
    row = m[(m.symbol == sym) & (m.month == pd.Period(month, "M"))].iloc[0]
    ar, inv = np.abs(r), 1.0 / dv
    A, AC, absret = np.mean(ar * inv) * 1e6, np.mean(inv) * 1e6, np.mean(ar)
    assert row["n_days"] == len(r)
    assert row["A"] == pytest.approx(A, rel=1e-12)
    assert row["A_C"] == pytest.approx(AC, rel=1e-12)
    assert row["absret"] == pytest.approx(absret, rel=1e-12)
    cov_pop = np.cov(ar, inv, ddof=0)[0, 1] * 1e6
    assert row["COV"] == pytest.approx(cov_pop, rel=1e-8, abs=1e-12)
    assert row["COV"] == pytest.approx(row["A"] - row["absret"] * row["A_C"], rel=1e-12)
    assert row["DIF"] == pytest.approx(np.log(A) - np.log(absret) - np.log(AC), rel=1e-10)
    slope = np.polyfit(dv, ar, 1)[0]
    assert row["b"] == pytest.approx(slope, rel=1e-6)
    assert row["CV2"] == pytest.approx(np.var(dv) / np.mean(dv) ** 2, rel=1e-8)
    assert row["dvol"] == pytest.approx(dv.mean(), rel=1e-12)
    assert row["AN"] == pytest.approx(np.mean(np.maximum(-r, 0) * inv) * 1e6, rel=1e-12)
    assert row["AP"] == pytest.approx(np.mean(np.maximum(r, 0) * inv) * 1e6, rel=1e-12)
    assert row["AN"] + row["AP"] == pytest.approx(row["A"], rel=1e-12)


def test_monthly_n_days_and_same_valid_days(toy):
    d, m = toy
    xj = m[(m.symbol == "X") & (m.month == pd.Period("2020-01", "M"))].iloc[0]
    assert xj["n_days"] == 27
    ym = m[(m.symbol == "Y") & (m.month == pd.Period("2020-03", "M"))].iloc[0]
    assert ym["n_days"] == 30
    assert np.isfinite(m[["A", "A_C", "absret", "COV", "DIF"]].to_numpy()).all()


def test_monthly_ret_month_end_consecutive_only(toy):
    d, m = toy
    close_end = d.groupby(["symbol", "month"])["close"].last()
    get = lambda s, mo: m[(m.symbol == s) & (m.month == pd.Period(mo, "M"))]["ret"].iloc[0]
    P = lambda mo: pd.Period(mo, "M")
    assert np.isnan(get("X", "2020-01"))
    assert get("X", "2020-02") == pytest.approx(close_end["X", P("2020-02")] / close_end["X", P("2020-01")] - 1)
    assert get("X", "2020-03") == pytest.approx(close_end["X", P("2020-03")] / close_end["X", P("2020-02")] - 1)
    assert np.isnan(get("Y", "2020-01"))
    assert np.isnan(get("Y", "2020-03"))
    assert len(m) == 5


def test_monthly_measures_accepts_raw_daily():
    dates = pd.date_range("2020-01-01", "2020-01-05")
    raw = _daily("Z", dates, [1, 2, 1, 2, 4], [1e6, 2e6, 1e6, 4e6, 2e6])
    m = monthly_measures(raw, scale=1e6)
    assert m["n_days"].iloc[0] == 4
    assert m["absret"].iloc[0] == pytest.approx(0.875)
    assert m["A_C"].iloc[0] == pytest.approx(np.mean([0.5, 1, 0.25, 0.5]))
    assert m["A"].iloc[0] == pytest.approx(np.mean([0.5, 0.5, 0.25, 0.5]))


def test_winsorize_cs_per_cross_section():
    df = pd.DataFrame({"month": [1] * 101 + [2] * 101,
                       "v": np.r_[np.arange(101.0), 1000 + np.arange(101.0)]})
    w = winsorize_cs(df, ["v"], "month", (0.01, 0.99))
    a, b = w["v"].to_numpy()[:101], w["v"].to_numpy()[101:]
    assert a.min() == pytest.approx(1.0) and a.max() == pytest.approx(99.0)
    assert b.min() == pytest.approx(1001.0) and b.max() == pytest.approx(1099.0)
    np.testing.assert_allclose(a[1:100], np.arange(1.0, 100.0))
    assert df["v"].max() == 1100.0


def test_winsorize_cs_keeps_nan():
    df = pd.DataFrame({"month": [1] * 5, "v": [1.0, np.nan, 3.0, 4.0, 100.0]})
    w = winsorize_cs(df, ["v"], "month", (0.0, 0.75))
    assert np.isnan(w["v"].iloc[1])
    assert w["v"].iloc[4] == pytest.approx(np.nanquantile(df["v"], 0.75))


def test_cs_residual_orthogonal_zero_mean():
    rng = np.random.default_rng(1)
    n = 50
    df = pd.DataFrame({"month": np.repeat([1, 2, 3], n), "x": rng.normal(size=3 * n)})
    df["y"] = np.repeat([1.0, -2.0, 5.0], n) + np.repeat([0.5, 3.0, -1.0], n) * df["x"] + rng.normal(size=3 * n)
    df.loc[7, "y"] = np.nan
    df = df.sample(frac=1, random_state=3)
    res = cs_residual(df, "y", "x", "month")
    assert isinstance(res, pd.Series)
    assert res.index.equals(df.index) or set(res.index) == set(df.index)
    res = res.reindex(df.index)
    assert np.isnan(res.loc[7])
    for _, g in df.assign(e=res).dropna().groupby("month"):
        assert abs(g["e"].mean()) < 1e-10
        assert abs((g["e"] * g["x"]).sum()) < 1e-8
        beta = np.polyfit(g["x"], g["y"], 1)
        np.testing.assert_allclose(g["e"], g["y"] - np.polyval(beta, g["x"]), atol=1e-10)


def test_cs_residual_single_month():
    df = pd.DataFrame({"month": [1] * 5, "x": [0.0, 1, 2, 3, 4], "y": [1.0, 3, 5, 7, 10]})
    res = cs_residual(df, "y", "x", "month")
    assert isinstance(res, pd.Series) and len(res) == 5
    beta = np.polyfit(df["x"], df["y"], 1)
    np.testing.assert_allclose(res.reindex(df.index), df["y"] - np.polyval(beta, df["x"]), atol=1e-10)


def test_mean_cs_corr():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"month": np.repeat([1, 2], 20), "a": rng.normal(size=40)})
    df["b"] = df["a"] * np.repeat([1.0, -1.0], 20)
    c = mean_cs_corr(df, ["a", "b"], "month")
    assert c.loc["a", "b"] == pytest.approx(0.0, abs=1e-12)
    assert c.loc["a", "a"] == pytest.approx(1.0)
