"""Unit tests for Fama-MacBeth, Newey-West, sorts, factor betas and IVOL on toy data."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import yaml

from factors import cmkt_monthly, daily_cmkt, ivol_monthly, rolling_betas
from fama_macbeth import fama_macbeth, nw_mean
from measures import daily_returns
from sorts import alpha, assign_quantiles, hl_factor, portfolio_returns, sort_table

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- Newey-West, Fama-MacBeth
def test_nw_mean_equals_statsmodels_hac():
    rng = np.random.default_rng(10)
    e = rng.normal(size=300)
    x = 0.2 + e + 0.6 * np.r_[0, e[:-1]]                 # MA(1): autocorrelated
    m, se, t = nw_mean(pd.Series(x), lags=6)
    ref = sm.OLS(x, np.ones((len(x), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    assert m == pytest.approx(x.mean())
    assert se == pytest.approx(ref.bse[0])
    assert t == pytest.approx(ref.tvalues[0])
    # hand-computed Bartlett HAC (statsmodels default: no small-sample correction)
    u = x - x.mean()
    T = len(u)
    s = (u @ u) / T + 2 * sum((1 - L / 7) * (u[L:] @ u[:-L]) / T for L in range(1, 7))
    assert se == pytest.approx(np.sqrt(s / T), rel=1e-10)


def test_nw_mean_drops_nan_and_iid_close_to_plain_t():
    rng = np.random.default_rng(11)
    x = rng.normal(0.1, 1.0, 2000)
    xs = pd.Series(np.r_[x, np.nan])
    m, se, t = nw_mean(xs, 6)
    t_plain = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    assert m == pytest.approx(x.mean())
    assert t == pytest.approx(t_plain, rel=0.10)


def test_fama_macbeth_recovers_slopes():
    rng = np.random.default_rng(12)
    T, N = 120, 200
    df = pd.DataFrame({"month": np.repeat(np.arange(T), N), "x": rng.normal(size=T * N),
                       "z": rng.normal(size=T * N)})
    df["y"] = 1 + 2 * df["x"] - 0.5 * df["z"] + rng.normal(0, 0.5, T * N)
    res = fama_macbeth(df, "y", ["x", "z"], lags=6).set_index("var")
    assert res.loc["x", "coef"] == pytest.approx(2.0, abs=0.01)
    assert res.loc["z", "coef"] == pytest.approx(-0.5, abs=0.01)
    assert res.loc["x", "t"] > 100
    assert res.attrs["n_months"] == T and res.attrs["avg_obs"] == N
    # mean of the per-month OLS slopes, exactly
    cs = res.attrs["coef_series"]
    g0 = df[df.month == 0]
    b0 = np.linalg.lstsq(np.column_stack([np.ones(N), g0[["x", "z"]]]), g0["y"], rcond=None)[0]
    np.testing.assert_allclose(cs.iloc[0].to_numpy(), b0[1:], rtol=1e-10)
    assert res.loc["x", "coef"] == pytest.approx(cs["x"].mean())
    assert res.loc["x", "t"] == pytest.approx(nw_mean(cs["x"], 6)[2])


def test_fama_macbeth_exact_and_min_obs():
    rows = []
    for t in range(10):
        x = np.arange(15.0) + t
        rows.append(pd.DataFrame({"month": t, "x": x, "y": 3.0 + (t % 2 + 1) * x}))
    df = pd.concat(rows, ignore_index=True)
    df.loc[df.month == 9, "y"] = np.nan                     # month 9 has no observations
    res = fama_macbeth(df, "y", ["x"], lags=2)
    assert res.attrs["n_months"] == 9
    assert res["coef"].iloc[0] == pytest.approx(np.mean([1, 2, 1, 2, 1, 2, 1, 2, 1]))
    assert res.attrs["avg_adj_r2"] == pytest.approx(1.0)


# ---------------------------------------------------------------- sorts
def _sort_panel(seed=20, T=60, N=50):
    rng = np.random.default_rng(seed)
    p = pd.DataFrame({"month": np.repeat(pd.period_range("2015-01", periods=T, freq="M"), N),
                      "s": rng.normal(size=T * N), "w": rng.uniform(1, 5, T * N)})
    p["ret"] = 0.5 * p["s"] + rng.normal(size=T * N)
    return p


def test_assign_quantiles_equal_sizes():
    p = _sort_panel(T=3, N=50)
    p.loc[0, "s"] = np.nan
    q = assign_quantiles(p, "s", 5)
    assert isinstance(q, pd.Series) and q.index.equals(p.index) or set(q.index) == set(p.index)
    q = q.reindex(p.index)
    assert np.isnan(q.loc[0])
    for m, g in p.assign(q=q).groupby("month"):
        counts = g["q"].value_counts().sort_index()
        assert list(counts.index) == [1, 2, 3, 4, 5]
        assert counts.max() - counts.min() <= 1
        ok = g.dropna(subset=["s"])
        # monotone: every Q1 signal below every Q5 signal
        assert ok.loc[ok.q == 1, "s"].max() < ok.loc[ok.q == 2, "s"].min()
        assert ok.loc[ok.q == 4, "s"].max() < ok.loc[ok.q == 5, "s"].min()


def test_assign_quantiles_single_month():
    p = pd.DataFrame({"month": 1, "s": np.arange(10.0)[::-1]})
    q = assign_quantiles(p, "s", 5).reindex(p.index)
    assert q.tolist() == [5, 5, 4, 4, 3, 3, 2, 2, 1, 1]


def test_portfolio_returns_ew_and_weighted():
    p = pd.DataFrame({"month": [1] * 4, "s": [1.0, 2, 3, 4], "ret": [1.0, 3, 10, 20], "w": [1.0, 3, 1, 1]})
    ew = portfolio_returns(p, "s", "ret", 2)
    assert ew.loc[1, 1] == pytest.approx(2.0) and ew.loc[1, 2] == pytest.approx(15.0)
    vw = portfolio_returns(p, "s", "ret", 2, weight="w")
    assert vw.loc[1, 1] == pytest.approx((1 * 1 + 3 * 3) / 4) and vw.loc[1, 2] == pytest.approx(15.0)
    assert list(ew.columns) == [1, 2]


def test_sort_table_hl_and_alpha():
    p = _sort_panel()
    rng = np.random.default_rng(21)
    months = p["month"].unique()
    f = pd.DataFrame({"F1": rng.normal(size=len(months)), "F2": rng.normal(size=len(months))},
                     index=pd.PeriodIndex(months, freq="M"))
    tab = sort_table(p, "s", "ret", 5, f, lags=6)
    pr = tab.attrs["portfolios"]
    hl = pr[5] - pr[1]
    assert tab.loc["raw", "H-L"] == pytest.approx(hl.mean())
    assert tab.loc["raw", "H-L"] == pytest.approx(tab.loc["raw", "Q5"] - tab.loc["raw", "Q1"])
    assert tab.loc["raw", "t"] == pytest.approx(nw_mean(hl, 6)[2])
    X = sm.add_constant(f.loc[hl.index].to_numpy())
    a_ref = sm.OLS(hl.to_numpy(), X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    assert tab.loc["alpha", "H-L"] == pytest.approx(a_ref.params[0])
    assert tab.loc["alpha", "t"] == pytest.approx(a_ref.tvalues[0])
    assert tab.loc["raw", "n_months"] == 60
    assert tab.loc["raw", "H-L"] > 0                     # ret increasing in signal
    hf = hl_factor(p, "s", "ret", 5)
    np.testing.assert_allclose(hf.to_numpy(), hl.to_numpy())


def test_alpha_exact_intercept():
    idx = pd.period_range("2020-01", periods=40, freq="M")
    rng = np.random.default_rng(22)
    f = pd.DataFrame({"F": rng.normal(size=40)}, index=idx)
    y = pd.Series(0.3 + 1.5 * f["F"].to_numpy(), index=idx)
    a, _ = alpha(y, f)
    assert a == pytest.approx(0.3, abs=1e-12)
    a0, t0 = alpha(y, None)
    assert a0 == pytest.approx(y.mean())


# ---------------------------------------------------------------- factors
def _monthly_beta_panel():
    months = pd.period_range("2015-01", periods=60, freq="M")
    rng = np.random.default_rng(30)
    f = pd.Series(rng.normal(0, 0.1, 60), index=months)
    a = pd.DataFrame({"symbol": "A", "month": months, "ret": 0.5 + 1.7 * f.to_numpy()})
    # B starts later: only months 40.. exist
    b = pd.DataFrame({"symbol": "B", "month": months[40:], "ret": -0.1 + 0.8 * f.to_numpy()[40:]})
    return pd.concat([a, b], ignore_index=True), f


def test_rolling_betas_exact_and_no_lookahead():
    monthly, f = _monthly_beta_panel()
    b = rolling_betas(monthly, f, window=36, min_obs=12).set_index(["symbol", "month"])["beta"]
    months = f.index
    # A: beta(t) defined from month index 12 (12 prior months) onward and equals 1.7
    assert np.isnan(b.get(("A", months[11]), np.nan))
    for t in months[12:]:
        assert b[("A", t)] == pytest.approx(1.7, rel=1e-9)
    # B: first month 40; needs 12 prior obs -> NaN through month 51, defined from 52
    for t in months[40:52]:
        assert np.isnan(b.get(("B", t), np.nan))
    assert b[("B", months[52])] == pytest.approx(0.8, rel=1e-9)

    # no use of month t: perturb month t's return of A, beta(t) unchanged; beta(t+1) changes
    t = months[45]
    m2 = monthly.copy()
    m2.loc[(m2.symbol == "A") & (m2.month == t), "ret"] += 5.0
    b2 = rolling_betas(m2, f, 36, 12).set_index(["symbol", "month"])["beta"]
    assert b2[("A", t)] == pytest.approx(b[("A", t)], rel=1e-12)
    assert abs(b2[("A", months[46])] - 1.7) > 1e-3
    # window: perturbation at month 45 drops out after 36 months (month 82 is outside the sample),
    # check with month 10: affects beta(11..46), not beta(47)
    m3 = monthly.copy()
    m3.loc[(m3.symbol == "A") & (m3.month == months[10]), "ret"] += 5.0
    b3 = rolling_betas(m3, f, 36, 12).set_index(["symbol", "month"])["beta"]
    assert abs(b3[("A", months[46])] - 1.7) > 1e-3
    assert b3[("A", months[47])] == pytest.approx(1.7, rel=1e-9)


def test_rolling_betas_matches_numpy_ols_with_noise():
    monthly, f = _monthly_beta_panel()
    rng = np.random.default_rng(31)
    monthly["ret"] = monthly["ret"] + rng.normal(0, 0.05, len(monthly))
    monthly.loc[5, "ret"] = np.nan                         # a missing month inside the window
    b = rolling_betas(monthly, f, 36, 12).set_index(["symbol", "month"])["beta"]
    A = monthly[monthly.symbol == "A"].set_index("month")["ret"]
    for i in [20, 36, 50, 59]:
        t = f.index[i]
        win = f.index[max(0, i - 36):i]
        y, x = A.loc[win], f.loc[win]
        ok = y.notna()
        ref = np.polyfit(x[ok], y[ok], 1)[0]
        assert b[("A", t)] == pytest.approx(ref, rel=1e-8)


def test_cmkt_monthly_uses_lagged_eligibility():
    months = pd.period_range("2020-01", periods=3, freq="M")
    monthly = pd.DataFrame({"symbol": ["A"] * 3 + ["B"] * 3, "month": list(months) * 2,
                            "ret": [np.nan, 0.1, 0.2, np.nan, 0.3, 0.5]})
    elig = pd.Series([True, True, True, True, False, True])   # B not eligible in Feb
    c = cmkt_monthly(monthly, elig)
    assert c[months[1]] == pytest.approx(0.2)                  # Jan eligibility: A and B
    assert c[months[2]] == pytest.approx(0.2)                  # Feb eligibility: A only


def _ivol_daily(seed=40, n_sym=4, start="2019-01-01", end="2020-03-31"):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    m = rng.normal(0, 0.02, len(dates))
    frames = []
    for k in range(n_sym):
        r = 0.001 + (0.5 + 0.3 * k) * m + rng.normal(0, 0.03 + 0.01 * k, len(dates))
        frames.append(pd.DataFrame({"symbol": f"S{k}", "date": dates, "close": 100 * np.cumprod(1 + r),
                                    "quote_volume": rng.lognormal(12, 1, len(dates))}))
    daily = pd.concat(frames, ignore_index=True)
    # S0 has a 20-day hole in 2019-06 and some zero-volume days
    daily = daily[~((daily.symbol == "S0") & daily.date.between("2019-06-05", "2019-06-24"))]
    daily.loc[daily.sample(frac=0.03, random_state=1).index, "quote_volume"] = 0.0
    return daily_returns(daily.reset_index(drop=True))


def test_ivol_matches_direct_ols():
    d = _ivol_daily()
    mk = d[d.valid].groupby("date")["r"].mean()
    np.testing.assert_allclose(daily_cmkt(d).to_numpy(), mk.to_numpy())
    iv = ivol_monthly(d, n_months=12, min_days=120).set_index(["symbol", "month"])["ivol"]
    for sym in ["S0", "S3"]:
        for end in ["2019-12", "2020-03", "2019-05"]:
            e = pd.Period(end, "M")
            g = d[(d.symbol == sym) & d.valid & (d.month > e - 12) & (d.month <= e)]
            x, y = g["date"].map(mk).to_numpy(), g["r"].to_numpy()
            X = np.column_stack([np.ones(len(x)), x])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ beta
            ref = np.sqrt(resid @ resid / (len(y) - 2))      # residual std, ddof = 2
            assert iv[(sym, e)] == pytest.approx(ref, rel=1e-7), (sym, end)
    # fewer than min_days valid days in the window -> NaN
    assert np.isnan(iv[("S0", pd.Period("2019-03", "M"))])     # ~ 89 valid days
    assert np.isfinite(iv[("S1", pd.Period("2019-05", "M"))])  # ~146 valid days


# ---------------------------------------------------------------- pandas-3 smoke test of the pipeline
def _random_daily(n_sym=60, start="2019-01-01", end="2022-12-31", seed=50):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    m = rng.normal(0, 0.03, len(dates))
    frames = []
    for k in range(n_sym):
        beg = rng.integers(0, 400) if k % 3 == 0 else 0          # staggered listings
        dd = dates[beg:]
        r = (0.5 + rng.uniform()) * m[beg:] + rng.normal(0, rng.uniform(0.02, 0.08), len(dd))
        qv = np.exp(rng.normal(np.log(10 ** rng.uniform(4.5, 8)), 1.0, len(dd)))
        df = pd.DataFrame({"symbol": f"C{k:02d}USDT", "date": dd, "close": 10 * np.cumprod(1 + np.clip(r, -0.9, None)),
                           "quote_volume": qv})
        frames.append(df.sample(frac=0.995, random_state=k).sort_values("date"))   # random gaps
    return pd.concat(frames, ignore_index=True)


def test_pipeline_smoke_pandas3():
    # regression guard: pd.concat of sort_table outputs used to fail in pandas 3 because
    # tab.attrs['portfolios'] is a DataFrame (fixed by analysis._no_attrs)
    import analysis
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    daily = _random_daily()
    panel, factors = analysis.build_panel(daily, cfg, "2020-01", "2022-12")
    assert len(panel) > 0 and panel["month"].nunique() == 36
    assert {"ret", "ret_adj", "lnA", "lnA_C", "ResLnA", "ResA", "mom", "ret_1", "ivol"} <= set(panel.columns)
    out = analysis.run_tests(panel, factors, cfg, full=True)
    for k in ["T3_sorts", "T4_fm", "T1_describe", "T1_corr", "T2_fm", "T5_imla", "U1_an_fm", "R_weighting", "sample"]:
        assert k in out and len(out[k]) > 0, k
    assert out["T4_fm"]["coef"].notna().all()
