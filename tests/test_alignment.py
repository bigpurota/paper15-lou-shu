"""Look-ahead guard: every lagged quantity in build_panel is recomputed independently from the
raw closes and volumes for randomly chosen (symbol, month t) cells.

A, A_C from month t-2; Ret[-1] = month t-1; Ret[-12,-2] = 11 months t-12..t-2 (fractions);
beta on [t-36, t-1] vs EW CMKT whose membership is eligibility in month m-1;
IVOL from daily OLS over months t-13..t-2 (ddof 2), at least 120 days.
"""
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import analysis
from data_binance import eligible_symbol
from test_estimators import _random_daily

CFG = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml").read_text())


@pytest.fixture(scope="module")
def built():
    cfg = copy.deepcopy(CFG)
    cfg["measures"]["winsor"] = [0.0, 1.0]                 # compare raw measures
    daily = _random_daily(n_sym=30)
    panel, _ = analysis.build_panel(daily, cfg, "2020-01", "2022-12")
    d = daily.sort_values(["symbol", "date"]).copy()
    d["month"] = d["date"].dt.to_period("M")
    d["r"] = d.groupby("symbol")["close"].pct_change()
    d.loc[d.groupby("symbol")["date"].diff() != pd.Timedelta(days=1), "r"] = np.nan
    return panel, d


def test_build_panel_has_no_look_ahead(built):
    panel, d = built
    ce = d.groupby(["symbol", "month"])["close"].last()
    mret = lambda s, m: ce[(s, m)] / ce[(s, m - 1)] - 1 if (s, m) in ce.index and (s, m - 1) in ce.index else np.nan
    v = d[d["r"].notna() & (d["quote_volume"] > 0)]
    st = v.groupby(["symbol", "month"]).agg(n=("r", "size"), dv=("quote_volume", "mean"))
    elig = set(st[(st.n >= CFG["universe"]["min_valid_days"]) & (st.dv >= CFG["universe"]["min_mean_dvol_usd"])].index)
    syms = sorted(d["symbol"].unique())
    cm = {}
    for m in sorted(d["month"].unique()):
        rs = [x for x in (mret(s, m) for s in syms if (s, m - 1) in elig) if np.isfinite(x)]
        if rs:
            cm[m] = np.mean(rs)
    mkt_d = v.groupby("date")["r"].mean()

    rng = np.random.default_rng(1)
    checked = {"beta": 0, "ivol": 0, "mom": 0}
    for i in rng.choice(len(panel), 40, replace=False):
        row = panel.iloc[i]
        s, t = row["symbol"], row["month"]
        g = v[(v.symbol == s) & (v.month == t - 2)]
        assert row["A"] == pytest.approx(1e6 * (g.r.abs() / g.quote_volume).mean(), rel=1e-10)
        assert row["A_C"] == pytest.approx(1e6 * (1 / g.quote_volume).mean(), rel=1e-10)
        assert row["ret"] == pytest.approx(100 * mret(s, t), rel=1e-10)
        np.testing.assert_allclose(row["ret_1"], mret(s, t - 1), rtol=1e-10)
        mm = [mret(s, t - k) for k in range(2, 13)]
        np.testing.assert_allclose(row["mom"], np.prod([1 + x for x in mm]) - 1, rtol=1e-10)
        assert row["CMKT"] == pytest.approx(cm[t], rel=1e-10)         # panel CMKT is a fraction
        xy = [(cm.get(t - k, np.nan), mret(s, t - k)) for k in range(1, 37)]
        xy = np.array([p for p in xy if np.all(np.isfinite(p))])
        b = np.polyfit(xy[:, 0], xy[:, 1], 1)[0] if len(xy) >= CFG["factors"]["beta_min_obs"] else np.nan
        np.testing.assert_allclose(row["beta"], b, rtol=1e-8)
        np.testing.assert_allclose(row["ret_adj"], 100 * (mret(s, t) - b * cm[t]), rtol=1e-8)
        w = v[(v.symbol == s) & (v.month >= t - 13) & (v.month <= t - 2)]
        if len(w) >= CFG["controls"]["ivol_min_days"]:
            X = np.column_stack([np.ones(len(w)), w["date"].map(mkt_d)])
            e = w["r"].to_numpy() - X @ np.linalg.lstsq(X, w["r"].to_numpy(), rcond=None)[0]
            iv = np.sqrt((e ** 2).sum() / (len(e) - 2))
        else:
            iv = np.nan
        np.testing.assert_allclose(row["ivol"], iv, rtol=1e-8)
        checked["beta"] += np.isfinite(b)
        checked["ivol"] += np.isfinite(iv)
        checked["mom"] += np.isfinite(row["mom"])
    assert min(checked.values()) >= 10, checked        # the non-NaN branches were exercised


def test_controls_are_fractions_returns_percent(built):
    panel, _ = built
    assert panel["ret_1"].abs().median() < 1.0 < panel["ret"].abs().median()


def test_leveraged_token_filter_keeps_real_coins():
    listed = {"BTCUSDT", "BTCUPUSDT", "BTCDOWNUSDT", "BNBUSDT", "BNBBULLUSDT", "JUPUSDT", "SYRUPUSDT", "USDCUSDT"}
    keep = {s for s in listed if eligible_symbol(s, CFG, listed)}
    assert keep == {"BTCUSDT", "BNBUSDT", "JUPUSDT", "SYRUPUSDT"}
    assert eligible_symbol("JUPUSDT", CFG) is None      # without the listing: conservative (old rule)
