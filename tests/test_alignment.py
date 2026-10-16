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


SIGNAL_FIELDS = ("A", "A_C", "ivol")                 # shifted by measures.lag_months in build_panel
OTHER_FIELDS = ("ret", "ret_1", "mom", "CMKT", "beta", "ret_adj")


def _daily_and_panel(lag: int | None = None):
    cfg = copy.deepcopy(CFG)
    cfg["measures"]["winsor"] = [0.0, 1.0]                 # compare raw measures
    if lag is not None:
        cfg["measures"]["lag_months"] = lag               # mutation: a look-ahead pipeline
    daily = _random_daily(n_sym=30)
    panel, _ = analysis.build_panel(daily, cfg, "2020-01", "2022-12")
    d = daily.sort_values(["symbol", "date"]).copy()
    d["month"] = d["date"].dt.to_period("M")
    d["r"] = d.groupby("symbol")["close"].pct_change()
    d.loc[d.groupby("symbol")["date"].diff() != pd.Timedelta(days=1), "r"] = np.nan
    return panel, d


@pytest.fixture(scope="module")
def built():
    return _daily_and_panel()


def _close(x, y, rtol, nan_ok):
    return bool(np.isclose(x, y, rtol=rtol, atol=0.0, equal_nan=nan_ok))


def recompute_check(panel, d, n_rows: int = 40) -> tuple[dict, dict]:
    """Independent recomputation (always with the PRE-REGISTERED timing: signal from month t-2) of
    randomly chosen panel rows.  Returns (mismatch counts per field, counts of non-NaN branches)."""
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
    bad = dict.fromkeys(SIGNAL_FIELDS + OTHER_FIELDS, 0)
    checked = {"beta": 0, "ivol": 0, "mom": 0}
    for i in rng.choice(len(panel), n_rows, replace=False):
        row = panel.iloc[i]
        s, t = row["symbol"], row["month"]
        g = v[(v.symbol == s) & (v.month == t - 2)]
        bad["A"] += not _close(row["A"], 1e6 * (g.r.abs() / g.quote_volume).mean(), 1e-10, False)
        bad["A_C"] += not _close(row["A_C"], 1e6 * (1 / g.quote_volume).mean(), 1e-10, False)
        bad["ret"] += not _close(row["ret"], 100 * mret(s, t), 1e-10, False)
        bad["ret_1"] += not _close(row["ret_1"], mret(s, t - 1), 1e-10, True)
        mm = [mret(s, t - k) for k in range(2, 13)]
        bad["mom"] += not _close(row["mom"], np.prod([1 + x for x in mm]) - 1, 1e-10, True)
        bad["CMKT"] += not _close(row["CMKT"], cm.get(t, np.nan), 1e-10, False)    # panel CMKT is a fraction
        xy = [(cm.get(t - k, np.nan), mret(s, t - k)) for k in range(1, 37)]
        xy = np.array([p for p in xy if np.all(np.isfinite(p))])
        b = np.polyfit(xy[:, 0], xy[:, 1], 1)[0] if len(xy) >= CFG["factors"]["beta_min_obs"] else np.nan
        bad["beta"] += not _close(row["beta"], b, 1e-8, True)
        bad["ret_adj"] += not _close(row["ret_adj"], 100 * (mret(s, t) - b * cm.get(t, np.nan)), 1e-8, True)
        w = v[(v.symbol == s) & (v.month >= t - 13) & (v.month <= t - 2)]
        if len(w) >= CFG["controls"]["ivol_min_days"]:
            X = np.column_stack([np.ones(len(w)), w["date"].map(mkt_d)])
            e = w["r"].to_numpy() - X @ np.linalg.lstsq(X, w["r"].to_numpy(), rcond=None)[0]
            iv = np.sqrt((e ** 2).sum() / (len(e) - 2))
        else:
            iv = np.nan
        bad["ivol"] += not _close(row["ivol"], iv, 1e-8, True)
        checked["beta"] += np.isfinite(b)
        checked["ivol"] += np.isfinite(iv)
        checked["mom"] += np.isfinite(row["mom"])
    return bad, checked


def test_build_panel_has_no_look_ahead(built):
    bad, checked = recompute_check(*built)
    assert not any(bad.values()), bad
    assert min(checked.values()) >= 10, checked        # the non-NaN branches were exercised


@pytest.mark.parametrize("lag", [1, 0])
def test_recompute_check_catches_look_ahead(lag):
    """Mutation test: the same check on a panel built with lag_months = 1 (or 0, i.e. same-month signal)
    must FAIL, and the failures must sit in the lagged signal fields, not in returns/controls."""
    assert CFG["measures"]["lag_months"] == 2                     # the pre-registered value is unchanged
    bad, _ = recompute_check(*_daily_and_panel(lag))
    assert bad["A"] == bad["A_C"] == 40, bad                      # every sampled row is flagged
    assert all(bad[f] == 0 for f in OTHER_FIELDS), bad            # ... and only through the shifted signal


def test_recompute_check_passes_with_explicit_lag_2():
    bad, _ = recompute_check(*_daily_and_panel(2))
    assert not any(bad.values()), bad


def test_controls_are_fractions_returns_percent(built):
    panel, _ = built
    assert panel["ret_1"].abs().median() < 1.0 < panel["ret"].abs().median()


def test_leveraged_token_filter_keeps_real_coins():
    listed = {"BTCUSDT", "BTCUPUSDT", "BTCDOWNUSDT", "BNBUSDT", "BNBBULLUSDT", "JUPUSDT", "SYRUPUSDT", "USDCUSDT"}
    keep = {s for s in listed if eligible_symbol(s, CFG, listed)}
    assert keep == {"BTCUSDT", "BNBUSDT", "JUPUSDT", "SYRUPUSDT"}
    assert eligible_symbol("JUPUSDT", CFG) is None      # without the listing: conservative (old rule)
