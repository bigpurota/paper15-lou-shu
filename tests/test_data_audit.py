"""Audit rules on synthetic cases: redenomination (split-adjusted), ticker reuse (new asset),
long relisting gap, zero-volume and broken rows, and genuine crashes/pumps that must stay."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import data_audit as da
from measures import daily_returns, monthly_measures

CFG = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml").read_text())


def _coin(sym, closes, dvol=1e6, start="2021-01-01", dates=None, opens=None):
    """Kline-like daily rows: open = previous close (24/7 market), base volume = dvol / close."""
    closes = np.asarray(closes, float)
    dates = pd.date_range(start, periods=len(closes), freq="D") if dates is None else pd.DatetimeIndex(dates)
    dv = np.full(len(closes), dvol) if np.isscalar(dvol) else np.asarray(dvol, float)
    op = np.r_[closes[0], closes[:-1]] if opens is None else np.asarray(opens, float)
    return pd.DataFrame({"symbol": sym, "date": dates, "open": op, "high": np.maximum(op, closes) * 1.01,
                         "low": np.minimum(op, closes) * 0.99, "close": closes,
                         "volume": dv / closes, "quote_volume": dv})


def _walk(n, seed, s=0.03, p0=1.0):
    rng = np.random.default_rng(seed)
    return p0 * np.exp(np.cumsum(rng.normal(0, s, n)))


def _redenominated(F=1000.0, halt=4, seed=0):
    """60 days, halt of `halt` days, then trading reopens at prev_close / F (F new units per old)."""
    p = _walk(90, seed)
    keep = np.r_[np.arange(40), np.arange(40 + halt - 1, 90)]       # dates 40..40+halt-2 missing
    dates = pd.date_range("2021-01-01", periods=90, freq="D")[keep]
    closes = p[keep].copy()
    closes[40:] /= F                                                # after the halt: new units
    opens = np.r_[closes[0], closes[:-1]]
    opens[40] = closes[39] / F                                      # reopens at prev_close / F
    return _coin("REDUSDT", closes, dates=dates, opens=opens), p[keep]


def test_redenomination_detected_and_split_adjusted():
    raw, true_price = _redenominated(F=1000.0)
    ev = da.detect_events(raw, CFG)
    assert list(ev["type"]) == ["redenomination"]
    e = ev.iloc[0]
    assert e["factor"] == pytest.approx(1000.0) and e["action"] == "split_adjust"
    clean = da.apply_events(raw, ev)
    # adjusted prices are proportional to the true (continuous) price path, and the last
    # (post-event) prices are untouched
    ratio = clean["close"].to_numpy() / true_price
    np.testing.assert_allclose(ratio, ratio[-1], rtol=1e-12)
    np.testing.assert_allclose(clean["close"].iloc[-1], raw["close"].iloc[-1])
    np.testing.assert_allclose(clean["volume"] * clean["close"], raw["volume"] * raw["close"], rtol=1e-12)
    mret = monthly_measures(daily_returns(clean[["symbol", "date", "close", "quote_volume"]]))["ret"].dropna()
    assert mret.abs().max() < 1.0                                    # no +99 900% month any more


def test_reverse_split_and_non_exact_reopen():
    raw, _ = _redenominated(F=0.01, halt=9, seed=3)                  # 100 old -> 1 new
    raw.loc[40, ["open"]] *= 0.93                                    # market moved during the halt
    ev = da.detect_events(raw, CFG)
    assert list(ev["type"]) == ["redenomination"] and ev["factor"].iloc[0] == pytest.approx(0.01)


def test_round_jump_without_volume_evidence_is_not_spliced():
    raw, _ = _redenominated(F=1000.0)
    raw["volume"] = 1e6                                               # base volume does not move
    ev = da.detect_events(raw, CFG)
    assert list(ev["type"]) == ["discontinuity"] and ev["action"].iloc[0] == "new_asset"


def test_ticker_reuse_becomes_new_asset():
    old = _coin("LUNXUSDT", np.r_[_walk(30, 1, p0=80.0), np.geomspace(1.0, 5e-5, 5)])
    new = _coin("LUNXUSDT", _walk(40, 2, p0=5.0), start=str(old["date"].max() + pd.Timedelta(days=18)))
    raw = pd.concat([old, new], ignore_index=True)
    ev = da.detect_events(raw, CFG)
    assert list(ev["type"]) == ["ticker_reuse"]
    clean = da.apply_events(raw, ev)
    assert set(clean["symbol"]) == {"LUNXUSDT", "LUNXUSDT#2"}
    assert (clean["symbol"] == "LUNXUSDT#2").sum() == len(new)
    np.testing.assert_allclose(clean["close"].to_numpy(), raw["close"].to_numpy())   # no price change


def test_long_relisting_gap_without_jump_is_new_asset():
    a = _coin("RELUSDT", _walk(40, 4))
    b = _coin("RELUSDT", _walk(40, 5, p0=a["close"].iloc[-1]), start=str(a["date"].max() + pd.Timedelta(days=60)))
    ev = da.detect_events(pd.concat([a, b], ignore_index=True), CFG)
    assert list(ev["type"]) == ["long_gap"] and ev["action"].iloc[0] == "new_asset"


def test_genuine_crash_and_pump_are_kept():
    # LUNA-like: -99.99% over three days with continuous trading (open = previous close)
    crash = _coin("CRSHUSDT", np.r_[_walk(30, 6, p0=80.0), [17.0, 1.08, 3.2e-4, 5e-5], _walk(10, 7, p0=5e-5)],
                  dvol=np.r_[np.full(30, 1e8), [2e9, 4e9, 2e9, 1e8], np.full(10, 1e7)])
    pump = _coin("PUMPUSDT", np.r_[_walk(30, 8), [10.0, 12.0, 3.0], _walk(10, 9, p0=3.0)])
    # a short maintenance halt (2 days) with a normal move
    halt = _coin("HLTUSDT", _walk(50, 10))
    halt = halt.drop(index=[20, 21]).reset_index(drop=True)
    raw = pd.concat([crash, pump, halt], ignore_index=True)
    ev = da.detect_events(raw, CFG)
    assert ev.empty, ev
    clean = da.apply_events(raw, ev)
    pd.testing.assert_frame_equal(clean, raw.sort_values(["symbol", "date"]).reset_index(drop=True))


def test_zero_volume_and_broken_rows_dropped():
    c = _coin("ZVUSDT", _walk(40, 11))
    c.loc[[38, 39], ["quote_volume", "volume"]] = 0.0                 # frozen tail (delisting)
    c.loc[10, "close"] = c.loc[10, "high"] * 3                         # close above the high
    ev = da.detect_events(c, CFG)
    assert sorted(ev["type"]) == ["broken_print", "zero_volume"]
    assert ev.loc[ev["type"] == "zero_volume", "evidence"].iloc[0].startswith("2 day(s)")
    clean = da.apply_events(c, ev)
    assert len(clean) == len(c) - 3


def test_close_only_input_still_works():
    raw, _ = _redenominated(F=1000.0)
    ev = da.detect_events(raw[["symbol", "date", "close", "quote_volume"]], CFG)
    # without open/volume the reopen is the close; a round jump is accepted on price evidence
    assert list(ev["type"]) == ["redenomination"]


def test_multiple_events_compound():
    raw, true_price = _redenominated(F=10.0, seed=12)
    # a second 1:100 split after another 4-day halt
    last = raw["close"].iloc[-1]
    later_dates = pd.date_range(raw["date"].max() + pd.Timedelta(days=4), periods=30, freq="D")
    closes = last / 100 * _walk(30, 13, s=0.02)
    tail = _coin("REDUSDT", closes, dates=later_dates, opens=np.r_[last / 100, closes[:-1]])
    full = pd.concat([raw, tail], ignore_index=True)
    ev = da.detect_events(full, CFG)
    assert list(ev["type"]) == ["redenomination", "redenomination"]
    assert list(ev["factor"]) == pytest.approx([10.0, 100.0])
    clean = da.apply_events(full, ev)
    r = clean["close"].pct_change().abs()
    assert r.max() < 0.5
    np.testing.assert_allclose(clean["close"].iloc[0], raw["close"].iloc[0] / 1000.0)


def test_winsorized_returns_robustness_block():
    import analysis
    from test_estimators import _random_daily
    panel, factors = analysis.build_panel(_random_daily(n_sym=40), CFG, "2020-01", "2022-12")
    m0 = panel["month"].iloc[0]
    i = panel.index[panel["month"] == m0][0]
    panel.loc[i, "ret"] = 1e6                                           # one absurd coin-month
    pw = analysis.winsorize_returns(panel, (0.01, 0.99))
    sub = panel.loc[panel["month"] == m0, "ret"]
    assert pw.loc[i, "ret"] == pytest.approx(sub.quantile(0.99))
    other = panel["month"] != m0
    base = analysis.winsorize_returns(panel.drop(index=i), (0.01, 0.99))
    pd.testing.assert_series_equal(pw.loc[other, "ret"], base.loc[other[other].index, "ret"])  # within month
    out = analysis.run_tests(panel, factors, CFG, full=True)
    r = out["R_winsor_returns"]
    assert set(r["test"]) == {"T3", "T4"}
    assert set(r.loc[r["test"] == "T3", "signal"]) == {"A", "A_C", "ResA"}
    assert set(r.loc[r["test"] == "T4", "model"]) == {"(1)", "(2)", "(3)", "(4)", "(5)"}
    assert "R_winsor_returns" not in analysis.run_tests(panel, factors, CFG, full=False)
