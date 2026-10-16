"""Known-answer tests for the MOEX upgrade (data_moex.py, run_moex.py) on toy data; no network."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

import data_moex as dm
import run_moex as rm

SPEC = dm.DATA_SPEC


def _cal(start, end, holidays=()):
    cal = pd.bdate_range(start, end)
    return cal[~cal.isin(pd.to_datetime(list(holidays)))]


# ---------------------------------------------------------------- ex-dividend dates
def test_ex_date_t2_regime_with_holiday():
    # SBER 2017: registry 14.06.2017 (Wed), 12.06 holiday -> last cum 09.06 (Fri), ex 13.06 (Tue)
    cal = _cal("2017-06-01", "2017-06-30", ["2017-06-12"])
    d = pd.DataFrame({"secid": ["SBER"], "registry_date": [pd.Timestamp("2017-06-14")],
                      "last_cum_date": [pd.NaT], "amount": [6.0]})
    assert dm.ex_dates(d, cal, SPEC["t1_from"]).iloc[0] == pd.Timestamp("2017-06-13")


def test_ex_date_t1_regime_and_weekend_registry():
    cal = _cal("2025-07-01", "2025-07-31")
    d = pd.DataFrame({"secid": ["SBER", "X"],
                      "registry_date": [pd.Timestamp("2025-07-18"), pd.Timestamp("2025-07-20")],  # Fri, Sun
                      "last_cum_date": [pd.NaT, pd.NaT], "amount": [34.84, 1.0]})
    ex = dm.ex_dates(d, cal, SPEC["t1_from"])
    assert ex.iloc[0] == pd.Timestamp("2025-07-18")      # last cum 17.07 (smart-lab T-1), ex = R
    assert ex.iloc[1] == pd.Timestamp("2025-07-18")      # Sunday R: buy Thu settles Fri <= R, ex Fri


def test_ex_date_from_given_last_cum_day():
    cal = _cal("2016-10-01", "2016-10-31")
    d = pd.DataFrame({"secid": ["MTSS"], "registry_date": [pd.Timestamp("2016-10-17")],
                      "last_cum_date": [pd.Timestamp("2016-10-14")], "amount": [11.99]})   # Fri
    assert dm.ex_dates(d, cal, SPEC["t1_from"]).iloc[0] == pd.Timestamp("2016-10-17")


# ---------------------------------------------------------------- daily construction
def _hist(secid, dates, close, value=1e6, volume=1000):
    n = len(dates)
    return pd.DataFrame({"secid": secid, "date": pd.to_datetime(dates), "close": close,
                         "value": np.broadcast_to(value, n).astype(float),
                         "volume": np.broadcast_to(volume, n).astype(float), "numtrades": 10})


def _no_splits():
    return pd.DataFrame(columns=["secid", "date", "before", "after"])


def _no_divs():
    return pd.DataFrame(columns=["secid", "ex_date", "amount"])


def test_dividend_added_back_on_ex_date():
    dates = ["2019-07-01", "2019-07-02", "2019-07-03"]
    h = _hist("AAA", dates, [100.0, 90.0, 91.0])
    divs = pd.DataFrame({"secid": ["AAA"], "ex_date": [pd.Timestamp("2019-07-02")], "amount": [10.0]})
    d, flags = dm.build_daily(h, _no_splits(), divs)
    assert d["r"].iloc[1] == pytest.approx(0.0)           # (90 + 10) / 100 - 1
    assert d["r"].iloc[2] == pytest.approx(91 / 90 - 1)
    assert d["tri"].iloc[-1] == pytest.approx(100 / 100 * 91 / 90)
    assert flags.empty


def test_dividend_on_non_trading_ex_date_goes_to_next_trade():
    cal_days = ["2019-07-01", "2019-07-02", "2019-07-03"]
    other = _hist("ZZZ", cal_days, [1.0, 1.0, 1.0])                # defines the market calendar
    h = pd.concat([other, _hist("AAA", ["2019-07-01", "2019-07-03"], [100.0, 95.0])])
    divs = pd.DataFrame({"secid": ["AAA"], "ex_date": [pd.Timestamp("2019-07-02")], "amount": [5.0]})
    d, _ = dm.build_daily(h, _no_splits(), divs)
    a = d[d["symbol"] == "AAA"]
    assert a["div"].iloc[1] == 5.0 and a["gross"].iloc[1] == pytest.approx(1.0)
    assert np.isnan(a["r"].iloc[1])                       # previous trade was not the previous market day


def test_reverse_split_adjusts_return_and_volume():
    # VTBR-like 5000:1 consolidation on 2024-07-15
    dates = ["2024-07-11", "2024-07-12", "2024-07-15", "2024-07-16"]
    h = _hist("VTB", dates, [0.02, 0.02, 101.0, 102.0], volume=[5e9, 5e9, 1e6, 1e6])
    sp = pd.DataFrame({"secid": ["VTB"], "date": [pd.Timestamp("2024-07-15")], "before": [5000], "after": [1]})
    d, flags = dm.build_daily(h, sp, _no_divs())
    assert d["r"].iloc[2] == pytest.approx(101.0 / (0.02 * 5000) - 1)
    assert d["volume_adj"].iloc[0] == pytest.approx(1e6)    # 5e9 old shares = 1e6 new shares
    assert d["volume_adj"].iloc[3] == pytest.approx(1e6)
    assert flags.empty


def test_unrecorded_jump_breaks_chain_and_is_flagged():
    dates = pd.bdate_range("2019-01-01", "2019-03-29")
    close = np.full(len(dates), 10.0)
    close[dates >= "2019-02-15"] = 1000.0                  # x100 jump without a recorded split
    d, flags = dm.build_daily(_hist("J", dates, close), _no_splits(), _no_divs())
    assert (flags["kind"] == "unexplained_jump").sum() == 1
    m = dm.monthly_tri_returns(d).set_index("month")["ret"]
    assert np.isnan(m[pd.Period("2019-02", "M")])            # Feb spans the break
    assert m[pd.Period("2019-03", "M")] == pytest.approx(0.0)  # Mar is inside the new segment


def test_halt_gap_is_missing_daily_but_monthly_return_kept():
    days = list(pd.bdate_range("2022-02-01", "2022-02-25")) + list(pd.bdate_range("2022-03-24", "2022-04-29"))
    close = np.where(pd.DatetimeIndex(days) < "2022-03-01", 100.0, 60.0)
    d, _ = dm.build_daily(_hist("H", days, close), _no_splits(), _no_divs())
    first_after = d[d["date"] == pd.Timestamp("2022-03-24")]
    assert np.isnan(first_after["r"].iloc[0]) and not first_after["valid"].iloc[0]
    assert d.loc[d["date"] == pd.Timestamp("2022-03-25"), "valid"].iloc[0]
    m = dm.monthly_tri_returns(d).set_index("month")["ret"]
    assert m[pd.Period("2022-03", "M")] == pytest.approx(-0.4)


def test_monthly_tri_returns_known_answer():
    dates = ["2020-01-30", "2020-01-31", "2020-02-27", "2020-02-28", "2020-04-30"]
    d, _ = dm.build_daily(_hist("M", dates, [10.0, 11.0, 12.0, 12.1, 13.0]), _no_splits(), _no_divs())
    m = dm.monthly_tri_returns(d).set_index("month")["ret"]
    assert np.isnan(m[pd.Period("2020-01", "M")])
    assert m[pd.Period("2020-02", "M")] == pytest.approx(12.1 / 11 - 1)
    assert np.isnan(m[pd.Period("2020-04", "M")])          # March missing -> not consecutive


# ---------------------------------------------------------------- parsers
def test_parse_dohod_and_smartlab():
    dohod = """<table><tr><th>Дата объявления дивиденда</th><th>Дата закрытия реестра</th>
      <th>Год для учета дивиденда</th><th>Дивиденд</th></tr>
      <tr><td>n/a</td><td>20.07.2027 (прогноз)</td><td>n/a</td><td>44.53</td></tr>
      <tr><td>21.04.2026</td><td>20.07.2026</td><td>2026</td><td>37.64</td></tr></table>"""
    a = dm.parse_dohod(dohod, "SBER")
    assert len(a) == 1 and a["amount"].iloc[0] == 37.64 and a["registry_date"].iloc[0] == pd.Timestamp("2026-07-20")
    sl = """<table><tr><td>Выплаченные</td></tr><tr><th>Тикер</th><th>дата T-1</th><th>дата отсечки</th>
      <th>Период</th><th>дивиденд</th></tr>
      <tr><td>SBER</td><td>17.07.2025</td><td>18.07.2025</td><td>2024 год</td><td>34,84₽</td></tr>
      <tr><td>SBERP</td><td>17.07.2025</td><td>18.07.2025</td><td>2024 год</td><td>34,84₽</td></tr></table>"""
    b = dm.parse_smartlab(sl, "SBER")
    assert len(b) == 1 and b["amount"].iloc[0] == pytest.approx(34.84)
    assert b["last_cum_date"].iloc[0] == pd.Timestamp("2025-07-17")


def test_parse_smartlab_reads_paid_table_after_expected_table():
    # real smart-lab layout (e.g. LKOH): 'expected' table first, then 'paid' table with the same header
    hdr = "<tr><th>Тикер</th><th>дата T-1</th><th>дата отсечки</th><th>Период</th><th>дивиденд</th></tr>"
    sl = (f"<table><tr><td>Ожидаемые дивиденды</td></tr>{hdr}"
          "<tr><td>LKOH</td><td>08.12.2026</td><td>09.12.2026</td><td>3кв 2026</td><td>535₽</td></tr></table>"
          f"<table><tr><td>Выплаченные дивиденды</td></tr>{hdr}"
          "<tr><td>LKOH</td><td>08.12.2026</td><td>09.12.2026</td><td>3кв 2026</td><td>535₽</td></tr>"
          "<tr><td>LKOH</td><td>30.04.2026</td><td>04.05.2026</td><td>4кв 2025</td><td>278₽</td></tr>"
          "<tr><td>LKOH</td><td>11.12.2025</td><td>12.12.2025</td><td>3кв 2025</td><td>397₽</td></tr></table>")
    b = dm.parse_smartlab(sl, "LKOH")
    assert sorted(b["amount"]) == [278.0, 397.0, 535.0]              # paid history kept; repeat across tables once


# ---------------------------------------------------------------- panel wiring (synthetic market)
def _synthetic(n=130, seed=0):
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range("2014-01-01", "2018-12-31")
    mkt = rng.normal(0, 0.01, len(cal))
    rows, info = [], []
    for i in range(n):
        s = f"S{i:03d}"
        r = 0.8 * mkt + rng.normal(0, 0.02, len(cal))
        close = 100 * np.exp(np.cumsum(r))
        value = np.exp(rng.normal(14 + 2 * rng.standard_normal(), 0.5, len(cal)))
        keep = rng.random(len(cal)) > 0.02
        rows.append(pd.DataFrame({"secid": s, "date": cal[keep], "close": close[keep], "value": value[keep],
                                  # issuesize = 1 and volume = value -> TO = VALUE, AT family = A family / 1e6
                                  "volume": value[keep], "numtrades": 5}))
        info.append({"secid": s, "type": "preferred_share" if i < 5 else "common_share", "issuesize": 1.0})
    hist = pd.concat(rows, ignore_index=True)
    daily, _ = dm.build_daily(hist, _no_splits(), _no_divs())
    return daily, pd.DataFrame(info)


@pytest.fixture(scope="module")
def synth():
    daily, info = _synthetic()
    spec = copy.deepcopy(rm.MOEX_SPEC)
    spec["period"]["return_end"] = "2018-12"
    spec["universe"]["min_stocks_first_month"] = 100
    spec["factors"]["beta_min_obs"] = 12
    monthly = rm.monthly_frames(daily, info, spec)
    panel, factors, meta = rm.build_moex_panel(monthly, daily, info, spec, 1e5)
    return daily, info, spec, monthly, panel, factors, meta


def test_turnover_family_equals_volume_family_when_to_is_value(synth):
    _, _, _, _, panel, _, _ = synth
    assert np.allclose(panel["AT"], panel["A"] / 1e6, rtol=1e-9, equal_nan=True)
    assert np.allclose(panel["ResLnAT"], panel["ResLnA"], atol=1e-9, equal_nan=True)


def test_preferred_excluded_and_lag(synth):
    daily, _, _, monthly, panel, _, meta = synth
    assert not panel["symbol"].isin([f"S{i:03d}" for i in range(5)]).any()
    # A of month t in the panel equals winsor-free A of month t-2 for an interior observation
    row = panel.dropna(subset=["A"]).iloc[len(panel) // 2]
    src = monthly[(monthly["symbol"] == row["symbol"]) & (monthly["month"] == row["month"] - 2)]
    lo, hi = panel[panel["month"] == row["month"]]["A"].min(), panel[panel["month"] == row["month"]]["A"].max()
    assert np.clip(src["A"].iloc[0], lo, hi) == pytest.approx(row["A"])
    assert pd.Period(meta["return_start"], "M") == pd.Period(meta["first_measure_month"], "M") + 2


def test_me_uses_previous_december(synth):
    daily, _, _, monthly, _, _, _ = synth
    s = "S050"
    dec = daily[(daily["symbol"] == s) & (pd.to_datetime(daily["date"]).dt.strftime("%Y-%m") == "2015-12")]
    me = monthly[(monthly["symbol"] == s) & (monthly["month"] == pd.Period("2016-06", "M"))]["ME"].iloc[0]
    assert me == pytest.approx(dec["close"].iloc[-1] * 1.0 / 1e6)


def test_tests_run_and_paper_table(synth, tmp_path):
    _, _, spec, _, panel, factors, _ = synth
    t3 = rm.sorts(panel, factors, 5, 6)
    t4 = rm.fm(panel, "ret_adj", 6)
    assert set(t3["signal"]) == {"A", "A_C", "ResA", "AT", "AT_C", "ResAT"}
    assert {"lnME", "mom", "ret_1"} <= set(t4["var"])
    # identical families -> identical FM slopes on the Amihud variables
    a = t4[(t4["family"] == "A (RUB volume)") & (t4["model"] == "(4)")].set_index("var")
    b = t4[(t4["family"] == "AT (turnover)") & (t4["model"] == "(4)")].set_index("var")
    assert a.loc["ResLnA", "coef"] == pytest.approx(b.loc["ResLnAT", "coef"])
    import yaml
    paper = yaml.safe_load((rm.SRC / "paper_numbers.yaml").read_text())
    pv = rm.paper_vs_moex(rm.describe(panel), rm.mean_cs_corr(panel, ["A", "AT", "A_C", "AT_C", "absret"], "month")
                          .reset_index(names="measure"), t3.assign(variant="main"), t4, paper, 1.96, 3.0)
    assert pv["moex"].notna().all() and len(pv) > 40
    assert pv.loc[pv["item"] == "corr(A, A_C)", "moex"].iloc[0] == pytest.approx(
        pv.loc[pv["item"] == "corr(AT, AT_C)", "moex"].iloc[0])


def test_ex_date_check_slope_one_for_true_dividends():
    rng = np.random.default_rng(1)
    rows, divs = [], []
    cal = pd.bdate_range("2019-01-01", "2019-12-31")
    for i in range(40):
        p = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(cal))))
        k = 100 + i
        D = p[k - 1] * rng.uniform(0.02, 0.15)
        p[k:] -= D * p[k:] / p[k]                       # price drops by D on the ex-date
        rows.append(_hist(f"D{i}", cal, p))
        divs.append({"secid": f"D{i}", "ex_date": cal[k], "amount": D})
    d, _ = dm.build_daily(pd.concat(rows), _no_splits(), pd.DataFrame(divs))
    chk = rm.ex_date_check(d).set_index("day")
    assert chk.loc["ex-date", "slope_price_return_on_minus_yield"] == pytest.approx(1.0, abs=0.15)
    assert abs(chk.loc["day before ex-date", "slope_price_return_on_minus_yield"]) < 0.5


def test_crosscheck_dividends_from_cached_pages(tmp_path, monkeypatch):
    monkeypatch.setattr(dm, "RAW", tmp_path)
    (tmp_path / "dividends" / "dohod").mkdir(parents=True)
    (tmp_path / "dividends" / "smartlab").mkdir(parents=True)
    (tmp_path / "dividends" / "dohod" / "SBER.html").write_text(
        "<table><tr><th>Дата объявления дивиденда</th><th>Дата закрытия реестра</th><th>Год</th><th>Дивиденд</th></tr>"
        "<tr><td>1</td><td>18.07.2025</td><td>2025</td><td>34.84</td></tr>"
        "<tr><td>1</td><td>14.06.2017</td><td>2017</td><td>6</td></tr></table>", encoding="utf-8")
    (tmp_path / "dividends" / "smartlab" / "SBER.html").write_text(
        "<table><tr><th>Тикер</th><th>дата T-1</th><th>дата отсечки</th><th>Период</th><th>дивиденд</th></tr>"
        "<tr><td>SBER</td><td>17.07.2025</td><td>18.07.2025</td><td>2024</td><td>34,84₽</td></tr>"
        "<tr><td>SBER</td><td>09.06.2017</td><td>14.06.2017</td><td>2016</td><td>6₽</td></tr></table>", encoding="utf-8")
    cal = _cal("2017-01-01", "2025-12-31", ["2017-06-12"])
    ev, summ = dm.crosscheck_dividends(["SBER"], cal)
    assert summ["events_matched"].iloc[0] == 2 and summ["share_ex_date_equal"].iloc[0] == 1.0


def test_main_end_to_end_on_synthetic(tmp_path, monkeypatch):
    import json
    import yaml
    daily, info = _synthetic(n=130, seed=3)
    hist_rows = daily.rename(columns={"symbol": "secid"})[["secid", "date", "close", "value", "volume"]]
    frames = {
        "universe": pd.DataFrame({"month": ["2015-01"] * len(info), "snap_date": "2015-01-30",
                                  "secid": info["secid"], "numtrades": 1, "value": 1.0}),
        "info": info, "history": hist_rows.assign(numtrades=1), "splits": _no_splits(),
        "dividends": pd.DataFrame(columns=["secid", "registry_date", "last_cum_date", "amount", "source", "ex_date"]),
        "dividend_pages": pd.DataFrame({"secid": info["secid"], "dohod": False, "smartlab": False}),
        "daily": daily, "flags": pd.DataFrame(columns=["secid", "date", "kind", "value"])}
    monkeypatch.setattr(dm, "load_or_build", lambda spec, workers=6, rebuild=False: frames)
    monkeypatch.setattr(dm, "RAW", tmp_path / "raw")
    cfg = {"seed": 1, "moex": {"period": {"return_end": "2018-12"}, "factors": {"beta_min_obs": 12},
                               "universe": {"min_stocks_first_month": 100}}}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg))
    res = tmp_path / "results"
    res.mkdir()
    (res / "LATEST").write_text("BASE_RUN_X\n")
    out = res / "upgrades" / "moex"
    run_id = rm.main([str(tmp_path / "config.yaml"), "--out", str(out), "--results-dir", str(res),
                      "--cache-dir", str(tmp_path / "cache")])
    man = json.loads((out / "run_manifest.json").read_text())
    assert man["run_id"] == run_id and man["base_run_id"] == "BASE_RUN_X"
    for k in ["git", "config_sha256", "spec_sha256", "code_sha256", "seed", "timings_sec"]:
        assert k in man
    for t in man["tables"]:
        assert (out / t).exists(), t
    s = (out / "SUMMARY.md").read_text()
    assert "MOEX" in s and "BASE_RUN_X" in s
    v = pd.read_csv(out / "variants_key.csv")
    base = ["main", "filter 1M RUB", "no dividends (price returns)", "main without M9 corrections"]
    assert set(v["variant"]) == set(base) | {f"{b}, terciles" for b in base} | {
        "main, raw returns", "diagnostic: raw returns, months of ret_adj FM"}
    assert v.set_index("variant").loc["main, raw returns", "HL_A_C"] != v.set_index("variant").loc["main, raw returns", "HL_A_C"]  # NaN: FM-only
    assert "t > 3" in s and "diagnostic" in s


def test_me_is_split_adjusted_to_current_issuesize():
    # 1:100 split in 2024 (GMKN-like): Dec-2023 price 15000 -> 150 in today's units; ISSUESIZE is today's
    days = list(pd.bdate_range("2023-12-01", "2023-12-29")) + list(pd.bdate_range("2024-04-01", "2024-04-30"))
    close = np.where(pd.DatetimeIndex(days) < "2024-04-08", 15000.0, 150.0)
    sp = pd.DataFrame({"secid": ["G"], "date": [pd.Timestamp("2024-04-08")], "before": [1], "after": [100]})
    d, _ = dm.build_daily(_hist("G", days, close), sp, _no_divs())
    assert d["r"][d["date"] == pd.Timestamp("2024-04-08")].iloc[0] == pytest.approx(0.0)
    info = pd.DataFrame({"secid": ["G"], "type": ["common_share"], "issuesize": [1.0e8]})
    m = rm.monthly_frames(d, info, rm.MOEX_SPEC)
    me = m.loc[m["month"] == pd.Period("2024-04", "M"), "ME"].iloc[0]
    assert me == pytest.approx(150.0 * 1.0e8 / 1e6)


def test_split_adjusted_dividend_converted_to_contemporaneous_units():
    # GMKN-like: dividend 2.48 quoted in today's units (after a later 1:100 split) = 248 per old share
    days = list(pd.bdate_range("2019-06-03", "2019-06-07")) + list(pd.bdate_range("2024-04-08", "2024-04-12"))
    close = np.array([15000.0, 15000.0, 14752.0, 14752.0, 14752.0, 150.0, 150.0, 150.0, 150.0, 150.0])
    sp = pd.DataFrame({"secid": ["G"], "date": [pd.Timestamp("2024-04-08")], "before": [1], "after": [100]})
    divs = pd.DataFrame({"secid": ["G"], "ex_date": [pd.Timestamp("2019-06-05")], "amount": [2.48]})
    d, _ = dm.build_daily(_hist("G", days, close), sp, divs)
    assert d["div"].iloc[2] == pytest.approx(248.0)
    assert d["r"].iloc[2] == pytest.approx(0.0)


def test_infer_unrecorded_consolidation_but_not_a_crash():
    # IRAO-like 100:1 consolidation (price x107, share volume / 100) vs a crash (price / 11, volume flat)
    d1 = pd.bdate_range("2014-11-03", "2014-12-18")
    d2 = pd.bdate_range("2015-01-20", "2015-03-02")
    days = d1.append(d2)
    n1, n2 = len(d1), len(d2)
    irao = _hist("IRAO", days, np.r_[np.full(n1, 0.00712), np.full(n2, 0.7641)],
                 volume=np.r_[np.full(n1, 1.0e10), np.full(n2, 1.0e8)])
    crash = _hist("RGSS", days, np.r_[np.full(n1, 5.67), np.full(n2, 0.52)], volume=1.0e6)
    hist = pd.concat([irao, crash], ignore_index=True)
    inf = dm.infer_splits(hist, _no_splits())
    assert list(inf["secid"]) == ["IRAO"] and inf["before"].iloc[0] == 100 and inf["after"].iloc[0] == 1
    d, flags = dm.build_daily(hist, _no_splits(), _no_divs())
    i = d[(d["symbol"] == "IRAO") & (d["date"] == d2[0])]
    assert i["gross"].iloc[0] == pytest.approx(0.7641 / (0.00712 * 100))
    assert d.loc[d["symbol"] == "IRAO", "volume_adj"].iloc[0] == pytest.approx(1.0e8)
    assert set(flags["kind"]) == {"inferred_split", "unexplained_jump"}
    assert flags.loc[flags["kind"] == "unexplained_jump", "secid"].tolist() == ["RGSS"]


def test_verified_split_and_large_dividend():
    # BELU-like 1:8 split across a short suspension: not a power of 10, inside the x8 guard -> only the
    # verified list fixes it; LNZL-like 65% dividend is dropped by the guard unless verified
    days = pd.bdate_range("2024-08-01", "2024-08-30")
    close = np.where(days < "2024-08-22", 4680.0, 585.0)
    h = _hist("B8", days, close)
    spec = copy.deepcopy(SPEC)
    spec["verified_splits"] = [{"secid": "B8", "date": "2024-08-22", "before": 1, "after": 8, "source": "toy"}]
    d0, _ = dm.build_daily(h, _no_splits(), _no_divs(), {**spec, "verified_splits": []})
    d1, fl = dm.build_daily(h, _no_splits(), _no_divs(), spec)
    assert d0["tri"].iloc[-1] == pytest.approx(585.0 / 4680.0)
    assert d1["tri"].iloc[-1] == pytest.approx(1.0)
    assert d1["vol_mult"].iloc[0] == 8.0 and "verified_split" in set(fl["kind"])
    dd = ["2021-07-08", "2021-07-09", "2021-07-12"]
    hl = _hist("L", dd, [23370.0, 23370.0, 8150.5])
    divs = pd.DataFrame({"secid": ["L"], "registry_date": [pd.Timestamp("2021-07-13")],
                         "ex_date": [pd.Timestamp("2021-07-12")], "amount": [15219.5]})
    spec["verified_large_dividends"] = []
    _, fl0 = dm.build_daily(hl, _no_splits(), divs, spec)
    assert fl0["kind"].tolist() == ["dividend_dropped"]
    spec["verified_large_dividends"] = [{"secid": "L", "registry_date": "2021-07-13", "source": "toy"}]
    d2, fl2 = dm.build_daily(hl, _no_splits(), divs, spec)
    assert d2["r"].iloc[2] == pytest.approx(0.0) and fl2["kind"].tolist() == ["verified_large_dividend"]
