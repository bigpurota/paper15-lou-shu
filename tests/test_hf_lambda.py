"""Known-answer tests for src/hf_lambda.py (upgrade 2: high-frequency lambda with exact trade signs)."""
from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import pytest

import hf_lambda as H

BAR = H.BAR_MS


# ------------------------------------------------------------------ helpers
def _zip(path, name, text):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(name, text)
    return path


def _simulate_trades(lam, n_bars=3000, rate=6.0, noise=2e-4, seed=0, t0_ms=1_717_200_000_000, us=False):
    """Trades with exact signs; bar return = lam * sum_k sign_k sqrt(dvol_k) + noise; price walks."""
    rng = np.random.default_rng(seed)
    rows, price, bars = [], 1.0, []
    aid = 0
    for n in range(n_bars):
        k = rng.poisson(rate)
        signs = rng.choice([-1.0, 1.0], size=k)
        dv = np.exp(rng.normal(6.0, 1.0, size=k))                      # dollar volume per trade
        sv = float((signs * np.sqrt(dv)).sum())
        r = lam * sv + noise * rng.standard_normal()
        start = t0_ms + n * BAR
        # all trades of the bar print at the end-of-bar price except the last, which defines the close
        p_end = price * np.exp(r)
        for j in range(k):
            ts = start + 1 + j * (BAR - 2) // max(k, 1)
            rows.append((aid, p_end, dv[j] / p_end, aid, aid, ts * (1000 if us else 1), signs[j] < 0, True))
            aid += 1
        if k:
            price = p_end
        bars.append({"bar": start // BAR, "svol": sv, "r": r if k else 0.0, "k": k,
                     "Q": float(dv.sum()), "B": float(dv[signs > 0].sum()), "close": price})
    trades = pd.DataFrame(rows, columns=["id", "p", "q", "f", "l", "ts", "bm", "best"])
    return trades, pd.DataFrame(bars)


def _trades_csv(trades: pd.DataFrame, header: bool = False) -> str:
    t = trades.copy()
    t["bm"] = t["bm"].map({True: "True", False: "False"})
    t["best"] = "True"
    buf = io.StringIO()
    if header:
        buf.write("agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker,is_best_match\n")
    t.to_csv(buf, header=False, index=False)
    return buf.getvalue()


def _klines_csv(bars: pd.DataFrame, us: bool = False, header: bool = False) -> str:
    lines = []
    if header:
        lines.append("open_time,open,high,low,close,volume,close_time,quote_volume,count,"
                     "taker_buy_volume,taker_buy_quote_volume,ignore")
    for b in bars.itertuples():
        ot = b.bar * BAR * (1000 if us else 1)
        lines.append(f"{ot},{b.close},{b.close},{b.close},{b.close},0,{ot + BAR - 1},{b.Q},{b.k},0,{b.B},0")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ estimators
def test_svol_bar_known_values():
    B, Q = np.array([4.0, 9.0, 0.0, 5.0]), np.array([13.0, 9.0, 16.0, 10.0])
    np.testing.assert_allclose(H.svol_bar(B, Q, "sqrt_sides"), [2 - 3, 3, -4, 0])
    np.testing.assert_allclose(H.svol_bar(B, Q, "net_sqrt"), [-np.sqrt(5), 3, -4, 0])
    # one trade per side or one-sided bars: sqrt_sides equals the trade-level sum exactly
    np.testing.assert_allclose(H.svol_bar([9.0], [9.0], "sqrt_sides"), [3.0])
    with pytest.raises(ValueError):
        H.svol_bar(B, Q, "bogus")


def test_svol_bar_clips_inconsistent_taker_volume():
    # B > Q (bad print) -> S clipped at 0, no NaN from sqrt of negative
    out = H.svol_bar([12.0], [10.0], "sqrt_sides")
    assert np.isfinite(out).all() and out[0] == pytest.approx(np.sqrt(12.0))


def test_bar_returns_requires_consecutive_bars():
    r = H.bar_returns(np.array([10, 11, 13, 14]), np.array([1.0, 1.1, 1.2, 0.6]))
    assert np.isnan(r[0]) and np.isnan(r[2])
    assert r[1] == pytest.approx(np.log(1.1)) and r[3] == pytest.approx(np.log(0.5))


def test_estimate_lambda_no_intercept_exact_and_noisy():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 30, 5000)
    e = H.estimate_lambda(2e-5 * x, x)
    assert e["lam"] == pytest.approx(2e-5, rel=1e-12) and e["n"] == 5000
    # eq. (11) has no intercept: a pure drift does not enter the slope formula sum(xr)/sum(x^2)
    r = 3e-5 * x + 1e-4 * rng.standard_normal(5000)
    e = H.estimate_lambda(r, x)
    assert e["lam"] == pytest.approx(float((x * r).sum() / (x * x).sum()))
    assert abs(e["lam"] - 3e-5) < 4 * e["se"] and e["t"] > 50
    assert np.isnan(H.estimate_lambda(np.array([1.0, 2.0]), np.array([0.0, 0.0]))["lam"])   # no variation in x


def test_usable_bars_filters():
    r = np.array([np.nan, 0.01, 0.9, -0.01, 0.0])
    Q = np.array([5.0, 5.0, 5.0, 0.0, 5.0])
    np.testing.assert_array_equal(H.usable_bars(r, Q, 0.5), [False, True, False, False, True])


def test_month_bar_range():
    lo, hi = H.month_bar_range("2024-02")
    assert lo * BAR == pd.Timestamp("2024-02-01").value // 10**6
    assert (hi + 1) * BAR == pd.Timestamp("2024-03-01").value // 10**6
    assert hi - lo + 1 == 29 * 288


# ------------------------------------------------------------------ readers
def test_read_kline_zip_header_and_microseconds(tmp_path):
    _, bars = _simulate_trades(1e-5, n_bars=50, seed=2)
    ms = H.read_kline_zip(_zip(tmp_path / "a.zip", "a.csv", _klines_csv(bars)))
    us = H.read_kline_zip(_zip(tmp_path / "b.zip", "b.csv", _klines_csv(bars, us=True, header=True)))
    pd.testing.assert_frame_equal(ms, us)
    assert (ms["bar"].to_numpy() == bars["bar"].to_numpy()).all()
    np.testing.assert_allclose(ms["B"], bars["B"])


def test_read_aggtrades_signs_and_bars(tmp_path):
    # two bars; isBuyerMaker=True means the taker SOLD -> negative sign
    t0 = 1_717_200_000_000
    rows = [(1, 10.0, 1.0, 1, 1, t0 + 1, False, True),     # buy  $10
            (2, 10.0, 0.4, 2, 2, t0 + 2, True, True),      # sell $4
            (3, 11.0, 1.0, 3, 3, t0 + BAR + 5, True, True)]  # sell $11 in the next bar
    tr = pd.DataFrame(rows, columns=["id", "p", "q", "f", "l", "ts", "bm", "best"])
    b = H.read_aggtrades_zip(_zip(tmp_path / "t.zip", "t.csv", _trades_csv(tr)))
    assert list(b["bar"]) == [t0 // BAR, t0 // BAR + 1]
    np.testing.assert_allclose(b["Q"], [14.0, 11.0])
    np.testing.assert_allclose(b["B"], [10.0, 0.0])
    np.testing.assert_allclose(b["svol_exact"], [np.sqrt(10) - 2.0, -np.sqrt(11)])
    np.testing.assert_allclose(b["close"], [10.0, 11.0])


def test_read_aggtrades_chunking_header_and_us_invariant(tmp_path):
    tr, _ = _simulate_trades(1e-5, n_bars=200, seed=3)
    full = H.read_aggtrades_zip(_zip(tmp_path / "a.zip", "a.csv", _trades_csv(tr)))
    tr_us = tr.assign(ts=tr["ts"] * 1000)
    chunked = H.read_aggtrades_zip(_zip(tmp_path / "b.zip", "b.csv", _trades_csv(tr_us, header=True)), chunksize=7)
    pd.testing.assert_frame_equal(full, chunked, check_dtype=False)


def test_complete_grid_fills_no_trade_bars():
    b = pd.DataFrame({"bar": [5, 8], "close": [1.0, 2.0], "Q": [3.0, 4.0], "B": [1.0, 1.0],
                      "svol_exact": [0.5, -0.5], "n_agg": [1, 1]})
    g = H.complete_grid(b, None, 9)
    assert list(g["bar"]) == [5, 6, 7, 8, 9]
    np.testing.assert_allclose(g["close"], [1, 1, 1, 2, 2])
    np.testing.assert_allclose(g["Q"], [3, 0, 0, 4, 0])


# ------------------------------------------------------------------ known answer: exact vs bar lambda
def test_exact_lambda_recovers_truth_and_kline_proxy_matches_aggtrades(tmp_path):
    lam = 2e-5
    trades, bars = _simulate_trades(lam, n_bars=4000, seed=4)
    month = pd.Timestamp(bars["bar"].iloc[0] * BAR, unit="ms").strftime("%Y-%m")
    lo, hi = H.month_bar_range(month)
    bars = bars[(bars["bar"] >= lo) & (bars["bar"] <= hi)]
    trades = trades[(trades["ts"] // BAR >= lo) & (trades["ts"] // BAR <= hi)]
    sym = tmp_path / "XUSDT"
    sym.mkdir()
    ap = _zip(sym / f"XUSDT-aggTrades-{month}.zip", "a.csv", _trades_csv(trades))
    kp = _zip(sym / f"XUSDT-5m-{month}.zip", "k.csv", _klines_csv(bars))
    res = H.aggtrades_month_lambda(str(ap), str(kp), 0.5)
    # exact trade-level lambda recovers the true slope
    assert res["agg_lam_exact"] == pytest.approx(lam, rel=0.03)
    # kline bars and aggTrades bars are the same data -> identical proxies
    assert res["kl_lam_sqrt_sides"] == pytest.approx(res["agg_lam_sqrt_sides"], rel=1e-9)
    assert res["integrity_med_absrel_Q"] == pytest.approx(0.0, abs=1e-9)
    assert res["integrity_share_close_equal"] == pytest.approx(1.0)
    # the worker on the kline file alone gives the same as the kline part of the validation worker
    kw = H.kline_month_lambda(str(kp), 0.5)
    assert kw["error"] == "" and kw["lam_sqrt_sides"] == pytest.approx(res["kl_lam_sqrt_sides"])
    assert kw["symbol"] == "XUSDT" and kw["measure_month"] == month


def test_bar_proxy_ranks_coins_like_true_lambda():
    """Across coins with different true lambda, the bar-level lambda preserves the ordering."""
    true = np.exp(np.linspace(np.log(5e-6), np.log(2e-4), 12))
    est = []
    for i, l in enumerate(true):
        _, b = _simulate_trades(l, n_bars=1500, rate=4.0, noise=1e-4, seed=10 + i)
        e = H.lambdas_from_bars(b[["bar", "close", "Q", "B"]], 0.5)
        est.append(e["lam_sqrt_sides"])
    rho = pd.Series(true).corr(pd.Series(est), method="spearman")
    assert rho > 0.95


def test_kline_worker_records_errors(tmp_path):
    d = tmp_path / "BADUSDT"
    d.mkdir()
    p = d / "BADUSDT-5m-2024-01.zip"
    p.write_bytes(b"not a zip")
    res = H.kline_month_lambda(str(p), 0.5)
    assert res["error"].startswith("BadZipFile")


# ------------------------------------------------------------------ panel-level tests
def _toy_panel(n_months=30, n=80, seed=5, priced="noncost"):
    """Panel with lnA = a*lnlam + noncost; returns load only on the named component."""
    rng = np.random.default_rng(seed)
    rows = []
    months = pd.period_range("2023-10", periods=n_months, freq="M")
    for m in months:
        lnlam = rng.normal(-10, 1, n)
        nonc = rng.normal(0, 1, n)
        lnA = 0.8 * lnlam + nonc
        lnA_C = lnA - rng.normal(0, 0.3, n)
        driver = nonc if priced == "noncost" else lnlam
        ret = 1.0 * driver + rng.normal(0, 1.0, n)
        for i in range(n):
            rows.append({"symbol": f"C{i:03d}USDT", "month": m, "ret": ret[i], "ret_adj": ret[i],
                         "lnA": lnA[i], "A": np.exp(lnA[i]), "lnA_C": lnA_C[i], "A_C": np.exp(lnA_C[i]),
                         "ResA": rng.normal(), "ResLnA": rng.normal(), "mom": rng.normal(0, .2), "ret_1": rng.normal(0, .1),
                         "_lam": np.exp(lnlam[i])})
    return pd.DataFrame(rows)


def _lam_table(panel, lag=2):
    L = panel[["symbol", "month", "_lam"]].copy()
    L["measure_month"] = (L["month"] - lag).astype(str)
    return pd.DataFrame({"symbol": L["symbol"], "measure_month": L["measure_month"],
                         "lam_sqrt_sides": L["_lam"], "lam_net_sqrt": L["_lam"] * 1.1, "n_bars": 5000})


def test_attach_lambda_lag_alignment_and_min_bars():
    panel = _toy_panel(n_months=4, n=20)
    lam = _lam_table(panel)
    lam.loc[0, "n_bars"] = 10                                   # one coin-month below min_bars
    p = H.attach_lambda(panel.drop(columns="_lam"), lam, H.SPEC, 2, "sqrt_sides", 1000, (0.0, 1.0))
    assert len(p) == len(panel) - 1
    m = p.merge(panel[["symbol", "month", "_lam"]], on=["symbol", "month"])
    np.testing.assert_allclose(m["lam"], m["_lam"])            # measure month t-2 -> panel month t
    np.testing.assert_allclose(p["lnlam"], np.log(p["lam"]))


def test_attach_lambda_negative_lambda_has_no_log():
    panel = _toy_panel(n_months=2, n=20)
    lam = _lam_table(panel)
    lam.loc[3, "lam_sqrt_sides"] = -1e-6
    p = H.attach_lambda(panel.drop(columns="_lam"), lam, H.SPEC, 2, "sqrt_sides", 1000, (0.0, 1.0))
    assert p["lnlam"].isna().sum() == 1


def test_cs_fit_resid_is_projection():
    df = _toy_panel(n_months=3, n=50).assign(lnlam=lambda d: np.log(d["_lam"]))
    fit, res = H.cs_fit_resid(df, "lnA", ["lnlam"], "month")
    np.testing.assert_allclose(fit + res, df["lnA"])
    for _, g in df.assign(res=res).groupby("month"):
        assert abs(np.corrcoef(g["res"], g["lnlam"])[0, 1]) < 1e-8
        assert abs(g["res"].mean()) < 1e-10


@pytest.mark.parametrize("priced", ["noncost", "cost"])
def test_table10_identifies_priced_component(priced):
    panel = _toy_panel(priced=priced)
    p = H.attach_lambda(panel.drop(columns="_lam"), _lam_table(panel), H.SPEC, 2, "sqrt_sides", 1000, (0.0, 1.0))
    out = H.table9_10(p, 6)
    t10 = out["T10"].set_index(["model", "var"])
    other = "cost" if priced == "noncost" else "noncost"
    assert t10.loc[("(3)", priced), "t"] > 5
    assert abs(t10.loc[("(3)", other), "t"]) < 3
    t9b = out["T9B"].set_index(["model", "var"])
    assert set(out["T9B"]["model"]) == {"(0) same sample", "(1)", "(2)"}
    if priced == "cost":                                       # cost part = function of ln lambda
        assert t9b.loc[("(1)", "lnlam"), "coef"] > 0.5
    t9a = out["T9A"]
    assert set(t9a["corr"]) == {"levels_pearson", "levels_spearman", "logs_pearson"}
    assert t9a.loc[(t9a["corr"] == "levels_pearson") & (t9a["var"] == "A"), "paper"].iloc[0] == 0.737


def test_table10_with_edge_adds_variant_and_floor_rule():
    panel = _toy_panel(n_months=12, n=60)
    rng = np.random.default_rng(0)
    edge = pd.DataFrame({"symbol": panel["symbol"], "measure_month": (panel["month"] - 2).astype(str),
                         "edge": np.exp(rng.normal(-6, 1, len(panel)))})
    edge.loc[:9, "edge"] = -0.001                              # non-positive estimates -> floored
    pe, share = H.attach_edge(panel.drop(columns="_lam"), edge, 2, (0.0, 1.0))
    assert share == pytest.approx(10 / len(panel))
    m0 = (panel["month"] == panel.loc[0, "month"]).to_numpy()
    pos_min = edge["edge"][m0 & (edge["edge"] > 0).to_numpy()].min()
    assert pe.loc[0, "EDGE"] == pytest.approx(pos_min)
    assert pe["lnedge"].notna().all()
    p = H.attach_lambda(pe, _lam_table(panel), H.SPEC, 2, "sqrt_sides", 1000, (0.0, 1.0))
    out = H.table9_10(p, 6)
    assert set(out["T10"]["cost_benchmarks"]) == {"lambda", "lambda+EDGE"}


def test_load_edge(tmp_path):
    assert H.load_edge(tmp_path / "none.parquet")[0] is None
    pth = tmp_path / "e.parquet"
    pd.DataFrame({"symbol": ["A", "B", "C"], "month": ["2024-01", "2024-01", "2024-02"],
                  "s2": [1e-4, -4e-6, 4e-4], "edge": [0.01, -0.002, 0.02]}).to_parquet(pth)
    e, status = H.load_edge(pth)
    assert len(e) == 3 and "'edge'" in status
    assert list(e["measure_month"]) == ["2024-01", "2024-01", "2024-02"]


def test_select_validation_cap_and_strata():
    panel = _toy_panel(n_months=1, n=50)
    panel["month"] = pd.Period("2024-08", "M")                 # = validation month 2024-06 + lag 2
    spec = H.SPEC | {"validation_months": ["2024-06", "2025-06"], "validation_per_quintile": 3, "aggtrades_max_mb": 300}
    sizes = {}
    for i, s in enumerate(panel["symbol"]):
        for m in spec["validation_months"]:
            sizes[H.agg_key(s, m)] = (400 if i % 4 == 0 else 10) * 2**20
    del sizes[H.agg_key("C001USDT", "2025-06")]
    sel = H.select_validation(panel, spec, 2, 7, sizes)
    chosen = sel[sel["status"] == "selected"]
    assert chosen.groupby("quintile_A_C").size().tolist() == [3] * 5
    assert not chosen["symbol"].isin([f"C{i:03d}USDT" for i in range(0, 50, 4)] + ["C001USDT"]).any()
    assert set(sel["status"]) <= {"selected", "skipped_over_cap", "skipped_missing_file"}
    again = H.select_validation(panel, spec, 2, 7, sizes)
    pd.testing.assert_frame_equal(sel, again)                  # seeded, deterministic


def test_needed_coin_months_window():
    panel = _toy_panel(n_months=6, n=3)
    spec = H.SPEC | {"measure_start": "2023-09", "measure_end": "2023-10"}
    need = H.needed_coin_months(panel, spec, 2)                # panel months 2023-11, 2023-12
    assert sorted(need["measure_month"].unique()) == ["2023-09", "2023-10"]
    assert len(need) == 6


def test_aggtrades_listing_is_cached(tmp_path, monkeypatch):
    calls = []

    def fake_list(prefix):
        calls.append(prefix)
        sym = prefix.split("/")[4]
        return {H.agg_key(sym, "2024-06"): 5, H.agg_key(sym, "2025-06"): 7, H.agg_key(sym, "2023-01"): 9}

    monkeypatch.setattr(H, "RAW_AGG", tmp_path)
    monkeypatch.setattr(H, "list_sizes", fake_list)
    a = H.aggtrades_listing(["AUSDT", "BUSDT"], ["2024-06", "2025-06"], 2)
    assert len(a) == 4 and len(calls) == 2 and H.agg_key("AUSDT", "2023-01") not in a
    b = H.aggtrades_listing(["AUSDT", "BUSDT"], ["2024-06", "2025-06"], 2)
    assert a == b and len(calls) == 2                           # second call served from listing.json
    H.aggtrades_listing(["AUSDT", "CUSDT"], ["2024-06"], 2)
    assert len(calls) == 3                                      # only the new symbol is listed


def test_relabelled_asset_ids_use_the_binance_symbol_files():
    assert H.raw_symbol("LUNAUSDT#2") == "LUNAUSDT" and H.raw_symbol("BTCUSDT") == "BTCUSDT"
    assert H.kline_key("CVCUSDT#2", "2024-01") == "data/spot/monthly/klines/CVCUSDT/5m/CVCUSDT-5m-2024-01.zip"
    assert H.local_path(H.agg_key("FTTUSDT#3", "2024-06")).parent.name == "FTTUSDT"


def test_diagnostics_variants_and_exit_rows():
    panel = _toy_panel(n_months=20, n=60, priced="cost")
    panel["ivol"] = np.random.default_rng(1).uniform(0.01, 0.1, len(panel))
    panel["lnVOL"] = np.random.default_rng(2).normal(15, 2, len(panel))
    last_m = panel["month"].max()
    exit_month = panel["month"].min() + 9
    panel = panel[~((panel["symbol"] == "C000USDT") & (panel["month"] > exit_month))]   # C000 leaves the panel
    base = panel.drop(columns="_lam")
    p = H.attach_lambda(base, _lam_table(panel), H.SPEC, 2, "sqrt_sides", 1000, (0.0, 1.0))
    tab, slopes = H.diagnostics(p, base, 6)
    v = list(tab["variant"].unique())
    assert v[0] == "main (same as T9B)" and len(v) == 8
    t = tab.set_index(["variant", "model", "var"])
    main_n = t.loc[("main (same as T9B)", "(1)", "lnlam"), "n_months"]
    halves = [x for x in v if "half" in x]
    assert sum(t.loc[(h, "(1)", "lnlam"), "n_months"] for h in halves) == main_n
    ex = [x for x in v if x.startswith("ex exit")][0]
    assert (tab.loc[tab["variant"] == ex, "n_rows_dropped_vs_main"] == 1).all()   # only C000 at its exit month
    assert last_m != exit_month
    assert ("+ ivol control", "(1)", "ivol") in t.index
    assert ("+ lnVOL control (size proxy; verify stage)", "(2)", "lnVOL") in t.index
    assert (tab.loc[tab["var"] == "lnlam", "coef"] > 0.5).all()                   # returns load on ln lambda
    r = tab[tab["var"] == "lnlam"]
    np.testing.assert_allclose(r["coef_per_cs_sd"], r["coef"] * r["cs_sd_lnlam"])
    assert set(slopes["model"]) == {"(0)", "(1)"} and slopes.groupby("model").size().eq(main_n).all()
