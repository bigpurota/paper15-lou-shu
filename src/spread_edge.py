"""Upgrade 3 (edge_spread): EDGE bid-ask spread benchmark and economic significance.

EDGE is the efficient OHLC spread estimator of Ardia, Guidotti & Kroencke (2024, JFE 161,
103916), implemented here from the paper's moment conditions; the official `bidask` package is
used ONLY as a reference test (tests/test_spread_edge.py and a run-time check on real coin-months).

What the script does (all outputs in results/upgrades/edge_spread/):
  1. OHLC per coin-day re-parsed from data/raw/klines/<SYM>/*.zip  -> data/cache/ohlc_daily.parquet
     (rebuilt only when a zip is newer than the cache), aligned to the audited daily panel of the
     base run (same coin-days; prices rescaled by close_daily / close_raw where the audit rescaled).
  2. EDGE per coin-month from daily OHLC of that month only (pairs of consecutive calendar days,
     both inside the month), >= `universe.min_valid_days` pairs   -> data/cache/edge_monthly.parquet
  3. Table 9 Panel C analogue: mean cross-sectional corr(EDGE, A / A_C / ResA); FM (signals at t-2,
     CMKT-adjusted return, controls mom, ret_1) on ln EDGE and on ln EDGE + ln A_C.
  4. Table 10 analogue: ln A = a + b ln EDGE + e within each month; FM on cost (fit) / non-cost (e).
  5. Premium vs costs (task 13.1): quintile long-short turnover x half-spread (EDGE/2) per side,
     net-of-cost H-L return with Newey-West t.

Variants (all reported, none selected on results): EDGE window m1 (one month, main) or m12 (signed
root of the mean monthly s2 over 12 months ending at t-2, >= 6 months); half-spread for costs
"pooled" (main: sqrt of the mean s2 of the leg's coins, since one coin-month of daily bars is mostly
noise) or per-coin "floor" / "abs" (abs = upper bound).

Non-positive EDGE (the estimator returns a signed root; s^2 < 0 happens when the true spread is small
relative to noise). Rule (set in the code; nothing shows that it predates the results):
  main  "floor": within each month's panel cross-section, non-positive estimates are set to the
        smallest positive estimate of that month (they are read as "spread too small to measure"),
        then winsorized 1/99% per month as in the paper, then logged;
  robustness "drop": non-positive estimates are dropped;
  robustness "abs" : |signed root| (the default output of bidask.edge, sign=False).
Costs use the same rule (floored half-spread), with "abs" as the conservative variant.

Usage:  python -I src/spread_edge.py config.yaml [--rebuild-ohlc] [--out results/upgrades/edge_spread]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import hashlib
import importlib.metadata as im
import json
import subprocess
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml

from fama_macbeth import fama_macbeth, nw_mean
from measures import cs_residual, mean_cs_corr, winsorize_cs
from sorts import alpha, assign_quantiles
import data_audit
import run as base

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "klines"
CACHE = ROOT / "data" / "cache"
OHLC_PATH = CACHE / "ohlc_daily.parquet"
EDGE_PATH = CACHE / "edge_monthly.parquet"
OUT = ROOT / "results" / "upgrades" / "edge_spread"
CODE_FILES = ["spread_edge.py", "fama_macbeth.py", "measures.py", "sorts.py", "data_binance.py", "data_audit.py"]
RULES = ["floor", "drop", "abs"]
COST_RULES = ["pooled", "floor", "abs"]
WINDOWS = {"m1": 1, "m12": 12}
FEE_PER_SIDE = 0.001

PAPER = {
    "source_t9a": "Table 9 Panel A (p. 53): corr(ES, .)",
    "corr_ES_A": 0.731, "corr_ES_A_C": 0.705, "corr_ES_ResA": 0.215,
    "source_t9c": "Table 9 Panel C (p. 53)",
    "fm_lnES_alone": {"coef": -0.248, "t": -1.94},
    "fm_lnAC_with_benchmarks": {"coef": 0.210, "t": 3.71},
    "source_t10": "Table 10 left panel (p. 54); cost = fit of lnA on ln lambda, QS, ES, RS",
    "t10_cost_alone": {"coef": -0.047, "t": -1.11},
    "t10_noncost_alone": {"coef": 0.186, "t": 3.04},
    "t10_both_cost": {"coef": 0.015, "t": 0.35},
    "t10_both_noncost": {"coef": 0.191, "t": 3.10},
}


def edge_from_ohlc(o, h, l, c) -> dict:
    """EDGE (Ardia, Guidotti & Kroencke 2024) from one calendar-complete sequence of OHLC prices.

    Missing days are NaN rows (the caller reindexes to the calendar), so a pair (t-1, t) with a
    missing day drops out of every moment that needs that day; each moment is averaged over the
    pairs where it is defined (the missing-data convention of the authors' reference code).
    Returns s2 (squared-spread estimate, may be < 0), the signed root `edge` = sign(s2) sqrt|s2|
    (0.01 = 1%), and n_pairs = number of pairs with all eight prices present.
    """
    def _log(x):
        x = np.asarray(x, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(x > 0, np.log(x), np.nan)

    lo_, lh, ll, lc = (_log(x) for x in (o, h, l, c))
    nan = {"s2": np.nan, "edge": np.nan, "n_pairs": 0}
    if len(lo_) < 3:
        return nan
    mid = (lh + ll) / 2
    hp, lp, cp, mp = lh[:-1], ll[:-1], lc[:-1], mid[:-1]
    oc, hc, lcur, mc = lo_[1:], lh[1:], ll[1:], mid[1:]
    n_pairs = int((np.isfinite(hp) & np.isfinite(lp) & np.isfinite(cp) & np.isfinite(oc)
                   & np.isfinite(hc) & np.isfinite(lcur)).sum())
    a1 = mc - oc
    a2 = oc - mp
    a3 = mc - cp
    a4 = cp - mp
    a5 = oc - cp

    def _ind(cond, *need):
        """Indicator that is NaN where any input it depends on is missing."""
        miss = np.zeros(len(cond), bool)
        for x in need:
            miss |= np.isnan(x)
        return np.where(miss, np.nan, cond.astype(float))

    tau = _ind((hc != lcur) | (lcur != cp), hc, lcur, cp)
    with np.errstate(invalid="ignore"), _quiet():
        q_open = np.nanmean(tau * _ind(oc != hc, oc, hc)) + np.nanmean(tau * _ind(oc != lcur, oc, lcur))
        q_close = np.nanmean(tau * _ind(cp != hp, cp, hp)) + np.nanmean(tau * _ind(cp != lp, cp, lp))
        p_tau = np.nanmean(tau)
        if np.nansum(tau) < 2 or q_open == 0 or q_close == 0:
            return {**nan, "n_pairs": n_pairs}
        b1 = a1 - np.nanmean(a1) / p_tau * tau
        b3 = a3 - np.nanmean(a3) / p_tau * tau
        b5 = a5 - np.nanmean(a5) / p_tau * tau
        m1 = -4.0 / q_open * b1 * a2 - 4.0 / q_close * b3 * a4
        m2 = -4.0 / q_open * b1 * a5 - 4.0 / q_close * b5 * a4
        e1, e2 = np.nanmean(m1), np.nanmean(m2)
        v1, v2 = np.nanmean(m1 ** 2) - e1 ** 2, np.nanmean(m2 ** 2) - e2 ** 2
    s2 = (v2 * e1 + v1 * e2) / (v1 + v2) if v1 + v2 > 0 else (e1 + e2) / 2.0
    if not np.isfinite(s2):
        return {**nan, "n_pairs": n_pairs}
    w2 = float(v1 / (v1 + v2)) if v1 + v2 > 0 else 0.5
    return {"s2": float(s2), "edge": float(np.sign(s2) * np.sqrt(abs(s2))), "n_pairs": n_pairs, "w2": w2}


class _quiet:
    """Silence 'Mean of empty slice' warnings from nanmean on all-NaN inputs."""
    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)
    def __exit__(self, *exc):
        self._w.__exit__(*exc)


def calendar_month(g: pd.DataFrame) -> pd.DataFrame:
    """One coin-month of OHLC reindexed to every calendar day between its first and last row."""
    g = g.set_index("date").sort_index()
    return g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="D"))


def edge_monthly(ohlc: pd.DataFrame, min_pairs: int = 10) -> pd.DataFrame:
    """One row per (symbol, month): edge (signed), edge_abs, s2, n_days, n_pairs.

    Only days inside the month are used (the pair last-day-of-previous-month -> first day is not);
    missing days are NaN rows. Coin-months with fewer than `min_pairs` complete pairs get NaN.
    """
    d = ohlc.assign(month=ohlc["date"].dt.to_period("M"))
    rows = []
    for (sym, mon), g in d.groupby(["symbol", "month"], sort=True):
        g = calendar_month(g[["date", "open", "high", "low", "close"]])
        r = edge_from_ohlc(g["open"].to_numpy(), g["high"].to_numpy(), g["low"].to_numpy(), g["close"].to_numpy())
        if r["n_pairs"] < min_pairs:
            r["s2"] = r["edge"] = r["w2"] = np.nan
        rows.append((sym, str(mon), r["edge"], r["s2"], int(g["open"].notna().sum()), r["n_pairs"], r.get("w2", np.nan)))
    out = pd.DataFrame(rows, columns=["symbol", "month", "edge", "s2", "n_days", "n_pairs", "w2"])
    out["edge_abs"] = out["edge"].abs()
    return out


def edge_window(em: pd.DataFrame, months: int, min_months: int | None = None) -> pd.DataFrame:
    """EDGE over a trailing window: signed root of the mean monthly s2 over `months` calendar months
    ending in the estimation month (at least `min_months` = months // 2 monthly estimates)."""
    if months == 1:
        return em
    min_months = min_months or months // 2
    w = em.pivot(index="month", columns="symbol", values="s2")
    w.index = pd.PeriodIndex(w.index, freq="M")
    w = w.reindex(pd.period_range(w.index.min(), w.index.max(), freq="M"))
    m = w.rolling(months, min_periods=min_months).mean()
    out = m.stack().rename("s2").reset_index()
    out.columns = ["month", "symbol", "s2"]
    out["month"] = out["month"].astype(str)
    out["edge"] = np.sign(out["s2"]) * np.sqrt(out["s2"].abs())
    out["edge_abs"] = out["edge"].abs()
    return out[["symbol", "month", "edge", "s2", "edge_abs"]]


def _zip_fingerprint() -> float:
    return max((p.stat().st_mtime for p in RAW.glob("*/*.zip")), default=0.0)


def build_ohlc(symbols: list[str] | None = None, path: Path = OHLC_PATH, rebuild: bool = False) -> pd.DataFrame:
    """Re-parse daily klines zips into (symbol, date, open, high, low, close, quote_volume)."""
    if path.exists() and not rebuild and path.stat().st_mtime >= _zip_fingerprint():
        return pd.read_parquet(path)
    from data_binance import _read_zip
    symbols = symbols or sorted(p.name for p in RAW.iterdir() if p.is_dir())
    frames = []
    for sym in symbols:
        files = sorted((RAW / sym).glob("*.zip"))
        if not files:
            continue
        df = pd.concat([_read_zip(f) for f in files], ignore_index=True)
        t = pd.to_numeric(df["open_time"])
        t = t.where(t < 1e14, t // 1000)
        frames.append(pd.DataFrame({
            "symbol": sym, "date": pd.to_datetime(t, unit="ms").dt.normalize(),
            **{c: pd.to_numeric(df[c], errors="coerce") for c in ["open", "high", "low", "close", "quote_volume"]}}))
    ohlc = pd.concat(frames, ignore_index=True).drop_duplicates(["symbol", "date"])
    ohlc = ohlc.sort_values(["symbol", "date"]).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    ohlc.to_parquet(path, index=False)
    return ohlc


def align_to_daily(ohlc: pd.DataFrame, daily: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Keep the coin-days of the (audited) daily panel and rescale OHLC by close_daily / close_raw.

    If the audit removed days or rescaled prices (redenominations), EDGE sees the same data as the
    Amihud measures. Returns the aligned frame and counts for the manifest.
    """
    dd = daily[["symbol", "date", "close"]].rename(columns={"close": "close_daily", "symbol": "asset"}).copy()
    dd["asset"] = dd["asset"].astype(str)
    dd["symbol"] = dd["asset"].str.split("#").str[0]
    dd["date"] = pd.to_datetime(dd["date"]).dt.tz_localize(None).dt.normalize()
    oo = ohlc.copy()
    oo["date"] = pd.to_datetime(oo["date"]).dt.tz_localize(None).dt.normalize()
    m = oo.merge(dd, on=["symbol", "date"], how="inner")
    n_new_ids = int((m["asset"] != m["symbol"]).sum())
    m["symbol"] = m["asset"]
    m = m.drop(columns=["asset"])
    k = m["close_daily"] / m["close"]
    rescale = (k - 1).abs() > 1e-9
    for col in ["open", "high", "low", "close"]:
        m[col] = np.where(rescale, m[col] * k, m[col])
    stats = {"ohlc_rows_raw": int(len(ohlc)), "ohlc_rows_aligned": int(len(m)),
             "rows_dropped_not_in_daily": int(len(ohlc) - len(m)), "rows_rescaled": int(rescale.sum()),
             "rows_new_asset_id": n_new_ids,
             "rows_bad_ohlc": int(((m["high"] < m["low"]) | (m[["open", "high", "low", "close"]] <= 0).any(axis=1)).sum())}
    return m.drop(columns=["close_daily"]), stats


def edge_signal(edge: pd.Series, month: pd.Series, rule: str) -> pd.Series:
    """Level of EDGE used in logs and costs under a non-positive-estimate rule (see module doc)."""
    e = edge.astype(float)
    if rule == "abs":
        return e.abs().where(e.abs() > 0)
    if rule == "drop":
        return e.where(e > 0)
    if rule == "floor":
        pos_min = e.where(e > 0).groupby(month).transform("min")
        return e.where((e > 0) | e.isna(), pos_min)
    raise ValueError(rule)


def attach_edge(panel: pd.DataFrame, em: pd.DataFrame, lag: int, rule: str, winsor) -> pd.DataFrame:
    """Merge EDGE of month t-lag into the return-month panel; add EDGE (winsorized) and lnEDGE."""
    p = panel.copy()
    p["month"] = p["month"].astype(str)
    est = (pd.PeriodIndex(p["month"], freq="M") - lag).astype(str)
    e = em[["symbol", "month", "edge"]].rename(columns={"month": "est_month", "edge": "edge_raw"})
    p = p.assign(est_month=est).merge(e, on=["symbol", "est_month"], how="left")
    p["EDGE"] = edge_signal(p["edge_raw"], p["month"], rule)
    p = winsorize_cs(p, ["EDGE"], "month", tuple(winsor))
    p["lnEDGE"] = np.log(p["EDGE"])
    return p


def _fm(panel, models, lags, controls=("mom", "ret_1")) -> pd.DataFrame:
    rows = []
    for name, xs in models.items():
        res = fama_macbeth(panel, "ret_adj", xs + list(controls), lags=lags)
        for _, r in res.iterrows():
            rows.append({"model": name, "var": r["var"], "coef": r["coef"], "t": r["t"],
                         "n_months": res.attrs["n_months"], "avg_obs": res.attrs["avg_obs"],
                         "adj_r2": res.attrs["avg_adj_r2"]})
    return pd.DataFrame(rows)


def table9c(p: pd.DataFrame, lags: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Correlations of EDGE with A, A_C, ResA and FM on ln EDGE (+ ln A_C), on one common sample."""
    corr = []
    s = p.dropna(subset=["EDGE", "A", "A_C", "ResA"])
    for method in ["pearson", "spearman"]:
        c = mean_cs_corr(s, ["EDGE", "A", "A_C", "ResA"], "month", method)
        corr.append({"method": method, **{f"corr_EDGE_{k}": c.loc["EDGE", k] for k in ["A", "A_C", "ResA"]},
                     "n_months": s["month"].nunique(), "avg_obs": s.groupby("month").size().mean()})
    c = mean_cs_corr(s.dropna(subset=["lnEDGE", "lnA", "lnA_C"]), ["lnEDGE", "lnA", "lnA_C"], "month")
    corr.append({"method": "pearson_logs", "corr_EDGE_A": c.loc["lnEDGE", "lnA"],
                 "corr_EDGE_A_C": c.loc["lnEDGE", "lnA_C"], "corr_EDGE_ResA": np.nan,
                 "n_months": s["month"].nunique(), "avg_obs": s.groupby("month").size().mean()})
    common = p.dropna(subset=["lnEDGE", "lnA_C", "lnA", "ret_adj", "mom", "ret_1"])
    fm = _fm(common, {"(1) lnEDGE": ["lnEDGE"], "(2) lnEDGE+lnA_C": ["lnEDGE", "lnA_C"],
                      "(ref) lnA_C": ["lnA_C"], "(ref) lnA": ["lnA"]}, lags)
    return pd.DataFrame(corr), fm


def table10(p: pd.DataFrame, lags: int) -> pd.DataFrame:
    """ln A = a + b ln EDGE + e per month: cost = fit, non-cost = e; FM on cost / non-cost / both."""
    s = p.dropna(subset=["lnA", "lnEDGE"]).copy()
    s["NonCost"] = cs_residual(s, "lnA", "lnEDGE", "month")
    s["Cost"] = s["lnA"] - s["NonCost"]
    b = s.groupby("month").apply(lambda g: np.polyfit(g["lnEDGE"], g["lnA"], 1)[0] if len(g) > 2 else np.nan)
    s = s.dropna(subset=["Cost", "NonCost", "ret_adj", "mom", "ret_1"])
    fm = _fm(s, {"(1) cost": ["Cost"], "(2) non-cost": ["NonCost"], "(3) both": ["Cost", "NonCost"]}, lags)
    r2 = s.groupby("month").apply(lambda g: np.corrcoef(g["lnA"], g["lnEDGE"])[0, 1] ** 2 if len(g) > 2 else np.nan)
    fm["first_stage_r2_mean"] = float(r2.mean())
    cser = fama_macbeth(s, "ret_adj", ["Cost", "mom", "ret_1"], lags=lags).attrs["coef_series"]["Cost"]
    fm["first_stage_slope_share_neg"] = float((b < 0).mean())
    fm["first_stage_slope_min_abs"] = float(b.abs().min())
    fm["cost_coef_monthly_median"] = float(cser.median())
    fm["cost_coef_monthly_max_abs"] = float(cser.abs().max())
    return fm


def _half_spread(p: pd.DataFrame, em: pd.DataFrame, rule: str) -> pd.Series:
    """Half-spread (fraction) charged when the month-t portfolio is formed at the end of t-1.

    EDGE of month t-1 (the trading month), else t-2; the non-positive rule applied within the
    month's panel cross-section; still missing -> median of the coin's quintile in that month.
    """
    col = "s2" if rule == "pooled" else "edge"
    e = em.set_index(["symbol", "month"])[col]
    per = pd.PeriodIndex(p["month"], freq="M")
    out = pd.Series(np.nan, index=p.index)
    for lag in [1, 2]:
        key = pd.MultiIndex.from_arrays([p["symbol"], (per - lag).astype(str)])
        v = pd.Series(e.reindex(key).to_numpy(), index=p.index)
        out = out.fillna(v)
    if rule == "pooled":
        imputed = out.isna()
        s2 = out.groupby([p["month"], p["q"]]).transform("mean").clip(lower=0)
        return (np.sqrt(s2) / 2.0).rename("hs"), imputed
    lvl = edge_signal(out, p["month"], rule)
    imputed = lvl.isna()
    lvl = lvl.fillna(lvl.groupby([p["month"], p["q"]]).transform("median"))
    return (lvl / 2.0).rename("hs"), imputed


def costs(panel: pd.DataFrame, em: pd.DataFrame, signal: str, q: int, rule: str, lags: int,
          factors: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Monthly turnover, cost and net H-L for the EW top-minus-bottom quintile portfolio on `signal`.

    Half-spread rules: "pooled" (main) = sqrt(mean s2 over the leg's coins)/2, the same for every coin
    of the leg (per-coin EDGE from 30 daily bars is mostly noise; the mean of s2 is unbiased);
    "floor"/"abs" = per-coin EDGE/2 under the non-positive rules ("abs" is an upper bound).
    Weights: 1/N in each leg at formation. Turnover of a leg at t = sum_i |w_it - w~_i,t-1|, where
    w~ are last month's weights after drift with realised returns (exits sell to 0, entries buy).
    Cost of a leg = sum_i |dw_i| * half-spread_i (paid once per side, i.e. on each trade);
    long-short cost = cost_long + cost_short (per $1 long and $1 short). The first month (full build)
    is excluded from averages. Returns (monthly frame, summary row frame).
    """
    p = panel[["month", "symbol", signal, "ret"]].dropna().copy()
    p["month"] = p["month"].astype(str)
    p["q"] = assign_quantiles(p, signal, q, "month")
    p = p.dropna(subset=["q"])
    p["hs"], imputed = _half_spread(p, em, rule)
    p["r"] = p["ret"] / 100.0
    months = sorted(p["month"].unique())
    rows = []
    prev = {}
    for m in months:
        g = p[p["month"] == m]
        rec = {"month": m}
        for leg, qq in [("long", q), ("short", 1)]:
            x = g[g["q"] == qq]
            w = pd.Series(1.0 / len(x), index=x["symbol"].to_numpy())
            hs = pd.Series(x["hs"].to_numpy(), index=w.index)
            old = prev.get(leg, pd.Series(dtype=float))
            u = w.index.union(old.index)
            dw = (w.reindex(u, fill_value=0.0) - old.reindex(u, fill_value=0.0)).abs()
            hs_u = hs.reindex(u).fillna(prev.get(leg + "_hs", pd.Series(dtype=float)).reindex(u)).fillna(hs.median())
            rec[f"turnover_{leg}"] = float(dw.sum())
            rec[f"cost_{leg}"] = float((dw * hs_u).sum()) * 100.0
            rec[f"ret_{leg}"] = float(x["r"].mean()) * 100.0
            rec[f"hs_{leg}"] = float(x["hs"].mean()) * 100.0
            r = pd.Series(x["r"].to_numpy(), index=w.index)
            drift = w * (1 + r)
            prev[leg] = drift / drift.sum()
            prev[leg + "_hs"] = hs
        rows.append(rec)
    ts = pd.DataFrame(rows)
    ts["first_month"] = ts.index == 0
    ts["hl_gross"] = ts["ret_long"] - ts["ret_short"]
    ts["cost_ls"] = ts["cost_long"] + ts["cost_short"]
    ts["hl_net"] = ts["hl_gross"] - ts["cost_ls"]
    ts["fee_ls"] = (ts["turnover_long"] + ts["turnover_short"]) * FEE_PER_SIDE * 100.0
    ts["hl_net_fee"] = ts["hl_net"] - ts["fee_ls"]
    ts.insert(0, "signal", signal)
    ts.insert(1, "rule", rule)
    s = ts[~ts["first_month"]].set_index("month")
    row = {"signal": signal, "rule": rule, "n_months": len(s),
           "turnover_long": s["turnover_long"].mean(), "turnover_short": s["turnover_short"].mean(),
           "half_spread_long_pct": s["hs_long"].mean(), "half_spread_short_pct": s["hs_short"].mean(),
           "share_months_hs_zero_long": float((s["hs_long"] == 0).mean()),
           "share_months_hs_zero_short": float((s["hs_short"] == 0).mean()),
           "cost_ls_pct": s["cost_ls"].mean(), "share_hs_imputed": float(imputed.mean()),
           "fee_per_side_pct": FEE_PER_SIDE * 100.0, "fee_ls_pct": s["fee_ls"].mean()}
    for c in ["hl_gross", "hl_net", "hl_net_fee"]:
        m_, _, t_ = nw_mean(s[c], lags)
        row[f"{c}_pct"], row[f"{c}_t"] = m_, t_
    row["hl_net_pct_per_year"] = 12 * row["hl_net_pct"]
    row["breakeven_cost_multiple"] = row["hl_gross_pct"] / row["cost_ls_pct"] if row["cost_ls_pct"] > 0 else np.nan
    to = (s["turnover_long"] + s["turnover_short"]).mean()
    row["breakeven_cost_per_side_pct"] = row["hl_gross_pct"] / to if to > 0 else np.nan
    if factors is not None:
        f = factors.copy()
        f.index = f.index.astype(str)
        for c in ["hl_gross", "hl_net", "hl_net_fee"]:
            a, t_ = alpha(s[c], f, lags)
            row[f"{c}_alpha_pct"], row[f"{c}_alpha_t"] = a, t_
    return ts, pd.DataFrame([row])


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info() -> dict:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        st = subprocess.run(["git", "status", "--porcelain", "--", "src", "tests", "config.yaml", "Makefile", "mk", "requirements.txt"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
        return {"sha": sha, "dirty": bool(st.strip()), "dirty_scope": "code+config (src, tests, config.yaml, Makefile, mk, requirements.txt); generated outputs excluded"}
    except (OSError, subprocess.CalledProcessError):
        return {"sha": None, "dirty": None}


def bidask_check(ohlc: pd.DataFrame, em: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Compare our EDGE with bidask.edge(sign=True) on n random real coin-months (gaps as NaN rows)."""
    try:
        import bidask
    except ImportError:
        return pd.DataFrame([{"status": "bidask not installed"}])
    ver = im.version("bidask")
    ok = em.dropna(subset=["edge"])
    gap = ok[ok["n_pairs"] < ok["n_days"] - 1]
    g = gap.sample(n=min(n // 3, len(gap)), random_state=seed)
    rest = ok.drop(g.index)
    cand = pd.concat([g, rest.sample(n=min(n - len(g), len(rest)), random_state=seed)])
    d = ohlc.assign(month=ohlc["date"].dt.to_period("M").astype(str)).set_index(["symbol", "month"]).sort_index()
    rows = []
    for _, r in cand.iterrows():
        full = calendar_month(d.loc[(r["symbol"], r["month"])].reset_index(drop=True))
        ref = bidask.edge(full["open"].to_numpy(), full["high"].to_numpy(), full["low"].to_numpy(),
                          full["close"].to_numpy(), sign=True)
        rows.append({"symbol": r["symbol"], "month": r["month"], "edge_ours": r["edge"], "edge_bidask": ref,
                     "abs_diff": abs(r["edge"] - ref), "n_gaps": int(full["open"].isna().sum()),
                     "bidask_version": ver})
    return pd.DataFrame(rows)


def _f(x, nd=2):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def _sig(t, hlz):
    a = abs(t)
    return (f"significant and clears HLZ |t|>{hlz:g}" if a > hlz else
            (f"significant at 1.96 but does not clear HLZ |t|>{hlz:g}" if a > 1.96 else f"not significant (|t| < 1.96 < {hlz:g} HLZ)"))


def write_summary(out: Path, ctx: dict) -> None:
    hlz = ctx["hlz"]
    corr, fm9, t10, cs, chk, ed = ctx["corr"], ctx["fm9"], ctx["t10"], ctx["cost"], ctx["check"], ctx["edge_desc"]
    main9 = fm9[(fm9["rule"] == "floor") & (fm9["window"] == "m1")]
    w12 = fm9[(fm9["rule"] == "floor") & (fm9["window"] == "m12")]
    g = lambda df, m, v: df[(df["model"] == m) & (df["var"] == v)].iloc[0]
    c = corr[(corr["rule"] == "floor") & (corr["window"] == "m1") & (corr["method"] == "pearson")].iloc[0]
    c12 = corr[(corr["rule"] == "floor") & (corr["window"] == "m12") & (corr["method"] == "pearson")].iloc[0]
    e1 = g(main9, "(1) lnEDGE", "lnEDGE")
    e2 = g(main9, "(2) lnEDGE+lnA_C", "lnEDGE")
    a2 = g(main9, "(2) lnEDGE+lnA_C", "lnA_C")
    t = t10[(t10["rule"] == "floor") & (t10["window"] == "m1")]
    c1, n2 = g(t, "(1) cost", "Cost"), g(t, "(2) non-cost", "NonCost")
    c3, n3 = g(t, "(3) both", "Cost"), g(t, "(3) both", "NonCost")
    k = cs[(cs["signal"] == "A_C") & (cs["rule"] == "pooled")].iloc[0]
    ka = cs[(cs["signal"] == "A_C") & (cs["rule"] == "abs")].iloc[0]
    P = PAPER
    one = fm9[(fm9["model"] == "(1) lnEDGE") & (fm9["var"] == "lnEDGE")]
    cst = t10[(t10["model"] == "(1) cost") & (t10["var"] == "Cost")]
    nco = t10[(t10["model"] == "(2) non-cost") & (t10["var"] == "NonCost")]
    focal = pd.concat([fm9[fm9["var"].isin(["lnEDGE", "lnA_C"])], t10[t10["var"].isin(["Cost", "NonCost"])]])
    nv = len(one)
    sgn = cst.set_index(["window", "rule"])["coef"].le(0) & nco.set_index(["window", "rule"])["coef"].gt(0)
    pattern = int(sgn.sum())
    n_gap_chk = int((chk["n_gaps"] > 0).sum()) if len(chk) and "n_gaps" in chk else 0
    L = [f"# Upgrade 3: the EDGE spread and the economic significance of the premium",
         "",
         f"Base run: `{ctx['base_run_id']}`; upgrade run: `{ctx['run_id']}`. Every number below comes from the CSVs in this folder.",
         "",
         "## What we tested",
         "- EDGE (Ardia, Guidotti, Kroencke 2024): our own implementation on daily OHLC within the month (≥"
         f"{ctx['min_pairs']} pairs of adjacent days). Cross-check against the `bidask` package {ctx['bidask_version']}: "
         f"{len(chk)} coin-months (random draw stratified so that up to a third are months with missing days; "
         f"{n_gap_chk} of them have gaps; `edge_bidask_check.csv`), max |difference| = {chk['abs_diff'].max():.1e}."
         if len(chk) and "abs_diff" in chk else
         "- EDGE: the cross-check against `bidask` was not run (package not installed).",
         f"- Non-positive estimates: {ed['share_nonpos']:.1%} of coin-months with an estimate (median EDGE {ed['median_edge_pct']:.3f}%): "
         "on daily candles a coin's monthly spread estimate is almost pure noise. The main rule is floor "
         "(the smallest positive estimate of the month); robustness: drop, abs and a 12-month window (all variants in the CSVs).",
         "- Analogue of Table 9 Panel C (EDGE instead of ES), analogue of Table 10 (cost/non-cost parts of ln A), long–short costs of the A_C quintiles.",
         "",
         "## Key numbers (floor rule) vs the paper",
         "| Statistic | Paper (ES, NYSE/AMEX) | Us (EDGE, Binance) |",
         "|---|---|---|",
         f"| corr(spread, A) | {P['corr_ES_A']:.3f} | {c['corr_EDGE_A']:.3f} |",
         f"| corr(spread, A_C) | {P['corr_ES_A_C']:.3f} | {c['corr_EDGE_A_C']:.3f} |",
         f"| corr(spread, Res.A) | {P['corr_ES_ResA']:.3f} | {c['corr_EDGE_ResA']:.3f} |",
         f"| FM: ln spread alone | {P['fm_lnES_alone']['coef']:.3f} (t {P['fm_lnES_alone']['t']:.2f}) | {e1['coef']:.3f} (t {e1['t']:.2f}) |",
         f"| FM: ln A_C together with the spread (paper: model 4, + QS, ES, RS, λ) | {P['fm_lnAC_with_benchmarks']['coef']:.3f} (t {P['fm_lnAC_with_benchmarks']['t']:.2f}) | {a2['coef']:.3f} (t {a2['t']:.2f}); ln EDGE {e2['coef']:.3f} (t {e2['t']:.2f}) |",
         f"| T10: cost alone | {P['t10_cost_alone']['coef']:.3f} (t {P['t10_cost_alone']['t']:.2f}) | {c1['coef']:.3f} (t {c1['t']:.2f}) |",
         f"| T10: non-cost alone | {P['t10_noncost_alone']['coef']:.3f} (t {P['t10_noncost_alone']['t']:.2f}) | {n2['coef']:.3f} (t {n2['t']:.2f}) |",
         f"| T10: together, cost / non-cost | {P['t10_both_cost']['coef']:.3f} (t {P['t10_both_cost']['t']:.2f}) / {P['t10_both_noncost']['coef']:.3f} (t {P['t10_both_noncost']['t']:.2f}) | {c3['coef']:.3f} (t {c3['t']:.2f}) / {n3['coef']:.3f} (t {n3['t']:.2f}) |",
         f"| 12-month window: corr(EDGE, A_C); ln EDGE in FM | — | {c12['corr_EDGE_A_C']:.3f}; {g(w12, '(1) lnEDGE', 'lnEDGE')['coef']:.3f} (t {g(w12, '(1) lnEDGE', 'lnEDGE')['t']:.2f}) |",
         "",
         "## Premium vs costs (A_C quintiles, EW, %/month; leg half-spread = sqrt(mean s²)/2)",
         f"- Gross H–L: {_f(k['hl_gross_pct'])} (t {_f(k['hl_gross_t'])}); turnover long/short: {_f(k['turnover_long'])}/{_f(k['turnover_short'])} per month; "
         f"mean half-spread long/short: {_f(k['half_spread_long_pct'])}%/{_f(k['half_spread_short_pct'])}%.",
         f"- Long–short costs: {_f(k['cost_ls_pct'])}%/month; net H–L: {_f(k['hl_net_pct'])} (t {_f(k['hl_net_t'])}), "
         f"{_f(k['hl_net_pct_per_year'], 1)}%/year; cost cushion: ×{_f(k['breakeven_cost_multiple'])}.",
         f"- Conservative variant (|EDGE| per coin, abs rule; noise inflates it relative to EDGE, but it is not a guaranteed upper bound on the true spread): {_f(ka['cost_ls_pct'])}%/month, net H–L {_f(ka['hl_net_pct'])} (t {_f(ka['hl_net_t'])}).",
         f"- Reliability of the main estimate: in {k['share_months_hs_zero_long']:.0%} of months for the long leg and {k['share_months_hs_zero_short']:.0%} for the short leg "
         "the leg's mean s² is ≤ 0 and its half-spread is set to 0, so daily-candle EDGE cannot resolve the spread even at the portfolio level; the cost figure must not be read as a measurement.",
         f"- EDGE-free bound: the cost per side (half-spread + fee) at which the mean net H–L is 0: {_f(k['breakeven_cost_per_side_pct'])}% "
         f"(at a long+short turnover of {_f(k['turnover_long'] + k['turnover_short'])} per month).",
         f"- Extra variant (not pre-registered): + a fee of {_f(k['fee_per_side_pct'], 2)}% per side = {_f(k['fee_ls_pct'])}%/month; "
         f"net H–L after spread and fee {_f(k['hl_net_fee_pct'])} (t {_f(k['hl_net_fee_t'])}), alpha {_f(k['hl_net_fee_alpha_pct'])} (t {_f(k['hl_net_fee_alpha_t'])}).",
         "",
         "## Interpretation",
         f"- ln EDGE alone: {_sig(e1['t'], hlz)} (sign {'+' if e1['coef'] > 0 else '−'}); ln A_C controlling for EDGE: {_sig(a2['t'], hlz)}.",
         f"- Non-cost part of ln A: {_sig(n2['t'], hlz)}; cost part: {_sig(c1['t'], hlz)}."
         + (" As in the paper, the part of the Amihud measure unrelated to the spread is priced." if abs(n2['t']) > 1.96 and abs(c1['t']) <= 1.96 and n2['coef'] > 0 else
            " The paper's pattern (only the non-cost part is priced) is not statistically reproduced on our data."),
         f"- Across all {nv} variants (window × rule): ln EDGE alone > 0 in {int((one['coef'] > 0).sum())} of {nv} (the paper's sign is −); "
         f"cost part > 0 in {int((cst['coef'] > 0).sum())} of {nv}; non-cost > 0 in {int((nco['coef'] > 0).sum())} of {nv}. "
         f"Max |t| over EDGE / A_C / cost / non-cost across all variants = {focal['t'].abs().max():.2f}"
         + (" (< 1.96). " if focal['t'].abs().max() < 1.96 else " (≥ 1.96 in at least one variant). ")
         + f"The paper's sign pattern (cost ≤ 0, non-cost > 0) holds in {pattern} of {nv} variants.",
         "- The premium " + ("survives costs: the net H–L is significant." if k['hl_net_t'] > 1.96 else
                       ("is positive after costs but not statistically significant (nor is the gross one)." if k['hl_net_pct'] > 0 else
                        "does not survive costs: the net H–L is ≤ 0.")),
         "",
         "## Caveats",
         "- EDGE is an estimate from daily candles, not an observed spread; for liquid coins the true spread is small and the estimate is noisy (hence negative s²). "
         "The task (task.md, line 59) envisages hourly bars; this is a deviation.",
         f"- Crypto trades 24/7: the median |open − previous day's close| = {ed['median_abs_open_gap_bp']:.1f} bp (open = previous close on {ed['share_open_eq_prev_close']:.0%} of days) "
         f"against a median daily high−low range of {ed['median_hl_range_bp']:.0f} bp. The median weight of EDGE's second moment (on this gap) = {ed['median_weight_moment2']:.4f}: "
         "the estimate is driven almost entirely by the bounce between the last trade of one day and the first trade of the next.",
         f"- T10, cost part: Cost = a_t + b_t·ln EDGE, so the monthly coefficient = (coefficient on ln EDGE)/b_t. The slope b_t < 0 in {c1['first_stage_slope_share_neg']:.0%} of months, "
         f"min |b_t| = {c1['first_stage_slope_min_abs']:.4f}; monthly cost coefficients: median {c1['cost_coef_monthly_median']:.2f}, max |·| {c1['cost_coef_monthly_max_abs']:.0f}. "
         "The FM mean of the cost part is driven by a few months, so only t, not magnitude, can be compared with the paper.",
         "- Costs: the main estimate is the half-spread only (the fee is a separate line above); it excludes the price impact of large orders and the cost/availability of shorting (many illiquid coins cannot be borrowed).",
         "- The decomposition of ln A into cost/non-cost is estimated on the panel of coins with a return in t (as in FM), not on the full t−2 cross-section.",
         ]
    (out / "SUMMARY.md").write_text("\n".join(L) + "\n")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--rebuild-ohlc", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    ap.add_argument("--n-check", type=int, default=300)
    a = ap.parse_args(argv)
    timings, t0 = {}, time.perf_counter()

    def lap(name):
        nonlocal t0
        timings[name] = round(time.perf_counter() - t0, 3)
        print(f"[edge_spread] {name}: {timings[name]:.1f}s", flush=True)
        t0 = time.perf_counter()

    cfg_path = Path(a.config)
    cfg = yaml.safe_load(cfg_path.read_text())
    seed, lags, q = int(cfg["seed"]), int(cfg["tests"]["nw_lags"]), int(cfg["tests"]["quantiles"])
    lag, winsor, min_pairs = int(cfg["measures"]["lag_months"]), cfg["measures"]["winsor"], int(cfg["universe"]["min_valid_days"])
    res_dir = Path(a.results_dir)
    base_id = (res_dir / "LATEST").read_text().strip()
    base_dir = res_dir / base_id
    base_man = json.loads((base_dir / "run_manifest.json").read_text())
    panel_path = base.resolve_panel_cache(base_man, base_id)
    daily_path = base.resolve_repo_path(base_man["data_source"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    ohlc_raw = build_ohlc(rebuild=a.rebuild_ohlc)
    lap("ohlc")
    daily = pd.read_parquet(daily_path)
    audit_info = {"enabled": False}
    if data_audit.audit_config(cfg)["enabled"]:
        daily, ev, _ = data_audit.audit(daily, cfg)
        audit_info = {"enabled": True, "events_by_type": {k: int(v) for k, v in ev["type"].value_counts().items()},
                      "daily_rows_clean": int(len(daily)), "assets_clean": int(daily["symbol"].nunique())}
    ohlc, align = align_to_daily(ohlc_raw, daily)
    align["audit"] = audit_info
    del ohlc_raw, daily
    lap("align")
    em = edge_monthly(ohlc, min_pairs)
    em.to_parquet(EDGE_PATH, index=False)
    lap("edge")
    chk = bidask_check(ohlc, em, a.n_check, seed)
    chk.to_csv(out / "edge_bidask_check.csv", index=False)
    lap("bidask_check")

    panel = pd.read_parquet(panel_path)
    panel["month"] = panel["month"].astype(str)
    fac = pd.read_csv(base_dir / "factors.csv", dtype={"month": str}).set_index("month")[["CMKT", "CMOM"]]
    ok = em["edge"].notna()
    ed = {"n_coin_months": int(ok.sum()), "share_nonpos": float((em.loc[ok, "edge"] <= 0).mean()),
          "share_zero": float((em.loc[ok, "edge"] == 0).mean()), "median_edge_pct": float(em.loc[ok, "edge"].median() * 100),
          "n_missing_lt_min_pairs": int((~ok).sum()),
          "median_weight_moment2": float(em.loc[ok, "w2"].median()), "mean_weight_moment2": float(em.loc[ok, "w2"].mean())}
    oo = ohlc.sort_values(["symbol", "date"])
    consec = oo.groupby("symbol")["date"].diff() == pd.Timedelta(days=1)
    pc = oo.groupby("symbol")["close"].shift()
    ed["share_open_eq_prev_close"] = float((oo["open"] == pc)[consec].mean())
    ed["median_abs_open_gap_bp"] = float((np.log(oo["open"] / pc).abs()[consec]).median() * 1e4)
    ed["median_hl_range_bp"] = float(((oo["high"] - oo["low"]) / oo["close"]).median() * 1e4)
    del oo, consec, pc
    corr_l, fm_l, t10_l, cost_l, ts_l, cov_l = [], [], [], [], [], []
    for wname, wm in WINDOWS.items():
        emw = edge_window(em, wm)
        for rule in RULES:
            p = attach_edge(panel, emw, lag, rule, winsor)
            tag = {"window": wname, "rule": rule}
            cov_l.append({**tag, "panel_rows": len(p), "rows_with_edge_raw": int(p["edge_raw"].notna().sum()),
                          "rows_with_lnEDGE": int(p["lnEDGE"].notna().sum()),
                          "share_nonpos_in_panel": float((p["edge_raw"] <= 0).sum() / max(p["edge_raw"].notna().sum(), 1))})
            cr, fm = table9c(p, lags)
            corr_l.append(cr.assign(**tag))
            fm_l.append(fm.assign(**tag))
            t10_l.append(table10(p, lags).assign(**tag))
    for rule in COST_RULES:
        for sig in ["A_C", "A", "ResA"]:
            ts, row = costs(panel, em, sig, q, rule, lags, fac)
            ts_l.append(ts)
            cost_l.append(row)
    checks = {"bidask_max_abs_diff": float(chk["abs_diff"].max()) if "abs_diff" in chk else None,
              "bidask_n_coin_months": int(len(chk)), "bidask_n_with_gaps": int((chk.get("n_gaps", pd.Series(dtype=int)) > 0).sum())}
    t3p = base_dir / "T3_portfolios.csv"
    if t3p.exists():
        base_hl = pd.read_csv(t3p, dtype={"month": str})
        mine = pd.concat(ts_l, ignore_index=True)
        mine = mine[mine["rule"] == COST_RULES[0]][["signal", "month", "hl_gross"]]
        mm = mine.merge(base_hl[["signal", "month", "HL"]], on=["signal", "month"], how="outer")
        checks["gross_hl_vs_base_T3_max_abs_diff"] = float((mm["hl_gross"] - mm["HL"]).abs().max())
        checks["gross_hl_vs_base_T3_unmatched_months"] = int(mm[["hl_gross", "HL"]].isna().any(axis=1).sum())
    lap("tests")
    corr, fm9, t10, cost = (pd.concat(x, ignore_index=True) for x in (corr_l, fm_l, t10_l, cost_l))
    ed_desc = em.loc[ok, "edge"].describe(percentiles=[.01, .1, .25, .5, .75, .9, .99]).to_frame("edge").T
    ed_desc = ed_desc.assign(**ed)
    tables = {"T9C_corr": corr, "T9C_fm": fm9, "T10_fm": t10, "costs_summary": cost,
              "costs_monthly": pd.concat(ts_l, ignore_index=True), "edge_describe": ed_desc,
              "coverage": pd.DataFrame(cov_l),
              "paper_numbers": pd.json_normalize(PAPER, sep=".").T.reset_index().rename(columns={"index": "key", 0: "value"})}
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cfg_hash = sha256_file(cfg_path)
    run_id = f"{stamp}_{cfg_hash[:8]}_edge_spread"
    try:
        bver = im.version("bidask")
    except im.PackageNotFoundError:
        bver = None
    write_summary(out, {"hlz": float(cfg["tests"]["hlz_threshold"]), "corr": corr, "fm9": fm9, "t10": t10,
                        "cost": cost, "check": chk, "edge_desc": ed, "base_run_id": base_id, "run_id": run_id,
                        "min_pairs": min_pairs, "bidask_version": bver})
    lap("write")
    manifest = {
        "run_id": run_id, "upgrade": "edge_spread", "created_utc": stamp,
        "base_run_id": base_id, "base_config_sha256": base_man.get("config_sha256"),
        "git": git_info(), "config_path": base.rel_path(cfg_path), "config_sha256": cfg_hash,
        "code_sha256": {f: sha256_file(ROOT / "src" / f) for f in CODE_FILES},
        "data_sha256": {"panel": sha256_file(panel_path), "daily": sha256_file(daily_path),
                        "base_factors_csv": sha256_file(base_dir / "factors.csv"),
                        "ohlc_daily": sha256_file(OHLC_PATH), "edge_monthly": sha256_file(EDGE_PATH)},
        "data_paths": {"panel": base.rel_path(panel_path), "daily": base.rel_path(daily_path),
                       "ohlc_daily": base.rel_path(OHLC_PATH), "edge_monthly": base.rel_path(EDGE_PATH)},
        "alignment": align, "edge": ed, "checks": checks, "seed": seed,
        "versions": {"python": sys.version.split()[0], "bidask": bver,
                     **{k: im.version(k) for k in ["pandas", "numpy", "statsmodels"]}},
        "spec": {"lag_months": lag, "nw_lags": lags, "quantiles": q, "min_pairs": min_pairs, "winsor": winsor,
                 "rules": RULES, "main_rule": "floor", "windows": WINDOWS, "main_window": "m1",
                 "cost_rules": COST_RULES, "main_cost_rule": "pooled", "fee_per_side_extra_variant": FEE_PER_SIDE, "controls": ["mom", "ret_1"], "dep_var": "ret_adj"},
        "timings_sec": timings | {"total": round(sum(timings.values()), 3)},
        "tables": sorted(f"{k}.csv" for k in tables) + ["edge_bidask_check.csv", "SUMMARY.md"],
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"[edge_spread] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
