"""Upgrade 6: Lou & Shu (2017) on a second market, MOEX board TQBR (Russian common shares).

Same machinery as the Binance run (measures.monthly_measures, winsorize_cs, cs_residual,
factors.*, fama_macbeth, sorts.sort_table), applied to
  A, A_C   : daily RUB volume VALUE                       (x 1e6, as Table 1 of the paper)
  AT, AT_C : daily turnover TO = VOLUME / ISSUESIZE       (current ISSUESIZE, split-adjusted volume)
with the controls the paper uses where they exist here: ln(ME) (ME = close x ISSUESIZE at the end
of the previous year), Ret[-12,-2], Ret[-1], IVOL. No B/M (deviation M5).

Reproduces T1 (corr A-A_C, AT-AT_C, Q75/Q25), T3 (quintiles; terciles as robustness), T4 models
(1)-(5) for A and AT, NW(6), and writes a paper-vs-MOEX CSV and a code-generated SUMMARY.md.

Pre-registered choices (fixed before any MOEX result was computed; see MOEX_SPEC):
  * universe: TYPE == common_share (no preferred shares, depositary receipts, funds), as share
    codes 10/11 in the paper; >= 10 valid days in the estimation month;
    main filter: mean daily VALUE >= 100,000 RUB in the estimation month (removes only near-dead
    listings, keeps the illiquid tail where the effect lives, as the paper keeps all NYSE/AMEX stocks);
    robustness: >= 1,000,000 RUB (about USD 10-15k, the order of the Binance filter).
  * first estimation month: the first month with >= 100 eligible stocks (TQBR became the main
    board only in mid-2014); returns until 2026-09.
  * risk adjustment: EW market of the eligible universe (CMKT analogue), betas over [t-60, t-1] with
    >= 24 months as in the paper; sorting alphas on MKT + SMB (terciles of ln ME) + MOM (terciles of
    Ret[-12,-2]); r_f = 0 (no effect on FM slopes or H-L spreads).
  * variants reported, all of them: main; filter 1M RUB; price returns without dividends; raw
    (not risk-adjusted) returns in FM; terciles instead of quintiles.

Usage:  python -I src/run_moex.py config.yaml [--out results/upgrades/moex] [--workers 6]
Reads:  results/LATEST (base run id, recorded for provenance; MOEX data is independent of it)
Writes: results/upgrades/moex/*.csv, SUMMARY.md, run_manifest.json; data/cache/moex_panel.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import copy
import hashlib
import json
import logging
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml

import data_moex
from analysis import _fm_block, _no_attrs
from factors import cmkt_monthly, ivol_monthly, rolling_betas
from fama_macbeth import fama_macbeth, nw_mean
from measures import cs_residual, mean_cs_corr, monthly_measures, winsorize_cs
from sorts import hl_factor, portfolio_returns, sort_table

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CODE_FILES = ["data_moex.py", "run_moex.py", "measures.py", "factors.py", "fama_macbeth.py", "sorts.py", "analysis.py"]
log = logging.getLogger("run_moex")

MOEX_SPEC = {
    "data": data_moex.DATA_SPEC,
    "universe": {"types": ["common_share"], "min_valid_days": 10, "min_mean_value_rub": 1.0e5,
                 "robust_min_mean_value_rub": 1.0e6, "min_stocks_first_month": 100},
    "measures": {"scale_A": 1.0e6, "scale_AT": 1.0, "winsor": [0.01, 0.99], "lag_months": 2},
    "period": {"return_end": "2026-09"},
    "factors": {"beta_window": 60, "beta_min_obs": 24, "factor_quantiles": 3},
    "controls": {"mom_window": [-12, -2], "ivol_months": 12, "ivol_min_days": 120},
    "tests": {"quantiles": 5, "robust_quantiles": 3, "nw_lags": 6, "hlz_threshold": 3.0, "t_crit": 1.96},
}

# Lou & Shu (2017) turnover-based numbers (Panel B of Tables 3 and 4; Table 1). The dollar-volume
# numbers are read from src/paper_numbers.yaml.
PAPER_AT = {
    "source": "Lou & Shu (2017): Table 1 (p. 43), Table 3 Panel B (p. 45), Table 4 Panel B (p. 46)",
    "table1": {"AT": {"q25": 5.68, "q50": 14.46, "q75": 35.87}, "AT_C": {"q25": 423.06, "q50": 1169.49, "q75": 2989.59},
               "corr_A_AT": 0.691, "corr_AT_ATC": 0.746, "corr_A_ATC": 0.312, "corr_AT_absret": 0.347,
               "corr_ATC_absret": -0.040},
    "table3B": {"AT": {"raw": (0.39, 2.64), "alpha": (0.49, 3.65)},
                "AT_C": {"raw": (0.33, 2.39), "alpha": (0.55, 4.47)},
                "ResAT": {"raw": (0.18, 0.74), "alpha": (-0.03, -0.21)}},
    "table4B": {"(1)": {"lnAT": (0.163, 3.95)}, "(2)": {"lnAT_C": (0.223, 5.77)}, "(3)": {"ResLnAT": (-0.246, -3.02)},
                "(4)": {"lnAT_C": (0.198, 5.10), "ResLnAT": (-0.207, -2.53)},
                "(5)": {"lnAT_C": (0.192, 5.12), "ResLnAT": (-0.267, -3.79), "ivol": (-2.768, -0.54)}},
    "table4A_lnME": {"(1)": (0.080, 2.04), "(2)": (0.144, 3.14), "(3)": (-0.067, -2.48), "(4)": (0.069, 1.82), "(5)": (0.062, 1.74)},
}

# MOEX-specific deviations from the paper (written to deviations.csv; proposed for DEVIATIONS.md).
DEVIATIONS_MOEX = [
    ("M1", "NYSE/AMEX 1964-2012, CRSP", "MOEX TQBR, common shares, measures from mid-2014, returns to 2026-09",
     "Second market (upgrade 6); TQBR became the main T+ board only in mid-2014 (EQBR before)",
     "Far fewer months and stocks (about 100-250 per month) -> lower power"),
    ("M2", "Share codes 10/11", "TYPE == common_share (no preferred shares, DRs, funds); mean daily VALUE >= 100k RUB "
     "(robustness 1M RUB); >= 10 valid days", "Closest analogue of ordinary common shares; filter fixed before results",
     "Thresholds are nominal RUB (inflation ignored)"),
    ("M3", "Shares outstanding from CRSP each month", "ISSUESIZE as of today for all months; volume split-adjusted to today's units",
     "ISS gives only the current ISSUESIZE", "Turnover and ME are mis-measured for firms with buybacks/SPOs; "
     "measurement error attenuates AT_C and ln ME"),
    ("M4", "CRSP total returns", "Dividends from dohod.ru (fallback smart-lab.ru), ex-date from registry date and "
     "T+2/T+1 rule; splits from ISS /statistics/engines/stock/splits", "ISS /securities/<SECID>/dividends is retired; "
     "CCI requires login", "Missing dividends bias returns of payers (large, liquid stocks) down; variant without dividends reported"),
    ("M5", "Controls ln ME, B/M, mom, rev", "ln ME, mom, rev (no B/M)", "No book values in ISS", "Control set smaller"),
    ("M6", "FF3-adjusted returns, FF3+UMD alphas", "Beta-adjusted on EW TQBR market (60m, >= 24); alphas on MKT + SMB "
     "(ln ME terciles) + MOM (Ret[-12,-2] terciles); r_f = 0", "No Russian FF factors with HML", "Risk adjustment coarser"),
    ("M7", "Continuous trading", "Trading halt 2022-02-28..2022-03-23: no daily return across it; Feb->Mar 2022 monthly "
     "return kept", "Daily return over a month-long gap is not a daily return", "March 2022 has < 10 valid days -> no measures"),
    ("M8", "PERMNO links across ticker changes", "SECID is the identifier; renamed/re-registered tickers are separate stocks",
     "ISS history is per SECID", "A few series end early (like a delisting at the last price)"),
    ("M9", "CRSP corporate-action adjustments", "Two verified manual corrections after a scan of all daily moves > x2.5 "
     "not explained by a recorded split/dividend: BELU 1:8 split on 2024-08-22 (not in ISS split history; ratio 8 "
     "escapes the power-of-10 inference rule); LNZL 2021-07 dividend 15,219.50 RUB kept although its yield (65%) "
     "exceeds the 50% error guard", "Public sources (MOEX notice / company board decision), see data_flags.csv",
     "Removes a spurious -88% month (BELU 2024-08) and a spurious large negative month (LNZL 2021-07); "
     "the main specification without these corrections is reported as a variant"),
]

WINSOR_COLS = ["A", "A_C", "absret", "dvol", "AT", "AT_C", "TO", "ivol"]
LOGS = [("A", "lnA"), ("A_C", "lnA_C"), ("absret", "ln_absret"), ("dvol", "lnVOL"),
        ("AT", "lnAT"), ("AT_C", "lnAT_C"), ("TO", "lnTO")]


# ---------------------------------------------------------------- helpers
def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        out[k] = deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info() -> dict:
    import subprocess
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        st = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
        return {"sha": sha, "dirty": bool(st.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"sha": None, "dirty": None}


def csv_ready(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index()
    for c in df.columns:
        if isinstance(df[c].dtype, pd.PeriodDtype):
            df[c] = df[c].astype(str)
    return df


# ---------------------------------------------------------------- panel
def _as_period(s: pd.Series) -> pd.Series:
    return s if isinstance(s.dtype, pd.PeriodDtype) else pd.PeriodIndex(s.astype(str), freq="M").to_series(index=s.index)


def monthly_frames(daily: pd.DataFrame, info: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """One row per (symbol, month): A-family on VALUE, AT-family on turnover, ret (TRI), ME inputs."""
    d = daily.copy()
    d["month"] = _as_period(d["month"])
    d["date"] = pd.to_datetime(d["date"])
    iss = info.set_index("secid")["issuesize"]
    d["issuesize"] = d["symbol"].map(iss)
    d["close_adj"] = d["close"] / d["vol_mult"]                     # price in today's share units
    mcfg = spec["measures"]

    dA = d.assign(quote_volume=d["value"], close=d["tri"])
    mA = monthly_measures(dA, scale=float(mcfg["scale_A"])).drop(columns=["ret"])
    to = d["volume_adj"] / d["issuesize"]
    dT = d.assign(quote_volume=to, close=d["tri"], valid=d["valid"] & (to > 0) & np.isfinite(to))
    mT = monthly_measures(dT, scale=float(mcfg["scale_AT"]))[["symbol", "month", "A", "A_C", "dvol", "n_days"]]
    mT = mT.rename(columns={"A": "AT", "A_C": "AT_C", "dvol": "TO", "n_days": "n_days_T"})
    ret = data_moex.monthly_tri_returns(d)
    m = ret.merge(mA, on=["symbol", "month"], how="left").merge(mT, on=["symbol", "month"], how="left")
    m["n_days"] = m["n_days"].fillna(0).astype(int)

    # ME at the end of the previous calendar year: last December close (split-adjusted) x ISSUESIZE
    dec = d[d["date"].dt.month == 12].sort_values("date").groupby(["symbol", d["date"].dt.year])["close_adj"].last()
    me = (dec * dec.index.get_level_values(0).map(iss).to_numpy() / 1e6).rename("ME").reset_index()
    me.columns = ["symbol", "year", "ME"]                                  # million RUB
    me["year"] = me["year"] + 1                                            # applies to returns of year+1
    m["year"] = m["month"].dt.year
    m = m.merge(me, on=["symbol", "year"], how="left").drop(columns=["year"])
    return m


def first_measure_month(monthly: pd.DataFrame, info: pd.DataFrame, spec: dict) -> pd.Period:
    """First estimation month with >= min_stocks_first_month eligible stocks under the MAIN filter
    (fixed once and shared by all variants, so that variants differ only in what they vary)."""
    ucfg = spec["universe"]
    common = set(info.loc[info["type"].isin(ucfg["types"]), "secid"])
    m = monthly
    e = m["symbol"].isin(common) & (m["n_days"] >= ucfg["min_valid_days"]) & (m["dvol"] >= ucfg["min_mean_value_rub"])
    counts = m.loc[e].groupby("month").size()
    ok = counts[counts >= ucfg["min_stocks_first_month"]]
    if ok.empty:
        raise ValueError("no month reaches min_stocks_first_month eligible stocks")
    return ok.index.min()


def build_moex_panel(monthly: pd.DataFrame, daily: pd.DataFrame, info: pd.DataFrame, spec: dict,
                     min_value: float, first_meas: pd.Period | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (panel, factors, meta) mirroring analysis.build_panel, with AT family and ln ME."""
    ucfg, mcfg, fcfg, ccfg = spec["universe"], spec["measures"], spec["factors"], spec["controls"]
    common = set(info.loc[info["type"].isin(ucfg["types"]), "secid"])
    m = monthly.copy()
    eligible = m["symbol"].isin(common) & (m["n_days"] >= ucfg["min_valid_days"]) & (m["dvol"] >= min_value)
    if first_meas is None:
        first_meas = first_measure_month(m, info, spec)
    eligible &= m["month"] >= first_meas
    lag = int(mcfg["lag_months"])
    start, end = first_meas + lag, pd.Period(spec["period"]["return_end"], "M")

    d = daily.copy()
    d["month"] = _as_period(d["month"])
    d["date"] = pd.to_datetime(d["date"])
    ivol = ivol_monthly(d, ccfg["ivol_months"], ccfg["ivol_min_days"])
    sig = m.loc[eligible].drop(columns=["ret", "ME"]).merge(ivol, on=["symbol", "month"], how="left")
    sig = winsorize_cs(sig, WINSOR_COLS, "month", tuple(mcfg["winsor"]))
    with np.errstate(divide="ignore", invalid="ignore"):
        for c, lc in LOGS:
            sig[lc] = np.log(sig[c]).replace(-np.inf, np.nan)
    sig["ResA"] = cs_residual(sig, "A", "A_C", "month")
    sig["ResLnA"] = cs_residual(sig, "lnA", "lnA_C", "month")
    sig["ResAT"] = cs_residual(sig, "AT", "AT_C", "month")
    sig["ResLnAT"] = cs_residual(sig, "lnAT", "lnAT_C", "month")
    sig["month"] = sig["month"] + lag

    cmkt = cmkt_monthly(m, eligible)
    betas = rolling_betas(m, cmkt, fcfg["beta_window"], fcfg["beta_min_obs"])
    R = m.pivot(index="month", columns="symbol", values="ret")
    R = R.reindex(pd.period_range(R.index.min(), R.index.max(), freq="M"))
    lo, hi = ccfg["mom_window"]
    mom = np.exp(np.log1p(R).shift(-hi).rolling(hi - lo + 1, min_periods=hi - lo + 1).sum()) - 1
    ctrl = pd.DataFrame({"ret_1": R.shift(1).stack(future_stack=True),
                         "mom": mom.stack(future_stack=True)}).reset_index()
    ctrl.columns = ["month", "symbol", "ret_1", "mom"]

    p = m[["symbol", "month", "ret", "ME"]].merge(sig, on=["symbol", "month"], how="inner")
    p = p.merge(ctrl, on=["symbol", "month"], how="left").merge(betas, on=["symbol", "month"], how="left")
    p["lnME"] = np.log(p["ME"])
    p["CMKT"] = p["month"].map(cmkt)
    p["ret_adj"] = p["ret"] - p["beta"] * p["CMKT"]
    for c in ["ret", "ret_adj"]:
        p[c] = 100.0 * p[c]
    p = p[(p["month"] >= start) & (p["month"] <= end)].dropna(subset=["ret"]).reset_index(drop=True)

    q = fcfg["factor_quantiles"]
    cmom = hl_factor(p, "mom", "ret", q).rename("CMOM")
    csmb = (-hl_factor(p, "lnME", "ret", q)).rename("CSMB")               # small minus big
    factors = pd.concat([(100.0 * cmkt).rename("CMKT"), csmb, cmom], axis=1).loc[np.sort(p["month"].unique())]
    meta = {"first_measure_month": str(first_meas), "return_start": str(start), "return_end": str(end),
            "min_mean_value_rub": min_value}
    return p, factors, meta


# ---------------------------------------------------------------- tests
FM_A = {"(1)": ["lnA"], "(2)": ["lnA_C"], "(3)": ["ResLnA"], "(4)": ["lnA_C", "ResLnA"],
        "(5)": ["lnA_C", "ResLnA", "ivol"]}
FM_AT = {"(1)": ["lnAT"], "(2)": ["lnAT_C"], "(3)": ["ResLnAT"], "(4)": ["lnAT_C", "ResLnAT"],
         "(5)": ["lnAT_C", "ResLnAT", "ivol"]}
CONTROLS = ["lnME", "mom", "ret_1"]


def describe(panel: pd.DataFrame) -> pd.DataFrame:
    cols = ["A", "AT", "A_C", "AT_C", "absret", "dvol", "TO", "ME"]
    d = panel[cols].describe(percentiles=[.1, .25, .5, .75, .9]).T
    d["Q75/Q25"] = d["75%"] / d["25%"]
    return d.reset_index(names="measure")


def sorts(panel, factors, q, lags, signals=("A", "A_C", "ResA", "AT", "AT_C", "ResAT")) -> pd.DataFrame:
    fac = factors[["CMKT", "CSMB", "CMOM"]]
    out = []
    for s in signals:
        tab = sort_table(panel, s, "ret", q, fac, lags)
        tab.insert(0, "signal", s)
        tab.insert(1, "quantiles", q)
        out.append(_no_attrs(tab.reset_index(names="row")))
    return pd.concat(out, ignore_index=True)


def fm(panel, y, lags) -> pd.DataFrame:
    a = _fm_block(panel, y, FM_A, lags, CONTROLS)
    a.insert(0, "family", "A (RUB volume)")
    t = _fm_block(panel, y, FM_AT, lags, CONTROLS)
    t.insert(0, "family", "AT (turnover)")
    return pd.concat([a, t], ignore_index=True)


def run_variant(name: str, monthly, daily, info, spec, min_value, first_meas=None):
    lags = spec["tests"]["nw_lags"]
    panel, factors, meta = build_moex_panel(monthly, daily, info, spec, min_value, first_meas)
    out = {"T3": sorts(panel, factors, spec["tests"]["quantiles"], lags),
           "T3_terciles": sorts(panel, factors, spec["tests"]["robust_quantiles"], lags),
           "T4": fm(panel, "ret_adj", lags)}
    for v in out.values():
        v.insert(0, "variant", name)
    meta.update({"variant": name, "n_months": int(panel["month"].nunique()), "n_stock_months": int(len(panel)),
                 "n_stocks": int(panel["symbol"].nunique()),
                 "avg_stocks_per_month": float(panel.groupby("month").size().mean()),
                 "avg_obs_fm_A1": float(out["T4"].query("family=='A (RUB volume)' and model=='(1)'")["avg_obs"].iloc[0]),
                 "n_months_fm_A1": int(out["T4"].query("family=='A (RUB volume)' and model=='(1)'")["n_months"].iloc[0])})
    return out, panel, factors, meta


def alpha_detail(panel, factors, q, lags, signals=("A", "A_C", "ResA", "AT", "AT_C", "ResAT")) -> pd.DataFrame:
    """H-L alpha regressions in full: months used, raw H-L on the SAME months, factor loadings (NW t),
    and the mean ln ME of the bottom / top quantile (is the illiquid leg a small-stock leg?)."""
    import statsmodels.api as sm
    fac = factors[["CMKT", "CSMB", "CMOM"]]
    rows = []
    for sig in signals:
        pr = portfolio_returns(panel, sig, "ret", q)
        hl = (pr[q] - pr[1]).rename("y")
        df = pd.concat([hl, fac], axis=1, join="inner").dropna()
        res = sm.OLS(df["y"], sm.add_constant(df[fac.columns])).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
        raw_all, raw_same = nw_mean(hl, lags), nw_mean(df["y"], lags)
        pq = panel[["month", sig, "lnME"]].dropna(subset=[sig]).copy()
        pq["qq"] = pq.groupby("month")[sig].transform(
            lambda x: pd.qcut(x.rank(method="first"), q, labels=False) + 1 if x.notna().sum() >= q else np.nan)
        me = pq.groupby("qq")["lnME"].mean()
        r = {"signal": sig, "quantiles": q, "n_months_raw": int(hl.notna().sum()), "raw_hl": raw_all[0],
             "raw_hl_t": raw_all[2], "n_months_alpha": int(len(df)), "raw_hl_same_months": raw_same[0],
             "raw_hl_same_months_t": raw_same[2], "alpha": float(res.params["const"]),
             "alpha_t": float(res.tvalues["const"])}
        for f in fac.columns:
            r[f"load_{f}"], r[f"load_{f}_t"] = float(res.params[f]), float(res.tvalues[f])
        r["lnME_bottom_q"], r["lnME_top_q"] = float(me.get(1, np.nan)), float(me.get(q, np.nan))
        rows.append(r)
    return pd.DataFrame(rows)


def diag_by_year(panel, q, lags) -> pd.DataFrame:
    """POST-HOC (added at the VERIFY stage): per calendar year, the mean quintile H-L on A_C (raw) and
    the mean FM(4) slope on ln A_C (ret_adj), and both full-sample statistics with that year left out.
    Every year is dropped in turn (no year singled out)."""
    pr = portfolio_returns(panel, "A_C", "ret", q)
    hl = (pr[q] - pr[1]).dropna()
    fmr = fama_macbeth(panel, "ret_adj", FM_A["(4)"] + CONTROLS, lags=lags)
    b = fmr.attrs["coef_series"]["lnA_C"].dropna()
    years = sorted(set(hl.index.year) | set(b.index.year))
    rows = []
    for y in years:
        h_in, b_in = hl[hl.index.year == y], b[b.index.year == y]
        h_out, b_out = nw_mean(hl[hl.index.year != y], lags), nw_mean(b[b.index.year != y], lags)
        rows.append({"year": y, "n_months_hl": len(h_in), "hl_A_C_mean": h_in.mean() if len(h_in) else np.nan,
                     "hl_A_C_share_of_cum": h_in.sum() / hl.sum(),
                     "hl_A_C_excl_year": h_out[0], "hl_A_C_excl_year_t": h_out[2],
                     "n_months_fm": len(b_in), "fm4_lnA_C_mean": b_in.mean() if len(b_in) else np.nan,
                     "fm4_lnA_C_excl_year": b_out[0], "fm4_lnA_C_excl_year_t": b_out[2]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- data diagnostics
def ex_date_check(daily: pd.DataFrame) -> pd.DataFrame:
    """Price drop on ex-dates vs dividend yield: slope ~ 1 if ex-dates are right, ~ 0 on the day before."""
    d = daily.sort_values(["symbol", "date"]).copy()
    d["prev_close"] = d.groupby("symbol")["close"].shift(1)
    d["pr"] = d["close"] * d["split"] / d["prev_close"] - 1.0                 # price-only return
    d["y"] = d["div"] / d["prev_close"]
    d["y_next"] = d.groupby("symbol")["y"].shift(-1)
    rows = []
    for name, ycol in [("ex-date", "y"), ("day before ex-date", "y_next")]:
        e = d[(d[ycol] > 0) & d["r"].notna()]
        x = -e[ycol].to_numpy()
        yv = e["pr"].to_numpy()
        slope = float(np.polyfit(x, yv, 1)[0]) if len(e) > 2 else np.nan
        rows.append({"day": name, "n_events": int(len(e)), "mean_div_yield": float(e[ycol].mean()),
                     "mean_price_return": float(e["pr"].mean()),
                     "slope_price_return_on_minus_yield": slope})
    return pd.DataFrame(rows)


def universe_counts(universe, info, monthly, spec) -> pd.DataFrame:
    u = universe.merge(info[["secid", "type"]], on="secid", how="left")
    u["month"] = _as_period(u["month"])
    listed = u.groupby("month").size().rename("listed_TQBR_month_end")
    com = u[u["type"] == "common_share"].groupby("month").size().rename("listed_common")
    m = monthly.copy()
    common = set(info.loc[info["type"].isin(spec["universe"]["types"]), "secid"])
    base = m["symbol"].isin(common) & (m["n_days"] >= spec["universe"]["min_valid_days"])
    e1 = m[base & (m["dvol"] >= spec["universe"]["min_mean_value_rub"])].groupby("month").size().rename("eligible_main")
    e2 = m[base & (m["dvol"] >= spec["universe"]["robust_min_mean_value_rub"])].groupby("month").size().rename("eligible_1m")
    return pd.concat([listed, com, e1, e2], axis=1).reset_index(names="month")


# ---------------------------------------------------------------- paper vs MOEX
def _pick(df, **kw):
    q = df
    for k, v in kw.items():
        q = q[q[k] == v]
    return q


def paper_vs_moex(t1d, t1c, t3, t4, paper: dict, tcrit: float, hlz: float) -> pd.DataFrame:
    ls = paper["lou_shu_2017"]
    rows = []

    def add(item, p_val, p_t, m_val, m_t, kind):
        r = {"item": item, "paper": p_val, "paper_t": p_t, "moex": m_val, "moex_t": m_t}
        if kind == "t" and p_val is not None and np.isfinite(m_val):
            r["same_sign"] = bool(np.sign(p_val) == np.sign(m_val))
            r["moex_sig_1.96"] = bool(abs(m_t) > tcrit)
            r["moex_sig_hlz3"] = bool(abs(m_t) > hlz)
        rows.append(r)

    c = t1c.set_index("measure")
    add("corr(A, A_C)", ls["table1_panelB"]["corr_A_AC"], None, c.loc["A", "A_C"], None, "level")
    add("corr(AT, AT_C)", PAPER_AT["table1"]["corr_AT_ATC"], None, c.loc["AT", "AT_C"], None, "level")
    add("corr(A, AT)", PAPER_AT["table1"]["corr_A_AT"], None, c.loc["A", "AT"], None, "level")
    add("corr(A, |Ret|)", ls["table1_panelB"]["corr_A_absret"], None, c.loc["A", "absret"], None, "level")
    add("corr(A_C, |Ret|)", ls["table1_panelB"]["corr_AC_absret"], None, c.loc["A_C", "absret"], None, "level")
    add("corr(AT_C, |Ret|)", PAPER_AT["table1"]["corr_ATC_absret"], None, c.loc["AT_C", "absret"], None, "level")
    dd = t1d.set_index("measure")
    t1a = ls["table1_panelA"]
    for meas, pq in [("A", t1a["A"]), ("A_C", t1a["A_C"]), ("absret", t1a["absret"]),
                     ("AT", PAPER_AT["table1"]["AT"]), ("AT_C", PAPER_AT["table1"]["AT_C"])]:
        add(f"Q75/Q25 of {meas}", pq["q75"] / pq["q25"], None, dd.loc[meas, "Q75/Q25"], None, "level")

    t3p = ls["table3_panelA"]
    for sig in ["A", "A_C", "ResA", "AT", "AT_C", "ResAT"]:
        for row in ["raw", "alpha"]:
            if sig in t3p:
                pv, pt = t3p[sig][row]["hl"], t3p[sig][row]["t"]
            else:
                pv, pt = PAPER_AT["table3B"][sig][row]
            mr = _pick(t3, signal=sig, row=row).iloc[0]
            lab = "raw" if row == "raw" else "alpha (paper: FF3+UMD; MOEX: MKT+SMB+MOM)"
            add(f"Quintile H-L on {sig}, {lab} (%/mo)", pv, pt, float(mr["H-L"]), float(mr["t"]), "t")

    t4p = ls["table4_panelA"]
    for fam, models, ptab in [("A (RUB volume)", FM_A, t4p), ("AT (turnover)", FM_AT, PAPER_AT["table4B"])]:
        for mod, xs in models.items():
            for x in xs:
                pe = ptab[mod][x]
                pv, pt = (pe["coef"], pe["t"]) if isinstance(pe, dict) else pe
                mr = _pick(t4, family=fam, model=mod, var=x).iloc[0]
                add(f"FM {mod} {x}", pv, pt, float(mr["coef"]), float(mr["t"]), "t")
    for mod in FM_A:
        pv, pt = PAPER_AT["table4A_lnME"][mod]
        mr = _pick(t4, family="A (RUB volume)", model=mod, var="lnME").iloc[0]
        add(f"FM {mod} lnME (A family)", pv, pt, float(mr["coef"]), float(mr["t"]), "t")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- SUMMARY.md (generated from CSVs)
def _f(x, nd=2):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def summary_md(out: Path, base_run_id: str, run_id: str) -> str:
    pv = pd.read_csv(out / "paper_vs_moex.csv")
    meta = pd.read_csv(out / "sample.csv")
    var = pd.read_csv(out / "variants_key.csv")
    exd = pd.read_csv(out / "ex_date_check.csv")
    cov = pd.read_csv(out / "dividend_coverage.csv").iloc[0]
    xc = pd.read_csv(out / "dividend_crosscheck.csv").iloc[0]
    main = meta[meta["variant"] == "main"].iloc[0]
    g = lambda item: pv[pv["item"] == item].iloc[0]

    L = [f"# Апгрейд 6: второй рынок — MOEX, режим TQBR", "",
         f"_Сгенерировано `src/run_moex.py`, run_id `{run_id}`, базовый прогон `{base_run_id}`. "
         "Все числа — из CSV в этой папке._", "",
         "## Что проверяли",
         "Те же меры и тесты, что и на Binance, на обыкновенных акциях Мосбиржи (режим TQBR): "
         "`A`, `A_C` по рублёвому объёму `VALUE`, `AT`, `AT_C` по обороту `VOLUME/ISSUESIZE`; "
         "T1 (корреляции, Q75/Q25), T3 (квинтили, терцили — робастность), T4 (модели 1–5 для `A` и `AT`), "
         "контроли `ln ME`, `Ret[-12,-2]`, `Ret[-1]`, Newey–West(6).", "",
         f"Выборка: доходности {main['return_start']}–{main['return_end']} "
         f"({int(main['n_months'])} мес.), {int(main['n_stocks'])} акций, "
         f"в среднем {main['avg_stocks_per_month']:.0f} акций в месяц "
         f"(фильтр: средний дневной `VALUE` ≥ {main['min_mean_value_rub']:,.0f} руб.); "
         f"FM модели (1): {int(main['n_months_fm_A1'])} мес., в среднем {main['avg_obs_fm_A1']:.0f} наблюдений.", "",
         "## Ключевые числа: статья против MOEX", "",
         "| Показатель | Статья | t | MOEX | t |", "|---|---|---|---|---|"]
    keys = ["corr(A, A_C)", "corr(AT, AT_C)", "Q75/Q25 of A_C", "Q75/Q25 of absret",
            "Quintile H-L on A, raw (%/mo)", "Quintile H-L on A_C, raw (%/mo)", "Quintile H-L on ResA, raw (%/mo)",
            "Quintile H-L on AT, raw (%/mo)", "Quintile H-L on AT_C, raw (%/mo)", "Quintile H-L on ResAT, raw (%/mo)",
            "FM (1) lnA", "FM (2) lnA_C", "FM (4) lnA_C", "FM (4) ResLnA",
            "FM (1) lnAT", "FM (2) lnAT_C", "FM (4) lnAT_C", "FM (4) ResLnAT"]
    for k in keys:
        r = g(k)
        L.append(f"| {k} | {_f(r['paper'], 3)} | {_f(r['paper_t'])} | {_f(r['moex'], 3)} | {_f(r['moex_t'])} |")
    L.append("")

    # interpretation strictly from the numbers; thresholds and data-spec values from the run manifest
    man = json.loads((out / "run_manifest.json").read_text())
    spec = man["spec"]
    tc, hz = float(spec["tests"]["t_crit"]), float(spec["tests"]["hlz_threshold"])
    ad = pd.read_csv(out / "T3_alpha_detail.csv").set_index("signal")
    yr = pd.read_csv(out / "diag_by_year.csv")
    uc = pd.read_csv(out / "universe_counts.csv")
    flags = pd.read_csv(out / "data_flags.csv")
    devs = pd.read_csv(out / "deviations.csv")
    paper_sample = yaml.safe_load((SRC / "paper_numbers.yaml").read_text())["lou_shu_2017"]["sample"]

    def verdict(item):
        r = g(item)
        t = r["moex_t"]
        sig = f"значим (|t| > {tc})" if abs(t) > tc else "не значим"
        hlz = f", проходит порог HLZ t > {hz}" if abs(t) > hz else ""
        same = "тот же знак, что в статье" if np.sign(r["moex"]) == np.sign(r["paper"]) else "знак противоположен статье"
        return f"{_f(r['moex'], 3)} (t = {_f(t)}): {sig}{hlz}, {same}"
    vk = var.set_index("variant")

    def vt(v, col, nd=3):
        x, t = vk.loc[v, col], vk.loc[v, f"{col}_t"]
        return f"{_f(x, nd)} (t = {_f(t)})"

    def sig(v, col):
        t = vk.loc[v, f"{col}_t"]
        return bool(np.isfinite(t) and t > tc)

    vol_fm = sig("main", "FM4_lnA_C") and sig("main", "FM4_lnAT_C")
    vol_fm_hlz = vk.loc["main", "FM4_lnA_C_t"] > hz and vk.loc["main", "FM4_lnAT_C_t"] > hz
    vol_items = [i for i in pv["item"] if i.startswith("FM (") and i.split()[-1] in ("lnA", "lnA_C", "lnAT", "lnAT_C")]
    hlz_pass = [i for i in vol_items if g(i)["moex_t"] > hz]
    raw_v, diag_v = "main, raw returns", "diagnostic: raw returns, months of ret_adj FM"
    res_pos = any(g(k)["moex_t"] > tc for k in ["FM (4) ResLnA", "FM (4) ResLnAT", "Quintile H-L on ResA, raw (%/mo)"])
    a_ac = ad.loc["A_C"]
    absorbed = abs(a_ac["alpha"]) < abs(a_ac["raw_hl_same_months"]) / 3
    top = yr.loc[yr["hl_A_C_share_of_cum"].idxmax()]
    L += ["## Интерпретация (по числам выше)",
          f"Пороги: |t| > {tc} (5%) и t > {hz} (Harvey–Liu–Zhu, 2016).", "",
          f"- Объёмная компонента в FM (зависимая — доходность, скорректированная на бету к EW-рынку, как в статье): "
          f"`ln A_C` в модели (4) — {verdict('FM (4) lnA_C')}; `ln AT_C` — {verdict('FM (4) lnAT_C')}; "
          f"модель (1) `ln A` — {verdict('FM (1) lnA')}.",
          f"- Компонента price impact: `Res.lnA` в модели (4) — {verdict('FM (4) ResLnA')}; "
          f"`Res.lnAT` — {verdict('FM (4) ResLnAT')}; "
          + ("ни одна не значимо положительна — в этой части вывод статьи («price impact не прайсится») воспроизводится, "
             "но и отрицательного знака статьи значимо нет." if not res_pos else
             "есть значимо положительная оценка — вывод статьи в этой части не воспроизводится."),
          f"- Сортировки (сырые доходности): H–L по `A` — {verdict('Quintile H-L on A, raw (%/mo)')}; по `A_C` — "
          f"{verdict('Quintile H-L on A_C, raw (%/mo)')}; по `Res.A` — {verdict('Quintile H-L on ResA, raw (%/mo)')}.",
          f"- Альфы H–L на MKT+SMB+MOM: `A` — {verdict('Quintile H-L on A, alpha (paper: FF3+UMD; MOEX: MKT+SMB+MOM) (%/mo)')}; "
          f"`A_C` — {verdict('Quintile H-L on A_C, alpha (paper: FF3+UMD; MOEX: MKT+SMB+MOM) (%/mo)')}. "
          f"Альфа считается на {int(a_ac['n_months_alpha'])} мес. (нужны все три фактора), сырой H–L — на "
          f"{int(a_ac['n_months_raw'])}; на тех же {int(a_ac['n_months_alpha'])} мес. сырой H–L по `A_C` — "
          f"{_f(a_ac['raw_hl_same_months'], 3)} (t = {_f(a_ac['raw_hl_same_months_t'])}), по `A` — "
          f"{_f(ad.loc['A', 'raw_hl_same_months'], 3)} (t = {_f(ad.loc['A', 'raw_hl_same_months_t'])}). "
          f"Нагрузки H–L по `A_C` (`T3_alpha_detail.csv`): SMB {_f(a_ac['load_CSMB'])} (t = {_f(a_ac['load_CSMB_t'])}), "
          f"MOM {_f(a_ac['load_CMOM'])} (t = {_f(a_ac['load_CMOM_t'])}), MKT {_f(a_ac['load_CMKT'])} "
          f"(t = {_f(a_ac['load_CMKT_t'])}); средний ln ME нижнего/верхнего квинтиля — {_f(a_ac['lnME_bottom_q'])} / "
          f"{_f(a_ac['lnME_top_q'])}. "
          + ("Сырой спред (на тех же месяцах) почти целиком поглощается факторами: альфа меньше трети сырого H–L, "
             "основная нагрузка — на SMB (неликвидный квинтиль состоит из мелких бумаг)." if absorbed
             else "Альфа сохраняет заметную часть сырого спреда."),
          f"- Чувствительность к поправке на риск: FM (4) `ln A_C` на сырых доходностях — {vt(raw_v, 'FM4_lnA_C')}, "
          f"`ln A` в модели (1) — {vt(raw_v, 'FM1_lnA')}; на сырых доходностях за те же месяцы, что основной FM "
          f"(пост-хок диагностика) — {vt(diag_v, 'FM4_lnA_C')} и {vt(diag_v, 'FM1_lnA')}.",
          f"- Концентрация по годам (пост-хок, `diag_by_year.csv`, каждый год исключается по очереди): наибольший вклад "
          f"в накопленный H–L по `A_C` даёт {int(top['year'])} год — {100 * top['hl_A_C_share_of_cum']:.0f}% "
          f"(в среднем {_f(top['hl_A_C_mean'])}%/мес. за {int(top['n_months_hl'])} мес.); без него H–L = "
          f"{_f(top['hl_A_C_excl_year'], 3)} (t = {_f(top['hl_A_C_excl_year_t'])}), а FM (4) `ln A_C` = "
          f"{_f(top['fm4_lnA_C_excl_year'], 3)} (t = {_f(top['fm4_lnA_C_excl_year_t'])}); минимальный t FM (4) `ln A_C` "
          f"при исключении любого года — {_f(yr['fm4_lnA_C_excl_year_t'].min())}.",
          "- Итог: "
          + (("объёмная компонента значима в основном FM во всех моделях обеих семей (|t| > {})".format(tc)
              if all(g(i)["moex_t"] > tc for i in vol_items) else "объёмная компонента значима в основном FM в модели (4) "
              "обеих семей (не во всех моделях, см. `paper_vs_moex.csv`)") if vol_fm else
             "объёмная компонента в модели (4) основного FM не значима")
          + (f"; порог HLZ t > {hz} проходят только: " + ", ".join(f"{i} (t = {_f(g(i)['moex_t'])})" for i in hlz_pass)
             + (" — в модели (4) нет" if not vol_fm_hlz else "") if hlz_pass else f"; порог HLZ t > {hz} не проходит ни одна оценка")
          + ("; на сырых доходностях она " + ("тоже значима" if sig(raw_v, "FM4_lnA_C") else "не значима")
             + ", в сортировках по `A_C` " + ("значима" if g('Quintile H-L on A_C, raw (%/mo)')['moex_t'] > tc else "не значима")
             + ", в альфах портфелей " + ("значима" if g('Quintile H-L on A_C, alpha (paper: FF3+UMD; MOEX: MKT+SMB+MOM) (%/mo)')['moex_t'] > tc else "не значима"))
          + ". Направление (объём, а не price impact) совпадает со статьёй, но доказательность на MOEX слабее и "
          "зависит от спецификации; см. все варианты ниже.", ""]

    L += ["## Все варианты (ничего не отбирали)", "",
          "Варианты, заданные в `MOEX_SPEC` до первого расчёта: main, filter 1M RUB, no dividends, raw returns, terciles "
          "(файлы MOEX не были закоммичены до расчёта, поэтому порядок подтверждается записями сессии, а не git). "
          "Добавлены после первого прогона и помечены: `main without M9 corrections` (эффект двух проверенных "
          "исправлений данных), `diagnostic: raw returns, months of ret_adj FM`, `T3_alpha_detail.csv` и "
          "`diag_by_year.csv` (на этапе проверки).", "",
          "| Вариант | H–L A_C raw (t) | H–L A_C alpha (t) | H–L Res.A raw (t) | FM(1) lnA (t) | FM(4) lnA_C (t) | FM(4) ResLnA (t) | FM(4) lnAT_C (t) | FM(4) ResLnAT (t) |",
          "|---|---|---|---|---|---|---|---|---|"]

    def c(r, col, nd=2):
        return "—" if not np.isfinite(r[col]) else f"{_f(r[col], nd)} ({_f(r[col + '_t'])})"
    for _, r in var.iterrows():
        L.append(f"| {r['variant']} | {c(r, 'HL_A_C')} | {c(r, 'HL_A_C_alpha')} | {c(r, 'HL_ResA')} | {c(r, 'FM1_lnA', 3)} | "
                 f"{c(r, 'FM4_lnA_C', 3)} | {c(r, 'FM4_ResLnA', 3)} | {c(r, 'FM4_lnAT_C', 3)} | {c(r, 'FM4_ResLnAT', 3)} |")
    ex = exd.set_index("day")
    ds = spec["data"]
    um = uc[(uc["month"] >= str(main["first_measure_month"])) &
            (uc["month"] <= str(pd.Period(str(main["return_end"]), "M") - int(spec["measures"]["lag_months"])))]
    um = um["eligible_main"].dropna()
    nq = int(spec["tests"]["quantiles"])
    big = flags[flags["kind"] == "verified_large_dividend"].set_index("secid")["value"]
    m9 = "; ".join([f"сплит {v['secid']} {v['before']}:{v['after']} {v['date']} (нет в истории сплитов ISS)"
                    for v in ds.get("verified_splits", [])]
                   + [f"дивиденд {v['secid']} (реестр {v['registry_date']}, доходность {100 * float(big.loc[v['secid']]):.0f}% "
                      f"выше защитного порога {100 * float(ds['max_div_yield']):.0f}%)"
                      for v in ds.get("verified_large_dividends", []) if v["secid"] in big.index])
    L += ["", "## Оговорки",
          "- `ISSUESIZE` — только текущий (ISS не отдаёт историю): turnover и `ME` в прошлом считаются по "
          "сегодняшнему числу акций (объём пересчитан в текущие единицы с учётом сплитов).",
          f"- Дивиденды: эндпоинт ISS `/securities/<SECID>/dividends` выведен из эксплуатации, поэтому взяты с "
          f"dohod.ru (запасной источник smart-lab.ru): страницы есть у {int(cov['secids_with_page'])} из "
          f"{int(cov['secids'])} бумаг, {int(cov['events'])} выплат. Проверка дат отсечки: падение цены в "
          f"экс-дату к дивидендной доходности — наклон {_f(ex.loc['ex-date', 'slope_price_return_on_minus_yield'])} "
          f"(накануне — {_f(ex.loc['day before ex-date', 'slope_price_return_on_minus_yield'])}); "
          f"сверка источников: из {int(xc['events_matched'])} выплат, найденных и на dohod.ru, и на smart-lab.ru, "
          f"экс-даты совпали в {100 * xc['share_ex_date_equal']:.0f}% случаев "
          f"(±1 торговый день — {100 * xc['share_ex_date_within_1_trading_day']:.0f}%); "
          f"{int(xc['events_dohod_unmatched'])} из {int(xc['events_dohod'])} выплат dohod.ru не найдены на smart-lab.ru "
          "с той же суммой (±2%) и датой (±5 дней) — у smart-lab короче история и иначе пересчитаны суммы после сплитов.",
          f"- Остановка торгов {ds['halt'][0]}..{ds['halt'][1]}: дневная доходность через разрыв не считается; месячная "
          "доходность февраль→март 2022 сохранена (это реальная доходность держателя).",
          f"- Нет B/M; поправка на риск — EW-рынок TQBR (беты {spec['factors']['beta_window']} мес., минимум "
          f"{spec['factors']['beta_min_obs']}), альфа — MKT+SMB+MOM, r_f = 0.",
          f"- {int(um.min())}–{int(um.max())} акций в месяц (месяцы оценки мер): мощность ниже, чем у статьи "
          f"({paper_sample['months']} мес., в среднем {paper_sample['avg_obs']:,} акций); квинтили по "
          f"{int(um.min()) // nq}–{int(um.max()) // nq} бумаг, поэтому терцили приведены как робастность.",
          "- Тикеры, сменившие SECID (например, TCSG→T), — разные бумаги; доходность на стыке не склеивается.",
          f"- Исправления данных после первого прогона (M9, проверены по публичным источникам): {m9}. "
          "Флаги — `data_flags.csv`; вариант без исправлений — в таблице выше.",
          "- Шкала `ivol` (у нас — ст. откл. дневных остатков в долях) может отличаться от статьи: для `ivol` сравниваем знак и t, не величину.",
          f"- Полный список отступлений — `deviations.csv` ({devs['id'].iloc[0]}–{devs['id'].iloc[-1]}).", ""]
    return "\n".join(L)


VK_SORT = [("A", "raw"), ("A_C", "raw"), ("ResA", "raw"), ("AT_C", "raw"), ("ResAT", "raw"),
           ("A_C", "alpha"), ("ResA", "alpha")]
VK_FM = [("A (RUB volume)", "(1)", "lnA"), ("A (RUB volume)", "(4)", "lnA_C"), ("A (RUB volume)", "(4)", "ResLnA"),
         ("AT (turnover)", "(4)", "lnAT_C"), ("AT (turnover)", "(4)", "ResLnAT")]


def variants_key(t3q, t3t, t4_all) -> pd.DataFrame:
    """One row per variant (every variant that was run): key H-L spreads and FM slopes with t-stats.
    Sort-only variants (terciles) have empty FM cells; FM-only variants (raw returns) empty sort cells."""
    sources = [(v, t3q[t3q["variant"] == v], t4_all[t4_all["variant"] == v]) for v in t3q["variant"].unique()]
    sources += [(f"{v}, terciles", t3t[t3t["variant"] == v], t4_all.iloc[0:0]) for v in t3t["variant"].unique()]
    sources += [(v, t3q.iloc[0:0], t4_all[t4_all["variant"] == v])
                for v in t4_all["variant"].unique() if v not in set(t3q["variant"])]
    rows = []
    for v, a, b in sources:
        r = {"variant": v}
        for s, row in VK_SORT:
            x = a[(a["signal"] == s) & (a["row"] == row)]
            sfx = "" if row == "raw" else "_alpha"
            r[f"HL_{s}{sfx}"] = float(x["H-L"].iloc[0]) if len(x) else np.nan
            r[f"HL_{s}{sfx}_t"] = float(x["t"].iloc[0]) if len(x) else np.nan
        for fam, mod, var_ in VK_FM:
            x = b[(b["family"] == fam) & (b["model"] == mod) & (b["var"] == var_)]
            key = f"FM{mod.strip('()')}_{var_}"
            r[key] = float(x["coef"].iloc[0]) if len(x) else np.nan
            r[f"{key}_t"] = float(x["t"].iloc[0]) if len(x) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> str:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--out", default=str(ROOT / "results" / "upgrades" / "moex"))
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--cache-dir", default=str(ROOT / "data" / "cache"))
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg_path = Path(a.config)
    cfg = yaml.safe_load(cfg_path.read_text())
    spec = deep_update(MOEX_SPEC, cfg.get("moex", {}))
    spec_hash = hashlib.sha256(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()
    latest = Path(a.results_dir) / "LATEST"
    base_run_id = latest.read_text().strip() if latest.exists() else None
    seed = int(cfg.get("seed", 0))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"moex_{stamp}_{spec_hash[:8]}"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}

    def stage(name):
        class _T:
            def __enter__(self):
                self.t = time.perf_counter()
                log.info("stage %s ...", name)
            def __exit__(self, *exc):
                timings[name] = round(time.perf_counter() - self.t, 3)
        return _T()

    with stage("data"):
        fr = data_moex.load_or_build(spec["data"], a.workers)
        info, daily = fr["info"], fr["daily"]
    with stage("monthly"):
        monthly = monthly_frames(daily, info, spec)
    tables: dict[str, pd.DataFrame] = {}
    t3s, t3ts, t4s, metas = [], [], [], []
    with stage("main"):
        fm0 = first_measure_month(monthly, info, spec)
        res, panel, factors, meta = run_variant("main", monthly, daily, info, spec, spec["universe"]["min_mean_value_rub"], fm0)
        t3s.append(res["T3"]); t3ts.append(res["T3_terciles"]); t4s.append(res["T4"]); metas.append(meta)
        tables["T1_describe"] = describe(panel)
        corr_cols = ["A", "AT", "A_C", "AT_C", "absret"]
        tables["T1_corr"] = mean_cs_corr(panel, corr_cols, "month").reset_index(names="measure")
        tables["T1_corr_spearman"] = mean_cs_corr(panel, corr_cols, "month", "spearman").reset_index(names="measure")
        raw = fm(panel, "ret", spec["tests"]["nw_lags"])
        raw.insert(0, "variant", "main, raw returns")
        t4s.append(raw)
        # post-hoc diagnostic (not pre-registered): raw returns restricted to the months of the ret_adj FM,
        # to separate the effect of the shorter sample (betas need >= 24 months) from the beta adjustment
        adj_months = panel.loc[panel["beta"].notna(), "month"].unique()
        raw2 = fm(panel[panel["month"].isin(adj_months)], "ret", spec["tests"]["nw_lags"])
        raw2.insert(0, "variant", "diagnostic: raw returns, months of ret_adj FM")
        t4s.append(raw2)
        tables["factors"] = factors.reset_index(names="month")
        tables["T3_alpha_detail"] = alpha_detail(panel, factors, spec["tests"]["quantiles"], spec["tests"]["nw_lags"])
        tables["diag_by_year"] = diag_by_year(panel, spec["tests"]["quantiles"], spec["tests"]["nw_lags"])
        panel_out = panel
    with stage("variants"):
        r2, _, _, m2 = run_variant("filter 1M RUB", monthly, daily, info, spec, spec["universe"]["robust_min_mean_value_rub"], fm0)
        t3s.append(r2["T3"]); t3ts.append(r2["T3_terciles"]); t4s.append(r2["T4"]); metas.append(m2)
        # price returns without dividends: same daily data, dividends set to zero
        dn = data_moex.build_daily(fr["history"], fr["splits"], fr["dividends"].iloc[0:0], spec["data"])[0]
        mn = monthly_frames(dn, info, spec)
        r3, _, _, m3 = run_variant("no dividends (price returns)", mn, dn, info, spec, spec["universe"]["min_mean_value_rub"], fm0)
        t3s.append(r3["T3"]); t3ts.append(r3["T3_terciles"]); t4s.append(r3["T4"]); metas.append(m3)
        # main specification without the verified manual corrections (M9), to show their effect
        spec_nc = copy.deepcopy(spec)
        spec_nc["data"]["verified_splits"], spec_nc["data"]["verified_large_dividends"] = [], []
        dc = data_moex.build_daily(fr["history"], fr["splits"], fr["dividends"], spec_nc["data"])[0]
        mc = monthly_frames(dc, info, spec_nc)
        r4, _, _, m4 = run_variant("main without M9 corrections", mc, dc, info, spec_nc,
                                   spec["universe"]["min_mean_value_rub"], fm0)
        t3s.append(r4["T3"]); t3ts.append(r4["T3_terciles"]); t4s.append(r4["T4"]); metas.append(m4)
    with stage("report"):
        tables["T3_sorts"] = pd.concat(t3s, ignore_index=True)
        tables["T3_sorts_terciles"] = pd.concat(t3ts, ignore_index=True)
        tables["T4_fm"] = pd.concat(t4s, ignore_index=True)
        tables["sample"] = pd.DataFrame(metas)
        tables["variants_key"] = variants_key(tables["T3_sorts"], tables["T3_sorts_terciles"], tables["T4_fm"])
        paper = yaml.safe_load((SRC / "paper_numbers.yaml").read_text())
        mt3 = tables["T3_sorts"][tables["T3_sorts"]["variant"] == "main"]
        mt4 = tables["T4_fm"][tables["T4_fm"]["variant"] == "main"]
        tables["paper_vs_moex"] = paper_vs_moex(tables["T1_describe"], tables["T1_corr"], mt3, mt4, paper,
                                                spec["tests"]["t_crit"], spec["tests"]["hlz_threshold"])
        tables["universe_counts"] = universe_counts(fr["universe"], info, monthly, spec)
        tables["ex_date_check"] = ex_date_check(daily)
        pages = fr["dividend_pages"]
        common = info[info["type"] == "common_share"]
        tables["dividend_coverage"] = pd.DataFrame([{
            "secids": int(len(common)), "secids_with_page": int((pages["dohod"] | pages["smartlab"]).sum()),
            "dohod_pages": int(pages["dohod"].sum()), "smartlab_only": int((~pages["dohod"] & pages["smartlab"]).sum()),
            "events": int(len(fr["dividends"])), "events_with_ex_date": int(fr["dividends"]["ex_date"].notna().sum()),
            "secids_with_events": int(fr["dividends"]["secid"].nunique())}])
        cal = data_moex.market_calendar(fr["history"])
        ev, summ = data_moex.crosscheck_dividends(sorted(common["secid"]), cal, spec["data"])
        tables["dividend_crosscheck"] = summ
        tables["dividend_crosscheck_events"] = ev
        tables["data_flags"] = fr["flags"]
        tables["splits_used"] = fr["splits"]
        tables["deviations"] = pd.DataFrame(DEVIATIONS_MOEX, columns=["id", "paper", "moex", "reason", "effect"])
        for name, df in tables.items():
            csv_ready(df).to_csv(out / f"{name}.csv", index=False)
        Path(a.cache_dir).mkdir(parents=True, exist_ok=True)
        csv_ready(panel_out).to_parquet(Path(a.cache_dir) / "moex_panel.parquet", index=False)

    data_files = sorted(Path(a.cache_dir).glob("moex_*.parquet"))
    manifest = {
        "run_id": run_id, "created_utc": stamp, "upgrade": "moex",
        "base_run_id": base_run_id,
        "base_run_note": "MOEX data and tests are independent of the Binance base run; its id is recorded for provenance.",
        "git": git_info(),
        "config_path": str(cfg_path.resolve()), "config_sha256": sha256_file(cfg_path),
        "spec": spec, "spec_sha256": spec_hash,
        "code_sha256": {f: sha256_file(SRC / f) for f in CODE_FILES},
        "data_sha256": {p.name: sha256_file(p) for p in data_files if p.name != "moex_panel.parquet"},
        "raw_cache": str((ROOT / "data" / "raw" / "moex").resolve()),
        "seed": seed,
        "timings_sec": timings | {"total": round(sum(timings.values()), 3)},
        "rows": {"daily_rows": int(len(daily)), "daily_symbols": int(daily["symbol"].nunique()),
                 "panel_stock_months": int(len(panel_out)), "panel_months": int(panel_out["month"].nunique())},
        "tables": sorted(f"{k}.csv" for k in tables) + ["SUMMARY.md"],
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    (out / "SUMMARY.md").write_text(summary_md(out, base_run_id, run_id))
    log.info("moex upgrade %s written to %s", run_id, out)
    print(run_id)
    return run_id


if __name__ == "__main__":
    main()
