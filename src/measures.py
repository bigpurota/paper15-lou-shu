"""Monthly Amihud-type measures from a daily panel (Lou & Shu 2017, Appendix A).

Daily input columns: symbol, date, close, quote_volume (dollar volume).
All monthly measures use the same set of valid days d: return defined
(previous calendar day has a close) and dollar volume > 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def daily_returns(daily: pd.DataFrame) -> pd.DataFrame:
    """Add r (simple close-to-close return) and valid flag; r is NaN after a gap."""
    d = daily.sort_values(["symbol", "date"]).copy()
    prev_close = d.groupby("symbol")["close"].shift(1)
    prev_date = d.groupby("symbol")["date"].shift(1)
    consecutive = (d["date"] - prev_date) == pd.Timedelta(days=1)
    d["r"] = np.where(consecutive, d["close"] / prev_close - 1.0, np.nan)
    d["valid"] = d["r"].notna() & (d["quote_volume"] > 0) & np.isfinite(d["r"])
    d["month"] = d["date"].dt.to_period("M")
    return d


def monthly_measures(daily: pd.DataFrame, scale: float = 1e6) -> pd.DataFrame:
    """One row per (symbol, month): A, A_C, absret, COV, DIF, VOLUME, AN, AP, b, CV2, ret.

    A      = mean_d |r_d| / DVol_d                 (x scale)
    A_C    = mean_d 1 / DVol_d                     (x scale)
    absret = mean_d |r_d|
    COV    = A - absret * A_C  = cov_d(|r|, 1/DVol) with 1/D normalisation (Amihud & Noh 2021, eq. 2)
    DIF    = ln A - ln absret - ln A_C              (Amihud & Noh 2021, eq. 8)
    b, CV2 = slope of |r| on DVol (with constant) and squared coef. of variation of DVol (A&N eq. 7)
    ret    = month-end close / previous month-end close - 1 (consecutive months only)
    """
    d = daily if "valid" in daily else daily_returns(daily)
    v = d[d["valid"]].copy()
    v["ar"] = v["r"].abs()
    v["inv"] = 1.0 / v["quote_volume"]
    v["ratio"] = v["ar"] * v["inv"]
    v["ratio_n"] = (-v["r"].clip(upper=0)) * v["inv"]
    v["ratio_p"] = v["r"].clip(lower=0) * v["inv"]
    v["dv2"] = v["quote_volume"] ** 2
    v["ar_dv"] = v["ar"] * v["quote_volume"]
    g = v.groupby(["symbol", "month"], sort=False)
    m = g.agg(n_days=("r", "size"), A=("ratio", "mean"), A_C=("inv", "mean"),
              absret=("ar", "mean"), AN=("ratio_n", "mean"), AP=("ratio_p", "mean"),
              dvol=("quote_volume", "mean"), dv2=("dv2", "mean"), ar_dv=("ar_dv", "mean"))
    for c in ["A", "A_C", "AN", "AP"]:
        m[c] *= scale
    m["COV"] = m["A"] - m["absret"] * m["A_C"]
    var_dv = m["dv2"] - m["dvol"] ** 2
    m["b"] = (m["ar_dv"] - m["absret"] * m["dvol"]) / var_dv.where(var_dv > 0)
    m["CV2"] = var_dv / m["dvol"] ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        m["DIF"] = np.log(m["A"]) - np.log(m["absret"]) - np.log(m["A_C"])
    m = m.drop(columns=["dv2", "ar_dv"])

    last = d.groupby(["symbol", "month"], sort=True)["close"].last().rename("close_end").reset_index()
    last["prev_close"] = last.groupby("symbol")["close_end"].shift(1)
    prev_month = last.groupby("symbol")["month"].shift(1)
    consecutive = prev_month == (last["month"] - 1)
    last["ret"] = np.where(consecutive, last["close_end"] / last["prev_close"] - 1.0, np.nan)
    out = last[["symbol", "month", "ret"]].merge(m.reset_index(), on=["symbol", "month"], how="left")
    out["n_days"] = out["n_days"].fillna(0).astype(int)
    return out.sort_values(["symbol", "month"]).reset_index(drop=True)


def winsorize_cs(df: pd.DataFrame, cols: list[str], by: str, q: tuple[float, float]) -> pd.DataFrame:
    """Winsorize each column at quantiles q within each cross-section `by`."""
    out = df.copy()
    for c in cols:
        lo = out.groupby(by)[c].transform(lambda s: s.quantile(q[0]))
        hi = out.groupby(by)[c].transform(lambda s: s.quantile(q[1]))
        out[c] = out[c].clip(lower=lo, upper=hi)
    return out


def cs_residual(df: pd.DataFrame, y: str, x: str, by: str) -> pd.Series:
    """Residual of y on [1, x] estimated separately in each cross-section `by`."""
    res = pd.Series(np.nan, index=df.index, dtype=float)
    for _, g in df[[y, x]].groupby(df[by], sort=False):
        g = g.dropna()
        if len(g) < 3:
            continue
        X = np.column_stack([np.ones(len(g)), g[x].to_numpy(float)])
        beta, *_ = np.linalg.lstsq(X, g[y].to_numpy(float), rcond=None)
        res.loc[g.index] = g[y].to_numpy(float) - X @ beta
    return res


def mean_cs_corr(df: pd.DataFrame, cols: list[str], by: str, method: str = "pearson") -> pd.DataFrame:
    """Time-series average of monthly cross-sectional correlation matrices (Table 1, Panel B)."""
    mats = [g[cols].corr(method=method) for _, g in df.groupby(by) if len(g) >= 10]
    return sum(mats) / len(mats)
