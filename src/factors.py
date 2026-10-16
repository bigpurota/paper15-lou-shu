"""Crypto factor analogues and risk-adjusted returns.

CMKT  : EW monthly return of the eligible universe (eligible in month t-1).
betas : rolling OLS of monthly returns on CMKT over [t-W, t-1], at least `min_obs` months
        (analogue of the FF3 adjustment of Brennan, Chordia & Subrahmanyam 1998, eq. 8 of the paper).
IVOL  : std of residuals of daily returns on daily EW CMKT over the 12 months ending in the
        estimation month (computed from monthly sufficient statistics).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def cmkt_monthly(monthly: pd.DataFrame, eligible: pd.Series) -> pd.Series:
    """EW return in month t over symbols eligible in month t-1 (no look-ahead in membership)."""
    m = monthly[["symbol", "month", "ret"]].copy()
    elig = monthly.loc[eligible, ["symbol", "month"]].copy()
    elig["month"] = elig["month"] + 1
    m = m.merge(elig, on=["symbol", "month"], how="inner").dropna(subset=["ret"])
    return m.groupby("month")["ret"].mean().rename("CMKT")


def rolling_betas(monthly: pd.DataFrame, factor: pd.Series, window: int, min_obs: int) -> pd.DataFrame:
    """beta_{i,t} from months [t-window, t-1] (pairwise-complete), long format symbol, month, beta."""
    R = monthly.pivot(index="month", columns="symbol", values="ret")
    full = pd.period_range(R.index.min(), R.index.max(), freq="M")
    R = R.reindex(full)
    f = factor.reindex(full)
    M = R.notna() & f.notna().to_numpy()[:, None]
    x = pd.DataFrame(np.where(M, f.to_numpy()[:, None], 0.0), index=full, columns=R.columns)
    y = R.where(M, 0.0)
    roll = lambda z: z.rolling(window, min_periods=1).sum().shift(1)
    n, sx, sy = roll(M.astype(float)), roll(x), roll(y)
    sxx, sxy = roll(x * x), roll(x * y)
    var = sxx - sx ** 2 / n
    beta = (sxy - sx * sy / n) / var.where(var > 1e-12)
    beta = beta.where(n >= min_obs)
    out = beta.stack().rename("beta").reset_index()
    out.columns = ["month", "symbol", "beta"]
    return out


def daily_cmkt(daily: pd.DataFrame) -> pd.Series:
    """EW daily return across all valid symbol-days."""
    v = daily[daily["valid"]]
    return v.groupby("date")["r"].mean().rename("m")


def ivol_monthly(daily: pd.DataFrame, n_months: int, min_days: int) -> pd.DataFrame:
    """IVOL indexed by the last month of its window (symbol, month, ivol)."""
    m = daily_cmkt(daily)
    v = daily.loc[daily["valid"], ["symbol", "month", "date", "r"]].copy()
    v["x"] = v["date"].map(m)
    v = v.dropna(subset=["x"])
    v["xx"], v["yy"], v["xy"] = v["x"] ** 2, v["r"] ** 2, v["x"] * v["r"]
    s = v.groupby(["symbol", "month"])[["r", "x", "xx", "yy", "xy"]].sum()
    s["n"] = v.groupby(["symbol", "month"]).size()
    out = []
    for sym, g in s.groupby(level="symbol"):
        g = g.droplevel("symbol")
        g = g.reindex(pd.period_range(g.index.min(), g.index.max(), freq="M"), fill_value=0.0)
        r = g.rolling(n_months, min_periods=1).sum()
        n = r["n"]
        sxx = r["xx"] - r["x"] ** 2 / n
        syy = r["yy"] - r["r"] ** 2 / n
        sxy = r["xy"] - r["x"] * r["r"] / n
        ssr = syy - sxy ** 2 / sxx.where(sxx > 0)
        ivol = np.sqrt((ssr / (n - 2)).clip(lower=0)).where(n >= min_days)
        out.append(pd.DataFrame({"symbol": sym, "month": ivol.index, "ivol": ivol.to_numpy()}))
    return pd.concat(out, ignore_index=True)
