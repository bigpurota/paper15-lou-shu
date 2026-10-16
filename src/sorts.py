"""Portfolio sorts: quantile portfolios, H-L spreads, factor alphas (Newey-West t)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from fama_macbeth import nw_mean


def assign_quantiles(panel: pd.DataFrame, signal: str, q: int, by: str = "month") -> pd.Series:
    """Quantile 1..q of `signal` within each cross-section (ties broken by rank order)."""
    def _q(s: pd.Series) -> pd.Series:
        ok = s.notna()
        out = pd.Series(np.nan, index=s.index)
        if ok.sum() >= q:
            out[ok] = pd.qcut(s[ok].rank(method="first"), q, labels=False) + 1
        return out
    return panel.groupby(by, group_keys=False)[signal].apply(_q)


def portfolio_returns(panel: pd.DataFrame, signal: str, ret: str, q: int,
                      by: str = "month", weight: str | None = None) -> pd.DataFrame:
    """Months x quantiles matrix of portfolio returns (EW, or weighted by column `weight`)."""
    p = panel[[by, signal, ret] + ([weight] if weight else [])].dropna().copy()
    p["q"] = assign_quantiles(p, signal, q, by)
    p = p.dropna(subset=["q"])
    if weight:
        p["wr"] = p[ret] * p[weight]
        g = p.groupby([by, "q"])
        pr = g["wr"].sum() / g[weight].sum()
    else:
        pr = p.groupby([by, "q"])[ret].mean()
    pr = pr.unstack("q")
    pr.columns = [int(c) for c in pr.columns]
    return pr


def alpha(series: pd.Series, factors: pd.DataFrame | None, lags: int = 6) -> tuple[float, float]:
    """Intercept and its NW t-stat from a time-series regression on factors (raw mean if None)."""
    if factors is None or factors.shape[1] == 0:
        m, _, t = nw_mean(series, lags)
        return m, t
    df = pd.concat([series.rename("y"), factors], axis=1, join="inner").dropna()
    X = sm.add_constant(df[factors.columns])
    res = sm.OLS(df["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(res.params["const"]), float(res.tvalues["const"])


def sort_table(panel: pd.DataFrame, signal: str, ret: str, q: int, factors: pd.DataFrame | None,
               lags: int = 6, by: str = "month", weight: str | None = None) -> pd.DataFrame:
    """Rows 'raw' and 'alpha': mean return / alpha per quantile plus H-L and its NW t."""
    pr = portfolio_returns(panel, signal, ret, q, by, weight)
    hl = pr[q] - pr[1]
    out = {}
    for name, f in [("raw", None), ("alpha", factors)]:
        row = {f"Q{k}": alpha(pr[k], f, lags)[0] for k in pr.columns}
        a, t = alpha(hl, f, lags)
        row.update({"H-L": a, "t": t, "n_months": int(hl.notna().sum())})
        out[name] = row
    tab = pd.DataFrame(out).T
    tab.attrs["portfolios"] = pr
    return tab


def hl_factor(panel: pd.DataFrame, signal: str, ret: str, q: int, by: str = "month") -> pd.Series:
    """EW top-quantile minus bottom-quantile return (e.g. IMLA_C, CMOM)."""
    pr = portfolio_returns(panel, signal, ret, q, by)
    return (pr[q] - pr[1]).rename(f"HL_{signal}")
