"""Fama-MacBeth (1973) regressions with Newey-West (1987) t-statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def nw_mean(x: pd.Series | np.ndarray, lags: int = 6) -> tuple[float, float, float]:
    """Mean of a time series, its Newey-West standard error and t-statistic.

    Implemented as OLS of x on a constant with HAC covariance (Bartlett kernel).
    """
    x = pd.Series(x).dropna().to_numpy(dtype=float)
    res = sm.OLS(x, np.ones((len(x), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(res.params[0]), float(res.bse[0]), float(res.tvalues[0])


def cs_ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, float]:
    """OLS with intercept; returns (coefficients without intercept, adjusted R^2)."""
    n, k = X.shape
    Z = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    resid = y - Z @ beta
    tss = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (resid ** 2).sum() / tss if tss > 0 else np.nan
    adj = 1 - (1 - r2) * (n - 1) / (n - k - 1) if n > k + 1 else np.nan
    return beta[1:], adj


def fama_macbeth(panel: pd.DataFrame, y: str, xs: list[str], by: str = "month",
                 lags: int = 6, min_obs: int | None = None) -> pd.DataFrame:
    """Monthly cross-sectional OLS of y on xs; returns a table with mean coef, NW t, N, adj R^2.

    Coefficients on returns are reported in percent per month when y is in percent.
    """
    min_obs = min_obs or (len(xs) + 10)
    rows, stats = [], []
    for t, g in panel.groupby(by, sort=True):
        g = g[[y] + xs].dropna()
        if len(g) < min_obs:
            continue
        b, adj = cs_ols(g[y].to_numpy(float), g[xs].to_numpy(float))
        rows.append(pd.Series(b, index=xs, name=t))
        stats.append((len(g), adj))
    coefs = pd.DataFrame(rows)
    out = []
    for x in xs:
        m, se, tval = nw_mean(coefs[x], lags)
        out.append({"var": x, "coef": m, "se": se, "t": tval})
    res = pd.DataFrame(out)
    res.attrs["n_months"] = len(coefs)
    res.attrs["avg_obs"] = float(np.mean([s[0] for s in stats])) if stats else np.nan
    res.attrs["avg_adj_r2"] = float(np.nanmean([s[1] for s in stats])) if stats else np.nan
    res.attrs["coef_series"] = coefs
    return res
