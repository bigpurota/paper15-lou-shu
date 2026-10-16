"""Build the month-t test panel (signals from t-2) and run the paper's tests.

Panel columns:
  ret, ret_adj (CMKT-adjusted) in percent per month; controls ret_1 = Ret[-1], mom = Ret[-12,-2]
  as fractions (as in Lou & Shu, so their FM slopes are comparable to the paper's 0.430 / -6.755),
  signals measured in month t-2: A, A_C, absret, COV, DIF, dvol, ivol, ...,
  logs lnA, lnA_C, ln_absret, lnVOL, residuals ResA (A on A_C), ResLnA (lnA on lnA_C).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from factors import cmkt_monthly, ivol_monthly, rolling_betas
from fama_macbeth import fama_macbeth, nw_mean
from measures import cs_residual, daily_returns, mean_cs_corr, monthly_measures, winsorize_cs
from sorts import alpha, hl_factor, sort_table

WINSOR_COLS = ["A", "A_C", "absret", "dvol", "AN", "AP", "COV", "DIF", "b", "CV2", "ivol"]


def build_panel(daily: pd.DataFrame, cfg: dict, start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (panel, factors) for return months start..end."""
    if "valid" not in daily:
        daily = daily_returns(daily)
    mcfg, ucfg, fcfg, ccfg = cfg["measures"], cfg["universe"], cfg["factors"], cfg["controls"]
    monthly = monthly_measures(daily, scale=float(mcfg["scale_A"]))
    eligible = (monthly["n_days"] >= ucfg["min_valid_days"]) & (monthly["dvol"] >= ucfg["min_mean_dvol_usd"])

    # signals (estimation month), winsorized and residualised within the estimation cross-section
    ivol = ivol_monthly(daily, ccfg["ivol_months"], ccfg["ivol_min_days"])
    sig = monthly.loc[eligible].drop(columns=["ret"]).merge(ivol, on=["symbol", "month"], how="left")
    sig = winsorize_cs(sig, WINSOR_COLS, "month", tuple(mcfg["winsor"]))
    with np.errstate(divide="ignore", invalid="ignore"):
        for c, lc in [("A", "lnA"), ("A_C", "lnA_C"), ("absret", "ln_absret"), ("dvol", "lnVOL")]:
            sig[lc] = np.log(sig[c]).replace(-np.inf, np.nan)
    sig["ResA"] = cs_residual(sig, "A", "A_C", "month")
    sig["ResLnA"] = cs_residual(sig, "lnA", "lnA_C", "month")
    sig["month"] = sig["month"] + int(mcfg["lag_months"])

    # returns, controls, risk adjustment
    cmkt = cmkt_monthly(monthly, eligible)
    betas = rolling_betas(monthly, cmkt, fcfg["beta_window"], fcfg["beta_min_obs"])
    R = monthly.pivot(index="month", columns="symbol", values="ret")
    R = R.reindex(pd.period_range(R.index.min(), R.index.max(), freq="M"))
    lo, hi = ccfg["mom_window"]
    mom = np.exp(np.log1p(R).shift(-hi).rolling(hi - lo + 1, min_periods=hi - lo + 1).sum()) - 1
    ctrl = pd.DataFrame({"ret_1": R.shift(1).stack(future_stack=True),
                         "mom": mom.stack(future_stack=True)}).reset_index()
    ctrl.columns = ["month", "symbol", "ret_1", "mom"]

    p = monthly[["symbol", "month", "ret"]].merge(sig, on=["symbol", "month"], how="inner")
    p = p.merge(ctrl, on=["symbol", "month"], how="left").merge(betas, on=["symbol", "month"], how="left")
    p["CMKT"] = p["month"].map(cmkt)
    p["ret_adj"] = p["ret"] - float(cfg["returns"]["rf"]) - p["beta"] * p["CMKT"]
    for c in ["ret", "ret_adj"]:          # dependent variables in %; controls stay fractions
        p[c] = 100.0 * p[c]
    p = p[(p["month"] >= pd.Period(start, "M")) & (p["month"] <= pd.Period(end, "M"))]
    p = p.dropna(subset=["ret"]).reset_index(drop=True)

    cmom = hl_factor(p, "mom", "ret", fcfg["factor_quantiles"]).rename("CMOM")
    factors = pd.concat([(100.0 * cmkt).rename("CMKT"), cmom], axis=1).loc[p["month"].unique()].sort_index()
    return p, factors


def _no_attrs(df: pd.DataFrame) -> pd.DataFrame:
    """Drop .attrs (sort_table stores a DataFrame there; pd.concat cannot compare those)."""
    df.attrs = {}
    return df


def _fm_block(panel, y, models, lags, controls):
    rows = []
    for name, xs in models.items():
        res = fama_macbeth(panel, y, xs + controls, lags=lags)
        for _, r in res.iterrows():
            rows.append({"model": name, "var": r["var"], "coef": r["coef"], "t": r["t"],
                         "n_months": res.attrs["n_months"], "avg_obs": res.attrs["avg_obs"],
                         "adj_r2": res.attrs["avg_adj_r2"]})
    return pd.DataFrame(rows)


def describe(panel: pd.DataFrame) -> pd.DataFrame:
    """Table 1, Panel A: pooled moments and percentiles of the measures (firm-months)."""
    cols = ["A", "A_C", "absret", "dvol", "COV", "DIF"]
    d = panel[cols].describe(percentiles=[.1, .25, .5, .75, .9]).T
    d["Q75/Q25"] = d["75%"] / d["25%"]
    return d


def winsorize_returns(panel: pd.DataFrame, q: tuple[float, float]) -> pd.DataFrame:
    """ret and ret_adj winsorized at quantiles q within each month (other columns untouched)."""
    return winsorize_cs(panel, ["ret", "ret_adj"], "month", tuple(q))


def winsorized_returns_block(panel, fac, cfg, t4_models, controls) -> pd.DataFrame:
    t = cfg["tests"]
    q = cfg["returns"].get("winsorize_returns_q", [0.01, 0.99])
    pw = winsorize_returns(panel, q)
    rows = []
    for s in ["A", "A_C", "ResA"]:
        tab = _no_attrs(sort_table(pw, s, "ret", t["quantiles"], fac, t["nw_lags"]).reset_index(names="row"))
        tab.insert(0, "signal", s)
        tab.insert(0, "test", "T3")
        rows.append(tab)
    fm = _fm_block(pw, "ret_adj", t4_models, t["nw_lags"], controls)
    fm.insert(0, "test", "T4")
    rows.append(fm)
    out = pd.concat(rows, ignore_index=True)
    out.insert(1, "winsor_q", f"{q[0]:g}-{q[1]:g}")
    return out


def run_tests(panel: pd.DataFrame, factors: pd.DataFrame, cfg: dict, full: bool = True) -> dict[str, pd.DataFrame]:
    t = cfg["tests"]
    lags, q = t["nw_lags"], t["quantiles"]
    controls = ["mom", "ret_1"]
    fac = factors[["CMKT", "CMOM"]]
    out: dict[str, pd.DataFrame] = {}

    # T3: quintile sorts on A, A_C, Res.A (raw EW and CMKT+CMOM alpha)
    t3 = []
    for s in ["A", "A_C", "ResA"]:
        tab = sort_table(panel, s, "ret", q, fac, lags)
        tab.insert(0, "signal", s)
        t3.append(_no_attrs(tab.reset_index(names="row")))
    out["T3_sorts"] = pd.concat(t3, ignore_index=True)

    # T4: FM on CMKT-adjusted returns
    t4_models = {"(1)": ["lnA"], "(2)": ["lnA_C"], "(3)": ["ResLnA"],
                 "(4)": ["lnA_C", "ResLnA"], "(5)": ["lnA_C", "ResLnA", "ivol"]}
    out["T4_fm"] = _fm_block(panel, "ret_adj", t4_models, lags, controls)
    if not full:
        return out

    # T1: descriptive stats and average cross-sectional correlations
    out["T1_describe"] = describe(panel).reset_index(names="measure")
    corr_cols = ["A", "A_C", "absret", "ResA", "COV"]
    out["T1_corr"] = mean_cs_corr(panel, corr_cols, "month").reset_index(names="measure")
    out["T1_corr_spearman"] = mean_cs_corr(panel, corr_cols, "month", "spearman").reset_index(names="measure")

    # T2: decomposition
    out["T2_fm"] = _fm_block(panel, "ret_adj", {"(1)": ["lnA"], "(3)": ["lnA_C", "ln_absret"]}, lags, controls)

    # T5: IMLA_C factor test (Table A.1)
    imla = hl_factor(panel, "A_C", "ret", t["imla_quantiles"]).rename("IMLA_C")
    hlA = sort_table(panel, "A", "ret", q, None, lags).attrs["portfolios"]
    hlA = (hlA[q] - hlA[1]).rename("HL_A")
    f1 = imla.to_frame()
    f3 = pd.concat([imla, fac], axis=1)
    t5 = []
    for name, f in [("raw", None), ("IMLA_C", f1), ("IMLA_C+CMKT+CMOM", f3)]:
        a, tv = alpha(hlA, f, lags)
        t5.append({"model": name, "alpha_HL_A": a, "t": tv})
    out["T5_imla"] = pd.DataFrame(t5)
    out["factors"] = pd.concat([fac, imla], axis=1).reset_index(names="month")

    # T6: trading volume directly
    out["T6_fm"] = _fm_block(panel, "ret_adj", {"(1)": ["lnVOL"], "(2)": ["lnVOL", "ResLnA"]}, lags, controls)

    # T7: January placebo for ln A_C
    fm = fama_macbeth(panel, "ret_adj", ["lnA_C"] + controls, lags=lags)
    cs = fm.attrs["coef_series"]["lnA_C"]
    jan = cs[[m.month == 1 for m in cs.index]]
    non = cs[[m.month != 1 for m in cs.index]]
    out["T7_january"] = pd.DataFrame([
        {"subset": "January", "coef": jan.mean(), "t": jan.mean() / (jan.std(ddof=1) / np.sqrt(len(jan))), "n": len(jan)},
        {"subset": "Feb-Dec", "coef": nw_mean(non, lags)[0], "t": nw_mean(non, lags)[2], "n": len(non)}])

    # Upgrade 1: Amihud & Noh (2021) missing covariance term
    an_models = {"LS: lnA_C+ln|Ret|": ["lnA_C", "ln_absret"],
                 "AN: lnA_C+ln|Ret|+DIF": ["lnA_C", "ln_absret", "DIF"],
                 "AN: ResLnA+lnA_C+ivol": ["lnA_C", "ResLnA", "ivol"]}
    out["U1_an_fm"] = _fm_block(panel, "ret_adj", an_models, lags, controls)
    u1 = []
    for s in ["COV", "DIF"]:
        tab = sort_table(panel, s, "ret", q, fac, lags)
        tab.insert(0, "signal", s)
        u1.append(_no_attrs(tab.reset_index(names="row")))
    out["U1_an_sorts"] = pd.concat(u1, ignore_index=True)
    out["U1_an_describe"] = pd.DataFrame([{
        "share_COV_negative": float((panel["COV"] < 0).mean()),
        "mean_DIF": float(panel["DIF"].mean()),
        "median_b": float(panel["b"].median()), "median_CV2": float(panel["CV2"].median()),
        "corr_DIF_lnA": float(mean_cs_corr(panel, ["DIF", "lnA"], "month").iloc[0, 1]),
        "corr_DIF_ResLnA": float(mean_cs_corr(panel, ["DIF", "ResLnA"], "month").iloc[0, 1])}])

    # Robustness: raw returns (Table A.3 Panel A), subperiods, liquid half, DVol-weighted sorts
    out["R_raw_fm"] = _fm_block(panel, "ret", t4_models, lags, controls)
    months = np.sort(panel["month"].unique())
    half = months[len(months) // 2]
    rows = []
    for name, sub in [("first half", panel[panel["month"] < half]), ("second half", panel[panel["month"] >= half])]:
        fmb = _fm_block(sub, "ret_adj", {"(2)": ["lnA_C"], "(4)": ["lnA_C", "ResLnA"]}, lags, controls)
        fmb.insert(0, "subsample", name)
        rows.append(fmb)
        for s in ["A", "A_C", "ResA"]:
            tab = sort_table(sub, s, "ret", q, None, lags).loc["raw"]
            rows.append(pd.DataFrame([{"subsample": name, "model": f"sort {s}", "var": "H-L",
                                       "coef": tab["H-L"], "t": tab["t"], "n_months": tab["n_months"]}]))
    out["R_subperiods"] = pd.concat(rows, ignore_index=True)

    liquid = panel[panel["dvol"] >= panel.groupby("month")["dvol"].transform("median")]
    vw_rows = []
    for name, sub, w in [("EW, all", panel, None), ("EW, liquid half", liquid, None), ("DVol-weighted, all", panel, "dvol")]:
        for s in ["A", "A_C", "ResA"]:
            tab = sort_table(sub, s, "ret", q, fac, lags, weight=w)
            for row in ["raw", "alpha"]:
                vw_rows.append({"sample": name, "signal": s, "row": row, "H-L": tab.loc[row, "H-L"], "t": tab.loc[row, "t"]})
    out["R_weighting"] = pd.DataFrame(vw_rows)

    # Robustness (pre-registered, returns.winsorize_returns): T3 sorts and T4 FM with the
    # dependent returns winsorized within each month; factors (CMKT, CMOM) are left unchanged
    out["R_winsor_returns"] = winsorized_returns_block(panel, fac, cfg, t4_models, controls)

    out["sample"] = pd.DataFrame([{
        "n_months": panel["month"].nunique(), "first": str(months[0]), "last": str(months[-1]),
        "n_coin_months": len(panel), "n_coins": panel["symbol"].nunique(),
        "avg_coins_per_month": panel.groupby("month").size().mean(),
        "avg_coins_fm": panel.dropna(subset=["ret_adj", "lnA", "mom", "ret_1"]).groupby("month").size().mean()}])
    return out
