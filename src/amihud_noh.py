"""Upgrade 1: Amihud & Noh (2021) -- the term missing from the Lou & Shu (2017) decomposition.

    ILLIQ = mean(|r|/dvol) = LSIlliq + cov(|r|, 1/dvol),   LSIlliq = mean|r| * mean(1/dvol)  (A&N eq. 6)
    cov(|r|, 1/dvol) ~ -b * CV^2                                                         (A&N eq. 7)
    DIF = lnILLIQ - lnLSIlliq ~ ln(1 - b * CV^2 / LSIlliq)                                (A&N eq. 8)

Parts (all FM: monthly OLS of month-t return on signals measured in s = t-2, controls Ret[-12,-2]
and Ret[-1], Newey-West t with the config's lags; dependent variable ret_adj (CMKT-adjusted, our
main spec) and ret (raw; closest to A&N's excess return since rf = 0)):

(a) monthly measures from the base panel: share of coin-months with COV < 0, mean DIF,
    corr(DIF, lnA); FM [lnA_C, ln|Ret|] (LS eq. 6), [lnA_C, ln|Ret|, DIF] (A&N Table 1 col. 3),
    [lnLSIlliq, DIF] (col. 2), [ResLnA, lnA_C, IVOL] (col. 4 = LS Table 4 model 5).
(b) A&N-style 12-month measures from daily data (window of 12 calendar months ending in s,
    >= 200 valid days, the single highest day of ILLIQ, |R| and IDVOL each dropped, extreme 1% of
    ILLIQ, |R|, IDVOL, DIF censored in each s), lagged two months; the same FM plus fDIF.
(c) the IVOL channel: how much of Res.lnA is |Ret| / volatility.
(d) a verdict table: the decisive tests with t against 1.96 and the HLZ bar.

Usage:  python -I src/amihud_noh.py config.yaml [--base-run RUN_ID] [--out results/upgrades/amihud_noh]
                                              [--allow-data-mismatch]
The base run id is read from results/LATEST at run time unless --base-run is given.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json
import logging
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml

from fama_macbeth import fama_macbeth, nw_mean
from measures import cs_residual, daily_returns, mean_cs_corr, monthly_measures

ROOT = Path(__file__).resolve().parents[1]
OUT_DEFAULT = ROOT / "results" / "upgrades" / "amihud_noh"
CODE_FILES = ["amihud_noh.py", "measures.py", "fama_macbeth.py", "run.py", "data_audit.py"]
CONTROLS = ["mom", "ret_1"]
SENS_MIN_OBS = (None, 30, 50)   # minimum coins per FM cross-section; None = base rule k + 10
log = logging.getLogger("amihud_noh")

# Main choices of this upgrade. NOT pre-registered: this module was written together with its results; only the
# monthly DIF model (V1 monthly) was in src/analysis.py before the first real-data run.
AN_SPEC = {
    "window_months": 12,          # A&N: 12 months ending in s (Amihud 2002)
    "min_days": 200,              # A&N fn. 9: >= 200 trading days with return and volume
    "drop_max_day": True,         # A&N fn. 10: drop the day with the highest value of each variable
    "censor": [0.01, 0.99],       # A&N fn. 10: censor coins in the extreme 1% of ILLIQ, |R|, IDVOL, DIF
    "censor_cols": ["ILLIQ", "absR", "IDVOL", "DIF"],
    "dvol_scale": 1e6,            # dvol in millions of USDT, as A&N (logs make this irrelevant)
    "b_cv2_all_days": True,       # b and CV^2 on all valid days of the window (A&N do not drop a day)
    "dep_vars": ["ret_adj", "ret"],   # main = ret_adj (as our T4); ret = robustness (A&N: excess return)
}

# Numbers from Amihud & Noh (2021) that are not in src/paper_numbers.yaml (text and Table C.1).
AN_TEXT = {
    "source": "Amihud & Noh (2021): p. 4 (text), p. 6 (text), Table 1 (p. 15), Table C.1 (p. 20)",
    "mean_DIF": -0.17, "sd_DIF": 0.09, "corr_DIF_lnILLIQ": -0.37, "corr_DIF_lnLSIlliq": -0.41,
    "corr_DIF_ln_absR": -0.24, "corr_DIF_lnIDVOL": -0.38,
    "sd_ln_absR": 0.34, "sd_lnIDVOL": 1.89, "corr_lnILLIQ_lnIDVOL": 0.99,
    "share_b_negative": 0.011, "mean_CV2": 1.447, "mean_b": 0.043,
    "corr_lnb_lnLSIlliq": 0.87, "corr_lnCV2_lnLSIlliq": 0.62,
    "r2_DIF_on_components_int": 0.45, "r2_DIF_on_components_noint": 0.73,
    "r2_DIF_on_lnCV2_lnb_int": 0.40, "r2_DIF_on_lnCV2_lnb_noint": 0.67,
    "col1_lnILLIQ": (0.102, 2.45),
    "fDIF_int": (1.261, 3.29), "fDIF_noint": (1.022, 3.63),
    "fDIF_bcv_int": (2.236, 3.55), "fDIF_bcv_noint": (1.026, 3.96),
}


# ============================================================ (b) 12-month A&N measures
def _wide(s: pd.Series, months: pd.PeriodIndex) -> pd.DataFrame:
    return s.unstack("symbol").reindex(months).fillna(0.0)


def rolling_an_measures(daily: pd.DataFrame, window: int = 12, min_days: int = 200,
                        drop_max: bool = True, dvol_scale: float = 1e6) -> pd.DataFrame:
    """A&N 12-month measures indexed by the last month s of the window.

    Over the valid days d of months s-window+1..s (calendar months; gaps simply contribute no days):
      ILLIQ = mean_d |r|/dv, absR = mean_d |r|, IDVOL = mean_d 1/dv   (dv = quote volume / dvol_scale),
      each with its own single highest day removed when drop_max (mean = (sum - max) / (n - 1));
      LSIlliq = absR * IDVOL; DIF = lnILLIQ - lnLSIlliq; COV12 = ILLIQ - LSIlliq;
      b = cov(|r|, dv) / var(dv), CV2 = var(dv) / mean(dv)^2 (all valid days, population moments);
      DIF_approx = ln(1 - b CV2 / LSIlliq) (A&N eq. 8; NaN when the argument is <= 0).
    Returns symbol, month, n_days and the measures plus their logs, for windows with n_days >= min_days.
    """
    d = daily if "valid" in daily else daily_returns(daily)
    v = d.loc[d["valid"], ["symbol", "month", "r", "quote_volume"]].copy()
    v["ar"] = v["r"].abs()
    v["dv"] = v["quote_volume"] / dvol_scale
    v["inv"] = 1.0 / v["dv"]
    v["illiq"] = v["ar"] * v["inv"]
    v["dv2"] = v["dv"] ** 2
    v["ardv"] = v["ar"] * v["dv"]
    g = v.groupby(["symbol", "month"])
    sums = g[["ar", "inv", "illiq", "dv", "dv2", "ardv"]].sum()
    maxs = g[["ar", "inv", "illiq"]].max()
    cnt = g.size()
    months = pd.period_range(v["month"].min(), v["month"].max(), freq="M")

    roll_sum = lambda s: _wide(s, months).rolling(window, min_periods=1).sum()
    roll_max = lambda s: _wide(s, months).rolling(window, min_periods=1).max()   # values are >= 0
    n = roll_sum(cnt)
    k = 1.0 if drop_max else 0.0
    out = {"n_days": n}
    for col, name in [("illiq", "ILLIQ"), ("ar", "absR"), ("inv", "IDVOL")]:
        S = roll_sum(sums[col])
        M = roll_max(maxs[col]) if drop_max else 0.0
        out[name] = (S - k * M) / (n - k)
    S_ar, S_dv, S_dv2, S_ardv = (roll_sum(sums[c]) for c in ["ar", "dv", "dv2", "ardv"])
    m_dv, m_ar = S_dv / n, S_ar / n
    var_dv = S_dv2 / n - m_dv ** 2
    cov = S_ardv / n - m_ar * m_dv
    out["b"] = cov / var_dv.where(var_dv > 0)
    out["CV2"] = var_dv / m_dv ** 2

    long = pd.DataFrame({k_: w.stack(future_stack=True) for k_, w in out.items()})
    long.index.names = ["month", "symbol"]
    long = long.reset_index()
    long = long[long["n_days"] >= min_days].copy()
    long["n_days"] = long["n_days"].astype(int)
    long["LSIlliq"] = long["absR"] * long["IDVOL"]
    long["COV12"] = long["ILLIQ"] - long["LSIlliq"]
    with np.errstate(divide="ignore", invalid="ignore"):
        for c, lc in [("ILLIQ", "lnILLIQ"), ("LSIlliq", "lnLSIlliq"), ("absR", "ln_absR"),
                      ("IDVOL", "lnIDVOL"), ("b", "ln_b"), ("CV2", "lnCV2")]:
            long[lc] = np.log(long[c].where(long[c] > 0))
        long["DIF"] = long["lnILLIQ"] - long["lnLSIlliq"]
        arg = 1.0 - long["b"] * long["CV2"] / long["LSIlliq"]
        long["DIF_approx"] = np.log(arg.where(arg > 0))
    cols = ["symbol", "month", "n_days", "ILLIQ", "absR", "IDVOL", "LSIlliq", "COV12", "DIF", "DIF_approx",
            "b", "CV2", "lnILLIQ", "lnLSIlliq", "ln_absR", "lnIDVOL", "ln_b", "lnCV2"]
    return long[cols].sort_values(["symbol", "month"]).reset_index(drop=True)


def eligible_months(daily: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """(symbol, month) pairs that pass the base universe filters in that month (as analysis.build_panel)."""
    m = monthly_measures(daily, scale=float(cfg["measures"]["scale_A"]))
    u = cfg["universe"]
    ok = (m["n_days"] >= u["min_valid_days"]) & (m["dvol"] >= u["min_mean_dvol_usd"])
    return m.loc[ok, ["symbol", "month"]].reset_index(drop=True)


def censor_cs(df: pd.DataFrame, cols: list[str], by: str, q: tuple[float, float]) -> pd.DataFrame:
    """Drop rows whose value of any of `cols` lies outside the [q0, q1] quantiles of its cross-section.

    Quantiles are computed on the uncensored cross-section; rows with NaN in a column are dropped.
    """
    keep = pd.Series(True, index=df.index)
    for c in cols:
        lo = df.groupby(by)[c].transform(lambda s: s.quantile(q[0]))
        hi = df.groupby(by)[c].transform(lambda s: s.quantile(q[1]))
        keep &= df[c].notna() & (df[c] >= lo) & (df[c] <= hi)
    return df.loc[keep].copy()


def cs_fit(df: pd.DataFrame, y: str, xs: list[str], by: str, intercept: bool = True,
           min_obs: int = 10) -> tuple[pd.Series, pd.Series]:
    """Fitted values of monthly cross-sectional OLS of y on xs, and the R^2 of each month.

    With intercept the R^2 is centred; without it is uncentred (as A&N report 0.45 vs 0.73).
    """
    fit = pd.Series(np.nan, index=df.index, dtype=float)
    r2 = {}
    for t, g in df[[y] + xs].groupby(df[by], sort=True):
        g = g.dropna()
        if len(g) < min_obs:
            continue
        X = g[xs].to_numpy(float)
        if intercept:
            X = np.column_stack([np.ones(len(g)), X])
        yy = g[y].to_numpy(float)
        beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
        f = X @ beta
        fit.loc[g.index] = f
        tss = ((yy - yy.mean()) ** 2).sum() if intercept else (yy ** 2).sum()
        r2[t] = 1.0 - ((yy - f) ** 2).sum() / tss if tss > 0 else np.nan
    return fit, pd.Series(r2, dtype=float)


def build_an12(daily: pd.DataFrame, cfg: dict, eligible: pd.DataFrame | None = None,
               spec: dict = AN_SPEC) -> tuple[pd.DataFrame, pd.DataFrame]:
    """12-month measures at s for coins eligible in s, censored, with RlnILLIQ and fDIF variants.

    Returns (an12 at measurement month s, censoring log per month).
    """
    m = rolling_an_measures(daily, spec["window_months"], spec["min_days"], spec["drop_max_day"],
                            spec["dvol_scale"])
    if eligible is not None:
        m = m.merge(eligible, on=["symbol", "month"], how="inner")
    n_before = m.groupby("month").size()
    m = censor_cs(m, spec["censor_cols"], "month", tuple(spec["censor"]))
    n_after = m.groupby("month").size()
    log_ = pd.DataFrame({"n_before": n_before, "n_after": n_after}).fillna(0).astype(int)
    m = m.reset_index(drop=True)
    m["RlnILLIQ"] = cs_residual(m, "lnILLIQ", "lnIDVOL", "month")
    comps3, comps2 = ["lnCV2", "ln_b", "lnLSIlliq"], ["lnCV2", "ln_b"]
    for name, xs, icpt in [("fDIF_int", comps3, True), ("fDIF_noint", comps3, False),
                           ("fDIF_bcv_int", comps2, True), ("fDIF_bcv_noint", comps2, False)]:
        m[name], _ = cs_fit(m, "DIF", xs, "month", intercept=icpt)
    return m, log_.reset_index(names="month")


def lag_merge(an12: pd.DataFrame, panel: pd.DataFrame, lag: int) -> pd.DataFrame:
    """Attach measures of month s to the base-panel row of month t = s + lag (inner join)."""
    a = an12.copy()
    a["s"] = a["month"]
    a["month"] = a["month"] + lag
    keep = ["symbol", "month", "ret", "ret_adj", "mom", "ret_1", "ivol"]
    return panel[keep].merge(a, on=["symbol", "month"], how="inner")


# ============================================================ FM helpers
def fm_block(panel: pd.DataFrame, models: dict[str, list[str]], dep_vars: list[str], lags: int,
             sample: str, controls: list[str] = CONTROLS, min_obs: int | None = None) -> pd.DataFrame:
    """FM for each model x dependent variable. min_obs=None is the base rule (k + 10 coins per month);
    min_n is the smallest cross-section that entered the time-series average."""
    rows = []
    for y in dep_vars:
        for name, xs in models.items():
            cols = [y] + xs + controls
            n_cs = panel[cols].dropna().groupby(panel["month"]).size()
            n_cs = n_cs[n_cs >= (min_obs or len(xs + controls) + 10)]
            if len(n_cs) < 2:      # no usable cross-section (small synthetic data, high min_obs)
                rows += [{"sample": sample, "dep": y, "model": name, "var": x, "coef": np.nan, "t": np.nan,
                          "n_months": len(n_cs), "avg_obs": np.nan, "min_n": 0, "adj_r2": np.nan}
                         for x in xs + controls]
                continue
            res = fama_macbeth(panel, y, xs + controls, lags=lags, min_obs=min_obs)
            for _, r in res.iterrows():
                rows.append({"sample": sample, "dep": y, "model": name, "var": r["var"], "coef": r["coef"],
                             "t": r["t"], "n_months": res.attrs["n_months"], "avg_obs": res.attrs["avg_obs"],
                             "min_n": int(n_cs.min()) if len(n_cs) else 0, "adj_r2": res.attrs["avg_adj_r2"]})
    return pd.DataFrame(rows)


def ts_mean_t(df: pd.DataFrame, col: str, lags: int, by: str = "month") -> tuple[float, float]:
    """Time-series mean of the monthly cross-sectional mean of col and its NW t."""
    s = df.groupby(by)[col].mean().dropna()
    m, _, t = nw_mean(s, lags)
    return m, t


def avg_cs_r2(df: pd.DataFrame, y: str, xs: list[str], by: str = "month") -> float:
    _, r2 = cs_fit(df, y, xs, by, intercept=True)
    return float(r2.mean())


def pair_corr(df: pd.DataFrame, a: str, b: str, by: str = "month") -> float:
    return float(mean_cs_corr(df.dropna(subset=[a, b]), [a, b], by).iloc[0, 1])


# ============================================================ models
MONTHLY_MODELS = {
    "LS eq6: lnA_C+ln|Ret|": ["lnA_C", "ln_absret"],
    "AN col3: lnA_C+ln|Ret|+DIF": ["lnA_C", "ln_absret", "DIF"],
    "AN col2: lnLSIlliq+DIF": ["lnLSIlliq", "DIF"],
    "AN col4 = LS T4(5): ResLnA+lnA_C+ivol": ["ResLnA", "lnA_C", "ivol"],
}
MONTHLY_IVOL_MODELS = {
    "C1 = LS T4(4): lnA_C+ResLnA": ["lnA_C", "ResLnA"],
    "C2: lnA_C+ResLnA+ln|Ret|": ["lnA_C", "ResLnA", "ln_absret"],
    "C3: lnA_C+ResLnA+ivol": ["lnA_C", "ResLnA", "ivol"],
    "C4: lnA_C+ResLnA+ln|Ret|+ivol": ["lnA_C", "ResLnA", "ln_absret", "ivol"],
    "C5: lnA_C+ResLnA_perp": ["lnA_C", "ResLnA_perp"],
}
AN12_MODELS = {
    "AN col1: lnILLIQ": ["lnILLIQ"],
    "LS eq6: lnIDVOL+ln|R|": ["lnIDVOL", "ln_absR"],
    "AN col2: lnLSIlliq+DIF": ["lnLSIlliq", "DIF"],
    "AN col3: ln|R|+lnIDVOL+DIF": ["ln_absR", "lnIDVOL", "DIF"],
    "AN col4: RlnILLIQ+lnIDVOL+ivol": ["RlnILLIQ", "lnIDVOL", "ivol"],
    "AN fDIF (3 comps, intercept)": ["ln_absR", "lnIDVOL", "fDIF_int"],
    "AN fDIF (3 comps, no intercept)": ["ln_absR", "lnIDVOL", "fDIF_noint"],
    "AN fDIF (lnCV2, ln b, intercept)": ["ln_absR", "lnIDVOL", "fDIF_bcv_int"],
    "AN fDIF (lnCV2, ln b, no intercept)": ["ln_absR", "lnIDVOL", "fDIF_bcv_noint"],
}
AN12_IVOL_MODELS = {
    "C2: lnIDVOL+RlnILLIQ+ln|R|": ["lnIDVOL", "RlnILLIQ", "ln_absR"],
    "C4: lnIDVOL+RlnILLIQ+ln|R|+ivol": ["lnIDVOL", "RlnILLIQ", "ln_absR", "ivol"],
    "C5: lnIDVOL+RlnILLIQ_perp": ["lnIDVOL", "RlnILLIQ_perp"],
}

# Decisive tests: (id, question, sample, model, var, LS prediction, A&N prediction).
# Prediction codes: "+" significantly positive, "-" significantly negative, "0" not significant,
# "-/0" either (LS 2017, Introduction: the coefficient on the residual Amihud measure "is either
# insignificant or significantly negative"; corrected at verification, it was "-").
VERDICT_TESTS = [
    ("V1", "The omitted term DIF is priced beyond ln A_C and ln|Ret| (A&N col. 3)",
     "monthly", "AN col3: lnA_C+ln|Ret|+DIF", "DIF", "0", "+"),
    ("V1", "The omitted term DIF is priced beyond ln A_C and ln|Ret| (A&N col. 3)",
     "12m", "AN col3: ln|R|+lnIDVOL+DIF", "DIF", "0", "+"),
    ("V2", "DIF is priced beyond lnLSIlliq (A&N col. 2)",
     "monthly", "AN col2: lnLSIlliq+DIF", "DIF", "0", "+"),
    ("V2", "DIF is priced beyond lnLSIlliq (A&N col. 2)",
     "12m", "AN col2: lnLSIlliq+DIF", "DIF", "0", "+"),
    ("V3", "Sign of Res.lnA given ln A_C and IVOL (LS T4(5) = A&N col. 4)",
     "monthly", "AN col4 = LS T4(5): ResLnA+lnA_C+ivol", "ResLnA", "-/0", "+"),
    ("V3", "Sign of Res.lnA given ln A_C and IVOL (LS T4(5) = A&N col. 4)",
     "12m", "AN col4: RlnILLIQ+lnIDVOL+ivol", "RlnILLIQ", "-/0", "+"),
    ("V4", "fDIF (DIF fitted from lnCV2, ln b, lnLSIlliq) is priced (A&N p. 6)",
     "12m", "AN fDIF (3 comps, intercept)", "fDIF_int", "0", "+"),
    ("V5", "Res.lnA after controlling for ln|Ret| and IVOL (volatility channel)",
     "monthly", "C4: lnA_C+ResLnA+ln|Ret|+ivol", "ResLnA", "-/0", "+"),
    ("V5", "Res.lnA after controlling for ln|Ret| and IVOL (volatility channel)",
     "12m", "C4: lnIDVOL+RlnILLIQ+ln|R|+ivol", "RlnILLIQ", "-/0", "+"),
    ("V6", "Volume (ln A_C / lnIDVOL) is priced given ln|Ret| (both sides agree)",
     "monthly", "LS eq6: lnA_C+ln|Ret|", "lnA_C", "+", "+"),
    ("V6", "Volume (ln A_C / lnIDVOL) is priced given ln|Ret| (both sides agree)",
     "12m", "LS eq6: lnIDVOL+ln|R|", "lnIDVOL", "+", "+"),
]


def classify(t: float, bar: float) -> str:
    if not np.isfinite(t):
        return "n/a"
    return "+" if t >= bar else ("-" if t <= -bar else "0")


LS_NULL = "LS (H0 not rejected)"


def supports(outcome: str, ls: str, an: str) -> str:
    """Which camp the outcome supports. A significant sign is positive evidence; an insignificant
    coefficient where one camp predicts zero only fails to reject that camp (LS_NULL), it is not proof."""
    if outcome == "n/a":
        return "n/a"
    hit = [name for name, p in [("LS", ls), ("A&N", an)] if outcome in p.split("/")]
    if outcome == "0":
        return LS_NULL if hit == ["LS"] else ("neither" if not hit else "A&N (H0 not rejected)")
    return {0: "neither", 1: hit[0] if hit else "neither", 2: "both"}[len(hit)]


VERDICT_TEXT = {
    "AN": "the data lean towards A&N: the omitted term / Res.lnA carry positively priced information "
          "(at least one test clears the HLZ bar).",
    "AN_weak": "A&N lead only at the 1.96 level; no test clears the HLZ bar, so the conclusion is weak.",
    "LS": "the data lean towards LS: the significant discriminating results have the sign LS predict.",
    "none": "the data do not discriminate between the hypotheses: there are no significant discriminating results, "
            "or they offset each other; an insignificant DIF is consistent with LS, but in a short sample it may "
            "reflect low power rather than evidence.",
}


def verdict_counts(v: pd.DataFrame, hlz: float, dep: str = "ret_adj") -> pd.DataFrame:
    """Pre-registered rule over the main-spec discriminating tests (V1-V5; V0 is a premise, V6 not
    discriminating): 'AN' if A&N wins at 1.96 and at least one A&N result clears the HLZ bar;
    'AN_weak' if A&N wins at 1.96 only; 'LS' if significant LS-signed results outnumber A&N's; else 'none'."""
    hk = f"{hlz:g}"
    d = v[(v["dep"] == dep) & (v["test"] != "V0") & (v["LS_predicts"] != v["AN_predicts"])]
    row = {"dep": dep, "n_tests": len(d)}
    for bar in ["1.96", hk]:
        col = d[f"supports_{bar}"]
        row[f"AN_{bar}"] = int((col == "A&N").sum())
        row[f"LS_{bar}"] = int((col == "LS").sum())
        row[f"LS_null_{bar}"] = int((col == LS_NULL).sum())
        row[f"none_{bar}"] = int((col == "neither").sum())
    if row["AN_1.96"] > row["LS_1.96"]:
        code = "AN" if row[f"AN_{hk}"] >= 1 else "AN_weak"
    elif row["LS_1.96"] > row["AN_1.96"]:
        code = "LS"
    else:
        code = "none"
    row["verdict"] = code
    row["verdict_text"] = VERDICT_TEXT[code]
    return pd.DataFrame([row])


def verdict_table(fm: pd.DataFrame, premise: pd.DataFrame, bars: tuple[float, float]) -> pd.DataFrame:
    """One row per decisive test x dependent variable; outcome at |t| >= 1.96 and >= HLZ bar."""
    rows = []
    for _, r in premise.iterrows():   # V0: the premise cov(|r|, 1/dvol) < 0 <=> mean DIF < 0
        rows.append({"test": "V0", "question": "A&N premise: cov(|r|, 1/DVol) < 0, i.e. mean DIF < 0",
                     "sample": r["sample"], "dep": "-", "model": "mean of monthly CS mean DIF", "var": "DIF",
                     "coef": r["mean_DIF"], "t": r["t_mean_DIF"], "n_months": r["n_months"],
                     "LS_predicts": "0", "AN_predicts": "-"})
    for tid, q, sample, model, var, ls, an in VERDICT_TESTS:
        sel = fm[(fm["sample"] == sample) & (fm["model"] == model) & (fm["var"] == var)]
        for _, r in sel.iterrows():
            rows.append({"test": tid, "question": q, "sample": sample, "dep": r["dep"], "model": model,
                         "var": var, "coef": r["coef"], "t": r["t"], "n_months": r["n_months"],
                         "LS_predicts": ls, "AN_predicts": an})
    v = pd.DataFrame(rows)
    lo, hi = bars
    v["outcome_1.96"] = [classify(t, lo) for t in v["t"]]
    v[f"outcome_{hi:g}"] = [classify(t, hi) for t in v["t"]]
    v["supports_1.96"] = [supports(o, a, b) for o, a, b in zip(v["outcome_1.96"], v["LS_predicts"], v["AN_predicts"])]
    v[f"supports_{hi:g}"] = [supports(o, a, b) for o, a, b in
                             zip(v[f"outcome_{hi:g}"], v["LS_predicts"], v["AN_predicts"])]
    v["main_spec"] = v["dep"].isin(["ret_adj", "-"])
    return v


# ============================================================ the whole upgrade
def run_upgrade(daily: pd.DataFrame, panel: pd.DataFrame, cfg: dict, spec: dict = AN_SPEC,
                timings: dict | None = None) -> dict[str, pd.DataFrame]:
    """All tables of the upgrade. `panel` is the base panel (month as Period)."""
    timings = {} if timings is None else timings
    lags = int(cfg["tests"]["nw_lags"])
    bars = (1.96, float(cfg["tests"]["hlz_threshold"]))
    deps = spec["dep_vars"]
    lag = int(cfg["measures"]["lag_months"])
    out: dict[str, pd.DataFrame] = {}

    # ---- (a) monthly version on the base panel
    t0 = time.perf_counter()
    p = panel.copy()
    p["lnLSIlliq"] = p["ln_absret"] + p["lnA_C"]
    p["ResLnA_perp"] = cs_residual_multi(p, "ResLnA", ["ln_absret", "ivol"], "month")
    m_dif, t_dif = ts_mean_t(p, "DIF", lags)
    share_cov = p.dropna(subset=["COV"]).groupby("month")["COV"].apply(lambda s: (s < 0).mean())
    desc_m = {"sample": "monthly", "n_coin_months": int(p["DIF"].notna().sum()),
              "n_months": int(p.dropna(subset=["DIF"])["month"].nunique()),
              "share_COV_negative_pooled": float((p["COV"].dropna() < 0).mean()),
              "share_COV_negative_avg_month": float(share_cov.mean()),
              "mean_DIF": m_dif, "t_mean_DIF": t_dif, "median_DIF": float(p["DIF"].median()),
              "corr_DIF_lnA": pair_corr(p, "DIF", "lnA"), "corr_DIF_lnA_C": pair_corr(p, "DIF", "lnA_C"),
              "corr_DIF_ln_absret": pair_corr(p, "DIF", "ln_absret"),
              "corr_DIF_lnLSIlliq": pair_corr(p, "DIF", "lnLSIlliq"),
              # base signals are winsorized one by one (analysis.WINSOR_COLS), so the identity
              # lnA = ln|Ret| + lnA_C + DIF holds exactly only away from the tails
              "share_identity_broken": float(((p["lnA"] - p["ln_absret"] - p["lnA_C"] - p["DIF"]).abs()
                                              > 1e-8).mean())}
    fm_m = fm_block(p, MONTHLY_MODELS | MONTHLY_IVOL_MODELS, deps, lags, "monthly")
    timings["monthly"] = round(time.perf_counter() - t0, 3)

    # ---- (b) 12-month A&N measures
    t0 = time.perf_counter()
    elig = eligible_months(daily, cfg)
    an12, cens = build_an12(daily, cfg, elig, spec)
    q = lag_merge(an12, panel, lag)
    q["RlnILLIQ_perp"] = cs_residual_multi(q, "RlnILLIQ", ["ln_absR", "ivol"], "month")
    timings["an12_measures"] = round(time.perf_counter() - t0, 3)
    t0 = time.perf_counter()
    m12, t12 = ts_mean_t(an12, "DIF", lags)
    _, r2_int = cs_fit(an12, "DIF", ["lnCV2", "ln_b", "lnLSIlliq"], "month", True)
    _, r2_noint = cs_fit(an12, "DIF", ["lnCV2", "ln_b", "lnLSIlliq"], "month", False)
    _, r2b_int = cs_fit(an12, "DIF", ["lnCV2", "ln_b"], "month", True)
    _, r2b_noint = cs_fit(an12, "DIF", ["lnCV2", "ln_b"], "month", False)
    desc_12 = {"sample": "12m", "n_coin_months": int(an12["DIF"].notna().sum()),
               "n_months": int(an12["month"].nunique()),
               "n_coin_months_in_return_panel": int(q["DIF"].notna().sum()),
               "n_return_months": int(q["month"].nunique()),
               "share_COV_negative_pooled": float((an12["COV12"] < 0).mean()),
               "share_COV_negative_avg_month": float(an12.groupby("month")["COV12"].apply(lambda s: (s < 0).mean()).mean()),
               "mean_DIF": m12, "t_mean_DIF": t12, "median_DIF": float(an12["DIF"].median()),
               "sd_DIF_avg_month": float(an12.groupby("month")["DIF"].std().mean()),
               "corr_DIF_lnILLIQ": pair_corr(an12, "DIF", "lnILLIQ"),
               "corr_DIF_lnLSIlliq": pair_corr(an12, "DIF", "lnLSIlliq"),
               "corr_DIF_ln_absR": pair_corr(an12, "DIF", "ln_absR"),
               "corr_DIF_lnIDVOL": pair_corr(an12, "DIF", "lnIDVOL"),
               "corr_DIF_DIF_approx": pair_corr(an12, "DIF", "DIF_approx"),
               "share_DIF_approx_undefined": float(an12["DIF_approx"].isna().mean()),
               "share_b_negative": float((an12["b"] < 0).mean()),
               "share_b_nonpositive": float((an12["b"] <= 0).mean()),
               "mean_b_avg_month": float(an12.groupby("month")["b"].mean().mean()),
               "mean_CV2_avg_month": float(an12.groupby("month")["CV2"].mean().mean()),
               "corr_lnb_lnLSIlliq": pair_corr(an12, "ln_b", "lnLSIlliq"),
               "corr_lnCV2_lnLSIlliq": pair_corr(an12, "lnCV2", "lnLSIlliq"),
               "r2_DIF_on_components_int": float(r2_int.mean()),
               "r2_DIF_on_components_noint": float(r2_noint.mean()),
               "r2_DIF_on_lnCV2_lnb_int": float(r2b_int.mean()),
               "r2_DIF_on_lnCV2_lnb_noint": float(r2b_noint.mean()),
               "sd_fDIF_int_avg_month": float(an12.groupby("month")["fDIF_int"].std().mean()),
               "cs_sd_ret_adj_avg_month": float(q.groupby("month")["ret_adj"].std().mean()),
               "censored_share": float(1 - cens["n_after"].sum() / max(cens["n_before"].sum(), 1))}
    out["describe"] = pd.DataFrame([desc_m, desc_12])
    c1 = ["lnILLIQ", "lnLSIlliq", "ln_absR", "lnIDVOL", "DIF"]
    out["an12_corr"] = mean_cs_corr(an12.dropna(subset=c1), c1, "month").reset_index(names="measure")
    mom = an12.groupby("month")[c1].agg(["mean", "std"]).mean()
    out["an12_moments"] = mom.unstack().reset_index(names="measure")
    out["an12_censoring"] = cens
    fm_12 = fm_block(q, AN12_MODELS | AN12_IVOL_MODELS, deps, lags, "12m")
    timings["an12_fm"] = round(time.perf_counter() - t0, 3)

    # ---- (c) the IVOL channel
    t0 = time.perf_counter()
    ch = []
    for sample, df, res, ar in [("monthly", p, "ResLnA", "ln_absret"), ("12m", q, "RlnILLIQ", "ln_absR")]:
        df = df.dropna(subset=[res, ar, "ivol", "DIF"])     # one common sample for all R^2 (nested models)
        ch.append({"sample": sample, "residual": res,
                   f"corr_res_ln_absret": pair_corr(df, res, ar),
                   "corr_res_ivol": pair_corr(df, res, "ivol"),
                   "corr_res_DIF": pair_corr(df, res, "DIF"),
                   "corr_ln_absret_ivol": pair_corr(df, ar, "ivol"),
                   "r2_res_on_ln_absret": avg_cs_r2(df, res, [ar]),
                   "r2_res_on_ivol": avg_cs_r2(df, res, ["ivol"]),
                   "r2_res_on_ln_absret_ivol": avg_cs_r2(df, res, [ar, "ivol"]),
                   "r2_res_on_ln_absret_DIF": avg_cs_r2(df, res, [ar, "DIF"]),
                   "n_coin_months": int(df[[res, ar, "ivol"]].dropna().shape[0])})
    out["ivol_channel"] = pd.DataFrame(ch)
    p["DIF_perp"] = cs_residual_multi(p, "DIF", ["ln_absret", "lnA_C", "ivol"], "month")
    out["ivol_channel"]["corr_res_perp_DIF_perp"] = [pair_corr(p, "ResLnA_perp", "DIF_perp"), np.nan]
    timings["ivol_channel"] = round(time.perf_counter() - t0, 3)

    fm = pd.concat([fm_m, fm_12], ignore_index=True)
    out["fm"] = fm
    premise = out["describe"][["sample", "mean_DIF", "t_mean_DIF", "n_months"]]
    out["verdict"] = verdict_table(fm, premise, bars)
    out["verdict_counts"] = pd.concat([verdict_counts(out["verdict"], bars[1], dep) for dep in deps],
                                      ignore_index=True)
    # sensitivity (added at verification, not pre-registered): early FM months have 15-23 coins
    vmodels = {t[3] for t in VERDICT_TESTS}
    mm = {k: v_ for k, v_ in (MONTHLY_MODELS | MONTHLY_IVOL_MODELS).items() if k in vmodels}
    m12_ = {k: v_ for k, v_ in (AN12_MODELS | AN12_IVOL_MODELS).items() if k in vmodels}
    sv, sc = [], []
    for mo in SENS_MIN_OBS:
        fm_s = fm if mo is None else pd.concat([fm_block(p, mm, deps, lags, "monthly", min_obs=mo),
                                                fm_block(q, m12_, deps, lags, "12m", min_obs=mo)], ignore_index=True)
        vt = verdict_table(fm_s, premise, bars)
        vt = vt[vt["test"] != "V0"].merge(fm_s[["sample", "dep", "model", "var", "min_n"]],
                                          on=["sample", "dep", "model", "var"], how="left")
        label = "k+10 (main)" if mo is None else str(mo)
        sv.append(vt.assign(min_obs=label))
        sc.append(pd.concat([verdict_counts(vt, bars[1], dep) for dep in deps], ignore_index=True).assign(min_obs=label))
    out["sensitivity_min_obs"] = pd.concat(sv, ignore_index=True)[
        ["min_obs", "test", "sample", "dep", "var", "coef", "t", "n_months", "min_n", "supports_1.96",
         f"supports_{bars[1]:g}"]]
    out["sensitivity_min_obs_counts"] = pd.concat(sc, ignore_index=True)
    out["paper_reference"] = paper_reference()
    out["an12_measures_sample"] = an12.groupby("month").agg(
        n=("symbol", "size"), mean_DIF=("DIF", "mean"), mean_lnILLIQ=("lnILLIQ", "mean"),
        share_b_nonpos=("b", lambda s: float((s <= 0).mean()))).reset_index()
    return out


def cs_residual_multi(df: pd.DataFrame, y: str, xs: list[str], by: str, min_obs: int = 10) -> pd.Series:
    """Residual of y on [1, xs] within each cross-section (NaN where any input is missing)."""
    fit, _ = cs_fit(df, y, xs, by, intercept=True, min_obs=min_obs)
    return df[y] - fit



# key-regression rows (ret_adj) that have one paper value: (sample, our model, our var, paper model, paper var)
SCALE_PAIRS = [("monthly", "AN col3: lnA_C+ln|Ret|+DIF", "DIF", "AN col3", "DIF"),
               ("12m", "AN col3: ln|R|+lnIDVOL+DIF", "DIF", "AN col3", "DIF"),
               ("monthly", "AN col2: lnLSIlliq+DIF", "DIF", "AN col2", "DIF"),
               ("12m", "AN col2: lnLSIlliq+DIF", "DIF", "AN col2", "DIF"),
               ("monthly", "LS eq6: lnA_C+ln|Ret|", "lnA_C", "LS T2(3)", "lnA_C"),
               ("12m", "LS eq6: lnIDVOL+ln|R|", "lnIDVOL", "LS T2(3)", "lnA_C"),
               ("monthly", "LS eq6: lnA_C+ln|Ret|", "ln_absret", "LS T2(3)", "ln_absret"),
               ("12m", "LS eq6: lnIDVOL+ln|R|", "ln_absR", "LS T2(3)", "ln_absret"),
               ("12m", "AN fDIF (3 comps, intercept)", "fDIF_int", "AN fDIF_int", "fDIF_int")]


def scale_ratios(fm: pd.DataFrame, pr: pd.DataFrame) -> pd.DataFrame:
    """|our coefficient / paper coefficient| for SCALE_PAIRS (text of SUMMARY.md; build_docs.py uses the same pairs)."""
    pr = pr.reset_index() if "model" not in pr.columns else pr
    rows = []
    for sample, model, var, pm, pv in SCALE_PAIRS:
        o = fm[(fm["sample"] == sample) & (fm["dep"] == "ret_adj") & (fm["model"] == model) & (fm["var"] == var)]
        p = pr[(pr["model"] == pm) & (pr["var"] == pv)]
        if len(o) == 1 and len(p) == 1:
            a, b = float(o["coef"].iloc[0]), float(p["coef"].iloc[0])
            rows.append({"sample": sample, "model": model, "var": var, "ours": a, "paper": b,
                         "ratio": abs(a / b), "opposite_sign": a * b < 0})
    return pd.DataFrame(rows)


def paper_reference() -> pd.DataFrame:
    """Paper numbers to set next to ours: from src/paper_numbers.yaml plus AN_TEXT above."""
    pn = yaml.safe_load((Path(__file__).resolve().parent / "paper_numbers.yaml").read_text())
    ls, an = pn["lou_shu_2017"], pn["amihud_noh_2021"]
    rows = [
        ("LS T2(3)", "lnA_C", ls["table2_model3"]["lnA_C"]), ("LS T2(3)", "ln_absret", ls["table2_model3"]["ln_absret"]),
        ("LS T4(4)", "lnA_C", ls["table4_panelA"]["(4)"]["lnA_C"]), ("LS T4(4)", "ResLnA", ls["table4_panelA"]["(4)"]["ResLnA"]),
        ("LS T4(5)", "lnA_C", ls["table4_panelA"]["(5)"]["lnA_C"]), ("LS T4(5)", "ResLnA", ls["table4_panelA"]["(5)"]["ResLnA"]),
        ("LS T4(5)", "ivol", ls["table4_panelA"]["(5)"]["ivol"]),
        ("AN col2", "lnLSIlliq", an["col2"]["lnLSIlliq"]), ("AN col2", "DIF", an["col2"]["DIF"]),
        ("AN col3", "ln_absR", an["col3"]["ln_absret"]), ("AN col3", "lnIDVOL", an["col3"]["lnIDVOL"]),
        ("AN col3", "DIF", an["col3"]["DIF"]),
        ("AN col4", "lnIDVOL", an["col4"]["lnIDVOL"]), ("AN col4", "RlnILLIQ", an["col4"]["RlnILLIQ"]),
        ("AN col4", "ivol", an["col4"]["IdioVol"]),
    ]
    out = [{"model": m, "var": v, "coef": d["coef"], "t": d["t"], "source": "src/paper_numbers.yaml"} for m, v, d in rows]
    for k in ["col1_lnILLIQ", "fDIF_int", "fDIF_noint", "fDIF_bcv_int", "fDIF_bcv_noint"]:
        c, t = AN_TEXT[k]
        out.append({"model": f"AN {k}", "var": k, "coef": c, "t": t, "source": AN_TEXT["source"]})
    for k, val in AN_TEXT.items():
        if isinstance(val, float):
            out.append({"model": "AN descriptive", "var": k, "coef": val, "t": np.nan, "source": AN_TEXT["source"]})
    return pd.DataFrame(out)


# ============================================================ SUMMARY.md (generated)
def _f(x, nd=3):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def _get(fm, sample, dep, model, var):
    r = fm[(fm["sample"] == sample) & (fm["dep"] == dep) & (fm["model"] == model) & (fm["var"] == var)]
    return (np.nan, np.nan, 0) if r.empty else (float(r["coef"].iloc[0]), float(r["t"].iloc[0]), int(r["n_months"].iloc[0]))


def summary_md(tables: dict[str, pd.DataFrame], manifest: dict, hlz: float) -> str:
    d = tables["describe"].set_index("sample")
    fm, v, ch = tables["fm"], tables["verdict"], tables["ivol_channel"].set_index("sample")
    pr = tables["paper_reference"].set_index(["model", "var"])
    dm, d12 = d.loc["monthly"], d.loc["12m"]
    hk = f"{hlz:g}"
    L = [f"# Upgrade 1: Amihud & Noh (2021) vs Lou & Shu (2017)", "",
         f"Base run: `{manifest['base_run_id']}`; upgrade run: `{manifest['run_id']}`; config and code sha256 in "
         "`run_manifest.json` (`make verify` rechecks them). "
         f"Every number below is generated by code (`src/amihud_noh.py`) from the CSVs in this folder.", ""]
    if not manifest.get("data_matches_base", True):
        L += ["**WARNING: daily.parquet does not match the base run's data (hash). Do not interpret the numbers.**", ""]
    L += ["## What we tested",
          "- `ILLIQ = LSIlliq + cov(|r|, 1/DVol)`: Lou & Shu implicitly set the covariance to zero. "
          "`DIF = lnILLIQ − ln|R| − lnIDVOL` is the omitted term (A&N, eqs. 6–8).",
          "- (a) monthly measures of the base panel; (b) 12-month A&N measures (≥200 days, the single highest day "
          "excluded, 1% censoring), lagged 2 months; (c) the IVOL channel; (d) the verdict. FM, Newey–West with 6 lags, "
          "controls Ret[−12,−2] and Ret[−1]; the main dependent variable is the CMKT-adjusted return (`ret_adj`), "
          "the raw return (`ret`) is a robustness check.", "",
          "## Premise: is the covariance negative?",
          f"- Monthly measures: share of coin-months with COV<0 = {_f(dm['share_COV_negative_pooled'])}; "
          f"mean DIF = {_f(dm['mean_DIF'])} (t = {_f(dm['t_mean_DIF'], 2)}); corr(DIF, lnA) = {_f(dm['corr_DIF_lnA'])}.",
          f"- 12 months: share COV<0 = {_f(d12['share_COV_negative_pooled'])}; mean DIF = {_f(d12['mean_DIF'])} "
          f"(A&N: {_f(AN_TEXT['mean_DIF'], 2)}); corr(DIF, lnILLIQ) = {_f(d12['corr_DIF_lnILLIQ'])} "
          f"(A&N: {_f(AN_TEXT['corr_DIF_lnILLIQ'], 2)}); share b≤0 = {_f(d12['share_b_nonpositive'])} "
          f"(A&N: {_f(AN_TEXT['share_b_negative'])}); R² of DIF on (lnCV², ln b, lnLSIlliq) = "
          f"{_f(d12['r2_DIF_on_components_int'], 2)} (A&N: {_f(AN_TEXT['r2_DIF_on_components_int'], 2)}); "
          f"corr(DIF, ln(1−b·CV²/LSIlliq)) = {_f(d12['corr_DIF_DIF_approx'])}.", "",
          "## Key regressions (main specification, `ret_adj`)",
          "| Model | Variable | Us: monthly | Us: 12 months | Papers |", "|---|---|---|---|---|"]
    key = [("AN col3", "DIF", "AN col3: lnA_C+ln|Ret|+DIF", "AN col3: ln|R|+lnIDVOL+DIF", "DIF", ("AN col3", "DIF")),
           ("AN col2", "DIF", "AN col2: lnLSIlliq+DIF", "AN col2: lnLSIlliq+DIF", "DIF", ("AN col2", "DIF")),
           ("LS eq6", "ln A_C", "LS eq6: lnA_C+ln|Ret|", "LS eq6: lnIDVOL+ln|R|", ("lnA_C", "lnIDVOL"), ("LS T2(3)", "lnA_C")),
           ("LS eq6", "ln|Ret|", "LS eq6: lnA_C+ln|Ret|", "LS eq6: lnIDVOL+ln|R|", ("ln_absret", "ln_absR"), ("LS T2(3)", "ln_absret")),
           ("col4/T4(5)", "Res.lnA", "AN col4 = LS T4(5): ResLnA+lnA_C+ivol", "AN col4: RlnILLIQ+lnIDVOL+ivol",
            ("ResLnA", "RlnILLIQ"), None),
           ("C4 (+ln|Ret|+IVOL)", "Res.lnA", "C4: lnA_C+ResLnA+ln|Ret|+ivol", "C4: lnIDVOL+RlnILLIQ+ln|R|+ivol",
            ("ResLnA", "RlnILLIQ"), "none"),
           ("fDIF", "fDIF", None, "AN fDIF (3 comps, intercept)", "fDIF_int", ("AN fDIF_int", "fDIF_int"))]
    for name, label, mm, m12, var, ref in key:
        vm, v12 = (var if isinstance(var, str) else var[0]), (var if isinstance(var, str) else var[1])
        cm = _get(fm, "monthly", "ret_adj", mm, vm) if mm else (np.nan, np.nan, 0)
        c12 = _get(fm, "12m", "ret_adj", m12, v12)
        if ref == "none":
            paper = ("A&N col. 4 with ln|R| in place of IdioVol: 'similar' (fn. 14, no numbers), i.e. RlnILLIQ > 0; "
                     "neither paper uses both controls together; LS: not reported")
        elif ref is None:
            a, b = pr.loc[("LS T4(5)", "ResLnA")], pr.loc[("AN col4", "RlnILLIQ")]
            paper = f"LS {_f(a['coef'])} ({_f(a['t'], 2)}); A&N {_f(b['coef'])} ({_f(b['t'], 2)})"
        else:
            r = pr.loc[ref]
            paper = f"{_f(r['coef'])} ({_f(r['t'], 2)})"
        L.append(f"| {name} | {label} | {_f(cm[0])} ({_f(cm[1], 2)}) | {_f(c12[0])} ({_f(c12[1], 2)}) | {paper} |")
    L += ["", "Coefficients in % per month, NW t-statistics in parentheses. Papers: LS use FF3-adjusted returns; A&N use excess returns with Size and B/M controls.", ""]

    c_m, c_12 = ch.loc["monthly"], ch.loc["12m"]
    L += ["## The IVOL channel: what Res.lnA is",
          f"- corr(Res.lnA, ln|Ret|) = {_f(c_m['corr_res_ln_absret'])} (monthly), {_f(c_12['corr_res_ln_absret'])} (12 months); "
          f"corr(Res.lnA, IVOL) = {_f(c_m['corr_res_ivol'])} / {_f(c_12['corr_res_ivol'])}; "
          f"corr(Res.lnA, DIF) = {_f(c_m['corr_res_DIF'])} / {_f(c_12['corr_res_DIF'])}.",
          f"- Mean R² of Res.lnA on ln|Ret| = {_f(c_m['r2_res_on_ln_absret'], 2)} / {_f(c_12['r2_res_on_ln_absret'], 2)}; "
          f"on ln|Ret|+IVOL = {_f(c_m['r2_res_on_ln_absret_ivol'], 2)} / {_f(c_12['r2_res_on_ln_absret_ivol'], 2)}; "
          f"on ln|Ret|+DIF = {_f(c_m['r2_res_on_ln_absret_DIF'], 2)} / {_f(c_12['r2_res_on_ln_absret_DIF'], 2)} "
          "(almost an identity: lnA = ln|Ret| + lnA_C + DIF).", ""]
    share = c_m["r2_res_on_ln_absret_ivol"]
    if np.isfinite(share):
        L.append(f"- Conclusion on (c): ln|Ret| and IVOL explain {'most' if share >= 0.5 else 'a minority'} "
                 f"of the variation in Res.lnA for the monthly measures (R² = {_f(share, 2)}); the part of Res.lnA ⟂ (ln|Ret|, IVOL) is close to "
                 f"DIF ⟂ (ln|Ret|, ln A_C, IVOL): corr = {_f(c_m['corr_res_perp_DIF_perp'], 2)}.")
    c1 = _get(fm, "monthly", "ret_adj", "C1 = LS T4(4): lnA_C+ResLnA", "ResLnA")
    c3 = _get(fm, "monthly", "ret_adj", "C3: lnA_C+ResLnA+ivol", "ResLnA")
    c4 = _get(fm, "monthly", "ret_adj", "C4: lnA_C+ResLnA+ln|Ret|+ivol", "ResLnA")
    c5 = _get(fm, "monthly", "ret_adj", "C5: lnA_C+ResLnA_perp", "ResLnA_perp")
    L += [f"- FM (monthly, `ret_adj`), Res.lnA: no control {_f(c1[0])} (t {_f(c1[1], 2)}); +IVOL {_f(c3[0])} "
          f"(t {_f(c3[1], 2)}); +ln|Ret|+IVOL {_f(c4[0])} (t {_f(c4[1], 2)}); the part of Res.lnA orthogonal to "
          f"ln|Ret| and IVOL: {_f(c5[0])} (t {_f(c5[1], 2)}).", ""]

    # fDIF: every variant (V4 uses only the main fDIF_int) and the Taylor diagnostic of eq. (8)
    L += ["## fDIF and the Taylor approximation (A&N eqs. 7–8)",
          f"- corr(DIF, ln(1−b·CV²/LSIlliq)) = {_f(d12['corr_DIF_DIF_approx'])}; share of windows where the argument is ≤ 0 "
          f"(approximation undefined) = {_f(d12['share_DIF_approx_undefined'])}; mean CV² = "
          f"{_f(d12['mean_CV2_avg_month'], 2)} (A&N: {_f(AN_TEXT['mean_CV2'], 3)}); mean b = "
          f"{_f(d12['mean_b_avg_month'], 4)} (A&N: {_f(AN_TEXT['mean_b'], 3)}; dvol scales are not comparable); "
          f"corr(ln b, lnLSIlliq) = {_f(d12['corr_lnb_lnLSIlliq'], 2)} (A&N: {_f(AN_TEXT['corr_lnb_lnLSIlliq'], 2)}); "
          f"corr(lnCV², lnLSIlliq) = {_f(d12['corr_lnCV2_lnLSIlliq'], 2)} (A&N: {_f(AN_TEXT['corr_lnCV2_lnLSIlliq'], 2)}).",
          f"- R² of DIF on its components: (lnCV², ln b, lnLSIlliq) {_f(d12['r2_DIF_on_components_int'], 2)} with an intercept / "
          f"{_f(d12['r2_DIF_on_components_noint'], 2)} without (A&N: {_f(AN_TEXT['r2_DIF_on_components_int'], 2)} / "
          f"{_f(AN_TEXT['r2_DIF_on_components_noint'], 2)}); (lnCV², ln b) {_f(d12['r2_DIF_on_lnCV2_lnb_int'], 2)} / "
          f"{_f(d12['r2_DIF_on_lnCV2_lnb_noint'], 2)} (A&N: {_f(AN_TEXT['r2_DIF_on_lnCV2_lnb_int'], 2)} / {_f(AN_TEXT['r2_DIF_on_lnCV2_lnb_noint'], 2)}). R² without an intercept is uncentred and "
          "inflated by the non-zero mean of DIF.",
          "", "| fDIF variant | coef (t), `ret_adj` | coef (t), `ret` | A&N |", "|---|---|---|---|"]
    for model, var in [("AN fDIF (3 comps, intercept)", "fDIF_int"), ("AN fDIF (3 comps, no intercept)", "fDIF_noint"),
                       ("AN fDIF (lnCV2, ln b, intercept)", "fDIF_bcv_int"),
                       ("AN fDIF (lnCV2, ln b, no intercept)", "fDIF_bcv_noint")]:
        a, r_ = _get(fm, "12m", "ret_adj", model, var), _get(fm, "12m", "ret", model, var)
        ref = pr.loc[(f"AN {var}", var)]
        L.append(f"| {model}{' (V4, main variant, not pre-registered)' if var == 'fDIF_int' else ''} | {_f(a[0])} ({_f(a[1], 2)}) | "
                 f"{_f(r_[0])} ({_f(r_[1], 2)}) | {_f(ref['coef'])} ({_f(ref['t'], 2)}) |")
    rat = scale_ratios(fm, pr)
    L += ["", f"Our coefficients are on a different scale from the papers': in the key-regression rows with a single paper "
          f"value, |ours / paper| ranges from {_f(rat['ratio'].min(), 1)} to {_f(rat['ratio'].max(), 1)} "
          f"(median {_f(rat['ratio'].median(), 1)}; {int((rat['ratio'] < 10).sum())} of {len(rat)} below 10; "
          f"{int(rat['opposite_sign'].sum())} with the opposite sign), "
          f"which is consistent with coin returns being far more dispersed (a hypothesis): the cross-sectional s.d. of `ret_adj` is ≈ {_f(d12['cs_sd_ret_adj_avg_month'], 1)}% per month. For fDIF the regressor "
          f"also has a small spread: sd(fDIF) = {_f(d12['sd_fDIF_int_avg_month'], 3)} vs sd(DIF) = "
          f"{_f(d12['sd_DIF_avg_month'], 3)} (≈ √R²). We therefore compare only sign and t with A&N.", ""]

    main = v[v["main_spec"]]
    L += ["## Verdict: who is right on our data (main specification)",
          "| Test | Sample | Variable | coef | t | LS | A&N | |t|≥1.96 → | |t|≥" + hk + " → |",
          "|---|---|---|---|---|---|---|---|---|"]
    for _, r in main.iterrows():
        L.append(f"| {r['test']} | {r['sample']} | {r['var']} | {_f(r['coef'])} | {_f(r['t'], 2)} | {r['LS_predicts']} | "
                 f"{r['AN_predicts']} | {r['supports_1.96']} | {r['supports_' + hk]} |")
    vcs = tables["verdict_counts"].set_index("dep")
    vc = vcs.loc["ret_adj"]
    L += ["", f"Discriminating tests (V1–V5, where LS and A&N predict different outcomes): {vc['n_tests']}. "
          f"At |t|≥1.96: for A&N {vc['AN_1.96']}, for LS (significant LS sign) {vc['LS_1.96']}, "
          f"LS H0 not rejected {vc['LS_null_1.96']}, for neither {vc['none_1.96']}. "
          f"At |t|≥{hk}: for A&N {vc['AN_' + hk]}, for LS {vc['LS_' + hk]}.",
          f"**Outcome of our verdict rule (not pre-registered: the rule was written together with its results):** {vc['verdict_text']}", ""]
    if "ret" in vcs.index:
        vr = vcs.loc["ret"]
        L += [f"Robustness (raw return `ret`): for A&N {vr['AN_1.96']}/{vr['AN_' + hk]}, for LS "
              f"{vr['LS_1.96']}/{vr['LS_' + hk]} (at 1.96/{hk}); outcome: `{vr['verdict']}`.", ""]
        drv = v[(v["dep"] == "ret") & (v["supports_1.96"] == "A&N") & (v["test"] != "V0")]
        if len(drv):
            L += ["The A&N lead on `ret` comes from: " + "; ".join(
                f"{r['test']} {r['sample']} {r['var']} t = {_f(r['t'], 2)}" for _, r in drv.iterrows()) + ".", ""]
    col3 = _get(fm, "12m", "ret_adj", "AN col3: ln|R|+lnIDVOL+DIF", "DIF")
    c2_12 = _get(fm, "12m", "ret_adj", "C2: lnIDVOL+RlnILLIQ+ln|R|", "RlnILLIQ")
    m_col3 = _get(fm, "monthly", "ret_adj", "AN col3: lnA_C+ln|Ret|+DIF", "DIF")
    m_c2 = _get(fm, "monthly", "ret_adj", "C2: lnA_C+ResLnA+ln|Ret|", "ResLnA")
    fm_months = sorted({int(x) for x in fm["n_months"]})
    mom = tables["an12_moments"].set_index("measure")
    cr = tables["an12_corr"].set_index("measure")
    sv = tables["sensitivity_min_obs"]
    sc = tables["sensitivity_min_obs_counts"]
    labels = list(dict.fromkeys(sv["min_obs"]))
    L += ["## Sensitivity to cross-section size (added during verification, not pre-registered)",
          f"The first FM months contain few coins (smallest cross-section in the main specification: "
          f"{int(sv.loc[sv['min_obs'] == labels[0], 'min_n'].min())} coins). Below are the same tests when at least "
          "30 or 50 coins per month are required (all variants are shown; the main specification was not changed).", "",
          "| Test | Sample | Variable | " + " | ".join(f"t, min={x}" for x in labels) + " |",
          "|---|---|---|" + "---|" * len(labels)]
    base_rows = sv[(sv["min_obs"] == labels[0]) & (sv["dep"] == "ret_adj")]
    for _, r in base_rows.iterrows():
        ts = []
        for x in labels:
            m_ = sv[(sv["min_obs"] == x) & (sv["dep"] == "ret_adj") & (sv["test"] == r["test"]) &
                    (sv["sample"] == r["sample"]) & (sv["var"] == r["var"])]
            ts.append(f"{_f(m_['t'].iloc[0], 2)} ({int(m_['n_months'].iloc[0])} mo.)" if len(m_) else "n/a")
        L.append(f"| {r['test']} | {r['sample']} | {r['var']} | " + " | ".join(ts) + " |")
    L += ["", "Rule outcome under different minimums (`ret_adj` / `ret`): " + "; ".join(
          f"min={x}: `{sc[(sc['min_obs'] == x) & (sc['dep'] == 'ret_adj')]['verdict'].iloc[0]}` / "
          f"`{sc[(sc['min_obs'] == x) & (sc['dep'] == 'ret')]['verdict'].iloc[0]}`" for x in labels)
          + f". Discriminating results with |t| ≥ {hk} across all minimums and both dependent variables: "
          f"{int((sv['t'].abs() >= hlz).sum())}. "
          + ("The rule outcome changes with the minimum, so it cannot be considered robust."
             if sc["verdict"].nunique() > 1 else "The rule outcome does not depend on the minimum."), ""]
    L += [
          "## Caveats",
          "- The LS prediction for Res.lnA / RlnILLIQ (V3, V5) is coded as '−/0': LS (2017, Introduction) write that "
          "the coefficient on the residual measure is 'either insignificant or significantly negative'. It was "
          "corrected from '−' during verification; the rule does not depend on this (it compares only significant "
          "results); only the labels change from 'neither' to 'LS (H0 not rejected)'.",
          f"- The tests are not independent. Since lnILLIQ = ln|R| + lnIDVOL + DIF, the model [lnIDVOL, RlnILLIQ, ln|R|] is "
          f"a reparametrisation of col. 3: the 12-month RlnILLIQ coefficient = {_f(c2_12[0])} (t {_f(c2_12[1], 2)}), DIF in col. 3 = "
          f"{_f(col3[0])} (t {_f(col3[1], 2)}). V1–V5 use the same regressors in different combinations and are strongly dependent; "
          "the 'who wins' count is descriptive, not a formal test.",
          f"- In the monthly panel the signals are winsorised separately (base run), so the identity lnA = ln|Ret| + lnA_C + DIF "
          f"fails in {_f(100 * dm['share_identity_broken'], 1)}% of coin-months: monthly C2 Res.lnA = {_f(m_c2[0])} "
          f"(t {_f(m_c2[1], 2)}) vs DIF in col. 3 = {_f(m_col3[0])} (t {_f(m_col3[1], 2)}).",
          f"- 12 months: the cross-sectional spread of ln|R| is small (sd = {_f(mom.loc['ln_absR', 'std'], 2)} vs "
          f"{_f(mom.loc['lnIDVOL', 'std'], 2)} for lnIDVOL; A&N: {_f(AN_TEXT['sd_ln_absR'], 2)} and "
          f"{_f(AN_TEXT['sd_lnIDVOL'], 2)}), so corr(lnILLIQ, lnIDVOL) = {_f(cr.loc['lnILLIQ', 'lnIDVOL'], 3)} "
          f"(A&N: {_f(AN_TEXT['corr_lnILLIQ_lnIDVOL'], 2)}, so this is not specific to crypto). Beyond volume, ILLIQ carries "
          "only ln|R| and DIF.",
          "- IVOL in the 12-month models is the base IVOL (12 months, window ending at s = t−2, residuals on the daily EW CMKT, "
          "not FF3 as in A&N).",
          "- No Size or B/M control (D2, D3): A&N and LS have them; ln A_C partly plays the role of size.",
          f"- Months with measures: {int(dm['n_months'])}; FM months: {fm_months[0]}–{fm_months[-1]} vs 588 (LS) "
          f"and 744 (A&N): power is lower, so the HLZ bar t>{hk} is shown next to 1.96.",
          "- `b` and CV² use all days of the window (A&N, fn. 10, do not say whether the highest day is excluded for them); "
          "for ILLIQ, |R| and IDVOL the single highest day of each is excluded. The 1% censoring is on ILLIQ, |R|, IDVOL "
          "and DIF (there is no Size).",
          "- Specification search: every model of this upgrade (fm.csv, both dependent variables, 4 fDIF variants) "
          "is reported in full. Nothing shows that the specification or the verdict rule predates the results "
          "(they were written together); only the monthly DIF model (V1 monthly) was in the code "
          "before the first real-data run.",
          f"- Robustness on raw returns: rows with `dep = ret` in `verdict.csv` and `fm.csv`.", ""]
    return "\n".join(L)


# ============================================================ entry point
def prepare_daily(path: Path, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Load the daily file exactly as run.py does, including the data audit (asset ids SYM#k,
    redenomination adjustments), so that symbols and prices match the base panel."""
    import data_audit
    import run as base
    daily = base.load_daily(path)
    info = {"enabled": False}
    if data_audit.audit_config(cfg)["enabled"]:
        n_raw = len(daily)
        daily, events, _ = data_audit.audit(daily, cfg)
        info = {"enabled": True, "daily_rows_raw": int(n_raw), "daily_rows_clean": int(len(daily)),
                "events_by_type": {k: int(v) for k, v in events["type"].value_counts().items()}}
    return daily_returns(daily), info


def _load_panel(path: Path) -> pd.DataFrame:
    p = pd.read_parquet(path)
    p["month"] = pd.PeriodIndex(p["month"].astype(str), freq="M")
    p["symbol"] = p["symbol"].astype(str)
    return p


def main(argv: list[str] | None = None) -> Path:
    import run as base   # provenance helpers and the base loader (same daily cleaning as the base run)

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--base-run", default=None, help="base run id (default: contents of results/LATEST)")
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    ap.add_argument("--cache-dir", default=str(ROOT / "data" / "cache"))
    ap.add_argument("--daily", default=None, help="daily parquet (default: the base run's data_source)")
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--allow-data-mismatch", action="store_true",
                    help="proceed when the daily file hash differs from the base run's (smoke runs only)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    timings: dict[str, float] = {}
    t_all = time.perf_counter()

    cfg_path = Path(a.config)
    cfg = yaml.safe_load(cfg_path.read_text())
    res_dir = Path(a.results_dir)
    base_id = a.base_run or (res_dir / "LATEST").read_text().strip()
    base_manifest_path = res_dir / base_id / "run_manifest.json"
    bm = json.loads(base_manifest_path.read_text())
    panel_path = base.resolve_panel_cache(bm, base_id, Path(a.cache_dir))
    daily_path = Path(a.daily) if a.daily else base.resolve_repo_path(bm["data_source"])

    t0 = time.perf_counter()
    daily_sha = base.sha256_file(daily_path)
    match = daily_sha == bm.get("data_sha256")
    if not match:
        msg = f"daily data {daily_path} (sha {daily_sha[:12]}) differs from base run {base_id} ({str(bm.get('data_sha256'))[:12]})"
        if not a.allow_data_mismatch:
            raise SystemExit(msg + "; re-run the base pipeline (make run) or pass --allow-data-mismatch")
        log.warning(msg)
    daily, audit_info = prepare_daily(daily_path, cfg)
    panel = _load_panel(panel_path)
    timings["load"] = round(time.perf_counter() - t0, 3)

    tables = run_upgrade(daily, panel, cfg, AN_SPEC, timings)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    for name, df in tables.items():
        base._csv_ready(df).to_csv(out / f"{name}.csv", index=False)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    src = Path(__file__).resolve().parent
    manifest = {
        "run_id": f"{stamp}_amihud_noh",
        "upgrade": "amihud_noh",
        "base_run_id": base_id,
        "created_utc": stamp,
        "git": base.git_info(),
        "config_path": base.rel_path(cfg_path),
        "config_sha256": base.sha256_file(cfg_path),
        "code_sha256": {f: base.sha256_file(src / f) for f in CODE_FILES + ["paper_numbers.yaml"]},
        "inputs": {
            "daily": {"path": base.rel_path(daily_path), "sha256": daily_sha},
            "base_panel": {"path": base.rel_path(panel_path), "sha256": base.sha256_file(panel_path)},
            "base_manifest": {"path": base.rel_path(base_manifest_path), "sha256": base.sha256_file(base_manifest_path)},
        },
        "data_matches_base": bool(match),
        "audit": audit_info,
        "seed": int(cfg["seed"]),
        "seed_note": "no randomness in this upgrade",
        "spec": AN_SPEC,
        "nw_lags": int(cfg["tests"]["nw_lags"]),
        "t_bars": [1.96, float(cfg["tests"]["hlz_threshold"])],
        "versions": base.versions(),
        "rows": {"panel_coin_months": int(len(panel)),
                 "an12_coin_months_in_return_panel": int(tables["describe"].set_index("sample").loc["12m", "n_coin_months_in_return_panel"])},
        "tables": sorted(f"{k}.csv" for k in tables) + ["SUMMARY.md"],
    }
    (out / "SUMMARY.md").write_text(summary_md(tables, manifest, float(cfg["tests"]["hlz_threshold"])))
    timings["write"] = round(time.perf_counter() - t0, 3)
    timings["total"] = round(time.perf_counter() - t_all, 3)
    manifest["timings_sec"] = timings
    manifest["peak_rss_mb"] = round(base.peak_rss_mb(), 1)
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    log.info("amihud_noh written to %s (base run %s)", out, base_id)
    return out


if __name__ == "__main__":
    main()
