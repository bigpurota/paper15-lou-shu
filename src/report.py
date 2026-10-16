""""Paper vs us" table (CSV + Markdown), robustness and power diagnostics, LaTeX number macros.

Usage:  python -I src/report.py [--results-dir results] [--run-id RUN_ID (default: results/LATEST)]
Reads:  results/<run_id>/*.csv (base run), results/simulations/*.csv (known-answer simulations),
        the run's cached panel (run_manifest.json -> panel_cache) for a few panel diagnostics,
        src/paper_numbers.yaml (numbers copied from the papers).
Writes: results/<run_id>/paper_vs_us.csv, paper_vs_us.md, robustness_vs_base.csv,
        report_diagnostics.csv, report_manifest.json, tex/numbers.tex

Every number in paper_vs_us.md comes from these inputs: the "why" texts are templates whose
numbers are filled in from the CSVs. A missing input gives "n/a", never a typed-in number.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import hashlib
import json
import math
import time

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
PAPER = Path(__file__).resolve().parent / "paper_numbers.yaml"
T_CRIT = 1.96
Z_POWER80 = 0.8416212335729143          # Phi^{-1}(0.80): MDE = (1.96 + 0.84) * SE
CONTROLS = ["mom", "ret_1"]

# (id, section, label, kind); kind 't' = rule on sign and t-stat, 'level' = sign and order of magnitude,
# 'ratio' = order of magnitude only (positive by construction, so the sign carries no information).
# Ids are letters only (they become LaTeX macro names).
METRICS = [
    ("corrAAC", "T1", "corr(A, A_C)", "level"),
    ("corrAabs", "T1", "corr(A, |Ret|)", "level"),
    ("corrACabs", "T1", "corr(A_C, |Ret|)", "level"),
    ("QratioAC", "T1", "Q75/Q25 of A_C", "ratio"),
    ("QratioAbs", "T1", "Q75/Q25 of |Ret|", "ratio"),
    ("TtwolnAC", "T2", "FM ln A_C, with ln|Ret| (Table 2 model 3)", "t"),
    ("TtwolnAbs", "T2", "FM ln|Ret|, with ln A_C (Table 2 model 3)", "t"),
    ("HLArawCoef", "T3", "H-L on A, raw (%/mo)", "t"),
    ("HLAalphaCoef", "T3", "H-L on A, alpha (%/mo)", "t"),
    ("HLACrawCoef", "T3", "H-L on A_C, raw (%/mo)", "t"),
    ("HLACalphaCoef", "T3", "H-L on A_C, alpha (%/mo)", "t"),
    ("HLResArawCoef", "T3", "H-L on Res.A, raw (%/mo)", "t"),
    ("HLResAalphaCoef", "T3", "H-L on Res.A, alpha (%/mo)", "t"),
    ("FMlnA", "T4", "FM ln A, model (1)", "t"),
    ("FMlnAC", "T4", "FM ln A_C, model (2)", "t"),
    ("FMResLnA", "T4", "FM Res.lnA, model (3)", "t"),
    ("FMlnACfour", "T4", "FM ln A_C, model (4) with Res.lnA", "t"),
    ("FMResLnAfour", "T4", "FM Res.lnA, model (4) with ln A_C", "t"),
    ("FMlnACfive", "T4", "FM ln A_C, model (5) with Res.lnA + IVOL", "t"),
    ("FMResLnAivol", "T4", "FM Res.lnA, model (5) with ln A_C + IVOL", "t"),
    ("FMivol", "T4", "FM IVOL, model (5)", "t"),
    ("TsixLnVOL", "T6", "FM ln DVol, with Res.lnA (Table 6 col. 1)", "t"),
    ("TsixResLnA", "T6", "FM Res.lnA, with ln DVol (Table 6 col. 1)", "t"),
    ("IMLAalpha", "A.1", "alpha of H-L(A) on IMLA_C (%/mo)", "t"),
    ("IMLAalphaFive", "A.1", "alpha of H-L(A) on IMLA_C + factors (%/mo)", "t"),
    ("RawlnA", "A.3", "FM ln A, raw returns, model (1)", "t"),
    ("RawlnAC", "A.3", "FM ln A_C, raw returns, model (2)", "t"),
    ("RawResLnA", "A.3", "FM Res.lnA, raw returns, model (3)", "t"),
    ("RawResLnAfive", "A.3", "FM Res.lnA, raw returns, model (5)", "t"),
    ("SubAlnAC", "A.3", "FM ln A_C (2), first half vs 1964-1988", "t"),
    ("SubAResLnA", "A.3", "FM Res.lnA (4), first half vs 1964-1988", "t"),
    ("SubBlnAC", "A.3", "FM ln A_C (2), second half vs 1989-2012", "t"),
    ("SubBResLnA", "A.3", "FM Res.lnA (4), second half vs 1989-2012", "t"),
    ("ANResLnA", "A&N", "A&N (2021) col. 4: RlnILLIQ with ln A_C + IVOL", "t"),
    ("ANDIF", "A&N", "A&N (2021) col. 3: DIF with ln A_C + ln|Ret|", "t"),
]

SECTIONS = {
    "T1": "Table 1: descriptive statistics",
    "T2": "Table 2: decomposition of ln A",
    "T3": "Table 3: quintile sorts (EW, signal in t-2)",
    "T4": "Table 4: Fama-MacBeth, risk-adjusted returns",
    "T6": "Table 6: dollar volume directly",
    "A.1": "Internet Appendix A.1: IMLA_C factor",
    "A.3": "Internet Appendix A.3: raw returns and sub-periods",
    "A&N": "Amihud & Noh (2021) critique (upgrade 1, base run)",
}

# Why our number differs / matches. Templates: {name} is filled from the diagnostics
# (report_diagnostics.csv) and from the row itself ({us}, {us_t}, {paper}, {paper_t}, {mde},
# {months_needed}). "Hypothesis:" marks explanations we have not tested.
WHY = {
    "corrAAC": "Matches. A and A_C share the 1/DVol term, and across coins ln DVol varies much more than "
               "ln|Ret| (Q75/Q25 {qratioAC} for A_C vs {qratioAbs} for |Ret|), so A is mostly A_C, as in the paper.",
    "corrAabs": "Opposite sign. Across coins dollar volume rises with volatility: the average monthly "
                "cross-sectional corr(ln DVol, ln|Ret|) is {corrVolAbs}, and the slope of ln DVol on ln|Ret| "
                "averages {elasMean} (median {elasMed}). Dividing |Ret| by DVol therefore cancels the |Ret| "
                "variation: corr(ln A, ln|Ret|) = {corrLnAAbs}. Hypothesis: in stocks, size drives both low "
                "volume and high volatility, so they move together; in crypto, volatility attracts trading "
                "(retail attention, pumps, possibly wash trading).",
    "corrACabs": "Opposite sign, same mechanism: volatile coins have higher DVol and so lower A_C "
                 "(corr(ln A_C, ln|Ret|) = {corrLnACAbs}). In crypto a 'liquid' (low A_C) coin tends to be a "
                 "volatile one, while the |Ret| part of A sits in Res. A (corr(Res.A, |Ret|) = {corrResAAbs}).",
    "QratioAC": "Liquidity is much less dispersed than among NYSE/AMEX stocks. The universe is Binance USDT "
                "pairs and the $10k/day filter removes the illiquid tail (D8); median DVol is {medDvolM} $M/day. "
                "Hypothesis: listing on Binance itself acts as a liquidity screen.",
    "QratioAbs": "Relative dispersion is similar, but the level is much higher: median daily |Ret| is "
                 "{absretMed}% here vs {paperAbsretMed}% in the paper.",
    "TtwolnAC": "Same sign, not significant (t {us_t}). The positive A_C part and negative |Ret| part of the "
                "paper's decomposition both appear, but only the |Ret| part is significant. There is no ln ME "
                "or B/M control (D2, D3). MDE80 = {mde}.",
    "TtwolnAbs": "Matches: ln|Ret| is negative and significant at 1.96 (t {us_t}) but below the HLZ bar of "
                 "{hlz}. Slopes are in %/mo per log unit, and coin returns are far more dispersed (s.d. {retSd}%/mo), "
                 "so all our slopes are larger in absolute terms. Hypothesis: this is the same volatility effect "
                 "that Res. ln A picks up in Table 4 (corr(Res.A, |Ret|) = {corrResAAbs}).",
    "HLArawCoef": "Same sign with a larger point estimate, but not significant. Coin-month returns are very "
                  "noisy (s.d. {retSd}%/mo, p99 {retP99}%), and we have {nMonths} months (paper {paperMonths}) with "
                  "{avgCoins} coins per month, so MDE80 = {mde}%/mo against the paper's {paper}. Most influential month: {topMonthA} (H-L "
                  "{topValA}%, CMKT {cmktTopA}%); without it the mean is {exMeanA} (t {exTA}).",
    "HLAalphaCoef": "Same sign, not significant. Alpha is on CMKT+CMOM, not FF3+UMD (D6). It is smaller than "
                    "the raw spread ({rawHLA}), so part of the raw spread is CMKT/CMOM exposure. MDE80 = {mde}.",
    "HLACrawCoef": "Same sign, not significant (MDE80 = {mde}%/mo; at this point estimate t would reach 1.96 "
                   "after about {months_needed} months, an estimate assuming SE falls as 1/sqrt(T)). Most influential month: {topMonthAC} (H-L {topValAC}%, "
                   "CMKT {cmktTopAC}%, cross-sectional mean coin return {meanRetTopAC}%); without it the spread is "
                   "{exMeanAC} (t {exTAC}). Median monthly H-L {medianAC}; positive in {sharePosAC}% of months. "
                   "By half: {subAHLAC} (t {subAHLACt}) in {subARange} vs {subBHLAC} (t {subBHLACt}) in {subBRange}.",
    "HLACalphaCoef": "Same sign, not significant. Alpha is on CMKT+CMOM (D6) and is smaller than the raw "
                     "spread ({rawHLAC}). MDE80 = {mde}.",
    "HLResArawCoef": "Matches the paper: no significant spread on the price-impact residual (quintiles Q1..Q5: "
                     "{qResA}). With MDE80 = {mde}, a null here is weak evidence on its own.",
    "HLResAalphaCoef": "Matches (not significant). Alpha is on CMKT+CMOM (D6). Two of the weighting variants are "
                       "significant: liquid half {liqResAalpha} (t {liqResAalphaT}) and DVol weights {vwResAalpha} "
                       "(t {vwResAalphaT}), with the sign of the negative Res. ln A slope in FM; their raw-return "
                       "spreads are not ({liqResAraw}, t {liqResArawT}; {vwResAraw}, t {vwResArawT}). See the "
                       "robustness table for all variants.",
    "FMlnA": "Same sign, not significant. Returns are CMKT-adjusted with 36-month betas (D4), with no ln ME or "
             "B/M controls (D2, D3). FM uses {fmMonths} months and {fmAvgObs} coins per month (paper "
             "{paperMonths} and {paperAvgObs}). MDE80 = {mde}.",
    "FMlnAC": "Same sign, not significant. Per cross-sectional s.d. of ln A_C, our slope is {psUs}%/mo vs the "
              "paper's {psPaper} (paper s.d. from IQR/1.349, an estimate), with MDE80 = {mdePs} per s.d. "
              "Without the most influential month ({fmTopMonthTwo}) the slope is {fmExTwo} (t {fmExTwoT}). "
              "First half {subAFm} (t {subAFmT}), second half {subBFm} (t {subBFmT}).",
    "FMResLnA": "Matches: negative and significant at 1.96 (t {us_t}), but below the HLZ bar of {hlz}. In "
                "world-0 simulations, NW(6) gives |t|>1.96 in {fprFmThree}% of runs (95% CI {fprFmThreeLo}-"
                "{fprFmThreeHi}), and the empirical 5% critical |t| is {critFmThree}; our |t| is above it. "
                "Without the most influential month ({fmTopMonthThree}) the slope is {fmExThree} "
                "(t {fmExThreeT}). Hypothesis: a volatility effect rather than a price-impact premium. Res. A is "
                "positively correlated with |Ret| ({corrResAAbs}), and ln|Ret| has the same negative sign in T2.",
    "FMlnACfour": "Same sign, not significant (t {us_t}). Without {fmTopMonthFourAC} the slope is "
                  "{fmExFourAC} (t {fmExFourACT}).",
    "FMResLnAfour": "Matches at 1.96 (t {us_t}), below HLZ {hlz}. Without {fmTopMonthFourRes} the slope is "
                    "{fmExFourRes} (t {fmExFourResT}).",
    "FMlnACfive": "Same sign, not significant (t {us_t}). MDE80 = {mde}.",
    "FMResLnAivol": "Matches at 1.96 (t {us_t}); adding IVOL moves the Res. ln A slope from {fmFourRes} "
                    "to {us}. Our IVOL is a residual from a one-factor daily CMKT model (D7), and both IVOL and "
                    "Res. ln A load on volatility.",
    "FMivol": "Different: IVOL is significantly negative here (t {us_t}) but not in the paper. The levels are "
              "not comparable (ours is a daily-residual s.d. from a one-factor CMKT model, D7); only sign and t "
              "are compared. Winsorized returns give t {winIvolT}.",
    "TsixLnVOL": "Same sign, not significant (t {us_t}). Our Res. ln A is the residual on ln A_C, whereas the "
                 "paper's Table 6 uses the residual on ln VOLUME; ln A_C is close to -ln DVol (mean cs corr "
                 "{corrLnACVol}).",
    "TsixResLnA": "Matches at 1.96 (t {us_t}): the residual stays negative with ln DVol in the regression. The "
                  "residual is still taken on ln A_C, not on ln VOLUME as in the paper (see the row above).",
    "IMLAalpha": "Matches: controlling for IMLA_C (A_C tercile H-L) removes the A spread (raw {rawHLA} to "
                 "{us}), as in the paper. Weak evidence, since the raw spread was already insignificant.",
    "IMLAalphaFive": "Matches (not significant). Our factors are IMLA_C + CMKT + CMOM, the paper's are "
                     "IMLA_C + FF3 + UMD (D6).",
    "RawlnA": "Same sign, not significant (t {us_t}). On raw returns the paper's own slope is also weaker "
              "than on FF3-adjusted returns.",
    "RawlnAC": "Same sign, not significant (t {us_t}); the paper's t {paper_t} is also below 1.96.",
    "RawResLnA": "Our raw-return residual slope is significant (t {us_t}), while the paper's is not. On raw "
                 "returns the Res. ln A effect is stronger here than in the paper.",
    "RawResLnAfive": "Matches: negative and significant at 1.96 (t {us_t}).",
    "SubAlnAC": "Same sign; significant at 1.96 (t {us_t}), but below the HLZ bar, in our first half "
                "({subARange}, {subAObs} coins per FM month on average vs {subBObs} in the second half).",
    "SubAResLnA": "Matches at 1.96 (t {us_t}) in the first half.",
    "SubBlnAC": "Opposite sign: in the second half ({subBRange}, {subBObs} coins per FM month) the ln A_C slope "
                "is {us} (t {us_t}). The paper's 1989-2012 model (2) value is printed as {paper}, ten times its "
                "other ln A_C slopes, probably a typo; its model (4) slope after 1989 is only {paperSubCFour} "
                "(t {paperSubCFourT}). Both samples show a weaker A_C effect later on.",
    "SubBResLnA": "Same sign, not significant in the second half (t {us_t}).",
    "ANResLnA": "This is the same regression as model (5), set against A&N col. 4. 'no' means the sign is "
                "opposite to A&N's (+), not that a regression is missing; our sign agrees with Lou & Shu. A&N "
                "use 12-month measures on CRSP 1955-2016 with excess returns, ours are monthly and CMKT-adjusted.",
    "ANDIF": "Opposite sign, not significant (t {us_t}). DIF = ln A - ln|Ret| - ln A_C is A&N's missing "
             "covariance term. COV < 0 in {shareCovNeg}% of coin-months, i.e. 1/DVol is low on high-|r| days. "
             "Our measures are monthly (A&N use 12-month windows), with no size or B/M controls (D2, D3). "
             "DIF-sorted H-L: {difHL} (t {difHLT}).",
}


# ----------------------------------------------------------------------------------------- helpers

def nan_tuple():
    return (math.nan, None)


def ffmt(x, d: int = 2) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not math.isfinite(x) else f"{x:.{d}f}"


class SafeDict(dict):
    def __missing__(self, key):
        return "n/a"


def _read(rd: Path, name: str) -> pd.DataFrame | None:
    f = rd / f"{name}.csv"
    return pd.read_csv(f) if f.exists() else None


def _fm(df: pd.DataFrame | None, model: str, var: str, **filt) -> tuple[float, float | None]:
    if df is None:
        return nan_tuple()
    m = (df["model"] == model) & (df["var"] == var)
    for k, v in filt.items():
        m &= df[k] == v
    r = df[m]
    return (float(r["coef"].iloc[0]), float(r["t"].iloc[0])) if len(r) else nan_tuple()


def _hlz(cfg_path: Path = ROOT / "config.yaml") -> float:
    try:
        return float(yaml.safe_load(cfg_path.read_text())["tests"]["hlz_threshold"])
    except Exception:
        return 3.0


def _months_split(months: list[str]) -> tuple[str, str]:
    """Same split as analysis.run_tests: first half = months before months[len // 2]."""
    months = sorted(months)
    half = months[len(months) // 2]
    first = [m for m in months if m < half]
    second = [m for m in months if m >= half]
    return f"{first[0]}..{first[-1]}", f"{second[0]}..{second[-1]}"


# ------------------------------------------------------------------------------------ paper / ours

def paper_values(p: dict) -> dict[str, tuple[float, float | None]]:
    ls, an = p["lou_shu_2017"], p["amihud_noh_2021"]
    t1a, t1b, t3, t4 = ls["table1_panelA"], ls["table1_panelB"], ls["table3_panelA"], ls["table4_panelA"]
    a3 = ls["tableA3"]
    v = {"corrAAC": (t1b["corr_A_AC"], None), "corrAabs": (t1b["corr_A_absret"], None),
         "corrACabs": (t1b["corr_AC_absret"], None),
         "QratioAC": (t1a["A_C"]["q75"] / t1a["A_C"]["q25"], None),
         "QratioAbs": (t1a["absret"]["q75"] / t1a["absret"]["q25"], None)}
    tw = ls["table2_model3"]
    v["TtwolnAC"] = (tw["lnA_C"]["coef"], tw["lnA_C"]["t"])
    v["TtwolnAbs"] = (tw["ln_absret"]["coef"], tw["ln_absret"]["t"])
    for sig, key in [("A", "A"), ("A_C", "AC"), ("ResA", "ResA")]:
        for row in ["raw", "alpha"]:
            v[f"HL{key}{row}Coef"] = (t3[sig][row]["hl"], t3[sig][row]["t"])
    ct = lambda d: (d["coef"], d["t"])  # noqa: E731
    v["FMlnA"] = ct(t4["(1)"]["lnA"])
    v["FMlnAC"] = ct(t4["(2)"]["lnA_C"])
    v["FMResLnA"] = ct(t4["(3)"]["ResLnA"])
    v["FMlnACfour"] = ct(t4["(4)"]["lnA_C"])
    v["FMResLnAfour"] = ct(t4["(4)"]["ResLnA"])
    v["FMlnACfive"] = ct(t4["(5)"]["lnA_C"])
    v["FMResLnAivol"] = ct(t4["(5)"]["ResLnA"])
    v["FMivol"] = ct(t4["(5)"]["ivol"])
    v["TsixLnVOL"] = ct(ls["table6"]["(1)"]["lnVOL"])
    v["TsixResLnA"] = ct(ls["table6"]["(1)"]["ResLnA"])
    ia = ls["tableA1_panelA"]
    v["IMLAalpha"] = ct({"coef": ia["imla_one_factor"]["alpha"], "t": ia["imla_one_factor"]["t"]})
    v["IMLAalphaFive"] = ct({"coef": ia["imla_five_factor"]["alpha"], "t": ia["imla_five_factor"]["t"]})
    v["RawlnA"] = ct(a3["panelA_raw"]["(1)"]["lnA"])
    v["RawlnAC"] = ct(a3["panelA_raw"]["(2)"]["lnA_C"])
    v["RawResLnA"] = ct(a3["panelA_raw"]["(3)"]["ResLnA"])
    v["RawResLnAfive"] = ct(a3["panelA_raw"]["(5)"]["ResLnA"])
    v["SubAlnAC"] = ct(a3["panelB_1964_1988"]["(2)"]["lnA_C"])
    v["SubAResLnA"] = ct(a3["panelB_1964_1988"]["(4)"]["ResLnA"])
    v["SubBlnAC"] = ct(a3["panelC_1989_2012"]["(2)"]["lnA_C"])
    v["SubBResLnA"] = ct(a3["panelC_1989_2012"]["(4)"]["ResLnA"])
    v["ANResLnA"] = ct(an["col4"]["RlnILLIQ"])
    v["ANDIF"] = ct(an["col3"]["DIF"])
    return {k: (float(a), None if b is None else float(b)) for k, (a, b) in v.items()}


def our_values(rd: Path) -> dict[str, tuple[float, float | None]]:
    """Read our numbers from the run's CSVs; a missing table gives NaN for its rows."""
    v: dict[str, tuple[float, float | None]] = {}
    corr = _read(rd, "T1_corr")
    if corr is not None:
        c = corr.set_index("measure")
        v["corrAAC"], v["corrAabs"], v["corrACabs"] = ((float(c.loc[a, b]), None) for a, b in
                                                       [("A", "A_C"), ("A", "absret"), ("A_C", "absret")])
    desc = _read(rd, "T1_describe")
    if desc is not None:
        d = desc.set_index("measure")
        v["QratioAC"] = (float(d.loc["A_C", "75%"] / d.loc["A_C", "25%"]), None)
        v["QratioAbs"] = (float(d.loc["absret", "75%"] / d.loc["absret", "25%"]), None)
    t2 = _read(rd, "T2_fm")
    v["TtwolnAC"], v["TtwolnAbs"] = _fm(t2, "(3)", "lnA_C"), _fm(t2, "(3)", "ln_absret")
    t3 = _read(rd, "T3_sorts")
    if t3 is not None:
        t3 = t3.set_index(["signal", "row"])
        for sig, key in [("A", "A"), ("A_C", "AC"), ("ResA", "ResA")]:
            for row in ["raw", "alpha"]:
                r = t3.loc[(sig, row)]
                v[f"HL{key}{row}Coef"] = (float(r["H-L"]), float(r["t"]))
    t4 = _read(rd, "T4_fm")
    v["FMlnA"], v["FMlnAC"], v["FMResLnA"] = _fm(t4, "(1)", "lnA"), _fm(t4, "(2)", "lnA_C"), _fm(t4, "(3)", "ResLnA")
    v["FMlnACfour"], v["FMResLnAfour"] = _fm(t4, "(4)", "lnA_C"), _fm(t4, "(4)", "ResLnA")
    v["FMlnACfive"], v["FMResLnAivol"], v["FMivol"] = (_fm(t4, "(5)", "lnA_C"), _fm(t4, "(5)", "ResLnA"),
                                                       _fm(t4, "(5)", "ivol"))
    t6 = _read(rd, "T6_fm")
    v["TsixLnVOL"], v["TsixResLnA"] = _fm(t6, "(2)", "lnVOL"), _fm(t6, "(2)", "ResLnA")
    t5 = _read(rd, "T5_imla")
    if t5 is not None:
        for mid, model in [("IMLAalpha", "IMLA_C"), ("IMLAalphaFive", "IMLA_C+CMKT+CMOM")]:
            r = t5[t5["model"] == model]
            if len(r):
                v[mid] = (float(r["alpha_HL_A"].iloc[0]), float(r["t"].iloc[0]))
    raw = _read(rd, "R_raw_fm")
    v["RawlnA"], v["RawlnAC"] = _fm(raw, "(1)", "lnA"), _fm(raw, "(2)", "lnA_C")
    v["RawResLnA"], v["RawResLnAfive"] = _fm(raw, "(3)", "ResLnA"), _fm(raw, "(5)", "ResLnA")
    sp = _read(rd, "R_subperiods")
    v["SubAlnAC"] = _fm(sp, "(2)", "lnA_C", subsample="first half")
    v["SubAResLnA"] = _fm(sp, "(4)", "ResLnA", subsample="first half")
    v["SubBlnAC"] = _fm(sp, "(2)", "lnA_C", subsample="second half")
    v["SubBResLnA"] = _fm(sp, "(4)", "ResLnA", subsample="second half")
    u1 = _read(rd, "U1_an_fm")
    v["ANResLnA"] = _fm(u1, "AN: ResLnA+lnA_C+ivol", "ResLnA")
    v["ANDIF"] = _fm(u1, "AN: lnA_C+ln|Ret|+DIF", "DIF")
    return {k: v.get(k, nan_tuple()) for k, _, _, _ in METRICS}


def match(kind: str, paper: tuple, us: tuple) -> str:
    """Rule-based verdict (see the legend in paper_vs_us.md)."""
    pv, pt = paper
    uv, ut = us
    if uv is None or not math.isfinite(uv):
        return "n/a"
    same = math.copysign(1, pv) == math.copysign(1, uv)
    if kind == "ratio":
        return "yes" if abs(math.log10(abs(uv) / abs(pv))) <= 0.5 else "no"
    if kind == "level":
        if not same:
            return "no"
        return "yes" if abs(math.log10(abs(uv) / abs(pv))) <= 0.5 else "sign only"
    sig_us = ut is not None and abs(ut) >= T_CRIT
    if abs(pt) >= T_CRIT:                     # paper significant
        return "no" if not same else ("yes" if sig_us else "sign only")
    if not sig_us:                            # paper insignificant: so are we
        return "yes"
    return "sign only" if same else "no"


def mde80(coef: float, t: float | None) -> float:
    """Minimum detectable effect at 80% power, two-sided 5%: (1.96 + 0.84) * SE, SE = |coef / t|."""
    if t is None or not math.isfinite(t) or t == 0 or not math.isfinite(coef):
        return math.nan
    return (T_CRIT + Z_POWER80) * abs(coef / t)


def months_needed(t: float | None, n: float) -> float:
    """Months for |t| to reach 1.96 at the same point estimate if SE scales as 1/sqrt(T) (estimate)."""
    if t is None or not math.isfinite(t) or t == 0 or not math.isfinite(n):
        return math.nan
    return n * (T_CRIT / t) ** 2


# ------------------------------------------------------------------------------------- diagnostics

class Diag:
    """Collects diagnostics: id (letters only), value, label (text), description, source."""

    def __init__(self):
        self.rows: list[dict] = []

    def add(self, did: str, value=math.nan, desc: str = "", source: str = "", label: str | None = None,
            d: int = 2):
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = math.nan
        self.rows.append({"id": did, "value": value, "label": label if label is not None else ffmt(value, d),
                          "description": desc, "source": source})

    def ctx(self) -> dict[str, str]:
        return {r["id"]: r["label"] for r in self.rows}

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=["id", "value", "label", "description", "source"])


def _panel(rd: Path, man: dict) -> pd.DataFrame | None:
    p = man.get("panel_cache")
    cands = [Path(p)] if p else []
    cands.append(ROOT / "data" / "cache" / f"panel_{rd.name}.parquet")
    for c in cands:
        if c.exists():
            return pd.read_parquet(c)
    return None


def _fm_loo(panel: pd.DataFrame, xs: list[str], var: str, lags: int):
    """FM on ret_adj (as T4); returns (coef, t, most influential month, coef and t without it)."""
    from fama_macbeth import fama_macbeth, nw_mean
    res = fama_macbeth(panel, "ret_adj", xs + CONTROLS, lags=lags)
    s = res.attrs["coef_series"][var]
    top = (s - s.mean()).abs().idxmax()
    m, _, t = nw_mean(s.drop(top), lags)
    full = res.set_index("var").loc[var]
    return float(full["coef"]), float(full["t"]), str(top), float(s[top]), m, t


def build_diagnostics(rd: Path, sim_dir: Path, man: dict, paper_yaml: dict, us: dict, lags: int = 6) -> Diag:
    from fama_macbeth import nw_mean
    D = Diag()
    ls = paper_yaml["lou_shu_2017"]
    D.add("hlz", _hlz(), "Harvey-Liu-Zhu (2016) t bar, config tests.hlz_threshold", "config.yaml", d=1)
    D.add("paperMonths", ls["sample"]["months"], "Lou & Shu months", "paper_numbers.yaml", d=0)
    D.add("paperAvgObs", ls["sample"]["avg_obs"], "Lou & Shu avg firms per FM month", "paper_numbers.yaml", d=0)
    D.add("paperAbsretMed", 100 * ls["table1_panelA"]["absret"]["q50"], "paper median daily |Ret|, %",
          "paper_numbers.yaml", d=1)

    s = _read(rd, "sample")
    if s is not None:
        r = s.iloc[0]
        D.add("nMonths", r["n_months"], "return months in sorts", "sample.csv", d=0)
        D.add("nCoins", r["n_coins"], "distinct coins", "sample.csv", d=0)
        D.add("avgCoins", r["avg_coins_per_month"], "avg coins per month (sorts)", "sample.csv", d=0)
    t4 = _read(rd, "T4_fm")
    if t4 is not None:
        D.add("fmMonths", t4["n_months"].iloc[0], "FM months (T4)", "T4_fm.csv", d=0)
        D.add("fmAvgObs", t4["avg_obs"].iloc[0], "avg coins per FM month (T4)", "T4_fm.csv", d=0)
        D.add("fmFourRes", _fm(t4, "(4)", "ResLnA")[0], "Res.lnA slope, model (4)", "T4_fm.csv")

    rdist = _read(rd, "ret_distribution")
    if rdist is not None:
        q = rdist.set_index("stat")
        D.add("retSd", q.loc["sd", "ret"], "s.d. of monthly coin returns, %", "ret_distribution.csv", d=1)
        D.add("retP1", q.loc["p1", "ret"], "p1 of monthly coin returns, %", "ret_distribution.csv", d=1)
        D.add("retP99", q.loc["p99", "ret"], "p99 of monthly coin returns, %", "ret_distribution.csv", d=1)
        D.add("retMax", q.loc["p100", "ret"], "max monthly coin return, %", "ret_distribution.csv", d=1)
        D.add("retAdjMin", q.loc["p0", "ret_adj"], "min CMKT-adjusted return, %", "ret_distribution.csv", d=1)
        D.add("retAdjMax", q.loc["p100", "ret_adj"], "max CMKT-adjusted return, %", "ret_distribution.csv", d=1)
    fac = _read(rd, "factors")
    if fac is not None:
        i = fac["CMKT"].idxmax()
        D.add("cmktMax", fac.loc[i, "CMKT"], "max monthly CMKT, %", "factors.csv", d=1)
        D.add("cmktMaxMonth", label=str(fac.loc[i, "month"]), desc="month of max CMKT", source="factors.csv")
        D.add("cmktMean", fac["CMKT"].mean(), "mean CMKT, %/mo", "factors.csv")

    desc = _read(rd, "T1_describe")
    if desc is not None:
        dd = desc.set_index("measure")
        D.add("medDvolM", dd.loc["dvol", "50%"] / 1e6, "median avg daily DVol, $M", "T1_describe.csv", d=1)
        D.add("absretMean", 100 * dd.loc["absret", "mean"], "mean of monthly avg daily |Ret|, %", "T1_describe.csv")
        D.add("absretMed", 100 * dd.loc["absret", "50%"], "median of monthly avg daily |Ret|, %", "T1_describe.csv")
        D.add("qratioAC", dd.loc["A_C", "75%"] / dd.loc["A_C", "25%"], "Q75/Q25 A_C", "T1_describe.csv", d=1)
        D.add("qratioAbs", dd.loc["absret", "75%"] / dd.loc["absret", "25%"], "Q75/Q25 |Ret|", "T1_describe.csv")
    corr = _read(rd, "T1_corr")
    if corr is not None:
        D.add("corrResAAbs", corr.set_index("measure").loc["ResA", "absret"], "mean cs corr(Res.A, |Ret|)",
              "T1_corr.csv")
    u1d = _read(rd, "U1_an_describe")
    if u1d is not None:
        D.add("shareCovNeg", 100 * u1d["share_COV_negative"].iloc[0], "share of coin-months with COV<0, %",
              "U1_an_describe.csv", d=1)
    u1s = _read(rd, "U1_an_sorts")
    if u1s is not None:
        r = u1s[(u1s["signal"] == "DIF") & (u1s["row"] == "raw")]
        if len(r):
            D.add("difHL", r["H-L"].iloc[0], "H-L on DIF, raw %/mo", "U1_an_sorts.csv")
            D.add("difHLT", r["t"].iloc[0], "t of H-L on DIF", "U1_an_sorts.csv")
    D.add("rawHLA", us["HLArawCoef"][0], "raw H-L on A", "T3_sorts.csv")
    D.add("rawHLAC", us["HLACrawCoef"][0], "raw H-L on A_C", "T3_sorts.csv")
    w = _read(rd, "R_winsor_returns")
    if w is not None:
        D.add("winIvolT", _fm(w[w["test"] == "T4"], "(5)", "ivol")[1], "t of IVOL, model (5), winsorized returns",
              "R_winsor_returns.csv")

    # sorts: most influential month, leave-one-out, median, share positive; sub-period sort H-L
    ps_ = _read(rd, "panel_summary")
    psum = ps_.set_index("month")["mean_ret"] if ps_ is not None else None
    t3s = _read(rd, "T3_sorts")
    if t3s is not None:
        r = t3s[(t3s["signal"] == "ResA") & (t3s["row"] == "raw")]
        if len(r):
            D.add("qResA", label=" / ".join(ffmt(r[f"Q{k}"].iloc[0]) for k in range(1, 6)),
                  desc="raw EW quintile returns, Res.A sort, %/mo", source="T3_sorts.csv")
    rw = _read(rd, "R_weighting")
    if rw is not None:
        for samp, key in [("EW, liquid half", "liq"), ("DVol-weighted, all", "vw")]:
            for row, nm in [("alpha", "alpha"), ("raw", "raw")]:
                r = rw[(rw["sample"] == samp) & (rw["signal"] == "ResA") & (rw["row"] == row)]
                if len(r):
                    D.add(f"{key}ResA{nm}", r["H-L"].iloc[0], f"Res.A H-L {row}, {samp}", "R_weighting.csv")
                    D.add(f"{key}ResA{nm}T", r["t"].iloc[0], "t of the same", "R_weighting.csv")
    port = _read(rd, "T3_portfolios")
    if port is not None:
        for sig, key in [("A", "A"), ("A_C", "AC"), ("ResA", "ResA")]:
            g = port[port["signal"] == sig].set_index("month")["HL"].dropna().sort_index()
            top = (g - g.mean()).abs().idxmax()
            m, _, t = nw_mean(g.drop(top), lags)
            D.add(f"topMonth{key}", label=str(top), desc=f"most influential month for raw H-L {sig}",
                  source="T3_portfolios.csv")
            D.add(f"topVal{key}", g[top], f"raw H-L {sig} in that month, %", "T3_portfolios.csv", d=1)
            D.add(f"exMean{key}", m, f"raw H-L {sig} without that month, %/mo", "T3_portfolios.csv")
            D.add(f"exT{key}", t, f"NW t of raw H-L {sig} without that month", "T3_portfolios.csv")
            D.add(f"median{key}", g.median(), f"median monthly raw H-L {sig}", "T3_portfolios.csv")
            D.add(f"sharePos{key}", 100 * (g > 0).mean(), f"% of months with raw H-L {sig} > 0",
                  "T3_portfolios.csv", d=0)
            if fac is not None:
                cm = fac.set_index("month")["CMKT"]
                D.add(f"cmktTop{key}", cm.get(top, math.nan), f"CMKT in the most influential month ({sig}), %",
                      "factors.csv", d=1)
            if psum is not None:
                D.add(f"meanRetTop{key}", psum.get(top, math.nan),
                      f"cross-sectional mean coin return in that month ({sig}), %", "panel_summary.csv", d=1)
        a, b = _months_split(port["month"].unique().tolist())
        D.add("subARange", label=a, desc="first half (sort months)", source="T3_portfolios.csv")
        D.add("subBRange", label=b, desc="second half (sort months)", source="T3_portfolios.csv")
    sp = _read(rd, "R_subperiods")
    if sp is not None:
        for half, k in [("first half", "A"), ("second half", "B")]:
            c, t = _fm(sp, "sort A_C", "H-L", subsample=half)
            D.add(f"sub{k}HLAC", c, f"raw H-L A_C, {half}", "R_subperiods.csv")
            D.add(f"sub{k}HLACt", t, f"t, raw H-L A_C, {half}", "R_subperiods.csv")
            c, t = _fm(sp, "(2)", "lnA_C", subsample=half)
            D.add(f"sub{k}Fm", c, f"FM (2) ln A_C slope, {half}", "R_subperiods.csv")
            D.add(f"sub{k}FmT", t, f"t, FM (2) ln A_C, {half}", "R_subperiods.csv")
            r = sp[(sp["subsample"] == half) & (sp["model"] == "(2)")]
            D.add(f"sub{k}Obs", r["avg_obs"].iloc[0] if len(r) else math.nan,
                  f"avg coins per FM month, {half}", "R_subperiods.csv", d=0)
    a3c = ls["tableA3"]["panelC_1989_2012"]["(4)"]["lnA_C"]
    D.add("paperSubCFour", a3c["coef"], "paper A.3 Panel C model (4) ln A_C", "paper_numbers.yaml", d=3)
    D.add("paperSubCFourT", a3c["t"], "t of the same", "paper_numbers.yaml")

    # panel diagnostics: volume-volatility link, per-s.d. effects, FM leave-one-month-out
    panel = _panel(rd, man)
    src = "panel cache (run_manifest.json: panel_cache)"
    if panel is not None:
        from measures import mean_cs_corr
        cm = mean_cs_corr(panel, ["lnVOL", "ln_absret", "lnA", "lnA_C"], "month")
        D.add("corrVolAbs", cm.loc["lnVOL", "ln_absret"], "mean cs corr(ln DVol, ln|Ret|)", src)
        D.add("corrLnAAbs", cm.loc["lnA", "ln_absret"], "mean cs corr(ln A, ln|Ret|)", src)
        D.add("corrLnACAbs", cm.loc["lnA_C", "ln_absret"], "mean cs corr(ln A_C, ln|Ret|)", src)
        D.add("corrLnACVol", cm.loc["lnA_C", "lnVOL"], "mean cs corr(ln A_C, ln DVol)", src)
        sl = [np.polyfit(g["ln_absret"], g["lnVOL"], 1)[0]
              for _, g in panel[["month", "ln_absret", "lnVOL"]].dropna().groupby("month") if len(g) >= 10]
        D.add("elasMean", np.mean(sl), "mean monthly cs slope of ln DVol on ln|Ret|", src)
        D.add("elasMed", np.median(sl), "median monthly cs slope of ln DVol on ln|Ret|", src)
        sd = panel.groupby("month")["lnA_C"].std().mean()
        D.add("sdLnAC", sd, "mean monthly cs s.d. of ln A_C", src)
        c, t = us["FMlnAC"]
        D.add("psUs", c * sd, "FM (2) ln A_C slope per 1 cs s.d., %/mo", src)
        D.add("mdePs", mde80(c, t) * sd, "MDE80 of FM (2) ln A_C per 1 cs s.d., %/mo", src)
        for xs, var, key, model in [(["lnA_C"], "lnA_C", "Two", "(2)"), (["ResLnA"], "ResLnA", "Three", "(3)"),
                                    (["lnA_C", "ResLnA"], "lnA_C", "FourAC", "(4)"),
                                    (["lnA_C", "ResLnA"], "ResLnA", "FourRes", "(4)")]:
            c, _, top, vtop, m, tt = _fm_loo(panel, xs, var, lags)
            D.add(f"fmTopMonth{key}", label=top, desc=f"most influential month, FM {model} {var}", source=src)
            D.add(f"fmTopVal{key}", vtop, f"FM {model} {var}: monthly slope in that month", src)
            D.add(f"fmEx{key}", m, f"FM {model} {var}: slope without that month", src)
            D.add(f"fmEx{key}T", tt, f"FM {model} {var}: NW t without that month", src)
            # consistency check: the slope recomputed from the cached panel must equal T4_fm.csv
            D.add(f"fmReproDiff{key}", abs(c - _fm(t4, model, var)[0]),
                  f"|FM {model} {var} from panel - T4_fm.csv| (should be ~0)", src, d=6)
    ps_paper = ls["table4_panelA"]["(2)"]["lnA_C"]["coef"] * math.log(
        ls["table1_panelA"]["A_C"]["q75"] / ls["table1_panelA"]["A_C"]["q25"]) / 1.349
    D.add("psPaper", ps_paper, "paper FM (2) ln A_C slope per 1 s.d. (IQR/1.349 of pooled ln A_C, estimate)",
          "paper_numbers.yaml")

    # simulations: size of the tests in world 0 and power in world V
    ws = _read(sim_dir, "sim_world_summary")
    if ws is not None:
        w0 = ws[ws["world"].astype(str) == "0"].set_index("stat")
        for stat, key in [("fm2_lnA_C", "FmTwo"), ("fm3_ResLnA", "FmThree"), ("hl_A_C", "HlAC"),
                          ("fm3_ResLnA_nw0", "FmThreeNWzero")]:
            if stat in w0.index:
                r = w0.loc[stat]
                D.add(f"fpr{key}", 100 * r["share_abs_t_gt_1.96"], f"world 0: % of runs with |t|>1.96, {stat}",
                      "sim_world_summary.csv", d=1)
                D.add(f"fpr{key}Lo", 100 * r["ci95_lo"], "95% CI low", "sim_world_summary.csv", d=1)
                D.add(f"fpr{key}Hi", 100 * r["ci95_hi"], "95% CI high", "sim_world_summary.csv", d=1)
                D.add(f"fprThree{key}", 100 * r["share_abs_t_gt_3"], f"world 0: % of runs with |t|>3, {stat}",
                      "sim_world_summary.csv", d=1)
    w0r = _read(sim_dir, "sim_world0_reps")
    if w0r is not None:
        for stat, key in [("fm2_lnA_C", "FmTwo"), ("fm3_ResLnA", "FmThree"), ("hl_A_C", "HlAC"), ("hl_A", "HlA")]:
            col = f"{stat}_t"
            if col in w0r:
                D.add(f"crit{key}", w0r[col].abs().quantile(0.95), f"world 0: empirical 5% critical |t|, {stat}",
                      "sim_world0_reps.csv")
        if "hl_A_C_coef" in w0r:
            # sampling s.d. of the H-L A_C spread across world-0 runs vs our real NW standard error
            # (both: EW quintile spread, %/mo, ~300 coins x ~90 months)
            sim_se = w0r["hl_A_C_coef"].std(ddof=1)
            c, t = us["HLACrawCoef"]
            real_se = abs(c / t) if t else math.nan
            D.add("simSeHlAC", sim_se, "world 0: s.d. of H-L A_C across runs, %/mo", "sim_world0_reps.csv")
            D.add("realSeHlAC", real_se, "our NW s.e. of H-L A_C, %/mo", "T3_sorts.csv")
            D.add("seRatioHlAC", real_se / sim_se, "real s.e. / simulated s.e., H-L A_C", "T3_sorts.csv + sims", d=1)
        if "n_months" in w0r and "avg_coins" in w0r:
            D.add("simMonths", w0r["n_months"].mean(), "world 0: months per simulated sample", "sim_world0_reps.csv", d=0)
            D.add("simCoins", w0r["avg_coins"].mean(), "world 0: avg coins per month", "sim_world0_reps.csv", d=0)
            D.add("simReps", len(w0r), "world 0: number of runs", "sim_world0_reps.csv", d=0)
        if "mean_absret_daily" in w0r:
            D.add("simAbsret", 100 * w0r["mean_absret_daily"].mean(), "world 0: mean daily |r|, %",
                  "sim_world0_reps.csv")
    pw = _read(sim_dir, "sim_power")
    if pw is not None:
        D.add("powReps", pw["n_reps"].min(), "world V: runs per premium level (min)", "sim_power.csv", d=0)
        for stat, key in [("fm2_lnA_C", "FmTwo"), ("hl_A_C", "HlAC")]:
            for prem, pk in [(0.0025, "Quarter"), (0.005, "Half"), (0.01, "One")]:
                r = pw[(pw["stat"] == stat) & np.isclose(pw["premium"], prem)]
                if len(r):
                    D.add(f"pow{key}{pk}", 100 * r["share_t_gt_1.96"].iloc[0],
                          f"world V power (t>1.96), {stat}, premium {prem * 100:g}%/mo per s.d.",
                          "sim_power.csv", d=0)
    pl = _read(rd, "placebo")
    if pl is not None:
        sh = pl[pl["test"] == "shuffle within month"]
        if len(sh):
            D.add("placeboShareMax", 100 * sh["share_abs_t_gt_1.96"].max(),
                  "shuffle placebo: max % of |t|>1.96 across stats", "placebo.csv", d=0)
            D.add("placeboReps", sh["n"].iloc[0], "shuffle placebo reps", "placebo.csv", d=0)
        lk = pl[pl["test"] == "leak (lag 0)"].set_index("var")
        for var, key in [("H-L A_C", "HLAC"), ("FM(2) lnA_C", "FmTwo")]:
            if var in lk.index:
                D.add(f"leakT{key}", lk.loc[var, "t"], f"t with lag-0 (leaked) signal, {var}", "placebo.csv")
    jan = _read(rd, "T7_january")
    if jan is not None:
        j = jan.set_index("subset")
        D.add("janCoef", j.loc["January", "coef"], "ln A_C slope, Januaries", "T7_january.csv")
        D.add("janT", j.loc["January", "t"], "t, Januaries", "T7_january.csv")
    au = _read(rd, "audit_summary")
    if au is not None:
        a = au.set_index("type")
        for typ, key in [("redenomination", "auditRedenom"), ("ticker_reuse", "auditReuse"),
                         ("long_gap", "auditGap")]:
            D.add(key, a.loc[typ, "n_events"] if typ in a.index else 0, f"audit events: {typ}",
                  "audit_summary.csv", d=0)
    return D


# --------------------------------------------------------------------------------- robustness

def build_robustness(rd: Path, us: dict, diag: dict[str, float]) -> pd.DataFrame:
    """Our base estimate vs each variant; same conclusion = same sign and same side of |t| = 1.96."""
    rows = []
    base = {"HL A raw": us["HLArawCoef"], "HL A_C raw": us["HLACrawCoef"], "HL Res.A raw": us["HLResArawCoef"],
            "HL A alpha": us["HLAalphaCoef"], "HL A_C alpha": us["HLACalphaCoef"],
            "HL Res.A alpha": us["HLResAalphaCoef"],
            "FM ln A_C (2)": us["FMlnAC"], "FM Res.lnA (3)": us["FMResLnA"],
            "FM ln A_C (4)": us["FMlnACfour"], "FM Res.lnA (4)": us["FMResLnAfour"]}

    def add(stat, variant, val, src):
        b = base[stat]
        same = (math.isfinite(val[0]) and math.copysign(1, b[0]) == math.copysign(1, val[0])
                and (abs(b[1]) >= T_CRIT) == (val[1] is not None and abs(val[1]) >= T_CRIT))
        rows.append({"statistic": stat, "variant": variant, "base": b[0], "base_t": b[1], "variant_coef": val[0],
                     "variant_t": val[1], "same_conclusion": "yes" if same else "no", "source": src})

    w = _read(rd, "R_winsor_returns")
    if w is not None:
        t3 = w[w["test"] == "T3"].set_index(["signal", "row"])
        for stat, sig, row in [("HL A raw", "A", "raw"), ("HL A_C raw", "A_C", "raw"), ("HL Res.A raw", "ResA", "raw"),
                               ("HL A alpha", "A", "alpha"), ("HL A_C alpha", "A_C", "alpha"),
                               ("HL Res.A alpha", "ResA", "alpha")]:
            r = t3.loc[(sig, row)]
            add(stat, "returns winsorized 1/99 within month", (float(r["H-L"]), float(r["t"])), "R_winsor_returns.csv")
        t4w = w[w["test"] == "T4"]
        for stat, model, var in [("FM ln A_C (2)", "(2)", "lnA_C"), ("FM Res.lnA (3)", "(3)", "ResLnA"),
                                 ("FM ln A_C (4)", "(4)", "lnA_C"), ("FM Res.lnA (4)", "(4)", "ResLnA")]:
            add(stat, "returns winsorized 1/99 within month", _fm(t4w, model, var), "R_winsor_returns.csv")
    raw = _read(rd, "R_raw_fm")
    for stat, model, var in [("FM ln A_C (2)", "(2)", "lnA_C"), ("FM Res.lnA (3)", "(3)", "ResLnA"),
                             ("FM ln A_C (4)", "(4)", "lnA_C"), ("FM Res.lnA (4)", "(4)", "ResLnA")]:
        if raw is not None:
            add(stat, "raw instead of CMKT-adjusted returns", _fm(raw, model, var), "R_raw_fm.csv")
    rw = _read(rd, "R_weighting")
    if rw is not None:
        for samp in ["EW, liquid half", "DVol-weighted, all"]:
            for stat, sig, row in [("HL A raw", "A", "raw"), ("HL A_C raw", "A_C", "raw"),
                                   ("HL Res.A raw", "ResA", "raw"), ("HL A alpha", "A", "alpha"),
                                   ("HL A_C alpha", "A_C", "alpha"), ("HL Res.A alpha", "ResA", "alpha")]:
                r = rw[(rw["sample"] == samp) & (rw["signal"] == sig) & (rw["row"] == row)]
                if len(r):
                    add(stat, samp, (float(r["H-L"].iloc[0]), float(r["t"].iloc[0])), "R_weighting.csv")
    sp = _read(rd, "R_subperiods")
    if sp is not None:
        for half in ["first half", "second half"]:
            for stat, sig in [("HL A raw", "A"), ("HL A_C raw", "A_C"), ("HL Res.A raw", "ResA")]:
                add(stat, half, _fm(sp, f"sort {sig}", "H-L", subsample=half), "R_subperiods.csv")
            for stat, model, var in [("FM ln A_C (2)", "(2)", "lnA_C"), ("FM ln A_C (4)", "(4)", "lnA_C"),
                                     ("FM Res.lnA (4)", "(4)", "ResLnA")]:
                add(stat, half, _fm(sp, model, var, subsample=half), "R_subperiods.csv")
    for stat, key in [("HL A raw", "A"), ("HL A_C raw", "AC"), ("HL Res.A raw", "ResA")]:
        if f"exMean{key}" in diag:
            add(stat, "without most influential month", (diag[f"exMean{key}"], diag[f"exT{key}"]),
                "T3_portfolios.csv")
    for stat, key in [("FM ln A_C (2)", "Two"), ("FM Res.lnA (3)", "Three"), ("FM ln A_C (4)", "FourAC"),
                      ("FM Res.lnA (4)", "FourRes")]:
        if f"fmEx{key}" in diag:
            add(stat, "without most influential month", (diag[f"fmEx{key}"], diag[f"fmEx{key}T"]), "panel cache")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------- build

def build(rd: Path, sim_dir: Path | None = None, man: dict | None = None):
    """Return (paper_vs_us table, robustness table, diagnostics table)."""
    man = man or {}
    sim_dir = sim_dir or rd.parent / "simulations"
    pyaml = yaml.safe_load(PAPER.read_text())
    paper = paper_values(pyaml)
    us = our_values(rd)
    D = build_diagnostics(rd, sim_dir, man, pyaml, us)
    ctx = D.ctx()
    n_months = float(D.frame().set_index("id")["value"].get("fmMonths", math.nan))
    n_sort = float(D.frame().set_index("id")["value"].get("nMonths", math.nan))
    hlz = _hlz()
    rows = []
    for mid, sec, label, kind in METRICS:
        (pv, pt), (uv, ut) = paper[mid], us[mid]
        n = n_sort if mid.startswith("HL") or mid.startswith("IMLA") else n_months
        mde = mde80(uv, ut) if kind == "t" else math.nan
        mn = months_needed(ut, n) if kind == "t" else math.nan
        rowctx = SafeDict(ctx)
        rowctx.update({"us": ffmt(uv, 3), "us_t": ffmt(ut), "paper": ffmt(pv, 3), "paper_t": ffmt(pt),
                       "mde": ffmt(mde), "months_needed": ffmt(mn, 0), "hlz": ffmt(hlz, 1)})
        rows.append({"id": mid, "section": sec, "metric": label, "kind": kind, "paper": pv, "paper_t": pt,
                     "us": uv, "us_t": ut, "match": match(kind, paper[mid], us[mid]),
                     "us_abs_t_gt_hlz": (None if kind != "t" or ut is None or not math.isfinite(ut)
                                         else bool(abs(ut) > hlz)),
                     "us_mde80": mde, "months_needed_t196": mn,
                     "why": WHY[mid].format_map(rowctx)})
    df = pd.DataFrame(rows)
    dvals = {r["id"]: r["value"] for r in D.rows}
    rob = build_robustness(rd, us, dvals)
    return df, rob, D.frame()


# --------------------------------------------------------------------------------------- writers

def fmt(x: float, t: float | None) -> str:
    if x is None or not math.isfinite(x):
        return "n/a"
    d = 2 if abs(x) >= 10 else 3
    s = f"{x:.{d}f}"
    return s if t is None or not math.isfinite(t) else f"{s} (t {t:.2f})"


def md_cell(s: str) -> str:
    return str(s).replace("|", "\\|")


def tex_num(x: float | None, d: int) -> str:
    return "n/a" if x is None or not math.isfinite(x) else f"{x:.{d}f}"


def sample_label(rd: Path) -> str:
    s = rd / "sample.csv"
    if not s.exists():
        return "Binance"
    r = pd.read_csv(s).iloc[0]
    return f"Binance {r['first']}–{r['last']}, {int(r['n_months'])} mo."


def key_findings(df: pd.DataFrame, rob: pd.DataFrame, diag: pd.DataFrame) -> list[str]:
    """Bullet points whose numbers and qualitative words ("above"/"below", "yes"/"no") come from the tables."""
    c = SafeDict({r.id: r.label for r in diag.itertuples()})
    val = {r.id: r.value for r in diag.itertuples()}
    t = df[(df["kind"] == "t") & (df["section"] != "A&N")]
    an = df[(df["kind"] == "t") & (df["section"] == "A&N")]
    hlz = _hlz()
    n_sig = int((t["us_t"].abs() >= T_CRIT).sum())
    n_hlz = int((t["us_t"].abs() > hlz).sum())
    counts = t["match"].value_counts()
    r = df.set_index("id")
    f = lambda i: fmt(r.loc[i, "us"], r.loc[i, "us_t"])  # noqa: E731
    sig = lambda i: abs(r.loc[i, "us_t"]) >= T_CRIT  # noqa: E731
    same = int(rob["same_conclusion"].eq("yes").sum()) if len(rob) else 0
    res = rob[rob["statistic"].isin(["FM Res.lnA (3)", "FM Res.lnA (4)"])] if len(rob) else rob
    res_lost = sorted({f"{x.statistic}: {x.variant}" for x in res.itertuples() if x.same_conclusion == "no"})
    crit = val.get("critFmThree", math.nan)
    t3 = abs(r.loc["FMResLnA", "us_t"])
    lines = [
        f"- **Scorecard.** Of {len(t)} Lou & Shu coefficient rows, {int(counts.get('yes', 0))} match, "
        f"{int(counts.get('sign only', 0))} match in sign only and {int(counts.get('no', 0))} are 'no' (opposite sign); "
        f"{n_sig} of these t-statistics have |t| ≥ 1.96 and {n_hlz} have |t| > {hlz:g} (HLZ bar). Several rows "
        "are overlapping specifications of the same few signals, so they are not independent tests. "
        f"Against Amihud & Noh (2021), {int(an['match'].eq('no').sum())} of {len(an)} rows have the opposite "
        "sign (A&N col. 4 is the same regression as our model (5)). "
        f"{same} of {len(rob)} robustness variants keep the base conclusion (`robustness_vs_base.csv`).",
        f"- **Volume component (A_C).** Sorts give {f('HLACrawCoef')} and FM gives {f('FMlnAC')}; both have the "
        f"paper's sign, and they are {'significant' if sig('HLACrawCoef') or sig('FMlnAC') else 'not significant'}. "
        f"The most influential month for the sort is {c['topMonthAC']} (H-L {c['topValAC']}%, CMKT "
        f"{c['cmktTopAC']}%); without it the spread is {c['exMeanAC']} (t {c['exTAC']}). The FM slope is "
        f"{c['subAFm']} (t {c['subAFmT']}) in {c['subARange']} and {c['subBFm']} (t {c['subBFmT']}) in "
        f"{c['subBRange']}.",
        f"- **Price-impact residual (Res. ln A).** FM model (3) gives {f('FMResLnA')}, the paper's sign. "
        f"|t| is {'above' if t3 > crit else 'not above'} the world-0 5% critical value ({c['critFmThree']}) "
        f"and {'above' if t3 > hlz else 'below'} the HLZ bar of {hlz:g}. Across the robustness variants of models "
        f"(3) and (4), {int(res['same_conclusion'].eq('yes').sum())} of {len(res)} keep the conclusion"
        + (f"; it changes in: {'; '.join(res_lost)}." if res_lost else ".")
        + f" Hypothesis: this is a volatility effect, since corr(Res. A, |Ret|) = {c['corrResAAbs']} and "
        f"ln|Ret| is {f('TtwolnAbs')} in the T2 decomposition.",
        f"- **Power.** Monthly coin returns have s.d. {c['retSd']}%, and the standard error of the A_C spread is "
        f"{c['seRatioHlAC']} times its value in the world-0 simulation of similar size ({c['simMonths']} months, "
        f"{c['simCoins']} coins). The 80%-power MDE for "
        f"the A_C spread is {ffmt(r.loc['HLACrawCoef', 'us_mde80'])}%/mo against the paper's "
        f"{ffmt(r.loc['HLACrawCoef', 'paper'])}, so our insignificant spreads cannot reject a stock-sized premium.",
        f"- **Different market.** Across coins volatility and dollar volume move together (corr(ln DVol, "
        f"ln|Ret|) = {c['corrVolAbs']}, slope {c['elasMean']}), and corr(A, |Ret|) is "
        f"{fmt(r.loc['corrAabs', 'us'], None)}, against {fmt(r.loc['corrAabs', 'paper'], None)} for stocks.",
    ]
    return lines


def write_md(df: pd.DataFrame, rob: pd.DataFrame, diag: pd.DataFrame, rd: Path, man: dict) -> None:
    us_col = f"Us ({sample_label(rd)})"
    c = SafeDict({r.id: r.label for r in diag.itertuples()})
    val = {r.id: r.value for r in diag.itertuples()}
    hlz = _hlz()
    lines = [f"# Lou & Shu (2017) vs. us — run `{man.get('run_id', rd.name)}`", "",
             "_Generated by `src/report.py` from the CSVs of this run and `results/simulations/`. "
             "Do not edit by hand._", "",
             f"Data: `{Path(str(man.get('data_source', '?'))).name}` (sha256 `{str(man.get('data_sha256', ''))[:12]}`), "
             f"git `{str((man.get('git') or {}).get('sha'))[:10]}`"
             f"{' (dirty)' if (man.get('git') or {}).get('dirty') else ''}. "
             f"Paper: NYSE/AMEX 1964–2012, {c['paperMonths']} months, {c['paperAvgObs']} firms per FM month; "
             f"A&N (2021): CRSP 1955–2016. Us: {c['nMonths']} months, {c['avgCoins']} coins per month "
             f"({c['fmAvgObs']} in FM, {c['fmMonths']} FM months). Data audit: {c['auditRedenom']} redenominations "
             f"split-adjusted, {c['auditReuse']} ticker reuses and {c['auditGap']} long gaps split into new assets "
             "(`audit_events.csv`).", "",
             "## Key findings", ""] + key_findings(df, rob, diag) + [
             "", "## Paper vs us", "",
             f"t-statistics: Newey–West, 6 lags. Columns: *Match?* follows the rule below; *|t|>{hlz:g}* is our "
             "statistic against the Harvey–Liu–Zhu bar; *MDE80* is the effect we could detect with 80% power "
             "(two-sided 5%), (1.96+0.84)·SE, in the row's units.", "",
             f"| Metric | Paper | {us_col} | Match? | \\|t\\|>{hlz:g} | MDE80 | Why it differs / matches |",
             "|---|---|---|---|---|---|---|"]
    for sec, title in SECTIONS.items():
        sub = df[df["section"] == sec]
        if not len(sub):
            continue
        lines.append(f"| **{md_cell(title)}** | | | | | | |")
        for r in sub.itertuples():
            h = "" if r.us_abs_t_gt_hlz is None or (isinstance(r.us_abs_t_gt_hlz, float)
                                                    and not math.isfinite(r.us_abs_t_gt_hlz)) \
                else ("yes" if r.us_abs_t_gt_hlz else "no")
            cells = [r.metric, fmt(r.paper, r.paper_t), fmt(r.us, r.us_t), r.match, h,
                     ffmt(r.us_mde80) if r.kind == "t" else "", r.why]
            lines.append("| " + " | ".join(md_cell(x) for x in cells) + " |")
    lines += ["", "**Match rule.** Coefficients and spreads: if the paper is significant (|t| ≥ 1.96), *yes* = "
              "same sign and |t| ≥ 1.96, *sign only* = same sign but not significant, *no* = opposite sign. If the "
              "paper is not significant, *yes* = we are not significant either, *sign only* / *no* = we are "
              "significant with the same / opposite sign. Levels (correlations, Q75/Q25): *yes* = same sign and "
              "within half an order of magnitude (|log10(us/paper)| ≤ 0.5), *sign only* = same sign otherwise; "
              "Q75/Q25 ratios are positive by construction, so they get *yes* or *no* on magnitude alone. "
              "*Months needed* (in the A_C sort row) assumes the SE falls as 1/sqrt(T): an estimate. "
              "Units: H-L and alphas in %/month; FM slopes in %/month per unit of the regressor (paper: "
              "FF3-adjusted; us: CMKT-adjusted, D4). Deviations D1–D13 and journal entries J1–J2: `DEVIATIONS.md`.",
              "", "## Robustness of our results (base vs variant)", "",
              "*Same conclusion* = same sign and same side of |t| = 1.96 as the base estimate.", "",
              "| Statistic | Variant | Base | Variant estimate | Same conclusion? |", "|---|---|---|---|---|"]
    for r in rob.itertuples():
        lines.append("| " + " | ".join(md_cell(x) for x in [r.statistic, r.variant, fmt(r.base, r.base_t),
                                                              fmt(r.variant_coef, r.variant_t),
                                                              r.same_conclusion]) + " |")
    lines += ["", "## Size and power of the tests (known-answer simulations, `results/simulations/`)", "",
              f"- **World 0, no premium** ({c['simReps']} runs, {c['simAbsret']}% mean daily |r| vs {c['absretMean']}% in "
              f"our data): with NW(6) the share of runs with |t| > 1.96 is {c['fprFmTwo']}% for FM ln A_C, "
              f"{c['fprFmThree']}% for FM Res. ln A (95% CI {c['fprFmThreeLo']}–{c['fprFmThreeHi']}) and "
              f"{c['fprHlAC']}% for H-L A_C. Without the NW correction (lag 0), Res. ln A gives "
              f"{c['fprFmThreeNWzero']}%. Empirical 5% critical |t|: FM ln A_C {c['critFmTwo']}, FM Res. ln A "
              f"{c['critFmThree']}, H-L A_C {c['critHlAC']}, H-L A {c['critHlA']}. "
              + ("With about 90 months NW(6) over-rejects for Res. ln A (the 95% CI excludes 5%). "
                 if val.get("fprFmThreeLo", math.nan) > 5 else "The size of the Res. ln A test is close to 5%. ")
              + f"The share with |t| > 3 is {c['fprThreeFmThree']}% (Res. ln A).",
              f"- **World V, power** ({c['powReps']} runs per level; t > 1.96, FM ln A_C / H-L A_C) at a premium of 0.25, 0.5 and 1%/mo per "
              f"s.d. of the priced state: {c['powFmTwoQuarter']}/{c['powHlACQuarter']}%, "
              f"{c['powFmTwoHalf']}/{c['powHlACHalf']}%, {c['powFmTwoOne']}/{c['powHlACOne']}%. These overstate "
              "power on real data. The sampling s.d. of the H-L A_C spread across world-0 runs is "
              f"{c['simSeHlAC']}%/mo, while our real NW s.e. is {c['realSeHlAC']}%/mo ({c['seRatioHlAC']}x). Real "
              f"returns are noisier and have fat tails (s.d. {c['retSd']}%/mo, p99 {c['retP99']}%, max "
              f"{c['retMax']}%). The MDE80 column therefore uses the real standard errors. Fuller analysis (block "
              "bootstrap, leave-one-year-out, multiple testing): `results/upgrades/robustness/`.",
              f"- **Placebos** (`placebo.csv`): shuffling signals within month, at most {c['placeboShareMax']}% of "
              f"{c['placeboReps']} runs give |t| > 1.96 for any statistic. Using the month-t signal instead (lag 0, "
              f"a deliberate leak) gives t {c['leakTHLAC']} for H-L A_C and {c['leakTFmTwo']} for FM ln A_C, so "
              "a timing error in the signal lag would visibly change the results. January-only ln A_C slope: "
              f"{c['janCoef']} (t {c['janT']}).", "",
              "## Diagnostics used above", "",
              "All values in `report_diagnostics.csv` (id, value, description, source).", ""]
    (rd / "paper_vs_us.md").write_text("\n".join(lines))


def write_tex(df: pd.DataFrame, diag: pd.DataFrame, rd: Path) -> None:
    """\\newcommand macros: \\us<Id>, \\us<Id>T, \\paper<Id>, \\paper<Id>T, \\match<Id>, \\diag<Id>."""
    out = ["% generated by src/report.py — do not edit by hand"]
    for r in df.itertuples():
        mid = r.id
        d = 2 if r.kind == "t" or abs(r.paper) >= 10 else 3
        out.append(f"\\newcommand{{\\us{mid}}}{{{tex_num(r.us, d)}}}")
        out.append(f"\\newcommand{{\\paper{mid}}}{{{tex_num(r.paper, d)}}}")
        if r.kind == "t":
            out.append(f"\\newcommand{{\\us{mid}T}}{{{tex_num(r.us_t, 2)}}}")
            out.append(f"\\newcommand{{\\paper{mid}T}}{{{tex_num(r.paper_t, 2)}}}")
            out.append(f"\\newcommand{{\\mde{mid}}}{{{tex_num(r.us_mde80, 2)}}}")
        out.append(f"\\newcommand{{\\match{mid}}}{{{r.match}}}")
    for r in diag.itertuples():
        if r.id.isalpha():
            out.append(f"\\newcommand{{\\diag{r.id[0].upper() + r.id[1:]}}}{{{r.label}}}")
    s = rd / "sample.csv"
    if s.exists():
        r = pd.read_csv(s).iloc[0]
        out += [f"\\newcommand{{\\usNMonths}}{{{int(r['n_months'])}}}",
                f"\\newcommand{{\\usNCoins}}{{{int(r['n_coins'])}}}",
                f"\\newcommand{{\\usAvgCoins}}{{{r['avg_coins_per_month']:.0f}}}",
                f"\\newcommand{{\\usFirstMonth}}{{{r['first']}}}",
                f"\\newcommand{{\\usLastMonth}}{{{r['last']}}}"]
    (rd / "tex").mkdir(exist_ok=True)
    (rd / "tex" / "numbers.tex").write_text("\n".join(out) + "\n")


def _sha(p: Path) -> str | None:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def write_manifest(rd: Path, sim_dir: Path, man: dict, timings: dict) -> None:
    import subprocess
    try:
        sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True,
                                    text=True).stdout.strip())
    except Exception:
        sha, dirty = None, None
    here = Path(__file__).resolve().parent
    inputs = sorted(p for p in rd.glob("*.csv") if p.name not in {"paper_vs_us.csv", "robustness_vs_base.csv",
                                                                 "report_diagnostics.csv"})
    sims = sorted(sim_dir.glob("*.csv")) if sim_dir.exists() else []
    panel = man.get("panel_cache")
    m = {"stage": "report", "base_run_id": man.get("run_id", rd.name), "created_utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
         "git": {"sha": sha, "dirty": dirty},
         "code_sha256": {f: _sha(here / f) for f in ["report.py", "figures.py", "paper_numbers.yaml",
                                                      "fama_macbeth.py", "measures.py"]},
         "inputs_sha256": {p.name: _sha(p) for p in inputs},
         "simulations_sha256": {p.name: _sha(p) for p in sims},
         "panel_cache": panel, "panel_cache_sha256": _sha(Path(panel)) if panel else None,
         "config_sha256": _sha(ROOT / "config.yaml"), "seed": man.get("seed"),
         "timings_sec": timings}
    (rd / "report_manifest.json").write_text(json.dumps(m, indent=2))


def resolve_run(results_dir: Path, run_id: str | None) -> Path:
    run_id = run_id or (results_dir / "LATEST").read_text().strip()
    return results_dir / run_id


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    ap.add_argument("--run-id", default=None)
    a = ap.parse_args(argv)
    t0 = time.time()
    rd = resolve_run(Path(a.results_dir), a.run_id)
    sim_dir = Path(a.results_dir) / "simulations"
    mf = rd / "run_manifest.json"
    man = json.loads(mf.read_text()) if mf.exists() else {}
    df, rob, diag = build(rd, sim_dir, man)
    t1 = time.time()
    df.to_csv(rd / "paper_vs_us.csv", index=False)
    rob.to_csv(rd / "robustness_vs_base.csv", index=False)
    diag.to_csv(rd / "report_diagnostics.csv", index=False)
    write_md(df, rob, diag, rd, man)
    write_tex(df, diag, rd)
    write_manifest(rd, sim_dir, man, {"build": round(t1 - t0, 3), "write": round(time.time() - t1, 3)})
    print(df[["metric", "paper", "us", "us_t", "match"]].to_string(index=False))
    return rd


if __name__ == "__main__":
    main()
