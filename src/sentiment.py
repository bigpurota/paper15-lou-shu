"""Upgrades 4-5: mispricing vs liquidity tests of the volume premium (Lou & Shu 2017, Section 4.1.2/4.2.1).

Table 13 analogue (sentiment):
    R_{i,t} = a + b * Sent_{t-1} + c * CMKT_t [+ d * CMOM_t] + u_t,   NW(6)
    Sent = Crypto Fear & Greed index (alternative.me, daily since 2018-02); monthly value of month t-1 =
    average of the daily values (main) or the last daily value of the month (robustness).
    Paper (AT_C, FF4): short-leg b < 0 (-0.25, t -2.67), long b ~ 0, long-short b > 0 (0.24, t 1.81).
Table 12 analogue (market illiquidity):
    Illiq = residual of an AR(2) on the monthly EW cross-sectional mean of A (Section 5 construction,
    without the x(-1), so that high values = illiquidity rose). Robustness: AR(2) on ln(EW mean A);
    the log level itself (not stationary in crypto, listed for completeness).
    A liquidity story predicts a LARGER volume premium after high illiquidity (b > 0 for long-short);
    the paper (Pastor-Stambaugh measure) finds b = -0.048 (t -2.38).
Portfolios: EW quintiles of A_C measured at t-2 (Long = Q5 = highest A_C = lowest volume; Short = Q1),
raw returns in % (r_f = 0, D5). Also the FM slope series on ln A_C (base T4 model (2)), recomputed here.
State variables are z-scored over the return months used, so b is in %/month per 1 s.d.
(the Baker-Wurgler index used in the paper is also roughly unit-variance); b_raw = b / sd.
All variants are pre-declared in VARIANTS below and all are reported.

Usage:  python -I src/sentiment.py config.yaml [--results-dir results] [--out results/upgrades/sentiment]
Reads:  results/LATEST (base run id), results/<base>/run_manifest.json, factors.csv, T3_portfolios.csv,
        T4_fm.csv, T7_january.csv; the base panel cache; the daily data used by the base run;
        data/raw/fng/fng.json (downloaded once if missing).
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
import statsmodels.api as sm
import yaml

from fama_macbeth import fama_macbeth, nw_mean
from measures import monthly_measures, winsorize_cs
from sorts import portfolio_returns

ROOT = Path(__file__).resolve().parents[1]
FNG_URL = "https://api.alternative.me/fng/?limit=0&format=json"
FNG_PATH = ROOT / "data" / "raw" / "fng" / "fng.json"
FNG_MIN_DAYS = 20             # a month's Sent needs >= 20 daily values
ILLIQ_MIN_COINS = 30          # AR(2) estimation starts at the first month with >= 30 eligible coins
PORTS = ["Short", "2", "3", "4", "Long", "Long-Short", "FM_lnA_C"]
log = logging.getLogger("sentiment")

# Pre-declared variants (all reported). Main rows are flagged main=True.
STATE_VARS = {
    "sent_mean": ("Fear&Greed, mean of month t-1", True),
    "sent_eom": ("Fear&Greed, last day of month t-1", False),
    "illiq_ar2": ("AR(2) residual of EW mean A, month t-1", True),
    "illiq_ar2_log": ("AR(2) residual of ln EW mean A, month t-1", False),
    "illiq_level_log": ("ln EW mean A, month t-1 (level; trending)", False),
}
CONTROLS = {"CMKT+CMOM": ["CMKT", "CMOM"], "CMKT": ["CMKT"],
            "CMKT+CMOM+CMKT(t-1)": ["CMKT", "CMOM", "CMKT_l1"]}
MAIN_CONTROLS = "CMKT+CMOM"
# Added at verification (post hoc, NOT pre-declared): both states correlate with the market return of their own
# month t-1, so a lagged-market control separates the state effect from plain market reversal. Reported, never main.
POST_HOC_CONTROLS = {"CMKT+CMOM+CMKT(t-1)"}

# Paper, AT_C sorts (Tables 12-13). Panel B: b and t; Panel A: High-Low difference of Long-Short.
PAPER = {
    ("sentiment", "Short"): (-0.25, -2.67), ("sentiment", "Long"): (-0.01, -0.10),
    ("sentiment", "Long-Short"): (0.24, 1.81), ("sentiment", "L-S High-Low (Panel A)"): (0.36, 1.08),
    ("illiquidity", "Short"): (0.031, 2.18), ("illiquidity", "Long"): (-0.016, -1.11),
    ("illiquidity", "Long-Short"): (-0.048, -2.38), ("illiquidity", "L-S High-Low (Panel A)"): (-0.02, -0.08),
}


# ---------------------------------------------------------------- Fear & Greed
def download_fng(path: Path = FNG_PATH, url: str = FNG_URL) -> Path:
    """Download the full F&G history once (idempotent: skipped when the file exists)."""
    if path.exists() and path.stat().st_size > 0:
        return path
    import requests
    path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    tmp = path.with_suffix(".part")
    tmp.write_bytes(r.content)
    tmp.replace(path)
    return path


def parse_fng(raw: dict) -> pd.DataFrame:
    """Untrusted JSON -> daily frame (date, value). Validates types and the 0..100 range."""
    data = raw.get("data") if isinstance(raw, dict) else None
    if not isinstance(data, list) or not data:
        raise ValueError("F&G JSON: no 'data' list")
    rows = []
    for rec in data:
        ts, val = int(str(rec["timestamp"])), float(str(rec["value"]))
        if not (0.0 <= val <= 100.0):
            raise ValueError(f"F&G value out of range: {val}")
        rows.append((pd.Timestamp(ts, unit="s").normalize(), val))
    d = pd.DataFrame(rows, columns=["date", "value"])
    d = d.drop_duplicates("date", keep="first").sort_values("date").reset_index(drop=True)
    return d


def fng_monthly(daily: pd.DataFrame, min_days: int = FNG_MIN_DAYS) -> pd.DataFrame:
    """Monthly sent_mean (average), sent_eom (last value of the month), n_days; NaN if n_days < min_days."""
    d = daily.copy()
    d["month"] = d["date"].dt.to_period("M")
    g = d.sort_values("date").groupby("month")["value"]
    m = pd.DataFrame({"sent_mean": g.mean(), "sent_eom": g.last(), "n_days": g.size()})
    m.loc[m["n_days"] < min_days, ["sent_mean", "sent_eom"]] = np.nan
    return m


# ---------------------------------------------------------------- market illiquidity
def market_illiquidity(daily: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """EW cross-sectional mean of A per calendar month over the base run's eligible universe
    (same filters and winsorization as analysis.build_panel), plus the number of coins."""
    mcfg, ucfg = cfg["measures"], cfg["universe"]
    monthly = monthly_measures(daily, scale=float(mcfg["scale_A"]))
    elig = (monthly["n_days"] >= ucfg["min_valid_days"]) & (monthly["dvol"] >= ucfg["min_mean_dvol_usd"])
    m = winsorize_cs(monthly.loc[elig, ["symbol", "month", "A"]], ["A"], "month", tuple(mcfg["winsor"]))
    g = m.groupby("month")["A"]
    out = pd.DataFrame({"n_coins": g.count(), "ew_A": g.mean()})
    out["ln_ew_A"] = np.log(out["ew_A"])
    return out


def ar2_residual(x: pd.Series) -> tuple[pd.Series, dict]:
    """OLS x_t = c + phi1 x_{t-1} + phi2 x_{t-2} + e_t on a contiguous monthly series; residual e_t."""
    x = x.astype(float)
    df = pd.DataFrame({"y": x, "l1": x.shift(1), "l2": x.shift(2)}).dropna()
    res = sm.OLS(df["y"], sm.add_constant(df[["l1", "l2"]])).fit()
    params = {"c": float(res.params["const"]), "phi1": float(res.params["l1"]),
              "phi2": float(res.params["l2"]), "n": int(res.nobs), "r2": float(res.rsquared),
              "first": str(df.index.min()), "last": str(df.index.max())}
    return res.resid.rename(x.name), params


def illiq_states(illiq: pd.DataFrame, min_coins: int = ILLIQ_MIN_COINS) -> tuple[pd.DataFrame, list[dict]]:
    """Calendar-month state series illiq_ar2, illiq_ar2_log, illiq_level_log (estimation from the first
    month with >= min_coins coins; the sample must be contiguous from there)."""
    ok = illiq.index[illiq["n_coins"] >= min_coins]
    if len(ok) == 0:
        raise ValueError("no month with enough coins for the illiquidity series")
    s = illiq.loc[ok.min():].copy()
    full = pd.period_range(s.index.min(), s.index.max(), freq="M")
    if len(full) != len(s):
        raise ValueError("illiquidity series has gaps")
    out = pd.DataFrame(index=s.index)
    fits = []
    for name, col in [("illiq_ar2", "ew_A"), ("illiq_ar2_log", "ln_ew_A")]:
        e, p = ar2_residual(s[col].rename(name))
        out[name] = e
        fits.append({"variant": name, "series": col} | p)
    out["illiq_level_log"] = s["ln_ew_A"]
    return out, fits


# ---------------------------------------------------------------- portfolios and regressions
def lag_state(state: pd.Series, months: pd.Index) -> pd.Series:
    """Value for return month t = state of calendar month t-1."""
    s = state.copy()
    s.index = s.index + 1
    return s.reindex(months)


def premium_series(panel: pd.DataFrame, q: int) -> pd.DataFrame:
    """Monthly Short..Long (EW quintiles of A_C, raw % returns), Long-Short, and FM ln A_C slope
    (CMKT-adjusted returns, controls mom, ret_1: base T4 model (2))."""
    pr = portfolio_returns(panel, "A_C", "ret", q)
    out = pd.DataFrame({"Short": pr[1], "2": pr[2], "3": pr[3], "4": pr[4], "Long": pr[q]})
    out["Long-Short"] = out["Long"] - out["Short"]
    fm = fama_macbeth(panel, "ret_adj", ["lnA_C", "mom", "ret_1"])
    out["FM_lnA_C"] = fm.attrs["coef_series"]["lnA_C"]
    out.attrs = {}
    return out


def state_regression(y: pd.Series, state: pd.Series, factors: pd.DataFrame | None, lags: int) -> dict:
    """y_t = a + b*state_t + factors_t + u_t, NW(lags). `state` is already lagged and aligned to t."""
    parts = [y.rename("y"), state.rename("s")] + ([factors] if factors is not None and factors.shape[1] else [])
    df = pd.concat(parts, axis=1, join="inner").dropna()
    X = sm.add_constant(df.drop(columns="y"))
    res = sm.OLS(df["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return {"b": float(res.params["s"]), "t": float(res.tvalues["s"]),
            "a": float(res.params["const"]), "t_a": float(res.tvalues["const"]),
            "n_months": int(res.nobs), "r2": float(res.rsquared)}


def regime_table(prem: pd.DataFrame, state: pd.Series, lags: int) -> pd.DataFrame:
    """Panel A: mean return per portfolio after high / low state (split at the median of the lagged state
    over the months used), High-Low difference with NW t (regression on a high dummy)."""
    st = state.reindex(prem.index)
    keep = st.notna()
    med = st[keep].median()
    high = (st > med).astype(float)[keep]
    rows = []
    for regime, mask in [("High", high == 1), ("Low", high == 0)]:
        row = {"regime": regime, "n_months": int(mask.sum()), "median": float(med)}
        for c in PORTS:
            x = prem.loc[mask.index[mask], c]
            m, _, t = nw_mean(x, lags)
            row[c] = m
            row[f"t_{c}"] = t
        rows.append(row)
    row = {"regime": "High-Low", "n_months": int(keep.sum()), "median": float(med)}
    for c in PORTS:
        r = state_regression(prem.loc[keep, c], high, None, lags)
        row[c], row[f"t_{c}"] = r["b"], r["t"]
    rows.append(row)
    return pd.DataFrame(rows)


def zscore(s: pd.Series) -> tuple[pd.Series, float, float]:
    mu, sd = float(s.mean()), float(s.std(ddof=1))
    return (s - mu) / sd, mu, sd


def add_lagged_market(factors: pd.DataFrame) -> pd.DataFrame:
    """CMKT_l1 = CMKT of calendar month t-1 (NaN when month t-1 is not in `factors`)."""
    f = factors.copy()
    full = f.reindex(pd.period_range(f.index.min(), f.index.max(), freq="M"))
    f["CMKT_l1"] = full["CMKT"].shift(1).reindex(f.index)
    return f


def run_state_tests(prem: pd.DataFrame, states: pd.DataFrame, factors: pd.DataFrame, lags: int,
                    hlz: float = 3.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Panel B (all state variables x control sets x portfolios) and Panel A (median splits)."""
    months = prem.index
    factors = add_lagged_market(factors)
    b_rows, a_rows = [], []
    for sv, (label, main_sv) in STATE_VARS.items():
        if sv not in states:
            continue
        lagged = lag_state(states[sv], months)
        common = lagged.dropna().index.intersection(prem.dropna(subset=["Long-Short"]).index)
        z, mu, sd = zscore(lagged.loc[common])
        for cname, cols in CONTROLS.items():
            for p in PORTS:
                r = state_regression(prem.loc[common, p], z, factors.loc[common, cols], lags)
                b_rows.append({"state": sv, "label": label, "controls": cname, "portfolio": p,
                               "main": bool(main_sv and cname == MAIN_CONTROLS),
                               "post_hoc": cname in POST_HOC_CONTROLS} | r |
                              {"sd_state": sd, "mean_state": mu, "b_raw_units": r["b"] / sd,
                               "sig_196": abs(r["t"]) > 1.96, "sig_hlz": abs(r["t"]) > hlz})
        pa = regime_table(prem.loc[common], lagged.loc[common], lags)
        pa.insert(0, "label", label)
        pa.insert(0, "state", sv)
        a_rows.append(pa)
    return pd.DataFrame(b_rows), pd.concat(a_rows, ignore_index=True)


def paper_vs_us(panel_b: pd.DataFrame, panel_a: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (test, port), (pb, pt) in PAPER.items():
        sv = "sent_mean" if test == "sentiment" else "illiq_ar2"
        if port.startswith("L-S High-Low"):
            r = panel_a[(panel_a["state"] == sv) & (panel_a["regime"] == "High-Low")].iloc[0]
            ours_b, ours_t, n = r["Long-Short"], r["t_Long-Short"], r["n_months"]
        else:
            r = panel_b[(panel_b["state"] == sv) & (panel_b["controls"] == MAIN_CONTROLS)
                        & (panel_b["portfolio"] == port)].iloc[0]
            ours_b, ours_t, n = r["b"], r["t"], r["n_months"]
        rows.append({"test": test, "row": port, "paper_b": pb, "paper_t": pt, "ours_b": ours_b, "ours_t": ours_t,
                     "n_months": n, "same_sign": bool(np.sign(pb) == np.sign(ours_b)),
                     "ours_sig_196": abs(ours_t) > 1.96})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- summary (generated from the CSVs)
def _f(x, d=2):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def verdict_sentiment(pb: pd.DataFrame) -> str:
    g = pb[(pb["state"] == "sent_mean") & (pb["controls"] == MAIN_CONTROLS)].set_index("portfolio")
    s, ls = g.loc["Short"], g.loc["Long-Short"]
    if s["b"] < 0 and ls["b"] > 0 and abs(s["t"]) > 1.96 and abs(ls["t"]) > 1.96:
        return "предсказание mispricing подтверждается: b(short) < 0 и b(long–short) > 0, оба значимы при |t| > 1.96"
    if s["b"] < 0 and ls["b"] > 0:
        return "знаки совпадают с предсказанием mispricing (b(short) < 0, b(long–short) > 0), но хотя бы один коэффициент незначим при 1.96"
    return "предсказание mispricing не подтверждается: знаки b(short) / b(long–short) не совпадают с ожидаемыми (< 0 / > 0)"


def verdict_illiq(pb: pd.DataFrame) -> str:
    ls = pb[(pb["state"] == "illiq_ar2") & (pb["controls"] == MAIN_CONTROLS)
            & (pb["portfolio"] == "Long-Short")].iloc[0]
    if ls["b"] > 0 and abs(ls["t"]) > 1.96:
        return "премия значимо больше после роста рыночной неликвидности (b > 0, |t| > 1.96): это в пользу объяснения через ликвидность"
    if ls["b"] < 0 and abs(ls["t"]) > 1.96:
        return "премия значимо меньше после роста неликвидности (b < 0): против объяснения через ликвидность, как в статье"
    return "значимой связи премии с рыночной неликвидностью нет (|t| ≤ 1.96): рост премии, который предсказывает объяснение через ликвидность, не найден"


def robustness_line(pb: pd.DataFrame, main_state: str, states: list[str]) -> str:
    """Generated robustness sentence for the Long-Short slope: range of t over variants, counts vs 1.96 / 3,
    and the post-hoc lagged-market control for the main state."""
    g = pb[(pb["portfolio"] == "Long-Short") & (pb["state"].isin(states))]
    pre = g[~g["post_hoc"]]
    ph = g[g["post_hoc"] & (g["state"] == main_state)]
    txt = (f"Устойчивость Long-Short ({', '.join(states)}): t от {_f(pre['t'].min())} до {_f(pre['t'].max())} "
           f"по {len(pre)} заранее объявленным вариантам; abs(t) > 1.96 в {int((pre['t'].abs() > 1.96).sum())}, "
           f"abs(t) > 3 в {int((pre['t'].abs() > 3).sum())}")
    if len(ph):
        r = ph.iloc[0]
        txt += (f"; с добавленным контролем CMKT(t−1) (post hoc) для {main_state}: b = {_f(r['b'])} (t {_f(r['t'])}, "
                f"{int(r['n_months'])} мес.)")
    return txt + "."


def write_summary(out: Path, base_id: str, pvu: pd.DataFrame, pb: pd.DataFrame, pa: pd.DataFrame,
                  jan: pd.DataFrame | None, fng_m: pd.DataFrame, ar2: pd.DataFrame,
                  data_match: bool = True) -> None:
    L = [f"# Апгрейды 4–5: сентимент, рыночная неликвидность, январь", "",
         f"Базовый прогон: `{base_id}`. Все числа ниже сгенерированы `src/sentiment.py` из CSV в этой папке.", ""]
    if not data_match:
        L += ["**SMOKE-ПРОГОН: дневные данные изменились после базового прогона; числа недействительны, не интерпретировать.**", ""]
    L += [
         "## Что проверяли",
         "- Аналог Table 13: доходности квинтилей по `A_C` (t−2) и long–short регрессируются на Sent_{t−1} "
         "(Crypto Fear & Greed, среднее за месяц; z-оценка) с контролями CMKT + CMOM, NW(6).",
         "- Аналог Table 12: Illiq_{t−1} = остаток AR(2) по EW-среднему `A`; та же регрессия; плюс разбиение по медиане.",
         "- Ряд FM-наклонов на ln A_C (модель (2) из T4) — на тех же переменных состояния.", "",
         "## Статья против нас (основная спецификация)", "",
         "Наши b — в %/мес. на 1 s.d. переменной состояния; b статьи — на единицу её индекса (Baker–Wurgler ≈ 1 s.d.; "
         "уровень Pástor–Stambaugh — не 1 s.d.), поэтому сравнимы знаки и t, а не величины.", "",
         "| Тест | Строка | Статья b (t) | Мы b (t) | Месяцев | Знак совпал | Мы: abs(t) > 1.96 / > 3 (HLZ) |",
         "|---|---|---|---|---|---|---|"]
    yn = lambda x: "да" if x else "нет"
    for _, r in pvu.iterrows():
        L.append(f"| {r['test']} | {r['row']} | {_f(r['paper_b'], 3)} ({_f(r['paper_t'])}) | "
                 f"{_f(r['ours_b'], 3)} ({_f(r['ours_t'])}) | {int(r['n_months'])} | {yn(r['same_sign'])} | "
                 f"{yn(abs(r['ours_t']) > 1.96)} / {yn(abs(r['ours_t']) > 3.0)} |")
    fm = pb[(pb["main"]) & (pb["portfolio"] == "FM_lnA_C")]
    L += ["", "FM-наклон ln A_C на переменной состояния (CMKT + CMOM): " +
          "; ".join(f"{r['state']}: b = {_f(r['b'], 3)} (t {_f(r['t'])})" for _, r in fm.iterrows()), "",
          "## Все варианты (Panel B, b (t), месяцев); строки с * добавлены при проверке (post hoc)", "",
          "| Состояние | Контроли | Short | Long | Long-Short | FM ln A_C | Месяцев (L-S) |", "|---|---|---|---|---|---|---|"]
    for (sv, cn), g in pb.groupby(["state", "controls"], sort=False):
        g = g.set_index("portfolio")
        cell = lambda port: f"{_f(g.loc[port, 'b'], 2)} ({_f(g.loc[port, 't'])})"
        L.append(f"| {sv} | {cn}{' *' if g['post_hoc'].iloc[0] else ''}{' (осн.)' if g['main'].iloc[0] else ''} | "
                 f"{cell('Short')} | {cell('Long')} | {cell('Long-Short')} | {cell('FM_lnA_C')} | "
                 f"{int(g.loc['Long-Short', 'n_months'])} |")
    hl = pa[pa["regime"] == "High-Low"]
    L += ["", "Panel A (разбиение по медиане), Long-Short High−Low: " +
          "; ".join(f"{r['state']}: {_f(r['Long-Short'])} (t {_f(r['t_Long-Short'])})" for _, r in hl.iterrows()), ""]
    L += ["## Интерпретация (правила в коде, по t-статистикам)",
          f"- Сентимент: {verdict_sentiment(pb)}.",
          f"- Неликвидность: {verdict_illiq(pb)}.",
          f"- {robustness_line(pb, 'sent_mean', ['sent_mean', 'sent_eom'])}",
          f"- {robustness_line(pb, 'illiq_ar2', ['illiq_ar2', 'illiq_ar2_log', 'illiq_level_log'])}", ""]
    if jan is not None:
        j = jan.set_index("subset")
        L += ["## Январь (T7 базового прогона) — только плацебо",
              f"- Наклон ln A_C в январе: {_f(j.loc['January', 'coef'])} (t {_f(j.loc['January', 't'])}, "
              f"n = {int(j.loc['January', 'n'])}); февраль–декабрь: {_f(j.loc['Feb-Dec', 'coef'])} "
              f"(t {_f(j.loc['Feb-Dec', 't'])}, n = {int(j.loc['Feb-Dec', 'n'])}).",
              "- В крипте нет налогового года и window dressing по январю, поэтому это плацебо, а не тест mispricing; "
              "при ~7 январях мощности нет.", ""]
    nvar = pb[["state", "controls", "post_hoc"]].drop_duplicates()
    L += ["## Оговорки",
          f"- Перебрано {len(nvar)} пар (переменная состояния × контроли; из них {int(nvar['post_hoc'].sum())} "
          "добавлены при проверке — контроль CMKT(t−1)), все в `panel_b.csv` / `panel_a.csv`; "
          "основная спецификация объявлена заранее (`STATE_VARS`, `MAIN_CONTROLS` в коде).",
          f"- Fear & Greed: месяцев с данными {int(fng_m['sent_mean'].notna().sum())}; индекс строится по цене и объёму BTC "
          "(волатильность, моментум, соцсети), т.е. он частично механически связан с рынком — отсюда контроль CMKT.",
          f"- AR(2) по EW `A` оценён на {int(ar2.iloc[0]['n'])} месяцах остатков ({ar2.iloc[0]['first']}–{ar2.iloc[0]['last']}; "
          f"ряд начинается с первого месяца с ≥ {ILLIQ_MIN_COINS} монетами, первые 2 месяца уходят на лаги), φ1 = {_f(ar2.iloc[0]['phi1'])}, φ2 = {_f(ar2.iloc[0]['phi2'])}.",
          f"- AR(2) оценён на всей выборке (как у Pástor–Stambaugh и в разделе 5 статьи) — look-ahead в коэффициентах; "
          f"в статье в Table 12 — уровень меры Pástor–Stambaugh, а не наша инновация EW `A`.",
          "- Статья сортирует по `AT_C` (оборачиваемость), мы — по `A_C` (D2); FF4 заменён на CMKT + CMOM (D6).",
          f"- Порог HLZ: |t| > 3 отмечен в `panel_b.csv` (столбец `sig_hlz`).", ""]
    (out / "SUMMARY.md").write_text("\n".join(L))


# ---------------------------------------------------------------- entry point
def _csv(df: pd.DataFrame, path: Path) -> None:
    df = df.copy()
    df.attrs = {}
    if not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index()
    for c in df.columns:
        if isinstance(df[c].dtype, pd.PeriodDtype):
            df[c] = df[c].astype(str)
    df.to_csv(path, index=False)


def _to_period(df: pd.DataFrame, col: str = "month") -> pd.DataFrame:
    df[col] = pd.PeriodIndex(df[col].astype(str), freq="M")
    return df


def main(argv: list[str] | None = None) -> None:
    import run as base   # provenance helpers (sha256_file, git_info, versions, load_daily)
    import data_audit

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    ap.add_argument("--out", default=str(ROOT / "results" / "upgrades" / "sentiment"))
    ap.add_argument("--fng", default=str(FNG_PATH))
    ap.add_argument("--allow-data-mismatch", action="store_true",
                    help="smoke runs only: proceed when the daily data changed after the base run (flagged in manifest)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    timings: dict[str, float] = {}
    t0 = time.perf_counter()

    def lap(name):
        nonlocal t0
        timings[name] = round(time.perf_counter() - t0, 3)
        t0 = time.perf_counter()

    cfg_path = Path(a.config)
    cfg = yaml.safe_load(cfg_path.read_text())
    lags, q, hlz = cfg["tests"]["nw_lags"], cfg["tests"]["quantiles"], cfg["tests"]["hlz_threshold"]
    res_dir = Path(a.results_dir)
    base_id = (res_dir / "LATEST").read_text().strip()
    bdir = res_dir / base_id
    bman = json.loads((bdir / "run_manifest.json").read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log.info("base run %s", base_id)

    fng_path = download_fng(Path(a.fng))
    fng_d = parse_fng(json.loads(fng_path.read_text()))
    fng_m = fng_monthly(fng_d)
    lap("fng")

    daily_path = Path(bman["data_source"])
    daily_hash = base.sha256_file(daily_path)
    data_match = daily_hash == bman["data_sha256"]
    if not data_match and a.allow_data_mismatch:
        log.warning("daily data changed since base run %s: SMOKE ONLY, numbers not valid", base_id)
    elif not data_match:
        raise RuntimeError(f"daily data {daily_path} changed since base run {base_id} (sha mismatch)")
    daily = base.load_daily(daily_path)
    audit_on = bool(data_audit.audit_config(cfg)["enabled"])
    if audit_on:                       # same cleaning as the base run before build_panel
        daily = data_audit.audit(daily, cfg)[0]
    illiq = market_illiquidity(daily, cfg)
    ill_states, ar2_fits = illiq_states(illiq)
    lap("illiquidity")

    panel_path = Path(bman["panel_cache"])
    panel = _to_period(pd.read_parquet(panel_path))
    prem = premium_series(panel, q)
    factors = _to_period(pd.read_csv(bdir / "factors.csv")).set_index("month")[["CMKT", "CMOM"]]
    # cross-checks against the base run
    t3 = pd.read_csv(bdir / "T3_portfolios.csv")
    t3 = _to_period(t3[t3["signal"] == "A_C"]).set_index("month")
    chk_hl = float((t3["HL"] - prem["Long-Short"]).abs().max())
    t4 = pd.read_csv(bdir / "T4_fm.csv")
    t4c = float(t4[(t4["model"] == "(2)") & (t4["var"] == "lnA_C")]["coef"].iloc[0])
    chk_fm = abs(t4c - float(prem["FM_lnA_C"].mean()))
    if chk_hl > 1e-8 or chk_fm > 1e-8:
        raise RuntimeError(f"premium series do not match the base run (HL {chk_hl}, FM {chk_fm})")
    lap("premium")

    states = pd.concat([fng_m[["sent_mean", "sent_eom"]], ill_states], axis=1).sort_index()
    pb, pa = run_state_tests(prem, states, factors, lags, hlz)
    pvu = paper_vs_us(pb, pa)
    jan = pd.read_csv(bdir / "T7_january.csv") if (bdir / "T7_january.csv").exists() else None
    lap("tests")

    _csv(fng_m.rename_axis("month"), out / "fng_monthly.csv")
    _csv(illiq.join(ill_states, how="left").rename_axis("month"), out / "illiq_monthly.csv")
    _csv(pd.DataFrame(ar2_fits), out / "ar2_fit.csv")
    _csv(prem.rename_axis("month"), out / "premium_series.csv")
    _csv(states.rename_axis("month"), out / "states.csv")
    _csv(pb, out / "panel_b.csv")
    _csv(pa, out / "panel_a.csv")
    _csv(pvu, out / "paper_vs_us.csv")
    if jan is not None:
        _csv(jan.assign(note="placebo: no tax-year motive in crypto"), out / "january_placebo.csv")
    write_summary(out, base_id, pvu, pb, pa, jan, fng_m, pd.DataFrame(ar2_fits), data_match)
    lap("write")

    code_files = ["sentiment.py", "run.py", "measures.py", "sorts.py", "fama_macbeth.py", "data_audit.py"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = {
        "run_id": f"sentiment_{stamp}",
        "base_run_id": base_id,
        "created_utc": stamp,
        "git": base.git_info(),
        "config_path": str(cfg_path.resolve()),
        "config_sha256": base.sha256_file(cfg_path),
        "code_sha256": {f: base.sha256_file(ROOT / "src" / f) for f in code_files},
        "data": {"fng_json": str(fng_path.resolve()), "fng_sha256": base.sha256_file(fng_path),
                 "fng_url": FNG_URL, "daily": str(daily_path), "daily_sha256": daily_hash,
                 "panel_cache": str(panel_path), "panel_sha256": base.sha256_file(panel_path)},
        "seed": int(cfg["seed"]),
        "params": {"nw_lags": lags, "quantiles": q, "fng_min_days": FNG_MIN_DAYS,
                   "illiq_min_coins": ILLIQ_MIN_COINS, "main_controls": MAIN_CONTROLS,
                   "post_hoc_controls": sorted(POST_HOC_CONTROLS),
                   "state_vars": {k: v[0] for k, v in STATE_VARS.items()}},
        "checks": {"daily_matches_base_run": data_match, "daily_audit_applied": audit_on, "max_abs_diff_HL_vs_T3": chk_hl, "abs_diff_FM_lnA_C_vs_T4": chk_fm},
        "versions": base.versions(),
        "timings_sec": timings | {"total": round(sum(timings.values()), 3)},
        "outputs": sorted(p.name for p in out.iterdir() if p.name != "run_manifest.json"),
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("written to %s", out)


if __name__ == "__main__":
    main()
