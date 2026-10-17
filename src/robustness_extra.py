"""Overfitting / inference upgrade ("robustness"): block-bootstrap CIs, specification ledger,
leave-one-year-out stability, and minimum detectable effects (MDE).

Usage:  python -I src/robustness_extra.py config.yaml [--base RUN_ID] [--out DIR]
                                                     [--draws 2000] [--block 6] [--ledger-only]
Reads:  results/LATEST (base run id, unless --base), results/<base>/*.csv,
        data/cache/panel_<base>.parquet (rebuilt from the daily cache if missing),
        results/simulations/sim_world0_reps.csv, sim_power.csv, results/upgrades/*/*.csv.
Writes: results/upgrades/robustness/{bootstrap_ci, stability_loyo, stability_excl2021,
        world0_calibration, mde, ledger, ledger_summary, consistency_checks}.csv,
        run_manifest.json, SUMMARY.md (English; every number is read from the CSVs above).

Conventions (as in the base run): returns in percent per month, NW lags from config,
FM slopes are time-series means of monthly cross-sectional slopes (so a month-level
bootstrap of the slope series is a bootstrap of the FM estimate), H-L = Q5 - Q1 EW raw.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import hashlib
import json
import logging
import subprocess
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm

from fama_macbeth import fama_macbeth, nw_mean
from sorts import alpha, portfolio_returns
from run import rel_path, resolve_panel_cache, resolve_repo_path

ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger("robustness")

N_DRAWS = 2000
BLOCK = 6
POWER_TARGET = 0.80
N_SEED_REPS = 20
PAPER_SD_LNA_C = 2.53
CONTROLS = ["mom", "ret_1"]
FM_STATS = [("fm1_lnA", "(1)", ["lnA"], "lnA"),
            ("fm2_lnA_C", "(2)", ["lnA_C"], "lnA_C"),
            ("fm3_ResLnA", "(3)", ["ResLnA"], "ResLnA"),
            ("fm4_lnA_C", "(4)", ["lnA_C", "ResLnA"], "lnA_C"),
            ("fm4_ResLnA", "(4)", ["lnA_C", "ResLnA"], "ResLnA")]
HL_SIGNALS = ["A", "A_C", "ResA"]
CONTROL_VARS = {"mom", "ret_1", "ivol", "beta", "const", "CMKT", "CMOM"}
ESTIMATE_COLS = ["coef", "H-L", "alpha_HL_A", "alpha", "estimate", "mean_coef", "premium", "slope", "b"]
LEDGER_SKIP = {"paper_vs_us.csv", "paper_reference.csv", "paper_numbers.csv", "robustness_vs_base.csv",
               "report_diagnostics.csv"}
LEDGER_SKIP_PREFIXES = ("ret_", "audit_")
NOT_T_PREFIXES = ("mean_", "max_", "min_", "median_", "dvol", "vol", "ret", "price", "close")
LOADING_T = {"t_mkt-rf", "t_smb", "t_hml", "t_umd", "t_rmw", "t_cma", "t_factor", "t_cmkt",
             "t_cmom", "t_imla_c", "t_a", "t_const", "t_intercept"}
LEG_NAMES = {"short", "long", "1", "2", "3", "4", "5", "q1", "q2", "q3", "q4", "q5"}
OWN_NON_TESTS = {f"robustness/{f}" for f in ["ledger.csv", "ledger_summary.csv", "bootstrap_ci.csv",
                 "consistency_checks.csv", "stability_loyo_summary.csv", "mde.csv", "mde_inputs.csv",
                 "world0_calibration.csv"]}


def mbb_indices(T: int, block: int, n_draws: int, rng: np.random.Generator) -> np.ndarray:
    """Moving-block bootstrap (Kunsch 1989): n_draws x T matrix of time indices.

    Each draw concatenates ceil(T/block) blocks of `block` consecutive indices whose start is
    uniform on 0..T-block (non-circular), truncated to length T.
    """
    if block < 1 or block > T:
        raise ValueError(f"block={block} must be in 1..T={T}")
    nb = -(-T // block)
    starts = rng.integers(0, T - block + 1, size=(n_draws, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_draws, nb * block)
    return idx[:, :T]


def boot_means(x: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Mean of x over each resampled index row, ignoring NaN (months where the stat is absent)."""
    v = x[idx]
    with np.errstate(invalid="ignore"):
        return np.nanmean(v, axis=1)


def boot_alphas(y: np.ndarray, F: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Intercept of OLS y ~ 1 + F on each resampled set of months (rows with NaN dropped)."""
    out = np.full(idx.shape[0], np.nan)
    for i, row in enumerate(idx):
        yy, FF = y[row], F[row]
        ok = np.isfinite(yy) & np.isfinite(FF).all(axis=1)
        Z = np.column_stack([np.ones(ok.sum()), FF[ok]])
        if ok.sum() > Z.shape[1]:
            out[i] = np.linalg.lstsq(Z, yy[ok], rcond=None)[0][0]
    return out


def ci_row(name: str, est: float, nw_t: float, draws: np.ndarray, n_months: int) -> dict:
    d = draws[np.isfinite(draws)]
    se = float(d.std(ddof=1))
    lo, hi = np.quantile(d, [0.025, 0.975])
    opp = float((d <= 0).mean()) if est > 0 else float((d >= 0).mean())
    return {"stat": name, "estimate": est, "t_nw": nw_t, "n_months": n_months,
            "boot_se": se, "t_boot": est / se if se > 0 else np.nan,
            "ci95_lo": float(lo), "ci95_hi": float(hi),
            "ci_excludes_0": bool(lo > 0 or hi < 0),
            "share_draws_opposite_sign": opp, "n_draws_valid": int(d.size)}


def monthly_series(panel: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Months x stats frame: H-L of A, A_C, ResA (EW raw, %), and FM monthly slopes; plus factors."""
    t = cfg["tests"]
    q, lags = int(t["quantiles"]), int(t["nw_lags"])
    cols = {}
    for s in HL_SIGNALS:
        pr = portfolio_returns(panel, s, "ret", q)
        cols[f"hl_{s}"] = pr[q] - pr[1]
    for name, _, xs, var in FM_STATS:
        res = fama_macbeth(panel, "ret_adj", xs + CONTROLS, lags=lags)
        cols[name] = res.attrs["coef_series"][var]
    S = pd.DataFrame(cols).sort_index()
    S.index = pd.PeriodIndex(S.index, freq="M")
    full = pd.period_range(S.index.min(), S.index.max(), freq="M")
    return S.reindex(full), _factors(panel, cfg).reindex(full)


def _factors(panel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """CMKT (%) and CMOM (%) exactly as in analysis.build_panel."""
    cmkt = (100.0 * panel.groupby("month")["CMKT"].first()).rename("CMKT")
    pr = portfolio_returns(panel, "mom", "ret", int(cfg["factors"]["factor_quantiles"]))
    cmom = (pr[pr.columns.max()] - pr[1]).rename("CMOM")
    f = pd.concat([cmkt, cmom], axis=1)
    f.index = pd.PeriodIndex(f.index, freq="M")
    return f


def _seed_stability(draw_fn, K: int) -> dict:
    """Monte Carlo stability of the percentile CI: rerun the bootstrap with K other seeds and
    report the range of the CI bounds and the share of seeds whose CI excludes 0."""
    los, his = [], []
    for d in draw_fn(K):
        d = d[np.isfinite(d)]
        lo, hi = np.quantile(d, [0.025, 0.975])
        los.append(lo)
        his.append(hi)
    los, his = np.array(los), np.array(his)
    return {"ci95_lo_seed_min": float(los.min()), "ci95_lo_seed_max": float(los.max()),
            "ci95_hi_seed_min": float(his.min()), "ci95_hi_seed_max": float(his.max()),
            "share_seeds_ci_excludes_0": float(((los > 0) | (his < 0)).mean()), "n_seed_reps": K}


def bootstrap_table(S: pd.DataFrame, F: pd.DataFrame, lags: int, block: int, n_draws: int,
                    seed: int, n_seed_reps: int = N_SEED_REPS) -> pd.DataFrame:
    """Joint MBB over calendar months (same draws for every stat): raw means and H-L alphas.

    The reported CI uses `seed`; `n_seed_reps` extra seeds (SeedSequence([seed, k])) measure how
    much the CI bounds move with Monte Carlo noise, so a bound near 0 is not read as a finding."""
    rng = np.random.default_rng(seed)
    idx = mbb_indices(len(S), block, n_draws, rng)
    idx_k = [mbb_indices(len(S), block, n_draws, np.random.default_rng([seed, k]))
             for k in range(1, n_seed_reps + 1)]
    Fm = F[["CMKT", "CMOM"]].to_numpy(float)
    rows = []
    for c in S.columns:
        x = S[c].to_numpy(float)
        m, _, tv = nw_mean(S[c].dropna(), lags)
        r = ci_row(c, m, tv, boot_means(x, idx), int(np.isfinite(x).sum()))
        if n_seed_reps:
            r |= _seed_stability(lambda K: (boot_means(x, ii) for ii in idx_k), n_seed_reps)
        r["kind"] = "H-L raw (%/mo)" if c.startswith("hl_") else "FM slope (ret_adj, %)"
        rows.append(r)
        if c.startswith("hl_"):
            a, ta = alpha(S[c], F[["CMKT", "CMOM"]], lags)
            r = ci_row(c + "_alpha", a, ta, boot_alphas(x, Fm, idx), int(np.isfinite(x).sum()))
            if n_seed_reps:
                r |= _seed_stability(lambda K: (boot_alphas(x, Fm, ii) for ii in idx_k), n_seed_reps)
            r["kind"] = "H-L alpha CMKT+CMOM (%/mo)"
            rows.append(r)
    out = pd.DataFrame(rows)
    out.insert(1, "kind", out.pop("kind"))
    if n_seed_reps:
        s = out["share_seeds_ci_excludes_0"]
        out["ci_verdict"] = np.where(out["ci_excludes_0"] & (s == 1.0), "excludes 0 (all seeds)",
                            np.where(~out["ci_excludes_0"] & (s == 0.0), "contains 0 (all seeds)",
                                     "borderline (depends on seed)"))
    out["block"], out["n_draws"], out["seed"] = block, n_draws, seed
    return out


def _stat_on(S: pd.DataFrame, F: pd.DataFrame, keep: np.ndarray, col: str, lags: int, use_alpha: bool):
    s = S.loc[keep, col]
    if use_alpha:
        a, t = alpha(s, F.loc[keep, ["CMKT", "CMOM"]], lags)
        return a, t
    m, _, t = nw_mean(s.dropna(), lags)
    return m, t


STAB_STATS = [("hl_A_C", False), ("hl_A_C", True), ("fm2_lnA_C", False)]
EXCL_STATS = [("hl_A", False), ("hl_A_C", False), ("hl_ResA", False), ("hl_A_C", True),
              ("fm1_lnA", False), ("fm2_lnA_C", False), ("fm3_ResLnA", False),
              ("fm4_lnA_C", False), ("fm4_ResLnA", False)]


def _label(col, use_alpha):
    return col + ("_alpha" if use_alpha else "")


def loyo_table(S: pd.DataFrame, F: pd.DataFrame, lags: int, stats=STAB_STATS) -> pd.DataFrame:
    """Leave-one-calendar-year-out: drop all months of year y, NW(lags) on the remaining months
    (treated as contiguous). Row 'all' is the full sample."""
    years = sorted({p.year for p in S.index})
    rows = []
    for y in [None] + years:
        keep = np.array([p.year != y for p in S.index])
        for col, ua in stats:
            est, t = _stat_on(S, F, keep, col, lags, ua)
            rows.append({"dropped_year": "none" if y is None else str(y), "stat": _label(col, ua),
                         "estimate": est, "t_nw": t, "n_months": int(S.loc[keep, col].notna().sum())})
    out = pd.DataFrame(rows)
    full = out[out["dropped_year"] == "none"].set_index("stat")
    out["sign_flip_vs_full"] = [np.sign(r.estimate) != np.sign(full.loc[r.stat, "estimate"]) for r in out.itertuples()]
    out["abs_t_gt_1.96"] = out["t_nw"].abs() > 1.96
    return out


def loyo_summary(L: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for st, g in L.groupby("stat", sort=False):
        full = g[g["dropped_year"] == "none"].iloc[0]
        d = g[g["dropped_year"] != "none"]
        imin, imax = d["t_nw"].idxmin(), d["t_nw"].idxmax()
        rows.append({"stat": st, "estimate_full": full["estimate"], "t_full": full["t_nw"],
                     "n_years": len(d), "est_min": d["estimate"].min(), "est_max": d["estimate"].max(),
                     "t_min": d.loc[imin, "t_nw"], "year_t_min": d.loc[imin, "dropped_year"],
                     "t_max": d.loc[imax, "t_nw"], "year_t_max": d.loc[imax, "dropped_year"],
                     "n_sign_flips": int(d["sign_flip_vs_full"].sum()),
                     "n_abs_t_gt_1.96": int(d["abs_t_gt_1.96"].sum())})
    return pd.DataFrame(rows)


def excl_year_table(S: pd.DataFrame, F: pd.DataFrame, lags: int, year: int = 2021) -> pd.DataFrame:
    rows = []
    for col, ua in EXCL_STATS:
        for name, keep in [("full", np.ones(len(S), bool)),
                           (f"excl_{year}", np.array([p.year != year for p in S.index]))]:
            est, t = _stat_on(S, F, keep, col, lags, ua)
            rows.append({"sample": name, "stat": _label(col, ua), "estimate": est, "t_nw": t,
                         "n_months": int(S.loc[keep, col].notna().sum())})
    return pd.DataFrame(rows)


def world0_calibration(w0: pd.DataFrame, hlz: float) -> pd.DataFrame:
    """False-positive rates of NW t under world 0 and the simulated 5% critical value of |t|."""
    rows = []
    for c in [c for c in w0.columns if c.endswith("_t") and not c.endswith("_nw0_t")]:
        t = w0[c].dropna().abs()
        n = len(t)
        fpr = float((t > 1.96).mean())
        rows.append({"stat": c[:-2], "n_reps": n, "fpr_1.96": fpr,
                     "fpr_1.96_se": float(np.sqrt(fpr * (1 - fpr) / n)),
                     f"fpr_{hlz:g}": float((t > hlz).mean()),
                     "crit_abs_t_5pct": float(np.quantile(t, 0.95)),
                     "sd_t": float(w0[c].std(ddof=1))})
    return pd.DataFrame(rows)


def mde_from_power_curve(prem: np.ndarray, power: np.ndarray, target: float = POWER_TARGET) -> float:
    """Smallest premium where the (linearly interpolated) power curve reaches `target`; NaN if never.
    The curve is first made monotone (running max), since simulated power is noisy."""
    o = np.argsort(prem)
    p, w = np.asarray(prem, float)[o], np.maximum.accumulate(np.asarray(power, float)[o])
    if w[0] >= target:
        return float(p[0])
    k = np.argmax(w >= target)
    if w[k] < target:
        return np.nan
    return float(p[k - 1] + (target - w[k - 1]) * (p[k] - p[k - 1]) / (w[k] - w[k - 1]))


def mde_from_mean_t(prem: np.ndarray, mean_t: np.ndarray, crit: float = 1.96,
                    target: float = POWER_TARGET) -> float:
    """Premium where E[t] = crit + z_target under a linear fit of mean t on premium
    (assumes t ~ N(E[t], 1) under the alternative)."""
    b1, b0 = np.polyfit(np.asarray(prem, float), np.asarray(mean_t, float), 1)
    return float((crit + norm.ppf(target) - b0) / b1)


def mde_table(power: pd.DataFrame, w0cal: pd.DataFrame, sim_dims: dict, real_dims: dict,
              real_se: pd.DataFrame, target: float = POWER_TARGET) -> pd.DataFrame:
    """MDE at `target` power, in %/month per 1 s.d. of the signal (FM) or %/month of H-L.

    method 'sim_power_curve'  : interpolated simulated power (and its 95% CI band) at |t|>1.96;
    method 'sim_mean_t'       : linear E[t](premium), critical 1.96 and the world-0 calibrated one;
    method 'sim_scaled_to_real': sim MDE x sqrt(N_sim T_sim / (N_real T_real))  [estimate];
    method 'real_se'          : (crit + z_target) x SE of our estimate (NW and bootstrap) [estimate].
    """
    z = norm.ppf(target)
    rows = []
    scale = np.sqrt(sim_dims["N"] * sim_dims["T"] / (real_dims["N"] * real_dims["T"]))
    for st, g in power.groupby("stat", sort=False):
        g = g.sort_values("premium")
        prem = 100.0 * g["premium"].to_numpy(float)
        crit_sim = w0cal.set_index("stat")["crit_abs_t_5pct"].get(st, np.nan)
        base = {"stat": st, "world": g["world"].iloc[0], "units": "%/month per 1 s.d. of signal"}
        m_curve = mde_from_power_curve(prem, g["share_abs_t_gt_1.96"].to_numpy())
        m_lo = mde_from_power_curve(prem, g["ci95_hi"].to_numpy())
        m_hi = mde_from_power_curve(prem, g["ci95_lo"].to_numpy())
        rows.append(base | {"method": "sim_power_curve", "crit_t": 1.96, "mde": m_curve,
                            "mde_lo": m_lo, "mde_hi": m_hi, "n_reps_per_point": int(g["n_reps"].min())})
        for crit, tag in [(1.96, "1.96"), (crit_sim, "world0_5pct")]:
            if np.isfinite(crit):
                m = mde_from_mean_t(prem, g["mean_t"].to_numpy(), crit, target)
                rows.append(base | {"method": f"sim_mean_t_crit_{tag}", "crit_t": crit, "mde": m})
                rows.append(base | {"method": f"sim_scaled_to_real_crit_{tag}", "crit_t": crit,
                                    "mde": m * scale, "scale": scale})
    for r in real_se.itertuples():
        for crit, tag in [(1.96, "1.96"), (r.crit_world0, "world0_5pct")]:
            if not np.isfinite(crit):
                continue
            for se_kind, se in [("nw", r.se_nw), ("boot", r.se_boot)]:
                rows.append({"stat": r.stat, "world": "real", "units": r.units,
                             "method": f"real_se_{se_kind}_crit_{tag}", "crit_t": crit,
                             "mde": (crit + z) * se * r.sd_signal, "se": se, "sd_signal": r.sd_signal})
    out = pd.DataFrame(rows)
    out["power_target"] = target
    return out


def paper_effect_rows(paper: dict) -> pd.DataFrame:
    """Effect sizes of Lou & Shu in the same units as our MDE (for comparison).

    FM: slope x s.d. of ln A_C as stated by the paper (pp. 13-14: s.d. 2.53, so 0.183 x 2.53 = 0.46%/mo
    in model 2). An earlier version approximated the s.d. from Table 1 quartiles as
    ln(q75/q25)/1.349 = 3.53, which overstated the paper's effect by 40%; that value is kept in
    column `sd_signal_iqr_approx` for reference only. The paper does not say whether 2.53 is pooled
    or cross-sectional; ours is the mean monthly cross-sectional s.d. -- so the ratio is an estimate."""
    ls = paper["lou_shu_2017"]
    q = ls["table1_panelA"]["A_C"]
    sd_iqr = float(np.log(q["q75"] / q["q25"]) / (2 * norm.ppf(0.75)))
    sd = float(ls.get("table4_text", {}).get("sd_lnA_C", PAPER_SD_LNA_C))
    rows = [{"stat": "hl_A_C", "world": "paper", "method": "paper_effect", "units": "%/month of H-L (Q5-Q1)",
             "mde": float(ls["table3_panelA"]["A_C"]["raw"]["hl"]), "sd_signal": 1.0},
            {"stat": "hl_A", "world": "paper", "method": "paper_effect", "units": "%/month of H-L (Q5-Q1)",
             "mde": float(ls["table3_panelA"]["A"]["raw"]["hl"]), "sd_signal": 1.0}]
    for st, model, var in [("fm2_lnA_C", "(2)", "lnA_C"), ("fm4_lnA_C", "(4)", "lnA_C")]:
        c = float(ls["table4_panelA"][model][var]["coef"])
        rows.append({"stat": st, "world": "paper", "method": "paper_effect_per_sd_estimate",
                     "units": "%/month per 1 s.d. of signal", "mde": c * sd, "se": np.nan, "sd_signal": sd,
                     "sd_signal_iqr_approx": sd_iqr})
    return pd.DataFrame(rows)


def real_se_frame(boot: pd.DataFrame, panel: pd.DataFrame, w0cal: pd.DataFrame) -> pd.DataFrame:
    """SE of our key estimates (NW = est/t, bootstrap) and the scale that converts them to
    %/month per s.d. (mean monthly cross-sectional s.d. of the regressor in the FM sample)."""
    crit = w0cal.set_index("stat")["crit_abs_t_5pct"]
    rows = []
    for st in ["fm2_lnA_C", "fm4_lnA_C", "fm3_ResLnA", "hl_A_C", "hl_ResA", "hl_A"]:
        b = boot.set_index("stat").loc[st]
        if st.startswith("fm"):
            xs, var = next((x, v) for n, _, x, v in FM_STATS if n == st)
            sub = panel.dropna(subset=["ret_adj"] + xs + CONTROLS)
            sd, units = float(sub.groupby("month")[var].std(ddof=1).mean()), "%/month per 1 s.d. of signal"
        else:
            sd, units = 1.0, "%/month of H-L (Q5-Q1)"
        se_nw = abs(b["estimate"] / b["t_nw"]) if b["t_nw"] else np.nan
        rows.append({"stat": st, "se_nw": se_nw, "se_boot": b["boot_se"], "sd_signal": sd, "units": units,
                     "crit_world0": float(crit.get(st, np.nan))})
    return pd.DataFrame(rows)


def _t_columns(df: pd.DataFrame) -> list[str]:
    out = []
    for c in df.columns:
        lc = str(c).lower()
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if lc.startswith(("paper", "sd_", "share", "n_")):
            continue
        if lc.endswith("_t") and lc.startswith(NOT_T_PREFIXES):
            continue
        if lc in {"t", "tstat", "t_stat", "t_nw", "nw_t", "t_boot"} or lc.endswith("_t") or lc.startswith("t_"):
            out.append(c)
    return out


def _estimate_col(df: pd.DataFrame, tcol: str) -> str | None:
    lc = str(tcol)
    for pre in [lc[:-2] if lc.endswith("_t") else None, lc[2:] if lc.startswith("t_") else None]:
        if pre and pre in df.columns and pre not in ("nw", "boot") and pd.api.types.is_numeric_dtype(df[pre]):
            return pre
    for c in ESTIMATE_COLS:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            return c
    cols = list(df.columns)
    for c in reversed(cols[:cols.index(tcol)]):
        if pd.api.types.is_numeric_dtype(df[c]) and c not in _t_columns(df):
            return c
    return None


def ledger_from_csv(path: Path, source: str, category: str) -> pd.DataFrame:
    """One ledger row per (row, t-column) of a results CSV."""
    try:
        df = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
        return pd.DataFrame()
    tcols = _t_columns(df)
    if not tcols:
        return pd.DataFrame()
    id_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    rows = []
    for tcol in tcols:
        ecol = _estimate_col(df, tcol)
        for i, r in df.iterrows():
            t = r[tcol]
            if not np.isfinite(t):
                continue
            ident = " | ".join(f"{c}={r[c]}" for c in id_cols if pd.notna(r[c]))
            var = next((str(r[c]) for c in ["var", "stat", "signal"] if c in df.columns and pd.notna(r[c])), "")
            leg = str(r["portfolio"]).lower() if "portfolio" in df.columns and pd.notna(r["portfolio"]) else ""
            if "symbol" in df.columns:
                role = "measurement"
            elif path.stem.startswith("describe") or path.stem.endswith("_describe"):
                role = "descriptive"
            elif "leak" in ident:
                role = "diagnostic"
            elif category == "placebo" or "placebo" in path.stem or "shuffle" in ident:
                role = "placebo"
            elif var in CONTROL_VARS or str(tcol).lower() in LOADING_T:
                role = "control"
            elif str(tcol).lower()[2:] in LEG_NAMES or leg in LEG_NAMES:
                role = "descriptive"
            else:
                role = "test"
            rows.append({"source": source, "file": path.name, "row": int(i), "test": ident or f"row {i}",
                         "t_column": tcol, "estimate_column": ecol,
                         "estimate": float(r[ecol]) if ecol and pd.notna(r[ecol]) else np.nan,
                         "t": float(t), "role": role})
    return pd.DataFrame(rows)


def build_ledger(base_dir: Path, upgrades_dir: Path | None = None, exclude: set[str] | None = None,
                 hlz: float = 3.0) -> pd.DataFrame:
    """All t-statistics of the base run, then of every results/upgrades/<name>/*.csv.

    Duplicates (same t to 6 decimals already in the ledger, e.g. T2 model (1) = T4 model (1)) are
    kept but flagged, so that `unique` counts each distinct test once."""
    parts = []
    for p in sorted(base_dir.glob("*.csv")):
        if p.name in LEDGER_SKIP or p.name.startswith(LEDGER_SKIP_PREFIXES):
            continue
        parts.append(ledger_from_csv(p, "base", "placebo" if p.stem.startswith("placebo") else "base"))
    if upgrades_dir is not None and upgrades_dir.exists():
        for d in sorted(x for x in upgrades_dir.iterdir() if x.is_dir()):
            for p in sorted(d.glob("*.csv")):
                if p.name in LEDGER_SKIP or (exclude and f"{d.name}/{p.name}" in exclude):
                    continue
                parts.append(ledger_from_csv(p, f"upgrade:{d.name}", "upgrade"))
    parts = [x for x in parts if len(x)]
    if not parts:
        return pd.DataFrame()
    L = pd.concat(parts, ignore_index=True)
    key = [(k if k != 0 else ("zero", i)) for i, k in enumerate(L["t"].round(6))]
    seen: dict = {}
    dup = []
    for k, (src, f, r) in zip(key, zip(L["source"], L["file"], L["row"])):
        dup.append(seen.get(k, ""))
        seen.setdefault(k, f"{src}/{f}#{r}")
    L["duplicate_of"] = dup
    L["unique"] = L["duplicate_of"] == ""
    L["abs_t_gt_1.96"] = L["t"].abs() > 1.96
    L[f"abs_t_gt_{hlz:g}"] = L["t"].abs() > hlz
    L.insert(0, "test_id", np.arange(1, len(L) + 1))
    return L


def add_multiple_testing(L: pd.DataFrame, alpha_level: float = 0.05) -> pd.DataFrame:
    """Two-sided normal p-values; Bonferroni and Benjamini-Hochberg over the unique hypothesis tests
    (role == 'test'); controls and placebos are not part of the family."""
    L = L.copy()
    L["p_normal"] = 2 * norm.sf(L["t"].abs())
    fam = L["unique"] & (L["role"] == "test")
    m = int(fam.sum())
    L["in_family"] = fam
    L["p_bonferroni"] = np.where(fam, np.minimum(1.0, L["p_normal"] * max(m, 1)), np.nan)
    p = L.loc[fam, "p_normal"].to_numpy()
    q = np.full(len(p), np.nan)
    if m:
        o = np.argsort(p)
        adj = p[o] * m / np.arange(1, m + 1)
        adj = np.minimum.accumulate(adj[::-1])[::-1]
        q[o] = np.minimum(1.0, adj)
    L["q_bh"] = np.nan
    L.loc[fam, "q_bh"] = q
    L["family_size"] = m
    return L


def ledger_summary(L: pd.DataFrame, hlz: float, alpha_level: float = 0.05) -> pd.DataFrame:
    rows = []
    m = int(L["in_family"].sum()) if len(L) else 0
    for (src, role), g in L.groupby(["source", "role"]):
        u = g[g["unique"]]
        rows.append({"source": src, "role": role, "n_rows": len(g), "n_unique": len(u),
                     "n_abs_t_gt_1.96": int(u["abs_t_gt_1.96"].sum()),
                     f"n_abs_t_gt_{hlz:g}": int(u[f"abs_t_gt_{hlz:g}"].sum()),
                     "expected_false_1.96_if_all_null": 0.05 * len(u)})
    fam = L[L["in_family"]]
    rows.append({"source": "ALL", "role": "test (family)", "n_rows": int((L["role"] == "test").sum()),
                 "n_unique": m, "n_abs_t_gt_1.96": int(fam["abs_t_gt_1.96"].sum()),
                 f"n_abs_t_gt_{hlz:g}": int(fam[f"abs_t_gt_{hlz:g}"].sum()),
                 "expected_false_1.96_if_all_null": 0.05 * m,
                 "n_bonferroni_sig": int((fam["p_bonferroni"] < alpha_level).sum()),
                 "n_bh_sig": int((fam["q_bh"] < alpha_level).sum()),
                 "bonferroni_crit_t": float(norm.isf(alpha_level / (2 * max(m, 1))))})
    bf = L[L["unique"] & (L["role"] == "test") & (L["source"] == "base")]
    if len(bf):
        sub = add_multiple_testing(bf.drop(columns=["p_normal", "in_family", "p_bonferroni", "q_bh",
                                                    "family_size"], errors="ignore"), alpha_level)
        mb = len(sub)
        rows.append({"source": "BASE", "role": "test (family)", "n_rows": int(((L["role"] == "test") &
                     (L["source"] == "base")).sum()), "n_unique": mb,
                     "n_abs_t_gt_1.96": int(sub["abs_t_gt_1.96"].sum()),
                     f"n_abs_t_gt_{hlz:g}": int(sub[f"abs_t_gt_{hlz:g}"].sum()),
                     "expected_false_1.96_if_all_null": 0.05 * mb,
                     "n_bonferroni_sig": int((sub["p_bonferroni"] < alpha_level).sum()),
                     "n_bh_sig": int((sub["q_bh"] < alpha_level).sum()),
                     "bonferroni_crit_t": float(norm.isf(alpha_level / (2 * mb)))})
    return pd.DataFrame(rows)


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


def load_panel(base: str, base_dir: Path, cfg: dict) -> tuple[pd.DataFrame, Path, str]:
    """Base-run panel from data/cache; rebuilt from the daily cache with the base config if absent."""
    man = json.loads((base_dir / "run_manifest.json").read_text())
    path = resolve_panel_cache(man, base)
    if path.exists():
        p = pd.read_parquet(path)
        h = sha256_file(path)
    else:
        import analysis
        from run import load_daily
        daily_path = resolve_repo_path(man["data_source"])
        log.warning("panel cache %s missing: rebuilding from %s", path, daily_path)
        h_daily = sha256_file(daily_path)
        if h_daily != man.get("data_sha256"):
            raise RuntimeError("daily cache changed since the base run; rerun `make run` first")
        p, _ = analysis.build_panel(load_daily(daily_path), cfg, cfg["period"]["return_start"], cfg["period"]["return_end"])
        p["month"] = p["month"].astype(str)
        path, h = daily_path, h_daily
    p["month"] = pd.PeriodIndex(p["month"].astype(str), freq="M")
    return p, path, h


def consistency_checks(boot: pd.DataFrame, base_dir: Path) -> pd.DataFrame:
    """Our point estimates must equal the base run's T3/T4 (same code path)."""
    rows = []
    t3 = pd.read_csv(base_dir / "T3_sorts.csv")
    t4 = pd.read_csv(base_dir / "T4_fm.csv")
    b = boot.set_index("stat")
    for s in HL_SIGNALS:
        for row, st in [("raw", f"hl_{s}"), ("alpha", f"hl_{s}_alpha")]:
            ref = t3[(t3["signal"] == s) & (t3["row"] == row)].iloc[0]
            rows.append({"check": f"T3 {row} H-L {s}", "base": ref["H-L"], "ours": b.loc[st, "estimate"],
                         "base_t": ref["t"], "ours_t": b.loc[st, "t_nw"]})
    for name, model, _, var in FM_STATS:
        ref = t4[(t4["model"] == model) & (t4["var"] == var)].iloc[0]
        rows.append({"check": f"T4 {model} {var}", "base": ref["coef"], "ours": b.loc[name, "estimate"],
                     "base_t": ref["t"], "ours_t": b.loc[name, "t_nw"]})
    out = pd.DataFrame(rows)
    out["match"] = np.isclose(out["base"], out["ours"], rtol=1e-8, atol=1e-10) & \
        np.isclose(out["base_t"], out["ours_t"], rtol=1e-6, atol=1e-8)
    return out


def _f(x, d=2):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def _ci_word(r) -> str:
    if "ci_verdict" in r.index and isinstance(r["ci_verdict"], str):
        if r["ci_verdict"].startswith("borderline"):
            k = int(round(r["share_seeds_ci_excludes_0"] * r["n_seed_reps"]))
            return f"borderline ({k}/{int(r['n_seed_reps'])} seeds exclude 0)"
    return "yes" if r["ci_excludes_0"] else "no"


def write_summary(out: Path, base: str, tabs: dict, paper: dict, hlz: float, block: int, n_draws: int) -> str:
    boot, loyo_s, excl, w0, mde, lsum, chk = (tabs[k] for k in
        ["bootstrap_ci", "stability_loyo_summary", "stability_excl2021", "world0_calibration", "mde",
         "ledger_summary", "consistency_checks"])
    b = boot.set_index("stat")
    ls = paper["lou_shu_2017"]
    pap = {"hl_A": ls["table3_panelA"]["A"]["raw"], "hl_A_C": ls["table3_panelA"]["A_C"]["raw"],
           "hl_ResA": ls["table3_panelA"]["ResA"]["raw"],
           "hl_A_alpha": ls["table3_panelA"]["A"]["alpha"], "hl_A_C_alpha": ls["table3_panelA"]["A_C"]["alpha"],
           "hl_ResA_alpha": ls["table3_panelA"]["ResA"]["alpha"],
           "fm1_lnA": ls["table4_panelA"]["(1)"]["lnA"], "fm2_lnA_C": ls["table4_panelA"]["(2)"]["lnA_C"],
           "fm3_ResLnA": ls["table4_panelA"]["(3)"]["ResLnA"], "fm4_lnA_C": ls["table4_panelA"]["(4)"]["lnA_C"],
           "fm4_ResLnA": ls["table4_panelA"]["(4)"]["ResLnA"]}
    L = []
    L.append("# Upgrade 'robustness': overfitting and statistical inference\n")
    L.append(f"_Generated by `src/robustness_extra.py` from the CSVs in this folder; base run `{base}`. "
             "Not edited by hand._\n")
    if not chk["match"].all():
        L.append("**WARNING: the point estimates do not match T3/T4 of the base run (see consistency_checks.csv).**\n")
    L.append("## What was checked\n")
    L.append(f"1. Block bootstrap over months (moving block, block of {block} months, {n_draws} draws, seed from config): "
             "95% percentile CIs for the H–L of the A, A_C and Res.A quintiles (raw and CMKT+CMOM alpha) and for the FM slopes.")
    L.append("2. Specification ledger: every t-statistic of the base run and the upgrades, flags |t|>1.96 and "
             f"|t|>{hlz:g} (Harvey–Liu–Zhu), Bonferroni and BH over the family of unique tests.")
    L.append("3. Stability: drop one calendar year at a time; separately, drop 2021 (bull market).")
    L.append("4. Test size in world 0 and the minimum detectable effect (MDE) at 80% power.\n")

    L.append("## Bootstrap CIs vs the paper\n")
    L.append("| statistic | paper (t) | us | t NW | 95% bootstrap CI | CI excludes 0 |")
    L.append("|---|---|---|---|---|---|")
    for st in ["hl_A", "hl_A_C", "hl_ResA", "hl_A_alpha", "hl_A_C_alpha", "hl_ResA_alpha",
               "fm1_lnA", "fm2_lnA_C", "fm3_ResLnA", "fm4_lnA_C", "fm4_ResLnA"]:
        r, p = b.loc[st], pap[st]
        pe = p.get("hl", p.get("coef"))
        L.append(f"| {st} | {_f(pe)} ({_f(p['t'])}) | {_f(r['estimate'])} | {_f(r['t_nw'])} | "
                 f"[{_f(r['ci95_lo'])}; {_f(r['ci95_hi'])}] | {_ci_word(r)} |")
    n_ex = int(b.loc[[s for s in pap], "ci_excludes_0"].sum())
    txt = f"\nOf the {len(pap)} key statistics, the CI (seed from config) excludes zero for {n_ex}"
    if "ci_verdict" in b.columns:
        v = b.loc[[s for s in pap], "ci_verdict"]
        K = int(b["n_seed_reps"].iloc[0])
        txt += (f"; robustly under all {K} other bootstrap seeds for {int((v == 'excludes 0 (all seeds)').sum())}, "
                f"borderline (the result depends on the seed) for {int((v == 'borderline (depends on seed)').sum())}")
    L.append(txt + ". H–L in %/month; FM is the slope on the CMKT-adjusted return "
             "(% per unit of the regressor; the paper uses FF3). Our alpha is on CMKT+CMOM, the paper's on 4 factors.\n")

    L.append("## Stability across years\n")
    L.append("| statistic | full sample (t) | t min (year dropped) | t max (year dropped) | sign flips | years with \\|t\\|>1.96 |")
    L.append("|---|---|---|---|---|---|")
    for _, r in loyo_s.iterrows():
        L.append(f"| {r['stat']} | {_f(r['estimate_full'])} ({_f(r['t_full'])}) | {_f(r['t_min'])} ({r['year_t_min']}) | "
                 f"{_f(r['t_max'])} ({r['year_t_max']}) | {r['n_sign_flips']}/{r['n_years']} | "
                 f"{r['n_abs_t_gt_1.96']}/{r['n_years']} |")
    e = excl.pivot(index="stat", columns="sample", values=["estimate", "t_nw"])
    L.append("\nWithout 2021: " + "; ".join(
        f"{st} {_f(e.loc[st, ('estimate', 'full')])} (t {_f(e.loc[st, ('t_nw', 'full')])}) → "
        f"{_f(e.loc[st, ('estimate', 'excl_2021')])} (t {_f(e.loc[st, ('t_nw', 'excl_2021')])})"
        for st in ["hl_A_C", "fm2_lnA_C", "fm3_ResLnA"]) + ".\n")
    w0i = w0.set_index("stat")

    L.append("## Multiple testing\n")
    L.append("The family contains only rows with role `test`, without duplicates of the same t-statistic. "
             "Excluded: `control` (FM controls, factor loadings, intercepts), `placebo` (shuffling), "
             "`diagnostic` (lag 0, the leakage check, where a large |t| is expected), `descriptive` (single quintiles, "
             "descriptive moments), `measurement` (first-stage estimation of a measure per coin-month).\n")
    for fam_src, label in [("BASE", "Base run only (crypto, Lou–Shu)"),
                           ("ALL", "Everything: base run and all upgrades (including the US and the variants)")]:
        sel = lsum[lsum["source"] == fam_src]
        if not len(sel):
            continue
        a = sel.iloc[0]
        L.append(f"- {label}: unique tests {int(a['n_unique'])}; |t|>1.96 for {int(a['n_abs_t_gt_1.96'])} "
                 f"(≈{_f(a['expected_false_1.96_if_all_null'], 2)} expected if all nulls were true); "
                 f"|t|>{hlz:g} for {int(a[f'n_abs_t_gt_{hlz:g}'])}. After Bonferroni (threshold |t|>{_f(a['bonferroni_crit_t'])}) "
                 f"{int(a['n_bonferroni_sig'])} are significant, after BH (q<0.05) {int(a['n_bh_sig'])}.")
        if fam_src == "BASE" and (out / "ledger.csv").exists():
            led = pd.read_csv(out / "ledger.csv")
            bs = led[(led["source"] == "base") & led["unique"] & (led["role"] == "test") & led["abs_t_gt_1.96"]]
            n_res = int(bs["test"].str.contains(r"Res", regex=True).sum())
            fpr0 = float(w0i["fpr_1.96"].mean())
            L.append(f"  Of these {len(bs)} tests with |t|>1.96, {n_res} are variants of one hypothesis about Res.A/ResLnA "
                     "(raw returns, subperiods, weighting, winsorisation, models with controls), i.e. they are "
                     "strongly correlated, not independent. With the NW(6) size from world 0 "
                     f"({_f(100 * fpr0, 1)}% on average instead of 5%), ≈{_f(fpr0 * int(a['n_unique']), 1)} of "
                     f"{int(a['n_unique'])} would be expected if all nulls were true; because the tests are correlated, the spread "
                     "of this number is much wider than binomial.")
    L.append("")
    L.append("| source | role | rows | unique | \\|t\\|>1.96 | \\|t\\|>" + f"{hlz:g}" + " |")
    L.append("|---|---|---|---|---|---|")
    for _, r in lsum[~lsum["source"].isin(["ALL", "BASE"])].iterrows():
        L.append(f"| {r['source']} | {r['role']} | {int(r['n_rows'])} | {int(r['n_unique'])} | "
                 f"{int(r['n_abs_t_gt_1.96'])} | {int(r[f'n_abs_t_gt_{hlz:g}'])} |")
    L.append("\nThe full list is in `ledger.csv`.\n")

    L.append("## Test size and power (simulations)\n")
    w = w0.set_index("stat")
    L.append("World 0 (no premium): share of |t|>1.96: " + ", ".join(
        f"{s} {_f(100 * w.loc[s, 'fpr_1.96'], 1)}%" for s in w.index if s in ("fm2_lnA_C", "fm3_ResLnA", "hl_A_C")) +
        f"; share of |t|>{hlz:g}: " + ", ".join(
        f"{s} {_f(100 * w.loc[s, f'fpr_{hlz:g}'], 1)}%" for s in w.index if s in ("fm2_lnA_C", "fm3_ResLnA", "hl_A_C")) +
        ". Empirical 5% critical |t| in world 0: " + ", ".join(
        f"{s} {_f(w.loc[s, 'crit_abs_t_5pct'])}" for s in w.index if s in ("fm2_lnA_C", "fm3_ResLnA", "hl_A_C")) + ".\n")
    md = mde.set_index(["stat", "method"])

    def g(st, m):
        return md["mde"].get((st, m), np.nan)
    L.append("MDE at 80% power (estimates):")
    L.append(f"- from the simulated power curve (|t|>1.96), FM lnA_C model 2: {_f(g('fm2_lnA_C', 'sim_power_curve'))} %/month "
             f"per 1 s.d. (range from the 95% CI of power [{_f(md['mde_lo'].get(('fm2_lnA_C', 'sim_power_curve')))}; "
             f"{_f(md['mde_hi'].get(('fm2_lnA_C', 'sim_power_curve')))}]); H–L A_C: {_f(g('hl_A_C', 'sim_power_curve'))} "
             "%/month per 1 s.d. (units of the simulated premium, not H–L).")
    L.append(f"- simulation rescaled to our N×T (√(N·T) scaling), FM lnA_C: {_f(g('fm2_lnA_C', 'sim_scaled_to_real_crit_1.96'))} "
             f"%/month per 1 s.d.; with the world-0 critical value: {_f(g('fm2_lnA_C', 'sim_scaled_to_real_crit_world0_5pct'))}.")
    L.append(f"- from the standard error of our real estimate (NW): FM lnA_C {_f(g('fm2_lnA_C', 'real_se_nw_crit_1.96'))} "
             f"%/month per 1 s.d.; H–L A_C {_f(g('hl_A_C', 'real_se_nw_crit_1.96'))} %/month "
             f"(bootstrap SE: {_f(g('hl_A_C', 'real_se_boot_crit_1.96'))}).")
    ms, mr = g("fm2_lnA_C", "sim_scaled_to_real_crit_1.96"), g("fm2_lnA_C", "real_se_nw_crit_1.96")
    if np.isfinite(ms) and np.isfinite(mr):
        ratio = mr / ms
        L.append(f"- MDE from the real SE / MDE from the simulation (FM lnA_C) = {_f(ratio)} (an estimate: the simulated "
                 "MDE is rescaled to our N×T by √(N·T)): "
                 + ("the simulation substantially understates the noise of real returns, so we rely on the MDE from the real SE."
                    if ratio > 2 else "the simulation and the real SE agree in order of magnitude."
                    if ratio >= 0.5 else "the real SE is clearly smaller than the simulated one; the simulated MDE is conservative."))
    L.append("")
    for st, pm, name in [("hl_A_C", "paper_effect", "H–L A_C, %/month"),
                         ("fm2_lnA_C", "paper_effect_per_sd_estimate", "FM lnA_C (2), %/month per 1 s.d. (s.d. of ln A_C = 2.53 from the paper's text, pp. 13-14)")]:
        pe, mde_r, sig = g(st, pm), g(st, "real_se_nw_crit_1.96"), abs(b.loc[st, "t_nw"]) > 1.96
        if not (np.isfinite(pe) and np.isfinite(mde_r)):
            continue
        if mde_r > pe:
            verdict = ("our sample does not detect an effect of this size with 80% power"
                       + ("; our insignificant result does not prove the absence of a premium." if not sig
                          else ", yet our estimate is significant: it is larger than the paper's effect."))
        else:
            verdict = ("we would detect an effect of this size with ≥80% power"
                       + ("; since ours is insignificant, a premium of the paper's size is unlikely on our market." if not sig
                          else "; our estimate is significant."))
        L.append(f"- {name}: paper's effect {_f(pe)}, our MDE {_f(mde_r)} ({_f(mde_r / pe, 1)} times larger); {verdict}")
    L.append("\n## Caveats\n")
    L.append(f"- The block bootstrap with block {block} keeps autocorrelation only within a block; the percentile CI with "
             f"{int(boot['n_months'].min())}–{int(boot['n_months'].max())} months is approximate; the non-circular MBB "
             "underweights the ends of the sample.")
    if "ci_verdict" in boot.columns:
        L.append(f"- 'borderline' in the bootstrap table: the CI bound is so close to 0 that under other seeds (range of the lower "
                 "bound in `bootstrap_ci.csv`, columns ci95_lo_seed_*) the CI sometimes excludes and sometimes includes 0; we do "
                 "not count such a result as significant.")
    L.append(f"- The world-0 thresholds are estimated from {int(w0['n_reps'].min())} replications; the standard error of the "
             f"false-positive rate is ≈{_f(100 * float(w0['fpr_1.96_se'].mean()), 1)} pp, and the 95% quantile of |t| is noisy too.")
    L.append("- The paper's effect per 1 s.d. is taken from its text (s.d. 2.53); the paper does not say whether this is a pooled "
             "or an average cross-sectional s.d.; ours is the average cross-sectional s.d. The ratio of the MDE to the paper's effect is an estimate.")
    L.append("- Dropping a year joins the neighbouring months in the NW estimate; betas/CMOM are not re-estimated.")
    L.append("- The simulated MDE depends on the DGP (noise, number of coins); rescaling to the real N×T is an estimate, not a measurement.")
    lo = w0["fpr_1.96"] - 1.96 * w0["fpr_1.96_se"]
    rej = ", ".join(w0.loc[lo > 0.05, "stat"]) or "none"
    L.append(f"- The HLZ bar t>{hlz:g} is shown next to 1.96. In world 0 the NW(6) false-positive rate at 1.96 is "
             f"{_f(100 * w0['fpr_1.96'].min(), 1)}–{_f(100 * w0['fpr_1.96'].max(), 1)}%; its 95% CI (±1.96 s.e.) excludes 5% "
             f"only for {rej}, so 1.96 may be laxer than its nominal level, clearly so only for those statistics.")
    if not chk["match"].all():
        L.append("- The point estimates differ from the base run; the interpretation is invalid until this is fixed.")
    text = "\n".join(L) + "\n"
    (out / "SUMMARY.md").write_text(text)
    return text


def run_ledger(base_dir: Path, upgrades_dir: Path, out: Path, hlz: float) -> pd.DataFrame:
    L = build_ledger(base_dir, upgrades_dir, exclude=OWN_NON_TESTS, hlz=hlz)
    L = add_multiple_testing(L)
    L.to_csv(out / "ledger.csv", index=False)
    S = ledger_summary(L, hlz)
    S.to_csv(out / "ledger_summary.csv", index=False)
    return S


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--base", default=None, help="base run id (default: results/LATEST)")
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    ap.add_argument("--out", default=None, help="default: <results-dir>/upgrades/robustness")
    ap.add_argument("--draws", type=int, default=N_DRAWS)
    ap.add_argument("--block", type=int, default=BLOCK)
    ap.add_argument("--sims", default=str(ROOT / "results" / "simulations"))
    ap.add_argument("--ledger-only", action="store_true",
                    help="only rebuild ledger.csv over the base run and results/upgrades/*/ (run after all upgrades)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg_path = Path(a.config)
    cfg = yaml.safe_load(cfg_path.read_text())
    res_dir = Path(a.results_dir)
    base = a.base or (res_dir / "LATEST").read_text().strip()
    base_dir = res_dir / base
    upg_dir = res_dir / "upgrades"
    out = Path(a.out) if a.out else upg_dir / "robustness"
    out.mkdir(parents=True, exist_ok=True)
    lags, hlz, seed = int(cfg["tests"]["nw_lags"]), float(cfg["tests"]["hlz_threshold"]), int(cfg["seed"])
    timings: dict[str, float] = {}
    t0 = time.perf_counter()

    def tick(name):
        nonlocal t0
        timings[name] = round(time.perf_counter() - t0, 3)
        log.info("stage %s done in %.1fs", name, timings[name])
        t0 = time.perf_counter()

    if a.ledger_only:
        man_path = out / "run_manifest.json"
        if a.base is None and man_path.exists():
            mb = json.loads(man_path.read_text()).get("base_run_id")
            if mb and mb != base:
                log.warning("results/LATEST=%s but robustness tables are for %s: ledger uses %s; "
                            "rerun the full upgrade to move to LATEST", base, mb, mb)
            base = mb or base
            base_dir = res_dir / base
        S = run_ledger(base_dir, upg_dir, out, hlz)
        log.info("ledger rebuilt: %s", S.to_dict("records")[-1])
        man_path = out / "run_manifest.json"
        if man_path.exists():
            man = json.loads(man_path.read_text())
            man["ledger_rebuilt_utc"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            man["ledger_sources"] = sorted(str(p.relative_to(res_dir)) for p in upg_dir.glob("*/*.csv"))
            man["ledger_code_sha256"] = sha256_file(ROOT / "src" / "robustness_extra.py")
            man_path.write_text(json.dumps(man, indent=2))
            need = ["bootstrap_ci", "stability_loyo_summary", "stability_excl2021", "world0_calibration",
                    "mde", "consistency_checks"]
            if all((out / f"{k}.csv").exists() for k in need):
                tabs = {k: pd.read_csv(out / f"{k}.csv") for k in need} | {"ledger_summary": S}
                prm = man.get("params", {})
                paper = yaml.safe_load((ROOT / "src" / "paper_numbers.yaml").read_text())
                write_summary(out, man.get("base_run_id", base), tabs, paper, hlz,
                              int(prm.get("block", a.block)), int(prm.get("n_draws", a.draws)))
        return

    panel, panel_path, panel_hash = load_panel(base, base_dir, cfg)
    tick("load_panel")
    S, F = monthly_series(panel, cfg)
    tick("monthly_series")
    tabs: dict[str, pd.DataFrame] = {}
    tabs["bootstrap_ci"] = bootstrap_table(S, F, lags, a.block, a.draws, seed)
    tick("bootstrap")
    tabs["consistency_checks"] = consistency_checks(tabs["bootstrap_ci"], base_dir)
    tabs["stability_loyo"] = loyo_table(S, F, lags)
    tabs["stability_loyo_summary"] = loyo_summary(tabs["stability_loyo"])
    tabs["stability_excl2021"] = excl_year_table(S, F, lags, 2021)
    tick("stability")

    sims = Path(a.sims)
    w0 = pd.read_csv(sims / "sim_world0_reps.csv")
    power = pd.read_csv(sims / "sim_power.csv")
    preps = pd.read_csv(sims / "sim_power_reps.csv")
    tabs["world0_calibration"] = world0_calibration(w0, hlz)
    sample = pd.read_csv(base_dir / "sample.csv").iloc[0]
    t4 = pd.read_csv(base_dir / "T4_fm.csv")
    sim_dims = {"N": float(preps["avg_coins"].mean()), "T": float(preps["n_months"].mean())}
    real_dims = {"N": float(t4.loc[t4["model"] == "(2)", "avg_obs"].iloc[0]),
                 "T": float(t4.loc[t4["model"] == "(2)", "n_months"].iloc[0])}
    rse = real_se_frame(tabs["bootstrap_ci"], panel, tabs["world0_calibration"])
    paper = yaml.safe_load((ROOT / "src" / "paper_numbers.yaml").read_text())
    tabs["mde"] = pd.concat([mde_table(power, tabs["world0_calibration"], sim_dims, real_dims, rse),
                             paper_effect_rows(paper)], ignore_index=True)
    tabs["mde_inputs"] = pd.DataFrame([{"N_sim": sim_dims["N"], "T_sim": sim_dims["T"],
                                        "N_real_fm2": real_dims["N"], "T_real_fm2": real_dims["T"],
                                        "avg_coins_sorts": float(sample["avg_coins_per_month"]),
                                        "n_months_sorts": int(sample["n_months"])}])
    tick("mde")
    for k, v in tabs.items():
        v.to_csv(out / f"{k}.csv", index=False)
    tabs["ledger_summary"] = run_ledger(base_dir, upg_dir, out, hlz)
    tick("ledger")
    write_summary(out, base, tabs, paper, hlz, a.block, a.draws)
    tick("summary")

    code_files = ["src/robustness_extra.py", "src/fama_macbeth.py", "src/sorts.py", "src/paper_numbers.yaml"]
    inputs = [panel_path] + sorted(base_dir.glob("*.csv")) + [sims / f for f in
              ["sim_world0_reps.csv", "sim_power.csv", "sim_power_reps.csv"]]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = {
        "run_id": f"robustness_{stamp}",
        "upgrade": "robustness",
        "base_run_id": base,
        "created_utc": stamp,
        "git": git_info(),
        "config_path": rel_path(cfg_path),
        "config_sha256": sha256_file(cfg_path),
        "code_sha256": {f: sha256_file(ROOT / f) for f in code_files},
        "input_sha256": {rel_path(p): sha256_file(p) for p in inputs},
        "seed": seed,
        "params": {"block": a.block, "n_draws": a.draws, "n_seed_reps_ci_stability": N_SEED_REPS,
                   "seed_reps_scheme": "default_rng([seed, k]), k=1..n_seed_reps", "nw_lags": lags, "hlz_threshold": hlz,
                   "power_target": POWER_TARGET, "bootstrap": "moving block, non-circular, joint over months, percentile CI"},
        "consistency_all_match": bool(tabs["consistency_checks"]["match"].all()),
        "timings_sec": timings | {"total": round(sum(timings.values()), 3)},
        "tables": sorted(p.name for p in out.glob("*.csv")),
        "ledger_sources": sorted(str(p.relative_to(res_dir)) for p in upg_dir.glob("*/*.csv")),
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("robustness upgrade written to %s", out)


if __name__ == "__main__":
    main()
