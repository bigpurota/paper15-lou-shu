"""Known-answer simulations: worlds V, P, 0 run through the SAME pipeline as the real data
(task.md 4.4 and 13.1). Daily data are generated in the real-data format
(symbol, date, close, quote_volume) and passed to analysis.build_panel + analysis.run_tests.

DGP (per asset i, month t, day d in t; all latent states are monthly AR(1) processes):
  lv_{i,t}   log dollar-volume level = a_i + x_{i,t}; a_i ~ N(lv_mean, lv_sd_asset) gives the large
             cross-sectional dispersion of A_C (Q75/Q25 >> 10); x is AR(1) (lv_rho, lv_sd_time).
  DVol_d     = exp(lv_{i,t} + u_d), u_d ~ N(0, dvol_daily_sd), iid.
  l_{i,t}    log idiosyncratic volatility level, asset mean + AR(1); drawn INDEPENDENTLY of lv.
  r_d        = mu_{i,t}/n_days_t + beta_i m_d + s_{i,d} eps_d, beta_i ~ U[0.5, 1.5], m_d ~ N(0, mkt_sd),
             eps_d ~ standardised Student-t(t_df), simple returns, close = cumprod(1 + r).
World V (volume premium):  s_{i,d} = exp(l_{i,t})                (|r| independent of volume);
                           mu_{i,t} = k z_t(-lv_{i,t-2}).
World P (price impact):    s_{i,d} = exp(l_{i,t}) exp(g u_d - g^2 sd_u^2), g = 0.5, i.e.
                           |r_d| = lambda_{i,t} DVol_d^0.5 |eps_d| with
                           ln lambda_{i,t} = l_{i,t} - 0.5 lv_{i,t} - const.
                           Priced: mu_{i,t} = k z_t(l_{i,t-2}) = price impact IN EXCESS of the
                           square-root-of-volume law, independent of lv by construction.
                           (lambda itself cannot be independent of lv with realistic volatility:
                           with sd(lv) ~ 2.6 liquid coins would get daily vol > 40%.)
World 0 (no premium):      world-P return structure (incl. the within-month |r|-volume link), mu = 0.
k = cfg simulation.premium_per_sd (monthly, per 1 cross-sectional s.d. of the priced state);
z_t = cross-sectional z-score in month t. Calendar = real calendar days (28-31 per month), so
simulation.days_per_month is not used. First `warmup_months` (14) feed betas, momentum and IVOL.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
import traceback

import numpy as np
import pandas as pd
import yaml
from scipy.stats import binomtest

from analysis import build_panel, run_tests
from fama_macbeth import fama_macbeth
from measures import mean_cs_corr

WORLD_CODE = {"0": 0, "V": 1, "P": 2}
DGP_DEFAULTS = dict(warmup_months=14, first_month="2015-01",
                    lv_mean=float(np.log(5e6)), lv_sd_asset=2.5, lv_rho=0.9, lv_sd_time=0.6,
                    dvol_daily_sd=0.5, vol_mean=float(np.log(0.035)), vol_sd_asset=0.3,
                    vol_rho=0.8, vol_sd_time=0.25, mkt_sd=0.025, beta_lo=0.5, beta_hi=1.5,
                    t_df=5, impact_exp=0.5)
# statistics collected per replication: (key, test table, model/signal, variable)
FM_STATS = [("fm2_lnA_C", "(2)", "lnA_C"), ("fm3_ResLnA", "(3)", "ResLnA"),
            ("fm4_lnA_C", "(4)", "lnA_C"), ("fm4_ResLnA", "(4)", "ResLnA")]
SORT_STATS = [("hl_A", "A"), ("hl_A_C", "A_C"), ("hl_ResA", "ResA")]
# diagnostic only: same FM slopes with a 0-lag (heteroskedasticity-only) s.e. instead of NW(6)
NW0_STATS = [("fm2_lnA_C_nw0", "lnA_C"), ("fm3_ResLnA_nw0", "ResLnA")]
STAT_KEYS = [k for k, *_ in FM_STATS] + [k for k, _ in SORT_STATS] + [k for k, _ in NW0_STATS]
DIAG_KEYS = ["corr_A_AC", "AC_q75_q25", "absret_q75_q25", "mean_absret_daily", "avg_coins", "n_months"]
# pre-declared pass rules (share of replications)
PASS_SHARE_SIG, MAX_SHARE_FALSE = 0.8, 0.2


def dgp_params(cfg: dict) -> dict:
    sc = cfg.get("simulation", {})
    return {k: type(v)(sc.get(k, v)) for k, v in DGP_DEFAULTS.items()}


def _ar1(rng, n: int, T: int, rho: float, sd: float) -> np.ndarray:
    """n x T stationary AR(1) with unconditional s.d. `sd`."""
    x = np.empty((n, T))
    x[:, 0] = rng.normal(0.0, sd, n)
    e = rng.normal(0.0, sd * np.sqrt(1 - rho ** 2), (n, T))
    for t in range(1, T):
        x[:, t] = rho * x[:, t - 1] + e[:, t]
    return x


def _z(x: np.ndarray) -> np.ndarray:
    """Cross-sectional z-score of each column (month)."""
    return (x - x.mean(axis=0)) / x.std(axis=0)


def simulate_daily(world: str, cfg: dict, rng: np.random.Generator, premium: float | None = None,
                   n_assets: int | None = None, n_months: int | None = None) -> pd.DataFrame:
    """Daily panel (symbol, date, close, quote_volume) for world 'V', 'P' or '0'."""
    if world not in WORLD_CODE:
        raise ValueError(f"unknown world {world!r}")
    sc, p = cfg["simulation"], dgp_params(cfg)
    N = int(n_assets or sc["n_assets"])
    T = p["warmup_months"] + int(n_months or sc["n_months"])
    k = float(sc["premium_per_sd"] if premium is None else premium)

    months = pd.period_range(p["first_month"], periods=T, freq="M")
    dates = pd.date_range(months[0].start_time, months[-1].end_time.normalize(), freq="D")
    mi = np.asarray(dates.to_period("M").asi8 - months[0].ordinal)       # month index of each day
    ndays = np.bincount(mi, minlength=T)
    D = len(dates)

    lv = rng.normal(p["lv_mean"], p["lv_sd_asset"], (N, 1)) + _ar1(rng, N, T, p["lv_rho"], p["lv_sd_time"])
    lvol = rng.normal(p["vol_mean"], p["vol_sd_asset"], (N, 1)) + _ar1(rng, N, T, p["vol_rho"], p["vol_sd_time"])
    beta = rng.uniform(p["beta_lo"], p["beta_hi"], N)

    mu = np.zeros((N, T))
    if world != "0" and k != 0.0:
        state = -lv if world == "V" else lvol
        mu[:, 2:] = k * _z(state[:, :-2])

    su = p["dvol_daily_sd"]
    u = rng.normal(0.0, su, (N, D))
    dvol = np.exp(lv[:, mi] + u)
    s = np.exp(lvol[:, mi])
    if world in ("P", "0"):
        g = p["impact_exp"]
        s = s * np.exp(g * u - g ** 2 * su ** 2)                           # E[factor^2] = 1
    df = p["t_df"]
    eps = rng.standard_t(df, (N, D)) / np.sqrt(df / (df - 2))
    m = rng.normal(0.0, p["mkt_sd"], D)
    r = (mu / ndays)[:, mi] + beta[:, None] * m[None, :] + s * eps
    r = np.clip(r, -0.95, None)
    close = np.cumprod(1.0 + r, axis=1)

    return pd.DataFrame({
        "symbol": np.repeat([f"S{i:04d}" for i in range(N)], D),
        "date": np.tile(dates.to_numpy(), N),
        "close": close.ravel(),
        "quote_volume": dvol.ravel()})


def _rng(cfg: dict, world: str, rep: int, premium: float | None) -> np.random.Generator:
    tag = -1 if premium is None else int(round(premium * 1e6))
    ss = np.random.SeedSequence([int(cfg["seed"]), WORLD_CODE[world], int(rep), tag + 1_000_000])
    return np.random.default_rng(ss)


def run_world(world: str, cfg: dict, rep: int, premium: float | None = None,
              n_assets: int | None = None, n_months: int | None = None) -> dict:
    """Simulate one replication, run the real pipeline (T3, T4) and return key statistics."""
    t0 = time.perf_counter()
    daily = simulate_daily(world, cfg, _rng(cfg, world, rep, premium), premium, n_assets, n_months)
    p = dgp_params(cfg)
    months = daily["date"].dt.to_period("M")
    start, end = months.min() + p["warmup_months"], months.max()
    t1 = time.perf_counter()
    panel, factors = build_panel(daily, cfg, str(start), str(end))
    res = run_tests(panel, factors, cfg, full=False)
    t2 = time.perf_counter()

    k = 0.0 if world == "0" else float(cfg["simulation"]["premium_per_sd"] if premium is None else premium)
    out = {"world": world, "rep": rep, "premium": k}
    fm = res["T4_fm"].set_index(["model", "var"])
    for key, model, var in FM_STATS:
        out[f"{key}_coef"], out[f"{key}_t"] = fm.loc[(model, var), ["coef", "t"]]
    t3 = res["T3_sorts"]
    for key, sig in SORT_STATS:
        row = t3[(t3["signal"] == sig) & (t3["row"] == "raw")].iloc[0]
        out[f"{key}_coef"], out[f"{key}_t"] = float(row["H-L"]), float(row["t"])
    for key, var in NW0_STATS:
        r0 = fama_macbeth(panel, "ret_adj", [var, "mom", "ret_1"], lags=0).set_index("var")
        out[f"{key}_coef"], out[f"{key}_t"] = r0.loc[var, ["coef", "t"]]
    out["corr_A_AC"] = float(mean_cs_corr(panel, ["A", "A_C"], "month").iloc[0, 1])
    q = panel[["A_C", "absret"]].quantile([0.25, 0.75])
    out["AC_q75_q25"] = float(q.loc[0.75, "A_C"] / q.loc[0.25, "A_C"])
    out["absret_q75_q25"] = float(q.loc[0.75, "absret"] / q.loc[0.25, "absret"])
    out["mean_absret_daily"] = float(panel["absret"].mean())
    out["n_months"] = int(panel["month"].nunique())
    out["avg_coins"] = float(panel.groupby("month").size().mean())
    out["sec_simulate"], out["sec_pipeline"] = t1 - t0, t2 - t1
    return out


def _job(args) -> dict:
    world, cfg, rep, premium = args
    try:
        return run_world(world, cfg, rep, premium)
    except Exception:                                             # keep the batch alive, report it
        return {"world": world, "rep": rep, "premium": premium, "error": traceback.format_exc()}


def summarize(reps: pd.DataFrame, group: list[str]) -> pd.DataFrame:
    """Per group and statistic: mean coef, mean t, rejection shares with Clopper-Pearson 95% CI."""
    rows = []
    ok = reps[reps["error"].isna()] if "error" in reps else reps
    for keys, g in ok.groupby(group, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        for st in STAT_KEYS:
            t = g[f"{st}_t"].dropna()
            n, n_sig = len(t), int((t.abs() > 1.96).sum())
            ci = binomtest(n_sig, n).proportion_ci(0.95) if n else None
            rows.append({**dict(zip(group, keys)), "stat": st, "n_reps": n,
                         "mean_coef": g[f"{st}_coef"].mean(), "mean_t": t.mean(), "sd_t": t.std(ddof=1),
                         "share_abs_t_gt_1.96": n_sig / n if n else np.nan,
                         "ci95_lo": ci.low if ci else np.nan, "ci95_hi": ci.high if ci else np.nan,
                         "share_t_gt_1.96": float((t > 1.96).mean()),
                         "share_t_lt_-1.96": float((t < -1.96).mean()),
                         "share_abs_t_gt_3": float((t.abs() > 3).mean())})
    return pd.DataFrame(rows)


def checks(summary: pd.DataFrame) -> pd.DataFrame:
    """Pre-declared pass criteria (task.md 4.4)."""
    s = summary.set_index(["world", "stat"])
    get = lambda w, st, c: float(s.loc[(w, st), c]) if (w, st) in s.index else np.nan
    rows = []
    def add(world, crit, value, rule, ok):
        rows.append({"world": world, "criterion": crit, "value": value, "rule": rule, "pass": bool(ok)})
    v = get("V", "fm2_lnA_C", "share_t_gt_1.96")
    add("V", "share t(lnA_C, model 2) > 1.96", v, f">= {PASS_SHARE_SIG}", v >= PASS_SHARE_SIG)
    for st in ["fm3_ResLnA", "fm4_ResLnA"]:               # unpriced: two-sided, any significance counts
        v = get("V", st, "share_abs_t_gt_1.96")
        add("V", f"share |t({st})| > 1.96 (should be unpriced)", v, f"<= {MAX_SHARE_FALSE}", v <= MAX_SHARE_FALSE)
    v = get("V", "hl_A_C", "share_t_gt_1.96")
    add("V", "share t(H-L A_C sort) > 1.96", v, f">= {PASS_SHARE_SIG}", v >= PASS_SHARE_SIG)
    v = get("V", "hl_ResA", "share_abs_t_gt_1.96")
    add("V", "share |t(H-L ResA sort)| > 1.96 (should be unpriced)", v, f"<= {MAX_SHARE_FALSE}", v <= MAX_SHARE_FALSE)
    for st in ["fm3_ResLnA", "fm4_ResLnA", "hl_ResA"]:
        v = get("P", st, "share_t_gt_1.96")
        add("P", f"share t({st}) > 1.96", v, f">= {PASS_SHARE_SIG}", v >= PASS_SHARE_SIG)
    for st in ["fm2_lnA_C", "fm4_lnA_C"]:
        v = get("P", st, "share_abs_t_gt_1.96")
        add("P", f"share |t({st})| > 1.96 (ideally unpriced)", v, f"<= {MAX_SHARE_FALSE}", v <= MAX_SHARE_FALSE)
    for st in [k for k in STAT_KEYS if not k.endswith("_nw0")]:
        v, lo, hi = (get("0", st, c) for c in ["share_abs_t_gt_1.96", "ci95_lo", "ci95_hi"])
        add("0", f"false-positive rate |t({st})| > 1.96 [95% CI {lo:.3f}, {hi:.3f}]", v,
            "CI contains 0.05", lo <= 0.05 <= hi)
    return pd.DataFrame(rows)


def _git_sha() -> str:
    try:
        root = Path(__file__).resolve().parent.parent
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("config")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-sims", type=int, default=None, help="world-0 replications (default: config)")
    ap.add_argument("--reps", type=int, default=50, help="replications for worlds V and P")
    ap.add_argument("--power-reps", type=int, default=20)
    ap.add_argument("--power-grid", type=str, default="0,0.0025,0.005,0.01")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args(argv)

    for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
        os.environ.setdefault(v, "1")
    cfg_bytes = Path(a.config).read_bytes()
    cfg = yaml.safe_load(cfg_bytes)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n0_cfg = int(cfg["simulation"]["n_sims_world0"])
    n0 = a.n_sims or n0_cfg
    notes = []
    if n0 < n0_cfg:
        notes.append(f"world 0 replications reduced from {n0_cfg} (config) to {n0} via --n-sims")
    grid = [float(x) for x in a.power_grid.split(",")]

    batches = [("V", [("V", cfg, r, None) for r in range(a.reps)]),
               ("P", [("P", cfg, r, None) for r in range(a.reps)]),
               ("0", [("0", cfg, r, None) for r in range(n0)]),
               ("power_V", [("V", cfg, 1000 + r, k) for k in grid for r in range(a.power_reps)])]
    import multiprocessing as mp
    results, timing = {}, {}
    t_all = time.perf_counter()
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for name, jobs in batches:
            t0 = time.perf_counter()
            results[name] = pd.DataFrame(pool.map(_job, jobs, chunksize=1))
            timing[name] = round(time.perf_counter() - t0, 1)
            n_err = int(results[name]["error"].notna().sum()) if "error" in results[name] else 0
            print(f"[{name}] {len(jobs)} reps, {timing[name]} s, errors: {n_err}", flush=True)
    wall = round(time.perf_counter() - t_all, 1)

    worlds = pd.concat([results["V"], results["P"], results["0"]], ignore_index=True)
    worlds.to_csv(out / "sim_world_reps.csv", index=False)
    results["0"].to_csv(out / "sim_world0_reps.csv", index=False)
    summary = summarize(worlds, ["world"])
    summary.to_csv(out / "sim_world_summary.csv", index=False)
    results["power_V"].to_csv(out / "sim_power_reps.csv", index=False)
    power = summarize(results["power_V"], ["premium"])
    power = power[power["stat"].isin(["fm2_lnA_C", "fm4_lnA_C", "hl_A_C"])]
    power.insert(0, "world", "V")
    power.to_csv(out / "sim_power.csv", index=False)
    ok = worlds[worlds["error"].isna()] if "error" in worlds else worlds
    ok.groupby("world", sort=False)[DIAG_KEYS].mean().reset_index().to_csv(out / "sim_world_diagnostics.csv", index=False)
    chk = checks(summary)
    chk.to_csv(out / "sim_checks.csv", index=False)

    errors = {n: int(r["error"].notna().sum()) if "error" in r else 0 for n, r in results.items()}
    manifest = {
        "run_id": f"sim_{time.strftime('%Y%m%d_%H%M%S')}",
        "seed": int(cfg["seed"]), "seed_scheme": "SeedSequence([seed, world_code, rep, round(premium*1e6)+1e6 or 999999])",
        "git_sha": _git_sha(), "config_sha256": hashlib.sha256(cfg_bytes).hexdigest(),
        "code_sha256_src": hashlib.sha256(b"".join(f.read_bytes() for f in
                                                    sorted(Path(__file__).resolve().parent.glob("*.py")))).hexdigest(),
        "git_dirty": bool(subprocess.run(["git", "status", "--porcelain", "src"], capture_output=True, text=True,
                                         cwd=Path(__file__).resolve().parent.parent).stdout.strip()),
        "n_reps": {n: len(r) for n, r in results.items()}, "errors": errors,
        "n_sims_world0_config": n0_cfg, "power_grid": grid,
        "runtime_sec": timing, "wall_sec": wall, "workers": a.workers,
        "n_assets": int(cfg["simulation"]["n_assets"]), "n_months": int(cfg["simulation"]["n_months"]),
        "premium_per_sd": float(cfg["simulation"]["premium_per_sd"]),
        "dgp": dgp_params(cfg), "world0_structure": "world P returns (|r|-volume link), mu = 0",
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "notes": notes, "checks_passed": int(chk["pass"].sum()), "checks_total": len(chk)}
    (out / "sim_manifest.json").write_text(json.dumps(manifest, indent=2))
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary[["world", "stat", "n_reps", "mean_coef", "mean_t", "share_abs_t_gt_1.96",
                       "ci95_lo", "ci95_hi", "share_t_gt_1.96", "share_abs_t_gt_3"]].round(3).to_string(index=False))
        print(power[["premium", "stat", "n_reps", "mean_t", "share_t_gt_1.96", "share_abs_t_gt_3"]].round(3).to_string(index=False))
        print(chk.to_string(index=False))
    print(f"wall {wall} s; notes: {notes}")


if __name__ == "__main__":
    main()
