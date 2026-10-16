"""One reproducible run: daily data -> audit -> panel -> tests (+ placebos) -> results/<run_id>/.

Usage:  python -I src/run.py config.yaml [--daily data/cache/daily.parquet] [--tag TAG]
                                         [--results-dir results] [--cache-dir data/cache]
                                         [--synthetic N] [--no-placebo]
Writes: results/<run_id>/*.csv (incl. audit_events.csv, audit_summary.csv, ret_distribution.csv,
        ret_extremes.csv, ret_influence.csv), run_manifest.json; results/LATEST (= run_id);
        the full panel only to <cache-dir>/panel_<run_id>.parquet.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import copy
import hashlib
import importlib
import importlib.metadata as im
import inspect
import json
import logging
import platform
import resource
import subprocess
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml

import analysis
import data_audit
from sorts import portfolio_returns

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PACKAGES = ["pandas", "numpy", "statsmodels", "matplotlib", "scipy", "PyYAML", "pyarrow", "requests", "pytest"]
log = logging.getLogger("run")


# ---------------------------------------------------------------- provenance helpers
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_hash(src: Path = SRC) -> str:
    """sha256 over (file name, bytes) of src/*.py in sorted order."""
    h = hashlib.sha256()
    for p in sorted(src.glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def rel_path(path) -> str:
    """Path as written into manifests/CSVs/SUMMARY: relative to the repo root (no user name, portable).
    A location outside the repo (e.g. a test's tmp dir) becomes '../..'-relative to the root, so that
    resolve_repo_path() still round-trips it on the same machine."""
    import os
    p = Path(path)
    p = (p if p.is_absolute() else Path.cwd() / p).resolve()
    root = ROOT.resolve()
    try:
        return p.relative_to(root).as_posix()
    except ValueError:
        try:
            return Path(os.path.relpath(p, root)).as_posix()
        except ValueError:                       # other drive (Windows): nothing relative exists
            return p.name


def resolve_repo_path(p) -> Path:
    """Inverse of rel_path for paths read back from manifests (repo-relative; absolute in old ones)."""
    q = Path(p)
    return q if q.is_absolute() else ROOT / q


def resolve_panel_cache(manifest: dict, run_id: str, cache_dir: Path | None = None) -> Path:
    """Base-run panel parquet: manifest 'panel_cache' (repo-relative; absolute in older manifests),
    falling back to <cache_dir or data/cache>/panel_<run_id>.parquet when that path does not exist."""
    p = manifest.get("panel_cache")
    if p:
        q = Path(p)
        q = q if q.is_absolute() else ROOT / q
        if q.exists():
            return q
    return Path(cache_dir or ROOT / "data" / "cache") / f"panel_{run_id}.parquet"


def frame_hash(df: pd.DataFrame) -> str:
    """Content hash of an in-memory frame (used for --synthetic data)."""
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes()).hexdigest()


def git_info(root: Path = ROOT) -> dict:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        st = subprocess.run(["git", "status", "--porcelain", "--", "src", "tests", "config.yaml", "Makefile", "mk", "requirements.txt"], cwd=root, capture_output=True, text=True, check=True).stdout
        return {"sha": sha, "dirty": bool(st.strip()), "dirty_scope": "code+config (src, tests, config.yaml, Makefile, mk, requirements.txt); generated outputs excluded"}
    except (OSError, subprocess.CalledProcessError):
        return {"sha": None, "dirty": None}


def versions() -> dict:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for p in PACKAGES:
        try:
            out[p] = im.version(p)
        except im.PackageNotFoundError:
            out[p] = None
    return out


def peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 2**10   # bytes on macOS, KiB on Linux


# ---------------------------------------------------------------- data
OPTIONAL_DAILY = ["open", "high", "low", "volume", "n_trades"]     # used by the data audit


def load_daily(path: Path) -> pd.DataFrame:
    import pyarrow.parquet as pq
    have = set(pq.read_schema(path).names)
    cols = ["symbol", "date", "close", "quote_volume"] + [c for c in OPTIONAL_DAILY if c in have]
    d = pd.read_parquet(path, columns=cols)
    d["symbol"] = d["symbol"].astype(str)
    d["date"] = pd.to_datetime(d["date"]).dt.tz_localize(None).dt.normalize()
    for c in cols[2:]:
        d[c] = d[c].astype(float)
    return d.sort_values(["symbol", "date"]).reset_index(drop=True)


def synthetic_daily(n_symbols: int = 80, start: str = "2018-01-01", end: str = "2020-06-30",
                    seed: int = 0) -> pd.DataFrame:
    """Toy daily panel with staggered listings/delistings; for smoke tests only (no premium built in)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    T = len(dates)
    mkt = rng.normal(0.0, 0.03, T)
    rows = []
    for i in range(n_symbols):
        a, b = sorted(rng.integers(0, T, 2)) if rng.random() < 0.3 else (0, T)
        if b - a < 90:
            a, b = 0, T
        n = b - a
        beta, sig, lvl = rng.uniform(0.5, 1.5), rng.uniform(0.02, 0.08), rng.normal(13.0, 2.0)
        r = beta * mkt[a:b] + sig * rng.standard_normal(n)
        close = 10.0 * np.exp(np.cumsum(np.log1p(np.clip(r, -0.9, None))))
        dvol = np.exp(lvl + 0.5 * rng.standard_normal(n) + 5.0 * np.abs(r))
        keep = rng.random(n) > 0.01                       # a few missing days (gaps)
        rows.append(pd.DataFrame({"symbol": f"C{i:03d}USDT", "date": dates[a:b][keep],
                                  "close": close[keep], "quote_volume": dvol[keep]}))
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------- outputs
def _csv_ready(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index()
    for c in df.columns:
        if isinstance(df[c].dtype, pd.PeriodDtype):
            df[c] = df[c].astype(str)
    return df


def panel_summary(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-month counts and coverage (the full panel is not stored in results/)."""
    g = panel.groupby("month")
    s = pd.DataFrame({"n_coins": g.size(),
                      "n_ret_adj": g["ret_adj"].count(),
                      "n_lnA": g["lnA"].count(),
                      "n_mom": g["mom"].count(),
                      "n_ivol": g["ivol"].count(),
                      "mean_ret": g["ret"].mean(),
                      "median_dvol": g["dvol"].median()})
    return s.reset_index()


def quintile_portfolios(panel: pd.DataFrame, q: int) -> pd.DataFrame:
    """Long table month, signal, Q1..Qq, HL: monthly EW raw returns of the T3 sorts (for figures)."""
    out = []
    for s in ["A", "A_C", "ResA"]:
        pr = portfolio_returns(panel, s, "ret", q)
        pr.columns = [f"Q{c}" for c in pr.columns]
        pr["HL"] = pr[f"Q{q}"] - pr["Q1"]
        pr.insert(0, "signal", s)
        out.append(pr.reset_index(names="month"))
    return pd.concat(out, ignore_index=True)


def run_placebos(panel, factors, cfg, daily, start, end, seed):
    """Call placebo.run_placebos if the module exists; pass a lag-0 panel when it accepts one."""
    try:
        placebo = importlib.import_module("placebo")
    except ImportError as e:
        log.warning("placebo module not available, skipping placebos (%s)", e)
        return {}
    kwargs = {}
    if "panel_lag0" in inspect.signature(placebo.run_placebos).parameters:
        cfg0 = copy.deepcopy(cfg)
        cfg0["measures"]["lag_months"] = 0
        kwargs["panel_lag0"] = analysis.build_panel(daily, cfg0, start, end)[0]
    res = placebo.run_placebos(panel, factors, cfg, seed, **kwargs)
    if isinstance(res, pd.DataFrame):
        return {"placebo": res}
    return {k if k.startswith("P") else f"placebo_{k}": v for k, v in res.items()}


def main(argv: list[str] | None = None) -> str:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--daily", default=str(ROOT / "data" / "cache" / "daily.parquet"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    ap.add_argument("--cache-dir", default=str(ROOT / "data" / "cache"))
    ap.add_argument("--synthetic", type=int, default=0, metavar="N",
                    help="ignore --daily and generate N synthetic coins (smoke tests only)")
    ap.add_argument("--no-placebo", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg_path = Path(a.config)
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg_hash = sha256_file(cfg_path)
    seed = int(cfg["seed"])
    start, end = cfg["period"]["return_start"], cfg["period"]["return_end"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{stamp}_{cfg_hash[:8]}" + (f"_{a.tag}" if a.tag else "")
    out = Path(a.results_dir) / run_id
    out.mkdir(parents=True, exist_ok=False)
    timings: dict[str, float] = {}
    tables: dict[str, pd.DataFrame] = {}

    def stage(name):
        class _T:
            def __enter__(self):
                self.t = time.perf_counter()
                log.info("stage %s ...", name)
            def __exit__(self, *exc):
                timings[name] = round(time.perf_counter() - self.t, 3)
                log.info("stage %s done in %.1fs", name, timings[name])
        return _T()

    with stage("load"):
        if a.synthetic:
            daily = synthetic_daily(a.synthetic, seed=seed)
            data_src, data_hash = f"synthetic:{a.synthetic}", frame_hash(daily)
        else:
            daily = load_daily(Path(a.daily))
            data_src, data_hash = rel_path(a.daily), sha256_file(Path(a.daily))
    audit_info = {"enabled": False}
    if data_audit.audit_config(cfg)["enabled"]:
        with stage("audit"):
            n_raw = len(daily)
            daily, events, audit_summary = data_audit.audit(daily, cfg)
            tables["audit_events"], tables["audit_summary"] = events, audit_summary
            audit_info = {"enabled": True, "rules": data_audit.audit_config(cfg),
                          "events_by_type": {k: int(v) for k, v in events["type"].value_counts().items()},
                          "daily_rows_raw": int(n_raw), "daily_rows_clean": int(len(daily))}
    with stage("build_panel"):
        panel, factors = analysis.build_panel(daily, cfg, start, end)
    with stage("run_tests"):
        tables.update(analysis.run_tests(panel, factors, cfg))
        tables.update(data_audit.return_diagnostics(panel, daily, cfg["tests"]["quantiles"]))
        tables["T3_portfolios"] = quintile_portfolios(panel, cfg["tests"]["quantiles"])
        tables["panel_summary"] = panel_summary(panel)
    if not a.no_placebo:
        with stage("placebo"):
            tables.update(run_placebos(panel, factors, cfg, daily, start, end, seed))
    with stage("write"):
        for name, df in tables.items():
            _csv_ready(df).to_csv(out / f"{name}.csv", index=False)
        cache = Path(a.cache_dir)
        cache.mkdir(parents=True, exist_ok=True)
        _csv_ready(panel).to_parquet(cache / f"panel_{run_id}.parquet", index=False)

    months = panel["month"]
    manifest = {
        "run_id": run_id,
        "created_utc": stamp,
        "tag": a.tag,
        "git": git_info(),
        "config_path": rel_path(cfg_path),
        "config_sha256": cfg_hash,
        "code_sha256": code_hash(),
        "data_source": data_src,
        "data_sha256": data_hash,
        "seed": seed,
        "period": {"return_start": start, "return_end": end,
                   "first_panel_month": str(months.min()), "last_panel_month": str(months.max())},
        "versions": versions(),
        "timings_sec": timings | {"total": round(sum(timings.values()), 3)},
        "audit": audit_info,
        "rows": {"daily_rows": int(len(daily)), "daily_symbols": int(daily["symbol"].nunique()),
                 "panel_coin_months": int(len(panel)), "panel_coins": int(panel["symbol"].nunique()),
                 "panel_months": int(months.nunique())},
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "tables": sorted(f"{k}.csv" for k in tables),
        "panel_cache": rel_path(Path(a.cache_dir) / f"panel_{run_id}.parquet"),
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    (Path(a.results_dir) / "LATEST").write_text(run_id + "\n")
    log.info("run %s written to %s", run_id, out)
    print(run_id)
    return run_id


if __name__ == "__main__":
    main()
