"""Upgrade 2: high-frequency price-impact benchmark lambda with EXACT trade signs.

Lou & Shu (2017), eq. (11) and Appendix A:  r_n = lambda * SVol_n + u_n  over the 5-minute periods n
of a stock-month, SVol_n = sum_k sign_k * sqrt(dvol_k), signs from Lee-Ready (their weak spot after
2006).  On Binance the aggressor side is recorded by the exchange, so signs are exact:

* bar level (main sample): 5m spot klines carry taker-buy quote volume B_n and quote volume Q_n,
  hence S_n = Q_n - B_n.  Pre-registered proxy (MAIN):   SVol_n = sqrt(B_n) - sqrt(S_n)
  (each side treated as one aggregated trade; equals the trade-level sum when a bar has one trade
  per side).  Robustness (ALT):  SVol_n = sign(B_n - S_n) * sqrt(|B_n - S_n|)  (net order flow).
* trade level (validation subsample): aggTrades, sign_k = -1 if isBuyerMaker else +1, dvol_k = p_k*q_k,
  SVol_n = sum_k sign_k sqrt(dvol_k)  -- exactly the paper's formula with exact signs.

r_n = ln(P_n / P_{n-1}), P_n = last trade price of bar n (kline close; carried forward through bars
without trades).  The paper uses the quote midpoint; Binance has no public historical spot quotes, so
we use trade prices (bid-ask bounce: see SUMMARY caveats).  A bar is used if Q_n > 0, bar n-1 exists
and |r_n| <= MAX_ABS_R (redenomination / bad-print guard).  lambda = OLS slope WITHOUT intercept, as
in eq. (11).  A coin-month needs >= MIN_BARS such bars (paper: >= 10; we require far more).
Deviation from the paper (footnote 16 drops negative lambdas after taking logs): lambda is winsorized
1/99 within each month FIRST, so a lambda <= 0 below the 1st percentile is raised to it and kept
(3 coin-months in the main run; counted in coverage.csv as n_lambda_nonpos).

Then: Table 9 analogue (Panel A correlations of lambda with A, A_C, Res.A; Panel B Fama-MacBeth of
CMKT-adjusted returns on ln lambda [+ ln A_C], controls mom, ret_1, signals at t-2, NW 6 lags) and
Table 10 analogue (ln A split into the fitted "cost" part from monthly cross-sectional regressions on
ln lambda [and ln EDGE, if data/cache/edge_monthly.parquet exists] and the residual "non-cost" part).

Usage:  python -I src/hf_lambda.py config.yaml [--plan] [--download-only] [--workers N]
Reads:  results/LATEST -> data/cache/panel_<run_id>.parquet (base panel; never hard-coded)
Writes: data/raw/klines5m/<SYM>/*.zip, data/raw/aggtrades/<SYM>/*.zip (cached, idempotent),
        results/upgrades/hf_lambda/*.csv, run_manifest.json, SUMMARY.md
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import hashlib
import io
import json
import logging
import threading
import time
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import yaml

from run import rel_path, resolve_panel_cache

ROOT = Path(__file__).resolve().parents[1]
RAW_K5 = ROOT / "data" / "raw" / "klines5m"
RAW_AGG = ROOT / "data" / "raw" / "aggtrades"
CACHE = ROOT / "data" / "cache"
OUT = ROOT / "results" / "upgrades" / "hf_lambda"
LISTING_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
DATA_URL = "https://data.binance.vision"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
BAR_MS = 300_000
log = logging.getLogger("hf_lambda")

SPEC = {
    "measure_start": "2023-08",
    "measure_end": "2026-07",
    "svol_main": "sqrt_sides",
    "svol_alt": "net_sqrt",
    "min_bars": 1000,
    "min_bars_robust": 100,
    "max_abs_r": 0.5,
    "winsor": [0.01, 0.99],
    "validation_months": ["2024-06", "2025-06"],
    "validation_per_quintile": 7,
    "aggtrades_max_mb": 300,
    "download_workers": 24,
}

PAPER = {
    "T9A": {"A": 0.737, "A_C": 0.747, "ResA": 0.149},
    "T9B": {"(1)": {"lnlam": (-0.034, -0.67)},
            "(2)": {"lnlam": (-0.155, -2.51), "lnA_C": (0.180, 3.80)}},
    "T10": {"(1)": {"cost": (-0.047, -1.11)},
            "(2)": {"noncost": (0.186, 3.04)},
            "(3)": {"cost": (0.015, 0.35), "noncost": (0.191, 3.10)}},
    "sample": "NYSE/AMEX, 1983-2012 (Table 9: 358 months, ~1725 stocks); Table 10: 1964-2012",
}


def svol_bar(B: np.ndarray, Q: np.ndarray, kind: str) -> np.ndarray:
    """Bar-level signed sqrt dollar volume from taker-buy quote volume B and total quote volume Q."""
    B = np.asarray(B, float)
    S = np.clip(np.asarray(Q, float) - B, 0.0, None)
    B = np.clip(B, 0.0, None)
    if kind == "sqrt_sides":
        return np.sqrt(B) - np.sqrt(S)
    if kind == "net_sqrt":
        d = B - S
        return np.sign(d) * np.sqrt(np.abs(d))
    raise ValueError(kind)


def bar_returns(bar: np.ndarray, close: np.ndarray) -> np.ndarray:
    """r_n = ln(P_n / P_{n-1}) if bar n-1 is the immediately preceding 5m bar, else NaN."""
    bar = np.asarray(bar, np.int64)
    close = np.asarray(close, float)
    r = np.full(len(close), np.nan)
    if len(close) > 1:
        ok = (np.diff(bar) == 1) & (close[1:] > 0) & (close[:-1] > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            r[1:] = np.where(ok, np.log(close[1:] / close[:-1]), np.nan)
    return r


def estimate_lambda(r: np.ndarray, x: np.ndarray, use: np.ndarray | None = None) -> dict:
    """OLS of r on x without intercept (eq. 11). Returns lam, se, t, r2 (uncentred), n."""
    r, x = np.asarray(r, float), np.asarray(x, float)
    m = np.isfinite(r) & np.isfinite(x)
    if use is not None:
        m &= use
    r, x = r[m], x[m]
    n = int(len(r))
    sxx = float((x * x).sum())
    if n < 2 or sxx <= 0:
        return {"lam": np.nan, "se": np.nan, "t": np.nan, "r2": np.nan, "n": n}
    lam = float((x * r).sum() / sxx)
    u = r - lam * x
    se = float(np.sqrt((u * u).sum() / (n - 1) / sxx))
    syy = float((r * r).sum())
    return {"lam": lam, "se": se, "t": lam / se if se > 0 else np.nan,
            "r2": 1 - float((u * u).sum()) / syy if syy > 0 else np.nan, "n": n}


def usable_bars(r: np.ndarray, Q: np.ndarray, max_abs_r: float) -> np.ndarray:
    return np.isfinite(r) & (np.asarray(Q, float) > 0) & (np.abs(r) <= max_abs_r)


def lambdas_from_bars(bars: pd.DataFrame, max_abs_r: float, exact: bool = False) -> dict:
    """bars: bar, close, Q, B [, svol_exact], sorted by bar. Returns lambdas for each SVol definition."""
    r = bar_returns(bars["bar"].to_numpy(), bars["close"].to_numpy())
    Q, B = bars["Q"].to_numpy(float), bars["B"].to_numpy(float)
    use = usable_bars(r, Q, max_abs_r)
    out = {"n_bars_total": int(len(bars)), "n_bars_traded": int((Q > 0).sum()),
           "n_extreme": int((np.isfinite(r) & (Q > 0) & (np.abs(r) > max_abs_r)).sum()),
           "quote_volume": float(Q.sum()), "taker_buy_share": float(B.sum() / Q.sum()) if Q.sum() > 0 else np.nan}
    kinds = ["sqrt_sides", "net_sqrt"] + (["exact"] if exact else [])
    for k in kinds:
        x = bars["svol_exact"].to_numpy(float) if k == "exact" else svol_bar(B, Q, k)
        est = estimate_lambda(r, x, use)
        for key, v in est.items():
            out[f"{key}_{k}" if key != "n" else "n_bars"] = v
    return out


def _csv_frame(raw: bytes | io.IOBase, usecols: list[int], names: list[str], **kw) -> pd.DataFrame:
    df = pd.read_csv(raw, header=None, usecols=usecols, names=names, dtype=str, **kw)
    return df


def read_kline_zip(path: Path) -> pd.DataFrame:
    """Binance spot kline zip -> bar, close, Q, B, n_trades (bar = open_time_ms // 300000)."""
    with zipfile.ZipFile(path) as zf:
        raw = zf.read(zf.namelist()[0])
    df = pd.read_csv(io.BytesIO(raw), header=None, usecols=[0, 4, 7, 8, 10],
                     names=["open_time", "close", "Q", "n_trades", "B"], dtype=str)
    df = df.apply(pd.to_numeric, errors="coerce").dropna(subset=["open_time", "close"])
    t = df["open_time"].astype("int64")
    t = t.where(t < 10**14, t // 1000)
    out = pd.DataFrame({"bar": (t // BAR_MS).astype("int64"), "close": df["close"].astype(float),
                        "Q": df["Q"].fillna(0.0).astype(float), "B": df["B"].fillna(0.0).astype(float),
                        "n_trades": df["n_trades"].fillna(0).astype(float)})
    return out.drop_duplicates("bar").sort_values("bar").reset_index(drop=True)


def _bool(s: pd.Series) -> np.ndarray:
    return s.astype(str).str.strip().str.lower().isin(["true", "1"]).to_numpy()


def read_aggtrades_zip(path: Path, chunksize: int = 2_000_000) -> pd.DataFrame:
    """aggTrades zip -> 5m bars: bar, close (last price), Q, B, svol_exact, n_agg; streamed in chunks.

    Columns: agg_id, price, qty, first_id, last_id, time, is_buyer_maker, is_best_match.
    is_buyer_maker = True -> the taker sold (seller-initiated) -> sign -1.
    """
    parts = []
    with zipfile.ZipFile(path) as zf, zf.open(zf.namelist()[0]) as fh:
        reader = pd.read_csv(fh, header=None, usecols=[0, 1, 2, 5, 6], names=["id", "p", "q", "ts", "bm"],
                             dtype=str, chunksize=chunksize)
        for ch in reader:
            num = ch[["id", "p", "q", "ts"]].apply(pd.to_numeric, errors="coerce")
            ok = num.notna().all(axis=1).to_numpy()
            if not ok.any():
                continue
            num, bm = num[ok], _bool(ch.loc[ok, "bm"])
            ts = num["ts"].astype("int64").to_numpy()
            ts = np.where(ts < 10**14, ts, ts // 1000)
            order = np.lexsort((num["id"].to_numpy(), ts))
            p, q = num["p"].to_numpy(float)[order], num["q"].to_numpy(float)[order]
            ts, bm = ts[order], bm[order]
            dv = p * q
            sign = np.where(bm, -1.0, 1.0)
            g = pd.DataFrame({"bar": ts // BAR_MS, "close": p, "Q": dv, "B": np.where(sign > 0, dv, 0.0),
                              "svol_exact": sign * np.sqrt(dv), "n_agg": 1})
            parts.append(g.groupby("bar", sort=True).agg(close=("close", "last"), Q=("Q", "sum"), B=("B", "sum"),
                                                         svol_exact=("svol_exact", "sum"), n_agg=("n_agg", "sum")))
    if not parts:
        return pd.DataFrame(columns=["bar", "close", "Q", "B", "svol_exact", "n_agg"])
    bars = pd.concat(parts).groupby(level=0, sort=True).agg(
        close=("close", "last"), Q=("Q", "sum"), B=("B", "sum"), svol_exact=("svol_exact", "sum"), n_agg=("n_agg", "sum"))
    return bars.reset_index(names="bar")


def complete_grid(bars: pd.DataFrame, lo: int | None = None, hi: int | None = None) -> pd.DataFrame:
    """Insert bars without trades (Q = B = svol = 0, close carried forward), as 5m klines have them."""
    if bars.empty:
        return bars
    lo = int(bars["bar"].min()) if lo is None else lo
    hi = int(bars["bar"].max()) if hi is None else hi
    g = bars.set_index("bar").reindex(np.arange(lo, hi + 1))
    g["close"] = g["close"].ffill()
    for c in ["Q", "B", "svol_exact", "n_agg"]:
        if c in g:
            g[c] = g[c].fillna(0.0)
    return g.dropna(subset=["close"]).reset_index(names="bar")


def month_bar_range(month: str) -> tuple[int, int]:
    p = pd.Period(month, "M")
    a = int(p.start_time.value // 10**6) // BAR_MS
    b = int((p + 1).start_time.value // 10**6) // BAR_MS - 1
    return a, b


def kline_month_lambda(path: str, max_abs_r: float) -> dict:
    """Worker: one 5m kline file -> lambda estimates (picklable for the process pool)."""
    p = Path(path)
    sym, month = p.parent.name, p.stem[-7:]
    try:
        bars = read_kline_zip(p)
        lo, hi = month_bar_range(month)
        bars = bars[(bars["bar"] >= lo) & (bars["bar"] <= hi)]
        res = lambdas_from_bars(bars, max_abs_r)
        res["error"] = ""
    except Exception as e:
        res = {"error": f"{type(e).__name__}: {e}"[:200]}
    return {"symbol": sym, "measure_month": month, **res}


def aggtrades_month_lambda(path: str, kline_path: str | None, max_abs_r: float) -> dict:
    """Worker: one aggTrades file -> exact trade-level lambda + bar proxies from the same trades,
    and the kline-based proxies of the same coin-month (if the kline file exists)."""
    p = Path(path)
    sym, month = p.parent.name, p.stem[-7:]
    t0 = time.perf_counter()
    lo, hi = month_bar_range(month)
    agg = read_aggtrades_zip(p)
    agg = agg[(agg["bar"] >= lo) & (agg["bar"] <= hi)]
    agg = complete_grid(agg, None, hi)
    res = {"symbol": sym, "measure_month": month}
    res.update({f"agg_{k}": v for k, v in lambdas_from_bars(agg, max_abs_r, exact=True).items()})
    res["agg_n_trades"] = float(agg["n_agg"].sum()) if len(agg) else 0.0
    if kline_path and Path(kline_path).exists():
        kl = read_kline_zip(Path(kline_path))
        kl = kl[(kl["bar"] >= lo) & (kl["bar"] <= hi)]
        res.update({f"kl_{k}": v for k, v in lambdas_from_bars(kl, max_abs_r).items()})
        m = kl.merge(agg[["bar", "Q", "B", "close"]], on="bar", suffixes=("_kl", "_agg"))
        m = m[m["Q_kl"] > 0]
        res["integrity_n_bars"] = int(len(m))
        res["integrity_med_absrel_Q"] = float((m["Q_agg"] / m["Q_kl"] - 1).abs().median()) if len(m) else np.nan
        res["integrity_med_absrel_B"] = float(((m["B_agg"] - m["B_kl"]).abs() / m["Q_kl"]).median()) if len(m) else np.nan
        res["integrity_share_close_equal"] = float(np.isclose(m["close_agg"], m["close_kl"], rtol=1e-9).mean()) if len(m) else np.nan
    res["sec"] = round(time.perf_counter() - t0, 2)
    return res


_tls = threading.local()


def _session() -> requests.Session:
    if not hasattr(_tls, "s"):
        _tls.s = requests.Session()
    return _tls.s


def _get(url: str, params: dict | None = None, tries: int = 6, stream: bool = False) -> requests.Response:
    for k in range(tries):
        try:
            r = _session().get(url, params=params, timeout=120, stream=stream)
            if r.status_code == 404:
                return r
            r.raise_for_status()
            return r
        except requests.RequestException:
            if k == tries - 1:
                raise
            time.sleep(2 ** k)


def list_sizes(prefix: str) -> dict[str, int]:
    """S3 listing (paginated) -> {key: size} for .zip keys under prefix."""
    out, marker = {}, ""
    while True:
        r = _get(LISTING_URL, {"prefix": prefix, "marker": marker})
        root = ET.fromstring(r.content)
        keys = []
        for c in root.findall("s3:Contents", NS):
            k = c.findtext("s3:Key", namespaces=NS)
            keys.append(k)
            if k.endswith(".zip"):
                out[k] = int(c.findtext("s3:Size", namespaces=NS))
        if root.findtext("s3:IsTruncated", namespaces=NS) != "true" or not keys:
            return out
        marker = root.findtext("s3:NextMarker", namespaces=NS) or keys[-1]


def aggtrades_listing(symbols: list[str], months: list[str], workers: int) -> dict[str, int]:
    """aggTrades file sizes for symbols x months; the S3 listing is cached in data/raw/aggtrades/listing.json
    so that the validation sample does not change when Binance adds files later (delete it to refresh)."""
    path = RAW_AGG / "listing.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    todo = [s for s in symbols if s not in cache.get("listed_symbols", [])]
    if todo:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            res = list(ex.map(lambda s: list_sizes(f"data/spot/monthly/aggTrades/{s}/{s}-aggTrades-"), todo))
        sizes = cache.get("sizes", {}) | {k: v for d in res for k, v in d.items()}
        cache = {"listed_symbols": sorted(set(cache.get("listed_symbols", [])) | set(todo)), "sizes": sizes,
                 "listed_utc": cache.get("listed_utc", []) + [datetime.now(timezone.utc).isoformat(timespec="seconds")]}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache))
    want = {agg_key(s, m) for s in symbols for m in months}
    return {k: int(v) for k, v in cache["sizes"].items() if k in want}


def raw_symbol(sym: str) -> str:
    """Binance symbol of a panel asset id: the data audit relabels relisted/redenominated segments as
    <SYMBOL>#k (src/data_audit.py); their trades are in the files of <SYMBOL>."""
    return str(sym).split("#")[0]


def kline_key(sym: str, month: str) -> str:
    sym = raw_symbol(sym)
    return f"data/spot/monthly/klines/{sym}/5m/{sym}-5m-{month}.zip"


def agg_key(sym: str, month: str) -> str:
    sym = raw_symbol(sym)
    return f"data/spot/monthly/aggTrades/{sym}/{sym}-aggTrades-{month}.zip"


def local_path(key: str) -> Path:
    base = RAW_AGG if "/aggTrades/" in key else RAW_K5
    return base / key.split("/")[4] / Path(key).name


def download(key: str) -> dict:
    """Download key (+ verify its .CHECKSUM sha256) into data/raw; skip if already present."""
    dst = local_path(key)
    if dst.exists() and dst.stat().st_size > 0:
        return {"key": key, "status": "cached", "bytes": dst.stat().st_size}
    url = f"{DATA_URL}/{key}"
    for attempt in range(5):
        tmp = dst.with_suffix(".part")
        try:
            r = _get(url, stream=True)
            if r.status_code == 404:
                return {"key": key, "status": "missing", "bytes": 0}
            dst.parent.mkdir(parents=True, exist_ok=True)
            h = hashlib.sha256()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
                    h.update(chunk)
            c = _get(url + ".CHECKSUM")
        except requests.RequestException:
            tmp.unlink(missing_ok=True)
            time.sleep(2 ** attempt)
            continue
        want = c.text.split()[0].strip().lower() if c.status_code == 200 and c.text.strip() else None
        if want is None or want == h.hexdigest():
            tmp.rename(dst)
            return {"key": key, "status": "downloaded" if want else "downloaded_nochecksum", "bytes": dst.stat().st_size}
        tmp.unlink(missing_ok=True)
    return {"key": key, "status": "failed_after_retries", "bytes": 0}


def download_many(keys: list[str], workers: int) -> pd.DataFrame:
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, res in enumerate(ex.map(download, keys)):
            rows.append(res)
            if i % 500 == 0:
                log.info("  download %d/%d", i, len(keys))
    return pd.DataFrame(rows, columns=["key", "status", "bytes"])


def base_run() -> tuple[str, Path]:
    run_id = (ROOT / "results" / "LATEST").read_text().strip()
    man = json.loads((ROOT / "results" / run_id / "run_manifest.json").read_text())
    path = resolve_panel_cache(man, run_id, CACHE)
    return run_id, path


def load_panel(path: Path) -> pd.DataFrame:
    p = pd.read_parquet(path)
    p["month"] = pd.PeriodIndex(p["month"].astype(str), freq="M")
    return p


def needed_coin_months(panel: pd.DataFrame, spec: dict, lag: int) -> pd.DataFrame:
    """(symbol, measure_month) pairs of the base panel inside the measure window."""
    mm = panel["month"] - lag
    s = panel.loc[(mm >= pd.Period(spec["measure_start"], "M")) & (mm <= pd.Period(spec["measure_end"], "M")),
                  ["symbol"]].copy()
    s["measure_month"] = mm[s.index].astype(str)
    return s.drop_duplicates().sort_values(["measure_month", "symbol"]).reset_index(drop=True)


def select_validation(panel: pd.DataFrame, spec: dict, lag: int, seed: int, sizes: dict[str, int]) -> pd.DataFrame:
    """Stratified validation sample: A_C quintiles of the first validation month's cross-section,
    random order within quintile (seeded), take the first k coins whose aggTrades files exist and are
    <= cap in ALL validation months; every skipped candidate is logged with its reason."""
    vm = spec["validation_months"]
    cs = panel[panel["month"] == pd.Period(vm[0], "M") + lag][["symbol", "A_C"]].dropna()
    cs = cs.assign(quintile=pd.qcut(cs["A_C"].rank(method="first"), 5, labels=False) + 1)
    rng = np.random.default_rng(seed)
    cap = spec["aggtrades_max_mb"] * 2**20
    rows = []
    for q, g in cs.groupby("quintile"):
        g = g.iloc[rng.permutation(len(g))]
        taken = 0
        for sym in g["symbol"]:
            szs = [sizes.get(agg_key(sym, m)) for m in vm]
            if taken >= spec["validation_per_quintile"]:
                status = "not_needed"
            elif any(s is None for s in szs):
                status = "skipped_missing_file"
            elif any(s > cap for s in szs):
                status = "skipped_over_cap"
            else:
                status = "selected"
                taken += 1
            rows.append({"symbol": sym, "quintile_A_C": int(q), "status": status,
                         **{f"mb_{m}": (None if s is None else round(s / 2**20, 1)) for m, s in zip(vm, szs)}})
    sel = pd.DataFrame(rows)
    return sel[sel["status"] != "not_needed"].reset_index(drop=True)


def cs_fit_resid(df: pd.DataFrame, y: str, xs: list[str], by: str) -> tuple[pd.Series, pd.Series]:
    """Monthly cross-sectional OLS of y on [1, xs]: (fitted, residual) aligned with df."""
    fit = pd.Series(np.nan, index=df.index)
    res = pd.Series(np.nan, index=df.index)
    for _, g in df[[y] + xs].groupby(df[by], sort=False):
        g = g.dropna()
        if len(g) < len(xs) + 3:
            continue
        Z = np.column_stack([np.ones(len(g))] + [g[x].to_numpy(float) for x in xs])
        b, *_ = np.linalg.lstsq(Z, g[y].to_numpy(float), rcond=None)
        fit.loc[g.index] = Z @ b
        res.loc[g.index] = g[y].to_numpy(float) - Z @ b
    return fit, res


def attach_lambda(panel: pd.DataFrame, lam: pd.DataFrame, spec: dict, lag: int, kind: str, min_bars: int,
                  winsor: tuple[float, float]) -> pd.DataFrame:
    """Merge lambda of measure month t-lag onto panel rows of month t; winsorize; ln lambda."""
    from measures import winsorize_cs
    col = f"lam_{kind}"
    L = lam[["symbol", "measure_month", col, "n_bars"]].copy()
    L = L[L["n_bars"] >= min_bars]
    L["month"] = pd.PeriodIndex(L["measure_month"], freq="M") + lag
    p = panel.merge(L[["symbol", "month", col]].rename(columns={col: "lam"}), on=["symbol", "month"], how="inner")
    p = p.dropna(subset=["lam"])
    p = winsorize_cs(p, ["lam"], "month", tuple(winsor))
    with np.errstate(divide="ignore", invalid="ignore"):
        p["lnlam"] = np.where(p["lam"] > 0, np.log(p["lam"].where(p["lam"] > 0)), np.nan)
    return p


def _fm_block(panel: pd.DataFrame, y: str, models: dict, lags: int, controls: list[str]) -> pd.DataFrame:
    """analysis._fm_block, but a model with fewer than 3 usable months gives NaN rows instead of crashing."""
    from analysis import _fm_block as fm_block
    rows = []
    for name, xs in models.items():
        cols = [y] + xs + controls
        ok = panel.dropna(subset=cols).groupby("month").size() if len(panel) else pd.Series(dtype=int)
        if int((ok >= len(xs) + len(controls) + 10).sum()) < 3:
            rows.append(pd.DataFrame([{"model": name, "var": v, "coef": np.nan, "t": np.nan, "n_months": 0,
                                       "avg_obs": np.nan, "adj_r2": np.nan} for v in xs + controls]))
        else:
            rows.append(fm_block(panel, y, {name: xs}, lags, controls))
    return pd.concat(rows, ignore_index=True)


def table9_10(p: pd.DataFrame, lags: int) -> dict[str, pd.DataFrame]:
    """Panel A correlations, Panel B FM, Table 10 cost / non-cost FM on one lambda-merged panel."""
    from measures import mean_cs_corr
    controls = ["mom", "ret_1"]
    out = {}
    rows = []
    for name, cols, method in [("levels_pearson", ["lam", "A", "A_C", "ResA"], "pearson"),
                               ("levels_spearman", ["lam", "A", "A_C", "ResA"], "spearman"),
                               ("logs_pearson", ["lnlam", "lnA", "lnA_C", "ResLnA"], "pearson")]:
        pc = p.dropna(subset=cols)
        n_ok = int((pc.groupby("month").size() >= 10).sum()) if len(pc) else 0
        c = mean_cs_corr(pc, cols, "month", method) if n_ok else None
        for v in cols[1:]:
            rows.append({"corr": name, "var": v, "with": cols[0], "ours": float(c.loc[cols[0], v]) if n_ok else np.nan,
                         "paper": PAPER["T9A"].get(v, np.nan) if name == "levels_pearson" else np.nan,
                         "n_months": int(p.dropna(subset=cols)["month"].nunique())})
    out["T9A"] = pd.DataFrame(rows)

    s = p.dropna(subset=["lnlam", "lnA_C", "lnA", "ret_adj"] + controls)
    out["T9B"] = _fm_block(s, "ret_adj", {"(0) same sample": ["lnA_C"], "(1)": ["lnlam"], "(2)": ["lnlam", "lnA_C"]},
                           lags, controls)

    costs = {"lambda": ["lnlam"]}
    if "lnedge" in s and s["lnedge"].notna().any():
        costs["lambda+EDGE"] = ["lnlam", "lnedge"]
    t10 = []
    for cname, xs in costs.items():
        ss = s.dropna(subset=xs).copy()
        ss["cost"], ss["noncost"] = cs_fit_resid(ss, "lnA", xs, "month")
        fb = _fm_block(ss, "ret_adj", {"(1)": ["cost"], "(2)": ["noncost"], "(3)": ["cost", "noncost"]}, lags, controls)
        fb.insert(0, "cost_benchmarks", cname)
        g = ss.dropna(subset=["cost"]).groupby("month")
        fb["avg_cs_r2_lnA_on_cost"] = float((g["cost"].var() / g["lnA"].var()).mean()) if len(ss) else np.nan
        t10.append(fb)
    out["T10"] = pd.concat(t10, ignore_index=True)
    return out


def diagnostics(p: pd.DataFrame, panel: pd.DataFrame, lags: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """POST-HOC stability checks of Table 9 Panel B (added in the RUN stage after seeing the main result;
    not pre-registered, reported in full, main numbers unchanged).  Same common sample as table9_10.
    Variants: halves of the return months; without each coin's last panel month when it leaves the panel
    before the panel ends (delisting months, D10); raw instead of CMKT-adjusted returns; returns
    winsorized 1/99 within month (as base R_winsor_returns); + ivol control (lambda from trade prices
    co-moves with volatility, and ivol is priced in base T4 model 5); + lnVOL control (added in the
    VERIFY stage: the paper controls ln ME in Tables 9-10, we have no size control, ln dollar volume
    at t-2 is the closest proxy in the panel; skipped if the panel has no lnVOL).
    Returns (table, monthly slope series of models (0) and (1))."""
    from analysis import winsorize_returns
    from fama_macbeth import fama_macbeth
    controls = ["mom", "ret_1"]
    s = p.dropna(subset=["lnlam", "lnA_C", "lnA", "ret_adj"] + controls).copy()
    months = sorted(s["month"].unique())
    half = len(months) // 2
    last = panel.groupby("symbol")["month"].max()
    exit_row = s["symbol"].map(last).eq(s["month"]) & s["month"].ne(panel["month"].max())
    samples = {
        "main (same as T9B)": (s, "ret_adj"),
        f"first half ({months[0]}..{months[half - 1]})": (s[s["month"].isin(months[:half])], "ret_adj"),
        f"second half ({months[half]}..{months[-1]})": (s[s["month"].isin(months[half:])], "ret_adj"),
        "ex exit month (coin leaves panel)": (s[~exit_row], "ret_adj"),
        "raw ret (not CMKT-adjusted)": (s, "ret"),
        "ret_adj winsorized 1/99 within month": (winsorize_returns(s, (0.01, 0.99)), "ret_adj"),
        "+ ivol control": (s.dropna(subset=["ivol"]), "ret_adj"),
    }
    extras = {"+ ivol control": ["ivol"]}
    if "lnVOL" in s:
        name = "+ lnVOL control (size proxy; verify stage)"
        samples[name], extras[name] = (s.dropna(subset=["lnVOL"]), "ret_adj"), ["lnVOL"]
    rows = []
    for name, (d, y) in samples.items():
        extra = extras.get(name, [])
        models = {"(0)": ["lnA_C"], "(1)": ["lnlam"], "(2)": ["lnlam", "lnA_C"]}
        fb = _fm_block(d, y, {k: v + extra for k, v in models.items()}, lags, controls)
        fb = fb[~fb["var"].isin(controls)]
        fb.insert(0, "variant", name)
        fb.insert(1, "y", y)
        fb["cs_sd_lnlam"] = float(d.groupby("month")["lnlam"].std().mean()) if len(d) else np.nan
        fb["n_rows_dropped_vs_main"] = len(s) - len(d)
        rows.append(fb)
    tab = pd.concat(rows, ignore_index=True)
    tab["coef_per_cs_sd"] = np.where(tab["var"] == "lnlam", tab["coef"] * tab["cs_sd_lnlam"], np.nan)
    series = []
    for name, xs in [("(0)", ["lnA_C"]), ("(1)", ["lnlam"])]:
        r = fama_macbeth(s, "ret_adj", xs + controls, lags=lags)
        cs = r.attrs["coef_series"][xs[0]]
        series.append(pd.DataFrame({"model": name, "var": xs[0], "month": cs.index.astype(str), "slope": cs.to_numpy()}))
    return tab, pd.concat(series, ignore_index=True)


def _paper_t9b(model: str, var: str) -> tuple[float, float]:
    return PAPER["T9B"].get(model, {}).get(var, (np.nan, np.nan))


def _paper_t10(model: str, var: str) -> tuple[float, float]:
    return PAPER["T10"].get(model, {}).get(var, (np.nan, np.nan))


def load_edge(path: Path) -> tuple[pd.DataFrame | None, str]:
    """Optional monthly EDGE spreads from upgrade edge_spread (src/spread_edge.py): symbol, month
    (= ESTIMATION month) and the signed estimate `edge` (or the first numeric column containing 'edge')."""
    if not path.exists():
        msg = (f"absent: {rel_path(path)} not found (produced by `make upgrade-edge_spread`); "
               "variant 'T10 lambda+EDGE' NOT run")
        log.warning("EDGE file missing -> %s", msg)
        return None, msg
    e = pd.read_parquet(path)
    cands = ["edge"] if "edge" in e else [c for c in e.columns if "edge" in c.lower() and pd.api.types.is_numeric_dtype(e[c])]
    if "symbol" not in e or "month" not in e or not cands:
        msg = f"present but unrecognised columns {list(e.columns)}; variant 'T10 lambda+EDGE' NOT run"
        log.warning("EDGE file unusable -> %s", msg)
        return None, msg
    c = cands[0]
    e = e[["symbol", "month", c]].rename(columns={c: "edge"})
    e["measure_month"] = e["month"].astype(str).str[:7]
    return e[["symbol", "measure_month", "edge"]], f"used column '{c}' of {path.name} ({len(e)} coin-months)"


def attach_edge(panel: pd.DataFrame, edge: pd.DataFrame, lag: int, winsor) -> tuple[pd.DataFrame, float]:
    """Add lnedge to the FULL base panel with the main rule of upgrade edge_spread ("floor"): in each
    return-month cross-section non-positive EDGE -> smallest positive EDGE of that month; then
    winsorize 1/99% per month and log.  Returns (panel, share of floored estimates)."""
    from measures import winsorize_cs
    e = edge.copy()
    e["month"] = pd.PeriodIndex(e["measure_month"], freq="M") + lag
    p = panel.merge(e[["symbol", "month", "edge"]], on=["symbol", "month"], how="left")
    raw = p["edge"].astype(float)
    pos_min = raw.where(raw > 0).groupby(p["month"]).transform("min")
    floored = raw.notna() & (raw <= 0)
    p["EDGE"] = raw.where(~floored, pos_min)
    p = winsorize_cs(p, ["EDGE"], "month", tuple(winsor))
    with np.errstate(divide="ignore", invalid="ignore"):
        p["lnedge"] = np.log(p["EDGE"].where(p["EDGE"] > 0))
    share = float(floored.sum() / raw.notna().sum()) if raw.notna().any() else np.nan
    return p.drop(columns=["edge"]), share


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_files(paths: list[Path]) -> str:
    """sha256 over (relative name, sha256(file)) of the files in sorted order."""
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(str(p.relative_to(ROOT)).encode())
        h.update(sha256_file(p).encode())
    return h.hexdigest()


def git_info() -> dict:
    import subprocess
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        st = subprocess.run(["git", "status", "--porcelain", "--", "src", "tests", "config.yaml", "Makefile", "mk", "requirements.txt"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
        return {"sha": sha, "dirty": bool(st.strip()), "dirty_scope": "code+config (src, tests, config.yaml, Makefile, mk, requirements.txt); generated outputs excluded"}
    except (OSError, subprocess.CalledProcessError):
        return {"sha": None, "dirty": None}


def _fmt(x, d=3):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def _sig(t: float, hlz: float) -> str:
    if not np.isfinite(t):
        return "no estimate"
    a = abs(t)
    return "significant, |t| above the HLZ bar (size not calibrated at this T, see Test size)" if a > hlz else ("significant at 5%" if a > 1.96 else "not significant")


def write_summary(out: Path, hlz: float) -> None:
    t9a = pd.read_csv(out / "T9A_corr.csv")
    t9b = pd.read_csv(out / "T9B_fm.csv")
    t10 = pd.read_csv(out / "T10_fm.csv")
    cov = pd.read_csv(out / "coverage.csv")
    val = pd.read_csv(out / "validation_summary.csv")
    man = json.loads((out / "run_manifest.json").read_text())
    main = t9b[(t9b["spec"] == "main")]

    def fm(model, var, df=main):
        r = df[(df["model"] == model) & (df["var"] == var)]
        return (float(r["coef"].iloc[0]), float(r["t"].iloc[0]), int(r["n_months"].iloc[0]), float(r["avg_obs"].iloc[0])) \
            if len(r) else (np.nan, np.nan, 0, np.nan)

    L = []
    L.append("# Upgrade 2: λ from 5-minute bars signed by taker-buy volume (Binance spot)\n")
    L.append(f"_Generated by `src/hf_lambda.py` from the CSVs in this folder. Base run: `{man['base_run_id']}`; "
             f"run_id: `{man['run_id']}`._\n")
    L.append("**What we tested.** λ is the slope of the 5-minute log return on SVol (the paper's eq. 11, no intercept). "
             "The trade direction comes from the taker-buy volume that Binance reports in each 5-minute kline, i.e. the "
             "aggressor side aggregated within the bar (main SVol = √B − √S, robustness sign(B−S)·√|B−S|). The exact "
             "trade-by-trade Σ sign·√dvol from aggTrades (isBuyerMaker) is computed only on a validation subsample. "
             "We then build analogues of Table 9 (correlations, FM) and Table 10 (cost / non-cost parts of ln A).\n")
    c = cov.iloc[0]
    L.append(f"**Sample.** Measure months {man['spec']['measure_start']}…{man['spec']['measure_end']} "
             f"({int(c['n_measure_months'])} months; returns are two months later); coin-months in the base panel {int(c['n_panel_coin_months'])}, "
             f"with a 5m file {int(c['n_with_file'])}, with λ (≥{man['spec']['min_bars']} bars) {int(c['n_lambda_ok'])}, "
             f"of which λ ≤ 0: {int(c['n_lambda_nonpos'])}"
             + (" (the min_bars threshold removed no coin-month, so the min_bars_robust variant equals the main one)"
                if int(c["n_lambda_ok"]) == int(c["n_with_file"]) else "") + f". FM uses {int(fm('(1)', 'lnlam')[2])} months, "
             f"{_fmt(fm('(1)', 'lnlam')[3], 0)} coins on average (paper: 358 months, ~1725 stocks).\n")

    sel = pd.read_csv(out / "validation_selection.csv")
    n_skip_cap = int((sel["status"] == "skipped_over_cap").sum()) if len(sel) else 0
    n_skip_miss = int((sel["status"] == "skipped_missing_file").sum()) if len(sel) else 0
    vrow = val[(val["pair"] == "exact_vs_kl_sqrt_sides") & (val["months"] == "all")]
    if len(vrow):
        v = vrow.iloc[0]
        L.append(f"**Validation of the bar λ against the exact trade-by-trade λ.** {int(v['n'])} coin-months "
                 f"(stratified by A_C quintile): corr(ln λ_exact, ln λ_bar) = {_fmt(v['pearson_log'])}, "
                 f"Spearman = {_fmt(v['spearman'])}; median λ_bar/λ_exact = {_fmt(v['median_ratio'])}. "
                 f"Alternative proxy sign(B−S)·√|B−S|: corr = "
                 + _fmt(float(val.loc[(val["pair"] == "exact_vs_kl_net_sqrt") & (val["months"] == "all"), "pearson_log"].iloc[0])
                        if ((val["pair"] == "exact_vs_kl_net_sqrt") & (val["months"] == "all")).any() else np.nan)
                 + f". Candidates skipped: {n_skip_cap} (file > {man['spec']['aggtrades_max_mb']} MB), "
                 f"{n_skip_miss} (file missing in one of the months); see `validation_selection.csv`.\n")

    L.append("| Statistic | Paper | Us |\n|---|---|---|")
    pa = t9a[(t9a["spec"] == "main") & (t9a["corr"] == "levels_pearson")].set_index("var")
    for v in ["A", "A_C", "ResA"]:
        L.append(f"| corr(λ, {v}) | {_fmt(PAPER['T9A'][v])} | {_fmt(float(pa.loc[v, 'ours'])) if v in pa.index else '—'} |")
    for model, var in [("(1)", "lnlam"), ("(2)", "lnlam"), ("(2)", "lnA_C")]:
        b, t, *_ = fm(model, var)
        pb, pt = _paper_t9b(model, var)
        L.append(f"| FM {model} {var} | {_fmt(pb)} (t {_fmt(pt, 2)}) | {_fmt(b)} (t {_fmt(t, 2)}) |")
    t10m = t10[(t10["spec"] == "main") & (t10["cost_benchmarks"] == "lambda")]
    for model, var in [("(1)", "cost"), ("(2)", "noncost"), ("(3)", "cost"), ("(3)", "noncost")]:
        b, t, *_ = fm(model, var, t10m)
        pb, pt = _paper_t10(model, var)
        L.append(f"| T10 {model} {var} | {_fmt(pb)} (t {_fmt(pt, 2)}) | {_fmt(b)} (t {_fmt(t, 2)}) |")
    L.append("")

    b1, t1, n_m, _ = fm("(1)", "lnlam")
    b2, t2, *_ = fm("(2)", "lnlam")
    bc, tc, *_ = fm("(2)", "lnA_C")
    b0, t0, *_ = fm("(0) same sample", "lnA_C")
    nc, tnc, *_ = fm("(3)", "noncost", t10m)
    co, tco, *_ = fm("(3)", "cost", t10m)
    ca = float(pa.loc["A", "ours"]) if "A" in pa.index else np.nan
    cac = float(pa.loc["A_C", "ours"]) if "A_C" in pa.index else np.nan

    def level(c):
        return "no estimate" if not np.isfinite(c) else ("high" if c >= 0.5 else ("moderate" if c >= 0.3 else "low"))

    L.append("**Interpretation (of the numbers above).**")
    L.append(f"- Link between λ and the Amihud measures: corr(λ, A) = {_fmt(ca)} ({level(ca)}), corr(λ, A_C) = {_fmt(cac)} ({level(cac)}); "
             "in the paper both are about 0.74.")
    L.append(f"- ln λ alone: {_fmt(b1)} (t {_fmt(t1, 2)}), {_sig(t1, hlz)}; together with ln A_C: ln λ {_sig(t2, hlz)} "
             f"(t {_fmt(t2, 2)}), ln A_C {_sig(tc, hlz)} (t {_fmt(tc, 2)}). ln A_C alone on the same sample: t {_fmt(t0, 2)}.")
    L.append(f"- Table 10 (cost = projection of ln A on ln λ), model (3): the cost part is {_sig(tco, hlz)} "
             f"({_fmt(co)}, t {_fmt(tco, 2)}); the non-cost part is {_sig(tnc, hlz)} ({_fmt(nc)}, t {_fmt(tnc, 2)}).")
    alt = t9b[(t9b["spec"] == "alt_svol") & (t9b["model"] == "(1)") & (t9b["var"] == "lnlam")]
    if len(alt):
        L.append(f"- Robustness, SVol = sign(B−S)·√|B−S|: ln λ alone t {_fmt(float(alt['t'].iloc[0]), 2)}.")
    te = t10[(t10["spec"] == "main") & (t10["cost_benchmarks"] == "lambda+EDGE")]
    if len(te):
        ce, tce, *_ = fm("(3)", "cost", te)
        ne, tne, *_ = fm("(3)", "noncost", te)
        L.append(f"- Table 10 with λ and EDGE: cost part t {_fmt(tce, 2)} ({_sig(tce, hlz)}), non-cost part t {_fmt(tne, 2)} ({_sig(tne, hlz)}).")
    base_txt = "file not found"
    bt4 = ROOT / "results" / man["base_run_id"] / "T4_fm.csv"
    if bt4.exists():
        b4 = pd.read_csv(bt4)
        r4 = b4[(b4["model"] == "(2)") & (b4["var"] == "lnA_C")]
        if len(r4):
            base_txt = (f"{_fmt(float(r4['coef'].iloc[0]))}, t {_fmt(float(r4['t'].iloc[0]), 2)}, "
                        f"{int(r4['n_months'].iloc[0])} months")
    sign = lambda b: "—" if not np.isfinite(b) else ("negative" if b < 0 else "positive")
    L.append(f"- Signs vs the paper: ln λ alone is {sign(b1)} (paper: −0.034, t −0.67, i.e. λ on its own is not priced); "
             f"ln λ together with ln A_C is {sign(b2)} (paper: −0.155), ln A_C is {sign(bc)} (paper: +0.180). "
             f"In this window ln A_C without λ: {_fmt(b0)} (t {_fmt(t0, 2)}, {_sig(t0, hlz)}); in the base run over the full period "
             f"(T4 model (2): {base_txt}). The point estimates of the A_C premium in the λ window and over the full period differ "
             "in sign; the difference between them was not formally tested.")
    L.append("- In Table 10 model (1), the cost part within a month is a linear function of ln λ (cost = a_t + b_t·ln λ), so its t "
             "is close to the t of Table 9 model (1) but not equal to it, because the slope b_t varies across months.")
    L.append(f"- Thresholds: 1.96 and HLZ t > {hlz:g}. All variants (SVol proxy, min_bars, Spearman, logs) are in `T9B_fm.csv`, "
             "`T10_fm.csv` and `T9A_corr.csv`; the list is in `variants.csv`.\n")
    dpath = out / "T9B_diagnostics.csv"
    if dpath.exists():
        dg = pd.read_csv(dpath)
        L.append("\n**Panel B robustness checks (post hoc: added after we saw the main result; "
                 "not pre-registered; they do not change the main numbers).** Source: `T9B_diagnostics.csv`.\n")
        L.append("| Variant | (1) ln λ (t) | effect of 1 σ ln λ, %/mo | (2) ln λ (t) | (2) ln A_C (t) | (0) ln A_C (t) | months |\n|---|---|---|---|---|---|---|")
        for v, g in dg.groupby("variant", sort=False):
            def cell(m, var):
                r = g[(g["model"] == m) & (g["var"] == var)]
                return (f"{_fmt(float(r['coef'].iloc[0]))} ({_fmt(float(r['t'].iloc[0]), 2)})", r) if len(r) else ("—", r)
            c1, r1 = cell("(1)", "lnlam")
            per = _fmt(float(r1["coef_per_cs_sd"].iloc[0]), 2) if len(r1) else "—"
            nm = int(r1["n_months"].iloc[0]) if len(r1) else 0
            L.append(f"| {v} | {c1} | {per} | {cell('(2)', 'lnlam')[0]} | {cell('(2)', 'lnA_C')[0]} | {cell('(0)', 'lnA_C')[0]} | {nm} |")
        sp = out / "fm_slopes_monthly.csv"
        if sp.exists():
            sl = pd.read_csv(sp)
            s1 = sl[sl["model"] == "(1)"]["slope"]
            L.append(f"\nMonthly slopes of model (1) (`fm_slopes_monthly.csv`): {int((s1 < 0).sum())} of {len(s1)} negative, "
                     f"median {_fmt(float(s1.median()))}, min {_fmt(float(s1.min()))}, max {_fmt(float(s1.max()))}.\n")
    L.append(f"**Caveats.** Only {n_m} return months in FM (the paper has 358), so power is low: an insignificant result does not "
             "prove the absence of a premium, and significance over 36 months (one market cycle) does not guarantee stability. "
             "Returns use the trade price, not the midquote (spot quotes are not publicly archived), so bid-ask bounce "
             "inflates λ for coins with wide spreads. The bar SVol approximates the trade-by-trade SVol (see the validation). "
             f"Bars with |r| > {man['spec']['max_abs_r']} are excluded ({int(c['n_bars_extreme_dropped'])} bars, including real "
             "flash crashes). "
             + (f"The ≥ {man['spec']['min_bars']} bars requirement removed the {int(c['n_with_file']) - int(c['n_lambda_ok'])} "
                "most illiquid coin-months. " if int(c["n_lambda_ok"]) < int(c["n_with_file"]) else "")
             + "λ ≤ 0 is not dropped; cross-sectional 1/99 winsorisation raises it to the 1st percentile "
             f"({int(c['n_lambda_nonpos'])} such coin-months). "
             + "EDGE: " + man.get("edge_status", "—") + ".")
    sim_man = ROOT / "results" / "simulations" / "sim_manifest.json"
    sim_T = json.loads(sim_man.read_text()).get("n_months") if sim_man.exists() else None
    L.append(f"\n**Test size.** The world-0 calibration of NW(6) (`results/simulations/`, "
             f"`results/upgrades/robustness/world0_calibration.csv`) was run at "
             + (f"{sim_T} months" if sim_T else "the base sample length")
             + f"; at T = {n_m} months the size of the test, and so the meaning of |t| > {hlz:g} here, is not calibrated.")
    if not man.get("edge_variant_run", True):
        L.append("\n**WARNING: the Table 10 variant with λ+EDGE was NOT run**: " + man.get("edge_status", "")
                 + ". `T10_fm.csv` has no `lambda+EDGE` rows.")
    if man.get("offline"):
        L.append(f"\n**WARNING: offline run on incomplete data** ({int(c['n_not_downloaded_offline'])} files not downloaded); "
                 "do not interpret the numbers.")
    (out / "SUMMARY.md").write_text("\n".join(L) + "\n")


def plan(spec: dict, panel: pd.DataFrame, lag: int, workers: int) -> pd.DataFrame:
    """List 5m kline sizes for the base-panel coins over the full possible window (for the budget)."""
    syms = sorted(panel["symbol"].unique())
    with ThreadPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(lambda s: list_sizes(f"data/spot/monthly/klines/{s}/5m/"), syms))
    sizes = {k: v for d in res for k, v in d.items()}
    pm = panel[["symbol", "month"]].copy()
    pm["measure_month"] = (pm["month"] - lag).astype(str)
    pm["bytes"] = [sizes.get(kline_key(s, m), np.nan) for s, m in zip(pm["symbol"], pm["measure_month"])]
    by = pm.groupby("measure_month").agg(n=("symbol", "size"), n_file=("bytes", "count"), mb=("bytes", lambda b: b.sum() / 2**20))
    by["cum_gb_from_end"] = by["mb"][::-1].cumsum()[::-1] / 1024
    return by.reset_index()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--plan", action="store_true", help="only list sizes per measure month and exit")
    ap.add_argument("--download-only", action="store_true")
    ap.add_argument("--no-download", action="store_true",
                    help="offline: use only files already in data/raw (missing ones are logged as not_downloaded)")
    ap.add_argument("--strict-offline", action="store_true",
                    help="with --no-download: fail (exit != 0) if any needed file is missing instead of running on a subset")
    ap.add_argument("--allow-partial", action="store_true",
                    help="run even if some downloads failed after retries (default: exit != 0)")
    ap.add_argument("--workers", type=int, default=6, help="processes for lambda estimation")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg_path = Path(a.config)
    cfg = yaml.safe_load(cfg_path.read_text())
    spec = SPEC | (cfg.get("hf_lambda") or {})
    seed, lag = int(cfg["seed"]), int(cfg["measures"]["lag_months"])
    nw, hlz = int(cfg["tests"]["nw_lags"]), float(cfg["tests"]["hlz_threshold"])
    timings: dict[str, float] = {}
    T = time.perf_counter

    t = T()
    base_id, panel_path = base_run()
    panel = load_panel(panel_path)
    timings["load_panel"] = round(T() - t, 2)
    log.info("base run %s: %d rows", base_id, len(panel))
    if a.plan:
        print(plan(spec, panel, lag, spec["download_workers"]).to_string())
        return

    t = T()
    need = needed_coin_months(panel, spec, lag)
    kkeys = [kline_key(s, m) for s, m in zip(need["symbol"], need["measure_month"])]
    vm = spec["validation_months"]
    vsyms = sorted(panel.loc[panel["month"] == pd.Period(vm[0], "M") + lag, "symbol"].unique())
    if a.no_download:
        agg_sizes = {agg_key(s, m): local_path(agg_key(s, m)).stat().st_size for s in vsyms for m in vm
                     if local_path(agg_key(s, m)).exists()}
    else:
        agg_sizes = aggtrades_listing(vsyms, vm, spec["download_workers"])
    sel = select_validation(panel, spec, lag, seed, agg_sizes)
    vkeys = [agg_key(s, m) for s in sel.loc[sel["status"] == "selected", "symbol"] for m in vm]
    vkl = [kline_key(s, m) for s in sel.loc[sel["status"] == "selected", "symbol"] for m in vm]
    if a.no_download:
        def local_status(keys):
            return pd.DataFrame([{"key": k, "status": "cached" if local_path(k).exists() else "not_downloaded",
                                  "bytes": local_path(k).stat().st_size if local_path(k).exists() else 0} for k in keys],
                                columns=["key", "status", "bytes"])
        dl, dla = local_status(sorted(set(kkeys + vkl))), local_status(vkeys)
    else:
        dl = download_many(sorted(set(kkeys + vkl)), spec["download_workers"])
        dla = download_many(vkeys, max(2, spec["download_workers"] // 2))
    timings["download"] = round(T() - t, 2)
    dl_all = pd.concat([dl.assign(kind="klines5m"), dla.assign(kind="aggTrades")], ignore_index=True)
    n_off = int((dl_all["status"] == "not_downloaded").sum())
    if n_off:
        log.warning("offline (--no-download): %d needed files are not in data/raw; results are on INCOMPLETE data "
                    "(listed in downloads_missing.csv, flagged in SUMMARY.md)", n_off)
        if a.strict_offline:
            raise SystemExit(f"--strict-offline: {n_off} needed files missing from data/raw, e.g. "
                             + ", ".join(dl_all.loc[dl_all['status'] == 'not_downloaded', 'key'].head(3)))
    log.info("downloads: %s", dl_all.groupby(["kind", "status"]).size().to_dict())
    n_failed = int((dl_all["status"] == "failed_after_retries").sum())
    if n_failed and not a.allow_partial:
        raise SystemExit(f"{n_failed} downloads failed after retries (e.g. "
                         + ", ".join(dl_all.loc[dl_all['status'] == 'failed_after_retries', 'key'].head(3))
                         + "). Rerun `make upgrade-hf_lambda`: cached files are kept and only the rest is fetched "
                         "(or pass --allow-partial to run on the subset; it is then flagged in SUMMARY.md).")
    if a.download_only:
        return

    t = T()
    paths = sorted({str(local_path(k)) for k in kkeys if local_path(k).exists()})
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        lam_rows = list(ex.map(kline_month_lambda, paths, [spec["max_abs_r"]] * len(paths), chunksize=32))
    lam = pd.DataFrame(lam_rows)
    for k in ["sqrt_sides", "net_sqrt"]:
        if f"lam_{k}" not in lam:
            lam[f"lam_{k}"] = np.nan
    if "n_bars" not in lam:
        lam["n_bars"] = 0
    lam["n_bars"] = lam["n_bars"].fillna(0).astype(int)
    lam = lam.rename(columns={"symbol": "raw_symbol"})
    lam = need.assign(raw_symbol=need["symbol"].map(raw_symbol)).merge(lam, on=["raw_symbol", "measure_month"], how="inner")
    lam["shared_file"] = lam.duplicated(["raw_symbol", "measure_month"], keep=False)
    timings["lambda_klines"] = round(T() - t, 2)

    t = T()
    apaths = [str(local_path(k)) for k in vkeys if local_path(k).exists()]
    kpaths = [str(local_path(kline_key(Path(p).parent.name, Path(p).stem[-7:]))) for p in apaths]
    with ProcessPoolExecutor(max_workers=min(a.workers, 4)) as ex:
        vrows = list(ex.map(aggtrades_month_lambda, apaths, kpaths, [spec["max_abs_r"]] * len(apaths)))
    val = pd.DataFrame(vrows)
    timings["lambda_aggtrades"] = round(T() - t, 2)

    t = T()
    edge, edge_status = load_edge(CACHE / "edge_monthly.parquet")
    panel_e = panel
    if edge is not None:
        panel_e, floored = attach_edge(panel, edge, lag, spec["winsor"])
        edge_status += f"; main rule 'floor' as in edge_spread, floored share {floored:.3f}"
    variants = [("main", spec["svol_main"], spec["min_bars"]),
                ("alt_svol", spec["svol_alt"], spec["min_bars"]),
                ("min_bars_robust", spec["svol_main"], spec["min_bars_robust"])]
    T9A, T9B, T10, describe = [], [], [], []
    for name, kind, mb in variants:
        p = attach_lambda(panel_e, lam, spec, lag, kind, mb, spec["winsor"])
        r = table9_10(p, nw)
        if name == "main":
            diag, slopes = diagnostics(p, panel_e, nw)
        for k, lst in [("T9A", T9A), ("T9B", T9B), ("T10", T10)]:
            df = r[k]
            df.insert(0, "spec", name)
            lst.append(df)
        d = p["lam"].describe(percentiles=[.1, .25, .5, .75, .9])
        describe.append({"spec": name, "svol": kind, "min_bars": mb, **d.to_dict(),
                         "share_nonpos": float((p["lam"] <= 0).mean()), "n_months": int(p["month"].nunique())})
    t9a, t9b, t10 = (pd.concat(x, ignore_index=True) for x in (T9A, T9B, T10))
    t9b["paper_coef"] = [_paper_t9b(m, v)[0] if s == "main" else np.nan for s, m, v in zip(t9b["spec"], t9b["model"], t9b["var"])]
    t9b["paper_t"] = [_paper_t9b(m, v)[1] if s == "main" else np.nan for s, m, v in zip(t9b["spec"], t9b["model"], t9b["var"])]
    t10["paper_coef"] = [_paper_t10(m, v)[0] for m, v in zip(t10["model"], t10["var"])]
    t10["paper_t"] = [_paper_t10(m, v)[1] for m, v in zip(t10["model"], t10["var"])]
    timings["tests"] = round(T() - t, 2)

    lm = need.merge(lam, on=["symbol", "measure_month"], how="left")
    ok = lm["n_bars"].fillna(0) >= spec["min_bars"]
    cov_m = lm.assign(has_file=lm["error"].notna(), ok=ok, nonpos=ok & (lm["lam_sqrt_sides"] <= 0),
                      err=lm["error"].fillna("").astype(str).ne("")).groupby("measure_month").agg(
        n_panel=("symbol", "size"), n_with_file=("has_file", "sum"), n_error=("err", "sum"),
        n_lambda_ok=("ok", "sum"), n_lambda_nonpos=("nonpos", "sum"))
    cov_m.reset_index().to_csv(OUT_DIR / "coverage_by_month.csv", index=False)
    pd.DataFrame([{"n_measure_months": int(need["measure_month"].nunique()),
                   "n_panel_coin_months": int(len(need)), "n_with_file": int(cov_m["n_with_file"].sum()),
                   "n_missing_on_server": int((dl["status"] == "missing").sum()),
                   "n_not_downloaded_offline": int((dl["status"] == "not_downloaded").sum()),
                   "n_download_failed": int((dl_all["status"] == "failed_after_retries").sum()),
                   "n_read_errors": int(cov_m["n_error"].sum()),
                   "n_lambda_ok": int(cov_m["n_lambda_ok"].sum()), "n_lambda_nonpos": int(cov_m["n_lambda_nonpos"].sum()),
                   "n_relabelled_ids": int((lam["symbol"] != lam["raw_symbol"]).sum()),
                   "n_shared_file": int(lam["shared_file"].sum()),
                   "n_bars_extreme_dropped": int(lam.get("n_extreme", pd.Series(dtype=float)).fillna(0).sum()),
                   "klines5m_gb": round(dl["bytes"].sum() / 2**30, 3),
                   "aggtrades_gb": round(dla["bytes"].sum() / 2**30, 3)}]).to_csv(OUT_DIR / "coverage.csv", index=False)

    vs = []
    if len(val):
        val["months"] = val["measure_month"]
        pairs = [("exact_vs_kl_sqrt_sides", "agg_lam_exact", "kl_lam_sqrt_sides"),
                 ("exact_vs_kl_net_sqrt", "agg_lam_exact", "kl_lam_net_sqrt"),
                 ("exact_vs_agg_sqrt_sides", "agg_lam_exact", "agg_lam_sqrt_sides"),
                 ("kl_vs_agg_sqrt_sides", "kl_lam_sqrt_sides", "agg_lam_sqrt_sides")]
        for grp, g in [("all", val)] + list(val.groupby("measure_month")):
            for name, x, y in pairs:
                if x not in g or y not in g:
                    continue
                gg = g[(g[x] > 0) & (g[y] > 0) & (g.get("agg_n_bars", 0) >= spec["min_bars"])]
                vs.append({"pair": name, "months": grp, "n": len(gg),
                           "pearson_log": float(np.corrcoef(np.log(gg[x]), np.log(gg[y]))[0, 1]) if len(gg) > 2 else np.nan,
                           "spearman": float(gg[x].corr(gg[y], method="spearman")) if len(gg) > 2 else np.nan,
                           "median_ratio": float((gg[y] / gg[x]).median()) if len(gg) else np.nan})
    pd.DataFrame(vs, columns=["pair", "months", "n", "pearson_log", "spearman", "median_ratio"]).to_csv(
        OUT_DIR / "validation_summary.csv", index=False)

    t9a.to_csv(OUT_DIR / "T9A_corr.csv", index=False)
    t9b.to_csv(OUT_DIR / "T9B_fm.csv", index=False)
    diag.to_csv(OUT_DIR / "T9B_diagnostics.csv", index=False)
    slopes.to_csv(OUT_DIR / "fm_slopes_monthly.csv", index=False)
    t10.to_csv(OUT_DIR / "T10_fm.csv", index=False)
    pd.DataFrame(describe).to_csv(OUT_DIR / "lambda_describe.csv", index=False)
    lam.to_csv(OUT_DIR / "lambda_monthly.csv", index=False)
    val.to_csv(OUT_DIR / "validation_lambda.csv", index=False)
    sel.to_csv(OUT_DIR / "validation_selection.csv", index=False)
    dl_all.groupby(["kind", "status"]).agg(n=("key", "size"), gb=("bytes", lambda b: b.sum() / 2**30)).reset_index() \
        .to_csv(OUT_DIR / "downloads.csv", index=False)
    dl_all.loc[dl_all["status"].isin(["missing", "failed_after_retries", "not_downloaded"])].to_csv(OUT_DIR / "downloads_missing.csv", index=False)
    pd.DataFrame([{"spec": n, "svol": k, "min_bars": mb, "role": "main" if n == "main" else "robustness"}
                  for n, k, mb in variants] +
                 [{"spec": "spearman correlations", "svol": spec["svol_main"], "min_bars": spec["min_bars"], "role": "robustness"},
                  {"spec": "log-log correlations", "svol": spec["svol_main"], "min_bars": spec["min_bars"], "role": "robustness"},
                  {"spec": "T10 lambda+EDGE", "svol": "all", "min_bars": "all",
                   "role": "main" if edge is not None else "not run (no EDGE file)"}] +
                 [{"spec": f"T9B_diagnostics: {v}", "svol": spec["svol_main"], "min_bars": spec["min_bars"],
                   "role": "post-hoc diagnostic (added after seeing main result)"} for v in diag["variant"].unique()]).to_csv(OUT_DIR / "variants.csv", index=False)

    t = T()
    code_files = [ROOT / "src" / f for f in ["hf_lambda.py", "analysis.py", "fama_macbeth.py", "measures.py"]]
    raw_files = [local_path(k) for k in set(kkeys + vkl + vkeys) if local_path(k).exists()]
    data_hash = {"base_panel": sha256_file(panel_path), "raw_zips": sha256_files(raw_files), "n_raw_zips": len(raw_files)}
    if edge is not None:
        data_hash["edge_monthly"] = sha256_file(CACHE / "edge_monthly.parquet")
    timings["hash"] = round(T() - t, 2)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cfg_hash = sha256_file(cfg_path)
    manifest = {
        "run_id": f"hf_lambda_{stamp}_{cfg_hash[:8]}", "created_utc": stamp, "base_run_id": base_id,
        "base_panel": rel_path(panel_path), "git": git_info(), "config_sha256": cfg_hash,
        "code_sha256": {p.name: sha256_file(p) for p in code_files}, "data_sha256": data_hash,
        "seed": seed, "spec": spec, "offline": bool(a.no_download), "paper": {k: v for k, v in PAPER.items()},
        "edge_status": edge_status, "edge_variant_run": edge is not None, "timings_sec": timings | {"total": round(sum(timings.values()), 2)},
        "versions": {"python": sys.version.split()[0], "pandas": pd.__version__, "numpy": np.__version__},
        "tables": sorted(p.name for p in OUT_DIR.glob("*.csv")),
    }
    (OUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    write_summary(OUT_DIR, hlz)
    log.info("written to %s (%.0fs)", OUT_DIR, sum(timings.values()))


OUT_DIR = OUT
if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    main()
