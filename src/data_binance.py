"""Binance spot daily klines from data.binance.vision.

Survivorship: the symbol list comes from the S3 bucket listing, which keeps
pairs that were delisted, so the universe of every month includes them.

Usage:  python -I src/data_binance.py config.yaml
Writes: data/raw/klines/<SYMBOL>/*.zip  and  data/cache/daily.parquet
"""
from __future__ import annotations

import io
import re
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "klines"
CACHE = ROOT / "data" / "cache"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
              "quote_volume", "n_trades", "taker_buy_base", "taker_buy_quote", "ignore"]

_session = requests.Session()


def _get(url: str, params: dict | None = None, tries: int = 6) -> requests.Response:
    """GET with exponential backoff (the proxy drops connections under load)."""
    for k in range(tries):
        try:
            r = _session.get(url, params=params, timeout=60)
            r.raise_for_status()
            return r
        except requests.RequestException:
            if k == tries - 1:
                raise
            time.sleep(2 ** k)


def _list(listing_url: str, prefix: str, delimiter: str | None = "/") -> tuple[list[str], list[str]]:
    """Return (common prefixes, keys) under prefix, following S3 pagination."""
    prefixes, keys, marker = [], [], ""
    while True:
        params = {"prefix": prefix, "marker": marker}
        if delimiter:
            params["delimiter"] = delimiter
        r = _get(listing_url, params)
        root = ET.fromstring(r.content)
        prefixes += [p.text for p in root.findall("s3:CommonPrefixes/s3:Prefix", NS)]
        keys += [k.text for k in root.findall("s3:Contents/s3:Key", NS)]
        if root.findtext("s3:IsTruncated", namespaces=NS) != "true":
            return prefixes, keys
        marker = root.findtext("s3:NextMarker", namespaces=NS) or (keys[-1] if keys else prefixes[-1])


def eligible_symbol(symbol: str, cfg: dict, listed: set[str] | None = None) -> str | None:
    """Return the base asset if SYMBOL is an eligible <BASE><QUOTE> pair, else None.

    A base ending in a leveraged-token suffix (BTCUP, ETHBULL) is excluded only when the
    stem is itself a listed pair (BTCUSDT); with `listed` given, real coins such as JUP or
    SYRUP are kept. Without `listed` every such suffix is excluded (conservative).
    """
    quote = cfg["market"]["quote"]
    if not symbol.endswith(quote) or not re.fullmatch(r"[0-9A-Z]+", symbol):
        return None
    base = symbol[: -len(quote)]
    u = cfg["universe"]
    if not base or base in set(u["exclude_bases"]):
        return None
    if any(base.endswith(s) and len(base) > len(s)
           and (listed is None or base[: -len(s)] + quote in listed) for s in u["exclude_suffixes"]):
        return None
    return base


def list_symbols(cfg: dict) -> list[str]:
    prefix = "data/spot/monthly/klines/"
    prefixes, _ = _list(cfg["market"]["listing_url"], prefix)
    syms = [p[len(prefix):].strip("/") for p in prefixes]
    listed = set(syms)
    return sorted(s for s in syms if eligible_symbol(s, cfg, listed))


def _download_symbol(symbol: str, cfg: dict) -> int:
    m = cfg["market"]
    prefix = f"data/spot/monthly/klines/{symbol}/{m['interval']}/"
    _, keys = _list(m["listing_url"], prefix, delimiter=None)
    out = RAW / symbol
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for key in keys:
        if not key.endswith(".zip"):
            continue
        month = key[-11:-4]
        if not (m["first_data_month"] <= month <= m["last_data_month"]):
            continue
        dst = out / Path(key).name
        if dst.exists() and dst.stat().st_size > 0:
            continue
        r = _get(f"https://data.binance.vision/{key}")
        dst.write_bytes(r.content)
        n += 1
    return n


def _read_zip(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        name = zf.namelist()[0]
        raw = zf.read(name)
    df = pd.read_csv(io.BytesIO(raw), header=None)
    if isinstance(df.iloc[0, 0], str):          # some files carry a header row
        df = df.iloc[1:]
    df = df.iloc[:, :12]
    df.columns = KLINE_COLS
    return df


def build_daily(symbols: list[str], cfg: dict) -> pd.DataFrame:
    frames = []
    for sym in symbols:
        files = sorted((RAW / sym).glob("*.zip"))
        if not files:
            continue
        df = pd.concat([_read_zip(f) for f in files], ignore_index=True)
        t = pd.to_numeric(df["open_time"])
        # Binance switched spot timestamps from ms to us in 2025
        t = t.where(t < 1e14, t // 1000)
        df = pd.DataFrame({
            "symbol": sym,
            "base": sym[: -len(cfg["market"]["quote"])],
            "date": pd.to_datetime(t, unit="ms").dt.normalize(),
            "close": pd.to_numeric(df["close"], errors="coerce"),
            "quote_volume": pd.to_numeric(df["quote_volume"], errors="coerce"),
            "n_trades": pd.to_numeric(df["n_trades"], errors="coerce"),
        })
        frames.append(df)
    daily = pd.concat(frames, ignore_index=True)
    daily = daily.drop_duplicates(["symbol", "date"]).sort_values(["symbol", "date"])
    return daily.reset_index(drop=True)


def main(cfg_path: str) -> None:
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    symbols = list_symbols(cfg)
    print(f"eligible symbols: {len(symbols)}", flush=True)
    with ThreadPoolExecutor(max_workers=8) as ex:
        for i, n in enumerate(ex.map(lambda s: _download_symbol(s, cfg), symbols)):
            if i % 50 == 0:
                print(f"  {i}/{len(symbols)} symbols", flush=True)
    daily = build_daily(symbols, cfg)
    CACHE.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(CACHE / "daily.parquet", index=False)
    (CACHE / "symbols.txt").write_text("\n".join(symbols) + "\n")
    print(f"daily rows: {len(daily):,}; symbols with data: {daily['symbol'].nunique()}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "config.yaml"))
