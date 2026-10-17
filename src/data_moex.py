"""MOEX (Moscow Exchange) board TQBR: survivorship-free universe, daily history, splits, dividends.

Sources (all public, no login):
  universe   /iss/history/engines/stock/markets/shares/boards/TQBR/securities.json?date=<month-end>
             -> every SECID listed on TQBR at each month-end (not today's list).
  history    /iss/history/engines/stock/markets/shares/boards/TQBR/securities/<SECID>.json
             -> TRADEDATE, CLOSE, VALUE (RUB), VOLUME (shares), NUMTRADES (pages of 100, `start=`).
  info       /iss/securities/<SECID>.json -> TYPE (common_share, preferred_share, depositary_receipt, ...)
             and ISSUESIZE (shares outstanding *today only*; see DEVIATIONS M3).
  splits     /iss/statistics/engines/stock/splits.json -> tradedate, secid, before, after.
  dividends  the ISS endpoint /iss/securities/<SECID>/dividends.json is retired (it now returns the
             security description; it is not in /iss/reference, and /iss/cci/corp-actions/dividends
             requires a MOEX Passport login). We therefore take dividends per share and registry
             ("record") dates from dohod.ru (www.dohod.ru/ik/analytics/dividend/<secid>) and, when
             that page does not exist, from smart-lab.ru (smart-lab.ru/q/<SECID>/dividend/, which also
             gives the last cum-dividend day "T-1"). Both are third-party pages: parsed as untrusted
             text with the standard-library HTML parser, cached as raw HTML under data/raw/moex/.

Everything downloaded is cached per request in data/raw/moex/ (atomic writes), so re-running is
idempotent and offline once the cache is complete.

Daily construction (`build_daily`), one row per (SECID, trade day with VALUE > 0 and CLOSE):
  * ex-dividend date from the registry date R and the settlement regime (T+2 until 2023-07-28,
    T+1 from 2023-07-31): last cum day c = the latest trading day whose settlement day
    (c + L trading days) is <= R; ex-date = next trading day after c.  smart-lab rows give c directly.
  * total-return gross_k = (close_k + D_k) / close_{k-1} * after/before (splits between the two trades).
  * daily return r_k is defined only when the previous trade of the stock was on the previous
    market trading day and the interval does not span the 2022-02-28..2022-03-23 trading halt
    (the first post-halt day is missing, not a one-day return).
  * tri: total-return index chained over all trades (monthly returns use month-end tri, so the
    February->March 2022 holding-period return is kept).
  * volume_adj: VOLUME converted to today's share units (split-adjusted) to match ISSUESIZE.

Usage:  python -I src/data_moex.py config.yaml [--workers 6]
Writes: data/raw/moex/**  and  data/cache/moex_{universe,info,history,splits,dividends,daily}.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "moex"
CACHE = ROOT / "data" / "cache"
ISS = "https://iss.moex.com/iss"
BOARD = "TQBR"
HIST_COLS = "TRADEDATE,SECID,CLOSE,VALUE,VOLUME,NUMTRADES"
log = logging.getLogger("data_moex")

DATA_SPEC = {
    "board": BOARD,
    "universe_first_month": "2014-01",
    "history_from": "2014-01-01",
    "history_till": "2026-09-30",
    "halt": ["2022-02-28", "2022-03-23"],
    "t1_from": "2023-07-31",
    "max_div_yield": 0.5,
    "split_guard": 8.0,
    "div_split_adjusted": True,
    "infer_splits": True,
    "infer_price_tol": 0.15,
    "infer_volume_tol": 3.0,
    "infer_window": 20,
    "verified_splits": [
        {"secid": "BELU", "date": "2024-08-22", "before": 1, "after": 8,
         "source": "MOEX notice on BELU suspension before the share distribution (last trading day 15.08.2024); "
                   "Novabev 1:8 (7 additional shares per share)"},
    ],
    "verified_large_dividends": [
        {"secid": "LNZL", "registry_date": "2021-07-13",
         "source": "Board recommendation, 15,219.50 RUB per common share for 2020, registry date 13.07.2021"},
    ],
}

_local = threading.local()


def _session() -> requests.Session:
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
        _local.s.headers["User-Agent"] = "Mozilla/5.0 (research replication; RS3 HSE)"
    return _local.s


def _get(url: str, params: dict | None = None, tries: int = 6, ok404: bool = False) -> requests.Response | None:
    for k in range(tries):
        try:
            r = _session().get(url, params=params, timeout=60)
            if ok404 and r.status_code == 404:
                return None
            r.raise_for_status()
            return r
        except requests.RequestException:
            if k == tries - 1:
                raise
            time.sleep(2 ** k)
    return None


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}_{threading.get_ident()}")
    tmp.write_bytes(data)
    tmp.replace(path)


def _iss_table(url: str, params: dict, block: str) -> tuple[list[str], list[list]]:
    """All pages of one ISS block (pages of 100 rows via `start=`)."""
    cols, rows, start = None, [], 0
    while True:
        js = _get(url, {**params, "iss.meta": "off", "start": start}).json()
        b = js[block]
        cols = b["columns"]
        rows += b["data"]
        cur = js.get(f"{block}.cursor")
        if cur and cur["data"]:
            idx, total, size = cur["data"][0]
            if idx + size >= total:
                break
            start = idx + size
        elif len(b["data"]) < 100:
            break
        else:
            start += len(b["data"])
    return cols, rows


def _cached_json(path: Path, fetch) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    obj = fetch()
    _atomic_write(path, json.dumps(obj).encode())
    return obj


def month_ends(first: str, last: str) -> list[pd.Timestamp]:
    return [p.to_timestamp(how="end").normalize() for p in pd.period_range(first, last, freq="M")]


def fetch_universe_month(month_end: pd.Timestamp) -> dict:
    """Board snapshot on the last date <= month_end that has rows (steps back up to 10 days)."""
    def fetch():
        url = f"{ISS}/history/engines/stock/markets/shares/boards/{BOARD}/securities.json"
        for back in range(11):
            d = (month_end - pd.Timedelta(days=back)).strftime("%Y-%m-%d")
            cols, rows = _iss_table(url, {"date": d, "history.columns": HIST_COLS}, "history")
            if rows:
                return {"date": d, "columns": cols, "data": rows}
        return {"date": None, "columns": [], "data": []}
    return _cached_json(RAW / "universe" / f"{month_end:%Y-%m}.json", fetch)


def fetch_info(secid: str) -> dict:
    def fetch():
        js = _get(f"{ISS}/securities/{secid}.json", {"iss.meta": "off", "iss.only": "description"}).json()
        return {r[0]: r[2] for r in js["description"]["data"]}
    return _cached_json(RAW / "info" / f"{secid}.json", fetch)


def fetch_history(secid: str, start: str, till: str) -> dict:
    def fetch():
        url = f"{ISS}/history/engines/stock/markets/shares/boards/{BOARD}/securities/{secid}.json"
        cols, rows = _iss_table(url, {"from": start, "till": till, "history.columns": HIST_COLS}, "history")
        return {"from": start, "till": till, "columns": cols, "data": rows}
    return _cached_json(RAW / "history" / f"{secid}_{start}_{till}.json", fetch)


def fetch_splits() -> dict:
    def fetch():
        cols, rows = _iss_table(f"{ISS}/statistics/engines/stock/splits.json", {}, "splits")
        return {"columns": cols, "data": rows}
    return _cached_json(RAW / "splits.json", fetch)


def fetch_dividend_pages(secid: str) -> dict[str, Path | None]:
    """Raw HTML of the dohod.ru and smart-lab.ru dividend pages (None if the page does not exist)."""
    out = {}
    for src, url in [("dohod", f"https://www.dohod.ru/ik/analytics/dividend/{secid.lower()}"),
                     ("smartlab", f"https://smart-lab.ru/q/{secid}/dividend/")]:
        path = RAW / "dividends" / src / f"{secid}.html"
        miss = path.with_suffix(".404")
        if path.exists():
            out[src] = path
        elif miss.exists():
            out[src] = None
        else:
            r = _get(url, ok404=True)
            time.sleep(0.3)
            if r is None:
                _atomic_write(miss, b"")
                out[src] = None
            else:
                _atomic_write(path, r.content)
                out[src] = path
    return out


class _Tables(HTMLParser):
    """Minimal HTML table extractor (stdlib only; text of each td/th, whitespace-collapsed)."""
    def __init__(self):
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self._row = None
        self._cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.tables.append([])
        elif tag == "tr" and self.tables:
            self._row = []
            self.tables[-1].append(self._row)
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def html_tables(html: str) -> list[list[list[str]]]:
    p = _Tables()
    p.feed(html)
    return p.tables


def _num(s: str) -> float:
    s = s.replace("₽", "").replace("\xa0", "").replace(" ", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


def _div_frame(rows: list[dict]) -> pd.DataFrame:
    """Dividend rows with fixed dtypes (also when empty)."""
    df = pd.DataFrame(rows, columns=["secid", "registry_date", "last_cum_date", "amount"])
    for c in ["registry_date", "last_cum_date"]:
        df[c] = pd.to_datetime(df[c]).astype("datetime64[ns]")
    df["amount"] = df["amount"].astype(float)
    df["secid"] = df["secid"].astype(object)
    return df


def parse_dohod(html: str, secid: str) -> pd.DataFrame:
    """Rows: secid, registry_date, amount (RUB per share); forecasts '(прогноз)' are dropped."""
    for t in html_tables(html):
        if t and t[0][:2] == ["Дата объявления дивиденда", "Дата закрытия реестра"]:
            rows = []
            for r in t[1:]:
                if len(r) < 4 or "прогноз" in r[1]:
                    continue
                d = pd.to_datetime(r[1], format="%d.%m.%Y", errors="coerce")
                a = _num(r[3])
                if pd.notna(d) and np.isfinite(a) and a > 0:
                    rows.append({"secid": secid, "registry_date": d, "last_cum_date": pd.NaT, "amount": a})
            return _div_frame(rows)
    return _div_frame([])


def parse_smartlab(html: str, secid: str) -> pd.DataFrame:
    """Rows of every dividend table with the (ticker, T-1, record date) header for this ticker.

    A page can carry an 'expected dividends' table *before* the 'paid dividends' table (same header);
    reading only the first matching table silently dropped the whole paid history (VERIFY stage fix).
    Rows of all matching tables are pooled; a row that repeats a row of an *earlier table* (listed both
    as expected and paid) is kept once. Repeats within one table are kept (two payments with the same
    record date and amount are summed downstream, as for dohod.ru). Future record dates are harmless:
    they fall outside the price history.
    """
    rows, seen = [], set()
    for t in html_tables(html):
        hdr = next((r for r in t[:3] if r[:3] == ["Тикер", "дата T-1", "дата отсечки"]), None)
        if hdr is None:
            continue
        this = []
        for r in t:
            if len(r) < 5 or r[0] != secid or "₽" not in r[4]:
                continue
            c = pd.to_datetime(r[1], format="%d.%m.%Y", errors="coerce")
            d = pd.to_datetime(r[2], format="%d.%m.%Y", errors="coerce")
            a = _num(r[4])
            if pd.notna(d) and np.isfinite(a) and a > 0 and (c, d, a) not in seen:
                this.append({"secid": secid, "registry_date": d, "last_cum_date": c, "amount": a})
        seen |= {(x["last_cum_date"], x["registry_date"], x["amount"]) for x in this}
        rows += this
    return _div_frame(rows)


def universe_frame(snapshots: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for month, snap in snapshots.items():
        if not snap["data"]:
            continue
        df = pd.DataFrame(snap["data"], columns=snap["columns"])
        df["month"] = month
        df["snap_date"] = snap["date"]
        rows.append(df[["month", "snap_date", "SECID", "NUMTRADES", "VALUE"]])
    u = pd.concat(rows, ignore_index=True).rename(columns=str.lower)
    return u.drop_duplicates(["month", "secid"])


def history_frame(hist: dict, secid: str) -> pd.DataFrame:
    df = pd.DataFrame(hist["data"], columns=hist["columns"])
    if df.empty:
        return pd.DataFrame(columns=["secid", "date", "close", "value", "volume", "numtrades"])
    df = df.rename(columns={"TRADEDATE": "date", "SECID": "secid", "CLOSE": "close", "VALUE": "value",
                            "VOLUME": "volume", "NUMTRADES": "numtrades"})
    df["secid"] = secid
    df["date"] = pd.to_datetime(df["date"])
    for c in ["close", "value", "volume", "numtrades"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["secid", "date", "close", "value", "volume", "numtrades"]]


def market_calendar(hist: pd.DataFrame) -> pd.DatetimeIndex:
    """Trading days of the board = dates on which at least one SECID traded."""
    return pd.DatetimeIndex(np.sort(hist.loc[hist["value"] > 0, "date"].unique()))


def ex_dates(divs: pd.DataFrame, cal: pd.DatetimeIndex, t1_from: str) -> pd.Series:
    """Ex-dividend date for each row: next trading day after the last cum day.

    last cum day: given (smart-lab 'T-1'), else the latest trading day c with settle(c) <= R, where
    settle(c) = c + L trading days, L = 2 before `t1_from` (T+2 regime), 1 from it (T+1).
    """
    cal = pd.DatetimeIndex(cal).sort_values()
    out = []
    for _, d in divs.iterrows():
        c = d.get("last_cum_date", pd.NaT)
        if pd.isna(c):
            R = d["registry_date"]
            L = 1 if R >= pd.Timestamp(t1_from) else 2
            j = cal.searchsorted(R, side="right") - 1
            i = j - L
            if i < 0:
                out.append(pd.NaT)
                continue
            c = cal[i]
        k = cal.searchsorted(pd.Timestamp(c), side="right")
        out.append(cal[k] if k < len(cal) else pd.NaT)
    return pd.Series(out, index=divs.index, dtype="datetime64[ns]")


def infer_splits(hist: pd.DataFrame, splits: pd.DataFrame, spec: dict = DATA_SPEC) -> pd.DataFrame:
    """Unrecorded splits by the rule in DATA_SPEC (columns secid, date, before, after)."""
    h = hist[(hist["value"] > 0) & hist["close"].notna() & (hist["close"] > 0)].sort_values(["secid", "date"])
    rec = set(zip(splits["secid"], pd.to_datetime(splits["date"])))
    out, w = [], int(spec["infer_window"])
    for sec, g in h.groupby("secid", sort=False):
        c, v, dts = g["close"].to_numpy(float), g["volume"].to_numpy(float), g["date"].to_numpy()
        ratio = c[1:] / c[:-1]
        for i in np.flatnonzero(np.abs(np.log(ratio)) > np.log(spec["split_guard"])):
            k = int(np.round(np.log10(ratio[i])))
            if k == 0 or abs(ratio[i] / 10.0 ** k - 1) > spec["infer_price_tol"]:
                continue
            if any((sec, pd.Timestamp(dts[i + 1]) - pd.Timedelta(days=dd)) in rec for dd in range(0, 40)):
                continue
            vb, va = np.median(v[max(0, i + 1 - w): i + 1]), np.median(v[i + 1: i + 1 + w])
            if vb <= 0 or va <= 0:
                continue
            vol_ratio = va / vb
            if abs(np.log(vol_ratio * 10.0 ** k)) <= np.log(spec["infer_volume_tol"]):
                before, after = (10 ** k, 1) if k > 0 else (1, 10 ** (-k))
                out.append({"secid": sec, "date": pd.Timestamp(dts[i + 1]), "before": before, "after": after})
    return pd.DataFrame(out, columns=["secid", "date", "before", "after"])


def build_daily(hist: pd.DataFrame, splits: pd.DataFrame, divs: pd.DataFrame, spec: dict = DATA_SPEC,
                cal: pd.DatetimeIndex | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily trade-day panel with split- and dividend-adjusted returns.

    hist   : secid, date, close, value, volume
    splits : secid, date, before, after       (after/before new shares per old share)
    divs   : secid, ex_date, amount           (RUB per share in today's units if spec['div_split_adjusted'])
    Returns (daily, flags): daily has symbol, date, close, value, volume, volume_adj, div, split,
    gross, r, valid, tri, seg, month, vol_mult (today's shares per share of that date; price in today's
    units = close / vol_mult); flags lists dropped dividends / unexplained jumps.
    """
    h = hist[(hist["value"] > 0) & hist["close"].notna() & (hist["close"] > 0)].copy()
    h = h.sort_values(["secid", "date"]).reset_index(drop=True)
    cal = market_calendar(hist) if cal is None else pd.DatetimeIndex(cal)
    flags = []
    splits = splits[["secid", "date", "before", "after"]].copy()
    ver = [v for v in spec.get("verified_splits", []) if v["secid"] in set(h["secid"])]
    if ver:
        vs = pd.DataFrame([{k: v[k] for k in ["secid", "date", "before", "after"]} for v in ver])
        vs["date"] = pd.to_datetime(vs["date"])
        splits = pd.concat([splits, vs], ignore_index=True)
        for v in ver:
            flags.append({"secid": v["secid"], "date": pd.Timestamp(v["date"]), "kind": "verified_split",
                          "value": float(v["before"]) / float(v["after"])})
    keep_big = {(v["secid"], pd.Timestamp(v["registry_date"])) for v in spec.get("verified_large_dividends", [])}
    if spec.get("infer_splits", False):
        inf = infer_splits(hist, splits, spec)
        for _, s in inf.iterrows():
            flags.append({"secid": s["secid"], "date": s["date"], "kind": "inferred_split",
                          "value": float(s["before"]) / float(s["after"])})
        splits = pd.concat([splits, inf], ignore_index=True) if len(inf) else splits
    h["prev_date"] = h.groupby("secid")["date"].shift(1)
    h["prev_close"] = h.groupby("secid")["close"].shift(1)

    h["split"] = 1.0
    h["vol_mult"] = 1.0
    sp = splits.copy()
    for _, s in sp.iterrows():
        m = h["secid"] == s["secid"]
        f = float(s["after"]) / float(s["before"])
        hit = m & (h["date"] >= s["date"]) & (h["prev_date"] < s["date"])
        h.loc[hit, "split"] *= f
        h.loc[m & (h["date"] < s["date"]), "vol_mult"] *= f
    h["volume_adj"] = h["volume"] * h["vol_mult"]

    h["div"] = 0.0
    if len(divs):
        for sec, dv in divs.dropna(subset=["ex_date"]).groupby("secid"):
            idx = h.index[h["secid"] == sec]
            if len(idx) == 0:
                continue
            dates = h.loc[idx, "date"].to_numpy()
            for _, d in dv.iterrows():
                k = np.searchsorted(dates, np.datetime64(d["ex_date"]), side="left")
                if k == 0 or k >= len(idx):
                    continue
                row = idx[k]
                pc = h.at[row, "prev_close"]
                amt = d["amount"] * (h.at[row, "vol_mult"] if spec.get("div_split_adjusted", True) else 1.0)
                verified = (sec, pd.Timestamp(d["registry_date"])) in keep_big if "registry_date" in d else False
                if verified:
                    flags.append({"secid": sec, "date": h.at[row, "date"], "kind": "verified_large_dividend",
                                  "value": amt / pc})
                elif amt / pc > spec["max_div_yield"]:
                    flags.append({"secid": sec, "date": h.at[row, "date"], "kind": "dividend_dropped",
                                  "value": amt / pc})
                    continue
                h.at[row, "div"] += amt

    h["gross"] = (h["close"] + h["div"]) * h["split"] / h["prev_close"]
    jump = h["gross"].notna() & (np.abs(np.log(h["gross"])) > np.log(spec["split_guard"])) & (h["split"] == 1.0)
    for _, j in h[jump].iterrows():
        flags.append({"secid": j["secid"], "date": j["date"], "kind": "unexplained_jump", "value": j["gross"]})
    h.loc[jump, "gross"] = np.nan

    pos = pd.Series(np.arange(len(cal)), index=cal)
    p_now = h["date"].map(pos)
    p_prev = h["prev_date"].map(pos)
    consecutive = (p_now - p_prev) == 1
    h0, h1 = pd.Timestamp(spec["halt"][0]), pd.Timestamp(spec["halt"][1])
    spans_halt = (h["prev_date"] < h0) & (h["date"] > h1)
    h["r"] = np.where(consecutive & ~spans_halt, h["gross"] - 1.0, np.nan)
    h["valid"] = h["r"].notna() & np.isfinite(h["r"]) & (h["value"] > 0)

    h["seg"] = (h["gross"].isna() | (h["secid"] != h["secid"].shift(1))).cumsum()
    gg = h["gross"].where(h["gross"].notna(), 1.0)
    h["tri"] = np.exp(np.log(gg).groupby(h["seg"]).cumsum())
    h["month"] = h["date"].dt.to_period("M")
    h = h.rename(columns={"secid": "symbol"})
    cols = ["symbol", "date", "month", "close", "value", "volume", "volume_adj", "vol_mult", "div", "split",
            "gross", "r", "valid", "tri", "seg"]
    return h[cols].reset_index(drop=True), pd.DataFrame(flags, columns=["secid", "date", "kind", "value"])


def monthly_tri_returns(daily: pd.DataFrame) -> pd.DataFrame:
    """ret_m = tri at last trade of month m / tri at last trade of month m-1 - 1, same chain only."""
    last = daily.sort_values(["symbol", "date"]).groupby(["symbol", "month"]).agg(
        tri=("tri", "last"), seg=("seg", "last"), seg_first=("seg", "first")).reset_index()
    prev = last.groupby("symbol").shift(1)
    same = (prev["month"] == last["month"] - 1) & (prev["seg"] == last["seg_first"]) & \
           (last["seg"] == last["seg_first"])
    last["ret"] = np.where(same, last["tri"] / prev["tri"] - 1.0, np.nan)
    return last[["symbol", "month", "ret"]]


def crosscheck_dividends(secids: list[str], cal: pd.DatetimeIndex, spec: dict = DATA_SPEC,
                         tol_days: int = 5, tol_amount: float = 0.02) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare dohod.ru (registry date -> ex-date by the settlement rule) with smart-lab.ru (T-1 given).

    Uses only cached pages (no download). Returns (per-event matches, one-row summary):
    events are matched when registry dates differ by <= tol_days and amounts by <= tol_amount.
    """
    lo, hi = pd.Timestamp(spec["history_from"]), pd.Timestamp(spec["history_till"])
    rows = []
    for s in secids:
        pd_, ps = RAW / "dividends" / "dohod" / f"{s}.html", RAW / "dividends" / "smartlab" / f"{s}.html"
        if not (pd_.exists() and ps.exists()):
            continue
        a = parse_dohod(pd_.read_text(encoding="utf-8", errors="replace"), s)
        b = parse_smartlab(ps.read_text(encoding="utf-8", errors="replace"), s)
        a = a[(a["registry_date"] >= lo) & (a["registry_date"] <= hi)].reset_index(drop=True)
        b = b[(b["registry_date"] >= lo) & (b["registry_date"] <= hi)].reset_index(drop=True)
        if a.empty and b.empty:
            continue
        a["ex"] = ex_dates(a, cal, spec["t1_from"]) if len(a) else pd.Series(dtype="datetime64[ns]")
        b["ex"] = ex_dates(b, cal, spec["t1_from"]) if len(b) else pd.Series(dtype="datetime64[ns]")
        used = set()
        for _, x in a.iterrows():
            cand = b[(np.abs((b["registry_date"] - x["registry_date"]).dt.days) <= tol_days) &
                     (np.abs(b["amount"] / x["amount"] - 1) <= tol_amount) & ~b.index.isin(used)]
            if len(cand):
                y = cand.iloc[0]
                used.add(cand.index[0])
                rows.append({"secid": s, "registry_dohod": x["registry_date"], "amount": x["amount"],
                             "ex_rule": x["ex"], "ex_smartlab": y["ex"], "matched": True})
            else:
                rows.append({"secid": s, "registry_dohod": x["registry_date"], "amount": x["amount"],
                             "ex_rule": x["ex"], "ex_smartlab": pd.NaT, "matched": False})
        for k, y in b.iterrows():
            if k not in used:
                rows.append({"secid": s, "registry_dohod": pd.NaT, "amount": y["amount"], "ex_rule": pd.NaT,
                             "ex_smartlab": y["ex"], "matched": False})
    ev = pd.DataFrame(rows, columns=["secid", "registry_dohod", "amount", "ex_rule", "ex_smartlab", "matched"])
    m = ev[ev["matched"]]
    summ = pd.DataFrame([{
        "secids_both_pages": int(ev["secid"].nunique()),
        "events_dohod": int(ev["registry_dohod"].notna().sum()),
        "events_smartlab_only": int((ev["registry_dohod"].isna()).sum()),
        "events_dohod_unmatched": int((ev["registry_dohod"].notna() & ~ev["matched"]).sum()),
        "events_matched": int(len(m)),
        "share_ex_date_equal": float((m["ex_rule"] == m["ex_smartlab"]).mean()) if len(m) else np.nan,
        "share_ex_date_within_1_trading_day": float(
            (np.abs(cal.searchsorted(m["ex_rule"]) - cal.searchsorted(m["ex_smartlab"])) <= 1).mean()) if len(m) else np.nan,
    }])
    return ev, summ


def download_all(spec: dict = DATA_SPEC, workers: int = 6) -> dict[str, pd.DataFrame]:
    """Download (or read from the raw cache) everything and return parsed frames."""
    months = month_ends(spec["universe_first_month"], pd.Timestamp(spec["history_till"]).strftime("%Y-%m"))
    with ThreadPoolExecutor(workers) as ex:
        snaps = dict(zip([f"{m:%Y-%m}" for m in months], ex.map(fetch_universe_month, months)))
    uni = universe_frame(snaps)
    secids = sorted(uni["secid"].unique())
    log.info("universe: %d month-ends, %d distinct SECIDs", uni["month"].nunique(), len(secids))

    with ThreadPoolExecutor(workers) as ex:
        infos = dict(zip(secids, ex.map(fetch_info, secids)))
    info = pd.DataFrame([{"secid": s, "type": i.get("TYPE"), "issuesize": pd.to_numeric(i.get("ISSUESIZE"), errors="coerce"),
                          "isin": i.get("ISIN"), "name": i.get("NAME"), "listlevel": i.get("LISTLEVEL")}
                         for s, i in infos.items()])
    common = sorted(info.loc[info["type"] == "common_share", "secid"])
    log.info("common shares: %d of %d", len(common), len(secids))

    def _h(s):
        return history_frame(fetch_history(s, spec["history_from"], spec["history_till"]), s)
    with ThreadPoolExecutor(workers) as ex:
        hist = pd.concat(list(ex.map(_h, common)), ignore_index=True)

    sp = fetch_splits()
    splits = pd.DataFrame(sp["data"], columns=sp["columns"]).rename(columns={"tradedate": "date"})
    splits["date"] = pd.to_datetime(splits["date"])
    splits = splits[splits["secid"].isin(common)].reset_index(drop=True)

    with ThreadPoolExecutor(2) as ex:
        pages = dict(zip(common, ex.map(fetch_dividend_pages, common)))
    dv = []
    for s, pg in pages.items():
        if pg["dohod"] is not None:
            d = parse_dohod(pg["dohod"].read_text(encoding="utf-8", errors="replace"), s)
            d["source"] = "dohod"
        elif pg["smartlab"] is not None:
            d = parse_smartlab(pg["smartlab"].read_text(encoding="utf-8", errors="replace"), s)
            d["source"] = "smartlab"
        else:
            continue
        dv.append(d)
    divs = pd.concat([d for d in dv if len(d)], ignore_index=True) if dv else pd.DataFrame(
        columns=["secid", "registry_date", "last_cum_date", "amount", "source"])
    pages_df = pd.DataFrame([{"secid": s, "dohod": p["dohod"] is not None, "smartlab": p["smartlab"] is not None}
                             for s, p in pages.items()])
    return {"universe": uni, "info": info, "history": hist, "splits": splits, "dividends": divs,
            "dividend_pages": pages_df}


def build_all(frames: dict[str, pd.DataFrame], spec: dict = DATA_SPEC) -> dict[str, pd.DataFrame]:
    hist, divs = frames["history"], frames["dividends"].copy()
    cal = market_calendar(hist)
    hist = hist[hist["date"] <= pd.Timestamp(spec["history_till"])]
    divs = divs[(divs["registry_date"] >= pd.Timestamp(spec["history_from"])) &
                (divs["registry_date"] <= pd.Timestamp(spec["history_till"]) + pd.Timedelta(days=10))]
    divs = divs.groupby(["secid", "registry_date"], as_index=False).agg(
        last_cum_date=("last_cum_date", "first"), amount=("amount", "sum"), source=("source", "first"))
    divs["ex_date"] = ex_dates(divs, cal, spec["t1_from"])
    daily, flags = build_daily(hist, frames["splits"], divs, spec, cal)
    return {"daily": daily, "dividends": divs, "flags": flags}


def load_or_build(spec: dict = DATA_SPEC, workers: int = 6, rebuild: bool = False) -> dict[str, pd.DataFrame]:
    """Cached parquet frames; downloads/builds only what is missing."""
    names = ["universe", "info", "history", "splits", "dividends", "dividend_pages", "daily", "flags"]
    paths = {n: CACHE / f"moex_{n}.parquet" for n in names}
    stamp_path = CACHE / "moex_build.json"
    stamp = build_stamp(spec)
    fresh = stamp_path.exists() and json.loads(stamp_path.read_text()) == stamp
    if not rebuild and fresh and all(p.exists() for p in paths.values()):
        return {n: pd.read_parquet(p) for n, p in paths.items()}
    frames = download_all(spec, workers)
    built = build_all(frames, spec)
    frames["dividends"] = built["dividends"]
    frames["daily"] = built["daily"]
    frames["flags"] = built["flags"]
    CACHE.mkdir(parents=True, exist_ok=True)
    for n, df in frames.items():
        d = df.copy()
        for c in d.columns:
            if isinstance(d[c].dtype, pd.PeriodDtype):
                d[c] = d[c].astype(str)
        d.to_parquet(paths[n], index=False)
    stamp_path.write_text(json.dumps(stamp, indent=2))
    return load_or_build(spec, workers, rebuild=False)


def build_stamp(spec: dict) -> dict:
    """Parquet caches are rebuilt (from the raw cache, offline) when the builder code or spec changes."""
    import hashlib
    return {"data_moex_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "spec_sha256": hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--rebuild", action="store_true", help="re-parse raw cache into data/cache/moex_*.parquet")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    fr = load_or_build(DATA_SPEC, a.workers, a.rebuild)
    for n, df in fr.items():
        log.info("%s: %d rows", n, len(df))


if __name__ == "__main__":
    main()
