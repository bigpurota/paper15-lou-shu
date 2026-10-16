"""Data-quality audit of the Binance daily klines: find price discontinuities that are not market
moves and repair them CRSP-style before any measure or return is computed.

Why this is needed: Binance klines are not adjusted for token redenominations, splits and swaps
(CRSP adjusts stock splits with its cumulative factor), and tickers are sometimes reused for a new
token (LUNA after the Terra collapse). One unadjusted 1000:1 redenomination makes a coin-month
return of +170 000% and dominates every equal-weighted portfolio.

Key fact used by the rules: Binance trades 24/7, so the open of day d equals the close of day d-1
for every genuine move (crashes included: LUNA on 2022-05-12 opened at 1.0012 x the previous close
and fell intraday). A redenomination is done during a trading halt, and trading reopens at
prev_close / F with F a power of ten (new units per old unit); base volume then moves by ~F while
dollar volume stays continuous.

Event types (row = one event; `factor` is F for redenominations, else open/prev_close or NaN):
  redenomination       open/prev_close within `round_tol_log10` of 10^-k (|k| >= 1), gap below
                       `new_asset_min_gap_days`, and base volume moves by ~F with dollar volume
                       continuous (`vol_tol_log10`, `dvol_tol_log10`). Treatment: all earlier prices
                       of the asset are divided by F and base volume multiplied by F (CRSP-like),
                       so daily returns, monthly returns and A are continuous.
  discontinuity        same price signature but the volume evidence fails, or a non-round jump of
                       at least `min_factor` across a halt (e.g. a relaunch at a new price).
                       Treatment: new asset id from the event date on (no splicing).
  ticker_reuse         halt of at least `new_asset_min_gap_days` with a price jump >= `min_factor`
                       (LUNA 2022-05-31 = Terra 2.0). Treatment: new asset id.
  long_gap             halt of at least `new_asset_min_gap_days` without such a jump (delisting
                       and relisting). Treatment: new asset id, so no return, beta or momentum
                       window spans a stale price.
  zero_volume          days with no trades (Binance repeats the last close). Treatment: row dropped.
  broken_print         non-positive/non-finite price or close outside [low, high].
                       Treatment: row dropped.
  open_mismatch        |log10(open/prev_close)| >= log10(`open_mismatch_ratio`) without a halt and
                       not classified above (e.g. end of a pre-market phase). Treatment: none,
                       reported only (all returns are close-to-close).
A new asset id is <SYMBOL>#k (k = 2, 3, ...).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULTS = {
    "enabled": True,
    "min_factor": 8.0,                 # |price jump across a halt| that cannot be ignored
    "round_tol_log10": 0.1,            # open/prev_close within 10^(+-0.1) (~26%) of a power of ten
    "new_asset_min_gap_days": 14,
    "volume_window_days": 7,           # trading days before/after the event for volume medians
    "vol_tol_log10": 1.0,              # base-volume ratio within one order of magnitude of F
    "dvol_tol_log10": 1.0,             # dollar-volume ratio within one order of magnitude of 1
    "open_mismatch_ratio": 1.5,
}
PRICE_COLS = ["open", "high", "low", "close"]
EVENT_COLS = ["symbol", "date", "type", "factor", "action", "evidence"]
ACTIONS = {"redenomination": "split_adjust", "discontinuity": "new_asset", "ticker_reuse": "new_asset",
           "long_gap": "new_asset", "zero_volume": "drop_row", "broken_print": "drop_row",
           "open_mismatch": "none"}


def audit_config(cfg: dict | None) -> dict:
    return DEFAULTS | dict((cfg or {}).get("audit") or {})


def _bad_rows(d: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(zero_volume, broken_print) boolean masks."""
    zero = ~(d["quote_volume"] > 0)
    px = [c for c in PRICE_COLS if c in d]
    vals = d[px].to_numpy(float)
    broken = pd.Series(~np.isfinite(vals).all(axis=1) | (vals <= 0).any(axis=1), index=d.index)
    if {"high", "low"} <= set(d):
        eps = 1e-9 * d["close"].abs()
        broken |= (d["close"] > d["high"] + eps) | (d["close"] < d["low"] - eps) | (d["high"] < d["low"])
    return zero & ~broken, broken


def _runs(dates: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """Group sorted dates into runs of consecutive days: (first, last, n)."""
    out, start, prev, n = [], None, None, 0
    for t in dates:
        if prev is not None and t - prev == pd.Timedelta(days=1):
            prev, n = t, n + 1
            continue
        if start is not None:
            out.append((start, prev, n))
        start, prev, n = t, t, 1
    if start is not None:
        out.append((start, prev, n))
    return out


def _vol_ratio(x: pd.DataFrame, i: int, col: str, w: int) -> float:
    if col not in x:
        return np.nan
    pre = x[col].iloc[max(0, i - w):i]
    post = x[col].iloc[i:i + w]
    a, b = pre[pre > 0].median(), post[post > 0].median()
    return float(b / a) if a > 0 and b > 0 else np.nan


def detect_events(daily: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """All audit events of a daily panel (symbol, date, close, quote_volume [, open, high, low, volume])."""
    a = audit_config(cfg)
    d = daily.sort_values(["symbol", "date"]).reset_index(drop=True)
    events = []
    zero, broken = _bad_rows(d)
    for (sym, kind), g in pd.concat([d.loc[zero, ["symbol", "date"]].assign(k="zero_volume"),
                                     d.loc[broken, ["symbol", "date"]].assign(k="broken_print")]).groupby(["symbol", "k"]):
        for first, last, n in _runs(g["date"].sort_values()):
            events.append({"symbol": sym, "date": first, "type": kind, "factor": np.nan,
                           "evidence": f"{n} day(s) {first:%Y-%m-%d}..{last:%Y-%m-%d}"})

    lmin, tol, w = np.log10(a["min_factor"]), a["round_tol_log10"], int(a["volume_window_days"])
    mism = np.log10(a["open_mismatch_ratio"])
    for sym, x in d[~(zero | broken)].groupby("symbol", sort=True):
        x = x.reset_index(drop=True)
        gap = x["date"].diff().dt.days
        ref = x["open"] if "open" in x else x["close"]       # reopening price after a halt
        lr = np.log10(ref / x["close"].shift(1))
        cand = x.index[(gap > 1) | (lr.abs() >= mism)]
        for i in cand:
            g, l = float(gap[i]), float(lr[i])
            k = round(l)
            is_round = abs(k) >= 1 and abs(l - k) <= tol
            F = 10.0 ** (-k) if is_round else np.nan
            vr, dvr = _vol_ratio(x, i, "volume", w), _vol_ratio(x, i, "quote_volume", w)
            vol_ok = (np.isfinite(vr) and abs(np.log10(vr) - np.log10(F)) <= a["vol_tol_log10"]) if is_round else False
            dvol_ok = np.isfinite(dvr) and abs(np.log10(dvr)) <= a["dvol_tol_log10"]
            if "volume" not in x:                            # no base volume: price evidence only
                vol_ok = is_round
            ev = (f"gap={g:.0f}d; prev_close={x['close'].iloc[i - 1]:.6g}; reopen={ref.iloc[i]:.6g}; "
                  f"reopen/prev_close={10 ** l:.6g}; base_vol_ratio={vr:.4g}; dollar_vol_ratio={dvr:.4g}")
            if g >= a["new_asset_min_gap_days"]:
                kind, fac = ("ticker_reuse" if abs(l) >= lmin else "long_gap"), 10 ** l
            elif is_round and vol_ok and dvol_ok:
                kind, fac = "redenomination", F
            elif abs(l) >= lmin:
                kind, fac = "discontinuity", 10 ** l
            elif g == 1 and abs(l) >= mism:
                kind, fac = "open_mismatch", 10 ** l
            else:
                continue                                     # ordinary short halt, price continuous
            events.append({"symbol": sym, "date": x["date"].iloc[i], "type": kind, "factor": fac, "evidence": ev})
    ev = pd.DataFrame(events, columns=["symbol", "date", "type", "factor", "evidence"])
    ev["action"] = ev["type"].map(ACTIONS)
    return ev[EVENT_COLS].sort_values(["symbol", "date", "type"]).reset_index(drop=True)


def apply_events(daily: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Repair the daily panel: drop bad rows, split into new asset ids, split-adjust earlier prices."""
    d = daily.sort_values(["symbol", "date"]).reset_index(drop=True).copy()
    drop = events[events["action"] == "drop_row"]
    if len(drop):
        zero, broken = _bad_rows(d)
        d = d[~(zero | broken)].reset_index(drop=True)
    # new asset ids: cumulative count of breaks per symbol
    brk = events[events["action"] == "new_asset"]
    if len(brk):
        seg = pd.Series(0, index=d.index)
        for sym, e in brk.groupby("symbol"):
            m = d["symbol"] == sym
            for t in e["date"]:
                seg[m & (d["date"] >= t)] += 1
        d["symbol"] = np.where(seg > 0, d["symbol"].astype(str) + "#" + (seg + 1).astype(str), d["symbol"].astype(str))
        d["symbol"] = d["symbol"].astype(str)
    # split adjustment within the asset that contains the event (after the re-labelling)
    adj = events[events["action"] == "split_adjust"]
    if len(adj):
        div = pd.Series(1.0, index=d.index)
        for _, e in adj.iterrows():
            base_sym = d["symbol"].str.split("#").str[0]
            # the asset id that holds the event date
            hold = d.loc[(base_sym == e["symbol"]) & (d["date"] == e["date"]), "symbol"]
            if hold.empty:
                continue
            div[(d["symbol"] == hold.iloc[0]) & (d["date"] < e["date"])] *= float(e["factor"])
        for c in PRICE_COLS:
            if c in d:
                d[c] = d[c] / div
        if "volume" in d:
            d["volume"] = d["volume"] * div
    return d.sort_values(["symbol", "date"]).reset_index(drop=True)


def audit(daily: pd.DataFrame, cfg: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(clean daily, events, summary by type)."""
    ev = detect_events(daily, cfg)
    clean = apply_events(daily, ev)
    summ = (ev.groupby(["type", "action"]).agg(n_events=("symbol", "size"), n_symbols=("symbol", "nunique"))
            .reset_index())
    tot = pd.DataFrame([{"type": "TOTAL", "action": "", "n_events": len(ev), "n_symbols": ev["symbol"].nunique()},
                        {"type": "rows_in", "action": "", "n_events": len(daily), "n_symbols": daily["symbol"].nunique()},
                        {"type": "rows_out", "action": "", "n_events": len(clean), "n_symbols": clean["symbol"].nunique()}])
    return clean, ev, pd.concat([summ, tot], ignore_index=True)


# ---------------------------------------------------------------- diagnostics of the final panel
def return_diagnostics(panel: pd.DataFrame, daily: pd.DataFrame, q: int = 5, top: int = 10) -> dict[str, pd.DataFrame]:
    """Distribution of panel returns, the most extreme coin-months and their influence on the sorts.

    ret_distribution : percentiles of ret / ret_adj (percent per month), pooled over coin-months
    ret_extremes     : `top` highest and lowest ret with daily evidence from month t
                       (max/min daily return, number of days, mean daily dollar volume,
                       first/last close of the month, audit-adjusted)
    ret_influence    : coin-months with the largest contribution to the time-series mean of their
                       quintile portfolio in the T3 sorts, contribution = ret / (n_{q,t} * T), in
                       percentage points of the reported quintile mean
    """
    from sorts import assign_quantiles

    pct = [0, .001, .01, .05, .1, .25, .5, .75, .9, .95, .99, .999, 1]
    dist = pd.DataFrame({c: panel[c].quantile(pct) for c in ["ret", "ret_adj"]})
    dist.index = [f"p{100 * x:g}" for x in pct]
    dist.loc["mean"] = panel[["ret", "ret_adj"]].mean()
    dist.loc["sd"] = panel[["ret", "ret_adj"]].std()
    dist.loc["n"] = panel[["ret", "ret_adj"]].count()
    dist = dist.reset_index(names="stat")

    d = daily.sort_values(["symbol", "date"]).copy()
    d["month"] = d["date"].dt.to_period("M")
    prev = d.groupby("symbol")["date"].shift(1)
    d["r"] = np.where((d["date"] - prev) == pd.Timedelta(days=1), d["close"] / d.groupby("symbol")["close"].shift(1) - 1, np.nan)
    dm = d.groupby(["symbol", "month"]).agg(n_days=("close", "size"), max_daily_r=("r", "max"), min_daily_r=("r", "min"),
                                           mean_dvol_t=("quote_volume", "mean"), first_close=("close", "first"),
                                           last_close=("close", "last")).reset_index()
    p = panel.dropna(subset=["ret"])
    ext = pd.concat([p.nlargest(top, "ret").assign(side="top"), p.nsmallest(top, "ret").assign(side="bottom")])
    ext = ext[["side", "symbol", "month", "ret", "ret_adj", "A", "A_C", "dvol"]].rename(columns={"dvol": "dvol_t-2"})
    ext = ext.merge(dm, on=["symbol", "month"], how="left")

    rows = []
    for s in ["A", "A_C", "ResA"]:
        x = p[["symbol", "month", s, "ret"]].dropna().copy()
        x["q"] = assign_quantiles(x, s, q)
        x = x.dropna(subset=["q"])
        n_qt = x.groupby(["month", "q"])["ret"].transform("size")
        T = x["month"].nunique()
        x["contribution_pp"] = x["ret"] / (n_qt * T)
        x["quintile_mean"] = x.groupby("q")["contribution_pp"].transform("sum")
        x["signal"] = s
        rows.append(x.reindex(x["contribution_pp"].abs().sort_values(ascending=False).index).head(top))
    infl = pd.concat(rows, ignore_index=True)[["signal", "q", "symbol", "month", "ret", "contribution_pp", "quintile_mean"]]
    return {"ret_distribution": dist, "ret_extremes": ext.reset_index(drop=True), "ret_influence": infl}
