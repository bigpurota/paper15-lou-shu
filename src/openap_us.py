"""Upgrade 7: out-of-sample in time on US data (Chen & Zimmermann Open Source Asset Pricing).

Tests, on the published OpenAP long-short (L-S) series:
  (1) mean L-S return and Newey-West t of Illiquidity (Amihud 2002) and DolVol in the Lou-Shu
      sample 1964-2012 vs the post-sample 2013-latest (publication decay), plus the difference;
  (2) Table A.1 analogue: alpha of the Illiquidity L-S on the DolVol L-S (one factor; the paper
      uses IMLA_C, a tercile spread on A_C = 1/dollar volume) and on DolVol L-S + FF3 + UMD
      (five factors), in both subperiods;
  (3) correlation of the two L-S series.

Sign convention (OpenAP, SignalDoc column `Sign`): the published L-S return is
`Sign * (top quantile - bottom quantile)` of the signal, i.e. it is long the side that the original
paper says earns more. Illiquidity has Sign=+1 (long illiquid), DolVol has Sign=-1 (long LOW
dollar volume). So the DolVol L-S is oriented like the paper's IMLA_C (high A_C = low volume). The
signs are read from SignalDoc at run time and checked empirically (both L-S series must load
positively on SMB, since illiquid / low-volume stocks are small).

Usage:  python -I src/openap_us.py config.yaml [--results-dir results] [--out results/upgrades/openap_us]
Data:   data/raw/openap/release_<rel>/  and  data/raw/french/   (cached; downloads are idempotent)
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
import re
import subprocess
import time
import zipfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
import yaml

from run import rel_path

ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger("openap_us")

OPENAP_RELEASE = "202510"
OPENAP_FILES = {
    "PredictorLSretWide.csv": "10sOryk_ddjkXagaajTKUk1nwJs2ZLRiI",
    "SignalDoc.csv": "1rdi0jTPSA6xtn6TpQAMT5WyEczQ1gK59",
}
OPENAP_MIRROR = ("https://raw.githubusercontent.com/edwardlg/assip-2026-projects/"
                 "9c4d74d3b27c9ac26713f8d7f8ec20fcdac9291e/04_anomaly_publication_decay/data/raw/{name}")
FRENCH_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{name}"
FRENCH_FILES = ["F-F_Research_Data_Factors_CSV.zip", "F-F_Momentum_Factor_CSV.zip"]

OPENAP_DEFAULT_LS_QUANTILE = 0.2
OPENAP_TIMING = {
    "DolVol": "log dollar volume of a single month, t−3 (signal = volume at l2, applied to the next month's return)",
    "Illiquidity": "12-month average, formed in June and held for 12 months (lag 1–12 months, no skip month)",
}

MAIN_SIGNALS = ["Illiquidity", "DolVol"]
CONTEXT_SIGNALS = ["std_turn", "zerotrade1M", "zerotrade6M", "zerotrade12M", "zerotrade"]
PERIODS = {"LS_1964_2012": ("1964-01", "2012-12"), "post_2013": ("2013-01", None)}
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120 Safari/537.36"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def looks_like_html(head: bytes) -> bool:
    h = head.lstrip()[:200].lower()
    return h.startswith(b"<!doctype html") or h.startswith(b"<html")


def _get(url: str, timeout: int = 300) -> bytes:
    r = requests.get(url, headers=UA, timeout=timeout)
    r.raise_for_status()
    return r.content


def fetch(dest: Path, urls: list[str], kind: str = "csv") -> dict:
    """Download dest from the first url that returns real data; skip if a valid file is cached.

    Writes <dest>.source.json with the url used, time and sha256. Returns that record.
    """
    meta_path = dest.with_name(dest.name + ".source.json")
    if dest.exists() and dest.stat().st_size > 0 and not looks_like_html(dest.read_bytes()[:512]):
        rec = json.loads(meta_path.read_text()) if meta_path.exists() else {"url": None}
        rec.update({"file": rel_path(dest), "sha256": sha256_file(dest), "cached": True})
        return rec
    dest.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in urls:
        try:
            body = _get(url)
        except requests.RequestException as e:
            errors.append(f"{url}: {e}")
            continue
        if looks_like_html(body[:512]):
            title = re.search(rb"<title>(.*?)</title>", body[:4000], re.S)
            errors.append(f"{url}: HTML page instead of data ({title.group(1).decode(errors='replace') if title else '?'})")
            continue
        if kind == "zip" and not zipfile.is_zipfile(io.BytesIO(body)):
            errors.append(f"{url}: not a zip")
            continue
        dest.write_bytes(body)
        rec = {"file": rel_path(dest), "url": url, "downloaded_utc": datetime.now(timezone.utc).isoformat(),
               "sha256": sha256_file(dest), "tried_before": errors}
        meta_path.write_text(json.dumps(rec, indent=2))
        rec["cached"] = False
        log.info("downloaded %s from %s", dest.name, url)
        return rec
    raise RuntimeError(f"could not download {dest.name}; tried:\n  " + "\n  ".join(errors))


def drive_url(file_id: str) -> str:
    return f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"


def download_all(raw: Path = ROOT / "data" / "raw") -> dict:
    recs = {}
    od = raw / "openap" / f"release_{OPENAP_RELEASE}"
    for name, fid in OPENAP_FILES.items():
        recs[f"openap/{name}"] = fetch(od / name, [drive_url(fid), OPENAP_MIRROR.format(name=name)])
    for name in FRENCH_FILES:
        recs[f"french/{name}"] = fetch(raw / "french" / name, [FRENCH_URL.format(name=name)], kind="zip")
    return recs


def load_ls_wide(path: Path | str) -> pd.DataFrame:
    """OpenAP PredictorLSretWide: date x signal, monthly L-S returns in percent. Index = Period[M]."""
    df = pd.read_csv(path)
    idx = pd.PeriodIndex(pd.to_datetime(df["date"]), freq="M", name="month")
    return df.drop(columns="date").set_index(idx).sort_index().astype(float)


def load_signal_doc(path: Path | str) -> pd.DataFrame:
    return pd.read_csv(path)


def parse_french_csv(text: str) -> pd.DataFrame:
    """Monthly block of a Kenneth French CSV (first table: rows 'YYYYMM, v1, v2, ...'). Percent."""
    lines = text.splitlines()
    hdr = next(i for i, l in enumerate(lines) if l.startswith(",") and re.match(r"^\s*\d{6}\s*,", lines[i + 1] or ""))
    cols = [c.strip() for c in lines[hdr].split(",")[1:]]
    rows = []
    for l in lines[hdr + 1:]:
        m = re.match(r"^\s*(\d{6})\s*,(.*)$", l)
        if not m:
            break
        vals = [float(v) for v in m.group(2).split(",")]
        rows.append([m.group(1)] + vals)
    df = pd.DataFrame(rows, columns=["yyyymm"] + cols)
    df["month"] = pd.PeriodIndex(pd.to_datetime(df.pop("yyyymm"), format="%Y%m"), freq="M")
    df = df.set_index("month")
    return df.mask(df.isin([-99.99, -999.0]))


def load_french(ff_zip: Path | str, mom_zip: Path | str) -> pd.DataFrame:
    def _read(p):
        with zipfile.ZipFile(p) as z:
            return z.read(z.namelist()[0]).decode("latin-1")
    ff = parse_french_csv(_read(ff_zip))
    mom = parse_french_csv(_read(mom_zip))
    mom.columns = ["UMD" if c.strip().lower() == "mom" else c for c in mom.columns]
    return ff.join(mom[["UMD"]], how="inner")


def ols_hac(y: pd.Series, X: pd.DataFrame | None, lags: int) -> dict:
    """OLS of y on const (+X) with Newey-West (Bartlett, `lags`) covariance. Returns a flat dict."""
    d = pd.concat([y.rename("y")] + ([X] if X is not None else []), axis=1).dropna()
    Z = sm.add_constant(d.drop(columns="y"), has_constant="add") if X is not None else np.ones((len(d), 1))
    res = sm.OLS(d["y"].to_numpy(float), np.asarray(Z, dtype=float)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    names = ["alpha"] + (list(X.columns) if X is not None else [])
    out = {"n": int(len(d)), "start": str(d.index.min()), "end": str(d.index.max())}
    for i, nm in enumerate(names):
        out[f"{nm}"] = float(res.params[i])
        out[f"t_{nm}"] = float(res.tvalues[i])
    out["r2"] = float(res.rsquared) if X is not None else np.nan
    return out


def slice_period(df: pd.DataFrame | pd.Series, start: str | None, end: str | None):
    idx = df.index
    m = np.ones(len(idx), bool)
    if start:
        m &= idx >= pd.Period(start, "M")
    if end:
        m &= idx <= pd.Period(end, "M")
    return df[m]


def mean_tests(ls: pd.DataFrame, periods: dict, lags: int) -> pd.DataFrame:
    """Mean and NW t per signal x period, plus the post-minus-pre difference (dummy regression)."""
    rows = []
    names = list(periods)
    for s in ls.columns:
        for p, (a, b) in periods.items():
            r = ols_hac(slice_period(ls[s], a, b), None, lags)
            rows.append({"signal": s, "period": p, "n": r["n"], "start": r["start"], "end": r["end"],
                         "mean": r["alpha"], "t": r["t_alpha"]})
        if len(names) == 2:
            pre = slice_period(ls[s], *periods[names[0]]).dropna()
            post = slice_period(ls[s], *periods[names[1]]).dropna()
            y = pd.concat([pre, post])
            X = pd.DataFrame({"post": np.r_[np.zeros(len(pre)), np.ones(len(post))]}, index=y.index)
            r = ols_hac(y, X, lags)
            rows.append({"signal": s, "period": f"{names[1]}_minus_{names[0]}", "n": r["n"], "start": r["start"],
                         "end": r["end"], "mean": r["post"], "t": r["t_post"]})
    return pd.DataFrame(rows)


def spanning_tests(ls: pd.DataFrame, ff: pd.DataFrame, y: str, x: str, periods: dict, lags: int) -> pd.DataFrame:
    """Table A.1 analogue: alpha of the y L-S on the x L-S, alone and with FF3 + UMD; plus FF4 of y alone."""
    data = ls[[y, x]].join(ff[["Mkt-RF", "SMB", "HML", "UMD"]], how="inner")
    specs = {"one_factor": [x], "five_factor": [x, "Mkt-RF", "SMB", "HML", "UMD"],
             "ff4_only": ["Mkt-RF", "SMB", "HML", "UMD"], "raw": []}
    rows = []
    for p, (a, b) in periods.items():
        d = slice_period(data, a, b)
        for spec, xs in specs.items():
            r = ols_hac(d[y], d[xs] if xs else None, lags)
            row = {"dep": y, "period": p, "spec": spec, "n": r["n"], "start": r["start"], "end": r["end"],
                   "alpha": r["alpha"], "t_alpha": r["t_alpha"], "r2": r["r2"]}
            for c in ["DolVol", "Mkt-RF", "SMB", "HML", "UMD"]:
                if c == x or c in xs:
                    row[f"b_{'factor' if c == x else c}"] = r.get(c, np.nan)
                    row[f"t_{'factor' if c == x else c}"] = r.get(f"t_{c}", np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def correlations(ls: pd.DataFrame, a: str, b: str, periods: dict) -> pd.DataFrame:
    rows = []
    for p, (s, e) in {"full": (None, None), **periods}.items():
        d = slice_period(ls[[a, b]], s, e).dropna()
        rows.append({"a": a, "b": b, "period": p, "n": len(d), "start": str(d.index.min()), "end": str(d.index.max()),
                     "pearson": d[a].corr(d[b]), "spearman": d[a].corr(d[b], method="spearman")})
    return pd.DataFrame(rows)


def decay_lag_sensitivity(ls: pd.DataFrame, periods: dict, lags_list=(0, 3, 6, 12), primary: int = 6) -> pd.DataFrame:
    """Robustness of the post-minus-pre t to the NW lag choice (the primary choice, 6 lags as in config.yaml = `primary`)."""
    out = []
    for L in lags_list:
        m = mean_tests(ls, periods, L)
        m = m[m["period"].str.contains("_minus_")].assign(nw_lags=L, primary=(L == primary))
        out.append(m[["signal", "period", "nw_lags", "primary", "n", "mean", "t"]])
    return pd.concat(out, ignore_index=True)


def sign_checks(ls: pd.DataFrame, ff: pd.DataFrame, doc: pd.DataFrame, signals: list[str], lags: int) -> pd.DataFrame:
    """Documented sign (SignalDoc) and the empirical SMB loading of each L-S (expected > 0 here)."""
    d = ls.join(ff, how="inner")
    rows = []
    for s in signals:
        r = ols_hac(d[s], d[["SMB"]], lags)
        meta = doc.loc[doc["Acronym"] == s]
        sign = meta["Sign"].iloc[0] if len(meta) and "Sign" in meta else np.nan
        rows.append({"signal": s, "role": "main" if s in MAIN_SIGNALS else "context", "doc_sign": sign,
                     "long_side": ("top (high signal)" if sign == 1 else "bottom (low signal)" if sign == -1 else "?"),
                     "corr_SMB": d[s].corr(d["SMB"]), "b_SMB": r["SMB"], "t_SMB": r["t_SMB"],
                     "smb_check_pass": (bool(r["SMB"] > 0 and r["t_SMB"] > 1.96) if s in MAIN_SIGNALS else None),
                     "first_month": str(ls[s].first_valid_index()), "last_month": str(ls[s].last_valid_index())})
    return pd.DataFrame(rows)


def signal_meta(doc: pd.DataFrame, signals: list[str]) -> pd.DataFrame:
    keep = [c for c in ["Acronym", "Sign", "Authors", "Year", "Journal", "SampleStartYear", "SampleEndYear",
                        "LongDescription", "Detailed Definition", "Stock Weight", "LS Quantile", "Quantile Filter",
                        "Portfolio Period", "Start Month", "Filter", "Cat.Signal", "Cat.Form", "Return", "T-Stat"]
            if c in doc.columns]
    out = doc.loc[doc["Acronym"].isin(signals), keep].reset_index(drop=True)
    lq = out["LS Quantile"] if "LS Quantile" in out else pd.Series(np.nan, index=out.index)
    out["LS Quantile effective"] = lq.fillna(OPENAP_DEFAULT_LS_QUANTILE)
    return out


def _f(x, nd=2):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def _na(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return str(x).strip()


def build_summary(means: pd.DataFrame, span: pd.DataFrame, corr: pd.DataFrame, signs: pd.DataFrame,
                  paper: dict, base_run: str, hlz: float, data_note: str, meta: pd.DataFrame | None = None,
                  sens: pd.DataFrame | None = None) -> str:
    pre, post = list(PERIODS)
    lab = {pre: "1964–2012", post: "2013–" }
    m = means.set_index(["signal", "period"])
    sp = span.set_index(["period", "spec"])
    a1 = paper["tableA1_panelA"]
    t3 = paper["table3_panelA"]["A"]
    def mrow(s, p):
        r = m.loc[(s, p)]
        return f"{_f(r['mean'])} (t {_f(r['t'])}, N={int(r['n'])}, {r['start']}–{r['end']})"
    L = ["# Upgrade 7: OpenAP (US), out of sample in time", "",
         f"Generated by `src/openap_us.py`; base run (crypto, recorded for traceability only): `{base_run}`.",
         f"Data: {data_note}", "",
         "**What we tested.** The Chen–Zimmermann long–short series `Illiquidity` (Amihud, 2002) and `DolVol` (dollar volume). "
         "(1) The premium in the Lou–Shu sample 1964–2012 and after 2013; (2) a Table A.1 analogue: the alpha of `Illiquidity` on `DolVol` "
         "(1 factor) and on `DolVol`+FF3+UMD (5 factors); (3) the correlation of the series. Newey–West t with 6 lags; "
         f"thresholds 1.96 and HLZ t>{hlz:g}.", "",
         "**OpenAP signs.** " + "; ".join(f"`{r.signal}`: Sign={r.doc_sign:+.0f}, long = {r.long_side}, "
                                           f"SMB loading {_f(r.b_SMB)} (t {_f(r.t_SMB)}), sign check "
                                           f"{'passed' if r.smb_check_pass else 'NOT passed'}"
                                           for r in signs.itertuples() if r.role == "main") +
         ". The `DolVol` L-S is low volume minus high volume, i.e. oriented like the paper's IMLA_C (high A_C = low volume). "
         "The other signals (`std_turn`, `zerotrade*`) are context only, in table 1.", "",
         "## 1. Mean L-S return, % per month", "",
         f"| signal | {lab[pre]} | {lab[post]} | difference (after − before) |", "|---|---|---|---|"]
    for s in means["signal"].unique():
        d = m.loc[(s, f"{post}_minus_{pre}")]
        L.append(f"| {s} | {mrow(s, pre)} | {mrow(s, post)} | {_f(d['mean'])} (t {_f(d['t'])}) |")
    L += ["", f"Paper (Table 3, EW quintiles of A, NYSE/AMEX): H−L {_f(t3['raw']['hl'])} (t {_f(t3['raw']['t'])}), "
              f"4-factor alpha {_f(t3['alpha']['hl'])} (t {_f(t3['alpha']['t'])}).", "",
          "## 2. Table A.1 analogue: alpha of the `Illiquidity` L-S", "",
          f"| specification | {lab[pre]} | {lab[post]} |", "|---|---|---|"]
    names = {"raw": "no factors (mean)", "ff4_only": "FF3+UMD", "one_factor": "1 factor: DolVol L-S",
             "five_factor": "5 factors: DolVol L-S + FF3 + UMD"}
    for spec, nm in names.items():
        cells = []
        for p in (pre, post):
            r = sp.loc[(p, spec)]
            b = f", β_DolVol {_f(r['b_factor'])}" if spec in ("one_factor", "five_factor") else ""
            cells.append(f"{_f(r['alpha'])} (t {_f(r['t_alpha'])}{b}, R² {_f(r['r2'])})" if spec != "raw"
                         else f"{_f(r['alpha'])} (t {_f(r['t_alpha'])})")
        L.append(f"| {nm} | {cells[0]} | {cells[1]} |")
    L += ["", f"Paper (Table A.1, Panel A, H−L of A quintiles): 1 factor IMLA_C {_f(a1['imla_one_factor']['alpha'])} "
              f"(t {_f(a1['imla_one_factor']['t'])}); 5 factors {_f(a1['imla_five_factor']['alpha'])} "
              f"(t {_f(a1['imla_five_factor']['t'])})."]
    b1 = paper.get("tableA1_panelB")
    if b1:
        L.append(f"Paper (Table A.1, Panel B, annual measures, closer to the 12-month OpenAP `Illiquidity`): "
                 f"1 factor {_f(b1['imla_one_factor']['alpha'])} (t {_f(b1['imla_one_factor']['t'])}); "
                 f"5 factors {_f(b1['imla_five_factor']['alpha'])} (t {_f(b1['imla_five_factor']['t'])}).")
    L += ["", "## 3. Correlation of the series", ""]
    for r in corr.itertuples():
        span_txt = f", {r.start}–{r.end}" if hasattr(r, "start") else ""
        L.append(f"- {r.period}: Pearson {_f(r.pearson)}, Spearman {_f(r.spearman)}, N={r.n}{span_txt}")
    o_pre, o_post = sp.loc[(pre, "one_factor")], sp.loc[(post, "one_factor")]
    r_pre, r_post = m.loc[("Illiquidity", pre)], m.loc[("Illiquidity", post)]
    def sig(t):
        return ("significant and clears HLZ" if abs(t) > hlz else "significant at 5% but does not clear HLZ" if abs(t) > 1.96
                else "not significant")
    d_pre = m.loc[("Illiquidity", f"{post}_minus_{pre}")]
    d_dv = m.loc[("DolVol", f"{post}_minus_{pre}")]
    absorbed = abs(r_pre["t"]) > 1.96 and abs(o_pre["t_alpha"]) <= 1.96
    def d_sig(signal, t):
        """sig() of an 'after - before' difference, unless clearing HLZ depends on the NW lag choice."""
        if sens is not None and len(sens):
            gi = sens[sens["signal"] == signal]
            if len(gi) and abs(t) > hlz and (gi["t"].abs() <= hlz).any():
                return "significant at 5%, borderline at HLZ (see the NW-lag sensitivity below)"
        return sig(t)
    f_pre, f_post = sp.loc[(pre, "ff4_only")], sp.loc[(post, "ff4_only")]
    L += ["", "## Interpretation (of the numbers above)", "",
          f"- `Illiquidity` premium in 1964–2012: {_f(r_pre['mean'])}% (t {_f(r_pre['t'])}, {sig(r_pre['t'])}); after 2013: "
          f"{_f(r_post['mean'])}% (t {_f(r_post['t'])}, {sig(r_post['t'])}); difference {_f(d_pre['mean'])} (t {_f(d_pre['t'])}, "
          f"{d_sig('Illiquidity', d_pre['t'])}). For `DolVol` the difference is {_f(d_dv['mean'])} (t {_f(d_dv['t'])}, {d_sig('DolVol', d_dv['t'])}).",
          f"- Alpha of `Illiquidity` on the `DolVol` L-S: {_f(o_pre['alpha'])}% (t {_f(o_pre['t_alpha'])}, {sig(o_pre['t_alpha'])}) "
          f"in 1964–2012 and {_f(o_post['alpha'])}% (t {_f(o_post['t_alpha'])}, {sig(o_post['t_alpha'])}) after 2013. "
          + ("In the Lou–Shu sample the significant Amihud premium becomes insignificant after controlling for the volume factor, "
             "qualitatively as in the paper's Table A.1."
             if absorbed else
             "The Table A.1 result (the Amihud premium is absorbed by the volume factor) is not cleanly reproduced "
             "on OpenAP data in 1964–2012; see the numbers."),
          (f"- Caveat to this conclusion: the `Illiquidity` alpha on FF3+UMD alone is also {sig(sp.loc[(pre, 'ff4_only')]['t_alpha'])} "
           f"({_f(sp.loc[(pre, 'ff4_only')]['alpha'])}%, t {_f(sp.loc[(pre, 'ff4_only')]['t_alpha'])}), so in the OpenAP version "
           "size/value/momentum already remove the premium, and the contribution of the volume factor itself is harder to isolate than in the paper."
           if abs(sp.loc[(pre, 'ff4_only')]['t_alpha']) <= 1.96 else
           f"- The `Illiquidity` alpha on FF3+UMD is significant ({_f(sp.loc[(pre, 'ff4_only')]['alpha'])}%, "
           f"t {_f(sp.loc[(pre, 'ff4_only')]['t_alpha'])}), so it is the volume factor that removes it."),
          ("- After 2013 there is no premium even without factors, so the 'volume or price' question is not identified "
           "in this subperiod." if abs(r_post["t"]) <= 1.96 else
           "- After 2013 the premium remains significant; see its alpha on the volume factor above."),
          (f"- The FF3+UMD alpha of `Illiquidity` is {_f(f_pre['alpha'])}% (t {_f(f_pre['t_alpha'])}, {sig(f_pre['t_alpha'])}) "
           f"in 1964–2012 and {_f(f_post['alpha'])}% (t {_f(f_post['t_alpha'])}, {sig(f_post['t_alpha'])}) after 2013, and its "
           f"SMB loading moves from {_f(f_pre['b_SMB'])} (t {_f(f_pre['t_SMB'])}) to {_f(f_post['b_SMB'])} (t {_f(f_post['t_SMB'])}). "
           + ("The alpha is insignificant in both periods, so there was no significant abnormal premium that could vanish; "
              "the fall of the raw L-S is consistent with lower size (SMB) exposure. Neither the alpha difference nor this "
              "SMB attribution is tested formally."
              if abs(f_pre["t_alpha"]) <= 1.96 and abs(f_post["t_alpha"]) <= 1.96 else
              "The alpha difference is not tested formally.")),
          ]
    if sens is not None and len(sens):
        parts, fragile = [], False
        for s, g in sens.groupby("signal", sort=False):
            parts.append(f"`{s}`: " + ", ".join(f"{int(r.nw_lags)} lags t {_f(r.t)}" for r in g.itertuples()))
            prim = g[g["primary"]]
            if len(prim) and abs(prim["t"].iloc[0]) > hlz and (g["t"].abs() <= hlz).any():
                fragile = True
        L.append("- Sensitivity of the t of the 'after − before' difference to the number of NW lags (the main specification, 6, "
                 "follows the paper and the base `config.yaml`; the others are checks only, `decay_nw_sensitivity.csv`): " + "; ".join(parts) + ". "
                 + ("Clearing the HLZ bar is not robust to the choice of lags: the decay is significant at 1.96 and borderline at HLZ."
                    if fragile else "The conclusion relative to 1.96 and HLZ does not depend on the choice of lags."))
    L += ["", "## Caveats", "",
          "- This is not our own computation on CRSP but ready-made OpenAP portfolios (a cross-check, not a substitute for our own code).",
          "- The OpenAP construction differs from the paper (Lou–Shu: NYSE/AMEX, monthly measure A at t−2, EW quintiles; "
          "the IMLA_C factor uses A_C terciles): OpenAP has its own universe, filters, windows and quantiles (lines below and `signal_meta.csv`); "
          "the volume factor here is the DolVol L-S, not IMLA_C; FF3+UMD are the VW French factors.",
          "- The post-2013 subperiod is short; the 'after − before' difference tests for decay, not for zero.",
          "- The 2013 break is the end of the Lou–Shu sample, not the publication date of the signals themselves"
          + (" (" + ", ".join(f"`{r['Acronym']}`: {r['Authors']} {_na(r['Year'])}" for _, r in meta.iterrows()
                             if r["Acronym"] in MAIN_SIGNALS and "Year" in meta and "Authors" in meta) + ")"
             if meta is not None and "Year" in meta and "Authors" in meta else "")
          + "; this is a test 'outside the Lou–Shu sample', not a test of publication decay in the McLean–Pontiff (2016) sense: "
            "part of 1964–2012 already lies after publication.",
          "- The two L-S series use different universes: OpenAP `Illiquidity` is NYSE only (exchcd==1) with price > 5 (filter from SignalDoc), "
          "while `DolVol` uses all CRSP stocks without a filter, including NASDAQ, which Lou–Shu exclude precisely because of its inflated volume. "
          "The Table A.1 regression here is therefore an NYSE portfolio on a whole-market factor.",
          *[f"- Timing of `{k}` (from the OpenAP code): {v}." for k, v in OPENAP_TIMING.items()],
          *(["- The official Google Drive link returned 'Quota exceeded'; the file was taken from a public GitHub copy "
             "(the same v2.0.0 release, same file id; pinned commit), sha256 in `run_manifest.json`. "
             "On a rerun, download it again from Drive and compare the sha256."] if "drive.usercontent" not in data_note else []),
          ]
    if meta is not None:
        for _, r in meta.iterrows():
            if r["Acronym"] in MAIN_SIGNALS:
                L.append(f"- Construction of `{r['Acronym']}` per SignalDoc: weights {_na(r.get('Stock Weight'))}; "
                         f"rebalanced every {_na(r.get('Portfolio Period'))} months; filter {_na(r.get('Filter'))}; "
                         f"L-S quantile {_na(r.get('LS Quantile effective', r.get('LS Quantile')))}"
                         f"{' (OpenAP default, quintiles)' if pd.isna(r.get('LS Quantile')) else ''}, "
                         f"breakpoints over all stocks after the filter; "
                         f"definition: {_na(r.get('Detailed Definition'))}")
    return "\n".join(L) + "\n"


def git_info() -> dict:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        st = subprocess.run(["git", "status", "--porcelain", "--", "src", "tests", "config.yaml", "Makefile", "mk", "requirements.txt"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
        return {"sha": sha, "dirty": bool(st.strip()), "dirty_scope": "code+config (src, tests, config.yaml, Makefile, mk, requirements.txt); generated outputs excluded"}
    except (OSError, subprocess.CalledProcessError):
        return {"sha": None, "dirty": None}


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    ap.add_argument("--out", default=str(ROOT / "results" / "upgrades" / "openap_us"))
    ap.add_argument("--raw", default=str(ROOT / "data" / "raw"))
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    timings: dict[str, float] = {}
    t0 = time.perf_counter()

    cfg_path = Path(a.config)
    cfg = yaml.safe_load(cfg_path.read_text())
    lags = int(cfg["tests"]["nw_lags"])
    hlz = float(cfg["tests"].get("hlz_threshold", 3.0))
    seed = int(cfg["seed"])
    latest = Path(a.results_dir) / "LATEST"
    base_run = latest.read_text().strip() if latest.exists() else None
    paper_path = ROOT / "src" / "paper_numbers.yaml"
    paper = yaml.safe_load(paper_path.read_text())["lou_shu_2017"]

    t = time.perf_counter()
    recs = download_all(Path(a.raw))
    timings["download"] = round(time.perf_counter() - t, 3)

    t = time.perf_counter()
    od = Path(a.raw) / "openap" / f"release_{OPENAP_RELEASE}"
    ls_all = load_ls_wide(od / "PredictorLSretWide.csv")
    doc = load_signal_doc(od / "SignalDoc.csv")
    ff = load_french(Path(a.raw) / "french" / FRENCH_FILES[0], Path(a.raw) / "french" / FRENCH_FILES[1])
    missing = [s for s in MAIN_SIGNALS if s not in ls_all.columns]
    if missing:
        raise RuntimeError(f"OpenAP file lacks {missing}; columns like: {[c for c in ls_all.columns if 'ol' in c][:20]}")
    context = [s for s in CONTEXT_SIGNALS if s in ls_all.columns]
    signals = MAIN_SIGNALS + context
    ls = ls_all[signals]
    timings["load"] = round(time.perf_counter() - t, 3)

    t = time.perf_counter()
    periods = {k: v for k, v in PERIODS.items()}
    tables = {
        "means": mean_tests(ls, periods, lags),
        "spanning": spanning_tests(ls, ff, "Illiquidity", "DolVol", periods, lags),
        "correlation": correlations(ls, "Illiquidity", "DolVol", periods),
        "decay_nw_sensitivity": decay_lag_sensitivity(ls[MAIN_SIGNALS], periods, primary=lags),
        "sign_checks": sign_checks(ls, ff, doc, signals, lags),
        "signal_meta": signal_meta(doc, signals),
        "ls_series": ls.join(ff, how="left").reset_index().assign(month=lambda d: d["month"].astype(str)),
    }
    tables["paper_comparison"] = pd.DataFrame([
        {"item": "TableA1_one_factor_alpha", "paper": paper["tableA1_panelA"]["imla_one_factor"]["alpha"],
         "paper_t": paper["tableA1_panelA"]["imla_one_factor"]["t"],
         **_pick(tables["spanning"], "LS_1964_2012", "one_factor")},
        {"item": "TableA1_five_factor_alpha", "paper": paper["tableA1_panelA"]["imla_five_factor"]["alpha"],
         "paper_t": paper["tableA1_panelA"]["imla_five_factor"]["t"],
         **_pick(tables["spanning"], "LS_1964_2012", "five_factor")},
        {"item": "Table3_A_raw_HL", "paper": paper["table3_panelA"]["A"]["raw"]["hl"],
         "paper_t": paper["table3_panelA"]["A"]["raw"]["t"],
         **_pick(tables["spanning"], "LS_1964_2012", "raw")},
        {"item": "Table3_A_4factor_alpha_HL", "paper": paper["table3_panelA"]["A"]["alpha"]["hl"],
         "paper_t": paper["table3_panelA"]["A"]["alpha"]["t"],
         **_pick(tables["spanning"], "LS_1964_2012", "ff4_only")},
    ] + [
        {"item": f"TableA1_panelB_{k}_alpha", "paper": v["alpha"], "paper_t": v["t"],
         **_pick(tables["spanning"], "LS_1964_2012", k.replace("imla_", ""))}
        for k, v in (paper.get("tableA1_panelB") or {}).items() if k.startswith("imla_")
    ])
    timings["tests"] = round(time.perf_counter() - t, 3)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False)
    src_ls = recs["openap/PredictorLSretWide.csv"].get("url") or "cache"
    data_note = (f"OpenAP release {OPENAP_RELEASE} (`PredictorLSretWide.csv`, source: {src_ls}); "
                 "Kenneth French FF3 + Momentum (monthly).")
    (out / "SUMMARY.md").write_text(build_summary(tables["means"], tables["spanning"], tables["correlation"],
                                                  tables["sign_checks"], paper, str(base_run), hlz, data_note,
                                                  tables["signal_meta"], tables["decay_nw_sensitivity"]))
    timings["total"] = round(time.perf_counter() - t0, 3)
    code_files = [Path(__file__).resolve(), paper_path]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = {
        "upgrade": "openap_us",
        "run_id": stamp + "_openap_us",
        "created_utc": stamp,
        "base_run_id": base_run,
        "base_run_used_for": "provenance only (US data do not depend on the crypto panel)",
        "git": git_info(),
        "config_path": rel_path(cfg_path), "config_sha256": sha256_file(cfg_path),
        "code_sha256": {p.name: sha256_file(p) for p in code_files},
        "input_data": recs,
        "openap_release": OPENAP_RELEASE,
        "signals": signals, "periods": PERIODS, "nw_lags": lags, "hlz_threshold": hlz,
        "seed": seed, "seed_used": False,
        "timings_sec": timings,
        "tables": sorted(f"{k}.csv" for k in tables) + ["SUMMARY.md"],
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    log.info("openap_us written to %s", out)
    return out


def _pick(span: pd.DataFrame, period: str, spec: str) -> dict:
    r = span[(span["period"] == period) & (span["spec"] == spec)].iloc[0]
    return {"ours": r["alpha"], "ours_t": r["t_alpha"], "ours_n": int(r["n"]), "ours_period": f"{r['start']}..{r['end']}"}


if __name__ == "__main__":
    main()
