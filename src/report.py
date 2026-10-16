""""Paper vs us" table (CSV + Markdown) and LaTeX number macros for the slides.

Usage:  python -I src/report.py [--results-dir results] [--run-id RUN_ID (default: results/LATEST)]
Writes: results/<run_id>/paper_vs_us.csv, paper_vs_us.md, tex/numbers.tex
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json
import math

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
PAPER = Path(__file__).resolve().parent / "paper_numbers.yaml"
T_CRIT = 1.96

# Why our number may differ; factual, pointing to DEVIATIONS.md. Result-dependent parts stay TBD.
WHY = {
    "corrAAC": "Same construction; level depends on the cross-sectional dispersion of |r| vs. DVol (D1). TBD after real data.",
    "corrAabs": "Level and dispersion of crypto |r| differ from NYSE/AMEX stocks (D1). TBD after real data.",
    "corrACabs": "As above (D1). TBD after real data.",
    "QratioAC": "Dispersion of dollar volume across coins vs. stocks; the $10k volume filter cuts the illiquid tail (D8). TBD after real data.",
    "QratioAbs": "Crypto volatility level and dispersion differ from stocks (D1). TBD after real data.",
    "HLArawCoef": "Far fewer months than 588 and fewer assets -> lower power (D1); crypto returns are much more volatile than stock returns. TBD after real data.",
    "HLAalphaCoef": "Alpha on CMKT+CMOM instead of FF3+UMD (D6); short sample (D1). TBD after real data.",
    "HLACrawCoef": "Short sample, fewer assets (D1). TBD after real data.",
    "HLACalphaCoef": "CMKT+CMOM alpha (D6); short sample (D1). TBD after real data.",
    "HLResArawCoef": "Paper's spread is insignificant; ours tests the same null on a short sample (D1). TBD after real data.",
    "HLResAalphaCoef": "CMKT+CMOM alpha (D6). TBD after real data.",
    "FMlnA": "CMKT-adjusted returns, 36m betas (D4); no ln(ME), B/M controls (D2, D3); r_f = 0 (D5, no effect on slopes). TBD after real data.",
    "FMlnAC": "As for ln A (D2-D4). TBD after real data.",
    "FMResLnA": "As for ln A (D2-D4). TBD after real data.",
    "FMResLnAivol": "IVOL vs daily EW CMKT, window ends in t-2 (D7); no size control (D2). TBD after real data.",
    "ANResLnA": "Same regression as the row above; Amihud & Noh use 12-month measures and get the opposite sign to Lou & Shu. TBD after real data.",
    "ANDIF": "Our measures are monthly (A&N: 12-month windows), no size/B/M controls (D2, D3). A&N col. 2 (with ln LSIlliq) gives 1.219 (t 4.42). TBD after real data.",
    "IMLAalpha": "IMLA_C built from our tercile sorts on A_C; one-factor alpha. TBD after real data.",
}

# id, label, kind ('t' = rule on sign and t-stat, 'level' = rule on sign and order of magnitude)
METRICS = [
    ("corrAAC", "corr(A, A_C)", "level"),
    ("corrAabs", "corr(A, |Ret|)", "level"),
    ("corrACabs", "corr(A_C, |Ret|)", "level"),
    ("QratioAC", "Q75/Q25 of A_C", "level"),
    ("QratioAbs", "Q75/Q25 of |Ret|", "level"),
    ("HLArawCoef", "H-L on A, raw (%/mo)", "t"),
    ("HLAalphaCoef", "H-L on A, alpha (%/mo)", "t"),
    ("HLACrawCoef", "H-L on A_C, raw (%/mo)", "t"),
    ("HLACalphaCoef", "H-L on A_C, alpha (%/mo)", "t"),
    ("HLResArawCoef", "H-L on Res.A, raw (%/mo)", "t"),
    ("HLResAalphaCoef", "H-L on Res.A, alpha (%/mo)", "t"),
    ("FMlnA", "FM ln A, model (1)", "t"),
    ("FMlnAC", "FM ln A_C, model (2)", "t"),
    ("FMResLnA", "FM Res.lnA, model (3)", "t"),
    ("FMResLnAivol", "FM Res.lnA, model (5) with ln A_C + IVOL", "t"),
    ("ANResLnA", "A&N (2021) col. 4: RlnILLIQ with ln A_C + IVOL", "t"),
    ("ANDIF", "A&N (2021) col. 3: DIF with ln A_C + ln|Ret|", "t"),
    ("IMLAalpha", "alpha of H-L(A) on IMLA_C (%/mo)", "t"),
]


def paper_values(p: dict) -> dict[str, tuple[float, float | None]]:
    ls, an = p["lou_shu_2017"], p["amihud_noh_2021"]
    t1a, t1b, t3, t4 = ls["table1_panelA"], ls["table1_panelB"], ls["table3_panelA"], ls["table4_panelA"]
    v = {"corrAAC": (t1b["corr_A_AC"], None), "corrAabs": (t1b["corr_A_absret"], None),
         "corrACabs": (t1b["corr_AC_absret"], None),
         "QratioAC": (t1a["A_C"]["q75"] / t1a["A_C"]["q25"], None),
         "QratioAbs": (t1a["absret"]["q75"] / t1a["absret"]["q25"], None)}
    for sig, key in [("A", "A"), ("A_C", "AC"), ("ResA", "ResA")]:
        for row in ["raw", "alpha"]:
            v[f"HL{key}{row}Coef"] = (t3[sig][row]["hl"], t3[sig][row]["t"])
    v["FMlnA"] = tuple(t4["(1)"]["lnA"].values())
    v["FMlnAC"] = tuple(t4["(2)"]["lnA_C"].values())
    v["FMResLnA"] = tuple(t4["(3)"]["ResLnA"].values())
    v["FMResLnAivol"] = tuple(t4["(5)"]["ResLnA"].values())
    v["ANResLnA"] = tuple(an["col4"]["RlnILLIQ"].values())
    v["ANDIF"] = tuple(an["col3"]["DIF"].values())
    v["IMLAalpha"] = tuple(ls["tableA1_panelA"]["imla_one_factor"].values())
    return {k: (float(a), None if b is None else float(b)) for k, (a, b) in v.items()}


def our_values(rd: Path) -> dict[str, tuple[float, float | None]]:
    """Read our numbers from the run's CSVs; a missing table gives NaN for its rows."""
    nan = (math.nan, None)
    v: dict[str, tuple[float, float | None]] = {}

    def read(name):
        f = rd / f"{name}.csv"
        return pd.read_csv(f) if f.exists() else None

    corr = read("T1_corr")
    if corr is not None:
        c = corr.set_index("measure")
        v["corrAAC"], v["corrAabs"], v["corrACabs"] = ((float(c.loc[a, b]), None) for a, b in
                                                       [("A", "A_C"), ("A", "absret"), ("A_C", "absret")])
    desc = read("T1_describe")
    if desc is not None:
        d = desc.set_index("measure")
        v["QratioAC"] = (float(d.loc["A_C", "75%"] / d.loc["A_C", "25%"]), None)
        v["QratioAbs"] = (float(d.loc["absret", "75%"] / d.loc["absret", "25%"]), None)
    t3 = read("T3_sorts")
    if t3 is not None:
        t3 = t3.set_index(["signal", "row"])
        for sig, key in [("A", "A"), ("A_C", "AC"), ("ResA", "ResA")]:
            for row in ["raw", "alpha"]:
                r = t3.loc[(sig, row)]
                v[f"HL{key}{row}Coef"] = (float(r["H-L"]), float(r["t"]))

    def fm(df, model, var):
        r = df[(df["model"] == model) & (df["var"] == var)]
        return (float(r["coef"].iloc[0]), float(r["t"].iloc[0])) if len(r) else nan

    t4 = read("T4_fm")
    if t4 is not None:
        v["FMlnA"], v["FMlnAC"] = fm(t4, "(1)", "lnA"), fm(t4, "(2)", "lnA_C")
        v["FMResLnA"], v["FMResLnAivol"] = fm(t4, "(3)", "ResLnA"), fm(t4, "(5)", "ResLnA")
    u1 = read("U1_an_fm")
    if u1 is not None:
        v["ANResLnA"] = fm(u1, "AN: ResLnA+lnA_C+ivol", "ResLnA")
        v["ANDIF"] = fm(u1, "AN: lnA_C+ln|Ret|+DIF", "DIF")
    t5 = read("T5_imla")
    if t5 is not None:
        r = t5[t5["model"] == "IMLA_C"]
        if len(r):
            v["IMLAalpha"] = (float(r["alpha_HL_A"].iloc[0]), float(r["t"].iloc[0]))
    return {k: v.get(k, nan) for k, _, _ in METRICS}


def match(kind: str, paper: tuple, us: tuple) -> str:
    """Rule-based verdict (see the legend in paper_vs_us.md)."""
    pv, pt = paper
    uv, ut = us
    if uv is None or not math.isfinite(uv):
        return "n/a"
    same = math.copysign(1, pv) == math.copysign(1, uv)
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


def build(rd: Path) -> pd.DataFrame:
    paper = paper_values(yaml.safe_load(PAPER.read_text()))
    us = our_values(rd)
    rows = []
    for mid, label, kind in METRICS:
        (pv, pt), (uv, ut) = paper[mid], us[mid]
        rows.append({"id": mid, "metric": label, "kind": kind, "paper": pv, "paper_t": pt,
                     "us": uv, "us_t": ut, "match": match(kind, paper[mid], us[mid]), "why": WHY[mid]})
    return pd.DataFrame(rows)


def sample_label(rd: Path) -> str:
    s = rd / "sample.csv"
    if not s.exists():
        return "Binance"
    r = pd.read_csv(s).iloc[0]
    return f"Binance {r['first']}–{r['last']}, {int(r['n_months'])} mo."


def write_md(df: pd.DataFrame, rd: Path, man: dict) -> None:
    us_col = f"Us ({sample_label(rd)})"
    lines = [f"# Lou & Shu (2017) vs. us — run `{man.get('run_id', rd.name)}`", "",
             f"Data: `{man.get('data_source', '?')}` (sha256 `{str(man.get('data_sha256', ''))[:12]}`), "
             f"git `{str((man.get('git') or {}).get('sha'))[:10]}`"
             f"{' (dirty)' if (man.get('git') or {}).get('dirty') else ''}. "
             "Paper: NYSE/AMEX 1964–2012, 588 months; A&N (2021): CRSP 1955–2016.", "",
             f"| Metric | Paper | {us_col} | Match? | Why it differs |", "|---|---|---|---|---|"]
    for r in df.itertuples():
        cells = [r.metric, fmt(r.paper, r.paper_t), fmt(r.us, r.us_t), r.match, r.why]
        lines.append("| " + " | ".join(md_cell(c) for c in cells) + " |")
    lines += ["", "**Match rule.** Coefficients / spreads: if the paper is significant (|t| ≥ 1.96), "
              "*yes* = same sign and |t| ≥ 1.96, *sign only* = same sign but insignificant, *no* = opposite sign; "
              "if the paper is insignificant, *yes* = we are insignificant too, *sign only* / *no* = we are "
              "significant with the same / opposite sign. Levels (correlations, Q75/Q25): *yes* = same sign and "
              "within half an order of magnitude (|log10(us/paper)| ≤ 0.5), *sign only* = same sign otherwise. "
              "Deviations D1–D11: see `DEVIATIONS.md`.", ""]
    (rd / "paper_vs_us.md").write_text("\n".join(lines))


def write_tex(df: pd.DataFrame, rd: Path) -> None:
    """\\newcommand macros: \\us<Id>, \\us<Id>T, \\paper<Id>, \\paper<Id>T, \\match<Id>."""
    out = ["% generated by src/report.py — do not edit by hand"]
    for r in df.itertuples():
        mid = r.id
        d = 2 if r.kind == "t" or abs(r.paper) >= 10 else 3
        out.append(f"\\newcommand{{\\us{mid}}}{{{tex_num(r.us, d)}}}")
        out.append(f"\\newcommand{{\\paper{mid}}}{{{tex_num(r.paper, d)}}}")
        if r.kind == "t":
            out.append(f"\\newcommand{{\\us{mid}T}}{{{tex_num(r.us_t, 2)}}}")
            out.append(f"\\newcommand{{\\paper{mid}T}}{{{tex_num(r.paper_t, 2)}}}")
        out.append(f"\\newcommand{{\\match{mid}}}{{{r.match}}}")
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


def resolve_run(results_dir: Path, run_id: str | None) -> Path:
    run_id = run_id or (results_dir / "LATEST").read_text().strip()
    return results_dir / run_id


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    ap.add_argument("--run-id", default=None)
    a = ap.parse_args(argv)
    rd = resolve_run(Path(a.results_dir), a.run_id)
    mf = rd / "run_manifest.json"
    man = json.loads(mf.read_text()) if mf.exists() else {}
    df = build(rd)
    df.to_csv(rd / "paper_vs_us.csv", index=False)
    write_md(df, rd, man)
    write_tex(df, rd)
    print(df[["metric", "paper", "us", "us_t", "match"]].to_string(index=False))
    return rd


if __name__ == "__main__":
    main()
